#include "dns/gpu/runner_comp.hpp"

#include "dns/interrupt.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/derived.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/derived_remap.hpp"
#include "dns/gpu/internal/frame_scratch.hpp"
#include "dns/gpu/internal/rans_stats.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/internal/steady_march.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/rotortable.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace dns::gpu {

using internal::fail;
using internal::finishTiming;
using internal::rampFrac;

std::expected<RunStats, RunError>
runCompressible(const Grid& g, const CompState& ic, const Sdf& sdf,
                const std::array<double, 6>* bbox, const CompRun& r,
                RunOutput& out)
{
    const auto gg = GpuGrid::from(g);
    const bool nscbc = r.far == FarField::Nscbc;
    const bool stretched = r.stretchBeta > 0;
    const bool focal = r.focalActive;
    // RANS on the immersed wall (masterplan_flowrans P5): off is every path
    // below bit for bit.
    const bool rans = r.rans != RansModel::None;
    const bool steady = r.steadyTol > 0;
    assert(!steady || rans); // the validator: immersed --steady needs --rans

    // The PointRemaps of the two curvilinear meshes; both are the exact
    // identity when their parameters are off, so building them unconditionally
    // costs nothing. NSCBC and the derived frames evaluate the active one on
    // device (kernel template argument) so differences use physical, not
    // reference, distances.
    const TanhStretchY remap{double(g.ny) * g.hy, r.stretchBeta};
    const AxisRemap3 fremap = focalRemap(g, r.focalActive, r.focalLo,
                                         r.focalHi, r.focalW, r.focalRatio);

    // Curvilinear mesh: nodes from the PointRemap, finite-volume tables
    // derived from the nodes (discrete face closure), solver on the
    // curvilinear path. LES rides along on both paths (the SGS operators read
    // the mesh metrics); the validator guarantees at most one of the two
    // meshes is active.
    std::optional<CurvMetrics> mtr;
    if (stretched)
    {
        auto m = CurvMetrics::build(CurvMesh::stretchedY(g, r.stretchBeta));
        if (!m) return fail(2, "stretched mesh: " + m.error());
        mtr = std::move(*m);
    }
    else if (focal)
    {
        auto m = CurvMetrics::build(CurvMesh::fromRemap(g, fremap));
        if (!m) return fail(2, "focal mesh: " + m.error());
        mtr = std::move(*m);
    }
    else if (rans)
    {
        // The closure lives on the curvilinear path only: a uniform mesh is
        // its identity-map instance (the cost ratio is recorded in
        // docs/rans.md).
        auto m = CurvMetrics::build(CurvMesh::uniform(g));
        if (!m) return fail(2, "uniform mesh: " + m.error());
        mtr = std::move(*m);
    }
    auto dev = mtr ? DeviceComp::make(*mtr, r.tr, r.les, r.sgs)
                   : DeviceComp::make(g, r.tr, r.les, r.sgs);
    // The compensated state: always on the steady march, opt-in on a time
    // march (--compensated). --penalize, a plain-state edit, is rejected
    // with it by the validator.
    const bool compensated = steady || r.compensated;
    auto ds = DeviceCompState::make(g.size(), ransScalars(r.rans), compensated);
    auto red = ReduceScratch::make();
    if (!dev || !ds || !red) return fail(6, "GPU allocation failed");

    // --penalize: device lambda for the band decay (DeviceCutGeometry::make
    // reuses the standard upload; the aperture buffers ride along unused).
    std::optional<DeviceCutGeometry> dPen;
    if (r.penalize > 0)
    {
        auto p = DeviceCutGeometry::make(*r.cutGeom);
        if (!p) return fail(6, "GPU allocation failed");
        dPen = std::move(*p);
    }

    DeviceCompFar::Config fc;
    fc.rhoInf = r.rhoInf;
    fc.uInf = r.uInf;
    fc.pInf = r.pInf;
    fc.gamma = r.tr.gamma;
    // Resolved by validate(); not re-derived here (see runner_incomp).
    if (r.spongeCells <= 0)
        return fail(6, "far-field band width was not resolved");
    fc.ns = r.spongeCells;
    fc.openY = r.open;
    fc.openZ = r.open && g.nz > 1;
    fc.noslipY = r.noslipY;
    fc.phiInf[0] = r.phiInf[0];
    fc.phiInf[1] = r.phiInf[1];
    if (nscbc)
    {   // x is handled characteristically; an optional thin strip inside the
        // NSCBC rows absorbs the residual reflection that pumps box modes above
        // Re 800 (pure NSCBC diverges at Re2000 / 2048^2).
        fc.sigMax = r.nscbcStrip > 0 ? 0.5 : 0.0;
        if (r.nscbcStrip > 0) fc.ns = r.nscbcStrip;
    }
    auto far = DeviceCompFar::make(g, fc);
    if (!far) return fail(6, "far-field init failed");

    std::optional<DeviceCompNscbc> nsc;
    if (nscbc)
    {
        DeviceCompNscbc::Config nc;
        nc.rhoInf = r.rhoInf;
        nc.uInf = r.uInf;
        nc.pInf = r.pInf;
        nc.gamma = r.tr.gamma;
        nc.betaTrans = r.nscbcBeta;
        nc.betaAuto = r.nscbcAbc == NscbcAbc::Adaptive;
        nc.betaModal = r.nscbcAbc == NscbcAbc::Modal;
        nc.etaIn = r.nscbcEtaIn;
        nc.etaInShear = r.nscbcEtaInShear;
        nc.etaLat = r.nscbcEtaLat;
        nc.openY = r.open;
        nc.openZ = r.open && g.nz > 1;
        nc.potCorrect = r.nscbcPot;
        nc.xc = r.cx;
        nc.yc = r.cy;
        nc.zc = r.cz;
        nc.aBody = 0.5 * r.D;
        nc.floorRows = r.floorRows;
        nc.thermalInflow = r.nscbcThermalInflow;
        nc.Rgas = r.tr.Rgas;
        nc.TInf = r.pInf / (r.rhoInf * r.tr.Rgas);
        auto m = DeviceCompNscbc::make(g, nc);
        if (!m) return fail(6, "NSCBC init failed");
        nsc = std::move(*m);
    }

    // --nscbc-sink: the scale-selective band under the characteristic rows
    // (gpu/comp_sink.hpp). Validator guarantees the uniform mesh; 0 = off,
    // no object, no kernel, bit-for-bit the bare boundary.
    std::optional<DeviceCompSink> snk;
    if (nscbc && r.nscbcSink > 0)
    {
        auto m = DeviceCompSink::make(
            g, {.nb = r.nscbcSink,
                .alpha = r.nscbcSinkAmp,
                .cInf = std::sqrt(r.tr.gamma * r.pInf / r.rhoInf),
                .openY = r.open,
                .openZ = r.open && g.nz > 1});
        if (!m) return fail(6, "sink init failed");
        snk = std::move(*m);
    }

    // The ghost wall classifies against the physical cell centres of whichever
    // mesh is active; the focal remap is the exact identity when off, so the
    // Cartesian tables are unchanged.
    const GhostWallThermal wallBc{
        .mode = r.wallIsothermal ? GhostWallThermal::Mode::Isothermal
                                 : GhostWallThermal::Mode::Adiabatic,
        .Twall = r.Twall,
        .Rgas = r.tr.Rgas};
    // A spinning body needs two things a static one does not: its surface
    // MOVES (the mirror images about omega x r, not about rest, or the blade is
    // a rotating hole that drives nothing), and its geometry CHANGES with time.
    // The second is the azimuth table: every bin classified up front, replayed
    // by the loop, because reclassifying costs a signed-distance sweep and the
    // run takes thousands of steps per revolution.
    std::optional<RotorTable> rotorTable;
    std::optional<DeviceCompGhost> ghostOwned;
    DeviceCompGhost* ghost = nullptr;
    if (r.rotor.active() && r.rotorMrf)
    {
        // --rotor-frame mrf: the frozen-rotor approximation. ONE classification
        // (azimuth 0, exactly the static path plus the wall velocity), no
        // azimuth table, and the rotation reaches the fluid as the rotating-
        // frame equations inside a blended cylinder around the swept volume
        // (gpu/comp.hpp MrfZone; formulation in internal/comp_flux.hpp).
        const int a = rotorAxisIndex(r.rotor.axis);
        const int b = (a + 1) % 3, c2 = (a + 2) % 3;
        const double* cs = r.rotor.shaft.data();
        // ONE host classification up front, and the zone is sized from its
        // interface faces -- the SAME measurement the table path makes
        // (rotortable.hip). The body AABB's in-plane corner (rotorSweepBounds)
        // is exact for the swept BOX but overestimates a tip radius by up to
        // sqrt(2): a 4-blade cross has a square AABB whose corner is empty
        // space, and the zone sized from it reached past the focal plateau
        // into the mesh blend (measured 2026-07-31: 4-blade prop, reported
        // tip 0.5608 vs real 0.3927, r2 0.718 against a plateau half-width
        // 0.589, diverged at t = 3.07).
        const GhostWallMotion motion = rotorWallMotion(r.rotor);
        const CompGhostTables tbls = classifyCompGhost(
            g, sdf, bbox ? bbox->data() : nullptr, fremap, &motion);
        if (tbls.nGhost() == 0 || tbls.nFace() == 0)
            return fail(3, "rotor: the body has no mirrored ghost cells -- it "
                           "is not resolved by this grid at all");
        double tipR = 0.0, aHalfMeas = 0.0;
        for (std::size_t f = 0; f + 2 < tbls.fpos.size(); f += 3)
        {
            const double db = double(tbls.fpos[f + b]) - cs[b];
            const double dc = double(tbls.fpos[f + c2]) - cs[c2];
            tipR = std::max(tipR, std::hypot(db, dc));
            aHalfMeas = std::max(
                aHalfMeas, std::abs(double(tbls.fpos[f + a]) - cs[a]));
        }
        // The tip-Mach guard is the same hard stop the table path applies --
        // the KG flux has no shock capturing either way.
        const double cInf = std::sqrt(r.tr.gamma * r.pInf / r.rhoInf);
        if (auto e = internal::checkTipMach(r.rotor.omega, tipR, r.rotor.radius,
                                            cInf, r.uInf / cInf,
                                            "measured tip radius", "body", "", "");
            !e)
            return std::unexpected(e.error());

        // Zone: f = 1 must cover the blade plus the ghost wall's mirror band,
        // f = 0 must be reached well before any domain face so the NSCBC rows
        // and the sink never see the frame. Margins in fine cells and in D.
        assert(r.hFine > 0);   // dns_main fills it from hFine(Case)
        const double hF = r.hFine;
        const double aHalf = aHalfMeas;
        MrfZone z;
        z.axis = a;
        z.omega = r.rotor.omega;
        for (int d = 0; d < 3; ++d) z.center[d] = cs[d];
        z.r1 = tipR + std::max(0.05 * r.D, 4.0 * hF);
        z.r2 = z.r1 + std::max(0.15 * r.D, 8.0 * hF);
        z.a1 = aHalf + std::max(0.05 * r.D, 4.0 * hF);
        z.a2 = z.a1 + std::max(0.15 * r.D, 8.0 * hF);
        // A z shaft at nz == 1 is the exact 2D restriction: the zone is
        // z-invariant, so the axial blend never engages.
        const bool axialDegenerate = a == 2 && g.nz == 1;
        if (axialDegenerate)
        {
            z.a1 = 1e30;
            z.a2 = 2e30;
        }
        // Clearance: the support must end at least (boundary bands + 4) cells
        // inside every face. Reference axis lengths equal the physical ones
        // (the focal map is endpoint-preserving); the margin uses the COARSE
        // spacing, the largest a cell near a face can be.
        const double L[3] = {g.nx * g.hx, g.ny * g.hy, g.nz * g.hz};
        const double hC[3] = {g.hx * (focal ? r.focalRatio : 1.0),
                              g.hy * (focal ? r.focalRatio : 1.0),
                              g.hz * (focal ? r.focalRatio : 1.0)};
        const int bandCells = std::max(r.nscbcStrip, r.nscbcSink) + 4;
        auto clear = [&](int d) { return std::min(cs[d], L[d] - cs[d]) -
                                         bandCells * hC[d]; };
        double lat = clear(b);
        if (g.nz > 1 || c2 != 2) lat = std::min(lat, clear(c2));
        if (z.r2 >= lat || (!axialDegenerate && z.a2 >= clear(a)))
            return fail(2, std::format(
                "MRF zone (r2 {:.3g}, a2 {:.3g}) reaches within {} cells of a "
                "domain face (lateral clearance {:.3g}, axial {:.3g}): the "
                "rotating frame must vanish before the boundary rows. Enlarge "
                "--dr or the box",
                z.r2, z.a2, bandCells, lat, clear(a)));
        std::printf("rotor: MRF zone about the shaft: f=1 to r %.4g, 0 past "
                    "%.4g; axial %.4g/%.4g (frozen rotor -- an "
                    "APPROXIMATION; blade fixed at azimuth 0)\n",
                    z.r1, z.r2, z.a1, z.a2);
        std::fflush(stdout);
        if (dev->setMrf(z, fremap) != hipSuccess)
            return fail(6, "MRF zone init failed");

        CompGhostCapacity cap;
        cap.cover(tbls);
        auto gh = DeviceCompGhost::fromTables(g, tbls, cap, r.tr.gamma, r.rhoInf,
                                              r.pInf, focal, wallBc);
        if (!gh) return fail(6, "ghost wall init failed");
        // The deep-solid verdict the table path reports per bin, measured on
        // the one frozen azimuth: the count the dry-run's thin-blade warning
        // promises, and what gpu_mrf_rotor2 asserts is nonzero.
        std::printf("rotor: frozen azimuth: %d ghosts, %d deep, %d faces\n",
                    tbls.nGhost(), tbls.nDeep(), tbls.nFace());
        if (gh->deepCells().empty())
            std::printf("rotor: WARNING no deep-solid cells -- the body is "
                        "nowhere thicker than the mirror band. Thicken it "
                        "(--blade-thicken) or refine.\n");
        ghostOwned = std::move(*gh);
        ghost = &*ghostOwned;
    }
    else if (r.rotor.active())
    {
        // The turn alone: `sdf` is the body AT AZIMUTH 0 and already carries
        // --blade-thicken (casegeom applied it), so re-applying the dilation
        // here would double it. Dropping it is exact rather than approximate --
        // a rotation is an isometry, so dilating then turning and turning then
        // dilating are the same body.
        Rotor turn = r.rotor;
        turn.thicken = 0.0;
        // The short-circuit box must cover the SWEPT volume, not the body's
        // pose at one angle: a bin whose blade reached outside it would be
        // classified as plain fluid and vanish.
        std::array<double, 6> swept{};
        if (bbox) swept = rotorSweepBounds(turn, *bbox);
        const auto t0 = std::chrono::steady_clock::now();
        auto tbl = buildRotorTable(g, turn, [&sdf] { return sdf; },
                                   bbox ? swept.data() : nullptr, fremap);
        if (!tbl) return fail(3, tbl.error());
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                .count();
        std::printf("rotor: %d azimuth bins built in %.2f s, %.1f MB host "
                    "(max %d ghosts, %d deep, %d faces)\n",
                    r.rotor.bins, secs, double(tbl->bytes()) / (1 << 20),
                    tbl->capacity.nGhost, tbl->capacity.nDeep,
                    tbl->capacity.nFace);
        // --rotor-mach is quoted on the NOMINAL tip radius D/2, because that is
        // all the validator can know without opening the body. The real tip may
        // be further out -- a lens, a scene, a mesh whose blades overhang -- and
        // then the fastest material point in the flow is faster than the flag
        // says. The KG central flux has no shock capturing, so this is a hard
        // stop rather than a warning; it is checked HERE because the extent is
        // only measurable once the body has been voxelized.
        const double cInf = std::sqrt(r.tr.gamma * r.pInf / r.rhoInf);
        if (auto e = internal::checkTipMach(
                r.rotor.omega, tbl->tipRadius, r.rotor.radius, cInf,
                r.uInf / cInf, "measured tip radius", "body",
                "The flux is Kennedy-Gruber central with no shock capturing. ",
                ", or use a body that fits inside its own diameter");
            !e)
            return std::unexpected(e.error());
        // The mirror band must be at least two cells thick per side, so a body
        // with NO deep-solid core anywhere is one whose two surfaces read
        // through each other. Reported, not rejected: it is a resolution
        // verdict on the user's own mesh, and only measurable once that mesh
        // has been voxelized.
        if (tbl->capacity.nDeep == 0)
            std::printf("rotor: WARNING no deep-solid cells at any azimuth -- "
                        "the body is nowhere thicker than the mirror band, so "
                        "its two surfaces read through each other. Thicken it "
                        "(--blade-thicken) or refine.\n");
        std::fflush(stdout);
        auto gh = DeviceCompGhost::fromTables(g, tbl->bins[0], tbl->capacity,
                                              r.tr.gamma, r.rhoInf, r.pInf, focal,
                                              wallBc);
        if (!gh) return fail(6, "ghost wall init failed");
        rotorTable = std::move(*tbl);
        ghostOwned = std::move(*gh);
        ghost = &*ghostOwned;
    }
    else
    {
        auto gh = DeviceCompGhost::make(g, sdf, r.tr.gamma, r.rhoInf, r.pInf,
                                        bbox ? bbox->data() : nullptr, fremap,
                                        wallBc);
        if (!gh) return fail(6, "ghost wall init failed");
        ghostOwned = std::move(*gh);
        ghost = &*ghostOwned;
    }
    // A resolution verdict like the deep-solid count: ghosts whose normal
    // sample found no fluid are imaged from their fluid neighbours instead
    // (classifyCompGhost) -- the wall stays closed, but is locally a staircase.
    {
        int fb = ghost->nFallback();
        if (rotorTable)
            for (const auto& t : rotorTable->bins) fb = std::max(fb, t.fallback);
        if (fb > 0)
            std::printf("ghost wall: %d ghost cells imaged from their fluid "
                        "neighbours (no usable wall-normal sample: overlapping "
                        "mesh shells, or a body thin or concave at the grid "
                        "scale)\n", fb);
    }
    // Bin currently resident on the device. Bin 0 is loaded above, and the
    // index is recomputed from the step's own time (never accumulated), so a
    // long run cannot drift off the table.
    int rotorBinLoaded = 0;
    long rotorSwaps = 0;
    // The body mask the artefacts show. A rotor's blade moves, so it is
    // refreshed at FRAME time (rare) rather than at bin time (often), and by
    // toggling the two bins' solid lists rather than rewriting the grid -- the
    // cost is the surface, not the volume.
    std::vector<float> rotorMask;
    int rotorMaskBin = -1;
    if (r.rotor.active()) rotorMask.assign(g.size(), 1.0f);

    // RANS: the closure on the curvilinear solver. The fluid mask (cells
    // that are neither solid nor a far mirror row) is where the SST guard
    // counts, the residual norms and the stop integrals sum; mu_t gets the
    // ghost wall's odd image and the mirror rows' even copy.
    std::vector<std::uint8_t> fluid;
    if (rans)
    {
        fluid.assign(g.size(), 1);
        std::vector<std::uint8_t> solid(g.size(), 0);
        for (int c : ghost->solidCells()) {
            fluid[std::size_t(c)] = 0;
            solid[std::size_t(c)] = 1;
        }
        for (int k = 0; k < g.nz; ++k)
            for (int i = 0; i < g.nx; ++i)
                for (int j : {0, g.ny - 1})
                    if (!r.open) fluid[g.idx(i, j, k)] = 0;
        if (!r.open && g.nz > 1)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    for (int k : {0, g.nz - 1}) fluid[g.idx(i, j, k)] = 0;
        ScalarBoundary bnd;
        bnd.open[0] = 1;   // x: the NSCBC / sponge faces
        bnd.open[1] = r.open ? 1 : 0;
        bnd.open[2] = r.open && g.nz > 1 ? 1 : 0;
        bnd.phiInf[0] = r.phiInf[0];
        bnd.phiInf[1] = r.phiInf[1];
        DeviceCompGhost* gp = ghost;
        DeviceCompFar* fp = &*far;
        if (!dev->enableRans({.model = r.rans,
                              .wallDist = r.wallDist,
                              .mutHook = [gp, fp](Real* nut, Real* f1) {
                                  gp->mirrorMut(nut, f1);
                                  fp->applyMut(nut, f1);
                              },
                              .fluidMask = fluid,
                              .bnd = bnd,
                              .solidMask = std::move(solid)}))
            return fail(6, "GPU allocation failed (RANS)");
    }
    // LES: the SGS mu_t takes the SAME images, so the body face carries the
    // molecular stress (up to the staircase) and the y/z mirror rows hold a
    // copy of their neighbour, not a gradient wrapped across the box.
    else if (r.les > 0)
    {
        DeviceCompGhost* gp = ghost;
        DeviceCompFar* fp = &*far;
        dev->setMutHook([gp, fp](Real* nut, Real* f1) {
            gp->mirrorMut(nut, f1);
            fp->applyMut(nut, f1);
        });
    }
    if (compensated && !dev->enableCompensation())
        return fail(6, "GPU allocation failed (compensated state)");

    // --penalize: the band decay rides the stage hook (runs once per RK
    // stage, so each application carries dt/3; penDt is set per step below).
    // It precedes the ghost fill so the mirrors sample the penalized field.
    double penDt = 0;
    auto applyBc = [&](DeviceCompState& s) {
        if (dPen && penDt > 0)
            penalizeMomentum(GpuGrid::from(g), dPen->lambda, Real(r.penalize),
                             Real(penDt), s);
        far->apply(s);
        ghost->apply(s);
        if (rans) ghost->applyTurb(s, r.rans, r.tr, r.phiInf[1]);
    };
    dev->setStageHook(applyBc);
    if (nsc)
        dev->setRhsHook([&](const DeviceCompState& s, DeviceCompState& out,
                            const StageCombine& cmb) {
            if (stretched) nsc->apply(*dev, s, out, cmb, remap);
            else if (focal) nsc->apply(*dev, s, out, cmb, fremap);
            else nsc->apply(*dev, s, out, cmb);
            // The sink ADDS on top of the characteristic rows too, on
            // purpose: their mesh-scale content is exactly what the
            // decomposition mishandles (docs/nscbc.md, the hand-off).
            if (snk) snk->apply(s, out, cmb);
        });

    if (ds->upload(ic) != hipSuccess)
        return fail(6, "initial condition upload failed");
    // RANS: every scalar at its freestream value, rho_ic phi_inf.
    for (int i = 0; i < ds->nT; ++i)
    {
        std::vector<double> t0(g.size());
        for (std::size_t c = 0; c < g.size(); ++c)
            t0[c] = ic.rho[c] * r.phiInf[i];
        if (ds->uploadTurb(i, t0) != hipSuccess)
            return fail(6, "initial condition upload failed");
    }
    if (auto h = internal::checkDevice("the initial condition"); !h)
        return std::unexpected(h.error());
    const double u0 = r.ramp > 0 ? 0.0 : r.uInf;
    far->setFreestream(u0);
    if (nsc) nsc->setFreestream(u0);
    applyBc(*ds);

    // qw is the wall heat flux (an energy RATE into the body, comp_ghost.hpp);
    // it is identically zero on the default adiabatic wall.
    // A rotor adds thrust and shaft torque, and their coefficients on the
    // propeller convention (n = revolutions per unit time, D the disc
    // diameter). Only when it is spinning: a static run's stats file must keep
    // exactly the columns it has always had.
    // RANS appends its columns (y+ is the linear-sublayer estimate of the
    // staircase wall, DeviceCompGhost::wallYplus) and a steady march its own;
    // a DNS/LES run keeps exactly the columns it always had.
    std::vector<std::string> cols =
        r.rotor.active()
            ? std::vector<std::string>{"t",  "dt", "cd", "cl",     "cs",
                                       "wmax", "qw", "thrust", "torque",
                                       "ct", "cq"}
            : std::vector<std::string>{"t", "dt", "cd", "cl", "cs", "wmax",
                                       "qw"};
    if (rans)
        cols.insert(cols.end(),
                    {"mutMax", "yplusMax", "yplusMean", "floorHits"});
    if (steady)
        cols.insert(cols.end(),
                    {"iter", "resid", "resU", "resE", "resT", "remaining"});
    if (auto e = out.declareSeries("stats", cols); !e)
        return fail(4, e.error());
    const int shaftAxis = rotorAxisIndex(r.rotor.axis);
    const auto [ctDen, cqDen] = internal::propellerDen(r.rhoInf, r.rotor.omega, r.D);

    const float vortScale =
        float(r.curlMax > 0 ? r.curlMax
                            : (r.signedVorticity ? 8.0 : 15.0) * r.uInf / r.D);
    const float speedMax = float(1.6 * r.uInf);
    const char* vortName = r.signedVorticity ? "vort" : "curl";
    // --temp: T - Tinf on the diverging map, so the freestream is the neutral
    // background exactly as --speed-map delta does for velocity; the clip is
    // shared with the O-grid runner (internal/frame_scratch.hpp).
    const double TInf = r.pInf / (r.rhoInf * r.tr.Rgas);
    const double machSq = r.uInf * r.uInf * r.rhoInf / (r.tr.gamma * r.pInf);
    const float tempScale = internal::thermalFrameScale(
        TInf, machSq, r.tr.gamma, r.wallIsothermal, r.Twall);
    // Display raster: on a clustered mesh the frame arrays are resampled onto
    // the uniform physical grid RunOutput was built with (dns/framemap.hpp).
    const bool resample = r.raster && r.raster->active();
    const std::size_t rn = resample ? r.raster->renderGrid().size() : 0;
    std::vector<float> vortR(rn), speedR(rn);
    std::vector<float> tempR(r.tempFrames ? rn : 0);
    std::vector<float> rotorMaskR(r.rotor.active() ? rn : 0);
    internal::FrameScratch fs;
    auto writeFrame = [&](double t) {
        if (!fs.ensure(g.size(), r.tempFrames ? 3 : 2, true)) return;
        if (r.tempFrames) dev->primitives(*ds);
        fs.centreFromConserved(gg, *ds);
        if (r.signedVorticity && stretched)
            vorticityZRemap(gg, remap, fs.u(), fs.v(), fs.vort());
        else if (r.signedVorticity && focal)
            vorticityZRemap(gg, fremap, fs.u(), fs.v(), fs.vort());
        else if (r.signedVorticity)
            vorticityZCentred(gg, fs.u(), fs.v(), fs.vort());
        else curlMagCentred(gg, fs.u(), fs.v(), fs.w(), fs.vort());
        speedCentred(gg, fs.u(), fs.v(), fs.w(), fs.speed());
        fs.take(0, fs.vort());
        fs.take(1, fs.speed(), r.speedDelta ? r.uInf : 0.0);
        if (r.tempFrames) fs.take(2, dev->T(), TInf);
        const float* vortOut = fs.channel(0);
        const float* speedOut = fs.channel(1);
        const float* tempOut = r.tempFrames ? fs.channel(2) : nullptr;
        if (resample)
        {
            r.raster->apply(fs.channel(0), vortR.data());
            r.raster->apply(fs.channel(1), speedR.data());
            vortOut = vortR.data();
            speedOut = speedR.data();
            if (r.tempFrames)
            {
                r.raster->apply(fs.channel(2), tempR.data());
                tempOut = tempR.data();
            }
        }
        const FrameField ff[3]{
            {vortName, vortOut, 0, vortScale, r.signedVorticity,
             r.vortAsinh && r.signedVorticity ? vortScale / 32.0f : 0.0f},
            internal::speedField(speedOut, r.speedDelta, r.uInf, speedMax),
            {"temp", tempOut, -tempScale, tempScale, true}};
        fs.emit(out, t, {ff, r.tempFrames ? 3u : 2u});
    };

    RunStats st;
    double t = 0.0;
    internal::StatsCadence cad{r.statsEvery};
    double dtStab = 0.0;
    int sinceDt = 1 << 30;
    // Potential-correction wake source: EMA of F_drag/(rhoInf uInf), fed only
    // past the quiet-start ramp (the ramp's inertial force is not drag).
    internal::WakeEma wake{.tLast = r.ramp};
    const auto wall0 = std::chrono::steady_clock::now();

    // --- RANS measurements (fluid mask) ------------------------------------
    // max mu_t/mu_inf, the wall's first-cell y+ estimate, the SST guard hits,
    // and the stop integrals sum_fluid V t_i.
    DeviceBuffer<double> vol64;
    DeviceBuffer<std::uint8_t> fluidDev;
    if (rans)
    {
        auto v = DeviceBuffer<double>::make(g.size());
        auto f = DeviceBuffer<std::uint8_t>::make(g.size());
        if (!v || !f || v->upload(mtr->vol.data()) != hipSuccess ||
            f->upload(fluid.data()) != hipSuccess)
            return fail(6, "GPU allocation failed (RANS measurements)");
        vol64 = std::move(*v);
        fluidDev = std::move(*f);
    }
    struct RansExtra
    {
        double mutMax = 0, ypMax = 0, ypMean = 0, hits = 0, turb[2] = {0, 0};
    };
    auto measureRans = [&]() {
        RansExtra x;
        dev->updateMut(*ds);
        x.mutMax = internal::reduce(
            *red, g.size(),
            internal::MaskedMax{fluidDev.data(), dev->mut().data(),
                                1.0 / r.tr.muRef},
            rocprim::maximum<double>(), -HUGE_VAL);
        const auto yp = ghost->wallYplus(*dev, *ds);
        x.ypMax = yp.max;
        x.ypMean = yp.mean;
        x.hits = double(dev->floorHits());
        for (int i = 0; i < ds->nT; ++i)
            x.turb[i] = internal::reduce(
                *red, g.size(),
                internal::MaskedVolSum{fluidDev.data(), vol64.data(),
                                       ds->t[i].data(), 0.0},
                rocprim::plus<double>(), 0.0);
        return x;
    };
    const double span = g.nz > 1 ? g.nz * g.hz : 1.0;

    // --emit surface on the immersed ghost wall, once at the end: the tally
    // per interface face, split into pressure / convective / viscous
    // (DeviceCompGhost::surfaceFaces), gauge on pInf. The header line carries
    // what a reader needs to normalize: the moment pivot, qRef * span and the
    // propeller denominators. Reads the state only, so the run is unchanged.
    // Its label is the run's: the RANS label when there is one, else the LES
    // label or DNS, plus the frozen-rotor note on an MRF run (r.label is set
    // only on a RANS run -- the progress line of a DNS/LES run is unchanged).
    auto surfaceLabel = [&] {
        std::string l = !r.label.empty() ? r.label
                        : r.les > 0      ? lesRunLabel(r.sgs, r.les, g.nz == 1)
                                         : std::string("DNS");
        if (r.rotor.active() && r.rotorMrf)
            l += " [MRF frozen rotor: an APPROXIMATION]";
        return l;
    };
    auto writeSurface = [&]() -> std::expected<void, RunError> {
        if (!out.wantsSurface()) return {};
        auto fs = ghost->surfaceFaces(*dev, *ds, r.pInf);
        if (!fs) return fail(6, "the surface tally split failed");
        if (auto h = internal::checkDevice("the surface tally", &*red); !h)
            return std::unexpected(h.error());
        std::vector<std::vector<double>> rows;
        rows.reserve(fs->size());
        for (const auto& f : *fs)
            rows.push_back({f.x[0],  f.x[1],  f.x[2],  double(f.axis),
                            double(f.sign), f.area, f.s[0], f.s[1], f.s[2],
                            f.p,     f.f[0],  f.f[1],  f.f[2],  f.fp[0],
                            f.fp[1], f.fp[2], f.fc[0], f.fc[1], f.fc[2],
                            f.fv[0], f.fv[1], f.fv[2]});
        const std::string cs[] = {"x",   "y",   "z",   "axis", "sign", "area",
                                  "sx",  "sy",  "sz",  "p",    "fx",   "fy",
                                  "fz",  "fpx", "fpy", "fpz",  "fcx",  "fcy",
                                  "fcz", "fvx", "fvy", "fvz"};
        const auto& pv = r.rotor.shaft;
        const std::string comment = std::format(
            "{} | immersed ghost wall: one row per interface face, flux fluid "
            "-> body in solver units, gauge on pInf {:.10g} | pivot {:.10g} "
            "{:.10g} {:.10g} | qRef*span {:.10g} | ctDen {:.10g} | cqDen "
            "{:.10g} | shaft {}",
            surfaceLabel(), r.pInf, pv[0],
            pv[1], pv[2], r.qRef * span, ctDen, cqDen,
            r.rotor.active() ? shaftAxis : -1);
        if (auto e = out.writeSurface(comment, cs, rows, 10); !e)
            return fail(4, e.error());
        return {};
    };

    if (steady)
    {
        // The pseudo-time march (docs/rans.md, "Immersed-wall RANS"): the
        // stop is on cd and the turbulence integrals, never the residual,
        // whose norms are reported in freestream units (mom D/U^2, energy
        // D/U^3, SA nu~ 1/U^2, SST k D/U^3 and ln omega D/U).
        const bool sst = ransSst(r.rans);
        const double tU = r.D / r.uInf;
        double resid = HUGE_VAL, resU = 0, resE = 0, resT = 0;
        DeviceCompGhost::Force F{};
        RansExtra x;
        internal::SteadyMarchIo io;
        io.afterStep = [&](DeviceCompState& s) { applyBc(s); };
        io.sample = [&](long) -> std::expected<std::vector<double>, RunError> {
            const ResidualNorms n = dev->residual(*ds, *red);
            F = ghost->force(*dev, *ds, r.rotor.shaft);
            st.cd = F.fx / (r.qRef * span);
            st.cl = F.fy / (r.qRef * span);
            st.cs = F.fz / (r.qRef * span);
            x = measureRans();
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            resU = n.mom * tU / r.uInf;
            resE = n.E * tU / (r.uInf * r.uInf);
            resT = sst ? std::max(n.t[0] * tU / (r.uInf * r.uInf), n.t[1] * tU)
                       : n.t[0] / (r.uInf * r.uInf);
            resid = std::max({resU, resE, resT});
            if (!std::isfinite(resid) || !std::isfinite(st.cd))
                return fail(7, "the residual is not finite (the state "
                               "diverged)");
            std::vector<double> q = {st.cd};
            for (int i = 0; i < ds->nT; ++i) q.push_back(x.turb[i]);
            return q;
        };
        io.row = [&](const internal::SteadyRow& sr) {
            const double wmax =
                r.cfl / sr.dtRef * std::min({g.hx, g.hy, g.hz});
            const double row[] = {sr.t,     sr.dtRef,  st.cd,    st.cl,
                                  st.cs,    wmax,      F.heat,   x.mutMax,
                                  x.ypMax,  x.ypMean,  x.hits,
                                  double(sr.iter), resid, resU, resE, resT,
                                  sr.remaining};
            out.row("stats", row);
            if (!sr.print) return;
            std::printf("iter=%ld cd=%.5f cl=%+.2e mutMax=%.3g y+max=%.3g "
                        "remaining=%.2e resid=%.3e (U %.2e E %.2e T %.2e) "
                        "dtRef=%.2e %s\n",
                        sr.iter, st.cd, st.cl, x.mutMax, x.ypMax, sr.remaining,
                        resid, resU, resE, resT, sr.dtRef, r.label.c_str());
            std::fflush(stdout);
        };
        const internal::SteadyMarchParams sp{.cfl = r.cfl,
                                             .dtMargin = r.dtMargin,
                                             .collapse = 1e-6,
                                             .tol = r.steadyTol,
                                             .dtRecompute = r.dtRecompute,
                                             .cap = r.steadySteps,
                                             .statsEvery = r.statsEvery,
                                             .series = 1 + ds->nT,
                                             .observables = 1};
        auto res = internal::explicitSteadyMarch(*dev, *ds, *red, sp, io);
        if (!res) return std::unexpected(res.error());
        {
            const char* names[3] = {"cd", sst ? "int rho k" : "int rho nu~",
                                    "int rho omega"};
            std::fputs(internal::steadyFloorReport(
                           res->floors,
                           std::span<const char* const>(names, 1 + ds->nT))
                           .c_str(),
                       stdout);
            std::fflush(stdout);
        }
        st.steps = res->steps;
        st.tStop = res->t;
        if (auto h = internal::checkDevice("the end of the run", &*red); !h)
            return std::unexpected(h.error());
        if (auto e = writeSurface(); !e) return std::unexpected(e.error());
        finishTiming(st, wall0, out);
        if (!res->converged && !stopRequested())
            return fail(8, std::format("steady state not reached in {} "
                                       "iterations (remaining change {:.3e} "
                                       "> {:.3g})",
                                       res->steps, res->remaining,
                                       r.steadyTol));
        st.converged = res->converged;
        return st;
    }

    while (t < r.tEnd)
    {
        // A stop signal ends the run the SAME way reaching tEnd does: return
        // normally, so every destructor runs -- the stats file is flushed and
        // the shared-memory segment is unlinked. Terminating mid-step instead
        // leaves a segment that outlives its own run, flagged as live.
        if (stopRequested()) break;
        // 1e-6: the compressible collapse threshold. NOT shared with the
        // channel's 1e-9 -- the dt scales differ.
        if (auto e = internal::recomputeDt(
                sinceDt, dtStab, r.dtRecompute, r.dtMargin, 1e-6,
                [&] { return dev->stableDt(*ds, *red, r.cfl); });
            !e)
            return fail(7, e.error());
        const double dt = std::min(r.tEnd - t, dtStab);

        // Advance the geometry to this step's azimuth, ONCE per step rather
        // than once per RK stage. The tip moves a small fraction of a cell in a
        // step at an acoustic dt, so the intra-step lag is far below the wall's
        // own discretization error, while a per-stage swap would triple the
        // copies to chase it. The bin comes from the absolute time, so it
        // cannot drift; a step crossing several bins simply lands on the last.
        if (rotorTable)
        {
            const int b = rotorBin(r.rotor, rotorAzimuth(r.rotor, t));
            if (b != rotorBinLoaded)
            {
                if (ghost->load(rotorTable->bins[std::size_t(b)]) != hipSuccess)
                    return fail(6, "rotor azimuth swap failed");
                rotorBinLoaded = b;
                ++rotorSwaps;
            }
        }

        penDt = dt / 3.0;
        dev->step(*ds, Real(dt));
        if (t < r.ramp)
        {   // Quiet start: boost the frame, then retarget the far field.
            const double uNew = r.uInf * rampFrac(t + dt, r.ramp);
            if (nsc)
            {
                boostX(gg, *ds, uNew - nsc->freestream());
                nsc->setFreestream(uNew);
                far->setFreestream(uNew);
            }
            else
            {
                far->boost(*ds, uNew - far->freestream());
                far->setFreestream(uNew);
            }
        }
        applyBc(*ds);   // leave the state BC/ghost-consistent between steps
        t += dt;
        ++st.steps;

        if (cad.due(st.steps, t, r.tEnd))
        {
            const auto F = ghost->force(*dev, *ds, r.rotor.shaft);
            const RansExtra rx = rans ? measureRans() : RansExtra{};
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            st.cd = F.fx / (r.qRef * span);
            st.cl = F.fy / (r.qRef * span);
            st.cs = F.fz / (r.qRef * span);
            if (nsc && r.nscbcPot && t > r.ramp)
            {
                const double Qinst = std::clamp(F.fx / (r.rhoInf * r.uInf),
                                                0.0, 2.0 * r.D * r.uInf * span);
                nsc->setWakeSource(wake.update(t, Qinst, r.D, r.uInf));
            }
            const double wmax =
                r.cfl / dtStab * std::min({g.hx, g.hy, g.hz});
            const double f[3] = {F.fx, F.fy, F.fz};
            const double q[3] = {F.qx, F.qy, F.qz};
            // Thrust is the force along the shaft, torque the moment about it;
            // both signed, so a rotor pushing the wrong way reads negative
            // rather than plausible.
            const double thrust = f[shaftAxis], torque = q[shaftAxis];
            if (r.rotor.active())
            {
                const double row[] = {t,      dt,     st.cd,  st.cl,
                                      st.cs,  wmax,   F.heat, thrust,
                                      torque, ctDen > 0 ? thrust / ctDen : 0.0,
                                      cqDen > 0 ? torque / cqDen : 0.0};
                out.row("stats", row);
            }
            else if (rans)
            {
                const double row[] = {t,        dt,       st.cd,     st.cl,
                                      st.cs,    wmax,     F.heat,    rx.mutMax,
                                      rx.ypMax, rx.ypMean, rx.hits};
                out.row("stats", row);
            }
            else
            {
                const double row[] = {t, dt, st.cd, st.cl, st.cs, wmax, F.heat};
                out.row("stats", row);
            }
            if (cad.printDue(st.steps, t, r.tEnd))
            {
                if (r.rotor.active())
                    std::printf("t=%.2f/%.2f T=%+.4g Q=%+.4g cd=%.3f "
                                "wmax=%.2f dt=%.2e\n",
                                t, r.tEnd, thrust, torque, st.cd, wmax, dt);
                else if (rans)
                    std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f cs=%+.3f "
                                "wmax=%.2f dt=%.2e mutMax=%.3g y+max=%.3g "
                                "%s\n",
                                t, r.tEnd, st.cd, st.cl, st.cs, wmax, dt,
                                rx.mutMax, rx.ypMax, r.label.c_str());
                else
                    std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f cs=%+.3f "
                                "wmax=%.2f dt=%.2e\n",
                                t, r.tEnd, st.cd, st.cl, st.cs, wmax, dt);
                std::fflush(stdout);
                cad.printed(st.steps);
            }
            cad.logged(st.steps);
        }
        // The output schedule is independent of the quiet-start ramp: frames
        // during spin-up show the (physical, ramped) transient. The ramp gate
        // that used to suppress them hid the first `ramp` time units of every
        // run for no numerical reason -- the sinks pace themselves via
        // fdt/pdt/wdt either way.
        if (out.wantsAnyFrame(t))
        {
            if (rotorTable && rotorMaskBin != rotorBinLoaded)
            {
                if (rotorMaskBin >= 0)
                    for (int c : rotorTable->bins[std::size_t(rotorMaskBin)].solid)
                        rotorMask[std::size_t(c)] = 1.0f;
                for (int c : rotorTable->bins[std::size_t(rotorBinLoaded)].solid)
                    rotorMask[std::size_t(c)] = 0.0f;
                rotorMaskBin = rotorBinLoaded;
                // Through the SAME display raster the fields go through: on a
                // focal mesh RunOutput was built with a render-grid mask, and
                // handing it a solver-grid one would be a size mismatch, not a
                // slightly wrong picture.
                if (resample)
                {
                    r.raster->apply(rotorMask.data(), rotorMaskR.data());
                    out.updateMask(rotorMaskR);
                }
                else
                    out.updateMask(rotorMask);
            }
            writeFrame(t);
        }
    }
    if (auto h = internal::checkDevice("the end of the run", &*red); !h)
        return std::unexpected(h.error());
    // Swaps per step is the diagnostic that says whether the bin count matched
    // the time step: far below 1 means most steps saw a body that had not
    // moved (correct, and free), while 1 means every step crossed a bin and the
    // tip is outrunning the table.
    if (rotorTable && st.steps > 0)
        std::printf("rotor: %ld azimuth swaps over %ld steps (%.2f per step, "
                    "%.1f revolutions)\n",
                    rotorSwaps, st.steps, double(rotorSwaps) / double(st.steps),
                    std::abs(r.rotor.omega) * t / (2.0 * std::numbers::pi));
    if (auto e = writeSurface(); !e) return std::unexpected(e.error());
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
