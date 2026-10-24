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
#include "dns/gpu/internal/run_loop.hpp"
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
    auto dev = mtr ? DeviceComp::make(*mtr, r.tr, r.les, r.sgs)
                   : DeviceComp::make(g, r.tr, r.les, r.sgs);
    auto ds = DeviceCompState::make(g.size());
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
        GhostWallVelocity uw = [rot = r.rotor](double x, double y, double z2) {
            return rotorWallVelocity(rot, {x, y, z2});
        };
        const CompGhostTables tbls = classifyCompGhost(
            g, sdf, bbox ? bbox->data() : nullptr, fremap, &uw);
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
        // the one frozen azimuth.
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
    const std::vector<std::string> cols =
        r.rotor.active()
            ? std::vector<std::string>{"t",  "dt", "cd", "cl",     "cs",
                                       "wmax", "qw", "thrust", "torque",
                                       "ct", "cq"}
            : std::vector<std::string>{"t", "dt", "cd", "cl", "cs", "wmax",
                                       "qw"};
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
            if (auto h = internal::checkDevice("a stats row", &*red); !h)
                return std::unexpected(h.error());
            const double span = g.nz > 1 ? g.nz * g.hz : 1.0;
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
    finishTiming(st, wall0, out);
    return st;
}

} // namespace dns::gpu
