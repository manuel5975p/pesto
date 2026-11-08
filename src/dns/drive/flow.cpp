// dns flow on the structured solvers: incompressible, compressible (Cartesian, O-grid), sequenced scenes.
#include "dns/drive/internal.hpp"

#include "dns/casegeom.hpp"
#include "dns/config.hpp"
#include "dns/flow_ic.hpp"
#include "dns/framemap.hpp"
#include "dns/geometry.hpp"
#include "dns/meshio.hpp"
#include "dns/ogrid.hpp"
#include "dns/operators.hpp"
#include "dns/output.hpp"
#include "dns/remap.hpp"
#include "dns/scene.hpp"
#include "dns/transport.hpp"
#include "dns/walldist.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/comp.hpp"
#include "dns/gpu/runner_comp.hpp"
#include "dns/gpu/runner_incomp.hpp"
#include "dns/gpu/runner_ogrid.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <fstream>
#include <iterator>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace dns::drive {

using std::numbers::pi;
#if DNS_WITH_GPU
namespace {


struct Freestream
{
    double pInf, mu, Tinf;
    Transport gas;
};

// The freestream state and gas of a compressible body run on diameter D. Tinf
// is the box-unit temperature T = p / (rho R) = 1 / (gamma M^2): the reference
// the viscosity law and the wall ratio are measured against, and the
// dimensional anchor --tinf maps onto (dns/transport.hpp).
Freestream compFreestream(const Case& c, double D)
{
    const double gamma = kGammaAir, rhoInf = kRhoInf, Rgas = kRgasBox;
    const double pInf = rhoInf * c.uInf * c.uInf / (gamma * c.mach * c.mach);
    const double mu = rhoInf * c.uInf * D / c.re;
    const double Tinf = pInf / (rhoInf * Rgas);
    return {pInf, mu, Tinf,
            Transport::air(mu, Tinf, c.tinfK, c.prandtl, gamma, Rgas,
                           c.sutherland)};
}
} // namespace


int runFlowIncompressible(const Case& c, CaseGeometry& geo,
                          const VectorField* ic0,
                          VectorField* finalOut,
                          bool suppressOutput)
{
    const auto& g = c.grid;
    const auto st = sanitizeCut(g, geo.cut);   // before the ring: body cells only
    const bool fringe = c.far == FarField::Fringe || c.far == FarField::Periodic;
    // --body floor: the slab crosses the inlet/outlet planes. Measure the
    // fully-solid interior row count and the cut row's open fraction from the
    // BODY cut now -- closeBoxBoundary is about to zero these very apertures
    // -- so the tunnel BC can zero its Dirichlet/convective values there and
    // keep the prescribed fluxes and global mass balance exact.
    // (measureFloorRows carries the sliver-promotion rule and its measurement;
    // see dns/geometry.hpp.)
    int floorRows = -1;
    double floorFrac = 1.0;
    if (c.bodyKind == BodyKind::Floor)
    {
        const FloorRows f = measureFloorRows(g, geo.cut);
        floorRows = f.rows;
        floorFrac = f.frac;
    }
    if (!fringe) closeBoxBoundary(g, geo.cut); // fringe: the ring stays periodic
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed, st.minLambda,
                st.minAperture);

    // Frontal area / reference dynamic pressure: one definition, in the
    // geometry layer (dns/geometry.hpp).
    const ReferenceQ rq = referenceQIncomp(g, geo.cut, 1.0, c.uInf);
    if (!rq.measured)
        return die(exitInput, "the body produced no solid cells (too small?)");
    const double qRef = rq.q;

    // IC: freestream + near-wake kick (dns/flow_ic.hpp). With grid sequencing
    // the coarse level's converged field replaces it.
    auto U = VectorField::zeros(g);
    if (ic0) U = *ic0;
    else incompressibleBodyIC(c, geo, U);

    // Phase 3c: everything the run will write is created and checked now.
    // The written window is the interior: the ghost ring never reaches a frame.
    const auto gg = gpu::GpuGrid::from(g);
    FrameGeometry fg = FrameGeometry::centred(g, geo.D);
    fg.mask.assign(g.size(), 1.0f);
    for (std::size_t q = 0; q < g.size(); ++q) fg.mask[q] = float(geo.cut.lambda[q]);
    fg.i0 = 1;
    fg.j0 = 1;
    fg.k0 = gg.kInLo();
    fg.w = g.nx - 2;
    fg.h = g.ny - 2;
    fg.d = gg.kInCount();

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"re\":{}", c.re), suppressOutput);
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::IncompRun rr;
    rr.nu = c.nu;
    rr.uInf = c.uInf;
    rr.qRef = qRef;
    rr.D = geo.D;
    rr.cfl = c.cfl;
    rr.tEnd = c.ramp + c.tEnd;
    rr.tol = c.tol;
    rr.maxIters = c.maxIters;
    rr.les = c.cs;
    rr.sgs = c.sgsModel;
    rr.cutFlux = c.cutFlux;
    rr.penalize = c.penalize;
    rr.ramp = c.ramp;
    // The dt reduction is one mandatory host sync per step; --dt-recompute
    // amortizes it exactly as on the compressible path. Defaults to 1 here,
    // so no incompressible trajectory moves unless the user asks.
    rr.dtRecompute = c.dtRecompute;
    rr.vortMax = c.vortMax * c.uInf / geo.D;   // flag is in uInf/D units
    rr.vortAsinh = c.vortAsinh;
    rr.speedDelta = c.speedDelta;
    rr.fringe = fringe;
    rr.open = c.open;
    rr.pot = c.farPot;
    rr.steadyTol = c.steadyTol;
    rr.cx = c.placement.cx;
    rr.cy = c.placement.cy;
    rr.cz = c.placement.cz;
    rr.fringeCells = c.spongeCells;
    rr.fringeAxes = c.fringeAxes;
    rr.floorRows = floorRows;
    rr.floorFrac = floorFrac;

    rr.box = c.bodyKind == BodyKind::Scene
                 ? gpu::bodyBox(gg, geo.bbox)
                 : gpu::bodyBox(gg, c.placement.cx, c.placement.cy,
                                c.placement.cz, 0.5 * geo.D);
    if (g.nz == 1 || geo.zSpans)
    {   // z-invariant bodies span the whole interior in z.
        rr.box.k0 = gg.kInLo();
        rr.box.k1 = gg.kInHi();
    }
    if (c.bodyKind == BodyKind::Floor)
    {   // the slab spans every interior column (ghost ring is one cell).
        rr.box.i0 = 1;
        rr.box.i1 = gg.nx - 2;
    }

    // Per-object force integration: one tally box per scene object. Only
    // emitted when the boxes are pairwise disjoint -- overlapping boxes would
    // each count the shared surface, so those scenes get the total only.
    if (geo.objectBoxes.size() > 1) {
        auto t = gpu::objectTallyBoxes(gg, double(g.nz) * g.hz, geo.objectBoxes);
        if (t) rr.tallies = std::move(*t);
        else
            std::fprintf(stderr,
                         "dns: warning: force boxes of '%s' and '%s' overlap; per-object forces suppressed (total "
                         "cd/cl only)\n",
                         t.error().first.c_str(), t.error().second.c_str());
    }

    auto res = gpu::runIncompressible(g, geo.cut, geo.sdf, U, rr, *out, finalOut);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
}

int runFlowOGrid(const Case& c, CaseGeometry& geo)
{
    const auto& g = c.grid;
    const double rhoInf = kRhoInf;
    const Freestream fs = compFreestream(c, geo.D);
    const double pInf = fs.pInf, Tinf = fs.Tinf;
    const Transport& gas = fs.gas;

    const OGridSpec spec{.cx = c.placement.cx,
                         .cy = c.placement.cy,
                         .cz = c.placement.cz,
                         .rFar = c.ogridRFar,
                         .beta = c.ogridBeta,
                         .span = c.ogridSpan};
    const RevolvedSpec rev =
        c.ogridNphi > 0
            ? RevolvedSpec{.wedge = false, .dphi = 2.0 * pi / c.ogridNphi,
                           .nphi = c.ogridNphi}
            : RevolvedSpec{.wedge = true, .dphi = c.ogridWedgeDeg * pi / 180.0};
    auto og = c.ogridRevolved ? buildRevolvedOGrid(g, geo.sdf, spec, rev)
                              : buildOGrid(g, geo.sdf, spec);
    if (!og) return die(exitInput, "o-grid: " + og.error());
    std::printf("# o-grid: rWall %.4f..%.4f  h1 %.5g (%.1f cells in D/sqrt(Re))"
                "  growth %.4f  wall-normal dev %.3g deg\n",
                og->rWallMin, og->rWallMax, og->h1,
                og->h1 > 0 ? geo.D / std::sqrt(c.re) / og->h1 : 0.0,
                og->growth, og->normalDevMax * 180.0 / pi);

    // --init-from (full revolution): the converged wedge, read and checked
    // against this mesh before any output exists.
    std::optional<WedgeStart> wedgeStart;
    if (!c.ogridInitFrom.empty())
    {
        auto slurp = [](const std::string& p) -> std::optional<std::vector<char>> {
            std::ifstream f(p, std::ios::binary);
            if (!f) return std::nullopt;
            return std::vector<char>(std::istreambuf_iterator<char>(f), {});
        };
        const std::string mp = c.ogridInitFrom + "_mesh.bin",
                          fp = c.ogridInitFrom + "_final.bin";
        const auto mb = slurp(mp), fb = slurp(fp);
        if (!mb || !fb)
            return die(exitInput, std::format("--init-from: cannot read {} (a "
                                              "wedge run with --emit mesh "
                                              "writes it)",
                                              !mb ? mp : fp));
        std::vector<std::string> names;
        if (c.rans == RansModel::SaNeg) names = {"nutilde"};
        else if (ransSst(c.rans)) names = {"k", "omega"};
        const double centre[3] = {c.placement.cx, c.placement.cy,
                                  c.placement.cz};
        auto ws = revolveWedgeState(*mb, *fb, *og, centre, names, pInf);
        if (!ws)
            return die(exitInput, "--init-from " + c.ogridInitFrom + ": " +
                                      ws.error());
        wedgeStart = std::move(*ws);
        std::printf("# initial state: the wedge %s revolved over %d planes\n",
                    c.ogridInitFrom.c_str(), g.nz);
    }

    // Frames are the index-space raster of the annulus (runner_ogrid.hpp says
    // why), so the render grid IS the logical grid and there is no mask: every
    // cell is fluid.
    FrameGeometry fg = FrameGeometry::centred(g, geo.D);
    fg.mask.assign(g.size(), 1.0f);

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"re\":{},\"mach\":{}", c.re,
                       c.mach));
    auto out = RunOutput::make(oc, c.meta, g, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::OGridRun rr;
    rr.tr = gas;
    rr.rhoInf = rhoInf;
    rr.uInf = c.uInf;
    rr.pInf = pInf;
    rr.thermal = c.bodyIsothermal ? gpu::OGridThermal::Isothermal
                                  : gpu::OGridThermal::Adiabatic;
    rr.Twall = c.twallRatio * Tinf;
    // The resolved rotor, verbatim (validator-owned); on the O-grid only
    // --rotor-frame mrf reaches here, and active() is false without
    // --rotor-mach.
    rr.rotor = c.rotor;
    rr.rotorMrf = c.rotorMrf;
    rr.D = geo.D;
    // Per unit span on a z-extruded mesh (the runner divides by the span);
    // the frontal disc pi D^2/4 on a revolved one (a whole body).
    rr.qRef = 0.5 * rhoInf * c.uInf * c.uInf *
              (c.ogridRevolved ? 0.25 * pi * geo.D * geo.D : geo.D);
    rr.les = c.cs;
    rr.sgs = c.sgsModel;
    rr.cfl = c.cfl;
    rr.tEnd = c.ramp + c.tEnd;
    rr.ramp = c.ramp;
    rr.dtRecompute = c.dtRecompute;
    rr.signedVorticity = c.vortSigned || g.nz == 1;
    rr.curlMax = c.vortMax * c.uInf / geo.D;
    rr.vortAsinh = c.vortAsinh;
    rr.speedDelta = c.speedDelta;
    rr.tempFrames = c.tempFrames;
    rr.cx = c.placement.cx;
    rr.cy = c.placement.cy;
    rr.cz = c.placement.cz;

    // RANS / steady (docs/rans.md, "dns flow"): the exact distance to the
    // DISCRETE wall polyline, the freestream scalars, the run label. The
    // wall distance also feeds d1 and y+ of a laminar steady or --emit
    // surface run.
    const bool rans = c.rans != RansModel::None;
    rr.rans = c.rans;
    rr.phiInf[0] = c.ransPhiInf[0];
    rr.phiInf[1] = c.ransPhiInf[1];
    rr.steadyTol = c.compSteady ? c.steadyTol : 0.0;
    rr.steadySteps = c.steadySteps;
    rr.steadyImplicit =
        c.compSteady && c.steadyScheme == SteadyScheme::Implicit;
    rr.compensated = c.compensated;
    // --steady-sample: the explicit march's stop/stats cadence (the validator
    // rejects it with the implicit one); a time march keeps its 200.
    if (c.compSteady)
    {
        rr.statsEvery = c.steadySample;
        rr.ltsMax = c.ltsMax;
        rr.precondK = c.precondK;
    }
    rr.label = flowRunLabel(c);
    if (rans || c.compSteady || c.emit.contains(Emit::Surface) ||
        c.emit.contains(Emit::Mesh))
        rr.wallDist = c.ogridRevolved
                          ? revolvedWallDistance(og->mesh, og->nTheta,
                                                 og->rev.wedge)
                          : ogridWallDistance(og->mesh);
    // --emit mesh: the mesh and its d1 now, before any device allocation.
    if (out->wantsMesh())
    {
        const double centre[3] = {c.placement.cx, c.placement.cy,
                                  c.placement.cz};
        const auto bytes =
            encodeMesh(*og, centre, wallFirstCellDistance(g, rr.wallDist));
        if (auto e = out->writeMeshArtefact("mesh", bytes); !e)
            return die(exitOutput, e.error());
    }
    if (!rr.label.empty()) std::printf("# %s\n", rr.label.c_str());

    // Quiet start at rest (--ramp 0: the freestream), --init-from's revolved wedge, the
    // trip (dns/flow_ic.hpp); RANS scalars at their freestream value everywhere.
    CompState ic = ogridCompressibleIC(c, geo, og->mesh, rhoInf, pInf, wedgeStart ? &*wedgeStart : nullptr,
                                             std::span(rr.icTurb, std::size_t(ransScalars(c.rans))));

    const auto res = gpu::runOGrid(g, *og, ic, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    if (auto e = out->finish(); !e) return die(exitOutput, e.error());
    std::printf("done: %ld steps, cd=%.4f cl=%.4f\n", res->steps, res->cd,
                res->cl);
    return 0;
}

int runFlowCompressible(const Case& c, CaseGeometry& geo)
{
    const auto& g = c.grid;
    const double gamma = kGammaAir, rhoInf = kRhoInf;
    const Freestream fs = compFreestream(c, geo.D);
    const double pInf = fs.pInf, Tinf = fs.Tinf;
    const Transport& gas = fs.gas;

    // Frontal area / reference dynamic pressure (used for qRef ONLY, never for
    // the solve -- the compressible body is the sharp ghost wall). One
    // definition, in the geometry layer; see geometry.hpp for why REFERENCE
    // spacings are the correct weights even under --focal (measured).
    const AxisRemap3 fRemap = focalRemap(g, c.focalActive, c.focalLo,
                                         c.focalHi, c.focalW, c.focalRatio);
    ReferenceQ rq;
    if (c.bodyKind != BodyKind::None)
    {
        auto sc = geo.cut;
        (void)sanitizeCut(g, sc);
        rq = referenceQComp(g, sc, rhoInf, c.uInf, geo.D);
    }
    else
        rq.q = 0.5 * rhoInf * c.uInf * c.uInf * geo.D;
    const double qRef = rq.q;

    // IC: tapered freestream + solenoidal wake trip (dns/flow_ic.hpp). The
    // mesh maps go in because the taper and the trip must sit where the cells
    // physically are; both are the exact identity on a uniform grid.
    const TanhStretchY yRemap{double(g.ny) * g.hy, c.stretchBeta};
    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    compressibleBodyIC(c, geo, fRemap, yRemap, rhoInf, pInf, gamma, host, mask);

    // Display raster for the active mesh maps (dns/framemap.hpp).
    const PhysicalRaster raster = rasterFor(g, fRemap, yRemap);
    const Grid& rg = raster.renderGrid();

    FrameGeometry fg = FrameGeometry::centred(rg, geo.D);
    fg.mask = raster.apply(mask);

    RunMeta rmeta = c.meta;
    if (raster.active()) rmeta.setGrid(rg);

    const auto oc = outputConfig(
        c, std::format("{{\"app\":\"dns\",\"re\":{},\"mach\":{}", c.re, c.mach));
    auto out = RunOutput::make(oc, rmeta, rg, std::move(fg));
    if (!out) return die(exitOutput, out.error());

    gpu::CompRun rr;
    rr.raster = &raster;
    rr.tr = gas;
    rr.wallIsothermal = c.bodyIsothermal;
    rr.Twall = c.twallRatio * Tinf;
    rr.nscbcThermalInflow = c.nscbcThermalInflow;
    rr.uInf = c.uInf;
    rr.rhoInf = rhoInf;
    rr.pInf = pInf;
    rr.D = geo.D;
    rr.qRef = qRef;
    rr.cfl = c.cfl;
    rr.tEnd = c.ramp + c.tEnd;
    rr.les = c.cs;
    rr.sgs = c.sgsModel;
    // No compressible --cut-flux (validator-rejected; see gpu/comp.hpp), so
    // --penalize is the only consumer of the aperture geometry here.
    rr.cutGeom = c.penalize > 0 ? &geo.cut : nullptr;
    rr.penalize = c.penalize;
    rr.curlMax = c.vortMax * c.uInf / geo.D;   // flag is in uInf/D units
    rr.vortAsinh = c.vortAsinh;
    rr.speedDelta = c.speedDelta;
    rr.tempFrames = c.tempFrames;
    rr.thinSeal = c.thinSeal;
    rr.far = c.far;
    rr.open = c.open;
    rr.spongeCells = c.spongeCells;
    rr.nscbcBeta = c.nscbcBeta;
    rr.nscbcAbc = c.nscbcAbc;
    rr.nscbcEtaIn = c.nscbcEtaIn;
    rr.nscbcEtaInShear = c.nscbcEtaInShear;
    rr.nscbcEtaLat = c.nscbcEtaLat;
    rr.nscbcStrip = c.nscbcStrip;
    rr.nscbcSink = c.nscbcSink;
    rr.nscbcSinkAmp = c.nscbcSinkAmp;
    rr.nscbcPot = c.farPot;
    rr.cx = c.placement.cx;
    rr.cy = c.placement.cy;
    rr.cz = c.placement.cz;
    rr.ramp = c.ramp;
    rr.dtRecompute = c.dtRecompute;
    rr.stretchBeta = c.stretchBeta;
    rr.noslipY = c.noslipY;
    rr.focalActive = c.focalActive;
    rr.focalRatio = c.focalRatio;
    for (int a = 0; a < 3; ++a)
    {
        rr.focalLo[a] = c.focalLo[a];
        rr.focalHi[a] = c.focalHi[a];
        rr.focalW[a] = c.focalW[a];
    }
    // --body floor: fully-solid row count at the x planes, same rule and same
    // implementation as the incompressible tunnel (dns/geometry.hpp).
    if (c.bodyKind == BodyKind::Floor)
        rr.floorRows = measureFloorRows(g, geo.cut).rows;
    // 3D runs report |curl| (sequential map) unless --vort-signed asks for
    // the centre-plane signed wz; native 2D always reports signed wz so the
    // diverging map keeps the sign of the shed vortices.
    rr.signedVorticity = g.nz == 1 || c.vortSigned;
    // The resolved rotor, verbatim: the validator owns the shaft, rate, bins
    // and dilation, and `active()` is false unless --rotor-mach was given.
    rr.rotor = c.rotor;
    rr.rotorMrf = c.rotorMrf;
    rr.hFine = hFine(c);
    rr.compensated = c.compensated;
    // RANS on the immersed ghost wall (docs/rans.md, "Immersed-wall RANS"):
    // d = |sdf| at the physical centres, the freestream scalars, the label.
    if (c.rans != RansModel::None)
    {
        rr.rans = c.rans;
        rr.wallDist = immersedWallDistance(g, geo.sdf, fRemap);
        rr.phiInf[0] = c.ransPhiInf[0];
        rr.phiInf[1] = c.ransPhiInf[1];
        rr.steadyTol = c.compSteady ? c.steadyTol : 0.0;
        rr.steadySteps = c.steadySteps;
        rr.label = flowRunLabel(c);
        std::printf("# %s\n", rr.label.c_str());
    }
    // The frozen rotor's laminar steady march (--rotor-frame mrf --steady):
    // the same pseudo-time knobs, and its label.
    else if (c.compSteady)
    {
        rr.steadyTol = c.steadyTol;
        rr.steadySteps = c.steadySteps;
        rr.label = flowRunLabel(c);
        std::printf("# %s\n", rr.label.c_str());
    }
    // --shock: the label is not the empty DNS/LES one any more.
    else if (c.shock != ShockMode::None)
    {
        rr.label = flowRunLabel(c);
        std::printf("# %s\n", rr.label.c_str());
    }
    rr.shock = c.shock;
    rr.shockDtAuto = c.shockDtAuto;
    // --steady-sample: the stop/stats cadence of the pseudo-time march only;
    // a time march keeps its 200.
    if (c.compSteady)
    {
        rr.statsEvery = c.steadySample;
        rr.ltsMax = c.ltsMax;
        rr.precondK = c.precondK;
    }

    auto res = gpu::runCompressible(g, host, geo.sdf,
                                    geo.hasBbox ? &geo.bbox : nullptr, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
}

int runFlowSequenced(const Case& c, const RawConfig& raw, const Scene* scene,
                     CaseGeometry& fine)
{
    const int L = c.seqLevels;
    VectorField carry = VectorField::zeros(c.grid);   // replaced at level 0
    const VectorField* ic = nullptr;
    Grid gPrev = c.grid;
    double hPrev = c.h;

    for (int l = 0; l < L; ++l)
    {
        const int shift = L - 1 - l;
        RawConfig lr = raw;
        lr.n = raw.n >> shift;
        if (raw.given.contains("--ny")) lr.ny = std::max(1, raw.ny >> shift);
        if (raw.given.contains("--nz") && raw.nz > 1)
            lr.nz = std::max(4, raw.nz >> shift);
        lr.seq = 1;   // the level itself is a plain steady solve

        auto lc = validate(lr);
        if (!lc) { std::fputs(format(lc.error()).c_str(), stderr); return lc.error().code; }

        const bool last = l == L - 1;
        std::optional<CaseGeometry> coarse;
        if (!last)
        {
            auto g = buildCaseGeometry(*lc, scene);
            if (!g) return die(exitInput, g.error());
            coarse = std::move(*g);
        }
        CaseGeometry& geo = last ? fine : *coarse;

        VectorField prolonged = VectorField::zeros(lc->grid);
        if (l > 0)
        {
            prolonged = prolongMac(gPrev, hPrev, carry, lc->grid, lc->h);
            ic = &prolonged;
        }

        std::printf("# seq level %d/%d: n=%d h=%.6g (%.1f cells/D)%s\n", l + 1, L,
                    lc->n, lc->h, lc->placement.D / lc->h,
                    last ? "  [final]" : "");
        std::fflush(stdout);

        VectorField fin = VectorField::zeros(lc->grid);
        const int rc = runFlowIncompressible(*lc, geo, ic, last ? nullptr : &fin,
                                             !last);
        if (rc != 0) return rc;
        if (!last)
        {
            carry = std::move(fin);
            gPrev = lc->grid;
            hPrev = lc->h;
        }
    }
    return 0;
}
#endif // DNS_WITH_GPU


} // namespace dns::drive
