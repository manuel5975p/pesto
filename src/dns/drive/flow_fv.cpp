// dns flow on a FaceMesh: simpleFoam cases, compressible / incompressible FV (O-grid, box, polyMesh), --solver simple.
#include "dns/drive/internal.hpp"

#include "dns/casegeom.hpp"
#include "dns/config.hpp"
#include "dns/flow_ic.hpp"
#include "dns/facemesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/output.hpp"
#include "dns/fvincomp.hpp"
#include "dns/fvnozzle.hpp"
#include "dns/fvshock.hpp"
#include "dns/fvsimple.hpp"
#include "dns/fvstructured.hpp"
#include "dns/remap.hpp"
#include "dns/scene.hpp"
#include "dns/taylor_green.hpp"
#include "dns/transport.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/runner_fv.hpp"
#include "dns/gpu/runner_fvincomp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/runner_fvcomp.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <fstream>
#include <iterator>
#include <numbers>
#include <string>
#include <vector>

namespace dns::drive {

using std::numbers::pi;

int runFvFlow(const Case& c)
{
    const auto t0 = std::chrono::steady_clock::now();
    auto L = loadFvCase(c.fvCase);
    if (!L) return die(exitInput, L.error());
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
#if DNS_WITH_GPU
    if (!c.quiet)
        std::printf("# FV case read in %.2f s\n%s", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                    describeFoamCase(L->c).c_str());
    const gpu::FvRunSpec sp{c.outPrefix,
                            c.fvMaxIters,
                            c.quiet,
                            !c.emit.contains(Emit::None),
                            fvNumerics(c.fvPrecision),
                            c.fvCoupled,
                            gpu::FvCoupledOpts{c.fvCoupledEta, c.fvCoupledAlpha, c.fvCoupledCheck},
                            c.fvs.refArea};
    auto r = gpu::runFvSimple(sp, *L);
    if (!r) return die(r.error().code, r.error().message);
    std::printf("done: %ld iterations%s, %.2f ms/iteration\n", r->steps, r->converged ? " (converged)" : "",
                r->msPerStep);
    return 0;
#else
    (void)t0;
    return exitNoGpu;
#endif
}
#if DNS_WITH_GPU
namespace {


// The compressible FV family's common tail (docs/fvcomp.md): the frame plane,
// the sink, the run. Phase 3 creates every output before the device.
// wallDist: per cell, RANS only (fvRansWallDistance).
int runFvCompOn(const Case& c, const FaceMesh& fm, const FvBcSpec& spec, const FvFreestream& fs, const FvState& ic,
                double D, double refArea, std::vector<double> wallDist = {})
{
    auto raster = FvRaster::make(fm, {.longSide = c.fvFrameN, .box = c.fvFrameBox});
    if (!raster) return die(exitInput, raster.error());
    FrameGeometry fg = FrameGeometry::centred(raster->grid(), D);
    fg.mask = raster->mask();
    RunMeta meta = c.meta;
    meta.setGrid(raster->grid());
    auto out = RunOutput::make(outputConfig(c, std::format("{{\"app\":\"dns\",\"re\":{},\"mach\":{}", c.re, c.mach)),
                               meta, raster->grid(), std::move(fg));
    if (!out) return die(exitOutput, out.error());
    const Transport gas = Transport::air(c.inviscid ? 0.0 : fs.rho * fs.U * D / c.re, fs.T, c.tinfK, c.prandtl, fs.gamma,
                                         kRgasBox, c.sutherland);
    gpu::FvCompRun rr{.label = fvCompLabel(c, fm.dim == 2),
                      .solver = {.tr = gas,
                                 .bc = fvBcsOf(fm, spec, fs, 1.0),
                                 .precision = c.fvPrecision,
                                 .csLes = c.cs,
                                 .sgs = c.sgsModel,
                                 .compensated = fvCompStateCompensated(c),
                                 .rans = {.model = c.rans,
                                          .wallDist = std::move(wallDist),
                                          .phiInf = {c.ransPhiInf[0], c.ransPhiInf[1]}},
                                 .tallied = fvTallyOf(fm, spec),
                                 .pRef = fs.pRest > 0.0 ? fs.pRest : fs.p,
                                 .shock = {.mode = c.shock, .eps4 = c.shockEps4}},
                      .spec = spec,
                      .fs = fs,
                      .D = D,
                      .refArea = refArea,
                      .cfl = c.cfl,
                      .tEnd = c.ramp + c.tEnd,
                      .ramp = c.ramp,
                      .dtRecompute = c.dtRecompute,
                      .curlMax = c.vortMax * fs.U / D,
                      .vortAsinh = c.vortAsinh,
                      .speedDelta = c.speedDelta,
                      .quiet = c.quiet,
                      .steadyTol = c.compSteady ? c.steadyTol : 0.0,
                      .steadySteps = c.steadySteps};
    if (spec.nozzle) {
        const auto total = std::ranges::find(spec.patches, FvPatchKind::Total, &FvPatchSpec::kind);
        rr.nozzle = FvNozzleRun{.astar = spec.nozzle->astar,
                                .axis = spec.nozzle->axis,
                                .pRef = rr.solver.pRef,
                                .p0 = total == spec.patches.end() ? 0.0 : total->p0 * fs.p,
                                .t0 = total == spec.patches.end() ? 0.0 : total->t0 * fs.T};
        if (spec.thrust)
            rr.thrust = FvThrustCase{.axis = rr.nozzle->axis,
                                     .pa = spec.thrust->pa * fs.p,
                                     .p0 = rr.nozzle->p0,
                                     .t0 = rr.nozzle->t0,
                                     .astar = rr.nozzle->astar,
                                     .gamma = fs.gamma,
                                     .Rgas = kRgasBox};
    }
    rr.shockDtAuto = c.shockDtAuto;
    rr.foamWrite = c.foamWrite;
    rr.foamPrefix = c.outPrefix;
    auto res = gpu::runFvComp(fm, c.rans == RansModel::None ? ic : fvWithScalars(ic, c.rans, rr.solver.rans.phiInf), rr,
                              *raster, *out);
    if (!res) return die(res.error().code, res.error().message);
    if (auto e = out->finish(); !e) return die(exitOutput, e.error());
    std::printf("done: %ld steps, cd=%.4f cl=%.4f, %.3f ms/step\n", res->steps, res->cd, res->cl, res->msPerStep);
    return 0;
}

// --solver fv on an O-grid: the fluid rows through fromCurvMesh, rows 1 and
// ny-1's node lines the Wall and Far patches. Same mesh, same IC as runFlowOGrid.
int runFvOGrid(const Case& c, CaseGeometry& geo)
{
    const auto& g = c.grid;
    const OGridSpec spec{.cx = c.placement.cx, .cy = c.placement.cy, .cz = c.placement.cz, .rFar = c.ogridRFar,
                         .beta = c.ogridBeta, .span = c.ogridSpan};
    auto og = buildOGrid(g, geo.sdf, spec);
    if (!og) return die(exitInput, "o-grid: " + og.error());
    const StructuredSpec ss = fvCompOgridSpec(g);
    auto fm = FaceMesh::fromCurvMesh(og->mesh, ss, CellOrder::Morton);
    if (!fm) return die(exitInput, "FaceMesh: " + fm.error());
    FvBcSpec bc{.patches = {{.name = "wall", .kind = FvPatchKind::Wall, .isothermal = c.bodyIsothermal,
                             .tWall = c.twallRatio, .tallied = true},
                            {.name = "far", .kind = FvPatchKind::Far}}};
    const FvFreestream fs = fvFreestreamOf(c.mach, kGammaAir, kRgasBox, {1, 0, 0});
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
    // runFlowOGrid's start: at rest under a ramp, plus the trip.
    const CompState ic = ogridCompressibleIC(c, geo, og->mesh, kRhoInf, fs.p, nullptr, {});
    std::vector<double> wd;
    if (c.rans != RansModel::None) {
        auto pm = FaceMesh::polyOf(og->mesh, ss);
        if (!pm) return die(exitInput, "FaceMesh: " + pm.error());
        auto d = fvRansWallDistance(std::move(*pm), *fm, bc);
        if (!d) return die(exitInput, d.error());
        wd = std::move(*d);
    }
    return runFvCompOn(c, *fm, bc, fs, fvStateOf(*fm, ic), geo.D, geo.D * double(g.nz) * g.hz, std::move(wd));
}

// --solver fv with --body none: the uniform, --stretch or --focal box.
int runFvBox(const Case& c, CaseGeometry& geo)
{
    const auto& g = c.grid;
    const AxisRemap3 fRemap = focalRemap(g, c.focalActive, c.focalLo, c.focalHi, c.focalW, c.focalRatio);
    const CurvMesh mesh = c.focalActive         ? CurvMesh::fromRemap(g, fRemap)
                          : c.stretchBeta > 0.0 ? CurvMesh::stretchedY(g, c.stretchBeta)
                                                : CurvMesh::uniform(g);
    const StructuredSpec ss = fvCompBoxSpec(g, c.far == FarField::Periodic, c.noslipY);
    auto fm = FaceMesh::fromCurvMesh(mesh, ss, CellOrder::Morton);
    if (!fm) return die(exitInput, "FaceMesh: " + fm.error());
    const FvBcSpec bc = fvCompBoxBc(*fm);
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
    const FvFreestream fs = fvFreestreamOf(c.mach, kGammaAir, kRgasBox, {1, 0, 0});
    const TanhStretchY yRemap{double(g.ny) * g.hy, c.stretchBeta};
    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    compressibleBodyIC(c, geo, fRemap, yRemap, kRhoInf, fs.p, kGammaAir, host, mask);
    std::vector<double> wd;
    if (c.rans != RansModel::None) {
        auto pm = FaceMesh::polyOf(mesh, ss);
        if (!pm) return die(exitInput, "FaceMesh: " + pm.error());
        auto d = fvRansWallDistance(std::move(*pm), *fm, bc);
        if (!d) return die(exitInput, d.error());
        wd = std::move(*d);
    }
    return runFvCompOn(c, *fm, bc, fs, fvStateOf(*fm, host), geo.D, geo.D * double(g.nz) * g.hz, std::move(wd));
}
} // namespace

#endif // DNS_WITH_GPU

std::filesystem::path fvPolyDir(const std::string& dir)
{
    const std::filesystem::path p = std::filesystem::path(dir) / "constant" / "polyMesh";
    return std::filesystem::exists(p / "boundary") ? p : std::filesystem::path(dir);
}

std::expected<bool, std::pair<int, std::string>> fvPolyCheck(const Case& c)
{
    using Err = std::unexpected<std::pair<int, std::string>>;
    auto man = readPolyManifest(fvPolyDir(c.fvMeshDir));
    if (!man) return Err({exitInput, man.error()});
    std::vector<FvMeshPatch> mp;
    for (const FoamPatch& p : man->patches) mp.push_back({p.name, foamPatchTypeName(p.type)});
    const bool wedgeOk = c.sub == Subcommand::Flow && c.shock != ShockMode::None;
    if (auto r = checkFvBc(c.fvBc, mp, wedgeOk); !r) return Err({exitInput, c.fvMeshDir + ": " + r.error()});
    const bool wedge = std::ranges::any_of(mp, [](const FvMeshPatch& m) { return m.type == "wedge"; });
    if (wedgeOk && wedge && (c.lesLabel || c.rans != RansModel::None))
        return Err({2, "a `wedge` mesh takes no --les, --rans (an axisymmetric mean flow)"});
    if (wedgeOk && c.lesLabel) return Err({2, std::string(fvPair(FvModule::Shock, FvModule::LesSmag).message)});
    const bool twoD = fvMeshIs2D(mp);
    if (twoD && c.lesLabel && c.sgsModel == SgsModel::Sigma)
        return Err({2, "--sgs: sigma: D_sigma is identically ZERO for any 2D flow; this mesh has `empty` patches"});
    return twoD;
}

int runFvCompPoly(const Case& c)
{
    auto twoD = fvPolyCheck(c);
    if (!twoD) return die(twoD.error().first, twoD.error().second);
    auto pm = readPolyMesh(fvPolyDir(c.fvMeshDir));
    if (!pm) return die(exitInput, pm.error());
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton, 0, {.wedge = c.shock != ShockMode::None});
    if (!fm) return die(exitInput, fm.error());
    // --shock reads mesh lines: a cell with a face and no opposite is refused (masterplan_combnozzle R8).
    if (c.shock != ShockMode::None)
        if (auto ok = fvShockMeshCheck(*fm); !ok) return die(exitInput, c.fvMeshDir + ": " + ok.error());
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
#if DNS_WITH_GPU
    FvFreestream fs = fvFreestreamOf(c.mach, c.fvGamma, kRgasBox, c.fvBc.dir);
    // A nozzle starts at rest at [nozzle] p, t; the ramp raises the Total row's p0.
    if (c.fvBc.nozzle) fs.pRest = c.fvBc.nozzle->p * fs.p;
    const double u0 = c.ramp > 0 ? 0.0 : fs.U;
    const FvState ic = c.fvBc.nozzle ? fvRestState(*fm, fs.pRest, c.fvBc.nozzle->t * fs.T, fs.gamma, kRgasBox)
                                     : fvUniformState(*fm, fs, u0);
    const double area = c.fvBc.area > 0 ? c.fvBc.area : c.fvBc.length * fvMeshDepth(*fm);
    std::vector<double> wd;
    if (c.rans != RansModel::None) {
        auto d = fvRansWallDistance(std::move(*pm), *fm, c.fvBc);
        if (!d) return die(exitInput, c.fvMeshDir + ": " + d.error());
        wd = std::move(*d);
    }
    return runFvCompOn(c, *fm, c.fvBc, fs, ic, c.fvBc.length, area, std::move(wd));
#else
    return exitNoGpu;
#endif
}
#if DNS_WITH_GPU
namespace {


// The time-accurate FV solver (docs/fvincomp.md): a FaceMesh, its --bc table,
// the initial field; gpu/runner_fvincomp owns the loop.
gpu::FvIncompRun fvIncompRun(const Case& c, std::string mesh)
{
    return gpu::FvIncompRun{.outPrefix = c.outPrefix,
                            .meshName = std::move(mesh),
                            .ctl = FvIncompControls{.nu = c.nu,
                                                    .tol = c.fvTol,
                                                    .les = c.cs,
                                                    .maxIter = c.fvMaxPcg,
                                                    .sgs = c.sgsModel,
                                                    .precision = c.fvPrecision},
                            .cfl = c.cfl,
                            .tEnd = c.tEnd,
                            .quiet = c.quiet,
                            .stats = !c.emit.contains(Emit::None)};
}
} // namespace

#endif

std::expected<FvIncompBcs, std::string> readFvIncompBcs(const Case& c)
{
    std::ifstream in(c.fvBcFile);
    if (!in) return std::unexpected(c.fvBcFile + ": cannot read");
    const std::string text{std::istreambuf_iterator<char>(in), {}};
    auto b = parseFvIncompBcs(text);
    if (!b) return std::unexpected(c.fvBcFile + ": " + b.error());
    return b;
}
namespace {


// A wall temperature belongs to dns conv.
std::expected<void, std::string> noWallT(const Case& c, const FvIncompBcs& b)
{
    for (const FvIncompPatch& p : b.patches)
        if (p.hasT) return std::unexpected(std::format("{}: --bc: patch.{}.t belongs to dns conv", c.fvBcFile, p.name));
    return {};
}
} // namespace


int runFvIncompManifest(const Case& c)
{
    auto m = readPolyManifest(fvPolyDir(c.fvMeshDir));
    if (!m) return die(exitInput, m.error());
    auto b = readFvIncompBcs(c);
    if (!b) return die(exitInput, b.error());
    if (c.sub != Subcommand::Conv)
        if (auto e = noWallT(c, *b); !e) return die(exitInput, e.error());
    auto s = describeFvIncompBcs(*b, *m);
    if (!s) return die(exitInput, c.fvBcFile + ": " + s.error());
    std::printf("  counts     %lld cells, %lld faces\n%s", (long long)m->nCells, (long long)m->nFaces, s->c_str());
    return 0;
}

int runFvIncompPoly(Case c)
{
    auto pm = readPolyMesh(fvPolyDir(c.fvMeshDir));
    if (!pm) return die(exitInput, pm.error());
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton);
    if (!fm) return die(exitInput, c.fvMeshDir + ": " + fm.error());
    auto b = readFvIncompBcs(c);
    if (!b) return die(exitInput, b.error());
    if (auto e = noWallT(c, *b); !e) return die(exitInput, e.error());
    if (auto e = bindFvIncompBcs(*b, *fm); !e) return die(exitInput, c.fvBcFile + ": " + e.error());
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
#if DNS_WITH_GPU
    c.nu = b->uRef * b->length / c.re;
    c.tEnd = c.ft * b->length / b->uRef;
    auto r = fvIncompRun(c, "polyMesh " + c.fvMeshDir);
    r.pm = &*pm;
    std::vector<double> U0(3 * std::size_t(fm->nC));
    for (std::size_t q = 0; q < 3; ++q)
        std::fill_n(U0.begin() + std::ptrdiff_t(q * std::size_t(fm->nC)), fm->nC, b->u0[q]);
    auto st = gpu::runFvIncomp(r, *fm, *b, U0);
    if (!st) return die(st.error().code, st.error().message);
    std::printf("done: %ld steps, t = %g, %.2f ms/step, cd %.6f cl %.6f cdf %.6f\n", st->steps, st->tStop,
                st->msPerStep, st->cd, st->cl, st->cdf);
    return 0;
#else
    return exitNoGpu;
#endif
}

int runFvIncompOGrid(const Case& c, const Scene* scene)
{
    auto geo = buildCaseGeometry(c, scene);
    if (!geo) return die(exitInput, geo.error());
    const OGridSpec spec{.cx = c.placement.cx,
                         .cy = c.placement.cy,
                         .cz = c.placement.cz,
                         .rFar = c.ogridRFar,
                         .beta = c.ogridBeta,
                         .span = c.ogridSpan};
    auto og = buildOGrid(c.grid, geo->sdf, spec);
    if (!og) return die(exitInput, "o-grid: " + og.error());
    auto fm = fvOgridMesh(*og);
    if (!fm) return die(exitInput, "o-grid: " + fm.error());
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
#if DNS_WITH_GPU
    FvIncompBcs b{.patches = {{.name = "wall", .kind = FvIncompBc::Wall, .force = true},
                              {.name = "far", .kind = FvIncompBc::Far, .u = {c.uInf, 0, 0}}},
                  .uRef = c.uInf,
                  .length = geo->D,
                  .area = geo->D * c.ogridSpan,
                  .u0 = {c.uInf, 0, 0}};
    if (auto e = bindFvIncompBcs(b, *fm); !e) return die(exitInput, e.error());
    const std::size_t nC = std::size_t(fm->nC);
    std::vector<double> U0(3 * nC, 0.0);
    std::fill_n(U0.begin(), nC, c.uInf);
    if (c.ogridTrip > 0) addOgridTripFv(c, geo->D, fm->C, U0); // the solenoidal vortex of runFlowOGrid
    auto run = fvIncompRun(c, "O-grid, body-fitted");
    run.ctl.nu = c.uInf * geo->D / c.re;
    auto st = gpu::runFvIncomp(run, *fm, b, U0);
    if (!st) return die(st.error().code, st.error().message);
    std::printf("done: %ld steps, t = %g, %.2f ms/step, cd %.6f cl %.6f cdf %.6f\n", st->steps, st->tStop,
                st->msPerStep, st->cd, st->cl, st->cdf);
    return 0;
#else
    return exitNoGpu;
#endif
}

int runFvIncompTgv(const Case& c)
{
    const Grid& g = c.grid;
    auto fm = fvPeriodicBox(g, FvPerturb(c.fvPerturb), c.fvPerturbAmp);
    if (!fm) return die(exitInput, "tgv: " + fm.error());
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns tgv --solver fv is GPU-only)");
#if DNS_WITH_GPU
    const bool twoD = g.nz == 1;
    const std::vector<double> U0 = taylorGreenAt(fm->C, twoD);
    auto r = fvIncompRun(c, c.fvPerturb == 0 ? "periodic box" : "perturbed periodic box");
    r.seriesDt = twoD ? 0.0 : 0.1;
    r.seriesVolume = twoD ? 1.0 : g.cellVolume() * double(g.size());
    r.quiet = c.quiet || !twoD; // the 3D stdout is the series
    gpu::FvIncompState fin;
    auto st = gpu::runFvIncomp(r, *fm, FvIncompBcs{}, U0, &fin);
    if (!st) return die(st.error().code, st.error().message);
    if (twoD && c.emit.contains(Emit::Error)) {
        const TaylorGreenFvError e = taylorGreenErrorAt(fm->C, fm->V, fin.U, c.nu, st->tStop);
        std::printf("n=%d nu=%g T=%g steps=%ld\nmax error vs analytic: %.3e\nL2 error vs analytic: %.3e\n", g.nx, c.nu,
                    st->tStop, st->steps, e.max, e.l2);
    }
    std::fprintf(stderr, "done: %ld steps, %.2f ms/step (FV)\n", st->steps, st->msPerStep);
    return 0;
#else
    return exitNoGpu;
#endif
}

int runFvStructured(const Case& c)
{
    Sdf sdf;
    if (c.fvs.mesh == FvStructMesh::OGrid) {
        auto geo = buildCaseGeometry(c, nullptr);
        if (!geo) return die(exitInput, geo.error());
        sdf = std::move(geo->sdf);
    }
    auto pm = fvStructuredMesh(c.fvs, sdf);
    if (!pm) return die(exitInput, pm.error());
    if (auto w = writeFoamCase(c.fvCase, *pm, fvStructuredTexts(c.fvs)); !w) return die(exitOutput, w.error());
    if (c.fvExport.empty()) return runFvFlow(c);
    std::printf("case written to %s: %s, %ld cells, %ld faces\n", c.fvExport.c_str(),
                fvStructuredLabel(c.fvs).c_str(), long(pm->man.nCells), long(pm->man.nFaces));
    return 0;
}

int runFlow(const Case& c, [[maybe_unused]] const RawConfig& raw, const Scene* scene)
{
    // Input before device: a bad mesh is exit 3 on every machine, GPU or not.
    [[maybe_unused]] const auto t0 = std::chrono::steady_clock::now();
    auto geo = buildCaseGeometry(c, scene);
    if (!geo) return die(exitInput, geo.error());
#if DNS_WITH_GPU
    if (c.fvMode == FvCompMode::Structured) return c.ogrid ? runFvOGrid(c, *geo) : runFvBox(c, *geo);
#endif
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns flow is GPU-only)");
#if DNS_WITH_GPU
    if (c.seqLevels > 1) return runFlowSequenced(c, raw, scene, *geo);
    std::printf("# geometry built in %.2f s (D = %.4f, %.1f cells)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                    .count(),
                geo->D, geo->D / c.h);
    if (c.ogrid) return runFlowOGrid(c, *geo);
    return c.physics == Physics::Compressible ? runFlowCompressible(c, *geo)
                                              : runFlowIncompressible(c, *geo);
#else
    return exitNoGpu;
#endif
}


} // namespace dns::drive
