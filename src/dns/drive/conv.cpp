// dns conv: structured and FaceMesh (--solver fv, --polymesh) Rayleigh-Benard.
#include "dns/drive/internal.hpp"

#include "dns/config.hpp"
#include "dns/flow_ic.hpp"
#include "dns/facemesh.hpp"
#include "dns/output.hpp"
#include "dns/fvcomp_case.hpp"
#include "dns/fvconv.hpp"
#include "dns/fvincomp.hpp"
#include "dns/transport.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/runner_conv.hpp"
#include "dns/gpu/runner_fvconv.hpp"
#include "dns/gpu/runner_rb.hpp"
#endif

#include <cstdio>
#include <format>
#include <numbers>
#include <string>
#include <vector>

namespace dns::drive {

using std::numbers::pi;
namespace {

// --- conv ------------------------------------------------------------------

// The artefact sink of a convection run. The two branches are two MODELS and
// share no physics code, but they write the same artefacts with the same
// metadata and the same centre planes -- so the sink is built once and the
// only thing either branch supplies is its own ghost mask.
[[maybe_unused]] std::expected<RunOutput, std::string> convOutput(const Case& c,
                                                                  std::vector<float> mask)
{
    FrameGeometry fg = FrameGeometry::centred(c.grid, c.convH);
    fg.mask = std::move(mask);
    return RunOutput::make(
        outputConfig(c,
                     std::format("{{\"app\":\"dns\",\"rayleigh\":{},"
                                 "\"prandtl\":{}",
                                 c.rayleigh, c.prandtl)),
        c.meta, c.grid, std::move(fg));
}
} // namespace

#if DNS_WITH_GPU
namespace {


// dns conv --solver fv | --polymesh (docs/fvcomp.md, docs/fvincomp.md). The IC is
// convIC's and rbIC's analytic fields at the cell centres; the box is a FaceMesh
// with its floor at y = 0, so both routes are one route.
gpu::FvConvRun fvConvRun(const Case& c, const FaceMesh& fm, FvConvPlates plates, double kapFace)
{
    return {.label = fvConvLabel(c, fm.dim == 2),
            .H = c.convH,
            .Tbot = c.tBot,
            .Ttop = c.tTop,
            .kapDiff = c.kappaDiff,
            .kapFace = kapFace,
            .uFf = c.uFreeFall,
            .cfl = c.cfl,
            .tEnd = c.tEnd,
            .dtRecompute = c.dtRecompute,
            .profileDt = c.profileDt,
            .bins = c.fvConvBins,
            .plates = std::move(plates),
            .quiet = c.quiet};
}

int runConvFvOn(const Case& c, const FaceMesh& fm)
{
    auto raster = FvRaster::make(fm, {.longSide = c.fvFrameN, .box = c.fvFrameBox});
    if (!raster) return die(exitInput, raster.error());
    const bool poly = !c.fvMeshDir.empty();
    auto output = [&]() {
        FrameGeometry fg = FrameGeometry::centred(raster->grid(), c.convH);
        fg.mask = raster->mask();
        RunMeta meta = c.meta;
        meta.setGrid(raster->grid());
        return RunOutput::make(outputConfig(c, std::format("{{\"app\":\"dns\",\"rayleigh\":{},\"prandtl\":{}",
                                                           c.rayleigh, c.prandtl)),
                               meta, raster->grid(), std::move(fg));
    };
    if (c.physics == Physics::Incompressible) {
        FvIncompBcs b;
        if (poly) {
            auto rb = readFvIncompBcs(c);
            if (!rb) return die(exitInput, rb.error());
            b = std::move(*rb);
        } else {
            b = fvConvIncompBcsOf(fvConvBoxWalls(fm, c.tBot, c.tTop));
        }
        auto walls = fvConvWalls(b);
        if (!walls) return die(exitInput, c.fvBcFile + ": " + walls.error());
        if (auto e = bindFvIncompBcs(b, fm); !e) return die(exitInput, c.fvBcFile + ": " + e.error());
        auto plates = fvConvPlates(fm, *walls, c.tBot, c.tTop, c.convH);
        if (!plates) return die(exitInput, plates.error());
        auto out = output();
        if (!out) return die(exitOutput, out.error());
        std::vector<double> U0, T0;
        rbFvFields(c, fm, U0, T0);
        const FvIncompControls ctl{.nu = c.nu,
                                   .tol = c.fvTol,
                                   .les = c.cs,
                                   .maxIter = c.fvMaxPcg,
                                   .sgs = c.sgsModel,
                                   .precision = c.fvPrecision,
                                   .kappa = c.kappaDiff,
                                   .Tref = 0.0,
                                   .betaG = {0.0, c.gravity, 0.0}};
        gpu::FvConvRun rr = fvConvRun(c, fm, std::move(*plates), c.kappaDiff);
        rr.statsEvery = 50;
        auto res = gpu::runFvConvRb(fm, b, ctl, U0, T0, rr, *raster, *out);
        if (!res) return die(res.error().code, res.error().message);
        return reportDone(*res, *out);
    }
    const FvBcSpec spec = poly ? c.fvBc : fvConvCompBcOf(fvConvBoxWalls(fm, c.tBot, c.tTop));
    auto walls = fvConvWalls(spec);
    if (!walls) return die(exitInput, c.fvBcFile + ": " + walls.error());
    auto plates = fvConvPlates(fm, *walls, c.tBot, c.tTop, c.convH);
    if (!plates) return die(exitInput, plates.error());
    auto out = output();
    if (!out) return die(exitOutput, out.error());
    // The layer's units: Tref = rho_ref = R = 1, p_ref = 1, walls absolute.
    const FvFreestream fs{.rho = 1.0, .U = 1.0, .p = c.pRef, .T = 1.0, .gamma = kGammaAir};
    const FvState ic = convFvState(c, fm);
    const Transport gas = Transport::air(c.nu, 1.0, c.tinfK, c.prandtl, kGammaAir, kRgasBox, c.sutherland);
    const gpu::FvCompConfig cfg{.tr = gas,
                                .bc = fvBcsOf(fm, spec, fs, 1.0),
                                .precision = c.fvPrecision,
                                .compensated = c.compensated,
                                .pRef = fs.p,
                                .gravity = {0.0, -c.gravity, 0.0}};
    auto res = gpu::runFvConvComp(fm, ic, cfg, fvConvRun(c, fm, std::move(*plates), c.nu * gas.cp() / c.prandtl), *raster,
                                  *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
}
} // namespace

#endif // DNS_WITH_GPU

int runConvFvManifest(const Case& c)
{
    std::printf("  label      %s\n", fvConvLabel(c, false).c_str());
    if (c.physics == Physics::Incompressible) {
        auto b = readFvIncompBcs(c);
        if (!b) return die(exitInput, b.error());
        if (auto w = fvConvWalls(*b); !w) return die(exitInput, c.fvBcFile + ": " + w.error());
        return runFvIncompManifest(c);
    }
    auto two = fvPolyCheck(c);
    if (!two) return die(two.error().first, two.error().second);
    if (auto w = fvConvWalls(c.fvBc); !w) return die(exitInput, c.fvBcFile + ": " + w.error());
    return 0;
}
namespace {


int runConvFv(const Case& c)
{
    if (c.fvMeshDir.empty()) {
        auto fm = fvConvBox(c.grid, c.closedSides, FvPerturb(c.fvPerturb), c.fvPerturbAmp);
        if (!fm) return die(exitInput, "FaceMesh: " + fm.error());
        if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns conv is GPU-only)");
#if DNS_WITH_GPU
        return runConvFvOn(c, *fm);
#else
        return exitNoGpu;
#endif
    }
    if (c.physics == Physics::Compressible) {
        auto two = fvPolyCheck(c);
        if (!two) return die(two.error().first, two.error().second);
    }
    auto pm = readPolyMesh(fvPolyDir(c.fvMeshDir));
    if (!pm) return die(exitInput, pm.error());
    auto fm = FaceMesh::fromPolyMesh(*pm, CellOrder::Morton);
    if (!fm) return die(exitInput, c.fvMeshDir + ": " + fm.error());
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns conv is GPU-only)");
#if DNS_WITH_GPU
    return runConvFvOn(c, *fm);
#else
    return exitNoGpu;
#endif
}
} // namespace


int runConv(const Case& c)
{
    if (c.fvConv) return runConvFv(c);
    if (!haveGpu(c))
        return die(exitNoGpu, "no GPU device (dns conv is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    const double gamma = kGammaAir;

    // Boussinesq branch: a different model, not a cheaper solver for the same
    // one (gpu/boussinesq.hpp). It shares the case derivation and the artefact
    // sink and nothing else.
    if (c.physics == Physics::Incompressible)
    {
        auto U = VectorField::zeros(g);
        auto T = zeros(g);
        std::vector<float> mask(g.size(), 1.0f);
        rbIC(c, U, T, mask);

        auto out = convOutput(c, std::move(mask));
        if (!out) return die(exitOutput, out.error());

        gpu::RbRun rr;
        rr.rayleigh = c.rayleigh;
        rr.prandtl = c.prandtl;
        rr.nu = c.nu;
        rr.kappa = c.kappaDiff;
        rr.gAlpha = c.gravity;
        rr.H = c.convH;
        rr.Tbot = c.tBot;
        rr.Ttop = c.tTop;
        rr.Tref = 0.0;
        rr.patch = c.hotPatch;
        rr.closedSides = c.closedSides;
        rr.cfl = c.cfl;
        rr.tEnd = c.tEnd;
        rr.profileDt = c.profileDt;

        auto res = gpu::runRb(g, U, T, rr, *out);
        if (!res) return die(res.error().code, res.error().message);
        return reportDone(*res, *out);
    }

    auto host = CompState::zeros(g);
    std::vector<float> mask(g.size(), 1.0f);
    convIC(c, gamma, host, mask);

    auto out = convOutput(c, std::move(mask));
    if (!out) return die(exitOutput, out.error());

    gpu::ConvRun rr;
    rr.gamma = gamma;
    rr.mu = c.nu;            // rho_ref = 1
    rr.Pr = c.prandtl;
    rr.tinfK = c.tinfK;
    rr.sutherland = c.sutherland;
    rr.gravity = c.gravity;
    rr.H = c.convH;
    rr.Tbot = c.tBot;
    rr.Ttop = c.tTop;
    rr.patch = c.hotPatch;
    rr.closedSides = c.closedSides;
    rr.rayleigh = c.rayleigh;
    rr.uFf = c.uFreeFall;
    rr.cfl = c.cfl;
    rr.tEnd = c.tEnd;
    rr.dtRecompute = c.dtRecompute;
    rr.profileDt = c.profileDt;

    auto res = gpu::runConv(g, host, rr, *out);
    if (!res) return die(res.error().code, res.error().message);
    return reportDone(*res, *out);
#else
    return exitNoGpu;
#endif
}



} // namespace dns::drive
