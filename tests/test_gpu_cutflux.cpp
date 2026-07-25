// fusedRhsStageCut (gpu/cutflux.hpp), the apertured cut-cell momentum RHS.
//
// Checks:
//   1. All-open geometry: the cut kernel's output is BIT-IDENTICAL to
//      fusedRhsStage on a random field (the flag is all-zero and the fast
//      path is the verbatim stencils.hpp code).
//   2. Sphere geometry: every cell OUTSIDE the buildCutFlag band is
//      bit-identical to the plain kernel; the band is non-empty and the two
//      kernels genuinely differ somewhere inside it (the apertured path
//      engaged).
//   3. nz == 1 cylinder: w is EXACTLY zero after the cut kernel on a
//      two-component field, and stays exactly zero over 100 RK3 + projection
//      steps (the 2D degeneracy invariant; fluxDiff makes the z-cancellation
//      FMA-contraction-proof by construction).
//   4. Inviscid energy stability, the cut-flux version of the
//      energy-stability canary: off-node sphere, nu = 0, random projected
//      field, 300 RK3 + cut projection steps with the cut-aware Courant
//      bound; kinetic energy must not grow beyond time-integration noise
//      (the apertured skew form is exactly skew-symmetric in the
//      lambda~-weighted inner product).
//
// No HIP device -> skip (exit 77).
#include "gpu_test_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/gpu/cutflux.hpp"
#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>

using namespace dns;

namespace {

using dnstest::check;



VectorField randomField(const Grid& g, unsigned seed, bool with_w)
{
    auto U = VectorField::zeros(g);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] = d(rng);
        U.v[c] = d(rng);
        if (with_w) U.w[c] = d(rng);
    }
    return U;
}

CutGeometry openCut(const Grid& g)
{
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        cut.lambda[c] = 1.0;
        cut.ax[c] = cut.ay[c] = cut.az[c] = 1.0;
    }
    return cut;
}

// Run both kernels on the same inputs; return the downloaded pair.
struct Pair
{
    VectorField plain, cutout;
};

std::optional<Pair> runBoth(const Grid& g, const CutGeometry& cut,
                            const VectorField& U, double nu)
{
    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto dFlag = gpu::DeviceCutFlag::make(buildCutFlag(g, cut));
    auto dU = gpu::DeviceField::make(g.size());
    auto dA = gpu::DeviceField::make(g.size());
    auto dB = gpu::DeviceField::make(g.size());
    if (!dCut || !dFlag || !dU || !dA || !dB) return std::nullopt;
    (void)gpu::uploadField(U, *dU);
    // a = 0.25, x = in: exercises the full axpy shape with one field.
    gpu::fusedRhsStage(gg, gpu::Real(nu), gpu::Real(0), 0, 0, 0, gpu::Real(0.25),
                       *dU, gpu::Real(0.75), *dU, *dU, gpu::Real(1e-3), *dA);
    gpu::fusedRhsStageCut(gg, gpu::Real(nu), gpu::Real(0), 0, 0, 0,
                          gpu::Real(0.25), *dU, gpu::Real(0.75), *dU, *dU,
                          *dCut, *dFlag, gpu::Real(1e-3), *dB);
    Pair p{VectorField::zeros(g), VectorField::zeros(g)};
    if (gpu::downloadField(*dA, p.plain) != hipSuccess) return std::nullopt;
    if (gpu::downloadField(*dB, p.cutout) != hipSuccess) return std::nullopt;
    return p;
}

} // namespace

int main()
{
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    // --- (1) all-open geometry: bit identity everywhere -------------------
    {
        const int n = 32;
        const auto g = Grid::make(n, n, n, 1.0, 1.0, 1.0);
        const auto cut = openCut(g);
        const auto flag = buildCutFlag(g, cut);
        bool flagZero = true;
        for (auto f : flag) flagZero = flagZero && f == 0;
        check(flagZero, "all-open geometry: flag is empty");
        const auto U = randomField(g, 7, true);
        auto p = runBoth(g, cut, U, 0.01);
        if (!p) { std::printf("alloc failed\n"); return 1; }
        bool same = true;
        for (std::size_t c = 0; c < g.size(); ++c)
            same = same && p->plain.u[c] == p->cutout.u[c]
                        && p->plain.v[c] == p->cutout.v[c]
                        && p->plain.w[c] == p->cutout.w[c];
        check(same, "all-open geometry: bit-identical to fusedRhsStage");
    }

    // --- (2) sphere: bit identity outside the flag band -------------------
    {
        const int n = 48;
        const double L = 2 * pi;
        const auto g = Grid::make(n, n, n, L, L, L);
        const Sphere s{L / 2, L / 2 + 0.31 * g.hx, L / 2 - 0.17 * g.hx, L / 6};
        auto cut = CutGeometry::make(g, s);
        (void)sanitizeCut(g, cut);
        const auto flag = buildCutFlag(g, cut);
        std::size_t nFlag = 0;
        for (auto f : flag) nFlag += f;
        check(nFlag > 0 && nFlag < g.size() / 2, "sphere: flag band is a band");
        auto U = randomField(g, 11, true);
        maskSolidFaces(g, cut, U);
        auto p = runBoth(g, cut, U, 0.01);
        if (!p) { std::printf("alloc failed\n"); return 1; }
        bool outside = true, insideDiffers = false;
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            const bool eq = p->plain.u[c] == p->cutout.u[c]
                         && p->plain.v[c] == p->cutout.v[c]
                         && p->plain.w[c] == p->cutout.w[c];
            if (!flag[c] && !eq) outside = false;
            if (flag[c] && !eq) insideDiffers = true;
        }
        check(outside, "sphere: bit-identical outside the flag band");
        check(insideDiffers, "sphere: apertured path engaged inside the band");
    }

    // --- (3) nz == 1: w stays exactly zero --------------------------------
    {
        const int n = 48;
        const double L = 2 * pi;
        const auto g = Grid::make(n, n, 1, L, L, 1.0);
        const CylinderZ cyl{0.5 * L, 0.45 * L, 0.15 * L};
        auto cut = CutGeometry::make(g, cyl);
        (void)sanitizeCut(g, cut);
        auto U = randomField(g, 13, false);
        maskSolidFaces(g, cut, U);

        const auto gg = gpu::GpuGrid::from(g);
        auto dCut = gpu::DeviceCutGeometry::make(cut);
        auto dFlag = gpu::DeviceCutFlag::make(buildCutFlag(g, cut));
        auto dU = gpu::DeviceField::make(g.size());
        auto dU0 = gpu::DeviceField::make(g.size());
        auto dS = gpu::DeviceField::make(g.size());
        auto red = gpu::ReduceScratch::make();
        auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
        auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, 1e-5);
        if (!dCut || !dFlag || !dU || !dU0 || !dS || !red || !scr || !poisson)
        { std::printf("alloc failed\n"); return 1; }
        (void)gpu::uploadField(U, *dU);
        gpu::maskSolidFaces(gg, *dCut, *dU);
        (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(gpu::Real));
        gpu::projectCut(gg, *dCut, *poisson, gpu::Real(1), *dU, *scr);

        auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out,
                         double a, double dt) {
            gpu::fusedRhsStageCut(gg, gpu::Real(0.002), gpu::Real(0), 0, 0, 0,
                                  gpu::Real(a), *dU0, gpu::Real(1.0 - a), in,
                                  in, *dCut, *dFlag, gpu::Real(dt), out);
            gpu::maskSolidFaces(gg, *dCut, out);
            gpu::projectCut(gg, *dCut, *poisson, gpu::Real(dt), out, *scr);
        };
        for (int step = 0; step < 100; ++step)
        {
            const double conv = gpu::maxCourantPerDtCut(
                gg, *dU, dCut->ax, dCut->ay, dCut->az, dCut->lambda, *red);
            const double dt = 0.5 / conv;
            std::swap(*dU0, *dU);
            stage(*dU0, *dU, 0.0, dt);
            stage(*dU, *dS, 0.75, dt);
            stage(*dS, *dU, 1.0 / 3.0, dt);
        }
        auto R = VectorField::zeros(g);
        (void)gpu::downloadField(*dU, R);
        double wMax = 0, uMax = 0;
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            wMax = std::max(wMax, std::abs(R.w[c]));
            uMax = std::max(uMax, std::abs(R.u[c]));
        }
        check(wMax == 0.0, "nz==1: w exactly zero after 100 cut-flux steps");
        check(std::isfinite(uMax) && uMax > 0, "nz==1: u finite and alive");
    }

    // --- (4) inviscid energy stability ------------------------------------
    {
        const int n = 64;
        const double L = 2 * pi;
        const auto g = Grid::make(n, n, n, L, L, L);
        const double h = g.hx;
        const Sphere s{L / 2 + 0.137 * h, L / 2 - 0.293 * h, L / 2 + 0.051 * h,
                       L / 5};
        auto cut = CutGeometry::make(g, s);
        (void)sanitizeCut(g, cut);
        auto U = randomField(g, 42, true);
        maskSolidFaces(g, cut, U);

        const auto gg = gpu::GpuGrid::from(g);
        auto dCut = gpu::DeviceCutGeometry::make(cut);
        auto dFlag = gpu::DeviceCutFlag::make(buildCutFlag(g, cut));
        auto dU = gpu::DeviceField::make(g.size());
        auto dU0 = gpu::DeviceField::make(g.size());
        auto dS = gpu::DeviceField::make(g.size());
        auto red = gpu::ReduceScratch::make();
        auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
        auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, 1e-5);
        if (!dCut || !dFlag || !dU || !dU0 || !dS || !red || !scr || !poisson)
        { std::printf("alloc failed\n"); return 1; }
        (void)gpu::uploadField(U, *dU);
        gpu::maskSolidFaces(gg, *dCut, *dU);
        (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(gpu::Real));
        gpu::projectCut(gg, *dCut, *poisson, gpu::Real(1), *dU, *scr);

        const double K0 = gpu::kineticEnergy(gg, *dU, *red);
        auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out,
                         double a, double dt) {
            gpu::fusedRhsStageCut(gg, gpu::Real(0), gpu::Real(0), 0, 0, 0,
                                  gpu::Real(a), *dU0, gpu::Real(1.0 - a), in,
                                  in, *dCut, *dFlag, gpu::Real(dt), out);
            gpu::maskSolidFaces(gg, *dCut, out);
            gpu::projectCut(gg, *dCut, *poisson, gpu::Real(dt), out, *scr);
        };
        double kMax = K0;
        const int steps = 300;
        for (int step = 0; step < steps; ++step)
        {
            const double conv = gpu::maxCourantPerDtCut(
                gg, *dU, dCut->ax, dCut->ay, dCut->az, dCut->lambda, *red);
            const double dt = 0.5 / conv;
            std::swap(*dU0, *dU);
            stage(*dU0, *dU, 0.0, dt);
            stage(*dU, *dS, 0.75, dt);
            stage(*dS, *dU, 1.0 / 3.0, dt);
            if (step % 100 == 0 || step == steps - 1)
            {
                const double K = gpu::kineticEnergy(gg, *dU, *red);
                kMax = std::max(kMax, K);
                std::printf("step %3d K/K0=%.6f\n", step, K / K0);
                if (!std::isfinite(K)) { check(false, "energy finite"); break; }
            }
        }
        check(kMax <= K0 * 1.02, "inviscid cut-flux energy never grows");
    }

    std::printf("%s\n", dnstest::gFailures == 0 ? "ALL OK" : "FAILURES");
    return dnstest::finish();
}
