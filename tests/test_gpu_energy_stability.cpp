// Inviscid energy-stability canary: the harshest version of the scenario that
// blew up the divergence-form scheme (2026-07-15). Off-node sphere (cut cells
// on the shoulder), nu = 0, random projected field, hundreds of RK3 + cut
// projection steps. With skew-symmetric convection and the orthogonal
// projection, kinetic energy must never grow beyond time-integration noise;
// an energy-injecting operator explodes here within tens of steps.
#include "gpu_test_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>

using namespace dns;

int main()
{
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    const int n = 64;
    const double L = 2 * pi;
    const auto g = Grid::make(n, n, n, L, L, L);
    const double h = g.hx;

    const Sphere s{L / 2 + 0.137 * h, L / 2 - 0.293 * h, L / 2 + 0.051 * h, L / 5};
    auto cut = CutGeometry::make(g, s);
    (void)sanitizeCut(g, cut);

    auto U = VectorField::zeros(g);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] = d(rng); U.v[c] = d(rng); U.w[c] = d(rng);
    }
    maskSolidFaces(g, cut, U);

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto dU = gpu::DeviceField::make(g.size());
    auto dU0 = gpu::DeviceField::make(g.size());
    auto dS = gpu::DeviceField::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    if (!dCut || !dU || !dU0 || !dS || !red || !scr)
    {
        std::fprintf(stderr, "FAIL: GPU alloc\n");
        return 1;
    }
    auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, 1e-5);
    if (!poisson)
    {
        std::fprintf(stderr, "FAIL: solver make\n");
        return 1;
    }
    (void)gpu::uploadField(U, *dU);
    gpu::maskSolidFaces(gg, *dCut, *dU);
    (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(gpu::Real));
    gpu::projectCut(gg, *dCut, *poisson, gpu::Real(1), *dU, *scr);

    const double K0 = gpu::kineticEnergy(gg, *dU, *red);
    std::printf("K0 = %.6f\n", K0);

    auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double a,
                     double dt) {
        gpu::fusedRhsStage(gg, gpu::Real(0), gpu::Real(0), 0, 0, 0, gpu::Real(a),
                           *dU0, gpu::Real(1.0 - a), in, gpu::Real(dt), out);
        gpu::maskSolidFaces(gg, *dCut, out);
        gpu::projectCut(gg, *dCut, *poisson, gpu::Real(dt), out, *scr);
    };

    const int steps = 300;
    double kMax = K0;
    for (int step = 0; step < steps; ++step)
    {
        const double conv = gpu::maxCourantPerDt(gg, *dU, *red);
        const double dt = 0.5 / conv;
        std::swap(*dU0, *dU);
        stage(*dU0, *dU, 0.0, dt);
        stage(*dU, *dS, 0.75, dt);
        stage(*dS, *dU, 1.0 / 3.0, dt);
        if (step % 50 == 0 || step == steps - 1)
        {
            const double K = gpu::kineticEnergy(gg, *dU, *red);
            kMax = std::max(kMax, K);
            std::printf("step %3d K=%.6f K/K0=%.6f\n", step, K, K / K0);
            if (!std::isfinite(K))
            {
                std::printf("FAIL: energy is not finite\n");
                return 1;
            }
        }
    }

    // Inviscid + energy-conserving convection + orthogonal projection: K can
    // only shrink, modulo RK3 time error and the FP32 projection residual.
    // The divergence form exceeded this bound by orders of magnitude here.
    if (kMax > K0 * 1.02)
    {
        std::printf("FAIL: energy grew (Kmax/K0 = %.4f)\n", kMax / K0);
        return 1;
    }
    std::printf("energy bounded: Kmax/K0 = %.6f\n", kMax / K0);
    return 0;
}
