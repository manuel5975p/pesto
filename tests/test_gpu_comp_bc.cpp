// Tunnel BCs: uniform freestream is a fixed point to FP32 rounding (2D and
// 3D), free-slip walls generate exactly zero wall vorticity.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_bc.hpp"
#include "gpu_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using namespace dns;
using std::numbers::pi;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

// Max deviation from freestream after nSteps in an empty tunnel.
double freestreamDrift(int nx, int ny, int nz)
{
    const double gamma = 1.4, rhoInf = 1.0, uInf = 1.0;
    const double pInf = 1.0 / (gamma * 0.04); // M 0.2
    const auto g = Grid::make(nx, ny, nz, nx * 0.01, ny * 0.01, nz > 1 ? nz * 0.01 : 1.0);
    auto dev = gpu::DeviceComp::make(g, gamma, 1e-3);
    auto bc = gpu::DeviceCompBc::make(g, gamma, {rhoInf, uInf, pInf, true, true});
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !bc || !ds || !red) std::exit(EXIT_FAILURE);
    dev->setStageHook([&](gpu::DeviceCompState& st) { bc->apply(st); });

    auto host = CompState::zeros(g);
    const double EInf = pInf / (gamma - 1.0) + 0.5 * rhoInf * uInf * uInf;
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.mx[c] = rhoInf * uInf;
        host.E[c] = EInf;
    }
    (void)ds->upload(host);
    for (int it = 0; it < 100; ++it) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
    }
    (void)ds->download(host);
    double drift = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c) {
        drift = std::max(drift, std::abs(host.rho[c] - rhoInf));
        drift = std::max(drift, std::abs(host.mx[c] - rhoInf * uInf));
        drift = std::max(drift, std::abs(host.my[c]));
        drift = std::max(drift, std::abs(host.mz[c]));
        drift = std::max(drift, std::abs(host.E[c] - EInf) / EInf);
    }
    return drift;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const double d2 = freestreamDrift(96, 48, 1);
    std::printf("freestream drift 2D: %.2e\n", d2);
    check(d2 < 5e-6, "2D uniform freestream fixed to FP32 rounding");

    const double d3 = freestreamDrift(48, 24, 24);
    std::printf("freestream drift 3D: %.2e\n", d3);
    check(d3 < 5e-6, "3D uniform freestream fixed to FP32 rounding");

    // Disturbed flow: wall rows must carry exactly zero wall vorticity
    // (tangential mirrored, normal antimirrored).
    {
        const double gamma = 1.4, pInf = 1.0 / (gamma * 0.04);
        const int nx = 96, ny = 48;
        const auto g = Grid::make(nx, ny, 1, 2 * pi, pi, 1.0);
        auto dev = gpu::DeviceComp::make(g, gamma, 1e-3);
        auto bc = gpu::DeviceCompBc::make(g, gamma, {1.0, 1.0, pInf, true, true});
        auto ds = gpu::DeviceCompState::make(g.size());
        auto red = gpu::ReduceScratch::make();
        dev->setStageHook([&](gpu::DeviceCompState& st) { bc->apply(st); });
        auto host = CompState::zeros(g);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
                const double u = 1.0 + 0.2 * std::sin(x) * std::cos(2 * y);
                const double v = 0.1 * std::cos(x) * std::sin(2 * y);
                const std::size_t c = g.idx(i, j, 0);
                host.rho[c] = 1.0;
                host.mx[c] = u;
                host.my[c] = v;
                host.E[c] = pInf / (gamma - 1.0) + 0.5 * (u * u + v * v);
            }
        (void)ds->upload(host);
        for (int it = 0; it < 50; ++it)
            dev->step(*ds, gpu::Real(dev->stableDt(*ds, *red)));
        bc->apply(*ds);
        (void)ds->download(host);
        // Corner vorticity on the wall planes (between rows 0|1 and ny-2|ny-1).
        double wmax = 0.0;
        for (int i = 1; i < nx - 1; ++i)
            for (const int jw : {1, ny - 1}) {
                const double u0 = host.mx[g.idx(i, jw - 1, 0)] / host.rho[g.idx(i, jw - 1, 0)];
                const double u1 = host.mx[g.idx(i, jw, 0)] / host.rho[g.idx(i, jw, 0)];
                const double vA = host.my[g.idx(i - 1, jw - 1, 0)] + host.my[g.idx(i - 1, jw, 0)];
                const double vB = host.my[g.idx(i + 1, jw - 1, 0)] + host.my[g.idx(i + 1, jw, 0)];
                const double w = (vB - vA) / (4 * g.hx) - (u1 - u0) / g.hy;
                wmax = std::max(wmax, std::abs(w));
            }
        std::printf("wall vorticity max: %.2e\n", wmax);
        check(wmax < 1e-6, "free-slip walls: zero wall vorticity");
    }

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
