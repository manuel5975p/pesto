// Compressible ghost wall: quiescent invariance (no spurious currents or
// force), impulsive freestream stability + positive drag on the circle.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;



} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const int n = 128;
    const double L = 2 * pi, R = L / 16, cx = L / 2, cy = L / 2 + 0.013;
    const double gamma = 1.4, rhoInf = 1.0, pInf = 1.0 / (gamma * 0.04); // M 0.2
    const auto g = Grid::make(n, n, 1, L, L, 1.0); // hz = 1: unit depth
    auto sdf = [&](double x, double y, double) {
        return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)) - R;
    };
    const double mu = rhoInf * 2 * R / 100.0; // Re 100 on uInf = 1

    auto dev = gpu::DeviceComp::make(g, gamma, mu);
    auto ghost = gpu::DeviceCompGhost::make(g, sdf, gamma, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ghost || !ds || !red) return EXIT_FAILURE;
    dev->setStageHook([&](gpu::DeviceCompState& st) { ghost->apply(st); });

    // Quiescent gas + wall must stay exactly quiescent: mirror of zero is
    // zero, copied p/rho are uniform.
    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gamma - 1.0);
    }
    (void)ds->upload(host);
    for (int it = 0; it < 50; ++it) dev->step(*ds, gpu::Real(1e-3));
    ghost->apply(*ds);
    (void)ds->download(host);
    double vmax = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
        vmax = std::max({vmax, std::abs(host.mx[c]), std::abs(host.my[c])});
    const auto F0 = ghost->force(*dev, *ds);
    std::printf("quiescent: max|m| %.2e force (%.2e, %.2e)\n", vmax, F0.fx, F0.fy);
    check(vmax < 1e-5, "quiescent gas stays at rest");
    check(std::abs(F0.fx) < 1e-4 && std::abs(F0.fy) < 1e-4, "no spurious force");

    // Tapered freestream: stable evolution, positive drag, bounded speed.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double d = sdf((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0.0);
            const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
            const double u = f * f * (3 - 2 * f);
            host.rho[c] = rhoInf;
            host.mx[c] = rhoInf * u;
            host.my[c] = 0.0;
            host.E[c] = pInf / (gamma - 1.0) + 0.5 * rhoInf * u * u;
        }
    (void)ds->upload(host);
    double t = 0.0;
    while (t < 2.0) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
        t += dt;
    }
    ghost->apply(*ds);
    const auto F = ghost->force(*dev, *ds);
    (void)ds->download(host);
    double smax = 0.0;
    bool finite = true;
    for (std::size_t c = 0; c < g.size(); ++c) {
        const double u = host.mx[c] / host.rho[c], v = host.my[c] / host.rho[c];
        if (!std::isfinite(u) || !std::isfinite(v)) finite = false;
        smax = std::max(smax, std::sqrt(u * u + v * v));
    }
    const double cd = F.fx / (0.5 * rhoInf * 2 * R);
    std::printf("freestream t=2: cd %.3f max|u| %.3f\n", cd, smax);
    check(finite, "freestream evolution finite");
    check(cd > 0.5 && cd < 30.0, "drag positive and sane (early transient)");
    check(smax < 3.0, "speed bounded");

    // The tally used to atomicAdd in FP32: cancelling +- contributions made the
    // reported cd irreproducible run to run, non-determinism in the primary
    // validated quantity. It now stages per face in FP64 and reduces
    // deterministically, so repeats must agree to the last bit.
    {
        bool same = true;
        for (int r = 0; r < 4; ++r) {
            const auto Fr = ghost->force(*dev, *ds);
            if (Fr.fx != F.fx || Fr.fy != F.fy || Fr.fz != F.fz) same = false;
        }
        check(same, "force tally is bit-reproducible across repeats");
        std::printf("cd (FP64 tally) = %.17g\n", cd);
    }

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
