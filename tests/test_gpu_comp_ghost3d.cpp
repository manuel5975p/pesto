// Compressible sharp ghost wall in TRUE 3D (nz>1): quiescent invariance (no
// spurious currents or force), impulsive freestream stability + positive drag
// on a sphere, and near-symmetry of the transverse force. Exercises the 3D
// paths of DeviceCompGhost the 2D cylinder test cannot: z-normals, +z/-z
// interface faces (ndir=6), mz momentum in the mirror fill and force tally.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
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

    const int n = 64;
    const double L = 2 * pi, R = L / 8;
    const double cx = L / 2 + 0.013, cy = L / 2 + 0.017, cz = L / 2 + 0.011;
    const double gamma = 1.4, rhoInf = 1.0, pInf = 1.0 / (gamma * 0.04); // M 0.2
    const auto g = Grid::make(n, n, n, L, L, L);
    auto sdf = [&](double x, double y, double z) {
        return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) +
                         (z - cz) * (z - cz)) -
               R;
    };
    const double mu = rhoInf * 2 * R / 100.0; // Re 100 on uInf = 1

    auto dev = gpu::DeviceComp::make(g, gamma, mu);
    auto ghost = gpu::DeviceCompGhost::make(g, sdf, gamma, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ghost || !ds || !red) return EXIT_FAILURE;
    dev->setStageHook([&](gpu::DeviceCompState& st) { ghost->apply(st); });

    // Quiescent gas + wall must stay exactly quiescent in 3D.
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
        vmax = std::max({vmax, std::abs(host.mx[c]), std::abs(host.my[c]),
                         std::abs(host.mz[c])});
    const auto F0 = ghost->force(*dev, *ds);
    std::printf("quiescent: max|m| %.2e force (%.2e, %.2e, %.2e)\n", vmax, F0.fx,
                F0.fy, F0.fz);
    check(vmax < 1e-5, "3D quiescent gas stays at rest");
    check(std::abs(F0.fx) < 1e-4 && std::abs(F0.fy) < 1e-4 &&
              std::abs(F0.fz) < 1e-4,
          "3D no spurious force");

    // Tapered freestream in +x: stable, positive drag, near-symmetric transverse
    // force, bounded speed.
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double d =
                    sdf((i + 0.5) * g.hx, (j + 0.5) * g.hy, (k + 0.5) * g.hz);
                const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
                const double u = f * f * (3 - 2 * f);
                host.rho[c] = rhoInf;
                host.mx[c] = rhoInf * u;
                host.my[c] = 0.0;
                host.mz[c] = 0.0;
                host.E[c] = pInf / (gamma - 1.0) + 0.5 * rhoInf * u * u;
            }
    (void)ds->upload(host);
    double t = 0.0;
    while (t < 1.0) {
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
        const double u = host.mx[c] / host.rho[c], v = host.my[c] / host.rho[c],
                     w = host.mz[c] / host.rho[c];
        if (!std::isfinite(u) || !std::isfinite(v) || !std::isfinite(w))
            finite = false;
        smax = std::max(smax, std::sqrt(u * u + v * v + w * w));
    }
    // Sphere reference area pi R^2.
    const double cd = F.fx / (0.5 * rhoInf * pi * R * R);
    std::printf("freestream t=1: cd %.3f (fy %.2e fz %.2e) max|u| %.3f\n", cd,
                F.fy, F.fz, smax);
    check(finite, "3D freestream evolution finite");
    check(cd > 0.3 && cd < 30.0, "3D drag positive and sane (early transient)");
    check(std::abs(F.fy) < 0.25 * std::abs(F.fx) &&
              std::abs(F.fz) < 0.25 * std::abs(F.fx),
          "3D transverse force small (symmetry)");
    check(smax < 3.0, "3D speed bounded");

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
