// Mesh insertion into the 3D compressible solver: a triangle-mesh body
// (MeshSdf over an icosphere) driven through DeviceCompGhost must reproduce the
// analytic-sphere sdf drag within mesh-discretization error. Proves the ghost
// wall accepts an arbitrary mesh sdf in true 3D (the compressible analogue of
// the incompressible val_mesh_sphere check).
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/meshsdf.hpp"
#include "gpu_test_util.hpp"
#include "mesh_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
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

// Run a tapered-freestream sphere to time tEnd and return the drag force fx and
// the transverse magnitude, for a given signed-distance body.
struct Run { double fx, ftrans, smax; bool finite; };

Run sphereRun(const Grid& g, double gamma, double mu, double rhoInf,
              double pInf, const std::function<double(double, double, double)>& sdf,
              double tEnd)
{
    auto dev = gpu::DeviceComp::make(g, gamma, mu);
    auto ghost = gpu::DeviceCompGhost::make(g, sdf, gamma, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ghost || !ds || !red) return {0, 0, 0, false};
    dev->setStageHook([&](gpu::DeviceCompState& st) { ghost->apply(st); });

    auto host = CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double d =
                    sdf((i + 0.5) * g.hx, (j + 0.5) * g.hy, (k + 0.5) * g.hz);
                const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
                const double u = f * f * (3 - 2 * f);
                host.rho[c] = rhoInf;
                host.mx[c] = rhoInf * u;
                host.E[c] = pInf / (gamma - 1.0) + 0.5 * rhoInf * u * u;
            }
    (void)ds->upload(host);
    double t = 0.0;
    while (t < tEnd) {
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
    return {F.fx, std::sqrt(F.fy * F.fy + F.fz * F.fz), smax, finite};
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const int n = 64;
    const double L = 2 * pi, R = L / 8;
    const double cx = L / 2 + 0.013, cy = L / 2 + 0.017, cz = L / 2 + 0.011;
    const double gamma = 1.4, rhoInf = 1.0, pInf = 1.0 / (gamma * 0.04); // M 0.2
    const double mu = rhoInf * 2 * R / 100.0;                            // Re 100
    const auto g = Grid::make(n, n, n, L, L, L);
    const double tEnd = 0.8;

    // Analytic sphere sdf (reference).
    auto analytic = [&](double x, double y, double z) {
        return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy) +
                         (z - cz) * (z - cz)) -
               R;
    };
    // Mesh body: icosphere of the same radius/centre, wrapped in MeshSdf.
    const MeshSdf meshSdf(meshtest::icosphere(R, {cx, cy, cz}, 3), false);
    check(meshSdf.watertight(), "icosphere mesh watertight");
    auto mesh = [&](double x, double y, double z) { return meshSdf(x, y, z); };

    const Run a = sphereRun(g, gamma, mu, rhoInf, pInf, analytic, tEnd);
    const Run m = sphereRun(g, gamma, mu, rhoInf, pInf, mesh, tEnd);
    if (!a.finite && a.fx == 0.0) return test::skipExitCode; // device alloc fail

    const double qA = 0.5 * rhoInf * pi * R * R;
    const double cdA = a.fx / qA, cdM = m.fx / qA;
    const double rel = std::abs(cdM - cdA) / cdA;
    std::printf("cd analytic %.3f  mesh %.3f  rel diff %.1f%%\n", cdA, cdM,
                100 * rel);
    std::printf("transverse: analytic %.2e mesh %.2e ; max|u| a %.3f m %.3f\n",
                a.ftrans, m.ftrans, a.smax, m.smax);

    check(a.finite && m.finite, "both runs finite");
    check(cdM > 0.3 && cdM < 30.0, "mesh drag positive and sane");
    check(rel < 0.15, "mesh cd matches analytic sphere within 15%");
    check(m.ftrans < 0.25 * std::abs(m.fx), "mesh transverse force small");

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
