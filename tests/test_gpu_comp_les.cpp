// GPU compressible Smagorinsky LES (3D): csLes==0 is bit-identical to molecular
// DNS, the SGS stress stays conservative (mass/momentum/total-energy drift
// bounded), and csLes>0 adds dissipation (lower kinetic energy than DNS on an
// underresolved 3D Taylor-Green vortex). Runs a genuine 3D grid (nz>1) — the
// compressible SGS model is 3D-valid, unlike the 2D-only incompressible port.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
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



// Compressible Taylor-Green vortex at Mach M, constant background density.
CompState tgvState(const Grid& g, double gamma, double M)
{
    auto s = CompState::zeros(g);
    const double rho0 = 1.0, V0 = 1.0;
    const double p0 = rho0 * V0 * V0 / (gamma * M * M);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy,
                             z = (k + 0.5) * g.hz;
                const double u = V0 * std::sin(x) * std::cos(y) * std::cos(z);
                const double v = -V0 * std::cos(x) * std::sin(y) * std::cos(z);
                const double w = 0.0;
                const double p = p0 + rho0 * V0 * V0 / 16.0 *
                                          (std::cos(2 * x) + std::cos(2 * y)) *
                                          (std::cos(2 * z) + 2.0);
                const std::size_t c = g.idx(i, j, k);
                s.rho[c] = rho0;
                s.mx[c] = rho0 * u;
                s.my[c] = rho0 * v;
                s.mz[c] = rho0 * w;
                s.E[c] = p / (gamma - 1) + 0.5 * rho0 * (u * u + v * v + w * w);
            }
    return s;
}

struct Sums { double mass, px, py, pz, E, ke; };

Sums sums(const CompState& s)
{
    Sums r{0, 0, 0, 0, 0, 0};
    for (std::size_t c = 0; c < s.rho.size(); ++c) {
        r.mass += s.rho[c];
        r.px += s.mx[c];
        r.py += s.my[c];
        r.pz += s.mz[c];
        r.E += s.E[c];
        r.ke += 0.5 * (s.mx[c] * s.mx[c] + s.my[c] * s.my[c] + s.mz[c] * s.mz[c]) /
                s.rho[c];
    }
    return r;
}

double maxAbsDiff(const CompState& a, const CompState& b)
{
    double m = 0.0;
    auto scan = [&](const std::vector<double>& x, const std::vector<double>& y) {
        for (std::size_t i = 0; i < x.size(); ++i)
            m = std::max(m, std::abs(x[i] - y[i]));
    };
    scan(a.rho, b.rho); scan(a.mx, b.mx); scan(a.my, b.my);
    scan(a.mz, b.mz); scan(a.E, b.E);
    return m;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const int n = 32;
    const double L = 2 * pi, gamma = 1.4, M = 0.1;
    const auto g = Grid::make(n, n, n, L, L, L);
    const double mu = 1.0 / 1600.0; // Re 1600, deliberately underresolved at 32^3
    const auto ic = tgvState(g, gamma, M);

    auto scratch = gpu::ReduceScratch::make();
    if (!scratch) return EXIT_FAILURE;

    // --- csLes == 0 bit-identical to the no-LES default ctor ---
    auto devA = gpu::DeviceComp::make(g, gamma, mu);        // default (no LES)
    auto devB = gpu::DeviceComp::make(g, gamma, mu, 0.0);   // LES flag, Cs = 0
    if (!devA || !devB) return test::skipExitCode;
    auto dsA = gpu::DeviceCompState::make(g.size());
    auto dsB = gpu::DeviceCompState::make(g.size());
    if (!dsA || !dsB) return EXIT_FAILURE;
    (void)dsA->upload(ic);
    (void)dsB->upload(ic);
    const double dt0 = 0.5 * devA->stableDt(*dsA, *scratch);
    for (int it = 0; it < 20; ++it) {
        devA->step(*dsA, gpu::Real(dt0));
        devB->step(*dsB, gpu::Real(dt0));
    }
    CompState a = ic, b = ic;
    (void)dsA->download(a);
    (void)dsB->download(b);
    check(devA->les() == false && devB->les() == false, "Cs=0 reports LES off");
    check(maxAbsDiff(a, b) == 0.0, "Cs=0 bit-identical to molecular DNS");

    // --- LES on: 3D run, conservation, extra dissipation vs DNS ---
    auto devDns = gpu::DeviceComp::make(g, gamma, mu);
    auto devLes = gpu::DeviceComp::make(g, gamma, mu, 0.18); // Smagorinsky Cs
    if (!devDns || !devLes) return EXIT_FAILURE;
    check(devLes->les() == true, "Cs>0 reports LES on");
    auto dsDns = gpu::DeviceCompState::make(g.size());
    auto dsLes = gpu::DeviceCompState::make(g.size());
    if (!dsDns || !dsLes) return EXIT_FAILURE;
    (void)dsDns->upload(ic);
    (void)dsLes->upload(ic);

    const Sums s0 = sums(ic);
    const int steps = 300;
    for (int it = 0; it < steps; ++it) {
        const double dtL = 0.5 * devLes->stableDt(*dsLes, *scratch);
        devLes->step(*dsLes, gpu::Real(dtL));
        const double dtD = 0.5 * devDns->stableDt(*dsDns, *scratch);
        devDns->step(*dsDns, gpu::Real(dtD));
    }
    CompState hLes = ic, hDns = ic;
    (void)dsLes->download(hLes);
    (void)dsDns->download(hDns);
    const Sums sLes = sums(hLes), sDns = sums(hDns);

    // Conservative SGS: mass, momentum, total energy drift bounded (FP32).
    const double scale = std::max(std::abs(s0.mass), double(g.size()));
    check(std::abs(sLes.mass - s0.mass) < 1e-4 * scale, "LES conserves mass");
    check(std::abs(sLes.px - s0.px) < 1e-3 * scale &&
              std::abs(sLes.py - s0.py) < 1e-3 * scale &&
              std::abs(sLes.pz - s0.pz) < 1e-3 * scale,
          "LES conserves momentum");
    check(std::abs(sLes.E - s0.E) < 1e-4 * std::abs(s0.E),
          "LES conserves total energy");

    // Extra dissipation: SGS drains kinetic energy faster than molecular DNS.
    std::printf("KE: init %.4f  DNS %.4f  LES %.4f\n", s0.ke, sDns.ke, sLes.ke);
    check(std::isfinite(sLes.ke) && sLes.ke < sDns.ke,
          "LES dissipates more KE than DNS (3D)");

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
