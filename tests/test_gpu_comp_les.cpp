// GPU compressible Smagorinsky LES (3D): csLes==0 is bit-identical to molecular
// DNS, the SGS stress stays conservative (mass/momentum/total-energy drift
// bounded), and csLes>0 adds dissipation (lower kinetic energy than DNS on an
// underresolved 3D Taylor-Green vortex). Runs a genuine 3D grid (nz>1) — the
// SGS model is 3D-valid in both physics (the incompressible one since
// 2026-07-20). Periodic box, no wall: the mu_t image hook is not involved.
// The LES viscous dt bound (W15) reads the max of the solver's own FP32 mu_t:
// on a 3D state (smag and sigma) that max agrees with the host FP64
// evaluation of the model (the deleted FP64 reducer, kept here as the oracle)
// to 1e-6 relative, stableDt leaves mut() at that state's mu_t, the bound is
// inert where it does not bind (== a DNS twin's dt exactly) and is the
// formula where it does.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/sgs.hpp"
#include "dns/transport.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

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

struct Sums { double mass, px, py, pz, E, ke, absM; };

Sums sums(const CompState& s)
{
    Sums r{0, 0, 0, 0, 0, 0, 0};
    for (std::size_t c = 0; c < s.rho.size(); ++c) {
        r.mass += s.rho[c];
        r.px += s.mx[c];
        r.py += s.my[c];
        r.pz += s.mz[c];
        r.absM += std::abs(s.mx[c]) + std::abs(s.my[c]) + std::abs(s.mz[c]);
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

// The TGV plus a 3D w and a density modulation, so the sigma operator is
// nonzero (the bare TGV has w == 0: D_sigma == 0) and rho enters mu_t.
CompState tgv3dState(const Grid& g, double gamma, double M)
{
    auto s = tgvState(g, gamma, M);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy,
                             z = (k + 0.5) * g.hz;
                const std::size_t c = g.idx(i, j, k);
                const double r = 1.0 + 0.1 * std::cos(x + y) * std::sin(z);
                const double u = s.mx[c], v = s.my[c];
                const double w = 0.3 * std::sin(x) * std::sin(2 * y) *
                                 std::cos(z);
                const double p = (gamma - 1) *
                    (s.E[c] - 0.5 * (u * u + v * v)); // rho0 == 1
                s.rho[c] = r;
                s.mx[c] = r * u;
                s.my[c] = r * v;
                s.mz[c] = r * w;
                s.E[c] = p / (gamma - 1) + 0.5 * r * (u * u + v * v + w * w);
            }
    return s;
}

// Host FP64 max over cells of rho (C Delta)^2 op(grad u), u = m/rho widened
// per neighbour, central differences on the periodic uniform box: the FP64
// functor stableDt used before W15, verbatim in arithmetic.
double nutMaxFp64(const Grid& g, const CompState& s, double cs, int sigma)
{
    const double d2 = std::pow(g.hx * g.hy * g.hz, 2.0 / 3.0);
    auto vel = [&](std::size_t c, int comp) {
        const auto& m = comp == 0 ? s.mx : comp == 1 ? s.my : s.mz;
        return m[c] / s.rho[c];
    };
    double best = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t p[3] = {g.idx((i + 1) % g.nx, j, k),
                                          g.idx(i, (j + 1) % g.ny, k),
                                          g.idx(i, j, (k + 1) % g.nz)};
                const std::size_t m[3] = {g.idx((i + g.nx - 1) % g.nx, j, k),
                                          g.idx(i, (j + g.ny - 1) % g.ny, k),
                                          g.idx(i, j, (k + g.nz - 1) % g.nz)};
                const double inv[3] = {0.5 / g.hx, 0.5 / g.hy, 0.5 / g.hz};
                double gd[3][3];
                for (int cc = 0; cc < 3; ++cc)
                    for (int d = 0; d < 3; ++d)
                        gd[cc][d] = (vel(p[d], cc) - vel(m[d], cc)) * inv[d];
                const double nut = s.rho[g.idx(i, j, k)] * cs * cs * d2 *
                                   sgsRate(sigma, gd);
                best = std::max(best, nut);
            }
    return best;
}

// The LES dt bound on one 3D state for one model at Cs `cs` (W15). bind: Cs
// chosen so the viscous limit is the binding one.
void lesDtBound(const Grid& g, double gamma, double mu, const CompState& st,
                double cs, SgsModel model, bool bind,
                gpu::ReduceScratch& red)
{
    const char* name = model == SgsModel::Sigma ? "sigma" : "smag";
    auto dev = gpu::DeviceComp::make(g, gamma, mu, cs, model);
    auto twin = gpu::DeviceComp::make(g, gamma, mu); // DNS: no SGS bound
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !twin || !ds) {
        check(false, std::string(name) + ": LES dt rig builds");
        return;
    }
    (void)ds->upload(st);
    CompState h = st;
    (void)ds->download(h); // the FP32 state, widened: the oracle's input
    const double dtL = dev->stableDt(*ds, red);
    const double dtD = twin->stableDt(*ds, red);
    std::vector<gpu::Real> nut(g.size());
    (void)dev->mut().download(nut.data());
    const double f32 = double(*std::ranges::max_element(nut));
    const double f64 = nutMaxFp64(g, h, cs, model == SgsModel::Sigma ? 1 : 0);
    // mut() is s's mu_t after stableDt: the same field updateSgsNut gives.
    std::vector<gpu::Real> again(g.size());
    dev->updateSgsNut(*ds);
    (void)dev->mut().download(again.data());
    const double rhoMin = *std::ranges::min_element(h.rho);
    const double fac = Transport::constant(gamma, mu).diffusivityFactor();
    // The device grid's spacing is FP32 (GpuGrid), and stableDt reads it.
    const double hmin = std::min({double(gpu::Real(g.hx)), double(gpu::Real(g.hy)),
                                  double(gpu::Real(g.hz))});
    auto bound = [&](double nm) {
        return 0.25 * hmin * hmin * rhoMin / ((mu + nm) * fac);
    };
    std::printf("%s Cs %.3g: mu_t max FP32 %.9e FP64 %.9e (rel %.2e); dt LES "
                "%.9e DNS %.9e bound(FP32) %.9e bound(FP64) %.9e\n",
                name, cs, f32, f64, std::abs(f32 / f64 - 1), dtL, dtD,
                bound(f32), bound(f64));
    check(f64 > 0 && std::abs(f32 / f64 - 1) < 1e-6,
          std::string(name) + ": FP32 mu_t max == FP64 model max to 1e-6");
    check(nut == again,
          std::string(name) + ": stableDt leaves mut() at the state's mu_t");
    if (!bind) {
        check(dtL == dtD,
              std::string(name) + ": non-binding LES bound leaves dt exact");
        return;
    }
    check(dtL < dtD, std::string(name) + ": the LES viscous bound binds");
    check(std::abs(dtL / bound(f32) - 1) < 1e-14,
          std::string(name) + ": binding dt is the formula on the FP32 max");
    check(std::abs(dtL / bound(f64) - 1) < 1e-6,
          std::string(name) + ": binding dt == the FP64-functor dt to 1e-6");
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
    // Momentum against its own scale sum|m| (the TGV's total is zero, so the
    // old 1e-3 * N band was ~33 absolute and pinned nothing).
    const double mBand = 1e-6 * s0.absM;
    std::printf("momentum drift after %d steps: px %.3e  py %.3e  pz %.3e  "
                "(band 1e-6 sum|m| = %.3e)\n", steps, sLes.px - s0.px,
                sLes.py - s0.py, sLes.pz - s0.pz, mBand);
    check(std::abs(sLes.px - s0.px) < mBand &&
              std::abs(sLes.py - s0.py) < mBand &&
              std::abs(sLes.pz - s0.pz) < mBand,
          "LES conserves momentum to 1e-6 sum|m|");
    check(std::abs(sLes.E - s0.E) < 1e-4 * std::abs(s0.E),
          "LES conserves total energy");

    // Extra dissipation: SGS drains kinetic energy faster than molecular DNS.
    std::printf("KE: init %.4f  DNS %.4f  LES %.4f\n", s0.ke, sDns.ke, sLes.ke);
    check(std::isfinite(sLes.ke) && sLes.ke < sDns.ke,
          "LES dissipates more KE than DNS (3D)");

    // --- the LES dt bound: FP32 mu_t max vs the FP64 functor (W15) ---
    const auto ic3 = tgv3dState(g, gamma, M);
    lesDtBound(g, gamma, mu, ic3, 0.17, SgsModel::Smagorinsky, false, *scratch);
    lesDtBound(g, gamma, mu, ic3, 1.35, SgsModel::Sigma, false, *scratch);
    lesDtBound(g, gamma, mu, ic3, 5.0, SgsModel::Smagorinsky, true, *scratch);
    lesDtBound(g, gamma, mu, ic3, 40.0, SgsModel::Sigma, true, *scratch);

    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
