// Temporal order of the integrators, isolated from space: ONE fixed grid, so
// the semi-discrete operator is identical across runs and only dt changes.
// Each run goes to t = 1 at dt, dt/2, dt/4 and is compared with the FP64 run
// on the same grid at dt/16; the observed order log2(e(dt)/e(dt/2)) must be
// the nominal one, band [2.7, 3.3] for order 3.
//   * Comp (CPU, FP64) and DeviceComp (FP32): SSP-RK3, order 3.
//   * dns::TimeStepper (CPU, FP64) and gpu::TimeStepper (FP32),
//     incompressible: SSP-RK3 with an EXACT discrete projection after every
//     stage (the FFT Poisson solve inverts the very div-grad it is built
//     from), so on the periodic box it is RK3 applied to du/dt = P L(u):
//     nominal order 3, same band. Pure Taylor-Green is NOT used: its advection
//     is a gradient, the run is a linear decay at rate 2 nu and RK3's error at
//     2 nu dt sits under FP32 rounding. A second streamfunction mode makes the
//     nonlinearity real.
// The device runs carry FP32 rounding on top of the time error. The rounding
// of each device run is MEASURED -- device vs host at the SAME dt, where the
// time errors are identical -- and must stay under 1/10 of that run's time
// error, or the band would measure rounding instead of the integrator. That is
// what sets the coarse dt and the flow (stated at each case): the time error
// falls as dt^3 while the accumulated rounding grows with the step count, and
// the textbook setups (r0 = 1 vortex at the CFL 0.4 dt, k = 1 Taylor-Green)
// measured rounding at 0.6x and 5x of the dt/4 time error.
// The host half runs without a device; the test then skips (77) unless it
// failed.
#include "dns/comp.hpp"
#include "dns/field.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/timestepper.hpp"
#include "dns/timestepper.hpp"
#include "permute_test_util.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

constexpr double kGamma = 1.4, kMu = 1e-2, kT = 1.0;
constexpr double kLo = 2.7, kHi = 3.3;

void orderCheck(const std::string& tag, const double e[3])
{
    const double o1 = std::log2(e[0] / e[1]), o2 = std::log2(e[1] / e[2]);
    std::printf("%s: e(dt) %.3e e(dt/2) %.3e e(dt/4) %.3e orders %.3f %.3f "
                "(expect [%.1f, %.1f])\n",
                tag.c_str(), e[0], e[1], e[2], o1, o2, kLo, kHi);
    check(o1 >= kLo && o1 <= kHi && o2 >= kLo && o2 <= kHi,
          tag + ": observed order in [2.7, 3.3]");
}

// Device rounding of each run (device vs host at the same dt) against that
// run's time error.
void floorCheck(const std::string& tag, const double e[3], const double fl[3])
{
    double worst = 0.0;
    for (int m = 0; m < 3; ++m) worst = std::max(worst, fl[m] / e[m]);
    std::printf("%s: FP32 rounding per run %.3e %.3e %.3e, worst rounding / "
                "time error %.3f\n", tag.c_str(), fl[0], fl[1], fl[2], worst);
    check(worst < 0.1, tag + ": FP32 rounding < 1/10 of the time error");
}

// --- compressible --------------------------------------------------------

// Isentropic vortex (test_comp's, an exact Euler solution for any core
// radius r0), L = 10, strength 5, stream (1, 1), core radius r0 = 0.5 (1.6
// cells): the small core spins fast against the acoustic dt, which is what
// lifts the time error clear of FP32 rounding (r0 = 1: rounding 0.18 of the
// dt/4 time error; r0 = 0.5: 0.04 -- measured, then chosen).
CompState vortex(const Grid& g)
{
    auto s = CompState::zeros(g);
    const double L = g.nx * g.hx, x0 = 5.0, y0 = 5.0, beta = 5.0, r0 = 0.5;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            double dx = (i + 0.5) * g.hx - x0, dy = (j + 0.5) * g.hy - y0;
            dx -= L * std::round(dx / L);
            dy -= L * std::round(dy / L);
            dx /= r0;
            dy /= r0;
            const double r2 = dx * dx + dy * dy;
            const double f = beta / (2.0 * pi) * std::exp(0.5 * (1.0 - r2));
            const double u = 1.0 - f * dy, v = 1.0 + f * dx;
            const double T = 1.0 - (kGamma - 1.0) * beta * beta
                                       / (8.0 * kGamma * pi * pi) * std::exp(1.0 - r2);
            const double rho = std::pow(T, 1.0 / (kGamma - 1.0));
            const std::size_t c = g.idx(i, j, 0);
            s.rho[c] = rho;
            s.mx[c] = rho * u;
            s.my[c] = rho * v;
            s.E[c] = rho * T / (kGamma - 1.0) + 0.5 * rho * (u * u + v * v);
        }
    return s;
}

CompState cpuComp(const Grid& g, const CompState& s0, int steps)
{
    const Comp c = Comp::make(g, kGamma, kMu);
    auto s = s0;
    for (int n = 0; n < steps; ++n) c.step(s, kT / steps);
    return s;
}

bool gpuComp(const Grid& g, const CompState& s0, int steps, CompState& out)
{
    auto dev = gpu::DeviceComp::make(g, kGamma, kMu);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !ds) return false;
    (void)ds->upload(s0);
    for (int n = 0; n < steps; ++n) dev->step(*ds, gpu::Real(kT / steps));
    out = CompState::zeros(g);
    (void)ds->download(out);
    return true;
}

// Coarse dt = 1/12: 1.9x stableDt at cfl 0.4 (~0.043 on this grid), i.e.
// (|u|+c) dt/h ~ 0.78 per axis -- inside SSP-RK3's central stability limit.
// At dt = 1/20 (r0 = 1) the dt/4 run's FP32 rounding was ~0.6 of its time
// error.
constexpr int kN0 = 12;

void compressible(bool device)
{
    const auto g = Grid::make(32, 32, 1, 10.0, 10.0, 10.0 / 32);
    const auto s0 = vortex(g);
    const auto ref = cpuComp(g, s0, 16 * kN0);
    CompState host[3];
    double e[3];
    for (int m = 0; m < 3; ++m) {
        host[m] = cpuComp(g, s0, kN0 << m);
        e[m] = permtest::relState(host[m], ref);
    }
    if (!device) {
        std::printf("vortex: stableDt(cfl 0.4) %.4e, coarse dt %.4e\n",
                    *Comp::make(g, kGamma, kMu).stableDt(s0), kT / kN0);
        orderCheck("Comp (CPU) SSP-RK3", e);
        return;
    }
    double eg[3], fl[3];
    for (int m = 0; m < 3; ++m) {
        CompState s;
        if (!gpuComp(g, s0, kN0 << m, s)) { check(false, "GPU setup"); return; }
        eg[m] = permtest::relState(s, ref);
        fl[m] = permtest::relState(s, host[m]);
    }
    orderCheck("DeviceComp SSP-RK3", eg);
    floorCheck("DeviceComp SSP-RK3", e, fl);
}

// --- incompressible ------------------------------------------------------

constexpr double kNu = 0.01;

// Taylor-Green at k = XK plus an oblique second mode, from a NODE
// streamfunction differenced onto the MAC faces: discretely divergence-free,
// so the first projection does not perturb it. nz = 4 (z-invariant), as
// gpu_tgv.
VectorField incompIC(const Grid& g, int kk)
{
    auto psi = [kk](double x, double y) {
        return (std::sin(kk * x) * std::sin(kk * y)
                + 0.5 * std::cos(kk * x + 0.3) * std::sin(2 * kk * y + 0.7)) / kk;
    };
    VectorField U = VectorField::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double x = i * g.hx, y = j * g.hy;
                const std::size_t c = g.idx(i, j, k);
                U.u[c] = (psi(x, y + g.hy) - psi(x, y)) / g.hy;
                U.v[c] = -(psi(x + g.hx, y) - psi(x, y)) / g.hx;
            }
    return U;
}

double relField(const VectorField& a, const VectorField& b)
{
    const auto d = test::fieldDiff(a, b);
    return d.maxErr / d.scale;
}

VectorField cpuIncomp(const Grid& g, const VectorField& U0, int steps)
{
    TimeStepper ts(g, {kNu, 0.4});
    auto U = U0;
    for (int s = 0; s < steps; ++s) ts.step(U, kT / steps);
    return U;
}

bool gpuIncomp(const Grid& g, const VectorField& U0, int steps, VectorField& U)
{
    U = U0;
    auto D = gpu::DeviceField::make(g.size());
    auto ts = gpu::TimeStepper::make(g, {kNu, 0.4});
    if (!D || !ts) return false;
    (void)gpu::uploadField(U, *D);
    for (int s = 0; s < steps; ++s) ts->step(*D, kT / steps);
    (void)gpu::downloadField(*D, U);
    return true;
}

void incompressible(bool device)
{
    const int n = 32;
    const auto g = Grid::make(n, n, 4, 2 * pi, 2 * pi, 8 * pi / n);
    // k = 3 (and 6 in y for the second mode) with coarse dt = 1/8. The time
    // error per unit time grows as omega^4 dt^3 while the CFL dt does not
    // depend on k, so the higher mode is what lifts the dt/4 run's time error
    // to >= 10x its FP32 rounding (k = 1 at dt = 1/16: rounding 5x the time
    // error; measured, then chosen). The grid still resolves both modes (10.7
    // and 5.3 cells per wavelength); this is a TIME test on a fixed grid.
    const int kk = 3, n0 = 8;
    const VectorField U0 = incompIC(g, kk);
    if (!device) {
        double umax = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c)
            umax = std::max({umax, std::abs(U0.u[c]), std::abs(U0.v[c])});
        std::printf("incompressible: max|u| %.3f, coarse dt %.4e, Courant "
                    "max|u| dt/h %.3f\n", umax, kT / n0, umax * kT / n0 / g.hx);
    }
    const VectorField ref = cpuIncomp(g, U0, 16 * n0);
    VectorField host[3];
    double e[3];
    for (int m = 0; m < 3; ++m) {
        host[m] = cpuIncomp(g, U0, n0 << m);
        e[m] = relField(host[m], ref);
    }
    if (!device) {
        orderCheck("incompressible TimeStepper (CPU)", e);
        return;
    }
    double eg[3], fl[3];
    for (int m = 0; m < 3; ++m) {
        VectorField U;
        if (!gpuIncomp(g, U0, n0 << m, U)) { check(false, "GPU setup"); return; }
        eg[m] = relField(U, ref);
        fl[m] = relField(U, host[m]);
    }
    orderCheck("incompressible gpu::TimeStepper", eg);
    floorCheck("incompressible gpu::TimeStepper", e, fl);
}

} // namespace

int main()
{
    compressible(false);
    incompressible(false);
    if (!gpu::deviceAvailable()) {
        if (dnstest::gFailures) return dnstest::finish();
        dnstest::skip("no HIP device (CPU half passed)");
    }
    compressible(true);
    incompressible(true);
    return dnstest::finish();
}
