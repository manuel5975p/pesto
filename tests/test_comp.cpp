// Compressible NS: isentropic vortex convergence, acoustic speed, low-Mach
// Taylor-Green decay, exact conservation, nz=1 vs z-invariant 3D parity.
#include "dns/comp.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;



// Isentropic vortex (beta strength) centred at (x0,y0), background (uInf,vInf),
// T0 = 1, rho0 = 1, R = 1: exact solution advects rigidly.
CompState vortexState(const Comp& c, double x0, double y0, double beta,
                        double uInf, double vInf)
{
    auto s = CompState::zeros(c.g);
    const double L = c.g.nx * c.g.hx;
    for (int j = 0; j < c.g.ny; ++j)
        for (int i = 0; i < c.g.nx; ++i) {
            // Nearest periodic image of the vortex centre.
            double dx = (i + 0.5) * c.g.hx - x0;
            double dy = (j + 0.5) * c.g.hy - y0;
            dx -= L * std::round(dx / L);
            dy -= L * std::round(dy / L);
            const double r2 = dx * dx + dy * dy;
            const double f = beta / (2.0 * pi) * std::exp(0.5 * (1.0 - r2));
            const double u = uInf - f * dy;
            const double v = vInf + f * dx;
            const double T = 1.0 - (c.gamma() - 1.0) * beta * beta /
                                       (8.0 * c.gamma() * pi * pi) *
                                       std::exp(1.0 - r2);
            const double rho = std::pow(T, 1.0 / (c.gamma() - 1.0));
            const double p = rho * T;
            const std::size_t k = c.g.idx(i, j, 0);
            s.rho[k] = rho;
            s.mx[k] = rho * u;
            s.my[k] = rho * v;
            s.E[k] = p / (c.gamma() - 1.0) + 0.5 * rho * (u * u + v * v);
        }
    return s;
}

void advance(const Comp& c, CompState& s, double T)
{
    double t = 0.0;
    while (t < T) {
        const double dt = std::min(*c.stableDt(s), T - t);
        c.step(s, dt);
        t += dt;
    }
}

// L2 density error of the vortex after advecting one full period.
double vortexError(int n)
{
    const double L = 10.0;
    const auto g = Grid::make(n, n, 1, L, L, L / n);
    const auto c = Comp::make(g, 1.4, 0.0);
    auto s = vortexState(c, 5.0, 5.0, 5.0, 1.0, 1.0);
    const auto exact = s;
    advance(c, s, L); // (uInf,vInf)=(1,1): back to start after t = L
    double e2 = 0.0;
    for (std::size_t k = 0; k < g.size(); ++k) {
        const double d = s.rho[k] - exact.rho[k];
        e2 += d * d;
    }
    return std::sqrt(e2 / double(g.size()));
}

void testVortex()
{
    const double e64 = vortexError(64);
    const double e128 = vortexError(128);
    const double order = std::log2(e64 / e128);
    std::printf("vortex L2(rho): 64 %.3e  128 %.3e  order %.2f\n", e64, e128, order);
    check(e128 < 3e-3, "vortex error small at 128^2");
    check(order > 1.7, "vortex convergence order ~2");
}

void testAcoustic()
{
    // Right-going simple wave on a quiet background (rho0 = 1, p0 = 1/gamma,
    // c0 = 1): u' = c0 rho'/rho0. Track the pulse peak after t = 3.
    const int n = 256;
    const double L = 20.0;
    const auto g = Grid::make(n, n / 32, 1, L, L / 32, 1.0);
    const auto c = Comp::make(g, 1.4, 0.0);
    auto s = CompState::zeros(g);
    const double eps = 1e-4, x0 = 5.0, sig = 0.6;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * g.hx;
            const double dr = eps * std::exp(-(x - x0) * (x - x0) / (sig * sig));
            const double rho = 1.0 + dr;
            const double p = 1.0 / c.gamma() + dr; // dp = c0^2 drho
            const double u = dr;                 // du = c0 drho / rho0
            const std::size_t k = g.idx(i, j, 0);
            s.rho[k] = rho;
            s.mx[k] = rho * u;
            s.E[k] = p / (c.gamma() - 1.0) + 0.5 * rho * u * u;
        }
    advance(c, s, 3.0);
    int imax = 0;
    for (int i = 0; i < g.nx; ++i)
        if (s.rho[g.idx(i, 0, 0)] > s.rho[g.idx(imax, 0, 0)]) imax = i;
    const double xpk = (imax + 0.5) * g.hx;
    std::printf("acoustic peak at %.3f (expect %.3f)\n", xpk, x0 + 3.0);
    check(std::abs(xpk - (x0 + 3.0)) < 3.0 * g.hx, "acoustic speed = c0");
}

void testTaylorGreen()
{
    // Low-Mach TGV: kinetic energy decays like exp(-4 nu t) in the
    // incompressible limit.
    const int n = 64;
    const auto g = Grid::make(n, n, 1, 2 * pi, 2 * pi, 2 * pi / n);
    const double M = 0.05, mun = 0.01;
    const auto c = Comp::make(g, 1.4, mun);
    auto s = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
            const double u = M * std::sin(x) * std::cos(y);
            const double v = -M * std::cos(x) * std::sin(y);
            const double p = 1.0 / c.gamma() + 0.25 * M * M * (std::cos(2 * x) + std::cos(2 * y));
            const std::size_t k = g.idx(i, j, 0);
            s.rho[k] = 1.0;
            s.mx[k] = u;
            s.my[k] = v;
            s.E[k] = p / (c.gamma() - 1.0) + 0.5 * (u * u + v * v);
        }
    auto ke = [&](const CompState& q) {
        double K = 0.0;
        for (std::size_t k = 0; k < g.size(); ++k)
            K += 0.5 * (q.mx[k] * q.mx[k] + q.my[k] * q.my[k]) / q.rho[k];
        return K;
    };
    const double K0 = ke(s);
    const double T = 2.0;
    advance(c, s, T);
    const double ratio = ke(s) / K0, exact = std::exp(-4.0 * mun * T);
    std::printf("TGV decay: K/K0 %.5f exact %.5f\n", ratio, exact);
    check(std::abs(ratio - exact) < 0.01 * exact, "low-Mach TGV decay ~ exp(-4 nu t)");
}

void testConservation()
{
    const int n = 48;
    const auto g = Grid::make(n, n, 1, 2 * pi, 2 * pi, 2 * pi / n);
    const auto c = Comp::make(g, 1.4, 0.02);
    auto s = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
            const double rho = 1.0 + 0.2 * std::sin(x + 2 * y);
            const double u = 0.3 * std::sin(2 * x) * std::cos(y);
            const double v = 0.3 * std::cos(x + y);
            const double p = 1.0 + 0.1 * std::cos(x) * std::sin(y);
            const std::size_t k = g.idx(i, j, 0);
            s.rho[k] = rho;
            s.mx[k] = rho * u;
            s.my[k] = rho * v;
            s.E[k] = p / (c.gamma() - 1.0) + 0.5 * rho * (u * u + v * v);
        }
    auto sums = [&](const CompState& q, double out[4]) {
        out[0] = out[1] = out[2] = out[3] = 0.0;
        for (std::size_t k = 0; k < g.size(); ++k) {
            out[0] += q.rho[k];
            out[1] += q.mx[k];
            out[2] += q.my[k];
            out[3] += q.E[k];
        }
    };
    double s0[4], s1[4];
    sums(s, s0);
    for (int it = 0; it < 100; ++it) c.step(s, *c.stableDt(s));
    sums(s, s1);
    bool ok = true;
    for (int q = 0; q < 4; ++q) {
        const double scale = std::max(std::abs(s0[q]), double(g.size()));
        if (std::abs(s1[q] - s0[q]) > 1e-11 * scale) ok = false;
    }
    std::printf("conservation drift: rho %.2e mx %.2e my %.2e E %.2e\n",
                s1[0] - s0[0], s1[1] - s0[1], s1[2] - s0[2], s1[3] - s0[3]);
    check(ok, "flux form conserves rho, m, E to roundoff");
}

} // namespace

void test2d3dParity()
{
    // z-invariant 3D run must match the nz=1 run slice for slice.
    const int n = 32;
    const auto g2 = Grid::make(n, n, 1, 10.0, 10.0, 1.0);
    const auto g3 = Grid::make(n, n, 4, 10.0, 10.0, 4.0);
    const auto c2 = Comp::make(g2, 1.4, 2e-3);
    const auto c3 = Comp::make(g3, 1.4, 2e-3);
    auto s2 = vortexState(c2, 5.0, 5.0, 5.0, 1.0, 1.0);
    auto s3 = CompState::zeros(g3);
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t a = g2.idx(i, j, 0), b = g3.idx(i, j, k);
                s3.rho[b] = s2.rho[a];
                s3.mx[b] = s2.mx[a];
                s3.my[b] = s2.my[a];
                s3.E[b] = s2.E[a];
            }
    const double dt = 0.9 * *c2.stableDt(s2);
    for (int it = 0; it < 10; ++it) {
        c2.step(s2, dt);
        c3.step(s3, dt);
    }
    double err = 0.0, wmax = 0.0;
    for (int k = 0; k < 4; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t a = g2.idx(i, j, 0), b = g3.idx(i, j, k);
                err = std::max(err, std::abs(s3.rho[b] - s2.rho[a]));
                err = std::max(err, std::abs(s3.mx[b] - s2.mx[a]));
                err = std::max(err, std::abs(s3.E[b] - s2.E[a]));
                wmax = std::max(wmax, std::abs(s3.mz[b]));
            }
    std::printf("2d/3d parity: err %.2e, max|mz| %.2e\n", err, wmax);
    check(err < 1e-13 && wmax < 1e-15, "nz=1 == z-invariant 3D, mz stays 0");
}

// A diverged state must be REPORTED, not asserted away. In Release the old
// asserts vanished and sqrt(gamma*p/r) with p < 0 returned a NaN dt, which the
// time loop rode forever -- silent non-termination instead of exit 7.
void testDivergedStableDt()
{
    const int n = 8;
    const Grid g = Grid::make(n, n, 1, 1.0, 1.0, 1.0);
    const Comp c = Comp::make(g, 1.4, 0.0);

    CompState s = CompState::zeros(g);
    for (std::size_t i = 0; i < g.size(); ++i) { s.rho[i] = 1.0; s.E[i] = 2.5; }
    check(c.stableDt(s).has_value(), "stableDt succeeds on a healthy state");

    CompState neg = s;
    neg.rho[g.idx(3, 4, 0)] = -1e-3;
    check(!c.stableDt(neg).has_value(),
          "stableDt reports a NEGATIVE-density state, not a NaN dt");

    CompState lowp = s;
    lowp.E[g.idx(2, 2, 0)] = -5.0;   // E < |m|^2/2rho  =>  p < 0
    check(!c.stableDt(lowp).has_value(),
          "stableDt reports a NEGATIVE-pressure state, not a NaN dt");

    CompState nan = s;
    nan.rho[g.idx(1, 1, 0)] = std::nan("");
    check(!c.stableDt(nan).has_value(), "stableDt reports a NaN density");
}

int main()
{
    testVortex();
    testAcoustic();
    testTaylorGreen();
    testConservation();
    test2d3dParity();
    testDivergedStableDt();
    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
