// Full solver vs analytic 2D Taylor-Green decay + spatial convergence order.
#include "dns/operators.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using namespace dns;
using std::numbers::pi;

namespace {

double runTgv(int n, double nu, double T)
{
    const auto g = Grid::make(n, n, 1, 2 * pi, 2 * pi, 2 * pi / n);
    auto U = VectorField::zeros(g);
    taylorGreen(g, nu, 0.0, U);

    TimeStepper ts(g, {.nu = nu, .cfl = 0.4});
    double t = 0.0;
    while (t < T)
    {
        const double dt = ts.suggestDt(U, T - t);
        ts.step(U, dt);
        t += dt;
    }
    return taylorGreenError(g, nu, t, U);
}

} // namespace

int main()
{
    const double nu = 0.05, T = 0.5;

    const double e32 = runTgv(32, nu, T);
    const double e64 = runTgv(64, nu, T);
    const double order = std::log2(e32 / e64);

    std::printf("err n=32: %.3e   n=64: %.3e   observed order: %.2f\n", e32, e64, order);

    int fails = 0;
    if (e64 > 2e-3) { std::puts("FAIL: error too large"); ++fails; }
    if (order < 1.7) { std::puts("FAIL: not 2nd order"); ++fails; }

    // A diverged (NaN-seeded) field must NOT get a usable dt: std::max(NaN,
    // diff) used to drop the NaN and hand back dtMax, so `dns tgv` marched a
    // NaN field to tEnd and exited 0. The caller's `dt > 0` check (exit 7)
    // relies on this.
    {
        const auto g = Grid::make(16, 16, 1, 2 * pi, 2 * pi, 2 * pi / 16);
        auto U = VectorField::zeros(g);
        taylorGreen(g, nu, 0.0, U);
        U.v[g.idx(5, 7, 0)] = std::nan("");
        TimeStepper ts(g, {.nu = nu, .cfl = 0.4});
        const double dt = ts.suggestDt(U, 1.0);
        std::printf("NaN-seeded suggestDt: %g\n", dt);
        if (dt > 0 && std::isfinite(dt))
        {
            std::puts("FAIL: a NaN field got a finite positive dt");
            ++fails;
        }
    }
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
