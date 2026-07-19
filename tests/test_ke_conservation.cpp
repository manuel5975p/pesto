// Inviscid, unforced HIT: the divergence-form MAC advection with per-stage
// projection conserves discrete kinetic energy exactly in space (Morinishi
// Div. scheme); the only drift is SSP-RK3 time error, O(dt^3) per step, i.e.
// O(dt^2) over a fixed horizon. Verify smallness AND the dt^2 convergence.
#include "dns/hit_ic.hpp"
#include "dns/operators.hpp"
#include "dns/timestepper.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

namespace {

double keDrift(const dns::Grid& g, const dns::VectorField& U0, double dt, double tEnd)
{
    auto U = U0;
    dns::TimeStepper ts(g, {0.0, 1.0});
    const double k0 = dns::kineticEnergy(g, U);
    for (double t = 0.0; t < tEnd - 0.5 * dt; t += dt) ts.step(U, dt);
    return std::abs(dns::kineticEnergy(g, U) - k0) / k0;
}

} // namespace

int main()
{
    using std::numbers::pi;
    const int n = 32;
    const auto g = dns::Grid::make(n, n, n, 2 * pi, 2 * pi, 2 * pi);
    auto U0 = dns::VectorField::zeros(g);
    dns::hitInitialCondition(g, 1.0, 42, U0);

    const double tEnd = 0.5;
    const double d1 = keDrift(g, U0, 5.0e-3, tEnd);
    const double d2 = keDrift(g, U0, 2.5e-3, tEnd);
    const double order = std::log2(d1 / d2);
    std::printf("drift(dt)=%.3e drift(dt/2)=%.3e order %.2f\n", d1, d2, order);

    bool ok = true;
    if (!(d2 < 1e-5))
    {
        std::fprintf(stderr, "FAIL: drift %.3e >= 1e-5 at dt=2.5e-3\n", d2);
        ok = false;
    }
    if (!(order > 1.5))
    {
        std::fprintf(stderr, "FAIL: drift order %.2f <= 1.5 (spatial leak?)\n", order);
        ok = false;
    }
    return ok ? 0 : 1;
}
