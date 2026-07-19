// 3D Taylor-Green grid convergence: Richardson order of K(T) across
// n = 16/32/64 at Re = 100 must exceed 1.7 (2nd-order space, RK3 time).
#include "dns/operators.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

namespace {

double keAtTime(int n, double nu, double tEnd)
{
    using std::numbers::pi;
    const auto g = dns::Grid::make(n, n, n, 2 * pi, 2 * pi, 2 * pi);
    auto U = dns::VectorField::zeros(g);
    dns::taylorGreen3D(g, U);
    dns::TimeStepper ts(g, {nu, 0.5});
    double t = 0.0;
    while (t < tEnd)
    {
        const double dt = ts.suggestDt(U, tEnd - t);
        ts.step(U, dt);
        t += dt;
    }
    return dns::kineticEnergy(g, U) / (g.cellVolume() * double(g.size()));
}

} // namespace

int main()
{
    const double nu = 0.01, tEnd = 1.0;
    const double k16 = keAtTime(16, nu, tEnd);
    const double k32 = keAtTime(32, nu, tEnd);
    const double k64 = keAtTime(64, nu, tEnd);

    const double d1 = std::abs(k16 - k32), d2 = std::abs(k32 - k64);
    const double order = std::log2(d1 / d2);
    std::printf("K(T): %.10e %.10e %.10e  order %.2f\n", k16, k32, k64, order);
    if (!(d2 > 0) || !(order > 1.7))
    {
        std::fprintf(stderr, "FAIL: observed order %.2f <= 1.7\n", order);
        return 1;
    }
    return 0;
}
