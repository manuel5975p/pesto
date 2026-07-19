// 3D Taylor-Green vortex driver (Re=1600 benchmark): prints a dissipation
// time series (t, K, -dK/dt, eps_visc) for comparison against spectral data.
// Uses the GPU integrator when built with DNS_WITH_HIP and a device is present.
#include "dns/operators.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"

#if DNS_WITH_HIP
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/timestepper.hpp"
#endif

#include <charconv>
#include <cstdio>
#include <numbers>
#include <string_view>

namespace {

double parseOr(std::string_view s, double fallback)
{
    double v{};
    auto [p, ec] = std::from_chars(s.begin(), s.end(), v);
    return ec == std::errc{} && p == s.end() ? v : fallback;
}

struct Probe { double k, eps; };

// Emit one CSV row; kPrev/tPrev updated across calls.
void emit(double t, const Probe& p, double& kPrev, double& tPrev)
{
    const double dkdt = t > tPrev ? -(p.k - kPrev) / (t - tPrev) : 0.0;
    std::printf("%.6f,%.10e,%.10e,%.10e\n", t, p.k, dkdt, p.eps);
    kPrev = p.k;
    tPrev = t;
}

template <class SuggestDt, class Step, class Stats>
long integrate(double tEnd, double outEvery, SuggestDt suggestDt, Step step,
               Stats stats)
{
    double t = 0.0, nextOut = 0.0, kPrev = 0.0, tPrev = 0.0;
    long steps = 0;
    while (t < tEnd)
    {
        if (t >= nextOut)
        {
            emit(t, stats(), kPrev, tPrev);
            nextOut += outEvery;
        }
        const double dt = suggestDt(tEnd - t);
        step(dt);
        t += dt;
        ++steps;
    }
    emit(t, stats(), kPrev, tPrev);
    return steps;
}

} // namespace

int main(int argc, char** argv)
{
    using std::numbers::pi;
    const int n = argc > 1 ? int(parseOr(argv[1], 64)) : 64;
    const double re = argc > 2 ? parseOr(argv[2], 1600.0) : 1600.0;
    const double tEnd = argc > 3 ? parseOr(argv[3], 20.0) : 20.0;
    const double cfl = 0.9, outEvery = 0.1;

    const auto g = dns::Grid::make(n, n, n, 2 * pi, 2 * pi, 2 * pi);
    const double vol = g.cellVolume() * double(g.size());
    const dns::StepParams sp{1.0 / re, cfl};
    auto U = dns::VectorField::zeros(g);
    dns::taylorGreen3D(g, U);

    long steps = 0;
#if DNS_WITH_HIP
    if (dns::gpu::deviceAvailable())
    {
        auto ts = dns::gpu::TimeStepper::make(g, sp);
        auto D = dns::gpu::DeviceField::make(g.size());
        if (!ts || !D)
        {
            std::fprintf(stderr, "GPU init failed\n");
            return 1;
        }
        (void)dns::gpu::uploadField(U, *D);
        const auto gg = ts->grid();
        auto& red = ts->reduceScratch();

        std::printf("# 3D TGV n=%d Re=%g cfl=%g backend=hip\n", n, re, cfl);
        std::printf("t,K,dKdt,epsVisc\n");
        steps = integrate(
            tEnd, outEvery, [&](double dtMax) { return ts->suggestDt(*D, dtMax); },
            [&](double dt) { ts->step(*D, dt); },
            [&] {
                return Probe{dns::gpu::kineticEnergy(gg, *D, red) / vol,
                             sp.nu * dns::gpu::velocityGradientSq(gg, *D, red)};
            });
        std::fprintf(stderr, "done: %ld steps (hip)\n", steps);
        return 0;
    }
#endif

    dns::TimeStepper ts(g, sp);
    std::printf("# 3D TGV n=%d Re=%g cfl=%g backend=cpu\n", n, re, cfl);
    std::printf("t,K,dKdt,epsVisc\n");
    steps = integrate(
        tEnd, outEvery, [&](double dtMax) { return ts.suggestDt(U, dtMax); },
        [&](double dt) { ts.step(U, dt); },
        [&] {
            return Probe{dns::kineticEnergy(g, U) / vol,
                         sp.nu * dns::velocityGradientSq(g, U)};
        });
    std::fprintf(stderr, "done: %ld steps (cpu)\n", steps);
    return 0;
}
