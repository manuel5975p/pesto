// GPU step-time benchmark: forced-HIT-like settings at a given N, reports
// ms/step and VRAM. Usage: bench [n] [steps]
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/timestepper.hpp"
#include "dns/taylor_green.hpp"

#include <charconv>
#include <chrono>
#include <cstdio>
#include <numbers>
#include <string_view>

int main(int argc, char** argv)
{
    using namespace dns;
    using std::numbers::pi;

    if (!gpu::deviceAvailable())
    {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }
    auto parse = [](const char* s, int fallback) {
        int v{};
        const std::string_view sv(s);
        auto [p, ec] = std::from_chars(sv.begin(), sv.end(), v);
        return ec == std::errc{} && p == sv.end() ? v : fallback;
    };
    const int n = argc > 1 ? parse(argv[1], 256) : 256;
    const int steps = argc > 2 ? parse(argv[2], 20) : 20;

    const auto g = Grid::make(n, n, n, 2 * pi, 2 * pi, 2 * pi);
    auto U = VectorField::zeros(g);
    taylorGreen3D(g, U);

    auto ts = gpu::TimeStepper::make(g, {1e-3, 0.9});
    if (!ts)
    {
        std::fprintf(stderr, "TimeStepper::make: %s\n", hipGetErrorString(ts.error()));
        return 1;
    }
    auto D = gpu::DeviceField::make(g.size());
    if (!D)
    {
        std::fprintf(stderr, "DeviceField::make: %s\n", hipGetErrorString(D.error()));
        return 1;
    }
    (void)gpu::uploadField(U, *D);
    ts->setForcing({0.1, {0, 0, 0}}); // exercise the forcing path too

    const double dt = 0.2 * (2 * pi / n); // fixed benchmark dt, CFL-ish
    for (int s = 0; s < 3; ++s) ts->step(*D, dt); // warm-up (incl. FFT tuning)
    (void)hipDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) ts->step(*D, dt);
    (void)hipDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / steps;

    std::size_t freeB = 0, totalB = 0;
    (void)hipMemGetInfo(&freeB, &totalB);
    const double usedGiB = double(totalB - freeB) / (1 << 30);

    std::printf("n=%d  %.2f ms/step  (%.1f Mcells, %.2f GiB VRAM in use)\n", n, ms,
                g.size() / 1e6, usedGiB);
    return 0;
}
