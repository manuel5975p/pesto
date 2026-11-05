// dns bench.
#include "dns/drive/internal.hpp"

#include "dns/config.hpp"
#include "dns/flow_ic.hpp"
#include "dns/taylor_green.hpp"
#include "dns/transport.hpp"

#if DNS_WITH_GPU
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/timestepper.hpp"
#endif

#include <chrono>
#include <cstdio>
#include <numbers>
#include <optional>
#include <string>

namespace dns::drive {

using std::numbers::pi;
#if DNS_WITH_GPU
namespace {

// --- bench -----------------------------------------------------------------

// Device memory this process holds, from the free-memory delta across the
// allocation. hipMemGetInfo's absolute "in use" also counts the display server
// and every other process on the card, which made the reported figure both
// wrong and machine-dependent.
std::optional<double> benchVramGiB(std::size_t freeBefore)
{
    std::size_t freeB = 0, totalB = 0;
    if (hipMemGetInfo(&freeB, &totalB) != hipSuccess) return std::nullopt;
    return freeBefore > freeB ? double(freeBefore - freeB) / (1 << 30) : 0.0;
}

// ms per call of `step`, timed over `steps` calls after 3 warm-up calls, each
// phase closed by a checked synchronization; or the device failure that
// invalidated the measurement.
template <class Step>
std::expected<double, std::string> benchTime(int steps, Step&& step)
{
    for (int s = 0; s < 3; ++s) step();   // warm-up (incl. FFT tuning)
    if (auto e = hipDeviceSynchronize(); e != hipSuccess)
        return std::unexpected(std::string("warm-up: ") + hipGetErrorString(e));
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) step();
    if (auto e = hipDeviceSynchronize(); e != hipSuccess)
        return std::unexpected(std::string("timed steps: ") + hipGetErrorString(e));
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count()
                      / steps;
    if (auto e = hipGetLastError(); e != hipSuccess)
        return std::unexpected(std::string("launch: ") + hipGetErrorString(e));
    return ms;
}

void benchReport(const Grid& g, const char* what, double ms, double vramGiB)
{
    std::printf("n=%d  %s  %.2f ms/step  (%.1f Mcells, %.1f Mcell-updates/s, "
                "%.2f GiB VRAM)\n",
                g.nx, what, ms, g.size() / 1e6, g.size() / (ms * 1e3), vramGiB);
}
} // namespace

#endif

int runBench(const Case& c)
{
    if (!haveGpu(c)) return die(exitNoGpu, "no GPU device (dns bench is GPU-only)");
#if DNS_WITH_GPU
    const auto& g = c.grid;
    auto U = VectorField::zeros(g);
    taylorGreen3D(g, U);
    const double dt = 0.2 * (2 * pi / g.nx); // fixed benchmark dt, CFL-ish
    std::size_t freeBefore = 0, totalB = 0;
    if (hipMemGetInfo(&freeBefore, &totalB) != hipSuccess)
        return die(exitResource, "GPU memory query failed");

    if (c.physics == Physics::Compressible)
    {
        constexpr double gamma = kGammaAir;
        auto D = gpu::DeviceComp::make(g, gamma, c.nu, c.cs, c.sgsModel);
        auto S = gpu::DeviceCompState::make(g.size());
        if (!D || !S) return die(exitResource, "GPU init failed");
        if (S->upload(benchCompState(g, U, gamma, c.mach)) != hipSuccess)
            return die(exitResource, "GPU upload failed");
        const auto vram = benchVramGiB(freeBefore);
        if (!vram) return die(exitResource, "GPU memory query failed");
        const auto ms =
            benchTime(c.steps, [&] { D->step(*S, gpu::Real(dt)); });
        if (!ms) return die(exitResource, "GPU failure in bench " + ms.error());
        benchReport(g, c.cs > 0 ? "compressible LES" : "compressible", *ms,
                    *vram);
        return 0;
    }

    auto ts = gpu::TimeStepper::make(g, {c.nu, 0.9});
    auto D = gpu::DeviceField::make(g.size());
    if (!ts || !D) return die(exitResource, "GPU init failed");
    if (gpu::uploadField(U, *D) != hipSuccess)
        return die(exitResource, "GPU upload failed");
    ts->setForcing({0.1, {0, 0, 0}}); // exercise the forcing path too
    const auto vram = benchVramGiB(freeBefore);
    if (!vram) return die(exitResource, "GPU memory query failed");
    const auto ms = benchTime(c.steps, [&] { ts->step(*D, dt); });
    if (!ms) return die(exitResource, "GPU failure in bench " + ms.error());
    benchReport(g, "incompressible", *ms, *vram);
    return 0;
#else
    return exitNoGpu;
#endif
}


} // namespace dns::drive
