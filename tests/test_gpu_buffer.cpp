// DeviceBuffer/DeviceField round-trip and move semantics.
#include "gpu_test_util.hpp"

#include "dns/gpu/device_buffer.hpp"

#include <numbers>
#include <utility>

int main()
{
    using namespace dns;
    test::requireDeviceOrSkip();

    const auto g = Grid::make(24, 16, 8, 1.0, 2.0, 3.0);
    const auto U0 = test::randomField(g, 1);

    auto D = gpu::DeviceField::make(g.size());
    if (!D)
    {
        std::fprintf(stderr, "FAIL: DeviceField::make: %s\n",
                     hipGetErrorString(D.error()));
        return 1;
    }
    if (gpu::uploadField(U0, *D) != hipSuccess) return 1;

    // Move must transfer ownership without copies or leaks.
    gpu::DeviceField E = std::move(*D);

    auto U1 = VectorField::zeros(g);
    if (gpu::downloadField(E, U1) != hipSuccess) return 1;

    const auto d = test::fieldDiff(U1, U0);
    // float round-trip of doubles in [-1,1]: error bounded by half ulp ~ 6e-8
    return test::checkClose("roundtrip", d.maxErr, d.scale, 1e-6) ? 0 : 1;
}
