// Device projection makes a random field divergence-free to fp32 FFT roundoff.
#include "gpu_test_util.hpp"

#include "dns/gpu/projection.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/operators.hpp"

#include <numbers>

int main()
{
    using namespace dns;
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    const auto g = Grid::make(32, 32, 32, 2 * pi, 2 * pi, 2 * pi);
    const auto U = test::randomField(g, 8);
    const auto gg = gpu::GpuGrid::from(g);

    auto D = *gpu::DeviceField::make(g.size());
    (void)gpu::uploadField(U, D);
    auto solver = *gpu::DevicePoissonSolver::make(gg);
    auto scratch = *gpu::DeviceProjectionScratch::make(g.size());
    auto red = *gpu::ReduceScratch::make();

    const double before = gpu::maxAbsDivergence(gg, D, red);
    gpu::project(gg, solver, gpu::Real(0.01), D, scratch);
    const double after = gpu::maxAbsDivergence(gg, D, red);

    // Relative to the pre-projection divergence: fp32 FFT leaves ~1e-6..1e-5.
    std::printf("maxDiv before %.3e after %.3e ratio %.3e\n", before, after,
                after / before);
    if (!(after < 1e-4 * before))
    {
        std::fprintf(stderr, "FAIL: divergence reduction only %.3e\n", after / before);
        return 1;
    }
    return 0;
}
