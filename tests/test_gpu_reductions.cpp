// FP64 device reductions vs the CPU double oracle (tight rtol: both fp64-accumulate;
// only the fp32 narrowing of the inputs differs).
#include "gpu_test_util.hpp"

#include "dns/gpu/reductions.hpp"
#include "dns/operators.hpp"

#include <numbers>

namespace {

bool close(const char* what, double got, double ref, double rtol)
{
    const double err = std::abs(got - ref);
    const double scale = std::max(std::abs(ref), 1e-300);
    std::printf("%-18s gpu %.10e cpu %.10e rel %.3e %s\n", what, got, ref,
                err / scale, err <= rtol * scale ? "ok" : "FAIL");
    return err <= rtol * scale;
}

} // namespace

int main()
{
    using namespace dns;
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    const auto g = Grid::make(48, 32, 24, 2 * pi, 1.5 * pi, 1.0 * pi);
    auto U = test::randomField(g, 5);
    // Nonzero mean so meanVelocity/fluctuationEnergy are exercised properly.
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] += 0.3;
        U.v[c] -= 0.1;
        U.w[c] += 0.05;
    }

    const auto gg = gpu::GpuGrid::from(g);
    auto D = *gpu::DeviceField::make(g.size());
    (void)gpu::uploadField(U, D);
    auto red = *gpu::ReduceScratch::make();

    // fp32 inputs => ~1e-7 per-element error; fp64 sums keep it there.
    const double rtol = 1e-5;
    bool ok = true;
    ok &= close("kineticEnergy", gpu::kineticEnergy(gg, D, red), kineticEnergy(g, U), rtol);
    const auto mg = gpu::meanVelocity(gg, D, red);
    const auto mc = meanVelocity(g, U);
    ok &= close("mean.x", mg.x, mc.x, 1e-4);
    ok &= close("mean.y", mg.y, mc.y, 1e-4);
    ok &= close("mean.z", mg.z, mc.z, 1e-4);
    ok &= close("fluctuationEnergy", gpu::fluctuationEnergy(gg, D, red),
                fluctuationEnergy(g, U), rtol);
    ok &= close("maxCourantPerDt", gpu::maxCourantPerDt(gg, D, red),
                maxCourantPerDt(g, U), 1e-5);
    ok &= close("velocityGradientSq", gpu::velocityGradientSq(gg, D, red),
                velocityGradientSq(g, U), 1e-4);

    // maxAbsDivergence of a random field: compare against CPU (absolute-ish).
    ok &= close("maxAbsDivergence", gpu::maxAbsDivergence(gg, D, red),
                maxAbsDivergence(g, U), 1e-4);

    return ok ? 0 : 1;
}
