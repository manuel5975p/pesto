// FP64 device reductions vs the CPU double oracle (tight rtol: both fp64-accumulate;
// only the fp32 narrowing of the inputs differs).
#include "gpu_test_util.hpp"

#include "dns/gpu/internal/run_loop.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/operators.hpp"

#include <limits>
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

    // A single NaN anywhere must reach a max-reduction as +inf, never be
    // dropped by a `>` that is false against NaN -- at the first cell, the
    // middle and the last alike (the reduction tree places them differently).
    for (const std::size_t at : {std::size_t(0), g.size() / 2, g.size() - 1})
    {
        auto Un = U;
        Un.u[at] = std::numeric_limits<double>::quiet_NaN();
        if (gpu::uploadField(Un, D) != hipSuccess) return 1;
        const double cpd = gpu::maxCourantPerDt(gg, D, red);
        const double div = gpu::maxAbsDivergence(gg, D, red);
        const double mab = gpu::maxAbsScalar(D.u, red);
        const bool inf = std::isinf(cpd) && std::isinf(div) && std::isinf(mab);
        std::printf("NaN at %zu: courant %g div %g maxabs %g %s\n", at, cpd, div,
                    mab, inf ? "ok" : "FAIL");
        ok &= inf;
        ok &= std::isinf(maxCourantPerDt(g, Un));   // the CPU oracle too
    }

    // deviceHealthy: a provoked runtime error is reported once, and the next
    // check is clean (hipGetLastError clears it).
    {
        // A null destination is rejected up front (hipErrorInvalidValue),
        // before any transfer -- it cannot fault the device context.
        const hipError_t provoked =
            hipMemcpy(nullptr, red.out.data(), sizeof(double),
                      hipMemcpyDeviceToHost);
        std::printf("provoked hipMemcpy: %s\n", hipGetErrorString(provoked));
        const auto bad = gpu::internal::deviceHealthy("the test");
        std::printf("provoked: %s\n", bad ? "healthy (FAIL)" : bad.error().c_str());
        ok &= !bad.has_value();
        const auto good = gpu::internal::deviceHealthy("the test");
        std::printf("next check: %s\n", good ? "healthy" : good.error().c_str());
        ok &= good.has_value();
        // A failure recorded in the scratch is reported and then cleared.
        red.err = hipErrorInvalidValue;
        ok &= !gpu::internal::deviceHealthy("the test", &red).has_value();
        ok &= red.err == hipSuccess
              && gpu::internal::deviceHealthy("the test", &red).has_value();
    }

    return ok ? 0 : 1;
}
