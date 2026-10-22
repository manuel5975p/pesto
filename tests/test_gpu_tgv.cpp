// End-to-end integrator checks:
//  (a) 3D trajectory parity: GPU fp32 vs CPU double over 20 identical steps.
//  (b) 2D Taylor-Green on the GPU converges at order > 1.7 vs the analytic field.
#include "gpu_test_util.hpp"

#include "dns/gpu/runner_periodic.hpp"
#include "dns/gpu/timestepper.hpp"
#include "dns/operators.hpp"
#include "dns/taylor_green.hpp"
#include "dns/timestepper.hpp"

#include <numbers>

namespace {

using namespace dns;
using std::numbers::pi;

bool trajectoryParity()
{
    const auto g = Grid::make(16, 16, 16, 2 * pi, 2 * pi, 2 * pi);
    auto Uc = VectorField::zeros(g);
    taylorGreen3D(g, Uc);

    const StepParams sp{1.0 / 400.0, 0.5};
    const double dt = 0.02;

    auto D = *gpu::DeviceField::make(g.size());
    (void)gpu::uploadField(Uc, D);
    auto gts = *gpu::TimeStepper::make(g, sp);

    TimeStepper cts(g, sp);
    for (int s = 0; s < 20; ++s)
    {
        cts.step(Uc, dt);
        gts.step(D, dt);
    }
    auto Ug = VectorField::zeros(g);
    (void)gpu::downloadField(D, Ug);
    const auto d = test::fieldDiff(Ug, Uc);
    return test::checkClose("3D parity", d.maxErr, d.scale, 2e-3);
}

double tgv2dError(int n, double nu, double tEnd)
{
    // nz = 4 (z-invariant solution): rocFFT 3D plans with a unit dimension are
    // not worth relying on, and this also exercises the z index paths.
    const auto g = Grid::make(n, n, 4, 2 * pi, 2 * pi, 8 * pi / n);
    auto U = VectorField::zeros(g);
    taylorGreen(g, nu, 0.0, U);
    auto D = *gpu::DeviceField::make(g.size());
    (void)gpu::uploadField(U, D);
    auto ts = *gpu::TimeStepper::make(g, {nu, 0.4});

    double t = 0.0;
    while (t < tEnd)
    {
        const double dt = ts.suggestDt(D, tEnd - t);
        ts.step(D, dt);
        t += dt;
    }
    (void)gpu::downloadField(D, U);
    return taylorGreenError(g, nu, t, U);
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    bool ok = trajectoryParity();

    const double nu = 0.05, tEnd = 0.5;
    const double e32 = tgv2dError(32, nu, tEnd);
    const double e64 = tgv2dError(64, nu, tEnd);
    const double order = std::log2(e32 / e64);
    std::printf("2D TGV err32 %.3e err64 %.3e order %.2f\n", e32, e64, order);
    if (!(e64 < 2e-3) || !(order > 1.7))
    {
        std::fprintf(stderr, "FAIL: 2D TGV e64=%.3e order=%.2f\n", e64, order);
        ok = false;
    }

    // A NaN-seeded IC is a numerical failure (exit 7) of the 3D runner, not a
    // NaN marched to tEnd: the Courant max-reduction reads the NaN as +inf, so
    // dt is 0 and the runner's dt check stops it.
    {
        const auto g = Grid::make(16, 16, 16, 2 * pi, 2 * pi, 2 * pi);
        auto U = VectorField::zeros(g);
        taylorGreen3D(g, U);
        U.w[g.idx(3, 9, 11)] = std::nan("");
        gpu::TgvRun tr;
        tr.nu = 1.0 / 400.0;
        tr.tEnd = 0.2;
        tr.volume = std::pow(2 * pi, 3);
        const auto st = gpu::runTgv3d(g, U, tr, nullptr);
        std::printf("NaN-seeded runTgv3d: %s (code %d)\n",
                    st ? "ran to tEnd" : st.error().message.c_str(),
                    st ? 0 : st.error().code);
        if (st || st.error().code != 7)
        {
            std::fprintf(stderr, "FAIL: NaN IC did not stop with exit 7\n");
            ok = false;
        }
    }
    return ok ? 0 : 1;
}
