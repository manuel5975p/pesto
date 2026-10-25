#include "dns/gpu/timestepper.hpp"

#include "dns/gpu/kernels.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns::gpu {

std::expected<TimeStepper, hipError_t> TimeStepper::make(const Grid& g, StepParams p)
{
    assert(p.nu >= 0);
    assert(p.cfl > 0 && p.cfl <= 1);
    const GpuGrid gg = GpuGrid::from(g);

    auto poisson = DevicePoissonSolver::make(gg);
    if (!poisson) return std::unexpected(poisson.error());
    auto scratch = DeviceProjectionScratch::make(gg.size());
    if (!scratch) return std::unexpected(scratch.error());
    auto s1 = DeviceField::make(gg.size());
    if (!s1) return std::unexpected(s1.error());
    auto s2 = DeviceField::make(gg.size());
    if (!s2) return std::unexpected(s2.error());
    auto red = ReduceScratch::make();
    if (!red) return std::unexpected(red.error());

    return TimeStepper(gg, p, std::move(*poisson), std::move(*scratch),
                       std::move(*s1), std::move(*s2), std::move(*red));
}

TimeStepper::TimeStepper(GpuGrid g, StepParams p, DevicePoissonSolver poisson,
                         DeviceProjectionScratch scratch, DeviceField s1,
                         DeviceField s2, ReduceScratch red)
    : g_(g), p_(p), poisson_(std::move(poisson)), scratch_(std::move(scratch)),
      s1_(std::move(s1)), s2_(std::move(s2)), red_(std::move(red))
{
}

double TimeStepper::suggestDt(const DeviceField& U, double dtMax)
{
    const double conv = maxCourantPerDt(g_, U, red_);
    // A NaN Courant number is a diverged field: max(NaN, diff) would drop it
    // and return dtMax. NaN out instead, so the caller's dt check fails.
    if (!(conv >= 0)) return std::numeric_limits<double>::quiet_NaN();
    const double hmin = std::min({double(g_.hx), double(g_.hy), double(g_.hz)});
    const double diff = p_.nu > 0 ? 6.0 * p_.nu / (hmin * hmin) : 0.0;
    const double lim = std::max(conv, diff);
    double dt = lim > 0 ? std::min(dtMax, p_.cfl / lim) : dtMax;
    if (forcing_.A > 0)
        dt = std::min(dt, 0.2 / forcing_.A);
    return dt;
}

// One fused kernel per stage; s1_/s2_ ping-pong because the stage output may
// not alias the stencil input.
void TimeStepper::stage(Real a, const DeviceField& x, Real b, const DeviceField& in,
                        Real dt, DeviceField& out)
{
    fusedRhsStage(g_, Real(p_.nu), Real(forcing_.A), Real(forcing_.mean.x),
                  Real(forcing_.mean.y), Real(forcing_.mean.z), a, x, b, in, dt, out);
}

void TimeStepper::step(DeviceField& U, double dt)
{
    assert(dt > 0);
    const Real h = Real(dt);

    // stage 1: s1 = U + dt L(U)
    stage(Real(0), U, Real(1), U, h, s1_);
    project(g_, poisson_, h, s1_, scratch_);

    // stage 2: s2 = 3/4 U + 1/4 (s1 + dt L(s1))
    stage(Real(0.75), U, Real(0.25), s1_, h, s2_);
    project(g_, poisson_, h, s2_, scratch_);

    // stage 3: U = 1/3 U + 2/3 (s2 + dt L(s2))
    // 1 - Real(2/3) is exact, so the weights sum to exactly 1 in FP32.
    stage(Real(1) - Real(2.0 / 3.0), U, Real(2.0 / 3.0), s2_, h, U);
    project(g_, poisson_, h, U, scratch_);
}

} // namespace dns::gpu
