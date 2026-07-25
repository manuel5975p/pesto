// Device SSP-RK3 fractional-step integrator; mirrors dns::TimeStepper with all
// fields resident on the GPU. GPU-only header.
#pragma once

#include "dns/forcing.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/poisson.hpp"
#include "dns/gpu/projection.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/timestepper.hpp"

#include <expected>

namespace dns::gpu {

class TimeStepper
{
public:
    // Allocates all device scratch and FFT plans up front.
    static std::expected<TimeStepper, hipError_t> make(const Grid& g, StepParams p);

    // Forcing coefficients used by rhs until the next call; {} disables.
    void setForcing(ForcingState f) { forcing_ = f; }

    // Stable dt from convective/diffusive/forcing limits (device reductions).
    double suggestDt(const DeviceField& U, double dtMax);

    // Advance U by dt (SSP-RK3, projection after each stage). Pre: dt > 0.
    void step(DeviceField& U, double dt);

    // Shared handle for app-side statistics on the same device state.
    const GpuGrid& grid() const { return g_; }

private:
    TimeStepper(GpuGrid g, StepParams p, DevicePoissonSolver poisson,
                DeviceProjectionScratch scratch, DeviceField s1, DeviceField s2,
                ReduceScratch red);

    void stage(Real a, const DeviceField& x, Real b, const DeviceField& in, Real dt,
               DeviceField& out);

    GpuGrid g_;
    StepParams p_;
    ForcingState forcing_{};
    DevicePoissonSolver poisson_;
    DeviceProjectionScratch scratch_;
    DeviceField s1_, s2_; // stage ping-pong (fused kernel cannot write in place)
    ReduceScratch red_;
};

} // namespace dns::gpu
