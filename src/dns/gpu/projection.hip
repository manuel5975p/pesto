#include "dns/gpu/projection.hpp"

#include "dns/gpu/kernels.hpp"

#include <cassert>

namespace dns::gpu {

void project(const GpuGrid& g, DevicePoissonSolver& poisson, Real dt, DeviceField& U,
             DeviceProjectionScratch& s)
{
    assert(dt > 0);
    divergence(g, U, s.div, Real(1) / dt);
    poisson.solve(s.div, s.phi);
    subtractGradient(g, s.phi, dt, U);
}

} // namespace dns::gpu
