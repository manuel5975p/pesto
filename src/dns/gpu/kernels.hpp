// Host launchers for the device stencil/update kernels; mirrors dns/operators.hpp.
// GPU-only header. All launchers are asynchronous on the default stream.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"

namespace dns::gpu {

// out = scale * div(U) at cell centres (scale folds the projection 1/dt).
void divergence(const GpuGrid& g, const DeviceField& U, DeviceScalar& out, Real scale);

// U -= scale * grad(p) on the faces.
void subtractGradient(const GpuGrid& g, const DeviceScalar& p, Real scale, DeviceField& U);

// out = -(U . grad)U, conservative form (same stencil as the CPU oracle).
void convection(const GpuGrid& g, const DeviceField& U, DeviceField& out);

// out += nu * lap(U) componentwise.
void addScaledLaplacian(const GpuGrid& g, Real nu, const DeviceField& U, DeviceField& out);

// out += A * (U - mean), the linear-forcing term.
void addForcing(const GpuGrid& g, Real A, Real mx, Real my, Real mz,
                const DeviceField& U, DeviceField& out);

// out = a*x + b*(z + dt*r): all three SSP-RK3 stage combinations.
void rkCombine(const GpuGrid& g, Real a, const DeviceField& x, Real b,
               const DeviceField& z, Real dt, const DeviceField& r, DeviceField& out);

// Fused RK stage: out = a*x + b*(in + dt*(conv(in) + nu*lap(in) + A*(in-mean))).
// One memory pass per stage. Pre: out does not alias in (stencil reads).
void fusedRhsStage(const GpuGrid& g, Real nu, Real A, Real mx, Real my, Real mz,
                   Real a, const DeviceField& x, Real b, const DeviceField& in,
                   Real dt, DeviceField& out);

// As above but the viscous term reads diff: nu*lap(diff). Embedded-boundary
// apps pass a ghost-filled copy of in so the wall is sub-cell sharp for
// diffusion while convection keeps the masked zeros (energy stability).
// Pre: out aliases neither in nor diff.
void fusedRhsStage(const GpuGrid& g, Real nu, Real A, Real mx, Real my, Real mz,
                   Real a, const DeviceField& x, Real b, const DeviceField& in,
                   const DeviceField& diff, Real dt, DeviceField& out);

// U -= mean componentwise (pins the mean mode under linear forcing).
void subtractMean(const GpuGrid& g, Real mx, Real my, Real mz, DeviceField& U);

} // namespace dns::gpu
