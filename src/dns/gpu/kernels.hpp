// Host launchers for the device stencil/update kernels; mirrors dns/operators.hpp.
// GPU-only header. All launchers are asynchronous on the default stream.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"

namespace dns::gpu {

// Optional tail applied to a fused stage's three outputs, inside the stage
// kernel, in this order:
//   1. Brinkman solid-band decay (--penalize): face *= exp(-idte*(1 - lam_face)),
//      lam_face the mean of the two adjacent cell fluid fractions; a face with
//      lam_face == 1 is left EXACTLY alone (branch, not exp(0) roundoff).
//   2. Solid-face mask: face = 0 wherever its own aperture is closed.
// Each part is off when its pointer is null, and the off case costs one uniform
// branch. Folding them here removes two full read-modify-write passes over the
// three arrays the stage has just written.
//
// Pre (part 1): idte = dt/eta >= 0. Pre (part 2): the three aperture arrays are
// the ones the geometry the stage output will be projected against carries.
struct StageTail
{
    const Real* lam = nullptr;                              // 1: null = off
    Real idte = 0;                                          // 1: dt / eta
    const Real* ax = nullptr;                               // 2: null = off
    const Real* ay = nullptr;
    const Real* az = nullptr;
};

// out = scale * div(U) at cell centres (scale folds the projection 1/dt).
void divergence(const GpuGrid& g, const DeviceField& U, DeviceScalar& out, Real scale);

// U -= scale * grad(p) on the faces. Pre: p does not alias U.
void subtractGradient(const GpuGrid& g, const DeviceScalar& p, Real scale, DeviceField& U);

// out = -(U . grad)U, conservative form (same stencil as the CPU oracle).
// Pre: out does not alias U.
void convection(const GpuGrid& g, const DeviceField& U, DeviceField& out);

// out += nu * lap(U) componentwise. Pre: out does not alias U.
void addScaledLaplacian(const GpuGrid& g, Real nu, const DeviceField& U, DeviceField& out);

// out += A * (U - mean), the linear-forcing term.
void addForcing(const GpuGrid& g, Real A, Real mx, Real my, Real mz,
                const DeviceField& U, DeviceField& out);

// out = a*x + b*(z + dt*r): all three SSP-RK3 stage combinations.
void rkCombine(const GpuGrid& g, Real a, const DeviceField& x, Real b,
               const DeviceField& z, Real dt, const DeviceField& r, DeviceField& out);

// Fused RK stage: out = a*x + b*(in + dt*(conv(in) + nu*lap(in) + A*(in-mean))),
// then `tail`. One memory pass per stage.
// Pre: out does not alias in (stencil reads). out MAY alias x (the SSP-RK3
// closing stage does exactly that), which is why only the stencil inputs are
// restrict-qualified inside.
void fusedRhsStage(const GpuGrid& g, Real nu, Real A, Real mx, Real my, Real mz,
                   Real a, const DeviceField& x, Real b, const DeviceField& in,
                   Real dt, DeviceField& out, const StageTail& tail = {});

// As above but the viscous term reads diff: nu*lap(diff). Embedded-boundary
// apps pass a ghost-filled copy of in so the wall is sub-cell sharp for
// diffusion while convection keeps the masked zeros (energy stability).
// Pre: out aliases neither in nor diff; in and diff may alias each other.
void fusedRhsStage(const GpuGrid& g, Real nu, Real A, Real mx, Real my, Real mz,
                   Real a, const DeviceField& x, Real b, const DeviceField& in,
                   const DeviceField& diff, Real dt, DeviceField& out,
                   const StageTail& tail = {});

// U -= mean componentwise (pins the mean mode under linear forcing).
void subtractMean(const GpuGrid& g, Real mx, Real my, Real mz, DeviceField& U);

} // namespace dns::gpu
