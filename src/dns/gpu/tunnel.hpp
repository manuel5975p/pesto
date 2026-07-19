// Incompressible far-field boundary treatments on the ghost ring.
// GPU-only header.
#pragma once

#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/reductions.hpp"

#include <expected>

namespace dns::gpu {

// Confined channel: Dirichlet inflow, convective outflow with exact global
// mass balance, FREE-SLIP (Neumann) lateral walls.
//   inlet  (i<=1)    u = uInf; v,w antimirrored so they vanish on x = hx
//   outlet (i==nx-1) convective advection out of the domain
//   y,z rings        tangential components mirrored (the wall generates NO
//                    vorticity), normal component 0 on the wall face and
//                    antimirrored in the ghost. Pressure is already Neumann
//                    there (closed apertures + DCT).
// A Dirichlet u = uInf far field clamps the velocity against the sheared
// interior flow and paints a vortex sheet along the wall. Never reintroduce it.
//
// Ghost corners are mirrored DIAGONALLY: an edge cell takes the single
// (jm,km) value, and the k == 0 w row is explicitly antimirrored. At nz == 1
// km == k identically, so this reduces exactly to the y-only treatment -- one
// kernel covers native 2D and full 3D with no mode switch. (Two variants of
// this kernel existed in the tree; the sequential y-then-z form mirrored an
// edge ghost across only one of the two walls, which is wrong for nz > 1.)
class DeviceTunnelBc
{
public:
    struct Config
    {
        double uInf = 1.0;
    };

    static std::expected<DeviceTunnelBc, hipError_t> make(const GpuGrid&, Config);

    // cdt = uInf*dt/hx drives the outlet from `in`. cdt == 0 with `in` aliased
    // to `out` is a pure ghost refresh.
    // Pre: same grid, same size.
    void apply(const DeviceField& in, DeviceField& out, double cdt) const;

    // Restore exact global mass balance on the open outlet plane.
    void correctOutletFlux(DeviceField&, ReduceScratch&, DeviceScalar& tmp) const;

    // Inlet/outlet faces carry prescribed flux at zero aperture, so
    // divergenceCut misses them; add the missing RHS terms. `scale` must match
    // the one passed to divergenceCut.
    void addFluxRhs(const DeviceField&, double scale, DeviceScalar& rhs) const;

private:
    DeviceTunnelBc(const GpuGrid& g, Config c) : g_(g), cfg_(c) {}
    GpuGrid g_;
    Config cfg_;
};

// Cosine-ramped fringe relaxing u -> (uInf,0,0) on the domain boundaries of a
// fully periodic box (the FFT-Poisson far field).
//   sig = sigMax * (1 - cos(pi r)) / 2, r = max over banded axes of the
//   normalised band depth (0 at the interior edge, 1 on the boundary).
// axes == 2 leaves z plain periodic: for a z-invariant (quasi-2D) run a z band
// would sponge the whole wake. axes == 3 bands all three.
// Pre: cells >= 1; axes == 2, or axes == 3 with nz >= 2*cells + 2.
class DeviceFringe
{
public:
    struct Config
    {
        double uInf = 1.0, sigMax = 0.5;
        int cells = 0;
        int axes = 3;
    };

    static std::expected<DeviceFringe, hipError_t> make(const GpuGrid&, Config);

    void apply(DeviceField&) const;

private:
    DeviceFringe(const GpuGrid& g, Config c) : g_(g), cfg_(c) {}
    GpuGrid g_;
    Config cfg_;
};

} // namespace dns::gpu
