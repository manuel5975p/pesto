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
// openLat: OPEN lateral faces instead of walls (sponge-free free stream).
// Tangential ghosts keep the zero-gradient mirror; the NORMAL velocity on the
// lateral boundary faces is zero-gradient from the nearest interior face (the
// displacement/dipole field vents; relaxing it is a wall in disguise). Those
// faces carry prescribed flux at zero aperture exactly like the inlet/outlet:
// addFluxRhs adds them to the projection RHS, correctOutletFlux closes the
// global balance uInf*Ain = Qout + Qlat at the outlet plane, and the pressure
// stays Neumann/DCT. openLat == false is bit-identical to the wall tunnel.
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
        bool openLat = false; // open lateral faces (y, and z when nz > 1)
        // Potential-corrected INLET (requires openLat): the Dirichlet inlet
        // becomes u = uInf + wake source + displacement doublet (v likewise),
        // removing the finite-domain bias of pinning u = uInf a few D
        // upstream of the body. The wake-source strength Q = F_drag / uInf is
        // fed via setWakeSource (EMA-smoothed by the caller). The lateral
        // faces stay free (zero-gradient) -- they already vent adaptively.
        // pot with Q == 0 and aBody == 0 reproduces the uniform inlet.
        bool pot = false;
        double xc = 0, yc = 0, zc = 0; // body centre (world coords)
        double aBody = 0;              // displacement radius (D/2)
        // Full-width solid floor crossing the inlet/outlet planes (--body
        // floor): interior rows j <= floorRows are fully solid there -- the
        // Dirichlet inlet and convective outlet become 0 -- and row
        // floorRows+1 is the cut row, open over fraction floorFrac of its
        // face. Prescribed fluxes, the outlet tally and the global mass
        // balance are weighted accordingly (the slab is x/z-invariant, so a
        // row count + one fraction describe both planes exactly).
        // floorRows == -1 (no floor) is BIT-IDENTICAL to the historical
        // tunnel.
        int floorRows = -1;
        double floorFrac = 1.0;
    };

    static std::expected<DeviceTunnelBc, hipError_t> make(const GpuGrid&, Config);

    // cdt = uInf*dt/hx drives the outlet from `in`. cdt == 0 with `in` aliased
    // to `out` is a pure ghost refresh. refreshOnly (open laterals, after the
    // projection): keep the prescribed lateral boundary fluxes the projection
    // balanced, update only the ghost rows.
    // Pre: same grid, same size.
    // Quiet start (--ramp): retarget the freestream the inlet/outlet/mirrors
    // impose. Host-side; takes effect on the next apply.
    void setFreestream(double u) { cfg_.uInf = u; }

    void apply(const DeviceField& in, DeviceField& out, double cdt,
               bool refreshOnly = false) const;

    // Restore exact global mass balance on the open outlet plane.
    void correctOutletFlux(DeviceField&, ReduceScratch&, DeviceScalar& tmp) const;

    // Inlet/outlet faces carry prescribed flux at zero aperture, so
    // divergenceCut misses them; add the missing RHS terms. `scale` must match
    // the one passed to divergenceCut.
    void addFluxRhs(const DeviceField&, double scale, DeviceScalar& rhs) const;

    // Wake-source strength Q = F_drag / uInf for the potential-corrected
    // inlet. Pre: Q >= 0, finite. Ignored unless Config::pot.
    void setWakeSource(double Q) { Q_ = Q; }

private:
    DeviceTunnelBc(const GpuGrid& g, Config c) : g_(g), cfg_(c) {}
    GpuGrid g_;
    Config cfg_;
    double Q_ = 0;
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

    // Quiet start (--ramp): retarget the freestream the fringe relaxes to.
    void setFreestream(double u) { cfg_.uInf = u; }

    void apply(DeviceField&) const;

private:
    DeviceFringe(const GpuGrid& g, Config c) : g_(g), cfg_(c) {}
    GpuGrid g_;
    Config cfg_;
};

} // namespace dns::gpu
