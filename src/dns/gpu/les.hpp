// Optional incompressible Smagorinsky LES. nut = (Cs*Delta)^2 |S| scaled by
// the fluid fraction lambda; the eddy-diffusion substep adds
// b*dt*div(nut grad u) per RK stage in conservative flux form. Full 3D strain
// and fluxes; at nz == 1 every z term is exactly zero and the model is
// bit-identical to the historical 2D kernel (Delta^2 = hx*hy there,
// (hx hy hz)^(2/3) in 3D).
//
// NOTE gradient form div(nut grad u), NOT the symmetric stress div(2 nut S).
// The difference is dj(nut di uj): zero for constant nut with div u = 0,
// nonzero for varying nut. Still flux-conservative and dissipative
// (nut |grad u|^2 >= 0); Cs is nominal.
//
// Output is a FILTERED field: statistics depend on grid AND Cs by construction,
// not as a discretisation error. Label such runs LES and state Cs; never
// present them as converged predictions. 2D caveat: Smagorinsky models a 3D
// forward energy cascade, while 2D cascades energy up and enstrophy down, so a
// 2D LES run is a stabilised/regularised 2D Euler flow -- qualitative only.
//
// Cut-cell caveat: |S| uses central differences that reach into the solid where
// u is masked to 0, so raw nut PEAKS at the wall (no van Driest damping) and
// the flux form ignores the apertures ax/ay (staircase body). Mitigation: nut
// is scaled by the fluid fraction lambda -- zero in solid, damped in cut cells.
// Coarse, not exact.
//
// GPU-only header.
#pragma once

#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/index.hpp"

#include <expected>

namespace dns::gpu {

class DeviceSmagorinsky
{
public:
    // Pre: cs > 0.
    static std::expected<DeviceSmagorinsky, hipError_t> make(const GpuGrid&,
                                                             double cs);

    // nut at cell centres from the MAC field. U must be the GHOST-FILLED view
    // the viscous term reads, matching the field addEddyDiffusion is given.
    // The domain-boundary ring gets nut = 0: the wrap stencil would otherwise
    // read across the box, and zero is the correct no-eddy-flux condition.
    void computeNut(const DeviceField& U, const DeviceCutGeometry&);

    // out += scale * div(nut grad U), conservative flux form with face- and
    // edge-averaged nut. Explicitly stable: dt*nut/h^2 ~ Cs^2 * CFL << 1.
    void addEddyDiffusion(const DeviceField& U, Real scale, DeviceField& out) const;

    // The SGS viscosity field, for the body SGS-friction tally. Never enters
    // the cdf column.
    const DeviceScalar& nut() const { return nut_; }

private:
    DeviceSmagorinsky(const GpuGrid& g, Real cs2d2, DeviceScalar nut)
        : g_(g), cs2d2_(cs2d2), nut_(std::move(nut))
    {}

    GpuGrid g_;
    Real cs2d2_;
    DeviceScalar nut_;
};

} // namespace dns::gpu
