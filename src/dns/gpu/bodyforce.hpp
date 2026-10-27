// Discrete surface integrals over an embedded body inside an index box.
// GPU-only header.
#pragma once

#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/reductions.hpp"

#include <array>
#include <expected>

namespace dns::gpu {

// Index bounding box of the body plus a margin, so the tallies scan O(body)
// not O(grid). Clamped strictly inside the interior, so lateral boundary faces
// never count.
struct BodyBox
{
    int i0, i1, j0, j1, k0, k1;
};

// r is the body radius; cz is ignored when the grid is native 2D.
// Pre: h > 0 on every axis.
BodyBox bodyBox(const GpuGrid&, double cx, double cy, double cz, double r,
                int margin = 2);

// Same clamping from an explicit world box {xlo,ylo,zlo,xhi,yhi,zhi} -- the
// per-object scene tallies, whose extents are not cubes.
BodyBox bodyBox(const GpuGrid&, const std::array<double, 6>& worldBox,
                int margin = 2);

// Index-box overlap test: two overlapping tally boxes would each count the
// shared surface, so the caller must not emit per-object series for them.
bool overlaps(const BodyBox&, const BodyBox&);

class DeviceBodyForce
{
public:
    // cd, cl: total force / qRef. cdf: the FRICTION part only, and always
    // nu-only -- the SGS eddy viscosity never enters cdf (project LES stance:
    // the cdf column is a molecular-viscosity diagnostic). cs: z-component,
    // exactly 0 in native 2D.
    struct Coeffs
    {
        double cd, cl, cs, cdf;
    };

    static std::expected<DeviceBodyForce, hipError_t> make(const GpuGrid&, BodyBox);

    // Pressure: F = -sum phi_c (a_xp - a_c) A_face -- the exact discrete
    //   aperture identity for the surface integral of phi.
    // Friction: F = nu sum over links from an open face to a CLOSED neighbour
    //   of (q - q_ghost) A/d -- the Laplacian flux the momentum update applies,
    //   with the closed neighbour holding the ghost-fill mirror value.
    //   ALL THREE link directions are tallied unconditionally. At nz == 1 the
    //   z neighbours wrap to the cell itself, whose aperture passed the
    //   > openTol guard that gated the call, so the z links contribute exactly
    //   zero: the 3D form IS the 2D form, with no mode switch to get wrong.
    // ghosted = the ghost-filled view the viscous term actually saw, NOT the
    //   state (feeding mirror values to convection injects energy).
    // nut != nullptr adds the SGS wall drain (all three link directions, the
    // eddy-diffusion substep's own weights) to cd/cl/cs but NOT to cdf.
    // Pre: geometry sanitized; phi is the FIRST RK stage's pressure.
    Coeffs measure(const DeviceCutGeometry&, const DeviceScalar& phi,
                   const DeviceField& ghosted, double nu, double qRef,
                   const DeviceScalar* nut, ReduceScratch&);

private:
    DeviceBodyForce(const GpuGrid& g, BodyBox b, DeviceScalar fx, DeviceScalar fy,
                    DeviceScalar fz)
        : g_(g), box_(b), fx_(std::move(fx)), fy_(std::move(fy)), fz_(std::move(fz))
    {}

    GpuGrid g_;
    BodyBox box_;
    DeviceScalar fx_, fy_, fz_;
};

} // namespace dns::gpu
