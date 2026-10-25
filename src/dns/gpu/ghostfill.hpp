// Sharp (sub-cell) no-slip momentum wall: mirror ghost values in solid faces.
// maskSolidFaces leaves hard zeros in every closed face, which the momentum
// stencils read as the wall -- a staircase pinned to grid lines, O(h). This
// module instead fills each solid face that a fluid stencil can read with
//   q_ghost = (sdf / d) q(image),   image = wall foot + d * n_wall,
// q(image) by the trilinear interpolant over a cell whose weighted corners are
// ALL open faces of the same component; d = |sdf| (the mirror) where that cell
// is clean, else the smallest d (1/16-cell steps, up to a cell diagonal past
// the mirror) where it is. The profile linear along the normal then vanishes
// at the true boundary and the image is exact for linear fields: the momentum
// wall becomes sub-cell accurate. Ghost faces all have zero aperture,
// hence the apertured projection/continuity never sees their values; only the
// advection/diffusion stencils do. GPU-only header.
#pragma once

#include "dns/cutcell.hpp"
#include "dns/gpu/device_buffer.hpp"

#include <expected>
#include <functional>

namespace dns::gpu {

// Signed distance to the body in world coords: negative inside the solid,
// approximately a true distance near the wall (|grad sdf| ~ 1 there).
// Host-side, build time only.
using Sdf = std::function<double(double x, double y, double z)>;

// Precomputed mirror-interpolation table, resident on device.
class DeviceGhostFill
{
public:
    // Build from the FINAL geometry (after sanitizeCut / closeBoxBoundary).
    // Ghost faces: aperture closed, sdf(center) < 0, and an open same-component
    // face within the 3x3x3 index neighborhood (everything a stencil reads).
    // Box-ring faces (sdf >= 0) are left to the app's BC kernels.
    // Pre: body at least ~2 cells clear of any closed domain ring.
    static std::expected<DeviceGhostFill, hipError_t>
    make(const Grid& g, const CutGeometry& cut, const Sdf& sdf);

    DeviceGhostFill(DeviceGhostFill&&) noexcept = default;
    DeviceGhostFill& operator=(DeviceGhostFill&&) noexcept = default;

    // Overwrite every ghost face: q[face] = sum_m w_m * q[nbr_m] (weights carry
    // the mirror sign and sum to sdf/d in [-1, 0) over open faces;
    // |q_ghost| <= max|q_nbr|).
    // Pre: U's solid faces hold zeros (maskSolidFaces ran on it).
    void apply(DeviceField& U) const;

    // diff = in with ghost faces filled: the field the viscous term reads.
    // Asynchronous on the default stream, like every other launcher here.
    // Pre: buffers same size, diff does not alias in.
    void fillFrom(const DeviceField& in, DeviceField& diff) const;

    std::size_t count() const { return count_; }
    // Ghost faces with no clean interpolation cell within reach (geometry
    // unresolved there: gaps / concave corners under ~2 cells), imaged by the
    // renormalized fallback instead. 0 on any resolved convex body.
    std::size_t fallbackCount() const { return fallback_; }

private:
    DeviceGhostFill(DeviceBuffer<int> face, DeviceBuffer<int> comp,
                    DeviceBuffer<int> nbr, DeviceBuffer<Real> w, std::size_t n,
                    std::size_t fallback)
        : face_(std::move(face)), comp_(std::move(comp)), nbr_(std::move(nbr)),
          w_(std::move(w)), count_(n), fallback_(fallback)
    {
    }

    // Ghost faces of dst = the mirror sum over src's open faces. src == dst is
    // legal: a ghost face is never one of its own samples.
    void fill(const DeviceField& src, DeviceField& dst) const;

    DeviceBuffer<int> face_; // ghost face linear index
    DeviceBuffer<int> comp_; // 0 = u, 1 = v, 2 = w
    DeviceBuffer<int> nbr_;  // 8 sample indices per ghost face
    DeviceBuffer<Real> w_;   // 8 weights per ghost face (0 for excluded samples)
    std::size_t count_ = 0;
    std::size_t fallback_ = 0;
};

} // namespace dns::gpu
