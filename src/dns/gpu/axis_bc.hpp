// The axis of a REVOLVED O-grid (dns/ogrid.hpp buildRevolvedOGrid,
// docs/rans.md "Axisymmetric wedge and the revolved O-grid"): the ghost
// columns past the axis of revolution and, on the wedge, the two ghost planes.
// Installed in the stage hook AFTER the O-grid rows (gpu/ogrid_bc), and its
// mu_t twin after theirs. GPU-only header.
//
// Axisymmetry is a mesh plus this hook, never a kernel branch: the axis faces
// have S == 0 exactly (no flux of any kind crosses the axis) and a limit
// metric (dns/curvmesh.hpp); the ghost columns hold the ACROSS-AXIS
// CONTINUATION of the flow, which is what the axis cell's centre gradients
// and its j-/k-faces' tangential stencils read:
//   wedge  column (i_g, j, 1) = R_x(pi) of (i_s, j, 1): (rho, rho u, -rho v,
//          -rho w, E, scalars), (i_g, i_s) = (0, 1) and (nx-1, nx-2), every j
//          (the ghost rows included -- the O-grid rows run first); then the
//          ghost planes (i, j, 0) = R_x(-dphi) and (i, j, 2) = R_x(+dphi) of
//          (i, j, 1) for every i and j (rotations about x commute, so no
//          per-column sign);
//   full   (i_g, j, k) = (i_s, j, (k + nphi/2) mod nphi), every field: the
//          same physical point, Cartesian components need no rotation.
// Ghost-column GEOMETRY is a placeholder no real cell reads.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/ogrid.hpp"

#include <expected>

namespace dns::gpu {

class DeviceAxisBc
{
public:
    // Pre: g a revolved O-grid grid (nx = nTheta + 2 >= 6); rev.wedge =>
    // g.nz == 3, else g.nz == rev.nphi (even, >= 4).
    static std::expected<DeviceAxisBc, hipError_t> make(const Grid& g,
                                                        const RevolvedSpec& rev);

    // The ghost columns, then (wedge) the ghost planes; the low words of
    // every written cell zeroed on a compensated state. The planes rewrite
    // EVERY field of planes 0 and 2 from plane 1 alone, which is what lets
    // the wedge's stage kernels skip those planes (DeviceComp::setStageRange).
    void apply(DeviceCompState& s) const;

    // mu_t and F1 (may be null) by the same copies: scalars, no rotation.
    void applyMut(Real* nut, Real* f1) const;

private:
    DeviceAxisBc() = default;

    GpuGrid g_{};
    bool wedge_ = true;
    Real c_ = 1, s_ = 0; // cos and sin of dphi, rounded once from FP64
};

} // namespace dns::gpu
