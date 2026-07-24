// Sharp sub-cell no-slip adiabatic wall for the compressible solver (2D/3D):
// solid cells near the surface become mirror ghosts (velocity extrapolated
// through the wall, p/rho copied), refilled per RK stage via the solver stage
// hook; drag/lift is the exact discrete momentum flux through the fluid/ghost
// interface. GPU-only header.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/remap.hpp"

#include <expected>
#include <functional>
#include <vector>

namespace dns::gpu {

class DeviceCompGhost
{
public:
    // sdf(x, y, z) > 0 in fluid, < 0 in solid, approximately a distance.
    using Sdf = std::function<double(double, double, double)>;

    // Freestream (rhoInf, pInf) fills deep-solid cells (quiescent interior);
    // sdf resolved by the grid (body radius >> h). bbox (optional, 6 doubles
    // [lo x,y,z, hi x,y,z] = the body's axis-aligned bounds): cells whose centre
    // is provably farther than the near-wall band from that box are taken as
    // plain fluid WITHOUT calling sdf -- a compact body then costs O(bbox cells)
    // sdf evals instead of O(grid), the dominant cost for a mesh sdf. nullptr =
    // evaluate every cell (unchanged; analytic sdfs / whole-domain bodies).
    static std::expected<DeviceCompGhost, hipError_t> make(
        const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
        const double* bbox = nullptr);

    // Focal (curvilinear) variant: cell centres are the remap of the uniform
    // reference centres, so every distance, normal and band test is taken in
    // PHYSICAL space while the trilinear stencil weights stay in index space
    // (the separable monotone map keeps the cell-centre lattice a valid
    // interpolation lattice; the per-axis inverse takes the sample point
    // back). Classification is entirely host-side, so the inverse never
    // reaches a kernel. An all-inactive remap reproduces the Cartesian
    // tables bit-for-bit. Pre: the solver passed to force() is the
    // curvilinear DeviceComp built from the same remap's nodes.
    static std::expected<DeviceCompGhost, hipError_t> make(
        const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
        const double* bbox, const AxisRemap3& remap);

    // Overwrite ghost cells with wall states and deep-solid cells with the
    // quiescent interior; install via DeviceComp::setStageHook.
    void apply(DeviceCompState& s);

    struct Force { double fx, fy, fz; };

    // Momentum flux fluid -> body through the interface (calls
    // solver.primitives(s); state must be ghost-filled, i.e. after apply).
    Force force(DeviceComp& solver, const DeviceCompState& s);

    // True when make() was given an active focal remap: the force tally then
    // integrates the curvilinear face flux (area carried by S) instead of the
    // Cartesian one.
    bool curvilinear() const { return curv_; }

    // Cells with sdf < 0 (mask/diagnostics).
    const std::vector<int>& solidCells() const { return solid_; }
    // Solid cells held quiescent instead of mirrored. The mirror band
    // between these and the fluid must stay at least two cells thick along
    // every axis -- the interior stencil reaches two cells, and a quiescent
    // cell inside its reach is a discontinuity against the fluid.
    const std::vector<int>& deepCells() const { return deepHost_; }

private:
    DeviceCompGhost() = default;

    GpuGrid g_{};
    Real gamma_ = Real(1.4), rhoInf_ = Real(1), pInf_ = Real(1);
    int nGhost_ = 0, nDeep_ = 0, nFace_ = 0;
    DeviceBuffer<int> gcell_;    // ghost cell index
    DeviceBuffer<int> gnbr_;     // 8 trilinear source cells per ghost
    DeviceBuffer<Real> gw_;      // 8 renormalized weights per ghost
    DeviceBuffer<Real> gscale_;  // velocity extrapolation factor dw/ds
    DeviceBuffer<int> deep_;     // deep-solid cell index
    DeviceBuffer<int> fcell_;    // fluid cell of an interface face
    DeviceBuffer<int> fdir_;     // 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z (ghost side)
    // Per-face FP64 staging for the force tally, laid out [fx | fy | fz], each
    // nFace_ long. The tally used to atomicAdd in FP32: cancelling +-
    // contributions lost precision and atomic ordering made the reported cd
    // irreproducible run to run. A deterministic FP64 reduction replaces it.
    DeviceBuffer<double> facc_;
    bool curv_ = false;
    ReduceScratch red_;
    std::vector<int> solid_;
    std::vector<int> deepHost_;
};

} // namespace dns::gpu
