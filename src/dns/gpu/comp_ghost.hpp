// Sharp sub-cell no-slip wall for the compressible solver (2D/3D): solid cells
// near the surface become mirror ghosts (velocity extrapolated through the
// wall; thermodynamics either adiabatic -- p/rho copied -- or isothermal),
// refilled per RK stage via the solver stage hook; drag/lift/wall heat are the
// exact discrete fluxes through the fluid/ghost interface. GPU-only header.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/remap.hpp"

#include <expected>
#include <functional>
#include <vector>

namespace dns::gpu {

// Thermal condition on an immersed body surface. Adiabatic is the historical
// wall (zero wall-normal dT/dn: p and rho copied from the sample point) and is
// the default, so every existing call site stays bit-for-bit unchanged.
// Isothermal holds the wall FACE at Twall by the same linear image the velocity
// mirror uses. Namespace scope, not nested, only so it can be a defaulted
// argument of DeviceCompGhost::make.
// Pre: Twall > 0 and Rgas > 0 when Isothermal.
struct GhostWallThermal
{
    enum class Mode { Adiabatic, Isothermal };
    Mode mode = Mode::Adiabatic;
    double Twall = 0;  // solver units; used iff Isothermal, must be > 0 then
    double Rgas = 1.0; // specific gas constant of the state, p = rho R T
};

class DeviceCompGhost
{
public:
    // sdf(x, y, z) > 0 in fluid, < 0 in solid, approximately a distance.
    using Sdf = std::function<double(double, double, double)>;
    using WallThermal = GhostWallThermal;

    // Freestream (rhoInf, pInf) fills deep-solid cells (quiescent interior);
    // sdf resolved by the grid (body radius >> h). bbox (optional, 6 doubles
    // [lo x,y,z, hi x,y,z] = the body's axis-aligned bounds): cells whose centre
    // is provably farther than the near-wall band from that box are taken as
    // plain fluid WITHOUT calling sdf -- a compact body then costs O(bbox cells)
    // sdf evals instead of O(grid), the dominant cost for a mesh sdf. nullptr =
    // evaluate every cell (unchanged; analytic sdfs / whole-domain bodies).
    // wall selects the surface thermal condition; the default is the adiabatic
    // wall this class has always had. Fails with hipErrorInvalidValue on an
    // isothermal wall with Twall <= 0 or Rgas <= 0.
    static std::expected<DeviceCompGhost, hipError_t> make(
        const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
        const double* bbox = nullptr, WallThermal wall = {});

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
        const double* bbox, const AxisRemap3& remap, WallThermal wall = {});

    // Overwrite ghost cells with wall states and deep-solid cells with the
    // quiescent interior (at Twall when the wall is isothermal); install via
    // DeviceComp::setStageHook.
    void apply(DeviceCompState& s);

    // Surface tallies, all in solver units and all with the same orientation:
    // the flux carried from the FLUID into the BODY across the fluid/ghost
    // interface. fx/fy/fz are forces; heat is an energy RATE, so heat > 0 means
    // the fluid loses energy to the body (a cold wall) and heat < 0 that the
    // body heats the fluid. With A the wetted area and D the body scale, the
    // mean wall flux is heat/A and the Nusselt number
    // Nu = (heat / A) D / (kappa (Tinf - Twall)).
    //
    // heat is the CONDUCTION alone -- the Fourier term kappa dT/dn of those
    // same faces, with kappa the face-averaged, Sutherland-scaled value the
    // flux itself uses. The convective and viscous-work parts of the face
    // energy flux are deliberately excluded: on a staircase no-slip wall the
    // face velocity is the average of a fluid cell and its mirror ghost, so
    // those terms are the mirror's imperfection rather than heat, and they do
    // not vanish on an insulated wall -- reporting them made the adiabatic
    // column read a nonzero rate that no user could tell from a real flux.
    // fx/fy/fz remain the FULL momentum flux (pressure + convection + viscous
    // stress), which is what drag is.
    //
    // An adiabatic wall therefore reports heat == 0.0 EXACTLY: its image sets
    // the wall-normal dT/dn to zero, so its wall flux is zero by definition of
    // the boundary condition. kBodyFlux in comp_ghost.hip records the measured
    // size of the mirror-band artifact this suppresses, and the ~half-percent
    // of it that still rides inside the isothermal number.
    struct Force { double fx, fy, fz, heat; };

    // Momentum flux and wall heat fluid -> body through the interface (calls
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
    // Thermal wall in device precision. iso_ == 0 runs the original adiabatic
    // arithmetic untouched, which is what pins the published cd/cl anchors.
    int iso_ = 0;
    Real Twall_ = Real(0), Rgas_ = Real(1);
    int nGhost_ = 0, nDeep_ = 0, nFace_ = 0;
    DeviceBuffer<int> gcell_;    // ghost cell index
    DeviceBuffer<int> gnbr_;     // 8 trilinear source cells per ghost
    DeviceBuffer<Real> gw_;      // 8 renormalized weights per ghost
    DeviceBuffer<Real> gscale_;  // velocity extrapolation factor dw/ds
    DeviceBuffer<int> deep_;     // deep-solid cell index
    DeviceBuffer<int> fcell_;    // fluid cell of an interface face
    DeviceBuffer<int> fdir_;     // 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z (ghost side)
    // Per-face FP64 staging for the surface tally, laid out
    // [fx | fy | fz | heat], each nFace_ long. The tally used to atomicAdd in
    // FP32: cancelling +- contributions lost precision and atomic ordering made
    // the reported cd irreproducible run to run. A deterministic FP64 reduction
    // replaces it.
    DeviceBuffer<double> facc_;
    bool curv_ = false;
    ReduceScratch red_;
    std::vector<int> solid_;
    std::vector<int> deepHost_;
};

} // namespace dns::gpu
