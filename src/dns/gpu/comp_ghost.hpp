// Sharp sub-cell no-slip wall for the compressible solver (2D/3D): solid cells
// near the surface become mirror ghosts (velocity extrapolated through the
// wall; thermodynamics either adiabatic -- p/rho copied -- or isothermal),
// refilled per RK stage via the solver stage hook; drag/lift/wall heat are the
// exact discrete fluxes through the fluid/ghost interface. GPU-only header.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/remap.hpp"

#include <array>
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

// sdf(x, y, z) > 0 in fluid, < 0 in solid, approximately a distance.
using GhostSdf = std::function<double(double, double, double)>;

// Surface velocity of the body at a WORLD point on its wall. Passing one makes
// the wall an inhomogeneous no-slip surface: the mirror images the velocity
// about u_wall instead of about rest. This module stays ignorant of WHY the
// wall moves -- a rotor supplies dns::rotorWallVelocity, a translating or
// pitching body would supply its own -- so no motion model lives here. nullptr
// keeps the resting wall, arithmetic untouched.
using GhostWallVelocity =
    std::function<std::array<double, 3>(double x, double y, double z)>;

// Everything the ghost wall uploads, before it reaches the device: the pure
// HOST result of classifying one body against one grid.
//
// Split out of DeviceCompGhost::make so a MOVING body can precompute one of
// these per azimuth -- off the device, on any thread, with no HIP call and no
// allocation the run has to keep -- and replay them into a single set of device
// buffers (dns/rotortable.hpp). A static body never sees it: make() still
// classifies and uploads in one step.
struct CompGhostTables
{
    std::vector<int> gcell;  // ghost cell index
    std::vector<int> gnbr;   // 8 trilinear source cells per ghost
    std::vector<Real> gw;    // 8 renormalized weights per ghost
    std::vector<Real> gscale;// velocity extrapolation factor dw/ds per ghost
    std::vector<Real> uwall; // wall velocity per ghost, xyz; empty = at rest
    std::vector<int> deep;   // deep-solid cells (quiescent fill)
    std::vector<int> fcell;  // fluid cell of an interface face
    std::vector<int> fdir;   // 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z (ghost side)
    std::vector<Real> fpos;  // physical centre of each interface face, xyz
    std::vector<int> solid;  // every cell with sdf < 0 (mask / diagnostics)
    // Ghosts imaged from their fluid neighbours because the normal sample was
    // unusable (diagnostic: thin or concave geometry at the grid scale).
    int fallback = 0;

    int nGhost() const { return int(gcell.size()); }
    int nDeep() const { return int(deep.size()); }
    int nFace() const { return int(fcell.size()); }
};

// Largest table a set of azimuths needs; the device buffers are sized once from
// this so replaying a bin is a copy, never a reallocation.
struct CompGhostCapacity
{
    int nGhost = 0, nDeep = 0, nFace = 0;

    void cover(const CompGhostTables& t);
};

// The classification make() runs, on its own. Pure host: no HIP call, no device
// memory, and `sdf` is evaluated only where the bbox short-circuit lets it be.
// Safe to run concurrently for different azimuths as long as sdf and uwall are.
// Post: every solid cell a fluid cell's face flux reads (its face and edge
// neighbours) is a ghost, never deep -- so the interface faces are EVERY
// fluid/solid face (a closed surface, sum n A == 0) and the fluid never sees a
// resting pInf cell. Pre: bbox is null or 6 doubles [lo xyz, hi xyz].
CompGhostTables classifyCompGhost(const Grid& g, const GhostSdf& sdf,
                                  const double* bbox, const AxisRemap3& remap,
                                  const GhostWallVelocity* uwall);

class DeviceCompGhost
{
public:
    using Sdf = GhostSdf;
    using WallThermal = GhostWallThermal;
    using WallVelocity = GhostWallVelocity;

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
    // uwall != nullptr makes the surface MOVE at that velocity (see
    // WallVelocity); it is sampled once per ghost cell at the foot of the wall
    // normal, so it costs nothing per step and rides the azimuth table for a
    // rotor. nullptr is the resting wall, bit for bit.
    static std::expected<DeviceCompGhost, hipError_t> make(
        const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
        const double* bbox = nullptr, WallThermal wall = {},
        const WallVelocity* uwall = nullptr);

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
        const double* bbox, const AxisRemap3& remap, WallThermal wall = {},
        const WallVelocity* uwall = nullptr);

    // Build from already-classified tables, with the device buffers sized for
    // `cap` rather than for `t` -- the entry point a moving body uses, so that
    // every later load() is a copy into buffers that already fit. `curv` says
    // whether the tables came from an active remap (the force tally needs it);
    // it cannot be inferred from the tables themselves.
    // Pre: cap covers t.
    static std::expected<DeviceCompGhost, hipError_t>
    fromTables(const Grid& g, const CompGhostTables& t, CompGhostCapacity cap,
               double gamma, double rhoInf, double pInf, bool curv,
               WallThermal wall = {});

    // Replace the resident tables with another azimuth's. No allocation, no
    // reclassification: the counts change, the buffers do not.
    // Pre: t fits the capacity this was built with.
    hipError_t load(const CompGhostTables& t);

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
    // qx/qy/qz are the MOMENT of that same force about the point passed to
    // force(): sum (r_face - pivot) x f_face over the same interface faces and
    // the same staged FP64 values, so torque and drag can never disagree about
    // which faces are the body. For a rotor the shaft-axis component is the
    // shaft torque and the shaft-axis force is thrust.
    struct Force { double fx, fy, fz, heat, qx, qy, qz; };

    // Momentum flux, wall heat and moment fluid -> body through the interface
    // (calls solver.primitives(s); state must be ghost-filled, i.e. after
    // apply). `pivot` is the world point the moment is taken about.
    Force force(DeviceComp& solver, const DeviceCompState& s,
                const std::array<double, 3>& pivot = {});

    // True when make() was given an active focal remap: the force tally then
    // integrates the curvilinear face flux (area carried by S) instead of the
    // Cartesian one.
    bool curvilinear() const { return curv_; }

    // True iff make() was given a wall velocity: the surface moves.
    bool moving() const { return moving_; }

    // Cells with sdf < 0 (mask/diagnostics).
    const std::vector<int>& solidCells() const { return solid_; }
    // Solid cells held quiescent instead of mirrored. The mirror band
    // between these and the fluid must stay at least two cells thick along
    // every axis -- the interior stencil reaches two cells, and a quiescent
    // cell inside its reach is a discontinuity against the fluid. None of
    // them is ever inside a fluid cell's flux stencil (classifyCompGhost).
    const std::vector<int>& deepCells() const { return deepHost_; }
    // Ghosts of the resident tables imaged from their fluid neighbours.
    int nFallback() const { return fallback_; }

private:
    DeviceCompGhost() = default;

    GpuGrid g_{};
    Real gamma_ = Real(1.4), rhoInf_ = Real(1), pInf_ = Real(1);
    // Thermal wall in device precision. iso_ == 0 runs the original adiabatic
    // arithmetic untouched, which is what pins the published cd/cl anchors.
    int iso_ = 0;
    Real Twall_ = Real(0), Rgas_ = Real(1);
    int nGhost_ = 0, nDeep_ = 0, nFace_ = 0, fallback_ = 0;
    DeviceBuffer<int> gcell_;    // ghost cell index
    DeviceBuffer<int> gnbr_;     // 8 trilinear source cells per ghost
    DeviceBuffer<Real> gw_;      // 8 renormalized weights per ghost
    DeviceBuffer<Real> gscale_;  // velocity extrapolation factor dw/ds
    // Wall velocity at each ghost's own wall point, xyz interleaved; empty on a
    // resting wall, which is what selects the untouched arithmetic in kGhostFill.
    DeviceBuffer<Real> uwall_;
    DeviceBuffer<int> deep_;     // deep-solid cell index
    DeviceBuffer<int> fcell_;    // fluid cell of an interface face
    DeviceBuffer<int> fdir_;     // 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z (ghost side)
    // Physical centre of each interface face, xyz interleaved. Carried rather
    // than recomputed in the kernel because on a focal mesh the position is the
    // REMAP of the reference face, which the tables know and CurvView does not
    // -- so the moment arm is one expression for both meshes.
    DeviceBuffer<Real> fpos_;
    // Per-face FP64 staging for the surface tally, laid out
    // [fx | fy | fz | heat | qx | qy | qz], each nFace_ long. The tally used to atomicAdd in
    // FP32: cancelling +- contributions lost precision and atomic ordering made
    // the reported cd irreproducible run to run. A deterministic FP64 reduction
    // replaces it.
    DeviceBuffer<double> facc_;
    bool curv_ = false;
    bool moving_ = false;
    CompGhostCapacity cap_{};
    ReduceScratch red_;
    std::vector<int> solid_;
    std::vector<int> deepHost_;
};

} // namespace dns::gpu
