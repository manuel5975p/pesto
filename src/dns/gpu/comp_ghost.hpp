// Sharp sub-cell no-slip wall for the compressible solver (2D/3D): solid cells
// near the surface become mirror ghosts (velocity extrapolated through the
// wall; thermodynamics either adiabatic -- T copied -- or isothermal; pressure
// imaged by the wall-normal momentum balance, grad p = -rho a_wall, sampled on
// the wall normal), refilled per RK stage via the solver stage hook;
// drag/lift/wall heat are the exact discrete fluxes through the fluid/ghost
// interface. GPU-only header.
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

// Kinematics of the wall material point at a WORLD point on the surface: its
// velocity u and its acceleration a. Both are needed because the wall enters
// two images. The velocity mirror images about u instead of about rest. The
// pressure image carries the wall-normal momentum balance: at a no-slip wall
// the fluid moves with the wall, so rho Du/Dt = rho a there and
//   grad p |wall = -rho a + div tau
// (no body force on any ghost-wall path). A resting wall has a == 0 and keeps
// the zero-gradient copy; a spinning one has the centripetal a = w x (w x r),
// which the copy drops at first order (half a cell of rho w^2 r of pressure
// on every face that sees the shaft). The viscous term is not imaged -- see
// kGhostFill. This module stays ignorant of WHY the wall moves -- a rotor
// supplies dns::rotorWallVelocity/rotorWallAcceleration, a translating or
// pitching body would supply its own -- so no motion model lives here.
struct GhostWallKinematics
{
    std::array<double, 3> u{}, a{};
};
// nullptr keeps the resting wall, arithmetic untouched.
using GhostWallMotion =
    std::function<GhostWallKinematics(double x, double y, double z)>;

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
    std::vector<int> gnbr;   // 8 trilinear source cells per ghost (velocity)
    std::vector<Real> gw;    // 8 renormalized weights per ghost
    // The thermodynamic (rho, p) stencil, 8 + 8 per ghost: the same as the
    // velocity one where that is all fluid, otherwise ON the normal a little
    // further out (classifyCompGhost says why).
    std::vector<int> tnbr;
    std::vector<Real> tw;
    std::vector<Real> gscale;// velocity extrapolation factor dw/ds per ghost
    // Wall distance of each ghost centre (-sdf): the RANS omega image needs
    // the sample's wall distance d_s = dw / a (dns/rans.hpp, applyTurb).
    std::vector<Real> gdw;
    // The FLUID face neighbours of each ghost, kFaceNbr per ghost, -1 padded
    // (the cells across its interface faces, in face order): the mu_t image
    // is clamped against them (mirrorMut).
    static constexpr int kFaceNbr = 6;
    std::vector<int> gfnb;
    std::vector<Real> uwall; // wall velocity per ghost, xyz; empty = at rest
    // Per ghost, a_wall . (x_sample - x_ghost): the pressure image's increment
    // per unit sample density, p_ghost = p_sample + rho_sample * aspan. Present
    // exactly when uwall is.
    std::vector<Real> aspan;
    std::vector<int> deep;   // deep-solid cells (quiescent fill)
    std::vector<int> fcell;  // fluid cell of an interface face
    std::vector<int> fdir;   // 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z (ghost side)
    std::vector<Real> fpos;  // physical centre of each interface face, xyz
    // Wall velocity AT each interface face centre, xyz; present exactly when
    // uwall is. The impermeable face of a moving wall carries u_wall . n.
    std::vector<Real> fuwall;
    std::vector<int> solid;  // every cell with sdf < 0 (mask / diagnostics)
    // Ghosts imaged from their fluid neighbours because the normal sample was
    // unusable (diagnostic: thin or concave geometry at the grid scale).
    int fallback = 0;

    // TWO-SIDED FACES. Where a section is under ~2 cells thick (sharp edges,
    // crests) a ghost is face-adjacent to fluid on both surfaces but carries
    // ONE image, mirrored from the surface nearest its centre. A fluid cell on
    // the other side would flux against the far surface's pressure and
    // mirrored velocity. Each such interface face -- its ghost's sdf normal
    // points away from the fluid cell -- gets its own SIDE image, built like a
    // ghost's but along the sdf normal at the FLUID cell, about that cell's
    // wall point. The solver's rhs hook and the tally read it for that face
    // only; the ghost cell keeps its image for every other face. Per side
    // face: the face index into fcell, the velocity and thermo stencils (8
    // each), a = dw/dsmp, and with a moving wall its wall velocity (xyz) and
    // pressure span, sampled at the side image's own wall point.
    std::vector<int> sideFace;
    std::vector<int> sideNbr;
    std::vector<Real> sideW;
    std::vector<int> sideTnbr;
    std::vector<Real> sideTw;
    std::vector<Real> sideScale;
    std::vector<Real> sideUwall; // 3 per side face; empty = at rest
    std::vector<Real> sideAspan; // with sideUwall
    // The fluid cells carrying side faces, ascending, and each one's run
    // [sideStart[q], sideStart[q+1]) of side faces (a fluid cell's faces are
    // contiguous in face order): the rhs pass sums a cell's faces in one
    // thread, deterministically. sideStart has nSideCell() + 1 entries, or
    // none when there are no side faces.
    std::vector<int> sideCell;
    std::vector<int> sideStart;
    // Wrong-side faces left on the shared image: no usable sample on the
    // fluid cell's side (diagnostic, like fallback).
    int sideSkipped = 0;

    int nGhost() const { return int(gcell.size()); }
    int nDeep() const { return int(deep.size()); }
    int nFace() const { return int(fcell.size()); }
    int nSide() const { return int(sideFace.size()); }
    int nSideCell() const { return int(sideCell.size()); }
};

// Largest table a set of azimuths needs; the device buffers are sized once from
// this so replaying a bin is a copy, never a reallocation.
struct CompGhostCapacity
{
    int nGhost = 0, nDeep = 0, nFace = 0, nSide = 0, nSideCell = 0;

    void cover(const CompGhostTables& t);
};

// The classification make() runs, on its own. Pure host: no HIP call, no device
// memory, and `sdf` is evaluated only where the bbox short-circuit lets it be.
// Safe to run concurrently for different azimuths as long as sdf and uwall are.
// Post: every solid cell a fluid cell's face flux reads (its face and edge
// neighbours) is a ghost, never deep -- so the interface faces are EVERY
// fluid/solid face (a closed surface, sum n A == 0) and the fluid never sees a
// resting pInf cell. Every wrong-side face has a side image or is counted in
// sideSkipped. Pre: bbox is null or 6 doubles [lo xyz, hi xyz].
CompGhostTables classifyCompGhost(const Grid& g, const GhostSdf& sdf,
                                  const double* bbox, const AxisRemap3& remap,
                                  const GhostWallMotion* motion);

class DeviceCompGhost
{
public:
    using Sdf = GhostSdf;
    using WallThermal = GhostWallThermal;
    using WallMotion = GhostWallMotion;

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
    // motion != nullptr makes the surface MOVE (see WallMotion); it is sampled
    // once per ghost cell at the foot of the wall normal, so it costs nothing
    // per step and rides the azimuth table for a rotor. nullptr is the resting
    // wall, bit for bit.
    static std::expected<DeviceCompGhost, hipError_t> make(
        const Grid& g, const Sdf& sdf, double gamma, double rhoInf, double pInf,
        const double* bbox = nullptr, WallThermal wall = {},
        const WallMotion* motion = nullptr);

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
        const WallMotion* motion = nullptr);

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

    // --- RANS (immersed ghost wall; masterplan_flowrans 4.9) ---------------
    // The turbulence scalars of the ghosts, after apply() (they carry the
    // ghost density apply() wrote), through the SAME linear image the
    // velocity uses: nu~_g = -a sum w nu~_s, k_g = -a sum w k_s (specific
    // values), omega_g = (1 + a) omega_w - a sum w omega_s with omega_w =
    // 60 nu_w / (beta1 d_s^2), d_s = dw / a the sample's wall distance and nu_w
    // from the thermo sample -- an UNVERIFIED choice for a sub-cell wall,
    // labelled and measured on the immersed channel. Deep cells: nu~ = k = 0,
    // omega = omega_inf. A compensated state gets the low words of every ghost
    // and deep cell (all fields) zeroed. Pre: s.nT == ransScalars(model).
    void applyTurb(DeviceCompState& s, RansModel model, const Transport& tr,
                   double omegaInf) const;
    // mu_t / F1 images for the mutHook: mu_t_g = -min(a sum w mu_t_s, min of
    // its fluid face neighbours' mu_t) -- the odd image (wall face average ~ 0,
    // never computed from ghost scalars), clamped so every staircase face
    // keeps mu + mu_t >= mu; F1_g = sum w F1_s; deep cells mu_t = 0, F1 = 1.
    // f1 may be null (SA). Pre: the fluid mu_t is final.
    void mirrorMut(Real* nut, Real* f1) const;
    // The first-cell y+ the immersed wall resolves, per interface face from
    // its FLUID cell c: y+ = sqrt(|u_c| d_c / nu_c), the linear-sublayer
    // estimate (u_tau^2 = nu |u_c| / d_c) -- an estimate on a staircase wall,
    // reported as one. max and mean over the faces. Pre: solver.rans().
    struct YPlus { double max, mean; };
    YPlus wallYplus(DeviceComp& solver, const DeviceCompState& s);

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
    // flux itself uses (plus the eddy conductivity under RANS/LES). The convective and viscous-work parts of the face
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
    // Under RANS or LES the viscous part (and heat's conduction) carries the
    // face mu_t -- with its wall image, mirrorMut -- and the solver's eddy
    // conductivity, through the flux the solver itself uses, so drag, torque
    // and heat are the budget of the solved equations (gpu_les_wall pins the
    // momentum closure); without either, the molecular tally bit for bit.
    Force force(DeviceComp& solver, const DeviceCompState& s,
                const std::array<double, 3>& pivot = {});

    // One interface face of the same tally, split (--emit surface on the
    // ghost wall: which part of the body carries which part of the force).
    // s = the face area vector oriented into the body (Cartesian: sign * area
    // along `axis`; focal: the metric S); p = the KG face pressure the flux
    // uses, minus pRef. f = the face's full momentum flux minus pRef s (gauge:
    // the closed surface has sum s == 0, so sum f == force() to round-off),
    // fp = p s its pressure part, fc its convective part (rho v_a u at the
    // impermeable face, applyRhs: exactly 0 on a resting or MRF wall, the
    // momentum the moving wall's face sweeps on a rotor), and fv = f - fp -
    // fc the viscous stress. Parts in FP64 from the same FP32
    // face values; f is the default tally's own expression.
    struct SurfaceFace
    {
        double x[3];
        int axis, sign;
        double area, s[3], p, f[3], fp[3], fc[3], fv[3];
    };
    // Same preconditions and mu_t as force(); one row per resident face, in
    // table order. Fails only on a device error (allocation, copy).
    std::expected<std::vector<SurfaceFace>, hipError_t>
    surfaceFaces(DeviceComp& solver, const DeviceCompState& s, double pRef);

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
    // Two-sided faces of the resident tables carrying a side image, and the
    // wrong-side faces left on the shared one (CompGhostTables::sideFace).
    int nSide() const { return nSide_; }
    int nSideSkipped() const { return sideSkipped_; }
    // The rhs half of the wall faces, through stageAdd; install it in the
    // solver's rhs hook (force() and surfaceFaces() tally the same fluxes).
    // IMPERMEABLE FACES: every fluid/ghost face carries no advective flux
    // relative to the wall -- rest: none; MRF: none relative to the frame,
    // pressure work p w_n; moving: rho u_wall.n {1, u, e} + p u_wall.n -- and
    // keeps its pressure flux, viscous stress, work and conduction. The face
    // mean of a fluid cell and its odd linear image does not zero u_n (it goes
    // as s^2 at a no-slip wall, and the staircase face sits up to h/2 off the
    // wall), so without this the body was a net mass source of 0.1-1.1%
    // rho U c (exp_mrfB). TWO-SIDED faces: their flux reads the side image
    // (the solver's own flux: Cartesian or curvilinear, MRF frame, mu_t with
    // the odd image clamped as mirrorMut clamps). One thread per fluid cell,
    // faces in table order: deterministic. Pre: called from the rhs hook (the
    // stage's primitives and mu_t are of s), s ghost-filled by apply().
    void applyRhs(const DeviceComp& solver, const DeviceCompState& s,
                  DeviceCompState& out, const StageCombine& cmb) const;

private:
    DeviceCompGhost() = default;
    // force()/surfaceFaces() shared front half: primitives, mu_t, and the
    // face-flux kernel into acc (kSplit stages the parts, see kBodyFlux).
    // False on the focal-without-curvilinear-solver programmer error.
    template <bool kSplit>
    bool stageFlux(DeviceComp& solver, const DeviceCompState& s, double* acc,
                   double pRef);

    GpuGrid g_{};
    Real gamma_ = Real(1.4), rhoInf_ = Real(1), pInf_ = Real(1);
    // Thermal wall in device precision. iso_ == 0 runs the original adiabatic
    // arithmetic untouched, which is what pins the published cd/cl anchors.
    int iso_ = 0;
    Real Twall_ = Real(0), Rgas_ = Real(1);
    int nGhost_ = 0, nDeep_ = 0, nFace_ = 0, fallback_ = 0;
    int nSide_ = 0, nSideCell_ = 0, sideSkipped_ = 0;
    // Side images of the two-sided faces (CompGhostTables::side*).
    DeviceBuffer<int> sideFace_, sideNbr_, sideTnbr_, sideCell_, sideStart_;
    DeviceBuffer<Real> sideW_, sideTw_, sideScale_, sideUwall_, sideAspan_;
    DeviceBuffer<int> gcell_;    // ghost cell index
    DeviceBuffer<int> gnbr_;     // 8 trilinear source cells per ghost
    DeviceBuffer<Real> gw_;      // 8 renormalized weights per ghost
    DeviceBuffer<int> tnbr_;     // 8 thermo (rho, p) source cells per ghost
    DeviceBuffer<Real> tw_;      // and their weights
    DeviceBuffer<Real> gscale_;  // velocity extrapolation factor dw/ds
    DeviceBuffer<Real> gdw_;     // ghost wall distance (RANS omega image)
    DeviceBuffer<int> gfnb_;     // fluid face neighbours per ghost (mu_t clamp)
    // Wall velocity at each ghost's own wall point, xyz interleaved; empty on a
    // resting wall, which is what selects the untouched arithmetic in kGhostFill.
    DeviceBuffer<Real> uwall_;
    DeviceBuffer<Real> aspan_;   // a_wall . (x_sample - x_ghost); with uwall_
    DeviceBuffer<int> deep_;     // deep-solid cell index
    DeviceBuffer<int> fcell_;    // fluid cell of an interface face
    DeviceBuffer<int> fdir_;     // 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z (ghost side)
    DeviceBuffer<Real> fuw_;     // wall velocity at each face centre; moving
    // The fluid cells owning interface faces, in face order, and each one's
    // face run [wstart[q], wstart[q+1]): the impermeable pass's CSR, built by
    // load() from fcell (a fluid cell's faces are contiguous).
    DeviceBuffer<int> wcell_, wstart_;
    int nWallCell_ = 0;
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
