// Boundary conditions of a body-fitted O-grid (dns/ogrid.hpp) for the
// compressible curvilinear solver: the inner no-slip wall row, the outer
// characteristic far-field row, and the exact surface tally on the wall face.
// The scheme itself is unchanged -- DeviceComp::make(const CurvMetrics&, ...)
// already carries a fully general curvilinear mesh; this header owns only the
// two rows the mesh leaves to a boundary condition. GPU-only header.
//
// TOPOLOGY (dns/ogrid.hpp, not re-derived here).
//   i  angular, nx cells, CYCLIC and CLOCKWISE: theta_i = -2 pi i / nx at node
//      i, so the outward radial normal at the CENTRE of angular cell i is
//      n = (cos th, sin th, 0) with th = -2 pi (i + 1/2) / nx -- which is also
//      the bisector of the far face's two nodes, i.e. the direction of that
//      face's area vector. Anything that integrates around the body must carry
//      that sign or cd/cl flip.
//   j  radial, ny cells INCLUDING two ghost rows: cell 0 is the inner ghost
//      (inside the solid), cells 1..ny-2 are fluid, cell ny-1 is the outer
//      ghost. Cell 0 has no valid lo face (its lo neighbour wraps to the far
//      boundary), so no flux divergence may be taken there: BOTH ghost rows are
//      overwritten every RK stage, exactly as the channel wall is, and their
//      rhs is discarded.
//   k  spanwise, Translate-periodic; nz == 1 is the exact 2D restriction.
//
// WHY BOTH ROWS ARE A STAGE HOOK AND NOT AN RHS HOOK. Neither row carries a
// solution: their state is defined, not advanced. That is the whole difference
// from the deleted comp_bc, which overwrote the state of INTERIOR cells after
// their conservative update and injected the acoustic transient that got it
// deleted. Here the overwrite lands on rows the time integrator does not own.
// The cost is stated at DeviceOGridBc::apply.
#pragma once

#include "dns/gpu/comp.hpp"

#include <expected>

namespace dns::gpu {

// Thermal condition of the inner (body) wall row.
enum class OGridThermal
{
    // rho and E copied, all three momentum components antimirrored: the
    // ghost temperature equals the first fluid cell's, so the Fourier flux
    // through the wall face vanishes identically. Bit-identical arithmetic to
    // the channel's adiabatic branch.
    Adiabatic,
    // T0 = 2 Twall - T1 (floored positive), p0 = p1, rho0 = p0 / (R T0), E0
    // rebuilt on the antimirrored velocity: the channel's kChannelWallY
    // arithmetic, which applies here UNCHANGED because the O-grid puts the
    // ghost row at the exact geometric reflection of the first fluid row about
    // the wall face, so the reflection factor is exactly 1.
    Isothermal,
};

// The two O-grid boundary rows, applied together as one stage hook.
class DeviceOGridBc
{
public:
    struct Config
    {
        double gamma = 1.4, Rgas = 1.0;
        OGridThermal thermal = OGridThermal::Adiabatic;
        double Twall = 1.0;   // isothermal only
        // Freestream the far-field row relaxes toward; uInf is along +x and is
        // retargeted per step by a quiet-start ramp (setFreestream).
        double rhoInf = 1.0, uInf = 1.0, pInf = 1.0;
        // Rudy-Strikwerda pressure anchor of the far row, in (0, 1]. The
        // incoming acoustic invariant is
        //   L- = (1 - sigma) * [ (p - pInf) - rho c (un - unInf) ],
        // so sigma == 1 takes it entirely from the freestream (the classical
        // characteristic far field: a hard pressure anchor) and sigma -> 0
        // leaves it at its interior value (zero gradient: no anchor, and
        // reflecting). Both limits agree EXACTLY on a purely outgoing acoustic
        // wave, for which that bracket vanishes identically -- so sigma trades
        // pressure anchoring against how much of a wake's own pressure
        // signature is reflected, and nothing else. Default 1.
        double sigma = 1.0;
        // Dead-band width (fraction of the local sound speed) of the tanh
        // blend between the outflow (entropy/shear one-sided from inside) and
        // inflow (entropy/shear prescribed) forms of the far row, so the
        // switch across un = 0 is C-infinity. Both forms vanish at the
        // freestream, so the fixed point is exact for any blend weight -- the
        // same argument comp_nscbc's lateral blend makes.
        double band = 0.01;
    };

    // ny >= 4, gamma > 1, Rgas > 0, rhoInf > 0, pInf > 0, |uInf| subsonic,
    // sigma in (0, 1], band > 0, and Twall > 0 when isothermal.
    static std::expected<DeviceOGridBc, hipError_t> make(const Grid& g,
                                                         Config c);

    // Overwrite the inner wall row (j = 0) and the outer far row (j = ny-1);
    // install via DeviceComp::setStageHook. Reads and writes are disjoint.
    //
    // WHAT THE FAR ROW COSTS. It is a ghost-row STATE condition, not an rhs
    // overwrite, which is legitimate here only because row ny-1 carries no
    // solution (see the header note) -- and it is much simpler, because the
    // radial normal is not an axis and every wave has to be re-expressed in the
    // local frame anyway. Three things are given up against a rate-form NSCBC:
    //   * no Yoo-Im transverse correction. That term is a d/dt construct with
    //     no state analogue, so a strong vortex crossing the far boundary
    //     reflects more than comp_nscbc's x-outflow would. The O-grid answer is
    //     to put rFar far enough out, which is cheap on an annulus.
    //   * the condition is imposed once per stage on the state rather than
    //     ridden through the RK combination, so the boundary is first order in
    //     time. The interior stays SSP-RK3.
    //   * the far row is second order in SPACE by reflection (ghost = 2 *
    //     boundary - interior), which is exact only because the Extend node
    //     continuation makes the outer ghost cell the exact reflection of the
    //     last fluid cell about the far circle. Do not port this to a mesh
    //     where that is not true.
    void apply(DeviceCompState& s) const;

    // Retarget the far-field freestream (quiet-start ramp). The run boosts the
    // frame with boostX and then calls this, exactly as runner_comp does.
    void setFreestream(double u) { u_ = u; }
    double freestream() const { return u_; }

    struct Surface
    {
        double fx, fy, fz; // force ON the body (fluid -> body)
        // Energy flux from the fluid INTO the body through the wall face, same
        // orientation as the force: positive = the body gains energy (a wall
        // colder than the fluid). At a no-slip wall the face velocity is
        // exactly zero, so the convective and viscous-work terms vanish
        // identically and this is the Fourier term alone -- a caller divides by
        // kappa * dT * area to get a Nusselt number.
        double heat;
    };

    // Exact discrete surface integral over the wall face row: the wall face is
    // the lo face of cell j = 1, i.e. the +1-face area vector S[1] of the inner
    // ghost row, so this is faceFluxCurv summed over ONE row of genuine mesh
    // faces, oriented into the body, in a deterministic FP64 reduction.
    //
    // This is why the O-grid exists. The immersed-body tally
    // (DeviceCompGhost::force) sums a staircase of axis-aligned Cartesian faces
    // that only approximates the surface, needs an sdf classification, an
    // aperture convention and a sub-cell mirror to define the wall state at
    // all, and carries the sub-cell boundary layer's error into cd directly.
    // Here the wall IS a mesh face row: the area vectors are the body's own,
    // the closed surface they form sums to zero identically (a uniform state
    // gives exactly zero force), and no geometry is approximated anywhere.
    //
    // Molecular stress only, like the ghost tally -- the SGS eddy viscosity is
    // deliberately not added, so `cdf` means the same thing on both paths.
    // Pre: the state is wall-filled (call apply first); calls solver.primitives.
    Surface surface(DeviceComp& solver, const DeviceCompState& s);

private:
    DeviceOGridBc() = default;

    GpuGrid g_{};
    Config c_{};
    double u_ = 0;      // live (ramped) freestream speed
    double cInf_ = 0;   // freestream sound speed
    DeviceScalar cosT_, sinT_; // nx entries: the cell-centre radial normal
    DeviceBuffer<double> acc_; // 4 * nx * nz staging for the surface tally
    ReduceScratch red_;
};

} // namespace dns::gpu
