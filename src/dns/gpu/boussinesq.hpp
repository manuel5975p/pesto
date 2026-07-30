// Boussinesq convection pieces for the INCOMPRESSIBLE MAC solver: the
// cell-centred temperature field with its own advection-diffusion, the
// buoyancy source, the no-slip isothermal y walls, and the per-y-plane moment
// sums the Nusselt numbers are built from.
//
// WHY THIS EXISTS ALONGSIDE gpu/comp_conv
// --------------------------------------
// comp_conv puts gravity in the fully compressible solver, where density
// responds to temperature through the equation of state and the time step is
// acoustic. That is the honest physics, and it is also why a run costs 1/M
// steps: real air convects at M ~ 1e-3, so a faithful Mach number is a
// thousand-fold cost and the affordable ones over-stratify the layer.
//
// Boussinesq IS the M -> 0 limit taken analytically instead of paid for:
// divergence-free velocity, density variation retained ONLY in the buoyancy
// term. There is then no sound speed, no acoustic dt and no adiabatic
// gradient at all -- the superadiabatic correction comp_conv must carry is
// identically zero here, so Ra and Nu are both on the full imposed drop.
//
// The price is that it is a different model, not a cheaper solver for the same
// one: it cannot represent acoustics, compressive heating, or a stratified
// layer deep enough for g/cp to matter.
//
// GRID CONVENTION. y carries ONE ghost row at j = 0 and j = ny-1, so interior
// cells are j in [1, ny-2] and the two wall faces are the v-faces j = 1 and
// j = ny-1. With periodic sides x and z are periodic over their full extent,
// which matches PoissonBc::PeriodicXZWallY exactly; with CLOSED sides
// (`--sides closed`) x -- and z when nz > 1 -- carry the same one-cell ghost
// ring, the lateral wall faces are the u-faces i = 1 and i = nx-1 (w-faces
// k = 1 and k = nz-1), and the pressure problem becomes PoissonBc::NeumannBox.
// `closed` is that choice, threaded through every entry point below; false
// reproduces the periodic solver bit-for-bit.
//
// GHOST OWNERSHIP. Every kernel here (and every core kernel the runner calls)
// writes ghost cells with whatever its interior formula gives -- the wall fill
// OWNS them and runs after each stage and after each projection, so no value a
// ghost cell holds between those points is ever read as physics.
//
// GPU-only header.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"

#include <vector>

namespace dns::gpu {

// No-slip isothermal plates. Velocity: the wall-normal component is zeroed ON
// the two wall faces and the tangential ghosts are ANTImirrored, so u and w
// vanish at the wall by symmetry (this is the one place a convection cell
// differs from the tunnel, whose lateral walls are free-slip and therefore
// mirror the tangential components instead). Temperature: T_ghost = 2 Twall -
// T_in, the linear image that puts T = Twall on the wall face.
// `patch` is the side fraction of a CENTRED hot patch on the floor, in (0,1]:
// the floor holds Tbot where |x - Lx/2| <= patch*Lx/2 and likewise in z, and
// the lid temperature everywhere else, so patch = 1 is the uniformly heated
// plate and is bit-for-bit unchanged. At nz == 1 the z test is vacuous and the
// patch is the middle strip in x -- the 2D reading of "part of the floor". It
// is centred on the FLUID extent, so it does not drift once lateral ghost
// columns exist.
//
// `closed` gives the lateral faces the same treatment one axis over, with the
// one difference that is the point of the distinction: a side wall is
// ADIABATIC (T mirrored, dT/dn = 0). The drop across the PLATES is the
// experiment; a conducting side wall would be a second, uncontrolled source.
// Axis order is y, then x, then z, so a corner ghost holds the last axis's
// mirror of an already-mirrored cell -- the diagonal reflection, which is what
// the cross terms of the viscous stencil want.
void applyRbWall(const GpuGrid& g, Real Tbot, Real Ttop, Real patch, bool closed,
                 DeviceField& U, DeviceScalar& T);

// Mirror the pressure into the wall ghost cells: dp/dn = 0 at a wall. The
// Poisson solve writes them as ZERO, and subtractGradient would then read that
// zero as a real value and drive a spurious velocity through the wall face. A
// corner ghost takes the DIAGONAL interior cell, so the whole fill is a pure
// read of interior values and needs no axis ordering.
// Pre: call between solve() and subtractGradient().
void mirrorPressureWalls(const GpuGrid& g, bool closed, DeviceScalar& p);

// out = a*Tx + b*(T + dt*(-div(u T) + kappa lap T)) on the INTERIOR rows,
// the temperature twin of kernels.hpp's fusedRhsStage. Advection is the
// conservative centred form on the MAC faces, so the wall face's v = 0 carries
// exactly no heat and all wall transport is the Fourier term.
// Pre: out aliases neither T nor U.
void stageTemperature(const GpuGrid& g, Real kappa, Real a,
                      const DeviceScalar& Tx, Real b, const DeviceField& U,
                      const DeviceScalar& T, Real dt, DeviceScalar& out);

// U.v += s * gAlpha * (T - Tref) on the interior y faces, with T averaged onto
// the face. This is the ENTIRE buoyancy model: the only place density enters.
void addBuoyancy(const GpuGrid& g, Real gAlpha, Real Tref, Real s,
                 const DeviceScalar& T, DeviceField& U);

// Zero every WALL ghost cell of a cell-centred field: the two y rows always,
// the lateral columns when they are walls. The divergence diagnostic needs
// this: a ghost cell holds the antimirrored image, whose divergence is
// legitimately nonzero and ~1e5 times the interior's, so a plain whole-grid max
// reports the mirror instead of the projection.
void zeroGhostCells(const GpuGrid& g, bool closed, DeviceScalar& q);

// Deterministic FP64 per-y-plane moment sums: for every row j,
//   out[j*4 + 0..3] = sum over (i,k) of {T, v_c, v_c T, |u_c|^2}
// with v_c and u_c the cell-centred velocities. One thread per row, fixed
// (i,k) order -- bit-reproducible run to run. `host` is resized to g.ny * 4.
// The sums run over the FLUID (i,k) only: a lateral ghost column holds the
// wall mirror, and averaging it in would bias every plane mean.
hipError_t rbPlaneStats(const GpuGrid& g, bool closed, const DeviceField& U,
                        const DeviceScalar& T, DeviceBuffer<double>& scratch,
                        std::vector<double>& host);

} // namespace dns::gpu
