// The conserved ghost state of a no-slip wall, and the index arithmetic that
// finds the ghost/source pair on a y face.
//
// Three walls in this tree write the SAME ghost state -- the channel plates
// (gpu/comp_channel), the convection cell's plates (gpu/comp_conv) and the
// O-grid body surface (gpu/ogrid_bc) -- and each used to carry its own copy of
// it. That is exactly the arithmetic that must not drift: a wall temperature
// floor or a mirrored-velocity convention that differs by branch is a physics
// difference nothing in the test tree would attribute to the right place.
//
// The expressions are verbatim what the three kernels had, in the same order,
// so every measured anchor is bit-for-bit unmoved.
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/gpu/index.hpp"

#include <cstddef>

namespace dns::gpu::internal {

// Thread t of a 2 * nx * nz launch -> the y-ghost cell `c` and the interior
// cell `s` it mirrors, with `hi` true on the top wall. Ghost rows are {0,
// ny-1} and their sources {1, ny-2}, so writes and reads are disjoint.
// Returns false when t is past the end of the launch.
__host__ __device__ inline bool yGhostPair(const GpuGrid& g, std::size_t t,
                                           int& i, int& k, bool& hi,
                                           std::size_t& c, std::size_t& s)
{
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= 2 * nxz) return false;
    hi = t >= nxz;
    i = int(t % g.nx);
    k = int((t / g.nx) % g.nz);
    c = gidx(g, i, hi ? g.ny - 1 : 0, k);
    s = gidx(g, i, hi ? g.ny - 2 : 1, k);
    return true;
}

// The same pair on a wall normal to x (axis 0) or z (axis 2): thread t of a
// 2 * (face) launch, the face sweeping EVERY index of the other two axes (ghost
// rows included) -- (j, k) beside a walled i, (i, j) beside a walled k. So a
// corner ghost is this axis's mirror of a cell an earlier pass already
// mirrored: the diagonal reflection. Index pair only; the state edit is the
// caller's physics. Returns false when t is past the end of the launch.
// Pre: axis is 0 or 2.
__host__ __device__ inline bool sideGhostPair(const GpuGrid& g, int axis,
                                              std::size_t t, bool& hi,
                                              std::size_t& c, std::size_t& s)
{
    const int nA = axis == 0 ? g.nx : g.nz;
    const int nB = axis == 0 ? g.ny : g.nx;   // the two swept axes, B then C
    const int nC = axis == 0 ? g.nz : g.ny;
    const std::size_t face = std::size_t(nB) * nC;
    if (t >= 2 * face) return false;
    hi = t >= face;
    const int b = int(t % nB), cc = int((t / nB) % nC);
    const int a0 = hi ? nA - 1 : 0, a1 = hi ? nA - 2 : 1;
    c = axis == 0 ? gidx(g, a0, b, cc) : gidx(g, b, cc, a0);
    s = axis == 0 ? gidx(g, a1, b, cc) : gidx(g, b, cc, a1);
    return true;
}

// Adiabatic no-slip: rho and E copied, ALL momentum antimirrored, so every
// velocity component vanishes at the wall face by symmetry. |m| is unchanged,
// so the copied E keeps p mirrored -- T0 == T1 and the Fourier flux through
// the face is identically zero.
__host__ __device__ inline void adiabaticGhost(std::size_t c, std::size_t s,
                                               Real* rho, Real* mx, Real* my,
                                               Real* mz, Real* E)
{
    rho[c] = rho[s];
    mx[c] = -mx[s];
    my[c] = -my[s];
    mz[c] = -mz[s];
    E[c] = E[s];
}

// Isothermal no-slip: the ghost thermodynamic state is built so the wall face
// holds T = Twall and the wall-normal momentum balance with the mirrored
// velocity. T0 = 2 Twall - T1, FLOORED at Twall/20 so a transient cannot
// construct a negative-temperature ghost; p0 = p1 + rho1 * fspan; rho0 =
// p0/(R T0); m0 = -rho0 u1; E0 from (p0, rho0, |u1|).
// fspan = -f . (x1 - x0), f the body force per unit mass: at a resting no-slip
// wall the fluid does not accelerate, so grad p = rho f there (plus the
// viscous term, not imaged -- comp_ghost.hip kGhostFill says why), and the
// copy p0 = p1 is that balance only where f . n == 0. A gravity plate is
// where it is not (comp_conv: the hydrostatic gradient). fspan == 0 is the
// copy bit-for-bit: p1 + rho1 * 0 is p1.
// Pre: rho[s] > 0, gam > 1, Rgas > 0, Twall > 0.
__host__ __device__ inline void isothermalGhost(std::size_t c, std::size_t s,
                                                Real gam, Real Rgas, Real Twall,
                                                Real fspan, Real* rho, Real* mx,
                                                Real* my, Real* mz, Real* E)
{
    const Real r1 = rho[s];
    const Real u1 = mx[s] / r1, v1 = my[s] / r1, w1 = mz[s] / r1;
    const Real ke1 = Real(0.5) * (u1 * u1 + v1 * v1 + w1 * w1);
    const Real p1 = (gam - Real(1)) * (E[s] - r1 * ke1);
    const Real T0 = fmaxf(Real(2) * Twall - p1 / (Rgas * r1), Real(0.05) * Twall);
    const Real p0 = p1 + r1 * fspan;
    const Real r0 = p0 / (Rgas * T0);
    rho[c] = r0;
    mx[c] = -r0 * u1;
    my[c] = -r0 * v1;
    mz[c] = -r0 * w1;
    E[c] = p0 / (gam - Real(1)) + r0 * ke1;
}

} // namespace dns::gpu::internal
