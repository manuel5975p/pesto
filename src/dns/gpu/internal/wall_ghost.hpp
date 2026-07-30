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
// holds T = Twall and dp/dn = 0 with the mirrored velocity. T0 = 2 Twall - T1,
// FLOORED at Twall/20 so a transient cannot construct a negative-temperature
// ghost; p0 = p1; rho0 = p0/(R T0); m0 = -rho0 u1; E0 from (p0, rho0, |u1|).
// Pre: rho[s] > 0, gam > 1, Rgas > 0, Twall > 0.
__host__ __device__ inline void isothermalGhost(std::size_t c, std::size_t s,
                                                Real gam, Real Rgas, Real Twall,
                                                Real* rho, Real* mx, Real* my,
                                                Real* mz, Real* E)
{
    const Real r1 = rho[s];
    const Real u1 = mx[s] / r1, v1 = my[s] / r1, w1 = mz[s] / r1;
    const Real ke1 = Real(0.5) * (u1 * u1 + v1 * v1 + w1 * w1);
    const Real p1 = (gam - Real(1)) * (E[s] - r1 * ke1);
    const Real T0 = fmaxf(Real(2) * Twall - p1 / (Rgas * r1), Real(0.05) * Twall);
    const Real r0 = p1 / (Rgas * T0);
    rho[c] = r0;
    mx[c] = -r0 * u1;
    my[c] = -r0 * v1;
    mz[c] = -r0 * w1;
    E[c] = p1 / (gam - Real(1)) + r0 * ke1;
}

} // namespace dns::gpu::internal
