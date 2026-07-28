// Device-side grid mirror and periodic index arithmetic; matches Grid::idx.
// GPU-only header: not part of the public dns/ include set.
#pragma once

#include "dns/grid.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace dns::gpu {

using Real = float; // device field precision; reductions accumulate in double

// Trivially-copyable Grid mirror with Real spacings (kernel argument).
struct GpuGrid
{
    int nx, ny, nz;
    Real hx, hy, hz;

    static GpuGrid from(const Grid& g)
    {
        return {g.nx, g.ny, g.nz, Real(g.hx), Real(g.hy), Real(g.hz)};
    }

    __host__ __device__ std::size_t size() const
    {
        return std::size_t(nx) * ny * nz;
    }

    // Interior k range on a grid carrying a one-cell ghost ring: [1, nz-2],
    // or the single plane k = 0 in native 2D (nz == 1, no z ghosts). A form
    // that hardcodes [1, nz-2] is an EMPTY range at nz == 1 and silently
    // disables whatever it guards.
    __host__ __device__ int kInLo() const { return nz == 1 ? 0 : 1; }
    __host__ __device__ int kInHi() const { return nz == 1 ? 0 : nz - 2; }

    // Interior cell count along z (1 in native 2D).
    __host__ __device__ int kInCount() const { return kInHi() - kInLo() + 1; }
};

// Lateral ghost-ring offset for the convection box (`dns conv --sides`): 1 on
// an axis whose faces are WALLS -- the ghost column the wall fill owns -- and 0
// on a periodic one, which carries no ghosts at all. z never carries one at
// nz == 1: the 2D restriction is exact periodic wrap, not a degenerate wall.
// One definition, read by both physics branches and by the host-side plane
// normalisations, so the kernel's interior and the divisor cannot disagree.
__host__ __device__ inline int sideGhostX(bool closed) { return closed ? 1 : 0; }

__host__ __device__ inline int sideGhostZ(const GpuGrid& g, bool closed)
{
    return closed && g.nz > 1 ? 1 : 0;
}

// Periodic wrap for stencil offsets; |i - [0,n)| <= 1.
__host__ __device__ inline int wrap(int i, int n)
{
    return i < 0 ? i + n : (i >= n ? i - n : i);
}

__host__ __device__ inline std::size_t gidx(const GpuGrid& g, int i, int j, int k)
{
    i = wrap(i, g.nx); j = wrap(j, g.ny); k = wrap(k, g.nz);
    return std::size_t(i) + std::size_t(g.nx) * (std::size_t(j) + std::size_t(g.ny) * k);
}

// Inverse of the flattened index: c -> (i, j, k), i fastest.
// 32-bit arithmetic throughout: every grid is validated below INT_MAX cells,
// and a 64-bit div/mod pair at the head of every kernel is not free.
__host__ __device__ inline void decompose(const GpuGrid& g, std::size_t c,
                                          int& i, int& j, int& k)
{
    const int ci = int(c);
    i = ci % g.nx;
    j = (ci / g.nx) % g.ny;
    k = ci / (g.nx * g.ny);
}

// Wrap-corrected linear offsets from cell (i,j,k) to its six face neighbours:
// neighbour index = c + p[d] (or + m[d]), no index recomputation and no
// per-access wrap branch. Computing these ONCE per thread replaces every
// gidx() in a stencil kernel; the compressible face flux alone made ~150 of
// them per cell.
//
// Two properties make the offsets reusable across a whole stencil:
//   - offset d depends only on the index along d, so the offsets taken at a
//     cell stay valid at any neighbour that differs from it along a DIFFERENT
//     axis (which is how one face flux reaches both of its cells' tangents);
//   - a degenerate axis (nz == 1) yields p[2] == m[2] == 0, the exact
//     self-wrap gidx already produced, so 2D stays bit-identical.
struct Offsets
{
    std::ptrdiff_t p[3], m[3];
};

// Pre: |i - [0,nx)| <= 1 and likewise j, k -- the same contract as gidx, and
// wrapped the same way. A caller stepping one cell off the grid (the sparse
// boundary and surface-force kernels do: they ask for the face at j-1 with
// j == 0) MUST land on the wrapped cell's offsets, not on offsets computed
// from j == -1: those two disagree exactly at the wrap, which is where the
// boundary kernels live.
__host__ __device__ inline Offsets offsetsAt(const GpuGrid& g, int i, int j,
                                             int k)
{
    i = wrap(i, g.nx);
    j = wrap(j, g.ny);
    k = wrap(k, g.nz);
    const std::ptrdiff_t sy = g.nx, sz = std::ptrdiff_t(g.nx) * g.ny;
    Offsets o;
    o.p[0] = i + 1 == g.nx ? 1 - g.nx : 1;
    o.m[0] = i == 0 ? g.nx - 1 : -1;
    o.p[1] = j + 1 == g.ny ? sy * (1 - g.ny) : sy;
    o.m[1] = j == 0 ? sy * (g.ny - 1) : -sy;
    o.p[2] = k + 1 == g.nz ? sz * (1 - g.nz) : sz;
    o.m[2] = k == 0 ? sz * (g.nz - 1) : -sz;
    return o;
}

} // namespace dns::gpu
