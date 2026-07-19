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
};

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
__host__ __device__ inline void decompose(const GpuGrid& g, std::size_t c,
                                          int& i, int& j, int& k)
{
    i = int(c % g.nx);
    j = int((c / g.nx) % g.ny);
    k = int(c / (std::size_t(g.nx) * g.ny));
}

} // namespace dns::gpu
