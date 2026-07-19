// Uniform periodic Cartesian grid with MAC staggering; owns index arithmetic.
// 2D runs use nz == 1: periodic wrap makes every z stencil cancel exactly, so
// the 3D operators ARE the 2D ones (w stays 0; solvers skip the z transform).
#pragma once

#include <cassert>
#include <cstddef>

namespace dns {

// Staggering convention (all arrays are nx*ny*nz, fully periodic):
//   p(i,j,k)  cell centre   ((i+.5)hx, (j+.5)hy, (k+.5)hz)
//   u(i,j,k)  x-face        (i*hx,     (j+.5)hy, (k+.5)hz)   left face of cell i
//   v(i,j,k)  y-face        ((i+.5)hx, j*hy,     (k+.5)hz)
//   w(i,j,k)  z-face        ((i+.5)hx, (j+.5)hy, k*hz)
struct Grid
{
    int nx, ny, nz;
    double hx, hy, hz;

    // n* > 0, l* > 0
    static Grid make(int nx, int ny, int nz, double lx, double ly, double lz)
    {
        assert(nx > 0 && ny > 0 && nz > 0);
        assert(lx > 0 && ly > 0 && lz > 0);
        return {nx, ny, nz, lx / nx, ly / ny, lz / nz};
    }

    std::size_t size() const { return std::size_t(nx) * ny * nz; }
    double cellVolume() const { return hx * hy * hz; }

    // Periodic wrap of one axis index; |d| <= n.
    static int wrap(int i, int n)
    {
        assert(i >= -n && i <= 2 * n - 1);
        return i < 0 ? i + n : (i >= n ? i - n : i);
    }

    std::size_t idx(int i, int j, int k) const
    {
        i = wrap(i, nx); j = wrap(j, ny); k = wrap(k, nz);
        return std::size_t(i) + std::size_t(nx) * (std::size_t(j) + std::size_t(ny) * k);
    }
};

} // namespace dns
