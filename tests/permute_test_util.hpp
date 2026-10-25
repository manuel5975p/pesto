// Axis permutations of a compressible state and its grid, for the symmetry
// tests: the equations are invariant under every rotation and reflection, so
// step(P s) must equal P step(s) whatever the axis an operator was written
// along. Host only; shared by the CPU and GPU permutation tests.
#pragma once

#include "dns/comp.hpp"
#include "dns/grid.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace permtest {

using namespace dns;

// Cyclic x -> y -> z (a proper rotation): old axis x becomes new y, y becomes
// z, z becomes x. New cell (k, i, j) holds old cell (i, j, k); momenta follow.
inline Grid cyclicGrid(const Grid& g)
{
    return Grid::make(g.nz, g.nx, g.ny, g.nz * g.hz, g.nx * g.hx, g.ny * g.hy);
}

inline CompState cyclic(const Grid& g, const CompState& s)
{
    const Grid p = cyclicGrid(g);
    auto o = CompState::zeros(p);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t a = g.idx(i, j, k), b = p.idx(k, i, j);
                o.rho[b] = s.rho[a];
                o.mx[b] = s.mz[a];
                o.my[b] = s.mx[a];
                o.mz[b] = s.my[a];
                o.E[b] = s.E[a];
            }
    return o;
}

// Mirror x -> -x (a reflection): cell i -> nx-1-i, mx -> -mx. Same grid.
inline CompState mirrorX(const Grid& g, const CompState& s)
{
    auto o = CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t a = g.idx(i, j, k),
                                  b = g.idx(g.nx - 1 - i, j, k);
                o.rho[b] = s.rho[a];
                o.mx[b] = -s.mx[a];
                o.my[b] = s.my[a];
                o.mz[b] = s.mz[a];
                o.E[b] = s.E[a];
            }
    return o;
}

// 2D transpose x <-> y (a reflection, nz == 1): cell (j, i) holds old (i, j).
inline CompState transpose2d(const Grid& g, const CompState& s)
{
    const Grid p = Grid::make(g.ny, g.nx, 1, g.ny * g.hy, g.nx * g.hx, g.hz);
    auto o = CompState::zeros(p);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t a = g.idx(i, j, 0), b = p.idx(j, i, 0);
            o.rho[b] = s.rho[a];
            o.mx[b] = s.my[a];
            o.my[b] = s.mx[a];
            o.mz[b] = s.mz[a];
            o.E[b] = s.E[a];
        }
    return o;
}

// max |a - b| / max |b| for one field; 0 when b is identically zero and a
// matches it.
inline double rel(const std::vector<double>& a, const std::vector<double>& b)
{
    double e = 0.0, m = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
        m = std::max(m, std::abs(b[i]));
    }
    return m > 0.0 ? e / m : e;
}

// Worst field-relative difference over all five conserved fields.
inline double relState(const CompState& a, const CompState& b)
{
    return std::max({rel(a.rho, b.rho), rel(a.mx, b.mx), rel(a.my, b.my),
                     rel(a.mz, b.mz), rel(a.E, b.E)});
}

} // namespace permtest
