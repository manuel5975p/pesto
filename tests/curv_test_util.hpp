// Shared helpers for the curvilinear compressible tests (CPU and GPU): cell
// centroids, smooth periodic states, the isentropic vortex, V-weighted sums.
#pragma once

#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <vector>

namespace curvtest {

using namespace dns;

inline std::array<double, 3> centroid(const CurvMesh& m, int i, int j, int k)
{
    std::array<double, 3> c{0, 0, 0};
    for (int dk = 0; dk < 2; ++dk)
        for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di) {
                const auto p = m.node(i + di, j + dj, k + dk);
                for (int d = 0; d < 3; ++d) c[d] += 0.125 * p[d];
            }
    return c;
}

// Smooth periodic state (all five fields active); w and z-variation only when
// nz > 1, so nz == 1 states are exactly 2D.
inline CompState smoothState(const CurvMesh& m, double gamma)
{
    const Grid& g = m.g;
    auto s = CompState::zeros(g);
    const double lx = g.nx * g.hx, ly = g.ny * g.hy, lz = g.nz * g.hz;
    using std::numbers::pi;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const auto p = centroid(m, i, j, k);
                const double ax = 2 * pi * p[0] / lx, ay = 2 * pi * p[1] / ly;
                const double az = g.nz > 1 ? 2 * pi * p[2] / lz : 0.0;
                const double cz = g.nz > 1 ? std::cos(az) : 1.0;
                const double rho = 1.0 + 0.2 * std::sin(ax) * std::cos(ay) * cz;
                const double u = 0.5 + 0.3 * std::cos(ax) * std::sin(ay) * cz;
                const double v = -0.2 + 0.25 * std::sin(ax + ay) * cz;
                const double w =
                    g.nz > 1 ? 0.2 * std::sin(az) * std::cos(ax) : 0.0;
                const double pr = 1.0 + 0.1 * std::cos(ax) * std::cos(ay) * cz;
                const std::size_t c = g.idx(i, j, k);
                s.rho[c] = rho;
                s.mx[c] = rho * u;
                s.my[c] = rho * v;
                s.mz[c] = rho * w;
                s.E[c] = pr / (gamma - 1.0) +
                         0.5 * rho * (u * u + v * v + w * w);
            }
    return s;
}

// Isentropic vortex density at (x, y): centre (x0, y0), strength beta.
inline double vortexRho(double x, double y, double x0, double y0, double beta,
                        double gamma)
{
    using std::numbers::pi;
    const double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
    const double T = 1.0 - (gamma - 1) * beta * beta / (8 * gamma * pi * pi) *
                               std::exp(1 - r2);
    return std::pow(T, 1.0 / (gamma - 1));
}

// Isentropic vortex in a (1,1) freestream on the mesh; exact Euler solution,
// so at time t the field is the IC advected by (t, t).
inline CompState vortexState(const CurvMesh& m, double beta, double gamma)
{
    using std::numbers::pi;
    const Grid& g = m.g;
    auto s = CompState::zeros(g);
    const double L = g.nx * g.hx, x0 = L / 2, y0 = L / 2;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const auto p = centroid(m, i, j, k);
                const double dx = p[0] - x0, dy = p[1] - y0;
                const double r2 = dx * dx + dy * dy;
                const double f = beta / (2 * pi) * std::exp(0.5 * (1.0 - r2));
                const double u = 1.0 - f * dy, v = 1.0 + f * dx;
                const double rho = vortexRho(p[0], p[1], x0, y0, beta, gamma);
                const double pr = std::pow(rho, gamma);
                const std::size_t c = g.idx(i, j, k);
                s.rho[c] = rho;
                s.mx[c] = rho * u;
                s.my[c] = rho * v;
                s.E[c] = pr / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
            }
    return s;
}

// Volume-weighted totals (rho, mx, my, mz, E) -- the conserved quantities on
// a curvilinear mesh.
inline std::array<double, 5> totals(const CurvMetrics& m, const CompState& s)
{
    std::array<double, 5> t{0, 0, 0, 0, 0};
    for (std::size_t c = 0; c < s.rho.size(); ++c) {
        t[0] += m.vol[c] * s.rho[c];
        t[1] += m.vol[c] * s.mx[c];
        t[2] += m.vol[c] * s.my[c];
        t[3] += m.vol[c] * s.mz[c];
        t[4] += m.vol[c] * s.E[c];
    }
    return t;
}

inline double relLinf(const std::vector<double>& a, const std::vector<double>& b)
{
    double e = 0.0, mx = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
        mx = std::max(mx, std::abs(b[i]));
    }
    return e / mx;
}

} // namespace curvtest
