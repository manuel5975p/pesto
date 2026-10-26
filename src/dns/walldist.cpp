#include "dns/walldist.hpp"

#include "dns/meshsdf.hpp"
#include "dns/trimesh.hpp"

#include <algorithm>
#include <array>
#include <utility>
#include <cassert>
#include <cmath>
#include <thread>

namespace dns {

namespace {

// Distance from (px, py) to the segment a-b, exact in FP64 up to rounding.
double segmentDistance(double px, double py, double ax, double ay, double bx,
                       double by)
{
    const double ex = bx - ax, ey = by - ay;
    const double l2 = ex * ex + ey * ey;
    double t = l2 > 0.0 ? ((px - ax) * ex + (py - ay) * ey) / l2 : 0.0;
    t = std::clamp(t, 0.0, 1.0);
    return std::hypot(px - (ax + t * ex), py - (ay + t * ey));
}

} // namespace

std::vector<double> ogridWallDistance(const CurvMesh& m, int threads)
{
    const Grid& g = m.g;
    assert(g.ny >= 4 && g.nx >= 3);
    std::vector<double> wx(g.nx), wy(g.nx);
    for (int i = 0; i < g.nx; ++i) {
        const auto p = m.node(i, 1, 0);
        wx[i] = p[0];
        wy[i] = p[1];
    }
    // d of the in-plane centroid of cell (i, j, 0), min over the polyline in
    // a fixed segment order.
    std::vector<double> plane(std::size_t(g.nx) * g.ny);
    auto row = [&](int j) {
        const int js = std::clamp(j, 1, g.ny - 2); // ghost -> source row
        for (int i = 0; i < g.nx; ++i) {
            double cx = 0, cy = 0;
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di) {
                    const auto p = m.node(i + di, js + dj, 0);
                    cx += 0.25 * p[0];
                    cy += 0.25 * p[1];
                }
            double d = HUGE_VAL;
            for (int s = 0; s < g.nx; ++s) {
                const int t = s + 1 == g.nx ? 0 : s + 1;
                d = std::min(d, segmentDistance(cx, cy, wx[s], wy[s], wx[t],
                                                wy[t]));
            }
            plane[std::size_t(j) * g.nx + i] = d;
        }
    };
    const int nt = std::max(
        1, std::min(threads > 0 ? threads
                                : int(std::thread::hardware_concurrency()),
                    g.ny));
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < nt; ++t)
            pool.emplace_back([&, t] {
                for (int j = t; j < g.ny; j += nt) row(j);
            });
    }
    std::vector<double> d(g.size());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                d[g.idx(i, j, k)] = plane[std::size_t(j) * g.nx + i];
    return d;
}

std::vector<double> revolvedWallDistance(const CurvMesh& m, int nTheta,
                                         bool wedge, int threads)
{
    const Grid& g = m.g;
    assert(g.nx == nTheta + 2 && g.ny >= 4 && (!wedge || g.nz == 3));
    const int kLo = wedge ? 1 : 0, kHi = wedge ? 1 : g.nz - 1;
    // The real wall quads, 4 triangles each through the quad centroid.
    TriMesh tm;
    for (int k = kLo; k <= kHi; ++k)
        for (int i = 1; i <= nTheta; ++i) {
            const std::array<std::array<double, 3>, 4> q = {
                m.node(i, 1, k), m.node(i + 1, 1, k), m.node(i + 1, 1, k + 1),
                m.node(i, 1, k + 1)};
            std::array<double, 3> c{0, 0, 0};
            for (const auto& p : q)
                for (int a = 0; a < 3; ++a) c[a] += 0.25 * p[a];
            const int b = int(tm.verts.size());
            for (const auto& p : q) tm.verts.push_back(p);
            tm.verts.push_back(c);
            for (int e = 0; e < 4; ++e)
                tm.tris.push_back({b + e, b + (e + 1) % 4, b + 4});
        }
    const MeshSdf bvh(std::move(tm));
    std::vector<double> d(g.size(), 0.0);
    auto row = [&](int j) {
        const int js = std::clamp(j, 1, g.ny - 2);
        for (int k = kLo; k <= kHi; ++k)
            for (int i = 1; i <= nTheta; ++i) {
                double c[3] = {0, 0, 0};
                for (int dk = 0; dk < 2; ++dk)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            const auto p = m.node(i + di, js + dj, k + dk);
                            for (int a = 0; a < 3; ++a) c[a] += 0.125 * p[a];
                        }
                d[g.idx(i, j, k)] = bvh.distance(c[0], c[1], c[2]);
            }
    };
    const int nt = std::max(
        1, std::min(threads > 0 ? threads
                                : int(std::thread::hardware_concurrency()),
                    g.ny));
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < nt; ++t)
            pool.emplace_back([&, t] {
                for (int j = t; j < g.ny; j += nt) row(j);
            });
    }
    // Ghost columns (the cell they continue across the axis), then the wedge
    // planes (plane 1's, every column).
    for (int k = kLo; k <= kHi; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (const auto& [ig, is] : {std::pair{0, 1},
                                         std::pair{nTheta + 1, nTheta}}) {
                const int ks = wedge ? k : (k + g.nz / 2) % g.nz;
                d[g.idx(ig, j, k)] = d[g.idx(is, j, ks)];
            }
    if (wedge)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                for (const int k : {0, 2}) d[g.idx(i, j, k)] = d[g.idx(i, j, 1)];
    return d;
}

std::vector<double> wallFirstCellDistance(const Grid& g,
                                          const std::vector<double>& d)
{
    assert(d.size() == g.size() && g.ny >= 2);
    std::vector<double> d1(std::size_t(g.nx) * g.nz);
    for (int k = 0; k < g.nz; ++k)
        for (int i = 0; i < g.nx; ++i)
            d1[std::size_t(k) * g.nx + i] = d[g.idx(i, 1, k)];
    return d1;
}

std::vector<double> channelWallDistance(const CurvMesh& m, double delta)
{
    const Grid& g = m.g;
    assert(delta > 0.0 && g.ny >= 4);
    std::vector<double> d(g.size());
    // The channel mesh is a y-only map, so the eight-node mean y of a cell is
    // the mean of its two node rows -- taken in that form, which rounds once.
    auto centroidY = [&](int i, int j, int k) {
        return 0.5 * (m.node(i, j, k)[1] + m.node(i, j + 1, k)[1]);
    };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const int js = std::clamp(j, 1, g.ny - 2); // ghost -> mirrored row
                const double y = centroidY(i, js, k);
                d[g.idx(i, j, k)] = std::min(y, 2.0 * delta - y);
            }
    return d;
}

std::vector<double> immersedWallDistance(const Grid& g, const Sdf& sdf,
                                         const AxisRemap3& remap, int threads)
{
    const double floor = 1e-9 * std::min({g.hx, g.hy, g.hz});
    std::vector<double> d(g.size());
    auto row = [&](int r) {
        const int j = r % g.ny, k = r / g.ny;
        for (int i = 0; i < g.nx; ++i) {
            const double X[3] = {(i + 0.5) * g.hx, (j + 0.5) * g.hy,
                                 (k + 0.5) * g.hz};
            double x[3];
            remap.map(X, x);
            d[g.idx(i, j, k)] = std::max(std::abs(sdf(x[0], x[1], x[2])), floor);
        }
    };
    const int rows = g.ny * g.nz;
    const int nt = std::max(
        1, std::min(threads > 0 ? threads
                                : int(std::thread::hardware_concurrency()),
                    rows));
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < nt; ++t)
            pool.emplace_back([&, t] {
                for (int r = t; r < rows; r += nt) row(r);
            });
    }
    return d;
}

} // namespace dns
