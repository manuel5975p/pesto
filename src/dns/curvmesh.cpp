#include "dns/curvmesh.hpp"

#include <cassert>
#include <cmath>
#include <format>

namespace dns {
namespace {

struct V3
{
    double x = 0, y = 0, z = 0;
};

V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V3 operator*(double s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V3 cross(V3 a, V3 b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

int floorDiv(int i, int n)
{
    int q = i / n;
    if (i % n < 0) --q;
    return q;
}

// One axis of node(), resolved: the stored rows this index reads, their
// weights, and the box image added to that axis' coordinate. Two rows are
// enough -- Extend is the only mode that needs more than one.
struct AxisTerms
{
    int idx[2] = {0, 0};
    double w[2] = {1.0, 0.0};
    int n = 1;          // active terms
    double shift = 0.0; // Translate image, added to this axis' coordinate
};

AxisTerms axisTerms(AxisWrap mode, int i, int n, double h)
{
    if (mode == AxisWrap::Extend) {
        assert(n >= 2); // build() rejects a degenerate Extend axis
        // node(i) = node(0) + i (node(1) - node(0)) below the bottom, and
        // node(n-1) + s (node(n-1) - node(n-2)), s = i-n+1, above the top.
        if (i < 0) return {{0, 1}, {1.0 - i, double(i)}, 2, 0.0};
        if (i >= n) {
            const double s = i - n + 1;
            return {{n - 1, n - 2}, {1.0 + s, -s}, 2, 0.0};
        }
        return {{i, 0}, {1.0, 0.0}, 1, 0.0};
    }
    const int q = floorDiv(i, n);
    const double shift = mode == AxisWrap::Translate ? q * n * h : 0.0;
    return {{i - q * n, 0}, {1.0, 0.0}, 1, shift};
}

// Signed volume of tetra (o, a, b, c) times 6.
double tet6(V3 o, V3 a, V3 b, V3 c) { return dot(cross(a - o, b - o), c - o); }

// Row-major inverse of m; false when |det| vanishes or flips sign.
bool invert3(const double m[3][3], double inv[3][3], double& det)
{
    inv[0][0] = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    inv[0][1] = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    inv[0][2] = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    inv[1][0] = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    inv[1][1] = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    inv[1][2] = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    inv[2][0] = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    inv[2][1] = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    inv[2][2] = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    det = m[0][0] * inv[0][0] + m[0][1] * inv[1][0] + m[0][2] * inv[2][0];
    if (!(det > 0.0)) return false;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) inv[r][c] /= det;
    return true;
}

} // namespace

CurvMesh CurvMesh::uniform(const Grid& g)
{
    CurvMesh m{g, {}, {}, {}};
    const std::size_t n = g.size();
    m.x.resize(n);
    m.y.resize(n);
    m.z.resize(n);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                m.x[c] = i * g.hx;
                m.y[c] = j * g.hy;
                m.z[c] = k * g.hz;
            }
    return m;
}

CurvMesh CurvMesh::sinusoidal(const Grid& g, double amp)
{
    return fromRemap(g, SinusoidalRemap{g.nx * g.hx, g.ny * g.hy, g.nz * g.hz,
                                        amp, g.nz == 1});
}

CurvMesh CurvMesh::stretchedY(const Grid& g, double beta)
{
    return fromRemap(g, TanhStretchY{g.ny * g.hy, beta});
}

CurvMesh CurvMesh::channelY(const Grid& g, double beta, double delta)
{
    return fromRemap(g, ChannelStretchY{g.ny * g.hy, g.ny, beta, delta});
}

std::array<double, 3> CurvMesh::node(int i, int j, int k) const
{
    const AxisTerms A = axisTerms(wrap[0], i, g.nx, g.hx);
    const AxisTerms B = axisTerms(wrap[1], j, g.ny, g.hy);
    const AxisTerms C = axisTerms(wrap[2], k, g.nz, g.hz);
    // The (0,0,0) term carries weight exactly 1 whenever every axis is
    // single-term, so all-Translate reproduces `x[c] + q nx hx` bit-for-bit.
    const std::size_t c0 = g.idx(A.idx[0], B.idx[0], C.idx[0]);
    const double w0 = A.w[0] * B.w[0] * C.w[0];
    double p[3] = {w0 * x[c0], w0 * y[c0], w0 * z[c0]};
    for (int c = 0; c < C.n; ++c)
        for (int b = 0; b < B.n; ++b)
            for (int a = 0; a < A.n; ++a) {
                if (a == 0 && b == 0 && c == 0) continue;
                const std::size_t s = g.idx(A.idx[a], B.idx[b], C.idx[c]);
                const double w = A.w[a] * B.w[b] * C.w[c];
                p[0] += w * x[s];
                p[1] += w * y[s];
                p[2] += w * z[s];
            }
    return {p[0] + A.shift, p[1] + B.shift, p[2] + C.shift};
}

std::expected<CurvMetrics, std::string> CurvMetrics::build(const CurvMesh& m)
{
    const Grid& g = m.g;
    const std::size_t n = g.size();
    if (g.nx < 2 || g.ny < 2)
        return std::unexpected("curvilinear mesh needs nx, ny >= 2");
    if (m.x.size() != n || m.y.size() != n || m.z.size() != n)
        return std::unexpected(
            std::format("node arrays must hold {} entries (one per cell)", n));
    if (g.nz == 1)
        for (std::size_t c = 0; c < n; ++c)
            if (m.z[c] != m.z[0])
                return std::unexpected(
                    "nz == 1 requires a planar node layer (constant z), or the "
                    "exact 2D restriction breaks");
    const int na[3] = {g.nx, g.ny, g.nz};
    for (int a = 0; a < 3; ++a)
        if (m.wrap[a] == AxisWrap::Extend && na[a] < 2)
            return std::unexpected(std::format(
                "axis {} is Extend but has {} cell(s): linear extension needs "
                "two node rows",
                a, na[a]));

    CurvMetrics t;
    t.g = g;
    t.wrap = m.wrap;
    for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c) t.S[a][c].assign(n, 0.0);
        for (int e = 0; e < 9; ++e) t.Ji[a][e].assign(n, 0.0);
        t.aov[a].assign(n, 0.0);
    }
    t.vol.assign(n, 0.0);

    auto nd = [&](int i, int j, int k) -> V3 {
        const auto p = m.node(i, j, k);
        return {p[0], p[1], p[2]};
    };
    auto centroid = [&](int i, int j, int k) {
        V3 s{};
        for (int dk = 0; dk < 2; ++dk)
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di) s = s + nd(i + di, j + dj, k + dk);
        return 0.125 * s;
    };
    // 4 corners of the face at offset `off` (0 = lo, 1 = hi) along axis a,
    // ordered P00, P10, P11, P01 in the (t1, t2) tangent plane.
    auto face = [&](int i, int j, int k, int a, int off, V3 P[4]) {
        const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
        int o[3] = {0, 0, 0};
        o[a] = off;
        auto corner = [&](int d1, int d2) {
            int e[3] = {o[0], o[1], o[2]};
            e[t1] += d1;
            e[t2] += d2;
            return nd(i + e[0], j + e[1], k + e[2]);
        };
        P[0] = corner(0, 0);
        P[1] = corner(1, 0);
        P[2] = corner(1, 1);
        P[3] = corner(0, 1);
    };

    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const V3 o = nd(i, j, k);
                double v6 = 0.0;
                const V3 cc0 = centroid(i, j, k);
                for (int a = 0; a < 3; ++a) {
                    V3 P[4];
                    face(i, j, k, a, 1, P);
                    // Exact bilinear-quad area vector, outward +a.
                    const V3 Sv = 0.5 * cross(P[2] - P[0], P[3] - P[1]);
                    t.S[a][0][c] = Sv.x;
                    t.S[a][1][c] = Sv.y;
                    t.S[a][2][c] = Sv.z;
                    // Diagonal P0-P2 on every face: neighbours triangulate the
                    // shared surface identically, so the cells tile the box.
                    v6 += tet6(o, P[0], P[1], P[2]) + tet6(o, P[0], P[2], P[3]);
                    V3 Q[4];
                    face(i, j, k, a, 0, Q);
                    v6 += tet6(o, Q[0], Q[2], Q[1]) + tet6(o, Q[0], Q[3], Q[2]);

                    // dx/dxi at the hi face: normal from cell centroids,
                    // tangents from face edge midpoints (dxi = 1).
                    const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
                    const V3 cc1 = centroid(i + (a == 0), j + (a == 1), k + (a == 2));
                    V3 col[3];
                    col[a] = cc1 - cc0;
                    col[t1] = 0.5 * ((P[1] + P[2]) - (P[0] + P[3]));
                    col[t2] = 0.5 * ((P[3] + P[2]) - (P[0] + P[1]));
                    const double M[3][3] = {{col[0].x, col[1].x, col[2].x},
                                            {col[0].y, col[1].y, col[2].y},
                                            {col[0].z, col[1].z, col[2].z}};
                    double inv[3][3], det = 0.0;
                    if (!invert3(M, inv, det))
                        return std::unexpected(std::format(
                            "degenerate face Jacobian at cell ({}, {}, {}) axis "
                            "{} (det {})",
                            i, j, k, a, det));
                    for (int d = 0; d < 3; ++d)
                        for (int cx = 0; cx < 3; ++cx)
                            t.Ji[a][d * 3 + cx][c] = inv[d][cx];
                }
                const double vol = v6 / 6.0;
                if (!(vol > 0.0))
                    return std::unexpected(std::format(
                        "non-positive cell volume at ({}, {}, {}): {}", i, j, k,
                        vol));
                t.vol[c] = vol;
            }

    // Widths need the neighbour's face norms, hence the second pass.
    t.hmin = 1e300;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                for (int a = 0; a < 3; ++a) {
                    const int li[3] = {i - (a == 0), j - (a == 1), k - (a == 2)};
                    // An Extend axis has no lo neighbour at index 0: the wrap
                    // would read the far end's face. Clamp to the cell's own
                    // face, the only local width there is.
                    const bool clamp =
                        m.wrap[a] == AxisWrap::Extend && li[a] < 0;
                    const std::size_t lo =
                        clamp ? c : g.idx(li[0], li[1], li[2]);
                    auto norm = [&](std::size_t f) {
                        return std::sqrt(t.S[a][0][f] * t.S[a][0][f] +
                                         t.S[a][1][f] * t.S[a][1][f] +
                                         t.S[a][2][f] * t.S[a][2][f]);
                    };
                    t.aov[a][c] = 0.5 * (norm(c) + norm(lo)) / t.vol[c];
                    if (a < 2 || g.nz > 1)
                        t.hmin = std::min(t.hmin, 1.0 / t.aov[a][c]);
                }
            }
    return t;
}

} // namespace dns
