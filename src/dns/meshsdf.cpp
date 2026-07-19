#include "dns/meshsdf.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <numeric>
#include <unordered_map>

namespace dns {

namespace {

using A3 = std::array<double, 3>;

A3 sub(const A3& a, const A3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
A3 add(const A3& a, const A3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
A3 mul(const A3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
double dot(const A3& a, const A3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
A3 cross(const A3& a, const A3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
double norm(const A3& a) { return std::sqrt(dot(a, a)); }
A3 normalized(const A3& a)
{
    const double n = norm(a);
    return n > 0 ? mul(a, 1.0 / n) : A3{0, 0, 0};
}

// Closest point on triangle abc to p (Ericson, RTCD 5.1.5), reporting the
// owning feature: 0..2 vertex a/b/c, 3..5 edge ab/bc/ca, 6 interior.
A3 closestOnTri(const A3& p, const A3& a, const A3& b, const A3& c, int& feat)
{
    const A3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0 && d2 <= 0) { feat = 0; return a; }
    const A3 bp = sub(p, b);
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) { feat = 1; return b; }
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0)
    {
        feat = 3;
        return add(a, mul(ab, d1 / (d1 - d3)));
    }
    const A3 cp = sub(p, c);
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) { feat = 2; return c; }
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0)
    {
        feat = 5;
        return add(a, mul(ac, d2 / (d2 - d6)));
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0)
    {
        feat = 4;
        return add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    const double denom = 1.0 / (va + vb + vc);
    feat = 6;
    return add(a, add(mul(ab, vb * denom), mul(ac, vc * denom)));
}

double boxD2(const A3& lo, const A3& hi, const A3& p)
{
    double s = 0;
    for (int a = 0; a < 3; ++a)
    {
        const double d = std::max({lo[a] - p[a], 0.0, p[a] - hi[a]});
        s += d * d;
    }
    return s;
}

std::uint64_t edgeKey(int a, int b)
{
    return (std::uint64_t(std::uint32_t(std::min(a, b))) << 32)
         | std::uint32_t(std::max(a, b));
}

} // namespace

MeshSdf::MeshSdf(TriMesh mesh, bool robust)
    : verts_(std::move(mesh.verts)), robust_(robust)
{
    assert(!verts_.empty() && !mesh.tris.empty());
    A3 lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
    for (const auto& v : verts_)
        for (int a = 0; a < 3; ++a)
        {
            lo[a] = std::min(lo[a], v[a]);
            hi[a] = std::max(hi[a], v[a]);
        }
    const double diag2 = dot(sub(hi, lo), sub(hi, lo));

    // Keep only triangles with area (degenerates break normals, add nothing).
    for (const auto& t : mesh.tris)
    {
        const A3 cr = cross(sub(verts_[t[1]], verts_[t[0]]),
                            sub(verts_[t[2]], verts_[t[0]]));
        if (norm(cr) > 1e-14 * diag2) tris_.push_back(t);
    }
    assert(!tris_.empty());

    // Watertight: each directed edge once, its opposite present.
    {
        std::unordered_map<std::uint64_t, std::array<int, 2>> cnt;
        for (const auto& t : tris_)
            for (int e = 0; e < 3; ++e)
            {
                const int a = t[e], b = t[(e + 1) % 3];
                ++cnt[edgeKey(a, b)][a < b ? 0 : 1];
            }
        watertight_ = std::ranges::all_of(cnt, [](const auto& kv) {
            return kv.second[0] == 1 && kv.second[1] == 1;
        });
    }

    // Feature pseudonormals: face = geometric normal, vertex = angle-weighted
    // sum of incident face normals, edge = sum of the two adjacent faces.
    faceN_.resize(tris_.size());
    vertN_.assign(verts_.size(), {0, 0, 0});
    std::unordered_map<std::uint64_t, A3> edgeAcc;
    for (std::size_t t = 0; t < tris_.size(); ++t)
    {
        const auto& tr = tris_[t];
        const A3 &a = verts_[tr[0]], &b = verts_[tr[1]], &c = verts_[tr[2]];
        faceN_[t] = normalized(cross(sub(b, a), sub(c, a)));
        const A3* v[3]{&a, &b, &c};
        for (int e = 0; e < 3; ++e)
        {
            const A3 e1 = sub(*v[(e + 1) % 3], *v[e]);
            const A3 e2 = sub(*v[(e + 2) % 3], *v[e]);
            const double ang = std::atan2(norm(cross(e1, e2)), dot(e1, e2));
            vertN_[tr[e]] = add(vertN_[tr[e]], mul(faceN_[t], ang));
            auto [it, fresh] =
                edgeAcc.try_emplace(edgeKey(tr[e], tr[(e + 1) % 3]), A3{0, 0, 0});
            it->second = add(it->second, faceN_[t]);
        }
    }
    for (auto& n : vertN_) n = normalized(n);
    edgeN_.resize(tris_.size());
    for (std::size_t t = 0; t < tris_.size(); ++t)
        for (int e = 0; e < 3; ++e)
            edgeN_[t][e] =
                normalized(edgeAcc.at(edgeKey(tris_[t][e], tris_[t][(e + 1) % 3])));

    // Median-split BVH over triangle centroids.
    std::vector<A3> cent(tris_.size());
    for (std::size_t t = 0; t < tris_.size(); ++t)
        cent[t] = mul(add(add(verts_[tris_[t][0]], verts_[tris_[t][1]]),
                          verts_[tris_[t][2]]),
                      1.0 / 3.0);
    order_.resize(tris_.size());
    std::iota(order_.begin(), order_.end(), 0);
    nodes_.reserve(2 * tris_.size() / 4 + 2);
    build(0, int(tris_.size()), cent);
}

int MeshSdf::build(int begin, int end, const std::vector<A3>& cent)
{
    const int ni = int(nodes_.size());
    Node nd{{1e300, 1e300, 1e300}, {-1e300, -1e300, -1e300}, -1, -1, 0, 0};
    A3 clo{1e300, 1e300, 1e300}, chi{-1e300, -1e300, -1e300};
    for (int i = begin; i < end; ++i)
        for (int v = 0; v < 3; ++v)
        {
            const A3& p = verts_[tris_[order_[i]][v]];
            for (int a = 0; a < 3; ++a)
            {
                nd.lo[a] = std::min(nd.lo[a], p[a]);
                nd.hi[a] = std::max(nd.hi[a], p[a]);
            }
        }
    for (int i = begin; i < end; ++i)
        for (int a = 0; a < 3; ++a)
        {
            clo[a] = std::min(clo[a], cent[order_[i]][a]);
            chi[a] = std::max(chi[a], cent[order_[i]][a]);
        }
    nodes_.push_back(nd);
    if (end - begin <= 8)
    {
        nodes_[ni].start = begin;
        nodes_[ni].count = end - begin;
        return ni;
    }
    const A3 ext = sub(chi, clo);
    const int axis =
        int(std::ranges::max_element(ext) - ext.begin());
    const int mid = (begin + end) / 2;
    std::nth_element(order_.begin() + begin, order_.begin() + mid,
                     order_.begin() + end, [&](int s, int t) {
                         return cent[s][axis] < cent[t][axis];
                     });
    const int a = build(begin, mid, cent);
    const int b = build(mid, end, cent);
    nodes_[ni].a = a;
    nodes_[ni].b = b;
    return ni;
}

void MeshSdf::nearest(int ni, const A3& p, Hit& h) const
{
    const Node& nd = nodes_[ni];
    if (nd.count > 0)
    {
        for (int i = nd.start; i < nd.start + nd.count; ++i)
        {
            const int t = order_[i];
            int feat;
            const A3 cp = closestOnTri(p, verts_[tris_[t][0]], verts_[tris_[t][1]],
                                       verts_[tris_[t][2]], feat);
            const A3 d = sub(p, cp);
            const double d2 = dot(d, d);
            if (d2 < h.d2) h = {d2, t, cp, feat};
        }
        return;
    }
    int c0 = nd.a, c1 = nd.b;
    double d0 = boxD2(nodes_[c0].lo, nodes_[c0].hi, p);
    double d1 = boxD2(nodes_[c1].lo, nodes_[c1].hi, p);
    if (d1 < d0)
    {
        std::swap(c0, c1);
        std::swap(d0, d1);
    }
    if (d0 < h.d2) nearest(c0, p, h);
    if (d1 < h.d2) nearest(c1, p, h);
}

double MeshSdf::operator()(double x, double y, double z) const
{
    const A3 p{x, y, z};
    Hit h{std::numeric_limits<double>::max(), -1, {}, 6};
    nearest(0, p, h);
    const double d = std::sqrt(h.d2);
    if (robust_) return winding(x, y, z) > 0.5 ? -d : d;
    const auto& t = tris_[h.tri];
    const A3& n = h.feat == 6 ? faceN_[h.tri]
                : h.feat < 3  ? vertN_[t[h.feat]]
                              : edgeN_[h.tri][h.feat - 3];
    return dot(sub(p, h.cp), n) < 0 ? -d : d;
}

double MeshSdf::winding(double x, double y, double z) const
{
    const A3 p{x, y, z};
    double sum = 0;
    for (const auto& t : tris_)
    {
        const A3 a = sub(verts_[t[0]], p), b = sub(verts_[t[1]], p),
                 c = sub(verts_[t[2]], p);
        const double la = norm(a), lb = norm(b), lc = norm(c);
        const double den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la
                         + dot(c, a) * lb;
        sum += 2.0 * std::atan2(dot(a, cross(b, c)), den);
    }
    return sum / (4.0 * std::numbers::pi);
}


} // namespace dns
