#include "dns/fvwalldist.hpp"

#include "dns/parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <numeric>

namespace dns {
namespace {

using P3 = std::array<double, 3>;
constexpr double kSmall = 1e-15;      // OpenFOAM SMALL
constexpr double kGreat = 1e15;       // OpenFOAM GREAT
constexpr double kRootVSmall = 1e-150; // OpenFOAM ROOTVSMALL (double)
constexpr double kVSmall = 1e-300;
constexpr double kPropagationTol = 0.01; // FaceCellWaveBase::propagationTol_

P3 sub(const P3& a, const P3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
P3 add(const P3& a, const P3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
P3 scale(double s, const P3& a) { return {s * a[0], s * a[1], s * a[2]}; }
double dot(const P3& a, const P3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double magSqr(const P3& a) { return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]; }
double mag(const P3& a) { return std::sqrt(magSqr(a)); }
P3 cross(const P3& a, const P3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// triangle::nearestPointClassify's distance (Ericson, OpenFOAM's branch order).
double triDist(const P3& p, const P3& a, const P3& b, const P3& c)
{
    const P3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return mag(sub(a, p));
    const P3 bp = sub(p, b);
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return mag(sub(b, p));
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        if ((d1 - d3) < kRootVSmall) return mag(sub(a, p));
        const double v = d1 / (d1 - d3);
        return mag(sub(add(a, scale(v, ab)), p));
    }
    const P3 cp = sub(p, c);
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return mag(sub(c, p));
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        if ((d2 - d6) < kRootVSmall) return mag(sub(a, p));
        const double w = d2 / (d2 - d6);
        return mag(sub(add(a, scale(w, ac)), p));
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        if (((d4 - d3) + (d5 - d6)) < kRootVSmall) return mag(sub(b, p));
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return mag(sub(add(b, scale(w, sub(c, b))), p));
    }
    if ((va + vb + vc) < kRootVSmall) {
        const P3 ctr = scale(1.0 / 3.0, add(add(a, b), c));
        return mag(sub(ctr, p));
    }
    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom, w = vc * denom;
    const P3 q = add(add(a, scale(v, ab)), scale(w, ac));
    return mag(sub(q, p));
}

struct Poly
{
    const PolyMesh& pm;
    P3 pt(std::int32_t i) const { return {pm.px[std::size_t(i)], pm.py[std::size_t(i)], pm.pz[std::size_t(i)]}; }
    int size(std::int64_t f) const { return pm.faceStart[std::size_t(f + 1)] - pm.faceStart[std::size_t(f)]; }
    std::int32_t node(std::int64_t f, int i) const { return pm.faceNodes[std::size_t(pm.faceStart[std::size_t(f)] + i)]; }
    // face::centre (area-weighted fan about the point average, unprojected areas).
    P3 centre(std::int64_t f) const
    {
        const int n = size(f);
        if (n == 3) return scale(1.0 / 3.0, add(add(pt(node(f, 0)), pt(node(f, 1))), pt(node(f, 2))));
        P3 avg{0, 0, 0};
        for (int i = 0; i < n; ++i) avg = add(avg, pt(node(f, i)));
        avg = scale(1.0 / double(n), avg);
        double sumA = 0;
        P3 sumAc{0, 0, 0};
        for (int i = 0; i < n; ++i) {
            const P3 a = pt(node(f, i)), b = pt(node(f, i == n - 1 ? 0 : i + 1));
            const P3 ttc = add(add(a, b), avg);
            const double ta = mag(cross(sub(a, avg), sub(b, avg)));
            sumA += ta;
            sumAc = add(sumAc, scale(ta, ttc));
        }
        if (sumA > kVSmall) return scale(1.0 / (3.0 * sumA), sumAc);
        return avg;
    }
    // face::nearestPoint(p).distance()
    double faceDist(std::int64_t f, const P3& p) const
    {
        const int n = size(f);
        if (n == 3) return triDist(p, pt(node(f, 0)), pt(node(f, 1)), pt(node(f, 2)));
        const P3 ctr = centre(f);
        double best = kGreat;
        for (int i = 0; i < n; ++i) {
            const double d = triDist(p, pt(node(f, i)), pt(node(f, (i + 1) % n)), ctr);
            if (std::abs(d) < std::abs(best)) best = d;
        }
        return best;
    }
};

// The combined wall patch (every wall-type patch in boundary order): its mesh
// faces, meshPoints (order of first appearance), pointFaces (ascending local
// faces), and per local face its point neighbours (self + faces sharing a point).
struct WallPatch
{
    std::vector<std::int64_t> faces;
    std::vector<std::int32_t> meshPoints;
    std::vector<std::vector<int>> pointFaces;    // by local point
    std::vector<std::vector<int>> localFacePts;  // local point indices per local face
    std::vector<int> nbr(int lf) const
    {
        std::vector<int> out{lf};
        for (int lp : localFacePts[std::size_t(lf)])
            for (int g : pointFaces[std::size_t(lp)])
                if (std::ranges::find(out, g) == out.end()) out.push_back(g);
        return out;
    }
};

WallPatch wallPatch(const PolyMesh& pm, const Poly& P)
{
    WallPatch w;
    for (const FoamPatch& p : pm.man.patches)
        if (p.type == FoamPatchType::Wall)
            for (std::int64_t i = 0; i < p.nFaces; ++i) w.faces.push_back(p.startFace + i);
    std::vector<int> local(pm.px.size(), -1);
    w.localFacePts.resize(w.faces.size());
    for (std::size_t lf = 0; lf < w.faces.size(); ++lf) {
        const std::int64_t f = w.faces[lf];
        for (int i = 0; i < P.size(f); ++i) {
            const std::int32_t v = P.node(f, i);
            if (local[std::size_t(v)] < 0) {
                local[std::size_t(v)] = int(w.meshPoints.size());
                w.meshPoints.push_back(v);
                w.pointFaces.emplace_back();
            }
            w.localFacePts[lf].push_back(local[std::size_t(v)]);
        }
    }
    for (std::size_t lf = 0; lf < w.faces.size(); ++lf)
        for (int lp : w.localFacePts[lf]) {
            auto& pf = w.pointFaces[std::size_t(lp)];
            if (pf.empty() || pf.back() != int(lf)) pf.push_back(int(lf));
        }
    return w;
}

// smallestDist over local faces `set`.
double smallestDist(const Poly& P, const WallPatch& w, const P3& p, const std::vector<int>& set)
{
    double mn = kGreat;
    for (int lf : set) {
        const double d = P.faceDist(w.faces[std::size_t(lf)], p);
        if (d < mn) mn = d;
    }
    return mn;
}

// --- exact: nearest point over all wall triangles (an AABB tree) -------------
struct Tri
{
    P3 a, b, c;
};

struct Bvh
{
    struct Node
    {
        P3 lo, hi;
        int first = 0, count = 0, left = -1, right = -1;
    };
    std::vector<Tri> tris;
    std::vector<int> idx;
    std::vector<Node> nodes;

    static void grow(P3& lo, P3& hi, const P3& p)
    {
        for (int q = 0; q < 3; ++q) {
            lo[std::size_t(q)] = std::min(lo[std::size_t(q)], p[std::size_t(q)]);
            hi[std::size_t(q)] = std::max(hi[std::size_t(q)], p[std::size_t(q)]);
        }
    }
    int build(int first, int count)
    {
        Node nd;
        nd.lo = {kGreat, kGreat, kGreat};
        nd.hi = {-kGreat, -kGreat, -kGreat};
        for (int i = first; i < first + count; ++i) {
            const Tri& t = tris[std::size_t(idx[std::size_t(i)])];
            grow(nd.lo, nd.hi, t.a);
            grow(nd.lo, nd.hi, t.b);
            grow(nd.lo, nd.hi, t.c);
        }
        nd.first = first;
        nd.count = count;
        const int me = int(nodes.size());
        nodes.push_back(nd);
        if (count <= 4) return me;
        int ax = 0;
        for (int q = 1; q < 3; ++q)
            if (nd.hi[std::size_t(q)] - nd.lo[std::size_t(q)] > nd.hi[std::size_t(ax)] - nd.lo[std::size_t(ax)]) ax = q;
        auto key = [&](int i) {
            const Tri& t = tris[std::size_t(i)];
            return t.a[std::size_t(ax)] + t.b[std::size_t(ax)] + t.c[std::size_t(ax)];
        };
        const int mid = first + count / 2;
        std::nth_element(idx.begin() + first, idx.begin() + mid, idx.begin() + first + count,
                         [&](int x, int y) { return key(x) < key(y) || (key(x) == key(y) && x < y); });
        const int l = build(first, mid - first);
        const int r = build(mid, first + count - mid);
        nodes[std::size_t(me)].left = l;
        nodes[std::size_t(me)].right = r;
        nodes[std::size_t(me)].count = 0;
        return me;
    }
    static double boxDist2(const Node& n, const P3& p)
    {
        double s = 0;
        for (int q = 0; q < 3; ++q) {
            const double d = std::max({n.lo[std::size_t(q)] - p[std::size_t(q)], 0.0, p[std::size_t(q)] - n.hi[std::size_t(q)]});
            s += d * d;
        }
        return s;
    }
    double nearest(const P3& p) const
    {
        double best = kGreat;
        int stack[128];
        int sp = 0;
        stack[sp++] = 0;
        while (sp) {
            const Node& n = nodes[std::size_t(stack[--sp])];
            if (boxDist2(n, p) > best * best) continue;
            if (n.left < 0) {
                for (int i = n.first; i < n.first + n.count; ++i) {
                    const Tri& t = tris[std::size_t(idx[std::size_t(i)])];
                    best = std::min(best, triDist(p, t.a, t.b, t.c));
                }
                continue;
            }
            const double dl = boxDist2(nodes[std::size_t(n.left)], p), dr = boxDist2(nodes[std::size_t(n.right)], p);
            if (dl < dr) {
                stack[sp++] = n.right;
                stack[sp++] = n.left;
            } else {
                stack[sp++] = n.left;
                stack[sp++] = n.right;
            }
        }
        return best;
    }
};

} // namespace

std::expected<FvWallDist, std::string> fvWallDistance(const PolyMesh& pm, const FaceMesh& fm, bool exact,
                                                      bool correctWalls)
{
    const Poly P{pm};
    const std::size_t nCells = std::size_t(pm.man.nCells), nFaces = std::size_t(pm.man.nFaces);
    const std::size_t nInt = std::size_t(pm.man.nInternalFaces);
    if (std::size_t(fm.nC) != nCells) return std::unexpected("wall distance: FaceMesh and polyMesh cell counts differ");
    const bool cyclic = std::ranges::any_of(pm.man.patches, [](const FoamPatch& p) { return p.type == FoamPatchType::Cyclic; });
    const WallPatch w = wallPatch(pm, P);
    if (w.faces.empty()) return std::unexpected("wall distance: the mesh has no wall-type patch (RAS needs one)");

    // Cell centres in polyMesh order (the FaceMesh geometry: OpenFOAM's).
    std::vector<P3> cc(nCells);
    std::vector<int> fmCell(nCells, -1);
    for (std::size_t c = 0; c < nCells; ++c) {
        const std::size_t s = std::size_t(fm.srcCell[c]);
        fmCell[s] = int(c);
        cc[s] = {fm.C[0][c], fm.C[1][c], fm.C[2][c]};
    }
    FvWallDist out;
    std::vector<double> dist(nCells, 0.0);

    if (exact) {
        Bvh b;
        for (std::int64_t f : w.faces) {
            const int n = P.size(f);
            if (n == 3) {
                b.tris.push_back({P.pt(P.node(f, 0)), P.pt(P.node(f, 1)), P.pt(P.node(f, 2))});
                continue;
            }
            const P3 ctr = P.centre(f);
            for (int i = 0; i < n; ++i) b.tris.push_back({P.pt(P.node(f, i)), P.pt(P.node(f, (i + 1) % n)), ctr});
        }
        b.idx.resize(b.tris.size());
        std::iota(b.idx.begin(), b.idx.end(), 0);
        b.build(0, int(b.tris.size()));
        forBlocks(workerCount(0, nCells), nCells, [&](int, std::size_t lo, std::size_t hi) {
            for (std::size_t c = lo; c < hi; ++c) dist[c] = b.nearest(cc[c]);
        });
    } else {
        if (cyclic)
            return std::unexpected("wallDist meshWave on a mesh with cyclic patches: OpenFOAM's cyclic transfer of "
                                   "the wave is not reproduced (use method exactDistance)");
        // Face centres in polyMesh order.
        std::vector<P3> fc(nFaces);
        for (std::size_t f = 0; f < fm.srcFace.size(); ++f)
            fc[std::size_t(fm.srcFace[f])] = {fm.Cf[0][f], fm.Cf[1][f], fm.Cf[2][f]};
        // primitiveMesh::calcCells: per cell the faces it OWNS (ascending, the
        // boundary ones included), THEN the faces it neighbours (ascending) --
        // not one ascending list; the wave's update order (and with the 1%
        // propagation tolerance its result) depends on it.
        std::vector<int> cStart(nCells + 1, 0), cFace;
        for (std::size_t f = 0; f < nFaces; ++f) {
            ++cStart[std::size_t(pm.owner[f]) + 1];
            if (f < nInt) ++cStart[std::size_t(pm.neighbour[f]) + 1];
        }
        for (std::size_t c = 0; c < nCells; ++c) cStart[c + 1] += cStart[c];
        cFace.resize(std::size_t(cStart[nCells]));
        {
            std::vector<int> fill(cStart.begin(), cStart.end() - 1);
            for (std::size_t f = 0; f < nFaces; ++f) cFace[std::size_t(fill[std::size_t(pm.owner[f])]++)] = int(f);
            for (std::size_t f = 0; f < nInt; ++f) cFace[std::size_t(fill[std::size_t(pm.neighbour[f])]++)] = int(f);
        }
        // FaceCellWave<wallPoint>, serial: wallPoint {origin, distSqr}, invalid
        // distSqr = -GREAT (valid iff > -SMALL).
        std::vector<P3> fo(nFaces, {kGreat, kGreat, kGreat}), co(nCells, {kGreat, kGreat, kGreat});
        std::vector<double> fd(nFaces, -kGreat), cd(nCells, -kGreat);
        std::vector<std::uint8_t> fChanged(nFaces, 0), cChanged(nCells, 0);
        std::vector<int> changedFaces, changedCells;
        for (std::int64_t f : w.faces) {
            fo[std::size_t(f)] = fc[std::size_t(f)];
            fd[std::size_t(f)] = 0.0;
            fChanged[std::size_t(f)] = 1;
            changedFaces.push_back(int(f));
        }
        // wallPoint::update(pt, w2, tol)
        auto update = [](const P3& pt, const P3& origin, P3& myO, double& myD) {
            const double dist2 = magSqr(sub(pt, origin));
            if (!(myD > -kSmall)) {
                myD = dist2;
                myO = origin;
                return true;
            }
            const double diff = myD - dist2;
            if (diff < 0) return false;
            if ((diff < kSmall) || ((myD > kSmall) && (diff / myD < kPropagationTol))) return false;
            myD = dist2;
            myO = origin;
            return true;
        };
        const long maxIter = long(nCells) + 1;
        for (long iter = 0; iter < maxIter; ++iter) {
            // faceToCell
            for (int f : changedFaces) {
                const P3 o = fo[std::size_t(f)];
                const int own = pm.owner[std::size_t(f)];
                if (co[std::size_t(own)] != o && update(cc[std::size_t(own)], o, co[std::size_t(own)], cd[std::size_t(own)]))
                    if (!cChanged[std::size_t(own)]) {
                        cChanged[std::size_t(own)] = 1;
                        changedCells.push_back(own);
                    }
                if (std::size_t(f) < nInt) {
                    const int nei = pm.neighbour[std::size_t(f)];
                    if (co[std::size_t(nei)] != o && update(cc[std::size_t(nei)], o, co[std::size_t(nei)], cd[std::size_t(nei)]))
                        if (!cChanged[std::size_t(nei)]) {
                            cChanged[std::size_t(nei)] = 1;
                            changedCells.push_back(nei);
                        }
                }
                fChanged[std::size_t(f)] = 0;
            }
            changedFaces.clear();
            const std::size_t nC = changedCells.size();
            if (nC == 0) break;
            // cellToFace
            for (int c : changedCells) {
                const P3 o = co[std::size_t(c)];
                for (int k = cStart[std::size_t(c)]; k < cStart[std::size_t(c) + 1]; ++k) {
                    const int f = cFace[std::size_t(k)];
                    if (fo[std::size_t(f)] != o && update(fc[std::size_t(f)], o, fo[std::size_t(f)], fd[std::size_t(f)]))
                        if (!fChanged[std::size_t(f)]) {
                            fChanged[std::size_t(f)] = 1;
                            changedFaces.push_back(f);
                        }
                }
                cChanged[std::size_t(c)] = 0;
            }
            changedCells.clear();
            if (changedFaces.empty()) break;
        }
        for (std::size_t c = 0; c < nCells; ++c) {
            if (cd[c] > -kSmall) dist[c] = std::sqrt(cd[c]);
            else {
                dist[c] = cd[c];
                ++out.nUnset;
            }
        }
        if (out.nUnset)
            return std::unexpected(std::format("wallDist meshWave: {} cells not reached by the wave (a region "
                                               "without a wall)",
                                               out.nUnset));
        if (correctWalls) { // cellDistFuncs::correctBoundaryCells(walls, doPointCells = true)
            std::vector<std::uint8_t> nearest(nCells, 0);
            for (std::size_t lf = 0; lf < w.faces.size(); ++lf) {
                const std::size_t c = std::size_t(pm.owner[std::size_t(w.faces[lf])]);
                dist[c] = smallestDist(P, w, cc[c], w.nbr(int(lf)));
                nearest[c] = 1;
            }
            // pointCells, ascending cell index per point
            std::vector<int> pStart(pm.px.size() + 1, 0), pCell;
            {
                std::vector<std::vector<int>> pc(pm.px.size());
                for (std::size_t c = 0; c < nCells; ++c)
                    for (int k = cStart[c]; k < cStart[c + 1]; ++k) {
                        const std::int64_t f = cFace[std::size_t(k)];
                        for (int i = 0; i < P.size(f); ++i) {
                            auto& v = pc[std::size_t(P.node(f, i))];
                            if (v.empty() || v.back() != int(c)) v.push_back(int(c));
                        }
                    }
                for (std::size_t p = 0; p < pc.size(); ++p) pStart[p + 1] = pStart[p] + int(pc[p].size());
                pCell.reserve(std::size_t(pStart.back()));
                for (const auto& v : pc) pCell.insert(pCell.end(), v.begin(), v.end());
            }
            for (std::size_t lp = 0; lp < w.meshPoints.size(); ++lp) {
                const std::size_t v = std::size_t(w.meshPoints[lp]);
                for (int k = pStart[v]; k < pStart[v + 1]; ++k) {
                    const std::size_t c = std::size_t(pCell[std::size_t(k)]);
                    if (nearest[c]) continue;
                    dist[c] = smallestDist(P, w, cc[c], w.pointFaces[lp]);
                    nearest[c] = 1;
                }
            }
        }
    }
    out.y.resize(nCells);
    for (std::size_t s = 0; s < nCells; ++s) out.y[std::size_t(fmCell[s])] = dist[s];

    // nearWallDist (combined wall patch): per wall face.
    std::vector<int> fmFace(nFaces, -1);
    for (std::size_t f = 0; f < fm.srcFace.size(); ++f)
        if (int(f) >= fm.nI) fmFace[std::size_t(fm.srcFace[f])] = int(f);
    out.yWall.assign(std::size_t(fm.nB), 0.0);
    for (std::size_t lf = 0; lf < w.faces.size(); ++lf) {
        const std::int64_t f = w.faces[lf];
        const int g = fmFace[std::size_t(f)];
        if (g < 0) return std::unexpected("wall distance: a wall face is not a FaceMesh boundary face");
        const std::size_t c = std::size_t(pm.owner[std::size_t(f)]);
        out.yWall[std::size_t(g - fm.nI)] = smallestDist(P, w, cc[c], w.nbr(int(lf)));
    }
    return out;
}

} // namespace dns
