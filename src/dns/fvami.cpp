#include "dns/fvami.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>

// OpenFOAM's operand order everywhere in this file; its build (gcc -O3, x86-64)
// contracts nothing, neither may this one: compile with -ffp-contract=off.
// File:line references are OpenFOAM-v2606/src.

namespace dns {
namespace {

using V3 = std::array<double, 3>;

V3 add(const V3& a, const V3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V3 sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V3 mul(double s, const V3& a) { return {s * a[0], s * a[1], s * a[2]}; }
V3 div(const V3& a, double s) { return {a[0] / s, a[1] / s, a[2] / s}; }
V3 neg(const V3& a) { return {-a[0], -a[1], -a[2]}; }
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V3 cross(const V3& a, const V3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double magSqr(const V3& a) { return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]; }
double mag(const V3& a) { return std::sqrt(magSqr(a)); }
double sqr(double x) { return x * x; }

constexpr double kGreat = 1e15, kVSmall = 1e-300, kRootVSmall = 1e-150;
constexpr double kPi = std::numbers::pi;      // constant::mathematical::pi = M_PI
constexpr double kInterTol = 1e-6;            // faceAreaIntersect::tol
constexpr double kRestartMinWeight = 0.95;    // faceAreaWeightAMI::restartUncoveredSourceFace
constexpr double kMaxBoundsError = 0.05;      // advancingFrontAMI::checkPatches

// Vector::normalise(ROOTVSMALL)
V3 normalised(const V3& v)
{
    const double s = mag(v);
    if (s < kRootVSmall) return {0, 0, 0};
    return div(v, s);
}
double clamp1(double x) { return std::min(std::max(x, -1.0), 1.0); } // Foam::clamp(x, -1, 1)

// triangle::areaNormal
V3 triAreaNormal(const V3& a, const V3& b, const V3& c) { return mul(0.5, cross(sub(b, a), sub(c, a))); }

using Face = std::span<const std::int32_t>;
using Points = std::vector<V3>;

// face::average(points)
V3 faceAverage(Face f, const Points& p)
{
    V3 avg{0, 0, 0};
    for (const std::int32_t i : f) avg = add(avg, p[i]);
    return div(avg, double(f.size()));
}
// face::areaNormal(points): the fan about the point average (a triangle directly).
V3 faceAreaNormal(Face f, const Points& p)
{
    const std::size_t nP = f.size();
    if (nP == 3) return triAreaNormal(p[f[0]], p[f[1]], p[f[2]]);
    const V3 centrePoint = faceAverage(f, p);
    V3 n{0, 0, 0};
    for (std::size_t pI = 0; pI < nP; ++pI)
        n = add(n, triAreaNormal(p[f[pI]], p[f[pI == nP - 1 ? 0 : pI + 1]], centrePoint));
    return n;
}
// face::centre(points)
V3 faceCentre(Face f, const Points& p)
{
    const std::size_t nP = f.size();
    if (nP == 3) return mul(1.0 / 3.0, add(add(p[f[0]], p[f[1]]), p[f[2]]));
    const V3 centrePoint = faceAverage(f, p);
    double sumA = 0;
    V3 sumAc{0, 0, 0};
    for (std::size_t pI = 0; pI < nP; ++pI) {
        const V3& thisPoint = p[f[pI]];
        const V3& nextPoint = p[f[pI == nP - 1 ? 0 : pI + 1]];
        const V3 ttc = add(add(thisPoint, nextPoint), centrePoint);
        const double ta = mag(cross(sub(thisPoint, centrePoint), sub(nextPoint, centrePoint)));
        sumA += ta;
        sumAc = add(sumAc, mul(ta, ttc));
    }
    if (sumA > kVSmall) return div(sumAc, 3.0 * sumA);
    return centrePoint;
}

// face::mostConcaveAngle (OpenFOAM/meshes/meshShapes/face/face.C:58)
int mostConcaveAngle(Face f, const Points& p, const std::vector<V3>& edges, double& maxAngle)
{
    const V3 n = faceAreaNormal(f, p);
    const int nE = int(edges.size());
    int index = 0;
    maxAngle = -kGreat;
    for (int i = 0; i < nE; ++i) {
        const V3& leftEdge = edges[i == 0 ? nE - 1 : i - 1];
        const V3& rightEdge = edges[i];
        const V3 edgeNormal = cross(rightEdge, leftEdge);
        const double edgeCos = dot(leftEdge, rightEdge);
        const double edgeAngle = std::acos(clamp1(edgeCos));
        const double angle = dot(edgeNormal, n) > 0 ? kPi + edgeAngle : kPi - edgeAngle;
        if (maxAngle < angle) {
            maxAngle = angle;
            index = i;
        }
    }
    return index;
}

// face::split(SPLITTRIANGLE, ...) (face.C:106): what face::triangles(points,
// DynamicList<face>&) appends, `triMode mesh`.
void faceTriangles(Face f, const Points& p, std::vector<std::array<std::int32_t, 3>>& out)
{
    const int n = int(f.size());
    if (n == 3) {
        out.push_back({f[0], f[1], f[2]});
        return;
    }
    std::vector<V3> edges(static_cast<std::size_t>(n)); // face::calcEdgeVectors
    for (int i = 0; i < n; ++i) edges[i] = normalised(sub(p[f[(i + 1) % n]], p[f[i]]));
    if (n == 4) {
        double minAngle;
        const int startIndex = mostConcaveAngle(f, p, edges, minAngle);
        const int nextIndex = (startIndex + 1) % n, splitIndex = (nextIndex + 1) % n;
        out.push_back({f[startIndex], f[nextIndex], f[splitIndex]});
        out.push_back({f[splitIndex], f[(splitIndex + 1) % n], f[startIndex]});
        return;
    }
    double minAngle = 1;
    const int startIndex = mostConcaveAngle(f, p, edges, minAngle);
    const double bisectAngle = minAngle / 2;
    const V3& rightEdge = edges[startIndex];
    int index = (startIndex + 2) % n;
    int minIndex = index;
    double minDiff = kPi;
    for (int i = 0; i < n - 3; ++i) {
        const V3 splitEdge = normalised(sub(p[f[index]], p[f[startIndex]]));
        const double splitCos = dot(splitEdge, rightEdge);
        const double splitAngle = std::acos(clamp1(splitCos));
        const double angleDiff = std::fabs(splitAngle - bisectAngle);
        if (angleDiff < minDiff) {
            minDiff = angleDiff;
            minIndex = index;
        }
        index = (index + 1) % n;
    }
    const int diff = minIndex > startIndex ? minIndex - startIndex : minIndex + n - startIndex;
    const int nPoints1 = diff + 1, nPoints2 = n - diff + 1;
    std::vector<std::int32_t> face1(static_cast<std::size_t>(nPoints1)), face2(static_cast<std::size_t>(nPoints2));
    index = startIndex;
    for (int i = 0; i < nPoints1; ++i, index = (index + 1) % n) face1[i] = f[index];
    index = minIndex;
    for (int i = 0; i < nPoints2; ++i, index = (index + 1) % n) face2[i] = f[index];
    faceTriangles(face1, p, out);
    faceTriangles(face2, p, out);
}

// triangle::nearestPointClassify(p).distance() (triangleI.H:760)
double triNearestDist(const V3& a, const V3& b, const V3& c, const V3& p)
{
    const V3 ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const double d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return mag(sub(a, p));
    const V3 bp = sub(p, b);
    const double d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return mag(sub(b, p));
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        if ((d1 - d3) < kRootVSmall) return mag(sub(a, p));
        const double v = d1 / (d1 - d3);
        const V3 nearPt = add(a, mul(v, ab));
        return mag(sub(nearPt, p));
    }
    const V3 cp = sub(p, c);
    const double d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return mag(sub(c, p));
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        if ((d2 - d6) < kRootVSmall) return mag(sub(a, p));
        const double w = d2 / (d2 - d6);
        const V3 nearPt = add(a, mul(w, ac));
        return mag(sub(nearPt, p));
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        if (((d4 - d3) + (d5 - d6)) < kRootVSmall) return mag(sub(b, p));
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const V3 nearPt = add(b, mul(w, sub(c, b)));
        return mag(sub(nearPt, p));
    }
    if ((va + vb + vc) < kRootVSmall) {
        const V3 nearPt = mul(1.0 / 3.0, add(add(a, b), c));
        return mag(sub(nearPt, p));
    }
    const double denom = 1.0 / (va + vb + vc);
    const double v = vb * denom, w = vc * denom;
    const V3 nearPt = add(add(a, mul(v, ab)), mul(w, ac));
    return mag(sub(nearPt, p));
}

// boundBox
struct BB
{
    V3 mn{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    V3 mx{-std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(),
          -std::numeric_limits<double>::max()};
    void add(const V3& p)
    {
        for (int d = 0; d < 3; ++d) {
            mn[d] = std::min(mn[d], p[d]);
            mx[d] = std::max(mx[d], p[d]);
        }
    }
    V3 centre() const { return mul(0.5, dns::add(mn, mx)); }
    double magSqr() const { return sqr(mx[0] - mn[0]) + sqr(mx[1] - mn[1]) + sqr(mx[2] - mn[2]); } // min_.distSqr(max_)
    void inflate(double factor) // grow(factor * mag())
    {
        const double delta = factor * std::sqrt(magSqr());
        for (int d = 0; d < 3; ++d) {
            mn[d] -= delta;
            mx[d] += delta;
        }
    }
    bool contains(const V3& pt) const
    {
        return mn[0] <= pt[0] && pt[0] <= mx[0] && mn[1] <= pt[1] && pt[1] <= mx[1] && mn[2] <= pt[2] &&
               pt[2] <= mx[2];
    }
    bool overlaps(const BB& b) const // box_box_overlaps
    {
        return mn[0] <= b.mx[0] && b.mn[0] <= mx[0] && mn[1] <= b.mx[1] && b.mn[1] <= mx[1] && mn[2] <= b.mx[2] &&
               b.mn[2] <= mx[2];
    }
    int nDim() const
    {
        int ngood = 0;
        for (int d = 0; d < 3; ++d) {
            const double dist = mx[d] - mn[d];
            if (dist < 0) return -1;
            if (dist > 0) ++ngood;
        }
        return ngood;
    }
    V3 hexCorner(int i) const
    {
        switch (i) {
        case 0: return mn;
        case 1: return {mx[0], mn[1], mn[2]};
        case 2: return {mx[0], mx[1], mn[2]};
        case 3: return {mn[0], mx[1], mn[2]};
        case 4: return {mn[0], mn[1], mx[2]};
        case 5: return {mx[0], mn[1], mx[2]};
        case 6: return mx;
        default: return {mn[0], mx[1], mx[2]};
        }
    }
    // treeBoundBox::corner(octant)
    V3 corner(int octant) const
    {
        return {(octant & 1) ? mx[0] : mn[0], (octant & 2) ? mx[1] : mn[1], (octant & 4) ? mx[2] : mn[2]};
    }
    // treeBoundBox::subBbox(octant)
    BB subBbox(int octant) const
    {
        const V3 mid = centre();
        BB bb;
        bb.mn = mid;
        bb.mx = mid;
        for (int d = 0; d < 3; ++d) {
            if (octant & (1 << d)) bb.mx[d] = mx[d];
            else bb.mn[d] = mn[d];
        }
        return bb;
    }
};

// boundBox::box_sphere_overlaps
bool boxSphereOverlaps(const V3& corner0, const V3& corner1, const V3& centre, double radiusSqr)
{
    double distSqr = 0;
    for (int d = 0; d < 3; ++d) {
        const double d0 = corner0[d] - centre[d];
        const double d1 = corner1[d] - centre[d];
        if ((d0 > 0) != (d1 > 0)) continue;
        distSqr += std::min(d0 * d0, d1 * d1);
        if (distSqr > radiusSqr) return false;
    }
    return true;
}

struct MinMax
{
    double mn, mx;
    explicit MinMax(double v) : mn(v), mx(v) {}
    MinMax(double a, double b) : mn(a), mx(b) {}
    void add(double v)
    {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    bool overlaps(const MinMax& b) const { return mn <= b.mx && b.mn <= mx; }
};

// boundBox::intersects(const triPointRef&) (OpenFOAM/meshes/boundBox/boundBox.cxx:253)
bool bbIntersectsTri(const BB& bb, const V3& a, const V3& b, const V3& c)
{
    if (bb.nDim() != 3) return false;
    if (bb.contains(a) || bb.contains(b) || bb.contains(c)) return true;
    const auto projectBox = [&bb](const V3& axis) {
        MinMax e(dot(axis, bb.hexCorner(0)));
        for (int i = 1; i < 8; ++i) e.add(dot(axis, bb.hexCorner(i)));
        return e;
    };
    for (int d = 0; d < 3; ++d) {
        MinMax t(a[d]);
        t.add(b[d]);
        t.add(c[d]);
        if (!t.overlaps(MinMax(bb.mn[d], bb.mx[d]))) return false;
    }
    V3 axis = triAreaNormal(a, b, c);
    if (!MinMax(dot(axis, a)).overlaps(projectBox(axis))) return false;
    const std::array<V3, 3> edgeVecs{sub(c, b), sub(a, c), sub(b, a)}; // vecA, vecB, vecC
    for (const V3& edgeVec : edgeVecs)
        for (int d = 0; d < 3; ++d) {
            axis = {0, 0, 0};
            axis[d] = 1;
            axis = cross(edgeVec, axis);
            MinMax t(dot(axis, a));
            t.add(dot(axis, b));
            t.add(dot(axis, c));
            if (!t.overlaps(projectBox(axis))) return false;
        }
    return true;
}

// A primitivePatch as cyclicAMIPolyPatch::resetAMI builds patch0 / nbrPatch0:
// localFaces on the patch's own points (meshPoints order: first appearance).
struct Patch
{
    int n = 0;
    std::vector<std::int32_t> fStart, fNodes;
    Points p;
    std::vector<V3> normal;                       // faceNormals(): face::unitNormal
    std::vector<int> ffStart, ff;                 // faceFaces()
    std::vector<int> triStart;                    // srcTris_ / tgtTris_
    std::vector<std::array<std::int32_t, 3>> tri;
    std::vector<double> magSf;                    // srcMagSf_ / tgtMagSf_
    std::vector<V3> centre;                       // face::centre (target: tree, nearest point)
    std::vector<BB> fbb;                          // treeBoundBox(points, f) (target)

    Face face(int i) const { return {fNodes.data() + fStart[i], std::size_t(fStart[i + 1] - fStart[i])}; }
    std::span<const int> faceFaces(int i) const { return {ff.data() + ffStart[i], std::size_t(ffStart[i + 1] - ffStart[i])}; }
    std::span<const std::array<std::int32_t, 3>> tris(int i) const
    {
        return {tri.data() + triStart[i], std::size_t(triStart[i + 1] - triStart[i])};
    }
};

// PrimitivePatch::calcAddressing's faceFaces (PrimitivePatchAddressing.C:45):
// a face's list = the lower faces that found it, then the higher faces it
// finds edge by edge through pointFaces of the edge's start.
void calcFaceFaces(Patch& P)
{
    const int nP = int(P.p.size());
    std::vector<int> pfStart(static_cast<std::size_t>(nP) + 1, 0);
    for (const std::int32_t v : P.fNodes) ++pfStart[v + 1];
    for (int i = 0; i < nP; ++i) pfStart[i + 1] += pfStart[i];
    std::vector<int> pf(P.fNodes.size()), fill(pfStart.begin(), pfStart.end() - 1);
    for (int f = 0; f < P.n; ++f)
        for (const std::int32_t v : P.face(f)) pf[fill[v]++] = f;

    std::vector<std::vector<int>> ff(static_cast<std::size_t>(P.n));
    std::vector<char> detected(P.fNodes.size(), 0);
    for (int facei = 0; facei < P.n; ++facei) {
        const Face curF = P.face(facei);
        const int nE = int(curF.size());
        for (int edgeI = 0; edgeI < nE; ++edgeI) {
            if (detected[P.fStart[facei] + edgeI]) continue;
            const std::int32_t e0 = curF[edgeI], e1 = curF[(edgeI + 1) % nE];
            for (int k = pfStart[e0]; k < pfStart[e0 + 1]; ++k) {
                const int curNei = pf[k];
                if (curNei <= facei) continue;
                const Face nf = P.face(curNei);
                const int nNE = int(nf.size());
                for (int neiEdgeI = 0; neiEdgeI < nNE; ++neiEdgeI) {
                    const std::int32_t s0 = nf[neiEdgeI], s1 = nf[(neiEdgeI + 1) % nNE];
                    if (!((s0 == e0 && s1 == e1) || (s0 == e1 && s1 == e0))) continue;
                    ff[facei].push_back(curNei);
                    ff[curNei].push_back(facei);
                    detected[P.fStart[curNei] + neiEdgeI] = 1;
                }
            }
        }
    }
    P.ffStart.assign(static_cast<std::size_t>(P.n) + 1, 0);
    for (int f = 0; f < P.n; ++f) P.ffStart[f + 1] = P.ffStart[f] + int(ff[f].size());
    P.ff.resize(static_cast<std::size_t>(P.ffStart[P.n]));
    for (int f = 0; f < P.n; ++f) std::copy(ff[f].begin(), ff[f].end(), P.ff.begin() + P.ffStart[f]);
}

Patch makePatch(const PolyMesh& pm, std::span<const double> px, std::span<const double> py,
                std::span<const double> pz, const FoamPatch& fp, std::vector<std::int32_t>& map, bool target)
{
    Patch P;
    P.n = int(fp.nFaces);
    P.fStart.assign(static_cast<std::size_t>(P.n) + 1, 0);
    std::vector<std::int32_t> used;
    for (int f = 0; f < P.n; ++f) {
        const std::int64_t gf = fp.startFace + f;
        for (std::int32_t k = pm.faceStart[gf]; k < pm.faceStart[gf + 1]; ++k) {
            const std::int32_t v = pm.faceNodes[k];
            if (map[v] < 0) {
                map[v] = std::int32_t(P.p.size());
                used.push_back(v);
                P.p.push_back({px[v], py[v], pz[v]});
            }
            P.fNodes.push_back(map[v]);
        }
        P.fStart[f + 1] = std::int32_t(P.fNodes.size());
    }
    for (const std::int32_t v : used) map[v] = -1;

    P.normal.resize(static_cast<std::size_t>(P.n));
    P.magSf.resize(static_cast<std::size_t>(P.n));
    P.triStart.assign(static_cast<std::size_t>(P.n) + 1, 0);
    for (int f = 0; f < P.n; ++f) {
        const Face fc = P.face(f);
        P.normal[f] = normalised(faceAreaNormal(fc, P.p));
        faceTriangles(fc, P.p, P.tri); // advancingFrontAMI::triangulatePatch, tmMesh
        P.triStart[f + 1] = int(P.tri.size());
        double m = 0; // areaNormalisationMode project
        for (const auto& t : P.tris(f)) m += dot(triAreaNormal(P.p[t[0]], P.p[t[1]], P.p[t[2]]), P.normal[f]);
        P.magSf[f] = m;
    }
    calcFaceFaces(P);
    if (target) {
        P.centre.resize(static_cast<std::size_t>(P.n));
        P.fbb.resize(static_cast<std::size_t>(P.n));
        for (int f = 0; f < P.n; ++f) {
            const Face fc = P.face(f);
            if (fc.size() != 3) P.centre[f] = faceCentre(fc, P.p);
            for (const std::int32_t v : fc) P.fbb[f].add(P.p[v]);
        }
    }
    return P;
}

// face::nearestPoint(sample, points).distance() (faceIntersection.C:207)
double faceNearestDist(const Patch& P, int facei, const V3& sample)
{
    const Face f = P.face(facei);
    const Points& p = P.p;
    if (f.size() == 3) return triNearestDist(p[f[0]], p[f[1]], p[f[2]], sample);
    const V3& ctr = P.centre[facei];
    double nearest = kGreat;
    const std::size_t nP = f.size();
    for (std::size_t pI = 0; pI < nP; ++pI) {
        const double cur = triNearestDist(p[f[pI]], p[f[pI + 1 == nP ? 0 : pI + 1]], ctr, sample);
        if (std::fabs(cur) < std::fabs(nearest)) nearest = cur;
    }
    return nearest;
}

// indexedOctree<treeDataPrimitivePatch<primitivePatch>> as
// AMIInterpolation::createTree builds it (AMIInterpolation.C:86), and its
// findNearest with findNearestMaskedOp.
class Octree
{
public:
    static std::expected<Octree, std::string> make(const Patch& P)
    {
        Octree t;
        t.P_ = &P;
        if (P.n == 0) return t;
        BB bb;
        for (const V3& v : P.p) bb.add(v);
        bb.inflate(0.01);
        constexpr int maxLevels = 8;
        constexpr double maxLeafRatio = 10, maxDuplicity = 3.0;
        t.contents_.emplace_back(static_cast<std::size_t>(P.n));
        for (int i = 0; i < P.n; ++i) t.contents_[0][i] = i;
        auto top = t.divide(bb, 0);
        if (!top) return std::unexpected(top.error());
        t.nodes_.push_back(*top);
        for (int nLevels = 1; nLevels < maxLevels; ++nLevels) {
            std::int64_t nEntries = 0;
            for (const auto& c : t.contents_) nEntries += std::int64_t(c.size());
            if (double(nEntries) > maxDuplicity * P.n) break;
            const std::size_t nOldNodes = t.nodes_.size();
            // splitNodes(label(maxLeafRatio), nodes, contents): old nodes only
            for (std::size_t nodeI = 0; nodeI < nOldNodes; ++nodeI)
                for (int octant = 0; octant < 8; ++octant) {
                    const int index = t.nodes_[nodeI].sub[octant];
                    if (index < 0 || !(index & 1)) continue; // empty or node
                    const int contentI = index >> 1;
                    if (int(t.contents_[contentI].size()) <= int(maxLeafRatio)) continue;
                    auto subNode = t.divide(t.nodes_[nodeI].bb.subBbox(octant), contentI);
                    if (!subNode) return std::unexpected(subNode.error());
                    const int sz = int(t.nodes_.size());
                    t.nodes_.push_back(*subNode);
                    t.nodes_[nodeI].sub[octant] = sz << 1;
                }
            if (nOldNodes == t.nodes_.size()) break;
        }
        return t;
    }

    // tree.findNearest(sample, startDistSqr, findNearestMaskedOp(tree, exclude)).index(), -1 = miss
    int findNearest(const V3& sample, double startDistSqr, std::span<const int> exclude) const
    {
        double nearestDistSqr = startDistSqr;
        int nearestShape = -1;
        if (!nodes_.empty()) findNearest(0, sample, nearestDistSqr, nearestShape, exclude);
        return nearestShape;
    }

private:
    struct Node
    {
        BB bb;
        std::array<int, 8> sub{}; // -1 empty, 2 i node i, 2 i + 1 content i
    };

    // treeDataPrimitivePatch::overlaps(index, searchBox) (treeDataPrimitivePatch.C:393)
    bool overlaps(int index, const BB& searchBox) const
    {
        const Patch& P = *P_;
        if (!searchBox.overlaps(P.fbb[index])) return false;
        const Face f = P.face(index);
        const Points& p = P.p;
        if (f.size() == 3) return bbIntersectsTri(searchBox, p[f[0]], p[f[1]], p[f[2]]);
        for (const std::int32_t v : f)
            if (searchBox.contains(p[v])) return true;
        const V3& fc = P.centre[index];
        const std::size_t nP = f.size();
        for (std::size_t fp = 0; fp < nP; ++fp)
            if (bbIntersectsTri(searchBox, p[f[fp]], p[f[fp + 1 == nP ? 0 : fp + 1]], fc)) return true;
        return false;
    }

    // indexedOctree::divide(bb, contents, contentIndex) (indexedOctree.C:70)
    std::expected<Node, std::string> divide(const BB& bb, int contentIndex)
    {
        if (bb.mn[0] >= bb.mx[0] || bb.mn[1] >= bb.mx[1] || bb.mn[2] >= bb.mx[2])
            return std::unexpected(std::format("Badly formed bounding box:({} {} {}) ({} {} {})", bb.mn[0], bb.mn[1],
                                               bb.mn[2], bb.mx[0], bb.mx[1], bb.mx[2]));
        std::array<std::vector<int>, 8> divided;
        for (int octant = 0; octant < 8; ++octant) {
            const BB subBb = bb.subBbox(octant);
            for (const int index : contents_[contentIndex])
                if (overlaps(index, subBb)) divided[octant].push_back(index);
        }
        Node nod;
        nod.bb = bb;
        bool replaceNode = true;
        for (int octant = 0; octant < 8; ++octant) {
            if (divided[octant].empty()) {
                nod.sub[octant] = -1;
                continue;
            }
            if (replaceNode) {
                contents_[contentIndex] = std::move(divided[octant]);
                replaceNode = false;
            } else {
                contentIndex = int(contents_.size());
                contents_.push_back(std::move(divided[octant]));
            }
            nod.sub[octant] = (contentIndex << 1) | 1;
        }
        return nod;
    }

    // treeBoundBox::searchOrder (treeBoundBoxI.H)
    static std::array<int, 8> searchOrder(const BB& bb, const V3& pt)
    {
        V3 dist = sub(bb.centre(), pt);
        int octant = 0;
        for (int d = 0; d < 3; ++d)
            if (dist[d] < 0) {
                octant |= 1 << d;
                dist[d] *= -1;
            }
        int mn = 0, mid = 0, mx = 0;
        if (dist[0] < dist[1]) {
            if (dist[1] < dist[2]) mn = 1, mid = 2, mx = 4;
            else if (dist[2] < dist[0]) mn = 4, mid = 1, mx = 2;
            else mn = 1, mid = 4, mx = 2;
        } else {
            if (dist[2] < dist[1]) mn = 4, mid = 2, mx = 1;
            else if (dist[0] < dist[2]) mn = 2, mid = 1, mx = 4;
            else mn = 2, mid = 4, mx = 1;
        }
        std::array<int, 8> o{};
        o[0] = octant;
        o[1] = octant ^ mn;
        o[2] = octant ^ mid;
        o[3] = octant ^ mx;
        o[4] = o[1] ^ mid;
        o[5] = o[1] ^ mx;
        o[6] = o[2] ^ mx;
        o[7] = o[4] ^ mx;
        return o;
    }

    // indexedOctree::findNearest(nodeI, ...) (indexedOctree.C:396)
    void findNearest(int nodeI, const V3& sample, double& nearestDistSqr, int& nearestShape,
                     std::span<const int> exclude) const
    {
        const Node& nod = nodes_[nodeI];
        for (const int octant : searchOrder(nod.bb, sample)) {
            const int index = nod.sub[octant];
            if (index < 0) continue;
            if (!(index & 1)) {
                const BB& subBb = nodes_[index >> 1].bb;
                if (boxSphereOverlaps(subBb.mn, subBb.mx, sample, nearestDistSqr))
                    findNearest(index >> 1, sample, nearestDistSqr, nearestShape, exclude);
                continue;
            }
            if (!boxSphereOverlaps(nod.bb.centre(), nod.bb.corner(octant), sample, nearestDistSqr)) continue;
            for (const int shape : contents_[index >> 1]) { // findNearestMaskedOp
                if (std::ranges::find(exclude, shape) != exclude.end()) continue;
                const double distSqr = sqr(faceNearestDist(*P_, shape, sample));
                if (distSqr < nearestDistSqr) {
                    nearestDistSqr = distSqr;
                    nearestShape = shape;
                }
            }
        }
    }

    const Patch* P_ = nullptr;
    std::vector<Node> nodes_;
    std::vector<std::vector<int>> contents_;
};

using Tri = std::array<V3, 3>;
struct TriList // FixedList<triPoints, 10>
{
    std::array<Tri, 10> t;
    int n = 0;
    void push(const V3& a, const V3& b, const V3& c) { t[n++] = {a, b, c}; }
};

// faceAreaIntersect::planeIntersection
V3 planeIntersection(const std::array<double, 3>& d, const Tri& t, int negI, int posI)
{
    const double dp = d[posI], dn = d[negI];
    return div(sub(mul(dp, t[negI]), mul(dn, t[posI])), -dn + dp);
}

// faceAreaIntersect::triSliceWithPlane (faceAreaIntersect.C:47); the plane is
// (origin, normal), signedDistance = (p - origin) & normal.
void triSliceWithPlane(const Tri& tri, const V3& origin, const V3& normal, TriList& tris, double len)
{
    std::array<double, 3> d{};
    int nCoPlanar = 0, nPos = 0, posI = -1, negI = -1, copI = -1;
    for (int i = 0; i < 3; ++i) {
        d[i] = dot(sub(tri[i], origin), normal);
        if (std::fabs(d[i]) < kInterTol * len) {
            ++nCoPlanar;
            copI = i;
            d[i] = 0.0;
        } else if (d[i] > 0) {
            ++nPos;
            posI = i;
        } else {
            negI = i;
        }
    }
    if (nPos == 3 || (nPos == 2 && nCoPlanar == 1) || (nPos == 1 && nCoPlanar == 2)) {
        tris.t[tris.n++] = tri;
        return;
    }
    if (nPos == 2 && nCoPlanar == 0) {
        const int i0 = negI, i1 = (i0 + 1) % 3, i2 = (i1 + 1) % 3;
        const V3 p01 = planeIntersection(d, tri, i0, i1);
        const V3 p02 = planeIntersection(d, tri, i0, i2);
        tris.push(tri[i1], tri[i2], p02);
        tris.push(tri[i1], p02, p01);
        return;
    }
    if (nPos != 1) return;
    const int i0 = posI;
    if (nCoPlanar == 0) {
        const int i1 = (i0 + 1) % 3, i2 = (i1 + 1) % 3;
        const V3 p01 = planeIntersection(d, tri, i1, i0);
        const V3 p02 = planeIntersection(d, tri, i2, i0);
        tris.push(tri[i0], p01, p02);
        return;
    }
    const int i1 = negI, i2 = copI;
    const V3 p01 = planeIntersection(d, tri, i1, i0);
    if ((i0 + 1) % 3 == i1) tris.push(tri[i0], p01, tri[i2]);
    else tris.push(tri[i0], tri[i2], p01);
}

// faceAreaIntersect::triangleIntersect (faceAreaIntersect.C:222), area only.
void triangleIntersect(const Tri& src, const V3& tgt0, const V3& tgt1, const V3& tgt2, const V3& n, double& area)
{
    TriList work1, work2;
    const double srcArea = mag(triAreaNormal(src[0], src[1], src[2]));
    if (srcArea < kRootVSmall) return;
    const double t = std::sqrt(srcArea);
    {
        const double s = mag(sub(tgt1, tgt0));
        if (s < kRootVSmall) return;
        const V3 n0 = cross(sub(tgt0, tgt1), mul(-s, n));
        const double magSqrN0 = magSqr(n0);
        if (magSqrN0 < kRootVSmall) return;
        triSliceWithPlane(src, tgt0, div(n0, std::sqrt(magSqrN0)), work1, t);
    }
    if (work1.n == 0) return;
    {
        const double s = mag(sub(tgt2, tgt1));
        if (s < kRootVSmall) return;
        const V3 n1 = cross(sub(tgt1, tgt2), mul(-s, n));
        const double magSqrN1 = magSqr(n1);
        if (magSqrN1 < kRootVSmall) return;
        const V3 nrm = div(n1, std::sqrt(magSqrN1));
        for (int i = 0; i < work1.n; ++i) triSliceWithPlane(work1.t[i], tgt1, nrm, work2, t);
        if (work2.n == 0) return;
    }
    {
        const double s = mag(sub(tgt2, tgt0));
        if (s < kRootVSmall) return;
        const V3 n2 = cross(sub(tgt2, tgt0), mul(-s, n));
        const double magSqrN2 = magSqr(n2);
        if (magSqrN2 < kRootVSmall) return;
        const V3 nrm = div(n2, std::sqrt(magSqrN2));
        work1.n = 0;
        for (int i = 0; i < work2.n; ++i) triSliceWithPlane(work2.t[i], tgt2, nrm, work1, t);
        for (int i = 0; i < work1.n; ++i) area += mag(triAreaNormal(work1.t[i][0], work1.t[i][1], work1.t[i][2]));
    }
}

// The walk of one AMI: faceAreaWeightAMI's members on (src, tgt).
struct Walk
{
    const Patch& src;
    const Patch& tgt;
    const Octree& tree;
    std::vector<std::vector<int>> srcAddr, tgtAddr;
    std::vector<std::vector<double>> srcWght, tgtWght;
    std::vector<std::string>* warnings = nullptr;
    bool warnedNormal = false;

    // advancingFrontAMI::isCandidate (maxDistance2, minCosAngle off)
    bool isCandidate(int srcFacei, int tgtFacei) const
    {
        return !(src.magSf[srcFacei] < kRootVSmall || tgt.magSf[tgtFacei] < kRootVSmall);
    }

    // The projection normal of calcInterArea / overlaps; false = "Invalid normal".
    bool interNormal(int srcFacei, int tgtFacei, V3& n)
    {
        n = add(neg(src.normal[srcFacei]), tgt.normal[tgtFacei]);
        const double magN = mag(n);
        if (magN > kRootVSmall) {
            n = div(n, magN);
            return true;
        }
        if (!warnedNormal) {
            warnings->push_back(std::format("Invalid normal for source face {} target face {}", srcFacei, tgtFacei));
            warnedNormal = true;
        }
        return false;
    }

    // faceAreaWeightAMI::calcInterArea -> faceAreaIntersect::calc; threshold
    // < 0. With a threshold: faceAreaWeightAMI::overlaps (stops when area > threshold).
    double interArea(int srcFacei, int tgtFacei, double threshold, bool& over)
    {
        over = false;
        double area = 0;
        if (!isCandidate(srcFacei, tgtFacei)) return area;
        V3 n;
        if (!interNormal(srcFacei, tgtFacei, n)) return area;
        for (const auto& triA : src.tris(srcFacei)) {
            const Tri tpA{src.p[triA[0]], src.p[triA[1]], src.p[triA[2]]};
            for (const auto& triB : tgt.tris(tgtFacei)) {
                triangleIntersect(tpA, tgt.p[triB[2]], tgt.p[triB[1]], tgt.p[triB[0]], n, area);
                if (threshold >= 0 && area > threshold) {
                    over = true;
                    return area;
                }
            }
        }
        return area;
    }
    bool overlaps(int srcFacei, int tgtFacei, double threshold)
    {
        bool over = false;
        interArea(srcFacei, tgtFacei, threshold, over);
        return over;
    }

    // advancingFrontAMI::findTargetFace (advancingFrontAMI.C:317)
    int findTargetFace(int srcFacei, std::span<const int> excludeFaces, int srcFacePti) const
    {
        const Face srcFace = src.face(srcFacei);
        BB bb;
        for (const std::int32_t v : srcFace) bb.add(src.p[v]);
        const V3 srcPt = srcFacePti == -1 ? bb.centre() : src.p[srcFace[srcFacePti]];
        int sample = tree.findNearest(srcPt, 0.25 * bb.magSqr(), excludeFaces);
        if (sample < 0) sample = tree.findNearest(srcPt, sqr(kGreat), excludeFaces);
        if (sample >= 0 && isCandidate(srcFacei, sample)) return sample;
        return -1;
    }

    // advancingFrontAMI::appendNbrFaces on the target patch
    void appendNbrFaces(int facei, const std::vector<int>& visitedFaces, std::vector<int>& faceIDs) const
    {
        static const double thetaCos = std::cos(89.0 * kPi / 180.0);
        for (const int nbrFacei : tgt.faceFaces(facei)) {
            if (std::ranges::find(visitedFaces, nbrFacei) != visitedFaces.end()) continue;
            if (std::ranges::find(faceIDs, nbrFacei) != faceIDs.end()) continue;
            const double cosI = dot(tgt.normal[facei], tgt.normal[nbrFacei]);
            if (cosI > thetaCos) faceIDs.push_back(nbrFacei);
        }
    }

    // faceAreaWeightAMI::processSourceFace
    bool processSourceFace(int srcFacei, int tgtStartFacei, std::vector<int>& nbrFaces, std::vector<int>& visitedFaces)
    {
        if (tgtStartFacei == -1) return false;
        nbrFaces.push_back(tgtStartFacei);
        appendNbrFaces(tgtStartFacei, visitedFaces, nbrFaces);
        bool faceProcessed = false;
        while (!nbrFaces.empty()) {
            const int tgtFacei = nbrFaces.back();
            nbrFaces.pop_back();
            visitedFaces.push_back(tgtFacei);
            bool over = false;
            const double area = interArea(srcFacei, tgtFacei, -1, over);
            if (area / src.magSf[srcFacei] > kInterTol) {
                srcAddr[srcFacei].push_back(tgtFacei);
                srcWght[srcFacei].push_back(area);
                tgtAddr[tgtFacei].push_back(srcFacei);
                tgtWght[tgtFacei].push_back(area);
                appendNbrFaces(tgtFacei, visitedFaces, nbrFaces);
                faceProcessed = true;
            }
        }
        return faceProcessed;
    }

    // faceAreaWeightAMI::setNextFaces; an error = its FatalError.
    std::expected<bool, std::string> setNextFaces(int& startSeedi, int& srcFacei, int& tgtFacei,
                                                  const std::vector<char>& mapFlag, int nMapped,
                                                  std::vector<int>& seedFaces, const std::vector<int>& visitedFaces)
    {
        if (nMapped == 0) return false;
        const int nSrc = src.n;
        const auto findNext = [&](int i) {
            for (++i; i < nSrc; ++i)
                if (mapFlag[i]) return i;
            return -1;
        };
        const std::span<const int> srcNbrFaces = src.faceFaces(srcFacei);
        tgtFacei = -1;
        bool valuesSet = false;
        for (const int faceS : srcNbrFaces) {
            if (!(mapFlag[faceS] && seedFaces[faceS] == -1)) continue;
            for (const int faceT : visitedFaces) {
                const double threshold = src.magSf[faceS] * kInterTol;
                if (!overlaps(faceS, faceT, threshold)) continue;
                seedFaces[faceS] = faceT;
                if (!valuesSet) {
                    srcFacei = faceS;
                    tgtFacei = faceT;
                    valuesSet = true;
                }
            }
        }
        if (valuesSet) return true;

        int facei = startSeedi;
        if (!mapFlag[startSeedi]) facei = findNext(facei);
        const int startSeedi0 = facei;
        bool foundNextSeed = false;
        while (facei != -1) {
            if (!foundNextSeed) {
                startSeedi = facei;
                foundNextSeed = true;
            }
            if (seedFaces[facei] != -1) {
                srcFacei = facei;
                tgtFacei = seedFaces[facei];
                return true;
            }
            facei = findNext(facei);
        }
        facei = startSeedi0;
        while (facei != -1) {
            srcFacei = facei;
            tgtFacei = findTargetFace(srcFacei, visitedFaces, -1);
            if (tgtFacei >= 0) return true;
            facei = findNext(facei);
        }
        // mustMatchFaces() = requireMatch && !applyLowWeightCorrection()
        const V3 c = faceCentre(src.face(srcFacei), src.p);
        return std::unexpected(std::format("Unable to set target face for source face {} with centre: ({:g} {:g} {:g})",
                                           srcFacei, c[0], c[1], c[2]));
    }

    // faceAreaWeightAMI::calcAddressing
    std::expected<void, std::string> calcAddressing(int srcFacei, int tgtFacei)
    {
        int nFacesRemaining = src.n;
        std::vector<int> nbrFaces, visitedFaces;
        std::vector<int> seedFaces(static_cast<std::size_t>(src.n), -1);
        seedFaces[srcFacei] = tgtFacei;
        std::vector<char> mapFlag(static_cast<std::size_t>(src.n), 1);
        int startSeedi = 0;
        for (;;) {
            nbrFaces.clear();
            visitedFaces.clear();
            processSourceFace(srcFacei, tgtFacei, nbrFaces, visitedFaces);
            if (mapFlag[srcFacei]) {
                mapFlag[srcFacei] = 0;
                --nFacesRemaining;
            }
            const auto cont =
                setNextFaces(startSeedi, srcFacei, tgtFacei, mapFlag, nFacesRemaining, seedFaces, visitedFaces);
            if (!cont) return std::unexpected(cont.error());
            if (!*cont) return {};
        }
    }

    // faceAreaWeightAMI::restartUncoveredSourceFace
    void restartUncoveredSourceFace()
    {
        std::vector<int> nbrFaces, visitedFaces;
        for (int srcFacei = 0; srcFacei < src.n; ++srcFacei) {
            double s = 0;
            for (const double w : srcWght[srcFacei]) s += w;
            const double t = s / src.magSf[srcFacei];
            if (!(t < kRestartMinWeight)) continue;
            const int nFp = int(src.face(srcFacei).size());
            for (int fpi = 0; fpi < nFp; ++fpi) {
                const int tgtFacei = findTargetFace(srcFacei, srcAddr[srcFacei], fpi);
                if (tgtFacei == -1) continue;
                nbrFaces.clear();
                visitedFaces = srcAddr[srcFacei];
                processSourceFace(srcFacei, tgtFacei, nbrFaces, visitedFaces);
            }
        }
    }
};

// AMIInterpolation::normaliseWeights(patchAreas, ..., conformal = true)
FoamAmiSide normalise(const std::vector<std::vector<int>>& addr, std::vector<std::vector<double>>& wght,
                      const std::vector<double>& magSf)
{
    FoamAmiSide S;
    const std::size_t n = addr.size();
    S.start.assign(n + 1, 0);
    S.wSum.resize(n);
    S.magSf = magSf;
    for (std::size_t f = 0; f < n; ++f) S.start[f + 1] = S.start[f] + int(addr[f].size());
    S.addr.reserve(static_cast<std::size_t>(S.start[n]));
    S.w.reserve(static_cast<std::size_t>(S.start[n]));
    for (std::size_t f = 0; f < n; ++f) {
        std::vector<double>& w = wght[f];
        if (w.empty()) {
            S.wSum[f] = 0;
            continue;
        }
        double denom = magSf[f];
        double s = 0;
        for (const double x : w) s += x;
        const double t = s / denom;
        denom = s; // conformal
        for (double& x : w) x /= denom;
        S.wSum[f] = t;
        S.addr.insert(S.addr.end(), addr[f].begin(), addr[f].end());
        S.w.insert(S.w.end(), w.begin(), w.end());
    }
    return S;
}

} // namespace

std::expected<FoamAmi, std::string> foamCyclicAmi(const PolyMesh& pm, std::span<const double> px,
                                                  std::span<const double> py, std::span<const double> pz,
                                                  int srcPatch, int tgtPatch)
{
    const int nPatches = int(pm.man.patches.size());
    if (srcPatch < 0 || srcPatch >= nPatches || tgtPatch < 0 || tgtPatch >= nPatches || srcPatch == tgtPatch)
        return std::unexpected(std::format("foamCyclicAmi: patch indices {} {} out of range", srcPatch, tgtPatch));
    const FoamPatch& sp = pm.man.patches[srcPatch];
    const FoamPatch& tp = pm.man.patches[tgtPatch];
    if (sp.type != FoamPatchType::CyclicAMI || tp.type != FoamPatchType::CyclicAMI)
        return std::unexpected(std::format("foamCyclicAmi: patches {} and {} are not both cyclicAMI", sp.name, tp.name));
    if (!(srcPatch < tgtPatch)) // cyclicAMIPolyPatch::AMI()
        return std::unexpected("AMI interpolator only available to owner patch");
    if (px.size() != pm.px.size() || py.size() != pm.px.size() || pz.size() != pm.px.size())
        return std::unexpected("foamCyclicAmi: point arrays do not match the mesh");

    std::vector<std::int32_t> map(pm.px.size(), -1);
    const Patch src = makePatch(pm, px, py, pz, sp, map, false);
    const Patch tgt = makePatch(pm, px, py, pz, tp, map, true);

    FoamAmi A;
    // advancingFrontAMI::checkPatches (requireMatch): a warning only
    if (src.n > 0 && tgt.n > 0) {
        BB bbSrc, bbTgt;
        for (const V3& v : src.p) bbSrc.add(v);
        for (const V3& v : tgt.p) bbTgt.add(v);
        BB bbTgtInf = bbTgt;
        bbTgtInf.inflate(kMaxBoundsError);
        if (!(bbTgtInf.contains(bbSrc.mn) && bbTgtInf.contains(bbSrc.mx)))
            A.warnings.push_back("Source and target patch bounding boxes are not similar");
    }

    std::vector<std::vector<int>> srcAddr(static_cast<std::size_t>(src.n)), tgtAddr(static_cast<std::size_t>(tgt.n));
    std::vector<std::vector<double>> srcWght(static_cast<std::size_t>(src.n)), tgtWght(static_cast<std::size_t>(tgt.n));

    // advancingFrontAMI::initialiseWalk
    if (src.n > 0 && tgt.n == 0)
        A.warnings.push_back(std::format("{} source faces but no target faces", src.n));
    if (src.n > 0 && tgt.n > 0) {
        auto tree = Octree::make(tgt);
        if (!tree) return std::unexpected(tree.error());
        Walk W{src, tgt, *tree, std::move(srcAddr), std::move(tgtAddr), std::move(srcWght), std::move(tgtWght),
               &A.warnings};
        // faceAreaWeightAMI::calculate passes srcFacei = tgtFacei = 0, so
        // initialiseWalk's search ((srcFacei == -1) || (tgtFacei == -1)) never
        // runs: the walk starts on the pair (0, 0) whether it overlaps or not
        // ("Unable to find initial target face" is unreachable here).
        if (auto r = W.calcAddressing(0, 0); !r) return std::unexpected(r.error());
        W.restartUncoveredSourceFace(); // restartUncoveredSourceFace 1 (the default)
        srcAddr = std::move(W.srcAddr);
        tgtAddr = std::move(W.tgtAddr);
        srcWght = std::move(W.srcWght);
        tgtWght = std::move(W.tgtWght);
    }
    A.src = normalise(srcAddr, srcWght, src.magSf);
    A.tgt = normalise(tgtAddr, tgtWght, tgt.magSf);
    return A;
}

std::vector<double> foamAmiInterpolate(const FoamAmiSide& s, std::span<const double> fld, int nComp)
{
    const std::size_t n = s.wSum.size(), nc = std::size_t(nComp);
    std::vector<double> result(n * nc, 0.0);
    for (std::size_t f = 0; f < n; ++f)
        for (int i = s.start[f]; i < s.start[f + 1]; ++i)
            for (std::size_t c = 0; c < nc; ++c) result[f * nc + c] += s.w[i] * fld[std::size_t(s.addr[i]) * nc + c];
    return result;
}

FoamAmiSum foamAmiSum(const FoamAmiSide& s)
{
    FoamAmiSum r;
    r.min = std::numeric_limits<double>::max();
    r.max = -std::numeric_limits<double>::max();
    double sum = 0;
    for (const double t : s.wSum) {
        r.min = std::min(r.min, t);
        r.max = std::max(r.max, t);
        sum += t;
        if (t == 0) ++r.nZero;
    }
    r.average = sum / double(s.wSum.size());
    return r;
}

std::string foamAmiSumLine(const FoamAmiSide& s, const char* sourceOrTarget, int precision)
{
    const FoamAmiSum r = foamAmiSum(s);
    return std::format("AMI: Patch {} sum(weights) min:{:.{}g} max:{:.{}g} average:{:.{}g}", sourceOrTarget, r.min,
                       precision, r.max, precision, r.average, precision);
}

} // namespace dns
