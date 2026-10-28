#include "dns/facemesh.hpp"

#include "dns/la/graph.hpp"
#include "dns/parallel.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>

namespace dns {
namespace {

using std::string;
using V3 = std::array<double, 3>;

// OpenFOAM doubleScalar VSMALL / ROOTVSMALL.
constexpr double kVSmall = 1.0e-300;
constexpr double kRootVSmall = 1.0e-150;

V3 operator+(const V3& a, const V3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
V3 operator-(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
V3 operator-(const V3& a) { return {-a[0], -a[1], -a[2]}; }
V3 operator*(double s, const V3& a) { return {s * a[0], s * a[1], s * a[2]}; }
V3 operator/(const V3& a, double s) { return {a[0] / s, a[1] / s, a[2] / s}; }
V3& operator+=(V3& a, const V3& b) { return a = a + b; }
// OpenFOAM's `&` and `^`, same operand order.
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V3 cross(const V3& a, const V3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double mag(const V3& a) { return std::sqrt(dot(a, a)); }

V3 point(const PolyMesh& m, std::int32_t i)
{
    return {m.px[std::size_t(i)], m.py[std::size_t(i)], m.pz[std::size_t(i)]};
}

// primitiveMeshTools::makeFaceCentresAndAreas, one face.
void faceGeometry(const PolyMesh& m, std::size_t f, V3& Cf, V3& S)
{
    const std::size_t a = std::size_t(m.faceStart[f]), n = std::size_t(m.faceStart[f + 1]) - a;
    const auto P = [&](std::size_t q) { return point(m, m.faceNodes[a + q]); };
    if (n == 3) {
        const V3 p0 = P(0), p1 = P(1), p2 = P(2);
        Cf = (1.0 / 3.0) * (p0 + p1 + p2);
        S = 0.5 * cross(p1 - p0, p2 - p0);
        return;
    }
    V3 fC{0, 0, 0};
    for (std::size_t q = 0; q < n; ++q) fC += P(q);
    fC = fC / double(n);
    V3 sumN{0, 0, 0}, sumAc{0, 0, 0};
    double sumA = 0;
    for (std::size_t q = 0; q < n; ++q) {
        const V3 thisP = P(q), nextP = P(q == n - 1 ? 0 : q + 1);
        const V3 c = thisP + nextP + fC;
        const V3 nv = cross(nextP - thisP, fC - thisP);
        const double am = mag(nv);
        sumN += nv;
        sumA += am;
        sumAc += am * c;
    }
    if (sumA < kRootVSmall) {
        Cf = fC;
        S = {0, 0, 0};
        return;
    }
    Cf = ((1.0 / 3.0) * sumAc) / sumA;
    S = 0.5 * sumN;
}

// The FV data in SOURCE numbering, before any ordering.
struct SrcMesh
{
    int nC = 0, dim = 3, emptyAxis = -1;
    std::vector<V3> C;
    std::vector<double> V;
    // internal faces (polyMesh internal + merged cyclic pairs)
    std::vector<int> iOwn, iNei;
    std::vector<std::int64_t> iSrc;
    std::vector<V3> iS, iCf, iD;
    // boundary faces, patch by patch (non-cyclic patches, polyMesh order)
    std::vector<int> bOwn;
    std::vector<std::int64_t> bSrc;
    std::vector<V3> bS, bCf, bD;
    std::vector<MeshPatch> patches;  // first/count into the b* arrays
    std::vector<std::string> zoneNames;
    std::vector<std::vector<int>> zoneCells;
};

// First error of a forBlocks sweep: one slot per block, the lowest block's
// is the serial loop's first.
struct BlockErrors
{
    std::vector<std::string> e;
    explicit BlockErrors(int nw) : e(std::size_t(nw)) {}
    std::expected<void, string> first() const
    {
        for (const auto& s : e)
            if (!s.empty()) return std::unexpected(s);
        return {};
    }
};

std::expected<SrcMesh, string> buildSource(const PolyMesh& pm, int threads)
{
    const auto& man = pm.man;
    const std::size_t nF = std::size_t(man.nFaces), nI = std::size_t(man.nInternalFaces);
    const int nC = int(man.nCells);
    if (nC < 1) return std::unexpected("mesh: no cells");
    if (pm.faceStart.size() != nF + 1 || pm.owner.size() != nF || pm.neighbour.size() != nI
        || pm.px.size() != std::size_t(man.nPoints) || pm.py.size() != pm.px.size()
        || pm.pz.size() != pm.px.size())
        return std::unexpected("mesh: PolyMesh arrays do not match its manifest counts");
    for (const auto& p : man.patches)
        if (p.type == FoamPatchType::Wedge)
            return std::unexpected("mesh: patch '" + p.name
                                   + "' is a wedge: axisymmetric FaceMeshes are not built yet "
                                     "(masterplan_meshsolver P8)");
    for (std::size_t f = 0; f < nF; ++f)
        if (pm.owner[f] < 0 || pm.owner[f] >= nC || (f < nI && (pm.neighbour[f] <= pm.owner[f]
                                                                 || pm.neighbour[f] >= nC)))
            return std::unexpected(std::format("mesh: face {} has bad owner/neighbour", f));

    SrcMesh s;
    s.nC = nC;

    // Face geometry, one face per index.
    std::vector<V3> Cf(nF), S(nF);
    {
        const int nw = workerCount(threads, nF);
        forBlocks(nw, nF, [&](int, std::size_t lo, std::size_t hi) {
            for (std::size_t f = lo; f < hi; ++f) faceGeometry(pm, f, Cf[f], S[f]);
        });
    }

    // Source CSR in OpenFOAM's accumulation order: owned faces (ascending),
    // then neighbour faces (ascending); neighbour entries stored as ~f.
    std::vector<int> cs(std::size_t(nC) + 1, 0), cf;
    for (std::size_t f = 0; f < nF; ++f) ++cs[std::size_t(pm.owner[f]) + 1];
    for (std::size_t f = 0; f < nI; ++f) ++cs[std::size_t(pm.neighbour[f]) + 1];
    std::partial_sum(cs.begin(), cs.end(), cs.begin());
    cf.resize(std::size_t(cs.back()));
    {
        std::vector<int> fill(cs.begin(), cs.end() - 1);
        for (std::size_t f = 0; f < nF; ++f) cf[std::size_t(fill[std::size_t(pm.owner[f])]++)] = int(f);
        for (std::size_t f = 0; f < nI; ++f)
            cf[std::size_t(fill[std::size_t(pm.neighbour[f])]++)] = ~int(f);
    }

    // Cell geometry (makeCellCentresAndVols) + closure/volume validity.
    s.C.resize(std::size_t(nC));
    s.V.resize(std::size_t(nC));
    {
        const int nw = workerCount(threads, std::size_t(nC));
        BlockErrors errs(nw);
        forBlocks(nw, std::size_t(nC), [&](int w, std::size_t lo, std::size_t hi) {
            for (std::size_t c = lo; c < hi; ++c) {
                const int b = cs[c], e = cs[c + 1];
                if (e - b < 4) {
                    errs.e[std::size_t(w)] = std::format("mesh: cell {} has {} faces (< 4)", c, e - b);
                    return;
                }
                V3 cEst{0, 0, 0};
                for (int q = b; q < e; ++q) {
                    const int f = cf[std::size_t(q)];
                    cEst += Cf[std::size_t(f >= 0 ? f : ~f)];
                }
                cEst = cEst / double(e - b);
                V3 ctr{0, 0, 0}, sumS{0, 0, 0};
                double vol = 0, sumMagS = 0;
                for (int q = b; q < e; ++q) {
                    const int f = cf[std::size_t(q)];
                    const std::size_t g = std::size_t(f >= 0 ? f : ~f);
                    const double pyr3 = f >= 0 ? dot(S[g], Cf[g] - cEst) : dot(S[g], cEst - Cf[g]);
                    const V3 pc = 0.75 * Cf[g] + 0.25 * cEst;
                    ctr += pyr3 * pc;
                    vol += pyr3;
                    sumS += f >= 0 ? S[g] : -S[g];
                    sumMagS += mag(S[g]);
                }
                if (std::abs(vol) > kVSmall)
                    ctr = ctr / vol;
                else
                    ctr = cEst;
                vol *= 1.0 / 3.0;
                if (!(mag(sumS) <= 1e-9 * sumMagS)) {
                    errs.e[std::size_t(w)] = std::format(
                        "mesh: open cell {} (|sum S| / sum |S| = {:.3e} > 1e-9)", c, mag(sumS) / sumMagS);
                    return;
                }
                if (!(vol > 0)) {
                    errs.e[std::size_t(w)] = std::format("mesh: cell {} has volume {} <= 0", c, vol);
                    return;
                }
                s.C[c] = ctr;
                s.V[c] = vol;
            }
        });
        if (auto r = errs.first(); !r) return std::unexpected(r.error());
    }

    // Bounding-box diagonal: the cyclic matching scale.
    double diag = 0;
    {
        V3 lo{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
              std::numeric_limits<double>::max()};
        V3 hi = -lo;
        for (std::size_t i = 0; i < pm.px.size(); ++i) {
            const V3 p = point(pm, std::int32_t(i));
            for (int a = 0; a < 3; ++a) {
                lo[std::size_t(a)] = std::min(lo[std::size_t(a)], p[std::size_t(a)]);
                hi[std::size_t(a)] = std::max(hi[std::size_t(a)], p[std::size_t(a)]);
            }
        }
        diag = mag(hi - lo);
    }

    // Empty axis: every empty face normal to one axis, two per cell.
    std::vector<char> isEmpty(nF, 0);
    for (const auto& p : man.patches) {
        if (p.type != FoamPatchType::Empty) continue;
        for (std::int64_t q = 0; q < p.nFaces; ++q) {
            const std::size_t f = std::size_t(p.startFace + q);
            isEmpty[f] = 1;
            const V3& n = S[f];
            const double m = mag(n);
            int a = 0;
            for (int k = 1; k < 3; ++k)
                if (std::abs(n[std::size_t(k)]) > std::abs(n[std::size_t(a)])) a = k;
            const double off = std::sqrt(std::max(0.0, dot(n, n) - n[std::size_t(a)] * n[std::size_t(a)]));
            if (!(m > 0) || off > 1e-9 * m)
                return std::unexpected(std::format("mesh: empty face {} (patch '{}') is not normal to "
                                                   "x, y or z: `empty` must be a one-cell extrusion "
                                                   "along x|y|z",
                                                   f, p.name));
            if (s.emptyAxis >= 0 && s.emptyAxis != a)
                return std::unexpected(std::format("mesh: empty faces normal to two axes ({} and {})",
                                                   "xyz"[s.emptyAxis], "xyz"[a]));
            s.emptyAxis = a;
        }
    }
    if (s.emptyAxis >= 0) {
        s.dim = 2;
        const std::size_t a = std::size_t(s.emptyAxis);
        std::vector<int> nPlus(std::size_t(nC), 0), nMinus(std::size_t(nC), 0);
        for (std::size_t f = nI; f < nF; ++f)
            if (isEmpty[f]) ++(S[f][a] > 0 ? nPlus : nMinus)[std::size_t(pm.owner[f])];
        for (int c = 0; c < nC; ++c)
            if (nPlus[std::size_t(c)] != 1 || nMinus[std::size_t(c)] != 1)
                return std::unexpected(std::format(
                    "mesh: cell {} has {} +{} and {} -{} empty faces: `empty` must be a one-cell "
                    "extrusion along x|y|z",
                    c, nPlus[std::size_t(c)], "xyz"[a], nMinus[std::size_t(c)], "xyz"[a]));
        for (std::size_t f = 0; f < nF; ++f)
            if (!isEmpty[f] && std::abs(S[f][a]) > 1e-12 * mag(S[f]))
                return std::unexpected(std::format(
                    "mesh: face {} is not perpendicular to the empty axis {} (|S.e|/|S| = {:.3e} > "
                    "1e-12): `empty` must be a one-cell extrusion",
                    f, "xyz"[a], std::abs(S[f][a]) / mag(S[f])));
    }

    // Internal faces: polyMesh internal, then the merged cyclic pairs.
    auto pushInternal = [&](int o, int n, std::int64_t src, const V3& Sv, const V3& Cfv, const V3& dv) {
        s.iOwn.push_back(o);
        s.iNei.push_back(n);
        s.iSrc.push_back(src);
        s.iS.push_back(Sv);
        s.iCf.push_back(Cfv);
        s.iD.push_back(dv);
    };
    for (std::size_t f = 0; f < nI; ++f) {
        const int o = pm.owner[f], n = pm.neighbour[f];
        pushInternal(o, n, std::int64_t(f), S[f], Cf[f], s.C[std::size_t(n)] - s.C[std::size_t(o)]);
    }
    std::vector<char> done(man.patches.size(), 0);
    for (std::size_t i = 0; i < man.patches.size(); ++i) {
        const auto& A = man.patches[i];
        if (A.type != FoamPatchType::Cyclic || done[i]) continue;
        const auto j = std::size_t(std::ranges::find(man.patches, A.neighbourPatch, &FoamPatch::name)
                                   - man.patches.begin());
        if (j >= man.patches.size() || man.patches[j].nFaces != A.nFaces)
            return std::unexpected("mesh: cyclic '" + A.name + "' has no matching neighbourPatch");
        const auto& B = man.patches[j];
        done[i] = done[j] = 1;
        V3 sep{0, 0, 0};
        for (std::int64_t q = 0; q < A.nFaces; ++q) {
            const std::size_t fa = std::size_t(A.startFace + q), fb = std::size_t(B.startFace + q);
            const V3 t = Cf[fb] - Cf[fa];
            if (q == 0) sep = A.separation ? V3{(*A.separation)[0], (*A.separation)[1], (*A.separation)[2]} : t;
            if (!(mag(t - sep) <= 1e-8 * diag))
                return std::unexpected(std::format(
                    "mesh: cyclic '{}'/'{}' unmatched at pair {}: centre offset ({:.9g} {:.9g} {:.9g}) "
                    "vs separation ({:.9g} {:.9g} {:.9g}) (> 1e-8 of the bbox diagonal)",
                    A.name, B.name, q, t[0], t[1], t[2], sep[0], sep[1], sep[2]));
            if (!(mag(S[fa] + S[fb]) <= 1e-8 * mag(S[fa])))
                return std::unexpected(std::format("mesh: cyclic '{}'/'{}' unmatched at pair {}: face "
                                                   "areas differ",
                                                   A.name, B.name, q));
            const int ca = pm.owner[fa], cb = pm.owner[fb];
            if (ca == cb)
                return std::unexpected(std::format("mesh: cyclic '{}' couples cell {} to itself (one "
                                                   "cell across a periodic axis)",
                                                   A.name, ca));
            if (ca < cb)
                pushInternal(ca, cb, std::int64_t(fa), S[fa], Cf[fa],
                             (s.C[std::size_t(cb)] + (Cf[fa] - Cf[fb])) - s.C[std::size_t(ca)]);
            else
                pushInternal(cb, ca, std::int64_t(fb), S[fb], Cf[fb],
                             (s.C[std::size_t(ca)] + (Cf[fb] - Cf[fa])) - s.C[std::size_t(cb)]);
        }
    }

    // Boundary faces, non-cyclic patches in order.
    for (const auto& p : man.patches) {
        if (p.type == FoamPatchType::Cyclic) continue;
        MeshPatch mp{p.name, p.type, int(s.bOwn.size()), int(p.nFaces)};
        for (std::int64_t q = 0; q < p.nFaces; ++q) {
            const std::size_t f = std::size_t(p.startFace + q);
            const int o = pm.owner[f];
            s.bOwn.push_back(o);
            s.bSrc.push_back(std::int64_t(f));
            s.bS.push_back(S[f]);
            s.bCf.push_back(Cf[f]);
            s.bD.push_back(Cf[f] - s.C[std::size_t(o)]);
        }
        s.patches.push_back(std::move(mp));
    }

    // 2D: the empty-axis component of S and d is EXACTLY 0 off the empty faces.
    if (s.emptyAxis >= 0) {
        const std::size_t a = std::size_t(s.emptyAxis);
        for (std::size_t f = 0; f < s.iS.size(); ++f) s.iS[f][a] = s.iD[f][a] = 0.0;
        for (const auto& p : s.patches)
            if (p.type != FoamPatchType::Empty)
                for (int q = p.first; q < p.first + p.count; ++q)
                    s.bS[std::size_t(q)][a] = s.bD[std::size_t(q)][a] = 0.0;
    }

    for (std::size_t f = 0; f < s.iS.size(); ++f)
        if (!(dot(s.iD[f], s.iS[f]) > 0))
            return std::unexpected(std::format("mesh: face {} between cells {} and {}: d.S <= 0 "
                                               "(inverted or badly non-convex cell)",
                                               s.iSrc[f], s.iOwn[f], s.iNei[f]));

    s.zoneNames = man.cellZones;
    for (const auto& z : pm.zoneCells) s.zoneCells.emplace_back(z.begin(), z.end());
    return s;
}

// --- orderings ----------------------------------------------------------------

std::uint64_t spread3(std::uint64_t v)
{
    v &= 0x1fffff;
    v = (v | v << 32) & 0x1f00000000ffffULL;
    v = (v | v << 16) & 0x1f0000ff0000ffULL;
    v = (v | v << 8) & 0x100f00f00f00f00fULL;
    v = (v | v << 4) & 0x10c30c30c30c30c3ULL;
    v = (v | v << 2) & 0x1249249249249249ULL;
    return v;
}

// perm[new] = source cell, by Morton key of C, ties by source index.
std::vector<int> mortonPerm(const SrcMesh& s)
{
    V3 lo = s.C[0], hi = s.C[0];
    for (const V3& c : s.C)
        for (std::size_t a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], c[a]);
            hi[a] = std::max(hi[a], c[a]);
        }
    constexpr double kMax = 2097151.0;  // 21 bits
    std::vector<std::uint64_t> key(s.C.size());
    for (std::size_t c = 0; c < s.C.size(); ++c) {
        std::uint64_t k = 0;
        for (std::size_t a = 0; a < 3; ++a) {
            const double r = hi[a] - lo[a];
            const double q = r > 0 ? std::clamp(std::floor((s.C[c][a] - lo[a]) / r * kMax), 0.0, kMax) : 0.0;
            k |= spread3(std::uint64_t(q)) << a;
        }
        key[c] = k;
    }
    std::vector<int> perm(s.C.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::ranges::sort(perm, [&](int a, int b) {
        return key[std::size_t(a)] != key[std::size_t(b)] ? key[std::size_t(a)] < key[std::size_t(b)] : a < b;
    });
    return perm;
}

// The FaceMesh of s under perm (perm[new] = source cell).
FaceMesh assemble(const SrcMesh& s, const std::vector<int>& perm, CellOrder order)
{
    FaceMesh m;
    m.order = order;
    m.nC = s.nC;
    m.dim = s.dim;
    m.emptyAxis = s.emptyAxis;
    std::vector<int> newOf(std::size_t(s.nC));
    for (int n = 0; n < s.nC; ++n) newOf[std::size_t(perm[std::size_t(n)])] = n;

    for (std::size_t a = 0; a < 3; ++a) m.C[a].resize(std::size_t(s.nC));
    m.V.resize(std::size_t(s.nC));
    m.srcCell.resize(std::size_t(s.nC));
    for (int n = 0; n < s.nC; ++n) {
        const std::size_t c = std::size_t(perm[std::size_t(n)]);
        for (std::size_t a = 0; a < 3; ++a) m.C[a][std::size_t(n)] = s.C[c][a];
        m.V[std::size_t(n)] = s.V[c];
        m.srcCell[std::size_t(n)] = std::int64_t(c);
    }

    const std::size_t nI = s.iOwn.size(), nB = s.bOwn.size();
    m.nI = int(nI);
    m.nB = int(nB);
    for (std::size_t a = 0; a < 3; ++a) {
        m.S[a].resize(nI + nB);
        m.Cf[a].resize(nI + nB);
        m.d[a].resize(nI + nB);
    }
    m.own.resize(nI + nB);
    m.nei.resize(nI);
    m.srcFace.resize(nI + nB);

    // Internal: re-orient to own < nei (negating S and d exactly), then sort.
    std::vector<int> io(nI);
    std::vector<int> lo(nI), hi(nI);
    for (std::size_t f = 0; f < nI; ++f) {
        const int a = newOf[std::size_t(s.iOwn[f])], b = newOf[std::size_t(s.iNei[f])];
        lo[f] = std::min(a, b);
        hi[f] = std::max(a, b);
    }
    std::iota(io.begin(), io.end(), 0);
    std::ranges::sort(io, [&](int x, int y) {
        const std::size_t p = std::size_t(x), q = std::size_t(y);
        if (lo[p] != lo[q]) return lo[p] < lo[q];
        if (hi[p] != hi[q]) return hi[p] < hi[q];
        return s.iSrc[p] < s.iSrc[q];
    });
    for (std::size_t k = 0; k < nI; ++k) {
        const std::size_t f = std::size_t(io[k]);
        const bool swap = newOf[std::size_t(s.iOwn[f])] > newOf[std::size_t(s.iNei[f])];
        m.own[k] = lo[f];
        m.nei[k] = hi[f];
        m.srcFace[k] = s.iSrc[f];
        for (std::size_t a = 0; a < 3; ++a) {
            m.S[a][k] = swap ? -s.iS[f][a] : s.iS[f][a];
            m.d[a][k] = swap ? -s.iD[f][a] : s.iD[f][a];
            m.Cf[a][k] = s.iCf[f][a];
        }
    }

    // Boundary: per patch, sorted by (own, srcFace).
    for (const auto& p : s.patches) {
        std::vector<int> bo(std::size_t(p.count));
        std::iota(bo.begin(), bo.end(), p.first);
        std::ranges::sort(bo, [&](int x, int y) {
            const int ox = newOf[std::size_t(s.bOwn[std::size_t(x)])],
                      oy = newOf[std::size_t(s.bOwn[std::size_t(y)])];
            return ox != oy ? ox < oy : s.bSrc[std::size_t(x)] < s.bSrc[std::size_t(y)];
        });
        for (int q = 0; q < p.count; ++q) {
            const std::size_t f = std::size_t(bo[std::size_t(q)]), k = nI + std::size_t(p.first + q);
            m.own[k] = newOf[std::size_t(s.bOwn[f])];
            m.srcFace[k] = s.bSrc[f];
            for (std::size_t a = 0; a < 3; ++a) {
                m.S[a][k] = s.bS[f][a];
                m.Cf[a][k] = s.bCf[f][a];
                m.d[a][k] = s.bD[f][a];
            }
        }
        m.patches.push_back(p);
    }

    // CSR, ascending face index.
    m.cellStart.assign(std::size_t(s.nC) + 1, 0);
    for (std::size_t f = 0; f < nI + nB; ++f) ++m.cellStart[std::size_t(m.own[f]) + 1];
    for (std::size_t f = 0; f < nI; ++f) ++m.cellStart[std::size_t(m.nei[f]) + 1];
    std::partial_sum(m.cellStart.begin(), m.cellStart.end(), m.cellStart.begin());
    m.cellFace.resize(std::size_t(m.cellStart.back()));
    std::vector<int> fill(m.cellStart.begin(), m.cellStart.end() - 1);
    for (std::size_t f = 0; f < nI + nB; ++f) {
        m.cellFace[std::size_t(fill[std::size_t(m.own[f])]++)] = int(f);
        if (f < nI) m.cellFace[std::size_t(fill[std::size_t(m.nei[f])]++)] = int(f);
    }

    m.zoneNames = s.zoneNames;
    for (const auto& z : s.zoneCells) {
        std::vector<int> cz;
        cz.reserve(z.size());
        for (const int c : z) cz.push_back(newOf[std::size_t(c)]);
        std::ranges::sort(cz);
        m.zoneCells.push_back(std::move(cz));
    }
    return m;
}

} // namespace

std::span<const char* const> cellOrderNames()
{
    static constexpr const char* kNames[] = {"source", "morton", "colour"};
    return kNames;
}

std::expected<FaceMesh, std::string> FaceMesh::fromPolyMesh(const PolyMesh& pm, CellOrder order, int threads)
{
    auto s = buildSource(pm, threads);
    if (!s) return std::unexpected(s.error());
    if (order == CellOrder::Source) {
        std::vector<int> id(std::size_t(s->nC));
        std::iota(id.begin(), id.end(), 0);
        return assemble(*s, id, order);
    }
    const auto morton = mortonPerm(*s);
    if (order == CellOrder::Morton) return assemble(*s, morton, order);

    // ColourMajor: greedy colouring of the Morton mesh, then stable by colour.
    const FaceMesh mm = assemble(*s, morton, CellOrder::Morton);
    const la::CellGraph g = la::graphOf(mm);
    for (int r = 0; r < g.n; ++r)
        if (g.rowPtr[std::size_t(r) + 1] - g.rowPtr[std::size_t(r)] > 254)
            return std::unexpected(std::format("mesh: cell {} has more than 254 neighbours "
                                               "(colour-major ordering needs < 255 colours)",
                                               mm.srcCell[std::size_t(r)]));
    const auto colours = la::greedyColour(g);
    const auto cperm = la::colourMajorOrder(colours);  // new -> Morton index
    std::vector<int> perm(cperm.size());
    for (std::size_t n = 0; n < cperm.size(); ++n) perm[n] = morton[std::size_t(cperm[n])];
    FaceMesh m = assemble(*s, perm, order);
    m.colour.resize(cperm.size());
    for (std::size_t n = 0; n < cperm.size(); ++n) m.colour[n] = colours[std::size_t(cperm[n])];
    m.nColours = colours.empty() ? 0 : int(*std::ranges::max_element(colours)) + 1;
    return m;
}

std::expected<PolyMesh, std::string> FaceMesh::polyOf(const CurvMesh& m, const StructuredSpec& sp)
{
    const Grid& g = m.g;
    const int n[3] = {g.nx, g.ny, g.nz};
    if (m.axisFaces[0] != -1 || m.axisFaces[1] != -1)
        return std::unexpected("structured mesh: revolved meshes with axis faces are not supported "
                               "(masterplan_meshsolver P8)");
    if (m.wrap[2] == AxisWrap::MirrorZ)
        return std::unexpected("structured mesh: the axisymmetric wedge (MirrorZ) is not supported "
                               "(masterplan_meshsolver P8)");
    if (m.x.size() != g.size() || m.y.size() != g.size() || m.z.size() != g.size())
        return std::unexpected("structured mesh: node arrays do not match the grid");
    for (const auto t : sp.type)
        if (t == FoamPatchType::Wedge)
            return std::unexpected("structured mesh: wedge patches are not supported (P8)");
    const bool twoD = g.nz == 1;
    if (twoD && (sp.type[4] != FoamPatchType::Empty || sp.type[5] != FoamPatchType::Empty))
        return std::unexpected("structured mesh: nz == 1 is 2D: the k-/k+ patches must be empty");
    for (int f = 0; f < 6; ++f)
        if (sp.type[std::size_t(f)] == FoamPatchType::Empty && !(twoD && f >= 4))
            return std::unexpected("structured mesh: empty patches only on k-/k+ at nz == 1");

    int nc[3];
    bool periodic[3], shared[3];
    for (int a = 0; a < 3; ++a) {
        const std::size_t ua = std::size_t(a);
        if (sp.lo[ua] < 0 || sp.hi[ua] >= n[a] || sp.lo[ua] > sp.hi[ua])
            return std::unexpected(std::format("structured mesh: axis {} range [{}, {}] outside [0, {})",
                                               "ijk"[a], sp.lo[ua], sp.hi[ua], n[a]));
        nc[a] = sp.hi[ua] - sp.lo[ua] + 1;
        const bool cl = sp.type[2 * ua] == FoamPatchType::Cyclic,
                   ch = sp.type[2 * ua + 1] == FoamPatchType::Cyclic;
        if (cl != ch)
            return std::unexpected(std::format("structured mesh: axis {}: one side cyclic, the other not",
                                               "ijk"[a]));
        periodic[a] = cl;
        if (periodic[a] && (sp.lo[ua] != 0 || sp.hi[ua] != n[a] - 1))
            return std::unexpected(std::format("structured mesh: periodic axis {} needs the full range",
                                               "ijk"[a]));
        if (periodic[a] && m.wrap[ua] != AxisWrap::Cyclic && m.wrap[ua] != AxisWrap::Translate)
            return std::unexpected(std::format("structured mesh: periodic axis {} needs a Cyclic or "
                                               "Translate wrap",
                                               "ijk"[a]));
        shared[a] = periodic[a] && m.wrap[ua] == AxisWrap::Cyclic;
    }
    // Patch names: every emitted patch named, all distinct.
    std::vector<std::string> names;
    for (int f = 0; f < 6; ++f) {
        if (shared[f / 2]) continue;
        const auto& nm = sp.name[std::size_t(f)];
        if (nm.empty())
            return std::unexpected(std::format("structured mesh: patch {} has no name", f));
        if (std::ranges::find(names, nm) != names.end())
            return std::unexpected("structured mesh: patch name '" + nm + "' used twice");
        names.push_back(nm);
    }

    const int np[3] = {shared[0] ? nc[0] : nc[0] + 1, shared[1] ? nc[1] : nc[1] + 1,
                       shared[2] ? nc[2] : nc[2] + 1};
    const std::int64_t nPts = std::int64_t(np[0]) * np[1] * np[2],
                       nCells = std::int64_t(nc[0]) * nc[1] * nc[2];
    if (nPts > std::numeric_limits<std::int32_t>::max() / 4 || nCells > std::numeric_limits<std::int32_t>::max() / 8)
        return std::unexpected("structured mesh: too large for int32 polyMesh labels");

    PolyMesh pm;
    pm.px.resize(std::size_t(nPts));
    pm.py.resize(std::size_t(nPts));
    pm.pz.resize(std::size_t(nPts));
    for (int k = 0; k < np[2]; ++k)
        for (int j = 0; j < np[1]; ++j)
            for (int i = 0; i < np[0]; ++i) {
                const auto p = m.node(sp.lo[0] + i, sp.lo[1] + j, sp.lo[2] + k);
                const std::size_t id = std::size_t(i) + std::size_t(np[0]) * (std::size_t(j) + std::size_t(np[1]) * std::size_t(k));
                pm.px[id] = p[0];
                pm.py[id] = p[1];
                pm.pz[id] = p[2];
            }
    auto pid = [&](int i, int j, int k) {
        if (shared[0]) i %= nc[0];
        if (shared[1]) j %= nc[1];
        if (shared[2]) k %= nc[2];
        return std::int32_t(i + np[0] * (j + np[1] * k));
    };
    auto cid = [&](int i, int j, int k) { return std::int32_t(i + nc[0] * (j + nc[1] * k)); };

    struct Face
    {
        std::array<std::int32_t, 4> v;
        std::int32_t own, nei;
    };
    std::vector<Face> internal;
    std::array<std::vector<Face>, 6> bnd;
    // The 4 corner points of the face of cell (i,j,k) at offset off along a,
    // in the (t1, t2) order whose area vector points +a.
    auto corners = [&](int i, int j, int k, int a, int off) {
        const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
        std::array<std::int32_t, 4> v{};
        const int d1[4] = {0, 1, 1, 0}, d2[4] = {0, 0, 1, 1};
        for (int q = 0; q < 4; ++q) {
            int e[3] = {i, j, k};
            e[a] += off;
            e[t1] += d1[q];
            e[t2] += d2[q];
            v[std::size_t(q)] = pid(e[0], e[1], e[2]);
        }
        return v;
    };
    for (int k = 0; k < nc[2]; ++k)
        for (int j = 0; j < nc[1]; ++j)
            for (int i = 0; i < nc[0]; ++i) {
                const int ix[3] = {i, j, k};
                for (int a = 0; a < 3; ++a) {
                    const std::int32_t c = cid(i, j, k);
                    if (ix[a] == 0 && !shared[a]) {  // lo boundary face, outward -a
                        auto v = corners(i, j, k, a, 0);
                        bnd[std::size_t(2 * a)].push_back({{v[0], v[3], v[2], v[1]}, c, -1});
                    }
                    auto v = corners(i, j, k, a, 1);
                    if (ix[a] < nc[a] - 1 || shared[a]) {
                        int e[3] = {i, j, k};
                        e[a] = (ix[a] + 1) % nc[a];
                        const std::int32_t nb = cid(e[0], e[1], e[2]);
                        if (nb == c)
                            return std::unexpected(std::format("structured mesh: periodic axis {} has "
                                                               "one cell (a cell coupled to itself)",
                                                               "ijk"[a]));
                        if (c < nb)
                            internal.push_back({v, c, nb});
                        else
                            internal.push_back({{v[0], v[3], v[2], v[1]}, nb, c});
                    } else
                        bnd[std::size_t(2 * a + 1)].push_back({v, c, -1});
                }
            }
    std::ranges::stable_sort(internal, [](const Face& x, const Face& y) {
        return x.own != y.own ? x.own < y.own : x.nei < y.nei;
    });

    auto& man = pm.man;
    pm.faceStart.push_back(0);
    auto put = [&](const Face& f) {
        pm.faceNodes.insert(pm.faceNodes.end(), f.v.begin(), f.v.end());
        pm.faceStart.push_back(std::int32_t(pm.faceNodes.size()));
        pm.owner.push_back(f.own);
    };
    for (const auto& f : internal) {
        put(f);
        pm.neighbour.push_back(f.nei);
    }
    const double L[3] = {g.nx * g.hx, g.ny * g.hy, g.nz * g.hz};
    std::int64_t at = std::int64_t(internal.size());
    for (int f = 0; f < 6; ++f) {
        const int a = f / 2;
        if (shared[a]) continue;
        FoamPatch p;
        p.name = sp.name[std::size_t(f)];
        p.type = sp.type[std::size_t(f)];
        p.nFaces = std::int64_t(bnd[std::size_t(f)].size());
        p.startFace = at;
        if (p.type == FoamPatchType::Cyclic) {
            p.neighbourPatch = sp.name[std::size_t(f ^ 1)];
            std::array<double, 3> sep{0, 0, 0};
            sep[std::size_t(a)] = (f % 2 == 0 ? 1.0 : -1.0) * L[a];
            p.separation = sep;
        }
        for (const auto& fc : bnd[std::size_t(f)]) put(fc);
        at += p.nFaces;
        man.patches.push_back(std::move(p));
    }
    man.nPoints = nPts;
    man.nCells = nCells;
    man.nFaces = at;
    man.nInternalFaces = std::int64_t(internal.size());
    man.binary = false;
    man.labelBits = 32;
    return pm;
}

std::expected<FaceMesh, std::string> FaceMesh::fromCurvMesh(const CurvMesh& cm, const StructuredSpec& sp,
                                                             CellOrder order, int threads)
{
    auto pm = polyOf(cm, sp);
    if (!pm) return std::unexpected(pm.error());
    auto fm = fromPolyMesh(*pm, order, threads);
    if (!fm) return std::unexpected(fm.error());
    const int ni = sp.hi[0] - sp.lo[0] + 1, nj = sp.hi[1] - sp.lo[1] + 1;
    for (auto& s : fm->srcCell) {
        const int p = int(s);
        s = std::int64_t(cm.g.idx(sp.lo[0] + p % ni, sp.lo[1] + (p / ni) % nj, sp.lo[2] + p / (ni * nj)));
    }
    return fm;
}

// --- quality ------------------------------------------------------------------

MeshQuality meshQuality(const FaceMesh& fm, const PolyMesh& pm, int threads)
{
    MeshQuality q;
    const std::size_t nC = std::size_t(fm.nC), nI = std::size_t(fm.nI), nF = nI + std::size_t(fm.nB);
    auto vec = [](const std::array<std::vector<double>, 3>& v, std::size_t i) {
        return V3{v[0][i], v[1][i], v[2][i]};
    };
    assert(std::int64_t(nC) == pm.man.nCells);
    std::vector<int> newOf(nC);
    for (std::size_t n = 0; n < nC; ++n) {
        assert(fm.srcCell[n] >= 0 && fm.srcCell[n] < std::int64_t(nC));
        newOf[std::size_t(fm.srcCell[n])] = int(n);
    }

    // Per-face orthogonality and skewness (polyMeshTools definitions).
    std::vector<double> ortho(nI), skew(nF);
    {
        const int nw = workerCount(threads, nF);
        forBlocks(nw, nF, [&](int, std::size_t lo, std::size_t hi) {
            for (std::size_t f = lo; f < hi; ++f) {
                const V3 S = vec(fm.S, f), Cf = vec(fm.Cf, f);
                const std::size_t src = std::size_t(fm.srcFace[f]);
                const std::size_t a = std::size_t(pm.faceStart[src]), b = std::size_t(pm.faceStart[src + 1]);
                V3 Cpf, d;
                double fd0;
                if (f < nI) {
                    d = vec(fm.d, f);
                    ortho[f] = dot(d, S) / (mag(d) * mag(S) + kRootVSmall);
                    // Cf lies in the frame of the srcFace's polyMesh owner.
                    const int frame = newOf[std::size_t(pm.owner[src])];
                    if (frame == fm.nei[f]) {
                        Cpf = Cf - vec(fm.C, std::size_t(fm.nei[f]));
                        d = -d;
                    } else
                        Cpf = Cf - vec(fm.C, std::size_t(fm.own[f]));
                    fd0 = 0.2 * mag(d);
                } else {
                    Cpf = Cf - vec(fm.C, std::size_t(fm.own[f]));
                    const double ms = mag(S);
                    const V3 nrm = ms < kRootVSmall ? V3{0, 0, 0} : S / ms;
                    d = dot(nrm, Cpf) * nrm;
                    fd0 = 0.4 * mag(d);
                }
                const V3 sv = Cpf - (dot(S, Cpf) / (dot(S, d) + kRootVSmall)) * d;
                const V3 svHat = sv / (mag(sv) + kRootVSmall);
                double fd = fd0 + kRootVSmall;
                for (std::size_t p = a; p < b; ++p)
                    fd = std::max(fd, std::abs(dot(svHat, point(pm, pm.faceNodes[p]) - Cf)));
                skew[f] = mag(sv) / fd;
            }
        });
    }
    double minOrtho = 1, sumOrtho = 0;
    for (std::size_t f = 0; f < nI; ++f) {
        minOrtho = std::min(minOrtho, ortho[f]);
        sumOrtho += ortho[f];
    }
    constexpr double kDeg = 180.0 / 3.14159265358979323846;
    q.nonOrthMax = nI ? kDeg * std::acos(std::clamp(minOrtho, -1.0, 1.0)) : 0.0;
    q.nonOrthAvg = nI ? kDeg * std::acos(std::clamp(sumOrtho / double(nI), -1.0, 1.0)) : 0.0;
    q.skewMax = nF ? *std::ranges::max_element(skew) : 0.0;

    // Volumes (total in source order, as checkMesh sums), aspect, CFL width.
    q.vMin = std::numeric_limits<double>::max();
    q.vMax = 0;
    for (std::size_t s = 0; s < nC; ++s) q.vTotal += fm.V[std::size_t(newOf[s])];
    std::vector<double> width(nC);
    for (std::size_t c = 0; c < nC; ++c) {
        q.vMin = std::min(q.vMin, fm.V[c]);
        q.vMax = std::max(q.vMax, fm.V[c]);
        V3 sumMag{0, 0, 0};
        double sumS = 0;
        const int nfc = fm.cellStart[c + 1] - fm.cellStart[c];
        ++q.facesPerCell[std::size_t(std::min(nfc, 31))];
        for (int k = fm.cellStart[c]; k < fm.cellStart[c + 1]; ++k) {
            const std::size_t f = std::size_t(fm.cellFace[std::size_t(k)]);
            const V3 S = vec(fm.S, f);
            for (std::size_t a = 0; a < 3; ++a) sumMag[a] += std::abs(S[a]);
            const bool empty = f >= nI && [&] {
                for (const auto& p : fm.patches)
                    if (int(f - nI) >= p.first && int(f - nI) < p.first + p.count)
                        return p.type == FoamPatchType::Empty;
                return false;
            }();
            if (!empty) sumS += mag(S);
        }
        double mn = std::numeric_limits<double>::max(), mx = -mn;
        for (int a = 0; a < 3; ++a) {
            if (a == fm.emptyAxis) continue;
            mn = std::min(mn, sumMag[std::size_t(a)]);
            mx = std::max(mx, sumMag[std::size_t(a)]);
        }
        double ar = mx / (mn + kRootVSmall);
        if (fm.dim == 3)
            ar = std::max(ar, 1.0 / 6.0 * (sumMag[0] + sumMag[1] + sumMag[2])
                                  / std::pow(std::max(kRootVSmall, fm.V[c]), 2.0 / 3.0));
        q.aspectMax = std::max(q.aspectMax, ar);
        width[c] = fm.dim * fm.V[c] / sumS;
    }
    std::ranges::sort(width);
    q.cflWidthMin = width.front();
    q.cflWidthMax = width.back();
    q.cflWidthP1 = width[std::size_t(0.01 * double(nC - 1))];
    q.cflWidthMedian = width[(nC - 1) / 2];
    for (std::size_t f = 0; f < nI; ++f) {
        const double a = fm.V[std::size_t(fm.own[f])], b = fm.V[std::size_t(fm.nei[f])];
        q.growthMax = std::max(q.growthMax, std::max(a / b, b / a));
    }
    for (std::size_t f = 0; f < nF; ++f) {
        const std::size_t src = std::size_t(fm.srcFace[f]);
        ++q.nodesPerFace[std::size_t(std::min(pm.faceStart[src + 1] - pm.faceStart[src], 31))];
    }
    return q;
}

std::string meshReport(const FaceMesh& fm, const MeshQuality& q)
{
    std::string r = std::format("mesh: {} cells, {} internal faces, {} boundary faces, {}D{}\n", fm.nC,
                                fm.nI, fm.nB, fm.dim,
                                fm.emptyAxis >= 0 ? std::format(" (empty axis {})", "xyz"[fm.emptyAxis])
                                                  : std::string());
    r += std::format("order: {}{}\n", cellOrderNames()[std::size_t(fm.order)],
                     fm.order == CellOrder::ColourMajor ? std::format(" ({} colours)", fm.nColours)
                                                        : std::string());
    r += "patches:\n";
    for (const auto& p : fm.patches)
        r += std::format("  {:<24} {:<14} {:>9} faces\n", p.name, foamPatchTypeName(p.type), p.count);
    if (!fm.zoneNames.empty()) {
        r += "cell zones:\n";
        for (std::size_t z = 0; z < fm.zoneNames.size(); ++z)
            r += std::format("  {:<24} {:>9} cells\n", fm.zoneNames[z], fm.zoneCells[z].size());
    }
    r += std::format("volume: min {:.10g}  max {:.10g}  total {:.10g}\n", q.vMin, q.vMax, q.vTotal);
    r += std::format("non-orthogonality: max {:.10g} deg  average {:.10g} deg\n", q.nonOrthMax, q.nonOrthAvg);
    r += std::format("skewness max {:.10g}   aspect ratio max {:.10g}\n", q.skewMax, q.aspectMax);
    r += std::format("CFL width (dim V / sum |S|): min {:.6g}  p1 {:.6g}  median {:.6g}  max {:.6g}  "
                     "(median/min {:.3g})\n",
                     q.cflWidthMin, q.cflWidthP1, q.cflWidthMedian, q.cflWidthMax,
                     q.cflWidthMedian / q.cflWidthMin);
    r += std::format("growth max (V_nb / V over internal faces): {:.6g}\n", q.growthMax);
    auto hist = [](const std::array<int, 32>& h) {
        std::string s;
        for (std::size_t i = 0; i < h.size(); ++i)
            if (h[i]) s += std::format("  {}{}:{}", i, i == 31 ? "+" : "", h[i]);
        return s;
    };
    r += "faces per cell:" + hist(q.facesPerCell) + "\n";
    r += "nodes per face:" + hist(q.nodesPerFace) + "\n";
    if (q.growthMax > 1.25)
        r += std::format("warning: adjacent cell volume ratio {:.3g} > 1.25 (refinement jumps reflect "
                         "near-grid content; docs/mesh.md)\n",
                         q.growthMax);
    return r;
}

} // namespace dns
