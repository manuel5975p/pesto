// fvOptions actuationDiskSource, host side (docs/fv.md "fvOptions"): OpenFOAM
// v2606's polyMesh::findCell for the monitor points and the cellZone's volume
// fractions. Declarations: dns/fvsimple.hpp.
#include "dns/fvsimple.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <span>

namespace dns {
namespace {

using V3 = std::array<double, 3>;

V3 sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
V3 cross(const V3& a, const V3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double mag(const V3& a) { return std::sqrt(dot(a, a)); }

constexpr double kSmall = 1e-15, kRootVSmall = 1e-150, kGreat = 1e15, kVGreat = 1e300;

// tetrahedron(a, b, c, d) of OpenFOAM (tetrahedronI.H), verbatim.
struct Tet
{
    V3 a, b, c, d;

    double magV() const { return (1.0 / 6.0) * dot(cross(sub(b, a), sub(c, a)), sub(d, a)); }
    double circumRadius() const
    {
        const V3 A = sub(b, a), B = sub(c, a), C = sub(d, a);
        const double lambda = dot(C, C) - dot(A, C), mu = dot(B, B) - dot(A, B);
        const V3 ba = cross(B, A), ca = cross(C, A);
        const V3 num{lambda * ba[0] - mu * ca[0], lambda * ba[1] - mu * ca[1], lambda * ba[2] - mu * ca[2]};
        const double denom = dot(C, ba);
        if (std::abs(denom) < kRootVSmall) return kGreat;
        return mag(V3{0.5 * (A[0] + num[0] / denom), 0.5 * (A[1] + num[1] / denom), 0.5 * (A[2] + num[2] / denom)});
    }
    double quality() const
    {
        const double r = std::min(circumRadius(), kGreat);
        return magV() / (8.0 / (9.0 * std::sqrt(3.0)) * (r * r * r) + kRootVSmall);
    }
    // triangle(p0, p1, p2).unitNormal(): 0.5 (p1 - p0) ^ (p2 - p0), normalise(ROOTVSMALL)
    static V3 unitNormal(const V3& p0, const V3& p1, const V3& p2)
    {
        const V3 x = cross(sub(p1, p0), sub(p2, p0));
        V3 n{0.5 * x[0], 0.5 * x[1], 0.5 * x[2]};
        const double s = mag(n);
        if (s < kRootVSmall) return {0, 0, 0};
        return {n[0] / s, n[1] / s, n[2] / s};
    }
    // tetrahedron::inside: outside only on a positive distance > SMALL from a side.
    bool inside(const V3& p) const
    {
        if (dot(sub(p, b), unitNormal(b, c, d)) > kSmall) return false;
        if (dot(sub(p, a), unitNormal(a, d, c)) > kSmall) return false;
        if (dot(sub(p, a), unitNormal(a, b, d)) > kSmall) return false;
        if (dot(sub(p, a), unitNormal(a, c, b)) > kSmall) return false;
        return true;
    }
};

struct Finder
{
    const PolyMesh& pm;
    const FaceMesh& fm;
    std::vector<int> fmCell; // OpenFOAM cell -> FaceMesh cell

    V3 point(std::int32_t i) const { return {pm.px[std::size_t(i)], pm.py[std::size_t(i)], pm.pz[std::size_t(i)]}; }
    V3 centre(std::int64_t foamCell) const
    {
        const std::size_t c = std::size_t(fmCell[std::size_t(foamCell)]);
        return {fm.C[0][c], fm.C[1][c], fm.C[2][c]};
    }
    std::span<const std::int32_t> face(std::int64_t f) const
    {
        return {pm.faceNodes.data() + pm.faceStart[std::size_t(f)],
                std::size_t(pm.faceStart[std::size_t(f) + 1] - pm.faceStart[std::size_t(f)])};
    }

    // polyMeshTetDecomposition::minQuality(mesh, cC, fI, isOwner, faceBasePtI)
    double minQuality(const V3& cC, std::int64_t fI, bool isOwner, int base) const
    {
        const auto f = face(fI);
        const int n = int(f.size());
        const V3 tetBasePt = point(f[std::size_t(base)]);
        double q = kVGreat;
        for (int tetPt = 1; tetPt < n - 1; ++tetPt) {
            const int facePt = (tetPt + base) % n, other = (facePt + 1) % n;
            const V3 pA = point(f[std::size_t(isOwner ? facePt : other)]);
            const V3 pB = point(f[std::size_t(isOwner ? other : facePt)]);
            q = std::min(q, Tet{cC, tetBasePt, pA, pB}.quality());
        }
        return q;
    }

    // tetBasePtIs[fI] (findFaceBasePts, tol = minTetQuality = SMALL^2): the
    // first face point whose fan has min quality > tol seen from the owner
    // and (internal face) the neighbour; none: 0 (tetIndices::faceTriIs).
    int basePoint(std::int64_t fI) const
    {
        const double tol = kSmall * kSmall;
        const auto f = face(fI);
        const bool internal = fI < pm.man.nInternalFaces;
        const V3 oCc = centre(pm.owner[std::size_t(fI)]);
        const V3 nCc = internal ? centre(pm.neighbour[std::size_t(fI)]) : V3{};
        for (int b = 0; b < int(f.size()); ++b) {
            const double own = minQuality(oCc, fI, true, b);
            const double q = internal ? std::min(own, minQuality(nCc, fI, false, b)) : own;
            if (q > tol) return b;
        }
        return 0;
    }

    // polyMeshTetDecomposition::findTet(mesh, celli, p) != none
    bool inCell(std::int64_t celli, std::span<const std::int64_t> faces, const V3& p) const
    {
        const V3 cC = centre(celli);
        for (const std::int64_t fI : faces) {
            const auto f = face(fI);
            const int n = int(f.size()), base = basePoint(fI);
            const bool own = pm.owner[std::size_t(fI)] == celli;
            for (int tetPt = 1; tetPt < n - 1; ++tetPt) {
                const int facePt = (tetPt + base) % n, other = (facePt + 1) % n;
                // tetIndices::faceTriIs: owner (base, facePt, other), neighbour flipped
                const V3 t0 = point(f[std::size_t(base)]);
                const V3 t1 = point(f[std::size_t(own ? facePt : other)]);
                const V3 t2 = point(f[std::size_t(own ? other : facePt)]);
                if (Tet{cC, t0, t1, t2}.inside(p)) return true;
            }
        }
        return false;
    }
};

// findCell for each point: {the lowest cell holding it (-1: none), how many hold it}.
std::expected<std::vector<std::pair<std::int64_t, int>>, std::string> findCells(const PolyMesh& pm, const FaceMesh& fm,
                                                                                std::span<const V3> pts)
{
    const std::size_t nC = std::size_t(pm.man.nCells), nF = std::size_t(pm.man.nFaces),
                      nI = std::size_t(pm.man.nInternalFaces);
    if (std::size_t(fm.nC) != nC || fm.srcCell.size() != nC)
        return std::unexpected("findCell: the FaceMesh is not this polyMesh's");
    Finder F{pm, fm, std::vector<int>(nC, -1)};
    for (std::size_t c = 0; c < nC; ++c) F.fmCell[std::size_t(fm.srcCell[c])] = int(c);
    std::vector<std::pair<std::int64_t, int>> out;
    std::vector<std::uint8_t> bits(nC);
    for (const V3& p : pts) {
        // Candidates: the cells whose point bounding box holds p (a superset
        // of the cells whose tets, hulls of the cell's points, hold it within
        // SMALL): 6 flags per cell, one pass over the faces.
        std::ranges::fill(bits, std::uint8_t(0));
        const double pad[3] = {1e-12 * (1.0 + std::abs(p[0])), 1e-12 * (1.0 + std::abs(p[1])),
                               1e-12 * (1.0 + std::abs(p[2]))};
        for (std::size_t f = 0; f < nF; ++f) {
            std::uint8_t b = 0;
            for (const std::int32_t i : F.face(std::int64_t(f))) {
                const V3 x = F.point(i);
                for (int q = 0; q < 3; ++q) {
                    if (x[std::size_t(q)] <= p[std::size_t(q)] + pad[q]) b |= std::uint8_t(1u << (2 * q));
                    if (x[std::size_t(q)] >= p[std::size_t(q)] - pad[q]) b |= std::uint8_t(1u << (2 * q + 1));
                }
            }
            bits[std::size_t(pm.owner[f])] |= b;
            if (f < nI) bits[std::size_t(pm.neighbour[f])] |= b;
        }
        std::map<std::int64_t, std::vector<std::int64_t>> cand; // ascending cell index -> its faces
        for (std::size_t c = 0; c < nC; ++c)
            if (bits[c] == 63) cand[std::int64_t(c)];
        std::int64_t hit = -1;
        int holders = 0;
        if (!cand.empty()) {
            for (std::size_t f = 0; f < nF; ++f) {
                if (auto it = cand.find(pm.owner[f]); it != cand.end()) it->second.push_back(std::int64_t(f));
                if (f < nI)
                    if (auto it = cand.find(pm.neighbour[f]); it != cand.end()) it->second.push_back(std::int64_t(f));
            }
            for (const auto& [c, faces] : cand) {
                for (const std::int64_t f : faces)
                    for (const FoamPatch& pa : pm.man.patches)
                        if (pa.type == FoamPatchType::Cyclic && f >= pa.startFace && f < pa.startFace + pa.nFaces)
                            return std::unexpected(std::format(
                                "findCell: point ({:g} {:g} {:g}) is next to cyclic patch '{}' (cell {}): the tet base "
                                "points of a coupled face are not built",
                                p[0], p[1], p[2], pa.name, c));
                if (F.inCell(c, faces, p)) {
                    if (hit < 0) hit = c;
                    ++holders;
                }
            }
        }
        out.emplace_back(hit, holders);
    }
    return out;
}

} // namespace

std::expected<std::int64_t, std::string> foamFindCell(const PolyMesh& pm, const FaceMesh& fm,
                                                      const std::array<double, 3>& p)
{
    const V3 one[1] = {p};
    auto r = findCells(pm, fm, one);
    if (!r) return std::unexpected(r.error());
    return (*r)[0].first;
}

std::expected<void, std::string> buildDiskSetup(const PolyMesh& pm, const FaceMesh& fm, const FoamCase& c,
                                                FvSimpleSetup& s)
{
    s.disks.clear();
    if (c.disks.empty()) return {};
    const std::size_t nC = std::size_t(pm.man.nCells);
    std::vector<int> fmCell(nC, -1);
    for (std::size_t i = 0; i < nC; ++i) fmCell[std::size_t(fm.srcCell[i])] = int(i);
    std::vector<V3> pts;
    for (const FvDisk& k : c.disks) pts.insert(pts.end(), k.monitor.begin(), k.monitor.end());
    auto found = findCells(pm, fm, pts);
    if (!found) return std::unexpected(found.error());
    std::size_t at = 0;
    for (const FvDisk& k : c.disks) {
        const std::string where = "fvOptions/" + k.name + " (actuationDiskSource)";
        FvDiskSetup d;
        const auto z = std::ranges::find(pm.man.cellZones, k.cellZone);
        if (z == pm.man.cellZones.end()) return std::unexpected(where + ": cellZone '" + k.cellZone + "' is not in the mesh");
        // cellSetOption: cells_ = the zone, V_ = sum of their volumes in that order
        for (const std::int32_t cell : pm.zoneCells[std::size_t(z - pm.man.cellZones.begin())]) {
            d.cells.push_back(fmCell[std::size_t(cell)]);
            d.V += fm.V[std::size_t(d.cells.back())];
        }
        if (!(d.V > 1e-300))
            return std::unexpected(where + ": cellZone '" + k.cellZone + "' has no volume (OpenFOAM applies nothing)");
        for (const int cc : d.cells) d.frac.push_back(fm.V[std::size_t(cc)] / d.V);
        for (std::size_t i = 0; i < k.monitor.size(); ++i, ++at) {
            const std::int64_t cell = (*found)[at].first;
            d.monitorHolders.push_back((*found)[at].second);
            const V3& p = k.monitor[i];
            if (cell < 0)
                return std::unexpected(std::format("{}: No owner cell found for point ({:g} {:g} {:g})", where, p[0],
                                                   p[1], p[2]));
            if (std::ranges::find(d.monitorSrc, cell) == d.monitorSrc.end()) d.monitorSrc.push_back(cell);
        }
        std::ranges::sort(d.monitorSrc);
        for (const std::int64_t cell : d.monitorSrc) d.monitor.push_back(fmCell[std::size_t(cell)]);
        s.disks.push_back(std::move(d));
    }
    return {};
}

} // namespace dns
