#include "dns/fvshock.hpp"

#include "dns/shockflux.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <ranges>

namespace dns {
namespace {

// The cell across CSR entry e of cell c, -1 on a boundary face.
int across(const FaceMesh& m, std::size_t c, int e)
{
    const std::size_t f = std::size_t(m.cellFace[std::size_t(e)]);
    if (f >= std::size_t(m.nI)) return -1;
    return std::size_t(m.own[f]) == c ? m.nei[f] : m.own[f];
}

// The CSR entry of face f in cell c. Pre: f is a face of c.
int entryOf(const FaceMesh& m, std::size_t c, int f)
{
    int e = m.cellStart[c];
    while (m.cellFace[std::size_t(e)] != f) ++e;
    return e;
}

// Unit outward normal of entry e of cell c.
std::array<double, 3> normalOf(const FaceMesh& m, std::size_t c, int e)
{
    const std::size_t f = std::size_t(m.cellFace[std::size_t(e)]);
    const double sign = std::size_t(m.own[f]) == c ? 1.0 : -1.0;
    const double S[3] = {m.S[0][f], m.S[1][f], m.S[2][f]};
    const double mag = std::sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
    return {sign * S[0] / mag, sign * S[1] / mag, sign * S[2] / mag};
}

// opp[e] per CSR entry: the rule of fvShockLines, -1 for an inactive entry or none.
std::vector<int> oppositeEntries(const FaceMesh& m, const FvFaceGeom& geom)
{
    std::vector<int> opp(m.cellFace.size(), -1);
    for (std::size_t c = 0; c < std::size_t(m.nC); ++c)
        for (int e = m.cellStart[c]; e < m.cellStart[c + 1]; ++e) {
            if (!geom.active[std::size_t(m.cellFace[std::size_t(e)])]) continue;
            const auto ne = normalOf(m, c, e);
            int best = -1;
            double bestCos = -0.5;
            for (int q = m.cellStart[c]; q < m.cellStart[c + 1]; ++q) {
                const std::size_t fq = std::size_t(m.cellFace[std::size_t(q)]);
                if (q == e || !geom.active[fq]) continue;
                const auto nq = normalOf(m, c, q);
                const double cs = ne[0] * nq[0] + ne[1] * nq[1] + ne[2] * nq[2];
                if (cs > -0.5) continue;
                const bool tie = best >= 0 && cs == bestCos && m.srcFace[fq] < m.srcFace[std::size_t(m.cellFace[std::size_t(best)])];
                if (best < 0 || cs < bestCos || tie) {
                    best = q;
                    bestCos = cs;
                }
            }
            opp[std::size_t(e)] = best;
        }
    return opp;
}

// Two cells out of c through entry e: the neighbour, then the opposite of the face entered
// by. -1 where the line stops.
int twoOut(const FaceMesh& m, const std::vector<int>& opp, std::size_t c, int e)
{
    const int c1 = across(m, c, e);
    if (c1 < 0) return -1;
    const int o = opp[std::size_t(entryOf(m, std::size_t(c1), m.cellFace[std::size_t(e)]))];
    return o < 0 ? -1 : across(m, std::size_t(c1), o);
}

} // namespace

FvShockLines fvShockLines(const FaceMesh& m, const FvFaceGeom& geom)
{
    const std::vector<int> opp = oppositeEntries(m, geom);
    FvShockLines l;
    l.lineStart.assign(std::size_t(m.nC) + 1, 0);
    for (std::size_t c = 0; c < std::size_t(m.nC); ++c) {
        for (int e = m.cellStart[c]; e < m.cellStart[c + 1]; ++e) {
            const int o = opp[std::size_t(e)];
            if (o < 0) continue;
            // One line per opposite pair; a one-way pick keeps its own line.
            if (!(e < o || opp[std::size_t(o)] != e)) continue;
            const int up = twoOut(m, opp, c, e), dn = twoOut(m, opp, c, o);
            if (up < 0 || dn < 0) continue;
            l.line.push_back(dn);
            l.line.push_back(up);
        }
        l.lineStart[c + 1] = int(l.line.size() / 2);
    }
    l.faceLine.assign(2 * std::size_t(m.nI), -1);
    for (std::size_t f = 0; f < std::size_t(m.nI); ++f) {
        const std::size_t own = std::size_t(m.own[f]), nei = std::size_t(m.nei[f]);
        const int eo = opp[std::size_t(entryOf(m, own, int(f)))], en = opp[std::size_t(entryOf(m, nei, int(f)))];
        if (eo >= 0) l.faceLine[2 * f] = across(m, own, eo);
        if (en >= 0) l.faceLine[2 * f + 1] = across(m, nei, en);
    }
    return l;
}

long fvUnlinedCells(const FaceMesh& m, const FvFaceGeom& geom)
{
    const std::vector<int> opp = oppositeEntries(m, geom);
    long n = 0;
    for (std::size_t c = 0; c < std::size_t(m.nC); ++c) {
        const int first = m.cellStart[c], last = m.cellStart[c + 1];
        if (m.wedgeHalf > 0.0 && last - first == 5) continue; // an axis prism
        const bool unlined = std::ranges::any_of(std::views::iota(first, last), [&](int e) {
            return geom.active[std::size_t(m.cellFace[std::size_t(e)])] && opp[std::size_t(e)] < 0;
        });
        n += unlined ? 1 : 0;
    }
    return n;
}

std::expected<void, std::string> fvShockMeshCheck(const FaceMesh& fm)
{
    auto g = fvFaceGeom(fm, FvFaceGradRule::AlongD, 1.0);
    if (!g) return std::unexpected(g.error());
    const long n = fvUnlinedCells(fm, *g);
    if (n == 0) return {};
    return std::unexpected(std::format("mesh: --shock needs a mesh line through every face: {} of {} cells have a face with no "
                                       "opposite (tets, triangles; docs/nozzle.md \"Sensor\", issues.md Z11)",
                                       n, fm.nC));
}

void fvShockSense(const FvShockLines& l, std::span<const double> p, std::span<std::uint8_t> b)
{
    for (std::size_t c = 0; c < p.size(); ++c) {
        double v = 0.0;
        for (int q = l.lineStart[c]; q < l.lineStart[c + 1]; ++q) {
            const double jump = shockJump(p[std::size_t(l.line[2 * std::size_t(q)])], p[c], p[std::size_t(l.line[2 * std::size_t(q) + 1])]);
            v = jump > v || jump != jump ? jump : v; // a NaN sticks
        }
        b[c] = shockByte(v);
    }
}

} // namespace dns
