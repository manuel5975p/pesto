#include "dns/fvles.hpp"

#include "dns/tfles.hpp"

#include <cmath>
#include <format>

namespace dns {

std::vector<double> fvLesDelta2(const FaceMesh& fm)
{
    const std::size_t nI = std::size_t(fm.nI);
    // A 2D mesh's depth: V / (cell area) from the empty face of its owner.
    const double thick = fm.emptyAxis >= 0 ? [&] {
        for (const MeshPatch& p : fm.patches)
            if (p.type == FoamPatchType::Empty && p.count > 0) {
                const std::size_t f = nI + std::size_t(p.first);
                const std::size_t c = std::size_t(fm.own[f]);
                return fm.V[c] / std::abs(fm.S[std::size_t(fm.emptyAxis)][f]);
            }
        return 1.0;
    }() : 1.0;
    std::vector<double> d2(std::size_t(fm.nC));
    for (std::size_t c = 0; c < d2.size(); ++c) {
        const double r3 = std::cbrt(fm.V[c]);
        d2[c] = fm.emptyAxis >= 0 ? fm.V[c] / thick : r3 * r3;
    }
    return d2;
}

std::expected<FvTflesWalk, std::string> fvTflesWalk(const FaceMesh& fm, const FvFaceGeom& geom)
{
    FvTflesWalk w;
    for (int a = 0; a < 3; ++a)
        if (a != fm.emptyAxis) w.axis[w.nAxes++] = a;
    const std::size_t nC = std::size_t(fm.nC), nI = std::size_t(fm.nI);
    w.next.assign(std::size_t(2 * w.nAxes) * nC, -1);
    for (std::size_t c = 0; c < nC; ++c)
        for (int s = 0; s < w.nAxes; ++s)
            for (int dir = 0; dir < 2; ++dir) {
                const std::size_t a = std::size_t(w.axis[s]);
                const double sg = dir == 0 ? 1.0 : -1.0;
                double best = 0.0;
                int pick = -1;
                for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
                    const std::size_t f = std::size_t(fm.cellFace[std::size_t(q)]);
                    if (!geom.active[f]) continue;
                    const double o = std::size_t(fm.own[f]) == c ? 1.0 : -1.0;
                    const double mag = std::hypot(fm.S[0][f], fm.S[1][f], fm.S[2][f]);
                    const double al = sg * o * fm.S[a][f] / mag;
                    // Exact ties (the sub-faces of a split face) go to the lowest SOURCE
                    // face: the walk is the same under every cell numbering (gpu_fv_matrix (d)).
                    const bool tie = pick >= 0 && al == best && fm.srcFace[f] < fm.srcFace[std::size_t(pick)];
                    if (al > best || tie) {
                        best = al;
                        pick = int(f);
                    }
                }
                if (pick < 0)
                    return std::unexpected(std::format("TFLES sensor walk: cell {} (source {}) has no face toward {}e_{}",
                                                       c, fm.srcCell[c], dir == 0 ? '+' : '-', "xyz"[a]));
                const std::size_t f = std::size_t(pick);
                if (f >= nI) continue; // a boundary face: the walk stops
                w.next[(std::size_t(s) * 2 + std::size_t(dir)) * nC + c] =
                    std::size_t(fm.own[f]) == c ? fm.nei[f] : fm.own[f];
            }
    // Edge layers: owners of active boundary faces are layer 0, then face neighbours.
    std::vector<int> layer(nC, kTflesEdgeRows);
    for (std::size_t b = 0; b < std::size_t(fm.nB); ++b)
        if (geom.active[nI + b]) layer[std::size_t(fm.own[nI + b])] = 0;
    for (int k = 1; k < kTflesEdgeRows; ++k)
        for (std::size_t f = 0; f < nI; ++f) {
            const std::size_t o = std::size_t(fm.own[f]), n = std::size_t(fm.nei[f]);
            if (layer[o] == k - 1 && layer[n] > k) layer[n] = k;
            if (layer[n] == k - 1 && layer[o] > k) layer[o] = k;
        }
    w.edge.assign(nC, 0);
    for (std::size_t c = 0; c < nC; ++c) w.edge[c] = layer[c] < kTflesEdgeRows ? 1 : 0;
    w.lapW.assign(fm.cellFace.size(), 0.0);
    for (std::size_t q = 0; q < fm.cellFace.size(); ++q) {
        const std::size_t f = std::size_t(fm.cellFace[q]);
        if (f >= nI) continue;
        w.lapW[q] = geom.corr[0][f] * fm.S[0][f] + geom.corr[1][f] * fm.S[1][f] + geom.corr[2][f] * fm.S[2][f];
    }
    return w;
}

} // namespace dns
