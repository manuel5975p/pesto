// cyclicAMI on the FV path's host setup (docs/fvpimple.md "cyclicAMI"): the
// matrix graph with the AMI couplings and FvSimpleSetup::ami -- the patch
// geometry of cyclicAMIFvPatch (makeWeights, delta) and basicFvGeometryScheme
// (nonOrthDeltaCoeffs, nonOrthCorrectionVectors on a coupled patch), operand
// for operand. Compiled -ffp-contract=off.
#include "dns/fvsimple.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

namespace dns {
namespace {

using V3 = std::array<double, 3>;

double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
double mag(const V3& a) { return std::sqrt(dot(a, a)); }

// The polyMesh patch index of a FaceMesh / case patch name, -1: none.
int patchIndex(const PolyManifest& man, const std::string& name)
{
    const auto it = std::ranges::find(man.patches, name, &FoamPatch::name);
    return it == man.patches.end() ? -1 : int(it - man.patches.begin());
}

} // namespace

std::expected<FvGraph, std::string> buildFvGraph(const PolyMesh& pm, const FaceMesh& fm)
{
    FvGraph G;
    const auto& pt = pm.man.patches;
    for (std::size_t i = 0; i < pt.size(); ++i) {
        if (pt[i].type != FoamPatchType::CyclicAMI) continue;
        const int nb = patchIndex(pm.man, pt[i].neighbourPatch);
        if (nb < 0) return std::unexpected("cyclicAMI '" + pt[i].name + "': neighbourPatch not in the mesh");
        if (nb < int(i)) continue; // the owner is the lower patch index (cyclicAMIPolyPatch::owner())
        auto a = foamCyclicAmi(pm, pm.px, pm.py, pm.pz, int(i), nb);
        if (!a) return std::unexpected("cyclicAMI '" + pt[i].name + "' / '" + pt[std::size_t(nb)].name + "': " + a.error());
        G.ami.push_back(FvAmiPair{int(i), nb, std::move(*a)});
    }
    if (G.ami.empty()) {
        G.g = la::graphOf(fm);
        G.pat = la::ellPattern(G.g);
        return G;
    }
    // FaceMesh cell of an OpenFOAM cell (Source order: the identity, not assumed).
    std::vector<int> cellOf(std::size_t(fm.nC), -1);
    for (int c = 0; c < fm.nC; ++c) cellOf[std::size_t(fm.srcCell[std::size_t(c)])] = c;
    std::vector<std::pair<int, int>> extra;
    for (const FvAmiPair& p : G.ami) {
        const FoamPatch& ps = pt[std::size_t(p.src)];
        const FoamPatch& pg = pt[std::size_t(p.tgt)];
        auto side = [&](const FoamAmiSide& S, const FoamPatch& me, const FoamPatch& other) {
            for (std::int64_t i = 0; i < me.nFaces; ++i) {
                const int P = cellOf[std::size_t(pm.owner[std::size_t(me.startFace + i)])];
                for (int j = S.start[std::size_t(i)]; j < S.start[std::size_t(i) + 1]; ++j)
                    extra.emplace_back(P, cellOf[std::size_t(pm.owner[std::size_t(other.startFace + S.addr[std::size_t(j)])])]);
            }
        };
        side(p.ami.src, ps, pg);
        side(p.ami.tgt, pg, ps);
    }
    for (const auto& [r, c] : extra)
        if (r == c) return std::unexpected("cyclicAMI: a cell is coupled to itself across the interface");
    const la::CellGraph plain = la::graphOf(fm);
    for (const auto& [r, c] : extra)
        if (la::graphSlot(plain, r, c) >= 0)
            return std::unexpected(std::format("cyclicAMI: cells {} and {} are joined both by a face and by a cyclicAMI "
                                               "address (not built)",
                                               fm.srcCell[std::size_t(r)], fm.srcCell[std::size_t(c)]));
    G.g = la::graphOf(fm, extra);
    G.pat = la::ellPattern(G.g);
    return G;
}

std::expected<void, std::string> bindFvAmi(const FaceMesh& fm, const FoamCase& c, const la::CellGraph& g,
                                           const la::EllPattern& pat, const std::vector<FvAmiPair>& ami,
                                           FvSimpleSetup& s)
{
    FvAmiSetup& A = s.ami;
    const std::size_t nB = std::size_t(s.nB);
    A = FvAmiSetup{};
    A.bA.assign(nB, -1);
    A.aStart.push_back(0);
    // fvPatch geometry of boundary face b: nf() = Sf / magSf, coupledFvPatch::delta() = Cf - Cn.
    auto nOf = [&](int b) { return V3{s.bN[std::size_t(b)], s.bN[nB + std::size_t(b)], s.bN[2 * nB + std::size_t(b)]}; };
    auto dOf = [&](int b) {
        const std::size_t f = std::size_t(s.bFace[std::size_t(b)]);
        return V3{fm.d[0][f], fm.d[1][f], fm.d[2][f]};
    };
    for (const FvAmiPair& p : ami) {
        // c.patches is in polyMesh boundary order (every patch).
        if (p.src < 0 || p.tgt < 0 || std::size_t(std::max(p.src, p.tgt)) >= s.patchFaces.size())
            return std::unexpected("cyclicAMI: a pair's patch is not the case's");
        const std::vector<int>& fs = s.patchFaces[std::size_t(p.src)];
        const std::vector<int>& ft = s.patchFaces[std::size_t(p.tgt)];
        FvAmiSetup::Report rep;
        rep.src = c.patches[std::size_t(p.src)].name;
        rep.tgt = c.patches[std::size_t(p.tgt)].name;
        auto side = [&](const FoamAmiSide& S, const std::vector<int>& mine, const std::vector<int>& other,
                        std::array<double, 3>& sum, int& nFaces, int& low) -> std::expected<void, std::string> {
            if (S.start.size() != mine.size() + 1)
                return std::unexpected("cyclicAMI: the AMI's face count is not the patch's");
            nFaces = int(mine.size());
            double mn = 1e300, mx = -1e300, tot = 0;
            for (std::size_t i = 0; i < mine.size(); ++i) {
                const int b = mine[i];
                if (b < 0) return std::unexpected("cyclicAMI: a patch face is not a boundary face of the setup");
                const double ws = i < S.wSum.size() ? S.wSum[i] : 0.0;
                mn = std::min(mn, ws);
                mx = std::max(mx, ws);
                tot += ws;
                low += ws < 0.2;
                const int a = A.nA++;
                A.aB.push_back(b);
                A.bA[std::size_t(b)] = a;
                const int P = s.bOwn[std::size_t(b)];
                // interpolate(nbr field): sum_j w_j fld[addr_j] from Zero, address order
                double nbrDelta = 0;
                V3 nbrD{0, 0, 0};
                for (int j = S.start[i]; j < S.start[i + 1]; ++j) {
                    const int bn = other[std::size_t(S.addr[std::size_t(j)])];
                    if (bn < 0) return std::unexpected("cyclicAMI: an addressed face is not a boundary face");
                    const double w = S.w[std::size_t(j)];
                    const int N = s.bOwn[std::size_t(bn)];
                    const int slot = la::graphSlot(g, P, N);
                    if (slot < 0) return std::unexpected("cyclicAMI: the matrix graph lacks a coupling (buildFvGraph)");
                    A.aCell.push_back(N);
                    A.aFace.push_back(bn);
                    A.aSlot.push_back(slot);
                    A.aEll.push_back(pat.slotPos[std::size_t(slot)]);
                    A.aW.push_back(w);
                    const V3 dn = dOf(bn);
                    nbrDelta += w * dot(nOf(bn), dn);
                    for (std::size_t q = 0; q < 3; ++q) nbrD[q] += w * dn[q];
                }
                A.aStart.push_back(int(A.aCell.size()));
                // cyclicAMIFvPatch::makeWeights
                const V3 n = nOf(b), dP = dOf(b);
                const double di = std::abs(dot(n, dP)), dni = std::abs(nbrDelta);
                A.wt.push_back(dni / (di + dni));
                // cyclicAMIFvPatch::delta(); basicFvGeometryScheme::nonOrthDeltaCoeffs / nonOrthCorrectionVectors
                const V3 delta{dP[0] - nbrD[0], dP[1] - nbrD[1], dP[2] - nbrD[2]};
                const double dcNO = 1.0 / std::max(dot(n, delta), 0.05 * mag(delta));
                s.bDc[std::size_t(b)] = dcNO;
                for (std::size_t q = 0; q < 3; ++q) {
                    A.d.push_back(delta[q]); // face-major here, transposed below
                    A.k.push_back(n[q] - delta[q] * dcNO);
                }
            }
            sum = {mn, mx, mine.empty() ? 0.0 : tot / double(mine.size())};
            return {};
        };
        if (auto r = side(p.ami.src, fs, ft, rep.srcSum, rep.srcFaces, rep.srcLow); !r) return r;
        if (auto r = side(p.ami.tgt, ft, fs, rep.tgtSum, rep.tgtFaces, rep.tgtLow); !r) return r;
        A.report.push_back(std::move(rep));
    }
    // d, k component-major
    const std::size_t nA = std::size_t(A.nA);
    std::vector<double> d(3 * nA), k(3 * nA);
    for (std::size_t a = 0; a < nA; ++a)
        for (std::size_t q = 0; q < 3; ++q) {
            d[q * nA + a] = A.d[3 * a + q];
            k[q * nA + a] = A.k[3 * a + q];
        }
    A.d = std::move(d);
    A.k = std::move(k);
    // rows: the cells with AMI faces, their faces in ascending boundary-face order
    std::vector<int> order(nA);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](int x, int y) {
        const int cx = s.bOwn[std::size_t(A.aB[std::size_t(x)])], cy = s.bOwn[std::size_t(A.aB[std::size_t(y)])];
        return cx != cy ? cx < cy : A.aB[std::size_t(x)] < A.aB[std::size_t(y)];
    });
    for (const int a : order) {
        const int cell = s.bOwn[std::size_t(A.aB[std::size_t(a)])];
        if (A.rCell.empty() || A.rCell.back() != cell) {
            A.rCell.push_back(cell);
            A.rStart.push_back(int(A.rA.size()));
        }
        A.rA.push_back(a);
    }
    A.rStart.push_back(int(A.rA.size()));
    return {};
}

} // namespace dns
