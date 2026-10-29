// Host half of the solver library (docs/la.md): block shapes, the sliced-ELL
// pattern, structured graphs (== graphOf(fromCurvMesh)), line sets (gridLines,
// strongLines, lineSet's rejections) and the line oracle's two restatements.
#include "check.hpp"
#include "la_test_util.hpp"

#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/grid.hpp"
#include "dns/la/ell.hpp"
#include "dns/la/graph.hpp"
#include "dns/la/lines.hpp"
#include "dns/la/structured.hpp"
#include "dns/polymesh.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dns;
using namespace dns::la;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

template <class E>
auto orDie(E&& e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    return std::move(*e);
}

bool contains(const std::string& s, const std::string& tok) { return s.find(tok) != std::string::npos; }

void shapes()
{
    std::printf("block shapes\n");
    for (const int B : supportedBlockSizes()) {
        const auto s = denseShape(B);
        check(s && s->B == B && s->Bd == B && s->nS == 0, "dense B = " + std::to_string(B));
    }
    const auto c11 = blockShape(11, 5, true), c11p = blockShape(11, 5, false), c6 = blockShape(6, 5, true);
    check(c11 && c11->nS == 1 && c11->shared(), "B 11 Bd 5 shared: nS 1");
    check(c11p && c11p->nS == 6 && !c11p->shared(), "B 11 Bd 5 per-row: nS 6");
    check(c6 && c6->nS == 1, "B 6 Bd 5: nS 1");
    const auto b4 = denseShape(4), d3 = blockShape(7, 3, true), d5 = blockShape(5, 5, true), d1 = blockShape(3, 5, true);
    check(!b4 && contains(b4.error(), "B = 4"), "B 4 rejected by name: " + (b4 ? std::string() : b4.error()));
    check(!d3 && contains(d3.error(), "Bd = 3"), "Bd 3 rejected by name");
    check(d5 && d5->nS == 0, "Bd == B ignores sharedScalar");
    check(!d1 && contains(d1.error(), "Bd = 5"), "Bd > B rejected");
}

CellGraph fromMesh(std::array<int, 3> n, std::array<bool, 3> per)
{
    const Grid g = Grid::make(n[0], n[1], n[2], n[0] * 0.25, n[1] * 0.5, n[2] * 0.125);
    StructuredSpec sp{{0, 0, 0}, {n[0] - 1, n[1] - 1, n[2] - 1}, {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"}, {}};
    for (int a = 0; a < 3; ++a) {
        const FoamPatchType t = per[std::size_t(a)] ? FoamPatchType::Cyclic : FoamPatchType::Wall;
        sp.type[std::size_t(2 * a)] = sp.type[std::size_t(2 * a + 1)] = t;
    }
    if (n[2] == 1) sp.type[4] = sp.type[5] = FoamPatchType::Empty;
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Source), "structured");
    return graphOf(fm);
}

void structured()
{
    std::printf("structuredGraph == graphOf(fromCurvMesh)\n");
    struct Case
    {
        std::array<int, 3> n;
        std::array<bool, 3> per;
        const char* tag;
    };
    for (const Case& cs : {Case{{6, 5, 4}, {true, true, true}, "6x5x4 periodic"},
                           Case{{5, 4, 3}, {false, false, false}, "5x4x3 walled"},
                           Case{{7, 5, 1}, {true, false, false}, "7x5 2D x-periodic"},
                           Case{{2, 5, 3}, {true, true, false}, "2x5x3 (2-wide periodic x)"},
                           Case{{3, 1, 1}, {false, false, false}, "3x1x1 (degenerate y)"}}) {
        if (cs.n[1] == 1) { // fromCurvMesh admits only z degenerate: a host-only check
            const CellGraph s = structuredGraph(cs.n, cs.per);
            check(s.rowPtr == std::vector<int>{0, 1, 3, 4} && s.col == std::vector<int>{1, 0, 2, 1},
                  std::string(cs.tag) + ": a 1D chain");
            continue;
        }
        const CellGraph a = structuredGraph(cs.n, cs.per), b = fromMesh(cs.n, cs.per);
        check(a.n == b.n && a.rowPtr == b.rowPtr && a.col == b.col, cs.tag);
    }
}

void ell()
{
    std::printf("ELL pattern\n");
    const PolyMesh pm = orDie(readPolyMesh(fs::path(DNS_MESH_DATA) / "cyl2d_snappy" / "constant/polyMesh"), "cyl2d");
    const FaceMesh fm = orDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), "cyl2d");
    for (const CellGraph& g : {graphOf(fm), structuredGraph({13, 9, 7}, {true, false, true})}) {
        const EllPattern p = ellPattern(g);
        bool ok = p.nSlices == (g.n + 63) / 64 && p.sliceBase.back() == p.nE;
        for (int c = 0; c < g.n && ok; ++c) {
            const int s = c / kSliceRows, base = p.sliceBase[std::size_t(s)] + c % kSliceRows;
            const int deg = g.rowPtr[std::size_t(c) + 1] - g.rowPtr[std::size_t(c)];
            for (int k = 0; k < p.sliceWidth[std::size_t(s)]; ++k) {
                const int E = base + kSliceRows * k;
                ok = ok && (k < deg ? p.col[std::size_t(E)] == g.col[std::size_t(g.rowPtr[std::size_t(c)] + k)] &&
                                          p.slotPos[std::size_t(g.rowPtr[std::size_t(c)] + k)] == E
                                    : p.col[std::size_t(E)] == -1);
            }
        }
        std::vector<int> seen(std::size_t(p.nE), 0);
        for (const int E : p.slotPos) ok = ok && ++seen[std::size_t(E)] == 1;
        check(ok, "rows " + std::to_string(g.n) + ": slots map to distinct entries, ascending, padded -1 (nE " +
                      std::to_string(p.nE) + ", max width " + std::to_string(p.maxWidth) + ")");
        check(ellPattern(g) == p, "deterministic");
    }
}

void lines()
{
    std::printf("line sets\n");
    const CellGraph g = structuredGraph({12, 9, 1}, {false, false, false});
    for (int axis = 0; axis < 2; ++axis) {
        const LineSet gl = gridLines({12, 9, 1}, axis);
        std::vector<std::vector<int>> ch;
        for (int l = 0; l < gl.count(); ++l)
            ch.emplace_back(gl.cells.begin() + gl.start[std::size_t(l)], gl.cells.begin() + gl.start[std::size_t(l) + 1]);
        const auto v = lineSet(g, ch);
        check(v && *v == gl, "gridLines axis " + std::to_string(axis) + " is a valid line set (" +
                                 std::to_string(gl.count()) + " lines)");
    }
    // rejections by name
    const std::vector<std::vector<int>> twice{{0, 1}, {1, 2}}, gap{{0, 2}}, out{{0, 999}}, empty{{}};
    const auto e1 = lineSet(g, twice), e2 = lineSet(g, gap), e3 = lineSet(g, out), e4 = lineSet(g, empty);
    check(!e1 && contains(e1.error(), "in two chains"), "a row in two chains rejected");
    check(!e2 && contains(e2.error(), "not adjacent"), "non-adjacent rows rejected");
    check(!e3 && contains(e3.error(), "out of range"), "row out of range rejected");
    check(!e4 && contains(e4.error(), "empty chain"), "empty chain rejected");
    const std::vector<std::vector<int>> one{{13, 14, 15}};
    const auto s = lineSet(g, one);
    check(s && s->count() == 1 + g.n - 3 && s->start[1] == 3 && s->cells[3] == 0, "uncovered rows become singletons");

    // strongLines: y couplings 100x x couplings recover the y grid lines.
    std::vector<double> w(g.col.size());
    for (int c = 0; c < g.n; ++c)
        for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k)
            w[std::size_t(k)] = std::abs(g.col[std::size_t(k)] - c) == 12 ? 100.0 : 1.0;
    const LineSet sl = strongLines(g, w, 4.0);
    check(sl == gridLines({12, 9, 1}, 1), "anisotropic weights: strongLines == gridLines(axis y)");
    check(strongLines(g, w, 4.0) == sl, "strongLines deterministic");
    std::vector<double> iso(g.col.size(), 1.0);
    check(strongLines(g, iso, 4.0).count() == g.n, "isotropic weights: all singletons");
    check(strongLines(g, w, 200.0).count() == g.n, "ratio above the anisotropy: all singletons");

    // A real mesh: |S|/|d| on cyl2d_snappy's layers.
    const PolyMesh pm = orDie(readPolyMesh(fs::path(DNS_MESH_DATA) / "cyl2d_snappy" / "constant/polyMesh"), "cyl2d");
    const FaceMesh fm = orDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), "cyl2d");
    const CellGraph gm = graphOf(fm);
    std::vector<double> wm(gm.col.size(), 0.0);
    for (int f = 0; f < fm.nI; ++f) {
        const std::size_t F = std::size_t(f);
        const double S = std::hypot(fm.S[0][F], fm.S[1][F], fm.S[2][F]), d = std::hypot(fm.d[0][F], fm.d[1][F], fm.d[2][F]);
        wm[std::size_t(gm.ownSlot[F])] += S / d;
        wm[std::size_t(gm.neiSlot[F])] += S / d;
    }
    const LineSet ml = strongLines(gm, wm, 4.0);
    int multi = 0;
    for (int l = 0; l < ml.count(); ++l) multi += ml.start[std::size_t(l) + 1] - ml.start[std::size_t(l)] > 1;
    std::printf("  cyl2d_snappy |S|/|d| ratio 4: %d lines of length > 1, max length %d\n", multi, ml.maxLength());
    check(multi > 0 && ml.cells.size() == std::size_t(gm.n), "cyl2d_snappy: strong lines found, every row once");
}

void oracle()
{
    std::printf("line oracle: blockband == dense\n");
    const CellGraph g = structuredGraph({6, 17, 1}, {false, false, false});
    const LineSet ls = gridLines({6, 17, 1}, 1);
    for (const int B : {1, 3, 5, 7}) {
        const BlockMatrix m = latest::randomMatrix(g, *denseShape(B), 11u + unsigned(B));
        const std::vector<double> r = latest::randomVec(std::size_t(g.n * B), 3u);
        std::vector<double> za, zb, sa, sb;
        latest::lineSolve(m, ls, r, za, sa, true);
        latest::lineSolve(m, ls, r, zb, sb, false);
        const double e = latest::maxRel(za, zb, sa);
        check(e <= 1e-13, "B " + std::to_string(B) + ": band vs dense rel " + std::to_string(e));
    }
}

} // namespace

int main()
{
    shapes();
    structured();
    ell();
    lines();
    oracle();
    return dnstest::finish();
}
