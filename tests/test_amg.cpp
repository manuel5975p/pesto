// Host half of the AMG (docs/la.md, masterplan_meshsolver P5): the pairwise
// aggregation's invariants on structured and polyhedral graphs (every row in
// one connected aggregate of <= 2^passes rows, coarse graphs neighbour-unique
// and symmetric, every fine slot summed exactly once), the Galerkin
// restatement vs an explicit P^T A P, the semi-coarsening along a strong axis,
// option / weight rejections by name, determinism; `setup`: the Setup gate
// (6M-cell 7-point graph <= 3 s host, two setups identical).
#include "check.hpp"
#include "la_test_util.hpp"

#include "dns/facemesh.hpp"
#include "dns/la/amg.hpp"
#include "dns/la/graph.hpp"
#include "dns/la/structured.hpp"
#include "dns/polymesh.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
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

FaceMesh loadMesh(const char* name, CellOrder order)
{
    const PolyMesh pm = orDie(readPolyMesh(fs::path(DNS_MESH_DATA) / name / "constant/polyMesh"), name);
    return orDie(FaceMesh::fromPolyMesh(pm, order), name);
}

// Neighbour-unique ascending rows, no self entry, symmetric pattern.
bool wellFormed(const CellGraph& g)
{
    for (int c = 0; c < g.n; ++c)
        for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) {
            const int j = g.col[std::size_t(k)];
            if (j < 0 || j >= g.n || j == c) return false;
            if (k > g.rowPtr[std::size_t(c)] && g.col[std::size_t(k) - 1] >= j) return false;
            const auto b = g.col.begin() + g.rowPtr[std::size_t(j)], e = g.col.begin() + g.rowPtr[std::size_t(j) + 1];
            if (!std::binary_search(b, e, c)) return false;
        }
    return true;
}

// Every aggregate connected in the fine graph (BFS inside the member set).
bool aggregatesConnected(const CellGraph& g, const AmgTransfer& t)
{
    std::vector<int> mark(std::size_t(g.n), -1);
    for (int I = 0; I + 1 < int(t.memberPtr.size()); ++I) {
        const int b = t.memberPtr[std::size_t(I)], e = t.memberPtr[std::size_t(I) + 1];
        if (b == e) return false;
        std::vector<int> q{t.member[std::size_t(b)]};
        mark[std::size_t(q[0])] = I;
        for (std::size_t h = 0; h < q.size(); ++h)
            for (int k = g.rowPtr[std::size_t(q[h])]; k < g.rowPtr[std::size_t(q[h]) + 1]; ++k) {
                const int j = g.col[std::size_t(k)];
                if (t.agg[std::size_t(j)] == I && mark[std::size_t(j)] != I) {
                    mark[std::size_t(j)] = I;
                    q.push_back(j);
                }
            }
        if (int(q.size()) != e - b) return false;
    }
    return true;
}

// Transfer consistency: members == agg, intra slots inside, off slots join
// the coarse slot's two aggregates, every fine slot exactly once.
bool transferConsistent(const CellGraph& g, const CellGraph& gc, const AmgTransfer& t, int maxSize)
{
    if (int(t.agg.size()) != g.n || int(t.memberPtr.size()) != gc.n + 1) return false;
    std::vector<int> used(g.col.size(), 0);
    auto rowOf = [&](int slot) {
        return int(std::upper_bound(g.rowPtr.begin(), g.rowPtr.end(), slot) - g.rowPtr.begin()) - 1;
    };
    for (int I = 0; I < gc.n; ++I) {
        const int sz = t.memberPtr[std::size_t(I) + 1] - t.memberPtr[std::size_t(I)];
        if (sz < 1 || sz > maxSize) return false;
        for (int k = t.memberPtr[std::size_t(I)]; k < t.memberPtr[std::size_t(I) + 1]; ++k) {
            if (t.agg[std::size_t(t.member[std::size_t(k)])] != I) return false;
            if (k > t.memberPtr[std::size_t(I)] && t.member[std::size_t(k) - 1] >= t.member[std::size_t(k)]) return false;
        }
        for (int k = t.intraPtr[std::size_t(I)]; k < t.intraPtr[std::size_t(I) + 1]; ++k) {
            const int s = t.intraSlot[std::size_t(k)];
            if (t.agg[std::size_t(rowOf(s))] != I || t.agg[std::size_t(g.col[std::size_t(s)])] != I) return false;
            ++used[std::size_t(s)];
        }
        for (int cs = gc.rowPtr[std::size_t(I)]; cs < gc.rowPtr[std::size_t(I) + 1]; ++cs) {
            const int J = gc.col[std::size_t(cs)];
            for (int k = t.offPtr[std::size_t(cs)]; k < t.offPtr[std::size_t(cs) + 1]; ++k) {
                const int s = t.offSlot[std::size_t(k)];
                if (t.agg[std::size_t(rowOf(s))] != I || t.agg[std::size_t(g.col[std::size_t(s)])] != J) return false;
                if (k > t.offPtr[std::size_t(cs)] && t.offSlot[std::size_t(k) - 1] >= s) return false;
                ++used[std::size_t(s)];
            }
        }
    }
    return std::ranges::all_of(used, [](int u) { return u == 1; });
}

// Explicit P^T A P (dense over coarse rows, logical blocks) vs amgGalerkin.
double galerkinError(const BlockMatrix& m, const CellGraph& gc, const AmgTransfer& t)
{
    const BlockMatrix c = amgGalerkin(m, gc, t);
    const int B = m.shape.B, nc = gc.n;
    std::map<std::pair<int, int>, std::vector<double>> ref;
    auto blk = [&](int I, int J) -> std::vector<double>& {
        auto& v = ref[{I, J}];
        if (v.empty()) v.assign(std::size_t(B * B), 0.0);
        return v;
    };
    for (int i = 0; i < m.n; ++i) {
        auto& d = blk(t.agg[std::size_t(i)], t.agg[std::size_t(i)]);
        for (int r = 0; r < B; ++r)
            for (int q = 0; q < B; ++q) d[std::size_t(r * B + q)] += m.diagAt(i, r, q);
        for (int k = m.rowPtr[std::size_t(i)]; k < m.rowPtr[std::size_t(i) + 1]; ++k) {
            auto& o = blk(t.agg[std::size_t(i)], t.agg[std::size_t(m.col[std::size_t(k)])]);
            for (int r = 0; r < B; ++r)
                for (int q = 0; q < B; ++q) o[std::size_t(r * B + q)] += m.offAt(k, r, q);
        }
    }
    double err = 0, scale = 0;
    for (int I = 0; I < nc; ++I) {
        const auto& d = ref[{I, I}];
        for (int r = 0; r < B; ++r)
            for (int q = 0; q < B; ++q) {
                err = std::max(err, std::abs(d[std::size_t(r * B + q)] - c.diagAt(I, r, q)));
                scale = std::max(scale, std::abs(d[std::size_t(r * B + q)]));
            }
        for (int k = gc.rowPtr[std::size_t(I)]; k < gc.rowPtr[std::size_t(I) + 1]; ++k) {
            const auto& o = ref[{I, gc.col[std::size_t(k)]}];
            for (int r = 0; r < B; ++r)
                for (int q = 0; q < B; ++q) err = std::max(err, std::abs(o[std::size_t(r * B + q)] - c.offAt(k, r, q)));
            ref.erase({I, gc.col[std::size_t(k)]});
        }
        ref.erase({I, I});
    }
    // every reference block has a home in the coarse pattern
    for (const auto& [key, v] : ref)
        for (const double x : v) err = std::max(err, std::abs(x));
    return err / std::max(scale, 1e-300);
}

struct Case
{
    std::string tag;
    CellGraph g;
};

std::vector<Case> graphs()
{
    std::vector<Case> v;
    v.push_back({"2D 5-point 37x23", structuredGraph({37, 23, 1}, {false, false, false})});
    v.push_back({"3D 7-point 13x9x7", structuredGraph({13, 9, 7}, {false, false, false})});
    v.push_back({"3D cyclic 8x6x5", structuredGraph({8, 6, 5}, {true, true, true})});
    v.push_back({"2D 64x64", structuredGraph({64, 64, 1}, {false, false, false})});
    v.push_back({"cyl2d_snappy Morton", graphOf(loadMesh("cyl2d_snappy", CellOrder::Morton))});
    v.push_back({"cyl2d_snappy Source", graphOf(loadMesh("cyl2d_snappy", CellOrder::Source))});
    v.push_back({"box_refined Morton", graphOf(loadMesh("box_refined", CellOrder::Morton))});
    return v;
}

void hierarchies()
{
    std::printf("aggregation invariants\n");
    for (const auto& [tag, g] : graphs()) {
        const BlockMatrix m = latest::randomMatrix(g, *denseShape(1), 3u);
        for (const int passes : {1, 2}) {
            AmgSetup s;
            s.passes = passes;
            s.coarseUnknowns = 16;
            s.lineRatio = 4;
            const auto w = std::vector<double>(g.col.size(), 1.0);
            const AmgHierarchy h = orDie(amgHierarchy(g, w, 1, s), tag);
            const AmgHierarchy h2 = orDie(amgHierarchy(g, w, 1, s), tag);
            check(h == h2, tag + " passes " + std::to_string(passes) + ": two setups identical");
            bool ok = true;
            for (int l = 0; l + 1 < h.levels(); ++l) {
                const CellGraph& gf = l == 0 ? g : h.graphs[std::size_t(l - 1)];
                const CellGraph& gc = h.graphs[std::size_t(l)];
                ok = ok && wellFormed(gc) && aggregatesConnected(gf, h.transfers[std::size_t(l)]) &&
                     transferConsistent(gf, gc, h.transfers[std::size_t(l)], 1 << passes);
                ok = ok && int(h.lines[std::size_t(l)].cells.size()) == gf.n;
            }
            std::string lv;
            for (const int r : h.rows) lv += " " + std::to_string(r);
            std::printf("  %-22s passes %d: levels%s  grid %.3f op %.3f\n", tag.c_str(), passes, lv.c_str(),
                        h.gridComplexity(), h.operatorComplexity());
            check(ok, tag + " passes " + std::to_string(passes) + ": aggregates connected, <= " +
                          std::to_string(1 << passes) + " rows, coarse graphs well formed, every slot once");
            check(h.rows.back() <= 16 || h.levels() == s.maxLevels, tag + ": reaches the coarse size");
            // Galerkin vs explicit P^T A P on level 0 (scalar and a split block shape)
            const double e1 = galerkinError(m, h.graphs[0], h.transfers[0]);
            const BlockMatrix m7 = latest::randomMatrix(g, *blockShape(7, 5, true), 9u);
            const double e7 = galerkinError(m7, h.graphs[0], h.transfers[0]);
            check(e1 <= 1e-14 && e7 <= 1e-14, tag + ": amgGalerkin == explicit P^T A P (B1 " + std::to_string(e1) +
                                                  ", B7/5 " + std::to_string(e7) + ")");
        }
    }
}

void structure()
{
    std::printf("aggregate shapes\n");
    // isotropic 64 x 64: every level-1 aggregate has 4 rows (2x2 counted)
    const std::array<int, 3> n{64, 64, 1};
    const CellGraph g = structuredGraph(n, {false, false, false});
    AmgSetup s;
    s.coarseUnknowns = 64;
    const AmgHierarchy h = orDie(amgHierarchy(g, std::vector<double>(g.col.size(), 1.0), 1, s), "iso");
    int four = 0, squares = 0;
    for (int I = 0; I < h.rows[1]; ++I) {
        const int b = h.transfers[0].memberPtr[std::size_t(I)], e = h.transfers[0].memberPtr[std::size_t(I) + 1];
        if (e - b != 4) continue;
        ++four;
        int imin = 1 << 30, imax = -1, jmin = 1 << 30, jmax = -1;
        for (int k = b; k < e; ++k) {
            const int c = h.transfers[0].member[std::size_t(k)];
            imin = std::min(imin, c % n[0]);
            imax = std::max(imax, c % n[0]);
            jmin = std::min(jmin, c / n[0]);
            jmax = std::max(jmax, c / n[0]);
        }
        squares += imax - imin == 1 && jmax - jmin == 1;
    }
    std::printf("  64x64 isotropic: %d of %d level-1 aggregates have 4 rows, %d of them 2x2\n", four, h.rows[1], squares);
    check(h.rows[1] == 64 * 64 / 4 && four == h.rows[1], "64x64 isotropic: every level-1 aggregate has 4 rows");
    // y 100x stronger: every level-1 aggregate is a 1 x 4 column in y
    const BlockMatrix an = [&] {
        BlockMatrix m = BlockMatrix::zeros(g, *denseShape(1));
        for (int c = 0; c < g.n; ++c)
            for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k)
                m.off[std::size_t(k)] = std::abs(g.col[std::size_t(k)] - c) == n[0] ? -100.0 : -1.0;
        return m;
    }();
    const AmgHierarchy ha = orDie(amgHierarchy(g, amgStrength(an), 1, s), "aniso");
    bool column = true;
    for (int I = 0; I < ha.rows[1]; ++I) {
        const int b = ha.transfers[0].memberPtr[std::size_t(I)], e = ha.transfers[0].memberPtr[std::size_t(I) + 1];
        for (int k = b; k < e; ++k) column = column && ha.transfers[0].member[std::size_t(k)] % n[0] ==
                                                           ha.transfers[0].member[std::size_t(b)] % n[0];
    }
    std::printf("  64x64 y-strong x100: level-1 rows %d, all aggregates inside one x column: %d\n", ha.rows[1], int(column));
    check(ha.rows[1] == 64 * 16 && column, "y-strong: semi-coarsening along y (1 x 4 aggregates)");
    // a constant-annihilating (zero row sum) Laplacian stays so on every level
    BlockMatrix lap = BlockMatrix::zeros(g, *denseShape(1));
    for (int c = 0; c < g.n; ++c)
        for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) {
            lap.off[std::size_t(k)] = -1.0;
            lap.diag[std::size_t(c)] += 1.0;
        }
    double worst = 0;
    BlockMatrix cur = lap;
    for (int l = 0; l + 1 < h.levels(); ++l) {
        cur = amgGalerkin(cur, h.graphs[std::size_t(l)], h.transfers[std::size_t(l)]);
        for (int c = 0; c < cur.n; ++c) {
            double s = cur.diag[std::size_t(c)];
            for (int k = cur.rowPtr[std::size_t(c)]; k < cur.rowPtr[std::size_t(c) + 1]; ++k) s += cur.off[std::size_t(k)];
            worst = std::max(worst, std::abs(s));
        }
    }
    check(worst == 0.0, "zero row sums preserved exactly on every level (" + std::to_string(worst) + ")");
}

void rejections()
{
    std::printf("rejections\n");
    const CellGraph g = structuredGraph({8, 8, 1}, {false, false, false});
    const std::vector<double> w(g.col.size(), 1.0);
    const auto a = amgHierarchy(g, std::vector<double>(3, 1.0), 1, {});
    check(!a && contains(a.error(), "3 slot weights"), "weight count: " + (a ? std::string() : a.error()));
    std::vector<double> neg = w;
    neg[5] = -1;
    const auto b = amgHierarchy(g, neg, 1, {});
    check(!b && contains(b.error(), "slot weight 5"), "negative weight: " + (b ? std::string() : b.error()));
    neg[5] = std::nan("");
    check(!amgHierarchy(g, neg, 1, {}), "NaN weight rejected");
    AmgSetup s;
    s.coarseUnknowns = 5000;
    const auto c = amgHierarchy(g, w, 1, s);
    check(!c && contains(c.error(), "coarseUnknowns"), "coarseUnknowns > 1024: " + (c ? std::string() : c.error()));
    s = {};
    s.lineRatio = 0.5;
    check(!amgHierarchy(g, w, 1, s), "lineRatio 0.5 rejected");
    s = {};
    s.passes = 0;
    check(!amgHierarchy(g, w, 1, s), "passes 0 rejected");
    // 2000 rows without couplings: no aggregate can form
    CellGraph iso;
    iso.n = 2000;
    iso.rowPtr.assign(2001, 0);
    const auto d = amgHierarchy(iso, {}, 1, {});
    check(!d && contains(d.error(), "stalled"), "uncoupled rows: " + (d ? std::string() : d.error()));
    // all-zero weights on a coupled graph: same stall
    const CellGraph g2 = structuredGraph({40, 40, 1}, {false, false, false});
    const auto e = amgHierarchy(g2, std::vector<double>(g2.col.size(), 0.0), 1, {});
    check(!e && contains(e.error(), "stalled"), "zero weights stall by name");
    // small enough: one level, no transfer
    const auto f = amgHierarchy(structuredGraph({10, 10, 1}, {false, false, false}),
                                std::vector<double>(structuredGraph({10, 10, 1}, {false, false, false}).col.size(), 1.0), 1, {});
    check(f && f->levels() == 1, "100 rows <= 256: one level");
}

int setupGate()
{
    std::printf("Setup gate: host hierarchy of a 6M-cell 7-point graph\n");
    const std::array<int, 3> n{182, 182, 182};
    const CellGraph g = structuredGraph(n, {false, false, false});
    const std::vector<double> w(g.col.size(), 1.0);
    std::printf("  graph %d rows, %zu slots\n", g.n, g.col.size());
    for (const double lr : {0.0, 4.0}) {
        AmgSetup s;
        s.lineRatio = lr;
        double best = 1e30;
        AmgHierarchy first;
        bool same = true;
        for (int rep = 0; rep < 2; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            AmgHierarchy h = orDie(amgHierarchy(g, w, 1, s), "6M");
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, sec);
            std::printf("  lineRatio %.0f run %d: %.3f s, levels %d (", lr, rep, sec, h.levels());
            for (const int r : h.rows) std::printf(" %d", r);
            std::printf(" ), grid %.3f op %.3f\n", h.gridComplexity(), h.operatorComplexity());
            if (rep == 0) first = std::move(h);
            else same = h == first;
        }
        check(same, "6M setup (lineRatio " + std::to_string(int(lr)) + "): two setups identical");
        check(best <= 3.0, "6M setup (lineRatio " + std::to_string(int(lr)) + ") <= 3 s (" + std::to_string(best) + " s)");
    }
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "all";
    if (mode == "setup") return setupGate();
    hierarchies();
    structure();
    rejections();
    return dnstest::finish();
}
