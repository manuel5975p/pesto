// FaceMesh and CellGraph (dns/facemesh, dns/la/graph; masterplan_meshsolver P1):
//   F1  geometry vs OpenFOAM's own checkMesh log (parsed, not transcribed)
//   F2  per-cell closure, total volume, d.S > 0
//   F3  fromCurvMesh: exact uniform periodic box, 2D O-grid vs CurvMetrics,
//       sinusoidal periodic mesh vs CurvMetrics
//   F4  Source / Morton / ColourMajor: identical geometry under srcCell/srcFace;
//       valid colouring, 2 colours on an even periodic box
//   F5  threads 1/2/3/7/hw byte-identical      F6  two builds byte-identical
//   F7  CellGraph symmetric, slots consistent, no self entries
//   F8  (`test_facemesh mb`, local, rc 77 without runs/foam_mesh/mB) the
//       831k-cell snappy mesh vs its checkMesh numbers, build time
#include "dns/facemesh.hpp"

#include "dns/bodies.hpp"
#include "dns/curvmesh.hpp"
#include "dns/la/graph.hpp"
#include "dns/ogrid.hpp"
#include "check.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <thread>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

fs::path dataDir(const char* name) { return fs::path(DNS_MESH_DATA) / name; }

PolyMesh readOrDie(const fs::path& d)
{
    auto m = readPolyMesh(d);
    if (!m) {
        std::printf("FATAL: %s\n", m.error().c_str());
        std::exit(1);
    }
    return std::move(*m);
}

FaceMesh buildOrDie(std::expected<FaceMesh, std::string> m, const std::string& what)
{
    if (!m) {
        std::printf("FATAL: %s: %s\n", what.c_str(), m.error().c_str());
        std::exit(1);
    }
    return std::move(*m);
}

double rel(double a, double b) { return std::abs(a - b) / std::max(std::abs(b), 1e-300); }

// --- checkMesh log ------------------------------------------------------------

struct CheckMesh
{
    double vMin = -1, vMax = -1, vTotal = -1, nonOrthMax = -1, nonOrthAvg = -1, skew = -1, aspect = -1;
    long cells = -1;
};

CheckMesh parseCheckMesh(const fs::path& log)
{
    std::ifstream f(log);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    CheckMesh c;
    const std::string num = "([-+0-9.eE]+)";
    std::smatch m;
    if (std::regex_search(s, m, std::regex("Min volume = " + num + "\\. Max volume = " + num
                                           + "\\.\\s+Total volume = " + num + "\\.")))
        c.vMin = std::stod(m[1]), c.vMax = std::stod(m[2]), c.vTotal = std::stod(m[3]);
    if (std::regex_search(s, m, std::regex("Mesh non-orthogonality Max: " + num + " average: " + num)))
        c.nonOrthMax = std::stod(m[1]), c.nonOrthAvg = std::stod(m[2]);
    if (std::regex_search(s, m, std::regex("Max skewness = " + num))) c.skew = std::stod(m[1]);
    if (std::regex_search(s, m, std::regex("Max aspect ratio = " + num))) c.aspect = std::stod(m[1]);
    if (std::regex_search(s, m, std::regex("\\n\\s+cells:\\s+([0-9]+)"))) c.cells = std::stol(m[1]);
    return c;
}

void compareCheckMesh(const MeshQuality& q, const FaceMesh& fm, const CheckMesh& c, const std::string& tag)
{
    check(c.vTotal > 0 && c.nonOrthMax >= 0 && c.skew >= 0 && c.aspect > 0 && c.cells > 0,
          tag + ": checkMesh log parsed");
    check(long(fm.nC) == c.cells, std::format("{}: {} cells == checkMesh {}", tag, fm.nC, c.cells));
    check(rel(q.vTotal, c.vTotal) <= 1e-9 && rel(q.vMin, c.vMin) <= 1e-9 && rel(q.vMax, c.vMax) <= 1e-9,
          std::format("{}: volume total/min/max {:.15g}/{:.15g}/{:.15g} vs {:.15g}/{:.15g}/{:.15g} (1e-9)", tag,
                      q.vTotal, q.vMin, q.vMax, c.vTotal, c.vMin, c.vMax));
    check(rel(q.nonOrthMax, c.nonOrthMax) <= 1e-6 && rel(q.nonOrthAvg, c.nonOrthAvg) <= 1e-6,
          std::format("{}: non-orthogonality max/avg {:.12g}/{:.12g} vs {:.12g}/{:.12g} (1e-6)", tag,
                      q.nonOrthMax, q.nonOrthAvg, c.nonOrthMax, c.nonOrthAvg));
    check(rel(q.skewMax, c.skew) <= 1e-6,
          std::format("{}: skewness max {:.12g} vs {:.12g} (1e-6)", tag, q.skewMax, c.skew));
    check(rel(q.aspectMax, c.aspect) <= 1e-6,
          std::format("{}: aspect max {:.12g} vs {:.12g} (1e-6)", tag, q.aspectMax, c.aspect));
}

// --- generic checks -------------------------------------------------------------

struct Closure
{
    double worst = 0;   // max |sum S| / sum |S|
    double minDS = 1e300;
};

Closure closure(const FaceMesh& m)
{
    Closure r;
    for (int c = 0; c < m.nC; ++c) {
        double s[3] = {0, 0, 0}, sm = 0;
        for (int k = m.cellStart[std::size_t(c)]; k < m.cellStart[std::size_t(c) + 1]; ++k) {
            const std::size_t f = std::size_t(m.cellFace[std::size_t(k)]);
            const double sg = m.own[f] == c ? 1.0 : -1.0;
            for (int a = 0; a < 3; ++a) s[a] += sg * m.S[std::size_t(a)][f];
            sm += std::hypot(m.S[0][f], m.S[1][f], m.S[2][f]);
        }
        r.worst = std::max(r.worst, std::hypot(s[0], s[1], s[2]) / sm);
    }
    for (int f = 0; f < m.nI; ++f) {
        const std::size_t u = std::size_t(f);
        r.minDS = std::min(r.minDS, m.d[0][u] * m.S[0][u] + m.d[1][u] * m.S[1][u] + m.d[2][u] * m.S[2][u]);
    }
    return r;
}

template <class T>
bool bytes(const std::vector<T>& a, const std::vector<T>& b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool byteIdentical(const FaceMesh& a, const FaceMesh& b)
{
    bool ok = a.nC == b.nC && a.nI == b.nI && a.nB == b.nB && a.dim == b.dim && a.emptyAxis == b.emptyAxis
              && bytes(a.own, b.own) && bytes(a.nei, b.nei) && bytes(a.V, b.V) && bytes(a.cellStart, b.cellStart)
              && bytes(a.cellFace, b.cellFace) && a.patches == b.patches && a.zoneNames == b.zoneNames
              && a.zoneCells == b.zoneCells && bytes(a.srcCell, b.srcCell) && bytes(a.srcFace, b.srcFace)
              && bytes(a.colour, b.colour) && a.nColours == b.nColours && a.order == b.order;
    for (std::size_t q = 0; q < 3; ++q)
        ok = ok && bytes(a.S[q], b.S[q]) && bytes(a.Cf[q], b.Cf[q]) && bytes(a.d[q], b.d[q]) && bytes(a.C[q], b.C[q]);
    return ok;
}

// F7
void graphChecks(const FaceMesh& m, const std::string& tag, int wantDegree = -1)
{
    const la::CellGraph g = la::graphOf(m);
    bool sym = true, sorted = true, noSelf = true, slots = true, degree = true;
    for (int r = 0; r < g.n; ++r) {
        const int b = g.rowPtr[std::size_t(r)], e = g.rowPtr[std::size_t(r) + 1];
        if (wantDegree >= 0 && e - b != wantDegree) degree = false;
        for (int q = b; q < e; ++q) {
            const int c = g.col[std::size_t(q)];
            if (c == r) noSelf = false;
            if (q > b && g.col[std::size_t(q) - 1] >= c) sorted = false;
            const auto cb = g.col.begin() + g.rowPtr[std::size_t(c)], ce = g.col.begin() + g.rowPtr[std::size_t(c) + 1];
            if (!std::binary_search(cb, ce, r)) sym = false;
        }
    }
    for (int f = 0; f < m.nI; ++f) {
        const int o = m.own[std::size_t(f)], n = m.nei[std::size_t(f)];
        const int os = g.ownSlot[std::size_t(f)], ns = g.neiSlot[std::size_t(f)];
        if (g.col[std::size_t(os)] != n || g.col[std::size_t(ns)] != o || os < g.rowPtr[std::size_t(o)]
            || os >= g.rowPtr[std::size_t(o) + 1] || ns < g.rowPtr[std::size_t(n)] || ns >= g.rowPtr[std::size_t(n) + 1])
            slots = false;
    }
    check(sym && sorted && noSelf, tag + ": CellGraph symmetric, ascending, no self entry");
    check(slots, tag + ": ownSlot/neiSlot index nei in row own and own in row nei");
    if (wantDegree >= 0) check(degree, std::format("{}: every row has {} entries", tag, wantDegree));
}

// --- F1 / F2 -----------------------------------------------------------------

void f1f2()
{
    std::printf("F1/F2 checked-in meshes vs checkMesh\n");
    for (const char* name : {"cyl2d_snappy", "box_refined", "box_cyclic", "box_cyclic_bin"}) {
        const PolyMesh pm = readOrDie(dataDir(name) / "constant" / "polyMesh");
        const FaceMesh fm = buildOrDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), name);
        const MeshQuality q = meshQuality(fm, pm);
        const CheckMesh c = parseCheckMesh(dataDir(name) / "log.checkMesh");
        const std::string n(name);
        if (n == "cyl2d_snappy" || n == "box_refined") {
            compareCheckMesh(q, fm, c, n);
            int face5 = 0, cell7 = 0;
            for (std::size_t k = 5; k < 32; ++k) face5 += q.nodesPerFace[k];
            for (std::size_t k = 7; k < 32; ++k) cell7 += q.facesPerCell[k];
            check(face5 > 0 && (n == "box_refined" || cell7 > 0),
                  std::format("{}: data exercises polyhedra ({} faces >= 5 nodes, {} cells >= 7 faces)", n,
                              face5, cell7));
        }
        const Closure cl = closure(fm);
        check(cl.worst <= 1e-12, std::format("{}: per-cell |sum S| / sum|S| = {:.2e} <= 1e-12", n, cl.worst));
        check(rel(q.vTotal, c.vTotal) <= 1e-9, std::format("{}: sum V == checkMesh total", n));
        check(cl.minDS > 0, std::format("{}: d.S > 0 on every internal face (min {:.3e})", n, cl.minDS));
        if (n == "cyl2d_snappy") {
            bool zero = true;
            for (const auto& p : fm.patches)
                if (p.type != FoamPatchType::Empty)
                    for (int f = p.first; f < p.first + p.count; ++f)
                        zero = zero && fm.S[2][std::size_t(fm.nI + f)] == 0.0 && fm.d[2][std::size_t(fm.nI + f)] == 0.0;
            for (int f = 0; f < fm.nI; ++f) zero = zero && fm.S[2][std::size_t(f)] == 0.0 && fm.d[2][std::size_t(f)] == 0.0;
            check(fm.dim == 2 && fm.emptyAxis == 2 && zero,
                  "cyl2d_snappy: 2D, empty axis z, S_z and d_z exactly 0 off the empty faces");
        }
        if (n == "box_cyclic") {
            check(fm.nI == 46 + 6 + 12 && fm.nB == 16, "box_cyclic: cyclic pairs merged (64 internal, 16 wall)");
            graphChecks(fm, "box_cyclic");
        }
        if (n == "cyl2d_snappy" || n == "box_refined") graphChecks(fm, n);
    }
}

// --- F3 ---------------------------------------------------------------------

std::array<int, 3> ijk(const Grid& g, std::int64_t idx)
{
    return {int(idx % g.nx), int((idx / g.nx) % g.ny), int(idx / (std::int64_t(g.nx) * g.ny))};
}

// Max |S_fm - sign S_curv| / |S| over every face of fm (built with srcCell =
// Grid::idx), the structured face identified by its two cells.
double faceAreaError(const FaceMesh& fm, const CurvMetrics& cm, const StructuredSpec& sp)
{
    const Grid& g = cm.g;
    const int n[3] = {g.nx, g.ny, g.nz};
    double worst = 0;
    auto cmp = [&](std::size_t f, std::size_t cell, int a, double sign) {
        double e = 0, m = 0;
        for (std::size_t q = 0; q < 3; ++q) {
            const double cs = sign * cm.S[std::size_t(a)][q][cell];
            e = std::max(e, std::abs(fm.S[q][f] - cs));
            m = std::max(m, std::abs(cs));
        }
        worst = std::max(worst, e / m);
    };
    for (int f = 0; f < fm.nI; ++f) {
        const auto A = ijk(g, fm.srcCell[std::size_t(fm.own[std::size_t(f)])]);
        const auto B = ijk(g, fm.srcCell[std::size_t(fm.nei[std::size_t(f)])]);
        int a = 0;
        while (A[std::size_t(a)] == B[std::size_t(a)]) ++a;
        // L: the cell whose +a face this is (B == L + 1 mod n).
        const bool ownIsL = (A[std::size_t(a)] + 1) % n[a] == B[std::size_t(a)];
        const auto L = ownIsL ? A : B;
        cmp(std::size_t(f), g.idx(L[0], L[1], L[2]), a, ownIsL ? 1.0 : -1.0);
    }
    for (const auto& p : fm.patches) {
        const int side = int(std::ranges::find(sp.name, p.name) - sp.name.begin());
        const int a = side / 2;
        for (int q = p.first; q < p.first + p.count; ++q) {
            const std::size_t f = std::size_t(fm.nI + q);
            auto c = ijk(g, fm.srcCell[std::size_t(fm.own[f])]);
            if (side % 2 == 1)
                cmp(f, g.idx(c[0], c[1], c[2]), a, 1.0);
            else {
                c[std::size_t(a)] -= 1;
                cmp(f, g.idx(c[0], c[1], c[2]), a, -1.0);
            }
        }
    }
    return worst;
}

void f3()
{
    std::printf("F3 fromCurvMesh\n");
    {   // (a) uniform periodic box, power-of-two spacings
        const Grid g = Grid::make(6, 5, 4, 6 * 0.25, 5 * 0.5, 4 * 0.125);
        const StructuredSpec sp{{0, 0, 0}, {5, 4, 3}, {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"},
                                {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                                 FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic}};
        const FaceMesh fm = buildOrDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Source), "box6x5x4");
        const double h[3] = {g.hx, g.hy, g.hz};
        bool sExact = true, sixFaces = true;
        double vErr = 0, dErr = 0;
        for (int f = 0; f < fm.nI; ++f) {
            int a = 0, nz = 0;
            for (int q = 0; q < 3; ++q)
                if (fm.S[std::size_t(q)][std::size_t(f)] != 0.0) a = q, ++nz;
            const double area = h[(a + 1) % 3] * h[(a + 2) % 3];
            if (nz != 1 || std::abs(fm.S[std::size_t(a)][std::size_t(f)]) != area) sExact = false;
            const double dd = std::hypot(fm.d[0][std::size_t(f)], fm.d[1][std::size_t(f)], fm.d[2][std::size_t(f)]);
            dErr = std::max(dErr, std::abs(dd - h[a]) / h[a]
                                      + std::abs(std::abs(fm.d[std::size_t(a)][std::size_t(f)]) - dd) / h[a]);
        }
        for (int c = 0; c < fm.nC; ++c) {
            vErr = std::max(vErr, rel(fm.V[std::size_t(c)], g.cellVolume()));
            if (fm.cellStart[std::size_t(c) + 1] - fm.cellStart[std::size_t(c)] != 6) sixFaces = false;
        }
        check(fm.nC == 120 && fm.nB == 0 && fm.nI == 360, "(a) 6x5x4 periodic: 120 cells, 360 internal faces, no boundary");
        check(sExact, "(a) S = the exact face areas, axis-aligned (bitwise)");
        check(vErr <= 4.5e-16, std::format("(a) V = hx hy hz to {:.2e} (<= 2 ulp)", vErr));
        check(dErr <= 1e-15, std::format("(a) d = h e_a on every face, cyclic included ({:.2e})", dErr));
        check(sixFaces, "(a) 6 faces per cell");
        graphChecks(fm, "(a)", 6);
    }
    {   // (b) 2D O-grid 64 x 24 vs CurvMetrics
        const Grid g = ogridGrid(64, 24, 1, 0.5, 5.0, 1.0);
        const auto og = buildOGrid(g, sdfOf(CylinderZ{0.0, 0.0, 0.5}),
                                   OGridSpec{.cx = 0, .cy = 0, .cz = 0, .rFar = 5.0, .beta = 2.0, .span = 1.0});
        if (!og) {
            check(false, "(b) O-grid builds: " + og.error());
            return;
        }
        const auto cm = CurvMetrics::build(og->mesh);
        if (!cm) {
            check(false, "(b) O-grid metrics build: " + cm.error());
            return;
        }
        const StructuredSpec sp{{0, 1, 0}, {63, 22, 0}, {"", "", "wall", "far", "back", "front"},
                                {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall,
                                 FoamPatchType::Patch, FoamPatchType::Empty, FoamPatchType::Empty}};
        const FaceMesh fm = buildOrDie(FaceMesh::fromCurvMesh(og->mesh, sp, CellOrder::Source), "ogrid");
        double vErr = 0;
        for (int c = 0; c < fm.nC; ++c)
            vErr = std::max(vErr, rel(fm.V[std::size_t(c)], cm->vol[std::size_t(fm.srcCell[std::size_t(c)])]));
        const double sErr = faceAreaError(fm, *cm, sp);
        auto count = [&](const char* nm) {
            for (const auto& p : fm.patches)
                if (p.name == nm) return p.count;
            return -1;
        };
        check(fm.dim == 2 && fm.emptyAxis == 2 && fm.nC == 64 * 22, "(b) 2D, empty axis z, 64 x 22 fluid cells");
        check(count("wall") == 64 && count("far") == 64 && count("back") == 64 * 22 && count("front") == 64 * 22,
              "(b) patches: wall 64, far 64, back/front 1408 each; angular axis internal");
        check(sErr <= 1e-14, std::format("(b) per face |S - CurvMetrics::S| / |S| = {:.2e} <= 1e-14", sErr));
        check(vErr <= 1e-13, std::format("(b) per cell |V - vol| / V = {:.2e} <= 1e-13", vErr));
        check(closure(fm).worst <= 1e-12, "(b) closed cells");
    }
    {   // (c) sinusoidal periodic mesh vs CurvMetrics
        const Grid g = Grid::make(8, 6, 4, 2.0, 1.2, 0.6);
        const CurvMesh m = CurvMesh::sinusoidal(g, 0.05);
        const auto cm = CurvMetrics::build(m);
        const StructuredSpec sp{{0, 0, 0}, {7, 5, 3}, {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"},
                                {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                                 FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic}};
        const FaceMesh fm = buildOrDie(FaceMesh::fromCurvMesh(m, sp, CellOrder::Morton), "sinusoidal");
        if (!cm) {
            check(false, "(c) CurvMetrics builds");
            return;
        }
        double vSum = 0, vDiff = 0;
        for (int c = 0; c < fm.nC; ++c) {
            vSum += fm.V[std::size_t(c)];
            vDiff = std::max(vDiff, rel(fm.V[std::size_t(c)], cm->vol[std::size_t(fm.srcCell[std::size_t(c)])]));
        }
        const double sErr = faceAreaError(fm, *cm, sp);
        check(sErr <= 1e-13, std::format("(c) per face |S - CurvMetrics::S| / |S| = {:.2e} <= 1e-13", sErr));
        check(rel(vSum, 2.0 * 1.2 * 0.6) <= 1e-12, std::format("(c) sum V = box volume ({:.2e})", rel(vSum, 1.44)));
        std::printf("  (c) per-cell |V - CurvMetrics::vol| / V max %.3e (warp order, not gated)\n", vDiff);
        check(closure(fm).worst <= 1e-12 && closure(fm).minDS > 0, "(c) closed cells, d.S > 0");
    }
}

// --- F4 ---------------------------------------------------------------------

// X equals the Source-ordered mesh up to the permutation (bitwise, sign-corrected).
bool sameGeometry(const FaceMesh& src, const FaceMesh& x)
{
    if (src.nC != x.nC || src.nI != x.nI || src.nB != x.nB) return false;
    std::map<std::int64_t, int> cellAt, faceAt;
    for (int c = 0; c < src.nC; ++c) cellAt[src.srcCell[std::size_t(c)]] = c;
    for (int f = 0; f < src.nI + src.nB; ++f) faceAt[src.srcFace[std::size_t(f)]] = f;
    for (int c = 0; c < x.nC; ++c) {
        const std::size_t s = std::size_t(cellAt.at(x.srcCell[std::size_t(c)]));
        if (x.V[std::size_t(c)] != src.V[s]) return false;
        for (std::size_t a = 0; a < 3; ++a)
            if (x.C[a][std::size_t(c)] != src.C[a][s]) return false;
    }
    for (int f = 0; f < x.nI + x.nB; ++f) {
        const std::size_t u = std::size_t(f), s = std::size_t(faceAt.at(x.srcFace[u]));
        const auto so = src.srcCell[std::size_t(src.own[s])];
        const double sg = x.srcCell[std::size_t(x.own[u])] == so ? 1.0 : -1.0;
        if (f >= x.nI && sg < 0) return false;  // boundary faces never flip
        for (std::size_t a = 0; a < 3; ++a)
            if (x.S[a][u] != sg * src.S[a][s] || x.d[a][u] != sg * src.d[a][s] || x.Cf[a][u] != src.Cf[a][s])
                return false;
    }
    if (x.patches.size() != src.patches.size()) return false;
    for (std::size_t p = 0; p < x.patches.size(); ++p)
        if (x.patches[p].count != src.patches[p].count || x.patches[p].name != src.patches[p].name) return false;
    return true;
}

bool validColouring(const FaceMesh& m)
{
    if (m.colour.size() != std::size_t(m.nC)) return false;
    for (int f = 0; f < m.nI; ++f)
        if (m.colour[std::size_t(m.own[std::size_t(f)])] == m.colour[std::size_t(m.nei[std::size_t(f)])]) return false;
    return std::ranges::is_sorted(m.colour)
           && int(*std::ranges::max_element(m.colour)) + 1 == m.nColours;
}

void f4()
{
    std::printf("F4 orderings\n");
    for (const char* name : {"cyl2d_snappy", "box_refined"}) {
        const PolyMesh pm = readOrDie(dataDir(name) / "constant" / "polyMesh");
        const FaceMesh s = buildOrDie(FaceMesh::fromPolyMesh(pm, CellOrder::Source), name);
        const FaceMesh mo = buildOrDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), name);
        const FaceMesh co = buildOrDie(FaceMesh::fromPolyMesh(pm, CellOrder::ColourMajor), name);
        check(sameGeometry(s, mo), std::string(name) + ": Morton == Source geometry (bitwise, sign-corrected)");
        check(sameGeometry(s, co), std::string(name) + ": ColourMajor == Source geometry (bitwise, sign-corrected)");
        check(validColouring(co), std::format("{}: colour-major, distance-1 valid, {} colours", name, co.nColours));
        bool csr = true;
        for (const FaceMesh* m : {&s, &mo, &co})
            for (int c = 0; c < m->nC; ++c)
                csr = csr && std::is_sorted(m->cellFace.begin() + m->cellStart[std::size_t(c)],
                                            m->cellFace.begin() + m->cellStart[std::size_t(c) + 1]);
        check(csr, std::string(name) + ": CSR faces ascending in every ordering");
        bool ownLt = true;
        for (const FaceMesh* m : {&s, &mo, &co})
            for (int f = 0; f < m->nI; ++f) ownLt = ownLt && m->own[std::size_t(f)] < m->nei[std::size_t(f)];
        check(ownLt, std::string(name) + ": own < nei on every internal face");
    }
    {
        const Grid g = Grid::make(8, 6, 4, 1.0, 0.75, 0.5);
        const StructuredSpec sp{{0, 0, 0}, {7, 5, 3}, {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"},
                                {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                                 FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic}};
        const FaceMesh co = buildOrDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::ColourMajor), "box8");
        check(co.nColours == 2 && validColouring(co), std::format("even periodic box: {} colours (want 2)", co.nColours));
    }
}

// --- F5 / F6 -----------------------------------------------------------------

void f5f6()
{
    std::printf("F5/F6 thread-count identity, repeatability\n");
    const int hw = int(std::max(1u, std::thread::hardware_concurrency()));
    for (const char* name : {"cyl2d_snappy", "box_refined"}) {
        const PolyMesh pm = readOrDie(dataDir(name) / "constant" / "polyMesh");
        const FaceMesh ref = buildOrDie(FaceMesh::fromPolyMesh(pm, CellOrder::ColourMajor, 1), name);
        const la::CellGraph gref = la::graphOf(ref);
        bool same = true;
        for (const int t : {2, 3, 7, hw, 4 * hw}) {
            const FaceMesh m = buildOrDie(FaceMesh::fromPolyMesh(pm, CellOrder::ColourMajor, t), name);
            same = same && byteIdentical(ref, m) && la::graphOf(m) == gref;
        }
        check(same, std::format("{}: threads 1/2/3/7/{}/{} byte-identical FaceMesh and CellGraph", name, hw, 4 * hw));
        const FaceMesh again = buildOrDie(FaceMesh::fromPolyMesh(pm, CellOrder::ColourMajor, 1), name);
        check(byteIdentical(ref, again), std::string(name) + ": two builds byte-identical");
    }
}

// --- F8 (local) ----------------------------------------------------------------

int f8()
{
#ifndef DNS_LOCAL_MESH_MB
    dnstest::skip("DNS_LOCAL_MESH_MB not configured");
#else
    const fs::path dir = fs::path(DNS_LOCAL_MESH_MB) / "constant" / "polyMesh";
    if (!fs::exists(dir / "owner")) dnstest::skip(std::string("no local mesh at ") + dir.string());
    std::printf("F8 mB (831k-cell snappy mesh)\n");
    const auto t0 = std::chrono::steady_clock::now();
    const PolyMesh pm = readOrDie(dir);
    const auto t1 = std::chrono::steady_clock::now();
    const FaceMesh fm = buildOrDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), "mB");
    const auto t2 = std::chrono::steady_clock::now();
    const MeshQuality q = meshQuality(fm, pm);
    const double tr = std::chrono::duration<double>(t1 - t0).count(),
                 tb = std::chrono::duration<double>(t2 - t1).count();
    std::printf("  read %.2f s, FaceMesh build %.2f s (hw threads)\n", tr, tb);
    // The pinned numbers are mB's log.checkMesh (run on 8 ranks: its face
    // count 2622899 includes the processor faces; the reconstructed mesh on
    // disk has 2600459 -- the FaceMesh's internal + boundary count).
    check(fm.nC == 831578 && fm.nI + fm.nB == 2600459,
          std::format("mB: {} cells, {} faces (831578, 2600459)", fm.nC, fm.nI + fm.nB));
    check(rel(q.vTotal, 5.899292884) <= 1e-9, std::format("mB: total volume {:.12g} vs 5.899292884", q.vTotal));
    check(rel(q.nonOrthMax, 65.06425232) <= 1e-6 && rel(q.nonOrthAvg, 9.764831307) <= 1e-6,
          std::format("mB: non-orth {:.12g}/{:.12g} vs 65.06425232/9.764831307", q.nonOrthMax, q.nonOrthAvg));
    check(rel(q.skewMax, 3.265908371) <= 1e-6, std::format("mB: skewness {:.12g} vs 3.265908371", q.skewMax));
    check(rel(q.aspectMax, 16.64481943) <= 1e-6, std::format("mB: aspect {:.12g} vs 16.64481943", q.aspectMax));
    check(tr + tb <= 30.0, std::format("mB: read + build {:.2f} s <= 30 s", tr + tb));
    check(closure(fm).worst <= 1e-12 && closure(fm).minDS > 0, "mB: closed cells, d.S > 0");
    return dnstest::finish();
#endif
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "mb") return f8();
    f1f2();
    f3();
    f4();
    f5f6();
    return dnstest::finish();
}
