// G7 (docs/nozzle.md): the sensor's mesh lines. Host. A uniform periodic box is the structured
// stride-2 sensor's bytes; prisms carry no radial line; a mesh rotated 45 deg keeps two
// distinct lines per cell where the structured +-e_a walk picks one; bitwise the same lines
// under every CellOrder, ties (a snappy mesh) included.
#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/foamio.hpp"
#include "dns/fvgrad.hpp"
#include "dns/fvshock.hpp"
#include "dns/shockflux.hpp"

#include "check.hpp"
#include "shock_ref.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <numbers>
#include <set>
#include <vector>

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

template <class T>
T must(std::expected<T, std::string> e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL: %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    return std::move(*e);
}

FvShockLines linesOf(const FaceMesh& m)
{
    return fvShockLines(m, must(fvFaceGeom(m, FvFaceGradRule::AlongD, 1.0), "geom"));
}

void periodic()
{
    std::printf("uniform periodic box == the structured sensor's bytes\n");
    const int n = 64;
    Grid g;
    g.nx = n;
    g.ny = 6;
    g.nz = 1;
    g.hx = g.hy = 1.0 / n;
    g.hz = 1.0;
    const StructuredSpec sp{{0, 0, 0},
                            {g.nx - 1, g.ny - 1, 0},
                            {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                             FoamPatchType::Empty, FoamPatchType::Empty}};
    const FaceMesh m = must(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Morton), "box");
    const FvShockLines l = linesOf(m);
    check(l.line.size() == 2 * 2 * std::size_t(m.nC), "two lines per cell");
    shockref::State<double> s;
    s.q[0].assign(std::size_t(n), 1.0);
    s.q[1].assign(std::size_t(n), 0.0);
    s.q[2].resize(std::size_t(n));
    for (int i = 0; i < n; ++i) { // a double shock, smooth waves between
        const double x = (i + 0.5) / n, p = (x > 0.2 && x < 0.55 ? 5.0 : 1.0) + 0.05 * std::sin(6.0 * 2 * std::numbers::pi * x);
        s.q[2][std::size_t(i)] = p / (shockref::kGamma - 1.0);
    }
    const shockref::Cfg<double> c{};
    const std::vector<std::uint8_t> ref = shockref::senseBytes(s, c);
    std::vector<double> p(std::size_t(m.nC));
    for (std::size_t q = 0; q < p.size(); ++q) {
        const std::size_t I = std::size_t(m.srcCell[q]) % std::size_t(n);
        p[q] = shockref::pressureOf(s.q[0][I], s.q[1][I], s.q[2][I]);
    }
    std::vector<std::uint8_t> b(p.size());
    fvShockSense(l, p, b);
    long diff = 0, sensed = 0;
    for (std::size_t q = 0; q < p.size(); ++q) {
        const std::uint8_t r = ref[std::size_t(m.srcCell[q]) % std::size_t(n)];
        diff += b[q] != r;
        sensed += b[q] != 0;
    }
    std::printf("  %ld cells sensed, %ld bytes differ\n", sensed, diff);
    check(diff == 0 && sensed > 0, "bytes equal the structured sensor's, live");
}

// x-axis distance of cells c-2, c+2 of line l of cell c: the line's own span.
void prisms()
{
    std::printf("wedge prisms\n");
    const fs::path d = fs::path(DNS_MESH_DATA) / ".." / "fv" / "nozzle_W";
    const PolyMesh pm = must(readPolyMesh(d / "constant/polyMesh"), "W");
    const FaceMesh m = must(FaceMesh::fromPolyMesh(pm, CellOrder::Source, 0, {.wedge = true}), "W");
    const FvShockLines l = linesOf(m);
    const int nx = 80;
    int nCount[8] = {};
    bool rows = true, interior = true;
    for (int c = 0; c < m.nC; ++c) {
        const int i = c % nx, j = c / nx;
        const int nl = l.lineStart[std::size_t(c) + 1] - l.lineStart[std::size_t(c)];
        ++nCount[std::min(nl, 7)];
        const bool xw = i >= 2 && i < nx - 2;
        if (j < 2) rows = rows && nl == (xw ? 1 : 0);
        if (j >= 2 && j < 14 && xw) interior = interior && nl == 2;
    }
    std::printf("  cells with 0, 1, 2, 3 lines: %d %d %d %d\n", nCount[0], nCount[1], nCount[2], nCount[3]);
    check(rows, "rows 0, 1: no radial line, the x line alone");
    check(interior, "interior: x and radial; a side-to-side line never exists");
}

// Two cells out of c along +e_a, the structured axis walk's pick: the face most aligned with e_a,
// ties to the lowest srcFace, the cell beyond, then again.
int walkAxis(const FaceMesh& m, int c, int a)
{
    for (int k = 0; k < 2; ++k) {
        int best = -1;
        double bc = 0.0;
        for (int q = m.cellStart[std::size_t(c)]; q < m.cellStart[std::size_t(c) + 1]; ++q) {
            const std::size_t f = std::size_t(m.cellFace[std::size_t(q)]);
            if (f >= std::size_t(m.nI)) continue;
            const double sg = m.own[f] == c ? 1.0 : -1.0;
            const double s = sg * m.S[std::size_t(a)][f] / std::sqrt(m.S[0][f] * m.S[0][f] + m.S[1][f] * m.S[1][f]);
            // fvTflesWalk's rule: ties to the lowest srcFace.
            const bool tie = std::abs(s - bc) <= 1e-12 && best >= 0 && m.srcFace[f] < m.srcFace[std::size_t(best)];
            if (s > bc + 1e-12 || tie) {
                bc = s;
                best = int(f);
            }
        }
        if (best < 0) return -1;
        c = m.own[std::size_t(best)] == c ? m.nei[std::size_t(best)] : m.own[std::size_t(best)];
    }
    return c;
}

void rotated()
{
    std::printf("a mesh rotated 135 deg (45 deg to both axes, the +-e_a tie falls on one face)\n");
    Grid g;
    g.nx = 24;
    g.ny = 24;
    g.nz = 1;
    g.hx = g.hy = 1.0;
    g.hz = 1.0;
    const StructuredSpec sp{{0, 0, 0},
                            {g.nx - 1, g.ny - 1, 0},
                            {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Patch,
                             FoamPatchType::Empty, FoamPatchType::Empty}};
    PolyMesh pm = must(FaceMesh::polyOf(CurvMesh::uniform(g), sp), "polyOf");
    const double r = std::sqrt(0.5); // exact ties: one constant, not cos and sin of a rounded angle
    for (std::size_t i = 0; i < pm.px.size(); ++i) {
        const double x = pm.px[i], y = pm.py[i];
        pm.px[i] = -r * x - r * y;
        pm.py[i] = r * x - r * y;
    }
    const FaceMesh m = must(FaceMesh::fromPolyMesh(pm, CellOrder::Source), "rotated");
    const FvShockLines l = linesOf(m);
    bool two = true;
    long walkSame = 0, n = 0;
    for (int c = 0; c < m.nC; ++c) {
        const int i = c % g.nx, j = c / g.nx;
        if (i < 2 || i >= g.nx - 2 || j < 2 || j >= g.ny - 2) continue;
        ++n;
        const int b = l.lineStart[std::size_t(c)], e = l.lineStart[std::size_t(c) + 1];
        std::set<int> ends;
        for (int q = 2 * b; q < 2 * e; ++q) ends.insert(l.line[std::size_t(q)]);
        two = two && e - b == 2 && ends.size() == 4;
        walkSame += walkAxis(m, c, 0) == walkAxis(m, c, 1);
    }
    std::printf("  %ld interior cells; the +-e_a walk's two axes pick one cell on %ld\n", n, walkSame);
    check(two, "2 distinct lines per interior cell");
    check(walkSame > n / 2, "control: the +-e_a walk collapses its axes here");
}

void orders(const fs::path& poly, FaceMeshOptions opt, const char* tag, const fs::path& pfile)
{
    std::printf("%s under Source, Morton, Colour\n", tag);
    const PolyMesh pm = must(readPolyMesh(poly), tag);
    std::vector<std::uint8_t> ref;
    std::vector<int> refLine;
    int nI = 0;
    for (CellOrder o : {CellOrder::Source, CellOrder::Morton, CellOrder::ColourMajor}) {
        const FaceMesh m = must(FaceMesh::fromPolyMesh(pm, o, 0, opt), tag);
        const FvShockLines l = linesOf(m);
        std::vector<double> p(std::size_t(m.nC));
        std::vector<double> src(p.size());
        if (pfile.empty()) {
            for (std::size_t c = 0; c < src.size(); ++c) src[c] = 1.0 + 0.5 * std::sin(double(c) * c * 1e-3);
        } else {
            src = must(readFoamInternalField(pfile, 1, m.nC), "p");
        }
        for (std::size_t c = 0; c < p.size(); ++c) p[c] = src[std::size_t(m.srcCell[c])];
        std::vector<std::uint8_t> b(p.size()), bs(p.size());
        fvShockSense(l, p, b);
        for (std::size_t c = 0; c < p.size(); ++c) bs[std::size_t(m.srcCell[c])] = b[c];
        // P-, N+ by source face and source cell, oriented by source owner.
        std::vector<int> fl(2 * std::size_t(pm.neighbour.size()) + 2 * std::size_t(m.nI), -2);
        nI = m.nI;
        for (std::size_t f = 0; f < std::size_t(m.nI); ++f) {
            const std::size_t src2 = std::size_t(m.srcFace[f]);
            if (src2 >= pm.neighbour.size()) continue; // a cyclic: pm numbers it past the internals
            const bool flip = pm.owner[src2] != m.srcCell[std::size_t(m.own[f])];
            const int a = l.faceLine[2 * f], c = l.faceLine[2 * f + 1];
            const int x = a < 0 ? -1 : int(m.srcCell[std::size_t(a)]), y = c < 0 ? -1 : int(m.srcCell[std::size_t(c)]);
            fl[2 * src2] = flip ? y : x;
            fl[2 * src2 + 1] = flip ? x : y;
        }
        if (o == CellOrder::Source) {
            ref = bs;
            refLine = fl;
            continue;
        }
        check(bs == ref, std::format("{}: bytes equal Source's", tag));
        check(fl == refLine, std::format("{}: P-, N+ equal Source's", tag));
    }
    std::printf("  %d internal faces\n", nI);
}

} // namespace

int main()
{
    periodic();
    prisms();
    rotated();
    const fs::path w = fs::path(DNS_MESH_DATA) / ".." / "fv" / "nozzle_B";
    orders(w / "constant/polyMesh", {}, "nozzle_B", w / "ref/2.4/p");
    const fs::path s = fs::path(DNS_MESH_DATA) / "cyl2d_snappy" / "constant/polyMesh";
    orders(s, {}, "cyl2d_snappy (hanging faces)", {});
    orders(fs::path(DNS_MESH_DATA) / "box_refined" / "constant/polyMesh", {}, "box_refined (exact ties: the srcFace rule)", {});
    return dnstest::finish();
}
