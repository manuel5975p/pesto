// The RAS wall distance of the FV path (dns/fvwalldist, phase BL) vs
// OpenFOAM v2606's own on the identical mesh: checkMesh -writeFields
// '(wallDistance)' (wallDist::New(mesh).y() with the case's fvSchemes method)
// stored as tests/data/fv/<case>/ref/wallDistance.
//   meshWave (sst_plate, sa_skew): FaceCellWave of wallPoint, serial order,
//     propagationTol 0.01, + correctBoundaryCells -- must agree to rounding
//     (1e-12 relative): an order or tolerance slip moves whole cells by ~1%.
//   exactDistance (sst_skew): nearest point on the wall faces (1e-12).
// Plus the invariants: exact <= meshWave everywhere (meshWave is a distance
// to wall-face CENTRES away from the walls), the wall-face cells equal in both.
// Usage: test_fvwalldist [CASE_DIR WALLDIST_FILE] (a case outside the suite).
#include "dns/foamio.hpp"
#include "dns/fvsimple.hpp"
#include "dns/fvwalldist.hpp"

#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <string>

#ifndef DNS_FV_DATA
#error "DNS_FV_DATA must point at tests/data/fv"
#endif

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

void compare(const fs::path& dir, const fs::path& ref, double gate)
{
    std::printf("wall distance %s\n", dir.string().c_str());
    auto L = loadFvCase(dir);
    if (!L) {
        check(false, L.error());
        return;
    }
    const auto& r = L->c.ras;
    auto w = fvWallDistance(L->pm, L->fm, r.exactDist, r.correctWalls);
    if (!w) {
        check(false, w.error());
        return;
    }
    auto foam = readFoamInternalField(ref, 1, L->pm.man.nCells);
    if (!foam) {
        check(false, foam.error());
        return;
    }
    double worst = 0, scale = 0;
    std::size_t at = 0, n12 = 0, n6 = 0, n3 = 0;
    for (std::size_t c = 0; c < w->y.size(); ++c) {
        const double theirs = (*foam)[std::size_t(L->fm.srcCell[c])];
        const double d = std::abs(w->y[c] - theirs) / std::max(std::abs(theirs), 1e-300);
        if (d > worst) {
            worst = d;
            at = c;
        }
        n12 += d > 1e-12;
        n6 += d > 1e-6;
        n3 += d > 1e-3;
        scale = std::max(scale, theirs);
    }
    std::printf("    cells off by > 1e-12 / 1e-6 / 1e-3 relative: %zu / %zu / %zu\n", n12, n6, n3);
    if (n3) { // where: wall-face cells, wall-point cells, interior (debugging aid)
        auto ex = fvWallDistance(L->pm, L->fm, true, true);
        std::vector<std::uint8_t> wallFace(w->y.size(), 0), wallPoint(L->pm.px.size(), 0);
        for (const auto& p : L->pm.man.patches)
            if (p.type == FoamPatchType::Wall)
                for (std::int64_t f = p.startFace; f < p.startFace + p.nFaces; ++f) {
                    for (int k = L->pm.faceStart[std::size_t(f)]; k < L->pm.faceStart[std::size_t(f + 1)]; ++k)
                        wallPoint[std::size_t(L->pm.faceNodes[std::size_t(k)])] = 1;
                }
        std::vector<int> fmOf(w->y.size());
        for (std::size_t c = 0; c < w->y.size(); ++c) fmOf[std::size_t(L->fm.srcCell[c])] = int(c);
        for (const auto& p : L->pm.man.patches)
            if (p.type == FoamPatchType::Wall)
                for (std::int64_t f = p.startFace; f < p.startFace + p.nFaces; ++f)
                    wallFace[std::size_t(fmOf[std::size_t(L->pm.owner[std::size_t(f)])])] = 1;
        std::size_t nwf = 0, nwp = 0, nin = 0, below = 0;
        for (std::size_t c = 0; c < w->y.size(); ++c) {
            const double theirs = (*foam)[std::size_t(L->fm.srcCell[c])];
            if (std::abs(w->y[c] - theirs) / theirs <= 1e-3) continue;
            if (wallFace[c]) ++nwf;
            else {
                bool touches = false;
                for (int k = L->fm.cellStart[c]; k < L->fm.cellStart[c + 1] && !touches; ++k) {
                    const std::int64_t f = L->fm.srcFace[std::size_t(L->fm.cellFace[std::size_t(k)])];
                    for (int j = L->pm.faceStart[std::size_t(f)]; j < L->pm.faceStart[std::size_t(f + 1)]; ++j)
                        touches = touches || wallPoint[std::size_t(L->pm.faceNodes[std::size_t(j)])];
                }
                (touches ? nwp : nin) += 1;
            }
            below += w->y[c] < theirs;
            if (nwf + nwp + nin <= 5 && ex)
                std::printf("      cell %lld: mine %.9e theirs %.9e exact %.9e%s\n", (long long)L->fm.srcCell[c],
                            w->y[c], theirs, ex->y[c], wallFace[c] ? " (wall-face cell)" : "");
        }
        std::printf("    > 1e-3: %zu wall-face cells, %zu wall-point cells, %zu interior; mine below theirs %zu\n", nwf,
                    nwp, nin, below);
    }
    check(worst < gate, std::format("{} cells, {}: worst relative difference {:.2e} at cell {} (gate {:.0e})",
                                    w->y.size(), r.exactDist ? "exactDistance" : "meshWave", worst,
                                    L->fm.srcCell[at], gate));
    // exact <= meshWave (meshWave only ever overestimates)
    auto e = fvWallDistance(L->pm, L->fm, true, true);
    auto m = fvWallDistance(L->pm, L->fm, false, true);
    if (!e || !m) return;
    double over = 0;
    for (std::size_t c = 0; c < e->y.size(); ++c) over = std::max(over, (e->y[c] - m->y[c]) / m->y[c]);
    check(over < 1e-12, std::format("exact <= meshWave everywhere (max (exact - meshWave)/meshWave {:.2e})", over));
    double wallCells = 0;
    for (int b = 0; b < L->fm.nB; ++b)
        if (e->yWall[std::size_t(b)] > 0) {
            const std::size_t c = std::size_t(L->fm.own[std::size_t(L->fm.nI + b)]);
            wallCells = std::max(wallCells, std::abs(e->y[c] - m->y[c]) / m->y[c]);
        }
    std::printf("    wall-face cells: |exact - meshWave| / meshWave <= %.2e\n", wallCells);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 4) { // CASE C_FILE: the FaceMesh cell centres vs OpenFOAM's (writeCellCentres, 17 digits)
        auto L = loadFvCase(argv[1]);
        if (!L) return std::printf("%s\n", L.error().c_str()), 1;
        auto C = readFoamInternalField(argv[2], 3, L->pm.man.nCells);
        if (!C) return std::printf("%s\n", C.error().c_str()), 1;
        double mx = 0;
        std::size_t nbit = 0;
        for (std::size_t c = 0; c < std::size_t(L->fm.nC); ++c) {
            const std::size_t s = std::size_t(L->fm.srcCell[c]);
            bool same = true;
            for (int q = 0; q < 3; ++q) {
                const double a = L->fm.C[std::size_t(q)][c], b = (*C)[3 * s + std::size_t(q)];
                mx = std::max(mx, std::abs(a - b));
                same = same && a == b;
            }
            nbit += !same;
        }
        std::printf("cell centres: max |FaceMesh - OpenFOAM| %.3e, %zu of %d cells not bitwise equal (17 digits)\n",
                    mx, nbit, L->fm.nC);
        return 0;
    }
    if (argc == 3) {
        if (!fs::exists(argv[2])) {
            std::printf("skip: %s not present (local run directory)\n", argv[2]);
            return 77;
        }
        // OpenFOAM writes wallDistance at the case's writePrecision (12 here)
        compare(argv[1], argv[2], 1e-11);
        return dnstest::finish();
    }
    for (const char* c : {"sst_plate", "sst_skew", "sa_skew"}) {
        const fs::path d = fs::path(DNS_FV_DATA) / c;
        compare(d, d / "ref" / "wallDistance", 1e-12);
    }
    return dnstest::finish();
}
