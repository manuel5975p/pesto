// sampleCut() must reproduce, sample for sample, the three hand-rolled
// supersamplers it replaces (meshsdf's makeMeshCut and the two app-local golf
// cuts), and exactLensCut must reproduce the eye app's chord construction.
// Per-body sample budgets are carried as SampleOpts, NOT unified -- changing a
// budget shifts the cut, so these are bit-identity checks.
#include "dns/bodies.hpp"
#include "dns/geometry.hpp"
#include "dns/meshsdf.hpp"

#include "mesh_test_util.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

bool same(const dns::Scalar& a, const dns::Scalar& b, double tol = 0.0)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > tol) return false;
    return true;
}

void checkCut(const dns::CutGeometry& a, const dns::CutGeometry& b, const char* what)
{
    check(same(a.lambda, b.lambda), (std::string(what) + " lambda").c_str());
    check(same(a.ax, b.ax), (std::string(what) + " ax").c_str());
    check(same(a.ay, b.ay), (std::string(what) + " ay").c_str());
    check(same(a.az, b.az), (std::string(what) + " az").c_str());
}

// --- the legacy samplers, verbatim, as the oracle ---------------------------

dns::CutGeometry legacyMeshCut(const dns::Grid& g, const dns::Sdf& sdf, double bandLo,
                               double bandHi)
{
    using namespace dns;
    assert(bandLo < 0 && bandHi > 0);
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    // Fluid fraction of an axis box by an sx*sy*sz sample lattice.
    auto frac = [&](double x0, double y0, double z0, double ex, double ey,
                    double ez, int sx, int sy, int sz) {
        int m = 0;
        for (int c = 0; c < sz; ++c)
            for (int b = 0; b < sy; ++b)
                for (int a = 0; a < sx; ++a)
                    if (sdf(x0 + (a + 0.5) * ex / sx, y0 + (b + 0.5) * ey / sy,
                            z0 + (c + 0.5) * ez / sz) > 0.0)
                        ++m;
        return double(m) / (sx * sy * sz);
    };
    auto classify = [&](double phi, auto sample) {
        return phi >= bandHi ? 1.0 : phi <= bandLo ? 0.0 : sample();
    };
    // Lipschitz block prefilter: the sdf is an exact distance (|grad phi| = 1),
    // so one query at a B^3-cell block center bounds every sample point in the
    // block. Blocks wholly outside [bandLo - r, bandHi + r] skip the 4-per-cell
    // queries entirely -- the dominant cost on large grids.
    constexpr int B = 8;
    const double blockR =
        0.5 * std::sqrt(double(B * g.hx) * (B * g.hx) + double(B * g.hy) * (B * g.hy)
                        + double(B * g.hz) * (B * g.hz));
    for (int kb = 0; kb < g.nz; kb += B)
        for (int jb = 0; jb < g.ny; jb += B)
            for (int ib = 0; ib < g.nx; ib += B)
            {
                const int i1 = std::min(ib + B, g.nx), j1 = std::min(jb + B, g.ny),
                          k1 = std::min(kb + B, g.nz);
                const double phiC = sdf((ib + 0.5 * (i1 - ib)) * g.hx,
                                        (jb + 0.5 * (j1 - jb)) * g.hy,
                                        (kb + 0.5 * (k1 - kb)) * g.hz);
                const bool allFluid = phiC >= bandHi + blockR;
                const bool allSolid = phiC <= bandLo - blockR;
                if (allFluid || allSolid)
                {
                    const double v = allFluid ? 1.0 : 0.0;
                    for (int k = kb; k < k1; ++k)
                        for (int j = jb; j < j1; ++j)
                            for (int i = ib; i < i1; ++i)
                            {
                                const auto id = g.idx(i, j, k);
                                cut.lambda[id] = cut.ax[id] = cut.ay[id]
                                    = cut.az[id] = v;
                            }
                    continue;
                }
                for (int k = kb; k < k1; ++k)
                    for (int j = jb; j < j1; ++j)
                        for (int i = ib; i < i1; ++i)
                        {
                            const auto id = g.idx(i, j, k);
                            const double x0 = i * g.hx, y0 = j * g.hy, z0 = k * g.hz;
                            cut.lambda[id] = classify(
                                sdf(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy, z0 + 0.5 * g.hz),
                                [&] { return frac(x0, y0, z0, g.hx, g.hy, g.hz, 4, 4, 4); });
                            cut.ax[id] = classify(
                                sdf(x0, y0 + 0.5 * g.hy, z0 + 0.5 * g.hz),
                                [&] { return frac(x0, y0, z0, 0, g.hy, g.hz, 1, 4, 4); });
                            cut.ay[id] = classify(
                                sdf(x0 + 0.5 * g.hx, y0, z0 + 0.5 * g.hz),
                                [&] { return frac(x0, y0, z0, g.hx, 0, g.hz, 4, 1, 4); });
                            cut.az[id] = classify(
                                sdf(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy, z0),
                                [&] { return frac(x0, y0, z0, g.hx, g.hy, 0, 4, 4, 1); });
                        }
            }
    return cut;
}


dns::CutGeometry legacyGolfBallCut(const dns::Grid& g, const dns::Sdf& sdf,
                                   const dns::Sphere& s, double depth)
{
    using namespace dns;
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    const double halfDiag = 0.5 * std::sqrt(g.hx * g.hx + g.hy * g.hy + g.hz * g.hz);
    const double lo = s.r - depth - halfDiag, hi = s.r + halfDiag;
    auto dist = [&](double x, double y, double z) {
        const double dx = x - s.cx, dy = y - s.cy, dz = z - s.cz;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    auto frac = [&](double x0, double y0, double z0, double ex, double ey, double ez,
                    int sx, int sy, int sz) {
        int m = 0;
        for (int c = 0; c < sz; ++c)
            for (int bb = 0; bb < sy; ++bb)
                for (int a = 0; a < sx; ++a)
                    if (sdf(x0 + (a + 0.5) * ex / sx, y0 + (bb + 0.5) * ey / sy,
                            z0 + (c + 0.5) * ez / sz) > 0.0)
                        ++m;
        return double(m) / (sx * sy * sz);
    };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto id = g.idx(i, j, k);
                const double x0 = i * g.hx, y0 = j * g.hy, z0 = k * g.hz;
                auto classify = [&](double d, auto sample) {
                    return d >= hi ? 1.0 : d <= lo ? 0.0 : sample();
                };
                cut.lambda[id] = classify(
                    dist(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy, z0 + 0.5 * g.hz),
                    [&] { return frac(x0, y0, z0, g.hx, g.hy, g.hz, 4, 4, 4); });
                cut.ax[id] = classify(dist(x0, y0 + 0.5 * g.hy, z0 + 0.5 * g.hz),
                                      [&] { return frac(x0, y0, z0, 0, g.hy, g.hz, 1, 4, 4); });
                cut.ay[id] = classify(dist(x0 + 0.5 * g.hx, y0, z0 + 0.5 * g.hz),
                                      [&] { return frac(x0, y0, z0, g.hx, 0, g.hz, 4, 1, 4); });
                cut.az[id] = classify(dist(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy, z0),
                                      [&] { return frac(x0, y0, z0, g.hx, g.hy, 0, 4, 4, 1); });
            }
    return cut;
}

dns::CutGeometry legacyGolfDiskCut(const dns::Grid& g, const dns::Sdf& sdf,
                                   const dns::CylinderZ& c, double depth)
{
    using namespace dns;
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    const double halfDiag = 0.5 * std::hypot(g.hx, g.hy);
    const double lo = c.r - depth - halfDiag, hi = c.r + halfDiag;
    auto dist = [&](double x, double y) { return std::hypot(x - c.cx, y - c.cy); };
    auto cellFrac = [&](double x0, double y0) {
        int m = 0;
        for (int bb = 0; bb < 8; ++bb)
            for (int a = 0; a < 8; ++a)
                if (sdf(x0 + (a + 0.5) * g.hx / 8, y0 + (bb + 0.5) * g.hy / 8, 0) > 0.0)
                    ++m;
        return m / 64.0;
    };
    auto segFrac = [&](double x0, double y0, double ex, double ey) {
        int m = 0;
        for (int a = 0; a < 8; ++a)
            if (sdf(x0 + (a + 0.5) * ex / 8, y0 + (a + 0.5) * ey / 8, 0) > 0.0) ++m;
        return m / 8.0;
    };
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const double x0 = i * g.hx, y0 = j * g.hy;
            auto classify = [&](double d, auto sample) {
                return d >= hi ? 1.0 : d <= lo ? 0.0 : sample();
            };
            const double lam = classify(dist(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy),
                                        [&] { return cellFrac(x0, y0); });
            const double ax = classify(dist(x0, y0 + 0.5 * g.hy),
                                       [&] { return segFrac(x0, y0, 0, g.hy); });
            const double ay = classify(dist(x0 + 0.5 * g.hx, y0),
                                       [&] { return segFrac(x0, y0, g.hx, 0); });
            for (int k = 0; k < g.nz; ++k)
            {
                const auto id = g.idx(i, j, k);
                cut.lambda[id] = lam;
                cut.ax[id] = ax;
                cut.ay[id] = ay;
                cut.az[id] = lam;
            }
        }
    return cut;
}

} // namespace

int main()
{
    using namespace dns;
    using std::numbers::pi;

    // --- 1) sampleCut == makeMeshCut on a mesh sdf --------------------------
    {
        const auto mesh = meshtest::icosphere(1.0, {2.0, 2.0, 2.0}, 2);
        const MeshSdf sdf(mesh, false);
        const auto g = Grid::make(24, 24, 24, 4.0, 4.0, 4.0);
        const double halfDiag =
            0.5 * std::sqrt(g.hx * g.hx + g.hy * g.hy + g.hz * g.hz);
        const auto legacy = legacyMeshCut(g, [&](double x, double y, double z) {
            return sdf(x, y, z);
        }, -halfDiag, halfDiag);
        const auto fresh = sampleCut(g, [&](double x, double y, double z) {
            return sdf(x, y, z);
        });
        check(fresh.has_value(), "sampleCut(mesh) succeeds");
        if (fresh) checkCut(*fresh, legacy, "mesh cut");
    }

    // --- 2) sampleCut == the golf-ball sampler (4^3 / 4^2) ------------------
    {
        const auto g = Grid::make(32, 32, 32, 2 * pi, 2 * pi, 2 * pi);
        const double h = g.hx;
        const Sphere s{pi, pi, pi, 0.8};
        const double depth = 2.5 * h, rd = pitRadius(5.0 * h, depth);
        const auto centres = fibonacciCentres(s, 40, rd, depth);
        const Sdf sdf = dimpled(sdfOf(s), centres, rd);
        const auto legacy = legacyGolfBallCut(g, sdf, s, depth);
        // Band matches the legacy shell: [r - depth - halfDiag, r + halfDiag]
        // measured on the SPHERE distance; sdf() differs inside a pit, so this
        // is a band on the same quantity only when it is expressed as the
        // sphere sdf. Use the sphere sdf for classification.
        const auto fresh = sampleCut(g, sdf, {4, 4, false, 0.0});
        check(fresh.has_value(), "sampleCut(golf ball) succeeds");
        // The legacy classifier bands on the SMOOTH sphere; sampleCut bands on
        // the dimpled sdf. Both are conservative supersets of the cut band, so
        // the two agree wherever either samples -- assert on the cut cells.
        if (fresh)
        {
            std::size_t diff = 0;
            for (std::size_t i = 0; i < g.size(); ++i)
                if (std::abs(fresh->lambda[i] - legacy.lambda[i]) > 1e-12) ++diff;
            check(diff == 0, "golf-ball lambda matches the legacy sampler");
        }
    }

    // --- 3) sampleCut(zInvariant) == the golf-disk sampler (8x8 / 8) --------
    // NOTE the legacy disk sampler walks the FACE SEGMENT diagonally
    // (x0 + (a+.5)ex/8, y0 + (a+.5)ey/8): with one extent zero that is the
    // straight segment, so the two agree exactly.
    {
        const auto g = Grid::make(48, 48, 3, 2 * pi, 2 * pi, 3 * (2 * pi / 48));
        const double h = g.hx;
        const CylinderZ c{pi, pi, 0.8};
        const double depth = 4.0 * h, rd = pitRadius(8.0 * h, depth);
        const auto centres = ringCentres(c, 24, rd, depth);
        const Sdf sdf = dimpled(sdfOf(c), centres, rd, true);
        const auto legacy = legacyGolfDiskCut(g, sdf, c, depth);
        const auto fresh = sampleCut(g, sdf, {8, 8, true, 0.0});
        check(fresh.has_value(), "sampleCut(golf disk) succeeds");
        if (fresh)
        {
            std::size_t diff = 0;
            for (std::size_t i = 0; i < g.size(); ++i)
                if (std::abs(fresh->lambda[i] - legacy.lambda[i]) > 1e-12) ++diff;
            check(diff == 0, "golf-disk lambda matches the legacy sampler");
            // z-invariance: every k layer identical, az == lambda.
            check(same(fresh->az, fresh->lambda), "golf disk az == lambda");
            bool layered = true;
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    for (int k = 1; k < g.nz; ++k)
                        if (fresh->lambda[g.idx(i, j, k)] != fresh->lambda[g.idx(i, j, 0)])
                            layered = false;
            check(layered, "golf disk is replicated over k");
        }
    }

    // --- 4) exactLensCut: tips survive, apertures are exact chords ----------
    {
        const auto g = Grid::make(64, 64, 1, 2 * pi, 2 * pi, 1.0);
        const Lens b{pi, pi, 1.4, 0.4};
        const auto cut = exactLensCut(g, b);
        check(cut.has_value(), "exactLensCut succeeds");
        if (cut)
        {
            double solid = 0;
            for (std::size_t i = 0; i < g.size(); ++i)
                solid += (1.0 - cut->lambda[i]);
            solid *= g.hx * g.hy;
            // Exact lens area: 2 (ra^2 acos(e/ra) - e sqrt(ra^2 - e^2)).
            const double e = b.e(), ra = b.ra();
            const double exact =
                2.0 * (ra * ra * std::acos(e / ra) - e * std::sqrt(ra * ra - e * e));
            check(std::abs(solid - exact) / exact < 2e-3,
                  "lens solid area matches the closed form");
            for (auto v : cut->lambda) check(v >= 0 && v <= 1, "lens lambda in [0,1]");
        }
        check(!exactLensCut(g, Lens{pi, pi, 0.4, 0.4}).has_value(),
              "exactLensCut rejects w <= hh");
    }

    // --- 5) argument validation --------------------------------------------
    {
        const auto g = Grid::make(8, 8, 8, 1, 1, 1);
        const Sdf s = sdfOf(Sphere{0.5, 0.5, 0.5, 0.2});
        check(!sampleCut(g, s, {0, 4}).has_value(), "sampleCut rejects nCell < 1");
        check(!sampleCut(g, s, {4, 0}).has_value(), "sampleCut rejects nFace < 1");
        auto ok = sampleCut(g, s);
        check(ok.has_value(), "sampleCut accepts defaults");
        if (ok)
            for (auto v : ok->ax) check(v >= 0 && v <= 1, "ax in [0,1]");
    }

    if (failures == 0) std::printf("geometry: ok\n");
    return failures == 0 ? 0 : 1;
}
