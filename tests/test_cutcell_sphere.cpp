// Sphere cut-cell geometry + small-cell sanitization + apertured projection.
//  1) fluid-fraction volume deficit matches 4/3 pi r^3 (exact-geometry check).
//  2) x-face aperture deficit on a plane matches the cross-circle area.
//  3) sanitizeCut invariants: open faces join fluid cells, fluid cells keep an
//     open face, thresholds respected.
//  4) projectCut drives the apertured divergence to zero on a noise field.
#include "dns/cutcell.hpp"
#include "dns/operators.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <random>

using namespace dns;

int main()
{
    using std::numbers::pi;
    const int n = 32;
    const double L = 2 * pi;
    const auto g = Grid::make(n, n, n, L, L, L);
    const double h = g.hx;

    // Centre deliberately off-node so tangencies produce arbitrarily small cut
    // cells -- the case sanitizeCut exists for.
    const Sphere s{L / 2 + 0.137 * h, L / 2 - 0.293 * h, L / 2 + 0.051 * h, L / 5};
    auto cut = CutGeometry::make(g, s);

    bool ok = true;
    auto check = [&](const char* what, double err, double tol) {
        std::printf("%-28s %.3e (tol %.1e)\n", what, err, tol);
        if (!(err < tol)) { std::printf("FAIL: %s\n", what); ok = false; }
    };

    // --- 1) solid volume ---
    {
        double solid = 0.0;
        for (double l : cut.lambda) solid += 1.0 - l;
        solid *= g.cellVolume();
        const double exact = 4.0 / 3.0 * pi * s.r * s.r * s.r;
        check("volume rel err", std::abs(solid - exact) / exact, 1e-9);
    }

    // --- 2) x-face aperture deficit vs cross-circle area ---
    {
        const int i = int(std::floor(s.cx / h)) + 2; // plane cutting the sphere
        const double d = i * h - s.cx;
        const double exact = pi * (s.r * s.r - d * d);
        double solid = 0.0;
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                solid += (1.0 - cut.ax[g.idx(i, j, k)]) * g.hy * g.hz;
        check("aperture rel err", std::abs(solid - exact) / exact, 1e-9);
    }

    // --- 3) sanitize ---
    const double lambdaMin = 1e-2, apertureMin = 1e-2;
    const auto st = sanitizeCut(g, cut, lambdaMin, apertureMin);
    std::printf("sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed,
                st.minLambda, st.minAperture);
    if (st.minLambda < lambdaMin || st.minAperture < apertureMin)
    {
        std::printf("FAIL: thresholds not enforced\n");
        ok = false;
    }
    const double tol = CutGeometry::openTol;
    for (int k = 0; k < n && ok; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const auto c = g.idx(i, j, k);
                const bool solidC = cut.lambda[c] <= 0.0;
                // open face must join two fluid cells
                if ((cut.ax[c] > tol && (solidC || cut.lambda[g.idx(i - 1, j, k)] <= 0.0))
                 || (cut.ay[c] > tol && (solidC || cut.lambda[g.idx(i, j - 1, k)] <= 0.0))
                 || (cut.az[c] > tol && (solidC || cut.lambda[g.idx(i, j, k - 1)] <= 0.0)))
                {
                    std::printf("FAIL: open face on solid cell at %d %d %d\n", i, j, k);
                    ok = false;
                }
                // fluid cell must keep at least one open face
                if (!solidC
                    && !(cut.ax[c] > tol || cut.ax[g.idx(i + 1, j, k)] > tol
                      || cut.ay[c] > tol || cut.ay[g.idx(i, j + 1, k)] > tol
                      || cut.az[c] > tol || cut.az[g.idx(i, j, k + 1)] > tol))
                {
                    std::printf("FAIL: isolated fluid cell at %d %d %d\n", i, j, k);
                    ok = false;
                }
            }
    if (!ok) return EXIT_FAILURE;

    // --- 4) projection ---
    auto U = VectorField::zeros(g);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] = d(rng); U.v[c] = d(rng); U.w[c] = d(rng);
    }
    maskSolidFaces(g, cut, U);

    CutPoissonSolver poisson(g, cut, 20000, 1e-12);
    auto scr = CutProjectionScratch::make(g);
    projectCut(g, cut, poisson, 0.01, U, scr);

    const double e = maxAbsDivergenceCut(g, cut, U);
    std::printf("max |apertured div| after projectCut: %.3e\n", e);
    return ok && e < 1e-8 ? EXIT_SUCCESS : EXIT_FAILURE;
}
