// Cut-cell geometry + apertured projection.
//  1) 45-deg rotated box, edges through grid nodes => every cut cell halved.
//  2) apertures are consistent (closed inside solid, open in free stream).
//  3) projectCut drives the apertured divergence to zero on a noise field.
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

    // Center + half-diagonal + z-slab all on grid nodes: diamond edges have
    // slope +-1 and pass through nodes, so each cut cell is split corner-to-
    // corner (area 1/2). z-caps sit on nodes (no partial cells there).
    const RotatedCube cube{16 * h, 16 * h, 8 * h, 10 * h, 22 * h};
    const auto cut = CutGeometry::make(g, cube);

    // --- 1) halving ---
    int nCut = 0, nSolid = 0, nFluid = 0, bad = 0;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        const double l = cut.lambda[c];
        if (std::abs(l - 1.0) < 1e-12) ++nFluid;
        else if (std::abs(l) < 1e-12) ++nSolid;
        else if (std::abs(l - 0.5) < 1e-12) ++nCut;
        else { ++bad; if (bad <= 5) std::printf("  bad lambda=%.15g at %zu\n", l, c); }
    }
    std::printf("cells: fluid=%d solid=%d cut(0.5)=%d off=%d\n",
                nFluid, nSolid, nCut, bad);
    if (bad != 0 || nCut == 0 || nSolid == 0)
    {
        std::printf("FAIL: geometry not cleanly halved\n");
        return EXIT_FAILURE;
    }

    // --- 2) aperture sanity: face deep in solid closed, face far away open ---
    {
        const auto cIn = g.idx(16, 16, 16);       // diamond centre, in slab
        const auto cOut = g.idx(1, 1, 16);        // corner of box, free stream
        if (!(cut.ax[cIn] < 1e-12 && cut.ay[cIn] < 1e-12))
        {
            std::printf("FAIL: interior faces not closed (ax=%.3g ay=%.3g)\n",
                        cut.ax[cIn], cut.ay[cIn]);
            return EXIT_FAILURE;
        }
        if (!(cut.ax[cOut] > 1 - 1e-12 && cut.az[cOut] > 1 - 1e-12))
        {
            std::printf("FAIL: free-stream faces not open\n");
            return EXIT_FAILURE;
        }
    }

    // --- 3) projection ---
    auto U = VectorField::zeros(g);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] = d(rng); U.v[c] = d(rng); U.w[c] = d(rng);
    }
    maskSolidFaces(g, cut, U);

    auto ps = CutPoissonSolver::make(g, cut, 5000, 1e-12);
    auto& poisson = *ps;
    auto scr = CutProjectionScratch::make(g);
    projectCut(g, cut, poisson, 0.01, U, scr);

    const double e = maxAbsDivergenceCut(g, cut, U);
    std::printf("max |apertured div| after projectCut: %.3e\n", e);
    if (e >= 1e-8) return EXIT_FAILURE;

    // --- 4) a NaN residual is DIVERGENCE, not an instant solve ---
    // `NaN > target` is false, so without the isfinite guard the CG loop exits
    // at iteration 0 and returns 0 -- indistinguishable from a perfect
    // warm-started solve. It must report non-convergence (maxIters) instead.
    {
        const int maxIters = 50;
        auto pd = CutPoissonSolver::make(g, cut, maxIters, 1e-12);
        if (!pd) { std::printf("FAIL: solver make\n"); return EXIT_FAILURE; }
        Scalar rhs = zeros(g), p = zeros(g);
        for (std::size_t c = 0; c < g.size(); ++c)
            if (cut.lambda[c] > 0.0) { rhs[c] = std::nan(""); break; }
        const int it = pd->solve(rhs, p, /*warmStart=*/false);
        std::printf("CG on a NaN-seeded rhs returned %d (maxIters %d)\n",
                    it, maxIters);
        if (it != maxIters)
        {
            std::printf("FAIL: NaN residual reported as a converged solve\n");
            return EXIT_FAILURE;
        }
    }

    // An all-solid geometry is a user input, not a programmer error.
    {
        CutGeometry solid{zeros(g), zeros(g), zeros(g), zeros(g)};
        if (CutPoissonSolver::make(g, solid, 10, 1e-12).has_value())
        {
            std::printf("FAIL: all-solid geometry accepted\n");
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
