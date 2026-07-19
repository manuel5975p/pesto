// NeumannBox Poisson: manufactured interior solution p = prod cos(pi m s) with
// zero normal gradient at the box faces, rhs = discrete mirror Laplacian. The
// DCT solve must reproduce p exactly up to fp32; the PCG path on a fully open
// interior must converge in O(1) iterations since the preconditioner is exact.
#include "gpu_test_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/poisson.hpp"
#include "dns/operators.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace {

using namespace dns;

bool ghost(const Grid& g, int i, int j, int k)
{
    return i == 0 || i == g.nx - 1 || j == 0 || j == g.ny - 1 || k == 0
        || k == g.nz - 1;
}

// Mirror (zero-flux) neighbor lookup on the interior box [1, n-2].
int mir(int i, int n) { return i < 1 ? 1 : (i > n - 2 ? n - 2 : i); }

} // namespace

int main()
{
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    const int mx = 32, my = 24, mz = 8;
    const auto g = Grid::make(mx + 2, my + 2, mz + 2, 1.1 * (mx + 2) / mx,
                              0.9 * (my + 2) / my, 1.0 * (mz + 2) / mz);

    // Neumann eigenfunctions of the cell-centered mirror Laplacian.
    Scalar pRef = zeros(g), rhs = zeros(g);
    for (int k = 1; k <= mz; ++k)
        for (int j = 1; j <= my; ++j)
            for (int i = 1; i <= mx; ++i)
                pRef[g.idx(i, j, k)] = std::cos(pi * 3 * (i - 0.5) / mx)
                                     * std::cos(pi * 2 * (j - 0.5) / my)
                                     * std::cos(pi * 1 * (k - 0.5) / mz);
    for (int k = 1; k <= mz; ++k)
        for (int j = 1; j <= my; ++j)
            for (int i = 1; i <= mx; ++i)
            {
                const auto c = g.idx(i, j, k);
                auto at = [&](int a, int b, int d) {
                    return pRef[g.idx(mir(a, g.nx), mir(b, g.ny), mir(d, g.nz))];
                };
                rhs[c] = (at(i + 1, j, k) - 2 * pRef[c] + at(i - 1, j, k)) / (g.hx * g.hx)
                       + (at(i, j + 1, k) - 2 * pRef[c] + at(i, j - 1, k)) / (g.hy * g.hy)
                       + (at(i, j, k + 1) - 2 * pRef[c] + at(i, j, k - 1)) / (g.hz * g.hz);
            }

    const auto gg = gpu::GpuGrid::from(g);
    auto dRhs = *gpu::DeviceScalar::make(g.size());
    auto dP = *gpu::DeviceScalar::make(g.size());
    std::vector<gpu::Real> tmp(rhs.begin(), rhs.end());
    (void)dRhs.upload(tmp.data());

    // Direct DCT solve.
    auto solver = gpu::DevicePoissonSolver::make(gg, gpu::PoissonBc::NeumannBox);
    if (!solver)
    {
        std::fprintf(stderr, "FAIL: solver make: %s\n",
                     hipGetErrorString(solver.error()));
        return 1;
    }
    solver->solve(dRhs, dP);
    (void)dP.download(tmp.data());

    double err = 0.0, scale = 0.0, ghostErr = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                if (ghost(g, i, j, k))
                    ghostErr = std::max(ghostErr, std::abs(double(tmp[c])));
                else
                {
                    err = std::max(err, std::abs(double(tmp[c]) - pRef[c]));
                    scale = std::max(scale, std::abs(pRef[c]));
                }
            }
    bool ok = test::checkClose("neumann dct", err, scale, 1e-5);
    ok = test::checkClose("ghost ring zero", ghostErr, 1.0, 1e-12) && ok;

    // Full PCG path on an all-open interior (no body): exact preconditioner,
    // convergence in a couple of iterations.
    CutGeometry cut{Scalar(g.size(), 1.0), Scalar(g.size(), 1.0),
                    Scalar(g.size(), 1.0), Scalar(g.size(), 1.0)};
    closeBoxBoundary(g, cut);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    if (!dCut)
    {
        std::fprintf(stderr, "FAIL: cut upload\n");
        return 1;
    }
    auto pcg = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 100, 1e-6,
                                                 gpu::PoissonBc::NeumannBox);
    if (!pcg)
    {
        std::fprintf(stderr, "FAIL: pcg make\n");
        return 1;
    }
    // The PCG operator is the apertured mirror Laplacian == the rhs built above.
    const int it = pcg->solve(dRhs, dP);
    (void)dP.download(tmp.data());
    double errPcg = 0.0;
    for (int k = 1; k <= mz; ++k)
        for (int j = 1; j <= my; ++j)
            for (int i = 1; i <= mx; ++i)
            {
                const auto c = g.idx(i, j, k);
                errPcg = std::max(errPcg, std::abs(double(tmp[c]) - pRef[c]));
            }
    std::printf("pcg iters: %d\n", it);
    ok = test::checkClose("neumann pcg", errPcg, scale, 1e-4) && ok;
    if (it > 5)
    {
        std::fprintf(stderr, "FAIL: exact preconditioner took %d iters\n", it);
        ok = false;
    }
    return ok ? 0 : 1;
}
