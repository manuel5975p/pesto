// 2D (nz == 1) device Poisson: periodic and NeumannBox against manufactured
// solutions built from the discrete operators, exactly as the 3D tests do.
// nz == 1 is the exact 2D restriction; the solver must handle it natively
// (2D FFT plan, z DCT pass skipped).
#include "gpu_test_util.hpp"

#include "dns/gpu/poisson.hpp"
#include "dns/operators.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace {

using namespace dns;

// Mirror (zero-flux) neighbor lookup on the interior box [1, n-2].
int mir(int i, int n) { return i < 1 ? 1 : (i > n - 2 ? n - 2 : i); }

bool periodicCase()
{
    using std::numbers::pi;
    const auto g = Grid::make(48, 32, 1, 2 * pi, 2 * pi, 1.0);

    Scalar pRef = zeros(g), rhs = zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            pRef[g.idx(i, j, 0)] =
                std::cos((i + 0.5) * g.hx) * std::cos(2 * (j + 0.5) * g.hy);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const auto c = g.idx(i, j, 0);
            rhs[c] = (pRef[g.idx(i + 1, j, 0)] - 2 * pRef[c] + pRef[g.idx(i - 1, j, 0)])
                         / (g.hx * g.hx)
                   + (pRef[g.idx(i, j + 1, 0)] - 2 * pRef[c] + pRef[g.idx(i, j - 1, 0)])
                         / (g.hy * g.hy);
        }

    auto solver = gpu::DevicePoissonSolver::make(gpu::GpuGrid::from(g));
    if (!solver)
    {
        std::fprintf(stderr, "FAIL: periodic 2d make: %s\n",
                     hipGetErrorString(solver.error()));
        return false;
    }
    auto dRhs = *gpu::DeviceScalar::make(g.size());
    auto dP = *gpu::DeviceScalar::make(g.size());
    std::vector<gpu::Real> tmp(rhs.begin(), rhs.end());
    (void)dRhs.upload(tmp.data());
    solver->solve(dRhs, dP);
    (void)dP.download(tmp.data());

    double err = 0.0, scale = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        err = std::max(err, std::abs(double(tmp[c]) - pRef[c]));
        scale = std::max(scale, std::abs(pRef[c]));
    }
    return test::checkClose("periodic 2d", err, scale, 1e-5);
}

bool neumannCase()
{
    using std::numbers::pi;
    const int mx = 32, my = 24;
    const auto g = Grid::make(mx + 2, my + 2, 1, 1.1 * (mx + 2) / mx,
                              0.9 * (my + 2) / my, 1.0);

    Scalar pRef = zeros(g), rhs = zeros(g);
    for (int j = 1; j <= my; ++j)
        for (int i = 1; i <= mx; ++i)
            pRef[g.idx(i, j, 0)] = std::cos(pi * 3 * (i - 0.5) / mx)
                                 * std::cos(pi * 2 * (j - 0.5) / my);
    for (int j = 1; j <= my; ++j)
        for (int i = 1; i <= mx; ++i)
        {
            const auto c = g.idx(i, j, 0);
            auto at = [&](int a, int b) {
                return pRef[g.idx(mir(a, g.nx), mir(b, g.ny), 0)];
            };
            rhs[c] = (at(i + 1, j) - 2 * pRef[c] + at(i - 1, j)) / (g.hx * g.hx)
                   + (at(i, j + 1) - 2 * pRef[c] + at(i, j - 1)) / (g.hy * g.hy);
        }

    auto solver = gpu::DevicePoissonSolver::make(gpu::GpuGrid::from(g),
                                                 gpu::PoissonBc::NeumannBox);
    if (!solver)
    {
        std::fprintf(stderr, "FAIL: neumann 2d make: %s\n",
                     hipGetErrorString(solver.error()));
        return false;
    }
    auto dRhs = *gpu::DeviceScalar::make(g.size());
    auto dP = *gpu::DeviceScalar::make(g.size());
    std::vector<gpu::Real> tmp(rhs.begin(), rhs.end());
    (void)dRhs.upload(tmp.data());
    solver->solve(dRhs, dP);
    (void)dP.download(tmp.data());

    double err = 0.0, scale = 0.0, ghostErr = 0.0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const auto c = g.idx(i, j, 0);
            if (i == 0 || i == g.nx - 1 || j == 0 || j == g.ny - 1)
                ghostErr = std::max(ghostErr, std::abs(double(tmp[c])));
            else
            {
                err = std::max(err, std::abs(double(tmp[c]) - pRef[c]));
                scale = std::max(scale, std::abs(pRef[c]));
            }
        }
    bool ok = test::checkClose("neumann 2d", err, scale, 1e-5);
    return test::checkClose("ghost ring zero", ghostErr, 1.0, 1e-12) && ok;
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    bool ok = periodicCase();
    ok = neumannCase() && ok;
    return ok ? 0 : 1;
}
