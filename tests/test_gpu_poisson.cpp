// Device Poisson vs manufactured solution p = cos(x)cos(2y)cos(z) with the RHS
// built from the discrete Laplacian (so the solve should be exact up to fp32 FFT).
#include "gpu_test_util.hpp"

#include "dns/gpu/poisson.hpp"
#include "dns/operators.hpp"

#include <numbers>
#include <vector>

int main()
{
    using namespace dns;
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    const auto g = Grid::make(32, 32, 16, 2 * pi, 2 * pi, 2 * pi);

    // Manufactured p with zero mean; rhs = discrete FD Laplacian of p.
    Scalar pRef = zeros(g), rhs = zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy,
                             z = (k + 0.5) * g.hz;
                pRef[g.idx(i, j, k)] = std::cos(x) * std::cos(2 * y) * std::cos(z);
            }
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                rhs[c] =
                    (pRef[g.idx(i + 1, j, k)] - 2 * pRef[c] + pRef[g.idx(i - 1, j, k)])
                        / (g.hx * g.hx)
                  + (pRef[g.idx(i, j + 1, k)] - 2 * pRef[c] + pRef[g.idx(i, j - 1, k)])
                        / (g.hy * g.hy)
                  + (pRef[g.idx(i, j, k + 1)] - 2 * pRef[c] + pRef[g.idx(i, j, k - 1)])
                        / (g.hz * g.hz);
            }

    const auto gg = gpu::GpuGrid::from(g);
    auto solver = gpu::DevicePoissonSolver::make(gg);
    if (!solver)
    {
        std::fprintf(stderr, "FAIL: solver make: %s\n", hipGetErrorString(solver.error()));
        return 1;
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
    return test::checkClose("poisson", err, scale, 1e-5) ? 0 : 1;
}
