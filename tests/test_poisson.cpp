// Poisson solvers vs manufactured solution p = cos(x)cos(2y)[cos(z)], run on
// a 3D grid and on the degenerate nz == 1 grid (the exact 2D restriction).
#include "dns/poisson_internal.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using namespace dns;
using std::numbers::pi;

namespace {

// rhs of the *discrete* 2nd-order Laplacian applied to p, so any exact
// solver must reproduce p to roundoff.
void discreteLaplacian(const Grid& g, const Scalar& p, Scalar& out)
{
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                out[c] = (p[g.idx(i + 1, j, k)] - 2 * p[c] + p[g.idx(i - 1, j, k)]) / (g.hx * g.hx)
                       + (p[g.idx(i, j + 1, k)] - 2 * p[c] + p[g.idx(i, j - 1, k)]) / (g.hy * g.hy)
                       + (p[g.idx(i, j, k + 1)] - 2 * p[c] + p[g.idx(i, j, k - 1)]) / (g.hz * g.hz);
            }
}

double maxErr(const Grid& g, const Scalar& a, const Scalar& b)
{
    double e = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c)
        e = std::max(e, std::abs(a[c] - b[c]));
    return e;
}

int check(PoissonSolver& s, const Grid& g, const Scalar& ref, const Scalar& rhs, double tol,
          const char* name)
{
    Scalar p = zeros(g);
    s.solve(rhs, p);
    const double e = maxErr(g, p, ref);
    std::printf("%-12s err=%.3e tol=%.0e\n", name, e, tol);
    return e < tol ? 0 : 1;
}

int runCase(const Grid& g, const char* tag)
{
    Scalar ref = zeros(g), rhs = zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                ref[g.idx(i, j, k)] = std::cos((i + 0.5) * g.hx)
                                    * std::cos(2 * (j + 0.5) * g.hy)
                                    * (g.nz > 1 ? std::cos((k + 0.5) * g.hz) : 1.0);
    discreteLaplacian(g, ref, rhs);

    int fails = 0;
    char name[32];
#if DNS_WITH_FFTW
    {
        auto s = makeFftSolver(g);
        std::snprintf(name, sizeof name, "fft %s", tag);
        fails += check(*s, g, ref, rhs, 1e-11, name);
    }
#endif
    {
        auto s = makeJacobiSolver(g, 40000, 1e-11);
        std::snprintf(name, sizeof name, "jacobi %s", tag);
        fails += check(*s, g, ref, rhs, 1e-5, name);
    }
    return fails;
}

} // namespace

int main()
{
    int fails = runCase(Grid::make(32, 32, 16, 2 * pi, 2 * pi, 2 * pi), "3d");
    fails += runCase(Grid::make(48, 32, 1, 2 * pi, 2 * pi, 1.0), "2d");
    return fails == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
