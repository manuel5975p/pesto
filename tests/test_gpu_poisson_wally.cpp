// PeriodicXZWallY Poisson: the wall-bounded box (periodic x and z, mirror
// Neumann on the two y wall faces). Manufactured solution, discrete rhs, and
// the solve must reproduce it to fp32 -- the y direction is inverted by an
// exact tridiagonal solve, so there is no transform error to hide behind.
//
// Three properties, each of which has a distinct way of being wrong:
//   1. Reproduction of an arbitrary (non-eigenfunction) field, up to the
//      additive constant a pure-Neumann direction leaves free.
//   2. The 2D restriction nz == 1 -- the z plan is skipped entirely there.
//   3. The singular (kx, kz) = (0, 0) mode: a rhs whose x/z mean is a
//      non-trivial function of y is exactly the case the pin-and-degauge
//      branch handles, and getting it wrong is invisible in 1 and 2.
#include "gpu_test_util.hpp"

#include "dns/gpu/poisson.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace {

using namespace dns;

// Mirror across the two y wall faces; periodic wrap in x and z.
int wrap(int i, int n) { return i < 0 ? i + n : (i >= n ? i - n : i); }
int mirY(int j, int ny) { return j < 1 ? 1 : (j > ny - 2 ? ny - 2 : j); }

// Discrete Laplacian with those boundary rules, on the interior y rows.
Scalar discreteLaplacian(const Grid& g, const Scalar& p)
{
    Scalar rhs = zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 1; j <= g.ny - 2; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                auto at = [&](int a, int b, int d) {
                    return p[g.idx(wrap(a, g.nx), mirY(b, g.ny), wrap(d, g.nz))];
                };
                rhs[c] = (at(i + 1, j, k) - 2 * p[c] + at(i - 1, j, k))
                             / (g.hx * g.hx)
                       + (at(i, j + 1, k) - 2 * p[c] + at(i, j - 1, k))
                             / (g.hy * g.hy)
                       + (g.nz > 1 ? (at(i, j, k + 1) - 2 * p[c]
                                      + at(i, j, k - 1)) / (g.hz * g.hz)
                                   : 0.0);
            }
    return rhs;
}

// Solve, then compare on the interior with the free constant removed. Also
// reports the largest value left in the ghost rows, which must be zero.
bool roundTrip(const char* what, const Grid& g, const Scalar& pRef,
               double rtol = 1e-5)
{
    const Scalar rhs = discreteLaplacian(g, pRef);
    const auto gg = gpu::GpuGrid::from(g);
    auto dRhs = *gpu::DeviceScalar::make(g.size());
    auto dP = *gpu::DeviceScalar::make(g.size());
    std::vector<gpu::Real> tmp(rhs.begin(), rhs.end());
    (void)dRhs.upload(tmp.data());

    auto solver =
        gpu::DevicePoissonSolver::make(gg, gpu::PoissonBc::PeriodicXZWallY);
    if (!solver)
    {
        std::fprintf(stderr, "FAIL: %s solver make: %s\n", what,
                     hipGetErrorString(solver.error()));
        return false;
    }
    solver->solve(dRhs, dP);
    (void)dP.download(tmp.data());

    // A pure-Neumann direction fixes the solution only up to a constant.
    double mGot = 0, mRef = 0;
    std::size_t n = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 1; j <= g.ny - 2; ++j)
            for (int i = 0; i < g.nx; ++i, ++n)
            {
                mGot += double(tmp[g.idx(i, j, k)]);
                mRef += pRef[g.idx(i, j, k)];
            }
    mGot /= double(n);
    mRef /= double(n);

    double err = 0, scale = 0, ghostErr = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                if (j == 0 || j == g.ny - 1)
                    ghostErr = std::max(ghostErr, std::abs(double(tmp[c])));
                else
                {
                    err = std::max(err,
                                   std::abs((double(tmp[c]) - mGot)
                                            - (pRef[c] - mRef)));
                    scale = std::max(scale, std::abs(pRef[c] - mRef));
                }
            }
    bool ok = test::checkClose(what, err, scale, rtol);
    ok = test::checkClose("  ghost rows zero", ghostErr, 1.0, 1e-12) && ok;
    return ok;
}

} // namespace

int main()
{
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    bool ok = true;

    // 1. 3D, a field with no special structure: several x/z harmonics against
    //    several mirror-y ones, so every mode of the solve is exercised.
    {
        const int nx = 32, my = 24, nz = 16;
        const auto g = Grid::make(nx, my + 2, nz, 1.3, 0.9 * (my + 2) / my, 1.1);
        Scalar p = zeros(g);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 1; j <= my; ++j)
                for (int i = 0; i < g.nx; ++i)
                    p[g.idx(i, j, k)] =
                        std::cos(2 * pi * 3 * i / nx) * std::cos(pi * 2 * (j - 0.5) / my)
                        + 0.7 * std::sin(2 * pi * 1 * i / nx)
                              * std::cos(pi * 5 * (j - 0.5) / my)
                              * std::cos(2 * pi * 2 * k / nz)
                        + 0.4 * std::cos(2 * pi * 4 * k / nz)
                              * std::cos(pi * 1 * (j - 0.5) / my);
        ok = roundTrip("3D periodic-xz wall-y", g, p) && ok;
    }

    // 2. The 2D restriction: nz == 1 skips the z transform entirely.
    {
        const int nx = 64, my = 32;
        const auto g = Grid::make(nx, my + 2, 1, 2.0, 1.0 * (my + 2) / my, 1.0);
        Scalar p = zeros(g);
        for (int j = 1; j <= my; ++j)
            for (int i = 0; i < g.nx; ++i)
                p[g.idx(i, j, 0)] =
                    std::cos(2 * pi * 5 * i / nx) * std::cos(pi * 3 * (j - 0.5) / my)
                    + 0.3 * std::sin(2 * pi * 2 * i / nx)
                          * std::cos(pi * 1 * (j - 0.5) / my);
        ok = roundTrip("2D (nz == 1)", g, p) && ok;
    }

    // 3. The singular mode alone: p depends on y ONLY, so the entire solution
    //    lives in (kx, kz) = (0, 0) -- the pinned, de-gauged branch. If that
    //    branch is wrong this is the only test that sees it.
    {
        const int nx = 16, my = 40, nz = 8;
        const auto g = Grid::make(nx, my + 2, nz, 1.0, 1.0 * (my + 2) / my, 1.0);
        Scalar p = zeros(g);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 1; j <= my; ++j)
                for (int i = 0; i < g.nx; ++i)
                    p[g.idx(i, j, k)] = std::cos(pi * 2 * (j - 0.5) / my)
                                      + 0.5 * std::cos(pi * 7 * (j - 0.5) / my);
        ok = roundTrip("y-only (singular kx=kz=0 mode)", g, p) && ok;
    }

    // 4. A y-independent field: the solution is a constant, i.e. exactly the
    //    part the gauge fixes away. The de-gauging must not leave a ramp.
    {
        const int nx = 32, my = 16, nz = 8;
        const auto g = Grid::make(nx, my + 2, nz, 1.0, 1.0 * (my + 2) / my, 1.0);
        Scalar p = zeros(g);
        for (int k = 0; k < g.nz; ++k)
            for (int j = 1; j <= my; ++j)
                for (int i = 0; i < g.nx; ++i)
                    p[g.idx(i, j, k)] = std::cos(2 * pi * 2 * i / nx)
                                      * std::cos(2 * pi * 1 * k / nz);
        ok = roundTrip("y-independent", g, p) && ok;
    }

    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
