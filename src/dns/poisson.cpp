// Pressure Poisson solvers: the damped-Jacobi reference and, where FFTW is
// available, the exact spectral inversion of the 2nd-order FD Laplacian.
// One TU: the two are alternative implementations of one interface, and
// makeDefaultSolver picks between them at compile time.
#include "dns/poisson.hpp"

#include "dns/poisson_internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace dns {

namespace {

class JacobiSolver final : public PoissonSolver
{
public:
    JacobiSolver(const Grid& g, int maxIters, double tol)
        : g_(g), maxIters_(maxIters), tol_(tol), tmp_(zeros(g)) {}

    void solve(const Scalar& rhs, Scalar& p) override
    {
        const double ax = 1.0 / (g_.hx * g_.hx);
        const double ay = 1.0 / (g_.hy * g_.hy);
        const double az = 1.0 / (g_.hz * g_.hz);
        const double diag = 2.0 * (ax + ay + az);

        const double mean = std::accumulate(rhs.begin(), rhs.end(), 0.0) / double(g_.size());

        p.assign(g_.size(), 0.0);
        for (int it = 0; it < maxIters_; ++it)
        {
            double resMax = 0.0;
            for (int k = 0; k < g_.nz; ++k)
                for (int j = 0; j < g_.ny; ++j)
                    for (int i = 0; i < g_.nx; ++i)
                    {
                        const auto c = g_.idx(i, j, k);
                        const double nb =
                            ax * (p[g_.idx(i + 1, j, k)] + p[g_.idx(i - 1, j, k)])
                          + ay * (p[g_.idx(i, j + 1, k)] + p[g_.idx(i, j - 1, k)])
                          + az * (p[g_.idx(i, j, k + 1)] + p[g_.idx(i, j, k - 1)]);
                        const double f = rhs[c] - mean;
                        tmp_[c] = (nb - f) / diag;
                        resMax = std::max(resMax, std::abs(nb - diag * p[c] - f));
                    }
            std::swap(p, tmp_);
            if (resMax < tol_)
                break;
        }

        const double pMean = std::accumulate(p.begin(), p.end(), 0.0) / double(g_.size());
        for (auto& x : p) x -= pMean;
    }

private:
    Grid g_;
    int maxIters_;
    double tol_;
    Scalar tmp_;
};

} // namespace

std::unique_ptr<PoissonSolver> makeJacobiSolver(const Grid& g, int maxIters, double tol)
{
    return std::make_unique<JacobiSolver>(g, maxIters, tol);
}

#if !DNS_WITH_FFTW
std::unique_ptr<PoissonSolver> makeJacobiFallback(const Grid& g)
{
    return makeJacobiSolver(g, 20000, 1e-10);
}
#endif

} // namespace dns

#if DNS_WITH_FFTW

#include <fftw3.h>

#include <cmath>
#include <numbers>

namespace dns {

namespace {

// Solves the discrete 2nd-order FD Laplacian exactly in Fourier space:
// eigenvalue per mode m on axis n:  (2 cos(2 pi m / n) - 2) / h^2.
class FftSolver final : public PoissonSolver
{
public:
    explicit FftSolver(const Grid& g)
        : g_(g),
          nc_(std::size_t(g.nz) * g.ny * (g.nx / 2 + 1)),
          real_(fftw_alloc_real(g.size())),
          cplx_(fftw_alloc_complex(nc_))
    {
        // storage is i-fastest => row-major dims (nz, ny, nx)
        fwd_ = fftw_plan_dft_r2c_3d(g.nz, g.ny, g.nx, real_, cplx_, FFTW_MEASURE);
        bwd_ = fftw_plan_dft_c2r_3d(g.nz, g.ny, g.nx, cplx_, real_, FFTW_MEASURE);
    }

    ~FftSolver() override
    {
        fftw_destroy_plan(fwd_);
        fftw_destroy_plan(bwd_);
        fftw_free(real_);
        fftw_free(cplx_);
    }

    FftSolver(const FftSolver&) = delete;
    FftSolver& operator=(const FftSolver&) = delete;

    void solve(const Scalar& rhs, Scalar& p) override
    {
        using std::numbers::pi;
        std::copy(rhs.begin(), rhs.end(), real_);
        fftw_execute(fwd_);

        const int nxh = g_.nx / 2 + 1;
        const double norm = 1.0 / double(g_.size());
        for (int kz = 0; kz < g_.nz; ++kz)
            for (int ky = 0; ky < g_.ny; ++ky)
                for (int kx = 0; kx < nxh; ++kx)
                {
                    const std::size_t c = std::size_t(kx) + std::size_t(nxh) * (std::size_t(ky) + std::size_t(g_.ny) * kz);
                    const double lam =
                        (2.0 * std::cos(2.0 * pi * kx / g_.nx) - 2.0) / (g_.hx * g_.hx)
                      + (2.0 * std::cos(2.0 * pi * ky / g_.ny) - 2.0) / (g_.hy * g_.hy)
                      + (2.0 * std::cos(2.0 * pi * kz / g_.nz) - 2.0) / (g_.hz * g_.hz);
                    const double s = (lam == 0.0) ? 0.0 : norm / lam; // zero mode -> zero mean
                    cplx_[c][0] *= s;
                    cplx_[c][1] *= s;
                }

        fftw_execute(bwd_);
        p.assign(real_, real_ + g_.size());
    }

private:
    Grid g_;
    std::size_t nc_;
    double* real_;
    fftw_complex* cplx_;
    fftw_plan fwd_{};
    fftw_plan bwd_{};
};

} // namespace

std::unique_ptr<PoissonSolver> makeFftSolver(const Grid& g)
{
    return std::make_unique<FftSolver>(g);
}

} // namespace dns
#endif // DNS_WITH_FFTW
