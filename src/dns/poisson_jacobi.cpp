#include "dns/poisson.hpp"

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

} // namespace dns
