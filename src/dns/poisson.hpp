// Pressure Poisson solvers: lap(p) = rhs on the periodic cell-centre grid.
// The GPU backend slots in behind this same interface.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"

#include <memory>

namespace dns {

class PoissonSolver
{
public:
    virtual ~PoissonSolver() = default;

    // Solve lap(p) = rhs; rhs mean is projected out (periodic compatibility).
    // p is overwritten; result has zero mean.
    virtual void solve(const Scalar& rhs, Scalar& p) = 0;
};

#if DNS_WITH_FFTW
// Exact (to roundoff) spectral solve of the 2nd-order FD Laplacian, O(N log N).
std::unique_ptr<PoissonSolver> makeFftSolver(const Grid& g);
#else
// The damped-Jacobi reference at the settings makeDefaultSolver needs. The
// solver itself is implementation-only (poisson_internal.hpp).
std::unique_ptr<PoissonSolver> makeJacobiFallback(const Grid& g);
#endif

// Best available solver for this build.
inline std::unique_ptr<PoissonSolver> makeDefaultSolver(const Grid& g)
{
#if DNS_WITH_FFTW
    return makeFftSolver(g);
#else
    return makeJacobiFallback(g);
#endif
}

} // namespace dns
