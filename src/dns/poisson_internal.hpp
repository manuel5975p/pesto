// Implementation-only Poisson entry points. Not part of the public interface:
// the damped-Jacobi solver is a slow reference kept for the no-FFTW build and
// for cross-checking the spectral solve, never a solver a caller should select.
#pragma once

#include "dns/poisson.hpp"

namespace dns {

// Damped-Jacobi reference solver. iters/tol: stop on either.
std::unique_ptr<PoissonSolver> makeJacobiSolver(const Grid& g, int maxIters,
                                                double tol);

} // namespace dns
