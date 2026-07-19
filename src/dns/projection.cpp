#include "dns/projection.hpp"

#include "dns/operators.hpp"

#include <cassert>

namespace dns {

void project(const Grid& g, PoissonSolver& poisson, double dt, VectorField& U,
             ProjectionScratch& s)
{
    assert(dt > 0);
    divergence(g, U, s.div);
    for (auto& x : s.div) x /= dt;
    poisson.solve(s.div, s.phi);
    subtractGradient(g, s.phi, dt, U);
}

} // namespace dns
