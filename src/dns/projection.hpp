// Chorin projection: make a face velocity field discretely divergence-free.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"
#include "dns/poisson.hpp"

namespace dns {

// Workspace to avoid per-step allocation.
struct ProjectionScratch
{
    Scalar div, phi;
    static ProjectionScratch make(const Grid& g) { return {zeros(g), zeros(g)}; }
};

// Projects U in place: solve lap(phi) = div(U)/dt, U -= dt grad(phi).
// Post: divergence(U) ~ 0 (to solver tolerance). phi*dt is the pressure increment.
void project(const Grid& g, PoissonSolver& poisson, double dt, VectorField& U,
             ProjectionScratch& s);

} // namespace dns
