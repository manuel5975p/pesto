// SSP-RK3 fractional-step integrator for incompressible Navier-Stokes.
#pragma once

#include "dns/field.hpp"
#include "dns/forcing.hpp"
#include "dns/grid.hpp"
#include "dns/poisson.hpp"
#include "dns/projection.hpp"

namespace dns {

struct StepParams
{
    double nu;   // kinematic viscosity, >= 0 (0 = incompressible Euler)
    double cfl;  // Courant target for suggestDt, in (0, 1]
};

class TimeStepper
{
public:
    TimeStepper(const Grid& g, StepParams p);

    // Forcing coefficients used by rhs until the next call; {} disables.
    void setForcing(ForcingState f) { forcing_ = f; }

    // Stable dt from convective/diffusive/forcing limits; U nonzero or dtMax used.
    double suggestDt(const VectorField& U, double dtMax) const;

    // Advance U by dt (SSP-RK3, projection after each stage). Pre: dt > 0.
    void step(VectorField& U, double dt);

private:
    void rhs(const VectorField& U, VectorField& out);

    Grid g_;
    StepParams p_;
    ForcingState forcing_{};
    std::unique_ptr<PoissonSolver> poisson_;
    ProjectionScratch scratch_;
    VectorField k_, s1_, s2_;
};

} // namespace dns
