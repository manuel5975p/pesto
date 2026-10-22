#include "dns/timestepper.hpp"

#include "dns/projection.hpp"

#include "dns/operators.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns {

TimeStepper::TimeStepper(const Grid& g, StepParams p)
    : g_(g), p_(p),
      poisson_(makeDefaultSolver(g)),
      scratch_(ProjectionScratch::make(g)),
      k_(VectorField::zeros(g)),
      s1_(VectorField::zeros(g)),
      s2_(VectorField::zeros(g))
{
    assert(p_.nu >= 0);
    assert(p_.cfl > 0 && p_.cfl <= 1);
}

double TimeStepper::suggestDt(const VectorField& U, double dtMax) const
{
    const double conv = maxCourantPerDt(g_, U);
    // A NaN Courant number is a diverged field: max(NaN, diff) would drop it
    // and return dtMax. NaN out instead, so the caller's dt check fails.
    if (!(conv >= 0)) return std::numeric_limits<double>::quiet_NaN();
    const double hmin2 = std::pow(std::min({g_.hx, g_.hy, g_.hz}), 2);
    const double diff = p_.nu > 0 ? 6.0 * p_.nu / hmin2 : 0.0;
    const double lim = std::max(conv, diff);
    double dt = lim > 0 ? std::min(dtMax, p_.cfl / lim) : dtMax;
    if (forcing_.A > 0)
        dt = std::min(dt, 0.2 / forcing_.A); // resolve the forcing rate
    return dt;
}

void TimeStepper::rhs(const VectorField& U, VectorField& out)
{
    convection(g_, U, out); // out = -(U.grad)U
    if (p_.nu > 0)
    {
        laplacian(g_, U, k_);
        for (std::size_t c = 0; c < g_.size(); ++c)
        {
            out.u[c] += p_.nu * k_.u[c];
            out.v[c] += p_.nu * k_.v[c];
            out.w[c] += p_.nu * k_.w[c];
        }
    }
    if (forcing_.A > 0)
    {
        const auto [A, m] = forcing_;
        for (std::size_t c = 0; c < g_.size(); ++c)
        {
            out.u[c] += A * (U.u[c] - m.x);
            out.v[c] += A * (U.v[c] - m.y);
            out.w[c] += A * (U.w[c] - m.z);
        }
    }
}

void TimeStepper::step(VectorField& U, double dt)
{
    assert(dt > 0);
    const auto n = g_.size();

    auto axpy = [n](VectorField& y, double a, const VectorField& x,
                    double b, const VectorField& z) {
        for (std::size_t c = 0; c < n; ++c)
        {
            y.u[c] = a * x.u[c] + b * z.u[c];
            y.v[c] = a * x.v[c] + b * z.v[c];
            y.w[c] = a * x.w[c] + b * z.w[c];
        }
    };

    // stage 1: s1 = U + dt L(U)
    rhs(U, s2_);
    axpy(s1_, 1.0, U, 0.0, U);
    for (std::size_t c = 0; c < n; ++c)
    {
        s1_.u[c] += dt * s2_.u[c];
        s1_.v[c] += dt * s2_.v[c];
        s1_.w[c] += dt * s2_.w[c];
    }
    project(g_, *poisson_, dt, s1_, scratch_);

    // stage 2: s1 = 3/4 U + 1/4 (s1 + dt L(s1))
    rhs(s1_, s2_);
    for (std::size_t c = 0; c < n; ++c)
    {
        s1_.u[c] = 0.75 * U.u[c] + 0.25 * (s1_.u[c] + dt * s2_.u[c]);
        s1_.v[c] = 0.75 * U.v[c] + 0.25 * (s1_.v[c] + dt * s2_.v[c]);
        s1_.w[c] = 0.75 * U.w[c] + 0.25 * (s1_.w[c] + dt * s2_.w[c]);
    }
    project(g_, *poisson_, dt, s1_, scratch_);

    // stage 3: U = 1/3 U + 2/3 (s1 + dt L(s1))
    rhs(s1_, s2_);
    for (std::size_t c = 0; c < n; ++c)
    {
        U.u[c] = (1.0 / 3.0) * U.u[c] + (2.0 / 3.0) * (s1_.u[c] + dt * s2_.u[c]);
        U.v[c] = (1.0 / 3.0) * U.v[c] + (2.0 / 3.0) * (s1_.v[c] + dt * s2_.v[c]);
        U.w[c] = (1.0 / 3.0) * U.w[c] + (2.0 / 3.0) * (s1_.w[c] + dt * s2_.w[c]);
    }
    project(g_, *poisson_, dt, U, scratch_);
}


// --- Chorin projection (was projection.cpp; one TU with the stepper that is
// its only caller) --------------------------------------------------------
#include "dns/operators.hpp"

#include <cassert>



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
