// Linear (Lundgren) forcing with fixed-eps controller (Carroll-Blanquart):
// f = A(t) (u - <u>) injects exactly epsTarget per unit mass at every instant.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"
#include "dns/operators.hpp"

#include <algorithm>

namespace dns {

// Per-step forcing coefficients; compute once per step, reuse for all stages.
struct ForcingState
{
    double A = 0.0;           // forcing rate [1/time]; 0 = forcing off
    Vec3 mean{0.0, 0.0, 0.0}; // mean velocity subtracted inside f
};

// A = eps / (2 max(k', kFloor)); the floor caps A during low-energy startup.
// Stationary state: <eps_visc> = epsTarget, u' -> (0.19 L epsTarget)^(1/3).
struct LinearForcing
{
    double epsTarget; // > 0, injected power per unit mass
    double kFloor;    // > 0, clamp; use kStationary / 5

    ForcingState update(const Grid& g, const VectorField& U) const
    {
        const double kPrime = fluctuationEnergy(g, U);
        const double A = epsTarget / (2.0 * std::max(kPrime, kFloor));
        return {A, meanVelocity(g, U)};
    }
};

} // namespace dns
