// Random solenoidal initial condition for forced HIT with a model spectrum
// E(k) ~ (k/kp)^4 exp(-2 (k/kp)^2), kp = 4 k0; discretely projected on the MAC grid.
#pragma once

#include "dns/field.hpp"
#include "dns/grid.hpp"

namespace dns {

#if DNS_WITH_FFTW
// Seeds U with zero-mean turbulence of rms velocity uPrime (per component 3D rms).
// Pre: periodic grid, uPrime > 0. Post: discretely divergence-free, zero mean.
void hitInitialCondition(const Grid& g, double uPrime, unsigned seed, VectorField& U);
#endif

} // namespace dns
