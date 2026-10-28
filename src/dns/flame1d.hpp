// Initial states of `dns flame` (docs/chem.md): the oracle flame laid onto the
// grid. Geometry layer (reads the validated Case; no GPU, no filesystem).
#pragma once

#include "dns/chem.hpp"
#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/grid.hpp"

#include <functional>
#include <vector>

namespace dns {

// The low-Mach premixed state of a temperature profile t(x) = T/Tu on the
// cell centres (x along the grid's x axis, j/k-uniform), in box units
// (rho_u = R = 1, Tu = p0): the isobaric density rho0 = Tu/T, then the
// momentum balance p = p0 - m^2 (1/rho0 - 1) and rho = p/T, u = m/rho, E =
// p/(gamma-1) + rho u^2/2, Y = (sigma - t)/(sigma - 1) (Shvab-Zeldovich,
// exact for Le = 1 and unity Y_u), rhoY = rho Y.
// Pre: Tu > 0, sigma > 1, t(x) in [1, sigma]. q and rhoY are resized.
void premixedState(const Grid& g, double gamma, double Tu, double sigma,
                   double m, const std::function<double(double)>& t,
                   CompState& q, std::vector<double>& rhoY);

// The `dns flame` IC: the oracle flame (Case::flameEig's table) with its mid
// temperature at x = flameXf0, mass flux m = flameUIn. Pre: c.sub == Flame.
void flameIC(const Case& c, CompState& q, std::vector<double>& rhoY);

} // namespace dns
