// Initial states of `dns flame` (docs/chem.md): the oracle flame laid onto the
// grid. Geometry layer (reads the validated Case; no GPU, no filesystem).
#pragma once

#include "dns/chem.hpp"
#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/grid.hpp"
#include "dns/tfles.hpp"
#include "dns/transport.hpp"
#include "dns/turb_ic.hpp"

#include <array>
#include <functional>
#include <span>
#include <vector>

namespace dns {

// One point of the low-Mach premixed state at t = T/Tu (box units, below):
// premixedState's expressions, shared with the FV IC (dns/fvflame).
struct PremixedPoint
{
    double rho, u, E, Y;
};
PremixedPoint premixedPoint(double gamma, double Tu, double sigma, double m, double t);

// The low-Mach premixed state of a temperature profile t(x, y) = T/Tu on the
// cell centres (x along the grid's x axis, k-uniform), in box units
// (rho_u = R = 1, Tu = p0): the isobaric density rho0 = Tu/T, then the
// momentum balance p = p0 - m^2 (1/rho0 - 1) and rho = p/T, u = m/rho, E =
// p/(gamma-1) + rho u^2/2, Y = (sigma - t)/(sigma - 1) (Shvab-Zeldovich,
// exact for Le = 1 and unity Y_u), rhoY = rho Y. A y-dependent t is a
// wrinkled front laid row by row (x momentum only: the start is not the
// curved flame's solution and relaxes acoustically).
// Pre: Tu > 0, sigma > 1, t(x, y) in [1, sigma]. q and rhoY are resized.
void premixedState(const Grid& g, double gamma, double Tu, double sigma,
                   double m, const std::function<double(double, double)>& t,
                   CompState& q, std::vector<double>& rhoY);

// Front displacement of `--wrinkle A --wrinkle-modes N` at height y of a box
// Ly tall: sum over k = 1..N of (A/sqrt(N)) cos(2 pi k y/Ly + phi_k), phi_k
// = 2 pi frac(k^2 golden) (deterministic, no RNG, spread along the front).
// A = 0 gives exactly 0.
// Pre: Ly > 0, N >= 1.
double flameWrinkle(double A, int N, double y, double Ly);

// The `dns flame` IC: the oracle flame (Case::flameEig's table) with its mid
// temperature at x = flameXf0 + flameWrinkle(y), mass flux m = flameUIn.
// --tfles F > 1: the profile at (x - x_f)/F, the thickened flame (bunsenIC
// likewise on the cone distance). Pre: c.sub == Flame.
void flameIC(const Case& c, CompState& q, std::vector<double>& rhoY);

// The --equivalence IC (model C2): flameIC's state (the premixedEigen2
// profile), its progress s_p = rhoY/rho re-read as the two reactants: Y_O =
// Y_O1 s_p, Y_F = Y_F,rem + Y_O/s (Shvab-Zeldovich, Le = 1); a lean
// --mix-phi premix Y_F = Y_F1 s_p, Y_O = Y_O,rem + s Y_F (its Y_F1, Y_O1,
// burnt Tb). rhoY[0] = rho Y_F, rhoY[1] = rho Y_O (resized).
// Pre: c.flamePremix2.
void flameIC2(const Case& c, CompState& q, std::vector<double> (&rhoY)[2]);

// flameIC2's progress -> reactants rule on its own (shared with the low-Mach
// IC): rhoY[1] = rho s_p on entry; on return rhoY[0] = rho Y_F, rhoY[1] = rho
// Y_O of the burnt premix (rich: Y_O = Y_O1 s_p, Y_F = Y_F,rem + Y_O/s; lean:
// Y_F = Y_F1 s_p, Y_O = Y_O,rem + s Y_F). Pre: c.flamePremix2, rho and rhoY[1]
// of one size.
void premix2Reactants(const Case& c, const std::vector<double>& rho,
                      std::vector<double> (&rhoY)[2]);

// The `dns flame --mode bunsen` IC: the oracle profile on the signed distance
// to the ideal cone (bunsenConeDistance), t_f; blended by the inflow
// profile's weights J, P, C of y: T/Tu = t_f J + tPilot P + tCo C, Y =
// Y(t_f) J. Uniform p = p0, v = 0, and each row's inflow mass flux
// carried through: u = u_in(y) t / t_in(y) (x momentum only: the start
// relaxes acoustically and convectively). nz > 1: extruded over z (the slot
// is z-invariant). Pre: c.flameBunsen.
void bunsenIC(const Case& c, CompState& q, std::vector<double>& rhoY);

// The `dns flame --mode diffusion` IC: the inflow row (diffusionInflow, the
// Burke-Schumann sheet of the inflow mixture fraction) extruded along x --
// a burning sheet from the exit plane to the outflow, which the run closes
// into its tip within about one flow-through. p = p0 = T0, rho = 1/t, the
// row's inflow u. rhoY[0] = rho Y_F, rhoY[1] = rho Y_O (resized).
// --ignition L (c.flameIgnition > 0): Z(y) w(x), w = (1 - tanh((x - L)/edge))
// /2, laid as its Burke-Schumann state (a closed sheet L tall), u from the
// row's to the coflow's. --mode candle with g > 0: p = p_a(x) on every cell
// (candleAmbientP; horizontally uniform), rho = p_a/(R T); --quiescent: the
// oxidizer at rest everywhere. 0 / g 0 = the extruded row bit for bit.
// Pre: c.flameDiffusion.
void diffusionIC(const Case& c, CompState& q, std::vector<double> (&rhoY)[2]);

// The frozen turbulence box of `--turb` (dns/turb_ic.hpp): u' = flameTurb,
// l_t = flameTurbL, the periodic box flameTurbNbx x ny x nz cells of the run's
// h (L_bx = L_z), seeded by --seed. Pre: c.flameTurb > 0.
TurbSpec flameTurbSpec(const Case& c);

// The `--turb` IC: the fresh gas carries the box's continuation of the
// inflow sweep (the box at x_b = x, i.e. cell i reads box column i mod nbx),
// windowed off ahead of the flame: u' w(x), w = (1 - tanh((x - (x_f0 - 1.5
// l_t))/(0.25 l_t)))/2. m += rho u' w, E += the kinetic-energy change (p kept).
// Pre: c.flameTurb > 0, box from turbBox(flameTurbSpec(c)).
void addTurbulence(const Case& c, const std::array<std::vector<double>, 3>& box,
                   CompState& q);

// The `--mode bunsen --turb` IC: the box (cell i reads box column i mod nbx,
// as addTurbulence) weighted by the cell's fresh fraction Y = rhoY/rho -- the
// jet core and the unburnt cone carry it, the burnt coflow and products do
// not. m += rho Y u', E += the kinetic-energy change (p kept).
// Pre: c.flameBunsen, c.flameTurb > 0, box from turbBox(flameTurbSpec(c)),
// rhoY from bunsenIC.
void addBunsenTurbulence(const Case& c,
                         const std::array<std::vector<double>, 3>& box,
                         const std::vector<double>& rhoY, CompState& q);

// The `dns flame --mode candle --wick` IC (dns/wick.hpp): the hydrostatic
// ambient column p_a(x) on every cell at rest, and around the wick a closed
// Burke-Schumann sheet: Z = e^(-d/kWickIcDecay) w(x), d the distance to the
// wick box, w = (1 - tanh((x - (x_tip + kWickIcAbove))/0.5))/2, Z = 0 below
// the wax top; rho = p_a/(R T). Solid cells: ambient air at rest with EVERY
// rho Y = 0 (the walled kernel never moves them). rhoY[0] = rho Y_F, [1] =
// rho Y_O, [2] = rho Y_s = 0 (resized; [2] empty without soot).
// Pre: c.flameWick.
inline constexpr double kWickIcDecay = 0.5;
inline constexpr double kWickIcAbove = 4;
void wickIC(const Case& c, CompState& q, std::vector<double> (&rhoY)[3]);

// The premix whose products leave: --equivalence's burnt one at --mix-phi (C2, C2b),
// else the case's; sigma its expansion ratio (C0: the flame model's).
struct FlameBurnt
{
    PartialPremixModel burn;
    double sigma = 1;
};
FlameBurnt flameBurnt(const Case& c);

// --tfles F (structured and FV, both physics): the thickened flame's setup in box
// units (mu = Pr, rho_u = 1, Tu = c.flameTu), the sensor's radius `dilate` cells.
// Pre: c.flameTfles > 0.
TflesSetup flameTflesSetup(const Case& c, double sigma, int dilate);

// The flame's gas in box units at T (Tu, T0): mu = Pr at Tref = T, so lambda = mu cp /
// Pr = cp (delta_D = 1); the case's viscosity law.
Transport flameGas(const Case& c, double T);

// The compressible flame's x-lo inflow rows, row j + ny k (NSCBC Prof rows): velocity,
// jet weight, the density target (rho; 1/(R T) under --nscbc-tinflow), rho, fuel and
// oxidizer mass fractions (yO empty: one reactant).
struct FlameInflowRows
{
    std::vector<double> u, jet, target, rho, yF, yO;
};

// --mode bunsen: bunsenInflowAt per row (slot or round nozzle), p0 = Tu. Pre: c.flameBunsen.
FlameInflowRows bunsenInflowRows(const Case& c);

// --mode diffusion | candle: the Burke-Schumann sheet of Z(y) (quiescentInflow with
// --quiescent); under gravity gx > 0 rho = pIn / T (the ambient column at h/2), else
// p0 = T0. Pre: c.flameDiffusion.
FlameInflowRows diffusionInflowRows(const Case& c, double gx, double pIn);

// --wick: the coflow air at rest on the ambient column, rho = pIn / T0 (rho empty).
// Pre: c.flameWick.
FlameInflowRows wickInflowRows(const Case& c, double pIn);

// --mode bunsen --turb: the inflow sees the box times the jet weight of its row (cell i
// of row r reads box column i mod nbx: q / nbx is the row), the coflow stays laminar.
// Pre: box from turbBox(flameTurbSpec(c)), rows nbx wide, jet one weight per row.
void bunsenTurbJetWeight(std::array<std::vector<double>, 3>& box, std::span<const double> jet, int nbx);

} // namespace dns
