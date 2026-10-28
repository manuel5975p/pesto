// The non-premixed (diffusion) flame of `dns flame --mode diffusion`
// (docs/chem.md): the two-reactant single-step model F + s O -> (1+s) P, its
// coupling functions, the Burke-Schumann flame-sheet state, the inflow
// profile of the slot jet, and the jet measurement the runner and the tests
// read. Pure host; no GPU, no filesystem (config layer).
//
// Model C1: calorically perfect, gamma 1.4 fixed, EQUAL molecular weights
// (R, cp composition-free), Le = 1, both streams at T0. Y_F and Y_O carried;
// products and inert are the balance. omega = A rho^2 Y_F Y_O e^(-Ta/T)
// (fuel mass per volume and time, dns::arrheniusRate2); oxidizer consumed at
// s omega, heat q omega per unit fuel. Units: rho0 = R = 1, the jet speed
// U = 1, length l = D0/U with D0 = mu/(rho0 Pr) (so mu = Pr and the slot
// width in l IS the jet Peclet number), T0 = p0 = 1/(gamma M^2), M = U/c0.
#pragma once

#include "dns/bunsen.hpp"

#include <span>

namespace dns {

// Pre (validated): s > 0, yF1, yO2 in (0, 1], 1 <= sigma <= 10 (1 = no heat
// release), beta >= 0, Da > 0.
struct DiffusionModel
{
    double s = 1;       // stoichiometric oxidizer-to-fuel mass ratio
    double yF1 = 1;     // fuel stream Y_F (balance inert)
    double yO2 = 1;     // oxidizer stream Y_O (balance inert)
    double sigma = 5;   // flame-sheet temperature ratio T_ad(Z_st) / T0
    double beta = 8;    // Zeldovich number at T_ad: Ta = beta sigma^2/(sigma-1) T0
    double Da = 100;    // A rho0 e^(-Ta/T_ad) in 1/(l/U) (the rate at the sheet)

    // Stoichiometric mixture fraction yO2 / (s yF1 + yO2).
    double zst() const { return yO2 / (s * yF1 + yO2); }
    // Heat release per unit fuel over cp T0: (sigma - 1)/(yF1 zst), so the
    // Burke-Schumann sheet sits at T_ad = sigma T0.
    double qOverCpT0() const { return (sigma - 1.0) / (yF1 * zst()); }
    // Activation temperature over T0; 0 with no heat release (sigma == 1).
    double theta() const
    {
        return sigma > 1 ? beta * sigma * sigma / (sigma - 1.0) : 0.0;
    }
    // ln A in box units: ln Da + Ta/T_ad (rho0 = 1).
    double lnA() const;
};

// NSCBC outflow pressure anchor of the jet (DeviceCompNscbc::Config::sigma;
// 0.25 elsewhere). The jet's heat release is a steady volume source the
// outflow must pass: at 0.25 (relaxation L/(sigma c) ~ 20 l/U_jet) the box
// sat at p/p0 1.67 by t = 14 (W 16, s 1, sigma 5, M 0.05) and every cold
// stream compressed to T/T0 1.6-1.7. Measured in docs/chem.md.
inline constexpr double kDiffOutflowSigma = 4.0;

// Mixture fraction Z = (s Y_F - Y_O + yO2)/(s yF1 + yO2): 1 in the fuel
// stream, 0 in the oxidizer, affine in the coupling function
// beta = Y_F - Y_O/s (no source).
double mixtureFraction(const DiffusionModel& m, double YF, double YO);

// The Burke-Schumann (infinitely fast) state at mixture fraction Z in [0, 1]:
// lean side Y_F = 0, Y_O = yO2 (1 - Z/zst); rich side Y_O = 0, Y_F = yF1 (Z -
// zst)/(1 - zst); t = T/T0 = 1 + qOverCpT0 (yF1 Z - Y_F) (Shvab-Zeldovich),
// peak sigma at Z = zst.
struct SheetState
{
    double YF, YO, t;
};
SheetState burkeSchumann(const DiffusionModel& m, double Z);

// Inverse of erfc on (0, 2) to FP64 round-off (Newton from a bisection start).
double erfcInv(double y);

// The slot jet inflow (x-lo, a BunsenGeom with uJet = 1): the profile
// weights J, P, C of bunsenInflow, mixture fraction Z = J + zst P (the pilot
// strips are burnt gas AT stoichiometry), the Burke-Schumann state of Z, u =
// J + uPilot P + uCo C (a pulse scales J). The jet edge is then a burning
// flame sheet at the exit plane: the diffusion flame's anchor.
struct DiffusionInflowRow
{
    double u, Z, YF, YO, t, jet;
};
DiffusionInflowRow diffusionInflow(const DiffusionModel& m, const BunsenGeom& b,
                                   double y);

// The jet read off FP32 fields (cell (i, j) at i + nx j, square cells h,
// slot centred on yc with yc/h - 1/2 an integer + 1/2):
//   tip:  the first axis row (mean of the two columns either side of yc)
//         where Z drops below zst, interpolated; NaN = the fuel core reaches
//         the last row (open tip).
//   base: the lowest row whose max omega >= 0.1 max omega (the flame's
//         anchor height; 0 = attached).
//   half: the Z = zst half-width at x = tip/2 (each side walked outward
//         from the axis, averaged); NaN without a tip.
//   zMin, zMax, the mixture-fraction bounds; hits: cells with Z, Y_F or Y_O
//   outside [-1e-6, 1 + 1e-6] (Y_F against yF1, Y_O against yO2).
struct JetMeasure
{
    double xTip = 0, xBase = 0, half = 0;
    double zMin = 0, zMax = 0, yfMin = 0, yoMin = 0, hits = 0;
};
JetMeasure measureJet(const DiffusionModel& m, std::span<const float> YF,
                      std::span<const float> YO, std::span<const float> omega,
                      int nx, int ny, double h, double yc);

// --mode candle's ambient column: isothermal at T0 under gravity g along -x,
// p_a(x) = p0 exp(-g x / (R T0)) with x from the inflow face (x = 0). The IC
// pressure, the inflow density targets, the outflow pressure target and the
// NSCBC hydrostatic waves all read it. g == 0 returns p0 exactly.
// Pre: p0 > 0, g >= 0, RT0 > 0.
double candleAmbientP(double p0, double g, double RT0, double x);

// --mode candle's dimensionless groups (box units: U_fuel = 1, l, gamma 1.4):
//   froude      U^2/(g W) = 1/(g W): jet momentum vs buoyancy (<< 1 = a
//               buoyancy-controlled flame)
//   uFreeFall   sqrt(2 g L (sigma - 1)/sigma): the inviscid free-fall speed
//               of flame-sheet gas over the box length L -- a SCALE, not a
//               bound (defaults: 28.4 U; measured |u|max 35 U, the periodic
//               box's recirculation adds to the plume)
//   machBuoyant M uFreeFall
//   adiabatic   (g/cp) L / ((sigma - 1) T0) = g L (gamma - 1) M^2/(sigma - 1):
//               the adiabatic lapse over the box vs the flame's temperature
//               rise (conv's "adiabatic fraction")
//   hydroDrop   1 - p_a(L)/p0 = 1 - exp(-gamma M^2 g L): the ambient
//               column's stratification
//   flicker     0.479 sqrt(g/W): Cetegen & Ahmed's f = 1.5 D^(-1/2) Hz (D in
//               m) rewritten as f = (1.5/sqrt(9.81 m/s^2)) sqrt(g/D), D -> W,
//               in U/l -- a pool-fire correlation, a scale only
// Pre: g > 0, W > 0, L > 0, mach > 0, m.sigma > 1.
struct CandleScales
{
    double froude, uFreeFall, machBuoyant, adiabatic, hydroDrop, flicker;
};
CandleScales candleScales(const DiffusionModel& m, double g, double W,
                          double L, double mach);

// --mode candle --quiescent's inflow row: the oxidizer stream at rest at T0
// (Z = 0, u = 0, Y_O = yO2, no fuel anywhere).
DiffusionInflowRow quiescentInflow(const DiffusionModel& m);

// Burke-Schumann tip height of a CONSTANT-density planar slot jet of width W
// at uniform speed 1 and diffusivity 1 (axial diffusion neglected): the axis
// mixture fraction erf(W / (4 sqrt(x))) reaches zst at x = W^2 / (16
// erfinv(zst)^2). The heated jet's own D(T) rises with T, so the measured
// tip sits BELOW it; printed as the scale, never a validation.
double sheetTipConstantDensity(double W, double zst);

} // namespace dns
