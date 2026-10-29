// `dns flame --mode candle --wick` geometry and soot layer (docs/chem.md,
// "Wick candle"): a candle body (an immersed solid block) with a wick standing
// up from its top, fuel vapour leaving the wick surface through a source band,
// and a one-equation semi-empirical soot model carried as a third species.
// Pure host: the body SDF, the wall temperature over the surface, the per-cell
// region codes the runner's source and balances read, the soot constants, and
// the flame / soot measurement. No GPU, no filesystem (config layer).
//
// Box units are the candle's (dns/diffusion.hpp): rho0 = R = 1, U = 1 (the
// slot candle's fuel speed; here only the unit), l = D0/U, T0 = p0 =
// 1/(gamma M^2). x is up (gravity along -x).
#pragma once

#include "dns/diffusion.hpp"
#include "dns/grid.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace dns {

// The candle, x up, axis y = yc. Body: the box [xBase, xTop] x [yc - wCandle/2,
// yc + wCandle/2] (its top is the wax surface). Wick: the box [xTop - wWick,
// xTop + hWick] x [yc - wWick/2, yc + wWick/2] (sunk wWick into the body so the
// union has no seam). Walls: isothermal, the wick (wall points above xTop) at
// tWick T0 -- the fuel's vaporisation temperature -- the body at T0. Fuel:
// mass rate Q per unit depth and time, Y_F = yF, leaving the wick surface at
// tWick T0 (WickRegion::Source).
// Pre (validated): wWick, hWick, wCandle > 0, wCandle > wWick, xTop > xBase,
// tWick >= 1, Q > 0, yF in (0, 1].
struct WickSpec
{
    double yc = 0, xBase = 0, xTop = 0, wCandle = 0;
    double wWick = 0, hWick = 0;
    double tWick = 2, tCandle = 1;
    double Q = 1, yF = 1;

    double xTip() const { return xTop + hWick; }
};

// sdf > 0 in fluid, < 0 in solid: the min of the two boxes' exact signed
// distances (exact outside the union, a lower bound inside).
double wickSdf(const WickSpec& s, double x, double y);

// The wall temperature over T0 at a wall point (x, y): tWick on the wick
// above the wax surface (x > xTop), tCandle elsewhere (the body).
double wickWallT(const WickSpec& s, double x, double y);

// Per-cell region codes (gidx order, centres at (i+0.5)h).
enum class WickRegion : std::uint8_t
{
    Fluid = 0,
    Solid = 1,  // sdf < 0 at the centre: ghost or deep, never fluid
    Source = 2, // fluid cell face-adjacent to a WICK solid cell (centre x >
                // xTop): the fuel vapour enters here
};
// Pre: g.nz == 1, the body at least 2 cells off every box face.
std::vector<std::uint8_t> wickRegions(const Grid& g, const WickSpec& s);

// --- Soot (one-equation, semi-empirical; docs/chem.md "Soot") -------------
// rho Y_s carried as species 2. Formation from the fuel on the rich side,
//   omega_f  = A_f rho Y_F e^(-Ta_f/T)                 (fuel -> soot, no heat)
// oxidation by the oxidizer (first order in soot mass: fixed particle size),
//   omega_ox = A_ox rho^2 Y_s Y_O e^(-Ta_ox/T)          (soot + s O -> products)
// with the fuel's stoichiometry and heat release: soot is the fuel's carbon
// in condensed form, so T_ad and the mixture fraction keep their meaning. No
// molecular diffusion (Sc_soot -> infinity; the MC limiter's dissipation
// only), no thermophoresis, no radiation. Ta_f = 50 T0 and Ta_ox = 60 T0
// (about 15000 K and 18000 K at T0 = 300 K: between Moss's surface growth
// 12100 K and Leung-Lindstedt's nucleation 21100 K, and Lee's oxidation
// 19800 K); the pre-exponentials are Damkohler numbers at T_ad, TUNED FOR
// APPEARANCE (a sooting yellow zone burned out at the tip), not calibrated.
inline constexpr double kSootThetaF = 50;
inline constexpr double kSootThetaOx = 60;
// Defaults (docs/chem.md scouting table, defaults, tt 8, t > 5): Da_ox 2e3
// let 90% of the soot out of the top (smoking: a filament up the axis);
// 5e4 still 9% out; 2e5 and 1e6 burn it out at the tip (out/formed 3e-4 /
// 3e-5). Da_f 10 forms 4.7% of the fuel into soot, peak f_v ~1 ppm.
inline constexpr double kSootDafDefault = 10.0;
inline constexpr double kSootDaoxDefault = 2e5;
// The wick candle's main-step Damkohler number (--damkohler absent): at the
// slot candle's 5000 the wick's IC sheet LIFTED OFF and went out by t = 3
// (exit 8; the buoyant draft past the wick base outran the edge flame);
// 2e4 / 5e4 hold the base at 1.6-0.6 / 0.75 l above the wax (scouts, tt 4).
// Neither binds the step (dtChem 1.3e-3 vs the acoustic 1.1e-4).
inline constexpr double kWickDaDefault = 5e4;
// Soot density over the ambient air density (1800 kg/m^3 / 1.177 kg/m^3):
// f_v = rho Y_s / kSootDensityBox in box units (rho0 = 1).
inline constexpr double kSootDensityBox = 1800.0 / 1.177;

struct SootModel
{
    double Daf = kSootDafDefault;   // A_f e^(-Ta_f/T_ad), per l/U
    double Daox = kSootDaoxDefault; // A_ox rho0 e^(-Ta_ox/T_ad), per l/U
    double thetaF = kSootThetaF, thetaOx = kSootThetaOx;

    // ln A in box units at sheet temperature ratio sigma = T_ad/T0.
    double lnAf(double sigma) const;
    double lnAox(double sigma) const;
};

// The flame read off FP32 fields (cell (i, j) at i + nx j, square cells h),
// fluid cells only (solid[c] != 0 skipped):
//   xTip:   the first axis row ABOVE the wick tip (mean of the two columns
//           either side of yc) where Z drops below zst, interpolated; NaN =
//           open (no crossing below the last row).
//   height: xTip - s.xTip() (the visible flame height above the wick).
//   half:   the Z = zst half-width at x = (s.xTip() + xTip)/2; NaN without.
//   xBase:  the lowest fluid row whose max omega >= 0.1 max omega.
//   xSoot:  the highest row whose max Y_s >= 0.1 max Y_s (the sooting zone's
//           top: the luminous height); NaN without soot.
//   sootMax, zMin, zMax, hits (Z, Y_F, Y_O outside [-1e-6, 1 + 1e-6] or Y_s
//   below -1e-6), as measureJet.
struct WickMeasure
{
    double xTip = 0, height = 0, half = 0, xBase = 0, xSoot = 0;
    double sootMax = 0, zMin = 0, zMax = 0, hits = 0;
};
WickMeasure measureWick(const DiffusionModel& m, const WickSpec& s,
                        std::span<const float> YF, std::span<const float> YO,
                        std::span<const float> Ys,
                        std::span<const float> omega,
                        std::span<const std::uint8_t> solid, int nx, int ny,
                        double h);

} // namespace dns
