// The low-Mach flame solver's config layer (masterplan real_bunsen S4,
// docs/chem.md "Low-Mach"): the FFT-size rule, the defaults, and the initial
// state laid from the FP64 oracle. Pure host: no GPU, no filesystem.
//
// Box units: rho_u = S_L = R = 1, delta_D = 1, Tu = P0 = 1 (so T = 1/rho
// exactly the EOS), cp = gamma/(gamma - 1) = 3.5, mu = Pr at Tu.
//
// Staggered (MAC) layout, i fastest:
//   cells   c = i + nx (j + ny k)
//   x faces f = i + (nx + 1)(j + ny k), i in [0, nx]: face i is cell i's
//           lo face; face 0 the inflow, face nx the outflow (both distinct)
//   y faces the cell's lo face (j - 1/2), indexed like the cell (periodic)
//   z faces the cell's lo face (k - 1/2), indexed like the cell (periodic)
//   open laterals (S6): the y = 0 face is cell (i, 0, k)'s lo face, the
//           y = Ly face lives in myHi[i + nx k]; z alike (mzHi[i + nx j])
#pragma once

#include "dns/chem.hpp"
#include "dns/config.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace dns {

// The convective CFL default of the central MAC SSP-RK3 (no acoustics).
inline constexpr double kLmCflDefault = 0.8;

// The chemistry dt fraction K of the bound K / max J (--chem-dt). MEASURED
// (2026-10-31, the M1 case 1D: phi 1.3 C2b, sigma 6.88, Sutherland, Pr 0.52,
// 4 cells/delta_th): K 0.5 binds 1.43x below diffusion, K 1 and 2 leave
// diffusion binding and move S_L by -3.1e-5 vs 0.5 -- explicit chemistry
// does not bind at K 1 (no point-implicit S4b needed). RK3's real-axis
// bound is 2.51.
inline constexpr double kLmChemDtDefault = 1.0;

// `--dt-rule cell`'s positivity number K of dt <= K / max_c (2 C_c + D_c)
// (masterplan real_bunsen S9, issues LM10; DeviceLowMach::positivityDt).
// K = 1 is forward Euler's positivity bound, which SSP-RK3 inherits (its SSP
// coefficient is 1). MEASURED (4 cells/delta_th, 2D rim slot U 10, 3D round
// pilot burner, g 0 and real g): K 1 bounded (0 hits), K 1.5 not (1.6e3 -
// 1.9e4 hits); docs/chem.md "Low-Mach dt rule".
inline constexpr double kLmPosKDefault = 1.0;

// The checkpoint "solver" record of runLowMach (dns_main's phase 1.5 checks a
// --restart against it; runFlame writes "flame-compressible").
inline constexpr const char* kLowMachSolverName = "flame-lowmach";
inline constexpr const char* kFlameCompSolverName = "flame-compressible";

// The premixed weight of a cell's flame index xi (Takeno; D14): w_p =
// smoothstep(-0.2, 0.2, xi) = t^2 (3 - 2t), t = clamp((xi + 0.2)/0.4, 0, 1).
// The ONE definition of the premixed/diffusion split: the low-Mach runner's
// burnPre = integral omega w_p, burnDif = integral omega (1 - w_p), and the
// renderer's two emission grids (S9).
inline double lmPremixedWeight(double xi)
{
    const double t = std::clamp((xi + 0.2) / 0.4, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Real gravity in model units (masterplan real_bunsen S7, mapping M1): g
// delta_D,model / S_L^2 with GRI-3.0's CH4/air phi 1.3 (Cantera 3.2.0,
// tests/data/chem/ch4_gri30.csv: S_L 0.23123712 m/s, delta_th 0.61874103 mm)
// and g 9.80665 m/s^2; delta_D,model = delta_th / dthOverDD (the model's own
// delta_th/delta_D, 4.70964 at phi 1.3 C2: 0.02410). For the dry-run and
// the --gravity message only (the flag is in model units).
inline double lmRealGravity(double dthOverDD)
{
    const double sl = 0.23123712, dth = 0.61874103e-3, g = 9.80665;
    return g * (dth / dthOverDD) / (sl * sl);
}

// --rim's defaults (masterplan real_bunsen S8): the room draft below the rim
// (D15: 0.4 S_L = 0.1 m/s at M1, keeps x-lo an inflow outside the tube) and
// the tube inside the box below the exit (delta_th; the final case's 7.5).
inline constexpr double kLmRoomDraft = 0.4;
inline constexpr double kLmTubeLengthDefault = 7.5;
// The --rim IC's products column: this many delta_th past the ideal tip.
inline constexpr double kLmRimColumn = 4.0;

// The tube's signed distance at (x, y, z) (masterplan real_bunsen S8; b.rimLip
// > 0): the wall R_i = W/2 <= r <= R_o = W/2 + rimLip, x <= xRim (it stands
// on the inflow face and continues below the box), r = |y - yc| (slot: two
// lips) or hypot(y - yc, z - zc) (round). The exact distance in the (x, r)
// half plane (a true sdf; negative inside the wall). The ONE tube definition:
// the solid map, the Poiseuille rows and the renderer's json "rim" read it.
double lmTubeSdf(const BunsenGeom& b, double x, double y, double z);

// The solid map of grid g: 1 where the cell CENTRE is inside the tube (sdf <
// 0), else 0 (the staircase; LmPressure's face codes derive from it). Empty
// when b.rimLip == 0.
std::vector<std::uint8_t> lmTubeSolid(const BunsenGeom& b, const Grid& g);

// n's prime factors are all <= 7 (hipFFT's fast radices). Pre: n >= 1.
bool lmFftSmooth(int n);
// The nearest 7-smooth sizes >= n and <= n (>= 1). Pre: n >= 1.
int lmNextSmooth(int n);
int lmPrevSmooth(int n);

// The low-Mach state on the host, FP64 (uploaded as FP32): rho, rho Y_k per
// cell; the MAC mass fluxes; the dynamic pressure p (cells). Open laterals
// (S6): myHi (nx nz, the y = Ly faces, i + nx k) and mzHi (nx ny, i + nx j);
// empty = 0 on upload, and always empty with periodic laterals.
struct LowMachState
{
    std::vector<double> rho, p, mx, my, mz, myHi, mzHi;
    std::array<std::vector<double>, 2> rhoY;
    int nY = 1;
};

// The `dns flame --physics lowmach` IC. Premixed (1D/2D/3D): the oracle
// profile T(x - x_f0 - wrinkle(y)) of Case::flameEig (Tu = 1), rho = 1/T, Y
// by Shvab-Zeldovich (C0: Y = (sigma - T)/(sigma - 1); --equivalence: the
// C2 reactants of flameIC2's rule); every x face carries the inflow mass flux
// m = u_in, my = mz = 0, p = m^2 (T_last - T) per row (the 1D momentum
// balance p + m u = const, p = 0 at the burnt end). Bunsen (--mode bunsen,
// masterplan real_bunsen S5): lowmachBunsenIC. Pre: c.physics == LowMach,
// c.sub == Flame, premixed or bunsen mode.
LowMachState lowmachIC(const Case& c);

// A partially premixed cell of the --equivalence burner (masterplan
// real_bunsen S5) from its mass fractions of three gases: fresh rich premix
// a (Y_F1, Y_O1, T = Tu), the jet's burnt products b (Y_F,rem, Y_O 0) whose
// T-weighted part is tB, air cAir (Y_O2) with tAir; the products + air part
// is laid BURNT TO COMPLETION (Burke-Schumann: the deficient reactant
// consumed, s Y_F - Y_O and T + (q/cp) Y_F kept), the fresh part untouched.
// So no products/air overlap is ever a reactive mixture (a box-long hot
// column against air, laid as a linear blend, exploded at t = 0: measured,
// --length 90 diverged in the first steps). Z = a + b exactly.
struct LmMix
{
    double t, yF, yO;
};
LmMix lmPartialPremixMix(const PartialPremixModel& pm, double a, double b,
                        double tB, double cAir, double tAir);

// The low-Mach Bunsen's x-lo inflow FACE state at (y, z): bunsenInflowAt's
// weights J (jet), P (pilot strips), C (coflow), speed u, m = u/t (rho =
// 1/t). C0: t = J + tPilot P + tCo C, Y = J (fresh jet, burnt or inert
// strips and coflow); --equivalence (C2): the jet the fresh rich premix,
// the pilot strips its burnt products at tPilot, the coflow AIR at tCo,
// lmPartialPremixMix(J, P, tPilot P, C, tCo C): the pilot/air edge enters
// as its Burke-Schumann state. Pre: c.flameBunsen, c.physics == LowMach.
// --rim (S8, c.lmRim): the rows are SHARP (no tanh): inside the tube the
// developed Poiseuille profile of the jet (lmPoiseuille), fresh premix at Tu;
// the tube's wall (R_i <= r <= R_o) m = 0, t = tRim, Y = 0 (a closed face);
// outside the room draft (uCo) of cold room air (C2: air; C0: Y = 0).
struct LmInflowRow
{
    double m, t, yF, yO, jet, pilot, co;
};
LmInflowRow lmInflowRow(const Case& c, double y, double z);

// The developed tube flow of --rim (D12), u(r) per unit bulk speed at the
// cell centres of grid g: the discrete steady solution of the solver's
// viscous operator on the staircase. Slot: the parabola c ((R_e + d)^2 -
// r^2) with R_e the staircase's wall face (round(R_i / h) cells) and d =
// (sqrt(W_e^2 + h^2) - W_e)/2, W_e = 2 R_e -- the antimirror no-slip ghost
// (u_g = -u_1) makes the parabola through r = R_e + d EXACT for the 3-point
// Laplacian; round: 2 (1 - r^2/R_i^2) (the staircase approximates the
// circle). Both scaled so the mean over the tube's fluid cells is 1 (the
// bulk speed). Returns the per-row factor (ny nz, row j + ny k; 0 outside
// the tube). Pre: b.rimLip > 0.
std::vector<double> lmPoiseuille(const BunsenGeom& b, const Grid& g);
// The scale of that profile (1 / the mean of the unscaled shape over the
// tube's inflow cells); the config stores it in b.rimScale, lmInflowRow
// reads it. Pre: b.rimLip > 0, the tube holds a cell.
double lmPoiseuilleScale(const BunsenGeom& b, const Grid& g);

// The conduction into the tube (S8, stats qRim): the sum over every
// fluid/solid face (solid from lmTubeSolid's map; x, y, and z when nz > 1;
// y/z wrapping when periodic) of lambda_w (T_c - T_w) / (h/2) times the face
// area, lambda_w = cp mu(T_w)/Pr -- the isothermal wall's face flux exactly
// as the temperature predictor forms it (rho D(T_w) (T_c - T_w)/(h/2), times
// cp). FP64, cell order. Positive = heat into the tube.
// Pre: solid.size() == T.size() == g.size().
double lmWallHeat(const Grid& g, std::span<const std::uint8_t> solid,
                  std::span<const float> T, double lambdaW, double Tw,
                  bool openY, bool openZ);

// The low-Mach Bunsen IC: bunsenIC's ideal cone (the oracle profile on the
// cone distance, tf; the jet's progress s = (sigma - tf)/(sigma - 1)). C0: t
// = tf J + tPilot P + tCo C, Y = s J. C2: lmPartialPremixMix(J s, J (1 - s)
// + P, sigma J (1 - s) + tPilot P, C, tCo C) -- the inner cone premixed, the
// products/air interface a closed-form Burke-Schumann sheet. rho = 1/t,
// every x face of a row the row's inflow m, my = mz = 0, p = 0. nz > 1: the
// slot extruded over z, the round nozzle at r = hypot(y - yc, z - zc).
// c.lmInert (--reaction off, S6): every cell its row's inflow state (no
// cone). Open laterals start with myHi = mzHi = 0 (empty).
// --rim (S8): fresh premix in the tube (x < xRim), the ideal cone standing
// on the rim (the cone distance at x - xRim), its products a SHORT column
// (r < R_i, up to kLmRimColumn delta_th past the tip: the candle's
// --ignition lesson, a box-tall hot column rolls up) laid against the room
// air as its Burke-Schumann sheet (tanh edges of half-width b.edge), room
// air elsewhere; the
// tube's cells rho = 1/tRim, rho Y = 0; every x face touching the tube 0,
// the jet and draft rows their inflow m, the lip rows above the rim 0.
// Pre: c.flameBunsen, c.physics == LowMach.
LowMachState lowmachBunsenIC(const Case& c);

} // namespace dns
