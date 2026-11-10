// The nozzle's library half (docs/nozzle.md): the rest start, the ideal mass flow, the stats
// row of a run on a Total inlet, the thrust ledger's roles and _thrust.csv rows. Host, no GPU, no
// physics in the driver.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvref.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace dns {

// Uniform rest at p, T: the IC of a nozzle run, raised by a ramp on the Total row's p0.
// Pre: p, T > 0.
FvState fvRestState(const FaceMesh& fm, double p, double T, double gamma, double Rgas);

// Ideal choked mass flow through astar of a gas at stagnation p0, T0 (what Cd divides by).
// Pre: all > 0.
double fvIdealMassFlow(double p0, double T0, double astar, double gamma, double Rgas);

// The stats row's definitions: astar, axis (unit), p0 and t0 stagnation in box units.
struct FvNozzleRun
{
    double astar = 0;
    std::array<double, 3> axis{1, 0, 0};
    double pRef = 0, p0 = 0, t0 = 0;
};

// mdotIn, mdotOut > 0 through the Total and Outflow patches; Fwall the gauge wall force
// along the axis; Fin, Fexit the outward patch fluxes of momentum along it; Cd = mdotOut /
// ideal, Cf = Fexit / (p0 A*).
struct FvNozzleRow
{
    double mdotIn = 0, mdotOut = 0, Fwall = 0, Fin = 0, Fexit = 0, Cd = 0, Cf = 0;
};

// bc: a row per patch. mass, mom[3]: DeviceFvComp::patchFlux of rows 0, 1..3, per patch.
// fw: the tallied force. Pre: spans sized alike; run.astar > 0.
FvNozzleRow fvNozzleRow(std::span<const FvBc> bc, std::span<const double> mass, const std::array<std::vector<double>, 3>& mom,
                        const std::array<double, 3>& fw, const FvNozzleRun& run, double gamma, double Rgas);

// --- Thrust: the impulse ledger's rows (docs/nozzle.md "Thrust", masterplan_combnozzle R2) ---

// A patch's part in the momentum identity, from its BC kind alone: in (Total, Inflow, Far), exit
// (Outflow), side (a `wedge` row, or a Slip on a symmetryPlane / symmetry mesh patch: the planar axis),
// wall (every other Wall, Slip), none (Empty). Every non-empty boundary face has exactly one role.
enum class FvRole : std::uint8_t { In, Exit, Wall, Side, None };
inline constexpr int kFvRoles = 4; // In .. Side

// Per FaceMesh patch its role. Pre: bc holds a row per patch first (FvConfig::bc's rule).
std::vector<FvRole> fvRoles(std::span<const FvBc> bc, const FaceMesh& fm);

// The thrust row's six state rows: mass, the three momenta, E, the fuel's rho Y (0 without species).
using FvRow6 = std::array<double, 6>;

// The ledger per role (patches summed in patch order) on the six rows, and the omega integral, each
// patch read compensated (FvLedger::at, omegaSum). fuelRow: the fuel's state row (5 + fuel), < 0 without species.
struct FvRoleSums
{
    std::array<FvRow6, kFvRoles> I{};
    double omega = 0;
};
FvRoleSums fvRoleSums(const FvLedger& l, std::span<const FvRole> roles, int fuelRow);

// sum_c V q_c on the six rows, Neumaier-compensated in ascending cell order (s as downloaded: hi + lo).
FvRow6 fvTotals6(const FaceMesh& fm, const FvState& s, int fuelRow);

// The constants of a thrust row: axis a (unit), ambient pa, the Total row's p0, t0 (box units), the
// throat area astar, the heat release q (0 inert), the gas, exitArea = sum_exit S.a. station: the
// reacting nozzle's burnt stagnation plane along a (fvStation; NaN: no station columns).
struct FvThrustCase
{
    std::array<double, 3> axis{1, 0, 0};
    double pa = 0, p0 = 0, t0 = 0, astar = 0, q = 0, gamma = 1.4, Rgas = 1, exitArea = 0;
    double station = NAN;
};

// sum_exit S.a over the active Outflow faces. Pre: roles = fvRoles of fm.
double fvExitArea(const FaceMesh& fm, std::span<const FvRole> roles, const std::array<double, 3>& axis);

// What one evaluation of the current state gives (ev and its rhs out, DeviceFvComp::evaluate):
// Texit = sum_exit (F - pa S).a; Twall = -sum_{in, wall, side} (F - pa S).a - (sum_c V dq/dt).a, the
// same number by the gather's telescoping; Fmom = sum_exit mdot_f (u_B.a), Fpres = sum_exit (p_B - pa)
// S.a (B the boundary state); exitSuper the exit area fraction with u_B.n >= c_B.
struct FvExitSample
{
    double Texit = 0, Twall = 0, Fmom = 0, Fpres = 0, exitSuper = 0;
};
FvExitSample fvExitSample(const FaceMesh& fm, std::span<const FvRole> roles, const FvEval& ev, const FvState& rhs,
                          const FvThrustCase& k);

// Sample-free means over a window t1 < t2 between two ledger reads a, b: mdotIn, mdotOut, the vacuum
// thrust Tvac = Delta I_exit.a / Delta t (pRef 0), CfVac = Tvac/(p0 A*), Cf = (Tvac - pa exitArea)/(p0 A*),
// cstar = p0 A*/mdotOut.
struct FvThrustWindow
{
    double mdotIn = 0, mdotOut = 0, Tvac = 0, CfVac = 0, Cf = 0, cstar = 0;
};
FvThrustWindow fvThrustWindow(const FvRoleSums& a, double t1, const FvRoleSums& b, double t2, const FvThrustCase& k);

struct FvStation;

// The reacting nozzle's exact storage (masterplan_combclose 2.2): Mst = sum V rho over the cells with axis.C > station,
// Mth over the cells flagged past the throat (fvPastThroat); Neumaier sums in ascending cell order of the downloaded
// state. The throat flux of a window is mdotOut + Delta Mth / Delta t, sample-free.
struct FvStorage
{
    double Mst = 0, Mth = 0;
};
FvStorage fvStorage(const FaceMesh& fm, const FvState& s, const std::array<double, 3>& axis, double station,
                    std::span<const std::uint8_t> pastThroat);

// _thrust.csv: the header and one row (R2's columns). res_k = P_k - P0_k + sum_roles I_k - q omega [E]
// + omega [fuel]: 0 to rounding. station (the reacting nozzle's): p0b T0b CdB CfBvac CfB Mst Mth appended, Cd
// and Cf on the burnt station's (p0b, T0b) with the window's mdotOut, Tvac. Pre: b, m non-null iff station.
std::vector<std::string> fvThrustColumns(bool station = false);
std::vector<double> fvThrustRow(double t, long step, double dt, const FvRow6& P, const FvRow6& P0, const FvRoleSums& now,
                                const FvThrustWindow& w, const FvExitSample& x, const FvThrustCase& k,
                                const FvStation* b = nullptr, const FvStorage* m = nullptr);

// The exit plane's Mach number from window-mean fluxes per area: f1 mass, f2 momentum + pressure along
// the normal, f3 energy (the quasi-1D state the conserved fluxes carry). supersonic picks the root.
// Pre: f1 > 0, a real root exists (else NaN).
double fvExitMach(double f1, double f2, double f3, double gamma, bool supersonic);

// masterplan_combnozzle 2.3, the burnt stagnation station: over the internal faces whose owner and
// neighbour centroids straddle s = station along axis, the face state is the mean of the two cells'
// primitives; T0 = T + |u|^2/(2 cp), p0 = p (T0/T)^(gamma/(gamma - 1)), both weighted by |rho u.S|.
// faces 0: no face crosses (p0, T0 NaN).
struct FvStation
{
    double p0 = 0, T0 = 0, mdot = 0;
    int faces = 0;
};
FvStation fvStation(const FaceMesh& fm, const FvState& s, const std::array<double, 3>& axis, double station, double gamma,
                    double Rgas);

// --- The reacting quasi-1D nozzle (docs/nozzle.md "Reacting", masterplan_combnozzle 2.1, R5) ---

// The constant-area deflagration of a fresh state (Tu, pu, uu): complete burn of q, exact Rayleigh-Hugoniot
// (p + m u, cp T + u^2/2 + q conserved), the subsonic root by bisection on u_b in [u_u, (P/m) gamma/(gamma+1)];
// stagnation T0b, p0b of the burnt gas. q: the heat released (q (1 - Y) for a partial burn). Pre: Tu, pu, uu > 0.
struct FvBurnt
{
    double ub = 0, pb = 0, Tb = 0, Mb = 0, T0b = 0, p0b = 0;
};
FvBurnt fvRayleighBurnt(double gamma, double Rgas, double cp, double q, double Tu, double pu, double uu);

// The composition oracle (masterplan_combnozzle 2.1): Total (p0u, T0u) fresh gas, a planar flame in a chamber
// of area eps_c A*, the burnt gas choked at A* (m eps_c = Gamma p0b / sqrt(R T0b), bisection on M_u in [1e-6,
// 0.3]) and expanded isentropically to eps_e (the supersonic root). q: the SOLVER's heat release, cp (sigma - 1)
// Tu with Tu the box's static fresh T (never one normalised on T0u). CFvac on p0b, A*. sigma only feeds the
// low-Mach estimate MuLowM = Gamma / (sqrt(gamma sigma) eps_c). Error: eps_c <= 1 (no throat), no choke root in the
// bracket, a thermally choked chamber (M_b >= 1), eps_e < 1. g, R, cp, q echo the inputs (the IC's Rayleigh line).
struct FvReactingDesign
{
    double g = 0, R = 0, cp = 0, q = 0;
    double Gamma = 0, Mu = 0, Tu = 0, pu = 0, rhou = 0, uu = 0, m = 0;
    FvBurnt b{};
    double Me = 0, peOverP0b = 0, CFvac = 0, MuLowM = 0;
};
std::expected<FvReactingDesign, std::string> fvReactingNozzleDesign(double gamma, double Rgas, double cp, double q,
                                                                    double sigma, double epsC, double epsE, double p0u,
                                                                    double T0u);

// The supersonic (sup) or subsonic Mach number of area ratio ar >= 1 (bisection; ar <= 1 gives 1).
double fvAreaMach(double ar, double gamma, bool sup);

// A multi-row planar [nozzle] chamber (combscale S5): r = (C - origin).lift; Hc = max over Wall-role faces owned by
// chamber cells of (Cf - origin).lift; seed = --wrinkle A times delta_th (0: planar).
struct FvNozzleShape
{
    std::array<double, 3> origin{}, lift{0, 1, 0};
    double Hc = 0, seed = 0;
};

// The shape of a [nozzle] flame mesh: nullopt on a one-row chamber (no chamber bin holds two cells).
// Pre: probe.spec().chamber, roles = fvRoles of fm. Post: Hc > 0 when set.
std::optional<FvNozzleShape> fvNozzleShapeOf(const FaceMesh& fm, const FvFlameProbe& probe, std::span<const FvRole> roles,
                                             const std::array<double, 3>& origin, const std::array<double, 3>& lift);

// The front's transverse amplitudes on the chamber (combscale S5, G6.4): xIa_n = 2 sum_chamber V rhoY cos(n pi r/Hc) /
// (rhoY_ref A_ref), n = 1, 2, in length units; rhoY_ref, A_ref the probe's S_L normalisation (its reference slab's mean
// rho Y - yBurnt rho, probe.area()). Neumaier sums in ascending cell order, FP64. A planar front reads ~1e-16 x_I; a
// front displaced by a cos(pi r/Hc) reads xIa1 = a. Pre: s holds the fuel row.
std::array<double, 2> fvFrontAmplitudes(const FaceMesh& fm, const FvFlameProbe& probe, const FvNozzleShape& shape,
                                        const FvState& s);

// The reacting nozzle's IC (masterplan_combnozzle P4, masterplan_combclose 2.1): per slab of a chamber probe its
// area A = V_slab/h; the chamber bins on the Rayleigh line of the design's (m, P, H): p + m u and cp T + u^2/2 +
// q (1 - Y) conserved (fvRayleighBurnt at q (1 - Y)), Y = (sigma - t)/(sigma - 1) from the flame's T/Tu profile
// t(s - xf) (s from the probe's low end); Y 0 is d.b. Every later slab burnt and isentropic at (d.b.p0b, d.b.T0b),
// subsonic up to the least-area slab (fvThroatBin), supersonic past it, M of A/astar, Y 0. Pre: probe.spec().chamber,
// d from fvReactingNozzleDesign.
// shape (combscale S5, a multi-row planar chamber): the chamber's profile at t(s - xf - seed cos(pi r/Hc)); the
// nozzle half carries v = u r dA_b/A_b along lift (dA_b central over the bin centres, one-sided at the ends: the
// generator's u y_w' r/y_w), E += rho v^2/2. nullptr: the one-row body verbatim (v 0).
FvState fvReactingNozzleIC(const FaceMesh& fm, const FvFlameProbe& probe, const FvReactingDesign& d,
                           const std::function<double(double)>& t, double xf, double astar,
                           const FvNozzleShape* shape = nullptr);

// Per probe bin its area V_bin/h (cells summed in ascending order): the IC's slab areas.
std::vector<double> fvBinAreas(const FaceMesh& fm, const FvFlameProbe& probe);

// The throat bin: the least-area non-empty bin at or past nCh (the first on a tie); nCh if none is non-empty.
// Pre: nCh < A.size().
int fvThroatBin(std::span<const double> A, int nCh);

// Per cell 1 iff probe.binOf(c) > the throat bin (fvThroatBin of fvBinAreas, the IC's rule): Mth's cells.
// Pre: probe.spec().chamber.
std::vector<std::uint8_t> fvPastThroat(const FaceMesh& fm, const FvFlameProbe& probe);

} // namespace dns
