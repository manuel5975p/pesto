// The nozzle's library half (docs/nozzle.md): the rest start, the ideal mass flow, the stats
// row of a run on a Total inlet, the thrust ledger's roles and _thrust.csv rows. Host, no GPU, no
// physics in the driver.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvref.hpp"

#include <array>
#include <cstdint>
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

// The ledger per role (patches summed in patch order) on the six rows, and the omega integral.
// fuelRow: the fuel's state row (5 + fuel), < 0 without species.
struct FvRoleSums
{
    std::array<FvRow6, kFvRoles> I{};
    double omega = 0;
};
FvRoleSums fvRoleSums(const FvLedger& l, std::span<const FvRole> roles, int fuelRow);

// sum_c V q_c on the six rows, Neumaier-compensated in ascending cell order (s as downloaded: hi + lo).
FvRow6 fvTotals6(const FaceMesh& fm, const FvState& s, int fuelRow);

// The constants of a thrust row: axis a (unit), ambient pa, the Total row's p0, t0 (box units), the
// throat area astar, the heat release q (0 inert), the gas, exitArea = sum_exit S.a.
struct FvThrustCase
{
    std::array<double, 3> axis{1, 0, 0};
    double pa = 0, p0 = 0, t0 = 0, astar = 0, q = 0, gamma = 1.4, Rgas = 1, exitArea = 0;
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

// _thrust.csv: the header and one row (R2's columns). res_k = P_k - P0_k + sum_roles I_k - q omega [E]
// + omega [fuel]: 0 to rounding.
std::vector<std::string> fvThrustColumns();
std::vector<double> fvThrustRow(double t, long step, double dt, const FvRow6& P, const FvRow6& P0, const FvRoleSums& now,
                                const FvThrustWindow& w, const FvExitSample& x, const FvThrustCase& k);

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

} // namespace dns
