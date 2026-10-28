// The 2D slot Bunsen burner of `dns flame --mode bunsen` (docs/chem.md): the
// burner geometry, the x-lo inflow profile, the ideal cone the IC is laid on,
// and the cone measurement the runner and the tests read. Pure host; no GPU,
// no filesystem (config layer).
#pragma once

#include <span>
#include <vector>

namespace dns {

// Slot burner, box units (lengths delta_D, speeds S_L). x is the jet axis
// (inflow plane x = 0), y the periodic transverse axis, slot centred on yc.
struct BunsenGeom
{
    double W = 0;          // slot (jet) width
    double uJet = 5;       // fresh jet speed
    double uCo = 1;        // coflow speed
    double wCo = 0;        // coflow width per side
    double tCo = 1;        // coflow temperature / Tu: 1 cold inert, sigma pilot
    double pilot = 0;      // strip per side between jet and coflow, Y = 0
    double tPilot = 1;     // its temperature / Tu (sigma = burnt pilot)
    double uPilot = 0;     // its speed (0 + tPilot 1 = a cold rim)
    double edge = 0;       // tanh half-width of every profile edge, > 0
    double yc = 0;         // slot centre (a cell face: ny even)
    double coneH = 0;      // ideal cone height (W/2) sqrt(uJet^2 - 1)
    double pulse = 0;      // jet speed modulation amplitude (fraction of uJet)
    double pulseFreq = 0;  // its frequency (1/tau)
};

// The x-lo inflow state at height y: speed u, temperature ratio t = T/Tu,
// fuel fraction Y, and the jet, pilot and coflow weights. Edges
// are tanh blends of half-width edge: jet weight J = 0.5(1 - tanh((r -
// W/2)/edge)), coflow weight C = 0.5(1 + tanh((r - W/2 - pilot)/edge)), the
// pilot strip P = 1 - J - C; r = |y - yc|. u = uJet J + uPilot P + uCo C
// (a pulse scales the J part), t = J + tPilot P + tCo C, Y = J. A burnt
// coflow without a strip (tCo = sigma, pilot 0) carries the Shvab-Zeldovich
// enthalpy of the fresh jet exactly: t = sigma - (sigma - 1) Y.
// Pre: b.edge > 0.
struct BunsenInflowRow
{
    double u, t, Y, jet, pilot, co;
};
BunsenInflowRow bunsenInflow(const BunsenGeom& b, double y);

// Signed distance from (x, y) to the ideal cone: the two flanks from
// (0, yc +- W/2) to the tip (coneH, yc), positive on the burnt side
// (outside the cone or past the tip), negative inside the fresh core.
// Pre: b.W > 0, b.coneH > 0.
double bunsenConeDistance(const BunsenGeom& b, double x, double y);

// The cone read off a temperature field T (FP32, cell (i, j) at i + nx j,
// x = (i + 1/2) h, y = (j + 1/2) h, square cells h) at the isotherm Tmid.
//   tip:  the first Tmid crossing along the slot axis (mean of the two
//         columns either side of yc), interpolated; NaN if the axis never
//         crosses (blow-through) or starts hot (flash-back).
//   rows: an axis-cold row's flank crossings y_L, y_R = the first Tmid
//         crossings walking outward from the axis (never across the wrap).
//         With an omega field, a crossing counts only if it BURNS: max
//         omega within `win` cells of it >= 0.1 max omega (a hot pilot or
//         coflow strip crosses Tmid too, without a flame).
//   base: the lowest such row with both crossings; NaN if none.
//   fit:  rows with x in [base + 0.25 (tip - base), base + 0.75 (tip -
//         base)] -- the straight flank, away from tip and base -- least
//         squares of the half-width (y_R - y_L)/2 = a - x tanA, and each side
//         about yc for tanL, tanR; rms = the half-width fit residual.
// iLo..iHi (inclusive) are the fit rows, for the caller's own integrals.
// ok is false when fewer than 4 rows fit (no measurable flank).
// Pre: nx >= 2, ny >= 4 even-split about yc (yc/h - 1/2 integer + 1/2).
struct ConeMeasure
{
    bool ok = false;
    bool axisHot = false; // the axis is at or above Tmid at the inflow row
    double xTip = 0, xBase = 0;
    int iLo = 0, iHi = -1, rows = 0;
    double tanA = 0, tanL = 0, tanR = 0, rms = 0;
    std::vector<int> fi;            // the fit rows' i
    std::vector<double> fyL, fyR;   // and their crossings
};
ConeMeasure measureCone(std::span<const float> T, int nx, int ny, double h,
                        double Tmid, double yc,
                        std::span<const float> omega = {}, int win = 4);

// The kinematic mass burning flux of the fitted flanks: m = rho (u, v) . n at
// each fit row's Tmid crossing (rho, u, v linear between the two cells the
// crossing lies between; n the outward flank normal of that side's fit),
// averaged over both sides. A steady flame passes rho_u S_L through every
// isotherm, so m = 1 in box units for a flank burning at the oracle speed.
// NaN when !m.ok. Pre: fields like measureCone's.
double flankMassFlux(const ConeMeasure& m, std::span<const float> rho,
                     std::span<const float> u, std::span<const float> v,
                     int nx, double h);

} // namespace dns
