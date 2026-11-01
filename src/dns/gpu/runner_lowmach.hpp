// The `dns flame --physics lowmach` time loop (masterplan real_bunsen S4;
// docs/chem.md "Low-Mach"): the 1D/2D/3D freely propagating premixed flame on
// the zero-Mach solver (gpu/lowmach.hpp) -- inflow FACE at x-lo, p = 0
// outflow at x-hi, y/z periodic (the burner: or OPEN, S6), no NSCBC, no sink
// band.
//
// GPU-only header.
#pragma once

#include "dns/bunsen.hpp"
#include "dns/checkpoint.hpp"
#include "dns/chem.hpp"
#include "dns/gpu/lowmach.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/lowmach.hpp"
#include "dns/output.hpp"

#include <expected>
#include <string>

namespace dns::gpu {

struct LowMachRun
{
    LowMachSetup sp;         // the solver's model, inflow and transport
    double sigma = 5;        // the burnt premix's Tb/Tu (Tmid, frame scale)
    double uIn = 1;          // inflow speed / S_L
    double dth = 1;          // oracle thermal thickness (guard, dth column)
    double cfl = 0.8;
    double chemK = kLmChemDtDefault; // chemistry dt K / max J (--chem-dt)
    // --dt-rule cell (S9, issues LM10): dt also <= posK / max_c (2 C_c +
    // D_c) (DeviceLowMach::positivityDt); false = the S4-S8 bounds alone,
    // bit for bit. Stats gain a dtPos column.
    bool dtCell = false;
    double posK = kLmPosKDefault;
    double tEnd = 10;
    int dtRecompute = 16;
    double dtMargin = 0.95;
    int statsEvery = 50;
    double profileDt = 0;
    std::string label;       // flameRunLabel()
    // --equivalence (model C2): SL's fresh reference is rho (Y_F - yFBurnt)
    // (runner_flame's rule); dzMin/dzMax (Z - Z_inflow) appended.
    double yFBurnt = 0;
    bool premix2 = false;
    // --mode bunsen (masterplan real_bunsen S5): the burner (the per-row
    // inflow is sp.rowM/rowT/rowY, rows j + ny k), the two-reactant model
    // for Z and Z_st (premix2).
    bool bunsen = false;
    BunsenGeom geom;
    PartialPremixModel premix;
    // --reaction off on the burner (S6): no chemistry (sp.lnA = -inf), the
    // anchoring guards off (there is no cone).
    bool inert = false;
    // --rim (S8): the immersed tube (sp.solid, sp.wallT; its geometry in
    // geom.rimLip / xRim): the rim columns and the rim's anchoring guards.
    bool rim = false;
    // Checkpoint / restart (dns/checkpoint.hpp; solver "flame-lowmach"):
    // runner_flame's rules -- a checkpoint at the END of a step at the first
    // boundary at or past k every, never at tEnd; wall-limit and SIGINT stop
    // there.
    CheckpointPolicy ckpt;
};

// Stats columns: t, dt, dtChem, dtDiff, SL, SLdisp, xf, dth, Tmax, mass,
// massTally, omegaEdge, yHits, uFresh, pFresh, TFresh, yMin, yMax, pJump
// (pFresh - p of the last column: the flame's dynamic pressure drop), szDefect
// (max |H - H_in|, H = T + (q/cp) Y_F), divErr, eosErr, pcgIt (0: the direct
// solve), then dzMin, dzMax (the mixture fraction's range minus the inflow
// premix's Z) with --equivalence. Profiles (--profile-dt): t, x,
// T, rho, u, p, Y, omega (column means). Exit 8 when the flame drifts within
// 2 delta_th of either face.
//
// --mode bunsen (S5): stats t, dt, dtChem, dtDiff, xTip (the axis' first
// Tmid crossing: the inner cone), xBase, tanA, tanL, tanR, rms, sinRatio =
// sin(alpha) U_jet, SLflank, mFlank (measureCone / flankMassFlux, as
// runner_flame), burn = integral omega, fuelIn (the inflow FACE's exact fuel
// flux, convective + diffusive), fuelNom (rho u Y of the rows), burnRatio
// (burn/fuelIn; --equivalence burn/(fuelIn - fuelOut)), Tmax, mass,
// massErr ((mass - FP64 tally)/tally), yHits (Y_F and Y_O outside [-1e-6,
// 1 + 1e-6]), yMin, yMax (Y_F), divErr, eosErr, pcgIt, uMax; --equivalence
// adds xTipOut (the axis' first downward Z = Z_st crossing: the outer
// cone; NaN = open), fuelOut, burnPre = integral omega w_p(xi), burnDif =
// integral omega (1 - w_p), preFrac = burnPre/burn, zMin, zMax; nz > 1 adds
// ST (burn over the ideal cone's area at the measured tip and base). The
// slot (2D, 3D) is measured on SPAN-MEAN planes, every integral per unit
// span (nz 8 extrusion = 2D bitwise); the round nozzle's cone on the axial
// plane, integrals over the box. Frames: temp, fuel, [oxid,] omega,
// [findex,] speed. Exit 8: flash-back, blow-through, blow-off (runner_flame's
// guards, 2 delta_th from the outflow; none when inert).
//
// Open laterals (S6, sp.openY): the bunsen stats add mdotIn, mdotOut,
// mdotLat (net in through the laterals), uLatMin, uLatMax (outward normal
// velocity on the lateral faces; DeviceLowMach::boundarySums), and C0 a
// fuelOut column; fuelOut (both models) is then x-hi + the laterals with the
// ambient rule, burnRatio = burn/(fuelIn - fuelOut). The slot's fluxes per
// unit span. --profile-dt on the burner: the "jet" series per x column (t,
// x, uc, Tc, Yc, J, M, Q, b; runLowMach's jetProfiles).
//
// --rim (S8): pcgIt = the most PCG iterations of the last step's three
// stage solves (else 0: the direct solve); the bunsen stats add xRimBase
// (xBase - xRim: the flame base's stand-off above the rim), rBase (the base
// row's flank half-width on the cone plane: where the base sits radially),
// qRim (lmWallHeat: the conduction into the tube, per unit span on the
// slot). Exit 8 adds: the base below the rim (flash-back into the tube);
// blow-off when xRimBase passes half the cone height above the rim. A PCG
// that does not converge is exit 8.
//
// --dt-rule cell (S9): every stats series appends dtPos (the positivity
// bound of the last dt recompute, before the margin).
std::expected<RunStats, RunError> runLowMach(const Grid&, const LowMachState& ic,
                                             const LowMachRun&, RunOutput&);

} // namespace dns::gpu
