// The `dns flame` time loop (docs/chem.md): a 1D freely propagating premixed
// flame in an open x box -- NSCBC inflow at x-lo carrying fresh gas (Y = 1)
// at the inflow speed, NSCBC outflow at x-hi anchored at the burnt pressure,
// y periodic, the scale-selective sink under both faces. Uniform mesh only.
//
// Order inside the rhs hook, and why: NSCBC overwrites continuity on the
// open-face cells -> the species re-carry rescales rhoY there by the density
// the mean flow just produced (it must see that density before anything else
// writes rho) -> the sink -> the species twin of the sink (same band operator
// on rhoY, so a uniform Y passes the band untouched).
//
// GPU-only header.
#pragma once

#include "dns/bunsen.hpp"
#include "dns/comp.hpp"
#include "dns/diffusion.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/output.hpp"
#include "dns/sgs.hpp"
#include "dns/tfles.hpp"
#include "dns/transport.hpp"

#include <array>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns::gpu {

struct FlameRun
{
    Transport tr;            // mu = Pr at Tref = Tu (box units)
    SpeciesSetup sp;         // nY 1, fuel 0, openX, yIn = 1
    double Tu = 1, sigma = 5;
    double uIn = 1;          // inflow speed (the NSCBC target), / S_L
    double pB = 1;           // outflow pressure target: the IC's burnt p
    double etaIn = 100;      // NSCBC inflow relax
    // NSCBC outflow pressure anchor (Config::sigma). 0.25 = the shared
    // default (1D and bunsen keep it). The diffusion jet's dilatation held
    // the box at p/p0 1.67 under 0.25 (relaxation L/(sigma c) ~ 20 l/U);
    // dns_main sets kDiffOutflowSigma there.
    double outSigma = 0.25;
    bool thermalInflow = false; // entropy inflow holds T = Tu, not rho = 1
    int sinkNb = 16;         // 0 = no sink
    double sinkAmp = 1;
    double dth = 1;          // oracle thermal thickness (drift guard)
    double cfl = 0.4;
    double tEnd = 10;
    int dtRecompute = 16;
    double dtMargin = 0.95;
    int statsEvery = 200;
    double profileDt = 0;
    std::string label;       // flameRunLabel(), printed with the progress line

    // --mode bunsen (dns/bunsen.hpp). bunsen == false is the 1D runner bit
    // for bit. Per-row x-lo inflow at j + ny k: the u target, the jet weight
    // the pulse scales, the entropy target (rho, or 1/(R T) with
    // thermalInflow) -- DeviceCompNscbc::Config::inflow* -- and rho, Y of the
    // row for the nominal fuel inflow sum rho u Y (the fuel row itself rides
    // sp.yInRow).
    bool bunsen = false;
    BunsenGeom geom;
    std::vector<double> rowU, rowJet, rowTarget, rowRho, rowY;

    // --mode diffusion (dns/diffusion.hpp): the bunsen per-row inflow
    // machinery (geom with uJet = 1, rowU/rowJet/rowTarget/rowRho; rowY the
    // fuel, rowYO the oxidizer, both on sp.yInRow), sp.ox = 1, the jet stats
    // (measureJet) and the extinction guard. Exclusive with bunsen.
    bool diffusion = false;
    DiffusionModel diff;
    std::vector<double> rowYO;

    // --mode candle (diffusion == true): gravity g along -x through THE
    // source (addGravity, axis 0, after the NSCBC overwrite), the NSCBC
    // hydrostatic waves hydroLo/Hi (DeviceCompNscbc::Config), pB = the
    // ambient p_a at the outflow row. gravity == 0 launches neither: the
    // diffusion runner bit for bit. Candle stats append uMax (max |u| over
    // the box), uOutMin (min u on the outflow row: backflow < 0) and pMean is
    // normalised by the ambient column's mean. quiescent: no flame guard.
    bool candle = false, quiescent = false;
    double gravity = 0, hydroLo = 0, hydroHi = 0;

    // 3D premixed in turbulence (--turb; docs/chem.md, "3D and turbulence").
    // turbBox = turbBox(flameTurbSpec) (turbNbx x ny x nz, i fastest); empty
    // = no turbulent inflow, the laminar runner bit for bit. Each step the
    // inflow face samples the box at the swept distance X = sum u_in dt
    // (gpu/flame_turb.hpp -> DeviceCompNscbc::setTurbulentInflow). uPrime and
    // lt (box units) set the uPrime stats slab and the frame scales. hold: the
    // inflow-speed controller (kHold*) holding the fresh-mass thickness at its
    // first-row value. At nz > 1 the stats append area, I0, uPrime, uInflow,
    // xFresh, xLead, the frames add vort, speed, and exit 8 fires when the
    // brush's leading edge reaches the inflow band (flash-back).
    // Bunsen at nz > 1 (docs/chem.md, "3D turbulent slot Bunsen"): rows at
    // j + ny k; turbBox is the box times the row's jet weight J (dns_main);
    // the sweep runs at U_jet (+ pulse); the cone is measured on span-mean
    // fields; stats append ST, area, I0; frames add vort, speed.
    std::array<std::vector<double>, 3> turbBox;
    int turbNbx = 0;
    double uPrime = 0, lt = 0;
    bool hold = false;

    // LES (--les CS --sgs M; premixed, nz > 1; docs/chem.md, "LES species and
    // TFLES"): the SGS stress and eddy conductivity of DeviceComp, the species
    // SGS flux mu_t/Sc_t (Sc_t = Pr_t), and mu_t on the open x rows copied
    // from their neighbours (mutCopyOpenXHook). csLes == 0 = the DNS runner
    // bit for bit.
    double csLes = 0;
    SgsModel sgs = SgsModel::Sigma;
    // TFLES (--tfles F): gpu/tfles updated from each step's input state, its
    // F E, E/F, Omega handed to the species kernels; the diffusive dt bound
    // tflesStableDt min-ed in; stats append Ebar (the Omega-weighted mean E)
    // and FEmax. false = the runner bit for bit.
    bool tflesOn = false;
    TflesSetup tfles;
};

// --hold-flame: at every stats row u_in <- S_ema - (x_I - x_I0)/kHoldTau,
// x_I = integral of rho Y dx / (rho_u Y_u) the fresh-mass thickness (exactly
// dx_I/dt = u_in - S_c: inflow minus consumption, whatever the brush's
// shape), x_I0 its first-row value, S_ema the consumption speed's moving
// average over kHoldEmaTau. Rate-limited to kHoldMaxRate * (time since the
// last row), clamped to [kHoldUMin, kHoldUMax]. Units: flame times, S_L.
// MEASURED (2026-10-29, u'_in 4, 6 cells): holding the plane-mean T crossing
// x_f instead failed -- the brush skewed, x_f moved DOWNSTREAM while S_c rose
// 2.2 -> 3, the controller cut u_in to 1.16 and the leading edge flashed back
// into the inflow band (S_c 8, I0 2.2). Constants chosen, not bracketed.
inline constexpr double kHoldTau = 2.0;
inline constexpr double kHoldEmaTau = 1.0;
inline constexpr double kHoldMaxRate = 1.0;
inline constexpr double kHoldUMin = 0.5, kHoldUMax = 10.0;
// The 3D flash-back guard: exit 8 when the plane-mean progress c = (T - Tu)/
// (Tb - Tu) first reaches kLeadC within one delta_th of the inflow band.
inline constexpr double kLeadC = 0.05;

// `ic` + `rhoY` (one vector per species, sp.nY of them) from flameIC (1D),
// bunsenIC or diffusionIC. 1D: exit 8 when the flame drifts within band + 2
// delta_th of either face. Bunsen: exit 8 when the cone is not anchored --
// the slot axis starts hot (flash-back), never crosses Tmid (blow-through),
// the tip reaches band + 2 delta_th of the outflow, or the base lifts past
// half the tip height. Diffusion: exit 8 when the flame goes out (Tmax below
// the mid temperature (1 + sigma) T0 / 2) or lifts off past half the box. A
// yHits > 0 at the end is a warning line with the overshoot magnitude.
std::expected<RunStats, RunError> runFlame(
    const Grid&, const CompState& ic,
    std::span<const std::vector<double>> rhoY, const FlameRun&, RunOutput&);

} // namespace dns::gpu
