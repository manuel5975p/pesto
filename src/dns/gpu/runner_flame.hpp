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
#include "dns/transport.hpp"

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
};

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
