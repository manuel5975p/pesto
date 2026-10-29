// The `dns flame --mode candle --wick` time loop (docs/chem.md, "Wick
// candle"): the candle's diffusion flame (two reactants, gravity -x, the
// hydrostatic NSCBC box) around an IMMERSED candle body + wick -- one
// isothermal DeviceCompGhost (wick at the vaporisation temperature, body at
// T0, GhostWallThermal::TwallAt), its pressure image hydrostatic (the wall
// "acceleration" +g x: grad p = -rho g at the wall) -- with the fuel vapour
// injected through a source band beside the wick and (optionally) soot as a
// third species. The slot candle (runner_flame) is untouched.
//
// Order inside the rhs hook: NSCBC -> gravity -> species re-carry -> inflow
// pin -> sink -> species sink -> wall faces (applyRhs) -> fuel source. Stage
// hook: the ghost fill.
//
// GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/diffusion.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/output.hpp"
#include "dns/transport.hpp"
#include "dns/wick.hpp"

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace dns::gpu {

struct WickRun
{
    Transport tr;           // mu = Pr at T0 (box units)
    SpeciesSetup sp;        // fuel 0, ox 1, soot 2 (or -1), openX, per-row
                            // air inflow, ambientHi; solid set by the runner
    DiffusionModel diff;
    WickSpec spec;
    std::vector<std::uint8_t> regions; // wickRegions(g, spec)
    double T0 = 1, gravity = 0;
    double pOut = 1;        // outflow target p_a(x_out)
    double hydroLo = 0, hydroHi = 0;
    double uCo = 1;         // air coflow from below
    double etaIn = 100;
    bool thermalInflow = false;
    int sinkNb = 16;
    double sinkAmp = 1;
    double cfl = 0.4, tEnd = 10;
    int dtRecompute = 16;
    double dtMargin = 0.95;
    int statsEvery = 200;
    // Per-row x-lo inflow (DeviceCompNscbc::Config::inflow*): air at uCo.
    std::vector<double> rowU, rowJet, rowTarget;
    std::string label;      // wickRunLabel()
};

// `ic` + `rhoY` (sp.nY vectors) from wickIC. Exit 7 on a non-finite state, 8
// when the flame goes out (max fluid T below (1 + sigma) T0 / 2 after t = 2).
std::expected<RunStats, RunError> runWick(const Grid&, const CompState& ic,
                                          std::span<const std::vector<double>> rhoY,
                                          const WickRun&, RunOutput&);

} // namespace dns::gpu
