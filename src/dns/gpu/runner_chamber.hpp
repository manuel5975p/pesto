// The `dns flame --mode chamber` time loop (docs/chem.md, "Chamber"): the
// premixed single-step flame of runner_flame inside a closed 2D chamber.
//
// Pieces, all on the uniform 2D mesh:
//  - the chamber + holder are ONE immersed adiabatic ghost wall
//    (DeviceCompGhost, stage hook apply(), rhs hook applyRhs(): impermeable
//    faces), and the species stage runs its walled instantiation
//    (SpeciesSetup::solid): no species flux through a fluid/solid face.
//  - the injector is a volumetric mass-flow source on the injector-band cells:
//    rho += S, E += S cp Tu (the injected gas's enthalpy at rest), rhoY += S
//    (Y = 1), S uniform per cell, total = the requested mdot times a smooth
//    ramp, held for the whole step (an SSP-RK3 step then adds exactly
//    S dt).
//  - the ambient: x NSCBC (cold inert inflow at the coflow, outflow anchored
//    at p0), sink + species sink, y periodic.
// Rhs hook order: NSCBC -> species re-carry -> sink -> species sink -> wall
// faces -> injector (the last two are additive on cells disjoint from the
// boundary rows).
//
// GPU-only header.
#pragma once

#include "dns/chamber.hpp"
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/output.hpp"
#include "dns/transport.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace dns::gpu {

struct ChamberRun
{
    Transport tr;             // mu = Pr at Tref = Tu (box units)
    SpeciesSetup sp;          // nY 1, openX, yIn 0 (inert ambient); solid set here
    double Tu = 1, sigma = 5; // Tu = p0
    double uCo = 2;           // ambient coflow (NSCBC inflow target)
    double etaIn = 100;
    bool thermalInflow = false;
    int sinkNb = 16;
    double sinkAmp = 1;
    ChamberSpec spec;
    std::vector<std::uint8_t> regions; // chamberRegions(g, spec)
    double mdot = 0;          // injected mass per unit time and depth
    double ramp = 1;          // sin^2 ramp length; 0 = full rate at t = 0
    bool reaction = true;     // guards (flash-back / blow-off) only when on
    double cfl = 0.4, tEnd = 60;
    int dtRecompute = 16;
    double dtMargin = 0.95;
    int statsEvery = 20;
    std::string label;
};

// The injection rate factor at time t: sin^2(pi t / (2 ramp)) below ramp, 1
// after (ramp <= 0: 1). Host and test share it.
double chamberRamp(double t, double ramp);

// `ic` + `rhoY` from chamberIC. Exit 8 on flash-back or blow-off (reaction
// on), 7 on a non-finite state, 3 when the ghost classification disagrees
// with r.regions about which cells are solid.
std::expected<RunStats, RunError> runChamber(const Grid&, const CompState& ic,
                                             const std::vector<double>& rhoY,
                                             const ChamberRun&, RunOutput&);

} // namespace dns::gpu
