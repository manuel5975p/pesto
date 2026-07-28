// The Rayleigh-Benard time loop: no-slip ISOTHERMAL y walls at two different
// temperatures on the uniform mesh, periodic or closed (no-slip adiabatic)
// lateral faces, gravity in -y. The heated
// floor is the only forcing; there is no mean flow and no freestream.
//
// The solver is fully compressible, so buoyancy needs no model: gravity acts
// on the density the equation of state already gives, and the run is
// stratified (the hydrostatic density scale height is R Tref / g). Two
// consequences the caller must respect and validate() enforces:
//
//   * the layer only convects on the SUPERADIABATIC part of the imposed
//     temperature drop, dTsuper = dT - (g/cp) H (the Schwarzschild criterion),
//     so `rayleigh` is defined on dTsuper and dTsuper > 0 is a hard input
//     requirement;
//   * the Nusselt numbers are reported on the FULL imposed dT, so the
//     quiescent conduction state reads exactly Nu = 1.
//
// GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/gpu/runner_incomp.hpp" // RunStats, RunError
#include "dns/output.hpp"

#include <expected>

namespace dns::gpu {

struct ConvRun
{
    double gamma = 1.4, Rgas = 1.0;
    double mu = 0;              // dynamic viscosity at Tref (rho_ref = 1)
    double Pr = 0.72;
    double tinfK = 288.15;      // Sutherland anchor; read only when sutherland
    bool sutherland = false;
    double gravity = 0;         // magnitude; the vector is (0, -gravity, 0)
    double H = 1.0;             // layer height, wall face to wall face
    double Tbot = 1, Ttop = 1;  // isothermal wall temperatures, Tbot > Ttop
    // Side fraction of a centred hot patch on the floor; 1 = the whole plate.
    // With patch < 1 the reported nuBot is the net FLOOR-AREA-AVERAGED flux
    // (the cold surround absorbs heat), not a classical Nusselt number.
    double patch = 1.0;
    // Lateral faces: false = periodic, true = no-slip ADIABATIC walls on x, and
    // on z when nz > 1 (`--sides closed`). Periodic is bit-for-bit the solver
    // before the flag existed.
    bool closedSides = false;
    double Tref = 1;            // mid-layer reference temperature
    double rayleigh = 0;        // superadiabatic Ra (diagnostics + manifest)
    double uFf = 0;             // free-fall velocity, sqrt(g dTsuper/Tref H)
    double cfl = 0.4;
    double tEnd = 0;            // in time units; the app converts --tt free-fall times
    int dtRecompute = 16;
    double dtMargin = 0.95;
    int statsEvery = 200;       // steps between stats rows
    double profileDt = 0;       // y-profile dump interval; 0 = off
};

// `ic` is the hydrostatic conduction state plus the solenoidal trip; the ghost
// rows are overwritten by the wall mirror before the first step.
std::expected<RunStats, RunError>
runConv(const Grid&, const CompState& ic, const ConvRun&, RunOutput&);

} // namespace dns::gpu
