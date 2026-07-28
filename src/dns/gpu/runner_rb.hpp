// The Boussinesq Rayleigh-Benard time loop: incompressible SSP-RK3 with
// projection, a cell-centred temperature advanced on the same stages, and
// buoyancy as the only coupling back into momentum. No-slip isothermal plates
// on y; the lateral faces are periodic or no-slip adiabatic walls (`--sides`).
//
// This is the M -> 0 limit of runner_conv, taken analytically. There is no
// sound speed and therefore no acoustic time step -- dt is set by advection,
// diffusion and the free-fall acceleration -- which is what makes a
// physically faithful air cell affordable here and not there.
//
// FREE-FALL SCALING, and it is not the compressible branch's. H = 1, the
// imposed drop is 1 (plates at +/- 1/2), and g*alpha = 1, so the free-fall
// velocity and time are both exactly 1 and
//   nu = sqrt(Pr/Ra),  kappa = 1/sqrt(Ra Pr).
// There is no Mach number, no stratification and no adiabatic gradient, so --
// unlike the compressible branch -- Ra and Nu are both on the FULL imposed
// drop and no superadiabatic correction exists to disagree about.
//
// GPU-only header.
#pragma once

#include "dns/field.hpp"
#include "dns/gpu/runner_incomp.hpp" // RunStats, RunError
#include "dns/output.hpp"

#include <expected>

namespace dns::gpu {

struct RbRun
{
    double rayleigh = 1e6;
    double prandtl = 0.72;
    double nu = 0;        // sqrt(Pr/Ra), resolved by validate()
    double kappa = 0;     // 1/sqrt(Ra Pr)
    double gAlpha = 1.0;  // free-fall scaling pins this to 1
    double H = 1.0;       // plate separation
    double Tbot = 0.5, Ttop = -0.5, Tref = 0.0;
    // Side fraction of a centred hot patch on the FLOOR; 1 = the whole plate
    // (bit-for-bit the uniform case). Outside the patch the floor sits at the
    // lid temperature. With patch < 1 the reported nuBot is the net
    // FLOOR-AREA-AVERAGED heat flux -- the cold surround absorbs heat, so it
    // is no longer a classical Nusselt number and the Ra-Nu scaling does not
    // apply to it.
    double patch = 1.0;
    // Lateral faces: false = periodic (an unbounded layer sampled over one
    // box), true = no-slip ADIABATIC side walls on x, and on z when nz > 1.
    // It also selects the pressure problem -- PeriodicXZWallY vs the all-
    // Neumann box -- because a closed side is a zero-flux face for p as well.
    bool closedSides = false;
    double cfl = 0.8;
    double tEnd = 0;      // in free-fall times, which here ARE time units
    int statsEvery = 50;
    double profileDt = 0;
};

// `ic` is the linear conduction profile plus the trip; `icU` its velocity
// (at rest but for the trip). The wall BC is applied before the first step.
std::expected<RunStats, RunError>
runRb(const Grid&, const VectorField& icU, const Scalar& icT, const RbRun&,
      RunOutput&);

} // namespace dns::gpu
