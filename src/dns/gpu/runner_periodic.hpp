// The two PERIODIC time loops: 3D Taylor-Green and forced homogeneous
// isotropic turbulence. No boundary conditions, no body, no projection wall --
// which is exactly why they are separate from runner_{incomp,comp,channel}.
//
// The app owns dispatch, resource acquisition and the Case -> RunSpec
// mapping; every time loop is in gpu/runner_*.
#pragma once

#include "dns/gpu/runner_incomp.hpp"   // RunError, RunStats
#include "dns/hit_ic.hpp"
#include "dns/output.hpp"

#include <expected>
#include <functional>

namespace dns::gpu {

class PeriodicFrames;

struct TgvRun
{
    double nu = 0;
    double cfl = 0.9;
    double tEnd = 0;
    double outEvery = 0.1;   // dissipation-series cadence, in t
    double volume = 1.0;     // box volume; K is reported per unit volume
};

// 3D Taylor-Green. Writes the `t,K,dKdt,epsVisc` series to STDOUT -- that is
// the documented output of `dns tgv` (a dissipation time series in the format
// the reference curves use), not a diagnostic. `frames` may be null.
std::expected<RunStats, RunError>
runTgv3d(const Grid& g, const VectorField& ic, const TgvRun& r,
         PeriodicFrames* frames);

struct HitRun
{
    double nu = 0;
    double epsF = 0;       // forcing dissipation target
    double kFloor = 0;     // linear-forcing floor, k_stationary/5
    double tEnd = 0;
    double tSpin = 0;      // statistics start after this
    double tEddy = 0;      // one eddy turnover, box units
    int statsEvery = 10;   // steps between stats rows
    bool spectrum = false; // --emit spectrum
    bool snapshot = false; // --emit snapshot
    double lBoxSI = 0;     // carried into the snapshot metadata
    std::function<double(double)> timeToSI;   // ditto
};

// Stationary averages over the post-spin-up window, in box units.
struct HitAverages
{
    double epsVisc = 0, kPrime = 0, reLambda = 0;
};

// Forced HIT with the fixed-eps linear-forcing controller
// (Carroll-Blanquart): stationary <eps_visc> = eps_f. `frames` may be null.
// Every artefact -- stats rows, spectra, snapshots -- goes through `out`.
std::expected<HitAverages, RunError>
runHit(const Grid& g, const VectorField& ic, const HitRun& r, RunOutput& out,
       PeriodicFrames* frames, RunStats& st);

} // namespace dns::gpu
