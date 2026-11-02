// `dns flame --physics lowmach --solver fv` and `--polymesh DIR --bc FILE`: the
// planar premixed flame on the low-Mach FV solver (gpu/fv_incomp DeviceFvLowMach,
// docs/fvincomp.md "Low-Mach"). GPU-only header.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/fvraster.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/output.hpp"

#include <expected>
#include <string>

namespace dns::gpu {

struct FvLowMachRun
{
    FvLowMachModel model;
    FvIncompControls ctl;
    FvLowMachBcs bcs;       // bound by runFvLowMach
    double cfl = 0.8, chemK = 1, dtMargin = 0.95, tEnd = 10;
    int dtRecompute = 16, statsEvery = 50;
    double profileDt = 0;
    bool quiet = false;
    std::string label;
    const FvFlameProbe* flame = nullptr; // non-null
};

// Stats: gpu/runner_lowmach's premixed columns (t, dt, dtChem, dtDiff, SL, SLdisp, xf,
// dth, Tmax, mass, massTally, omegaEdge, yHits, uFresh, pFresh, TFresh, yMin, yMax,
// pJump, szDefect, divErr, eosErr, pcgIt: the step's PCG iterations), then burn, fuel,
// eDefect (Sum V rho T - its tally). Profiles (--profile-dt): t, x, T, rho, u, p, Y,
// omega slab means. Frames: temp, fuel, omega. Equal steps to tEnd. Exit 8: drift
// within 2 delta_th of either end; a pressure solve that hit its cap warns on stderr.
std::expected<RunStats, RunError> runFvLowMach(const FaceMesh& fm, const FvLowMachIC& ic, FvLowMachRun r,
                                               const FvRaster& raster, RunOutput& out);

} // namespace dns::gpu
