// The time loop of the compressible FV family (docs/fvcomp.md): quiet-start
// ramp, SSP-RK3 under the Sigma dt, force and budget rows, frames on a uniform
// plane through the mesh. One loop for every FaceMesh. GPU-only header.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvcomp_case.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvraster.hpp"
#include "dns/gpu/fv_comp.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/output.hpp"

#include <expected>
#include <string>

namespace dns::gpu {

struct FvCompRun
{
    std::string label;      // the run label: stats comment, first output line
    FvCompConfig solver;    // gas, table at frac = run start, precision, LES, tally
    FvBcSpec spec;          // the patches, for the ramp's table updates
    FvFreestream fs;
    double D = 1, refArea = 1;   // force coefficient reference area; frame scales
    double cfl = 0.4;
    double tEnd = 0;             // ramp included
    double ramp = 0;             // > 0: ic at rest, the freestream raised by Galilean boosts
    int dtRecompute = 16;
    double dtMargin = 0.95;
    int statsEvery = 200;
    double curlMax = 0;          // 0 = 8/D signed, 15/D magnitude
    bool vortAsinh = false, speedDelta = false;
    bool quiet = false;
    // > 0: --steady TOL, local-time-stepping pseudo-time, tEnd unused; stops on
    // steadyFloor over cd, cdf (observables) and the monitors kinetic energy +
    // the scalars' volume integrals. Exit 8 past steadySteps.
    double steadyTol = 0;
    long steadySteps = 0;
    // dns flame (solver.chem.nY > 0): the chemistry dt joins the bound, the stats
    // are the flame's (FvFlameProbe, the structured runner's columns), frames temp,
    // fuel, omega; exit 8 when x_f drifts within 2 delta_th of either end. Null: off.
    const FvFlameProbe* flame = nullptr;
    double profileDt = 0;        // flame: --profile-dt, the `profiles` series of slab means
    // dns flame --mode diffusion: the same chemistry dt, the structured jet's columns
    // (FvJetProbe, fuelIn / fuelOut the open patches' species flux), frames temp, fuel,
    // oxid, omega; exit 8 on extinction (TmaxIn below the mid T) or lift-off past half
    // the mesh. Null: off.
    const FvJetProbe* jet = nullptr;
};

// Pre: ic sized for fm; out made on raster.grid(). Errors: 3 setup (mesh and
// boundary table), 4 output, 6 device, 7 numerical.
std::expected<RunStats, RunError> runFvComp(const FaceMesh& fm, const dns::FvState& ic, const FvCompRun& r,
                                            const FvRaster& raster, RunOutput& out);

} // namespace dns::gpu
