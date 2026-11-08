// The time loop of the compressible FV family (docs/fvcomp.md): quiet-start
// ramp, SSP-RK3 under the Sigma dt, force and budget rows, frames on a uniform
// plane through the mesh. One loop for every FaceMesh. GPU-only header.
#pragma once

#include "dns/checkpoint.hpp"
#include "dns/chem.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvcomp_case.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvnozzle.hpp"
#include "dns/fvraster.hpp"
#include "dns/gpu/fv_comp.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/output.hpp"
#include "dns/turb_ic.hpp"

#include <expected>
#include <optional>
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
    // flame, --equivalence (models C2, C2b): the stats append zMin, zMax, the premix's
    // mixtureFraction(Y_F, Y_O) over the cells (runner_flame's columns). Empty: off.
    std::optional<PartialPremixModel> premix{};
    // dns flame --mode diffusion: the same chemistry dt, the structured jet's columns
    // (FvJetProbe, fuelIn / fuelOut the open patches' species flux), frames temp, fuel,
    // oxid, omega; exit 8 on extinction (TmaxIn below the mid T) or lift-off past half
    // the mesh. Null: off.
    const FvJetProbe* jet = nullptr;
    // dns flame --checkpoint-dt / --wall-limit / --restart (dns/checkpoint.hpp, solver
    // kFvFlameCompSolverName): the time-accurate loop only, ramp 0. Inactive: the
    // loop bit for bit.
    CheckpointPolicy ckpt{};
    // --turb (M5, solver.turb): the inflow Far patch's per-face rows, rows turbFirst ..
    // of the table (solver.bcRow), one per sample point of turbSweep (the patch's face
    // centres, boundary order), rewritten before every step from the box swept to X =
    // sum fs.U dt (FP64 host; fvFarTurbRow), held over its stages. Null: off.
    const TurbSweep* turbSweep = nullptr;
    const FvPatchSpec* turbPatch = nullptr;
    int turbFirst = 0;
    // A nozzle on a Total inlet (docs/nozzle.md): the ramp raises the Total row's p0, no
    // boost; the stats append mdotIn mdotOut Fwall Fin Fexit Cd Cf. Empty: off.
    std::optional<FvNozzleRun> nozzle{};
    // --shock hll: the sticky dt cadence (internal::shockDtEvery), unless --dt-recompute.
    bool shockDtAuto = false;
    // > 0: --foam-write, PREFIX_foam/<t>/{p,T,U,rho} every foamWrite, source cell order.
    double foamWrite = 0;
    std::string foamPrefix{};
};

// Pre: ic sized for fm; out made on raster.grid(); ckpt active only with ramp 0
// and steadyTol 0. Errors: 3 setup (mesh and boundary table) and a --restart
// checkpoint that does not fit, 4 output, 6 device, 7 numerical.
std::expected<RunStats, RunError> runFvComp(const FaceMesh& fm, const dns::FvState& ic, const FvCompRun& r,
                                            const FvRaster& raster, RunOutput& out);

} // namespace dns::gpu
