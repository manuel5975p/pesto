// `dns flame --physics lowmach --solver fv` and `--polymesh DIR --bc FILE`: the
// planar premixed flame on the low-Mach FV solver (gpu/fv_incomp DeviceFvLowMach,
// docs/fvincomp.md "Low-Mach"). GPU-only header.
#pragma once

#include "dns/bunsen.hpp"
#include "dns/checkpoint.hpp"
#include "dns/chem.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/fvraster.hpp"
#include "dns/fvturb.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/output.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <functional>
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
    // Model C2 (--equivalence): the fresh fuel (the SZ defect's H = T + q/cp Y_F at
    // the inflow) and the mixture fraction Z = (s Y_F - Y_O + yO2)/zDen of species 0,
    // 1, read as dzMin, dzMax against the inflow's zIn. premix2 false: C0's columns.
    bool premix2 = false;
    double yFresh = 1, zS = 0, zO2 = 0, zDen = 1, zIn = 0;
    // --dt-rule cell: dt also <= posK / max_c (2 C_c + D_c); 0 = off.
    double posK = 0;
    // --mode bunsen (2D slot): per-face inflow rows on `patch` (row of the face centre and
    // its owner's srcCell), gpu/runner_lowmach's cone columns measured on a uniform plane
    // of pixel h whose lower corner is lo (the inflow plane, slot centre at lo + yc):
    // pixel = cell on the flags' box. Per unit span (depth).
    struct Burner
    {
        bool on = false;
        std::string patch;
        std::function<FvLmRow(const std::array<double, 3>& cf, std::int64_t ownerSrc)> row;
        BunsenGeom geom;
        PartialPremixModel premix; // premix2's
        std::array<double, 2> lo{};
        double h = 1, depth = 1;
        // --rim on the box: the tube is the `rim` Wall patch; xRimBase, rBase, qRim
        // columns and the rim's anchoring guards (x above geom.xRim).
        bool rim = false;
    } burner{};
    // --checkpoint-dt / --wall-limit / --restart (dns/checkpoint.hpp, solver
    // kFvLowMachSolverName; DeviceFvLowMach::save's words). Inactive: the loop bit for bit.
    CheckpointPolicy ckpt{};
    // --turb (M5): the box swept past the faces of inflow patch turbPatch, their bU and
    // rUb rewritten before every step (fvLowMachTurbRows at X = sum turbU dt of the
    // step's start, FP64 host), held over its stages. Null: off.
    const FvTurb* turb = nullptr;
    std::string turbPatch{};
    double turbU = 1;
};

// Stats: gpu/runner_lowmach's premixed columns (t, dt, dtChem, dtDiff, SL, SLdisp, xf,
// dth, Tmax, mass, massTally, omegaEdge, yHits, uFresh, pFresh, TFresh, yMin, yMax,
// pJump, szDefect, divErr, eosErr, pcgIt: the step's PCG iterations), then burn, fuel,
// eDefect (Sum V rho T - its tally), with premix2 dzMin, dzMax. Profiles (--profile-dt): t, x, T, rho, u, p, Y,
// omega slab means. Frames: temp, fuel, omega. Equal steps to tEnd. Exit 8: drift
// within 2 delta_th of either end; a pressure solve that hit its cap warns on stderr.
// Burner: gpu/runner_lowmach's burner columns, + its open (an `open` patch: mdotIn,
// mdotOut, mdotLat, uLatMin, uLatMax, C0 fuelOut) and rim columns. M4: LES (model.cs)
// appends mutMax, TFLES (model.tfles) FEmax, Ebar (DeviceFvLowMach::eddyStats).
std::expected<RunStats, RunError> runFvLowMach(const FaceMesh& fm, const FvLowMachIC& ic, FvLowMachRun r,
                                               const FvRaster& raster, RunOutput& out);

} // namespace dns::gpu
