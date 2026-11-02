// The Rayleigh-Benard time loops on a FaceMesh (dns conv --solver fv): both
// physics, one statistics definition, the structured runners' columns. Plane
// means are V-weighted over height bins, so on the uniform box a bin IS a row
// and nuBot, nuTop, nuFlux are runner_conv's and runner_rb's numbers. nuBotF,
// nuTopF: the conservative wall-face heat flux. GPU-only header.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvconv.hpp"
#include "dns/fvincomp.hpp"
#include "dns/fvraster.hpp"
#include "dns/gpu/fv_comp.hpp"
#include "dns/gpu/run_result.hpp"
#include "dns/output.hpp"

#include <expected>
#include <string>
#include <vector>

namespace dns::gpu {

struct FvConvRun
{
    std::string label;
    double H = 1, Tbot = 1, Ttop = 1;
    double kapDiff = 0;   // thermal diffusivity at the reference state: Nu's conduction scale
    double kapFace = 0;   // what a wall face's conduction carries per unit gradient
    double uFf = 1;       // frame scales
    double cfl = 0.4, tEnd = 0;
    int dtRecompute = 16;
    double dtMargin = 0.95;
    int statsEvery = 200; // steps; the Boussinesq loop takes ~40x fewer, run with 50
    double profileDt = 0; // 0 = off
    int bins = 0;         // height bins of the plane statistics
    FvConvPlates plates;
    bool quiet = false;
};

// Pre: ic sized for fm, cfg.gravity set; out made on raster.grid(). Errors: 3 setup,
// 4 output, 6 device, 7 numerical.
std::expected<RunStats, RunError> runFvConvComp(const FaceMesh& fm, const dns::FvState& ic, const FvCompConfig& cfg,
                                                const FvConvRun& r, const FvRaster& raster, RunOutput& out);

// Pre: b bound to fm, U0 [3 nC], T0 [nC], ctl.kappa, ctl.betaG set. Same errors.
std::expected<RunStats, RunError> runFvConvRb(const FaceMesh& fm, const FvIncompBcs& b, const FvIncompControls& ctl,
                                              const std::vector<double>& U0, const std::vector<double>& T0,
                                              const FvConvRun& r, const FvRaster& raster, RunOutput& out);

} // namespace dns::gpu
