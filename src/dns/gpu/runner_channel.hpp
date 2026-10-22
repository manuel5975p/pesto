// The compressible channel-flow time loop: periodic x/z, no-slip isothermal
// y walls on the ChannelStretchY curvilinear mesh, constant streamwise body
// force. Wall units: delta = 1, uTau = 1, f = 1, nu = 1/reTau -- the box
// momentum balance then pins tau_wall = 1 identically, so reTau is the run's
// friction Reynolds number by construction (in the mean).
//
// GPU-only header.
#pragma once

#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/gpu/run_result.hpp" // RunStats, RunError
#include "dns/output.hpp"
#include "dns/transport.hpp"

#include <expected>

namespace dns::gpu {

struct ChannelRun
{
    double gamma = kGammaAir, Rgas = kRgasBox;
    double mu = 0;              // = rho nu = 1/reTau in wall units
    double reTau = 180;
    double ubEst = 0;           // log-law bulk-velocity estimate (validate())
    // Streamwise body-force density. NOT configurable and deliberately so:
    // the channel runs in wall units (delta = 1, uTau = 1, nu = 1/reTau), and
    // the box momentum balance pins tau_wall = f*delta = 1. Changing it would
    // silently redefine what --re-tau means.
    static constexpr double fBody = 1.0;
    double delta = 1.0;         // channel half-height
    double pRef = 1.0;          // background pressure (sets the bulk Mach)
    double Twall = 1.0;         // isothermal wall temperature
    bool isothermal = true;
    double stretchBeta = 2.0;   // ChannelStretchY clustering
    double cfl = 0.4;
    double tEnd = 30;           // eddy turnovers; delta/uTau = 1
    int dtRecompute = 16;
    double dtMargin = 0.95;
    int statsEvery = 200;       // steps between stats rows
    double profileDt = 1.0;     // y-profile dump interval; 0 = off
    double curlMax = 0;         // frame vorticity scale; 0 = reTau/4
};

// `ic` is the initial conservative state (perturbed laminar profile); the
// ghost rows are overwritten by the wall mirror before the first step.
std::expected<RunStats, RunError>
runChannel(const Grid&, const CompState& ic, const ChannelRun&, RunOutput&);

} // namespace dns::gpu
