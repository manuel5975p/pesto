// The explicit local-time-stepping steady march of --steady (docs/rans.md,
// "Stopping on the integrals"): localDt every dtRecompute iterations, one
// SSP-RK3 step at the per-cell dt, the runner's boundary refresh, and every
// statsEvery iterations a sample of the runner's stop series into SteadyStop.
// Shared by every runner that marches in pseudo-time; the runner supplies its
// integrals and its stats row, so the physics differences stay in the runner
// (gpu/internal/run_loop.hpp says why the loops are otherwise not merged).
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/gpu/comp.hpp"
#include "dns/gpu/internal/run_loop.hpp"
#include "dns/interrupt.hpp"
#include "dns/steady_stop.hpp"

#include <cmath>
#include <expected>
#include <format>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace dns::gpu::internal {

struct SteadyMarchParams
{
    double cfl = 0.4;        // local CFL of every cell
    double dtMargin = 0.95;  // dtRef = margin * min_c dt_c
    double collapse = 1e-6;  // dtRef below this: exit 7 (per-physics, run_loop)
    // > 0: ALSO exit 7 when dtRef falls below collapseRel x the first dtRef.
    // A mesh whose smallest cell legitimately sits under `collapse` (the
    // revolved wedge's axis cells, width rho dphi: ~1e-6 at 256 x 192 x 5 deg
    // laminar) sets collapse = 0 and keeps this divergence guard instead.
    double collapseRel = 0;
    double tol = 0;          // --steady TOL; > 0
    int dtRecompute = 16;    // iterations between localDt calls
    long cap = 0;            // --steady-steps; reaching it ends the march
    long statsEvery = 200;   // iterations between stop samples
    int series = 1;          // stop series per sample
    // -1: the channel's geometric rule (steadyRemaining); >= 0: the flow's
    // jitter-floored rule (steadyFloor), the first `observables` series being
    // reported values and the rest monitors.
    int observables = -1;
    // --precond: DeviceComp::stepPrecond instead of step (setPrecond ran,
    // so localDt already bounds the preconditioned wave). false = the march
    // as before.
    bool precond = false;
};

// The stop the params ask for.
inline SteadyStop makeSteadyStop(const SteadyMarchParams& p)
{
    return p.observables < 0 ? SteadyStop(p.series)
                             : SteadyStop(p.series, p.observables);
}

// What the runner sees of one stats sample, after the stop has heard it.
struct SteadyRow
{
    long iter;
    double t, dtRef;   // accumulated dtRef (NOT a physical time), last dtRef
    double remaining;  // SteadyStop::remaining(tol); +inf until certifiable
    bool converged;    // remaining < tol
    bool last;         // converged or at the cap: the runner's final row
    bool print;        // the progress line is due
    // The Newton-Krylov march only (gpu/internal/nk_march.hpp): the CFL and
    // step length of the step that produced this state, rejections so far,
    // and Krylov iterations of that step. 0 on the explicit march.
    double cfl = 0, alpha = 0;
    long rejects = 0, krylov = 0;
};

struct SteadyMarchIo
{
    // The boundary rows after every step (the stage hook's, between steps).
    std::function<void(DeviceCompState&)> afterStep;
    // One sample at the current state: the stop series (FP64), or the run's
    // error (exit 6 device, 7 non-finite). Pre: s boundary-consistent.
    std::function<std::expected<std::vector<double>, RunError>(long iter)> sample;
    // The stats row (and progress line when r.print) of the sample just
    // taken; at r.last also the runner's final outputs.
    std::function<void(const SteadyRow& r)> row;
};

struct SteadyResult
{
    long steps = 0;
    double t = 0;          // accumulated dtRef
    bool converged = false;
    double remaining = HUGE_VAL;
    // The floored rule's last per-series reading (empty under the geometric
    // rule): the trend and the jitter floor the runner reports.
    std::vector<SteadyFloor> floors;
};

// The floored rule's end-of-run line: per series its trend and jitter floor
// (relative), "-" where the window is still too short. Empty under the
// geometric rule. Pre: names.size() == floors.size() when non-empty.
inline std::string steadyFloorReport(const std::vector<SteadyFloor>& floors,
                                     std::span<const char* const> names)
{
    if (floors.empty()) return {};
    std::string s = "# steady stop (jitter-floored, relative):";
    for (std::size_t i = 0; i < floors.size(); ++i) {
        const SteadyFloor& f = floors[i];
        s += std::format("{} {} ", i ? " |" : "", names[i]);
        s += std::isfinite(f.jitter)
                 ? std::format("trend {:.2e} jitter {:.2e}", f.trend, f.jitter)
                 : std::string("window too short");
    }
    return s + "\n";
}

// The march. Pre: dev.localDt is callable (enableRans or enableSteady ran),
// s is boundary-consistent, p.tol > 0, p.cap >= 1. Leaves the lts ratios ON.
// Stops early (converged false) on a stop request; the runner turns a
// non-converged return into exit 8.
inline std::expected<SteadyResult, RunError>
explicitSteadyMarch(DeviceComp& dev, DeviceCompState& s, ReduceScratch& red,
                    const SteadyMarchParams& p, const SteadyMarchIo& io)
{
    dev.setLts(true);
    SteadyStop stop = makeSteadyStop(p);
    StatsCadence cad{p.statsEvery};
    SteadyResult res;
    double dtRef = 0.0, dt0 = 0.0;
    int sinceDt = 1 << 30;
    while (!res.converged && res.steps < p.cap) {
        if (stopRequested()) break;
        if (auto e = recomputeDt(sinceDt, dtRef, p.dtRecompute, p.dtMargin,
                                 p.collapse,
                                 [&] { return dev.localDt(s, red, p.cfl); });
            !e)
            return fail(7, e.error());
        if (dt0 == 0) dt0 = dtRef;
        if (p.collapseRel > 0 && dtRef < p.collapseRel * dt0)
            return fail(7, std::format("dt collapsed to {:.3e} from {:.3e} -- "
                                       "the state diverged",
                                       dtRef, dt0));
        if (p.precond) dev.stepPrecond(s, Real(dtRef));
        else dev.step(s, Real(dtRef));
        io.afterStep(s);
        res.t += dtRef;
        ++res.steps;
        if (res.steps - cad.lastLog < cad.every && res.steps < p.cap) continue;
        auto q = io.sample(res.steps);
        if (!q) return std::unexpected(q.error());
        // Equally spaced samples only: the cap row is off the cadence.
        if (res.steps % cad.every == 0) {
            stop.push(std::span<const double>(*q));
            res.remaining = stop.remaining(p.tol);
        }
        res.converged = res.remaining < p.tol;
        const bool last = res.converged || res.steps >= p.cap;
        const bool print =
            last || res.steps - cad.lastPrint >= 10 * cad.every;
        io.row({res.steps, res.t, dtRef, res.remaining, res.converged, last,
                print, 0, 0, 0, 0});
        if (print) cad.printed(res.steps);
        cad.logged(res.steps);
    }
    res.floors = stop.floors();
    return res;
}

} // namespace dns::gpu::internal
