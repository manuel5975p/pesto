// The Newton-Krylov pseudo-time march of `dns flow --steady TOL
// --steady-scheme implicit` (docs/rans.md, "Newton-Krylov pseudo-time";
// masterplan_flowrans 4.7): per iteration one linear solve (gpu/nk), a
// whole-step length, a trial state, and accept / reject on its scaled
// residual. The CFL ramp and the arming are PseudoTime's (kNkPolicy): the
// integrals are heard by the stop ONLY on an armed step (undamped at the cap),
// and any other step clears their history -- an unarmed sample certified a
// damped stall on the channel.
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/gpu/internal/steady_march.hpp"
#include "dns/gpu/nk.hpp"
#include "dns/pseudo_time.hpp"

#include <cmath>
#include <format>

namespace dns::gpu::internal {

// Pre: nk made on dev with the runner's hook; s boundary-consistent, plain;
// p.tol > 0, p.cap >= 1 (p.cfl, dtMargin, dtRecompute, statsEvery unread:
// one row per Newton iteration). The progress line every 10 iterations.
inline std::expected<SteadyResult, RunError>
nkSteadyMarch(DeviceNk& nk, DeviceCompState& s, ReduceScratch& red,
              const SteadyMarchParams& p, const SteadyMarchIo& io)
{
    PseudoTime pt(kNkPolicy);
    SteadyStop stop = makeSteadyStop(p);
    SteadyResult res;
    if (nk.load(s) != hipSuccess) return fail(6, "Newton-Krylov load failed");
    nk.store(s);
    double rn = nk.residual(s, red);
    if (!std::isfinite(rn))
        return fail(7, "the residual of the initial state is not finite");
    long rejects = 0, lastPrint = 0;
    double alpha = 0, cflStep = 0;
    long krylov = 0;
    bool stalled = false; // the last solve hit the Krylov cap unconverged
    auto reject = [&]() -> std::expected<void, RunError> {
        stop = makeSteadyStop(p);
        ++rejects;
        nk.store(s);
        rn = nk.residual(s, red);
        if (pt.rejected()) return {};
        return fail(7, std::format("Newton-Krylov pseudo-time collapsed: CFL "
                                   "< {:g} after {} rejections",
                                   kNkCflMin, rejects));
    };
    while (!res.converged && res.steps < p.cap) {
        if (stopRequested()) break;
        ++res.steps;
        cflStep = pt.cfl();
        // kNkStallKrylov: at the cap after a stalled solve, one cycle.
        const bool atCap = cflStep >= kNkPolicy.cflMax;
        auto lin = nk.solve(s, cflStep, red,
                            atCap && stalled ? kNkStallKrylov : 0);
        if (!lin) {
            // A singular line pivot or a GMRES breakdown: a rejection.
            if (auto e = reject(); !e) return std::unexpected(e.error());
            continue;
        }
        krylov = lin->krylov;
        stalled = !lin->converged;
        alpha = nk.stepLength(red);
        if (!(alpha > 0)) {
            if (auto e = reject(); !e) return std::unexpected(e.error());
            continue;
        }
        nk.trial(alpha, s);
        const double rt = nk.residual(s, red);
        if (!std::isfinite(rt) || rt > kNkRejectGrowth * rn) {
            if (auto e = reject(); !e) return std::unexpected(e.error());
            continue;
        }
        nk.accept();
        pt.accepted(alpha);
        rn = rt;
        auto q = io.sample(res.steps);
        if (!q) return std::unexpected(q.error());
        if (pt.armed())
            stop.push(std::span<const double>(*q));
        else
            stop = makeSteadyStop(p);
        res.remaining = stop.remaining(p.tol);
        res.converged = res.remaining < p.tol;
        const bool last = res.converged || res.steps >= p.cap;
        const bool print = last || res.steps - lastPrint >= 10;
        io.row({res.steps, res.t, 0.0, res.remaining, res.converged, last,
                print, cflStep, alpha, rejects, krylov});
        if (print) lastPrint = res.steps;
    }
    res.floors = stop.floors();
    return res;
}

} // namespace dns::gpu::internal
