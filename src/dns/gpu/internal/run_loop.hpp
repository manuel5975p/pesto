// The parts of a time-loop runner that are genuinely shared and carry NO
// ordering: the error constructor, the periodic stable-dt recompute, and the
// wall-clock epilogue.
//
// WHAT IS DELIBERATELY NOT HERE
// -----------------------------
// The loop bodies themselves. The six runners (incomp, comp, ogrid, channel,
// conv, rb) look similar and are NOT interchangeable, and the differences are
// physics:
//
//   * comp boosts the frame BETWEEN dev->step and applyBc (quiet start);
//   * channel applies the wall AFTER step and never boosts;
//   * the incompressible force tally reads the FIRST stage's phi;
//   * dtRecompute = 16 with dtMargin = 0.95 is a deliberate,
//     trajectory-changing performance default;
//   * the dt-collapse thresholds differ per physics (1e-6 comp, 1e-9 channel,
//     1e-4 incompressible) because the dt scales differ.
//
// A merged runner that standardised any of those would be a physics change
// wearing a refactor's clothes (masterplan_drift R1). So each runner keeps its
// own loop, verbatim, and only the ordering-free scaffolding is shared.
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/gpu/reductions.hpp"
#include "dns/gpu/run_result.hpp"   // RunStats, RunError
#include "dns/output.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <expected>
#include <format>
#include <numbers>
#include <string>
#include <utility>

namespace dns::gpu::internal {

inline std::unexpected<RunError> fail(int code, std::string m)
{
    return std::unexpected(RunError{code, std::move(m)});
}

// The device's health at a checkpoint: a sticky launch/runtime error, a
// failed synchronization, a reduction that failed through `red` since the
// last check (its err is then cleared), or a solver's own sticky `status`.
// Every runner calls it after the initial upload, at every stats row and at
// the end of the run; a failure is exit 6. The message names `where`.
inline std::expected<void, std::string>
deviceHealthy(const char* where, ReduceScratch* red = nullptr,
              hipError_t status = hipSuccess)
{
    hipError_t e = hipGetLastError();
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e == hipSuccess && red && red->err != hipSuccess)
    {
        e = red->err;
        red->err = hipSuccess;
    }
    if (e == hipSuccess) e = status;
    if (e == hipSuccess) return {};
    return std::unexpected(
        std::format("GPU failure at {}: {}", where, hipGetErrorString(e)));
}

// deviceHealthy() as a runner's error: exit 6.
inline std::expected<void, RunError>
checkDevice(const char* where, ReduceScratch* red = nullptr,
            hipError_t status = hipSuccess)
{
    if (auto h = deviceHealthy(where, red, status); !h) return fail(6, h.error());
    return {};
}

// The stats-row and progress-line cadences of the wall-bounded and the
// compressible body loops: a row every `every` steps, the progress line every
// 10 * `every`, both also at tEnd. (The incompressible loop keeps its own
// `steps % every` rule, which also skips the end-of-run sliver step.)
//
// The progress line has its OWN counter, and that is the whole reason this
// struct exists. Sharing the row counter made `steps - last >= 10 * every`
// unreachable -- it resets every `every` steps -- so the line printed only at
// tEnd and dns-ui's live plots stayed empty for the whole run. Found in the
// channel, conv and rb loops (2026-07-26) and again in comp and ogrid.
struct StatsCadence
{
    long every;
    long lastLog = 0, lastPrint = 0;

    bool due(long steps, double t, double tEnd) const
    {
        return steps - lastLog >= every || t >= tEnd;
    }
    bool printDue(long steps, double t, double tEnd) const
    {
        return steps - lastPrint >= 10 * every || t >= tEnd;
    }
    void logged(long steps) { lastLog = steps; }
    void printed(long steps) { lastPrint = steps; }
};

// The y-profile dump schedule: every `dt` of simulated time (dt <= 0: never),
// and once more at tEnd.
struct ProfileSchedule
{
    double dt, next;

    explicit ProfileSchedule(double every) : dt(every), next(every > 0 ? every : -1)
    {
    }
    bool due(double t, double tEnd) const
    {
        return next >= 0 && (t >= next || t >= tEnd);
    }
    // Pre: due(t, tEnd) was true. Skips every slot a long step jumped over.
    void done(double t)
    {
        while (next <= t) next += dt;
    }
};

// Quiet-start freestream schedule: smoothstep 0 -> 1 over [0, ramp]; 1 from
// the start when ramp <= 0.
inline double rampFrac(double t, double ramp)
{
    if (ramp <= 0 || t >= ramp) return 1.0;
    const double s = t / ramp;
    return s * s * (3.0 - 2.0 * s);
}

// The EMA of a potential-corrected far field's wake source (measured drag over
// rho uInf), smoothed over ~2 D/uInf so the shedding oscillation does not feed
// back into the far-field target. The caller clamps the sample and decides
// when to feed it (past the start transient). Returns the updated average.
struct WakeEma
{
    double q = 0.0, tLast;

    double update(double t, double qInst, double D, double uInf)
    {
        const double alpha = std::min(1.0, (t - tLast) / (2.0 * D / uInf));
        q += alpha * (qInst - q);
        tLast = t;
        return q;
    }
};

// A rotor's hard tip-Mach stop: prints the measured tip and fails with exit 2
// once tip + freestream reaches Mach 1 -- the KG flux has no shock capturing,
// and the extent is only measurable once the body has been voxelized.
// `label` names the radius in the printout, `subject` the thing reaching it;
// `pre`/`post` wrap the "Lower --rotor-mach below X" advice.
inline std::expected<void, RunError>
checkTipMach(double omega, double tipR, double nominalR, double cInf,
             double machInf, const char* label, const char* subject,
             const char* pre, const char* post)
{
    const double machTip = std::abs(omega) * tipR / cInf;
    std::printf("rotor: %s %.4g (nominal %.4g) -> tip Mach %.3f, +freestream "
                "%.3f\n",
                label, tipR, nominalR, machTip, machTip + machInf);
    if (machTip + machInf < 1.0) return {};
    return fail(2, std::format(
        "the {} reaches {:.4g} from the shaft, not the nominal {:.4g}, so its "
        "real tip Mach is {:.3f} and with the freestream {:.3f} >= 1. {}Lower "
        "--rotor-mach below {:.3f}{}",
        subject, tipR, nominalR, machTip, machTip + machInf, pre,
        nominalR / tipR * (1.0 - machInf) * 0.95, post));
}

// The propeller coefficient denominators rho n^2 D^4 (thrust) and rho n^2 D^5
// (torque), n = |omega| / 2 pi revolutions per unit time. Both are 0 at
// omega = 0, where the coefficients are undefined and reported as 0.
struct PropellerDen
{
    double ct, cq;
};
inline PropellerDen propellerDen(double rho, double omega, double D)
{
    const double rotorN = std::abs(omega) / (2.0 * std::numbers::pi);
    const double ctDen = rho * rotorN * rotorN * std::pow(D, 4);
    return {ctDen, ctDen * D};
}

// Periodic stable-dt recompute with the collapse guard, shared by the
// incompressible loop and the four compressible ones (comp, ogrid, channel,
// conv). `sinceDt` and `dtStab` are the caller's loop state and
// are updated in place; `stable` returns the raw solver bound.
//
// Returns an error STRING (not a RunError) so the caller keeps its own exit
// code and message context. Both failures are exit 7 today.
//
// `collapse` is per-physics on purpose -- see the header comment.
template <class StableFn>
std::expected<void, std::string>
recomputeDt(int& sinceDt, double& dtStab, int recomputeEvery, double margin,
            double collapse, StableFn&& stable)
{
    if (++sinceDt >= recomputeEvery)
    {
        const double s = stable();
        if (!std::isfinite(s) || s <= 0)
            return std::unexpected("stable dt is not finite and positive "
                                   "(the state diverged)");
        dtStab = margin * s;
        sinceDt = 0;
    }
    if (dtStab < collapse)
        return std::unexpected("dt collapsed -- the state diverged");
    return {};
}

// One-sided 2nd-order wall gradient of a quantity whose WALL VALUE is pinned
// to zero, from its first two off-wall values: f'(0) = (f1 d2^2 - f2 d1^2) /
// (d1 d2 (d2 - d1)). Both wall-bounded runners measure a wall flux this way --
// the channel its friction velocity from u, the convection cell its Nusselt
// number from T - Twall -- and both must subtract the pinned value themselves.
// Pre: 0 < d1 < d2.
inline double wallGradientPinned(double f1, double f2, double d1, double d2)
{
    return (f1 * d2 * d2 - f2 * d1 * d1) / (d1 * d2 * (d2 - d1));
}

// Wall-clock ms/step, recorded on the stats and handed to the output sink.
// Identical in every runner.
inline void finishTiming(RunStats& st,
                         std::chrono::steady_clock::time_point wall0,
                         RunOutput& out)
{
    st.msPerStep = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - wall0)
                       .count()
                   / double(st.steps ? st.steps : 1);
    out.setMsPerStep(st.msPerStep);
}

} // namespace dns::gpu::internal
