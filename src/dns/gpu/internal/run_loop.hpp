// The parts of a time-loop runner that are genuinely shared and carry NO
// ordering: the error constructor, the periodic stable-dt recompute, and the
// wall-clock epilogue.
//
// WHAT IS DELIBERATELY NOT HERE
// -----------------------------
// The loop bodies themselves. runner_incomp, runner_comp and runner_channel
// look similar and are NOT interchangeable, and the differences are physics:
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

#include "dns/gpu/runner_incomp.hpp"   // RunError, RunStats

#include <chrono>
#include <cmath>
#include <expected>
#include <string>
#include <utility>

namespace dns::gpu::internal {

inline std::unexpected<RunError> fail(int code, std::string m)
{
    return std::unexpected(RunError{code, std::move(m)});
}

// Periodic stable-dt recompute with the collapse guard, shared by the two
// compressible loops. `sinceDt` and `dtStab` are the caller's loop state and
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

// Wall-clock ms/step, recorded on the stats and handed to the output sink.
// Identical in all three runners.
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
