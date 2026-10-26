#pragma once

#include <cmath>
#include <span>
#include <vector>

namespace dns {

// Stop rule of the steady pseudo-time march (docs/rans.md, "Stopping on the
// integrals"). The FP32 residual cannot certify the slow mode, so the march
// stops on what it converges: equally spaced samples of a few integral
// quantities, each extrapolated geometrically to its fixed point.

// Below this relative change per window a series is FLAT: it is at the FP32
// sampling noise (measured ~3e-8 on the channel integrals) and its ratios
// carry no information. 2^-20 = 16 FP32 ulps.
inline constexpr double kSteadyFlat = 0x1p-20;
// The smallest --steady TOL the rule can certify: 4 kSteadyFlat, rounded up.
inline constexpr double kSteadyTolMin = 1e-6;

// Estimated |x_inf - x_last| / |x_last| of ONE series from its last four
// samples at spacing m = size/6 (so the window is the second half of the
// series: early transients are outside it). +inf unless every sample in the
// window is monotone to kSteadyFlat (four samples alone alias an oscillation
// into a geometric-looking one). Geometric when both difference ratios lie in
// [0, 1): |d3| r/(1-r), r the larger ratio. Otherwise +inf, except a flat
// window (every |d| <= min(tol/4, kSteadyFlat)): max|d|. A single series is
// still fooled while decelerating INTO a turning point; monitor several.
// Pre: tol > 0. +inf for fewer than 6 samples or x_last == 0.
double steadyRemaining(std::span<const double> x, double tol);

// --- the jitter-floored rule of `dns flow --steady` (docs/rans.md, "Stopping
// on a jitter floor") -----------------------------------------------------
// An external flow's integrals do not settle to kSteadyFlat: the FP32 floor
// moves the state by ~1 ulp per cell every iteration, which a pressure-
// integrated coefficient amplifies by p/q (~36 at M 0.2), so cd jitters at
// 3e-7 (cylinder Re 40) .. 5e-5 (sphere Re 1e4) relative -- white, not a
// trend. The rule separates the two: the TREND is read off block means
// (which average the jitter down) and the JITTER is measured and reported.

// A monitor's jitter above this is an oscillation, not FP32 noise: the
// measured floors reach 4.2e-4 (the wake deficit W of the sphere wedge at
// Re 1e4), a pseudo-time limit cycle 4e-3 (SST sphere Re 1e4 L2: cd) and 3e-2
// (its W). 2^-10 = 9.8e-4.
inline constexpr double kSteadyJitterMax = 0x1p-10;
// Samples per block, minimum: the window needs 3 blocks of 4 sub-blocks.
inline constexpr int kSteadyBlockMin = 8;

// What the floored rule read off one series' window.
struct SteadyFloor
{
    double trend = HUGE_VAL;  // relative remaining change of the block means
    double jitter = HUGE_VAL; // J: rms about the blocks' own lines, relative
    double remaining = HUGE_VAL; // the series' verdict (< tol: certified)
};

// One series under the floored rule. window = the last 3b samples (b =
// floor(n/6) rounded down to a multiple of 4, >= kSteadyBlockMin: the second
// half), three blocks of b, 12 sub-blocks of b/4. sigma of a block mean from
// the second differences of the sub-block means (blind to a linear trend,
// so a decaying mode cannot inflate it; it sees correlation up to b/4
// samples); delta = 3 sqrt(2) sigma, the detection limit of a block-mean
// difference d. |d1|, |d2| <= delta: trend = max|d| + delta. Same sign and
// |d2| + delta < |d1| - delta: geometric, r = (|d2| + delta)/(|d1| - delta),
// trend = (|d2| + delta) r/(1 - r). Otherwise +inf. An OBSERVABLE (a
// reported value: cd, cdf) is judged on the last sample, remaining = trend +
// sqrt(2) J; a MONITOR (a slow-mode integral) on its trend alone, +inf when J
// > kSteadyJitterMax. Pre: tol > 0. All +inf when too short or x_last == 0.
SteadyFloor steadyFloor(std::span<const double> x, bool observable);

// The history of the monitored integrals, one row per regular stats row.
class SteadyStop
{
public:
    // The channel's geometric rule (steadyRemaining). Pre: n >= 1 series.
    explicit SteadyStop(int n);
    // The flow's jitter-floored rule (steadyFloor): the first `observables`
    // series are observables, the rest monitors. Pre: 0 <= observables <= n.
    SteadyStop(int n, int observables);
    // Appends one sample per series. Pre: q.size() == number of series.
    void push(std::span<const double> q);
    // max over the series of the rule's verdict: converged iff < tol.
    double remaining(double tol) const;
    // The floored rule's per-series reading (empty under the geometric rule).
    std::vector<SteadyFloor> floors() const;
    bool floored() const { return observables_ >= 0; }

private:
    std::vector<std::vector<double>> series_;
    int observables_ = -1; // -1: the geometric rule
};

} // namespace dns
