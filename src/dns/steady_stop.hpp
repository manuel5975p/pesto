#pragma once

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

// The history of the monitored integrals, one row per regular stats row.
class SteadyStop
{
public:
    // Pre: n >= 1 series per row.
    explicit SteadyStop(int n);
    // Appends one sample per series. Pre: q.size() == number of series.
    void push(std::span<const double> q);
    // max over the series of steadyRemaining: converged iff < tol.
    double remaining(double tol) const;

private:
    std::vector<std::vector<double>> series_;
};

} // namespace dns
