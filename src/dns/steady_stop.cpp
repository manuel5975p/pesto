#include "dns/steady_stop.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

namespace dns {

namespace {

// d1 -> d2 contracts monotonically: same sign, |d2| < |d1| (0/0 counts).
bool contracting(double d1, double d2, double& r)
{
    if (d1 == 0)
    {
        r = 0;
        return d2 == 0;
    }
    r = d2 / d1;
    return r >= 0 && r < 1;
}

// The smaller of the window's largest drop and largest rise: 0 for a
// monotone window, the reversal for one that turns or oscillates.
double reversal(std::span<const double> w)
{
    double hi = w[0], lo = w[0], drop = 0, rise = 0;
    for (const double v : w)
    {
        drop = std::max(drop, hi - v);
        rise = std::max(rise, v - lo);
        hi = std::max(hi, v);
        lo = std::min(lo, v);
    }
    return std::min(drop, rise);
}

} // namespace

double steadyRemaining(std::span<const double> x, double tol)
{
    assert(tol > 0);
    const std::size_t m = x.size() / 6;
    if (m < 1) return HUGE_VAL;
    const std::size_t last = x.size() - 1;
    const double scale = std::abs(x[last]);
    if (!(scale > 0)) return HUGE_VAL;
    // Four samples alias an oscillation into a geometric-looking sequence;
    // every row in between must be monotone to the sampling noise.
    if (reversal(x.subspan(last - 3 * m)) > kSteadyFlat * scale) return HUGE_VAL;
    std::array<double, 3> d;
    for (std::size_t i = 0; i < 3; ++i)
        d[i] = (x[last - (2 - i) * m] - x[last - (3 - i) * m]) / scale;
    double r1 = 0, r2 = 0;
    if (contracting(d[0], d[1], r1) && contracting(d[1], d[2], r2))
    {
        const double r = std::max(r1, r2);
        return std::abs(d[2]) * r / (1 - r);
    }
    // Sign changes or growth: a turning point, an oscillation or noise. Only
    // a window flat to the sampling noise is certified, never extrapolated.
    const double dMax = std::ranges::max(d, {}, [](double v) { return std::abs(v); });
    return std::abs(dMax) <= std::min(tol / 4, kSteadyFlat) ? std::abs(dMax)
                                                            : HUGE_VAL;
}

SteadyStop::SteadyStop(int n) : series_(std::size_t(std::max(n, 0)))
{
    assert(n >= 1); // no series would certify everything
}

void SteadyStop::push(std::span<const double> q)
{
    assert(q.size() == series_.size());
    for (std::size_t i = 0; i < q.size(); ++i) series_[i].push_back(q[i]);
}

double SteadyStop::remaining(double tol) const
{
    double worst = 0;
    for (const auto& s : series_) worst = std::max(worst, steadyRemaining(s, tol));
    return worst;
}

} // namespace dns
