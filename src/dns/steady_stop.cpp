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

namespace {

// Mean and least-squares slope of x about its own centre index.
struct Line
{
    double mean, slope, tc;
};

Line fitLine(std::span<const double> x)
{
    const double n = double(x.size()), tc = 0.5 * (n - 1);
    double m = 0;
    for (const double v : x) m += v;
    m /= n;
    double sxy = 0, sxx = 0;
    for (std::size_t k = 0; k < x.size(); ++k) {
        const double t = double(k) - tc;
        sxy += t * (x[k] - m);
        sxx += t * t;
    }
    return {m, sxx > 0 ? sxy / sxx : 0.0, tc};
}

} // namespace

SteadyFloor steadyFloor(std::span<const double> x, bool observable)
{
    SteadyFloor f;
    const std::size_t b = x.size() / 6 / 4 * 4;
    if (b < std::size_t(kSteadyBlockMin)) return f;
    const double scale = std::abs(x.back());
    if (!(scale > 0)) return f;
    const std::span<const double> w = x.subspan(x.size() - 3 * b);
    const std::size_t L = b / 4;
    // Sub-block means; their second differences give a block mean's noise.
    std::array<double, 12> M{};
    for (std::size_t s = 0; s < 12; ++s) {
        double a = 0;
        for (std::size_t k = 0; k < L; ++k) a += w[s * L + k];
        M[s] = a / double(L) / scale;
    }
    double dd = 0;
    for (std::size_t s = 1; s + 1 < 12; ++s) {
        const double e = M[s + 1] - 2 * M[s] + M[s - 1];
        dd += e * e;
    }
    const double sigmaSub = std::sqrt(dd / 10.0 / 6.0);
    const double sigma = 0.5 * sigmaSub;
    const double delta = 3.0 * std::sqrt(2.0) * sigma;
    std::array<double, 3> m{};
    double ss = 0;
    for (std::size_t q = 0; q < 3; ++q) {
        const auto blk = w.subspan(q * b, b);
        const Line ln = fitLine(blk);
        m[q] = ln.mean / scale;
        for (std::size_t k = 0; k < b; ++k) {
            const double e = blk[k] - ln.mean - ln.slope * (double(k) - ln.tc);
            ss += e * e;
        }
    }
    f.jitter = std::sqrt(ss / double(3 * b)) / scale;
    const double d1 = m[1] - m[0], d2 = m[2] - m[1];
    // Three block means alias a swing whose turn lies inside the window into
    // a contracting one; the sub-block means must be monotone to their own
    // detection limit (the jitter-tolerant form of the geometric rule's
    // monotone window).
    const double rev = reversal(M);
    if (std::abs(d1) <= kSteadyFlat && std::abs(d2) <= kSteadyFlat &&
        rev <= kSteadyFlat) {
        // Flat to the FP32 sampling floor over the whole window (the
        // geometric rule's flat branch): converged by any standard, whatever
        // the noise statistics (an FP64 turbulence integral moving by 1e-9
        // reads as "non-monotone" against its own tiny sigma).
        f.trend = std::max(std::abs(d1), std::abs(d2));
    } else if (rev > 3.0 * std::sqrt(2.0) * sigmaSub) {
        // trend stays +inf
    } else if (std::abs(d1) <= delta && std::abs(d2) <= delta) {
        f.trend = std::max(std::abs(d1), std::abs(d2)) + delta;
    } else if (d1 * d2 > 0) {
        const double lo = std::abs(d1) - delta, hi = std::abs(d2) + delta;
        if (lo > 0 && hi < lo) {
            const double r = hi / lo;
            f.trend = hi * r / (1 - r);
        }
    }
    f.remaining = observable ? f.trend + std::sqrt(2.0) * f.jitter
                  : f.jitter <= kSteadyJitterMax ? f.trend
                                                 : HUGE_VAL;
    return f;
}

SteadyStop::SteadyStop(int n) : series_(std::size_t(std::max(n, 0)))
{
    assert(n >= 1); // no series would certify everything
}

SteadyStop::SteadyStop(int n, int observables)
    : series_(std::size_t(std::max(n, 0))), observables_(observables)
{
    assert(n >= 1 && observables >= 0 && observables <= n);
}

void SteadyStop::push(std::span<const double> q)
{
    assert(q.size() == series_.size());
    for (std::size_t i = 0; i < q.size(); ++i) series_[i].push_back(q[i]);
}

double SteadyStop::remaining(double tol) const
{
    double worst = 0;
    if (!floored()) {
        for (const auto& s : series_)
            worst = std::max(worst, steadyRemaining(s, tol));
        return worst;
    }
    for (const SteadyFloor& f : floors()) worst = std::max(worst, f.remaining);
    return worst;
}

std::vector<SteadyFloor> SteadyStop::floors() const
{
    std::vector<SteadyFloor> out;
    if (!floored()) return out;
    for (std::size_t i = 0; i < series_.size(); ++i)
        out.push_back(steadyFloor(series_[i], int(i) < observables_));
    return out;
}

} // namespace dns
