// dns/steady_stop: the steady march's stop rule on integral quantities. A pure
// geometric tail is extrapolated exactly; growth, oscillation and too-short
// histories are never certified; a flat noisy window is; and a turning point
// that fools ONE series is caught by a companion series on the slow mode --
// the reason the channel monitors several integrals, not ub alone.
// Host-only, milliseconds.
#include "dns/steady_stop.hpp"

#include "check.hpp"

#include <cmath>
#include <format>
#include <functional>
#include <random>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

std::vector<double> sample(const std::function<double(int)>& f, int n)
{
    std::vector<double> x(n);
    for (int k = 0; k < n; ++k) x[k] = f(k);
    return x;
}

// x_k = 1 + A rho^k: the estimate IS the remaining change, at every length.
void geometricExact()
{
    for (const double A : {0.03, -0.05})
    {
        double worst = 0;
        for (int n = 6; n <= 600; ++n)
        {
            const auto x = sample([&](int k) { return 1 + A * std::pow(0.99, k); }, n);
            const double truth = std::abs(x.back() - 1) / std::abs(x.back());
            worst = std::max(worst, std::abs(steadyRemaining(x, 1e-6) / truth - 1));
        }
        check(worst < 1e-9, std::format("geometric tail A={}: estimate == remaining "
                                        "(worst rel {:.1e})", A, worst));
    }
    // The stop it implies is never early: first n with estimate < tol has
    // true remaining < tol.
    for (const double tol : {1e-3, 1e-5})
        for (int n = 6; n <= 3000; ++n)
        {
            const auto x = sample([](int k) { return 1 + 0.03 * std::pow(0.995, k); }, n);
            if (steadyRemaining(x, tol) >= tol) continue;
            check(std::abs(x.back() - 1) / x.back() < tol,
                  std::format("geometric stop at tol {} is not early (n {})", tol, n));
            break;
        }
}

void undecidable()
{
    check(std::isinf(steadyRemaining(std::vector<double>(5, 1.0), 1e-3)),
          "fewer than 6 samples: never certified");
    check(steadyRemaining(std::vector<double>(6, 1.0), 1e-3) == 0,
          "6 equal samples: exactly 0");
    check(std::isinf(steadyRemaining(std::vector<double>(12, 0.0), 1e-3)),
          "x_last == 0: no relative scale, never certified");
    check(std::isinf(steadyRemaining(
              sample([](int k) { return 1 + 1e-3 * k; }, 60), 1e-3)),
          "linear drift (ratio 1): never certified");
    check(std::isinf(steadyRemaining(
              sample([](int k) { return 1 + 1e-4 * std::pow(1.01, k); }, 60), 1e-3)),
          "growth above the flat threshold: never certified");
}

// Four samples at spacing m alias an oscillation into a contracting sequence
// (without the monotone-window guard: an undamped 1e-4 wave certified at tol
// 1e-5 on 52 lengths, a damped one 8.7x early). With it: an undamped wave is
// never certified, a damped one only once flat, and never early.
void oscillation()
{
    for (const double tol : {1e-5, kSteadyTolMin})
    {
        int certified = 0;
        for (int n = 6; n <= 3000; ++n)
            certified += steadyRemaining(
                             sample([](int k) { return 1 + 1e-4 * std::cos(0.37 * k); }, n),
                             tol) < tol;
        check(certified == 0,
              std::format("undamped oscillation never certified at tol {} ({} lengths)",
                          tol, certified));
    }
    for (const double tol : {1e-4, 1e-5, kSteadyTolMin})
    {
        bool stopped = false, early = false;
        for (int n = 6; n <= 8000 && !stopped; ++n)
        {
            const auto x = sample(
                [](int k) { return 1 + 0.01 * std::exp(-k / 200.0) * std::cos(k / 7.0); },
                n);
            if (steadyRemaining(x, tol) >= tol) continue;
            stopped = true;
            early = std::abs(x.back() - 1) > tol;
        }
        check(stopped && !early,
              std::format("damped oscillation at tol {}: certified, never early", tol));
    }
}

// Converged to FP32 sampling noise (the measured ~3e-8): certified within a
// few rows, at every TOL down to kSteadyTolMin.
void noise()
{
    for (const double tol : {1e-3, 1e-5, kSteadyTolMin})
    {
        std::mt19937 gen(7);
        std::normal_distribution<double> nd(0, 3e-8);
        std::vector<double> x;
        int firstOk = -1;
        for (int n = 1; n <= 600 && firstOk < 0; ++n)
        {
            x.push_back(15.8 * (1 + nd(gen)));
            if (n >= 60 && steadyRemaining(x, tol) < tol) firstOk = n;
        }
        check(firstOk >= 0 && firstOk <= 70,
              std::format("noise 3e-8 certified at tol {} within 10 rows (row {})",
                          tol, firstOk));
    }
    // Noise ABOVE the flat threshold with sign changes is not certified by the
    // flat branch: 1e-5 noise at tol 1e-3 must not pass on max|d| alone.
    std::mt19937 gen(11);
    std::normal_distribution<double> nd(0, 1e-5);
    std::vector<double> x(60);
    for (double& v : x) v = 1 + nd(gen);
    x[59] = x[49] - 1e-5, x[39] = x[49] + 1e-5, x[29] = x[39] - 1e-5; // + - + window
    check(std::isinf(steadyRemaining(x, 1e-3)),
          "a sign-changing window above kSteadyFlat is not flat");
}

// One series with a fast mode against an opposite slow mode passes through a
// geometric-looking deceleration into its turning point 5% off its limit (the
// measured channel ub does this: alone it certifies TOL 1e-3 at 3400
// iterations, 2.8% low). A second series carrying the slow mode alone keeps
// the pair from stopping there, and the pair's stop is inside tol on both.
void turningPoint()
{
    const double tol = 1e-3;
    auto X = [](int k) {
        return 1 + 0.03 * std::exp(-k / 10.0) - 0.06 * std::exp(-k / 300.0);
    };
    auto Y = [](int k) { return 1 + 0.05 * std::exp(-k / 300.0); };
    bool fooled = false;
    for (int n = 6; n <= 200 && !fooled; ++n)
    {
        const auto x = sample(X, n);
        fooled = steadyRemaining(x, tol) < tol && std::abs(x.back() - 1) > 10 * tol;
    }
    check(fooled, "a single series IS fooled at its turning point (the reason "
                  "the stop monitors several integrals)");

    SteadyStop stop(2);
    for (int k = 0; k < 8000; ++k)
    {
        const double q[2] = {X(k), Y(k)};
        stop.push(q);
        if (stop.remaining(tol) >= tol) continue;
        const double ex = std::abs(q[0] - 1) / q[0], ey = std::abs(q[1] - 1) / q[1];
        check(ex < tol && ey < tol,
              std::format("pair stops inside tol (row {}, x {:.1e}, y {:.1e})", k, ex, ey));
        return;
    }
    check(false, "pair never stopped");
}

void slowestSeriesGoverns()
{
    SteadyStop stop(3);
    for (int k = 0; k < 600; ++k)
    {
        const double q[3] = {1 + 0.01 * std::pow(0.9, k), 2.0,
                             1 + 0.01 * std::pow(0.995, k)};
        stop.push(q);
    }
    const double slow = 0.01 * std::pow(0.995, 599);
    check(std::abs(stop.remaining(1e-3) / (slow / (1 + slow)) - 1) < 1e-9,
          "remaining is the slowest series' tail");
}

// --- the jitter-floored rule (dns flow) ---------------------------------------

std::vector<double> noisy(const std::function<double(int)>& f, int n, double sd,
                          unsigned seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd(0, sd);
    std::vector<double> x(n);
    for (int k = 0; k < n; ++k) x[k] = f(k) * (1 + nd(gen));
    return x;
}

// First n in [48, nMax] whose rule certifies tol, else -1; `err` = the last
// sample's true relative distance from `limit` there.
int firstStop(const std::function<std::vector<double>(int)>& hist, int nMax,
              double tol, bool observable, double limit, double& err)
{
    for (int n = 48; n <= nMax; ++n)
    {
        const auto x = hist(n);
        if (!(steadyFloor(x, observable).remaining < tol)) continue;
        err = std::abs(x.back() - limit) / std::abs(limit);
        return n;
    }
    return -1;
}

// White FP32-like jitter on a converged value (the measured floors: 3e-7
// cylinder Re 40, 5e-5 sphere Re 1e4): certified at the first full window as
// an observable iff TOL clears trend + sqrt(2) J, the jitter J read back.
void flooredNoise()
{
    for (const double sd : {3e-7, 5e-5})
    {
        const auto x = noisy([](int) { return 0.4248; }, 48, sd, 3);
        const SteadyFloor f = steadyFloor(x, true);
        check(std::abs(f.jitter / sd - 1) < 0.35,
              std::format("jitter {:.1e} read back as {:.2e}", sd, f.jitter));
        check(f.remaining < 3 * sd + f.trend && f.trend < 3 * sd,
              std::format("white jitter {:.1e}: no trend (trend {:.2e})", sd,
                          f.trend));
        check(f.remaining < 1e-3, std::format("jitter {:.1e} certifies TOL 1e-3", sd));
        check(!(f.remaining < sd), std::format("jitter {:.1e} never certifies "
                                               "TOL below it", sd));
    }
    check(std::isinf(steadyFloor(std::vector<double>(47, 1.0), true).remaining),
          "47 samples: a block under kSteadyBlockMin, never certified");
    check(steadyFloor(std::vector<double>(48, 1.0), true).remaining == 0,
          "48 equal samples: exactly 0");
    check(std::isinf(steadyFloor(std::vector<double>(96, 0.0), true).remaining),
          "x_last == 0: never certified");
}

// Geometric tails, clean and under jitter: never early (the last sample
// within tol of the limit at the first certification).
void flooredGeometric()
{
    for (const double sd : {0.0, 3e-7, 1e-6})
        for (const double tol : {1e-3, 1e-4, 1e-5})
            for (const double rate : {0.99, 0.995})
            {
                double err = 0;
                const int n = firstStop(
                    [&](int m) {
                        return noisy([&](int k) { return 1 + 0.03 * std::pow(rate, k); },
                                     m, sd, 5);
                    },
                    6000, tol, true, 1.0, err);
                check(n > 0 && err < tol,
                      std::format("geometric rate {} jitter {:.0e} tol {}: stops ({}), "
                                  "never early (err {:.1e})",
                                  rate, sd, tol, n, err));
            }
}

// Drift is never certified once the window resolves it: linear (no fixed
// point) above the noise from the first window, and a slow drift buried in
// jitter (1e-7 / sample under 1e-6) from the window that resolves it (b
// 48 samples: drift 4.8e-6 per block against a detection limit 6e-7) --
// before that it is indistinguishable from jitter by ANY rule on that window,
// and the trend it reports (drift + detection limit) is what it is.
void flooredDrift()
{
    double err = 0;
    for (const bool obs : {true, false})
    {
        check(firstStop([](int m) { return sample([](int k) { return 1 + 1e-5 * k; }, m); },
                        3000, 1e-3, obs, 1.0, err) < 0,
              std::format("linear drift 1e-5 / sample: never certified ({})",
                          obs ? "observable" : "monitor"));
        check(firstStop([](int m) {
                  return noisy([](int k) { return 1 + 1e-6 * k; }, m, 1e-6, 9);
              }, 3000, 1e-4, obs, 1.0, err) < 0,
              std::format("drift 1e-6 / sample under 1e-6 jitter: never certified "
                          "({})", obs ? "observable" : "monitor"));
        int late = 0;
        for (int n = 288; n <= 3000; n += 7)
            late += steadyFloor(noisy([](int k) { return 1 + 1e-7 * k; }, n, 1e-6, 9),
                                obs)
                        .remaining < 1e-5;
        check(late == 0, std::format("slow drift 1e-7 / sample under 1e-6 jitter: "
                                     "never certified at 1e-5 once resolved ({}, "
                                     "{} windows)", obs ? "observable" : "monitor",
                                     late));
    }
}

// Oscillations above the floor are never certified; a damped one only once
// its swing is under tol, never early. As a monitor, an undamped swing above
// kSteadyJitterMax is refused although its block means are flat. (An
// observable's undamped swing of amplitude A is certified iff trend + A <
// tol: every sample is then inside tol of the fixed point. A monitor's swing
// below kSteadyJitterMax is the stated floor: not told from FP32 jitter.)
void flooredOscillation()
{
    double err = 0;
    for (const double tol : {1e-3, 1e-4, 1e-5, kSteadyTolMin})
    {
        for (const double w : {0.37, 1.9})
            check(firstStop([&](int m) {
                      return sample([&](int k) { return 1 + 1.5 * tol * std::cos(w * k); },
                                    m);
                  }, 3000, tol, true, 1.0, err) < 0,
                  std::format("undamped 1.5 tol oscillation (w {}) never certified at "
                              "tol {}", w, tol));
        // A period (126 samples) longer than the early windows: a window
        // shorter than the period decelerating into a crest IS a turning
        // point (the one-series limitation, turningPoint()); once the window
        // spans the period it is never certified.
        int late = 0;
        for (int n = 260; n <= 3000; n += 3)
            late += steadyFloor(sample([&](int k) { return 1 + 1.5 * tol * std::cos(0.05 * k); },
                                       n),
                                true)
                        .remaining < tol;
        check(late == 0, std::format("slow undamped oscillation (period 126) never "
                                     "certified once the window spans it, tol {} ({})",
                                     tol, late));
    }
    // Period 2 (the measured SST sphere Re 1e4 L2 limit cycle: acf -0.9).
    check(firstStop([](int m) {
              return sample([](int k) { return 0.41 * (1 + 4e-3 * (k % 2 ? 1 : -1)); }, m);
          }, 3000, 1e-3, true, 0.41, err) < 0,
          "period-2 limit cycle 4e-3: never certified at tol 1e-3");
    check(firstStop([](int m) {
              return sample([](int k) { return 1 + 3e-3 * std::cos(0.37 * k); }, m);
          }, 3000, 1e-3, false, 1.0, err) < 0,
          "monitor oscillation 3e-3 > kSteadyJitterMax: never certified");
    check(firstStop([](int m) {
              return sample([](int k) { return 1 + 3e-3 * std::cos(0.37 * k); }, m);
          }, 3000, 1e-3, true, 1.0, err) < 0,
          "observable oscillation 3e-3 at tol 1e-3: never certified");
    for (const double tol : {1e-3, 1e-4, 1e-5})
    {
        const int n = firstStop([](int m) {
            return sample([](int k) {
                return 1 + 0.01 * std::exp(-k / 200.0) * std::cos(k / 7.0);
            }, m);
        }, 8000, tol, true, 1.0, err);
        check(n > 0 && err < tol,
              std::format("damped oscillation at tol {}: certified ({}), never early "
                          "(err {:.1e})", tol, n, err));
    }
    // The same swing under jitter.
    for (const double tol : {1e-3, 1e-4})
    {
        const int n = firstStop([](int m) {
            return noisy([](int k) {
                return 1 + 0.01 * std::exp(-k / 200.0) * std::cos(k / 7.0);
            }, m, 1e-6, 13);
        }, 8000, tol, true, 1.0, err);
        check(n > 0 && err < tol,
              std::format("damped oscillation + 1e-6 jitter at tol {}: certified ({}), "
                          "never early (err {:.1e})", tol, n, err));
    }
}

// The turning-point lesson under the floored rule: one series alone is fooled
// at its turning point, the pair with a slow-mode monitor is not.
void flooredTurningPoint()
{
    const double tol = 1e-3;
    auto X = [](int k) {
        return 1 + 0.03 * std::exp(-k / 10.0) - 0.06 * std::exp(-k / 300.0);
    };
    auto Y = [](int k) { return 1 + 0.05 * std::exp(-k / 300.0); };
    SteadyStop stop(2, 1);
    for (int k = 0; k < 8000; ++k)
    {
        const double q[2] = {X(k), Y(k)};
        stop.push(q);
        if (!(stop.remaining(tol) < tol)) continue;
        const double ex = std::abs(q[0] - 1) / q[0], ey = std::abs(q[1] - 1) / q[1];
        check(ex < tol && ey < tol,
              std::format("floored pair stops inside tol (row {}, x {:.1e}, y {:.1e})",
                          k, ex, ey));
        const auto f = stop.floors();
        check(f.size() == 2 && std::isfinite(f[0].jitter) && std::isfinite(f[1].jitter),
              "floors() reports both series");
        return;
    }
    check(false, "floored pair never stopped");
}

// The rule split: SteadyStop(n) stays the channel's geometric rule, bit for
// bit steadyRemaining; floors() is empty there.
void ruleSplit()
{
    SteadyStop geo(1);
    std::vector<double> x;
    for (int k = 0; k < 300; ++k)
    {
        const double q[1] = {1 + 0.03 * std::pow(0.99, k)};
        geo.push(q);
        x.push_back(q[0]);
    }
    check(geo.remaining(1e-4) == steadyRemaining(x, 1e-4),
          "SteadyStop(n) == steadyRemaining exactly");
    check(geo.floors().empty() && !geo.floored(), "geometric rule: no floors");
    SteadyStop fl(1, 1);
    for (const double v : x)
    {
        const double q[1] = {v};
        fl.push(q);
    }
    check(fl.floored() && fl.remaining(1e-4) == steadyFloor(x, true).remaining,
          "SteadyStop(n, k) == steadyFloor exactly");
}

} // namespace

int main()
{
    geometricExact();
    undecidable();
    oscillation();
    noise();
    turningPoint();
    slowestSeriesGoverns();
    flooredNoise();
    flooredGeometric();
    flooredDrift();
    flooredOscillation();
    flooredTurningPoint();
    ruleSplit();
    return dnstest::finish();
}
