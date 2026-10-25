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

} // namespace

int main()
{
    geometricExact();
    undecidable();
    oscillation();
    noise();
    turningPoint();
    slowestSeriesGoverns();
    return dnstest::finish();
}
