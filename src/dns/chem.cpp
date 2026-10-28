#include "dns/chem.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns {
namespace {

// One shooting pass at mass flux m. State (p, x) integrated in the logit
// variable u of s = (Tb - T)/(Tb - Tu): ds/du = s (1 - s) clusters steps at
// BOTH ends, where p -> 0 linearly (hot) and p -> m cp (T - Tu) (cold), so
// dx/du = -(lambda/p) dT/du stays bounded there instead of the 1/s of a
// uniform-T march.
struct Shot
{
    bool hitZero = false; // p reached 0 before the cold end: m too large
    double resid = 0;     // p - m cp (T - Tu) at the cold end
};

constexpr double kS0 = 1e-9;     // hot-end start, s
constexpr double kS1 = 1 - 1e-7; // cold-end stop, s

double logit(double s) { return std::log(s / (1 - s)); }
double logistic(double u) { return 1 / (1 + std::exp(-u)); }

Shot shoot(const FlameProblem& f, double m, int n, FlameEigen* table)
{
    const double dT = f.Tb - f.Tu, q = f.cp * dT;
    // Hot-end start: p = a (Tb - T), a the positive root of a^2 + m cp a =
    // lambda_b B with q omega ~ B (Tb - T) (linearised about Tb).
    const double B = f.A * f.rho(f.Tb) * f.cp * std::exp(-f.Ta / f.Tb);
    const double lb = f.lambda(f.Tb);
    const double a =
        0.5 * (-m * f.cp + std::sqrt(m * m * f.cp * f.cp + 4 * lb * B));
    auto rhs = [&](double u, double p, double& dp, double& dx) {
        const double s = logistic(u), T = f.Tb - dT * s;
        const double dTdu = -dT * s * (1 - s);
        const double lam = f.lambda(T);
        const double om = f.A * f.rho(T) * s * std::exp(-f.Ta / T);
        dp = (m * f.cp - lam * q * om / p) * dTdu;
        dx = lam / p * dTdu;
    };
    const double u0 = logit(kS0), u1 = logit(kS1), h = (u1 - u0) / n;
    double p = a * dT * kS0, x = 0;
    auto record = [&](double u) {
        if (!table) return;
        const double T = f.Tb - dT * logistic(u);
        table->T.push_back(T);
        table->x.push_back(x);
        table->dTdx.push_back(p / f.lambda(T));
    };
    record(u0);
    for (int k = 0; k < n; ++k) {
        const double u = u0 + k * h;
        double k1p, k1x, k2p, k2x, k3p, k3x, k4p, k4x;
        rhs(u, p, k1p, k1x);
        const double p2 = p + 0.5 * h * k1p;
        if (!(p2 > 0)) return {true, 0};
        rhs(u + 0.5 * h, p2, k2p, k2x);
        const double p3 = p + 0.5 * h * k2p;
        if (!(p3 > 0)) return {true, 0};
        rhs(u + 0.5 * h, p3, k3p, k3x);
        const double p4 = p + h * k3p;
        if (!(p4 > 0)) return {true, 0};
        rhs(u + h, p4, k4p, k4x);
        p += h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p);
        x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x);
        if (!(p > 0)) return {true, 0};
        record(u + h);
    }
    const double Tend = f.Tb - dT * kS1;
    return {false, p - m * f.cp * (Tend - f.Tu)};
}

// The cold residual is decreasing in m; "hit zero" counts as negative.
bool tooFast(const Shot& s) { return s.hitZero || s.resid < 0; }

} // namespace

std::expected<FlameEigen, std::string> flameEigen(const FlameProblem& f, int n)
{
    assert(n >= 100 && f.Tu > 0 && f.Tb > f.Tu && f.cp > 0 && f.A > 0 &&
           f.Ta > 0 && f.rho && f.lambda);
    // Bracket by doubling/halving from m = 1, then bisect to round-off.
    double lo = 1, hi = 1;
    if (tooFast(shoot(f, 1.0, n, nullptr))) {
        for (int k = 0; k < 200 && tooFast(shoot(f, lo, n, nullptr)); ++k)
            lo *= 0.5;
        if (tooFast(shoot(f, lo, n, nullptr)))
            return std::unexpected("flame eigenvalue: no slow bracket");
    } else {
        for (int k = 0; k < 200 && !tooFast(shoot(f, hi, n, nullptr)); ++k)
            hi *= 2;
        if (!tooFast(shoot(f, hi, n, nullptr)))
            return std::unexpected("flame eigenvalue: no fast bracket");
    }
    for (int k = 0; k < 200 && hi - lo > 1e-15 * hi; ++k) {
        const double mid = 0.5 * (lo + hi);
        (tooFast(shoot(f, mid, n, nullptr)) ? hi : lo) = mid;
    }
    FlameEigen e;
    e.m = 0.5 * (lo + hi);
    shoot(f, e.m, n, &e);
    // Increasing-T order, x shifted to 0 at the mid temperature.
    std::ranges::reverse(e.T);
    std::ranges::reverse(e.x);
    std::ranges::reverse(e.dTdx);
    const double Tm = 0.5 * (f.Tu + f.Tb);
    const auto it = std::ranges::upper_bound(e.T, Tm);
    assert(it != e.T.begin() && it != e.T.end());
    const std::size_t k = std::size_t(it - e.T.begin());
    const double w = (Tm - e.T[k - 1]) / (e.T[k] - e.T[k - 1]);
    const double x0 = e.x[k - 1] + w * (e.x[k] - e.x[k - 1]);
    for (double& v : e.x) v -= x0;
    // Newton on the Hermite profile itself, so T(0) is the mid temperature
    // of the interpolant the IC reads, not of the linear chord.
    for (int it = 0; it < 4; ++it) {
        const double d = (Tm - flameProfile(e, 0.0)) / e.dTdx[k];
        for (double& v : e.x) v -= d;
    }
    e.dth = (f.Tb - f.Tu) / std::ranges::max(e.dTdx);
    return e;
}

std::expected<PremixedEigen, std::string> premixedEigen(const PremixedModel& md,
                                                        int n)
{
    // Non-dimensional: T in Tu, rho = 1/T (isobaric, p = R = 1), cp = 1,
    // lambda = lambdaRatio(T), A = 1; then rescale to A = Lambda, m = 1.
    FlameProblem f;
    f.Tu = 1;
    f.Tb = md.sigma;
    f.cp = 1;
    f.A = 1;
    f.Ta = md.theta();
    f.rho = [](double T) { return 1 / T; };
    f.lambda = [md](double T) { return md.lambdaRatio(T); };
    auto e = flameEigen(f, n);
    if (!e) return std::unexpected(e.error());
    PremixedEigen r;
    const double m1 = e->m;
    r.Lambda = 1 / (m1 * m1);
    r.tab = std::move(*e);
    // x -> x m1, dT/dx -> dT/dx / m1: the A -> A / m1^2 similarity.
    for (double& v : r.tab.x) v *= m1;
    for (double& v : r.tab.dTdx) v /= m1;
    r.tab.dth *= m1;
    r.tab.m = 1;
    r.dthOverDD = r.tab.dth;
    return r;
}

double flameProfile(const FlameEigen& e, double x)
{
    assert(e.T.size() >= 2);
    if (x <= e.x.front()) return e.T.front();
    if (x >= e.x.back()) return e.T.back();
    const auto it = std::ranges::upper_bound(e.x, x);
    const std::size_t k = std::size_t(it - e.x.begin()) - 1;
    const double h = e.x[k + 1] - e.x[k], t = (x - e.x[k]) / h;
    const double t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * e.T[k] + (t3 - 2 * t2 + t) * h * e.dTdx[k] +
           (-2 * t3 + 3 * t2) * e.T[k + 1] + (t3 - t2) * h * e.dTdx[k + 1];
}

} // namespace dns
