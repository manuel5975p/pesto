#include "dns/chem.hpp"

#include "dns/parallel.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <format>

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
    bool stiff = false;   // a step went stiff (h J > kStiffStep): m too large
    bool preheat = false; // p* fell below kColdCut of the preheat line
    double resid = 0;     // p - m cp (T - Tu) at the cold end
    double tCut = 0;      // rate switched off below this T (0 = never)
};

constexpr double kS0 = 1e-9;     // hot-end start, s
constexpr double kS1 = 1 - 1e-7; // cold-end stop, s
// h J above this: RK4 left its accuracy region on the quasi-steady manifold
// (stability bound 2.785). Slow shots peak at h J ~ 2e-3 (n 2e4, sigma
// 1.2-10, beta 2-20, measured), a collapsed fast shot reaches O(1).
constexpr double kStiffStep = 1.0;
// The bracket scan: m = 2^kScanLo .. 2^kScanHi (81 powers of two).
constexpr int kScanLo = -60, kScanHi = 20;

double logit(double s) { return std::log(s / (1 - s)); }
double logistic(double u) { return 1 / (1 + std::exp(-u)); }

Shot shoot(const FlameProblem& f, double m, int n, FlameEigen* table)
{
    const double dT = f.Tb - f.Tu, q = f.cp * dT;
    // Hot-end start: p = a (Tb - T), a the positive root of a^2 + m cp a =
    // lambda_b B with q omega ~ B (Tb - T) (linearised about Tb).
    double B = f.A * f.rho(f.Tb) * f.cp * std::exp(-f.Ta / f.Tb);
    if (f.rateFactor) B *= f.rateFactor(f.Tb, 0.0);
    const double lb = f.lambda(f.Tb);
    const double a =
        0.5 * (-m * f.cp + std::sqrt(m * m * f.cp * f.cp + 4 * lb * B));
    // The reaction term lambda q omega of the last evaluation (0 once cut)
    // and its s: the stiffness and cold-boundary tests read the base point's.
    bool cut = false;
    double g = 0, sLast = 0;
    auto rhs = [&](double u, double p, double& dp, double& dx) {
        const double s = logistic(u), T = f.Tb - dT * s;
        const double dTdu = -dT * s * (1 - s);
        const double lam = f.lambda(T);
        double om = f.A * f.rho(T) * s * std::exp(-f.Ta / T);
        if (f.rateFactor) om *= f.rateFactor(T, s);
        g = cut ? 0.0 : lam * q * om;
        sLast = s;
        dp = (m * f.cp - g / p) * dTdu;
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
    Shot r;
    auto zero = [&r] {
        r.hitZero = true;
        return r;
    };
    // 0: hot end (p rising), 1: past the peak of p, 2: the inert preheat
    // zone reached (p* < kColdCut m cp (T - Tu)).
    int phase = 0;
    for (int k = 0; k < n; ++k) {
        const double u = u0 + k * h;
        double k1p, k1x, k2p, k2x, k3p, k3x, k4p, k4x;
        rhs(u, p, k1p, k1x);
        {
            const double s = sLast, dTdu = dT * s * (1 - s); // |dT/du|
            if (h * g * dTdu / (p * p) > kStiffStep) {
                r.stiff = true;
                return r;
            }
            const double c = g / (m * f.cp * (m * f.cp * dT * (1 - s)));
            if (phase == 0 && k1p < 0) phase = 1;
            if (phase == 1 && c <= kColdCut) {
                phase = 2;
                r.preheat = true;
            } else if (phase == 2 && !cut && c > kColdCut) {
                cut = true;
                r.tCut = f.Tb - dT * s;
                rhs(u, p, k1p, k1x);
            }
        }
        const double p2 = p + 0.5 * h * k1p;
        if (!(p2 > 0)) return zero();
        rhs(u + 0.5 * h, p2, k2p, k2x);
        const double p3 = p + 0.5 * h * k2p;
        if (!(p3 > 0)) return zero();
        rhs(u + 0.5 * h, p3, k3p, k3x);
        const double p4 = p + h * k3p;
        if (!(p4 > 0)) return zero();
        rhs(u + h, p4, k4p, k4x);
        p += h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p);
        x += h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x);
        if (!(p > 0)) return zero();
        record(u + h);
    }
    const double Tend = f.Tb - dT * kS1;
    r.resid = p - m * f.cp * (Tend - f.Tu);
    return r;
}

bool tooFast(const Shot& s) { return s.hitZero || s.stiff || s.resid < 0; }

} // namespace

std::expected<FlameEigen, std::string> flameEigen(const FlameProblem& f, int n)
{
    assert(n >= 100 && f.Tu > 0 && f.Tb > f.Tu && f.cp > 0 && f.A > 0 &&
           f.Ta > 0 && f.rho && f.lambda);
    // The scan: exactly one slow -> fast transition (issues.md 72: the old
    // doubling/halving trusted a monotone verdict and bisected a spurious
    // one, Lambda = 2^14 at sigma 6.5 Sutherland).
    constexpr int nScan = kScanHi - kScanLo + 1;
    std::array<bool, nScan> fast{};
    for (int k = 0; k < nScan; ++k)
        fast[std::size_t(k)] =
            tooFast(shoot(f, std::ldexp(1.0, kScanLo + k), n, nullptr));
    if (fast.front())
        return std::unexpected(std::format(
            "flame eigenvalue: no slow bracket (every m >= 2^{} is too fast)",
            kScanLo));
    if (!fast.back())
        return std::unexpected(std::format(
            "flame eigenvalue: no fast bracket (every m <= 2^{} is too slow)",
            kScanHi));
    int first = -1, last = -1, count = 0;
    for (int k = 0; k + 1 < nScan; ++k)
        if (fast[std::size_t(k)] != fast[std::size_t(k + 1)]) {
            if (first < 0) first = k;
            last = k;
            ++count;
        }
    if (count != 1)
        return std::unexpected(std::format(
            "flame eigenvalue: non-monotone shot at m in [{:.6g}, {:.6g}]; "
            "raise n",
            std::ldexp(1.0, kScanLo + first), std::ldexp(1.0, kScanLo + last + 1)));
    // The shipped bracket: [lo, 1] below m = 1, [1, hi] above (its halving or
    // doubling stopped at the same power of two), so the bisection below
    // takes the old sequence of midpoints.
    double lo = std::ldexp(1.0, kScanLo + first), hi = 2 * lo;
    if (hi <= 1) hi = 1;
    else lo = 1;
    for (int k = 0; k < 200 && hi - lo > 1e-15 * hi; ++k) {
        const double mid = 0.5 * (lo + hi);
        (tooFast(shoot(f, mid, n, nullptr)) ? hi : lo) = mid;
    }
    FlameEigen e;
    e.m = 0.5 * (lo + hi);
    const Shot fin = shoot(f, e.m, n, &e);
    if (!fin.preheat)
        return std::unexpected(std::format(
            "flame eigenvalue: no inert preheat zone (the unburnt mixture's "
            "own rate never falls below {:g} of the preheat line: the cold-"
            "boundary difficulty; raise beta or sigma)",
            kColdCut));
    if (fin.tCut > 0 && fin.tCut - f.Tu > kColdCutMaxRise * (f.Tb - f.Tu))
        return std::unexpected(std::format(
            "flame eigenvalue: cold-boundary difficulty (the unburnt "
            "mixture's own rate is cut at {:.3g} of the rise above Tu, above "
            "{:g}: the eigenvalue is the cutoff's; raise beta or sigma)",
            (fin.tCut - f.Tu) / (f.Tb - f.Tu), kColdCutMaxRise));
    e.tCut = fin.tCut;
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

namespace {

// Non-dimensional: T in Tu, rho = 1/T (isobaric, p = R = 1), cp = 1, A = 1;
// then rescale the table to A = Lambda, m = 1 (x -> x m1, dT/dx -> dT/dx /
// m1: the A -> A / m1^2 similarity).
std::expected<PremixedEigen, std::string> rescaled(const FlameProblem& f, int n)
{
    auto e = flameEigen(f, n);
    if (!e) return std::unexpected(e.error());
    PremixedEigen r;
    const double m1 = e->m;
    r.Lambda = 1 / (m1 * m1);
    r.tab = std::move(*e);
    for (double& v : r.tab.x) v *= m1;
    for (double& v : r.tab.dTdx) v /= m1;
    r.tab.dth *= m1;
    r.tab.m = 1;
    r.dthOverDD = r.tab.dth;
    return r;
}

// The C2 jet's model at a premix of equivalence ratio phi (the jet diluted
// with air, or any fuel/air premix): Tb from the jet's q, the jet's Ta,
// A = 1. Rich (phi > 1): O deficient, rateFactor = rho (s Y_F,rem + Y_O1
// s_p). Lean (phi < 1): F deficient, Y_F = Y_F1 s_p, Y_O = Y_O,rem + s Y_F,
// Y_O,rem = Y_O1 - s Y_F1, q omega = cp dT A rho s_p e^(-Ta/T) rho (Y_O,rem
// + s Y_F1 s_p), Tb = 1 + q Y_F1/(cp Tu).
// law (null = C 1): C(xi) of the premix's fresh state, one constant across
// the flame (xi is conserved), as a factor of the rate.
FlameProblem premix2Problem(const PartialPremixModel& md, double phi,
                            const PhiRateLaw<double>* law = nullptr)
{
    PartialPremixModel at = md;
    at.phi = phi;
    FlameProblem f;
    f.Tu = 1;
    // The jet itself at sigma exactly (not through q's round trip).
    f.Tb = md.sigmaAt(phi);
    f.cp = 1;
    f.A = 1;
    f.Ta = md.theta();
    f.rho = [](double T) { return 1 / T; };
    f.lambda = [md](double T) { return md.lambdaRatio(T); };
    const double C = law ? phiRateC(*law, md, phi) : 1.0;
    if (phi > 1 && !law) {
        const double sYrem = at.s * at.yFrem(), yO1 = at.yO1();
        f.rateFactor = [sYrem, yO1](double T, double sp) {
            return (1 / T) * (sYrem + yO1 * sp);
        };
    } else if (phi > 1) {
        const double sYrem = at.s * at.yFrem(), yO1 = at.yO1();
        f.rateFactor = [sYrem, yO1, C](double T, double sp) {
            return (1 / T) * (sYrem + yO1 * sp) * C;
        };
    } else {
        const double yOrem = at.yOrem(), sYF1 = at.s * at.yF1();
        f.rateFactor = [yOrem, sYF1, C](double T, double sp) {
            return (1 / T) * (yOrem + sYF1 * sp) * C;
        };
    }
    return f;
}

// Clamped cubic spline (zero end slopes) through (xi_k, lnC_k): the second
// derivatives m2 by the Thomas algorithm, FP64. Pre: n >= 3, xi ascending.
void clampedSpline(PhiRateLaw<double>& L)
{
    const int n = L.n;
    std::array<double, PhiRateLaw<double>::kMaxNodes> a{}, b{}, c{}, d{};
    for (int k = 0; k < n; ++k) {
        const double hl = k > 0 ? L.xi[k] - L.xi[k - 1] : 0.0;
        const double hr = k + 1 < n ? L.xi[k + 1] - L.xi[k] : 0.0;
        const double sl = k > 0 ? (L.lnC[k] - L.lnC[k - 1]) / hl : 0.0;
        const double sr = k + 1 < n ? (L.lnC[k + 1] - L.lnC[k]) / hr : 0.0;
        a[std::size_t(k)] = hl;
        b[std::size_t(k)] = 2 * (hl + hr);
        c[std::size_t(k)] = hr;
        d[std::size_t(k)] = 6 * (sr - sl);
    }
    for (int k = 1; k < n; ++k) {
        const double w = a[std::size_t(k)] / b[std::size_t(k - 1)];
        b[std::size_t(k)] -= w * c[std::size_t(k - 1)];
        d[std::size_t(k)] -= w * d[std::size_t(k - 1)];
    }
    L.m2[n - 1] = d[std::size_t(n - 1)] / b[std::size_t(n - 1)];
    for (int k = n - 2; k >= 0; --k)
        L.m2[k] = (d[std::size_t(k)] - c[std::size_t(k)] * L.m2[k + 1]) /
                  b[std::size_t(k)];
}

// GRI-3.0 rows (tests/data/chem/ch4_gri30.csv, the SL column): every 0.05
// from 0.5 to 1.6, 0.975 / 1.025 / 1.075 at the peak, 1.7 beyond the jet's
// 1.6 (the zero-slope clamp's interval stays outside --equivalence's range,
// 0.5 likewise below --mix-phi's 0.6). The csv's other rows (0.575, 0.625,
// 1.125, 1.325, 1.375, 1.425, 1.575) are the held-out checks of test_flame1d.
constexpr Ch4SpeedNode kCh4Gri30[] = {
    {0.5, 0.047820312},  {0.55, 0.077558148}, {0.6, 0.11248302},
    {0.65, 0.15057316},  {0.7, 0.19010289},   {0.75, 0.22946721},
    {0.8, 0.26711442},   {0.85, 0.30161877},  {0.9, 0.33123534},
    {0.95, 0.35484518},  {0.975, 0.36388604}, {1.0, 0.37086092},
    {1.025, 0.37554031}, {1.05, 0.37775225},  {1.075, 0.37737606},
    {1.1, 0.37404952},   {1.15, 0.3579195},   {1.2, 0.32777164},
    {1.25, 0.28393749},  {1.3, 0.23123712},   {1.35, 0.17864667},
    {1.4, 0.13648153},   {1.45, 0.11423922},  {1.5, 0.099707104},
    {1.55, 0.088421817}, {1.6, 0.079144066},  {1.7, 0.064614159}};

} // namespace

std::expected<PremixedEigen, std::string> premixedEigen(const PremixedModel& md,
                                                        int n)
{
    FlameProblem f;
    f.Tu = 1;
    f.Tb = md.sigma;
    f.cp = 1;
    f.A = 1;
    f.Ta = md.theta();
    f.rho = [](double T) { return 1 / T; };
    f.lambda = [md](double T) { return md.lambdaRatio(T); };
    return rescaled(f, n);
}

std::expected<PremixedEigen, std::string>
premixedEigen2(const PartialPremixModel& md, int n)
{
    assert(md.phi > 1 && md.s > 0 && md.yO2 > 0 && md.sigma > 1);
    return rescaled(premix2Problem(md, md.phi), n);
}

std::expected<PremixedEigen, std::string>
premixedEigen2At(const PartialPremixModel& md, double phi,
                 const PhiRateLaw<double>* law, int n)
{
    assert(phi > 0 && phi != 1 && md.phi > 1);
    auto jet = flameEigen(premix2Problem(md, md.phi), n);
    if (!jet) return std::unexpected(jet.error());
    auto at = flameEigen(premix2Problem(md, phi, law), n);
    if (!at) return std::unexpected(at.error());
    // A = Lambda2 = 1/m_jet^2: the A = 1 table's x scales by m_jet, m by
    // 1/m_jet (the sqrt(A) similarity), rho_u = 1 in both.
    const double mj = jet->m;
    PremixedEigen r;
    r.Lambda = 1 / (mj * mj);
    r.tab = std::move(*at);
    r.tab.m = r.tab.m / mj;
    for (double& v : r.tab.x) v *= mj;
    for (double& v : r.tab.dTdx) v /= mj;
    r.tab.dth *= mj;
    r.dthOverDD = r.tab.dth;
    return r;
}

std::expected<double, std::string>
premixedSpeedRatio(const PartialPremixModel& md, double phi,
                   const PhiRateLaw<double>* law, int n)
{
    auto e = premixedEigen2At(md, phi, law, n);
    if (!e) return std::unexpected(e.error());
    return e->tab.m;
}

std::span<const Ch4SpeedNode> ch4Gri30Speeds() { return kCh4Gri30; }

double phiRateC(const PhiRateLaw<double>& L, const PartialPremixModel& md,
                double phi)
{
    assert(L.n >= 2 && phi > 0);
    PartialPremixModel at = md;
    at.phi = phi;
    return std::exp(phiRateLn(L, at.yF1(), at.yO1()));
}

std::expected<PhiRateLaw<double>, std::string>
phiRateLaw(const PartialPremixModel& md, std::span<const Ch4SpeedNode> nodes,
           int n)
{
    assert(md.phi > 1 && md.s > 0 && md.yO2 > 0);
    std::vector<Ch4SpeedNode> use;
    for (const Ch4SpeedNode& v : nodes)
        if (v.phi != 1) use.push_back(v);
    if (use.size() < 3)
        return std::unexpected("A(phi) law: fewer than 3 usable nodes");
    if (use.size() > std::size_t(PhiRateLaw<double>::kMaxNodes))
        return std::unexpected(std::format(
            "A(phi) law: {} nodes, at most {}", use.size(),
            PhiRateLaw<double>::kMaxNodes));
    // The UNCORRECTED eigenvalue at each node (A = 1), one solve per node.
    std::vector<std::expected<FlameEigen, std::string>> eig(
        use.size(), std::unexpected(std::string()));
    forItems(workerCount(0, use.size()), use.size(), [&](int, std::size_t k) {
        eig[k] = flameEigen(premix2Problem(md, use[k].phi), n);
    });
    PhiRateLaw<double> L;
    L.n = int(use.size());
    L.s = md.s;
    L.yO2 = md.yO2;
    L.invDen = 1 / (md.s + md.yO2);
    for (std::size_t k = 0; k < use.size(); ++k) {
        if (!eig[k])
            return std::unexpected(std::format("A(phi) law: node phi {:g}: {}",
                                               use[k].phi, eig[k].error()));
        PartialPremixModel at = md;
        at.phi = use[k].phi;
        L.xi[k] = at.yF1();
        L.lnC[k] = 2 * std::log(use[k].SL / eig[k]->m);
        if (k > 0 && !(L.xi[k] > L.xi[k - 1]))
            return std::unexpected("A(phi) law: nodes not ascending in phi");
    }
    // Normalise: ln C(xi_jet) = 0 (a constant shift; the spline is linear in
    // its data, so the second derivatives are rebuilt from the shifted values).
    clampedSpline(L);
    const double v = phiRateLn(L, md.yF1(), md.yO1());
    for (int k = 0; k < L.n; ++k) L.lnC[k] -= v;
    clampedSpline(L);
    return L;
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
