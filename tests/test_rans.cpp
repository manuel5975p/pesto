// dns/rans.hpp closed forms and dns/walldist: the SA-neg constants and
// functions against their definitions (TMR spalart.html), the S~ limiter, the
// log-layer identity that pins cw1, the negative branch, the point-implicit
// combination, float/double agreement, and the channel wall distance.
// Host-only, milliseconds.
#include "dns/rans.hpp"

#include "dns/curvmesh.hpp"
#include "dns/remap.hpp"
#include "dns/walldist.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <random>

using namespace dns;
using dnstest::check;

namespace {

// Standard SA WITH ft2, written out independently of rans.hpp (TMR text):
// the positive branch of SA-neg must be exactly this. Returns {R, scale},
// scale = |production| + |destruction|: the operation order differs from
// rans.hpp, so the comparison is relative to the terms that cancel in R.
RansPoint<double> standardSa(double nut, double nu, double Om, double d)
{
    const double cb1 = 0.1355, sig = 2.0 / 3.0, cb2 = 0.622, k = 0.41;
    const double cw1 = cb1 / (k * k) + (1 + cb2) / sig;
    const double chi = nut / nu, c3 = chi * chi * chi;
    const double fv1 = c3 / (c3 + 7.1 * 7.1 * 7.1);
    const double fv2 = 1 - chi / (1 + chi * fv1);
    const double ft2 = 1.2 * std::exp(-0.5 * chi * chi);
    const double Sb = nut / (k * k * d * d) * fv2;
    const double St = Sb >= -0.7 * Om
                          ? Om + Sb
                          : Om + Om * (0.49 * Om + 0.9 * Sb) / ((0.9 - 1.4) * Om - Sb);
    const double r = St == 0 ? 10.0 : std::min(nut / (St * k * k * d * d), 10.0);
    const double g = r + 0.3 * (std::pow(r, 6) - r);
    const double fw = g * std::pow(65.0 / (std::pow(g, 6) + 64.0), 1.0 / 6.0);
    const double P = cb1 * (1 - ft2) * St * nut;
    const double D = (cw1 * fw - cb1 / (k * k) * ft2) * (nut / d) * (nut / d);
    return {P - D, std::abs(P) + std::abs(D)};
}

void constants()
{
    check(std::abs(saFv1(sa::cv1) - 0.5) < 1e-14, "fv1(cv1) = 1/2");
    check(std::abs(saFw(1.0) - 1.0) < 1e-14, "fw(r = 1) = 1");
    check(std::abs(sa::cw1 - 3.239067816775729) < 1e-14, "cw1 = 3.2390678...");
    check(saR(1.0, 1e-12, 1.0) == 10.0, "r capped at 10");
    check(saR(1.0, 0.0, 1.0) == 10.0, "S~ = 0 gives r = 10");
    check(saFn(0.0) == 1.0 && saFn(2.0) == 1.0, "fn = 1 for chi >= 0");
}

void limiter()
{
    double worstJump = 0, minSt = 1e300;
    for (double Om : {1e-3, 0.1, 1.0, 37.0, 1e4}) {
        const double at = -sa::c2 * Om;
        const double lo = saStilde(Om, std::nextafter(at, -1e300));
        const double hi = saStilde(Om, at);
        worstJump = std::max(worstJump, std::abs(hi - lo) / Om);
        for (int i = 0; i < 2000; ++i) {
            const double Sb = -50.0 * Om + i * 0.05 * Om;
            minSt = std::min(minSt, saStilde(Om, Sb) / Om);
        }
    }
    std::printf("S~ limiter: jump %.2e, min S~/Omega %.4f\n", worstJump, minSt);
    check(worstJump < 1e-12, "S~ continuous at Sbar = -c2 Omega");
    check(minSt > 0, "S~ > 0 on the sweep");
}

// Log layer: nu~ = kappa u y, Omega = u/(kappa y). Production - destruction
// + the analytic diffusion (1+cb2) kappa^2 u^2 / sigma vanishes as chi -> inf
// -- the identity cw1 = cb1/kappa^2 + (1+cb2)/sigma encodes. At finite chi
// fv2 ~ 1/chi shifts S~ and r = 1/(1 + fv2), leaving (cb1 + cw1 kappa^2
// fw'(1)) / chi ~ 1.1e-3 P at chi = 1e4 (measured), so the balance is pinned
// at chi = 1e6 and the 1/chi decay is checked. A wrong cw1 by 1e-3 would
// leave ~1.2e-2 P whatever chi is.
void logLayer()
{
    const double u = 1.0, y = 0.1, k = sa::kappa;
    const double prod = sa::cb1 * u * u;
    const double diff = (1 + sa::cb2) * k * k * u * u / sa::sigma;
    auto res = [&](double chi) {
        const double nut = k * u * y, nu = nut / chi;
        return (saSource(nut, nu, u / (k * y), y).R + diff) / prod;
    };
    const double r4 = res(1e4), r6 = res(1e6);
    std::printf("log layer: |res|/P %.3e at chi 1e4, %.3e at chi 1e6 (ratio "
                "%.1f)\n",
                std::abs(r4), std::abs(r6), r4 / r6);
    check(std::abs(r6) < 1e-3, "log-layer balance at chi 1e6 (pins cw1)");
    check(std::abs(r4 / r6 - 100.0) < 1.0, "the residual decays as 1/chi");
}

void negative()
{
    const double nu = 1e-3, Om = 3.0, d = 0.05;
    const double eps = 1e-14;
    const auto rp = saSource(eps, nu, Om, d), rn = saSource(-eps, nu, Om, d);
    check(std::abs(rp.R) < 1e-12 && std::abs(rn.R) < 1e-12,
          "R continuous through nu~ = 0 (both sides -> 0)");
    check(saSource(0.0, nu, Om, d).R == 0.0, "R(0) = 0");
    check(saMut(1.2, -0.3, nu) == 0.0 && saMut(1.2, 0.0, nu) == 0.0,
          "mu_t = 0 for nu~ <= 0");
    check(saFn(0.0) == 1.0, "fn(0) = 1");
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> U(0, 1);
    bool jPos = true, same = true;
    double worst = 0;
    for (int i = 0; i < 100000; ++i) {
        const double nuv = std::pow(10.0, -6 + 4 * U(rng));
        const double chi = (U(rng) * 2 - 0.5) * std::pow(10.0, 4 * U(rng) - 1);
        const double om = std::pow(10.0, -3 + 6 * U(rng)) * (U(rng) < 0.05 ? 0 : 1);
        const double dd = std::pow(10.0, -5 + 5 * U(rng));
        const auto P = saSource(chi * nuv, nuv, om, dd);
        jPos = jPos && P.J >= 0.0 && std::isfinite(P.R);
        if (chi >= 0) {
            const auto S = standardSa(chi * nuv, nuv, om, dd);
            const double sc = std::max(S.J, 1e-300);
            const double rel = std::abs(P.R - S.R) / sc;
            worst = std::max(worst, rel);
            same = same && rel < 1e-14;
        }
    }
    std::printf("SA-neg: positive branch vs standard SA, worst rel %.2e\n", worst);
    check(jPos, "J >= 0 and R finite on 1e5 random states");
    check(same, "positive branch == standard SA incl. ft2 (1e-14)");
}

void pointImplicitForm()
{
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> U(0, 1);
    // R = 0, comb = q: q (1 + cr J) / (1 + cr J) in float is six roundings
    // (J q, cr (J q), q + ..., cr J, 1 + ..., the quotient), each <= u =
    // 2^-24 relative: |out - q| <= 6u |q| <= 6 ulp of q in the worst binade
    // position. The masterplan's 1 ulp is below that bound (measured worst:
    // 3 ulp); the measured worst case is printed.
    bool pos = true;
    double worstUlp = 0;
    for (int i = 0; i < 100000; ++i) {
        const float q = float(U(rng) * 10), cr = float(U(rng)),
                    J = float(U(rng) * 100);
        const float out = pointImplicit(q, cr, 0.0f, J, q);
        const float ulp = std::nextafter(q, 1e30f) - q;
        worstUlp = std::max(worstUlp, double(std::abs(out - q) / ulp));
        const float comb = float(U(rng) * 5), Jq = float(U(rng) * 50);
        const float R = -Jq * q + float(U(rng)); // R + J q >= 0
        pos = pos && pointImplicit(comb, cr, R, Jq, q) >= 0.0f;
    }
    std::printf("point-implicit fixed point: worst %.2f ulp (float)\n", worstUlp);
    check(worstUlp <= 6.0, "R = 0 and comb = q give out = q (<= 6 ulp, the "
                           "6-rounding bound)");
    check(pointImplicit(3.0, 0.5, 0.0, 7.0, 3.0) == 3.0 &&
              pointImplicit(1.25f, 0.5f, 0.0f, 0.0f, 9.0f) == 1.25f,
          "exact where the arithmetic is: J = 0 is comb + cr R");
    check(pos, "out >= 0 for comb >= 0 and R + J q >= 0");
}

void floatDouble()
{
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> U(0, 1);
    double worst = 0;
    for (int i = 0; i < 20000; ++i) {
        const double nu = std::pow(10.0, -5 + 3 * U(rng));
        const double chi = (U(rng) * 1.2 - 0.2) * std::pow(10.0, 3 * U(rng));
        const double om = std::pow(10.0, -2 + 4 * U(rng));
        const double d = std::pow(10.0, -3 + 3 * U(rng));
        const auto D = saSource(chi * nu, nu, om, d);
        const auto F =
            saSource(float(chi * nu), float(nu), float(om), float(d));
        // Scale: the larger of production and destruction magnitudes.
        const double sc = std::abs(sa::cb1 * om * chi * nu) +
                          std::abs(sa::cw1 * (chi * nu / d) * (chi * nu / d)) +
                          1e-300;
        worst = std::max(worst, std::abs(double(F.R) - D.R) / sc);
    }
    std::printf("float vs double saSource: worst rel %.2e\n", worst);
    check(worst < 1e-5, "float and double sources agree (1e-5 rel)");
}

// SST log layer (F1 = 1, set 1): k = u^2/sqrt(beta*), omega = u/(sqrt(beta*)
// kappa y), nu_t = k/omega = kappa u y. The k equation balances exactly
// (P = beta* omega k, k uniform -> no diffusion). The omega equation's
// residual is (gamma1 - [beta1/beta* - sigma_w1 kappa^2/sqrt(beta*)]) u^2 /
// (kappa y)^2: zero with the DERIVED gamma1 (SST-Vm) and (5/9 - 0.55317)
// ~ 0.43% of the production with the 2003 gamma1 (SST-2003m). Both pinned.
void sstLogLayer()
{
    const double u = 1.0, y = 0.05, k2 = sst::kappa * sst::kappa;
    const double k = u * u / 0.3, om = u / (0.3 * sst::kappa * y);
    const double Om = u / (sst::kappa * y);
    const double mut = sstMut(1.0, k, om, Om, 1.0);
    check(std::abs(mut - sst::kappa * u * y) < 1e-15, "SST: log-layer mu_t = "
                                                       "kappa u y");
    const double diffW = sst::sigmaW1 * u * u / (0.3 * y * y);
    for (RansModel m : {RansModel::SstVm, RansModel::Sst2003m}) {
        const auto P = sstSource(m, 1.0, k, om, Om, mut, 1.0, 0.0);
        const double prod = sstGamma1(m) * Om * Om;
        const double rw = (P.Rw + diffW) / prod;
        const double want = m == RansModel::SstVm
                                ? 0.0
                                : (5.0 / 9.0 - (0.075 / 0.09 - 0.5 * k2 / 0.3)) /
                                      (5.0 / 9.0);
        std::printf("SST %s log layer: k residual %.2e, omega residual %.4e of "
                    "P (want %.4e)\n",
                    std::string(ransName(m)).c_str(), P.Rk / (mut * Om * Om), rw,
                    want);
        check(std::abs(P.Rk) < 1e-12 * mut * Om * Om,
              std::string(ransName(m)) + ": log-layer k equation balances");
        check(std::abs(rw - want) < 1e-12,
              std::string(ransName(m)) +
                  ": log-layer omega residual pins gamma1 (0 derived, 0.43% "
                  "for 5/9)");
    }
    check(std::abs(sstGamma1(RansModel::SstVm) - 0.5531666666666667) < 1e-12 &&
              sstGamma1(RansModel::Sst2003m) == 5.0 / 9.0 &&
              sstGamma2(RansModel::Sst2003m) == 0.44,
          "gamma1/gamma2: derived (Vm), 5/9 and 0.44 (2003m)");
}

void sstFunctions()
{
    const double k = 0.5, om = 50.0, nu = 1e-3, rho = 1.0, cd = 1e-3;
    const auto nearW = sstBlend(k, om, 1e-6, nu, rho, cd);
    const auto farW = sstBlend(k, om, 1e3, nu, rho, cd);
    check(nearW.F1 == 1.0 && nearW.F2 == 1.0, "F1, F2 -> 1 as d -> 0");
    check(farW.F1 < 1e-12, "F1 -> 0 as d -> infinity");
    check(sstBlendC(1.0, sst::sigmaK1, sst::sigmaK2) == sst::sigmaK1 &&
              sstBlendC(0.0, sst::sigmaK1, sst::sigmaK2) == sst::sigmaK2 &&
              sstBlendC(1.0, sst::beta1, sst::beta2) == sst::beta1 &&
              sstBlendC(0.0, sst::sigmaW1, sst::sigmaW2) == sst::sigmaW2,
          "blended constants hit set 1 / set 2 exactly at F1 = 1 / 0");
    // Production limiter: P = mut R^2 against c beta* rho omega k, c = 10
    // (2003m) / 20 (Vm).
    const double kk = 1.0, w = 1.0, mt = 1.0;
    for (auto [m, c] : {std::pair{RansModel::Sst2003m, 10.0},
                        std::pair{RansModel::SstVm, 20.0}}) {
        const double lim = c * sst::betaStar * kk * w;
        const double Rlo = std::sqrt(0.5 * lim), Rhi = std::sqrt(4.0 * lim);
        const auto a = sstSource(m, 1.0, kk, w, Rlo, mt, 1.0, 0.0);
        const auto b = sstSource(m, 1.0, kk, w, Rhi, mt, 1.0, 0.0);
        check(std::abs(a.Rk - (0.5 * lim - sst::betaStar * w * kk)) < 1e-14 &&
                  std::abs(b.Rk - (lim - sst::betaStar * w * kk)) < 1e-14,
              std::string(ransName(m)) + ": k production limiter switches at " +
                  std::to_string(int(c)) + " beta* rho omega k");
        // omega production: 2003m limited too (gamma rho lim / mu_t), Vm not.
        const double g = sstGamma1(m);
        const double wantB = m == RansModel::Sst2003m ? g * lim / mt
                                                      : g * Rhi * Rhi;
        check(std::abs(b.Rw - (wantB - sst::beta1 * w * w)) < 1e-12,
              std::string(ransName(m)) + ": omega production uses " +
                  (m == RansModel::Sst2003m ? "P~ (limited)" : "P (unlimited)"));
    }
    // mu_t branch: a1 omega vs R F2.
    check(std::abs(sstMut(1.0, 2.0, 10.0, 1.0, 1.0) - 2.0 / 10.0) < 1e-16,
          "mu_t = rho k / omega when a1 omega > R F2");
    check(std::abs(sstMut(1.0, 2.0, 1.0, 10.0, 0.5) - sst::a1 * 2.0 / 5.0) <
              1e-15,
          "mu_t = rho a1 k / (R F2) when R F2 > a1 omega");
    check(sstMut(1.0, 0.0, 1.0, 1.0, 1.0) == 0.0 && sstMut(1.0, -1.0, 1.0, 1.0, 1.0) == 0.0,
          "mu_t = 0 for k <= 0");
    // The negative cross-diffusion is implicit (J), the positive explicit.
    const auto neg = sstSource(RansModel::SstVm, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, -1.0);
    const auto pos = sstSource(RansModel::SstVm, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0);
    check(neg.Jw > pos.Jw && pos.Jw == 2.0 * sst::beta2 * 2.0,
          "negative cross-diffusion feeds J_omega; positive does not");
}

void wallDistance()
{
    const int nyT = 34;
    const double beta = 2.0, h1 = ChannelStretchY::firstCell(nyT, beta, 1.0);
    const Grid g = Grid::make(4, nyT, 3, 2.0, 2.0 + 2.0 * h1, 1.5);
    const auto m = CurvMesh::channelY(g, beta, 1.0);
    const auto d = channelWallDistance(m, 1.0);
    const ChannelStretchY R{g.ny * g.hy, g.ny, beta, 1.0};
    bool exact = true, ghost = true, pos = true;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const int js = std::clamp(j, 1, g.ny - 2);
                const double y = 0.5 * (R.mapY(js * g.hy) + R.mapY((js + 1) * g.hy));
                const double want = std::min(y, 2.0 - y);
                const double got = d[g.idx(i, j, k)];
                exact = exact && std::abs(got - want) <=
                                     2 * std::numeric_limits<double>::epsilon() * want;
                pos = pos && got > 0;
                if (j == 0 || j == g.ny - 1)
                    ghost = ghost && got == d[g.idx(i, js, k)];
            }
    check(exact, "wall distance = min(y, 2 - y) at the node-mean centroid");
    check(ghost, "ghost rows carry the mirrored row's d");
    check(pos, "d > 0 on every cell");
}

} // namespace

int main()
{
    constants();
    limiter();
    logLayer();
    negative();
    pointImplicitForm();
    floatDouble();
    sstLogLayer();
    sstFunctions();
    wallDistance();
    return dnstest::finish();
}
