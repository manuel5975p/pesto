#include "dns/prop/bemt.hpp"

#include "dns/prop/numeric.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>

namespace dns::prop {
namespace {

using std::numbers::pi;

constexpr double kEps = 1e-7;        // bracket offset from the phi = 0 pole
constexpr int kRePasses = 3;         // W0 = |V, Omega r|, then two from solved W
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Prandtl tip x hub loss at radius r, inflow |sin phi|. In (0, 1].
double lossFactor(int B, double r, double R, double rh, double sphi)
{
    const double s = std::max(std::fabs(sphi), 1e-12);
    const double ft = 0.5 * B * (R - r) / (r * s);
    const double fh = 0.5 * B * (r - rh) / (rh * s);
    const double F = (4.0 / (pi * pi)) * std::acos(std::exp(-ft)) * std::acos(std::exp(-fh));
    return std::max(F, 1e-9);
}

// Turbine-convention axial induction for loading k_t = -k > 2/3 (Buhl).
double buhlAt(double k, double F)
{
    const double kt = -k;
    const double g1 = 2.0 * F * kt - (10.0 / 9.0 - F);
    const double g2 = 2.0 * F * kt - F * (4.0 / 3.0 - F);
    const double g3 = 2.0 * F * kt - (25.0 / 9.0 - 2.0 * F);
    if (std::fabs(g3) < 1e-6) return 1.0 - 1.0 / (2.0 * std::sqrt(g2));
    return (g1 - std::sqrt(g2)) / g3;
}

struct Station
{
    double resid, alpha, cl, cd, cn, ct, F, k;
};

// Everything the residual and the loads need at element i, inflow phi.
Station stationTerms(double phi, std::size_t i, const BemtBlade& bl, double lam,
                     const Polar& polar, double re)
{
    const double s = std::sin(phi), c = std::cos(phi);
    const double alpha = bl.beta[i] - phi;
    const ClCd cc = polar.at(alpha, re);
    const double cn = cc.cl * c - cc.cd * s;   // along the thrust, upstream
    const double ct = cc.cl * s + cc.cd * c;   // against the rotation
    const double F = lossFactor(bl.B, bl.r[i], bl.R, bl.rh, s);
    const double k = bl.sigma[i] * cn / (4.0 * F * s * s);
    const double ax = k >= -2.0 / 3.0 ? s * (1.0 - k)   // sin(phi)/(1 + a), momentum
                                      : s / (1.0 - buhlAt(k, F));
    const double resid = ax - lam * c - lam * bl.sigma[i] * ct / (4.0 * F * s);
    return {resid, alpha, cc.cl, cc.cd, cn, ct, F, k};
}

// Ning's bracketed root for element i; NaN when no bracket changes sign.
double solvePhi(std::size_t i, const BemtBlade& bl, double lam, const Polar& polar,
                double re)
{
    const auto f = [&](double p) { return stationTerms(p, i, bl, lam, polar, re).resid; };
    const double h = pi / 2;
    const double brackets[4][2] = {{kEps, h}, {h, pi - kEps}, {-h, -kEps}, {-pi + kEps, -h}};
    for (const auto& br : brackets)
    {
        const double flo = f(br[0]), fhi = f(br[1]);
        if (flo == 0.0) return br[0];
        if (flo * fhi < 0.0) return num::brentq(f, br[0], br[1], 1e-12, 1e-12, 200);
    }
    return kNaN;
}

double nanMax(std::span<const double> v)
{
    double m = kNaN;
    for (double x : v)
        if (!std::isnan(x) && (std::isnan(m) || x > m)) m = x;
    return m;
}

// np.nansum: NaN entries count as 0.
double nanSum(std::span<const double> v)
{
    std::vector<double> w(v.begin(), v.end());
    for (double& x : w)
        if (std::isnan(x)) x = 0.0;
    return num::sum(w);
}

} // namespace

BemtBlade bemtBlade(const Planform& pf, int blades, double hubRR, int n, double D)
{
    assert(blades >= 1 && hubRR > 0 && hubRR < 1 && n >= 4 && D > 0);
    const double R = 0.5, rh = hubRR * 0.5;
    std::vector<double> rb(std::size_t(n) + 1);
    for (int k = 0; k <= n; ++k)
    {
        const double x = 0.5 * (1.0 - std::cos(pi * double(k) / double(n)));
        rb[std::size_t(k)] = rh + (R - rh) * x;
    }
    BemtBlade bl;
    bl.B = blades;
    bl.D = D;
    bl.R = R * D;
    bl.rh = rh * D;
    for (int k = 0; k < n; ++k)
    {
        const double rm = 0.5 * (rb[std::size_t(k)] + rb[std::size_t(k) + 1]);
        const Section s = pf.at(rm);
        bl.r.push_back(rm * D);
        bl.dr.push_back((rb[std::size_t(k) + 1] - rb[std::size_t(k)]) * D);
        bl.c.push_back(s.chord * D);
        bl.beta.push_back(s.beta);
    }
    for (std::size_t i = 0; i < bl.r.size(); ++i)
        bl.sigma.push_back(bl.B * bl.c[i] / (2.0 * pi * bl.r[i]));
    return bl;
}

BemtResult bemtSolve(double J, const BemtBlade& bl, const Operating& op,
                     const Polar& polar)
{
    const double n = op.revPerSec();
    const double om = 2.0 * pi * n;
    const double V = J * n * bl.D;
    const std::size_t N = bl.r.size();
    std::vector<double> W(N);
    for (std::size_t i = 0; i < N; ++i) W[i] = std::hypot(V, om * bl.r[i]);

    BemtResult res;
    BemtSpan& s = res.span;
    const int passes = polar.thin() ? 1 : kRePasses;
    for (int pass = 0; pass < passes; ++pass)
    {
        s = BemtSpan{};
        for (std::size_t i = 0; i < N; ++i)
        {
            s.re.push_back(W[i] * bl.c[i] / op.nu);
            s.mach.push_back(W[i] / op.sound);
        }
        for (std::size_t i = 0; i < N; ++i)
        {
            const double lam = V / (om * bl.r[i]);
            const double phi = solvePhi(i, bl, lam, polar, s.re[i]);
            if (!std::isfinite(phi))
            {
                for (auto* v : {&s.phi, &s.alpha, &s.cl, &s.cd, &s.cn, &s.ct, &s.F,
                                &s.k, &s.kp, &s.W, &s.Va, &s.Vt})
                    v->push_back(kNaN);
                continue;
            }
            const Station t = stationTerms(phi, i, bl, lam, polar, s.re[i]);
            const double kp =
                bl.sigma[i] * t.ct / (4.0 * t.F * std::sin(phi) * std::cos(phi));
            const double vt = om * bl.r[i] / (1.0 + kp);   // Omega r (1 - a')
            const double w = std::fabs(vt / std::cos(phi));
            const double va = w * std::sin(phi);           // V (1 + a)
            s.phi.push_back(phi);
            s.alpha.push_back(t.alpha);
            s.cl.push_back(t.cl);
            s.cd.push_back(t.cd);
            s.cn.push_back(t.cn);
            s.ct.push_back(t.ct);
            s.F.push_back(t.F);
            s.k.push_back(t.k);
            s.kp.push_back(kp);
            s.W.push_back(w);
            s.Va.push_back(va);
            s.Vt.push_back(vt);
        }
        for (std::size_t i = 0; i < N; ++i)
            if (std::isfinite(s.W[i])) W[i] = s.W[i];
    }

    std::vector<double> dT(N), dQ(N), absdT(N), err(N, kNaN);
    for (std::size_t i = 0; i < N; ++i)
    {
        const double q = 0.5 * op.rho * (s.W[i] * s.W[i]) * bl.c[i] * bl.B;
        s.dTdr.push_back(q * s.cn[i]);
        s.dQdr.push_back(q * s.ct[i] * bl.r[i]);
        s.a.push_back(V > 0 ? (s.Va[i] - V) / V : kNaN);
        s.ap.push_back(1.0 - s.Vt[i] / (om * bl.r[i]));
        dT[i] = s.dTdr[i] * bl.dr[i];
        dQ[i] = s.dQdr[i] * bl.dr[i];
        absdT[i] = std::fabs(s.dTdr[i]);
        // Momentum check: annulus thrust 4 pi r rho Va (Va - V) F, where it holds.
        const double mom =
            4.0 * pi * bl.r[i] * op.rho * s.Va[i] * (s.Va[i] - V) * s.F[i];
        if (s.k[i] >= -2.0 / 3.0 && s.phi[i] > 0 && s.phi[i] < pi / 2)
            err[i] = std::fabs(s.dTdr[i] - mom);
        if (!std::isfinite(s.phi[i])) ++res.unsolved;
    }
    double scale = nanMax(absdT);
    if (!(scale != 0.0) || std::isnan(scale)) scale = 1.0;   // `nanmax or 1.0`
    const double em = nanMax(err);
    res.momErr = std::isnan(em) ? 0.0 : em / scale;
    res.T = nanSum(dT);
    res.Q = nanSum(dQ);
    res.P = 2.0 * pi * n * res.Q;
    res.re75 = num::interp(0.75 * bl.R, bl.r, s.re);
    return res;
}

BemtCurve bemtCurve(const Case& c, const BemtBlade& bl, const Polar& polar)
{
    BemtCurve out;
    const double n = c.op.revPerSec(), om = 2.0 * pi * n;
    double w0 = 0;
    for (double J : c.js)
        for (double r : bl.r) w0 = std::max(w0, std::hypot(J * n * c.op.diameter, om * r));
    out.machHi = 1.15 * w0 / c.op.sound;
    for (double J : c.js)
    {
        const BemtResult r = bemtSolve(J, bl, c.op, polar);
        out.rows.push_back(curvePoint(J, r.T, r.Q, r.re75, true, c.op));
        out.worstMomErr = std::max(out.worstMomErr, r.momErr);
        out.unsolved += r.unsolved;
    }
    return out;
}

std::string bemtSpanTable(double J, const BemtBlade& bl, const BemtResult& res)
{
    const BemtSpan& s = res.span;
    constexpr double deg = 180.0 / pi;
    std::string t = std::format("span at J = {:g}: T = {:.4f} N, Q = {:.5f} N m\n", J,
                                res.T, res.Q);
    t += "   r/R   alpha    phi        a       a'      cl       cd       Re    dT/dr"
         "    dQ/dr\n";
    for (std::size_t i = 0; i < bl.r.size(); ++i)
        t += std::format("  {:5.3f} {:6.2f} {:6.2f} {:8.4f} {:8.4f} {:7.4f} {:8.5f} "
                         "{:8.3g} {:8.3f} {:8.4f}\n",
                         bl.r[i] / bl.R, s.alpha[i] * deg, s.phi[i] * deg, s.a[i],
                         s.ap[i], s.cl[i], s.cd[i], s.re[i], s.dTdr[i], s.dQdr[i]);
    return t;
}

} // namespace dns::prop
