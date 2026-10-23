#include "dns/prop/liftline.hpp"

#include "dns/prop/numeric.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <limits>
#include <numbers>
#include <thread>
#include <utility>

namespace dns::prop {
namespace {

using std::numbers::pi;
using Vec = std::vector<double>;

constexpr double kRTip = 0.5;            // tip radius of the unit-diameter blade
constexpr double kTanbMin = 0.01;        // floor on the wake pitch: never a flat sheet
constexpr int kHomotopySteps = 8;        // saturated-linear -> real polar
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kGammaTol = 1e-7;       // of the circulation scale 0.5 omega R c_max

double radians(double d) { return d * (pi / 180.0); }
const double kHMu = radians(1.5);        // slope stencil of the stall viscosity

// np.maximum: NaN in either operand propagates.
double npMax(double a, double b)
{
    if (std::isnan(a) || std::isnan(b)) return kNaN;
    return a > b ? a : b;
}

double maxAbs(const Vec& v)
{
    double m = 0.0;
    for (double x : v) m = npMax(m, std::fabs(x));
    return m;
}

// --- wake --------------------------------------------------------------------

// Wake angles of one helix: half step on the first turn (the near field of the
// control points), dpsi to one diameter, 2 dpsi beyond.
Vec helixPsi(double xper, double length, double D, double dpsi)
{
    const double end = length / xper;
    const double near = D / xper;
    Vec psi = num::arange(0.0, 2 * pi, 0.5 * dpsi);
    for (const Vec& part : {num::arange(2 * pi, near, dpsi),
                            num::arange(std::max(near, 2 * pi), end, 2 * dpsi)})
        psi.insert(psi.end(), part.begin(), part.end());
    std::erase_if(psi, [end](double p) { return !(p < end); });
    std::ranges::sort(psi);
    psi.erase(std::unique(psi.begin(), psi.end()), psi.end());
    psi.push_back(end);
    return psi;
}

// Straight segments of the reference blade's trailing helices, blade ->
// downstream. hub = image-hub radius: each helix at r gets a partner at
// hub^2/r with the same axial advance per radian and opposite strength; the
// root helix is its own image and cancels.
struct Wake
{
    std::vector<Vec3> a, b;
    std::vector<int> owner;   // node index
    Vec wt;                   // +-1
};

Wake wake(const Vec& nodes, const Vec& tanb, double length, double D, double dpsi,
          std::optional<double> hub)
{
    Wake w;
    for (std::size_t k = 0; k < nodes.size(); ++k)
    {
        const double r = nodes[k], t = tanb[k];
        std::vector<std::pair<double, double>> radii{{r, 1.0}};
        if (hub)
        {
            if (std::fabs(r - *hub) <= 1e-12 * *hub) continue;
            radii.emplace_back(*hub * *hub / r, -1.0);
        }
        const Vec psi = helixPsi(r * t, length, D, dpsi);
        for (const auto& [ri, wt] : radii)
        {
            std::vector<Vec3> pts(psi.size());
            for (std::size_t q = 0; q < psi.size(); ++q)
                pts[q] = {psi[q] * r * t, ri * std::cos(psi[q]), -ri * std::sin(psi[q])};
            for (std::size_t q = 0; q + 1 < pts.size(); ++q)
            {
                w.a.push_back(pts[q]);
                w.b.push_back(pts[q + 1]);
                w.owner.push_back(int(k));
                w.wt.push_back(wt);
            }
        }
    }
    return w;
}

// Runs body(i) for i in [0, n) on a few threads. Each i is independent and
// sequential inside, so the result does not depend on the thread count.
template <class F>
void parallelFor(std::size_t n, F&& body)
{
    const std::size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t nt = std::min<std::size_t>(std::min<std::size_t>(hw, 16), n);
    if (nt <= 1)
    {
        for (std::size_t i = 0; i < n; ++i) body(i);
        return;
    }
    std::vector<std::jthread> pool;
    for (std::size_t t = 0; t < nt; ++t)
        pool.emplace_back([&, t] {
            for (std::size_t i = t; i < n; i += nt) body(i);
        });
}

struct Influence
{
    num::Matrix ax, at;
};

// Induced (axial, tangential) velocity at the reference control points per
// unit Gamma of each horseshoe, all blades. Blade b's field at the reference
// points is the reference field at the points turned by -2 pi b/B, turned
// back: one wake, B point sets. Pre: nodes.size() == cps.size() + 1, tanb > 0.
Influence influence(const Vec& nodes, const Vec& cps, const Vec& tanb, int blades,
                    double core, double length, double D, double dpsi,
                    std::optional<double> hub)
{
    const std::size_t n = cps.size(), K = nodes.size();
    assert(K == n + 1);
    Vec th(static_cast<std::size_t>(blades));
    for (int b = 0; b < blades; ++b) th[std::size_t(b)] = 2.0 * pi * double(b) / blades;
    std::vector<Vec3> P;
    for (double t : th)
        for (double c : cps) P.push_back({0.0, c * std::cos(t), -c * std::sin(t)});
    const Wake w = wake(nodes, tanb, length, D, dpsi, hub);

    // U[p][j]: bound segment j (node j+1 -> node j, so Gamma > 0 is -e_r and
    // rho W x Gamma pulls upstream) plus helix j (+Gamma_j downstream) minus
    // helix j+1.
    std::vector<std::vector<Vec3>> U(P.size(), std::vector<Vec3>(n));
    parallelFor(P.size(), [&](std::size_t p) {
        std::vector<Vec3> G(K, Vec3{0, 0, 0});
        for (std::size_t m = 0; m < w.a.size(); ++m)
        {
            const Vec3 v = segmentVelocity(P[p], w.a[m], w.b[m], core);
            Vec3& g = G[std::size_t(w.owner[m])];
            for (int d = 0; d < 3; ++d) g[std::size_t(d)] += v[std::size_t(d)] * w.wt[m];
        }
        for (std::size_t j = 0; j < n; ++j)
        {
            const Vec3 bnd = segmentVelocity(P[p], {0.0, nodes[j + 1], 0.0},
                                             {0.0, nodes[j], 0.0}, core);
            for (int d = 0; d < 3; ++d)
                U[p][j][std::size_t(d)] =
                    bnd[std::size_t(d)] + G[j][std::size_t(d)] - G[j + 1][std::size_t(d)];
        }
    });
    Influence inf{num::Matrix(int(n)), num::Matrix(int(n))};
    for (int b = 0; b < blades; ++b)
    {
        const double st = std::sin(th[std::size_t(b)]), ct = std::cos(th[std::size_t(b)]);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
                const Vec3& u = U[std::size_t(b) * n + i][j];
                inf.ax(int(i), int(j)) += u[0];
                inf.at(int(i), int(j)) += u[1] * st + u[2] * ct;
            }
    }
    return inf;
}

// --- one operating point -------------------------------------------------------

struct State
{
    Vec g, ua, ut, va, vt, W, phi, alpha, re, cl, cd, cla, mu;
};

// liftline.Point: blade, flow, polar -> velocities and loads.
class Model
{
public:
    Model(const LiftGeometry& geo, const Polar* polar, double V, double om, double rho,
          double nu, int blades, bool hubImage, double stallVisc)
        : geo_(geo), polar_(polar), V_(V), om_(om), rho_(rho), nu_(nu), B_(blades),
          hubImage_(hubImage)
    {
        const std::size_t n = geo.cps.size();
        for (std::size_t i = 0; i < n; ++i) dr_.push_back(geo.nodes[i + 1] - geo.nodes[i]);
        R_ = geo.nodes.back();
        // Chattot's artificial viscosity, live only where dcl/dalpha < 0: mu =
        // kappa pi c |cl_a| / (32 dr) gives the grid-scale sawtooth back its
        // stability margin for kappa >= 1 and vanishes as dr -> 0.
        for (std::size_t i = 0; i < n; ++i)
            visc_.push_back(stallVisc * pi * geo.chord[i] / (32.0 * dr_[i]));
        d2diag_.assign(n, -2.0);
        d2diag_[n - 1] -= 1.0;                    // tip: Gamma = 0 at the node, odd ghost
        d2diag_[0] += hubImage ? 1.0 : -1.0;      // wall: even; free: odd
    }

    std::size_t size() const { return geo_.cps.size(); }
    double R() const { return R_; }
    double om() const { return om_; }
    int blades() const { return B_; }
    const Vec& dr() const { return dr_; }
    void setBlend(double b) { blend_ = b; }

    // (d2 @ g): the second difference with its end ghosts.
    Vec d2(const Vec& g) const
    {
        const std::size_t n = g.size();
        Vec out(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            double s = d2diag_[i] * g[i];
            if (i > 0) s += g[i - 1];
            if (i + 1 < n) s += g[i + 1];
            out[i] = s;
        }
        return out;
    }
    double d2at(std::size_t i, std::size_t j) const
    {
        if (i == j) return d2diag_[i];
        return (i + 1 == j || j + 1 == i) ? 1.0 : 0.0;
    }

    // The polar's own lift slope and zero-lift angle at Re `re`, clipped to its
    // own cl range on +-30 deg: the monotone start of the homotopy.
    void saturate(double re)
    {
        const Vec deg = num::linspace(-30.0, 30.0, 241);
        Vec al(deg.size()), cl(deg.size());
        for (std::size_t i = 0; i < deg.size(); ++i)
        {
            al[i] = radians(deg[i]);
            cl[i] = polar_->at(al[i], re).cl;
        }
        double a0 = 0.0, best = kInf;
        std::size_t pick = 0;
        bool found = false;
        for (std::size_t i = 0; i + 1 < cl.size(); ++i)
            if (cl[i] < 0.0 && cl[i + 1] >= 0.0 && std::fabs(al[i]) < best)
            {
                best = std::fabs(al[i]);
                pick = i;
                found = true;
            }
        if (found)
            a0 = al[pick] - cl[pick] * (al[pick + 1] - al[pick]) / (cl[pick + 1] - cl[pick]);
        const double h = radians(2.0);
        const double cp = polar_->at(a0 + h, re).cl, cm = polar_->at(a0 - h, re).cl;
        a0_ = a0;
        slope_ = std::max((cp - cm) / (2 * h), 1.0);
        clmin_ = *std::ranges::min_element(cl);
        clmax_ = *std::ranges::max_element(cl);
    }

    ClCd clcd(double alpha, double re) const
    {
        ClCd v = polar_->at(alpha, re);
        if (blend_ < 1.0)
        {
            const double w = num::mod(alpha + pi, 2 * pi) - pi;
            const double lin = std::clamp(slope_ * (w - a0_), clmin_, clmax_);
            v.cl = blend_ * v.cl + (1.0 - blend_) * lin;
        }
        return v;
    }

    State state(Vec g, Vec ua, Vec ut) const
    {
        State s;
        const std::size_t n = size();
        s.g = std::move(g);
        s.ua = std::move(ua);
        s.ut = std::move(ut);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double va = V_ + s.ua[i];
            const double vt = om_ * geo_.cps[i] - s.ut[i];
            const double W = std::hypot(va, vt);
            const double phi = std::atan2(va, vt);
            const double alpha = geo_.beta[i] - phi;
            const double re = W * geo_.chord[i] / nu_;
            const ClCd c = clcd(alpha, re);
            s.va.push_back(va);
            s.vt.push_back(vt);
            s.W.push_back(W);
            s.phi.push_back(phi);
            s.alpha.push_back(alpha);
            s.re.push_back(re);
            s.cl.push_back(c.cl);
            s.cd.push_back(c.cd);
        }
        return s;
    }

    // Per-unit-span thrust and torque, all blades (propulsive signs).
    std::pair<Vec, Vec> loads(const State& s) const
    {
        Vec dT, dQ;
        for (std::size_t i = 0; i < size(); ++i)
        {
            const double drag = 0.5 * rho_ * s.W[i] * geo_.chord[i] * s.cd[i];
            dT.push_back(B_ * (rho_ * s.g[i] * s.vt[i] - drag * s.va[i]));
            dQ.push_back(B_ * geo_.cps[i] * (rho_ * s.g[i] * s.va[i] + drag * s.vt[i]));
        }
        return {dT, dQ};
    }

    // Uniform actuator-disk inflow u0 consistent with strip thrust at u0: the
    // wake pitch J = 0 gets, having no freestream to pitch it.
    double seed() const
    {
        const double area = pi * (R_ * R_);
        const Vec zero(size(), 0.0);
        auto thrust = [&](double u) {
            const State s = state(zero, Vec(size(), u), zero);
            Vec dT(size());
            for (std::size_t i = 0; i < size(); ++i)
                dT[i] = B_ * 0.5 * rho_ * (s.W[i] * s.W[i]) * geo_.chord[i]
                        * (s.cl[i] * std::cos(s.phi[i]) - s.cd[i] * std::sin(s.phi[i]))
                        * dr_[i];
            return num::sum(dT);
        };
        auto gap = [&](double u) {
            const double T = std::max(thrust(u), -0.49 * rho_ * area * V_ * V_);
            return u - (-0.5 * V_ + std::sqrt(0.25 * V_ * V_ + T / (2 * rho_ * area)));
        };
        const double lo = -0.49 * V_, hi = om_ * R_;
        if (gap(lo) < 0.0 && 0.0 < gap(hi))
            return num::brentq(gap, lo, hi, 1e-12 * hi, 4 * std::numeric_limits<double>::epsilon(),
                               100);
        return 0.0;
    }

    // Hydrodynamic pitch at the nodes for the next wake (see liftline.py).
    Vec tanb(const State& s, const Vec& tanb, WakePitch mode) const
    {
        const Vec& nodes = geo_.nodes;
        const double floorV = 1e-9 * om_ * R_;
        Vec t(nodes.size());
        if (mode == WakePitch::Local)
        {
            Vec tc(size());
            for (std::size_t i = 0; i < size(); ++i) tc[i] = s.va[i] / npMax(s.vt[i], floorV);
            for (std::size_t k = 0; k < nodes.size(); ++k)
                t[k] = npMax(num::interp(nodes[k], geo_.cps, tc), kTanbMin);
            return t;
        }
        const std::size_t K = nodes.size();
        Vec g(K + 1, 0.0);
        for (std::size_t i = 0; i < size(); ++i) g[i + 1] = s.g[i];
        Vec cyl(K);
        for (std::size_t k = 0; k < K; ++k)
            cyl[k] = -B_ * (g[k + 1] - g[k]) / (4 * pi * nodes[k] * tanb[k]);
        if (hubImage_) cyl[0] = 0.0;   // root helix = its image
        for (std::size_t k = 0; k < K; ++k)
        {
            double ua = 0.0;
            for (std::size_t m = 0; m < K; ++m)
                if (nodes[m] > nodes[k]) ua += cyl[m];
            ua += 0.5 * cyl[k];
            double gn = k + 1 == K ? 0.0 : num::interp(nodes[k], geo_.cps, s.g);
            const double ut = B_ * gn / (4 * pi * nodes[k]);
            t[k] = npMax((V_ + ua) / npMax(om_ * nodes[k] - ut, floorV), kTanbMin);
        }
        return t;
    }

    const LiftGeometry& geo() const { return geo_; }
    double V() const { return V_; }
    const Vec& visc() const { return visc_; }

private:
    const LiftGeometry& geo_;
    const Polar* polar_;
    double V_, om_, rho_, nu_;
    int B_;
    bool hubImage_;
    Vec dr_, visc_, d2diag_;
    double R_ = 0;
    double blend_ = 1.0;
    double a0_ = 0, slope_ = 1, clmin_ = 0, clmax_ = 0;
};

// The Gamma residual on a frozen wake and its Jacobian (liftline.solve_gamma).
class GammaSystem
{
public:
    GammaSystem(const Model& m, const Influence& inf)
        : m_(m), inf_(inf)
    {
        scale_ = 0.5 * m.om() * m.R() * *std::ranges::max_element(m.geo().chord);
    }
    double scale() const { return scale_; }

    std::pair<Vec, State> res(const Vec& g) const
    {
        constexpr double h = 1e-3;
        State s = m_.state(g, num::matvec(inf_.ax, g), num::matvec(inf_.at, g));
        const Vec d2g = m_.d2(g);
        const std::size_t n = g.size();
        Vec r(n);
        bool finite = true;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double clp = m_.clcd(s.alpha[i] + h, s.re[i]).cl;
            const double clm = m_.clcd(s.alpha[i] - h, s.re[i]).cl;
            s.cla.push_back((clp - clm) / (2 * h));
            // mu from a slope over +-H_MU: continuous in alpha, so the residual
            // has a root even with a panel sitting on the kink.
            const double cp = m_.clcd(s.alpha[i] + kHMu, s.re[i]).cl;
            const double cm = m_.clcd(s.alpha[i] - kHMu, s.re[i]).cl;
            s.mu.push_back(m_.visc()[i] * npMax((cm - cp) / (2 * kHMu), 0.0));
            r[i] = (g[i] - 0.5 * s.W[i] * m_.geo().chord[i] * s.cl[i] - s.mu[i] * d2g[i])
                   / scale_;
            finite = finite && std::isfinite(r[i]);
        }
        if (!finite) std::ranges::fill(r, kInf);
        return {std::move(r), std::move(s)};
    }

    num::Matrix jac(const State& s) const
    {
        const int n = int(s.W.size());
        num::Matrix J(n);
        const Vec& c = m_.geo().chord;
        for (int i = 0; i < n; ++i)
        {
            const std::size_t I = std::size_t(i);
            const double W = s.W[I], va = s.va[I], vt = s.vt[I];
            for (int j = 0; j < n; ++j)
            {
                const double ax = inf_.ax(i, j), at = inf_.at(i, j);
                const double dW = (va * ax - vt * at) / W;
                const double dphi = (vt * ax + va * at) / (W * W);
                const double eye = i == j ? 1.0 : 0.0;
                J(i, j) = (eye - s.mu[I] * m_.d2at(I, std::size_t(j))
                           - 0.5 * c[I] * (s.cl[I] * dW - (W * s.cla[I]) * dphi))
                          / scale_;
            }
        }
        return J;
    }

private:
    const Model& m_;
    const Influence& inf_;
    double scale_ = 1;
};

Vec axpy(const Vec& g, double lam, const Vec& dg)
{
    Vec out(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) out[i] = g[i] + lam * dg[i];
    return out;
}

Vec negated(Vec v)
{
    for (double& x : v) x = -x;
    return v;
}

// Pseudo-transient continuation: implicit steps of dGamma/dtau = -R, the
// fallback past a fold where Newton has no nearby root. Step grows by
// switched evolution relaxation (tau ~ 1/|R|).
std::pair<Vec, bool> ptc(const GammaSystem& sys, const Vec& g0, int iters = 400)
{
    Vec g = g0;
    auto [r, s] = sys.res(g);
    double nr = num::norm(r);
    double tau = 0.1;
    for (int it = 0; it < iters; ++it)
    {
        if (!std::isfinite(nr)) return {g, false};
        if (maxAbs(r) < kGammaTol) return {g, true};
        num::Matrix J = sys.jac(s);
        for (int i = 0; i < J.n; ++i) J(i, i) += 1.0 / (tau * sys.scale());
        auto dg = num::solve(std::move(J), negated(r));
        if (!dg) return {g, false};
        auto [rt, st] = sys.res(axpy(g, 1.0, *dg));
        const double nt = num::norm(rt);
        if (!std::isfinite(nt))
        {
            tau *= 0.25;
            continue;
        }
        g = axpy(g, 1.0, *dg);
        r = std::move(rt);
        s = std::move(st);
        tau = std::min(tau * std::max(std::min(nr / std::max(nt, 1e-300), 4.0), 0.25), 1e12);
        nr = nt;
    }
    return {g, maxAbs(r) < kGammaTol};
}

// Gamma on a frozen wake: Newton with backtracking, then MINPACK hybrd, then
// pseudo-transient continuation. A plain fixed point is no option: the end
// panels of a cosine grid have self-influence ~ 1/(panel width).
std::pair<Vec, bool> solveGamma(const Model& m, const Influence& inf, const Vec& g0)
{
    const GammaSystem sys(m, inf);
    Vec g = g0;
    auto [r, s] = sys.res(g);
    double nr = num::norm(r);
    for (int it = 0; it < 80; ++it)
    {
        if (maxAbs(r) < kGammaTol) return {g, true};
        auto dg = num::solve(sys.jac(s), negated(r));
        if (!dg) break;
        double lam = 1.0;
        bool accepted = false;
        while (lam > 1e-6)
        {
            auto [rt, st] = sys.res(axpy(g, lam, *dg));
            const double nt = num::norm(rt);
            if (nt < (1.0 - 0.1 * lam) * nr)
            {
                g = axpy(g, lam, *dg);
                r = std::move(rt);
                s = std::move(st);
                nr = nt;
                accepted = true;
                break;
            }
            lam *= 0.5;
        }
        if (!accepted) break;
    }
    if (maxAbs(r) < kGammaTol) return {g, true};
    const Vec x = num::hybrd(
        [&](std::span<const double> xs, std::span<double> out) {
            const Vec rr = sys.res(Vec(xs.begin(), xs.end())).first;
            std::ranges::copy(rr, out.begin());
        },
        g, 1e-13);
    if (maxAbs(sys.res(x).first) < kGammaTol) return {x, true};
    return ptc(sys, g0);
}

// Gamma by continuation from the stall-saturated linear polar (monotone, one
// solution) to the real one on a frozen wake: the branch connected to attached
// flow, deterministically. A failed step is halved; at a fold the last try is
// a cold jump to blend 1.
std::pair<Vec, bool> homotopy(Model& m, const Influence& inf, Vec g)
{
    double b = 0.0, step = 1.0 / kHomotopySteps;
    m.setBlend(0.0);
    bool ok;
    std::tie(g, ok) = solveGamma(m, inf, g);
    while (ok && b < 1.0)
    {
        const double nb = std::min(b + step, 1.0);
        m.setBlend(nb);
        auto [gt, okt] = solveGamma(m, inf, g);
        if (okt)
        {
            g = std::move(gt);
            b = nb;
            step = std::min(2 * step, 1.0 / kHomotopySteps);
        }
        else if (step > 1.0 / (64 * kHomotopySteps))
            step *= 0.5;
        else
            break;
    }
    if (b < 1.0 || !ok)
    {
        m.setBlend(1.0);
        std::tie(g, ok) = solveGamma(m, inf, g);
    }
    m.setBlend(1.0);
    return {g, ok};
}

// Core radius: 5% of the closest control-point-to-node gap, so no control
// point ever sits in a trailing filament's core.
double autoCore(const LiftGeometry& geo)
{
    double gap = kInf;
    for (std::size_t i = 0; i < geo.cps.size(); ++i)
        gap = std::min(gap, std::min(geo.cps[i] - geo.nodes[i], geo.nodes[i + 1] - geo.cps[i]));
    return 0.05 * gap;
}

} // namespace

Vec3 segmentVelocity(const Vec3& p, const Vec3& a, const Vec3& b, double core)
{
    const Vec3 r1{p[0] - a[0], p[1] - a[1], p[2] - a[2]};
    const Vec3 r2{p[0] - b[0], p[1] - b[1], p[2] - b[2]};
    const Vec3 r0{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    const Vec3 cr{r1[1] * r2[2] - r1[2] * r2[1], r1[2] * r2[0] - r1[0] * r2[2],
                  r1[0] * r2[1] - r1[1] * r2[0]};
    const double cr2 = cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2];
    const double n1 = std::sqrt(r1[0] * r1[0] + r1[1] * r1[1] + r1[2] * r1[2]) + 1e-300;
    const double n2 = std::sqrt(r2[0] * r2[0] + r2[1] * r2[1] + r2[2] * r2[2]) + 1e-300;
    const double k = r0[0] * (r1[0] / n1 - r2[0] / n2) + r0[1] * (r1[1] / n1 - r2[1] / n2)
                     + r0[2] * (r1[2] / n1 - r2[2] / n2);
    const double l2 = r0[0] * r0[0] + r0[1] * r0[1] + r0[2] * r0[2];
    const double cl2 = core * core * l2;
    const double den = std::sqrt(cl2 * cl2 + cr2 * cr2) + 1e-300;
    const double f = k / den;
    return {cr[0] * f / (4.0 * pi), cr[1] * f / (4.0 * pi), cr[2] * f / (4.0 * pi)};
}

LiftGeometry liftGeometry(const Planform& pf, double hubRR, int panels, HubModel hub,
                          double D)
{
    assert(panels >= 1 && hubRR > 0 && hubRR < 1);
    const double rh = hubRR * kRTip;
    auto f = [hub](double t) {
        return hub == HubModel::Image ? std::sin(0.5 * pi * t)
                                      : 0.5 * (1 - std::cos(pi * t));
    };
    LiftGeometry g;
    for (int k = 0; k <= panels; ++k)
        g.nodes.push_back((rh + (kRTip - rh) * f(double(k) / panels)) * D);
    for (int k = 0; k < panels; ++k)
    {
        const double r = rh + (kRTip - rh) * f((double(k) + 0.5) / panels);
        const Section s = pf.at(r);
        g.cps.push_back(r * D);
        g.chord.push_back(s.chord * D);
        g.beta.push_back(s.beta);
    }
    return g;
}

LiftPoint liftlineSolve(double J, const Case& c, const LiftGeometry& geo,
                        const Polar& polar)
{
    const double D = c.op.diameter, n = c.op.revPerSec();
    const double om = 2 * pi * n;
    const LiftlineOptions& o = c.ll;
    Model m(geo, &polar, J * n * D, om, c.op.rho, c.op.nu, c.blades,
            o.hub == HubModel::Image, o.stallVisc);
    const std::size_t N = m.size();

    const double u0 = m.seed();
    const Vec zero(N, 0.0);
    State s = m.state(zero, Vec(N, u0), zero);
    // Strip circulation tapered by Prandtl's factor at the free end(s): Newton
    // starts on the attached branch instead of next to a free end's 1/h.
    Vec g(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        const double sphi = std::max(std::fabs(std::sin(s.phi[i])), 0.05);
        double f = m.blades() * (m.R() - geo.cps[i]) / (2 * geo.cps[i] * sphi);
        if (o.hub == HubModel::None)
            f = std::min(f, m.blades() * (geo.cps[i] - geo.nodes[0]) / (2 * geo.cps[i] * sphi));
        g[i] = 0.5 * s.W[i] * geo.chord[i] * s.cl[i] * 2 / pi * std::acos(std::exp(-f));
    }
    Vec tanb(geo.nodes.size());
    for (std::size_t k = 0; k < tanb.size(); ++k)
        tanb[k] = npMax((m.V() + u0) / (om * geo.nodes[k]), kTanbMin);
    const double core = o.core != 0.0 ? o.core * D * kRTip : autoCore(geo);
    const std::optional<double> hub =
        o.hub == HubModel::Image ? std::optional<double>(geo.nodes[0]) : std::nullopt;
    bool ok = false;
    double dpitch = kInf;
    m.saturate(num::interp(0.75 * m.R(), geo.cps, s.re));
    int it = 0;
    for (; it <= o.alignIters; ++it)
    {
        const Influence inf = influence(geo.nodes, geo.cps, tanb, c.blades, core,
                                        o.wakeLength * D, D, radians(o.segDeg), hub);
        ok = false;
        if (it > 0) std::tie(g, ok) = solveGamma(m, inf, g);
        if (!ok) std::tie(g, ok) = homotopy(m, inf, g);
        s = m.state(g, num::matvec(inf.ax, g), num::matvec(inf.at, g));
        if (!ok) break;   // no wake to realign from: flag, don't guess
        Vec next = tanb;
        for (int k = 0; k < (o.wakePitch == WakePitch::Mean ? 30 : 1); ++k)
            next = m.tanb(s, next, o.wakePitch);
        // The alignment residual: the wake solved on vs the wake it implies.
        dpitch = 0.0;
        for (std::size_t k = 0; k < next.size(); ++k)
            dpitch = npMax(dpitch, std::fabs(next[k] - tanb[k]) / next[k]);
        if (dpitch < o.alignTol) break;
        tanb = std::move(next);
    }
    LiftPoint p;
    auto [dT, dQ] = m.loads(s);
    Vec tdr(N), qdr(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        tdr[i] = dT[i] * m.dr()[i];
        qdr[i] = dQ[i] * m.dr()[i];
    }
    p.T = num::sum(tdr);
    p.Q = num::sum(qdr);
    p.re75 = num::interp(0.75 * m.R(), geo.cps, s.re);
    p.converged = ok && dpitch < o.alignTol;
    p.u0 = u0;
    p.dpitch = dpitch;
    p.core = core;
    p.iters = std::min(it, o.alignIters);
    p.g = s.g;
    p.ua = s.ua;
    p.ut = s.ut;
    p.va = s.va;
    p.vt = s.vt;
    p.W = s.W;
    p.phi = s.phi;
    p.alpha = s.alpha;
    p.re = s.re;
    p.cl = s.cl;
    p.cd = s.cd;
    p.dT = std::move(dT);
    p.dQ = std::move(dQ);
    return p;
}

LiftCurve liftlineCurve(const Case& c, const LiftGeometry& geo, const Polar& polar)
{
    LiftCurve out;
    for (double J : c.js)
    {
        LiftPoint p = liftlineSolve(J, c, geo, polar);
        out.rows.push_back(curvePoint(J, p.T, p.Q, p.re75, p.converged, c.op));
        if (c.dumpSpan && *c.dumpSpan == J) out.span = std::move(p);
    }
    if (c.dumpSpan && !out.span) out.span = liftlineSolve(*c.dumpSpan, c, geo, polar);
    return out;
}

std::string liftlineSpanTable(double J, const LiftGeometry& geo, const LiftPoint& p)
{
    const double R = geo.nodes.back();
    constexpr double deg = 180.0 / pi;
    std::string t = std::format(
        "  span at J={:g}: u0 seed {:.4f} m/s, {} realignments, core {:.2e} m, "
        "converged {}\n",
        J, p.u0, p.iters, p.core, int(p.converged));
    t += "  r/R      Gamma     alpha     phi       u_a       u_t       cl       cd"
         "        Re        dT/dr     dQ/dr\n";
    for (std::size_t i = 0; i < geo.cps.size(); ++i)
        t += std::format("  {:.4f} {:9.4f} {:8.3f} {:8.3f} {:9.4f} {:9.4f} {:8.4f} "
                         "{:8.5f} {:9.0f} {:9.4f} {:9.5f}\n",
                         geo.cps[i] / R, p.g[i], p.alpha[i] * deg, p.phi[i] * deg,
                         p.ua[i], p.ut[i], p.cl[i], p.cd[i], p.re[i], p.dT[i], p.dQ[i]);
    return t;
}

// --- self-test ----------------------------------------------------------------

namespace {

// One Betz-optimum check: lightly loaded, induced velocity normal to the rigid
// helicoid = (w/2) cos beta_i. Efficiency vs actuator disk; with tol, the
// Gamma shape vs Prandtl's approximation to Goldstein.
SelfCheck betz(int nb, double w, std::optional<double> tol)
{
    const int n = 40;
    const double R = 0.5, rh = 0.05, V = 1.0, lam = 0.25;
    const double om = V / (lam * R);
    LiftGeometry geo;
    for (int k = 0; k <= n; ++k)
        geo.nodes.push_back(rh + (R - rh) * 0.5 * (1 - std::cos(pi * double(k) / n)));
    for (int k = 0; k < n; ++k)
        geo.cps.push_back(rh + (R - rh) * 0.5 * (1 - std::cos(pi * (double(k) + 0.5) / n)));
    geo.chord.assign(std::size_t(n), 1.0);
    geo.beta.assign(std::size_t(n), 0.0);
    Vec tn, tc;
    for (double r : geo.nodes) tn.push_back((V + 0.5 * w) / (om * r));
    for (double r : geo.cps) tc.push_back((V + 0.5 * w) / (om * r));
    const Influence inf = influence(geo.nodes, geo.cps, tn, nb, autoCore(geo), 20.0, 1.0,
                                    radians(5.0), std::nullopt);
    num::Matrix M(n);
    Vec rhs(static_cast<std::size_t>(n)), cb(static_cast<std::size_t>(n)), sb(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
    {
        const std::size_t I = std::size_t(i);
        cb[I] = std::cos(std::atan(tc[I]));
        sb[I] = std::sin(std::atan(tc[I]));
        for (int j = 0; j < n; ++j) M(i, j) = inf.ax(i, j) * cb[I] + inf.at(i, j) * sb[I];
        rhs[I] = 0.5 * w * cb[I];
    }
    const Vec g = num::solve(M, rhs).value_or(Vec(std::size_t(n), kNaN));
    const Vec ua = num::matvec(inf.ax, g), ut = num::matvec(inf.at, g);
    Vec tp(static_cast<std::size_t>(n)), pp(static_cast<std::size_t>(n));
    for (std::size_t i = 0; i < std::size_t(n); ++i)
    {
        const double dr = geo.nodes[i + 1] - geo.nodes[i];
        tp[i] = g[i] * (om * geo.cps[i] - ut[i]) * dr;
        pp[i] = geo.cps[i] * g[i] * (V + ua[i]) * dr;
    }
    const double T = nb * num::sum(tp);
    const double P = om * nb * num::sum(pp);
    const double eta = T * V / P;
    const double ct = T / (0.5 * V * V * pi * R * R);
    const double ead = 2.0 / (1.0 + std::sqrt(1.0 + ct));
    bool ok = eta < ead && eta > 0.95 * ead;
    std::string msg = std::format("B={} w/V={}: eta_i {:.4f} < actuator disk {:.4f} "
                                  "(CT_disk {:.4f})",
                                  nb, w, eta, ead, ct);
    if (tol)
    {
        const double st = std::sin(std::atan(tc.back()));
        double dev = 0.0, gpMax = 0.0;
        Vec gp(static_cast<std::size_t>(n));
        for (std::size_t i = 0; i < std::size_t(n); ++i)
        {
            const double F = 2 / pi * std::acos(std::exp(-nb * (R - geo.cps[i]) / (2 * R * st)));
            gp[i] = 2 * pi * geo.cps[i] * F * w * sb[i] * cb[i] / nb;
            gpMax = std::max(gpMax, gp[i]);
        }
        for (std::size_t i = 0; i < std::size_t(n); ++i)
            if (geo.cps[i] > 0.4 * R && geo.cps[i] < 0.95 * R)
                dev = std::max(dev, std::fabs(g[i] - gp[i]));
        dev /= gpMax;
        ok = ok && dev < *tol;
        msg += std::format("; Gamma vs Prandtl max dev {:.4f} of peak on r/R 0.4-0.95", dev);
    }
    return {"betz", ok, msg};
}

} // namespace

std::vector<SelfCheck> liftlineSelfTest()
{
    std::vector<SelfCheck> out;
    // (1) Straight segment vs Gamma/(4 pi h) (cos t1 - cos t2).
    const Vec3 A{0.0, 0.0, -1.0}, B{0.0, 0.0, 1.5};
    double err = 0.0;
    for (const Vec3& p : {Vec3{0.3, 0.0, 0.2}, Vec3{0.0, -2.0, 3.0}, Vec3{0.05, 0.0, -1.4}})
    {
        const Vec3 v = segmentVelocity(p, A, B, 0.0);
        const double h = std::hypot(p[0], p[1]);
        const double c1 = (p[2] - A[2]) / std::hypot(p[0] - A[0], p[1] - A[1], p[2] - A[2]);
        const double c2 = (p[2] - B[2]) / std::hypot(p[0] - B[0], p[1] - B[1], p[2] - B[2]);
        const double s = (c1 - c2) / (4 * pi * h);
        const Vec3 ref{s * -p[1] / h, s * p[0] / h, 0.0};
        double num = 0.0, den = 0.0;
        for (int d = 0; d < 3; ++d)
        {
            num = std::max(num, std::fabs(v[std::size_t(d)] - ref[std::size_t(d)]));
            den = std::max(den, std::fabs(ref[std::size_t(d)]));
        }
        err = std::max(err, num / den);
    }
    out.push_back({"segment", err < 1e-12,
                   std::format("max rel err {:.1e} vs closed form", err)});
    // On its own line: zero; at h = core: Vatistas 1/sqrt2.
    const double core = 0.01;
    const Vec3 on = segmentVelocity({0.0, 0.0, 3.0}, A, B, core);
    const double v = segmentVelocity({core, 0.0, 0.2}, A, B, core)[1];
    const double v0 = segmentVelocity({core, 0.0, 0.2}, A, B, 0.0)[1];
    const double onMax = std::max({std::fabs(on[0]), std::fabs(on[1]), std::fabs(on[2])});
    out.push_back({"core",
                   onMax == 0.0 && std::fabs(v / v0 - 1.0 / std::sqrt(2.0)) < 1e-12,
                   std::format("on-line {:.0e}, h=core ratio {:.12f} (1/sqrt2)", onMax,
                               v / v0)});

    // (2) Semi-infinite helix cylinder, B = 24: u_x = gamma/2 at the disk and
    // gamma = B Gamma / p far downstream (actuator disk: half the far wake).
    {
        const int nb = 24;
        const double R = 0.5, p = 0.4, L = 40.0;
        const double tb = p / (2 * pi * R);
        const Wake w = wake({R}, {tb}, L * R, 2 * R, radians(5.0), std::nullopt);
        std::vector<Vec3> pts;
        for (double x : {0.0, 0.5 * L * R})
            for (double r : {0.15, 0.3})
            {
                const Vec ts = num::linspace(0, 2 * pi, 7);
                for (std::size_t q = 0; q + 1 < ts.size(); ++q)
                    pts.push_back({x, r * std::cos(ts[q]), r * std::sin(ts[q])});
            }
        Vec u(pts.size(), 0.0);
        parallelFor(pts.size(), [&](std::size_t q) {
            for (int b = 0; b < nb; ++b)
            {
                const double t = 2 * pi * b / nb, c = std::cos(t), s = std::sin(t);
                // P @ [[1,0,0],[0,c,-s],[0,s,c]]: the point turned by -t.
                const Vec3 pr{pts[q][0], pts[q][1] * c + pts[q][2] * s,
                              -pts[q][1] * s + pts[q][2] * c};
                for (std::size_t m = 0; m < w.a.size(); ++m)
                    u[q] += segmentVelocity(pr, w.a[m], w.b[m], 1e-6)[0];
            }
        });
        const double gam = nb / p;
        const double disk = num::mean(std::span(u).first(12));
        const double far = num::mean(std::span(u).subspan(12));
        out.push_back(
            {"cylinder",
             std::fabs(disk / (-0.5 * gam) - 1) < 0.01 && std::fabs(far / -gam - 1) < 0.01,
             std::format("disk {:.4f} gamma (0.5), far {:.4f} gamma (1), disk/far {:.4f}; "
                         "sign: a +Gamma downstream tip helix decelerates, as the bound "
                         "-e_r needs",
                         disk / -gam, far / -gam, disk / far)});
    }

    // (3) Betz optimum vs Prandtl/Goldstein shape and actuator-disk efficiency.
    out.push_back(betz(8, 0.02, 0.03));
    out.push_back(betz(2, 0.02, std::nullopt));
    out.push_back(betz(2, 0.2, std::nullopt));
    return out;
}

} // namespace dns::prop
