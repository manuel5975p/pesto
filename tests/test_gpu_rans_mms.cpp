// Residual-based manufactured solutions for the RANS scalar equations on the
// body-fitted O-grid annulus (masterplan_flowrans P1, docs/rans.md "dns
// flow"). A smooth manufactured state (rho, u, p, nu~ | k, omega, d) is put on
// every cell centroid, the GPU residual kernel evaluates R_h at it, and the
// continuous operator L is evaluated on the host in FP64 at the same points
// (nested 4th-order central differences; its own error is measured against a
// halved step and must sit <= 1e-3 of the finest discretization error).
// Three levels of fixed beta; the volume-weighted L2 of (R_h - L) / L2(L)
// must fall monotonically with the finest-pair order p_obs in
// [p - 0.25, p + 0.5]:
//   u = 0 (diffusion, sources, cb2, grad rho; SST: diffusion + destruction)
//         -> p = 2
//   u != 0 (first-order upwind convection on the mean mass flux) -> p = 1
//   SA convection isolated (d = 1e6, tiny nu and nu~) -> p = 1
// SST runs in its F1 = F2 = 1 branch (d = 1e-3 drives both blend arguments
// past their clamp of 10, a1 omega > S): the smooth inner k-omega branch,
// where no max/min of the closure switches inside the domain. Seconds.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/rans.hpp"
#include "dns/gpu/axis_bc.hpp"
#include "dns/gpu/comp.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <format>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

constexpr double kGam = 1.4, kR = 0.5, kRFar = 2.5, kBeta = 1.0;
constexpr double kWave = 6.0; // manufactured wavenumber: h k ~ 0.1 at the finest

// One manufactured case: which closure, whether there is a velocity, and the
// scales. Every field is an analytic function of (x, y, z). The annulus
// cases are z-independent (nz == 1); the WEDGE cases (revolved) are
// axisymmetric about x: scalars are even functions of rho = |(y, z)|, the
// velocity is (u(x, rho), g(x) (y, z)) -- smooth through the axis.
struct Mfg
{
    RansModel model = RansModel::SaNeg;
    bool flow = true;
    double mu = 0.02, U0 = 1.0;
    double phiScale = 1.0;      // SA: nu~ in units of nu; SST: k scale
    double omega0 = 50.0;       // SST omega scale
    double dConst = 0.0;        // > 0: a constant wall distance
    bool revolved = false;      // the sphere's wedge instead of the annulus

    // The transverse profile: cos(k y + a) on the annulus, cos(k rho) on
    // the wedge (even in rho).
    double tr(double y, double z, double a) const
    {
        return revolved ? std::cos(kWave * std::hypot(y, z))
                        : std::cos(kWave * y + a);
    }
    double rho(double x, double y, double z) const
    {
        return 1.0 + 0.1 * std::sin(kWave * x + 0.3) * tr(y, z, -0.2);
    }
    double u(double, double y, double z) const
    {
        return flow ? U0 * (0.8 + 0.2 * tr(y, z, 0.1)) : 0.0;
    }
    // The transverse velocity: v on the annulus; g(x) y on the wedge.
    double v(double x, double y, double) const
    {
        if (!flow) return 0.0;
        const double gx = U0 * 0.1 * std::sin(kWave * x - 0.4);
        return revolved ? gx * y : gx;
    }
    double w(double x, double, double z) const
    {
        return flow && revolved ? U0 * 0.1 * std::sin(kWave * x - 0.4) * z
                                : 0.0;
    }
    double p(double x, double y, double z) const
    {
        return 20.0 * (1.0 + 0.01 * std::cos(kWave * x) * tr(y, z, 0.0));
    }
    // SA nu~ (chi in [0.25, 0.75] x phiScale: fv2 > 0, no limiter switch) or
    // SST k.
    double phi0(double x, double y, double z) const
    {
        const double s = 0.5 + 0.25 * std::sin(kWave * x - 0.5) *
                                   tr(y, z, 0.7 - 1.5707963267948966);
        if (model == RansModel::SaNeg) return phiScale * mu / rho(x, y, z) * s;
        return phiScale * 2.0 * s;
    }
    double phi1(double x, double y, double z) const
    {
        return omega0 * (1.0 + 0.3 * std::cos(kWave * x + 0.2) *
                                   tr(y, z, -0.3 - 1.5707963267948966));
    }
    double d(double x, double y, double z) const
    {
        if (dConst > 0) return dConst;
        return (revolved ? std::sqrt(x * x + y * y + z * z) : std::hypot(x, y)) -
               kR + 0.3;
    }
};

using F3 = std::function<double(double, double, double)>;

// 4th-order central first derivatives.
double dd(const F3& f, double x, double y, double z, double h, int a)
{
    auto at = [&](double s) {
        return f(x + (a == 0) * s, y + (a == 1) * s, z + (a == 2) * s);
    };
    return (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h);
}

// The continuous scalar operator(s) at (x, y, z), FD step h. {L0, L1}.
std::array<double, 2> continuous(const Mfg& M, double x, double y, double z,
                                 double h)
{
    const F3 rho = [&](double a, double b, double c) { return M.rho(a, b, c); };
    const F3 ph0 = [&](double a, double b, double c) { return M.phi0(a, b, c); };
    const F3 ph1 = [&](double a, double b, double c) { return M.phi1(a, b, c); };
    const F3 vel[3] = {
        [&](double a, double b, double c) { return M.u(a, b, c); },
        [&](double a, double b, double c) { return M.v(a, b, c); },
        [&](double a, double b, double c) { return M.w(a, b, c); }};
    double gu[3][3]; // gu[c][e] = d u_c / d x_e
    for (int c = 0; c < 3; ++c)
        for (int e = 0; e < 3; ++e) gu[c][e] = dd(vel[c], x, y, z, h, e);
    const double r = M.rho(x, y, z);
    // div of a flux F(x) whose component a is f(a, point).
    auto div = [&](auto f) {
        double s = 0;
        for (int a = 0; a < 3; ++a)
            s += dd([&](double p, double q, double t) { return f(a, p, q, t); },
                    x, y, z, h, a);
        return s;
    };
    auto grad = [&](const F3& f, double p, double q, double t, int a) {
        return dd(f, p, q, t, h, a);
    };
    if (M.model == RansModel::SaNeg) {
        const double sig = sa::sigma;
        const double conv = div([&](int a, double p, double q, double t) {
            return M.rho(p, q, t) * vel[a](p, q, t) * M.phi0(p, q, t);
        });
        const double diff = div([&](int a, double p, double q, double t) {
            return (M.mu + M.rho(p, q, t) * M.phi0(p, q, t)) / sig *
                   grad(ph0, p, q, t, a);
        });
        const double nt = M.phi0(x, y, z), nu = M.mu / r;
        double gg = 0, rg = 0;
        for (int a = 0; a < 3; ++a) {
            const double gn = grad(ph0, x, y, z, a);
            gg += gn * gn;
            rg += grad(rho, x, y, z, a) * gn;
        }
        const double w0 = gu[2][1] - gu[1][2], w1 = gu[0][2] - gu[2][0],
                     w2 = gu[1][0] - gu[0][1];
        const double Om = std::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
        const double src = r * saSource(nt, nu, Om, M.d(x, y, z)).R;
        return {-conv + diff + src + sa::cb2 / sig * r * gg -
                    saDiffCoef(nu, nt) / sig * rg,
                0.0};
    }
    // SST, F1 = F2 = 1: mu_t = rho k / omega, the inner (k-omega) constants.
    double S2 = 0;
    for (int c = 0; c < 3; ++c)
        for (int e = 0; e < 3; ++e) {
            const double sce = 0.5 * (gu[c][e] + gu[e][c]);
            S2 += 2 * sce * sce;
        }
    const auto mut = [&](double p, double q, double t) {
        return M.rho(p, q, t) * M.phi0(p, q, t) / M.phi1(p, q, t);
    };
    std::array<double, 2> L{};
    for (int e = 0; e < 2; ++e) {
        const F3& ph = e == 0 ? ph0 : ph1;
        const double sg = e == 0 ? sst::sigmaK1 : sst::sigmaW1;
        const double conv = div([&](int a, double p, double q, double t) {
            return M.rho(p, q, t) * vel[a](p, q, t) * ph(p, q, t);
        });
        const double diff = div([&](int a, double p, double q, double t) {
            return (M.mu + sg * mut(p, q, t)) * grad(ph, p, q, t, a);
        });
        L[e] = -conv + diff;
    }
    const double k = M.phi0(x, y, z), w = M.phi1(x, y, z);
    L[0] += mut(x, y, z) * S2 - sst::betaStar * r * w * k;
    L[1] += sstGamma1(M.model) * r * S2 - sst::beta1 * r * w * w;
    return L;
}

struct Level
{
    double err[2] = {0, 0}, fdErr[2] = {0, 0}, h = 0;
};

// R_h - L on one level: the annulus (nx angular, ny radial incl. 2 ghost
// rows) or the sphere's 5-degree wedge (nx meridional + 2 ghost columns).
std::expected<Level, std::string> level(const Mfg& M, int nx, int ny)
{
    const RevolvedSpec rv{.wedge = true, .dphi = 5.0 * 3.141592653589793 / 180};
    const Grid g = M.revolved ? ogridRevolvedGrid(nx, ny, rv, kR, kRFar)
                              : ogridGrid(nx, ny, 1, kR, kRFar, 1.0);
    const Sdf body = [&](double x, double y, double z) {
        return (M.revolved ? std::sqrt(x * x + y * y + z * z) : std::hypot(x, y)) -
               kR;
    };
    auto og = M.revolved
                  ? buildRevolvedOGrid(g, body, {.rFar = kRFar, .beta = kBeta}, rv)
                  : buildOGrid(g, body,
                               {.rFar = kRFar, .beta = kBeta, .span = 1.0});
    if (!og) return std::unexpected(og.error());
    auto m = CurvMetrics::build(og->mesh);
    if (!m) return std::unexpected(m.error());
    // Centroids: the node mean (CurvMetrics' and the wall distance's).
    std::vector<std::array<double, 3>> cc(g.size());
    const int nk = g.nz == 1 ? 1 : 2;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                std::array<double, 3> s{0, 0, 0};
                for (int dk = 0; dk < nk; ++dk)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            const auto q = og->mesh.node(i + di, j + dj, k + dk);
                            for (int a = 0; a < 3; ++a) s[a] += q[a] / (4.0 * nk);
                        }
                cc[g.idx(i, j, k)] = s;
            }
    auto h = CompState::zeros(g);
    std::vector<double> t0(g.size()), t1(g.size()), d(g.size());
    for (std::size_t c = 0; c < g.size(); ++c) {
        const auto [x, y, z] = cc[c];
        const double r = M.rho(x, y, z), u = M.u(x, y, z), v = M.v(x, y, z),
                     w = M.w(x, y, z);
        h.rho[c] = r;
        h.mx[c] = r * u;
        h.my[c] = r * v;
        h.mz[c] = r * w;
        h.E[c] = M.p(x, y, z) / (kGam - 1) + 0.5 * r * (u * u + v * v + w * w);
        t0[c] = r * M.phi0(x, y, z);
        t1[c] = r * M.phi1(x, y, z);
        d[c] = std::max(M.d(x, y, z), 1e-3);
    }
    auto dev = gpu::DeviceComp::make(*m, kGam, M.mu);
    auto ds = gpu::DeviceCompState::make(g.size(), ransScalars(M.model));
    auto out = gpu::DeviceCompState::make(g.size(), ransScalars(M.model));
    if (!dev || !ds || !out) return std::unexpected("GPU alloc");
    std::vector<std::uint8_t> mask(g.size(), 0);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 1; j <= g.ny - 2; ++j)
            for (int i = 0; i < g.nx; ++i)
                mask[g.idx(i, j, k)] =
                    !M.revolved || (k == 1 && i >= 1 && i <= g.nx - 2);
    if (!dev->enableRans({.model = M.model,
                          .wallDist = d,
                          .mutHook = {},
                          .fluidMask = mask}))
        return std::unexpected("enableRans");
    if (ds->upload(h) != hipSuccess || ds->uploadTurb(0, t0) != hipSuccess ||
        (ds->nT > 1 && ds->uploadTurb(1, t1) != hipSuccess))
        return std::unexpected("upload");
    // The wedge: the axis ghosts and the ghost planes by the production hook
    // (the continuation), every other cell exact.
    if (M.revolved) {
        auto ax = gpu::DeviceAxisBc::make(g, rv);
        if (!ax) return std::unexpected("axis bc");
        ax->apply(*ds);
    }
    dev->residualInto(*ds, *out);
    Level L;
    L.h = 1.0 / std::sqrt(double(g.nx - (M.revolved ? 2 : 0)) * (g.ny - 2));
    double num[2] = {0, 0}, den[2] = {0, 0}, fd[2] = {0, 0};
    std::vector<float> R[2];
    for (int e = 0; e < ds->nT; ++e) {
        R[e].resize(g.size());
        if (out->t[e].download(R[e].data()) != hipSuccess)
            return std::unexpected("download");
    }
    const double hfd = 2e-3;
    for (std::size_t c = 0; c < g.size(); ++c) {
        if (!mask[c]) continue;
        const auto [x, y, z] = cc[c];
        const auto a = continuous(M, x, y, z, hfd);
        const auto b = continuous(M, x, y, z, 0.5 * hfd);
        for (int e = 0; e < ds->nT; ++e) {
            const double dv = double(R[e][c]) - a[e];
            num[e] += m->vol[c] * dv * dv;
            den[e] += m->vol[c] * a[e] * a[e];
            fd[e] += m->vol[c] * (a[e] - b[e]) * (a[e] - b[e]);
        }
    }
    for (int e = 0; e < ds->nT; ++e) {
        L.err[e] = std::sqrt(num[e] / den[e]);
        L.fdErr[e] = std::sqrt(fd[e] / den[e]);
    }
    return L;
}

void ladder(const std::string& name, const Mfg& M, double p)
{
    const int lv[3][2] = {{32, 24}, {64, 48}, {128, 96}};
    Level L[3];
    for (int l = 0; l < 3; ++l) {
        auto r = level(M, lv[l][0], lv[l][1]);
        if (!r) { check(false, name + ": " + r.error()); return; }
        L[l] = *r;
    }
    for (int e = 0; e < ransScalars(M.model); ++e) {
        const std::string q =
            name + (M.model == RansModel::SaNeg ? " nu~" : e ? " omega" : " k");
        const double p01 = std::log(L[0].err[e] / L[1].err[e]) /
                           std::log(L[0].h / L[1].h);
        const double p12 = std::log(L[1].err[e] / L[2].err[e]) /
                           std::log(L[1].h / L[2].h);
        std::printf("%s: rel L2 %.3e %.3e %.3e  order %.2f %.2f  (FD error "
                    "%.1e of the finest)\n",
                    q.c_str(), L[0].err[e], L[1].err[e], L[2].err[e], p01, p12,
                    L[2].fdErr[e] / L[2].err[e]);
        check(L[2].fdErr[e] <= 1e-3 * L[2].err[e],
              q + ": the FP64 reference's own error <= 1e-3 of the finest");
        check(L[0].err[e] > L[1].err[e] && L[1].err[e] > L[2].err[e],
              q + ": monotone convergence");
        check(p12 >= p - 0.25 && p12 <= p + 0.5,
              std::format("{}: finest-pair order {:.2f} in [{:.2f}, {:.2f}]", q,
                          p12, p - 0.25, p + 0.5));
    }
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    ladder("SA u = 0", {.model = RansModel::SaNeg, .flow = false}, 2.0);
    ladder("SA full", {.model = RansModel::SaNeg, .flow = true}, 1.0);
    ladder("SA convection", {.model = RansModel::SaNeg, .flow = true,
                             .mu = 1e-6, .phiScale = 100.0, .dConst = 1e6},
           1.0);
    ladder("SST u = 0", {.model = RansModel::Sst2003m, .flow = false,
                         .dConst = 1e-3},
           2.0);
    ladder("SST full", {.model = RansModel::Sst2003m, .flow = true,
                        .dConst = 1e-3},
           1.0);
    // The axisymmetric wedge (P2b): the same terms through the hoop
    // geometry, the degenerate axis faces and the continuation ghosts.
    ladder("wedge SA u = 0", {.model = RansModel::SaNeg, .flow = false,
                              .revolved = true},
           2.0);
    ladder("wedge SA full", {.model = RansModel::SaNeg, .flow = true,
                             .revolved = true},
           1.0);
    ladder("wedge SST u = 0", {.model = RansModel::Sst2003m, .flow = false,
                               .dConst = 1e-3, .revolved = true},
           2.0);
    ladder("wedge SST full", {.model = RansModel::Sst2003m, .flow = true,
                              .dConst = 1e-3, .revolved = true},
           1.0);
    return dnstest::finish();
}
