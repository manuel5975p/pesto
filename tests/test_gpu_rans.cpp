// GPU RANS (SA-neg) on the compressible curvilinear solver, against the FP64
// oracle and against the RANS-off solver itself:
//   G1  an lts buffer of 1.0f is the uniform step bit for bit
//   G2  RANS on with nu~ == 0 is RANS off bit for bit; nu~ stays exactly 0
//   G3  a uniform specific scalar stays uniform under transport alone
//       (the duplicated mass flux equals faceFluxCurv's), and is conserved
//   G4  the wall face carries mu_t == 0 exactly; the ghost scalar algebra
//   G5  the residual at the oracle's discrete fixed point is at the FP32
//       floor; a perturbed state is far above it
//   G6  the oracle state is a fixed point of the LTS march
//   G7  ... with RANDOM lts ratios (an additive hook ignoring lts drifts O(1))
//   G8  SA-neg stress: a negative nu~ patch stays finite with mu_t == 0
//   G9  --steady from the perturbed state converges onto the oracle
// docs/rans.md. Seconds.
#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_channel.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/rans_channel1d.hpp"
#include "dns/walldist.hpp"
#include "check.hpp"
#include "curv_test_util.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

constexpr double kGamma = 1.4;

// One channel rig: mesh, solver, wall, body-force hook -- the runner's wiring,
// without the runner. rans == None leaves RANS off.
struct Rig
{
    Grid g{1, 1, 1, 1, 1, 1};
    gpu::GpuGrid gg{};
    std::unique_ptr<CurvMetrics> m;
    std::unique_ptr<gpu::DeviceComp> dev;
    std::unique_ptr<gpu::DeviceCompState> ds;
    std::unique_ptr<gpu::DeviceChannelWall> wall;
    std::unique_ptr<gpu::ReduceScratch> red;
    std::vector<double> d;
    double mu = 0, Twall = 0;
};

std::unique_ptr<Rig> makeRig(const Channel1dCase& cc, int nx, int nz,
                             RansModel model)
{
    auto r = std::make_unique<Rig>();
    r->g = channel1dGrid(cc, nx, nz, 4.0, 1.0);
    r->gg = gpu::GpuGrid::from(r->g);
    const auto mesh = CurvMesh::channelY(r->g, cc.beta, 1.0);
    auto m = CurvMetrics::build(mesh);
    if (!m) return nullptr;
    r->m = std::make_unique<CurvMetrics>(std::move(*m));
    r->mu = 1.0 / cc.reTau;
    r->Twall = channelPRef(cc.reTau, cc.mach);
    r->d = channelWallDistance(mesh, 1.0);
    const double d1 = r->d[r->g.idx(0, 1, 0)];
    auto dev = gpu::DeviceComp::make(*r->m, kGamma, r->mu);
    auto ds = gpu::DeviceCompState::make(r->g.size(), ransScalars(model));
    auto wall = gpu::DeviceChannelWall::make(
        r->g, {kGamma, 1.0, r->Twall, true,
               ransSst(model) ? sst::omegaWallFactor * r->mu / (d1 * d1) : 0.0});
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !wall || !red) return nullptr;
    r->dev = std::make_unique<gpu::DeviceComp>(std::move(*dev));
    r->ds = std::make_unique<gpu::DeviceCompState>(std::move(*ds));
    r->wall = std::make_unique<gpu::DeviceChannelWall>(std::move(*wall));
    r->red = std::make_unique<gpu::ReduceScratch>(std::move(*red));
    if (model != RansModel::None &&
        !r->dev->enableRans({.model = model, .wallDist = r->d,
                             .mirrorYLo = true, .mirrorYHi = true}))
        return nullptr;
    Rig* p = r.get();
    p->dev->setStageHook([p](gpu::DeviceCompState& s) { p->wall->apply(s); });
    p->dev->setRhsHook([p](const gpu::DeviceCompState&, gpu::DeviceCompState& o,
                           const gpu::StageCombine& cmb) {
        gpu::addBodyForceX(p->gg, 1.0, cmb, p->dev->u(), o);
    });
    return r;
}

// The march the runner does: step, wall.
void march(Rig& r, int n, double dt)
{
    for (int i = 0; i < n; ++i) {
        r.dev->step(*r.ds, gpu::Real(dt));
        r.wall->apply(*r.ds);
    }
}

bool sameBits(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (float(a[i]) != float(b[i]) || std::signbit(a[i]) != std::signbit(b[i]))
            return false;
    return true;
}

bool sameState(const CompState& a, const CompState& b)
{
    return sameBits(a.rho, b.rho) && sameBits(a.mx, b.mx) && sameBits(a.my, b.my) &&
           sameBits(a.mz, b.mz) && sameBits(a.E, b.E);
}

// A deterministic, non-trivial laminar-channel state with all three velocity
// components (so every flux and stencil term is live), ghost rows by the wall.
CompState laminarState(const Grid& g, double Twall)
{
    auto s = CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double y = (j + 0.5) / g.ny, x = double(i) / g.nx,
                             z = double(k) / g.nz;
                const double rho = 1.0 + 0.01 * std::sin(6.28 * x + 1.0);
                const double u = 4 * y * (1 - y) + 0.05 * std::sin(6.28 * z);
                const double v = 0.02 * std::sin(6.28 * x) * y * (1 - y);
                const double w = 0.03 * std::cos(6.28 * x + 6.28 * z);
                const std::size_t c = g.idx(i, j, k);
                s.rho[c] = rho;
                s.mx[c] = rho * u;
                s.my[c] = rho * v;
                s.mz[c] = rho * w;
                s.E[c] = rho * Twall / (kGamma - 1) +
                         0.5 * rho * (u * u + v * v + w * w);
            }
    return s;
}

double stableOf(Rig& r) { return 0.9 * r.dev->stableDt(*r.ds, *r.red, 0.4); }

// --- G1 -----------------------------------------------------------------------
void ltsNull()
{
    Channel1dCase cc;
    cc.ny = 32;
    cc.reTau = 2.0;
    auto a = makeRig(cc, 8, 4, RansModel::None);
    auto b = makeRig(cc, 8, 4, RansModel::None);
    if (!a || !b) { check(false, "G1 rig"); return; }
    const auto s0 = laminarState(a->g, a->Twall);
    (void)a->ds->upload(s0);
    (void)b->ds->upload(s0);
    a->wall->apply(*a->ds);
    b->wall->apply(*b->ds);
    (void)b->dev->uploadLts(std::vector<gpu::Real>(b->g.size(), 1.0f));
    b->dev->setLts(true);
    const double dt = stableOf(*a);
    march(*a, 50, dt);
    march(*b, 50, dt);
    auto ha = CompState::zeros(a->g), hb = CompState::zeros(b->g);
    (void)a->ds->download(ha);
    (void)b->ds->download(hb);
    check(sameState(ha, hb), "G1: lts == 1.0f is the uniform step bit for bit "
                             "(50 laminar steps)");
}

// --- G2 -----------------------------------------------------------------------
void laminarLimit()
{
    Channel1dCase cc;
    cc.ny = 32;
    cc.reTau = 2.0;
    auto off = makeRig(cc, 8, 4, RansModel::None);
    auto on = makeRig(cc, 8, 4, RansModel::SaNeg);
    if (!off || !on) { check(false, "G2 rig"); return; }
    const auto s0 = laminarState(off->g, off->Twall);
    (void)off->ds->upload(s0);
    (void)on->ds->upload(s0);
    (void)on->ds->uploadTurb(0, std::vector<double>(on->g.size(), 0.0));
    off->wall->apply(*off->ds);
    on->wall->apply(*on->ds);
    const double dt = stableOf(*off); // the SAME fixed dt for both
    march(*off, 200, dt);
    march(*on, 200, dt);
    auto ha = CompState::zeros(off->g), hb = CompState::zeros(on->g);
    (void)off->ds->download(ha);
    (void)on->ds->download(hb);
    std::vector<double> t0;
    (void)on->ds->downloadTurb(0, t0);
    const bool zero = t0.size() == on->g.size() &&
                      std::ranges::all_of(t0, [](double v) { return v == 0.0; });
    check(sameState(ha, hb), "G2: RANS on with nu~ == 0 is RANS off bit for bit "
                             "(200 steps, 5 conserved arrays)");
    check(zero, "G2: nu~ stays exactly 0");
}

// --- G3 -----------------------------------------------------------------------
void uniformScalar()
{
    const int n = 48;
    const auto g = Grid::make(n, n, 1, 10.0, 10.0, 1.0);
    const auto mesh = CurvMesh::sinusoidal(g, 0.04);
    auto m = CurvMetrics::build(mesh);
    if (!m) { check(false, "G3 metrics"); return; }
    auto dev = gpu::DeviceComp::make(*m, kGamma, 1e-3);
    auto ds = gpu::DeviceCompState::make(g.size(), 1);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !red ||
        !dev->enableRans({.model = RansModel::SaNeg,
                          .wallDist = std::vector<double>(g.size(), 1.0),
                          .transportOnly = true})) {
        check(false, "G3 alloc");
        return;
    }
    const auto s0 = curvtest::vortexState(mesh, 5.0, kGamma);
    // nu~ = 2^-13 (nu~/nu ~ 0.12: mu_t ~ fv1 is negligible). A POWER OF TWO:
    // rho nu~ is then rho scaled exactly, the upwind flux is the mass flux
    // scaled exactly, and the J == 0 combination is the mean flow's own
    // expression -- so if massFlux matches faceFluxCurv bit for bit, nu~
    // stays EXACTLY uniform. Any drift is the duplicated expression.
    const double c = 0x1p-13;
    std::vector<double> t(g.size());
    for (std::size_t q = 0; q < g.size(); ++q)
        t[q] = double(float(s0.rho[q])) * c;
    (void)ds->upload(s0);
    (void)ds->uploadTurb(0, t);
    auto mass = [&](const std::vector<double>& v) {
        double s = 0;
        for (std::size_t q = 0; q < v.size(); ++q) s += v[q] * m->vol[q];
        return s;
    };
    std::vector<double> t1;
    auto h = CompState::zeros(g);
    (void)ds->downloadTurb(0, t1);
    (void)ds->download(h);
    const double M0 = mass(t1), R0 = mass(h.rho);
    const double dt = 0.9 * dev->stableDt(*ds, *red, 0.4);
    for (int i = 0; i < 100; ++i) dev->step(*ds, gpu::Real(dt));
    (void)ds->download(h);
    (void)ds->downloadTurb(0, t1);
    double worst = 0;
    for (std::size_t q = 0; q < g.size(); ++q) {
        const float nt = float(t1[q]) / float(h.rho[q]);
        worst = std::max(worst, std::abs(double(nt) - c) / (c * 0x1p-23));
    }
    const double dM = std::abs(mass(t1) - M0) / M0;
    const double dR = std::abs(mass(h.rho) - R0) / R0;
    std::printf("G3: uniform nu~ after 100 steps: worst %.2f ulp; sum rho nu~ "
                "vol drift %.2e (the mean flow's own sum rho vol: %.2e)\n",
                worst, dM, dR);
    check(worst <= 4.0, "G3: uniform specific scalar stays uniform (<= 4 ulp)");
    // The scalar is conserved exactly as well as the density is -- bit for
    // bit -- and, with the RK3 weights summing to exactly 1, both to round-off.
    check(dM == dR, "G3: sum rho nu~ vol drifts exactly as sum rho vol");
    check(dM < 1e-6, "G3: sum rho nu~ vol conserved to 1e-6 over 100 steps");
}

// --- the oracle state on the device -------------------------------------------
struct OracleRig
{
    std::unique_ptr<Rig> rig;
    Channel1dSolution sol;
};

std::unique_ptr<OracleRig> oracleRig(RansModel model = RansModel::SaNeg,
                                     int ny = 64, double reTau = 180)
{
    Channel1dCase cc;
    cc.ny = ny;
    cc.reTau = reTau;
    cc.model = model;
    auto s = solveChannel1d(cc);
    if (!s) {
        std::printf("oracle: %s\n", s.error().c_str());
        return nullptr;
    }
    auto o = std::make_unique<OracleRig>();
    o->sol = std::move(*s);
    o->rig = makeRig(cc, 4, 1, model);
    if (!o->rig) return nullptr;
    return o;
}

// One conserved scalar (SA) or the SST pair (t2 non-empty).
void upload(Rig& r, const CompState& h, const std::vector<double>& t,
            const std::vector<double>& t2 = {})
{
    (void)r.ds->upload(h);
    (void)r.ds->uploadTurb(0, t);
    if (!t2.empty()) (void)r.ds->uploadTurb(1, t2);
    r.wall->apply(*r.ds);
}

// The runner's normalisation (docs/rans.md): SST's scalar part is the larger
// of the k and ln(omega) rates.
double resid(Rig& r, double ubEst, gpu::ResidualNorms* out = nullptr)
{
    const auto n = r.dev->residual(*r.ds, 1, r.g.ny - 2, *r.red);
    if (out) *out = n;
    return std::max({n.mom, n.E / ubEst, n.t[0], n.t[1]});
}

// FP32 floor of the residual: the spread over equally valid float roundings
// of the SAME double state (each field nudged by a random +-1 ulp). A residual
// at the oracle state no larger than a few of these is at the representation
// floor, not at a discretization mismatch.
double fp32Floor(Rig& r, const CompState& h, const std::vector<double>& t,
                 double ubEst, const std::vector<double>& t2 = {})
{
    std::mt19937 rng(5);
    double worst = 0;
    for (int trial = 0; trial < 4; ++trial) {
        auto nudge = [&](std::vector<double> v) {
            for (double& x : v) {
                const float f = float(x);
                x = rng() & 1 ? std::nextafter(f, 1e30f) : std::nextafter(f, -1e30f);
            }
            return v;
        };
        CompState p = h;
        p.rho = nudge(p.rho);
        p.mx = nudge(p.mx);
        p.E = nudge(p.E);
        upload(r, p, nudge(t), t2.empty() ? t2 : nudge(t2));
        worst = std::max(worst, resid(r, ubEst));
    }
    return worst;
}

double g5Floor = 0; // measured in G5, reused by G9's tolerance

// --- G5 -----------------------------------------------------------------------
void residualAtOracle()
{
    auto o = oracleRig();
    if (!o) { check(false, "G5 oracle rig"); return; }
    Rig& r = *o->rig;
    const auto h = toCompState(o->sol, r.g);
    const auto t = toTurb(o->sol, r.g);
    gpu::ResidualNorms n0{};
    upload(r, h, t);
    const double r0 = resid(r, o->sol.ubEst, &n0);
    const double floor = fp32Floor(r, h, t, o->sol.ubEst);
    g5Floor = floor;
    // Perturbed: u x 0.95, nu~ x 0.7 (T unchanged).
    auto hp = h;
    auto tp = t;
    for (std::size_t q = 0; q < hp.rho.size(); ++q) {
        const double u = h.mx[q] / h.rho[q];
        const double e = h.E[q] - 0.5 * h.rho[q] * u * u;
        hp.mx[q] = h.rho[q] * 0.95 * u;
        hp.E[q] = e + 0.5 * h.rho[q] * 0.95 * u * 0.95 * u;
        tp[q] = 0.7 * t[q];
    }
    gpu::ResidualNorms np{};
    upload(r, hp, tp);
    const double rp = resid(r, o->sol.ubEst, &np);
    std::printf("G5: resid at oracle %.3e (U %.2e E %.2e T %.2e), FP32 floor "
                "%.3e, perturbed %.3e (U %.2e E %.2e T %.2e)\n",
                r0, n0.mom, n0.E / o->sol.ubEst, n0.t[0], floor, rp, np.mom,
                np.E / o->sol.ubEst, np.t[0]);
    check(r0 <= 3.0 * floor, "G5: residual at the oracle state is at the FP32 "
                             "floor (<= 3x)");
    check(rp >= 100.0 * r0, "G5: perturbed state >= 100x the oracle residual");
}

// Drift of u (Linf / ub) and of the first scalar (nu~ or k; Linf / max) from
// the oracle profile.
std::pair<double, double> drift(Rig& r, const Channel1dSolution& s)
{
    auto h = CompState::zeros(r.g);
    std::vector<double> t;
    (void)r.ds->download(h);
    (void)r.ds->downloadTurb(0, t);
    const std::vector<double>& ref = ransSst(s.model) ? s.k : s.nut;
    double du = 0, dn = 0, nmax = 0;
    for (int j = 1; j <= r.g.ny - 2; ++j)
        for (int i = 0; i < r.g.nx; ++i) {
            const std::size_t q = r.g.idx(i, j, 0);
            du = std::max(du, std::abs(h.mx[q] / h.rho[q] - s.u[j - 1]));
            dn = std::max(dn, std::abs(t[q] / h.rho[q] - ref[j - 1]));
            nmax = std::max(nmax, ref[j - 1]);
        }
    return {du / s.ub, dn / nmax};
}

// --- G6 / G7 ------------------------------------------------------------------
void fixedPoint(bool randomLts)
{
    auto o = oracleRig();
    if (!o) { check(false, "G6/7 oracle rig"); return; }
    Rig& r = *o->rig;
    upload(r, toCompState(o->sol, r.g), toTurb(o->sol, r.g));
    const int steps = randomLts ? 500 : 2000;
    double dt = 0;
    if (randomLts) {
        // Ratios in [1, 10] on a reference a tenth of the smallest local dt,
        // so every cell stays inside its own stable step.
        const double dref = r.dev->localDt(*r.ds, *r.red, 0.4);
        std::mt19937 rng(9);
        std::uniform_real_distribution<float> U(1.0f, 10.0f);
        std::vector<gpu::Real> l(r.g.size());
        for (auto& x : l) x = U(rng);
        (void)r.dev->uploadLts(l);
        dt = 0.095 * dref;
    } else {
        dt = 0.95 * r.dev->localDt(*r.ds, *r.red, 0.4);
    }
    r.dev->setLts(true);
    for (int i = 0; i < steps; ++i) {
        if (!randomLts && i % 16 == 0)
            dt = 0.95 * r.dev->localDt(*r.ds, *r.red, 0.4);
        r.dev->step(*r.ds, gpu::Real(dt));
        r.wall->apply(*r.ds);
    }
    const auto [du, dn] = drift(r, o->sol);
    std::printf("%s: %d LTS steps from the oracle: u drift %.2e (Linf/ub), "
                "nu~ drift %.2e (Linf/max)\n",
                randomLts ? "G7" : "G6", steps, du, dn);
    // Plan bounds 1e-4 / 1e-3, nu~ tightened after measuring (G6 1.1e-4,
    // G7 7.9e-5; u 3.5e-5 / 2.7e-5).
    check(du < 1e-4 && dn < 5e-4,
          randomLts ? "G7: random lts keeps the oracle fixed point"
                    : "G6: the oracle state is a fixed point of the LTS march");
}

// --- G4 -----------------------------------------------------------------------
void wallFace()
{
    auto o = oracleRig();
    if (!o) { check(false, "G4 oracle rig"); return; }
    Rig& r = *o->rig;
    upload(r, toCompState(o->sol, r.g), toTurb(o->sol, r.g));
    r.dev->updateMut(*r.ds);
    std::vector<gpu::Real> mut(r.g.size());
    (void)r.dev->mut().download(mut.data());
    auto h = CompState::zeros(r.g);
    std::vector<double> t;
    (void)r.ds->download(h);
    (void)r.ds->downloadTurb(0, t);
    bool face = true, ghost = true, pos = false;
    for (int i = 0; i < r.g.nx; ++i)
        for (auto [gj, sj] : {std::pair{0, 1}, {r.g.ny - 1, r.g.ny - 2}}) {
            const std::size_t gq = r.g.idx(i, gj, 0), sq = r.g.idx(i, sj, 0);
            face = face && gpu::Real(0.5) * (mut[gq] + mut[sq]) == 0.0f;
            pos = pos || mut[sq] > 0.0f;
            const float want = -float(h.rho[gq]) * float(t[sq]) / float(h.rho[sq]);
            ghost = ghost && std::abs(float(t[gq]) - want) <=
                                 std::abs(want) * 0x1p-23f;
        }
    check(pos, "G4: mu_t > 0 in the first fluid row (the test is not vacuous)");
    check(face, "G4: 0.5 (mu_t,ghost + mu_t,1) == 0.0f on both plates");
    check(ghost, "G4: ghost rho nu~ = -rho_g nu~_1 (1 ulp)");
}

// --- G8 -----------------------------------------------------------------------
void negativePatch()
{
    auto o = oracleRig();
    if (!o) { check(false, "G8 oracle rig"); return; }
    Rig& r = *o->rig;
    const auto h = toCompState(o->sol, r.g);
    auto t = toTurb(o->sol, r.g);
    const double nu = r.mu / 1.0;
    for (int j = 20; j <= 26; ++j)
        for (int i = 0; i < r.g.nx; ++i) {
            const std::size_t q = r.g.idx(i, j, 0);
            t[q] = -5.0 * nu * h.rho[q];
        }
    upload(r, h, t);
    r.dev->setLts(true);
    double dt = 0;
    for (int i = 0; i < 500; ++i) {
        if (i % 16 == 0) dt = 0.95 * r.dev->localDt(*r.ds, *r.red, 0.4);
        r.dev->step(*r.ds, gpu::Real(dt));
        r.wall->apply(*r.ds);
    }
    auto hs = CompState::zeros(r.g);
    std::vector<double> ts;
    (void)r.ds->download(hs);
    (void)r.ds->downloadTurb(0, ts);
    r.dev->updateMut(*r.ds);
    std::vector<gpu::Real> mut(r.g.size());
    (void)r.dev->mut().download(mut.data());
    bool finite = true, zero = true;
    int neg = 0;
    double minNt = 0;
    for (int j = 1; j <= r.g.ny - 2; ++j)
        for (int i = 0; i < r.g.nx; ++i) {
            const std::size_t q = r.g.idx(i, j, 0);
            finite = finite && std::isfinite(hs.rho[q]) && std::isfinite(hs.mx[q]) &&
                     std::isfinite(hs.E[q]) && std::isfinite(ts[q]) &&
                     std::isfinite(mut[q]);
            if (ts[q] < 0) {
                ++neg;
                zero = zero && mut[q] == 0.0f;
            }
            minNt = std::min(minNt, ts[q] / hs.rho[q] / nu);
        }
    std::printf("G8: after 500 LTS steps: %d cells still nu~ < 0 (min nu~/nu "
                "%.3f)\n", neg, minNt);
    check(finite, "G8: SA-neg patch stays finite (no NaN)");
    check(zero, "G8: mu_t == 0 on every nu~ < 0 cell");
}

// The steady LTS march for `n` iterations with the runner's dt cadence;
// returns the largest residual sampled (every 200) over the last `tail`.
double steadyMarch(Rig& r, long n, long tail, double ubEst)
{
    r.dev->setLts(true);
    double dt = 0, worst = 0;
    for (long it = 0; it < n; ++it) {
        if (it % 16 == 0) dt = 0.95 * r.dev->localDt(*r.ds, *r.red, 0.4);
        r.dev->step(*r.ds, gpu::Real(dt));
        r.wall->apply(*r.ds);
        if ((it + 1) % 200 == 0 && it >= n - tail)
            worst = std::max(worst, resid(r, ubEst));
    }
    return worst;
}

// --- G9 -----------------------------------------------------------------------
// MEASURED (2026-10-24, docs/rans.md "The FP32 update floor"): the march
// cannot take the residual to 10x the static floor of G5. Once |R_c| dt_c <
// ulp(q_c)/2 the FP32 state stops changing, so the march stalls on a DYNAMIC
// floor ulp(q)/(2 dt_c) -- resid ~1-2 here (E ~ p/(gamma-1) dominates at
// M 0.2; resE carries it), ~40x the static floor, and reached within ~1e3
// iterations from the oracle itself. The slow (ub) mode's residual is below
// that floor long before it has converged, so the residual cannot certify it
// (a cold start sits at the floor with ub 1.6% low). What IS checked: from
// the perturbed state the march lands on the oracle's discrete solution, and
// its residual sits on the same plateau as a march started AT the oracle.
void steadyConverges()
{
    auto o = oracleRig();
    if (!o) { check(false, "G9 oracle rig"); return; }
    Rig& r = *o->rig;
    const auto h = toCompState(o->sol, r.g);
    const auto t = toTurb(o->sol, r.g);
    upload(r, h, t);
    const double plateau = steadyMarch(r, 20000, 10000, o->sol.ubEst);
    auto hp = h;
    auto tp = t;
    for (std::size_t q = 0; q < hp.rho.size(); ++q) {
        const double u = h.mx[q] / h.rho[q];
        const double e = h.E[q] - 0.5 * h.rho[q] * u * u;
        hp.mx[q] = h.rho[q] * 0.95 * u;
        hp.E[q] = e + 0.5 * h.rho[q] * 0.95 * u * 0.95 * u;
        tp[q] = 0.7 * t[q];
    }
    upload(r, hp, tp);
    const auto t0 = std::chrono::steady_clock::now();
    // The ub error decays with an e-folding of ~6e4 iterations (measured:
    // -0.357% at 1.6e5, +0.048% at 4e5); 2.2e5 is ~30 s.
    const long n = 220000;
    const double res = steadyMarch(r, n, 10000, o->sol.ubEst);
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto hs = CompState::zeros(r.g);
    (void)r.ds->download(hs);
    double mom = 0, mass = 0, du = 0;
    for (int j = 1; j <= r.g.ny - 2; ++j) {
        const std::size_t q = r.g.idx(0, j, 0);
        const double dyj = r.m->vol[q] / r.m->S[1][1][q];
        mom += hs.mx[q] * dyj;
        mass += hs.rho[q] * dyj;
        du = std::max(du, std::abs(hs.mx[q] / hs.rho[q] - o->sol.u[j - 1]));
    }
    const double ub = mom / mass;
    std::printf("G9: %ld LTS iterations from the perturbed state, %.1f s: ub "
                "%.6f vs oracle %.6f (%+.3f%%), U+ Linf %.3f%%; resid %.3e on "
                "the plateau %.3e (static floor %.3e, 10x = the plan's "
                "unreachable TOL)\n",
                n, sec, ub, o->sol.ub, 100 * (ub - o->sol.ub) / o->sol.ub,
                100 * du / o->sol.ub, res, plateau, g5Floor);
    check(std::abs(ub - o->sol.ub) <= 3e-3 * o->sol.ub,
          "G9: ub within 0.3% of the oracle");
    check(du <= 1e-2 * o->sol.ub, "G9: U+ within 1% (Linf) of the oracle");
    check(res <= 2.0 * plateau,
          "G9: the residual sits on the oracle-start plateau (<= 2x)");
}

// --- SST (phase B) -------------------------------------------------------------

// Laminar limit with k == 0: mu_t = 0 exactly, so SST on is SST off bit for
// bit on the mean flow, and k stays exactly 0 (omega is free, here 10).
void sstLaminarLimit(RansModel m)
{
    Channel1dCase cc;
    cc.ny = 32;
    cc.reTau = 2.0;
    cc.model = m;
    auto off = makeRig(cc, 8, 4, RansModel::None);
    auto on = makeRig(cc, 8, 4, m);
    if (!off || !on) { check(false, "S-G2 rig"); return; }
    const auto s0 = laminarState(off->g, off->Twall);
    (void)off->ds->upload(s0);
    off->wall->apply(*off->ds);
    std::vector<double> w(on->g.size());
    for (std::size_t q = 0; q < w.size(); ++q) w[q] = 10.0 * s0.rho[q];
    upload(*on, s0, std::vector<double>(on->g.size(), 0.0), w);
    const double dt = stableOf(*off);
    march(*off, 200, dt);
    march(*on, 200, dt);
    auto ha = CompState::zeros(off->g), hb = CompState::zeros(on->g);
    (void)off->ds->download(ha);
    (void)on->ds->download(hb);
    std::vector<double> k;
    (void)on->ds->downloadTurb(0, k);
    const bool zero = k.size() == on->g.size() &&
                      std::ranges::all_of(k, [](double v) { return v == 0.0; });
    const std::string n(ransName(m));
    check(sameState(ha, hb), n + ": k == 0 is RANS off bit for bit (200 steps)");
    check(zero, n + ": k stays exactly 0");
    check(on->dev->floorHits() == 0, n + ": 0 positivity-guard hits");
}

void sstSuite(RansModel m, bool converge)
{
    const std::string n(ransName(m));
    auto o = oracleRig(m);
    if (!o) { check(false, n + " oracle rig"); return; }
    Rig& r = *o->rig;
    const auto h = toCompState(o->sol, r.g);
    const auto tk = toTurb(o->sol, r.g, 0), tw = toTurb(o->sol, r.g, 1);

    // S-G4: the wall face.
    upload(r, h, tk, tw);
    r.dev->updateMut(*r.ds);
    std::vector<gpu::Real> mut(r.g.size());
    (void)r.dev->mut().download(mut.data());
    auto hd = CompState::zeros(r.g);
    std::vector<double> dk, dw;
    (void)r.ds->download(hd);
    (void)r.ds->downloadTurb(0, dk);
    (void)r.ds->downloadTurb(1, dw);
    bool face = true, kg = true, wg = true;
    for (int i = 0; i < r.g.nx; ++i)
        for (auto [gj, sj] : {std::pair{0, 1}, {r.g.ny - 1, r.g.ny - 2}}) {
            const std::size_t gq = r.g.idx(i, gj, 0), sq = r.g.idx(i, sj, 0);
            face = face && gpu::Real(0.5) * (mut[gq] + mut[sq]) == 0.0f;
            const double rw = 0.5 * (hd.rho[gq] + hd.rho[sq]);
            const double wf = 0.5 * (dw[gq] / hd.rho[gq] + dw[sq] / hd.rho[sq]);
            wg = wg && std::abs(wf - o->sol.omegaCoef / rw) <=
                           1e-5 * o->sol.omegaCoef / rw;
            kg = kg && std::abs(dk[gq] / hd.rho[gq] + dk[sq] / hd.rho[sq]) <=
                           2e-7 * std::abs(dk[sq] / hd.rho[sq]);
        }
    check(face, n + ": mu_t == 0.0f on both wall faces");
    check(kg, n + ": ghost k = -k_1 (specific)");
    check(wg, n + ": the wall face holds omega_w = 60 nu_w/(beta1 d1^2)");

    // S-G5: the residual at the oracle.
    upload(r, h, tk, tw);
    gpu::ResidualNorms n0{};
    const double r0 = resid(r, o->sol.ubEst, &n0);
    const double floor = fp32Floor(r, h, tk, o->sol.ubEst, tw);
    auto hp = h;
    auto kp = tk, wp = tw;
    for (std::size_t q = 0; q < hp.rho.size(); ++q) {
        const double u = h.mx[q] / h.rho[q];
        const double e = h.E[q] - 0.5 * h.rho[q] * u * u;
        hp.mx[q] = h.rho[q] * 0.95 * u;
        hp.E[q] = e + 0.5 * h.rho[q] * 0.95 * u * 0.95 * u;
        kp[q] = 0.7 * tk[q];
        wp[q] = 1.3 * tw[q];
    }
    upload(r, hp, kp, wp);
    const double rp = resid(r, o->sol.ubEst);
    std::printf("%s G5: resid at oracle %.3e (U %.2e E %.2e k %.2e lnw %.2e), "
                "FP32 floor %.3e, perturbed %.3e\n",
                n.c_str(), r0, n0.mom, n0.E / o->sol.ubEst, n0.t[0], n0.t[1],
                floor, rp);
    check(r0 <= 3.0 * floor, n + ": residual at the oracle is at the FP32 "
                                 "floor (<= 3x)");
    check(rp >= 100.0 * r0, n + ": perturbed state >= 100x the oracle "
                                "residual");

    // S-G6: the oracle is a fixed point of the LTS march.
    upload(r, h, tk, tw);
    const double plateau = steadyMarch(r, 2000, 1000, o->sol.ubEst);
    const auto [du, dn] = drift(r, o->sol);
    std::printf("%s G6: 2000 LTS steps from the oracle: u drift %.2e, k drift "
                "%.2e, residual plateau %.3e, %llu guard hits\n",
                n.c_str(), du, dn, plateau, r.dev->floorHits());
    check(du < 1e-4 && dn < 1e-3,
          n + ": the oracle state is a fixed point of the LTS march");

    // Positivity stress: the oracle's k and omega scaled by a random factor
    // in [0.5, 2] per cell (a rough field), 200 LTS steps, ZERO guard hits.
    std::mt19937 rng(17);
    std::uniform_real_distribution<double> U(0.5, 2.0);
    auto kr = tk, wr = tw;
    for (std::size_t q = 0; q < kr.size(); ++q) {
        kr[q] *= U(rng);
        wr[q] *= U(rng);
    }
    upload(r, h, kr, wr);
    const unsigned long long h0 = r.dev->floorHits();
    (void)steadyMarch(r, 200, 0, o->sol.ubEst);
    auto hs = CompState::zeros(r.g);
    std::vector<double> ks, ws;
    (void)r.ds->download(hs);
    (void)r.ds->downloadTurb(0, ks);
    (void)r.ds->downloadTurb(1, ws);
    bool finite = true;
    for (std::size_t q = 0; q < ks.size(); ++q)
        finite = finite && std::isfinite(hs.mx[q]) && std::isfinite(hs.E[q]) &&
                 std::isfinite(ks[q]) && std::isfinite(ws[q]);
    std::printf("%s positivity stress: %llu guard hits in 200 steps\n",
                n.c_str(), r.dev->floorHits() - h0);
    check(finite, n + ": rough k/omega field stays finite");
    check(r.dev->floorHits() == h0, n + ": rough field, 0 positivity-guard hits");
    if (!converge) return;

    // S-G9: from the perturbed state onto the oracle's discrete solution.
    upload(r, hp, kp, wp);
    const auto t0 = std::chrono::steady_clock::now();
    const long iters = 220000;
    const double res = steadyMarch(r, iters, 10000, o->sol.ubEst);
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    (void)r.ds->download(hs);
    double mom = 0, mass = 0, dmax = 0;
    for (int j = 1; j <= r.g.ny - 2; ++j) {
        const std::size_t q = r.g.idx(0, j, 0);
        const double dyj = r.m->vol[q] / r.m->S[1][1][q];
        mom += hs.mx[q] * dyj;
        mass += hs.rho[q] * dyj;
        dmax = std::max(dmax, std::abs(hs.mx[q] / hs.rho[q] - o->sol.u[j - 1]));
    }
    const double ub = mom / mass;
    std::printf("%s G9: %ld LTS iterations, %.1f s: ub %.6f vs oracle %.6f "
                "(%+.3f%%), U+ Linf %.3f%%, resid %.3e, %llu guard hits\n",
                n.c_str(), iters, sec, ub, o->sol.ub,
                100 * (ub - o->sol.ub) / o->sol.ub, 100 * dmax / o->sol.ub, res,
                r.dev->floorHits());
    check(std::abs(ub - o->sol.ub) <= 3e-3 * o->sol.ub,
          n + ": ub within 0.3% of the oracle");
    check(dmax <= 1e-2 * o->sol.ub, n + ": U+ within 1% (Linf) of the oracle");
    check(r.dev->floorHits() == h0, n + ": 0 guard hits converging");
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    ltsNull();
    laminarLimit();
    uniformScalar();
    wallFace();
    residualAtOracle();
    fixedPoint(false);
    fixedPoint(true);
    negativePatch();
    steadyConverges();
    for (RansModel m : {RansModel::Sst2003m, RansModel::SstVm})
        sstLaminarLimit(m);
    sstSuite(RansModel::Sst2003m, true);
    sstSuite(RansModel::SstVm, false);
    return dnstest::finish();
}

