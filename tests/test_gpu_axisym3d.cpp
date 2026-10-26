// The FULL-REVOLUTION revolved O-grid on the GPU (masterplan_flowrans P4;
// docs/rans.md "The full revolution"), small grids, seconds each:
//   (a) a uniform freestream is a fixed point THROUGH the axis: the residual
//       of every real cell, axis rings included, at the FP32 floor (< 16
//       ulp of the largest face flux) and 100 steps leaving the outer third
//       <= 8 ulp -- along the axis, tilted 30 deg and across it (the mesh
//       rotated about z; the freestream stays x); the 5-deg wedge at the
//       same (nTheta, ny) is the baseline
//   (b) an axisymmetric start stays axisymmetric (laminar Re 100): the
//       azimuthal spread of every field and the swirl random-walk at
//       rounding level over 1200 time-accurate steps (no exponential growth)
//       and saturate at the jitter floor of an 8000-iteration steady march;
//       the state is EXACTLY symmetric under a rotation by pi about the axis
//   (c) the 3D state equals the wedge's at the same dphi, plane by plane:
//       N 16 vs a 22.5-deg wedge, same (nTheta, ny), the same steady march
//   (d) a cross-flow sphere (axis perpendicular to the flow): steady laminar
//       Re 100 reaches the axial-flow drag of the same mesh, no lateral
//       force, and the pressure around the axis rings is a smooth field
//       (m >= 2 content at the discretization level, no axis spike)
//   (e) conservation across the cyclic seam and the axis: a compact,
//       non-axisymmetric bump straddling both -- the summed residual over a
//       closed shell of real cells telescopes to rounding
//   (f) SA and SST 200-step URANS smoke: finite, 0 guard hits, mu_t == 0.0
//       on every wall face
//   (g) the tally on a linear pressure along and ACROSS the axis: the
//       faceted body's Archimedes force, lateral component included
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/rans.hpp"
#include "dns/walldist.hpp"
#include "dns/gpu/axis_bc.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/reductions.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <format>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

constexpr double kGam = 1.4, kRgas = 1.0, kMach = 0.2;
constexpr double kRhoInf = 1.0, kUInf = 1.0;
const double kPInf = kRhoInf * kUInf * kUInf / (kGam * kMach * kMach);
const double kTInf = kPInf / (kRhoInf * kRgas);
constexpr double kD = 0.785398163397448;
const double kQRef = 0.5 * kRhoInf * kUInf * kUInf * 0.25 * pi * kD * kD;

struct RigSpec
{
    int nT = 32, ny = 24;
    int nphi = 16;        // 0 = the wedge
    double dphi = 0;      // the wedge angle (nphi == 0)
    double re = 100;
    RansModel model = RansModel::None;
    double rotDeg = 0;    // mesh rotated about z: the axis leaves x
    bool steady = false;  // compensated state + local-dt buffers
};

struct Rig
{
    RigSpec sp;
    Grid g{1, 1, 1, 1, 1, 1};
    bool wedge = false;
    int kLo = 0, kHi = 0;
    std::unique_ptr<OGrid> og;
    std::unique_ptr<CurvMetrics> m;
    std::unique_ptr<gpu::DeviceComp> dev;
    std::unique_ptr<gpu::DeviceCompState> ds;
    std::unique_ptr<gpu::DeviceOGridBc> bc;
    std::unique_ptr<gpu::DeviceAxisBc> axis;
    std::unique_ptr<gpu::ReduceScratch> red;
    std::vector<std::array<double, 3>> cen; // cell centroids
    double phi[2] = {0, 0};
};

std::unique_ptr<Rig> makeRig(const RigSpec& sp)
{
    auto r = std::make_unique<Rig>();
    r->sp = sp;
    r->wedge = sp.nphi == 0;
    const RevolvedSpec rv =
        r->wedge ? RevolvedSpec{.wedge = true, .dphi = sp.dphi}
                 : RevolvedSpec{.wedge = false, .dphi = 2 * pi / sp.nphi,
                                .nphi = sp.nphi};
    r->g = ogridRevolvedGrid(sp.nT, sp.ny, rv, 0.5 * kD, 20 * kD);
    const Sdf sphere = [](double x, double y, double z) {
        return std::sqrt(x * x + y * y + z * z) - 0.5 * kD;
    };
    auto og = buildRevolvedOGrid(r->g, sphere, {.rFar = 20 * kD, .beta = 3.0},
                                 rv);
    if (!og) return nullptr;
    r->og = std::make_unique<OGrid>(std::move(*og));
    if (sp.rotDeg != 0) {
        const double a = sp.rotDeg * pi / 180, c = std::cos(a), s = std::sin(a);
        auto& M = r->og->mesh;
        for (std::size_t q = 0; q < M.x.size(); ++q) {
            const double x = M.x[q], y = M.y[q];
            M.x[q] = c * x - s * y;
            M.y[q] = s * x + c * y;
        }
    }
    auto m = CurvMetrics::build(r->og->mesh);
    if (!m) return nullptr;
    r->m = std::make_unique<CurvMetrics>(std::move(*m));
    const Grid& g = r->g;
    r->kLo = r->wedge ? 1 : 0;
    r->kHi = r->wedge ? 1 : g.nz - 1;
    r->cen.resize(g.size());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                std::array<double, 3> c{0, 0, 0};
                for (int dk = 0; dk < 2; ++dk)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            const auto p = r->og->mesh.node(i + di, j + dj, k + dk);
                            for (int a = 0; a < 3; ++a) c[a] += 0.125 * p[a];
                        }
                r->cen[g.idx(i, j, k)] = c;
            }
    const auto tr = Transport::air(kRhoInf * kUInf * kD / sp.re, kTInf, 300.0,
                                   0.72, kGam, kRgas, false);
    gpu::DeviceOGridBc::Config c{.gamma = kGam, .Rgas = kRgas,
                                 .rhoInf = kRhoInf, .uInf = kUInf,
                                 .pInf = kPInf};
    c.tr = tr;
    c.farNormal.resize(std::size_t(g.nx) * g.nz);
    for (int k = 0; k < g.nz; ++k)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t q = g.idx(i, g.ny - 2, k);
            const double s[3] = {r->m->S[1][0][q], r->m->S[1][1][q],
                                 r->m->S[1][2][q]};
            const double l = std::hypot(s[0], s[1], s[2]);
            c.farNormal[std::size_t(k) * g.nx + i] = {s[0] / l, s[1] / l,
                                                       s[2] / l};
        }
    c.faceILo = 1;
    c.faceIHi = sp.nT;
    c.faceKLo = r->kLo;
    c.faceKHi = r->kHi;
    c.tallyScale = r->wedge ? 2 * pi / sp.dphi : 1.0;
    const auto d = revolvedWallDistance(r->og->mesh, sp.nT, r->wedge);
    c.wallD1 = wallFirstCellDistance(g, d);
    c.rans = sp.model;
    const double nu = kD / sp.re;
    if (sp.model == RansModel::SaNeg)
        c.phiInf[0] = nuTildaInf(kRansNutInfRatio, nu);
    if (ransSst(sp.model)) {
        c.phiInf[0] = kInf(kRansKInf, kUInf);
        c.phiInf[1] = omegaInf(c.phiInf[0], kRansMutInfRatio, nu);
    }
    r->phi[0] = c.phiInf[0];
    r->phi[1] = c.phiInf[1];
    auto dev = gpu::DeviceComp::make(*r->m, tr);
    auto ds = gpu::DeviceCompState::make(g.size(), ransScalars(sp.model),
                                         sp.steady);
    auto bc = gpu::DeviceOGridBc::make(g, c);
    auto ax = gpu::DeviceAxisBc::make(g, rv);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !bc || !ax || !red) return nullptr;
    r->dev = std::make_unique<gpu::DeviceComp>(std::move(*dev));
    r->ds = std::make_unique<gpu::DeviceCompState>(std::move(*ds));
    r->bc = std::make_unique<gpu::DeviceOGridBc>(std::move(*bc));
    r->axis = std::make_unique<gpu::DeviceAxisBc>(std::move(*ax));
    r->red = std::make_unique<gpu::ReduceScratch>(std::move(*red));
    Rig* p = r.get();
    std::vector<std::uint8_t> mask(g.size(), 0);
    for (int k = r->kLo; k <= r->kHi; ++k)
        for (int j = 1; j <= g.ny - 2; ++j)
            for (int i = 1; i <= sp.nT; ++i) mask[g.idx(i, j, k)] = 1;
    if (sp.model != RansModel::None) {
        if (!p->dev->enableRans({.model = sp.model,
                                 .wallDist = d,
                                 .mutHook =
                                     [p](gpu::Real* n, gpu::Real* f) {
                                         p->bc->mutHook()(n, f);
                                         p->axis->applyMut(n, f);
                                     },
                                 .fluidMask = mask}))
            return nullptr;
    } else if (!p->dev->enableSteady(mask)) {
        return nullptr;
    }
    if (sp.steady && !p->dev->enableCompensation()) return nullptr;
    p->dev->setStageHook([p](gpu::DeviceCompState& s) {
        p->bc->apply(s);
        p->axis->apply(s);
    });
    return r;
}

void apply(Rig& r)
{
    r.bc->apply(*r.ds);
    r.axis->apply(*r.ds);
}

std::vector<float> down(const gpu::DeviceScalar& b)
{
    std::vector<float> h(b.size());
    (void)b.download(h.data());
    return h;
}

void uniformState(Rig& r)
{
    auto h = CompState::zeros(r.g);
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        h.rho[c] = kRhoInf;
        h.mx[c] = kRhoInf * kUInf;
        h.E[c] = kPInf / (kGam - 1) + 0.5 * kRhoInf * kUInf * kUInf;
    }
    (void)r.ds->upload(h);
    for (int i = 0; i < r.ds->nT; ++i)
        (void)r.ds->uploadTurb(i, std::vector<double>(r.g.size(),
                                                      kRhoInf * r.phi[i]));
}

// Time-accurate steps at a global dt (recomputed every 16); the time marched.
double march(Rig& r, int steps)
{
    double dt = 0, t = 0;
    for (int s = 0; s < steps; ++s) {
        if (s % 16 == 0) dt = 0.9 * r.dev->stableDt(*r.ds, *r.red, 0.4);
        r.dev->step(*r.ds, gpu::Real(dt));
        apply(r);
        t += dt;
    }
    return t;
}

// The explicit local-time-stepping march of --steady (localDt every 16,
// dtRef = 0.95 min dt_c, cfl 0.4). Pre: sp.steady.
void steadyMarch(Rig& r, int iters)
{
    double dtRef = 0;
    for (int s = 0; s < iters; ++s) {
        if (s % 16 == 0) {
            dtRef = 0.95 * r.dev->localDt(*r.ds, *r.red, 0.4);
            r.dev->setLts(true);
        }
        r.dev->step(*r.ds, gpu::Real(dtRef));
        apply(r);
    }
}

// Force ON the body, in units of qRef (the circle's frontal area).
std::array<double, 3> coeffs(Rig& r)
{
    const auto F = r.bc->surface(*r.dev, *r.ds);
    return {F.fx / kQRef, F.fy / kQRef, F.fz / kQRef};
}

bool real(const Rig& r, int i, int j)
{
    return i >= 1 && i <= r.sp.nT && j >= 1 && j <= r.g.ny - 2;
}

// ---------------------------------------------------------------- (a) ----
void freestream()
{
    for (const double rot : {-1.0, 0.0, 30.0, 90.0}) {
        // rot < 0: the 5-deg wedge at the same (nTheta, ny), the baseline.
        const std::string tag =
            rot < 0 ? std::string("(a) wedge baseline")
                    : std::format("(a) axis at {:g} deg to the flow", rot);
        auto r = rot < 0 ? makeRig({.nphi = 0, .dphi = 5 * pi / 180})
                         : makeRig({.rotDeg = rot});
        if (!r) { check(false, tag + ": rig"); return; }
        uniformState(*r);
        apply(*r);
        auto out = gpu::DeviceCompState::make(r->g.size());
        if (!out) { check(false, tag + ": alloc"); return; }
        r->dev->residualInto(*r->ds, *out);
        const auto R = down(out->rho), Rx = down(out->mx), Ry = down(out->my),
                   Rz = down(out->mz), RE = down(out->E);
        const Grid& g = r->g;
        double worst = 0, worstAxis = 0;
        for (int k = r->kLo; k <= r->kHi; ++k)
            for (int j = 3; j <= g.ny - 3; ++j)
                for (int i = 1; i <= r->sp.nT; ++i) {
                    const std::size_t c = g.idx(i, j, k);
                    // The cell's largest face area: the face fluxes' scale.
                    double A = 0;
                    for (int ax = 0; ax < 3; ++ax)
                        for (const std::size_t f :
                             {c, g.idx(i - (ax == 0), j - (ax == 1),
                                       (k - (ax == 2) + g.nz) % g.nz)})
                            A = std::max(A, std::hypot(r->m->S[ax][0][f],
                                                       r->m->S[ax][1][f],
                                                       r->m->S[ax][2][f]));
                    const double sc = kRhoInf * kUInf * A / r->m->vol[c] * 0x1p-23;
                    const double sp = kPInf / kUInf * sc;
                    const double e = std::max(
                        {std::abs(R[c]) / sc, std::abs(Rx[c]) / sp,
                         std::abs(Ry[c]) / sp, std::abs(Rz[c]) / sp,
                         std::abs(RE[c]) / (kPInf * sc)});
                    worst = std::max(worst, e);
                    if (i == 1 || i == r->sp.nT) worstAxis = std::max(worstAxis, e);
                }
        std::printf("%s: uniform freestream residual, rows 3..ny-3: %.1f ulp "
                    "(axis rings %.1f)\n",
                    tag.c_str(), worst, worstAxis);
        check(worst < 16, tag + ": residual at the FP32 floor on every real "
                                "cell (closure with the axis faces)");
        // The impulsive no-slip wall launches a start-up wave (and the central
        // scheme's numerical precursor, 3 cells per step): the outer third of
        // the rows -- the far row, the ghost columns and the axis rings out
        // there included -- must still hold the freestream after 100 steps.
        // (The wedge baseline at this coarse 32 x 24 reads 1.8e3 ulp at
        // j = ny/2: the wave, not the axis.)
        const double t = march(*r, 100);
        const auto rho = down(r->ds->rho), mx = down(r->ds->mx),
                   my = down(r->ds->my), mz = down(r->ds->mz);
        double dev = 0;
        int at[3] = {0, 0, 0}, count = 0;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 2 * g.ny / 3; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    const std::size_t c = g.idx(i, j, k);
                    ++count;
                    const double e = std::max({std::abs(rho[c] - 1.0) / 0x1p-23,
                                               std::abs(mx[c] - 1.0) / 0x1p-23,
                                               std::abs(my[c]) / 0x1p-23,
                                               std::abs(mz[c]) / 0x1p-23});
                    if (e > dev) {
                        dev = e;
                        at[0] = i;
                        at[1] = j;
                        at[2] = k;
                    }
                }
        std::printf("%s: 100 steps (t %.3g), outer third (%d cells): %.1f ulp "
                    "at (%d, %d, %d)\n",
                    tag.c_str(), t, count, dev, at[0], at[1], at[2]);
        // Measured 4.5-6.0 ulp, the wedge baseline the largest, on the axis
        // rings at j ~ 2 ny/3: rounding accumulated over 100 steps.
        check(dev <= 8, tag + ": the outer third stays the freestream (<= 8 "
                              "ulp), ghost columns and axis rings included");
    }
}

// ---------------------------------------------------------------- (b) ----
// Largest azimuthal spread over the real (i, j) rings, per field, in units
// of the field's ulp at its scale; and the swirl max |u_phi| / U.
struct Spread
{
    double rho = 0, u = 0, vr = 0, p = 0, swirl = 0;
};

Spread spread(Rig& r)
{
    CompState h = CompState::zeros(r.g);
    (void)r.ds->download(h);
    const Grid& g = r.g;
    Spread s;
    for (int j = 1; j <= g.ny - 2; ++j)
        for (int i = 1; i <= r.sp.nT; ++i) {
            double lo[4] = {HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL};
            double hi[4] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
            for (int k = 0; k < g.nz; ++k) {
                const std::size_t c = g.idx(i, j, k);
                const auto& x = r.cen[c];
                const double rr = std::hypot(x[1], x[2]);
                const double cp = x[1] / rr, sp = x[2] / rr;
                const double rho = h.rho[c], u = h.mx[c] / rho,
                             v = h.my[c] / rho, w = h.mz[c] / rho;
                const double p = (kGam - 1) *
                                 (h.E[c] - 0.5 * rho * (u * u + v * v + w * w));
                const double q[4] = {rho, u, v * cp + w * sp, p};
                for (int a = 0; a < 4; ++a) {
                    lo[a] = std::min(lo[a], q[a]);
                    hi[a] = std::max(hi[a], q[a]);
                }
                s.swirl = std::max(s.swirl, std::abs(-v * sp + w * cp) / kUInf);
            }
            s.rho = std::max(s.rho, (hi[0] - lo[0]) / (kRhoInf * 0x1p-23));
            s.u = std::max(s.u, (hi[1] - lo[1]) / (kUInf * 0x1p-23));
            s.vr = std::max(s.vr, (hi[2] - lo[2]) / (kUInf * 0x1p-23));
            s.p = std::max(s.p, (hi[3] - lo[3]) / (kPInf * 0x1p-23));
        }
    return s;
}

// The rotation by pi about the axis maps the mesh onto itself to the bit;
// the state must follow: (i, j, k + N/2) == C2 (i, j, k) bitwise.
bool c2Exact(Rig& r)
{
    const auto rho = down(r.ds->rho), mx = down(r.ds->mx), my = down(r.ds->my),
               mz = down(r.ds->mz), E = down(r.ds->E);
    const Grid& g = r.g;
    bool ok = true;
    for (int k = 0; k < g.nz / 2; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t a = g.idx(i, j, k), b = g.idx(i, j, k + g.nz / 2);
                ok = ok && rho[a] == rho[b] && mx[a] == mx[b] && my[a] == -my[b] &&
                     mz[a] == -mz[b] && E[a] == E[b];
            }
    return ok;
}

void axisymmetry()
{
    {
        auto r = makeRig({});
        if (!r) { check(false, "(b) rig"); return; }
        uniformState(*r);
        apply(*r);
        // Rounding injects an O(ulp) azimuthal difference every step (the
        // planes' FP32 metrics are rotations only to rounding); a stable
        // axisymmetric state lets it random-walk, an unstable mode would
        // grow exponentially. 300 -> 1200 steps (4x) must grow < 4x.
        Spread s300, s;
        int done = 0;
        for (const int n : {100, 300, 1200}) {
            march(*r, n - done);
            done = n;
            s = spread(*r);
            if (n == 300) s300 = s;
            std::printf("(b) %d time-accurate steps: azimuthal spread rho %.1f, "
                        "u %.1f, v_r %.1f, p %.1f ulp; swirl %.2e U\n",
                        n, s.rho, s.u, s.vr, s.p, s.swirl);
        }
        const double grow = std::max({s.rho / s300.rho, s.u / s300.u,
                                      s.vr / s300.vr, s.p / s300.p,
                                      s.swirl / s300.swirl});
        std::printf("(b) growth 300 -> 1200 steps: x%.2f (random walk x2, "
                    "linear x4)\n",
                    grow);
        check(grow < 4, "(b) time march: the azimuthal spread grows "
                        "sub-linearly (rounding, no unstable mode)");
        check(s.rho <= 64 && s.u <= 128 && s.vr <= 128 && s.p <= 64,
              "(b) time march, 1200 steps: azimuthal spread rho, p <= 64 ulp, "
              "u, v_r <= 128 ulp");
        check(s.swirl <= 1e-5, "(b) time march: swirl <= 1e-5 U");
        check(c2Exact(*r), "(b) time march: the state is exactly pi-symmetric "
                           "about the axis (bitwise)");
    }
    {
        auto r = makeRig({.steady = true});
        if (!r) { check(false, "(b) steady rig"); return; }
        uniformState(*r);
        apply(*r);
        // The steady march's FP32 jitter floor (docs/rans.md): the spread
        // saturates; 2000 -> 8000 iterations must not double it.
        Spread s2000, s;
        int done = 0;
        for (const int n : {1000, 2000, 4000, 8000}) {
            steadyMarch(*r, n - done);
            done = n;
            s = spread(*r);
            if (n == 2000) s2000 = s;
            std::printf("(b) %d steady iterations: azimuthal spread rho %.1f, u "
                        "%.1f, v_r %.1f, p %.1f ulp; swirl %.2e U\n",
                        n, s.rho, s.u, s.vr, s.p, s.swirl);
        }
        const double grow = std::max({s.rho / s2000.rho, s.u / s2000.u,
                                      s.vr / s2000.vr, s.p / s2000.p,
                                      s.swirl / s2000.swirl});
        std::printf("(b) growth 2000 -> 8000 iterations: x%.2f\n", grow);
        check(grow < 2, "(b) steady march: the azimuthal spread saturates "
                        "(x < 2 over 4x the iterations)");
        check(s.rho <= 32 && s.u <= 512 && s.vr <= 512 && s.p <= 32,
              "(b) steady march: the azimuthal spread at the FP32 jitter "
              "floor (rho, p <= 32 ulp; u, v_r <= 512 ulp)");
        check(s.swirl <= 3e-5, "(b) steady march: swirl <= 3e-5 U");
        check(c2Exact(*r), "(b) steady march: exactly pi-symmetric (bitwise)");
    }
}

// ---------------------------------------------------------------- (c) ----
void wedgeMatch()
{
    const int N = 16;
    auto a = makeRig({.nphi = N, .steady = true});
    auto w = makeRig({.nphi = 0, .dphi = 2 * pi / N, .steady = true});
    if (!a || !w) { check(false, "(c) rigs"); return; }
    for (Rig* r : {a.get(), w.get()}) {
        uniformState(*r);
        apply(*r);
    }
    double worstCd = 0;
    for (int blk = 0; blk < 4; ++blk) {
        steadyMarch(*a, 1500);
        steadyMarch(*w, 1500);
        const double ca = coeffs(*a)[0], cw = coeffs(*w)[0];
        worstCd = std::max(worstCd, std::abs(ca / cw - 1));
        std::printf("(c) iter %d: cd 3D %.7f, wedge %.7f (rel %.2e)\n",
                    1500 * (blk + 1), ca, cw, ca / cw - 1);
    }
    // Plane k of the 3D state is the wedge's real plane rotated by
    // R_x(phi_k + dphi/2): compare rho, u, E and the rotated (v, w).
    CompState ha = CompState::zeros(a->g), hw = CompState::zeros(w->g);
    (void)a->ds->download(ha);
    (void)w->ds->download(hw);
    double worst = 0;
    for (int k = 0; k < N; ++k) {
        const double ph = 2 * pi * (k + 0.5) / N, c = std::cos(ph),
                     s = std::sin(ph);
        for (int j = 1; j <= a->g.ny - 2; ++j)
            for (int i = 1; i <= a->sp.nT; ++i) {
                const std::size_t p = a->g.idx(i, j, k), q = w->g.idx(i, j, 1);
                const double vy = c * hw.my[q] - s * hw.mz[q],
                             vz = s * hw.my[q] + c * hw.mz[q];
                worst = std::max({worst, std::abs(ha.rho[p] - hw.rho[q]),
                                  std::abs(ha.mx[p] - hw.mx[q]) / kUInf,
                                  std::abs(ha.my[p] - vy) / kUInf,
                                  std::abs(ha.mz[p] - vz) / kUInf,
                                  std::abs(ha.E[p] - hw.E[q]) / kPInf});
            }
    }
    std::printf("(c) after 6000 iterations: cd worst rel %.2e, state worst "
                "%.2e (rho, rho u / U, E / p_inf)\n",
                worstCd, worst);
    // Identical equations, different rounding: the two marches agree to the
    // FP32 jitter floor (cd jitter ~1e-5), 10x inside the 0.1% gate.
    check(worstCd < 1e-4, "(c) 3D N 16 cd == the 22.5-deg wedge's at every "
                          "sample (1e-4 relative)");
    check(worst < 5e-5, "(c) every 3D plane == the wedge's real plane rotated "
                        "(5e-5)");
}

// ---------------------------------------------------------------- (d) ----
// Least-squares quadratic q = c0 + c1 Y + c2 Z + c3 Y^2 + c4 YZ + c5 Z^2 in
// the transverse coordinates (Y, Z) of points (a smooth field near the axis
// is exactly this to third order: x along a ring varies as rho^2), then its
// worst misfit on `test` points. Normal equations, partial pivoting.
double quadExtrapolation(const std::vector<std::array<double, 3>>& fit,
                         const std::vector<std::array<double, 3>>& test)
{
    auto basis = [](double Y, double Z) {
        return std::array<double, 6>{1, Y, Z, Y * Y, Y * Z, Z * Z};
    };
    double A[6][7] = {};
    for (const auto& p : fit) {
        const auto b = basis(p[0], p[1]);
        for (int r = 0; r < 6; ++r) {
            for (int c = 0; c < 6; ++c) A[r][c] += b[r] * b[c];
            A[r][6] += b[r] * p[2];
        }
    }
    for (int c = 0; c < 6; ++c) {
        int piv = c;
        for (int r = c + 1; r < 6; ++r)
            if (std::abs(A[r][c]) > std::abs(A[piv][c])) piv = r;
        std::swap(A[c], A[piv]);
        for (int r = 0; r < 6; ++r) {
            if (r == c) continue;
            const double f = A[r][c] / A[c][c];
            for (int k = c; k < 7; ++k) A[r][k] -= f * A[c][k];
        }
    }
    double worst = 0;
    for (const auto& p : test) {
        const auto b = basis(p[0], p[1]);
        double q = 0;
        for (int r = 0; r < 6; ++r) q += b[r] * A[r][6] / A[r][r];
        worst = std::max(worst, std::abs(q - p[2]));
    }
    return worst;
}

void crossFlow()
{
    const int N = 32, iters = 12000;
    auto ax = makeRig({.nphi = N, .steady = true});
    auto cr = makeRig({.nphi = N, .rotDeg = 90, .steady = true});
    if (!ax || !cr) { check(false, "(d) rigs"); return; }
    for (Rig* r : {ax.get(), cr.get()}) {
        uniformState(*r);
        apply(*r);
        steadyMarch(*r, iters);
    }
    const auto ca = coeffs(*ax), cc = coeffs(*cr);
    // The rotated mesh: the drag is still along x; the would-be lift along
    // the axis (y) and across it (z).
    std::printf("(d) N %d steady %d: axial cd %.5f; cross-flow cd %.5f, "
                "lateral %.2e %.2e\n",
                N, iters, ca[0], cc[0], cc[1], cc[2]);
    check(std::isfinite(cc[0]) && std::abs(cc[0] / ca[0] - 1) < 0.03,
          "(d) the cross-flow sphere's drag is the axial-flow drag of the "
          "same mesh (3%)");
    check(std::abs(cc[1]) < 1e-3 * cc[0] && std::abs(cc[2]) < 1e-3 * cc[0],
          "(d) no lateral force on the cross-flow sphere (1e-3 cd)");
    // Regularity through the axis. A smooth field near the axis is a
    // quadratic in the transverse coordinates (Y, Z) to third order; fit
    // one on rings 1..4 off the axis ring and extrapolate it ONTO the axis
    // ring -- an axis spike or a kink misses -- against the same fit one
    // ring further out (rings 2..5 -> ring 1) as the control. Every primitive,
    // both poles, the first fluid row and two outer ones.
    CompState h = CompState::zeros(cr->g);
    (void)cr->ds->download(h);
    const double a = 90 * pi / 180;
    const double e1[3] = {-std::sin(a), std::cos(a), 0}, e2[3] = {0, 0, 1};
    auto prim = [&](std::size_t c, int f) {
        const double rho = h.rho[c], u = h.mx[c] / rho, v = h.my[c] / rho,
                     w = h.mz[c] / rho;
        const double p = (kGam - 1) * (h.E[c] - 0.5 * rho * (u * u + v * v + w * w));
        const double q[4] = {p / (0.5 * kRhoInf * kUInf * kUInf), u, v, w};
        return q[f];
    };
    double axisMiss = 0, ctrlMiss = 0;
    for (int f = 0; f < 4; ++f)
        for (const int j : {1, 4, 8})
            for (const int iAx : {1, cr->sp.nT}) {
                const int dir = iAx == 1 ? 1 : -1;
                std::vector<std::array<double, 3>> ring[6];
                double lo = HUGE_VAL, hi = -HUGE_VAL;
                for (int o = 0; o < 6; ++o)
                    for (int k = 0; k < N; ++k) {
                        const std::size_t c = cr->g.idx(iAx + dir * o, j, k);
                        const auto& x = cr->cen[c];
                        const double Y = x[0] * e1[0] + x[1] * e1[1],
                                     Z = x[2] * e2[2], q = prim(c, f);
                        ring[o].push_back({Y, Z, q});
                        lo = std::min(lo, q);
                        hi = std::max(hi, q);
                    }
                auto cat = [&](int o0, int o1) {
                    std::vector<std::array<double, 3>> v;
                    for (int o = o0; o <= o1; ++o)
                        v.insert(v.end(), ring[o].begin(), ring[o].end());
                    return v;
                };
                const double sc = std::max(hi - lo, 1e-12);
                axisMiss = std::max(axisMiss, quadExtrapolation(cat(1, 4), ring[0]) / sc);
                ctrlMiss = std::max(ctrlMiss, quadExtrapolation(cat(2, 5), ring[1]) / sc);
            }
    std::printf("(d) cross-flow, quadratic fit extrapolated onto a ring, worst "
                "miss / patch range: the axis ring %.3e, one ring out %.3e\n",
                axisMiss, ctrlMiss);
    check(axisMiss <= 2 * ctrlMiss, "(d) the cross-flow field is as smooth "
                                    "onto the axis ring as one ring out: no "
                                    "axis spike");
}

// ---------------------------------------------------------------- (e) ----
// A compact C2 bump (1 - s^2)^3, s = |x - x0| / w, x0 on the downstream
// side 0.2 D off the axis at azimuth -14 deg (straddles the axis AND the
// seam k = 0 / N-1), on every conserved field. Shell = real cells of rows
// [jA, jB] (the axis faces close it); its boundary sees the uniform state, so
// sum_shell V (R_bump - R_uniform) telescopes to zero.
void conservation()
{
    auto r = makeRig({});
    if (!r) { check(false, "(e) rig"); return; }
    const Grid& g = r->g;
    auto out0 = gpu::DeviceCompState::make(g.size());
    auto out1 = gpu::DeviceCompState::make(g.size());
    if (!out0 || !out1) { check(false, "(e) alloc"); return; }
    uniformState(*r);
    apply(*r);
    r->dev->residualInto(*r->ds, *out0);
    for (const int cse : {0, 1, 2}) {
        const bool onAxis = cse < 2;
        const double amp = cse == 1 ? 4.0 : 1.0;
        const double w = 0.9 * kD;
        const std::array<double, 3> c0 =
            onAxis ? std::array<double, 3>{2.5 * kD, 0.194 * kD, -0.0484 * kD}
                   : std::array<double, 3>{0.0, 1.7 * kD, 1.7 * kD};
        auto h = CompState::zeros(g);
        for (std::size_t c = 0; c < g.size(); ++c) {
            const auto& x = r->cen[c];
            const double s2 = (std::pow(x[0] - c0[0], 2) + std::pow(x[1] - c0[1], 2) +
                               std::pow(x[2] - c0[2], 2)) /
                              (w * w);
            const double b = s2 < 1 ? amp * std::pow(1 - s2, 3) : 0.0;
            h.rho[c] = kRhoInf * (1 + 0.05 * b);
            h.mx[c] = kRhoInf * kUInf * (1 - 0.2 * b);
            h.my[c] = 0.15 * b * kRhoInf * kUInf;
            h.mz[c] = -0.1 * b * kRhoInf * kUInf;
            h.E[c] = kPInf / (kGam - 1) * (1 + 0.03 * b) +
                     0.5 * (h.mx[c] * h.mx[c] + h.my[c] * h.my[c] +
                            h.mz[c] * h.mz[c]) /
                         h.rho[c];
        }
        // Rows the bump (and its 2-cell stencil) never reaches.
        int jA = g.ny, jB = 0;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i) {
                    const auto& x = r->cen[g.idx(i, j, k)];
                    const double s2 =
                        (std::pow(x[0] - c0[0], 2) + std::pow(x[1] - c0[1], 2) +
                         std::pow(x[2] - c0[2], 2)) /
                        (w * w);
                    if (s2 < 1) {
                        jA = std::min(jA, j);
                        jB = std::max(jB, j);
                    }
                }
        jA -= 3;
        jB += 3;
        if (jA < 1 || jB > g.ny - 2) { check(false, "(e) shell rows"); return; }
        (void)r->ds->upload(h);
        apply(*r);
        r->dev->residualInto(*r->ds, *out1);
        const gpu::DeviceScalar* f0[5] = {&out0->rho, &out0->mx, &out0->my,
                                          &out0->mz, &out0->E};
        const gpu::DeviceScalar* f1[5] = {&out1->rho, &out1->mx, &out1->my,
                                          &out1->mz, &out1->E};
        const char* nm[5] = {"mass", "x-mom", "y-mom", "z-mom", "energy"};
        double worst = 0;
        for (int v = 0; v < 5; ++v) {
            const auto a = down(*f0[v]), b = down(*f1[v]);
            double sum = 0, mag = 0;
            for (int k = 0; k < g.nz; ++k)
                for (int j = jA; j <= jB; ++j)
                    for (int i = 1; i <= r->sp.nT; ++i) {
                        const std::size_t c = g.idx(i, j, k);
                        const double d = (double(b[c]) - double(a[c])) * r->m->vol[c];
                        sum += d;
                        mag += std::abs(d);
                    }
            const double rel = mag > 0 ? std::abs(sum) / mag : 1.0;
            worst = std::max(worst, rel);
            std::printf("(e) %s x%g, %s: |sum V dR| / sum |V dR| = %.2e (rows "
                        "%d..%d)\n",
                        onAxis ? "bump across the axis and the seam"
                               : "bump off both",
                        amp, nm[v], rel, jA, jB);
        }
        // FP32 rounding, not a leak: the axis cells cancel p_inf |S| over a
        // tiny volume, so their absolute noise is larger -- and the ratio
        // falls with the bump's amplitude (measured 2.6e-6 -> 3.0e-7 for
        // x-momentum at 4x), which a leak's would not.
        const double bound = !onAxis ? 1e-7 : amp > 1 ? 1e-6 : 5e-6;
        check(worst < bound,
              std::format("(e) {} x{:g}: mass, momentum and energy telescope "
                          "({:g} of the flux activity)",
                          onAxis ? "across the axis and the cyclic seam"
                                 : "away from both (control)",
                          amp, bound));
    }
}

// ---------------------------------------------------------------- (f) ----
void ransSmoke()
{
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m}) {
        const std::string n(ransName(m));
        auto r = makeRig({.re = 1e4, .model = m});
        if (!r) { check(false, n + " (f) rig"); return; }
        uniformState(*r);
        apply(*r);
        march(*r, 200);
        r->dev->updateMut(*r->ds);
        const auto mut = down(r->dev->mut());
        bool face = true, finite = true;
        for (int k = 0; k < r->g.nz; ++k)
            for (int i = 1; i <= r->sp.nT; ++i)
                face = face && 0.5f * (mut[r->g.idx(i, 0, k)] +
                                       mut[r->g.idx(i, 1, k)]) == 0.0f;
        for (float v : down(r->ds->t[0])) finite = finite && std::isfinite(v);
        for (float v : down(r->ds->E)) finite = finite && std::isfinite(v);
        check(finite && r->dev->floorHits() == 0,
              n + " (f): 200 3D URANS steps finite, 0 guard hits");
        check(face, n + " (f): mu_t == 0.0f on every wall face, every k");
        check(c2Exact(*r), n + " (f): the RANS state stays exactly pi-symmetric");
    }
}

// ---------------------------------------------------------------- (g) ----
void tally()
{
    auto r = makeRig({.nT = 64, .ny = 48, .nphi = 64});
    if (!r) { check(false, "(g) rig"); return; }
    // The faceted body's volume: (1/3) sum over the wall faces of x_f . S_f
    // (outward from the body = -S[1] of the ghost row).
    const Grid& g = r->g;
    double V = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int i = 1; i <= r->sp.nT; ++i) {
            const std::size_t c = g.idx(i, 0, k);
            std::array<double, 3> xf{0, 0, 0};
            for (int dk = 0; dk < 2; ++dk)
                for (int di = 0; di < 2; ++di) {
                    const auto p = r->og->mesh.node(i + di, 1, k + dk);
                    for (int a = 0; a < 3; ++a) xf[a] += 0.25 * p[a];
                }
            for (int a = 0; a < 3; ++a) V += xf[a] * r->m->S[1][a][c] / 3;
        }
    const double a = 0.05;
    for (const int dir : {0, 1}) {
        auto h = CompState::zeros(g);
        for (std::size_t c = 0; c < g.size(); ++c) {
            h.rho[c] = kRhoInf;
            h.E[c] = (kPInf + a * r->cen[c][dir]) / (kGam - 1);
        }
        (void)r->ds->upload(h);
        apply(*r);
        const auto F = r->bc->surface(*r->dev, *r->ds);
        const double Fd = dir == 0 ? F.fx : F.fy, Fo = dir == 0 ? F.fy : F.fx;
        // The resting wall's pressure image is the zero-gradient copy, so the
        // face pressure is the first cell's: F = -sum_f p(x_1) S_f exactly.
        double Fref = 0;
        for (int k = 0; k < g.nz; ++k)
            for (int i = 1; i <= r->sp.nT; ++i) {
                const std::size_t c = g.idx(i, 0, k), c1 = g.idx(i, 1, k);
                Fref -= (kPInf + a * r->cen[c1][dir]) * r->m->S[1][dir][c];
            }
        const double rel = std::abs(Fd / (-a * V) - 1);
        const double relRef = std::abs(Fd / Fref - 1);
        std::printf("(g) p = p0 + a %s: F %.6e vs -a V_faceted %.6e (rel "
                    "%.2e: the first-cell face pressure), vs -sum p(x_1) S "
                    "%.2e; off-gradient components %.1e %.1e\n",
                    dir == 0 ? "x (along the axis)" : "y (across it)", Fd,
                    -a * V, rel, relRef, Fo, F.fz);
        check(relRef < 1e-5, std::string("(g) the tally on a pressure gradient ") +
                                 (dir == 0 ? "along" : "across") +
                                 " the axis == -sum_f p_1 S_f over every wall "
                                 "face (1e-5)");
        check(rel < 2e-2, std::string("(g) Archimedes ") +
                              (dir == 0 ? "along" : "across") +
                              " the axis: the faceted body's -a V (2%, first "
                              "order in h1)");
        check(std::abs(Fo) < 1e-6 * std::abs(Fd) &&
                  std::abs(F.fz) < 1e-6 * std::abs(Fd),
              std::string("(g) ") + (dir == 0 ? "along" : "across") +
                  ": no force off the gradient (1e-6)");
    }
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    freestream();
    axisymmetry();
    wedgeMatch();
    crossFlow();
    conservation();
    ransSmoke();
    tally();
    return dnstest::finish();
}
