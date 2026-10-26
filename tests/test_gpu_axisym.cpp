// The axisymmetric wedge on the GPU (masterplan_flowrans P2a; docs/rans.md
// "Axisymmetric wedge and the revolved O-grid"):
//   (a) a uniform x-freestream is a fixed point: its residual on the real
//       cells is at the FP32 floor (every cell's closure holds with the
//       degenerate axis faces), and 100 steps leave the outer rows <= 4 ulp
//   (b) the side-face mass flux of a random axisymmetric no-swirl state
//       vanishes to O(ulp): <= 8 ulp * |rho v| A
//   (c) the ghost columns are R_x(pi) and the ghost planes R_x(-+dphi) of
//       the real plane, bit-exact against a host recomputation
//   (d) w == 0 in the real plane is preserved over 200 steps (|w| <= 1e-6 U)
//   (e) a 200-step laminar Re 100 wedge smoke: finite, positive drag
// Seconds.
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
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <random>
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
constexpr double kDphi = 5.0 * 3.14159265358979323846 / 180.0;

struct Rig
{
    Grid g{1, 1, 1, 1, 1, 1};
    std::unique_ptr<OGrid> og;
    std::unique_ptr<CurvMetrics> m;
    std::unique_ptr<gpu::DeviceComp> dev;
    std::unique_ptr<gpu::DeviceCompState> ds;
    std::unique_ptr<gpu::DeviceOGridBc> bc;
    std::unique_ptr<gpu::DeviceAxisBc> axis;
    std::unique_ptr<gpu::ReduceScratch> red;
    int nT = 0;
    double phi[2] = {0, 0};
};

std::unique_ptr<Rig> makeRig(int nT, int ny, double re,
                             RansModel model = RansModel::None)
{
    auto r = std::make_unique<Rig>();
    const RevolvedSpec rv{.wedge = true, .dphi = kDphi};
    r->nT = nT;
    r->g = ogridRevolvedGrid(nT, ny, rv, 0.5 * kD, 20 * kD);
    const Sdf sphere = [](double x, double y, double z) {
        return std::sqrt(x * x + y * y + z * z) - 0.5 * kD;
    };
    auto og = buildRevolvedOGrid(r->g, sphere, {.rFar = 20 * kD, .beta = 3.0},
                                 rv);
    if (!og) return nullptr;
    r->og = std::make_unique<OGrid>(std::move(*og));
    auto m = CurvMetrics::build(r->og->mesh);
    if (!m) return nullptr;
    r->m = std::make_unique<CurvMetrics>(std::move(*m));
    const Grid& g = r->g;
    const auto tr = Transport::air(kRhoInf * kUInf * kD / re, kTInf, 300.0,
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
    c.faceIHi = nT;
    c.faceKLo = c.faceKHi = 1;
    c.tallyScale = 2 * pi / kDphi;
    const auto d = revolvedWallDistance(r->og->mesh, nT, true);
    c.wallD1 = wallFirstCellDistance(g, d);
    c.rans = model;
    const double nu = kD / re;
    if (model == RansModel::SaNeg) c.phiInf[0] = nuTildaInf(kRansNutInfRatio, nu);
    if (ransSst(model)) {
        c.phiInf[0] = kInf(kRansKInf, kUInf);
        c.phiInf[1] = omegaInf(c.phiInf[0], kRansMutInfRatio, nu);
    }
    r->phi[0] = c.phiInf[0];
    r->phi[1] = c.phiInf[1];
    auto dev = gpu::DeviceComp::make(*r->m, tr);
    auto ds = gpu::DeviceCompState::make(g.size(), ransScalars(model));
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
    if (model != RansModel::None) {
        std::vector<std::uint8_t> mask(g.size(), 0);
        for (int j = 1; j <= g.ny - 2; ++j)
            for (int i = 1; i <= nT; ++i) mask[g.idx(i, j, 1)] = 1;
        if (!p->dev->enableRans({.model = model,
                                 .wallDist = d,
                                 .mutHook =
                                     [p](gpu::Real* n, gpu::Real* f) {
                                         p->bc->mutHook()(n, f);
                                         p->axis->applyMut(n, f);
                                     },
                                 .fluidMask = mask}))
            return nullptr;
    }
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

// Node-mean centroid.
std::array<double, 3> centroid(const CurvMesh& m, int i, int j, int k)
{
    std::array<double, 3> c{0, 0, 0};
    for (int dk = 0; dk < 2; ++dk)
        for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di) {
                const auto p = m.node(i + di, j + dj, k + dk);
                for (int a = 0; a < 3; ++a) c[a] += 0.125 * p[a];
            }
    return c;
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

// An axisymmetric no-swirl state: on every cell, (rho, u, v_rho, p) of its
// centroid's (x, rho), velocity (u, v_rho rho^). Random smooth modes.
void axisymState(Rig& r, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(-1, 1);
    const double a[4] = {U(rng), U(rng), U(rng), U(rng)};
    auto h = CompState::zeros(r.g);
    for (int k = 0; k < r.g.nz; ++k)
        for (int j = 0; j < r.g.ny; ++j)
            for (int i = 0; i < r.g.nx; ++i) {
                const auto x = centroid(r.og->mesh, i, j, k);
                const double rho = std::hypot(x[1], x[2]);
                const double rh[2] = {rho > 0 ? x[1] / rho : 1.0,
                                      rho > 0 ? x[2] / rho : 0.0};
                const double d = 1.0 + 0.1 * std::sin(3 * x[0] + a[0]) *
                                            std::cos(2 * rho + a[1]);
                const double u = 0.8 + 0.2 * std::cos(2 * x[0] + a[2]);
                const double vr = 0.3 * std::sin(2 * rho + a[3]) * rho;
                const double p = kPInf * (1 + 0.01 * std::cos(x[0] + rho));
                const std::size_t c = r.g.idx(i, j, k);
                h.rho[c] = d;
                h.mx[c] = d * u;
                h.my[c] = d * vr * rh[0];
                h.mz[c] = d * vr * rh[1];
                h.E[c] = p / (kGam - 1) + 0.5 * d * (u * u + vr * vr);
            }
    (void)r.ds->upload(h);
}

void freestream()
{
    auto r = makeRig(64, 48, 100);
    if (!r) { check(false, "(a) rig"); return; }
    uniformState(*r);
    apply(*r);
    auto out = gpu::DeviceCompState::make(r->g.size());
    if (!out) { check(false, "(a) alloc"); return; }
    r->dev->residualInto(*r->ds, *out);
    const auto R = down(out->rho), Rx = down(out->mx), Ry = down(out->my);
    const Grid& g = r->g;
    // Scale: |rho U A| / V of each cell, in ulps of the face fluxes.
    double worst = 0;
    for (int j = 3; j <= g.ny - 3; ++j)
        for (int i = 1; i <= r->nT; ++i) {
            const std::size_t c = g.idx(i, j, 1);
            const double A = std::hypot(r->m->S[1][0][c], r->m->S[1][1][c],
                                        r->m->S[1][2][c]);
            const double sc = kRhoInf * kUInf * A / r->m->vol[c] * 0x1p-23;
            worst = std::max({worst, std::abs(R[c]) / sc,
                              std::abs(Rx[c]) / (kPInf / kUInf * sc),
                              std::abs(Ry[c]) / (kPInf / kUInf * sc)});
        }
    std::printf("(a): uniform freestream residual on rows 3..ny-3: %.1f ulp "
                "of the face fluxes\n",
                worst);
    check(worst < 16, "(a) the uniform freestream's residual is at the FP32 "
                      "floor (closure holds with the axis faces)");
    const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
    for (int s = 0; s < 100; ++s) {
        r->dev->step(*r->ds, gpu::Real(dt));
        apply(*r);
    }
    const auto rho = down(r->ds->rho), mx = down(r->ds->mx);
    double dev = 0;
    for (int j = g.ny / 2; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            for (int k = 0; k < 3; ++k) {
                const std::size_t c = g.idx(i, j, k);
                dev = std::max({dev, std::abs(rho[c] - 1.0) / 0x1p-23,
                                std::abs(mx[c] - 1.0) / 0x1p-23});
            }
    std::printf("(a): outer half after 100 steps: %.1f ulp\n", dev);
    check(dev <= 4, "(a) the outer rows stay at the freestream (<= 4 ulp)");
}

void sideFlux()
{
    auto r = makeRig(64, 48, 100);
    if (!r) { check(false, "(b) rig"); return; }
    axisymState(*r, 7);
    apply(*r);
    const Grid& g = r->g;
    const auto rho = down(r->ds->rho), my = down(r->ds->my),
               mz = down(r->ds->mz), mx = down(r->ds->mx);
    double worst = 0;
    for (int j = 1; j <= g.ny - 2; ++j)
        for (int i = 1; i <= r->nT; ++i) {
            // The +k face of plane 1 (between planes 1 and 2) and of plane 0.
            for (const int k : {0, 1}) {
                const std::size_t a = g.idx(i, j, k), b = g.idx(i, j, k + 1);
                const double S[3] = {r->m->S[2][0][a], r->m->S[2][1][a],
                                     r->m->S[2][2][a]};
                const double A = std::hypot(S[0], S[1], S[2]);
                double m[3];
                for (int q = 0; q < 3; ++q) {
                    const float* f = q == 0 ? mx.data() : q == 1 ? my.data()
                                                                  : mz.data();
                    m[q] = 0.5 * (double(f[a]) / rho[a] + double(f[b]) / rho[b]) *
                           0.5 * (double(rho[a]) + rho[b]);
                }
                const double flux = m[0] * S[0] + m[1] * S[1] + m[2] * S[2];
                const double vm = std::hypot(m[1], m[2]) + std::abs(m[0]);
                if (vm * A > 0)
                    worst = std::max(worst, std::abs(flux) / (vm * A * 0x1p-23));
            }
        }
    std::printf("(b): side-face mass flux of an axisymmetric state: %.2f ulp "
                "of |rho v| A\n",
                worst);
    check(worst <= 8, "(b) the wedge side faces carry no mass (<= 8 ulp)");
}

void ghostsExact()
{
    auto r = makeRig(64, 48, 100);
    if (!r) { check(false, "(c) rig"); return; }
    axisymState(*r, 11);
    apply(*r);
    const Grid& g = r->g;
    const auto rho = down(r->ds->rho), mx = down(r->ds->mx),
               my = down(r->ds->my), mz = down(r->ds->mz), E = down(r->ds->E);
    const float c = float(std::cos(kDphi)), s = float(std::sin(kDphi));
    bool cols = true, planes = true;
    for (int j = 0; j < g.ny; ++j)
        for (const auto& [ig, is] : {std::pair{0, 1}, {g.nx - 1, g.nx - 2}}) {
            const std::size_t d = g.idx(ig, j, 1), q = g.idx(is, j, 1);
            cols = cols && rho[d] == rho[q] && mx[d] == mx[q] &&
                   my[d] == -my[q] && mz[d] == -mz[q] && E[d] == E[q];
        }
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t q = g.idx(i, j, 1);
            for (const int k : {0, 2}) {
                const float sk = k == 0 ? -s : s;
                const std::size_t d = g.idx(i, j, k);
                const float yv = std::fma(c, my[q], -(sk * mz[q]));
                const float zv = std::fma(sk, my[q], c * mz[q]);
                planes = planes && rho[d] == rho[q] && mx[d] == mx[q] &&
                         E[d] == E[q] && my[d] == yv && mz[d] == zv;
            }
        }
    check(cols, "(c) ghost columns == R_x(pi) of their source, bitwise");
    check(planes, "(c) ghost planes == R_x(-+dphi) of the real plane, bitwise");
}

void noSwirl()
{
    auto r = makeRig(64, 48, 100);
    if (!r) { check(false, "(d) rig"); return; }
    uniformState(*r);
    apply(*r);
    const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
    for (int s = 0; s < 200; ++s) {
        r->dev->step(*r->ds, gpu::Real(dt));
        apply(*r);
    }
    const Grid& g = r->g;
    const auto rho = down(r->ds->rho), mz = down(r->ds->mz);
    double w = 0;
    for (int j = 1; j <= g.ny - 2; ++j)
        for (int i = 1; i <= r->nT; ++i) {
            const std::size_t c = g.idx(i, j, 1);
            w = std::max(w, std::abs(double(mz[c]) / rho[c]));
        }
    std::printf("(d): max |w| in the real plane after 200 steps: %.2e U\n", w);
    check(w <= 1e-6, "(d) w == 0 in the real plane is preserved (<= 1e-6 U)");
}

void smoke()
{
    auto r = makeRig(64, 48, 100);
    if (!r) { check(false, "(e) rig"); return; }
    uniformState(*r);
    apply(*r);
    double dt = 0;
    for (int s = 0; s < 200; ++s) {
        if (s % 16 == 0) dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
        r->dev->step(*r->ds, gpu::Real(dt));
        apply(*r);
    }
    const auto F = r->bc->surface(*r->dev, *r->ds);
    const double cd = F.fx / (0.5 * kRhoInf * kUInf * kUInf * 0.25 * pi * kD * kD);
    bool finite = std::isfinite(cd);
    for (float v : down(r->ds->E)) finite = finite && std::isfinite(v);
    std::printf("(e): 200 impulsive steps, cd %.4f (t = %.3g)\n", cd, 200 * dt);
    check(finite && cd > 0, "(e) 200-step laminar Re 100 wedge: finite, drag > 0");
}

// (f) RANS on the wedge: 200 URANS steps at Re 1e4, 0 guard hits, mu_t
// exactly 0.0 on every real wall face.
void ransSmoke()
{
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m}) {
        const std::string n(ransName(m));
        auto r = makeRig(64, 48, 1e4, m);
        if (!r) { check(false, n + " (f) rig"); return; }
        uniformState(*r);
        apply(*r);
        double dt = 0;
        for (int s = 0; s < 200; ++s) {
            if (s % 16 == 0) dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
            r->dev->step(*r->ds, gpu::Real(dt));
            apply(*r);
        }
        r->dev->updateMut(*r->ds);
        const auto mut = down(r->dev->mut());
        bool face = true, finite = true;
        for (int i = 1; i <= r->nT; ++i)
            face = face && 0.5f * (mut[r->g.idx(i, 0, 1)] +
                                   mut[r->g.idx(i, 1, 1)]) == 0.0f;
        for (float v : down(r->ds->t[0])) finite = finite && std::isfinite(v);
        check(finite && r->dev->floorHits() == 0,
              n + " (f): 200 wedge URANS steps finite, 0 guard hits");
        check(face, n + " (f): mu_t == 0.0f on every real wall face");
    }
}

// (g) the tally's x 2 pi/dphi on a body at rest in a linear pressure field
// p = p0 + a x: the revolved closed surface feels Archimedes' -a V_sphere
// (the face pressure is the first cell's, an O(h1) error), and its lateral
// forces are exactly 0 by symmetry.
void tallyScaling()
{
    auto r = makeRig(128, 96, 100);
    if (!r) { check(false, "(g) rig"); return; }
    const double a = 0.05;
    auto h = CompState::zeros(r->g);
    for (int k = 0; k < r->g.nz; ++k)
        for (int j = 0; j < r->g.ny; ++j)
            for (int i = 0; i < r->g.nx; ++i) {
                const auto x = centroid(r->og->mesh, i, j, k);
                const std::size_t c = r->g.idx(i, j, k);
                h.rho[c] = kRhoInf;
                h.E[c] = (kPInf + a * x[0]) / (kGam - 1);
            }
    (void)r->ds->upload(h);
    r->bc->apply(*r->ds); // the wall row only matters (a resting image)
    r->axis->apply(*r->ds);
    const auto F = r->bc->surface(*r->dev, *r->ds);
    const double V = pi / 6 * kD * kD * kD;
    const double rel = std::abs(F.fx / (-a * V) - 1);
    std::printf("(g): wedge tally x 2 pi/dphi on p = p0 + a x: Fx %.6e vs "
                "-a V %.6e (rel %.2e)\n",
                F.fx, -a * V, rel);
    check(rel < 2e-2, "(g) the scaled wedge tally is Archimedes' force (2%)");
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    freestream();
    sideFlux();
    ghostsExact();
    noSwirl();
    smoke();
    ransSmoke();
    tallyScaling();
    return dnstest::finish();
}
