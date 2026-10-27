// RANS on the immersed ghost wall (masterplan_flowrans P5, docs/rans.md
// "Immersed-wall RANS"):
//   (a) G2-flow: on a --focal mesh with the ghost wall and NSCBC, SA with
//       nu~ == 0 (nu~_inf = 0) and SST with k == 0 leave the mean flow AND the
//       body tally bit for bit the RANS-off run's (50 steps, fixed dt)
//   (b) G3-flow: a uniform specific scalar rides a non-uniform flow through
//       the NSCBC faces (transportOnly, the open-face flux carrying the same
//       value in): 0 ulp in the interior; the x-face drift is the FAILED
//       boundary clause, pinned as a measured ceiling
//   (c) the wall-linear image: a planar wall tilted 30 deg at a sub-cell
//       offset, nu~ = k = mu_t = c d in the fluid: every ghost reproduces
//       c sdf(ghost) to FP32 rounding, and omega_g is the (1 + a) omega_w -
//       a omega_s image
//   (d) 200-step SA and SST URANS smoke on an immersed cylinder: finite,
//       mu_t == 0 on deep cells, SA 0 guard hits (SST's staircase hits are
//       an open finding, counted)
//   (e) the clamped mu_t image: an under-resolved Re 1e5 law-of-the-wall
//       layer, SA and SST, where the plain odd image takes faces to -41 mu:
//       every staircase face keeps mu + mu_t >= mu
// Seconds.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/rans.hpp"
#include "dns/remap.hpp"
#include "dns/walldist.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/reductions.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

constexpr double kGam = 1.4, kRgas = 1.0, kMach = 0.2;
constexpr double kRhoInf = 1.0, kUInf = 1.0;
const double kPInf = kRhoInf * kUInf * kUInf / (kGam * kMach * kMach);
const double kTInf = kPInf / (kRhoInf * kRgas);

// One immersed run's objects, built the way runner_comp builds them.
struct Rig
{
    Grid g{1, 1, 1, 1, 1, 1};
    AxisRemap3 remap{};
    std::unique_ptr<CurvMetrics> m;
    Transport tr{};
    RansModel model = RansModel::None;
    double phi[2] = {0, 0};
    std::unique_ptr<gpu::DeviceComp> dev;
    std::unique_ptr<gpu::DeviceCompState> ds;
    std::unique_ptr<gpu::DeviceCompFar> far;
    std::unique_ptr<gpu::DeviceCompNscbc> nsc;
    std::unique_ptr<gpu::DeviceCompGhost> ghost;   // null: no body
    std::unique_ptr<gpu::ReduceScratch> red;
    std::vector<std::uint8_t> fluid;
    std::function<void(gpu::DeviceCompState&)> bc;   // the stage hook
};

// The rig's body: a circle of D = L/4 at (0.4 L, 0.5 L), L = 2 pi.
Sdf rigCircle()
{
    const double L = 2 * std::numbers::pi;
    const double D = 0.25 * L, cx = 0.4 * L, cy = 0.5 * L;
    return [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - 0.5 * D;
    };
}

struct Opt
{
    RansModel model = RansModel::None;
    bool rans = false;       // enableRans (with model None: RANS off)
    bool body = true;
    bool focal = true;
    bool transportOnly = false;
    double phi[2] = {0, 0};
    double re = 100;
    int n = 96;
    bool wallValue = true;   // ScalarBoundary::solid (the runner's setting)
};

std::unique_ptr<Rig> makeRig(const Opt& o)
{
    auto r = std::make_unique<Rig>();
    r->model = o.model;
    r->phi[0] = o.phi[0];
    r->phi[1] = o.phi[1];
    const double L = 2 * std::numbers::pi;
    r->g = Grid(o.n, o.n, 1, L / o.n, L / o.n, 1.0);
    const double D = 0.25 * L, cx = 0.4 * L, cy = 0.5 * L;
    const double lo[3] = {cx - 0.8 * D, cy - 0.8 * D, 0},
                 hi[3] = {cx + 0.8 * D, cy + 0.8 * D, 0},
                 w[3] = {0.3 * D, 0.3 * D, 0};
    r->remap = focalRemap(r->g, o.focal, lo, hi, w, 1.5);
    auto m = CurvMetrics::build(o.focal ? CurvMesh::fromRemap(r->g, r->remap)
                                        : CurvMesh::uniform(r->g));
    if (!m) return nullptr;
    r->m = std::make_unique<CurvMetrics>(std::move(*m));
    const double mu = kRhoInf * kUInf * D / o.re;
    r->tr = Transport::air(mu, kTInf, 300.0, 0.72, kGam, kRgas, false);
    auto dev = gpu::DeviceComp::make(*r->m, r->tr);
    auto ds = gpu::DeviceCompState::make(r->g.size(), ransScalars(o.model));
    gpu::DeviceCompFar::Config fc;
    fc.rhoInf = kRhoInf;
    fc.uInf = kUInf;
    fc.pInf = kPInf;
    fc.gamma = kGam;
    fc.ns = 8;
    fc.sigMax = 0.0;   // NSCBC: no strip
    fc.phiInf[0] = o.phi[0];
    fc.phiInf[1] = o.phi[1];
    auto far = gpu::DeviceCompFar::make(r->g, fc);
    gpu::DeviceCompNscbc::Config nc;
    nc.rhoInf = kRhoInf;
    nc.uInf = kUInf;
    nc.pInf = kPInf;
    nc.gamma = kGam;
    nc.Rgas = kRgas;
    nc.TInf = kTInf;
    auto nsc = gpu::DeviceCompNscbc::make(r->g, nc);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !far || !nsc || !red) return nullptr;
    r->dev = std::make_unique<gpu::DeviceComp>(std::move(*dev));
    r->ds = std::make_unique<gpu::DeviceCompState>(std::move(*ds));
    r->far = std::make_unique<gpu::DeviceCompFar>(std::move(*far));
    r->nsc = std::make_unique<gpu::DeviceCompNscbc>(std::move(*nsc));
    r->red = std::make_unique<gpu::ReduceScratch>(std::move(*red));
    const Sdf circle = rigCircle();
    if (o.body) {
        auto gh = gpu::DeviceCompGhost::make(r->g, circle, kGam, kRhoInf,
                                             kPInf, nullptr, r->remap);
        if (!gh) return nullptr;
        r->ghost = std::make_unique<gpu::DeviceCompGhost>(std::move(*gh));
    }
    r->fluid.assign(r->g.size(), 1);
    if (r->ghost)
        for (int c : r->ghost->solidCells()) r->fluid[std::size_t(c)] = 0;
    for (int i = 0; i < r->g.nx; ++i)
        for (int j : {0, r->g.ny - 1}) r->fluid[r->g.idx(i, j, 0)] = 0;
    Rig* p = r.get();
    if (o.rans) {
        gpu::ScalarBoundary bnd;
        bnd.open[0] = 1;
        bnd.phiInf[0] = o.phi[0];
        bnd.phiInf[1] = o.phi[1];
        const auto d = o.body ? immersedWallDistance(r->g, circle, r->remap)
                              : std::vector<double>(r->g.size(), 1.0);
        std::vector<std::uint8_t> solid;
        if (r->ghost && o.wallValue) {
            solid.assign(r->g.size(), 0);
            for (int c : r->ghost->solidCells()) solid[std::size_t(c)] = 1;
        }
        if (!r->dev->enableRans({.model = o.model,
                                 .wallDist = d,
                                 .mutHook =
                                     [p](gpu::Real* nut, gpu::Real* f1) {
                                         if (p->ghost) p->ghost->mirrorMut(nut, f1);
                                         p->far->applyMut(nut, f1);
                                     },
                                 .fluidMask = r->fluid,
                                 .transportOnly = o.transportOnly,
                                 .bnd = bnd,
                                 .solidMask = std::move(solid)}))
            return nullptr;
    }
    p->bc = [p](gpu::DeviceCompState& s) {
        p->far->apply(s);
        if (!p->ghost) return;
        p->ghost->apply(s);
        if (p->dev->rans())
            p->ghost->applyTurb(s, p->model, p->tr, p->phi[1]);
    };
    p->dev->setStageHook(p->bc);
    p->dev->setRhsHook([p](const gpu::DeviceCompState& s,
                           gpu::DeviceCompState& out,
                           const gpu::StageCombine& cmb) {
        p->nsc->apply(*p->dev, s, out, cmb, p->remap);
    });
    return r;
}

// Freestream plus a smooth vortical/thermal perturbation (so the flow is
// non-uniform everywhere), scalars at rho * phi.
void upload(Rig& r, double phiIc0, double phiIc1)
{
    auto h = CompState::zeros(r.g);
    const double L = r.g.nx * r.g.hx;
    for (int j = 0; j < r.g.ny; ++j)
        for (int i = 0; i < r.g.nx; ++i) {
            const std::size_t c = r.g.idx(i, j, 0);
            const double x = (i + 0.5) / r.g.nx, y = (j + 0.5) / r.g.ny;
            const double rho = kRhoInf * (1 + 0.02 * std::sin(2 * M_PI * x) *
                                                  std::cos(2 * M_PI * y));
            const double u = kUInf * (1 + 0.05 * std::sin(2 * M_PI * y));
            const double v = 0.05 * kUInf * std::sin(2 * M_PI * x);
            h.rho[c] = rho;
            h.mx[c] = rho * u;
            h.my[c] = rho * v;
            h.E[c] = kPInf / (kGam - 1) + 0.5 * rho * (u * u + v * v);
            (void)L;
        }
    (void)r.ds->upload(h);
    for (int i = 0; i < r.ds->nT; ++i) {
        std::vector<double> t(r.g.size());
        for (std::size_t c = 0; c < t.size(); ++c)
            t[c] = h.rho[c] * (i ? phiIc1 : phiIc0);
        (void)r.ds->uploadTurb(i, t);
    }
    r.bc(*r.ds);
}

std::vector<float> down(const gpu::DeviceScalar& b)
{
    std::vector<float> h(b.size());
    (void)b.download(h.data());
    return h;
}

bool same(const gpu::DeviceScalar& a, const gpu::DeviceScalar& b)
{
    const auto x = down(a), y = down(b);
    return std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

// (a)
void laminarLimit()
{
    auto off = makeRig({});
    auto sa = makeRig({.model = RansModel::SaNeg, .rans = true});
    auto sst = makeRig({.model = RansModel::Sst2003m, .rans = true,
                        .phi = {0.0, 10.0}});
    if (!off || !sa || !sst) { check(false, "(a) rigs"); return; }
    for (Rig* r : {off.get(), sa.get(), sst.get()}) upload(*r, 0.0, 10.0);
    const double dt = 0.9 * off->dev->stableDt(*off->ds, *off->red, 0.4);
    for (int s = 0; s < 50; ++s)
        for (Rig* r : {off.get(), sa.get(), sst.get()}) {
            r->dev->step(*r->ds, gpu::Real(dt));
            r->bc(*r->ds);
        }
    for (Rig* r : {sa.get(), sst.get()}) {
        const std::string n = r == sa.get() ? "SA nu~ == 0" : "SST k == 0";
        check(same(off->ds->rho, r->ds->rho) && same(off->ds->mx, r->ds->mx) &&
                  same(off->ds->my, r->ds->my) && same(off->ds->E, r->ds->E),
              "(a) " + n + ": focal + ghost + NSCBC mean flow == RANS off bit "
                           "for bit (50 steps)");
        const auto F0 = off->ghost->force(*off->dev, *off->ds);
        const auto F1 = r->ghost->force(*r->dev, *r->ds);
        check(F0.fx == F1.fx && F0.fy == F1.fy,
              "(a) " + n + ": the body tally == RANS off bit for bit");
        const auto t = down(r->ds->t[0]);
        check(std::ranges::all_of(t, [](float v) { return v == 0.0f; }),
              "(a) " + n + ": the scalar stays exactly 0 (ghost images too)");
    }
}

// (b)
void uniformTransport()
{
    const double phi = 0x1p-13;
    auto r = makeRig({.model = RansModel::SaNeg, .rans = true, .body = false,
                      .focal = true, .transportOnly = true, .phi = {phi, 0}});
    if (!r) { check(false, "(b) rig"); return; }
    upload(*r, phi, 0);
    const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
    for (int s = 0; s < 100; ++s) {
        r->dev->step(*r->ds, gpu::Real(dt));
        r->bc(*r->ds);
    }
    const auto rho = down(r->ds->rho), t = down(r->ds->t[0]);
    const float ref = float(phi);
    double interior = 0, edge = 0;
    for (int j = 1; j < r->g.ny - 1; ++j)
        for (int i = 0; i < r->g.nx; ++i) {
            const std::size_t c = r->g.idx(i, j, 0);
            const float q = t[c] / rho[c];
            const double ulps = std::abs(double(q) - ref) /
                                double(std::nextafter(ref, 1.0f) - ref);
            // The x faces' drift is carried downstream at u: after 100 steps
            // it has reached column 17 (measured), so the interior is
            // counted from column 24 on.
            const bool band = i < 24 || i > r->g.nx - 25;
            if (band) edge = std::max(edge, ulps);
            else interior = std::max(interior, ulps);
        }
    std::printf("  (b) phi/phi_ref - 1 after 100 steps: interior %.1f ulp, "
                "x-face band %.3g ulp\n",
                interior, edge);
    check(interior == 0.0, "(b) a uniform specific scalar rides the density "
                           "EXACTLY in the interior (0 ulp)");
    // GATE G3-flow's boundary clause (<= 2 ulp). The NSCBC rows overwrite
    // the continuity rhs with the characteristic form, which no scalar FLUX
    // can match (the plan's open-face flux rho_c u_c . S alone drifted phi by
    // 1.36e4 ulp, 8e-4 relative, in 100 steps). The open-face cells now carry
    // the scalar by the density the mean flow produced (the phi R_rho form,
    // rans::openScalarConsistency): measured 0 ulp (gfx1100, 2026-10-26).
    check(edge <= 2.0, "(b) the NSCBC x-face band rides the density to <= 2 "
                       "ulp (the phi R_rho form; 1.36e4 ulp before)");
}

// (c) the planar wall, read back on the host.
void wallLinear()
{
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m}) {
        const double L = 2 * std::numbers::pi;
        const int n = 64;
        const Grid g(n, n, 1, L / n, L / n, 1.0);
        const double th = std::numbers::pi / 6, nx = std::cos(th),
                     ny = std::sin(th);
        const double x0 = 0.5 * L + 0.37 * g.hx;
        const Sdf plane = [=](double x, double y, double) {
            return (x - x0) * nx + (y - 0.5 * L) * ny;
        };
        auto mtr = CurvMetrics::build(CurvMesh::uniform(g));
        const auto tr = Transport::air(0.01, kTInf, 300.0, 0.72, kGam, kRgas,
                                       false);
        auto dev = gpu::DeviceComp::make(*mtr, tr);
        auto ds = gpu::DeviceCompState::make(g.size(), ransScalars(m));
        auto gh = gpu::DeviceCompGhost::make(g, plane, kGam, kRhoInf, kPInf);
        if (!mtr || !dev || !ds || !gh) { check(false, "(c) setup"); return; }
        const AxisRemap3 id = focalRemap(g, false, nullptr, nullptr, nullptr, 1);
        const auto d = immersedWallDistance(g, plane, id);
        if (!dev->enableRans({.model = m, .wallDist = d})) {
            check(false, "(c) enableRans");
            return;
        }
        // Uniform gas at rest; nu~ = c sdf (SA: the linear sublayer) or
        // k = c sdf^2 (SST: the quadratic one), omega uniform on fluid.
        const bool sstK = m == RansModel::Sst2003m;
        const double cc = 0.3, om = 50.0;
        auto h = CompState::zeros(g);
        std::vector<double> t0(g.size()), t1(g.size(), om);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                h.rho[c] = 1.3;
                h.E[c] = kPInf / (kGam - 1);
                const double s = plane((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0);
                t0[c] = 1.3 * cc * (sstK ? s * s : s);
                t1[c] = 1.3 * om;
            }
        (void)ds->upload(h);
        (void)ds->uploadTurb(0, t0);
        if (ds->nT > 1) (void)ds->uploadTurb(1, t1);
        gh->apply(*ds);
        gh->applyTurb(*ds, m, tr, 1.0);
        std::vector<double> g0, g1;
        (void)ds->downloadTurb(0, g0);
        if (ds->nT > 1) (void)ds->downloadTurb(1, g1);
        const auto rho = down(ds->rho);
        // Every solid cell the fluid reads is a ghost; check those within one
        // cell of the wall (deep cells hold 0).
        double err = 0, scale = cc * g.hx, kmin = 1e30, kerr = 0;
        int checked = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double s = plane((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0);
                if (s >= 0 || s < -1.5 * g.hx) continue;
                if (i < 3 || i > n - 4 || j < 3 || j > n - 4) continue;
                const double v = g0[c] / rho[c];
                err = std::max(err, std::abs(v - cc * s));
                kmin = std::min(kmin, v);
                // k_g - c sdf_g^2 = a^2 c E, E >= 0 the sample's bilinear
                // interpolation error of sdf^2 (<= h^2 / 4 per axis).
                kerr = std::max(kerr, std::abs(v - cc * s * s) / (cc * g.hx * g.hx));
                ++checked;
            }
        const std::string nm = m == RansModel::SaNeg ? "SA nu~" : "SST k";
        if (!sstK) {
            std::printf("  (c) %s: %d ghosts, max |phi_g - c sdf| / (c h) = %.3e\n",
                        nm.c_str(), checked, err / scale);
            check(checked > 20 && err <= 1e-5 * scale,
                  "(c) " + nm + " = c d is reproduced on every ghost (a linear "
                                "image of a linear field)");
        } else {
            std::printf("  (c) %s: %d ghosts, min k_g %.3e, max |k_g - c sdf^2| / "
                        "(c h^2) = %.3e\n",
                        nm.c_str(), checked, kmin, kerr);
            check(checked > 20 && kmin >= 0,
                  "(c) SST k_g = a^2 sum w k_s >= 0 on every ghost (the even, "
                  "quadratic sublayer image: never negative)");
            check(kerr <= 0.5, "(c) SST k = c d^2 reproduced on every ghost to "
                               "the sample's interpolation error (<= c h^2/2)");
        }
        // mu_t through mirrorMut: the same image, the wall face ~ 0.
        auto nut = gpu::DeviceBuffer<gpu::Real>::make(g.size());
        std::vector<gpu::Real> hn(g.size());
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                hn[g.idx(i, j, 0)] = gpu::Real(
                    cc * plane((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0));
        (void)nut->upload(hn.data());
        gh->mirrorMut(nut->data(), nullptr);
        const auto mn = down(*nut);
        // The image is clamped by the ghost's fluid face neighbours (a ghost
        // deeper than its nearest fluid neighbour is far would make that face
        // mu_t negative): c sdf where it is not, -min(nbr) where it is.
        const auto tb = gpu::classifyCompGhost(g, plane, nullptr, id, nullptr);
        std::vector<double> clampAt(g.size(), -1e30);
        for (int q = 0; q < tb.nGhost(); ++q)
            for (int s = 0; s < gpu::CompGhostTables::kFaceNbr; ++s) {
                const int f = tb.gfnb[gpu::CompGhostTables::kFaceNbr * q + s];
                if (f < 0) break;
                double& v = clampAt[std::size_t(tb.gcell[q])];
                v = std::max(v, -double(hn[f]));
            }
        double emu = 0;
        int bound = 0, free = 0;
        for (int j = 3; j < n - 3; ++j)
            for (int i = 3; i < n - 3; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double s = plane((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0);
                if (s >= 0 || s < -1.5 * g.hx) continue;
                const bool b = clampAt[c] > cc * s;
                (b ? bound : free) += 1;
                emu = std::max(emu, std::abs(mn[c] - (b ? clampAt[c] : cc * s)));
            }
        std::printf("  (c) %s mu_t image: %d ghosts reproduce c d, %d clamped to "
                    "-min(nbr)\n",
                    nm.c_str(), free, bound);
        check(emu <= 1e-5 * scale && bound > 0 && free > 0,
              "(c) " + nm + ": mu_t_g = -min(a sum w mu_t_s, min fluid face-"
                            "neighbour mu_t) -- c d where unclamped");
        if (m != RansModel::Sst2003m) continue;
        // omega_g: (1 + a) omega_w - a omega with omega_w = 60 nu/(b1 d_s^2):
        // every ghost holds a value above omega_w (omega_w >> omega here).
        double minRatio = 1e30;
        for (int j = 3; j < n - 3; ++j)
            for (int i = 3; i < n - 3; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double s = plane((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0);
                if (s >= 0 || s < -1.5 * g.hx) continue;
                minRatio = std::min(minRatio, g1[c] / rho[c] / om);
            }
        check(minRatio > 1.0 && std::isfinite(minRatio),
              "(c) SST omega_g is the omega_w-anchored image (finite, above "
              "the interior omega next to the wall)");
    }
}

// (d)
void smoke()
{
    for (const bool wv : {false, true})
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m}) {
        const double nu = kRhoInf * kUInf * (0.25 * 2 * std::numbers::pi) /
                          1000.0;
        double phi[2] = {nuTildaInf(kRansNutInfRatio, nu), 0};
        if (ransSst(m)) {
            phi[0] = kInf(kRansKInf, kUInf);
            phi[1] = omegaInf(phi[0], kRansMutInfRatio, nu);
        }
        auto r = makeRig({.model = m, .rans = true, .focal = false,
                          .phi = {phi[0], phi[1]}, .re = 1000, .n = 128,
                          .wallValue = wv});
        if (!r) { check(false, "(d) rig"); return; }
        upload(*r, phi[0], phi[1]);
        for (int s = 0; s < 200; ++s) {
            const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
            r->dev->step(*r->ds, gpu::Real(dt));
            r->bc(*r->ds);
        }
        r->dev->updateMut(*r->ds);
        const auto rho = down(r->ds->rho), t = down(r->ds->t[0]);
        const auto mut = down(r->dev->mut());
        bool finite = true;
        for (std::size_t c = 0; c < rho.size(); ++c)
            finite = finite && std::isfinite(rho[c]) && std::isfinite(t[c]);
        bool deepZero = true;
        for (int c : r->ghost->deepCells()) deepZero = deepZero && mut[c] == 0;
        const std::string n = m == RansModel::SaNeg ? "SA" : "SST";
        const auto F = r->ghost->force(*r->dev, *r->ds);
        std::printf("  (d) %s 200 steps (wall value %d): cd-like fx %.4f, hits %llu\n",
                    n.c_str(), int(wv), F.fx, r->dev->floorHits());
        check(finite && std::isfinite(F.fx), "(d) " + n + " 200-step immersed "
                                                          "URANS is finite");
        // 0 hits for both, with and without the wall-value convection. SST
        // used to trip the COUNTED k guard here (710 hits in 200 steps, 4
        // staircase-corner cells) through the ODD ghost image k_g = -a k_s
        // < 0; the even quadratic image k_g = a^2 k_s removes it alone (and
        // the wall-value convection keeps a ghost value from being carried
        // through an interface face at all).
        check(r->dev->floorHits() == 0,
              "(d) " + n + (wv ? "" : " (no wall-value convection)") +
                  ": 0 guard hits");
        check(deepZero, "(d) " + n + ": mu_t == 0 on the deep cells");
    }
}

// (e) the bounded mu_t image on an under-resolved immersed URANS (128^2,
// D = 32 cells, Re 1e5, y+ ~ 30-80, 200 steps). On every staircase face
// (fluid c | ghost g) the face viscosity mu + 0.5 (mu_t_g + mu_t_c) must stay
// >= mu. The unclamped odd image -a sum w mu_t_s is recomputed on the host
// from the same tables and must drive some face NEGATIVE here -- the rig
// exercises the clamp -- and every ghost must be -min(image, min over its
// fluid face neighbours' mu_t).
void boundedMut()
{
    const int dijk[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                            {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m}) {
        const double re = 1e5;
        const double mu = kRhoInf * kUInf * (0.25 * 2 * std::numbers::pi) / re;
        double phi[2] = {nuTildaInf(kRansNutInfRatio, mu / kRhoInf), 0};
        if (ransSst(m)) {
            phi[0] = kInf(kRansKInf, kUInf);
            phi[1] = omegaInf(phi[0], kRansMutInfRatio, mu / kRhoInf);
        }
        auto r = makeRig({.model = m, .rans = true, .focal = false,
                          .phi = {phi[0], phi[1]}, .re = re, .n = 128});
        if (!r) { check(false, "(e) rig"); return; }
        upload(*r, phi[0], phi[1]);
        // A developed wall layer, so the image sees a law-of-the-wall mu_t
        // from the first step (from a freestream start 200 steps grow next to
        // none: 56 SA / 51 SST faces below mu, none negative, measured):
        // nu_t = kappa u_tau d e^(-d/2delta), u_tau = 0.38 Re^-0.1 U (the
        // validator's bluff-body estimate), delta = D/20. SA: nu~ = that
        // (fv1 ~ 1 at chi ~ kappa y+); SST: the log-layer equilibrium pair
        // k = u_tau^2/sqrt(beta*) e^(-d/delta), omega = sqrt(k)/(beta*^1/4
        // kappa d). Freestream values are floors.
        const auto rho = down(r->ds->rho);
        const double ut = 0.38 * std::pow(re, -0.1) * kUInf, kap = 0.41;
        const double delta = 0.05 * 0.25 * 2 * std::numbers::pi;
        std::vector<double> t0(r->g.size()), t1(r->g.size());
        const Sdf sdf = rigCircle();
        for (int j = 0; j < r->g.ny; ++j)
            for (int i = 0; i < r->g.nx; ++i) {
                const std::size_t c = r->g.idx(i, j, 0);
                const double d = std::max(
                    sdf((i + 0.5) * r->g.hx, (j + 0.5) * r->g.hy, 0), 0.0);
                const double k = ut * ut / 0.3 * std::exp(-d / delta);
                const double nut = kap * ut * d * std::exp(-0.5 * d / delta);
                const double om = std::sqrt(k) / (std::sqrt(0.3) * kap *
                                                  std::max(d, 1e-3 * r->g.hx));
                t0[c] = rho[c] * std::max(ransSst(m) ? k : nut, phi[0]);
                t1[c] = rho[c] * std::max(om, phi[1]);
            }
        (void)r->ds->uploadTurb(0, t0);
        if (ransSst(m)) (void)r->ds->uploadTurb(1, t1);
        r->bc(*r->ds);
        for (int s = 0; s < 200; ++s) {
            const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
            r->dev->step(*r->ds, gpu::Real(dt));
            r->bc(*r->ds);
        }
        r->dev->updateMut(*r->ds);
        const auto mut = down(r->dev->mut());
        const auto yp = r->ghost->wallYplus(*r->dev, *r->ds);
        const Grid& g = r->g;
        const auto T =
            gpu::classifyCompGhost(g, rigCircle(), nullptr, r->remap, nullptr);
        const std::string n = m == RansModel::SaNeg ? "SA" : "SST";
        check(T.solid == r->ghost->solidCells(),
              "(e) " + n + ": the host tables are the solver's");
        std::vector<int> ghostOf(g.size(), -1);
        std::vector<double> img(std::size_t(T.nGhost())), lim;
        for (int q = 0; q < T.nGhost(); ++q) {
            ghostOf[std::size_t(T.gcell[q])] = q;
            double s = 0;
            for (int k = 0; k < 8; ++k)
                s += double(T.gw[8 * q + k]) * mut[T.gnbr[8 * q + k]];
            img[q] = double(T.gscale[q]) * s;
        }
        lim = img;
        int below = 0, negative = 0, wouldBelow = 0, wouldNeg = 0;
        double minTot = 1e30;
        for (int f = 0; f < T.nFace(); ++f) {
            const int c = T.fcell[f], d = T.fdir[f];
            const int i = c % g.nx, j = (c / g.nx) % g.ny, k = c / (g.nx * g.ny);
            const std::size_t gc =
                g.idx(i + dijk[d][0], j + dijk[d][1], k + dijk[d][2]);
            const int q = ghostOf[gc];
            if (q < 0) { check(false, "(e) face without a ghost"); return; }
            lim[q] = std::min(lim[q], double(mut[c]));
            const double tot = mu + 0.5 * (double(mut[gc]) + mut[c]);
            const double would = mu + 0.5 * (-img[q] + mut[c]);
            below += tot < mu;
            negative += tot < 0;
            wouldBelow += would < mu;
            wouldNeg += would < 0;
            minTot = std::min(minTot, tot / mu);
        }
        double eLim = 0;
        for (int q = 0; q < T.nGhost(); ++q)
            eLim = std::max(eLim, std::abs(mut[T.gcell[q]] + lim[q])
                                      / (std::abs(lim[q]) + 1e-3 * mu));
        const double dtEnd = r->dev->stableDt(*r->ds, *r->red, 0.4);
        const auto F = r->ghost->force(*r->dev, *r->ds);
        std::printf("  (e) %s Re 1e5 128^2 200 steps: y+ max %.1f mean %.1f, "
                    "%d faces; unclamped image: %d below mu, %d negative; "
                    "solver: %d below mu, %d negative, min (mu+mu_t)/mu %.4f, "
                    "max |mu_t_g + min(img, nbr)| rel %.2e, fx %.4f, dt %.3g, "
                    "hits %llu\n",
                    n.c_str(), yp.max, yp.mean, T.nFace(), wouldBelow, wouldNeg,
                    below, negative, minTot, eLim, F.fx, dtEnd,
                    r->dev->floorHits());
        // Unclamped, this rig blew up: SST's dt fell to 1.8e-9 by step 200
        // (SA ran on with faces down to -39 mu); clamped both hold dt ~2e-3.
        check(std::isfinite(F.fx) && dtEnd > 1e-4,
              "(e) " + n + ": the run is finite with a healthy dt");
        check(wouldNeg > 0, "(e) " + n + ": the unclamped odd image makes some "
                                         "face's total viscosity NEGATIVE here");
        check(below == 0 && negative == 0,
              "(e) " + n + ": every staircase face keeps mu + mu_t >= mu");
        check(eLim <= 1e-5, "(e) " + n + ": mu_t_g = -min(a sum w mu_t_s, "
                                         "min fluid face-neighbour mu_t)");
    }
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    laminarLimit();
    uniformTransport();
    wallLinear();
    smoke();
    boundedMut();
    return dnstest::finish();
}
