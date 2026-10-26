// RANS on the body-fitted O-grid (masterplan_flowrans P1, docs/rans.md
// "dns flow"):
//   (a) the RANS tables (d1, omega_w) on a RANS-off solver change no bit
//       (the byte-identity protocol covers RANS off vs HEAD end to end)
//   (b) SA nu~ == 0 (nu~_inf = 0) and SST k == 0: the mean flow is RANS off
//       bit for bit over 50 steps at a FIXED dt; the scalar stays exactly 0
//   (c) the far row is a fixed point of a uniform freestream WITH uniform
//       scalars: the BC alone to 4 ulp, and the outer rows after 100 steps
//   (d) wall face mu_t == 0.0 exactly; the far ghost mu_t/F1 an even copy;
//       the wall face holds omega_w = 60 nu_f/(beta1 d1^2)
//   (e) G3-ogrid: a uniform specific scalar 2^-13 rides the density exactly
//       (transportOnly) everywhere the far blend cannot reach
//   (f) 200-step SA and SST URANS smoke on 256 x 96: finite, 0 guard hits
//   (g) the compensated LTS march: the low words of rows 0 and ny-1 are 0
//       after every apply
// Seconds.
#include "dns/bodies.hpp"
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/rans.hpp"
#include "dns/walldist.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/reductions.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace dns;
using dnstest::check;

namespace {

constexpr double kGam = 1.4, kRgas = 1.0, kMach = 0.2;
constexpr double kRhoInf = 1.0, kUInf = 1.0;
const double kPInf = kRhoInf * kUInf * kUInf / (kGam * kMach * kMach);
const double kTInf = kPInf / (kRhoInf * kRgas);
constexpr double kD = 0.785398163397448, kRFar = 20 * kD;

struct Rig
{
    Grid g{1, 1, 1, 1, 1, 1};
    std::unique_ptr<OGrid> og;
    std::unique_ptr<CurvMetrics> m;
    std::vector<double> d;
    std::unique_ptr<gpu::DeviceComp> dev;
    std::unique_ptr<gpu::DeviceCompState> ds;
    std::unique_ptr<gpu::DeviceOGridBc> bc;
    std::unique_ptr<gpu::ReduceScratch> red;
    double mu = 0;
    double phi[2] = {0, 0};
};

struct Opt
{
    RansModel model = RansModel::None;
    double re = 40;
    int nx = 128, ny = 64;
    double phi[2] = {-1, -1};   // < 0: the validator defaults
    bool transportOnly = false;
    bool compensated = false;
    bool tables = true;         // hand the BC d1 even without RANS
};

std::unique_ptr<Rig> makeRig(const Opt& o)
{
    auto r = std::make_unique<Rig>();
    r->g = ogridGrid(o.nx, o.ny, 1, 0.5 * kD, kRFar, 1.0);
    const Sdf circle = [](double x, double y, double) {
        return std::hypot(x, y) - 0.5 * kD;
    };
    auto og = buildOGrid(r->g, circle, {.rFar = kRFar, .beta = 3.0, .span = 1.0});
    if (!og) return nullptr;
    r->og = std::make_unique<OGrid>(std::move(*og));
    auto m = CurvMetrics::build(r->og->mesh);
    if (!m) return nullptr;
    r->m = std::make_unique<CurvMetrics>(std::move(*m));
    r->d = ogridWallDistance(r->og->mesh);
    r->mu = kRhoInf * kUInf * kD / o.re;
    const double nu = r->mu / kRhoInf;
    if (o.model == RansModel::SaNeg)
        r->phi[0] = o.phi[0] >= 0 ? o.phi[0] : nuTildaInf(kRansNutInfRatio, nu);
    if (ransSst(o.model)) {
        r->phi[0] = o.phi[0] >= 0 ? o.phi[0] : kInf(kRansKInf, kUInf);
        r->phi[1] = o.phi[1] >= 0 ? o.phi[1]
                                  : omegaInf(kInf(kRansKInf, kUInf),
                                             kRansMutInfRatio, nu);
    }
    const auto tr = Transport::air(r->mu, kTInf, 300.0, 0.72, kGam, kRgas, false);
    auto dev = gpu::DeviceComp::make(*r->m, tr);
    auto ds = gpu::DeviceCompState::make(r->g.size(), ransScalars(o.model),
                                         o.compensated);
    gpu::DeviceOGridBc::Config bcc{.gamma = kGam, .Rgas = kRgas,
                                   .rhoInf = kRhoInf, .uInf = kUInf,
                                   .pInf = kPInf};
    bcc.tr = tr;
    if (o.model != RansModel::None || o.tables)
        bcc.wallD1 = wallFirstCellDistance(r->g, r->d);
    bcc.rans = o.model;
    bcc.phiInf[0] = r->phi[0];
    bcc.phiInf[1] = r->phi[1];
    auto bc = gpu::DeviceOGridBc::make(r->g, bcc);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !bc || !red) return nullptr;
    r->dev = std::make_unique<gpu::DeviceComp>(std::move(*dev));
    r->ds = std::make_unique<gpu::DeviceCompState>(std::move(*ds));
    r->bc = std::make_unique<gpu::DeviceOGridBc>(std::move(*bc));
    r->red = std::make_unique<gpu::ReduceScratch>(std::move(*red));
    const auto fluid = gpu::rowMask(r->g, 1, r->g.ny - 2);
    if (o.model != RansModel::None &&
        !r->dev->enableRans({.model = o.model,
                             .wallDist = r->d,
                             .mutHook = r->bc->mutHook(),
                             .fluidMask = fluid,
                             .transportOnly = o.transportOnly}))
        return nullptr;
    if (o.compensated && o.model == RansModel::None &&
        !r->dev->enableSteady(fluid))
        return nullptr;
    if (o.compensated && !r->dev->enableCompensation()) return nullptr;
    Rig* p = r.get();
    p->dev->setStageHook([p](gpu::DeviceCompState& s) { p->bc->apply(s); });
    return r;
}

// The impulsive freestream with every scalar at its freestream value.
void uploadFreestream(Rig& r)
{
    auto h = CompState::zeros(r.g);
    for (std::size_t c = 0; c < r.g.size(); ++c) {
        h.rho[c] = kRhoInf;
        h.mx[c] = kRhoInf * kUInf;
        h.E[c] = kPInf / (kGam - 1.0) + 0.5 * kRhoInf * kUInf * kUInf;
    }
    (void)r.ds->upload(h);
    for (int i = 0; i < r.ds->nT; ++i)
        (void)r.ds->uploadTurb(i, std::vector<double>(r.g.size(),
                                                      kRhoInf * r.phi[i]));
    r.bc->apply(*r.ds);
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
    return x.size() == y.size() &&
           std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

bool sameMean(const gpu::DeviceCompState& a, const gpu::DeviceCompState& b)
{
    return same(a.rho, b.rho) && same(a.mx, b.mx) && same(a.my, b.my) &&
           same(a.mz, b.mz) && same(a.E, b.E);
}

// (a) and (b).
void laminarLimits()
{
    auto off = makeRig({.tables = false});
    auto tab = makeRig({.tables = true});
    auto sa = makeRig({.model = RansModel::SaNeg, .phi = {0, 0}});
    auto sst = makeRig({.model = RansModel::Sst2003m, .phi = {0, -1}});
    if (!off || !tab || !sa || !sst) { check(false, "(a,b) rigs"); return; }
    for (Rig* r : {off.get(), tab.get(), sa.get(), sst.get()}) uploadFreestream(*r);
    const double dt = 0.9 * off->dev->stableDt(*off->ds, *off->red, 0.4);
    for (int s = 0; s < 50; ++s)
        for (Rig* r : {off.get(), tab.get(), sa.get(), sst.get()}) {
            r->dev->step(*r->ds, gpu::Real(dt));
            r->bc->apply(*r->ds);
        }
    check(sameMean(*off->ds, *tab->ds),
          "(a) d1/omega_w tables on a RANS-off solver change no bit (50 steps)");
    check(sameMean(*off->ds, *sa->ds),
          "(b) SA nu~ == 0: mean flow == RANS off bit for bit (50 steps)");
    check(sameMean(*off->ds, *sst->ds),
          "(b) SST k == 0: mean flow == RANS off bit for bit (50 steps)");
    const auto t = down(sa->ds->t[0]);
    const auto k = down(sst->ds->t[0]);
    check(std::ranges::all_of(t, [](float v) { return v == 0.0f; }) &&
              std::ranges::all_of(k, [](float v) { return v == 0.0f; }),
          "(b) nu~ and k stay exactly 0");
}

// (c) the far row's fixed point, scalars included.
void farFixedPoint()
{
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m}) {
        const std::string n(ransName(m));
        auto r = makeRig({.model = m, .transportOnly = m == RansModel::SaNeg});
        if (!r) { check(false, n + " (c) rig"); return; }
        uploadFreestream(*r);
        const Grid& g = r->g;
        auto rel = [&](int i0, int i1, int jLo) {
            double worst = 0;
            const auto rho = down(r->ds->rho);
            for (int i = 0; i < r->ds->nT; ++i) {
                const auto t = down(r->ds->t[i]);
                for (int j = jLo; j < g.ny; ++j)
                    for (int ii = i0; ii < i1; ++ii) {
                        const std::size_t c = g.idx(ii, j, 0);
                        const double ph = double(t[c]) / double(rho[c]);
                        worst = std::max(worst, std::abs(ph - r->phi[i]) /
                                                    (r->phi[i] * 0x1p-23));
                    }
            }
            return worst;
        };
        const double bc0 = rel(0, g.nx, g.ny - 1);
        std::printf("%s (c): far ghost specific scalars after apply: %.2f ulp\n",
                    n.c_str(), bc0);
        check(bc0 <= 4.0, n + " (c): far row reproduces phi_inf (<= 4 ulp)");
        const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
        for (int s = 0; s < 100; ++s) {
            r->dev->step(*r->ds, gpu::Real(dt));
            r->bc->apply(*r->ds);
        }
        // Outside the wall's acoustic cone (t ~ 0.01, the outer cells are
        // ~0.5 D wide): the outer quarter of the annulus.
        const double fp = rel(0, g.nx, 3 * g.ny / 4);
        std::printf("%s (c): outer rows after 100 steps: %.2f ulp\n", n.c_str(),
                    fp);
        // SST's k and omega decay ambiently (beta* omega k, beta omega^2):
        // a physical rate, not the boundary's; SA runs transport only.
        if (m == RansModel::SaNeg)
            check(fp <= 4.0, n + " (c): outer rows stay at phi_inf (<= 4 ulp)");
        else
            check(fp < 1e3, n + " (c): outer rows near phi_inf (ambient decay "
                               "only)");
    }
}

// (d) the wall and far images.
void images()
{
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m}) {
        const std::string n(ransName(m));
        auto r = makeRig({.model = m, .re = 1e4});
        if (!r) { check(false, n + " (d) rig"); return; }
        uploadFreestream(*r);
        const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
        for (int s = 0; s < 20; ++s) {
            r->dev->step(*r->ds, gpu::Real(dt));
            r->bc->apply(*r->ds);
        }
        r->dev->updateMut(*r->ds);
        const Grid& g = r->g;
        const auto mut = down(r->dev->mut());
        const auto f1 = ransSst(m) ? down(r->dev->f1()) : std::vector<float>{};
        const auto rho = down(r->ds->rho);
        const auto t0 = down(r->ds->t[0]);
        const auto t1 = ransSst(m) ? down(r->ds->t[1]) : std::vector<float>{};
        bool face = true, far = true, f1ok = true, kimg = true, wimg = true;
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t w0 = g.idx(i, 0, 0), w1 = g.idx(i, 1, 0);
            const std::size_t fN = g.idx(i, g.ny - 1, 0),
                              fS = g.idx(i, g.ny - 2, 0);
            face = face && 0.5f * (mut[w0] + mut[w1]) == 0.0f;
            far = far && mut[fN] == mut[fS];
            if (!f1.empty()) f1ok = f1ok && f1[w0] == f1[w1] && f1[fN] == f1[fS];
            const double s0 = double(t0[w0]) / rho[w0],
                         s1 = double(t0[w1]) / rho[w1];
            kimg = kimg && std::abs(s0 + s1) <= 3e-7 * std::abs(s1);
            if (!t1.empty()) {
                const double d1 = r->d[w1];
                const double rf = 0.5 * (double(rho[w0]) + double(rho[w1]));
                const double ww = sst::omegaWallFactor * r->mu / (d1 * d1) / rf;
                const double wf = 0.5 * (double(t1[w0]) / rho[w0] +
                                         double(t1[w1]) / rho[w1]);
                wimg = wimg && std::abs(wf - ww) <= 1e-5 * ww;
            }
        }
        check(face, n + " (d): mu_t == 0.0f on every wall face");
        check(far, n + " (d): far ghost mu_t an even copy of row ny-2");
        check(f1ok, n + " (d): F1 even on both rows");
        check(kimg, n + " (d): wall ghost scalar = -specific (FP32 rounding)");
        check(wimg, n + " (d): the wall face holds omega_w = 60 nu_f/(beta1 "
                        "d1^2) per face");
    }
}

// (e) G3-ogrid: transport of a uniform specific scalar.
void uniformTransport()
{
    const double c = 0x1p-13;
    auto r = makeRig({.model = RansModel::SaNeg, .phi = {c, 0},
                      .transportOnly = true});
    if (!r) { check(false, "(e) rig"); return; }
    uploadFreestream(*r);
    const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
    for (int s = 0; s < 100; ++s) {
        r->dev->step(*r->ds, gpu::Real(dt));
        r->bc->apply(*r->ds);
    }
    const Grid& g = r->g;
    const auto rho = down(r->ds->rho), t = down(r->ds->t[0]);
    int jExact = g.ny - 2; // the deepest row the far blend has not reached
    double worstIn = 0;
    for (int j = g.ny - 2; j >= 1; --j) {
        double w = 0;
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t q = g.idx(i, j, 0);
            const float nt = t[q] / rho[q];
            w = std::max(w, std::abs(double(nt) - c) / (c * 0x1p-23));
        }
        if (w > 0) {
            jExact = j - 1;
            break;
        }
        worstIn = std::max(worstIn, w);
    }
    double worstOut = 0;
    for (int j = jExact + 1; j <= g.ny - 2; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t q = g.idx(i, j, 0);
            worstOut = std::max(worstOut,
                                std::abs(double(t[q] / rho[q]) - c) /
                                    (c * 0x1p-23));
        }
    std::printf("(e): uniform nu~ exact (0 ulp) on rows 1..%d of %d; the far "
                "blend's rows within %.2f ulp\n",
                jExact, g.ny - 2, worstOut);
    check(jExact >= (3 * (g.ny - 2)) / 4,
          "(e): a uniform specific scalar rides the density EXACTLY on the "
          "wall-side three quarters (0 ulp)");
    check(worstOut <= 4.0, "(e): the far blend's rows within 4 ulp");
}

// (f) URANS smoke.
void smoke()
{
    for (RansModel m : {RansModel::SaNeg, RansModel::Sst2003m}) {
        const std::string n(ransName(m));
        auto r = makeRig({.model = m, .re = 1e4, .nx = 256, .ny = 96});
        if (!r) { check(false, n + " (f) rig"); return; }
        uploadFreestream(*r);
        double dt = 0;
        for (int s = 0; s < 200; ++s) {
            if (s % 16 == 0) dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
            r->dev->step(*r->ds, gpu::Real(dt));
            r->bc->apply(*r->ds);
        }
        bool finite = true;
        for (auto* b : {&r->ds->rho, &r->ds->mx, &r->ds->E, &r->ds->t[0]})
            for (float v : down(*b)) finite = finite && std::isfinite(v);
        check(finite && std::isfinite(dt) && dt > 0,
              n + " (f): 200 URANS steps on 256 x 96 stay finite");
        check(r->dev->floorHits() == 0, n + " (f): 0 guard hits");
    }
}

// (g) compensated LTS.
void compensatedRows()
{
    for (RansModel m : {RansModel::None, RansModel::SaNeg, RansModel::Sst2003m}) {
        const std::string n = m == RansModel::None ? "laminar"
                                                   : std::string(ransName(m));
        auto r = makeRig({.model = m, .compensated = true});
        if (!r) { check(false, n + " (g) rig"); return; }
        uploadFreestream(*r);
        r->dev->setLts(true);
        double dt = 0;
        bool zero = true;
        for (int s = 0; s < 64; ++s) {
            if (s % 16 == 0) dt = 0.95 * r->dev->localDt(*r->ds, *r->red, 0.4);
            r->dev->step(*r->ds, gpu::Real(dt));
            r->bc->apply(*r->ds);
            const Grid& g = r->g;
            std::vector<const gpu::DeviceScalar*> lo;
            for (auto& l : r->ds->lo) lo.push_back(&l);
            for (int i = 0; i < r->ds->nT; ++i) lo.push_back(&r->ds->tlo[i]);
            for (const auto* l : lo) {
                const auto h = down(*l);
                for (int i = 0; i < g.nx; ++i)
                    zero = zero && h[g.idx(i, 0, 0)] == 0.0f &&
                           h[g.idx(i, g.ny - 1, 0)] == 0.0f;
            }
        }
        const auto n0 = r->dev->residual(*r->ds, *r->red);
        check(zero, n + " (g): low words of rows 0 and ny-1 are 0 after every "
                        "apply (64 compensated LTS steps)");
        check(std::isfinite(n0.mom), n + " (g): the compensated march is finite");
    }
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    laminarLimits();
    farFixedPoint();
    images();
    uniformTransport();
    smoke();
    compensatedRows();
    return dnstest::finish();
}
