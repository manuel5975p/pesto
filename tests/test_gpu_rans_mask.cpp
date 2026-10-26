// The P0 generalization of the RANS plumbing (masterplan_flowrans section 3.2)
// is the channel's arithmetic, bit for bit:
//   M1  residual over the fluid mask == residual over the row range, bitwise
//       (every norm), and residualInto() writes exactly residualField()
//   M2  the channel mutHook writes mu_t odd (wall face exactly 0.0) and F1
//       even across both ghost rows
//   M3  the SST guard counts only masked cells: 0 hits with the channel mask,
//       > 0 with every cell counted (the antimirrored ghost k is negative)
//   M4  testSource enters R exactly (one FP32 add) and touches nothing else
//   M5  laminar --steady: localDt without RANS equals SST's at k == 0 bitwise
//       (mu_t = 0 there), and the residual/compensated LTS march run at nT 0
// Seconds.
#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_channel.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/rans_channel1d.hpp"
#include "dns/walldist.hpp"
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

constexpr double kGamma = 1.4;

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
};

// The channel runner's wiring. maskAll: an EMPTY fluid mask (every cell
// counted) instead of the channel's rows. compensated: the steady state.
std::unique_ptr<Rig> makeRig(const Channel1dCase& cc, RansModel model,
                             bool maskAll = false, bool compensated = false,
                             const gpu::Real* const* src = nullptr)
{
    auto r = std::make_unique<Rig>();
    r->g = channel1dGrid(cc, 4, 1, 4.0, 1.0);
    r->gg = gpu::GpuGrid::from(r->g);
    const auto mesh = CurvMesh::channelY(r->g, cc.beta, 1.0);
    auto m = CurvMetrics::build(mesh);
    if (!m) return nullptr;
    r->m = std::make_unique<CurvMetrics>(std::move(*m));
    const double mu = 1.0 / cc.reTau;
    r->d = channelWallDistance(mesh, 1.0);
    const double d1 = r->d[r->g.idx(0, 1, 0)];
    auto dev = gpu::DeviceComp::make(*r->m, kGamma, mu);
    auto ds = gpu::DeviceCompState::make(r->g.size(), ransScalars(model),
                                         compensated);
    auto wall = gpu::DeviceChannelWall::make(
        r->g, {kGamma, 1.0, channelPRef(cc.reTau, cc.mach), true,
               ransSst(model) ? sst::omegaWallFactor * mu / (d1 * d1) : 0.0});
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !wall || !red) return nullptr;
    r->dev = std::make_unique<gpu::DeviceComp>(std::move(*dev));
    r->ds = std::make_unique<gpu::DeviceCompState>(std::move(*ds));
    r->wall = std::make_unique<gpu::DeviceChannelWall>(std::move(*wall));
    r->red = std::make_unique<gpu::ReduceScratch>(std::move(*red));
    const auto mask = maskAll ? std::vector<std::uint8_t>{}
                              : gpu::rowMask(r->g, 1, r->g.ny - 2);
    if (model != RansModel::None) {
        gpu::RansSetup rs{.model = model,
                          .wallDist = r->d,
                          .mutHook = gpu::mutMirrorYHook(r->gg, true, true),
                          .fluidMask = mask};
        if (src) {
            rs.testSource[0] = src[0];
            rs.testSource[1] = src[1];
        }
        if (!r->dev->enableRans(rs)) return nullptr;
    } else if (!r->dev->enableSteady(mask)) {
        return nullptr;
    }
    if (compensated && !r->dev->enableCompensation()) return nullptr;
    Rig* p = r.get();
    p->dev->setStageHook([p](gpu::DeviceCompState& s) { p->wall->apply(s); });
    p->dev->setRhsHook([p](const gpu::DeviceCompState&, gpu::DeviceCompState& o,
                           const gpu::StageCombine& cmb) {
        gpu::addBodyForceX(p->gg, 1.0, cmb, p->dev->u(), o);
    });
    return r;
}

// The oracle state, 5% off in u and 30% off in the scalars: a residual well
// above the FP32 floor in every equation.
void uploadPerturbed(Rig& r, const Channel1dSolution& sol)
{
    auto h = toCompState(sol, r.g);
    for (std::size_t q = 0; q < h.rho.size(); ++q) {
        const double u = h.mx[q] / h.rho[q];
        const double e = h.E[q] - 0.5 * h.rho[q] * u * u;
        h.mx[q] = h.rho[q] * 0.95 * u;
        h.E[q] = e + 0.5 * h.rho[q] * 0.9025 * u * u;
    }
    (void)r.ds->upload(h);
    for (int i = 0; i < r.ds->nT; ++i) {
        auto t = toTurb(sol, r.g, i);
        for (double& v : t) v *= i == 0 ? 0.7 : 1.3;
        (void)r.ds->uploadTurb(i, t);
    }
    r.wall->apply(*r.ds);
}

bool sameNorms(const gpu::ResidualNorms& a, const gpu::ResidualNorms& b)
{
    return std::memcmp(&a, &b, sizeof a) == 0;
}

std::vector<float> down(const gpu::DeviceScalar& b)
{
    std::vector<float> h(b.size());
    (void)b.download(h.data());
    return h;
}

bool sameBuf(const gpu::DeviceScalar& a, const gpu::DeviceScalar& b)
{
    const auto x = down(a), y = down(b);
    return x.size() == y.size() &&
           std::memcmp(x.data(), y.data(), x.size() * sizeof(float)) == 0;
}

void maskedResidual(RansModel model)
{
    const std::string n(ransName(model));
    Channel1dCase cc;
    cc.model = model;
    auto sol = solveChannel1d(cc);
    if (!sol) { check(false, n + " oracle: " + sol.error()); return; }
    auto r = makeRig(cc, model);
    if (!r) { check(false, n + " rig"); return; }
    uploadPerturbed(*r, *sol);
    const auto a = r->dev->residual(*r->ds, *r->red);
    const auto b = r->dev->residual(*r->ds, 1, r->g.ny - 2, *r->red);
    std::printf("%s M1: mask %.6e %.6e %.6e %.6e %.6e\n", n.c_str(), a.rho,
                a.mom, a.E, a.t[0], a.t[1]);
    check(sameNorms(a, b) && a.mom > 0 && a.t[0] > 0,
          n + " M1: masked residual == row-range residual, bitwise");
    auto out = gpu::DeviceCompState::make(r->g.size(), r->ds->nT);
    if (!out) { check(false, n + " M1 alloc"); return; }
    r->dev->residualInto(*r->ds, *out);
    const auto& f = r->dev->residualField();
    bool same = sameBuf(out->rho, f.rho) && sameBuf(out->mx, f.mx) &&
                sameBuf(out->my, f.my) && sameBuf(out->mz, f.mz) &&
                sameBuf(out->E, f.E);
    for (int i = 0; i < r->ds->nT; ++i) same = same && sameBuf(out->t[i], f.t[i]);
    check(same, n + " M1: residualInto == residualField, every array bitwise");

    // M2: the mu_t images.
    r->dev->updateMut(*r->ds);
    const auto mut = down(r->dev->mut());
    const auto f1 = ransSst(model) ? down(r->dev->f1()) : std::vector<float>{};
    bool odd = true, even = true;
    for (int i = 0; i < r->g.nx; ++i)
        for (auto [gj, sj] : {std::pair{0, 1}, {r->g.ny - 1, r->g.ny - 2}}) {
            const std::size_t gq = r->g.idx(i, gj, 0), sq = r->g.idx(i, sj, 0);
            odd = odd && mut[gq] == -mut[sq] && 0.5f * (mut[gq] + mut[sq]) == 0.0f;
            if (!f1.empty()) even = even && f1[gq] == f1[sq];
        }
    check(odd, n + " M2: mutHook writes mu_t odd, wall face exactly 0.0");
    check(even, n + " M2: mutHook writes F1 even");
}

void guardMask()
{
    Channel1dCase cc;
    cc.model = RansModel::Sst2003m;
    auto sol = solveChannel1d(cc);
    if (!sol) { check(false, "M3 oracle"); return; }
    auto a = makeRig(cc, cc.model);
    auto b = makeRig(cc, cc.model, true);
    if (!a || !b) { check(false, "M3 rig"); return; }
    for (Rig* r : {a.get(), b.get()}) {
        uploadPerturbed(*r, *sol);
        const double dt = 0.9 * r->dev->stableDt(*r->ds, *r->red, 0.4);
        for (int i = 0; i < 5; ++i) {
            r->dev->step(*r->ds, gpu::Real(dt));
            r->wall->apply(*r->ds);
        }
    }
    const auto ha = a->dev->floorHits(), hb = b->dev->floorHits();
    std::printf("M3: guard hits, channel mask %llu, every cell %llu\n", ha, hb);
    check(ha == 0, "M3: the channel mask counts 0 hits");
    check(hb > 0, "M3: counting every cell sees the antimirrored ghost k");
    check(sameBuf(a->ds->rho, b->ds->rho) && sameBuf(a->ds->t[0], b->ds->t[0]) &&
              sameBuf(a->ds->t[1], b->ds->t[1]),
          "M3: the mask changes the count only, never the state");
}

void testSourceExact()
{
    Channel1dCase cc;
    cc.model = RansModel::Sst2003m;
    auto sol = solveChannel1d(cc);
    if (!sol) { check(false, "M4 oracle"); return; }
    auto plain = makeRig(cc, cc.model);
    if (!plain) { check(false, "M4 rig"); return; }
    const std::size_t n = plain->g.size();
    std::vector<float> s0(n), s1(n);
    for (std::size_t q = 0; q < n; ++q) {
        s0[q] = float(1e-3 * std::sin(0.37 * double(q)));
        s1[q] = float(-2e-2 * std::cos(0.11 * double(q)));
    }
    auto d0 = gpu::DeviceScalar::make(n);
    auto d1 = gpu::DeviceScalar::make(n);
    if (!d0 || !d1 || d0->upload(s0.data()) || d1->upload(s1.data())) {
        check(false, "M4 source upload");
        return;
    }
    const gpu::Real* src[2] = {d0->data(), d1->data()};
    auto with = makeRig(cc, cc.model, false, false, src);
    if (!with) { check(false, "M4 rig (source)"); return; }
    uploadPerturbed(*plain, *sol);
    uploadPerturbed(*with, *sol);
    (void)plain->dev->residual(*plain->ds, *plain->red);
    (void)with->dev->residual(*with->ds, *with->red);
    const auto& fa = plain->dev->residualField();
    const auto& fb = with->dev->residualField();
    const auto ka = down(fa.t[0]), kb = down(fb.t[0]);
    const auto wa = down(fa.t[1]), wb = down(fb.t[1]);
    bool exact = true;
    for (std::size_t q = 0; q < n; ++q)
        exact = exact && kb[q] == ka[q] + s0[q] && wb[q] == wa[q] + s1[q];
    check(exact, "M4: R_k, R_omega gain exactly the FP32 source add");
    check(sameBuf(fa.rho, fb.rho) && sameBuf(fa.mx, fb.mx) &&
              sameBuf(fa.E, fb.E),
          "M4: the mean-flow rhs is untouched");
}

void laminarSteady()
{
    Channel1dCase cc;
    cc.reTau = 20;
    cc.model = RansModel::Sst2003m;
    auto lam = makeRig(cc, RansModel::None, false, true);
    auto sst = makeRig(cc, RansModel::Sst2003m);
    if (!lam || !sst) { check(false, "M5 rig"); return; }
    cc.model = RansModel::None;
    auto sol = solveChannel1d(cc);
    if (!sol) { check(false, "M5 oracle: " + sol.error()); return; }
    const auto h = toCompState(*sol, lam->g);
    (void)lam->ds->upload(h);
    lam->wall->apply(*lam->ds);
    (void)sst->ds->upload(h);
    (void)sst->ds->uploadTurb(0, std::vector<double>(sst->g.size(), 0.0));
    (void)sst->ds->uploadTurb(1, std::vector<double>(sst->g.size(), 1.0));
    sst->wall->apply(*sst->ds);
    // k == 0: SST's mu_t bound is 0, so both diffusive bounds are molecular.
    const double a = lam->dev->localDt(*lam->ds, *lam->red, 0.4);
    const double b = sst->dev->localDt(*sst->ds, *sst->red, 0.4);
    std::vector<double> ca(lam->g.size()), cb(sst->g.size());
    (void)lam->dev->localDtCells().download(ca.data());
    (void)sst->dev->localDtCells().download(cb.data());
    check(a == b && a > 0 && ca == cb,
          "M5: laminar localDt == SST's at k == 0, every cell bitwise");
    const auto n0 = lam->dev->residual(*lam->ds, *lam->red);
    lam->dev->setLts(true);
    double dt = 0.95 * a;
    for (int i = 0; i < 400; ++i) {
        if (i % 16 == 0) dt = 0.95 * lam->dev->localDt(*lam->ds, *lam->red, 0.4);
        lam->dev->step(*lam->ds, gpu::Real(dt));
        lam->wall->apply(*lam->ds);
    }
    const auto n1 = lam->dev->residual(*lam->ds, *lam->red);
    std::printf("M5: laminar oracle residual mom %.3e -> %.3e after 400 "
                "compensated LTS steps\n",
                n0.mom, n1.mom);
    check(std::isfinite(n1.mom) && n1.t[0] == 0 && n1.t[1] == 0,
          "M5: nT == 0 residual finite, no scalar norms");
    check(n1.mom < 10 * n0.mom + 1e-12,
          "M5: the laminar fixed point holds under the compensated LTS march");
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    maskedResidual(RansModel::SaNeg);
    maskedResidual(RansModel::Sst2003m);
    guardMask();
    testSourceExact();
    laminarSteady();
    return dnstest::finish();
}
