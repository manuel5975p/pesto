// Device compressible solver is axis-blind: step(P s) == P step(s) for the
// cyclic rotation x -> y -> z (momenta permuted) and the mirror x -> -x (mx
// negated), on an anisotropic box. Variants: a ghost-wall sphere centred in a
// cubic box (classification, image and tally must permute with the state, and
// the force vector with them), and LES on (Smagorinsky and sigma: both
// operators are rotation invariant and Delta^2 = (hx hy hz)^(2/3) is
// permutation invariant). FP32 fields: 1e-6 of each field's max.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "curv_test_util.hpp"
#include "permute_test_util.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>

using namespace dns;
using dnstest::check;

namespace {

constexpr double kGamma = 1.4, kMu = 0.01;
constexpr int kSteps = 10;
constexpr double kTol = 1e-6;

using Perm = CompState (*)(const Grid&, const CompState&);

struct Body
{
    double cx, cy, cz, r;
};

struct Run
{
    CompState s;
    std::array<double, 3> f{0, 0, 0};
    bool ok = false;
};

// kSteps steps at dt on g, optionally with a ghost sphere and LES.
Run advance(const Grid& g, const CompState& s0, double dt,
            const std::optional<Body>& body, double cs, SgsModel sgs)
{
    Run r;
    auto dev = gpu::DeviceComp::make(g, kGamma, kMu, cs, sgs);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !ds) return r;
    std::optional<gpu::DeviceCompGhost> gh;
    if (body) {
        const Body b = *body;
        auto m = gpu::DeviceCompGhost::make(
            g,
            [b](double x, double y, double z) {
                return std::sqrt((x - b.cx) * (x - b.cx) + (y - b.cy) * (y - b.cy)
                                 + (z - b.cz) * (z - b.cz)) - b.r;
            },
            kGamma, 1.0, 1.0);
        if (!m) return r;
        gh.emplace(std::move(*m));
        dev->setStageHook([&](gpu::DeviceCompState& st) { gh->apply(st); });
    }
    (void)ds->upload(s0);
    for (int it = 0; it < kSteps; ++it) dev->step(*ds, gpu::Real(dt));
    if (gh) {
        gh->apply(*ds);
        const auto F = gh->force(*dev, *ds);
        r.f = {F.fx, F.fy, F.fz};
    }
    r.s = CompState::zeros(g);
    (void)ds->download(r.s);
    r.ok = true;
    return r;
}

double stableDt(const Grid& g, const CompState& s)
{
    auto dev = gpu::DeviceComp::make(g, kGamma, kMu);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !red) return 0.0;
    (void)ds->upload(s);
    return dev->stableDt(*ds, *red);
}

void permuteCase(const std::string& tag, const Grid& g, const CompState& s0,
                 const Grid& gp, Perm P, const std::optional<Body>& body,
                 const std::optional<Body>& bodyP,
                 std::array<double, 3> (*Pf)(std::array<double, 3>),
                 double cs = 0.0, SgsModel sgs = SgsModel::Smagorinsky)
{
    const double dt = 0.5 * stableDt(g, s0);
    const Run a = advance(g, s0, dt, body, cs, sgs);
    const Run b = advance(gp, P(g, s0), dt, bodyP, cs, sgs);
    if (!a.ok || !b.ok || !(dt > 0)) { check(false, tag + ": GPU setup"); return; }
    const double e = permtest::relState(b.s, P(g, a.s));
    std::printf("%-34s dt %.3e  max rel |step(P s) - P step(s)| %.3e\n",
                tag.c_str(), dt, e);
    check(e < kTol, tag + ": step(P s) == P step(s) to 1e-6");
    if (cs > 0.0) {
        // The model must actually act, or the LES rows pin nothing new.
        const Run dns = advance(g, s0, dt, body, 0.0, sgs);
        const double d = dns.ok ? permtest::relState(a.s, dns.s) : 0.0;
        std::printf("%-34s LES vs molecular after %d steps: rel %.3e\n",
                    tag.c_str(), kSteps, d);
        check(d > 100 * kTol, tag + ": the SGS model changes the flow");
    }
    if (!body) return;
    const auto fa = Pf(a.f);
    const double fn = std::sqrt(fa[0] * fa[0] + fa[1] * fa[1] + fa[2] * fa[2]);
    const double fe = std::sqrt((b.f[0] - fa[0]) * (b.f[0] - fa[0])
                                + (b.f[1] - fa[1]) * (b.f[1] - fa[1])
                                + (b.f[2] - fa[2]) * (b.f[2] - fa[2]));
    std::printf("%-34s force P F (%.6e, %.6e, %.6e) vs F(P s) (%.6e, %.6e, "
                "%.6e): rel %.3e\n", tag.c_str(), fa[0], fa[1], fa[2], b.f[0],
                b.f[1], b.f[2], fn > 0 ? fe / fn : fe);
    check(fn > 0 && fe < 1e-5 * fn, tag + ": force(P s) == P force(s) to 1e-5");
}

std::array<double, 3> cyclicF(std::array<double, 3> f) { return {f[2], f[0], f[1]}; }
std::array<double, 3> mirrorF(std::array<double, 3> f) { return {-f[0], f[1], f[2]}; }

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    // Plain periodic box, anisotropic spacings.
    {
        const auto g = Grid::make(24, 24, 24, 2.0, 2.5, 3.0);
        const auto s0 = curvtest::smoothState(CurvMesh::uniform(g), kGamma);
        permuteCase("cyclic", g, s0, permtest::cyclicGrid(g), permtest::cyclic,
                    {}, {}, cyclicF);
        permuteCase("mirror", g, s0, g, permtest::mirrorX, {}, {}, mirrorF);
        permuteCase("cyclic LES smag 0.17", g, s0, permtest::cyclicGrid(g),
                    permtest::cyclic, {}, {}, cyclicF, 0.17,
                    SgsModel::Smagorinsky);
        permuteCase("mirror LES smag 0.17", g, s0, g, permtest::mirrorX, {}, {},
                    mirrorF, 0.17, SgsModel::Smagorinsky);
        permuteCase("cyclic LES sigma 1.35", g, s0, permtest::cyclicGrid(g),
                    permtest::cyclic, {}, {}, cyclicF, 1.35, SgsModel::Sigma);
        permuteCase("mirror LES sigma 1.35", g, s0, g, permtest::mirrorX, {}, {},
                    mirrorF, 1.35, SgsModel::Sigma);
    }

    // Ghost sphere centred in a cubic box: the body maps onto itself under
    // both permutations, so the whole wall must permute with the state.
    {
        const double L = 3.0;
        const auto g = Grid::make(24, 24, 24, L, L, L);
        const auto s0 = curvtest::smoothState(CurvMesh::uniform(g), kGamma);
        const Body b{L / 2, L / 2, L / 2, 0.6};
        permuteCase("ghost sphere cyclic", g, s0, permtest::cyclicGrid(g),
                    permtest::cyclic, b, b, cyclicF);
        permuteCase("ghost sphere mirror", g, s0, g, permtest::mirrorX, b, b,
                    mirrorF);
    }
    return dnstest::finish();
}
