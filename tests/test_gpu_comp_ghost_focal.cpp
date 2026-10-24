// Compressible ghost wall on a FOCAL (per-axis clustered) mesh: the inactive
// remap must reproduce the Cartesian construction exactly, and with the body
// inside the fine plateau the wall must keep a quiescent gas quiescent, report
// no spurious force, and produce sane positive drag in a freestream. A body
// placed off the box centre, with the plateau stated physically through
// focalRemap, must sit on the fine cells (the --focal contract).
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;



const double Lbox = 2 * pi, Rb = Lbox / 16;
const double cxB = Lbox / 2, cyB = Lbox / 2 + 0.013;
const double gam = 1.4, rhoInf = 1.0, pInf = 1.0 / (gam * 0.04); // M 0.2

double circle(double x, double y, double)
{
    return std::sqrt((x - cxB) * (x - cxB) + (y - cyB) * (y - cyB)) - Rb;
}

// The plateau holds the body plus a comfortable guard band, mirroring the
// containment rule the validator enforces. wx scales the x plateau: equal
// half-widths give ISOTROPIC fine cells, unequal ones the anisotropic fine
// cells a wake-following tube actually produces.
AxisRemap3 focalAboutBody(double R, double wx = 3.0, double bf = 0.5)
{
    const double lo = cxB - wx * Rb, hi = cxB + wx * Rb;
    return AxisRemap3{Focal1D::make(Lbox, lo, hi, bf * (hi - lo), R),
                      Focal1D::make(Lbox, cyB - 3 * Rb, cyB + 3 * Rb,
                                    3 * Rb, R),
                      Focal1D::identity(1.0)};
}

// Inactive focal maps must give the ghost wall byte-identical geometry: the
// classification is the same arithmetic through an identity branch.
void degenerateMatchesCartesian(const Grid& g)
{
    const AxisRemap3 I{Focal1D::identity(Lbox), Focal1D::identity(Lbox),
                       Focal1D::identity(1.0)};
    auto cart = gpu::DeviceCompGhost::make(g, circle, gam, rhoInf, pInf);
    auto foc = gpu::DeviceCompGhost::make(g, circle, gam, rhoInf, pInf,
                                          nullptr, I);
    if (!cart || !foc) { check(false, "ghost make"); return; }
    check(cart->solidCells() == foc->solidCells(),
          "inactive focal remap classifies the same solid cells");
    check(!foc->curvilinear(), "inactive focal remap is not curvilinear");

    // Same wall states after a fill from the same field.
    auto dc = gpu::DeviceComp::make(g, gam, 1e-3);
    auto sa = gpu::DeviceCompState::make(g.size());
    auto sb = gpu::DeviceCompState::make(g.size());
    if (!dc || !sa || !sb) { check(false, "GPU alloc"); return; }
    auto host = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double u = 0.7 + 0.1 * std::sin(3.0 * i * g.hx),
                         v = 0.2 * std::cos(2.0 * j * g.hy);
            host.rho[c] = rhoInf;
            host.mx[c] = rhoInf * u;
            host.my[c] = rhoInf * v;
            host.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * (u * u + v * v);
        }
    (void)sa->upload(host);
    (void)sb->upload(host);
    cart->apply(*sa);
    foc->apply(*sb);
    auto ha = CompState::zeros(g), hb = CompState::zeros(g);
    (void)sa->download(ha);
    (void)sb->download(hb);
    bool same = true;
    for (std::size_t c = 0; c < g.size(); ++c)
        if (ha.rho[c] != hb.rho[c] || ha.mx[c] != hb.mx[c] ||
            ha.my[c] != hb.my[c] || ha.E[c] != hb.E[c])
            same = false;
    check(same, "inactive focal ghost fill is bit-identical");
}

// The mirror band must be at least two cells thick along EVERY axis: the
// interior stencil reaches two cells, so a quiescent deep-solid cell within
// that reach is a hard discontinuity against the fluid. The threshold that
// separates deep from mirrored is a physical depth, so on anisotropic cells
// it has to follow the WIDEST local spacing -- setting it from the smallest
// leaves the band under two cells thick along the wide axis, and the
// measured Re100 cylinder drag flipped sign (cd 1.49 -> -7.6).
void mirrorBandDepth(const Grid& g, const AxisRemap3& R,
                     const gpu::DeviceCompGhost::Sdf& sdf, const char* tag)
{
    auto ghost = gpu::DeviceCompGhost::make(g, sdf, gam, rhoInf, pInf,
                                            nullptr, R);
    if (!ghost) { check(false, "band ghost make"); return; }
    std::vector<char> solid(g.size(), 0), deep(g.size(), 0);
    for (int c : ghost->solidCells()) solid[c] = 1;
    for (int c : ghost->deepCells()) deep[c] = 1;
    int thin = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                if (!deep[g.idx(i, j, k)]) continue;
                const int off[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                for (int a = 0; a < (g.nz == 1 ? 2 : 3); ++a)
                    for (int s = -2; s <= 2; ++s) {
                        if (s == 0) continue;
                        if (!solid[g.idx(i + s * off[a][0], j + s * off[a][1],
                                         k + s * off[a][2])])
                            ++thin;
                    }
            }
    std::printf("%s: %d deep cells, %d fluid cells within 2 of a deep cell\n",
                tag, int(ghost->deepCells().size()), thin);
    check(thin == 0, tag);
}

void mirrorBandDepth(const Grid& g, double wx, double bf, const char* tag)
{
    mirrorBandDepth(g, focalAboutBody(4.0, wx, bf), circle, tag);
}

// The --focal CONTRACT on the ghost wall: the plateau is stated PHYSICALLY
// and built by focalRemap (the one builder every consumer uses), and the body
// sits OFF the box centre, where map(X) != X -- the case the reference-space
// construction got wrong (the fine region landed ~R times narrower and
// shifted, the body on coarse cells). Every solid cell must then see the
// fine spacing, and the solid/deep counts must be the FINE lattice's.
void worldPlacedBody(const Grid& g)
{
    std::printf("--- world-placed body off the box centre ---\n");
    const double ox = Lbox / 3, oy = Lbox / 2 + 0.013;
    const gpu::DeviceCompGhost::Sdf off = [=](double x, double y, double) {
        return std::hypot(x - ox, y - oy) - Rb;
    };
    const double lo[3] = {ox - 1.5 * Rb, oy - 1.5 * Rb, 0.0};
    const double hi[3] = {ox + 1.5 * Rb, oy + 1.5 * Rb, 0.0};
    const double w[3] = {1.5 * Rb, 1.5 * Rb, 0.0};
    const AxisRemap3 R = focalRemap(g, true, lo, hi, w, 4.0);
    const double hFx = R.fx.c * R.fx.r * g.hx, hFy = R.fy.c * R.fy.r * g.hy;
    check(R.fx.clears(4 * g.hx) && R.fy.clears(4 * g.hy),
          "world-placed plateau clears the faces");
    check(std::abs(R.fx.map(R.fx.a) - lo[0]) < 1e-12
              && std::abs(R.fy.map(R.fy.b) - hi[1]) < 1e-12,
          "world-placed plateau is the physical box");

    auto counts = [&](const AxisRemap3& M, double& worst) {
        const auto T = gpu::classifyCompGhost(g, off, nullptr, M, nullptr);
        const double fx = M.fx.c * M.fx.r * g.hx, fy = M.fy.c * M.fy.r * g.hy;
        worst = 0.0;
        for (int c : T.solid)
        {
            const int i = c % g.nx, j = (c / g.nx) % g.ny;
            worst = std::max(
                {worst, std::abs(M.fx.deriv((i + 0.5) * g.hx) * g.hx / fx - 1.0),
                 std::abs(M.fy.deriv((j + 0.5) * g.hy) * g.hy / fy - 1.0)});
        }
        return T;
    };
    double worst = 0.0;
    const auto T = counts(R, worst);
    // Fine-lattice expectations: cells inside the disc, and inside the disc
    // shrunk by the 2-cell mirror band (the deep threshold is 2 x the widest
    // local spacing).
    const double hMax = std::max(hFx, hFy);
    const double solidExp = pi * Rb * Rb / (hFx * hFy);
    const double deepExp = pi * (Rb - 2 * hMax) * (Rb - 2 * hMax) / (hFx * hFy);
    std::printf("  h_fine %.4e x %.4e (ref %.4e): solid %d (fine lattice "
                "%.0f), deep %d (%.0f), ghosts %d; worst spacing dev %.2e\n",
                hFx, hFy, g.hx, int(T.solid.size()), solidExp, T.nDeep(),
                deepExp, T.nGhost(), worst);
    check(worst < 1e-12, "every solid cell of a world-placed body is h_fine");
    check(std::abs(T.solid.size() / solidExp - 1.0) < 0.03,
          "solid count is the fine lattice's");
    check(std::abs(T.nDeep() / deepExp - 1.0) < 0.10,
          "deep-solid count is the fine lattice's");

    // The pre-fix construction (physical numbers read as reference ones):
    // the body lands on blend/coarse cells. Measured here: solid 1063 vs
    // 1148, but the solid cells span up to 4x h_fine -- the spacing, not the
    // count, is what discriminates.
    const AxisRemap3 B{Focal1D::make(Lbox, lo[0], hi[0], w[0], 4.0),
                       Focal1D::make(Lbox, lo[1], hi[1], w[1], 4.0),
                       Focal1D::identity(1.0)};
    double worstB = 0.0;
    const auto TB = counts(B, worstB);
    std::printf("  reference-space construction: solid %d, deep %d, worst "
                "spacing dev %.2f\n",
                int(TB.solid.size()), TB.nDeep(), worstB);
    check(worstB > 0.5,
          "the reference-space construction puts the body off the fine "
          "region (the check discriminates)");

    // On the device: the mirror band stays two cells deep, and a uniform
    // state carries no static force on the off-centre body.
    mirrorBandDepth(g, R, off, "world-placed: mirror band is 2 cells deep");
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto cpu = Comp::make(mesh, gam, 0.0);
    if (!cpu) { check(false, "world-placed CPU make"); return; }
    auto dev = gpu::DeviceComp::make(*cpu->curv, gam, 0.0);
    auto ghost = gpu::DeviceCompGhost::make(g, off, gam, rhoInf, pInf, nullptr, R);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !ghost || !ds) { check(false, "world-placed GPU alloc"); return; }
    check(int(ghost->deepCells().size()) == T.nDeep(),
          "device ghost wall reports the host deep count");
    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gam - 1.0);
    }
    (void)ds->upload(host);
    ghost->apply(*ds);
    const auto F = ghost->force(*dev, *ds);
    const double scale = 0.5 * rhoInf * 2 * Rb;
    std::printf("  static force (%.3e, %.3e)\n", F.fx, F.fy);
    check(std::abs(F.fx) < 1e-4 * scale && std::abs(F.fy) < 1e-4 * scale,
          "world-placed body: no static force");
}

// The same freestream transient on the plain Cartesian mesh, computed once:
// the reference the focal runs are pinned against.
double cartesianCd(const Grid& g)
{
    static double cached = 0.0;
    if (cached != 0.0) return cached;
    const double mu = rhoInf * 2 * Rb / 100.0;
    auto dev = gpu::DeviceComp::make(g, gam, mu);
    auto ghost = gpu::DeviceCompGhost::make(g, circle, gam, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ghost || !ds || !red) return 1.0;
    dev->setStageHook([&](gpu::DeviceCompState& st) { ghost->apply(st); });
    auto host = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double d = circle((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0.0);
            const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
            const double u = f * f * (3 - 2 * f);
            host.rho[c] = rhoInf;
            host.mx[c] = rhoInf * u;
            host.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * u * u;
        }
    (void)ds->upload(host);
    double t = 0.0;
    while (t < 2.0) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
        t += dt;
    }
    ghost->apply(*ds);
    cached = ghost->force(*dev, *ds).fx / (0.5 * rhoInf * 2 * Rb);
    return cached;
}

// The body sits in the fine plateau: the wall sees locally uniform cells, so
// the Cartesian guarantees must survive.
void focalWall(const Grid& g, double wx, double bf, const char* tag)
{
    std::printf("--- focal wall, %s ---\n", tag);
    const AxisRemap3 R = focalAboutBody(4.0, wx, bf);
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto cpu = Comp::make(mesh, gam, rhoInf * 2 * Rb / 100.0);
    if (!cpu) { check(false, "focal CPU make"); return; }
    auto dev = gpu::DeviceComp::make(*cpu->curv, gam, rhoInf * 2 * Rb / 100.0);
    auto ghost = gpu::DeviceCompGhost::make(g, circle, gam, rhoInf, pInf,
                                            nullptr, R);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ghost || !ds || !red) { check(false, "GPU alloc"); return; }
    check(ghost->curvilinear(), "active focal ghost reports curvilinear");
    check(!ghost->solidCells().empty(), "focal ghost found the body");
    dev->setStageHook([&](gpu::DeviceCompState& st) { ghost->apply(st); });

    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gam - 1.0);
    }
    (void)ds->upload(host);
    for (int it = 0; it < 50; ++it) dev->step(*ds, gpu::Real(2e-4));
    ghost->apply(*ds);
    (void)ds->download(host);
    double vmax = 0.0;
    bool mzZero = true;
    for (std::size_t c = 0; c < g.size(); ++c) {
        vmax = std::max({vmax, std::abs(host.mx[c]), std::abs(host.my[c])});
        if (host.mz[c] != 0.0) mzZero = false;
    }
    const auto F0 = ghost->force(*dev, *ds);
    std::printf("focal quiescent: max|m| %.2e force (%.2e, %.2e)\n", vmax,
                F0.fx, F0.fy);
    check(vmax < 1e-5, "quiescent gas stays at rest on the focal mesh");
    check(std::abs(F0.fx) < 1e-4 && std::abs(F0.fy) < 1e-4,
          "no spurious force on the focal mesh");
    check(mzZero, "nz == 1: mz stays exactly zero with the focal wall");

    // Tapered freestream, physical cell centres from the remap.
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double x = R.fx.map((i + 0.5) * g.hx),
                         y = R.fy.map((j + 0.5) * g.hy);
            const double d = circle(x, y, 0.0);
            const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
            const double u = f * f * (3 - 2 * f);
            host.rho[c] = rhoInf;
            host.mx[c] = rhoInf * u;
            host.my[c] = 0.0;
            host.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * u * u;
        }
    (void)ds->upload(host);
    double t = 0.0;
    while (t < 2.0) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
        t += dt;
    }
    ghost->apply(*ds);
    const auto F = ghost->force(*dev, *ds);
    (void)ds->download(host);
    double smax = 0.0;
    bool finite = true;
    for (std::size_t c = 0; c < g.size(); ++c) {
        const double u = host.mx[c] / host.rho[c],
                     v = host.my[c] / host.rho[c];
        if (!std::isfinite(u) || !std::isfinite(v)) finite = false;
        smax = std::max(smax, std::sqrt(u * u + v * v));
    }
    const double cd = F.fx / (0.5 * rhoInf * 2 * Rb);
    std::printf("focal freestream t=2: cd %.3f max|u| %.3f\n", cd, smax);
    check(finite, "focal freestream evolution finite");
    // Pinned against the Cartesian mesh at the same instant rather than a
    // fixed band: the focal mesh resolves the body better, so the transient
    // differs, but a sign flip or an order-of-magnitude swing is the failure
    // this test exists to catch.
    const double ref = cartesianCd(g);
    std::printf("  cd focal %.3f vs Cartesian %.3f (ratio %.3f)\n", cd, ref,
                cd / ref);
    check(cd > 0.4 * ref && cd < 2.0 * ref,
          "focal drag tracks the Cartesian drag");
    check(smax < 3.0, "focal speed bounded");

    bool same = true;
    for (int rep = 0; rep < 4; ++rep) {
        const auto Fr = ghost->force(*dev, *ds);
        if (Fr.fx != F.fx || Fr.fy != F.fy || Fr.fz != F.fz) same = false;
    }
    check(same, "focal force tally is bit-reproducible");
}

// A uniform state (no flow, uniform pressure) must produce ZERO net force
// whatever the mesh: every interface face sees the same pressure and the
// closed surface's area vectors sum to zero. This is where an anisotropic
// fine region -- the shape a wake-following tube actually has -- is checked,
// separately from any flow evolution.
void staticForce(const Grid& g, double wx, double bf, const char* tag)
{
    const AxisRemap3 R = focalAboutBody(4.0, wx, bf);
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto cpu = Comp::make(mesh, gam, 0.0); // inviscid: pressure only
    if (!cpu) { check(false, "static focal make"); return; }
    auto dev = gpu::DeviceComp::make(*cpu->curv, gam, 0.0);
    auto ghost = gpu::DeviceCompGhost::make(g, circle, gam, rhoInf, pInf,
                                            nullptr, R);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !ghost || !ds) { check(false, "GPU alloc"); return; }
    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gam - 1.0);
    }
    (void)ds->upload(host);
    ghost->apply(*ds);
    const auto F = ghost->force(*dev, *ds);
    const double scale = 0.5 * rhoInf * 2 * Rb;
    std::printf("%s: fine dx/dy %.3f, static force (%.3e, %.3e) -> cd %.3e\n",
                tag, (R.fx.c * R.fx.r * g.hx) / (R.fy.c * R.fy.r * g.hy),
                F.fx, F.fy, F.fx / scale);
    check(std::abs(F.fx) < 1e-4 * scale && std::abs(F.fy) < 1e-4 * scale,
          tag);
}

// Wall heat on the CURVILINEAR tally: kBodyFluxCurv's Fourier term maps the
// index-space temperature gradient through the face inverse Jacobian and
// contracts it with the face area vector, so a clustered mesh must reproduce
// the Cartesian signs and the exact linearity in Twall - Tinf. A uniform
// quiescent gas is used, where the tally is pure conduction and nothing else.
void focalWallHeat(const Grid& g, double wx, double bf, const char* tag)
{
    using WT = gpu::GhostWallThermal;
    const AxisRemap3 R = focalAboutBody(4.0, wx, bf);
    const double mu = rhoInf * 2 * Rb / 100.0;
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto cpu = Comp::make(mesh, gam, mu);
    if (!cpu) { check(false, "focal heat CPU make"); return; }
    auto dev = gpu::DeviceComp::make(*cpu->curv, gam, mu);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !ds) { check(false, "focal heat GPU alloc"); return; }
    const double Tinf = pInf / rhoInf; // R = 1
    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gam - 1.0);
    }
    auto heatAt = [&](WT w) {
        auto gh = gpu::DeviceCompGhost::make(g, circle, gam, rhoInf, pInf,
                                             nullptr, R, w);
        if (!gh) { check(false, "focal heat ghost make"); return 0.0; }
        check(gh->curvilinear(), "focal heat ghost is curvilinear");
        (void)ds->upload(host);
        gh->apply(*ds);
        return gh->force(*dev, *ds).heat;
    };
    const double qAd = heatAt(WT{});
    const double qHot = heatAt(
        WT{.mode = WT::Mode::Isothermal, .Twall = 1.5 * Tinf, .Rgas = 1.0});
    const double qCold = heatAt(
        WT{.mode = WT::Mode::Isothermal, .Twall = 0.5 * Tinf, .Rgas = 1.0});
    const double q10 = heatAt(
        WT{.mode = WT::Mode::Isothermal, .Twall = 1.1 * Tinf, .Rgas = 1.0});
    const double q20 = heatAt(
        WT{.mode = WT::Mode::Isothermal, .Twall = 1.2 * Tinf, .Rgas = 1.0});
    std::printf("%s: heat adiabatic %.3e hot %.4e cold %.4e ratio %.4f\n", tag,
                qAd, qHot, qCold, q20 / q10);
    check(qAd == 0.0, "focal adiabatic wall reports exactly zero heat");
    check(qHot < 0.0, "focal hot wall drives heat into the fluid");
    check(qCold > 0.0, "focal cold wall pulls heat out of the fluid");
    check(std::abs(q20 / q10 - 2.0) < 0.02,
          "focal wall heat is linear in Twall - Tinf for small differences");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const auto g = Grid::make(128, 128, 1, Lbox, Lbox, 1.0);
    degenerateMatchesCartesian(g);
    worldPlacedBody(g);
    mirrorBandDepth(g, 3.0, 0.5, "isotropic: mirror band is 2 cells deep");
    mirrorBandDepth(g, 5.0, 0.25, "anisotropic: mirror band is 2 cells deep");
    staticForce(g, 3.0, 0.5, "isotropic fine cells: no static force");
    staticForce(g, 5.0, 0.25, "anisotropic fine cells: no static force");
    focalWallHeat(g, 3.0, 0.5, "isotropic fine cells: wall heat");
    focalWallHeat(g, 5.0, 0.25, "anisotropic fine cells: wall heat");
    focalWall(g, 3.0, 0.5, "isotropic fine cells");
    // Anisotropic fine cells are what a wake-following tube actually
    // produces, and they are what broke the mirror band: with the
    // deep-solid threshold set by the SMALLEST spacing the band was under
    // two cells thick along the wide axis, the interior stencil reached
    // quiescent deep cells, and the measured Re100 cylinder drag flipped
    // sign (cd 1.49 -> -7.6). Keep this case green.
    focalWall(g, 5.0, 0.25, "anisotropic fine cells");
    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
