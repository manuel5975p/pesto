// Similarity invariances of the full compressible body rig: ghost-wall
// cylinder, NSCBC x faces with the default sink, free-slip y walls.
//
// (a) Thermodynamic scaling: rho, p, mu (and so E, m, kappa) x2 at fixed u.
//     Sound speed, nu and every rate are unchanged, so the scaled run is the
//     base run times 2 -- and since every scaled quantity is multiplied by a
//     POWER OF TWO, every FP32 operation of a homogeneous scheme reproduces the
//     base result times 2 EXACTLY (no rounding difference can arise). Expected:
//     state_b == 2 state_a bit for bit, forces x2 bit for bit.
// (b) Length scaling: L, h, D, hz and mu x2 at fixed rho, u, p (Re, M fixed)
//     and dt x2, same step count. Every flux / h, viscous term mu/h^2 and dt
//     product sees the same mantissas, so the state is identical bit for bit,
//     force per face x4 (area hy hz x4), cd identical.
// Any departure is a non-homogeneous term in the scheme (an absolute floor,
// an epsilon, a hard-coded length) and is reported with the cell count and
// the largest relative difference.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_far.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/comp_nscbc.hpp"
#include "dns/gpu/comp_sink.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

using namespace dns;

namespace {

using dnstest::check;

constexpr double gam = 1.4, uInf = 1.0;

struct Scale
{
    double thermo = 1.0; // rho, p, mu multiplier
    double length = 1.0; // L, h, D, mu multiplier (and dt)
};

struct Out
{
    CompState s;
    gpu::DeviceCompGhost::Force F{};
    bool ok = false;
};

// The base field (thermo = length = 1) on the base grid, built once; scaled
// runs multiply it (thermo) or reuse it index for index (length).
CompState baseState(const Grid& g, double cx, double cy, double R, double rhoInf,
                    double pInf)
{
    auto h = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double d = std::hypot((i + 0.5) * g.hx - cx,
                                        (j + 0.5) * g.hy - cy) - R;
            const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
            const double u = uInf * f * f * (3 - 2 * f);
            const double v = 0.05 * f * std::sin(2.0 * (i + 0.5) / g.nx * 6.283185307179586);
            h.rho[c] = rhoInf;
            h.mx[c] = rhoInf * u;
            h.my[c] = rhoInf * v;
            h.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * (u * u + v * v);
        }
    return h;
}

Out run(const Scale& sc, const CompState& ic, double dt, int steps)
{
    Out o;
    const double k = sc.length, m = sc.thermo;
    const int nx = 128, ny = 64;
    const double h0 = 1.0 / 16.0;
    const auto g = Grid::make(nx, ny, 1, k * nx * h0, k * ny * h0, k * 1.0);
    const double R = k * 0.5, cx = k * (2.5 + 0.3125 * h0),
                 cy = k * (2.0 + 0.1875 * h0);
    const double rhoInf = m * 1.0, pInf = m * (1.0 / (gam * 0.04));
    const double mu = m * k * (1.0 / 128.0);
    const gpu::GhostSdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - R;
    };
    auto dev = gpu::DeviceComp::make(g, gam, mu);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, gam, rhoInf, pInf);
    auto far = gpu::DeviceCompFar::make(
        g, {.rhoInf = rhoInf, .uInf = uInf, .pInf = pInf, .gamma = gam,
            .ns = 8, .sigMax = 0.0});
    gpu::DeviceCompNscbc::Config nc;
    nc.rhoInf = rhoInf;
    nc.uInf = uInf;
    nc.pInf = pInf;
    nc.gamma = gam;
    auto nsc = gpu::DeviceCompNscbc::make(g, nc);
    auto snk = gpu::DeviceCompSink::make(
        g, {.nb = 16, .alpha = 1.0, .cInf = std::sqrt(gam * pInf / rhoInf)});
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !far || !nsc || !snk || !ds) return o;
    dev->setStageHook([&](gpu::DeviceCompState& s) {
        far->apply(s);
        gh->apply(s);
    });
    dev->setRhsHook([&](const gpu::DeviceCompState& s, gpu::DeviceCompState& out,
                        const gpu::StageCombine& cmb) {
        nsc->apply(*dev, s, out, cmb);
        snk->apply(s, out, cmb);
    });
    auto h = ic;
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] *= m;
        h.mx[c] *= m;
        h.my[c] *= m;
        h.mz[c] *= m;
        h.E[c] *= m;
    }
    (void)ds->upload(h);
    for (int s = 0; s < steps; ++s) dev->step(*ds, gpu::Real(k * dt));
    far->apply(*ds);
    gh->apply(*ds);
    o.F = gh->force(*dev, *ds);
    o.s = CompState::zeros(g);
    (void)ds->download(o.s);
    o.ok = true;
    return o;
}

// Cells where b != f * a, and the largest relative difference.
void compare(const CompState& a, const CompState& b, double f, int& nDiff,
             double& maxRel)
{
    nDiff = 0;
    maxRel = 0.0;
    const std::vector<double>* A[5] = {&a.rho, &a.mx, &a.my, &a.mz, &a.E};
    const std::vector<double>* B[5] = {&b.rho, &b.mx, &b.my, &b.mz, &b.E};
    for (int q = 0; q < 5; ++q)
        for (std::size_t c = 0; c < A[q]->size(); ++c) {
            const double x = f * (*A[q])[c], y = (*B[q])[c];
            if (x == y) continue;
            ++nDiff;
            maxRel = std::max(maxRel, std::abs(x - y)
                                          / std::max(std::abs(x), 1e-30));
        }
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const int nx = 128, ny = 64;
    const double h0 = 1.0 / 16.0;
    const auto g0 = Grid::make(nx, ny, 1, nx * h0, ny * h0, 1.0);
    const double pInf0 = 1.0 / (gam * 0.04);
    const auto ic = baseState(g0, 2.5 + 0.3125 * h0, 2.0 + 0.1875 * h0, 0.5, 1.0,
                              pInf0);
    // dt from the base state (acoustic CFL ~0.3), fixed for every run.
    const double c0 = std::sqrt(gam * pInf0);
    const double dt = 0.3 * h0 / (uInf + c0);
    // t = 0.625: the start-up acoustics (c ~ 5.3) have crossed the 2.5 D
    // to the inflow face and the y walls, so NSCBC and the sink act on them.
    const int steps = 200;

    const Out a = run({1.0, 1.0}, ic, dt, steps);
    const Out b = run({2.0, 1.0}, ic, dt, steps);
    const Out c = run({1.0, 2.0}, ic, dt, steps);
    if (!a.ok || !b.ok || !c.ok) { check(false, "rig setup"); return dnstest::finish(); }
    std::printf("base: F (%.9e, %.9e), %d steps dt %.4e\n", a.F.fx, a.F.fy,
                steps, dt);
    check(std::isfinite(a.F.fx) && a.F.fx != 0.0, "base force finite, nonzero");

    int nd = 0;
    double mr = 0.0;
    compare(a.s, b.s, 2.0, nd, mr);
    std::printf("(a) rho,p,mu x2: %d values differ from 2x base (max rel %.2e); "
                "F ratio (%.17g, %.17g)\n", nd, mr, b.F.fx / a.F.fx,
                b.F.fy / a.F.fy);
    check(nd == 0, "(a) thermo x2: state is exactly 2x the base, bit for bit");
    check(b.F.fx == 2.0 * a.F.fx && b.F.fy == 2.0 * a.F.fy,
          "(a) thermo x2: force is exactly 2x the base");

    compare(a.s, c.s, 1.0, nd, mr);
    const double cdA = a.F.fx / (0.5 * 1.0 * uInf * uInf * 1.0 * 1.0);
    const double cdC = c.F.fx / (0.5 * 1.0 * uInf * uInf * 2.0 * 2.0);
    std::printf("(b) L,h,mu,dt x2: %d values differ from base (max rel %.2e); "
                "F ratio %.17g; cd %.12f vs %.12f\n", nd, mr, c.F.fx / a.F.fx,
                cdA, cdC);
    check(nd == 0, "(b) length x2: state identical to the base, bit for bit");
    check(c.F.fx == 4.0 * a.F.fx && c.F.fy == 4.0 * a.F.fy,
          "(b) length x2: force is exactly 4x (area x4)");
    check(cdA == cdC, "(b) length x2: cd identical");
    return dnstest::finish();
}
