// CPU curvilinear compressible solver: metric exactness (closure, tiling,
// Cartesian degeneracy), freestream preservation, conservation, the exact
// nz == 1 restriction, isentropic-vortex convergence, and mesh rejection.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "curv_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace dns;
using namespace curvtest;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

void metricsUniform()
{
    const auto g = Grid::make(8, 6, 4, 2.0, 1.2, 0.6);
    const auto m = CurvMetrics::build(CurvMesh::uniform(g));
    if (!m) { check(false, "uniform metrics build"); return; }
    const double A[3] = {g.hy * g.hz, g.hz * g.hx, g.hx * g.hy};
    const double H[3] = {g.hx, g.hy, g.hz};
    double err = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c) {
        err = std::max(err, std::abs(m->vol[c] - g.cellVolume()));
        for (int a = 0; a < 3; ++a) {
            for (int cx = 0; cx < 3; ++cx)
                err = std::max(err,
                               std::abs(m->S[a][cx][c] - (cx == a ? A[a] : 0.0)));
            for (int d = 0; d < 3; ++d)
                for (int cx = 0; cx < 3; ++cx)
                    err = std::max(err, std::abs(m->Ji[a][d * 3 + cx][c] -
                                                 (d == cx ? 1.0 / H[d] : 0.0)));
            err = std::max(err, std::abs(m->aov[a][c] - 1.0 / H[a]));
        }
    }
    std::printf("uniform metric err %.2e, hmin %.6f\n", err, m->hmin);
    check(err < 1e-13, "uniform metrics equal Cartesian analytics");
    check(std::abs(m->hmin - 0.15) < 1e-13, "hmin is the smallest spacing");
}

void closureAndTiling()
{
    const double lx = 2.0, ly = 1.7, lz = 1.3;
    const auto g = Grid::make(12, 10, 8, lx, ly, lz);
    const auto m = CurvMetrics::build(CurvMesh::sinusoidal(g, 0.05));
    if (!m) { check(false, "distorted metrics build"); return; }
    double close = 0.0, smax = 0.0, tot = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                tot += m->vol[c];
                for (int cx = 0; cx < 3; ++cx) {
                    double sum = 0.0;
                    for (int a = 0; a < 3; ++a) {
                        const std::size_t lo =
                            g.idx(i - (a == 0), j - (a == 1), k - (a == 2));
                        sum += m->S[a][cx][c] - m->S[a][cx][lo];
                        smax = std::max(smax, std::abs(m->S[a][cx][c]));
                    }
                    close = std::max(close, std::abs(sum));
                }
            }
    std::printf("face closure %.2e (max |S| %.2e), vol defect %.2e\n", close,
                smax, tot - lx * ly * lz);
    check(close < 1e-14, "per-cell face-vector closure is exact");
    check(std::abs(tot - lx * ly * lz) < 1e-12 * lx * ly * lz,
          "distorted cells tile the box volume");
}

void uniformDegeneracy()
{
    const auto g = Grid::make(16, 12, 6, 2.0, 1.5, 0.9);
    const auto cart = Comp::make(g, 1.4, 3e-3);
    const auto mesh = CurvMesh::uniform(g);
    const auto curv = Comp::make(mesh, 1.4, 3e-3);
    if (!curv) { check(false, "uniform curvilinear make"); return; }
    const auto s = smoothState(mesh, 1.4);
    auto ra = CompState::zeros(g), rb = CompState::zeros(g);
    cart.rhs(s, ra);
    curv->rhs(s, rb);
    const double e =
        std::max({relLinf(rb.rho, ra.rho), relLinf(rb.mx, ra.mx),
                  relLinf(rb.my, ra.my), relLinf(rb.mz, ra.mz),
                  relLinf(rb.E, ra.E)});
    std::printf("uniform-mesh rhs vs Cartesian rel Linf %.2e\n", e);
    check(e < 1e-12, "uniform-mesh curvilinear rhs equals Cartesian");
}

void freestream()
{
    for (const bool flat : {false, true}) {
        const auto g = flat ? Grid::make(16, 16, 1, 2.0, 2.0, 1.0)
                            : Grid::make(12, 12, 12, 2.0, 2.0, 2.0);
        const auto mesh = CurvMesh::sinusoidal(g, 0.05);
        const auto c = Comp::make(mesh, 1.4, 2e-3);
        if (!c) { check(false, "distorted make"); continue; }
        auto s = CompState::zeros(g);
        const double rho = 1.2, u = 1.0, v = 0.5, w = flat ? 0.0 : 0.25,
                     p = 0.9;
        for (std::size_t i = 0; i < g.size(); ++i) {
            s.rho[i] = rho;
            s.mx[i] = rho * u;
            s.my[i] = rho * v;
            s.mz[i] = rho * w;
            s.E[i] = p / 0.4 + 0.5 * rho * (u * u + v * v + w * w);
        }
        auto r = CompState::zeros(g);
        c->rhs(s, r);
        double e = 0.0;
        for (std::size_t i = 0; i < g.size(); ++i)
            e = std::max({e, std::abs(r.rho[i]), std::abs(r.mx[i]),
                          std::abs(r.my[i]), std::abs(r.mz[i]),
                          std::abs(r.E[i])});
        const double dt = c->stableDt(s);
        auto s2 = s;
        for (int it = 0; it < 5; ++it) c->step(s2, dt);
        double drift = 0.0;
        for (std::size_t i = 0; i < g.size(); ++i)
            drift = std::max({drift, std::abs(s2.rho[i] - rho),
                              std::abs(s2.E[i] - s.E[i])});
        std::printf("%s freestream: rhs %.2e, 5-step drift %.2e, dt %.3e\n",
                    flat ? "2D" : "3D", e, drift, dt);
        check(e < 1e-11, flat ? "2D freestream rhs vanishes"
                              : "3D freestream rhs vanishes");
        check(drift < 1e-11, flat ? "2D freestream preserved over steps"
                                  : "3D freestream preserved over steps");
    }
}

void conservation()
{
    const auto g = Grid::make(12, 12, 12, 2.0, 2.0, 2.0);
    const auto mesh = CurvMesh::sinusoidal(g, 0.05);
    const auto c = Comp::make(mesh, 1.4, 5e-3);
    if (!c) { check(false, "distorted make"); return; }
    auto s = smoothState(mesh, 1.4);
    const auto t0 = totals(*c->curv, s);
    const double dt = 0.5 * c->stableDt(s);
    for (int it = 0; it < 10; ++it) c->step(s, dt);
    const auto t1 = totals(*c->curv, s);
    double e = 0.0;
    for (int q = 0; q < 5; ++q)
        e = std::max(e, std::abs(t1[q] - t0[q]) /
                            std::max(std::abs(t0[q]), 1.0));
    std::printf("conservation drift %.2e (rho %.2e E %.2e)\n", e,
                t1[0] - t0[0], t1[4] - t0[4]);
    check(e < 1e-12, "mass/momentum/energy conserved on distorted mesh");
}

void twoDExact()
{
    const auto g = Grid::make(24, 24, 1, 10.0, 10.0, 1.0);
    const auto mesh = CurvMesh::sinusoidal(g, 0.04);
    const auto c = Comp::make(mesh, 1.4, 2e-3);
    if (!c) { check(false, "2D distorted make"); return; }
    auto s = vortexState(mesh, 5.0, 1.4);
    const double dt = 0.9 * c->stableDt(s);
    for (int it = 0; it < 10; ++it) c->step(s, dt);
    bool mzZero = true, finite = true;
    for (std::size_t i = 0; i < g.size(); ++i) {
        if (s.mz[i] != 0.0) mzZero = false;
        if (!std::isfinite(s.rho[i]) || !std::isfinite(s.E[i])) finite = false;
    }
    check(mzZero, "nz == 1: mz stays exactly zero on a distorted mesh");
    check(finite, "nz == 1 distorted run stays finite");
}

double vortexError(int n, double T)
{
    const double L = 10.0;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const auto mesh = CurvMesh::sinusoidal(g, 0.04);
    const auto c = Comp::make(mesh, 1.4, 0.0); // inviscid: pure advection
    if (!c) return -1.0;
    auto s = vortexState(mesh, 5.0, 1.4);
    const int steps = int(std::ceil(T / (0.9 * c->stableDt(s))));
    const double dt = T / steps;
    for (int it = 0; it < steps; ++it) c->step(s, dt);
    // Exact solution: the IC advected by (T, T), periodically wrapped.
    double num = 0.0, den = 0.0;
    auto wrapc = [&](double d) {
        while (d < -L / 2) d += L;
        while (d >= L / 2) d -= L;
        return d;
    };
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const auto p = centroid(mesh, i, j, 0);
            const double dx = wrapc(p[0] - T - L / 2), dy = wrapc(p[1] - T - L / 2);
            const double ex =
                vortexRho(dx, dy, 0.0, 0.0, 5.0, 1.4);
            const std::size_t cix = g.idx(i, j, 0);
            const double dvol = c->curv->vol[cix];
            num += dvol * (s.rho[cix] - ex) * (s.rho[cix] - ex);
            den += dvol;
        }
    return std::sqrt(num / den);
}

void vortexConvergence()
{
    const double e24 = vortexError(24, 0.5), e48 = vortexError(48, 0.5);
    const double ratio = e24 / e48;
    std::printf("vortex L2(rho) err: n=24 %.3e n=48 %.3e ratio %.2f\n", e24,
                e48, ratio);
    check(e24 > 0 && e48 > 0, "vortex runs completed");
    check(ratio > 2.7, "second-order convergence on the distorted mesh");
}

void rejection()
{
    const auto g = Grid::make(12, 12, 1, 2.0, 2.0, 1.0);
    check(!CurvMetrics::build(CurvMesh::sinusoidal(g, 0.5)),
          "inverted cells are rejected");
    auto short_ = CurvMesh::uniform(g);
    short_.x.pop_back();
    check(!CurvMetrics::build(short_), "wrong node-array size is rejected");
    auto bent = CurvMesh::uniform(g);
    bent.z[3] = 0.1;
    check(!CurvMetrics::build(bent),
          "non-planar layer at nz == 1 is rejected");
    const auto g1 = Grid::make(1, 8, 1, 1.0, 1.0, 1.0);
    check(!CurvMetrics::build(CurvMesh::uniform(g1)),
          "degenerate x axis is rejected");
}

} // namespace

int main()
{
    metricsUniform();
    closureAndTiling();
    uniformDegeneracy();
    freestream();
    conservation();
    twoDExact();
    vortexConvergence();
    rejection();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
