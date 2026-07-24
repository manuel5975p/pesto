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

// The TanhStretchY PointRemap and the mesh derived from it: exact identity at
// beta == 0, endpoint/symmetry/monotonicity of the mapping, analytic Jacobian
// vs finite differences, inverse-Jacobian and metric-tensor consistency, and
// valid finite-volume metrics that still tile the box.
void stretchedRemap()
{
    const auto g = Grid::make(8, 16, 1, 2.0, 1.0, 1.0);
    const auto u = CurvMesh::uniform(g);
    const auto s0 = CurvMesh::stretchedY(g, 0.0);
    check(s0.x == u.x && s0.y == u.y && s0.z == u.z,
          "stretchedY(beta = 0) is uniform bit-for-bit");

    const double beta = 2.0, ly = g.ny * g.hy;
    const TanhStretchY R{ly, beta};
    check(R.mapY(0.0) == 0.0 && std::abs(R.mapY(ly) - ly) < 1e-15,
          "tanh stretch preserves the y endpoints");
    bool mono = true, sym = true;
    double prev = R.mapY(0.0);
    for (int j = 1; j <= g.ny; ++j) {
        const double y = R.mapY(j * g.hy);
        if (!(y > prev)) mono = false;
        prev = y;
        const double yr = R.mapY(ly - j * g.hy);
        if (std::abs(y + yr - ly) > 1e-13) sym = false;
    }
    check(mono, "tanh stretch is monotone");
    check(sym, "tanh stretch is symmetric about mid-box");
    check(R.mapY(0.5 * g.hy) < 0.5 * g.hy,
          "cells compress toward the wall");

    // Analytic Jacobian against central differences, and the derived
    // inverse-Jacobian / metric-tensor helpers.
    const double X[3] = {0.3, 0.11, 0.0};
    double J[3][3], Ji[3][3], G[3][3];
    R.jacobian(X, J);
    const double d = 1e-6;
    const double fd = (R.mapY(X[1] + d) - R.mapY(X[1] - d)) / (2 * d);
    check(std::abs(J[1][1] - fd) < 1e-8 && J[0][0] == 1.0 && J[2][2] == 1.0 &&
              J[0][1] == 0.0 && J[1][0] == 0.0,
          "remap Jacobian matches finite differences");
    check(remapInverseJacobian(R, X, Ji) &&
              std::abs(Ji[1][1] * J[1][1] - 1.0) < 1e-13,
          "inverse Jacobian inverts the Jacobian");
    remapMetric(R, X, G);
    check(std::abs(G[1][1] - J[1][1] * J[1][1]) < 1e-13 &&
              std::abs(G[0][0] - 1.0) < 1e-15 && std::abs(G[0][1]) < 1e-15,
          "metric tensor is J^T J");
    check(std::abs(remapDet(R, X) - J[1][1]) < 1e-15,
          "det J is the 1D compression factor");

    // Finite-volume tables from the stretched nodes stay valid and tile the
    // box; the smallest width is the wall cell.
    const auto m = CurvMetrics::build(CurvMesh::stretchedY(g, beta));
    if (!m) { check(false, "stretched metrics build"); return; }
    double tot = 0.0;
    for (const double v : m->vol) tot += v;
    const double box = g.nx * g.hx * ly * g.nz * g.hz;
    check(std::abs(tot - box) < 1e-12 * box, "stretched cells tile the box");
    const double hWall = R.mapY(g.hy) - R.mapY(0.0);
    check(std::abs(m->hmin - hWall) < 1e-10, "hmin is the wall-cell height");
}

// Focal1D / AxisRemap3: bit-exact inactive identity, exact plateau and far
// spacings, Jacobian vs finite differences, exact C^2 periodic extension,
// inverse round-trip, and valid finite-volume metrics on a 3-axis focal mesh.
void focalRemap()
{
    const auto g = Grid::make(16, 16, 8, 4.0, 4.0, 2.0);
    const auto u = CurvMesh::uniform(g);

    const AxisRemap3 id{Focal1D::identity(4.0), Focal1D::identity(4.0),
                        Focal1D::identity(2.0)};
    const auto m0 = CurvMesh::fromRemap(g, id);
    check(m0.x == u.x && m0.y == u.y && m0.z == u.z,
          "inactive AxisRemap3 is uniform bit-for-bit");
    check(!Focal1D::make(4.0, 1.0, 2.0, 0.5, 1.0).active &&
              !Focal1D::make(4.0, 2.0, 1.0, 0.5, 4.0).active &&
              !Focal1D::make(4.0, 1.0, 2.0, 0.0, 4.0).active,
          "invalid Focal1D parameters degrade to the identity");

    const double L = 4.0, a = 1.5, b = 2.5, w = 0.5, R = 4.0;
    const Focal1D f = Focal1D::make(L, a, b, w, R);
    check(f.map(0.0) == 0.0 && std::abs(f.map(L) - L) < 1e-14,
          "focal map preserves the endpoints");

    bool mono = true;
    double prev = f.map(0.0);
    const double h = L / 64;
    for (int j = 1; j <= 64; ++j) {
        const double x = f.map(j * h);
        if (!(x > prev)) mono = false;
        prev = x;
    }
    check(mono, "focal map is monotone");
    const double fine = f.map(2.0 + h) - f.map(2.0);
    const double coarse = f.map(0.25 + h) - f.map(0.25);
    check(std::abs(fine - f.c * f.r * h) < 5e-15 &&
              std::abs(coarse - f.c * h) < 5e-15,
          "plateau spacing c r h, far spacing c h exactly");
    check(std::abs(coarse / fine - R) < 1e-12,
          "cell-size ratio is exactly R");

    bool periodic = true;
    for (const double X : {-0.7, 0.3, 1.8, 2.6, 3.9, 4.4, 9.1}) {
        if (std::abs(f.map(X + L) - (f.map(X) + L)) > 1e-13) periodic = false;
        const double fd =
            (f.map(X + 1e-6) - f.map(X - 1e-6)) / 2e-6;
        if (std::abs(fd - f.deriv(X)) > 1e-7) periodic = false;
    }
    check(periodic, "periodic extension and deriv match everywhere");
    check(f.deriv(1e-9) == f.c && f.deriv(L - 1e-9) == f.c,
          "slope is exactly c at both faces (C2 across the wrap)");

    bool inv = true;
    for (int j = 0; j <= 200; ++j) {
        const double X = -L + 3.0 * L * j / 200.0;
        if (std::abs(f.inverse(f.map(X)) - X) > 1e-12) inv = false;
    }
    check(inv, "inverse(map(X)) round-trips through both blends");

    const AxisRemap3 A{Focal1D::make(4.0, 1.0, 3.0, 0.5, 4.0),
                       Focal1D::make(4.0, 1.5, 2.5, 0.5, 4.0),
                       Focal1D::make(2.0, 0.7, 1.3, 0.3, 2.0)};
    const double X[3] = {0.8, 1.7, 1.2};
    double J[3][3];
    A.jacobian(X, J);
    bool diag = true;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (i != j && J[i][j] != 0.0) diag = false;
    check(diag && remapDet(A, X) > 0.0,
          "AxisRemap3 Jacobian is diagonal with positive det");
    double Ji[3][3];
    check(remapInverseJacobian(A, X, Ji) &&
              std::abs(Ji[0][0] * J[0][0] - 1.0) < 1e-13,
          "inverse Jacobian inverts the diagonal");

    const auto m = CurvMetrics::build(CurvMesh::fromRemap(g, A));
    if (!m) { check(false, "focal metrics build"); return; }
    double tot = 0.0;
    for (const double v : m->vol) tot += v;
    const double box = 4.0 * 4.0 * 2.0;
    check(std::abs(tot - box) < 1e-12 * box, "focal cells tile the box");
    const double hfine = std::min({A.fx.c * A.fx.r * g.hx,
                                   A.fy.c * A.fy.r * g.hy,
                                   A.fz.c * A.fz.r * g.hz});
    check(std::abs(m->hmin - hfine) < 1e-10 * hfine,
          "hmin is the fine plateau spacing");

    const auto g2 = Grid::make(32, 32, 1, 4.0, 4.0, 1.0);
    const AxisRemap3 A2{A.fx, A.fy, Focal1D::identity(1.0)};
    check(bool(CurvMetrics::build(CurvMesh::fromRemap(g2, A2))),
          "nz == 1 stays planar with x/y focal");
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
    stretchedRemap();
    focalRemap();
    rejection();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
