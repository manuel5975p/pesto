// Body-fitted O-grid: per-axis node continuation (Cyclic/Extend), the exact
// polar annulus for a circular contour, finite-volume metric validity on a
// non-Translate mesh (closure, positive volumes, analytic areas), the exact
// uniform limit at beta = 0, the wall-mirror inner ghost row, planarity and
// pure extrusion in z, contour rejection, and a non-circular contour.
#include "dns/ogrid.hpp"

#include "dns/bodies.hpp"
#include "dns/curvmesh.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <format>
#include <string>
#include <numbers>
#include <vector>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

// --- helpers ---------------------------------------------------------------

// The exact bilinear-quad area vector of one cell face, the same formula
// CurvMetrics::build uses -- recomputed here so the per-cell closure can be
// checked on faces the S tables do not store (the lo face of an Extend axis).
std::array<double, 3> faceS(const CurvMesh& m, int i, int j, int k, int a,
                            int off)
{
    const int t1 = (a + 1) % 3, t2 = (a + 2) % 3;
    int o[3] = {0, 0, 0};
    o[a] = off;
    auto corner = [&](int d1, int d2) {
        int e[3] = {o[0], o[1], o[2]};
        e[t1] += d1;
        e[t2] += d2;
        return m.node(i + e[0], j + e[1], k + e[2]);
    };
    const auto P0 = corner(0, 0), P1 = corner(1, 0), P2 = corner(1, 1),
               P3 = corner(0, 1);
    const double u[3] = {P2[0] - P0[0], P2[1] - P0[1], P2[2] - P0[2]};
    const double v[3] = {P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]};
    return {0.5 * (u[1] * v[2] - u[2] * v[1]), 0.5 * (u[2] * v[0] - u[0] * v[2]),
            0.5 * (u[0] * v[1] - u[1] * v[0])};
}

// Stretched circle: level sets are exact ellipses, gradient direction exact.
Sdf ellipseSdf(double cx, double cy, double a, double b)
{
    return [=](double x, double y, double) {
        return (std::hypot((x - cx) / a, (y - cy) / b) - 1.0) * std::min(a, b);
    };
}

// Two disjoint disks: the ray through both crosses the surface three times.
Sdf twoDisksSdf()
{
    return [](double x, double y, double) {
        return std::min(std::hypot(x, y) - 0.5, std::hypot(x - 2.0, y) - 0.4);
    };
}

// --- part 1: per-axis node continuation -------------------------------------

void axisWrapModes()
{
    const auto g = Grid::make(8, 6, 4, 2.0, 1.2, 0.6);
    const auto u = CurvMesh::sinusoidal(g, 0.05);
    check(u.wrap[0] == AxisWrap::Translate && u.wrap[1] == AxisWrap::Translate &&
              u.wrap[2] == AxisWrap::Translate,
          "a mesh defaults to Translate on every axis");

    bool translate = true;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const auto p = u.node(i, j, 1), q = u.node(i + g.nx, j - g.ny, 1);
            if (q[0] != p[0] + g.nx * g.hx || q[1] != p[1] - g.ny * g.hy ||
                q[2] != p[2])
                translate = false;
        }
    check(translate, "Translate images add exactly one box length (bit-exact)");

    auto c = u;
    c.wrap = {AxisWrap::Cyclic, AxisWrap::Extend, AxisWrap::Translate};
    bool cyc = true, ext = true;
    for (int j = 0; j < g.ny; ++j) {
        const auto p = c.node(0, j, 2), q = c.node(g.nx, j, 2);
        if (q != p) cyc = false;
    }
    for (int i = 0; i < g.nx; ++i) {
        const auto top = c.node(i, g.ny, 2), t1 = c.node(i, g.ny - 1, 2),
                   t2 = c.node(i, g.ny - 2, 2);
        const auto bot = c.node(i, -1, 2), b0 = c.node(i, 0, 2),
                   b1 = c.node(i, 1, 2);
        for (int d = 0; d < 3; ++d) {
            if (std::abs(top[d] - (2 * t1[d] - t2[d])) > 1e-14) ext = false;
            if (std::abs(bot[d] - (2 * b0[d] - b1[d])) > 1e-14) ext = false;
        }
    }
    check(cyc, "Cyclic closes the axis exactly: node(n) == node(0)");
    check(ext, "Extend continues linearly off both ends");

    auto bad = CurvMesh::uniform(Grid::make(8, 6, 1, 2.0, 1.2, 1.0));
    bad.wrap[2] = AxisWrap::Extend;
    check(!CurvMetrics::build(bad), "Extend on a degenerate axis is rejected");
}

// --- the reference O-grid used by most checks -------------------------------

constexpr int kNx = 64, kNy = 34;
constexpr double kR = 0.5, kRFar = 5.0, kSpan = 1.0, kCx = 0.3, kCy = -0.2;

OGridSpec circleSpec(double beta = 0.0)
{
    return {.cx = kCx, .cy = kCy, .cz = 0.0, .rFar = kRFar, .beta = beta,
            .span = kSpan};
}

void circleExact()
{
    const auto g = ogridGrid(kNx, kNy, 1, kR, kRFar, kSpan);
    const auto og =
        buildOGrid(g, sdfOf(CylinderZ{kCx, kCy, kR}), circleSpec());
    if (!og) {
        check(false, std::string("circle O-grid builds: ") + og.error());
        return;
    }
    double errNode = 0.0, errWall = 0.0;
    for (int j = 1; j < kNy; ++j) {
        const double t = double(j - 1) / (kNy - 2);
        const double rad = kR + ogridRadialDist(t, 0.0) * (kRFar - kR);
        for (int i = 0; i < kNx; ++i) {
            const double th = -2.0 * pi * i / kNx;
            const auto p = og->mesh.node(i, j, 0);
            errNode = std::max({errNode,
                                std::abs(p[0] - (kCx + rad * std::cos(th))),
                                std::abs(p[1] - (kCy + rad * std::sin(th)))});
        }
    }
    for (int i = 0; i < kNx; ++i) {
        errWall = std::max(errWall, std::abs(og->rWall[i] - kR));
        const double th = -2.0 * pi * i / kNx;
        errWall = std::max({errWall,
                            std::abs(og->wallNormal[i][0] - std::cos(th)),
                            std::abs(og->wallNormal[i][1] - std::sin(th)),
                            std::abs(og->wallNormal[i][2])});
    }
    std::printf("circle O-grid: node err %.3e, wall radius/normal err %.3e, "
                "h1 %.6f, growth %.4f, normal dev %.2e rad\n",
                errNode, errWall, og->h1, og->growth, og->normalDevMax);
    check(errNode < 1e-12, "circle O-grid nodes are the exact polar annulus");
    check(errWall < 1e-12, "wall radius and normal are exact for a circle");
    check(std::abs(og->rWallMin - kR) < 1e-12 &&
              std::abs(og->rWallMax - kR) < 1e-12,
          "reported wall radius bracket is the circle radius");
    check(og->normalDevMax < 1e-7, // acos loses half the mantissa near 1
          "radial edges are exactly wall-normal on a circle");
}

void metricsAndClosure()
{
    const auto g = ogridGrid(kNx, kNy, 4, kR, kRFar, kSpan);
    const auto og =
        buildOGrid(g, sdfOf(CylinderZ{kCx, kCy, kR}), circleSpec(2.0));
    if (!og) {
        check(false, std::string("clustered O-grid builds: ") + og.error());
        return;
    }
    const auto m = CurvMetrics::build(og->mesh);
    if (!m) {
        check(false, std::string("O-grid metrics build: ") + m.error());
        return;
    }
    check(m->wrap[0] == AxisWrap::Cyclic && m->wrap[1] == AxisWrap::Extend &&
              m->wrap[2] == AxisWrap::Translate,
          "metrics carry the mesh's per-axis wrap");

    double close = 0.0, smax = 0.0, vmin = 1e300, tableClose = 0.0, fluid = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                vmin = std::min(vmin, m->vol[c]);
                if (j >= 1 && j <= g.ny - 2) fluid += m->vol[c];
                for (int cx = 0; cx < 3; ++cx) {
                    double raw = 0.0, tab = 0.0;
                    for (int a = 0; a < 3; ++a) {
                        const auto hi = faceS(og->mesh, i, j, k, a, 1);
                        const auto lo = faceS(og->mesh, i, j, k, a, 0);
                        raw += hi[cx] - lo[cx];
                        smax = std::max(smax, std::abs(hi[cx]));
                        const std::size_t nb =
                            g.idx(i - (a == 0), j - (a == 1), k - (a == 2));
                        tab += m->S[a][cx][c] - m->S[a][cx][nb];
                    }
                    close = std::max(close, std::abs(raw));
                    if (j >= 1) tableClose = std::max(tableClose, std::abs(tab));
                }
            }
    const double dth = 2.0 * pi / kNx;
    const double poly =
        0.5 * kNx * std::sin(dth) * (kRFar * kRFar - kR * kR) * kSpan;
    const double disc = pi * (kRFar * kRFar - kR * kR) * kSpan;
    std::printf("O-grid closure %.2e (max |S| %.2e), table closure %.2e, "
                "min vol %.3e\nfluid volume %.12g vs polygon %.12g "
                "(annulus %.12g, rel %.2e)\n",
                close, smax, tableClose, vmin, fluid, poly, disc,
                std::abs(fluid - disc) / disc);
    check(close < 1e-14, "per-cell face-vector closure is exact on Cyclic/Extend");
    check(tableClose < 1e-14,
          "the stored S tables close for every cell with a lo neighbour");
    check(vmin > 0.0, "every cell volume is positive, ghost rows included");
    check(std::abs(fluid - poly) < 1e-12 * poly,
          "fluid cells tile the polygonal annulus exactly");
    check(std::abs(fluid - disc) < dth * dth * disc,
          "the polygonal annulus is the true annulus to O(dtheta^2)");
}

void uniformLimit()
{
    bool bitwise = true;
    for (int i = 0; i <= 64; ++i) {
        const double t = double(i) / 64;
        if (ogridRadialDist(t, 0.0) != t) bitwise = false;
    }
    check(bitwise, "beta = 0 radial distribution is t bit-for-bit");
    check(ogridRadialDist(0.0, 3.0) == 0.0 && ogridRadialDist(1.0, 3.0) == 1.0,
          "the clustered distribution hits both ends exactly");

    const auto g = ogridGrid(kNx, kNy, 1, kR, kRFar, kSpan);
    const auto sdf = sdfOf(CylinderZ{kCx, kCy, kR});
    const auto u = buildOGrid(g, sdf, circleSpec());
    const auto c = buildOGrid(g, sdf, circleSpec(2.5));
    if (!u || !c) {
        check(false, "uniform and clustered O-grids build");
        return;
    }
    const double h = (kRFar - kR) / (kNy - 2);
    double sp = 0.0;
    for (int j = 1; j <= kNy - 2; ++j) {
        const auto a = u->mesh.node(0, j, 0), b = u->mesh.node(0, j + 1, 0);
        sp = std::max(sp, std::abs(std::hypot(b[0] - a[0], b[1] - a[1]) - h));
    }
    std::printf("beta = 0 radial spacing err %.2e (h %.6f); clustered h1 %.6f "
                "growth %.4f\n",
                sp, h, c->h1, c->growth);
    check(sp < 1e-13, "beta = 0 gives uniform radial spacing");
    check(std::abs(u->h1 - h) < 1e-13 && std::abs(u->growth - 1.0) < 1e-13,
          "beta = 0 reports the uniform first cell and unit growth");
    check(c->h1 < 0.25 * h && c->growth > 1.0,
          "beta > 0 clusters toward the wall with a bounded growth ratio");
}

void ghostMirrorAndExtrusion()
{
    const auto g = ogridGrid(kNx, kNy, 4, kR, kRFar, kSpan);
    const auto og =
        buildOGrid(g, ellipseSdf(kCx, kCy, 0.6, 0.35), circleSpec(2.0));
    if (!og) {
        check(false, std::string("ellipse O-grid builds: ") + og.error());
        return;
    }
    double mid = 0.0, extr = 0.0, dz = 0.0;
    for (int k = 0; k < g.nz; ++k)
        for (int i = 0; i < kNx; ++i) {
            const auto p0 = og->mesh.node(i, 0, k), p1 = og->mesh.node(i, 1, k),
                       p2 = og->mesh.node(i, 2, k);
            for (int d = 0; d < 2; ++d)
                mid = std::max(mid, std::abs(0.5 * (p0[d] + p2[d]) - p1[d]));
            const auto q = og->mesh.node(i, 1, 0);
            extr = std::max({extr, std::abs(p1[0] - q[0]), std::abs(p1[1] - q[1])});
            dz = std::max(dz, std::abs(p1[2] - (-0.5 * kSpan + k * g.hz)));
        }
    std::printf("ghost midpoint err %.2e, extrusion err %.2e, z err %.2e\n", mid,
                extr, dz);
    check(mid < 1e-14, "the wall node row is exactly midway between ghost and "
                       "first fluid row");
    check(extr < 1e-15, "nz > 1 is a pure extrusion (x, y independent of k)");
    check(dz < 1e-15, "spanwise nodes are the uniform layer offsets");

    const auto g1 = ogridGrid(kNx, kNy, 1, kR, kRFar, kSpan);
    const auto flat =
        buildOGrid(g1, ellipseSdf(kCx, kCy, 0.6, 0.35), circleSpec(2.0));
    check(flat && CurvMetrics::build(flat->mesh).has_value(),
          "nz == 1 stays planar and its metrics build");
}

void nonCircular()
{
    struct Case
    {
        const char* name;
        Sdf sdf;
    };
    const Case cases[] = {
        {"ellipse 0.6 x 0.35", ellipseSdf(kCx, kCy, 0.6, 0.35)},
        {"lens (sharp tips)", sdfOf(Lens{kCx, kCy, 0.6, 0.25})},
    };
    for (const auto& cs : cases) {
        const auto g = ogridGrid(kNx, kNy, 1, kR, kRFar, kSpan);
        const auto og = buildOGrid(g, cs.sdf, circleSpec(2.0));
        if (!og) {
            check(false, std::string(cs.name) + " builds: " + og.error());
            continue;
        }
        const auto m = CurvMetrics::build(og->mesh);
        if (!m) {
            check(false, std::string(cs.name) + " metrics: " + m.error());
            continue;
        }
        const double vmin = *std::ranges::min_element(m->vol);
        const double deg = og->normalDevMax * 180.0 / pi;
        std::printf("%s: rw [%.6f, %.6f], h1 %.6f (max %.6f), growth %.4f, "
                    "wall-normal dev %.3f deg, min vol %.3e\n",
                    cs.name, og->rWallMin, og->rWallMax, og->h1, og->h1Max,
                    og->growth, deg, vmin);
        check(vmin > 0.0, std::string(cs.name) + ": no folded cells");
        check(deg < 2.0,
              std::string(cs.name) + ": near-wall cells within 2 deg of normal");
        check(og->growth < 1.35,
              std::string(cs.name) + ": adjacent radial growth stays under 1.35");
    }
}

void rejection()
{
    const auto g = ogridGrid(kNx, kNy, 1, kR, kRFar, kSpan);
    auto fails = [&](const char* what, const Grid& gg, const Sdf& s,
                     const OGridSpec& sp) {
        const auto r = buildOGrid(gg, s, sp);
        check(!r, what);
        if (!r) std::printf("      %s\n", r.error().c_str());
    };
    fails("a body that is not star-shaped about the centre is rejected", g,
          twoDisksSdf(), circleSpec());
    fails("a body larger than rFar is rejected", g,
          sdfOf(CylinderZ{kCx, kCy, 2.0 * kRFar}), circleSpec());
    fails("a centre outside the body is rejected", g,
          sdfOf(CylinderZ{kCx + 3.0, kCy, kR}), circleSpec());
    fails("a z-varying contour is rejected", g,
          [](double x, double y, double z) {
              return std::hypot(x - kCx, y - kCy) - (kR + 0.05 * z);
          },
          circleSpec());
    fails("a grid whose angular axis is not 2 pi is rejected",
          Grid::make(kNx, kNy, 1, 6.0, 1.0, kSpan),
          sdfOf(CylinderZ{kCx, kCy, kR}), circleSpec());
    fails("a grid whose spanwise axis misses span is rejected",
          Grid::make(kNx, kNy, 1, 2.0 * pi, 1.0, 2.0 * kSpan),
          sdfOf(CylinderZ{kCx, kCy, kR}), circleSpec());
    fails("too few radial rows is rejected",
          Grid::make(kNx, 3, 1, 2.0 * pi, 1.0, kSpan),
          sdfOf(CylinderZ{kCx, kCy, kR}), circleSpec());
    fails("too few angular cells is rejected",
          Grid::make(2, kNy, 1, 2.0 * pi, 1.0, kSpan),
          sdfOf(CylinderZ{kCx, kCy, kR}), circleSpec());
    fails("a negative clustering beta is rejected", g,
          sdfOf(CylinderZ{kCx, kCy, kR}),
          OGridSpec{.cx = kCx, .cy = kCy, .cz = 0.0, .rFar = kRFar,
                    .beta = -1.0, .span = kSpan});
}

} // namespace

// The validator's flags-only O-grid numbers: the nominal-circle h1 it prints
// IS the built mesh's first cell for a cylinder, and the boundary-layer
// warning counts cells through the clustered radial distribution, not the
// nominal hy (which read "0.0 cells" at Re 1e5 before).
void validatorNumbers()
{
    auto c = dnstest::runCli({"flow", "--physics", "compressible", "--body",
                              "cylinder", "--ogrid", "--n", "64", "--ny", "96",
                              "--ogrid-far", "5", "--re", "100", "--dry-run"});
    check(c.has_value(), "o-grid case validates");
    if (!c) return;
    const double R = 0.5 * c->placement.D;
    const auto og = buildOGrid(c->grid, sdfOf(CylinderZ{0.0, 0.0, R}),
                               OGridSpec{0.0, 0.0, 0.0, c->ogridRFar,
                                         c->ogridBeta, c->ogridSpan});
    check(og.has_value(), "o-grid of the validated case builds");
    if (og) {
        const double rel = std::abs(c->ogridH1 - og->h1) / og->h1;
        std::printf("o-grid h1: validator %.12e, mesh %.12e (rel %.1e)\n",
                    c->ogridH1, og->h1, rel);
        check(rel < 1e-9, "dry-run h1 equals buildOGrid(...).h1 (cylinder)");
    }
    check(c->describe().find("o-grid outer row") != std::string::npos,
          "describe names the o-grid outer row, not nscbc");

    auto hi = dnstest::runCli({"flow", "--physics", "compressible", "--body",
                               "cylinder", "--ogrid", "--re", "1e5", "--n",
                               "512", "--ny", "256", "--dry-run"});
    check(hi.has_value(), "o-grid Re 1e5 validates (warning only)");
    if (!hi) return;
    const double D = hi->placement.D;
    const double cells = ogridRadialCells(D / std::sqrt(1e5), 256, 0.5 * D,
                                          hi->ogridRFar, hi->ogridBeta);
    // Bisection against the closed form: D(t*) is the target length.
    const double tStar = cells / 254.0;
    const double back = ogridRadialDist(tStar, hi->ogridBeta) *
                        (hi->ogridRFar - 0.5 * D);
    std::printf("o-grid Re 1e5: BL %.4f cells (D(t*) %.6e vs %.6e)\n", cells,
                back, D / std::sqrt(1e5));
    check(cells > 0.0, "BL cell count is positive");
    check(std::abs(back - D / std::sqrt(1e5)) < 1e-12 * D,
          "bisection solves D(t*) = (D/sqrt Re)/(rFar - D/2)");
    const std::string want = std::format(
        "boundary layer is {:.1f} cells (D/sqrt(Re)/h); this is "
        "under-resolved for DNS", cells);
    check(std::ranges::find(hi->warnings, want) != hi->warnings.end(),
          "BL warning quotes the clustered-distribution count");
    check(std::ranges::find(hi->warnings,
                            std::string("o-grid far row is unmeasured above "
                                        "Re 800 (the NSCBC Re-800 rule does "
                                        "not apply to it)")) !=
              hi->warnings.end(),
          "Re > 800 on the o-grid warns");
}

// The revolved wedge (buildRevolvedOGrid, masterplan_flowrans section 4.5):
// degenerate axis faces with their limit metric, positive volumes everywhere
// (the past-axis placeholder columns take their neighbour's), the wedge's
// side faces exact z-mirrors, MirrorZ's one node past the top.
void revolvedWedge()
{
    std::printf("--- revolved wedge ---\n");
    const int nT = 32, ny = 20;
    const double R = 0.5, rFar = 6.0, dphi = 5.0 * pi / 180.0;
    const RevolvedSpec rv{.wedge = true, .dphi = dphi};
    const Grid g = ogridRevolvedGrid(nT, ny, rv, R, rFar);
    const Sdf sphere = [=](double x, double y, double z) {
        return std::sqrt(x * x + y * y + z * z) - R;
    };
    auto og = buildRevolvedOGrid(g, sphere, {.rFar = rFar, .beta = 2.0}, rv);
    if (!og) { check(false, "revolved build: " + og.error()); return; }
    check(g.nx == nT + 2 && g.nz == 3 && og->nTheta == nT && og->revolved,
          "wedge grid: nx = nTheta + 2, nz = 3");
    const auto& m = og->mesh;
    // Nodes on the axis: rho == 0 to the bit.
    bool onAxis = true;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < ny; ++j)
            for (const int i : {1, nT + 1}) {
                const auto p = m.node(i, j, k);
                onAxis = onAxis && p[1] == 0.0 && p[2] == 0.0;
            }
    check(onAxis, "axis node columns sit at rho == 0 exactly");
    bool mirror = true;
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= g.nx; ++i) {
            const auto a = m.node(i, j, 0), b = m.node(i, j, 3);
            const auto c = m.node(i, j, 1), d = m.node(i, j, 2);
            mirror = mirror && a[0] == b[0] && a[1] == b[1] && a[2] == -b[2] &&
                     c[0] == d[0] && c[1] == d[1] && c[2] == -d[2];
        }
    check(mirror, "MirrorZ: node(3) == z-mirror of node(0); planes 1 and 2 "
                  "exact mirrors");
    auto mt = CurvMetrics::build(m);
    if (!mt) { check(false, "revolved metrics: " + mt.error()); return; }
    bool zeroS = true, rows = true, limit = true, pos = true, sides = true;
    double worstLim = 0;
    for (int k = 0; k < 3; ++k)
        for (int j = 0; j < ny; ++j) {
            for (const int i : {0, nT}) {
                const std::size_t c = g.idx(i, j, k);
                zeroS = zeroS && mt->S[0][0][c] == 0.0 &&
                        mt->S[0][1][c] == 0.0 && mt->S[0][2][c] == 0.0;
                for (int e = 3; e < 9; ++e) rows = rows && mt->Ji[0][e][c] == 0.0;
                if (k != 1 || j == 0 || j == ny - 1) continue;
                // Row i on the real plane (phi = 0 centroid): +-y^/(2 rho_F).
                const int f = i == 0 ? 1 : nT;
                double rc = 0;
                for (int dk = 0; dk < 2; ++dk)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di)
                            rc += 0.125 * m.node(f + di, j + dj, k + dk)[1];
                const double want = (i == 0 ? 1.0 : -1.0) / (2.0 * rc);
                const double jx = mt->Ji[0][0][c], jy = mt->Ji[0][1][c],
                             jz = mt->Ji[0][2][c];
                worstLim = std::max({worstLim, std::abs(jy - want) / std::abs(want),
                                     std::abs(jx) / std::abs(want),
                                     std::abs(jz) / std::abs(want)});
            }
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                pos = pos && mt->vol[c] > 0.0;
                if (k != 0) continue;
                const std::size_t c1 = g.idx(i, j, 1);
                sides = sides && mt->S[2][0][c1] == -mt->S[2][0][c] &&
                        mt->S[2][1][c1] == -mt->S[2][1][c] &&
                        mt->S[2][2][c1] == mt->S[2][2][c];
            }
        }
    limit = worstLim < 1e-12;
    std::printf("axis limit row vs +-y^/(2 rho_F): worst rel %.2e\n", worstLim);
    check(zeroS, "axis faces: S == (0, 0, 0) exactly");
    check(rows, "axis faces: Ji rows j and k == 0");
    check(limit, "axis faces: row i is +-y^/(2 rho_F) on the real plane");
    check(pos, "every volume > 0, placeholder columns and ghost planes "
               "included");
    check(sides, "the wedge's two side faces are exact z-mirrors (bitwise)");
    // An sdf that is not a body of revolution about x is rejected.
    const Sdf ell = [=](double x, double y, double z) {
        return std::sqrt(x * x + y * y + 4 * z * z) - R;
    };
    auto bad = buildRevolvedOGrid(g, ell, {.rFar = rFar, .beta = 2.0}, rv);
    check(!bad && bad.error().find("body of revolution") != std::string::npos,
          "a non-axisymmetric sdf is rejected by name");
}

int main()
{
    revolvedWedge();
    axisWrapModes();
    circleExact();
    metricsAndClosure();
    uniformLimit();
    ghostMirrorAndExtrusion();
    nonCircular();
    rejection();
    validatorNumbers();
    return dnstest::finish();
}
