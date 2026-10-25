// A moving ghost wall in a gas that moves WITH it is a fixed point.
//
// Uniform gas at velocity U and a body whose wall velocity is the same
// constant U everywhere (a translating body in its co-moving gas): the image
// u_ghost = (1+a) U - a u_sample = U for every a, so every ghost holds the
// freestream, the flux divergence of a uniform state is zero, and nothing may
// move. The only admissible residue is FP32 rounding of (1+a) U - a U.
// Checked after 50 steps: max|u - U| over every non-deep cell within 3x the
// round-off floor of the same rig with the wall at rest, and the tallied force
// < 1e-6 pInf A (A = tallied area) -- uniform pressure on a closed surface.
// 2D cylinder and 3D sphere, off-lattice centres.
//
// And a SPINNING wall in gas that co-rotates with it, where the wall
// accelerates. A swirl u_theta(r) = w r inside R1, blended C1 to rest by R2,
// rho uniform, mu = 0, p from the radial balance dp/dr = rho u_theta^2 / r, is
// an exact steady Euler state, and a cylinder of radius R < R1 concentric with
// it, spinning at w, is a wall it satisfies: no slip (u = w R there) and the
// wall-normal momentum balance dp/dn = rho w^2 R = -rho a_wall . n. One step
// from it measures the wall closure's residual: max over the fluid cells
// within 3 cells of the wall of |d(rho u)/dt|, over rho w^2 R. With the wall's
// momentum balance in the pressure image that is the boundary cell's local
// truncation, O(h); with the zero-gradient copy (diagnostic: the wall
// acceleration withheld) it is O(1), h-independent -- the first fluid row is
// short half a cell of the cyclostrophic gradient. Both walls that move: the
// Cartesian ghost (direct rotor) and the O-grid wall row (--rotor-frame mrf,
// zone f = 1 everywhere).
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/rotor.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/rotortable.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

constexpr double gam = 1.4, rhoInf = 1.0;
const double pInf = 1.0 / (gam * 0.04); // Mach 0.2 on unit speed

struct Drift
{
    double du = 0.0; // max |u - U| over non-deep cells, absolute
    double fr = 0.0; // |F| / (pInf A)
};

// 50 steps of the uniform gas at U against the wall moving at U (moving ==
// false: the RESTING wall, which needs U == 0 to be a fixed point).
Drift run(const std::string& tag, const Grid& g, const gpu::GhostSdf& sdf,
          std::array<double, 3> U, bool moving, bool verbose)
{
    const gpu::GhostWallMotion uw = [U](double, double, double) {
        return gpu::GhostWallKinematics{U, {0.0, 0.0, 0.0}};
    };
    const double mu = 0.02;
    auto dev = gpu::DeviceComp::make(g, gam, mu);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, gam, rhoInf, pInf, nullptr, {},
                                         moving ? &uw : nullptr);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) { check(false, tag + ": setup"); return {}; }
    if (moving) check(gh->moving(), tag + ": wall is moving");
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });

    const double u2 = U[0] * U[0] + U[1] * U[1] + U[2] * U[2];
    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.mx[c] = rhoInf * U[0];
        host.my[c] = rhoInf * U[1];
        host.mz[c] = rhoInf * U[2];
        host.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * u2;
    }
    (void)ds->upload(host);
    gh->apply(*ds);
    const double dt = dev->stableDt(*ds, *red);
    for (int s = 0; s < 50; ++s) dev->step(*ds, gpu::Real(dt));
    gh->apply(*ds);
    (void)ds->download(host);

    std::vector<char> deep(g.size(), 0);
    for (int c : gh->deepCells()) deep[std::size_t(c)] = 1;
    double du = 0.0, dp = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c) {
        if (deep[c]) continue;
        const double r = host.rho[c];
        const double u = host.mx[c] / r, v = host.my[c] / r, w = host.mz[c] / r;
        du = std::max({du, std::abs(u - U[0]), std::abs(v - U[1]),
                       std::abs(w - U[2])});
        const double p = (gam - 1.0) * (host.E[c] - 0.5 * r * (u * u + v * v + w * w));
        dp = std::max(dp, std::abs(p - pInf) / pInf);
    }
    const auto T = gpu::classifyCompGhost(g, sdf, nullptr,
                                          {Focal1D::identity(g.nx * g.hx),
                                           Focal1D::identity(g.ny * g.hy),
                                           Focal1D::identity(g.nz * g.hz)},
                                          nullptr);
    const double area[3] = {g.hy * g.hz, g.hx * g.hz, g.hx * g.hy};
    double A = 0.0;
    for (int d : T.fdir) A += area[d / 2];
    const auto F = gh->force(*dev, *ds);
    const double fr = std::sqrt(F.fx * F.fx + F.fy * F.fy + F.fz * F.fz) / (pInf * A);
    if (verbose)
        std::printf("%s: 50 steps dt %.3e: max|u-U| %.2e max|p-pInf|/pInf %.2e "
                    "|F|/(pInf A) %.2e (A %.4f, %d faces)\n",
                    tag.c_str(), dt, du, dp, fr, A, T.nFace());
    return {du, fr};
}

// The co-moving checks. The exact answer is u == U and F == 0; the only
// admissible residue is FP32 round-off, and that floor is MEASURED rather than
// assumed: the resting wall in resting gas is also an exact fixed point, so
// the moving image may add nothing beyond the noise the resting wall already
// makes. A fixed absolute bound cannot be set below it -- pInf / |U|^2 ~ 18
// here, so one ulp of E is ~4e-7 of velocity through p, and 50 steps measured
// 2.2e-6 (2D) / 5.4e-6 (3D) with the wall at rest.
void comoving(const std::string& tag, const Grid& g, const gpu::GhostSdf& sdf,
              std::array<double, 3> U)
{
    const double Umag = std::sqrt(U[0] * U[0] + U[1] * U[1] + U[2] * U[2]);
    const Drift m = run(tag + " co-moving", g, sdf, U, true, true);
    const Drift r = run(tag + " resting floor", g, sdf, {0, 0, 0}, false, true);
    std::printf("%s: max|u-U|/|U| %.2e; resting-wall round-off floor "
                "max|u| %.2e -> co-moving/floor %.2f\n",
                tag.c_str(), m.du / Umag, r.du, m.du / std::max(r.du, 1e-300));
    check(m.du < 3.0 * std::max(r.du, 1e-9),
          tag + ": co-moving drift within 3x the resting-wall round-off floor");
    check(m.fr < 1e-6, tag + ": no force on the co-moving body (1e-6 pInf A)");
}

// ---- the spinning wall in co-rotating gas -----------------------------------

// The swirl about the origin: w r inside r1, C1 smoothstep to rest at r2;
// p(r) = pInf - int_r^r2 rho u^2/r' dr', tabulated (1e5 intervals, linear:
// error ~1e-11 of rho w^2 R^2).
struct Swirl
{
    double w, r1, r2;
    std::vector<double> P; // p - pInf on [0, r2]
    double ut(double r) const
    {
        if (r >= r2) return 0.0;
        if (r <= r1) return w * r;
        const double s = (r - r1) / (r2 - r1);
        return w * r * (1.0 - s * s * (3.0 - 2.0 * s));
    }
    static Swirl make(double w, double r1, double r2)
    {
        Swirl sw{w, r1, r2, {}};
        const int n = 100000;
        sw.P.assign(n + 1, 0.0);
        const double dr = r2 / n;
        for (int k = n - 1; k >= 0; --k) {
            const double ra = k * dr, rb = (k + 1) * dr, rm = 0.5 * (ra + rb);
            auto f = [&](double r) { return r > 0 ? rhoInf * sw.ut(r) * sw.ut(r) / r : 0.0; };
            sw.P[k] = sw.P[k + 1] - dr * (f(ra) + 4 * f(rm) + f(rb)) / 6.0;
        }
        return sw;
    }
    double p(double r) const
    {
        if (r >= r2) return pInf;
        const double t = r / r2 * double(P.size() - 1);
        const std::size_t k = std::min(std::size_t(t), P.size() - 2);
        return pInf + P[k] + (t - double(k)) * (P[k + 1] - P[k]);
    }
    void put(CompState& h, std::size_t c, double x, double y) const
    {
        const double r = std::hypot(x, y), u = r > 0 ? ut(r) / r : 0.0;
        const double vx = -u * y, vy = u * x;
        h.rho[c] = rhoInf;
        h.mx[c] = rhoInf * vx;
        h.my[c] = rhoInf * vy;
        h.mz[c] = 0.0;
        h.E[c] = p(r) / (gam - 1.0) + 0.5 * rhoInf * (vx * vx + vy * vy);
    }
};

// max |d(rho u)/dt| / (rho w^2 R) over the cells flagged `near`, one step.
double residual(const CompState& a, const CompState& b, const std::vector<char>& near,
                double dt, double scale)
{
    double r = 0.0;
    for (std::size_t c = 0; c < near.size(); ++c)
        if (near[c])
            r = std::max(r, std::hypot(b.mx[c] - a.mx[c], b.my[c] - a.my[c]) / dt);
    return r / scale;
}

// Cartesian ghost wall, direct rotor: n x n box of side 2 pi, R = L/16.
double swirlGhost(int n, bool dropAccel)
{
    const double L = 2 * pi, h = L / n, R = L / 16, w = 0.5 / R;
    const double cx = L / 2 + 0.37 * h, cy = L / 2 + 0.61 * h;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const Swirl sw = Swirl::make(w, 2 * R, 4 * R);
    const gpu::GhostSdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - R;
    };
    const Rotor rot{.axis = RotorAxis::Z, .shaft = {cx, cy, 0.5}, .omega = w,
                    .radius = R, .bins = 1, .thicken = 0};
    const gpu::GhostWallMotion full = gpu::rotorWallMotion(rot);
    const gpu::GhostWallMotion motion =
        dropAccel ? gpu::GhostWallMotion([full](double x, double y, double z) {
            auto k = full(x, y, z);
            k.a = {0.0, 0.0, 0.0};
            return k;
        })
                  : full;
    auto dev = gpu::DeviceComp::make(g, gam, 0.0);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, gam, rhoInf, pInf, nullptr, {}, &motion);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) { check(false, "swirlGhost setup"); return 0; }
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
    auto a = CompState::zeros(g);
    std::vector<char> near(g.size(), 0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const double x = (i + 0.5) * h - cx, y = (j + 0.5) * h - cy;
            sw.put(a, g.idx(i, j, 0), x, y);
            const double d = std::hypot(x, y) - R;
            near[g.idx(i, j, 0)] = d > 0.0 && d < 3 * h;
        }
    (void)ds->upload(a);
    gh->apply(*ds);
    (void)ds->download(a);
    const double dt = dev->stableDt(*ds, *red);
    dev->step(*ds, gpu::Real(dt));
    auto b = CompState::zeros(g);
    (void)ds->download(b);
    return residual(a, b, near, dt, rhoInf * w * w * R);
}

// O-grid wall row, --rotor-frame mrf with f = 1 everywhere: R, rFar = 5 R,
// uniform radial spacing h1 = R / (16 lvl), nx = 100 lvl.
double swirlOGrid(int lvl, bool dropAccel)
{
    const double R = 1.0, rFar = 5.0 * R, w = 0.5 / R;
    const int nx = 100 * lvl, ny = 2 + 64 * lvl;
    const auto g = ogridGrid(nx, ny, 1, R, rFar, 1.0);
    auto og = buildOGrid(g, [R](double x, double y, double) { return std::hypot(x, y) - R; },
                         {.cx = 0, .cy = 0, .cz = 0, .rFar = rFar, .beta = 0, .span = 1.0});
    if (!og) { check(false, "swirlOGrid build: " + og.error()); return 0; }
    auto m = CurvMetrics::build(og->mesh);
    if (!m) { check(false, "swirlOGrid metrics"); return 0; }
    const Swirl sw = Swirl::make(w, 1.5 * R, 3.5 * R);
    auto dev = gpu::DeviceComp::make(*m, gam, 0.0);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !red) { check(false, "swirlOGrid alloc"); return 0; }
    gpu::MrfZone z;
    z.axis = 2;
    z.omega = w;
    z.r1 = 1e30; z.r2 = 2e30; z.a1 = 1e30; z.a2 = 2e30;
    if (dev->setMrfCurv(z, og->mesh) != hipSuccess) { check(false, "swirlOGrid mrf"); return 0; }
    gpu::DeviceOGridBc::Config bc;
    bc.gamma = gam;
    bc.rhoInf = rhoInf;
    bc.uInf = 0.0;
    bc.pInf = pInf;
    bc.omega = w;
    bc.wallPos = ogridWallMidpoints(*og);
    bc.wallSpan = ogridWallSpans(*og);
    if (dropAccel)
        for (auto& d : bc.wallSpan) d = {0.0, 0.0};
    auto wall = gpu::DeviceOGridBc::make(g, bc);
    if (!wall) { check(false, "swirlOGrid bc"); return 0; }
    dev->setStageHook([&](gpu::DeviceCompState& s) { wall->apply(s); });
    auto a = CompState::zeros(g);
    std::vector<char> near(g.size(), 0);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            double x = 0, y = 0;
            for (int dj = 0; dj < 2; ++dj)
                for (int di = 0; di < 2; ++di) {
                    const auto p = og->mesh.node(i + di, j + dj, 0);
                    x += 0.25 * p[0];
                    y += 0.25 * p[1];
                }
            sw.put(a, g.idx(i, j, 0), x, y);
            near[g.idx(i, j, 0)] = j >= 1 && j <= 3;
        }
    (void)ds->upload(a);
    wall->apply(*ds);
    (void)ds->download(a);
    const double dt = dev->stableDt(*ds, *red);
    dev->step(*ds, gpu::Real(dt));
    auto b = CompState::zeros(g);
    (void)ds->download(b);
    return residual(a, b, near, dt, rhoInf * w * w * R);
}

template <class Rig>
void spinning(const std::string& tag, Rig rig, int lo, int hi)
{
    const double r1 = rig(lo, false), r2 = rig(hi, false);
    const double c1 = rig(lo, true), c2 = rig(hi, true);
    std::printf("%s spinning in co-rotating gas: wall residual |d(rho u)/dt|/(rho w^2 R) "
                "%.4e -> %.4e (order %.2f) | copy %.4e -> %.4e (order %.2f)\n",
                tag.c_str(), r1, r2, std::log2(r1 / r2), c1, c2, std::log2(c1 / c2));
    check(std::log2(r1 / r2) >= 0.8,
          tag + ": the co-rotating state's wall residual falls at order >= 0.8 "
                "(the boundary cell's O(h) truncation; measured " +
              std::to_string(std::log2(r1 / r2)) + ")");
    check(r2 < 0.25 * c2, tag + ": the wall acceleration term removes the copy's "
                                "h-independent residual (below a quarter of it)");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const double L = 2 * pi;
    {
        const auto g = Grid::make(128, 128, 1, L, L, 1.0);
        const double cx = L / 2 + 0.37 * g.hx, cy = L / 2 + 0.61 * g.hy,
                     R = L / 16;
        comoving("cylinder 2D", g,
                 [=](double x, double y, double) {
                     return std::hypot(x - cx, y - cy) - R;
                 },
                 {0.6, -0.35, 0.0});
    }
    {
        const auto g = Grid::make(40, 40, 40, L, L, L);
        const double cx = L / 2 + 0.37 * g.hx, cy = L / 2 + 0.61 * g.hy,
                     cz = L / 2 + 0.23 * g.hz, R = L / 6;
        comoving("sphere 3D", g,
                 [=](double x, double y, double z) {
                     return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)
                                      + (z - cz) * (z - cz)) - R;
                 },
                 {0.4, 0.3, -0.5});
    }
    spinning("ghost Cartesian", swirlGhost, 128, 256);
    spinning("O-grid mrf", swirlOGrid, 1, 2);
    return dnstest::finish();
}
