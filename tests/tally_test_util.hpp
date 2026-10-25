// Shared rigs for the IMPOSED surface-tally tests (gpu_tally_*): one circular
// body of radius R, a prescribed 2D primitive field (rho, u, v, p) written at
// the physical cell centres, the path's own boundary fill, then the path's own
// surface tally -- no time stepping. Four paths, each at a resolution level
// lvl (lvl 1: R/h = 16, lvl 2: R/h = 32 on the finest spacing):
//   ghostCartesian  DeviceCompGhost on a uniform box
//   ghostFocal      DeviceCompGhost on an anisotropic --focal mesh (ratio 3,
//                   blend 0.5 of the plateau width per side)
//   ogrid           DeviceOGridBc::surface on the body-fitted annulus
//   cutCell         DeviceBodyForce::measure (pressure part; nu = 0)
// The three compressible rigs optionally SPIN the wall (Spin): a rigid rotation
// about a z shaft, imaged by the moving wall, with a rotating-frame zone
// (f = 1 everywhere) installed so the tally transports the wall-relative
// velocity -- the --rotor-frame mrf configuration.
// GPU-only header.
#pragma once

#include "dns/bodies.hpp"
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/cutcell.hpp"
#include "dns/ogrid.hpp"
#include "dns/rotor.hpp"
#include "dns/gpu/bodyforce.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/rotortable.hpp"
#include "check.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace tallytest {

using namespace dns;

inline constexpr double kGam = 1.4, kRho = 1.0, kR = 1.0;

// A measured number for a check message, at the precision a finding needs.
inline std::string num(double v)
{
    char b[32];
    std::snprintf(b, sizeof b, "%.4e", v);
    return b;
}

struct Prim2
{
    double rho, u, v, p;
};
// Field at a physical point given RELATIVE to the body centre (x, y); every
// rig, and every moment pivot below, is stated relative to that centre.
using FieldFn = std::function<Prim2(double x, double y)>;

// Force on the body (fluid -> body) per unit span, and its z moment about the
// pivot handed to the rig (NaN where the path has no moment tally).
struct Tally
{
    double fx = 0, fy = 0, qz = std::numeric_limits<double>::quiet_NaN();
    bool ok = false;
    double hFine = 0; // finest wall-adjacent spacing the path used
    // Cut-cell only: aperture-measured volumes sum (1-a) h^3 over x / y faces.
    double volAx = 0, volAy = 0;
    // Ghost paths only: the EXACT field's pressure force (and its moment about
    // the same pivot) through the very staircase faces the tally sums, p taken
    // at each face centre -- what the tally would read with a perfect wall
    // image. tally - stair isolates the wall closure; stair - exact is the
    // staircase body's own geometry (a lattice-point count, no smooth order).
    double sfx = std::numeric_limits<double>::quiet_NaN(), sfy = sfx, sqz = sfx;
};

// The staircase reference above, from the classification's interface faces.
// (cx, cy): body centre; pivot: world point.
inline void stairFlux(const Grid& g, const AxisRemap3& R, const gpu::GhostSdf& sdf,
                      const FieldFn& f, double cx, double cy,
                      std::array<double, 2> pivot, Tally& t)
{
    const auto T = gpu::classifyCompGhost(g, sdf, nullptr, R, nullptr);
    const Focal1D* ax[2] = {&R.fx, &R.fy};
    const double h[2] = {g.hx, g.hy};
    t.sfx = t.sfy = t.sqz = 0.0;
    for (int q = 0; q < T.nFace(); ++q) {
        const int c = T.fcell[std::size_t(q)], d = T.fdir[std::size_t(q)];
        const int a = d / 2, b = 1 - a;
        const int idx[2] = {c % g.nx, (c / g.nx) % g.ny};
        const double sgn = (d & 1) ? -1.0 : 1.0;
        double xf[2];
        xf[a] = ax[a]->map((idx[a] + 0.5 + 0.5 * sgn) * h[a]);
        xf[b] = ax[b]->map((idx[b] + 0.5) * h[b]);
        const double area = (ax[b]->map((idx[b] + 1) * h[b]) - ax[b]->map(idx[b] * h[b]))
                            * g.hz;
        const double fa = sgn * f(xf[0] - cx, xf[1] - cy).p * area;
        const double fx = a == 0 ? fa : 0.0, fy = a == 1 ? fa : 0.0;
        t.sfx += fx;
        t.sfy += fy;
        t.sqz += (xf[0] - pivot[0]) * fy - (xf[1] - pivot[1]) * fx;
    }
}

// A wall spinning at omega about a z shaft at `shaft` (relative to the body
// centre), the gas in the rotating-frame zone with it. dropAccel is a
// DIAGNOSTIC: the wall keeps its velocity but its acceleration is withheld
// from the pressure image, which is exactly the zero-gradient copy -- the
// closure the solver used before the image carried the wall's momentum
// balance. The moment is then taken about the shaft.
struct Spin
{
    double omega = 0;
    std::array<double, 2> shaft{};
    bool dropAccel = false;
};

// The rotating-frame zone of a Spin: f = 1 over the whole box.
inline gpu::MrfZone zoneOf(const Spin& s, double sx, double sy)
{
    gpu::MrfZone z;
    z.axis = 2;
    z.omega = s.omega;
    z.center[0] = sx;
    z.center[1] = sy;
    z.center[2] = 0.5;
    z.r1 = 1e30;
    z.r2 = 2e30;
    z.a1 = 1e30;
    z.a2 = 2e30;
    return z;
}

// The wall kinematics of a Spin about the world shaft (sx, sy).
inline gpu::GhostWallMotion motionOf(const Spin& s, double sx, double sy)
{
    const Rotor rot{.axis = RotorAxis::Z, .shaft = {sx, sy, 0.5},
                    .omega = s.omega, .radius = kR, .bins = 1, .thicken = 0};
    const gpu::GhostWallMotion m = gpu::rotorWallMotion(rot);
    if (!s.dropAccel) return m;
    return [m](double x, double y, double z) {
        auto k = m(x, y, z);
        k.a = {0.0, 0.0, 0.0};
        return k;
    };
}

inline CompState stateOf(const Grid& g, const std::function<std::array<double, 2>(int, int)>& centre,
                         const FieldFn& f)
{
    auto s = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const auto x = centre(i, j);
            const Prim2 q = f(x[0], x[1]);
            const std::size_t c = g.idx(i, j, 0);
            s.rho[c] = q.rho;
            s.mx[c] = q.rho * q.u;
            s.my[c] = q.rho * q.v;
            s.E[c] = q.p / (kGam - 1.0) + 0.5 * q.rho * (q.u * q.u + q.v * q.v);
        }
    return s;
}

// Ghost wall on a uniform box of side 4R, n = 64 lvl: R/h = 16 lvl. The centre
// sits off the lattice by a fixed fraction of h so both levels see the same
// relative placement.
inline Tally ghostCartesian(int lvl, const FieldFn& f, double pInf,
                            std::array<double, 2> pivotOff,
                            const Spin* spin = nullptr)
{
    const int n = 64 * lvl;
    const double L = 4.0 * kR, h = L / n;
    const double cx = L / 2 + 0.37 * h, cy = L / 2 + 0.61 * h;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const gpu::GhostSdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - kR;
    };
    Tally t;
    auto dev = gpu::DeviceComp::make(g, kGam, 0.0);
    std::optional<gpu::GhostWallMotion> motion;
    if (spin) motion = motionOf(*spin, cx + spin->shaft[0], cy + spin->shaft[1]);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, kGam, kRho, pInf, nullptr, {},
                                         motion ? &*motion : nullptr);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !ds) { dnstest::check(false, "ghostCartesian setup"); return t; }
    if (spin &&
        dev->setMrf(zoneOf(*spin, cx + spin->shaft[0], cy + spin->shaft[1]),
                    {Focal1D::identity(L), Focal1D::identity(L),
                     Focal1D::identity(1.0)}) != hipSuccess) {
        dnstest::check(false, "ghostCartesian setMrf");
        return t;
    }
    const auto s = stateOf(
        g, [&](int i, int j) { return std::array{(i + 0.5) * h - cx, (j + 0.5) * h - cy}; },
        f);
    (void)ds->upload(s);
    gh->apply(*ds);
    const auto F = gh->force(*dev, *ds, {cx + pivotOff[0], cy + pivotOff[1], 0.0});
    t = {F.fx, F.fy, F.qz, true, h, 0, 0};
    stairFlux(g, {Focal1D::identity(L), Focal1D::identity(L), Focal1D::identity(1.0)},
              sdf, f, cx, cy, {cx + pivotOff[0], cy + pivotOff[1]}, t);
    return t;
}

// The anisotropic focal mesh: plateaus stated physically around the body
// (x half-width 1.25 R, y half-width 2.5 R, so the two axes normalize to
// different fine spacings), ratio 3, blend half the plateau width per side,
// box 12 R. n is chosen so the COARSER fine spacing is R/(16 lvl).
inline Tally ghostFocal(int lvl, const FieldFn& f, double pInf,
                        std::array<double, 2> pivotOff, double* aspect = nullptr,
                        const Spin* spin = nullptr)
{
    const double L = 12.0 * kR, ratio = 3.0;
    const double hx0 = 1.25 * kR, hy0 = 2.5 * kR;
    // c of Focal1D::fromPhysical depends on L, not on n.
    auto cOf = [&](double half) {
        const double r = 1.0 / ratio, w = 0.5 * 2 * half;
        return 1.0 + (1.0 - r) * (2 * half / r + 2.0 * w / (1.0 + r)) / L;
    };
    const double cMax = std::max(cOf(hx0), cOf(hy0));
    const int n = int(std::ceil(16.0 * lvl * cMax / ratio * L / kR));
    const double h = L / n;
    const double cx = L / 2 + 0.37 * h, cy = L / 2 + 0.61 * h;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const double lo[3] = {cx - hx0, cy - hy0, 0}, hi[3] = {cx + hx0, cy + hy0, 0};
    const double w[3] = {hx0, hy0, 0};
    const AxisRemap3 R = focalRemap(g, true, lo, hi, w, ratio);
    const double hfx = R.fx.c * R.fx.r * h, hfy = R.fy.c * R.fy.r * h;
    if (aspect) *aspect = std::max(hfx, hfy) / std::min(hfx, hfy);
    Tally t;
    if (!R.fx.clears(3 * h) || !R.fy.clears(3 * h)) {
        dnstest::check(false, "focal plateau clears the faces");
        return t;
    }
    const gpu::GhostSdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - kR;
    };
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto cpu = Comp::make(mesh, kGam, 0.0);
    if (!cpu) { dnstest::check(false, "focal metrics: " + cpu.error()); return t; }
    auto dev = gpu::DeviceComp::make(*cpu->curv, kGam, 0.0);
    std::optional<gpu::GhostWallMotion> motion;
    if (spin) motion = motionOf(*spin, cx + spin->shaft[0], cy + spin->shaft[1]);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, kGam, kRho, pInf, nullptr, R, {},
                                         motion ? &*motion : nullptr);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !ds) { dnstest::check(false, "ghostFocal setup"); return t; }
    if (spin &&
        dev->setMrf(zoneOf(*spin, cx + spin->shaft[0], cy + spin->shaft[1]), R)
            != hipSuccess) {
        dnstest::check(false, "ghostFocal setMrf");
        return t;
    }
    const auto s = stateOf(g,
                           [&](int i, int j) {
                               return std::array{R.fx.map((i + 0.5) * h) - cx,
                                                 R.fy.map((j + 0.5) * h) - cy};
                           },
                           f);
    (void)ds->upload(s);
    gh->apply(*ds);
    const auto F = gh->force(*dev, *ds, {cx + pivotOff[0], cy + pivotOff[1], 0.0});
    t = {F.fx, F.fy, F.qz, true, std::max(hfx, hfy), 0, 0};
    stairFlux(g, R, sdf, f, cx, cy, {cx + pivotOff[0], cy + pivotOff[1]}, t);
    return t;
}

// Body-fitted O-grid, centre at the origin, rFar = 2.5 R, uniform radial
// spacing: nx = 100 lvl (angular 2 pi R / nx ~ R/16), ny = 2 + 24 lvl
// (radial h1 = R/16). DeviceOGridBc reports the moment only for a moving
// wall, so when a pivot is asked for the wall spins at omega = 1e-30 about
// it: that builds the lever-arm tables, and a wall speed of ~1e-30 is far
// below FP32 resolution of any field here. With a Spin the wall turns for
// real about the shaft, the moment is taken about the shaft, and the solver
// carries the f = 1 rotating-frame zone (setMrfCurv).
inline Tally ogrid(int lvl, const FieldFn& f, double pInf, bool wantMoment,
                   std::array<double, 2> pivotOff, const Spin* spin = nullptr)
{
    const int nx = 100 * lvl, ny = 2 + 24 * lvl;
    const double rFar = 2.5 * kR;
    const auto g = ogridGrid(nx, ny, 1, kR, rFar, 1.0);
    Tally t;
    auto og = buildOGrid(g, sdfOf(CylinderZ{0.0, 0.0, kR}),
                         {.cx = 0, .cy = 0, .cz = 0, .rFar = rFar, .beta = 0,
                          .span = 1.0});
    if (!og) { dnstest::check(false, "O-grid build: " + og.error()); return t; }
    auto m = CurvMetrics::build(og->mesh);
    if (!m) { dnstest::check(false, "O-grid metrics: " + m.error()); return t; }
    gpu::DeviceOGridBc::Config cfg; // adiabatic wall, gas at rest outside
    cfg.gamma = kGam;
    cfg.rhoInf = kRho;
    cfg.uInf = 0.0;
    cfg.pInf = pInf;
    if (wantMoment || spin) {
        cfg.omega = spin ? spin->omega : 1e-30;
        cfg.cx = spin ? spin->shaft[0] : pivotOff[0];
        cfg.cy = spin ? spin->shaft[1] : pivotOff[1];
        cfg.wallPos = ogridWallMidpoints(*og);
        cfg.wallSpan = ogridWallSpans(*og);
        if (spin && spin->dropAccel)
            for (auto& d : cfg.wallSpan) d = {0.0, 0.0};
    }
    auto dev = gpu::DeviceComp::make(*m, kGam, 0.0);
    auto bc = gpu::DeviceOGridBc::make(g, cfg);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !bc || !ds) { dnstest::check(false, "ogrid setup"); return t; }
    if (spin && dev->setMrfCurv(zoneOf(*spin, spin->shaft[0], spin->shaft[1]),
                                og->mesh) != hipSuccess) {
        dnstest::check(false, "ogrid setMrfCurv");
        return t;
    }
    const auto s = stateOf(g,
                           [&](int i, int j) {
                               std::array<double, 2> c{0, 0};
                               for (int dj = 0; dj < 2; ++dj)
                                   for (int di = 0; di < 2; ++di) {
                                       const auto p = og->mesh.node(i + di, j + dj, 0);
                                       c[0] += 0.25 * p[0];
                                       c[1] += 0.25 * p[1];
                                   }
                               return c;
                           },
                           f);
    (void)ds->upload(s);
    bc->apply(*ds);
    const auto S = bc->surface(*dev, *ds);
    t = {S.fx, S.fy, (wantMoment || spin) ? S.tz : t.qz, true, og->h1Max, 0, 0};
    return t;
}

// Incompressible cut-cell tally: interior box of side 4R with n = 64 lvl
// cells plus the one-cell ghost ring every incompressible case carries, hz = 1.
// phi is the field's pressure (a gauge: the incompressible pressure carries no
// absolute level), apertures sanitized as the solver does.
inline Tally cutCell(int lvl, const FieldFn& f)
{
    const int n = 64 * lvl, nx = n + 2;
    const double L = 4.0 * kR, h = L / n;
    const auto g = Grid::make(nx, nx, 1, nx * h, nx * h, 1.0);
    const CylinderZ cyl{h + L / 2 + 0.37 * h, h + L / 2 + 0.61 * h, kR};
    auto cut = CutGeometry::make(g, cyl);
    (void)sanitizeCut(g, cut);
    const auto gg = gpu::GpuGrid::from(g);
    const auto box = gpu::bodyBox(gg, cyl.cx, cyl.cy, 0, cyl.r);
    Tally t;
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto bf = gpu::DeviceBodyForce::make(gg, box);
    auto phi = gpu::DeviceScalar::make(g.size());
    auto q = gpu::DeviceField::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dCut || !bf || !phi || !q || !red) { dnstest::check(false, "cutCell setup"); return t; }
    std::vector<gpu::Real> ph(g.size(), 0), z(g.size(), 0);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            ph[g.idx(i, j, 0)] =
                gpu::Real(f((i + 0.5) * h - cyl.cx, (j + 0.5) * h - cyl.cy).p);
    (void)phi->upload(ph.data());
    (void)q->u.upload(z.data());
    (void)q->v.upload(z.data());
    (void)q->w.upload(z.data());
    const auto c = bf->measure(*dCut, *phi, *q, 0.0, 1.0, nullptr, *red);
    t = {c.cd, c.cl, t.qz, true, h, 0, 0};
    // Aperture-measured volumes over the faces strictly inside the tally box
    // rows (the summation-by-parts range of kBodyPressure).
    for (int j = box.j0; j <= box.j1; ++j)
        for (int i = box.i0 + 1; i <= box.i1; ++i)
            t.volAx += (1.0 - cut.ax[g.idx(i, j, 0)]) * h * h;
    for (int j = box.j0 + 1; j <= box.j1; ++j)
        for (int i = box.i0; i <= box.i1; ++i)
            t.volAy += (1.0 - cut.ay[g.idx(i, j, 0)]) * h * h;
    return t;
}

} // namespace tallytest
