// Surface tallies against circular Couette flow: the exact field
//   u_theta(r) = A r + B / r   between R1 (the body) and R2,
// IMPOSED on every fluid cell, the wall/ghost fill applied, then tallied --
// no time stepping, so what is measured is the tally and the wall image alone.
//
// Exact: sigma_rtheta = mu r d(u/r)/dr = -2 mu B / r^2, so the torque the
// fluid exerts on the inner cylinder, per unit depth, is
//   Q = R1 * sigma_rtheta(R1) * 2 pi R1 = -4 pi mu B,
//   B = (Omega1 - Omega2) R1^2 R2^2 / (R2^2 - R1^2),
// i.e. +4 pi mu Omega2 R1^2 R2^2/(R2^2-R1^2) with the inner wall at rest and
// -4 pi mu Omega1 R1^2 R2^2/(R2^2-R1^2) with the outer one at rest. Net force 0.
// The outer wall is never needed: the tally reads only the inner wall's faces.
//
// Paths: ghost Cartesian, ghost focal (anisotropic plateau), O-grid (torque
// tz), cut-cell friction (DeviceBodyForce). The cut-cell tally reports a
// FORCE, not a moment, and its box is an index box: tallying the upper half
// of the body gives Fx = -2 R1 sigma_rtheta(R1) = 4 mu B / R1 exactly. The
// cut-cell wall cannot move (no wall-velocity hook), so it has no
// rotating-inner case.
//
// Expected accuracy: the staircase / mirror wall is first order in the shear
// it tallies, so 5% at R/h = 16 and an observed order >= 0.8 over h, h/2.
// The O-grid wall face IS the surface, but its face gradient is the one-sided
// (u1 - u0) / dn of a value mirror: first order too; same bounds apply.
// Cut cell: the linear image misses the profile's wall curvature by O(h^2)
// per ghost, an O(h) friction error on an IMPOSED field (measured -4.5% /
// -2.4% / -1.3% at R/h 16/32/64). Its earlier +6% plateau was the image cell
// renormalized over its open corners (gpu/ghostfill.hip). At the solver's own
// steady state (cutSteady) the drained friction is far closer: discrete
// conservation hands the wall the bulk's flux.
//
// Measurement mode (`test_gpu_tally_couette_torque sweep`, not part of the
// ctest verdict): relative error of every path over a resolution sweep.
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
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/rotortable.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

constexpr double kGam = 1.4, kRho = 1.0, kMu = 0.05;
const double kPInf = 1.0 / (kGam * 0.04); // c = 5
constexpr double kR1 = 1.0, kR2 = 2.0;

// Couette between R1 (rate Om1) and R2 (rate Om2).
struct Couette
{
    double om1, om2, A, B;

    static Couette make(double om1, double om2)
    {
        const double d = kR2 * kR2 - kR1 * kR1;
        return {om1, om2, (om2 * kR2 * kR2 - om1 * kR1 * kR1) / d,
                (om1 - om2) * kR1 * kR1 * kR2 * kR2 / d};
    }
    double uth(double r) const { return A * r + B / r; }
    // dp/dr = rho u^2 / r, p(R1) = pInf.
    double p(double r) const
    {
        return kPInf + kRho * (0.5 * A * A * (r * r - kR1 * kR1) +
                               2 * A * B * std::log(r / kR1) -
                               0.5 * B * B * (1 / (r * r) - 1 / (kR1 * kR1)));
    }
    double torque(double mu) const { return -4 * pi * mu * B; }
};

// Velocity of the exact field at (x, y) about (cx, cy).
std::array<double, 2> vel(const Couette& cf, double x, double y, double cx,
                          double cy)
{
    const double dx = x - cx, dy = y - cy, r = std::hypot(dx, dy);
    const double u = cf.uth(r) / r;
    return {-u * dy, u * dx};
}

void putCell(CompState& h, std::size_t c, const Couette& cf, double x, double y,
             double cx, double cy)
{
    const auto v = vel(cf, x, y, cx, cy);
    const double p = cf.p(std::hypot(x - cx, y - cy));
    h.rho[c] = kRho;
    h.mx[c] = kRho * v[0];
    h.my[c] = kRho * v[1];
    h.mz[c] = 0.0;
    h.E[c] = p / (kGam - 1.0) + 0.5 * kRho * (v[0] * v[0] + v[1] * v[1]);
}

struct Tally
{
    double q = 0, f = 0, rh = 0; // torque, |net force|, R1 / h (finest axis)
    bool ok = false;
};

// Ghost wall, Cartesian (focal == false) or on an anisotropic focal plateau.
Tally ghostTally(int n, const Couette& cf, bool focal)
{
    Tally t;
    const double L = focal ? 8.0 : 4.0;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const double cx = L / 2 + 0.013 * g.hx, cy = L / 2 + 0.017 * g.hy;
    AxisRemap3 R{Focal1D::identity(L), Focal1D::identity(L),
                 Focal1D::identity(1.0)};
    if (focal) {
        // Plateau 4 R1 wide in x, 2.5 R1 in y, blend R1: fine cells 0.71 h by
        // 0.58 h, aspect 1.21 -- anisotropic on purpose.
        const double lo[3] = {cx - 2 * kR1, cy - 1.25 * kR1, 0};
        const double hi[3] = {cx + 2 * kR1, cy + 1.25 * kR1, 0};
        const double w[3] = {kR1, kR1, 0};
        R = focalRemap(g, true, lo, hi, w, 3.0);
        if (!R.fx.clears(2 * g.hx) || !R.fy.clears(2 * g.hy)) {
            check(false, "focal plateau clears the box faces");
            return t;
        }
    }
    const double hfx = focal ? R.fx.c * R.fx.r * g.hx : g.hx;
    const double hfy = focal ? R.fy.c * R.fy.r * g.hy : g.hy;
    t.rh = kR1 / std::max(hfx, hfy);

    const gpu::GhostSdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - kR1;
    };
    const Rotor rot{.axis = RotorAxis::Z, .shaft = {cx, cy, 0.5},
                    .omega = cf.om1, .radius = kR1, .bins = 1, .thicken = 0};
    const gpu::GhostWallMotion uw = gpu::rotorWallMotion(rot);
    const bool moving = cf.om1 != 0.0;

    std::optional<gpu::DeviceComp> dev;
    if (focal) {
        const auto cpu = Comp::make(CurvMesh::fromRemap(g, R), kGam, kMu);
        if (!cpu) { check(false, "focal CPU make"); return t; }
        auto d = gpu::DeviceComp::make(*cpu->curv, kGam, kMu);
        if (d) dev.emplace(std::move(*d));
    } else {
        auto d = gpu::DeviceComp::make(g, kGam, kMu);
        if (d) dev.emplace(std::move(*d));
    }
    auto gh = gpu::DeviceCompGhost::make(g, sdf, kGam, kRho, kPInf, nullptr, R,
                                         {}, moving ? &uw : nullptr);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !ds) { check(false, "ghost setup"); return t; }
    if (gh->curvilinear() != focal) { check(false, "ghost curvilinear flag"); return t; }

    auto h = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            putCell(h, g.idx(i, j, 0), cf, R.fx.map((i + 0.5) * g.hx),
                    R.fy.map((j + 0.5) * g.hy), cx, cy);
    (void)ds->upload(h);
    gh->apply(*ds);
    const auto F = gh->force(*dev, *ds, {cx, cy, 0.5});
    t.q = F.qz;
    t.f = std::hypot(F.fx, F.fy);
    t.ok = true;
    return t;
}

// O-grid between R1 and rFar = R2; the wall row rotates at omegaWall.
Tally ogridTally(int nx, int ny, const Couette& cf, double omegaWall)
{
    Tally t;
    const double cx = 0.0, cy = 0.0;
    const auto g = ogridGrid(nx, ny, 1, kR1, kR2, 1.0);
    auto og = buildOGrid(g, sdfOf(CylinderZ{cx, cy, kR1}),
                         {.cx = cx, .cy = cy, .cz = 0.0, .rFar = kR2,
                          .beta = 0.0, .span = 1.0});
    if (!og) { check(false, "O-grid build: " + og.error()); return t; }
    auto m = CurvMetrics::build(og->mesh);
    if (!m) { check(false, "O-grid metrics: " + m.error()); return t; }
    auto dev = gpu::DeviceComp::make(*m, Transport::constant(kGam, kMu));
    gpu::DeviceOGridBc::Config bc;
    bc.gamma = kGam;
    bc.rhoInf = kRho;
    bc.uInf = 0.0;
    bc.pInf = kPInf;
    bc.omega = omegaWall;
    bc.cx = cx;
    bc.cy = cy;
    bc.wallPos = ogridWallMidpoints(*og);
    bc.wallSpan = ogridWallSpans(*og);
    auto wall = gpu::DeviceOGridBc::make(g, bc);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !wall || !ds) { check(false, "O-grid setup"); return t; }
    t.rh = kR1 / std::max(og->h1Max, 2 * pi * kR1 / nx);

    auto h = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            double x = 0, y = 0;
            for (int d = 0; d < 4; ++d) {
                const auto p = og->mesh.node(i + (d & 1), j + (d >> 1), 0);
                x += 0.25 * p[0];
                y += 0.25 * p[1];
            }
            putCell(h, g.idx(i, j, 0), cf, x, y, cx, cy);
        }
    (void)ds->upload(h);
    wall->apply(*ds);
    const auto F = wall->surface(*dev, *ds);
    t.q = F.tz;
    t.f = std::hypot(F.fx, F.fy);
    t.ok = true;
    return t;
}

// Cut-cell friction (incompressible, MAC faces): Fx on the UPPER half of the
// body (index box above the centre) and on the whole body.
struct CutTally
{
    double fxHalf = 0, fxFull = 0, rh = 0;
    std::size_t fallback = 0; // ghost faces on the fallback image
    bool ok = false;
};

CutTally cutTally(int n, const Couette& cf)
{
    CutTally t;
    const double Lin = 6.0, h = Lin / n;
    const int nx = n + 2;
    const auto g = Grid::make(nx, nx, 1, nx * h, nx * h, h);
    // Centre ON a v-face line (y = jc h), so the half-body box splits between
    // two u-face rows; x off the lattice.
    const int jc = n / 2 + 1;
    const double cx = h + 0.5 * Lin + 0.31 * h, cy = jc * h;
    const CylinderZ cyl{cx, cy, kR1};
    auto cut = CutGeometry::make(g, cyl);
    (void)sanitizeCut(g, cut);
    closeBoxBoundary(g, cut);
    const gpu::Sdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - kR1;
    };
    auto dcut = gpu::DeviceCutGeometry::make(cut);
    auto gf = gpu::DeviceGhostFill::make(g, cut, sdf);
    auto in = gpu::DeviceField::make(g.size());
    auto gh = gpu::DeviceField::make(g.size());
    auto phi = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dcut || !gf || !in || !gh || !phi || !red) {
        check(false, "cut-cell setup");
        return t;
    }
    constexpr double tol = CutGeometry::openTol;
    std::vector<gpu::Real> u(g.size(), 0), v(g.size(), 0), w(g.size(), 0);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            if (cut.ax[c] > tol)
                u[c] = gpu::Real(vel(cf, i * h, (j + 0.5) * h, cx, cy)[0]);
            if (cut.ay[c] > tol)
                v[c] = gpu::Real(vel(cf, (i + 0.5) * h, j * h, cx, cy)[1]);
        }
    if (in->u.upload(u.data()) != hipSuccess ||
        in->v.upload(v.data()) != hipSuccess ||
        in->w.upload(w.data()) != hipSuccess ||
        gh->w.upload(w.data()) != hipSuccess ||
        phi->upload(w.data()) != hipSuccess) {
        check(false, "cut-cell upload");
        return t;
    }
    (void)gh->u.upload(u.data());
    (void)gh->v.upload(v.data());
    gf->fillFrom(*in, *gh);
    const auto gg = gpu::GpuGrid::from(g);
    gpu::BodyBox full = gpu::bodyBox(gg, cx, cy, 0.0, kR1, 3);
    gpu::BodyBox half = full;
    half.j0 = jc;
    auto bfF = gpu::DeviceBodyForce::make(gg, full);
    auto bfH = gpu::DeviceBodyForce::make(gg, half);
    if (!bfF || !bfH) { check(false, "body box"); return t; }
    // qRef = hz: the coefficients come back as force per unit depth.
    t.fxFull = bfF->measure(*dcut, *phi, *gh, kMu / kRho, g.hz, nullptr, *red).cdf;
    t.fxHalf = bfH->measure(*dcut, *phi, *gh, kMu / kRho, g.hz, nullptr, *red).cdf;
    t.rh = kR1 / h;
    t.fallback = gf->fallbackCount();
    t.ok = true;
    return t;
}

// The same cut-cell wall, TIME-STEPPED: the exact field is the initial state,
// every face at r >= R2 is pinned to it, and the momentum diffusion the
// incompressible stage applies (ghost fill -> nu lap -> solid mask) runs to
// its steady state. Circular Couette is harmonic (lap u = 0), so it is the
// fixed point of the continuous operator; what the discrete operator settles
// to, and the friction it then drains, is the wall the SOLVER enforces --
// not the tally's reading of an imposed field.
CutTally cutSteady(int n, const Couette& cf, double tEnd)
{
    CutTally t;
    const double Lin = 6.0, h = Lin / n;
    const int nx = n + 2;
    const auto g = Grid::make(nx, nx, 1, nx * h, nx * h, h);
    const int jc = n / 2 + 1;
    const double cx = h + 0.5 * Lin + 0.31 * h, cy = jc * h;
    auto cut = CutGeometry::make(g, CylinderZ{cx, cy, kR1});
    (void)sanitizeCut(g, cut);
    closeBoxBoundary(g, cut);
    const gpu::Sdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - kR1;
    };
    // The update mask: the solid faces plus the pinned outer ring r >= R2.
    CutGeometry free = cut;
    constexpr double tol = CutGeometry::openTol;
    std::vector<gpu::Real> u(g.size(), 0), v(g.size(), 0), z(g.size(), 0);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double xu = i * h, yu = (j + 0.5) * h;
            const double xv = (i + 0.5) * h, yv = j * h;
            if (cut.ax[c] > tol) u[c] = gpu::Real(vel(cf, xu, yu, cx, cy)[0]);
            if (cut.ay[c] > tol) v[c] = gpu::Real(vel(cf, xv, yv, cx, cy)[1]);
            if (std::hypot(xu - cx, yu - cy) >= kR2) free.ax[c] = 0;
            if (std::hypot(xv - cx, yv - cy) >= kR2) free.ay[c] = 0;
        }
    auto dcut = gpu::DeviceCutGeometry::make(cut);
    auto dfree = gpu::DeviceCutGeometry::make(free);
    auto gf = gpu::DeviceGhostFill::make(g, cut, sdf);
    auto U = gpu::DeviceField::make(g.size());
    auto G = gpu::DeviceField::make(g.size());
    auto R = gpu::DeviceField::make(g.size());
    auto phi = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dcut || !dfree || !gf || !U || !G || !R || !phi || !red) {
        check(false, "cut-cell steady setup");
        return t;
    }
    (void)U->u.upload(u.data());
    (void)U->v.upload(v.data());
    (void)U->w.upload(z.data());
    (void)R->w.upload(z.data());
    (void)phi->upload(z.data());
    const auto gg = gpu::GpuGrid::from(g);
    // Explicit diffusion at 0.2 of the 2D stability bound h^2 / (4 nu).
    const double dt = 0.2 * h * h / kMu;
    const long steps = long(std::ceil(tEnd / dt));
    for (long s = 0; s < steps; ++s) {
        gf->fillFrom(*U, *G);
        (void)R->u.upload(z.data());
        (void)R->v.upload(z.data());
        gpu::addScaledLaplacian(gg, gpu::Real(kMu), *G, *R);
        gpu::maskSolidFaces(gg, *dfree, *R);
        gpu::rkCombine(gg, 0, *U, 1, *U, gpu::Real(dt), *R, *U);
    }
    gf->fillFrom(*U, *G);
    gpu::BodyBox full = gpu::bodyBox(gg, cx, cy, 0.0, kR1, 3);
    gpu::BodyBox half = full;
    half.j0 = jc;
    auto bfF = gpu::DeviceBodyForce::make(gg, full);
    auto bfH = gpu::DeviceBodyForce::make(gg, half);
    if (!bfF || !bfH) { check(false, "body box"); return t; }
    t.fxFull = bfF->measure(*dcut, *phi, *G, kMu / kRho, g.hz, nullptr, *red).cdf;
    t.fxHalf = bfH->measure(*dcut, *phi, *G, kMu / kRho, g.hz, nullptr, *red).cdf;
    t.rh = kR1 / h;
    t.fallback = gf->fallbackCount();
    t.ok = true;
    return t;
}

double order(double eCoarse, double eFine) { return std::log2(eCoarse / eFine); }

// The two-grid verdict shared by every path: 5% at the coarse grid
// (R/h >= 16), error falling at order >= 0.8.
void verdict(const std::string& tag, double rhC, double eC, double rhF,
             double eF)
{
    const double ord = order(eC, eF);
    std::printf("  %s: R/h %.1f err %.3e | R/h %.1f err %.3e | order %.2f\n",
                tag.c_str(), rhC, eC, rhF, eF, ord);
    check(eC < 0.05, tag + ": within 5% at R/h ~16 (measured " +
                         std::to_string(100 * eC) + "%)");
    check(ord >= 0.8, tag + ": error falls at order >= 0.8 (measured " +
                          std::to_string(ord) + ")");
}

void ghostCase(const char* name, bool focal, const Couette& cf, int n0)
{
    const double Q = cf.torque(kMu);
    std::printf("--- %s, Omega1 %.3f Omega2 %.3f: exact torque %.6e ---\n",
                name, cf.om1, cf.om2, Q);
    const Tally a = ghostTally(n0, cf, focal), b = ghostTally(2 * n0, cf, focal);
    if (!a.ok || !b.ok) return;
    std::printf("  n %d: torque %.6e |F| %.3e   n %d: torque %.6e |F| %.3e\n",
                n0, a.q, a.f, 2 * n0, b.q, b.f);
    check(a.q * Q > 0 && b.q * Q > 0,
          std::string(name) + ": torque has the exact sign");
    verdict(std::string(name) + " torque", a.rh, std::abs(a.q / Q - 1), b.rh,
            std::abs(b.q / Q - 1));
    // Net force is exactly 0; scaled by the torque's force scale Q / R1.
    const double fs = std::abs(Q) / kR1;
    check(a.f < 0.05 * fs && b.f < 0.05 * fs,
          std::string(name) + ": net force ~0 (|F| R1/|Q| " +
              std::to_string(a.f / fs) + ", " + std::to_string(b.f / fs) + ")");
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();

    // Wall speeds 0.5 = Mach 0.1: the pressure field and the convective
    // momentum flux stay small against the shear.
    const Couette outer = Couette::make(0.0, 0.5 / kR2);
    const Couette inner = Couette::make(0.5 / kR1, 0.0);

    // Measurement mode (`test_gpu_tally_couette_torque sweep`): relative
    // torque / friction error over a resolution sweep, per path.
    if (argc > 1 && std::string(argv[1]) == "cut") {
        const double fx = 4 * kMu * outer.B / kR1;
        const double tEnd = argc > 2 ? std::stod(argv[2]) : 20.0;
        for (int n : {48, 64, 96, 128, 192, 256, 384}) {
            const CutTally a = cutTally(n, outer), s = cutSteady(n, outer, tEnd);
            std::printf("n %4d R/h %5.1f imposed %+.4e steady %+.4e (full %.2e)\n",
                        n, a.rh, a.fxHalf / fx - 1, s.fxHalf / fx - 1,
                        s.fxFull / fx);
        }
        return 0;
    }
    if (argc > 1) {
        const double Q = outer.torque(kMu), fx = 4 * kMu * outer.B / kR1;
        for (int n : {48, 64, 80, 96, 128, 160, 192, 256, 320, 384}) {
            const Tally a = ghostTally(n, outer, false),
                        b = ghostTally(n, outer, true);
            const CutTally c = cutTally(n, outer);
            const Tally o = ogridTally(n * 100 / 64, 2 + n / 4, outer, 1e-9);
            std::printf("n %4d cart R/h %5.1f %+.3e | focal R/h %5.1f %+.3e | "
                        "cut R/h %5.1f %+.3e | ogrid R/h %5.1f %+.3e\n",
                        n, a.rh, a.q / Q - 1, b.rh, b.q / Q - 1, c.rh,
                        c.fxHalf / fx - 1, o.rh, o.q / Q - 1);
        }
        return 0;
    }

    ghostCase("ghost Cartesian, inner at rest", false, outer, 64);
    ghostCase("ghost Cartesian, inner rotating", false, inner, 64);
    ghostCase("ghost focal, inner at rest", true, outer, 96);
    ghostCase("ghost focal, inner rotating", true, inner, 96);

    // O-grid: arc and radial spacing R1/16 at the coarse grid. Its torque
    // column exists only with a wall rate (lever-arm tables), so the resting
    // inner wall turns at 1e-9 -- a wall speed of 1e-9, below FP32 resolution
    // of the fluid velocity, i.e. the resting mirror in all but name.
    for (const auto& [name, cf] :
         {std::pair{"O-grid, inner at rest", outer},
          std::pair{"O-grid, inner rotating", inner}}) {
        const double Q = cf.torque(kMu);
        const double om = cf.om1 != 0.0 ? cf.om1 : 1e-9;
        std::printf("--- %s: exact torque %.6e ---\n", name, Q);
        const Tally a = ogridTally(100, 18, cf, om),
                    b = ogridTally(200, 34, cf, om);
        if (!a.ok || !b.ok) continue;
        std::printf("  coarse: torque %.6e |F| %.3e  fine: torque %.6e |F| %.3e\n",
                    a.q, a.f, b.q, b.f);
        check(a.q * Q > 0 && b.q * Q > 0,
              std::string(name) + ": torque has the exact sign");
        verdict(std::string(name) + " torque", a.rh, std::abs(a.q / Q - 1),
                b.rh, std::abs(b.q / Q - 1));
        const double fs = std::abs(Q) / kR1;
        check(a.f < 0.05 * fs && b.f < 0.05 * fs,
              std::string(name) + ": net force ~0");
    }

    // Cut-cell friction, inner at rest only (no moving cut-cell wall).
    {
        const double fx = 4 * kMu * outer.B / kR1;
        std::printf("--- cut-cell friction, inner at rest: exact upper-half "
                    "Fx %.6e ---\n", fx);
        const CutTally a = cutTally(96, outer), b = cutTally(192, outer);
        if (a.ok && b.ok) {
            std::printf("  n 96: Fx half %.6e full %.3e   n 192: Fx half %.6e "
                        "full %.3e\n", a.fxHalf, a.fxFull, b.fxHalf, b.fxFull);
            check(a.fxHalf * fx > 0 && b.fxHalf * fx > 0,
                  "cut-cell: upper-half friction has the exact sign");
            verdict("cut-cell upper-half friction", a.rh,
                    std::abs(a.fxHalf / fx - 1), b.rh,
                    std::abs(b.fxHalf / fx - 1));
            check(std::abs(a.fxFull) < 0.05 * std::abs(fx) &&
                      std::abs(b.fxFull) < 0.05 * std::abs(fx),
                  "cut-cell: whole-body friction Fx ~0");
            // A resolved convex body needs no fallback image anywhere: every
            // ghost reads a clean trilinear cell.
            check(a.fallback == 0 && b.fallback == 0,
                  "cut-cell: no ghost face on the fallback image (" +
                      std::to_string(a.fallback) + ", " +
                      std::to_string(b.fallback) + ")");
        }
    }

    // Cut-cell friction at the SOLVER's steady state (momentum diffusion run
    // to its fixed point, outer ring pinned): what the solver drains through
    // the wall once its own discrete solution has settled. Three levels.
    // Measured (2026-10-24): -1.71e-3 / -4.76e-4 / -1.79e-4 at R/h 16/32/64,
    // orders 1.85, 1.41. Before the clean-cell image it was +3.26e-3 /
    // +2.25e-3 / +9.39e-4 -- non-monotone over R/h 8..64, order 0.53 on the
    // first pair.
    {
        const double fx = 4 * kMu * outer.B / kR1;
        std::printf("--- cut-cell friction, steady diffusion fixed point ---\n");
        const CutTally s[3] = {cutSteady(96, outer, 20.0),
                               cutSteady(192, outer, 20.0),
                               cutSteady(384, outer, 20.0)};
        if (s[0].ok && s[1].ok && s[2].ok) {
            double e[3];
            for (int l = 0; l < 3; ++l) {
                e[l] = std::abs(s[l].fxHalf / fx - 1);
                std::printf("  R/h %5.1f: Fx half %.6e err %.3e\n", s[l].rh,
                            s[l].fxHalf, e[l]);
            }
            check(e[0] < 0.01, "cut-cell steady friction: within 1% at R/h 16 "
                               "(measured " + std::to_string(100 * e[0]) + "%)");
            for (int l = 0; l < 2; ++l)
                check(order(e[l], e[l + 1]) >= 1.0,
                      "cut-cell steady friction: order >= 1 from R/h " +
                          std::to_string(int(s[l].rh)) + " (measured " +
                          std::to_string(order(e[l], e[l + 1])) + ")");
        }
    }
    return dnstest::finish();
}
