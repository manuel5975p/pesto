// The ghost wall's surface tally is a CLOSED surface, and the flow never sees a
// deep-solid cell.
//
// The tally (kBodyFlux) sums the face flux over every fluid/ghost face. That
// is the body's force only if those faces are ALL the faces between fluid and
// solid: a fluid face against a DEEP cell (held at rest at pInf, not mirrored)
// is a piece of wall the fluid feels and the tally never reads. Then
// sum(n A) != 0 and a uniform pressure -- which exerts no net force on any
// closed body -- reports one (measured on a 3-blade prop mesh: up to 16% of
// the wetted faces missing, a spurious CT of +0.025 from pInf alone). The same
// face is also a wrong WALL: rest at pInf instead of the mirror image, so on a
// rotor a resting patch on a moving blade.
//
// How a near-wall cell ends up deep: its wall-normal sample finds no fluid.
// The commonest cause is a mesh of OVERLAPPING shells (hub and blades exported
// as separate closed parts, no boolean): inside the overlap the nearest
// triangle can be a BURIED face of the other part, so the sdf normal points
// into the body and the sample lands in solid. The two-shell mesh below is
// that case in miniature; the analytic bodies are the controls.
//
// Properties, per body (thick, thin down to ~2 cells, concave junction,
// overlapping mesh shells, and two lattice offsets each):
//   - every fluid/solid face is tallied, so sum(n A) == 0 to round-off;
//   - no deep cell inside the flux stencil of any fluid cell (face + edge
//     neighbours): the deep-solid notion survives only where nothing reads it;
//   - a uniform gas at rest exerts zero force on the body (device tally);
//   - a spinning thin body in quiescent gas feels a torque opposing its spin.
//
// Optional: `test_gpu_comp_ghost_closure <mesh.stl> <n>` runs the census on a
// rotor mesh about the x shaft (swept diameter pi/4) on an n^3 uniform box --
// a host-only measurement mode, not part of the ctest pass/fail.
#include "dns/bodies.hpp"
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/meshsdf.hpp"
#include "dns/rotor.hpp"
#include "dns/trimesh.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"
#include "mesh_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numbers>
#include <set>
#include <string>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

constexpr double gamma_ = 1.4, rhoInf = 1.0;
const double pInf = 1.0 / (gamma_ * 0.04); // Mach 0.2 on a unit speed

struct Census
{
    int faces = 0;     // tallied interface faces
    int wetted = 0;    // every fluid/solid face
    int holes = 0;     // fluid faces against a deep cell
    int reach = 0;     // deep cells inside a fluid cell's flux stencil
    int ghosts = 0, deep = 0, fallback = 0;
    double area = 0.0; // area of the tallied faces
    double sumNA[3] = {0, 0, 0};

    double closure() const
    {
        return std::sqrt(sumNA[0] * sumNA[0] + sumNA[1] * sumNA[1]
                         + sumNA[2] * sumNA[2]) / std::max(area, 1e-300);
    }
};

// Host census of one classification: what the tally covers against what the
// fluid actually borders.
Census census(const Grid& g, const gpu::CompGhostTables& t)
{
    Census c;
    std::vector<char> solid(g.size(), 0), deep(g.size(), 0);
    for (int s : t.solid) solid[std::size_t(s)] = 1;
    for (int d : t.deep) deep[std::size_t(d)] = 1;
    c.ghosts = t.nGhost();
    c.deep = t.nDeep();
    c.fallback = t.fallback;
    c.faces = t.nFace();
    const double area[3] = {g.hy * g.hz, g.hx * g.hz, g.hx * g.hy};
    const int dijk[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                            {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    for (int f = 0; f < t.nFace(); ++f) {
        const int d = t.fdir[std::size_t(f)];
        const double sgn = (d & 1) ? -1.0 : 1.0;
        c.sumNA[d / 2] += sgn * area[d / 2];
        c.area += area[d / 2];
    }
    const int ndir = g.nz == 1 ? 4 : 6;
    const int kr = g.nz == 1 ? 0 : 1;
    std::set<std::size_t> reached;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                if (solid[g.idx(i, j, k)]) continue;
                for (int d = 0; d < ndir; ++d) {
                    const std::size_t nc = g.idx(i + dijk[d][0], j + dijk[d][1],
                                                 k + dijk[d][2]);
                    if (solid[nc]) ++c.wetted;
                    if (deep[nc]) ++c.holes;
                }
                // Face and edge neighbours: what faceFlux reads around a
                // fluid cell (the normal pair plus the tangential averages).
                for (int dk = -kr; dk <= kr; ++dk)
                    for (int dj = -1; dj <= 1; ++dj)
                        for (int di = -1; di <= 1; ++di) {
                            const int nnz = (di != 0) + (dj != 0) + (dk != 0);
                            if (nnz == 0 || nnz == 3) continue;
                            const std::size_t nc = g.idx(i + di, j + dj, k + dk);
                            if (deep[nc]) reached.insert(nc);
                        }
            }
    c.reach = int(reached.size());
    return c;
}

CompState quiescent(const Grid& g)
{
    auto h = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] = rhoInf;
        h.E[c] = pInf / (gamma_ - 1.0);
    }
    return h;
}

// Uniform gas at rest at pInf, ghost-filled, tallied on the device: the net
// force relative to pInf times the tallied area.
double restForce(const Grid& g, const gpu::GhostSdf& sdf, double area)
{
    auto dev = gpu::DeviceComp::make(g, gamma_, 0.01);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, gamma_, rhoInf, pInf);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !ds) { check(false, "restForce setup"); return 1e300; }
    (void)ds->upload(quiescent(g));
    gh->apply(*ds);
    const auto F = gh->force(*dev, *ds);
    return std::sqrt(F.fx * F.fx + F.fy * F.fy + F.fz * F.fz)
           / (pInf * std::max(area, 1e-300));
}

void report(const std::string& name, const Census& c, double frest)
{
    std::printf("%-30s ghosts %6d (fallback %4d) deep %6d faces %6d wetted %6d "
                "holes %5d reach %5d |sum nA|/A %.2e |F_rest|/(pInf A) %.2e\n",
                name.c_str(), c.ghosts, c.fallback, c.deep, c.faces, c.wetted,
                c.holes, c.reach, c.closure(), frest);
}

AxisRemap3 identityRemap(const Grid& g)
{
    return {Focal1D::identity(g.nx * g.hx), Focal1D::identity(g.ny * g.hy),
            Focal1D::identity(g.nz * g.hz)};
}

// One body on one lattice: classify, census, rest force, and the checks.
void closedBody(const std::string& name, const Grid& g, const gpu::GhostSdf& sdf)
{
    const auto t = gpu::classifyCompGhost(g, sdf, nullptr, identityRemap(g),
                                          nullptr);
    const Census c = census(g, t);
    const double fr = restForce(g, sdf, c.area);
    report(name, c, fr);
    check(c.faces > 0, name + ": has interface faces");
    check(c.holes == 0, name + ": no fluid face against a deep cell");
    check(c.wetted == c.faces, name + ": every fluid/solid face is tallied");
    check(c.closure() < 1e-12, name + ": sum(n A) == 0 (closed surface)");
    check(c.reach == 0, name + ": no deep cell inside a fluid flux stencil");
    check(fr < 1e-5, name + ": uniform gas at rest exerts no net force");
}

// Signed distance of a box of half extents (a, b, c) about the origin.
double sdfBox(double x, double y, double z, double a, double b, double c)
{
    const double qx = std::abs(x) - a, qy = std::abs(y) - b,
                 qz = std::abs(z) - c;
    const double ox = std::max(qx, 0.0), oy = std::max(qy, 0.0),
                 oz = std::max(qz, 0.0);
    return std::sqrt(ox * ox + oy * oy + oz * oz)
           + std::min(std::max({qx, qy, qz}), 0.0);
}

// Thin lens of half-height hh, tilted by `tilt` about z, centred at (cx, cy).
// Rotation is an isometry, so the field stays a distance.
gpu::GhostSdf thinLens(double cx, double cy, double w, double hh, double tilt)
{
    const Sdf lens = sdfOf(Lens{0.0, 0.0, w, hh});
    const double c = std::cos(tilt), s = std::sin(tilt);
    return [=](double x, double y, double z) {
        const double dx = x - cx, dy = y - cy;
        return lens(c * dx + s * dy, -s * dx + c * dy, z);
    };
}

// A hub sphere and one thin pitched blade, union: the concave junction is
// where a normal-sampled image runs into the other body. Blade spans +y,
// pitched by `pitch` about y, `thick` = full thickness.
gpu::GhostSdf hubBlade(double cx, double cy, double cz, double hub,
                       double span, double chord, double thick, double pitch)
{
    const double c = std::cos(pitch), s = std::sin(pitch);
    return [=](double x, double y, double z) {
        const double dx = x - cx, dy = y - cy, dz = z - cz;
        const double sh = std::sqrt(dx * dx + dy * dy + dz * dz) - hub;
        // Blade frame: rotate (x, z) about y, root at the hub centre.
        const double bx = c * dx + s * dz, bz = -s * dx + c * dz;
        const double sb = sdfBox(bx, dy - 0.5 * span, bz, 0.5 * chord,
                                 0.5 * span, 0.5 * thick);
        return std::min(sh, sb);
    };
}

// Two closed icospheres concatenated into ONE mesh with no boolean: the buried
// halves stay inside the other sphere, exactly like hub and blades exported as
// overlapping parts. Winding-number sign (robust), as such meshes need.
gpu::GhostSdf overlappingShells(std::array<double, 3> c, double r, double sep)
{
    TriMesh m = meshtest::icosphere(r, {c[0] - 0.5 * sep, c[1], c[2]}, 3);
    const TriMesh b = meshtest::icosphere(r, {c[0] + 0.5 * sep, c[1], c[2]}, 3);
    const int off = int(m.verts.size());
    m.verts.insert(m.verts.end(), b.verts.begin(), b.verts.end());
    for (const auto& t : b.tris) m.tris.push_back({t[0] + off, t[1] + off, t[2] + off});
    auto ms = std::make_shared<MeshSdf>(std::move(m), true);
    return [ms](double x, double y, double z) { return (*ms)(x, y, z); };
}

struct Spin
{
    double torque = 0, ke = 0;
};

// Spin a body from rest in a quiescent gas for tEnd; torque about the shaft.
Spin spinUp(const Grid& g, const gpu::GhostSdf& sdf, const Rotor& rot,
            double mu, double tEnd)
{
    const gpu::GhostWallVelocity uw = [&rot](double x, double y, double z) {
        return rotorWallVelocity(rot, {x, y, z});
    };
    auto dev = gpu::DeviceComp::make(g, gamma_, mu);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, gamma_, rhoInf, pInf, nullptr,
                                         {}, &uw);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) { check(false, "spinUp setup"); return {}; }
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
    auto host = quiescent(g);
    (void)ds->upload(host);
    double t = 0.0;
    while (t < tEnd) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
        t += dt;
    }
    gh->apply(*ds);
    const auto F = gh->force(*dev, *ds, rot.shaft);
    (void)ds->download(host);
    Spin out;
    const int a = rotorAxisIndex(rot.axis);
    out.torque = a == 0 ? F.qx : a == 1 ? F.qy : F.qz;
    std::vector<char> solid(g.size(), 0);
    for (int s : gh->solidCells()) solid[std::size_t(s)] = 1;
    const double vol = g.hx * g.hy * g.hz;
    for (std::size_t c = 0; c < g.size(); ++c) {
        if (solid[c]) continue;
        const double m2 = host.mx[c] * host.mx[c] + host.my[c] * host.my[c]
                          + host.mz[c] * host.mz[c];
        out.ke += 0.5 * m2 / host.rho[c] * vol;
    }
    return out;
}

// Measurement mode: a rotor mesh about the x shaft, swept diameter pi/4, on an
// n^3 uniform box 1.3 D wide. Census only, no device.
int meshCensus(const std::string& path, int n)
{
    auto mesh = loadStl(path);
    if (!mesh) {
        std::printf("load %s: %s\n", path.c_str(), mesh.error().c_str());
        return 3;
    }
    const double D = pi / 4, Lb = 1.3 * D;
    const double scale = D / sweptDiameter(*mesh, 0);
    transform(*mesh, scale, 0.0, 0.0, {0.5 * Lb, 0.5 * Lb, 0.5 * Lb},
              MeshAnchor::Origin);
    auto ms = std::make_shared<MeshSdf>(std::move(*mesh), true);
    const gpu::GhostSdf sdf = [ms](double x, double y, double z) {
        return (*ms)(x, y, z);
    };
    const auto g = Grid::make(n, n, n, Lb, Lb, Lb);
    const auto t = gpu::classifyCompGhost(g, sdf, nullptr, identityRemap(g),
                                          nullptr);
    const Census c = census(g, t);
    // Host only: on a uniform rest state the tally is pInf sum(n A) exactly,
    // so the closure column IS the rest force (the device check pins that).
    report(path + " n" + std::to_string(n), c, c.closure());
    std::printf("  sum nA = (%.3e, %.3e, %.3e), h = %.4e, D/h = %.1f\n",
                c.sumNA[0], c.sumNA[1], c.sumNA[2], g.hx, D / g.hx);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 3) return meshCensus(argv[1], std::atoi(argv[2]));
    test::requireDeviceOrSkip();

    const double L = 2 * pi;
    // Two lattice offsets per body: the holes are a property of where the
    // surface cuts the lattice, so one placement can hide them.
    const double off[2][3] = {{0.013, 0.017, 0.011}, {0.37, 0.61, 0.23}};

    for (int o = 0; o < 2; ++o) {
        const std::string tag = " off" + std::to_string(o);
        {
            const auto g = Grid::make(64, 64, 1, L, L, 1.0);
            const double h = g.hx;
            const double cx = L / 2 + off[o][0] * h, cy = L / 2 + off[o][1] * h;
            const double R = L / 8;
            closedBody("cylinder 2D" + tag, g, [=](double x, double y, double) {
                return std::hypot(x - cx, y - cy) - R;
            });
        }
        {
            const auto g = Grid::make(48, 48, 48, L, L, L);
            const double h = g.hx;
            const double cx = L / 2 + off[o][0] * h, cy = L / 2 + off[o][1] * h,
                         cz = L / 2 + off[o][2] * h, R = L / 6;
            closedBody("sphere 3D" + tag, g, [=](double x, double y, double z) {
                return std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy)
                                 + (z - cz) * (z - cz)) - R;
            });
        }
        for (const double cells : {3.0, 2.5, 2.0}) {
            const auto g = Grid::make(128, 128, 1, L, L, 1.0);
            const double h = g.hx;
            const double cx = L / 2 + off[o][0] * h, cy = L / 2 + off[o][1] * h;
            char nm[64];
            std::snprintf(nm, sizeof nm, "lens 2D %.1f cells", cells);
            closedBody(nm + tag, g,
                       thinLens(cx, cy, L / 6, 0.5 * cells * h, 0.35));
        }
        for (const double cells : {3.0, 2.0}) {
            const auto g = Grid::make(64, 64, 64, L, L, L);
            const double h = g.hx;
            const double cx = L / 2 + off[o][0] * h, cy = L / 3 + off[o][1] * h,
                         cz = L / 2 + off[o][2] * h;
            char nm[64];
            std::snprintf(nm, sizeof nm, "hub+blade 3D %.1f cells", cells);
            closedBody(nm + tag, g,
                       hubBlade(cx, cy, cz, 4.0 * h, 22.0 * h, 10.0 * h,
                                cells * h, 0.6));
        }
        {
            const auto g = Grid::make(40, 40, 40, L, L, L);
            const double h = g.hx;
            closedBody("overlapping shells 3D" + tag, g,
                       overlappingShells({L / 2 + off[o][0] * h,
                                          L / 2 + off[o][1] * h,
                                          L / 2 + off[o][2] * h},
                                         L / 6, L / 6));
        }
    }

    // A MOVING thin body. The wall faces that used to fall back to a deep cell
    // were resting patches on a moving surface; now every one is imaged about
    // omega x r. Spin-up from rest must oppose the spin and drive the gas.
    {
        const auto g = Grid::make(128, 128, 1, L, L, 1.0);
        const double h = g.hx;
        const double cx = L / 2 + 0.37 * h, cy = L / 2 + 0.61 * h;
        const double w = L / 6;
        const Rotor rot{.axis = RotorAxis::Z, .shaft = {cx, cy, 0.5},
                        .omega = 1.0 / w, .radius = w, .bins = 1,
                        .thicken = 0.0};
        const auto s = spinUp(g, thinLens(cx, cy, w, 1.0 * h, 0.35), rot,
                              rhoInf * 2 * w / 40.0, 0.5);
        std::printf("spinning lens 2D 2.0 cells: torque %.6e ke %.6e\n",
                    s.torque, s.ke);
        check(std::isfinite(s.torque) && std::isfinite(s.ke),
              "spinning thin lens: finite");
        check(s.torque < 0.0, "spinning thin lens: torque opposes the spin");
        check(s.ke > 0.0, "spinning thin lens: the wall drives the gas");
    }
    {
        const auto g = Grid::make(64, 64, 64, L, L, L);
        const double h = g.hx;
        const double cx = L / 2 + 0.37 * h, cy = L / 3 + 0.61 * h,
                     cz = L / 2 + 0.23 * h;
        const double span = 22.0 * h;
        const Rotor rot{.axis = RotorAxis::X, .shaft = {cx, cy, cz},
                        .omega = 1.0 / span, .radius = span, .bins = 1,
                        .thicken = 0.0};
        const auto s = spinUp(g, hubBlade(cx, cy, cz, 4.0 * h, span, 10.0 * h,
                                          2.0 * h, 0.6),
                              rot, rhoInf * span / 40.0, 0.5);
        std::printf("spinning hub+blade 3D 2.0 cells: torque %.6e ke %.6e\n",
                    s.torque, s.ke);
        check(std::isfinite(s.torque) && std::isfinite(s.ke),
              "spinning hub+blade: finite");
        check(s.torque < 0.0, "spinning hub+blade: torque opposes the spin");
        check(s.ke > 0.0, "spinning hub+blade: the wall drives the gas");
    }
    // The same overlapping-shell mesh, spinning about the axis through both
    // centres: every hole of the resting classification was a resting patch.
    {
        const auto g = Grid::make(40, 40, 40, L, L, L);
        const double h = g.hx;
        const std::array<double, 3> c{L / 2 + 0.37 * h, L / 2 + 0.61 * h,
                                      L / 2 + 0.23 * h};
        const Rotor rot{.axis = RotorAxis::Y, .shaft = c, .omega = 1.0 / (L / 3),
                        .radius = L / 3, .bins = 1, .thicken = 0.0};
        const auto s = spinUp(g, overlappingShells(c, L / 6, L / 6), rot,
                              rhoInf * (L / 3) / 40.0, 0.5);
        std::printf("spinning overlapping shells 3D: torque %.6e ke %.6e\n",
                    s.torque, s.ke);
        check(std::isfinite(s.torque) && std::isfinite(s.ke),
              "spinning overlapping shells: finite");
        check(s.torque < 0.0,
              "spinning overlapping shells: torque opposes the spin");
    }

    return dnstest::finish();
}
