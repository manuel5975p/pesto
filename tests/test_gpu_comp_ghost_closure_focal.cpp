// The ghost wall's closed-surface census on a FOCAL (anisotropic, clustered)
// mesh -- the companion of test_gpu_comp_ghost_closure, which only ever
// classifies through the identity remap.
//
// On a focal mesh the classification runs in physical space (mapped centres,
// local spacings, inverse-mapped samples) and the tally integrates the
// CURVILINEAR face flux, whose area vectors are the metric tables S. The
// interface is still a set of index-space faces, so the same three properties
// must hold, now with the metric face vectors:
//   - every fluid/solid face is tallied and no deep cell sits in a fluid
//     cell's flux stencil (holes 0, reach 0);
//   - sum(S) over the tallied faces == 0 (a closed surface; exact telescoping
//     of the bilinear node faces, so round-off only);
//   - a uniform gas at rest exerts no net force (device curvilinear tally).
// Bodies: an off-centre cylinder (2D), an off-centre sphere (3D), and the
// overlapping two-shell mesh (3D), each on a ratio-3 plateau whose per-axis
// widths differ, so the fine cells are ANISOTROPIC.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/meshsdf.hpp"
#include "dns/trimesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"

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

constexpr double gamma_ = 1.4, rhoInf = 1.0, ratio = 3.0;
const double pInf = 1.0 / (gamma_ * 0.04); // Mach 0.2 on a unit speed

struct Census
{
    int faces = 0, wetted = 0, holes = 0, reach = 0;
    int ghosts = 0, deep = 0, fallback = 0;
    double area = 0.0;               // sum |S| over the tallied faces
    double sumNA[3] = {0, 0, 0};     // sum of oriented metric face vectors

    double closure() const
    {
        return std::sqrt(sumNA[0] * sumNA[0] + sumNA[1] * sumNA[1]
                         + sumNA[2] * sumNA[2]) / std::max(area, 1e-300);
    }
};

// Host census of one classification with the METRIC face vectors: the +a face
// of cell c is S[a](c); a -face of the fluid cell is the +face of its back
// neighbour, oriented into the body.
Census census(const Grid& g, const CurvMetrics& m, const gpu::CompGhostTables& t)
{
    Census c;
    std::vector<char> solid(g.size(), 0), deep(g.size(), 0);
    for (int s : t.solid) solid[std::size_t(s)] = 1;
    for (int d : t.deep) deep[std::size_t(d)] = 1;
    c.ghosts = t.nGhost();
    c.deep = t.nDeep();
    c.fallback = t.fallback;
    c.faces = t.nFace();
    for (int f = 0; f < t.nFace(); ++f) {
        const int d = t.fdir[std::size_t(f)], a = d / 2;
        const int cell = t.fcell[std::size_t(f)];
        const int i = cell % g.nx, j = (cell / g.nx) % g.ny,
                  k = cell / (g.nx * g.ny);
        const bool minus = d & 1;
        const std::size_t own = g.idx(i - (minus && a == 0), j - (minus && a == 1),
                                      k - (minus && a == 2));
        const double sgn = minus ? -1.0 : 1.0;
        double n2 = 0.0;
        for (int q = 0; q < 3; ++q) {
            const double s = m.S[a][q][own];
            c.sumNA[q] += sgn * s;
            n2 += s * s;
        }
        c.area += std::sqrt(n2);
    }
    const int dijk[6][3] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                            {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
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

// Ratio-3 plateau stated physically about (cx, cy, cz) with per-axis half
// widths hw (hw[2] == 0 keeps z the identity), blend = half the plateau width.
// Different half widths give different normalizations c, so different fine
// spacings: anisotropic fine cells.
AxisRemap3 plateau(const Grid& g, const double ctr[3], const double hw[3])
{
    double lo[3], hi[3], w[3];
    for (int a = 0; a < 3; ++a) {
        lo[a] = ctr[a] - hw[a];
        hi[a] = ctr[a] + hw[a];
        w[a] = hw[a];
    }
    return focalRemap(g, true, lo, hi, w, ratio);
}

// Uniform gas at rest, ghost-filled from the census's own tables, tallied on
// the curvilinear device path: |F| relative to pInf times the metric area.
double restForce(const Grid& g, const CurvMetrics& m,
                 const gpu::CompGhostTables& t, double area)
{
    auto dev = gpu::DeviceComp::make(m, gamma_, 0.01);
    gpu::CompGhostCapacity cap;
    cap.cover(t);
    auto gh = gpu::DeviceCompGhost::fromTables(g, t, cap, gamma_, rhoInf, pInf,
                                               true);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !ds) { check(false, "restForce setup"); return 1e300; }
    check(gh->curvilinear(), "focal ghost reports curvilinear");
    auto h = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] = rhoInf;
        h.E[c] = pInf / (gamma_ - 1.0);
    }
    (void)ds->upload(h);
    gh->apply(*ds);
    const auto F = gh->force(*dev, *ds);
    return std::sqrt(F.fx * F.fx + F.fy * F.fy + F.fz * F.fz)
           / (pInf * std::max(area, 1e-300));
}

// The per-face split (surfaceFaces, --emit surface) on the curvilinear tally:
// after one step of a flowing state, each row's s is the oriented metric face
// vector (|s| == area, sum s == 0 over the closed surface), the rows' gauge
// totals sum to force(), and p s + fc + fv == f per row.
void splitFocal(const std::string& name, const Grid& g, const CurvMetrics& m,
                const gpu::CompGhostTables& t)
{
    auto dev = gpu::DeviceComp::make(m, gamma_, 0.01);
    gpu::CompGhostCapacity cap;
    cap.cover(t);
    auto gh = gpu::DeviceCompGhost::fromTables(g, t, cap, gamma_, rhoInf, pInf,
                                               true);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) { check(false, name + ": split setup"); return; }
    auto h = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] = rhoInf;
        h.mx[c] = 0.8;
        h.my[c] = -0.3;
        h.mz[c] = g.nz > 1 ? 0.2 : 0.0;
        h.E[c] = pInf / (gamma_ - 1.0)
                 + 0.5 * (h.mx[c] * h.mx[c] + h.my[c] * h.my[c]
                          + h.mz[c] * h.mz[c]) / rhoInf;
    }
    (void)ds->upload(h);
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
    gh->apply(*ds);
    dev->step(*ds, gpu::Real(dev->stableDt(*ds, *red)));
    gh->apply(*ds);
    const auto F = gh->force(*dev, *ds);
    const auto rows = gh->surfaceFaces(*dev, *ds, pInf);
    check(rows.has_value() && int(rows->size()) == t.nFace(),
          name + ": one split row per interface face");
    if (!rows) return;
    double f[3] = {0, 0, 0}, s[3] = {0, 0, 0}, gross = 0, area = 0, row = 0;
    for (const auto& r : *rows) {
        for (int c = 0; c < 3; ++c) {
            f[c] += r.f[c];
            s[c] += r.s[c];
            gross += std::abs(r.f[c]);
            row = std::max(row, std::abs(r.fp[c] + r.fc[c] + r.fv[c] - r.f[c]));
        }
        area += r.area;
    }
    const double Ff[3] = {F.fx, F.fy, F.fz};
    double ef = 0;
    for (int c = 0; c < 3; ++c) ef = std::max(ef, std::abs(f[c] - Ff[c]));
    const double closure = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]) / area;
    std::printf("%s split: F (%.6e %.6e %.6e) rows-F %.2e (gross %.3e), "
                "|sum s|/A %.1e, max row residual %.1e\n",
                name.c_str(), F.fx, F.fy, F.fz, ef, gross, closure, row);
    check(closure < 1e-12, name + ": split s is the closed metric surface");
    check(ef < 1e-6 * gross, name + ": split row sums == force() (1e-6 of gross)");
    check(row <= 1e-12 * gross, name + ": p s + fc + fv == f per row");
}

// The focal (faceFluxCurv) counterpart of the Cartesian inviscid check: at
// mu = 0 the face flux is p S + rho u_a u only, so fv must be FP32 rounding
// of the gross absolute flux -- inside an MRF zone (pointwise frame velocity
// on the remapped mesh, the rotor runner's path), where u_a = U.S - w.S.
void splitFocalInviscidMrf(const std::string& name, const Grid& g,
                           const CurvMetrics& m, const gpu::CompGhostTables& t,
                           const AxisRemap3& R, const double ctr[3],
                           const double hw[3])
{
    auto dev = gpu::DeviceComp::make(m, gamma_, 0.0);
    gpu::CompGhostCapacity cap;
    cap.cover(t);
    auto gh = gpu::DeviceCompGhost::fromTables(g, t, cap, gamma_, rhoInf, pInf,
                                               true);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) { check(false, name + ": setup"); return; }
    gpu::MrfZone z;
    z.axis = 0;
    z.omega = 0.5 / hw[1];
    for (int c = 0; c < 3; ++c) z.center[c] = ctr[c];
    z.r1 = hw[1];
    z.r2 = 2.0 * hw[1];
    z.a1 = hw[0];
    z.a2 = 2.0 * hw[0];
    check(dev->setMrf(z, R) == hipSuccess, name + ": setMrf");
    auto h = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] = rhoInf;
        h.mx[c] = 0.8;
        h.my[c] = -0.3;
        h.mz[c] = g.nz > 1 ? 0.2 : 0.0;
        h.E[c] = pInf / (gamma_ - 1.0)
                 + 0.5 * (h.mx[c] * h.mx[c] + h.my[c] * h.my[c]
                          + h.mz[c] * h.mz[c]) / rhoInf;
    }
    (void)ds->upload(h);
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
    gh->apply(*ds);
    dev->step(*ds, gpu::Real(dev->stableDt(*ds, *red)));
    gh->apply(*ds);
    const auto rows = gh->surfaceFaces(*dev, *ds, pInf);
    check(rows.has_value() && !rows->empty(), name + ": inviscid surfaceFaces");
    if (!rows) return;
    double fv = 0, fc = 0, gross = 0;
    for (const auto& r : *rows)
        for (int c = 0; c < 3; ++c) {
            fv += std::abs(r.fv[c]);
            fc += std::abs(r.fc[c]);
            gross += std::abs(r.f[c] + pInf * r.s[c]);
        }
    std::printf("%s inviscid MRF split: sum|fv| %.3e, sum|fc| %.3e, gross %.3e "
                "(fv/gross %.2e)\n",
                name.c_str(), fv, fc, gross, fv / gross);
    check(fc > 1e-3 * gross, name + ": the convective part is not vacuous");
    check(fv < 1e-6 * gross,
          name + ": mu = 0 leaves no viscous part (FP32 rounding of the gross)");
}

void closedBody(const std::string& name, const Grid& g, const gpu::GhostSdf& sdf,
                const double ctr[3], const double hw[3])
{
    const AxisRemap3 R = plateau(g, ctr, hw);
    const double hf[3] = {R.fx.c * R.fx.r * g.hx, R.fy.c * R.fy.r * g.hy,
                          R.fz.active ? R.fz.c * R.fz.r * g.hz : g.hz};
    const bool fits = R.fx.clears(4 * g.hx) && R.fy.clears(4 * g.hy)
                      && R.fz.clears(4 * g.hz);
    check(R.fx.active && R.fy.active && (g.nz == 1 || R.fz.active) && fits,
          name + ": plateau active and clears the faces");
    std::printf("%s: h_fine %.4f x %.4f x %.4f (ref %.4f), aspect x/y %.3f\n",
                name.c_str(), hf[0], hf[1], hf[2], g.hx, hf[0] / hf[1]);
    check(std::abs(hf[0] / hf[1] - 1.0) > 0.05,
          name + ": fine cells are anisotropic (> 5%)");
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto m = CurvMetrics::build(mesh);
    if (!m) { check(false, name + ": metrics: " + m.error()); return; }
    // bbox: the plateau box, which holds the body; it only skips sdf calls on
    // cells provably far from it (the mesh sdf is the dominant cost).
    const double bbox[6] = {ctr[0] - hw[0], ctr[1] - hw[1], ctr[2] - hw[2],
                            ctr[0] + hw[0], ctr[1] + hw[1], ctr[2] + hw[2]};
    const auto t = gpu::classifyCompGhost(g, sdf, bbox, R, nullptr);
    const Census c = census(g, *m, t);
    const double fr = restForce(g, *m, t, c.area);
    std::printf("%-28s ghosts %6d (fallback %3d) deep %6d faces %6d wetted %6d "
                "holes %d reach %d |sum S|/A %.2e |F_rest|/(pInf A) %.2e\n",
                name.c_str(), c.ghosts, c.fallback, c.deep, c.faces, c.wetted,
                c.holes, c.reach, c.closure(), fr);
    check(c.faces > 0, name + ": has interface faces");
    check(c.deep > 0, name + ": has deep cells (the property is not vacuous)");
    check(c.holes == 0, name + ": no fluid face against a deep cell");
    check(c.wetted == c.faces, name + ": every fluid/solid face is tallied");
    check(c.closure() < 1e-12, name + ": sum(S) == 0 over the metric faces");
    check(c.reach == 0, name + ": no deep cell inside a fluid flux stencil");
    check(fr < 1e-5, name + ": uniform gas at rest exerts no net force");
    if (name.starts_with("sphere 3D")) {
        splitFocal(name, g, *m, t);
        splitFocalInviscidMrf(name, g, *m, t, R, ctr, hw);
    }
}

gpu::GhostSdf overlappingShells(std::array<double, 3> c, double r, double sep)
{
    TriMesh m = meshtest::icosphere(r, {c[0] - 0.5 * sep, c[1], c[2]}, 3);
    const TriMesh b = meshtest::icosphere(r, {c[0] + 0.5 * sep, c[1], c[2]}, 3);
    const int off = int(m.verts.size());
    m.verts.insert(m.verts.end(), b.verts.begin(), b.verts.end());
    for (const auto& t : b.tris)
        m.tris.push_back({t[0] + off, t[1] + off, t[2] + off});
    auto ms = std::make_shared<MeshSdf>(std::move(m), true);
    return [ms](double x, double y, double z) { return (*ms)(x, y, z); };
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const double L = 2 * pi;
    // Two lattice offsets per body: holes are a property of where the surface
    // cuts the lattice.
    const double off[2][3] = {{0.013, 0.017, 0.011}, {0.37, 0.61, 0.23}};
    for (int o = 0; o < 2; ++o) {
        const std::string tag = " off" + std::to_string(o);
        {
            const auto g = Grid::make(96, 96, 1, L, L, 1.0);
            const double R = L / 16;
            const double ctr[3] = {0.45 * L + off[o][0] * g.hx,
                                   0.55 * L + off[o][1] * g.hy, 0.5};
            const double hw[3] = {2.5 * R, 1.5 * R, 0.0};
            closedBody("cylinder 2D" + tag, g,
                       [=](double x, double y, double) {
                           return std::hypot(x - ctr[0], y - ctr[1]) - R;
                       },
                       ctr, hw);
        }
        {
            const auto g = Grid::make(48, 48, 48, L, L, L);
            const double R = L / 10;
            const double ctr[3] = {0.47 * L + off[o][0] * g.hx,
                                   0.53 * L + off[o][1] * g.hy,
                                   0.48 * L + off[o][2] * g.hz};
            const double hw[3] = {1.6 * R, 1.2 * R, 1.4 * R};
            closedBody("sphere 3D" + tag, g,
                       [=](double x, double y, double z) {
                           return std::sqrt((x - ctr[0]) * (x - ctr[0])
                                            + (y - ctr[1]) * (y - ctr[1])
                                            + (z - ctr[2]) * (z - ctr[2])) - R;
                       },
                       ctr, hw);
        }
        {
            const auto g = Grid::make(48, 48, 48, L, L, L);
            const double r = L / 12;
            const double ctr[3] = {0.5 * L + off[o][0] * g.hx,
                                   0.48 * L + off[o][1] * g.hy,
                                   0.52 * L + off[o][2] * g.hz};
            const double hw[3] = {1.8 * r, 1.2 * r, 1.3 * r};
            closedBody("overlapping shells 3D" + tag, g,
                       overlappingShells({ctr[0], ctr[1], ctr[2]}, r, r),
                       ctr, hw);
        }
    }
    return dnstest::finish();
}
