#include "ui/view.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace ui {
namespace {

// --- palette ----------------------------------------------------------------

// Fluid fraction lambda -> colour. Blue is a nearly-full cell, red a nearly
// closed one, because the near-zero cells are the ones that matter: they are
// what sanitizeCut() solidifies and what stalls the FP32 PCG when it does not.
Rgba lambdaColor(double l, uint8_t alpha)
{
    const double t = std::clamp(1.0 - l, 0.0, 1.0);   // 0 open .. 1 closed
    // 4-stop ramp: steel blue -> teal -> amber -> red.
    static const float stops[4][3] = {{0.30f, 0.55f, 0.95f},
                                      {0.25f, 0.80f, 0.70f},
                                      {0.95f, 0.75f, 0.25f},
                                      {0.92f, 0.30f, 0.28f}};
    const double s = t * 3.0;
    const int i = std::min(2, int(s));
    const float f = float(s - i);
    auto ch = [&](int k) {
        return uint8_t(255.0f * (stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f));
    };
    return {ch(0), ch(1), ch(2), alpha};
}

Rgba apertureColor(double a, uint8_t alpha)
{
    // Apertures use the same ramp so a face and its cell can be compared, but
    // desaturated so the two layers are distinguishable when both are on.
    const Rgba c = lambdaColor(a, alpha);
    return {uint8_t((c.r + 150) / 2), uint8_t((c.g + 150) / 2),
            uint8_t((c.b + 150) / 2), alpha};
}

const Rgba kBox = rgb(0x5a6b7d);
const Rgba kBoxInner = rgb(0x8fa3b8);
const Rgba kGrid = rgb(0x2c3541);
const Rgba kGridMajor = rgb(0x46586c);
const Rgba kFocal = rgb(0x4fa3ff);
const Rgba kSurface = rgb(0xc8d2dc);
const Rgba kSolid = rgb(0x39434f);

void line(std::vector<VtxL>& v, const Vec3& a, const Vec3& b, Rgba c)
{
    v.push_back({{a.x, a.y, a.z}, c});
    v.push_back({{b.x, b.y, b.z}, c});
}

} // namespace

void appendBoxLines(std::vector<VtxL>& v, const Vec3& lo, const Vec3& hi, Rgba c)
{
    const float x[2] = {lo.x, hi.x}, y[2] = {lo.y, hi.y}, z[2] = {lo.z, hi.z};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            v.push_back({{x[0], y[i], z[j]}, c});
            v.push_back({{x[1], y[i], z[j]}, c});
            v.push_back({{x[i], y[0], z[j]}, c});
            v.push_back({{x[i], y[1], z[j]}, c});
            v.push_back({{x[i], y[j], z[0]}, c});
            v.push_back({{x[i], y[j], z[1]}, c});
        }
}

namespace {

void boxLines(std::vector<VtxL>& v, const Vec3& lo, const Vec3& hi, Rgba c)
{
    appendBoxLines(v, lo, hi, c);
}

void quad(std::vector<Vtx3>& v, const Vec3& a, const Vec3& b, const Vec3& c,
          const Vec3& d, const Vec3& n, Rgba col)
{
    const Vtx3 A{{a.x, a.y, a.z}, {n.x, n.y, n.z}, col};
    const Vtx3 B{{b.x, b.y, b.z}, {n.x, n.y, n.z}, col};
    const Vtx3 C{{c.x, c.y, c.z}, {n.x, n.y, n.z}, col};
    const Vtx3 D{{d.x, d.y, d.z}, {n.x, n.y, n.z}, col};
    v.push_back(A); v.push_back(B); v.push_back(C);
    v.push_back(A); v.push_back(C); v.push_back(D);
}

void boxTris(std::vector<Vtx3>& v, const Vec3& lo, const Vec3& hi, Rgba c)
{
    quad(v, {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z},
         {lo.x, hi.y, lo.z}, {0, 0, -1}, c);
    quad(v, {lo.x, lo.y, hi.z}, {hi.x, lo.y, hi.z}, {hi.x, hi.y, hi.z},
         {lo.x, hi.y, hi.z}, {0, 0, 1}, c);
    quad(v, {lo.x, lo.y, lo.z}, {hi.x, lo.y, lo.z}, {hi.x, lo.y, hi.z},
         {lo.x, lo.y, hi.z}, {0, -1, 0}, c);
    quad(v, {lo.x, hi.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z},
         {lo.x, hi.y, hi.z}, {0, 1, 0}, c);
    quad(v, {lo.x, lo.y, lo.z}, {lo.x, hi.y, lo.z}, {lo.x, hi.y, hi.z},
         {lo.x, lo.y, hi.z}, {-1, 0, 0}, c);
    quad(v, {hi.x, lo.y, lo.z}, {hi.x, hi.y, lo.z}, {hi.x, hi.y, hi.z},
         {hi.x, lo.y, hi.z}, {1, 0, 0}, c);
}

// --- marching tetrahedra ----------------------------------------------------
//
// The cube is split into 6 tetrahedra and each tet is resolved by its own
// 16-case table. This is deliberately NOT marching cubes: the tet table is
// eight lines instead of a 256-entry triTable, the surface is watertight by
// construction, and the extra triangles cost nothing at the sample counts here
// (the field is sampled over the BODY's bounding box, not the whole domain).

const int kTetOfCube[6][4] = {{0, 5, 1, 6}, {0, 1, 2, 6}, {0, 2, 3, 6},
                              {0, 3, 7, 6}, {0, 7, 4, 6}, {0, 4, 5, 6}};

// Corner offsets in the standard cube ordering used by kTetOfCube.
const int kCorner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                           {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

struct Sample
{
    Vec3 p;
    double v;
};

Vec3 lerpIso(const Sample& a, const Sample& b)
{
    const double d = b.v - a.v;
    const float t = float(std::abs(d) < 1e-30 ? 0.5 : (0.0 - a.v) / d);
    return {a.p.x + (b.p.x - a.p.x) * t, a.p.y + (b.p.y - a.p.y) * t,
            a.p.z + (b.p.z - a.p.z) * t};
}

void emitTri(std::vector<Vtx3>& out, const Vec3& a, const Vec3& b, const Vec3& c,
             Rgba col)
{
    Vec3 n = normalize(cross(b - a, c - a));
    const Vtx3 A{{a.x, a.y, a.z}, {n.x, n.y, n.z}, col};
    const Vtx3 B{{b.x, b.y, b.z}, {n.x, n.y, n.z}, col};
    const Vtx3 C{{c.x, c.y, c.z}, {n.x, n.y, n.z}, col};
    out.push_back(A);
    out.push_back(B);
    out.push_back(C);
}

// Solid is sdf < 0, so the surface is the 0 level set and "inside" is the
// negative side.
void marchTet(std::vector<Vtx3>& out, const Sample s[4], Rgba col)
{
    int code = 0;
    for (int i = 0; i < 4; ++i)
        if (s[i].v < 0) code |= 1 << i;
    switch (code) {
    case 0x0:
    case 0xF:
        return;
    case 0x1:
    case 0xE:
        emitTri(out, lerpIso(s[0], s[1]), lerpIso(s[0], s[2]), lerpIso(s[0], s[3]), col);
        return;
    case 0x2:
    case 0xD:
        emitTri(out, lerpIso(s[1], s[0]), lerpIso(s[1], s[3]), lerpIso(s[1], s[2]), col);
        return;
    case 0x4:
    case 0xB:
        emitTri(out, lerpIso(s[2], s[0]), lerpIso(s[2], s[1]), lerpIso(s[2], s[3]), col);
        return;
    case 0x8:
    case 0x7:
        emitTri(out, lerpIso(s[3], s[0]), lerpIso(s[3], s[2]), lerpIso(s[3], s[1]), col);
        return;
    case 0x3:
    case 0xC: {
        const Vec3 a = lerpIso(s[0], s[2]), b = lerpIso(s[0], s[3]);
        const Vec3 c = lerpIso(s[1], s[3]), d = lerpIso(s[1], s[2]);
        emitTri(out, a, b, c, col);
        emitTri(out, a, c, d, col);
        return;
    }
    case 0x5:
    case 0xA: {
        const Vec3 a = lerpIso(s[0], s[1]), b = lerpIso(s[0], s[3]);
        const Vec3 c = lerpIso(s[2], s[3]), d = lerpIso(s[2], s[1]);
        emitTri(out, a, b, c, col);
        emitTri(out, a, c, d, col);
        return;
    }
    case 0x9:
    case 0x6: {
        const Vec3 a = lerpIso(s[0], s[1]), b = lerpIso(s[0], s[2]);
        const Vec3 c = lerpIso(s[3], s[2]), d = lerpIso(s[3], s[1]);
        emitTri(out, a, b, c, col);
        emitTri(out, a, c, d, col);
        return;
    }
    default:
        return;
    }
}

} // namespace

double nodeCoord(const Build& b, int axis, int i)
{
    const dns::Grid& g = b.cas().grid;
    const double h[3] = {g.hx, g.hy, g.hz};
    const double X = double(i) * h[axis];
    if (!b.remapActive) return X;
    const dns::Focal1D* f[3] = {&b.remap.fx, &b.remap.fy, &b.remap.fz};
    return f[axis]->map(X);
}

void domainBounds(const Build& b, Vec3& centre, float& radius)
{
    if (!b.ok) {
        centre = {0, 0, 0};
        radius = 1;
        return;
    }
    const dns::Grid& g = b.cas().grid;
    const float lx = float(g.nx * g.hx), ly = float(g.ny * g.hy),
                lz = float(g.nz == 1 ? 0.0 : g.nz * g.hz);
    centre = {lx * 0.5f, ly * 0.5f, lz * 0.5f};
    radius = 0.5f * std::sqrt(lx * lx + ly * ly + lz * lz);
    if (radius <= 0) radius = 1;
}

void destroyView(Renderer& r, ViewMeshes& m)
{
    r.destroy(m.lines);
    r.destroy(m.surface);
    r.destroy(m.cells);
}

void buildViewGeometry(const Build& b, const ViewOptions& o, ViewGeometry& out)
{
    std::vector<VtxL>& lines = out.lines;
    std::vector<Vtx3>& surf = out.surface;
    std::vector<Vtx3>& cells = out.cells;
    lines.clear();
    surf.clear();
    cells.clear();
    out.stats = ViewStats{};

    if (!b.ok) return;

    const dns::Grid& g = b.cas().grid;
    const bool flat = g.nz == 1;
    const int n[3] = {g.nx, g.ny, g.nz};
    const double h[3] = {g.hx, g.hy, g.hz};

    // Node coordinate tables per axis, remap applied once.
    std::array<std::vector<double>, 3> node;
    for (int a = 0; a < 3; ++a) {
        node[a].resize(size_t(n[a]) + 1);
        for (int i = 0; i <= n[a]; ++i) node[a][size_t(i)] = nodeCoord(b, a, i);
    }
    const double lz = flat ? 0.0 : node[2].back();
    auto nodeAt = [&](int a, int i) {
        return float(a == 2 && flat ? 0.0 : node[a][size_t(i)]);
    };

    // --- domain box ---------------------------------------------------------
    if (o.showBox) {
        boxLines(lines, {0, 0, 0},
                 {float(node[0].back()), float(node[1].back()), float(lz)}, kBox);
        // The incompressible grid carries a one-cell ghost ring; the INTERIOR
        // box is what the run actually solves, and mistaking one for the other
        // is how a body ends up a cell from the boundary.
        if (b.cas().physics == dns::Physics::Incompressible) {
            const Vec3 lo{nodeAt(0, 1), nodeAt(1, 1), flat ? 0.0f : nodeAt(2, 1)};
            const Vec3 hi{nodeAt(0, n[0] - 1), nodeAt(1, n[1] - 1),
                          flat ? 0.0f : nodeAt(2, n[2] - 1)};
            boxLines(lines, lo, hi, kBoxInner);
        }
    }

    // --- grid lines ---------------------------------------------------------
    //
    // Every line at n = 512 is a grey wash, so lines are strided; the stride is
    // reported to the UI because a viewer that silently draws one line in eight
    // is lying about the resolution.
    if (o.showGrid) {
        const int maxLines = 220;
        int stride = o.gridStride > 0
                         ? o.gridStride
                         : std::max({1, n[0] / maxLines, n[1] / maxLines});
        out.stats.strideUsed = stride;
        const float zPlane = flat ? 0.0f : nodeAt(2, o.sliceAxis == 2
                                                         ? std::clamp(o.sliceIndex, 0,
                                                                      n[2])
                                                         : 0);
        const float x1 = nodeAt(0, n[0]), y1 = nodeAt(1, n[1]);
        for (int i = 0; i <= n[0]; i += stride) {
            const float x = nodeAt(0, i);
            line(lines, {x, 0, zPlane}, {x, y1, zPlane},
                 (i % (stride * 8) == 0) ? kGridMajor : kGrid);
        }
        for (int j = 0; j <= n[1]; j += stride) {
            const float y = nodeAt(1, j);
            line(lines, {0, y, zPlane}, {x1, y, zPlane},
                 (j % (stride * 8) == 0) ? kGridMajor : kGrid);
        }
        // In 3D also rule the far x-z and y-z faces, so the spanwise spacing
        // (and any z clustering) is visible without rotating the camera.
        if (!flat) {
            for (int i = 0; i <= n[0]; i += stride) {
                const float x = nodeAt(0, i);
                line(lines, {x, 0, 0}, {x, 0, float(lz)}, kGrid);
            }
            for (int k = 0; k <= n[2]; k += stride) {
                const float z = nodeAt(2, k);
                line(lines, {0, 0, z}, {x1, 0, z}, kGrid);
                line(lines, {0, 0, z}, {0, y1, z}, kGrid);
            }
        }
        // Focal plateau: the whole reason to run a clustered mesh is where
        // this box sits relative to the body, so it is drawn in the accent
        // colour, over the grid.
        if (b.remapActive) {
            Vec3 lo{0, 0, 0}, hi{0, 0, 0};
            const dns::Focal1D* f[3] = {&b.remap.fx, &b.remap.fy, &b.remap.fz};
            float* lop[3] = {&lo.x, &lo.y, &lo.z};
            float* hip[3] = {&hi.x, &hi.y, &hi.z};
            for (int a = 0; a < 3; ++a) {
                if (f[a]->active) {
                    *lop[a] = float(f[a]->map(f[a]->a));
                    *hip[a] = float(f[a]->map(f[a]->b));
                } else {
                    *lop[a] = 0;
                    *hip[a] = float(a == 2 && flat ? 0.0 : node[a].back());
                }
            }
            boxLines(lines, lo, hi, kFocal);
        }
    }

    // --- body surface -------------------------------------------------------
    if (o.showSurface && b.geo.sdf) {
        // Sample over the body's own bounding box (padded), not the domain:
        // a 96^3 lattice over a 1D box resolves the wall far better than the
        // same budget spread over an 8D x 8D tunnel.
        double lo[3], hi[3];
        if (b.hasSolidBox) {
            for (int a = 0; a < 3; ++a) {
                lo[a] = b.solidBox[size_t(a)];
                hi[a] = b.solidBox[size_t(a) + 3];
                const double pad = 0.08 * std::max(1e-9, hi[a] - lo[a]) + 2 * h[a];
                lo[a] -= pad;
                hi[a] += pad;
                lo[a] = std::max(lo[a], 0.0);
                hi[a] = std::min(hi[a], double(n[a]) * h[a]);
            }
        } else {
            for (int a = 0; a < 3; ++a) {
                lo[a] = 0;
                hi[a] = double(n[a]) * h[a];
            }
        }
        if (flat) {
            lo[2] = 0;
            hi[2] = h[2];
        }

        int ns[3];
        for (int a = 0; a < 3; ++a) {
            const double span = hi[a] - lo[a];
            ns[a] = std::clamp(int(span / std::max(h[a], 1e-30) * 1.5), 2, 96);
            if (span <= 0) ns[a] = 1;
        }
        if (flat) ns[2] = 1;

        // The lattice is uniform in REFERENCE space (lo/hi are the solid box
        // in index units, so it is dense where the cells are), and each node
        // is placed AND sampled at its physical position: the solver's body
        // is the world sdf at mapped centres (comp_ghost), so that is the one
        // that is drawn.
        auto pos = [&](int a, int i) {
            const double X = lo[a] + (hi[a] - lo[a]) * double(i) / double(ns[a]);
            if (!b.remapActive) return X;
            const dns::Focal1D* f[3] = {&b.remap.fx, &b.remap.fy, &b.remap.fz};
            return f[a]->map(X);
        };

        if (flat) {
            // Marching squares in the z-centre plane: the 2D body outline,
            // drawn as lines so it stays crisp against the grid.
            const double zc = 0.5 * h[2];
            std::vector<double> f0(size_t(ns[0] + 1) * size_t(ns[1] + 1));
            for (int j = 0; j <= ns[1]; ++j)
                for (int i = 0; i <= ns[0]; ++i)
                    f0[size_t(j) * size_t(ns[0] + 1) + size_t(i)] =
                        b.geo.sdf(pos(0, i), pos(1, j), zc);
            for (int j = 0; j < ns[1]; ++j)
                for (int i = 0; i < ns[0]; ++i) {
                    const double v[4] = {
                        f0[size_t(j) * size_t(ns[0] + 1) + size_t(i)],
                        f0[size_t(j) * size_t(ns[0] + 1) + size_t(i + 1)],
                        f0[size_t(j + 1) * size_t(ns[0] + 1) + size_t(i + 1)],
                        f0[size_t(j + 1) * size_t(ns[0] + 1) + size_t(i)]};
                    const double px[4] = {pos(0, i), pos(0, i + 1), pos(0, i + 1),
                                          pos(0, i)};
                    const double py[4] = {pos(1, j), pos(1, j), pos(1, j + 1),
                                          pos(1, j + 1)};
                    Vec3 hits[4];
                    int nh = 0;
                    for (int e = 0; e < 4 && nh < 4; ++e) {
                        const int a = e, c = (e + 1) % 4;
                        if ((v[a] < 0) == (v[c] < 0)) continue;
                        const double t = v[a] / (v[a] - v[c]);
                        hits[nh++] = {float(px[a] + (px[c] - px[a]) * t),
                                      float(py[a] + (py[c] - py[a]) * t), 0.0f};
                    }
                    if (nh >= 2) line(lines, hits[0], hits[1], kSurface);
                    if (nh == 4) line(lines, hits[2], hits[3], kSurface);
                }
        } else {
            std::vector<double> fv(size_t(ns[0] + 1) * size_t(ns[1] + 1) *
                                   size_t(ns[2] + 1));
            const size_t sx = 1, sy = size_t(ns[0] + 1),
                         sz = size_t(ns[0] + 1) * size_t(ns[1] + 1);
            for (int k = 0; k <= ns[2]; ++k)
                for (int j = 0; j <= ns[1]; ++j)
                    for (int i = 0; i <= ns[0]; ++i)
                        fv[size_t(i) * sx + size_t(j) * sy + size_t(k) * sz] =
                            b.geo.sdf(pos(0, i), pos(1, j), pos(2, k));
            Sample corner[8];
            for (int k = 0; k < ns[2]; ++k)
                for (int j = 0; j < ns[1]; ++j)
                    for (int i = 0; i < ns[0]; ++i) {
                        bool anyNeg = false, anyPos = false;
                        for (int c = 0; c < 8; ++c) {
                            const int ci = i + kCorner[c][0], cj = j + kCorner[c][1],
                                      ck = k + kCorner[c][2];
                            const double val =
                                fv[size_t(ci) * sx + size_t(cj) * sy + size_t(ck) * sz];
                            corner[c] = {{float(pos(0, ci)), float(pos(1, cj)),
                                          float(pos(2, ck))},
                                         val};
                            (val < 0 ? anyNeg : anyPos) = true;
                        }
                        if (!anyNeg || !anyPos) continue;
                        for (const auto& t : kTetOfCube) {
                            const Sample s[4] = {corner[t[0]], corner[t[1]],
                                                 corner[t[2]], corner[t[3]]};
                            marchTet(surf, s, kSurface);
                        }
                    }
        }
    }

    // --- cut cells, solid mask, apertures -----------------------------------
    if (o.showCut || o.showSolid || o.showApertures) {
        int k0 = 0, k1 = n[2] - 1;
        int i0 = 0, i1 = n[0] - 1, j0 = 0, j1 = n[1] - 1;
        if (!flat && !o.wholeBody) {
            const int c = std::clamp(o.sliceIndex, 0, n[o.sliceAxis] - 1);
            const int w = std::max(0, o.slabCells);
            int* lo[3] = {&i0, &j0, &k0};
            int* hi[3] = {&i1, &j1, &k1};
            *lo[o.sliceAxis] = std::max(0, c - w);
            *hi[o.sliceAxis] = std::min(n[o.sliceAxis] - 1, c + w);
        }

        std::size_t drawn = 0;
        const auto& cut = b.geo.cut;
        for (int k = k0; k <= k1 && !out.stats.capped; ++k)
            for (int j = j0; j <= j1 && !out.stats.capped; ++j)
                for (int i = i0; i <= i1; ++i) {
                    const std::size_t id = g.idx(i, j, k);
                    const double l = cut.lambda[id];
                    const bool isSolid = l <= 0;
                    const bool isCut = l > 0 && l < 1.0;
                    if (!isCut && !isSolid) continue;
                    if (isSolid && !o.showSolid) continue;
                    if (isCut && !o.showCut && !o.showApertures) continue;
                    if (++drawn > std::size_t(o.maxCells)) {
                        out.stats.capped = true;
                        break;
                    }
                    // In 2D the cells are pushed a sliver BEHIND the z = 0
                    // plane the grid lines and the body outline are drawn on,
                    // so the mesh stays legible on top of the lambda fill
                    // instead of z-fighting with it.
                    const Vec3 lo{nodeAt(0, i), nodeAt(1, j),
                                  flat ? -0.05f * float(h[0]) : nodeAt(2, k)};
                    const Vec3 hi{nodeAt(0, i + 1), nodeAt(1, j + 1),
                                  flat ? -0.002f * float(h[0]) : nodeAt(2, k + 1)};
                    if (isSolid) {
                        boxTris(cells, lo, hi, {kSolid.r, kSolid.g, kSolid.b, 190});
                        ++out.stats.solidDrawn;
                        continue;
                    }
                    if (o.showCut) {
                        boxTris(cells, lo, hi, lambdaColor(l, 150));
                        ++out.stats.cutDrawn;
                    }
                    if (o.showApertures) {
                        // The three low faces of this cell, drawn slightly
                        // inset so they read on top of the cell box.
                        const double ap[3] = {cut.ax[id], cut.ay[id], cut.az[id]};
                        const float e = 0.04f;
                        const Vec3 d{(hi.x - lo.x) * e, (hi.y - lo.y) * e,
                                     (hi.z - lo.z) * e};
                        if (ap[0] > 0 && ap[0] < 1)
                            quad(cells, {lo.x + d.x, lo.y, lo.z},
                                 {lo.x + d.x, hi.y, lo.z}, {lo.x + d.x, hi.y, hi.z},
                                 {lo.x + d.x, lo.y, hi.z}, {1, 0, 0},
                                 apertureColor(ap[0], 220));
                        if (ap[1] > 0 && ap[1] < 1)
                            quad(cells, {lo.x, lo.y + d.y, lo.z},
                                 {hi.x, lo.y + d.y, lo.z}, {hi.x, lo.y + d.y, hi.z},
                                 {lo.x, lo.y + d.y, hi.z}, {0, 1, 0},
                                 apertureColor(ap[1], 220));
                        if (!flat && ap[2] > 0 && ap[2] < 1)
                            quad(cells, {lo.x, lo.y, lo.z + d.z},
                                 {hi.x, lo.y, lo.z + d.z}, {hi.x, hi.y, lo.z + d.z},
                                 {lo.x, hi.y, lo.z + d.z}, {0, 0, 1},
                                 apertureColor(ap[2], 220));
                    }
                }
    }

}

void buildView(Renderer& r, const Build& b, const ViewOptions& o, ViewMeshes& out)
{
    ViewGeometry geo;
    buildViewGeometry(b, o, geo);
    out.stats = geo.stats;
    (void)r.upload(out.lines, std::span<const VtxL>(geo.lines));
    (void)r.upload(out.surface, std::span<const Vtx3>(geo.surface));
    (void)r.upload(out.cells, std::span<const Vtx3>(geo.cells));
}

} // namespace ui
