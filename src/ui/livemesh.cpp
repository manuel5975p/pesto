#include "ui/livemesh.hpp"

#include <algorithm>
#include <cmath>

namespace ui {
namespace {

// Normalize into [0,1] against the publisher's FIXED range. Using a per-frame
// min/max instead would make a steady wake pulse as the extremes wander, which
// is the exact flicker the fixed range exists to prevent.
inline float norm(float v, float lo, float hi)
{
    return hi > lo ? std::clamp((v - lo) / (hi - lo), 0.0f, 1.0f) : 0.0f;
}

// The reference speed and the half-span of the deviation about it. The span is
// measured from the data because a magnitude's fixed range says nothing about
// how far it strays from the tunnel: at Re 200 the wake deficit is a few
// percent of uInf, and scaling it by [0, 1.6] would leave the whole picture one
// flat colour.
struct RefScale
{
    float ref = 0, span = 1;
};

RefScale referenceOf(RefMode mode, float refValue, float contrast,
                     const std::vector<float>& v, const std::vector<float>& mask)
{
    RefScale r;
    if (v.empty()) return r;
    const bool useMask = mask.size() == v.size();
    if (mode == RefMode::Mean) {
        double acc = 0;
        std::size_t n = 0;
        for (std::size_t i = 0; i < v.size(); ++i) {
            if (useMask && mask[i] < 0.5f) continue;   // solid cells are not flow
            acc += v[i];
            ++n;
        }
        r.ref = n ? float(acc / double(n)) : 0.0f;
    } else {
        r.ref = refValue;
    }
    float worst = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (useMask && mask[i] < 0.5f) continue;
        worst = std::max(worst, std::abs(v[i] - r.ref));
    }
    // A floor, so a uniform field does not divide by ~0 and explode into noise.
    r.span = std::max(worst, 1e-6f) / std::max(0.05f, contrast);
    return r;
}

// Signed deviation -> [0,1] with the reference at 0.5, which is where the
// diverging ramp is darkest.
inline float normRel(float v, const RefScale& r)
{
    return std::clamp(0.5f + 0.5f * (v - r.ref) / r.span, 0.0f, 1.0f);
}

inline Rgba shade(float t, bool diverging, float alpha)
{
    unsigned char r, g, b;
    fieldColor(t, diverging, r, g, b);
    return {r, g, b, static_cast<unsigned char>(std::clamp(alpha, 0.0f, 1.0f) * 255.0f)};
}

// One axis-aligned quad as two triangles, with a constant normal and colour.
void quad(std::vector<Vtx3>& out, const Vec3& a, const Vec3& b, const Vec3& c,
          const Vec3& d, const Vec3& n, Rgba col)
{
    // Vtx3 is a plain aggregate of float[3]s, not Vec3s.
    auto v = [&](const Vec3& p) {
        return Vtx3{{p.x, p.y, p.z}, {n.x, n.y, n.z}, col};
    };
    out.push_back(v(a));
    out.push_back(v(b));
    out.push_back(v(c));
    out.push_back(v(a));
    out.push_back(v(c));
    out.push_back(v(d));
}

// The same quad with a colour PER CORNER, which the fog needs: a slice built
// out of flat quads is a mosaic of hard-edged tiles, and a stack of mosaics is
// what reads as "layered". Corners are shared between neighbouring quads, so
// the interpolated colour is continuous across the whole slice.
void quadC(std::vector<Vtx3>& out, const Vec3& a, const Vec3& b, const Vec3& c,
           const Vec3& d, const Vec3& n, Rgba ca, Rgba cb, Rgba cc, Rgba cd)
{
    auto v = [&](const Vec3& p, Rgba col) {
        return Vtx3{{p.x, p.y, p.z}, {n.x, n.y, n.z}, col};
    };
    out.push_back(v(a, ca));
    out.push_back(v(b, cb));
    out.push_back(v(c, cc));
    out.push_back(v(a, ca));
    out.push_back(v(c, cc));
    out.push_back(v(d, cd));
}

// How much a sample contributes: zero below the threshold, ramped smoothly to
// full over `soft`. The ramp is the point -- a hard cut on a smooth field draws
// its own iso-contour, and the fog comes out as a set of nested shells with
// visible edges instead of as fog.
inline float fogGate(float mag, float floor, float soft)
{
    if (mag <= floor) return 0.0f;
    if (soft <= 1e-6f) return 1.0f;
    const float u = std::clamp((mag - floor) / soft, 0.0f, 1.0f);
    return u * u * (3.0f - 2.0f * u);
}

} // namespace

int fogOrder(const Vec3& viewDir, bool& ascending)
{
    const float ax = std::abs(viewDir.x), ay = std::abs(viewDir.y),
                az = std::abs(viewDir.z);
    const int axis = (ax >= ay && ax >= az) ? 0 : (ay >= az ? 1 : 2);
    const float d = axis == 0 ? viewDir.x : (axis == 1 ? viewDir.y : viewDir.z);
    // viewDir points AWAY from the eye, so a positive component means the far
    // end of the volume is at the high index -- and back-to-front is therefore
    // descending. Getting this backwards composites the fog inside out.
    ascending = d < 0;
    return axis;
}

void buildLivePlane(const LiveView& lf, const LiveMeshOptions& o, bool diverging,
                    float lo, float hi, std::vector<Vtx3>& out, LiveMeshStats& st)
{
    st = {};
    if (!o.showPlane || !lf.planes) return;
    const auto& planes = *lf.planes;
    if (planes.empty()) return;

    // Pick the requested axis if the run published it (a 2D run publishes only
    // the xy plane), else fall back to the first.
    const LivePlane* p = nullptr;
    for (const auto& q : planes)
        if (q.axis == o.plane) p = &q;
    if (!p) p = &planes.front();
    if (p->value.empty()) return;

    // An off-centre plane is sliced out of the volume HERE, not asked of the
    // solver: the volume is already published, so dragging the plane is a
    // memory read rather than a round trip through the run.
    const int VW = lf.volW, VH = lf.volH, VD = lf.volD;
    const bool haveVol = lf.hasVolume();
    std::vector<float> sliced;
    int sliceIdx = -1;
    if (haveVol) {
        const int nAlong = p->axis == 0 ? VW : (p->axis == 1 ? VH : VD);
        sliceIdx = std::clamp(int(o.planeAt * float(nAlong - 1) + 0.5f), 0,
                              nAlong - 1);
        const auto& v = *lf.volume;
        auto at = [&](int i, int j, int k) {
            return v[std::size_t(k) * VW * VH + std::size_t(j) * VW + i];
        };
        sliced.resize(std::size_t(p->w) * p->h);
        for (int b = 0; b < p->h; ++b)
            for (int a = 0; a < p->w; ++a) {
                const int i = p->axis == 0 ? sliceIdx : a;
                const int j = p->axis == 0 ? a : (p->axis == 1 ? sliceIdx : b);
                const int k = p->axis == 2 ? sliceIdx : b;
                sliced[std::size_t(b) * p->w + a] =
                    (i < VW && j < VH && k < VD) ? at(i, j, k) : 0.0f;
            }
    }
    // The published centre plane is the fallback: a 2D run, or one whose volume
    // exceeded the budget, still gets its cut plane.
    const std::vector<float>& val = sliced.empty() ? p->value : sliced;
    st.slicedFromVolume = !sliced.empty();
    st.sliceIndex = sliceIdx;

    // Relative colouring re-centres the ramp on a reference speed and switches
    // to the DIVERGING look, whatever the field's own hint says.
    const bool rel = o.refMode != RefMode::Off;
    const RefScale rs =
        rel ? referenceOf(o.refMode, o.refValue, o.contrast, val, p->mask)
            : RefScale{};
    st.refUsed = rs.ref;
    st.spanUsed = rs.span;
    const bool div = rel ? true : diverging;

    // Stride so a large plane stays inside the quad budget.
    int stride = 1;
    while (std::size_t(p->w / stride) * std::size_t(p->h / stride) >
               std::size_t(o.maxQuads) &&
           stride < 64)
        ++stride;
    st.stride = stride;

    const double hx = lf.hx, hy = lf.hy, hz = lf.hz;
    // World origin of the window, and the two in-plane axes. The published
    // window starts at (i0,j0,k0) in full-grid indices, so a viewer that
    // assumed 0 would draw every plane one ghost cell off.
    const double x0 = lf.i0 * hx, y0 = lf.j0 * hy, z0 = lf.k0 * hz;
    // Each plane sits at the CENTRE of the domain on its own axis -- that is
    // where the solver cut it. The window size is what says where the centre
    // is; deriving it from the volume put the xy plane on the floor of the box
    // whenever the volume was not being copied.
    // ...and it sits wherever the slice was taken: at planeAt of the domain
    // when the volume gave us a slice, at the centre when it did not.
    const double fx = st.slicedFromVolume ? double(o.planeAt) : 0.5;
    const double midX = x0 + fx * (lf.winW > 0 ? lf.winW : p->w) * hx;
    const double midY = y0 + fx * (lf.winH > 0 ? lf.winH : p->h) * hy;
    const double midZ = z0 + fx * (lf.winD > 0 ? lf.winD : 1) * hz;

    for (int j = 0; j + stride <= p->h; j += stride) {
        for (int i = 0; i + stride <= p->w; i += stride) {
            // Average the block a stride covers: dropping samples aliases a
            // wake into stripes, averaging just blurs it.
            float acc = 0;
            float solid = 0;
            int n = 0;
            for (int jj = j; jj < j + stride; ++jj)
                for (int ii = i; ii < i + stride; ++ii) {
                    const std::size_t k = std::size_t(jj) * p->w + ii;
                    acc += val[k];
                    if (!p->mask.empty()) solid += p->mask[k] < 0.5f ? 1.0f : 0.0f;
                    ++n;
                }
            if (n == 0) continue;
            if (o.hideSolid && solid > 0.5f * n) continue;
            const float mean = acc / n;
            const Rgba col = shade(rel ? normRel(mean, rs) : norm(mean, lo, hi), div,
                                   o.planeAlpha);

            const double a0 = double(i), a1 = double(i + stride);
            const double b0 = double(j), b1 = double(j + stride);
            Vec3 A, B, C, D, N;
            if (p->axis == 2) {          // xy plane at the mid z
                const float zz = float(midZ);
                A = {float(x0 + a0 * hx), float(y0 + b0 * hy), zz};
                B = {float(x0 + a1 * hx), float(y0 + b0 * hy), zz};
                C = {float(x0 + a1 * hx), float(y0 + b1 * hy), zz};
                D = {float(x0 + a0 * hx), float(y0 + b1 * hy), zz};
                N = {0, 0, 1};
            } else if (p->axis == 1) {   // xz plane at the mid y
                const float y = float(midY);
                A = {float(x0 + a0 * hx), y, float(z0 + b0 * hz)};
                B = {float(x0 + a1 * hx), y, float(z0 + b0 * hz)};
                C = {float(x0 + a1 * hx), y, float(z0 + b1 * hz)};
                D = {float(x0 + a0 * hx), y, float(z0 + b1 * hz)};
                N = {0, 1, 0};
            } else {                     // yz plane at the mid x
                const float x = float(midX);
                A = {x, float(y0 + a0 * hy), float(z0 + b0 * hz)};
                B = {x, float(y0 + a1 * hy), float(z0 + b0 * hz)};
                C = {x, float(y0 + a1 * hy), float(z0 + b1 * hz)};
                D = {x, float(y0 + a0 * hy), float(z0 + b1 * hz)};
                N = {1, 0, 0};
            }
            quad(out, A, B, C, D, N, col);
            ++st.quads;
        }
    }
}

void buildLiveFog(const LiveView& lf, const LiveMeshOptions& o, bool diverging,
                  float lo, float hi, const Vec3& viewDir,
                  std::vector<Vtx3>& out, LiveMeshStats& st)
{
    st = {};
    if (!o.showFog || !lf.hasVolume()) return;
    const int W = lf.volW, H = lf.volH, D = lf.volD;
    const auto& vol = *lf.volume;

    bool ascending = false;
    const int axis = fogOrder(viewDir, ascending);
    st.fogAxis = axis;
    st.fogAscending = ascending;
    const int nSlice = axis == 0 ? W : (axis == 1 ? H : D);
    const int step = std::max(1, nSlice / std::max(1, o.fogSlices));
    st.fogStep = step;

    // In-plane stride, from the quad budget shared with the plane view.
    const int pw = axis == 0 ? H : W;
    const int ph = axis == 2 ? H : D;
    int stride = 1;
    while (std::size_t(pw / stride) * std::size_t(ph / stride) *
                   std::size_t(nSlice / step) >
               std::size_t(o.maxQuads) &&
           stride < 64)
        ++stride;
    st.stride = stride;
    const int nA = pw / stride, nB = ph / stride;   // quads across a slice
    if (nA < 1 || nB < 1) return;

    const bool rel = o.refMode != RefMode::Off;
    static const std::vector<float> kNoMask;
    const RefScale rs =
        rel ? referenceOf(o.refMode, o.refValue, o.contrast, vol, kNoMask)
            : RefScale{};
    st.refUsed = rs.ref;
    st.spanUsed = rs.span;
    const bool div = rel ? true : diverging;

    const double hx = lf.hx, hy = lf.hy, hz = lf.hz;
    const double x0 = lf.i0 * hx, y0 = lf.j0 * hy, z0 = lf.k0 * hz;

    // Density is authored per CELL of depth and the stack only samples every
    // `step`-th cell, so each slice has to stand in for `step` of them:
    // 1-(1-a)^step is the opacity that many cells would have accumulated.
    // Without it, halving the slice count halves the thickness of the fog and
    // the density slider stops meaning anything.
    const float aCell = std::clamp(o.fogAlpha, 0.0f, 0.98f);
    const float aSlice = 1.0f - std::pow(1.0f - aCell, float(step));

    auto at = [&](int i, int j, int k) {
        return vol[std::size_t(k) * W * H + std::size_t(j) * W + i];
    };
    // A sample on the slice, in the plane's own (a, b) indices.
    auto sampleAt = [&](int sl, int a, int b) {
        const int i = axis == 0 ? sl : a;
        const int j = axis == 0 ? a : (axis == 1 ? sl : b);
        const int k = axis == 2 ? sl : b;
        return at(std::min(i, W - 1), std::min(j, H - 1), std::min(k, D - 1));
    };

    // Colour is carried on the quad CORNERS, so it interpolates across a slice
    // instead of tiling it. The nodes are shared between neighbouring quads:
    // one row of (nA+1) values per node row, computed once and reused.
    std::vector<Rgba> row0(std::size_t(nA) + 1), row1(std::size_t(nA) + 1);
    std::vector<float> gate0(std::size_t(nA) + 1), gate1(std::size_t(nA) + 1);

    for (int s = 0; s < nSlice; s += step) {
        const int sl = ascending ? s : (nSlice - 1 - s);
        // Node values for the first row; every later row reuses its predecessor.
        auto nodeRow = [&](int B, std::vector<Rgba>& col, std::vector<float>& gate) {
            const int b = std::min(B * stride, ph - 1);
            for (int A = 0; A <= nA; ++A) {
                const int a = std::min(A * stride, pw - 1);
                // Box-average the block the node stands for: sampling one cell
                // out of a 4x4 stride aliases a wake into stripes.
                float acc = 0;
                int n = 0;
                for (int bb = b - stride / 2; bb <= b + stride / 2; ++bb)
                    for (int aa = a - stride / 2; aa <= a + stride / 2; ++aa) {
                        if (aa < 0 || bb < 0 || aa >= pw || bb >= ph) continue;
                        acc += sampleAt(sl, aa, bb);
                        ++n;
                    }
                const float mean = n ? acc / float(n) : 0.0f;
                const float t = rel ? normRel(mean, rs) : norm(mean, lo, hi);
                // Quiet fluid must be transparent, or the fog is a fog bank:
                // only what departs from the middle of the range contributes.
                // With a reference speed, "the middle" IS the tunnel speed, so
                // this becomes "show me where the flow is not freestream".
                const float mag = std::abs(t - 0.5f) * 2.0f;
                const float g = fogGate(mag, o.fogFloor, o.fogSoft);
                gate[std::size_t(A)] = g;
                col[std::size_t(A)] = shade(t, div, aSlice * mag * g);
            }
        };
        nodeRow(0, row0, gate0);

        for (int B = 0; B < nB; ++B) {
            nodeRow(B + 1, row1, gate1);
            for (int A = 0; A < nA; ++A) {
                // Drop the quad only when every corner is below the threshold:
                // a quad with one lit corner is the soft edge of the fog, and
                // dropping it is what puts a hard rim around the wake.
                const std::size_t i0 = std::size_t(A), i1 = i0 + 1;
                if (gate0[i0] <= 0 && gate0[i1] <= 0 && gate1[i0] <= 0 &&
                    gate1[i1] <= 0)
                    continue;

                const double a0 = double(A * stride), a1 = double((A + 1) * stride);
                const double b0 = double(B * stride), b1 = double((B + 1) * stride);
                Vec3 P0, P1, P2, P3, N;
                if (axis == 0) {
                    const float x = float(x0 + (sl + 0.5) * hx);
                    P0 = {x, float(y0 + a0 * hy), float(z0 + b0 * hz)};
                    P1 = {x, float(y0 + a1 * hy), float(z0 + b0 * hz)};
                    P2 = {x, float(y0 + a1 * hy), float(z0 + b1 * hz)};
                    P3 = {x, float(y0 + a0 * hy), float(z0 + b1 * hz)};
                    N = {1, 0, 0};
                } else if (axis == 1) {
                    const float y = float(y0 + (sl + 0.5) * hy);
                    P0 = {float(x0 + a0 * hx), y, float(z0 + b0 * hz)};
                    P1 = {float(x0 + a1 * hx), y, float(z0 + b0 * hz)};
                    P2 = {float(x0 + a1 * hx), y, float(z0 + b1 * hz)};
                    P3 = {float(x0 + a0 * hx), y, float(z0 + b1 * hz)};
                    N = {0, 1, 0};
                } else {
                    const float z = float(z0 + (sl + 0.5) * hz);
                    P0 = {float(x0 + a0 * hx), float(y0 + b0 * hy), z};
                    P1 = {float(x0 + a1 * hx), float(y0 + b0 * hy), z};
                    P2 = {float(x0 + a1 * hx), float(y0 + b1 * hy), z};
                    P3 = {float(x0 + a0 * hx), float(y0 + b1 * hy), z};
                    N = {0, 0, 1};
                }
                quadC(out, P0, P1, P2, P3, N, row0[i0], row0[i1], row1[i1],
                      row1[i0]);
                ++st.quads;
            }
            row0.swap(row1);
            gate0.swap(gate1);
        }
        ++st.slices;
    }
    st.capped = st.quads >= std::size_t(o.maxQuads);
}

} // namespace ui
