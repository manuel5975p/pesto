#include "ui/gizmo.hpp"

#include <algorithm>
#include <cmath>

namespace ui {
namespace {

constexpr Rgba kAxisCol[3] = {{224, 82, 79, 255},    // x red
                              {103, 192, 90, 255},   // y green
                              {79, 163, 255, 255}};  // z blue
constexpr Rgba kScreenCol{200, 210, 220, 255};
constexpr Rgba kHotCol{255, 224, 102, 255};

Rgba mixc(Rgba a, Rgba b, float t)
{
    auto f = [&](std::uint8_t x, std::uint8_t y) {
        return std::uint8_t(float(x) + (float(y) - float(x)) * t);
    };
    return {f(a.r, b.r), f(a.g, b.g), f(a.b, b.b), f(a.a, b.a)};
}

Vtx3 vtx(const Vec3& p, const Vec3& n, Rgba c)
{
    return Vtx3{{p.x, p.y, p.z}, {n.x, n.y, n.z}, c};
}

void tri(std::vector<Vtx3>& o, const Vec3& a, const Vec3& b, const Vec3& c,
         const Vec3& n, Rgba col)
{
    o.push_back(vtx(a, n, col));
    o.push_back(vtx(b, n, col));
    o.push_back(vtx(c, n, col));
}

void quad(std::vector<Vtx3>& o, const Vec3& a, const Vec3& b, const Vec3& c,
          const Vec3& d, const Vec3& n, Rgba col)
{
    tri(o, a, b, c, n, col);
    tri(o, a, c, d, n, col);
}

// Two unit vectors orthogonal to `a`, for sweeping a prism or a cone around it.
void perp(const Vec3& a, Vec3& u, Vec3& v)
{
    const Vec3 pick = std::abs(a.z) < 0.9f ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    u = normalize(cross(a, pick));
    v = cross(a, u);
}

bool isAxis(GizmoPart p) { return p >= GizmoPart::AxisX && p <= GizmoPart::AxisZ; }
int axisIndex(GizmoPart p) { return int(p) - int(GizmoPart::AxisX); }
int planeIndex(GizmoPart p) { return int(p) - int(GizmoPart::PlaneX); }

} // namespace

GizmoFrame makeGizmoFrame(const Vec3& origin, const Vec3& eye, const Vec3& camFwd,
                          const Vec3& camRight, const Vec3& camUp,
                          float worldPerPx, float uiScale, const bool enabled[3])
{
    GizmoFrame f;
    f.origin = origin;
    f.camFwd = camFwd;
    f.camRight = camRight;
    f.camUp = camUp;
    f.worldPerPx = worldPerPx;
    f.uiScale = uiScale;
    const Vec3 e[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    for (int i = 0; i < 3; ++i) {
        // Point the handles toward the viewer, so they are not buried in the
        // body. The sign lives here alone; gizmoSolve uses frame.axis too, and
        // it cancels in (tLine - grabT), so a flipped axis still drags the
        // physically correct way.
        const float s = dot(e[i], origin - eye) > 0 ? -1.0f : 1.0f;
        f.axis[i] = e[i] * s;
        f.enabled[i] = enabled ? enabled[i] : true;
    }
    return f;
}

void gizmoBuild(std::vector<Vtx3>& out, const GizmoFrame& f, const GizmoStyle& st,
                GizmoPart hot, GizmoPart active)
{
    const float s = f.worldPerPx * f.uiScale;
    const float L = st.shaftPx * s;
    const float headL = st.headLenPx * s;
    const float shaftR = st.shaftRadPx * s;
    const float headR = st.headRadPx * s;

    auto colorFor = [&](GizmoPart p, Rgba base) {
        if (active == p) return kHotCol;
        if (active == GizmoPart::None && hot == p) return mixc(base, kHotCol, 0.6f);
        return base;
    };

    // Axis shafts (prisms) + heads (cones).
    for (int i = 0; i < 3; ++i) {
        if (!f.enabled[i]) continue;
        const GizmoPart part = GizmoPart(int(GizmoPart::AxisX) + i);
        // An axis pointing almost straight at the camera projects to a dot; the
        // plane handles cover that direction instead.
        if (std::abs(dot(f.axis[i], f.camFwd)) > st.parallelCut) continue;
        const Rgba col = colorFor(part, kAxisCol[i]);
        const Vec3 a = f.axis[i];
        Vec3 u, v;
        perp(a, u, v);
        const Vec3 p0 = f.origin + a * (L * 0.12f);
        const Vec3 p1 = f.origin + a * (L - headL);
        const int n = st.coneSides;
        for (int k = 0; k < n; ++k) {
            const float t0 = 6.2831853f * float(k) / n;
            const float t1 = 6.2831853f * float(k + 1) / n;
            const Vec3 r0 = u * std::cos(t0) + v * std::sin(t0);
            const Vec3 r1 = u * std::cos(t1) + v * std::sin(t1);
            quad(out, p0 + r0 * shaftR, p1 + r0 * shaftR, p1 + r1 * shaftR,
                 p0 + r1 * shaftR, r0, col);
            // Cone: base ring to the apex, plus the base disc.
            const Vec3 apex = f.origin + a * L;
            tri(out, p1 + r0 * headR, apex, p1 + r1 * headR, r0, col);
            tri(out, p1, p1 + r1 * headR, p1 + r0 * headR, -a, col);
        }
    }

    // Plane handles: a small quad in each coordinate plane, offset from the
    // origin so it does not swallow the axis shafts.
    for (int i = 0; i < 3; ++i) {
        const GizmoPart part = GizmoPart(int(GizmoPart::PlaneX) + i);
        const int a1 = (i + 1) % 3, a2 = (i + 2) % 3;
        if (!f.enabled[a1] || !f.enabled[a2]) continue;
        const Vec3 n = f.axis[i];
        if (std::abs(dot(n, f.camFwd)) < st.edgeOnCut) continue;   // edge-on
        const Vec3 e1 = f.axis[a1], e2 = f.axis[a2];
        const float o0 = st.planeOffPx * s, o1 = (st.planeOffPx + st.planeSizePx) * s;
        Rgba col = colorFor(part, kAxisCol[i]);
        col.a = (active == part || hot == part) ? 190 : 90;
        quad(out, f.origin + e1 * o0 + e2 * o0, f.origin + e1 * o1 + e2 * o0,
             f.origin + e1 * o1 + e2 * o1, f.origin + e1 * o0 + e2 * o1, n, col);
    }

    // Screen handle: a camera-facing disc at the origin.
    {
        const Rgba col = colorFor(GizmoPart::Screen, kScreenCol);
        const float r = st.screenRadPx * s;
        const Vec3 n = -f.camFwd;
        const int n2 = 20;
        for (int k = 0; k < n2; ++k) {
            const float t0 = 6.2831853f * float(k) / n2;
            const float t1 = 6.2831853f * float(k + 1) / n2;
            tri(out, f.origin,
                f.origin + f.camRight * (r * std::cos(t0)) + f.camUp * (r * std::sin(t0)),
                f.origin + f.camRight * (r * std::cos(t1)) + f.camUp * (r * std::sin(t1)),
                n, col);
        }
    }
}

GizmoHit gizmoPick(const GizmoFrame& f, const GizmoStyle& st, const Mat4& viewProj,
                   float vpW, float vpH, const Vec2& mouse)
{
    GizmoHit hit;
    const float s = f.worldPerPx * f.uiScale;
    const float L = st.shaftPx * s;

    Vec2 originPx;
    if (!worldToScreen(viewProj, f.origin, vpW, vpH, originPx)) return hit;

    // 1. The screen handle wins at the centre: it sits under the shafts, so
    //    testing it last would make it unreachable.
    if (length(mouse - originPx) <= st.screenRadPx * f.uiScale) {
        hit.part = GizmoPart::Screen;
        return hit;
    }

    // 2. Plane quads.
    for (int i = 0; i < 3; ++i) {
        const int a1 = (i + 1) % 3, a2 = (i + 2) % 3;
        if (!f.enabled[a1] || !f.enabled[a2]) continue;
        if (std::abs(dot(f.axis[i], f.camFwd)) < st.edgeOnCut) continue;
        const Vec3 e1 = f.axis[a1], e2 = f.axis[a2];
        const float o0 = st.planeOffPx * s, o1 = (st.planeOffPx + st.planeSizePx) * s;
        const Vec3 corners[4] = {f.origin + e1 * o0 + e2 * o0,
                                 f.origin + e1 * o1 + e2 * o0,
                                 f.origin + e1 * o1 + e2 * o1,
                                 f.origin + e1 * o0 + e2 * o1};
        Vec2 poly[4];
        bool ok = true;
        for (int k = 0; k < 4 && ok; ++k)
            ok = worldToScreen(viewProj, corners[k], vpW, vpH, poly[k]);
        if (ok && pointInConvex2(mouse, poly, 4)) {
            hit.part = GizmoPart(int(GizmoPart::PlaneX) + i);
            return hit;
        }
    }

    // 3. Axis shafts, nearest in pixels.
    float best = 1e30f;
    for (int i = 0; i < 3; ++i) {
        if (!f.enabled[i]) continue;
        if (std::abs(dot(f.axis[i], f.camFwd)) > st.parallelCut) continue;
        Vec2 tip;
        if (!worldToScreen(viewProj, f.origin + f.axis[i] * L, vpW, vpH, tip))
            continue;
        const float d = distPointSegment2(mouse, originPx, tip);
        if (d <= (st.shaftRadPx + st.pickPadPx) * f.uiScale && d < best) {
            best = d;
            hit.part = GizmoPart(int(GizmoPart::AxisX) + i);
            hit.pixelDist = d;
        }
    }
    return hit;
}

bool gizmoBeginDrag(GizmoDrag& g, const GizmoFrame& f, GizmoPart part, const Ray& r)
{
    g = {};
    if (part == GizmoPart::None) return false;
    g.part = part;
    g.originAtGrab = f.origin;

    if (isAxis(part)) {
        g.axis = f.axis[axisIndex(part)];
        float tLine = 0, sRay = 0;
        if (!closestLineRay(f.origin, g.axis, r, tLine, sRay)) return false;
        g.grabT = tLine;   // THE grab anchor: grabbing the tip must not
                           // teleport the object's centre to the cursor.
        g.active = true;
        return true;
    }

    // Plane and screen handles both slide in a plane frozen at grab time. The
    // screen plane must be frozen: if it tracked the camera, orbiting mid-drag
    // would move the object.
    const Vec3 n = part == GizmoPart::Screen
                       ? -f.camFwd
                       : Vec3{planeIndex(part) == 0 ? 1.0f : 0.0f,
                              planeIndex(part) == 1 ? 1.0f : 0.0f,
                              planeIndex(part) == 2 ? 1.0f : 0.0f};
    g.plane = planeThrough(f.origin, n);
    float t = 0;
    if (!rayPlane(r, g.plane, t)) return false;
    g.grabPoint = r.o + r.d * t;
    g.active = true;
    return true;
}

Vec3 gizmoSolve(const GizmoDrag& g, const Ray& r)
{
    if (!g.active) return g.originAtGrab;
    if (isAxis(g.part)) {
        float tLine = 0, sRay = 0;
        if (!closestLineRay(g.originAtGrab, g.axis, r, tLine, sRay))
            return g.originAtGrab;   // view swung parallel: hold, do not fling
        return g.originAtGrab + g.axis * (tLine - g.grabT);
    }
    float t = 0;
    if (!rayPlane(r, g.plane, t)) return g.originAtGrab;
    return g.originAtGrab + ((r.o + r.d * t) - g.grabPoint);
}

Vec3 snapOrigin(const SnapSettings& s, const Vec3& base, const Vec3& candidate,
                const bool axisFree[3])
{
    if (s.mode == SnapSettings::Mode::Free) return candidate;
    Vec3 out = candidate;
    const float c[3] = {candidate.x, candidate.y, candidate.z};
    const float b[3] = {base.x, base.y, base.z};
    float r[3] = {c[0], c[1], c[2]};
    for (int a = 0; a < 3; ++a) {
        if (axisFree && !axisFree[a]) continue;   // constrained: leave exact
        if (s.mode == SnapSettings::Mode::Grid) {
            if (s.h[a] > 0)
                r[a] = float(std::round(double(c[a]) / s.h[a]) * s.h[a]);
        } else {
            const double q = s.fraction * s.D;
            if (q > 0)
                r[a] = float(double(b[a]) +
                             std::round((double(c[a]) - double(b[a])) / q) * q);
        }
    }
    out = {r[0], r[1], r[2]};
    return out;
}

} // namespace ui
