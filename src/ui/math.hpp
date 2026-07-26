// The three-line linear algebra this viewer needs. Column-major 4x4 like GLSL,
// right-handed world, and a Vulkan-convention projection (clip z in [0,1], y
// down) so nothing has to flip a matrix at the last moment.
//
// World axes are the SOLVER's: x streamwise, y vertical/wall-normal, z
// spanwise. The camera's "up" is +z, which is the same convention docs/
// viewer-format.md pins for the web viewer -- keeping them equal means a
// screenshot from either tool can be compared without mirroring.
#pragma once

#include <cmath>

namespace ui {

struct Vec3
{
    float x = 0, y = 0, z = 0;

    Vec3() = default;
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
};

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a)
{
    const float l = length(a);
    return l > 0 ? a * (1.0f / l) : Vec3{0, 0, 1};
}

// Column-major: m[col * 4 + row], matching std140/GLSL mat4.
struct Mat4
{
    float m[16]{};

    static Mat4 identity()
    {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1;
        return r;
    }

    Mat4 operator*(const Mat4& b) const
    {
        Mat4 r;
        for (int c = 0; c < 4; ++c)
            for (int i = 0; i < 4; ++i) {
                float s = 0;
                for (int k = 0; k < 4; ++k) s += m[k * 4 + i] * b.m[c * 4 + k];
                r.m[c * 4 + i] = s;
            }
        return r;
    }

    // Transform a point (w = 1); returns clip space, w in outW.
    Vec3 transform(const Vec3& p, float& outW) const
    {
        const float x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
        const float y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
        const float z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
        outW = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
        return {x, y, z};
    }
};

inline Mat4 lookAt(const Vec3& eye, const Vec3& at, const Vec3& up)
{
    const Vec3 f = normalize(at - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0] = s.x;  r.m[4] = s.y;  r.m[8] = s.z;   r.m[12] = -dot(s, eye);
    r.m[1] = u.x;  r.m[5] = u.y;  r.m[9] = u.z;   r.m[13] = -dot(u, eye);
    r.m[2] = -f.x; r.m[6] = -f.y; r.m[10] = -f.z; r.m[14] = dot(f, eye);
    return r;
}

// Vulkan clip space: z in [0,1], y pointing down in NDC (the m[5] negation).
inline Mat4 perspective(float fovyRad, float aspect, float zn, float zf)
{
    const float t = 1.0f / std::tan(fovyRad * 0.5f);
    Mat4 r;
    r.m[0] = t / aspect;
    r.m[5] = -t;
    r.m[10] = zf / (zn - zf);
    r.m[11] = -1.0f;
    r.m[14] = zn * zf / (zn - zf);
    return r;
}

// Orthographic, same conventions. The 2D (nz == 1) view uses this: a cut-cell
// aperture read off a perspective image is a misread aperture.
inline Mat4 orthographic(float halfW, float halfH, float zn, float zf)
{
    Mat4 r = Mat4::identity();
    r.m[0] = 1.0f / halfW;
    r.m[5] = -1.0f / halfH;
    r.m[10] = 1.0f / (zn - zf);
    r.m[14] = zn / (zn - zf);
    return r;
}

// --- rays, planes, boxes -----------------------------------------------------
//
// Everything a translate gizmo and an object pick need, as pure functions. They
// live here rather than inline in app.cpp so `dns-ui --check` can drive them
// with no window and no device -- which is the only way the drag solve gets
// tested at all.

struct Vec2
{
    float x = 0, y = 0;
    Vec2() = default;
    Vec2(float a, float b) : x(a), y(b) {}
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
};
inline float dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
inline float length(const Vec2& a) { return std::sqrt(dot(a, a)); }

// `d` is NOT required to be unit length; each user documents what it means.
struct Ray { Vec3 o, d; };
struct Aabb3 { Vec3 lo, hi; };
// dot(n, p) == dist, n unit.
struct Plane { Vec3 n; float dist = 0; };

inline Plane planeThrough(const Vec3& p, const Vec3& n)
{
    const Vec3 u = normalize(n);
    return {u, dot(u, p)};
}

// Slab test. tEnter may be negative when the origin is inside the box.
inline bool rayAabb(const Ray& r, const Aabb3& b, float& tEnter, float& tExit)
{
    float t0 = -1e30f, t1 = 1e30f;
    const float o[3] = {r.o.x, r.o.y, r.o.z};
    const float d[3] = {r.d.x, r.d.y, r.d.z};
    const float lo[3] = {b.lo.x, b.lo.y, b.lo.z};
    const float hi[3] = {b.hi.x, b.hi.y, b.hi.z};
    for (int a = 0; a < 3; ++a) {
        if (std::abs(d[a]) < 1e-12f) {
            if (o[a] < lo[a] || o[a] > hi[a]) return false;
            continue;
        }
        float ta = (lo[a] - o[a]) / d[a], tb = (hi[a] - o[a]) / d[a];
        if (ta > tb) std::swap(ta, tb);
        t0 = std::max(t0, ta);
        t1 = std::min(t1, tb);
    }
    if (t1 < std::max(t0, 0.0f)) return false;
    tEnter = t0;
    tExit = t1;
    return true;
}

// False when the ray is (near) parallel to the plane.
inline bool rayPlane(const Ray& r, const Plane& p, float& t, float eps = 1e-7f)
{
    const float dn = dot(p.n, r.d);
    if (std::abs(dn) < eps) return false;
    t = (p.dist - dot(p.n, r.o)) / dn;
    return true;
}

// Closest approach between the INFINITE line a + u*t and the ray. Returns the
// line parameter and the ray parameter. False when the two are within eps of
// parallel -- the caller must then hold its previous value rather than solve,
// or the object flies to infinity as the view swings through the axis.
inline bool closestLineRay(const Vec3& a, const Vec3& u, const Ray& r,
                           float& tLine, float& sRay, float eps = 1e-6f)
{
    const Vec3 w0 = a - r.o;
    const float A = dot(u, u), B = dot(u, r.d), C = dot(r.d, r.d);
    const float D = dot(u, w0), E = dot(r.d, w0);
    const float det = A * C - B * B;
    if (std::abs(det) < eps) return false;
    tLine = (B * E - C * D) / det;
    sRay = (A * E - B * D) / det;
    return true;
}

// Pixel-space distance from p to the segment ab.
inline float distPointSegment2(const Vec2& p, const Vec2& a, const Vec2& b)
{
    const Vec2 ab = b - a;
    const float len2 = dot(ab, ab);
    if (len2 < 1e-12f) return length(p - a);
    float t = dot(p - a, ab) / len2;
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    return length(p - (a + ab * t));
}

// Point in a convex polygon, either winding.
inline bool pointInConvex2(const Vec2& p, const Vec2* poly, int n)
{
    bool neg = false, pos = false;
    for (int i = 0; i < n; ++i) {
        const Vec2& a = poly[i];
        const Vec2& b = poly[(i + 1) % n];
        const float cr = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if (cr < 0) neg = true;
        if (cr > 0) pos = true;
        if (neg && pos) return false;
    }
    return true;
}

// World -> VIEWPORT pixels, top-left origin. False when the point is behind the
// eye. The projections above already negate m[5], so NDC y points down and
// there is no second flip here: getting that wrong mirrors every gizmo hit-test
// vertically, which is why `--check` round-trips this against Camera::ray.
inline bool worldToScreen(const Mat4& viewProj, const Vec3& p, float vpW,
                          float vpH, Vec2& out)
{
    float w = 0;
    const Vec3 clip = viewProj.transform(p, w);
    if (w <= 1e-6f) return false;
    out = {(clip.x / w * 0.5f + 0.5f) * vpW, (clip.y / w * 0.5f + 0.5f) * vpH};
    return true;
}

} // namespace ui
