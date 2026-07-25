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

} // namespace ui
