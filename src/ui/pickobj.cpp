#include "ui/pickobj.hpp"

#include <algorithm>
#include <cmath>

namespace ui {
namespace {

// Intersect the ray with the z slab [zLo, zHi]; narrows [t0,t1] in place.
bool clipZ(const Ray& r, double zLo, double zHi, float& t0, float& t1)
{
    if (std::abs(r.d.z) < 1e-12f)
        return r.o.z >= zLo && r.o.z <= zHi;   // parallel: inside or nothing
    float ta = float((zLo - r.o.z) / r.d.z), tb = float((zHi - r.o.z) / r.d.z);
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
    return t1 >= std::max(t0, 0.0f);
}

// Quadratic for an (elliptic) cylinder about (cx, cy) with radii (rx, ry),
// infinite in z. Returns the [t0, t1] interval.
bool ellipticCylinder(const Ray& r, double cx, double cy, double rx, double ry,
                      float& t0, float& t1)
{
    // Scale into the unit circle: an ellipse is a circle in stretched
    // coordinates, and the same quadratic then covers the lens too.
    const double ex = (r.o.x - cx) / rx, ey = (r.o.y - cy) / ry;
    const double dx = r.d.x / rx, dy = r.d.y / ry;
    const double a = dx * dx + dy * dy;
    const double b = 2.0 * (ex * dx + ey * dy);
    const double c = ex * ex + ey * ey - 1.0;
    if (a < 1e-18) {
        // Looking straight down the axis: hit iff the origin is inside.
        if (c > 0) return false;
        t0 = -1e30f;
        t1 = 1e30f;
        return true;
    }
    const double disc = b * b - 4.0 * a * c;
    if (disc < 0) return false;
    const double sq = std::sqrt(disc);
    t0 = float((-b - sq) / (2.0 * a));
    t1 = float((-b + sq) / (2.0 * a));
    return true;
}

} // namespace

bool rayObject(const dns::SceneObject& o, const dns::Placement& ref,
               const dns::Grid& g, const std::array<double, 6>* meshBox,
               const Ray& r, float& tHit)
{
    const double lzFull = g.nz == 1 ? g.hz : double(g.nz) * g.hz;
    const double cx = ref.cx + o.x * ref.D, cy = ref.cy + o.y * ref.D,
                 cz = ref.cz + o.z * ref.D;
    const double rad = 0.5 * o.d * ref.D;

    float t0 = -1e30f, t1 = 1e30f;
    switch (o.kind) {
    case dns::SceneObjectKind::Sphere: {
        // The one kind that actually reads cz.
        const Vec3 oc{float(r.o.x - cx), float(r.o.y - cy), float(r.o.z - cz)};
        const float a = dot(r.d, r.d);
        const float b = 2.0f * dot(oc, r.d);
        const float c = dot(oc, oc) - float(rad * rad);
        const float disc = b * b - 4 * a * c;
        if (disc < 0 || a <= 0) return false;
        const float sq = std::sqrt(disc);
        t0 = (-b - sq) / (2 * a);
        t1 = (-b + sq) / (2 * a);
        break;
    }
    case dns::SceneObjectKind::Cylinder:
        if (!ellipticCylinder(r, cx, cy, rad, rad, t0, t1)) return false;
        if (!clipZ(r, 0.0, lzFull, t0, t1)) return false;
        break;
    case dns::SceneObjectKind::Lens: {
        // Not the exact two-disk intersection, but the ELLIPSE that bounds it:
        // strictly tighter than the AABB, which is `aspect` times too wide in x
        // (a factor of 3.5 at the default).
        const double w = o.aspect * rad;
        if (!ellipticCylinder(r, cx, cy, w, rad, t0, t1)) return false;
        if (!clipZ(r, 0.0, lzFull, t0, t1)) return false;
        break;
    }
    case dns::SceneObjectKind::Cube: {
        // The AABB that BOUNDS the cube, not the cube: the cross-section is the
        // L1 ball of half-diagonal rad, so this over-covers the four corners by
        // half a cell-diagonal. Picking a bound is the established compromise
        // here (see the lens above); the z extent, though, must be the real
        // one -- dns::sceneCube owns that rule, the editor does not restate it.
        const dns::RotatedCube c = dns::sceneCube(g, cx, cy, cz, rad);
        const Aabb3 box{{float(cx - rad), float(cy - rad), float(c.z0)},
                        {float(cx + rad), float(cy + rad), float(c.z1)}};
        if (!rayAabb(r, box, t0, t1)) return false;
        break;
    }
    case dns::SceneObjectKind::Mesh: {
        if (!meshBox) return false;
        const auto& b = *meshBox;
        const Aabb3 box{{float(b[0]), float(b[1]), float(b[2])},
                        {float(b[3]), float(b[4]), float(b[5])}};
        if (!rayAabb(r, box, t0, t1)) return false;
        break;
    }
    }
    if (t1 < 0) return false;              // entirely behind the eye
    tHit = t0 > 0 ? t0 : 0.0f;             // inside the shape: hit at zero
    return true;
}

int cycleSelection(std::span<const int> hits, int current, bool sameSpot)
{
    if (hits.empty()) return -1;
    if (!sameSpot) return hits.front();
    // Same pixel again: advance past whatever is selected, wrapping. This is
    // the Blender/Unreal rule, and the only way to reach an occluded object.
    for (std::size_t i = 0; i < hits.size(); ++i)
        if (hits[i] == current) return hits[(i + 1) % hits.size()];
    return hits.front();
}

} // namespace ui
