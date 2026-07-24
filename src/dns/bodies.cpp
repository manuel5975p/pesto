#include "dns/bodies.hpp"

#include <cassert>
#include <cmath>
#include <numbers>

namespace dns {

Sdf sdfOf(const Sphere& s)
{
    return [s](double x, double y, double z) {
        const double dx = x - s.cx, dy = y - s.cy, dz = z - s.cz;
        return std::sqrt(dx * dx + dy * dy + dz * dz) - s.r;
    };
}

Sdf sdfOf(const CylinderZ& c)
{
    return [c](double x, double y, double) {
        return std::hypot(x - c.cx, y - c.cy) - c.r;
    };
}

Sdf sdfOf(const RotatedCube& b)
{
    // L1 ball in x-y crossed with a z slab. Exact outside, a conservative
    // (Chebyshev-style) underestimate inside -- enough for the ghost band.
    return [b](double x, double y, double z) {
        const double dxy = (std::abs(x - b.cx) + std::abs(y - b.cy) - b.r)
                           / std::numbers::sqrt2;
        const double dz = std::max(b.z0 - z, z - b.z1);
        return std::max(dxy, dz);
    };
}

Sdf sdfOf(const Lens& b)
{
    // Intersection of two disks: max of the two disk distances.
    const double e = b.e(), ra = b.ra();
    return [b, e, ra](double x, double y, double) {
        const double dx = x - b.cx;
        const double d1 = std::hypot(dx, y - (b.cy - e)) - ra;
        const double d2 = std::hypot(dx, y - (b.cy + e)) - ra;
        return std::max(d1, d2);
    };
}

Sdf sdfOf(const Fence& f)
{
    // Semi-infinite slab: exact distance everywhere (the only rounding is at
    // the two top corners, where the Euclidean corner distance applies).
    return [f](double x, double y, double) {
        const double dx = std::abs(x - f.cx) - f.w;
        const double dy = y - f.yTop;
        if (dx > 0.0 && dy > 0.0) return std::hypot(dx, dy);
        return std::max(dx, dy);
    };
}

Sdf sdfOf(const Floor& f)
{
    // amp == 0: half-space, the plane distance is exact. Rippled: the
    // vertical gap divided by the max surface-slope factor is a conservative
    // near-distance (exact for a plane of that slope), which is all the cut
    // sampler's Lipschitz early-out needs.
    if (f.amp == 0.0 || f.n <= 0)
        return [f](double, double y, double) { return y - f.yTop; };
    // EXACT distance to the piecewise-linear surface (flats + the flank
    // segments of the up-to-three nearest teeth): the compressible ghost
    // wall mirrors across the surface and needs |grad sdf| ~ 1 -- a
    // slope-normalized vertical gap (conservative underestimate) made the
    // mirror points land wrong and NaN'd the fill at the first step.
    return [f](double x, double y, double) {
        auto seg = [&](double ax, double ay, double bx, double by) {
            const double vx = bx - ax, vy = by - ay;
            double t = ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy);
            t = std::min(std::max(t, 0.0), 1.0);
            const double dx = x - (ax + t * vx), dy = y - (ay + t * vy);
            return std::sqrt(dx * dx + dy * dy);
        };
        const double m0 = std::floor((x - f.x0) / f.pitch);
        double d = 1e300, ys = f.yTop;
        bool overTooth = false;
        for (double m = m0 - 1; m <= m0 + 1; m += 1.0)
        {
            if (m < 0 || m > double(f.n - 1)) continue;
            const double cm = f.x0 + (m + 0.5) * f.pitch;
            d = std::min(d, seg(cm - f.halfw, f.yTop, cm, f.yTop + f.amp));
            d = std::min(d, seg(cm, f.yTop + f.amp, cm + f.halfw, f.yTop));
            if (std::abs(x - cm) <= f.halfw)
            {
                overTooth = true;
                ys = f.yTop + f.amp * (1.0 - std::abs(x - cm) / f.halfw);
            }
        }
        if (!overTooth) d = std::min(d, std::abs(y - f.yTop));
        return y < ys ? -d : d;
    };
}

double pitRadius(double a, double depth)
{
    assert(depth > 0);
    return (a * a + depth * depth) / (2.0 * depth);
}

Sdf dimpled(Sdf base, std::span<const std::array<double, 3>> centres, double rd,
            bool zInvariant)
{
    assert(rd > 0 && !centres.empty());
    std::vector<std::array<double, 3>> q(centres.begin(), centres.end());
    return [base = std::move(base), q = std::move(q), rd, zInvariant](
               double x, double y, double z) {
        double phi = base(x, y, z);
        for (const auto& c : q)
        {
            const double ex = x - c[0], ey = y - c[1];
            const double ez = zInvariant ? 0.0 : z - c[2];
            phi = std::max(phi, rd - std::sqrt(ex * ex + ey * ey + ez * ez));
        }
        return phi;
    };
}

std::vector<std::array<double, 3>> fibonacciCentres(const Sphere& s, int n, double rd,
                                                    double depth)
{
    assert(n > 0 && depth > 0);
    using std::numbers::pi;
    const double qr = s.r + rd - depth;
    const double ga = pi * (3.0 - std::sqrt(5.0)); // golden angle
    std::vector<std::array<double, 3>> out;
    out.reserve(std::size_t(n));
    for (int i = 0; i < n; ++i)
    {
        const double z = 1.0 - 2.0 * (i + 0.5) / n;
        const double rho = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double th = ga * i;
        out.push_back({s.cx + qr * rho * std::cos(th),
                       s.cy + qr * rho * std::sin(th), s.cz + qr * z});
    }
    return out;
}

std::vector<std::array<double, 3>> ringCentres(const CylinderZ& c, int n, double rd,
                                               double depth)
{
    assert(n > 0 && depth > 0);
    using std::numbers::pi;
    const double qr = c.r + rd - depth;
    std::vector<std::array<double, 3>> out;
    out.reserve(std::size_t(n));
    for (int i = 0; i < n; ++i)
    {   // deterministic irrational phase: no dimple sits axis-aligned
        const double th = 2.0 * pi * i / n + 0.317;
        out.push_back({c.cx + qr * std::cos(th), c.cy + qr * std::sin(th), 0.0});
    }
    return out;
}

} // namespace dns
