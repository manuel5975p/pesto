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
