#include "dns/rotor.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numbers>
#include <utility>

namespace dns {
namespace {

constexpr double twoPi = 2.0 * std::numbers::pi;

// The two axes the rotation acts in, cyclically after the shaft axis. Keeping
// this one function is why nothing below has a per-axis branch.
std::pair<int, int> planeAxes(RotorAxis a)
{
    const int s = rotorAxisIndex(a);
    return {(s + 1) % 3, (s + 2) % 3};
}

} // namespace

double rotorAzimuth(const Rotor& r, double t)
{
    return r.omega * t;
}

int rotorBin(const Rotor& r, double theta)
{
    assert(r.bins > 0);
    // Wrap in bin units, not radians: the integer remainder is exact, so a
    // revolution closes on the same bin index however many turns have passed.
    const double f = theta * double(r.bins) / twoPi;
    long long m = (long long)std::floor(f) % (long long)r.bins;
    if (m < 0) m += r.bins;
    return int(m);
}

double rotorBinAngle(const Rotor& r, int m)
{
    assert(r.bins > 0 && m >= 0 && m < r.bins);
    return (double(m) + 0.5) * twoPi / double(r.bins);
}

int rotorNaturalBins(const Rotor& r, double hFine)
{
    assert(hFine > 0 && r.radius > 0);
    return std::max(1, int(std::ceil(twoPi * r.radius / hFine)));
}

Sdf rotorSdf(const Rotor& r, const Sdf& base, double theta)
{
    assert(r.thicken >= 0);
    const auto [b, c] = planeAxes(r.axis);
    const int s = rotorAxisIndex(r.axis);
    // Inverse rotation of the query point: the body turns by +theta, so a
    // world point is expressed in body coordinates by turning it back.
    const double ct = std::cos(theta), st = std::sin(theta);
    return [inner = base, shaft = r.shaft, delta = r.thicken, s, b, c, ct, st](
               double x, double y, double z) {
        const double p[3] = {x, y, z};
        const double db = p[b] - shaft[b], dc = p[c] - shaft[c];
        double q[3];
        q[s] = p[s];
        q[b] = shaft[b] + db * ct + dc * st;
        q[c] = shaft[c] - db * st + dc * ct;
        return inner(q[0], q[1], q[2]) - delta;
    };
}

std::array<double, 3> rotorWallVelocity(const Rotor& r,
                                        const std::array<double, 3>& p)
{
    const auto [b, c] = planeAxes(r.axis);
    const int s = rotorAxisIndex(r.axis);
    // omega * (e_s x d): zero along the shaft, (-d_c, +d_b) in the plane.
    std::array<double, 3> u{};
    u[s] = 0.0;
    u[b] = -r.omega * (p[c] - r.shaft[c]);
    u[c] = r.omega * (p[b] - r.shaft[b]);
    return u;
}

std::array<double, 3> rotorWallAcceleration(const Rotor& r,
                                            const std::array<double, 3>& p)
{
    const auto [b, c] = planeAxes(r.axis);
    const int s = rotorAxisIndex(r.axis);
    const double w2 = r.omega * r.omega;
    std::array<double, 3> a{};
    a[s] = 0.0;
    a[b] = -w2 * (p[b] - r.shaft[b]);
    a[c] = -w2 * (p[c] - r.shaft[c]);
    return a;
}

double rotorOmegaFromTipMach(double tipMach, double cInf, double radius)
{
    assert(radius > 0);
    return tipMach * cInf / radius;
}

std::array<double, 6> rotorSweptBox(const Rotor& r, double halfSpan, double pad)
{
    assert(r.radius > 0 && halfSpan >= 0 && pad >= 0);
    const auto [b, c] = planeAxes(r.axis);
    const int s = rotorAxisIndex(r.axis);
    double half[3];
    half[s] = halfSpan + pad;
    half[b] = half[c] = r.radius + pad;
    return {r.shaft[0] - half[0], r.shaft[1] - half[1], r.shaft[2] - half[2],
            r.shaft[0] + half[0], r.shaft[1] + half[1], r.shaft[2] + half[2]};
}

std::array<double, 6> rotorSweepBounds(const Rotor& r,
                                       const std::array<double, 6>& box)
{
    const auto [b, c] = planeAxes(r.axis);
    const int s = rotorAxisIndex(r.axis);
    // The turn moves points on circles about the shaft, so the swept radius is
    // the farthest corner of the box's own in-plane rectangle.
    double rad = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
        {
            const double db = box[b + 3 * i] - r.shaft[b];
            const double dc = box[c + 3 * j] - r.shaft[c];
            rad = std::max(rad, std::hypot(db, dc));
        }
    std::array<double, 6> out{};
    out[s] = box[s];
    out[s + 3] = box[s + 3];
    out[b] = r.shaft[b] - rad;
    out[b + 3] = r.shaft[b] + rad;
    out[c] = r.shaft[c] - rad;
    out[c + 3] = r.shaft[c] + rad;
    return out;
}

} // namespace dns
