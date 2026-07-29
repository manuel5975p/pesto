// Rotor kinematics and the case derivation behind them.
//
// Two properties carry the whole design and both are pinned here:
//   - the bin index is EXACT integer arithmetic mod bins, so a revolution
//     closes on the same table however many turns have passed (the azimuth
//     table is only cyclic because of this);
//   - rotorSdf rotates the QUERY POINT, so it is an exact rigid motion of
//     whatever sdf it wraps -- distances are preserved and a full turn
//     reproduces the original field bit for bit.
#include "dns/bodies.hpp"
#include "dns/cli.hpp"
#include "dns/config.hpp"
#include "dns/rotor.hpp"
#include "dns/trimesh.hpp"

#include "check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <expected>
#include <numbers>
#include <string>
#include <vector>

namespace {

using dnstest::check;
using namespace dns;

constexpr double pi = std::numbers::pi;

Rotor makeRotor(RotorAxis axis, double omega, int bins)
{
    return {.axis = axis,
            .shaft = {0.3, -0.2, 0.7},
            .omega = omega,
            .radius = 0.5,
            .bins = bins,
            .thicken = 0.0};
}

// --- bin arithmetic ---------------------------------------------------------

void testBins()
{
    const Rotor r = makeRotor(RotorAxis::Z, 2.0, 16);

    check(rotorBin(r, 0.0) == 0, "bin: azimuth 0 is bin 0");
    check(rotorBin(r, 2 * pi - 1e-12) == 15, "bin: just under a turn is the last bin");

    // The property the table depends on: N turns later, the SAME bin index.
    bool wraps = true;
    for (int m = 0; m < r.bins; ++m)
    {
        const double th = rotorBinAngle(r, m);
        for (int turn = -3; turn <= 3; ++turn)
            wraps = wraps && rotorBin(r, th + turn * 2 * pi) == m;
    }
    check(wraps, "bin: index is exactly periodic over +-3 revolutions");

    // Negative azimuth (a rotor spun the other way) normalizes, never returns
    // a negative index into the table.
    bool nonneg = true;
    for (int i = 1; i <= 200; ++i)
    {
        const int m = rotorBin(r, -0.037 * i);
        nonneg = nonneg && m >= 0 && m < r.bins;
    }
    check(nonneg, "bin: negative azimuth stays in [0, bins)");

    // Bin centres, so quantization error is half a bin either side.
    check(std::abs(rotorBinAngle(r, 0) - pi / 16.0) < 1e-15,
          "bin: bin 0 is built at its centre angle");
    bool inOwnBin = true;
    for (int m = 0; m < r.bins; ++m)
        inOwnBin = inOwnBin && rotorBin(r, rotorBinAngle(r, m)) == m;
    check(inOwnBin, "bin: every bin's build angle lands back in that bin");

    // theta is computed from absolute time, never accumulated.
    check(rotorAzimuth(r, 3.5) == r.omega * 3.5, "azimuth: exactly omega * t");
}

void testNaturalBins()
{
    Rotor r = makeRotor(RotorAxis::X, 1.0, 0);
    r.radius = 1.0;
    // One bin boundary per cell of tip travel.
    check(rotorNaturalBins(r, 0.01) == int(std::ceil(2 * pi / 0.01)),
          "natural bins: 2 pi R / h at the tip");
    check(rotorNaturalBins(r, 1e9) == 1, "natural bins: never below 1");
}

// --- rigid motion -----------------------------------------------------------

// A sphere off the shaft: rotating it must move it, and the sdf must stay a
// true distance (that is what sampleCut's Lipschitz prefilter and the ghost
// wall's finite-difference normal both assume).
void testRigidMotion()
{
    const Rotor r = makeRotor(RotorAxis::Z, 1.0, 16);
    const Sphere s{r.shaft[0] + 0.3, r.shaft[1], r.shaft[2], 0.1};
    const Sdf base = sdfOf(s);

    const Sdf id = rotorSdf(r, base, 0.0);
    bool same = true;
    for (int i = -6; i <= 6; ++i)
        for (int j = -6; j <= 6; ++j)
            for (int k = -6; k <= 6; ++k)
            {
                const double x = r.shaft[0] + 0.1 * i, y = r.shaft[1] + 0.1 * j,
                             z = r.shaft[2] + 0.1 * k;
                same = same && id(x, y, z) == base(x, y, z);
            }
    check(same, "rotorSdf: theta 0 is the base sdf, bit for bit");

    // A full turn is the identity to round-off (not bit-for-bit: cos/sin of
    // 2 pi are not exactly 1 and 0).
    const Sdf full = rotorSdf(r, base, 2 * pi);
    double worst = 0;
    for (int i = -6; i <= 6; ++i)
        for (int j = -6; j <= 6; ++j)
        {
            const double x = r.shaft[0] + 0.1 * i, y = r.shaft[1] + 0.1 * j;
            worst = std::max(worst, std::abs(full(x, y, r.shaft[2])
                                             - base(x, y, r.shaft[2])));
        }
    check(worst < 1e-12, "rotorSdf: a full turn reproduces the field ("
                             + std::to_string(worst) + ")");

    // Quarter turn about z, right-handed: the sphere at +x lands at +y.
    const Sdf quarter = rotorSdf(r, base, 0.5 * pi);
    check(std::abs(quarter(r.shaft[0], r.shaft[1] + 0.3, r.shaft[2]) + 0.1) < 1e-12,
          "rotorSdf: +90 deg about z carries +x to +y (right-handed)");
    check(quarter(r.shaft[0] + 0.3, r.shaft[1], r.shaft[2]) > 0,
          "rotorSdf: the body left where it was");

    // Distance is preserved: rotation is an isometry, so the sdf's magnitude
    // at a rotated probe equals the original at the original probe.
    const double px = r.shaft[0] + 0.42, py = r.shaft[1] + 0.11;
    const double th = 0.7, ct = std::cos(th), st = std::sin(th);
    const double qx = r.shaft[0] + 0.42 * ct - 0.11 * st;
    const double qy = r.shaft[1] + 0.42 * st + 0.11 * ct;
    const Sdf rot = rotorSdf(r, base, th);
    check(std::abs(rot(qx, qy, r.shaft[2]) - base(px, py, r.shaft[2])) < 1e-12,
          "rotorSdf: distances are preserved (rigid motion, not a scaling)");
}

// Dilation is a Minkowski sum with a ball: exactly a constant offset of a true
// distance field, which is why it can be a three-line lambda and still leave
// every consumer's Lipschitz assumption intact.
void testThicken()
{
    Rotor r = makeRotor(RotorAxis::Z, 1.0, 16);
    r.thicken = 0.05;
    const Sphere s{r.shaft[0], r.shaft[1], r.shaft[2], 0.1};
    const Sdf f = rotorSdf(r, sdfOf(s), 0.0);
    // The zero set moved out by exactly `thicken`.
    check(std::abs(f(r.shaft[0] + 0.15, r.shaft[1], r.shaft[2])) < 1e-12,
          "thicken: the surface moved outward by exactly the dilation");
    check(std::abs(f(r.shaft[0] + 0.35, r.shaft[1], r.shaft[2]) - 0.2) < 1e-12,
          "thicken: the field is offset, not rescaled");
}

// --- wall velocity ----------------------------------------------------------

void testWallVelocity()
{
    for (auto axis : {RotorAxis::X, RotorAxis::Y, RotorAxis::Z})
    {
        const Rotor r = makeRotor(axis, 3.0, 16);
        const int a = rotorAxisIndex(axis);
        const auto u0 = rotorWallVelocity(r, r.shaft);
        check(u0[0] == 0 && u0[1] == 0 && u0[2] == 0,
              "wall velocity: zero on the shaft axis");

        // A probe one radius out along the axis AFTER the shaft one.
        const int b = (a + 1) % 3, c = (a + 2) % 3;
        std::array<double, 3> p = r.shaft;
        p[b] += r.radius;
        const auto u = rotorWallVelocity(r, p);
        check(u[a] == 0.0, "wall velocity: no component along the shaft");
        check(std::abs(u[c] - r.omega * r.radius) < 1e-15 && u[b] == 0.0,
              "wall velocity: omega x r is right-handed about the shaft");
        const double sp = std::hypot(u[0], u[1], u[2]);
        check(std::abs(sp - std::abs(r.omega) * r.radius) < 1e-15,
              "wall velocity: tip speed is |omega| R");
    }
}

void testSweptBox()
{
    const Rotor r = makeRotor(RotorAxis::Y, 1.0, 16);
    const auto b = rotorSweptBox(r, 0.1, 0.02);
    // The disc spans the tip radius in the two plane axes and only the blade's
    // own half-span along the shaft: that asymmetry is the whole point of
    // knowing the swept extent without reading a mesh.
    check(std::abs((b[3] - b[0]) - 2 * (r.radius + 0.02)) < 1e-15,
          "swept box: x spans the tip diameter");
    check(std::abs((b[4] - b[1]) - 2 * (0.1 + 0.02)) < 1e-15,
          "swept box: the shaft axis spans the blade's own extent");
    check(std::abs((b[5] - b[2]) - 2 * (r.radius + 0.02)) < 1e-15,
          "swept box: z spans the tip diameter");
}

// --- mesh placement ---------------------------------------------------------
//
// A rotor mesh is placed and sized by its SHAFT and its SWEPT circle, not by
// its bounding box. For an odd blade count the two disagree, and anchoring on
// the box spins the model about a point that is not its axis.
void testMeshPlacement()
{
    // Three "tips" at 120 degrees on a unit-radius circle about the x axis:
    // the same asymmetry a three-bladed propeller has.
    TriMesh m;
    for (int b = 0; b < 3; ++b)
    {
        const double a = 2 * pi * b / 3;
        m.verts.push_back({0.0, std::cos(a), std::sin(a)});
    }
    m.verts.push_back({0.2, 0.0, 0.0});
    m.tris.push_back({0, 1, 2});
    m.tris.push_back({0, 1, 3});

    check(std::abs(sweptDiameter(m, 0) - 2.0) < 1e-12,
          "swept diameter is twice the largest radius about the shaft");
    const Aabb bb0 = bounds(m);
    const double ext = std::max({bb0.hi[0] - bb0.lo[0], bb0.hi[1] - bb0.lo[1],
                                 bb0.hi[2] - bb0.lo[2]});
    check(ext < 0.95 * sweptDiameter(m, 0),
          "and for three blades the bounding box is SMALLER than it");

    // Origin anchor: the mesh's own origin lands on `center`, so the shaft
    // does. The tips then all sit one scaled radius from it.
    TriMesh o = m;
    transform(o, 0.5, 0.0, 0.0, {1.0, 2.0, 3.0}, MeshAnchor::Origin);
    double rmin = 1e30, rmax = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        const double r = std::hypot(o.verts[std::size_t(i)][1] - 2.0,
                                    o.verts[std::size_t(i)][2] - 3.0);
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
    }
    check(std::abs(rmax - 0.5) < 1e-12 && std::abs(rmin - 0.5) < 1e-12,
          "the origin anchor puts every tip at the same radius from the shaft");
    check(std::abs(o.verts[3][0] - (1.0 + 0.1)) < 1e-12,
          "and scales about the origin, not about the box");

    // Box anchor, the static-body default: the tips are NOT equidistant from
    // `center` any more, which is exactly the failure a rotor must not have.
    TriMesh c = m;
    transform(c, 0.5, 0.0, 0.0, {1.0, 2.0, 3.0}, MeshAnchor::BoxCenter);
    double spread = 0.0;
    for (int i = 0; i < 3; ++i)
        spread = std::max(spread,
                          std::abs(std::hypot(c.verts[std::size_t(i)][1] - 2.0,
                                              c.verts[std::size_t(i)][2] - 3.0)
                                   - 0.5));
    check(spread > 0.05,
          "the box anchor offsets the shaft (which is why a rotor must not "
          "use it)");
}

// --- case derivation --------------------------------------------------------

std::expected<Case, ConfigError> parseCase(std::vector<const char*> args)
{
    std::vector<const char*> argv{"dns"};
    argv.insert(argv.end(), args.begin(), args.end());
    auto raw = cli::parse(int(argv.size()), const_cast<char**>(argv.data()));
    if (!raw) return std::unexpected(ConfigError{1, "parse", "cli"});
    return validate(*raw);
}

void testCase()
{
    const std::vector<const char*> base = {
        "flow", "--physics", "compressible", "--body", "cube", "--n", "256",
        "--nz", "1", "--far", "nscbc"};

    // Inertness: no --rotor-mach, no rotor, and the describe() text is the
    // static one. This is the property that keeps the measured anchors valid.
    {
        auto c = parseCase(base);
        check(c.has_value() && !c->rotor.active(),
              "case: no --rotor-mach leaves the rotor inactive");
        check(c && c->describe().find("rotor") == std::string::npos,
              "case: a static run's dry-run says nothing about a rotor");
    }

    // Resolved kinematics. c = uInf/mach is the ONE sound-speed definition;
    // tip speed must come out as tipMach * c.
    {
        auto args = base;
        args.insert(args.end(), {"--rotor-mach", "0.6", "--rotor-axis", "z"});
        auto c = parseCase(args);
        check(c.has_value(), "case: a 2D z-shaft rotor validates");
        if (c)
        {
            check(c->rotor.active(), "case: the rotor is active");
            check(std::abs(c->rotor.radius - 0.5 * c->placement.D) < 1e-15,
                  "case: tip radius is half the placement diameter");
            const double cInf = c->uInf / c->mach;
            check(std::abs(c->rotor.omega * c->rotor.radius - 0.6 * cInf) < 1e-12,
                  "case: tip speed is tipMach * freestream sound speed");
            const double hFine = std::min(c->grid.hx, c->grid.hy);
            check(c->rotor.bins == rotorNaturalBins(c->rotor, hFine),
                  "case: default bins are the natural count");
            check(c->describe().find("SPINNING") != std::string::npos,
                  "case: dry-run reports the rotor");
        }
    }

    // --blade-thicken is in FINE cells and reaches the sdf in world units.
    {
        auto args = base;
        args.insert(args.end(), {"--rotor-mach", "0.6", "--rotor-axis", "z",
                                 "--blade-thicken", "2"});
        auto c = parseCase(args);
        check(c.has_value(), "case: --blade-thicken validates");
        if (c)
        {
            const double hFine = std::min(c->grid.hx, c->grid.hy);
            check(std::abs(c->rotor.thicken - 2 * hFine) < 1e-15,
                  "case: thicken is cells * h_fine in world units");
            check(c->describe().find("THICKENED") != std::string::npos,
                  "case: a thickened blade is declared in the dry-run");
        }
    }

    // Rejections that protect a real invariant, one each.
    struct Bad { std::vector<const char*> extra; const char* what; };
    const Bad bad[] = {
        {{"--rotor-mach", "0.6"},
         "an x shaft at nz == 1 sweeps out of the plane"},
        {{"--rotor-mach", "0.9", "--rotor-axis", "z"},
         "tip + freestream Mach >= 1 has no shock capturing"},
        {{"--rotor-mach", "0.6", "--rotor-axis", "z", "--rotor-span", "0.9"},
         "--rotor-span past 0.5 leaves the placement cube"},
        {{"--rotor-mach", "0.6", "--rotor-axis", "z", "--rotor-bins", "4"},
         "under 8 bins the body teleports"},
        {{"--rotor-mach", "0.6", "--rotor-axis", "z", "--blade-thicken", "-1"},
         "a negative dilation is not a body"},
        {{"--rotor-bins", "64"},
         "a rotor flag without --rotor-mach"},
        {{"--rotor-mach", "0.6", "--rotor-axis", "z", "--cut-flux"},
         "apertures are built once from a static body"},
        {{"--rotor-mach", "0.6", "--rotor-axis", "z", "--seq", "2"},
         "a spinning body has no per-level restart"},
    };
    for (const auto& b : bad)
    {
        auto args = base;
        args.insert(args.end(), b.extra.begin(), b.extra.end());
        check(!parseCase(args).has_value(), std::string("reject: ") + b.what);
    }

    // A rotor is exempt from the --focal mesh rejection because its swept
    // extent follows from --dr and --rotor-span, never from the file.
    {
        const std::vector<const char*> m = {
            "flow", "--physics", "compressible", "--body", "prop.stl",
            "--focal", "auto", "--n", "256", "--ny", "256", "--nz", "128",
            "--dr", "6", "--far", "nscbc"};
        check(!parseCase(m).has_value(),
              "focal: a static mesh body still has no known AABB");
        auto args = m;
        args.insert(args.end(), {"--rotor-mach", "0.5"});
        auto c = parseCase(args);
        check(c.has_value(), "focal: a mesh ROTOR resolves a plateau anyway");
        if (c)
        {
            // The plateau must contain the swept disc, which is what the
            // exemption claims. Shaft along x: radius in y and z.
            const auto sw = rotorSweptBox(c->rotor, c->rotorSpanD * c->placement.D,
                                          0.0);
            bool holds = true;
            for (int a = 0; a < 3; ++a)
                if (c->focalHi[a] > c->focalLo[a])
                    holds = holds && c->focalLo[a] <= sw[a]
                            && c->focalHi[a] >= sw[a + 3];
            check(holds, "focal: the resolved plateau contains the swept volume");
        }
    }
}

} // namespace

int main()
{
    testBins();
    testNaturalBins();
    testRigidMotion();
    testThicken();
    testWallVelocity();
    testSweptBox();
    testMeshPlacement();
    testCase();
    return dnstest::finish();
}
