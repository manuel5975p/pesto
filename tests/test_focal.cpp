// --focal is a PHYSICAL contract: the box (auto or explicit) names the region
// of the world that gets the fine spacing, and the body is placed in world
// coordinates. Focal1D's own parameters are REFERENCE (index-space)
// positions, so the plateau has to be inverted through the map before it is
// built. It once was not: flowFocal handed world numbers to Focal1D::make, the
// physical fine region came out ~R times narrower and shifted off the body,
// and --dry-run still printed the fine spacing as if the body sat on it.
//
// Every check here is on the map every consumer actually builds
// (dns::focalRemap of the validated Case), in PHYSICAL coordinates:
//   * the plateau endpoints map to the resolved [lo, hi];
//   * the local physical spacing at the body centre, and on every cell over
//     the body's extent plus the 3-fine-cell guard, IS h_fine to 1e-12;
//   * the plateau physically contains the body plus that guard;
//   * the blend support clears the faces, so the map is endpoint-preserving.
// A discriminating check pins that the OLD construction fails the same test.
#include "dns/config.hpp"
#include "dns/geometry.hpp"
#include "dns/remap.hpp"

#include "check.hpp"
#include "cli_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using dnstest::runCli;

namespace {

std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> out;
    std::size_t p = 0;
    while (p < s.size())
    {
        const std::size_t q = s.find(' ', p);
        const std::size_t e = q == std::string::npos ? s.size() : q;
        if (e > p) out.push_back(s.substr(p, e - p));
        p = e + 1;
    }
    return out;
}

const Focal1D& axisOf(const AxisRemap3& m, int a)
{
    return a == 0 ? m.fx : a == 1 ? m.fy : m.fz;
}

// Largest relative deviation of the physical cell width from hF over the
// reference cells overlapping the physical interval [x0, x1].
double worstSpacing(const Focal1D& f, double h, double x0, double x1, double hF)
{
    const int i0 = int(std::floor(f.inverse(x0) / h));
    const int i1 = int(std::floor(f.inverse(x1) / h));
    double worst = 0.0;
    for (int i = i0; i <= i1; ++i)
    {
        const double d = f.map((i + 1) * h) - f.map(i * h);
        const double s = f.deriv((i + 0.5) * h) * h;
        worst = std::max({worst, std::abs(d / hF - 1.0), std::abs(s / hF - 1.0)});
    }
    return worst;
}

// Validates `cmd`, then checks the physical contract on every clustered axis.
// `axes` is how many axes must cluster; `zSpan` exempts z from containment
// (the cylinder spans the box).
void physicalContract(const std::string& tag, const std::string& cmd, int axes,
                      bool zSpan = false)
{
    std::printf("--- %s ---\n", tag.c_str());
    auto c = runCli(split(cmd));
    if (!c)
    {
        std::printf("  rejected: %s\n", c.error().message.c_str());
        check(false, tag + ": validates");
        return;
    }
    check(c->focalActive, tag + ": focal active");
    const Grid& g = c->grid;
    const AxisRemap3 m = focalRemap(g, c->focalActive, c->focalLo, c->focalHi,
                                    c->focalW, c->focalRatio);
    const int n[3] = {g.nx, g.ny, g.nz};
    const double h[3] = {g.hx, g.hy, g.hz};
    const double ctr[3] = {c->placement.cx, c->placement.cy, c->placement.cz};
    const double D = c->placement.D;
    const char* ax = "xyz";
    int clustered = 0;
    for (int a = 0; a < 3; ++a)
    {
        const Focal1D& f = axisOf(m, a);
        if (c->focalHi[a] <= c->focalLo[a])
        {
            check(!f.active, std::format("{}: {} degenerate pair is the identity",
                                         tag, ax[a]));
            continue;
        }
        ++clustered;
        const std::string t = std::format("{}: {}", tag, ax[a]);
        const double L = n[a] * h[a];
        const double lo = c->focalLo[a], hi = c->focalHi[a];
        const double hF = c->focalHFine[a];
        check(f.active, t + " map active");
        std::printf("  %c physical [%.6g, %.6g] ref [%.6g, %.6g] w %.4g hF %.4e "
                    "(R %.3g, c %.4g)\n",
                    ax[a], lo, hi, f.a, f.b, f.w, hF, c->focalRatio, f.c);
        // Plateau endpoints are the PHYSICAL box.
        check(std::abs(f.map(f.a) - lo) < 1e-12 * L
                  && std::abs(f.map(f.b) - hi) < 1e-12 * L,
              t + " plateau maps exactly onto the physical box");
        // The blend is physical too: map(a-w) = lo - W.
        check(std::abs(f.map(f.a - f.w) - (lo - c->focalW[a])) < 1e-12 * L
                  && std::abs(f.map(f.b + f.w) - (hi + c->focalW[a])) < 1e-12 * L,
              t + " blend ends at lo - W and hi + W physically");
        check(std::abs(hF - f.c * f.r * h[a]) <= 1e-14 * hF,
              t + " reported h_fine is the built map's plateau spacing");
        check(std::abs(c->focalHCoarse[a] / hF - c->focalRatio) < 1e-12,
              t + " coarse/fine is exactly --focal-ratio");
        // Endpoint-preserving, support clear of the faces.
        check(f.a - f.w >= 4 * h[a] && f.b + f.w <= L - 4 * h[a],
              t + " blend support clears both faces by >= 4 cells");
        check(f.map(0.0) == 0.0 && std::abs(f.map(L) - L) < 1e-12 * L,
              t + " map preserves the box");

        if (a == 2 && zSpan) continue;
        // Containment, physically.
        const double bLo = ctr[a] - 0.5 * D, bHi = ctr[a] + 0.5 * D;
        const double guard = 3.0 * hF;
        check(lo <= bLo - guard && hi >= bHi + guard,
              t + " plateau physically holds the body + 3 fine cells");
        // Spacing at the body centre: the reference cell that holds it.
        const int ic = int(std::floor(f.inverse(ctr[a]) / h[a]));
        const double dc = f.map((ic + 1) * h[a]) - f.map(ic * h[a]);
        std::printf("  %c spacing at body centre %.6e (h_fine %.6e, ref h %.4e)\n",
                    ax[a], dc, hF, h[a]);
        check(std::abs(dc / hF - 1.0) < 1e-12,
              t + " physical spacing at the body centre is h_fine");
        // Every cell overlapping body +- 2 h_fine: with the 3-cell guard
        // inside the plateau those cells lie WHOLLY on it (the cell holding
        // the guard's own end may start in the blend, whose quintic onset is
        // only C^2-flat, so it is not held to 1e-12).
        const double worst = worstSpacing(f, h[a], bLo - 2 * hF, bHi + 2 * hF, hF);
        std::printf("  %c worst spacing deviation over body + 2 h_fine %.2e\n",
                    ax[a], worst);
        check(worst < 1e-12,
              t + " physical spacing is h_fine over the whole body + guard");
        check(worstSpacing(f, h[a], bLo - guard, bHi + guard, hF) < 1e-6,
              t + " the guard's end cell is fine to the blend onset");
        // The body is D / h_fine fine cells across, as --dry-run says.
        const double cells = (f.inverse(bHi) - f.inverse(bLo)) / h[a];
        check(std::abs(cells - D / hF) < 1e-9 * cells,
              t + " body spans D / h_fine reference cells");
    }
    check(clustered == axes,
          std::format("{}: {} axes cluster (got {})", tag, axes, clustered));
}

// The pre-fix construction (world numbers straight into Focal1D::make) on a
// body away from the box centre: the physical fine region misses the body.
// This is the property the contract test above discriminates.
void oldConstructionMissesBody()
{
    std::printf("--- old construction ---\n");
    auto c = runCli(split("flow --physics compressible --body cylinder "
                          "--focal auto --n 256 --nz 1"));
    if (!c) { check(false, "cylinder auto validates"); return; }
    const Grid& g = c->grid;
    const double L = g.nx * g.hx;
    const Focal1D bad = Focal1D::make(L, c->focalLo[0], c->focalHi[0],
                                      c->focalW[0], c->focalRatio);
    const double hFbad = bad.c * bad.r * g.hx;
    const double bLo = c->placement.cx - 0.5 * c->placement.D;
    const double bHi = c->placement.cx + 0.5 * c->placement.D;
    const double worst = worstSpacing(bad, g.hx, bLo, bHi, hFbad);
    std::printf("  reference-space plateau: physical [%.4f, %.4f], body "
                "[%.4f, %.4f], worst spacing deviation %.3f\n",
                bad.map(bad.a), bad.map(bad.b), bLo, bHi, worst);
    check(worst > 0.1, "the reference-space construction is NOT fine over the "
                       "body (the test discriminates)");
    check(bad.map(bad.a) > bLo || bad.map(bad.b) < bHi,
          "the reference-space construction does not hold the body physically");
}

// No --focal: the map every consumer builds is the identity, bit for bit.
void absentIsIdentity()
{
    auto c = runCli(split("flow --physics compressible --body cylinder "
                          "--n 128 --nz 1"));
    if (!c) { check(false, "plain compressible validates"); return; }
    check(!c->focalActive, "no --focal: inactive");
    const AxisRemap3 m = focalRemap(c->grid, c->focalActive, c->focalLo,
                                    c->focalHi, c->focalW, c->focalRatio);
    bool same = !m.fx.active && !m.fy.active && !m.fz.active;
    for (int i = 0; i <= 128; ++i)
    {
        const double X = i * c->grid.hx;
        same = same && m.fx.map(X) == X && m.fy.map(X) == X && m.fz.map(X) == X;
    }
    check(same, "no --focal: focalRemap is the exact identity");
}

// fromPhysical against hand-checkable numbers, independent of the validator.
void fromPhysicalUnit()
{
    const double L = 8.0;
    for (double R : {2.0, 4.0, 16.0})
        for (double W : {0.25, 1.0})
        {
            const Focal1D f = Focal1D::fromPhysical(L, 3.0, 3.5, W, R);
            const std::string t = std::format("fromPhysical R {} W {}", R, W);
            check(f.active, t + ": active");
            check(std::abs(f.map(f.a) - 3.0) < 1e-13
                      && std::abs(f.map(f.b) - 3.5) < 1e-13,
                  t + ": plateau endpoints");
            check(std::abs(f.map(f.a - f.w) - (3.0 - W)) < 1e-13
                      && std::abs(f.map(f.b + f.w) - (3.5 + W)) < 1e-13,
                  t + ": blend endpoints");
            check(std::abs(f.deriv(0.5 * (f.a + f.b)) - f.c / R) < 1e-15,
                  t + ": plateau slope c/R");
            check(std::abs(f.map(L) - L) < 1e-13, t + ": endpoint preserving");
        }
    check(!Focal1D::fromPhysical(L, 3.0, 3.5, 0.5, 1.0).active
              && !Focal1D::fromPhysical(L, 3.5, 3.0, 0.5, 4.0).active
              && !Focal1D::fromPhysical(L, 3.0, 3.5, 0.0, 4.0).active,
          "fromPhysical: invalid parameters give the identity");
    // A wide plateau still fits in reference space -- c grows and the coarse
    // cells with it -- but a blend that physically reaches a face cannot.
    check(Focal1D::fromPhysical(L, 2.0, 6.0, 0.5, 4.0).clears(0.0),
          "fromPhysical: a wide plateau is paid for by coarser far cells");
    check(!Focal1D::fromPhysical(L, 0.4, 6.0, 0.5, 4.0).clears(0.0),
          "fromPhysical: a blend past the face fails clears()");
}

// sampleCutMapped is the solver's body in index space: identity = sampleCut
// bit for bit, and on a focal mesh its fluid fraction tracks the sign of the
// world sdf at the MAPPED centres (what the ghost wall classifies on).
void mappedCut()
{
    std::printf("--- sampleCutMapped ---\n");
    const double L = 2 * std::numbers::pi;
    const Grid g = Grid::make(128, 128, 1, L, L, 1.0);
    const double cx = L / 3, cy = L / 2 + 0.013, rb = L / 16;
    const Sdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - rb;
    };
    const AxisRemap3 id{Focal1D::identity(L), Focal1D::identity(L),
                        Focal1D::identity(1.0)};
    auto a = sampleCut(g, sdf, {8, 8});
    auto b = sampleCutMapped(g, sdf, id, {8, 8});
    check(a && b && a->lambda == b->lambda && a->ax == b->ax && a->ay == b->ay
              && a->az == b->az,
          "sampleCutMapped on the identity is sampleCut bit for bit");

    const double lo[3] = {cx - 1.5 * rb, cy - 1.5 * rb, 0.0};
    const double hi[3] = {cx + 1.5 * rb, cy + 1.5 * rb, 0.0};
    const double w[3] = {1.5 * rb, 1.5 * rb, 0.0};
    const AxisRemap3 m = focalRemap(g, true, lo, hi, w, 4.0);
    auto c = sampleCutMapped(g, sdf, m, {8, 8});
    if (!c) { check(false, "sampleCutMapped on a focal mesh"); return; }
    int agree = 0, total = 0, solid = 0, signSolid = 0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const double l = c->lambda[g.idx(i, j, 0)];
            const bool neg = sdf(m.fx.map((i + 0.5) * g.hx),
                                 m.fy.map((j + 0.5) * g.hy), 0.0) < 0.0;
            solid += l <= 0.0;
            signSolid += neg;
            if (l <= 0.0 || l >= 1.0)
            {
                ++total;
                agree += (l <= 0.0) == neg;
            }
        }
    const double hFx = m.fx.c * m.fx.r * g.hx, hFy = m.fy.c * m.fy.r * g.hy;
    const double expect = std::numbers::pi * rb * rb / (hFx * hFy);
    std::printf("  solid %d, mapped-centre sign %d, fine-lattice area %.0f\n",
                solid, signSolid, expect);
    check(agree == total, "every uncut cell agrees with the mapped-centre sign");
    check(std::abs(signSolid / expect - 1.0) < 0.03,
          "the mapped cut's body is the fine-lattice body");
}

} // namespace

int main()
{
    mappedCut();
    fromPhysicalUnit();
    absentIsIdentity();
    oldConstructionMissesBody();

    const std::string comp = "flow --physics compressible ";
    physicalContract("cylinder auto R4 2D",
                     comp + "--body cylinder --focal auto --n 256 --nz 1", 2);
    physicalContract("cylinder auto R2 Re300 2D (README table)",
                     comp + "--body cylinder --focal auto --focal-ratio 2 "
                            "--re 300 --n 256 --nz 1", 2);
    physicalContract("cube auto 3D",
                     comp + "--body cube --focal auto --focal-blend 0.8 "
                            "--re 2000 --les 1.35 --sgs sigma --nscbc-strip 4 "
                            "--n 128 --nz 128", 3);
    physicalContract("sphere auto 3D",
                     comp + "--body sphere --focal auto --focal-blend 0.8 "
                            "--n 128 --nz 128", 3);
    physicalContract("cylinder auto 3D (z spans)",
                     comp + "--body cylinder --focal auto --n 192 --nz 16", 2,
                     true);
    physicalContract("mesh rotor auto 3D R8",
                     comp + "--body prop.stl --robust --n 384 --ny 256 --nz 256 "
                            "--focal auto --focal-ratio 8 --focal-blend 1 "
                            "--rotor-mach 0.41 --rotor-axis x --rotor-frame mrf "
                            "--mach 0.08 --re 220", 3);

    // Explicit boxes, built from the placement so they sit off the box centre
    // (cx is at ~L/3, where map(X) != X is largest).
    {
        auto c = runCli(split(comp + "--body sphere --n 128 --nz 128"));
        check(c.has_value(), "sphere placement validates");
        if (c)
        {
            const auto& p = c->placement;
            const double e = 0.5 * p.D + 0.2 * p.D;
            const std::string box = std::format(
                "{:.9g},{:.9g},{:.9g},{:.9g},{:.9g},{:.9g}", p.cx - e,
                p.cx + e + 0.5 * p.D, p.cy - e, p.cy + e, p.cz - e, p.cz + e);
            physicalContract("sphere explicit 3D",
                             comp + "--body sphere --n 128 --nz 128 "
                                    "--focal-blend 0.7 --focal " + box, 3);
            physicalContract("sphere explicit 3D R2 blend 0.8",
                             comp + "--body sphere --n 128 --nz 128 --focal-ratio 2 "
                                    "--focal-blend 0.8 --focal " + box, 3);
            // x and y only: z stays uniform.
            const std::string boxXY = std::format(
                "{:.9g},{:.9g},{:.9g},{:.9g},0,0", p.cx - e, p.cx + e,
                p.cy - e, p.cy + e);
            physicalContract("sphere explicit x,y only",
                             comp + "--body sphere --n 128 --nz 128 "
                                    "--focal-blend 0.7 --focal " + boxXY, 2);
        }
    }

    // Rejections that are only right in the physical frame.
    {
        auto c = runCli(split(comp + "--body cylinder --n 256 --nz 1"));
        if (c)
        {
            const auto& p = c->placement;
            // A box that holds the body in REFERENCE terms only: [cx-D, cx+D]
            // in world units is fine, but a box of the body's own width with
            // no guard is not.
            const std::string tight = std::format(
                "{:.9g},{:.9g},{:.9g},{:.9g},0,0", p.cx - 0.5 * p.D,
                p.cx + 0.5 * p.D, p.cy - p.D, p.cy + p.D);
            check(!runCli(split(comp + "--body cylinder --n 256 --nz 1 --focal "
                                + tight)).has_value(),
                  "explicit box without the physical guard is rejected");
        }
        // Unaffordable: z is 4 D deep at --nz 64; a sphere plateau at R 4
        // takes ~4.6 D of reference cells, so the coarse cells grow until the
        // 4-cell face margin no longer fits beside the blend. Accepted before
        // the fix only because the plateau was never where the body is.
        auto u = runCli(split(comp + "--body sphere --focal auto "
                                     "--focal-blend 0.8 --n 128 --nz 64"));
        check(!u && u.error().code == 2,
              "a plateau the grid cannot afford is rejected (exit 2)");
    }

    if (dnstest::gFailures == 0) std::printf("focal: ok\n");
    return dnstest::finish();
}
