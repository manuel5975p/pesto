// Archimedes on every surface tally, imposed (no time stepping), on states the
// solved equations admit at the wall.
//
// WHY NOT THE TEXTBOOK STATE. A gas at rest under a uniform pressure gradient
// with no body force is not a state of the solved equations at a wall: a
// no-slip wall's normal momentum balance is grad p . n = rho (f - a_wall) . n
// (+ viscous), which is ZERO for a resting wall without a body force, while
// that field has grad p . n = G . n there. Every consistent wall closure must
// then read the field half a cell off the wall -- the O-grid's error was
// exactly h1/(2R) on it, first order, for that reason and no other. So the two
// compressible cases below are states that satisfy the wall's own balance:
//
//   1. FROUDE-KRYLOV + ADDED MASS (resting wall). The pressure the instant a
//      far-field gradient G starts accelerating a gas at rest past a fixed
//      cylinder: p = pInf + G . x (1 + R^2/r^2), harmonic, dp/dn = 0 on the
//      wall. Exact force F = -2 G V (2D, unit span, V = pi R^2): Archimedes
//      -G V plus the added mass of the displaced fluid -G V. Moment about the
//      centre 0, about a pivot b (c - b) x F.
//   2. ROTATING ARCHIMEDES (moving wall). The gas in rigid rotation about a
//      z shaft at s, p = pInf + rho w^2 |x - s|^2 / 2 (an exact steady Euler
//      state), the cylinder turning with it, the rotating-frame zone installed
//      so the tally transports the wall-relative velocity (--rotor-frame mrf).
//      The wall ACCELERATES, a = -w^2 (x - s), and the balance is grad p =
//      -rho a there -- the term the wall image carries. Exact force
//      -int grad p dV = -rho w^2 V (c - s), the centripetal force on the
//      displaced mass; moment about the shaft 0. Diagnosis beside it: the same
//      rig with the wall acceleration withheld from the image (the
//      zero-gradient copy the solver used before), first order.
//
// Expected accuracy, from the discretization (not from the code): the face
// pressure is the average of the fluid cell and an image that is now the
// field's own continuation to O(h^2); the remaining error is the polygon /
// staircase quadrature of a smooth integrand. Second order. Bands, both
// cases: 1% at R/h 16, and the wall closure (tally minus the exact field's
// flux through the same faces, tally_test_util.hpp stairFlux; the whole error
// on the O-grid) at a mean observed order >= 1.6 over R/h 16 -> 64. At R/h 64
// the spin closure is at the FP32 floor (~2e-6 of |F|: pInf is 50x the
// pressure differences here), which only lowers the measured order.
//
// 3. The incompressible cut-cell tally, unchanged: kBodyPressure is
//    -sum phi (a_+ - a) A; summed by parts along a row of a linear phi it is
//    EXACTLY -G times the aperture-measured volume sum (1 - a) h^2
//    (bodyforce.hpp). Pinned at the FP32 precision of the device fields,
//    1e-5 relative. Against pi R^2 that volume is a uniform-rule quadrature of
//    the chord 2 sqrt(R^2 - x^2): the sqrt endpoints make it O(h^1.5)
//    (generalized Euler-Maclaurin, endpoint term |zeta(-1/2, theta)| <= 0.21
//    each), so |V/pi R^2 - 1| <= 0.5 (h/R)^1.5. No moment. That tally reads
//    phi at cell centres through apertures and has no wall image, so the
//    linear field is a fair test of it.
#include "tally_test_util.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

using namespace dns;
using namespace tallytest;
using dnstest::check;
using std::numbers::pi;

namespace {

const double pInf = 1.0 / (kGam * 0.04); // M 0.2 on a unit speed
constexpr double eps = 1e-2;
const double G = pInf * eps / (4.0 * kR);
const double V = pi * kR * kR; // unit span

// Pivot relative to the centre, off both axes.
constexpr std::array<double, 2> kPivot{0.7 * kR, -2.3 * kR};

// (c - b) x F, z component, for a pivot b relative to the centre c = 0.
double armCross(std::array<double, 2> b, double fx, double fy)
{
    return (-b[0]) * fy - (-b[1]) * fx;
}

// ---- 1. Froude-Krylov + added mass ------------------------------------------

// Gas at rest, p = pInf + G s (1 + R^2/r^2) along axis s (0 = x, 1 = y). The
// continuation inside the body is cut off at r < R/2 (never read by a tally).
FieldFn accelerating(int axis)
{
    return [axis](double x, double y) {
        const double r2 = x * x + y * y;
        const double s = axis == 0 ? x : y;
        const double fac = r2 < 0.25 * kR * kR ? 0.0 : 1.0 + kR * kR / r2;
        return Prim2{kRho, 0.0, 0.0, pInf + G * s * fac};
    };
}

struct Level
{
    double err = 0; // |F - F_exact| / |F_exact|
    double cl = 0;  // closure: |F - F_stair| / |F_exact| (ghost); = err (O-grid)
    double geo = 0; // staircase geometry: |F_stair - F_exact| / |F_exact|
    double qc = 0;  // |Q about centre| / (|F_exact| R)
    double qb = 0;  // |Q_b - (c - b) x F_exact| / (|F_exact| |c - b|)
};

// Closure and geometry parts of a tally against the exact force (Fx, Fy).
void split(const Tally& t, double Fx, double Fy, double F, Level& l)
{
    l.err = std::hypot(t.fx - Fx, t.fy - Fy) / F;
    const bool stair = std::isfinite(t.sfx);
    l.cl = stair ? std::hypot(t.fx - t.sfx, t.fy - t.sfy) / F : l.err;
    l.geo = stair ? std::hypot(t.sfx - Fx, t.sfy - Fy) / F : 0.0;
}

// The checks every case makes, over R/h 16, 32, 64. The ORDER is checked on
// the closure part: on the ghost paths the total also carries the staircase
// body's lattice-count geometry, which has no smooth order to observe (it is
// printed, and the total is banded). It is the MEAN order over the two
// doublings: the closure error is a sum of O(h^2) per-face terms of both
// signs, so one doubling can land on a partial cancellation (measured on the
// focal KJ rig: 0.94 then 2.74, while every per-ghost image error fell 3.7x
// and 4.1x).
void checkLevels(const std::string& tag, const Level (&l)[3])
{
    const double o1 = std::log2(l[0].cl / l[1].cl), o2 = std::log2(l[1].cl / l[2].cl);
    const double order = 0.5 * (o1 + o2);
    std::printf("%-26s err R/h16 %.4e 32 %.4e 64 %.4e | closure %.4e %.4e %.4e orders "
                "%.3f %.3f mean %.3f | staircase geometry %.4e %.4e %.4e\n",
                tag.c_str(), l[0].err, l[1].err, l[2].err, l[0].cl, l[1].cl, l[2].cl, o1,
                o2, order, l[0].geo, l[1].geo, l[2].geo);
    check(l[0].err <= 0.01, tag + ": |F - F_exact| <= 1% of |F_exact| at R/h 16 "
                                  "(measured " + num(l[0].err) + ")");
    check(order >= 1.6, tag + ": wall closure converges at mean order >= 1.6 over "
                              "R/h 16 -> 64 (measured " + num(order) + ")");
}

template <class Rig>
Level measureFK(const std::string& name, Rig rig, int axis, int lvl, double shiftTol)
{
    const double Fx = axis == 0 ? -2 * G * V : 0.0, Fy = axis == 1 ? -2 * G * V : 0.0;
    const double F = 2 * G * V, arm = std::hypot(kPivot[0], kPivot[1]);
    const Tally tb = rig(lvl, accelerating(axis), kPivot);
    const Tally tc = rig(lvl, accelerating(axis), {0.0, 0.0});
    Level l;
    split(tb, Fx, Fy, F, l);
    l.qc = std::abs(tc.qz) / (F * kR);
    l.qb = std::abs(tb.qz - armCross(kPivot, Fx, Fy)) / (F * arm);
    // Q_b = Q_c + (c - b) x F_measured over the SAME staged face values, up to
    // the rounding of the lever arms -- FP64 in the ghost tally (kBodyMoment),
    // an FP32 difference wpx - cx in the O-grid one (kOGridWallFlux).
    const double shift = std::abs(tb.qz - tc.qz - armCross(kPivot, tb.fx, tb.fy)) / (F * arm);
    std::printf("  %s lvl %d: F (%+.9e, %+.9e)  Q_b %+.9e  Q_c %+.9e  shift %.3e\n",
                name.c_str(), lvl, tb.fx, tb.fy, tb.qz, tc.qz, shift);
    if (lvl == 1)
        check(shift <= shiftTol, name + ": moment shift theorem Q_b = Q_c + (c - b) "
                                        "x F on the staged faces (measured " +
                                     num(shift) + ", tol " + num(shiftTol) + ")");
    return l;
}

template <class Rig>
void froudeKrylov(const std::string& path, Rig rig, double shiftTol)
{
    for (int axis = 0; axis < 2; ++axis) {
        const std::string tag = path + (axis == 0 ? " grad x" : " grad y");
        const Level l[3] = {measureFK(path, rig, axis, 1, shiftTol),
                            measureFK(path, rig, axis, 2, shiftTol),
                            measureFK(path, rig, axis, 4, shiftTol)};
        const Level& l1 = l[0];
        std::printf("%-26s |Qc|/(F R) %.3e  Qb err %.3e\n", (tag + " FK").c_str(),
                    l1.qc, l1.qb);
        checkLevels(tag + " FK", l);
        check(l1.qc <= 0.01, tag + ": moment about the centre ~ 0, <= 1% of |F| R "
                                   "(measured " + num(l1.qc) + ")");
        check(l1.qb <= 0.01, tag + ": moment about b = (c - b) x F to 1% (measured " +
                                 num(l1.qb) + ")");
    }
}

// ---- 2. Rotating Archimedes -------------------------------------------------

constexpr double kOmega = 0.25;
constexpr std::array<double, 2> kShaft{1.5 * kR, -0.8 * kR}; // relative to the centre

FieldFn rigidRotation()
{
    return [](double x, double y) {
        const double dx = x - kShaft[0], dy = y - kShaft[1];
        return Prim2{kRho, -kOmega * dy, kOmega * dx,
                     pInf + 0.5 * kRho * kOmega * kOmega * (dx * dx + dy * dy)};
    };
}

template <class Rig>
void rotating(const std::string& path, Rig rig)
{
    // -rho w^2 V (c - s) with c = 0.
    const double Fx = kRho * kOmega * kOmega * V * kShaft[0],
                 Fy = kRho * kOmega * kOmega * V * kShaft[1];
    const double F = std::hypot(Fx, Fy);
    Level l[3], lc[3];
    double q[3];
    for (int k = 0; k < 3; ++k) {
        const int lvl = 1 << k;
        const Spin s{.omega = kOmega, .shaft = kShaft, .dropAccel = false};
        const Spin c{.omega = kOmega, .shaft = kShaft, .dropAccel = true};
        const Tally t = rig(lvl, s), tc = rig(lvl, c);
        split(t, Fx, Fy, F, l[k]);
        split(tc, Fx, Fy, F, lc[k]);
        q[k] = std::abs(t.qz) / (F * std::hypot(kShaft[0], kShaft[1]));
        std::printf("  %s spin lvl %d: F (%+.9e, %+.9e) exact (%+.9e, %+.9e) err %.4e "
                    "closure %.4e |Q_shaft|/(F d) %.3e | copy: err %.4e closure %.4e\n",
                    path.c_str(), lvl, t.fx, t.fy, Fx, Fy, l[k].err, l[k].cl, q[k],
                    lc[k].err, lc[k].cl);
    }
    std::printf("%-26s closure R/h 16/32/64 %.4e %.4e %.4e (orders %.3f, %.3f) | copy "
                "%.4e %.4e %.4e (orders %.3f, %.3f)\n",
                (path + " spin").c_str(), l[0].cl, l[1].cl, l[2].cl,
                std::log2(l[0].cl / l[1].cl), std::log2(l[1].cl / l[2].cl), lc[0].cl,
                lc[1].cl, lc[2].cl, std::log2(lc[0].cl / lc[1].cl),
                std::log2(lc[1].cl / lc[2].cl));
    const std::string tag = path + " spin";
    checkLevels(tag, l);
    check(q[0] <= 0.01, tag + ": moment about the shaft ~ 0, <= 1% of |F| |c - s| "
                              "(measured " + num(q[0]) + ")");
    check(l[1].cl < 0.25 * lc[1].cl,
          tag + ": the wall-acceleration term is live: closure error at R/h 32 below "
                "a quarter of the copy's (" + num(l[1].cl) + " vs " + num(lc[1].cl) + ")");
}

// ---- 3. cut-cell identity ---------------------------------------------------

void cutCellPath(int axis)
{
    const std::string tag = std::string("cut-cell") + (axis == 0 ? " grad x" : " grad y");
    // Gauge: phi = G s (the pInf level is not an incompressible quantity).
    const FieldFn f = [axis](double x, double y) {
        return Prim2{kRho, 0.0, 0.0, G * (axis == 0 ? x : y)};
    };
    for (int lvl = 1; lvl <= 2; ++lvl) {
        const Tally t = cutCell(lvl, f);
        const double F = axis == 0 ? t.fx : t.fy, Fo = axis == 0 ? t.fy : t.fx;
        const double Vap = axis == 0 ? t.volAx : t.volAy;
        const double ident = std::abs(F + G * Vap) / (G * Vap);
        const double rel = std::abs(Vap / V - 1.0);
        const double hr = t.hFine / kR;
        std::printf("%-22s R/h %2.0f: F %.10e  -G V_ap %.10e  ident %.2e  "
                    "|F_other|/GV %.2e  |V_ap/piR^2 - 1| %.3e (bound %.3e)\n",
                    tag.c_str(), 1.0 / hr, F, -G * Vap, ident,
                    std::abs(Fo) / (G * V), rel, 0.5 * std::pow(hr, 1.5));
        check(ident <= 1e-5, tag + ": F == -G * aperture-measured volume (FP32 "
                                   "fields, 1e-5); measured " + num(ident));
        check(std::abs(Fo) <= 1e-5 * G * V,
              tag + ": no force across the gradient (measured " +
                  num(std::abs(Fo) / (G * V)) + ")");
        check(rel <= 0.5 * std::pow(hr, 1.5),
              tag + ": aperture volume vs pi R^2 within 0.5 (h/R)^1.5 (measured " +
                  num(rel) + ")");
    }
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const auto t0 = std::chrono::steady_clock::now();
    std::printf("G = %.6e, Froude-Krylov + added mass |F| = 2 G pi R^2 = %.6e\n", G,
                2 * G * V);

    double aspect = 0;
    (void)ghostFocal(1, accelerating(0), pInf, kPivot, &aspect);
    std::printf("focal fine-cell aspect %.3f (ratio 3, blend 0.5)\n", aspect);
    check(aspect > 1.2, "focal fine cells are anisotropic");

    froudeKrylov("ghost Cartesian",
                 [](int lvl, const FieldFn& f, std::array<double, 2> pv) {
                     return ghostCartesian(lvl, f, pInf, pv);
                 },
                 1e-12);
    froudeKrylov("ghost focal",
                 [](int lvl, const FieldFn& f, std::array<double, 2> pv) {
                     return ghostFocal(lvl, f, pInf, pv);
                 },
                 1e-12);
    froudeKrylov("O-grid",
                 [](int lvl, const FieldFn& f, std::array<double, 2> pv) {
                     return ogrid(lvl, f, pInf, true, pv);
                 },
                 1e-6);

    std::printf("rotating Archimedes: omega %.3f, shaft (%.2f, %.2f) R from the centre, "
                "|F| = rho w^2 pi R^2 |c - s| = %.6e\n",
                kOmega, kShaft[0], kShaft[1],
                kRho * kOmega * kOmega * V * std::hypot(kShaft[0], kShaft[1]));
    rotating("ghost Cartesian", [](int lvl, const Spin& s) {
        return ghostCartesian(lvl, rigidRotation(), pInf, s.shaft, &s);
    });
    rotating("ghost focal", [](int lvl, const Spin& s) {
        return ghostFocal(lvl, rigidRotation(), pInf, s.shaft, nullptr, &s);
    });
    rotating("O-grid", [](int lvl, const Spin& s) {
        return ogrid(lvl, rigidRotation(), pInf, false, s.shaft, &s);
    });

    for (int axis = 0; axis < 2; ++axis) cutCellPath(axis);

    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("runtime %.2f s\n", sec);
    return dnstest::finish();
}
