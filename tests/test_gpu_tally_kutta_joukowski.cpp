// Kutta-Joukowski on every surface tally, imposed (no time stepping): the
// force of the KJ wall-pressure distribution, Fx = 0 (d'Alembert), Fy =
// -rho U Gamma per unit span -- which pins the SIGN of cl: a CCW circulation
// speeds the flow under the body and pushes it DOWN.
//
// WHY NOT THE POTENTIAL FLOW ITSELF. The inviscid flow past the cylinder slips
// along the wall at 2 U sin(theta) + Gamma/(2 pi R); every wall here is
// no-slip, so its image of that field is not the field (the face velocity
// halves, the convective flux through the staircase faces is wrong at O(1)
// per face), and its pressure has dp/dn = rho u_t^2/R at the wall where a
// resting no-slip wall's own momentum balance says 0. Measured, every path
// was then first order with a large constant (O-grid 5.8% at R/h 16, matching
// the model 1 - (1 + (1+e)^-2)/(2(1+e)), e = h1/(2R)) -- the state's
// inconsistency with the wall, not a defect of the tally. So the state here is
// one the resting wall admits: the gas at rest, its pressure on the wall the
// Bernoulli (KJ) distribution
//   p_w = pInf + rho U^2/2 (1 - g^2 - 4 sin^2 th + 4 g sin th), g = Gamma/(2 pi R U),
// continued off the wall harmonically with zero normal derivative:
//   cos 2th -> cos 2th ((r/R)^2 + (R/r)^2)/2,  sin th -> sin th (r/R + R/r)/2.
// (The pressure of a gas being accelerated from rest by a potential field.)
// The exact force is the integral of p_w alone: (0, -rho U Gamma).
//
// Expected accuracy, from the discretization: the image continues this field
// to O(h^2), so the wall closure is second order (checked on tally minus the
// exact field's flux through the same faces); the staircase body's geometry
// is a lattice count with no smooth order (printed). Bands: 1% at R/h 16,
// closure mean order >= 1.6 over R/h 16 -> 64. The cut-cell tally reads phi at
// cell centres through
// apertures (no wall image): 3% and order >= 1, as for any first-order
// staircase-aperture quadrature.
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

constexpr double U = 1.0, kMach = 0.1;
const double pInf = kRho * U * U / (kGam * kMach * kMach);
const double Gam0 = pi * kR * U; // g = 0.5: two surface stagnation points

// The compatible rest state; cut off inside r < R/2 (never read by a tally).
FieldFn restKJ(double Gam, double pRef)
{
    return [Gam, pRef](double x, double y) {
        const double r2 = x * x + y * y, r = std::sqrt(r2);
        if (r < 0.5 * kR) return Prim2{kRho, 0.0, 0.0, pRef};
        const double c = x / r, s = y / r, gt = Gam / (2.0 * pi * kR * U);
        const double c2 = c * c - s * s, q = r / kR;
        const double p = pRef + 0.5 * kRho * U * U *
                                    (-(1.0 + gt * gt) + c2 * (q * q + 1.0 / (q * q)) +
                                     2.0 * gt * s * (q + 1.0 / q));
        return Prim2{kRho, 0.0, 0.0, p};
    };
}

template <class Rig>
void path(const std::string& name, Rig rig, bool wallImage)
{
    const double Fy = -kRho * U * Gam0, F = std::abs(Fy);
    double e[3], cl[3];
    for (int k = 0; k < 3; ++k) {
        const int lvl = 1 << k;
        const Tally t = rig(lvl, Gam0);
        e[k] = std::hypot(t.fx, t.fy - Fy) / F;
        const bool stair = std::isfinite(t.sfx);
        cl[k] = stair ? std::hypot(t.fx - t.sfx, t.fy - t.sfy) / F : e[k];
        std::printf("%-16s R/h %2.0f: F = (%+.6e, %+.6e)  exact (0, %+.6e)  err %.4e  "
                    "closure %.4e%s\n",
                    name.c_str(), kR / t.hFine, t.fx, t.fy, Fy, e[k], cl[k],
                    stair ? "" : " (= err)");
        check(t.fy < 0.0, name + ": CCW circulation gives Fy < 0 (lvl " +
                              std::to_string(lvl) + ")");
    }
    const Tally neg = rig(1, -Gam0);
    std::printf("%-16s Gamma < 0: Fy %+.6e\n", name.c_str(), neg.fy);
    check(neg.fy > 0.0, name + ": CW circulation gives Fy > 0");
    const double o1 = std::log2(cl[0] / cl[1]), o2 = std::log2(cl[1] / cl[2]);
    std::printf("%-16s closure order 16->32 %.3f, 32->64 %.3f, mean %.3f\n",
                name.c_str(), o1, o2, 0.5 * (o1 + o2));
    if (wallImage) {
        // Mean over two doublings: see gpu_tally_archimedes checkLevels.
        check(e[0] <= 0.01, name + ": |F - (0, -rho U Gamma)| <= 1% at R/h 16 "
                                   "(measured " + num(e[0]) + ")");
        check(0.5 * (o1 + o2) >= 1.6,
              name + ": wall closure mean order >= 1.6 over R/h 16 -> 64 (measured " +
                  num(0.5 * (o1 + o2)) + ")");
    } else {
        check(e[0] <= 0.03, name + ": |F - (0, -rho U Gamma)| <= 3% at R/h 16 "
                                   "(measured " + num(e[0]) + ")");
        check(o1 >= 1.0, name + ": observed order >= 1 over R/h 16 -> 32 (measured " +
                             num(o1) + ")");
    }
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const auto t0 = std::chrono::steady_clock::now();
    std::printf("U %.1f, M %.2f, pInf %.4f, Gamma %.6f, rho U Gamma %.6f\n", U,
                kMach, pInf, Gam0, kRho * U * Gam0);
    path("ghost Cartesian",
         [](int lvl, double gm) {
             return ghostCartesian(lvl, restKJ(gm, pInf), pInf, {0, 0});
         },
         true);
    path("ghost focal",
         [](int lvl, double gm) {
             return ghostFocal(lvl, restKJ(gm, pInf), pInf, {0, 0});
         },
         true);
    path("O-grid",
         [](int lvl, double gm) {
             return ogrid(lvl, restKJ(gm, pInf), pInf, false, {0, 0});
         },
         true);
    path("cut-cell", [](int lvl, double gm) { return cutCell(lvl, restKJ(gm, 0.0)); },
         false);
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("runtime %.2f s\n", sec);
    return dnstest::finish();
}
