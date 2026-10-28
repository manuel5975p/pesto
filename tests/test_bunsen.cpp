// The slot Bunsen geometry (dns/bunsen.hpp, docs/chem.md), host only: the
// inflow profile's limits and symmetry, the cone distance, and measureCone on
// synthetic ideal cones -- the angle the runner reports must be the angle
// that was drawn, on the run's own grids, before any GPU number is trusted.
#include "dns/bunsen.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <format>
#include <vector>

namespace {

using dnstest::check;
using namespace dns;

BunsenGeom geom(double W, double U, double yc)
{
    BunsenGeom b;
    b.W = W;
    b.uJet = U;
    b.uCo = 1;
    b.wCo = W;
    b.tCo = 5;
    b.edge = 0.4;
    b.yc = yc;
    b.coneH = 0.5 * W * std::sqrt(U * U - 1);
    return b;
}

// A tanh flame of thickness d on the ideal cone of b, sampled on nx x ny
// cells of size h: Tu = 1 inside, Tb = 5 outside.
std::vector<float> coneField(const BunsenGeom& b, int nx, int ny, double h,
                             double d)
{
    std::vector<float> T(std::size_t(nx) * ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double s =
                bunsenConeDistance(b, (i + 0.5) * h, (j + 0.5) * h);
            T[std::size_t(j) * nx + i] =
                float(1 + 4 * 0.5 * (1 + std::tanh(s / d)));
        }
    return T;
}

} // namespace

int main()
{
    {
        BunsenGeom b = geom(19.2, 5, 28.8);
        const auto c = bunsenInflow(b, b.yc);
        check(std::abs(c.u - 5) < 1e-12 && std::abs(c.Y - 1) < 1e-12 &&
                  std::abs(c.t - 1) < 1e-12,
              "slot centre: u = U_jet, Y = 1, T = Tu");
        const auto f = bunsenInflow(b, b.yc + 0.5 * b.W + 0.5 * b.wCo);
        check(std::abs(f.u - 1) < 1e-9 && f.Y < 1e-9 && std::abs(f.t - 5) < 1e-9,
              "coflow: u = U_co, Y = 0, T = tCo");
        const auto e = bunsenInflow(b, b.yc + 0.5 * b.W);
        check(std::abs(e.Y - 0.5) < 1e-12 && std::abs(e.co - 0.5) < 1e-12,
              "edge: Y = co = 1/2 on the slot edge (no lip)");
        const auto l = bunsenInflow(b, b.yc - 0.5 * b.W);
        check(l.u == e.u && l.Y == e.Y && l.t == e.t, "profile symmetric in y");
        check(std::abs(e.t - (5 - 4 * e.Y)) < 1e-12,
              "burnt pilot: T/Tu = sigma - (sigma - 1) Y (Shvab-Zeldovich)");
        check(e.pilot == 0, "no strip: pilot weight exactly 0");
        b.pilot = 4;
        b.tPilot = 5;
        b.uPilot = 0.5;
        b.tCo = 1;
        const auto m = bunsenInflow(b, b.yc + 0.5 * b.W + 2);
        // 5 edge widths from both edges: tanh(5) leaves 4.5e-5 of each.
        check(std::abs(m.u - 0.5) < 1e-3 && m.Y < 1e-4 &&
                  std::abs(m.t - 5) < 1e-3 && std::abs(m.pilot - 1) < 1e-4,
              "pilot strip: u = uPilot, Y = 0, T = tPilot (to 1e-3)");
        const auto o = bunsenInflow(b, b.yc + 0.5 * b.W + 4 + 0.5 * b.wCo);
        check(std::abs(o.t - 1) < 1e-9 && std::abs(o.u - 1) < 1e-9,
              "cold coflow beyond the strip");
    }
    {
        const BunsenGeom b = geom(10, 3, 20);
        check(std::abs(bunsenConeDistance(b, b.coneH, b.yc)) < 1e-12,
              "distance 0 at the tip");
        check(std::abs(bunsenConeDistance(b, 0, b.yc + 5)) < 1e-12,
              "distance 0 at the base corner");
        check(bunsenConeDistance(b, 1, b.yc) < 0 &&
                  bunsenConeDistance(b, b.coneH + 1, b.yc) > 0 &&
                  bunsenConeDistance(b, 1, b.yc + 6) > 0,
              "sign: fresh core < 0, past the tip and outside > 0");
        const double sa = 1.0 / 3; // sin alpha of U = 3
        check(std::abs(bunsenConeDistance(b, b.coneH / 2, b.yc) +
                       0.5 * b.W / 2 * std::sqrt(1 - sa * sa)) < 1e-12,
              "axis point at half height: distance (W/4) cos alpha");
    }
    // measureCone on the runner's own geometry: h = delta_th / cells with
    // delta_th = 1.60165, W = 12 delta_th, coflow W per side, flame 1 delta_th.
    for (const double cells : {8.0, 16.0})
        for (const double U : {3.0, 5.0}) {
            const double dth = 1.60165, h = dth / cells;
            const int half = int(std::ceil(1.5 * 12 * cells - 1e-9));
            const int ny = 2 * half;
            BunsenGeom b = geom(12 * dth, U, half * h);
            const int nx = int(std::ceil((1.25 * b.coneH / dth + 8) * cells));
            const auto T = coneField(b, nx, ny, h, 0.5 * dth);
            const ConeMeasure m = measureCone(T, nx, ny, h, 3.0, b.yc);
            const double tanA = 0.5 * b.W / b.coneH;
            const double eA = m.tanA / tanA - 1;
            std::printf("  cells %g U %g: tan %.6f (ideal %.6f, %+.2e) L %.6f "
                        "R %.6f tip %.4f (ideal %.4f) base %.4f rows %d rms "
                        "%.2e\n",
                        cells, U, m.tanA, tanA, eA, m.tanL, m.tanR, m.xTip,
                        b.coneH, m.xBase, m.rows, m.rms);
            const std::string tag = std::format(" (cells {:g}, U {:g})", cells, U);
            check(m.ok && std::abs(eA) < 2e-3,
                  "flank tan(alpha) within 0.2% of the drawn cone" + tag);
            check(std::abs(m.tanL - m.tanR) < 1e-6 * tanA,
                  "left and right flanks agree" + tag);
            // The tanh profile's Tmid isotherm IS the cone, except the
            // rounded tip, which the 2D diffusion profile pulls inward.
            check(std::abs(m.xTip - b.coneH) < 2 * dth,
                  "tip within 2 delta_th of the drawn tip" + tag);
            check(m.xBase < h, "base on the first row" + tag);
            // Uniform rho = 1, u = U through the drawn cone: rho u.n =
            // U sin(alpha) = S_L = 1.
            const std::vector<float> one(T.size(), 1.0f), uu(T.size(), float(U)),
                zero(T.size(), 0.0f);
            const double mf = flankMassFlux(m, one, uu, zero, nx, h);
            std::printf("  mass flux %.6f\n", mf);
            check(std::abs(mf - 1) < 1e-3,
                  "flank mass flux of a uniform U jet = U sin(alpha) = 1" + tag);
            // omega qualification: a reaction sheet on the cone changes
            // nothing; the same sheet cut below X0 (a lifted flame whose
            // hot gas still crosses Tmid) moves the base to X0.
            std::vector<float> w(T.size());
            const double X0 = 0.3 * b.coneH;
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const double s = bunsenConeDistance(b, (i + 0.5) * h,
                                                        (j + 0.5) * h);
                    w[std::size_t(j) * nx + i] =
                        float(std::exp(-(s - 0.3 * dth) * (s - 0.3 * dth) /
                                       (0.1 * dth * dth)));
                }
            const int win = int(std::ceil(0.5 * dth / h));
            const ConeMeasure q = measureCone(T, nx, ny, h, 3.0, b.yc, w, win);
            check(q.ok && q.tanA == m.tanA && q.xBase == m.xBase,
                  "burning sheet on the cone: the same cone" + tag);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; (i + 0.5) * h < X0; ++i)
                    w[std::size_t(j) * nx + i] = 0;
            const ConeMeasure l = measureCone(T, nx, ny, h, 3.0, b.yc, w, win);
            check(l.xBase >= X0 && l.xBase < X0 + h,
                  "no reaction below X0: the base is X0" + tag);
        }
    {
        const BunsenGeom b = geom(19.2, 5, 28.8);
        const int nx = 50, ny = 72;
        std::vector<float> hot(std::size_t(nx) * ny, 5.0f), cold(hot.size(), 1.0f);
        const ConeMeasure a = measureCone(hot, nx, ny, 0.8, 3.0, b.yc);
        check(a.axisHot && !a.ok && std::isnan(a.xTip),
              "all hot: axisHot (flash-back), no cone");
        const ConeMeasure c = measureCone(cold, nx, ny, 0.8, 3.0, b.yc);
        check(!c.axisHot && !c.ok && std::isnan(c.xTip),
              "all cold: no tip (blow-through), no cone");
    }
    return dnstest::finish();
}
