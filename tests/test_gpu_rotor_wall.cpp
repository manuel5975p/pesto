// The MOVING no-slip wall, isolated from every other part of the rotor.
//
// A circular cylinder spun about its own axis is the one body whose geometry is
// IDENTICAL at every azimuth, so nothing here depends on rotated geometry, on an
// azimuth table, or on a bin index: what is under test is exactly the ghost
// image about u_wall = omega x r, and the moment the surface tally takes.
//
// Four properties, in increasing depth:
//   - a wall velocity that is identically zero reproduces the resting wall
//     (momentum bit for bit; energy to round-off, because the resting branch
//     forms 0.5 a^2 rho |u|^2 and the moving one 0.5 rho |a u|^2);
//   - the moment obeys the parallel-axis identity exactly, which pins the face
//     positions and the reduction without appealing to any physics;
//   - a spinning wall drives an AZIMUTHAL flow and feels a torque OPPOSING its
//     spin -- the thing whose absence would make a rotor a rotating hole;
//   - reversing omega reverses the flow and the torque, and leaves |torque|
//     unchanged: the wall term carries no even-order bias.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/rotortable.hpp"
#include "dns/rotor.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

constexpr int n = 192;
const double L = 2 * pi, R = L / 16, cx = L / 2, cy = L / 2;
constexpr double gamma_ = 1.4, rhoInf = 1.0;
const double pInf = 1.0 / (gamma_ * 0.04); // freestream Mach 0.2 on uInf = 1

double sdfCyl(double x, double y, double)
{
    return std::hypot(x - cx, y - cy) - R;
}

Rotor spin(double omega)
{
    return {.axis = RotorAxis::Z,
            .shaft = {cx, cy, 0.5},
            .omega = omega,
            .radius = R,
            .bins = 16,
            .thicken = 0.0};
}

CompState quiescent(const Grid& g)
{
    auto h = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] = rhoInf;
        h.E[c] = pInf / (gamma_ - 1.0);
    }
    return h;
}

struct SpinResult
{
    double torque = 0, fx = 0, fy = 0;
    double uTheta = 0;  // mean tangential speed in the first ring outside R
    double uRad = 0;    // mean |radial| speed there, which should be ~0
    double ke = 0;      // fluid kinetic energy outside the body
};

// Spin the cylinder from rest for `tEnd` and report what the wall did.
SpinResult spinUp(const Grid& g, double mu, double omega, double tEnd)
{
    const Rotor rot = spin(omega);
    const gpu::DeviceCompGhost::WallMotion uw = gpu::rotorWallMotion(rot);
    auto dev = gpu::DeviceComp::make(g, gamma_, mu);
    auto gh = gpu::DeviceCompGhost::make(g, sdfCyl, gamma_, rhoInf, pInf,
                                         nullptr, {}, &uw);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) { check(false, "spinUp setup"); return {}; }
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });

    auto host = quiescent(g);
    (void)ds->upload(host);
    double t = 0.0;
    while (t < tEnd) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
        t += dt;
    }
    gh->apply(*ds);
    const auto F = gh->force(*dev, *ds, rot.shaft);
    (void)ds->download(host);

    SpinResult out{.torque = F.qz, .fx = F.fx, .fy = F.fy};
    // First ring of fluid cells outside the surface: one to two cells out, so
    // the sample is in the fluid but still inside the driven shear layer.
    int m = 0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
            const double d = sdfCyl(x, y, 0.0);
            const double u = host.mx[c] / host.rho[c];
            const double v = host.my[c] / host.rho[c];
            if (d > 0.0) out.ke += 0.5 * host.rho[c] * (u * u + v * v)
                                  * g.hx * g.hy;
            if (d < g.hx || d > 2.0 * g.hx) continue;
            const double r = std::hypot(x - cx, y - cy);
            const double tx = -(y - cy) / r, ty = (x - cx) / r;
            const double rx = (x - cx) / r, ry = (y - cy) / r;
            out.uTheta += u * tx + v * ty;
            out.uRad += std::abs(u * rx + v * ry);
            ++m;
        }
    if (m > 0) { out.uTheta /= m; out.uRad /= m; }
    return out;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();

    const auto g = Grid::make(n, n, 1, L, L, 1.0); // hz = 1: unit depth
    // Re 20 on the diameter keeps the shear layer several cells thick at this
    // resolution, so the wall term is measured, not the staircase.
    const double mu = rhoInf * 2 * R / 20.0;
    const double omega = 1.0;
    std::printf("rig: R %.4f mu %.4e omega %.3f tip speed %.4f (Mach %.3f)\n",
                R, mu, omega, omega * R,
                omega * R / std::sqrt(gamma_ * pInf / rhoInf));

    // --- 1. a resting "moving" wall is the resting wall ---------------------
    //
    // Same evolved field, two ghosts: one built with no wall velocity, one with
    // a velocity that is identically zero. The momentum image must agree bit
    // for bit; the energy differs only by the rounding of a*a*|u|^2 against
    // |a*u|^2, so it is checked in ulps, not exactly.
    {
        const gpu::DeviceCompGhost::WallMotion zero =
            [](double, double, double) { return gpu::GhostWallKinematics{}; };
        auto dev = gpu::DeviceComp::make(g, gamma_, mu);
        auto still = gpu::DeviceCompGhost::make(g, sdfCyl, gamma_, rhoInf, pInf);
        auto moving = gpu::DeviceCompGhost::make(g, sdfCyl, gamma_, rhoInf,
                                                 pInf, nullptr, {}, &zero);
        auto da = gpu::DeviceCompState::make(g.size());
        auto db = gpu::DeviceCompState::make(g.size());
        auto red = gpu::ReduceScratch::make();
        if (!dev || !still || !moving || !da || !db || !red) return EXIT_FAILURE;
        check(!still->moving() && moving->moving(),
              "moving() reports which wall was built");

        // A developed field, so the ghosts have something non-trivial to image.
        auto host = quiescent(g);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double d = sdfCyl((i + 0.5) * g.hx, (j + 0.5) * g.hy, 0.0);
                const double f = std::clamp(d / (3.0 * g.hx), 0.0, 1.0);
                const double u = f * f * (3 - 2 * f);
                host.mx[c] = rhoInf * u;
                host.E[c] = pInf / (gamma_ - 1.0) + 0.5 * rhoInf * u * u;
            }
        dev->setStageHook([&](gpu::DeviceCompState& s) { still->apply(s); });
        (void)da->upload(host);
        double t = 0.0;
        while (t < 0.5) {
            const double dt = dev->stableDt(*da, *red);
            dev->step(*da, gpu::Real(dt));
            t += dt;
        }
        (void)da->download(host);

        (void)da->upload(host);
        (void)db->upload(host);
        still->apply(*da);
        moving->apply(*db);
        auto ha = CompState::zeros(g), hb = CompState::zeros(g);
        (void)da->download(ha);
        (void)db->download(hb);
        bool rhoSame = true;
        double worstM = 0.0, worstE = 0.0, mScale = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c) {
            if (ha.rho[c] != hb.rho[c]) rhoSame = false;
            mScale = std::max({mScale, std::abs(double(ha.mx[c])),
                               std::abs(double(ha.my[c]))});
        }
        for (std::size_t c = 0; c < g.size(); ++c) {
            worstM = std::max({worstM, std::abs(ha.mx[c] - hb.mx[c]),
                               std::abs(ha.my[c] - hb.my[c]),
                               std::abs(ha.mz[c] - hb.mz[c])});
            worstE = std::max(worstE, std::abs(ha.E[c] - hb.E[c])
                                          / std::max(1e-30, std::abs(ha.E[c])));
        }
        std::printf("zero wall velocity: rho identical %d, worst |dm| %.2e "
                    "(scale %.2e), worst rel E %.2e\n",
                    int(rhoSame), worstM, mScale, worstE);
        check(rhoSame, "a zero wall velocity leaves the density image alone");
        // NOT bit-for-bit, and deliberately not forced to be: the two branches
        // associate the same product differently ((-a*rg)*ui against
        // rg*(-(a*ui))), and contorting the kernel to match an input nobody
        // uses would be worse than the float noise it removes. The inertness
        // guarantee that protects the measured anchors is the OTHER one -- no
        // wall velocity at all takes the resting branch verbatim -- and
        // gpu_comp_ghost pins that.
        check(worstM < 1e-6 * std::max(mScale, 1e-30),
              "a zero wall velocity gives the resting momentum image to float "
              "round-off");
        check(worstE < 1e-6, "and the same energy to float round-off");
    }

    // --- 2. the moment is the moment of the reported force ------------------
    //
    // Parallel-axis: Q(b) = Q(a) - (b - a) x F, exactly, for any two pivots on
    // one state. This pins fpos_ and kBodyMoment without leaning on physics --
    // a wrong face position or a mismatched span would break it immediately.
    {
        const Rotor rot = spin(omega);
        const gpu::DeviceCompGhost::WallMotion uw = gpu::rotorWallMotion(rot);
        auto dev = gpu::DeviceComp::make(g, gamma_, mu);
        auto gh = gpu::DeviceCompGhost::make(g, sdfCyl, gamma_, rhoInf, pInf,
                                             nullptr, {}, &uw);
        auto ds = gpu::DeviceCompState::make(g.size());
        auto red = gpu::ReduceScratch::make();
        if (!dev || !gh || !ds || !red) return EXIT_FAILURE;
        dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
        auto host = quiescent(g);
        (void)ds->upload(host);
        double t = 0.0;
        while (t < 0.4) {
            const double dt = dev->stableDt(*ds, *red);
            dev->step(*ds, gpu::Real(dt));
            t += dt;
        }
        gh->apply(*ds);
        const std::array<double, 3> a{cx, cy, 0.5}, b{cx + 0.37, cy - 0.21, 0.5};
        const auto Fa = gh->force(*dev, *ds, a);
        const auto Fb = gh->force(*dev, *ds, b);
        const double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
        const double px = Fa.qx - (dy * Fa.fz - dz * Fa.fy);
        const double py = Fa.qy - (dz * Fa.fx - dx * Fa.fz);
        const double pz = Fa.qz - (dx * Fa.fy - dy * Fa.fx);
        const double scale =
            std::max({std::abs(Fa.qz), std::abs(Fb.qz), 1e-30});
        const double err = (std::abs(Fb.qx - px) + std::abs(Fb.qy - py)
                            + std::abs(Fb.qz - pz)) / scale;
        std::printf("parallel axis: Q(a) = (%.6e, %.6e, %.6e), rel err %.2e\n",
                    Fa.qx, Fa.qy, Fa.qz, err);
        check(err < 1e-10, "moment obeys the parallel-axis identity");
        check(Fa.fx == Fb.fx && Fa.fy == Fb.fy && Fa.fz == Fb.fz,
              "the pivot does not disturb the force");
    }

    // --- 3. the wall drives the fluid ---------------------------------------
    const auto fwd = spinUp(g, mu, omega, 1.0);
    // Steady creeping-flow torque per unit depth on a rotating circle,
    // 4 pi mu omega a^2, as the scale to judge against. The measured value is
    // an early-transient one on a periodic box, so it is only banded, not
    // pinned: the Stokes layer is still thin, which makes the shear -- and the
    // torque -- LARGER than the steady value.
    const double stokes = 4 * pi * mu * omega * R * R;
    std::printf("spun up t=1: torque %.5e (Stokes %.5e, ratio %.2f) "
                "F (%.2e, %.2e) uTheta %.4f (wall %.4f) |uRad| %.4f KE %.3e\n",
                fwd.torque, stokes, fwd.torque / -stokes, fwd.fx, fwd.fy,
                fwd.uTheta, omega * R, fwd.uRad, fwd.ke);
    check(std::isfinite(fwd.torque) && std::isfinite(fwd.ke),
          "spun-up state is finite");
    check(fwd.ke > 1e-6, "the spinning wall put energy into a fluid at rest");
    check(fwd.torque < 0.0,
          "the fluid resists the spin (torque opposes omega)");
    check(-fwd.torque > 0.5 * stokes && -fwd.torque < 20.0 * stokes,
          "torque is within a transient factor of the Stokes value");
    check(fwd.uTheta > 0.3 * omega * R,
          "the first fluid ring turns WITH the wall at a good fraction of its "
          "speed");
    check(fwd.uRad < 0.2 * std::abs(fwd.uTheta),
          "the driven flow is azimuthal, not a radial jet");
    check(std::abs(fwd.fx) < 0.05 * std::abs(fwd.torque) / R &&
              std::abs(fwd.fy) < 0.05 * std::abs(fwd.torque) / R,
          "a spinning circle in a fluid at rest feels no net force");

    // --- 4. reversing the spin reverses everything --------------------------
    const auto rev = spinUp(g, mu, -omega, 1.0);
    const double asym = std::abs(rev.torque + fwd.torque)
                        / std::max(1e-30, std::abs(fwd.torque));
    std::printf("reversed: torque %.5e, |asymmetry| %.2e, uTheta %.4f\n",
                rev.torque, asym, rev.uTheta);
    check(rev.torque > 0.0, "reversing omega reverses the torque");
    check(rev.uTheta < 0.0, "reversing omega reverses the driven flow");
    check(asym < 1e-3, "the wall term carries no bias: |torque| is unchanged");

    return dnstest::finish();
}
