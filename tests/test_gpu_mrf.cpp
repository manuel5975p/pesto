// The rotating-frame zone of --rotor-frame mrf, isolated at the solver level.
//
// Two rigs, each pinning one of the two claims the formulation makes:
//
//   1. TRANSPARENCY. A uniform axial freestream through the zone is an exact
//      solution of the rotating-frame equations (the Coriolis term is zero on
//      a swirl-free field, and the blended transport field f * omega x r is
//      solenoidal BY CONSTRUCTION -- grad f is perpendicular to w). What
//      remains is the O(h^2) discrete divergence of the blend band, so the
//      measured deviation must be small and must NOT grow secularly.
//
//   2. FRAME PARITY. A spinning cylinder in a quiescent box is axisymmetric,
//      and for axisymmetric outer conditions the frozen-rotor approximation
//      is EXACT: the MRF run and the direct (inertial, moving-wall) run solve
//      the same physics through different equations. Torque and the driven
//      swirl must agree to truncation level. A crossflow would NOT qualify --
//      MRF co-rotates the far field, which is precisely its approximation --
//      so the quiescent rig is the one that isolates the machinery.
#include "dns/comp.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/remap.hpp"
#include "dns/rotor.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

AxisRemap3 identityRemap(const Grid& g)
{
    return {Focal1D::identity(g.nx * g.hx), Focal1D::identity(g.ny * g.hy),
            Focal1D::identity(g.nz * g.hz)};
}

// --- 1. uniform axial freestream through the zone ---------------------------

void testTransparency()
{
    const int n = 64;
    const auto g = Grid::make(n, n, n, 2 * pi, 2 * pi, 2 * pi);
    const double gamma_ = 1.4, rhoInf = 1.0, uInf = 1.0, mach = 0.2;
    const double pInf = rhoInf * uInf * uInf / (gamma_ * mach * mach);

    auto dev = gpu::DeviceComp::make(g, gamma_, 0.0);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !red) { check(false, "transparency setup"); return; }

    gpu::MrfZone z;
    z.axis = 0;
    z.omega = 2.0; // tip speed 2 at r1 (Mach 0.4 against c = 5)
    z.center[0] = pi; z.center[1] = pi; z.center[2] = pi;
    z.r1 = 1.0; z.r2 = 1.6;
    z.a1 = 1.0; z.a2 = 1.6;
    check(dev->setMrf(z, identityRemap(g)) == hipSuccess, "setMrf");
    check(dev->mrfActive(), "mrfActive after setMrf");

    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.mx[c] = rhoInf * uInf;
        host.E[c] = pInf / (gamma_ - 1.0) + 0.5 * rhoInf * uInf * uInf;
    }
    (void)ds->upload(host);

    auto deviation = [&] {
        (void)ds->download(host);
        double dv = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c) {
            const double u = host.mx[c] / host.rho[c];
            const double v = host.my[c] / host.rho[c];
            const double w = host.mz[c] / host.rho[c];
            dv = std::max({dv, std::abs(u - uInf), std::abs(v), std::abs(w),
                           std::abs(double(host.rho[c]) - rhoInf)});
        }
        return dv;
    };

    double t = 0.0, devMid = 0.0;
    int steps = 0;
    while (t < 2.0) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
        t += dt;
        ++steps;
        if (devMid == 0.0 && t >= 1.0) devMid = deviation();
    }
    const double devEnd = deviation();
    std::printf("transparency: %d steps to t=2, max deviation %.3e (t=1: "
                "%.3e)\n", steps, devEnd, devMid);
    // Band: measured 2026-07-31 on gfx1201, 64^3: 6.4e-6 at t=1, 1.1e-5 at
    // t=2 (FP32 roundoff of the corner-potential telescoping). The pointwise
    // blend this replaced measured 3.2e-2 on the same rig -- a formulation
    // regression lands 3 orders above this band.
    check(devEnd < 1e-4, "uniform axial flow passes the zone");
    check(devEnd < 2.0 * std::max(devMid, 1e-6),
          "the blend residual does not grow secularly");
}

// --- 2. spinning cylinder in a quiescent box: MRF vs direct -----------------

struct SpinResult
{
    double torque = 0, uTheta = 0;
};

// Spin-up of the fluid around a cylinder rotating about its own axis; `mrf`
// picks the formulation. Wall velocity is omega x r in BOTH -- absolute
// velocity is what the state stores either way.
SpinResult spinUp(bool mrf, double tEnd)
{
    const int n = 192;
    const double L = 2 * pi, R = L / 16, cx = L / 2, cy = L / 2;
    const double gamma_ = 1.4, rhoInf = 1.0;
    const double pInf = 1.0 / (gamma_ * 0.04);
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const double mu = rhoInf * 2 * R / 20.0;

    const Rotor rot{.axis = RotorAxis::Z,
                    .shaft = {cx, cy, 0.5},
                    .omega = 1.0,
                    .radius = R,
                    .bins = 1,
                    .thicken = 0.0};
    const auto sdf = [&](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - R;
    };
    const gpu::DeviceCompGhost::WallVelocity uw = [&rot](double x, double y,
                                                         double z2) {
        return rotorWallVelocity(rot, {x, y, z2});
    };
    auto dev = gpu::DeviceComp::make(g, gamma_, mu);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, gamma_, rhoInf, pInf, nullptr,
                                         {}, &uw);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) { check(false, "spinUp setup"); return {}; }
    if (mrf) {
        gpu::MrfZone z;
        z.axis = 2;
        z.omega = rot.omega;
        z.center[0] = cx; z.center[1] = cy; z.center[2] = 0.5;
        z.r1 = 2.0 * R; z.r2 = 3.5 * R;
        z.a1 = 1e30; z.a2 = 2e30; // z shaft at nz == 1: z-invariant zone
        if (dev->setMrf(z, identityRemap(g)) != hipSuccess) {
            check(false, "spinUp setMrf");
            return {};
        }
    }
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });

    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gamma_ - 1.0);
    }
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

    SpinResult out{.torque = F.qz};
    int m = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
            const double d = sdf(x, y, 0.0);
            if (d < g.hx || d > 2.0 * g.hx) continue;
            const double r = std::hypot(x - cx, y - cy);
            const double tx = -(y - cy) / r, ty = (x - cx) / r;
            out.uTheta += (host.mx[c] / host.rho[c]) * tx +
                          (host.my[c] / host.rho[c]) * ty;
            ++m;
        }
    if (m > 0) out.uTheta /= m;
    return out;
}

void testFrameParity()
{
    const double tEnd = 3.0;
    const auto direct = spinUp(false, tEnd);
    const auto mrf = spinUp(true, tEnd);
    std::printf("parity: direct torque %.6e uTheta %.6e | mrf torque %.6e "
                "uTheta %.6e\n",
                direct.torque, direct.uTheta, mrf.torque, mrf.uTheta);
    check(direct.torque < 0.0 && mrf.torque < 0.0,
          "both frames report a torque opposing the spin");
    // Axisymmetric outer conditions: frozen rotor is exact, so the two runs
    // differ only at truncation level. Measured 2026-07-31 on gfx1201:
    // torque -8.9619e-2 direct vs -8.9628e-2 mrf (1.0e-4 relative), uTheta
    // gap 3.7e-5 relative. Banded ~200x.
    check(std::abs(mrf.torque - direct.torque) <
              0.02 * std::abs(direct.torque),
          "torque agrees between the frames (axisymmetric rig)");
    check(std::abs(mrf.uTheta - direct.uTheta) <
              0.04 * std::abs(direct.uTheta),
          "driven swirl agrees between the frames");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    testTransparency();
    testFrameParity();
    return dnstest::finish();
}
