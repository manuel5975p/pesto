// The rotating-frame zone (--rotor-frame mrf) keeps the solver conservative.
//
// In absolute-velocity form the frame enters as a transport correction inside
// the face flux (shared by both cells of a face) and a Coriolis source
// -f omega x m on the MOMENTUM only. Mass and total energy therefore have no
// source at all: the Coriolis force does no work on the absolute velocity it
// is built from ((omega x m) . m = 0). So on a fully periodic box with no body
// the volume totals of rho and E must stay put to FP32 roundoff while the
// momentum is genuinely turned -- on the uniform mesh (identity remap, the
// separable corner-potential tables) and on an active focal mesh (the same
// tables through the remap, curvilinear flux).
//
// Measured 2026-10-24 (gfx1201): the 1e-6 band FAILS on both meshes at
// 1.15e-6 over 50 steps, and the INERTIAL solver drifts by the same amount --
// the zone itself adds nothing (pinned below). The drift is linear in the
// step count (2.29e-6 at 100 steps): DeviceComp::step's stage-3 weights
// Real(1/3) + Real(2/3) sum to 1 + 2.98e-8 in FP32, a per-step gain on every
// conserved total. Left failing on purpose: it is a solver finding.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/remap.hpp"

#include "curv_test_util.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

constexpr double kGam = 1.4;

// Volume totals (rho, mx, my, mz, E): the conserved quantities.
std::array<double, 5> totals(const std::vector<double>& vol, const CompState& s)
{
    std::array<double, 5> t{0, 0, 0, 0, 0};
    for (std::size_t c = 0; c < s.rho.size(); ++c) {
        t[0] += vol[c] * s.rho[c];
        t[1] += vol[c] * s.mx[c];
        t[2] += vol[c] * s.my[c];
        t[3] += vol[c] * s.mz[c];
        t[4] += vol[c] * s.E[c];
    }
    return t;
}

// 50 steps of smoothState with the zone on (and once with it off, to show the
// zone actually acts). Pre: R describes the mesh `dev` was built on.
void run(const std::string& tag, const Grid& g, const AxisRemap3& R,
         bool curvilinear)
{
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto m = CurvMetrics::build(mesh);
    if (!m) { check(false, tag + ": metrics " + m.error()); return; }
    const double mu = 2e-3;
    auto make = [&] {
        return curvilinear ? gpu::DeviceComp::make(*m, kGam, mu)
                           : gpu::DeviceComp::make(g, kGam, mu);
    };
    auto dev = make();
    auto ref = make();
    auto ds = gpu::DeviceCompState::make(g.size());
    auto dr = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ref || !ds || !dr || !red) { check(false, tag + ": alloc"); return; }

    const double L = g.nx * g.hx;
    gpu::MrfZone z;
    z.axis = 2;
    z.omega = 0.6; // tip speed 0.6 * r2 = 0.66 against c ~ 1.2
    z.center[0] = 0.5 * L; z.center[1] = 0.5 * L; z.center[2] = 0.5 * L;
    z.r1 = 0.7; z.r2 = 1.1;
    z.a1 = 0.7; z.a2 = 1.1;
    check(dev->setMrf(z, R) == hipSuccess, tag + ": setMrf");

    // The reference totals are those of the state AS STORED (FP32), not of
    // the double IC: the upload rounding alone is ~1e-6 of the total.
    auto s0 = curvtest::smoothState(mesh, kGam);
    (void)ds->upload(s0);
    (void)dr->upload(s0);
    (void)ds->download(s0);
    const double dt = 0.5 * dev->stableDt(*ds, *red);
    for (int it = 0; it < 50; ++it) {
        dev->step(*ds, gpu::Real(dt));
        ref->step(*dr, gpu::Real(dt));
    }
    auto s1 = CompState::zeros(g), sr = CompState::zeros(g);
    (void)ds->download(s1);
    (void)dr->download(sr);

    const auto t0 = totals(m->vol, s0), t1 = totals(m->vol, s1),
               tr = totals(m->vol, sr);
    const double dRho = std::abs(t1[0] - t0[0]) / std::abs(t0[0]);
    const double dE = std::abs(t1[4] - t0[4]) / std::abs(t0[4]);
    // Momentum the frame turned: the MRF run against the inertial one.
    double dm = 0.0, mscale = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c) {
        dm = std::max({dm, std::abs(s1.mx[c] - sr.mx[c]),
                       std::abs(s1.my[c] - sr.my[c])});
        mscale = std::max(mscale, std::abs(s0.mx[c]));
    }
    std::printf("%s: dt %.3e, rel drift rho %.3e E %.3e | inertial-run drift "
                "rho %.3e | max|m_mrf - m_inertial| %.3e (scale %.3e)\n",
                tag.c_str(), dt, dRho, dE,
                std::abs(tr[0] - t0[0]) / std::abs(t0[0]), dm, mscale);
    char msg[160];
    std::snprintf(msg, sizeof msg,
                  ": total mass conserved to 1e-6 over 50 steps (measured %.3e)",
                  dRho);
    check(dRho < 1e-6, tag + msg);
    std::snprintf(msg, sizeof msg,
                  ": total energy conserved to 1e-6 over 50 steps (measured "
                  "%.3e)", dE);
    check(dE < 1e-6, tag + msg);
    check(dm > 1e-3 * mscale, tag + ": the zone actually changes the momentum");
    // The zone adds no source of its own: whatever the inertial solver drifts,
    // the MRF run drifts by the same amount.
    const double dRhoI = std::abs(tr[0] - t0[0]) / std::abs(t0[0]);
    check(std::abs(dRho - dRhoI) < 1e-8,
          tag + ": the zone adds no mass drift beyond the inertial solver's");
    // Diagnostic for the absolute drift: if it is linear in the step count it
    // is a per-step bias, not roundoff noise. The FP32 SSP-RK3 stage-3 weights
    // Real(1/3) + Real(2/3) sum to 1 + 2.98e-8, which scales the state by that
    // factor every step wherever the product does not round back.
    for (int it = 0; it < 50; ++it) ref->step(*dr, gpu::Real(dt));
    (void)dr->download(sr);
    const auto t100 = totals(m->vol, sr);
    std::printf("%s: inertial rel drift rho after 50 steps %.3e, after 100 "
                "steps %.3e (Real(1/3)+Real(2/3)-1 = %.3e)\n",
                tag.c_str(), dRhoI, std::abs(t100[0] - t0[0]) / std::abs(t0[0]),
                double(gpu::Real(1.0 / 3.0)) + double(gpu::Real(2.0 / 3.0)) - 1.0);
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    const int n = 32;
    const double L = 2 * pi;
    const auto g = Grid::make(n, n, n, L, L, L);
    run("identity remap (uniform)", g,
        AxisRemap3{Focal1D::identity(L), Focal1D::identity(L),
                   Focal1D::identity(L)},
        false);
    // Fine plateau over the zone on every axis, ratio 2, physical blend 0.6.
    const double lo = 0.5 * L - 1.2, hi = 0.5 * L + 1.2;
    const Focal1D f = Focal1D::fromPhysical(L, lo, hi, 0.6, 2.0);
    check(f.active && f.clears(2 * g.hx), "focal plateau is active and clears");
    run("focal remap (ratio 2)", g, AxisRemap3{f, f, f}, true);
    return dnstest::finish();
}
