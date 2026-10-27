// Impermeable fluid/ghost interface faces (DeviceCompGhost::applyRhs).
//
// The KG face flux averages the fluid cell with the ghost's linear image. That
// does not zero the normal velocity at the face (u_n goes as s^2 at a no-slip
// wall, and a staircase face sits up to h/2 off the true wall), so before the
// impermeable faces a resting body was a net mass source of 0.1-1.1% rho U c
// (exp_mrfB). With them the fluid's mass changes only by FP32 rounding.
//
// Rig: a fully periodic box (no far field, nothing else crosses a boundary)
// with a t/c 0.35 ellipse at 7 deg incidence, a sub-cell offset off the
// lattice, in a uniform stream. The fluid mass (FP64 sum over non-solid cells,
// times the cell volume the device divides by) after 50 steps:
//   (a) WITHOUT the rhs hook (the old permeable faces) it drifts -- the check
//       is not vacuous;
//   (b) WITH applyRhs it holds to FP32 summation, far below (a);
//   (c) the tally's convective part is exactly 0 (surfaceFaces fc).
// Variants: Cartesian, focal (curvilinear: the face area vector), and an MRF
// zone spinning the ellipse in a crossflow (no flux relative to the frame).
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/remap.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

using namespace dns;
using std::numbers::pi;

namespace {

using dnstest::check;

constexpr double gamma_ = 1.4, rhoInf = 1.0, mu = 2e-3, mach = 0.26;
const double pInf = 1.0 / (gamma_ * mach * mach);

struct Rig
{
    std::string name;
    bool focal = false;
    bool mrf = false;
};

struct Drift
{
    double rel = 0;   // |M(50) - M(0)| / M(0)
    double fcMax = 0; // max |fc| over the tallied faces at the end
    int faces = 0;
};

// The ellipse rig after 50 steps, with or without the wall faces' rhs.
Drift run(const Rig& rig, bool hook)
{
    Grid g;
    g.nx = 128;
    g.ny = 96;
    g.nz = 1;
    g.hx = g.hy = g.hz = 1.0 / 96;
    const double Lx = g.nx * g.hx, Ly = g.ny * g.hy;
    const double cx = 0.45 * Lx + 0.37 * g.hx, cy = 0.5 * Ly + 0.29 * g.hy;
    const double c = 0.3, t = 0.35, aoa = 7.0 * pi / 180.0;
    const double lo[3] = {cx - 0.3, cy - 0.2, 0.0},
                 hi[3] = {cx + 0.3, cy + 0.2, 0.0}, w[3] = {0.1, 0.1, 0.0};
    AxisRemap3 R{Focal1D::identity(Lx), Focal1D::identity(Ly),
                 Focal1D::identity(g.nz * g.hz)};
    if (rig.focal) {
        R.fx = Focal1D::fromPhysical(Lx, lo[0], hi[0], w[0], 2.0);
        R.fy = Focal1D::fromPhysical(Ly, lo[1], hi[1], w[1], 2.0);
    }
    // Ellipse sdf: the radial scaling of the unit-circle distance, which is
    // close to a distance near the surface at t/c 0.35.
    const gpu::GhostSdf sdf = [=](double x, double y, double) {
        const double dx = x - cx, dy = y - cy;
        const double xr = std::cos(aoa) * dx - std::sin(aoa) * dy;
        const double yr = std::sin(aoa) * dx + std::cos(aoa) * dy;
        const double a = 0.5 * c, b = 0.5 * t * c;
        const double q = std::hypot(xr / a, yr / b);
        return (q - 1.0) * std::min(a, b);
    };
    std::optional<CurvMetrics> cm;
    std::vector<double> vol(g.size(), g.hx * g.hy * g.hz);
    if (rig.focal) {
        auto m = CurvMetrics::build(CurvMesh::fromRemap(g, R));
        if (!m) { check(false, rig.name + ": metrics"); return {}; }
        cm = std::move(*m);
        // The device divides by the FP32 1/vol it uploaded.
        for (std::size_t q = 0; q < g.size(); ++q)
            vol[q] = 1.0 / double(gpu::Real(1.0 / cm->vol[q]));
    }
    auto dev = rig.focal ? gpu::DeviceComp::make(*cm, gamma_, mu)
                         : gpu::DeviceComp::make(g, gamma_, mu);
    if (!dev) { check(false, rig.name + ": solver"); return {}; }
    const double omega = 4.0 / c; // tip speed ~0.6 U
    const gpu::GhostWallMotion spin = [=](double x, double y, double) {
        return gpu::GhostWallKinematics{
            .u = {-omega * (y - cy), omega * (x - cx), 0.0},
            .a = {-omega * omega * (x - cx), -omega * omega * (y - cy), 0.0}};
    };
    if (rig.mrf) {
        gpu::MrfZone z;
        z.axis = 2;
        z.omega = omega;
        z.center[0] = cx;
        z.center[1] = cy;
        z.r1 = 0.6 * c;
        z.r2 = 1.2 * c;
        z.a1 = 1e30;
        z.a2 = 2e30;
        check(dev->setMrf(z, R) == hipSuccess, rig.name + ": setMrf");
    }
    const auto tb = gpu::classifyCompGhost(g, sdf, nullptr, R,
                                           rig.mrf ? &spin : nullptr);
    gpu::CompGhostCapacity cap;
    cap.cover(tb);
    auto gh = gpu::DeviceCompGhost::fromTables(g, tb, cap, gamma_, rhoInf,
                                               pInf, rig.focal);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!gh || !ds || !red) { check(false, rig.name + ": setup"); return {}; }
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
    if (hook)
        dev->setRhsHook([&](const gpu::DeviceCompState& s,
                            gpu::DeviceCompState& out,
                            const gpu::StageCombine& cmb) {
            gh->applyRhs(*dev, s, out, cmb);
        });
    auto h = CompState::zeros(g);
    for (std::size_t q = 0; q < g.size(); ++q) {
        h.rho[q] = rhoInf;
        h.mx[q] = rhoInf;
        h.my[q] = 0.1 * rhoInf;
        h.E[q] = pInf / (gamma_ - 1.0) + 0.5 * rhoInf * 1.01;
    }
    (void)ds->upload(h);
    gh->apply(*ds);
    std::vector<char> solid(g.size(), 0);
    for (int q : gh->solidCells()) solid[std::size_t(q)] = 1;
    auto mass = [&] {
        (void)ds->download(h);
        double m = 0.0;
        for (std::size_t q = 0; q < g.size(); ++q)
            if (!solid[q]) m += h.rho[q] * vol[q];
        return m;
    };
    const double m0 = mass();
    for (int s = 0; s < 50; ++s)
        dev->step(*ds, gpu::Real(dev->stableDt(*ds, *red)));
    gh->apply(*ds);
    Drift out;
    out.rel = std::abs(mass() / m0 - 1.0);
    out.faces = tb.nFace();
    const auto rows = gh->surfaceFaces(*dev, *ds, pInf);
    check(rows.has_value() && !rows->empty(), rig.name + ": surfaceFaces");
    if (rows)
        for (const auto& r : *rows)
            for (int a = 0; a < 3; ++a)
                out.fcMax = std::max(out.fcMax, std::abs(r.fc[a]));
    return out;
}

void rig(const Rig& r)
{
    const Drift off = run(r, false), on = run(r, true);
    std::printf("%s: %d faces; fluid mass drift over 50 steps: permeable "
                "%.3e, impermeable %.3e (ratio %.0f); max|fc| %.1e\n",
                r.name.c_str(), on.faces, off.rel, on.rel,
                off.rel / std::max(on.rel, 1e-300), on.fcMax);
    check(off.rel > 1e-6, r.name + ": the permeable faces leak (not vacuous)");
    // FP32 stage rounding over 150 stages: measured 3e-8 (focal), 5e-8
    // (Cartesian), 1.5e-7 (MRF: larger face velocities) against 1e-5..4e-4
    // permeable.
    check(on.rel < 3e-7, r.name + ": impermeable faces hold the fluid mass "
                                  "to FP32 rounding (3e-7)");
    check(on.rel < 1e-2 * off.rel,
          r.name + ": the leak is gone (100x below the permeable drift)");
    check(on.fcMax == 0.0,
          r.name + ": the tally's convective part is exactly 0");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    rig({.name = "cartesian ellipse"});
    rig({.name = "focal ellipse", .focal = true});
    rig({.name = "MRF spinning ellipse", .mrf = true});
    return dnstest::finish();
}
