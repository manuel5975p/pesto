// Wall heat tally against steady radial conduction: the exact field
//   T(r) = T1 + (T2 - T1) ln(r/R1) / ln(R2/R1),   gas at rest, uniform p,
// IMPOSED on every fluid cell around an isothermal cylinder held at T1, the
// wall image applied, then tallied -- no time stepping.
//
// Exact: Fourier flux q = -kappa dT/dr e_r. The tally's `heat` is the rate
// fluid -> body (positive = the body gains energy), so per unit depth
//   heat = kappa dT/dr(R1) * 2 pi R1 = -2 pi kappa (T1 - T2) / ln(R2/R1),
// negative here: the wall (T1 = 1.2 Tinf) is the hot side and heats the gas.
//
// Paths: ghost Cartesian, ghost focal (anisotropic plateau), O-grid. The
// isothermal image is linear through the wall face, so the face gradient is
// first order on the staircase: 5% at R/h = 16, observed order >= 0.8.
//
// Measurement mode (`test_gpu_tally_conduction sweep`, not part of the ctest
// verdict): the relative error over a resolution sweep for both ghost paths,
// with the count of Cartesian ghosts whose sample stencil dropped a solid cell.
#include "dns/bodies.hpp"
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>
#include <string>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

constexpr double kGam = 1.4, kMu = 0.05;
const double kPInf = 1.0 / (kGam * 0.04);
const double kTInf = kPInf; // rhoInf = 1, R = 1
constexpr double kR1 = 1.0, kR2 = 2.0;
const double kT1 = 1.2 * kTInf, kT2 = kTInf;

double Tof(double r)
{
    return kT1 + (kT2 - kT1) * std::log(r / kR1) / std::log(kR2 / kR1);
}

double exactHeat()
{
    const double kappa = Transport::constant(kGam, kMu).kappa(kTInf);
    return -2 * pi * kappa * (kT1 - kT2) / std::log(kR2 / kR1);
}

void putCell(CompState& h, std::size_t c, double r)
{
    const double T = Tof(r);
    h.rho[c] = kPInf / T;
    h.mx[c] = h.my[c] = h.mz[c] = 0.0;
    h.E[c] = kPInf / (kGam - 1.0);
}

struct Tally
{
    double heat = 0, rh = 0;
    bool ok = false;
};

Tally ghostTally(int n, bool focal)
{
    using WT = gpu::GhostWallThermal;
    Tally t;
    const double L = focal ? 8.0 : 4.0;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const double cx = L / 2 + 0.013 * g.hx, cy = L / 2 + 0.017 * g.hy;
    AxisRemap3 R{Focal1D::identity(L), Focal1D::identity(L),
                 Focal1D::identity(1.0)};
    if (focal) {
        // Same anisotropic plateau as the Couette test: fine cells 0.71 h by
        // 0.58 h.
        const double lo[3] = {cx - 2 * kR1, cy - 1.25 * kR1, 0};
        const double hi[3] = {cx + 2 * kR1, cy + 1.25 * kR1, 0};
        const double w[3] = {kR1, kR1, 0};
        R = focalRemap(g, true, lo, hi, w, 3.0);
        if (!R.fx.clears(2 * g.hx) || !R.fy.clears(2 * g.hy)) {
            check(false, "focal plateau clears the box faces");
            return t;
        }
    }
    const double hfx = focal ? R.fx.c * R.fx.r * g.hx : g.hx;
    const double hfy = focal ? R.fy.c * R.fy.r * g.hy : g.hy;
    t.rh = kR1 / std::max(hfx, hfy);
    const gpu::GhostSdf sdf = [=](double x, double y, double) {
        return std::hypot(x - cx, y - cy) - kR1;
    };
    std::optional<gpu::DeviceComp> dev;
    if (focal) {
        const auto cpu = Comp::make(CurvMesh::fromRemap(g, R), kGam, kMu);
        if (!cpu) { check(false, "focal CPU make"); return t; }
        auto d = gpu::DeviceComp::make(*cpu->curv, kGam, kMu);
        if (d) dev.emplace(std::move(*d));
    } else {
        auto d = gpu::DeviceComp::make(g, kGam, kMu);
        if (d) dev.emplace(std::move(*d));
    }
    auto gh = gpu::DeviceCompGhost::make(
        g, sdf, kGam, 1.0, kPInf, nullptr, R,
        WT{.mode = WT::Mode::Isothermal, .Twall = kT1, .Rgas = 1.0});
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !gh || !ds) { check(false, "ghost setup"); return t; }
    auto h = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            putCell(h, g.idx(i, j, 0),
                    std::hypot(R.fx.map((i + 0.5) * g.hx) - cx,
                               R.fy.map((j + 0.5) * g.hy) - cy));
    (void)ds->upload(h);
    gh->apply(*ds);
    t.heat = gh->force(*dev, *ds).heat;
    t.ok = true;
    return t;
}

Tally ogridTally(int nx, int ny)
{
    Tally t;
    const auto g = ogridGrid(nx, ny, 1, kR1, kR2, 1.0);
    auto og = buildOGrid(g, sdfOf(CylinderZ{0.0, 0.0, kR1}),
                         {.cx = 0.0, .cy = 0.0, .cz = 0.0, .rFar = kR2,
                          .beta = 0.0, .span = 1.0});
    if (!og) { check(false, "O-grid build: " + og.error()); return t; }
    auto m = CurvMetrics::build(og->mesh);
    if (!m) { check(false, "O-grid metrics: " + m.error()); return t; }
    auto dev = gpu::DeviceComp::make(*m, Transport::constant(kGam, kMu));
    gpu::DeviceOGridBc::Config bc;
    bc.gamma = kGam;
    bc.thermal = gpu::OGridThermal::Isothermal;
    bc.Twall = kT1;
    bc.rhoInf = 1.0;
    bc.uInf = 0.0;
    bc.pInf = kPInf;
    auto wall = gpu::DeviceOGridBc::make(g, bc);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !wall || !ds) { check(false, "O-grid setup"); return t; }
    t.rh = kR1 / std::max(og->h1Max, 2 * pi * kR1 / nx);
    auto h = CompState::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            double x = 0, y = 0;
            for (int d = 0; d < 4; ++d) {
                const auto p = og->mesh.node(i + (d & 1), j + (d >> 1), 0);
                x += 0.25 * p[0];
                y += 0.25 * p[1];
            }
            // The inner ghost row sits inside r < R1 where ln(r/R1) < 0 is
            // still fine (it is overwritten by the wall image anyway).
            putCell(h, g.idx(i, j, 0), std::hypot(x, y));
        }
    (void)ds->upload(h);
    wall->apply(*ds);
    t.heat = wall->surface(*dev, *ds).heat;
    t.ok = true;
    return t;
}

void verdict(const std::string& tag, const Tally& a, const Tally& b, double Q)
{
    const double eC = std::abs(a.heat / Q - 1), eF = std::abs(b.heat / Q - 1);
    const double ord = std::log2(eC / eF);
    std::printf("  %s: R/h %.1f heat %.6e err %.3e | R/h %.1f heat %.6e err "
                "%.3e | order %.2f\n",
                tag.c_str(), a.rh, a.heat, eC, b.rh, b.heat, eF, ord);
    check(a.heat < 0.0 && b.heat < 0.0,
          tag + ": hot wall heats the gas (heat < 0, exact sign)");
    check(eC < 0.05, tag + ": within 5% at R/h ~16 (measured " +
                         std::to_string(100 * eC) + "%)");
    check(ord >= 0.8, tag + ": error falls at order >= 0.8 (measured " +
                          std::to_string(ord) + ")");
}

} // namespace

int main(int argc, char**)
{
    test::requireDeviceOrSkip();
    const double Q = exactHeat();
    if (argc > 1) {
        for (int n : {48, 64, 80, 96, 128, 160, 192, 256, 320, 384}) {
            const Tally a = ghostTally(n, false), b = ghostTally(n, true);
            // Ghosts whose sample stencil drops a solid cell (a zero
            // renormalized weight), and the fallback count.
            const auto g = Grid::make(n, n, 1, 4.0, 4.0, 1.0);
            const double cx = 2.0 + 0.013 * g.hx, cy = 2.0 + 0.017 * g.hy;
            const auto T = gpu::classifyCompGhost(
                g, [=](double x, double y, double) {
                    return std::hypot(x - cx, y - cy) - kR1; },
                nullptr, AxisRemap3{Focal1D::identity(4.0),
                                    Focal1D::identity(4.0),
                                    Focal1D::identity(1.0)}, nullptr);
            int cut = 0;
            for (int q = 0; q < T.nGhost(); ++q)
                for (int s = 0; s < 4; ++s) // nz == 1: z pairs repeat
                    if (T.gw[8 * q + s] == 0) { ++cut; break; }
            std::printf("n %4d cart R/h %6.1f err %+.4e | focal R/h %6.1f err "
                        "%+.4e | ghosts %d with a dropped solid sample %d "
                        "fallback %d\n",
                        n, a.rh, a.heat / Q - 1, b.rh, b.heat / Q - 1,
                        T.nGhost(), cut, T.fallback);
        }
        return 0;
    }
    std::printf("exact heat per unit depth %.6e (kappa %.4e, T1/T2 %.2f)\n", Q,
                Transport::constant(kGam, kMu).kappa(kTInf), kT1 / kT2);

    {
        const Tally a = ghostTally(64, false), b = ghostTally(128, false);
        if (a.ok && b.ok) verdict("ghost Cartesian", a, b, Q);
    }
    {
        const Tally a = ghostTally(96, true), b = ghostTally(192, true);
        if (a.ok && b.ok) verdict("ghost focal", a, b, Q);
    }
    {
        const Tally a = ogridTally(100, 18), b = ogridTally(200, 34);
        if (a.ok && b.ok) verdict("O-grid", a, b, Q);
    }
    return dnstest::finish();
}
