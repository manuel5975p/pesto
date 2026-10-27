// Two-sided ghost faces (comp_ghost.hpp, CompGhostTables::sideFace).
//
// Where a section is under ~2 cells thick, a ghost cell is face-adjacent to
// fluid on BOTH surfaces but carries one image, mirrored from the surface
// nearest its centre. Before the side images, the fluid cell on the other
// surface fluxed against that image -- the far surface's pressure and mirrored
// velocity -- and the force tally read the same face (issues.md #14,
// exp_mrf14b: up to -81% Cn on a thin lens, and the convective "flux through a
// resting wall").
//
// Rig: two thin periodic membranes (wavy in x, and in z in 3D; thickness
// 1.3..4.2 cells, so their necks are two-sided) split a periodic box into two
// gases at rest at different pressures. A rigid wall holding a pressure jump
// at rest is an EXACT steady state, and the discrete one is exact too when
// every face reads its own side: interior faces see one uniform pressure, the
// wall faces a zero-gradient copy of it (and of the temperature, which jumps
// too on the adiabatic rigs). So:
//   (a) every interface face's KG pressure is its fluid cell's gas pressure
//       (surfaceFaces): a wrong-side face reads the average of the two gases,
//       an error of half the jump;
//   (b) each membrane's tallied normal force is the jump times its area;
//   (c) the gas stays at rest: max |u| after 60 steps against the acoustic
//       scale of the jump, dp / (rho c). A wrong-side face drives a flow of
//       that order through the neck.
// Variants: 2D and 3D Cartesian, a focal (curvilinear, anisotropic) mesh, an
// isothermal wall at the gas temperature, and a "moving" wall at zero
// velocity (the rotor table's uwall path).
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

constexpr double gamma_ = 1.4, rhoInf = 1.0, mu = 0.01;
const double pInf = 1.0 / (gamma_ * 0.04);
const double pHi = 1.05 * pInf, pLo = 0.95 * pInf;
// Membrane centre lines: sub-cell offsets that differ, so the two necks sit
// on different lattice positions.
constexpr double yc[2] = {0.25, 0.75};
constexpr double yoff[2] = {0.30, 0.65}; // in normal cells

struct Rig
{
    std::string name;
    int n, nz;
    bool focal = false;
    bool iso = false;
    bool moving = false;
};

void run(const Rig& rig)
{
    Grid g;
    g.nx = rig.n;
    g.ny = rig.n;
    g.nz = rig.nz;
    g.hx = g.hy = 1.0 / rig.n;
    g.hz = 1.0 / rig.n;
    const double lo[3] = {0.0, 0.18, 0.0}, hi[3] = {0.0, 0.82, 0.0},
                 w[3] = {0.0, 0.12, 0.0};
    const AxisRemap3 R = focalRemap(g, rig.focal, lo, hi, w, 2.0);
    // The spacing normal to the membranes where they sit.
    const double hn = rig.focal ? R.fy.c * R.fy.r * g.hy : g.hy;
    double c0[2];
    for (int m = 0; m < 2; ++m) c0[m] = yc[m] + yoff[m] * hn;
    // Half thickness of membrane m at (x, z): 1.3 .. 4.2 normal cells (the
    // 3D one also waves in z), so each membrane has a two-sided neck.
    auto halfT = [&](int m, double x, double z) {
        double t = 2.75 + 1.45 * std::cos(2 * pi * x + m);
        if (g.nz > 1) t += 0.4 * std::sin(2 * pi * z + 0.7 * m) - 0.4;
        return 0.5 * std::max(t, 1.3) * hn;
    };
    // sdf: distance to the nearer membrane, the slab formula scaled by the
    // slope so it stays close to a distance.
    const gpu::GhostSdf sdf = [&](double x, double y, double z) {
        double best = 1e300;
        for (int m = 0; m < 2; ++m) {
            const double e = 1e-4;
            const double dx = (halfT(m, x + e, z) - halfT(m, x - e, z)) / (2 * e);
            const double dz = (halfT(m, x, z + e) - halfT(m, x, z - e)) / (2 * e);
            const double d = (std::abs(y - c0[m]) - halfT(m, x, z))
                             / std::sqrt(1 + dx * dx + dz * dz);
            best = std::min(best, d);
        }
        return best;
    };
    // The gas of a point: high between the membranes, low outside.
    auto gasP = [&](double y) { return y > c0[0] && y < c0[1] ? pHi : pLo; };

    std::optional<CurvMetrics> cm;
    if (rig.focal) {
        auto m = CurvMetrics::build(CurvMesh::fromRemap(g, R));
        if (!m) { check(false, rig.name + ": metrics: " + m.error()); return; }
        cm = std::move(*m);
    }
    auto dev = rig.focal ? gpu::DeviceComp::make(*cm, gamma_, mu)
                         : gpu::DeviceComp::make(g, gamma_, mu);
    if (!dev) { check(false, rig.name + ": solver"); return; }
    const double Rgas = dev->transport().Rgas;
    const double T0 = pInf / (rhoInf * Rgas);
    const gpu::GhostWallThermal wall{
        .mode = rig.iso ? gpu::GhostWallThermal::Mode::Isothermal
                        : gpu::GhostWallThermal::Mode::Adiabatic,
        .Twall = T0,
        .Rgas = Rgas};
    const gpu::GhostWallMotion still = [](double, double, double) {
        return gpu::GhostWallKinematics{};
    };
    const auto tb = gpu::classifyCompGhost(g, sdf, nullptr, R,
                                           rig.moving ? &still : nullptr);
    gpu::CompGhostCapacity cap;
    cap.cover(tb);
    auto gh = gpu::DeviceCompGhost::fromTables(g, tb, cap, gamma_, rhoInf,
                                               pInf, rig.focal, wall);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!gh || !ds || !red) { check(false, rig.name + ": setup"); return; }
    std::printf("%s: %d ghosts, %d deep, %d faces, %d side faces (%d skipped), "
                "hn %.4f\n",
                rig.name.c_str(), tb.nGhost(), tb.nDeep(), tb.nFace(),
                tb.nSide(), tb.sideSkipped, hn);
    check(tb.nSide() > 0, rig.name + ": the necks are two-sided (premise)");
    check(tb.sideSkipped == 0, rig.name + ": every wrong-side face imaged");
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });
    dev->setRhsHook([&](const gpu::DeviceCompState& s,
                        gpu::DeviceCompState& out,
                        const gpu::StageCombine& cmb) {
        gh->applyRhs(*dev, s, out, cmb);
    });

    // Physical cell centres (the remap of the reference centres).
    auto ccy = [&](int j) { return R.fy.map((j + 0.5) * g.hy); };
    auto h = CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double p = gasP(ccy(j));
                // Isothermal rig: both gases at the wall temperature. Else
                // a temperature jump too (an adiabatic wall holds it), so a
                // wrong-side face also conducts.
                h.rho[c] = rig.iso ? p / (Rgas * T0)
                                   : (p == pHi ? rhoInf : 1.2 * rhoInf);
                h.E[c] = p / (gamma_ - 1.0);
            }
    (void)ds->upload(h);
    gh->apply(*ds);

    // (a) + (b) at the ghost-filled initial state.
    auto rows = gh->surfaceFaces(*dev, *ds, pInf);
    check(rows.has_value() && !rows->empty(), rig.name + ": surfaceFaces");
    if (!rows) return;
    const double dp = pHi - pLo;
    double pErr = 0.0, fy[2] = {0.0, 0.0}, area[2] = {0.0, 0.0};
    for (const auto& r : *rows) {
        // The fluid cell's side: back from the face, against the orientation
        // into the body (it sits >= 0.65 hn off the centre line, so a quarter
        // cell decides it); its gas sets the pressure it must see.
        const double yf = r.x[1] - (r.axis == 1 ? 0.25 * r.sign * hn : 0.0);
        pErr = std::max(pErr, std::abs(r.p + pInf - gasP(yf)) / dp);
        const int m = r.x[1] < 0.5 ? 0 : 1;
        fy[m] += r.f[1];
        if (r.axis == 1) area[m] += r.area * (r.sign > 0 ? 1.0 : 0.0);
    }
    // Lower membrane: low gas below pushes up, high gas above pushes down.
    const double Ax = area[0];
    const double fyErr = std::max(std::abs(fy[0] / ((pLo - pHi) * Ax) - 1),
                                  std::abs(fy[1] / ((pHi - pLo) * area[1]) - 1));
    std::printf("%s: max |p_face - p_gas| / dp %.2e; fy %.6e %.6e (rel err "
                "%.2e)\n",
                rig.name.c_str(), pErr, fy[0], fy[1], fyErr);
    check(pErr < 1e-5, rig.name + ": every face reads its own gas (1e-5 dp)");
    check(fyErr < 1e-5, rig.name + ": membrane force = jump x area (1e-5)");

    // (c) at rest after 60 steps.
    for (int s = 0; s < 60; ++s)
        dev->step(*ds, gpu::Real(dev->stableDt(*ds, *red)));
    gh->apply(*ds);
    (void)ds->download(h);
    std::vector<char> solid(g.size(), 0);
    for (int c : gh->solidCells()) solid[std::size_t(c)] = 1;
    double umax = 0.0;
    for (std::size_t c = 0; c < g.size(); ++c) {
        if (solid[c]) continue;
        const double m2 = h.mx[c] * h.mx[c] + h.my[c] * h.my[c]
                          + h.mz[c] * h.mz[c];
        umax = std::max(umax, std::sqrt(m2) / h.rho[c]);
    }
    const double cs = std::sqrt(gamma_ * pInf / rhoInf);
    const double uScale = dp / (rhoInf * cs);
    std::printf("%s: after 60 steps max|u| / (dp / rho c) %.2e\n",
                rig.name.c_str(), umax / uScale);
    check(std::isfinite(umax) && umax < 1e-3 * uScale,
          rig.name + ": the membranes hold the jump at rest (1e-3)");
}

// Thick convex corners are NOT two-sided (exp_corner). At a square's 90 deg
// corner the corner ghost's normal points along one face and its fluid
// neighbour across the other sees the adjacent face: both two-sided tests
// pass, yet the body runs on behind the ghost. Side-imaging those faces made
// a square cylinder offset-sensitive at Re 100 (cd spread 0.7% -> 8.7%,
// cl_rms 3.6% -> 48% over a 0.2-cell shift), so a ghost in THICK solid (solid
// two cells behind it along -n_g AND a deep cell within three) keeps the
// shared image: a thick body classifies no side face (its tables are HEAD's;
// fix A gave 16 / 2 / 224 on the dns square, diamond and spun cube). A thin
// plate with the same square ends still does. Host classification only.
void thickCorners()
{
    // Rounded-free box sdf, half sizes b, rotated by `rot` about x then z.
    auto boxSdf = [](const double c[3], const double b[3], double rz,
                     double rx) {
        return gpu::GhostSdf([=](double x, double y, double z) {
            double p[3] = {x - c[0], y - c[1], z - c[2]};
            const double y1 = std::cos(rx) * p[1] + std::sin(rx) * p[2];
            const double z1 = -std::sin(rx) * p[1] + std::cos(rx) * p[2];
            p[1] = y1;
            p[2] = z1;
            const double x2 = std::cos(rz) * p[0] + std::sin(rz) * p[1];
            const double y2 = -std::sin(rz) * p[0] + std::cos(rz) * p[1];
            p[0] = x2;
            p[1] = y2;
            double out = 0.0, in = -1e300;
            for (int a = 0; a < 3; ++a) {
                const double q = std::abs(p[a]) - b[a];
                out += std::max(q, 0.0) * std::max(q, 0.0);
                in = std::max(in, q);
            }
            return out > 0.0 ? std::sqrt(out) : in;
        });
    };
    struct Case
    {
        std::string name;
        int n, nz;
        double half[3], rz, rx;
        bool thin;
    };
    const double big = 10.0; // past the box: a 2D or z-spanning prism
    const Case cases[] = {
        {"square", 64, 1, {0.2, 0.2, big}, 0.0, 0.0, false},
        {"square 13 deg", 64, 1, {0.2, 0.2, big}, 0.23, 0.0, false},
        {"diamond", 64, 1, {0.15, 0.15, big}, pi / 4, 0.0, false},
        {"cube spun about x", 32, 32, {0.2, 0.2, 0.2}, 0.0, 0.5, false},
        {"plate 1.3 cells", 64, 1, {0.2, 0.65 / 64, big}, 0.0, 0.0, true},
    };
    // Three sub-cell placements per body: which face of a corner is the
    // ghost's own depends on them, the classification must not.
    const double offs[3][2] = {{0.13, 0.41}, {0.37, 0.08}, {0.5, 0.5}};
    for (const Case& k : cases) {
        Grid g;
        g.nx = g.ny = k.n;
        g.nz = k.nz;
        g.hx = g.hy = g.hz = 1.0 / k.n;
        const double z0[3] = {0.0, 0.0, 0.0};
        const AxisRemap3 R = focalRemap(g, false, z0, z0, z0, 1.0);
        for (const auto& o : offs) {
            // The plate's centre line never on a cell centre: a ghost there
            // has no normal (neighbour-imaged, never side-imaged).
            const double oy = o[1] + (k.thin ? 0.21 : 0.0);
            const double c[3] = {0.5 + o[0] * g.hx, 0.5 + oy * g.hy,
                                 k.nz > 1 ? 0.5 + 0.21 * g.hz : 0.5};
            const auto tb = gpu::classifyCompGhost(
                g, boxSdf(c, k.half, k.rz, k.rx), nullptr, R, nullptr);
            std::printf("thick corners: %s offset (%.2f, %.2f): %d ghosts, %d "
                        "deep, %d side faces\n",
                        k.name.c_str(), o[0], o[1], tb.nGhost(), tb.nDeep(),
                        tb.nSide());
            if (k.thin) {
                check(tb.nSide() > 0,
                      k.name + ": a thin plate's faces stay two-sided");
                continue;
            }
            check(tb.nDeep() > 0, k.name + ": the body is thick (premise)");
            check(tb.nSide() == 0 && tb.sideSkipped == 0,
                  k.name + ": a thick body's convex corners are one-sided");
        }
    }
}

} // namespace

int main()
{
    thickCorners();
    test::requireDeviceOrSkip();
    run({"2D", 64, 1});
    run({"3D", 32, 32});
    run({"2D focal", 64, 1, true});
    run({"2D isothermal", 64, 1, false, true});
    run({"2D moving (u = 0)", 64, 1, false, false, true});
    return dnstest::finish();
}
