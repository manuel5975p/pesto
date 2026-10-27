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
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/transport.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/rotortable.hpp"
#include "dns/remap.hpp"
#include "dns/rotor.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"
#include "validation_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numbers>
#include <regex>
#include <string>
#include <string_view>

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
    // Band: measured 2026-10-27 on gfx1100, 64^3: 5.5e-6 at t=1, 8.9e-6 at
    // t=2 (gfx1201 2026-07-31: 6.4e-6 / 1.1e-5; FP32 roundoff of the
    // corner-potential telescoping). The pointwise
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
    const gpu::DeviceCompGhost::WallMotion uw = gpu::rotorWallMotion(rot);
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
    // The per-face split (--emit surface) on a MOVING wall, both frames: its
    // rows' moment is the tallied torque, and the parts say where it lives.
    if (const auto rows = gh->surfaceFaces(*dev, *ds, pInf); rows) {
        double q = 0, qp = 0, qc = 0, qv = 0, gross = 0;
        for (const auto& r : *rows) {
            const double dx = r.x[0] - cx, dy = r.x[1] - cy;
            q += dx * r.f[1] - dy * r.f[0];
            qp += dx * r.fp[1] - dy * r.fp[0];
            qc += dx * r.fc[1] - dy * r.fc[0];
            qv += dx * r.fv[1] - dy * r.fv[0];
            gross += std::abs(dx * r.f[1] - dy * r.f[0]);
        }
        std::printf("%s split: torque rows %.6e vs tally %.6e; pressure %.3e "
                    "convective %.3e viscous %.3e\n",
                    mrf ? "mrf" : "direct", q, F.qz, qp, qc, qv);
        check(std::abs(q - F.qz) < 1e-6 * gross,
              "split rows' torque == the tally's (moving wall)");
    } else {
        check(false, "surfaceFaces on the spinning cylinder");
    }
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

void testFrameParity(const SpinResult& direct)
{
    const double tEnd = 3.0;
    const auto mrf = spinUp(true, tEnd);
    std::printf("parity: direct torque %.6e uTheta %.6e | mrf torque %.6e "
                "uTheta %.6e\n",
                direct.torque, direct.uTheta, mrf.torque, mrf.uTheta);
    check(direct.torque < 0.0 && mrf.torque < 0.0,
          "both frames report a torque opposing the spin");
    // Axisymmetric outer conditions: frozen rotor is exact, so the two runs
    // differ only at truncation level. Measured 2026-10-27 on gfx1100: torque
    // -8.94427e-2 direct vs -8.94411e-2 mrf (1.8e-5 relative), uTheta gap
    // 1.7e-5 relative (2026-07-31 gfx1201, before the wall-acceleration
    // pressure image: 1.0e-4 / 3.7e-5). Banded 1e-3: >= 10x either record.
    std::printf("parity: torque gap %.2e, uTheta gap %.2e (relative)\n",
                std::abs(mrf.torque / direct.torque - 1),
                std::abs(mrf.uTheta / direct.uTheta - 1));
    check(std::abs(mrf.torque - direct.torque) <
              1e-3 * std::abs(direct.torque),
          "torque agrees between the frames (axisymmetric rig, 1e-3)");
    check(std::abs(mrf.uTheta - direct.uTheta) <
              1e-3 * std::abs(direct.uTheta),
          "driven swirl agrees between the frames (1e-3)");
}

// --- 3. the node-psi zone on the body-fitted O-grid (setMrfCurv) ------------
//
// Same two claims, general-mesh path. Transparency has no separable table to
// lean on here: the correction is the per-node stream potential, whose face
// differences must telescope on the annulus exactly as the corner form does
// on the box. Parity reuses the SAME physical rig as the Cartesian one --
// spinning cylinder, quiescent far field, identical R, mu, omega, t -- so the
// O-grid MRF torque is pinned against the Cartesian DIRECT (moving-wall) run:
// two wall discretizations, two frames, one flow.

Transport rigGas(double gamma_, double mu)
{
    return Transport::air(mu, 1.0, 300.0, 0.72, gamma_, 1.0, false);
}

// Uniform quiescent state on a circular-cylinder O-grid with the zone active:
// one step, no boundary hook, and the rows a polluted ghost row cannot have
// reached yet must be unmoved to FP32-telescoping roundoff. Rows 0..2 and
// ny-3..ny-1 are excluded: the ghost rows have no valid lo face, so their own
// rhs is garbage by contract and it spreads one row per RK stage.
void testOGridTransparency()
{
    const double R = 0.5, rFar = 5.0;
    const int nx = 128, ny = 64;
    const double gamma_ = 1.4, rhoInf = 1.0;
    const double pInf = 1.0 / (gamma_ * 0.04); // c = 5
    const auto g = ogridGrid(nx, ny, 1, R, rFar, 1.0);
    const Sdf sdf = [R](double x, double y, double) {
        return std::hypot(x, y) - R;
    };
    auto og = buildOGrid(g, sdf, {.cx = 0, .cy = 0, .cz = 0, .rFar = rFar,
                                  .beta = 0, .span = 1.0});
    if (!og) { check(false, "ogrid transparency build: " + og.error()); return; }
    auto mtr = CurvMetrics::build(og->mesh);
    if (!mtr) { check(false, "metrics: " + mtr.error()); return; }
    auto dev = gpu::DeviceComp::make(*mtr, rigGas(gamma_, 1e-3), 0.0,
                                     SgsModel::Smagorinsky);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !red) { check(false, "ogrid transparency alloc"); return; }

    gpu::MrfZone z;
    z.axis = 2;
    z.omega = 2.0;
    z.center[0] = z.center[1] = z.center[2] = 0.0;
    z.r1 = 1.5; z.r2 = 2.5;   // blend strictly inside the annulus
    z.a1 = 1e30; z.a2 = 2e30; // z-extruded: degenerate axial blend
    check(dev->setMrfCurv(z, og->mesh) == hipSuccess, "setMrfCurv");

    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gamma_ - 1.0);
    }
    (void)ds->upload(host);
    const double dt = dev->stableDt(*ds, *red);
    dev->step(*ds, gpu::Real(dt));
    (void)ds->download(host);

    double dv = 0.0;
    for (int j = 3; j <= g.ny - 4; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            dv = std::max({dv, std::abs(host.mx[c] / host.rho[c]),
                           std::abs(host.my[c] / host.rho[c]),
                           std::abs(host.mz[c] / host.rho[c]),
                           std::abs(double(host.rho[c]) - rhoInf)});
        }
    std::printf("ogrid transparency: dt %.3e, interior deviation %.3e\n", dt,
                dv);
    check(dv < 1e-4, "a quiescent state passes the node-psi zone (annulus)");
}

// Spin-up of the SAME cylinder on its own body-fitted annulus, MRF frame:
// wall row mirrored about omega x r, node-psi zone over the wall, torque off
// the wall-face tally. Same R, mu, omega and end time as spinUp().
SpinResult spinUpOGrid(double tEnd)
{
    const double L = 2 * pi, R = L / 16, rFar = pi;
    const int nx = 128, ny = 96;
    const double gamma_ = 1.4, rhoInf = 1.0;
    const double pInf = 1.0 / (gamma_ * 0.04);
    const double mu = rhoInf * 2 * R / 20.0;
    const auto g = ogridGrid(nx, ny, 1, R, rFar, 1.0);
    const Sdf sdf = [R](double x, double y, double) {
        return std::hypot(x, y) - R;
    };
    auto og = buildOGrid(g, sdf, {.cx = 0, .cy = 0, .cz = 0, .rFar = rFar,
                                  .beta = 2.0, .span = 1.0});
    if (!og) { check(false, "ogrid spinUp build: " + og.error()); return {}; }
    auto mtr = CurvMetrics::build(og->mesh);
    if (!mtr) { check(false, "ogrid spinUp metrics"); return {}; }
    auto dev = gpu::DeviceComp::make(*mtr, rigGas(gamma_, mu), 0.0,
                                     SgsModel::Smagorinsky);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !ds || !red) { check(false, "ogrid spinUp alloc"); return {}; }

    gpu::MrfZone z;
    z.axis = 2;
    z.omega = 1.0;
    z.center[0] = z.center[1] = z.center[2] = 0.0;
    z.r1 = 2.0 * R; z.r2 = 3.5 * R; // the Cartesian rig's own fractions
    z.a1 = 1e30; z.a2 = 2e30;
    if (dev->setMrfCurv(z, og->mesh) != hipSuccess) {
        check(false, "ogrid spinUp setMrfCurv");
        return {};
    }

    gpu::DeviceOGridBc::Config bc;
    bc.gamma = gamma_;
    bc.Rgas = 1.0;
    bc.rhoInf = rhoInf;
    bc.uInf = 0.0;
    bc.pInf = pInf;
    bc.omega = z.omega;
    bc.cx = 0.0;
    bc.cy = 0.0;
    bc.wallPos = ogridWallMidpoints(*og);
    bc.wallSpan = ogridWallSpans(*og);
    auto wall = gpu::DeviceOGridBc::make(g, bc);
    if (!wall) { check(false, "ogrid spinUp bc"); return {}; }
    dev->setStageHook([&](gpu::DeviceCompState& s) { wall->apply(s); });

    auto host = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        host.rho[c] = rhoInf;
        host.E[c] = pInf / (gamma_ - 1.0);
    }
    (void)ds->upload(host);
    wall->apply(*ds);
    double t = 0.0;
    while (t < tEnd) {
        const double dt = dev->stableDt(*ds, *red);
        dev->step(*ds, gpu::Real(dt));
        t += dt;
    }
    wall->apply(*ds);
    const auto F = wall->surface(*dev, *ds);
    (void)ds->download(host);

    // Tangential average over the SAME wall-distance band the Cartesian rig
    // samples (its own grid spacing), cell centres off the mesh nodes.
    const double h = L / 192;
    SpinResult out{.torque = F.tz};
    int m = 0;
    for (int j = 1; j < ny - 1; ++j)
        for (int i = 0; i < nx; ++i) {
            const auto a = og->mesh.node(i, j, 0);
            const auto b = og->mesh.node(i + 1, j, 0);
            const auto c2 = og->mesh.node(i, j + 1, 0);
            const auto d2 = og->mesh.node(i + 1, j + 1, 0);
            const double x = 0.25 * (a[0] + b[0] + c2[0] + d2[0]);
            const double y = 0.25 * (a[1] + b[1] + c2[1] + d2[1]);
            const double r = std::hypot(x, y);
            if (r - R < h || r - R > 2.0 * h) continue;
            const std::size_t c = g.idx(i, j, 0);
            const double tx = -y / r, ty = x / r;
            out.uTheta += (host.mx[c] / host.rho[c]) * tx +
                          (host.my[c] / host.rho[c]) * ty;
            ++m;
        }
    if (m > 0) out.uTheta /= m;
    return out;
}

void testOGridFrameParity(const SpinResult& direct)
{
    const auto og = spinUpOGrid(3.0);
    std::printf("ogrid parity: direct torque %.6e uTheta %.6e | ogrid mrf "
                "torque %.6e uTheta %.6e\n",
                direct.torque, direct.uTheta, og.torque, og.uTheta);
    check(og.torque < 0.0,
          "the O-grid MRF torque opposes the spin");
    // Two wall discretizations (mesh-face row vs ghost staircase), two
    // frames, one flow: agreement is truncation-level, not roundoff.
    // Measured 2026-10-27 on gfx1100: torque 0.224%, uTheta 0.626%; banded
    // >= 3x (a correct wall fix may legitimately move them).
    std::printf("ogrid parity: torque gap %.3f%%, uTheta gap %.3f%%\n",
                100 * std::abs(og.torque / direct.torque - 1),
                100 * std::abs(og.uTheta / direct.uTheta - 1));
    check(std::abs(og.torque - direct.torque) < 0.01 * std::abs(direct.torque),
          "O-grid MRF torque matches the Cartesian direct run (1%)");
    check(std::abs(og.uTheta - direct.uTheta) < 0.02 * std::abs(direct.uTheta),
          "O-grid MRF driven swirl matches the Cartesian direct run (2%)");
}

// --- 4. bladed MRF end to end: the rotor2 L1 regression pin -----------------
//
// Nothing else pins bladed thrust and torque, so a tally or wall change that
// moves them (78860e3 moved CT -2.4%) passed unseen. This runs the BUILT
// binary on the parity suite's rotor2 L1 grid (tools/parity/cases.toml,
// D/h_fine 39.1 on x) for 1 D/U and pins ct/cq to the values it measured.
// A REGRESSION PIN, not a literature band: CT at 1 D/U is unsettled (the
// 4.5 D/U L1 run is 8% lower), which is fine for a pin. The grid is the
// coarsest that keeps a deep-solid core (blade t/h 3.6..4.4: below that the
// two surfaces read through each other), and the test asserts that core.
// Fixture: tests/data/rotor2.stl = tools/parity/geom.py write_rotor_stl with
// propeller.py `--blades 2 --root-chord 0.40 --tip-chord 0.30 --thickness
// 0.35 --pitch 1.0` (swept diameter scaled to 1).
#if defined(DNS_BIN) && defined(DNS_ROTOR2_STL)
void testRotor2Pin()
{
    const std::string prefix = "gpu_mrf_rotor2";
    const std::string out = prefix + "_stdout.txt";
    // --ramp 0.4 + ft 0.054512: tEnd = 0.4 + 0.054512 * 7.06858 = pi/4 = 1 D/U.
    const int rc = valtest::runBinaryTo(
        "\"" DNS_BIN "\" flow --physics compressible --body \"" DNS_ROTOR2_STL
        "\" --robust --n 150 --ny 100 --nz 100 --dr 6 --focal auto "
        "--focal-ratio 5 --focal-blend 1 --rotor-mach 0.41 --rotor-axis x "
        "--rotor-frame mrf --mach 0.082219 --re 220 --kick 0 --ramp 0.4 "
        "--ft 0.054512 --emit stats --emit surface --out " + prefix, out);
    check(rc == 0, "rotor2 L1 MRF run exits 0");
    if (rc != 0) return;

    std::ifstream f(out);
    const std::string log((std::istreambuf_iterator<char>(f)), {});
    check(log.find("frozen rotor -- an APPROXIMATION") != std::string::npos,
          "the run labels MRF a frozen-rotor approximation");
    std::smatch m;
    const std::regex re(R"(frozen azimuth: (\d+) ghosts, (\d+) deep, (\d+) faces)");
    const bool found = std::regex_search(log, m, re);
    check(found, "the run reports the measured deep-solid count");
    if (found) {
        std::printf("rotor2: %s ghosts, %s deep, %s faces\n", m[1].str().c_str(),
                    m[2].str().c_str(), m[3].str().c_str());
        check(std::stoi(m[2].str()) > 0,
              "the blade keeps a deep-solid core (both surfaces mirror)");
    }

    const auto st = valtest::loadCsv(prefix + "_stats.csv");
    check(st.has_value() && !st->col("t").empty(), "rotor2 stats rows written");
    if (!st || st->col("t").empty()) return;
    const double t = st->col("t").back(), ct = st->col("ct").back(),
                 cq = st->col("cq").back();
    std::printf("rotor2 L1 at t %.6f (1 D/U = %.6f): ct %.6g cq %.6g\n", t,
                pi / 4, ct, cq);
    check(std::abs(t - pi / 4) < 1e-4, "run reaches 1 D/U");
    // Measured 2026-10-27 on gfx1100 with the impermeable interface faces
    // (DeviceCompGhost::applyRhs; before them, with the two-sided faces: ct
    // 0.0508646, cq -0.0874821; before those, W7: ct 0.0565675, cq
    // -0.0911887). Banded 1%: a regression pin.
    constexpr double ctPin = 0.040455, cqPin = -0.0738025;
    std::printf("rotor2 pin: ct %+.3f%%, cq %+.3f%%\n", 100 * (ct / ctPin - 1),
                100 * (cq / cqPin - 1));
    check(std::abs(ct / ctPin - 1) < 0.01, "rotor2 L1 ct at 1 D/U pinned (1%)");
    check(std::abs(cq / cqPin - 1) < 0.01, "rotor2 L1 cq at 1 D/U pinned (1%)");

    // The ghost-wall split (--emit surface, W6): one row per interface face,
    // summing to the tallied thrust and torque (the stats file's 6 digits),
    // pressure + convective + viscous == total, and the parts pinned.
    std::ifstream sf(prefix + "_surface.csv");
    std::string head;
    std::getline(sf, head);
    std::smatch pm;
    const bool hasPivot = std::regex_search(
        head, pm, std::regex(R"(pivot (\S+) (\S+) (\S+))"));
    check(hasPivot, "surface header carries the moment pivot");
    check(head.starts_with("# DNS [MRF frozen rotor: an APPROXIMATION] |"),
          "surface header labels MRF a frozen-rotor approximation");
    const auto sv = valtest::loadCsv(prefix + "_surface.csv");
    check(sv.has_value() && found &&
              sv->rows() == std::size_t(std::stoi(m[3].str())),
          "one surface row per interface face");
    if (sv && hasPivot) {
        const double py = std::stod(pm[2].str()), pz = std::stod(pm[3].str());
        const double thrust = st->col("thrust").back(),
                     torque = st->col("torque").back();
        const double ctDen = thrust / ct, cqDen = torque / cq;
        const auto& y = sv->col("y");
        const auto& z = sv->col("z");
        double fx[4] = {0, 0, 0, 0}, mx[4] = {0, 0, 0, 0}, gross = 0;
        const char* part[4] = {"f", "fp", "fc", "fv"};
        for (std::size_t i = 0; i < sv->rows(); ++i)
            for (int p = 0; p < 4; ++p) {
                const std::string s = part[p];
                const double fxi = sv->col(s + "x")[i];
                fx[p] += fxi;
                mx[p] += (y[i] - py) * sv->col(s + "z")[i] -
                         (z[i] - pz) * sv->col(s + "y")[i];
                if (p == 0) gross += std::abs(fxi);
            }
        std::printf("rotor2 split: ct %.6g = p %.6g + c %.6g + v %.6g; cq %.6g "
                    "= p %.6g + c %.6g + v %.6g\n",
                    fx[0] / ctDen, fx[1] / ctDen, fx[2] / ctDen, fx[3] / ctDen,
                    mx[0] / cqDen, mx[1] / cqDen, mx[2] / cqDen, mx[3] / cqDen);
        check(std::abs(fx[0] / thrust - 1) < 1e-5,
              "surface rows sum to the tallied thrust");
        check(std::abs(mx[0] / torque - 1) < 1e-5,
              "surface rows' moment is the tallied torque");
        check(std::abs(fx[1] + fx[2] + fx[3] - fx[0]) < 1e-8 * gross,
              "pressure + convective + viscous == total");
        // Measured 2026-10-27 on gfx1100, same run as the ct/cq pin, with the
        // impermeable faces (before: p 0.00993798, c 0.00135217, v 0.0395745
        // with the two-sided faces; W6: p 0.0138105, c 0.00556694, v
        // 0.0371901). The MRF wall advects nothing relative to the frame, so
        // the convective part is exactly zero, not pinned.
        constexpr double ctP = -0.00550761, ctV = 0.0459626;
        check(std::abs(fx[1] / ctDen / ctP - 1) < 0.01, "rotor2 ct pressure part pinned (1%)");
        check(fx[2] == 0.0 && mx[2] == 0.0,
              "rotor2 convective part exactly 0 (impermeable relative to the frame)");
        check(std::abs(fx[3] / ctDen / ctV - 1) < 0.01, "rotor2 ct viscous part pinned (1%)");
    }

    std::error_code ec;
    for (const char* suf : {"_stdout.txt", "_stats.csv", ".json", "_mask.bin",
                            "_surface.csv"})
        std::filesystem::remove(prefix + suf, ec);
}
#endif

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    // `rotor2`: the bladed end-to-end pin alone (its own ctest entry,
    // gpu_mrf_rotor2), so the solver-level rigs keep their own budget.
    if (argc > 1 && std::string_view(argv[1]) == "rotor2") {
#if defined(DNS_BIN) && defined(DNS_ROTOR2_STL)
        testRotor2Pin();
#else
        check(false, "rotor2 needs DNS_BIN and DNS_ROTOR2_STL");
#endif
        return dnstest::finish();
    }
    testTransparency();
    const auto direct = spinUp(false, 3.0);
    testFrameParity(direct);
    testOGridTransparency();
    testOGridFrameParity(direct);
    return dnstest::finish();
}
