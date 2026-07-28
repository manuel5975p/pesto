// Solver side of the body-fitted O-grid: the two boundary rows
// (gpu/ogrid_bc.hpp) and the time loop (gpu/runner_ogrid.hpp).
//
// The checks, in the order they catch things:
//   * a uniform freestream is a fixed point -- one metric, wrap or BC sign
//     error anywhere shows up here first;
//   * a quiescent gas stays quiescent and the closed wall face row reports
//     exactly zero force and zero heat;
//   * no-slip actually holds at the wall face, and nz == 1 keeps mz EXACTLY 0;
//   * the isothermal wall holds Twall on the wall face, the adiabatic wall
//     passes ~no heat, and the isothermal heat has the physical sign;
//   * an acoustic pulse leaves through the far boundary;
//   * a short viscous cylinder run stays finite and yields positive drag of a
//     sane magnitude. NO literature band -- bands belong in the quarantined
//     val_* tier (docs/validation.md).
#include "dns/bodies.hpp"
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/output.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "dns/gpu/runner_ogrid.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

// Freestream in box units at Mach 0.2: rho = u = 1, p = 1 / (gamma M^2).
constexpr double kGam = 1.4, kRgas = 1.0, kMach = 0.2;
constexpr double kRhoInf = 1.0, kUInf = 1.0;
const double kPInf = kRhoInf * kUInf * kUInf / (kGam * kMach * kMach);
const double kTInf = kPInf / (kRhoInf * kRgas);
const double kCInf = std::sqrt(kGam * kPInf / kRhoInf);

constexpr double kR = 0.5, kRFar = 5.0, kSpan = 1.0, kCx = 0.0, kCy = 0.0;

Transport gasOf(double mu)
{
    return Transport::air(mu, kTInf, 300.0, 0.72, kGam, kRgas, false);
}

OGridSpec specOf(double beta)
{
    return {.cx = kCx, .cy = kCy, .cz = 0.0, .rFar = kRFar, .beta = beta,
            .span = kSpan};
}

// Physical cell centre: the mean of the cell's corner nodes (one z layer at
// nz == 1, where every node shares the same z).
std::array<double, 3> cellCentre(const CurvMesh& m, int i, int j, int k)
{
    const int nk = m.g.nz == 1 ? 1 : 2;
    std::array<double, 3> s{0, 0, 0};
    for (int dk = 0; dk < nk; ++dk)
        for (int dj = 0; dj < 2; ++dj)
            for (int di = 0; di < 2; ++di)
            {
                const auto p = m.node(i + di, j + dj, k + dk);
                for (int d = 0; d < 3; ++d) s[d] += p[d];
            }
    const double inv = 1.0 / (4.0 * nk);
    for (double& v : s) v *= inv;
    return s;
}

void fillUniform(const Grid& g, CompState& h, double rho, double u, double p)
{
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        h.rho[c] = rho;
        h.mx[c] = rho * u;
        h.my[c] = 0.0;
        h.mz[c] = 0.0;
        h.E[c] = p / (kGam - 1.0) + 0.5 * rho * u * u;
    }
}

struct Prim { double rho, u, v, w, p, T; };

Prim primOf(const CompState& h, std::size_t c)
{
    const double r = h.rho[c];
    const double u = h.mx[c] / r, v = h.my[c] / r, w = h.mz[c] / r;
    const double p =
        (kGam - 1.0) * (h.E[c] - 0.5 * r * (u * u + v * v + w * w));
    return {r, u, v, w, p, p / (kRgas * r)};
}

// The O-grid plus everything the GPU side needs, built once per case.
struct Case
{
    Grid g;
    OGrid og;
    CurvMetrics m;
};

std::expected<Case, std::string> makeCase(int nx, int ny, int nz, double beta)
{
    const auto g = ogridGrid(nx, ny, nz, kR, kRFar, kSpan);
    auto og = buildOGrid(g, sdfOf(CylinderZ{kCx, kCy, kR}), specOf(beta));
    if (!og) return std::unexpected(og.error());
    auto m = CurvMetrics::build(og->mesh);
    if (!m) return std::unexpected(m.error());
    return Case{.g = g, .og = std::move(*og), .m = std::move(*m)};
}

// --- 1. freestream preservation ---------------------------------------------
//
// The single most valuable check: a uniform freestream must be a fixed point of
// the metric flux AND of the far-field row. The inner wall is no-slip, so it
// legitimately disturbs the freestream -- but only within its stencil reach,
// which is one radial cell per RK stage, i.e. 3 per step. The check therefore
// runs a few steps and inspects the rows beyond that cone, plus the far ghost
// row itself, which must come back EXACTLY the freestream.
// Run at nz == 1 (the exact 2D restriction) and at nz > 1: neither the metric
// flux nor either boundary row branches on z, so the same numbers must come out
// of both, and the spanwise-periodic path has to preserve the freestream too.
void freestreamFixedPoint(int nz)
{
    std::printf("--- freestream fixed point, nz = %d ---\n", nz);
    const int nx = 96, ny = 50, nSteps = 5;
    auto cs = makeCase(nx, ny, nz, 0.0);
    if (!cs) { check(false, "O-grid builds: " + cs.error()); return; }
    const Grid& g = cs->g;

    auto dev = gpu::DeviceComp::make(cs->m, gasOf(0.01));
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto bc = gpu::DeviceOGridBc::make(
        g, {.gamma = kGam, .Rgas = kRgas,
            .thermal = gpu::OGridThermal::Adiabatic, .Twall = kTInf,
            .rhoInf = kRhoInf, .uInf = kUInf, .pInf = kPInf});
    if (!dev || !ds || !red || !bc) { check(false, "GPU alloc"); return; }

    auto host = CompState::zeros(g);
    fillUniform(g, host, kRhoInf, kUInf, kPInf);
    (void)ds->upload(host);
    bc->apply(*ds);

    // The far row alone, before any stepping. The deviation form makes the
    // freestream a fixed point of the characteristic algebra exactly, so what
    // is left is the FP32 recovery of p out of E -- a fixed relative eps E / p,
    // NOT something that grows with uInf, which is what rotating the total
    // velocity instead would have produced.
    auto after = CompState::zeros(g);
    (void)ds->download(after);
    double gErr = 0.0;
    bool mzGhostZero = true;
    for (int k = 0; k < g.nz; ++k)
        for (int i = 0; i < nx; ++i)
        {
            const std::size_t c = g.idx(i, ny - 1, k);
            const Prim q = primOf(after, c);
            gErr = std::max({gErr, std::abs(q.rho - kRhoInf) / kRhoInf,
                             std::abs(q.u - kUInf) / kUInf,
                             std::abs(q.v) / kUInf,
                             std::abs(q.p - kPInf) / kPInf});
            if (after.mz[c] != 0.0) mzGhostZero = false;
        }
    std::printf("far ghost row on a uniform freestream: max rel err %.2e\n",
                gErr);
    check(gErr < 1e-5,
          "the far-field row reproduces a uniform freestream to FP32 rounding");
    if (nz == 1)
        check(mzGhostZero, "nz == 1: the far row leaves mz EXACTLY zero");

    dev->setStageHook([&](gpu::DeviceCompState& s) { bc->apply(s); });
    const double dt = 0.5 * dev->stableDt(*ds, *red, 0.4);
    for (int s = 0; s < nSteps; ++s) dev->step(*ds, gpu::Real(dt));
    bc->apply(*ds);
    (void)ds->download(after);

    const int jFrom = 1 + 3 * nSteps + 2; // outside the wall's stencil cone
    double dRho = 0, dU = 0, dV = 0, dP = 0;
    for (int k = 0; k < g.nz; ++k)
        for (int j = jFrom; j <= ny - 1; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const Prim q = primOf(after, g.idx(i, j, k));
                dRho = std::max(dRho, std::abs(q.rho - kRhoInf) / kRhoInf);
                dU = std::max(dU, std::abs(q.u - kUInf) / kUInf);
                dV = std::max(dV, std::abs(q.v) / kUInf);
                dP = std::max(dP, std::abs(q.p - kPInf) / kPInf);
            }
    std::printf("rows %d..%d after %d steps (dt %.2e): drho %.2e du %.2e "
                "dv %.2e dp %.2e\n",
                jFrom, ny - 1, nSteps, dt, dRho, dU, dV, dP);
    check(dRho < 1e-5 && dU < 1e-5 && dV < 1e-5 && dP < 1e-5,
          "a uniform freestream is a fixed point to FP32 rounding");
}

// --- 2. quiescent gas, zero force, zero heat --------------------------------
//
// The closed wall face row's area vectors sum to zero, so a uniform state can
// produce no net force whatever the contour -- the same invariant
// test_gpu_comp_ghost_focal::staticForce pins for the immersed wall.
void quiescentZeroForce()
{
    std::printf("--- quiescent gas ---\n");
    const int nx = 96, ny = 50;
    auto cs = makeCase(nx, ny, 1, 2.0);
    if (!cs) { check(false, "O-grid builds: " + cs.error()); return; }
    const Grid& g = cs->g;

    auto dev = gpu::DeviceComp::make(cs->m, gasOf(0.01));
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto bc = gpu::DeviceOGridBc::make(
        g, {.gamma = kGam, .Rgas = kRgas,
            .thermal = gpu::OGridThermal::Adiabatic, .Twall = kTInf,
            .rhoInf = kRhoInf, .uInf = 0.0, .pInf = kPInf});
    if (!dev || !ds || !red || !bc) { check(false, "GPU alloc"); return; }

    auto host = CompState::zeros(g);
    fillUniform(g, host, kRhoInf, 0.0, kPInf);
    (void)ds->upload(host);
    bc->apply(*ds);

    const auto F0 = bc->surface(*dev, *ds);
    const double scale = 0.5 * kRhoInf * kUInf * kUInf * 2 * kR; // qRef
    std::printf("static: force (%.3e, %.3e, %.3e) heat %.3e\n", F0.fx, F0.fy,
                F0.fz, F0.heat);
    check(std::abs(F0.fx) < 1e-4 * scale && std::abs(F0.fy) < 1e-4 * scale &&
              std::abs(F0.fz) < 1e-4 * scale,
          "a uniform state produces zero net force on the wall face row");
    check(std::abs(F0.heat) < 1e-4, "a uniform state passes no wall heat");

    dev->setStageHook([&](gpu::DeviceCompState& s) { bc->apply(s); });
    const double dt = 0.5 * dev->stableDt(*ds, *red, 0.4);
    for (int s = 0; s < 50; ++s) dev->step(*ds, gpu::Real(dt));
    bc->apply(*ds);
    (void)ds->download(host);

    double vmax = 0.0;
    bool mzZero = true, finite = true;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        vmax = std::max({vmax, std::abs(host.mx[c]), std::abs(host.my[c])});
        if (host.mz[c] != 0.0) mzZero = false;
        if (!std::isfinite(host.rho[c]) || !std::isfinite(host.E[c]))
            finite = false;
    }
    const auto F = bc->surface(*dev, *ds);
    std::printf("after 50 steps: max|m| %.3e force (%.3e, %.3e) heat %.3e\n",
                vmax, F.fx, F.fy, F.heat);
    check(finite, "quiescent evolution stays finite");
    check(vmax < 1e-4, "a quiescent gas stays quiescent on the O-grid");
    check(mzZero, "nz == 1: mz stays EXACTLY zero");
    check(std::abs(F.fx) < 1e-4 * scale && std::abs(F.fy) < 1e-4 * scale,
          "no spurious force after evolution");
    check(std::abs(F.heat) < 1e-4, "no spurious wall heat after evolution");

    bool same = true;
    for (int rep = 0; rep < 4; ++rep)
    {
        const auto Fr = bc->surface(*dev, *ds);
        if (Fr.fx != F.fx || Fr.fy != F.fy || Fr.fz != F.fz ||
            Fr.heat != F.heat)
            same = false;
    }
    check(same, "the wall surface tally is bit-reproducible");
}

// --- 3. the wall itself: no-slip, isothermal, adiabatic ---------------------

struct WallResult
{
    double faceSpeed = 0;  // max |u| interpolated to the wall face
    double faceTErr = 0;   // max |T_face - Twall| / Twall
    double heat = 0;
    double firstRowSpeed = 0;
    bool mzZero = true;
    bool finite = true;
    bool ok = false;
};

WallResult runWall(const Case& cs, gpu::OGridThermal thermal, double Twall,
                   int steps)
{
    const Grid& g = cs.g;
    WallResult w;
    auto dev = gpu::DeviceComp::make(cs.m, gasOf(0.01));
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto bc = gpu::DeviceOGridBc::make(
        g, {.gamma = kGam, .Rgas = kRgas, .thermal = thermal, .Twall = Twall,
            .rhoInf = kRhoInf, .uInf = kUInf, .pInf = kPInf});
    if (!dev || !ds || !red || !bc) return w;

    auto host = CompState::zeros(g);
    fillUniform(g, host, kRhoInf, kUInf, kPInf);
    (void)ds->upload(host);
    dev->setStageHook([&](gpu::DeviceCompState& s) { bc->apply(s); });
    bc->apply(*ds);
    const double dt = 0.5 * dev->stableDt(*ds, *red, 0.4);
    for (int s = 0; s < steps; ++s) dev->step(*ds, gpu::Real(dt));
    bc->apply(*ds);
    (void)ds->download(host);

    for (int k = 0; k < g.nz; ++k)
        for (int i = 0; i < g.nx; ++i)
        {
            const Prim q0 = primOf(host, g.idx(i, 0, k));
            const Prim q1 = primOf(host, g.idx(i, 1, k));
            const double uf = 0.5 * (q0.u + q1.u), vf = 0.5 * (q0.v + q1.v),
                         wf = 0.5 * (q0.w + q1.w);
            w.faceSpeed = std::max(w.faceSpeed,
                                   std::sqrt(uf * uf + vf * vf + wf * wf));
            w.faceTErr = std::max(
                w.faceTErr, std::abs(0.5 * (q0.T + q1.T) - Twall) / Twall);
            w.firstRowSpeed =
                std::max(w.firstRowSpeed,
                         std::sqrt(q1.u * q1.u + q1.v * q1.v + q1.w * q1.w));
        }
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        if (host.mz[c] != 0.0) w.mzZero = false;
        if (!std::isfinite(host.rho[c]) || !std::isfinite(host.E[c]))
            w.finite = false;
    }
    w.heat = bc->surface(*dev, *ds).heat;
    w.ok = true;
    return w;
}

void wallConditions()
{
    std::printf("--- wall conditions ---\n");
    const auto cs = makeCase(96, 50, 1, 2.0);
    if (!cs) { check(false, "O-grid builds: " + cs.error()); return; }

    const double Twall = 1.5 * kTInf;
    const WallResult ad = runWall(*cs, gpu::OGridThermal::Adiabatic, kTInf, 200);
    const WallResult iso =
        runWall(*cs, gpu::OGridThermal::Isothermal, Twall, 200);
    if (!ad.ok || !iso.ok) { check(false, "wall runs built"); return; }

    // kappa * Tinf * perimeter: the natural scale of a wall heat integral here.
    const Transport gas = gasOf(0.01);
    const double qScale = gas.kappa(kTInf) * kTInf * 2 * pi * kR;
    std::printf("adiabatic : face|u| %.2e, row1 |u| %.3f, heat %.3e\n",
                ad.faceSpeed, ad.firstRowSpeed, ad.heat);
    std::printf("isothermal: face|u| %.2e, face T err %.2e, heat %.3e "
                "(scale %.3e)\n",
                iso.faceSpeed, iso.faceTErr, iso.heat, qScale);

    check(ad.finite && iso.finite, "both wall runs stay finite");
    check(ad.mzZero && iso.mzZero, "nz == 1: mz stays EXACTLY zero at the wall");
    check(ad.faceSpeed < 1e-6 * kUInf && iso.faceSpeed < 1e-6 * kUInf,
          "no-slip holds: the wall-face velocity is zero");
    check(ad.firstRowSpeed < 0.5 * kUInf,
          "the wall actually slows the first fluid row");
    check(iso.faceTErr < 1e-5,
          "the isothermal wall holds Twall on the wall face");
    // The wall is hotter than the gas, so energy leaves the body: `heat` is
    // oriented INTO the body and must be negative.
    check(iso.heat < -0.05 * qScale,
          "a hot isothermal wall passes heat OUT of the body");
    check(std::abs(ad.heat) < 0.02 * std::abs(iso.heat),
          "the adiabatic wall passes ~no heat");
}

// --- 4. an acoustic pulse must leave through the far boundary ---------------
void acousticEscape()
{
    std::printf("--- acoustic escape ---\n");
    const int nx = 96, ny = 50;
    auto cs = makeCase(nx, ny, 1, 0.0);
    if (!cs) { check(false, "O-grid builds: " + cs.error()); return; }
    const Grid& g = cs->g;

    auto dev = gpu::DeviceComp::make(cs->m, gasOf(0.0)); // inviscid: pure sound
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto bc = gpu::DeviceOGridBc::make(
        g, {.gamma = kGam, .Rgas = kRgas,
            .thermal = gpu::OGridThermal::Adiabatic, .Twall = kTInf,
            .rhoInf = kRhoInf, .uInf = 0.0, .pInf = kPInf});
    if (!dev || !ds || !red || !bc) { check(false, "GPU alloc"); return; }

    // Isentropic pressure ring at mid radius, gas otherwise at rest: a pure
    // acoustic pulse, half of it outbound and half of it bouncing off the body.
    const double amp = 0.05, r0 = 0.5 * (kR + kRFar), wid = 0.12 * (kRFar - kR);
    auto host = CompState::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const std::size_t c = g.idx(i, j, k);
                const auto x = cellCentre(cs->og.mesh, i, j, k);
                const double rr = std::hypot(x[0] - kCx, x[1] - kCy);
                const double s = (rr - r0) / wid;
                const double p = kPInf * (1.0 + amp * std::exp(-s * s));
                const double rho = kRhoInf * std::pow(p / kPInf, 1.0 / kGam);
                host.rho[c] = rho;
                host.E[c] = p / (kGam - 1.0);
            }
    (void)ds->upload(host);
    dev->setStageHook([&](gpu::DeviceCompState& s) { bc->apply(s); });
    bc->apply(*ds);

    // Ten radial transits: the outbound half leaves immediately, the inbound
    // half bounces off the body once and follows it out.
    const double tEnd = 10.0 * (kRFar - kR) / kCInf;
    double t = 0.0;
    int steps = 0;
    while (t < tEnd)
    {
        const double dt = std::min(tEnd - t, dev->stableDt(*ds, *red, 0.4));
        if (!(dt > 0) || !std::isfinite(dt)) break;
        dev->step(*ds, gpu::Real(dt));
        t += dt;
        ++steps;
    }
    bc->apply(*ds);
    (void)ds->download(host);

    double res = 0.0;
    bool finite = true;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 1; j <= ny - 2; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const Prim q = primOf(host, g.idx(i, j, k));
                if (!std::isfinite(q.p)) finite = false;
                res = std::max(res, std::abs(q.p - kPInf) / (amp * kPInf));
            }
    std::printf("t %.3f (%d steps): residual |p-pInf| / initial amplitude "
                "%.4f\n",
                t, steps, res);
    check(finite, "the acoustic run stays finite");
    check(res < 0.15,
          "an acoustic pulse leaves through the far boundary without a large "
          "reflection");
}

// --- 5. a short viscous cylinder run through the runner ---------------------
void cylinderRun()
{
    std::printf("--- Re 100 cylinder, runOGrid ---\n");
    const int nx = 128, ny = 66;
    auto cs = makeCase(nx, ny, 1, 2.0);
    if (!cs) { check(false, "O-grid builds: " + cs.error()); return; }
    const Grid& g = cs->g;
    std::printf("mesh: h1 %.5f (max %.5f) growth %.4f normal dev %.2e rad\n",
                cs->og.h1, cs->og.h1Max, cs->og.growth, cs->og.normalDevMax);

    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "dns_test_gpu_ogrid";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp, ec);

    RunMeta meta;
    meta.nx = g.nx; meta.ny = g.ny; meta.nz = g.nz;
    meta.hx = g.hx; meta.hy = g.hy; meta.hz = g.hz;
    meta.D = 2 * kR; meta.re = 100; meta.mach = kMach; meta.frameDt = 3.0;
    meta.body = "cylinder"; meta.farField = "ogrid";
    meta.physics = "compressible";

    RunOutput::Config oc;
    oc.prefix = (tmp / "ogrid").string();
    oc.kinds = {Emit::Stats, Emit::Bin};
    oc.fdt = 3.0;
    auto out = RunOutput::make(oc, meta, g, FrameGeometry{});
    if (!out) { check(false, "RunOutput::make: " + out.error()); return; }

    const double D = 2 * kR, Re = 100.0;
    gpu::OGridRun r;
    r.tr = gasOf(kRhoInf * kUInf * D / Re);
    r.rhoInf = kRhoInf;
    r.uInf = kUInf;
    r.pInf = kPInf;
    r.thermal = gpu::OGridThermal::Adiabatic;
    r.Twall = kTInf;
    r.D = D;
    r.qRef = 0.5 * kRhoInf * kUInf * kUInf * D;
    r.cfl = 0.4;
    r.tEnd = 6.0;
    r.ramp = 1.5;
    r.statsEvery = 200;

    // Quiet start: the IC is at REST, exactly consistent with the no-slip wall.
    auto ic = CompState::zeros(g);
    fillUniform(g, ic, kRhoInf, 0.0, kPInf);

    const auto st = gpu::runOGrid(g, cs->og, ic, r, *out);
    if (!st)
    {
        check(false, "runOGrid: " + st.error().message);
        return;
    }
    std::printf("steps %ld, %.3f ms/step, cd %.4f cl %+.4f cs %+.4e\n",
                st->steps, st->msPerStep, st->cd, st->cl, st->cs);
    check(st->steps > 0, "the O-grid runner takes steps");
    check(std::isfinite(st->cd) && std::isfinite(st->cl),
          "the reported coefficients are finite");
    // No literature band here on purpose: this is a smoke test, and a band
    // belongs in the quarantined val_* tier. Positive drag of the right order
    // is what a sign, orientation or metric error would break.
    check(st->cd > 0.3 && st->cd < 4.0,
          "the cylinder drag is positive and of a sane magnitude");
    check(std::abs(st->cl) < 2.0, "the lift stays bounded");
    check(std::filesystem::exists(oc.prefix + "_stats.csv"),
          "the run wrote its stats series");
    std::filesystem::remove_all(tmp, ec);
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    freestreamFixedPoint(1);
    freestreamFixedPoint(4);
    quiescentZeroForce();
    wallConditions();
    acousticEscape();
    cylinderRun();
    return dnstest::finish();
}
