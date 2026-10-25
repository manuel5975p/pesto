// Rotational equivariance of the body-fitted O-grid (circular cylinder).
//
// With nx divisible by 4 the polar annulus maps onto itself under a rotation
// by 90 deg: node i lands on node i - nx/4 (theta_i = -2 pi i / nx, the
// clockwise convention of dns/ogrid.hpp). So a freestream turned by +90 deg
// must give the same flow turned by +90 deg:
//   B(i) = R90 A(i + nx/4),   F_B = R90 F_A = (-F_A.y, F_A.x).
// Any metric, wrap-offset or sign error that is not rotation-equivariant
// (e.g. one that treats the theta = 0 seam differently) breaks it.
//
// DeviceOGridBc's far row takes a freestream along +x only (Config::uInf), so
// a 90 deg freestream cannot be stated through it. The 90 deg rig therefore
// keeps the class's WALL row and replaces its far row, in BOTH runs, by a
// Dirichlet ghost row holding the freestream state -- one far treatment,
// rotation-equivariant by construction, so what is compared is the interior
// metric flux and the wall. The class's own characteristic far row is pinned
// separately at 180 deg (uInf = +1 vs -1, which it does express):
//   C(i) = R180 A(i + nx/2),  F_C = -F_A.
// Last, a uniform freestream at 37 deg (no axis alignment, no symmetry) with
// Dirichlet ghost rows on both ends must be a fixed point to FP32 rounding:
// the discrete face closure of the mesh, off every axis.
#include "dns/bodies.hpp"
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/ogrid.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/ogrid_bc.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

constexpr double kGam = 1.4, kRgas = 1.0, kMach = 0.2;
constexpr double kRhoInf = 1.0, kUInf = 1.0;
const double kPInf = kRhoInf * kUInf * kUInf / (kGam * kMach * kMach);
const double kTInf = kPInf / (kRhoInf * kRgas);
constexpr double kR = 0.5, kRFar = 5.0;
constexpr int kNx = 96, kNy = 50, kSteps = 50;

struct Case
{
    Grid g;
    OGrid og;
    CurvMetrics m;
};

std::expected<Case, std::string> makeCase(double beta)
{
    const auto g = ogridGrid(kNx, kNy, 1, kR, kRFar, 1.0);
    auto og = buildOGrid(g, sdfOf(CylinderZ{0.0, 0.0, kR}),
                         {.cx = 0, .cy = 0, .cz = 0, .rFar = kRFar,
                          .beta = beta, .span = 1.0});
    if (!og) return std::unexpected(og.error());
    auto m = CurvMetrics::build(og->mesh);
    if (!m) return std::unexpected(m.error());
    return Case{.g = g, .og = std::move(*og), .m = std::move(*m)};
}

CompState uniform(const Grid& g, double u, double v)
{
    auto h = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        h.rho[c] = kRhoInf;
        h.mx[c] = kRhoInf * u;
        h.my[c] = kRhoInf * v;
        h.E[c] = kPInf / (kGam - 1.0) + 0.5 * kRhoInf * (u * u + v * v);
    }
    return h;
}

// Copy row j of every field of `src` into `dst` (nz == 1: one contiguous row).
void copyRow(const Grid& g, const gpu::DeviceCompState& src,
             gpu::DeviceCompState& dst, int j)
{
    const std::size_t off = std::size_t(j) * g.nx, bytes = g.nx * sizeof(gpu::Real);
    const gpu::DeviceScalar* s[5] = {&src.rho, &src.mx, &src.my, &src.mz, &src.E};
    gpu::DeviceScalar* d[5] = {&dst.rho, &dst.mx, &dst.my, &dst.mz, &dst.E};
    for (int f = 0; f < 5; ++f)
        (void)hipMemcpy(d[f]->data() + off, s[f]->data() + off, bytes,
                        hipMemcpyDeviceToDevice);
}

struct Result
{
    CompState s;
    double fx = 0, fy = 0;
    bool ok = false;
};

enum class Far { Dirichlet, Characteristic };

// Impulsive start from a uniform freestream at (u, v), kSteps at fixed dt.
// Dirichlet: class wall row + freestream ghost far row. Characteristic: the
// class's own far row with uInf = u (v must be 0).
Result run(const Case& cs, double u, double v, Far far, double dt)
{
    const Grid& g = cs.g;
    Result r;
    auto dev = gpu::DeviceComp::make(cs.m, Transport::air(0.01, kTInf, 300.0,
                                                          0.72, kGam, kRgas,
                                                          false));
    auto ds = gpu::DeviceCompState::make(g.size());
    auto fs = gpu::DeviceCompState::make(g.size());
    auto bc = gpu::DeviceOGridBc::make(
        g, {.gamma = kGam, .Rgas = kRgas,
            .thermal = gpu::OGridThermal::Adiabatic, .Twall = kTInf,
            .rhoInf = kRhoInf, .uInf = far == Far::Characteristic ? u : 0.0,
            .pInf = kPInf});
    if (!dev || !ds || !fs || !bc) { check(false, "run alloc"); return r; }
    r.s = uniform(g, u, v);
    (void)ds->upload(r.s);
    (void)fs->upload(r.s);
    auto hook = [&](gpu::DeviceCompState& s) {
        bc->apply(s);
        if (far == Far::Dirichlet) copyRow(g, *fs, s, g.ny - 1);
    };
    dev->setStageHook(hook);
    hook(*ds);
    for (int it = 0; it < kSteps; ++it) dev->step(*ds, gpu::Real(dt));
    hook(*ds);
    const auto F = bc->surface(*dev, *ds);
    (void)ds->download(r.s);
    r.fx = F.fx;
    r.fy = F.fy;
    r.ok = true;
    return r;
}

// max over cells of |B(i) - Rot A(i + shift)| in velocity (over U) and
// pressure (over the dynamic pressure); rot = quarter turns, CCW.
double rotatedDiff(const Grid& g, const CompState& A, const CompState& B,
                   int quarter)
{
    const int shift = quarter * g.nx / 4;
    double e = 0.0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t a = g.idx((i + shift) % g.nx, j, 0);
            const std::size_t b = g.idx(i, j, 0);
            double ua = A.mx[a] / A.rho[a], va = A.my[a] / A.rho[a];
            for (int q = 0; q < quarter; ++q) {
                const double t = ua;
                ua = -va;
                va = t;
            }
            const double ub = B.mx[b] / B.rho[b], vb = B.my[b] / B.rho[b];
            auto pOf = [](const CompState& s, std::size_t c) {
                const double ke = 0.5 * (s.mx[c] * s.mx[c] + s.my[c] * s.my[c])
                                  / s.rho[c];
                return (kGam - 1.0) * (s.E[c] - ke);
            };
            e = std::max({e, std::abs(ub - ua) / kUInf,
                          std::abs(vb - va) / kUInf,
                          std::abs(pOf(B, b) - pOf(A, a))
                              / (0.5 * kRhoInf * kUInf * kUInf)});
        }
    return e;
}

void rotation90(const Case& cs, double dt)
{
    std::printf("--- 0 vs 90 deg freestream (Dirichlet far row) ---\n");
    const Result A = run(cs, kUInf, 0.0, Far::Dirichlet, dt);
    const Result B = run(cs, 0.0, kUInf, Far::Dirichlet, dt);
    if (!A.ok || !B.ok) return;
    const double Fm = std::hypot(A.fx, A.fy);
    const double eF = std::hypot(B.fx + A.fy, B.fy - A.fx) / Fm;
    const double eS = rotatedDiff(cs.g, A.s, B.s, 1);
    std::printf("F(0) = (%.6e, %.6e)  F(90) = (%.6e, %.6e)  |F90 - R90 F0|/|F0| "
                "%.3e  state diff %.3e\n",
                A.fx, A.fy, B.fx, B.fy, eF, eS);
    check(A.fx > 0.0, "0 deg: the impulsive start drags the body downstream");
    check(std::abs(A.fy) < 1e-4 * Fm,
          "0 deg: no lift on the x-symmetric mesh (mirror symmetry)");
    check(eF < 1e-4, "90 deg: (fx, fy) -> (-fy, fx) to 1e-4");
    check(eS < 1e-4, "90 deg: state = R90 state(0) shifted by nx/4 to 1e-4");
}

void rotation180(const Case& cs, double dt)
{
    std::printf("--- 0 vs 180 deg freestream (characteristic far row) ---\n");
    const Result A = run(cs, kUInf, 0.0, Far::Characteristic, dt);
    const Result C = run(cs, -kUInf, 0.0, Far::Characteristic, dt);
    if (!A.ok || !C.ok) return;
    const double Fm = std::hypot(A.fx, A.fy);
    const double eF = std::hypot(C.fx + A.fx, C.fy + A.fy) / Fm;
    const double eS = rotatedDiff(cs.g, A.s, C.s, 2);
    std::printf("F(0) = (%.6e, %.6e)  F(180) = (%.6e, %.6e)  |F180 + F0|/|F0| "
                "%.3e  state diff %.3e\n",
                A.fx, A.fy, C.fx, C.fy, eF, eS);
    check(eF < 1e-4, "180 deg: F -> -F to 1e-4 through the characteristic row");
    check(eS < 1e-4,
          "180 deg: state = R180 state(0) shifted by nx/2 to 1e-4");
}

// Uniform freestream at 37 deg, both ghost rows held at the freestream.
void freestream37(const Case& cs, double dt)
{
    std::printf("--- uniform freestream at 37 deg ---\n");
    const Grid& g = cs.g;
    const double a = 37.0 * pi / 180.0, u = kUInf * std::cos(a),
                 v = kUInf * std::sin(a);
    auto dev = gpu::DeviceComp::make(cs.m, Transport::air(0.01, kTInf, 300.0,
                                                          0.72, kGam, kRgas,
                                                          false));
    auto ds = gpu::DeviceCompState::make(g.size());
    auto fs = gpu::DeviceCompState::make(g.size());
    if (!dev || !ds || !fs) { check(false, "37 deg alloc"); return; }
    auto h = uniform(g, u, v);
    (void)ds->upload(h);
    (void)fs->upload(h);
    dev->setStageHook([&](gpu::DeviceCompState& s) {
        copyRow(g, *fs, s, 0);
        copyRow(g, *fs, s, g.ny - 1);
    });
    for (int it = 0; it < kSteps; ++it) dev->step(*ds, gpu::Real(dt));
    (void)ds->download(h);
    double e = 0.0;
    for (int j = 1; j < g.ny - 1; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double ke = 0.5 * (h.mx[c] * h.mx[c] + h.my[c] * h.my[c])
                              / h.rho[c];
            const double p = (kGam - 1.0) * (h.E[c] - ke);
            e = std::max({e, std::abs(h.rho[c] - kRhoInf) / kRhoInf,
                          std::abs(h.mx[c] / h.rho[c] - u) / kUInf,
                          std::abs(h.my[c] / h.rho[c] - v) / kUInf,
                          std::abs(p - kPInf) / kPInf});
        }
    std::printf("max rel deviation over the fluid rows after %d steps: %.3e\n",
                kSteps, e);
    check(e < 1e-5, "37 deg freestream is a fixed point to FP32 rounding");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    static_assert(kNx % 4 == 0, "the 90 deg map needs nx divisible by 4");
    const auto cs = makeCase(2.0);
    if (!cs) { check(false, "O-grid builds: " + cs.error()); return dnstest::finish(); }
    // One dt for every run, so no rotation-dependent rounding in stableDt can
    // enter the comparison.
    double dt = 0.0;
    {
        auto dev = gpu::DeviceComp::make(cs->m, Transport::air(0.01, kTInf, 300.0,
                                                               0.72, kGam, kRgas,
                                                               false));
        auto ds = gpu::DeviceCompState::make(cs->g.size());
        auto red = gpu::ReduceScratch::make();
        if (!dev || !ds || !red) { check(false, "dt alloc"); return dnstest::finish(); }
        (void)ds->upload(uniform(cs->g, kUInf, 0.0));
        dt = 0.5 * dev->stableDt(*ds, *red, 0.4);
    }
    rotation90(*cs, dt);
    rotation180(*cs, dt);
    freestream37(*cs, dt);
    return dnstest::finish();
}
