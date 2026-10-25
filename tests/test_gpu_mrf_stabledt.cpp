// The acoustic dt bound sees the RELATIVE velocity inside the rotating-frame
// zone (--rotor-frame mrf).
//
// In the zone the convective eigenvalues are u_rel +- c with
// u_rel = u - f omega x (p - c0), so for gas AT REST the bound must be set by
// |f omega x r| + c, not by c alone: the absolute form under-counts by the
// frame speed exactly where a rotor mesh is finest. Expectation, evaluated
// cell by cell on the host from the zone parameters (not read back from the
// solver):
//   uniform mesh  : wmax = max_cells max_axis (|u_rel,a| + c) / h_a
//   focal mesh    : wmax = max_cells (|u_rel| + c) * max_axis aov_a(cell)
// and dt = cfl / wmax, so dt_mrf / dt_inertial is the ratio of the two maxima.
// The uniform path is also held to the plain magnitude form
// max (|f omega x r| + c) / h to 1e-3 (the per-axis bound may only be LESS
// restrictive than it, and by no more than the grid's angular sampling).
// mu = 0, so the viscous limit cannot mask the acoustic one.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/remap.hpp"

#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
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

// Host twin of the zone's blend and frame velocity (comp_flux.hpp mrfRamp /
// mrfFrameVel), double precision. z shaft, degenerate axial blend.
double ramp(double q, double q1, double q2)
{
    if (q <= q1) return 1.0;
    if (q >= q2) return 0.0;
    const double s = (q - q1) / (q2 - q1);
    return 1.0 - s * s * (3.0 - 2.0 * s);
}

std::array<double, 2> frameVel(const gpu::MrfZone& z, double x, double y)
{
    const double dx = x - z.center[0], dy = y - z.center[1];
    const double f = ramp(std::hypot(dx, dy), z.r1, z.r2);
    return {-f * z.omega * dy, f * z.omega * dx};
}

CompState rest(const Grid& g)
{
    auto s = CompState::zeros(g);
    for (std::size_t c = 0; c < g.size(); ++c) {
        s.rho[c] = 1.0;
        s.E[c] = (1.0 / kGam) / (kGam - 1.0); // p = 1/gamma, c = 1
    }
    return s;
}

// Uniform 2D mesh, zone tip speed 0.5 c at r1.
void uniformMesh()
{
    const int n = 128;
    const double L = 2 * pi, cs = 1.0;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    gpu::MrfZone z;
    z.axis = 2;
    z.center[0] = 0.5 * L; z.center[1] = 0.5 * L; z.center[2] = 0.5;
    z.r1 = 1.0; z.r2 = 1.6;
    z.a1 = 1e30; z.a2 = 2e30;
    z.omega = 0.5 * cs / z.r1;

    auto inert = gpu::DeviceComp::make(g, kGam, 0.0);
    auto mrf = gpu::DeviceComp::make(g, kGam, 0.0);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!inert || !mrf || !ds || !red) { check(false, "uniform alloc"); return; }
    const AxisRemap3 I{Focal1D::identity(L), Focal1D::identity(L),
                       Focal1D::identity(1.0)};
    check(mrf->setMrf(z, I) == hipSuccess, "uniform: setMrf");
    (void)ds->upload(rest(g));
    const double dtI = inert->stableDt(*ds, *red);
    const double dtM = mrf->stableDt(*ds, *red);

    double wAxis = 0.0, wMag = 0.0, tip = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const auto w = frameVel(z, (i + 0.5) * g.hx, (j + 0.5) * g.hy);
            wAxis = std::max({wAxis, (std::abs(w[0]) + cs) / g.hx,
                              (std::abs(w[1]) + cs) / g.hy});
            const double sp = std::hypot(w[0], w[1]);
            tip = std::max(tip, sp);
            wMag = std::max(wMag, (sp + cs) / std::min(g.hx, g.hy));
        }
    const double base = cs / std::min(g.hx, g.hy);
    const double expAxis = base / wAxis, expMag = base / wMag;
    const double got = dtM / dtI;
    std::printf("uniform: dt inertial %.6e mrf %.6e ratio %.8f | expected "
                "per-axis %.8f magnitude %.8f | max frame speed %.4f c\n",
                dtI, dtM, got, expAxis, expMag, tip / cs);
    check(std::abs(dtI - 0.4 * std::min(g.hx, g.hy) / cs) < 1e-6 * dtI,
          "uniform: inertial dt at rest is cfl h / c");
    check(std::abs(got / expAxis - 1.0) < 1e-6,
          "uniform: dt_mrf/dt_inertial = per-axis (|u_rel,a| + c)/h form to "
          "1e-6");
    check(std::abs(got / expMag - 1.0) < 1e-3,
          "uniform: dt_mrf/dt_inertial = (|f omega x r| + c)/h form to 1e-3");
    check(got < 1.0 / 1.4, "uniform: the frame speed actually tightens dt");
}

// Focal 2D mesh, zone inside the fine plateau: the curvilinear bound.
void focalMesh()
{
    const int n = 96;
    const double L = 2 * pi, cs = 1.0;
    const auto g = Grid::make(n, n, 1, L, L, 1.0);
    const Focal1D f = Focal1D::fromPhysical(L, 0.5 * L - 1.0, 0.5 * L + 1.0,
                                            0.8, 3.0);
    const AxisRemap3 R{f, f, Focal1D::identity(1.0)};
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto m = CurvMetrics::build(mesh);
    if (!m) { check(false, "focal metrics " + m.error()); return; }
    gpu::MrfZone z;
    z.axis = 2;
    z.center[0] = 0.5 * L; z.center[1] = 0.5 * L; z.center[2] = 0.5;
    z.r1 = 0.5; z.r2 = 0.8;
    z.a1 = 1e30; z.a2 = 2e30;
    z.omega = 0.5 * cs / z.r1;

    auto inert = gpu::DeviceComp::make(*m, kGam, 0.0);
    auto mrf = gpu::DeviceComp::make(*m, kGam, 0.0);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!inert || !mrf || !ds || !red) { check(false, "focal alloc"); return; }
    check(mrf->setMrf(z, R) == hipSuccess, "focal: setMrf");
    (void)ds->upload(rest(g));
    const double dtI = inert->stableDt(*ds, *red);
    const double dtM = mrf->stableDt(*ds, *red);

    double wI = 0.0, wM = 0.0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double a = std::max(m->aov[0][c], m->aov[1][c]);
            const auto w = frameVel(z, f.map((i + 0.5) * g.hx),
                                    f.map((j + 0.5) * g.hy));
            wI = std::max(wI, cs * a);
            wM = std::max(wM, (std::hypot(w[0], w[1]) + cs) * a);
        }
    const double exp = wI / wM, got = dtM / dtI;
    std::printf("focal: h_fine %.4e h_coarse %.4e, dt inertial %.6e mrf "
                "%.6e ratio %.8f expected %.8f\n",
                f.c * f.r * g.hx, f.c * g.hx, dtI, dtM, got, exp);
    check(std::abs(dtI - 0.4 / wI) < 1e-6 * dtI,
          "focal: inertial dt at rest is cfl / max(c aov)");
    check(std::abs(got / exp - 1.0) < 1e-6,
          "focal: dt_mrf/dt_inertial = (|u_rel| + c) aov form to 1e-6");
    check(got < 1.0 / 1.4, "focal: the frame speed tightens dt on the fine "
                           "cells");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    uniformMesh();
    focalMesh();
    return dnstest::finish();
}
