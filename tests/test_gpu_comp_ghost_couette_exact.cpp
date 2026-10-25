// Plane Couette flow through the compressible ghost wall: an EXACT steady
// solution the moving-wall image reproduces to round-off.
//
// A floor (solid y < y0) moves at U through a constant wall velocity, a ceiling
// (solid y > y1) rests, both off the lattice. u = U (y1 - y)/H, v = 0, uniform
// rho and p is a steady solution of the momentum equations (viscous heating is
// uniform, so p rises uniformly and drives nothing). The moving image
// u_ghost = (1+a) U_w - a u_sample extends a linear profile EXACTLY (trilinear
// sampling of a linear field is exact), so the discrete viscous flux is the
// same constant mu U/H on every y face, the momentum rhs vanishes, and the
// wall shear the tally reads is exact:
//   floor   fx = -mu U Lx hz / H   (the fluid drags the moving floor back)
//   ceiling fx = +mu U Lx hz / H
// The two walls share one ghost object, so the tally returns their SUM (0);
// the floor is separated by the moment about a pivot ON the ceiling face row:
// qz = -sum (y_face - y_c) fx_face = -(y_f - y_c) F_floor.
// Run on the uniform mesh (kBodyFlux) and on a focal-y mesh whose ratio-2
// plateau holds the gap (kBodyFluxCurv).
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/comp_ghost.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

using namespace dns;

namespace {

using dnstest::check;

constexpr double gam = 1.4, rhoInf = 1.0, U = 1.0, mu = 0.01;
const double pInf = 1.0 / (gam * 0.04); // Mach 0.2 on U

void couette(const std::string& tag, bool focal)
{
    std::printf("--- %s ---\n", tag.c_str());
    const int nx = 32, ny = 64;
    const double Lx = 1.0, Ly = 2.0;
    const auto g = Grid::make(nx, ny, 1, Lx, Ly, 1.0);
    const double h = g.hy;
    const double y0 = 0.75 + 0.3 * h, y1 = 1.25 + 0.37 * h, H = y1 - y0;
    const AxisRemap3 R{
        Focal1D::identity(Lx),
        focal ? Focal1D::fromPhysical(Ly, y0 - 0.1, y1 + 0.1, 0.15, 2.0)
              : Focal1D::identity(Ly),
        Focal1D::identity(1.0)};
    if (focal)
        check(R.fy.active && R.fy.clears(4 * h), tag + ": plateau clears");
    const gpu::GhostSdf sdf = [=](double, double y, double) {
        return std::min(y - y0, y1 - y);
    };
    // Uniform translation: no wall acceleration.
    const gpu::GhostWallMotion uw = [=](double, double y, double) {
        return gpu::GhostWallKinematics{{y < 0.5 * (y0 + y1) ? U : 0.0, 0.0, 0.0},
                                        {0.0, 0.0, 0.0}};
    };

    std::optional<CurvMetrics> m;
    if (focal) {
        auto mm = CurvMetrics::build(CurvMesh::fromRemap(g, R));
        if (!mm) { check(false, tag + ": metrics " + mm.error()); return; }
        m = std::move(*mm);
    }
    auto dev = focal ? gpu::DeviceComp::make(*m, gam, mu)
                     : gpu::DeviceComp::make(g, gam, mu);
    auto gh = gpu::DeviceCompGhost::make(g, sdf, gam, rhoInf, pInf, nullptr, R,
                                         {}, &uw);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dev || !gh || !ds || !red) { check(false, tag + ": setup"); return; }
    check(gh->curvilinear() == focal, tag + ": tally path as intended");
    dev->setStageHook([&](gpu::DeviceCompState& s) { gh->apply(s); });

    // The two face rows (identical tables to the ones the ghost uploaded).
    const auto T = gpu::classifyCompGhost(g, sdf, nullptr, R, &uw);
    double yf = 1e300, yc = -1e300;
    for (int f = 0; f < T.nFace(); ++f) {
        const double y = T.fpos[3 * std::size_t(f) + 1];
        yf = std::min(yf, y);
        yc = std::max(yc, y);
    }

    std::vector<double> yc_(ny);
    std::vector<char> fluid(ny);
    for (int j = 0; j < ny; ++j) {
        yc_[j] = R.fy.map((j + 0.5) * g.hy);
        fluid[j] = sdf(0.0, yc_[j], 0.0) > 0.0;
    }
    auto host = CompState::zeros(g);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double u = U * (y1 - yc_[j]) / H;
            host.rho[c] = rhoInf;
            host.mx[c] = rhoInf * u;
            host.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * u * u;
        }
    (void)ds->upload(host);
    gh->apply(*ds);

    auto residual = [&](double& du, double& dv) {
        auto q = CompState::zeros(g);
        (void)ds->download(q);
        du = dv = 0.0;
        for (int j = 0; j < ny; ++j) {
            if (!fluid[j]) continue;
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                du = std::max(du, std::abs(q.mx[c] / q.rho[c]
                                           - U * (y1 - yc_[j]) / H) / U);
                dv = std::max(dv, std::abs(q.my[c] / q.rho[c]) / U);
            }
        }
    };

    // The tally on the exact field, before any step.
    const double fExact = mu * U * Lx * g.hz / H;
    auto tally = [&](const char* when) {
        const auto F = gh->force(*dev, *ds, {0.5 * Lx, yc, 0.0});
        const double ff = -F.qz / (yf - yc);
        const double fc = F.fx - ff;
        std::printf("%s: floor fx %.8e (exact %.8e, rel %.2e) ceiling fx %.8e "
                    "(rel %.2e) net %.2e\n",
                    when, ff, -fExact, std::abs(ff + fExact) / fExact, fc,
                    std::abs(fc - fExact) / fExact, F.fx);
        check(std::abs(ff + fExact) < 1e-5 * fExact,
              tag + " " + when + ": floor shear = -mu U Lx/H to 1e-5");
        check(std::abs(fc - fExact) < 1e-5 * fExact,
              tag + " " + when + ": ceiling shear = +mu U Lx/H to 1e-5");
    };
    tally("t=0");

    const double dt = dev->stableDt(*ds, *red);
    dev->step(*ds, gpu::Real(dt));
    gh->apply(*ds);
    double du = 0, dv = 0;
    residual(du, dv);
    std::printf("after 1 step (dt %.3e): max|u - u_exact|/U %.2e max|v|/U %.2e\n",
                dt, du, dv);
    check(du < 1e-6 && dv < 1e-6,
          tag + ": linear Couette profile is a fixed point after 1 step (1e-6)");
    tally("1 step");

    for (int s = 0; s < 9; ++s) dev->step(*ds, gpu::Real(dt));
    gh->apply(*ds);
    residual(du, dv);
    std::printf("after 10 steps: max|u - u_exact|/U %.2e max|v|/U %.2e\n", du, dv);
    check(du < 1e-6 && dv < 1e-6,
          tag + ": profile stays fixed over 10 steps (1e-6)");
}

// Wall shear of the exact linear profile vs the ceiling's distance below the
// nearest ghost centre, dw = delta h, no time stepping. The image samples at
// dsmp = max(dw, 0.75 h) into the fluid; when delta < 1/4 that point lies
// between the wall and the first fluid centre, i.e. inside the trilinear cell
// whose other corner is the ghost ITSELF (weight zeroed, renormalized away).
// The sample then silently reads the first fluid centre at (1 - delta) h while
// a = delta/0.75 still assumes 0.75 h, so the imaged face gradient is
//   (1 - delta)(1 + delta/0.75) G    instead of G
// -- an O(1) shear error (max +2.08% at delta = 1/8) that does not shrink with
// h. The exact answer is G for every delta; the prediction is printed beside
// the measurement so a failure says which mechanism it is.
void offsetSweep()
{
    std::printf("--- ceiling offset sweep (Cartesian, t = 0 tally) ---\n");
    const int nx = 32, ny = 64;
    const double Lx = 1.0, Ly = 2.0;
    const auto g = Grid::make(nx, ny, 1, Lx, Ly, 1.0);
    const double h = g.hy, y0 = 24.3 * h;
    auto dev = gpu::DeviceComp::make(g, gam, mu);
    auto ds = gpu::DeviceCompState::make(g.size());
    if (!dev || !ds) { check(false, "sweep setup"); return; }
    for (const double delta : {0.05, 0.125, 0.2, 0.3, 0.5, 0.8}) {
        const double y1 = (40.5 - delta) * h, H = y1 - y0;
        const gpu::GhostSdf sdf = [=](double, double y, double) {
            return std::min(y - y0, y1 - y);
        };
        const gpu::GhostWallMotion uw = [=](double, double y, double) {
            return gpu::GhostWallKinematics{
                {y < 0.5 * (y0 + y1) ? U : 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        };
        auto gh = gpu::DeviceCompGhost::make(g, sdf, gam, rhoInf, pInf, nullptr,
                                             {}, &uw);
        if (!gh) { check(false, "sweep ghost"); return; }
        auto host = CompState::zeros(g);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t c = g.idx(i, j, 0);
                const double u = U * (y1 - (j + 0.5) * h) / H;
                host.rho[c] = rhoInf;
                host.mx[c] = rhoInf * u;
                host.E[c] = pInf / (gam - 1.0) + 0.5 * rhoInf * u * u;
            }
        (void)ds->upload(host);
        gh->apply(*ds);
        // Ceiling faces are at y = 40 h; floor faces at 24 h.
        const double yf = 24.0 * h, yc = 40.0 * h;
        const auto F = gh->force(*dev, *ds, {0.5 * Lx, yc, 0.0});
        const double ff = -F.qz / (yf - yc), fc = F.fx - ff;
        const double fExact = mu * U * Lx / H;
        const double pred = delta < 0.25 ? (1 - delta) * (1 + delta / 0.75) : 1.0;
        char msg[160];
        std::snprintf(msg, sizeof msg,
                      "ceiling dw = %.3f h: wall shear exact to 1e-5 "
                      "(measured/exact %.6f, mechanism predicts %.6f)",
                      delta, fc / fExact, pred);
        std::printf("  floor/exact %.7f  %s\n", -ff / fExact, msg);
        check(std::abs(fc / fExact - 1.0) < 1e-5, msg);
    }
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    couette("Cartesian", false);
    couette("focal-y", true);
    offsetSweep();
    return dnstest::finish();
}
