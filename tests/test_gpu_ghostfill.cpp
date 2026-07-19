// Ghost-fill (sharp viscous wall) checks:
//  1) Mirror reconstruction: seed every face with q = sdf(face center) -- the
//     linear-in-wall-distance profile that vanishes on the true boundary.
//     After maskSolidFaces + ghost fill, each ghost face must recover ~sdf
//     (negative inside); a sign/index bug produces O(field) errors, the
//     correct fill is off by interpolation error O(h^2/r) except where the
//     image stencil loses corners (O(h) worst case). Also: open faces must be
//     untouched, and w must stay 0-consistent in native 2D.
//  2) Energy: the energy-stability scenario (off-node sphere, random projected
//     field, 300 RK3 + cut projection steps) using the split-stage form
//     (convection reads the masked state, diffusion reads the ghost-filled
//     copy). Inviscid: K bounded as in test_gpu_energy_stability -- the ghost
//     values must never leak into the state. Viscous: the ghost wall must
//     dissipate (K falls, still bounded above).
#include "gpu_test_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>

using namespace dns;
using std::numbers::pi;

namespace {

// Fill every face of every component with sdf(face center).
VectorField sdfField(const Grid& g, const gpu::Sdf& sdf)
{
    auto U = VectorField::zeros(g);
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto c = g.idx(i, j, k);
                U.u[c] = sdf(i * g.hx, (j + 0.5) * g.hy, (k + 0.5) * g.hz);
                U.v[c] = sdf((i + 0.5) * g.hx, j * g.hy, (k + 0.5) * g.hz);
                U.w[c] = sdf((i + 0.5) * g.hx, (j + 0.5) * g.hy, k * g.hz);
            }
    return U;
}

// Reconstruction check for one body. Returns false on failure.
bool checkReconstruction(const char* name, const Grid& g, CutGeometry cut,
                         const gpu::Sdf& sdf)
{
    (void)sanitizeCut(g, cut);
    auto ghost = gpu::DeviceGhostFill::make(g, cut, sdf);
    auto dU = gpu::DeviceField::make(g.size());
    auto dG = gpu::DeviceField::make(g.size());
    if (!ghost || !dU || !dG)
    {
        std::fprintf(stderr, "FAIL: GPU alloc\n");
        return false;
    }
    std::printf("%s: %zu ghost faces\n", name, ghost->count());
    if (ghost->count() == 0)
    {
        std::printf("FAIL: no ghost faces built\n");
        return false;
    }

    auto U = sdfField(g, sdf);
    auto ref = U;
    maskSolidFaces(g, cut, ref); // the state the stage field is in
    (void)gpu::uploadField(ref, *dU);
    if (ghost->fillFrom(*dU, *dG) != hipSuccess)
    {
        std::fprintf(stderr, "FAIL: fillFrom\n");
        return false;
    }
    auto out = VectorField::zeros(g);
    (void)gpu::downloadField(*dG, out);

    const double tol = CutGeometry::openTol;
    const double h = g.hx;
    double maxErr = 0.0, sumErr = 0.0, maxOpenDrift = 0.0;
    std::size_t nGhost = 0;
    auto scan = [&](const Scalar& a, const Scalar& q, const Scalar& exact,
                    const Scalar& masked) {
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            if (a[c] > tol)
            {
                maxOpenDrift = std::max(maxOpenDrift,
                                        std::abs(q[c] - masked[c]));
                continue;
            }
            if (q[c] == 0.0) continue; // deep solid: stayed masked
            const double e = std::abs(q[c] - exact[c]);
            maxErr = std::max(maxErr, e);
            sumErr += e;
            ++nGhost;
        }
    };
    scan(cut.ax, out.u, U.u, ref.u);
    scan(cut.ay, out.v, U.v, ref.v);
    scan(cut.az, out.w, U.w, ref.w);

    const double meanErr = nGhost ? sumErr / double(nGhost) : 0.0;
    std::printf("%s: filled=%zu maxErr/h=%.3f meanErr/h=%.3f openDrift=%.2e\n",
                name, nGhost, maxErr / h, meanErr / h, maxOpenDrift);
    bool ok = true;
    ok &= test::checkClose("ghost max", maxErr, h, 0.8);
    ok &= test::checkClose("ghost mean", meanErr, h, 0.1);
    ok &= test::checkClose("open faces", maxOpenDrift, 1.0, 1e-6);
    return ok;
}

// Energy behavior of the split stage (conv: masked state, diff: ghost copy).
// nu = 0: K must stay bounded (<= 1.02 K0) -- ghost values must not leak.
// nu > 0: additionally K must fall (the ghost wall dissipates).
bool energyRun(double nu)
{
    const int n = 64;
    const double L = 2 * pi;
    const auto g = Grid::make(n, n, n, L, L, L);
    const double h = g.hx;
    const Sphere s{L / 2 + 0.137 * h, L / 2 - 0.293 * h, L / 2 + 0.051 * h, L / 5};
    const gpu::Sdf sdf = [&](double x, double y, double z) {
        const double dx = x - s.cx, dy = y - s.cy, dz = z - s.cz;
        return std::sqrt(dx * dx + dy * dy + dz * dz) - s.r;
    };
    auto cut = CutGeometry::make(g, s);
    (void)sanitizeCut(g, cut);

    auto U = VectorField::zeros(g);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        U.u[c] = d(rng); U.v[c] = d(rng); U.w[c] = d(rng);
    }
    maskSolidFaces(g, cut, U);

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto ghost = gpu::DeviceGhostFill::make(g, cut, sdf);
    auto dU = gpu::DeviceField::make(g.size());
    auto dU0 = gpu::DeviceField::make(g.size());
    auto dS = gpu::DeviceField::make(g.size());
    auto dG = gpu::DeviceField::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    if (!dCut || !ghost || !dU || !dU0 || !dS || !dG || !red || !scr)
    {
        std::fprintf(stderr, "FAIL: GPU alloc\n");
        return false;
    }
    auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, 1e-5);
    if (!poisson)
    {
        std::fprintf(stderr, "FAIL: solver make\n");
        return false;
    }
    (void)gpu::uploadField(U, *dU);
    gpu::maskSolidFaces(gg, *dCut, *dU);
    (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(gpu::Real));
    gpu::projectCut(gg, *dCut, *poisson, gpu::Real(1), *dU, *scr);

    const double K0 = gpu::kineticEnergy(gg, *dU, *red);
    std::printf("nu=%g: K0 = %.6f (ghost faces: %zu)\n", nu, K0, ghost->count());

    auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double a,
                     double dt) {
        (void)ghost->fillFrom(in, *dG);
        gpu::fusedRhsStage(gg, gpu::Real(nu), gpu::Real(0), 0, 0, 0, gpu::Real(a),
                           *dU0, gpu::Real(1.0 - a), in, *dG, gpu::Real(dt), out);
        gpu::maskSolidFaces(gg, *dCut, out);
        gpu::projectCut(gg, *dCut, *poisson, gpu::Real(dt), out, *scr);
    };

    const int steps = 300;
    double kMax = K0, kEnd = K0;
    for (int step = 0; step < steps; ++step)
    {
        const double conv = gpu::maxCourantPerDt(gg, *dU, *red);
        const double dt = std::min(0.5 / conv, 0.2 * h * h / std::max(nu, 1e-30));
        std::swap(*dU0, *dU);
        stage(*dU0, *dU, 0.0, dt);
        stage(*dU, *dS, 0.75, dt);
        stage(*dS, *dU, 1.0 / 3.0, dt);
        if (step % 50 == 0 || step == steps - 1)
        {
            const double K = gpu::kineticEnergy(gg, *dU, *red);
            kMax = std::max(kMax, K);
            kEnd = K;
            std::printf("step %3d K=%.6f K/K0=%.6f\n", step, K, K / K0);
            if (!std::isfinite(K))
            {
                std::printf("FAIL: energy is not finite\n");
                return false;
            }
        }
    }
    if (kMax > K0 * 1.02)
    {
        std::printf("FAIL: energy grew (Kmax/K0 = %.4f)\n", kMax / K0);
        return false;
    }
    if (nu > 0 && kEnd >= K0)
    {
        std::printf("FAIL: viscous run did not dissipate (Kend/K0 = %.4f)\n",
                    kEnd / K0);
        return false;
    }
    std::printf("nu=%g energy ok: Kmax/K0 = %.6f Kend/K0 = %.6f\n", nu, kMax / K0,
                kEnd / K0);
    return true;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    bool ok = true;

    { // 2D cylinder, native nz == 1, off-node center
        const int n = 64;
        const double L = 2 * pi;
        const auto g = Grid::make(n, n, 1, L, L, L / n);
        const double h = g.hx;
        const CylinderZ c{L / 2 + 0.31 * h, L / 2 + 0.17 * h, L / 6};
        const gpu::Sdf sdf = [&](double x, double y, double) {
            return std::hypot(x - c.cx, y - c.cy) - c.r;
        };
        ok &= checkReconstruction("cylinder2d", g, CutGeometry::make(g, c), sdf);
    }
    { // 3D sphere, off-node
        const int n = 48;
        const double L = 2 * pi;
        const auto g = Grid::make(n, n, n, L, L, L);
        const double h = g.hx;
        const Sphere s{L / 2 + 0.137 * h, L / 2 - 0.293 * h, L / 2 + 0.051 * h,
                       L / 6};
        const gpu::Sdf sdf = [&](double x, double y, double z) {
            const double dx = x - s.cx, dy = y - s.cy, dz = z - s.cz;
            return std::sqrt(dx * dx + dy * dy + dz * dz) - s.r;
        };
        ok &= checkReconstruction("sphere3d", g, CutGeometry::make(g, s), sdf);
    }
    ok &= energyRun(0.0);
    ok &= energyRun(0.05);

    if (!ok) return 1;
    std::printf("all ghost-fill checks passed\n");
    return 0;
}
