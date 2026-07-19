// 2D cut-cell (nz == 1) vs the same z-invariant problem run in 3D: a cylinder
// with a z-invariant velocity field must give the same projected field on
// every slice, since nz == 1 is the exact 2D restriction of the 3D kernels.
// Covered both periodic and NeumannBox (confined box) pressure solves, plus
// the divergence-removal check on the 2D run itself.
#include "gpu_test_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/gpu/cutcell.hpp"

#include <cstdio>
#include <numbers>
#include <random>
#include <vector>

using namespace dns;

namespace {

// z-invariant random u,v (replicated over k), w = 0.
VectorField randomField2d(const Grid& g, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto U = VectorField::zeros(g);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const double u = dist(rng), v = dist(rng);
            for (int k = 0; k < g.nz; ++k)
            {
                U.u[g.idx(i, j, k)] = u;
                U.v[g.idx(i, j, k)] = v;
            }
        }
    return U;
}

// Run projectCut on grid g and return the projected field (downloaded).
bool runProjected(const Grid& g, const CutGeometry& cut, gpu::PoissonBc bc,
                  const VectorField& U0, VectorField& out, double& divAfter)
{
    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto dU = gpu::DeviceField::make(g.size());
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    auto red = gpu::ReduceScratch::make();
    if (!dCut || !dU || !scr || !red)
    {
        std::fprintf(stderr, "FAIL: GPU alloc\n");
        return false;
    }
    auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, 1e-7, bc);
    if (!poisson)
    {
        std::fprintf(stderr, "FAIL: solver make: %s\n",
                     hipGetErrorString(poisson.error()));
        return false;
    }
    (void)gpu::uploadField(U0, *dU);
    gpu::maskSolidFaces(gg, *dCut, *dU);
    const int it = gpu::projectCut(gg, *dCut, *poisson, gpu::Real(0.01), *dU, *scr);
    std::printf("nz=%d pcg iterations: %d\n", g.nz, it);
    divAfter = gpu::maxAbsDivergenceCut(gg, *dCut, *dU, *red);
    out = VectorField::zeros(g);
    return gpu::downloadField(*dU, out) == hipSuccess;
}

// max |a2d - slice k of a3d| over u,v and the |a3d| scale.
test::FieldDiff sliceDiff(const Grid& g2, const Grid& g3, const VectorField& a2,
                          const VectorField& a3, int k)
{
    test::FieldDiff d{0.0, 0.0};
    for (int j = 0; j < g2.ny; ++j)
        for (int i = 0; i < g2.nx; ++i)
        {
            const auto c2 = g2.idx(i, j, 0), c3 = g3.idx(i, j, k);
            d.maxErr = std::max({d.maxErr, std::abs(a2.u[c2] - a3.u[c3]),
                                 std::abs(a2.v[c2] - a3.v[c3])});
            d.scale = std::max({d.scale, std::abs(a3.u[c3]), std::abs(a3.v[c3])});
        }
    return d;
}

bool runCase(gpu::PoissonBc bc, const char* tag)
{
    using std::numbers::pi;
    const int n = 48, nz3 = 8;
    const double L = 2 * pi;
    const auto g2 = Grid::make(n, n, 1, L, L, 1.0);
    const auto g3 = Grid::make(n, n, nz3, L, L, L * nz3 / n);

    const CylinderZ cyl{0.5 * L, 0.45 * L, 0.15 * L};
    auto cut2 = CutGeometry::make(g2, cyl);
    auto cut3 = CutGeometry::make(g3, cyl);
    sanitizeCut(g2, cut2);
    sanitizeCut(g3, cut3);
    if (bc == gpu::PoissonBc::NeumannBox)
    {
        closeBoxBoundary(g2, cut2);
        closeBoxBoundary(g3, cut3);
    }

    const auto U2 = randomField2d(g2, 7);
    auto U3 = VectorField::zeros(g3);
    for (int k = 0; k < g3.nz; ++k)
        for (int j = 0; j < g3.ny; ++j)
            for (int i = 0; i < g3.nx; ++i)
            {
                U3.u[g3.idx(i, j, k)] = U2.u[g2.idx(i, j, 0)];
                U3.v[g3.idx(i, j, k)] = U2.v[g2.idx(i, j, 0)];
            }

    VectorField P2, P3;
    double div2 = 0.0, div3 = 0.0;
    if (!runProjected(g2, cut2, bc, U2, P2, div2)) return false;
    if (!runProjected(g3, cut3, bc, U3, P3, div3)) return false;

    bool ok = true;
    std::printf("[%s] max |apertured div| 2d: %.3e 3d: %.3e\n", tag, div2, div3);
    if (div2 > 1e-4 || div3 > 1e-4)
    {
        std::printf("FAIL: divergence not removed\n");
        ok = false;
    }
    // w must remain identically zero in both runs (z terms cancel exactly).
    double wMax = 0.0;
    for (double w : P2.w) wMax = std::max(wMax, std::abs(w));
    for (double w : P3.w) wMax = std::max(wMax, std::abs(w));
    ok = test::checkClose("w stays zero", wMax, 1.0, 1e-12) && ok;
    // Every slice of the 3D run must match the 2D run (independent solves at
    // FP32 with different FFT shapes: modest tolerance). The confined 3D box
    // is also closed in z; its ghost slices are solid, so skip them there.
    const int kLo = bc == gpu::PoissonBc::NeumannBox ? 1 : 0;
    for (int k = kLo; k < g3.nz - kLo; ++k)
    {
        const auto d = sliceDiff(g2, g3, P2, P3, k);
        char what[32];
        std::snprintf(what, sizeof what, "%s slice %d", tag, k);
        ok = test::checkClose(what, d.maxErr, d.scale, 2e-3) && ok;
    }
    return ok;
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    bool ok = runCase(gpu::PoissonBc::Periodic, "periodic");
    ok = runCase(gpu::PoissonBc::NeumannBox, "neumann") && ok;
    return ok ? 0 : 1;
}
