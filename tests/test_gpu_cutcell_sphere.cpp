// GPU cut-cell on a sphere (arbitrary cuts, sanitized small cells):
//  1) apertured divergence parity vs the CPU oracle.
//  2) the FFT-preconditioned CG still converges in few iterations -- the
//     small-cell sanitization must keep the operator well conditioned.
//  3) projectCut drives the apertured divergence to the FP32 floor.
#include "gpu_test_util.hpp"

#include "dns/cutcell.hpp"
#include "dns/gpu/cutcell.hpp"

#include <cstdio>
#include <numbers>
#include <vector>

using namespace dns;

int main()
{
    using std::numbers::pi;
    test::requireDeviceOrSkip();

    const int n = 32;
    const double L = 2 * pi;
    const auto g = Grid::make(n, n, n, L, L, L);
    const double h = g.hx;

    // Off-node centre: tangencies generate arbitrarily small cut cells.
    const Sphere sph{L / 2 + 0.137 * h, L / 2 - 0.293 * h, L / 2 + 0.051 * h, L / 5};
    auto cut = CutGeometry::make(g, sph);
    const auto st = sanitizeCut(g, cut);
    std::printf("sanitize: cells=%zu faces=%zu pockets=%zu minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed, st.minAperture);

    auto U = test::randomField(g, 42);
    maskSolidFaces(g, cut, U);

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto dU = gpu::DeviceField::make(g.size());
    auto dDiv = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    if (!dCut || !dU || !dDiv || !red || !scr)
    {
        std::fprintf(stderr, "FAIL: GPU alloc\n");
        return 1;
    }
    auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, 1e-7);
    if (!poisson)
    {
        std::fprintf(stderr, "FAIL: solver make: %s\n",
                     hipGetErrorString(poisson.error()));
        return 1;
    }

    (void)gpu::uploadField(U, *dU);
    gpu::maskSolidFaces(gg, *dCut, *dU);

    bool ok = true;

    // --- 1) apertured divergence parity ---
    {
        Scalar divRef = zeros(g);
        divergenceCut(g, cut, U, divRef);
        gpu::divergenceCut(gg, *dCut, *dU, *dDiv, gpu::Real(1));
        std::vector<gpu::Real> divGpu(g.size());
        (void)dDiv->download(divGpu.data());
        double err = 0.0, scale = 0.0;
        for (std::size_t c = 0; c < g.size(); ++c)
        {
            err = std::max(err, std::abs(double(divGpu[c]) - divRef[c]));
            scale = std::max(scale, std::abs(divRef[c]));
        }
        ok = test::checkClose("divergenceCut", err, scale, 1e-5) && ok;
    }

    // --- 2) pressure solve converges and satisfies M(p) = rhs (see
    // test_gpu_cutcell for why the residual, not p itself, is the target) ---
    const double dt = 0.01;
    {
        Scalar rhs = zeros(g);
        divergenceCut(g, cut, U, rhs);
        for (auto& x : rhs) x /= dt;

        std::vector<gpu::Real> tmp(rhs.begin(), rhs.end());
        (void)dDiv->upload(tmp.data());
        const int it = poisson->solve(*dDiv, scr->phi);
        std::printf("pcg iterations: %d\n", it);
        std::vector<gpu::Real> pTmp(g.size());
        (void)scr->phi.download(pTmp.data());
        Scalar p(pTmp.begin(), pTmp.end());

        auto Ug = U;
        subtractGradientCut(g, cut, p, dt, Ug);
        const double resid = maxAbsDivergenceCut(g, cut, Ug);
        const double rhsScale = maxAbsDivergenceCut(g, cut, U);
        ok = test::checkClose("cut Mp=rhs resid", resid, rhsScale, 1e-4) && ok;
        if (it >= 150)
        {
            std::printf("FAIL: preconditioner ineffective (%d iterations)\n", it);
            ok = false;
        }
    }

    // --- 3) full projection to the FP32 floor ---
    {
        gpu::projectCut(gg, *dCut, *poisson, gpu::Real(dt), *dU, *scr);
        const double e = gpu::maxAbsDivergenceCut(gg, *dCut, *dU, *red);
        std::printf("max |apertured div| after projectCut: %.3e\n", e);
        if (e > 1e-4)
        {
            std::printf("FAIL: divergence not removed\n");
            ok = false;
        }
    }

    return ok ? 0 : 1;
}
