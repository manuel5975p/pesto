// GPU cut-cell parity vs the CPU oracle on the node-aligned rotated cube:
//  1) apertured divergence + solid-face masking match the CPU operators.
//  2) the FFT-preconditioned CG pressure matches the CPU CG pressure.
//  3) projectCut drives the apertured divergence to ~0 (FP32 floor) in few
//     iterations (the preconditioner must actually work).
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

    // Node-aligned geometry: every cut cell exactly halved (see test_cutcell).
    const RotatedCube cube{16 * h, 16 * h, 8 * h, 10 * h, 22 * h};
    const auto cut = CutGeometry::make(g, cube);

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

    // --- 2) the GPU pressure satisfies the discrete equation M(p) = rhs ---
    // Absolute pressure is not a valid parity target: M has a body-localized
    // near-null mode invisible to the residual and a constant null mode, so a
    // residual-stopped PCG and a deep CPU solve differ by those modes. The
    // physical check is that grad(p) actually balances rhs -> small residual.
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

        // Residual r = M(p) - rhs on fluid cells, relative to ||rhs||_inf.
        // M(p) = -apply where CutPoissonSolver::apply returns -M; reuse the CPU
        // apertured operator by measuring the divergence of dt*grad(p) instead:
        // divCut(dt grad p) should equal dt*rhs = divCut(U) on fluid.
        auto Ug = U;
        subtractGradientCut(g, cut, p, dt, Ug); // Ug = U - dt grad p
        const double resid = maxAbsDivergenceCut(g, cut, Ug);
        const double rhsScale = maxAbsDivergenceCut(g, cut, U);
        ok = test::checkClose("cut Mp=rhs resid", resid, rhsScale, 1e-4) && ok;
        if (it >= 100)
        {
            std::printf("FAIL: preconditioner ineffective (%d iterations)\n", it);
            ok = false;
        }
    }

    // --- 3) full projection drives the apertured divergence to the FP32 floor ---
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
