// --far tunnel --open: the sponge-free incompressible wind tunnel.
// Dirichlet inlet + convective outlet (unchanged) + OPEN lateral faces:
// zero-gradient normal velocity whose flux enters the projection RHS, global
// mass balance closed at the outlet, pressure Neumann/DCT throughout.
//
// Checks:
//   1. Freestream preservation, 2D: (uInf,0,0) run through the full
//      BC -> divergence -> DCT-CG -> gradient cycle stays uInf to FP32
//      round-off, with ~zero CG work (the RHS is identically balanced).
//   2. Same in 3D (y AND z faces open).
//   3. End-to-end: a short cylinder run (`dns flow --far tunnel --open`)
//      exits 0 with a healthy projection (maxDiv small) and finite forces.
//
// No HIP device -> skip (exit 77).

#include "dns/cutcell.hpp"
#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "dns/gpu/tunnel.hpp"
#include "validation_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;



// All-fluid cut geometry with the box boundary closed (the tunnel far field).
dns::CutGeometry openBox(const dns::Grid& g)
{
    using namespace dns;
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        cut.lambda[c] = 1.0;
        cut.ax[c] = cut.ay[c] = cut.az[c] = 1.0;
    }
    closeBoxBoundary(g, cut);
    return cut;
}

// Freestream through N full projection cycles; returns max |q - freestream|.
double freestreamDrift(const dns::Grid& g, int cycles, int& cgIters)
{
    using namespace dns;
    const auto gg = gpu::GpuGrid::from(g);
    const auto cut = openBox(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    auto dU = gpu::DeviceField::make(g.size());
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    auto tmp = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto bc = gpu::DeviceTunnelBc::make(gg, {1.0, true});
    if (!(dCut && dU && scr && tmp && red && bc)) return 1e9;
    auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 200, 1e-6,
                                                     gpu::PoissonBc::NeumannBox);
    if (!poisson) return 1e9;

    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = 1.0;
    (void)gpu::uploadField(U, *dU);
    (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(gpu::Real));
    bc->apply(*dU, *dU, 0.0);

    const double dt = 0.01;
    cgIters = 0;
    for (int s = 0; s < cycles; ++s)
    {
        bc->apply(*dU, *dU, 1.0 * dt / g.hx);
        bc->correctOutletFlux(*dU, *red, *tmp);
        gpu::divergenceCut(gg, *dCut, *dU, scr->div, gpu::Real(1.0 / dt));
        bc->addFluxRhs(*dU, 1.0 / dt, scr->div);
        cgIters += poisson->solve(scr->div, scr->phi, true).iters;
        gpu::subtractGradientCut(gg, *dCut, scr->phi, gpu::Real(dt), *dU);
        bc->apply(*dU, *dU, 0.0);
    }
    auto R = VectorField::zeros(g);
    (void)gpu::downloadField(*dU, R);
    double m = 0;
    for (std::size_t c = 0; c < g.size(); ++c)
    {
        m = std::max(m, std::abs(R.u[c] - 1.0));
        m = std::max(m, std::abs(R.v[c]));
        m = std::max(m, std::abs(R.w[c]));
    }
    return m;
}

int runApp(const std::string& args)
{
    return valtest::runBinary("\"" DNS_BIN "\" flow --body cylinder " + args);
}

void cleanup(const std::string& prefix)
{
    for (const char* suf : {"_stats.csv", ".json", "_vort.bin", "_speed.bin",
                            "_mask.bin"})
        std::remove((prefix + suf).c_str());
}

} // namespace

int main()
{
    using namespace dns;

    dns::test::requireDeviceOrSkip();

    // --- 1) freestream preservation, 2D ------------------------------------
    {
        int iters = 0;
        const double drift =
            freestreamDrift(Grid::make(34, 34, 1, 34.0, 34.0, 1.0), 20, iters);
        std::printf("2d drift %.3e, cg iters %d\n", drift, iters);
        check(drift < 5e-6, "2d freestream preserved through 20 cycles");
        check(iters <= 20, "2d freestream needs ~no CG work");
    }

    // --- 2) freestream preservation, 3D ------------------------------------
    {
        int iters = 0;
        const double drift =
            freestreamDrift(Grid::make(20, 20, 12, 20.0, 20.0, 12.0), 20, iters);
        std::printf("3d drift %.3e, cg iters %d\n", drift, iters);
        check(drift < 5e-6, "3d freestream preserved through 20 cycles");
        check(iters <= 20, "3d freestream needs ~no CG work");
    }

    // --- 3) end-to-end short cylinder run ----------------------------------
    {
        const std::string prefix = "tunnel_open_test";
        const int rc = runApp("--far tunnel --open --n 256 --nz 1 --re 60 "
                              "--dr 8 --ft 2 --fdt 1e9 --out " + prefix);
        check(rc == 0, "--far tunnel --open run exits 0");
        // By column NAME, never by index (the schema is documented that way).
        const auto tab = valtest::loadCsv(prefix + "_stats.csv");
        check(tab && tab->rows() > 5, "stats CSV has data rows");
        if (tab && tab->rows() > 0)
        {
            const auto& t = tab->col("t");
            const auto& cd = tab->col("cd");
            const auto& div = tab->col("maxDiv");
            double maxDiv = 0;
            bool finite = true;
            for (std::size_t q = 0; q < tab->rows(); ++q)
            {
                if (t[q] > 1.0) maxDiv = std::max(maxDiv, div[q]);
                finite = finite && std::isfinite(cd[q]);
            }
            std::printf("cd(t=%.2f) = %.4f, max maxDiv(t>1) = %.3e\n",
                        t.back(), cd.back(), maxDiv);
            check(finite, "cd finite throughout");
            check(maxDiv < 1e-3, "projection healthy (maxDiv < 1e-3 past start)");
            check(cd.back() > 0.5 && cd.back() < 4.0, "cd in a sane startup band");
        }
        cleanup(prefix);
    }

    std::printf("%s\n", dnstest::gFailures == 0 ? "ALL OK" : "FAILURES");
    return dnstest::finish();
}
