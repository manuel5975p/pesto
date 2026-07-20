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
#include "dns/gpu/tunnel.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

constexpr int skipExitCode = 77;
bool pass = true;

void check(bool ok, const char* what)
{
    std::printf("%-56s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) pass = false;
}

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
        cgIters += poisson->solve(scr->div, scr->phi, true);
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
    const std::string cmd = "\"" DNS_BIN "\" flow --body cylinder " + args
                          + " > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    if (rc == -1 || !WIFEXITED(rc)) return -1;
    return WEXITSTATUS(rc);
}

struct Row
{
    double t, cd, maxDiv;
};

std::vector<Row> readStats(const std::string& prefix)
{
    std::vector<Row> rows;
    std::ifstream f(prefix + "_stats.csv");
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
        std::stringstream ls(line);
        std::string cell;
        double col[8]{};
        bool ok = true;
        for (int q = 0; q < 8 && ok; ++q)
        {
            ok = bool(std::getline(ls, cell, ','));
            if (ok) col[q] = std::strtod(cell.c_str(), nullptr);
        }
        if (ok) rows.push_back({col[0], col[3], col[7]});
    }
    return rows;
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

    if (!gpu::deviceAvailable())
    {
        std::printf("no HIP device, skipping\n");
        return skipExitCode;
    }

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
        const auto rows = readStats(prefix);
        check(rows.size() > 5, "stats CSV has data rows");
        if (!rows.empty())
        {
            double maxDiv = 0;
            bool finite = true;
            for (const auto& r : rows)
            {
                if (r.t > 1.0) maxDiv = std::max(maxDiv, r.maxDiv);
                finite = finite && std::isfinite(r.cd);
            }
            std::printf("cd(t=%.2f) = %.4f, max maxDiv(t>1) = %.3e\n",
                        rows.back().t, rows.back().cd, maxDiv);
            check(finite, "cd finite throughout");
            check(maxDiv < 1e-3, "projection healthy (maxDiv < 1e-3 past start)");
            check(rows.back().cd > 0.5 && rows.back().cd < 4.0,
                  "cd in a sane startup band");
        }
        cleanup(prefix);
    }

    std::printf("%s\n", pass ? "ALL OK" : "FAILURES");
    return pass ? 0 : 1;
}
