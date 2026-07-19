// Flow around a 45-deg rotated cube via the cut-cell embedded boundary (GPU).
// Same scheme as the CPU cube_cut app — sharp no-through-flow on cut faces +
// apertured pressure projection — with geometry built exactly on the CPU and
// all field work on device; the pressure solve is FFT-preconditioned CG.
// Usage: cube_cut_gpu [n] [--re ReD] [--dr DRatio] [--ft flowThroughs]
//                     [--flat nz] [--fdt frameDt] [--tol rms] [--out prefix]
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

namespace {

using namespace dns;
using gpu::Real;

// Snap a world coordinate to the nearest grid node so diamond edges/caps stay
// node-aligned (the exact-halving condition).
double snapNode(double x, double h) { return std::round(x / h) * h; }

double parseOr(const char* s, double fallback)
{
    double v{};
    const std::string_view sv(s);
    auto [q, ec] = std::from_chars(sv.begin(), sv.end(), v);
    return ec == std::errc{} && q == sv.end() ? v : fallback;
}

// Relax the outflow strip back to the free stream (smoothstep ramp) and record
// the aperture-weighted streamwise momentum each cell injects into mom[] so a
// single reduction yields the drag.
__global__ void kFringeCut(gpu::GpuGrid g, const Real* ax, Real x0, Real x1,
                           Real uInf, Real s0, Real* u, Real* v, Real* w, Real* mom)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    auto ramp = [&](Real x) {
        if (x < x0 || x > x1) return Real(0);
        const Real xi = (x - x0) / (x1 - x0);
        return s0 * xi * xi * (3 - 2 * xi);
    };
    const Real su = ramp(i * g.hx), sc = ramp((i + Real(0.5)) * g.hx);
    const Real du = su * (uInf - u[c]);
    mom[c] = ax[c] * du;
    u[c] += du;
    v[c] -= sc * v[c];
    w[c] -= sc * w[c];
}

hipError_t downloadSlab(const gpu::DeviceScalar& q, const gpu::GpuGrid& g, int k,
                        std::vector<Real>& out)
{
    const std::size_t slab = std::size_t(g.nx) * g.ny;
    out.resize(slab);
    return hipMemcpy(out.data(), q.data() + slab * std::size_t(k),
                     slab * sizeof(Real), hipMemcpyDeviceToHost);
}

} // namespace

int main(int argc, char** argv)
{
    using std::numbers::pi;

    int n = 96, nzFlat = 0;
    double reD = 300.0, flowThroughs = 6.0, dRatio = 8.0, frameDt = 0.15;
    double tol = 1e-4;
    std::string prefix = "cube_cut_gpu";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--flat") nzFlat = int(next(nzFlat)); // quasi-2D: thin z, prism
        else if (s == "--fdt") frameDt = next(frameDt);
        else if (s == "--tol") tol = next(tol);
        else if (s == "--out" && a + 1 < argc) prefix = argv[++a];
        else n = int(parseOr(argv[a], n));
    }
    if (!gpu::deviceAvailable())
    {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }

    const double L = 2 * pi, uInf = 1.0;
    const double h = L / n;
    // rocFFT rejects 3D plans with near-unit dimensions: keep nz >= 4.
    const int nz = nzFlat > 0 ? std::max(nzFlat, 4) : n;
    const auto g = Grid::make(n, n, nz, L, L, nz * h); // cubic cells (hz == h)

    // Diamond half-diagonal R -> square side R*sqrt(2) = "D". Center on nodes so
    // every cut cell is halved. In flat mode the prism spans all z (2D cylinder-
    // like body); in 3D mode z-caps are node-aligned too.
    const double R = snapNode(L / dRatio / std::numbers::sqrt2, h);
    const double D = R * std::numbers::sqrt2;
    const double cx = snapNode(L / 3.0, h), cy = snapNode(L / 2.0, h);
    const double halfZ = snapNode(D / 2.0, h);
    const RotatedCube cube = nzFlat > 0
        ? RotatedCube{cx, cy, R, -1.0, nz * h + 1.0}
        : RotatedCube{cx, cy, R, L / 2 - halfZ, L / 2 + halfZ};
    const auto cut = CutGeometry::make(g, cube);

    const double nu = uInf * D / reD;
    std::printf("# cube_cut_gpu n=%d nz=%d Re_D=%g nu=%.4e D=%.4f (%.1f cells)\n",
                n, nz, reD, nu, D, D / h);

    // Verify the halving invariant up front (cheap, and the whole point).
    int nCut = 0, offHalf = 0;
    for (double l : cut.lambda)
        if (l > 1e-12 && l < 1 - 1e-12) { ++nCut; if (std::abs(l - 0.5) > 1e-9) ++offHalf; }
    std::printf("# cut cells: %d, off-half: %d\n", nCut, offHalf);

    // Clean laminar free stream (wind-channel inflow). A small deterministic
    // trip in the strip one D behind the cube breaks the wake symmetry so
    // shedding starts without seeding the whole field with noise.
    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = uInf;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > cx + R && x < cx + R + D && std::abs(y - cy) < R)
                    U.v[g.idx(i, j, k)] = 0.05 * uInf;
            }

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    // Sharp viscous wall: diffusion reads a ghost-filled copy (mirror values,
    // no-slip at the true sub-cell surface); convection keeps masked zeros.
    auto dGhost = gpu::DeviceGhostFill::make(
        g, cut, [&](double x, double y, double z) {
            const double d2 = (std::abs(x - cube.cx) + std::abs(y - cube.cy)
                               - cube.r) / std::numbers::sqrt2;
            return std::max({d2, cube.z0 - z, z - cube.z1});
        });
    auto dU = gpu::DeviceField::make(g.size());
    auto dU0 = gpu::DeviceField::make(g.size());
    auto dS = gpu::DeviceField::make(g.size());
    auto dG = gpu::DeviceField::make(g.size()); // ghost-filled diffusion view
    auto dMom = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    if (!dCut || !dGhost || !dU || !dU0 || !dS || !dG || !dMom || !red || !scr)
    {
        std::fprintf(stderr, "GPU alloc failed\n");
        return 1;
    }
    auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, tol);
    if (!poisson)
    {
        std::fprintf(stderr, "GPU Poisson init failed\n");
        return 1;
    }
    (void)gpu::uploadField(U, *dU);
    gpu::maskSolidFaces(gg, *dCut, *dU);
    (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(Real)); // warm-start seed

    // One SSP-RK3 stage: out = a*U0 + (1-a)*(in + dt*(conv + nu lap)), masked and
    // cut-projected. The fused kernel must not write onto its stencil input.
    int pcgIters = 0;
    auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double a,
                     double dt) {
        (void)dGhost->fillFrom(in, *dG); // sub-cell wall for the viscous term
        gpu::fusedRhsStage(gg, Real(nu), Real(0), 0, 0, 0, Real(a), *dU0,
                           Real(1.0 - a), in, *dG, Real(dt), out);
        gpu::maskSolidFaces(gg, *dCut, out);
        pcgIters += gpu::projectCut(gg, *dCut, *poisson, Real(dt), out, *scr);
    };

    const double tEnd = flowThroughs * L / uInf;
    const double xf0 = 0.85 * L, sigmaFringe = 40.0;
    // Frontal area: D x span. In flat mode the prism spans the whole (thin)
    // z-extent, so the span is Lz, not D.
    const double span = nzFlat > 0 ? nz * h : D;
    const double qRef = 0.5 * uInf * uInf * D * span;
    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,K,cd,cfl,maxDiv,pcgIters\n";

    // Startup meta so a viewer can render before the run finishes.
    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frameDt\":" << frameDt
        << ",\"D\":" << D << ",\"reD\":" << reD << ",\"cx\":" << cx / h
        << ",\"cy\":" << cy / h << ",\"R\":" << R / h << "}\n";

    // Mid-plane slice output for the movie: signed z-vorticity at cell corners.
    const int kc = nz / 2;
    std::ofstream vortFile(prefix + "_vort.bin", std::ios::binary);
    int frames = 0;
    double nextFrame = 0.0;
    std::vector<Real> u0, v0;
    std::vector<float> frame(std::size_t(n) * n);
    auto writeFrame = [&]() {
        (void)downloadSlab(dU->u, gg, kc, u0);
        (void)downloadSlab(dU->v, gg, kc, v0);
        auto at = [&](const std::vector<Real>& q, int i, int j) {
            return double(q[std::size_t(gpu::wrap(j, n)) * n
                            + std::size_t(gpu::wrap(i, n))]);
        };
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const double wz = (at(v0, i, j) - at(v0, i - 1, j)) / h
                                - (at(u0, i, j) - at(u0, i, j - 1)) / h;
                frame[std::size_t(j) * n + i] = float(wz);
            }
        vortFile.write(reinterpret_cast<const char*>(frame.data()),
                       std::streamsize(frame.size() * sizeof(float)));
        ++frames;
    };
    // Solid mask (mid-plane fluid fraction) so the renderer can draw the body.
    {
        std::ofstream mf(prefix + "_mask.bin", std::ios::binary);
        std::vector<float> m(std::size_t(n) * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                m[std::size_t(j) * n + i] = float(cut.lambda[g.idx(i, j, kc)]);
        mf.write(reinterpret_cast<const char*>(m.data()),
                 std::streamsize(m.size() * sizeof(float)));
    }

    double t = 0.0, dt = 1e-3;
    long steps = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < tEnd)
    {
        const double conv = gpu::maxCourantPerDt(gg, *dU, *red);
        const double diff = nu > 0 ? 6.0 * nu / (h * h) : 0.0;
        dt = std::min(tEnd - t, 0.8 / std::max(conv, diff));

        // SSP-RK3 on the dU/dS ping-pong (dU0 keeps the step's initial state).
        std::swap(*dU0, *dU);
        pcgIters = 0;
        stage(*dU0, *dU, 0.0, dt);          // dU = U0 + dt L
        stage(*dU, *dS, 0.75, dt);          // dS = 3/4 U0 + 1/4(dU + dt L)
        stage(*dS, *dU, 1.0 / 3.0, dt);     // dU = 1/3 U0 + 2/3(dS + dt L)

        kFringeCut<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
            gg, dCut->ax.data(), Real(xf0), Real(L), Real(uInf),
            Real(std::min(1.0, sigmaFringe * dt)), dU->u.data(), dU->v.data(),
            dU->w.data(), dMom->data());
        gpu::checkLastKernel("fringeCut");
        const double dMomSum = gpu::sumScalar(*dMom, *red) * g.cellVolume();
        pcgIters += gpu::projectCut(gg, *dCut, *poisson, Real(dt), *dU, *scr);

        if (t >= nextFrame) { writeFrame(); nextFrame += frameDt; }

        if (steps % 20 == 0)
        {
            const double K = gpu::kineticEnergy(gg, *dU, *red);
            const double cd = dMomSum / dt / qRef; // fringe injection == drag
            const double cfl = dt * conv;
            const double maxDiv = gpu::maxAbsDivergenceCut(gg, *dCut, *dU, *red);
            stats << t << ',' << dt << ',' << K << ',' << cd << ',' << cfl << ','
                  << maxDiv << ',' << pcgIters << '\n';
            stats.flush();
            if (steps % 200 == 0)
            {
                std::printf("t=%.2f/%.2f cd=%.3f K=%.1f maxDiv=%.2e pcg=%d\n",
                            t, tEnd, cd, K, maxDiv, pcgIters);
                std::fflush(stdout);
            }
        }
        t += dt;
        ++steps;
    }
    (void)hipDeviceSynchronize();
    const double wallMs = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - wall0).count();

    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frames\":" << frames
        << ",\"frameDt\":" << frameDt << ",\"D\":" << D << ",\"reD\":" << reD
        << ",\"cx\":" << cx / h << ",\"cy\":" << cy / h << ",\"R\":" << R / h
        << ",\"msPerStep\":" << wallMs / double(steps) << "}\n";
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n",
                steps, frames, wallMs / double(steps));
    return 0;
}
