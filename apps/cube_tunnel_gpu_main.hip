// Flow around a 45-deg rotated cube in a true numerical flow channel (GPU):
// no periodic directions. Laminar Dirichlet inflow, convective outflow with
// exact mass-flux correction, free-slip lateral walls, and the cut-cell
// embedded body. Pressure Poisson is PCG preconditioned by the exact DCT
// (NeumannBox) inverse Laplacian; drag/lift come from the discrete surface
// integral (aperture-pressure identity + masked-Laplacian viscous sink).
// The grid carries a one-cell ghost ring: interior is n x n x nz cells.
// Usage: cube_tunnel_gpu [n] [--re ReD] [--dr DRatio] [--ft flowThroughs]
//                        [--flat nz] [--fdt frameDt] [--tol rms] [--out prefix]
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

double snapNode(double x, double h) { return std::round(x / h) * h; }

double parseOr(const char* s, double fallback)
{
    double v{};
    const std::string_view sv(s);
    auto [q, ec] = std::from_chars(sv.begin(), sv.end(), v);
    return ec == std::errc{} && q == sv.end() ? v : fallback;
}

// Ghost-ring boundary conditions, applied after maskSolidFaces zeroed every
// closed face. cdt = Uc*dt/hx drives the convective outlet from the stage
// input state (pass cdt = 0 with in == out aliased for a pure refresh).
//   inlet   (i<=1):        u = uInf; v,w antimirrored so they vanish on x=hx
//   outlet  (i==nx-1):     convective advection out of the domain
//   walls   (j,k rings):   freestream far-field -- u = uInf, v = w = 0. This is
//     an OPEN lateral BC (no mirror), so the domain does not confine the body
//     into a periodic array the way a symmetry/free-slip wall would; the DCT
//     Neumann pressure BC dp/dn = 0 is still correct under a prescribed far
//     field. Wall distance (--dr) sets the residual blockage.
__global__ void kBcTunnel(gpu::GpuGrid g, Real uInf, Real cdt, const Real* inu,
                          const Real* inv, const Real* inw, Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    // FREE-SLIP lateral walls (Neumann): tangential mirrored (no wall-generated
    // vorticity), normal 0 on the wall face, antimirrored ghost. Pressure is
    // already Neumann (closed apertures + DCT).
    const int jm = j == 0 ? 1 : j == ny - 1 ? ny - 2 : j;
    const int km = k == 0 ? 1 : k == nz - 1 ? nz - 2 : k;

    if (i <= 1) u[c] = uInf;
    else if (i == nx - 1)
        u[c] = inu[c] - cdt * (inu[c] - inu[gidx(g, nx - 2, j, k)]);
    else if (jm != j || km != k) u[c] = u[gidx(g, i, jm, km)];

    if (i == 0) v[c] = -v[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        v[c] = inv[c] - cdt * (inv[c] - inv[gidx(g, nx - 2, j, k)]);
    else if (j == 1 || j == ny - 1) v[c] = 0; // impermeable wall faces
    else if (j == 0) v[c] = -v[gidx(g, i, 2, km)]; // antimirror across y = hy
    else if (km != k) v[c] = v[gidx(g, i, j, km)];

    if (i == 0) w[c] = -w[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        w[c] = inw[c] - cdt * (inw[c] - inw[gidx(g, nx - 2, j, k)]);
    else if (k == 1 || k == nz - 1) w[c] = 0; // impermeable wall faces
    else if (k == 0) w[c] = -w[gidx(g, i, jm, 2)]; // antimirror across z = hz
    else if (jm != j) w[c] = w[gidx(g, i, jm, k)];
}

// Per-cell tally of the outlet-plane u so one sumScalar yields the volume flux.
__global__ void kOutletTally(gpu::GpuGrid g, const Real* u, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    const bool plane = i == g.nx - 1 && j >= 1 && j <= g.ny - 2 && k >= 1
                    && k <= g.nz - 2;
    out[c] = plane ? u[c] : Real(0);
}

// u_outlet += corr on the open outlet plane (exact global mass balance).
__global__ void kOutletCorrect(gpu::GpuGrid g, Real corr, Real* u)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    if (i == g.nx - 1 && j >= 1 && j <= g.ny - 2 && k >= 1 && k <= g.nz - 2)
        u[c] += corr;
}

// The inlet/outlet faces carry prescribed fluxes but zero apertures, so
// divergenceCut misses them; add the missing terms to the projection RHS.
__global__ void kInOutFluxRhs(gpu::GpuGrid g, Real scale, Real uInf, const Real* u,
                              Real* rhs)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    if (j < 1 || j > g.ny - 2 || k < 1 || k > g.nz - 2) return;
    if (i == 1) rhs[c] -= scale * uInf / g.hx;
    else if (i == g.nx - 2) rhs[c] += scale * u[gidx(g, g.nx - 1, j, k)] / g.hx;
}

// Pressure force the projection exerts on the body inside a bounding box:
// F = -sum phi_c * (a_xp - a_c) * A_face (discrete aperture identity, the
// exact surface integral of phi over the embedded boundary).
__global__ void kBodyPressure(gpu::GpuGrid g, const Real* lam, const Real* ax,
                              const Real* ay, const Real* phi, int i0, int i1,
                              int j0, int j1, int k0, int k1, Real* fx, Real* fy)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    fx[c] = 0;
    fy[c] = 0;
    if (lam[c] <= 0 || i < i0 || i > i1 || j < j0 || j > j1 || k < k0 || k > k1)
        return;
    fx[c] = -phi[c] * (ax[gidx(g, i + 1, j, k)] - ax[c]) * g.hy * g.hz;
    fy[c] = -phi[c] * (ay[gidx(g, i, j + 1, k)] - ay[c]) * g.hx * g.hz;
}


hipError_t downloadSlab(const gpu::DeviceScalar& q, const gpu::GpuGrid& g, int k,
                        std::vector<Real>& out)
{
    const std::size_t slab = std::size_t(g.nx) * g.ny;
    out.resize(slab);
    return hipMemcpy(out.data(), q.data() + slab * std::size_t(k),
                     slab * sizeof(Real), hipMemcpyDeviceToHost);
}

// Strided x-z slice at fixed j: out[k*nx + i] = q(i, j, k).
hipError_t downloadSideSlab(const gpu::DeviceScalar& q, const gpu::GpuGrid& g,
                            int j, std::vector<Real>& out)
{
    out.resize(std::size_t(g.nx) * g.nz);
    const std::size_t rowB = std::size_t(g.nx) * sizeof(Real);
    return hipMemcpy2D(out.data(), rowB,
                       q.data() + std::size_t(j) * g.nx,
                       std::size_t(g.nx) * g.ny * sizeof(Real), rowB, g.nz,
                       hipMemcpyDeviceToHost);
}

} // namespace

int main(int argc, char** argv)
{
    using std::numbers::pi;

    int n = 96, nzFlat = 0;
    double reD = 300.0, flowThroughs = 6.0, dRatio = 8.0, frameDt = 0.15;
    double tol = 1e-4;
    std::string prefix = "cube_tunnel_gpu";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--flat") nzFlat = int(next(nzFlat)); // quasi-2D prism
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

    // Interior n x n x nzi cells of size h; one ghost cell on every side.
    const double L = 2 * pi, uInf = 1.0;
    const double h = L / n;
    const int nzi = nzFlat > 0 ? std::max(nzFlat, 4) : n;
    const int nx = n + 2, ny = n + 2, nzt = nzi + 2;
    const auto g = Grid::make(nx, ny, nzt, nx * h, ny * h, nzt * h);

    // Body in grid coordinates: physical position + the ghost offset h.
    const double R = snapNode(L / dRatio / std::numbers::sqrt2, h);
    const double D = R * std::numbers::sqrt2;
    const double cx = h + snapNode(L / 3.0, h), cy = h + snapNode(L / 2.0, h);
    const double halfZ = snapNode(D / 2.0, h);
    const double zc = h + snapNode(nzi * h / 2.0, h);
    const RotatedCube cube = nzFlat > 0
        ? RotatedCube{cx, cy, R, -1.0, nzt * h + 1.0}
        : RotatedCube{cx, cy, R, zc - halfZ, zc + halfZ};
    auto cut = CutGeometry::make(g, cube);
    closeBoxBoundary(g, cut);

    const double nu = uInf * D / reD;
    std::printf("# cube_tunnel_gpu n=%d nz=%d Re_D=%g nu=%.4e D=%.4f (%.1f cells)\n",
                n, nzi, reD, nu, D, D / h);

    int nCut = 0, offHalf = 0;
    for (double l : cut.lambda)
        if (l > 1e-12 && l < 1 - 1e-12) { ++nCut; if (std::abs(l - 0.5) > 1e-9) ++offHalf; }
    std::printf("# cut cells: %d, off-half: %d\n", nCut, offHalf);

    // Clean laminar free stream; deterministic trip one D behind the cube.
    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = uInf;
    for (int k = 0; k < nzt; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > cx + R && x < cx + R + D && std::abs(y - cy) < R)
                    U.v[g.idx(i, j, k)] = 0.05 * uInf;
            }

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    // Sharp viscous wall: diffusion reads a ghost-filled copy (mirror values,
    // no-slip at the true sub-cell surface); convection keeps masked zeros.
    // Diamond sdf is exact away from the corners; the z-slab caps compose by max.
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
    auto dFx = gpu::DeviceScalar::make(g.size());
    auto dFy = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    if (!dCut || !dGhost || !dU || !dU0 || !dS || !dG || !dFx || !dFy || !red
        || !scr)
    {
        std::fprintf(stderr, "GPU alloc failed\n");
        return 1;
    }
    auto poisson = gpu::DeviceCutPoissonSolver::make(gg, *dCut, 500, tol,
                                                     gpu::PoissonBc::NeumannBox);
    if (!poisson)
    {
        std::fprintf(stderr, "GPU Poisson init failed\n");
        return 1;
    }
    (void)gpu::uploadField(U, *dU);
    gpu::maskSolidFaces(gg, *dCut, *dU);
    (void)hipMemset(scr->phi.data(), 0, g.size() * sizeof(Real)); // warm-start seed

    const double planeArea = double(n) * nzi * h * h;
    auto bcApply = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double cdt) {
        kBcTunnel<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
            gg, Real(uInf), Real(cdt), in.u.data(), in.v.data(), in.w.data(),
            out.u.data(), out.v.data(), out.w.data());
        gpu::checkLastKernel("bcTunnel");
    };
    auto fluxCorrect = [&](gpu::DeviceField& out) {
        kOutletTally<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
            gg, out.u.data(), dFx->data());
        gpu::checkLastKernel("outletTally");
        const double qOut = gpu::sumScalar(*dFx, *red) * h * h;
        const double corr = (uInf * planeArea - qOut) / planeArea;
        kOutletCorrect<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
            gg, Real(corr), out.u.data());
        gpu::checkLastKernel("outletCorrect");
    };
    bcApply(*dU, *dU, 0.0); // seed ghosts/prescribed faces

    // One SSP-RK3 stage with confined-channel BCs and the ghost-flux projection.
    int pcgIters = 0;
    auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double a,
                     double dt) {
        (void)dGhost->fillFrom(in, *dG); // sub-cell wall for the viscous term
        gpu::fusedRhsStage(gg, Real(nu), Real(0), 0, 0, 0, Real(a), *dU0,
                           Real(1.0 - a), in, *dG, Real(dt), out);
        gpu::maskSolidFaces(gg, *dCut, out);
        bcApply(in, out, uInf * dt / h);
        fluxCorrect(out);
        gpu::divergenceCut(gg, *dCut, out, scr->div, Real(1.0 / dt));
        kInOutFluxRhs<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
            gg, Real(1.0 / dt), Real(uInf), out.u.data(), scr->div.data());
        gpu::checkLastKernel("inOutFluxRhs");
        pcgIters += poisson->solve(scr->div, scr->phi, true);
        gpu::subtractGradientCut(gg, *dCut, scr->phi, Real(dt), out);
        bcApply(out, out, 0.0); // refresh mirrors after the correction
    };

    // Bounding box (grid indices) for the body-force reduction: the body plus
    // margin, strictly inside the interior so boundary faces never count.
    const int bi0 = std::max(2, int((cx - R) / h) - 2);
    const int bi1 = std::min(nx - 3, int((cx + R) / h) + 3);
    const int bj0 = std::max(2, int((cy - R) / h) - 2);
    const int bj1 = std::min(ny - 3, int((cy + R) / h) + 3);
    const int bk0 = nzFlat > 0 ? 1 : std::max(2, int((zc - halfZ) / h) - 2);
    const int bk1 = nzFlat > 0 ? nzt - 2 : std::min(nzt - 3, int((zc + halfZ) / h) + 3);

    const double tEnd = flowThroughs * L / uInf;
    const double span = nzFlat > 0 ? nzi * h : D;
    const double qRef = 0.5 * uInf * uInf * D * span;
    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,K,cd,cl,cfl,maxDiv,pcgIters\n";

    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frameDt\":" << frameDt
        << ",\"D\":" << D << ",\"reD\":" << reD << ",\"cx\":" << cx / h - 1
        << ",\"cy\":" << cy / h - 1 << ",\"R\":" << R / h << "}\n";

    // Interior slices at cell corners: signed z-vorticity on the mid-z plane
    // (top view) and y-vorticity on the body-center x-z plane (side view).
    const int kc = nzt / 2;
    const int jc = int(cy / h + 0.5);
    std::ofstream vortFile(prefix + "_vort.bin", std::ios::binary);
    std::ofstream sideFile(prefix + "_side_vort.bin", std::ios::binary);
    int frames = 0;
    double nextFrame = 0.0;
    std::vector<Real> u0, v0, w0;
    std::vector<float> frame(std::size_t(n) * n);
    std::vector<float> side(std::size_t(n) * nzi);
    auto writeFrame = [&]() {
        (void)downloadSlab(dU->u, gg, kc, u0);
        (void)downloadSlab(dU->v, gg, kc, v0);
        auto at = [&](const std::vector<Real>& q, int i, int j) {
            return double(q[std::size_t(j) * nx + std::size_t(i)]);
        };
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const int ii = i + 1, jj = j + 1;
                const double wz = (at(v0, ii, jj) - at(v0, ii - 1, jj)) / h
                                - (at(u0, ii, jj) - at(u0, ii, jj - 1)) / h;
                frame[std::size_t(j) * n + i] = float(wz);
            }
        vortFile.write(reinterpret_cast<const char*>(frame.data()),
                       std::streamsize(frame.size() * sizeof(float)));

        (void)downloadSideSlab(dU->u, gg, jc, u0);
        (void)downloadSideSlab(dU->w, gg, jc, w0);
        for (int k = 0; k < nzi; ++k)
            for (int i = 0; i < n; ++i)
            {
                const int ii = i + 1, kk = k + 1;
                const double wy = (at(u0, ii, kk) - at(u0, ii, kk - 1)) / h
                                - (at(w0, ii, kk) - at(w0, ii - 1, kk)) / h;
                side[std::size_t(k) * n + i] = float(wy);
            }
        sideFile.write(reinterpret_cast<const char*>(side.data()),
                       std::streamsize(side.size() * sizeof(float)));
        ++frames;
    };
    {
        std::ofstream mf(prefix + "_mask.bin", std::ios::binary);
        std::vector<float> m(std::size_t(n) * n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                m[std::size_t(j) * n + i] = float(cut.lambda[g.idx(i + 1, j + 1, kc)]);
        mf.write(reinterpret_cast<const char*>(m.data()),
                 std::streamsize(m.size() * sizeof(float)));
        std::ofstream sf(prefix + "_side_mask.bin", std::ios::binary);
        std::vector<float> sm(std::size_t(n) * nzi);
        for (int k = 0; k < nzi; ++k)
            for (int i = 0; i < n; ++i)
                sm[std::size_t(k) * n + i] = float(cut.lambda[g.idx(i + 1, jc, k + 1)]);
        sf.write(reinterpret_cast<const char*>(sm.data()),
                 std::streamsize(sm.size() * sizeof(float)));
    }

    double t = 0.0, dt = 1e-3;
    long steps = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < tEnd)
    {
        const double conv = gpu::maxCourantPerDt(gg, *dU, *red);
        const double diff = nu > 0 ? 6.0 * nu / (h * h) : 0.0;
        dt = std::min(tEnd - t, 0.8 / std::max(conv, diff));

        std::swap(*dU0, *dU);
        pcgIters = 0;
        stage(*dU0, *dU, 0.0, dt);
        double cd = 0.0, cl = 0.0;
        if (steps % 20 == 0)
        {
            // Surface pressure integral from the FIRST stage's phi: that stage
            // advances a full dt, so phi is the static pressure (stage-3 phi
            // under-reads the force by 1/3). Form drag dominates at Re ~ 3e2;
            // skin friction is a few-percent correction.
            kBodyPressure<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                gg, dCut->lambda.data(), dCut->ax.data(), dCut->ay.data(),
                scr->phi.data(), bi0, bi1, bj0, bj1, bk0, bk1, dFx->data(),
                dFy->data());
            gpu::checkLastKernel("bodyPressure");
            cd = gpu::sumScalar(*dFx, *red) / qRef;
            cl = gpu::sumScalar(*dFy, *red) / qRef;
        }
        stage(*dU, *dS, 0.75, dt);
        stage(*dS, *dU, 1.0 / 3.0, dt);

        if (t >= nextFrame) { writeFrame(); nextFrame += frameDt; }

        if (steps % 20 == 0)
        {
            const double K = gpu::kineticEnergy(gg, *dU, *red);
            const double cfl = dt * conv;

            // Continuity residual including the prescribed in/out fluxes.
            gpu::divergenceCut(gg, *dCut, *dU, scr->div, Real(1));
            kInOutFluxRhs<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                gg, Real(1), Real(uInf), dU->u.data(), scr->div.data());
            gpu::checkLastKernel("inOutFluxRhs");
            const double maxDiv = gpu::maxAbsScalar(scr->div, *red);

            stats << t << ',' << dt << ',' << K << ',' << cd << ',' << cl << ','
                  << cfl << ',' << maxDiv << ',' << pcgIters << '\n';
            stats.flush();
            if (steps % 200 == 0)
            {
                std::printf("t=%.2f/%.2f cd=%.3f cl=%+.3f "
                            "K=%.1f maxDiv=%.2e pcg=%d\n",
                            t, tEnd, cd, cl, K, maxDiv, pcgIters);
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
        << ",\"cx\":" << cx / h - 1 << ",\"cy\":" << cy / h - 1
        << ",\"R\":" << R / h << ",\"msPerStep\":" << wallMs / double(steps) << "}\n";
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n",
                steps, frames, wallMs / double(steps));
    return 0;
}
