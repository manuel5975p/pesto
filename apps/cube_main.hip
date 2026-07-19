// Flow around a cube (GPU): direct-forcing immersed boundary on the periodic
// solver — cube faces masked each step, bulk velocity pinned to U = 1, drag
// from the momentum balance of that pin. Slices + stats for offline rendering.
// Usage: cube [n] [--re ReD] [--ft flowThroughs] [--out prefix]
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/poisson.hpp"
#include "dns/gpu/projection.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/timestepper.hpp"

#include <hip/hip_runtime.h>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace dns;
using gpu::Real;

// Axis-aligned solid cube; face samples inside are forced to rest.
struct CubeMask
{
    Real x0, x1, y0, y1, z0, z1;
};

__global__ void kMaskCube(gpu::GpuGrid g, CubeMask c, Real* u, Real* v, Real* w)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, idx, i, j, k);
    const Real xc = (i + Real(0.5)) * g.hx, xf = i * g.hx;
    const Real yc = (j + Real(0.5)) * g.hy, yf = j * g.hy;
    const Real zc = (k + Real(0.5)) * g.hz, zf = k * g.hz;
    if (xf >= c.x0 && xf <= c.x1 && yc >= c.y0 && yc <= c.y1 && zc >= c.z0 && zc <= c.z1)
        u[idx] = 0;
    if (xc >= c.x0 && xc <= c.x1 && yf >= c.y0 && yf <= c.y1 && zc >= c.z0 && zc <= c.z1)
        v[idx] = 0;
    if (xc >= c.x0 && xc <= c.x1 && yc >= c.y0 && yc <= c.y1 && zf >= c.z0 && zf <= c.z1)
        w[idx] = 0;
}

void maskCube(const gpu::GpuGrid& g, const CubeMask& c, gpu::DeviceField& U)
{
    kMaskCube<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(g, c, U.u.data(),
                                                               U.v.data(), U.w.data());
    gpu::checkLastKernel("kMaskCube");
}

// Fringe region: relax the outflow strip back to the free stream so the
// periodic wrap-around feeds clean flow to the cube (isolated body, not an
// array). Smoothstep-ramped, dt-proportional blend.
__global__ void kFringe(gpu::GpuGrid g, Real x0, Real x1, Real uInf, Real s0,
                        Real* u, Real* v, Real* w)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, idx, i, j, k);
    auto ramp = [&](Real x) {
        if (x < x0 || x > x1) return Real(0);
        const Real xi = (x - x0) / (x1 - x0);
        return s0 * xi * xi * (3 - 2 * xi);
    };
    const Real su = ramp(i * g.hx), sc = ramp((i + Real(0.5)) * g.hx);
    u[idx] += su * (uInf - u[idx]);
    v[idx] -= sc * v[idx];
    w[idx] -= sc * w[idx];
}

void fringe(const gpu::GpuGrid& g, Real x0, Real x1, Real uInf, Real s0,
            gpu::DeviceField& U)
{
    kFringe<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(g, x0, x1, uInf, s0,
                                                             U.u.data(), U.v.data(),
                                                             U.w.data());
    gpu::checkLastKernel("kFringe");
}

hipError_t downloadSlab(const gpu::DeviceScalar& q, const gpu::GpuGrid& g, int k,
                        std::vector<Real>& out)
{
    const std::size_t slab = std::size_t(g.nx) * g.ny;
    out.resize(slab);
    return hipMemcpy(out.data(), q.data() + slab * std::size_t(k),
                     slab * sizeof(Real), hipMemcpyDeviceToHost);
}

struct SliceWriter
{
    std::ofstream speed, vort, pres, ux;
    int frames = 0;

    explicit SliceWriter(const std::string& prefix)
        : speed(prefix + "_speed.bin", std::ios::binary),
          vort(prefix + "_vort.bin", std::ios::binary),
          pres(prefix + "_p.bin", std::ios::binary),
          ux(prefix + "_ux.bin", std::ios::binary)
    {
    }

    void append(std::ofstream& f, const std::vector<float>& q)
    {
        f.write(reinterpret_cast<const char*>(q.data()),
                std::streamsize(q.size() * sizeof(float)));
    }
};

// Mid-plane slices; pressure from lap(p) = -div((u.grad)u), obstacle forces omitted.
void writeFrame(const gpu::GpuGrid& g, const gpu::DeviceField& U, gpu::DeviceField& R,
                gpu::DeviceScalar& rhs, gpu::DeviceScalar& p,
                gpu::DevicePoissonSolver& poisson, SliceWriter& out)
{
    const int kc = g.nz / 2, nx = g.nx, ny = g.ny;
    const std::size_t slab = std::size_t(nx) * ny;

    gpu::convection(g, U, R);
    gpu::divergence(g, R, rhs, Real(-1));
    poisson.solve(rhs, p);

    std::vector<Real> um, u0, v0, vm, w0, wp, pc;
    (void)downloadSlab(U.u, g, kc, u0);
    (void)downloadSlab(U.u, g, kc - 1, um);
    (void)downloadSlab(U.v, g, kc, v0);
    (void)downloadSlab(U.v, g, kc - 1, vm);
    (void)downloadSlab(U.w, g, kc, w0);
    (void)downloadSlab(U.w, g, kc + 1, wp);
    (void)downloadSlab(p, g, kc, pc);

    auto at = [&](const std::vector<Real>& q, int i, int j) {
        return double(q[std::size_t(gpu::wrap(j, ny)) * nx
                        + std::size_t(gpu::wrap(i, nx))]);
    };

    std::vector<float> speed(slab), vort(slab), uxs(slab);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
        {
            const double uc = 0.5 * (at(u0, i, j) + at(u0, i + 1, j));
            const double vc = 0.5 * (at(v0, i, j) + at(v0, i, j + 1));
            const double wc = 0.5 * (at(w0, i, j) + at(wp, i, j));
            const double wx = (at(w0, i, j) - at(w0, i, j - 1)) / g.hy
                            - (at(v0, i, j) - at(vm, i, j)) / g.hz;
            const double wy = (at(u0, i, j) - at(um, i, j)) / g.hz
                            - (at(w0, i, j) - at(w0, i - 1, j)) / g.hx;
            const double wz = (at(v0, i, j) - at(v0, i - 1, j)) / g.hx
                            - (at(u0, i, j) - at(u0, i, j - 1)) / g.hy;
            const std::size_t c = std::size_t(j) * nx + i;
            speed[c] = float(std::sqrt(uc * uc + vc * vc + wc * wc));
            vort[c] = float(std::sqrt(wx * wx + wy * wy + wz * wz));
            uxs[c] = float(uc);
        }

    out.append(out.speed, speed);
    out.append(out.vort, vort);
    out.append(out.pres, std::vector<float>(pc.begin(), pc.end()));
    out.append(out.ux, uxs);
    ++out.frames;
}

double parseOr(const char* s, double fallback)
{
    double v{};
    const std::string_view sv(s);
    auto [q, ec] = std::from_chars(sv.begin(), sv.end(), v);
    return ec == std::errc{} && q == sv.end() ? v : fallback;
}

} // namespace

int main(int argc, char** argv)
{
    using std::numbers::pi;

    int n = 512;
    double reD = 2000.0, flowThroughs = 8.0, dRatio = 8.0;
    std::string prefix = "cube";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--out" && a + 1 < argc) prefix = argv[++a];
        else n = int(parseOr(argv[a], n));
    }
    if (!gpu::deviceAvailable())
    {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }

    const double L = 2 * pi, D = L / dRatio, uInf = 1.0, nu = uInf * D / reD;
    const auto g = Grid::make(n, n, n, L, L, L);
    const double cx = L / 6.0;
    const CubeMask cube{Real(cx - D / 2), Real(cx + D / 2),
                        Real(L / 2 - D / 2), Real(L / 2 + D / 2),
                        Real(L / 2 - D / 2), Real(L / 2 + D / 2)};
    const double xf0 = 0.85 * L, sigmaFringe = 40.0;
    std::printf("# cube n=%d Re_D=%g nu=%.4e D=%.4f (%d cells) U=1 T=%g flow-throughs\n",
                n, reD, nu, D, int(D / g.hx + 0.5), flowThroughs);

    // Uniform stream + small noise to break the wake's symmetry.
    auto U = VectorField::zeros(g);
    std::mt19937 rng(7);
    std::normal_distribution<double> pert(0.0, 0.02);
    for (auto& x : U.u) x = uInf + pert(rng);
    for (auto& x : U.v) x = pert(rng);
    for (auto& x : U.w) x = pert(rng);

    auto ts = gpu::TimeStepper::make(g, {nu, 0.9});
    auto D_ = gpu::DeviceField::make(g.size());
    auto R = gpu::DeviceField::make(g.size());
    auto rhs = gpu::DeviceScalar::make(g.size());
    auto p = gpu::DeviceScalar::make(g.size());
    auto poisson = gpu::DevicePoissonSolver::make(gpu::GpuGrid::from(g));
    auto scr = gpu::DeviceProjectionScratch::make(g.size());
    if (!ts || !D_ || !R || !rhs || !p || !poisson || !scr)
    {
        std::fprintf(stderr, "GPU init failed\n");
        return 1;
    }
    (void)gpu::uploadField(U, *D_);
    const auto gg = ts->grid();
    auto& red = ts->reduceScratch();

    const double tEnd = flowThroughs * L / uInf, frameDt = 0.2;
    const double vol = L * L * L, qRef = 0.5 * uInf * uInf * D * D;
    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,K,epsVisc,cd,cfl,maxDiv\n";
    SliceWriter out(prefix);

    double t = 0.0, dt = 1e-3, nextFrame = 0.0;
    long steps = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < tEnd)
    {
        const bool sample = steps % 10 == 0;
        double cd = 0.0;
        if (sample)
        {
            const auto m0 = gpu::meanVelocity(gg, *D_, red);
            maskCube(gg, cube, *D_);
            const auto m1 = gpu::meanVelocity(gg, *D_, red);
            cd = vol * (m0.x - m1.x) / dt / qRef; // IBM momentum removal == drag
        }
        else
            maskCube(gg, cube, *D_);
        fringe(gg, Real(xf0), Real(L), Real(uInf), Real(std::min(1.0, sigmaFringe * dt)),
               *D_);
        gpu::project(gg, *poisson, Real(1), *D_, *scr);

        if (t >= nextFrame)
        {
            writeFrame(gg, *D_, *R, *rhs, *p, *poisson, out);
            nextFrame += frameDt;
        }
        if (sample)
        {
            const double K = gpu::kineticEnergy(gg, *D_, red);
            const double eps = nu * gpu::velocityGradientSq(gg, *D_, red);
            const double cfl = dt * gpu::maxCourantPerDt(gg, *D_, red);
            const double maxDiv = gpu::maxAbsDivergence(gg, *D_, red);
            stats << t << ',' << dt << ',' << K << ',' << eps << ',' << cd << ','
                  << cfl << ',' << maxDiv << '\n';
            if (steps % 1000 == 0)
            {
                stats.flush();
                std::printf("t=%.2f/%.2f cd=%.3f K=%.4f\n", t, tEnd, cd, K);
                std::fflush(stdout);
            }
        }

        dt = ts->suggestDt(*D_, tEnd - t);
        ts->step(*D_, dt);
        t += dt;
        ++steps;
    }
    writeFrame(gg, *D_, *R, *rhs, *p, *poisson, out);
    (void)hipDeviceSynchronize();
    const double wallMs = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - wall0).count();

    std::ofstream meta(prefix + ".json");
    meta << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frames\":" << out.frames
         << ",\"frameDt\":" << frameDt << ",\"D\":" << D << ",\"reD\":" << reD
         << ",\"tEnd\":" << tEnd << ",\"msPerStep\":" << wallMs / double(steps)
         << "}\n";
    std::printf("done: %ld steps, %d frames, %.2f ms/step (incl. IBM + diagnostics)\n",
                steps, out.frames, wallMs / double(steps));
    return 0;
}
