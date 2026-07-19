// Movie extractor (GPU): advances a run and dumps mid-plane slices of speed,
// vorticity magnitude and pressure as float32 frames for offline rendering.
// Usage: movie tgv [n] [--out prefix]   (Re 1600, t in [0,20], 200 frames)
//        movie hit [n] [--out prefix]   (spin 5 Te, then 3 Te at 30 frames/Te)
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/poisson.hpp"
#include "dns/gpu/timestepper.hpp"
#include "dns/hit_ic.hpp"
#include "dns/taylor_green.hpp"
#include "dns/units.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

namespace {

using dns::gpu::Real;

// One z-slab (nx*ny floats, contiguous: i fastest, k slowest).
hipError_t downloadSlab(const dns::gpu::DeviceScalar& q, const dns::gpu::GpuGrid& g,
                        int k, std::vector<Real>& out)
{
    const std::size_t slab = std::size_t(g.nx) * g.ny;
    out.resize(slab);
    return hipMemcpy(out.data(), q.data() + slab * std::size_t(k),
                     slab * sizeof(Real), hipMemcpyDeviceToHost);
}

struct SliceWriter
{
    std::ofstream speed, vort, pres;
    int frames = 0;

    explicit SliceWriter(const std::string& prefix)
        : speed(prefix + "_speed.bin", std::ios::binary),
          vort(prefix + "_vort.bin", std::ios::binary),
          pres(prefix + "_p.bin", std::ios::binary)
    {
    }

    void append(std::ofstream& f, const std::vector<float>& q)
    {
        f.write(reinterpret_cast<const char*>(q.data()),
                std::streamsize(q.size() * sizeof(float)));
    }
};

// Extract mid-plane slices; pressure from lap(p) = div(conv(U)).
void writeFrame(const dns::gpu::GpuGrid& g, const dns::gpu::DeviceField& U,
                dns::gpu::DeviceField& R, dns::gpu::DeviceScalar& rhs,
                dns::gpu::DeviceScalar& p, dns::gpu::DevicePoissonSolver& poisson,
                SliceWriter& out)
{
    const int kc = g.nz / 2, nx = g.nx, ny = g.ny;
    const std::size_t slab = std::size_t(nx) * ny;

    dns::gpu::convection(g, U, R);
    dns::gpu::divergence(g, R, rhs, Real(-1)); // lap(p) = -div((u.grad)u)
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
        return double(q[std::size_t(dns::gpu::wrap(j, ny)) * nx
                        + std::size_t(dns::gpu::wrap(i, nx))]);
    };

    std::vector<float> speed(slab), vort(slab);
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
        }

    out.append(out.speed, speed);
    out.append(out.vort, vort);
    out.append(out.pres, std::vector<float>(pc.begin(), pc.end()));
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
    using namespace dns;
    using std::numbers::pi;

    if (argc < 2 || (std::strcmp(argv[1], "tgv") && std::strcmp(argv[1], "hit")))
    {
        std::fprintf(stderr, "usage: movie (tgv|hit) [n] [--out prefix]\n");
        return 2;
    }
    const bool tgv = std::strcmp(argv[1], "tgv") == 0;
    int n = 256;
    std::string prefix = std::string("movie_") + argv[1];
    for (int a = 2; a < argc; ++a)
    {
        if (!std::strcmp(argv[a], "--out") && a + 1 < argc) prefix = argv[++a];
        else n = int(parseOr(argv[a], n));
    }
    if (!gpu::deviceAvailable())
    {
        std::fprintf(stderr, "no HIP device\n");
        return 1;
    }

    const auto g = Grid::make(n, n, n, 2 * pi, 2 * pi, 2 * pi);
    auto U = VectorField::zeros(g);
    const double nu = tgv ? 1.0 / 1600.0 : HitDesign::make(n, 1.5e-5, 60.0).nuBox();
    if (tgv) taylorGreen3D(g, U);
    else hitInitialCondition(g, 1.0, 1, U);

    auto ts = gpu::TimeStepper::make(g, {nu, 0.9});
    auto D = gpu::DeviceField::make(g.size());
    auto R = gpu::DeviceField::make(g.size());
    auto rhs = gpu::DeviceScalar::make(g.size());
    auto p = gpu::DeviceScalar::make(g.size());
    auto poisson = gpu::DevicePoissonSolver::make(gpu::GpuGrid::from(g));
    if (!ts || !D || !R || !rhs || !p || !poisson)
    {
        std::fprintf(stderr, "GPU init failed\n");
        return 1;
    }
    (void)gpu::uploadField(U, *D);
    const auto gg = ts->grid();
    auto& red = ts->reduceScratch();

    const double tE = 0.19 * 2 * pi;                       // HIT eddy turnover
    const double tStart = tgv ? 0.0 : 5.0 * tE;            // HIT: skip spin-up
    const double tEnd = tgv ? 20.0 : tStart + 3.0 * tE;
    const double frameDt = tgv ? 0.1 : tE / 30.0;
    const LinearForcing forcing{HitDesign::make(n, 1.5e-5, 60.0).epsBox(), 1.5 / 5.0};

    SliceWriter out(prefix);
    double t = 0.0, nextFrame = tStart;
    long steps = 0;
    while (t < tEnd)
    {
        if (!tgv)
        {
            const auto mean = gpu::meanVelocity(gg, *D, red);
            const double kPrime = gpu::fluctuationEnergy(gg, *D, red);
            ts->setForcing({forcing.epsTarget / (2.0 * std::max(kPrime, forcing.kFloor)),
                            {mean.x, mean.y, mean.z}});
            if (steps % 100 == 0)
                gpu::subtractMean(gg, Real(mean.x), Real(mean.y), Real(mean.z), *D);
        }
        if (t >= nextFrame)
        {
            writeFrame(gg, *D, *R, *rhs, *p, *poisson, out);
            nextFrame += frameDt;
        }
        const double dt = ts->suggestDt(*D, tEnd - t);
        ts->step(*D, dt);
        t += dt;
        ++steps;
    }
    writeFrame(gg, *D, *R, *rhs, *p, *poisson, out);

    std::ofstream meta(prefix + ".json");
    meta << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frames\":" << out.frames
         << ",\"frameDt\":" << frameDt << ",\"mode\":\"" << argv[1] << "\"}\n";
    std::printf("%s: %d frames, %ld steps\n", prefix.c_str(), out.frames, steps);
    return 0;
}
