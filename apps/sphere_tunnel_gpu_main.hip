// Flow around a sphere in a true numerical flow channel (GPU): no periodic
// directions, so no periodic-image confinement. Laminar Dirichlet inflow,
// convective outflow with exact mass-flux correction, OPEN free-stream lateral
// walls (u = uInf, v = w = 0 -- not mirrors, which would re-create the array),
// and the sanitized sphere cut-cell body. Pressure Poisson is PCG
// preconditioned by the exact DCT (NeumannBox) inverse Laplacian; drag/lift
// come from the discrete aperture-pressure surface integral plus the
// masked-link skin-friction tally.
// The grid carries a one-cell ghost ring: interior is n^3 cells.
// Usage: sphere_tunnel_gpu [n] [--re ReD] [--dr DRatio] [--ft flowThroughs]
//                          [--golf nDimples (0 = smooth sphere)]
//                          [--fdt frameDt] [--tol rms] [--out prefix]
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <array>
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
//   walls   (j,k rings):   freestream far-field -- u = uInf, v = w = 0 (open,
//     no mirror; dp/dn = 0 stays correct under a prescribed far field).
__global__ void kBcTunnel(gpu::GpuGrid g, Real uInf, Real cdt, const Real* inu,
                          const Real* inv, const Real* inw, Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    // FREE-SLIP lateral walls (Neumann): tangential components mirrored -- the
    // wall generates no vorticity -- normal component 0 on the wall face,
    // antimirrored in the ghost. Pressure is already Neumann (closed apertures
    // + DCT). Dirichlet u = uInf here paints a vortex sheet along the wall.
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

// Friction force the wall Laplacian links exert on the body: momentum leaves
// the fluid only through links from open faces to closed faces. Closed faces
// hold ghost-fill mirror values, so each link drains nu*(q - q_ghost)*A/d --
// the viscous flux the momentum update actually applied, with the wall at its
// true sub-cell position via the ghost value.
__global__ void kBodyFriction(gpu::GpuGrid g, Real nu, const Real* ax,
                              const Real* ay, const Real* u, const Real* v,
                              int i0, int i1, int j0, int j1, int k0, int k1,
                              Real* fx, Real* fy)
{
    constexpr Real openTol = Real(CutGeometry::openTol);
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    fx[c] = 0;
    fy[c] = 0;
    if (i < i0 || i > i1 || j < j0 || j > j1 || k < k0 || k > k1) return;
    const Real lx = g.hy * g.hz / g.hx; // link conductance A/d per direction
    const Real ly = g.hx * g.hz / g.hy;
    const Real lz = g.hx * g.hy / g.hz;
    auto closedLinks = [&](const Real* a, const Real* q) {
        Real s = 0;
        auto link = [&](std::size_t nb, Real l) {
            if (a[nb] <= openTol) s += l * (q[c] - q[nb]);
        };
        link(gidx(g, i - 1, j, k), lx);
        link(gidx(g, i + 1, j, k), lx);
        link(gidx(g, i, j - 1, k), ly);
        link(gidx(g, i, j + 1, k), ly);
        link(gidx(g, i, j, k - 1), lz);
        link(gidx(g, i, j, k + 1), lz);
        return s;
    };
    if (ax[c] > openTol) fx[c] = nu * closedLinks(ax, u);
    if (ay[c] > openTol) fy[c] = nu * closedLinks(ay, v);
}

// --- Golf ball: sphere minus a Fibonacci lattice of spherical-cap dimples ----
// Dimples are sized in grid units so they stay resolvable: footprint radius
// 5h, depth 2.5h (a real golf ball's ~0.01D dimples would vanish on any
// affordable grid). sdf < 0 in the solid; true distance away from dimple rims.
struct GolfBall
{
    Sphere s;
    std::vector<std::array<double, 3>> q; // dimple-sphere centers
    double rd;    // dimple-sphere radius
    double depth; // dimple depth (radial recess)

    static GolfBall make(const Sphere& s, int nDimples, double h)
    {
        using std::numbers::pi;
        const double depth = 2.5 * h, a = 5.0 * h;
        GolfBall b{s, {}, (a * a + depth * depth) / (2.0 * depth), depth};
        const double qr = s.r + b.rd - depth;
        const double ga = pi * (3.0 - std::sqrt(5.0)); // golden angle
        for (int i = 0; i < nDimples; ++i)
        {
            const double z = 1.0 - 2.0 * (i + 0.5) / nDimples;
            const double rho = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double th = ga * i;
            b.q.push_back({s.cx + qr * rho * std::cos(th),
                           s.cy + qr * rho * std::sin(th), s.cz + qr * z});
        }
        return b;
    }

    // Solid = sphere \ union(dimple balls): sdf = max(sphere, -dimples).
    double sdf(double x, double y, double z) const
    {
        const double dx = x - s.cx, dy = y - s.cy, dz = z - s.cz;
        double phi = std::sqrt(dx * dx + dy * dy + dz * dz) - s.r;
        for (const auto& c : q)
        {
            const double ex = x - c[0], ey = y - c[1], ez = z - c[2];
            phi = std::max(phi, rd - std::sqrt(ex * ex + ey * ey + ez * ez));
        }
        return phi;
    }
};

// Cut geometry by supersampling the sdf: lambda from 4^3 cell samples,
// apertures from 4^2 face samples; only the shell band around the dimpled
// surface is sampled (elsewhere trivially fluid/solid). Sampling error ~1/16
// of a face -- far above openTol, absorbed by sanitizeCut and the first-order
// boundary scheme (cf. the eye app's supersampled lambda).
CutGeometry makeGolfCut(const Grid& g, const GolfBall& b)
{
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    const double halfDiag =
        0.5 * std::sqrt(g.hx * g.hx + g.hy * g.hy + g.hz * g.hz);
    const double lo = b.s.r - b.depth - halfDiag, hi = b.s.r + halfDiag;
    auto dist = [&](double x, double y, double z) {
        const double dx = x - b.s.cx, dy = y - b.s.cy, dz = z - b.s.cz;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };
    // Fluid fraction of an axis box by an sx*sy*sz sample lattice.
    auto frac = [&](double x0, double y0, double z0, double ex, double ey,
                    double ez, int sx, int sy, int sz) {
        int m = 0;
        for (int c = 0; c < sz; ++c)
            for (int bb = 0; bb < sy; ++bb)
                for (int a = 0; a < sx; ++a)
                    if (b.sdf(x0 + (a + 0.5) * ex / sx, y0 + (bb + 0.5) * ey / sy,
                              z0 + (c + 0.5) * ez / sz) > 0.0)
                        ++m;
        return double(m) / (sx * sy * sz);
    };
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
            {
                const auto id = g.idx(i, j, k);
                const double x0 = i * g.hx, y0 = j * g.hy, z0 = k * g.hz;
                auto classify = [&](double d, auto sample) {
                    return d >= hi ? 1.0 : d <= lo ? 0.0 : sample();
                };
                cut.lambda[id] = classify(
                    dist(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy, z0 + 0.5 * g.hz),
                    [&] { return frac(x0, y0, z0, g.hx, g.hy, g.hz, 4, 4, 4); });
                cut.ax[id] = classify(
                    dist(x0, y0 + 0.5 * g.hy, z0 + 0.5 * g.hz),
                    [&] { return frac(x0, y0, z0, 0, g.hy, g.hz, 1, 4, 4); });
                cut.ay[id] = classify(
                    dist(x0 + 0.5 * g.hx, y0, z0 + 0.5 * g.hz),
                    [&] { return frac(x0, y0, z0, g.hx, 0, g.hz, 4, 1, 4); });
                cut.az[id] = classify(
                    dist(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy, z0),
                    [&] { return frac(x0, y0, z0, g.hx, g.hy, 0, 4, 4, 1); });
            }
    return cut;
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

    int n = 96;
    double reD = 300.0, flowThroughs = 6.0, dRatio = 8.0, frameDt = 0.15;
    double tol = 1e-4;
    int golf = 0; // number of dimples; 0 = smooth sphere
    std::string prefix = "sphere_tunnel_gpu";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--golf") golf = int(next(0));
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
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

    // Interior n^3 cells of size h; one ghost cell on every side.
    const double L = 2 * pi, uInf = 1.0;
    const double h = L / n;
    const int nx = n + 2;
    const auto g = Grid::make(nx, nx, nx, nx * h, nx * h, nx * h);

    // Body in grid coordinates: physical position + the ghost offset h.
    // Off-node on purpose (arbitrary cuts); sanitizeCut handles the tiny cells.
    const double D = L / dRatio;
    const Sphere sph{h + L / 3.0 + 0.31 * h, h + L / 2.0 + 0.17 * h,
                     h + L / 2.0 + 0.23 * h, D / 2.0};
    const GolfBall ball = GolfBall::make(sph, std::max(golf, 1), h);
    auto cut = golf > 0 ? makeGolfCut(g, ball) : CutGeometry::make(g, sph);
    const auto st = sanitizeCut(g, cut); // before the ring: body cells only
    closeBoxBoundary(g, cut);

    const double nu = uInf * D / reD;
    std::printf("# sphere_tunnel_gpu n=%d Re_D=%g nu=%.4e D=%.4f (%.1f cells)\n",
                n, reD, nu, D, D / h);
    if (golf > 0)
        std::printf("# golf: %d dimples, footprint 5h, depth 2.5h (%.3f D)\n",
                    golf, ball.depth / D);
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed,
                st.minLambda, st.minAperture);

    // Clean laminar free stream; deterministic trip one D behind the sphere.
    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = uInf;
    for (int k = 0; k < nx; ++k)
        for (int j = 0; j < nx; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > sph.cx + sph.r && x < sph.cx + sph.r + D
                    && std::abs(y - sph.cy) < sph.r)
                    U.v[g.idx(i, j, k)] = 0.05 * uInf;
            }

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    // Sharp viscous wall: diffusion reads a ghost-filled copy (mirror values,
    // no-slip at the true sub-cell surface); convection keeps masked zeros.
    auto dGhost = gpu::DeviceGhostFill::make(
        g, cut, [&](double x, double y, double z) {
            if (golf > 0) return ball.sdf(x, y, z);
            const double dx = x - sph.cx, dy = y - sph.cy, dz = z - sph.cz;
            return std::sqrt(dx * dx + dy * dy + dz * dz) - sph.r;
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

    const double planeArea = double(n) * n * h * h;
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

    // One SSP-RK3 stage with channel BCs and the ghost-flux projection.
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
    const int bi0 = std::max(2, int((sph.cx - sph.r) / h) - 2);
    const int bi1 = std::min(nx - 3, int((sph.cx + sph.r) / h) + 3);
    const int bj0 = std::max(2, int((sph.cy - sph.r) / h) - 2);
    const int bj1 = std::min(nx - 3, int((sph.cy + sph.r) / h) + 3);
    const int bk0 = std::max(2, int((sph.cz - sph.r) / h) - 2);
    const int bk1 = std::min(nx - 3, int((sph.cz + sph.r) / h) + 3);

    const double tEnd = flowThroughs * L / uInf;
    const double qRef = 0.5 * uInf * uInf * pi * D * D / 4.0; // frontal disk
    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters\n";

    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frameDt\":" << frameDt
        << ",\"D\":" << D << ",\"reD\":" << reD << ",\"cx\":" << sph.cx / h - 1
        << ",\"cy\":" << sph.cy / h - 1 << ",\"R\":" << sph.r / h
        << ",\"golf\":" << golf << "}\n";

    // Interior mid-z slices: signed z-vorticity at cell corners and velocity
    // magnitude at cell centres (top view).
    const int kc = nx / 2;
    std::ofstream vortFile(prefix + "_vort.bin", std::ios::binary);
    std::ofstream speedFile(prefix + "_speed.bin", std::ios::binary);
    int frames = 0;
    double nextFrame = 0.0;
    std::vector<Real> u0, v0, w0, w1;
    std::vector<float> frame(std::size_t(n) * n);
    auto writeFrame = [&]() {
        (void)downloadSlab(dU->u, gg, kc, u0);
        (void)downloadSlab(dU->v, gg, kc, v0);
        (void)downloadSlab(dU->w, gg, kc, w0);
        (void)downloadSlab(dU->w, gg, kc + 1, w1);
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
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
            {
                const int ii = i + 1, jj = j + 1;
                const double uc = 0.5 * (at(u0, ii, jj) + at(u0, ii + 1, jj));
                const double vc = 0.5 * (at(v0, ii, jj) + at(v0, ii, jj + 1));
                const double wc = 0.5 * (at(w0, ii, jj) + at(w1, ii, jj));
                frame[std::size_t(j) * n + i] =
                    float(std::sqrt(uc * uc + vc * vc + wc * wc));
            }
        speedFile.write(reinterpret_cast<const char*>(frame.data()),
                        std::streamsize(frame.size() * sizeof(float)));
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
    }

    double t = 0.0, dt = 1e-3;
    long steps = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < tEnd)
    {
        const double conv = gpu::maxCourantPerDt(gg, *dU, *red);
        const double diff = nu > 0 ? 6.0 * nu / (h * h) : 0.0;
        const double dtStab = 0.8 / std::max(conv, diff);
        dt = std::min(tEnd - t, dtStab);

        // Blowup forensics: if the CFL limiter (not the end-of-run clamp) has
        // collapsed dt, the field has diverged -- dump the argmax and stop.
        if (dtStab < 1e-4 && steps > 0)
        {
            auto host = VectorField::zeros(g);
            (void)gpu::downloadField(*dU, host);
            const Scalar* comp[3]{&host.u, &host.v, &host.w};
            const char* name[3]{"u", "v", "w"};
            for (int q = 0; q < 3; ++q)
            {
                std::size_t am = 0;
                double m = 0.0;
                for (std::size_t c = 0; c < g.size(); ++c)
                    if (std::abs((*comp[q])[c]) > m) { m = std::abs((*comp[q])[c]); am = c; }
                const int i = int(am % nx), j = int((am / nx) % nx),
                          k = int(am / (std::size_t(nx) * nx));
                std::printf("BLOWUP t=%.4f max|%s|=%.3e at i=%d j=%d k=%d "
                            "(lam=%.3f)\n", t, name[q], m, i, j, k,
                            cut.lambda[am]);
            }
            std::fflush(stdout);
            return 2;
        }

        std::swap(*dU0, *dU);
        pcgIters = 0;
        stage(*dU0, *dU, 0.0, dt);
        double cd = 0.0, cl = 0.0, cdf = 0.0;
        if (steps % 20 == 0)
        {
            // Surface pressure integral from the FIRST stage's phi: that stage
            // advances a full dt, so phi is the static pressure. The later
            // stages' effective substeps are shorter (phi ~ (2/3)p on stage 3)
            // and would bias the force low.
            kBodyPressure<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                gg, dCut->lambda.data(), dCut->ax.data(), dCut->ay.data(),
                scr->phi.data(), bi0, bi1, bj0, bj1, bk0, bk1, dFx->data(),
                dFy->data());
            gpu::checkLastKernel("bodyPressure");
            cd = gpu::sumScalar(*dFx, *red) / qRef;
            cl = gpu::sumScalar(*dFy, *red) / qRef;

            // Skin friction from the time-t field (matches phi's time level),
            // through the ghost-filled view the viscous term actually saw.
            (void)dGhost->fillFrom(*dU0, *dG);
            kBodyFriction<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                gg, Real(nu), dCut->ax.data(), dCut->ay.data(), dG->u.data(),
                dG->v.data(), bi0, bi1, bj0, bj1, bk0, bk1, dFx->data(),
                dFy->data());
            gpu::checkLastKernel("bodyFriction");
            cdf = gpu::sumScalar(*dFx, *red) / qRef;
            cd += cdf;
            cl += gpu::sumScalar(*dFy, *red) / qRef;
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
                  << cdf << ',' << cfl << ',' << maxDiv << ',' << pcgIters << '\n';
            stats.flush();
            if (steps % 200 == 0)
            {
                std::printf("t=%.2f/%.2f cd=%.3f (cdf=%.3f) cl=%+.3f "
                            "K=%.1f maxDiv=%.2e pcg=%d\n",
                            t, tEnd, cd, cdf, cl, K, maxDiv, pcgIters);
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
        << ",\"cx\":" << sph.cx / h - 1 << ",\"cy\":" << sph.cy / h - 1
        << ",\"R\":" << sph.r / h << ",\"msPerStep\":" << wallMs / double(steps)
        << "}\n";
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n",
                steps, frames, wallMs / double(steps));
    return 0;
}
