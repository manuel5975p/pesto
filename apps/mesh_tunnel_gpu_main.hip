// Flow around an arbitrary triangle mesh (STL / glTF / GLB) in the numerical
// flow channel (GPU): the sphere_tunnel_gpu pipeline with the analytic body
// replaced by a MeshSdf -- BVH closest-triangle signed distance with
// pseudonormal (or --robust winding-number) sign -- fed through the same
// supersampled cut-cell build, sanitizeCut, closeBoxBoundary, ghost-fill
// sharp viscous wall, and aperture-pressure + friction drag/lift tallies.
// The grid carries a one-cell ghost ring: interior is n^3 cells.
// --open replaces the channel with a fully periodic open far field (fringe-
// region method): no closeBoxBoundary, FFT-periodic pressure, and per stage a
// cosine-ramped sponge band along every x/y/z boundary relaxing u toward the
// free stream.
// Usage: mesh_tunnel_gpu model.{stl,gltf,glb} [n] [--re ReD] [--dr DRatio]
//          [--scale s (0 = fit max extent to L/DRatio)] [--pitch deg]
//          [--yaw deg] [--robust] [--open] [--ft flowThroughs] [--fdt frameDt]
//          [--tol rms] [--out prefix]
#include "dns/geometry.hpp"
#include "dns/meshsdf.hpp"
#include "dns/trimesh.hpp"

#include "dns/gpu/cutpoisson.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
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

bool hasMeshExt(std::string_view s)
{
    auto ends = [&](std::string_view e) {
        return s.size() > e.size()
            && s.substr(s.size() - e.size()) == e;
    };
    return ends(".stl") || ends(".gltf") || ends(".glb");
}

// Ghost-ring boundary conditions, applied after maskSolidFaces zeroed every
// closed face. cdt = Uc*dt/hx drives the convective outlet from the stage
// input state (pass cdt = 0 with in == out aliased for a pure refresh).
//   inlet   (i<=1):        u = uInf; v,w antimirrored so they vanish on x=hx
//   outlet  (i==nx-1):     convective advection out of the domain
//   walls   (j,k rings):   FREE-SLIP -- tangential mirrored, normal 0.
__global__ void kBcTunnel(gpu::GpuGrid g, Real uInf, Real cdt, const Real* inu,
                          const Real* inv, const Real* inw, Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    const int nx = g.nx, ny = g.ny, nz = g.nz;
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

// Open far-field fringe sponge (--open): relax the stage velocity toward the
// free stream over cosine-ramped bands along every x/y/z domain boundary.
// r = max over axes of normalized band depth (0 at the interior band edge,
// 1 on the boundary); sig = sigMax*(1-cos(pi r))/2. The body never reaches
// the bands, so no solid face is touched.
__global__ void kFringe(gpu::GpuGrid g, int ns, Real sigMax, Real uInf, Real* u,
                        Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    auto depth = [ns](int q, int nq) {
        if (q < ns) return Real(ns - q) / Real(ns);
        if (q >= nq - ns) return Real(q - (nq - ns) + 1) / Real(ns);
        return Real(0);
    };
    const Real r =
        std::max(depth(i, g.nx), std::max(depth(j, g.ny), depth(k, g.nz)));
    if (r <= 0) return;
    const Real sig = sigMax * Real(0.5) * (Real(1) - cosf(Real(M_PI) * r));
    u[c] += sig * (uInf - u[c]);
    v[c] -= sig * v[c];
    w[c] -= sig * w[c];
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
// the fluid only through links from open faces to closed faces holding
// ghost-fill mirror values; each link drains nu*(q - q_ghost)*A/d.
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
    double tol = 1e-4, scaleArg = 0.0, pitchDeg = 0.0, yawDeg = 0.0;
    bool robust = false, dumpVol = false;
    bool open = false; // fully periodic open far field (fringe sponge)
    double volDt = 0.75;
    std::string meshPath, prefix = "mesh_tunnel_gpu";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--scale") scaleArg = next(scaleArg);
        else if (s == "--pitch") pitchDeg = next(pitchDeg);
        else if (s == "--yaw") yawDeg = next(yawDeg);
        else if (s == "--robust") robust = true;
        else if (s == "--open") open = true;
        else if (s == "--vol") dumpVol = true;
        else if (s == "--voldt") { dumpVol = true; volDt = next(volDt); }
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--fdt") frameDt = next(frameDt);
        else if (s == "--tol") tol = next(tol);
        else if (s == "--out" && a + 1 < argc) prefix = argv[++a];
        else if (hasMeshExt(s)) meshPath = s;
        else n = int(parseOr(argv[a], n));
    }
    if (meshPath.empty())
    {
        std::fprintf(stderr,
                     "usage: mesh_tunnel_gpu model.{stl,gltf,glb} [n] [--re ReD] "
                     "[--dr DRatio] [--scale s] [--pitch deg] [--yaw deg] "
                     "[--robust] [--open] [--ft T] [--fdt dt] [--tol rms] "
                     "[--out prefix]\n");
        return 1;
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

    // Body placement: same off-node domain point as the sphere app; auto scale
    // fits the mesh's largest extent to D0 = L/dRatio unless --scale is given.
    const double D0 = L / dRatio;
    const std::array<double, 3> C{h + L / 3.0 + 0.31 * h, h + L / 2.0 + 0.17 * h,
                                  h + L / 2.0 + 0.23 * h};
    auto meshOr = loadMesh(meshPath);
    if (!meshOr)
    {
        std::fprintf(stderr, "%s\n", meshOr.error().c_str());
        return 1;
    }
    TriMesh mesh = std::move(*meshOr);
    const Aabb bb0 = bounds(mesh);
    const double ext0 = std::max({bb0.hi[0] - bb0.lo[0], bb0.hi[1] - bb0.lo[1],
                                  bb0.hi[2] - bb0.lo[2]});
    if (!(ext0 > 0))
    {
        std::fprintf(stderr, "degenerate mesh bounds\n");
        return 1;
    }
    const double scale = scaleArg > 0 ? scaleArg : D0 / ext0;
    transform(mesh, scale, pitchDeg * pi / 180.0, yawDeg * pi / 180.0, C);
    const Aabb bb = bounds(mesh);
    const double D = std::max({bb.hi[0] - bb.lo[0], bb.hi[1] - bb.lo[1],
                               bb.hi[2] - bb.lo[2]});
    const std::size_t nTris = mesh.tris.size();

    const auto sdfT0 = std::chrono::steady_clock::now();
    const MeshSdf sdf(std::move(mesh), robust);
    if (!sdf.watertight())
        std::printf("# WARNING: mesh not watertight%s\n",
                    robust ? "" : "; consider --robust");

    auto cutR = sampleCut(g, [&](double x, double y, double z) { return sdf(x, y, z); });
    if (!cutR)
    {
        std::fprintf(stderr, "cut geometry: %s\n", cutR.error().c_str());
        return 3;
    }
    auto cut = std::move(*cutR);
    const auto st = sanitizeCut(g, cut); // before the ring: body cells only

    // Projected frontal area from the apertures: a (j,k) stream column is
    // blocked by 1 - min_i ax (exact for convex bodies, honest for the rest).
    double aFront = 0.0;
    for (int k = 1; k <= n; ++k)
        for (int j = 1; j <= n; ++j)
        {
            double amin = 1.0;
            for (int i = 2; i <= n; ++i)
                amin = std::min(amin, cut.ax[g.idx(i, j, k)]);
            aFront += (1.0 - amin) * g.hy * g.hz;
        }
    if (!open) closeBoxBoundary(g, cut); // open: ring cells stay periodic fluid
    const double geomSec = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - sdfT0).count();
    if (aFront <= 0)
    {
        std::fprintf(stderr, "mesh produced no solid cells (too small?)\n");
        return 1;
    }

    const double nu = uInf * D / reD;
    std::printf("# mesh_tunnel_gpu %s n=%d Re_D=%g nu=%.4e D=%.4f (%.1f cells)"
                " open=%d\n",
                meshPath.c_str(), n, reD, nu, D, D / h, int(open));
    std::printf("# mesh: tris=%zu scale=%.4g pitch=%g yaw=%g robust=%d "
                "watertight=%d Afront=%.4f geomBuild=%.1fs\n",
                nTris, scale, pitchDeg, yawDeg, int(robust),
                int(sdf.watertight()), aFront, geomSec);
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed,
                st.minLambda, st.minAperture);

    // Clean laminar free stream; deterministic trip one D behind the body.
    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = uInf;
    for (int k = 0; k < nx; ++k)
        for (int j = 0; j < nx; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > bb.hi[0] && x < bb.hi[0] + D && std::abs(y - C[1]) < 0.5 * D)
                    U.v[g.idx(i, j, k)] = 0.05 * uInf;
            }

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    // Sharp viscous wall: diffusion reads a ghost-filled copy (mirror values,
    // no-slip at the true sub-cell surface); convection keeps masked zeros.
    auto dGhost = gpu::DeviceGhostFill::make(
        g, cut, [&](double x, double y, double z) { return sdf(x, y, z); });
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
    auto poisson = gpu::DeviceCutPoissonSolver::make(
        gg, *dCut, 500, tol,
        open ? gpu::PoissonBc::Periodic : gpu::PoissonBc::NeumannBox);
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
    // Fringe band: ~10% of the domain per side, cosine-ramped, sigMax 0.5.
    const int nsFringe = std::clamp(int(0.1 * nx), 8, nx / 2 - 1);
    auto fringeApply = [&](gpu::DeviceField& out) {
        kFringe<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
            gg, nsFringe, Real(0.5), Real(uInf), out.u.data(), out.v.data(),
            out.w.data());
        gpu::checkLastKernel("fringe");
    };
    if (!open) bcApply(*dU, *dU, 0.0); // seed ghosts/prescribed faces

    // One SSP-RK3 stage with channel BCs and the ghost-flux projection.
    // Open: fringe sponge instead of BCs; periodic divergence sums to zero, so
    // the projection RHS is compatible without the in/out flux terms.
    int pcgIters = 0;
    auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double a,
                     double dt) {
        (void)dGhost->fillFrom(in, *dG); // sub-cell wall for the viscous term
        gpu::fusedRhsStage(gg, Real(nu), Real(0), 0, 0, 0, Real(a), *dU0,
                           Real(1.0 - a), in, *dG, Real(dt), out);
        gpu::maskSolidFaces(gg, *dCut, out);
        if (open) fringeApply(out);
        else
        {
            bcApply(in, out, uInf * dt / h);
            fluxCorrect(out);
        }
        gpu::divergenceCut(gg, *dCut, out, scr->div, Real(1.0 / dt));
        if (!open)
        {
            kInOutFluxRhs<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                gg, Real(1.0 / dt), Real(uInf), out.u.data(), scr->div.data());
            gpu::checkLastKernel("inOutFluxRhs");
        }
        pcgIters += poisson->solve(scr->div, scr->phi, true);
        gpu::subtractGradientCut(gg, *dCut, scr->phi, Real(dt), out);
        if (!open) bcApply(out, out, 0.0); // refresh mirrors after the correction
    };

    // Bounding box (grid indices) for the body-force reduction: the body plus
    // margin, strictly inside the interior so boundary faces never count.
    const int bi0 = std::max(2, int(bb.lo[0] / h) - 2);
    const int bi1 = std::min(nx - 3, int(bb.hi[0] / h) + 3);
    const int bj0 = std::max(2, int(bb.lo[1] / h) - 2);
    const int bj1 = std::min(nx - 3, int(bb.hi[1] / h) + 3);
    const int bk0 = std::max(2, int(bb.lo[2] / h) - 2);
    const int bk1 = std::min(nx - 3, int(bb.hi[2] / h) + 3);

    const double tEnd = flowThroughs * L / uInf;
    const double qRef = 0.5 * uInf * uInf * aFront; // measured frontal area
    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters\n";

    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frameDt\":" << frameDt
        << ",\"D\":" << D << ",\"reD\":" << reD << ",\"cx\":" << C[0] / h - 1
        << ",\"cy\":" << C[1] / h - 1 << ",\"R\":" << 0.5 * D / h
        << ",\"mesh\":\"" << meshPath << "\",\"open\":" << int(open) << "}\n";

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

    // Periodic 3D dump for the animated viewer: cell-centred velocity per
    // volume frame appended to _vol.bin, and lambda written once. All n^3 in
    // grid coords, so a marching-cubes body and flow isosurfaces align exactly.
    std::ofstream volFile;
    int volFrames = 0;
    double nextVol = 0.0;
    VectorField volHost = VectorField::zeros(g);
    std::vector<float> volBuf(std::size_t(n) * n * n * 3);
    if (dumpVol)
    {
        volFile.open(prefix + "_vol.bin", std::ios::binary);
        std::ofstream lam(prefix + "_lam.bin", std::ios::binary);
        std::vector<float> lbuf(std::size_t(n) * n * n);
        std::size_t q = 0;
        for (int k = 1; k <= n; ++k)
            for (int j = 1; j <= n; ++j)
                for (int i = 1; i <= n; ++i)
                    lbuf[q++] = float(cut.lambda[g.idx(i, j, k)]);
        lam.write(reinterpret_cast<const char*>(lbuf.data()),
                  std::streamsize(lbuf.size() * sizeof(float)));
    }
    auto writeVolume = [&]() {
        (void)gpu::downloadField(*dU, volHost);
        auto uf = [&](int i, int j, int k) { return double(volHost.u[g.idx(i, j, k)]); };
        auto vf = [&](int i, int j, int k) { return double(volHost.v[g.idx(i, j, k)]); };
        auto wf = [&](int i, int j, int k) { return double(volHost.w[g.idx(i, j, k)]); };
        std::size_t p = 0;
        for (int k = 1; k <= n; ++k)
            for (int j = 1; j <= n; ++j)
                for (int i = 1; i <= n; ++i)
                {
                    volBuf[p++] = float(0.5 * (uf(i, j, k) + uf(i + 1, j, k)));
                    volBuf[p++] = float(0.5 * (vf(i, j, k) + vf(i, j + 1, k)));
                    volBuf[p++] = float(0.5 * (wf(i, j, k) + wf(i, j, k + 1)));
                }
        volFile.write(reinterpret_cast<const char*>(volBuf.data()),
                      std::streamsize(volBuf.size() * sizeof(float)));
        ++volFrames;
    };

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
            // advances a full dt, so phi is the static pressure.
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
        if (dumpVol && t >= nextVol) { writeVolume(); nextVol += volDt; }

        if (steps % 20 == 0)
        {
            const double K = gpu::kineticEnergy(gg, *dU, *red);
            const double cfl = dt * conv;

            // Continuity residual; tunnel adds the prescribed in/out fluxes,
            // open is plain periodic divergence.
            gpu::divergenceCut(gg, *dCut, *dU, scr->div, Real(1));
            if (!open)
            {
                kInOutFluxRhs<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                    gg, Real(1), Real(uInf), dU->u.data(), scr->div.data());
                gpu::checkLastKernel("inOutFluxRhs");
            }
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
        << ",\"cx\":" << C[0] / h - 1 << ",\"cy\":" << C[1] / h - 1
        << ",\"R\":" << 0.5 * D / h << ",\"mesh\":\"" << meshPath
        << "\",\"volFrames\":" << volFrames << ",\"n\":" << n
        << ",\"open\":" << int(open)
        << ",\"msPerStep\":" << wallMs / double(steps) << "}\n";
    std::printf("done: %ld steps, %d frames, %d volFrames, %.2f ms/step\n",
                steps, frames, volFrames, wallMs / double(steps));
    return 0;
}
