// Flow around a z-aligned circular cylinder in a numerical flow channel (GPU),
// quasi-2D: only a few cells in z with FREE-SLIP z-walls (mirror u,v; w = 0),
// which are exact for a z-invariant flow -- the run is effectively a 2D
// circle simulation. --nz 1 (or 0) runs natively 2D instead: nz == 1, no z
// ghosts, the degenerate z axis stays periodic (exact for z-invariant flow).
// Laminar Dirichlet inflow, convective outflow with exact
// mass-flux correction, open free-stream lateral y-walls, sanitized cylinder
// cut-cell body. Pressure Poisson is PCG preconditioned by the exact DCT
// (NeumannBox) inverse Laplacian; drag/lift come from the discrete
// aperture-pressure surface integral plus the masked-link skin-friction tally
// (x/y links only: the free-slip z ghosts carry no viscous flux).
// The grid carries a one-cell ghost ring: interior is n x n x nzInt cells.
// --open replaces the channel with a fully periodic open far field (fringe-
// region method): no closeBoxBoundary, FFT-periodic pressure, and per stage a
// cosine-ramped sponge band along every x/y boundary relaxing u toward the
// free stream (z stays plain periodic -- the flow is z-invariant).
// Usage: cylinder_tunnel_gpu [n] [--nz nzInt | --nz 1 for native 2D]
//                            [--re ReD] [--dr DRatio] [--open]
//                            [--ft flowThroughs] [--fdt frameDt] [--tol rms] [--cfl c]
//                            [--les Cs (Smagorinsky, 2D only, 0 = off)]
//                            [--out prefix]
#include "dns/output.hpp"
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"
#include "dns/gpu/bodyforce.hpp"
#include "dns/gpu/tunnel.hpp"

#include <hip/hip_runtime.h>

#include <filesystem>

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

// --- Optional Smagorinsky LES (2D, --les Cs, off by default) ----------------
// nut = (Cs^2 hx hy) |S| lambda at cell centres from the MAC field; the eddy-
// diffusion substep adds b*dt * div(nut grad u) per RK stage in conservative
// flux form (face/corner-averaged nut). Explicitly stable: dt*nut/h^2 ~ Cs^2 *
// CFL << 1. Purely app-local so the shared inviscid-exact kernels stay untouched.
// NOTE: gradient form div(nut grad u), NOT symmetric stress div(2 nut S). Diff
// is dj(nut di uj): =0 for const nut + div u=0, nonzero for varying nut. Still
// flux-conservative and dissipative (nut|grad u|^2 >= 0); Cs is nominal, output
// qualitative per project LES stance (grid+Cs dependent, not converged).
// Cut-cell caveat: |S| uses central differences that reach into the solid where
// u is masked to 0, so raw nut PEAKS at the wall (no van Driest) and the flux
// form ignores apertures ax/ay (staircase body). Mitigation: scale nut by the
// fluid fraction lambda -- zero in solid, damped in cut cells; the eddy-flux
// coefficient is the cell/corner nut (fluid-fraction plane, not the ax/ay MAC
// plane), so lambda is the consistent open-area weight. Coarse, not exact.
// Ghost ring gets nut = 0: the wrap stencil would read across the box, and
// zero is the correct no-eddy-flux condition at the domain boundary.
__global__ void kSmagNut(gpu::GpuGrid g, Real cs2hh, const Real* u, const Real* v,
                         const Real* lam, Real* nut)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    if (i == 0 || i == g.nx - 1 || j == 0 || j == g.ny - 1)
    {
        nut[c] = 0; // ghost ring: no eddy flux through the domain boundary
        return;
    }
    const Real dudx = (u[gidx(g, i + 1, j, k)] - u[c]) / g.hx;
    const Real dvdy = (v[gidx(g, i, j + 1, k)] - v[c]) / g.hy;
    auto uc = [&](int a, int b) {
        return Real(0.5) * (u[gidx(g, a, b, k)] + u[gidx(g, a + 1, b, k)]);
    };
    auto vc = [&](int a, int b) {
        return Real(0.5) * (v[gidx(g, a, b, k)] + v[gidx(g, a, b + 1, k)]);
    };
    const Real dudy = (uc(i, j + 1) - uc(i, j - 1)) / (2 * g.hy);
    const Real dvdx = (vc(i + 1, j) - vc(i - 1, j)) / (2 * g.hx);
    const Real sxy = Real(0.5) * (dudy + dvdx);
    const Real smag =
        std::sqrt(2 * dudx * dudx + 2 * dvdy * dvdy + 4 * sxy * sxy);
    nut[c] = cs2hh * smag * lam[c];
}

__global__ void kEddyDiffuse(gpu::GpuGrid g, Real s, const Real* nut,
                             const Real* u, const Real* v, Real* ou, Real* ov)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    const Real ix = Real(1) / (g.hx * g.hx), iy = Real(1) / (g.hy * g.hy);
    auto n4 = [&](int a, int b) { // nut at the (a,b) cell corner
        return Real(0.25) * (nut[gidx(g, a, b, k)] + nut[gidx(g, a - 1, b, k)]
                             + nut[gidx(g, a, b - 1, k)] + nut[gidx(g, a - 1, b - 1, k)]);
    };
    { // u-face at (i,j): cells i and i-1 in x, corners (i,j) and (i,j+1) in y
        const std::size_t xp = gidx(g, i + 1, j, k), xm = gidx(g, i - 1, j, k);
        const std::size_t yp = gidx(g, i, j + 1, k), ym = gidx(g, i, j - 1, k);
        const Real lap =
            ix * (nut[c] * (u[xp] - u[c]) - nut[xm] * (u[c] - u[xm]))
          + iy * (n4(i, j + 1) * (u[yp] - u[c]) - n4(i, j) * (u[c] - u[ym]));
        ou[c] += s * lap;
    }
    { // v-face at (i,j): cells j and j-1 in y, corners (i,j) and (i+1,j) in x
        const std::size_t xp = gidx(g, i + 1, j, k), xm = gidx(g, i - 1, j, k);
        const std::size_t yp = gidx(g, i, j + 1, k), ym = gidx(g, i, j - 1, k);
        const Real lap =
            iy * (nut[c] * (v[yp] - v[c]) - nut[ym] * (v[c] - v[ym]))
          + ix * (n4(i + 1, j) * (v[xp] - v[c]) - n4(i, j) * (v[c] - v[xm]));
        ov[c] += s * lap;
    }
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

// --- 2D golf ball: circle minus N dimple disks around the circumference -----
// Dimple footprint radius 8h, depth 4h -- near real golf proportions when
// D ~ 256h and still grid-resolved. sdf < 0 in the solid, z-invariant.
struct GolfDisk
{
    CylinderZ c;
    std::vector<std::array<double, 2>> q; // dimple-disk centers
    double rd, depth;

    static GolfDisk make(const CylinderZ& c, int nDimples, double h)
    {
        using std::numbers::pi;
        const double depth = 4.0 * h, a = 8.0 * h;
        GolfDisk b{c, {}, (a * a + depth * depth) / (2.0 * depth), depth};
        const double qr = c.r + b.rd - depth;
        for (int i = 0; i < nDimples; ++i)
        {   // deterministic irrational phase: no dimple sits axis-aligned
            const double th = 2.0 * pi * i / nDimples + 0.317;
            b.q.push_back({c.cx + qr * std::cos(th), c.cy + qr * std::sin(th)});
        }
        return b;
    }

    // Solid = disk \ union(dimple disks): sdf = max(disk, -dimples).
    double sdf(double x, double y) const
    {
        double phi = std::hypot(x - c.cx, y - c.cy) - c.r;
        for (const auto& p : q)
            phi = std::max(phi, rd - std::hypot(x - p[0], y - p[1]));
        return phi;
    }
};

// Cut geometry by supersampling the sdf on the dimpled shell band: lambda (and
// az) from 8x8 cell samples, ax/ay from 8 samples along the face segment; one
// x-y layer, replicated over k. Sampling noise ~1/8 of a face is absorbed by
// sanitizeCut and the first-order boundary scheme (cf. the eye app's lambda).
CutGeometry makeGolfCut(const Grid& g, const GolfDisk& b)
{
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    const double halfDiag = 0.5 * std::hypot(g.hx, g.hy);
    const double lo = b.c.r - b.depth - halfDiag, hi = b.c.r + halfDiag;
    auto dist = [&](double x, double y) {
        return std::hypot(x - b.c.cx, y - b.c.cy);
    };
    auto cellFrac = [&](double x0, double y0) {
        int m = 0;
        for (int bb = 0; bb < 8; ++bb)
            for (int a = 0; a < 8; ++a)
                if (b.sdf(x0 + (a + 0.5) * g.hx / 8,
                          y0 + (bb + 0.5) * g.hy / 8) > 0.0)
                    ++m;
        return m / 64.0;
    };
    auto segFrac = [&](double x0, double y0, double ex, double ey) {
        int m = 0;
        for (int a = 0; a < 8; ++a)
            if (b.sdf(x0 + (a + 0.5) * ex / 8, y0 + (a + 0.5) * ey / 8) > 0.0)
                ++m;
        return m / 8.0;
    };
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
        {
            const double x0 = i * g.hx, y0 = j * g.hy;
            auto classify = [&](double d, auto sample) {
                return d >= hi ? 1.0 : d <= lo ? 0.0 : sample();
            };
            const double lam =
                classify(dist(x0 + 0.5 * g.hx, y0 + 0.5 * g.hy),
                         [&] { return cellFrac(x0, y0); });
            const double ax = classify(dist(x0, y0 + 0.5 * g.hy),
                                       [&] { return segFrac(x0, y0, 0, g.hy); });
            const double ay = classify(dist(x0 + 0.5 * g.hx, y0),
                                       [&] { return segFrac(x0, y0, g.hx, 0); });
            for (int k = 0; k < g.nz; ++k)
            {
                const auto id = g.idx(i, j, k);
                cut.lambda[id] = lam;
                cut.ax[id] = ax;
                cut.ay[id] = ay;
                cut.az[id] = lam;
            }
        }
    return cut;
}

int main(int argc, char** argv)
{
    using std::numbers::pi;

    int n = 1024, nzInt = 8;
    double reD = 60.0, flowThroughs = 20.0, dRatio = 8.0, frameDt = 0.15;
    double cfl = 0.8;
    double tol = 1e-4;
    double csLes = 0.0; // Smagorinsky constant; 0 = no LES model
    int golf = 0;       // number of dimples; 0 = smooth circle
    bool open = false;  // fully periodic open far field (fringe sponge)
    std::string prefix = "cylinder_tunnel_gpu";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--golf") golf = int(next(0));
        else if (s == "--nz") nzInt = int(next(nzInt));
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--cfl") cfl = next(cfl);
        else if (s == "--les") csLes = next(csLes);
        else if (s == "--open") open = true;
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

    // Interior n x n x nzInt cells of size h; one ghost cell on every side.
    // nzInt <= 1 selects native 2D: nz == 1 with no z ghosts (periodic z).
    const double L = 2 * pi, uInf = 1.0;
    const double h = L / n;
    const int nx = n + 2, nz = nzInt <= 1 ? 1 : nzInt + 2;
    const int nzCells = nz == 1 ? 1 : nzInt; // interior z extent in cells
    const auto g = Grid::make(nx, nx, nz, nx * h, nx * h, nz * h);

    // Axis in grid coordinates: physical position + the ghost offset h.
    // Off-node on purpose (arbitrary cuts); sanitizeCut handles the tiny cells.
    const double D = L / dRatio;
    const CylinderZ cyl{h + L / 3.0 + 0.31 * h, h + L / 2.0 + 0.17 * h, D / 2.0};
    const GolfDisk ball = GolfDisk::make(cyl, std::max(golf, 1), h);
    auto cut = golf > 0 ? makeGolfCut(g, ball) : CutGeometry::make(g, cyl);
    if (golf > 0)
        std::printf("# golf: %d dimples, footprint 8h, depth 4h (%.3f D)\n",
                    golf, ball.depth / D);
    const auto st = sanitizeCut(g, cut); // before the ring: body cells only
    if (!open) closeBoxBoundary(g, cut); // open: ring cells stay periodic fluid

    const double nu = uInf * D / reD;
    std::printf("# cylinder_tunnel_gpu n=%d nz=%d%s Re_D=%g nu=%.4e D=%.4f (%.1f cells)"
                " les=%g open=%d\n",
                n, nzCells, nz == 1 ? " (native 2D)" : "", reD, nu, D, D / h, csLes,
                int(open));
    if (csLes > 0 && nz != 1)
    {
        std::fprintf(stderr, "--les is 2D only (nz == 1)\n");
        return 1;
    }
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed,
                st.minLambda, st.minAperture);

    // Clean laminar free stream; deterministic z-uniform trip one D behind the
    // cylinder to break the symmetric-wake transient early (Re 60 sheds anyway).
    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = uInf;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < nx; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > cyl.cx + cyl.r && x < cyl.cx + cyl.r + D
                    && std::abs(y - cyl.cy) < cyl.r)
                    U.v[g.idx(i, j, k)] = 0.05 * uInf;
            }

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    // Sharp viscous wall: the diffusion term reads a ghost-filled copy of the
    // stage field (mirror values, no-slip at the true sub-cell surface), while
    // convection keeps the masked zeros -- energy stability is untouched.
    auto dGhost = gpu::DeviceGhostFill::make(
        g, cut, [&](double x, double y, double) {
            if (golf > 0) return ball.sdf(x, y);
            return std::hypot(x - cyl.cx, y - cyl.cy) - cyl.r;
        });
    auto dU = gpu::DeviceField::make(g.size());
    auto dU0 = gpu::DeviceField::make(g.size());
    auto dS = gpu::DeviceField::make(g.size());
    auto dG = gpu::DeviceField::make(g.size()); // ghost-filled diffusion view
    auto dFx = gpu::DeviceScalar::make(g.size());
    auto dFy = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    auto dNut = gpu::DeviceScalar::make(csLes > 0 ? g.size() : 1);
    if (!dCut || !dGhost || !dU || !dU0 || !dS || !dG || !dFx || !dFy || !red
        || !scr || !dNut)
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

    auto tunnel = gpu::DeviceTunnelBc::make(gg, {uInf});
    // Fringe band: ~10% of the domain per side, cosine-ramped, sigMax 0.5.
    // x,y only: the flow is z-invariant, a z band would sponge the whole wake.
    const int nsFringe = std::clamp(int(0.1 * nx), 8, nx / 2 - 1);
    auto fringe = gpu::DeviceFringe::make(gg, {uInf, 0.5, nsFringe, 2});
    if (!tunnel || !fringe)
    {
        std::fprintf(stderr, "far-field init failed\n");
        return 2;
    }
    auto bcApply = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double cdt) {
        tunnel->apply(in, out, cdt);
    };
    auto fluxCorrect = [&](gpu::DeviceField& out) {
        tunnel->correctOutletFlux(out, *red, *dFx);
    };
    auto fringeApply = [&](gpu::DeviceField& out) { fringe->apply(out); };
    if (!open) bcApply(*dU, *dU, 0.0); // seed ghosts/prescribed faces

    // One SSP-RK3 stage with channel BCs and the ghost-flux projection.
    // Open: fringe sponge instead of BCs; periodic divergence sums to zero, so
    // the projection RHS is compatible without the in/out flux terms.
    int pcgIters = 0;
    auto stage = [&](const gpu::DeviceField& in, gpu::DeviceField& out, double a,
                     double dt) {
        (void)dGhost->fillFrom(in, *dG); // sub-cell wall for the viscous terms
        gpu::fusedRhsStage(gg, Real(nu), Real(0), 0, 0, 0, Real(a), *dU0,
                           Real(1.0 - a), in, *dG, Real(dt), out);
        if (csLes > 0)
        {
            kSmagNut<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                gg, Real(csLes * csLes * h * h), dG->u.data(), dG->v.data(),
                dCut->lambda.data(), dNut->data());
            gpu::checkLastKernel("smagNut");
            kEddyDiffuse<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                gg, Real((1.0 - a) * dt), dNut->data(), dG->u.data(), dG->v.data(),
                out.u.data(), out.v.data());
            gpu::checkLastKernel("eddyDiffuse");
        }
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
            tunnel->addFluxRhs(out, 1.0 / dt, scr->div);
        }
        pcgIters += poisson->solve(scr->div, scr->phi, true);
        gpu::subtractGradientCut(gg, *dCut, scr->phi, Real(dt), out);
        if (!open) bcApply(out, out, 0.0); // refresh mirrors after the correction
    };

    // Body-force tally box: the body plus margin, strictly inside the interior
    // so lateral boundary faces never count; the full interior span in z.
    auto box = gpu::bodyBox(gg, cyl.cx, cyl.cy, 0.0, cyl.r);
    box.k0 = gg.kInLo();
    box.k1 = gg.kInHi();
    auto bodyForce = gpu::DeviceBodyForce::make(gg, box);
    if (!bodyForce)
    {
        std::fprintf(stderr, "body-force init failed\n");
        return 2;
    }

    const double tEnd = flowThroughs * L / uInf;
    const double qRef = 0.5 * uInf * uInf * D * (nzCells * h); // frontal rectangle

    // Every run artefact goes through the one sink; the ghost ring is cropped
    // out by the (i0, j0) = (1, 1) window so frames stay n x n interior cells.
    const int kc = nz == 1 ? 0 : nz / 2;
    RunMeta rm;
    rm.nx = n; rm.ny = n; rm.nz = nzCells;
    rm.hx = h; rm.hy = h; rm.hz = h;
    rm.D = D; rm.re = reD; rm.frameDt = frameDt; rm.cs = csLes;
    rm.les = csLes > 0;
    rm.body = golf > 0 ? "cylinder-golf" : "cylinder";
    rm.farField = open ? "fringe" : "tunnel";
    rm.physics = "incompressible";

    FrameGeometry fg;
    fg.mask.assign(g.size(), 1.0f);
    for (std::size_t c = 0; c < g.size(); ++c) fg.mask[c] = float(cut.lambda[c]);
    fg.ic = nx / 2; fg.jc = nx / 2; fg.kc = kc; fg.D = D;
    fg.i0 = 1; fg.j0 = 1; fg.k0 = nz == 1 ? 0 : 1;
    fg.w = n; fg.h = n; fg.d = nzCells;

    RunOutput::Config oc;
    oc.prefix = prefix;
    oc.kinds = {Emit::Stats, Emit::Bin};
    if (nz == 1) oc.kinds.insert(Emit::Png);
    oc.fdt = frameDt;
    auto out = RunOutput::make(oc, rm, g, fg);
    if (!out) {
        std::fprintf(stderr, "output init failed: %s\n", out.error().c_str());
        return 4;
    }
    const std::string cols[] = {"t", "dt", "K", "cd", "cl",
                                "cdf", "cfl", "maxDiv", "pcgIters"};
    out->declareSeries("stats", cols);

    const float vortScale = float(8.0 * uInf / D), speedMax = float(1.6 * uInf);
    std::vector<Real> u0, v0, w0, w1;
    std::vector<float> vortF(g.size()), speedF(g.size());
    auto writeFrame = [&](double tNow) {
        (void)downloadSlab(dU->u, gg, kc, u0);
        (void)downloadSlab(dU->v, gg, kc, v0);
        (void)downloadSlab(dU->w, gg, kc, w0);
        // 2D: w is identically zero and there is no k+1 slab; reuse slab kc.
        (void)downloadSlab(dU->w, gg, nz == 1 ? kc : kc + 1, w1);
        auto at = [&](const std::vector<Real>& q, int i, int j) {
            return double(q[std::size_t(Grid::wrap(j, nx)) * nx
                            + std::size_t(Grid::wrap(i, nx))]);
        };
        for (int j = 0; j < nx; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const std::size_t c = g.idx(i, j, kc);
                const double wz = (at(v0, i, j) - at(v0, i - 1, j)) / h
                                - (at(u0, i, j) - at(u0, i, j - 1)) / h;
                vortF[c] = float(wz);
                const double uc = 0.5 * (at(u0, i, j) + at(u0, i + 1, j));
                const double vc = 0.5 * (at(v0, i, j) + at(v0, i, j + 1));
                const double wc = 0.5 * (at(w0, i, j) + at(w1, i, j));
                speedF[c] = float(std::sqrt(uc * uc + vc * vc + wc * wc));
            }
        const FrameField ff[2]{{"vort", vortF.data(), 0, vortScale, true},
                               {"speed", speedF.data(), 0, speedMax, false}};
        if (auto r = out->frame(tNow, FrameViews{ff}); !r)
            std::fprintf(stderr, "frame write failed: %s\n", r.error().c_str());
    };

    double t = 0.0, dt = 1e-3;
    long steps = 0;
    const auto wall0 = std::chrono::steady_clock::now();
    while (t < tEnd)
    {
        const double conv = gpu::maxCourantPerDt(gg, *dU, *red);
        const double diff = nu > 0 ? 6.0 * nu / (h * h) : 0.0;
        const double dtStab = cfl / std::max(conv, diff);
        dt = std::min(tEnd - t, dtStab);

        // Blowup forensics: if the CFL limiter (not the end-of-run clamp) has
        // collapsed dt, the field has diverged -- dump the argmax and stop.
        if (dtStab < 1e-4 && steps > 0)
        {
            auto host = VectorField::zeros(g);
            (void)gpu::downloadField(*dU, host);
            reportBlowup(g, host, cut.lambda, t);
            return 7;
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
            // and would bias the force low. Friction is read from the time-t
            // field (matching phi's time level) through the ghost-filled view:
            // the wall links the viscous term actually applied this step.
            (void)dGhost->fillFrom(*dU0, *dG);
            const auto co = bodyForce->measure(*dCut, scr->phi, *dG, nu, qRef,
                                               csLes > 0 ? &*dNut : nullptr, *red);
            cd = co.cd;
            cl = co.cl;
            cdf = co.cdf;
        }
        stage(*dU, *dS, 0.75, dt);
        stage(*dS, *dU, 1.0 / 3.0, dt);

        if (out->wantsFrame(t) || out->wantsPng(t)) writeFrame(t);

        if (steps % 20 == 0)
        {
            const double K = gpu::kineticEnergy(gg, *dU, *red);
            const double cfl = dt * conv;

            // Continuity residual; tunnel adds the prescribed in/out fluxes,
            // open is plain periodic divergence.
            gpu::divergenceCut(gg, *dCut, *dU, scr->div, Real(1));
            if (!open)
            {
                tunnel->addFluxRhs(*dU, 1.0, scr->div);
            }
            const double maxDiv = gpu::maxAbsScalar(scr->div, *red);

            const double rowv[] = {t,   dt,  K,      cd,
                                   cl,  cdf, cfl,    maxDiv,
                                   double(pcgIters)};
            (void)out->row("stats", rowv);
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

    out->setMsPerStep(wallMs / double(steps));
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n",
                steps, out->frames(), wallMs / double(steps));
    return 0;
}
