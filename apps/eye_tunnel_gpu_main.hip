// Flow around a z-aligned eye/lens body in a numerical flow channel (GPU):
// the solid is the intersection of two equal disks (sharp horizontal tips),
// height H = L/dRatio, width W = ar*H. The inscribed ball of diameter H is
// strictly interior to the lens, so the solid IS the lens; the ball is drawn
// by viewers from the json R only. Otherwise identical to cylinder_tunnel_gpu:
// quasi-2D free-slip z-walls or --nz 1 native 2D, laminar Dirichlet inflow,
// convective outflow with exact mass-flux correction, open free-stream lateral
// y-walls, sanitized cut-cell body, DCT(NeumannBox)-preconditioned PCG Poisson;
// drag/lift from the aperture-pressure surface integral plus the masked-link
// skin-friction tally (x/y links only).
// The grid carries a one-cell ghost ring: interior is n x n x nzInt cells.
// Usage: eye_tunnel_gpu [n] [--nz nzInt | --nz 1 for native 2D]
//                       [--re ReH] [--dr DRatio] [--ar widthOverHeight]
//                       [--ft flowThroughs] [--fdt frameDt] [--tol rms] [--cfl c]
//                       [--les Cs (Smagorinsky, 2D only, 0 = off)]
//                       [--out prefix]
#include "dns/fieldpng.hpp"
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/ghostfill.hpp"
#include "dns/gpu/les.hpp"
#include "dns/gpu/kernels.hpp"
#include "dns/gpu/reductions.hpp"

#include <hip/hip_runtime.h>

#include <filesystem>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace dns;
using gpu::Real;

// Interior k range: [1, nz-2] with a z ghost ring, the single plane k = 0 in
// native 2D (nz == 1).
__host__ __device__ inline int kInLo(int nz) { return nz == 1 ? 0 : 1; }
__host__ __device__ inline int kInHi(int nz) { return nz == 1 ? 0 : nz - 2; }

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
//   inlet   (i<=1):      u = uInf; v,w antimirrored so they vanish on x=hx
//   outlet  (i==nx-1):   convective advection out of the domain
//   y-walls (j rings):   freestream far-field -- u = uInf, v = w = 0
//   z-walls (k rings):   FREE-SLIP -- u,v mirrored, w = 0 on the wall faces
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
    // already Neumann (closed apertures + DCT). z mirror rows: none in native
    // 2D (nz == 1, no z ghosts).
    const int jm = j == 0 ? 1 : j == ny - 1 ? ny - 2 : j;
    const int km = nz == 1 ? k : (k == 0 ? 1 : k == nz - 1 ? nz - 2 : k);

    if (i <= 1) u[c] = uInf;
    else if (i == nx - 1)
        u[c] = inu[c] - cdt * (inu[c] - inu[gidx(g, nx - 2, j, k)]);
    else if (jm != j) u[c] = u[gidx(g, i, jm, k)];
    else if (km != k) u[c] = u[gidx(g, i, j, km)];

    if (i == 0) v[c] = -v[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        v[c] = inv[c] - cdt * (inv[c] - inv[gidx(g, nx - 2, j, k)]);
    else if (j == 1 || j == ny - 1) v[c] = 0; // impermeable wall faces
    else if (j == 0) v[c] = -v[gidx(g, i, 2, k)]; // antimirror across y = hy
    else if (km != k) v[c] = v[gidx(g, i, j, km)];

    if (i == 0) w[c] = -w[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        w[c] = inw[c] - cdt * (inw[c] - inw[gidx(g, nx - 2, j, k)]);
    else if (jm != j) w[c] = w[gidx(g, i, jm, k)];
    else if (nz == 1 || k <= 1 || k == nz - 1) w[c] = 0;
}

// Per-cell tally of the outlet-plane u so one sumScalar yields the volume flux.
__global__ void kOutletTally(gpu::GpuGrid g, const Real* u, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    const bool plane = i == g.nx - 1 && j >= 1 && j <= g.ny - 2 && k >= kInLo(g.nz)
                    && k <= kInHi(g.nz);
    out[c] = plane ? u[c] : Real(0);
}

// u_outlet += corr on the open outlet plane (exact global mass balance).
__global__ void kOutletCorrect(gpu::GpuGrid g, Real corr, Real* u)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    if (i == g.nx - 1 && j >= 1 && j <= g.ny - 2 && k >= kInLo(g.nz)
        && k <= kInHi(g.nz))
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
    if (j < 1 || j > g.ny - 2 || k < kInLo(g.nz) || k > kInHi(g.nz)) return;
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

// Friction force the masked Laplacian exerts on the body: momentum leaves the
// fluid only through Laplacian links from open faces to closed (masked-zero)
// faces, each draining nu*u*(A_link/d_link). x/y links only -- the cylinder is
// z-invariant and the free-slip z ghosts mirror the field, so z links carry no
// viscous flux (and the closed ghost-ring z faces must not be counted).
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
    // Closed neighbors hold ghost-fill mirror values, so each wall link drains
    // nu*(q - q_ghost)*A/d -- the Laplacian flux the momentum update applies.
    auto closedLinks = [&](const Real* a, const Real* q) {
        Real s = 0;
        auto link = [&](std::size_t nb, Real l) {
            if (a[nb] <= openTol) s += l * (q[c] - q[nb]);
        };
        link(gidx(g, i - 1, j, k), lx);
        link(gidx(g, i + 1, j, k), lx);
        link(gidx(g, i, j - 1, k), ly);
        link(gidx(g, i, j + 1, k), ly);
        return s;
    };
    if (ax[c] > openTol) fx[c] = nu * closedLinks(ax, u);
    if (ay[c] > openTol) fy[c] = nu * closedLinks(ay, v);
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
// SGS counterpart of kBodyFriction: kEddyDiffuse drains momentum through the
// same closed links, but with the local nut and its exact per-link weights
// (x: cell-centre nut[c]/nut[xm]; y: corner-averaged n4). Same A/d normalisation
// as kBodyFriction (nu -> nut), so this folds straight into cd.
__global__ void kBodySgsFriction(gpu::GpuGrid g, const Real* nut, const Real* ax,
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
    auto n4 = [&](int a, int b) { // nut at the (a,b) cell corner (as kEddyDiffuse)
        return Real(0.25) * (nut[gidx(g, a, b, k)] + nut[gidx(g, a - 1, b, k)]
                             + nut[gidx(g, a, b - 1, k)] + nut[gidx(g, a - 1, b - 1, k)]);
    };
    // As in kBodyFriction: closed neighbors hold ghost values, drain is per-link
    // nut*(q - q_ghost)*A/d.
    if (ax[c] > openTol)
    {
        Real s = 0;
        auto link = [&](std::size_t nb, Real nl) {
            if (ax[nb] <= openTol) s += nl * (u[c] - u[nb]);
        };
        link(gidx(g, i - 1, j, k), nut[gidx(g, i - 1, j, k)] * lx);
        link(gidx(g, i + 1, j, k), nut[c] * lx);
        link(gidx(g, i, j - 1, k), n4(i, j) * ly);
        link(gidx(g, i, j + 1, k), n4(i, j + 1) * ly);
        fx[c] = s;
    }
    if (ay[c] > openTol)
    {
        Real s = 0;
        auto link = [&](std::size_t nb, Real nl) {
            if (ay[nb] <= openTol) s += nl * (v[c] - v[nb]);
        };
        link(gidx(g, i, j - 1, k), nut[gidx(g, i, j - 1, k)] * ly);
        link(gidx(g, i, j + 1, k), nut[c] * ly);
        link(gidx(g, i - 1, j, k), n4(i, j) * lx);
        link(gidx(g, i + 1, j, k), n4(i + 1, j) * lx);
        fy[c] = s;
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

// Eye/lens body: intersection of two equal disks, radius ra = e + hh, centres
// at (cx, cy -+ e) with e = (w^2 - hh^2)/(2 hh) -- sharp tips at x = cx +- w,
// top/bottom at y = cy +- hh. Convex. Pre: w > hh.
struct Lens
{
    double cx, cy, w, hh; // half-width, half-height (= inscribed-ball radius)
    double e() const { return (w * w - hh * hh) / (2 * hh); }
    double ra() const { return e() + hh; }
};

// Cut geometry for the z-invariant lens: ax/ay exact via the closed-form solid
// chord intervals, lambda/az by 64x64 supersampling of boundary cells (area
// error ~2e-4 << sanitize thresholds; boundary scheme is first order anyway).
CutGeometry makeLensCut(const Grid& g, const Lens& b)
{
    CutGeometry cut{zeros(g), zeros(g), zeros(g), zeros(g)};
    for (auto& v : cut.lambda) v = 1;
    for (auto& v : cut.ax) v = 1;
    for (auto& v : cut.ay) v = 1;
    for (auto& v : cut.az) v = 1;
    const double e = b.e(), ra = b.ra(), ra2 = ra * ra;
    auto inside = [&](double x, double y) {
        const double dx2 = (x - b.cx) * (x - b.cx);
        const double d1 = y - (b.cy - e), d2 = y - (b.cy + e);
        return dx2 + d1 * d1 <= ra2 && dx2 + d2 * d2 <= ra2;
    };
    // Solid y-interval at fixed x: both disks share s = sqrt(ra^2 - dx^2), the
    // intersection is [cy+e-s, cy-e+s] (empty when s <= e, i.e. |dx| >= w).
    auto chordY = [&](double x, double& lo, double& hi) {
        const double q = ra2 - (x - b.cx) * (x - b.cx);
        const double s = q > 0 ? std::sqrt(q) : 0;
        lo = b.cy + e - s;
        hi = b.cy - e + s;
    };
    // Solid x-interval at fixed y: cx +- min(s1, s2) over the two disks.
    auto chordX = [&](double y, double& lo, double& hi) {
        const double d1 = y - (b.cy - e), d2 = y - (b.cy + e);
        const double q = ra2 - std::max(d1 * d1, d2 * d2);
        const double s = q > 0 ? std::sqrt(q) : 0;
        lo = b.cx - s;
        hi = b.cx + s;
    };
    const int i0 = std::max(0, int((b.cx - b.w) / g.hx) - 2);
    const int i1 = std::min(g.nx - 1, int((b.cx + b.w) / g.hx) + 2);
    const int j0 = std::max(0, int((b.cy - b.hh) / g.hy) - 2);
    const int j1 = std::min(g.ny - 1, int((b.cy + b.hh) / g.hy) + 2);
    constexpr int ns = 64;
    for (int j = j0; j <= j1; ++j)
        for (int i = i0; i <= i1; ++i)
        {
            const double x0 = i * g.hx, y0 = j * g.hy;
            // u-face at x0, y in [y0, y0+hy]; v-face at y0, x in [x0, x0+hx]
            double lo, hi;
            chordY(x0, lo, hi);
            const double sy = std::max(0.0, std::min(hi, y0 + g.hy) - std::max(lo, y0));
            chordX(y0, lo, hi);
            const double sx = std::max(0.0, std::min(hi, x0 + g.hx) - std::max(lo, x0));
            // Cell fluid fraction: all four corners inside a convex solid means
            // fully solid; otherwise supersample.
            double lam;
            if (inside(x0, y0) && inside(x0 + g.hx, y0) && inside(x0, y0 + g.hy)
                && inside(x0 + g.hx, y0 + g.hy))
                lam = 0;
            else
            {
                int hit = 0;
                for (int a = 0; a < ns; ++a)
                    for (int c = 0; c < ns; ++c)
                        hit += inside(x0 + (a + 0.5) * g.hx / ns,
                                      y0 + (c + 0.5) * g.hy / ns);
                lam = 1.0 - double(hit) / (ns * ns);
            }
            for (int k = 0; k < g.nz; ++k)
            {
                const std::size_t c = g.idx(i, j, k);
                cut.ax[c] = 1.0 - sy / g.hy;
                cut.ay[c] = 1.0 - sx / g.hx;
                cut.lambda[c] = lam;
                cut.az[c] = lam;
            }
        }
    return cut;
}

} // namespace

int main(int argc, char** argv)
{
    using std::numbers::pi;

    int n = 1024, nzInt = 8;
    double reD = 60.0, flowThroughs = 20.0, dRatio = 8.0, frameDt = 0.15;
    double aspect = 3.5; // lens width / height
    double cfl = 0.8;
    double tol = 1e-4;
    double csLes = 0.0; // Smagorinsky constant; 0 = no LES model
    std::string prefix = "eye_tunnel_gpu";
    for (int a = 1; a < argc; ++a)
    {
        const std::string_view s(argv[a]);
        auto next = [&](double d) { return a + 1 < argc ? parseOr(argv[++a], d) : d; };
        if (s == "--re") reD = next(reD);
        else if (s == "--nz") nzInt = int(next(nzInt));
        else if (s == "--dr") dRatio = next(dRatio);
        else if (s == "--ar") aspect = next(aspect);
        else if (s == "--ft") flowThroughs = next(flowThroughs);
        else if (s == "--cfl") cfl = next(cfl);
        else if (s == "--les") csLes = next(csLes);
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

    // Centre in grid coordinates: physical position + the ghost offset h.
    // Off-node on purpose (arbitrary cuts); sanitizeCut handles the tiny cells.
    const double D = L / dRatio; // eye height = inscribed-ball diameter
    if (aspect <= 1.0)
    {
        std::fprintf(stderr, "--ar must be > 1 (lens needs w > hh)\n");
        return 1;
    }
    const Lens eye{h + L / 3.0 + 0.31 * h, h + L / 2.0 + 0.17 * h,
                   aspect * D / 2.0, D / 2.0};
    auto cut = makeLensCut(g, eye);
    const auto st = sanitizeCut(g, cut); // before the ring: body cells only
    closeBoxBoundary(g, cut);

    const double nu = uInf * D / reD;
    std::printf("# eye_tunnel_gpu n=%d nz=%d%s Re_H=%g nu=%.4e H=%.4f (%.1f cells)"
                " W=%.4f ar=%g les=%g\n",
                n, nzCells, nz == 1 ? " (native 2D)" : "", reD, nu, D, D / h,
                2 * eye.w, aspect, csLes);
    if (csLes > 0 && nz != 1)
    {
        std::fprintf(stderr, "--les is 2D only (nz == 1)\n");
        return 1;
    }
    std::printf("# sanitize: cells=%zu faces=%zu pockets=%zu minL=%.3g minA=%.3g\n",
                st.cellsSolidified, st.facesClosed, st.pocketsClosed,
                st.minLambda, st.minAperture);

    // Clean laminar free stream; deterministic z-uniform trip one H behind the
    // trailing tip to break the symmetric-wake transient early.
    auto U = VectorField::zeros(g);
    for (auto& x : U.u) x = uInf;
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < nx; ++j)
            for (int i = 0; i < nx; ++i)
            {
                const double x = i * h, y = (j + 0.5) * h;
                if (x > eye.cx + eye.w && x < eye.cx + eye.w + D
                    && std::abs(y - eye.cy) < eye.hh)
                    U.v[g.idx(i, j, k)] = 0.05 * uInf;
            }

    const auto gg = gpu::GpuGrid::from(g);
    auto dCut = gpu::DeviceCutGeometry::make(cut);
    // Sharp viscous wall: diffusion reads a ghost-filled copy (mirror values,
    // no-slip at the true sub-cell surface); convection keeps masked zeros.
    // Lens = disk intersection, sdf = max of the two disk distances (exact
    // away from the tips; FD normal picks the active disk).
    auto dGhost = gpu::DeviceGhostFill::make(
        g, cut, [&](double x, double y, double) {
            const double e = eye.e(), ra = eye.ra();
            const double d1 = std::hypot(x - eye.cx, y - (eye.cy - e)) - ra;
            const double d2 = std::hypot(x - eye.cx, y - (eye.cy + e)) - ra;
            return std::max(d1, d2);
        });
    auto dU = gpu::DeviceField::make(g.size());
    auto dU0 = gpu::DeviceField::make(g.size());
    auto dS = gpu::DeviceField::make(g.size());
    auto dG = gpu::DeviceField::make(g.size()); // ghost-filled diffusion view
    auto dFx = gpu::DeviceScalar::make(g.size());
    auto dFy = gpu::DeviceScalar::make(g.size());
    auto red = gpu::ReduceScratch::make();
    auto scr = gpu::DeviceCutProjectionScratch::make(g.size());
    std::optional<gpu::DeviceSmagorinsky> les;
    if (csLes > 0)
    {
        auto m = gpu::DeviceSmagorinsky::make(gg, csLes);
        if (!m)
        {
            std::fprintf(stderr, "LES init failed\n");
            return 2;
        }
        les = std::move(*m);
    }
    auto dNut = gpu::DeviceScalar::make(1);
    if (!dCut || !dGhost || !dU || !dU0 || !dS || !dG || !dFx || !dFy || !red
        || !scr || !dNut)
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

    const double planeArea = double(n) * nzCells * h * h;
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
        (void)dGhost->fillFrom(in, *dG); // sub-cell wall for the viscous terms
        gpu::fusedRhsStage(gg, Real(nu), Real(0), 0, 0, 0, Real(a), *dU0,
                           Real(1.0 - a), in, *dG, Real(dt), out);
        if (les)
        {
            les->computeNut(*dG, *dCut);
            les->addEddyDiffusion(*dG, Real((1.0 - a) * dt), out);
        }
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
    // margin in x/y, strictly inside the interior so lateral boundary faces
    // never count; the full interior span in z (only x/y links are tallied).
    const int bi0 = std::max(2, int((eye.cx - eye.w) / h) - 2);
    const int bi1 = std::min(nx - 3, int((eye.cx + eye.w) / h) + 3);
    const int bj0 = std::max(2, int((eye.cy - eye.hh) / h) - 2);
    const int bj1 = std::min(nx - 3, int((eye.cy + eye.hh) / h) + 3);
    const int bk0 = kInLo(nz), bk1 = kInHi(nz);

    const double tEnd = flowThroughs * L / uInf;
    const double qRef = 0.5 * uInf * uInf * D * (nzCells * h); // frontal rectangle
    std::ofstream stats(prefix + "_stats.csv");
    stats << "t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters\n";

    std::ofstream(prefix + ".json")
        << "{\"nx\":" << n << ",\"ny\":" << n << ",\"frameDt\":" << frameDt
        << ",\"D\":" << D << ",\"W\":" << 2 * eye.w << ",\"reD\":" << reD
        << ",\"cx\":" << eye.cx / h - 1 << ",\"cy\":" << eye.cy / h - 1
        << ",\"R\":" << eye.hh / h << "}\n";

    // Interior mid-z slices: signed z-vorticity at cell corners and velocity
    // magnitude at cell centres (top view).
    const int kc = nz == 1 ? 0 : nz / 2;
    std::ofstream vortFile(prefix + "_vort.bin", std::ios::binary);
    std::ofstream speedFile(prefix + "_speed.bin", std::ios::binary);
    std::vector<float> maskImg(std::size_t(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            maskImg[std::size_t(j) * n + i] =
                float(cut.lambda[g.idx(i + 1, j + 1, kc)]);
    {
        std::ofstream mf(prefix + "_mask.bin", std::ios::binary);
        mf.write(reinterpret_cast<const char*>(maskImg.data()),
                 std::streamsize(maskImg.size() * sizeof(float)));
    }
    // Live PNGs of the 2D fields (vorticity diverging, |u| viridis).
    const std::string pngDir = "png_" + prefix;
    const bool pngs = nz == 1;
    if (pngs) std::filesystem::create_directories(pngDir);
    const float vortScale = float(8.0 * uInf / D), speedMax = float(1.6 * uInf);
    int frames = 0;
    double nextFrame = 0.0;
    std::vector<Real> u0, v0, w0, w1;
    std::vector<float> frame(std::size_t(n) * n);
    auto writeFrame = [&]() {
        (void)downloadSlab(dU->u, gg, kc, u0);
        (void)downloadSlab(dU->v, gg, kc, v0);
        (void)downloadSlab(dU->w, gg, kc, w0);
        // 2D: w is identically zero and there is no k+1 slab; reuse slab kc.
        (void)downloadSlab(dU->w, gg, nz == 1 ? kc : kc + 1, w1);
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
        char png[4096];
        if (pngs) {
            std::snprintf(png, sizeof png, "%s/vort_%05d.png", pngDir.c_str(),
                          frames);
            dns::writeDivergingPng(png, n, n, frame.data(), vortScale,
                                   maskImg.data());
        }
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
        if (pngs) {
            std::snprintf(png, sizeof png, "%s/speed_%05d.png", pngDir.c_str(),
                          frames);
            dns::writeSequentialPng(png, n, n, frame.data(), 0.0f, speedMax,
                                    maskImg.data());
        }
        ++frames;
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

            // SGS wall stress: the eddy term drains momentum through the same
            // closed links. dNut already holds nut of the time-t field dU0 (set
            // by this step's first stage), so mirror kEddyDiffuse's weights and
            // fold into cd. cdf column stays nu-only (documented).
            if (csLes > 0)
            {
                kBodySgsFriction<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
                    gg, les->nut().data(), dCut->ax.data(), dCut->ay.data(),
                    dG->u.data(), dG->v.data(), bi0, bi1, bj0, bj1, bk0, bk1,
                    dFx->data(), dFy->data());
                gpu::checkLastKernel("bodySgsFriction");
                cd += gpu::sumScalar(*dFx, *red) / qRef;
                cl += gpu::sumScalar(*dFy, *red) / qRef;
            }
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
        << ",\"frameDt\":" << frameDt << ",\"D\":" << D << ",\"W\":" << 2 * eye.w
        << ",\"reD\":" << reD << ",\"cx\":" << eye.cx / h - 1
        << ",\"cy\":" << eye.cy / h - 1 << ",\"R\":" << eye.hh / h
        << ",\"msPerStep\":" << wallMs / double(steps) << "}\n";
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n",
                steps, frames, wallMs / double(steps));
    return 0;
}
