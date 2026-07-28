#include "dns/gpu/boussinesq.hpp"

#include "dns/gpu/device_util.hpp"

namespace dns::gpu {

namespace {

// Cell (i,j,k) owns the y-faces j (below) and j+1 (above); the two wall faces
// are therefore j = 1 and j = ny-1, and rows 0 and ny-1 are ghosts.
// Centred hot patch: the floor column (i,k) is heated iff it lies within the
// patch in BOTH lateral directions. Written as a fraction of the FLUID index
// range so it is independent of the box extents and of whether that axis
// carries a ghost column, and vacuous at nz == 1 (one z cell, always inside),
// which makes the 2D case the middle strip in x.
__host__ __device__ inline bool inHotPatch(const GpuGrid& g, int i, int k,
                                           Real patch, int ox, int oz)
{
    const Real fx = (Real(i - ox) + Real(0.5)) / Real(g.nx - 2 * ox)
                  - Real(0.5);
    if (fabsf(fx) > Real(0.5) * patch) return false;
    if (g.nz == 1) return true;
    const Real fz = (Real(k - oz) + Real(0.5)) / Real(g.nz - 2 * oz)
                  - Real(0.5);
    return fabsf(fz) <= Real(0.5) * patch;
}

__global__ void kRbWall(GpuGrid g, Real Tbot, Real Ttop, Real patch, int ox,
                        int oz, Real* u, Real* v, Real* w, Real* T)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= 2 * nxz) return;
    const bool hi = t >= nxz;
    const int i = int(t % g.nx), k = int((t / g.nx) % g.nz);
    const std::size_t c = gidx(g, i, hi ? g.ny - 1 : 0, k);
    const std::size_t s = gidx(g, i, hi ? g.ny - 2 : 1, k);

    // Dirichlet plate: the linear image puts T = Twall on the wall face. The
    // floor carries Tbot only inside the patch; outside it, and on the whole
    // lid, the wall sits at Ttop.
    const Real Twall =
        hi ? Ttop : (inHotPatch(g, i, k, patch, ox, oz) ? Tbot : Ttop);
    T[c] = Real(2) * Twall - T[s];
    // No-slip: tangential components ANTImirrored so they vanish at the face
    // by symmetry. (A free-slip wall -- the tunnel's -- mirrors them instead;
    // that difference is the whole boundary condition.)
    u[c] = -u[s];
    w[c] = -w[s];
    // Wall-normal component pinned to zero ON the wall face.
    if (hi) v[c] = Real(0);
    else
    {
        v[gidx(g, i, 1, k)] = Real(0);
        v[c] = Real(0);
    }
}

// Lateral no-slip ADIABATIC wall on x: the plates' treatment one axis over.
// The wall faces are the u-faces i = 1 and i = nx-1, the tangential v and w are
// antimirrored into the ghost column, and T is MIRRORED rather than imaged --
// dT/dn = 0, an insulating side wall. Runs over EVERY j and k, including the y
// ghost rows, so a corner ghost is the x-mirror of a cell the y pass already
// mirrored: the diagonal reflection.
__global__ void kRbWallX(GpuGrid g, Real* u, Real* v, Real* w, Real* T)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nyz = std::size_t(g.ny) * g.nz;
    if (t >= 2 * nyz) return;
    const bool hi = t >= nyz;
    const int j = int(t % g.ny), k = int((t / g.ny) % g.nz);
    const std::size_t c = gidx(g, hi ? g.nx - 1 : 0, j, k);
    const std::size_t s = gidx(g, hi ? g.nx - 2 : 1, j, k);

    T[c] = T[s];
    v[c] = -v[s];
    w[c] = -w[s];
    if (hi) u[c] = Real(0);
    else
    {
        u[gidx(g, 1, j, k)] = Real(0);
        u[c] = Real(0);
    }
}

// The z twin, and it exists only at nz > 1: at nz == 1 the z stencil cancels by
// periodic wrap and there is no third dimension to wall off.
__global__ void kRbWallZ(GpuGrid g, Real* u, Real* v, Real* w, Real* T)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxy = std::size_t(g.nx) * g.ny;
    if (t >= 2 * nxy) return;
    const bool hi = t >= nxy;
    const int i = int(t % g.nx), j = int((t / g.nx) % g.ny);
    const std::size_t c = gidx(g, i, j, hi ? g.nz - 1 : 0);
    const std::size_t s = gidx(g, i, j, hi ? g.nz - 2 : 1);

    T[c] = T[s];
    u[c] = -u[s];
    v[c] = -v[s];
    if (hi) w[c] = Real(0);
    else
    {
        w[gidx(g, i, j, 1)] = Real(0);
        w[c] = Real(0);
    }
}

// dp/dn = 0 on every wall. A ghost cell reads the interior cell obtained by
// clamping each of its ghost indices inward, which for a corner is the DIAGONAL
// interior cell -- so every read is of an interior value and the fill needs no
// ordering between the axes. ox = 0 leaves x untouched (periodic), and then
// this is exactly the old y-only mirror.
__global__ void kMirrorPressure(GpuGrid g, int ox, int oz, Real* p)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const int ii = min(max(i, ox), g.nx - 1 - ox);
    const int jj = min(max(j, 1), g.ny - 2);
    const int kk = min(max(k, oz), g.nz - 1 - oz);
    if (ii == i && jj == j && kk == k) return;
    p[c] = p[gidx(g, ii, jj, kk)];
}

__global__ void kStageTemperature(GpuGrid g, Real kappa, Real a, const Real* Tx,
                                  Real b, const Real* __restrict__ u,
                                  const Real* __restrict__ v,
                                  const Real* __restrict__ w,
                                  const Real* __restrict__ T, Real dt, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    // Ghost rows carry no equation; the wall kernel rewrites them right after.
    if (j < 1 || j > g.ny - 2)
    {
        out[c] = a * Tx[c] + b * T[c];
        return;
    }
    const Offsets o = offsetsAt(g, i, j, k);
    const Real Tc = T[c];

    // Conservative centred advection on the MAC faces: cell c's low face on
    // axis A is the face stored AT c, its high face the one at c + o.p[A].
    // The wall face carries v = 0 exactly, so no heat is advected through a
    // plate and every bit of wall transport is the Fourier term below.
    const Real fxLo = u[c] * Real(0.5) * (T[c + o.m[0]] + Tc);
    const Real fxHi = u[c + o.p[0]] * Real(0.5) * (Tc + T[c + o.p[0]]);
    const Real fyLo = v[c] * Real(0.5) * (T[c + o.m[1]] + Tc);
    const Real fyHi = v[c + o.p[1]] * Real(0.5) * (Tc + T[c + o.p[1]]);
    Real adv = (fxHi - fxLo) / g.hx + (fyHi - fyLo) / g.hy;
    Real lap = (T[c + o.p[0]] - Real(2) * Tc + T[c + o.m[0]]) / (g.hx * g.hx)
             + (T[c + o.p[1]] - Real(2) * Tc + T[c + o.m[1]]) / (g.hy * g.hy);
    if (g.nz > 1)
    {
        const Real fzLo = w[c] * Real(0.5) * (T[c + o.m[2]] + Tc);
        const Real fzHi = w[c + o.p[2]] * Real(0.5) * (Tc + T[c + o.p[2]]);
        adv += (fzHi - fzLo) / g.hz;
        lap += (T[c + o.p[2]] - Real(2) * Tc + T[c + o.m[2]]) / (g.hz * g.hz);
    }
    out[c] = a * Tx[c] + b * (Tc + dt * (-adv + kappa * lap));
}

// The y-face at index j sits between cells j-1 and j, so the face temperature
// is the average of those two.
__global__ void kAddBuoyancy(GpuGrid g, Real gAlpha, Real Tref, Real s,
                             const Real* __restrict__ T, Real* v)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (j < 1 || j > g.ny - 1) return;
    const Offsets o = offsetsAt(g, i, j, k);
    const Real Tface = Real(0.5) * (T[c] + T[c + o.m[1]]);
    v[c] += s * gAlpha * (Tface - Tref);
}

__global__ void kZeroGhostCells(GpuGrid g, int ox, int oz, Real* q)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const bool in = i >= ox && i < g.nx - ox && j >= 1 && j < g.ny - 1
        && k >= oz && k < g.nz - oz;
    if (!in) q[c] = Real(0);
}

__global__ void kRbPlaneStats(GpuGrid g, int ox, int oz, const Real* u,
                              const Real* v, const Real* w, const Real* T,
                              double* out)
{
    const int j = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (j >= g.ny) return;
    double sums[4] = {0, 0, 0, 0};
    for (int k = oz; k < g.nz - oz; ++k)
        for (int i = ox; i < g.nx - ox; ++i)
        {
            const std::size_t c = gidx(g, i, j, k);
            const Offsets o = offsetsAt(g, i, j, k);
            const double uc = 0.5 * (double(u[c]) + double(u[c + o.p[0]]));
            const double vc = 0.5 * (double(v[c]) + double(v[c + o.p[1]]));
            const double wc =
                g.nz > 1 ? 0.5 * (double(w[c]) + double(w[c + o.p[2]])) : 0.0;
            const double Tc = T[c];
            sums[0] += Tc;
            sums[1] += vc;
            sums[2] += vc * Tc;
            sums[3] += uc * uc + vc * vc + wc * wc;
        }
    for (int q = 0; q < 4; ++q) out[std::size_t(j) * 4 + q] = sums[q];
}

} // namespace

void applyRbWall(const GpuGrid& g, Real Tbot, Real Ttop, Real patch, bool closed,
                 DeviceField& U, DeviceScalar& T)
{
    const int ox = sideGhostX(closed), oz = sideGhostZ(g, closed);
    // Plates first, then x, then z: each pass reads what the previous wrote, so
    // a corner ghost ends up the diagonal reflection of an interior cell.
    kRbWall<<<launchBlocks(2 * std::size_t(g.nx) * g.nz), blockSize>>>(
        g, Tbot, Ttop, patch, ox, oz, U.u.data(), U.v.data(), U.w.data(),
        T.data());
    checkLastKernel("kRbWall");
    if (ox == 0) return;
    kRbWallX<<<launchBlocks(2 * std::size_t(g.ny) * g.nz), blockSize>>>(
        g, U.u.data(), U.v.data(), U.w.data(), T.data());
    checkLastKernel("kRbWallX");
    if (oz == 0) return;
    kRbWallZ<<<launchBlocks(2 * std::size_t(g.nx) * g.ny), blockSize>>>(
        g, U.u.data(), U.v.data(), U.w.data(), T.data());
    checkLastKernel("kRbWallZ");
}

double rbPatchAreaFraction(const GpuGrid& g, double patch, bool closed)
{
    const int ox = sideGhostX(closed), oz = sideGhostZ(g, closed);
    std::size_t inside = 0;
    for (int k = oz; k < g.nz - oz; ++k)
        for (int i = ox; i < g.nx - ox; ++i)
            if (inHotPatch(g, i, k, Real(patch), ox, oz)) ++inside;
    return double(inside) / (double(g.nx - 2 * ox) * (g.nz - 2 * oz));
}

void mirrorPressureWalls(const GpuGrid& g, bool closed, DeviceScalar& p)
{
    kMirrorPressure<<<launchBlocks(g.size()), blockSize>>>(
        g, sideGhostX(closed), sideGhostZ(g, closed), p.data());
    checkLastKernel("kMirrorPressure");
}

void stageTemperature(const GpuGrid& g, Real kappa, Real a,
                      const DeviceScalar& Tx, Real b, const DeviceField& U,
                      const DeviceScalar& T, Real dt, DeviceScalar& out)
{
    kStageTemperature<<<launchBlocks(g.size()), blockSize>>>(
        g, kappa, a, Tx.data(), b, U.u.data(), U.v.data(), U.w.data(), T.data(),
        dt, out.data());
    checkLastKernel("kStageTemperature");
}

void addBuoyancy(const GpuGrid& g, Real gAlpha, Real Tref, Real s,
                 const DeviceScalar& T, DeviceField& U)
{
    kAddBuoyancy<<<launchBlocks(g.size()), blockSize>>>(g, gAlpha, Tref, s,
                                                        T.data(), U.v.data());
    checkLastKernel("kAddBuoyancy");
}

void zeroGhostCells(const GpuGrid& g, bool closed, DeviceScalar& q)
{
    kZeroGhostCells<<<launchBlocks(g.size()), blockSize>>>(
        g, sideGhostX(closed), sideGhostZ(g, closed), q.data());
    checkLastKernel("kZeroGhostCells");
}

hipError_t rbPlaneStats(const GpuGrid& g, bool closed, const DeviceField& U,
                        const DeviceScalar& T, DeviceBuffer<double>& scratch,
                        std::vector<double>& host)
{
    const std::size_t n = std::size_t(g.ny) * 4;
    if (scratch.size() < n)
    {
        auto b = DeviceBuffer<double>::make(n);
        if (!b) return b.error();
        scratch = std::move(*b);
    }
    kRbPlaneStats<<<launchBlocks(std::size_t(g.ny)), blockSize>>>(
        g, sideGhostX(closed), sideGhostZ(g, closed), U.u.data(), U.v.data(),
        U.w.data(), T.data(), scratch.data());
    checkLastKernel("kRbPlaneStats");
    host.resize(n);
    return hipMemcpy(host.data(), scratch.data(), n * sizeof(double),
                     hipMemcpyDeviceToHost);
}

} // namespace dns::gpu
