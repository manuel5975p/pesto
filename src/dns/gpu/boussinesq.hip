#include "dns/gpu/boussinesq.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/hotpatch.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/wall_ghost.hpp"

namespace dns::gpu {

namespace {

// Cell (i,j,k) owns the y-faces j (below) and j+1 (above); the two wall faces
// are therefore j = 1 and j = ny-1, and rows 0 and ny-1 are ghosts. The hot
// patch predicate is shared with the compressible branch (gpu/hotpatch.hpp).

__global__ void kRbWall(GpuGrid g, Real Tbot, Real Ttop, Real patch, int ox,
                        int oz, Real* u, Real* v, Real* w, Real* T)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, k;
    bool hi;
    std::size_t c, s;
    if (!internal::yGhostPair(g, t, i, k, hi, c, s)) return;

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

// Lateral no-slip ADIABATIC wall on x (axis 0) or z (axis 2): the plates'
// treatment one axis over. The wall faces are the normal-velocity faces at
// index 1 and n-1 of that axis, the two tangential components are
// antimirrored into the ghost column, and T is MIRRORED rather than imaged --
// dT/dn = 0, an insulating side wall. The index pair is the shared one
// (internal::sideGhostPair), so a corner ghost is the diagonal reflection. The
// z wall is launched only at nz > 1: at nz == 1 the z stencil cancels by
// periodic wrap and there is no third dimension to wall off.
// `n` is the wall-normal velocity, `t1`/`t2` the tangential ones.
__global__ void kRbWallSide(GpuGrid g, int axis, Real* n, Real* t1, Real* t2,
                            Real* T)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    bool hi;
    std::size_t c, s;
    if (!internal::sideGhostPair(g, axis, t, hi, c, s)) return;

    T[c] = T[s];
    t1[c] = -t1[s];
    t2[c] = -t2[s];
    if (hi) n[c] = Real(0);
    else
    {
        n[s] = Real(0);   // s is the low wall face, index 1 on this axis
        n[c] = Real(0);
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
    if (!inFluid(g, i, j, k, ox, oz)) q[c] = Real(0);
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
    kRbWallSide<<<launchBlocks(2 * std::size_t(g.ny) * g.nz), blockSize>>>(
        g, 0, U.u.data(), U.v.data(), U.w.data(), T.data());
    checkLastKernel("kRbWallSide x");
    if (oz == 0) return;
    kRbWallSide<<<launchBlocks(2 * std::size_t(g.nx) * g.ny), blockSize>>>(
        g, 2, U.w.data(), U.u.data(), U.v.data(), T.data());
    checkLastKernel("kRbWallSide z");
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
    if (auto e = internal::ensurePlaneScratch(scratch, n); e != hipSuccess)
        return e;
    kRbPlaneStats<<<launchBlocks(std::size_t(g.ny)), blockSize>>>(
        g, sideGhostX(closed), sideGhostZ(g, closed), U.u.data(), U.v.data(),
        U.w.data(), T.data(), scratch.data());
    checkLastKernel("kRbPlaneStats");
    return internal::fetchPlaneScratch(scratch, n, host);
}

} // namespace dns::gpu
