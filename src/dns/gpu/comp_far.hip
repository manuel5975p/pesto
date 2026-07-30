#include "dns/gpu/comp_far.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/internal/wall_ghost.hpp"

namespace dns::gpu {

namespace {

// Freestream sponge with a smooth cosine strength, 0 at the interior band
// edge, sigMax at the domain boundary. The x bands (inflow + acoustic outflow
// absorber) relax the full state to freestream. Open lateral bands relax
// density, pressure and TANGENTIAL velocity only: the boundary-normal
// momentum stays free, so the body's displacement flow vents through the
// band instead of being reflected (a v -> 0 sponge is a wall in disguise).
__global__ void kFarSponge(GpuGrid g, int ns, Real sigMax, Real gam, int openY,
                           int openZ, Real rhoI, Real uI, Real pI, Real* rho,
                           Real* mx, Real* my, Real* mz, Real* E)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    auto depth = [ns](int q, int n) {
        if (q < ns) return Real(ns - q) / Real(ns);
        if (q >= n - ns) return Real(q - (n - ns) + 1) / Real(ns);
        return Real(0);
    };
    auto sig = [sigMax](Real r) {
        return r > Real(0) ? sigMax * Real(0.5) * (Real(1) - cosf(Real(M_PI) * r))
                           : Real(0);
    };
    const Real sx = sig(depth(i, g.nx));
    const Real sy = openY ? sig(depth(j, g.ny)) : Real(0);
    const Real sz = openZ ? sig(depth(k, g.nz)) : Real(0);
    if (sx <= Real(0) && sy <= Real(0) && sz <= Real(0)) return;

    Real r = rho[c];
    Real u = mx[c] / r, v = my[c] / r, w = mz[c] / r;
    Real p = (gam - Real(1))
           * (E[c] - Real(0.5) * r * (u * u + v * v + w * w));
    // x bands: full freestream relax (inflow + wake outflow). Open lateral
    // bands: rho/p only -- a pressure far field. Velocity stays free there,
    // so the body's displacement/dipole field is not clamped to uniform
    // freestream at finite distance (that clamp IS the wall blockage).
    const Real srp = fmaxf(sx, fmaxf(sy, sz));
    u += sx * (uI - u);
    v -= sx * v;
    w -= sx * w;
    r += srp * (rhoI - r);
    p += srp * (pI - p);
    rho[c] = r;
    mx[c] = r * u;
    my[c] = r * v;
    mz[c] = r * w;
    E[c] = p / (gam - Real(1)) + Real(0.5) * r * (u * u + v * v + w * w);
}

// Mirror walls on the y extremes; the wall planes are the faces between rows
// {0,1} and {ny-2,ny-1} (rows 0 and ny-1 are ghost rows). Free-slip: only the
// normal momentum antimirrored. No-slip: ALL momentum antimirrored, so every
// velocity component vanishes at the wall face by symmetry; |m| is unchanged,
// so the copied E keeps p mirrored -- an adiabatic no-slip wall. Writes j in
// {0, ny-1}, reads {1, ny-2}: disjoint.
__global__ void kFarWallY(GpuGrid g, int noslip, Real* rho, Real* mx, Real* my,
                          Real* mz, Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, k;
    bool hi;
    std::size_t c, s;
    if (!internal::yGhostPair(g, t, i, k, hi, c, s)) return;
    rho[c] = rho[s];
    mx[c] = noslip ? -mx[s] : mx[s];
    my[c] = -my[s];
    mz[c] = noslip ? -mz[s] : mz[s];
    E[c] = E[s];
}

// Same on the z extremes (only launched when nz > 1).
__global__ void kFarWallZ(GpuGrid g, Real* rho, Real* mx, Real* my, Real* mz,
                          Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxy = std::size_t(g.nx) * g.ny;
    if (t >= 2 * nxy) return;
    const bool hi = t >= nxy;
    const int i = int(t % g.nx), j = int((t / g.nx) % g.ny);
    const std::size_t c = gidx(g, i, j, hi ? g.nz - 1 : 0);
    const std::size_t s = gidx(g, i, j, hi ? g.nz - 2 : 1);
    rho[c] = rho[s];
    mx[c] = mx[s];
    my[c] = my[s];
    mz[c] = -mz[s];
    E[c] = E[s];
}

} // namespace

std::expected<DeviceCompFar, hipError_t> DeviceCompFar::make(const Grid& g,
                                                             Config c)
{
    const bool nsOk = c.ns >= 1 && 2 * c.ns < g.nx;
    const bool wallsOk = (c.openY || g.ny >= 3) && (c.openZ || g.nz == 1 || g.nz >= 3);
    if (!nsOk || !wallsOk || !(c.rhoInf > 0) || !(c.pInf > 0) || !(c.gamma > 1))
        return std::unexpected(hipErrorInvalidValue);
    if (g.nz == 1) c.openZ = false;
    return DeviceCompFar(g, c);
}

DeviceCompFar::DeviceCompFar(const Grid& g, const Config& c)
    : g_(GpuGrid::from(g)), c_(c), u_(c.uInf)
{
}

void DeviceCompFar::apply(DeviceCompState& s) const
{
    kFarSponge<<<launchBlocks(g_.size()), blockSize>>>(
        g_, c_.ns, Real(c_.sigMax), Real(c_.gamma), int(c_.openY), int(c_.openZ),
        Real(c_.rhoInf), Real(u_), Real(c_.pInf), s.rho.data(), s.mx.data(),
        s.my.data(), s.mz.data(), s.E.data());
    checkLastKernel("kFarSponge");
    if (!c_.openY) {
        kFarWallY<<<launchBlocks(2 * std::size_t(g_.nx) * g_.nz), blockSize>>>(
            g_, int(c_.noslipY), s.rho.data(), s.mx.data(), s.my.data(),
            s.mz.data(), s.E.data());
        checkLastKernel("kFarWallY");
    }
    if (!c_.openZ && g_.nz > 1) {
        kFarWallZ<<<launchBlocks(2 * std::size_t(g_.nx) * g_.ny), blockSize>>>(
            g_, s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
        checkLastKernel("kFarWallZ");
    }
}

void DeviceCompFar::boost(DeviceCompState& s, double du) const
{
    boostX(g_, s, du);
}

} // namespace dns::gpu
