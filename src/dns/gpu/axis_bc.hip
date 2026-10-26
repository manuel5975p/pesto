#include "dns/gpu/axis_bc.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"

#include <cassert>
#include <cmath>

namespace dns::gpu {

namespace {

// The 5 + nT conserved fields and their low words (null = plain state).
struct Fields
{
    Real* q[7];
    Real* lo[7];
    int n;
};

Fields fieldsOf(DeviceCompState& s)
{
    Fields f{{s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data()},
             {},
             5 + s.nT};
    for (int i = 0; i < s.nT; ++i) f.q[5 + i] = s.t[i].data();
    if (s.compensated()) {
        for (int v = 0; v < 5; ++v) f.lo[v] = s.lo[v].data();
        for (int i = 0; i < s.nT; ++i) f.lo[5 + i] = s.tlo[i].data();
    }
    return f;
}

// dst = R_x(angle) src for the field set: (rho v, rho w) rotated by (c, s),
// everything else copied; dst's low words zeroed.
__device__ inline void rotateCopy(const Fields& f, std::size_t dst,
                                  std::size_t src, Real c, Real s)
{
    for (int v = 0; v < f.n; ++v) {
        if (v == 2 || v == 3) continue;
        f.q[v][dst] = f.q[v][src];
    }
    // Spelled as explicit fmas so the rounding is pinned (a test recomputes
    // it on the host bit for bit) rather than left to contraction.
    const Real my = f.q[2][src], mz = f.q[3][src];
    f.q[2][dst] = fmaf(c, my, -(s * mz));
    f.q[3][dst] = fmaf(s, my, c * mz);
    if (f.lo[0])
        for (int v = 0; v < f.n; ++v) f.lo[v][dst] = Real(0);
}

// Wedge ghost columns: thread t over 2 * ny, (i_g, i_s) = (0, 1) for the low
// half, (nx-1, nx-2) for the high; R_x(pi) is (c, s) = (-1, 0) EXACTLY.
__global__ void kAxisColumnsWedge(GpuGrid g, Fields f)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= 2 * std::size_t(g.ny)) return;
    const bool hi = t >= std::size_t(g.ny);
    const int j = int(t % g.ny);
    const int ig = hi ? g.nx - 1 : 0, is = hi ? g.nx - 2 : 1;
    rotateCopy(f, gidx(g, ig, j, 1), gidx(g, is, j, 1), Real(-1), Real(0));
}

// Wedge ghost planes: thread over nx * ny; plane 0 = R_x(-dphi), plane 2 =
// R_x(+dphi) of plane 1.
__global__ void kWedgePlanes(GpuGrid g, Fields f, Real c, Real s)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(g.nx) * g.ny) return;
    const int i = int(t % g.nx), j = int(t / g.nx);
    const std::size_t src = gidx(g, i, j, 1);
    rotateCopy(f, gidx(g, i, j, 0), src, c, -s);
    rotateCopy(f, gidx(g, i, j, 2), src, c, s);
}

// Full revolution: thread over 2 * ny * nz; the ghost takes the cell across
// the axis, (i_s, j, (k + nz/2) mod nz), every field verbatim.
__global__ void kAxisColumnsFull(GpuGrid g, Fields f)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t plane = std::size_t(g.ny) * g.nz;
    if (t >= 2 * plane) return;
    const bool hi = t >= plane;
    const std::size_t r = t % plane;
    const int j = int(r % g.ny), k = int(r / g.ny);
    const int ig = hi ? g.nx - 1 : 0, is = hi ? g.nx - 2 : 1;
    const int ks = (k + g.nz / 2) % g.nz;
    const std::size_t dst = gidx(g, ig, j, k), src = gidx(g, is, j, ks);
    for (int v = 0; v < f.n; ++v) {
        f.q[v][dst] = f.q[v][src];
        if (f.lo[0]) f.lo[v][dst] = Real(0);
    }
}

// mu_t / F1: the same index maps, scalars copied.
__global__ void kAxisMut(GpuGrid g, int wedge, Real* nut, Real* f1)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (wedge) {
        if (t >= 2 * std::size_t(g.ny)) return;
        const bool hi = t >= std::size_t(g.ny);
        const int j = int(t % g.ny);
        const std::size_t d = gidx(g, hi ? g.nx - 1 : 0, j, 1),
                          s = gidx(g, hi ? g.nx - 2 : 1, j, 1);
        nut[d] = nut[s];
        if (f1) f1[d] = f1[s];
        return;
    }
    const std::size_t plane = std::size_t(g.ny) * g.nz;
    if (t >= 2 * plane) return;
    const bool hi = t >= plane;
    const std::size_t r = t % plane;
    const int j = int(r % g.ny), k = int(r / g.ny);
    const std::size_t d = gidx(g, hi ? g.nx - 1 : 0, j, k),
                      s = gidx(g, hi ? g.nx - 2 : 1, j, (k + g.nz / 2) % g.nz);
    nut[d] = nut[s];
    if (f1) f1[d] = f1[s];
}

__global__ void kWedgePlanesMut(GpuGrid g, Real* nut, Real* f1)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(g.nx) * g.ny) return;
    const int i = int(t % g.nx), j = int(t / g.nx);
    const std::size_t s = gidx(g, i, j, 1);
    for (const int k : {0, 2}) {
        const std::size_t d = gidx(g, i, j, k);
        nut[d] = nut[s];
        if (f1) f1[d] = f1[s];
    }
}

} // namespace

std::expected<DeviceAxisBc, hipError_t> DeviceAxisBc::make(const Grid& g,
                                                           const RevolvedSpec& rev)
{
    const bool ok = g.nx >= 6 && g.ny >= 4 &&
                    (rev.wedge ? g.nz == 3 && rev.dphi > 0
                               : g.nz == rev.nphi && rev.nphi >= 4 &&
                                     rev.nphi % 2 == 0);
    if (!ok) return std::unexpected(hipErrorInvalidValue);
    DeviceAxisBc a;
    a.g_ = GpuGrid::from(g);
    a.wedge_ = rev.wedge;
    a.c_ = Real(std::cos(rev.dphi));
    a.s_ = Real(std::sin(rev.dphi));
    return a;
}

void DeviceAxisBc::apply(DeviceCompState& s) const
{
    const Fields f = fieldsOf(s);
    if (!wedge_) {
        const std::size_t n = 2 * std::size_t(g_.ny) * g_.nz;
        kAxisColumnsFull<<<launchBlocks(n), blockSize>>>(g_, f);
        checkLastKernel("kAxisColumnsFull");
        return;
    }
    kAxisColumnsWedge<<<launchBlocks(2 * std::size_t(g_.ny)), blockSize>>>(g_, f);
    checkLastKernel("kAxisColumnsWedge");
    kWedgePlanes<<<launchBlocks(std::size_t(g_.nx) * g_.ny), blockSize>>>(
        g_, f, c_, s_);
    checkLastKernel("kWedgePlanes");
}

void DeviceAxisBc::applyMut(Real* nut, Real* f1) const
{
    const std::size_t n = wedge_ ? 2 * std::size_t(g_.ny)
                                 : 2 * std::size_t(g_.ny) * g_.nz;
    kAxisMut<<<launchBlocks(n), blockSize>>>(g_, wedge_ ? 1 : 0, nut, f1);
    checkLastKernel("kAxisMut");
    if (!wedge_) return;
    kWedgePlanesMut<<<launchBlocks(std::size_t(g_.nx) * g_.ny), blockSize>>>(
        g_, nut, f1);
    checkLastKernel("kWedgePlanesMut");
}

} // namespace dns::gpu
