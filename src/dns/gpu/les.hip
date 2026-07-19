#include "dns/gpu/les.hpp"

#include "dns/gpu/device_util.hpp"

#include <hip/hip_runtime.h>

#include <cmath>

namespace dns::gpu {

namespace {

// nut = (Cs^2 hx hy) |S| lambda at cell centres from the MAC field.
__global__ void kSmagNut(GpuGrid g, Real cs2hh, const Real* u, const Real* v,
                         const Real* lam, Real* nut)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
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

__global__ void kEddyDiffuse(GpuGrid g, Real s, const Real* nut, const Real* u,
                             const Real* v, Real* ou, Real* ov)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
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

} // namespace

std::expected<DeviceSmagorinsky, hipError_t>
DeviceSmagorinsky::make(const GpuGrid& g, double cs)
{
    if (!(cs > 0) || g.nz != 1) return std::unexpected(hipErrorInvalidValue);
    auto nut = DeviceScalar::make(g.size());
    if (!nut) return std::unexpected(nut.error());
    return DeviceSmagorinsky(g, Real(cs * cs * g.hx * g.hy), std::move(*nut));
}

void DeviceSmagorinsky::computeNut(const DeviceField& U,
                                   const DeviceCutGeometry& cut)
{
    kSmagNut<<<launchBlocks(g_.size()), blockSize>>>(
        g_, cs2hh_, U.u.data(), U.v.data(), cut.lambda.data(), nut_.data());
    checkLastKernel("smagNut");
}

void DeviceSmagorinsky::addEddyDiffusion(const DeviceField& U, Real scale,
                                         DeviceField& out) const
{
    kEddyDiffuse<<<launchBlocks(g_.size()), blockSize>>>(
        g_, scale, nut_.data(), U.u.data(), U.v.data(), out.u.data(),
        out.v.data());
    checkLastKernel("eddyDiffuse");
}

} // namespace dns::gpu
