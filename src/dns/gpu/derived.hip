#include "dns/gpu/derived.hpp"

#include "dns/gpu/device_util.hpp"

#include <hip/hip_runtime.h>

#include <cmath>

namespace dns::gpu {

namespace {

__global__ void kVorticityZ(GpuGrid g, const Real* u, const Real* v, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    out[c] = (v[c] - v[gidx(g, i - 1, j, k)]) / g.hx
           - (u[c] - u[gidx(g, i, j - 1, k)]) / g.hy;
}

__global__ void kSpeedMac(GpuGrid g, const Real* u, const Real* v, const Real* w,
                          Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real uc = Real(0.5) * (u[c] + u[gidx(g, i + 1, j, k)]);
    const Real vc = Real(0.5) * (v[c] + v[gidx(g, i, j + 1, k)]);
    const Real wc = Real(0.5) * (w[c] + w[gidx(g, i, j, k + 1)]);
    out[c] = std::sqrt(uc * uc + vc * vc + wc * wc);
}

__global__ void kPrimitive(std::size_t n, const Real* rho, const Real* mx,
                           const Real* my, const Real* mz, Real* u, Real* v,
                           Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const Real r = rho[c] > 0 ? rho[c] : Real(1);
    u[c] = mx[c] / r;
    v[c] = my[c] / r;
    w[c] = mz[c] / r;
}

__global__ void kVorticityZCentred(GpuGrid g, const Real* u, const Real* v,
                                   Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    out[c] = (v[gidx(g, i + 1, j, k)] - v[gidx(g, i - 1, j, k)]) / (2 * g.hx)
           - (u[gidx(g, i, j + 1, k)] - u[gidx(g, i, j - 1, k)]) / (2 * g.hy);
}

__global__ void kCurlMagCentred(GpuGrid g, const Real* u, const Real* v,
                                const Real* w, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    // nz == 1: gidx(g, i, j, k+1) == gidx(g, i, j, k-1) == c, so every z
    // derivative is (q[c] - q[c]) / (2 hz) == 0 identically.
    const Real dwdy = (w[gidx(g, i, j + 1, k)] - w[gidx(g, i, j - 1, k)]) / (2 * g.hy);
    const Real dvdz = (v[gidx(g, i, j, k + 1)] - v[gidx(g, i, j, k - 1)]) / (2 * g.hz);
    const Real dudz = (u[gidx(g, i, j, k + 1)] - u[gidx(g, i, j, k - 1)]) / (2 * g.hz);
    const Real dwdx = (w[gidx(g, i + 1, j, k)] - w[gidx(g, i - 1, j, k)]) / (2 * g.hx);
    const Real dvdx = (v[gidx(g, i + 1, j, k)] - v[gidx(g, i - 1, j, k)]) / (2 * g.hx);
    const Real dudy = (u[gidx(g, i, j + 1, k)] - u[gidx(g, i, j - 1, k)]) / (2 * g.hy);
    const Real wx = dwdy - dvdz, wy = dudz - dwdx, wz = dvdx - dudy;
    out[c] = std::sqrt(wx * wx + wy * wy + wz * wz);
}

__global__ void kSpeedCentred(std::size_t n, const Real* u, const Real* v,
                              const Real* w, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    out[c] = std::sqrt(u[c] * u[c] + v[c] * v[c] + w[c] * w[c]);
}

} // namespace

void vorticityZ(const GpuGrid& g, const DeviceField& U, DeviceScalar& out)
{
    kVorticityZ<<<launchBlocks(g.size()), blockSize>>>(g, U.u.data(), U.v.data(),
                                                       out.data());
    checkLastKernel("vorticityZ");
}

void speedMac(const GpuGrid& g, const DeviceField& U, DeviceScalar& out)
{
    kSpeedMac<<<launchBlocks(g.size()), blockSize>>>(g, U.u.data(), U.v.data(),
                                                     U.w.data(), out.data());
    checkLastKernel("speedMac");
}

void primitiveVelocity(const GpuGrid& g, const DeviceScalar& rho,
                       const DeviceScalar& mx, const DeviceScalar& my,
                       const DeviceScalar& mz, DeviceScalar& u, DeviceScalar& v,
                       DeviceScalar& w)
{
    kPrimitive<<<launchBlocks(g.size()), blockSize>>>(
        g.size(), rho.data(), mx.data(), my.data(), mz.data(), u.data(),
        v.data(), w.data());
    checkLastKernel("primitiveVelocity");
}

void vorticityZCentred(const GpuGrid& g, const DeviceScalar& u,
                       const DeviceScalar& v, DeviceScalar& out)
{
    kVorticityZCentred<<<launchBlocks(g.size()), blockSize>>>(
        g, u.data(), v.data(), out.data());
    checkLastKernel("vorticityZCentred");
}

void curlMagCentred(const GpuGrid& g, const DeviceScalar& u, const DeviceScalar& v,
                    const DeviceScalar& w, DeviceScalar& out)
{
    kCurlMagCentred<<<launchBlocks(g.size()), blockSize>>>(
        g, u.data(), v.data(), w.data(), out.data());
    checkLastKernel("curlMagCentred");
}

void speedCentred(const GpuGrid& g, const DeviceScalar& u, const DeviceScalar& v,
                  const DeviceScalar& w, DeviceScalar& out)
{
    kSpeedCentred<<<launchBlocks(g.size()), blockSize>>>(
        g.size(), u.data(), v.data(), w.data(), out.data());
    checkLastKernel("speedCentred");
}

} // namespace dns::gpu
