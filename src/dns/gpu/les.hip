#include "dns/gpu/les.hpp"

#include "dns/gpu/device_util.hpp"

#include <hip/hip_runtime.h>

#include <cmath>

namespace dns::gpu {

namespace {

// nut = cs2d2 * op(grad u) * lambda at cell centres from the MAC field, full
// 3D gradients; op is |S| (sigma == 0, Smagorinsky) or D_sigma (sigma != 0).
// At nz == 1 every z term is EXACTLY zero (w == 0, k+-1 wraps to k), so the
// Smagorinsky value is bit-identical to the old 2D kernel -- and D_sigma is
// identically zero there (two-component flow), by design.
__global__ void kSmagNut(GpuGrid g, Real cs2d2, int sigma, const Real* u,
                         const Real* v, const Real* w, const Real* lam,
                         Real* nut)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (i == 0 || i == g.nx - 1 || j == 0 || j == g.ny - 1
        || (g.nz > 1 && (k == 0 || k == g.nz - 1)))
    {
        nut[c] = 0; // ghost ring: no eddy flux through the domain boundary
        return;
    }
    const Real dudx = (u[gidx(g, i + 1, j, k)] - u[c]) / g.hx;
    const Real dvdy = (v[gidx(g, i, j + 1, k)] - v[c]) / g.hy;
    const Real dwdz = (w[gidx(g, i, j, k + 1)] - w[c]) / g.hz;
    auto uc = [&](int a, int b, int q) {
        return Real(0.5) * (u[gidx(g, a, b, q)] + u[gidx(g, a + 1, b, q)]);
    };
    auto vc = [&](int a, int b, int q) {
        return Real(0.5) * (v[gidx(g, a, b, q)] + v[gidx(g, a, b + 1, q)]);
    };
    auto wc = [&](int a, int b, int q) {
        return Real(0.5) * (w[gidx(g, a, b, q)] + w[gidx(g, a, b, q + 1)]);
    };
    const Real dudy = (uc(i, j + 1, k) - uc(i, j - 1, k)) / (2 * g.hy);
    const Real dvdx = (vc(i + 1, j, k) - vc(i - 1, j, k)) / (2 * g.hx);
    const Real dudz = (uc(i, j, k + 1) - uc(i, j, k - 1)) / (2 * g.hz);
    const Real dwdx = (wc(i + 1, j, k) - wc(i - 1, j, k)) / (2 * g.hx);
    const Real dvdz = (vc(i, j, k + 1) - vc(i, j, k - 1)) / (2 * g.hz);
    const Real dwdy = (wc(i, j + 1, k) - wc(i, j - 1, k)) / (2 * g.hy);
    if (sigma)
    {
        const Real gd[3][3] = {{dudx, dudy, dudz},
                               {dvdx, dvdy, dvdz},
                               {dwdx, dwdy, dwdz}};
        nut[c] = cs2d2 * Real(sigmaDsigma(gd)) * lam[c];
        return;
    }
    const Real sxy = Real(0.5) * (dudy + dvdx);
    const Real sxz = Real(0.5) * (dudz + dwdx);
    const Real syz = Real(0.5) * (dvdz + dwdy);
    const Real smag = std::sqrt(2 * dudx * dudx + 2 * dvdy * dvdy
                                + 2 * dwdz * dwdz + 4 * sxy * sxy
                                + 4 * sxz * sxz + 4 * syz * syz);
    nut[c] = cs2d2 * smag * lam[c];
}

// out += s * div(nut grad q) per MAC component, conservative flux form with
// face- and edge-averaged nut. The z fluxes are appended after the x,y sum, so
// at nz == 1 (where they are exactly zero) ou/ov keep their old bits and ow
// adds an exact zero.
__global__ void kEddyDiffuse(GpuGrid g, Real s, const Real* nut, const Real* u,
                             const Real* v, const Real* w, Real* ou, Real* ov,
                             Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real ix = Real(1) / (g.hx * g.hx), iy = Real(1) / (g.hy * g.hy),
               iz = Real(1) / (g.hz * g.hz);
    auto nxy = [&](int a, int b) { // nut on the (a,b) cell edge in the xy plane
        return Real(0.25) * (nut[gidx(g, a, b, k)] + nut[gidx(g, a - 1, b, k)]
                             + nut[gidx(g, a, b - 1, k)] + nut[gidx(g, a - 1, b - 1, k)]);
    };
    auto nxz = [&](int a, int q) { // nut on the (a,q) cell edge in the xz plane
        return Real(0.25) * (nut[gidx(g, a, j, q)] + nut[gidx(g, a - 1, j, q)]
                             + nut[gidx(g, a, j, q - 1)] + nut[gidx(g, a - 1, j, q - 1)]);
    };
    auto nyz = [&](int b, int q) { // nut on the (b,q) cell edge in the yz plane
        return Real(0.25) * (nut[gidx(g, i, b, q)] + nut[gidx(g, i, b - 1, q)]
                             + nut[gidx(g, i, b, q - 1)] + nut[gidx(g, i, b - 1, q - 1)]);
    };
    const std::size_t xp = gidx(g, i + 1, j, k), xm = gidx(g, i - 1, j, k);
    const std::size_t yp = gidx(g, i, j + 1, k), ym = gidx(g, i, j - 1, k);
    const std::size_t zp = gidx(g, i, j, k + 1), zm = gidx(g, i, j, k - 1);
    { // u-face at (i,j,k): cells i and i-1 in x; xy edges in y, xz edges in z
        const Real lap =
            ix * (nut[c] * (u[xp] - u[c]) - nut[xm] * (u[c] - u[xm]))
          + iy * (nxy(i, j + 1) * (u[yp] - u[c]) - nxy(i, j) * (u[c] - u[ym]))
          + iz * (nxz(i, k + 1) * (u[zp] - u[c]) - nxz(i, k) * (u[c] - u[zm]));
        ou[c] += s * lap;
    }
    { // v-face at (i,j,k): cells j and j-1 in y; xy edges in x, yz edges in z
        const Real lap =
            iy * (nut[c] * (v[yp] - v[c]) - nut[ym] * (v[c] - v[ym]))
          + ix * (nxy(i + 1, j) * (v[xp] - v[c]) - nxy(i, j) * (v[c] - v[xm]))
          + iz * (nyz(j, k + 1) * (v[zp] - v[c]) - nyz(j, k) * (v[c] - v[zm]));
        ov[c] += s * lap;
    }
    { // w-face at (i,j,k): cells k and k-1 in z; xz edges in x, yz edges in y
        const Real lap =
            iz * (nut[c] * (w[zp] - w[c]) - nut[zm] * (w[c] - w[zm]))
          + ix * (nxz(i + 1, k) * (w[xp] - w[c]) - nxz(i, k) * (w[c] - w[xm]))
          + iy * (nyz(j + 1, k) * (w[yp] - w[c]) - nyz(j, k) * (w[c] - w[ym]));
        ow[c] += s * lap;
    }
}

} // namespace

std::expected<DeviceSmagorinsky, hipError_t>
DeviceSmagorinsky::make(const GpuGrid& g, double cs, SgsModel model)
{
    if (!(cs > 0)) return std::unexpected(hipErrorInvalidValue);
    auto nut = DeviceScalar::make(g.size());
    if (!nut) return std::unexpected(nut.error());
    // Delta^2 = hx*hy at nz == 1 (unit depth carries no filter width; the
    // expression is kept VERBATIM so 2D LES trajectories do not move), else
    // the isotropic (hx hy hz)^(2/3).
    const Real cs2d2 =
        g.nz == 1 ? Real(cs * cs * g.hx * g.hy)
                  : Real(cs * cs
                         * std::pow(double(g.hx) * double(g.hy) * double(g.hz),
                                    2.0 / 3.0));
    return DeviceSmagorinsky(g, cs2d2, model, std::move(*nut));
}

void DeviceSmagorinsky::computeNut(const DeviceField& U,
                                   const DeviceCutGeometry& cut)
{
    kSmagNut<<<launchBlocks(g_.size()), blockSize>>>(
        g_, cs2d2_, model_ == SgsModel::Sigma ? 1 : 0, U.u.data(), U.v.data(),
        U.w.data(), cut.lambda.data(), nut_.data());
    checkLastKernel("smagNut");
}

void DeviceSmagorinsky::addEddyDiffusion(const DeviceField& U, Real scale,
                                         DeviceField& out) const
{
    kEddyDiffuse<<<launchBlocks(g_.size()), blockSize>>>(
        g_, scale, nut_.data(), U.u.data(), U.v.data(), U.w.data(),
        out.u.data(), out.v.data(), out.w.data());
    checkLastKernel("eddyDiffuse");
}

} // namespace dns::gpu
