#include "dns/gpu/kernels.hpp"

#include "dns/gpu/projection.hpp"

#include <cassert>

#include "dns/gpu/device_util.hpp"

namespace dns::gpu {

namespace {

__device__ inline Real avg(Real a, Real b) { return Real(0.5) * (a + b); }

__global__ void kDivergence(GpuGrid g, const Real* u, const Real* v, const Real* w,
                            Real* out, Real scale)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    out[c] = scale * ((u[gidx(g, i + 1, j, k)] - u[c]) / g.hx
                    + (v[gidx(g, i, j + 1, k)] - v[c]) / g.hy
                    + (w[gidx(g, i, j, k + 1)] - w[c]) / g.hz);
}

__global__ void kSubtractGradient(GpuGrid g, const Real* p, Real scale,
                                  Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    u[c] -= scale * (p[c] - p[gidx(g, i - 1, j, k)]) / g.hx;
    v[c] -= scale * (p[c] - p[gidx(g, i, j - 1, k)]) / g.hy;
    w[c] -= scale * (p[c] - p[gidx(g, i, j, k - 1)]) / g.hz;
}

// Plain cell divergence at (i,j,k) -- the skew-symmetric correction input.
__device__ Real cellDiv(const GpuGrid& g, const Real* u, const Real* v,
                        const Real* w, int i, int j, int k)
{
    const std::size_t c = gidx(g, i, j, k);
    return (u[gidx(g, i + 1, j, k)] - u[c]) / g.hx
         + (v[gidx(g, i, j + 1, k)] - v[c]) / g.hy
         + (w[gidx(g, i, j, k + 1)] - w[c]) / g.hz;
}

// Convective fluxes per momentum component at face (i,j,k) in skew-symmetric
// form: divergence form -(d(ui uj)/dxj) plus 0.5*ui*div(u) at the face. The
// correction vanishes on a discretely solenoidal field; with cut cells (only
// the apertured divergence is projected to zero) and the FP32 projection
// residual it is what keeps the operator energy-conserving -- the plain
// divergence form injects energy ~ u^2*div(u) at cut faces and blows up
// under-resolved shear layers. Mirrors dns/operators.cpp exactly.
__device__ Real convX(const GpuGrid& g, const Real* u, const Real* v, const Real* w,
                      std::size_t c, int i, int j, int k)
{
        const Real uc  = avg(u[c], u[gidx(g, i + 1, j, k)]);
        const Real ucm = avg(u[gidx(g, i - 1, j, k)], u[c]);
        const Real vjp = avg(v[gidx(g, i, j + 1, k)], v[gidx(g, i - 1, j + 1, k)]);
        const Real vj  = avg(v[c], v[gidx(g, i - 1, j, k)]);
        const Real ujp = avg(u[c], u[gidx(g, i, j + 1, k)]);
        const Real ujm = avg(u[gidx(g, i, j - 1, k)], u[c]);
        const Real wkp = avg(w[gidx(g, i, j, k + 1)], w[gidx(g, i - 1, j, k + 1)]);
        const Real wk  = avg(w[c], w[gidx(g, i - 1, j, k)]);
        const Real ukp = avg(u[c], u[gidx(g, i, j, k + 1)]);
        const Real ukm = avg(u[gidx(g, i, j, k - 1)], u[c]);
        return -((uc * uc - ucm * ucm) / g.hx
               + (vjp * ujp - vj * ujm) / g.hy
               + (wkp * ukp - wk * ukm) / g.hz)
             + Real(0.5) * u[c] * avg(cellDiv(g, u, v, w, i - 1, j, k),
                                      cellDiv(g, u, v, w, i, j, k));
}

__device__ Real convY(const GpuGrid& g, const Real* u, const Real* v, const Real* w,
                      std::size_t c, int i, int j, int k)
{
        const Real vc  = avg(v[c], v[gidx(g, i, j + 1, k)]);
        const Real vcm = avg(v[gidx(g, i, j - 1, k)], v[c]);
        const Real uip = avg(u[gidx(g, i + 1, j, k)], u[gidx(g, i + 1, j - 1, k)]);
        const Real ui  = avg(u[c], u[gidx(g, i, j - 1, k)]);
        const Real vip = avg(v[c], v[gidx(g, i + 1, j, k)]);
        const Real vim = avg(v[gidx(g, i - 1, j, k)], v[c]);
        const Real wkp = avg(w[gidx(g, i, j, k + 1)], w[gidx(g, i, j - 1, k + 1)]);
        const Real wk  = avg(w[c], w[gidx(g, i, j - 1, k)]);
        const Real vkp = avg(v[c], v[gidx(g, i, j, k + 1)]);
        const Real vkm = avg(v[gidx(g, i, j, k - 1)], v[c]);
        return -((uip * vip - ui * vim) / g.hx
               + (vc * vc - vcm * vcm) / g.hy
               + (wkp * vkp - wk * vkm) / g.hz)
             + Real(0.5) * v[c] * avg(cellDiv(g, u, v, w, i, j - 1, k),
                                      cellDiv(g, u, v, w, i, j, k));
}

__device__ Real convZ(const GpuGrid& g, const Real* u, const Real* v, const Real* w,
                      std::size_t c, int i, int j, int k)
{
        const Real wc  = avg(w[c], w[gidx(g, i, j, k + 1)]);
        const Real wcm = avg(w[gidx(g, i, j, k - 1)], w[c]);
        const Real uip = avg(u[gidx(g, i + 1, j, k)], u[gidx(g, i + 1, j, k - 1)]);
        const Real ui  = avg(u[c], u[gidx(g, i, j, k - 1)]);
        const Real wip = avg(w[c], w[gidx(g, i + 1, j, k)]);
        const Real wim = avg(w[gidx(g, i - 1, j, k)], w[c]);
        const Real vjp = avg(v[gidx(g, i, j + 1, k)], v[gidx(g, i, j + 1, k - 1)]);
        const Real vj  = avg(v[c], v[gidx(g, i, j, k - 1)]);
        const Real wjp = avg(w[c], w[gidx(g, i, j + 1, k)]);
        const Real wjm = avg(w[gidx(g, i, j - 1, k)], w[c]);
        return -((uip * wip - ui * wim) / g.hx
               + (vjp * wjp - vj * wjm) / g.hy
               + (wc * wc - wcm * wcm) / g.hz)
             + Real(0.5) * w[c] * avg(cellDiv(g, u, v, w, i, j, k - 1),
                                      cellDiv(g, u, v, w, i, j, k));
}

// nu * 7-point Laplacian of q at (i,j,k).
__device__ Real scaledLap(const GpuGrid& g, Real nu, const Real* q, std::size_t c,
                          int i, int j, int k)
{
    const Real ax = Real(1) / (g.hx * g.hx);
    const Real ay = Real(1) / (g.hy * g.hy);
    const Real az = Real(1) / (g.hz * g.hz);
    return nu * (ax * (q[gidx(g, i + 1, j, k)] - 2 * q[c] + q[gidx(g, i - 1, j, k)])
               + ay * (q[gidx(g, i, j + 1, k)] - 2 * q[c] + q[gidx(g, i, j - 1, k)])
               + az * (q[gidx(g, i, j, k + 1)] - 2 * q[c] + q[gidx(g, i, j, k - 1)]));
}

__global__ void kConvection(GpuGrid g, const Real* u, const Real* v, const Real* w,
                            Real* ou, Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    ou[c] = convX(g, u, v, w, c, i, j, k);
    ov[c] = convY(g, u, v, w, c, i, j, k);
    ow[c] = convZ(g, u, v, w, c, i, j, k);
}

// One pass per RK stage: out = a*x + b*(in + dt*rhs), where
// rhs = convection(in) + nu*lap(diff) + A*(in - mean). diff usually IS in;
// embedded-boundary apps pass a ghost-filled copy so only the viscous term
// sees the sub-cell wall (convection keeps the energy-stable masked zeros).
// out must not alias in/diff (stencil).
__global__ void kFusedRhsStage(GpuGrid g, Real nu, Real A, Real mx, Real my, Real mz,
                               Real a, const Real* xu, const Real* xv, const Real* xw,
                               Real b, const Real* u, const Real* v, const Real* w,
                               const Real* du, const Real* dv, const Real* dw,
                               Real dt, Real* ou, Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    Real ru = convX(g, u, v, w, c, i, j, k) + scaledLap(g, nu, du, c, i, j, k)
            + A * (u[c] - mx);
    Real rv = convY(g, u, v, w, c, i, j, k) + scaledLap(g, nu, dv, c, i, j, k)
            + A * (v[c] - my);
    Real rw = convZ(g, u, v, w, c, i, j, k) + scaledLap(g, nu, dw, c, i, j, k)
            + A * (w[c] - mz);
    ou[c] = a * xu[c] + b * (u[c] + dt * ru);
    ov[c] = a * xv[c] + b * (v[c] + dt * rv);
    ow[c] = a * xw[c] + b * (w[c] + dt * rw);
}

__global__ void kAddScaledLaplacian(GpuGrid g, Real nu, const Real* u, const Real* v,
                                    const Real* w, Real* ou, Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Real ax = Real(1) / (g.hx * g.hx);
    const Real ay = Real(1) / (g.hy * g.hy);
    const Real az = Real(1) / (g.hz * g.hz);
    const std::size_t xp = gidx(g, i + 1, j, k), xm = gidx(g, i - 1, j, k);
    const std::size_t yp = gidx(g, i, j + 1, k), ym = gidx(g, i, j - 1, k);
    const std::size_t zp = gidx(g, i, j, k + 1), zm = gidx(g, i, j, k - 1);
    ou[c] += nu * (ax * (u[xp] - 2 * u[c] + u[xm]) + ay * (u[yp] - 2 * u[c] + u[ym])
                 + az * (u[zp] - 2 * u[c] + u[zm]));
    ov[c] += nu * (ax * (v[xp] - 2 * v[c] + v[xm]) + ay * (v[yp] - 2 * v[c] + v[ym])
                 + az * (v[zp] - 2 * v[c] + v[zm]));
    ow[c] += nu * (ax * (w[xp] - 2 * w[c] + w[xm]) + ay * (w[yp] - 2 * w[c] + w[ym])
                 + az * (w[zp] - 2 * w[c] + w[zm]));
}

__global__ void kAddForcing(std::size_t n, Real A, Real mx, Real my, Real mz,
                            const Real* u, const Real* v, const Real* w,
                            Real* ou, Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    ou[c] += A * (u[c] - mx);
    ov[c] += A * (v[c] - my);
    ow[c] += A * (w[c] - mz);
}

__global__ void kSubtractMean(std::size_t n, Real mx, Real my, Real mz, Real* u,
                              Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    u[c] -= mx;
    v[c] -= my;
    w[c] -= mz;
}

__global__ void kRkCombine(std::size_t n, Real a, const Real* xu, const Real* xv,
                           const Real* xw, Real b, const Real* zu, const Real* zv,
                           const Real* zw, Real dt, const Real* ru, const Real* rv,
                           const Real* rw, Real* ou, Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    ou[c] = a * xu[c] + b * (zu[c] + dt * ru[c]);
    ov[c] = a * xv[c] + b * (zv[c] + dt * rv[c]);
    ow[c] = a * xw[c] + b * (zw[c] + dt * rw[c]);
}

} // namespace

void divergence(const GpuGrid& g, const DeviceField& U, DeviceScalar& out, Real scale)
{
    kDivergence<<<launchBlocks(g.size()), blockSize>>>(g, U.u.data(), U.v.data(),
                                                       U.w.data(), out.data(), scale);
    checkLastKernel("divergence");
}

void subtractGradient(const GpuGrid& g, const DeviceScalar& p, Real scale, DeviceField& U)
{
    kSubtractGradient<<<launchBlocks(g.size()), blockSize>>>(g, p.data(), scale,
                                                             U.u.data(), U.v.data(),
                                                             U.w.data());
    checkLastKernel("subtractGradient");
}

void convection(const GpuGrid& g, const DeviceField& U, DeviceField& out)
{
    kConvection<<<launchBlocks(g.size()), blockSize>>>(g, U.u.data(), U.v.data(),
                                                       U.w.data(), out.u.data(),
                                                       out.v.data(), out.w.data());
    checkLastKernel("convection");
}

void addScaledLaplacian(const GpuGrid& g, Real nu, const DeviceField& U, DeviceField& out)
{
    kAddScaledLaplacian<<<launchBlocks(g.size()), blockSize>>>(
        g, nu, U.u.data(), U.v.data(), U.w.data(), out.u.data(), out.v.data(),
        out.w.data());
    checkLastKernel("addScaledLaplacian");
}

void addForcing(const GpuGrid& g, Real A, Real mx, Real my, Real mz,
                const DeviceField& U, DeviceField& out)
{
    kAddForcing<<<launchBlocks(g.size()), blockSize>>>(
        g.size(), A, mx, my, mz, U.u.data(), U.v.data(), U.w.data(), out.u.data(),
        out.v.data(), out.w.data());
    checkLastKernel("addForcing");
}

void rkCombine(const GpuGrid& g, Real a, const DeviceField& x, Real b,
               const DeviceField& z, Real dt, const DeviceField& r, DeviceField& out)
{
    kRkCombine<<<launchBlocks(g.size()), blockSize>>>(
        g.size(), a, x.u.data(), x.v.data(), x.w.data(), b, z.u.data(), z.v.data(),
        z.w.data(), dt, r.u.data(), r.v.data(), r.w.data(), out.u.data(),
        out.v.data(), out.w.data());
    checkLastKernel("rkCombine");
}

void fusedRhsStage(const GpuGrid& g, Real nu, Real A, Real mx, Real my, Real mz,
                   Real a, const DeviceField& x, Real b, const DeviceField& in,
                   const DeviceField& diff, Real dt, DeviceField& out)
{
    kFusedRhsStage<<<launchBlocks(g.size()), blockSize>>>(
        g, nu, A, mx, my, mz, a, x.u.data(), x.v.data(), x.w.data(), b, in.u.data(),
        in.v.data(), in.w.data(), diff.u.data(), diff.v.data(), diff.w.data(), dt,
        out.u.data(), out.v.data(), out.w.data());
    checkLastKernel("fusedRhsStage");
}

void fusedRhsStage(const GpuGrid& g, Real nu, Real A, Real mx, Real my, Real mz,
                   Real a, const DeviceField& x, Real b, const DeviceField& in,
                   Real dt, DeviceField& out)
{
    fusedRhsStage(g, nu, A, mx, my, mz, a, x, b, in, in, dt, out);
}

void subtractMean(const GpuGrid& g, Real mx, Real my, Real mz, DeviceField& U)
{
    kSubtractMean<<<launchBlocks(g.size()), blockSize>>>(g.size(), mx, my, mz,
                                                         U.u.data(), U.v.data(),
                                                         U.w.data());
    checkLastKernel("subtractMean");
}


// --- Chorin projection (was projection.hip) -------------------------------
void project(const GpuGrid& g, DevicePoissonSolver& poisson, Real dt, DeviceField& U,
             DeviceProjectionScratch& s)
{
    assert(dt > 0);
    divergence(g, U, s.div, Real(1) / dt);
    poisson.solve(s.div, s.phi);
    subtractGradient(g, s.phi, dt, U);
}

} // namespace dns::gpu
