#include "dns/gpu/kernels.hpp"

#include "dns/gpu/projection.hpp"

#include <cassert>

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/stencil.hpp"

namespace dns::gpu {

namespace {

using internal::applyStageTail;
using internal::avg;
using internal::Inv;
using internal::invOf;

// The four plain cell divergences one fused stage needs: at the centre cell and
// at each of its three minus-neighbours. Each convective correction averages
// the centre with one minus-neighbour, so the centre value used to be evaluated
// three times per thread.
struct DivQuad
{
    Real c, xm, ym, zm;
};

// Pre: `o` are the offsets at c. The minus cells differ from c along ONE axis,
// so o's other two offsets stay valid there (index.hpp), and the missing plus
// neighbour along that axis is c itself.
__device__ inline DivQuad divQuad(const Real* __restrict__ u,
                                  const Real* __restrict__ v,
                                  const Real* __restrict__ w, const Offsets& o,
                                  const Inv& iv, std::size_t c)
{
    const std::size_t xm = c + o.m[0], ym = c + o.m[1], zm = c + o.m[2];
    return {.c = (u[c + o.p[0]] - u[c]) * iv.x + (v[c + o.p[1]] - v[c]) * iv.y
               + (w[c + o.p[2]] - w[c]) * iv.z,
            .xm = (u[c] - u[xm]) * iv.x + (v[xm + o.p[1]] - v[xm]) * iv.y
                + (w[xm + o.p[2]] - w[xm]) * iv.z,
            .ym = (u[ym + o.p[0]] - u[ym]) * iv.x + (v[c] - v[ym]) * iv.y
                + (w[ym + o.p[2]] - w[ym]) * iv.z,
            .zm = (u[zm + o.p[0]] - u[zm]) * iv.x + (v[zm + o.p[1]] - v[zm]) * iv.y
                + (w[c] - w[zm]) * iv.z};
}

__global__ void kDivergence(GpuGrid g, const Real* __restrict__ u,
                            const Real* __restrict__ v, const Real* __restrict__ w,
                            Real* out, Real scale)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Inv iv = invOf(g);
    out[c] = scale * ((u[c + o.p[0]] - u[c]) * iv.x
                    + (v[c + o.p[1]] - v[c]) * iv.y
                    + (w[c + o.p[2]] - w[c]) * iv.z);
}

__global__ void kSubtractGradient(GpuGrid g, const Real* __restrict__ p, Real scale,
                                  Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Inv iv = invOf(g);
    u[c] -= scale * (p[c] - p[c + o.m[0]]) * iv.x;
    v[c] -= scale * (p[c] - p[c + o.m[1]]) * iv.y;
    w[c] -= scale * (p[c] - p[c + o.m[2]]) * iv.z;
}

// Convective fluxes per momentum component at face (i,j,k) in skew-symmetric
// form: divergence form -(d(ui uj)/dxj) plus 0.5*ui*div(u) at the face. The
// correction vanishes on a discretely solenoidal field; with cut cells (only
// the apertured divergence is projected to zero) and the FP32 projection
// residual it is what keeps the operator energy-conserving -- the plain
// divergence form injects energy ~ u^2*div(u) at cut faces and diverges
// under-resolved shear layers. Mirrors dns/operators.cpp exactly.
__device__ Real convX(const Real* __restrict__ u, const Real* __restrict__ v,
                      const Real* __restrict__ w, const Offsets& o, const Inv& iv,
                      std::size_t c, const DivQuad& d)
{
    const std::size_t xm = c + o.m[0];
    const Real uc  = avg(u[c], u[c + o.p[0]]);
    const Real ucm = avg(u[xm], u[c]);
    const Real vjp = avg(v[c + o.p[1]], v[xm + o.p[1]]);
    const Real vj  = avg(v[c], v[xm]);
    const Real ujp = avg(u[c], u[c + o.p[1]]);
    const Real ujm = avg(u[c + o.m[1]], u[c]);
    const Real wkp = avg(w[c + o.p[2]], w[xm + o.p[2]]);
    const Real wk  = avg(w[c], w[xm]);
    const Real ukp = avg(u[c], u[c + o.p[2]]);
    const Real ukm = avg(u[c + o.m[2]], u[c]);
    return -((uc * uc - ucm * ucm) * iv.x
           + (vjp * ujp - vj * ujm) * iv.y
           + (wkp * ukp - wk * ukm) * iv.z)
         + Real(0.5) * u[c] * avg(d.xm, d.c);
}

__device__ Real convY(const Real* __restrict__ u, const Real* __restrict__ v,
                      const Real* __restrict__ w, const Offsets& o, const Inv& iv,
                      std::size_t c, const DivQuad& d)
{
    const std::size_t ym = c + o.m[1];
    const Real vc  = avg(v[c], v[c + o.p[1]]);
    const Real vcm = avg(v[ym], v[c]);
    const Real uip = avg(u[c + o.p[0]], u[ym + o.p[0]]);
    const Real ui  = avg(u[c], u[ym]);
    const Real vip = avg(v[c], v[c + o.p[0]]);
    const Real vim = avg(v[c + o.m[0]], v[c]);
    const Real wkp = avg(w[c + o.p[2]], w[ym + o.p[2]]);
    const Real wk  = avg(w[c], w[ym]);
    const Real vkp = avg(v[c], v[c + o.p[2]]);
    const Real vkm = avg(v[c + o.m[2]], v[c]);
    return -((uip * vip - ui * vim) * iv.x
           + (vc * vc - vcm * vcm) * iv.y
           + (wkp * vkp - wk * vkm) * iv.z)
         + Real(0.5) * v[c] * avg(d.ym, d.c);
}

__device__ Real convZ(const Real* __restrict__ u, const Real* __restrict__ v,
                      const Real* __restrict__ w, const Offsets& o, const Inv& iv,
                      std::size_t c, const DivQuad& d)
{
    const std::size_t zm = c + o.m[2];
    const Real wc  = avg(w[c], w[c + o.p[2]]);
    const Real wcm = avg(w[zm], w[c]);
    const Real uip = avg(u[c + o.p[0]], u[zm + o.p[0]]);
    const Real ui  = avg(u[c], u[zm]);
    const Real wip = avg(w[c], w[c + o.p[0]]);
    const Real wim = avg(w[c + o.m[0]], w[c]);
    const Real vjp = avg(v[c + o.p[1]], v[zm + o.p[1]]);
    const Real vj  = avg(v[c], v[zm]);
    const Real wjp = avg(w[c], w[c + o.p[1]]);
    const Real wjm = avg(w[c + o.m[1]], w[c]);
    return -((uip * wip - ui * wim) * iv.x
           + (vjp * wjp - vj * wjm) * iv.y
           + (wc * wc - wcm * wcm) * iv.z)
         + Real(0.5) * w[c] * avg(d.zm, d.c);
}

// nu * 7-point Laplacian of q at c.
__device__ inline Real scaledLap(const Real* __restrict__ q, const Offsets& o,
                                 const Inv& iv, Real nu, std::size_t c)
{
    return nu * (iv.xx * (q[c + o.p[0]] - 2 * q[c] + q[c + o.m[0]])
               + iv.yy * (q[c + o.p[1]] - 2 * q[c] + q[c + o.m[1]])
               + iv.zz * (q[c + o.p[2]] - 2 * q[c] + q[c + o.m[2]]));
}

__global__ void kConvection(GpuGrid g, const Real* __restrict__ u,
                            const Real* __restrict__ v, const Real* __restrict__ w,
                            Real* ou, Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Inv iv = invOf(g);
    const DivQuad d = divQuad(u, v, w, o, iv, c);
    ou[c] = convX(u, v, w, o, iv, c, d);
    ov[c] = convY(u, v, w, o, iv, c, d);
    ow[c] = convZ(u, v, w, o, iv, c, d);
}

// One pass per RK stage: out = a*x + b*(in + dt*rhs), where
// rhs = convection(in) + nu*lap(diff) + A*(in - mean), followed by the optional
// StageTail. diff usually IS in; embedded-boundary apps pass a ghost-filled
// copy so only the viscous term sees the sub-cell wall (convection keeps the
// energy-stable masked zeros).
//
// The six stencil inputs are restrict-qualified: none of them is written here
// (out aliases neither), so their mutual aliasing -- in == diff is the common
// call -- is not a restrict violation, while a store through ou can no longer
// be assumed to clobber them. xu/xv/xw are deliberately NOT qualified: the
// SSP-RK3 closing stage passes out == x.
__global__ void kFusedRhsStage(GpuGrid g, Real nu, Real A, Real mx, Real my, Real mz,
                               Real a, const Real* xu, const Real* xv, const Real* xw,
                               Real b, const Real* __restrict__ u,
                               const Real* __restrict__ v, const Real* __restrict__ w,
                               const Real* __restrict__ du,
                               const Real* __restrict__ dv,
                               const Real* __restrict__ dw, Real dt, StageTail tail,
                               Real* ou, Real* ov, Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Inv iv = invOf(g);
    const DivQuad d = divQuad(u, v, w, o, iv, c);
    const Real ru = convX(u, v, w, o, iv, c, d) + scaledLap(du, o, iv, nu, c)
                  + A * (u[c] - mx);
    const Real rv = convY(u, v, w, o, iv, c, d) + scaledLap(dv, o, iv, nu, c)
                  + A * (v[c] - my);
    const Real rw = convZ(u, v, w, o, iv, c, d) + scaledLap(dw, o, iv, nu, c)
                  + A * (w[c] - mz);
    Real qu = a * xu[c] + b * (u[c] + dt * ru);
    Real qv = a * xv[c] + b * (v[c] + dt * rv);
    Real qw = a * xw[c] + b * (w[c] + dt * rw);
    applyStageTail(tail, o, c, qu, qv, qw);
    ou[c] = qu;
    ov[c] = qv;
    ow[c] = qw;
}

__global__ void kAddScaledLaplacian(GpuGrid g, Real nu, const Real* __restrict__ u,
                                    const Real* __restrict__ v,
                                    const Real* __restrict__ w, Real* ou, Real* ov,
                                    Real* ow)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Inv iv = invOf(g);
    ou[c] += scaledLap(u, o, iv, nu, c);
    ov[c] += scaledLap(v, o, iv, nu, c);
    ow[c] += scaledLap(w, o, iv, nu, c);
}

__global__ void kAddForcing(std::size_t n, Real A, Real mx, Real my, Real mz,
                            const Real* __restrict__ u, const Real* __restrict__ v,
                            const Real* __restrict__ w, Real* ou, Real* ov, Real* ow)
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
                           const Real* zw, Real dt, const Real* __restrict__ ru,
                           const Real* __restrict__ rv, const Real* __restrict__ rw,
                           Real* ou, Real* ov, Real* ow)
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
                   const DeviceField& diff, Real dt, DeviceField& out,
                   const StageTail& tail)
{
    kFusedRhsStage<<<launchBlocks(g.size()), blockSize>>>(
        g, nu, A, mx, my, mz, a, x.u.data(), x.v.data(), x.w.data(), b, in.u.data(),
        in.v.data(), in.w.data(), diff.u.data(), diff.v.data(), diff.w.data(), dt,
        tail, out.u.data(), out.v.data(), out.w.data());
    checkLastKernel("fusedRhsStage");
}

void fusedRhsStage(const GpuGrid& g, Real nu, Real A, Real mx, Real my, Real mz,
                   Real a, const DeviceField& x, Real b, const DeviceField& in,
                   Real dt, DeviceField& out, const StageTail& tail)
{
    fusedRhsStage(g, nu, A, mx, my, mz, a, x, b, in, in, dt, out, tail);
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
