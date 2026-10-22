#include "dns/gpu/cutpoisson.hpp"

#include "dns/gpu/device_util.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include "dns/gpu/internal/reduce.hpp"

#include <rocprim/iterator/transform_iterator.hpp>

#include <cassert>
#include <climits>
#include <cmath>
#include <utility>

namespace dns::gpu {

using internal::reduce;
using internal::reduceInto;

namespace {

__global__ void kApplyCut(GpuGrid g, const Real* __restrict__ lam,
                          const Real* __restrict__ ax, const Real* __restrict__ ay,
                          const Real* __restrict__ az, const Real* __restrict__ x,
                          Real* y)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (lam[c] <= 0) { y[c] = 0; return; }
    const Real ix = Real(1) / (g.hx * g.hx), iy = Real(1) / (g.hy * g.hy),
               iz = Real(1) / (g.hz * g.hz);
    const Offsets o = offsetsAt(g, i, j, k);
    const std::size_t xp = c + o.p[0], xm = c + o.m[0];
    const std::size_t yp = c + o.p[1], ym = c + o.m[1];
    const std::size_t zp = c + o.p[2], zm = c + o.m[2];
    const Real m = ix * (ax[xp] * (x[xp] - x[c]) - ax[c] * (x[c] - x[xm]))
                 + iy * (ay[yp] * (x[yp] - x[c]) - ay[c] * (x[c] - x[ym]))
                 + iz * (az[zp] * (x[zp] - x[c]) - az[c] * (x[c] - x[zm]));
    y[c] = -m;
}

// r = -(rhs) - (useQ ? q : 0) on fluid cells, 0 on solid (b = -rhs, r = b - A p).
__global__ void kResidual(std::size_t n, const Real* lam, const Real* rhs,
                          const Real* q, int useQ, Real* r)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    r[c] = lam[c] > 0 ? -rhs[c] - (useQ ? q[c] : Real(0)) : Real(0);
}

__global__ void kNegate(std::size_t n, const Real* x, Real* y)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    y[c] = -x[c];
}

// mean = sum/nFluid, once. The FP64 division (not reciprocal-multiply, for
// bit-parity with the CPU oracle) runs at 1/30 FP32 rate on gfx1100, so it
// must not be repeated per thread in the subtract pass.
__global__ void kFluidMean(const double* sum, double nFluid, Real* mean)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) *mean = Real(sum[0] / nFluid);
}

// x = fluid ? x - mean : 0. mean read from device (no host sync).
__global__ void kSubtractFluidMean(std::size_t n, const Real* lam, const Real* mean,
                                   Real* x)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    x[c] = lam[c] > 0 ? x[c] - mean[0] : Real(0);
}

// p += alpha d; r -= alpha q, with alpha read from device (the fused CG update).
__global__ void kCgUpdate(std::size_t n, const Real* alpha, const Real* d,
                          const Real* q, Real* p, Real* r)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const Real a = alpha[0];
    p[c] += a * d[c];
    r[c] -= a * q[c];
}

// d = z + beta d, with beta read from device (the CG direction update).
__global__ void kCgDirection(std::size_t n, const Real* z, const Real* beta, Real* d)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    d[c] = z[c] + beta[0] * d[c];
}

// alpha = (dq > 0) ? rz/dq : 0. Single thread; keeps the CG scalar on device.
__global__ void kAlpha(const double* rz, const double* dq, Real* alpha)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *alpha = dq[0] > 0.0 ? Real(rz[0] / dq[0]) : Real(0);
}

// beta = rzNew/rz, then rz <- rzNew (carry the preconditioned dot forward).
__global__ void kBeta(double* rz, const double* rzNew, Real* beta)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *beta = rz[0] > 0.0 ? Real(rzNew[0] / rz[0]) : Real(0);
        rz[0] = rzNew[0];
    }
}


// One reduction over n items through a transform functor, double accumulator.
// (Same pattern as reductions.hip; kept local so hipCUB/rocPRIM stays in TUs.)

// Same reduction, but the result is left in the device buffer dst -- NO host
// copy, so it never synchronizes. Used inside the CG loop to keep alpha/beta
// and the mean on device across iterations.

struct FluidDot
{
    const Real *lam, *a, *b;
    __host__ __device__ double operator()(int c) const
    {
        return lam[c] > 0 ? double(a[c]) * double(b[c]) : 0.0;
    }
};

struct FluidSum
{
    const Real *lam, *x;
    __host__ __device__ double operator()(int c) const
    {
        return lam[c] > 0 ? double(x[c]) : 0.0;
    }
};

struct PlainSum
{
    const Real* x;
    __host__ __device__ double operator()(int c) const { return double(x[c]); }
};

struct PlainAbs
{
    const Real* x;
    __host__ __device__ double operator()(int c) const
    {
        return std::abs(double(x[c]));
    }
};

} // namespace

std::expected<DeviceCutPoissonSolver, hipError_t>
DeviceCutPoissonSolver::make(const GpuGrid& g, const DeviceCutGeometry& cut,
                             int maxIters, double tol, PoissonBc bc)
{
    auto fft = DevicePoissonSolver::make(g, bc);
    if (!fft) return std::unexpected(fft.error());
    DeviceScalar ws[5];
    for (auto& w : ws)
    {
        auto b = DeviceScalar::make(g.size());
        if (!b) return std::unexpected(b.error());
        w = std::move(*b);
    }
    auto red = ReduceScratch::make();
    if (!red) return std::unexpected(red.error());
    auto sc = DeviceBuffer<double>::make(6); // rr, dq, rz, rzNew, sum, bb
    if (!sc) return std::unexpected(sc.error());
    auto ab = DeviceBuffer<Real>::make(3); // alpha, beta, mean
    if (!ab) return std::unexpected(ab.error());
    return DeviceCutPoissonSolver(g, cut, maxIters, tol, std::move(*fft),
                                  std::move(ws[0]), std::move(ws[1]), std::move(ws[2]),
                                  std::move(ws[3]), std::move(ws[4]), std::move(*red),
                                  std::move(*sc), std::move(*ab));
}

DeviceCutPoissonSolver::DeviceCutPoissonSolver(
    const GpuGrid& g, const DeviceCutGeometry& cut, int maxIters, double tol,
    DevicePoissonSolver fft, DeviceScalar r, DeviceScalar z, DeviceScalar d,
    DeviceScalar q, DeviceScalar t, ReduceScratch red, DeviceBuffer<double> sc,
    DeviceBuffer<Real> ab)
    : g_(g), cut_(&cut), maxIters_(maxIters), tol_(tol), fft_(std::move(fft)),
      r_(std::move(r)), z_(std::move(z)), d_(std::move(d)), q_(std::move(q)),
      t_(std::move(t)), red_(std::move(red)), sc_(std::move(sc)), ab_(std::move(ab))
{
}

void DeviceCutPoissonSolver::applyA(const DeviceScalar& x, DeviceScalar& y)
{
    kApplyCut<<<launchBlocks(g_.size()), blockSize>>>(
        g_, cut_->lambda.data(), cut_->ax.data(), cut_->ay.data(), cut_->az.data(),
        x.data(), y.data());
    checkLastKernel("applyCut");
}

// z = (-lap)^-1 r via the periodic FFT solve, restricted back to the fluid
// subspace (solid entries zeroed, fluid mean removed) to keep PCG symmetric.
// The restriction is load-bearing: without it the effective preconditioner is
// no longer P B P (z carries an O(1) fluid mean from the solid-cell values)
// and the iteration loses positive definiteness and diverges.
void DeviceCutPoissonSolver::precondition(const DeviceScalar& r, DeviceScalar& z)
{
    kNegate<<<launchBlocks(g_.size()), blockSize>>>(g_.size(), r.data(), t_.data());
    checkLastKernel("negate");
    fft_.solve(t_, z);
    projectMean(z);
}

void DeviceCutPoissonSolver::projectMean(DeviceScalar& x)
{
    double* sum = sc_.data() + 4; // reuse the [sum] slot; result stays on device
    record(reduceInto(red_.tmp, g_.size(), FluidSum{cut_->lambda.data(), x.data()},
                      rocprim::plus<double>(), 0.0, sum));
    kFluidMean<<<1, 1>>>(sum, double(cut_->nFluid), ab_.data() + 2);
    checkLastKernel("fluidMean");
    kSubtractFluidMean<<<launchBlocks(g_.size()), blockSize>>>(
        g_.size(), cut_->lambda.data(), ab_.data() + 2, x.data());
    checkLastKernel("subtractFluidMean");
}

void DeviceCutPoissonSolver::fluidDotInto(const DeviceScalar& a, const DeviceScalar& b,
                                          double* dst)
{
    record(reduceInto(red_.tmp, g_.size(),
                      FluidDot{cut_->lambda.data(), a.data(), b.data()},
                      rocprim::plus<double>(), 0.0, dst));
}

void DeviceCutPoissonSolver::record(hipError_t e)
{
    if (e != hipSuccess && red_.err == hipSuccess) red_.err = e;
}

hipError_t DeviceCutPoissonSolver::status() const
{
    return red_.err != hipSuccess ? red_.err : fft_.status();
}

PcgResult DeviceCutPoissonSolver::solve(const DeviceScalar& rhs, DeviceScalar& p,
                                        bool warmStart)
{
    assert(rhs.size() == g_.size() && p.size() == g_.size());
    const std::size_t n = g_.size();
    // sc_ slots: [0]=rr [1]=dq [2]=rz [3]=rzNew [4]=sum(projectMean) [5]=bb.
    // Every CG scalar lives here; the host reads only rr+dq once per iteration
    // to drive the loop -- alpha, beta and the mean are computed on device.
    double* const RR = sc_.data() + 0;
    double* const DQ = sc_.data() + 1;
    double* const RZ = sc_.data() + 2;
    double* const RZN = sc_.data() + 3;
    double* const BB = sc_.data() + 5;

    // Target is relative to ||b|| (b = -rhs, projected), NOT the initial
    // residual: a warm start begins with a tiny residual, and scaling the target
    // to it would demand unreachable FP32 precision and drive CG into breakdown.
    kNegate<<<launchBlocks(n), blockSize>>>(n, rhs.data(), t_.data());
    checkLastKernel("negate");
    projectMean(t_);
    fluidDotInto(t_, t_, BB);
    // Every host read starts as NaN: a copy that fails leaves it NaN, and the
    // finiteness tests below then report a failed solve instead of a value.
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    double bb = kNaN;
    record(hipMemcpy(&bb, BB, sizeof(double), hipMemcpyDeviceToHost)); // once/solve
    const double target = tol_ * tol_ * bb;

    if (!warmStart) record(hipMemset(p.data(), 0, n * sizeof(Real)));
    else
    {
        projectMean(p);
        applyA(p, q_);
    }
    kResidual<<<launchBlocks(n), blockSize>>>(n, cut_->lambda.data(), rhs.data(),
                                              q_.data(), warmStart ? 1 : 0, r_.data());
    checkLastKernel("residual");
    projectMean(r_);
    fluidDotInto(r_, r_, RR);

    precondition(r_, z_);
    record(hipMemcpy(d_.data(), z_.data(), n * sizeof(Real),
                     hipMemcpyDeviceToDevice));
    fluidDotInto(r_, z_, RZ);

    double rr = kNaN;
    record(hipMemcpy(&rr, RR, sizeof(double), hipMemcpyDeviceToHost)); // initial guard
    double chk[2]{kNaN, kNaN}; // [rr, dq] pulled together each iteration (one sync)

    // NaN compares false against everything, so `rr > target` alone would exit
    // the loop at iteration 0 and report a perfect solve on a field that has
    // already diverged. Require rr to be a finite number to CONTINUE, and
    // report non-convergence when it is not.
    //
    // The convergence test is evaluated EVERY iteration, and the D2H copy that
    // drives it is the loop's only host synchronization. Rationing it (test
    // every K, K > 1) is the obvious optimization and it does not pay -- it was
    // built and measured, 2026-07-26, cylinder Re60 256^2 tunnel, one iteration
    // costing 0.316 ms against a sync costing ~0.02 ms:
    //     K = 1: 39.8 pcg/step, 12.9 ms/step
    //     K = 2: 41.5 pcg/step, 13.4 ms/step
    //     K = 4: 44.0 pcg/step, 13.9 ms/step
    // The overshoot (up to K-1 wasted iterations per solve) costs an order of
    // magnitude more than the syncs it removes, in 2D and in 3D alike (sphere
    // 128^3: 52 -> 56 pcg/step, 58.0 -> 60.6 ms/step). Do not re-try it without
    // first making a CG iteration cheap.
    int it = 0;
    if (!std::isfinite(rr)) return {maxIters_, false, false};
    for (; it < maxIters_ && std::isfinite(rr) && rr > target; ++it)
    {
        // projectMean(q_) is load-bearing in FP32: M(d) telescopes to a zero
        // fluid sum only in exact arithmetic, and the roundoff mean leaks into
        // r via r -= alpha q, where it sits in the null space of A and stalls
        // the residual against a relative target. Keep it.
        applyA(d_, q_);
        projectMean(q_);
        fluidDotInto(d_, q_, DQ);
        kAlpha<<<1, 1>>>(RZ, DQ, ab_.data());        // alpha = rz/dq (guards dq<=0)
        checkLastKernel("alpha");
        kCgUpdate<<<launchBlocks(n), blockSize>>>(n, ab_.data(), d_.data(), q_.data(),
                                                  p.data(), r_.data());
        checkLastKernel("cgUpdate");
        fluidDotInto(r_, r_, RR);
        // Single host round-trip: RR and DQ are the contiguous slots [0],[1].
        chk[0] = chk[1] = kNaN;
        record(hipMemcpy(chk, RR, 2 * sizeof(double), hipMemcpyDeviceToHost));
        rr = chk[0];
        if (!std::isfinite(rr)) return {maxIters_, false, false}; // diverged
        // Converged, or a breakdown (dq <= 0): the loop stops either way, and
        // the verdict below says which -- a breakdown above target is NOT a
        // converged solve.
        if (rr <= target || chk[1] <= 0.0) { ++it; break; }
        precondition(r_, z_);
        fluidDotInto(r_, z_, RZN);
        kBeta<<<1, 1>>>(RZ, RZN, ab_.data() + 1);    // beta=rzNew/rz, then rz<-rzNew
        checkLastKernel("beta");
        kCgDirection<<<launchBlocks(n), blockSize>>>(n, z_.data(), ab_.data() + 1,
                                                     d_.data());
        checkLastKernel("cgDirection");
    }
    projectMean(p);
    return {it, rr <= target, true};
}

int projectCut(const GpuGrid& g, const DeviceCutGeometry& cut,
               DeviceCutPoissonSolver& poisson, Real dt, DeviceField& U,
               DeviceCutProjectionScratch& s)
{
    assert(dt > 0);
    divergenceCut(g, cut, U, s.div, Real(1) / dt);
    const int it = poisson.solve(s.div, s.phi, true).iters;
    subtractGradientCut(g, cut, s.phi, dt, U);
    maskSolidFaces(g, cut, U);
    return it;
}

} // namespace dns::gpu
