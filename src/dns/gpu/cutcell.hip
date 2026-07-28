#include "dns/gpu/cutcell.hpp"

#include "dns/gpu/device_util.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include "dns/gpu/internal/reduce.hpp"

#include <rocprim/iterator/transform_iterator.hpp>

#include <cassert>
#include <climits>
#include <cmath>
#include <utility>
#include <vector>

namespace dns::gpu {

using internal::reduce;

namespace {

constexpr Real openTol = Real(CutGeometry::openTol);

__global__ void kDivergenceCut(GpuGrid g, const Real* __restrict__ ax,
                               const Real* __restrict__ ay,
                               const Real* __restrict__ az,
                               const Real* __restrict__ u, const Real* __restrict__ v,
                               const Real* __restrict__ w, Real* out, Real scale)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const std::size_t xp = c + o.p[0], yp = c + o.p[1], zp = c + o.p[2];
    out[c] = scale * ((ax[xp] * u[xp] - ax[c] * u[c]) / g.hx
                    + (ay[yp] * v[yp] - ay[c] * v[c]) / g.hy
                    + (az[zp] * w[zp] - az[c] * w[c]) / g.hz);
}

__global__ void kSubtractGradientCut(GpuGrid g, const Real* __restrict__ ax,
                                     const Real* __restrict__ ay,
                                     const Real* __restrict__ az,
                                     const Real* __restrict__ p, Real scale,
                                     Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    if (ax[c] > openTol) u[c] -= scale * (p[c] - p[c + o.m[0]]) / g.hx;
    if (ay[c] > openTol) v[c] -= scale * (p[c] - p[c + o.m[1]]) / g.hy;
    if (az[c] > openTol) w[c] -= scale * (p[c] - p[c + o.m[2]]) / g.hz;
}

// exp(-dt*(1-lambda_face)/eta) split decay per face component; lambda_face is
// the average of the two adjacent cell fractions (the same staggering the
// cut-flux volume uses). Interior faces (lambda_face == 1) are untouched
// EXACTLY (factor branch, not exp(0) roundoff).
__global__ void kPenalize(GpuGrid g, const Real* __restrict__ lam, Real idte,
                          Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Real lu = Real(0.5) * (lam[c + o.m[0]] + lam[c]);
    const Real lv = Real(0.5) * (lam[c + o.m[1]] + lam[c]);
    const Real lw = Real(0.5) * (lam[c + o.m[2]] + lam[c]);
    if (lu < Real(1)) u[c] *= exp(-idte * (Real(1) - lu));
    if (lv < Real(1)) v[c] *= exp(-idte * (Real(1) - lv));
    if (lw < Real(1)) w[c] *= exp(-idte * (Real(1) - lw));
}

__global__ void kMaskSolidFaces(std::size_t n, const Real* __restrict__ ax,
                                const Real* __restrict__ ay,
                                const Real* __restrict__ az, Real* u, Real* v,
                                Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    if (ax[c] <= openTol) u[c] = 0;
    if (ay[c] <= openTol) v[c] = 0;
    if (az[c] <= openTol) w[c] = 0;
}

// y = -M(x): the negated apertured Laplacian, SPD on the fluid subspace
// modulo the constant mode; zero on solid cells.
// One reduction over n items through a transform functor, double accumulator.
// (Same pattern as reductions.hip; kept local so hipCUB/rocPRIM stays in TUs.)

struct AbsDivCut
{
    GpuGrid g;
    const Real *lam, *ax, *ay, *az, *u, *v, *w;
    __host__ __device__ double operator()(int c) const
    {
        if (lam[c] <= 0) return 0.0;
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        const Offsets o = offsetsAt(g, i, j, k);
        const std::size_t xp = c + o.p[0], yp = c + o.p[1], zp = c + o.p[2];
        const double d =
            (double(ax[xp]) * u[xp] - double(ax[c]) * u[c]) / g.hx
          + (double(ay[yp]) * v[yp] - double(ay[c]) * v[c]) / g.hy
          + (double(az[zp]) * w[zp] - double(az[c]) * w[c]) / g.hz;
        return fabs(d);
    }
};

std::expected<DeviceScalar, hipError_t> uploadNarrow(const Scalar& src)
{
    auto buf = DeviceScalar::make(src.size());
    if (!buf) return buf;
    std::vector<Real> tmp(src.begin(), src.end());
    if (auto e = buf->upload(tmp.data()); e != hipSuccess) return std::unexpected(e);
    return buf;
}

} // namespace

std::expected<DeviceCutGeometry, hipError_t> DeviceCutGeometry::make(const CutGeometry& cut)
{
    auto lam = uploadNarrow(cut.lambda);
    if (!lam) return std::unexpected(lam.error());
    auto ax = uploadNarrow(cut.ax);
    if (!ax) return std::unexpected(ax.error());
    auto ay = uploadNarrow(cut.ay);
    if (!ay) return std::unexpected(ay.error());
    auto az = uploadNarrow(cut.az);
    if (!az) return std::unexpected(az.error());
    std::size_t nFluid = 0;
    for (double l : cut.lambda)
        if (l > 0.0) ++nFluid;
    assert(nFluid > 0);
    return DeviceCutGeometry{std::move(*lam), std::move(*ax), std::move(*ay),
                             std::move(*az), nFluid};
}

void divergenceCut(const GpuGrid& g, const DeviceCutGeometry& cut,
                   const DeviceField& U, DeviceScalar& out, Real scale)
{
    kDivergenceCut<<<launchBlocks(g.size()), blockSize>>>(
        g, cut.ax.data(), cut.ay.data(), cut.az.data(), U.u.data(), U.v.data(),
        U.w.data(), out.data(), scale);
    checkLastKernel("divergenceCut");
}

void subtractGradientCut(const GpuGrid& g, const DeviceCutGeometry& cut,
                         const DeviceScalar& p, Real scale, DeviceField& U)
{
    kSubtractGradientCut<<<launchBlocks(g.size()), blockSize>>>(
        g, cut.ax.data(), cut.ay.data(), cut.az.data(), p.data(), scale,
        U.u.data(), U.v.data(), U.w.data());
    checkLastKernel("subtractGradientCut");
}

void maskSolidFaces(const GpuGrid& g, const DeviceCutGeometry& cut, DeviceField& U)
{
    kMaskSolidFaces<<<launchBlocks(g.size()), blockSize>>>(
        g.size(), cut.ax.data(), cut.ay.data(), cut.az.data(), U.u.data(),
        U.v.data(), U.w.data());
    checkLastKernel("maskSolidFaces");
}

void penalizeSolidBand(const GpuGrid& g, const DeviceCutGeometry& cut, Real eta,
                       Real dt, DeviceField& U)
{
    kPenalize<<<launchBlocks(g.size()), blockSize>>>(
        g, cut.lambda.data(), dt / eta, U.u.data(), U.v.data(), U.w.data());
    checkLastKernel("penalize");
}

double maxAbsDivergenceCut(const GpuGrid& g, const DeviceCutGeometry& cut,
                           const DeviceField& U, ReduceScratch& s)
{
    return reduce(s, g.size(),
                  AbsDivCut{g, cut.lambda.data(), cut.ax.data(), cut.ay.data(),
                            cut.az.data(), U.u.data(), U.v.data(), U.w.data()},
                  rocprim::maximum<double>(), 0.0);
}

} // namespace dns::gpu
