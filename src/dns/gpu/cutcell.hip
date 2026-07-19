#include "dns/gpu/cutcell.hpp"

#include "dns/gpu/device_util.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <cassert>
#include <climits>
#include <cmath>
#include <utility>
#include <vector>

namespace dns::gpu {

namespace {

constexpr Real openTol = Real(CutGeometry::openTol);

__global__ void kDivergenceCut(GpuGrid g, const Real* ax, const Real* ay,
                               const Real* az, const Real* u, const Real* v,
                               const Real* w, Real* out, Real scale)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const std::size_t xp = gidx(g, i + 1, j, k), yp = gidx(g, i, j + 1, k),
                      zp = gidx(g, i, j, k + 1);
    out[c] = scale * ((ax[xp] * u[xp] - ax[c] * u[c]) / g.hx
                    + (ay[yp] * v[yp] - ay[c] * v[c]) / g.hy
                    + (az[zp] * w[zp] - az[c] * w[c]) / g.hz);
}

__global__ void kSubtractGradientCut(GpuGrid g, const Real* ax, const Real* ay,
                                     const Real* az, const Real* p, Real scale,
                                     Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (ax[c] > openTol) u[c] -= scale * (p[c] - p[gidx(g, i - 1, j, k)]) / g.hx;
    if (ay[c] > openTol) v[c] -= scale * (p[c] - p[gidx(g, i, j - 1, k)]) / g.hy;
    if (az[c] > openTol) w[c] -= scale * (p[c] - p[gidx(g, i, j, k - 1)]) / g.hz;
}

__global__ void kMaskSolidFaces(std::size_t n, const Real* ax, const Real* ay,
                                const Real* az, Real* u, Real* v, Real* w)
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
template <class F, class Op>
double reduce(ReduceScratch& s, std::size_t n, F f, Op op, double init)
{
    assert(n <= std::size_t(INT_MAX));
    const auto in =
        rocprim::make_transform_iterator(rocprim::counting_iterator<int>(0), f);

    std::size_t bytes = 0;
    (void)rocprim::reduce(nullptr, bytes, in, s.out.data(), init, n, op);
    if (s.tmp.size() < bytes)
    {
        auto grown = DeviceBuffer<unsigned char>::make(bytes);
        assert(grown && "reduction temp allocation failed");
        s.tmp = std::move(*grown);
    }
    bytes = s.tmp.size();
    (void)rocprim::reduce(s.tmp.data(), bytes, in, s.out.data(), init, n, op);
    double h = 0.0;
    (void)hipMemcpy(&h, s.out.data(), sizeof(double), hipMemcpyDeviceToHost);
    return h;
}

struct AbsDivCut
{
    GpuGrid g;
    const Real *lam, *ax, *ay, *az, *u, *v, *w;
    __host__ __device__ double operator()(int c) const
    {
        if (lam[c] <= 0) return 0.0;
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        const std::size_t xp = gidx(g, i + 1, j, k), yp = gidx(g, i, j + 1, k),
                          zp = gidx(g, i, j, k + 1);
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

double maxAbsDivergenceCut(const GpuGrid& g, const DeviceCutGeometry& cut,
                           const DeviceField& U, ReduceScratch& s)
{
    return reduce(s, g.size(),
                  AbsDivCut{g, cut.lambda.data(), cut.ax.data(), cut.ay.data(),
                            cut.az.data(), U.u.data(), U.v.data(), U.w.data()},
                  rocprim::maximum<double>(), 0.0);
}

} // namespace dns::gpu
