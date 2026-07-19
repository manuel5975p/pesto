#include "dns/gpu/reductions.hpp"

#include "dns/gpu/device_util.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <cassert>
#include <climits>

namespace dns::gpu {

std::expected<ReduceScratch, hipError_t> ReduceScratch::make()
{
    auto out = DeviceBuffer<double>::make(1);
    if (!out) return std::unexpected(out.error());
    return ReduceScratch{{}, std::move(*out)};
}

namespace {

// One reduction over n items through a transform functor, double accumulator.
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

struct SqSum3
{
    const Real *u, *v, *w;
    __host__ __device__ double operator()(int c) const
    {
        const double a = u[c], b = v[c], d = w[c];
        return a * a + b * b + d * d;
    }
};

struct CompSum
{
    const Real* q;
    __host__ __device__ double operator()(int c) const { return double(q[c]); }
};

struct FlucSq
{
    const Real *u, *v, *w;
    double mx, my, mz;
    __host__ __device__ double operator()(int c) const
    {
        const double a = u[c] - mx, b = v[c] - my, d = w[c] - mz;
        return a * a + b * b + d * d;
    }
};

struct CourantSum
{
    const Real *u, *v, *w;
    double ihx, ihy, ihz;
    __host__ __device__ double operator()(int c) const
    {
        return fabs(double(u[c])) * ihx + fabs(double(v[c])) * ihy
             + fabs(double(w[c])) * ihz;
    }
};

struct AbsDiv
{
    GpuGrid g;
    const Real *u, *v, *w;
    __host__ __device__ double operator()(int c) const
    {
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        const double d = (double(u[gidx(g, i + 1, j, k)]) - double(u[c])) / g.hx
                       + (double(v[gidx(g, i, j + 1, k)]) - double(v[c])) / g.hy
                       + (double(w[gidx(g, i, j, k + 1)]) - double(w[c])) / g.hz;
        return fabs(d);
    }
};

struct GradSq
{
    GpuGrid g;
    const Real *u, *v, *w;
    __host__ __device__ double operator()(int c) const
    {
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        const double ix = 1.0 / g.hx, iy = 1.0 / g.hy, iz = 1.0 / g.hz;
        const std::size_t xp = gidx(g, i + 1, j, k);
        const std::size_t yp = gidx(g, i, j + 1, k);
        const std::size_t zp = gidx(g, i, j, k + 1);
        double s = 0.0, d;
        d = (double(u[xp]) - u[c]) * ix; s += d * d;
        d = (double(u[yp]) - u[c]) * iy; s += d * d;
        d = (double(u[zp]) - u[c]) * iz; s += d * d;
        d = (double(v[xp]) - v[c]) * ix; s += d * d;
        d = (double(v[yp]) - v[c]) * iy; s += d * d;
        d = (double(v[zp]) - v[c]) * iz; s += d * d;
        d = (double(w[xp]) - w[c]) * ix; s += d * d;
        d = (double(w[yp]) - w[c]) * iy; s += d * d;
        d = (double(w[zp]) - w[c]) * iz; s += d * d;
        return s;
    }
};

} // namespace

double kineticEnergy(const GpuGrid& g, const DeviceField& U, ReduceScratch& s)
{
    const double sum = reduce(s, g.size(), SqSum3{U.u.data(), U.v.data(), U.w.data()},
                              rocprim::plus<double>(), 0.0);
    return 0.5 * sum * double(g.hx) * double(g.hy) * double(g.hz);
}

Mean3 meanVelocity(const GpuGrid& g, const DeviceField& U, ReduceScratch& s)
{
    const double inv = 1.0 / double(g.size());
    return {reduce(s, g.size(), CompSum{U.u.data()}, rocprim::plus<double>(), 0.0) * inv,
            reduce(s, g.size(), CompSum{U.v.data()}, rocprim::plus<double>(), 0.0) * inv,
            reduce(s, g.size(), CompSum{U.w.data()}, rocprim::plus<double>(), 0.0) * inv};
}

double fluctuationEnergy(const GpuGrid& g, const DeviceField& U, ReduceScratch& s)
{
    const Mean3 m = meanVelocity(g, U, s);
    const double sum = reduce(
        s, g.size(), FlucSq{U.u.data(), U.v.data(), U.w.data(), m.x, m.y, m.z},
        rocprim::plus<double>(), 0.0);
    return 0.5 * sum / double(g.size());
}

double maxCourantPerDt(const GpuGrid& g, const DeviceField& U, ReduceScratch& s)
{
    return reduce(s, g.size(),
                  CourantSum{U.u.data(), U.v.data(), U.w.data(), 1.0 / g.hx,
                             1.0 / g.hy, 1.0 / g.hz},
                  rocprim::maximum<double>(), 0.0);
}

double maxAbsDivergence(const GpuGrid& g, const DeviceField& U, ReduceScratch& s)
{
    return reduce(s, g.size(), AbsDiv{g, U.u.data(), U.v.data(), U.w.data()},
                  rocprim::maximum<double>(), 0.0);
}

double velocityGradientSq(const GpuGrid& g, const DeviceField& U, ReduceScratch& s)
{
    const double sum = reduce(s, g.size(), GradSq{g, U.u.data(), U.v.data(), U.w.data()},
                              rocprim::plus<double>(), 0.0);
    return sum / double(g.size());
}

} // namespace dns::gpu
