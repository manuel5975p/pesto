// The rocPRIM transform-reduce driver, once.
//
// INTERNAL (gpu/internal/) and included only from .hip TUs, so no rocPRIM or
// HIP declaration reaches a public header -- the constraint the per-TU copies
// existed to satisfy. Same pattern as gpu/internal/comp_flux.hpp.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/reductions.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <cassert>
#include <climits>
#include <cstddef>
#include <limits>

namespace dns::gpu::internal {

// op-reduce f(0..n-1) into the device double `dst` -- NO host copy, so it
// never synchronizes. Used inside the CG loop to keep alpha/beta and the mean
// on device across iterations. Scratch grows on demand and is reused.
//
// Any failure (the size query, the scratch growth, the reduction itself)
// writes a quiet NaN to `dst` and returns the error: a failed reduction must
// never read as a plausible number -- every consumer of these values (dt, a
// convergence test, a reported coefficient) then fails loudly on the NaN.
template <class F, class Op>
hipError_t reduceInto(DeviceBuffer<unsigned char>& tmp, std::size_t n, F f, Op op,
                      double init, double* dst)
{
    assert(n <= std::size_t(INT_MAX));
    const auto in =
        rocprim::make_transform_iterator(rocprim::counting_iterator<int>(0), f);
    std::size_t bytes = 0;
    hipError_t e = rocprim::reduce(nullptr, bytes, in, dst, init, n, op);
    if (e == hipSuccess && tmp.size() < bytes)
    {
        auto grown = DeviceBuffer<unsigned char>::make(bytes);
        if (grown) tmp = std::move(*grown);
        else e = grown.error();
    }
    if (e == hipSuccess)
    {
        bytes = tmp.size();
        e = rocprim::reduce(tmp.data(), bytes, in, dst, init, n, op);
    }
    if (e != hipSuccess)
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        (void)hipMemcpy(dst, &nan, sizeof(double), hipMemcpyHostToDevice);
    }
    return e;
}

// The same reduction into a host double: reduceInto plus one copy. Returns a
// quiet NaN on any failure and records the first error in s.err.
template <class F, class Op>
double reduce(ReduceScratch& s, std::size_t n, F f, Op op, double init)
{
    hipError_t e = reduceInto(s.tmp, n, f, op, init, s.out.data());
    double h = 0.0;
    if (e == hipSuccess)
        e = hipMemcpy(&h, s.out.data(), sizeof(double), hipMemcpyDeviceToHost);
    if (e == hipSuccess) return h;
    if (s.err == hipSuccess) s.err = e;
    return std::numeric_limits<double>::quiet_NaN();
}

} // namespace dns::gpu::internal
