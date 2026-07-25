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

namespace dns::gpu::internal {

// op-reduce f(0..n-1) into a host double. Scratch grows on demand and is
// reused across calls.
//
// On a scratch-allocation failure the reduction is left to run on whatever
// scratch already exists and rocPRIM reports the shortfall -- it does not
// corrupt memory. `init` is returned in that case. The obvious alternative
// (dereferencing the valueless expected) is UB in Release, where assert() is a
// comment; that is the bug this shape exists to avoid.
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
        if (!grown) return init;
        s.tmp = std::move(*grown);
    }
    bytes = s.tmp.size();
    (void)rocprim::reduce(s.tmp.data(), bytes, in, s.out.data(), init, n, op);
    double h = 0.0;
    (void)hipMemcpy(&h, s.out.data(), sizeof(double), hipMemcpyDeviceToHost);
    return h;
}

// Same reduction, but the result is left in the device buffer `dst` -- NO host
// copy, so it never synchronizes. Used inside the CG loop to keep alpha/beta
// and the mean on device across iterations.
template <class F, class Op>
void reduceInto(DeviceBuffer<unsigned char>& tmp, std::size_t n, F f, Op op,
                double init, double* dst)
{
    assert(n <= std::size_t(INT_MAX));
    const auto in =
        rocprim::make_transform_iterator(rocprim::counting_iterator<int>(0), f);
    std::size_t bytes = 0;
    (void)rocprim::reduce(nullptr, bytes, in, dst, init, n, op);
    if (tmp.size() < bytes)
    {
        auto grown = DeviceBuffer<unsigned char>::make(bytes);
        if (!grown) return;
        tmp = std::move(*grown);
    }
    bytes = tmp.size();
    (void)rocprim::reduce(tmp.data(), bytes, in, dst, init, n, op);
}

} // namespace dns::gpu::internal
