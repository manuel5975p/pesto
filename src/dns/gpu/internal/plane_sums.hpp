// The scratch handling around a per-y-plane moment sum, once.
//
// The three wall-bounded solvers (channel, compressible convection cell,
// Boussinesq convection cell) each launch one thread per y row that sums its
// whole (i,k) plane in FP64 in a fixed order, then read the sums back. The
// sums themselves are per-physics; the buffer growth and the download are not,
// and all three had a copy.
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/gpu/device_buffer.hpp"

#include <cstddef>
#include <vector>

namespace dns::gpu::internal {

// Grows `scratch` to at least n doubles, REUSING it when it is already big
// enough -- a stats cadence fires thousands of times and must not allocate on
// each. Returns hipSuccess when the buffer is usable.
inline hipError_t ensurePlaneScratch(DeviceBuffer<double>& scratch,
                                     std::size_t n)
{
    if (scratch.size() >= n) return hipSuccess;
    auto b = DeviceBuffer<double>::make(n);
    if (!b) return b.error();
    scratch = std::move(*b);
    return hipSuccess;
}

// Reads the launched sums back, sizing `host` to match. Call after the kernel
// that filled `scratch`; the copy synchronises.
inline hipError_t fetchPlaneScratch(const DeviceBuffer<double>& scratch,
                                    std::size_t n, std::vector<double>& host)
{
    host.resize(n);
    return hipMemcpy(host.data(), scratch.data(), n * sizeof(double),
                     hipMemcpyDeviceToHost);
}

} // namespace dns::gpu::internal
