// GPU runtime helpers: device detection, launch error check, host<->device
// field transfer with double<->float conversion. GPU-only header.
#pragma once

#include "dns/field.hpp"
#include "dns/gpu/device_buffer.hpp"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdio>

namespace dns::gpu {

// True iff at least one HIP device is usable (tests skip otherwise).
bool deviceAvailable();

// Narrowing upload / widening download of a whole staggered field.
hipError_t uploadField(const VectorField& src, DeviceField& dst);
hipError_t downloadField(const DeviceField& src, VectorField& dst);

constexpr unsigned blockSize = 256; // 8 x wave32 on RDNA3

inline unsigned launchBlocks(std::size_t n)
{
    return unsigned((n + blockSize - 1) / blockSize);
}

// Internal invariant: kernel launches must not fail (bad config = bug).
inline void checkLastKernel(const char* what)
{
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess)
    {
        std::fprintf(stderr, "HIP kernel failure in %s: %s\n", what,
                     hipGetErrorString(e));
        assert(false && "HIP kernel launch failed");
    }
}

} // namespace dns::gpu
