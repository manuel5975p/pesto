// GPU runtime helpers: device detection, launch error check, host<->device
// field transfer with double<->float conversion. GPU-only header.
#pragma once

#include "dns/field.hpp"
#include "dns/gpu/device_buffer.hpp"

#include <hip/hip_runtime.h>

#include <cassert>
#include <cstdio>

namespace dns::gpu {

// True iff at least one GPU device is usable (tests skip otherwise).
bool deviceAvailable();

// Which toolchain compiled this backend. The sources are identical either way
// (src/compat/cuda/README.md); the label exists so a stored provenance line --
// e.g. the `backend=` field in the TGV CSV headers under data/validation --
// records which one actually produced the numbers.
#if DNS_WITH_CUDA
inline constexpr const char* backendName = "cuda";
#else
inline constexpr const char* backendName = "hip";
#endif

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
