// GPU runtime helpers: device detection, launch error check, host<->device
// field transfer with double<->float conversion. GPU-only header.
#pragma once

#include "dns/field.hpp"
#include "dns/gpu/device_buffer.hpp"

#include "dns/gpu/hip_runtime.hpp"

#include <cassert>
#include <cstdio>
#include <expected>
#include <vector>

namespace dns::gpu {

// True iff at least one GPU device is usable (tests skip otherwise).
bool deviceAvailable();

// Which toolchain compiled this backend. The sources are identical either way
// (src/compat/cuda/README.md); the label exists so a stored provenance line --
// e.g. the `backend=` field in the TGV CSV headers under data/validation --
// records which one actually produced the numbers. The build system's backend
// branch defines it, so no source needs to ask which backend it is.
inline constexpr const char* backendName = DNS_GPU_BACKEND_NAME;

// Host FP64 <-> device Real, rounding once per element: the one narrowing
// upload and widening download every device state goes through.
// hipErrorInvalidValue (nothing copied) when the two sizes differ.
[[nodiscard]] hipError_t uploadNarrow(const std::vector<double>& src,
                                      DeviceScalar& dst);
[[nodiscard]] hipError_t downloadWiden(const DeviceScalar& src,
                                       std::vector<double>& dst);
// A fresh device buffer holding Real(src).
std::expected<DeviceScalar, hipError_t> makeNarrow(const std::vector<double>& src);

// Narrowing upload / widening download of a whole staggered field.
[[nodiscard]] hipError_t uploadField(const VectorField& src, DeviceField& dst);
[[nodiscard]] hipError_t downloadField(const DeviceField& src, VectorField& dst);

// dst = src on the device, all three components in ONE async launch -- three
// D2D hipMemcpy calls were three blocking host synchronizations. Returns the
// launch status; hipErrorInvalidValue (nothing launched) on a size mismatch.
hipError_t copyField(DeviceField& dst, const DeviceField& src);

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
