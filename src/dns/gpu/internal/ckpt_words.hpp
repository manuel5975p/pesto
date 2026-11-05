// Raw device words to and from a Checkpoint record (dns/checkpoint.hpp): a
// float buffer is an F32 record, a double buffer an F64 one, the bit pattern
// either way -- the FV solvers' state at either --fv-precision round-trips
// exactly. A missing or mis-sized record is exit 3 (the input), a transfer
// failure exit 6. GPU-internal header: HIP TUs only.
#pragma once

#include "dns/checkpoint.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/run_result.hpp"

#include <expected>
#include <format>
#include <string>
#include <type_traits>
#include <vector>

namespace dns::gpu::internal {

// The whole buffer as record `name`. Pre: no kernel still writes b.
template <class T>
std::expected<void, RunError> putWords(Checkpoint& ck, const std::string& name, const DeviceBuffer<T>& b)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    (void)hipDeviceSynchronize();
    std::vector<T> h(b.size());
    if (b.download(h.data()) != hipSuccess)
        return std::unexpected(RunError{6, std::format("checkpoint: '{}' download failed", name)});
    if constexpr (std::is_same_v<T, float>) ck.putF32(name, h);
    else ck.putF64(name, h);
    return {};
}

// Record `name` into the whole buffer: same type, exactly b.size() words.
template <class T>
std::expected<void, RunError> getWords(const Checkpoint& ck, const std::string& name, DeviceBuffer<T>& b)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    std::vector<T> h(b.size());
    if constexpr (std::is_same_v<T, float>) {
        if (auto e = ck.f32Into(name, h.data(), h.size()); !e) return std::unexpected(RunError{3, "--restart: " + e.error()});
    } else {
        auto v = ck.f64(name);
        if (!v) return std::unexpected(RunError{3, "--restart: " + v.error()});
        if (v->size() != h.size())
            return std::unexpected(RunError{3, std::format("--restart: checkpoint '{}' holds {} values, the run needs {}",
                                                           name, v->size(), h.size())});
        h = std::move(*v);
    }
    if (b.upload(h.data()) != hipSuccess)
        return std::unexpected(RunError{6, std::format("--restart: '{}' upload failed", name)});
    return {};
}

} // namespace dns::gpu::internal
