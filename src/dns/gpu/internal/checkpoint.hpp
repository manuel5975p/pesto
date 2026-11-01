// The runner side of dns/checkpoint.hpp: a runner's loop-state struct and its
// device state to and from a Checkpoint. Header-only (templates over the
// loop struct; the file format lives in dns/checkpoint.cpp).
//
// A LOOP STRUCT is the one place a runner keeps every host variable its time
// loop carries from one step to the next; it exposes them, by name, through
//   template <class F> void fields(F&& f) { f("t", t); f("steps", steps); ... }
// (members double, long, int or bool). saveLoop/loadLoop walk that list, so
// a variable added to the loop OUTSIDE the struct is not checkpointed and
// breaks the restart byte rule silently (masterplan real_bunsen K12; the
// restart tests are the proof).
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/checkpoint.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/run_result.hpp"

#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <type_traits>
#include <vector>

namespace dns::gpu::internal {

template <class Loop>
void saveLoop(Checkpoint& ck, Loop& L)
{
    L.fields([&](const char* n, auto& v) {
        using T = std::remove_cvref_t<decltype(v)>;
        if constexpr (std::is_floating_point_v<T>)
            ck.putF64(std::string("loop.") + n, double(v));
        else
            ck.putI64(std::string("loop.") + n, std::int64_t(v));
    });
}

// Every member back from the "loop.*" records; a missing one is an error by
// name (a checkpoint of another loop layout).
template <class Loop>
std::expected<void, std::string> loadLoop(const Checkpoint& ck, Loop& L)
{
    std::string err;
    L.fields([&](const char* n, auto& v) {
        if (!err.empty()) return;
        using T = std::remove_cvref_t<decltype(v)>;
        const std::string key = std::string("loop.") + n;
        if constexpr (std::is_floating_point_v<T>) {
            auto x = ck.f64Scalar(key);
            if (!x) err = x.error();
            else v = T(*x);
        } else {
            auto x = ck.i64Scalar(key);
            if (!x) err = x.error();
            else v = T(*x);
        }
    });
    if (!err.empty()) return std::unexpected(err);
    return {};
}

// The conserved compressible state, raw FP32 bit patterns: "state.rho",
// "state.mx", "state.my", "state.mz", "state.E", the turbulence scalars
// "state.t0/1", the species "state.y0..", the low words of a compensated
// state "state.lo0..4", "state.tlo0/1". Download failures are exit 6.
inline std::expected<void, RunError> saveCompState(Checkpoint& ck,
                                                   const DeviceCompState& s)
{
    std::vector<float> h;
    auto put = [&](const std::string& name, const DeviceScalar& b) -> bool {
        h.resize(b.size());
        if (b.download(h.data()) != hipSuccess) return false;
        ck.putF32(name, h);
        return true;
    };
    bool ok = put("state.rho", s.rho) && put("state.mx", s.mx) &&
              put("state.my", s.my) && put("state.mz", s.mz) &&
              put("state.E", s.E);
    for (int i = 0; ok && i < s.nT; ++i) ok = put(std::format("state.t{}", i), s.t[i]);
    for (int k = 0; ok && k < s.nY; ++k) ok = put(std::format("state.y{}", k), s.y[k]);
    if (s.compensated()) {
        for (int i = 0; ok && i < 5; ++i) ok = put(std::format("state.lo{}", i), s.lo[i]);
        for (int i = 0; ok && i < s.nT; ++i)
            ok = put(std::format("state.tlo{}", i), s.tlo[i]);
    }
    if (!ok) return std::unexpected(RunError{6, "checkpoint: state download failed"});
    return {};
}

// saveCompState's inverse into an allocated state of the same layout. A
// missing or mis-sized record is exit 3 (the input), an upload failure 6.
inline std::expected<void, RunError> loadCompState(const Checkpoint& ck,
                                                   DeviceCompState& s)
{
    std::vector<float> h;
    std::expected<void, RunError> res;
    auto get = [&](const std::string& name, DeviceScalar& b) -> bool {
        h.resize(b.size());
        if (auto e = ck.f32Into(name, h.data(), h.size()); !e) {
            res = std::unexpected(RunError{3, "--restart: " + e.error()});
            return false;
        }
        if (b.upload(h.data()) != hipSuccess) {
            res = std::unexpected(RunError{6, "--restart: state upload failed"});
            return false;
        }
        return true;
    };
    bool ok = get("state.rho", s.rho) && get("state.mx", s.mx) &&
              get("state.my", s.my) && get("state.mz", s.mz) &&
              get("state.E", s.E);
    for (int i = 0; ok && i < s.nT; ++i) ok = get(std::format("state.t{}", i), s.t[i]);
    for (int k = 0; ok && k < s.nY; ++k) ok = get(std::format("state.y{}", k), s.y[k]);
    if (ok && s.compensated()) {
        for (int i = 0; ok && i < 5; ++i) ok = get(std::format("state.lo{}", i), s.lo[i]);
        for (int i = 0; ok && i < s.nT; ++i)
            ok = get(std::format("state.tlo{}", i), s.tlo[i]);
    }
    return res;
}

} // namespace dns::gpu::internal
