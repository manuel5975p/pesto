// FP64 integrals and extremes over the REAL fluid cells of a mesh (a 0/1
// mask): the stop series and stats columns of the RANS / steady runners.
// Deterministic -- every one goes through internal::reduce (rocPRIM, fixed
// order) -- as the ground rules require of anything reported.
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/internal/reduce.hpp"

#include <cmath>
#include <cstdint>

namespace dns::gpu::internal {

// sum_c mask_c vol_c (f_c - off). Pre: mask, vol, f sized n.
struct MaskedVolSum
{
    const std::uint8_t* mask;
    const double* vol;
    const Real* f;
    double off;
    __host__ __device__ double operator()(int c) const
    {
        return mask[c] ? vol[c] * (double(f[c]) - off) : 0.0;
    }
};

// max_c mask_c scale f_c; a non-finite counted cell is +inf (never dropped
// by the max). Unmasked cells read -inf.
struct MaskedMax
{
    const std::uint8_t* mask;
    const Real* f;
    double scale;
    __host__ __device__ double operator()(int c) const
    {
        if (!mask[c]) return -HUGE_VAL;
        const double v = scale * double(f[c]);
        return isfinite(v) ? v : HUGE_VAL;
    }
};

} // namespace dns::gpu::internal
