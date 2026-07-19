// Device Chorin projection; mirrors dns/projection.hpp. GPU-only header.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/poisson.hpp"

#include <expected>

namespace dns::gpu {

struct DeviceProjectionScratch
{
    DeviceScalar div, phi;

    static std::expected<DeviceProjectionScratch, hipError_t> make(std::size_t n)
    {
        auto d = DeviceScalar::make(n);
        if (!d) return std::unexpected(d.error());
        auto p = DeviceScalar::make(n);
        if (!p) return std::unexpected(p.error());
        return DeviceProjectionScratch{std::move(*d), std::move(*p)};
    }
};

// Projects U in place: solve lap(phi) = div(U)/dt, U -= dt grad(phi). Pre: dt > 0.
void project(const GpuGrid& g, DevicePoissonSolver& poisson, Real dt, DeviceField& U,
             DeviceProjectionScratch& s);

} // namespace dns::gpu
