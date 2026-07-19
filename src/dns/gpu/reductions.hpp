// FP64 device reductions mirroring the dns/operators.hpp statistics.
// GPU-only header; hipCUB stays inside the .hip TU. All calls synchronize.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"

#include <expected>

namespace dns::gpu {

// Reusable temp storage; grows on demand, allocation-free in steady state.
struct ReduceScratch
{
    DeviceBuffer<unsigned char> tmp;
    DeviceBuffer<double> out; // single accumulator

    static std::expected<ReduceScratch, hipError_t> make();
};

struct Mean3 { double x, y, z; };

// Same semantics as the CPU oracle (dns/operators.hpp), accumulated in double.
double kineticEnergy(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
Mean3 meanVelocity(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
double fluctuationEnergy(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
double maxCourantPerDt(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
double maxAbsDivergence(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
double velocityGradientSq(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);

} // namespace dns::gpu
