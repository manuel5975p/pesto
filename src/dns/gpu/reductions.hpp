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

// Deterministic FP64 sum of a device double array. Force/flux tallies stage
// their per-item contributions here instead of atomicAdd-ing in FP32: with
// cancelling +- contributions the FP32 atomic both loses precision and makes
// the reported coefficient irreproducible run to run (atomic ordering).
double sumDouble(const double* q, std::size_t n, ReduceScratch& s);

// Same semantics as the CPU oracle (dns/operators.hpp), accumulated in double.
double kineticEnergy(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
Mean3 meanVelocity(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
double fluctuationEnergy(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
double maxCourantPerDt(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
double maxAbsDivergence(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);
double velocityGradientSq(const GpuGrid& g, const DeviceField& U, ReduceScratch& s);

// Plain sum of a device scalar field in double (e.g. the fringe momentum
// tally). Moved here from gpu/cutcell.hpp: it is a reduction, not cut-cell
// geometry, and both the tunnel and the diagnostics paths need it.
double sumScalar(const DeviceScalar& q, ReduceScratch& s);

// max |q| in double (e.g. the continuity-residual diagnostic).
double maxAbsScalar(const DeviceScalar& q, ReduceScratch& s);

} // namespace dns::gpu
