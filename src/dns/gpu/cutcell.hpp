// Device cut-cell embedded boundary: the GEOMETRY axis -- apertured
// divergence/gradient/masking over an uploaded CutGeometry. Mirrors
// dns/cutcell.hpp with all fields resident on the GPU; geometry is built on the
// CPU (exact, double) and uploaded once. The CG pressure solve that consumes
// this geometry is the separate solver axis, in gpu/cutpoisson.hpp.
// GPU-only header.
#pragma once

#include "dns/cutcell.hpp"
#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/poisson.hpp"
#include "dns/gpu/reductions.hpp"

#include <expected>

namespace dns::gpu {

// Cell fluid fractions + face apertures on device (Real) with the fluid-cell
// count; values/indexing identical to the host CutGeometry it mirrors.
struct DeviceCutGeometry
{
    DeviceScalar lambda, ax, ay, az;
    std::size_t nFluid;

    static std::expected<DeviceCutGeometry, hipError_t> make(const CutGeometry& cut);
};

// out = scale * apertured div(U) at cell centres (scale folds the projection 1/dt).
void divergenceCut(const GpuGrid& g, const DeviceCutGeometry& cut,
                   const DeviceField& U, DeviceScalar& out, Real scale);

// U -= scale * grad(p) on open faces only (aperture > openTol).
void subtractGradientCut(const GpuGrid& g, const DeviceCutGeometry& cut,
                         const DeviceScalar& p, Real scale, DeviceField& U);

// Zero every face whose aperture is closed: enforces u.n = 0 on the solid.
void maskSolidFaces(const GpuGrid& g, const DeviceCutGeometry& cut, DeviceField& U);

// Brinkman-style solid-band penalization (--penalize eta): each velocity face
// decays by exp(-dt*(1-lambda_face)/eta), lambda_face the face-averaged fluid
// fraction. The decay strength is SMOOTH ALONG THE SURFACE (lambda is the
// exact volume fraction), unlike the open/closed bit of the mask -- that
// smoothness is the point: it damps the jagged staircase slip that feeds the
// stationary 2-Delta upstream vorticity. Exact split decay: unconditionally
// stable, no dt limit. Strictly dissipative and local; the drained momentum
// is NOT yet folded into the cd budget (experimental flag; tally lands with
// the default flip, cd reads slightly low meanwhile).
void penalizeSolidBand(const GpuGrid& g, const DeviceCutGeometry& cut, Real eta,
                       Real dt, DeviceField& U);

// max |apertured div| over fluid cells, accumulated in double.
double maxAbsDivergenceCut(const GpuGrid& g, const DeviceCutGeometry& cut,
                           const DeviceField& U, ReduceScratch& s);

} // namespace dns::gpu
