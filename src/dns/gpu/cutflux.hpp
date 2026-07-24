// Apertured cut-cell momentum RHS (--cut-flux): the sharp alternative to the
// staircase-masked momentum update. Convective and viscous fluxes are weighted
// by the exact face apertures the projection already uses, over lambda-weighted
// control volumes, so the wall enters momentum at its true sub-cell position
// -- this removes the stationary 2-Delta vorticity the binary mask sheds
// (measured: alternation ~25 at the sphere surface, ~2.2 half a diameter
// upstream, both SGS models).
//
// Dispatch: the plain fusedRhsStage kernel runs first over the whole grid,
// then a band-only kernel overwrites cells where `flag` (dns::buildCutFlag)
// is set with the apertured RHS. Interior cells therefore carry the plain
// kernel's exact bits BY CONSTRUCTION (same kernel binary; two TUs do NOT
// contract identical expressions identically -- measured).
// GPU-only header.
#pragma once

#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/index.hpp"

namespace dns::gpu {

// The dispatch mask on device; built once from dns::buildCutFlag.
struct DeviceCutFlag
{
    DeviceBuffer<unsigned char> flag;

    static std::expected<DeviceCutFlag, hipError_t>
    make(const std::vector<unsigned char>& hostFlag);
};

// out = a*x + b*(in + dt*rhs) with rhs = apertured convection(in)
// + nu*apertured lap(diff) + A*(in - mean). diff is the ghost-filled view;
// fluid->ghost links keep FULL conductance (the mirror places the sub-cell
// wall; kBodyFriction depends on that contract), fluid->fluid links scale by
// the averaged aperture. Everything divides by lambda_tilde = max(lambda_face,
// 1/2) -- the small-cell floor on the time term only; face fluxes stay exact.
void fusedRhsStageCut(const GpuGrid& g, Real nu, Real A, Real mx, Real my,
                      Real mz, Real a, const DeviceField& x, Real b,
                      const DeviceField& in, const DeviceField& diff,
                      const DeviceCutGeometry& cut, const DeviceCutFlag& flag,
                      Real dt, DeviceField& out);

} // namespace dns::gpu
