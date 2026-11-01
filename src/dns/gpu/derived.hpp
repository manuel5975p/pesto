// Derived visualisation fields computed ON DEVICE: the quantities every app
// used to build by pulling whole slabs (or, for the 3D mesh runs, ~1 GB of
// primitives per frame) across PCIe and differencing them on the host.
//
// The stencils are the ones the host loops used, evaluated in the same
// precision (differences in Real, accumulation of the magnitude in Real), so
// the frames are the same field -- this is a data-movement change, not a
// numerical one.
//
// GPU-only header.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/index.hpp"

namespace dns::gpu {

// Signed z-vorticity at cell corners from a MAC velocity field:
//   wz = (v(i,j) - v(i-1,j)) / hx - (u(i,j) - u(i,j-1)) / hy
// the exact stencil the 2D apps used. Defined for any nz; at nz == 1 it is the
// only nonzero vorticity component.
void vorticityZ(const GpuGrid&, const DeviceField& U, DeviceScalar& out);

// |u| at cell centres from a MAC field (face pairs averaged to the centre).
void speedMac(const GpuGrid&, const DeviceField& U, DeviceScalar& out);

// The three CELL-CENTRED velocity components of a MAC field, each the average
// of the cell's two opposing faces -- the same averaging speedMac does
// internally, exposed so a caller can feed the centred curl/speed helpers
// below (which the compressible runners already use) from a staggered field.
void centreVelocity(const GpuGrid&, const DeviceField& U, DeviceScalar& u,
                    DeviceScalar& v, DeviceScalar& w);

// |curl u| and |u| at cell centres from CELL-CENTRED velocity components
// (the compressible primitives rho-1 * m), 3D central differences with
// periodic wrap -- the comp_mesh host stencil. At nz == 1 the z differences
// cancel by address arithmetic, exactly as everywhere else in the tree.
// SIGNED z-vorticity from CELL-CENTRED components, central differences:
//   wz = (v(i+1) - v(i-1)) / 2hx - (u(j+1) - u(j-1)) / 2hy
// the 2D compressible app's stencil. Signed, because its frames use the
// diverging colour map -- |curl| would throw the sign of the shed vortices away.
void vorticityZCentred(const GpuGrid&, const DeviceScalar& u,
                       const DeviceScalar& v, DeviceScalar& out);

void curlMagCentred(const GpuGrid&, const DeviceScalar& u, const DeviceScalar& v,
                    const DeviceScalar& w, DeviceScalar& out);
void speedCentred(const GpuGrid&, const DeviceScalar& u, const DeviceScalar& v,
                  const DeviceScalar& w, DeviceScalar& out);

// The Takeno flame index xi = grad Y_F . grad Y_O / (|grad Y_F| |grad Y_O|)
// per cell (masterplan real_bunsen S5, D14): +1 where both reactants fall
// toward the same side (premixed burning), -1 across a diffusion sheet. xi
// := 0 where either |grad Y| < 2^-24 / h_min (the shared limiter's vote
// threshold: below it a gradient is FP32 noise). Central differences, y and
// z periodic, x ONE-SIDED at i = 0 and nx - 1 (x is an inflow/outflow axis,
// never wrapped); at nz == 1 the z differences cancel by address
// arithmetic. In Real; |xi| <= 1 up to rounding (a pure-x pair is exactly
// +-1). openY/openZ (the low-Mach open laterals, S6): that axis one-sided at
// its ends like x (false: the periodic stencil bit for bit). Pre: nx >= 2.
void flameIndexCentred(const GpuGrid&, const DeviceScalar& YF,
                       const DeviceScalar& YO, DeviceScalar& out,
                       bool openY = false, bool openZ = false);

// Cell-centred velocity from the conservative compressible state: q = m / rho,
// with rho <= 0 treated as 1 (the host loop's guard; a non-positive density is
// a numerical failure the caller reports separately).
void primitiveVelocity(const GpuGrid&, const DeviceScalar& rho,
                       const DeviceScalar& mx, const DeviceScalar& my,
                       const DeviceScalar& mz, DeviceScalar& u, DeviceScalar& v,
                       DeviceScalar& w);

} // namespace dns::gpu
