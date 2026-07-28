// Device-side pieces every incompressible stencil kernel shares: the aperture
// open/closed tolerance, the once-per-thread reciprocal spacings, and the
// device half of kernels.hpp's StageTail (so the plain fused stage in
// gpu/kernels.hip and the apertured overwrite in gpu/cutflux.hip write the same
// tail from the same source).
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/cutcell.hpp"        // CutGeometry::openTol
#include "dns/gpu/index.hpp"
#include "dns/gpu/kernels.hpp"    // StageTail

#include <hip/hip_runtime.h>

#include <cstddef>

namespace dns::gpu::internal {

inline constexpr Real openTol = Real(CutGeometry::openTol);

__device__ inline Real avg(Real a, Real b) { return Real(0.5) * (a + b); }

// Reciprocal spacings, formed ONCE per thread. Every stencil below multiplies
// by these; dividing in place cost ~27 FP32 divisions per fused-stage thread,
// and a division is not one instruction on either backend. The squared entries
// keep the Laplacian's own 1/(h*h) form rather than (1/h)^2.
struct Inv
{
    Real x, y, z, xx, yy, zz;
};

__device__ inline Inv invOf(const GpuGrid& g)
{
    return {.x = Real(1) / g.hx,
            .y = Real(1) / g.hy,
            .z = Real(1) / g.hz,
            .xx = Real(1) / (g.hx * g.hx),
            .yy = Real(1) / (g.hy * g.hy),
            .zz = Real(1) / (g.hz * g.hz)};
}

// Brinkman band decay then the solid-face mask, in place on one cell's three
// freshly computed outputs. Both parts are no-ops when their pointers are null,
// and the branches are uniform across the whole launch.
// Pre: `o` are the wrap-corrected offsets at c.
__device__ inline void applyStageTail(const StageTail& t, const Offsets& o,
                                      std::size_t c, Real& qu, Real& qv, Real& qw)
{
    if (t.lam)
    {
        const Real lu = avg(t.lam[c + o.m[0]], t.lam[c]);
        const Real lv = avg(t.lam[c + o.m[1]], t.lam[c]);
        const Real lw = avg(t.lam[c + o.m[2]], t.lam[c]);
        if (lu < Real(1)) qu *= exp(-t.idte * (Real(1) - lu));
        if (lv < Real(1)) qv *= exp(-t.idte * (Real(1) - lv));
        if (lw < Real(1)) qw *= exp(-t.idte * (Real(1) - lw));
    }
    if (t.ax)
    {
        if (t.ax[c] <= openTol) qu = 0;
        if (t.ay[c] <= openTol) qv = 0;
        if (t.az[c] <= openTol) qw = 0;
    }
}

} // namespace dns::gpu::internal
