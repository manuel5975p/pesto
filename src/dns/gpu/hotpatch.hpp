// The centred hot floor patch of `dns conv` (--hot-patch F), and the discrete
// area fraction its Nusselt normalisation needs.
//
// ONE definition, shared by BOTH branches. The compressible cell
// (gpu/comp_conv) and the Boussinesq one (gpu/boussinesq) are two models, but
// the patch is not physics -- it is which floor columns are held at Tbot, and
// the two branches' cross-validation is only meaningful if they heat exactly
// the same ones. They used to carry a copy each.
//
// GPU-only header.
#pragma once

#include "dns/gpu/index.hpp"

#include <cstddef>

namespace dns::gpu {

// The floor column (i,k) is heated iff it lies within the patch in BOTH
// lateral directions. Written as a fraction of the FLUID index range so it is
// independent of the box extents and of whether that axis carries a ghost
// column (ox/oz from sideGhostX/sideGhostZ), and vacuous at nz == 1 -- one z
// cell is always inside, which makes the 2D case the middle strip in x.
// Pre: 0 < patch <= 1; patch == 1 covers the whole floor.
__host__ __device__ inline bool inHotPatch(const GpuGrid& g, int i, int k,
                                           Real patch, int ox, int oz)
{
    const Real fx = (Real(i - ox) + Real(0.5)) / Real(g.nx - 2 * ox)
                  - Real(0.5);
    if (fabsf(fx) > Real(0.5) * patch) return false;
    if (g.nz == 1) return true;
    const Real fz = (Real(k - oz) + Real(0.5)) / Real(g.nz - 2 * oz)
                  - Real(0.5);
    return fabsf(fz) <= Real(0.5) * patch;
}

// Area fraction of the floor the patch actually covers, counted in CELLS --
// the discrete value, which is what the Nusselt normalisation must use rather
// than the nominal patch^2. Host-side; runs the kernel's own predicate, so the
// two cannot disagree about where the patch edge falls.
inline double hotPatchAreaFraction(const GpuGrid& g, double patch, bool closed)
{
    const int ox = sideGhostX(closed), oz = sideGhostZ(g, closed);
    std::size_t inside = 0;
    for (int k = oz; k < g.nz - oz; ++k)
        for (int i = ox; i < g.nx - ox; ++i)
            if (inHotPatch(g, i, k, Real(patch), ox, oz)) ++inside;
    return double(inside) / (double(g.nx - 2 * ox) * (g.nz - 2 * oz));
}

} // namespace dns::gpu
