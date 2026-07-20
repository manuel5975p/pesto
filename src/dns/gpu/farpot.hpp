// Freestream-relative potential far-field disturbance: wake source (strength
// Q = F_drag / (rho uInf)) + displacement doublet (dbl = uInf a^2 in 2D,
// uInf a^3 / 2 in 3D), evaluated at (x,y,z) relative to the body centre.
// Shared by the compressible NSCBC targets and the incompressible tunnel
// inlet. Q = 0 and dbl = 0 return exact zeros. rmin2 clamps r^2 so a
// boundary evaluation can never sit on the singularity.
// GPU-only header.
#pragma once

#include "dns/gpu/index.hpp"

#include <cmath>

namespace dns::gpu {

__host__ __device__ inline void farPotential(bool twoD, Real x, Real y, Real z,
                                             Real Q, Real dbl, Real rmin2,
                                             Real& du, Real& dv, Real& dw)
{
    du = dv = dw = 0;
    if (twoD) {
        const Real rq = x * x + y * y;
        const Real r2 = rq < rmin2 ? rmin2 : rq;
        const Real src = Q / (Real(2 * M_PI) * r2);
        const Real r4 = r2 * r2;
        du = src * x - dbl * (x * x - y * y) / r4;
        dv = src * y - dbl * Real(2) * x * y / r4;
    } else {
        const Real rq = x * x + y * y + z * z;
        const Real r2 = rq < rmin2 ? rmin2 : rq;
        const Real r1 = std::sqrt(r2), r3 = r2 * r1, r5 = r3 * r2;
        const Real src = Q / (Real(4 * M_PI) * r3);
        du = src * x + dbl * (Real(1) / r3 - Real(3) * x * x / r5);
        dv = src * y - dbl * Real(3) * x * y / r5;
        dw = src * z - dbl * Real(3) * x * z / r5;
    }
}

} // namespace dns::gpu
