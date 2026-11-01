// The ONE species face formula (masterplan real_bunsen D7): the MC slope, the
// shared limiter of the two-reactant kernels and its linear face value. Moved
// verbatim from gpu/species.hip so the compressible species kernels and the
// low-Mach solver (gpu/lowmach.hip) read the same limiter -- two copies would
// be a restated rule.
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/gpu/index.hpp"

#include <cstddef>

namespace dns::gpu::internal {

// One interior face L | R along an axis (R = L + one step) and the stencil
// cells LL = L - 1, RR = R + 1. firstL/firstR: that upwind side has no second
// neighbour (an open face's boundary cell: its LL/RR would wrap across the
// domain), so the face goes first order (premixed) or central (shared theta)
// when its upwind cell is that side.
struct Face
{
    std::size_t L, R, LL, RR;
    bool firstL, firstR;
};

// Monotonized-central slope: 0 unless d1, d2 share a sign, else
// sign * min(2|d1|, 2|d2|, |d1 + d2|/2). Exactly 0 on a uniform field.
__device__ inline Real mcSlope(Real d1, Real d2)
{
    if (!(d1 * d2 > Real(0))) return Real(0);
    const Real m = fminf(fminf(Real(2) * fabsf(d1), Real(2) * fabsf(d2)),
                         Real(0.5) * fabsf(d1 + d2));
    return d1 > Real(0) ? m : -m;
}

// The shared limiter of face f: theta = min over the VOTING species of
// mcSlope(d1, d2) / d2, in [0, 2] (2 = none votes). The face value Y_up +
// theta/2 d2 is LINEAR in Y across species: an affine combination of the
// species (the coupling functions, the mixture fraction) is transported
// exactly by the same formula, whatever theta is. A voter keeps its own TVD
// bound (theta <= its own MC ratio; Sweby's region is star-shaped toward 0).
// A species votes iff |d2| >= kVoteJump = 2^-24 (the FP32 resolution of a
// unit mass fraction): a reactant consumed to ~1e-40 on the far side of the
// sheet carries sign-flipping rounding noise whose "extrema" (theta = 0)
// forced first-order upwind on EVERY species across the whole far side --
// measured on the advected layer (gpu_diffusion_mix) Z 5.0e-3 below the
// analytic erf at the slab centre with every species voting, max error 1.5e-3
// with the threshold. A non-voter moves at most |d2| < 2^-24 past its own
// neighbours per face, and votes again once it varies.
inline constexpr float kVoteJump = 0x1p-24f;

// Y[0..nY) the species' cell values; slot `skip` never votes (the soot slot:
// it is in no coupling function and keeps its own MC limiter; -1 = every
// species votes). Upwind cell = an open-face boundary cell (no second
// neighbour): the CENTRAL face value (theta 1, between the two cells, so
// bounded) -- the premixed kernel's first-order upwind there is a
// convective-flux mismatch mdot (Y_R - Y_L)/2 against the central energy flux.
__device__ inline Real sharedThetaOf(const Real* const* Y, int nY, int skip,
                                     const Face& f, Real mdot)
{
    if (mdot >= Real(0) ? f.firstL : f.firstR) return Real(1);
    Real th = Real(2);
    for (int s = 0; s < nY; ++s) {
        if (s == skip) continue;
        const Real* y = Y[s];
        Real d1, d2;
        if (mdot >= Real(0)) {
            d1 = f.firstL ? Real(0) : y[f.L] - y[f.LL];
            d2 = y[f.R] - y[f.L];
        } else {
            d1 = f.firstR ? Real(0) : y[f.R] - y[f.RR];
            d2 = y[f.L] - y[f.R];
        }
        if (fabsf(d2) >= kVoteJump) th = fminf(th, mcSlope(d1, d2) / d2);
    }
    return th;
}

// The shared-limiter face value Y_f = Y_up + theta/2 (Y_down - Y_up) (the
// low-Mach temperature predictor's T_f too).
__device__ inline Real faceValue2(const Real* Y, const Face& f, Real mdot,
                                  Real th)
{
    return mdot >= Real(0) ? Y[f.L] + Real(0.5) * th * (Y[f.R] - Y[f.L])
                           : Y[f.R] + Real(0.5) * th * (Y[f.L] - Y[f.R]);
}

// The face flux mdot Y_f - rho D (Y_R - Y_L)/h with faceValue2's Y_f.
__device__ inline Real faceY2(const Real* Y, const Face& f, Real mdot,
                              Real th, Real rhoD, Real h)
{
    const Real yf = faceValue2(Y, f, mdot, th);
    return mdot * yf - rhoD * (Y[f.R] - Y[f.L]) / h;
}

} // namespace dns::gpu::internal
