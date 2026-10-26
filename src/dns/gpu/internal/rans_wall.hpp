// The ghost-cell turbulence scalars of a no-slip wall (docs/rans.md): every
// wall row in the tree -- the channel plates, the O-grid body surface --
// writes these, so the image algebra cannot drift between mesh families.
// Moved verbatim from comp_channel.hip: the channel's bits are unchanged.
//
// GPU-internal header: HIP TUs only.
#pragma once

#include "dns/gpu/device_buffer.hpp"

namespace dns::gpu::internal {

// SA nu~ (and SST k) antimirrored on the SPECIFIC value, rebuilt with the
// ghost's OWN density (an isothermal ghost has rho_c != rho_s), so the wall
// face holds 0. Returns the conserved ghost value. Pre: rhoS > 0.
__device__ inline Real saGhost(Real rhoC, Real rhoS, Real t0S)
{
    return -rhoC * t0S / rhoS;
}

// SST omega: the DIRICHLET image of TMR's wall value, omega_c = 2 omega_w -
// omega_s on the specific value, omega_w = coefMu / rho_f with rho_f the
// wall-face density and coefMu = 60 mu_f / (beta1 d1^2) (mu_f the face
// viscosity; the channel passes the constant-mu coefficient). Returns the
// conserved ghost value. Pre: rhoC, rhoS > 0.
__device__ inline Real sstOmegaGhost(Real coefMu, Real rhoC, Real rhoS,
                                     Real t1S)
{
    const Real ww = coefMu / (Real(0.5) * (rhoC + rhoS));
    return rhoC * (Real(2) * ww - t1S / rhoS);
}

} // namespace dns::gpu::internal
