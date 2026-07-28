// Initial conditions for the three body/wall flows, with their measured
// constants at their definitions.
//
// These lived in apps/dns_main.cpp. They are physics -- taper widths, trip
// amplitudes and placements, a streamfunction chosen so the trip respects
// no-slip -- and CLAUDE.md keeps physics out of the app layer. Siblings:
// dns/hit_ic.hpp (spectral HIT), dns/taylor_green.hpp (analytic TGV).
//
// Every one of these is DETERMINISTIC: fixed modes, fixed phases, no RNG. Two
// runs of the same case start from bit-identical fields, which is what makes
// the bit-identity pins in the test suite meaningful.
#pragma once

#include "dns/casegeom.hpp"
#include "dns/comp.hpp"
#include "dns/config.hpp"
#include "dns/field.hpp"
#include "dns/remap.hpp"

#include <vector>

namespace dns {

// Incompressible body flow: uniform stream (or rest under --ramp) plus a
// deterministic z-uniform v-kick in the near wake.
//
// The kick is 5% of uInf over the box one to one-and-a-half diameters behind
// the body and within half a diameter of the axis. It exists to break the
// symmetric-wake transient early; it is NOT a turbulence trip and its
// amplitude is not tuned to anything. Under grid sequencing the coarse
// level's converged field replaces the whole IC.
//
// Quiet start (--ramp > 0) begins at rest, which is exactly consistent with
// the no-slip body -- the impulsive uniform stream is what rings up the
// trapped stagnation-region waves (see CLAUDE.md's ripple entry). The kick
// stays either way; it only acts once flow arrives.
void incompressibleBodyIC(const Case& c, const CaseGeometry& geo,
                          VectorField& U);

// Compressible body flow. Fills `state` and marks solid cells 0 in `mask`.
//
// The freestream is tapered to rest across ~3 cells outside the surface with a
// smoothstep, so an impulsive start does not slam a discontinuity onto the
// ghost wall. Under --ramp the field starts at rest instead and the runner
// raises the freestream by silent Galilean boosts.
//
// The trip is a weak Gaussian VORTEX one diameter behind the body, amplitude
// --kick * uInf, width D/4, written directly as a solenoidal velocity field:
// divergence-free, hence acoustically silent. It is applied on the quiet start
// only -- that start is too symmetric to shed on its own.
//
// `remap` and `yRemap` are the active mesh maps (both the exact identity when
// off): the taper and the trip must sit where the cells PHYSICALLY are, not
// where their indices are.
void compressibleBodyIC(const Case& c, const CaseGeometry& geo,
                        const AxisRemap3& remap, const TanhStretchY& yRemap,
                        double rhoInf, double pInf, double gamma,
                        CompState& state, std::vector<float>& mask);

// Channel: laminar Poiseuille scaled to the log-law bulk target (so the flux
// does not have to be shed during transition) plus a deterministic
// divergence-free trip u' = curl(psi).
//
// The envelope gEnv = (y(2d-y)/d^2)^2 has gEnv = gEnv' = 0 at BOTH walls, so
// the trip respects no-slip exactly; being a curl it is solenoidal, hence
// acoustically silent. nz == 1 keeps psi_x = 0 so w stays exactly 0 -- the
// `2D is nz == 1, not a code path` invariant.
//
// The two ghost rows are marked 0 in `mask`: the wall mirror owns them.
void channelIC(const Case& c, const ChannelStretchY& remap, double delta,
               double rho0, double gamma, CompState& state,
               std::vector<float>& mask);

// Rayleigh-Benard: the HYDROSTATIC conduction state plus the same
// deterministic curl(psi) trip the channel uses.
//
// Starting from rest and uniform pressure would not be a state of this problem
// at all -- gravity would collapse the layer and ring the box acoustically for
// the whole run. So the IC integrates dp/dy = -rho g through the linear
// conduction profile T(y), which for a linear T is exactly
// p(y) = pRef (T(y)/Tref)^(-g/(R a)) with a = dT/dy, and takes rho from the
// equation of state. That state is a genuine (unstable) equilibrium: it stays
// put to machine precision until the trip grows.
//
// The trip is solenoidal, hence acoustically silent, and its envelope vanishes
// with zero derivative at both walls so it respects no-slip exactly.
//
// The two ghost rows are marked 0 in `mask`: the wall mirror owns them.
void convIC(const Case& c, double gamma, CompState& state,
            std::vector<float>& mask);

// Boussinesq Rayleigh-Benard (--physics incompressible): the linear conduction
// profile and the same solenoidal trip, on the MAC grid.
//
// No hydrostatic pressure is needed or wanted here -- the projection owns the
// pressure, and in the Boussinesq equations the hydrostatic part is absorbed
// into it by construction. That is the whole difference from convIC, which has
// to integrate dp/dy = -rho g explicitly because its density is a state
// variable.
//
// The trip is the analytic curl of a streamfunction sampled at the MAC face
// positions, so it is solenoidal to truncation rather than exactly; the first
// projection removes the remainder, exactly as it does for channelIC.
void rbIC(const Case& c, VectorField& U, Scalar& T);

} // namespace dns
