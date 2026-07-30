// Temperature-dependent transport properties of the working gas: dynamic
// viscosity mu(T) and Fourier conductivity kappa(T) = mu(T) cp / Pr. The gas is
// air -- diatomic and CALORICALLY PERFECT (gamma = 1.4 fixed, cv constant): the
// vibrational mode is frozen below ~600 K, so gamma(T) is deliberately not
// modelled and every energy conversion in the solver stays linear in T.
//
// Trivially copyable and DNS_HD, so one object serves the host, the CPU
// reference solver and HIP kernels taking it by value.
//
// UNITS. The solver works in box units (rho_inf = u_inf = 1, R = 1), where
// temperature carries the scale T_inf = p_inf / (rho_inf R) = 1 / (gamma M^2).
// Sutherland's law is not scale-free -- its 110.4 K constant is dimensional --
// so a temperature-dependent run needs one dimensional anchor, the freestream
// static temperature in kelvin. That anchor enters ONLY as the ratio
// S = (110.4 K / T_inf[K]) * T_inf[box], which is why a constant-viscosity run
// never needs it and every measured anchor in the repo is unaffected.
#pragma once

#include "dns/hd.hpp"

#include <math.h>

namespace dns {

struct Transport
{
    double muRef = 0.0;  // dynamic viscosity at Tref
    double Tref = 1.0;   // reference temperature (box units) == T_inf
    double S = 0.0;      // Sutherland constant, box units; 0 with the law off
    double Pr = 0.72;    // molecular Prandtl number (air)
    double gamma = 1.4;  // ratio of specific heats
    double Rgas = 1.0;   // specific gas constant, p = rho R T
    bool sutherland = false;

    // Sutherland's constant for air, in kelvin.
    static constexpr double sutherlandK = 110.4;

    // Air with the viscosity law selected. tinfK is the DIMENSIONAL freestream
    // static temperature that fixes the Sutherland scale; it is ignored (and
    // may be anything) when suth is false. Pre: muRef >= 0, Tref > 0, Pr > 0,
    // gamma > 1, Rgas > 0, and tinfK > 0 when suth.
    static Transport air(double muRef, double Tref, double tinfK, double Pr,
                         double gamma, double Rgas, bool suth)
    {
        return {.muRef = muRef,
                .Tref = Tref,
                .S = suth ? sutherlandK / tinfK * Tref : 0.0,
                .Pr = Pr,
                .gamma = gamma,
                .Rgas = Rgas,
                .sutherland = suth};
    }

    // The gas the solver had before transport was temperature-dependent:
    // constant mu, Pr = 0.72, R = 1. Bit-for-bit that solver, and the one
    // factory a caller with only (gamma, mu) needs -- it exists so no other
    // header has to offer a second (gamma, mu) constructor beside a Transport
    // one. Pre: gamma > 1, mu >= 0.
    static Transport constant(double gamma, double mu)
    {
        return air(mu, 1.0, 0.0, 0.72, gamma, 1.0, false);
    }

    DNS_HD double cp() const { return gamma * Rgas / (gamma - 1.0); }

    // mu(T) / muRef = (T/Tref)^(3/2) (Tref + S) / (T + S), and EXACTLY 1 with
    // the law off -- the constant-viscosity path stays bit-for-bit what it was.
    // Monotone increasing in T, which is what lets stableDt bound mu over the
    // field from a single max-T reduction. Pre: T > 0 (stableDt checks it).
    DNS_HD double ratio(double T) const
    {
        if (!sutherland) return 1.0;
        const double t = T / Tref, s = S / Tref;
        return t * sqrt(t) * (1.0 + s) / (t + s);
    }
    DNS_HD double mu(double T) const { return muRef * ratio(T); }
    DNS_HD double kappa(double T) const { return mu(T) * cp() / Pr; }

    // Largest diffusivity the explicit stability limit must respect: momentum
    // carries nu = mu/rho, but the energy equation carries kappa/(rho cv) =
    // gamma nu / Pr, which for air (Pr = 0.72) is 1.94x LARGER. Taking only the
    // momentum limit -- as the solver did before temperature was modelled --
    // overestimates the stable dt by that factor on any viscously limited run.
    // Returns the diffusivity multiplier on mu/rho.
    DNS_HD double diffusivityFactor() const
    {
        const double thermal = gamma / Pr;
        return thermal > 1.0 ? thermal : 1.0;
    }
};

} // namespace dns
