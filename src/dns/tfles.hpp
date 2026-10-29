// Thickened-flame LES (TFLES) of `dns flame` (docs/chem.md, "LES species and
// TFLES"): the model constants, Charlette's efficiency function, Colin's
// sub-grid velocity scaling and the flame sensor's normalisation. Host- and
// device-compilable (DNS_HD), so the GPU kernels (gpu/tfles.hip) and the host
// tests evaluate ONE expression.
//
// Formulation (F_max = --tfles, Omega in [0, 1] the flame sensor):
//   F  = 1 + (F_max - 1) Omega                       per cell
//   rho D_eff = F E mu/Pr + (1 - Omega) mu_t/Sc_t     species
//   kappa_eff = F E kappa + (1 - Omega) mu_t cp/Pr_t  energy (Sc_t = Pr_t)
//   omega_eff = (E/F) omega                          fuel and heat release
// Momentum is not thickened. Uniform F with E = 1 is the laminar flame
// stretched by F along x: S_L is unchanged (the discrete equations map onto
// the unthickened ones with h -> F h).
#pragma once

#include "dns/hd.hpp"

#include <cmath>
#include <numbers>

namespace dns {

// Flame sensor: tanh(kTflesSensorBeta w_s / w_s,max), w_s the Arrhenius rate
// at the REDUCED activation temperature kTflesTaFraction Ta (Legier, Poinsot
// & Veynante 2000: it reaches into the preheat zone), then spread TOWARD
// COLDER GAS by a T-gated max filter of radius ceil(kTflesDilate F_max
// delta_th / h) cells: Omega(c) = max of Omega_raw over the box cells n with
// T_n >= T_c (separable passes, each gated on its own centre cell; x clamped
// at the open faces, y and z periodic). The gate is what lets one radius
// cover the preheat zone (c >= 0.01) without smearing the burnt side: a
// symmetric max filter covering c = 0.01 left Omega 0.03-0.6 at 2 F delta_th
// past c = 0.99 (host scan on the thickened oracle, 2026-10-29). Measured
// choices (docs/chem.md), not bracketed against a turbulent flame.
inline constexpr double kTflesSensorBeta = 50.0;
inline constexpr double kTflesTaFraction = 0.5;
inline constexpr double kTflesDilate = 2.5;
// The outermost x rows on each side where E = 1 and u'_Delta = 0: the curl
// and its Laplacian would difference across the periodic wrap there.
inline constexpr int kTflesEdgeRows = 3;
// Charlette, Meneveau & Veynante (2002) static exponent, C_k, and Colin et
// al. (2000)'s operator constant c2.
inline constexpr double kCharletteBeta = 0.5;
inline constexpr double kCharletteCk = 1.5;
inline constexpr double kColinC2 = 2.0;

// Charlette et al. 2002 (Combust. Flame 131:159) power-law wrinkling factor,
// static beta, in the saturated form of Wang, Boileau & Veynante 2011 (the
// "- 1" makes Xi(Delta = delta_L) = 1):
//   Xi = (1 + min[r - 1, Gamma(r, u, Re) u])^beta,  r = Delta/delta_L,
//   u = u'_Delta/S_L, Re = u'_Delta Delta/nu,
//   Gamma = [((f_u^-a + f_D^-a)^(-1/a))^-b + f_Re^-b]^(-1/b),
//   f_u  = 4 sqrt(27 Ck/110) (18 Ck/55) u^2,
//   f_D  = sqrt(27 Ck pi^(4/3)/110 (r^(4/3) - 1)),
//   f_Re = sqrt(9/55 exp(-3/2 Ck pi^(4/3)/Re)) sqrt(Re),
//   a = 0.60 + 0.20 exp(-0.1 u) - 0.20 exp(-0.01 r), b = 1.4, Ck = 1.5.
// Returns exactly 1 for r <= 1, u <= 0 or Re <= 0 (no sub-grid wrinkling);
// otherwise 1 <= Xi <= r^beta.
template <class T>
DNS_HD T charletteEfficiency(T r, T u, T re, T beta)
{
    if (!(r > T(1)) || !(u > T(0)) || !(re > T(0))) return T(1);
    const T ck = T(kCharletteCk);
    const T pi43 = T(4.6011511144704899); // pi^(4/3)
    const T fu = T(4) * std::sqrt(T(27) * ck / T(110)) * (T(18) * ck / T(55)) *
                 u * u;
    const T fd = std::sqrt(T(27) * ck * pi43 / T(110) *
                           (std::pow(r, T(4) / T(3)) - T(1)));
    const T fre = std::sqrt(T(9) / T(55) * std::exp(T(-1.5) * ck * pi43 / re)) *
                  std::sqrt(re);
    const T a = T(0.6) + T(0.2) * std::exp(T(-0.1) * u) -
                T(0.2) * std::exp(T(-0.01) * r);
    const T b = T(1.4);
    const T fud = std::pow(std::pow(fu, -a) + std::pow(fd, -a), T(-1) / a);
    const T gam = std::pow(std::pow(fud, -b) + std::pow(fre, -b), T(-1) / b);
    const T w = gam * u < r - T(1) ? gam * u : r - T(1);
    return std::pow(T(1) + w, beta);
}

// Colin et al. 2000 sub-grid velocity at the filter Delta from the resolved
// field on spacing h: u'_Delta = c2 h^3 |lap(curl u)| (Delta/h)^(1/3), given
// the grid-scale value uh = c2 h^3 |lap(curl u)|. The curl removes the
// dilatation, so a planar laminar flame has u'_Delta = 0 exactly.
template <class T>
DNS_HD T colinUprime(T uh, T Delta, T h)
{
    return uh * std::cbrt(Delta / h);
}

// The sensor's normalisation: ln of max over t in [1, sigma) of
// exp(-thetaS / t) (1/t) (sigma - t)/(sigma - 1) -- the reduced-activation
// rate exp(-TaS/T) rho Y on the low-Mach Shvab-Zeldovich manifold (rho = Tu/T,
// Y = (Tb - T)/(Tb - Tu); every point of the oracle profile lies on it), with
// t = T/Tu, thetaS = TaS/Tu. lnAs = -tflesSensorLogMax(...) then makes
// exp(lnAs - TaS/T) rho Y peak at 1 along the laminar flame. FP64 bisection on
// d/dt ln f = thetaS/t^2 - 1/t - 1/(sigma - t) (decreasing on [1, sigma) for
// thetaS >= sigma/2: ln f is concave there); a non-positive slope at t = 1
// puts the maximum there. Pre: sigma > 1, thetaS > 0.
inline double tflesSensorLogMax(double sigma, double thetaS)
{
    auto slope = [&](double t) {
        return thetaS / (t * t) - 1.0 / t - 1.0 / (sigma - t);
    };
    auto lnf = [&](double t) {
        return -thetaS / t - std::log(t) + std::log((sigma - t) / (sigma - 1.0));
    };
    if (!(slope(1.0) > 0)) return lnf(1.0);
    double lo = 1.0, hi = sigma;
    for (int it = 0; it < 200; ++it) {
        const double m = 0.5 * (lo + hi);
        if (m <= lo || m >= hi) break;
        (slope(m) > 0 ? lo : hi) = m;
    }
    return lnf(0.5 * (lo + hi));
}

// The resolved TFLES setup the runner hands to gpu/tfles (box units of `dns
// flame`). Pre: F >= 1, deltaL > 0, nuU > 0, dilate >= 0; charlette needs a
// 3D grid (validator: it needs --les, which needs --nz >= 8).
struct TflesSetup
{
    double F = 1;             // F_max
    bool uniform = false;     // Omega == 1 everywhere (--tfles-sensor uniform)
    bool charlette = false;   // E by charletteEfficiency (else E == 1)
    double deltaL = 1;        // delta_L of the efficiency: the oracle delta_th
    double nuU = 0.72;        // fresh-gas kinematic viscosity (mu / rho_u)
    double effBeta = kCharletteBeta;
    double lnAs = 0, TaS = 0; // the sensor's rate (lnAs = -tflesSensorLogMax)
    double sensorBeta = kTflesSensorBeta;
    int dilate = 0;           // the sensor's max-filter radius, cells
};

} // namespace dns
