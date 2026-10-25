// RANS closure algebra: model selection, constants and the pointwise source
// terms, shared by the config layer (enum, names), the GPU kernels (float) and
// the FP64 host oracle (double). Host- and device-compilable (DNS_HD), the
// dns/sgs.hpp pattern. This header is the ONE home of every model constant.
//
// SA-neg is TMR's (https://tmbwg.github.io/turbmodels/spalart.html, read
// 2026-10-24): for nu~ >= 0 it IS standard SA -- WITH ft2 (ct3 = 1.2, ct4 =
// 0.5) and the note-1(c) S~ limiter; TMR: "(SA-noft2) is not compatible with
// (SA-neg)". For nu~ < 0 the negative branch below; mu_t = 0 there. nu~ is
// never clipped.
//
// The gradient terms (cb2 |grad nu~|^2, the conservative form's grad rho
// term, the face diffusion) need a stencil and live in the kernel / oracle;
// what lives here is everything pointwise.
#pragma once

#include "dns/hd.hpp"

#include <cmath>
#include <span>
#include <string_view>

namespace dns {

// The RANS closure a run solves; None = the DNS/LES path bit-for-bit.
enum class RansModel { None, SaNeg, Sst2003m, SstVm };

// Label spelling (reports, RunMeta, stats), indexed by the enum.
inline std::string_view ransName(RansModel m)
{
    constexpr std::string_view n[] = {"none", "SA-neg", "SST-2003m", "SST-Vm"};
    return n[int(m)];
}
// CLI spelling, indexed by the enum.
inline std::span<const char* const> ransCliNames()
{
    static constexpr const char* n[] = {"none", "sa-neg", "sst-2003m", "sst-vm"};
    return n;
}
// The closures `dns channel --rans` accepts: only what is BUILT is
// registered, so --help and the CLI's "one of:" list stay truthful.
inline std::span<const RansModel> ransBuiltChannel()
{
    static constexpr RansModel m[] = {RansModel::None, RansModel::SaNeg,
                                      RansModel::Sst2003m, RansModel::SstVm};
    return m;
}
// Conserved turbulence scalars the model carries: 0 | 1 (rho nu~) | 2 (rho k,
// rho omega).
inline int ransScalars(RansModel m)
{
    return m == RansModel::None ? 0 : m == RansModel::SaNeg ? 1 : 2;
}
inline bool ransSst(RansModel m)
{
    return m == RansModel::Sst2003m || m == RansModel::SstVm;
}

// Turbulent Prandtl number of the eddy conductivity mu_t cp / Pr_t (the LES
// slot's value, gpu/comp.hpp prt_).
inline constexpr double kPrandtlTurb = 0.9;

namespace sa {
// TMR spalart.html (2026-10-24), "Standard SA" constants and note 1(c).
inline constexpr double cb1 = 0.1355, sigma = 2.0 / 3.0, cb2 = 0.622,
                        kappa = 0.41, cw2 = 0.3, cw3 = 2.0, cv1 = 7.1,
                        ct3 = 1.2, ct4 = 0.5, cn1 = 16.0, c2 = 0.7, c3 = 0.9;
inline constexpr double cw1 = cb1 / (kappa * kappa) + (1.0 + cb2) / sigma;
inline constexpr double rMax = 10.0;   // r = min(nu~/(S~ kappa^2 d^2), 10)
} // namespace sa

// A pointwise model source per unit rho (a rate of the specific scalar) and
// J >= 0, the frozen-coefficient implicit rate -d(source)/d(scalar).
template <class T> struct RansPoint
{
    T R, J;
};

// fv1 = chi^3 / (chi^3 + cv1^3).
template <class T> DNS_HD T saFv1(T chi)
{
    const T c3 = chi * chi * chi;
    const T cv = T(sa::cv1);
    return c3 / (c3 + cv * cv * cv);
}

// mu_t = rho nu~ fv1(nu~/nu), and exactly 0 for nu~ <= 0 (SA-neg). Pre: nu > 0.
template <class T> DNS_HD T saMut(T rho, T nut, T nu)
{
    if (!(nut > T(0))) return T(0);
    return rho * nut * saFv1(nut / nu);
}

// fn = (cn1 + chi^3) / (cn1 - chi^3) on the negative branch, 1 otherwise.
template <class T> DNS_HD T saFn(T chi)
{
    if (!(chi < T(0))) return T(1);
    const T c3 = chi * chi * chi;
    return (T(sa::cn1) + c3) / (T(sa::cn1) - c3);
}

// The diffusion coefficient nu + nu~ fn(chi), evaluated by the caller ON THE
// FACE from face-averaged nu and nu~ (never the average of per-cell
// coefficients). Pre: nu > 0.
template <class T> DNS_HD T saDiffCoef(T nu, T nut)
{
    return nu + nut * saFn(nut / nu);
}

// The modified vorticity S~ with TMR note 1(c)'s limiter (c2 = 0.7, c3 =
// 0.9): continuous at Sbar = -c2 Omega and > 0 whenever Omega > 0.
// Pre: Omega >= 0.
template <class T> DNS_HD T saStilde(T Omega, T Sbar)
{
    if (Sbar >= -T(sa::c2) * Omega) return Omega + Sbar;
    const T c2 = T(sa::c2), c3 = T(sa::c3);
    return Omega + Omega * (c2 * c2 * Omega + c3 * Sbar) /
                       ((c3 - T(2) * c2) * Omega - Sbar);
}

// fw(r) = g [(1 + cw3^6)/(g^6 + cw3^6)]^(1/6), g = r + cw2 (r^6 - r).
template <class T> DNS_HD T saFw(T r)
{
    const T r2 = r * r, r6 = r2 * r2 * r2;
    const T g = r + T(sa::cw2) * (r6 - r);
    const T g2 = g * g, g6 = g2 * g2 * g2;
    const T c = T(sa::cw3), c2 = c * c, c6 = c2 * c2 * c2;
    return g * std::pow((T(1) + c6) / (g6 + c6), T(1) / T(6));
}

// r = min(nu~/(S~ kappa^2 d^2), 10), and 10 when S~ == 0 (TMR).
template <class T> DNS_HD T saR(T nut, T St, T d)
{
    if (St == T(0)) return T(sa::rMax);
    const T k2 = T(sa::kappa) * T(sa::kappa);
    const T r = nut / (St * k2 * d * d);
    return r < T(sa::rMax) ? r : T(sa::rMax);
}

// SA-neg pointwise source per unit rho: production - destruction (+ the ft2
// terms) for nu~ >= 0, TMR's negative branch otherwise. J >= 0 is the
// frozen-coefficient implicit rate (fw, ft2, S~ held fixed). Pre: d > 0,
// nu > 0, Omega >= 0. The cb2 and grad rho terms are NOT here (stencil).
template <class T> DNS_HD RansPoint<T> saSource(T nut, T nu, T Omega, T d)
{
    const T cb1 = T(sa::cb1), cw1 = T(sa::cw1);
    const T k2 = T(sa::kappa) * T(sa::kappa);
    const T id2 = T(1) / (d * d);
    if (nut < T(0)) {
        const T R = cb1 * (T(1) - T(sa::ct3)) * Omega * nut +
                    cw1 * nut * nut * id2;
        const T J = -cb1 * (T(1) - T(sa::ct3)) * Omega - T(2) * cw1 * nut * id2;
        return {R, J > T(0) ? J : T(0)};
    }
    const T chi = nut / nu;
    const T fv1 = saFv1(chi);
    const T fv2 = T(1) - chi / (T(1) + chi * fv1);
    const T ft2 = T(sa::ct3) * std::exp(-T(sa::ct4) * chi * chi);
    const T Sbar = nut * fv2 * id2 / k2;
    const T St = saStilde(Omega, Sbar);
    const T fw = saFw(saR(nut, St, d));
    const T dcoef = cw1 * fw - cb1 * ft2 / k2;
    const T R = cb1 * (T(1) - ft2) * St * nut - dcoef * nut * nut * id2;
    const T J = T(2) * dcoef * nut * id2;
    return {R, J > T(0) ? J : T(0)};
}

// --- SST (Menter) -------------------------------------------------------------
// TMR sst.html (read 2026-10-24). Two variants, both "m" -- the 2/3 rho k
// isotropic part is DROPPED from tau_ij in momentum and energy (so mu_t rides
// the eddy-viscosity slot unchanged):
//   SST-2003m  2003 constants (gamma1 = 5/9, gamma2 = 0.44), mu_t with
//              S = sqrt(2 S:S), P = mu_t S^2, limiter 10 beta* rho omega k in
//              BOTH equations (omega production gamma/nu_t P~), CDkw floor
//              1e-10.
//   SST-Vm     1994 SST with vorticity: mu_t with Omega, P = mu_t Omega^2,
//              limiter 20 beta* rho omega k in the k equation only, omega
//              production gamma/nu_t P (unlimited), gamma_i derived
//              (beta_i/beta* - sigma_wi kappa^2/sqrt(beta*)), CDkw floor 1e-20.
// Full SST-2003 (2/3 rho k kept) would touch kPrim/faceFlux: not built.
namespace sst {
inline constexpr double betaStar = 0.09, a1 = 0.31, kappa = 0.41;
inline constexpr double sigmaK1 = 0.85, sigmaW1 = 0.5, beta1 = 0.075;
inline constexpr double sigmaK2 = 1.0, sigmaW2 = 0.856, beta2 = 0.0828;
// The TMR wall value omega_w = 10 * 6 nu / (beta1 d1^2), d1 = the distance
// of the first point off the wall (here: the first cell centre).
inline constexpr double omegaWallFactor = 60.0 / beta1;
} // namespace sst

// gamma_i: 2003 fixes 5/9 and 0.44; the 1994 form derives them so that the
// log layer balances exactly (sqrt(0.09) = 0.3 exactly).
inline constexpr double sstGamma1(RansModel m)
{
    return m == RansModel::Sst2003m
               ? 5.0 / 9.0
               : sst::beta1 / sst::betaStar -
                     sst::sigmaW1 * sst::kappa * sst::kappa / 0.3;
}
inline constexpr double sstGamma2(RansModel m)
{
    return m == RansModel::Sst2003m
               ? 0.44
               : sst::beta2 / sst::betaStar -
                     sst::sigmaW2 * sst::kappa * sst::kappa / 0.3;
}
inline constexpr double sstCdFloor(RansModel m)
{
    return m == RansModel::Sst2003m ? 1e-10 : 1e-20;
}

// phi = F1 phi1 + (1 - F1) phi2.
template <class T> DNS_HD T sstBlendC(T F1, double c1, double c2)
{
    return F1 * T(c1) + (T(1) - F1) * T(c2);
}

// The cross-diffusion product CDkw = max(2 rho sigma_w2 (grad k . grad w)/w,
// floor). gkw = grad k . grad omega. Pre: omega > 0.
template <class T> DNS_HD T sstCd(T rho, T gkw, T omega, T floor)
{
    const T cd = T(2) * rho * T(sst::sigmaW2) * gkw / omega;
    return cd > floor ? cd : floor;
}

// F1, F2 of the SST blend. arg1 is clamped at 10 before the 4th power: exact,
// tanh(10^4) is already 1 in every precision, and it keeps the power finite.
// Pre: d > 0, nu > 0, omega > 0, k >= 0, cd > 0.
template <class T> struct SstBlend
{
    T F1, F2;
};
template <class T> DNS_HD SstBlend<T> sstBlend(T k, T omega, T d, T nu, T rho,
                                               T cd)
{
    const T sk = std::sqrt(k > T(0) ? k : T(0));
    const T a = sk / (T(sst::betaStar) * omega * d);
    const T b = T(500) * nu / (d * d * omega);
    const T c = T(4) * rho * T(sst::sigmaW2) * k / (cd * d * d);
    const T ab = a > b ? a : b;
    T arg1 = ab < c ? ab : c;
    if (arg1 > T(10)) arg1 = T(10);
    const T a2 = T(2) * a, arg2v = a2 > b ? a2 : b;
    const T arg2 = arg2v > T(10) ? T(10) : arg2v;
    const T q1 = arg1 * arg1;
    return {std::tanh(q1 * q1), std::tanh(arg2 * arg2)};
}

// mu_t = rho a1 k / max(a1 omega, R F2), R = S (2003m) or Omega (Vm); 0 for
// k <= 0. Pre: omega > 0.
template <class T> DNS_HD T sstMut(T rho, T k, T omega, T R, T F2)
{
    if (!(k > T(0))) return T(0);
    const T a = T(sst::a1) * omega, b = R * F2;
    return rho * T(sst::a1) * k / (a > b ? a : b);
}

// Pointwise SST sources per unit volume, with the implicit rates (frozen
// coefficients, both >= 0): k: P~ - beta* rho omega k; omega: production
// (gamma/nu_t P~ for 2003m, gamma/nu_t P for Vm, written as gamma rho R^2
// with the 2003m limiter divided through) - beta rho omega^2 + the
// cross-diffusion (1 - F1) CDkw. The cross-diffusion's NEGATIVE part goes into
// J_omega (it destroys omega), its positive part stays explicit.
// R = S (2003m) or Omega (Vm), the rate the production uses.
// Pre: omega > 0, rho > 0, mut >= 0.
template <class T> struct SstPoint
{
    T Rk, Jk, Rw, Jw;
};
template <class T>
DNS_HD SstPoint<T> sstSource(RansModel m, T rho, T k, T omega, T R, T mut,
                             T F1, T gkw)
{
    const bool m03 = m == RansModel::Sst2003m;
    const T bs = T(sst::betaStar);
    const T beta = sstBlendC(F1, sst::beta1, sst::beta2);
    const T gam = sstBlendC(F1, sstGamma1(m), sstGamma2(m));
    const T R2 = R * R;
    const T P = mut * R2;
    const T lim = T(m03 ? 10.0 : 20.0) * bs * rho * omega * k;
    const T Pt = P < lim ? P : lim;
    // omega production gamma/nu_t P (Vm) or gamma/nu_t P~ (2003m) = gamma rho
    // min(R^2, lim/mu_t): finite as mu_t -> 0 (it tends to gamma rho R^2).
    T Pw = gam * rho * R2;
    if (m03 && P > lim) Pw = gam * rho * lim / mut;
    const T cdw = T(2) * (T(1) - F1) * rho * T(sst::sigmaW2) * gkw / omega;
    SstPoint<T> s;
    s.Rk = Pt - bs * rho * omega * k;
    s.Jk = bs * omega;
    s.Rw = Pw - beta * rho * omega * omega + cdw;
    s.Jw = T(2) * beta * omega + (cdw < T(0) ? -cdw / (rho * omega) : T(0));
    return s;
}

// --- wall-bounded initial guesses (channel IC, oracle start) -----------------

// Reichardt's composite law of the wall, u+(y+) = ln(1 + kappa y+)/kappa +
// 7.8 [1 - exp(-y+/11) - (y+/11) exp(-y+/3)]: viscous sublayer to log layer in
// one smooth curve (kappa = 0.41, the SA value). A STARTING guess, nothing is
// measured against it.
inline double reichardtUplus(double yp)
{
    return std::log(1.0 + sa::kappa * yp) / sa::kappa +
           7.8 * (1.0 - std::exp(-yp / 11.0) -
                  yp / 11.0 * std::exp(-yp / 3.0));
}

// SA starting field nu~ = kappa u_tau d (1 - d/(2 delta)): the log-layer value
// at the wall, a parabola to the centre. Pre: 0 <= d <= delta.
inline double saInitialNut(double d, double uTau, double delta)
{
    return sa::kappa * uTau * d * (1.0 - 0.5 * d / delta);
}

// SST starting field: the log-layer k = u_tau^2/sqrt(beta*) ramped to 0 as
// (y+/10)^2 below y+ = 10, and omega = sqrt(omega_log^2 + omega_vis^2) with
// omega_log = u_tau/(sqrt(beta*) kappa d) and the viscous-sublayer omega_vis
// = 6 nu/(beta1 d^2). A STARTING guess only. Pre: d > 0, nu > 0.
struct KOmega
{
    double k, omega;
};
inline KOmega sstInitialKOmega(double d, double nu, double uTau)
{
    const double yp = d * uTau / nu;
    const double ramp = yp < 10.0 ? (yp / 10.0) * (yp / 10.0) : 1.0;
    const double wl = uTau / (0.3 * sst::kappa * d);
    const double wv = 6.0 * nu / (sst::beta1 * d * d);
    return {uTau * uTau / 0.3 * ramp, std::sqrt(wl * wl + wv * wv)};
}

// The point-implicit stage combination of one conserved scalar q:
//   out = (comb + cr (R + J q)) / (1 + cr J),  comb = ca qa + cb qb,
// exact at a steady state (R == 0, comb == q gives out == q up to the
// quotient's rounding) for any J >= 0. At J == 0 it is comb + cr R, the mean
// flow's own combination expression -- which is what lets a scalar riding
// the density be transported bit for bit (gpu_rans G3). Pre: cr >= 0, J >= 0.
template <class T> DNS_HD T pointImplicit(T comb, T cr, T R, T J, T q)
{
    return (comb + cr * (R + J * q)) / (T(1) + cr * J);
}

} // namespace dns
