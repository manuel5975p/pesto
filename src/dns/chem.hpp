// Premixed single-step chemistry, config layer (docs/chem.md): the model
// constants, the ONE Arrhenius expression the kernels, the oracle and the IC
// share, and the FP64 phase-plane eigenvalue oracle of the 1D freely
// propagating flame. Pure: no filesystem, no GPU -- validate() calls it to
// resolve the grid and print Lambda on --dry-run.
//
// Model C0 (P1): calorically perfect, gamma 1.4 fixed, equal molecular
// weights (so R and cp do not depend on composition), Le = 1, F -> P with
// one species carried (Y = Y_F; Y_P = 1 - Y), unity Y_u.
#pragma once

#include "dns/hd.hpp"

#include <expected>
#include <functional>
#include <math.h>
#include <span>
#include <string>
#include <vector>

namespace dns {

// The non-dimensional premixed model. Box units of `dns flame`: rho_u = 1,
// S_L = 1, R = 1, delta_D = lambda_u / (rho_u cp S_L) = 1, so A = Lambda.
// Pre (validated): 1 < sigma <= 10, 2 <= beta <= 20, Pr > 0, sOverTu >= 0.
struct PremixedModel
{
    double sigma = 5.0;    // expansion ratio Tb / Tu
    double beta = 8.0;     // Zeldovich number theta (sigma - 1) / sigma^2
    double Pr = 0.72;      // molecular Prandtl number; Le = 1
    bool sutherland = false;
    double sOverTu = 0.0;  // Sutherland S / Tu (0 with the law off)

    // Activation temperature over Tu: theta = beta sigma^2 / (sigma - 1).
    double theta() const { return beta * sigma * sigma / (sigma - 1.0); }
    // lambda(T)/lambda_u at t = T/Tu: Transport::ratio's expression.
    double lambdaRatio(double t) const
    {
        if (!sutherland) return 1.0;
        return t * sqrt(t) * (1.0 + sOverTu) / (t + sOverTu);
    }
};

// Fuel consumption rate omega = exp(lnA - Ta/T) rho max(Y, 0). The log form
// keeps the pre-exponential in FP32 range (Lambda ~ 4e6, e^-50 at Tu). The
// rate floors Y at zero; the STATE is never clamped (docs/chem.md).
// Pre: T > 0.
template <class R>
DNS_HD inline R arrheniusRate(R lnA, R Ta, R rho, R Y, R T)
{
    return exp(lnA - Ta / T) * rho * (Y > R(0) ? Y : R(0));
}

// Bimolecular fuel consumption rate of the diffusion flame F + s O -> (1+s) P
// (dns/diffusion.hpp): omega = exp(lnA - Ta/T) rho^2 max(Y_F, 0) max(Y_O, 0).
// Same floors, same log form. Pre: T > 0.
template <class R>
DNS_HD inline R arrheniusRate2(R lnA, R Ta, R rho, R YF, R YO, R T)
{
    return exp(lnA - Ta / T) * rho * rho * (YF > R(0) ? YF : R(0)) *
           (YO > R(0) ? YO : R(0));
}

// A dimensional (or non-dimensional) 1D flame: F -> P, Le = 1, Y_u = 1, so
// Y = (Tb - T)/(Tb - Tu) and q = cp (Tb - Tu). rho(T) and lambda(T) are the
// isobaric state; omega = A rho Y exp(-Ta/T) rateFactor(T, Y).
// rateFactor: a second reactant's concentration (premixedEigen2); empty = 1,
// the single-reactant arithmetic bit for bit. The hot-end start reads
// rateFactor(Tb, 0).
// Pre: 0 < Tu < Tb, cp > 0, A > 0, Ta > 0, rho and lambda positive on [Tu, Tb],
// rateFactor (if set) positive on [Tu, Tb] x [0, 1].
struct FlameProblem
{
    double Tu = 1, Tb = 5, cp = 1, A = 1, Ta = 50;
    std::function<double(double)> rho, lambda;
    std::function<double(double, double)> rateFactor;
};

// The phase-plane solution: mass flux m = rho_u S_L, thermal thickness
// delta_th = (Tb - Tu) / max dT/dx, and the profile table (T_k, x_k, dT/dx_k)
// with dT/dx = p / lambda (p = lambda dT/dx is the shooting variable), x
// increasing with T, x = 0 at T = (Tu + Tb)/2. tCut: the cold cutoff
// temperature below which the rate was switched off (kColdCut; 0 = none, the
// reaction runs to the cold end).
struct FlameEigen
{
    double m = 0, dth = 0, tCut = 0;
    std::vector<double> T, x, dTdx;
};

// The cold-boundary bound: a shot switches its rate off where the
// quasi-steady manifold p* = lambda q omega/(m cp) climbs back above this
// fraction of the inert preheat line m cp (T - Tu) on the cold side (the
// unburnt mixture's own reaction no longer negligible: the cold-boundary
// difficulty). Never active when p*/(m cp (T - Tu)) stays below it to the
// cold end -- every shot of a well-posed problem, bit for bit the uncut
// march. Measured (beta 8): Lambda moves 4e-5 per decade of this bound at
// sigma 2, the uncut march drifted as 1/n there (issues.md 72).
inline constexpr double kColdCut = 0.5;
// The cut must sit within this fraction of the rise above Tu, else the
// eigenvalue is the cutoff's, not the flame's (an error by name). Measured
// (kColdCut 0.5 vs 0.1): Lambda moves 3.5e-5 at a cut 1.3e-5 above Tu (sigma
// 2, beta 8), 0.4-0.6% at 1-2e-3 (beta 4-8 at low sigma), >2% beyond 5e-3.
inline constexpr double kColdCutMaxRise = 1e-4;

// Eigenvalue m of the steady flame by shooting from the hot end (analytic
// start p = a s) to the cold end in the logit variable of (Tb - T)/(Tb - Tu),
// RK4 with n steps. A shot is TOO FAST if p reaches 0, if its residual p - m
// cp (T - Tu) at the cold end is negative, or if a step goes stiff (h J > 1,
// J = lambda q omega |dT/du| / p^2: the trajectory collapsed onto the
// quasi-steady manifold p ~ p*, which only a too-fast shot does; RK4 there
// bounced p to a spurious positive residual, issues.md 72). Bracket: a scan of
// m = 2^-60 .. 2^20 (81 powers of two) that must show exactly ONE slow ->
// fast transition, then bisection over the shipped bracket ([lo, 1] or [1,
// hi]: the old doubling/halving from m = 1, so every well-posed problem keeps
// its bits). FP64.
// Pre: n >= 100. Errors: no bracket in the scan, a non-monotone scan, no
// inert preheat zone (the rate never falls below kColdCut of the preheat
// line), a cut above kColdCutMaxRise (both: the cold-boundary difficulty).
std::expected<FlameEigen, std::string> flameEigen(const FlameProblem&,
                                                  int n = 20000);

// The non-dimensional model's eigenvalue: Lambda = 1/m(A = 1)^2 (m scales as
// sqrt(A) exactly), so A = Lambda burns at S_L = 1. dthOverDD = delta_th /
// delta_D; the table is rescaled to A = Lambda (x in delta_D, T in Tu).
struct PremixedEigen
{
    double Lambda = 0, dthOverDD = 0;
    FlameEigen tab;
};
std::expected<PremixedEigen, std::string> premixedEigen(const PremixedModel&,
                                                        int n = 20000);

// Model C2 (docs/chem.md "Model C2"): the RICH premixed jet of a partially
// premixed flame on the two-reactant step F + s O -> (1+s) P of model C1
// (arrheniusRate2, omega = A rho^2 Y_F Y_O e^(-Ta/T)), Le = 1, equal W,
// calorically perfect. The jet is fuel + air at equivalence ratio phi > 1:
// f = phi yO2/s fuel per air, Y_F1 = f/(1+f), Y_O1 = yO2/(1+f). O is
// deficient: the inner premixed flame leaves Y_F,rem = (phi - 1) Y_O1/s,
// which burns against the surrounding air (Y_O = yO2, Z = 0) at Z_st. q
// (per unit fuel) is fixed by the INNER flame: sigma = 1 + q (Y_O1/s)/(cp
// Tu). Box units as model C0 (rho_u = S_L = R = 1, delta_D = 1), so A =
// Lambda2 burns the jet at S_L = 1.
// Pre (validated): 1 < phi <= 1.6, s > 0, 0 < yO2 <= 1, 1 < sigma <= 10,
// 2 <= beta <= 20, Pr > 0, sOverTu >= 0.
struct PartialPremixModel
{
    double phi = 1.3;      // jet equivalence ratio
    double s = 3.99;       // stoichiometric oxidizer-to-fuel mass ratio (CH4 + 2 O2)
    double yO2 = 0.233;    // the air's oxidizer mass fraction
    double sigma = 6.88;   // inner (rich premixed) flame Tb/Tu
    double beta = 8.0;     // Zeldovich number of the inner flame
    double Pr = 0.72;
    bool sutherland = false;
    double sOverTu = 0.0;

    double theta() const { return beta * sigma * sigma / (sigma - 1.0); }
    double yF1() const { return f() / (1.0 + f()); }
    double yO1() const { return yO2 / (1.0 + f()); }
    double yFrem() const { return (phi - 1.0) * yO1() / s; }
    // A LEAN premix's leftover oxidizer (phi < 1; the rim's air-diluted jet).
    double yOrem() const { return (1.0 - phi) * yO1(); }
    // Z = 1 in the jet (fresh or burnt), 0 in the air; Z_st where the jet's
    // leftover fuel meets the air stoichiometrically.
    double zst() const { return (yO2 / s) / (yFrem() + yO2 / s); }
    // q / (cp Tu): (sigma - 1) s / Y_O1.
    double qOverCpTu() const { return (sigma - 1.0) * s / yO1(); }
    // Tb/Tu of a premix at phi on the jet's q: 1 + q Y_def/(cp Tu), Y_def =
    // Y_O1/s rich, Y_F1 lean (complete combustion of the deficient reactant,
    // equal cp); the jet's own phi returns sigma exactly. Pre: phi > 0.
    double sigmaAt(double ph) const
    {
        if (ph == phi) return sigma;
        PartialPremixModel at = *this;
        at.phi = ph;
        return ph > 1 ? 1 + qOverCpTu() * at.yO1() / s
                      : 1 + qOverCpTu() * at.yF1();
    }
    // Burke-Schumann sheet temperature of the jet against the air: 1 + q Z_st
    // Y_F1 / (cp Tu) (the fresh jet's fuel at Z_st, all of it burnt).
    double sigmaSt() const { return 1.0 + qOverCpTu() * zst() * yF1(); }
    // Mixture fraction of (Y_F, Y_O) on the coupling function Y_F - Y_O/s.
    double mixtureFraction(double YF, double YO) const
    {
        return (s * YF - YO + yO2) / (s * yFrem() + yO2);
    }
    double lambdaRatio(double t) const
    {
        if (!sutherland) return 1.0;
        return t * sqrt(t) * (1.0 + sOverTu) / (t + sOverTu);
    }

private:
    double f() const { return phi * yO2 / s; }
};

// Model C2b (docs/chem.md "Model C2b", masterplan real_bunsen S1b): the
// two-reactant rate with a mixture-dependent pre-exponential, omega = A C(xi)
// rho^2 Y_F Y_O e^(-Ta/T), so the one-step's premixed S_L(phi) follows
// GRI-3.0 CH4/air (the uncorrected step's S_L RISES with phi on the rich side
// where CH4's falls 5x, issues.md 73). xi = (s Y_F - Y_O + Y_O2)/(s + Y_O2) is
// the fuel/air mixture fraction (= Y_F1 of a fresh premix): a Shvab-Zeldovich
// coupling function, conserved by the step and carried exactly by the shared
// limiter, so a premixed flame sees ONE C across it and the oracle's
// eigenvalue scales as sqrt(C). ln C is a cubic spline on nodes xi_k,
// clamped (zero slope) at both end nodes and constant beyond them: C1
// everywhere, C2 between the ends. n == 0: off (never evaluated: the
// callers select the uncorrected instantiation). The ONE definition the
// oracle (R = double) and the cell pass (R = Real) evaluate (phiRateLn).
template <class R>
struct PhiRateLaw
{
    static constexpr int kMaxNodes = 32;
    int n = 0;
    R s = 0, yO2 = 0, invDen = 0; // invDen = 1/(s + yO2)
    R xi[kMaxNodes] = {}, lnC[kMaxNodes] = {}, m2[kMaxNodes] = {};

    template <class Q>
    PhiRateLaw<Q> as() const
    {
        PhiRateLaw<Q> o;
        o.n = n;
        o.s = Q(s);
        o.yO2 = Q(yO2);
        o.invDen = Q(invDen);
        for (int k = 0; k < kMaxNodes; ++k) {
            o.xi[k] = Q(xi[k]);
            o.lnC[k] = Q(lnC[k]);
            o.m2[k] = Q(m2[k]);
        }
        return o;
    }
};

// ln C(xi(Y_F, Y_O)) of PhiRateLaw: the spline on the bracketing interval
// (ascending linear scan), the end values outside. A NaN xi returns NaN.
// Pre: L.n >= 2, L.xi strictly ascending.
template <class R>
DNS_HD inline R phiRateLn(const PhiRateLaw<R>& L, R YF, R YO)
{
    const R xi = (L.s * YF - YO + L.yO2) * L.invDen;
    if (xi <= L.xi[0]) return L.lnC[0];
    if (xi >= L.xi[L.n - 1]) return L.lnC[L.n - 1];
    int k = 0;
    while (xi > L.xi[k + 1]) ++k;
    const R h = L.xi[k + 1] - L.xi[k];
    const R t = (xi - L.xi[k]) / h, u = R(1) - t;
    return u * L.lnC[k] + t * L.lnC[k + 1] +
           h * h / R(6) *
               ((u * u * u - u) * L.m2[k] + (t * t * t - t) * L.m2[k + 1]);
}

// C of a fresh premix at phi (its Y_F1, Y_O1) under law L of md's s, Y_O2.
// Pre: L.n >= 2, phi > 0.
double phiRateC(const PhiRateLaw<double>& L, const PartialPremixModel& md,
                double phi);

// GRI-3.0 CH4/air laminar burning velocity, Tu 298 K, 1 atm (Cantera 3.2.0,
// mixture-averaged; tests/data/chem/ch4_gri30.csv, tools/chem/cantera_flame.py
// --real): phi ascending, S_L in m/s. The A(phi) law's target SHAPE.
struct Ch4SpeedNode
{
    double phi, SL;
};
std::span<const Ch4SpeedNode> ch4Gri30Speeds();

// Lambda2 = 1/m(A = 1)^2 of the jet's inner flame (the C2 rate in
// progress s_p = (Tb - T)/(Tb - Tu) = Y_O/Y_O1, Y_F = Y_F,rem + Y_O/s:
// rateFactor(T, s_p) = rho(T) (s Y_F,rem + Y_O1 s_p)); table and thickness as
// premixedEigen. Uncorrected (C = 1): the C2b law is normalised to C(jet) = 1,
// so Lambda2 is the A of both.
std::expected<PremixedEigen, std::string>
premixedEigen2(const PartialPremixModel&, int n = 20000);

// A premix of the jet's model at equivalence ratio phi (the jet diluted with
// air at the rim, or any CH4/air premix): the jet's A = Lambda2, q, Ta; the
// burnt temperature sigmaAt(phi). Lean (phi < 1): F deficient, Y_F = Y_F1
// s_p, Y_O = Y_O,rem + s Y_F. law (null = C 1): the C2b rate. Result: Lambda
// = the JET's Lambda2 (the run's A), tab.m = S_L(phi)/S_L(jet) (rho_u = 1),
// the profile table in the jet's delta_D, dthOverDD this flame's.
// Pre: phi > 0, phi != 1 (the hot-end start needs a leftover reactant).
std::expected<PremixedEigen, std::string>
premixedEigen2At(const PartialPremixModel& md, double phi,
                 const PhiRateLaw<double>* law = nullptr, int n = 20000);

// premixedEigen2At(md, phi, law).tab.m: the model's S_L(phi)/S_L(md.phi).
// law null: the uncorrected one-step (printed beside the Cantera curve);
// phi == md.phi gives 1 exactly. Pre: as premixedEigen2At.
std::expected<double, std::string>
premixedSpeedRatio(const PartialPremixModel& md, double phi,
                   const PhiRateLaw<double>* law = nullptr, int n = 20000);

// The C2b law of md's jet: node k at xi_k = Y_F1(phi_k), ln C_k = 2 ln(
// S_GRI(phi_k) / m_k) with m_k the UNCORRECTED model's eigenvalue there (C
// enters as sqrt(C)), shifted so ln C(xi_jet) = 0 (the jet keeps A =
// Lambda2 and S_L = 1). nodes: GRI-3.0 rows (default ch4Gri30Speeds();
// a subset for the leave-out test); phi == 1 rows are skipped (no hot-end
// start). One oracle solve per node, threaded (own-index writes: bit-
// identical at any worker count). Errors: an oracle failure at a node, fewer
// than 3 usable nodes, more than kMaxNodes.
std::expected<PhiRateLaw<double>, std::string>
phiRateLaw(const PartialPremixModel& md,
           std::span<const Ch4SpeedNode> nodes = ch4Gri30Speeds(),
           int n = 20000);

// T(x) of an oracle flame at x (x = 0 at the mid temperature): cubic Hermite
// on the (x_k, T_k) table with the exact slopes dT/dx_k (4th order), clamped
// to the table's end temperatures past its ends. For premixedEigen's table:
// x in delta_D, T in Tu. Pre: e.T.size() >= 2.
double flameProfile(const FlameEigen& e, double x);

} // namespace dns
