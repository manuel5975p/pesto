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

// A dimensional (or non-dimensional) 1D flame: F -> P, Le = 1, Y_u = 1, so
// Y = (Tb - T)/(Tb - Tu) and q = cp (Tb - Tu). rho(T) and lambda(T) are the
// isobaric state; omega = A rho Y exp(-Ta/T).
// Pre: 0 < Tu < Tb, cp > 0, A > 0, Ta > 0, rho and lambda positive on [Tu, Tb].
struct FlameProblem
{
    double Tu = 1, Tb = 5, cp = 1, A = 1, Ta = 50;
    std::function<double(double)> rho, lambda;
};

// The phase-plane solution: mass flux m = rho_u S_L, thermal thickness
// delta_th = (Tb - Tu) / max dT/dx, and the profile table (T_k, x_k, dT/dx_k)
// with dT/dx = p / lambda (p = lambda dT/dx is the shooting variable), x
// increasing with T, x = 0 at T = (Tu + Tb)/2.
struct FlameEigen
{
    double m = 0, dth = 0;
    std::vector<double> T, x, dTdx;
};

// Eigenvalue m of the steady flame by shooting from the hot end (analytic
// start p = a s) to the cold end in the logit variable of (Tb - T)/(Tb - Tu),
// RK4 with n steps, bisection on the cold residual p - m cp (T - Tu). FP64.
// Pre: n >= 100. Error: no bracket (the problem has no flame solution).
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

// T(x) of an oracle flame at x (x = 0 at the mid temperature): cubic Hermite
// on the (x_k, T_k) table with the exact slopes dT/dx_k (4th order), clamped
// to the table's end temperatures past its ends. For premixedEigen's table:
// x in delta_D, T in Tu. Pre: e.T.size() >= 2.
double flameProfile(const FlameEigen& e, double x);

} // namespace dns
