// SGS model selection and the sigma-model differential operator, shared by the
// pure config layer (enum only) and the GPU nut kernels (operator). Host- and
// device-compilable (DNS_HD), so the operator is unit-testable on the CPU.
//
// Model catalogue:
//   Smagorinsky  nut = (Cs Delta)^2 |S|            Cs ~ 0.17
//   Sigma        nut = (Cσ Delta)^2 D_sigma        Cσ ~ 1.35 (Nicoud et al.
//                                                  Phys. Fluids 23, 085106, 2011)
//
// D_sigma = σ3 (σ1 - σ2)(σ2 - σ3) / σ1^2, with σ1 >= σ2 >= σ3 >= 0 the
// singular values of the velocity-gradient tensor g_ij = du_i/dx_j. By
// construction D_sigma vanishes identically for any two-component or
// two-dimensional flow, pure shear, solid rotation, and axisymmetric
// strain, and decays as y^3 at a wall -- exactly the laminar/transitional
// states where Smagorinsky's |S| injects spurious eddy viscosity. That is
// why it is the transition-capable choice for tripped/dimpled-body runs.
#pragma once

#include "dns/hd.hpp"

#include <cmath>

namespace dns {

enum class SgsModel { Smagorinsky, Sigma };

// D_sigma from the velocity-gradient tensor gd[c][d] = d u_c / d x_d.
// Internally double: the singular values come from the invariants of
// G = g^T g via the trigonometric eigenvalue formula (Cardano), which
// cancels catastrophically in float for near-degenerate states -- the
// common case (pure shear, 2D) that must return EXACTLY the clamped zero.
// All intermediates are clamped so roundoff can never yield NaN or a
// negative viscosity; degenerate σ1 == 0 (uniform flow) returns 0.
template <class T>
DNS_HD double sigmaDsigma(const T gd[3][3])
{
    double G[3][3]; // G = g^T g, symmetric PSD
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            double s = 0;
            for (int c = 0; c < 3; ++c)
                s += double(gd[c][a]) * double(gd[c][b]);
            G[a][b] = s;
        }
    const double i1 = G[0][0] + G[1][1] + G[2][2];
    if (!(i1 > 0)) return 0.0; // uniform flow (or non-finite input)
    double trG2 = 0;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) trG2 += G[a][b] * G[b][a];
    const double i2 = 0.5 * (i1 * i1 - trG2);
    const double i3 = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1])
                    - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0])
                    + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);
    // Nicoud et al. (2011), eqs. (18)-(20).
    const double a1 = fmax(i1 * i1 / 9.0 - i2 / 3.0, 0.0);
    const double a2 = i1 * i1 * i1 / 27.0 - i1 * i2 / 6.0 + i3 / 2.0;
    const double sq = sqrt(a1);
    const double cosArg =
        sq > 0 ? fmin(fmax(a2 / (a1 * sq), -1.0), 1.0) : 1.0;
    const double a3 = acos(cosArg) / 3.0;
    constexpr double pi3 = 1.0471975511965977; // pi/3
    const double s1 = sqrt(fmax(i1 / 3.0 + 2.0 * sq * cos(a3), 0.0));
    const double s2 = sqrt(fmax(i1 / 3.0 - 2.0 * sq * cos(pi3 + a3), 0.0));
    const double s3 = sqrt(fmax(i1 / 3.0 - 2.0 * sq * cos(pi3 - a3), 0.0));
    if (!(s1 > 0)) return 0.0;
    // The trigonometric formula carries ~sqrt(machine eps) ~ 1e-8 RELATIVE
    // error in the singular values, so states with an exact sigma3 == 0 (any
    // 2D flow, pure shear, rotation) come out as sigma3 ~ 1e-8 sigma1. Those
    // states are the model's defining zeros -- floor sigma3 well above the
    // noise (genuine 3D turbulence has sigma3/sigma1 far beyond 1e-6).
    if (s3 <= 1e-6 * s1) return 0.0;
    // Ordering can be violated by roundoff only at degeneracy; each factor is
    // clamped so the operator stays a viscosity (>= 0), never an energy source.
    return s3 * fmax(s1 - s2, 0.0) * fmax(s2 - s3, 0.0) / (s1 * s1);
}

// The SGS rate that multiplies (C Delta)^2 to give nut / rho, for whichever
// model `sigma` selects:  sigma ? D_sigma(gd) : |S| = sqrt(2 S:S).
//
// Templated on T to stay bit-identical at every call site: the nut kernels
// accumulate in Real (float) and the dt-limit reducers in double, and
// T(0.5) / T(2) / sqrt(T) reproduce each verbatim. Do NOT simplify to
// double-only -- that changes the float sites' rounding, hence nut, hence dt,
// hence every trajectory.
template <class T>
DNS_HD T sgsRate(int sigma, const T gd[3][3])
{
    if (sigma) return T(sigmaDsigma(gd));
    T s2 = T(0);
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) {
            const T sij = T(0.5) * (gd[c][d] + gd[d][c]);
            s2 += T(2) * sij * sij;
        }
    return sqrt(s2);
}

} // namespace dns
