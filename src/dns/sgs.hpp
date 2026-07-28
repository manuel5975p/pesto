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

namespace detail {

// Per-precision tuning of the sigma operator, MEASURED (2026-07-26, 5e5 random
// rank-deficient tensors per row, tests/test_sgs_op.cpp reproduces it):
//   s3Floor  relative sigma3/sigma1 noise ceiling of an exactly rank-deficient
//            state. Measured max s3/s1: double 1.8e-11, float 4.0e-5 (both at
//            singular-value aspect 1e-6; at aspect 1 it is 7e-16 / 8e-8).
//            The floor sits above that and costs at most s3Floor * |g| of
//            eddy viscosity, since D_sigma <= sigma3.
//   newton   iterations of the Chebyshev-cubic solve below; one more changes
//            no bit in either precision.
template <class T> struct SigmaTune {
    static constexpr T s3Floor = T(1e-6);
    static constexpr int newton = 4;
};
template <> struct SigmaTune<float> {
    static constexpr float s3Floor = 1e-4f;
    static constexpr int newton = 3;
};

// t = cos(acos(c)/3), the root in [1/2, 1] of the Chebyshev cubic 4t^3-3t = c,
// without transcendentals. c in [-1, 1].
// Substituting t = 1/2 + w gives w^2 (6 + 4w) = 1 + c =: d, whose root w is in
// [0, 1/2]; w = sqrt(d/6) (1 - sqrt(d/6)/3) is its small-d expansion, good to
// 7% at the far end (d = 2), and Newton on 4w^3 + 6w^2 - d converges
// quadratically for every d > 0 (the double root at w = 0 is reached exactly
// by the d == 0 branch, so the iteration never sees it).
template <class T>
DNS_HD T chebRootThird(T c)
{
    const T d = T(1) + c;
    if (!(d > T(0))) return T(0.5); // c == -1: t = cos(pi/3)
    const T v = std::sqrt(d / T(6));
    T w = v * (T(1) - v / T(3));
    for (int it = 0; it < SigmaTune<T>::newton; ++it) {
        const T f = ((T(4) * w + T(6)) * w) * w - d;
        w -= f / (T(12) * w * (w + T(1)));
    }
    return T(0.5) + w;
}

} // namespace detail

// D_sigma from the velocity-gradient tensor gd[c][d] = d u_c / d x_d,
// evaluated entirely in T (float at the nut kernels, double at the dt-limit
// reducers) -- consumer GPUs run FP64 at 1/32 rate, so the operator is
// CONDITIONED rather than widened:
//   * g is scaled by amax = max |g_cd| first, so every intermediate is O(1)
//     for any input magnitude (verified over 1e+-30 in float) and no invariant
//     can over- or underflow. D_sigma is homogeneous of degree 1, so the exact
//     inverse scaling is a single multiply by amax at the end.
//   * the invariants are taken on the deviator B = G - (tr G/3) I of G = g^T g,
//     where a1 = tr(B^2)/6 is a sum of squares and a2 = det(B)/2 -- identically
//     the old i1^2/9 - i2/3 and i1^3/27 - i1 i2/6 + i3/2, but with the
//     catastrophic i1^2 - tr G^2 cancellation removed, not hidden in FP64.
//   * acos/3 and its three cosines are replaced by the algebraically identical
//     Chebyshev root t = cos(acos(c)/3) (detail::chebRootThird) plus
//     cos(pi/3 -/+ a3) = t/2 -/+ (sqrt 3 / 2) sqrt(1 - t^2).
//   * sigma3 comes from det(g)^2 = det(G) / (lam1 lam2), not from the third
//     trigonometric root: for a rank-deficient g (any 2D flow, pure shear,
//     rotation -- the model's defining zeros) det(g) is EXACTLY zero whenever
//     the missing direction is a coordinate axis, so those states return an
//     exact 0 in float, and elsewhere squaring the determinant carries the
//     sigma3 noise at eps rather than the trigonometric root's sqrt(eps).
// All intermediates are clamped so roundoff can never yield NaN or a negative
// viscosity; degenerate sigma1 == 0 (uniform flow) returns 0, and so does a
// non-finite gd or an amax below T's smallest normal.
template <class T>
DNS_HD T sigmaDsigma(const T gd[3][3])
{
    T amax = T(0);
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) amax = std::fmax(amax, std::fabs(gd[a][b]));
    if (!(amax > T(0))) return T(0); // uniform flow (or all-NaN input)
    const T r = T(1) / amax;
    T h[3][3]; // g / amax: entries in [-1, 1], at least one exactly +-1
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) h[a][b] = gd[a][b] * r;
    T G[3][3]; // G = h^T h, symmetric PSD
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) {
            T s = T(0);
            for (int c = 0; c < 3; ++c) s += h[c][a] * h[c][b];
            G[a][b] = s;
        }
    const T i1 = G[0][0] + G[1][1] + G[2][2];
    // Exact algebra puts i1 in [1, 9]; the test rejects a NaN gd and the
    // overflow of 1/amax at a subnormal amax (nut there is below rounding).
    if (!(i1 > T(0) && i1 < T(10))) return T(0);
    const T m = i1 / T(3);
    const T b0 = G[0][0] - m, b1 = G[1][1] - m, b2 = G[2][2] - m;
    // Nicoud et al. (2011), eqs. (18)-(20), in deviator form.
    const T a1 = (b0 * b0 + b1 * b1 + b2 * b2
                  + T(2) * (G[0][1] * G[0][1] + G[0][2] * G[0][2]
                            + G[1][2] * G[1][2])) / T(6);
    const T a2 = T(0.5) * (b0 * (b1 * b2 - G[1][2] * G[1][2])
                           - G[0][1] * (G[0][1] * b2 - G[1][2] * G[0][2])
                           + G[0][2] * (G[0][1] * G[1][2] - b1 * G[0][2]));
    const T sq = std::sqrt(a1);
    const T cosArg =
        sq > T(0) ? std::fmin(std::fmax(a2 / (a1 * sq), T(-1)), T(1)) : T(1);
    const T t = detail::chebRootThird(cosArg); // cos(acos(cosArg)/3)
    // sin of the same angle; 1 - t is exact for t in [1/2, 1] (Sterbenz).
    const T sn = std::sqrt(std::fmax((T(1) - t) * (T(1) + t), T(0)));
    constexpr T sqrt3 = T(1.7320508075688772);
    const T l1 = std::fmax(m + T(2) * sq * t, T(0));
    const T l2 = std::fmin(std::fmax(m - sq * t + sqrt3 * sq * sn, T(0)), l1);
    const T dg = h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
               - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
               + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0]);
    const T l1l2 = l1 * l2; // lam1 lam2 lam3 == det G == det(g/amax)^2
    const T l3 = l1l2 > T(0) ? std::fmin(dg * dg / l1l2, l2) : T(0);
    const T s1 = std::sqrt(l1), s2 = std::sqrt(l2), s3 = std::sqrt(l3);
    if (!(s1 > T(0))) return T(0);
    // A state with an exact sigma3 == 0 that is not axis-aligned still leaves
    // sigma3 at the roundoff of det(g); those states are the model's defining
    // zeros, so floor sigma3 above that noise (genuine 3D turbulence has
    // sigma3/sigma1 far beyond the floor). detail::SigmaTune records both.
    if (s3 <= detail::SigmaTune<T>::s3Floor * s1) return T(0);
    // Ordering can be violated by roundoff only at degeneracy; each factor is
    // clamped so the operator stays a viscosity (>= 0), never an energy source.
    // The trailing amax undoes the input scaling (D_sigma is degree-1
    // homogeneous), exactly when amax is a power of two.
    return amax * s3 * std::fmax(s1 - s2, T(0)) * std::fmax(s2 - s3, T(0))
           / (s1 * s1);
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
    if (sigma) return sigmaDsigma(gd);
    T s2 = T(0);
    for (int c = 0; c < 3; ++c)
        for (int d = 0; d < 3; ++d) {
            const T sij = T(0.5) * (gd[c][d] + gd[d][c]);
            s2 += T(2) * sij * sij;
        }
    return sqrt(s2);
}

} // namespace dns
