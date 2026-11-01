// Per-axis real transforms that diagonalise the cell-centred 2nd-order FD
// Laplacian under separable boundary conditions (masterplan real_bunsen S3).
// ONE definition of the eigenpairs and of the per-element pre/post steps of
// the fast (complex FFT) algorithms: the device solver (gpu::SeparablePoisson)
// instantiates them in float, the host test (`poisson_sep`) in double against
// direct sums. Host + device, no GPU types.
//
// Cells j = 0..n-1 at (j + 1/2) h, boundary on the faces j = -1/2, n - 1/2:
//   Periodic  wrap                                    DHT    cas(2 pi j k / n)
//   NN        p_-1 =  p_0,  p_n =  p_n-1 (Neumann)    DCT-II cos(pi k (j+1/2) / n)
//   DD        p_-1 = -p_0,  p_n = -p_n-1 (p = 0)      DST-II sin(pi (k+1)(j+1/2) / n)
//   ND        p_-1 =  p_0,  p_n = -p_n-1              DCT-IV cos(pi (k+1/2)(j+1/2) / n)
// Eigenvalue of -L on mode k: (2 sin(theta_k / 2) / h)^2 (sepTheta).
//
// A pass runs a complex FFT of length L = sepLineLen(kind, n) per line:
// n (DHT, Makhoul DCT-II, DST-II = the DCT-II of (-1)^j x, reversed) or n/2
// (DCT-IV, n even: z_j = (x_2j + i x_n-1-2j) W_j, X_2m = Re(U_m Z_m),
// X_n-1-2m = -Im(U_m Z_m)). Forward pass of a line x (c_k = sum_j x_j basis_k(j)):
//   v_j = sepFwdPre(kind, j, n, x, sepTwPre(kind, n, j))            j < L
//   V   = FFT_L(v), FORWARD (e^{-2 pi i j k / L}, unnormalised)
//   c  <- sepFwdPost(kind, m, n, V_m, sepTwPost(kind, n, m))        1 or 2 outputs
// Inverse pass of a coefficient line c (output y = x / sepNorm):
//   v_k = sepInvPre(kind, k, n, c, sepTwPre / sepTwPost)
//   V   = FFT_L(v), direction sepInvForward(kind) ? FORWARD : BACKWARD
//   y  <- sepInvPost(kind, j, n, V_j, sepTwPost(kind, n, j))
// The DHT is its own inverse (norm 1/n), the DCT-IV too (norm 2/n).
#pragma once

#include "dns/hd.hpp"

#include <cmath>
#include <cstdint>
#include <numbers>

namespace dns {

enum class SepKind : std::uint8_t { Periodic, NN, DD, ND };

inline const char* sepKindName(SepKind k)
{
    switch (k)
    {
    case SepKind::Periodic: return "periodic";
    case SepKind::NN: return "NN";
    case SepKind::DD: return "DD";
    case SepKind::ND: return "ND";
    }
    return "?";
}

// Dirichlet (p = 0, antimirror ghost) on the lo / hi face of the line.
DNS_HD inline bool sepLoDirichlet(SepKind k) { return k == SepKind::DD; }
DNS_HD inline bool sepHiDirichlet(SepKind k)
{
    return k == SepKind::DD || k == SepKind::ND;
}
// The axis owns a zero eigenvalue (the constant mode): Periodic, NN.
DNS_HD inline bool sepHasZeroMode(SepKind k)
{
    return k == SepKind::Periodic || k == SepKind::NN;
}

// Complex FFT length of one line. Pre: n even for ND.
DNS_HD inline int sepLineLen(SepKind kind, int n)
{
    return kind == SepKind::ND ? n / 2 : n;
}

// theta_k of mode k (eigenvalue (2 sin(theta/2)/h)^2). Pre: 0 <= k < n.
inline double sepTheta(SepKind kind, int n, int k)
{
    using std::numbers::pi;
    switch (kind)
    {
    case SepKind::Periodic: return 2.0 * pi * k / n;
    case SepKind::NN: return pi * k / n;
    case SepKind::DD: return pi * (k + 1) / n;
    case SepKind::ND: return pi * (k + 0.5) / n;
    }
    return 0.0;
}

// Eigenvalue of -L (positive semi-definite) on mode k, FP64.
inline double sepEigen(SepKind kind, int n, double h, int k)
{
    const double s = 2.0 * std::sin(0.5 * sepTheta(kind, n, k)) / h;
    return s * s;
}

// Mode k at cell j: the forward transform's kernel (unnormalised).
inline double sepBasis(SepKind kind, int n, int k, int j)
{
    using std::numbers::pi;
    const double jc = j + 0.5;
    switch (kind)
    {
    case SepKind::Periodic:
    {
        const double a = 2.0 * pi * double((long(j) * k) % n) / n;
        return std::cos(a) + std::sin(a);
    }
    case SepKind::NN: return std::cos(pi * k * jc / n);
    case SepKind::DD: return std::sin(pi * (k + 1) * jc / n);
    case SepKind::ND: return std::cos(pi * (k + 0.5) * jc / n);
    }
    return 0.0;
}

// inverse(forward(x)) = x / sepNorm: the factor the spectral scale applies.
inline double sepNorm(SepKind kind, int n)
{
    return kind == SepKind::ND ? 2.0 / n : 1.0 / n;
}

template <class T>
struct SepC
{
    T re, im;
};

template <class T>
DNS_HD inline SepC<T> sepMul(SepC<T> a, SepC<T> b)
{
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

// Pre-FFT twiddle (ND: W_j = e^{-i pi (4j+1) / (4n)}; others 1) and post-FFT
// twiddle (NN, DD: e^{-i pi m / (2n)}; ND: U_m = e^{-i pi m / n}; DHT 1), FP64.
inline SepC<double> sepTwPre(SepKind kind, int n, int j)
{
    if (kind != SepKind::ND) return {1.0, 0.0};
    const double t = std::numbers::pi * (4.0 * j + 1.0) / (4.0 * n);
    return {std::cos(t), -std::sin(t)};
}
inline SepC<double> sepTwPost(SepKind kind, int n, int m)
{
    double t = 0.0;
    if (kind == SepKind::NN || kind == SepKind::DD) t = std::numbers::pi * m / (2.0 * n);
    if (kind == SepKind::ND) t = std::numbers::pi * m / n;
    return {std::cos(t), -std::sin(t)};
}

// Even-then-reversed-odd reordering (Makhoul), the NN / DD line permutation.
DNS_HD inline int sepEvenOdd(int j, int n)
{
    return j < (n + 1) / 2 ? 2 * j : 2 * (n - 1 - j) + 1;
}

// x(i) reads entry i of the line. Element j < sepLineLen.
template <class T, class Get>
DNS_HD inline SepC<T> sepFwdPre(SepKind kind, int j, int n, Get x, SepC<T> w)
{
    switch (kind)
    {
    case SepKind::NN: return {x(sepEvenOdd(j, n)), T(0)};
    case SepKind::DD:
    {
        const int s = sepEvenOdd(j, n);
        const T v = x(s);
        return {(s & 1) ? -v : v, T(0)};
    }
    case SepKind::ND: return sepMul(SepC<T>{x(2 * j), x(n - 1 - 2 * j)}, w);
    default: return {x(j), T(0)};
    }
}

// One FFT output element becomes one (d1 < 0) or two line entries.
template <class T>
struct SepOut
{
    int d0;
    T v0;
    int d1;
    T v1;
};

template <class T>
DNS_HD inline SepOut<T> sepFwdPost(SepKind kind, int m, int n, SepC<T> V, SepC<T> t)
{
    switch (kind)
    {
    case SepKind::NN: return {m, t.re * V.re - t.im * V.im, -1, T(0)};
    case SepKind::DD: return {n - 1 - m, t.re * V.re - t.im * V.im, -1, T(0)};
    case SepKind::ND:
    {
        const SepC<T> y = sepMul(t, V);
        return {2 * m, y.re, n - 1 - 2 * m, -y.im};
    }
    default: return {m, V.re - V.im, -1, T(0)};
    }
}

// c(i) reads coefficient i of the line. NN / DD: Makhoul's inverse twiddle
// conj(t_k) (C_k - i C_{n-k}), C_n := 0 (DD: C_k = c_{n-1-k}). ND: the
// forward pre (w = sepTwPre); others: t = sepTwPost.
template <class T, class Get>
DNS_HD inline SepC<T> sepInvPre(SepKind kind, int k, int n, Get c, SepC<T> w, SepC<T> t)
{
    switch (kind)
    {
    case SepKind::Periodic: return {c(k), T(0)};
    case SepKind::ND: return sepFwdPre(kind, k, n, c, w);
    default:
    {
        const bool dd = kind == SepKind::DD;
        const T ck = c(dd ? n - 1 - k : k);
        const T cn = k == 0 ? T(0) : c(dd ? k - 1 : n - k);
        return {t.re * ck - t.im * cn, -t.im * ck - t.re * cn};
    }
    }
}

DNS_HD inline bool sepInvForward(SepKind kind)
{
    return kind == SepKind::Periodic || kind == SepKind::ND;
}

template <class T>
DNS_HD inline SepOut<T> sepInvPost(SepKind kind, int j, int n, SepC<T> v, SepC<T> t)
{
    switch (kind)
    {
    case SepKind::NN: return {sepEvenOdd(j, n), v.re, -1, T(0)};
    case SepKind::DD:
    {
        const int d = sepEvenOdd(j, n);
        return {d, (d & 1) ? -v.re : v.re, -1, T(0)};
    }
    case SepKind::ND: return sepFwdPost(kind, j, n, v, t);
    default: return {j, v.re - v.im, -1, T(0)};
    }
}

// Two lines per complex FFT (the packing that halves FFT and spectrum traffic):
//   RealIn   the pre values are REAL (DHT both ways, DCT-II / DST-II forward):
//            FFT(a + i b) = Z; line a's spectrum (Z_k + conj Z_{L-k}) / 2,
//            line b's (Z_k - conj Z_{L-k}) / (2i)  (sepSplit, Z_L := Z_0);
//   RealOut  the FFT OUTPUT is real (Makhoul's inverse: Hermitian V):
//            FFT(Va + i Vb) = va + i vb (sepJoin, then re / im);
//   Single   the DCT-IV already fills the complex FFT (x_2j + i x_n-1-2j).
enum class SepPack : std::uint8_t { Single, RealIn, RealOut };

DNS_HD inline SepPack sepPack(SepKind kind, bool inverse)
{
    if (kind == SepKind::ND) return SepPack::Single;
    if (kind == SepKind::Periodic || !inverse) return SepPack::RealIn;
    return SepPack::RealOut;
}

template <class T>
DNS_HD inline SepC<T> sepSplit(SepC<T> zk, SepC<T> zl, bool second)
{
    if (!second) return {T(0.5) * (zk.re + zl.re), T(0.5) * (zk.im - zl.im)};
    return {T(0.5) * (zk.im + zl.im), T(0.5) * (zl.re - zk.re)};
}

template <class T>
DNS_HD inline SepC<T> sepJoin(SepC<T> va, SepC<T> vb)
{
    return {va.re - vb.im, va.im + vb.re};
}

} // namespace dns
