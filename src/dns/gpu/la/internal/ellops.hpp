// Device helpers of gpu/la: the compiled-shape dispatch, the logical block of
// an ELL entry, and a small dense FP64 inverse. INTERNAL (.hip TUs only).
#pragma once

#include "dns/gpu/la/blockell.hpp"

#include <cmath>

namespace dns::gpu::la::internal {

template <int B_, int Bd_>
struct Shape
{
    static constexpr int B = B_, Bd = Bd_;
};

// Calls f(Shape<B, Bd>{}) for a compiled shape; false (nothing called) else.
// The list is dns::la::blockShape's: B in {1,3,5,6,7,9,11}, Bd == B or 5.
template <class F>
bool withShape(const dns::la::BlockShape& s, F&& f)
{
    switch (s.B * 16 + s.Bd) {
    case 1 * 16 + 1: f(Shape<1, 1>{}); return true;
    case 3 * 16 + 3: f(Shape<3, 3>{}); return true;
    case 5 * 16 + 5: f(Shape<5, 5>{}); return true;
    case 6 * 16 + 6: f(Shape<6, 6>{}); return true;
    case 6 * 16 + 5: f(Shape<6, 5>{}); return true;
    case 7 * 16 + 7: f(Shape<7, 7>{}); return true;
    case 7 * 16 + 5: f(Shape<7, 5>{}); return true;
    case 9 * 16 + 9: f(Shape<9, 9>{}); return true;
    case 9 * 16 + 5: f(Shape<9, 5>{}); return true;
    case 11 * 16 + 11: f(Shape<11, 11>{}); return true;
    case 11 * 16 + 5: f(Shape<11, 5>{}); return true;
    default: return false;
    }
}

// The logical B x B block of entry E (dense part + scalar diagonal), FP64.
template <int B, int Bd, class T>
__device__ inline void offBlock(const EllView<T>& A, int E, double (&M)[B][B])
{
    const std::size_t nE = std::size_t(A.nE), e = std::size_t(E);
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) M[r][q] = 0.0;
#pragma unroll
    for (int r = 0; r < Bd; ++r)
#pragma unroll
        for (int q = 0; q < Bd; ++q) M[r][q] = double(A.off[std::size_t(r * Bd + q) * nE + e]);
#pragma unroll
    for (int r = Bd; r < B; ++r) M[r][r] = double(A.sc[std::size_t(A.shape.nS == 1 ? 0 : r - Bd) * nE + e]);
}

template <int B, class T>
__device__ inline void diagBlock(const EllView<T>& A, int c, double (&M)[B][B])
{
    const std::size_t n = std::size_t(A.n);
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) M[r][q] = double(A.diag[std::size_t(r * B + q) * n + std::size_t(c)]);
}

// X = M^-1 by Gauss-Jordan with partial pivoting; M is destroyed. False on a
// zero or non-finite pivot.
template <int B>
__device__ inline bool invert(double (&M)[B][B], double (&X)[B][B])
{
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) X[r][q] = r == q ? 1.0 : 0.0;
    for (int k = 0; k < B; ++k) {
        int p = k;
        double best = fabs(M[k][k]);
        for (int r = k + 1; r < B; ++r)
            if (fabs(M[r][k]) > best) {
                best = fabs(M[r][k]);
                p = r;
            }
        if (!(best > 0.0) || !isfinite(best)) return false;
        if (p != k)
            for (int q = 0; q < B; ++q) {
                const double t = M[k][q];
                M[k][q] = M[p][q];
                M[p][q] = t;
                const double u = X[k][q];
                X[k][q] = X[p][q];
                X[p][q] = u;
            }
        const double inv = 1.0 / M[k][k];
        for (int q = 0; q < B; ++q) {
            M[k][q] *= inv;
            X[k][q] *= inv;
        }
        for (int r = 0; r < B; ++r) {
            if (r == k) continue;
            const double f = M[r][k];
            if (f == 0.0) continue;
            for (int q = 0; q < B; ++q) {
                M[r][q] -= f * M[k][q];
                X[r][q] -= f * X[k][q];
            }
        }
    }
    return true;
}

// Row c's off-diagonal ELL walk: acc[r] += A_cj x_j (Neg: -=) over the entries
// in ascending slot order, the first padding entry (col < 0) ending the row.
// B == 1 PREFETCHES (masterplan_performance step 1): kPrefetch column loads,
// then the predicated value and x loads, then the same FMA chain in the same
// order -- the dependent col -> x loads no longer serialize, the bits do not
// change. Every load stays inside the slice (width w), so padding is in bounds.
inline constexpr int kPrefetch = 8;

template <int B, int Bd, bool Neg, class T>
__device__ inline void ellRow(const EllView<T>& A, int c, const T* x, T (&acc)[B])
{
    const std::size_t n = std::size_t(A.n), nE = std::size_t(A.nE);
    const int sl = c / dns::la::kSliceRows, lane = c % dns::la::kSliceRows;
    const int w = A.sliceWidth[sl];
    int E = A.sliceBase[sl] + lane;
    if constexpr (B == 1) {
        for (int k0 = 0; k0 < w; k0 += kPrefetch, E += kPrefetch * dns::la::kSliceRows) {
            int j[kPrefetch];
            T a[kPrefetch], xv[kPrefetch];
#pragma unroll
            for (int u = 0; u < kPrefetch; ++u) j[u] = k0 + u < w ? A.col[E + u * dns::la::kSliceRows] : -1;
#pragma unroll
            for (int u = 0; u < kPrefetch; ++u) {
                a[u] = j[u] >= 0 ? A.off[std::size_t(E + u * dns::la::kSliceRows)] : T(0);
                xv[u] = j[u] >= 0 ? x[std::size_t(j[u])] : T(0);
            }
#pragma unroll
            for (int u = 0; u < kPrefetch; ++u) {
                if (j[u] < 0) return;
                if constexpr (Neg) acc[0] -= a[u] * xv[u];
                else acc[0] += a[u] * xv[u];
            }
        }
    } else {
        for (int k = 0; k < w; ++k, E += dns::la::kSliceRows) {
            const int j = A.col[E];
            if (j < 0) break;
            T xj[B];
#pragma unroll
            for (int q = 0; q < B; ++q) xj[q] = x[std::size_t(q) * n + std::size_t(j)];
#pragma unroll
            for (int r = 0; r < Bd; ++r)
#pragma unroll
                for (int q = 0; q < Bd; ++q) {
                    if constexpr (Neg) acc[r] -= A.off[std::size_t(r * Bd + q) * nE + std::size_t(E)] * xj[q];
                    else acc[r] += A.off[std::size_t(r * Bd + q) * nE + std::size_t(E)] * xj[q];
                }
#pragma unroll
            for (int r = Bd; r < B; ++r) {
                const T s = A.sc[std::size_t(A.shape.nS == 1 ? 0 : r - Bd) * nE + std::size_t(E)];
                if constexpr (Neg) acc[r] -= s * xj[r];
                else acc[r] += s * xj[r];
            }
        }
    }
}

// ellRow<1, 1, Neg> for nb systems sharing the off-diagonals (system q:
// x + q xs, acc[q]; only q in mask): each system's FMA chain is ellRow's, the
// column / value loads are made once.
template <bool Neg, class T>
__device__ inline void ellRowMulti(const EllView<T>& A, int c, int nb, unsigned mask, const T* x, std::size_t xs,
                                   T (&acc)[3])
{
    const int sl = c / dns::la::kSliceRows, lane = c % dns::la::kSliceRows;
    const int w = A.sliceWidth[sl];
    int E = A.sliceBase[sl] + lane;
    for (int k0 = 0; k0 < w; k0 += kPrefetch, E += kPrefetch * dns::la::kSliceRows) {
        int j[kPrefetch];
        T a[kPrefetch], xv[3][kPrefetch];
#pragma unroll
        for (int u = 0; u < kPrefetch; ++u) j[u] = k0 + u < w ? A.col[E + u * dns::la::kSliceRows] : -1;
#pragma unroll
        for (int u = 0; u < kPrefetch; ++u) {
            a[u] = j[u] >= 0 ? A.off[std::size_t(E + u * dns::la::kSliceRows)] : T(0);
#pragma unroll
            for (int q = 0; q < 3; ++q)
                xv[q][u] = j[u] >= 0 && q < nb && ((mask >> q) & 1u) ? x[std::size_t(q) * xs + std::size_t(j[u])] : T(0);
        }
#pragma unroll
        for (int u = 0; u < kPrefetch; ++u) {
            if (j[u] < 0) return;
#pragma unroll
            for (int q = 0; q < 3; ++q) {
                if constexpr (Neg) acc[q] -= a[u] * xv[q][u];
                else acc[q] += a[u] * xv[q][u];
            }
        }
    }
}

// acc = D_c x_c + sum_j A_cj x_j: kSpmv's row, the expression order of every
// SpMV-shaped kernel (residual, fused Jacobi sweep).
template <int B, int Bd, class T>
__device__ inline void spmvRow(const EllView<T>& A, int c, const T* x, T (&acc)[B])
{
    const std::size_t n = std::size_t(A.n);
    T xc[B];
#pragma unroll
    for (int q = 0; q < B; ++q) xc[q] = x[std::size_t(q) * n + std::size_t(c)];
#pragma unroll
    for (int r = 0; r < B; ++r) {
        T s = T(0);
#pragma unroll
        for (int q = 0; q < B; ++q) s += A.diag[std::size_t(r * B + q) * n + std::size_t(c)] * xc[q];
        acc[r] = s;
    }
    ellRow<B, Bd, false>(A, c, x, acc);
}

// The launch grid of n threads at the library's 256-thread blocks.
inline unsigned gridFor(std::size_t n) { return unsigned((n + 255) / 256); }
inline constexpr int kThreads = 256;

} // namespace dns::gpu::la::internal
