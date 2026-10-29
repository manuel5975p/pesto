#include "dns/gpu/la/amg.hpp"

#include "dns/gpu/la/internal/dots.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <optional>

namespace dns::gpu::la {

namespace {

using internal::diagBlock;
using internal::gridFor;
using internal::kThreads;
using internal::offBlock;

std::string hipText(const char* what, hipError_t e) { return std::string(what) + ": " + hipGetErrorString(e); }

template <class U>
std::expected<DeviceBuffer<U>, std::string> devVec(const std::vector<U>& h, const char* what)
{
    auto b = DeviceBuffer<U>::make(std::max<std::size_t>(1, h.size()));
    if (!b) return std::unexpected(hipText(what, b.error()));
    if (!h.empty())
        if (const hipError_t e = hipMemcpy(b->data(), h.data(), h.size() * sizeof(U), hipMemcpyHostToDevice);
            e != hipSuccess)
            return std::unexpected(hipText(what, e));
    return std::move(*b);
}

template <class U>
std::expected<DeviceBuffer<U>, std::string> devZero(std::size_t n, const char* what)
{
    auto b = DeviceBuffer<U>::make(std::max<std::size_t>(1, n));
    if (!b) return std::unexpected(hipText(what, b.error()));
    if (const hipError_t e = hipMemset(b->data(), 0, b->size() * sizeof(U)); e != hipSuccess)
        return std::unexpected(hipText(what, e));
    return std::move(*b);
}

// ---- refresh: coarse = P^T A P as fixed-order FP64 sums --------------------

// Coarse diagonal block of row I: the members' diagonal blocks ascending, then
// the logical off blocks of the slots inside the aggregate ascending (per
// entry, the order of dns::la::amgGalerkin).
template <int B, int Bd, class T>
__global__ void kGatherDiag(int nc, EllView<T> fine, const int* __restrict__ memberPtr,
                            const int* __restrict__ member, const int* __restrict__ intraPtr,
                            const int* __restrict__ intraE, T* cdiag)
{
    const int I = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (I >= nc) return;
    const std::size_t nf = std::size_t(fine.n);
    double acc[B][B];
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) acc[r][q] = 0.0;
    for (int k = memberPtr[I]; k < memberPtr[I + 1]; ++k) {
        const std::size_t i = std::size_t(member[k]);
#pragma unroll
        for (int r = 0; r < B; ++r)
#pragma unroll
            for (int q = 0; q < B; ++q) acc[r][q] += double(fine.diag[std::size_t(r * B + q) * nf + i]);
    }
    for (int k = intraPtr[I]; k < intraPtr[I + 1]; ++k) {
        double M[B][B];
        offBlock<B, Bd>(fine, intraE[k], M);
#pragma unroll
        for (int r = 0; r < B; ++r)
#pragma unroll
            for (int q = 0; q < B; ++q) acc[r][q] += M[r][q];
    }
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) cdiag[std::size_t(r * B + q) * std::size_t(nc) + std::size_t(I)] = T(acc[r][q]);
}

// Coarse slot s (entry dst[s]): the sum of its fine entries, dense part and
// scalars separately.
template <class T>
__global__ void kGatherOff(int nSlots, int nDense, int nS, const int* __restrict__ offPtr,
                           const int* __restrict__ offE, const int* __restrict__ dst, const T* __restrict__ fOff,
                           const T* __restrict__ fSc, std::size_t nEf, T* cOff, T* cSc, std::size_t nEc)
{
    const int s = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (s >= nSlots) return;
    const std::size_t E = std::size_t(dst[s]);
    for (int e = 0; e < nDense; ++e) {
        double acc = 0.0;
        for (int k = offPtr[s]; k < offPtr[s + 1]; ++k) acc += double(fOff[std::size_t(e) * nEf + std::size_t(offE[k])]);
        cOff[std::size_t(e) * nEc + E] = T(acc);
    }
    for (int e = 0; e < nS; ++e) {
        double acc = 0.0;
        for (int k = offPtr[s]; k < offPtr[s + 1]; ++k) acc += double(fSc[std::size_t(e) * nEf + std::size_t(offE[k])]);
        cSc[std::size_t(e) * nEc + E] = T(acc);
    }
}

// ---- transfer ---------------------------------------------------------------

// rc = P^T r: members ascending, FP64.
template <class T>
__global__ void kRestrict(int nc, int B, int nf, const int* __restrict__ memberPtr, const int* __restrict__ member,
                          const T* __restrict__ r, T* __restrict__ rc)
{
    const int I = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (I >= nc) return;
    for (int q = 0; q < B; ++q) {
        double s = 0.0;
        for (int k = memberPtr[I]; k < memberPtr[I + 1]; ++k)
            s += double(r[std::size_t(q) * std::size_t(nf) + std::size_t(member[k])]);
        rc[std::size_t(q) * std::size_t(nc) + std::size_t(I)] = T(s);
    }
}

// x += P e.
template <class T>
__global__ void kProlongAdd(int nf, int B, int nc, const int* __restrict__ agg, const T* __restrict__ e,
                            T* __restrict__ x)
{
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= nf) return;
    const std::size_t I = std::size_t(agg[i]);
    for (int q = 0; q < B; ++q)
        x[std::size_t(q) * std::size_t(nf) + std::size_t(i)] += e[std::size_t(q) * std::size_t(nc) + I];
}

// x += P e with e = T(c[0] c1) (the scaled-V correction, inner 1): kScaled's
// value, rounded to T, then kProlongAdd's add -- two roundings as in the two-
// kernel form: contraction OFF for clang (HIP), __dmul_rn for nvcc (CUDA shim
// builds, whose --fmad would otherwise fuse the product into the add).
template <class T>
__global__ void kProlongAddScaled(int nf, int B, int nc, const int* __restrict__ agg, const double* __restrict__ c,
                                  const T* __restrict__ c1, T* __restrict__ x)
{
#pragma clang fp contract(off)
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= nf) return;
    const std::size_t I = std::size_t(agg[i]);
    const double a = c[0];
    for (int q = 0; q < B; ++q) {
        const T e = T(__dmul_rn(a, double(c1[std::size_t(q) * std::size_t(nc) + I])));
        x[std::size_t(q) * std::size_t(nf) + std::size_t(i)] += e;
    }
}

// ---- inner Krylov steps (coefficients stay on the device) -------------------

// s = [(c1, v1), (c1, b), (d2, v1), (d2, v2), (d2, rt)]; coef = [a1, c1, d2].
__global__ void kCoef1(const double* s, double* coef)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const double a = s[0] > 0.0 ? s[1] / s[0] : 0.0;
    coef[0] = isfinite(a) ? a : 0.0;
}

// The two-step K-cycle combination (Notay 2008, Algorithm 2.1); a non-
// positive or non-finite rho2 falls back to the one-step line search.
__global__ void kCoef2(const double* s, double* coef)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const double rho1 = s[0], al1 = s[1], gam = s[2], bet = s[3], al2 = s[4];
    coef[1] = coef[0];
    coef[2] = 0.0;
    if (!(rho1 > 0.0)) return;
    const double rho2 = bet - gam * gam / rho1;
    if (!(rho2 > 0.0) || !isfinite(rho2)) return;
    const double c1 = al1 / rho1 - gam * al2 / (rho1 * rho2), c2 = al2 / rho2;
    if (!isfinite(c1) || !isfinite(c2)) return;
    coef[1] = c1;
    coef[2] = c2;
}

// out = b - c[0] v
template <class T>
__global__ void kSubScaled(std::size_t N, const T* b, const double* c, const T* v, T* out)
{
    const std::size_t i = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (i < N) out[i] = T(double(b[i]) - c[0] * double(v[i]));
}

// out = c[1] x + c[2] y
template <class T>
__global__ void kComb2(std::size_t N, const double* c, const T* x, const T* y, T* out)
{
    const std::size_t i = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (i < N) out[i] = T(c[1] * double(x[i]) + c[2] * double(y[i]));
}

// y = x - (per-component mean of x); sum[q] = sum_c x[q n + c]. y may alias x.
template <class T>
__global__ void kRemoveMean(int n, int B, const double* sum, const T* x, T* y)
{
    const std::size_t i = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= std::size_t(n) * std::size_t(B)) return;
    y[i] = T(double(x[i]) - sum[i / std::size_t(n)] / double(n));
}

// ---- coarsest: dense FP64 inverse --------------------------------------------

// M (m x m row-major, unknown q n + c) from the coarsest ELL matrix. Pre: M zeroed.
template <int B, int Bd, class T>
__global__ void kDenseBuild(EllView<T> A, double* M)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= A.n) return;
    const std::size_t n = std::size_t(A.n), m = n * B;
    double D[B][B];
    diagBlock<B>(A, c, D);
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) M[(std::size_t(r) * n + std::size_t(c)) * m + std::size_t(q) * n + std::size_t(c)] = D[r][q];
    const int sl = c / dns::la::kSliceRows, lane = c % dns::la::kSliceRows;
    int E = A.sliceBase[sl] + lane;
    for (int k = 0; k < A.sliceWidth[sl]; ++k, E += dns::la::kSliceRows) {
        const int j = A.col[E];
        if (j < 0) break;
        offBlock<B, Bd>(A, E, D);
#pragma unroll
        for (int r = 0; r < B; ++r)
#pragma unroll
            for (int q = 0; q < B; ++q)
                M[(std::size_t(r) * n + std::size_t(c)) * m + std::size_t(q) * n + std::size_t(j)] = D[r][q];
    }
}

constexpr int kDenseThreads = 1024;

// Gauss-Jordan with partial pivoting in ONE block (fixed order: pivot = max
// |M[r][k]|, ties to the lower row). nullSpace: first M += alpha_q 1 1^T on
// each component's block, alpha_q = mean diagonal / nc. X = M^-1 (scratch),
// Xt = X column-major in T for the apply. bad[0] = 1 on a pivot below
// 1e-14 x max |diag| or a non-finite inverse.
template <class T>
__global__ void __launch_bounds__(kDenseThreads) kDenseInvert(int m, int B, int nullSpace, double* M, double* X,
                                                              T* Xt, int* bad)
{
    __shared__ double sv[kDenseThreads];
    __shared__ int si[kDenseThreads];
    __shared__ double f[dns::la::kAmgMaxCoarseUnknowns];
    __shared__ double alpha[16];
    __shared__ double scale;
    __shared__ int fail;
    const int t = int(threadIdx.x), nt = int(blockDim.x);
    const std::size_t M1 = std::size_t(m);
    const int nc = m / B;
    if (t == 0) {
        double dmax = 0;
        for (int u = 0; u < m; ++u) dmax = fmax(dmax, fabs(M[std::size_t(u) * M1 + std::size_t(u)]));
        scale = dmax;
        fail = 0;
        for (int q = 0; q < B; ++q) {
            double s = 0;
            for (int c = 0; c < nc; ++c) {
                const std::size_t u = std::size_t(q) * std::size_t(nc) + std::size_t(c);
                s += M[u * M1 + u];
            }
            alpha[q] = nullSpace ? s / double(nc) / double(nc) : 0.0;
        }
    }
    __syncthreads();
    for (std::size_t idx = std::size_t(t); idx < M1 * M1; idx += std::size_t(nt)) {
        const int r = int(idx / M1), q = int(idx % M1);
        if (nullSpace && r / nc == q / nc) M[idx] += alpha[r / nc];
        X[idx] = r == q ? 1.0 : 0.0;
    }
    __syncthreads();
    for (int k = 0; k < m; ++k) {
        double best = -1;
        int bi = m;
        for (int r = k + t; r < m; r += nt) {
            const double v = fabs(M[std::size_t(r) * M1 + std::size_t(k)]);
            if (v > best) {
                best = v;
                bi = r;
            }
        }
        sv[t] = best;
        si[t] = bi;
        __syncthreads();
        for (int w = nt / 2; w > 0; w >>= 1) {
            if (t < w && (sv[t + w] > sv[t] || (sv[t + w] == sv[t] && si[t + w] < si[t]))) {
                sv[t] = sv[t + w];
                si[t] = si[t + w];
            }
            __syncthreads();
        }
        const int p = si[0];
        if (t == 0 && (!(sv[0] > 1e-14 * scale) || !isfinite(sv[0]))) fail = 1;
        __syncthreads();
        if (fail) break;
        if (p != k)
            for (int q = t; q < m; q += nt) {
                const std::size_t a = std::size_t(k) * M1 + std::size_t(q), b = std::size_t(p) * M1 + std::size_t(q);
                const double u = M[a], v = X[a];
                M[a] = M[b];
                X[a] = X[b];
                M[b] = u;
                X[b] = v;
            }
        __syncthreads();
        const double inv = 1.0 / M[std::size_t(k) * M1 + std::size_t(k)];
        __syncthreads();
        for (int q = t; q < m; q += nt) {
            M[std::size_t(k) * M1 + std::size_t(q)] *= inv;
            X[std::size_t(k) * M1 + std::size_t(q)] *= inv;
        }
        for (int r = t; r < m; r += nt) f[r] = r == k ? 0.0 : M[std::size_t(r) * M1 + std::size_t(k)];
        __syncthreads();
        for (std::size_t idx = std::size_t(t); idx < M1 * M1; idx += std::size_t(nt)) {
            const std::size_t r = idx / M1, q = idx % M1;
            const double fr = f[r];
            if (fr == 0.0) continue;
            if (q >= std::size_t(k)) M[idx] -= fr * M[std::size_t(k) * M1 + q];
            X[idx] -= fr * X[std::size_t(k) * M1 + q];
        }
        __syncthreads();
    }
    if (fail) {
        if (t == 0) bad[0] = 1;
        return;
    }
    int nonFinite = 0;
    for (std::size_t idx = std::size_t(t); idx < M1 * M1; idx += std::size_t(nt)) {
        const std::size_t r = idx / M1, q = idx % M1;
        const T v = T(X[idx]);
        nonFinite |= !isfinite(double(v));
        Xt[q * M1 + r] = v;
    }
    if (nonFinite) bad[0] = 1; // same value from every writer
}

// kDenseInvert for m <= kDenseRegMax with [M | X] RESIDENT IN REGISTERS
// (masterplan_performance step 5): thread t owns column t / 4 of [M | X]
// (2m <= 256 columns) at rows t % 4 + 4 i, i < 32; the pivot column and the
// scaled pivot row go through LDS. Every element sees the same operations in
// the same order as kDenseInvert (same pivot: max |M[r][k]| then the lowest
// row is a total order, so the reduction shape cannot change it; same swap,
// inv, row scale, f, `fr == 0` skip, `q >= k` rule, fail test and Xt store),
// so the outputs (Xt, bad) are the same bits; the global read-modify-write per
// pivot step is gone (kDenseInvert: ~0.9 ms at m = 114).
constexpr int kDenseRegMax = 128, kDenseRegRows = 32; // 4 threads per column, 32 rows each
static_assert(kDenseThreads == 4 * 2 * kDenseRegMax && 4 * kDenseRegRows == kDenseRegMax);

template <class T>
__global__ void __launch_bounds__(kDenseThreads) kDenseInvertReg(int m, int B, int nullSpace, const double* M, T* Xt,
                                                                 int* bad)
{
    constexpr int R = kDenseRegRows, P = 32; // P: pivot-search threads
    __shared__ double colk[kDenseRegMax], rowk[2 * kDenseRegMax], rowp[2 * kDenseRegMax];
    __shared__ double sv[P];
    __shared__ int si[P];
    __shared__ double alpha[16];
    __shared__ double scale, sinv;
    __shared__ int fail, piv;
    const int t = int(threadIdx.x), col = t >> 2, sub = t & 3;
    const std::size_t M1 = std::size_t(m);
    const int nc = m / B;
    const bool active = col < 2 * m, inM = col < m;
    if (t == 0) { // verbatim: the scale and the null-space shifts, sequential
        double dmax = 0;
        for (int u = 0; u < m; ++u) dmax = fmax(dmax, fabs(M[std::size_t(u) * M1 + std::size_t(u)]));
        scale = dmax;
        fail = 0;
        for (int q = 0; q < B; ++q) {
            double s = 0;
            for (int c = 0; c < nc; ++c) {
                const std::size_t u = std::size_t(q) * std::size_t(nc) + std::size_t(c);
                s += M[u * M1 + u];
            }
            alpha[q] = nullSpace ? s / double(nc) / double(nc) : 0.0;
        }
    }
    __syncthreads();
    double a[R];
#pragma unroll
    for (int i = 0; i < R; ++i) {
        const int r = sub + 4 * i;
        a[i] = 0.0;
        if (!active || r >= m) continue;
        if (inM) {
            double v = M[std::size_t(r) * M1 + std::size_t(col)];
            if (nullSpace && r / nc == col / nc) v += alpha[r / nc];
            a[i] = v;
        } else a[i] = r == col - m ? 1.0 : 0.0;
    }
    for (int k = 0; k < m; ++k) {
        if (col == k)
#pragma unroll
            for (int i = 0; i < R; ++i)
                if (sub + 4 * i < m) colk[sub + 4 * i] = a[i];
        __syncthreads();
        if (t < P) { // rows >= k, strided; NaN never a candidate (as kDenseInvert's `v > best` from -1)
            double best = -1;
            int bi = m;
            for (int r = k + t; r < m; r += P) {
                const double v = fabs(colk[r]);
                if (v > best) {
                    best = v;
                    bi = r;
                }
            }
            sv[t] = best;
            si[t] = bi;
        }
        __syncthreads();
        for (int w = P / 2; w > 0; w >>= 1) {
            if (t < w && (sv[t + w] > sv[t] || (sv[t + w] == sv[t] && si[t + w] < si[t]))) {
                sv[t] = sv[t + w];
                si[t] = si[t + w];
            }
            __syncthreads();
        }
        if (t == 0) {
            piv = si[0];
            if (!(sv[0] > 1e-14 * scale) || !isfinite(sv[0])) fail = 1;
        }
        __syncthreads();
        if (fail) break;
        const int p = piv, ik = k >> 2, ip = p >> 2;
        if (p != k) { // swap rows k and p of [M | X] through LDS
            if (active && sub == (k & 3))
#pragma unroll
                for (int i = 0; i < R; ++i)
                    if (i == ik) rowk[col] = a[i];
            if (active && sub == (p & 3))
#pragma unroll
                for (int i = 0; i < R; ++i)
                    if (i == ip) rowp[col] = a[i];
            __syncthreads();
            if (active && sub == (k & 3))
#pragma unroll
                for (int i = 0; i < R; ++i)
                    if (i == ik) a[i] = rowp[col];
            if (active && sub == (p & 3))
#pragma unroll
                for (int i = 0; i < R; ++i)
                    if (i == ip) a[i] = rowk[col];
        }
        if (col == k && sub == (k & 3))
#pragma unroll
            for (int i = 0; i < R; ++i)
                if (i == ik) sinv = 1.0 / a[i];
        __syncthreads();
        const double inv = sinv;
        if (active && sub == (k & 3)) // row k scaled, then the pivot row to LDS
#pragma unroll
            for (int i = 0; i < R; ++i)
                if (i == ik) {
                    a[i] *= inv;
                    rowk[col] = a[i];
                }
        if (col == k) // f: column k of the other rows
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int r = sub + 4 * i;
                if (r < m) colk[r] = r == k ? 0.0 : a[i];
            }
        __syncthreads();
        if (active) {
            const double pk = rowk[col];
#pragma unroll
            for (int i = 0; i < R; ++i) {
                const int r = sub + 4 * i;
                if (r >= m) continue;
                const double fr = colk[r];
                if (fr == 0.0) continue;
                if (!inM || col >= k) a[i] -= fr * pk;
            }
        }
        __syncthreads();
    }
    if (fail) {
        if (t == 0) bad[0] = 1;
        return;
    }
    int nonFinite = 0;
    if (active && !inM) {
        const std::size_t q = std::size_t(col - m);
#pragma unroll
        for (int i = 0; i < R; ++i) {
            const int r = sub + 4 * i;
            if (r >= m) continue;
            const T v = T(a[i]);
            nonFinite |= !isfinite(double(v));
            Xt[q * M1 + std::size_t(r)] = v;
        }
    }
    if (nonFinite) bad[0] = 1; // same value from every writer
}

// x = Xt^T b (Xt column-major), FP64 accumulation.
template <class T>
__global__ void kDenseApply(int m, const T* __restrict__ Xt, const T* __restrict__ b, T* __restrict__ x)
{
    const int r = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (r >= m) return;
    // loads kPrefetch ahead, the same ascending FMA chain
    constexpr int U = internal::kPrefetch;
    double s = 0.0;
    int q = 0;
    for (; q + U <= m; q += U) {
        T a[U], v[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            a[u] = Xt[std::size_t(q + u) * std::size_t(m) + std::size_t(r)];
            v[u] = b[q + u];
        }
#pragma unroll
        for (int u = 0; u < U; ++u) s += double(a[u]) * double(v[u]);
    }
    for (; q < m; ++q) s += double(Xt[std::size_t(q) * std::size_t(m) + std::size_t(r)]) * double(b[q]);
    x[r] = T(s);
}

} // namespace

// ---- Level --------------------------------------------------------------------

template <class T>
struct Amg<T>::Level
{
    std::unique_ptr<BlockEll<T>> own; // levels >= 1
    const BlockEll<T>* A = nullptr;
    int inner = 0;
    bool coarsest = false;
    // transfer from the finer level (levels >= 1)
    int nSlots = 0;
    DeviceBuffer<int> memberPtr, member, agg, intraPtr, intraE, offPtr, offE, offDst;
    // smoother (non-coarsest levels)
    std::optional<BlockJacobi<T>> jac;
    std::optional<MultiColourGS<T>> gs;
    std::optional<LineSmoother<T>> lines;
    // vectors: b, x (levels >= 1), res, work, inner-step vectors
    DeviceBuffer<T> b, x, res, work, c1, v1, rt, d2, v2;
    DeviceBuffer<double> partial, s, coef;
    // level 0 with constantNullSpace: the per-component mean projection
    DeviceBuffer<T> ones, rp;
    DeviceBuffer<double> npartial, nsum;
    // coarsest
    int m = 0;
    DeviceBuffer<double> dense, scratch;
    DeviceBuffer<T> inv;
    // level 0: refresh's flags, level l's factor at l, the coarsest inverse's at levels()
    DeviceBuffer<int> flags;
    std::size_t size() const { return A->size(); }
};

template <class T>
Amg<T>::Amg() = default;
template <class T>
Amg<T>::Amg(Amg&&) noexcept = default;
template <class T>
Amg<T>& Amg<T>::operator=(Amg&&) noexcept = default;
template <class T>
Amg<T>::~Amg() = default;

template <class T>
const BlockEll<T>& Amg<T>::matrix(int l) const
{
    return *lv_[std::size_t(l)]->A;
}

template <class T>
std::expected<Amg<T>, std::string> Amg<T>::make(const BlockEll<T>& A, const dns::la::AmgHierarchy& h,
                                                const AmgOptions& o)
{
    if (h.B != A.shape().B) return std::unexpected("amg: hierarchy block size " + std::to_string(h.B) + " != matrix " + std::to_string(A.shape().B));
    if (h.levels() < 1 || h.rows[0] != A.rows() || std::size_t(h.nnz[0]) != A.pattern().slotPos.size())
        return std::unexpected("amg: hierarchy not built on the matrix's graph (rows / slots differ)");
    if (o.sweeps < 1 || o.sweeps > 16) return std::unexpected("amg: sweeps outside [1, 16]");
    if (!(o.omega > 0 && o.omega < 2)) return std::unexpected("amg: omega outside (0, 2)");
    if (o.inner < 0 || o.inner > 2) return std::unexpected("amg: inner outside {0, 1, 2}");
    if (o.innerMinRows < 0) return std::unexpected("amg: innerMinRows < 0");
    const int L = h.levels();
    if (std::int64_t(h.rows[std::size_t(L - 1)]) * h.B > dns::la::kAmgMaxCoarseUnknowns)
        return std::unexpected("amg: coarsest level above the dense limit");
    Amg a;
    a.o_ = o;
    const dns::la::BlockShape shape = A.shape();
    for (int l = 0; l < L; ++l) {
        auto lv = std::make_unique<Level>();
        lv->coarsest = l == L - 1;
        if (l == 0) lv->A = &A;
        else {
            const dns::la::CellGraph& g = h.graphs[std::size_t(l - 1)];
            const dns::la::EllPattern pat = dns::la::ellPattern(g);
            auto m = BlockEll<T>::make(pat, shape);
            if (!m) return std::unexpected(m.error());
            lv->own = std::make_unique<BlockEll<T>>(std::move(*m));
            lv->A = lv->own.get();
            const dns::la::AmgTransfer& t = h.transfers[std::size_t(l - 1)];
            const dns::la::EllPattern& fp = a.lv_[std::size_t(l - 1)]->A->pattern();
            std::vector<int> intraE(t.intraSlot.size()), offE(t.offSlot.size());
            for (std::size_t k = 0; k < intraE.size(); ++k) intraE[k] = fp.slotPos[std::size_t(t.intraSlot[k])];
            for (std::size_t k = 0; k < offE.size(); ++k) offE[k] = fp.slotPos[std::size_t(t.offSlot[k])];
            lv->nSlots = int(g.col.size());
            for (auto [dst, src] : {std::pair{&lv->memberPtr, &t.memberPtr}, std::pair{&lv->member, &t.member},
                                    std::pair{&lv->agg, &t.agg}, std::pair{&lv->intraPtr, &t.intraPtr},
                                    std::pair{&lv->intraE, static_cast<const std::vector<int>*>(&intraE)},
                                    std::pair{&lv->offPtr, &t.offPtr},
                                    std::pair{&lv->offE, static_cast<const std::vector<int>*>(&offE)},
                                    std::pair{&lv->offDst, &lv->A->pattern().slotPos}}) {
                auto d = devVec(*src, "amg transfer");
                if (!d) return std::unexpected(d.error());
                *dst = std::move(*d);
            }
        }
        const std::size_t N = lv->size();
        // inner steps on coarse levels 1 and 2 always, deeper only at >= innerMinRows rows
        lv->inner = (l == 0 || lv->coarsest || (l > 2 && lv->A->rows() < o.innerMinRows)) ? 0 : o.inner;
        auto vec = [&](DeviceBuffer<T>& v, std::size_t n) -> std::expected<void, std::string> {
            auto d = devZero<T>(n, "amg vectors");
            if (!d) return std::unexpected(d.error());
            v = std::move(*d);
            return {};
        };
        if (l > 0)
            for (auto* v : {&lv->b, &lv->x})
                if (auto e = vec(*v, N); !e) return std::unexpected(e.error());
        if (lv->coarsest) {
            lv->m = int(N);
            auto d = devZero<double>(N * N, "amg coarsest");
            if (!d) return std::unexpected(d.error());
            lv->dense = std::move(*d);
            auto s = devZero<double>(N * N, "amg coarsest");
            if (!s) return std::unexpected(s.error());
            lv->scratch = std::move(*s);
            auto iv = devZero<T>(N * N, "amg coarsest");
            if (!iv) return std::unexpected(iv.error());
            lv->inv = std::move(*iv);
        } else {
            if (auto e = vec(lv->res, N); !e) return std::unexpected(e.error());
            if (auto e = vec(lv->work, 2 * N); !e) return std::unexpected(e.error());
            const AmgSmoother sm = l == 0 ? o.smoother : o.coarseSmoother;
            if (sm == AmgSmoother::Jacobi) {
                auto j = BlockJacobi<T>::make(*lv->A);
                if (!j) return std::unexpected(j.error());
                lv->jac.emplace(std::move(*j));
            } else if (sm == AmgSmoother::MultiColourGS) {
                const auto& col = h.colours[std::size_t(l)];
                if (col.empty())
                    return std::unexpected("amg: level " + std::to_string(l) +
                                           " has a row above 254 entries: no colouring for MC-GS");
                auto g = MultiColourGS<T>::make(*lv->A, col);
                if (!g) return std::unexpected(g.error());
                lv->gs.emplace(std::move(*g));
            } else {
                if (h.lines.size() != std::size_t(L))
                    return std::unexpected("amg: lines smoother needs a hierarchy built with lineRatio > 1");
                auto s = LineSmoother<T>::make(*lv->A, h.lines[std::size_t(l)]);
                if (!s) return std::unexpected(s.error());
                lv->lines.emplace(std::move(*s));
            }
        }
        if (lv->inner > 0) {
            for (auto* v : {&lv->c1, &lv->v1, &lv->rt, &lv->d2, &lv->v2})
                if (auto e = vec(*v, N); !e) return std::unexpected(e.error());
            auto p = devZero<double>(std::size_t(internal::dotParts(N)) * 3, "amg dots");
            if (!p) return std::unexpected(p.error());
            lv->partial = std::move(*p);
            auto s = devZero<double>(8, "amg dots");
            if (!s) return std::unexpected(s.error());
            lv->s = std::move(*s);
            auto c = devZero<double>(4, "amg dots");
            if (!c) return std::unexpected(c.error());
            lv->coef = std::move(*c);
        }
        if (l == 0 && o.constantNullSpace && !lv->coarsest) {
            const std::size_t n = std::size_t(lv->A->rows());
            if (auto e = vec(lv->rp, N); !e) return std::unexpected(e.error());
            auto on = devVec(std::vector<T>(n, T(1)), "amg null space");
            if (!on) return std::unexpected(on.error());
            lv->ones = std::move(*on);
            const std::size_t nB = std::size_t(shape.B);
            auto p = devZero<double>(std::size_t(internal::dotParts(n)) * nB, "amg null space");
            if (!p) return std::unexpected(p.error());
            lv->npartial = std::move(*p);
            auto sm = devZero<double>(nB, "amg null space");
            if (!sm) return std::unexpected(sm.error());
            lv->nsum = std::move(*sm);
        }
        a.lv_.push_back(std::move(lv));
    }
    auto fl = devZero<int>(std::size_t(L) + 1, "amg flags");
    if (!fl) return std::unexpected(fl.error());
    a.lv_[0]->flags = std::move(*fl);
    return a;
}

template <class T>
std::expected<void, std::string> Amg<T>::refresh()
{
    const int L = levels();
    const dns::la::BlockShape shape = lv_[0]->A->shape();
    for (int l = 1; l < L; ++l) {
        Level& C = *lv_[std::size_t(l)];
        const BlockEll<T>& F = *lv_[std::size_t(l - 1)]->A;
        const EllView<T> fv = F.view();
        const int nc = C.A->rows();
        internal::withShape(shape, [&](auto sh) {
            kGatherDiag<decltype(sh)::B, decltype(sh)::Bd, T><<<gridFor(std::size_t(nc)), kThreads>>>(
                nc, fv, C.memberPtr.data(), C.member.data(), C.intraPtr.data(), C.intraE.data(), C.own->diag());
        });
        if (C.nSlots > 0)
            kGatherOff<T><<<gridFor(std::size_t(C.nSlots)), kThreads>>>(
                C.nSlots, shape.offDense(), shape.nS, C.offPtr.data(), C.offE.data(), C.offDst.data(), fv.off, fv.sc,
                std::size_t(fv.nE), C.own->off(), C.own->sc(), std::size_t(C.A->pattern().nE));
    }
    // Every factor and the coarsest inverse flag into flags_[0, L] (the
    // coarsest's at L), read with ONE download; the lowest failing level's
    // error first, as the level-by-level reads reported it.
    int* fl = lv_[0]->flags.data();
    for (int l = 0; l < L; ++l) {
        Level& V = *lv_[std::size_t(l)];
        if (V.jac) V.jac->factorDeferred(fl + l);
        else if (V.gs) V.gs->factorDeferred(fl + l);
        else if (V.lines) V.lines->factorDeferred(fl + l);
    }
    Level& Cs = *lv_[std::size_t(L - 1)];
    const std::size_t m = std::size_t(Cs.m);
    if (const hipError_t e = hipMemsetAsync(Cs.dense.data(), 0, m * m * sizeof(double), 0); e != hipSuccess)
        return std::unexpected(hipText("amg coarsest", e));
    const EllView<T> cv = Cs.A->view();
    internal::withShape(shape, [&](auto sh) {
        kDenseBuild<decltype(sh)::B, decltype(sh)::Bd, T><<<gridFor(std::size_t(cv.n)), kThreads>>>(cv, Cs.dense.data());
    });
    if (Cs.m <= kDenseRegMax)
        kDenseInvertReg<T><<<1, kDenseThreads>>>(Cs.m, shape.B, o_.constantNullSpace ? 1 : 0, Cs.dense.data(),
                                                 Cs.inv.data(), fl + L);
    else
        kDenseInvert<T><<<1, kDenseThreads>>>(Cs.m, shape.B, o_.constantNullSpace ? 1 : 0, Cs.dense.data(),
                                              Cs.scratch.data(), Cs.inv.data(), fl + L);
    std::vector<int> h(std::size_t(L) + 1, 0);
    if (const hipError_t e = hipMemcpy(h.data(), fl, h.size() * sizeof(int), hipMemcpyDeviceToHost); e != hipSuccess)
        return std::unexpected(hipText("amg refresh", e));
    if (const hipError_t e = hipGetLastError(); e != hipSuccess) return std::unexpected(hipText("amg refresh", e));
    if (std::ranges::all_of(h, [](int v) { return v == 0; })) return {};
    (void)hipMemset(fl, 0, h.size() * sizeof(int));
    for (int l = 0; l < L; ++l) {
        if (h[std::size_t(l)] == 0) continue;
        const Level& V = *lv_[std::size_t(l)];
        return std::unexpected("amg level " + std::to_string(l) + ": " +
                               (V.lines ? LineSmoother<T>::kFactorError : BlockJacobi<T>::kFactorError));
    }
    return std::unexpected("amg: coarsest matrix (" + std::to_string(Cs.m) + " unknowns) singular or its inverse non-finite" +
                           (o_.constantNullSpace ? "" : " (a pure Neumann operator needs constantNullSpace)"));
}

template <class T>
void Amg<T>::smooth(int l, const T* b, T* x)
{
    Level& V = *lv_[std::size_t(l)];
    if (V.jac) V.jac->smooth(b, x, o_.sweeps, o_.omega, V.work.data());
    else if (V.gs) V.gs->smooth(b, x, o_.sweeps, 1.0);
    else V.lines->smooth(b, x, o_.sweeps, V.work.data(), o_.omega);
}

template <class T>
void Amg<T>::cycle(int l, const T* b, T* x)
{
    Level& V = *lv_[std::size_t(l)];
    if (V.coarsest) {
        kDenseApply<T><<<gridFor(std::size_t(V.m)), kThreads>>>(V.m, V.inv.data(), b, x);
        return;
    }
    const int B = V.A->shape().B, nf = V.A->rows();
    if (V.jac) V.jac->smoothFromZero(b, x, o_.sweeps, o_.omega, V.work.data()); // == zeroed x, then smooth
    else {
        (void)hipMemsetAsync(x, 0, V.size() * sizeof(T), 0);
        smooth(l, b, x);
    }
    V.A->residual(b, x, V.res.data());
    Level& C = *lv_[std::size_t(l + 1)];
    const int nc = C.A->rows();
    kRestrict<T><<<gridFor(std::size_t(nc)), kThreads>>>(nc, B, nf, C.memberPtr.data(), C.member.data(), V.res.data(),
                                                         C.b.data());
    if (correct(l + 1, C.b.data(), C.x.data()))
        kProlongAddScaled<T><<<gridFor(std::size_t(nf)), kThreads>>>(nf, B, nc, C.agg.data(), C.coef.data(),
                                                                     C.c1.data(), x);
    else kProlongAdd<T><<<gridFor(std::size_t(nf)), kThreads>>>(nf, B, nc, C.agg.data(), C.x.data(), x);
    smooth(l, b, x);
}

template <class T>
bool Amg<T>::correct(int k, const T* b, T* x)
{
    Level& C = *lv_[std::size_t(k)];
    if (C.inner == 0) {
        cycle(k, b, x);
        return false;
    }
    const std::size_t N = C.size();
    const unsigned g = gridFor(N);
    cycle(k, b, C.c1.data());
    C.A->spmv(C.c1.data(), C.v1.data());
    internal::DotArgs<T> d{};
    d.a[0] = C.c1.data();
    d.b[0] = C.v1.data();
    d.a[1] = C.c1.data();
    d.b[1] = b;
    internal::deviceDots(N, d, 2, C.partial.data(), C.s.data());
    kCoef1<<<1, 1>>>(C.s.data(), C.coef.data());
    if (C.inner == 1) return true; // x = coef c1 rides the caller's prolongation (kProlongAddScaled)
    kSubScaled<T><<<g, kThreads>>>(N, b, C.coef.data(), C.v1.data(), C.rt.data());
    cycle(k, C.rt.data(), C.d2.data());
    C.A->spmv(C.d2.data(), C.v2.data());
    d.a[0] = C.d2.data();
    d.b[0] = C.v1.data();
    d.a[1] = C.d2.data();
    d.b[1] = C.v2.data();
    d.a[2] = C.d2.data();
    d.b[2] = C.rt.data();
    internal::deviceDots(N, d, 3, C.partial.data(), C.s.data() + 2);
    kCoef2<<<1, 1>>>(C.s.data(), C.coef.data());
    kComb2<T><<<g, kThreads>>>(N, C.coef.data(), C.c1.data(), C.d2.data(), x);
    return false;
}

template <class T>
void Amg<T>::removeMean(const T* x, T* y)
{
    Level& V = *lv_[0];
    const int n = V.A->rows(), B = V.A->shape().B;
    internal::DotArgs<T> d{};
    for (int q = 0; q < B; ++q) {
        d.a[q] = x + std::size_t(q) * std::size_t(n);
        d.b[q] = V.ones.data();
    }
    internal::deviceDots(std::size_t(n), d, B, V.npartial.data(), V.nsum.data());
    kRemoveMean<T><<<gridFor(V.size()), kThreads>>>(n, B, V.nsum.data(), x, y);
}

template <class T>
void Amg<T>::apply(const T* r, T* z)
{
    if (!lv_[0]->ones.size()) {
        cycle(0, r, z);
        return;
    }
    removeMean(r, lv_[0]->rp.data());
    cycle(0, lv_[0]->rp.data(), z);
    removeMean(z, z);
}

template class Amg<float>;
template class Amg<double>;

} // namespace dns::gpu::la
