#include "dns/gpu/la/smooth.hpp"

#include "dns/gpu/la/internal/ellops.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>

namespace dns::gpu::la {

namespace {

using internal::diagBlock;
using internal::gridFor;
using internal::invert;
using internal::kThreads;
using internal::offBlock;
constexpr int kLane = dns::la::kSliceRows; // lines per group

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

// Reads back and clears a device failure counter.
std::expected<int, std::string> takeFlag(DeviceBuffer<int>& f, const char* what)
{
    int h = 0;
    if (const hipError_t e = f.download(&h); e != hipSuccess) return std::unexpected(hipText(what, e));
    if (h != 0)
        if (const hipError_t e = hipMemset(f.data(), 0, sizeof(int)); e != hipSuccess)
            return std::unexpected(hipText(what, e));
    return h;
}

// ---- block Jacobi ---------------------------------------------------------

template <int B, class T>
__device__ inline bool storeFinite(const double (&X)[B][B], T* dst, std::size_t stride, std::size_t at)
{
    bool ok = true;
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) {
            const T v = T(X[r][q]);
            ok = ok && isfinite(double(v));
            dst[std::size_t(r * B + q) * stride + at] = v;
        }
    return ok;
}

template <int B, class T>
__global__ void kDiagInv(EllView<T> A, T* inv, int* bad)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= A.n) return;
    double M[B][B], X[B][B];
    diagBlock<B>(A, c, M);
    const bool ok = invert<B>(M, X) && storeFinite<B>(X, inv, std::size_t(A.n), std::size_t(c));
    if (!ok) bad[0] = 1; // same value from every writer
}

// z = D^-1 r (x += omega D^-1 r when accumulate).
template <int B, class T>
__global__ void kDiagApply(int n, const T* __restrict__ inv, const T* __restrict__ r, T* __restrict__ z,
                           int accumulate, T omega)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= n) return;
    const std::size_t N = std::size_t(n);
    T rc[B];
#pragma unroll
    for (int q = 0; q < B; ++q) rc[q] = r[std::size_t(q) * N + std::size_t(c)];
#pragma unroll
    for (int p = 0; p < B; ++p) {
        T s = T(0);
#pragma unroll
        for (int q = 0; q < B; ++q) s += inv[std::size_t(p * B + q) * N + std::size_t(c)] * rc[q];
        const std::size_t i = std::size_t(p) * N + std::size_t(c);
        z[i] = accumulate ? z[i] + omega * s : s;
    }
}

// The first damped Jacobi step from x = 0: residual + kDiagApply(accumulate)
// on a zeroed x, fused. b - A 0 == b exactly (every product is +-0 and the
// accumulator starts at +0), and the update keeps kDiagApply's literal form
// `x + omega s` with x the literal +0, so the bits (sign of zero included) are
// the memset + sweep's.
template <int B, class T>
__global__ void kJacobiZero(int n, const T* __restrict__ inv, const T* __restrict__ b, T* __restrict__ x, T omega)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= n) return;
    const std::size_t N = std::size_t(n);
    T rc[B];
#pragma unroll
    for (int q = 0; q < B; ++q) rc[q] = b[std::size_t(q) * N + std::size_t(c)];
#pragma unroll
    for (int p = 0; p < B; ++p) {
        T s = T(0);
#pragma unroll
        for (int q = 0; q < B; ++q) s += inv[std::size_t(p * B + q) * N + std::size_t(c)] * rc[q];
        x[std::size_t(p) * N + std::size_t(c)] = T(0) + omega * s;
    }
}

// One damped Jacobi step x -> xn = x + omega D^-1 (b - A x): kSpmv's residual
// row and kDiagApply's update in one pass, same expressions. Ping-pong
// (xn != x): kDiagApply too updated x only after the whole residual existed.
template <int B, int Bd, class T>
__global__ void __launch_bounds__(256) kJacobiSweep(EllView<T> A, const T* __restrict__ inv, const T* __restrict__ b,
                                                    const T* __restrict__ x, T* __restrict__ xn, T omega)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= A.n) return;
    const std::size_t N = std::size_t(A.n);
    T acc[B], rc[B];
    internal::spmvRow<B, Bd>(A, c, x, acc);
#pragma unroll
    for (int q = 0; q < B; ++q) {
        const std::size_t i = std::size_t(q) * N + std::size_t(c);
        rc[q] = b[i] - acc[q];
    }
#pragma unroll
    for (int p = 0; p < B; ++p) {
        T s = T(0);
#pragma unroll
        for (int q = 0; q < B; ++q) s += inv[std::size_t(p * B + q) * N + std::size_t(c)] * rc[q];
        const std::size_t i = std::size_t(p) * N + std::size_t(c);
        xn[i] = x[i] + omega * s;
    }
}

// ---- multicolour GS -------------------------------------------------------

// One colour pass on the PRIVATE colour-major ELL P: slots r0 .. r0 + m - 1
// hold rows rows[slot] of this colour (P.n == A's rows; the slot indexes P's
// slices and inv, the row indexes b and x). Row c's walk is the cell-order
// walk's: same entries, same order, same expressions.
template <int B, int Bd, class T>
__global__ void __launch_bounds__(256) kGsColour(EllView<T> P, const int* __restrict__ rows, int r0, int m,
                                                 const T* __restrict__ inv, const T* __restrict__ b, T* x,
                                                 T omega)
{
    const int t = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (t >= m) return;
    const int i = r0 + t, c = rows[i];
    const std::size_t n = std::size_t(P.n);
    T acc[B];
#pragma unroll
    for (int r = 0; r < B; ++r) acc[r] = b[std::size_t(r) * n + std::size_t(c)];
    internal::ellRow<B, Bd, true>(P, i, x, acc);
#pragma unroll
    for (int p = 0; p < B; ++p) {
        T s = T(0);
#pragma unroll
        for (int q = 0; q < B; ++q) s += inv[std::size_t(p * B + q) * n + std::size_t(i)] * acc[q];
        const std::size_t ix = std::size_t(p) * n + std::size_t(c);
        x[ix] = omega == T(1) ? s : (T(1) - omega) * x[ix] + omega * s;
    }
}

// kGsColour<1, 1> for nb systems sharing the off-diagonals (system q: b, x at
// q n, D^-1 at q n of inv; only q in mask).
template <class T>
__global__ void __launch_bounds__(256) kGsColourMulti(EllView<T> P, const int* __restrict__ rows, int r0, int m,
                                                      int nb, unsigned mask, const T* __restrict__ inv,
                                                      const T* __restrict__ b, T* x, T omega)
{
    const int t = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (t >= m) return;
    const int i = r0 + t, c = rows[i];
    const std::size_t n = std::size_t(P.n);
    T acc[3];
#pragma unroll
    for (int q = 0; q < 3; ++q) acc[q] = q < nb && ((mask >> q) & 1u) ? b[std::size_t(q) * n + std::size_t(c)] : T(0);
    internal::ellRowMulti<true>(P, i, nb, mask, x, n, acc);
#pragma unroll
    for (int q = 0; q < 3; ++q) {
        if (q >= nb || !((mask >> q) & 1u)) continue;
        T s = T(0);
        s += inv[std::size_t(q) * n + std::size_t(i)] * acc[q];
        const std::size_t ix = std::size_t(q) * n + std::size_t(c);
        x[ix] = omega == T(1) ? s : (T(1) - omega) * x[ix] + omega * s;
    }
}

// kDiagInvPerm<1> of system q's diagonal (diags + q n) for q in mask.
template <class T>
__global__ void kDiagInvPermMulti(int n, int nb, unsigned mask, const int* __restrict__ rows, const T* diags, T* inv,
                                  int* f0, int* f1, int* f2)
{
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= n) return;
    const std::size_t N = std::size_t(n);
    for (int q = 0; q < nb; ++q) {
        if (!((mask >> q) & 1u)) continue;
        double M[1][1], X[1][1];
        M[0][0] = double(diags[std::size_t(q) * N + std::size_t(rows[i])]);
        const bool ok = invert<1>(M, X) && storeFinite<1>(X, inv + std::size_t(q) * N, N, std::size_t(i));
        if (!ok) (q == 0 ? f0 : q == 1 ? f1 : f2)[0] = 1; // same value from every writer
    }
}

// The private ELL's values: entry E' copies A's entry src[E'] (padding: 0).
template <class T>
__global__ void kGsGather(int nE, int nDense, int nS, const int* __restrict__ src, const T* __restrict__ off,
                          const T* __restrict__ sc, std::size_t nEA, T* __restrict__ pOff, T* __restrict__ pSc)
{
    const int E = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (E >= nE) return;
    const int s = src[E];
    for (int e = 0; e < nDense; ++e)
        pOff[std::size_t(e) * std::size_t(nE) + std::size_t(E)] = s < 0 ? T(0) : off[std::size_t(e) * nEA + std::size_t(s)];
    for (int e = 0; e < nS; ++e)
        pSc[std::size_t(e) * std::size_t(nE) + std::size_t(E)] = s < 0 ? T(0) : sc[std::size_t(e) * nEA + std::size_t(s)];
}

// kDiagInv of row rows[i], stored at slot i.
template <int B, class T>
__global__ void kDiagInvPerm(EllView<T> A, const int* __restrict__ rows, T* inv, int* bad)
{
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= A.n) return;
    double M[B][B], X[B][B];
    diagBlock<B>(A, rows[i], M);
    const bool ok = invert<B>(M, X) && storeFinite<B>(X, inv, std::size_t(A.n), std::size_t(i));
    if (!ok) bad[0] = 1; // same value from every writer
}

// ---- lines ----------------------------------------------------------------

// Line l (sorted) is lane l % 64 of group l / 64; position p of it is
// idx = gBase[group] + 64 p + lane.
struct LineView
{
    int nLines = 0, nP = 0;
    const int *len, *gBase, *cell, *eLo, *eHi;
};

template <int B, int Bd, class T>
__global__ void kLineFactor(LineView L, EllView<T> A, T* Dp, T* G, T* C, int* bad)
{
    const int l = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (l >= L.nLines) return;
    const std::size_t nP = std::size_t(L.nP);
    const int len = L.len[l];
    int idx = L.gBase[l / kLane] + l % kLane;
    double Cp[B][B];
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) Cp[r][q] = 0.0;
    bool ok = true;
    for (int p = 0; p < len; ++p, idx += kLane) {
        double Tm[B][B], Lb[B][B], X[B][B], Y[B][B];
        diagBlock<B>(A, L.cell[idx], Tm);
        if (p > 0) {
            offBlock<B, Bd>(A, L.eLo[idx], Lb);
#pragma unroll
            for (int r = 0; r < B; ++r)
#pragma unroll
                for (int q = 0; q < B; ++q) {
                    double s = 0.0;
#pragma unroll
                    for (int t = 0; t < B; ++t) s += Lb[r][t] * Cp[t][q];
                    Tm[r][q] -= s;
                }
        } else {
#pragma unroll
            for (int r = 0; r < B; ++r)
#pragma unroll
                for (int q = 0; q < B; ++q) Lb[r][q] = 0.0;
        }
        ok = ok && invert<B>(Tm, X);
        ok = ok && storeFinite<B>(X, Dp, nP, std::size_t(idx));
        // G = X Lb, C = X Ub (0 at the line end)
#pragma unroll
        for (int r = 0; r < B; ++r)
#pragma unroll
            for (int q = 0; q < B; ++q) {
                double s = 0.0;
#pragma unroll
                for (int t = 0; t < B; ++t) s += X[r][t] * Lb[t][q];
                Y[r][q] = s;
            }
        ok = ok && storeFinite<B>(Y, G, nP, std::size_t(idx));
        if (p + 1 < len) offBlock<B, Bd>(A, L.eHi[idx], Lb);
        else
#pragma unroll
            for (int r = 0; r < B; ++r)
#pragma unroll
                for (int q = 0; q < B; ++q) Lb[r][q] = 0.0;
#pragma unroll
        for (int r = 0; r < B; ++r)
#pragma unroll
            for (int q = 0; q < B; ++q) {
                double s = 0.0;
#pragma unroll
                for (int t = 0; t < B; ++t) s += X[r][t] * Lb[t][q];
                Cp[r][q] = s;
            }
        ok = ok && storeFinite<B>(Cp, C, nP, std::size_t(idx));
        if (!ok) break;
    }
    if (!ok) bad[0] = 1;
}

// z = M^-1 r: forward y_p = Dp_p r_p - G_p y_{p-1} into z, then back
// x_p = y_p - C_p x_{p+1}. Accumulated in T.
template <int B, class T>
__global__ void __launch_bounds__(256) kLineSolve(LineView L, int n, const T* __restrict__ Dp,
                                                  const T* __restrict__ G, const T* __restrict__ C,
                                                  const T* __restrict__ r, T* z)
{
    const int l = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (l >= L.nLines) return;
    const std::size_t nP = std::size_t(L.nP), N = std::size_t(n);
    const int len = L.len[l];
    const int idx0 = L.gBase[l / kLane] + l % kLane;
    T y[B];
#pragma unroll
    for (int q = 0; q < B; ++q) y[q] = T(0);
    int idx = idx0;
    for (int p = 0; p < len; ++p, idx += kLane) {
        const std::size_t c = std::size_t(L.cell[idx]), at = std::size_t(idx);
        T rc[B], yn[B];
#pragma unroll
        for (int q = 0; q < B; ++q) rc[q] = r[std::size_t(q) * N + c];
#pragma unroll
        for (int a = 0; a < B; ++a) {
            T s = T(0);
#pragma unroll
            for (int q = 0; q < B; ++q)
                s += Dp[std::size_t(a * B + q) * nP + at] * rc[q] - G[std::size_t(a * B + q) * nP + at] * y[q];
            yn[a] = s;
        }
#pragma unroll
        for (int q = 0; q < B; ++q) {
            y[q] = yn[q];
            z[std::size_t(q) * N + c] = yn[q];
        }
    }
    // y holds x of the last position.
    idx -= 2 * kLane;
    for (int p = len - 2; p >= 0; --p, idx -= kLane) {
        const std::size_t c = std::size_t(L.cell[idx]), at = std::size_t(idx);
        T xn[B];
#pragma unroll
        for (int a = 0; a < B; ++a) {
            T s = z[std::size_t(a) * N + c];
#pragma unroll
            for (int q = 0; q < B; ++q) s -= C[std::size_t(a * B + q) * nP + at] * y[q];
            xn[a] = s;
        }
#pragma unroll
        for (int q = 0; q < B; ++q) {
            y[q] = xn[q];
            z[std::size_t(q) * N + c] = xn[q];
        }
    }
}

template <class T>
__global__ void kAxpy(std::size_t N, const T* __restrict__ d, T* __restrict__ x)
{
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) x[i] += d[i];
}

template <class T>
__global__ void kAxpyScaled(std::size_t N, T omega, const T* __restrict__ d, T* __restrict__ x)
{
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < N) x[i] += omega * d[i];
}

} // namespace

// ---- BlockJacobi ------------------------------------------------------------

template <class T>
std::expected<BlockJacobi<T>, std::string> BlockJacobi<T>::make(const BlockEll<T>& A)
{
    BlockJacobi j;
    j.A_ = &A;
    const std::size_t B = std::size_t(A.shape().B);
    auto inv = devZero<T>(std::size_t(A.rows()) * B * B, "BlockJacobi");
    if (!inv) return std::unexpected(inv.error());
    j.inv_ = std::move(*inv);
    auto bad = devZero<int>(1, "BlockJacobi");
    if (!bad) return std::unexpected(bad.error());
    j.bad_ = std::move(*bad);
    return j;
}

template <class T>
void BlockJacobi<T>::factorDeferred(int* flag)
{
    if (A_->rows() == 0) return;
    const EllView<T> v = A_->view();
    internal::withShape(A_->shape(), [&](auto sh) {
        kDiagInv<decltype(sh)::B, T><<<gridFor(std::size_t(v.n)), kThreads>>>(v, inv_.data(), flag);
    });
}

template <class T>
std::expected<void, std::string> BlockJacobi<T>::factor()
{
    if (A_->rows() == 0) return {};
    factorDeferred(bad_.data());
    auto f = takeFlag(bad_, "BlockJacobi factor");
    if (!f) return std::unexpected(f.error());
    if (*f) return std::unexpected(kFactorError);
    return {};
}

template <class T>
void BlockJacobi<T>::apply(const T* r, T* z) const
{
    if (A_->rows() == 0) return;
    internal::withShape(A_->shape(), [&](auto sh) {
        kDiagApply<decltype(sh)::B, T><<<gridFor(std::size_t(A_->rows())), kThreads>>>(A_->rows(), inv_.data(), r, z,
                                                                                      0, T(1));
    });
}

// Steps s = 0 .. count-1 ping-pong between x and work so that the last one
// writes x: step s writes x iff count - s is odd. fromZero: step 0 is
// kJacobiZero. From x with an odd count, step 0 runs in place (the two-launch
// residual + kDiagApply), since a fused step cannot read and write x. (Fusing
// kJacobiZero into the first sweep -- the neighbours' x1 recomputed from inv,
// b -- is bitwise equal but was measured SLOWER on rotor2: 9.4 vs 6.5 us per
// level visit, the extra FP64 work per neighbour; not used.)
template <class T>
void BlockJacobi<T>::run(const T* b, T* x, bool fromZero, int count, double omega, T* work) const
{
    if (count < 1 || A_->rows() == 0) return;
    const int n = A_->rows();
    const EllView<T> v = A_->view();
    internal::withShape(A_->shape(), [&](auto sh) {
        constexpr int B = decltype(sh)::B, Bd = decltype(sh)::Bd;
        int s = 0;
        if (fromZero) {
            kJacobiZero<B, T><<<gridFor(std::size_t(n)), kThreads>>>(n, inv_.data(), b, count % 2 ? x : work,
                                                                      T(omega));
            s = 1;
        } else if (count % 2) {
            A_->residual(b, x, work);
            kDiagApply<B, T><<<gridFor(std::size_t(n)), kThreads>>>(n, inv_.data(), work, x, 1, T(omega));
            s = 1;
        }
        for (; s < count; ++s) {
            const bool toX = (count - s) % 2 == 1;
            kJacobiSweep<B, Bd, T><<<gridFor(std::size_t(n)), kThreads>>>(v, inv_.data(), b, toX ? work : x,
                                                                          toX ? x : work, T(omega));
        }
    });
}

template <class T>
void BlockJacobi<T>::smooth(const T* b, T* x, int sweeps, double omega, T* work) const
{
    run(b, x, false, sweeps, omega, work);
}

template <class T>
void BlockJacobi<T>::smoothFromZero(const T* b, T* x, int sweeps, double omega, T* work) const
{
    run(b, x, true, sweeps, omega, work);
}

// ---- MultiColourGS ----------------------------------------------------------

template <class T>
std::expected<MultiColourGS<T>, std::string> MultiColourGS<T>::make(const BlockEll<T>& A,
                                                                    std::span<const std::uint8_t> colours)
{
    const dns::la::EllPattern& p = A.pattern();
    if (colours.size() != std::size_t(p.n)) return std::unexpected("MultiColourGS: colour count != rows");
    for (int c = 0; c < p.n; ++c) {
        const int base = p.sliceBase[std::size_t(c / dns::la::kSliceRows)] + c % dns::la::kSliceRows;
        for (int k = 0; k < p.sliceWidth[std::size_t(c / dns::la::kSliceRows)]; ++k) {
            const int j = p.col[std::size_t(base + k * dns::la::kSliceRows)];
            if (j < 0) break;
            if (colours[std::size_t(j)] == colours[std::size_t(c)])
                return std::unexpected("MultiColourGS: rows " + std::to_string(c) + " and " + std::to_string(j) +
                                       " couple and share colour " + std::to_string(colours[std::size_t(c)]));
        }
    }
    MultiColourGS g;
    g.A_ = &A;
    std::vector<int> rows(std::size_t(p.n));
    std::iota(rows.begin(), rows.end(), 0);
    std::ranges::stable_sort(rows, {}, [&](int c) { return colours[std::size_t(c)]; });
    const int nCol = p.n == 0 ? 0 : int(*std::ranges::max_element(colours)) + 1;
    g.colourStart_.assign(std::size_t(nCol) + 1, 0);
    for (const std::uint8_t k : colours) ++g.colourStart_[std::size_t(k) + 1];
    std::partial_sum(g.colourStart_.begin(), g.colourStart_.end(), g.colourStart_.begin());
    // the private colour-major ELL: slot i holds row rows[i]; slices of 64 slots
    // padded to their longest row (padding col -1, src -1)
    constexpr int S = dns::la::kSliceRows;
    auto rowLen = [&](int c) {
        const int base = p.sliceBase[std::size_t(c / S)] + c % S;
        int k = 0;
        while (k < p.sliceWidth[std::size_t(c / S)] && p.col[std::size_t(base + k * S)] >= 0) ++k;
        return k;
    };
    const int nSl = (p.n + S - 1) / S;
    std::vector<int> base(std::size_t(nSl) + 1, 0), width(std::size_t(std::max(nSl, 1)), 0);
    for (int sl = 0; sl < nSl; ++sl) {
        int w = 0;
        for (int i = sl * S; i < std::min(p.n, (sl + 1) * S); ++i) w = std::max(w, rowLen(rows[std::size_t(i)]));
        width[std::size_t(sl)] = w;
        base[std::size_t(sl) + 1] = base[std::size_t(sl)] + w * S;
    }
    g.nE_ = base[std::size_t(nSl)];
    std::vector<int> col(std::size_t(std::max(g.nE_, 1)), -1), src(std::size_t(std::max(g.nE_, 1)), -1);
    for (int i = 0; i < p.n; ++i) {
        const int c = rows[std::size_t(i)], cb = p.sliceBase[std::size_t(c / S)] + c % S, L = rowLen(c);
        for (int k = 0; k < L; ++k) {
            const std::size_t E = std::size_t(base[std::size_t(i / S)] + i % S + k * S);
            src[E] = cb + k * S;
            col[E] = p.col[std::size_t(src[E])];
        }
    }
    base.pop_back();
    for (auto [dst, h] : {std::pair{&g.rows_, &rows}, std::pair{&g.base_, &base}, std::pair{&g.width_, &width},
                          std::pair{&g.col_, &col}, std::pair{&g.src_, &src}}) {
        auto d = devVec(*h, "MultiColourGS layout");
        if (!d) return std::unexpected(d.error());
        *dst = std::move(*d);
    }
    const dns::la::BlockShape sh = A.shape();
    for (auto [dst, n] : {std::pair{&g.off_, std::size_t(g.nE_) * std::size_t(sh.offDense())},
                          std::pair{&g.sc_, std::size_t(g.nE_) * std::size_t(sh.nS)},
                          std::pair{&g.inv_, std::size_t(p.n) * std::size_t(sh.B) * std::size_t(sh.B)}}) {
        auto d = devZero<T>(n, "MultiColourGS values");
        if (!d) return std::unexpected(d.error());
        *dst = std::move(*d);
    }
    auto bad = devZero<int>(1, "MultiColourGS");
    if (!bad) return std::unexpected(bad.error());
    g.bad_ = std::move(*bad);
    return g;
}

template <class T>
void MultiColourGS<T>::factorDeferred(int* flag, bool offChanged)
{
    const EllView<T> v = A_->view();
    if (v.n == 0) return;
    const dns::la::BlockShape sh = A_->shape();
    if (offChanged && nE_ > 0)
        kGsGather<T><<<gridFor(std::size_t(nE_)), kThreads>>>(nE_, sh.offDense(), sh.nS, src_.data(), v.off, v.sc,
                                                             std::size_t(v.nE), off_.data(), sc_.data());
    internal::withShape(sh, [&](auto s) {
        kDiagInvPerm<decltype(s)::B, T><<<gridFor(std::size_t(v.n)), kThreads>>>(v, rows_.data(), inv_.data(), flag);
    });
}

template <class T>
std::expected<void, std::string> MultiColourGS<T>::factor(bool offChanged)
{
    if (A_->rows() == 0) return {};
    factorDeferred(bad_.data(), offChanged);
    auto f = takeFlag(bad_, "MultiColourGS factor");
    if (!f) return std::unexpected(f.error());
    if (*f) return std::unexpected(kFactorError);
    return {};
}

template <class T>
void MultiColourGS<T>::passes(const T* b, T* x, std::span<const int> order, double omega) const
{
    const EllView<T> a = A_->view();
    const EllView<T> P{a.n, nE_, a.shape, base_.data(), width_.data(), col_.data(), nullptr, off_.data(), sc_.data()};
    internal::withShape(A_->shape(), [&](auto sh) {
        int last = -1;
        for (const int k : order) {
            const int r0 = colourStart_[std::size_t(k)], m = colourStart_[std::size_t(k) + 1] - r0;
            if (m == 0) continue;
            if (k == last && omega == 1.0) continue; // a direct repeat at omega 1: the same bits
            kGsColour<decltype(sh)::B, decltype(sh)::Bd, T><<<gridFor(std::size_t(m)), kThreads>>>(
                P, rows_.data(), r0, m, inv_.data(), b, x, T(omega));
            last = k;
        }
    });
}

template <class T>
std::expected<void, std::string> MultiColourGS<T>::factorMulti(int nb, unsigned mask, const T* diags,
                                                               int* const* flags, bool offChanged)
{
    const EllView<T> v = A_->view();
    if (A_->shape().B != 1 || nb < 1 || nb > 3) return std::unexpected("MultiColourGS: factorMulti needs B 1, 1..3 systems");
    if (invMulti_.size() < std::size_t(v.n) * 3) {
        auto d = devZero<T>(std::size_t(v.n) * 3, "MultiColourGS systems");
        if (!d) return std::unexpected(d.error());
        invMulti_ = std::move(*d);
    }
    nbMulti_ = nb;
    maskMulti_ = mask;
    if (v.n == 0) return {};
    if (offChanged && nE_ > 0)
        kGsGather<T><<<gridFor(std::size_t(nE_)), kThreads>>>(nE_, 1, 0, src_.data(), v.off, v.sc, std::size_t(v.nE),
                                                             off_.data(), sc_.data());
    kDiagInvPermMulti<T><<<gridFor(std::size_t(v.n)), kThreads>>>(v.n, nb, mask, rows_.data(), diags, invMulti_.data(),
                                                                 flags[0], nb > 1 ? flags[1] : flags[0],
                                                                 nb > 2 ? flags[2] : flags[0]);
    return {};
}

template <class T>
void MultiColourGS<T>::applyMulti(int nb, unsigned mask, const T* r, T* z) const
{
    const EllView<T> a = A_->view();
    if (a.n == 0) return;
    (void)hipMemsetAsync(z, 0, A_->size() * std::size_t(nb) * sizeof(T), 0);
    const EllView<T> P{a.n, nE_, a.shape, base_.data(), width_.data(), col_.data(), nullptr, off_.data(), sc_.data()};
    const int K = colours();
    int last = -1;
    for (int s = 0; s < 2 * K; ++s) { // one symmetric sweep at omega 1: forward, backward
        const int k = s < K ? s : 2 * K - 1 - s;
        const int r0 = colourStart_[std::size_t(k)], m = colourStart_[std::size_t(k) + 1] - r0;
        if (m == 0 || k == last) continue; // a direct repeat at omega 1: the same bits
        kGsColourMulti<T><<<gridFor(std::size_t(m)), kThreads>>>(P, rows_.data(), r0, m, nb, mask, invMulti_.data(), r,
                                                                 z, T(1));
        last = k;
    }
}

template <class T>
void MultiColourGS<T>::sweep(const T* b, T* x, bool forward, double omega) const
{
    const int K = colours();
    std::vector<int> order(std::size_t(std::max(K, 0)));
    for (int t = 0; t < K; ++t) order[std::size_t(t)] = forward ? t : K - 1 - t;
    passes(b, x, order, omega);
}

template <class T>
void MultiColourGS<T>::smooth(const T* b, T* x, int sweeps, double omega) const
{
    const int K = colours();
    std::vector<int> order;
    for (int s = 0; s < sweeps; ++s) {
        for (int t = 0; t < K; ++t) order.push_back(t);
        for (int t = K - 1; t >= 0; --t) order.push_back(t);
    }
    passes(b, x, order, omega);
}

template <class T>
void MultiColourGS<T>::apply(const T* r, T* z) const
{
    if (A_->rows() == 0) return;
    (void)hipMemsetAsync(z, 0, A_->size() * sizeof(T), 0);
    smooth(r, z, 1, 1.0);
}

// ---- LineSmoother -----------------------------------------------------------

template <class T>
std::expected<LineSmoother<T>, std::string> LineSmoother<T>::make(const BlockEll<T>& A, const dns::la::LineSet& ls)
{
    const dns::la::EllPattern& p = A.pattern();
    if (ls.n != p.n) return std::unexpected("LineSmoother: line set rows != matrix rows");
    if (ls.cells.size() != std::size_t(p.n) || ls.start.empty() || ls.start.front() != 0 ||
        ls.start.back() != p.n || !std::ranges::is_sorted(ls.start))
        return std::unexpected("LineSmoother: line set does not cover every row exactly once");
    {
        std::vector<char> seen(std::size_t(p.n), 0);
        for (const int c : ls.cells) {
            if (c < 0 || c >= p.n || seen[std::size_t(c)])
                return std::unexpected("LineSmoother: row " + std::to_string(c) + " out of range or in two lines");
            seen[std::size_t(c)] = 1;
        }
    }
    // The ELL entry of column j in row c.
    auto entry = [&](int c, int j) {
        const int base = p.sliceBase[std::size_t(c / dns::la::kSliceRows)] + c % dns::la::kSliceRows;
        for (int k = 0; k < p.sliceWidth[std::size_t(c / dns::la::kSliceRows)]; ++k) {
            const int E = base + k * dns::la::kSliceRows;
            if (p.col[std::size_t(E)] == j) return E;
            if (p.col[std::size_t(E)] < 0) break;
        }
        return -1;
    };
    const int nL = ls.count();
    std::vector<int> order(static_cast<std::size_t>(nL));
    std::iota(order.begin(), order.end(), 0);
    auto lenOf = [&](int l) { return ls.start[std::size_t(l) + 1] - ls.start[std::size_t(l)]; };
    std::ranges::stable_sort(order, std::greater<>{}, lenOf);
    const int nG = (nL + kLane - 1) / kLane;
    std::vector<int> gBase(std::size_t(nG) + 1, 0), len(static_cast<std::size_t>(nL));
    for (int g = 0; g < nG; ++g) gBase[std::size_t(g) + 1] = gBase[std::size_t(g)] + kLane * lenOf(order[std::size_t(g) * kLane]);
    const int nP = gBase[std::size_t(nG)];
    std::vector<int> cell(std::size_t(nP), 0), eLo(std::size_t(nP), -1), eHi(std::size_t(nP), -1);
    for (int s = 0; s < nL; ++s) {
        const int l = order[std::size_t(s)], L = lenOf(l);
        len[std::size_t(s)] = L;
        const int* cs = ls.cells.data() + ls.start[std::size_t(l)];
        for (int q = 0; q < L; ++q) {
            const std::size_t idx = std::size_t(gBase[std::size_t(s / kLane)] + q * kLane + s % kLane);
            cell[idx] = cs[q];
            if (q > 0) eLo[idx] = entry(cs[q], cs[q - 1]);
            if (q + 1 < L) eHi[idx] = entry(cs[q], cs[q + 1]);
            if ((q > 0 && eLo[idx] < 0) || (q + 1 < L && eHi[idx] < 0))
                return std::unexpected("LineSmoother: line " + std::to_string(l) + " joins rows the matrix does not couple");
        }
    }
    LineSmoother m;
    m.A_ = &A;
    m.nLines_ = nL;
    m.nP_ = nP;
    for (auto [dst, src] : {std::pair{&m.len_, &len}, std::pair{&m.gBase_, &gBase}, std::pair{&m.cell_, &cell},
                            std::pair{&m.eLo_, &eLo}, std::pair{&m.eHi_, &eHi}}) {
        auto d = devVec(*src, "LineSmoother layout");
        if (!d) return std::unexpected(d.error());
        *dst = std::move(*d);
    }
    auto bad = devZero<int>(1, "LineSmoother");
    if (!bad) return std::unexpected(bad.error());
    m.bad_ = std::move(*bad);
    const std::size_t BB = std::size_t(A.shape().B) * std::size_t(A.shape().B);
    for (auto* dst : {&m.Dp_, &m.G_, &m.C_}) {
        auto d = devZero<T>(BB * std::size_t(nP), "LineSmoother factors");
        if (!d) return std::unexpected(d.error());
        *dst = std::move(*d);
    }
    return m;
}

template <class T>
void LineSmoother<T>::factorDeferred(int* flag)
{
    if (nLines_ == 0) return;
    const LineView L{nLines_, nP_, len_.data(), gBase_.data(), cell_.data(), eLo_.data(), eHi_.data()};
    const EllView<T> v = A_->view();
    internal::withShape(A_->shape(), [&](auto sh) {
        kLineFactor<decltype(sh)::B, decltype(sh)::Bd, T><<<gridFor(std::size_t(nLines_)), kThreads>>>(
            L, v, Dp_.data(), G_.data(), C_.data(), flag);
    });
}

template <class T>
std::expected<void, std::string> LineSmoother<T>::factor()
{
    if (nLines_ == 0) return {};
    factorDeferred(bad_.data());
    auto f = takeFlag(bad_, "LineSmoother factor");
    if (!f) return std::unexpected(f.error());
    if (*f) return std::unexpected(kFactorError);
    return {};
}

template <class T>
void LineSmoother<T>::apply(const T* r, T* z) const
{
    if (nLines_ == 0) return;
    const LineView L{nLines_, nP_, len_.data(), gBase_.data(), cell_.data(), eLo_.data(), eHi_.data()};
    internal::withShape(A_->shape(), [&](auto sh) {
        kLineSolve<decltype(sh)::B, T><<<gridFor(std::size_t(nLines_)), kThreads>>>(L, A_->rows(), Dp_.data(),
                                                                                   G_.data(), C_.data(), r, z);
    });
}

template <class T>
void LineSmoother<T>::smooth(const T* b, T* x, int sweeps, T* work, double omega) const
{
    T* res = work;
    T* dx = work + A_->size();
    for (int s = 0; s < sweeps && nLines_ > 0; ++s) {
        A_->residual(b, x, res);
        apply(res, dx);
        if (omega == 1.0) kAxpy<T><<<gridFor(A_->size()), kThreads>>>(A_->size(), dx, x);
        else kAxpyScaled<T><<<gridFor(A_->size()), kThreads>>>(A_->size(), T(omega), dx, x);
    }
}

template class BlockJacobi<float>;
template class BlockJacobi<double>;
template class MultiColourGS<float>;
template class MultiColourGS<double>;
template class LineSmoother<float>;
template class LineSmoother<double>;

} // namespace dns::gpu::la
