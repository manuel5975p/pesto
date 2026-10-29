#include "dns/gpu/la/krylov.hpp"

#include "dns/gpu/la/internal/dots.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <vector>

namespace dns::gpu::la {

namespace {

using internal::gridFor;
using internal::kThreads;
static_assert(kMaxDots == internal::kMaxDotBatch);

struct Coef
{
    double c[kMaxDots];
};

// w = sw w + sv sum_{k < m} c_k V_k, accumulated in FP64, k ascending.
template <class T>
__global__ void kCombine(std::size_t N, int m, const T* V, std::size_t ld, Coef c, double sw, double sv, T* w)
{
    const std::size_t i = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= N) return;
    double s = 0.0;
    for (int k = 0; k < m; ++k) s += c.c[k] * double(V[std::size_t(k) * ld + i]);
    w[i] = T(sw * double(w[i]) + sv * s);
}

// out = a x + b y (+ c z); out may alias any input (same index).
template <class T>
__global__ void kLin(std::size_t N, double a, const T* x, double b, const T* y, double c, const T* z, T* out)
{
    const std::size_t i = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= N) return;
    double s = a * double(x[i]) + b * double(y[i]);
    if (z) s += c * double(z[i]);
    out[i] = T(s);
}

// The device scalar step: alpha = num / d[0] into coef = [alpha, -alpha, bad],
// bad = the solver's own test on that dot (PCG: non-finite or not > 0;
// BiCGStab: non-finite or == 0). The host reads d with the next batch and
// takes the same branch; kLinDev is a no-op while bad is set.
__global__ void kAlpha(double num, const double* d, int pcg, double* coef)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const double v = d[0];
    const bool bad = !isfinite(v) || (pcg ? !(v > 0) : v == 0);
    const double a = num / v;
    coef[0] = a;
    coef[1] = -a;
    coef[2] = bad ? 1.0 : 0.0;
}

// kLin with b = coef[k] read on the device (the same expression, operand for
// operand); skipped while coef[2] (bad) is set.
template <class T>
__global__ void kLinDev(std::size_t N, double a, const T* x, const double* coef, int k, const T* y, double c,
                        const T* z, T* out)
{
    if (coef[2] != 0.0) return;
    const std::size_t i = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= N) return;
    const double b = coef[k];
    double s = a * double(x[i]) + b * double(y[i]);
    if (z) s += c * double(z[i]);
    out[i] = T(s);
}

// out_[kSlot + j]: system j's device-step dot; the pre-flag ints from out_[kFlags]
constexpr int kSlot = kMaxDots, kFlags = kMaxDots + Krylov<double>::kMaxBatch, kOut = kFlags + 2;
static_assert(2 * sizeof(double) >= Krylov<double>::kMaxBatch * sizeof(int));

// Batched kLin: system j (blockIdx.y) at offset j N; mode 0 skip, 1 copy
// (out = x, exact), 2 a x + b y, 3 a x + b y + c z -- kLin's expression.
struct LinBatch
{
    int mode[Krylov<double>::kMaxBatch] = {};
    double a[Krylov<double>::kMaxBatch] = {}, b[Krylov<double>::kMaxBatch] = {}, c[Krylov<double>::kMaxBatch] = {};
};

template <class T>
__global__ void kLinBatch(std::size_t N, LinBatch L, const T* x, const T* y, const T* z, T* out)
{
    const int j = int(blockIdx.y), md = L.mode[j];
    const std::size_t i = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (md == 0 || i >= N) return;
    const std::size_t k = std::size_t(j) * N + i;
    if (md == 1) {
        out[k] = x[k];
        return;
    }
    const double a = L.a[j], b = L.b[j], c = L.c[j];
    double s = a * double(x[k]) + b * double(y[k]);
    if (md == 3) s += c * double(z[k]);
    out[k] = T(s);
}

// kAlpha per system j < nb in `active`: num[j] / d[j] into coef + 3 j.
struct NumBatch
{
    double v[Krylov<double>::kMaxBatch] = {};
};

__global__ void kAlphaBatch(unsigned active, NumBatch num, const double* d, int pcg, double* coef)
{
    const int j = int(threadIdx.x);
    if (j >= Krylov<double>::kMaxBatch || !((active >> j) & 1u)) return;
    const double v = d[j];
    const bool bad = !isfinite(v) || (pcg ? !(v > 0) : v == 0);
    const double a = num.v[j] / v;
    coef[3 * j] = a;
    coef[3 * j + 1] = -a;
    coef[3 * j + 2] = bad ? 1.0 : 0.0;
}

// kLinDev per system j in `active` (blockIdx.y), skipped while its bad is set.
template <class T>
__global__ void kLinDevBatch(std::size_t N, unsigned active, double a, const T* x, const double* coef, int k,
                             const T* y, T* out)
{
    const int j = int(blockIdx.y);
    if (!((active >> j) & 1u) || coef[3 * j + 2] != 0.0) return;
    const std::size_t i = std::size_t(blockIdx.x) * kThreads + threadIdx.x;
    if (i >= N) return;
    const std::size_t e = std::size_t(j) * N + i;
    const double b = coef[3 * j + k];
    double s = a * double(x[e]) + b * double(y[e]);
    out[e] = T(s);
}

std::string hipText(const char* what, hipError_t e) { return std::string(what) + ": " + hipGetErrorString(e); }

// BiCGStab restarts its shadow residual when |(rh, r)| <= this |rh| |r|:
// below the FP64 dot's own rounding, (rh, r) is numerically zero. Measured:
// 1e-12 restarted a converging run 27 times and stalled it (cell B3).
constexpr double kShadowRestart = 0x1p-52;

} // namespace

template <class T>
std::expected<Krylov<T>, std::string> Krylov<T>::make(std::size_t N, int restart, int batch)
{
    if (restart < 0 || restart > kMaxRestart)
        return std::unexpected("Krylov: restart " + std::to_string(restart) + " outside [0, " +
                               std::to_string(kMaxRestart) + "]");
    if (batch < 1 || batch > kMaxBatch)
        return std::unexpected("Krylov: batch " + std::to_string(batch) + " outside [1, " + std::to_string(kMaxBatch) + "]");
    Krylov k;
    k.N_ = N;
    k.restart_ = restart;
    k.batch_ = batch;
    k.parts_ = internal::dotParts(N);
    auto aux = DeviceBuffer<T>::make(std::max<std::size_t>(1, 8 * N * std::size_t(batch)));
    if (!aux) return std::unexpected(hipText("Krylov work vectors", aux.error()));
    k.aux_ = std::move(*aux);
    auto basis = DeviceBuffer<T>::make(std::max<std::size_t>(1, std::size_t(2 * restart + 1) * N));
    if (!basis) return std::unexpected(hipText("Krylov basis", basis.error()));
    k.basis_ = std::move(*basis);
    auto part = DeviceBuffer<double>::make(std::size_t(kMaxDots) * std::size_t(k.parts_));
    if (!part) return std::unexpected(hipText("Krylov dots", part.error()));
    k.partial_ = std::move(*part);
    static_assert(sizeof host_ == kOut * sizeof(double));
    auto out = DeviceBuffer<double>::make(kOut);
    if (!out) return std::unexpected(hipText("Krylov dots", out.error()));
    k.out_ = std::move(*out);
    if (const hipError_t e = hipMemset(k.out_.data(), 0, kOut * sizeof(double)); e != hipSuccess)
        return std::unexpected(hipText("Krylov dots", e));
    auto coef = DeviceBuffer<double>::make(3 * kMaxBatch);
    if (!coef) return std::unexpected(hipText("Krylov dots", coef.error()));
    k.coef_ = std::move(*coef);
    return k;
}

template <class T>
int* Krylov<T>::preFlag(int j)
{
    return reinterpret_cast<int*>(out_.data() + kFlags) + j;
}

template <class T>
std::expected<void, std::string> Krylov<T>::batch(std::span<const DotPair<T>> pairs)
{
    const int K = int(pairs.size());
    if (K < 1 || K > kMaxDots) return std::unexpected("Krylov dots: bad batch size");
    internal::DotArgs<T> d{};
    for (int k = 0; k < K; ++k) {
        d.a[k] = pairs[std::size_t(k)].a;
        d.b[k] = pairs[std::size_t(k)].b;
    }
    internal::deviceDots(N_, d, K, partial_.data(), out_.data());
    if (const hipError_t e = hipMemcpy(host_, out_.data(), sizeof host_, hipMemcpyDeviceToHost); e != hipSuccess)
        return std::unexpected(hipText("Krylov dots", e));
    return {};
}

template <class T>
std::expected<bool, std::string> Krylov<T>::takePreFlag(int j)
{
    int f = 0;
    char* h = reinterpret_cast<char*>(&host_[kFlags]) + std::size_t(j) * sizeof(int);
    std::memcpy(&f, h, sizeof f);
    if (f == 0) return false;
    if (const hipError_t e = hipMemset(preFlag(j), 0, sizeof(int)); e != hipSuccess)
        return std::unexpected(hipText("Krylov pre-flag", e));
    std::memset(h, 0, sizeof(int));
    return true;
}

template <class T>
std::expected<void, std::string> Krylov<T>::dots(std::span<const DotPair<T>> pairs, std::span<double> out)
{
    const int K = int(pairs.size());
    if (K < 1 || K > kMaxDots || out.size() < pairs.size()) return std::unexpected("Krylov dots: bad batch size");
    internal::DotArgs<T> d{};
    for (int k = 0; k < K; ++k) {
        d.a[k] = pairs[std::size_t(k)].a;
        d.b[k] = pairs[std::size_t(k)].b;
    }
    internal::deviceDots(N_, d, K, partial_.data(), out_.data());
    if (const hipError_t e = hipMemcpy(out.data(), out_.data(), std::size_t(K) * sizeof(double),
                                       hipMemcpyDeviceToHost);
        e != hipSuccess)
        return std::unexpected(hipText("Krylov dots", e));
    return {};
}

namespace {

template <class T>
void lin(std::size_t N, double a, const T* x, double b, const T* y, T* out, double c = 0, const T* z = nullptr)
{
    kLin<T><<<gridFor(N), kThreads>>>(N, a, x, b, y, c, z, out);
}

template <class T>
void combine(std::size_t N, int m, const T* V, const double* c, double sv, T* w)
{
    if (m == 0) return;
    Coef cf{};
    for (int k = 0; k < m; ++k) cf.c[k] = c[k];
    kCombine<T><<<gridFor(N), kThreads>>>(N, m, V, N, cf, 1.0, sv, w);
}

// M r into z (identity: a copy).
template <class T>
void precondition(const LinOp<T>& M, const T* r, T* z, std::size_t N)
{
    if (M) M(r, z);
    else (void)hipMemcpyAsync(z, r, N * sizeof(T), hipMemcpyDeviceToDevice, 0);
}

std::unexpected<std::string> nonFinite(const char* who) { return std::unexpected(std::string(who) + ": non-finite residual"); }

} // namespace

template <class T>
std::expected<KrylovResult, std::string> Krylov<T>::fgmres(const LinOp<T>& A, const LinOp<T>& M, const T* b, T* x,
                                                           const KrylovOptions& o)
{
    const int m = std::min(o.restart, restart_);
    if (m < 1) return std::unexpected("fgmres: restart < 1 or no basis allocated");
    const std::size_t N = N_;
    T* r = vec(0);
    T* w = vec(1);
    auto V = [&](int i) { return basis_.data() + std::size_t(i) * N; };
    auto Z = [&](int i) { return basis_.data() + std::size_t(restart_ + 1 + i) * N; };
    KrylovResult res;
    double nrm[kMaxDots];
    auto norm = [&](const T* v) -> std::expected<double, std::string> {
        const DotPair<T> p{v, v};
        if (auto e = dots(std::span(&p, 1), std::span(nrm, 1)); !e) return std::unexpected(e.error());
        return std::sqrt(nrm[0]);
    };
    A(x, w);
    lin<T>(N, 1.0, b, -1.0, w, r);
    auto beta = norm(r);
    if (!beta) return std::unexpected(beta.error());
    if (!std::isfinite(*beta)) return nonFinite("fgmres");
    res.r0 = res.residual = *beta;
    const double target = std::max(o.relTol * res.r0, o.absTol);
    if (res.r0 <= target) {
        res.converged = true;
        return res;
    }
    const std::size_t M1 = std::size_t(m);
    std::vector<double> H((M1 + 1) * M1), g(M1 + 1), cs(M1), sn(M1), y(M1);
    auto h = [&](int i, int j) -> double& { return H[std::size_t(i * m + j)]; };
    std::array<DotPair<T>, kMaxDots> pairs{};
    double h1[kMaxDots], h2[kMaxDots];
    double b0 = *beta;
    while (true) {
        lin<T>(N, 1.0 / b0, r, 0.0, r, V(0));
        std::ranges::fill(g, 0.0);
        g[0] = b0;
        int j = 0;
        double est = b0;
        while (j < m) {
            precondition(M, V(j), Z(j), N);
            A(Z(j), w);
            for (int i = 0; i <= j; ++i) pairs[std::size_t(i)] = {V(i), w};
            if (auto e = dots(std::span(pairs.data(), std::size_t(j + 1)), std::span(h1)); !e)
                return std::unexpected(e.error());
            combine<T>(N, j + 1, V(0), h1, -1.0, w);
            pairs[std::size_t(j + 1)] = {w, w};
            if (auto e = dots(std::span(pairs.data(), std::size_t(j + 2)), std::span(h2)); !e)
                return std::unexpected(e.error());
            combine<T>(N, j + 1, V(0), h2, -1.0, w);
            const double nw = h2[j + 1];
            double hn2 = nw;
            for (int i = 0; i <= j; ++i) {
                h(i, j) = h1[i] + h2[i];
                hn2 -= h2[i] * h2[i];
            }
            if (hn2 <= 1e-2 * nw) { // cancellation: measure the new vector directly
                auto d = norm(w);
                if (!d) return std::unexpected(d.error());
                hn2 = *d * *d;
            }
            const double hn = std::sqrt(std::max(hn2, 0.0));
            h(j + 1, j) = hn;
            for (int i = 0; i < j; ++i) {
                const double t = cs[std::size_t(i)] * h(i, j) + sn[std::size_t(i)] * h(i + 1, j);
                h(i + 1, j) = -sn[std::size_t(i)] * h(i, j) + cs[std::size_t(i)] * h(i + 1, j);
                h(i, j) = t;
            }
            const double d = std::hypot(h(j, j), hn);
            if (!(d > 0) || !std::isfinite(d)) return nonFinite("fgmres");
            cs[std::size_t(j)] = h(j, j) / d;
            sn[std::size_t(j)] = hn / d;
            h(j, j) = d;
            h(j + 1, j) = 0;
            g[std::size_t(j + 1)] = -sn[std::size_t(j)] * g[std::size_t(j)];
            g[std::size_t(j)] *= cs[std::size_t(j)];
            est = std::abs(g[std::size_t(j + 1)]);
            ++res.iterations;
            ++j;
            if (!std::isfinite(est)) return nonFinite("fgmres");
            if (est <= target || res.iterations >= o.maxIter || hn == 0) break;
            if (j < m) lin<T>(N, 1.0 / hn, w, 0.0, w, V(j));
        }
        for (int i = j - 1; i >= 0; --i) {
            double s = g[std::size_t(i)];
            for (int k = i + 1; k < j; ++k) s -= h(i, k) * y[std::size_t(k)];
            y[std::size_t(i)] = s / h(i, i);
        }
        combine<T>(N, j, Z(0), y.data(), 1.0, x);
        res.residual = est;
        if (est <= target || res.iterations >= o.maxIter) break;
        A(x, w);
        lin<T>(N, 1.0, b, -1.0, w, r);
        auto bt = norm(r);
        if (!bt) return std::unexpected(bt.error());
        if (!std::isfinite(*bt)) return nonFinite("fgmres");
        b0 = res.residual = *bt;
        if (b0 <= target) break;
    }
    res.relResidual = res.residual / res.r0;
    res.converged = res.residual <= target;
    if (const hipError_t e = hipGetLastError(); e != hipSuccess) return std::unexpected(hipText("fgmres", e));
    return res;
}

// PCG (masterplan_performance step 3): the pre-change iterate, value for value,
// with (a) the start's A x0 and normFactor terms riding the first batch,
// (b) r0 tested before the first M apply when minIter == 0, (c) alpha on the
// device: (p, q) -> kAlpha -> kLinDev, (p, q) read with the next batch, so one
// host sync per iteration. On a breakdown / non-finite (p, q) the device
// updates were skipped, so x and r are exactly where the old loop left them.
template <class T>
std::expected<KrylovResult, std::string> Krylov<T>::pcg(const LinOp<T>& A, const LinOp<T>& M, const T* b, T* x,
                                                        const KrylovOptions& o, const KrylovStart<T>& st)
{
    const std::size_t N = N_;
    T *r = vec(0), *z = vec(1), *p = vec(2), *q = vec(3), *zOld = vec(4);
    KrylovResult res;
    const double* d = host_;
    // OpenFOAM's measure (normFactor > 0): pair 1 is the L1 sum of r.
    double nf = o.normFactor;
    const bool l1 = nf > 0 || st.normTerms;
    const T* rr = l1 ? nullptr : r;
    auto measure = [&](double v) { return l1 ? v / nf : std::sqrt(v); };
    if (st.Ax) lin<T>(N, 1.0, b, -1.0, st.Ax, r);
    else {
        A(x, q);
        lin<T>(N, 1.0, b, -1.0, q, r);
    }
    double rzv = 0;
    const bool early = o.minIter == 0 && o.r0First;
    if (early) { // r0 (and normFactor) first; M only when not converged
        const DotPair<T> p0[2] = {{r, rr}, {st.normTerms, nullptr}};
        if (auto e = batch(std::span(p0, st.normTerms ? 2 : 1)); !e) return std::unexpected(e.error());
        auto fl = takePreFlag();
        if (!fl) return std::unexpected(fl.error());
        if (*fl) {
            res.flagged = true;
            return res;
        }
        if (st.normTerms) nf = d[1] + 1e-20;
        res.r0 = res.residual = measure(d[0]);
        if (!std::isfinite(res.r0)) return nonFinite("pcg");
    }
    const double target0 = early ? std::max(o.relTol * res.r0, o.absTol) : 0;
    if (!early || res.r0 > target0) {
        precondition(M, r, z, N);
        const DotPair<T> p1[3] = {{r, z}, {r, rr}, {st.normTerms, nullptr}};
        const std::size_t n1 = early ? 1 : st.normTerms ? 3 : 2;
        if (auto e = batch(std::span(p1, n1)); !e) return std::unexpected(e.error());
        if (!early) {
            auto fl = takePreFlag();
            if (!fl) return std::unexpected(fl.error());
            if (*fl) {
                res.flagged = true;
                return res;
            }
            if (st.normTerms) nf = d[2] + 1e-20;
            res.r0 = res.residual = measure(d[1]);
        }
        rzv = d[0];
        if (!std::isfinite(res.r0) || !std::isfinite(rzv)) return nonFinite("pcg");
        (void)hipMemcpyAsync(p, z, N * sizeof(T), hipMemcpyDeviceToDevice, 0);
    }
    const double target = std::max(o.relTol * res.r0, o.absTol);
    const internal::DotArgs<T> pqArgs = [&] {
        internal::DotArgs<T> a{};
        a.a[0] = p;
        a.b[0] = q;
        return a;
    }();
    while ((res.residual > target || res.iterations < o.minIter) && res.iterations < o.maxIter) {
        A(p, q);
        internal::deviceDots(N, pqArgs, 1, partial_.data(), out_.data() + kSlot);
        kAlpha<<<1, 1>>>(rzv, out_.data() + kSlot, 1, coef_.data());
        kLinDev<T><<<gridFor(N), kThreads>>>(N, 1.0, x, coef_.data(), 0, p, 0.0, nullptr, x);
        kLinDev<T><<<gridFor(N), kThreads>>>(N, 1.0, r, coef_.data(), 1, q, 0.0, nullptr, r);
        T *zn = z, *zo = zOld;
        if (o.flexible) std::swap(zn, zo); // keep z_k for (r_{k+1}, z_k): swap the two z buffers
        precondition(M, r, zn, N);
        const DotPair<T> rzn[3] = {{r, zn}, {r, rr}, {r, zo}};
        const std::size_t nd = o.flexible ? 3 : 2;
        if (auto e = batch(std::span(rzn, nd)); !e) return std::unexpected(e.error());
        const double pqv = d[kSlot];
        if (!std::isfinite(pqv)) return nonFinite("pcg");
        if (!(pqv > 0)) {
            res.breakdown = true; // A or M not positive definite (or p == 0)
            break;
        }
        z = zn;
        zOld = zo;
        ++res.iterations;
        res.residual = measure(d[1]);
        if (!std::isfinite(res.residual) || !std::isfinite(d[0])) return nonFinite("pcg");
        const double beta = o.flexible ? (d[0] - d[2]) / rzv : d[0] / rzv;
        rzv = d[0];
        lin<T>(N, 1.0, z, beta, p, p);
    }
    res.relResidual = res.r0 > 0 ? res.residual / res.r0 : 0.0;
    res.converged = res.residual <= target;
    if (const hipError_t e = hipGetLastError(); e != hipSuccess) return std::unexpected(hipText("pcg", e));
    return res;
}

// BiCGStab (masterplan_performance step 3): the pre-change iterate, value for
// value, with the start's A x0 / normFactor terms in the first batch and alpha
// = rho / (rh, v) on the device: (rh, v) rides the ts batch, two host syncs
// per iteration. A breakdown / non-finite (rh, v) skipped the device update
// of s, and the host takes the old branch before any later value is used.
template <class T>
std::expected<KrylovResult, std::string> Krylov<T>::bicgstab(const LinOp<T>& A, const LinOp<T>& M, const T* b,
                                                             T* x, const KrylovOptions& o, const KrylovStart<T>& st)
{
    const std::size_t N = N_;
    T *r = vec(0), *rh = vec(1), *p = vec(2), *v = vec(3), *ph = vec(4), *s = vec(5), *sh = vec(6), *t = vec(7);
    KrylovResult res;
    const double* d = host_;
    // OpenFOAM's measure (normFactor > 0) rides one extra L1 pair per batch;
    // the 2-norms stay for the shadow-restart test.
    double nf = o.normFactor;
    const bool l1 = nf > 0 || st.normTerms;
    if (st.Ax) lin<T>(N, 1.0, b, -1.0, st.Ax, r);
    else {
        A(x, v);
        lin<T>(N, 1.0, b, -1.0, v, r);
    }
    (void)hipMemcpyAsync(rh, r, N * sizeof(T), hipMemcpyDeviceToDevice, 0);
    const DotPair<T> rr[3] = {{r, r}, {r, nullptr}, {st.normTerms, nullptr}};
    if (auto e = batch(std::span(rr, st.normTerms ? 3 : l1 ? 2 : 1)); !e) return std::unexpected(e.error());
    auto fl = takePreFlag();
    if (!fl) return std::unexpected(fl.error());
    if (*fl) {
        res.flagged = true;
        return res;
    }
    if (st.normTerms) nf = d[2] + 1e-20;
    double r2 = std::sqrt(d[0]);
    res.r0 = res.residual = l1 ? d[1] / nf : r2;
    if (!std::isfinite(res.r0) || !std::isfinite(r2)) return nonFinite("bicgstab");
    const double target = std::max(o.relTol * res.r0, o.absTol);
    double rho = 1, alpha = 1, omega = 1, rhoNew = d[0], rhNorm = r2;
    internal::DotArgs<T> rvArgs{};
    rvArgs.a[0] = rh;
    rvArgs.b[0] = v;
    bool fresh = true;
    while ((res.residual > target || res.iterations < o.minIter) && res.iterations < o.maxIter) {
        if (std::abs(rhoNew) <= kShadowRestart * rhNorm * r2) {
            // Lanczos breakdown (r ~ orthogonal to the shadow): restart the
            // shadow residual at the current r (measured: rho -> 1e-18 |rh||r|).
            (void)hipMemcpyAsync(rh, r, N * sizeof(T), hipMemcpyDeviceToDevice, 0);
            rhNorm = r2;
            rhoNew = r2 * r2;
            fresh = true;
            ++res.restarts;
        }
        if (fresh) (void)hipMemcpyAsync(p, r, N * sizeof(T), hipMemcpyDeviceToDevice, 0);
        else {
            const double beta = (rhoNew / rho) * (alpha / omega);
            lin<T>(N, 1.0, r, beta, p, p, -beta * omega, v);
        }
        rho = rhoNew;
        fresh = false;
        precondition(M, p, ph, N);
        A(ph, v);
        internal::deviceDots(N, rvArgs, 1, partial_.data(), out_.data() + kSlot);
        kAlpha<<<1, 1>>>(rho, out_.data() + kSlot, 0, coef_.data());
        kLinDev<T><<<gridFor(N), kThreads>>>(N, 1.0, r, coef_.data(), 1, v, 0.0, nullptr, s);
        precondition(M, s, sh, N);
        A(sh, t);
        const DotPair<T> ts[4] = {{t, s}, {t, t}, {s, s}, {s, nullptr}};
        if (auto e = batch(std::span(ts, l1 ? 4 : 3)); !e) return std::unexpected(e.error());
        const double rvv = d[kSlot];
        if (!std::isfinite(rvv)) return nonFinite("bicgstab");
        if (rvv == 0) {
            res.breakdown = true;
            break;
        }
        alpha = rho / rvv;
        if (!std::isfinite(d[0] + d[1] + d[2])) return nonFinite("bicgstab");
        ++res.iterations;
        const double sMeasure = l1 ? d[3] / nf : std::sqrt(d[2]);
        if (sMeasure <= target && res.iterations >= o.minIter) { // converged at the half step
            lin<T>(N, 1.0, x, alpha, ph, x);
            res.residual = sMeasure;
            break;
        }
        if (d[1] == 0) {
            res.breakdown = true;
            break;
        }
        omega = d[0] / d[1];
        lin<T>(N, 1.0, x, alpha, ph, x, omega, sh);
        lin<T>(N, 1.0, s, -omega, t, r);
        const DotPair<T> rn[3] = {{rh, r}, {r, r}, {r, nullptr}};
        if (auto e = batch(std::span(rn, l1 ? 3 : 2)); !e) return std::unexpected(e.error());
        rhoNew = d[0];
        r2 = std::sqrt(d[1]);
        res.residual = l1 ? d[2] / nf : r2;
        if (!std::isfinite(res.residual) || !std::isfinite(r2)) return nonFinite("bicgstab");
        if (omega == 0) {
            res.breakdown = true;
            break;
        }
    }
    res.relResidual = res.r0 > 0 ? res.residual / res.r0 : 0.0;
    res.converged = res.residual <= target;
    if (const hipError_t e = hipGetLastError(); e != hipSuccess) return std::unexpected(hipText("bicgstab", e));
    return res;
}

// bicgstab, nb systems in lockstep (masterplan_performance step 7). Every
// system runs the solo loop above value for value: the same kernels' bodies
// (kLinBatch = kLin, kLinDevBatch = kLinDev, kAlphaBatch = kAlpha per system),
// its own pairs in each batch (a pair's result does not depend on the batch:
// fixed partials per N), its own host scalars, branches and stop. A system
// that stops (converged, breakdown, maxIter, failure) is masked out of every
// later update; A and M still run on all nb, but only its scratch sees them.
template <class T>
std::expected<typename Krylov<T>::BatchResult, std::string>
Krylov<T>::bicgstabBatch(int nb, unsigned mask, const LinOp<T>& A, const LinOp<T>& M, const T* b, T* x,
                         const KrylovOptions& o, const KrylovStart<T>& st)
{
    if (nb < 1 || nb > batch_) return std::unexpected("bicgstab: batch of " + std::to_string(nb) + " systems");
    const std::size_t N = N_, NB = N * std::size_t(nb);
    auto vec = [&](int i) { return aux_.data() + std::size_t(i) * NB; };
    T *r = vec(0), *rh = vec(1), *p = vec(2), *v = vec(3), *ph = vec(4), *s = vec(5), *sh = vec(6), *t = vec(7);
    auto at = [&](const T* base, int j) { return base + std::size_t(j) * N; };
    const double* d = host_;
    const bool l1 = o.normFactor > 0 || st.normTerms;
    struct Sys
    {
        KrylovResult res;
        double nf = 0, target = 0, rho = 1, alpha = 1, omega = 1, rhoNew = 0, rhNorm = 0, r2 = 0;
        bool fresh = true;
    };
    std::array<Sys, kMaxBatch> S{};
    BatchResult out;
    unsigned active = 0;
    for (int j = 0; j < nb; ++j)
        if ((mask >> j) & 1u) active |= 1u << j;
    auto on = [&](int j) { return ((active >> j) & 1u) != 0; };
    auto finish = [&](int j) { // the solo epilogue
        KrylovResult& R = S[std::size_t(j)].res;
        R.relResidual = R.r0 > 0 ? R.residual / R.r0 : 0.0;
        R.converged = R.residual <= S[std::size_t(j)].target;
        out[std::size_t(j)] = R;
        active &= ~(1u << j);
    };
    auto fail = [&](int j) {
        out[std::size_t(j)] = nonFinite("bicgstab");
        active &= ~(1u << j);
    };
    auto cont = [&](int j) {
        const Sys& q = S[std::size_t(j)];
        return (q.res.residual > q.target || q.res.iterations < o.minIter) && q.res.iterations < o.maxIter;
    };
    auto lin = [&](const LinBatch& L, const T* xx, const T* yy, const T* zz, T* oo) {
        if (std::ranges::any_of(L.mode, [](int m) { return m != 0; }))
            kLinBatch<T><<<dim3(gridFor(N), unsigned(nb)), kThreads>>>(N, L, xx, yy, zz, oo);
    };
    // pairs of the active systems; base[j]: system j's first result
    std::array<DotPair<T>, kMaxDots> pr{};
    std::array<int, kMaxBatch> base{};
    auto run = [&](auto&& pairsOf) -> std::expected<void, std::string> {
        int K = 0;
        for (int j = 0; j < nb; ++j) {
            if (!on(j)) continue;
            base[std::size_t(j)] = K;
            K += pairsOf(j, pr.data() + K);
        }
        return batch(std::span(pr.data(), std::size_t(K)));
    };

    LinBatch L0, Lc;
    for (int j = 0; j < nb; ++j)
        if (on(j)) {
            L0.mode[j] = 2;
            L0.a[j] = 1.0;
            L0.b[j] = -1.0;
            Lc.mode[j] = 1;
        }
    if (st.Ax) lin(L0, b, st.Ax, nullptr, r);
    else {
        A(x, v);
        lin(L0, b, v, nullptr, r);
    }
    lin(Lc, r, nullptr, nullptr, rh);
    if (auto e = run([&](int j, DotPair<T>* q) {
            q[0] = {at(r, j), at(r, j)};
            q[1] = {at(r, j), nullptr};
            q[2] = {st.normTerms ? at(st.normTerms, j) : nullptr, nullptr};
            return st.normTerms ? 3 : l1 ? 2 : 1;
        });
        !e)
        return std::unexpected(e.error());
    for (int j = 0; j < nb; ++j) {
        if (!on(j)) continue;
        Sys& q = S[std::size_t(j)];
        const double* dj = d + base[std::size_t(j)];
        auto fl = takePreFlag(j);
        if (!fl) return std::unexpected(fl.error());
        if (*fl) {
            KrylovResult R;
            R.flagged = true;
            out[std::size_t(j)] = R;
            active &= ~(1u << j);
            continue;
        }
        q.nf = st.normTerms ? dj[2] + 1e-20 : o.normFactor;
        q.r2 = std::sqrt(dj[0]);
        q.res.r0 = q.res.residual = l1 ? dj[1] / q.nf : q.r2;
        if (!std::isfinite(q.res.r0) || !std::isfinite(q.r2)) {
            fail(j);
            continue;
        }
        q.target = std::max(o.relTol * q.res.r0, o.absTol);
        q.rhoNew = dj[0];
        q.rhNorm = q.r2;
        if (!cont(j)) finish(j);
    }
    internal::DotArgs<T> rvArgs{};
    for (int j = 0; j < nb; ++j) {
        rvArgs.a[j] = at(rh, j);
        rvArgs.b[j] = at(v, j);
    }
    while (active) {
        LinBatch Lr, Lp;
        NumBatch rhoB;
        bool restart = false;
        for (int j = 0; j < nb; ++j) {
            if (!on(j)) continue;
            Sys& q = S[std::size_t(j)];
            if (std::abs(q.rhoNew) <= kShadowRestart * q.rhNorm * q.r2) {
                Lr.mode[j] = 1;
                restart = true;
                q.rhNorm = q.r2;
                q.rhoNew = q.r2 * q.r2;
                q.fresh = true;
                ++q.res.restarts;
            }
            if (q.fresh) Lp.mode[j] = 1;
            else {
                const double beta = (q.rhoNew / q.rho) * (q.alpha / q.omega);
                Lp.mode[j] = 3;
                Lp.a[j] = 1.0;
                Lp.b[j] = beta;
                Lp.c[j] = -beta * q.omega;
            }
            q.rho = q.rhoNew;
            q.fresh = false;
            rhoB.v[j] = q.rho;
        }
        if (restart) lin(Lr, r, nullptr, nullptr, rh);
        lin(Lp, r, p, v, p);
        precondition(M, p, ph, NB);
        A(ph, v);
        internal::deviceDots(N, rvArgs, nb, partial_.data(), out_.data() + kSlot);
        kAlphaBatch<<<1, kMaxBatch>>>(active, rhoB, out_.data() + kSlot, 0, coef_.data());
        kLinDevBatch<T><<<dim3(gridFor(N), unsigned(nb)), kThreads>>>(N, active, 1.0, r, coef_.data(), 1, v, s);
        precondition(M, s, sh, NB);
        A(sh, t);
        if (auto e = run([&](int j, DotPair<T>* q) {
                q[0] = {at(t, j), at(s, j)};
                q[1] = {at(t, j), at(t, j)};
                q[2] = {at(s, j), at(s, j)};
                q[3] = {at(s, j), nullptr};
                return l1 ? 4 : 3;
            });
            !e)
            return std::unexpected(e.error());
        LinBatch Lx, Lr2;
        unsigned stop = 0; // systems ending after this x update
        for (int j = 0; j < nb; ++j) {
            if (!on(j)) continue;
            Sys& q = S[std::size_t(j)];
            const double* dj = d + base[std::size_t(j)];
            const double rvv = d[kSlot + j];
            if (!std::isfinite(rvv)) {
                fail(j);
                continue;
            }
            if (rvv == 0) {
                q.res.breakdown = true;
                finish(j);
                continue;
            }
            q.alpha = q.rho / rvv;
            if (!std::isfinite(dj[0] + dj[1] + dj[2])) {
                fail(j);
                continue;
            }
            ++q.res.iterations;
            const double sMeasure = l1 ? dj[3] / q.nf : std::sqrt(dj[2]);
            if (sMeasure <= q.target && q.res.iterations >= o.minIter) { // converged at the half step
                Lx.mode[j] = 2;
                Lx.a[j] = 1.0;
                Lx.b[j] = q.alpha;
                q.res.residual = sMeasure;
                stop |= 1u << j;
                continue;
            }
            if (dj[1] == 0) {
                q.res.breakdown = true;
                finish(j);
                continue;
            }
            q.omega = dj[0] / dj[1];
            Lx.mode[j] = 3;
            Lx.a[j] = 1.0;
            Lx.b[j] = q.alpha;
            Lx.c[j] = q.omega;
            Lr2.mode[j] = 2;
            Lr2.a[j] = 1.0;
            Lr2.b[j] = -q.omega;
        }
        lin(Lx, x, ph, sh, x);
        lin(Lr2, s, t, nullptr, r);
        for (int j = 0; j < nb; ++j)
            if ((stop >> j) & 1u) finish(j);
        if (!active) break;
        if (auto e = run([&](int j, DotPair<T>* q) {
                q[0] = {at(rh, j), at(r, j)};
                q[1] = {at(r, j), at(r, j)};
                q[2] = {at(r, j), nullptr};
                return l1 ? 3 : 2;
            });
            !e)
            return std::unexpected(e.error());
        for (int j = 0; j < nb; ++j) {
            if (!on(j)) continue;
            Sys& q = S[std::size_t(j)];
            const double* dj = d + base[std::size_t(j)];
            q.rhoNew = dj[0];
            q.r2 = std::sqrt(dj[1]);
            q.res.residual = l1 ? dj[2] / q.nf : q.r2;
            if (!std::isfinite(q.res.residual) || !std::isfinite(q.r2)) {
                fail(j);
                continue;
            }
            if (q.omega == 0) {
                q.res.breakdown = true;
                finish(j);
                continue;
            }
            if (!cont(j)) finish(j);
        }
    }
    if (const hipError_t e = hipGetLastError(); e != hipSuccess) return std::unexpected(hipText("bicgstab", e));
    return out;
}

template class Krylov<float>;
template class Krylov<double>;

} // namespace dns::gpu::la
