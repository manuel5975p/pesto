// gpu/la performance kernels vs the PRE-CHANGE kernels (masterplan_performance,
// docs/la.md "Performance kernels"): every kernel below is the library's
// kernel as it was before the prefetch / fusion / register rewrites, kept
// verbatim as an oracle. The library must reproduce each one BIT FOR BIT
// (memcmp, float and double) on structured and polyhedral graphs:
//   spmv / residual (every compiled shape; prefetched at B = 1),
//   multicolour GS sweeps, smooth, apply (prefetched at B = 1),
//   block-Jacobi smooth (fused ping-pong sweep, 1..3 steps) and smoothFromZero
//   (b with +-0 and exact zeros),
//   the AMG preconditioner (zero-start Jacobi, fused sweeps, scaled-V folded
//   prolongation, register-resident dense coarsest inverse, one-download
//   refresh flags) against the old V / scaled-V / K cycle, pivot ties and the
//   null space included -- and the singular-coarsest error,
//   pcg / bicgstab (device alpha, KrylovStart, r0First, breakdown, maxIter,
//   minIter, the pre-flag) against the old two/three-sync loops,
//   bicgstabBatch (+ spmvMulti, MC-GS factorMulti / applyMulti) against one
//   bicgstab per system.
// A HIP TU (it defines kernels), like test_gpu_tunnel_bc.
#include "gpu_test_util.hpp"
#include "la_test_util.hpp"

#include "dns/facemesh.hpp"
#include "dns/gpu/la/amg.hpp"
#include "dns/gpu/la/blockell.hpp"
#include "dns/gpu/la/krylov.hpp"
#include "dns/gpu/la/internal/dots.hpp"
#include "dns/gpu/la/smooth.hpp"
#include "dns/la/amg.hpp"
#include "dns/la/graph.hpp"
#include "dns/la/structured.hpp"
#include "dns/polymesh.hpp"

#include <cstring>
#include <filesystem>
#include <numeric>
#include <optional>
#include <string>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dns;
using namespace dns::la;
using namespace dns::gpu::la;
using dnstest::check;
namespace fs = std::filesystem;

namespace {

constexpr int kT = 256;
unsigned grid(std::size_t n) { return unsigned((n + 255) / 256); }

// ===== the pre-change kernels, verbatim =====================================

template <int B, int Bd, class T>
__global__ void __launch_bounds__(256) kSpmvRef(EllView<T> A, const T* __restrict__ x, const T* b, T* y)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= A.n) return;
    const std::size_t n = std::size_t(A.n), nE = std::size_t(A.nE);
    T xc[B], acc[B];
#pragma unroll
    for (int q = 0; q < B; ++q) xc[q] = x[std::size_t(q) * n + std::size_t(c)];
#pragma unroll
    for (int r = 0; r < B; ++r) {
        T s = T(0);
#pragma unroll
        for (int q = 0; q < B; ++q) s += A.diag[std::size_t(r * B + q) * n + std::size_t(c)] * xc[q];
        acc[r] = s;
    }
    const int sl = c / dns::la::kSliceRows, lane = c % dns::la::kSliceRows;
    const int w = A.sliceWidth[sl];
    int E = A.sliceBase[sl] + lane;
    for (int k = 0; k < w; ++k, E += dns::la::kSliceRows) {
        const int j = A.col[E];
        if (j < 0) break;
        T xj[B];
#pragma unroll
        for (int q = 0; q < B; ++q) xj[q] = x[std::size_t(q) * n + std::size_t(j)];
#pragma unroll
        for (int r = 0; r < Bd; ++r)
#pragma unroll
            for (int q = 0; q < Bd; ++q) acc[r] += A.off[std::size_t(r * Bd + q) * nE + std::size_t(E)] * xj[q];
#pragma unroll
        for (int r = Bd; r < B; ++r)
            acc[r] += A.sc[std::size_t(A.shape.nS == 1 ? 0 : r - Bd) * nE + std::size_t(E)] * xj[r];
    }
#pragma unroll
    for (int r = 0; r < B; ++r) {
        const std::size_t i = std::size_t(r) * n + std::size_t(c);
        y[i] = b ? b[i] - acc[r] : acc[r];
    }
}

template <int B, class T>
__global__ void kDiagApplyRef(int n, const T* __restrict__ inv, const T* __restrict__ r, T* __restrict__ z,
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

template <int B, int Bd, class T>
__global__ void __launch_bounds__(256) kGsColourRef(EllView<T> A, const int* __restrict__ rows, int m,
                                                    const T* __restrict__ inv, const T* __restrict__ b, T* x,
                                                    T omega)
{
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= m) return;
    const int c = rows[i];
    const std::size_t n = std::size_t(A.n), nE = std::size_t(A.nE);
    T acc[B];
#pragma unroll
    for (int r = 0; r < B; ++r) acc[r] = b[std::size_t(r) * n + std::size_t(c)];
    const int sl = c / dns::la::kSliceRows, lane = c % dns::la::kSliceRows;
    const int w = A.sliceWidth[sl];
    int E = A.sliceBase[sl] + lane;
    for (int k = 0; k < w; ++k, E += dns::la::kSliceRows) {
        const int j = A.col[E];
        if (j < 0) break;
        T xj[B];
#pragma unroll
        for (int q = 0; q < B; ++q) xj[q] = x[std::size_t(q) * n + std::size_t(j)];
#pragma unroll
        for (int r = 0; r < Bd; ++r)
#pragma unroll
            for (int q = 0; q < Bd; ++q) acc[r] -= A.off[std::size_t(r * Bd + q) * nE + std::size_t(E)] * xj[q];
#pragma unroll
        for (int r = Bd; r < B; ++r)
            acc[r] -= A.sc[std::size_t(A.shape.nS == 1 ? 0 : r - Bd) * nE + std::size_t(E)] * xj[r];
    }
#pragma unroll
    for (int p = 0; p < B; ++p) {
        T s = T(0);
#pragma unroll
        for (int q = 0; q < B; ++q) s += inv[std::size_t(p * B + q) * n + std::size_t(c)] * acc[q];
        const std::size_t ix = std::size_t(p) * n + std::size_t(c);
        x[ix] = omega == T(1) ? s : (T(1) - omega) * x[ix] + omega * s;
    }
}

template <class T>
__global__ void kRestrictRef(int nc, int B, int nf, const int* __restrict__ memberPtr, const int* __restrict__ member,
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

template <class T>
__global__ void kProlongAddRef(int nf, int B, int nc, const int* __restrict__ agg, const T* __restrict__ e,
                               T* __restrict__ x)
{
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i >= nf) return;
    const std::size_t I = std::size_t(agg[i]);
    for (int q = 0; q < B; ++q)
        x[std::size_t(q) * std::size_t(nf) + std::size_t(i)] += e[std::size_t(q) * std::size_t(nc) + I];
}

__global__ void kCoef1Ref(const double* s, double* coef)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const double a = s[0] > 0.0 ? s[1] / s[0] : 0.0;
    coef[0] = isfinite(a) ? a : 0.0;
}

__global__ void kCoef2Ref(const double* s, double* coef)
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

template <class T>
__global__ void kSubScaledRef(std::size_t N, const T* b, const double* c, const T* v, T* out)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i < N) out[i] = T(double(b[i]) - c[0] * double(v[i]));
}

template <class T>
__global__ void kScaledRef(std::size_t N, const double* c, const T* x, T* out)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i < N) out[i] = T(c[0] * double(x[i]));
}

template <class T>
__global__ void kComb2Ref(std::size_t N, const double* c, const T* x, const T* y, T* out)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i < N) out[i] = T(c[1] * double(x[i]) + c[2] * double(y[i]));
}

template <int B, int Bd, class T>
__global__ void kDenseBuildRef(EllView<T> A, double* M)
{
    const int c = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (c >= A.n) return;
    const std::size_t n = std::size_t(A.n), m = n * B;
    double D[B][B];
    dns::gpu::la::internal::diagBlock<B>(A, c, D);
#pragma unroll
    for (int r = 0; r < B; ++r)
#pragma unroll
        for (int q = 0; q < B; ++q) M[(std::size_t(r) * n + std::size_t(c)) * m + std::size_t(q) * n + std::size_t(c)] = D[r][q];
    const int sl = c / dns::la::kSliceRows, lane = c % dns::la::kSliceRows;
    int E = A.sliceBase[sl] + lane;
    for (int k = 0; k < A.sliceWidth[sl]; ++k, E += dns::la::kSliceRows) {
        const int j = A.col[E];
        if (j < 0) break;
        dns::gpu::la::internal::offBlock<B, Bd>(A, E, D);
#pragma unroll
        for (int r = 0; r < B; ++r)
#pragma unroll
            for (int q = 0; q < B; ++q)
                M[(std::size_t(r) * n + std::size_t(c)) * m + std::size_t(q) * n + std::size_t(j)] = D[r][q];
    }
}

constexpr int kDenseThreads = 1024;

template <class T>
__global__ void __launch_bounds__(kDenseThreads) kDenseInvertRef(int m, int B, int nullSpace, double* M, double* X,
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
    if (nonFinite) bad[0] = 1;
}

template <class T>
__global__ void kDenseApplyRef(int m, const T* __restrict__ Xt, const T* __restrict__ b, T* __restrict__ x)
{
    const int r = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (r >= m) return;
    double s = 0.0;
    for (int q = 0; q < m; ++q) s += double(Xt[std::size_t(q) * std::size_t(m) + std::size_t(r)]) * double(b[q]);
    x[r] = T(s);
}

template <class T>
__global__ void kLinRef(std::size_t N, double a, const T* x, double b, const T* y, double c, const T* z, T* out)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i >= N) return;
    double s = a * double(x[i]) + b * double(y[i]);
    if (z) s += c * double(z[i]);
    out[i] = T(s);
}

__global__ void kSetFlag(int* f) { f[0] = 1; }

template <class T>
__global__ void kRemoveMeanRef(int n, int B, const double* sum, const T* x, T* y)
{
    const std::size_t i = std::size_t(blockIdx.x) * kT + threadIdx.x;
    if (i >= std::size_t(n) * std::size_t(B)) return;
    y[i] = T(double(x[i]) - sum[i / std::size_t(n)] / double(n));
}

// ===== helpers ===============================================================

template <class E>
auto orDie(E&& e, const std::string& what)
{
    if (!e) {
        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(e.error())>, hipError_t>)
            std::printf("FATAL %s: %s\n", what.c_str(), hipGetErrorString(e.error()));
        else std::printf("FATAL %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    if constexpr (!std::is_void_v<typename std::remove_cvref_t<E>::value_type>) return std::move(*e);
}

template <class U>
gpu::DeviceBuffer<U> devOf(const std::vector<U>& h)
{
    auto b = orDie(gpu::DeviceBuffer<U>::make(std::max<std::size_t>(1, h.size())), "alloc");
    if (!h.empty() && b.upload(h.data()) != hipSuccess) std::exit(1);
    return b;
}

template <class T>
gpu::DeviceBuffer<T> devVec(const std::vector<double>& h)
{
    return devOf(std::vector<T>(h.begin(), h.end()));
}

template <class T>
std::vector<unsigned char> bytes(const T* d, std::size_t n)
{
    std::vector<unsigned char> b(n * sizeof(T));
    if (n && hipMemcpy(b.data(), d, b.size(), hipMemcpyDeviceToHost) != hipSuccess) std::exit(1);
    return b;
}

template <class T>
void copyDev(T* dst, const T* src, std::size_t n)
{
    if (hipMemcpy(dst, src, n * sizeof(T), hipMemcpyDeviceToDevice) != hipSuccess) std::exit(1);
}

template <class T>
const char* tname() { return sizeof(T) == 4 ? "float" : "double"; }

std::string shapeName(BlockShape s)
{
    return "B" + std::to_string(s.B) + "/" + std::to_string(s.Bd) + "/" + std::to_string(s.nS);
}

std::vector<BlockShape> allShapes()
{
    std::vector<BlockShape> s;
    for (const int B : supportedBlockSizes()) {
        s.push_back(*denseShape(B));
        if (B > 5) s.push_back(*blockShape(B, 5, true));
    }
    s.push_back(*blockShape(11, 5, false));
    return s;
}

CellGraph meshGraph(const char* name)
{
    const PolyMesh pm = orDie(readPolyMesh(fs::path(DNS_MESH_DATA) / name / "constant/polyMesh"), name);
    return graphOf(orDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), name));
}

// A vector with exact +0, -0 and random entries.
std::vector<double> signedZeros(std::vector<double> v)
{
    for (std::size_t i = 0; i < v.size(); ++i)
        if (i % 7 == 1) v[i] = -0.0;
        else if (i % 7 == 4) v[i] = 0.0;
    return v;
}

// ===== kernels: spmv, residual, Jacobi, MC-GS ===============================

template <class T>
void kernelCase(const std::string& gname, const CellGraph& g, BlockShape s, unsigned seed)
{
    const std::string tag = gname + " " + shapeName(s) + " " + tname<T>();
    const BlockMatrix m = latest::roundTo<T>(latest::randomMatrix(g, s, seed));
    auto A = orDie(BlockEll<T>::make(ellPattern(g), s), tag);
    orDie(A.upload(m), tag);
    const std::size_t N = A.size();
    const EllView<T> v = A.view();
    auto x = devVec<T>(latest::randomVec(N, seed + 1)), b = devVec<T>(signedZeros(latest::randomVec(N, seed + 2)));
    auto y = devVec<T>(std::vector<double>(N, 0)), yr = devVec<T>(std::vector<double>(N, 0));
    bool ok = true;
    internal::withShape(s, [&](auto sh) {
        constexpr int B = decltype(sh)::B, Bd = decltype(sh)::Bd;
        // spmv and residual (also r aliasing b)
        A.spmv(x.data(), y.data());
        kSpmvRef<B, Bd, T><<<grid(std::size_t(v.n)), kT>>>(v, x.data(), nullptr, yr.data());
        ok = ok && bytes(y.data(), N) == bytes(yr.data(), N);
        A.residual(b.data(), x.data(), y.data());
        kSpmvRef<B, Bd, T><<<grid(std::size_t(v.n)), kT>>>(v, x.data(), b.data(), yr.data());
        ok = ok && bytes(y.data(), N) == bytes(yr.data(), N);
        copyDev(y.data(), b.data(), N);
        A.residual(y.data(), x.data(), y.data());
        ok = ok && bytes(y.data(), N) == bytes(yr.data(), N);
        check(ok, tag + ": spmv, residual (and r == b) == the pre-change kernel bit for bit");
        // Jacobi: smooth from x and from zero, 1..3 steps, vs residual + kDiagApply
        auto J = orDie(BlockJacobi<T>::make(A), tag);
        orDie(J.factor(), tag);
        auto work = devVec<T>(std::vector<double>(N, 0)), wr = devVec<T>(std::vector<double>(N, 0));
        auto xl = devVec<T>(std::vector<double>(N, 0)), xr = devVec<T>(std::vector<double>(N, 0));
        bool okJ = true;
        for (const double om : {0.67, 1.0})
            for (int sweeps = 1; sweeps <= 3; ++sweeps)
                for (const bool zero : {false, true}) {
                    copyDev(xl.data(), x.data(), N);
                    if (zero) {
                        copyDev(xr.data(), x.data(), N); // stale contents: smoothFromZero must ignore them
                        J.smoothFromZero(b.data(), xr.data(), sweeps, om, work.data());
                        if (hipMemset(xl.data(), 0, N * sizeof(T)) != hipSuccess) std::exit(1);
                    } else {
                        copyDev(xr.data(), x.data(), N);
                        J.smooth(b.data(), xr.data(), sweeps, om, work.data());
                    }
                    for (int k = 0; k < sweeps; ++k) {
                        kSpmvRef<B, Bd, T><<<grid(std::size_t(v.n)), kT>>>(v, xl.data(), b.data(), wr.data());
                        kDiagApplyRef<B, T><<<grid(std::size_t(v.n)), kT>>>(v.n, J.inverse(), wr.data(), xl.data(), 1,
                                                                           T(om));
                    }
                    okJ = okJ && bytes(xr.data(), N) == bytes(xl.data(), N);
                }
        check(okJ, tag + ": Jacobi smooth / smoothFromZero (1..3 steps, omega 0.67 and 1) == memset + residual + "
                         "kDiagApply bit for bit");
        // MC-GS
        const auto colours = greedyColour(g);
        auto G = orDie(MultiColourGS<T>::make(A, colours), tag);
        orDie(G.factor(), tag);
        std::vector<int> rows(std::size_t(g.n));
        std::iota(rows.begin(), rows.end(), 0);
        std::ranges::stable_sort(rows, {}, [&](int c) { return colours[std::size_t(c)]; });
        const int K = G.colours();
        std::vector<int> start(std::size_t(K) + 1, 0);
        for (const std::uint8_t k : colours) ++start[std::size_t(k) + 1];
        std::partial_sum(start.begin(), start.end(), start.begin());
        auto drows = devOf(rows);
        auto J2 = orDie(BlockJacobi<T>::make(A), tag);
        orDie(J2.factor(), tag);
        auto refSweep = [&](T* xx, bool fwd, double om) {
            for (int t = 0; t < K; ++t) {
                const int k = fwd ? t : K - 1 - t;
                const int r0 = start[std::size_t(k)], mm = start[std::size_t(k) + 1] - r0;
                if (mm == 0) continue;
                kGsColourRef<B, Bd, T><<<grid(std::size_t(mm)), kT>>>(v, drows.data() + r0, mm, J2.inverse(), b.data(),
                                                                     xx, T(om));
            }
        };
        bool okG = true;
        for (const double om : {1.0, 0.7})
            for (const bool fwd : {true, false}) {
                copyDev(xl.data(), x.data(), N);
                copyDev(xr.data(), x.data(), N);
                G.sweep(b.data(), xr.data(), fwd, om);
                refSweep(xl.data(), fwd, om);
                okG = okG && bytes(xr.data(), N) == bytes(xl.data(), N);
            }
        copyDev(xl.data(), x.data(), N);
        copyDev(xr.data(), x.data(), N);
        G.smooth(b.data(), xr.data(), 2, 1.0);
        for (int k = 0; k < 2; ++k) {
            refSweep(xl.data(), true, 1.0);
            refSweep(xl.data(), false, 1.0);
        }
        okG = okG && bytes(xr.data(), N) == bytes(xl.data(), N);
        G.apply(b.data(), xr.data());
        if (hipMemset(xl.data(), 0, N * sizeof(T)) != hipSuccess) std::exit(1);
        refSweep(xl.data(), true, 1.0);
        refSweep(xl.data(), false, 1.0);
        okG = okG && bytes(xr.data(), N) == bytes(xl.data(), N);
        // a diagonal-only change: factor(false) keeps the private off-diagonal copy
        BlockMatrix m2 = m;
        for (double& dv : m2.diag) dv = double(T(dv * 1.25));
        orDie(A.upload(m2), tag);
        orDie(G.factor(false), tag);
        orDie(J2.factor(), tag);
        G.apply(b.data(), xr.data());
        if (hipMemset(xl.data(), 0, N * sizeof(T)) != hipSuccess) std::exit(1);
        refSweep(xl.data(), true, 1.0);
        refSweep(xl.data(), false, 1.0);
        okG = okG && bytes(xr.data(), N) == bytes(xl.data(), N);
        check(okG, tag + ": MC-GS sweep (both directions, omega 1 / 0.7), smooth, apply, factor(false) after a "
                         "diagonal-only change == the pre-change kernel bit for bit (" + std::to_string(K) + " colours)");
    });
}

// ===== the AMG cycle, as it was =============================================

template <class T>
struct RefAmg
{
    struct Lv
    {
        const BlockEll<T>* A = nullptr;
        int inner = 0;
        bool coarsest = false;
        gpu::DeviceBuffer<int> memberPtr, member, agg;
        std::optional<BlockJacobi<T>> jac;
        gpu::DeviceBuffer<T> b, x, res, work, c1, v1, rt, d2, v2, Xt;
        gpu::DeviceBuffer<double> partial, s, coef, M, X;
        gpu::DeviceBuffer<int> bad;
        int m = 0;
    };
    std::vector<Lv> lv;
    AmgOptions o;

    // Pre: amg was made from (A0, h, o) with the Jacobi smoother on every level and refreshed.
    // Returns the old refresh's coarsest `bad` flag.
    int build(const Amg<T>& amg, const AmgHierarchy& h, const AmgOptions& opt)
    {
        o = opt;
        const int L = amg.levels();
        lv.resize(std::size_t(L));
        for (int l = 0; l < L; ++l) {
            Lv& V = lv[std::size_t(l)];
            V.A = &amg.matrix(l);
            V.coarsest = l == L - 1;
            V.inner = (l == 0 || V.coarsest || (l > 2 && V.A->rows() < o.innerMinRows)) ? 0 : o.inner;
            const std::size_t N = V.A->size();
            if (l > 0) {
                const AmgTransfer& t = h.transfers[std::size_t(l - 1)];
                V.memberPtr = devOf(t.memberPtr);
                V.member = devOf(t.member);
                V.agg = devOf(t.agg);
            }
            for (auto* p : {&V.b, &V.x, &V.res, &V.work, &V.c1, &V.v1, &V.rt, &V.d2, &V.v2})
                *p = devVec<T>(std::vector<double>(N, 0));
            V.partial = devOf(std::vector<double>(std::size_t(internal::dotParts(N)) * 3, 0));
            V.s = devOf(std::vector<double>(8, 0));
            V.coef = devOf(std::vector<double>(4, 0));
            if (!V.coarsest) {
                V.jac.emplace(orDie(BlockJacobi<T>::make(*V.A), "ref jacobi"));
                orDie(V.jac->factor(), "ref jacobi");
            }
        }
        Lv& C = lv.back();
        C.m = int(C.A->size());
        const std::size_t m = std::size_t(C.m);
        C.M = devOf(std::vector<double>(m * m, 0));
        C.X = devOf(std::vector<double>(m * m, 0));
        C.Xt = devVec<T>(std::vector<double>(m * m, 0));
        C.bad = devOf(std::vector<int>(1, 0));
        const EllView<T> cv = C.A->view();
        internal::withShape(C.A->shape(), [&](auto sh) {
            kDenseBuildRef<decltype(sh)::B, decltype(sh)::Bd, T><<<grid(std::size_t(cv.n)), kT>>>(cv, C.M.data());
        });
        kDenseInvertRef<T><<<1, kDenseThreads>>>(C.m, C.A->shape().B, o.constantNullSpace ? 1 : 0, C.M.data(),
                                                 C.X.data(), C.Xt.data(), C.bad.data());
        int bad = 0;
        if (C.bad.download(&bad) != hipSuccess) std::exit(1);
        return bad;
    }

    void residual(const BlockEll<T>& A, const T* b, const T* x, T* r)
    {
        const EllView<T> v = A.view();
        internal::withShape(A.shape(), [&](auto sh) {
            kSpmvRef<decltype(sh)::B, decltype(sh)::Bd, T><<<grid(std::size_t(v.n)), kT>>>(v, x, b, r);
        });
    }

    void smooth(Lv& V, const T* b, T* x)
    {
        for (int s = 0; s < o.sweeps; ++s) {
            residual(*V.A, b, x, V.work.data());
            internal::withShape(V.A->shape(), [&](auto sh) {
                kDiagApplyRef<decltype(sh)::B, T><<<grid(std::size_t(V.A->rows())), kT>>>(
                    V.A->rows(), V.jac->inverse(), V.work.data(), x, 1, T(o.omega));
            });
        }
    }

    void cycle(int l, const T* b, T* x)
    {
        Lv& V = lv[std::size_t(l)];
        if (V.coarsest) {
            kDenseApplyRef<T><<<grid(std::size_t(V.m)), kT>>>(V.m, V.Xt.data(), b, x);
            return;
        }
        const int B = V.A->shape().B, nf = V.A->rows();
        if (hipMemset(x, 0, V.A->size() * sizeof(T)) != hipSuccess) std::exit(1);
        smooth(V, b, x);
        residual(*V.A, b, x, V.res.data());
        Lv& C = lv[std::size_t(l + 1)];
        const int nc = C.A->rows();
        kRestrictRef<T><<<grid(std::size_t(nc)), kT>>>(nc, B, nf, C.memberPtr.data(), C.member.data(), V.res.data(),
                                                      C.b.data());
        correct(l + 1, C.b.data(), C.x.data());
        kProlongAddRef<T><<<grid(std::size_t(nf)), kT>>>(nf, B, nc, C.agg.data(), C.x.data(), x);
        smooth(V, b, x);
    }

    void spmv(const BlockEll<T>& A, const T* x, T* y) { residual(A, nullptr, x, y); }

    // Amg::apply: the cycle, with the per-component mean projection when constantNullSpace.
    void apply(const T* r, T* z)
    {
        if (!o.constantNullSpace || lv.size() < 2) { // Amg: the projection only below a smoothed level 0
            cycle(0, r, z);
            return;
        }
        const int n = lv[0].A->rows(), B = lv[0].A->shape().B;
        const std::size_t N = lv[0].A->size();
        auto ones = devVec<T>(std::vector<double>(std::size_t(n), 1.0));
        auto rp = devVec<T>(std::vector<double>(N, 0));
        auto part = devOf(std::vector<double>(std::size_t(internal::dotParts(std::size_t(n))) * std::size_t(B), 0));
        auto sum = devOf(std::vector<double>(std::size_t(B), 0));
        auto rm = [&](const T* x, T* y) {
            internal::DotArgs<T> d{};
            for (int q = 0; q < B; ++q) {
                d.a[q] = x + std::size_t(q) * std::size_t(n);
                d.b[q] = ones.data();
            }
            internal::deviceDots(std::size_t(n), d, B, part.data(), sum.data());
            kRemoveMeanRef<T><<<grid(N), kT>>>(n, B, sum.data(), x, y);
        };
        rm(r, rp.data());
        cycle(0, rp.data(), z);
        rm(z, z);
        if (hipDeviceSynchronize() != hipSuccess) std::exit(1);
    }

    void correct(int k, const T* b, T* x)
    {
        Lv& C = lv[std::size_t(k)];
        if (C.inner == 0) {
            cycle(k, b, x);
            return;
        }
        const std::size_t N = C.A->size();
        cycle(k, b, C.c1.data());
        spmv(*C.A, C.c1.data(), C.v1.data());
        internal::DotArgs<T> d{};
        d.a[0] = C.c1.data();
        d.b[0] = C.v1.data();
        d.a[1] = C.c1.data();
        d.b[1] = b;
        internal::deviceDots(N, d, 2, C.partial.data(), C.s.data());
        kCoef1Ref<<<1, 1>>>(C.s.data(), C.coef.data());
        if (C.inner == 1) {
            kScaledRef<T><<<grid(N), kT>>>(N, C.coef.data(), C.c1.data(), x);
            return;
        }
        kSubScaledRef<T><<<grid(N), kT>>>(N, b, C.coef.data(), C.v1.data(), C.rt.data());
        cycle(k, C.rt.data(), C.d2.data());
        spmv(*C.A, C.d2.data(), C.v2.data());
        d.a[0] = C.d2.data();
        d.b[0] = C.v1.data();
        d.a[1] = C.d2.data();
        d.b[1] = C.v2.data();
        d.a[2] = C.d2.data();
        d.b[2] = C.rt.data();
        internal::deviceDots(N, d, 3, C.partial.data(), C.s.data() + 2);
        kCoef2Ref<<<1, 1>>>(C.s.data(), C.coef.data());
        kComb2Ref<T><<<grid(N), kT>>>(N, C.coef.data(), C.c1.data(), C.d2.data(), x);
    }
};

struct AmgCfg
{
    int sweeps, inner, coarse;
    bool nullSpace = false;
};

template <class T>
void amgCase(const std::string& gname, const CellGraph& g, const BlockMatrix& m0, const AmgCfg& c, unsigned seed)
{
    const BlockMatrix m = latest::roundTo<T>(m0);
    AmgSetup su;
    su.coarseUnknowns = c.coarse;
    const AmgHierarchy h = orDie(amgHierarchy(g, amgStrength(m), m.shape.B, su), gname);
    auto A = orDie(BlockEll<T>::make(ellPattern(g), m.shape), gname);
    orDie(A.upload(m), gname);
    AmgOptions o;
    o.smoother = o.coarseSmoother = AmgSmoother::Jacobi;
    o.sweeps = c.sweeps;
    o.inner = c.inner;
    o.constantNullSpace = c.nullSpace;
    auto amg = orDie(Amg<T>::make(A, h, o), gname);
    orDie(amg.refresh(), gname);
    RefAmg<T> ref;
    const int bad = ref.build(amg, h, o);
    const std::size_t N = A.size();
    const std::string tag = gname + " " + shapeName(m.shape) + " " + tname<T>() + " sweeps " + std::to_string(c.sweeps) +
                            " inner " + std::to_string(c.inner) + " coarsest " +
                            std::to_string(ref.lv.back().m) + " unknowns, " + std::to_string(amg.levels()) + " levels";
    bool ok = bad == 0;
    auto z = devVec<T>(std::vector<double>(N, 0)), zr = devVec<T>(std::vector<double>(N, 0));
    for (unsigned rep = 0; rep < 2; ++rep) { // the second apply runs on the first's stale buffers
        auto r = devVec<T>(signedZeros(latest::randomVec(N, seed + rep)));
        amg.apply(r.data(), z.data());
        ref.apply(r.data(), zr.data());
        ok = ok && bytes(z.data(), N) == bytes(zr.data(), N);
    }
    check(ok, tag + ": apply == the pre-change cycle bit for bit (two applies)");
}

// A singular coarsest matrix: the new dense inverse fails where the old one did, by name.
template <class T>
void singularCase()
{
    const CellGraph g = structuredGraph({9, 7, 1}, {false, false, false});
    BlockMatrix m = BlockMatrix::zeros(g, *denseShape(1));
    for (int c = 0; c < g.n; ++c)
        for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) {
            m.off[std::size_t(k)] = -1.0;
            m.diag[std::size_t(c)] += 1.0;
        }
    AmgSetup su;
    su.coarseUnknowns = 1024;
    const AmgHierarchy h = orDie(amgHierarchy(g, amgStrength(m), 1, su), "singular");
    auto A = orDie(BlockEll<T>::make(ellPattern(g), m.shape), "singular");
    orDie(A.upload(m), "singular");
    auto amg = orDie(Amg<T>::make(A, h, AmgOptions{}), "singular");
    const auto f = amg.refresh();
    RefAmg<T> ref;
    const int bad = ref.build(amg, h, AmgOptions{});
    check(bad == 1 && !f &&
              f.error() == "amg: coarsest matrix (63 unknowns) singular or its inverse non-finite (a pure Neumann "
                           "operator needs constantNullSpace)",
          std::string("pure Neumann 9x7 without constantNullSpace ") + tname<T>() +
              ": the old inverse flags it and refresh fails by name (" + (f ? std::string("succeeded") : f.error()) +
              ")");
}


// ===== Krylov, as it was =====================================================

template <class T>
struct RefKrylov
{
    std::size_t N = 0;
    gpu::DeviceBuffer<T> aux;
    gpu::DeviceBuffer<double> partial, out;
    explicit RefKrylov(std::size_t n) : N(n)
    {
        aux = devVec<T>(std::vector<double>(8 * n, 0));
        partial = devOf(std::vector<double>(std::size_t(kMaxDots) * std::size_t(internal::dotParts(n)), 0));
        out = devOf(std::vector<double>(kMaxDots, 0));
    }
    T* vec(int i) { return aux.data() + std::size_t(i) * N; }
    void dots(std::span<const DotPair<T>> pairs, double* d)
    {
        internal::DotArgs<T> a{};
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            a.a[k] = pairs[k].a;
            a.b[k] = pairs[k].b;
        }
        internal::deviceDots(N, a, int(pairs.size()), partial.data(), out.data());
        if (hipMemcpy(d, out.data(), pairs.size() * sizeof(double), hipMemcpyDeviceToHost) != hipSuccess) std::exit(1);
    }
    void lin(double a, const T* x, double b, const T* y, T* o, double c = 0, const T* z = nullptr)
    {
        kLinRef<T><<<grid(N), kT>>>(N, a, x, b, y, c, z, o);
    }
    void precondition(const LinOp<T>& M, const T* r, T* z)
    {
        if (M) M(r, z);
        else if (hipMemcpy(z, r, N * sizeof(T), hipMemcpyDeviceToDevice) != hipSuccess) std::exit(1);
    }
    // the pre-change Krylov<T>::pcg, minus its error strings (nonFinite -> nullopt)
    std::optional<KrylovResult> pcg(const LinOp<T>& A, const LinOp<T>& M, const T* b, T* x, const KrylovOptions& o)
    {
        T *r = vec(0), *z = vec(1), *p = vec(2), *q = vec(3), *zOld = vec(4);
        KrylovResult res;
        double d[3];
        const bool l1 = o.normFactor > 0;
        const T* rr = l1 ? nullptr : r;
        auto measure = [&](double v) { return l1 ? v / o.normFactor : std::sqrt(v); };
        A(x, q);
        lin(1.0, b, -1.0, q, r);
        precondition(M, r, z);
        DotPair<T> rz[3] = {{r, z}, {r, rr}, {r, zOld}};
        dots(std::span(rz, 2), d);
        double rzv = d[0];
        res.r0 = res.residual = measure(d[1]);
        if (!std::isfinite(res.r0) || !std::isfinite(rzv)) return std::nullopt;
        const double target = std::max(o.relTol * res.r0, o.absTol);
        (void)hipMemcpy(p, z, N * sizeof(T), hipMemcpyDeviceToDevice);
        while ((res.residual > target || res.iterations < o.minIter) && res.iterations < o.maxIter) {
            A(p, q);
            const DotPair<T> pq{p, q};
            dots(std::span(&pq, 1), d);
            if (!std::isfinite(d[0])) return std::nullopt;
            if (!(d[0] > 0)) {
                res.breakdown = true;
                break;
            }
            const double alpha = rzv / d[0];
            lin(1.0, x, alpha, p, x);
            lin(1.0, r, -alpha, q, r);
            if (o.flexible) {
                std::swap(z, zOld);
                rz[0] = {r, z};
                rz[2] = {r, zOld};
            }
            precondition(M, r, z);
            const std::size_t nd = o.flexible ? 3 : 2;
            dots(std::span(rz, nd), d);
            ++res.iterations;
            res.residual = measure(d[1]);
            if (!std::isfinite(res.residual) || !std::isfinite(d[0])) return std::nullopt;
            const double beta = o.flexible ? (d[0] - d[2]) / rzv : d[0] / rzv;
            rzv = d[0];
            lin(1.0, z, beta, p, p);
        }
        res.relResidual = res.r0 > 0 ? res.residual / res.r0 : 0.0;
        res.converged = res.residual <= target;
        return res;
    }
    std::optional<KrylovResult> bicgstab(const LinOp<T>& A, const LinOp<T>& M, const T* b, T* x, const KrylovOptions& o)
    {
        T *r = vec(0), *rh = vec(1), *p = vec(2), *v = vec(3), *ph = vec(4), *s = vec(5), *sh = vec(6), *t = vec(7);
        KrylovResult res;
        double d[4];
        const bool l1 = o.normFactor > 0;
        A(x, v);
        lin(1.0, b, -1.0, v, r);
        (void)hipMemcpy(rh, r, N * sizeof(T), hipMemcpyDeviceToDevice);
        const DotPair<T> rr[2] = {{r, r}, {r, nullptr}};
        dots(std::span(rr, l1 ? 2 : 1), d);
        double r2 = std::sqrt(d[0]);
        res.r0 = res.residual = l1 ? d[1] / o.normFactor : r2;
        if (!std::isfinite(res.r0) || !std::isfinite(r2)) return std::nullopt;
        const double target = std::max(o.relTol * res.r0, o.absTol);
        double rho = 1, alpha = 1, omega = 1, rhoNew = d[0], rhNorm = r2;
        bool fresh = true;
        while ((res.residual > target || res.iterations < o.minIter) && res.iterations < o.maxIter) {
            if (std::abs(rhoNew) <= 0x1p-52 * rhNorm * r2) {
                (void)hipMemcpy(rh, r, N * sizeof(T), hipMemcpyDeviceToDevice);
                rhNorm = r2;
                rhoNew = r2 * r2;
                fresh = true;
                ++res.restarts;
            }
            if (fresh) (void)hipMemcpy(p, r, N * sizeof(T), hipMemcpyDeviceToDevice);
            else {
                const double beta = (rhoNew / rho) * (alpha / omega);
                lin(1.0, r, beta, p, p, -beta * omega, v);
            }
            rho = rhoNew;
            fresh = false;
            precondition(M, p, ph);
            A(ph, v);
            const DotPair<T> rv{rh, v};
            dots(std::span(&rv, 1), d);
            if (!std::isfinite(d[0])) return std::nullopt;
            if (d[0] == 0) {
                res.breakdown = true;
                break;
            }
            alpha = rho / d[0];
            lin(1.0, r, -alpha, v, s);
            precondition(M, s, sh);
            A(sh, t);
            const DotPair<T> ts[4] = {{t, s}, {t, t}, {s, s}, {s, nullptr}};
            dots(std::span(ts, l1 ? 4 : 3), d);
            if (!std::isfinite(d[0] + d[1] + d[2])) return std::nullopt;
            ++res.iterations;
            const double sMeasure = l1 ? d[3] / o.normFactor : std::sqrt(d[2]);
            if (sMeasure <= target && res.iterations >= o.minIter) {
                lin(1.0, x, alpha, ph, x);
                res.residual = sMeasure;
                break;
            }
            if (d[1] == 0) {
                res.breakdown = true;
                break;
            }
            omega = d[0] / d[1];
            lin(1.0, x, alpha, ph, x, omega, sh);
            lin(1.0, s, -omega, t, r);
            const DotPair<T> rn[3] = {{rh, r}, {r, r}, {r, nullptr}};
            dots(std::span(rn, l1 ? 3 : 2), d);
            rhoNew = d[0];
            r2 = std::sqrt(d[1]);
            res.residual = l1 ? d[2] / o.normFactor : r2;
            if (!std::isfinite(res.residual) || !std::isfinite(r2)) return std::nullopt;
            if (omega == 0) {
                res.breakdown = true;
                break;
            }
        }
        res.relResidual = res.r0 > 0 ? res.residual / res.r0 : 0.0;
        res.converged = res.residual <= target;
        return res;
    }
};

bool same(const KrylovResult& a, const KrylovResult& b)
{
    return a.iterations == b.iterations && a.restarts == b.restarts && std::memcmp(&a.r0, &b.r0, 8) == 0 &&
           std::memcmp(&a.residual, &b.residual, 8) == 0 && std::memcmp(&a.relResidual, &b.relResidual, 8) == 0 &&
           a.converged == b.converged && a.breakdown == b.breakdown && !a.flagged && !b.flagged;
}

__global__ void kNormTerms(int n, const double* Ax, const double* b, double* out)
{
    const int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n) out[i] = fabs(Ax[i] - 0.25 * b[i]) + fabs(b[i]);
}

// Library pcg / bicgstab (no start, and with a KrylovStart) vs the old loops.
void krylovCase(const std::string& gname, const CellGraph& g, bool spd, unsigned seed)
{
    using T = double;
    BlockMatrix m = latest::randomMatrix(g, *denseShape(1), seed);
    if (spd) { // symmetric: a_cj = a_jc, still dominant
        for (int c = 0; c < g.n; ++c)
            for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) {
                const int j = g.col[std::size_t(k)];
                if (j < c) m.off[std::size_t(k)] = m.off[std::size_t(latest::slotOf(m, j, c))];
            }
        for (double& v : m.diag) v = std::abs(v) + 8.0;
    }
    auto A = orDie(BlockEll<T>::make(ellPattern(g), m.shape), gname);
    orDie(A.upload(m), gname);
    const std::size_t N = A.size();
    BlockMatrix mi = m; // indefinite: PCG breaks down
    for (int c = 0; c < g.n; c += 2) mi.diag[std::size_t(c)] = -mi.diag[std::size_t(c)];
    auto Ai = orDie(BlockEll<T>::make(ellPattern(g), m.shape), gname);
    orDie(Ai.upload(mi), gname);
    auto J = orDie(BlockJacobi<T>::make(A), gname);
    orDie(J.factor(), gname);
    const LinOp<T> Mj = [&](const T* r, T* z) { J.apply(r, z); };
    auto K = orDie(Krylov<T>::make(N, 0), gname);
    RefKrylov<T> R(N);
    auto b = devVec<T>(latest::randomVec(N, seed + 1));
    auto x0 = devVec<T>(latest::randomVec(N, seed + 2));
    auto xa = devVec<T>(std::vector<double>(N, 0)), xb = devVec<T>(std::vector<double>(N, 0));
    auto Ax = devVec<T>(std::vector<double>(N, 0)), nt = devVec<T>(std::vector<double>(N, 0));
    auto part = devOf(std::vector<double>(std::size_t(internal::dotParts(N)), 0));
    auto sum = devOf(std::vector<double>(1, 0));
    int fails = 0, cases = 0;
    for (const bool pcg : {true, false})
        for (const bool indef : {false, true})
            for (const bool useStart : {false, true})
                for (const int minIt : {0, 2})
                    for (const int maxIt : {3, 200})
                        for (const double absTol : {0.0, 1e30}) // 1e30: converged at r0
                            for (const bool r0First : {true, false}) {
                            if (indef && !pcg) continue;
                            if (!pcg && !r0First) continue;
                            const BlockEll<T>& Op = indef ? Ai : A;
                            const LinOp<T> Aop = [&](const T* in, T* o2) { Op.spmv(in, o2); };
                            KrylovOptions o;
                            o.relTol = 1e-8;
                            o.absTol = absTol;
                            o.minIter = minIt;
                            o.maxIter = maxIt;
                            o.flexible = pcg && minIt == 0;
                            o.r0First = r0First;
                            KrylovStart<T> st{};
                            if (useStart) { // the FV solve's start: A x0 and normFactor terms on the device
                                Op.spmv(x0.data(), Ax.data());
                                kNormTerms<<<grid(N), kT>>>(int(N), Ax.data(), b.data(), nt.data());
                                internal::DotArgs<T> da{};
                                da.a[0] = nt.data();
                                da.b[0] = nullptr;
                                internal::deviceDots(N, da, 1, part.data(), sum.data());
                                double h = 0;
                                if (sum.download(&h) != hipSuccess) std::exit(1);
                                o.normFactor = h + 1e-20; // the reference's host normFactor
                                st = {Ax.data(), nt.data()};
                            }
                            copyDev(xa.data(), x0.data(), N);
                            copyDev(xb.data(), x0.data(), N);
                            KrylovOptions ol = o;
                            if (useStart) ol.normFactor = 0; // the library takes it from st
                            auto rl = pcg ? K.pcg(Aop, Mj, b.data(), xa.data(), ol, st)
                                          : K.bicgstab(Aop, Mj, b.data(), xa.data(), ol, st);
                            auto rr = pcg ? R.pcg(Aop, Mj, b.data(), xb.data(), o) : R.bicgstab(Aop, Mj, b.data(), xb.data(), o);
                            ++cases;
                            const bool ok = rl && rr && same(*rl, *rr) && bytes(xa.data(), N) == bytes(xb.data(), N);
                            if (!ok) {
                                ++fails;
                                std::printf("    mismatch: %s indef %d start %d minIter %d maxIter %d absTol %g: it %d/%d "
                                            "r0 %.17g/%.17g res %.17g/%.17g bd %d/%d\n",
                                            pcg ? "pcg" : "bicgstab", indef, useStart, minIt, maxIt, absTol,
                                            rl ? rl->iterations : -1, rr ? rr->iterations : -1, rl ? rl->r0 : 0,
                                            rr ? rr->r0 : 0, rl ? rl->residual : 0, rr ? rr->residual : 0,
                                            rl ? rl->breakdown : 0, rr ? rr->breakdown : 0);
                            }
                            }
    check(fails == 0, gname + (spd ? " SPD" : " non-symmetric") + ": pcg / bicgstab (start or not, minIter 0/2, maxIter "
                      "3/200, converged at r0, r0First or not, indefinite breakdown) == the pre-change loops bit for bit (" +
                      std::to_string(cases) + " cases)");
    // the pre-flag: the solve stops at its first readback, x untouched, flag cleared
    const LinOp<T> Aop = [&](const T* in, T* o2) { A.spmv(in, o2); };
    bool okF = true;
    for (const bool pcg : {true, false}) {
        kSetFlag<<<1, 1>>>(K.preFlag());
        copyDev(xa.data(), x0.data(), N);
        KrylovOptions o;
        auto rl = pcg ? K.pcg(Aop, Mj, b.data(), xa.data(), o) : K.bicgstab(Aop, Mj, b.data(), xa.data(), o);
        okF = okF && rl && rl->flagged && bytes(xa.data(), N) == bytes(x0.data(), N);
        auto r2 = pcg ? K.pcg(Aop, Mj, b.data(), xa.data(), o) : K.bicgstab(Aop, Mj, b.data(), xa.data(), o);
        okF = okF && r2 && !r2->flagged && r2->iterations > 0;
    }
    check(okF, gname + ": a set pre-flag stops pcg / bicgstab at the first readback (x untouched) and is cleared");
}

// bicgstabBatch (with spmvMulti, MultiColourGS factorMulti / applyMulti) vs
// one library bicgstab per system (itself == the pre-change loop, above).
void batchCase(const std::string& gname, const CellGraph& g, unsigned seed)
{
    using T = double;
    const BlockMatrix m = latest::randomMatrix(g, *denseShape(1), seed, 0.3);
    const std::size_t N = std::size_t(g.n);
    auto A = orDie(BlockEll<T>::make(ellPattern(g), m.shape), gname);
    orDie(A.upload(m), gname);
    auto G = orDie(MultiColourGS<T>::make(A, greedyColour(g)), gname);
    auto K1 = orDie(Krylov<T>::make(N, 0), gname);
    auto K3 = orDie(Krylov<T>::make(N, 0, 3), gname);
    // three diagonals: system j's dominance differs, so they stop at different iterations
    std::vector<double> d3(3 * N);
    for (int j = 0; j < 3; ++j)
        for (std::size_t c = 0; c < N; ++c) d3[std::size_t(j) * N + c] = m.diag[c] * (1.0 + 0.6 * j) + (c % 5 == 0 ? 0.2 * j : 0.0);
    auto diag3 = devVec<T>(d3);
    auto ones = devVec<T>(std::vector<double>(N, 1.0));
    auto b3 = devVec<T>(latest::randomVec(3 * N, seed + 1));
    auto x0 = devVec<T>(latest::randomVec(3 * N, seed + 2));
    auto xs = devVec<T>(std::vector<double>(3 * N, 0)), xb = devVec<T>(std::vector<double>(3 * N, 0));
    auto Ax = devVec<T>(std::vector<double>(3 * N, 0)), nt = devVec<T>(std::vector<double>(3 * N, 0));
    auto y1 = devVec<T>(std::vector<double>(3 * N, 0)), y3 = devVec<T>(std::vector<double>(3 * N, 0));
    BlockMatrix mj = m;
    auto uploadDiag = [&](int j) {
        for (std::size_t c = 0; c < N; ++c) mj.diag[c] = d3[std::size_t(j) * N + c];
        orDie(A.upload(mj), gname);
    };
    // the multi operators against the single ones, bitwise
    bool okOp = true;
    int* fl3[3] = {K3.preFlag(0), K3.preFlag(1), K3.preFlag(2)};
    orDie(G.factorMulti(3, 7u, diag3.data(), fl3, true), gname);
    A.spmvMulti(3, 7u, diag3.data(), x0.data(), N, y3.data());
    G.applyMulti(3, 7u, b3.data(), xb.data());
    for (int j = 0; j < 3; ++j) {
        uploadDiag(j);
        A.spmv(x0.data() + std::size_t(j) * N, y1.data() + std::size_t(j) * N);
        orDie(G.factor(), gname);
        G.apply(b3.data() + std::size_t(j) * N, xs.data() + std::size_t(j) * N);
    }
    okOp = bytes(y1.data(), 3 * N) == bytes(y3.data(), 3 * N) && bytes(xs.data(), 3 * N) == bytes(xb.data(), 3 * N);
    check(okOp, gname + ": spmvMulti / factorMulti + applyMulti == spmv / factor + apply per system, bit for bit");
    int fails = 0, cases = 0;
    for (const unsigned mask : {7u, 5u, 2u})
        for (const bool useStart : {false, true})
            for (const int minIt : {0, 2})
                for (const int maxIt : {2, 300})
                    for (const int special : {0, 1, 2}) { // 1: system 1 starts converged; 2: system 2's factor fails
                        KrylovOptions o;
                        o.relTol = 1e-9;
                        o.minIter = minIt;
                        o.maxIter = maxIt;
                        std::vector<double> d = d3;
                        if (special == 2) d[2 * N + N / 2] = 0.0;
                        auto dd = devVec<T>(d);
                        copyDev(xs.data(), x0.data(), 3 * N);
                        copyDev(xb.data(), x0.data(), 3 * N);
                        if (special == 1) o.absTol = 0; // set per system below via the start x
                        // system 1 at an exact zero residual: b1 := A1 x1
                        std::vector<KrylovResult> solo(3);
                        std::vector<bool> soloOk(3, true);
                        auto bb = devVec<T>(latest::randomVec(3 * N, seed + 1));
                        if (special == 1) {
                            for (std::size_t c = 0; c < N; ++c) mj.diag[c] = d[N + c];
                            orDie(A.upload(mj), gname);
                            A.spmv(x0.data() + N, bb.data() + N);
                        }
                        for (int j = 0; j < 3; ++j) {
                            if (!((mask >> j) & 1u)) continue;
                            for (std::size_t c = 0; c < N; ++c) mj.diag[c] = d[std::size_t(j) * N + c];
                            orDie(A.upload(mj), gname);
                            T* xj = xs.data() + std::size_t(j) * N;
                            KrylovStart<T> st{};
                            KrylovOptions oj = o;
                            if (useStart) {
                                A.spmv(xj, Ax.data());
                                kNormTerms<<<grid(N), kT>>>(int(N), Ax.data(), bb.data() + std::size_t(j) * N, nt.data());
                                st = {Ax.data(), nt.data()};
                            }
                            G.factorDeferred(K1.preFlag());
                            const LinOp<T> Aop = [&](const T* in, T* o2) { A.spmv(in, o2); };
                            const LinOp<T> Mop = [&](const T* r, T* z) { G.apply(r, z); };
                            auto r = K1.bicgstab(Aop, Mop, bb.data() + std::size_t(j) * N, xj, oj, st);
                            if (r) solo[std::size_t(j)] = *r;
                            else soloOk[std::size_t(j)] = false;
                        }
                        // batched
                        orDie(A.upload(m), gname);
                        orDie(G.factorMulti(3, mask, dd.data(), fl3, true), gname);
                        KrylovStart<T> st3{};
                        if (useStart) {
                            A.spmvMulti(3, mask, dd.data(), xb.data(), N, Ax.data());
                            for (int j = 0; j < 3; ++j)
                                kNormTerms<<<grid(N), kT>>>(int(N), Ax.data() + std::size_t(j) * N,
                                                            bb.data() + std::size_t(j) * N, nt.data() + std::size_t(j) * N);
                            st3 = {Ax.data(), nt.data()};
                        }
                        const LinOp<T> A3 = [&](const T* in, T* o2) { A.spmvMulti(3, mask, dd.data(), in, N, o2); };
                        const LinOp<T> M3 = [&](const T* r, T* z) { G.applyMulti(3, mask, r, z); };
                        auto rb = K3.bicgstabBatch(3, mask, A3, M3, bb.data(), xb.data(), o, st3);
                        ++cases;
                        bool ok = bool(rb);
                        for (int j = 0; ok && j < 3; ++j) {
                            if (!((mask >> j) & 1u)) continue;
                            const auto& q = (*rb)[std::size_t(j)];
                            const KrylovResult& sr = solo[std::size_t(j)];
                            if (!soloOk[std::size_t(j)]) ok = !q;
                            else if (sr.flagged) ok = q && q->flagged;
                            else ok = q && same(*q, sr);
                            ok = ok && bytes(xb.data() + std::size_t(j) * N, N) == bytes(xs.data() + std::size_t(j) * N, N);
                        }
                        for (int j = 0; ok && j < 3; ++j) // masked-out systems untouched
                            if (!((mask >> j) & 1u))
                                ok = bytes(xb.data() + std::size_t(j) * N, N) == bytes(x0.data() + std::size_t(j) * N, N);
                        if (!ok) {
                            ++fails;
                            std::printf("    batch mismatch: mask %u start %d minIter %d maxIter %d special %d\n", mask,
                                        useStart, minIt, maxIt, special);
                        }
                    }
    check(fails == 0, gname + ": bicgstabBatch == one bicgstab per system bit for bit (masks 7/5/2, start or not, "
                      "minIter, maxIter, a system converged at r0, a failed factor; " + std::to_string(cases) + " cases)");
}
} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    std::printf("gpu/la performance kernels vs the pre-change kernels (bitwise)\n");
    std::vector<std::pair<std::string, CellGraph>> gs;
    gs.emplace_back("2D 37x23", structuredGraph({37, 23, 1}, {false, false, false}));
    gs.emplace_back("3D cyclic 8x6x5", structuredGraph({8, 6, 5}, {true, true, true}));
    gs.emplace_back("cyl2d_snappy", meshGraph("cyl2d_snappy"));
    gs.emplace_back("box_refined", meshGraph("box_refined"));
    unsigned seed = 11;
    for (const auto& [name, g] : gs)
        for (const BlockShape s : allShapes()) {
            kernelCase<double>(name, g, s, seed);
            kernelCase<float>(name, g, s, seed + 1);
            seed += 5;
        }
    // AMG: the FV p configuration (Jacobi, 2 sweeps, scaled V) and its neighbours,
    // coarsest sizes 1..128 unknowns (the register dense inverse's range) and above.
    std::vector<std::pair<std::string, CellGraph>> ag;
    ag.emplace_back("2D 160x120", structuredGraph({160, 120, 1}, {false, false, false}));
    ag.emplace_back("3D 40x30x20", structuredGraph({40, 30, 20}, {false, false, false}));
    ag.emplace_back("cyl2d_snappy", gs[2].second);
    ag.emplace_back("box_refined", gs[3].second);
    const AmgCfg cfgs[] = {{2, 1, 128}, {1, 0, 128}, {3, 2, 128}, {2, 1, 32}, {2, 1, 7}, {1, 1, 200}};
    for (const auto& [name, g] : ag) {
        const BlockMatrix m1 = latest::randomMatrix(g, *denseShape(1), seed++);
        for (const AmgCfg& c : cfgs) {
            amgCase<double>(name, g, m1, c, seed);
            amgCase<float>(name, g, m1, c, seed + 1);
            seed += 3;
        }
        const BlockMatrix m3 = latest::randomMatrix(g, *denseShape(3), seed++);
        amgCase<double>(name, g, m3, {2, 1, 128}, seed);
        amgCase<float>(name, g, m3, {2, 1, 64}, seed + 1);
        const BlockMatrix m7 = latest::randomMatrix(g, *blockShape(7, 5, true), seed++);
        amgCase<double>(name, g, m7, {2, 1, 128}, seed);
    }
    // integer-valued Laplacians: exact pivot ties in the coarsest inverse; the
    // pure Neumann one with constantNullSpace (alpha shift, mean projection)
    for (const bool neumann : {false, true})
        for (const auto& [name, g] : {std::pair<std::string, CellGraph>{"2D 10x11", structuredGraph({10, 11, 1}, {false, false, false})},
                                      std::pair<std::string, CellGraph>{"2D 64x48", structuredGraph({64, 48, 1}, {false, false, false})}}) {
            BlockMatrix m = BlockMatrix::zeros(g, *denseShape(1));
            for (int c = 0; c < g.n; ++c) {
                m.diag[std::size_t(c)] = neumann ? 0.0 : 4.5;
                for (int k = g.rowPtr[std::size_t(c)]; k < g.rowPtr[std::size_t(c) + 1]; ++k) {
                    m.off[std::size_t(k)] = -1.0;
                    if (neumann) m.diag[std::size_t(c)] += 1.0;
                }
            }
            const std::string tag = name + (neumann ? " Neumann Laplacian (null space)" : " integer Laplacian");
            for (const AmgCfg c : {AmgCfg{2, 1, 1024, neumann}, AmgCfg{2, 1, 128, neumann}, AmgCfg{1, 0, 16, neumann}}) {
                if (c.coarse == 1024 && g.n > 128) continue;
                amgCase<double>(tag, g, m, c, seed);
                amgCase<float>(tag, g, m, c, seed + 1);
                seed += 2;
            }
        }
    singularCase<double>();
    singularCase<float>();
    krylovCase("cyl2d_snappy", gs[2].second, true, 77u);
    krylovCase("cyl2d_snappy", gs[2].second, false, 78u);
    krylovCase("box_refined", gs[3].second, true, 79u);
    krylovCase("2D 37x23", gs[0].second, false, 80u);
    batchCase("cyl2d_snappy", gs[2].second, 91u);
    batchCase("box_refined", gs[3].second, 92u);
    batchCase("2D 37x23", gs[0].second, 93u);
    return dnstest::finish();
}
