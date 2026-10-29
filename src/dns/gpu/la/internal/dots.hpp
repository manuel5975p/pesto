// Batched FP64 dot kernels of gpu/la (Krylov, AMG): one launch of fixed-order
// block partials (grid P x K), one fixed-order reduce (one block per dot).
// P depends on N only, so results are byte-identical on any device; shared-
// memory trees, no shuffles (wave32 = wave64). INTERNAL (.hip TUs only).
#pragma once

#include "dns/gpu/la/internal/ellops.hpp"

#include <algorithm>
#include <cstddef>

namespace dns::gpu::la::internal {

// Max inner products per batch (CGS2 over a full FGMRES basis, plus one).
inline constexpr int kMaxDotBatch = 66;

template <class T>
struct DotArgs
{
    const T* a[kMaxDotBatch];
    const T* b[kMaxDotBatch];
};

// Partials per dot for vectors of length N (fixed by N alone).
inline int dotParts(std::size_t N) { return int(std::clamp<std::size_t>((N + 2047) / 2048, 1, 1024)); }

namespace {

// Block partials of dot blockIdx.y: a fixed strided loop per thread, then a
// fixed shared-memory tree. b == null: the L1 sum of |a| instead (OpenFOAM's
// residual measure, KrylovOptions::normFactor).
template <class T>
__global__ void __launch_bounds__(256) kDotPartial(std::size_t N, DotArgs<T> d, double* partial)
{
    __shared__ double sh[kThreads];
    const int k = int(blockIdx.y), t = int(threadIdx.x);
    const T* a = d.a[k];
    const T* b = d.b[k];
    double s = 0.0;
    const std::size_t stride = std::size_t(gridDim.x) * kThreads;
    if (b)
        for (std::size_t i = std::size_t(blockIdx.x) * kThreads + std::size_t(t); i < N; i += stride)
            s += double(a[i]) * double(b[i]);
    else
        for (std::size_t i = std::size_t(blockIdx.x) * kThreads + std::size_t(t); i < N; i += stride)
            s += fabs(double(a[i]));
    sh[t] = s;
    __syncthreads();
    for (int w = kThreads / 2; w > 0; w >>= 1) {
        if (t < w) sh[t] += sh[t + w];
        __syncthreads();
    }
    if (t == 0) partial[std::size_t(k) * gridDim.x + blockIdx.x] = sh[0];
}

// maybe_unused: a TU that includes this header without calling deviceDots.
[[maybe_unused]] __global__ void __launch_bounds__(256) kDotFinal(int P, const double* partial, double* out)
{
    __shared__ double sh[kThreads];
    const int k = int(blockIdx.x), t = int(threadIdx.x);
    double s = 0.0;
    for (int i = t; i < P; i += kThreads) s += partial[std::size_t(k) * std::size_t(P) + std::size_t(i)];
    sh[t] = s;
    __syncthreads();
    for (int w = kThreads / 2; w > 0; w >>= 1) {
        if (t < w) sh[t] += sh[t + w];
        __syncthreads();
    }
    if (t == 0) out[k] = sh[0];
}

} // namespace

// out[k] = (a_k, b_k), K pairs of length N, left in DEVICE memory (no sync).
// Pre: 1 <= K <= kMaxDotBatch, N > 0, partial holds dotParts(N) * K doubles.
template <class T>
void deviceDots(std::size_t N, const DotArgs<T>& d, int K, double* partial, double* out)
{
    const int P = dotParts(N);
    kDotPartial<T><<<dim3(unsigned(P), unsigned(K)), kThreads>>>(N, d, partial);
    kDotFinal<<<unsigned(K), kThreads>>>(P, partial, out);
}

} // namespace dns::gpu::la::internal
