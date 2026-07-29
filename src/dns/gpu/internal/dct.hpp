// Makhoul same-length-FFT DCT-II / DCT-III kernels, shared by the Neumann
// Poisson solver (gpu/poisson.hip, 3D rotating layout) and the transverse-modal
// NSCBC face transform (gpu/nscbc_modal.hip, one batched axis).
//
//   v[j]     = x[2j]           j <  ceil(n/2)
//   v[n-1-j] = x[2j+1]
//   C_k      = Re( e^{-i pi k / (2n)} FFT_n(v)_k )
// inverse (DCT-III):
//   V_k = e^{i pi k / (2n)} (C_k - i C_{n-k}) / n,  C_n := 0
//   x[permute(j)] = Re( IFFT_n(V)_j )
// The 1/n in the inverse twiddle compensates the unnormalised hipFFT inverse,
// so DCT-III(DCT-II(x)) == x.
//
// Two output shapes per direction: the *Rot kernels write with the axes rotated
// (the 3D pass chain), the *1d kernels write in place (a single batched axis).
// Kernels live in an anonymous namespace: each including TU gets its own copy,
// and the ones it does not launch are [[maybe_unused]].
//
// GPU-only internal header: not part of the public dns/ include set.
#pragma once

#include "dns/gpu/index.hpp"

#include <hipfft/hipfft.h>

#include <cmath>
#include <numbers>

namespace dns::gpu {
namespace {

// Even-then-reversed-odd reordering of one length-n line.
__device__ inline int evenOdd(int j, int n)
{
    return j < (n + 1) / 2 ? 2 * j : 2 * (n - 1 - j) + 1;
}

[[maybe_unused]] __global__ void kDctPermute(std::size_t total, int n0,
                                             const Real* x, hipfftComplex* v)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int j = int(m % n0);
    const std::size_t b = m / n0;
    v[m] = hipfftComplex{x[b * n0 + evenOdd(j, n0)], Real(0)};
}

// The rotation is a (k, i1) transpose per i2 slab; stage a 32x32 tile through
// LDS so both the read (k-fastest) and the write (i1-fastest) coalesce.
constexpr int dctTile = 32;
constexpr int dctRows = 8;

[[maybe_unused]] __global__ void kDctTwiddleRot(int n0, int n1, int n2,
                                                const hipfftComplex* V, Real* out)
{
    __shared__ Real tile[dctTile][dctTile + 1];
    const int k0 = blockIdx.x * dctTile, i10 = blockIdx.y * dctTile;
    const int i2 = blockIdx.z;
    const int tx = threadIdx.x;
    for (int r = threadIdx.y; r < dctTile; r += dctRows)
    {
        const int k = k0 + tx, i1 = i10 + r;
        if (k < n0 && i1 < n1)
        {
            const std::size_t m = std::size_t(k)
                + std::size_t(n0) * (i1 + std::size_t(n1) * i2);
            const Real th = Real(std::numbers::pi) * k / Real(2 * n0);
            tile[r][tx] = std::cos(th) * V[m].x + std::sin(th) * V[m].y;
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < dctTile; r += dctRows)
    {
        const int i1 = i10 + tx, k = k0 + r;
        if (k < n0 && i1 < n1)
            out[std::size_t(i1) + std::size_t(n1) * (i2 + std::size_t(n2) * k)]
                = tile[tx][r];
    }
}

[[maybe_unused]] __global__ void kDctInvTwiddle(std::size_t total, int n0,
                                                const Real* C, hipfftComplex* V)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int k = int(m % n0);
    const std::size_t b = m / n0;
    const Real ck = C[b * n0 + k];
    const Real cnk = k == 0 ? Real(0) : C[b * n0 + (n0 - k)];
    const Real th = Real(std::numbers::pi) * k / Real(2 * n0);
    const Real inv = Real(1) / Real(n0);
    V[m] = hipfftComplex{(std::cos(th) * ck + std::sin(th) * cnk) * inv,
                         (std::sin(th) * ck - std::cos(th) * cnk) * inv};
}

// Same tiled (j, i1) transpose; evenOdd only relabels the output k-plane, the
// i1-contiguous writes stay coalesced.
[[maybe_unused]] __global__ void kDctInvPermuteRot(int n0, int n1, int n2,
                                                   const hipfftComplex* v,
                                                   Real* out)
{
    __shared__ Real tile[dctTile][dctTile + 1];
    const int j0 = blockIdx.x * dctTile, i10 = blockIdx.y * dctTile;
    const int i2 = blockIdx.z;
    const int tx = threadIdx.x;
    for (int r = threadIdx.y; r < dctTile; r += dctRows)
    {
        const int j = j0 + tx, i1 = i10 + r;
        if (j < n0 && i1 < n1)
            tile[r][tx]
                = v[std::size_t(j) + std::size_t(n0) * (i1 + std::size_t(n1) * i2)].x;
    }
    __syncthreads();
    for (int r = threadIdx.y; r < dctTile; r += dctRows)
    {
        const int i1 = i10 + tx, j = j0 + r;
        if (j < n0 && i1 < n1)
            out[std::size_t(i1)
                + std::size_t(n1) * (i2 + std::size_t(n2) * evenOdd(j, n0))]
                = tile[tx][r];
    }
}

// Single-axis forms: same twiddle, layout preserved (k where j was), so one
// batched pass is the whole transform. total = n0 * batch.
[[maybe_unused]] __global__ void kDctTwiddle1d(std::size_t total, int n0,
                                               const hipfftComplex* V, Real* out)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int k = int(m % n0);
    const Real th = Real(std::numbers::pi) * k / Real(2 * n0);
    out[m] = std::cos(th) * V[m].x + std::sin(th) * V[m].y;
}

[[maybe_unused]] __global__ void kDctInvPermute1d(std::size_t total, int n0,
                                                  const hipfftComplex* v,
                                                  Real* out)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int j = int(m % n0);
    const std::size_t b = m / n0;
    out[b * n0 + evenOdd(j, n0)] = v[m].x;
}

[[maybe_unused]] inline dim3 rotGrid(int n0, int n1, int n2)
{
    return dim3(unsigned((n0 + dctTile - 1) / dctTile),
                unsigned((n1 + dctTile - 1) / dctTile), unsigned(n2));
}
[[maybe_unused]] constexpr dim3 rotBlock{dctTile, dctRows, 1};

} // namespace
} // namespace dns::gpu
