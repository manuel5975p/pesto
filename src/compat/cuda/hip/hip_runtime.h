// hip/hip_runtime.h -- CUDA build only. See src/compat/cuda/README.md.
//
// Host-safe: this header is reached from plain CXX translation units
// (apps/dns_main.cpp, tests/test_gpu_*.cpp) via gpu/device_util.hpp, so it must
// compile under g++ with no nvcc extensions. <cuda_runtime.h> is written for
// exactly that; the __global__/__device__/blockIdx spellings the .hip sources
// use come from nvcc itself and never reach the host compiler.
#pragma once

#include <cuda_runtime.h>

using hipError_t = cudaError_t;

inline constexpr cudaError_t hipSuccess = cudaSuccess;
inline constexpr cudaError_t hipErrorInvalidValue = cudaErrorInvalidValue;
inline constexpr cudaError_t hipErrorUnknown = cudaErrorUnknown;

inline constexpr cudaMemcpyKind hipMemcpyHostToDevice = cudaMemcpyHostToDevice;
inline constexpr cudaMemcpyKind hipMemcpyDeviceToHost = cudaMemcpyDeviceToHost;
inline constexpr cudaMemcpyKind hipMemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;

// Macros rather than inline forwarders: cudaMalloc is a template
// (T** -> void**), and a hand-written wrapper would have to reproduce that
// deduction. A rename is exactly what is wanted here and nothing more.
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemset cudaMemset
#define hipDeviceSynchronize cudaDeviceSynchronize
#define hipGetLastError cudaGetLastError
#define hipGetErrorString cudaGetErrorString
#define hipGetDeviceCount cudaGetDeviceCount
#define hipMemGetInfo cudaMemGetInfo
