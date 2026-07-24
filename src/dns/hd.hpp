// DNS_HD: expands to __host__ __device__ under a HIP/CUDA compiler and to
// nothing under a plain C++ compiler, so the same mathematical structs
// (dns/remap.hpp) serve host mesh generation and device kernels alike.
#pragma once

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__) || defined(__CUDACC__)
#define DNS_HD __host__ __device__
#else
#define DNS_HD
#endif
