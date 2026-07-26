// hipfft/hipfft.h -- CUDA build only. See src/compat/cuda/README.md.
//
// Reached only from gpu/poisson.hip, so nvcc-only is acceptable here; it is
// kept host-clean anyway because <cufft.h> is.
#pragma once

#include <cufft.h>

using hipfftResult = cufftResult;
using hipfftHandle = cufftHandle;
using hipfftComplex = cufftComplex; // both float2; aggregate init unchanged

inline constexpr cufftResult HIPFFT_SUCCESS = CUFFT_SUCCESS;

inline constexpr cufftType HIPFFT_R2C = CUFFT_R2C;
inline constexpr cufftType HIPFFT_C2R = CUFFT_C2R;
inline constexpr cufftType HIPFFT_C2C = CUFFT_C2C;

inline constexpr int HIPFFT_FORWARD = CUFFT_FORWARD;   // -1 in both
inline constexpr int HIPFFT_BACKWARD = CUFFT_INVERSE;  // +1 in both

#define hipfftPlan1d cufftPlan1d
#define hipfftPlan2d cufftPlan2d
#define hipfftPlan3d cufftPlan3d
#define hipfftDestroy cufftDestroy
#define hipfftExecR2C cufftExecR2C
#define hipfftExecC2R cufftExecC2R
#define hipfftExecC2C cufftExecC2C
