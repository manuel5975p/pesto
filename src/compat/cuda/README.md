# src/compat/cuda — the HIP-on-CUDA shim

`DNS_WITH_CUDA=ON` builds the same GPU backend against the CUDA toolkit. It is
a *header shim*, not a port: the `.hip` sources are unchanged, still say
`hipMalloc` / `hipfftExecR2C` / `rocprim::reduce`, and are simply compiled as
CUDA (`LANGUAGE CUDA`, so nvcc gets `-x cu`). This directory is put on the
include path AHEAD of nothing else — there is no ROCm to shadow, because a CUDA
build does not have one — and supplies the three include roots the sources ask
for:

    hip/hip_runtime.h      -> cuda_runtime.h    (~20 names)
    hipfft/hipfft.h        -> cufft.h           (~15 names)
    rocprim/{device,iterator}/*  -> CUB + thrust iterators

The mapping is 1:1 by construction, which is the point: HIP's own headers do
exactly this when ROCm targets an NVIDIA device, so the surface is known to be
expressible. What is here is only what `grep -o 'hip[A-Za-z]*' src apps tests`
actually finds — deliberately not a general HIP emulation. Adding a new HIP call
to a `.hip` file that is not in these headers is a CUDA-build link/compile
error, which is the intended feedback: extend the shim in the same commit.

Semantic notes, i.e. the places where 1:1 needed a check rather than a typedef:

- `rocprim::reduce(tmp, bytes, in, out, init, n, op)` has `init` BEFORE the
  count; `cub::DeviceReduce::Reduce` has it after `op`. The wrapper reorders.
- rocPRIM's `plus`/`maximum`/`minimum` are spelled here as local functors
  rather than aliased to `cuda::std::plus` / `cuda::maximum`, whose names and
  headers have moved across CCCL releases. Two lines each, and version-proof.
- `hipfftComplex` is `float2`, `cufftComplex` is `float2`. The aggregate
  initialisation in `gpu/poisson.hip` works unchanged.
- `HIPFFT_FORWARD`/`HIPFFT_BACKWARD` are -1/+1 in both.
- The headers must stay compilable by the plain HOST compiler: `dns_main.cpp`
  and the `test_gpu_*.cpp` tests are CXX TUs that include `gpu/device_util.hpp`
  and hence `hip/hip_runtime.h`. `cuda_runtime.h` is host-safe; `cub.cuh` is
  not, which is why the rocPRIM shim is only reachable from `gpu/internal/`.

## Build gotchas, measured on this machine (2026-07-25, CUDA 13.3, RTX 3070)

- **CMake tops out at the `CUDA20` dialect.** Its NVIDIA-CUDA module has no
  `CMAKE_CUDA23_STANDARD_COMPILE_OPTION` as of 4.3, so a target inheriting
  `CXX_STANDARD 23` fails to GENERATE ("does not know the flags") long before
  anything compiles. The CUDA branch of the top-level CMakeLists teaches it the
  flag, but only after a `check_source_compiles` on `<expected>` — guarded on
  the variable being absent, so a newer CMake wins.
- **nvcc pins an older host compiler than the CXX build uses.** Here CUDA TUs go
  through GCC 15.3 while the CXX TUs use the system GCC 16.1, because nvcc 13.3
  refuses 16. Both link one libstdc++ and the whole suite passes; if that ever
  turns into a link or ODR failure, the fix is to build everything with the
  older one (`-DCMAKE_CXX_COMPILER=g++-15`), not to patch around it.
- **`std::min({a,b,c})` needs `<algorithm>` spelled out.** The 2-argument
  overloads arrive transitively via `<bits/stl_algobase.h>`; the
  initializer-list ones do not. clang/hipcc's include graph happened to supply
  it, nvcc's does not — so two `.hip` files were relying on luck, and now say
  `#include <algorithm>`.
- **No extended `__device__` lambdas.** nvcc cannot query one's return type
  from host code, which is exactly what a transform iterator does. Reduction
  inputs are named functors (`PlainSum`, `PlainSum64`, ...); keep them that way
  and `--extended-lambda` stays unnecessary.
- FP64 runs at 1/32 rate on consumer GeForce. The fields are FP32 and only the
  reductions are FP64, so this shows up as a modest per-step tax rather than a
  wall: 11.1 ms/step at 128³ on the RTX 3070.

Cross-checked, not assumed: `dns tgv --n 64 --re 1600` on CUDA reproduces the
stored HIP reference (`data/validation/ours_tgv64.csv`) to 3e-7 relative in K
and 1.6e-7 in epsVisc, at identical time steps -- i.e. FP32 round-off, not a
different discretisation. The full 55-test ctest suite passes.
