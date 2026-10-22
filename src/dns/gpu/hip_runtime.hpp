// <hip/hip_runtime.h> for headers a host compiler also sees.
// ROCm's non-clang branch (host_defines.h) defines __noinline__ EMPTY, which
// turns libstdc++ 16's [[__gnu__::__noinline__]] (<format>) into a hard error
// for every host .cpp that includes <format>/<chrono> after a GPU header.
// The project never spells __noinline__, so dropping the empty macro is safe.
#pragma once

#include <hip/hip_runtime.h>

#if !(defined(__clang__) && defined(__HIP__)) && defined(__noinline__)
#undef __noinline__
#endif
