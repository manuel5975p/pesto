// rocprim/iterator/counting_iterator.hpp -- CUDA build only.
// See src/compat/cuda/README.md.
#pragma once

#include <thrust/iterator/counting_iterator.h>

namespace rocprim {

// Same contract: a random-access iterator yielding i at position i, usable as
// a CUB/rocPRIM device input.
template <class T>
using counting_iterator = thrust::counting_iterator<T>;

} // namespace rocprim
