// rocprim/iterator/transform_iterator.hpp -- CUDA build only.
// See src/compat/cuda/README.md.
#pragma once

#include <thrust/iterator/transform_iterator.h>

namespace rocprim {

// rocPRIM and thrust agree on the argument order (iterator, then function).
using thrust::make_transform_iterator;

} // namespace rocprim
