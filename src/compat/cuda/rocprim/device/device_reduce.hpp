// rocprim/device/device_reduce.hpp -- CUDA build only.
// See src/compat/cuda/README.md.
//
// nvcc-only (cub.cuh is device code); reachable exclusively from
// gpu/internal/reduce.hpp and the two .hip TUs that include it.
#pragma once

#include <cub/device/device_reduce.cuh>

#include <cstddef>

namespace rocprim {

// rocPRIM ships these as class templates in <rocprim/functional.hpp>, pulled in
// transitively. Spelled out rather than aliased to cuda::std::plus /
// cuda::maximum, whose header and name have moved between CCCL releases.
template <class T>
struct plus
{
    __host__ __device__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <class T>
struct maximum
{
    __host__ __device__ T operator()(const T& a, const T& b) const
    {
        return a < b ? b : a;
    }
};

template <class T>
struct minimum
{
    __host__ __device__ T operator()(const T& a, const T& b) const
    {
        return b < a ? b : a;
    }
};

// NOTE the argument order: rocPRIM puts the initial value BEFORE the count,
// CUB puts it after the operator. This reordering is the whole reason the
// wrapper is a function and not a macro.
template <class InputIt, class OutputIt, class InitType, class BinaryOp>
inline cudaError_t reduce(void* temporary_storage, std::size_t& storage_bytes,
                          InputIt input, OutputIt output, InitType initial_value,
                          std::size_t size, BinaryOp reduce_op,
                          cudaStream_t stream = 0)
{
    return cub::DeviceReduce::Reduce(temporary_storage, storage_bytes, input,
                                     output, size, reduce_op, initial_value,
                                     stream);
}

} // namespace rocprim
