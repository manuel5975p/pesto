// RAII device memory: move-only buffers and the staggered device field.
// GPU-only header.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <expected>
#include <utility>

namespace dns::gpu {

template <class T>
class DeviceBuffer
{
public:
    static std::expected<DeviceBuffer, hipError_t> make(std::size_t n)
    {
        T* p = nullptr;
        if (auto e = hipMalloc(&p, n * sizeof(T)); e != hipSuccess)
            return std::unexpected(e);
        return DeviceBuffer(p, n);
    }

    DeviceBuffer() = default;
    DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_), n_(o.n_)
    {
        o.p_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept
    {
        std::swap(p_, o.p_);
        std::swap(n_, o.n_);
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if (p_) (void)hipFree(p_);
    }

    T* data() { return p_; }
    const T* data() const { return p_; }
    std::size_t size() const { return n_; }

    // Full-buffer synchronous transfers; host must hold size() elements.
    hipError_t upload(const T* host)
    {
        return hipMemcpy(p_, host, n_ * sizeof(T), hipMemcpyHostToDevice);
    }
    hipError_t download(T* host) const
    {
        return hipMemcpy(host, p_, n_ * sizeof(T), hipMemcpyDeviceToHost);
    }

private:
    DeviceBuffer(T* p, std::size_t n) : p_(p), n_(n) {}

    T* p_ = nullptr;
    std::size_t n_ = 0;
};

using Real = float;
using DeviceScalar = DeviceBuffer<Real>;

// MAC face components on device; same layout/staggering as dns::VectorField.
struct DeviceField
{
    DeviceScalar u, v, w;

    static std::expected<DeviceField, hipError_t> make(std::size_t n)
    {
        auto a = DeviceScalar::make(n);
        if (!a) return std::unexpected(a.error());
        auto b = DeviceScalar::make(n);
        if (!b) return std::unexpected(b.error());
        auto c = DeviceScalar::make(n);
        if (!c) return std::unexpected(c.error());
        return DeviceField{std::move(*a), std::move(*b), std::move(*c)};
    }
};

} // namespace dns::gpu
