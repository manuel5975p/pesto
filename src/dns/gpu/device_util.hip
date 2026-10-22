#include "dns/gpu/device_util.hpp"

#include <vector>

namespace dns::gpu {

bool deviceAvailable()
{
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

hipError_t uploadNarrow(const std::vector<double>& src, DeviceScalar& dst)
{
    if (src.size() != dst.size()) return hipErrorInvalidValue;
    std::vector<Real> tmp(src.begin(), src.end()); // narrow to Real
    return dst.upload(tmp.data());
}

hipError_t downloadWiden(const DeviceScalar& src, std::vector<double>& dst)
{
    if (src.size() != dst.size()) return hipErrorInvalidValue;
    std::vector<Real> tmp(src.size());
    if (auto e = src.download(tmp.data()); e != hipSuccess) return e;
    dst.assign(tmp.begin(), tmp.end());
    return hipSuccess;
}

std::expected<DeviceScalar, hipError_t> makeNarrow(const std::vector<double>& src)
{
    auto buf = DeviceScalar::make(src.size());
    if (!buf) return buf;
    if (auto e = uploadNarrow(src, *buf); e != hipSuccess) return std::unexpected(e);
    return buf;
}

hipError_t uploadField(const VectorField& src, DeviceField& dst)
{
    if (auto e = uploadNarrow(src.u, dst.u); e != hipSuccess) return e;
    if (auto e = uploadNarrow(src.v, dst.v); e != hipSuccess) return e;
    return uploadNarrow(src.w, dst.w);
}

hipError_t downloadField(const DeviceField& src, VectorField& dst)
{
    if (auto e = downloadWiden(src.u, dst.u); e != hipSuccess) return e;
    if (auto e = downloadWiden(src.v, dst.v); e != hipSuccess) return e;
    return downloadWiden(src.w, dst.w);
}

namespace {

__global__ void kCopyField(std::size_t n, const Real* __restrict__ su,
                           const Real* __restrict__ sv, const Real* __restrict__ sw,
                           Real* __restrict__ du, Real* __restrict__ dv,
                           Real* __restrict__ dw)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    du[c] = su[c];
    dv[c] = sv[c];
    dw[c] = sw[c];
}

} // namespace

hipError_t copyField(DeviceField& dst, const DeviceField& src)
{
    const std::size_t n = src.u.size();
    if (src.v.size() != n || src.w.size() != n || dst.u.size() != n
        || dst.v.size() != n || dst.w.size() != n)
        return hipErrorInvalidValue;
    if (n == 0) return hipSuccess;
    kCopyField<<<launchBlocks(n), blockSize>>>(n, src.u.data(), src.v.data(),
                                               src.w.data(), dst.u.data(),
                                               dst.v.data(), dst.w.data());
    return hipGetLastError();
}

} // namespace dns::gpu
