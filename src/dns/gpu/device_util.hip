#include "dns/gpu/device_util.hpp"

#include <vector>

namespace dns::gpu {

bool deviceAvailable()
{
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

namespace {

hipError_t uploadScalar(const Scalar& src, DeviceScalar& dst)
{
    assert(src.size() == dst.size());
    std::vector<Real> tmp(src.begin(), src.end()); // narrow to Real
    return dst.upload(tmp.data());
}

hipError_t downloadScalar(const DeviceScalar& src, Scalar& dst)
{
    assert(dst.size() == src.size());
    std::vector<Real> tmp(src.size());
    if (auto e = src.download(tmp.data()); e != hipSuccess) return e;
    dst.assign(tmp.begin(), tmp.end());
    return hipSuccess;
}

} // namespace

hipError_t uploadField(const VectorField& src, DeviceField& dst)
{
    if (auto e = uploadScalar(src.u, dst.u); e != hipSuccess) return e;
    if (auto e = uploadScalar(src.v, dst.v); e != hipSuccess) return e;
    return uploadScalar(src.w, dst.w);
}

hipError_t downloadField(const DeviceField& src, VectorField& dst)
{
    if (auto e = downloadScalar(src.u, dst.u); e != hipSuccess) return e;
    if (auto e = downloadScalar(src.v, dst.v); e != hipSuccess) return e;
    return downloadScalar(src.w, dst.w);
}

} // namespace dns::gpu
