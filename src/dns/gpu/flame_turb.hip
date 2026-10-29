#include "dns/gpu/flame_turb.hpp"

#include "dns/gpu/device_util.hpp"

#include <cassert>
#include <cmath>

namespace dns::gpu {

namespace {

__global__ void kTurbInflowRows(int nbx, int nrows, int i0, int i1, Real f,
                                const Real* tu, const Real* tv, const Real* tw,
                                Real* du, Real* dv, Real* dw)
{
    const int row = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (row >= nrows) return;
    const std::size_t a = std::size_t(i0) + std::size_t(nbx) * row;
    const std::size_t b = std::size_t(i1) + std::size_t(nbx) * row;
    du[row] = fmaf(f, tu[b] - tu[a], tu[a]);
    dv[row] = fmaf(f, tv[b] - tv[a], tv[a]);
    dw[row] = fmaf(f, tw[b] - tw[a], tw[a]);
}

} // namespace

TurbSlice turbSliceOf(double X, double h, int nbx)
{
    assert(h > 0 && nbx >= 1 && std::isfinite(X));
    double s = std::fmod(-X / h, double(nbx));
    if (s < 0) s += nbx;
    int i0 = int(std::floor(s));
    if (i0 >= nbx) i0 = nbx - 1; // s rounded up to nbx
    const float f = float(s - i0);
    return {i0, (i0 + 1) % nbx, f};
}

std::expected<DeviceTurbInflow, hipError_t>
DeviceTurbInflow::make(int nbx, int ny, int nz, double h,
                       const std::array<std::vector<double>, 3>& box)
{
    const std::size_t rows = std::size_t(ny) * nz;
    const std::size_t n = std::size_t(nbx) * rows;
    assert(box[0].size() == n && box[1].size() == n && box[2].size() == n);
    DeviceTurbInflow t;
    t.nbx_ = nbx;
    t.nrows_ = int(rows);
    t.h_ = h;
    DeviceScalar* tab[3] = {&t.tu_, &t.tv_, &t.tw_};
    DeviceScalar* row[3] = {&t.du_, &t.dv_, &t.dw_};
    std::vector<Real> f(n);
    const std::vector<Real> zero(rows, Real(0));
    for (int c = 0; c < 3; ++c) {
        auto a = DeviceScalar::make(n);
        auto b = DeviceScalar::make(rows);
        if (!a) return std::unexpected(a.error());
        if (!b) return std::unexpected(b.error());
        *tab[c] = std::move(*a);
        *row[c] = std::move(*b);
        for (std::size_t q = 0; q < n; ++q) f[q] = Real(box[std::size_t(c)][q]);
        if (auto e = tab[c]->upload(f.data()); e != hipSuccess)
            return std::unexpected(e);
        if (auto e = row[c]->upload(zero.data()); e != hipSuccess)
            return std::unexpected(e);
    }
    return t;
}

void DeviceTurbInflow::sample(double X)
{
    const TurbSlice s = turbSliceOf(X, h_, nbx_);
    kTurbInflowRows<<<launchBlocks(std::size_t(nrows_)), blockSize>>>(
        nbx_, nrows_, s.i0, s.i1, s.f, tu_.data(), tv_.data(), tw_.data(),
        du_.data(), dv_.data(), dw_.data());
    checkLastKernel("kTurbInflowRows");
}

} // namespace dns::gpu
