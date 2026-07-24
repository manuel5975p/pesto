// Remap-aware derived fields: the PointRemap (dns/remap.hpp) is a TEMPLATE
// ARGUMENT of the kernel and travels by value, so the device evaluates the
// analytic mapping/Jacobian directly instead of assuming (i+1/2) h positions.
// GPU-internal header (HIP TUs only): templated __global__ kernels
// instantiate in the including TU.
#pragma once

#include "dns/gpu/device_buffer.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"
#include "dns/remap.hpp"

namespace dns::gpu {

// Centred z-vorticity dv/dx - du/dy on a remapped mesh: each finite
// difference is divided by the PHYSICAL distance between the two neighbouring
// cell centres, obtained from the remap evaluated at the UNWRAPPED reference
// coordinate. A periodic-compatible remap (map(X + P e_a) = map(X) + P e_a)
// then yields the correct periodic-image position by itself; TanhStretchY is
// merely smooth beyond the y extremes, which only its ghost rows see. With
// IdentityRemap this reduces exactly to the uniform kVorticityZCentred.
template <class Remap>
__global__ void kVorticityZRemap(GpuGrid g, Remap R, const Real* u,
                                 const Real* v, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    auto centre = [&](int ii, int jj, int kk, double x[3]) {
        const double X[3] = {(ii + 0.5) * double(g.hx), (jj + 0.5) * double(g.hy),
                             (kk + 0.5) * double(g.hz)};
        R.map(X, x);
    };
    double xp[3], xm[3];
    centre(i + 1, j, k, xp);
    centre(i - 1, j, k, xm);
    const double dx = xp[0] - xm[0];
    centre(i, j + 1, k, xp);
    centre(i, j - 1, k, xm);
    const double dy = xp[1] - xm[1];
    out[c] = Real((v[gidx(g, i + 1, j, k)] - v[gidx(g, i - 1, j, k)]) / dx -
                  (u[gidx(g, i, j + 1, k)] - u[gidx(g, i, j - 1, k)]) / dy);
}

template <class Remap>
void vorticityZRemap(const GpuGrid& g, const Remap& r, const DeviceScalar& u,
                     const DeviceScalar& v, DeviceScalar& out)
{
    kVorticityZRemap<<<launchBlocks(g.size()), blockSize>>>(g, r, u.data(),
                                                            v.data(), out.data());
    checkLastKernel("kVorticityZRemap");
}

} // namespace dns::gpu
