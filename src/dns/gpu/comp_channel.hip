#include "dns/gpu/comp_channel.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/wall_ghost.hpp"

namespace dns::gpu {

namespace {

// The two plates. Both ghost states are the shared no-slip wall
// (internal/wall_ghost.hpp); this kernel only picks which one.
__global__ void kChannelWallY(GpuGrid g, int isothermal, Real gam, Real Rgas,
                              Real Twall, Real* rho, Real* mx, Real* my,
                              Real* mz, Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, k;
    bool hi;
    std::size_t c, s;
    if (!internal::yGhostPair(g, t, i, k, hi, c, s)) return;
    if (!isothermal) internal::adiabaticGhost(c, s, rho, mx, my, mz, E);
    else internal::isothermalGhost(c, s, gam, Rgas, Twall, rho, mx, my, mz, E);
}

// Additive rhs contribution: the stage kernel already wrote the interior
// combination, so the force enters scaled by the same cr (comp.hpp,
// StageCombine) that multiplied the interior rhs.
__global__ void kBodyForceX(GpuGrid g, Real f, Real cr, const Real* u,
                            Real* omx, Real* oE)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (j < 1 || j > g.ny - 2) return; // ghost rows: the stage hook owns them
    omx[c] += cr * f;
    oE[c] += cr * f * u[c];
}

// One thread per y row; each sums its whole (i,k) plane in FP64 in a fixed
// order. ~ny threads is a tiny launch, but the plane sums are exact-order
// deterministic, which the ground rules require of anything reported.
__global__ void kPlaneStats(GpuGrid g, const Real* rho, const Real* u,
                            const Real* v, const Real* w, const Real* T,
                            double* out)
{
    const int j = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (j >= g.ny) return;
    double s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < g.nz; ++k)
        for (int i = 0; i < g.nx; ++i)
        {
            const std::size_t c = gidx(g, i, j, k);
            const double r = rho[c];
            const double uu = u[c], vv = v[c], ww = w[c];
            s[0] += r;
            s[1] += r * uu;
            s[2] += r * vv;
            s[3] += r * uu * uu;
            s[4] += r * vv * vv;
            s[5] += r * ww * ww;
            s[6] += r * uu * vv;
            s[7] += r * double(T[c]);
        }
    for (int q = 0; q < 8; ++q) out[std::size_t(j) * 8 + q] = s[q];
}

} // namespace

std::expected<DeviceChannelWall, hipError_t> DeviceChannelWall::make(const Grid& g,
                                                                     Config c)
{
    if (g.ny < 4 || !(c.gamma > 1) || !(c.Rgas > 0) || !(c.Twall > 0))
        return std::unexpected(hipErrorInvalidValue);
    return DeviceChannelWall(GpuGrid::from(g), c);
}

void DeviceChannelWall::apply(DeviceCompState& s) const
{
    kChannelWallY<<<launchBlocks(2 * std::size_t(g_.nx) * g_.nz), blockSize>>>(
        g_, int(c_.isothermal), Real(c_.gamma), Real(c_.Rgas), Real(c_.Twall),
        s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data());
    checkLastKernel("kChannelWallY");
}

void addBodyForceX(const GpuGrid& g, double f, Real cr, const DeviceScalar& u,
                   DeviceCompState& out)
{
    kBodyForceX<<<launchBlocks(g.size()), blockSize>>>(
        g, Real(f), cr, u.data(), out.mx.data(), out.E.data());
    checkLastKernel("kBodyForceX");
}

hipError_t channelPlaneStats(const GpuGrid& g, const DeviceScalar& rho,
                             const DeviceScalar& u, const DeviceScalar& v,
                             const DeviceScalar& w, const DeviceScalar& T,
                             DeviceBuffer<double>& scratch,
                             std::vector<double>& host)
{
    const std::size_t n = std::size_t(g.ny) * 8;
    if (auto e = internal::ensurePlaneScratch(scratch, n); e != hipSuccess)
        return e;
    kPlaneStats<<<launchBlocks(std::size_t(g.ny)), blockSize>>>(
        g, rho.data(), u.data(), v.data(), w.data(), T.data(), scratch.data());
    checkLastKernel("kPlaneStats");
    return internal::fetchPlaneScratch(scratch, n, host);
}

} // namespace dns::gpu
