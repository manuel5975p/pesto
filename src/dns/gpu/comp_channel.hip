#include "dns/gpu/comp_channel.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/index.hpp"

namespace dns::gpu {

namespace {

// Ghost rows {0, ny-1} mirror rows {1, ny-2}; writes and reads are disjoint.
// Adiabatic (isothermal == 0): rho/E copied, ALL momentum antimirrored --
// the same arithmetic as comp_far's kFarWallY no-slip branch. Isothermal:
// the ghost thermodynamic state is built so the wall face holds T = Twall
// and dp/dn = 0, with the mirrored velocity: T0 = 2 Twall - T1 (floored at
// Twall/20 so a transient cannot construct a negative-temperature ghost),
// p0 = p1, rho0 = p0/(R T0), m0 = -rho0 u1, E0 from (p0, rho0, |u1|).
__global__ void kChannelWallY(GpuGrid g, int isothermal, Real gam, Real Rgas,
                              Real Twall, Real* rho, Real* mx, Real* my,
                              Real* mz, Real* E)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t nxz = std::size_t(g.nx) * g.nz;
    if (t >= 2 * nxz) return;
    const bool hi = t >= nxz;
    const int i = int(t % g.nx), k = int((t / g.nx) % g.nz);
    const std::size_t c = gidx(g, i, hi ? g.ny - 1 : 0, k);
    const std::size_t s = gidx(g, i, hi ? g.ny - 2 : 1, k);
    if (!isothermal)
    {
        rho[c] = rho[s];
        mx[c] = -mx[s];
        my[c] = -my[s];
        mz[c] = -mz[s];
        E[c] = E[s];
        return;
    }
    const Real r1 = rho[s];
    const Real u1 = mx[s] / r1, v1 = my[s] / r1, w1 = mz[s] / r1;
    const Real ke1 = Real(0.5) * (u1 * u1 + v1 * v1 + w1 * w1);
    const Real p1 = (gam - Real(1)) * (E[s] - r1 * ke1);
    const Real T1 = p1 / (Rgas * r1);
    const Real T0 = fmaxf(Real(2) * Twall - T1, Real(0.05) * Twall);
    const Real r0 = p1 / (Rgas * T0);
    rho[c] = r0;
    mx[c] = -r0 * u1;
    my[c] = -r0 * v1;
    mz[c] = -r0 * w1;
    E[c] = p1 / (gam - Real(1)) + r0 * ke1;
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
    if (scratch.size() < n)
    {
        auto b = DeviceBuffer<double>::make(n);
        if (!b) return b.error();
        scratch = std::move(*b);
    }
    kPlaneStats<<<launchBlocks(std::size_t(g.ny)), blockSize>>>(
        g, rho.data(), u.data(), v.data(), w.data(), T.data(), scratch.data());
    checkLastKernel("kPlaneStats");
    host.resize(n);
    return hipMemcpy(host.data(), scratch.data(), n * sizeof(double),
                     hipMemcpyDeviceToHost);
}

} // namespace dns::gpu
