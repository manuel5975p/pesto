#include "dns/gpu/tunnel.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>

namespace dns::gpu {
namespace {

// Ghost-ring boundary conditions, applied after maskSolidFaces zeroed every
// closed face. See DeviceTunnelBc's header comment for the physics; the corner
// treatment is diagonal and collapses exactly to the y-only form at nz == 1.
__global__ void kBcTunnel(GpuGrid g, Real uInf, Real cdt, const Real* inu,
                          const Real* inv, const Real* inw, Real* u, Real* v,
                          Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const int nx = g.nx, ny = g.ny, nz = g.nz;

    // Free-slip mirror rows. In native 2D there is no z ghost ring, so km == k
    // identically and every z clause below degenerates to the y-only form.
    const int jm = j == 0 ? 1 : j == ny - 1 ? ny - 2 : j;
    const int km = nz == 1 ? k : (k == 0 ? 1 : k == nz - 1 ? nz - 2 : k);

    if (i <= 1) u[c] = uInf;
    else if (i == nx - 1)
        u[c] = inu[c] - cdt * (inu[c] - inu[gidx(g, nx - 2, j, k)]);
    else if (jm != j || km != k) u[c] = u[gidx(g, i, jm, km)];

    if (i == 0) v[c] = -v[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        v[c] = inv[c] - cdt * (inv[c] - inv[gidx(g, nx - 2, j, k)]);
    else if (j == 1 || j == ny - 1) v[c] = 0; // impermeable wall faces
    else if (j == 0) v[c] = -v[gidx(g, i, 2, km)]; // antimirror across y = hy
    else if (km != k) v[c] = v[gidx(g, i, j, km)];

    if (i == 0) w[c] = -w[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        w[c] = inw[c] - cdt * (inw[c] - inw[gidx(g, nx - 2, j, k)]);
    // nz == 1: w is identically zero and k == 0 == nz - 1, so this clause
    // fires first and sets w = 0, exactly as the native-2D form did.
    else if (nz == 1 || k == 1 || k == nz - 1) w[c] = 0; // impermeable wall faces
    else if (k == 0) w[c] = -w[gidx(g, i, jm, 2)]; // antimirror across z = hz
    else if (jm != j) w[c] = w[gidx(g, i, jm, k)];
}

// Per-cell tally of the outlet-plane u so one sumScalar yields the volume flux.
__global__ void kOutletTally(GpuGrid g, const Real* u, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const bool plane = i == g.nx - 1 && j >= 1 && j <= g.ny - 2
                    && k >= g.kInLo() && k <= g.kInHi();
    out[c] = plane ? u[c] : Real(0);
}

// u_outlet += corr on the open outlet plane (exact global mass balance).
__global__ void kOutletCorrect(GpuGrid g, Real corr, Real* u)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (i == g.nx - 1 && j >= 1 && j <= g.ny - 2 && k >= g.kInLo()
        && k <= g.kInHi())
        u[c] += corr;
}

// The inlet/outlet faces carry prescribed fluxes but zero apertures, so
// divergenceCut misses them; add the missing terms to the projection RHS.
__global__ void kInOutFluxRhs(GpuGrid g, Real scale, Real uInf, const Real* u,
                              Real* rhs)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (j < 1 || j > g.ny - 2 || k < g.kInLo() || k > g.kInHi()) return;
    if (i == 1) rhs[c] -= scale * uInf / g.hx;
    else if (i == g.nx - 2) rhs[c] += scale * u[gidx(g, g.nx - 1, j, k)] / g.hx;
}

// Open far-field fringe sponge: relax the stage velocity toward the free
// stream over cosine-ramped bands along the banded domain boundaries.
__global__ void kFringe(GpuGrid g, int ns, int axes, Real sigMax, Real uInf,
                        Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    auto depth = [ns](int q, int nq) {
        if (q < ns) return Real(ns - q) / Real(ns);
        if (q >= nq - ns) return Real(q - (nq - ns) + 1) / Real(ns);
        return Real(0);
    };
    Real r = max(depth(i, g.nx), depth(j, g.ny));
    if (axes == 3) r = max(r, depth(k, g.nz));
    if (r <= 0) return;
    const Real sig = sigMax * Real(0.5) * (Real(1) - cosf(Real(M_PI) * r));
    u[c] += sig * (uInf - u[c]);
    v[c] -= sig * v[c];
    w[c] -= sig * w[c];
}

} // namespace

// ---------------------------------------------------------------------------

std::expected<DeviceTunnelBc, hipError_t>
DeviceTunnelBc::make(const GpuGrid& g, Config c)
{
    if (g.nx < 4 || g.ny < 4 || !(c.uInf > 0) || !std::isfinite(c.uInf))
        return std::unexpected(hipErrorInvalidValue);
    if (g.nz != 1 && g.nz < 4) return std::unexpected(hipErrorInvalidValue);
    return DeviceTunnelBc(g, c);
}

void DeviceTunnelBc::apply(const DeviceField& in, DeviceField& out, double cdt) const
{
    kBcTunnel<<<launchBlocks(g_.size()), blockSize>>>(
        g_, Real(cfg_.uInf), Real(cdt), in.u.data(), in.v.data(), in.w.data(),
        out.u.data(), out.v.data(), out.w.data());
    checkLastKernel("bcTunnel");
}

void DeviceTunnelBc::correctOutletFlux(DeviceField& out, ReduceScratch& red,
                                       DeviceScalar& tmp) const
{
    kOutletTally<<<launchBlocks(g_.size()), blockSize>>>(g_, out.u.data(), tmp.data());
    checkLastKernel("outletTally");
    const double planeArea =
        double(g_.ny - 2) * g_.kInCount() * double(g_.hy) * double(g_.hz);
    const double qOut = sumScalar(tmp, red) * double(g_.hy) * double(g_.hz);
    const double corr = (cfg_.uInf * planeArea - qOut) / planeArea;
    kOutletCorrect<<<launchBlocks(g_.size()), blockSize>>>(g_, Real(corr), out.u.data());
    checkLastKernel("outletCorrect");
}

void DeviceTunnelBc::addFluxRhs(const DeviceField& U, double scale,
                                DeviceScalar& rhs) const
{
    kInOutFluxRhs<<<launchBlocks(g_.size()), blockSize>>>(
        g_, Real(scale), Real(cfg_.uInf), U.u.data(), rhs.data());
    checkLastKernel("inOutFluxRhs");
}

// ---------------------------------------------------------------------------

std::expected<DeviceFringe, hipError_t>
DeviceFringe::make(const GpuGrid& g, Config c)
{
    if (c.cells < 1 || c.cells >= g.nx / 2 || c.cells >= g.ny / 2)
        return std::unexpected(hipErrorInvalidValue);
    if (c.axes != 2 && c.axes != 3) return std::unexpected(hipErrorInvalidValue);
    if (c.axes == 3 && g.nz < 2 * c.cells + 2)
        return std::unexpected(hipErrorInvalidValue);
    if (!std::isfinite(c.sigMax) || c.sigMax <= 0)
        return std::unexpected(hipErrorInvalidValue);
    return DeviceFringe(g, c);
}

void DeviceFringe::apply(DeviceField& U) const
{
    kFringe<<<launchBlocks(g_.size()), blockSize>>>(
        g_, cfg_.cells, cfg_.axes, Real(cfg_.sigMax), Real(cfg_.uInf), U.u.data(),
        U.v.data(), U.w.data());
    checkLastKernel("fringe");
}

} // namespace dns::gpu
