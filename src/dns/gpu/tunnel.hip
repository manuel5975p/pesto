#include "dns/gpu/tunnel.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/farpot.hpp"

#include <cmath>

namespace dns::gpu {
namespace {

// Ghost-ring boundary conditions, applied after maskSolidFaces zeroed every
// closed face. See DeviceTunnelBc's header comment for the physics; the corner
// treatment is diagonal and collapses exactly to the y-only form at nz == 1.
__global__ void kBcTunnel(GpuGrid g, int open, int slave, Real uInf, Real cdt,
                          const Real* inu, const Real* inv, const Real* inw,
                          Real* u, Real* v, Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const int nx = g.nx, ny = g.ny, nz = g.nz;

    // Free-slip mirror rows. In native 2D there is no z ghost ring, so km == k
    // identically and every z clause below degenerates to the y-only form.
    // The tangential mirror doubles as the open-lateral zero-gradient ghost,
    // so `open` only rewrites the NORMAL-component clauses.
    const int jm = j == 0 ? 1 : j == ny - 1 ? ny - 2 : j;
    const int km = nz == 1 ? k : (k == 0 ? 1 : k == nz - 1 ? nz - 2 : k);

    if (i <= 1) u[c] = uInf;
    else if (i == nx - 1)
        u[c] = inu[c] - cdt * (inu[c] - inu[gidx(g, nx - 2, j, k)]);
    else if (jm != j || km != k) u[c] = u[gidx(g, i, jm, km)];

    if (i == 0) v[c] = -v[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        v[c] = inv[c] - cdt * (inv[c] - inv[gidx(g, nx - 2, j, k)]);
    else if (open)
    {   // boundary faces and ghosts take the nearest interior face value:
        // dv/dy = 0, the lateral flux is free (vents the displacement field).
        // slave == 0 (the post-projection ghost refresh) keeps the boundary
        // FACES at the prescribed flux the projection just balanced and only
        // tracks the ghost row -- re-slaving there would undo div u = 0 in
        // the boundary cells.
        if (slave && j <= 1) v[c] = v[gidx(g, i, 2, km)];
        else if (slave && j == ny - 1) v[c] = v[gidx(g, i, ny - 2, km)];
        else if (!slave && j == 0) v[c] = v[gidx(g, i, 2, km)];
        else if (km != k) v[c] = v[gidx(g, i, j, km)];
    }
    else if (j == 1 || j == ny - 1) v[c] = 0; // impermeable wall faces
    else if (j == 0) v[c] = -v[gidx(g, i, 2, km)]; // antimirror across y = hy
    else if (km != k) v[c] = v[gidx(g, i, j, km)];

    if (i == 0) w[c] = -w[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        w[c] = inw[c] - cdt * (inw[c] - inw[gidx(g, nx - 2, j, k)]);
    // nz == 1: w is identically zero and k == 0 == nz - 1, so this clause
    // fires first and sets w = 0, exactly as the native-2D form did.
    else if (nz == 1) w[c] = 0;
    else if (open)
    {
        if (slave && k <= 1) w[c] = w[gidx(g, i, jm, 2)];
        else if (slave && k == nz - 1) w[c] = w[gidx(g, i, jm, nz - 2)];
        else if (!slave && k == 0) w[c] = w[gidx(g, i, jm, 2)];
        else if (jm != j) w[c] = w[gidx(g, i, jm, k)];
    }
    else if (k == 1 || k == nz - 1) w[c] = 0; // impermeable wall faces
    else if (k == 0) w[c] = -w[gidx(g, i, jm, 2)]; // antimirror across z = hz
    else if (jm != j) w[c] = w[gidx(g, i, jm, k)];
}

// Potential-corrected inlet: overwrite the two Dirichlet u columns and the
// i == 0 v/w ghosts with the freestream + wake-source + doublet target at the
// staggered MAC face positions. Launched AFTER kBcTunnel (sequenced, so the
// ghost antimirror reads settled values -- folding this into kBcTunnel
// perturbed its codegen and broke the pinned bit-identity with the historical
// kernel). One thread per (j, k) inlet line. Q = 0 and dbl = 0 rewrite the
// identical uInf / antimirror values.
__global__ void kPotInlet(GpuGrid g, Real Q, Real dbl, Real xc, Real yc,
                          Real zc, Real uInf, Real* u, Real* v, Real* w)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= std::size_t(g.ny) * g.nz) return;
    const int j = int(t % g.ny), k = int(t / g.ny);
    const bool twoD = g.nz == 1;
    const Real rmin2 = Real(4) * g.hx * g.hx;
    Real du, dv, dw;
    for (int i = 0; i <= 1; ++i)
    {   // u face at (i hx, (j+1/2) hy, (k+1/2) hz)
        farPotential(twoD, Real(i) * g.hx - xc,
                     (Real(j) + Real(0.5)) * g.hy - yc,
                     (Real(k) + Real(0.5)) * g.hz - zc, Q, dbl, rmin2, du, dv,
                     dw);
        u[gidx(g, i, j, k)] = uInf + du;
    }
    // v ghost antimirrors about the target on the inlet plane x = hx.
    farPotential(twoD, g.hx - xc, Real(j) * g.hy - yc,
                 (Real(k) + Real(0.5)) * g.hz - zc, Q, dbl, rmin2, du, dv, dw);
    v[gidx(g, 0, j, k)] = Real(2) * dv - v[gidx(g, 1, j, k)];
    if (g.nz > 1)
    {
        farPotential(twoD, g.hx - xc, (Real(j) + Real(0.5)) * g.hy - yc,
                     Real(k) * g.hz - zc, Q, dbl, rmin2, du, dv, dw);
        w[gidx(g, 0, j, k)] = Real(2) * dw - w[gidx(g, 1, j, k)];
    }
}

// Per-cell tally of the outflux so one sumScalar yields the volume flux: the
// outlet-plane u, plus (open laterals) the signed lateral face fluxes scaled
// to u-equivalent units by wy = hx/hy, wz = hx/hz (the caller multiplies the
// sum by hy*hz). open == 0 writes exactly the old outlet-plane tally.
__global__ void kOutletTally(GpuGrid g, int open, int pot, Real wy, Real wz,
                             const Real* u, const Real* v, const Real* w,
                             Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const bool inJ = j >= 1 && j <= g.ny - 2;
    const bool inK = k >= g.kInLo() && k <= g.kInHi();
    Real q = i == g.nx - 1 && inJ && inK ? u[c] : Real(0);
    // pot: the inlet flux varies per face; tally the NET outflux and let the
    // host close corr = -net / A (the uInf * A shortcut assumed a uniform
    // inlet).
    if (pot && i == 1 && inJ && inK) q -= u[c];
    if (open && i >= 1 && i <= g.nx - 2)
    {
        if (inK && j == 1) q -= wy * v[c];
        if (inK && j == g.ny - 1) q += wy * v[c];
        if (g.nz > 1 && inJ && k == 1) q -= wz * w[c];
        if (g.nz > 1 && inJ && k == g.nz - 1) q += wz * w[c];
    }
    out[c] = q;
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

// The inlet/outlet faces (and the open lateral faces) carry prescribed fluxes
// but zero apertures, so divergenceCut misses them; add the missing terms to
// the projection RHS. A corner cell owns several boundary faces and gets one
// term per face.
__global__ void kInOutFluxRhs(GpuGrid g, int open, int pot, Real scale,
                              Real uInf, const Real* u, const Real* v,
                              const Real* w, Real* rhs)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (j < 1 || j > g.ny - 2 || k < g.kInLo() || k > g.kInHi()) return;
    if (i == 1) rhs[c] -= scale * (pot ? u[c] : uInf) / g.hx;
    else if (i == g.nx - 2) rhs[c] += scale * u[gidx(g, g.nx - 1, j, k)] / g.hx;
    if (!open || i < 1 || i > g.nx - 2) return;
    if (j == 1) rhs[c] -= scale * v[c] / g.hy;
    if (j == g.ny - 2) rhs[c] += scale * v[gidx(g, i, g.ny - 1, k)] / g.hy;
    if (g.nz > 1)
    {
        if (k == 1) rhs[c] -= scale * w[c] / g.hz;
        if (k == g.nz - 2) rhs[c] += scale * w[gidx(g, i, j, g.nz - 1)] / g.hz;
    }
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
    if (c.pot && (!c.openLat || !(c.aBody >= 0) || !std::isfinite(c.aBody)
                  || !std::isfinite(c.xc) || !std::isfinite(c.yc)
                  || !std::isfinite(c.zc)))
        return std::unexpected(hipErrorInvalidValue);
    return DeviceTunnelBc(g, c);
}

void DeviceTunnelBc::apply(const DeviceField& in, DeviceField& out, double cdt,
                           bool refreshOnly) const
{
    kBcTunnel<<<launchBlocks(g_.size()), blockSize>>>(
        g_, int(cfg_.openLat), int(!refreshOnly), Real(cfg_.uInf), Real(cdt),
        in.u.data(), in.v.data(), in.w.data(), out.u.data(), out.v.data(),
        out.w.data());
    checkLastKernel("bcTunnel");
    if (cfg_.pot)
    {   // Doublet strength: 2D uInf a^2, 3D uInf a^3 / 2.
        const double a2 = cfg_.aBody * cfg_.aBody;
        const double dbl =
            g_.nz == 1 ? cfg_.uInf * a2 : 0.5 * cfg_.uInf * a2 * cfg_.aBody;
        const std::size_t lines = std::size_t(g_.ny) * g_.nz;
        kPotInlet<<<launchBlocks(lines), blockSize>>>(
            g_, Real(Q_), Real(dbl), Real(cfg_.xc), Real(cfg_.yc),
            Real(cfg_.zc), Real(cfg_.uInf), out.u.data(), out.v.data(),
            out.w.data());
        checkLastKernel("potInlet");
    }
}

void DeviceTunnelBc::correctOutletFlux(DeviceField& out, ReduceScratch& red,
                                       DeviceScalar& tmp) const
{
    kOutletTally<<<launchBlocks(g_.size()), blockSize>>>(
        g_, int(cfg_.openLat), int(cfg_.pot), Real(g_.hx / g_.hy),
        Real(g_.hx / g_.hz), out.u.data(), out.v.data(), out.w.data(),
        tmp.data());
    checkLastKernel("outletTally");
    const double planeArea =
        double(g_.ny - 2) * g_.kInCount() * double(g_.hy) * double(g_.hz);
    const double qOut = sumScalar(tmp, red) * double(g_.hy) * double(g_.hz);
    // pot: qOut is already the NET outflux (inlet tallied negative).
    const double corr = cfg_.pot ? -qOut / planeArea
                                 : (cfg_.uInf * planeArea - qOut) / planeArea;
    kOutletCorrect<<<launchBlocks(g_.size()), blockSize>>>(g_, Real(corr), out.u.data());
    checkLastKernel("outletCorrect");
}

void DeviceTunnelBc::addFluxRhs(const DeviceField& U, double scale,
                                DeviceScalar& rhs) const
{
    kInOutFluxRhs<<<launchBlocks(g_.size()), blockSize>>>(
        g_, int(cfg_.openLat), int(cfg_.pot), Real(scale), Real(cfg_.uInf),
        U.u.data(), U.v.data(), U.w.data(), rhs.data());
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
