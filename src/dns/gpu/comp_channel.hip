#include "dns/gpu/comp_channel.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/stage_write.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/wall_ghost.hpp"

#include <cassert>

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
    // The driving force is along x, tangential to both plates: f . n == 0.
    else internal::isothermalGhost(c, s, gam, Rgas, Twall, Real(0), rho, mx, my,
                                   mz, E);
}

// Additive rhs contribution: the stage kernel already wrote the interior
// combination, so the force enters scaled by the same crAt(c) (comp.hpp,
// StageCombine) that multiplied the interior rhs. Comp: the compensated write
// (internal/stage_write.hpp).
template <bool Comp>
__global__ void kBodyForceX(GpuGrid g, Real f, StageCombine cmb, const Real* u,
                            Real* omx, Real* oE)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (j < 1 || j > g.ny - 2) return; // ghost rows: the stage hook owns them
    const Real cr = cmb.crAt(c);
    internal::stageAdd<Comp>(cmb, 1, c, cr, f, omx);
    internal::stageAdd<Comp>(cmb, 4, c, cr * f, u[c], oE);
}

// The wall's ghost cells are SET, not accumulated: their low words are zero,
// so the compensated pair holds exactly the value the wall wrote.
struct GhostLo
{
    Real* p[7];
    int n;
};

__global__ void kChannelGhostLo(GpuGrid g, GhostLo lo)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, k;
    bool hi;
    std::size_t c, s;
    if (!internal::yGhostPair(g, t, i, k, hi, c, s)) return;
    for (int v = 0; v < lo.n; ++v) lo.p[v][c] = Real(0);
}

// The SA ghost scalar on both plates: mirror the SPECIFIC nu~ odd and rebuild
// the conserved value with the ghost's OWN density (the isothermal ghost has
// rho_g != rho_1), as the momentum mirror does. Runs after kChannelWallY.
__global__ void kChannelWallTurb(GpuGrid g, const Real* rho, Real* t0)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, k;
    bool hi;
    std::size_t c, s;
    if (!internal::yGhostPair(g, t, i, k, hi, c, s)) return;
    t0[c] = -rho[c] * t0[s] / rho[s];
}

// The SST ghost pair: k antimirrored like nu~; omega a DIRICHLET image of the
// TMR wall value, omega_g = 2 omega_w - omega_1, so the wall face holds
// omega_w = coef / rho_w (nu_w = mu / rho_w on the face).
__global__ void kChannelWallSst(GpuGrid g, Real coef, const Real* rho, Real* t0,
                                Real* t1)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i, k;
    bool hi;
    std::size_t c, s;
    if (!internal::yGhostPair(g, t, i, k, hi, c, s)) return;
    t0[c] = -rho[c] * t0[s] / rho[s];
    const Real ww = coef / (Real(0.5) * (rho[c] + rho[s]));
    t1[c] = rho[c] * (Real(2) * ww - t1[s] / rho[s]);
}

// The 5 + nT field pointers of a state, DeviceCompState order.
template <class P> struct FieldPtrs
{
    P p[7];
    int n;
};

template <class S> auto fieldPtrs(S& s)
{
    FieldPtrs<decltype(s.rho.data())> f{
        {s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(), s.E.data()},
        5 + s.nT};
    for (int i = 0; i < s.nT; ++i) f.p[5 + i] = s.t[i].data();
    return f;
}

// One thread per y row: column (0, j, 0) of every field, field-major.
__global__ void kChannelGatherColumn(GpuGrid g, FieldPtrs<const Real*> f,
                                     Real* packed)
{
    const int j = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (j >= g.ny) return;
    const std::size_t c = gidx(g, 0, j, 0);
    for (int v = 0; v < f.n; ++v) packed[std::size_t(v) * g.ny + j] = f.p[v][c];
}

// Every fluid-row cell takes its row's packed value; ghost rows untouched.
__global__ void kChannelSetRows(GpuGrid g, const Real* packed,
                                FieldPtrs<Real*> f)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    if (j < 1 || j > g.ny - 2) return;
    for (int v = 0; v < f.n; ++v) f.p[v][c] = packed[std::size_t(v) * g.ny + j];
}

hipError_t ensurePacked(DeviceBuffer<Real>& packed, std::size_t n)
{
    if (packed.size() == n) return hipSuccess;
    auto b = DeviceBuffer<Real>::make(n);
    if (!b) return b.error();
    packed = std::move(*b);
    return hipSuccess;
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

// Sum and max of one field per row, same one-thread-per-row fixed order.
__global__ void kPlaneScalar(GpuGrid g, const Real* f, double* out)
{
    const int j = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (j >= g.ny) return;
    double s = 0.0, m = -HUGE_VAL;
    for (int k = 0; k < g.nz; ++k)
        for (int i = 0; i < g.nx; ++i) {
            const double v = f[gidx(g, i, j, k)];
            s += v;
            m = v > m ? v : m;
        }
    out[std::size_t(j) * 2] = s;
    out[std::size_t(j) * 2 + 1] = m;
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
    const std::size_t nb = launchBlocks(2 * std::size_t(g_.nx) * g_.nz);
    if (s.compensated()) {
        GhostLo lo{{}, 5 + s.nT};
        for (int v = 0; v < 5; ++v) lo.p[v] = s.lo[v].data();
        for (int i = 0; i < s.nT; ++i) lo.p[5 + i] = s.tlo[i].data();
        kChannelGhostLo<<<nb, blockSize>>>(g_, lo);
        checkLastKernel("kChannelGhostLo");
    }
    if (s.nT == 0) return;
    if (s.nT == 1) {
        kChannelWallTurb<<<nb, blockSize>>>(g_, s.rho.data(), s.t[0].data());
        checkLastKernel("kChannelWallTurb");
        return;
    }
    assert(s.nT == 2 && c_.sstOmegaCoef > 0);
    kChannelWallSst<<<nb, blockSize>>>(g_, Real(c_.sstOmegaCoef), s.rho.data(),
                                       s.t[0].data(), s.t[1].data());
    checkLastKernel("kChannelWallSst");
}

void addBodyForceX(const GpuGrid& g, double f, const StageCombine& cmb,
                   const DeviceScalar& u, DeviceCompState& out)
{
    internal::withCompensation(cmb, [&](auto comp) {
        kBodyForceX<decltype(comp)::value><<<launchBlocks(g.size()), blockSize>>>(
            g, Real(f), cmb, u.data(), out.mx.data(), out.E.data());
    });
    checkLastKernel("kBodyForceX");
}

hipError_t channelGatherColumn(const GpuGrid& g, const DeviceCompState& s,
                               DeviceBuffer<Real>& packed,
                               std::vector<Real>& host)
{
    const auto f = fieldPtrs(s);
    const std::size_t n = std::size_t(f.n) * g.ny;
    if (auto e = ensurePacked(packed, n); e != hipSuccess) return e;
    kChannelGatherColumn<<<launchBlocks(std::size_t(g.ny)), blockSize>>>(
        g, f, packed.data());
    checkLastKernel("kChannelGatherColumn");
    host.resize(n);
    return packed.download(host.data());
}

hipError_t channelSetRows(const GpuGrid& g, const std::vector<Real>& host,
                          DeviceBuffer<Real>& packed, DeviceCompState& s)
{
    assert(!s.compensated()); // a plain-state edit (see boostX)
    const auto f = fieldPtrs(s);
    const std::size_t n = std::size_t(f.n) * g.ny;
    assert(host.size() == n);
    if (auto e = ensurePacked(packed, n); e != hipSuccess) return e;
    if (auto e = packed.upload(host.data()); e != hipSuccess) return e;
    kChannelSetRows<<<launchBlocks(g.size()), blockSize>>>(g, packed.data(), f);
    checkLastKernel("kChannelSetRows");
    return hipSuccess;
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

hipError_t channelPlaneScalarSums(const GpuGrid& g, const DeviceScalar& field,
                                  DeviceBuffer<double>& scratch,
                                  std::vector<double>& host)
{
    const std::size_t n = std::size_t(g.ny) * 2;
    if (auto e = internal::ensurePlaneScratch(scratch, n); e != hipSuccess)
        return e;
    kPlaneScalar<<<launchBlocks(std::size_t(g.ny)), blockSize>>>(
        g, field.data(), scratch.data());
    checkLastKernel("kPlaneScalar");
    return internal::fetchPlaneScratch(scratch, n, host);
}

} // namespace dns::gpu
