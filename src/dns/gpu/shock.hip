#include "dns/gpu/shock.hpp"

#include "dns/gpu/comp.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/internal/reduce.hpp"
#include "dns/gpu/internal/stage_write.hpp"

#include <rocprim/functional.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dns::gpu {

using internal::reduce;

namespace {

struct ShockAxes
{
    int a[3];
};

// Row q on an axis of n cells lies in the band a stride-2 read would wrap from.
__device__ inline bool inBand(int q, int n) { return q < 2 || q >= n - 2; }

// Cell c's byte (and raw v): the stride-2 jump, max over the axes. A solid at
// +-1 or +-2 reads the centre on that side: ghosts hold antimirrored velocity,
// deep cells the ambient, a thin wall is not read across, and a blocked cell keeps
// reading the jump on its open side (a smeared shock is a linear ramp to a whole
// stride one cell up, v ~0; silencing the axis left slanted-wall faces unsensed,
// issues.md S8). nz == 1: o2 is 0, the z term is exactly 0, no branch.
template <bool kSolid, bool kRaw>
__global__ void kShockSense(GpuGrid g, const Real* p, const std::uint8_t* solid, ShockAxes open,
                            std::uint8_t* s, Real* raw)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    bool quiet = false;
    if constexpr (kSolid) quiet = solid[c] != 0;
    const int q[3] = {i, j, k}, n[3] = {g.nx, g.ny, g.nz};
    for (int a = 0; a < 3; ++a) quiet = quiet || (open.a[a] && inBand(q[a], n[a]));
    if (quiet) {
        s[c] = 0;
        if constexpr (kRaw) raw[c] = Real(0);
        return;
    }
    const Offsets o = offsetsAt(g, i, j, k), o2 = offsets2At(g, i, j, k);
    Real v = 0;
    for (int a = 0; a < 3; ++a) {
        std::ptrdiff_t up = o2.p[a], dn = o2.m[a];
        if constexpr (kSolid) {
            if (solid[c + o.p[a]] || solid[c + o2.p[a]]) up = 0;
            if (solid[c + o.m[a]] || solid[c + o2.m[a]]) dn = 0;
        }
        const Real jump = shockJump(p[c + dn], p[c], p[c + up]);
        v = jump > v || jump != jump ? jump : v; // a NaN sticks
    }
    s[c] = shockByte(v);
    if constexpr (kRaw) raw[c] = v;
}

// Test only: sigma 1 on every cell the sensor may write (the teeth of the flame
// gate). solid may be null.
__global__ void kShockForce(GpuGrid g, const std::uint8_t* solid, ShockAxes open, std::uint8_t* s)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const int q[3] = {i, j, k}, n[3] = {g.nx, g.ny, g.nz};
    bool quiet = solid && solid[c];
    for (int a = 0; a < 3; ++a) quiet = quiet || (open.a[a] && inBand(q[a], n[a]));
    s[c] = quiet ? 0 : 255;
}

// active[c] iff any byte on the +-2 axis cross of c: the union of the windows
// of its six faces, so kShockFaces reads exactly what it needs and one byte
// keeps a quiet cell out.
__global__ void kShockMask(GpuGrid g, const std::uint8_t* s, std::uint8_t* active)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k), o2 = offsets2At(g, i, j, k);
    unsigned any = s[c];
    for (int a = 0; a < 3; ++a)
        any |= unsigned(s[c + o.p[a]]) | s[c + o.m[a]] | s[c + o2.p[a]] | s[c + o2.m[a]];
    active[c] = any ? 1 : 0;
}

__device__ inline ShockState<Real> stateOf(PrimView q, Real gam, std::size_t c)
{
    return {q.rho[c], {q.u[c], q.v[c], q.w[c]}, q.p[c], q.e[c], gam};
}

// The correction of cell c: both faces of every axis, F_KG + sigma (F_HLL - F_KG)
// over the stage's own primitives, written through the stage combination.
// Species: D rides a donor stream, linear in Y, so a uniform Y stays exactly
// uniform, which needs rho's write and the species' to fuse alike: both go through
// stageAdd (its own contraction, not this pragma's). contract(off): hi of c and lo
// of c + 1 must be the same bits (the precedent is kLmfGatherS; nvcc's --fmad
// ignores it, conservation is its check).
template <bool Comp, bool kSolid, bool Sp>
__global__ void kShockFaces(GpuGrid g, PrimView q, StageCombine cmb, Real gam, const std::uint8_t* s,
                            const std::uint8_t* active, const std::uint8_t* solid, ShockAxes open, const Real* Y0,
                            const Real* Y1, const Real* Y2, const Real* Y3, int nY, Real* orho, Real* omx, Real* omy,
                            Real* omz, Real* oE, Real* oy0, Real* oy1, Real* oy2, Real* oy3)
{
#pragma clang fp contract(off)
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    if (!active[c]) return; // writes nothing: byte identity
    if constexpr (kSolid)
        if (solid[c]) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k), o2 = offsets2At(g, i, j, k);
    const int qc[3] = {i, j, k}, nn[3] = {g.nx, g.ny, g.nz};
    const Real* Y[kMaxY] = {Y0, Y1, Y2, Y3};
    Real d[5] = {Real(0), Real(0), Real(0), Real(0), Real(0)};
    Real dy[kMaxY] = {Real(0), Real(0), Real(0), Real(0)};
    for (int a = 0; a < 3; ++a) {
        if (nn[a] == 1) continue;
        const Real h = a == 0 ? g.hx : a == 1 ? g.hy : g.hz;
        const Real nrm[3] = {a == 0 ? Real(1) : Real(0), a == 1 ? Real(1) : Real(0), a == 2 ? Real(1) : Real(0)};
        for (int side = 0; side < 2; ++side) {
            const bool hi = side == 0;
            // Faces 0|1, n-2|n-1 and the wrap carry nothing on a non-periodic axis.
            if (open.a[a]) {
                const int w = qc[a], m = nn[a];
                if (hi ? (w == 0 || w >= m - 2) : (w <= 1 || w == m - 1)) continue;
            }
            const std::size_t L = hi ? c : c + o.m[a], R = hi ? c + o.p[a] : c;
            if constexpr (kSolid)
                if (solid[L] || solid[R]) continue; // kWallRhs owns that face
            const std::size_t LL = hi ? c + o.m[a] : c + o2.m[a], RR = hi ? c + o2.p[a] : c + o.p[a];
            const std::uint8_t a0 = s[LL], a1 = s[L], a2 = s[R], a3 = s[RR];
            const std::uint8_t b0 = a0 > a1 ? a0 : a1, b1 = a2 > a3 ? a2 : a3;
            const std::uint8_t by = b0 > b1 ? b0 : b1;
            if (!by) continue;
            const ShockDelta<Real> D =
                shockFaceDelta(Real(by) / Real(255), stateOf(q, gam, L), stateOf(q, gam, R), nrm);
            const Real sgn = hi ? Real(1) : Real(-1);
            for (int v = 0; v < 5; ++v) d[v] -= sgn * D.v[v] / h;
            if constexpr (Sp) {
                const Real dp = fmaxf(D.v[0], Real(0)), dm = fminf(D.v[0], Real(0));
                for (int m = 0; m < nY; ++m) dy[m] -= sgn * (dp * Y[m][L] + dm * Y[m][R]) / h;
            }
        }
    }
    const Real cr = cmb.crAt(c);
    Real* out[5] = {orho, omx, omy, omz, oE};
    for (int v = 0; v < 5; ++v) internal::stageAdd<Comp>(cmb, v, c, cr, d[v], out[v]);
    if constexpr (Sp) {
        Real* oy[kMaxY] = {oy0, oy1, oy2, oy3};
        // Species are never compensated: the plain branch reads only out.
        for (int m = 0; m < nY; ++m) internal::stageAdd<false>(cmb, 0, c, cr, dy[m], oy[m]);
    }
}

struct ShockCount
{
    const std::uint8_t* s;
    __host__ __device__ double operator()(int c) const { return s[c] ? 1.0 : 0.0; }
};
struct ShockTopByte
{
    const std::uint8_t* s;
    __host__ __device__ double operator()(int c) const { return double(s[c]) / 255.0; }
};
struct ShockTopRaw
{
    const Real* raw;
    __host__ __device__ double operator()(int c) const
    {
        const double v = raw[c];
        return isfinite(v) ? v : HUGE_VAL;
    }
};
// A field's min over fluid cells; a non-finite value reads as -inf, never dropped.
struct ShockFluidMin
{
    const Real* f;
    const std::uint8_t* solid;
    __host__ __device__ double operator()(int c) const
    {
        if (solid && solid[c]) return HUGE_VAL;
        const double v = f[c];
        return isfinite(v) ? v : -HUGE_VAL;
    }
};
struct ShockRegionCount
{
    const std::uint8_t *s, *region;
    std::uint8_t lo, hi;
    __host__ __device__ double operator()(int c) const
    {
        return s[c] && region[c] >= lo && region[c] <= hi ? 1.0 : 0.0;
    }
};
// Sensed cells of [lo, hi] upstream of column `col`.
struct ShockRegionFar
{
    GpuGrid g;
    const std::uint8_t *s, *region;
    std::uint8_t lo, hi;
    int col;
    __host__ __device__ double operator()(int c) const
    {
        if (!s[c] || region[c] < lo || region[c] > hi) return 0.0;
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        return i < col ? 1.0 : 0.0;
    }
};
struct ShockRegionTop
{
    const std::uint8_t *s;
    const Real* raw;
    const std::uint8_t* region;
    std::uint8_t lo, hi;
    __host__ __device__ double operator()(int c) const
    {
        if (region[c] < lo || region[c] > hi) return 0.0;
        if (!raw) return double(s[c]) / 255.0;
        const double v = raw[c];
        return isfinite(v) ? v : HUGE_VAL;
    }
};
struct ShockOmegaMax
{
    const Real* omega;
    __host__ __device__ double operator()(int c) const
    {
        const double w = omega[c];
        return isfinite(w) ? w : HUGE_VAL;
    }
};
// raw v over the cells burning above `thr`.
struct ShockBurnTop
{
    const Real *raw, *omega;
    double thr;
    __host__ __device__ double operator()(int c) const
    {
        if (!(double(omega[c]) > thr)) return 0.0;
        const double v = raw[c];
        return isfinite(v) ? v : HUGE_VAL;
    }
};
// One quantity of the cavity cells [lo, hi] of column `col`, 0 elsewhere: 0 local
// Mach, 1 one, 2 p, 3 rho u, 4 rho u p0 (isentropic).
struct ShockThroat
{
    GpuGrid g;
    const std::uint8_t* region;
    std::uint8_t lo, hi;
    int col, what;
    const Real *rho, *u, *v, *w, *p;
    double gamma;
    __host__ __device__ double operator()(int c) const
    {
        if (region[c] < lo || region[c] > hi) return 0.0;
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        if (i != col) return 0.0;
        if (what == 1) return 1.0;
        const double pp = p[c], r = rho[c], a = u[c], b = v[c], d = w[c];
        if (what == 2) return pp;
        if (what == 3) return r * a;
        const double m2 = (a * a + b * b + d * d) / (gamma * pp / r);
        if (what == 0) return sqrt(m2);
        return r * a * pp * pow(1 + 0.5 * (gamma - 1) * m2, gamma / (gamma - 1));
    }
};
// Sensed cells burning above `thr`.
struct ShockBurnCount
{
    const std::uint8_t* s;
    const Real* omega;
    double thr;
    __host__ __device__ double operator()(int c) const { return s[c] && double(omega[c]) > thr ? 1.0 : 0.0; }
};
// Local Mach on one region byte; 0 elsewhere.
struct ShockRegionMach
{
    const std::uint8_t* region;
    std::uint8_t which;
    const Real *rho, *u, *v, *w, *p;
    double gamma;
    __host__ __device__ double operator()(int c) const
    {
        if (region[c] != which) return 0.0;
        const double a = u[c], b = v[c], d = w[c];
        return sqrt((a * a + b * b + d * d) / (gamma * double(p[c]) / double(rho[c])));
    }
};
// |u_n|/c over the NSCBC end rows: 0 elsewhere, +inf on a non-finite row.
struct ShockOpenMach
{
    GpuGrid g;
    ShockAxes nscbc;
    const Real *rho, *u, *v, *w, *p;
    double gamma;
    __host__ __device__ double operator()(int c) const
    {
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        const int q[3] = {i, j, k}, n[3] = {g.nx, g.ny, g.nz};
        const Real* un[3] = {u, v, w};
        double m = 0;
        for (int a = 0; a < 3; ++a) {
            if (!nscbc.a[a] || (q[a] != 0 && q[a] != n[a] - 1)) continue;
            const double x = fabs(double(un[a][c])) / sqrt(gamma * double(p[c]) / double(rho[c]));
            if (!isfinite(x)) return HUGE_VAL;
            if (x > m) m = x;
        }
        return m;
    }
};

// |u_n|/c of one NSCBC row cell, t over the rows, axis by axis, lo then hi, from
// the conserved state (no launch, no primitive). The row the --shock hll guard
// reads: an x-hi cell flowing out is the Super row's, 0 here; +inf if not finite.
struct ShockRowGuard
{
    GpuGrid g;
    ShockAxes nscbc;
    const Real *rho, *mx, *my, *mz, *E;
    double gamma;
    __host__ __device__ double operator()(int t) const
    {
        const int n[3] = {g.nx, g.ny, g.nz};
        const Real* mom[3] = {mx, my, mz};
        for (int a = 0; a < 3; ++a) {
            if (!nscbc.a[a]) continue;
            const int b = a == 0 ? 1 : 0, d = a == 2 ? 1 : 2;
            const int face = n[b] * n[d];
            if (t >= 2 * face) {
                t -= 2 * face;
                continue;
            }
            const bool hi = t >= face;
            const int tt = hi ? t - face : t;
            int q[3];
            q[a] = hi ? n[a] - 1 : 0;
            q[b] = tt % n[b];
            q[d] = tt / n[b];
            const std::size_t c = gidx(g, q[0], q[1], q[2]);
            const double r = rho[c], un = double(mom[a][c]) / r;
            if (a == 0 && hi && un > 0) return 0.0;
            const double u = double(mx[c]) / r, v = double(my[c]) / r, w = double(mz[c]) / r;
            const double p = (gamma - 1) * (double(E[c]) - 0.5 * r * (u * u + v * v + w * w));
            const double x = fabs(un) / sqrt(gamma * p / r);
            return isfinite(x) ? x : HUGE_VAL;
        }
        return 0.0;
    }
};

ShockAxes axesOf(const bool a[3]) { return {{a[0] ? 1 : 0, a[1] ? 1 : 0, a[2] ? 1 : 0}}; }

} // namespace

std::expected<DeviceShock, hipError_t> DeviceShock::make(const GpuGrid& g, const ShockSetup& s)
{
    DeviceShock d;
    d.g_ = g;
    d.set_ = s;
    const std::size_t n = g.size();
    auto a = DeviceBuffer<std::uint8_t>::make(n);
    if (!a) return std::unexpected(a.error());
    auto b = DeviceBuffer<std::uint8_t>::make(n);
    if (!b) return std::unexpected(b.error());
    d.s_ = std::move(*a);
    d.active_ = std::move(*b);
    if (!s.apply) {
        auto r = DeviceScalar::make(n);
        if (!r) return std::unexpected(r.error());
        d.raw_ = std::move(*r);
    }
    return d;
}

void DeviceShock::sense(const Real* p)
{
    const std::size_t n = g_.size();
    const ShockAxes open = axesOf(set_.open);
    if (set_.testForce) {
        kShockForce<<<launchBlocks(n), blockSize>>>(g_, set_.solid, open, s_.data());
    } else if (set_.solid) {
        if (set_.apply)
            kShockSense<true, false><<<launchBlocks(n), blockSize>>>(g_, p, set_.solid, open, s_.data(), nullptr);
        else
            kShockSense<true, true><<<launchBlocks(n), blockSize>>>(g_, p, set_.solid, open, s_.data(), raw_.data());
    } else {
        if (set_.apply)
            kShockSense<false, false><<<launchBlocks(n), blockSize>>>(g_, p, nullptr, open, s_.data(), nullptr);
        else
            kShockSense<false, true><<<launchBlocks(n), blockSize>>>(g_, p, nullptr, open, s_.data(), raw_.data());
    }
    checkLastKernel("kShockSense");
    kShockMask<<<launchBlocks(n), blockSize>>>(g_, s_.data(), active_.data());
    checkLastKernel("kShockMask");
}

double DeviceShock::count(ReduceScratch& r) const
{
    return reduce(r, g_.size(), ShockCount{s_.data()}, rocprim::plus<double>(), 0.0);
}

void DeviceShock::faces(const PrimView& q, const StageCombine& cmb, Real gamma, const Real* const* Y, int nY,
                        DeviceCompState& out) const
{
    assert(set_.apply && (nY == 0 || !cmb.ol[0]));
    const std::size_t n = g_.size();
    const ShockAxes open = axesOf(set_.open);
    auto launch = [&](auto comp, auto solid, auto sp) {
        const Real* y[kMaxY] = {};
        for (int k = 0; k < nY; ++k) y[k] = Y[k];
        kShockFaces<decltype(comp)::value, decltype(solid)::value, decltype(sp)::value>
            <<<launchBlocks(n), blockSize>>>(g_, q, cmb, gamma, s_.data(), active_.data(), set_.solid, open, y[0], y[1],
                                             y[2], y[3], nY, out.rho.data(), out.mx.data(), out.my.data(), out.mz.data(),
                                             out.E.data(), out.y[0].data(), out.y[1].data(), out.y[2].data(),
                                             out.y[3].data());
    };
    using T = std::true_type;
    using F = std::false_type;
    if (nY > 0) {
        if (set_.solid) launch(F{}, T{}, T{});
        else launch(F{}, F{}, T{});
        checkLastKernel("kShockFaces<species>");
        return;
    }
    internal::withCompensation(cmb, [&](auto comp) {
        if (set_.solid) launch(comp, T{}, F{});
        else launch(comp, F{}, F{});
    });
    checkLastKernel("kShockFaces");
}

ShockRow DeviceShock::row(const Real* rho, const Real* u, const Real* v, const Real* w, const Real* p, Real gamma,
                          const ShockRegions& reg, ReduceScratch& r) const
{
    const std::size_t n = g_.size();
    ShockRow o;
    o.cells = count(r);
    o.top = set_.apply ? reduce(r, n, ShockTopByte{s_.data()}, rocprim::maximum<double>(), 0.0)
                       : reduce(r, n, ShockTopRaw{raw_.data()}, rocprim::maximum<double>(), 0.0);
    o.rhoMin = reduce(r, n, ShockFluidMin{rho, set_.solid}, rocprim::minimum<double>(), HUGE_VAL);
    o.pMin = reduce(r, n, ShockFluidMin{p, set_.solid}, rocprim::minimum<double>(), HUGE_VAL);
    const ShockAxes nscbc = axesOf(set_.nscbc);
    if (nscbc.a[0] || nscbc.a[1] || nscbc.a[2])
        o.openMach = reduce(r, n, ShockOpenMach{g_, nscbc, rho, u, v, w, p, double(gamma)},
                            rocprim::maximum<double>(), 0.0);
    if (!reg.region) return o;
    auto cells = [&](std::uint8_t lo, std::uint8_t hi) {
        return reduce(r, n, ShockRegionCount{s_.data(), reg.region, lo, hi}, rocprim::plus<double>(), 0.0);
    };
    o.cavCells = cells(reg.cavLo, reg.cavHi);
    o.lipCells = cells(reg.lipLo, reg.lipHi);
    o.cavRest = cells(reg.cavLo, std::uint8_t(reg.lipLo - 1));
    o.cavFar = reduce(r, n, ShockRegionFar{g_, s_.data(), reg.region, reg.cavLo, reg.cavHi, reg.farCol},
                      rocprim::plus<double>(), 0.0);
    o.jetMach = reduce(r, n, ShockRegionMach{reg.region, reg.jetByte, rho, u, v, w, p, double(gamma)},
                       rocprim::maximum<double>(), 0.0);
    if (reg.throatCol >= 0) {
        auto throat = [&](int what) {
            return ShockThroat{g_, reg.region, reg.cavLo, reg.cavHi, reg.throatCol, what, rho, u, v, w, p, double(gamma)};
        };
        const double count = reduce(r, n, throat(1), rocprim::plus<double>(), 0.0);
        const double flux = reduce(r, n, throat(3), rocprim::plus<double>(), 0.0);
        o.throatMach = reduce(r, n, throat(0), rocprim::maximum<double>(), 0.0);
        o.throatP = reduce(r, n, throat(2), rocprim::plus<double>(), 0.0) / count;
        o.throatP0 = reduce(r, n, throat(4), rocprim::plus<double>(), 0.0) / flux;
    }
    o.exitMach = reduce(r, n, ShockRegionMach{reg.region, reg.lipLo, rho, u, v, w, p, double(gamma)},
                        rocprim::maximum<double>(), 0.0);
    o.cavTop = reduce(r, n,
                      ShockRegionTop{s_.data(), set_.apply ? nullptr : raw_.data(), reg.region, reg.cavLo, reg.cavHi},
                      rocprim::maximum<double>(), 0.0);
    return o;
}

double DeviceShock::rowGuard(const Real* rho, const Real* mx, const Real* my, const Real* mz, const Real* E,
                             Real gamma, ReduceScratch& r) const
{
    const int n[3] = {g_.nx, g_.ny, g_.nz};
    std::size_t rows = 0;
    for (int a = 0; a < 3; ++a)
        if (set_.nscbc[a]) rows += 2 * g_.size() / std::size_t(n[a]);
    if (rows == 0) return 0.0;
    return reduce(r, rows, ShockRowGuard{g_, axesOf(set_.nscbc), rho, mx, my, mz, E, double(gamma)},
                  rocprim::maximum<double>(), 0.0);
}

hipError_t DeviceShock::bytes(std::vector<std::uint8_t>& out) const
{
    out.resize(g_.size());
    return s_.download(out.data());
}

double DeviceShock::burnMax(const Real* omega, double& omegaPeak, ReduceScratch& r) const
{
    assert(!set_.apply);
    const std::size_t n = g_.size();
    omegaPeak = std::max(omegaPeak, reduce(r, n, ShockOmegaMax{omega}, rocprim::maximum<double>(), 0.0));
    return reduce(r, n, ShockBurnTop{raw_.data(), omega, 0.01 * omegaPeak}, rocprim::maximum<double>(), 0.0);
}

double DeviceShock::burnCells(const Real* omega, double& omegaPeak, ReduceScratch& r) const
{
    const std::size_t n = g_.size();
    omegaPeak = std::max(omegaPeak, reduce(r, n, ShockOmegaMax{omega}, rocprim::maximum<double>(), 0.0));
    return reduce(r, n, ShockBurnCount{s_.data(), omega, 0.01 * omegaPeak}, rocprim::plus<double>(), 0.0);
}

std::expected<std::vector<ShockPeak>, hipError_t> DeviceShock::peaks(std::size_t k) const
{
    assert(!set_.apply);
    std::vector<Real> v(g_.size());
    if (const hipError_t e = raw_.download(v.data()); e != hipSuccess) return std::unexpected(e);
    std::vector<ShockPeak> all;
    for (std::size_t c = 0; c < v.size(); ++c)
        if (v[c] > Real(0)) all.push_back({double(v[c]), c});
    const auto larger = [](const ShockPeak& a, const ShockPeak& b) {
        return a.v != b.v ? a.v > b.v : a.cell < b.cell;
    };
    k = std::min(k, all.size());
    std::ranges::partial_sort(all, all.begin() + std::ptrdiff_t(k), larger);
    all.resize(k);
    return all;
}

} // namespace dns::gpu
