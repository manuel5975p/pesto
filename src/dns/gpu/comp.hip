#include "dns/gpu/comp.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/stage_write.hpp"
#include "dns/gpu/rans.hpp"
#include "dns/remap.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include "dns/gpu/internal/reduce.hpp"

#include <rocprim/iterator/transform_iterator.hpp>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <utility>

namespace dns::gpu {

using internal::reduce;
namespace {

__global__ void kPrim(GpuGrid g, Real gamma, Real Rgas, const Real* rho,
                      const Real* mx, const Real* my, const Real* mz,
                      const Real* E, Real* u, Real* v, Real* w, Real* e, Real* p,
                      Real* T)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    primitivesAt(c, gamma, Rgas, rho, mx, my, mz, E, u, v, w, e, p, T);
}

// Dynamic eddy viscosity mu_t = rho (C Delta)^2 op(grad u) at cell centres
// from central-difference velocity gradients; op is |S| = sqrt(2 S_ij S_ij)
// (sigma == 0, Smagorinsky) or D_sigma (sigma != 0, dns/sgs.hpp).
// smagCd = C^2 Delta^2. z-shear terms cancel by periodic wrap when nz == 1.
__global__ void kCompSmagNut(GpuGrid g, Real smagCd, int sigma, const Real* rho,
                             const Real* u, const Real* v, const Real* w,
                             Real* nut)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Real inv[3] = {Real(0.5) / g.hx, Real(0.5) / g.hy, Real(0.5) / g.hz};
    const Real* vc[3] = {u, v, w};
    Real gd[3][3]; // gd[c][d] = d vc[c] / d x_d
    for (int cc = 0; cc < 3; ++cc)
        for (int d = 0; d < 3; ++d)
            gd[cc][d] = (vc[cc][c + o.p[d]] - vc[cc][c + o.m[d]]) * inv[d];
    nut[c] = rho[c] * smagCd * sgsRate(sigma, gd);
}

// Frame velocity at both faces of cell (i,j,k) along `a` (mz.active only):
// the hi face is the cell's own +face, the lo face the back neighbour's, with
// the same wrap contract as offsetsAt so the two sides of any face agree.
__device__ inline void mrfCellFaces(const MrfView& mz, const GpuGrid& g, int i,
                                    int j, int k, int a, Real whi[3],
                                    Real wlo[3])
{
    mrfFaceVel(mz, i, j, k, a, whi);
    int il = i, jl = j, kl = k;
    if (a == 0) il = i ? i - 1 : g.nx - 1;
    else if (a == 1) jl = j ? j - 1 : g.ny - 1;
    else kl = k ? k - 1 : g.nz - 1;
    mrfFaceVel(mz, il, jl, kl, a, wlo);
}

// Coriolis source of the rotating-frame zone, added to the momentum rhs of
// cell c: -f omega x m in absolute-velocity form ((axis, b, c2) cyclic:
// S_b = +f omega m_c2, S_c2 = -f omega m_b). Energy carries no source -- the
// term is perpendicular to u, so its work is identically zero. Zero on any
// swirl-free field, which is what keeps a uniform axial freestream exact.
__device__ inline void mrfSource(const MrfView& mz, PrimView q, std::size_t c,
                                 int i, int j, int k, Real& dmx, Real& dmy,
                                 Real& dmz)
{
    Real wc[3];
    // Cell-centre position: the per-cell table on a general mesh
    // (setMrfCurv), the separable per-axis tables otherwise.
    const Real px = mz.pc[0] ? mz.pc[0][c] : mz.cc[0][i];
    const Real py = mz.pc[1] ? mz.pc[1][c] : mz.cc[1][j];
    const Real pz = mz.pc[2] ? mz.pc[2][c] : mz.cc[2][k];
    const Real f = mrfFrameVel(mz, px, py, pz, wc);
    if (f == Real(0)) return;
    const Real fo = f * mz.omega;
    const int b = (mz.axis + 1) % 3, c2 = (mz.axis + 2) % 3;
    const Real mom[3] = {q.rho[c] * q.u[c], q.rho[c] * q.v[c],
                         q.rho[c] * q.w[c]};
    Real* dm[3] = {&dmx, &dmy, &dmz};
    *dm[b] += fo * mom[c2];
    *dm[c2] -= fo * mom[b];
}

// out = cmb.ca a + cmb.cb b + cmb.crAt(c) * (-div F): the rhs and the RK
// combination in one pass (see StageCombine in comp.hpp). Comp: the
// compensated write (internal/stage_write.hpp).
template <bool Comp>
__global__ void kCompStage(GpuGrid g, ViscView vv, PrimView q,
                           StageCombine cmb, Real* orho, Real* omx, Real* omy,
                           Real* omz, Real* oE, const Real* nut, Real kapScale,
                           MrfView mz)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    Real drho = 0, dmx = 0, dmy = 0, dmz = 0, dE = 0;
    for (int a = 0; a < 3; ++a) {
        const Real h = a == 0 ? g.hx : a == 1 ? g.hy : g.hz;
        Real whi[3], wlo[3];
        const Real *pwhi = nullptr, *pwlo = nullptr;
        if (mz.active) {
            mrfCellFaces(mz, g, i, j, k, a, whi, wlo);
            pwhi = whi;
            pwlo = wlo;
        }
        const Flux hi =
            faceFlux(g, vv, q, o, c, c + o.p[a], a, nut, kapScale, pwhi);
        const Flux lo =
            faceFlux(g, vv, q, o, c + o.m[a], c, a, nut, kapScale, pwlo);
        drho -= (hi.rho - lo.rho) / h;
        dmx -= (hi.mx - lo.mx) / h;
        dmy -= (hi.my - lo.my) / h;
        dmz -= (hi.mz - lo.mz) / h;
        dE -= (hi.E - lo.E) / h;
    }
    if (mz.active) mrfSource(mz, q, c, i, j, k, dmx, dmy, dmz);
    const Real rhs[5] = {drho, dmx, dmy, dmz, dE};
    Real* out[5] = {orho, omx, omy, omz, oE};
    const Real cr = cmb.crAt(c);
    for (int v = 0; v < 5; ++v)
        internal::stagePut<Comp>(cmb, v, c, cr, rhs[v], out[v]);
}

// Curvilinear twin of kCompSmagNut. The velocity gradient comes from the mesh
// metrics (curvVelGradCentre) instead of the uniform spacing, and the filter
// width is the cell's OWN Delta (curvDelta2) -- on a clustered mesh Delta
// varies by the focal ratio across the box, which is the whole point: the fine
// plateau filters at its own scale, not the box average. cs2 = C^2 alone; the
// uniform path folds a constant Delta^2 into smagCd instead.
__global__ void kCompSmagNutCurv(GpuGrid g, Real cs2, int sigma, CurvView m,
                                 const Real* rho, const Real* u, const Real* v,
                                 const Real* w, Real* nut)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Real* vc[3] = {u, v, w};
    Real gd[3][3];
    curvVelGradCentre(m, o, c,
                      [&](std::size_t cell, int comp) { return vc[comp][cell]; },
                      gd);
    const Real cd = cs2 * curvDelta2(g, m.invVol[c]);
    nut[c] = rho[c] * cd * sgsRate(sigma, gd);
}

// cells: the run of cells written (DeviceComp::setStageRange).
template <bool Comp>
__global__ void kCompStageCurv(GpuGrid g, ViscView vv, PrimView q,
                               CurvView m, StageCombine cmb, Real* orho,
                               Real* omx, Real* omy, Real* omz, Real* oE,
                               const Real* nut, Real kapScale, MrfView mz,
                               CellRange cells)
{
    const std::size_t c =
        cells.first + std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= cells.end) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    Real drho = 0, dmx = 0, dmy = 0, dmz = 0, dE = 0;
    for (int a = 0; a < 3; ++a) {
        Real whi[3], wlo[3];
        const Real *pwhi = nullptr, *pwlo = nullptr;
        Real qhi = 0, qlo = 0;
        if (mz.active) {
            if (mz.psi) {
                // Node-psi transport correction (general mesh): the lo face is
                // the back neighbour's +a face, same wrap as offsetsAt.
                qhi = mrfFaceQNode(mz, g, i, j, a);
                const int il = a == 0 ? (i ? i - 1 : g.nx - 1) : i;
                const int jl = a == 1 ? (j ? j - 1 : g.ny - 1) : j;
                qlo = mrfFaceQNode(mz, g, il, jl, a);
            } else {
                mrfCellFaces(mz, g, i, j, k, a, whi, wlo);
                pwhi = whi;
                pwlo = wlo;
            }
        }
        const Flux hi = faceFluxCurv(vv, q, m, o, c, c + o.p[a], a, nut,
                                     kapScale, pwhi, qhi);
        const Flux lo = faceFluxCurv(vv, q, m, o, c + o.m[a], c, a, nut,
                                     kapScale, pwlo, qlo);
        drho -= hi.rho - lo.rho;
        dmx -= hi.mx - lo.mx;
        dmy -= hi.my - lo.my;
        dmz -= hi.mz - lo.mz;
        dE -= hi.E - lo.E;
    }
    const Real iv = m.invVol[c];
    Real rhs[5] = {drho * iv, dmx * iv, dmy * iv, dmz * iv, dE * iv};
    // The source is a volume term: added AFTER the 1/vol of the flux
    // divergence, in the same per-volume units.
    if (mz.active) mrfSource(mz, q, c, i, j, k, rhs[1], rhs[2], rhs[3]);
    Real* out[5] = {orho, omx, omy, omz, oE};
    const Real cr = cmb.crAt(c);
    for (int v = 0; v < 5; ++v)
        internal::stagePut<Comp>(cmb, v, c, cr, rhs[v], out[v]);
}

// The FP64 primitive recovery every stableDt reduction starts from: velocity
// and pressure of one cell, straight off the conserved state. Written once so
// the acoustic, thermal and density bounds cannot disagree about it -- three
// copies of these four lines is exactly how a gamma or a kinetic-energy term
// drifts between them. Pre: rho[c] > 0 (stableDt reports a diverged field
// rather than asserting).
struct CellPrim
{
    double u, v, w, p, rho;
};

__host__ __device__ inline CellPrim cellPrim(const Real* rho, const Real* mx,
                                             const Real* my, const Real* mz,
                                             const Real* E, double gamma, int c)
{
    const double r = rho[c];
    const double u = mx[c] / r, v = my[c] / r, w = mz[c] / r;
    const double p = (gamma - 1.0) *
        (double(E[c]) - 0.5 * (u * u + v * v + w * w) * r);
    return {u, v, w, p, r};
}

// Frame velocity at cell c's centre (device path of the dt reductions); the
// inactive view returns without touching w. In the rotating-frame zone the
// convective eigenvalues are u_rel +- c, so the acoustic bound must see the
// RELATIVE velocity -- the absolute one under-counts by the frame speed
// (~ tip Mach) exactly where the mesh is finest.
__host__ __device__ inline void mrfCellVel(const MrfView& mz, const GpuGrid& g,
                                           int c, double w[3])
{
    if (!mz.active) return;
    int i, j, k;
    decompose(g, std::size_t(c), i, j, k);
    Real wr[3];
    const Real px = mz.pc[0] ? mz.pc[0][c] : mz.cc[0][i];
    const Real py = mz.pc[1] ? mz.pc[1][c] : mz.cc[1][j];
    const Real pz = mz.pc[2] ? mz.pc[2][c] : mz.cc[2][k];
    (void)mrfFrameVel(mz, px, py, pz, wr);
    w[0] -= double(wr[0]);
    w[1] -= double(wr[1]);
    w[2] -= double(wr[2]);
}

// Curvilinear acoustic bound: (|u| + c) / width, width = 1/aov per axis;
// |u| >= |u . n| makes the bound conservative on skewed cells.
struct WavePerDtCurv
{
    GpuGrid g;
    MrfView mrf;
    const Real *rho, *mx, *my, *mz, *E;
    const Real *ax, *ay, *az; // az == nullptr when nz == 1
    double gamma;
    __host__ __device__ double operator()(int c) const
    {
        const auto q = cellPrim(rho, mx, my, mz, E, gamma, c);
        double ur[3] = {q.u, q.v, q.w};
        mrfCellVel(mrf, g, c, ur);
        const double sp = sqrt(ur[0] * ur[0] + ur[1] * ur[1] + ur[2] * ur[2])
                        + sqrt(gamma * q.p / q.rho);
        // A non-finite cell is +inf to the max-reduction, never dropped: a
        // comparison against NaN is false, and the bound would read finite.
        if (!isfinite(sp)) return HUGE_VAL;
        double m = sp * double(ax[c] > ay[c] ? ax[c] : ay[c]);
        if (az && sp * az[c] > m) m = sp * az[c];
        return m;
    }
};

struct WavePerDt
{
    GpuGrid g;
    MrfView mrf;
    const Real *rho, *mx, *my, *mz, *E;
    double gamma, ihx, ihy, ihz; // degenerate axes pass 0
    __host__ __device__ double operator()(int c) const
    {
        const auto q = cellPrim(rho, mx, my, mz, E, gamma, c);
        double ur[3] = {q.u, q.v, q.w};
        mrfCellVel(mrf, g, c, ur);
        const double cs = sqrt(gamma * q.p / q.rho);
        double m = (fabs(ur[0]) + cs) * ihx;
        const double wy = (fabs(ur[1]) + cs) * ihy;
        const double wz = (fabs(ur[2]) + cs) * ihz;
        // Every term is tested, not the max: `wy > m` is false for a NaN wy.
        if (!isfinite(m + wy + wz)) return HUGE_VAL;
        if (wy > m) m = wy;
        if (wz > m) m = wz;
        return m;
    }
};

struct MinRho
{
    const Real* rho;
    __host__ __device__ double operator()(int c) const { return double(rho[c]); }
};

// T = p / (rho R) per cell. Sutherland's mu(T) is monotone increasing, so ONE
// max-T reduction bounds mu over the whole field -- no per-cell viscosity
// reduction is needed for the diffusive dt limit.
struct MaxT
{
    const Real *rho, *mx, *my, *mz, *E;
    double gamma, Rgas;
    __host__ __device__ double operator()(int c) const
    {
        const auto q = cellPrim(rho, mx, my, mz, E, gamma, c);
        const double T = q.p / (q.rho * Rgas);
        return isfinite(T) ? T : HUGE_VAL;   // never dropped by the max
    }
};

// The solver's own per-cell SGS mu_t (sgsNut), widened for the max-reduction
// of the LES viscous dt limit. It used to be an FP64 functor re-deriving the
// whole model per cell (sigma's singular values included): 1.5 / 2.5 ms of a
// 5.8 / 7.1 ms step on a 128^3 sphere (W15), for a bound that differs from
// this one only by the FP32 rounding of mu_t.
struct MaxNut
{
    const Real* nut;
    __host__ __device__ double operator()(int c) const
    {
        const double v = nut[c];
        return isfinite(v) ? v : HUGE_VAL;   // never dropped by the max
    }
};

// Galilean x-boost by du: (rho, m, E) -> (rho, mx + rho du, E + mx du +
// rho du^2 / 2). Exact frame change -- creates no divergence, no vorticity,
// no acoustic wave.
__global__ void kBoostX(std::size_t n, Real du, const Real* rho, Real* mx,
                        Real* E)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    E[c] += mx[c] * du + Real(0.5) * rho[c] * du * du;
    mx[c] += rho[c] * du;
}

// kBoostX on the compensated pair (--compensated): the same increments off
// the hi words, accumulated into (hi, lo) so a ramp step below ulp(q)/2 is
// kept. A separate kernel, so the plain one stays the pre-compensation code.
__global__ void kBoostXPair(std::size_t n, Real du, const Real* rho, Real* mx,
                            Real* mxl, Real* E, Real* El)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const Word2 e = addWord2(E[c], El[c], mx[c] * du + Real(0.5) * rho[c] * du * du);
    const Word2 m = addWord2(mx[c], mxl[c], rho[c] * du);
    E[c] = e.hi;
    El[c] = e.lo;
    mx[c] = m.hi;
    mxl[c] = m.lo;
}

} // namespace

void boostX(const GpuGrid& g, DeviceCompState& s, double du)
{
    if (du == 0.0) return;
    if (s.compensated())
    {
        kBoostXPair<<<launchBlocks(g.size()), blockSize>>>(
            g.size(), Real(du), s.rho.data(), s.mx.data(), s.lo[1].data(),
            s.E.data(), s.lo[4].data());
        checkLastKernel("kBoostXPair");
        return;
    }
    kBoostX<<<launchBlocks(g.size()), blockSize>>>(g.size(), Real(du),
                                                   s.rho.data(), s.mx.data(),
                                                   s.E.data());
    checkLastKernel("kBoostX");
}

std::expected<DeviceCurv, hipError_t> DeviceCurv::make(const CurvMetrics& m)
{
    const std::size_t n = m.vol.size();
    auto up = [](DeviceScalar& dst, const std::vector<double>& src) -> hipError_t {
        auto b = makeNarrow(src);
        if (!b) return b.error();
        dst = std::move(*b);
        return hipSuccess;
    };
    DeviceCurv d;
    d.hmin = m.hmin;
    for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c)
            if (auto e = up(d.S[a][c], m.S[a][c]); e != hipSuccess)
                return std::unexpected(e);
        for (int q = 0; q < 9; ++q)
            if (auto e = up(d.Ji[a][q], m.Ji[a][q]); e != hipSuccess)
                return std::unexpected(e);
        if (auto e = up(d.aov[a], m.aov[a]); e != hipSuccess)
            return std::unexpected(e);
    }
    if (!m.aovSteadyZ.empty())
        if (auto e = up(d.aovSteadyZ, m.aovSteadyZ); e != hipSuccess)
            return std::unexpected(e);
    std::vector<double> iv(n);
    for (std::size_t i = 0; i < n; ++i) iv[i] = 1.0 / m.vol[i];
    if (auto e = up(d.invVol, iv); e != hipSuccess) return std::unexpected(e);
    return d;
}

namespace {

hipError_t zero(DeviceScalar& b)
{
    return b.size() ? hipMemset(b.data(), 0, b.size() * sizeof(Real))
                    : hipSuccess;
}

// h = hi (+ lo when compensated): both floats widen exactly, so the FP64 sum
// is the pair's value up to one double rounding.
hipError_t widenPair(const DeviceScalar& hi, const DeviceScalar& lo,
                     std::vector<double>& h)
{
    if (auto e = downloadWiden(hi, h); e != hipSuccess) return e;
    if (lo.size() == 0) return hipSuccess;
    std::vector<double> l(h.size());
    if (auto e = downloadWiden(lo, l); e != hipSuccess) return e;
    for (std::size_t i = 0; i < h.size(); ++i) h[i] += l[i];
    return hipSuccess;
}

} // namespace

std::expected<DeviceCompState, hipError_t>
DeviceCompState::make(std::size_t n, int nT, bool compensated)
{
    assert(nT >= 0 && nT <= 2);
    DeviceCompState s;
    auto alloc = [n](DeviceScalar& b, bool zeroed) -> hipError_t {
        auto e = DeviceScalar::make(n);
        if (!e) return e.error();
        b = std::move(*e);
        return zeroed ? zero(b) : hipSuccess;
    };
    for (int v = 0; v < 5; ++v) {
        DeviceScalar* hi[5] = {&s.rho, &s.mx, &s.my, &s.mz, &s.E};
        if (auto e = alloc(*hi[v], false); e != hipSuccess)
            return std::unexpected(e);
        if (compensated)
            if (auto e = alloc(s.lo[v], true); e != hipSuccess)
                return std::unexpected(e);
    }
    for (int i = 0; i < nT; ++i) {
        if (auto e = alloc(s.t[i], false); e != hipSuccess)
            return std::unexpected(e);
        if (compensated)
            if (auto e = alloc(s.tlo[i], true); e != hipSuccess)
                return std::unexpected(e);
    }
    s.nT = nT;
    return s;
}

hipError_t DeviceCompState::uploadTurb(int i, const std::vector<double>& h)
{
    assert(i >= 0 && i < nT && h.size() == t[i].size());
    if (auto e = zero(tlo[i]); e != hipSuccess) return e;
    return uploadNarrow(h, t[i]);
}

hipError_t DeviceCompState::downloadTurb(int i, std::vector<double>& h) const
{
    assert(i >= 0 && i < nT);
    h.resize(t[i].size());
    return widenPair(t[i], tlo[i], h);
}

hipError_t DeviceCompState::upload(const CompState& h)
{
    for (auto& l : lo)
        if (auto e = zero(l); e != hipSuccess) return e;
    if (auto e = uploadNarrow(h.rho, rho); e != hipSuccess) return e;
    if (auto e = uploadNarrow(h.mx, mx); e != hipSuccess) return e;
    if (auto e = uploadNarrow(h.my, my); e != hipSuccess) return e;
    if (auto e = uploadNarrow(h.mz, mz); e != hipSuccess) return e;
    return uploadNarrow(h.E, E);
}

hipError_t DeviceCompState::download(CompState& h) const
{
    if (auto e = widenPair(rho, lo[0], h.rho); e != hipSuccess) return e;
    if (auto e = widenPair(mx, lo[1], h.mx); e != hipSuccess) return e;
    if (auto e = widenPair(my, lo[2], h.my); e != hipSuccess) return e;
    if (auto e = widenPair(mz, lo[3], h.mz); e != hipSuccess) return e;
    return widenPair(E, lo[4], h.E);
}

std::expected<DeviceComp, hipError_t> DeviceComp::make(const Grid& g,
                                                       const Transport& tr,
                                                       double csLes, SgsModel sgs)
{
    assert(tr.gamma > 1.0 && tr.muRef >= 0.0 && tr.Pr > 0.0 && tr.Tref > 0.0 &&
           csLes >= 0.0);
    DeviceComp d(GpuGrid::from(g), tr, Real(csLes), sgs);
    const std::size_t n = g.size();
    if (auto e = d.allocStages(0); !e) return std::unexpected(e.error());
    for (auto* b : {&d.u_, &d.v_, &d.w_, &d.e_, &d.p_, &d.T_}) {
        auto e = DeviceScalar::make(n);
        if (!e) return std::unexpected(e.error());
        *b = std::move(*e);
    }
    if (csLes > 0.0) {
        auto e = DeviceScalar::make(n);
        if (!e) return std::unexpected(e.error());
        d.nut_ = std::move(*e);
    }
    return d;
}

std::expected<DeviceComp, hipError_t> DeviceComp::make(const Grid& g,
                                                       double gamma, double mu,
                                                       double csLes, SgsModel sgs)
{
    return make(g, Transport::constant(gamma, mu), csLes,
                sgs);
}

std::expected<DeviceComp, hipError_t> DeviceComp::make(const CurvMetrics& m,
                                                       double gamma, double mu,
                                                       double csLes,
                                                       SgsModel sgs)
{
    return make(m, Transport::constant(gamma, mu), csLes,
                sgs);
}

std::expected<DeviceComp, hipError_t> DeviceComp::make(const CurvMetrics& m,
                                                       const Transport& tr,
                                                       double csLes,
                                                       SgsModel sgs)
{
    auto d = make(m.g, tr, csLes, sgs);
    if (!d) return d;
    auto cv = DeviceCurv::make(m);
    if (!cv) return std::unexpected(cv.error());
    d->curv_ = std::move(*cv);
    return d;
}

CurvView DeviceCurv::view() const
{
    CurvView cv;
    for (int a = 0; a < 3; ++a) {
        for (int c = 0; c < 3; ++c) cv.S[a][c] = S[a][c].data();
        for (int e = 0; e < 9; ++e) cv.Ji[a][e] = Ji[a][e].data();
    }
    cv.invVol = invVol.data();
    return cv;
}

CurvView DeviceComp::curvView() const { return curv_->view(); }

hipError_t DeviceComp::setMrf(const MrfZone& z, const AxisRemap3& remap)
{
    assert(z.omega != 0.0 && z.axis >= 0 && z.axis < 3 &&
           0.0 < z.r1 && z.r1 < z.r2 && 0.0 < z.a1 && z.a1 < z.a2);
    // Physical coordinate tables per axis: cell centres at the remap of
    // (i + 0.5) h, +faces at the remap of (i + 1) h. One table each -- the
    // remap is separable, so face positions never depend on the other axes.
    const int n[3] = {g_.nx, g_.ny, g_.nz};
    const double h[3] = {g_.hx, g_.hy, g_.hz};
    const Focal1D* f[3] = {&remap.fx, &remap.fy, &remap.fz};
    for (int a = 0; a < 3; ++a) {
        std::vector<Real> cc(static_cast<std::size_t>(n[a]));
        std::vector<Real> fc(static_cast<std::size_t>(n[a]));
        for (int i = 0; i < n[a]; ++i) {
            cc[std::size_t(i)] = Real(f[a]->map((i + 0.5) * h[a]));
            fc[std::size_t(i)] = Real(f[a]->map((i + 1.0) * h[a]));
        }
        for (auto [buf, src] : {std::pair{&mrfCc_[a], &cc},
                                std::pair{&mrfFc_[a], &fc}}) {
            auto b = DeviceScalar::make(std::size_t(n[a]));
            if (!b) return b.error();
            if (auto e = b->upload(src->data()); e != hipSuccess) return e;
            *buf = std::move(*b);
        }
    }
    mrf_ = z;
    return hipSuccess;
}

namespace {

// Host double twin of mrfPsiRad's closed form (same cubic-smoothstep
// integral, kept in lockstep with it): G(r) = int_0^r s fr(s) ds. The table
// is built once in double so the only FP32 rounding is the stored psi itself;
// the kernel's telescoping then cancels those stored values exactly.
double mrfPsiRadHost(double r1, double r2, double r)
{
    if (r <= r1) return 0.5 * r * r;
    const double d = r2 - r1;
    const double u = r >= r2 ? 1.0 : (r - r1) / d;
    const double u2 = u * u, u3 = u2 * u, u4 = u2 * u2;
    return 0.5 * r1 * r1 +
           d * (r1 * (u - u3 + 0.5 * u4) +
                d * (0.5 * u2 - 0.75 * u4 + 0.4 * u4 * u));
}

} // namespace

hipError_t DeviceComp::setMrfCurv(const MrfZone& z, const CurvMesh& mesh)
{
    // z shaft only, and the axial blend must be degenerate (fa == 1
    // everywhere): the node-psi table carries no axial factor, because a
    // z-extruded mesh has no axial boundary for the blend to shield. The
    // runner sets a1/a2 to the same 1e30 sentinel the 2D Cartesian zone uses.
    assert(z.omega != 0.0 && z.axis == 2 && 0.0 < z.r1 && z.r1 < z.r2 &&
           z.a1 >= 1e29 && z.a2 > z.a1);
    assert(mesh.g.nx == g_.nx && mesh.g.ny == g_.ny && mesh.g.nz == g_.nz);

    // Per-node stream potential psi = -omega * hz * G(r), (ny+1) rows of nx:
    // row ny is the +j face's top corner row of the last cell row, well
    // defined for every AxisWrap by CurvMesh::node. hz is baked in (the z
    // edges of every in-plane face have exactly that length on a z-extruded
    // mesh; 1 at nz == 1, the unit depth).
    const std::size_t nn = std::size_t(g_.ny + 1) * g_.nx;
    std::vector<Real> psi(nn);
    for (int j = 0; j <= g_.ny; ++j)
        for (int i = 0; i < g_.nx; ++i) {
            const auto p = mesh.node(i, j, 0);
            const double r = std::hypot(p[0] - z.center[0],
                                        p[1] - z.center[1]);
            psi[std::size_t(j) * g_.nx + i] =
                Real(-z.omega * g_.hz * mrfPsiRadHost(z.r1, z.r2, r));
        }

    // Per-cell centres (mean of the 8 corner nodes) for the Coriolis source
    // and the dt bound -- pointwise consumers, no telescoping at stake.
    std::vector<Real> cx(g_.size()), cy(g_.size()), cz(g_.size());
    for (int k = 0; k < g_.nz; ++k)
        for (int j = 0; j < g_.ny; ++j)
            for (int i = 0; i < g_.nx; ++i) {
                double s[3] = {0, 0, 0};
                for (int dk = 0; dk < 2; ++dk)
                    for (int dj = 0; dj < 2; ++dj)
                        for (int di = 0; di < 2; ++di) {
                            const auto p =
                                mesh.node(i + di, j + dj, k + dk);
                            s[0] += p[0];
                            s[1] += p[1];
                            s[2] += p[2];
                        }
                const std::size_t c = gidx(g_, i, j, k);
                cx[c] = Real(0.125 * s[0]);
                cy[c] = Real(0.125 * s[1]);
                cz[c] = Real(0.125 * s[2]);
            }

    auto up = [](DeviceScalar* dst, const std::vector<Real>& src)
        -> hipError_t {
        auto b = DeviceScalar::make(src.size());
        if (!b) return b.error();
        if (auto e = b->upload(src.data()); e != hipSuccess) return e;
        *dst = std::move(*b);
        return hipSuccess;
    };
    if (auto e = up(&mrfPsi_, psi); e != hipSuccess) return e;
    if (auto e = up(&mrfPc_[0], cx); e != hipSuccess) return e;
    if (auto e = up(&mrfPc_[1], cy); e != hipSuccess) return e;
    if (auto e = up(&mrfPc_[2], cz); e != hipSuccess) return e;
    mrf_ = z;
    return hipSuccess;
}

MrfView DeviceComp::mrfView() const
{
    const MrfZone& z = *mrf_;
    MrfView v;
    v.active = 1;
    v.axis = z.axis;
    v.omega = Real(z.omega);
    v.cx = Real(z.center[0]);
    v.cy = Real(z.center[1]);
    v.cz = Real(z.center[2]);
    v.r1 = Real(z.r1);
    v.r2 = Real(z.r2);
    v.a1 = Real(z.a1);
    v.a2 = Real(z.a2);
    for (int a = 0; a < 3; ++a) {
        v.cc[a] = mrfCc_[a].data();
        v.fc[a] = mrfFc_[a].data();
        v.pc[a] = mrfPc_[a].data();
    }
    // Exactly one family is engaged: setMrf leaves psi/pc null, setMrfCurv
    // leaves cc/fc null, and psi is the kernel-side switch.
    v.psi = mrfPsi_.data();
    return v;
}

DeviceComp::DeviceComp(const GpuGrid& g, const Transport& tr, Real csLes,
                       SgsModel sgs)
    : g_(g), tr_(tr), csLes_(csLes), sgs_(sgs), stageRange_{0, g.size()}
{
    // Delta^2 = (hx hy hz)^(2/3) in 3D; hx hy when nz == 1 (unit z-depth).
    const double d2 = g.nz > 1
        ? std::pow(double(g.hx) * g.hy * g.hz, 2.0 / 3.0)
        : double(g.hx) * g.hy;
    smagCd_ = Real(csLes * csLes * d2);
}

void DeviceComp::primitives(const DeviceCompState& s)
{
    const std::size_t n = g_.size();
    kPrim<<<launchBlocks(n), blockSize>>>(
        g_, Real(tr_.gamma), Real(tr_.Rgas), s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
        s.E.data(), u_.data(), v_.data(), w_.data(), e_.data(), p_.data(),
        T_.data());
    checkLastKernel("kPrim");
}

void DeviceComp::primitivesAndMut(const DeviceCompState& s)
{
    assert(rans());
    const bool sst = ransSst(ransModel_);
    if (sst) { // F1 and mu_t read the primitives' stencil: two launches
        primitives(s);
        const PrimView q{s.rho.data(), u_.data(), v_.data(), w_.data(),
                         e_.data(), p_.data(), T_.data()};
        const rans::RansCellCache cache{
            {ransPhi_[0].data(), ransPhi_[1].data()}, ransRate_.data(),
            ransGkw_.data()};
        rans::sstMutCurv(g_, viscView(tr_), q, curvView(), ransModel_,
                         s.t[0].data(), s.t[1].data(), wd_.data(), nut_.data(),
                         f1_.data(), cache);
    } else {
        rans::primMutCurv(g_, viscView(tr_), Real(tr_.gamma), Real(tr_.Rgas),
                          s.rho.data(), s.mx.data(), s.my.data(),
                          s.mz.data(), s.E.data(), s.t[0].data(), u_.data(),
                          v_.data(), w_.data(), e_.data(), p_.data(),
                          T_.data(), nut_.data(), ransPhi_[0].data());
    }
    cacheOf_ = s.t[0].data();
    if (mutHook_) mutHook_(nut_.data(), sst ? f1_.data() : nullptr);
}

MutHook mutMirrorYHook(const GpuGrid& g, bool lo, bool hi)
{
    return [g, lo, hi](Real* nut, Real* f1) {
        rans::mutMirrorY(g, lo, hi, nut, f1);
    };
}

std::vector<std::uint8_t> rowMask(const Grid& g, int jLo, int jHi)
{
    assert(0 <= jLo && jLo <= jHi && jHi < g.ny);
    std::vector<std::uint8_t> m(g.size(), 0);
    for (int k = 0; k < g.nz; ++k)
        for (int j = jLo; j <= jHi; ++j)
            for (int i = 0; i < g.nx; ++i) m[g.idx(i, j, k)] = 1;
    return m;
}

unsigned long long DeviceComp::floorHits() const
{
    if (hits_.size() == 0) return 0;
    unsigned long long h = 0;
    if (hits_.download(&h) != hipSuccess) return ~0ull; // reads as a failure
    return h;
}

void DeviceComp::updateMut(const DeviceCompState& s)
{
    primitivesAndMut(s);
}

void DeviceComp::sgsNut(const DeviceCompState& s)
{
    assert(les());
    const std::size_t n = g_.size();
    const int sig = sgs_ == SgsModel::Sigma ? 1 : 0;
    if (curv_) {
        kCompSmagNutCurv<<<launchBlocks(n), blockSize>>>(
            g_, csLes_ * csLes_, sig, curvView(), s.rho.data(), u_.data(),
            v_.data(), w_.data(), nut_.data());
        checkLastKernel("kCompSmagNutCurv");
    } else {
        kCompSmagNut<<<launchBlocks(n), blockSize>>>(
            g_, smagCd_, sig, s.rho.data(), u_.data(), v_.data(), w_.data(),
            nut_.data());
        checkLastKernel("kCompSmagNut");
    }
    // The wall image (LES has no F1), exactly where RANS takes it.
    if (mutHook_) mutHook_(nut_.data(), nullptr);
}

void DeviceComp::updateSgsNut(const DeviceCompState& s)
{
    primitives(s);
    sgsNut(s);
}

void DeviceComp::setStageRange(CellRange r)
{
    assert(curvilinear() && r.first < r.end && r.end <= g_.size());
    stageRange_ = r;
}

void DeviceComp::stage(const DeviceCompState& s, const StageCombine& cmb,
                       DeviceCompState& out, bool residualMode)
{
    // RANS: mu_t lagged within the stage exactly like LES -- from the stage
    // input, through the SAME nut slot of the mean-flow flux.
    if (rans()) primitivesAndMut(s);
    else primitives(s);
    const ViscView vv = viscView(tr_);
    const PrimView q{s.rho.data(), u_.data(), v_.data(), w_.data(), e_.data(),
                     p_.data(), T_.data()};
    const std::size_t n = g_.size();
    const Real kapLes = Real(tr_.cp() / prt_); // eddy conductivity scale
    const Real* nutp = nullptr;
    Real kapScale = Real(0);
    const MrfView mz = mrf_ ? mrfView() : MrfView{};
    assert(!cmb.ol[0] || cmb.ca == Real(1) - cmb.cb); // the increment form
    if (curv_) {
        const CurvView cv = curvView();
        const CellRange cells = stageRange_;
        auto curvStage = [&] {
            internal::withCompensation(cmb, [&](auto comp) {
                kCompStageCurv<decltype(comp)::value>
                    <<<launchBlocks(cells.size()), blockSize>>>(
                        g_, vv, q, cv, cmb, out.rho.data(), out.mx.data(),
                        out.my.data(), out.mz.data(), out.E.data(), nutp,
                        kapScale, mz, cells);
            });
            checkLastKernel("kCompStageCurv");
        };
        if (rans()) {
            nutp = nut_.data();
            kapScale = kapLes;
            curvStage();
            // The scalar stage reads primitivesAndMut's cache: of s, never
            // stale (out aliases neither s nor cmb.a, so curvStage left s
            // untouched).
            assert(cacheOf_ == s.t[0].data());
            if (ransSst(ransModel_)) {
                const rans::SstStageArgs a{
                    ransModel_,     s.t[0].data(),     s.t[1].data(),
                    nut_.data(),    f1_.data(),        wd_.data(),
                    out.rho.data(), Real(omegaFloor_), mask_.data(),
                    hits_.data(),   testSrc_[0],       testSrc_[1],
                    {{ransPhi_[0].data(), ransPhi_[1].data()},
                     ransRate_.data(), ransGkw_.data()}};
                rans::sstStageCurv(g_, vv, q, cv, mz, cmb, a, residualMode,
                                   out.t[0].data(), out.t[1].data(), cells,
                                   bnd_);
            } else {
                rans::saStageCurv(g_, vv, q, cv, mz, cmb, s.t[0].data(),
                                  ransPhi_[0].data(), wd_.data(), residualMode,
                                  ransTransport_, testSrc_[0], out.t[0].data(),
                                  cells, bnd_);
            }
            if (rhsHook_) rhsHook_(s, out, cmb);
            if (bnd_.any())
                rans::openScalarConsistency(g_, bnd_, q, cv, cmb, residualMode,
                                            s, out);
            return;
        }
        if (les()) {
            sgsNut(s);
            nutp = nut_.data();
            kapScale = kapLes;
        }
        curvStage();
        if (rhsHook_) rhsHook_(s, out, cmb);
        return;
    }
    if (les()) {
        sgsNut(s);
        nutp = nut_.data();
        kapScale = kapLes;
    }
    internal::withCompensation(cmb, [&](auto comp) {
        kCompStage<decltype(comp)::value><<<launchBlocks(n), blockSize>>>(
            g_, vv, q, cmb, out.rho.data(), out.mx.data(), out.my.data(),
            out.mz.data(), out.E.data(), nutp, kapScale, mz);
    });
    checkLastKernel("kCompStage");
    if (rhsHook_) rhsHook_(s, out, cmb);
}

namespace {

__global__ void kPenalizeMomentum(std::size_t n, const Real* lam, Real idte,
                                  Real* mx, Real* my, Real* mz)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const Real l = lam[c];
    if (l >= Real(1)) return; // interior untouched exactly
    const Real f = exp(-idte * (Real(1) - l));
    mx[c] *= f;
    my[c] *= f;
    mz[c] *= f;
}

} // namespace

void penalizeMomentum(const GpuGrid& g, const DeviceScalar& lam, Real eta,
                      Real dt, DeviceCompState& s)
{
    assert(!s.compensated()); // a plain-state edit (see boostX)
    kPenalizeMomentum<<<launchBlocks(g.size()), blockSize>>>(
        g.size(), lam.data(), dt / eta, s.mx.data(), s.my.data(), s.mz.data());
    checkLastKernel("penalizeMomentum");
}

void DeviceComp::step(DeviceCompState& s, Real dt)
{
    // Shu-Osher SSP-RK3. Each stage writes its combination straight out of the
    // rhs kernel, so two stage buffers suffice: q1_ is free again after stage
    // 2 consumed it, stage 3 writes there, and the result is swapped into s.
    //
    // The output NEVER aliases cmb.a, and must not: the boundary rhs hook runs
    // as a SECOND kernel over the same cells and re-reads a, which an aliased
    // output would already have overwritten. (It did, and every NSCBC check
    // failed.) The swap is a buffer-handle exchange, not a copy -- no caller
    // may hold a raw device pointer into the state across a step.
    assert(s.nT == q1_.nT); // the scalars ride the same buffers and swap
    assert(s.compensated() == compensated_); // ...and so do the low words
    const Real* lts = ltsOn_ ? lts_.data() : nullptr;
    // Every (ca, cb) below has ca == 1 - cb EXACTLY in FP32, which is what the
    // compensated increment form (StageCombine::put) relies on.
    auto cmb = [lts](const DeviceCompState& a, Real ca,
                     const DeviceCompState& b, Real cb, Real cr,
                     DeviceCompState& out) {
        StageCombine k{{a.rho.data(), a.mx.data(), a.my.data(), a.mz.data(),
                        a.E.data()},
                       {b.rho.data(), b.mx.data(), b.my.data(), b.mz.data(),
                        b.E.data()},
                       ca, cb, cr};
        for (int i = 0; i < a.nT; ++i) {
            k.ta[i] = a.t[i].data();
            k.tb[i] = b.t[i].data();
        }
        k.lts = lts;
        if (!out.compensated()) return k;
        for (int v = 0; v < 5; ++v) {
            k.al[v] = a.lo[v].data();
            k.bl[v] = b.lo[v].data();
            k.ol[v] = out.lo[v].data();
        }
        for (int i = 0; i < a.nT; ++i) {
            k.tal[i] = a.tlo[i].data();
            k.tbl[i] = b.tlo[i].data();
            k.otl[i] = out.tlo[i].data();
        }
        return k;
    };
    // Idempotent::yes: s is already the hook's output (setStageHook).
    if (hook_ && hookAtEntry_) hook_(s);
    stage(s, cmb(s, Real(1), s, Real(0), dt, q1_), q1_);
    if (hook_) hook_(q1_);
    stage(q1_, cmb(s, Real(0.75), q1_, Real(0.25), Real(0.25) * dt, q2_), q2_);
    if (hook_) hook_(q2_);
    // ca = 1 - cb, not Real(1/3): the two rounded thirds sum to 1 + 3e-8 in
    // FP32, which inflated every conserved total by ~2e-8 per step on any
    // non-uniform field (measured, gpu_mrf_conservation) and floored a steady
    // RANS residual at 2^-25 q / dt. 1 - Real(2/3) is exact (Sterbenz), so
    // ca + cb == 1 exactly; cr = cb dt is unchanged.
    const Real cb3 = Real(2.0 / 3.0), ca3 = Real(1) - cb3;
    stage(q2_, cmb(s, ca3, q2_, cb3, cb3 * dt, q1_), q1_);
    std::swap(s, q1_);
}

double DeviceComp::stableDt(const DeviceCompState& s, ReduceScratch& r,
                            double cfl)
{
    const MrfView mz = mrf_ ? mrfView() : MrfView{};
    const double wmax = curv_
        ? reduce(r, g_.size(),
                 WavePerDtCurv{g_, mz, s.rho.data(), s.mx.data(), s.my.data(),
                               s.mz.data(), s.E.data(), curv_->aov[0].data(),
                               curv_->aov[1].data(),
                               g_.nz > 1 ? curv_->aov[2].data() : nullptr,
                               double(tr_.gamma)},
                 rocprim::maximum<double>(), 0.0)
        : reduce(r, g_.size(),
                 WavePerDt{g_, mz, s.rho.data(), s.mx.data(), s.my.data(),
                           s.mz.data(), s.E.data(), double(tr_.gamma),
                           1.0 / g_.hx, 1.0 / g_.hy,
                           g_.nz > 1 ? 1.0 / g_.hz : 0.0},
                 rocprim::maximum<double>(), 0.0);
    double dt = cfl / wmax;
    // RANS: the diffusive bound is PER CELL (mu_t varies by orders of
    // magnitude across a boundary layer, and the global-extreme form below
    // would pair the largest mu_t with the smallest cell). RANS-only, so the
    // DNS/LES bound below stays byte for byte what every anchor ran with.
    if (rans()) {
        const double dtv = reduce(r, g_.size(), ransDiffDt(s),
                                  rocprim::minimum<double>(), 1e300);
        return dtv < dt ? dtv : dt;
    }
    // Sutherland: mu is largest where T is largest, and the law is monotone,
    // so the field maximum comes from one extra reduction rather than a
    // per-cell viscosity pass. Constant viscosity skips the reduction entirely.
    double muEff = tr_.muRef;
    if (tr_.sutherland && tr_.muRef > 0.0)
        muEff = tr_.mu(reduce(r, g_.size(),
                              MaxT{s.rho.data(), s.mx.data(), s.my.data(),
                                   s.mz.data(), s.E.data(), double(tr_.gamma),
                                   double(tr_.Rgas)},
                              rocprim::maximum<double>(), 0.0));
    if (les()) {
        // The mu_t the fluxes of a stage on s would read, image rows
        // included (the mutHook images are odd or even: never above the max).
        updateSgsNut(s);
        muEff += reduce(r, g_.size(), MaxNut{nut_.data()},
                        rocprim::maximum<double>(), 0.0);
    }
    if (muEff > 0.0) {
        const double rhomin = reduce(r, g_.size(), MinRho{s.rho.data()},
                                     rocprim::minimum<double>(), 1e300);
        double h;
        if (curv_) {
            h = curv_->hmin;
        } else {
            h = std::min<double>(g_.hx, g_.hy);
            if (g_.nz > 1) h = std::min<double>(h, g_.hz);
        }
        // The binding diffusivity is the THERMAL one: kappa/(rho cv) =
        // gamma nu / Pr, 1.94x the momentum nu for air. Bounding only nu -- as
        // this did before the energy equation carried a real conductivity --
        // overestimates the stable dt by that factor whenever the diffusive
        // limit binds. The SGS part rides the same (conservative) factor.
        const double dtv =
            0.25 * h * h * rhomin / (muEff * tr_.diffusivityFactor());
        if (dtv < dt) dt = dtv;
    }
    return dt;
}

// --- RANS ---------------------------------------------------------------------

namespace {

// dtc[c] = min(cfl / wave_c, diffusive_c): the local dt of --steady. Diff is
// rans::RansDiffDt or rans::LaminarDiffDt.
template <class Diff>
__global__ void kLocalDt(WavePerDtCurv wave, Diff diff, double cfl,
                         std::size_t n, double* dtc)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const double w = wave(int(c));
    const double a = w > 0.0 ? cfl / w : HUGE_VAL;
    const double v = diff(int(c));
    dtc[c] = a < v ? a : v;
}

__global__ void kLtsRatio(std::size_t n, const double* dtc, double dtRef,
                          Real* lts)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    lts[c] = Real(dtc[c] / dtRef);
}

struct DtAt
{
    const double* dtc;
    __host__ __device__ double operator()(int c) const { return dtc[c]; }
};

// Squared per-cell residual of one equation over the counted cells -- the
// fluid mask when set, else rows [jLo, jHi] -- and 0 elsewhere: 0 continuity,
// 1 |momentum|/rho, 2 R0/rho (energy, a scalar), 4 R0/rho[] with `rho` the
// conserved scalar itself (SST omega: the rate of ln omega).
struct ResidSq
{
    GpuGrid g;
    const std::uint8_t* mask;
    int jLo, jHi, which;
    const Real *rho, *R0, *R1, *R2, *R3; // R1..R3: momentum when which == 1
    __host__ __device__ double operator()(int c) const
    {
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        if (mask ? !mask[c] : (j < jLo || j > jHi)) return 0.0;
        if (which == 1) {
            const double r = rho[c];
            const double a = R1[c] / r, b = R2[c] / r, d = R3[c] / r;
            return a * a + b * b + d * d;
        }
        const double v = which == 0 ? double(R0[c]) : double(R0[c]) / rho[c];
        return v * v;
    }
};

} // namespace

rans::RansDiffDt DeviceComp::ransDiffDt(const DeviceCompState& s) const
{
    return rans::RansDiffDt{s.rho.data(), s.mx.data(), s.my.data(),
                            s.mz.data(), s.E.data(), s.t[0].data(),
                            ransSst(ransModel_) ? s.t[1].data() : nullptr,
                            curv_->aov[0].data(), curv_->aov[1].data(),
                            g_.nz > 1 ? curv_->aov[2].data() : nullptr, tr_,
                            double(prt_)};
}

std::expected<void, hipError_t>
DeviceComp::allocSteady(const std::vector<std::uint8_t>& mask)
{
    assert(mask.empty() || mask.size() == g_.size());
    const std::size_t n = g_.size();
    auto l = DeviceScalar::make(n);
    auto t = DeviceBuffer<double>::make(n);
    if (!l) return std::unexpected(l.error());
    if (!t) return std::unexpected(t.error());
    lts_ = std::move(*l);
    dtc_ = std::move(*t);
    mask_ = {};
    maskCount_ = double(n);
    if (mask.empty()) return {};
    auto m = DeviceBuffer<std::uint8_t>::make(n);
    if (!m) return std::unexpected(m.error());
    if (auto e = m->upload(mask.data()); e != hipSuccess)
        return std::unexpected(e);
    mask_ = std::move(*m);
    maskCount_ = double(std::ranges::count(mask, std::uint8_t(1)));
    return {};
}

std::expected<void, hipError_t>
DeviceComp::enableSteady(const std::vector<std::uint8_t>& fluidMask)
{
    assert(curvilinear());
    return allocSteady(fluidMask);
}

std::expected<void, hipError_t> DeviceComp::enableRans(const RansSetup& rs)
{
    assert(curvilinear() && !les() && rs.model != RansModel::None &&
           rs.wallDist.size() == g_.size() && rs.omegaFloor > 0.0);
    assert(std::ranges::all_of(rs.wallDist, [](double d) { return d > 0.0; }));
    const std::size_t n = g_.size();
    if (auto e = allocStages(ransScalars(rs.model)); !e) return e;
    auto nb = DeviceScalar::make(n);
    if (!nb) return std::unexpected(nb.error());
    nut_ = std::move(*nb);
    auto d = makeNarrow(rs.wallDist);
    if (!d) return std::unexpected(d.error());
    wd_ = std::move(*d);
    if (auto e = allocSteady(rs.fluidMask); !e) return e;
    // The scalar stage's cell cache (rans::RansCellCache): phi0 always, SST
    // also phi1, the rate and grad k . grad omega.
    DeviceScalar* cache[4] = {&ransPhi_[0], &ransPhi_[1], &ransRate_,
                              &ransGkw_};
    for (int b = 0; b < (ransSst(rs.model) ? 4 : 1); ++b) {
        auto e = DeviceScalar::make(n);
        if (!e) return std::unexpected(e.error());
        *cache[b] = std::move(*e);
    }
    if (ransSst(rs.model)) {
        auto f = DeviceScalar::make(n);
        auto h = DeviceBuffer<unsigned long long>::make(1);
        if (!f) return std::unexpected(f.error());
        if (!h) return std::unexpected(h.error());
        const unsigned long long zero = 0;
        if (auto e = h->upload(&zero); e != hipSuccess)
            return std::unexpected(e);
        f1_ = std::move(*f);
        hits_ = std::move(*h);
    }
    omegaFloor_ = rs.omegaFloor;
    ransModel_ = rs.model;
    setMutHook(rs.mutHook);
    testSrc_[0] = rs.testSource[0];
    testSrc_[1] = rs.testSource[1];
    ransTransport_ = rs.transportOnly;
    bnd_ = rs.bnd;
    assert(rs.solidMask.empty() || (rs.bnd.any() && rs.solidMask.size() == n));
    if (!rs.solidMask.empty()) {
        auto sm = DeviceBuffer<std::uint8_t>::make(n);
        if (!sm) return std::unexpected(sm.error());
        if (auto e = sm->upload(rs.solidMask.data()); e != hipSuccess)
            return std::unexpected(e);
        solid_ = std::move(*sm);
        bnd_.solid = solid_.data();
    }
    return {};
}

double DeviceComp::localDt(const DeviceCompState& s, ReduceScratch& r,
                           double cfl)
{
    assert(curvilinear() && dtc_.size() == g_.size());
    const std::size_t n = g_.size();
    const MrfView mz = mrf_ ? mrfView() : MrfView{};
    // The wedge's steady local dt reads the side faces' real spacing
    // (DeviceCurv::aovSteadyZ); every other mesh its time-march aov.
    const Real* az = curv_->aovSteadyZ.size() ? curv_->aovSteadyZ.data()
                     : g_.nz > 1              ? curv_->aov[2].data()
                                              : nullptr;
    const WavePerDtCurv wave{g_, mz, s.rho.data(), s.mx.data(), s.my.data(),
                             s.mz.data(), s.E.data(), curv_->aov[0].data(),
                             curv_->aov[1].data(), az, double(tr_.gamma)};
    if (rans()) {
        rans::RansDiffDt diff = ransDiffDt(s);
        diff.az = az;
        kLocalDt<<<launchBlocks(n), blockSize>>>(wave, diff, cfl, n,
                                                 dtc_.data());
    } else {
        const rans::LaminarDiffDt diff{
            s.rho.data(),         s.mx.data(),          s.my.data(),
            s.mz.data(),          s.E.data(),           curv_->aov[0].data(),
            curv_->aov[1].data(), az,                   tr_};
        kLocalDt<<<launchBlocks(n), blockSize>>>(wave, diff, cfl, n,
                                                 dtc_.data());
    }
    checkLastKernel("kLocalDt");
    const double dtRef = reduce(r, n, DtAt{dtc_.data()},
                                rocprim::minimum<double>(), 1e300);
    // A collapsed or non-finite reference is the caller's divergence report;
    // the ratios are only meaningful against a positive one.
    if (!(dtRef > 0.0) || !std::isfinite(dtRef)) return dtRef;
    kLtsRatio<<<launchBlocks(n), blockSize>>>(n, dtc_.data(), dtRef,
                                              lts_.data());
    checkLastKernel("kLtsRatio");
    return dtRef;
}

void DeviceComp::setLts(bool on)
{
    assert(!on || lts_.size() == g_.size());
    ltsOn_ = on;
}

std::expected<void, hipError_t> DeviceComp::allocStages(int nT)
{
    for (auto* q : {&q1_, &q2_}) {
        auto e = DeviceCompState::make(g_.size(), nT, compensated_);
        if (!e) return std::unexpected(e.error());
        *q = std::move(*e);
    }
    return {};
}

std::expected<void, hipError_t> DeviceComp::enableCompensation()
{
    compensated_ = true;
    return allocStages(nTurb());
}

hipError_t DeviceComp::uploadLts(const std::vector<Real>& ratios)
{
    if (ratios.size() != g_.size()) return hipErrorInvalidValue;
    if (lts_.size() != g_.size()) {
        auto e = DeviceScalar::make(g_.size());
        if (!e) return e.error();
        lts_ = std::move(*e);
    }
    return lts_.upload(ratios.data());
}

void DeviceComp::residualInto(const DeviceCompState& s, DeviceCompState& out)
{
    assert(s.nT == nTurb() && out.nT == s.nT && &out != &s);
    // ca = cb = 0, cr = 1, no lts: the mean-flow kernel and every hook then
    // write the pure rhs (0 a + 0 b + 1 R == R exactly); the scalar kernel
    // needs its explicit mode, because the point-implicit form would return
    // (R + J q)/(1 + J) instead. No low words are named, so the plain write.
    StageCombine k{{s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
                    s.E.data()},
                   {s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
                    s.E.data()},
                   Real(0), Real(0), Real(1)};
    for (int i = 0; i < s.nT; ++i) k.ta[i] = k.tb[i] = s.t[i].data();
    stage(s, k, out, true);
}

namespace {

// FP64 RMS norms of the pure rhs R (a stage buffer) of state s over the
// counted cells (mask, else rows [jLo, jHi]); cnt is their number.
ResidualNorms residualNorms(const GpuGrid& g, const DeviceCompState& s,
                            const DeviceCompState& R, const std::uint8_t* mask,
                            int jLo, int jHi, double cnt, ReduceScratch& r)
{
    const std::size_t n = g.size();
    auto rms = [&](int which, const Real* R0) {
        const ResidSq f{g, mask, jLo, jHi, which, s.rho.data(), R0,
                        R.mx.data(), R.my.data(), R.mz.data()};
        return std::sqrt(
            reduce(r, n, f, rocprim::plus<double>(), 0.0) / cnt);
    };
    ResidualNorms out;
    out.rho = rms(0, R.rho.data());
    out.mom = rms(1, nullptr);
    out.E = rms(2, R.E.data());
    if (s.nT > 0) out.t[0] = rms(2, R.t[0].data());
    if (s.nT > 1) { // SST omega: R_w / (rho omega)
        const ResidSq f{g, mask, jLo, jHi, 4, s.t[1].data(), R.t[1].data(),
                        nullptr, nullptr, nullptr};
        out.t[1] = std::sqrt(reduce(r, n, f, rocprim::plus<double>(), 0.0) / cnt);
    }
    return out;
}

} // namespace

ResidualNorms DeviceComp::residual(const DeviceCompState& s, ReduceScratch& r)
{
    assert(dtc_.size() == g_.size()); // enableRans or enableSteady ran
    residualInto(s, q2_);
    return residualNorms(g_, s, q2_, mask_.data(), 0, g_.ny - 1, maskCount_, r);
}

ResidualNorms DeviceComp::residual(const DeviceCompState& s, int jLo, int jHi,
                                   ReduceScratch& r)
{
    assert(0 <= jLo && jLo <= jHi && jHi < g_.ny);
    residualInto(s, q2_);
    const double cnt = double(jHi - jLo + 1) * g_.nx * g_.nz;
    return residualNorms(g_, s, q2_, nullptr, jLo, jHi, cnt, r);
}

} // namespace dns::gpu
