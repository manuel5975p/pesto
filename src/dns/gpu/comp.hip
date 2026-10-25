#include "dns/gpu/comp.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/device_util.hpp"
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
    const Real r = rho[c];
    u[c] = mx[c] / r;
    v[c] = my[c] / r;
    w[c] = mz[c] / r;
    e[c] = E[c] / r;
    const Real pr = (gamma - Real(1)) *
        (E[c] - Real(0.5) * (mx[c] * mx[c] + my[c] * my[c] + mz[c] * mz[c]) / r);
    p[c] = pr;
    T[c] = pr / (r * Rgas);
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
// combination in one pass (see StageCombine in comp.hpp).
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
        out[v][c] = cmb.ca * cmb.a[v][c] + cmb.cb * cmb.b[v][c] + cr * rhs[v];
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

__global__ void kCompStageCurv(GpuGrid g, ViscView vv, PrimView q,
                               CurvView m, StageCombine cmb, Real* orho,
                               Real* omx, Real* omy, Real* omz, Real* oE,
                               const Real* nut, Real kapScale, MrfView mz)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
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
        out[v][c] = cmb.ca * cmb.a[v][c] + cmb.cb * cmb.b[v][c] + cr * rhs[v];
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

// Dynamic eddy viscosity mu_t = rho (C Delta)^2 op(grad u) at cell c (op as in
// kCompSmagNut), computed straight from the conserved state (no primitive
// scratch) so it can feed the const stableDt viscous limit. smagCd = C^2 Delta^2.
struct SmagMax
{
    GpuGrid g;
    double smagCd;
    int sigma;
    const Real *rho, *mx, *my, *mz;
    __host__ __device__ double vel(int c, int comp) const
    {
        const Real m = comp == 0 ? mx[c] : comp == 1 ? my[c] : mz[c];
        return double(m) / double(rho[c]);
    }
    __host__ __device__ double operator()(int c) const
    {
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        const Offsets o = offsetsAt(g, i, j, k);
        const double inv[3] = {0.5 / g.hx, 0.5 / g.hy, 0.5 / g.hz};
        double gd[3][3];
        for (int cc = 0; cc < 3; ++cc)
            for (int d = 0; d < 3; ++d)
                gd[cc][d] = (vel(int(c + o.p[d]), cc) -
                             vel(int(c + o.m[d]), cc)) * inv[d];
        const double nut = double(rho[c]) * smagCd * sgsRate(sigma, gd);
        return isfinite(nut) ? nut : HUGE_VAL;   // never dropped by the max
    }
};

// Curvilinear twin of SmagMax: same operator, mesh-metric gradients and the
// per-cell filter width (kCompSmagNutCurv). cs2 = C^2; Delta^2 is per cell.
struct SmagMaxCurv
{
    GpuGrid g;
    double cs2;
    int sigma;
    CurvView m;
    const Real *rho, *mx, *my, *mz;
    __host__ __device__ double vel(std::size_t c, int comp) const
    {
        const Real q = comp == 0 ? mx[c] : comp == 1 ? my[c] : mz[c];
        return double(q) / double(rho[c]);
    }
    __host__ __device__ double operator()(int c) const
    {
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        const Offsets o = offsetsAt(g, i, j, k);
        double gd[3][3];
        curvVelGradCentre(m, o, std::size_t(c),
                          [&](std::size_t cell, int comp) {
                              return vel(cell, comp);
                          },
                          gd);
        const double cd = cs2 * double(curvDelta2(g, m.invVol[c]));
        const double nut = double(rho[c]) * cd * sgsRate(sigma, gd);
        return isfinite(nut) ? nut : HUGE_VAL;   // never dropped by the max
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

} // namespace

void boostX(const GpuGrid& g, DeviceCompState& s, double du)
{
    if (du == 0.0) return;
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
    std::vector<double> iv(n);
    for (std::size_t i = 0; i < n; ++i) iv[i] = 1.0 / m.vol[i];
    if (auto e = up(d.invVol, iv); e != hipSuccess) return std::unexpected(e);
    return d;
}

std::expected<DeviceCompState, hipError_t> DeviceCompState::make(std::size_t n,
                                                                 int nT)
{
    assert(nT >= 0 && nT <= 2);
    DeviceCompState s;
    for (auto* b : {&s.rho, &s.mx, &s.my, &s.mz, &s.E}) {
        auto e = DeviceScalar::make(n);
        if (!e) return std::unexpected(e.error());
        *b = std::move(*e);
    }
    for (int i = 0; i < nT; ++i) {
        auto e = DeviceScalar::make(n);
        if (!e) return std::unexpected(e.error());
        s.t[i] = std::move(*e);
    }
    s.nT = nT;
    return s;
}

hipError_t DeviceCompState::uploadTurb(int i, const std::vector<double>& h)
{
    assert(i >= 0 && i < nT && h.size() == t[i].size());
    return uploadNarrow(h, t[i]);
}

hipError_t DeviceCompState::downloadTurb(int i, std::vector<double>& h) const
{
    assert(i >= 0 && i < nT);
    h.resize(t[i].size());
    return downloadWiden(t[i], h);
}

hipError_t DeviceCompState::upload(const CompState& h)
{
    if (auto e = uploadNarrow(h.rho, rho); e != hipSuccess) return e;
    if (auto e = uploadNarrow(h.mx, mx); e != hipSuccess) return e;
    if (auto e = uploadNarrow(h.my, my); e != hipSuccess) return e;
    if (auto e = uploadNarrow(h.mz, mz); e != hipSuccess) return e;
    return uploadNarrow(h.E, E);
}

hipError_t DeviceCompState::download(CompState& h) const
{
    if (auto e = downloadWiden(rho, h.rho); e != hipSuccess) return e;
    if (auto e = downloadWiden(mx, h.mx); e != hipSuccess) return e;
    if (auto e = downloadWiden(my, h.my); e != hipSuccess) return e;
    if (auto e = downloadWiden(mz, h.mz); e != hipSuccess) return e;
    return downloadWiden(E, h.E);
}

std::expected<DeviceComp, hipError_t> DeviceComp::make(const Grid& g,
                                                       const Transport& tr,
                                                       double csLes, SgsModel sgs)
{
    assert(tr.gamma > 1.0 && tr.muRef >= 0.0 && tr.Pr > 0.0 && tr.Tref > 0.0 &&
           csLes >= 0.0);
    DeviceComp d(GpuGrid::from(g), tr, Real(csLes), sgs);
    const std::size_t n = g.size();
    for (auto* s : {&d.q1_, &d.q2_}) {
        auto e = DeviceCompState::make(n);
        if (!e) return std::unexpected(e.error());
        *s = std::move(*e);
    }
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
    : g_(g), tr_(tr), csLes_(csLes), sgs_(sgs)
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

void DeviceComp::ransMut(const DeviceCompState& s)
{
    assert(rans());
    const bool sst = ransSst(ransModel_);
    if (sst) {
        const PrimView q{s.rho.data(), u_.data(), v_.data(), w_.data(),
                         e_.data(), p_.data(), T_.data()};
        rans::sstMutCurv(g_, viscView(tr_), q, curvView(), ransModel_,
                         s.t[0].data(), s.t[1].data(), wd_.data(), nut_.data(),
                         f1_.data());
    } else {
        rans::mutCurv(g_, viscView(tr_), s.rho.data(), T_.data(),
                      s.t[0].data(), nut_.data());
    }
    rans::mutMirrorY(g_, ransMirrorLo_, ransMirrorHi_, nut_.data(),
                     sst ? f1_.data() : nullptr);
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
    primitives(s);
    ransMut(s);
}

void DeviceComp::stage(const DeviceCompState& s, const StageCombine& cmb,
                       DeviceCompState& out, bool residualMode)
{
    primitives(s);
    const ViscView vv = viscView(tr_);
    const PrimView q{s.rho.data(), u_.data(), v_.data(), w_.data(), e_.data(),
                     p_.data(), T_.data()};
    const std::size_t n = g_.size();
    const Real kapLes = Real(tr_.cp() / prt_); // eddy conductivity scale
    const Real* nutp = nullptr;
    Real kapScale = Real(0);
    const MrfView mz = mrf_ ? mrfView() : MrfView{};
    if (curv_) {
        const CurvView cv = curvView();
        if (rans()) {
            // mu_t lagged within the stage exactly like LES: from the stage
            // input, through the SAME nut slot of the mean-flow flux.
            ransMut(s);
            nutp = nut_.data();
            kapScale = kapLes;
            kCompStageCurv<<<launchBlocks(n), blockSize>>>(
                g_, vv, q, cv, cmb, out.rho.data(), out.mx.data(),
                out.my.data(), out.mz.data(), out.E.data(), nutp, kapScale, mz);
            checkLastKernel("kCompStageCurv");
            if (ransSst(ransModel_)) {
                const rans::SstStageArgs a{
                    ransModel_,         s.t[0].data(), s.t[1].data(),
                    nut_.data(),        f1_.data(),    wd_.data(),
                    out.rho.data(),     Real(omegaFloor_),
                    int(ransMirrorLo_), int(ransMirrorHi_), hits_.data()};
                rans::sstStageCurv(g_, vv, q, cv, mz, cmb, a, residualMode,
                                   out.t[0].data(), out.t[1].data());
            } else {
                rans::saStageCurv(g_, vv, q, cv, mz, cmb, s.t[0].data(),
                                  wd_.data(), residualMode, ransTransport_,
                                  out.t[0].data());
            }
            if (rhsHook_) rhsHook_(s, out, cmb);
            return;
        }
        if (csLes_ > Real(0)) {
            kCompSmagNutCurv<<<launchBlocks(n), blockSize>>>(
                g_, csLes_ * csLes_, sgs_ == SgsModel::Sigma ? 1 : 0, cv,
                s.rho.data(), u_.data(), v_.data(), w_.data(), nut_.data());
            checkLastKernel("kCompSmagNutCurv");
            nutp = nut_.data();
            kapScale = kapLes;
        }
        kCompStageCurv<<<launchBlocks(n), blockSize>>>(
            g_, vv, q, cv, cmb, out.rho.data(), out.mx.data(),
            out.my.data(), out.mz.data(), out.E.data(), nutp, kapScale, mz);
        checkLastKernel("kCompStageCurv");
        if (rhsHook_) rhsHook_(s, out, cmb);
        return;
    }
    if (csLes_ > Real(0)) {
        kCompSmagNut<<<launchBlocks(n), blockSize>>>(
            g_, smagCd_, sgs_ == SgsModel::Sigma ? 1 : 0, s.rho.data(),
            u_.data(), v_.data(), w_.data(), nut_.data());
        checkLastKernel("kCompSmagNut");
        nutp = nut_.data();
        kapScale = kapLes;
    }
    kCompStage<<<launchBlocks(n), blockSize>>>(
        g_, vv, q, cmb, out.rho.data(), out.mx.data(), out.my.data(),
        out.mz.data(), out.E.data(), nutp, kapScale, mz);
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
    const Real* lts = ltsOn_ ? lts_.data() : nullptr;
    auto cmb = [lts](const DeviceCompState& a, Real ca,
                     const DeviceCompState& b, Real cb, Real cr) {
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
        return k;
    };
    if (hook_) hook_(s);
    stage(s, cmb(s, Real(1), s, Real(0), dt), q1_);
    if (hook_) hook_(q1_);
    stage(q1_, cmb(s, Real(0.75), q1_, Real(0.25), Real(0.25) * dt), q2_);
    if (hook_) hook_(q2_);
    // ca = 1 - cb, not Real(1/3): the two rounded thirds sum to 1 + 3e-8 in
    // FP32, which inflated every conserved total by ~2e-8 per step on any
    // non-uniform field (measured, gpu_mrf_conservation) and floored a steady
    // RANS residual at 2^-25 q / dt. 1 - Real(2/3) is exact (Sterbenz), so
    // ca + cb == 1 exactly; cr = cb dt is unchanged.
    const Real cb3 = Real(2.0 / 3.0), ca3 = Real(1) - cb3;
    stage(q2_, cmb(s, ca3, q2_, cb3, cb3 * dt), q1_);
    std::swap(s, q1_);
}

double DeviceComp::stableDt(const DeviceCompState& s, ReduceScratch& r,
                            double cfl) const
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
    if (csLes_ > Real(0)) {
        const int sig = sgs_ == SgsModel::Sigma ? 1 : 0;
        const double nutMax = curv_
            ? reduce(r, g_.size(),
                     SmagMaxCurv{g_, double(csLes_) * double(csLes_), sig,
                                 curvView(), s.rho.data(), s.mx.data(),
                                 s.my.data(), s.mz.data()},
                     rocprim::maximum<double>(), 0.0)
            : reduce(r, g_.size(),
                     SmagMax{g_, double(smagCd_), sig, s.rho.data(),
                             s.mx.data(), s.my.data(), s.mz.data()},
                     rocprim::maximum<double>(), 0.0);
        muEff += nutMax;
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

// dtc[c] = min(cfl / wave_c, diffusive_c): the local dt of --steady.
__global__ void kLocalDt(WavePerDtCurv wave, rans::RansDiffDt diff, double cfl,
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

// Squared per-cell residual of one equation over rows [jLo, jHi], 0
// elsewhere: 0 continuity, 1 |momentum|/rho, 2 R0/rho (energy, a scalar),
// 4 R0/rho[] with `rho` the conserved scalar itself (SST omega: the rate of
// ln omega).
struct ResidSq
{
    GpuGrid g;
    int jLo, jHi, which;
    const Real *rho, *R0, *R1, *R2, *R3; // R1..R3: momentum when which == 1
    __host__ __device__ double operator()(int c) const
    {
        int i, j, k;
        decompose(g, std::size_t(c), i, j, k);
        if (j < jLo || j > jHi) return 0.0;
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

std::expected<void, hipError_t> DeviceComp::enableRans(const RansSetup& rs)
{
    assert(curvilinear() && !les() && rs.model != RansModel::None &&
           rs.wallDist.size() == g_.size() && rs.omegaFloor > 0.0);
    assert(std::ranges::all_of(rs.wallDist, [](double d) { return d > 0.0; }));
    const std::size_t n = g_.size();
    const int nT = ransScalars(rs.model);
    for (auto* q : {&q1_, &q2_}) {
        auto e = DeviceCompState::make(n, nT);
        if (!e) return std::unexpected(e.error());
        *q = std::move(*e);
    }
    for (auto* b : {&nut_, &lts_}) {
        auto e = DeviceScalar::make(n);
        if (!e) return std::unexpected(e.error());
        *b = std::move(*e);
    }
    auto d = makeNarrow(rs.wallDist);
    if (!d) return std::unexpected(d.error());
    wd_ = std::move(*d);
    auto t = DeviceBuffer<double>::make(n);
    if (!t) return std::unexpected(t.error());
    dtc_ = std::move(*t);
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
    ransMirrorLo_ = rs.mirrorYLo;
    ransMirrorHi_ = rs.mirrorYHi;
    ransTransport_ = rs.transportOnly;
    return {};
}

double DeviceComp::localDt(const DeviceCompState& s, ReduceScratch& r,
                           double cfl)
{
    assert(rans());
    const std::size_t n = g_.size();
    const MrfView mz = mrf_ ? mrfView() : MrfView{};
    const WavePerDtCurv wave{g_, mz, s.rho.data(), s.mx.data(), s.my.data(),
                             s.mz.data(), s.E.data(), curv_->aov[0].data(),
                             curv_->aov[1].data(),
                             g_.nz > 1 ? curv_->aov[2].data() : nullptr,
                             double(tr_.gamma)};
    kLocalDt<<<launchBlocks(n), blockSize>>>(wave, ransDiffDt(s), cfl, n,
                                             dtc_.data());
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

ResidualNorms DeviceComp::residual(const DeviceCompState& s, int jLo, int jHi,
                                   ReduceScratch& r)
{
    assert(rans() && s.nT == q2_.nT && 0 <= jLo && jLo <= jHi && jHi < g_.ny);
    // ca = cb = 0, cr = 1, no lts: the mean-flow kernel and every hook then
    // write the pure rhs (0 a + 0 b + 1 R == R exactly); the scalar kernel
    // needs its explicit mode, because the point-implicit form would return
    // (R + J q)/(1 + J) instead.
    StageCombine k{{s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
                    s.E.data()},
                   {s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
                    s.E.data()},
                   Real(0), Real(0), Real(1)};
    for (int i = 0; i < s.nT; ++i) k.ta[i] = k.tb[i] = s.t[i].data();
    stage(s, k, q2_, true);
    const std::size_t n = g_.size();
    const double cnt = double(jHi - jLo + 1) * g_.nx * g_.nz;
    auto rms = [&](int which, const Real* R0) {
        const ResidSq f{g_, jLo, jHi, which, s.rho.data(), R0, q2_.mx.data(),
                        q2_.my.data(), q2_.mz.data()};
        return std::sqrt(
            reduce(r, n, f, rocprim::plus<double>(), 0.0) / cnt);
    };
    ResidualNorms out;
    out.rho = rms(0, q2_.rho.data());
    out.mom = rms(1, nullptr);
    out.E = rms(2, q2_.E.data());
    out.t[0] = rms(2, q2_.t[0].data());
    if (s.nT > 1) { // SST omega: R_w / (rho omega)
        const ResidSq f{g_, jLo, jHi, 4, s.t[1].data(), q2_.t[1].data(),
                        nullptr, nullptr, nullptr};
        out.t[1] = std::sqrt(reduce(r, n, f, rocprim::plus<double>(), 0.0) / cnt);
    }
    return out;
}

} // namespace dns::gpu
