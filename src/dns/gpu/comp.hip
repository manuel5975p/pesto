#include "dns/gpu/comp.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/internal/comp_flux.hpp"
#include "dns/gpu/device_util.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include "dns/gpu/internal/reduce.hpp"

#include <rocprim/iterator/transform_iterator.hpp>

#include <cassert>
#include <climits>
#include <cmath>

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
    const Real inv[3] = {Real(0.5) / g.hx, Real(0.5) / g.hy, Real(0.5) / g.hz};
    const Real* vc[3] = {u, v, w};
    Real gd[3][3]; // gd[c][d] = d vc[c] / d x_d
    for (int cc = 0; cc < 3; ++cc) {
        gd[cc][0] = (vc[cc][gidx(g, i + 1, j, k)] -
                     vc[cc][gidx(g, i - 1, j, k)]) * inv[0];
        gd[cc][1] = (vc[cc][gidx(g, i, j + 1, k)] -
                     vc[cc][gidx(g, i, j - 1, k)]) * inv[1];
        gd[cc][2] = (vc[cc][gidx(g, i, j, k + 1)] -
                     vc[cc][gidx(g, i, j, k - 1)]) * inv[2];
    }
    nut[c] = rho[c] * smagCd * sgsRate(sigma, gd);
}

__global__ void kCompRhs(GpuGrid g, Real mu, Real kappa, PrimView q, Real* orho,
                         Real* omx, Real* omy, Real* omz, Real* oE,
                         const Real* nut, Real kapScale)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    Real drho = 0, dmx = 0, dmy = 0, dmz = 0, dE = 0;
    for (int a = 0; a < 3; ++a) {
        const int ox = a == 0, oy = a == 1, oz = a == 2;
        const Real h = a == 0 ? g.hx : a == 1 ? g.hy : g.hz;
        const Flux hi = faceFlux(g, mu, kappa, q, i, j, k, a, nut, kapScale);
        const Flux lo =
            faceFlux(g, mu, kappa, q, i - ox, j - oy, k - oz, a, nut, kapScale);
        drho -= (hi.rho - lo.rho) / h;
        dmx -= (hi.mx - lo.mx) / h;
        dmy -= (hi.my - lo.my) / h;
        dmz -= (hi.mz - lo.mz) / h;
        dE -= (hi.E - lo.E) / h;
    }
    orho[c] = drho;
    omx[c] = dmx;
    omy[c] = dmy;
    omz[c] = dmz;
    oE[c] = dE;
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
    const Real* vc[3] = {u, v, w};
    Real gd[3][3];
    curvVelGradCentre(g, m, i, j, k,
                      [&](std::size_t cell, int comp) { return vc[comp][cell]; },
                      gd);
    const Real cd = cs2 * curvDelta2(g, m.invVol[c]);
    nut[c] = rho[c] * cd * sgsRate(sigma, gd);
}

__global__ void kCompRhsCurv(GpuGrid g, Real mu, Real kappa, PrimView q,
                             CurvView m, Real* orho, Real* omx, Real* omy,
                             Real* omz, Real* oE, const Real* nut,
                             Real kapScale)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    Real drho = 0, dmx = 0, dmy = 0, dmz = 0, dE = 0;
    for (int a = 0; a < 3; ++a) {
        const int ox = a == 0, oy = a == 1, oz = a == 2;
        const Flux hi =
            faceFluxCurv(g, mu, kappa, q, m, i, j, k, a, nut, kapScale);
        const Flux lo = faceFluxCurv(g, mu, kappa, q, m, i - ox, j - oy, k - oz,
                                     a, nut, kapScale);
        drho -= hi.rho - lo.rho;
        dmx -= hi.mx - lo.mx;
        dmy -= hi.my - lo.my;
        dmz -= hi.mz - lo.mz;
        dE -= hi.E - lo.E;
    }
    const Real iv = m.invVol[c];
    orho[c] = drho * iv;
    omx[c] = dmx * iv;
    omy[c] = dmy * iv;
    omz[c] = dmz * iv;
    oE[c] = dE * iv;
}

// Curvilinear acoustic bound: (|u| + c) / width, width = 1/aov per axis;
// |u| >= |u . n| makes the bound conservative on skewed cells.
struct WavePerDtCurv
{
    const Real *rho, *mx, *my, *mz, *E;
    const Real *ax, *ay, *az; // az == nullptr when nz == 1
    double gamma;
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double u = mx[c] / r, v = my[c] / r, w = mz[c] / r;
        const double p = (gamma - 1.0) *
            (double(E[c]) - 0.5 * (u * u + v * v + w * w) * r);
        const double sp = sqrt(u * u + v * v + w * w) + sqrt(gamma * p / r);
        double m = sp * double(ax[c] > ay[c] ? ax[c] : ay[c]);
        if (az && sp * az[c] > m) m = sp * az[c];
        return m;
    }
};

// out = ca a + cb b + cr k, all five components.
__global__ void kAxpy5(std::size_t n, Real ca, const Real* const* av, Real cb,
                       const Real* const* bv, Real cr, const Real* const* kv,
                       Real* const* ov)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    for (int q = 0; q < 5; ++q)
        ov[q][c] = ca * av[q][c] + cb * bv[q][c] + cr * kv[q][c];
}

struct WavePerDt
{
    const Real *rho, *mx, *my, *mz, *E;
    double gamma, ihx, ihy, ihz; // degenerate axes pass 0
    __host__ __device__ double operator()(int c) const
    {
        const double r = rho[c];
        const double u = mx[c] / r, v = my[c] / r, w = mz[c] / r;
        const double p = (gamma - 1.0) *
            (double(E[c]) - 0.5 * (u * u + v * v + w * w) * r);
        const double cs = sqrt(gamma * p / r);
        double m = (fabs(u) + cs) * ihx;
        const double wy = (fabs(v) + cs) * ihy;
        const double wz = (fabs(w) + cs) * ihz;
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
        decompose(g, c, i, j, k);
        const double inv[3] = {0.5 / g.hx, 0.5 / g.hy, 0.5 / g.hz};
        const int nb[3][2] = {{int(gidx(g, i + 1, j, k)), int(gidx(g, i - 1, j, k))},
                              {int(gidx(g, i, j + 1, k)), int(gidx(g, i, j - 1, k))},
                              {int(gidx(g, i, j, k + 1)), int(gidx(g, i, j, k - 1))}};
        double gd[3][3];
        for (int cc = 0; cc < 3; ++cc)
            for (int d = 0; d < 3; ++d)
                gd[cc][d] = (vel(nb[d][0], cc) - vel(nb[d][1], cc)) * inv[d];
        return double(rho[c]) * smagCd * sgsRate(sigma, gd);
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
        double gd[3][3];
        curvVelGradCentre(g, m, i, j, k,
                          [&](std::size_t cell, int comp) {
                              return vel(cell, comp);
                          },
                          gd);
        const double cd = cs2 * double(curvDelta2(g, m.invVol[c]));
        return double(rho[c]) * cd * sgsRate(sigma, gd);
    }
};

// Local copy of the reduction driver (reductions.hip keeps its own private).

hipError_t copyOne(DeviceScalar& dst, const std::vector<double>& src, bool up,
                   std::vector<double>* down)
{
    std::vector<Real> tmp(dst.size());
    if (up) {
        for (std::size_t i = 0; i < tmp.size(); ++i) tmp[i] = Real(src[i]);
        return dst.upload(tmp.data());
    }
    if (auto e = dst.download(tmp.data()); e != hipSuccess) return e;
    for (std::size_t i = 0; i < tmp.size(); ++i) (*down)[i] = tmp[i];
    return hipSuccess;
}

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
    std::vector<Real> tmp(n);
    auto up = [&](DeviceScalar& dst,
                  const std::vector<double>& src) -> hipError_t {
        auto b = DeviceScalar::make(n);
        if (!b) return b.error();
        for (std::size_t i = 0; i < n; ++i) tmp[i] = Real(src[i]);
        if (auto e = b->upload(tmp.data()); e != hipSuccess) return e;
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

std::expected<DeviceCompState, hipError_t> DeviceCompState::make(std::size_t n)
{
    DeviceCompState s;
    for (auto* b : {&s.rho, &s.mx, &s.my, &s.mz, &s.E}) {
        auto e = DeviceScalar::make(n);
        if (!e) return std::unexpected(e.error());
        *b = std::move(*e);
    }
    return s;
}

hipError_t DeviceCompState::upload(const CompState& h)
{
    if (auto e = copyOne(rho, h.rho, true, nullptr); e != hipSuccess) return e;
    if (auto e = copyOne(mx, h.mx, true, nullptr); e != hipSuccess) return e;
    if (auto e = copyOne(my, h.my, true, nullptr); e != hipSuccess) return e;
    if (auto e = copyOne(mz, h.mz, true, nullptr); e != hipSuccess) return e;
    return copyOne(E, h.E, true, nullptr);
}

hipError_t DeviceCompState::download(CompState& h) const
{
    auto& self = const_cast<DeviceCompState&>(*this);
    if (auto e = copyOne(self.rho, {}, false, &h.rho); e != hipSuccess) return e;
    if (auto e = copyOne(self.mx, {}, false, &h.mx); e != hipSuccess) return e;
    if (auto e = copyOne(self.my, {}, false, &h.my); e != hipSuccess) return e;
    if (auto e = copyOne(self.mz, {}, false, &h.mz); e != hipSuccess) return e;
    return copyOne(self.E, {}, false, &h.E);
}

std::expected<DeviceComp, hipError_t> DeviceComp::make(const Grid& g,
                                                       double gamma, double mu,
                                                       double csLes, SgsModel sgs)
{
    assert(gamma > 1.0 && mu >= 0.0 && csLes >= 0.0);
    DeviceComp d(GpuGrid::from(g), Real(gamma), Real(mu), Real(csLes), sgs);
    const std::size_t n = g.size();
    for (auto* s : {&d.k_, &d.q1_, &d.q2_}) {
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
    auto pt = DeviceBuffer<const Real*>::make(20);
    if (!pt) return std::unexpected(pt.error());
    d.ptrs_ = std::move(*pt);
    return d;
}

std::expected<DeviceComp, hipError_t> DeviceComp::make(const CurvMetrics& m,
                                                       double gamma, double mu,
                                                       double csLes,
                                                       SgsModel sgs)
{
    auto d = make(m.g, gamma, mu, csLes, sgs);
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

DeviceComp::DeviceComp(const GpuGrid& g, Real gamma, Real mu, Real csLes,
                       SgsModel sgs)
    : g_(g), gamma_(gamma), mu_(mu), csLes_(csLes), sgs_(sgs)
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
        g_, gamma_, Rgas_, s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
        s.E.data(), u_.data(), v_.data(), w_.data(), e_.data(), p_.data(),
        T_.data());
    checkLastKernel("kPrim");
}

void DeviceComp::rhs(const DeviceCompState& s, DeviceCompState& out)
{
    primitives(s);
    const Real kappa = kappaCond();
    const PrimView q{s.rho.data(), u_.data(), v_.data(), w_.data(), e_.data(),
                     p_.data(), T_.data()};
    const std::size_t n = g_.size();
    const Real kapLes = gamma_ * Rgas_ / ((gamma_ - Real(1)) * prt_); // cp/Pr_t
    if (curv_) {
        const CurvView cv = curvView();
        const Real* nutp = nullptr;
        Real kapScale = Real(0);
        if (csLes_ > Real(0)) {
            kCompSmagNutCurv<<<launchBlocks(n), blockSize>>>(
                g_, csLes_ * csLes_, sgs_ == SgsModel::Sigma ? 1 : 0, cv,
                s.rho.data(), u_.data(), v_.data(), w_.data(), nut_.data());
            checkLastKernel("kCompSmagNutCurv");
            nutp = nut_.data();
            kapScale = kapLes;
        }
        kCompRhsCurv<<<launchBlocks(n), blockSize>>>(
            g_, mu_, kappa, q, cv, out.rho.data(), out.mx.data(),
            out.my.data(), out.mz.data(), out.E.data(), nutp, kapScale);
        checkLastKernel("kCompRhsCurv");
        if (rhsHook_) rhsHook_(s, out);
        return;
    }
    const Real* nutp = nullptr;
    Real kapScale = Real(0);
    if (csLes_ > Real(0)) {
        kCompSmagNut<<<launchBlocks(n), blockSize>>>(
            g_, smagCd_, sgs_ == SgsModel::Sigma ? 1 : 0, s.rho.data(),
            u_.data(), v_.data(), w_.data(), nut_.data());
        checkLastKernel("kCompSmagNut");
        nutp = nut_.data();
        kapScale = kapLes;
    }
    kCompRhs<<<launchBlocks(n), blockSize>>>(g_, mu_, kappa, q, out.rho.data(),
                                             out.mx.data(), out.my.data(),
                                             out.mz.data(), out.E.data(), nutp,
                                             kapScale);
    checkLastKernel("kCompRhs");
    if (rhsHook_) rhsHook_(s, out);
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
    const std::size_t n = g_.size();

    auto axpy = [&](Real ca, const DeviceCompState& a, Real cb,
                    const DeviceCompState& b, Real cr, const DeviceCompState& k,
                    DeviceCompState& o) {
        const Real* h[20] = {
            a.rho.data(), a.mx.data(), a.my.data(), a.mz.data(), a.E.data(),
            b.rho.data(), b.mx.data(), b.my.data(), b.mz.data(), b.E.data(),
            k.rho.data(), k.mx.data(), k.my.data(), k.mz.data(), k.E.data(),
            o.rho.data(), o.mx.data(), o.my.data(), o.mz.data(), o.E.data()};
        (void)ptrs_.upload(h);
        const Real* const* d = ptrs_.data();
        kAxpy5<<<launchBlocks(n), blockSize>>>(
            n, ca, d, cb, d + 5, cr, d + 10,
            const_cast<Real* const*>(reinterpret_cast<const Real* const*>(d + 15)));
        checkLastKernel("kAxpy5");
    };
    if (hook_) hook_(s);
    rhs(s, k_);
    axpy(Real(1), s, Real(0), s, dt, k_, q1_);
    if (hook_) hook_(q1_);
    rhs(q1_, k_);
    axpy(Real(0.75), s, Real(0.25), q1_, Real(0.25) * dt, k_, q2_);
    if (hook_) hook_(q2_);
    rhs(q2_, k_);
    axpy(Real(1.0 / 3.0), s, Real(2.0 / 3.0), q2_, Real(2.0 / 3.0) * dt, k_, s);
}

double DeviceComp::stableDt(const DeviceCompState& s, ReduceScratch& r,
                            double cfl) const
{
    const double wmax = curv_
        ? reduce(r, g_.size(),
                 WavePerDtCurv{s.rho.data(), s.mx.data(), s.my.data(),
                               s.mz.data(), s.E.data(), curv_->aov[0].data(),
                               curv_->aov[1].data(),
                               g_.nz > 1 ? curv_->aov[2].data() : nullptr,
                               double(gamma_)},
                 rocprim::maximum<double>(), 0.0)
        : reduce(r, g_.size(),
                 WavePerDt{s.rho.data(), s.mx.data(), s.my.data(), s.mz.data(),
                           s.E.data(), double(gamma_), 1.0 / g_.hx, 1.0 / g_.hy,
                           g_.nz > 1 ? 1.0 / g_.hz : 0.0},
                 rocprim::maximum<double>(), 0.0);
    double dt = cfl / wmax;
    double muEff = double(mu_);
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
        const double dtv = 0.25 * h * h * rhomin / muEff;
        if (dtv < dt) dt = dtv;
    }
    return dt;
}

} // namespace dns::gpu
