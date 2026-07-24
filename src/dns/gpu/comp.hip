#include "dns/gpu/comp.hpp"

#include "dns/curvmesh.hpp"
#include "dns/gpu/comp_flux.hpp"
#include "dns/gpu/device_util.hpp"

#include <rocprim/device/device_reduce.hpp>
#include <rocprim/iterator/counting_iterator.hpp>
#include <rocprim/iterator/transform_iterator.hpp>

#include <cassert>
#include <climits>
#include <cmath>

namespace dns::gpu {
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
    if (sigma) {
        nut[c] = rho[c] * smagCd * Real(sigmaDsigma(gd));
        return;
    }
    Real s2 = 0;
    for (int cc = 0; cc < 3; ++cc)
        for (int d = 0; d < 3; ++d) {
            const Real sij = Real(0.5) * (gd[cc][d] + gd[d][cc]);
            s2 += Real(2) * sij * sij;
        }
    nut[c] = rho[c] * smagCd * sqrt(s2);
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

// Local copy of cutcell.hip's uploader (kept TU-private there, same pattern
// as the reduce driver).
std::expected<DeviceScalar, hipError_t> uploadNarrow(const Scalar& src)
{
    auto buf = DeviceScalar::make(src.size());
    if (!buf) return buf;
    std::vector<Real> tmp(src.begin(), src.end());
    if (auto e = buf->upload(tmp.data()); e != hipSuccess)
        return std::unexpected(e);
    return buf;
}

struct CompCutView
{
    const Real *ax, *ay, *az, *lam;
    const unsigned char *flag, *solid;
};

// --cut-flux band overwrite for the uniform comp RHS: kCompRhs runs first over
// the whole grid (interior cells keep its exact bits); this kernel then
// recomputes the flagged FLUID cells with aperture-weighted face fluxes over
// the lambda-floored volume. Faces to a solid-centre neighbour keep the FULL
// flux -- the ghost-mirror wall closure, unchanged, so kBodyFlux's
// fluid<->ghost tally remains the exact budget of the applied update. The KG
// split face flux stays single-valued per face, so aperture weighting is
// plain finite volume with varying face areas: conservative up to the
// lambda~ divisor (small-cell floor, transient-only error).
__global__ void kCompRhsCutOverwrite(GpuGrid g, Real mu, Real kappa, PrimView q,
                                     CompCutView cv, Real* orho, Real* omx,
                                     Real* omy, Real* omz, Real* oE,
                                     const Real* nut, Real kapScale)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    if (!cv.flag[c] || cv.solid[c]) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    Real drho = 0, dmx = 0, dmy = 0, dmz = 0, dE = 0;
    for (int a = 0; a < 3; ++a) {
        const int ox = a == 0, oy = a == 1, oz = a == 2;
        const Real h = a == 0 ? g.hx : a == 1 ? g.hy : g.hz;
        const std::size_t hiN = gidx(g, i + ox, j + oy, k + oz);
        const std::size_t loN = gidx(g, i - ox, j - oy, k - oz);
        const Real* A = a == 0 ? cv.ax : a == 1 ? cv.ay : cv.az;
        const Real wHi = cv.solid[hiN] ? Real(1) : A[hiN];
        const Real wLo = cv.solid[loN] ? Real(1) : A[c];
        const Flux hi = faceFlux(g, mu, kappa, q, i, j, k, a, nut, kapScale);
        const Flux lo =
            faceFlux(g, mu, kappa, q, i - ox, j - oy, k - oz, a, nut, kapScale);
        drho -= (wHi * hi.rho - wLo * lo.rho) / h;
        dmx -= (wHi * hi.mx - wLo * lo.mx) / h;
        dmy -= (wHi * hi.my - wLo * lo.my) / h;
        dmz -= (wHi * hi.mz - wLo * lo.mz) / h;
        dE -= (wHi * hi.E - wLo * lo.E) / h;
    }
    const Real lt = fmax(cv.lam[c], Real(0.5));
    orho[c] = drho / lt;
    omx[c] = dmx / lt;
    omy[c] = dmy / lt;
    omz[c] = dmz / lt;
    oE[c] = dE / lt;
}

__global__ void kCompRhsCurv(GpuGrid g, Real mu, Real kappa, PrimView q,
                             CurvView m, Real* orho, Real* omx, Real* omy,
                             Real* omz, Real* oE)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    Real drho = 0, dmx = 0, dmy = 0, dmz = 0, dE = 0;
    for (int a = 0; a < 3; ++a) {
        const int ox = a == 0, oy = a == 1, oz = a == 2;
        const Flux hi = faceFluxCurv(g, mu, kappa, q, m, i, j, k, a);
        const Flux lo = faceFluxCurv(g, mu, kappa, q, m, i - ox, j - oy, k - oz, a);
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
    const Real* lam = nullptr;   // --cut-flux: 1/lambda~ volume amplification
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
        if (lam) m /= fmax(double(lam[c]), 0.5);
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
        if (sigma) return double(rho[c]) * smagCd * sigmaDsigma(gd);
        double s2 = 0;
        for (int cc = 0; cc < 3; ++cc)
            for (int d = 0; d < 3; ++d) {
                const double sij = 0.5 * (gd[cc][d] + gd[d][cc]);
                s2 += 2.0 * sij * sij;
            }
        return double(rho[c]) * smagCd * sqrt(s2);
    }
};

// Local copy of the reduction driver (reductions.hip keeps its own private).
template <class F, class Op>
double reduce(ReduceScratch& s, std::size_t n, F f, Op op, double init)
{
    assert(n <= std::size_t(INT_MAX));
    const auto in =
        rocprim::make_transform_iterator(rocprim::counting_iterator<int>(0), f);
    std::size_t bytes = 0;
    (void)rocprim::reduce(nullptr, bytes, in, s.out.data(), init, n, op);
    if (s.tmp.size() < bytes) {
        auto grown = DeviceBuffer<unsigned char>::make(bytes);
        // assert() is a comment in Release: the old form dereferenced a
        // valueless expected on an allocation failure.
        if (!grown) return init;
        s.tmp = std::move(*grown);
    }
    bytes = s.tmp.size();
    (void)rocprim::reduce(s.tmp.data(), bytes, in, s.out.data(), init, n, op);
    double h = 0.0;
    (void)hipMemcpy(&h, s.out.data(), sizeof(double), hipMemcpyDeviceToHost);
    return h;
}

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
                                                       double gamma, double mu)
{
    auto d = make(m.g, gamma, mu, 0.0);
    if (!d) return d;
    auto cv = DeviceCurv::make(m);
    if (!cv) return std::unexpected(cv.error());
    d->curv_ = std::move(*cv);
    return d;
}

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
    if (curv_) {
        CurvView cv;
        for (int a = 0; a < 3; ++a) {
            for (int c = 0; c < 3; ++c) cv.S[a][c] = curv_->S[a][c].data();
            for (int e = 0; e < 9; ++e) cv.Ji[a][e] = curv_->Ji[a][e].data();
        }
        cv.invVol = curv_->invVol.data();
        kCompRhsCurv<<<launchBlocks(n), blockSize>>>(
            g_, mu_, kappa, q, cv, out.rho.data(), out.mx.data(),
            out.my.data(), out.mz.data(), out.E.data());
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
        kapScale = gamma_ * Rgas_ / ((gamma_ - Real(1)) * prt_); // cp / Pr_t
    }
    kCompRhs<<<launchBlocks(n), blockSize>>>(g_, mu_, kappa, q, out.rho.data(),
                                             out.mx.data(), out.my.data(),
                                             out.mz.data(), out.E.data(), nutp,
                                             kapScale);
    checkLastKernel("kCompRhs");
    if (cutFlux_) {
        const CompCutView cv{cutAx_.data(), cutAy_.data(), cutAz_.data(),
                             cutLam_.data(), cutFlag_.data(), cutSolid_.data()};
        kCompRhsCutOverwrite<<<launchBlocks(n), blockSize>>>(
            g_, mu_, kappa, q, cv, out.rho.data(), out.mx.data(),
            out.my.data(), out.mz.data(), out.E.data(), nutp, kapScale);
        checkLastKernel("kCompRhsCutOverwrite");
    }
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

std::expected<void, hipError_t>
DeviceComp::attachCutFlux(const CutGeometry& cut,
                          const std::vector<unsigned char>& flag,
                          const std::vector<unsigned char>& solid)
{
    assert(!curv_ && flag.size() == g_.size() && solid.size() == g_.size());
    auto ax = uploadNarrow(cut.ax);
    if (!ax) return std::unexpected(ax.error());
    auto ay = uploadNarrow(cut.ay);
    if (!ay) return std::unexpected(ay.error());
    auto az = uploadNarrow(cut.az);
    if (!az) return std::unexpected(az.error());
    auto lam = uploadNarrow(cut.lambda);
    if (!lam) return std::unexpected(lam.error());
    auto fb = DeviceBuffer<unsigned char>::make(flag.size());
    auto sb = DeviceBuffer<unsigned char>::make(solid.size());
    if (!fb) return std::unexpected(fb.error());
    if (!sb) return std::unexpected(sb.error());
    if (auto e = fb->upload(flag.data()); e != hipSuccess)
        return std::unexpected(e);
    if (auto e = sb->upload(solid.data()); e != hipSuccess)
        return std::unexpected(e);
    cutAx_ = std::move(*ax);
    cutAy_ = std::move(*ay);
    cutAz_ = std::move(*az);
    cutLam_ = std::move(*lam);
    cutFlag_ = std::move(*fb);
    cutSolid_ = std::move(*sb);
    cutFlux_ = true;
    return {};
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
                           g_.nz > 1 ? 1.0 / g_.hz : 0.0,
                           cutFlux_ ? cutLam_.data() : nullptr},
                 rocprim::maximum<double>(), 0.0);
    double dt = cfl / wmax;
    double muEff = double(mu_);
    if (csLes_ > Real(0)) {
        const double nutMax = reduce(
            r, g_.size(),
            SmagMax{g_, double(smagCd_), sgs_ == SgsModel::Sigma ? 1 : 0,
                    s.rho.data(), s.mx.data(), s.my.data(), s.mz.data()},
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
