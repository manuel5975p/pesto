#include "dns/gpu/tfles.hpp"

#include "dns/chem.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/plane_sums.hpp"
#include "dns/gpu/internal/reduce.hpp"

#include <cassert>
#include <cmath>

namespace dns::gpu {
namespace {

// Omega_raw = tanh(beta w_s), w_s = exp(lnAs - TaS/T) rho max(Y, 0): the ONE
// Arrhenius expression at the reduced activation, normalised to peak 1 on
// the laminar manifold (tflesSensorLogMax).
__global__ void kTflesSensorRaw(GpuGrid g, const Real* rho, const Real* ry,
                                const Real* T, Real lnAs, Real TaS, Real beta,
                                Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    const Real r = rho[c];
    const Real ws = arrheniusRate(lnAs, TaS, r, ry[c] / r, T[c]);
    out[c] = tanhf(beta * ws);
}

// One separable pass of the T-gated exact max filter of radius rad along
// `axis`: out[c] = max of in[n] over the line cells n within rad with T[n] >=
// T[c] (c itself included) -- the sensor spreads toward colder gas only. x
// clamped to [0, nx - 1] (open faces), y and z periodic; an axis of n cells
// with 2 rad + 1 >= n is the whole line. Composed over the axes the gate is
// transitive (T_n >= T_m >= T_c), so every contribution is hotter than c.
template <int axis>
__global__ void kTflesMaxDilate(GpuGrid g, int rad, const Real* T,
                                const Real* in, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const int n = axis == 0 ? g.nx : axis == 1 ? g.ny : g.nz;
    const int q = axis == 0 ? i : axis == 1 ? j : k;
    const std::ptrdiff_t s = axis == 0 ? 1
                             : axis == 1 ? std::ptrdiff_t(g.nx)
                                         : std::ptrdiff_t(g.nx) * g.ny;
    const std::size_t base = c - std::size_t(q) * std::size_t(s);
    const Real Tc = T[c];
    Real m = in[c];
    auto take = [&](int p) {
        const std::size_t a = base + std::size_t(p) * std::size_t(s);
        if (T[a] >= Tc) m = fmaxf(m, in[a]);
    };
    if constexpr (axis == 0) {
        const int lo = q - rad < 0 ? 0 : q - rad;
        const int hi = q + rad > n - 1 ? n - 1 : q + rad;
        for (int p = lo; p <= hi; ++p) take(p);
    } else {
        if (2 * rad + 1 >= n) {
            for (int p = 0; p < n; ++p) take(p);
        } else {
            for (int d = -rad; d <= rad; ++d) take(((q + d) % n + n) % n);
        }
    }
    out[c] = m;
}

// curl u, central, wrap-aware (offsetsAt).
__global__ void kTflesCurl(GpuGrid g, const Real* u, const Real* v,
                           const Real* w, Real* cx, Real* cy, Real* cz)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Real ix = Real(0.5) / g.hx, iy = Real(0.5) / g.hy,
               iz = Real(0.5) / g.hz;
    auto d = [&](const Real* f, int a, Real inv) {
        return (f[c + o.p[a]] - f[c + o.m[a]]) * inv;
    };
    cx[c] = d(w, 1, iy) - d(v, 2, iz);
    cy[c] = d(u, 2, iz) - d(w, 0, ix);
    cz[c] = d(v, 0, ix) - d(u, 1, iy);
}

// Colin's grid-scale u' = c2 h^3 |lap(curl u)| (7-point Laplacian per
// component, wrap-aware); coef = c2 h^3.
__global__ void kTflesColinU(GpuGrid g, const Real* cx, const Real* cy,
                             const Real* cz, Real coef, Real* out)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    decompose(g, c, i, j, k);
    const Offsets o = offsetsAt(g, i, j, k);
    const Real ih[3] = {Real(1) / (g.hx * g.hx), Real(1) / (g.hy * g.hy),
                        Real(1) / (g.hz * g.hz)};
    const Real* f[3] = {cx, cy, cz};
    Real s2 = 0;
    for (int q = 0; q < 3; ++q) {
        Real l = 0;
        for (int a = 0; a < 3; ++a)
            l += (f[q][c + o.p[a]] + f[q][c + o.m[a]] - Real(2) * f[q][c]) * ih[a];
        s2 += l * l;
    }
    out[c] = coef * sqrtf(s2);
}

// F, E -> FE, EoF, Om. omega null = the uniform sensor (Omega 1); uh null =
// no efficiency (E 1).
__global__ void kTflesCombine(GpuGrid g, Real Fmax, const Real* omega,
                              const Real* uh, int edge, Real deltaL, Real nuU,
                              Real beta, Real* fe, Real* eof, Real* om)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    const Real W = omega ? omega[c] : Real(1);
    const Real F = Real(1) + (Fmax - Real(1)) * W;
    Real E = Real(1);
    if (uh) {
        int i, j, k;
        decompose(g, c, i, j, k);
        if (i >= edge && i < g.nx - edge) {
            const Real D = F * deltaL;
            const Real up = colinUprime(uh[c], D, g.hx);
            E = charletteEfficiency(F, up, up * D / nuU, beta);
        }
    }
    fe[c] = F * E;
    eof[c] = E / F;
    om[c] = W;
}

__global__ void kTflesColumns(GpuGrid g, const Real* fe, const Real* eof,
                              const Real* om, double* out)
{
    const int i = int(std::size_t(blockIdx.x) * blockDim.x + threadIdx.x);
    if (i >= g.nx) return;
    double s[DeviceTfles::kTflesSums] = {0.0, 0.0, 0.0};
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j) {
            const std::size_t c = gidx(g, i, j, k);
            const double W = om[c];
            s[0] += W;
            s[1] += W * sqrt(double(fe[c]) * double(eof[c]));
            s[2] = fmax(s[2], double(fe[c]));
        }
    for (int q = 0; q < DeviceTfles::kTflesSums; ++q)
        out[std::size_t(i) * DeviceTfles::kTflesSums + q] = s[q];
}

struct MaxOf
{
    const Real* f;
    __host__ __device__ double operator()(int c) const
    {
        const double v = f[c];
        return isfinite(v) ? v : HUGE_VAL; // never dropped by the max
    }
};

__global__ void kMutCopyOpenX(GpuGrid g, Real* nut)
{
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t rows = std::size_t(g.ny) * g.nz;
    if (t >= rows) return;
    const int j = int(t % g.ny), k = int(t / g.ny);
    nut[gidx(g, 0, j, k)] = nut[gidx(g, 1, j, k)];
    nut[gidx(g, g.nx - 1, j, k)] = nut[gidx(g, g.nx - 2, j, k)];
}

std::expected<DeviceScalar, hipError_t> alloc(std::size_t n)
{
    return DeviceScalar::make(n);
}

} // namespace

std::expected<DeviceTfles, hipError_t> DeviceTfles::make(const Grid& g,
                                                         const TflesSetup& s)
{
    assert(s.F >= 1 && s.deltaL > 0 && s.nuU > 0 && s.dilate >= 0);
    assert(!s.charlette || g.nz > 1);
    DeviceTfles t;
    t.g_ = GpuGrid::from(g);
    t.s_ = s;
    const std::size_t n = g.size();
    std::vector<DeviceScalar*> bufs = {&t.fe_, &t.eof_, &t.om_};
    if (!s.uniform) bufs.insert(bufs.end(), {&t.ra_, &t.rb_});
    if (s.charlette) bufs.insert(bufs.end(), {&t.cx_, &t.cy_, &t.cz_, &t.uh_});
    for (DeviceScalar* b : bufs) {
        auto a = alloc(n);
        if (!a) return std::unexpected(a.error());
        *b = std::move(*a);
    }
    // The constant field (uniform, no efficiency) is formed here, once.
    if (s.uniform && !s.charlette) t.combine(nullptr, nullptr);
    return t;
}

void DeviceTfles::combine(const Real* omega, const Real* uh)
{
    kTflesCombine<<<launchBlocks(g_.size()), blockSize>>>(
        g_, Real(s_.F), omega, uh, kTflesEdgeRows, Real(s_.deltaL),
        Real(s_.nuU), Real(s_.effBeta), fe_.data(), eof_.data(), om_.data());
    checkLastKernel("kTflesCombine");
}

void DeviceTfles::dilate(const DeviceScalar& raw, const DeviceScalar& T,
                         DeviceScalar& out)
{
    assert(!s_.uniform && &raw != &out);
    const std::size_t n = g_.size();
    const int r = s_.dilate;
    const Real* t = T.data();
    // raw -> rb (x) -> out (y) [-> rb (z) -> out, nz > 1]
    kTflesMaxDilate<0><<<launchBlocks(n), blockSize>>>(g_, r, t, raw.data(),
                                                         rb_.data());
    kTflesMaxDilate<1><<<launchBlocks(n), blockSize>>>(g_, r, t, rb_.data(),
                                                         out.data());
    if (g_.nz > 1) {
        kTflesMaxDilate<2><<<launchBlocks(n), blockSize>>>(g_, r, t, out.data(),
                                                             rb_.data());
        (void)hipMemcpy(out.data(), rb_.data(), n * sizeof(Real),
                        hipMemcpyDeviceToDevice);
    }
    checkLastKernel("kTflesMaxDilate");
}

void DeviceTfles::update(const DeviceCompState& s, const DeviceScalar& u,
                         const DeviceScalar& v, const DeviceScalar& w,
                         const DeviceScalar& T)
{
    if (s_.uniform && !s_.charlette) return; // constant, formed at make()
    const std::size_t n = g_.size();
    const Real* omega = nullptr;
    if (!s_.uniform) {
        kTflesSensorRaw<<<launchBlocks(n), blockSize>>>(
            g_, s.rho.data(), s.y[0].data(), T.data(), Real(s_.lnAs),
            Real(s_.TaS), Real(s_.sensorBeta), ra_.data());
        checkLastKernel("kTflesSensorRaw");
        // raw (ra) -> x (rb) -> y (ra) -> z (rb) | nz == 1: ends in ra.
        const int r = s_.dilate;
        const Real* t = T.data();
        kTflesMaxDilate<0><<<launchBlocks(n), blockSize>>>(g_, r, t, ra_.data(),
                                                             rb_.data());
        kTflesMaxDilate<1><<<launchBlocks(n), blockSize>>>(g_, r, t, rb_.data(),
                                                             ra_.data());
        omega = ra_.data();
        if (g_.nz > 1) {
            kTflesMaxDilate<2><<<launchBlocks(n), blockSize>>>(
                g_, r, t, ra_.data(), rb_.data());
            omega = rb_.data();
        }
        checkLastKernel("kTflesMaxDilate");
    }
    const Real* uh = nullptr;
    if (s_.charlette) {
        kTflesCurl<<<launchBlocks(n), blockSize>>>(g_, u.data(), v.data(),
                                                   w.data(), cx_.data(),
                                                   cy_.data(), cz_.data());
        checkLastKernel("kTflesCurl");
        const double h = g_.hx;
        kTflesColinU<<<launchBlocks(n), blockSize>>>(
            g_, cx_.data(), cy_.data(), cz_.data(), Real(kColinC2 * h * h * h),
            uh_.data());
        checkLastKernel("kTflesColinU");
        uh = uh_.data();
    }
    combine(omega, uh);
}

double DeviceTfles::maxFE(ReduceScratch& r) const
{
    return internal::reduce(r, g_.size(), MaxOf{fe_.data()},
                            rocprim::maximum<double>(), 0.0);
}

hipError_t DeviceTfles::columnStats(DeviceBuffer<double>& scratch,
                                    std::vector<double>& host) const
{
    const std::size_t n = std::size_t(g_.nx) * kTflesSums;
    if (auto e = internal::ensurePlaneScratch(scratch, n); e != hipSuccess)
        return e;
    kTflesColumns<<<launchBlocks(std::size_t(g_.nx)), blockSize>>>(
        g_, fe_.data(), eof_.data(), om_.data(), scratch.data());
    checkLastKernel("kTflesColumns");
    return internal::fetchPlaneScratch(scratch, n, host);
}

MutHook mutCopyOpenXHook(const GpuGrid& g)
{
    assert(g.nx >= 3);
    return [g](Real* nut, Real*) {
        const std::size_t rows = std::size_t(g.ny) * g.nz;
        kMutCopyOpenX<<<launchBlocks(rows), blockSize>>>(g, nut);
        checkLastKernel("kMutCopyOpenX");
    };
}

} // namespace dns::gpu
