#include "dns/gpu/nscbc_modal.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/dct.hpp"

#include <cmath>
#include <cstdio>

namespace dns::gpu {

namespace {

void checkFft(hipfftResult r, const char* what)
{
    if (r != HIPFFT_SUCCESS)
        std::fprintf(stderr, "dns: hipfft %s failed (status %d)\n", what, int(r));
}

// The two halves of the incoming-acoustic closure on the x-hi row, arithmetic
// for arithmetic the same as kNscbcCells' a == 0, hi branch (comp_nscbc.hip):
//   a = L5  = (un + c)(dp/dx + rho c du/dx)   -- the outgoing acoustic amplitude
//   b = Tp - rho c Tu                         -- the Yoo-Im transverse forcing
// The transverse derivatives are central because this path is only reachable
// with walled laterals (openY/openZ false), where kNscbcCells is central too;
// at nz == 1 the z pair wraps onto the same cell and cancels identically.
__global__ void kFacePlanes(GpuGrid g, Real gam, const Real* rho, const Real* u,
                            const Real* v, const Real* w, const Real* p,
                            Real* aOut, Real* bOut)
{
    const std::size_t n = std::size_t(g.ny) * g.nz;
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= n) return;
    const int j = int(m % g.ny), k = int(m / g.ny), i = g.nx - 1;
    const std::size_t c = gidx(g, i, j, k), cn = gidx(g, i - 1, j, k);

    const Real r = rho[c], pc = p[c];
    const Real vc[3] = {u[c], v[c], w[c]};
    const Real cs = sqrt(gam * pc / r);
    const Real dp_ = (p[c] - p[cn]) / g.hx;
    const Real dun = (u[c] - u[cn]) / g.hx;

    const Real* vf[3] = {u, v, w};
    const Real hh[3] = {g.hx, g.hy, g.hz};
    auto ct = [&](const Real* f, int d) {
        const int px = d == 0, py = d == 1, pz = d == 2;
        const Real fp = f[gidx(g, i + px, j + py, k + pz)];
        const Real fm = f[gidx(g, i - px, j - py, k - pz)];
        return (fp - fm) * (Real(0.5) / hh[d]);
    };

    Real Tp = 0, Tun = 0;
    for (int d = 1; d < 3; ++d)
    {
        Tp += vc[d] * ct(p, d) + gam * pc * ct(vf[d], d);
        Tun += vc[d] * ct(u, d);
    }
    aOut[m] = (vc[0] + cs) * (dp_ + r * cs * dun);
    bOut[m] = Tp - r * cs * Tun; // s = +1 on the hi face
}

// beta(theta) = cos t / (1 + cos t) from cos t = a / (a + b), capped at the
// configured beta (the exact function never exceeds 1/2, so the shipped 0.5
// does not bind). Degenerate inputs fall back to the cap, i.e. to today's
// arithmetic: a + b ~ 0 against a live a or b means the two disagree about the
// sign of the wave, which is noise, not an angle.
__device__ inline Real betaOf(Real a, Real b, Real cap)
{
    const Real scale = fabs(a) + fabs(b);
    const Real s = a + b;
    Real t = Real(1);
    if (scale > Real(0) && fabs(s) > Real(1e-6) * scale) t = a / s;
    t = fmin(fmax(t, Real(0)), Real(1));
    return fmin(t / (Real(1) + t), cap);
}

__global__ void kScaleByBeta(std::size_t n, const Real* a, const Real* b,
                             Real cap, Real* out)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= n) return;
    out[m] = betaOf(a[m], b[m], cap) * b[m];
}

} // namespace

struct NscbcModalAbc::Impl
{
    GpuGrid g{};
    Config c{};
    std::size_t n = 0; // face cells: ny * nz
    DeviceBuffer<Real> a, b, out;
    DeviceBuffer<hipfftComplex> cbuf; // Modal only
    hipfftHandle plan{};              // Modal only, length ny, batch nz

    ~Impl()
    {
        if (plan) (void)hipfftDestroy(plan);
    }

    // One batched DCT-II (forward) / DCT-III (inverse) over the j axis; src and
    // dst may alias (both passes stage through cbuf).
    void dctFwd(const Real* src, Real* dst)
    {
        kDctPermute<<<launchBlocks(n), blockSize>>>(n, g.ny, src, cbuf.data());
        checkLastKernel("nscbcDctPermute");
        checkFft(hipfftExecC2C(plan, cbuf.data(), cbuf.data(), HIPFFT_FORWARD),
                 "ExecC2C nscbc forward");
        kDctTwiddle1d<<<launchBlocks(n), blockSize>>>(n, g.ny, cbuf.data(), dst);
        checkLastKernel("nscbcDctTwiddle");
    }

    void dctInv(const Real* src, Real* dst)
    {
        kDctInvTwiddle<<<launchBlocks(n), blockSize>>>(n, g.ny, src, cbuf.data());
        checkLastKernel("nscbcDctInvTwiddle");
        checkFft(hipfftExecC2C(plan, cbuf.data(), cbuf.data(), HIPFFT_BACKWARD),
                 "ExecC2C nscbc backward");
        kDctInvPermute1d<<<launchBlocks(n), blockSize>>>(n, g.ny, cbuf.data(),
                                                         dst);
        checkLastKernel("nscbcDctInvPermute");
    }
};

std::expected<NscbcModalAbc, hipError_t> NscbcModalAbc::make(const GpuGrid& g,
                                                             Config c)
{
    if (g.nx < 2 || g.ny < 2 || !(c.gamma > 1) || !(c.beta >= 0))
        return std::unexpected(hipErrorInvalidValue);

    auto impl = std::make_unique<Impl>();
    impl->g = g;
    impl->c = c;
    impl->n = std::size_t(g.ny) * g.nz;

    for (DeviceBuffer<Real>* buf : {&impl->a, &impl->b, &impl->out})
    {
        auto m = DeviceBuffer<Real>::make(impl->n);
        if (!m) return std::unexpected(m.error());
        *buf = std::move(*m);
    }
    {
        auto cb = DeviceBuffer<hipfftComplex>::make(impl->n);
        if (!cb) return std::unexpected(cb.error());
        impl->cbuf = std::move(*cb);
        if (hipfftPlan1d(&impl->plan, g.ny, HIPFFT_C2C, g.nz) != HIPFFT_SUCCESS)
            return std::unexpected(hipErrorUnknown);
    }
    return NscbcModalAbc(std::move(impl));
}

NscbcModalAbc::NscbcModalAbc(std::unique_ptr<Impl> impl) : impl_(std::move(impl))
{
}
NscbcModalAbc::NscbcModalAbc(NscbcModalAbc&&) noexcept = default;
NscbcModalAbc& NscbcModalAbc::operator=(NscbcModalAbc&&) noexcept = default;
NscbcModalAbc::~NscbcModalAbc() = default;

void NscbcModalAbc::compute(const Real* rho, const Real* u, const Real* v,
                            const Real* w, const Real* p)
{
    Impl& s = *impl_;
    kFacePlanes<<<launchBlocks(s.n), blockSize>>>(s.g, Real(s.c.gamma), rho, u, v,
                                                  w, p, s.a.data(), s.b.data());
    checkLastKernel("nscbcFacePlanes");

    const Real cap = Real(s.c.beta);
    // Both planes into the cosine basis, beta per mode there, back.
    // A constant beta would come back through this chain as beta * b exactly
    // (DCT-III inverts DCT-II), so the transform adds no bias of its own.
    s.dctFwd(s.a.data(), s.a.data());
    s.dctFwd(s.b.data(), s.b.data());
    kScaleByBeta<<<launchBlocks(s.n), blockSize>>>(s.n, s.a.data(), s.b.data(),
                                                   cap, s.out.data());
    checkLastKernel("nscbcBetaModal");
    s.dctInv(s.out.data(), s.out.data());
}

const Real* NscbcModalAbc::plane() const { return impl_->out.data(); }

std::size_t NscbcModalAbc::bytes() const
{
    const Impl& s = *impl_;
    std::size_t n = 3 * s.n * sizeof(Real) + s.n * sizeof(hipfftComplex);
    // Measured 2026-07-29: rocFFT answers 0 for these plans -- it does not
    // expose its internal scratch through this call -- so what this returns is
    // the buffers this module owns, not the last byte on the card. Asked
    // anyway, because the CUDA backend does answer.
    std::size_t work = 0;
    if (s.plan && hipfftGetSize(s.plan, &work) == HIPFFT_SUCCESS) n += work;
    return n;
}

} // namespace dns::gpu
