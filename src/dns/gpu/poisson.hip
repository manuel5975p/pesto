#include "dns/gpu/poisson.hpp"

#include "dns/gpu/device_util.hpp"

#include <hipfft/hipfft.h>

#include <cassert>
#include <cmath>
#include <numbers>
#include <vector>

namespace dns::gpu {

static_assert(std::is_same_v<Real, float>, "hipFFT plans below are float");

namespace {

// ------------------------------- periodic ---------------------------------

// Multiply each mode by norm/lambda (modified-wavenumber FD eigenvalue),
// zero the DC mode. lam* are per-axis eigenvalue arrays.
__global__ void kPoissonScale(int nxh, int ny, int nz, const Real* lamx,
                              const Real* lamy, const Real* lamz, Real norm,
                              hipfftComplex* c)
{
    const std::size_t nc = std::size_t(nxh) * ny * nz;
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= nc) return;
    const int kx = int(m % nxh);
    const int ky = int((m / nxh) % ny);
    const int kz = int(m / (std::size_t(nxh) * ny));
    const Real lam = lamx[kx] + lamy[ky] + lamz[kz];
    const Real s = lam == Real(0) ? Real(0) : norm / lam;
    c[m].x *= s;
    c[m].y *= s;
}

std::expected<DeviceBuffer<Real>, hipError_t> periodicEigenvalues(int n, int count,
                                                                  double h)
{
    using std::numbers::pi;
    std::vector<Real> lam(count);
    for (int m = 0; m < count; ++m)
        lam[m] = Real((2.0 * std::cos(2.0 * pi * m / n) - 2.0) / (h * h));
    auto buf = DeviceBuffer<Real>::make(count);
    if (!buf) return buf;
    if (auto e = buf->upload(lam.data()); e != hipSuccess) return std::unexpected(e);
    return buf;
}

// ------------------------------ Neumann box --------------------------------
//
// Per-axis DCT-II of the interior box via Makhoul's same-length FFT:
//   v[j]        = x[2j]            j <  ceil(n/2)
//   v[n-1-j]    = x[2j+1]
//   C_k         = Re( e^{-i pi k / (2n)} FFT_n(v)_k )
// and its inverse (DCT-III):
//   V_k = e^{i pi k / (2n)} (C_k - i C_{n-k}),  C_n := 0
//   x[permute(j)] = Re( IFFT_n(V)_j )
// Each pass transforms the fastest axis of the current layout and writes its
// output with the axes rotated (n1 becomes fastest); three passes restore the
// original layout, so one plan set covers the whole 3D transform. The
// unnormalized hipFFT inverse is compensated by 1/n in the inverse twiddle.

__device__ inline int evenOdd(int j, int n)
{
    return j < (n + 1) / 2 ? 2 * j : 2 * (n - 1 - j) + 1;
}

__global__ void kDctPermute(std::size_t total, int n0, const Real* x,
                            hipfftComplex* v)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int j = int(m % n0);
    const std::size_t b = m / n0;
    v[m] = hipfftComplex{x[b * n0 + evenOdd(j, n0)], Real(0)};
}

// The rotation is a (k, i1) transpose per i2 slab; stage a 32x32 tile through
// LDS so both the read (k-fastest) and the write (i1-fastest) coalesce.
constexpr int dctTile = 32;
constexpr int dctRows = 8;

__global__ void kDctTwiddleRot(int n0, int n1, int n2, const hipfftComplex* V,
                               Real* out)
{
    __shared__ Real tile[dctTile][dctTile + 1];
    const int k0 = blockIdx.x * dctTile, i10 = blockIdx.y * dctTile;
    const int i2 = blockIdx.z;
    const int tx = threadIdx.x;
    for (int r = threadIdx.y; r < dctTile; r += dctRows)
    {
        const int k = k0 + tx, i1 = i10 + r;
        if (k < n0 && i1 < n1)
        {
            const std::size_t m = std::size_t(k)
                + std::size_t(n0) * (i1 + std::size_t(n1) * i2);
            const Real th = Real(std::numbers::pi) * k / Real(2 * n0);
            tile[r][tx] = std::cos(th) * V[m].x + std::sin(th) * V[m].y;
        }
    }
    __syncthreads();
    for (int r = threadIdx.y; r < dctTile; r += dctRows)
    {
        const int i1 = i10 + tx, k = k0 + r;
        if (k < n0 && i1 < n1)
            out[std::size_t(i1) + std::size_t(n1) * (i2 + std::size_t(n2) * k)]
                = tile[tx][r];
    }
}

__global__ void kDctInvTwiddle(std::size_t total, int n0, const Real* C,
                               hipfftComplex* V)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int k = int(m % n0);
    const std::size_t b = m / n0;
    const Real ck = C[b * n0 + k];
    const Real cnk = k == 0 ? Real(0) : C[b * n0 + (n0 - k)];
    const Real th = Real(std::numbers::pi) * k / Real(2 * n0);
    const Real inv = Real(1) / Real(n0);
    V[m] = hipfftComplex{(std::cos(th) * ck + std::sin(th) * cnk) * inv,
                         (std::sin(th) * ck - std::cos(th) * cnk) * inv};
}

// Same tiled (j, i1) transpose; evenOdd only relabels the output k-plane, the
// i1-contiguous writes stay coalesced.
__global__ void kDctInvPermuteRot(int n0, int n1, int n2, const hipfftComplex* v,
                                  Real* out)
{
    __shared__ Real tile[dctTile][dctTile + 1];
    const int j0 = blockIdx.x * dctTile, i10 = blockIdx.y * dctTile;
    const int i2 = blockIdx.z;
    const int tx = threadIdx.x;
    for (int r = threadIdx.y; r < dctTile; r += dctRows)
    {
        const int j = j0 + tx, i1 = i10 + r;
        if (j < n0 && i1 < n1)
            tile[r][tx]
                = v[std::size_t(j) + std::size_t(n0) * (i1 + std::size_t(n1) * i2)].x;
    }
    __syncthreads();
    for (int r = threadIdx.y; r < dctTile; r += dctRows)
    {
        const int i1 = i10 + tx, j = j0 + r;
        if (j < n0 && i1 < n1)
            out[std::size_t(i1)
                + std::size_t(n1) * (i2 + std::size_t(n2) * evenOdd(j, n0))]
                = tile[tx][r];
    }
}

// q_k *= 1/lam (zero mode -> 0) on the packed interior spectrum.
__global__ void kNeumannScale(std::size_t total, int mx, int my, const Real* lamx,
                              const Real* lamy, const Real* lamz, Real* q)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int kx = int(m % mx);
    const int ky = int((m / mx) % my);
    const int kz = int(m / (std::size_t(mx) * my));
    const Real lam = lamx[kx] + lamy[ky] + lamz[kz];
    q[m] = lam == Real(0) ? Real(0) : q[m] / lam;
}

// Ghost offset per axis: a degenerate axis (m == n, i.e. n == 1) has no ring.
__host__ __device__ inline int ghostOff(int m, int n) { return m == n ? 0 : 1; }

__global__ void kPackInterior(GpuGrid g, int mx, int my, int mz, const Real* src,
                              Real* dst)
{
    const std::size_t total = std::size_t(mx) * my * mz;
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int a = int(m % mx);
    const int b = int((m / mx) % my);
    const int c = int(m / (std::size_t(mx) * my));
    dst[m] = src[gidx(g, a + ghostOff(mx, g.nx), b + ghostOff(my, g.ny),
                      c + ghostOff(mz, g.nz))];
}

__global__ void kUnpackInterior(GpuGrid g, int mx, int my, int mz, const Real* src,
                                Real* dst)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= g.size()) return;
    int i, j, k;
    decompose(g, m, i, j, k);
    const int ox = ghostOff(mx, g.nx), oy = ghostOff(my, g.ny), oz = ghostOff(mz, g.nz);
    const bool in = i >= ox && i < ox + mx && j >= oy && j < oy + my && k >= oz
        && k < oz + mz;
    dst[m] = in ? src[std::size_t(i - ox)
                      + std::size_t(mx) * (std::size_t(j - oy) + std::size_t(my) * (k - oz))]
                : Real(0);
}

inline dim3 rotGrid(int n0, int n1, int n2)
{
    return dim3(unsigned((n0 + dctTile - 1) / dctTile),
                unsigned((n1 + dctTile - 1) / dctTile), unsigned(n2));
}
constexpr dim3 rotBlock{dctTile, dctRows, 1};

std::expected<DeviceBuffer<Real>, hipError_t> neumannEigenvalues(int m, double h)
{
    using std::numbers::pi;
    std::vector<Real> lam(m);
    for (int k = 0; k < m; ++k)
        lam[k] = Real((2.0 * std::cos(pi * k / m) - 2.0) / (h * h));
    auto buf = DeviceBuffer<Real>::make(m);
    if (!buf) return buf;
    if (auto e = buf->upload(lam.data()); e != hipSuccess) return std::unexpected(e);
    return buf;
}

} // namespace

struct DevicePoissonSolver::Impl
{
    GpuGrid g;
    PoissonBc bc = PoissonBc::Periodic;

    // Periodic mode.
    int nxh = 0;
    hipfftHandle fwd{}, bwd{};
    DeviceBuffer<Real> lamx, lamy, lamz, work;
    DeviceBuffer<hipfftComplex> spec;

    // NeumannBox mode: interior extents and the three per-axis batched plans
    // (each transforms the fastest axis of the rotating layout).
    int mx = 0, my = 0, mz = 0;
    hipfftHandle plan[3]{};
    DeviceBuffer<Real> packA, packB;
    DeviceBuffer<hipfftComplex> cbuf;

    ~Impl()
    {
        if (fwd) (void)hipfftDestroy(fwd);
        if (bwd) (void)hipfftDestroy(bwd);
        for (auto p : plan)
            if (p) (void)hipfftDestroy(p);
    }
};

std::expected<DevicePoissonSolver, hipError_t>
DevicePoissonSolver::make(const GpuGrid& g, PoissonBc bc)
{
    auto impl = std::make_unique<Impl>();
    impl->g = g;
    impl->bc = bc;

    assert(g.nx > 1 && g.ny > 1); // only nz may be degenerate (2D mode)

    if (bc == PoissonBc::Periodic)
    {
        impl->nxh = g.nx / 2 + 1;
        auto lx = periodicEigenvalues(g.nx, impl->nxh, g.hx);
        if (!lx) return std::unexpected(lx.error());
        auto ly = periodicEigenvalues(g.ny, g.ny, g.hy);
        if (!ly) return std::unexpected(ly.error());
        auto lz = periodicEigenvalues(g.nz, g.nz, g.hz);
        if (!lz) return std::unexpected(lz.error());
        impl->lamx = std::move(*lx);
        impl->lamy = std::move(*ly);
        impl->lamz = std::move(*lz);

        auto work = DeviceBuffer<Real>::make(g.size());
        if (!work) return std::unexpected(work.error());
        impl->work = std::move(*work);
        auto spec =
            DeviceBuffer<hipfftComplex>::make(std::size_t(impl->nxh) * g.ny * g.nz);
        if (!spec) return std::unexpected(spec.error());
        impl->spec = std::move(*spec);

        // storage is i-fastest => row-major dims (nz, ny, nx), same as the CPU
        // plan; nz == 1 (2D) uses a 2D plan (length-1 FFT dims are fragile).
        const bool ok = g.nz == 1
            ? hipfftPlan2d(&impl->fwd, g.ny, g.nx, HIPFFT_R2C) == HIPFFT_SUCCESS
              && hipfftPlan2d(&impl->bwd, g.ny, g.nx, HIPFFT_C2R) == HIPFFT_SUCCESS
            : hipfftPlan3d(&impl->fwd, g.nz, g.ny, g.nx, HIPFFT_R2C) == HIPFFT_SUCCESS
              && hipfftPlan3d(&impl->bwd, g.nz, g.ny, g.nx, HIPFFT_C2R) == HIPFFT_SUCCESS;
        if (!ok) return std::unexpected(hipErrorUnknown);
        return DevicePoissonSolver(std::move(impl));
    }

    // NeumannBox: one-cell ghost ring on every non-degenerate axis. A
    // degenerate axis (n == 1, the 2D case) carries no ghosts, contributes a
    // zero eigenvalue, and its DCT pass is skipped below.
    assert(g.nx >= 4 && g.ny >= 4 && (g.nz == 1 || g.nz >= 4));
    impl->mx = g.nx - 2;
    impl->my = g.ny - 2;
    impl->mz = g.nz == 1 ? 1 : g.nz - 2;
    const std::size_t total = std::size_t(impl->mx) * impl->my * impl->mz;

    auto lx = neumannEigenvalues(impl->mx, g.hx);
    if (!lx) return std::unexpected(lx.error());
    auto ly = neumannEigenvalues(impl->my, g.hy);
    if (!ly) return std::unexpected(ly.error());
    auto lz = neumannEigenvalues(impl->mz, g.hz);
    if (!lz) return std::unexpected(lz.error());
    impl->lamx = std::move(*lx);
    impl->lamy = std::move(*ly);
    impl->lamz = std::move(*lz);

    auto a = DeviceBuffer<Real>::make(total);
    if (!a) return std::unexpected(a.error());
    impl->packA = std::move(*a);
    auto b = DeviceBuffer<Real>::make(total);
    if (!b) return std::unexpected(b.error());
    impl->packB = std::move(*b);
    auto c = DeviceBuffer<hipfftComplex>::make(total);
    if (!c) return std::unexpected(c.error());
    impl->cbuf = std::move(*c);

    const int n[3] = {impl->mx, impl->my, impl->mz};
    for (int p = 0; p < 3; ++p)
        if (n[p] > 1
            && hipfftPlan1d(&impl->plan[p], n[p], HIPFFT_C2C, int(total / n[p]))
                   != HIPFFT_SUCCESS)
            return std::unexpected(hipErrorUnknown);
    return DevicePoissonSolver(std::move(impl));
}

DevicePoissonSolver::DevicePoissonSolver(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DevicePoissonSolver::DevicePoissonSolver(DevicePoissonSolver&&) noexcept = default;
DevicePoissonSolver& DevicePoissonSolver::operator=(DevicePoissonSolver&&) noexcept
    = default;
DevicePoissonSolver::~DevicePoissonSolver() = default;

void DevicePoissonSolver::solve(const DeviceScalar& rhs, DeviceScalar& p)
{
    Impl& s = *impl_;
    assert(rhs.size() == s.g.size() && p.size() == s.g.size());

    if (s.bc == PoissonBc::Periodic)
    {
        // R2C may clobber its input: stage rhs into the work buffer.
        (void)hipMemcpy(s.work.data(), rhs.data(), s.g.size() * sizeof(Real),
                        hipMemcpyDeviceToDevice);
        [[maybe_unused]] auto rf = hipfftExecR2C(s.fwd, s.work.data(), s.spec.data());
        assert(rf == HIPFFT_SUCCESS);

        const std::size_t nc = std::size_t(s.nxh) * s.g.ny * s.g.nz;
        const Real norm = Real(1.0 / double(s.g.size()));
        kPoissonScale<<<launchBlocks(nc), blockSize>>>(s.nxh, s.g.ny, s.g.nz,
                                                       s.lamx.data(), s.lamy.data(),
                                                       s.lamz.data(), norm,
                                                       s.spec.data());
        checkLastKernel("poissonScale");

        [[maybe_unused]] auto rb = hipfftExecC2R(s.bwd, s.spec.data(), p.data());
        assert(rb == HIPFFT_SUCCESS);
        return;
    }

    const std::size_t total = std::size_t(s.mx) * s.my * s.mz;
    const int dims[3] = {s.mx, s.my, s.mz};
    Real* cur = s.packA.data();
    Real* nxt = s.packB.data();

    kPackInterior<<<launchBlocks(total), blockSize>>>(s.g, s.mx, s.my, s.mz,
                                                      rhs.data(), cur);
    checkLastKernel("packInterior");

    // Forward: three DCT-II passes, each rotating the layout (x,y,z fastest in
    // turn). A size-1 axis is skipped whole: its DCT is the identity and the
    // rotation is a memory no-op ((1,n1,n2) flattened == (n1,n2,1) flattened).
    for (int a = 0; a < 3; ++a)
    {
        const int n0 = dims[a], n1 = dims[(a + 1) % 3], n2 = dims[(a + 2) % 3];
        if (n0 == 1) continue;
        kDctPermute<<<launchBlocks(total), blockSize>>>(total, n0, cur, s.cbuf.data());
        checkLastKernel("dctPermute");
        [[maybe_unused]] auto r =
            hipfftExecC2C(s.plan[a], s.cbuf.data(), s.cbuf.data(), HIPFFT_FORWARD);
        assert(r == HIPFFT_SUCCESS);
        kDctTwiddleRot<<<rotGrid(n0, n1, n2), rotBlock>>>(n0, n1, n2,
                                                          s.cbuf.data(), nxt);
        checkLastKernel("dctTwiddleRot");
        std::swap(cur, nxt);
    }

    kNeumannScale<<<launchBlocks(total), blockSize>>>(total, s.mx, s.my,
                                                      s.lamx.data(), s.lamy.data(),
                                                      s.lamz.data(), cur);
    checkLastKernel("neumannScale");

    // Inverse: three DCT-III passes with the same rotation cycle (size-1 axes
    // skipped as above).
    for (int a = 0; a < 3; ++a)
    {
        const int n0 = dims[a], n1 = dims[(a + 1) % 3], n2 = dims[(a + 2) % 3];
        if (n0 == 1) continue;
        kDctInvTwiddle<<<launchBlocks(total), blockSize>>>(total, n0, cur,
                                                           s.cbuf.data());
        checkLastKernel("dctInvTwiddle");
        [[maybe_unused]] auto r =
            hipfftExecC2C(s.plan[a], s.cbuf.data(), s.cbuf.data(), HIPFFT_BACKWARD);
        assert(r == HIPFFT_SUCCESS);
        kDctInvPermuteRot<<<rotGrid(n0, n1, n2), rotBlock>>>(n0, n1, n2,
                                                             s.cbuf.data(), nxt);
        checkLastKernel("dctInvPermuteRot");
        std::swap(cur, nxt);
    }

    kUnpackInterior<<<launchBlocks(s.g.size()), blockSize>>>(s.g, s.mx, s.my, s.mz,
                                                             cur, p.data());
    checkLastKernel("unpackInterior");
}

} // namespace dns::gpu
