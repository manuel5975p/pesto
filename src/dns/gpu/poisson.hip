#include "dns/gpu/poisson.hpp"

#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/dct.hpp"

#include <hipfft/hipfft.h>

#include <cassert>
#include <cstdio>
#include <cmath>
#include <numbers>
#include <vector>

namespace dns::gpu {

static_assert(std::is_same_v<Real, float>, "hipFFT plans below are float");

namespace {

// hipfftExec* statuses used to be [[maybe_unused]] + assert, i.e. DISCARDED in
// the Release build this project ships. solve() is void-returning (it sits
// under the PoissonSolver interface), so the status cannot be propagated
// without changing that interface; report it loudly instead of dropping it.
void checkFft(hipfftResult r, const char* what)
{
    if (r != HIPFFT_SUCCESS)
        std::fprintf(stderr, "dns: hipfft %s failed (status %d)\n", what, int(r));
}


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
// Per-axis DCT-II of the interior box, Makhoul same-length-FFT kernels shared
// with the NSCBC face transform (gpu/internal/dct.hpp). Each pass transforms
// the fastest axis of the current layout and writes its output with the axes
// rotated (n1 becomes fastest); three passes restore the original layout, so
// one plan set covers the whole 3D transform.

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

// -------------------------- periodic x/z, wall y ---------------------------
//
// The wall-bounded box: periodic in x and z, homogeneous Neumann on the two y
// wall faces. FFT over x and z diagonalises those two directions, leaving one
// INDEPENDENT tridiagonal system per (kx, kz) mode along y:
//
//   p[j-1]/hy^2 + (lamx + lamz - 2/hy^2) p[j] + p[j+1]/hy^2 = rhs[j]
//
// with the mirror condition folding the wall neighbour onto the first interior
// row, i.e. the end diagonals lose one 1/hy^2. Solved exactly by Thomas, one
// thread per mode. This is deliberately NOT the DCT route: y needs no
// transform at all, so the solve stays exact for any ny and the delicate
// Makhoul machinery above is not extended to a mixed-transform case.

__global__ void kPackYInterior(GpuGrid g, int my, const Real* src,
                               hipfftComplex* dst)
{
    const std::size_t total = std::size_t(g.nx) * my * g.nz;
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= total) return;
    const int i = int(m % g.nx);
    const int j = int((m / g.nx) % my);
    const int k = int(m / (std::size_t(g.nx) * my));
    dst[m] = hipfftComplex{src[gidx(g, i, j + 1, k)], Real(0)};
}

__global__ void kUnpackYInterior(GpuGrid g, int my, Real norm,
                                 const hipfftComplex* src, Real* dst)
{
    const std::size_t m = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (m >= g.size()) return;
    int i, j, k;
    decompose(g, m, i, j, k);
    if (j < 1 || j > my)
    {
        dst[m] = Real(0); // ghost rows belong to the wall mirror, not here
        return;
    }
    const std::size_t q =
        std::size_t(i) + std::size_t(g.nx) * (std::size_t(j - 1) + std::size_t(my) * k);
    dst[m] = src[q].x * norm;
}

// One thread per (kx, kz) mode; sweeps y serially. `cp` is per-mode Thomas
// scratch of my reals. The (0,0) mode is the pure-Neumann singular one: its
// solution is defined only up to a constant, so the first row is pinned to
// zero and the constant is removed afterwards by subtracting the mode's mean.
__global__ void kTridiagY(int nx, int my, int nz, Real ihy2, const Real* lamx,
                          const Real* lamz, hipfftComplex* q, Real* cp)
{
    const std::size_t modes = std::size_t(nx) * nz;
    const std::size_t t = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= modes) return;
    const int kx = int(t % nx), kz = int(t / nx);
    const Real lam = lamx[kx] + lamz[kz];
    const bool singular = kx == 0 && kz == 0;

    auto at = [&](int j) {
        return std::size_t(kx)
             + std::size_t(nx) * (std::size_t(j) + std::size_t(my) * kz);
    };
    auto diag = [&](int j) {
        // Mirror at both walls removes one off-diagonal coupling from the end
        // rows: b = lam - 1/hy^2 there, lam - 2/hy^2 in the interior.
        const Real ends = (j == 0 ? Real(1) : Real(0))
                        + (j == my - 1 ? Real(1) : Real(0));
        return lam - (Real(2) - ends) * ihy2;
    };

    // Forward elimination.
    Real b0 = singular ? Real(1) : diag(0);
    Real c0 = singular ? Real(0) : ihy2;
    hipfftComplex d0 = singular ? hipfftComplex{Real(0), Real(0)} : q[at(0)];
    cp[t * my] = c0 / b0;
    q[at(0)] = hipfftComplex{d0.x / b0, d0.y / b0};
    for (int j = 1; j < my; ++j)
    {
        const Real a = ihy2, c = j == my - 1 ? Real(0) : ihy2;
        const Real m = diag(j) - a * cp[t * my + (j - 1)];
        cp[t * my + j] = c / m;
        const hipfftComplex prev = q[at(j - 1)];
        const hipfftComplex cur = q[at(j)];
        q[at(j)] = hipfftComplex{(cur.x - a * prev.x) / m,
                                 (cur.y - a * prev.y) / m};
    }
    // Back substitution.
    for (int j = my - 2; j >= 0; --j)
    {
        const hipfftComplex nxt = q[at(j + 1)];
        const Real f = cp[t * my + j];
        hipfftComplex cur = q[at(j)];
        cur.x -= f * nxt.x;
        cur.y -= f * nxt.y;
        q[at(j)] = cur;
    }
    if (!singular) return;
    // Restore the zero-mean gauge the pin destroyed.
    Real sx = 0, sy = 0;
    for (int j = 0; j < my; ++j) { sx += q[at(j)].x; sy += q[at(j)].y; }
    sx /= Real(my);
    sy /= Real(my);
    for (int j = 0; j < my; ++j)
    {
        hipfftComplex cur = q[at(j)];
        cur.x -= sx;
        cur.y -= sy;
        q[at(j)] = cur;
    }
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

    // PeriodicXZWallY mode: batched 1D C2C plans along x and along z over the
    // y-interior slab, plus the Thomas scratch.
    hipfftHandle planX{}, planZ{};
    DeviceBuffer<Real> thomas;

    ~Impl()
    {
        if (fwd) (void)hipfftDestroy(fwd);
        if (bwd) (void)hipfftDestroy(bwd);
        for (auto p : plan)
            if (p) (void)hipfftDestroy(p);
        if (planX) (void)hipfftDestroy(planX);
        if (planZ) (void)hipfftDestroy(planZ);
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

    if (bc == PoissonBc::PeriodicXZWallY)
    {
        // Only y carries ghosts here: x and z are periodic over their full
        // extent, so their eigenvalues are the full-length periodic ones and
        // the C2C transforms run over every cell.
        assert(g.ny >= 4);
        impl->mx = g.nx;
        impl->my = g.ny - 2;
        impl->mz = g.nz;
        const std::size_t total = std::size_t(g.nx) * impl->my * g.nz;

        auto lx = periodicEigenvalues(g.nx, g.nx, g.hx);
        if (!lx) return std::unexpected(lx.error());
        auto lz = periodicEigenvalues(g.nz, g.nz, g.hz);
        if (!lz) return std::unexpected(lz.error());
        impl->lamx = std::move(*lx);
        impl->lamz = std::move(*lz);

        auto c = DeviceBuffer<hipfftComplex>::make(total);
        if (!c) return std::unexpected(c.error());
        impl->cbuf = std::move(*c);
        auto th = DeviceBuffer<Real>::make(total);
        if (!th) return std::unexpected(th.error());
        impl->thomas = std::move(*th);

        // x is contiguous: stride 1, one transform per (j,k). z is the slowest
        // axis: stride nx*my, and its batch index runs over the contiguous
        // (i,j) pair, so dist 1.
        //
        // inembed/onembed MUST be non-null here. hipFFT (like cuFFT) ignores
        // EVERY advanced-layout parameter when they are null and falls back to
        // the packed contiguous layout -- so a null-embed z plan silently
        // transforms the wrong axis instead of failing, which is exactly what
        // it did: 2D stayed green (no z pass at all) while every 3D case came
        // back wrong.
        int nX = g.nx, nZ = g.nz;
        if (hipfftPlanMany(&impl->planX, 1, &nX, &nX, 1, g.nx, &nX, 1, g.nx,
                           HIPFFT_C2C, impl->my * g.nz)
            != HIPFFT_SUCCESS)
            return std::unexpected(hipErrorUnknown);
        if (g.nz > 1
            && hipfftPlanMany(&impl->planZ, 1, &nZ, &nZ, g.nx * impl->my, 1, &nZ,
                              g.nx * impl->my, 1, HIPFFT_C2C, g.nx * impl->my)
                   != HIPFFT_SUCCESS)
            return std::unexpected(hipErrorUnknown);
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
        auto rf = hipfftExecR2C(s.fwd, s.work.data(), s.spec.data());
        checkFft(rf, "ExecR2C");

        const std::size_t nc = std::size_t(s.nxh) * s.g.ny * s.g.nz;
        const Real norm = Real(1.0 / double(s.g.size()));
        kPoissonScale<<<launchBlocks(nc), blockSize>>>(s.nxh, s.g.ny, s.g.nz,
                                                       s.lamx.data(), s.lamy.data(),
                                                       s.lamz.data(), norm,
                                                       s.spec.data());
        checkLastKernel("poissonScale");

        auto rb = hipfftExecC2R(s.bwd, s.spec.data(), p.data());
        checkFft(rb, "ExecC2R");
        return;
    }

    if (s.bc == PoissonBc::PeriodicXZWallY)
    {
        const std::size_t total = std::size_t(s.g.nx) * s.my * s.g.nz;
        kPackYInterior<<<launchBlocks(total), blockSize>>>(s.g, s.my, rhs.data(),
                                                           s.cbuf.data());
        checkLastKernel("packYInterior");

        checkFft(hipfftExecC2C(s.planX, s.cbuf.data(), s.cbuf.data(),
                               HIPFFT_FORWARD), "ExecC2C x forward");
        if (s.g.nz > 1)
            checkFft(hipfftExecC2C(s.planZ, s.cbuf.data(), s.cbuf.data(),
                                   HIPFFT_FORWARD), "ExecC2C z forward");

        const Real ihy2 = Real(1.0 / (s.g.hy * s.g.hy));
        kTridiagY<<<launchBlocks(std::size_t(s.g.nx) * s.g.nz), blockSize>>>(
            s.g.nx, s.my, s.g.nz, ihy2, s.lamx.data(), s.lamz.data(),
            s.cbuf.data(), s.thomas.data());
        checkLastKernel("tridiagY");

        if (s.g.nz > 1)
            checkFft(hipfftExecC2C(s.planZ, s.cbuf.data(), s.cbuf.data(),
                                   HIPFFT_BACKWARD), "ExecC2C z backward");
        checkFft(hipfftExecC2C(s.planX, s.cbuf.data(), s.cbuf.data(),
                               HIPFFT_BACKWARD), "ExecC2C x backward");

        // hipFFT's inverse is unnormalised on both transformed axes.
        const Real norm = Real(1.0 / (double(s.g.nx) * s.g.nz));
        kUnpackYInterior<<<launchBlocks(s.g.size()), blockSize>>>(
            s.g, s.my, norm, s.cbuf.data(), p.data());
        checkLastKernel("unpackYInterior");
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
        auto r =
            hipfftExecC2C(s.plan[a], s.cbuf.data(), s.cbuf.data(), HIPFFT_FORWARD);
        checkFft(r, "ExecC2C forward");
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
        auto r =
            hipfftExecC2C(s.plan[a], s.cbuf.data(), s.cbuf.data(), HIPFFT_BACKWARD);
        checkFft(r, "ExecC2C backward");
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
