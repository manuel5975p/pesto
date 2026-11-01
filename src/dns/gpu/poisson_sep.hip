// SeparablePoisson (gpu/poisson.hpp): per-axis real transforms from
// dns/septransform.hpp, one batched complex hipFFT per axis and direction.
//
// Pass structure (the NeumannBox chain of poisson.hip, generalised): each pass
// transforms the FASTEST axis of the current layout (lines x n0, line b =
// i1 + n1 i2) and writes it transposed (n0 x lines), i.e. with the axes
// rotated, so three passes restore x-fastest. A degenerate z (2D) is skipped
// whole: the rotation is then a memory no-op. Forward x, y, z (the spectral
// scale fused into the last post), inverse x, y, z (the last post writes p).
// One real and one complex staging buffer.
//
// Two real lines share one complex FFT (dns::SepPack) except on a DCT-IV axis.
// Lines pair along i1 (y on the x pass, x on the z pass) or i2 (x on the y
// pass) -- NEVER along z, the axis a 2D case lacks: a z-extruded box then pairs
// exactly the 2D lines, and the solve equals the 2D one per plane bit for bit
// (gpu_poisson_sep (3)).
#include "dns/gpu/poisson.hpp"

#include "dns/gpu/device_util.hpp"

#include <hipfft/hipfft.h>

#include <algorithm>
#include <cstdio>
#include <vector>

namespace dns::gpu {

static_assert(std::is_same_v<Real, float>, "hipFFT plans below are float");

namespace {

using C32 = SepC<float>;

__device__ inline C32 toC(hipfftComplex v) { return {v.x, v.y}; }

// One pass's geometry. Indices are 32-bit: make() caps the box at 2^31 cells.
struct PassGeom
{
    unsigned n0, L, n1, lines; // line length, FFT length, i1 extent, n1 n2
    unsigned np, hp;           // paired extent, ceil(np / 2)
    unsigned clines;           // complex lines (= FFT batch)
    bool overI1;               // pair along i1 (else i2)
    SepKind kind;
    bool inverse;
};

__device__ inline unsigned lineOf(const PassGeom& g, unsigned p, unsigned o)
{
    return g.overI1 ? p + g.n1 * o : o + g.n1 * p;
}

// Pre-FFT step: complex element (Q, j) from its one or two real lines. The
// first forward pass reads the caller's rhs times scale, keep-masked.
template <SepPack M>
__global__ void kSepPre(PassGeom g, const Real* in, const std::uint8_t* keep, Real scale,
                        const float2* twPre, const float2* twPost, hipfftComplex* v)
{
    const unsigned m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= g.clines * g.L) return;
    const unsigned j = m % g.L, Q = m / g.L;
    const C32 w = toC(twPre[j]), t = toC(twPost[j]);
    auto one = [&](unsigned b) {
        auto get = [&](int i) -> Real {
            const std::size_t c = std::size_t(b) * g.n0 + unsigned(i);
            if (keep && !keep[c]) return Real(0);
            return in[c] * scale;
        };
        return g.inverse ? sepInvPre<float>(g.kind, int(j), int(g.n0), get, w, t)
                         : sepFwdPre<float>(g.kind, int(j), int(g.n0), get, w);
    };
    C32 r;
    if constexpr (M == SepPack::Single)
        r = one(Q);
    else
    {
        const unsigned p = Q % g.hp, o = Q / g.hp;
        const C32 a = one(lineOf(g, 2 * p, o));
        const C32 b = 2 * p + 1 < g.np ? one(lineOf(g, 2 * p + 1, o)) : C32{0.0f, 0.0f};
        r = M == SepPack::RealIn ? C32{a.re, b.re} : sepJoin(a, b);
    }
    v[m] = hipfftComplex{r.re, r.im};
}

constexpr int kTile = 32;
constexpr int kRows = 8;

// The spectral scale fused into the last forward post: q <- -q norm / lam,
// lam = lx + ly + lz of the output's dense index; the zero mode -> 0.
struct ScaleArgs
{
    const Real *lx = nullptr, *ly = nullptr, *lz = nullptr;
    Real norm = 1;
    unsigned nx = 1, ny = 1;
};

// Post-FFT step, transposed: the (m, b) tile through LDS so the read
// (m-fastest) and the write (b-fastest, out[b + lines d]) coalesce. One FFT
// output becomes one or two line entries (DCT-IV: 2m and n0-1-2m). keep (last
// inverse pass) indexes the output, then x-fastest.
template <SepPack M>
__global__ void kSepPost(PassGeom g, const hipfftComplex* V, const float2* twPost, Real* out,
                         const std::uint8_t* keep, ScaleArgs sc)
{
    __shared__ Real t0[kTile][kTile + 1];
    __shared__ Real t1[kTile][kTile + 1];
    __shared__ int d0[kTile], d1[kTile];
    const unsigned b0 = blockIdx.x * kTile, m0 = blockIdx.y * kTile;
    const unsigned tx = threadIdx.x;
    for (unsigned r = threadIdx.y; r < kTile; r += kRows)
    {
        const unsigned m = m0 + tx, b = b0 + r;
        if (m >= g.L || b >= g.lines) continue;
        C32 X;
        if constexpr (M == SepPack::Single)
            X = toC(V[b * g.L + m]);
        else
        {
            const unsigned i1 = b % g.n1, i2 = b / g.n1;
            const unsigned p = g.overI1 ? i1 : i2, o = g.overI1 ? i2 : i1;
            const unsigned row = (p / 2 + g.hp * o) * g.L;
            const C32 zk = toC(V[row + m]);
            if constexpr (M == SepPack::RealIn)
                X = sepSplit(zk, toC(V[row + (m ? g.L - m : 0u)]), (p & 1u) != 0);
            else
                X = C32{(p & 1u) ? zk.im : zk.re, 0.0f};
        }
        const C32 t = toC(twPost[m]);
        const SepOut<float> e = g.inverse ? sepInvPost<float>(g.kind, int(m), int(g.n0), X, t)
                                          : sepFwdPost<float>(g.kind, int(m), int(g.n0), X, t);
        t0[r][tx] = e.v0;
        t1[r][tx] = e.v1;
        if (r == 0)
        {
            d0[tx] = e.d0;
            d1[tx] = e.d1;
        }
    }
    __syncthreads();
    auto store = [&](unsigned o, Real val) {
        if (sc.lx)
        {
            const unsigned ix = o % sc.nx, iy = (o / sc.nx) % sc.ny, iz = o / (sc.nx * sc.ny);
            const Real lam = sc.lx[ix] + sc.ly[iy] + sc.lz[iz];
            val = lam == Real(0) ? Real(0) : -(val * sc.norm) / lam;
        }
        out[o] = keep && !keep[o] ? Real(0) : val;
    };
    for (unsigned r = threadIdx.y; r < kTile; r += kRows)
    {
        const unsigned b = b0 + tx, m = m0 + r;
        if (m >= g.L || b >= g.lines) continue;
        store(b + g.lines * unsigned(d0[r]), t0[tx][r]);
        if (d1[r] >= 0) store(b + g.lines * unsigned(d1[r]), t1[tx][r]);
    }
}

std::string fftText(const char* what, hipfftResult r)
{
    return std::string("SeparablePoisson: hipFFT ") + what + " failed (status " +
           std::to_string(int(r)) + ")";
}

} // namespace

struct SeparablePoisson::Impl
{
    SepBox box;
    int n[3] = {1, 1, 1};
    SepKind kind[3]{};
    PassGeom geom[3]{};
    std::size_t total = 0;
    Real norm = 1;
    hipError_t status = hipSuccess;
    hipfftHandle plan[3]{};
    DeviceBuffer<float2> twPre[3], twPost[3];
    DeviceBuffer<Real> lam[3];
    DeviceBuffer<Real> R;
    DeviceBuffer<hipfftComplex> C;

    ~Impl()
    {
        for (auto p : plan)
            if (p) (void)hipfftDestroy(p);
    }

    void record(hipError_t e)
    {
        if (e != hipSuccess && status == hipSuccess) status = e;
    }
};

std::expected<SeparablePoisson, std::string> SeparablePoisson::make(const SepBox& b)
{
    auto fail = [](const std::string& s) { return std::unexpected("SeparablePoisson: " + s); };
    if (b.nx < 2 || b.ny < 2) return fail("nx and ny must be >= 2");
    if (b.nz < 1) return fail("nz must be >= 1");
    if (b.nz == 1 && b.kz != SepKind::Periodic)
        return fail("nz == 1 (2D) needs a periodic z: the 2D restriction has no z stencil");
    if (!(b.hx > 0 && b.hy > 0 && b.hz > 0)) return fail("spacings must be > 0");
    if (b.size() > std::size_t(INT32_MAX)) return fail("more than 2^31 cells");

    auto impl = std::make_unique<Impl>();
    Impl& s = *impl;
    s.box = b;
    const int n[3] = {b.nx, b.ny, b.nz};
    const double h[3] = {b.hx, b.hy, b.hz};
    const SepKind k[3] = {b.kx, b.ky, b.kz};
    s.total = b.size();
    double norm = 1.0;
    std::size_t cmax = 1;
    for (int a = 0; a < 3; ++a)
    {
        s.n[a] = n[a];
        s.kind[a] = k[a];
        if (k[a] == SepKind::ND && n[a] % 2)
            return fail(std::string("ND on axis ") + "xyz"[a] + " needs an even cell count (got " +
                        std::to_string(n[a]) + ")");
        const int L = n[a] == 1 ? 1 : sepLineLen(k[a], n[a]);

        std::vector<Real> lam(std::size_t(n[a]), Real(0));
        if (n[a] > 1)
        {
            norm *= sepNorm(k[a], n[a]);
            for (int q = 0; q < n[a]; ++q) lam[std::size_t(q)] = Real(sepEigen(k[a], n[a], h[a], q));
        }
        auto dl = DeviceBuffer<Real>::make(lam.size());
        if (!dl || dl->upload(lam.data()) != hipSuccess) return fail("device eigenvalue table");
        s.lam[a] = std::move(*dl);

        std::vector<float2> pre(static_cast<std::size_t>(L)), post(static_cast<std::size_t>(L));
        for (int q = 0; q < L; ++q)
        {
            const auto w = sepTwPre(k[a], n[a], q);
            const auto t = sepTwPost(k[a], n[a], q);
            pre[std::size_t(q)] = float2{float(w.re), float(w.im)};
            post[std::size_t(q)] = float2{float(t.re), float(t.im)};
        }
        auto dp = DeviceBuffer<float2>::make(pre.size());
        if (!dp || dp->upload(pre.data()) != hipSuccess) return fail("device twiddle table");
        auto dq = DeviceBuffer<float2>::make(post.size());
        if (!dq || dq->upload(post.data()) != hipSuccess) return fail("device twiddle table");
        s.twPre[a] = std::move(*dp);
        s.twPost[a] = std::move(*dq);

        if (n[a] == 1) continue;
        // Pairing along i1 except on the y pass, where i1 is z (see the top).
        PassGeom& g = s.geom[a];
        const int n1 = n[(a + 1) % 3], n2 = n[(a + 2) % 3];
        g.n0 = unsigned(n[a]);
        g.L = unsigned(L);
        g.n1 = unsigned(n1);
        g.lines = unsigned(n1) * unsigned(n2);
        g.overI1 = a != 1;
        g.np = unsigned(g.overI1 ? n1 : n2);
        g.hp = (g.np + 1) / 2;
        const bool single = sepPack(k[a], false) == SepPack::Single;
        g.clines = single ? g.lines : g.hp * unsigned(g.overI1 ? n2 : n1);
        g.kind = k[a];
        cmax = std::max(cmax, std::size_t(g.clines) * g.L);
        const hipfftResult r = hipfftPlan1d(&s.plan[a], L, HIPFFT_C2C, int(g.clines));
        if (r != HIPFFT_SUCCESS) return std::unexpected(fftText("plan", r));
    }
    s.norm = Real(norm);
    auto R = DeviceBuffer<Real>::make(s.total);
    if (!R) return fail("device staging buffer (" + std::to_string(s.total * 4) + " B)");
    s.R = std::move(*R);
    auto C = DeviceBuffer<hipfftComplex>::make(cmax);
    if (!C) return fail("device spectrum buffer (" + std::to_string(cmax * 8) + " B)");
    s.C = std::move(*C);
    return SeparablePoisson(std::move(impl));
}

SeparablePoisson::SeparablePoisson(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SeparablePoisson::SeparablePoisson(SeparablePoisson&&) noexcept = default;
SeparablePoisson& SeparablePoisson::operator=(SeparablePoisson&&) noexcept = default;
SeparablePoisson::~SeparablePoisson() = default;

hipError_t SeparablePoisson::status() const { return impl_->status; }
const SepBox& SeparablePoisson::box() const { return impl_->box; }

void SeparablePoisson::solve(const Real* rhs, Real* p, Real scale, const std::uint8_t* keep)
{
    Impl& s = *impl_;
    int last = -1;
    for (int a = 0; a < 3; ++a)
        if (s.n[a] > 1) last = a;

    auto pass = [&](int a, bool inverse, const Real* in, const std::uint8_t* inKeep, Real sc, Real* out,
                    const std::uint8_t* outKeep, bool fuseScale) {
        PassGeom g = s.geom[a];
        g.inverse = inverse;
        const SepPack pk = sepPack(g.kind, inverse);
        const std::size_t cm = std::size_t(g.clines) * g.L;
        const float2 *tw0 = s.twPre[a].data(), *tw1 = s.twPost[a].data();
        switch (pk)
        {
        case SepPack::Single:
            kSepPre<SepPack::Single><<<launchBlocks(cm), blockSize>>>(g, in, inKeep, sc, tw0, tw1, s.C.data());
            break;
        case SepPack::RealIn:
            kSepPre<SepPack::RealIn><<<launchBlocks(cm), blockSize>>>(g, in, inKeep, sc, tw0, tw1, s.C.data());
            break;
        case SepPack::RealOut:
            kSepPre<SepPack::RealOut><<<launchBlocks(cm), blockSize>>>(g, in, inKeep, sc, tw0, tw1, s.C.data());
            break;
        }
        const bool fwdFft = inverse ? sepInvForward(g.kind) : true;
        const hipfftResult r =
            hipfftExecC2C(s.plan[a], s.C.data(), s.C.data(), fwdFft ? HIPFFT_FORWARD : HIPFFT_BACKWARD);
        if (r != HIPFFT_SUCCESS)
        {
            std::fprintf(stderr, "dns: %s\n", fftText("ExecC2C", r).c_str());
            s.record(hipErrorUnknown);
        }
        ScaleArgs sa;
        if (fuseScale)
            sa = ScaleArgs{s.lam[0].data(), s.lam[1].data(), s.lam[2].data(), s.norm, unsigned(s.n[0]),
                           unsigned(s.n[1])};
        const dim3 grid((g.lines + kTile - 1) / kTile, (g.L + kTile - 1) / kTile);
        const dim3 block(kTile, kRows, 1);
        switch (pk)
        {
        case SepPack::Single:
            kSepPost<SepPack::Single><<<grid, block>>>(g, s.C.data(), tw1, out, outKeep, sa);
            break;
        case SepPack::RealIn:
            kSepPost<SepPack::RealIn><<<grid, block>>>(g, s.C.data(), tw1, out, outKeep, sa);
            break;
        case SepPack::RealOut:
            kSepPost<SepPack::RealOut><<<grid, block>>>(g, s.C.data(), tw1, out, outKeep, sa);
            break;
        }
    };

    bool first = true;
    for (int a = 0; a < 3; ++a)
    {
        if (s.n[a] == 1) continue;
        pass(a, false, first ? rhs : s.R.data(), first ? keep : nullptr, first ? scale : Real(1), s.R.data(),
             nullptr, a == last);
        first = false;
    }
    for (int a = 0; a < 3; ++a)
    {
        if (s.n[a] == 1) continue;
        pass(a, true, s.R.data(), nullptr, Real(1), a == last ? p : s.R.data(), a == last ? keep : nullptr,
             false);
    }
    s.record(hipGetLastError());
}

} // namespace dns::gpu
