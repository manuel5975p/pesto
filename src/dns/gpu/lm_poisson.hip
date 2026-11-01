// LmPressure (gpu/lm_poisson.hpp): the apertured low-Mach pressure solve.
#include "dns/gpu/lm_poisson.hpp"

#include "dns/gpu/device_util.hpp"

#include <array>
#include <cfloat>
#include <cmath>
#include <vector>

namespace dns::gpu {

namespace {

struct LmGeom
{
    int nx, ny, nz;
    SepKind kx, ky, kz;
};

// y = A x on one cell: fluid rows -L_ap (face sums lo then hi, x, y, z),
// solid rows identity. ih2 = 1/h^2 per axis.
template <class T>
__global__ void kLmApply(LmGeom g, T ihx2, T ihy2, T ihz2, const std::uint8_t* code, const T* x,
                         T* y)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t sy = std::size_t(g.nx), sz = std::size_t(g.nx) * g.ny;
    if (c >= sz * std::size_t(g.nz)) return;
    const std::uint8_t f = code[c];
    const T xc = x[c];
    if (f & kLmSolid)
    {
        y[c] = xc;
        return;
    }
    const int ci = int(c);
    const int i = ci % g.nx, j = (ci / g.nx) % g.ny, k = ci / (g.nx * g.ny);
    T s = T(0);
    auto axis = [&](int q, int n, std::size_t st, std::uint8_t lo, std::uint8_t hi, SepKind kind,
                    T ih2) {
        T a = T(0);
        if (f & lo) a += xc - x[q > 0 ? c - st : c + st * std::size_t(n - 1)];
        else if (q == 0 && sepLoDirichlet(kind)) a += T(2) * xc;
        if (f & hi) a += xc - x[q < n - 1 ? c + st : c - st * std::size_t(n - 1)];
        else if (q == n - 1 && sepHiDirichlet(kind)) a += T(2) * xc;
        s += a * ih2;
    };
    axis(i, g.nx, 1, kLmFaceXLo, kLmFaceXHi, g.kx, ihx2);
    axis(j, g.ny, sy, kLmFaceYLo, kLmFaceYHi, g.ky, ihy2);
    if (g.nz > 1) axis(k, g.nz, sz, kLmFaceZLo, kLmFaceZHi, g.kz, ihz2);
    y[c] = s;
}

// b = -rhs on fluid cells, 0 on solid (A = -L_ap).
__global__ void kLmRhs(std::size_t n, const std::uint8_t* fluid, const Real* rhs, Real* b)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n) return;
    b[c] = fluid[c] ? -rhs[c] : Real(0);
}

__global__ void kLmZeroSolid(std::size_t n, const std::uint8_t* fluid, Real* x)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= n || fluid[c]) return;
    x[c] = Real(0);
}

// Host face codes; nz == 1 has no z faces.
std::vector<std::uint8_t> hostFaceCodes(const SepBox& b, std::span<const std::uint8_t> solid)
{
    const int n[3] = {b.nx, b.ny, b.nz};
    const SepKind kind[3] = {b.kx, b.ky, b.kz};
    const std::size_t st[3] = {1, std::size_t(b.nx), std::size_t(b.nx) * b.ny};
    const std::uint8_t lo[3] = {kLmFaceXLo, kLmFaceYLo, kLmFaceZLo};
    const std::uint8_t hi[3] = {kLmFaceXHi, kLmFaceYHi, kLmFaceZHi};
    std::vector<std::uint8_t> code(b.size(), 0);
    auto isSolid = [&](std::size_t c) { return !solid.empty() && solid[c] != 0; };
    for (int k = 0; k < b.nz; ++k)
        for (int j = 0; j < b.ny; ++j)
            for (int i = 0; i < b.nx; ++i)
            {
                const std::size_t c = std::size_t(i) + st[1] * j + st[2] * k;
                if (isSolid(c))
                {
                    code[c] = kLmSolid;
                    continue;
                }
                const int q[3] = {i, j, k};
                std::uint8_t f = 0;
                for (int a = 0; a < 3; ++a)
                {
                    if (n[a] == 1) continue;
                    const bool per = kind[a] == SepKind::Periodic;
                    if (q[a] > 0 || per)
                    {
                        const std::size_t nb = q[a] > 0 ? c - st[a] : c + st[a] * std::size_t(n[a] - 1);
                        if (!isSolid(nb)) f |= lo[a];
                    }
                    if (q[a] < n[a] - 1 || per)
                    {
                        const std::size_t nb = q[a] < n[a] - 1 ? c + st[a] : c - st[a] * std::size_t(n[a] - 1);
                        if (!isSolid(nb)) f |= hi[a];
                    }
                }
                code[c] = f;
            }
    return code;
}

// Fluid cells not connected through open faces to a Dirichlet boundary face:
// count and the lowest such index (phi undetermined there).
std::pair<std::size_t, std::size_t> enclosed(const SepBox& b, const std::vector<std::uint8_t>& code)
{
    const int n[3] = {b.nx, b.ny, b.nz};
    const SepKind kind[3] = {b.kx, b.ky, b.kz};
    const std::size_t st[3] = {1, std::size_t(b.nx), std::size_t(b.nx) * b.ny};
    const std::uint8_t lo[3] = {kLmFaceXLo, kLmFaceYLo, kLmFaceZLo};
    const std::uint8_t hi[3] = {kLmFaceXHi, kLmFaceYHi, kLmFaceZHi};
    std::vector<std::uint8_t> seen(b.size(), 0);
    std::vector<std::size_t> queue;
    auto coord = [&](std::size_t c, int a) { return int((c / st[a]) % std::size_t(n[a])); };
    for (std::size_t c = 0; c < b.size(); ++c)
    {
        if (code[c] & kLmSolid) continue;
        bool dir = false;
        for (int a = 0; a < 3; ++a)
        {
            if (n[a] == 1) continue;
            const int q = coord(c, a);
            dir |= (q == 0 && sepLoDirichlet(kind[a])) || (q == n[a] - 1 && sepHiDirichlet(kind[a]));
        }
        if (dir)
        {
            seen[c] = 1;
            queue.push_back(c);
        }
    }
    for (std::size_t h = 0; h < queue.size(); ++h)
    {
        const std::size_t c = queue[h];
        for (int a = 0; a < 3; ++a)
        {
            if (n[a] == 1) continue;
            const int q = coord(c, a);
            if (code[c] & lo[a])
            {
                const std::size_t nb = q > 0 ? c - st[a] : c + st[a] * std::size_t(n[a] - 1);
                if (!seen[nb]) { seen[nb] = 1; queue.push_back(nb); }
            }
            if (code[c] & hi[a])
            {
                const std::size_t nb = q < n[a] - 1 ? c + st[a] : c - st[a] * std::size_t(n[a] - 1);
                if (!seen[nb]) { seen[nb] = 1; queue.push_back(nb); }
            }
        }
    }
    std::size_t count = 0, first = b.size();
    for (std::size_t c = 0; c < b.size(); ++c)
        if (!(code[c] & kLmSolid) && !seen[c])
        {
            if (!count) first = c;
            ++count;
        }
    return {count, first};
}

} // namespace

struct LmPressure::Impl
{
    SepBox box;
    LmPressureOptions opt;
    LmGeom geom{};
    double ih2[3] = {0, 0, 0};
    double normA = 0; // Gershgorin bound of |A|_2
    std::size_t solid = 0;
    SeparablePoisson sep;
    DeviceBuffer<std::uint8_t> code, fluid;
    DeviceBuffer<Real> b;
    std::unique_ptr<la::Krylov<float>> krylov;

    Impl(SeparablePoisson s) : sep(std::move(s)) {}
};

std::expected<LmPressure, std::string>
LmPressure::make(const SepBox& box, std::span<const std::uint8_t> solid, const LmPressureOptions& opt)
{
    auto sep = SeparablePoisson::make(box);
    if (!sep) return std::unexpected("LmPressure: " + sep.error());
    if (!solid.empty() && solid.size() != box.size())
        return std::unexpected("LmPressure: solid map has " + std::to_string(solid.size()) +
                               " cells, the box " + std::to_string(box.size()));
    if (!(opt.tol > 0) || opt.maxIter < 1)
        return std::unexpected("LmPressure: tol must be > 0 and maxIter >= 1");
    auto impl = std::make_unique<Impl>(std::move(*sep));
    Impl& s = *impl;
    s.box = box;
    s.opt = opt;
    s.geom = {box.nx, box.ny, box.nz, box.kx, box.ky, box.kz};
    const double h[3] = {box.hx, box.hy, box.hz};
    for (int a = 0; a < 3; ++a)
    {
        s.ih2[a] = 1.0 / (h[a] * h[a]);
        if (a < 2 || box.nz > 1) s.normA += 4.0 * s.ih2[a];
    }
    for (std::uint8_t v : solid) s.solid += v != 0;

    const auto code = hostFaceCodes(box, s.solid ? solid : std::span<const std::uint8_t>{});
    if (s.solid)
    {
        const bool anyDir = sepHiDirichlet(box.kx) || sepHiDirichlet(box.ky) ||
                            (box.nz > 1 && sepHiDirichlet(box.kz));
        if (!anyDir)
            return std::unexpected("LmPressure: an apertured solve needs a Dirichlet face "
                                   "(kind DD or ND on some axis): the fluid pressure is "
                                   "undetermined otherwise");
        const auto [count, first] = enclosed(box, code);
        if (count)
        {
            const std::size_t i = first % std::size_t(box.nx), j = (first / box.nx) % std::size_t(box.ny),
                              k = first / (std::size_t(box.nx) * box.ny);
            return std::unexpected("LmPressure: " + std::to_string(count) +
                                   " fluid cells are enclosed by solid away from every Dirichlet "
                                   "face (first at i " + std::to_string(i) + ", j " +
                                   std::to_string(j) + ", k " + std::to_string(k) +
                                   "): the pressure is undetermined there");
        }
    }
    std::vector<std::uint8_t> fl(box.size());
    for (std::size_t c = 0; c < box.size(); ++c) fl[c] = (code[c] & kLmSolid) ? 0 : 1;
    auto dc = DeviceBuffer<std::uint8_t>::make(box.size());
    if (!dc || dc->upload(code.data()) != hipSuccess) return std::unexpected("LmPressure: device face codes");
    auto df = DeviceBuffer<std::uint8_t>::make(box.size());
    if (!df || df->upload(fl.data()) != hipSuccess) return std::unexpected("LmPressure: device fluid map");
    s.code = std::move(*dc);
    s.fluid = std::move(*df);
    if (s.solid)
    {
        auto bb = DeviceBuffer<Real>::make(box.size());
        if (!bb) return std::unexpected("LmPressure: device rhs buffer");
        s.b = std::move(*bb);
        auto k = la::Krylov<float>::make(box.size(), 0);
        if (!k) return std::unexpected("LmPressure: " + k.error());
        s.krylov = std::make_unique<la::Krylov<float>>(std::move(*k));
    }
    return LmPressure(std::move(impl));
}

LmPressure::LmPressure(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LmPressure::LmPressure(LmPressure&&) noexcept = default;
LmPressure& LmPressure::operator=(LmPressure&&) noexcept = default;
LmPressure::~LmPressure() = default;

const std::uint8_t* LmPressure::faceCodes() const { return impl_->code.data(); }
const std::uint8_t* LmPressure::fluid() const { return impl_->fluid.data(); }
std::size_t LmPressure::solidCells() const { return impl_->solid; }
SeparablePoisson& LmPressure::spectral() { return impl_->sep; }
const SepBox& LmPressure::box() const { return impl_->box; }

void LmPressure::apply(const float* x, float* y) const
{
    const Impl& s = *impl_;
    kLmApply<float><<<launchBlocks(s.box.size()), blockSize>>>(
        s.geom, float(s.ih2[0]), float(s.ih2[1]), float(s.ih2[2]), s.code.data(), x, y);
}

void LmPressure::apply(const double* x, double* y) const
{
    const Impl& s = *impl_;
    kLmApply<double><<<launchBlocks(s.box.size()), blockSize>>>(s.geom, s.ih2[0], s.ih2[1], s.ih2[2],
                                                                s.code.data(), x, y);
}

std::expected<LmSolveResult, std::string> LmPressure::solve(const Real* rhs, Real* phi, bool warm)
{
    Impl& s = *impl_;
    const std::size_t N = s.box.size();
    LmSolveResult res;
    if (!s.solid)
    {
        s.sep.solve(rhs, phi);
        if (const hipError_t e = s.sep.status(); e != hipSuccess)
            return std::unexpected(std::string("LmPressure: spectral solve: ") + hipGetErrorString(e));
        res.direct = res.converged = true;
        return res;
    }
    kLmRhs<<<launchBlocks(N), blockSize>>>(N, s.fluid.data(), rhs, s.b.data());
    if (warm) kLmZeroSolid<<<launchBlocks(N), blockSize>>>(N, s.fluid.data(), phi);
    else if (const hipError_t e = hipMemsetAsync(phi, 0, N * sizeof(Real), 0); e != hipSuccess)
        return std::unexpected(std::string("LmPressure: ") + hipGetErrorString(e));

    la::Krylov<float>& kr = *s.krylov;
    double bb = 0;
    {
        const la::DotPair<float> p[1] = {{s.b.data(), s.b.data()}};
        if (auto e = kr.dots(std::span(p, 1), std::span(&bb, 1)); !e)
            return std::unexpected("LmPressure: " + e.error());
    }
    res.bNorm = std::sqrt(bb);
    if (bb == 0)
    {
        if (const hipError_t e = hipMemsetAsync(phi, 0, N * sizeof(Real), 0); e != hipSuccess)
            return std::unexpected(std::string("LmPressure: ") + hipGetErrorString(e));
        res.converged = true;
        return res;
    }
    const la::LinOp<float> A = [this](const float* x, float* y) { apply(x, y); };
    SeparablePoisson& sep = s.sep;
    const std::uint8_t* fl = s.fluid.data();
    const la::LinOp<float> M = [&sep, fl](const float* r, float* z) { sep.solve(r, z, Real(-1), fl); };
    la::KrylovOptions o;
    o.maxIter = s.opt.maxIter;
    o.relTol = 0;
    o.absTol = s.opt.tol * res.bNorm;
    auto k = kr.pcg(A, M, s.b.data(), phi, o);
    if (!k) return std::unexpected("LmPressure: " + k.error());
    if (const hipError_t e = sep.status(); e != hipSuccess)
        return std::unexpected(std::string("LmPressure: spectral preconditioner: ") + hipGetErrorString(e));
    res.iterations = k->iterations;
    res.converged = k->converged;
    res.relResidual = k->residual / res.bNorm;
    double xx = 0;
    {
        const la::DotPair<float> p[1] = {{phi, phi}};
        if (auto e = kr.dots(std::span(p, 1), std::span(&xx, 1)); !e)
            return std::unexpected("LmPressure: " + e.error());
    }
    res.floorRel = double(FLT_EPSILON) * s.normA * std::sqrt(xx) / res.bNorm;
    return res;
}

} // namespace dns::gpu
