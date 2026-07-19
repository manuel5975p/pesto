// DeviceTunnelBc / DeviceFringe.
//
// The tree carried two variants of the tunnel ghost-ring kernel. The unified
// one adopts the diagonal corner mirror (correct for a box free-slip on both y
// and z) plus the nz == 1 guard on the z mirror row. This test pins the two
// properties that decision rests on:
//   1) at nz == 1 the unified kernel is BIT-IDENTICAL to the old native-2D
//      variant (Class A), so no 2D run moves;
//   2) at nz > 1 an edge ghost is mirrored across BOTH walls, and a
//      z-invariant field stays z-invariant.
// It also pins that the outlet trio uses GpuGrid::kInterior, whose hardcoded
// [1, nz-2] form is an EMPTY range at nz == 1 -- adopting that would silently
// disable outlet flux correction in every 2D run.
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/tunnel.hpp"

#include "gpu_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace dns;
using gpu::Real;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("FAIL: %s\n", what); ++failures; }
}

// The pre-recomposition native-2D variant (Class A: cylinder:75, eye:62),
// verbatim, as the bit-identity oracle at nz == 1.
__global__ void kBcTunnelClassA(gpu::GpuGrid g, Real uInf, Real cdt, const Real* inu,
                                const Real* inv, const Real* inw, Real* u, Real* v,
                                Real* w)
{
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const int km = nz == 1 ? k : (k == 0 ? 1 : k == nz - 1 ? nz - 2 : k);
    const int jm = j == 0 ? 1 : j == ny - 1 ? ny - 2 : j;

    if (i <= 1) u[c] = uInf;
    else if (i == nx - 1)
        u[c] = inu[c] - cdt * (inu[c] - inu[gidx(g, nx - 2, j, k)]);
    else if (jm != j) u[c] = u[gidx(g, i, jm, k)];
    else if (km != k) u[c] = u[gidx(g, i, j, km)];

    if (i == 0) v[c] = -v[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        v[c] = inv[c] - cdt * (inv[c] - inv[gidx(g, nx - 2, j, k)]);
    else if (j == 1 || j == ny - 1) v[c] = 0;
    else if (j == 0) v[c] = -v[gidx(g, i, 2, k)];
    else if (km != k) v[c] = v[gidx(g, i, j, km)];

    if (i == 0) w[c] = -w[gidx(g, 1, j, k)];
    else if (i == nx - 1)
        w[c] = inw[c] - cdt * (inw[c] - inw[gidx(g, nx - 2, j, k)]);
    else if (jm != j) w[c] = w[gidx(g, i, jm, k)];
    else if (nz == 1 || k <= 1 || k == nz - 1) w[c] = 0;
}

std::vector<Real> noise(std::size_t n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> d(-1, 1);
    std::vector<Real> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

bool upload(gpu::DeviceField& f, const std::vector<Real>& u,
            const std::vector<Real>& v, const std::vector<Real>& w)
{
    const std::size_t b = u.size() * sizeof(Real);
    return hipMemcpy(f.u.data(), u.data(), b, hipMemcpyHostToDevice) == hipSuccess
        && hipMemcpy(f.v.data(), v.data(), b, hipMemcpyHostToDevice) == hipSuccess
        && hipMemcpy(f.w.data(), w.data(), b, hipMemcpyHostToDevice) == hipSuccess;
}

std::vector<Real> download(const gpu::DeviceScalar& q, std::size_t n)
{
    std::vector<Real> h(n);
    (void)hipMemcpy(h.data(), q.data(), n * sizeof(Real), hipMemcpyDeviceToHost);
    return h;
}

} // namespace

int main()
{
    if (!gpu::deviceAvailable()) { std::printf("no HIP device\n"); return 77; }

    // --- 1) nz == 1: unified kernel is bit-identical to Class A -------------
    {
        const auto g = Grid::make(34, 34, 1, 34.0, 34.0, 1.0);
        const auto gg = gpu::GpuGrid::from(g);
        const auto n = g.size();
        // w is identically zero in a native-2D state and the kernels preserve
        // that; seeding it with noise would compare the two variants on a state
        // the 2D restriction cannot produce (they order the w clauses
        // differently on the y-ghost rows, where the unified form pins w = 0
        // and the old one mirrored it -- unreachable while w == 0).
        const auto u0 = noise(n, 1), v0 = noise(n, 2);
        const std::vector<Real> w0(n, Real(0));

        auto in = gpu::DeviceField::make(n);
        auto a = gpu::DeviceField::make(n);
        auto b = gpu::DeviceField::make(n);
        check(in && a && b, "alloc");
        if (!(in && a && b)) return 1;
        check(upload(*in, u0, v0, w0), "upload in");
        check(upload(*a, u0, v0, w0), "upload a");
        check(upload(*b, u0, v0, w0), "upload b");

        const double cdt = 0.37;
        kBcTunnelClassA<<<gpu::launchBlocks(n), gpu::blockSize>>>(
            gg, Real(1.0), Real(cdt), in->u.data(), in->v.data(), in->w.data(),
            a->u.data(), a->v.data(), a->w.data());
        gpu::checkLastKernel("classA");

        auto bc = gpu::DeviceTunnelBc::make(gg, {1.0});
        check(bc.has_value(), "DeviceTunnelBc::make");
        if (bc) bc->apply(*in, *b, cdt);
        (void)hipDeviceSynchronize();

        const auto au = download(a->u, n), av = download(a->v, n),
                   aw = download(a->w, n);
        const auto bu = download(b->u, n), bv = download(b->v, n),
                   bw = download(b->w, n);
        std::size_t diff = 0;
        for (std::size_t c = 0; c < n; ++c)
            diff += (au[c] != bu[c]) + (av[c] != bv[c]) + (aw[c] != bw[c]);
        check(diff == 0, "nz == 1: unified kernel bit-identical to the 2D variant");
        for (std::size_t c = 0; c < n; ++c)
            if (bw[c] != Real(0)) { check(false, "nz == 1: w stays exactly zero"); break; }
    }

    // --- 2) nz > 1: edge ghosts mirror across BOTH walls --------------------
    {
        const int nz = 10;
        const auto g = Grid::make(16, 16, nz, 16.0, 16.0, double(nz));
        const auto gg = gpu::GpuGrid::from(g);
        const auto n = g.size();
        auto uh = noise(n, 7), vh = noise(n, 8), wh = noise(n, 9);
        auto f = gpu::DeviceField::make(n);
        check(f.has_value(), "alloc 3d");
        if (!f) return 1;
        check(upload(*f, uh, vh, wh), "upload 3d");
        auto bc = gpu::DeviceTunnelBc::make(gg, {1.0});
        if (bc) bc->apply(*f, *f, 0.0);
        (void)hipDeviceSynchronize();
        const auto u = download(f->u, n);
        auto at = [&](const std::vector<Real>& q, int i, int j, int k) {
            return q[g.idx(i, j, k)];
        };
        // The (j=0, k=0) edge ghost must equal the (1,1) diagonal interior
        // value, not the (1,0) or (0,1) single-wall mirror.
        bool diagonal = true;
        for (int i = 2; i < g.nx - 1; ++i)
            if (at(u, i, 0, 0) != at(u, i, 1, 1)) diagonal = false;
        check(diagonal, "nz > 1: edge ghost is the diagonal (jm,km) mirror");
    }

    // --- 3) a z-invariant field stays z-invariant ---------------------------
    {
        const int nz = 10;
        const auto g = Grid::make(16, 16, nz, 16.0, 16.0, double(nz));
        const auto gg = gpu::GpuGrid::from(g);
        const auto n = g.size();
        const auto plane = noise(std::size_t(g.nx) * g.ny, 11);
        std::vector<Real> uh(n), vh(n), wh(n, 0);
        for (int k = 0; k < nz; ++k)
            for (std::size_t c = 0; c < plane.size(); ++c)
            {
                uh[c + plane.size() * k] = plane[c];
                vh[c + plane.size() * k] = plane[c] * Real(0.5);
            }
        auto f = gpu::DeviceField::make(n);
        if (!f) return 1;
        check(upload(*f, uh, vh, wh), "upload z-invariant");
        auto bc = gpu::DeviceTunnelBc::make(gg, {1.0});
        if (bc) bc->apply(*f, *f, 0.0);
        (void)hipDeviceSynchronize();
        const auto u = download(f->u, n);
        bool inv = true;
        for (int k = 2; k <= nz - 3; ++k)
            for (std::size_t c = 0; c < plane.size(); ++c)
                if (u[c + plane.size() * k] != u[c + plane.size() * 1]) inv = false;
        check(inv, "free-slip z walls preserve a z-invariant field");
    }

    // --- 4) outlet flux correction is ALIVE at nz == 1 ----------------------
    // The hardcoded k in [1, nz-2] form is empty at nz == 1; if it were adopted
    // the corrected outflow would keep its unbalanced value.
    {
        const auto g = Grid::make(34, 34, 1, 34.0, 34.0, 1.0);
        const auto gg = gpu::GpuGrid::from(g);
        const auto n = g.size();
        std::vector<Real> uh(n, Real(0.5)), vh(n, 0), wh(n, 0); // half the inflow
        auto f = gpu::DeviceField::make(n);
        auto tmp = gpu::DeviceScalar::make(n);
        auto red = gpu::ReduceScratch::make();
        check(f && tmp && red, "alloc outlet");
        if (!(f && tmp && red)) return 1;
        check(upload(*f, uh, vh, wh), "upload outlet");
        auto bc = gpu::DeviceTunnelBc::make(gg, {1.0});
        if (bc) bc->correctOutletFlux(*f, *red, *tmp);
        (void)hipDeviceSynchronize();
        const auto u = download(f->u, n);
        double s = 0;
        for (int j = 1; j <= g.ny - 2; ++j) s += u[g.idx(g.nx - 1, j, 0)];
        const double want = double(g.ny - 2) * 1.0;
        check(std::abs(s - want) < 1e-3 * want,
              "outlet flux correction restores exact mass balance at nz == 1");
    }

    // --- 5) DeviceFringe argument validation --------------------------------
    {
        const auto g2 = gpu::GpuGrid::from(Grid::make(64, 64, 1, 64, 64, 1));
        check(gpu::DeviceFringe::make(g2, {1.0, 0.5, 8, 2}).has_value(),
              "fringe axes 2 at nz == 1");
        check(!gpu::DeviceFringe::make(g2, {1.0, 0.5, 8, 3}).has_value(),
              "fringe rejects a z band that does not fit");
        check(!gpu::DeviceFringe::make(g2, {1.0, 0.5, 0, 2}).has_value(),
              "fringe rejects cells < 1");
        check(!gpu::DeviceFringe::make(g2, {1.0, 0.5, 40, 2}).has_value(),
              "fringe rejects cells >= nx/2");
        const auto g3 = gpu::GpuGrid::from(Grid::make(64, 64, 64, 64, 64, 64));
        check(gpu::DeviceFringe::make(g3, {1.0, 0.5, 8, 3}).has_value(),
              "fringe axes 3 when the band fits");
    }

    // --- 6) DeviceTunnelBc argument validation ------------------------------
    {
        const auto bad = gpu::GpuGrid::from(Grid::make(2, 2, 1, 2, 2, 1));
        check(!gpu::DeviceTunnelBc::make(bad, {1.0}).has_value(),
              "tunnel rejects a grid with no interior");
        const auto ok = gpu::GpuGrid::from(Grid::make(16, 16, 1, 16, 16, 1));
        check(!gpu::DeviceTunnelBc::make(ok, {0.0}).has_value(),
              "tunnel rejects uInf <= 0");
    }

    if (failures == 0) std::printf("gpu_tunnel_bc: ok\n");
    return failures == 0 ? 0 : 1;
}
