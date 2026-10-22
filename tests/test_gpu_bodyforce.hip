// DeviceBodyForce.
//
// The tree carried two friction tallies: cylinder/eye summed x/y links, sphere/
// mesh summed x/y/z. The unified one sums all three unconditionally. The claim
// that this is a NO-OP for a z-invariant body is the thing that must be
// measured, not assumed:
//   * at nz == 1 the z neighbours gidx(g,i,j,k+-1) wrap to the cell itself,
//     whose aperture passed the > openTol guard that gated the call, so neither
//     z link is taken;
//   * for a z-invariant body at nz > 1 the z neighbours of a body face are
//     fluid, so they are skipped too.
// This test computes cdf both ways and requires bit-identity in both cases.
// It also pins determinism (repeat runs must agree to the last bit) and that
// the pressure tally's fz is exactly 0 in native 2D.
#include "dns/bodies.hpp"
#include "dns/cutcell.hpp"
#include "dns/geometry.hpp"
#include "dns/gpu/bodyforce.hpp"
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <vector>

using namespace dns;
using gpu::Real;

namespace {

using dnstest::check;

// The pre-recomposition x/y-only friction tally (cylinder:196, eye), verbatim,
// as the oracle.
__global__ void kBodyFrictionXY(gpu::GpuGrid g, Real nu, const Real* ax,
                                const Real* ay, const Real* u, const Real* v,
                                int i0, int i1, int j0, int j1, int k0, int k1,
                                Real* fx, Real* fy)
{
    constexpr Real openTol = Real(CutGeometry::openTol);
    const std::size_t c = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= g.size()) return;
    int i, j, k;
    gpu::decompose(g, c, i, j, k);
    fx[c] = 0;
    fy[c] = 0;
    if (i < i0 || i > i1 || j < j0 || j > j1 || k < k0 || k > k1) return;
    const Real lx = g.hy * g.hz / g.hx;
    const Real ly = g.hx * g.hz / g.hy;
    auto closedLinks = [&](const Real* a, const Real* q) {
        Real s = 0;
        auto link = [&](std::size_t nb, Real l) {
            if (a[nb] <= openTol) s += l * (q[c] - q[nb]);
        };
        link(gidx(g, i - 1, j, k), lx);
        link(gidx(g, i + 1, j, k), lx);
        link(gidx(g, i, j - 1, k), ly);
        link(gidx(g, i, j + 1, k), ly);
        return s;
    };
    if (ax[c] > openTol) fx[c] = nu * closedLinks(ax, u);
    if (ay[c] > openTol) fy[c] = nu * closedLinks(ay, v);
}

struct Case2D
{
    Grid g;
    CutGeometry cut;
    CylinderZ cyl;
};

// A sanitized z-cylinder in a ghost-ringed box, nzInt interior cells in z
// (nzInt <= 1 -> native 2D).
Case2D makeCylinder(int n, int nzInt)
{
    using std::numbers::pi;
    const double L = 2 * pi, h = L / n;
    const int nx = n + 2, nz = nzInt <= 1 ? 1 : nzInt + 2;
    const auto g = Grid::make(nx, nx, nz, nx * h, nx * h, nz == 1 ? h : nz * h);
    const double D = L / 8.0;
    const CylinderZ cyl{h + L / 3.0 + 0.31 * h, h + L / 2.0 + 0.17 * h, D / 2.0};
    auto cut = CutGeometry::make(g, cyl);
    (void)sanitizeCut(g, cut);
    closeBoxBoundary(g, cut);
    return {g, std::move(cut), cyl};
}

// A z-invariant velocity field: same x-y plane in every k layer, w == 0.
void fillZInvariant(const Grid& g, std::vector<Real>& u, std::vector<Real>& v,
                    std::vector<Real>& w, unsigned seed)
{
    const std::size_t slab = std::size_t(g.nx) * g.ny;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Real> d(-1, 1);
    std::vector<Real> pu(slab), pv(slab);
    for (std::size_t c = 0; c < slab; ++c) { pu[c] = d(rng); pv[c] = d(rng); }
    u.assign(g.size(), 0);
    v.assign(g.size(), 0);
    w.assign(g.size(), 0);
    for (int k = 0; k < g.nz; ++k)
        for (std::size_t c = 0; c < slab; ++c)
        {
            u[c + slab * k] = pu[c];
            v[c + slab * k] = pv[c];
        }
}

bool upload(gpu::DeviceField& f, const std::vector<Real>& u,
            const std::vector<Real>& v, const std::vector<Real>& w)
{
    const std::size_t b = u.size() * sizeof(Real);
    return hipMemcpy(f.u.data(), u.data(), b, hipMemcpyHostToDevice) == hipSuccess
        && hipMemcpy(f.v.data(), v.data(), b, hipMemcpyHostToDevice) == hipSuccess
        && hipMemcpy(f.w.data(), w.data(), b, hipMemcpyHostToDevice) == hipSuccess;
}

// cdf from the legacy x/y-only tally, over the same box and field.
double legacyCdf(const Grid& g, const gpu::DeviceCutGeometry& cut,
                 const gpu::DeviceField& q, const gpu::BodyBox& b, double nu,
                 double qRef, gpu::ReduceScratch& red)
{
    const auto gg = gpu::GpuGrid::from(g);
    auto fx = gpu::DeviceScalar::make(g.size());
    auto fy = gpu::DeviceScalar::make(g.size());
    if (!fx || !fy) return std::nan("");
    kBodyFrictionXY<<<gpu::launchBlocks(g.size()), gpu::blockSize>>>(
        gg, Real(nu), cut.ax.data(), cut.ay.data(), q.u.data(), q.v.data(), b.i0,
        b.i1, b.j0, b.j1, b.k0, b.k1, fx->data(), fy->data());
    gpu::checkLastKernel("frictionXY");
    return gpu::sumScalar(*fx, red) / qRef;
}

// Measured cdf from the unified 3-direction tally. Pressure is fed a zero phi
// so the reported cdf isolates friction exactly.
double unifiedCdf(const Grid& g, const gpu::DeviceCutGeometry& cut,
                  const gpu::DeviceField& q, const gpu::BodyBox& b, double nu,
                  double qRef, gpu::ReduceScratch& red, double* cs = nullptr)
{
    const auto gg = gpu::GpuGrid::from(g);
    auto bf = gpu::DeviceBodyForce::make(gg, b);
    auto phi = gpu::DeviceScalar::make(g.size());
    if (!bf || !phi) return std::nan("");
    (void)hipMemset(phi->data(), 0, g.size() * sizeof(Real));
    const auto c = bf->measure(cut, *phi, q, nu, qRef, nullptr, red);
    if (cs) *cs = c.cs;
    return c.cdf;
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    using std::numbers::pi;

    auto red = gpu::ReduceScratch::make();
    check(red.has_value(), "ReduceScratch");
    if (!red) return 1;

    const double nu = 1.0 * (2 * pi / 8.0) / 60.0;

    // --- 1) native 2D: 3-link tally == 2-link tally, bit for bit -----------
    {
        auto c = makeCylinder(96, 1);
        const auto gg = gpu::GpuGrid::from(c.g);
        auto dCut = gpu::DeviceCutGeometry::make(c.cut);
        auto q = gpu::DeviceField::make(c.g.size());
        check(dCut && q, "alloc 2d");
        if (!(dCut && q)) return 1;
        std::vector<Real> u, v, w;
        fillZInvariant(c.g, u, v, w, 5);
        check(upload(*q, u, v, w), "upload 2d");
        const auto box = gpu::bodyBox(gg, c.cyl.cx, c.cyl.cy, 0, c.cyl.r);
        const double qRef = 0.5 * (2 * pi / 8.0) * c.g.hz;
        double cs = 0;
        const double a = legacyCdf(c.g, *dCut, *q, box, nu, qRef, *red);
        const double b = unifiedCdf(c.g, *dCut, *q, box, nu, qRef, *red, &cs);
        std::printf("nz==1  cdf xy-only = %.17g\n       cdf xyz    = %.17g\n", a, b);
        check(a == b, "nz == 1: the z links contribute exactly zero");
        check(cs == 0.0, "nz == 1: cs is exactly zero");
        check(std::isfinite(a) && a != 0.0, "nz == 1: the tally is non-trivial");
    }

    // --- 2) z-invariant body at nz > 1: same, bit for bit ------------------
    {
        auto c = makeCylinder(64, 8);
        const auto gg = gpu::GpuGrid::from(c.g);
        auto dCut = gpu::DeviceCutGeometry::make(c.cut);
        auto q = gpu::DeviceField::make(c.g.size());
        check(dCut && q, "alloc 3d");
        if (!(dCut && q)) return 1;
        std::vector<Real> u, v, w;
        fillZInvariant(c.g, u, v, w, 6);
        check(upload(*q, u, v, w), "upload 3d");
        auto box = gpu::bodyBox(gg, c.cyl.cx, c.cyl.cy, 0, c.cyl.r);
        box.k0 = gg.kInLo();
        box.k1 = gg.kInHi();
        const double qRef = 0.5 * (2 * pi / 8.0) * (8 * c.g.hz);
        const double a = legacyCdf(c.g, *dCut, *q, box, nu, qRef, *red);
        const double b = unifiedCdf(c.g, *dCut, *q, box, nu, qRef, *red);
        std::printf("nz==10 cdf xy-only = %.17g\n       cdf xyz    = %.17g\n", a, b);
        check(a == b, "z-invariant body: the z links contribute exactly zero");
        check(std::isfinite(a) && a != 0.0, "nz > 1: the tally is non-trivial");
    }

    // --- 3) determinism: repeated tallies agree to the last bit ------------
    {
        auto c = makeCylinder(96, 1);
        const auto gg = gpu::GpuGrid::from(c.g);
        auto dCut = gpu::DeviceCutGeometry::make(c.cut);
        auto q = gpu::DeviceField::make(c.g.size());
        if (!(dCut && q)) return 1;
        std::vector<Real> u, v, w;
        fillZInvariant(c.g, u, v, w, 7);
        check(upload(*q, u, v, w), "upload det");
        const auto box = gpu::bodyBox(gg, c.cyl.cx, c.cyl.cy, 0, c.cyl.r);
        const double qRef = 0.5 * (2 * pi / 8.0) * c.g.hz;
        const double first = unifiedCdf(c.g, *dCut, *q, box, nu, qRef, *red);
        bool same = true;
        for (int r = 0; r < 4; ++r)
            if (unifiedCdf(c.g, *dCut, *q, box, nu, qRef, *red) != first) same = false;
        check(same, "the tally is bit-reproducible across repeats");
    }

    // --- 4) argument validation --------------------------------------------
    {
        const auto gg = gpu::GpuGrid::from(Grid::make(16, 16, 1, 16, 16, 1));
        check(!gpu::DeviceBodyForce::make(gg, {5, 4, 0, 3, 0, 0}).has_value(),
              "rejects an empty box");
        check(!gpu::DeviceBodyForce::make(gg, {0, 20, 0, 3, 0, 0}).has_value(),
              "rejects a box outside the grid");
        check(gpu::DeviceBodyForce::make(gg, {2, 6, 2, 6, 0, 0}).has_value(),
              "accepts a valid box");
    }

    if (dnstest::gFailures == 0) std::printf("gpu_bodyforce: ok\n");
    return dnstest::finish();
}
