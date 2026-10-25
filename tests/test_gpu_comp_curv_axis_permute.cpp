// The curvilinear (metric) compressible path is axis-blind too. A 2D vortex
// on a mesh clustered in x, transposed, is the same flow on the mesh
// clustered in y: after the same steps the two must agree after transposing
// back. In 3D, three DIFFERENT per-axis focal ratios on a box with three
// different extents and cell counts, rotated cyclically x -> y -> z. Any
// metric table that pairs an S or Ji component with the wrong axis fails here
// while every isotropic or single-axis mesh stays green.
// CPU (FP64) to 1e-12, device (FP32) to 1e-6 of each field's max. The CPU
// half runs without a device; the test then skips (77) unless it failed.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "curv_test_util.hpp"
#include "permute_test_util.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

using namespace dns;
using dnstest::check;
using std::numbers::pi;

namespace {

constexpr double kGamma = 1.4, kMu = 2e-3;
constexpr int kSteps = 10;

using Perm = CompState (*)(const Grid&, const CompState&);

// Isentropic vortex (strength 5) in a (1, 0.5) stream centred in the box, at
// the mesh's cell centroids.
CompState vortex(const CurvMesh& m)
{
    const Grid& g = m.g;
    auto s = CompState::zeros(g);
    const double x0 = 0.5 * g.nx * g.hx, y0 = 0.5 * g.ny * g.hy, beta = 5.0;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const auto p = curvtest::centroid(m, i, j, 0);
            const double dx = p[0] - x0, dy = p[1] - y0;
            const double f = beta / (2 * pi) * std::exp(0.5 * (1.0 - dx * dx - dy * dy));
            const double u = 1.0 - f * dy, v = 0.5 + f * dx;
            const double rho = curvtest::vortexRho(p[0], p[1], x0, y0, beta, kGamma);
            const std::size_t c = g.idx(i, j, 0);
            s.rho[c] = rho;
            s.mx[c] = rho * u;
            s.my[c] = rho * v;
            s.E[c] = std::pow(rho, kGamma) / (kGamma - 1.0)
                     + 0.5 * rho * (u * u + v * v);
        }
    return s;
}

struct Pair
{
    CurvMesh a, b; // mesh and its permuted twin
    CompState s0;  // IC on a
    Perm P;
};

void cpuCase(const std::string& tag, const Pair& c)
{
    const auto ca = Comp::make(c.a, kGamma, kMu);
    const auto cb = Comp::make(c.b, kGamma, kMu);
    if (!ca || !cb) { check(false, tag + ": CPU curvilinear make"); return; }
    const double dt = 0.5 * *ca->stableDt(c.s0);
    auto sa = c.s0;
    auto sb = c.P(c.a.g, c.s0);
    for (int it = 0; it < kSteps; ++it) {
        ca->step(sa, dt);
        cb->step(sb, dt);
    }
    const double e = permtest::relState(sb, c.P(c.a.g, sa));
    std::printf("CPU %-26s dt %.3e  max rel |step(P s) - P step(s)| %.3e\n",
                tag.c_str(), dt, e);
    check(e < 1e-12, "CPU " + tag + ": step(P s) == P step(s) to 1e-12");
}

void gpuCase(const std::string& tag, const Pair& c)
{
    const auto ma = CurvMetrics::build(c.a), mb = CurvMetrics::build(c.b);
    if (!ma || !mb) { check(false, tag + ": metrics"); return; }
    auto da = gpu::DeviceComp::make(*ma, kGamma, kMu);
    auto db = gpu::DeviceComp::make(*mb, kGamma, kMu);
    auto sa = gpu::DeviceCompState::make(c.a.g.size());
    auto sb = gpu::DeviceCompState::make(c.b.g.size());
    auto red = gpu::ReduceScratch::make();
    if (!da || !db || !sa || !sb || !red) { check(false, tag + ": GPU alloc"); return; }
    (void)sa->upload(c.s0);
    (void)sb->upload(c.P(c.a.g, c.s0));
    const double dt = 0.5 * da->stableDt(*sa, *red);
    for (int it = 0; it < kSteps; ++it) {
        da->step(*sa, gpu::Real(dt));
        db->step(*sb, gpu::Real(dt));
    }
    auto ha = CompState::zeros(c.a.g), hb = CompState::zeros(c.b.g);
    (void)sa->download(ha);
    (void)sb->download(hb);
    const double e = permtest::relState(hb, c.P(c.a.g, ha));
    std::printf("GPU %-26s dt %.3e  max rel |step(P s) - P step(s)| %.3e\n",
                tag.c_str(), dt, e);
    check(e < 1e-6, "GPU " + tag + ": step(P s) == P step(s) to 1e-6");
}

// 2D: focal in x on (48 x 40, 10 x 8) vs focal in y on the transpose.
Pair focal2d()
{
    const auto ga = Grid::make(48, 40, 1, 10.0, 8.0, 1.0);
    const auto gb = Grid::make(40, 48, 1, 8.0, 10.0, 1.0);
    const Focal1D f = Focal1D::make(10.0, 3.5, 6.0, 1.5, 3.0);
    const AxisRemap3 ra{f, Focal1D::identity(8.0), Focal1D::identity(1.0)};
    const AxisRemap3 rb{Focal1D::identity(8.0), f, Focal1D::identity(1.0)};
    Pair p{CurvMesh::fromRemap(ga, ra), CurvMesh::fromRemap(gb, rb), {},
           permtest::transpose2d};
    p.s0 = vortex(p.a);
    return p;
}

// 3D: ratios 2, 3, 4 on x, y, z of a (24, 20, 16) box of extents (3, 2.5, 2),
// and its cyclic rotation (old x -> new y, y -> z, z -> x).
Pair focal3d()
{
    const auto ga = Grid::make(24, 20, 16, 3.0, 2.5, 2.0);
    const auto gb = permtest::cyclicGrid(ga);
    const Focal1D fx = Focal1D::make(3.0, 1.0, 1.8, 0.4, 2.0);
    const Focal1D fy = Focal1D::make(2.5, 0.8, 1.5, 0.35, 3.0);
    const Focal1D fz = Focal1D::make(2.0, 0.6, 1.2, 0.3, 4.0);
    const AxisRemap3 ra{fx, fy, fz};
    const AxisRemap3 rb{fz, fx, fy};
    Pair p{CurvMesh::fromRemap(ga, ra), CurvMesh::fromRemap(gb, rb), {},
           permtest::cyclic};
    p.s0 = curvtest::smoothState(p.a, kGamma);
    return p;
}

} // namespace

int main()
{
    const Pair p2 = focal2d();
    const Pair p3 = focal3d();
    cpuCase("2D focal-x vs focal-y", p2);
    cpuCase("3D ratios 2/3/4 cyclic", p3);
    if (!gpu::deviceAvailable()) {
        if (dnstest::gFailures) return dnstest::finish();
        dnstest::skip("no HIP device (CPU half passed)");
    }
    gpuCase("2D focal-x vs focal-y", p2);
    gpuCase("3D ratios 2/3/4 cyclic", p3);
    return dnstest::finish();
}
