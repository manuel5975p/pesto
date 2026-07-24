// GPU curvilinear compressible solver vs the CPU reference: stableDt and
// step parity on a distorted mesh, uniform-mesh degeneracy against the
// Cartesian device path, FP32 freestream preservation and conservation
// drift, and a short 3D distorted viscous run.
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "curv_test_util.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace dns;
using namespace curvtest;

namespace {

int failures = 0;

void check(bool ok, const char* what)
{
    std::printf("%-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

bool parityDistorted2d()
{
    const int n = 64;
    const auto g = Grid::make(n, n, 1, 10.0, 10.0, 10.0 / n);
    const auto mesh = CurvMesh::sinusoidal(g, 0.04);
    const auto cpu = Comp::make(mesh, 1.4, 2e-3);
    if (!cpu) { check(false, "CPU curvilinear make"); return false; }
    auto sc = vortexState(mesh, 5.0, 1.4);

    auto dev = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3);
    if (!dev) return false;
    check(dev->curvilinear(), "device solver reports curvilinear");
    auto ds = gpu::DeviceCompState::make(g.size());
    auto scratch = gpu::ReduceScratch::make();
    if (!ds || !scratch) { check(false, "GPU allocation"); return true; }
    (void)ds->upload(sc);

    const double dtc = cpu->stableDt(sc);
    const double dtg = dev->stableDt(*ds, *scratch);
    std::printf("dt cpu %.6e gpu %.6e\n", dtc, dtg);
    check(std::abs(dtc - dtg) < 1e-4 * dtc, "stableDt parity on distorted mesh");

    const double dt = 0.9 * dtc;
    for (int it = 0; it < 20; ++it) {
        cpu->step(sc, dt);
        dev->step(*ds, gpu::Real(dt));
    }
    auto sg = CompState::zeros(g);
    (void)ds->download(sg);
    const double er = relLinf(sg.rho, sc.rho), em = relLinf(sg.mx, sc.mx),
                 ee = relLinf(sg.E, sc.E);
    std::printf("20-step rel Linf: rho %.2e mx %.2e E %.2e\n", er, em, ee);
    check(er < 1e-4 && em < 1e-4 && ee < 1e-4,
          "20-step CPU/GPU parity within FP32");

    // FP32 conservation drift of the volume-weighted totals over 100 steps.
    const auto t0 = totals(*cpu->curv, sg);
    for (int it = 0; it < 100; ++it) dev->step(*ds, gpu::Real(dt));
    (void)ds->download(sg);
    const auto t1 = totals(*cpu->curv, sg);
    bool ok = true;
    for (int q = 0; q < 5; ++q)
        if (std::abs(t1[q] - t0[q]) >
            1e-4 * std::max(std::abs(t0[q]), double(g.size()) * g.cellVolume()))
            ok = false;
    std::printf("FP32 drift: rho %.2e mx %.2e E %.2e\n", t1[0] - t0[0],
                t1[1] - t0[1], t1[4] - t0[4]);
    check(ok, "FP32 conservation drift bounded on distorted mesh");
    return true;
}

void uniformDegeneracy()
{
    const auto g = Grid::make(32, 32, 1, 10.0, 10.0, 10.0 / 32);
    const auto mesh = CurvMesh::uniform(g);
    const auto cpu = Comp::make(mesh, 1.4, 2e-3);
    if (!cpu) { check(false, "uniform curvilinear make"); return; }
    const auto s0 = vortexState(mesh, 5.0, 1.4);

    auto devCart = gpu::DeviceComp::make(g, 1.4, 2e-3);
    auto devCurv = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3);
    auto da = gpu::DeviceCompState::make(g.size());
    auto db = gpu::DeviceCompState::make(g.size());
    if (!devCart || !devCurv || !da || !db) { check(false, "GPU alloc"); return; }
    (void)da->upload(s0);
    (void)db->upload(s0);
    const double dt = 0.9 * cpu->stableDt(s0);
    for (int it = 0; it < 5; ++it) {
        devCart->step(*da, gpu::Real(dt));
        devCurv->step(*db, gpu::Real(dt));
    }
    auto sa = CompState::zeros(g), sb = CompState::zeros(g);
    (void)da->download(sa);
    (void)db->download(sb);
    const double e = std::max({relLinf(sb.rho, sa.rho), relLinf(sb.mx, sa.mx),
                               relLinf(sb.my, sa.my), relLinf(sb.E, sa.E)});
    std::printf("uniform-mesh curv vs Cartesian device rel Linf %.2e\n", e);
    check(e < 1e-4, "uniform-mesh device path matches Cartesian device path");
}

void freestream3d()
{
    const auto g = Grid::make(24, 24, 24, 2.0, 2.0, 2.0);
    const auto mesh = CurvMesh::sinusoidal(g, 0.05);
    const auto cpu = Comp::make(mesh, 1.4, 2e-3);
    if (!cpu) { check(false, "3D distorted make"); return; }
    auto s = CompState::zeros(g);
    const double rho = 1.0, u = 1.0, v = 0.5, w = 0.25, p = 1.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        s.rho[i] = rho;
        s.mx[i] = rho * u;
        s.my[i] = rho * v;
        s.mz[i] = rho * w;
        s.E[i] = p / 0.4 + 0.5 * rho * (u * u + v * v + w * w);
    }
    auto dev = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto scratch = gpu::ReduceScratch::make();
    if (!dev || !ds || !scratch) { check(false, "GPU alloc"); return; }
    (void)ds->upload(s);
    const double dt = 0.9 * dev->stableDt(*ds, *scratch);
    for (int it = 0; it < 50; ++it) dev->step(*ds, gpu::Real(dt));
    auto s2 = CompState::zeros(g);
    (void)ds->download(s2);
    double e = 0.0;
    for (std::size_t i = 0; i < g.size(); ++i)
        e = std::max({e, std::abs(s2.rho[i] - rho), std::abs(s2.mx[i] - rho * u),
                      std::abs(s2.my[i] - rho * v), std::abs(s2.mz[i] - rho * w),
                      std::abs(s2.E[i] - s.E[0])});
    std::printf("3D freestream 50-step max deviation %.2e (dt %.3e)\n", e, dt);
    check(e < 1e-4, "FP32 freestream preserved on 3D distorted mesh");
}

// Wall-normal tanh-stretched mesh (TanhStretchY remap): CPU/GPU stableDt
// parity and FP32 freestream preservation. 2D and small by design.
void stretched2d()
{
    const auto g = Grid::make(48, 48, 1, 2.0, 1.0, 1.0);
    const auto mesh = CurvMesh::stretchedY(g, 2.0);
    const auto cpu = Comp::make(mesh, 1.4, 2e-3);
    if (!cpu) { check(false, "CPU stretched make"); return; }
    auto s = CompState::zeros(g);
    const double rho = 1.0, u = 1.0, v = 0.5, p = 1.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        s.rho[i] = rho;
        s.mx[i] = rho * u;
        s.my[i] = rho * v;
        s.E[i] = p / 0.4 + 0.5 * rho * (u * u + v * v);
    }
    auto dev = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto scratch = gpu::ReduceScratch::make();
    if (!dev || !ds || !scratch) { check(false, "GPU alloc"); return; }
    (void)ds->upload(s);
    const double dtc = cpu->stableDt(s);
    const double dtg = dev->stableDt(*ds, *scratch);
    std::printf("stretched dt cpu %.6e gpu %.6e\n", dtc, dtg);
    check(std::abs(dtc - dtg) < 1e-4 * dtc, "stableDt parity on stretched mesh");
    for (int it = 0; it < 50; ++it) dev->step(*ds, gpu::Real(0.9 * dtg));
    auto s2 = CompState::zeros(g);
    (void)ds->download(s2);
    double e = 0.0;
    bool mzZero = true;
    for (std::size_t i = 0; i < g.size(); ++i) {
        e = std::max({e, std::abs(s2.rho[i] - rho), std::abs(s2.mx[i] - rho * u),
                      std::abs(s2.my[i] - rho * v), std::abs(s2.E[i] - s.E[0])});
        if (s2.mz[i] != 0.0) mzZero = false;
    }
    std::printf("stretched freestream 50-step max deviation %.2e\n", e);
    check(e < 1e-4, "FP32 freestream preserved on stretched mesh");
    check(mzZero, "nz == 1: mz stays exactly zero on stretched mesh");
}

// Focal mesh (AxisRemap3, x and y clustered on an interior plateau):
// CPU/GPU stableDt parity, FP32 freestream preservation, the exact nz == 1
// restriction, and the degenerate identity against the Cartesian device path.
void focal2d()
{
    const double lx = 4.0, ly = 4.0;
    const auto g = Grid::make(64, 64, 1, lx, ly, 1.0);
    const AxisRemap3 R{Focal1D::make(lx, 1.2, 2.8, 0.8, 4.0),
                       Focal1D::make(ly, 1.4, 2.6, 0.6, 4.0),
                       Focal1D::identity(1.0)};
    const auto mesh = CurvMesh::fromRemap(g, R);
    const auto cpu = Comp::make(mesh, 1.4, 2e-3);
    if (!cpu) { check(false, "CPU focal make"); return; }

    auto s = CompState::zeros(g);
    const double rho = 1.0, u = 1.0, v = 0.5, p = 1.0;
    for (std::size_t i = 0; i < g.size(); ++i) {
        s.rho[i] = rho;
        s.mx[i] = rho * u;
        s.my[i] = rho * v;
        s.E[i] = p / 0.4 + 0.5 * rho * (u * u + v * v);
    }
    auto dev = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto scratch = gpu::ReduceScratch::make();
    if (!dev || !ds || !scratch) { check(false, "GPU alloc"); return; }
    (void)ds->upload(s);
    const double dtc = cpu->stableDt(s);
    const double dtg = dev->stableDt(*ds, *scratch);
    std::printf("focal dt cpu %.6e gpu %.6e\n", dtc, dtg);
    check(std::abs(dtc - dtg) < 1e-4 * dtc, "stableDt parity on focal mesh");
    // The acoustic limit follows the FINE plateau spacing, not h: the whole
    // cost model of clustering. hmin must be the fine x/y spacing.
    const double hf = std::min(R.fx.c * R.fx.r * g.hx, R.fy.c * R.fy.r * g.hy);
    std::printf("focal hmin %.6e, fine spacing %.6e\n", cpu->curv->hmin, hf);
    check(std::abs(cpu->curv->hmin - hf) < 1e-9 * hf,
          "hmin is the fine plateau spacing");
    for (int it = 0; it < 50; ++it) dev->step(*ds, gpu::Real(0.9 * dtg));
    auto s2 = CompState::zeros(g);
    (void)ds->download(s2);
    double e = 0.0;
    bool mzZero = true;
    for (std::size_t i = 0; i < g.size(); ++i) {
        e = std::max({e, std::abs(s2.rho[i] - rho), std::abs(s2.mx[i] - rho * u),
                      std::abs(s2.my[i] - rho * v), std::abs(s2.E[i] - s.E[0])});
        if (s2.mz[i] != 0.0) mzZero = false;
    }
    std::printf("focal freestream 50-step max deviation %.2e\n", e);
    check(e < 1e-4, "FP32 freestream preserved on focal mesh");
    check(mzZero, "nz == 1: mz stays exactly zero on focal mesh");

    // Inactive focal maps: the curvilinear device path must reproduce the
    // Cartesian one exactly (same pin as uniformDegeneracy, via fromRemap).
    const AxisRemap3 I{Focal1D::identity(lx), Focal1D::identity(ly),
                       Focal1D::identity(1.0)};
    const auto um = CurvMesh::fromRemap(g, I);
    const auto ucpu = Comp::make(um, 1.4, 2e-3);
    if (!ucpu) { check(false, "CPU identity-focal make"); return; }
    auto dc = gpu::DeviceComp::make(g, 1.4, 2e-3);
    auto dk = gpu::DeviceComp::make(*ucpu->curv, 1.4, 2e-3);
    auto sa = gpu::DeviceCompState::make(g.size());
    auto sb = gpu::DeviceCompState::make(g.size());
    if (!dc || !dk || !sa || !sb) { check(false, "GPU alloc"); return; }
    auto sm = smoothState(um, 1.4);
    (void)sa->upload(sm);
    (void)sb->upload(sm);
    const double dt = 0.5 * dtg;
    for (int it = 0; it < 10; ++it) {
        dc->step(*sa, gpu::Real(dt));
        dk->step(*sb, gpu::Real(dt));
    }
    auto ha = CompState::zeros(g), hb = CompState::zeros(g);
    (void)sa->download(ha);
    (void)sb->download(hb);
    const double d = std::max({relLinf(hb.rho, ha.rho), relLinf(hb.mx, ha.mx),
                               relLinf(hb.my, ha.my), relLinf(hb.E, ha.E)});
    std::printf("identity-focal vs Cartesian rel Linf %.2e\n", d);
    check(d < 1e-5, "inactive focal maps reproduce the Cartesian path");
}

void distorted3dRun()
{
    const auto g = Grid::make(24, 24, 24, 2.0, 2.0, 2.0);
    const auto mesh = CurvMesh::sinusoidal(g, 0.05);
    const auto cpu = Comp::make(mesh, 1.4, 5e-3);
    if (!cpu) { check(false, "3D distorted make"); return; }
    auto s = smoothState(mesh, 1.4);
    auto dev = gpu::DeviceComp::make(*cpu->curv, 1.4, 5e-3);
    auto ds = gpu::DeviceCompState::make(g.size());
    auto scratch = gpu::ReduceScratch::make();
    if (!dev || !ds || !scratch) { check(false, "GPU alloc"); return; }
    (void)ds->upload(s);
    const auto t0 = totals(*cpu->curv, s);
    double dt = 0.9 * dev->stableDt(*ds, *scratch);
    for (int it = 0; it < 50; ++it) {
        dev->step(*ds, gpu::Real(dt));
        if (it % 16 == 15) dt = 0.9 * dev->stableDt(*ds, *scratch);
    }
    auto s2 = CompState::zeros(g);
    (void)ds->download(s2);
    bool finite = true;
    double rhomin = 1e300;
    for (std::size_t i = 0; i < g.size(); ++i) {
        if (!std::isfinite(s2.rho[i]) || !std::isfinite(s2.E[i])) finite = false;
        rhomin = std::min(rhomin, s2.rho[i]);
    }
    const auto t1 = totals(*cpu->curv, s2);
    double drift = 0.0;
    for (int q = 0; q < 5; ++q)
        drift = std::max(drift, std::abs(t1[q] - t0[q]) /
                                    std::max(std::abs(t0[q]), 1.0));
    std::printf("3D distorted run: rhomin %.3f, conservation drift %.2e\n",
                rhomin, drift);
    check(finite && rhomin > 0.0, "3D distorted viscous run stays physical");
    check(drift < 1e-4, "3D distorted run conserves the totals");
}

} // namespace

int main()
{
    test::requireDeviceOrSkip();
    if (!parityDistorted2d()) return test::skipExitCode;
    uniformDegeneracy();
    freestream3d();
    stretched2d();
    focal2d();
    distorted3dRun();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
