// GPU curvilinear compressible solver vs the CPU reference: stableDt and
// step parity on a distorted mesh, uniform-mesh degeneracy against the
// Cartesian device path, FP32 freestream preservation and conservation
// drift, a short 3D distorted viscous run, and the curvilinear LES
// (both SGS operators; the CPU reference has no LES, so the pins are
// against the Cartesian device path and the model's own invariants).
#include "dns/comp.hpp"
#include "dns/curvmesh.hpp"
#include "dns/gpu/comp.hpp"
#include "curv_test_util.hpp"
#include "gpu_test_util.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace dns;
using namespace curvtest;

namespace {

using dnstest::check;



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

    const double dtc = *cpu->stableDt(sc);
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
    const double dt = 0.9 * *cpu->stableDt(s0);
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
    const double dtc = *cpu->stableDt(s);
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
    const double dtc = *cpu->stableDt(s);
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

// Kinetic energy of a downloaded state, unweighted (the mesh is the same on
// both sides of every comparison below).
double kineticEnergy(const CompState& s)
{
    double k = 0.0;
    for (std::size_t i = 0; i < s.rho.size(); ++i)
        k += 0.5 * (s.mx[i] * s.mx[i] + s.my[i] * s.my[i] + s.mz[i] * s.mz[i]) /
             s.rho[i];
    return k;
}

// The LES viscous dt limit, 0.25 h^2 rho_min / (mu + nut_max), must be the
// SAME on the identity-map curvilinear path as on the Cartesian one: same
// nut_max reduction, one through min(h), the other through the mesh hmin. The
// acoustic bound is not comparable between the two paths (see lesCurv), so the
// model constant is exaggerated until the viscous limit provably binds -- the
// check asserts that precondition rather than assuming it.
void lesViscousDtDegeneracy()
{
    const auto g = Grid::make(24, 24, 24, 2.0, 2.0, 2.0);
    const auto mesh = CurvMesh::uniform(g);
    const auto cpu = Comp::make(mesh, 1.4, 2e-3);
    if (!cpu) { check(false, "uniform curvilinear LES-dt make"); return; }
    const auto s0 = smoothState(mesh, 1.4);
    const SgsModel models[2] = {SgsModel::Smagorinsky, SgsModel::Sigma};
    const double cbig[2] = {20.0, 60.0}; // D_sigma << |S| on a smooth field
    for (int q = 0; q < 2; ++q) {
        auto dCart = gpu::DeviceComp::make(g, 1.4, 2e-3, cbig[q], models[q]);
        auto dCurv =
            gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3, cbig[q], models[q]);
        auto dnsCart = gpu::DeviceComp::make(g, 1.4, 2e-3);
        auto dnsCurv = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3);
        auto da = gpu::DeviceCompState::make(g.size());
        auto scratch = gpu::ReduceScratch::make();
        if (!dCart || !dCurv || !dnsCart || !dnsCurv || !da || !scratch) {
            check(false, "GPU alloc");
            return;
        }
        (void)da->upload(s0);
        const double dta = dCart->stableDt(*da, *scratch);
        const double dtb = dCurv->stableDt(*da, *scratch);
        const double acoustic =
            std::min(dnsCart->stableDt(*da, *scratch),
                     dnsCurv->stableDt(*da, *scratch));
        std::printf("LES viscous dt: cart %.6e curv %.6e (acoustic %.6e)\n",
                    dta, dtb, acoustic);
        check(dta < 0.5 * acoustic && dtb < 0.5 * acoustic,
              q == 0 ? "smag: the LES viscous limit binds"
                     : "sigma: the LES viscous limit binds");
        check(std::abs(dta - dtb) < 1e-4 * dta,
              q == 0 ? "curv LES smag viscous dt matches Cartesian"
                     : "curv LES sigma viscous dt matches Cartesian");
    }
}

// LES on the curvilinear path. Three pins:
//   (a) on the IDENTITY map the metric-gradient / per-cell-Delta model must
//       reproduce the uniform-mesh LES, for BOTH operators -- that is the
//       whole claim that this is the same model, not a new one;
//   (b) at nz == 1 the sigma model must stay identically zero on a DISTORTED
//       mesh (its defining 2D zero survives the metric contraction), which
//       makes the run bit-identical to molecular DNS;
//   (c) on a genuinely clustered 3D focal mesh both operators run, stay
//       physical, and dissipate.
void lesCurv()
{
    const double cs = 0.17, csig = 1.35;

    { // (a) identity degeneracy, 3D, both operators
        const auto g = Grid::make(24, 24, 24, 2.0, 2.0, 2.0);
        const auto mesh = CurvMesh::uniform(g);
        const auto cpu = Comp::make(mesh, 1.4, 2e-3);
        if (!cpu) { check(false, "uniform curvilinear LES make"); return; }
        const auto s0 = smoothState(mesh, 1.4);
        const SgsModel models[2] = {SgsModel::Smagorinsky, SgsModel::Sigma};
        const double cval[2] = {cs, csig};
        const char* label[2] = {"uniform-mesh curv LES smag matches Cartesian",
                                "uniform-mesh curv LES sigma matches Cartesian"};
        for (int q = 0; q < 2; ++q) {
            auto dCart = gpu::DeviceComp::make(g, 1.4, 2e-3, cval[q], models[q]);
            auto dCurv = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3, cval[q],
                                               models[q]);
            auto da = gpu::DeviceCompState::make(g.size());
            auto db = gpu::DeviceCompState::make(g.size());
            auto scratch = gpu::ReduceScratch::make();
            if (!dCart || !dCurv || !da || !db || !scratch) {
                check(false, "GPU alloc");
                return;
            }
            (void)da->upload(s0);
            (void)db->upload(s0);
            const double dta = dCart->stableDt(*da, *scratch);
            const double dtb = dCurv->stableDt(*db, *scratch);
            // NOTE the two stableDt values are NOT comparable in general, LES
            // or not: the curvilinear acoustic bound uses |u| where the
            // Cartesian one uses |u . n| (deliberately conservative on skewed
            // cells), so they differ by a few percent whenever the acoustic
            // limit binds. lesViscousDtDegeneracy() below pins the LES half of
            // the bound, where they must agree exactly.
            const double dt = 0.5 * std::min(dta, dtb);
            for (int it = 0; it < 10; ++it) {
                dCart->step(*da, gpu::Real(dt));
                dCurv->step(*db, gpu::Real(dt));
            }
            auto ha = CompState::zeros(g), hb = CompState::zeros(g);
            (void)da->download(ha);
            (void)db->download(hb);
            const double e =
                std::max({relLinf(hb.rho, ha.rho), relLinf(hb.mx, ha.mx),
                          relLinf(hb.my, ha.my), relLinf(hb.mz, ha.mz),
                          relLinf(hb.E, ha.E)});
            std::printf("curv-vs-Cartesian LES %s rel Linf %.2e\n",
                        q == 0 ? "smag " : "sigma", e);
            check(e < 1e-4, label[q]);
        }
    }

    { // (b) sigma is exactly zero at nz == 1, on a DISTORTED mesh
        const auto g = Grid::make(48, 48, 1, 4.0, 4.0, 4.0 / 48);
        const auto mesh = CurvMesh::sinusoidal(g, 0.04);
        const auto cpu = Comp::make(mesh, 1.4, 2e-3);
        if (!cpu) { check(false, "2D distorted LES make"); return; }
        const auto s0 = smoothState(mesh, 1.4);
        auto dns = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3);
        auto les = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3, csig,
                                         SgsModel::Sigma);
        auto da = gpu::DeviceCompState::make(g.size());
        auto db = gpu::DeviceCompState::make(g.size());
        auto scratch = gpu::ReduceScratch::make();
        if (!dns || !les || !da || !db || !scratch) {
            check(false, "GPU alloc");
            return;
        }
        (void)da->upload(s0);
        (void)db->upload(s0);
        const double dt = 0.5 * dns->stableDt(*da, *scratch);
        for (int it = 0; it < 20; ++it) {
            dns->step(*da, gpu::Real(dt));
            les->step(*db, gpu::Real(dt));
        }
        auto ha = CompState::zeros(g), hb = CompState::zeros(g);
        (void)da->download(ha);
        (void)db->download(hb);
        bool same = true;
        for (std::size_t i = 0; i < g.size(); ++i)
            if (ha.rho[i] != hb.rho[i] || ha.mx[i] != hb.mx[i] ||
                ha.my[i] != hb.my[i] || ha.E[i] != hb.E[i])
                same = false;
        check(same, "nz == 1 distorted mesh: sigma LES is bit-identical to DNS");
    }

    { // (c) clustered 3D focal mesh: both operators run and dissipate
        const double lx = 4.0, ly = 4.0, lz = 4.0;
        const auto g = Grid::make(32, 32, 32, lx, ly, lz);
        const AxisRemap3 R{Focal1D::make(lx, 1.2, 2.8, 0.8, 3.0),
                           Focal1D::make(ly, 1.4, 2.6, 0.6, 3.0),
                           Focal1D::make(lz, 1.4, 2.6, 0.6, 3.0)};
        const auto mesh = CurvMesh::fromRemap(g, R);
        const auto cpu = Comp::make(mesh, 1.4, 2e-3);
        if (!cpu) { check(false, "3D focal LES make"); return; }
        const auto s0 = smoothState(mesh, 1.4);
        const double k0 = kineticEnergy(s0);
        const SgsModel models[2] = {SgsModel::Smagorinsky, SgsModel::Sigma};
        const double cval[2] = {cs, csig};
        for (int q = 0; q < 2; ++q) {
            auto dns = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3);
            auto les = gpu::DeviceComp::make(*cpu->curv, 1.4, 2e-3, cval[q],
                                             models[q]);
            auto da = gpu::DeviceCompState::make(g.size());
            auto db = gpu::DeviceCompState::make(g.size());
            auto scratch = gpu::ReduceScratch::make();
            if (!dns || !les || !da || !db || !scratch) {
                check(false, "GPU alloc");
                return;
            }
            (void)da->upload(s0);
            (void)db->upload(s0);
            const double dtDns = dns->stableDt(*da, *scratch);
            const double dtLes = les->stableDt(*db, *scratch);
            check(dtLes > 0.0 && dtLes <= dtDns * (1.0 + 1e-9),
                  q == 0 ? "focal LES smag tightens the dt limit"
                         : "focal LES sigma tightens the dt limit");
            const double dt = 0.5 * dtLes;
            for (int it = 0; it < 40; ++it) {
                dns->step(*da, gpu::Real(dt));
                les->step(*db, gpu::Real(dt));
            }
            auto ha = CompState::zeros(g), hb = CompState::zeros(g);
            (void)da->download(ha);
            (void)db->download(hb);
            bool finite = true;
            double rhomin = 1e300;
            for (std::size_t i = 0; i < g.size(); ++i) {
                if (!std::isfinite(hb.rho[i]) || !std::isfinite(hb.E[i]))
                    finite = false;
                rhomin = std::min(rhomin, hb.rho[i]);
            }
            const double kd = kineticEnergy(ha), kl = kineticEnergy(hb);
            std::printf("focal 3D %s: K0 %.6f DNS %.6f LES %.6f\n",
                        q == 0 ? "smag " : "sigma", k0, kd, kl);
            check(finite && rhomin > 0.0,
                  q == 0 ? "focal 3D LES smag run stays physical"
                         : "focal 3D LES sigma run stays physical");
            check(kl < kd, q == 0 ? "focal 3D LES smag drains kinetic energy"
                                  : "focal 3D LES sigma drains kinetic energy");
        }
    }
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
    lesCurv();
    lesViscousDtDegeneracy();
    distorted3dRun();
    return dnstest::gFailures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
