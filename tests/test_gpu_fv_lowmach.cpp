// The low-Mach reacting FV solver (docs/fvincomp.md "Low-Mach", DeviceFvLowMach).
//   L1 uniform power-of-two Y exact to the bit, FP32 compensated and FP64, burning or
//      not, on every mesh: rho Y == Y rho
//   L2 mass and species: inventory vs FP64 tally, burning march, y + burn == tally
//   L3 energy: the semi-discrete balance to rounding; the march's closes at the time
//      scheme's order (dt, dt/2)
//   L4 rerun byte-identical, both precisions
//   L5 2D: rho w exactly 0
//   L6 freestream on a non-orthogonal mesh
//   L7 vs gpu/lowmach, same mesh, same dt: rounding, <= 3 x the FP32 run's own
//      distance from FP64
//   L8 (`test_gpu_fv_lowmach momentum`, 150 s) the momentum path: wrinkled front, FV -
//      MAC rho v shrinks with h
//   flame1d (`test_gpu_fv_lowmach flame1d`): the binary route, 7.5 cells per delta_th
//   R1-R3 two reactants (model C2, C2b): Z kept, fuel/oxidizer tallies, energy rate
//   R7 the two-reactant C2b stage vs gpu/lowmach, same mesh, same dt: rounding
//   B1/B2 per-face inflow rows (the burner's): each face carries its row, tallies close
//   bunsen (`test_gpu_fv_lowmach bunsen`): the binary's 2D slot burner vs structured
//   burner (`test_gpu_fv_lowmach burner`, M3 steps 3-5):
//   O1/G1 a quiescent open box stays at rest exactly, with and without gravity
//   O2 a vortex through an open face: KE never grows with the backflow term, does without
//   O3 open faces: uniform Y bitwise, tallies, an air ambient's oxygen
//   O4/G3 vs gpu/lowmach with open laterals (and gravity), same cells, same dt: shrinks
//   G2 a stratified column under g = the column at g 0 (box; sheared, random measured)
//   R1 the tube as a rim Wall patch vs the structured staircase on the same cells
//   R2 the rim's conduction: the linear profile is its fixed point
#include "dns/chem.hpp"
#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvincomp.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/lowmach.hpp"
#include "dns/la/graph.hpp"
#include "dns/polymesh.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <limits>
#include <numbers>
#include <span>
#include <string>
#include <vector>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif
#ifndef DNS_BIN
#error "DNS_BIN must be the dns binary"
#endif

using namespace dns;
using dnstest::check;

namespace {

constexpr double kSigma = 5, kNoRate = -std::numeric_limits<double>::infinity();

template <class T>
T orDie(std::expected<T, std::string> e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL: %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    return std::move(*e);
}

struct Mesh
{
    std::string name;
    FaceMesh fm;
    FvLowMachBcs b;
};

StructuredSpec openSpec(const Grid& g)
{
    return {.lo = {0, 0, 0},
            .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
            .type = {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                     FoamPatchType::Empty, FoamPatchType::Empty}};
}

FvLowMachBcs boxBcs(double y0, double y1)
{
    FvBcSpec s = fvLowMachBoxBc();
    s.patches[0].y = {y0, y1, 0, 0};
    return orDie(fvLowMachBcs(s, 0.98, 1.0), "box bcs");
}

// The open box, a non-orthogonal box, a snappy mesh with a wall, slip sides and every
// kind of face. Fresh mixture y0, y1 at the inflows.
std::vector<Mesh> meshes(double y0, double y1)
{
    std::vector<Mesh> v;
    const Grid g = Grid::make(96, 8, 1, 24.0, 2.0, 1.0);
    v.push_back({"box", orDie(fvLowMachBox(g), "box"), boxBcs(y0, y1)});
    v.push_back({"sine", orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(g, 0.04), openSpec(g), CellOrder::Morton), "sine"),
                 boxBcs(y0, y1)});
    {
        namespace fs = std::filesystem;
        const PolyMesh pm = orDie(readPolyMesh(fs::path(DNS_MESH_DATA) / "cyl2d_snappy/constant/polyMesh"), "snappy");
        FvBcSpec s;
        s.patches = {{.name = "cylinder", .kind = FvPatchKind::Wall, .isothermal = true, .tWall = 1.5},
                     {.name = "inlet", .kind = FvPatchKind::Inflow, .hasY = true, .y = {y0, y1, 0, 0}},
                     {.name = "outlet", .kind = FvPatchKind::Outflow},
                     {.name = "lower", .kind = FvPatchKind::Slip},
                     {.name = "upper", .kind = FvPatchKind::Slip}};
        v.push_back({"snappy", orDie(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), "snappy"),
                     orDie(fvLowMachBcs(s, 0.98, 1.0), "snappy bcs")});
    }
    return v;
}

// A warm front along x at the middle of the mesh.
FvLowMachIC frontIC(const FaceMesh& fm, double y0, double y1, bool coupled)
{
    const std::size_t nC = std::size_t(fm.nC);
    const auto [lo, hi] = std::ranges::minmax(fm.C[0]);
    const double xc = 0.5 * (lo + hi), w = 0.08 * (hi - lo);
    FvLowMachIC ic{.rho = std::vector<double>(nC),
                   .ry0 = std::vector<double>(nC),
                   .ry1 = std::vector<double>(nC),
                   .ru = std::vector<double>(3 * nC),
                   .p = std::vector<double>(nC)};
    for (std::size_t c = 0; c < nC; ++c) {
        const double T = 1.0 + 0.5 * (kSigma - 1.0) * (1.0 + std::tanh((fm.C[0][c] - xc) / w));
        ic.rho[c] = 1.0 / T;
        ic.ry0[c] = ic.rho[c] * (coupled ? y0 * (kSigma - T) / (kSigma - 1.0) : y0);
        ic.ry1[c] = ic.rho[c] * y1;
        ic.ru[c] = 0.98;
    }
    return ic;
}

FvLowMachModel modelOf(double lnA, int nY)
{
    return {.tr = Transport::air(0.72, 1.0, 300.0, 0.72, kGammaAir, kRgasBox, false),
            .prGas = 1.0,
            .lnA = lnA,
            .Ta = 8.0,
            .qcp = kSigma - 1.0,
            .nY = nY};
}

struct Run
{
    FvIncompSetup s;
    FvLowMachTables t;
    la::CellGraph g;
};

Run setupOf(Mesh& m)
{
    if (auto b = bindFvIncompBcs(m.b.flow, m.fm); !b) {
        std::printf("FATAL: bind %s: %s\n", m.name.c_str(), b.error().c_str());
        std::exit(1);
    }
    FvIncompSetup s = orDie(buildFvIncompSetup(m.fm, m.b.flow), m.name);
    FvLowMachTables t = buildFvLowMachTables(m.fm, m.b, s, 1.0);
    return {.s = std::move(s), .t = std::move(t), .g = la::graphOf(m.fm)};
}

using gpu::DeviceFvLowMach;

// n steps at a dt frozen off the first bound: the march of a gate.
void march(DeviceFvLowMach& d, int n, double frac = 0.5)
{
    const auto b = orDie(d.stableDt(0.8, 1.0), "dt");
    const double dt = frac * std::min({b.conv, b.diff, b.chem});
    for (int i = 0; i < n; ++i) orDie(d.step(dt), "step");
}

FvIncompControls ctlOf(FvPrecision p) { return {.tol = p == FvPrecision::Fp64 ? 1e-10 : 1e-6, .precision = p}; }

void gateUniform()
{
    for (bool burn : {false, true})
        for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64})
            for (Mesh& m : meshes(burn ? 1.0 : 0.5, 0.25)) {
                Run r = setupOf(m);
                const FvLowMachIC ic = frontIC(m.fm, burn ? 1.0 : 0.5, 0.25, burn);
                DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, modelOf(burn ? 2.0 : kNoRate, 2), ctlOf(p), ic), "make");
                march(d, 40);
                const auto c = orDie(d.cells(), "cells");
                std::size_t bad0 = 0, bad1 = 0;
                double worst = 0;
                for (std::size_t q = 0; q < c.rho.size(); ++q) {
                    bad0 += !burn && c.ry[0][q] != 0.5 * c.rho[q];
                    bad1 += c.ry[1][q] != 0.25 * c.rho[q];
                    if (!burn) worst = std::max(worst, std::abs(c.ry[0][q] / c.rho[q] - 0.5));
                }
                if (bad0 + bad1 > 0) std::printf("  worst |Y - 0.5| %.3e\n", worst);
                std::printf("L1 %s %s %s: %zu, %zu cells off\n", m.name.c_str(), p == FvPrecision::Fp32 ? "fp32" : "fp64",
                            burn ? "burning" : "inert", bad0, bad1);
                check(bad0 == 0 && bad1 == 0, "L1 uniform Y bitwise: " + m.name);
            }
}

void gateConservation()
{
    for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64})
        for (Mesh& m : meshes(1.0, 0.25)) {
            Run r = setupOf(m);
            const FvLowMachIC ic = frontIC(m.fm, 1.0, 0.25, true);
            DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, modelOf(2.0, 2), ctlOf(p), ic), "make");
            const gpu::FvLowMachBudget b0 = orDie(d.budget(), "budget");
            march(d, 40);
            const gpu::FvLowMachTotals t = orDie(d.totals(), "totals"); // budget() rewrites the last stage's D
            const gpu::FvLowMachBudget b1 = orDie(d.budget(), "budget");
            const double em = std::abs(t.mass - t.massTally) / t.mass;
            const double ey0 = std::abs(t.y[0] + t.burn - t.yTally[0]) / t.y[0];
            const double ey1 = std::abs(t.y[1] - t.yTally[1]) / t.y[1];
            const double eE = std::max(std::abs(b0.defect) / b0.gross, std::abs(b1.defect) / b1.gross);
            std::printf("L2/L3 %s %s: mass %.2e, fuel %.2e, passive %.2e, burnt %.3g, energy rate %.2e, march %.2e (eos %.2e, "
                        "div %.2e)\n",
                        m.name.c_str(), p == FvPrecision::Fp32 ? "fp32" : "fp64", em, ey0, ey1, t.burn, eE,
                        std::abs(t.enthalpy - t.enthalpyTally) / t.enthalpy, t.eosErr, t.divErr);
            const bool f64 = p == FvPrecision::Fp64;
            check(em <= (f64 ? 1e-13 : 1e-7), "L2 mass tally: " + m.name);
            check(ey0 <= (f64 ? 1e-13 : 1e-7), "L2 fuel tally: " + m.name);
            check(ey1 <= (f64 ? 1e-13 : 1e-7), "L2 passive tally: " + m.name);
            check(eE <= (f64 ? 1e-13 : 1e-5), "L3 energy rate: " + m.name);
        }
}

// The march's balance on the burning front: dt, dt/2 to the same time.
void gateEnergyOrder()
{
    double prev = 0;
    for (int f : {1, 2, 4}) {
        Mesh m = std::move(meshes(1.0, 0.25)[0]);
        Run r = setupOf(m);
        const FvLowMachIC ic = frontIC(m.fm, 1.0, 0.25, true);
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, modelOf(2.0, 1), ctlOf(FvPrecision::Fp64), ic), "make");
        const double dt = 2e-3 / f; // below every bound of this box
        for (int i = 0; i < 200 * f; ++i) orDie(d.step(dt), "step");
        const gpu::FvLowMachTotals t = orDie(d.totals(), "totals");
        const double e = std::abs(t.enthalpy - t.enthalpyTally) / t.enthalpy;
        std::printf("L3 march dt/%d: %.3e%s\n", f, e, prev > 0 ? std::format(" (order {:.2f})", std::log2(prev / e)).c_str() : "");
        if (prev > 0) check(std::log2(prev / e) >= 0.95, "L3 march balance closes at order 1: m lags one stage");
        prev = e;
    }
}

void gateRerun()
{
    for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64})
        for (Mesh& m : meshes(1.0, 0.25)) {
            Run r = setupOf(m);
            const FvLowMachIC ic = frontIC(m.fm, 1.0, 0.25, true);
            auto once = [&] {
                DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, modelOf(2.0, 2), ctlOf(p), ic), "make");
                march(d, 40);
                return orDie(d.cells(), "cells");
            };
            const auto a = once(), b = once();
            const bool eq = a.rho == b.rho && a.ry[0] == b.ry[0] && a.ru == b.ru && a.p == b.p && a.phi == b.phi;
            check(eq, "L4 rerun identical: " + m.name);
        }
}

void gate2D()
{
    for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64})
        for (Mesh& m : meshes(1.0, 0.25)) {
            Run r = setupOf(m);
            const FvLowMachIC ic = frontIC(m.fm, 1.0, 0.25, true);
            DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, modelOf(2.0, 2), ctlOf(p), ic), "make");
            march(d, 20);
            const auto c = orDie(d.cells(), "cells");
            const std::size_t n = c.rho.size();
            const bool zero = std::all_of(c.ru.begin() + std::ptrdiff_t(2 * n), c.ru.end(), [](double v) { return v == 0.0; });
            check(zero, "L5 rho w exactly 0: " + m.name);
        }
}

void gateFreestream()
{
    for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
        Mesh m = std::move(meshes(1.0, 0.25)[1]);
        Run r = setupOf(m);
        const std::size_t nC = std::size_t(m.fm.nC);
        FvLowMachIC ic{.rho = std::vector<double>(nC, 1.0),
                       .ry0 = std::vector<double>(nC, 1.0),
                       .ry1 = {},
                       .ru = std::vector<double>(3 * nC, 0.0),
                       .p = std::vector<double>(nC, 0.0)};
        std::fill_n(ic.ru.begin(), nC, 0.98);
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, modelOf(kNoRate, 1), ctlOf(p), ic), "make");
        march(d, 20);
        const auto c = orDie(d.cells(), "cells");
        double e = 0, ev = 0;
        for (std::size_t q = 0; q < nC; ++q) {
            e = std::max(e, std::abs(c.ru[q] - 0.98));
            ev = std::max(ev, std::abs(c.rho[q] - 1.0));
        }
        std::printf("L6 freestream %s: rho u %.2e, rho %.2e\n", p == FvPrecision::Fp32 ? "fp32" : "fp64", e, ev);
        check(e <= (p == FvPrecision::Fp64 ? 1e-12 : 2e-4), "L6 freestream rho u");
        check(ev <= (p == FvPrecision::Fp64 ? 1e-12 : 2e-4), "L6 freestream rho");
    }
}

// Same mesh, same dt, same IC as gpu/lowmach: its algebra on faces.
void gateStructured()
{
    const int nx = 128, ny = 4, steps = 400;
    const double h = 0.25;
    const Grid g = Grid::make(nx, ny, 1, nx * h, ny * h, 1.0);
    LowMachState ls;
    ls.rho.assign(g.size(), 0.0);
    ls.p.assign(g.size(), 0.0);
    ls.my.assign(g.size(), 0.0);
    ls.mz.assign(g.size(), 0.0);
    ls.mx.assign(std::size_t(nx + 1) * ny, 0.98);
    ls.rhoY[0].assign(g.size(), 0.0);
    ls.nY = 1;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double T = 1.0 + 0.5 * (kSigma - 1.0) * (1.0 + std::tanh(((i + 0.5) * h - 16.0) / 1.5));
            ls.rho[g.idx(i, j, 0)] = 1.0 / T;
            ls.rhoY[0][g.idx(i, j, 0)] = (kSigma - T) / (kSigma - 1.0) / T;
        }
    const FvLowMachModel mdl = modelOf(2.0, 1);
    gpu::LowMachSetup sp;
    sp.tr = mdl.tr;
    sp.prGas = 1.0;
    sp.lnA = mdl.lnA;
    sp.Ta = mdl.Ta;
    sp.qcp = mdl.qcp;
    sp.mIn = 0.98;
    auto st = orDie(gpu::DeviceLowMach::make(g, sp), "structured");
    if (auto e = st.upload(ls); !e) check(false, e.error());
    const double dt = 2e-3;
    for (int i = 0; i < steps; ++i)
        if (auto e = st.step(dt); !e) check(false, e.error());
    const LowMachState sf = orDie(st.download(), "download");

    Mesh m{"box", orDie(fvLowMachBox(g), "box"), boxBcs(1.0, 0.0)};
    Run r = setupOf(m);
    const std::size_t nC = g.size();
    FvLowMachIC ic{.rho = std::vector<double>(nC),
                   .ry0 = std::vector<double>(nC),
                   .ry1 = {},
                   .ru = std::vector<double>(3 * nC, 0.0),
                   .p = std::vector<double>(nC, 0.0)};
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t src = std::size_t(m.fm.srcCell[q]);
        ic.rho[q] = ls.rho[src];
        ic.ry0[q] = ls.rhoY[0][src];
        ic.ru[q] = 0.98;
    }
    auto marchFv = [&](FvPrecision p) {
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, mdl, ctlOf(p), ic), "make");
        for (int i = 0; i < steps; ++i) orDie(d.step(dt), "step");
        return orDie(d.cells(), "cells");
    };
    const auto f32 = marchFv(FvPrecision::Fp32), f64 = marchFv(FvPrecision::Fp64);
    double dSt = 0, dSt64 = 0, dOwn = 0;
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t src = std::size_t(m.fm.srcCell[q]);
        dSt = std::max({dSt, std::abs(f32.rho[q] - sf.rho[src]), std::abs(f32.ry[0][q] - sf.rhoY[0][src])});
        dSt64 = std::max({dSt64, std::abs(f64.rho[q] - sf.rho[src]), std::abs(f64.ry[0][q] - sf.rhoY[0][src])});
        dOwn = std::max({dOwn, std::abs(f32.rho[q] - f64.rho[q]), std::abs(f32.ry[0][q] - f64.ry[0][q])});
    }
    std::printf("L7 FV32 vs structured %.3e, FV64 vs structured %.3e, FV32 vs FV64 %.3e\n", dSt, dSt64, dOwn);
    check(dSt <= 3.0 * dOwn, "L7 FV vs structured: rounding");
}

// --- Model C2 / C2b: two reactants (M3 step 1) -----------------------------------
// The rich jet premix of docs/chem.md (phi 1.3, s 3.99, Y_O2 0.233, sigma 6.88, beta 8;
// constant mu so the 128 x 4 box keeps dt 1e-3 under the diffusive bound).
struct Rich
{
    PartialPremixModel pm;
    PhiRateLaw<double> law;
    double lnA = 0;
};

const Rich& rich()
{
    static const Rich r = [] {
        Rich o;
        o.law = orDie(phiRateLaw(o.pm), "phiRateLaw");
        o.lnA = std::log(orDie(premixedEigen2(o.pm), "premixedEigen2").Lambda);
        return o;
    }();
    return r;
}

FvLowMachModel modelRich(bool law)
{
    const Rich& r = rich();
    return {.tr = Transport::air(0.72, 1.0, 300.0, 0.72, kGammaAir, kRgasBox, false),
            .prGas = 1.0,
            .lnA = r.lnA,
            .Ta = r.pm.theta(),
            .qcp = r.pm.qOverCpTu(),
            .nY = 2,
            .ox = 1,
            .stoich = r.pm.s,
            .law = law ? r.law : PhiRateLaw<double>{}};
}

// flameIC2's rule on a tanh front: Y_O = Y_O1 s_p, Y_F = Y_F,rem + Y_O/s.
double richT(double x, double xc, double w) { return 1.0 + 0.5 * (rich().pm.sigma - 1.0) * (1.0 + std::tanh((x - xc) / w)); }

void richYs(double T, double& yF, double& yO)
{
    const PartialPremixModel& pm = rich().pm;
    const double sp = (pm.sigma - T) / (pm.sigma - 1.0);
    yO = pm.yO1() * sp;
    yF = pm.yFrem() + yO / pm.s;
}

FvLowMachIC richIC(const FaceMesh& fm)
{
    const std::size_t nC = std::size_t(fm.nC);
    const auto [lo, hi] = std::ranges::minmax(fm.C[0]);
    const double xc = 0.5 * (lo + hi), w = 0.08 * (hi - lo);
    FvLowMachIC ic{.rho = std::vector<double>(nC),
                   .ry0 = std::vector<double>(nC),
                   .ry1 = std::vector<double>(nC),
                   .ru = std::vector<double>(3 * nC),
                   .p = std::vector<double>(nC)};
    for (std::size_t c = 0; c < nC; ++c) {
        const double T = richT(fm.C[0][c], xc, w);
        double yF, yO;
        richYs(T, yF, yO);
        ic.rho[c] = 1.0 / T;
        ic.ry0[c] = ic.rho[c] * yF;
        ic.ry1[c] = ic.rho[c] * yO;
        ic.ru[c] = 0.98;
    }
    return ic;
}

double richZ(double yF, double yO)
{
    const PartialPremixModel& pm = rich().pm;
    return (pm.s * yF - yO + pm.yO2) / (pm.s * pm.yFrem() + pm.yO2);
}

// R1 Z = 1 kept (Shvab-Zeldovich: one limiter, one face formula, the s omega sink);
// R2 fuel + burn and oxidizer + s burn vs their tallies, mass; R3 the semi-discrete
// energy balance; both instantiations (uncorrected, C2b), both precisions, every mesh.
void gateRich()
{
    const PartialPremixModel& pm = rich().pm;
    for (bool law : {false, true})
        for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64})
            for (Mesh& m : meshes(pm.yF1(), pm.yO1())) {
                Run r = setupOf(m);
                DeviceFvLowMach d =
                    orDie(DeviceFvLowMach::make(r.s, r.t, r.g, modelRich(law), ctlOf(p), richIC(m.fm)), "make");
                const gpu::FvLowMachBudget b0 = orDie(d.budget(), "budget");
                march(d, 40);
                const gpu::FvLowMachTotals t = orDie(d.totals(), "totals");
                const gpu::FvLowMachBudget b1 = orDie(d.budget(), "budget");
                const auto c = orDie(d.cells(), "cells");
                double dz = 0;
                for (std::size_t q = 0; q < c.rho.size(); ++q)
                    dz = std::max(dz, std::abs(richZ(c.ry[0][q] / c.rho[q], c.ry[1][q] / c.rho[q]) - 1.0));
                const double em = std::abs(t.mass - t.massTally) / t.mass;
                const double eF = std::abs(t.y[0] + t.burn - t.yTally[0]) / t.y[0];
                const double eO = std::abs(t.y[1] + pm.s * t.burn - t.yTally[1]) / t.y[1];
                const double eE = std::max(std::abs(b0.defect) / b0.gross, std::abs(b1.defect) / b1.gross);
                const bool f64 = p == FvPrecision::Fp64;
                std::printf("R %s %s %s: |Z - 1| %.2e, mass %.2e, fuel %.2e, ox %.2e, burnt %.3g, energy rate %.2e\n",
                            m.name.c_str(), f64 ? "fp64" : "fp32", law ? "C2b" : "C2", dz, em, eF, eO, t.burn, eE);
                const std::string tag = std::format(" ({} {} {})", m.name, f64 ? "fp64" : "fp32", law ? "C2b" : "C2");
                check(t.burn > 0, "R burning" + tag);
                check(dz <= (f64 ? 1e-12 : 1e-6), "R1 Z kept" + tag);
                check(em <= (f64 ? 1e-13 : 1e-7), "R2 mass tally" + tag);
                check(eF <= (f64 ? 1e-13 : 1e-7), "R2 fuel tally" + tag);
                check(eO <= (f64 ? 1e-13 : 1e-7), "R2 oxidizer tally" + tag);
                check(eE <= (f64 ? 1e-13 : 1e-5), "R3 energy rate" + tag);
            }
}

// R7: gpu/lowmach's two-reactant C2b stage (kLmCellPass<true, true>, kLmScalars<true>)
// on the same mesh, same dt, same IC: rounding, as L7.
void gateRichStructured()
{
    const int nx = 128, ny = 4, steps = 400;
    const double h = 0.25, dt = 1e-3;
    const Grid g = Grid::make(nx, ny, 1, nx * h, ny * h, 1.0);
    const PartialPremixModel& pm = rich().pm;
    LowMachState ls;
    ls.rho.assign(g.size(), 0.0);
    ls.p.assign(g.size(), 0.0);
    ls.my.assign(g.size(), 0.0);
    ls.mz.assign(g.size(), 0.0);
    ls.mx.assign(std::size_t(nx + 1) * ny, 0.98);
    ls.rhoY[0].assign(g.size(), 0.0);
    ls.rhoY[1].assign(g.size(), 0.0);
    ls.nY = 2;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double T = richT((i + 0.5) * h, 16.0, 1.5);
            double yF, yO;
            richYs(T, yF, yO);
            ls.rho[g.idx(i, j, 0)] = 1.0 / T;
            ls.rhoY[0][g.idx(i, j, 0)] = yF / T;
            ls.rhoY[1][g.idx(i, j, 0)] = yO / T;
        }
    const FvLowMachModel mdl = modelRich(true);
    gpu::LowMachSetup sp;
    sp.tr = mdl.tr;
    sp.prGas = 1.0;
    sp.nY = 2;
    sp.fuel = 0;
    sp.ox = 1;
    sp.stoich = pm.s;
    sp.lnA = mdl.lnA;
    sp.Ta = mdl.Ta;
    sp.qcp = mdl.qcp;
    sp.mIn = 0.98;
    sp.TIn = 1.0;
    sp.yIn[0] = pm.yF1();
    sp.yIn[1] = pm.yO1();
    sp.law = mdl.law;
    auto st = orDie(gpu::DeviceLowMach::make(g, sp), "structured");
    if (auto e = st.upload(ls); !e) check(false, e.error());
    for (int i = 0; i < steps; ++i)
        if (auto e = st.step(dt); !e) check(false, e.error());
    const LowMachState sf = orDie(st.download(), "download");

    Mesh m{"box", orDie(fvLowMachBox(g), "box"), boxBcs(pm.yF1(), pm.yO1())};
    Run r = setupOf(m);
    const std::size_t nC = g.size();
    FvLowMachIC ic{.rho = std::vector<double>(nC),
                   .ry0 = std::vector<double>(nC),
                   .ry1 = std::vector<double>(nC),
                   .ru = std::vector<double>(3 * nC, 0.0),
                   .p = std::vector<double>(nC, 0.0)};
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t src = std::size_t(m.fm.srcCell[q]);
        ic.rho[q] = ls.rho[src];
        ic.ry0[q] = ls.rhoY[0][src];
        ic.ry1[q] = ls.rhoY[1][src];
        ic.ru[q] = 0.98;
    }
    auto marchFv = [&](FvPrecision p) {
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, mdl, ctlOf(p), ic), "make");
        for (int i = 0; i < steps; ++i) orDie(d.step(dt), "step");
        return orDie(d.cells(), "cells");
    };
    const auto f32 = marchFv(FvPrecision::Fp32), f64 = marchFv(FvPrecision::Fp64);
    double dSt = 0, dSt64 = 0, dOwn = 0, burnt = 0;
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t src = std::size_t(m.fm.srcCell[q]);
        for (int k = 0; k < 3; ++k) {
            const double s = k == 0 ? sf.rho[src] : sf.rhoY[k - 1][src];
            const double a = k == 0 ? f32.rho[q] : f32.ry[std::size_t(k - 1)][q];
            const double b = k == 0 ? f64.rho[q] : f64.ry[std::size_t(k - 1)][q];
            dSt = std::max(dSt, std::abs(a - s));
            dSt64 = std::max(dSt64, std::abs(b - s));
            dOwn = std::max(dOwn, std::abs(a - b));
        }
        burnt = std::max(burnt, std::abs(ls.rhoY[1][src] - sf.rhoY[1][src]));
    }
    std::printf("R7 C2b: FV32 vs structured %.3e, FV64 vs structured %.3e, FV32 vs FV64 %.3e (max |d rho Y_O| over the "
                "march %.3e)\n",
                dSt, dSt64, dOwn, burnt);
    check(burnt > 1e-3, "R7 the front moved");
    check(dSt <= 3.0 * dOwn, "R7 FV vs structured, two reactants: rounding");
}

// B1/B2 (M3 step 2): per-face inflow rows (fvLowMachRows). A slot of hot burnt
// products in a fresh coflow enters the box row by row: every inflow face carries its
// row's m to rounding, mass, fuel and the passive species close against the FP64
// tallies, both precisions, burning.
void gateRows()
{
    const Grid g = Grid::make(48, 32, 1, 12.0, 8.0, 1.0);
    auto rowAt = [](double y) {
        const double J = 0.5 * (1.0 - std::tanh((std::abs(y - 4.0) - 1.5) / 0.3));
        return FvLmRow{.m = 1.0 + 2.0 * J, .t = 1.0 + (kSigma - 1.0) * (1.0 - J), .y = {J, 0.25}};
    };
    for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
        Mesh m{"box", orDie(fvLowMachBox(g), "box"), boxBcs(1.0, 0.25)};
        if (auto b = bindFvIncompBcs(m.b.flow, m.fm); !b) check(false, b.error());
        FvIncompSetup s = orDie(buildFvIncompSetup(m.fm, m.b.flow), "setup");
        FvLowMachTables t = buildFvLowMachTables(m.fm, m.b, s, 1.0);
        const std::size_t nI = std::size_t(m.fm.nI), nC = std::size_t(m.fm.nC);
        const int n = orDie(fvLowMachRows(m.fm, "xlo", s, t, 1.0,
                                          [&](std::size_t q) {
                                              return rowAt(m.fm.Cf[1][nI + std::size_t(s.bFace[q])]);
                                          }),
                            "rows");
        check(n == g.ny, "B1 one row per inflow face");
        // The IC: the rows extruded (rho u = m along x).
        FvLowMachIC ic{.rho = std::vector<double>(nC),
                       .ry0 = std::vector<double>(nC),
                       .ry1 = std::vector<double>(nC),
                       .ru = std::vector<double>(3 * nC, 0.0),
                       .p = std::vector<double>(nC, 0.0)};
        for (std::size_t c = 0; c < nC; ++c) {
            const FvLmRow r = rowAt(m.fm.C[1][c]);
            ic.rho[c] = 1.0 / r.t;
            ic.ry0[c] = ic.rho[c] * r.y[0];
            ic.ry1[c] = ic.rho[c] * 0.25;
            ic.ru[c] = r.m;
        }
        const la::CellGraph cg = la::graphOf(m.fm);
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(s, t, cg, modelOf(2.0, 2), ctlOf(p), ic), "make");
        march(d, 40);
        const auto f = orDie(d.faces(), "faces");
        double eRow = 0;
        for (std::size_t q = 0; q < std::size_t(s.nB); ++q) {
            if (!t.bYfix[q]) continue;
            const double mag = s.bMagS[q], want = -rowAt(m.fm.Cf[1][nI + std::size_t(s.bFace[q])]).m * mag;
            eRow = std::max(eRow, std::abs(f.phiB[q] - want) / std::abs(want));
        }
        const gpu::FvLowMachTotals to = orDie(d.totals(), "totals");
        const double em = std::abs(to.mass - to.massTally) / to.mass;
        const double ey0 = std::abs(to.y[0] + to.burn - to.yTally[0]) / to.y[0];
        const double ey1 = std::abs(to.y[1] - to.yTally[1]) / to.y[1];
        const bool f64 = p == FvPrecision::Fp64;
        std::printf("B %s: inflow m vs row %.2e, mass %.2e, fuel %.2e, passive %.2e, burnt %.3g\n", f64 ? "fp64" : "fp32",
                    eRow, em, ey0, ey1, to.burn);
        check(to.burn > 0, "B burning");
        check(eRow <= (f64 ? 1e-14 : 1e-6), "B1 inflow faces carry the rows' m");
        check(em <= (f64 ? 1e-13 : 1e-7), "B2 mass tally with rows");
        check(ey0 <= (f64 ? 1e-13 : 1e-7), "B2 fuel tally with rows");
        check(ey1 <= (f64 ? 1e-13 : 1e-7), "B2 passive tally with rows");
    }
}

// The path a 1D flame never reads: a wrinkled front, collocated rho u vs the MAC faces'
// cell means. Not rounding (the schemes differ): the distance shrinks with h.
void gateMomentum()
{
    const FvLowMachModel mdl = modelOf(2.0, 1);
    double prev = 0;
    for (int f : {1, 2, 4}) {
        const int nx = 64 * f, ny = 16 * f;
        const double h = 0.5 / f, dt = 1.6e-3 / (f * f);
        const int steps = int(std::lround(0.4 / dt));
        const Grid g = Grid::make(nx, ny, 1, nx * h, ny * h, 1.0);
        LowMachState ls;
        ls.rho.assign(g.size(), 0.0);
        ls.p.assign(g.size(), 0.0);
        ls.my.assign(g.size(), 0.0);
        ls.mz.assign(g.size(), 0.0);
        ls.mx.assign(std::size_t(nx + 1) * ny, 0.98);
        ls.rhoY[0].assign(g.size(), 0.0);
        ls.nY = 1;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double y = (j + 0.5) * h, x = (i + 0.5) * h;
                const double xf = 16.0 + 1.5 * std::cos(2.0 * std::numbers::pi * y / (ny * h));
                const double T = 1.0 + 0.5 * (kSigma - 1.0) * (1.0 + std::tanh((x - xf) / 1.5));
                ls.rho[g.idx(i, j, 0)] = 1.0 / T;
                ls.rhoY[0][g.idx(i, j, 0)] = (kSigma - T) / (kSigma - 1.0) / T;
            }
        gpu::LowMachSetup sp;
        sp.tr = mdl.tr;
        sp.prGas = 1.0;
        sp.lnA = mdl.lnA;
        sp.Ta = mdl.Ta;
        sp.qcp = mdl.qcp;
        sp.mIn = 0.98;
        auto st = orDie(gpu::DeviceLowMach::make(g, sp), "structured");
        if (auto e = st.upload(ls); !e) check(false, e.error());
        for (int i = 0; i < steps; ++i)
            if (auto e = st.step(dt); !e) check(false, e.error());
        const LowMachState sf = orDie(st.download(), "download");

        Mesh m{"box", orDie(fvLowMachBox(g), "box"), boxBcs(1.0, 0.0)};
        Run r = setupOf(m);
        const std::size_t nC = g.size();
        FvLowMachIC ic{.rho = std::vector<double>(nC),
                       .ry0 = std::vector<double>(nC),
                       .ry1 = {},
                       .ru = std::vector<double>(3 * nC, 0.0),
                       .p = std::vector<double>(nC, 0.0)};
        for (std::size_t q = 0; q < nC; ++q) {
            const std::size_t src = std::size_t(m.fm.srcCell[q]);
            ic.rho[q] = ls.rho[src];
            ic.ry0[q] = ls.rhoY[0][src];
            ic.ru[q] = 0.98;
        }
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, mdl, ctlOf(FvPrecision::Fp64), ic), "make");
        for (int i = 0; i < steps; ++i) orDie(d.step(dt), "step");
        const auto c = orDie(d.cells(), "cells");
        double du = 0, dv = 0, vmax = 0, dr = 0;
        for (std::size_t q = 0; q < nC; ++q) {
            const std::size_t src = std::size_t(m.fm.srcCell[q]);
            const int i = int(src % std::size_t(nx)), j = int(src / std::size_t(nx));
            const double mx = 0.5 * (sf.mx[std::size_t(i) + std::size_t(nx + 1) * j] +
                                     sf.mx[std::size_t(i + 1) + std::size_t(nx + 1) * j]);
            const double my = 0.5 * (sf.my[g.idx(i, j, 0)] + sf.my[g.idx(i, (j + 1) % ny, 0)]);
            du = std::max(du, std::abs(c.ru[q] - mx));
            dv = std::max(dv, std::abs(c.ru[nC + q] - my));
            vmax = std::max(vmax, std::abs(my));
            dr = std::max(dr, std::abs(c.rho[q] - sf.rho[src]));
        }
        std::printf("L8 %d x %d: rho v max %.4f, |FV - MAC| rho v %.3e, rho u %.3e, rho %.3e%s\n", nx, ny, vmax, dv, du, dr,
                    prev > 0 ? std::format(" (order {:.2f})", std::log2(prev / dv)).c_str() : "");
        if (prev > 0) check(std::log2(prev / dv) >= 1.5, "L8 momentum: FV - MAC shrinks at order >= 1.5");
        prev = dv;
    }
}

int runFlame1d()
{
    const std::string pre = "/tmp/dns_test_fvlm1d";
    const int rc = valtest::runBinary("\"" DNS_BIN "\" flame --physics lowmach --solver fv --cells-per-delta 7.5 "
                                      "--inflow-speed 0.98 --tt 12 --fv-precision fp64 --emit stats --quiet --out " + pre);
    check(rc == 0, "flame1d exits 0");
    if (rc != 0) return dnstest::finish();
    const auto t = valtest::loadCsv(pre + "_stats.csv");
    check(t.has_value(), "flame1d stats");
    if (!t) return dnstest::finish();
    double sl = 0, dth = 0, hits = 0;
    int n = 0;
    for (std::size_t i = 0; i < t->col("t").size(); ++i) {
        hits += t->col("yHits")[i];
        if (t->col("t")[i] < 6) continue;
        sl += t->col("SL")[i];
        dth += t->col("dth")[i];
        ++n;
    }
    sl /= n;
    dth /= n;
    std::printf("flame1d: S_L %.6f, delta_th %.5f, hits %g\n", sl, dth, hits);
    check(std::abs(sl - 0.99428) <= 2e-3, "flame1d S_L against the structured 7.5-cell value");
    check(hits == 0, "flame1d no Y outside its bounds");
    return dnstest::finish();
}

// `test_gpu_fv_lowmach bunsen`: the binary's 2D slot burner (G2a, 4 cells/delta_th, window
// [20, 25]) against the structured 4-cell run of the same flags (docs/fvincomp.md table).
int runBunsen()
{
    const std::string pre = "/tmp/dns_test_fvlm_bunsen";
    const int rc = valtest::runBinary("\"" DNS_BIN "\" flame --physics lowmach --solver fv --mode bunsen "
                                      "--cells-per-delta 4 --length 49 --tt 25 --emit stats --quiet --out " + pre);
    check(rc == 0, "bunsen exits 0 (anchored: the exit-8 guards passed)");
    if (rc != 0) return dnstest::finish();
    const auto t = valtest::loadCsv(pre + "_stats.csv");
    check(t.has_value(), "bunsen stats");
    if (!t) return dnstest::finish();
    double tip = 0, br = 0, me = 0, hits = 0;
    int n = 0;
    for (std::size_t i = 0; i < t->col("t").size(); ++i) {
        hits = std::max(hits, t->col("yHits")[i]);
        me = std::max(me, std::abs(t->col("massErr")[i]));
        if (t->col("t")[i] < 20) continue;
        tip += t->col("xTip")[i];
        br += t->col("burnRatio")[i];
        ++n;
    }
    tip /= n;
    br /= n;
    std::printf("bunsen: xTip %.4f (structured 44.304), burnRatio %.6f, max |massErr| %.2e, hits %g\n", tip, br, me, hits);
    // Set after the measurement (FV - structured +3.2e-3 at 4 cells, a scheme gap: LF5).
    check(std::abs(tip - 44.304) <= 1e-2, "bunsen xTip within 1e-2 of the structured 4-cell run");
    check(std::abs(br - 1.0) <= 1e-4, "bunsen burn = fuel inflow");
    check(me <= 1e-7, "bunsen mass vs its FP64 tally");
    check(hits == 0, "bunsen no Y outside its bounds");
    return dnstest::finish();
}

// --- M3 steps 3-5: open faces, gravity, the rim ------------------------------------
// One case, both solvers: the structured setup and state, and the same case on a
// FaceMesh of the same cells (fvLowMachBox: open y -> `open` patches, a solid map ->
// the `rim` Wall patch).
struct Pair
{
    Grid g;
    gpu::LowMachSetup sp;
    LowMachState ls;
};

// The hot-blob box of test_gpu_lowmach (T = 1 + 3 exp(-r^2/4) at (Lx/3, Ly/2), m_x = mIn,
// m_y = 0.3 rho sin(2 pi x/Lx)), inert, Y uniform yF / yO, open y at that ambient.
Pair blobPair(int nx, int ny, double h, int nY, double mIn, double yF, double yO, bool open)
{
    Pair c;
    c.g = Grid::make(nx, ny, 1, nx * h, ny * h, 1.0);
    gpu::LowMachSetup& s = c.sp;
    s.tr = Transport::air(0.72, 1.0, 0.0, 0.72, kGammaAir, kRgasBox, false);
    s.prGas = 1;
    s.nY = nY;
    s.fuel = 0;
    s.ox = nY == 2 ? 1 : -1; // gpu/lowmach lays two species out as fuel, oxidizer (inert here)
    s.lnA = -HUGE_VAL;
    s.Ta = 50;
    s.qcp = 4;
    s.mIn = mIn;
    s.TIn = 1;
    s.yIn[0] = yF;
    s.yIn[1] = yO;
    s.openY = open;
    s.ambT = 1;
    s.ambY[0] = yF;
    s.ambY[1] = yO;
    const Grid& g = c.g;
    LowMachState& q = c.ls;
    q.nY = nY;
    q.rho.assign(g.size(), 0);
    q.p.assign(g.size(), 0);
    q.my.assign(g.size(), 0);
    q.mz.assign(g.size(), 0);
    q.mx.assign(std::size_t(nx + 1) * ny, mIn);
    q.rhoY[0].assign(g.size(), 0);
    if (nY == 2) q.rhoY[1].assign(g.size(), 0);
    if (open) q.myHi.assign(std::size_t(nx), 0.0);
    const double Lx = nx * h, Ly = ny * h;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double x = (i + 0.5) * h, y = (j + 0.5) * h;
            const double r2 = (x - Lx / 3) * (x - Lx / 3) + (y - Ly / 2) * (y - Ly / 2);
            const double T = 1 + 3 * std::exp(-r2 / 4);
            const std::size_t c0 = g.idx(i, j, 0);
            q.rho[c0] = 1 / T;
            q.rhoY[0][c0] = q.rho[c0] * yF;
            if (nY == 2) q.rhoY[1][c0] = q.rho[c0] * yO;
            if (mIn != 0) q.my[c0] = 0.3 * q.rho[c0] * std::sin(2 * std::numbers::pi * x / Lx);
        }
    return c;
}

struct FvCase
{
    Mesh m;
    Run r;
    FvLowMachIC ic;
    FvLowMachModel model;
};

// The FV twin of a Pair: rho u the MAC faces' cell means, every other field cell for
// cell; the x-lo inflow at the structured face state (per row when sp.rowM is set).
FvCase fvOf(const Pair& c, std::span<const std::uint8_t> solid = {})
{
    const Grid& g = c.g;
    const gpu::LowMachSetup& sp = c.sp;
    FvBcSpec spec = fvLowMachBoxBc(sp.openY, sp.ambT, {sp.ambY[0], sp.ambY[1]});
    spec.patches[0].y = {sp.yIn[0], sp.yIn[1], 0, 0};
    spec.patches[0].tIn = sp.TIn;
    if (!solid.empty())
        spec.patches.push_back({.name = "rim", .kind = FvPatchKind::Wall, .isothermal = true, .tWall = sp.wallT});
    FvCase f{.m = {"box", orDie(fvLowMachBox(g, sp.openY, solid), "box"),
                   orDie(fvLowMachBcs(spec, sp.mIn * sp.TIn / sp.prGas, 1.0), "bcs")},
             .r = {},
             .ic = {},
             .model = {}};
    f.r = setupOf(f.m);
    if (!sp.rowM.empty()) {
        orDie(fvLowMachRows(f.m.fm, "xlo", f.r.s, f.r.t, sp.prGas,
                            [&](std::size_t q) {
                                const std::size_t row = std::size_t(f.m.fm.srcCell[std::size_t(f.r.s.bOwn[q])]) /
                                                        std::size_t(g.nx);
                                return FvLmRow{.m = sp.rowM[row],
                                               .t = sp.rowT[row],
                                               .y = {sp.rowY[0][row], sp.nY > 1 ? sp.rowY[1][row] : 0.0}};
                            }),
              "rows");
    }
    f.model = {.tr = sp.tr,
               .prGas = sp.prGas,
               .lnA = sp.lnA,
               .Ta = sp.Ta,
               .qcp = sp.qcp,
               .nY = sp.nY,
               .backflow = sp.backflow,
               .gravity = {-sp.gravity, 0.0, 0.0}};
    const std::size_t nC = std::size_t(f.m.fm.nC), nx = std::size_t(g.nx);
    f.ic = {.rho = std::vector<double>(nC),
            .ry0 = std::vector<double>(nC),
            .ry1 = sp.nY > 1 ? std::vector<double>(nC) : std::vector<double>{},
            .ru = std::vector<double>(3 * nC, 0.0),
            .p = std::vector<double>(nC)};
    const LowMachState& s = c.ls;
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t src = std::size_t(f.m.fm.srcCell[q]), i = src % nx, j = src / nx;
        f.ic.rho[q] = s.rho[src];
        f.ic.ry0[q] = s.rhoY[0][src];
        if (sp.nY > 1) f.ic.ry1[q] = s.rhoY[1][src];
        f.ic.p[q] = s.p[src];
        f.ic.ru[q] = 0.5 * (s.mx[i + (nx + 1) * j] + s.mx[i + 1 + (nx + 1) * j]);
        const double myh = j + 1 < std::size_t(g.ny) ? s.my[src + nx] : sp.openY ? s.myHi[i] : s.my[i];
        f.ic.ru[nC + q] = 0.5 * (s.my[src] + myh);
    }
    return f;
}

// FV and structured fields after a march: max |rho| and |rho u| (the MAC faces' cell
// means) distances over the FV cells.
// max and volume-weighted mean (l1) distances; the max of rho u and where it sits;
// "away": the max over cells farther than `away` from every point of `corners`.
struct Dist
{
    double rho = 0, ru = 0, scale = 0, l1rho = 0, l1ru = 0, x = 0, y = 0, ruAway = 0;
};
Dist distance(const Pair& c, const FvCase& f, const gpu::FvLowMachCells& fc, const LowMachState& sf,
              const std::vector<std::array<double, 2>>& corners = {}, double away = 0)
{
    const Grid& g = c.g;
    const std::size_t nC = fc.rho.size(), nx = std::size_t(g.nx);
    Dist d;
    double vol = 0;
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t src = std::size_t(f.m.fm.srcCell[q]), i = src % nx, j = src / nx;
        const double mx = 0.5 * (sf.mx[i + (nx + 1) * j] + sf.mx[i + 1 + (nx + 1) * j]);
        const double myh = j + 1 < std::size_t(g.ny) ? sf.my[src + nx] : c.sp.openY ? sf.myHi[i] : sf.my[i];
        const double my = 0.5 * (sf.my[src] + myh);
        const double V = f.m.fm.V[q], er = std::abs(fc.rho[q] - sf.rho[src]);
        const double eu = std::max(std::abs(fc.ru[q] - mx), std::abs(fc.ru[nC + q] - my));
        d.rho = std::max(d.rho, er);
        if (eu > d.ru) {
            d.ru = eu;
            d.x = f.m.fm.C[0][q];
            d.y = f.m.fm.C[1][q];
        }
        bool far = true;
        for (const auto& p : corners) far = far && std::hypot(f.m.fm.C[0][q] - p[0], f.m.fm.C[1][q] - p[1]) > away;
        if (far) d.ruAway = std::max(d.ruAway, eu);
        d.l1rho += V * er;
        d.l1ru += V * eu;
        vol += V;
        d.scale = std::max({d.scale, std::abs(mx), std::abs(my)});
    }
    d.l1rho /= vol;
    d.l1ru /= vol;
    return d;
}

std::string orders(double a0, double a1) { return std::format("{:.2f}", std::log2(a0 / a1)); }

LowMachState marchStructured(const Pair& c, int steps, double dt)
{
    auto st = orDie(gpu::DeviceLowMach::make(c.g, c.sp), "structured");
    if (auto e = st.upload(c.ls); !e) check(false, e.error());
    for (int i = 0; i < steps; ++i)
        if (auto e = st.step(dt); !e) check(false, e.error());
    return orDie(st.download(), "download");
}

// O1 / G1: a quiescent open box (rho = rho_a, Y the ambient's, m_in 0) stays at rest
// EXACTLY over 1000 steps of dt 0.05, with and without gravity, both precisions:
// gpu/lowmach's open rest and hydro rest gates.
void gateOpenRest()
{
    for (const double gv : {0.0, 1.0})
        for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
            Pair c = blobPair(32, 16, 0.25, 2, 0.0, 0.25, 0.5, true);
            c.sp.gravity = gv;
            for (double& v : c.ls.rho) v = 1;
            for (double& v : c.ls.rhoY[0]) v = 0.25;
            for (double& v : c.ls.rhoY[1]) v = 0.5;
            FvCase f = fvOf(c);
            DeviceFvLowMach d = orDie(DeviceFvLowMach::make(f.r.s, f.r.t, f.r.g, f.model, ctlOf(p), f.ic), "make");
            for (int i = 0; i < 1000; ++i) orDie(d.step(0.05), "step");
            const auto o = orDie(d.cells(), "cells");
            bool rest = true;
            for (std::size_t q = 0; q < o.rho.size(); ++q)
                rest = rest && o.rho[q] == 1 && o.ry[0][q] == 0.25 && o.ry[1][q] == 0.5 && o.p[q] == 0;
            for (double v : o.ru) rest = rest && v == 0;
            for (double v : o.phi) rest = rest && v == 0;
            for (double v : o.phiB) rest = rest && v == 0;
            std::printf("O1 rest g %g %s: %s\n", gv, p == FvPrecision::Fp32 ? "fp32" : "fp64", rest ? "exact" : "MOVED");
            check(rest, std::format("O1/G1 quiescent open box at g {}: rho u, m, p exactly 0 over 1000 steps", gv));
        }
}

// O2: a nearly inviscid vortex straddling the open y-hi face (test_gpu_lowmach's
// backflow case on faces): the kinetic energy never grows with the stabilization
// (beta 1) and does without it (beta 0).
void gateBackflow()
{
    double grow[2] = {0, 0};
    for (const double beta : {1.0, 0.0}) {
        Pair c = blobPair(64, 48, 0.25, 1, 0.0, 0.5, 0, true);
        c.sp.backflow = beta;
        c.sp.tr = Transport::air(0.01, 1.0, 0.0, 0.72, kGammaAir, kRgasBox, false);
        for (double& v : c.ls.rho) v = 1;
        for (double& v : c.ls.rhoY[0]) v = 0.5;
        FvCase f = fvOf(c);
        const Grid& g = c.g;
        const double xc = 0.5 * g.nx * g.hx, yc = (g.ny - 3) * g.hy, rc = 2.0, G = 30.0;
        const std::size_t nC = std::size_t(f.m.fm.nC);
        for (std::size_t q = 0; q < nC; ++q) {
            const double dx = f.m.fm.C[0][q] - xc, dy = f.m.fm.C[1][q] - yc, r2 = dx * dx + dy * dy;
            const double fr = r2 > 0 ? G / (2 * std::numbers::pi * r2) * (1 - std::exp(-r2 / (rc * rc))) : 0.0;
            f.ic.ru[q] = -fr * dy;
            f.ic.ru[nC + q] = fr * dx;
        }
        auto ke = [&](const gpu::FvLowMachCells& o) {
            double e = 0;
            for (std::size_t q = 0; q < nC; ++q)
                e += 0.5 * f.m.fm.V[q] * (o.ru[q] * o.ru[q] + o.ru[nC + q] * o.ru[nC + q]) / o.rho[q];
            return e;
        };
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(f.r.s, f.r.t, f.r.g, f.model, ctlOf(FvPrecision::Fp32), f.ic),
                                  "make");
        // The first step projects the laid field (the structured IC is the faces').
        orDie(d.step(0.01), "step");
        const double ke0 = ke(orDie(d.cells(), "cells"));
        double keMax = ke0, keEnd = ke0;
        for (int n = 1; n < 2000; ++n) {
            auto s = d.step(0.01);
            if (!s) {
                keMax = HUGE_VAL;
                break;
            }
            if ((n + 1) % 100 == 0) {
                keEnd = ke(orDie(d.cells(), "cells"));
                if (!std::isfinite(keEnd)) {
                    keMax = HUGE_VAL;
                    break;
                }
                keMax = std::max(keMax, keEnd);
            }
        }
        grow[beta == 1.0 ? 0 : 1] = keMax / ke0;
        std::printf("O2 backflow beta %g: KE max/KE0 %.6f (KE0 %.4g), end/KE0 %.4f\n", beta, keMax / ke0, ke0,
                    keEnd / ke0);
    }
    check(grow[0] <= 1 + 1e-6, "O2 backflow beta 1: the kinetic energy never grows (2000 steps)");
    check(grow[1] > 1.5, "O2 backflow beta 0: the open face injects kinetic energy (KE grows > 1.5x)");
}

// O3: the hot blob through open laterals: uniform Y bitwise, mass and species = their
// FP64 tallies, both precisions; a pure-fuel box under an AIR ambient: every oxygen
// atom came in through an open face (Y_O in [0, Y_O2], inventory = its tally).
void gateOpenTallies()
{
    for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
        const bool f64 = p == FvPrecision::Fp64;
        for (const bool air : {false, true}) {
            Pair c = blobPair(64, 32, 0.25, 2, 1.0, air ? 1.0 : 0.5, air ? 0.0 : 0.25, true);
            if (air) {
                c.sp.ambY[0] = 0.0;
                c.sp.ambY[1] = 0.233;
            }
            FvCase f = fvOf(c);
            DeviceFvLowMach d = orDie(DeviceFvLowMach::make(f.r.s, f.r.t, f.r.g, f.model, ctlOf(p), f.ic), "make");
            const auto b = orDie(d.stableDt(0.8, 1.0), "dt");
            const double dt = 0.5 * std::min(b.conv, b.diff);
            double through = 0, latIn = 0;
            for (int i = 0; i < 100; ++i) {
                orDie(d.step(dt), "step");
                const auto fx = orDie(d.faces(), "faces");
                for (std::size_t q = 0; q < fx.phiB.size(); ++q) {
                    through += dt * std::abs(fx.phiB[q]);
                    if (f.r.t.bOpen[q] == 2 && fx.phiB[q] < 0) latIn -= dt * fx.phiB[q];
                }
            }
            const auto o = orDie(d.cells(), "cells");
            const auto t = orDie(d.totals(), "totals");
            std::size_t bad = 0;
            double yoMin = 1, yoMax = 0;
            for (std::size_t q = 0; q < o.rho.size(); ++q) {
                if (!air) bad += (o.ry[0][q] != 0.5 * o.rho[q]) + (o.ry[1][q] != 0.25 * o.rho[q]);
                const double yo = o.ry[1][q] / o.rho[q];
                yoMin = std::min(yoMin, yo);
                yoMax = std::max(yoMax, yo);
            }
            const double em = std::abs(t.mass - t.massTally) / through;
            const double e0 = std::abs(t.y[0] - t.yTally[0]) / through, e1 = std::abs(t.y[1] - t.yTally[1]) / through;
            std::printf("O3 %s %s: %zu cells off, mass %.2e, y0 %.2e, y1 %.2e of throughput %.4g (lateral inflow %.4g), "
                        "Y_O [%.3g, %.6g]\n",
                        air ? "fuel box, air ambient" : "uniform Y", f64 ? "fp64" : "fp32", bad, em, e0, e1, through,
                        latIn, yoMin, yoMax);
            if (!air) check(bad == 0, "O3 uniform Y bitwise through open faces");
            check(em <= (f64 ? 1e-13 : 1e-7) && e0 <= (f64 ? 1e-13 : 1e-7) && e1 <= (f64 ? 1e-13 : 1e-7),
                  "O3 mass and species = their FP64 tallies with open faces");
            check(latIn > 0, "O3 the open laterals draw gas in");
            if (air) {
                // The same case on gpu/lowmach (its bound 1e-6). An open face upwinding by
                // the stage INPUT's flux sign carried the ambient's Y_O out of a face that
                // turned outward: Y_O -2.6e-5 here (measured), structured 3.7e-25.
                const LowMachState sf = marchStructured(c, 100, dt);
                double soMin = 1;
                for (std::size_t q = 0; q < sf.rho.size(); ++q) soMin = std::min(soMin, sf.rhoY[1][q] / sf.rho[q]);
                std::printf("   structured, same case, same dt: Y_O min %.3g\n", soMin);
                check(yoMax <= 0.233 + 1e-6 && t.y[1] > 0 && yoMin >= -1e-6,
                      "O3 air ambient: Y_O in [0, Y_O2] to 1e-6, the oxygen came through the open faces");
            }
        }
    }
}

// O4 / G3: the blob through open laterals vs gpu/lowmach on the same cells, same dt
// sequence (dt = 0.03 h^2: time error negligible), t 0.25; 3 levels. The schemes differ
// (collocated vs MAC momentum, the open face's cell vs face velocity), so the distance
// is a scheme gap: it must shrink.
void gateOpenStructured(double gv)
{
    Dist prev;
    for (int f = 1; f <= 4; f *= 2) {
        const double h = 0.5 / f, dt = 0.03 * h * h;
        const int steps = int(std::lround(0.25 / dt));
        Pair c = blobPair(32 * f, 16 * f, h, 1, 1.0, 0.5, 0, true);
        c.sp.gravity = gv;
        // The swirl vanishes on the open faces (sin(pi y/Ly)): both solvers start with
        // zero normal flux there (a p = 0 face keeps the predictor's normal flux, so a
        // laid one would be an IC difference, not the schemes').
        const Grid& g = c.g;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                c.ls.my[g.idx(i, j, 0)] = 0.3 * c.ls.rho[g.idx(i, j, 0)] *
                                          std::sin(2 * std::numbers::pi * (i + 0.5) * h / (g.nx * h)) *
                                          std::sin(std::numbers::pi * j / g.ny);
        const LowMachState sf = marchStructured(c, steps, dt);
        FvCase fv = fvOf(c);
        DeviceFvLowMach d =
            orDie(DeviceFvLowMach::make(fv.r.s, fv.r.t, fv.r.g, fv.model, ctlOf(FvPrecision::Fp64), fv.ic), "make");
        for (int i = 0; i < steps; ++i) orDie(d.step(dt), "step");
        const auto cells = orDie(d.cells(), "cells");
        const Dist e = distance(c, fv, cells, sf);
        // The net lateral outflow (per unit depth): the partition of the throughput.
        // c: without the corner columns (i = 0, nx - 1: an open face meeting x-lo, x-hi).
        double latF = 0, latS = 0, latFc = 0, latSc = 0;
        for (std::size_t q = 0; q < cells.phiB.size(); ++q) {
            if (fv.r.t.bOpen[q] != 2) continue;
            const int i = int(std::size_t(fv.m.fm.srcCell[std::size_t(fv.r.s.bOwn[q])]) % std::size_t(g.nx));
            latF += cells.phiB[q];
            if (i > 0 && i + 1 < g.nx) latFc += cells.phiB[q];
        }
        for (int i = 0; i < g.nx; ++i) {
            const double v = (sf.myHi[std::size_t(i)] - sf.my[g.idx(i, 0, 0)]) * g.hx;
            latS += v;
            if (i > 0 && i + 1 < g.nx) latSc += v;
        }
        std::printf("%s %d x %d: |FV - MAC| l1 rho %.3e, rho u %.3e; max rho %.3e, rho u %.3e at (%.3g, %.3g) (scale "
                    "%.3f); lateral outflow %.6f vs %.6f (%+.3e; without the corner columns %+.3e)%s\n",
                    gv > 0 ? "G3" : "O4", g.nx, g.ny, e.l1rho, e.l1ru, e.rho, e.ru, e.x, e.y, e.scale, latF, latS,
                    latF - latS, latFc - latSc,
                    prev.rho > 0 ? std::format(" (orders l1 {}, {}; max {}, {})", orders(prev.l1rho, e.l1rho),
                                               orders(prev.l1ru, e.l1ru), orders(prev.rho, e.rho), orders(prev.ru, e.ru))
                                       .c_str()
                                 : "");
        if (prev.rho > 0)
            check(e.l1rho < prev.l1rho && e.l1ru < prev.l1ru,
                  std::string(gv > 0 ? "G3" : "O4") + " FV - structured (l1) shrinks with h");
        prev = e;
    }
}


// G2: a stably stratified column (T 1 -> 1.5 upward, closed bottom: an x-lo face at m
// 0; p = 0 on top) under g 1 vs the same column at g 0, same dt: hydrostatically
// balanced (gpu/lowmach's hydro gate: |dm| <= 1e-4 rho sqrt(g L) on the box). The
// sheared and random meshes (level x = 0, Lx): the residual of the non-orthogonal
// correction, measured.
void gateHydro()
{
    const int nx = 64, ny = 16;
    const double h = 0.25, Lx = nx * h, gv = 1.0;
    const Grid g = Grid::make(nx, ny, 1, Lx, ny * h, 1.0);
    auto strat = [&](double x) { return 1.0 + 0.25 * (1 + std::tanh((x - 0.5 * Lx) / 2.0)); };
    struct M
    {
        std::string name;
        FaceMesh fm;
        bool slip;
    };
    std::vector<M> ms;
    ms.push_back({"box", orDie(fvLowMachBox(g), "box"), false});
    {
        // Sheared: every node moved along y by 8 h sin(pi x / Lx), x unmoved: x = 0 and Lx
        // stay level (a hydrostatic rest needs a level p = 0 face), cyclic y keeps its
        // translation, the y faces lean up to atan(8 pi h / Lx) = 21 deg.
        CurvMesh cm = CurvMesh::uniform(g);
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) cm.y[g.idx(i, j, 0)] += 8.0 * h * std::sin(std::numbers::pi * i / nx);
        ms.push_back({"sheared 21 deg", orDie(FaceMesh::fromCurvMesh(cm, openSpec(g), CellOrder::Morton), "sheared"),
                      false});
    }
    {
        CurvMesh cm = CurvMesh::uniform(g);
        for (int j = 1; j < ny; ++j)
            for (int i = 1; i < nx; ++i) {
                const std::size_t n = g.idx(i, j, 0);
                cm.x[n] += 0.3 * h * fvHashUnit(7, n, 0);
                cm.y[n] += 0.3 * h * fvHashUnit(7, n, 1);
            }
        StructuredSpec ss = openSpec(g);
        ss.type[2] = ss.type[3] = FoamPatchType::Patch;
        ms.push_back({"random 0.3 h", orDie(FaceMesh::fromCurvMesh(cm, ss, CellOrder::Morton), "random"), true});
    }
    for (M& mm : ms) {
        FvBcSpec spec = fvLowMachBoxBc();
        if (mm.slip) {
            spec.patches.push_back({.name = "ylo", .kind = FvPatchKind::Slip});
            spec.patches.push_back({.name = "yhi", .kind = FvPatchKind::Slip});
        }
        Mesh m{mm.name, std::move(mm.fm), orDie(fvLowMachBcs(spec, 0.0, 1.0), "bcs")};
        Run r = setupOf(m);
        const std::size_t nC = std::size_t(m.fm.nC);
        FvLowMachIC ic{.rho = std::vector<double>(nC),
                       .ry0 = std::vector<double>(nC),
                       .ry1 = {},
                       .ru = std::vector<double>(3 * nC, 0.0),
                       .p = std::vector<double>(nC, 0.0)};
        for (std::size_t q = 0; q < nC; ++q) {
            ic.rho[q] = 1.0 / strat(m.fm.C[0][q]);
            ic.ry0[q] = ic.rho[q];
        }
        FvLowMachModel md = modelOf(kNoRate, 1);
        md.tr = Transport::air(0.72, 1.0, 0.0, 0.72, kGammaAir, kRgasBox, false);
        double dm = 0, mMax = 0;
        std::vector<double> ru[2];
        for (int k = 0; k < 2; ++k) {
            md.gravity = {k == 0 ? -gv : 0.0, 0.0, 0.0};
            DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, md, ctlOf(FvPrecision::Fp64), ic), "make");
            for (int i = 0; i < 1000; ++i) orDie(d.step(0.01), "step");
            ru[k] = orDie(d.cells(), "cells").ru;
        }
        for (std::size_t q = 0; q < ru[0].size(); ++q) {
            dm = std::max(dm, std::abs(ru[0][q] - ru[1][q]));
            mMax = std::max(mMax, std::abs(ru[1][q]));
        }
        const double scale = std::sqrt(gv * Lx);
        std::printf("G2 hydro %s: max |rho u(g) - rho u(0)| %.3e = %.2e of the free-fall scale (conduction flow max "
                    "%.3e)\n",
                    m.name.c_str(), dm, dm / scale, mMax);
        // gpu/lowmach's bound, applied to the sheared and random meshes after their first
        // measurement (5.2e-5, 3.1e-6 of the scale).
        check(dm <= 1e-4 * scale, "G2 the column under g equals the column at g 0: " + m.name);
    }
}

// A slot tube on the box (2D, open y): lips 2 f cells thick at y in [2.5, 3) and [5, 5.5),
// x < 3, wall T_w 1.5; a cold jet m 1 through the tube, a room draft 0.2 outside, the
// passive Y 0.5 everywhere (inflow and ambient). The staircase IS the FaceMesh's rim
// patch here: gpu/lowmach's walled kernels are the oracle on the same cells.
Pair tubePair(int f, std::vector<std::uint8_t>& solid)
{
    const double h = 0.25 / f;
    Pair c = blobPair(48 * f, 32 * f, h, 1, 1.0, 0.5, 0, true);
    const Grid& g = c.g;
    solid.assign(g.size(), 0);
    auto lip = [](double y) { return (y >= 2.5 && y < 3.0) || (y >= 5.0 && y < 5.5); };
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i)
            if ((i + 0.5) * h < 3.0 && lip((j + 0.5) * h)) solid[g.idx(i, j, 0)] = 1;
    gpu::LowMachSetup& s = c.sp;
    s.solid = solid;
    s.wallT = 1.5;
    s.rowM.assign(std::size_t(g.ny), 0.0);
    s.rowT.assign(std::size_t(g.ny), 1.0);
    s.rowY[0].assign(std::size_t(g.ny), 0.5);
    for (int j = 0; j < g.ny; ++j) {
        const double y = (j + 0.5) * h;
        s.rowM[std::size_t(j)] = lip(y) ? 0.0 : (y >= 3.0 && y < 5.0) ? 1.0 : 0.2;
        if (lip(y)) s.rowT[std::size_t(j)] = 1.5;
    }
    LowMachState& q = c.ls;
    const std::size_t nx = std::size_t(g.nx);
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c0 = g.idx(i, j, 0);
            const double T = solid[c0] ? 1.5 : 1.0;
            q.rho[c0] = 1.0 / T;
            q.rhoY[0][c0] = solid[c0] ? 0.0 : 0.5 / T;
            q.my[c0] = 0;
        }
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i <= g.nx; ++i) {
            const bool shut = (i > 0 && solid[g.idx(i - 1, j, 0)]) || (i < g.nx && solid[g.idx(i, j, 0)]);
            q.mx[std::size_t(i) + (nx + 1) * std::size_t(j)] = shut ? 0.0 : s.rowM[std::size_t(j)];
        }
    return c;
}

// R1: the tube (impermeable, isothermal, species-tight Wall patch) vs gpu/lowmach's
// staircase on the same cells, same dt sequence, t 0.125; 3 levels. Uniform Y bitwise,
// mass and species = their FP64 tallies, every rim face carries phi_b == 0 and a zero
// species flux EXACTLY.
void gateRimStructured()
{
    Dist prev;
    for (int f = 1; f <= 4; f *= 2) {
        std::vector<std::uint8_t> solid;
        const Pair c = tubePair(f, solid);
        const double h = 0.25 / f, dt = 0.03 * h * h;
        const int steps = int(std::lround(0.125 / dt));
        const LowMachState sf = marchStructured(c, steps, dt);
        FvCase fv = fvOf(c, solid);
        DeviceFvLowMach d =
            orDie(DeviceFvLowMach::make(fv.r.s, fv.r.t, fv.r.g, fv.model, ctlOf(FvPrecision::Fp64), fv.ic), "make");
        for (int i = 0; i < steps; ++i) orDie(d.step(dt), "step");
        const auto o = orDie(d.cells(), "cells");
        const auto fx = orDie(d.faces(), "faces");
        const auto t = orDie(d.totals(), "totals");
        std::size_t bad = 0, leak = 0, nRim = 0;
        for (std::size_t q = 0; q < o.rho.size(); ++q) bad += o.ry[0][q] != 0.5 * o.rho[q];
        const MeshPatch& rim = *std::ranges::find(fv.m.fm.patches, std::string("rim"), &MeshPatch::name);
        double heat = 0;
        for (std::size_t q = 0; q < std::size_t(fv.r.s.nB); ++q) {
            const int bf = fv.r.s.bFace[q];
            if (bf < rim.first || bf >= rim.first + rim.count) continue;
            ++nRim;
            leak += fx.phiB[q] != 0.0 || fx.fy[0][q] != 0.0;
            heat += fx.heat[q];
        }
        const double em = std::abs(t.mass - t.massTally) / t.mass, ey = std::abs(t.y[0] - t.yTally[0]) / t.y[0];
        // The tube's corners (lip tips, the inflow face): a corner cell is a scheme number
        // (CLAUDE.md, the lid corner).
        const std::vector<std::array<double, 2>> tips = {{3.0, 2.5}, {3.0, 3.0}, {3.0, 5.0}, {3.0, 5.5},
                                                         {0.0, 2.5}, {0.0, 3.0}, {0.0, 5.0}, {0.0, 5.5}};
        const Dist e = distance(c, fv, o, sf, tips, 0.5);
        std::printf("R1 %d x %d (%zu rim faces): %zu cells off, %zu leaking rim faces, wall heat into the fluid %.4g, "
                    "mass %.2e, species %.2e; |FV - MAC| l1 rho %.3e, rho u %.3e; max rho %.3e, rho u %.3e at (%.3g, "
                    "%.3g), rho u 0.5 off the corners %.3e (scale %.3f)%s\n",
                    c.g.nx, c.g.ny, nRim, bad, leak, heat, em, ey, e.l1rho, e.l1ru, e.rho, e.ru, e.x, e.y, e.ruAway,
                    e.scale,
                    prev.rho > 0 ? std::format(" (orders l1 {}, {}; max {}, {}; off the tips {})", orders(prev.l1rho, e.l1rho),
                                               orders(prev.l1ru, e.l1ru), orders(prev.rho, e.rho), orders(prev.ru, e.ru),
                                               orders(prev.ruAway, e.ruAway))
                                       .c_str()
                                 : "");
        check(bad == 0, "R1 uniform Y bitwise beside the tube");
        check(leak == 0 && nRim > 0, "R1 every rim face: phi_b == 0, species flux == 0");
        check(heat > 0, "R1 the warm tube heats the jet");
        check(em <= 1e-13 && ey <= 1e-13, "R1 mass and species = their FP64 tallies");
        if (prev.rho > 0) {
            check(e.l1rho < prev.l1rho && e.l1ru < prev.l1ru, "R1 FV - structured (l1) shrinks with h");
            check(e.ruAway < prev.ruAway, "R1 FV - structured rho u off the tube's corners shrinks with h");
        }
        prev = e;
    }
}

// R2: conduction through the rim (gpu_lowmach_rim_cond on faces): a slab across y at x
// in [12, 13), T_w 1, the x-lo face at T 2 and m 0, open y, constant mu: the LINEAR
// profile between them is the fixed point, and the wall heat equals the inflow
// face's conduction and lambda dT / L L_y.
void gateRimCond()
{
    for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
        const int nx = 64, ny = 16;
        const double h = 0.25, L = 12.0;
        Pair c = blobPair(nx, ny, h, 1, 0.0, 0.0, 0, true);
        c.sp.TIn = 2.0;
        c.sp.wallT = 1.0;
        std::vector<std::uint8_t> solid(c.g.size(), 0);
        for (int j = 0; j < ny; ++j)
            for (int i = 48; i < 52; ++i) solid[c.g.idx(i, j, 0)] = 1;
        auto Tlin = [&](double x) { return x < L ? 2.0 - x / L : 1.0; };
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t q = c.g.idx(i, j, 0);
                c.ls.rho[q] = 1.0 / Tlin((i + 0.5) * h);
                c.ls.rhoY[0][q] = 0;
            }
        for (double& v : c.ls.mx) v = 0;
        FvCase fv = fvOf(c, solid);
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(fv.r.s, fv.r.t, fv.r.g, fv.model, ctlOf(p), fv.ic), "make");
        for (int i = 0; i < 500; ++i) orDie(d.step(0.01), "step");
        const auto o = orDie(d.cells(), "cells");
        const auto fx = orDie(d.faces(), "faces");
        double eT = 0, um = 0;
        for (std::size_t q = 0; q < o.rho.size(); ++q) {
            eT = std::max(eT, std::abs(o.T[q] - Tlin(fv.m.fm.C[0][q])));
            um = std::max({um, std::abs(o.ru[q]), std::abs(o.ru[o.rho.size() + q])});
        }
        double qRim = 0, qIn = 0;
        const MeshPatch& rim = *std::ranges::find(fv.m.fm.patches, std::string("rim"), &MeshPatch::name);
        for (std::size_t q = 0; q < std::size_t(fv.r.s.nB); ++q) {
            const int bf = fv.r.s.bFace[q];
            if (bf >= rim.first && bf < rim.first + rim.count) qRim -= fx.heat[q];
            if (fv.r.s.bCode[q] == std::uint8_t(FvIncompFaceBc::Inflow)) qIn += fx.heat[q];
        }
        const double want = (c.sp.tr.muRef / c.sp.tr.Pr) * (1.0 / L) * (ny * h);
        std::printf("R2 cond %s: max |T - T_lin| %.2e, max |rho u| %.2e; wall heat %.6g, inflow face %.6g, lambda dT/L "
                    "L_y %.6g (over cp)\n",
                    p == FvPrecision::Fp32 ? "fp32" : "fp64", eT, um, qRim, qIn, want);
        const double tol = p == FvPrecision::Fp32 ? 2e-6 : 1e-12;
        check(eT <= tol, "R2 the linear conduction profile is the rim's fixed point");
        check(std::abs(qRim - want) <= 1e-5 * want && std::abs(qIn - qRim) <= 1e-5 * want,
              "R2 the wall heat = lambda dT/L L_y = the inflow face's conduction");
    }
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "flame1d") return runFlame1d();
    if (argc > 1 && std::string(argv[1]) == "bunsen") return runBunsen();
    if (argc > 1 && std::string(argv[1]) == "rich") {
        gateRich();
        gateRichStructured();
        gateRows();
        return dnstest::finish();
    }
    if (argc > 1 && std::string(argv[1]) == "burner") {
        gateOpenRest();
        gateBackflow();
        gateOpenTallies();
        gateOpenStructured(0.0);
        gateOpenStructured(0.5);
        gateHydro();
        gateRimCond();
        gateRimStructured();
        return dnstest::finish();
    }
    if (argc > 1 && std::string(argv[1]) == "momentum") {
        gateMomentum();
        return dnstest::finish();
    }
    gateUniform();
    gateConservation();
    gateEnergyOrder();
    gateRerun();
    gate2D();
    gateFreestream();
    gateStructured();
    gateRich();
    gateRichStructured();
    gateRows();
    return dnstest::finish();
}
