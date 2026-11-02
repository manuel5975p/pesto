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

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "flame1d") return runFlame1d();
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
    return dnstest::finish();
}
