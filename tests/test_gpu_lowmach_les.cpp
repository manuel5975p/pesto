// M4: LES and TFLES on the low-Mach solvers, structured (gpu/lowmach) and FV
// (gpu/fv_incomp DeviceFvLowMach) together (docs/chem.md "Low-Mach LES", docs/fvincomp.md
// "Low-Mach LES" and "Low-Mach combinations").
// Hand gates (default):
//   L1 laminar limit, both solvers: TFLES F 1 (uniform and dynamic sensor) == DNS bitwise;
//      LES sigma on a planar 3D flame (rank-1 velocity gradient: D_sigma 0) == DNS
//      bitwise; LES Smagorinsky on a uniform inert stream (grad u 0) == DNS bitwise
//   X1 TFLES F 2 (uniform, dynamic sensor) FV vs structured, same mesh, same dt, 400
//      steps: rounding, <= 3 x the FP32 run's own distance from FP64 (L7 pattern)
//   X2 LES Smagorinsky Cs 0.17 (2D device level, the 1D flame) FV(FP64) vs
//      structured(FP32): mu_t of one state and rho after a march at h, h/2, h/4 -- the
//      cell-velocity definitions differ (collocated (rho u)/rho vs MAC face mean): the
//      gap shrinks; the DNS gap on the same march is the L7 rounding
// `matrix` (ctest gpu_lm_matrix): the generic gates on every live cell of dns/lmmodules
// that carries an M4 module, FV, 3D box, sine-perturbed box, open box:
//   (a) TFLES F 1 == the cell without TFLES, byte for byte (FP32; without LES: with it
//       Omega > 0 switches the SGS scalar flux off, F 1 is not LES alone)
//   (b) C0: a passive uniform Y 0.25 bitwise (FP32, FP64); C2/C2b: Z == 1 kept;
//       mass and fuel + burn vs their FP64 tallies
//   (c) rerun byte-identical (FP32, FP64)
//   (d) Source / Morton / ColourMajor numbering, FP64: to summation order
// `markdown`: docs/fvincomp.md's generated "Low-Mach combinations" block.
#include "dns/chem.hpp"
#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvincomp.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/fvturb.hpp"
#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/lowmach.hpp"
#include "dns/la/graph.hpp"
#include "dns/lmmodules.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

using namespace dns;
using dnstest::check;
using gpu::DeviceFvLowMach;
using gpu::DeviceLowMach;

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

Transport gas() { return Transport::air(0.72, 1.0, 300.0, 0.72, kGammaAir, kRgasBox, false); }

// The one-step model of test_gpu_fv_lowmach (lnA 2, Ta 8, sigma 5).
FvLowMachModel modelOf(double lnA, int nY)
{
    return {.tr = gas(), .prGas = 1.0, .lnA = lnA, .Ta = 8.0, .qcp = kSigma - 1.0, .nY = nY};
}

// A TFLES setup on the model above: F, the sensor (dynamic: lnAs of the C0 manifold,
// TaS = Ta/2, radius rad cells), the efficiency off.
TflesSetup tflesOf(double F, bool uniform, int rad, bool charlette = false)
{
    const double thetaS = kTflesTaFraction * 8.0;
    return {.F = F,
            .uniform = uniform,
            .charlette = charlette,
            .deltaL = 1.5,
            .nuU = 0.72,
            .lnAs = -tflesSensorLogMax(kSigma, thetaS),
            .TaS = thetaS,
            .dilate = rad};
}

FvLowMachBcs boxBcs(double y0, double y1, bool openY = false)
{
    FvBcSpec s = fvLowMachBoxBc(openY);
    s.patches[0].y = {y0, y1, 0, 0};
    return orDie(fvLowMachBcs(s, 0.98, 1.0), "box bcs");
}

struct Run
{
    FvIncompSetup s;
    FvLowMachTables t;
    la::CellGraph g;
};

Run setupOf(const FaceMesh& fm, FvLowMachBcs b, bool tfles)
{
    if (auto r = bindFvIncompBcs(b.flow, fm); !r) {
        std::printf("FATAL: bind: %s\n", r.error().c_str());
        std::exit(1);
    }
    FvIncompSetup s = orDie(buildFvIncompSetup(fm, b.flow), "setup");
    FvLowMachTables t = buildFvLowMachTables(fm, b, s, 1.0);
    if (tfles)
        if (auto w = fvLowMachTfles(fm, t); !w) check(false, w.error());
    return {.s = std::move(s), .t = std::move(t), .g = la::graphOf(fm)};
}

FvIncompControls ctlOf(FvPrecision p) { return {.tol = p == FvPrecision::Fp64 ? 1e-10 : 1e-6, .precision = p}; }

// --- the 1D box pair (L7's) ----------------------------------------------------------

// The structured state of the tanh front of test_gpu_fv_lowmach L7 on grid g (rows, z
// planes identical); burning C0.
LowMachState frontState(const Grid& g, double x0, double w)
{
    LowMachState ls;
    ls.rho.assign(g.size(), 0.0);
    ls.p.assign(g.size(), 0.0);
    ls.my.assign(g.size(), 0.0);
    ls.mz.assign(g.size(), 0.0);
    ls.mx.assign(std::size_t(g.nx + 1) * g.ny * g.nz, 0.98);
    ls.rhoY[0].assign(g.size(), 0.0);
    ls.nY = 1;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double T = 1.0 + 0.5 * (kSigma - 1.0) * (1.0 + std::tanh(((i + 0.5) * g.hx - x0) / w));
                ls.rho[g.idx(i, j, k)] = 1.0 / T;
                ls.rhoY[0][g.idx(i, j, k)] = (kSigma - T) / (kSigma - 1.0) / T;
            }
    return ls;
}

gpu::LowMachSetup structuredOf(const FvLowMachModel& m)
{
    gpu::LowMachSetup sp;
    sp.tr = m.tr;
    sp.prGas = 1.0;
    sp.lnA = m.lnA;
    sp.Ta = m.Ta;
    sp.qcp = m.qcp;
    sp.mIn = 0.98;
    sp.cs = m.cs;
    sp.sgs = m.sgs;
    sp.tfles = m.tfles;
    return sp;
}

LowMachState marchStructured(const Grid& g, const LowMachState& ic, const FvLowMachModel& m, int steps, double dt)
{
    auto st = orDie(DeviceLowMach::make(g, structuredOf(m)), "structured");
    if (auto e = st.upload(ic); !e) check(false, e.error());
    for (int i = 0; i < steps; ++i)
        if (auto e = st.step(dt); !e) check(false, e.error());
    return orDie(st.download(), "download");
}

// The FV box of g with the structured IC cell for cell (rho u = 0.98).
struct BoxFv
{
    FaceMesh fm;
    Run r;
    FvLowMachIC ic;
};

BoxFv boxFv(const Grid& g, const LowMachState& ls, bool tfles)
{
    BoxFv b{.fm = orDie(fvLowMachBox(g), "box")};
    b.r = setupOf(b.fm, boxBcs(1.0, 0.0), tfles);
    const std::size_t nC = g.size();
    b.ic = {.rho = std::vector<double>(nC),
            .ry0 = std::vector<double>(nC),
            .ry1 = {},
            .ru = std::vector<double>(3 * nC, 0.0),
            .p = std::vector<double>(nC, 0.0)};
    for (std::size_t q = 0; q < nC; ++q) {
        const std::size_t src = std::size_t(b.fm.srcCell[q]);
        b.ic.rho[q] = ls.rho[src];
        b.ic.ry0[q] = ls.rhoY[0][src];
        b.ic.ru[q] = 0.98;
    }
    return b;
}

gpu::FvLowMachCells marchFv(BoxFv& b, const FvLowMachModel& m, FvPrecision p, int steps, double dt)
{
    DeviceFvLowMach d = orDie(DeviceFvLowMach::make(b.r.s, b.r.t, b.r.g, m, ctlOf(p), b.ic), "make fv");
    for (int i = 0; i < steps; ++i) orDie(d.step(dt), "step");
    return orDie(d.cells(), "cells");
}

bool sameState(const LowMachState& a, const LowMachState& b)
{
    return a.rho == b.rho && a.rhoY[0] == b.rhoY[0] && a.mx == b.mx && a.my == b.my && a.mz == b.mz && a.p == b.p;
}

bool sameCells(const gpu::FvLowMachCells& a, const gpu::FvLowMachCells& b)
{
    return a.rho == b.rho && a.ry[0] == b.ry[0] && a.ry[1] == b.ry[1] && a.ru == b.ru && a.p == b.p && a.phi == b.phi &&
           a.phiB == b.phiB;
}

// L1: each module at its laminar limit is the solver without it, byte for byte.
void gateLaminarLimit()
{
    const int steps = 200;
    const double dt = 2e-3;
    struct Case
    {
        std::string name;
        int nz;
        bool inert;
        FvLowMachModel on;
    };
    std::vector<Case> cases;
    {
        FvLowMachModel m = modelOf(2.0, 1);
        m.tfles = tflesOf(1.0, true, 0);
        cases.push_back({"TFLES F 1 uniform", 1, false, m});
        m.tfles = tflesOf(1.0, false, 6);
        cases.push_back({"TFLES F 1 dynamic", 1, false, m});
        m.tfles.reset();
        m.cs = 1.35;
        m.sgs = SgsModel::Sigma;
        cases.push_back({"LES sigma, planar 3D", 4, false, m});
        FvLowMachModel s = modelOf(kNoRate, 1);
        s.cs = 0.17;
        cases.push_back({"LES smag, uniform inert stream", 4, true, s});
    }
    for (const Case& c : cases) {
        const Grid g = Grid::make(64, 4, c.nz, 64 * 0.25, 1.0, c.nz * 0.25);
        LowMachState ic = frontState(g, 8.0, 1.5);
        if (c.inert) {
            std::fill(ic.rho.begin(), ic.rho.end(), 1.0);
            std::fill(ic.rhoY[0].begin(), ic.rhoY[0].end(), 1.0);
        }
        FvLowMachModel off = c.on;
        off.cs = 0;
        off.tfles.reset();
        const LowMachState sOn = marchStructured(g, ic, c.on, steps, dt), sOff = marchStructured(g, ic, off, steps, dt);
        const bool st = sameState(sOn, sOff);
        bool fv = true;
        for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
            BoxFv b = boxFv(g, ic, c.on.tfles.has_value());
            fv = fv && sameCells(marchFv(b, c.on, p, steps, dt), marchFv(b, off, p, steps, dt));
        }
        std::printf("L1 %s: structured %s, FV %s\n", c.name.c_str(), st ? "bitwise" : "DIFFERS", fv ? "bitwise" : "DIFFERS");
        check(st, "L1 structured laminar limit: " + c.name);
        check(fv, "L1 FV laminar limit: " + c.name);
    }
}

// X1: TFLES F 2, FV vs structured on the same mesh, same dt: rounding.
void gateTflesStructured()
{
    const int nx = 128, ny = 4, steps = 400;
    const double h = 0.25, dt = 1e-3; // under F 2 (FE/Pr) both diffusive bounds (1.6e-3)
    const Grid g = Grid::make(nx, ny, 1, nx * h, ny * h, 1.0);
    const LowMachState ic = frontState(g, 16.0, 1.5);
    for (bool uniform : {true, false}) {
        FvLowMachModel m = modelOf(2.0, 1);
        m.tfles = tflesOf(2.0, uniform, 6);
        const LowMachState sf = marchStructured(g, ic, m, steps, dt);
        BoxFv b = boxFv(g, ic, true);
        const auto f32 = marchFv(b, m, FvPrecision::Fp32, steps, dt), f64 = marchFv(b, m, FvPrecision::Fp64, steps, dt);
        // The DNS on the same march: how far TFLES moved the state.
        FvLowMachModel dns = m;
        dns.tfles.reset();
        const LowMachState sd = marchStructured(g, ic, dns, steps, dt);
        double dSt = 0, dSt64 = 0, dOwn = 0, dMod = 0;
        for (std::size_t q = 0; q < g.size(); ++q) {
            const std::size_t s = std::size_t(b.fm.srcCell[q]);
            dSt = std::max({dSt, std::abs(f32.rho[q] - sf.rho[s]), std::abs(f32.ry[0][q] - sf.rhoY[0][s])});
            dSt64 = std::max({dSt64, std::abs(f64.rho[q] - sf.rho[s]), std::abs(f64.ry[0][q] - sf.rhoY[0][s])});
            dOwn = std::max({dOwn, std::abs(f32.rho[q] - f64.rho[q]), std::abs(f32.ry[0][q] - f64.ry[0][q])});
            dMod = std::max(dMod, std::abs(sf.rho[s] - sd.rho[s]));
        }
        std::printf("X1 TFLES F 2 %s: FV32 vs structured %.3e, FV64 vs structured %.3e, FV32 vs FV64 %.3e; TFLES "
                    "moved rho by %.3e\n",
                    uniform ? "uniform" : "dynamic", dSt, dSt64, dOwn, dMod);
        check(dMod > 1e-3, "X1 TFLES moves the state");
        check(dSt <= 3.0 * dOwn, std::string("X1 TFLES FV vs structured: rounding, ") + (uniform ? "uniform" : "dynamic"));
    }
}

// X2: Smagorinsky on the 1D flame: the scheme gap of mu_t shrinks with h.
void gateSmagStructured()
{
    double prevMu = 0, prevRho = 0;
    for (int f : {1, 2, 4}) {
        const int nx = 128 * f, ny = 4;
        const double h = 0.25 / f, dt = 0.03 * h * h, tEnd = 0.25;
        const int steps = int(std::lround(tEnd / dt));
        const Grid g = Grid::make(nx, ny, 1, nx * h, ny * h, 1.0);
        const LowMachState ic = frontState(g, 16.0, 1.5);
        FvLowMachModel m = modelOf(2.0, 1);
        m.cs = 0.17;
        // mu_t of the IC, both solvers.
        auto st = orDie(DeviceLowMach::make(g, structuredOf(m)), "structured");
        if (auto e = st.upload(ic); !e) check(false, e.error());
        st.refreshEddy();
        std::vector<float> mS(g.size());
        (void)st.mut().download(mS.data());
        BoxFv b = boxFv(g, ic, false);
        DeviceFvLowMach d = orDie(DeviceFvLowMach::make(b.r.s, b.r.t, b.r.g, m, ctlOf(FvPrecision::Fp64), b.ic), "fv");
        const auto c0 = orDie(d.cells(), "cells");
        double dMu = 0, muMax = 0;
        for (std::size_t q = 0; q < g.size(); ++q) {
            const std::size_t s = std::size_t(b.fm.srcCell[q]);
            dMu = std::max(dMu, std::abs(c0.mut[q] - double(mS[s])));
            muMax = std::max(muMax, double(mS[s]));
        }
        const LowMachState sf = marchStructured(g, ic, m, steps, dt);
        const auto f64 = marchFv(b, m, FvPrecision::Fp64, steps, dt);
        FvLowMachModel dns = m;
        dns.cs = 0;
        const LowMachState sd = marchStructured(g, ic, dns, steps, dt);
        const auto fd = marchFv(b, dns, FvPrecision::Fp64, steps, dt);
        double dRho = 0, dDns = 0, dMod = 0;
        for (std::size_t q = 0; q < g.size(); ++q) {
            const std::size_t s = std::size_t(b.fm.srcCell[q]);
            dRho = std::max(dRho, std::abs(f64.rho[q] - sf.rho[s]));
            dDns = std::max(dDns, std::abs(fd.rho[q] - sd.rho[s]));
            dMod = std::max(dMod, std::abs(sf.rho[s] - sd.rho[s]));
        }
        std::printf("X2 smag h/%d: mu_t max %.4e, FV - structured %.3e (%.3e of max)%s; rho after t %.2g: LES gap %.3e%s, "
                    "DNS gap %.3e, LES moved rho %.3e\n",
                    f, muMax, dMu, dMu / muMax,
                    prevMu > 0 ? std::format(" order {:.2f}", std::log2(prevMu / (dMu / muMax))).c_str() : "", tEnd,
                    dRho, prevRho > 0 ? std::format(" order {:.2f}", std::log2(prevRho / dRho)).c_str() : "", dDns, dMod);
        if (prevMu > 0) check(dMu / muMax < prevMu, "X2 mu_t gap shrinks with h");
        if (prevRho > 0) check(dRho < prevRho, "X2 rho gap shrinks with h");
        prevMu = dMu / muMax;
        prevRho = dRho;
    }
}

// --- the harness (matrix) -------------------------------------------------------------

struct MCase
{
    std::string name;
    std::array<FaceMesh, 3> fm; // Source, Morton, ColourMajor
    bool open = false;
};

StructuredSpec spec3(const Grid& g)
{
    return {.lo = {0, 0, 0},
            .hi = {g.nx - 1, g.ny - 1, g.nz - 1},
            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
            .type = {FoamPatchType::Patch, FoamPatchType::Patch, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                     FoamPatchType::Cyclic, FoamPatchType::Cyclic}};
}

std::vector<MCase> mcases()
{
    const Grid g = Grid::make(32, 6, 6, 32 * 0.5, 3.0, 3.0);
    constexpr CellOrder ord[3] = {CellOrder::Source, CellOrder::Morton, CellOrder::ColourMajor};
    std::vector<MCase> v(3);
    v[0].name = "box3d";
    v[1].name = "sine3d";
    v[2].name = "open3d";
    v[2].open = true;
    StructuredSpec so = spec3(g);
    so.type[2] = so.type[3] = FoamPatchType::Patch;
    for (int o = 0; o < 3; ++o) {
        v[0].fm[o] = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), spec3(g), ord[o]), "box3d");
        v[1].fm[o] = orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(g, 0.04), spec3(g), ord[o]), "sine3d");
        v[2].fm[o] = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), so, ord[o]), "open3d");
    }
    return v;
}

// The --bc of a mesh case: xlo inflow, xhi outflow, ylo/yhi open with the ambient (open).
// The ambient keeps the invariants: the passive Y 0.25 (C0), the fresh premix at Z 1 (C2).
FvLowMachBcs mBcs(const MCase& m, bool two)
{
    const PartialPremixModel pm;
    FvBcSpec s = fvLowMachBoxBc(m.open, 1.0, two ? std::array<double, kFvLmMaxY>{pm.yF1(), pm.yO1()}
                                                 : std::array<double, kFvLmMaxY>{0.0, 0.25});
    if (two) {
        s.patches[0].y = {pm.yF1(), pm.yO1(), 0, 0};
    } else {
        s.patches[0].y = {1.0, 0.25, 0, 0};
    }
    // The z cyclics of fvLowMachBoxBc's box are `empty` at nz == 1: none here.
    return orDie(fvLowMachBcs(s, 0.98, 1.0), "matrix bcs");
}

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

double richZ(double yF, double yO)
{
    const PartialPremixModel& pm = rich().pm;
    return (pm.s * yF - yO + pm.yO2) / (pm.s * pm.yFrem() + pm.yO2);
}

// A wrinkled warm front with a weak 3D swirl (every velocity-gradient component non-zero
// after the first projection): C0 + a passive uniform Y 0.25, or C2's reactants at Z 1.
FvLowMachIC mIC(const FaceMesh& fm, bool two)
{
    const std::size_t nC = std::size_t(fm.nC);
    const double xc = 8.0, w = 1.2, tau = 2 * std::numbers::pi;
    const double sig = two ? rich().pm.sigma : kSigma;
    FvLowMachIC ic{.rho = std::vector<double>(nC),
                   .ry0 = std::vector<double>(nC),
                   .ry1 = std::vector<double>(nC),
                   .ru = std::vector<double>(3 * nC),
                   .p = std::vector<double>(nC)};
    for (std::size_t c = 0; c < nC; ++c) {
        const double x = fm.C[0][c], y = fm.C[1][c], z = fm.C[2][c];
        const double xf = xc + 0.6 * std::sin(tau * y / 3.0) * std::cos(tau * z / 3.0);
        const double T = 1.0 + 0.5 * (sig - 1.0) * (1.0 + std::tanh((x - xf) / w));
        ic.rho[c] = 1.0 / T;
        if (two) {
            const PartialPremixModel& pm = rich().pm;
            const double sp = (pm.sigma - T) / (pm.sigma - 1.0);
            const double yO = pm.yO1() * sp, yF = pm.yFrem() + yO / pm.s;
            ic.ry0[c] = ic.rho[c] * yF;
            ic.ry1[c] = ic.rho[c] * yO;
        } else {
            ic.ry0[c] = ic.rho[c] * (kSigma - T) / (kSigma - 1.0);
            ic.ry1[c] = ic.rho[c] * 0.25;
        }
        ic.ru[c] = 0.98;
        ic.ru[nC + c] = 0.05 * std::sin(tau * z / 3.0);
        ic.ru[2 * nC + c] = 0.05 * std::sin(tau * y / 3.0);
    }
    return ic;
}

FvLowMachModel cellModel(const LmModuleSet& s, double F)
{
    using M = LmModule;
    const bool two = s.test(std::size_t(M::ChemC2)) || s.test(std::size_t(M::ChemC2b));
    FvLowMachModel m = modelOf(2.0, 2);
    if (two) {
        const Rich& r = rich();
        m.lnA = r.lnA;
        m.Ta = r.pm.theta();
        m.qcp = r.pm.qOverCpTu();
        m.ox = 1;
        m.stoich = r.pm.s;
        if (s.test(std::size_t(M::ChemC2b))) m.law = r.law;
    }
    if (s.test(std::size_t(M::LesSmag))) m.cs = 0.17;
    if (s.test(std::size_t(M::LesSigma))) {
        m.cs = 1.35;
        m.sgs = SgsModel::Sigma;
    }
    const bool tu = s.test(std::size_t(M::TflesUniform)), td = s.test(std::size_t(M::TflesDynamic));
    if (tu || td) m.tfles = tflesOf(F, tu, 3, s.test(std::size_t(M::TflesEfficiency)));
    if (s.test(std::size_t(M::Gravity))) m.gravity = {-0.5, 0.0, 0.0};
    return m;
}

struct MRun
{
    gpu::FvLowMachCells c;
    gpu::FvLowMachTotals t;
};

// turbAmp >= 0 (M5 Turb): the xlo faces' bU, rUb rewritten before every step from a
// small turb_ic box (u' turbAmp, l_t 0.5, 8^3 over 16 x 3 x 3) swept at 0.98; 0 = the
// tables bit for bit (the inert setting).
MRun mRun(const FaceMesh& fm, const FvLowMachBcs& b, const FvLowMachModel& m, FvPrecision p, int steps,
          double turbAmp = -1)
{
    Run r = setupOf(fm, b, m.tfles.has_value());
    const bool two = m.ox == 1;
    DeviceFvLowMach d = orDie(DeviceFvLowMach::make(r.s, r.t, r.g, m, ctlOf(p), mIC(fm, two)), "make");
    const auto bd = orDie(d.stableDt(0.8, 1.0), "dt");
    const double dt = 0.5 * std::min({bd.conv, bd.diff, bd.chem});
    std::optional<TurbSweep> sw;
    std::vector<int> q;
    if (turbAmp >= 0) {
        TurbSpec ts;
        ts.urms = turbAmp > 0 ? turbAmp : 1.0;
        ts.lt = 0.5;
        ts.Lx = 16.0;
        ts.Ly = ts.Lz = 3.0;
        ts.nx = ts.ny = ts.nz = 8;
        std::vector<TurbMode> modes = turbModes(ts);
        if (turbAmp == 0)
            for (auto& t : modes) t.uh[0] = t.uh[1] = t.uh[2] = 0.0;
        const auto in = orDie(fvPatchFaces(fm, "xlo"), "xlo");
        std::vector<std::array<double, 3>> pts;
        for (int k = 0; k < r.s.nB; ++k) {
            const auto it = std::ranges::find(in.b, r.s.bFace[std::size_t(k)]);
            if (it == in.b.end()) continue;
            q.push_back(k);
            pts.push_back(in.cf[std::size_t(it - in.b.begin())]);
        }
        sw = TurbSweep::make(modes, ts, pts);
    }
    std::array<std::vector<double>, 3> du;
    std::vector<double> bU, rUb;
    for (int i = 0; i < steps; ++i) {
        if (sw) {
            sw->sample(0.98 * dt * i, du);
            fvLowMachTurbRows(q, du, r.s.bU, r.t.rUb, r.s.bT, 1.0, bU, rUb);
            if (auto e = d.setInflow(bU, rUb); !e) check(false, e.error());
        }
        orDie(d.step(dt), "step");
    }
    MRun o{.t = orDie(d.totals(), "totals")};
    o.c = orDie(d.cells(), "cells");
    return o;
}

int runMatrix()
{
    using M = LmModule;
    // Every live set containing an M4 module: singles and pairs, closed under the
    // requirements, the burner-only modules (rim) left out (refused with LES/TFLES).
    const M m4[] = {M::LesSmag, M::LesSigma, M::TflesUniform, M::TflesDynamic, M::TflesEfficiency, M::Turb};
    std::vector<LmModuleSet> sets;
    auto add = [&](LmModuleSet s) {
        s = lmModulesClosure(s);
        if (!s.test(std::size_t(M::ChemC2)) && !s.test(std::size_t(M::ChemC2b))) s.set(std::size_t(M::ChemC0));
        if (!lmModulesCheck(s) || s.test(std::size_t(M::Rim))) return;
        if (std::ranges::find(sets, s) == sets.end()) sets.push_back(s);
    };
    for (const M a : m4) {
        add(lmModuleSet({a}));
        for (std::size_t b = 0; b < kLmModuleCount; ++b)
            if (M(b) != a && lmPair(a, M(b)).decision == FvPairDecision::Live) add(lmModuleSet({a, M(b)}));
    }
    std::vector<MCase> mc = mcases();
    const int steps = 10;
    int rows = 0;
    for (const LmModuleSet& s : sets) {
        const bool two = s.test(std::size_t(M::ChemC2)) || s.test(std::size_t(M::ChemC2b));
        const bool thick = s.test(std::size_t(M::TflesUniform)) || s.test(std::size_t(M::TflesDynamic));
        for (MCase& mesh : mc) {
            if (mesh.open != s.test(std::size_t(M::Open))) continue;
            const FvLowMachBcs b = mBcs(mesh, two);
            const FvLowMachModel model = cellModel(s, 2.0);
            const std::string tag = std::format("[{} on {}]", lmModuleSetName(s), mesh.name);
            std::string notes;
            // (a) TFLES F 1 == the cell without TFLES (FP32). Not with LES: Omega > 0
            // switches the SGS scalar flux off there, so F 1 is not LES alone (the
            // structured rule, docs/chem.md "LES species and the thickened flame").
            const bool les = s.test(std::size_t(M::LesSmag)) || s.test(std::size_t(M::LesSigma));
            if (thick && les) notes += " (a) n/a (LES: Omega switches the SGS flux);";
            if (thick && !les) {
                FvLowMachModel one = cellModel(s, 1.0), none = one;
                none.tfles.reset();
                const bool eq = sameCells(mRun(mesh.fm[1], b, one, FvPrecision::Fp32, steps).c,
                                          mRun(mesh.fm[1], b, none, FvPrecision::Fp32, steps).c);
                check(eq, "(a) TFLES F 1 is the cell without TFLES " + tag);
                notes += eq ? " (a) bitwise;" : " (a) DIFFERS;";
            }
            // M5 Turb: rows of u' 0 are the tables: the cell without Turb byte for byte.
            const bool turb = s.test(std::size_t(M::Turb));
            const double amp = turb ? 0.1 : -1.0;
            if (turb) {
                bool eq = true;
                for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64})
                    eq = eq && sameCells(mRun(mesh.fm[1], b, model, p, steps, 0.0).c, mRun(mesh.fm[1], b, model, p, steps).c);
                check(eq, "(a) turb u' 0 is the cell without Turb " + tag);
                notes += eq ? " (a) turb u' 0 bitwise;" : " (a) turb DIFFERS;";
            }
            // (b) invariants, (c) rerun.
            for (auto p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
                const bool f64 = p == FvPrecision::Fp64;
                const MRun r1 = mRun(mesh.fm[1], b, model, p, steps, amp), r2 = mRun(mesh.fm[1], b, model, p, steps, amp);
                const bool rerun = sameCells(r1.c, r2.c);
                check(rerun, std::format("(c) rerun {} {}", f64 ? "fp64" : "fp32", tag));
                double inv = 0;
                std::size_t off = 0;
                for (std::size_t q = 0; q < r1.c.rho.size(); ++q) {
                    if (two)
                        inv = std::max(inv, std::abs(richZ(r1.c.ry[0][q] / r1.c.rho[q], r1.c.ry[1][q] / r1.c.rho[q]) - 1.0));
                    else
                        off += r1.c.ry[1][q] != 0.25 * r1.c.rho[q];
                }
                const gpu::FvLowMachTotals& t = r1.t;
                const double em = std::abs(t.mass - t.massTally) / t.mass;
                const double ef = std::abs(t.y[0] + t.burn - t.yTally[0]) / t.y[0];
                const double tol = f64 ? 1e-13 : 1e-7;
                if (two) check(inv <= (f64 ? 1e-12 : 1e-6), std::format("(b) Z == 1 kept {} {}", f64 ? "fp64" : "fp32", tag));
                else check(off == 0, std::format("(b) uniform Y bitwise {} {}", f64 ? "fp64" : "fp32", tag));
                check(em <= tol, std::format("(b) mass tally {} {}", f64 ? "fp64" : "fp32", tag));
                check(ef <= tol, std::format("(b) fuel tally {} {}", f64 ? "fp64" : "fp32", tag));
                notes += std::format(" {}: {} {:.1e}, mass {:.1e}, fuel {:.1e}, rerun {};", f64 ? "fp64" : "fp32",
                                     two ? "|Z-1|" : "Y off", two ? inv : double(off), em, ef, rerun ? "=" : "DIFFERS");
            }
            // (d) numbering, FP64, mapped by srcCell.
            {
                std::array<MRun, 3> o;
                for (int k = 0; k < 3; ++k) o[std::size_t(k)] = mRun(mesh.fm[k], b, model, FvPrecision::Fp64, steps, amp);
                double dmax = 0;
                bool bit = true;
                for (int k : {0, 2}) {
                    std::vector<double> rho(o[1].c.rho.size()), ry(rho.size()), r1(rho.size()), y1(rho.size());
                    for (std::size_t q = 0; q < rho.size(); ++q) {
                        rho[std::size_t(mesh.fm[k].srcCell[q])] = o[std::size_t(k)].c.rho[q];
                        ry[std::size_t(mesh.fm[k].srcCell[q])] = o[std::size_t(k)].c.ry[0][q];
                        r1[std::size_t(mesh.fm[1].srcCell[q])] = o[1].c.rho[q];
                        y1[std::size_t(mesh.fm[1].srcCell[q])] = o[1].c.ry[0][q];
                    }
                    for (std::size_t q = 0; q < rho.size(); ++q) {
                        dmax = std::max({dmax, std::abs(rho[q] - r1[q]), std::abs(ry[q] - y1[q])});
                        bit = bit && rho[q] == r1[q] && ry[q] == y1[q];
                    }
                }
                check(dmax <= 1e-11, "(d) numbering " + tag);
                notes += bit ? " (d) bitwise" : std::format(" (d) {:.1e}", dmax);
            }
            std::printf("| %s | %s |%s |\n", lmModuleSetName(s).c_str(), mesh.name.c_str(), notes.c_str());
            std::fflush(stdout);
            ++rows;
        }
    }
    std::printf("gpu_lm_matrix: %zu live sets, %d rows\n", sets.size(), rows);
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    if (argc > 1 && std::string(argv[1]) == "markdown") {
        std::printf("%s", lmCombinationsMarkdown().c_str());
        return 0;
    }
    test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "matrix") return runMatrix();
    if (argc > 1 && std::string(argv[1]) == "smag") {
        gateSmagStructured();
        return dnstest::finish();
    }
    gateLaminarLimit();
    gateTflesStructured();
    return dnstest::finish();
}
