// M5: --turb on the FV flame routes and on the low-Mach solvers (docs/fvcomp.md
// "Turbulence", docs/chem.md "Low-Mach turbulence", dns/fvturb, dns/lowmach).
// Default (ctest gpu_fv_turb), the compressible FV family:
//   T1 IC: the FV point sum at the centroids (fvAddTurbulence) vs the structured
//      addTurbulence on turbBox, the flags' box via the real CLI: rounding (1e-12 u')
//   T2 zero amplitude: per-face Far rows rewritten every step with u' 0 (fvFarTurbRow,
//      setBcRows) == the laminar table, byte for byte (FP32, compensated, FP64)
//   T3 the rows reach the flow: u' 1 rows make the first cell layer's v, w nonzero, the
//      inflow mass rate within 1e-3 of the laminar run's (plane-mean u' = 0)
//   X3 FV vs structured, a turbulent burning slab on the same 3D periodic box, same dt,
//      200 steps: rounding (FV FP32 vs structured <= 3 x the FP32 run's own distance
//      from FP64; the X1/S8 pattern)
// `lowmach` (ctest gpu_lm_turb), both low-Mach solvers:
//   L1 zero amplitude: setInflowTurb(mIn, 0, 0) every step (structured, the Turb
//      instantiation) and setInflow(the make tables) (FV) == the solvers without, byte
//      for byte
//   L2 structured vs FV, the turbulent planar flame of the CLI (IC + swept rows), same
//      dt: NOT rounding (MAC vs collocated, staggered vs centred row positions): the
//      gap is measured at h and h/2 against the laminar gap of the same march and
//      must shrink with h
#include "fv_comp_test_util.hpp"

#include "cli_test_util.hpp"
#include "dns/flame1d.hpp"
#include "dns/fvincomp.hpp"
#include "dns/fvflame_case.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/fvturb.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/fv_incomp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "dns/gpu/lowmach.hpp"
#include "dns/la/graph.hpp"
#include "dns/lowmach.hpp"
#include "dns/turb_ic.hpp"
#include "gpu_test_util.hpp"

#include <cstring>
#include <numbers>

using namespace dnstest::fvc;
using dnstest::check;
using gpu::DeviceFvComp;
using gpu::DeviceFvLowMach;
using gpu::DeviceLowMach;
using gpu::FvCompConfig;

namespace {

Case caseOf(std::vector<std::string> args)
{
    auto c = dnstest::runCli(std::move(args));
    if (!c) {
        std::printf("FATAL: case: %s %s\n", c.error().field.c_str(), c.error().message.c_str());
        std::exit(1);
    }
    return std::move(*c);
}

bool sameBytes(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

// --- the compressible FV box of the case (dns/fvflame_case: runFvFlame's, C0) ------------------------

struct CompBox
{
    Case c;
    FaceMesh fm{};
    FvBcSpec spec{};
    FvFreestream fs{};
    Transport tr{};
    FvChem chem{};
    FvState ic{};
    FvTurb turb{};
};

CompBox compBox(const std::vector<std::string>& args)
{
    CompBox b{.c = caseOf(args)};
    const Case& c = b.c;
    FvFlameBox box = fvFlameBox(c);
    b.fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(c.grid), box.ss, CellOrder::Morton), "box");
    b.spec = std::move(box.spec);
    const FvFlameSetup s = fvFlameSetup(c, b.spec);
    b.fs = s.fs;
    b.tr = s.gas;
    b.chem = s.chem;
    b.ic = fvFlameIC(c, b.fm, b.spec, s);
    b.turb = fvFlameTurb(c);
    return b;
}

// The table with per-face xlo rows (fvFaceRows, runFvFlame's), their face list and the first row.
struct Table
{
    std::vector<FvBc> bc;
    std::vector<int> bcRow;
    int first = 0;
    FvPatchFaces in;
};

Table tableOf(const CompBox& b, bool rows)
{
    Table t{.bc = fvBcsOf(b.fm, b.spec, b.fs, 1.0), .bcRow = {}, .first = 0, .in = {}};
    t.first = int(t.bc.size());
    if (!rows) {
        t.in = orDie(fvPatchFaces(b.fm, "xlo"), "xlo");
        return t;
    }
    FvFaceRows r = orDie(fvFaceRows(b.fm, t.bc, "xlo"), "xlo");
    t.bcRow = std::move(r.bcRow);
    t.in = std::move(r.faces);
    return t;
}

// steps of the box from s; sw non-null: the rows of sw (amp scales the modes' field)
// rewritten every step at X = sum U dt. Returns the final state.
FvState marchBox(const CompBox& b, const FvState& s, FvPrecision pr, bool comp, int steps, double dt, const TurbSweep* sw)
{
    const Table t = tableOf(b, sw != nullptr);
    DeviceFvComp d = orDie(DeviceFvComp::make(b.fm, {.tr = b.tr,
                                                      .bc = t.bc,
                                                      .bcRow = t.bcRow,
                                                      .precision = pr,
                                                      .compensated = comp,
                                                      .chem = b.chem,
                                                      .turb = sw != nullptr}),
                           "box");
    orDie(d.upload(s), "upload");
    std::array<std::vector<double>, 3> u;
    std::vector<FvBc> rows(t.in.b.size());
    double X = 0;
    for (int n = 0; n < steps; ++n) {
        if (sw) {
            sw->sample(X, u);
            for (std::size_t i = 0; i < rows.size(); ++i)
                rows[i] = fvFarTurbRow(b.spec.patches[0], b.fs, {u[0][i], u[1][i], u[2][i]});
            orDie(d.setBcRows(t.first, rows), "rows");
        }
        d.step(dt);
        X += b.fs.U * dt;
    }
    FvState o;
    orDie(d.download(o), "download");
    return o;
}

const std::vector<std::string> kCompArgs = {"flame", "--solver",  "fv",   "--turb", "1",  "--turb-length", "1",
                                            "--cells-per-delta", "4", "--ny", "24", "--nz", "24", "--upstream",
                                            "6",     "--downstream", "4", "--tt", "1", "--quiet"};

void gateIC()
{
    std::printf("== T1 FV IC (point sum at the centroids) vs the structured addTurbulence (turbBox)\n");
    const CompBox b = compBox(kCompArgs);
    const Case& c = b.c;
    FvState fv = b.ic;
    fvAddTurbulence(b.fm, b.turb, fv);
    CompState host = CompState::zeros(c.grid);
    std::vector<double> ry;
    flameIC(c, host, ry);
    addTurbulence(c, turbBox(flameTurbSpec(c)), host);
    double em = 0, eE = 0, Escale = 0;
    for (std::size_t n = 0; n < std::size_t(b.fm.nC); ++n) {
        const std::size_t s = std::size_t(b.fm.srcCell[n]);
        em = std::max({em, std::abs(fv.var(1)[n] - host.mx[s]), std::abs(fv.var(2)[n] - host.my[s]),
                       std::abs(fv.var(3)[n] - host.mz[s])});
        eE = std::max(eE, std::abs(fv.var(4)[n] - host.E[s]));
        Escale = std::max(Escale, std::abs(host.E[s]));
    }
    std::printf("  %d cells, %zu modes: max |m_FV - m_structured| %.2e (u' %g), E %.2e of max E\n", b.fm.nC,
                b.turb.modes.size(), em, c.flameTurb, eE / Escale);
    check(em <= 1e-12 * c.flameTurb, "T1 FV IC momentum == structured to 1e-12 u' (direct sum vs FFTW c2r)");
    check(eE <= 1e-14 * Escale, "T1 FV IC energy == structured to 1e-14 of E");
}

void gateRows()
{
    std::printf("== T2 zero-amplitude rows == the laminar table; T3 the rows reach the flow\n");
    const CompBox b = compBox(kCompArgs);
    std::vector<TurbMode> zero = b.turb.modes;
    for (auto& t : zero) t.uh[0] = t.uh[1] = t.uh[2] = 0.0;
    const Table t = tableOf(b, false);
    const TurbSweep sz = TurbSweep::make(zero, b.turb.spec, t.in.cf), s1 = TurbSweep::make(b.turb.modes, b.turb.spec, t.in.cf);
    DeviceFvComp probe = orDie(DeviceFvComp::make(b.fm, {.tr = b.tr, .bc = t.bc, .chem = b.chem}), "probe");
    orDie(probe.upload(b.ic), "upload");
    const double dts = 0.5 * std::min(orDie(probe.stableDt({.cfl = 0.4}), "dt"), orDie(probe.chemDt(0.5), "chem"));
    const int steps = 60;
    struct Mode
    {
        FvPrecision pr;
        bool comp;
        const char* name;
    };
    for (const Mode md : {Mode{FvPrecision::Fp32, false, "fp32"}, Mode{FvPrecision::Fp32, true, "compensated"},
                          Mode{FvPrecision::Fp64, false, "fp64"}}) {
        const FvState a = marchBox(b, b.ic, md.pr, md.comp, steps, dts, nullptr);
        const FvState z = marchBox(b, b.ic, md.pr, md.comp, steps, dts, &sz);
        const bool eq = sameBytes(a.q, z.q);
        std::printf("  %s, %d steps: u' 0 rows vs laminar table %s\n", md.name, steps, eq ? "byte-identical" : "DIFFER");
        check(eq, std::string("T2 zero-amplitude rows == laminar, ") + md.name);
    }
    // T3: u' 1 rows, FP64.
    const FvState lam = marchBox(b, b.ic, FvPrecision::Fp64, false, steps, dts, nullptr);
    const FvState tur = marchBox(b, b.ic, FvPrecision::Fp64, false, steps, dts, &s1);
    double vmax = 0, m0 = 0, m1 = 0;
    for (std::size_t n = 0; n < std::size_t(b.fm.nC); ++n) {
        m0 += lam.var(0)[n] * b.fm.V[n];
        m1 += tur.var(0)[n] * b.fm.V[n];
        if (b.fm.C[0][n] < b.c.grid.hx) vmax = std::max({vmax, std::abs(tur.var(2)[n] / tur.var(0)[n]),
                                                       std::abs(tur.var(3)[n] / tur.var(0)[n])});
    }
    std::printf("  u' 1: first-layer max |v|, |w| %.3f u'; mass vs laminar %.2e\n", vmax / b.c.flameTurb, m1 / m0 - 1);
    check(vmax > 0.1 * b.c.flameTurb, "T3 the rows' v, w enter the first cell layer");
    check(std::abs(m1 / m0 - 1) < 1e-3, "T3 the inflow mass rate is the laminar one to 1e-3 (plane-mean u' 0)");
}

// X3: the S8 pattern in 3D with the turbulent IC: a burning slab in a periodic box.
void gateStructured()
{
    std::printf("== X3 FV vs structured, a turbulent burning slab, 3D periodic, same dt\n");
    const double M = 0.05, sigma = 5.0;
    PremixedModel md;
    md.sigma = sigma;
    const PremixedEigen eig = orDie(premixedEigen(md), "oracle");
    const double Tu = 1.0 / (kGamma * M * M), h = eig.dthOverDD / 6.0;
    const int nb = 16;
    const Grid g = Grid::make(48, nb, nb, 48 * h, nb * h, nb * h);
    const Transport tr = Transport::air(md.Pr, Tu, 288.15, md.Pr, kGamma, 1.0, false);
    const double q = tr.cp() * (sigma - 1.0) * Tu;
    TurbSpec ts;
    ts.urms = 1.0;
    ts.lt = 2.5 * h;
    ts.nx = ts.ny = ts.nz = nb;
    ts.Lx = ts.Ly = ts.Lz = nb * h;
    ts.seed = 3;
    const auto box = turbBox(ts);
    auto ic = CompState::zeros(g);
    std::vector<double> ry(g.size());
    const double Lx = g.nx * g.hx, w = 1.5 * eig.dthOverDD;
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t c = g.idx(i, j, k);
                const double x = (i + 0.5) * g.hx;
                const double t = 1.0 + 0.5 * (sigma - 1.0) * (std::tanh((x - 0.3 * Lx) / w) - std::tanh((x - 0.7 * Lx) / w));
                const double rho = 1.0 / t;
                const std::size_t bq = std::size_t(i % nb) + std::size_t(nb) * (j + std::size_t(nb) * k);
                const double u[3] = {box[0][bq], box[1][bq], box[2][bq]};
                ic.rho[c] = rho;
                ic.mx[c] = rho * u[0];
                ic.my[c] = rho * u[1];
                ic.mz[c] = rho * u[2];
                ic.E[c] = Tu / (kGamma - 1.0) + 0.5 * rho * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                ry[c] = rho * (sigma - t) / (sigma - 1.0);
            }
    constexpr int steps = 200;
    auto dev = gpu::DeviceComp::make(g, tr);
    auto red = gpu::ReduceScratch::make();
    if (!dev || !red) std::exit(EXIT_FAILURE);
    gpu::SpeciesSetup ss;
    ss.lnA = std::log(eig.Lambda);
    ss.Ta = md.theta() * Tu;
    ss.q = q;
    ss.invPr = 1.0 / md.Pr;
    if (!dev->enableSpecies(ss)) std::exit(EXIT_FAILURE);
    auto ds = gpu::DeviceCompState::make(g.size(), 0, false, 1);
    if (!ds || ds->upload(ic) != hipSuccess || ds->uploadSpecies(0, ry) != hipSuccess) std::exit(EXIT_FAILURE);
    const float dt = 0.5f * float(dev->stableDt(*ds, *red, 0.4));
    for (int n = 0; n < steps; ++n) dev->step(*ds, gpu::Real(dt));
    auto ho = CompState::zeros(g);
    std::vector<double> yo;
    if (ds->download(ho) != hipSuccess || ds->downloadSpecies(0, yo) != hipSuccess) std::exit(EXIT_FAILURE);
    FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "box");
    const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return FvBc{}; });
    const FvChem chem{.nY = 1, .fuel = 0, .lnA = ss.lnA, .Ta = ss.Ta, .q = q, .invPr = ss.invPr};
    const std::vector<double>* src[6] = {&ic.rho, &ic.mx, &ic.my, &ic.mz, &ic.E, &ry};
    const std::vector<double>* hv[6] = {&ho.rho, &ho.mx, &ho.my, &ho.mz, &ho.E, &yo};
    FvState s = FvState::zeros(fm.nC, 6), st = FvState::zeros(fm.nC, 6);
    for (int r = 0; r < 6; ++r)
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            s.var(r)[c] = (*src[r])[std::size_t(fm.srcCell[c])];
            st.var(r)[c] = (*hv[r])[std::size_t(fm.srcCell[c])];
        }
    FvState o32, o64;
    for (const FvPrecision pr : {FvPrecision::Fp32, FvPrecision::Fp64}) {
        DeviceFvComp d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .precision = pr, .chem = chem}), "fv");
        orDie(d.upload(s), "upload");
        for (int n = 0; n < steps; ++n) d.step(double(dt));
        orDie(d.download(pr == FvPrecision::Fp32 ? o32 : o64), "download");
    }
    double ke0 = 0, ke1 = 0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        ke0 += (s.var(2)[c] * s.var(2)[c] + s.var(3)[c] * s.var(3)[c]) / s.var(0)[c];
        ke1 += (o64.var(2)[c] * o64.var(2)[c] + o64.var(3)[c] * o64.var(3)[c]) / o64.var(0)[c];
    }
    const double a = relDiff(o32.q, st.q, 6, fm.nC), own = relDiff(o32.q, o64.q, 6, fm.nC),
                 st64 = relDiff(st.q, o64.q, 6, fm.nC);
    std::printf("  %d x %d x %d, %d steps, transverse KE %.3f of its IC; FV FP32 vs structured %.3e, structured vs FV "
                "FP64 %.3e, FV FP32 vs FP64 %.3e\n",
                g.nx, g.ny, g.nz, steps, ke1 / ke0, a, st64, own);
    check(ke1 < ke0 && ke1 > 1e-3 * ke0, "X3 the turbulence decays (Re_t ~ 1 at this box), not gone");
    check(a <= 3.0 * own, std::format("X3 FV FP32 vs structured {:.2e} <= 3 x own {:.2e}", a, own));
    check(st64 <= 3.0 * own, std::format("X3 structured vs FV FP64 {:.2e} <= 3 x own {:.2e}", st64, own));
}

// --- low-Mach ---------------------------------------------------------------------------

const std::vector<std::string> kLmArgs = {"flame", "--physics", "lowmach", "--turb", "1", "--turb-length", "1",
                                          "--cells-per-delta", "4", "--ny", "24", "--nz", "24", "--upstream", "6",
                                          "--downstream", "4", "--tt", "1", "--quiet"};

gpu::LowMachSetup lmSetup(const Case& c, bool turb)
{
    gpu::LowMachSetup sp;
    const double Tu = c.flameTu, sig = c.flameModel.sigma;
    sp.tr = flameGas(c, Tu);
    sp.prGas = Tu;
    sp.lnA = std::log(c.flameEig.Lambda);
    sp.Ta = c.flameModel.theta() * Tu;
    sp.qcp = (sig - 1) * Tu;
    sp.mIn = c.flameUIn;
    sp.TIn = Tu;
    sp.yIn[0] = 1.0;
    sp.turb = turb;
    return sp;
}

struct LmFv
{
    FaceMesh fm;
    FvIncompSetup s{};
    FvLowMachTables t{};
    la::CellGraph g{};
    FvLowMachModel model{};
    FvLowMachIC ic{};
};

// The FV low-Mach box of the case (dns/fvflame_case: runFvLowMachCase's planar route).
LmFv lmFv(const Case& c, bool turb)
{
    LmFv o{.fm = orDie(fvLowMachCaseBox(c), "box")};
    const FvBcSpec spec = fvLowMachCaseSpec(c);
    FvLowMachBcs b = orDie(fvLowMachBcs(spec, c.flameUIn, c.flameTu), "bcs");
    if (auto r = bindFvIncompBcs(b.flow, o.fm); !r) check(false, r.error());
    o.s = orDie(buildFvIncompSetup(o.fm, b.flow), "setup");
    o.t = buildFvLowMachTables(o.fm, b, o.s, c.flameTu);
    o.g = la::graphOf(o.fm);
    const double sig = flameBurnt(c).sigma;
    o.model = fvLowMachCaseModel(c, spec, sig);
    o.ic = fvLowMachCaseIC(c, o.fm, spec, FvBurnerFrame{}, sig);
    if (turb) fvLowMachAddTurbulence(o.fm, fvFlameTurb(c), o.ic);
    return o;
}

// The structured march: rows from `sw` (null: off; zero: the Turb instantiation at 0).
LowMachState lmMarch(const Case& c, const LowMachState& ic, bool turb, const LowMachTurbSweeps* sw, int steps, double dt)
{
    auto d = orDie(DeviceLowMach::make(c.grid, lmSetup(c, turb)), "structured");
    if (auto e = d.upload(ic); !e) check(false, e.error());
    std::vector<double> m, v, w;
    const std::size_t nr = std::size_t(c.grid.ny) * c.grid.nz;
    double X = 0;
    for (int n = 0; n < steps; ++n) {
        if (turb) {
            if (sw) {
                lowmachTurbRows(*sw, X, c.flameUIn, m, v, w);
            } else {
                m.assign(nr, c.flameUIn);
                v.assign(nr, 0.0);
                w.assign(nr, 0.0);
            }
            if (auto e = d.setInflowTurb(m, v, w); !e) check(false, e.error());
        }
        if (auto e = d.step(dt); !e) check(false, e.error());
        X += c.flameUIn * dt;
    }
    return orDie(d.download(), "download");
}

// The FV march: rows from sw over the xlo faces (null: off; zero sweep: the tables).
gpu::FvLowMachCells fvMarch(const LmFv& f, const Case& c, FvPrecision p, const TurbSweep* sw, bool setTables, int steps,
                            double dt)
{
    DeviceFvLowMach d = orDie(DeviceFvLowMach::make(f.s, f.t, f.g, f.model, {.tol = p == FvPrecision::Fp64 ? 1e-10 : 1e-6,
                                                                              .precision = p},
                                                     f.ic),
                              "fv");
    const auto in = orDie(fvPatchFaces(f.fm, "xlo"), "xlo");
    std::vector<int> q;
    for (int k = 0; k < f.s.nB; ++k)
        if (std::ranges::find(in.b, f.s.bFace[std::size_t(k)]) != in.b.end()) q.push_back(k);
    std::array<std::vector<double>, 3> du;
    std::vector<double> bU, rUb;
    double X = 0;
    for (int n = 0; n < steps; ++n) {
        if (sw) {
            sw->sample(X, du);
            fvLowMachTurbRows(q, du, f.s.bU, f.t.rUb, f.s.bT, c.flameTu, bU, rUb);
            orDie(d.setInflow(bU, rUb), "setInflow");
        } else if (setTables) {
            orDie(d.setInflow(f.s.bU, f.t.rUb), "setInflow");
        }
        orDie(d.step(dt), "step");
        X += c.flameUIn * dt;
    }
    return orDie(d.cells(), "cells");
}

bool sameLm(const LowMachState& a, const LowMachState& b)
{
    return a.rho == b.rho && a.rhoY[0] == b.rhoY[0] && a.mx == b.mx && a.my == b.my && a.mz == b.mz && a.p == b.p;
}

bool sameCells(const gpu::FvLowMachCells& a, const gpu::FvLowMachCells& b)
{
    return a.rho == b.rho && a.ry[0] == b.ry[0] && a.ru == b.ru && a.p == b.p && a.phi == b.phi && a.phiB == b.phiB;
}

std::vector<std::array<double, 3>> xloCentres(const LmFv& f)
{
    const auto in = orDie(fvPatchFaces(f.fm, "xlo"), "xlo");
    std::vector<std::array<double, 3>> pts;
    for (int k = 0; k < f.s.nB; ++k) {
        const auto it = std::ranges::find(in.b, f.s.bFace[std::size_t(k)]);
        if (it != in.b.end()) pts.push_back(in.cf[std::size_t(it - in.b.begin())]);
    }
    return pts;
}

void gateLmZero()
{
    std::printf("== L1 zero-amplitude rows == the low-Mach solvers without, both solvers\n");
    const Case c = caseOf(kLmArgs);
    LowMachState ic = lowmachIC(c);
    lowmachAddTurbulence(c, ic);
    const int steps = 40;
    const double dt = 0.004; // h = delta_th/4: inside the diffusive bound (0.011 at rho_b)
    const bool st = sameLm(lmMarch(c, ic, false, nullptr, steps, dt), lmMarch(c, ic, true, nullptr, steps, dt));
    std::printf("  structured: Turb instantiation at u' 0 %s\n", st ? "byte-identical" : "DIFFERS");
    check(st, "L1 structured: the Turb instantiation at u' 0 == off");
    const LmFv f = lmFv(c, true);
    std::vector<TurbMode> zero = turbModes(flameTurbSpec(c));
    for (auto& t : zero) t.uh[0] = t.uh[1] = t.uh[2] = 0.0;
    const TurbSweep sz = TurbSweep::make(zero, flameTurbSpec(c), xloCentres(f));
    for (const FvPrecision p : {FvPrecision::Fp32, FvPrecision::Fp64}) {
        const auto a = fvMarch(f, c, p, nullptr, false, steps, dt), b = fvMarch(f, c, p, &sz, false, steps, dt),
                   t = fvMarch(f, c, p, nullptr, true, steps, dt);
        const bool eq = sameCells(a, b) && sameCells(a, t);
        std::printf("  FV %s: u' 0 rows and the re-set tables %s\n", p == FvPrecision::Fp64 ? "fp64" : "fp32",
                    eq ? "byte-identical" : "DIFFER");
        check(eq, "L1 FV: u' 0 rows == off");
    }
}

void gateLmCross()
{
    std::printf("== L2 structured vs FV low-Mach, the turbulent planar flame, same dt (stated gap: MAC vs collocated)\n");
    double prev = 0;
    for (const int f : {1, 2}) {
        const std::string cpd = std::to_string(4 * f), n = std::to_string(24 * f), tl = "1";
        std::vector<std::string> a = kLmArgs;
        for (std::size_t i = 0; i + 1 < a.size(); ++i) {
            if (a[i] == "--cells-per-delta") a[i + 1] = cpd;
            if (a[i] == "--ny" || a[i] == "--nz") a[i + 1] = n;
        }
        const Case c = caseOf(a);
        const double dt = 0.004 / (f * f), tEnd = 0.4;
        const int steps = int(std::lround(tEnd / dt));
        double gap[2] = {0, 0}, scale = 0;
        for (const bool turb : {false, true}) {
            LowMachState ic = lowmachIC(c);
            if (turb) lowmachAddTurbulence(c, ic);
            const LowMachTurbSweeps sw = lowmachTurbSweeps(c);
            const LowMachState s = lmMarch(c, ic, turb, turb ? &sw : nullptr, steps, dt);
            const LmFv fv = lmFv(c, turb);
            const TurbSpec ts = flameTurbSpec(c);
            const TurbSweep fsw = TurbSweep::make(turbModes(ts), ts, xloCentres(fv));
            const auto o = fvMarch(fv, c, FvPrecision::Fp64, turb ? &fsw : nullptr, false, steps, dt);
            for (std::size_t q = 0; q < std::size_t(fv.fm.nC); ++q) {
                const std::size_t src = std::size_t(fv.fm.srcCell[q]);
                gap[turb] = std::max(gap[turb], std::abs(o.rho[q] - s.rho[src]));
                scale = std::max(scale, s.rho[src]);
            }
        }
        std::printf("  h = delta_th/%d, t %.2g (%d steps): max |rho_FV - rho_MAC| turbulent %.3e%s, laminar %.3e (rho max "
                    "%.3f)\n",
                    4 * f, tEnd, steps, gap[1],
                    prev > 0 ? std::format(" (order {:.2f})", std::log2(prev / gap[1])).c_str() : "", gap[0], scale);
        if (prev > 0) check(gap[1] < 0.6 * prev, "L2 the turbulent FV - MAC gap shrinks with h (a scheme gap)");
        prev = gap[1];
    }
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "lowmach") {
        gateLmZero();
        gateLmCross();
        return dnstest::finish();
    }
    gateIC();
    gateRows();
    gateStructured();
    return dnstest::finish();
}
