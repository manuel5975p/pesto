// Species and one-step chemistry on the compressible FV family (docs/fvcomp.md
// "Species"), against the FP64 oracle and the structured species kernels.
//   S1 double instantiation vs FvRef: every species pass, rhs, 3 steps, chemDt
//      <= 1e-13 on box_cyclic, box_refined, cyl2d_snappy
//   S2 float vs double: rhs and a 3-step state (O2's bounds)
//   S3 uniform power-of-two Y exact to the bit through every boundary kind, 40
//      steps, FP32, FP32 compensated, FP64
//   S4 a passive species leaves the mean flow bit for bit the nY == 0 solver
//   S5 conservation: mass, passive species and E + q rho Y over a burning march
//      (periodic, walled); Sum V R_Y = -Sum V omega
//   S6 rerun byte-identical, both precisions
//   S7 2D: species z flux and grad_z Y exactly 0
//   S8 FV vs the structured species kernels on a uniform periodic box, same dt:
//      rounding, <= 3 x the FP32 run's own distance from FP64
//   D1 two reactants, per-face rows: double vs FvRef <= 1e-13, float vs double
//   D2 S3, S4 again on the two-reactant kernels
//   D3 conservation: mass, E + q rho Y_F, coupling rho (Y_F - Y_O/s), tracer
//   flame1d (`test_gpu_fv_flame flame1d`): the binary route, 7.5 cells per delta_th
//   mix (`test_gpu_fv_flame mix`): gpu_diffusion_mix's layer, uniform and wavy mesh
//   jet (`test_gpu_fv_flame jet`): the binary route of --mode diffusion
//   reactor (`test_gpu_fv_flame reactor`): the 0D constant-volume reactor, G0.1-G0.4 of
//      masterplan_combnozzle P0, against fvref_reactor.hpp's closed quadrature
#include "fv_comp_test_util.hpp"
#include "fvref_reactor.hpp"

#include "dns/chem.hpp"
#include "dns/diffusion.hpp"
#include "dns/gpu/chem.hpp"
#include "dns/gpu/comp.hpp"
#include "dns/gpu/hip_runtime.hpp"
#include "gpu_test_util.hpp"

#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <unistd.h>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dnstest::fvc;
using dnstest::check;

namespace {

struct Case
{
    const char* name;
    FaceMesh fm;
    std::vector<FvBc> bc;
    Transport tr;
};

// A hot, non-stiff step: omega ~ 0.6 rho Y at T ~ 1.
constexpr FvChem kChem{.nY = 2, .fuel = 0, .lnA = 1.6, .Ta = 2.0, .q = 0.3, .invPr = 1.0 / 0.72};
constexpr double kNoRate = -std::numeric_limits<double>::infinity();

void withY(FvBc& b, double y0, double y1)
{
    b.y[0] = y0;
    b.y[1] = y1;
}

std::vector<Case> cases(double y0 = 0.7, double y1 = 0.3)
{
    std::vector<Case> v;
    {
        FaceMesh fm = loadMesh("box_cyclic");
        auto pick = [](const MeshPatch&) { return FvBc{}; };
        v.push_back({"box_cyclic (adiabatic walls), mu 0.05", std::move(fm), {}, Transport::constant(kGamma, 0.05)});
        v.back().bc = bcFor(v.back().fm, pick);
    }
    v.push_back({"box_refined (cyclic, split faces), Sutherland", loadMesh("box_refined"), {},
                 Transport::air(0.02, 0.9, 288.15, 0.72, kGamma, 1.0, true)});
    {
        FaceMesh fm = loadMesh("cyl2d_snappy");
        auto pick = [&](const MeshPatch& p) {
            FvBc b;
            if (p.name == "cylinder") {
                b.kind = FvBcKind::Wall;
            } else if (p.name == "inlet") {
                b.kind = FvBcKind::Inflow;
                b.uIn[0] = 0.5;
                b.tIn = 1.0;
            } else if (p.name == "outlet") {
                b.kind = FvBcKind::Outflow;
                b.pOut = 1.0;
                b.tBack = 1.0;
            } else if (p.name == "lower") {
                b.kind = FvBcKind::Slip;
            } else {
                b.kind = FvBcKind::Far;
                b.qFar[0] = 1.0;
                b.qFar[1] = 0.5;
                b.qFar[4] = 1.0 / (kGamma - 1.0) + 0.125;
            }
            withY(b, y0, y1);
            return b;
        };
        v.push_back({"cyl2d_snappy (wall, inflow, outflow, slip, far), mu 0.01", std::move(fm), {},
                     Transport::constant(kGamma, 0.01)});
        v.back().bc = bcFor(v.back().fm, pick);
    }
    return v;
}

// smoothState + a tanh front in Y0 (the limiter's clipped branches) and a smooth Y1.
FvState speciesState(const FaceMesh& fm, bool uz)
{
    const FvState base = smoothState(fm, uz, 0.6);
    FvState s = FvState::zeros(fm.nC, kFvMeanVars + 2);
    std::copy(base.q.begin(), base.q.end(), s.q.begin());
    const auto [lo, hi] = std::ranges::minmax(fm.C[0]);
    const auto [ly, hy] = std::ranges::minmax(fm.C[1]);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double X = (fm.C[0][c] - lo) / (hi - lo), Y = (fm.C[1][c] - ly) / (hy - ly);
        const double y0 = 0.5 * (1.0 + std::tanh((X - 0.5 - 0.1 * std::sin(6 * Y)) / 0.05));
        const double y1 = 0.3 + 0.2 * std::cos(5 * X + 2 * Y);
        s.var(5)[c] = s.var(0)[c] * y0;
        s.var(6)[c] = s.var(0)[c] * y1;
    }
    return s;
}

double vecsDiff(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b)
{
    double worst = 0.0;
    for (std::size_t k = 0; k < b.size(); ++k) {
        double e = 0.0, s = 0.0;
        for (std::size_t c = 0; c < b[k].size(); ++c) {
            e = std::max(e, std::abs(a[k][c] - b[k][c]));
            s = std::max(s, std::abs(b[k][c]));
        }
        worst = std::max(worst, s == 0.0 ? e : e / s);
    }
    return worst;
}

// Model C1, F + 2 O, the third row Z: a hot, non-stiff step as kChem's.
constexpr FvChem kChem2{.nY = 3, .fuel = 0, .lnA = 1.6, .Ta = 2.0, .q = 0.3, .invPr = 1.0 / 0.72, .ox = 1, .stoich = 2.0};
constexpr double kOx2 = 0.8; // the oxidizer stream's Y_O

double mix2(double yf, double yo) { return (2.0 * yf - yo + kOx2) / (2.0 + kOx2); }

FvState reactantState(const FaceMesh& fm, bool uz)
{
    const FvState base = smoothState(fm, uz, 0.6);
    FvState s = FvState::zeros(fm.nC, kFvMeanVars + 3);
    std::copy(base.q.begin(), base.q.end(), s.q.begin());
    const auto [lo, hi] = std::ranges::minmax(fm.C[0]);
    const auto [ly, hy] = std::ranges::minmax(fm.C[1]);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double X = (fm.C[0][c] - lo) / (hi - lo), Y = (fm.C[1][c] - ly) / (hy - ly);
        const double a = 0.5 * (1.0 + std::tanh((X - 0.5 - 0.1 * std::sin(6 * Y)) / 0.08));
        const double yf = 0.2 + 0.6 * a, yo = kOx2 * (0.8 - 0.6 * a);
        s.var(5)[c] = s.var(0)[c] * yf;
        s.var(6)[c] = s.var(0)[c] * yo;
        s.var(7)[c] = s.var(0)[c] * mix2(yf, yo);
    }
    return s;
}

// The table of m with one row per face on every open patch: a Y and inflow profile.
struct Rows
{
    std::vector<FvBc> bc;
    std::vector<int> row;
};

Rows faceRows(const Case& m)
{
    Rows o{.bc = m.bc, .row = std::vector<int>(std::size_t(m.fm.nB), 0)};
    for (std::size_t p = 0; p < m.fm.patches.size(); ++p) {
        const MeshPatch& mp = m.fm.patches[p];
        std::fill_n(o.row.begin() + mp.first, mp.count, int(p));
        const FvBcKind k = m.bc[p].kind;
        if (k != FvBcKind::Inflow && k != FvBcKind::Far && k != FvBcKind::Outflow) continue;
        for (int i = 0; i < mp.count; ++i) {
            FvBc b = m.bc[p];
            const double w = std::sin(0.7 * i);
            b.y[0] = 0.5 + 0.3 * w;
            b.y[1] = 0.4 - 0.2 * w;
            b.y[2] = mix2(b.y[0], b.y[1]);
            b.ownY = k == FvBcKind::Outflow && i % 2 == 0;
            b.characteristic = k == FvBcKind::Outflow && i % 3 == 0;
            if (k == FvBcKind::Inflow) {
                b.uIn[0] = 0.5 + 0.05 * w;
                b.tIn = 1.0 + 0.05 * w;
                b.diffusive = i % 2 == 0;
            }
            o.row[std::size_t(mp.first + i)] = int(o.bc.size());
            o.bc.push_back(b);
        }
    }
    return o;
}

void gateTwoReactants()
{
    std::printf("== D1 two reactants, per-face rows: double vs FvRef, float vs double\n");
    for (Case& m : cases()) {
        const Rows r = faceRows(m);
        const FvChem chem = kChem2;
        const FvRef ref = orDie(FvRef::make(m.fm, {.tr = m.tr, .bc = r.bc, .bcRow = r.row, .chem = chem}), m.name);
        DeviceFvComp dev = orDie(
            DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = r.bc, .bcRow = r.row, .precision = FvPrecision::Fp64, .chem = chem}), m.name);
        DeviceFvComp flt = orDie(
            DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = r.bc, .bcRow = r.row, .precision = FvPrecision::Fp32, .chem = chem}), m.name);
        FvState s = reactantState(m.fm, m.fm.dim == 3);
        orDie(dev.upload(s), "upload");
        orDie(flt.upload(s), "upload");
        FvEval eR, eD;
        FvState rR, rD, rF;
        ref.evaluate(s, eR, rR);
        orDie(dev.evaluate(eD, rD), "evaluate");
        orDie(flt.rhs(rF), "rhs");
        const int nV = ref.nV(), nC = m.fm.nC;
        const double dR = relDiff(rD.q, rR.q, nV, nC), dG = vecsDiff(eD.gradY, eR.gradY), dO = vecsDiff({eD.omega}, {eR.omega}),
                     dB = vecsDiff(eD.bY, eR.bY), dFy = vecsDiff(eD.Fy, eR.Fy);
        const double dChemR = ref.chemDt(s, 0.5), dChemD = orDie(dev.chemDt(0.5), "chemDt");
        const double dt = 0.25 * orDie(ref.stableDt(s), "dt");
        FvState q = s, qD, qF;
        ref.march(q, dt, 3);
        for (int n = 0; n < 3; ++n) {
            dev.step(dt);
            flt.step(dt);
        }
        orDie(dev.download(qD), "download");
        orDie(flt.download(qF), "download");
        const double dS = relDiff(qD.q, q.q, nV, nC), fR = relDiff(rF.q, rD.q, nV, nC), fS = relDiff(qF.q, qD.q, nV, nC);
        std::printf("  %s\n", m.name);
        check(dR <= 1e-13, std::format("rhs {:.2e} <= 1e-13", dR));
        check(dO <= 1e-13, std::format("omega {:.2e} <= 1e-13", dO));
        check(dG <= 1e-13, std::format("grad Y {:.2e} <= 1e-13", dG));
        check(dB <= 1e-13, std::format("boundary Y {:.2e} <= 1e-13", dB));
        check(dFy <= 1e-13, std::format("species fluxes {:.2e} <= 1e-13", dFy));
        check(dS <= 1e-13, std::format("3 steps {:.2e} <= 1e-13", dS));
        check(std::abs(dChemD / dChemR - 1.0) <= 1e-13, std::format("chem dt {:.2e} <= 1e-13", std::abs(dChemD / dChemR - 1.0)));
        check(fR <= 2e-5, std::format("float rhs {:.2e} <= 2e-5", fR));
        check(fS <= 2e-6, std::format("float 3 steps {:.2e} <= 2e-6", fS));
    }
}

void gateConservation2()
{
    std::printf("== D3 two-reactant conservation under burning, FP64\n");
    const Grid g = Grid::make(32, 24, 1, 1.0, 1.0, 1.0);
    FaceMesh per = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "box");
    std::vector<Case> v;
    v.push_back({"uniform periodic", std::move(per), {}, Transport::constant(kGamma, 0.02)});
    v.back().bc = bcFor(v.back().fm, [](const MeshPatch&) { return FvBc{}; });
    v.push_back({"box_cyclic (adiabatic walls)", loadMesh("box_cyclic"), {}, Transport::constant(kGamma, 0.05)});
    v.back().bc = bcFor(v.back().fm, [](const MeshPatch&) { return FvBc{}; });
    v.push_back({"box_refined (cyclic, split faces)", loadMesh("box_refined"), {},
                 Transport::air(0.02, 0.9, 288.15, 0.72, kGamma, 1.0, true)});
    for (Case& m : v) {
        const FvChem chem = kChem2;
        DeviceFvComp d =
            orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64, .chem = chem}), m.name);
        const FvRef ref = orDie(FvRef::make(m.fm, {.tr = m.tr, .bc = m.bc, .chem = chem}), m.name);
        FvState s = reactantState(m.fm, m.fm.dim == 3);
        orDie(d.upload(s), "upload");
        struct Sums
        {
            double mass, coupled, element, tracer, fuel, gross;
        };
        auto sums = [&](const FvState& q) {
            Sums o{0, 0, 0, 0, 0, 0};
            for (std::size_t c = 0; c < std::size_t(q.nC); ++c) {
                const double V = m.fm.V[c];
                o.mass += V * q.var(0)[c];
                o.coupled += V * (q.var(4)[c] + chem.q * q.var(5)[c]);
                o.element += V * (q.var(5)[c] - q.var(6)[c] / chem.stoich);
                o.gross += V * (q.var(5)[c] + q.var(6)[c] / chem.stoich);
                o.tracer += V * q.var(7)[c];
                o.fuel += V * q.var(5)[c];
            }
            return o;
        };
        FvState R;
        FvEval ev;
        orDie(d.evaluate(ev, R), "evaluate");
        double dFuel = 0.0, dOx = 0.0, burn = 0.0;
        for (std::size_t c = 0; c < std::size_t(s.nC); ++c) {
            dFuel += m.fm.V[c] * R.var(5)[c];
            dOx += m.fm.V[c] * R.var(6)[c];
            burn += m.fm.V[c] * ev.omega[c];
        }
        const Sums a = sums(s);
        const double dt = 0.25 * orDie(ref.stableDt(s), "dt");
        for (int n = 0; n < 200; ++n) d.step(dt);
        FvState o;
        orDie(d.download(o), "download");
        const Sums b = sums(o);
        const double eM = std::abs(b.mass / a.mass - 1.0), eE = std::abs(b.coupled / a.coupled - 1.0),
                     eC = std::abs(b.element - a.element) / a.gross, eT = std::abs(b.tracer / a.tracer - 1.0),
                     eB = std::abs(dFuel + burn) / burn, eO = std::abs(dOx + chem.stoich * burn) / burn;
        std::printf("  %s: fuel burnt %.3f%% over 200 steps\n", m.name, 100.0 * (1.0 - b.fuel / a.fuel));
        check(eM <= 1e-13, std::format("mass {:.2e} <= 1e-13", eM));
        check(eE <= 1e-13, std::format("E + q rho Y_F {:.2e} <= 1e-13", eE));
        check(eC <= 1e-13, std::format("rho (Y_F - Y_O/s), of gross {:.2e} <= 1e-13", eC));
        check(eT <= 1e-13, std::format("tracer {:.2e} <= 1e-13", eT));
        check(eB <= 1e-13, std::format("Sum V R_F + Sum V omega {:.2e} <= 1e-13", eB));
        check(eO <= 1e-13, std::format("Sum V R_O + s Sum V omega {:.2e} <= 1e-13", eO));
        check(b.fuel < a.fuel, "fuel falls");
    }
}

// gpu_diffusion_mix's layer on FaceMesh: a fuel slab normal to (1, 1)/sqrt 2 on a
// periodic 256^2 box at h 0.25, advected at u = (1, 0), isothermal, s 3, Da 50, IC
// the Burke-Schumann state at diffusion age 2, run t = 20.
struct MixOut
{
    double dz = 0, zMin = HUGE_VAL, zMax = -HUGE_VAL, yMin = HUGE_VAL, zErr = 0;
    long steps = 0;
};

double slabZ(double s, double t, double x0, double a, double b, double P, double D)
{
    const double d = 2 * std::sqrt(D * t);
    double z = 0;
    for (int k = -3; k <= 3; ++k)
        z += 0.5 * (std::erf((s - a - x0 + k * P) / d) - std::erf((s - b - x0 + k * P) / d));
    return z;
}

MixOut runMix(const FaceMesh& fm, const DiffusionModel& m, bool reacting, int ox, double D)
{
    const double L = 64, gam = kGamma, Pr = 0.72, M = 0.1, T0 = 1 / (gam * M * M), U = 1.0, run = 20.0, age = 2.0;
    const Transport tr = Transport::air(Pr * D, T0, 288.15, Pr, gam, 1.0, false);
    const FvChem chem{.nY = 3, .fuel = 0, .lnA = reacting ? std::log(m.Da) : kNoRate, .Ta = 0, .q = 0, .invPr = 1 / Pr, .ox = ox,
                      .stoich = m.s};
    const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return FvBc{}; });
    DeviceFvComp d = orDie(DeviceFvComp::make(fm, {.tr = tr, .bc = bc, .precision = FvPrecision::Fp32, .chem = chem}), "mix");
    const double P = L / std::numbers::sqrt2, a = 0.25 * P, b = 0.75 * P, cs = U / std::numbers::sqrt2;
    FvState ic = FvState::zeros(fm.nC, 8);
    const std::size_t nC = std::size_t(fm.nC);
    for (std::size_t c = 0; c < nC; ++c) {
        const double s = (fm.C[0][c] + fm.C[1][c]) / std::numbers::sqrt2;
        const SheetState st = burkeSchumann(m, slabZ(s, age, 0.0, a, b, P, D));
        ic.var(0)[c] = 1;
        ic.var(1)[c] = U;
        ic.var(4)[c] = T0 / (gam - 1) + 0.5 * U * U;
        ic.var(5)[c] = st.YF;
        ic.var(6)[c] = st.YO;
        ic.var(7)[c] = mixtureFraction(m, st.YF, st.YO);
    }
    orDie(d.upload(ic), "upload");
    MixOut o;
    double t = 0, dt = 0;
    while (t < run) {
        if (o.steps % 16 == 0) dt = 0.95 * std::min(orDie(d.stableDt({.cfl = 0.4}), "dt"), orDie(d.chemDt(gpu::kChemDt), "chem"));
        const double h = std::min(dt, run - t);
        d.step(h);
        t += h;
        ++o.steps;
    }
    FvState q;
    orDie(d.download(q), "download");
    for (std::size_t c = 0; c < nC; ++c) {
        const double r = q.var(0)[c], yf = q.var(5)[c] / r, yo = q.var(6)[c] / r, tz = q.var(7)[c] / r;
        const double z = mixtureFraction(m, yf, yo);
        o.dz = std::max(o.dz, std::abs(z - tz));
        o.zMin = std::min({o.zMin, z, tz});
        o.zMax = std::max({o.zMax, z, tz});
        o.yMin = std::min({o.yMin, yf, yo});
        const double s = (fm.C[0][c] + fm.C[1][c]) / std::numbers::sqrt2;
        o.zErr = std::max(o.zErr, std::abs(tz - slabZ(std::fmod(s, P), age + run, cs * run, a, b, P, D)));
    }
    return o;
}

void gateMix()
{
    DiffusionModel m;
    m.s = 3;
    m.yF1 = 1;
    m.yO2 = 1;
    m.sigma = 1;
    m.Da = 50;
    const Grid g = Grid::make(256, 256, 1, 64, 64, 1.0);
    const auto walk = [](long n) { return 4 * std::ldexp(1.0, -23) * std::sqrt(3.0 * double(n)); };
    for (const double wavy : {0.0, 0.02}) {
        // Scaled like CurvMesh::sinusoidal: skewed cells, the layer crosses every mesh line.
        const CurvMesh cm = wavy > 0 ? CurvMesh::sinusoidal(g, wavy) : CurvMesh::uniform(g);
        const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(cm, periodicSpec(g), CellOrder::Morton), "mix mesh");
        const MixOut r = runMix(fm, m, true, 1, 1.0), p2 = runMix(fm, m, false, 1, 0.02), p1 = runMix(fm, m, false, -1, 0.02);
        const std::string tag = wavy > 0 ? "wavy" : "uniform";
        std::printf("  %s: %ld steps, max|Z(Y) - tracer| reacting %.3e (walk %.3e), passive %.3e (walk %.3e), per-species %.3e\n",
                    tag.c_str(), r.steps, r.dz, walk(r.steps), p2.dz, walk(p2.steps), p1.dz);
        std::printf("  %s: Z in [%.3e, 1%+.3e], Y min %.3e, max|Z - erf| %.3e\n", tag.c_str(), r.zMin, r.zMax - 1, r.yMin, r.zErr);
        check(r.dz < walk(r.steps) && p2.dz < walk(p2.steps), tag + ": Z(Y) equals the tracer within the FP32 walk");
        check(p1.dz > 30 * p2.dz, tag + ": contrast: per-species limiting breaks it by > 30x");
        check(r.zMin > -1e-6 && r.zMax < 1 + 1e-6 && r.yMin > -1e-6, tag + ": Z, Y in [-1e-6, 1 + 1e-6]");
        check(r.zErr < 3e-3, tag + ": the tracer within 3e-3 of the analytic erf");
    }
}

void gateOracle()
{
    std::printf("== S1 double vs FvRef, S2 float vs double\n");
    for (Case& m : cases()) {
        const FvChem chem = kChem;
        const FvRef ref = orDie(FvRef::make(m.fm, {.tr = m.tr, .bc = m.bc, .chem = chem}), m.name);
        DeviceFvComp dev =
            orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64, .chem = chem}), m.name);
        DeviceFvComp flt =
            orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp32, .chem = chem}), m.name);
        FvState s = speciesState(m.fm, m.fm.dim == 3);
        orDie(dev.upload(s), "upload");
        orDie(flt.upload(s), "upload");
        FvEval eR, eD;
        FvState rR, rD, rF;
        ref.evaluate(s, eR, rR);
        orDie(dev.evaluate(eD, rD), "evaluate");
        orDie(flt.rhs(rF), "rhs");
        const int nV = ref.nV(), nC = m.fm.nC;
        const double dR = relDiff(rD.q, rR.q, nV, nC), dY = vecsDiff(eD.Y, eR.Y), dG = vecsDiff(eD.gradY, eR.gradY),
                     dO = vecsDiff({eD.omega}, {eR.omega}), dB = vecsDiff(eD.bY, eR.bY), dFy = vecsDiff(eD.Fy, eR.Fy);
        const double dF = vecsDiff({eD.F[0], eD.F[1], eD.F[2], eD.F[3], eD.F[4]},
                                   {eR.F[0], eR.F[1], eR.F[2], eR.F[3], eR.F[4]});
        const double dChemR = ref.chemDt(s, 0.5), dChemD = orDie(dev.chemDt(0.5), "chemDt");
        const double dt = 0.25 * orDie(ref.stableDt(s), "dt");
        FvState q = s, qD, qF;
        ref.march(q, dt, 3);
        for (int n = 0; n < 3; ++n) {
            dev.step(dt);
            flt.step(dt);
        }
        orDie(dev.download(qD), "download");
        orDie(flt.download(qF), "download");
        const double dS = relDiff(qD.q, q.q, nV, nC);
        std::printf("  %s\n", m.name);
        check(dR <= 1e-13, std::format("rhs {:.2e} <= 1e-13", dR));
        check(dY <= 1e-13, std::format("Y {:.2e} <= 1e-13", dY));
        check(dO <= 1e-13, std::format("omega {:.2e} <= 1e-13", dO));
        check(dG <= 1e-13, std::format("grad Y {:.2e} <= 1e-13", dG));
        check(dB <= 1e-13, std::format("boundary Y {:.2e} <= 1e-13", dB));
        check(dF <= 1e-13, std::format("mean fluxes {:.2e} <= 1e-13", dF));
        check(dFy <= 1e-13, std::format("species fluxes {:.2e} <= 1e-13", dFy));
        check(dS <= 1e-13, std::format("3 steps {:.2e} <= 1e-13", dS));
        check(std::abs(dChemD / dChemR - 1.0) <= 1e-13,
              std::format("chem dt {:.2e} <= 1e-13", std::abs(dChemD / dChemR - 1.0)));
        const double fR = relDiff(rF.q, rD.q, nV, nC), fS = relDiff(qF.q, qD.q, nV, nC);
        check(fR <= 2e-5, std::format("float rhs {:.2e} <= 2e-5", fR));
        check(fS <= 2e-6, std::format("float 3 steps {:.2e} <= 2e-6", fS));
    }
}

template <class R>
bool exactRatio(const FvState& s, int row, double y)
{
    for (std::size_t c = 0; c < std::size_t(s.nC); ++c) {
        const R rho = R(s.var(0)[c]), ry = R(s.var(row)[c]);
        if (ry != R(y) * rho) return false;
    }
    return true;
}

void gatePassive()
{
    std::printf("== S3 uniform power-of-two Y, S4 passive species vs nY == 0\n");
    const double y0 = 0.25, y1 = 0.5;
    for (const int ox : {-1, 1})
    for (Case& m : cases(y0, y1)) {
        FvChem chem = kChem;
        chem.lnA = kNoRate;
        chem.q = 0.0;
        chem.ox = ox; // ox 1: the two-reactant kernels, D2
        FvState base = smoothState(m.fm, m.fm.dim == 3, 0.3);
        FvState s = FvState::zeros(m.fm.nC, kFvMeanVars + 2);
        std::copy(base.q.begin(), base.q.end(), s.q.begin());
        for (std::size_t c = 0; c < std::size_t(m.fm.nC); ++c) {
            s.var(5)[c] = y0 * s.var(0)[c];
            s.var(6)[c] = y1 * s.var(0)[c];
        }
        const double dt = 0.25 * orDie(orDie(FvRef::make(m.fm, {.tr = m.tr, .bc = m.bc}), m.name).stableDt(base), "dt");
        struct Mode
        {
            const char* tag;
            FvPrecision pr;
            bool comp;
        };
        for (const Mode md : {Mode{"FP32", FvPrecision::Fp32, false}, Mode{"FP32 compensated", FvPrecision::Fp32, true},
                              Mode{"FP64", FvPrecision::Fp64, false}}) {
            DeviceFvComp d = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = md.pr, .compensated = md.comp,
                                                             .chem = chem}), m.name);
            DeviceFvComp d0 = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = md.pr, .compensated = md.comp}),
                                    m.name);
            orDie(d.upload(s), "upload");
            orDie(d0.upload(base), "upload");
            const auto lts = d.localDt({.cfl = 0.4});
            check(!lts && lts.error().find("with species") != std::string::npos, "local time stepping refused with species");
            for (int n = 0; n < 40; ++n) {
                d.step(dt);
                d0.step(dt);
            }
            FvState o, o0;
            orDie(d.download(o), "download");
            orDie(d0.download(o0), "download");
            const bool fp32 = md.pr == FvPrecision::Fp32;
            bool ok;
            if (md.comp) { // hi + lo in FP64 is exact and scales with rho's.
                ok = true;
                for (std::size_t c = 0; c < std::size_t(m.fm.nC); ++c)
                    ok = ok && o.var(5)[c] == y0 * o.var(0)[c] && o.var(6)[c] == y1 * o.var(0)[c];
            } else {
                ok = fp32 ? exactRatio<float>(o, 5, y0) && exactRatio<float>(o, 6, y1)
                          : exactRatio<double>(o, 5, y0) && exactRatio<double>(o, 6, y1);
            }
            check(ok, std::format("S3 {}, {}, ox {}: rho Y == Y rho bitwise, 40 steps", m.name, md.tag, ox));
            const bool same = std::memcmp(o.q.data(), o0.q.data(), o0.q.size() * sizeof(double)) == 0;
            check(same, std::format("S4 {}, {}, ox {}: mean rows == nY 0 bitwise", m.name, md.tag, ox));
        }
    }
}

void gateConservation()
{
    std::printf("== S5 conservation under burning, FP64\n");
    const Grid g = Grid::make(32, 24, 1, 1.0, 1.0, 1.0);
    FaceMesh per = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "box");
    std::vector<Case> v;
    v.push_back({"uniform periodic", std::move(per), {}, Transport::constant(kGamma, 0.02)});
    v.back().bc = bcFor(v.back().fm, [](const MeshPatch&) { return FvBc{}; });
    {
        FaceMesh fm = loadMesh("box_cyclic");
        v.push_back({"box_cyclic (adiabatic walls)", std::move(fm), {}, Transport::constant(kGamma, 0.05)});
        v.back().bc = bcFor(v.back().fm, [](const MeshPatch&) { return FvBc{}; });
    }
    v.push_back({"box_refined (cyclic, split faces)", loadMesh("box_refined"), {},
                 Transport::air(0.02, 0.9, 288.15, 0.72, kGamma, 1.0, true)});
    for (Case& m : v) {
        const FvChem chem = kChem;
        DeviceFvComp d =
            orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64, .chem = chem}), m.name);
        const FvRef ref = orDie(FvRef::make(m.fm, {.tr = m.tr, .bc = m.bc, .chem = chem}), m.name);
        FvState s = speciesState(m.fm, m.fm.dim == 3);
        orDie(d.upload(s), "upload");
        struct Sums
        {
            double mass, coupled, passive, fuel;
        };
        auto sums = [&](const FvState& q) {
            Sums o{0, 0, 0, 0};
            for (std::size_t c = 0; c < std::size_t(q.nC); ++c) {
                const double V = m.fm.V[c];
                o.mass += V * q.var(0)[c];
                o.coupled += V * (q.var(4)[c] + chem.q * q.var(5)[c]);
                o.passive += V * q.var(6)[c];
                o.fuel += V * q.var(5)[c];
            }
            return o;
        };
        FvState R;
        FvEval ev;
        orDie(d.evaluate(ev, R), "evaluate");
        double dFuel = 0.0, burn = 0.0;
        for (std::size_t c = 0; c < std::size_t(s.nC); ++c) {
            dFuel += m.fm.V[c] * R.var(5)[c];
            burn += m.fm.V[c] * ev.omega[c];
        }
        const Sums a = sums(s);
        const double dt = 0.25 * orDie(ref.stableDt(s), "dt");
        for (int n = 0; n < 200; ++n) d.step(dt);
        FvState o;
        orDie(d.download(o), "download");
        const Sums b = sums(o);
        const double eM = std::abs(b.mass / a.mass - 1.0), eE = std::abs(b.coupled / a.coupled - 1.0),
                     eP = std::abs(b.passive / a.passive - 1.0), eB = std::abs(dFuel + burn) / burn;
        std::printf("  %s: fuel burnt %.3f%% over 200 steps\n", m.name, 100.0 * (1.0 - b.fuel / a.fuel));
        check(eM <= 1e-13, std::format("mass {:.2e} <= 1e-13", eM));
        check(eE <= 1e-13, std::format("E + q rho Y {:.2e} <= 1e-13", eE));
        check(eP <= 1e-13, std::format("passive species {:.2e} <= 1e-13", eP));
        check(eB <= 1e-13, std::format("Sum V R_Y + Sum V omega {:.2e} <= 1e-13", eB));
        check(b.fuel < a.fuel, "fuel falls");
    }
}

void gateRerun()
{
    std::printf("== S6 rerun, S7 2D\n");
    for (Case& m : cases()) {
        for (const FvPrecision pr : {FvPrecision::Fp32, FvPrecision::Fp64}) {
            FvState a, b;
            for (FvState* o : {&a, &b}) {
                DeviceFvComp d = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = pr, .chem = kChem}), m.name);
                FvState s = speciesState(m.fm, m.fm.dim == 3);
                orDie(d.upload(s), "upload");
                const double dt = 0.25 * orDie(d.stableDt(), "dt");
                for (int n = 0; n < 40; ++n) d.step(dt);
                orDie(d.download(*o), "download");
            }
            check(a.q.size() == b.q.size() && std::memcmp(a.q.data(), b.q.data(), a.q.size() * sizeof(double)) == 0,
                  std::format("{}, {}: reruns byte-identical", m.name, fvPrecisionName(pr)));
        }
        if (m.fm.dim != 2) continue;
        DeviceFvComp d = orDie(DeviceFvComp::make(m.fm, {.tr = m.tr, .bc = m.bc, .precision = FvPrecision::Fp64, .chem = kChem}),
                               m.name);
        FvState s = speciesState(m.fm, false), r;
        orDie(d.upload(s), "upload");
        FvEval ev;
        orDie(d.evaluate(ev, r), "evaluate");
        const std::size_t e = std::size_t(m.fm.emptyAxis);
        double gz = 0.0;
        for (std::size_t k = 0; k < 2; ++k) gz = std::max(gz, maxAbs(ev.gradY[3 * k + e]));
        check(gz == 0.0, std::format("{}: grad Y along the empty axis exactly 0", m.name));
    }
}

// Uniform periodic box, a hot wrinkled slab burning outward from rest, the real
// model's constants (M 0.05: the FP32 noise floor is the point, not the flame).
void gateStructured()
{
    std::printf("== S8 FV vs the structured species kernels, uniform periodic, same dt\n");
    const double M = 0.05, sigma = 5.0;
    PremixedModel md;
    md.sigma = sigma;
    const PremixedEigen eig = orDie(premixedEigen(md), "oracle");
    const double Tu = 1.0 / (kGamma * M * M), h = eig.dthOverDD / 12.0;
    const Grid g = Grid::make(128, 32, 1, 128 * h, 32 * h, 1.0);
    const Transport tr = Transport::air(md.Pr, Tu, 288.15, md.Pr, kGamma, 1.0, false);
    const double q = tr.cp() * (sigma - 1.0) * Tu;
    auto ic = CompState::zeros(g);
    std::vector<double> ry(g.size());
    const double Lx = g.nx * g.hx, Ly = g.ny * g.hy, w = 1.5 * eig.dthOverDD;
    for (int j = 0; j < g.ny; ++j)
        for (int i = 0; i < g.nx; ++i) {
            const std::size_t c = g.idx(i, j, 0);
            const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
            const double f = 0.3 * Lx + 2 * h * std::cos(2 * std::numbers::pi * y / Ly);
            const double t = 1.0 + 0.5 * (sigma - 1.0) * (std::tanh((x - f) / w) - std::tanh((x - 0.7 * Lx) / w));
            const double rho = 1.0 / t;
            ic.rho[c] = rho;
            ic.E[c] = Tu / (kGamma - 1.0);
            ry[c] = rho * (sigma - t) / (sigma - 1.0);
        }
    constexpr int steps = 400;
    // The structured run.
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
    // Both FV precisions on the same mesh.
    FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), CellOrder::Morton), "box");
    const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return FvBc{}; });
    const FvChem chem{.nY = 1, .fuel = 0, .lnA = ss.lnA, .Ta = ss.Ta, .q = q, .invPr = ss.invPr};
    const std::vector<double>* src[6] = {&ic.rho, &ic.mx, &ic.my, &ic.mz, &ic.E, &ry};
    FvState s = FvState::zeros(fm.nC, 6), st = FvState::zeros(fm.nC, 6);
    const std::vector<double>* hv[6] = {&ho.rho, &ho.mx, &ho.my, &ho.mz, &ho.E, &yo};
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
    double burnt = 0.0, b0 = 0.0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        burnt += o64.var(5)[c];
        b0 += s.var(5)[c];
    }
    std::printf("  %d x %d, %d steps, fuel burnt %.3f%%\n", g.nx, g.ny, steps, 100.0 * (1.0 - burnt / b0));
    const double a = relDiff(o32.q, st.q, 6, fm.nC), own = relDiff(o32.q, o64.q, 6, fm.nC),
                 st64 = relDiff(st.q, o64.q, 6, fm.nC);
    std::printf("  FV FP32 vs structured %.3e, structured vs FV FP64 %.3e, FV FP32 vs FP64 %.3e\n", a, st64, own);
    check(a <= 3.0 * own, std::format("FV FP32 vs structured {:.2e} <= 3 x own {:.2e}", a, own));
    check(st64 <= 3.0 * own, std::format("structured vs FV FP64 {:.2e} <= 3 x own {:.2e}", st64, own));
}

// --mode diffusion through the binary, FP64. Bands: gpu_diffusion_jet's that apply.
int runJet()
{
#ifndef DNS_BIN
    dnstest::skip("no dns binary");
#else
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / std::format("dns_fvjet_{}", ::getpid());
    fs::create_directories(dir);
    const std::string out = (dir / "j").string();
    const std::string cmd = std::string(DNS_BIN) +
                            " flame --mode diffusion --solver fv --fv-precision fp64 --tt 80 "
                            "--emit stats --quiet --out " + out + " > /dev/null";
    if (std::system(cmd.c_str()) != 0) {
        check(false, "dns flame --mode diffusion --solver fv ran");
        return dnstest::finish();
    }
    std::ifstream in(out + "_stats.csv");
    std::string line;
    std::vector<std::vector<double>> rows;
    std::vector<std::string> head;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream s(line);
        std::string f;
        std::vector<std::string> cols;
        while (std::getline(s, f, ',')) cols.push_back(f);
        if (head.empty()) {
            head = cols;
            continue;
        }
        std::vector<double> v;
        for (const std::string& c : cols) v.push_back(std::stod(c));
        rows.push_back(std::move(v));
    }
    fs::remove_all(dir);
    auto col = [&](const char* n) { return std::size_t(std::ranges::find(head, n) - head.begin()); };
    auto mean = [&](const char* n) {
        const std::size_t k = col(n);
        double s = 0;
        int m = 0;
        for (const auto& r : rows)
            if (r[col("t")] >= 60.0) {
                s += r[k];
                ++m;
            }
        return s / double(m);
    };
    const double tip = mean("xTip"), tad = 5.0 / (1.4 * 0.05 * 0.05), tmax = mean("TmaxIn") / tad, burn = mean("burnRatio"),
                 pm = mean("pMean"), leak = mean("fuelOut") / mean("fuelIn");
    double hits = 0;
    for (const auto& r : rows) hits = std::max(hits, r[col("hits")]);
    std::printf("  jet: tip %.3f, half %.3f, TmaxIn/T_ad %.5f, burn/in %.5f, out/in %.2e, p/p0 %.5f, base %.3f, hits %g\n", tip,
                mean("half"), tmax, burn, leak, pm, mean("xBase"), hits);
    check(std::abs(burn - 1) < 0.01, "burn/in within 1%");
    check(std::abs(leak) < 1e-3, "closed tip: out/in < 1e-3");
    check(mean("xBase") <= 0.5, "attached: base <= 2 h");
    check(tmax > 0.95 && tmax < 1.005, "TmaxIn/T_ad in [0.95, 1.005]");
    check(std::abs(pm - 1) < 0.01, "p/p0 within 1%");
    check(hits == 0, "no Z or Y out of bounds");
    check(std::isfinite(tip), "the tip closes");
    return dnstest::finish();
#endif
}

int runFlame1d()
{
#ifndef DNS_BIN
    dnstest::skip("no dns binary");
#else
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / std::format("dns_fvflame_{}", ::getpid());
    fs::create_directories(dir);
    const std::string out = (dir / "f").string();
    const std::string cmd = std::string(DNS_BIN) +
                            " flame --solver fv --fv-precision fp64 --cells-per-delta 7.5 --inflow-speed 0.98 --tt 12 "
                            "--emit stats --quiet --out " + out + " > /dev/null";
    if (std::system(cmd.c_str()) != 0) {
        check(false, "dns flame --solver fv ran");
        return dnstest::finish();
    }
    std::ifstream in(out + "_stats.csv");
    std::string line;
    std::vector<std::string> head;
    double SL = 0, dth = 0, hits = -1, mass0 = 0, mass = 0;
    int cSL = -1, cD = -1, cH = -1, cM = -1;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream s(line);
        std::string f;
        std::vector<std::string> row;
        while (std::getline(s, f, ',')) row.push_back(f);
        if (head.empty()) {
            head = row;
            for (int i = 0; i < int(row.size()); ++i) {
                if (row[std::size_t(i)] == "SL") cSL = i;
                if (row[std::size_t(i)] == "dth") cD = i;
                if (row[std::size_t(i)] == "yHits") cH = i;
                if (row[std::size_t(i)] == "mass") cM = i;
            }
            continue;
        }
        SL = std::stod(row[std::size_t(cSL)]);
        dth = std::stod(row[std::size_t(cD)]);
        hits = std::stod(row[std::size_t(cH)]);
        mass = std::stod(row[std::size_t(cM)]);
        if (mass0 == 0) mass0 = mass;
    }
    fs::remove_all(dir);
    std::printf("  7.5 cells/delta_th: SL %.6f dth %.5f hits %g mass %.6f -> %.6f\n", SL, dth, hits, mass0, mass);
    check(std::abs(SL - 1.0) <= 0.02, std::format("S_L {:.5f} within 2% of the oracle at 7.5 cells", SL));
    check(std::abs(dth / 1.60165 - 1.0) <= 0.03, std::format("delta_th {:.5f} within 3%", dth));
    check(hits == 0, "no Y out of bounds");
    return dnstest::finish();
#endif
}

// A CSV series: header and rows of text fields (comment lines skipped).
struct Csv
{
    std::vector<std::string> head;
    std::vector<std::vector<std::string>> rows;
    int col(const std::string& n) const
    {
        const auto it = std::ranges::find(head, n);
        return it == head.end() ? -1 : int(it - head.begin());
    }
};

Csv readCsv(const std::string& path)
{
    Csv c;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream s(line);
        std::string f;
        std::vector<std::string> row;
        while (std::getline(s, f, ',')) row.push_back(f);
        if (c.head.empty()) c.head = std::move(row);
        else c.rows.push_back(std::move(row));
    }
    return c;
}

// Shock x C0 through the binary (masterplan_combnozzle P3; ctest gpu_fv_flame_shock): fv1d's 7.5-cell
// polyMesh (tools/chem/fv_flame_mesh.py: 197 x 4, y cyclic, z empty), M 0.042, FP64, tt 14:
//   G3.4  --shock hll --shock-d4 0 == no --shock: every column of the plain run's stats, byte for byte
//   G3.3  --shock hll: shockCells 0 on every row; |S(hll) - S(d4 0)| <= |S(d4 0) - S_ref| / 3 ([7, 14] means)
//   route a total inlet with y, [nozzle] astar, [thrust] pa station: rc 0, the stats' nozzle columns,
//         _thrust.csv rows with every res_k <= 1e-12 of P_k (the ledger on the flame route, FP64)
int runFlameShock()
{
#if !defined(DNS_BIN) || !defined(DNS_SOURCE_DIR)
    dnstest::skip("no dns binary");
#else
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / std::format("dns_fvflameshock_{}", ::getpid());
    fs::create_directories(dir);
    const double h = 1.601651 / 7.5;
    const std::string mesh = (dir / "mesh").string();
    const std::string gen = std::format("python3 {}/tools/chem/fv_flame_mesh.py --nx 197 --ny 4 --lx {:.17g} --ly {:.17g} "
                                        "--depth 1 --out {} > /dev/null",
                                        DNS_SOURCE_DIR, 197 * h, 4 * h, mesh);
    if (std::system(gen.c_str()) != 0) {
        check(false, "fv_flame_mesh.py wrote the 7.5-cell mesh");
        return dnstest::finish();
    }
    auto run = [&](const std::string& tag, const std::string& bc, const std::string& extra, double tt) {
        const std::string out = (dir / tag).string();
        const std::string cmd = std::format("{} flame --mach 0.042 --expansion 5 --zeldovich 8 --prandtl 0.72 --polymesh {} --bc {} "
                                            "--upstream 14.133333333333333 --inflow-speed 0.98 --tt {:g} --fv-precision fp64 "
                                            "--emit stats --quiet --out {} {} > /dev/null",
                                            DNS_BIN, mesh, bc, tt, out, extra);
        const int rc = std::system(cmd.c_str());
        check(rc == 0, std::format("dns flame {}: rc 0", tag));
        return out;
    };
    const std::string bc = mesh + "/bc.toml";
    const Csv none = readCsv(run("none", bc, "", 14) + "_stats.csv");
    const Csv d40 = readCsv(run("d40", bc, "--shock hll --shock-d4 0", 14) + "_stats.csv");
    const Csv hll = readCsv(run("hll", bc, "--shock hll", 14) + "_stats.csv");
    bool same = !none.rows.empty() && none.rows.size() == d40.rows.size();
    for (std::size_t r = 0; same && r < none.rows.size(); ++r)
        for (std::size_t k = 0; same && k < none.head.size(); ++k) {
            const int j = d40.col(none.head[k]);
            same = j >= 0 && d40.rows[r][std::size_t(j)] == none.rows[r][k];
        }
    check(same, std::format("G3.4 (7.5 cells): --shock hll --shock-d4 0 == no --shock, {} rows byte for byte", none.rows.size()));
    auto window = [](const Csv& c) {
        const int t = c.col("t"), s = c.col("SL");
        double sum = 0;
        int n = 0;
        for (const auto& r : c.rows)
            if (std::stod(r[std::size_t(t)]) >= 7.0) {
                sum += std::stod(r[std::size_t(s)]);
                ++n;
            }
        return n ? sum / n : NAN;
    };
    double sensed = 0;
    for (const auto& r : hll.rows) sensed = std::max(sensed, std::stod(r[std::size_t(hll.col("shockCells"))]));
    const double sH = window(hll), s0 = window(d40), sref = 1.0 - 1.4 * 0.042 * 0.042 * 4.0, band = std::abs(s0 - sref) / 3.0;
    std::printf("  7.5 cells M 0.042: S_L hll %.7f, d4 0 %.7f, Delta %+.3e, band %.3e; shockCells max %g over %zu rows\n", sH, s0,
                sH - s0, band, sensed, hll.rows.size());
    check(!hll.rows.empty() && sensed == 0, "G3.3: the sensor stays silent on every row");
    check(std::abs(sH - s0) <= band, "G3.3 (7.5 cells): |S(hll) - S(d4 0)| <= |S(d4 0) - S_ref| / 3");
    // The reacting nozzle's tables on the flame route (no nozzle contour: a constant-area smoke).
    const double mu = 0.98 * 0.042, t0 = 1.0 + 0.2 * mu * mu, p0 = std::pow(t0, 3.5);
    const std::string bcN = (dir / "bcN.toml").string();
    {
        std::ofstream f(bcN);
        f << std::format("[reference]\ndirection = [1, 0, 0]\nlift = [0, 1, 0]\n[nozzle]\nastar = {:.17g}\n[thrust]\npa = 1\n"
                         "station = 30\n[patch.xlo]\ntype = \"total\"\np0 = {:.17g}\nt0 = {:.17g}\ny = [1.0]\n[patch.xhi]\n"
                         "type = \"outflow\"\nburnt = true\n",
                         4 * h, p0, t0);
    }
    const std::string outN = run("nozzle", bcN, "--shock hll", 1);
    const Csv st = readCsv(outN + "_stats.csv"), th = readCsv(outN + "_thrust.csv");
    check(st.col("mdotIn") >= 0 && st.col("Cf") >= 0 && st.col("shockCells") >= 0, "route: the stats carry the shock and nozzle columns");
    double res = th.rows.empty() ? HUGE_VAL : 0.0;
    for (const auto& r : th.rows)
        for (const char* k : {"mass", "mx", "E", "ry"}) {
            const double P = std::abs(std::stod(r[std::size_t(th.col(std::string("P_") + k))]));
            res = std::max(res, std::abs(std::stod(r[std::size_t(th.col(std::string("res_") + k))])) / P);
        }
    std::printf("  route (total + y, [nozzle], [thrust]): %zu stats rows, %zu thrust rows, max |res_k| / P_k %.2e\n", st.rows.size(),
                th.rows.size(), res);
    check(!th.rows.empty() && res <= 1e-12, "route: the impulse ledger closes on the flame route (res_k <= 1e-12 P_k)");
    fs::remove_all(dir);
    return dnstest::finish();
#endif
}

// The 0D constant-volume reactor (masterplan_combnozzle P0, docs/chem.md "P0: the 0D reactor"): the
// uniform periodic 4 x 4 x 1 box at rest, C0 sigma 5 beta 8 at M 0.01 units, Y 1, T0 2.5 Tu,
// against the closed quadrature of fvref_reactor.hpp.
namespace rx = dnstest::reactor;

FvState reactorMarch(const rx::ReactorCase& rc, FvPrecision pr, bool comp, double dt, int steps,
                     const std::function<void(int, const FvState&)>& every = {}, int stride = 0)
{
    DeviceFvComp d = orDie(DeviceFvComp::make(rc.fm, {.tr = rc.tr, .bc = rc.bc, .precision = pr, .compensated = comp,
                                                      .chem = rc.chem}),
                           "reactor");
    orDie(d.upload(rc.ic), "upload");
    FvState o;
    for (int n = 1; n <= steps; ++n) {
        d.step(dt);
        if (stride > 0 && n % stride == 0) {
            orDie(d.download(o), "download");
            every(n, o);
        }
    }
    orDie(d.download(o), "download");
    return o;
}

int runReactor()
{
    const rx::ReactorCase rc = orDie(rx::reactorCase(), "reactor case");
    const rx::Reactor& cv = rc.ode;
    const double te = rx::timeTo(cv, 0.5), T5 = cv.T(0.5);
    std::printf("== 0D reactor: t_e = t(Y 1/2) = %.16f, T(1/2) = %.10f Tu, T_end = %.10f Tu\n", te, T5 / rc.Tu,
                cv.T(0.0) / rc.Tu);

    std::printf("== G0.1 identity: FP32, FP32 compensated, FP64, 2000 steps of t_e/2000\n");
    struct Mode
    {
        const char* tag;
        FvPrecision pr;
        bool comp;
    };
    for (const Mode md : {Mode{"FP32", FvPrecision::Fp32, false}, Mode{"FP32 compensated", FvPrecision::Fp32, true},
                          Mode{"FP64", FvPrecision::Fp64, false}}) {
        bool ok = true;
        int seen = 0;
        const FvState o = reactorMarch(rc, md.pr, md.comp, te / 2000, 2000,
                                       [&](int, const FvState& s) {
                                           ok = ok && rx::uniformAtRest(s);
                                           ++seen;
                                       },
                                       50);
        const double y = rx::cellY(o, 0);
        std::printf("  %s: Y(t_e) %.9f, %d rows checked\n", md.tag, y, seen);
        check(ok && rx::uniformAtRest(o), std::format("G0.1 {}: every cell bitwise cell 0's, u v w == 0.0", md.tag));
        check(seen == 40 && y > 0.3 && y < 0.6, std::format("G0.1 {}: live (40 rows, burnt to Y ~ 1/2)", md.tag));
    }

    std::printf("== G0.2 order in dt, FP64: Y(t_e) - 1/2 and T(t_e)/T(1/2) - 1; G0.2c vs exact SSP-RK3\n");
    std::vector<double> eY, eT;
    double finest = 0.0, worstRk = 0.0, kutta = 0.0;
    for (const auto& [N, ref] : rx::kRk3Shu) {
        const FvState o = reactorMarch(rc, FvPrecision::Fp64, false, te / N, N);
        const double dY = rx::cellY(o, 0) - 0.5, dT = rx::cellT(o, 0) / T5 - 1.0;
        eY.push_back(std::abs(dY));
        eT.push_back(std::abs(dT));
        worstRk = std::max(worstRk, std::abs(dY - ref));
        if (N == 2000) kutta = std::abs(dY - rx::kRk3Kutta2000);
        if (N == 8000) finest = dY;
        const std::size_t k = eY.size() - 1;
        std::printf("  N %6d: dY %.6e  dT %.6e  order Y %.3f T %.3f  |dY - RK3 exact| %.2e\n", N, dY, dT,
                    k ? std::log2(eY[k - 1] / eY[k]) : 0.0, k ? std::log2(eT[k - 1] / eT[k]) : 0.0, std::abs(dY - ref));
    }
    auto ord = [](const std::vector<double>& e, std::size_t k) { return std::log2(e[k] / e[k + 1]); };
    const double pY = std::min(ord(eY, 0), ord(eY, 1)), pT = std::min(ord(eT, 0), ord(eT, 1));
    rx::recordUnmet(pY >= 2.8 && pT >= 2.8 && std::abs(finest) <= 1e-9,
          std::format("G0.2 N 2000/4000/8000: order Y {:.3f}, T {:.3f} >= 2.8; finest |dY| {:.2e} <= 1e-9", pY, pT,
                      std::abs(finest)));
    const double yCp = rx::yAt(rx::boxReactor(false), te), ctl = std::abs(0.5 + finest - yCp);
    std::printf("  control: the constant-pressure oracle Y(t_e) %.9f misses the N 8000 run by %.3e\n", yCp, ctl);
    check(ctl >= 0.1, std::format("G0.2 control: the cp oracle misses by {:.3e} >= 1e-1 (fails the gate)", ctl));
    const double pYb = std::min(ord(eY, 3), ord(eY, 4)), pTb = std::min(ord(eT, 3), ord(eT, 4));
    check(pYb >= 2.8 && pTb >= 2.8, std::format("G0.2b (A1) N 16000/32000/64000: order Y {:.3f}, T {:.3f} >= 2.8", pYb, pTb));
    rx::recordUnmet(eY[5] <= 1e-9, std::format("G0.2b (A1) N 64000: |dY| {:.2e} <= 1e-9", eY[5]));
    check(worstRk <= 1e-8, std::format("G0.2c (A1) every rung vs the exact SSP-RK3 recurrence {:.2e} <= 1e-8", worstRk));
    check(kutta > 1e-6, std::format("G0.2c control: N 2000 vs Kutta's RK3 {:.2e} > 1e-6", kutta));

    const FvRef ref = orDie(FvRef::make(rc.fm, {.tr = rc.tr, .bc = rc.bc, .chem = rc.chem}, 1), "reactor FvRef");
    std::printf("== G0.3 full burn to 2 t_e, FP64: the plan's t_e/8000 (recorded) and t_e/64000 (A2)\n");
    const double S0 = rx::coupledEnergy(rc.ic, rc.fm, rc.chem.q), Tend = cv.T(0.0);
    for (const int N : {8000, 64000}) {
        double Sprev = S0, dS = 0.0, bound = HUGE_VAL;
        bool uni = true;
        int rows = 0;
        const FvState fin = reactorMarch(rc, FvPrecision::Fp64, false, te / N, 2 * N,
                                         [&](int, const FvState& s) {
                                             const double S = rx::coupledEnergy(s, rc.fm, rc.chem.q);
                                             dS = std::max(dS, std::abs(S - Sprev) / S0);
                                             Sprev = S;
                                             uni = uni && rx::uniformAtRest(s);
                                             bound = std::min(bound, ref.chemDt(s, gpu::kChemDt));
                                             ++rows;
                                         },
                                         1);
        const double eTend = std::abs(rx::cellT(fin, 0) / Tend - 1.0), yEnd = rx::cellY(fin, 0);
        std::printf("  N %d, %d steps: dt / min chemDt(kChemDt %.1f) = %.3f; Sum V (E + q rho Y) per step max %.2e, total "
                    "%.2e; T/T_end - 1 %.2e; Y %.4e\n",
                    N, rows, gpu::kChemDt, te / N / bound, dS, std::abs(Sprev / S0 - 1.0), eTend, yEnd);
        const std::string tag = N == 8000 ? "G0.3" : "G0.3b (A2)";
        check(dS <= 1e-14, std::format("{} t_e/{}: Sum V (E + q rho Y) per step {:.2e} <= 1e-14", tag, N, dS));
        check(uni && rows == 2 * N, std::format("{} t_e/{}: live, {} rows, every cell cell 0's", tag, N, 2 * N));
        const bool met = eTend <= 1e-12 && std::abs(yEnd) <= 1e-12;
        const std::string what = std::format("{} t_e/{}: final T vs T0 + gamma (sigma - 1) Tu {:.2e} <= 1e-12, |Y| {:.2e} "
                                             "<= 1e-12",
                                             tag, N, eTend, std::abs(yEnd));
        if (N == 8000)
            rx::recordUnmet(met, what);
        else
            check(met, what);
    }

    // G0.4 as planned: 100 consecutive steps of t_e/2000 from the IC. Across the runaway
    // (steps 1900..2000, Y 0.90 -> 0.51) the ODE itself amplifies rounding: there each step is
    // compared from the device's own state, and the continuous window's distance is reported next
    // to FvRef's growth of a 1e-14 rho Y perturbation over the same window (the conditioning).
    std::printf("== G0.4 FvRef vs device FP64, steps of t_e/2000\n");
    const double dt = te / 2000;
    DeviceFvComp d = orDie(DeviceFvComp::make(rc.fm, {.tr = rc.tr, .bc = rc.bc, .precision = FvPrecision::Fp64,
                                                      .chem = rc.chem}),
                           "reactor");
    orDie(d.upload(rc.ic), "upload");
    FvState o;
    for (int n = 0; n < 100; ++n) d.step(dt);
    orDie(d.download(o), "download");
    FvState h = rc.ic;
    ref.march(h, dt, 100);
    const double e0 = relDiff(o.q, h.q, ref.nV(), rc.fm.nC);
    std::printf("  steps 0..100: Y 1 -> %.6f, |device - FvRef| %.2e\n", rx::cellY(o, 0), e0);
    check(e0 <= 1e-13 && rx::cellY(o, 0) < 1.0, std::format("G0.4 steps 0..100: full state {:.2e} <= 1e-13 (live: Y falls)", e0));
    for (int n = 100; n < 1900; ++n) d.step(dt);
    FvState s1900;
    orDie(d.download(s1900), "download");
    double eStep = 0.0;
    for (int n = 0; n < 100; ++n) {
        FvState a;
        orDie(d.download(a), "download");
        d.step(dt);
        orDie(d.download(o), "download");
        ref.step(a, dt);
        eStep = std::max(eStep, relDiff(o.q, a.q, ref.nV(), rc.fm.nC));
    }
    FvState cont = s1900, nudged = s1900;
    for (std::size_t c = 0; c < std::size_t(rc.fm.nC); ++c) nudged.var(5)[c] *= 1.0 + 1e-14;
    ref.march(cont, dt, 100);
    ref.march(nudged, dt, 100);
    const double eCont = relDiff(o.q, cont.q, ref.nV(), rc.fm.nC), grow = relDiff(nudged.q, cont.q, ref.nV(), rc.fm.nC) / 1e-14;
    std::printf("  steps 1900..2000 (Y %.4f -> %.4f, dt/chemDt %.2f at t_e): per step |device - FvRef| max %.2e; "
                "continuous %.2e; a 1e-14 rho Y perturbation grows x%.1f\n",
                rx::cellY(s1900, 0), rx::cellY(o, 0), dt / ref.chemDt(o, gpu::kChemDt), eStep, eCont, grow);
    check(eStep <= 1e-13 && rx::cellY(o, 0) < 0.6,
          std::format("G0.4 steps 1900..2000, each from the device state: {:.2e} <= 1e-13 (live: Y < 0.6)", eStep));
    return dnstest::finish();
}

} // namespace

int main(int argc, char** argv)
{
    test::requireDeviceOrSkip();
    if (argc > 1 && std::string(argv[1]) == "reactor") return runReactor();
    if (argc > 1 && std::string(argv[1]) == "flame1d") return runFlame1d();
    if (argc > 1 && std::string(argv[1]) == "shock") return runFlameShock();
    if (argc > 1 && std::string(argv[1]) == "jet") return runJet();
    if (argc > 1 && std::string(argv[1]) == "mix") {
        gateMix();
        return dnstest::finish();
    }
    gateOracle();
    gatePassive();
    gateConservation();
    gateTwoReactants();
    gateConservation2();
    gateRerun();
    gateStructured();
    return dnstest::finish();
}
