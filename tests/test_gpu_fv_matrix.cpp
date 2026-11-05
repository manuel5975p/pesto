// MC: the generic gate matrix of the compressible FV stage (docs/fvcomp.md "Combinations",
// docs/masterplan_multiphysics.md "Composition, not pairs"). Every LIVE cell of the
// inventory (dns/fvmodules: each module alone and each pair, closed under the
// requirements; tests/fv_matrix_util.hpp) on every mesh that can host it:
//   (a) inert = the stage without the module, byte for byte (FP32): passive species ->
//       the mean rows of the cell without species; TFLES F 1 -> the cell without TFLES;
//       RANS scalars 0 -> the laminar mean rows. Other modules have no inert setting
//       (their off IS absence: tools/fv/comp_bit_identity.sh against the base binary)
//   (b) invariants on the closed boundary set: the rest (hydrostatic with gravity)
//       state's rhs is the analytic source (drive) to rounding of its pressure fluxes;
//       uniform power-of-two Y exact over 10 steps (FP32, compensated, FP64); mass and
//       E + q rho Y_F + rho Phi telescope (FP64, 10 steps; energy not with the drive);
//       Z(Y_F, Y_O) == the carried tracer (two reactants, FP64 and FP32 walk)
//   (c) double instantiation vs FvRef <= 1e-13: rhs, mean and species fluxes, omega,
//       mu_t, the TFLES coefficients, RANS fields, stableDt, 3 steps; float vs double
//   (d) Source / Morton / ColourMajor numbering: 3 FP64 steps, to summation order (the
//       CSR order changes with the numbering; test_fvref E4), bitwise stated if so
//   (e) rerun byte-identical, FP32 plain and compensated, 10 steps
// One row per cell. `test_gpu_fv_matrix markdown` prints docs/fvcomp.md's generated
// "Combinations" section (the inventory + this matrix' rows).
#include "fv_comp_test_util.hpp"
#include "fv_matrix_util.hpp"

#include "dns/gpu/hip_runtime.hpp"
#include "gpu_test_util.hpp"

#include <chrono>
#include <cstring>
#include <map>

#ifndef DNS_MESH_DATA
#error "DNS_MESH_DATA must point at tests/data/mesh"
#endif

using namespace dnstest::fvc;
using dnstest::check;
namespace fvm = dnstest::fvm;
using fvm::has;

namespace {

constexpr std::array<CellOrder, 3> kOrders{CellOrder::Source, CellOrder::Morton, CellOrder::ColourMajor};

struct MeshCase
{
    std::string name;
    std::array<FaceMesh, 3> fm;                       // kOrders
    std::array<std::vector<FvBc>, 3> open, closed;    // per order (patches by name)
    std::array<std::vector<FvBc>, 3> openRest;        // the open set with every boundary velocity 0
    Transport tr;
    bool hydro = false; // the closed set has walls across g: hydrostatic rest is a fixed point
};

FvBc wall()
{
    FvBc b;
    b.kind = FvBcKind::Wall;
    return b;
}

std::vector<MeshCase> meshes()
{
    std::vector<MeshCase> v;
    auto fill = [&](MeshCase m, auto&& build, auto&& openPick, auto&& closedPick) {
        for (std::size_t o = 0; o < kOrders.size(); ++o) {
            m.fm[o] = build(kOrders[o]);
            m.open[o] = bcFor(m.fm[o], openPick);
            m.closed[o] = bcFor(m.fm[o], closedPick);
            m.openRest[o] = m.open[o];
            for (FvBc& b : m.openRest[o]) {
                std::ranges::fill(b.uIn, 0.0);
                b.qFar[4] -= 0.5 * (b.qFar[1] * b.qFar[1] + b.qFar[2] * b.qFar[2] + b.qFar[3] * b.qFar[3]) / b.qFar[0];
                std::fill_n(b.qFar + 1, 3, 0.0);
            }
        }
        v.push_back(std::move(m));
    };
    const Grid g = Grid::make(8, 6, 5, 8 * 0.3, 6 * 0.45, 5 * 0.2);
    fill(MeshCase{.name = "box 8x6x5 periodic", .tr = Transport::constant(kGamma, 0.05)},
         [&](CellOrder o) { return orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), periodicSpec(g), o), "box"); },
         [](const MeshPatch&) { return wall(); }, [](const MeshPatch&) { return wall(); });
    fill(MeshCase{.name = "box_cyclic (y walls)", .tr = Transport::constant(kGamma, 0.05), .hydro = true},
         [](CellOrder o) { return loadMesh("box_cyclic", o); }, [](const MeshPatch&) { return wall(); },
         [](const MeshPatch&) { return wall(); });
    fill(MeshCase{.name = "box_refined (cyclic, split faces), Sutherland",
                  .tr = Transport::air(0.02, 0.9, 288.15, 0.72, kGamma, 1.0, true)},
         [](CellOrder o) { return loadMesh("box_refined", o); }, [](const MeshPatch&) { return wall(); },
         [](const MeshPatch&) { return wall(); });
    auto cylOpen = [](const MeshPatch& p) {
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
        b.y[0] = 0.7;
        b.y[1] = 0.3;
        b.y[2] = 0.7; // Z(0.7, 0.3) of kC1: the tracer row
        return b;
    };
    auto cylClosed = [](const MeshPatch& p) {
        FvBc b;
        b.kind = p.name == "cylinder" ? FvBcKind::Wall : FvBcKind::Slip;
        return b;
    };
    fill(MeshCase{.name = "cyl2d_snappy", .tr = Transport::constant(kGamma, 0.01), .hydro = true},
         [](CellOrder o) { return loadMesh("cyl2d_snappy", o); }, cylOpen, cylClosed);
    return v;
}

bool openSet(const std::vector<FvBc>& bc)
{
    return std::ranges::any_of(bc, [](const FvBc& b) {
        return b.kind == FvBcKind::Inflow || b.kind == FvBcKind::Outflow || b.kind == FvBcKind::Far;
    });
}

// Z of kC1 (Y_F1 = Y_O2 = 1): (s Y_F - Y_O + 1) / (s + 1).
double zOf(double yf, double yo) { return (fvm::kC1.stoich * yf - yo + 1.0) / (fvm::kC1.stoich + 1.0); }

// smoothState + a tanh front in the fuel, smooth other rows (C1: the oxidizer opposite,
// the tracer = Z), RANS scalars smooth and positive. Position only: every order agrees.
// untie: + 1e-3 of a hashed field of the position on e. Mirror-symmetric fields on a
// uniform mesh tie neighbouring T exactly, and the TFLES dynamic sensor's T-gate is a
// switch: rounding flips a tie by O(1) (tiedSensor below measures it).
FvState cellState(const FvConfig& cfg, const FaceMesh& fm, int nV, bool untie = true)
{
    const bool uz = fm.emptyAxis < 0;
    const FvState base = smoothState(fm, uz, 0.6);
    FvState s = FvState::zeros(fm.nC, nV);
    std::copy(base.q.begin(), base.q.end(), s.q.begin());
    const auto [lo, hi] = std::ranges::minmax(fm.C[0]);
    const auto [ly, hy] = std::ranges::minmax(fm.C[1]);
    const int nY = cfg.chem.nY;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double X = (fm.C[0][c] - lo) / (hi - lo), Y = (fm.C[1][c] - ly) / (hy - ly), rho = s.var(0)[c];
        const double front = 0.5 * (1.0 - std::tanh((X - 0.5 - 0.1 * std::sin(6 * Y)) / 0.08));
        if (untie) {
            const double h = std::sin(12.9898 * fm.C[0][c] + 78.233 * fm.C[1][c] + 37.719 * fm.C[2][c]) * 43758.5453;
            s.var(4)[c] *= 1.0 + 1e-3 * (h - std::floor(h) - 0.5);
        }
        if (nY > 0) {
            s.var(4)[c] += 0.8 * (1.0 - front) / (kGamma - 1.0);
            if (cfg.chem.ox >= 0) {
                const double yf = 0.8 * front, yo = 0.9 * (1.0 - front) + 0.05;
                s.var(5)[c] = rho * yf;
                s.var(6)[c] = rho * yo;
                s.var(7)[c] = rho * zOf(yf, yo);
            } else {
                s.var(5)[c] = rho * front;
                s.var(6)[c] = rho * (0.3 + 0.2 * std::cos(5 * X + 2 * Y));
            }
        }
        const int iT = kFvMeanVars + nY;
        if (cfg.rans.model == RansModel::SaNeg) s.var(iT)[c] = rho * 3.0 * cfg.tr.muRef * (1.0 + 0.5 * std::sin(3 * X + Y));
        if (ransSst(cfg.rans.model)) {
            s.var(iT)[c] = rho * 1e-3 * (1.0 + 0.5 * std::sin(3 * X + Y));
            s.var(iT + 1)[c] = rho * (1.0 + 0.3 * std::cos(2 * X - Y));
        }
    }
    return s;
}

// At rest: the matrix' ambient column (fvm::ambient: T0 = 1 / R, p hydrostatic along g, 1
// without gravity; the open rows' bits), species uniform powers of two, RANS scalars uniform.
FvState restState(const FvConfig& cfg, const FaceMesh& fm, int nV)
{
    FvState s = fvAmbientState(fm, fvm::ambient(cfg.tr), cfg.gravity, cfg.tr.gamma, cfg.tr.Rgas, nV);
    const double yk[4] = {0.25, 0.5, 0.125, 0.0625};
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double rho = s.var(0)[c];
        for (int k = 0; k < cfg.chem.nY; ++k) s.var(kFvMeanVars + k)[c] = rho * yk[k];
        const int iT = kFvMeanVars + cfg.chem.nY;
        if (cfg.rans.model == RansModel::SaNeg) s.var(iT)[c] = rho * 3.0 * cfg.tr.muRef;
        if (ransSst(cfg.rans.model)) {
            s.var(iT)[c] = rho * 1e-3;
            s.var(iT + 1)[c] = rho * 1.0;
        }
    }
    return s;
}

FvCompConfig devOf(const FvConfig& c, FvPrecision pr, bool comp = false)
{
    return {.tr = c.tr,
            .bc = c.bc,
            .bcRow = c.bcRow,
            .precision = pr,
            .csLes = c.les.cs,
            .sgs = c.les.sgs,
            .prt = c.les.prt,
            .compensated = comp,
            .chem = c.chem,
            .tfles = c.tfles,
            .rans = c.rans,
            .bodyForce = c.bodyForce,
            .gravity = c.gravity,
            .ambient = c.ambient,
            .turb = c.turb};
}

bool sameBytes(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

double vecDiff(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size()) return HUGE_VAL;
    double e = 0.0, s = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
        e = std::max(e, std::abs(a[i] - b[i]));
        s = std::max(s, std::abs(b[i]));
    }
    return s == 0.0 ? e : e / s;
}

double vecsDiff(const auto& a, const auto& b)
{
    double w = 0.0;
    for (std::size_t k = 0; k < b.size(); ++k) w = std::max(w, vecDiff(a[k], b[k]));
    return w;
}

// A device run: n steps (lts: localDt first), the state.
struct Run
{
    FvState s;
    bool ok = true;
};

Run march(const FaceMesh& fm, const FvCompConfig& c, const FvState& ic, double dt, int n, bool lts)
{
    auto d = DeviceFvComp::make(fm, c);
    if (!d) {
        std::printf("    make: %s\n", d.error().c_str());
        return {.ok = false};
    }
    orDie(d->upload(ic), "upload");
    double h = dt;
    if (lts) h = orDie(d->localDt({.cfl = 0.4}), "localDt") * 0.5;
    for (int i = 0; i < n; ++i) d->step(h);
    Run r;
    orDie(d->download(r.s), "download");
    return r;
}

struct Row
{
    std::string cell;
    int meshes = 0;
    double oracle = 0.0, f32 = 0.0, orders = 0.0, rest64 = 0.0, rest32 = 0.0, open64 = -1.0, open32 = -1.0, mass = -1.0, energy = -1.0, ztr64 = -1.0,
           ztr32 = -1.0;
    bool ordersBitwise = true, pow2 = true, pow2Run = false, rerun = true;
    std::string inert = "n/a";
    bool pass = true;
};

// The rest state's rhs minus the analytic source, over the mean and species rows,
// relative to the cell's pressure-flux scale p sum|S| / V (the rounding of that sum).
double restResidual(const FaceMesh& fm, const FvConfig& cfg, const FvState& s, const FvState& r, int rows)
{
    double e = 0.0;
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        double sa = 0.0;
        for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
            const std::size_t f = std::size_t(fm.cellFace[std::size_t(q)]);
            sa += std::hypot(fm.S[0][f], fm.S[1][f], fm.S[2][f]);
        }
        const double p = (cfg.tr.gamma - 1.0) * s.var(4)[c], scale = p * sa / fm.V[c];
        for (int v = 0; v < rows; ++v) {
            const double src = v >= 1 && v <= 3 ? cfg.bodyForce[std::size_t(v - 1)] : 0.0;
            e = std::max(e, std::abs(r.var(v)[c] - src) / scale);
        }
    }
    return e;
}

template <class R>
bool exactRatio(const FvState& s, int row, double y)
{
    for (std::size_t c = 0; c < std::size_t(s.nC); ++c)
        if (R(s.var(row)[c]) != R(y) * R(s.var(0)[c])) return false;
    return true;
}

Row runCell(const FvModuleSet& cell, const std::vector<MeshCase>& ms)
{
    Row row{.cell = fvModuleSetName(cell)};
    const bool lts = has(cell, FvModule::Steady), grav = has(cell, FvModule::Gravity);
    const bool drive = has(cell, FvModule::BodyForce);
    // Two reactants: C1, and C2b (the same transport, the law on the rate only).
    const bool c1 = has(cell, FvModule::ChemC1) || has(cell, FvModule::ChemC2b), chem = has(cell, FvModule::ChemC0) || c1;
    const bool tf = has(cell, FvModule::TflesUniform) || has(cell, FvModule::TflesDynamic);
    const bool eff = has(cell, FvModule::TflesEfficiency);
    const bool rans = has(cell, FvModule::RansSa) || has(cell, FvModule::RansSst);
    std::vector<std::string> inert;
    bool inertOk = true;
    for (const MeshCase& m : ms) {
        constexpr std::size_t kM = 1; // Morton: the cell order every gate but (d) runs on
        const FaceMesh& fm = m.fm[kM];
        // Gravity on the open set: its Far / Outflow faces read the hydrostatic ambient's rows.
        const auto& bcA = m.open[kM];
        const FvConfig cfg = fvm::matrixConfig(cell, fm, bcA, m.tr);
        const FvRef ref = orDie(FvRef::make(fm, cfg), row.cell + " on " + m.name);
        const int nV = ref.nV(), nC = fm.nC;
        const int mean = kFvMeanVars + cfg.chem.nY; // rows the rest / mass gates read
        ++row.meshes;
        const FvState s0 = cellState(cfg, fm, nV);
        const double dt = 0.25 * orDie(ref.stableDt(s0), "dt");

        // (c) FP64 device vs FvRef, float vs double.
        {
            auto dev = orDie(DeviceFvComp::make(fm, devOf(cfg, FvPrecision::Fp64)), "dev");
            auto flt = orDie(DeviceFvComp::make(fm, devOf(cfg, FvPrecision::Fp32)), "flt");
            orDie(dev.upload(s0), "upload");
            orDie(flt.upload(s0), "upload");
            FvEval eR, eD;
            FvState rR, rD, rF;
            ref.evaluate(s0, eR, rR);
            orDie(dev.evaluate(eD, rD), "evaluate");
            orDie(flt.rhs(rF), "rhs");
            double e = relDiff(rD.q, rR.q, nV, nC);
            e = std::max(e, vecsDiff(eD.F, eR.F));
            if (chem) e = std::max({e, vecsDiff(eD.Fy, eR.Fy), vecDiff(eD.omega, eR.omega)});
            if (cfg.les.cs > 0) e = std::max(e, vecDiff(eD.nut, eR.nut));
            if (tf) e = std::max({e, vecDiff(eD.fe, eR.fe), vecDiff(eD.eof, eR.eof), vecDiff(eD.om, eR.om)});
            if (rans) e = std::max(e, vecsDiff(eD.tfld, eR.tfld));
            const double dtD = orDie(dev.stableDt(), "dt");
            e = std::max(e, std::abs(dtD / orDie(ref.stableDt(s0), "dt") - 1.0));
            double f = relDiff(rF.q, rD.q, nV, nC);
            if (!lts) {
                FvState q = s0, qD, qF;
                ref.march(q, dt, 3);
                for (int n = 0; n < 3; ++n) {
                    dev.step(dt);
                    flt.step(dt);
                }
                orDie(dev.download(qD), "download");
                orDie(flt.download(qF), "download");
                e = std::max(e, relDiff(qD.q, q.q, nV, nC));
                f = std::max(f, relDiff(qF.q, qD.q, nV, nC));
            }
            row.oracle = std::max(row.oracle, e);
            if (f > 2e-5 || e > 1e-13) std::printf("    %s: oracle %.2e, f32 %.2e\n", m.name.c_str(), e, f);
            row.f32 = std::max(row.f32, f);
        }

        // (d) numberings, FP64, 3 steps, compared in source numbering.
        {
            std::vector<double> base;
            for (std::size_t o = 0; o < kOrders.size(); ++o) {
                const FaceMesh& fo = m.fm[o];
                const FvConfig co = fvm::matrixConfig(cell, fo, m.open[o], m.tr);
                const Run r = march(fo, devOf(co, FvPrecision::Fp64), cellState(co, fo, nV), dt, 3, lts);
                if (!r.ok) {
                    row.pass = false;
                    continue;
                }
                std::vector<double> bySrc(r.s.q.size());
                for (std::size_t v = 0; v < std::size_t(nV); ++v)
                    for (std::size_t c = 0; c < std::size_t(nC); ++c)
                        bySrc[v * std::size_t(nC) + std::size_t(fo.srcCell[c])] = r.s.q[v * std::size_t(nC) + c];
                if (base.empty()) {
                    base = std::move(bySrc);
                    continue;
                }
                const double eo = relDiff(bySrc, base, nV, nC);
                if (eo > 1e-13) std::printf("    %s, numbering %zu: %.2e\n", m.name.c_str(), o, eo);
                row.orders = std::max(row.orders, eo);
                row.ordersBitwise = row.ordersBitwise && sameBytes(bySrc, base);
            }
        }

        // (e) rerun, FP32 plain and compensated.
        for (const bool comp : {false, true}) {
            const Run a = march(fm, devOf(cfg, FvPrecision::Fp32, comp), s0, dt, 10, lts);
            const Run b = march(fm, devOf(cfg, FvPrecision::Fp32, comp), s0, dt, 10, lts);
            row.rerun = row.rerun && a.ok && b.ok && sameBytes(a.s.q, b.s.q);
        }

        // (b) invariants on the closed set.
        const auto& bcC = m.closed[kM];
        const FvConfig cc = fvm::matrixConfig(cell, fm, bcC, m.tr, {.passive = true});
        if (!grav || m.hydro) {
            const FvState rest = restState(cc, fm, nV);
            for (const FvPrecision pr : {FvPrecision::Fp64, FvPrecision::Fp32}) {
                auto d = orDie(DeviceFvComp::make(fm, devOf(cc, pr)), "rest");
                orDie(d.upload(rest), "rest");
                FvState r;
                orDie(d.rhs(r), "rest");
                const double e = restResidual(fm, cc, rest, r, mean);
                (pr == FvPrecision::Fp64 ? row.rest64 : row.rest32) =
                    std::max(pr == FvPrecision::Fp64 ? row.rest64 : row.rest32, e);
            }
        }
        // Gravity on the open set at rest: the ambient rows make the hydrostatic column a fixed
        // point through Far, Outflow and Inflow faces too (docs/fvcomp.md "Hydrostatic ambient").
        if (grav && m.hydro && openSet(m.openRest[kM])) {
            const FvConfig co = fvm::matrixConfig(cell, fm, m.openRest[kM], m.tr, {.passive = true});
            const FvState rest = restState(co, fm, nV);
            for (const FvPrecision pr : {FvPrecision::Fp64, FvPrecision::Fp32}) {
                auto d = orDie(DeviceFvComp::make(fm, devOf(co, pr)), "open rest");
                orDie(d.upload(rest), "open rest");
                FvState r;
                orDie(d.rhs(r), "open rest");
                double& w = pr == FvPrecision::Fp64 ? row.open64 : row.open32;
                w = std::max(w, restResidual(fm, co, rest, r, mean));
            }
        }
        if (chem) {
            FvState u = cellState(cc, fm, nV);
            const double yk[3] = {0.25, 0.5, 0.125};
            for (std::size_t c = 0; c < std::size_t(nC); ++c)
                for (int k = 0; k < cc.chem.nY; ++k) u.var(kFvMeanVars + k)[c] = yk[k] * u.var(0)[c];
            struct Mode
            {
                FvPrecision pr;
                bool comp;
            };
            for (const Mode md : {Mode{FvPrecision::Fp32, false}, Mode{FvPrecision::Fp32, true}, Mode{FvPrecision::Fp64, false}}) {
                const Run r = march(fm, devOf(cc, md.pr, md.comp), u, dt, 10, false);
                bool ok = r.ok;
                for (int k = 0; ok && k < cc.chem.nY; ++k) {
                    if (md.comp) {
                        for (std::size_t c = 0; ok && c < std::size_t(nC); ++c)
                            ok = r.s.var(kFvMeanVars + k)[c] == yk[k] * r.s.var(0)[c];
                    } else {
                        ok = md.pr == FvPrecision::Fp32 ? exactRatio<float>(r.s, kFvMeanVars + k, yk[k])
                                                        : exactRatio<double>(r.s, kFvMeanVars + k, yk[k]);
                    }
                }
                row.pow2 = row.pow2 && ok;
                row.pow2Run = true;
            }
        }
        if (!lts) {
            // Burning (the real chemistry), FP64, 10 steps.
            const FvConfig cb = fvm::matrixConfig(cell, fm, bcC, m.tr);
            const FvState s = cellState(cb, fm, nV);
            const Run r = march(fm, devOf(cb, FvPrecision::Fp64), s, dt, 10, false);
            auto totals = [&](const FvState& q, double& e) {
                double mass = 0.0;
                e = 0.0;
                for (std::size_t c = 0; c < std::size_t(nC); ++c) {
                    double phi = 0.0;
                    for (std::size_t a = 0; a < 3; ++a) phi -= cb.gravity[a] * fm.C[a][c];
                    const double heat = cb.chem.nY ? cb.chem.q * q.var(kFvMeanVars + cb.chem.fuel)[c] : 0.0;
                    mass += fm.V[c] * q.var(0)[c];
                    e += fm.V[c] * (q.var(4)[c] + heat + q.var(0)[c] * phi);
                }
                return mass;
            };
            double e0, e1;
            const double m0 = totals(s, e0), m1 = totals(r.s, e1);
            row.mass = std::max(row.mass, std::abs(m1 / m0 - 1.0));
            // Energy: no work term (drive), and the potential -g.x must be single-valued
            // (no cyclic pair along g).
            if (!drive && (!grav || m.hydro)) row.energy = std::max(row.energy, std::abs(e1 / e0 - 1.0));
            if (c1) {
                // Z == tracer: FP64 and the FP32 walk, 20 steps.
                for (const FvPrecision pr : {FvPrecision::Fp64, FvPrecision::Fp32}) {
                    const Run z = march(fm, devOf(cb, pr), s, dt, 20, false);
                    double dz = 0.0;
                    for (std::size_t c = 0; c < std::size_t(nC); ++c) {
                        const double rho = z.s.var(0)[c];
                        dz = std::max(dz, std::abs(zOf(z.s.var(5)[c] / rho, z.s.var(6)[c] / rho) - z.s.var(7)[c] / rho));
                    }
                    double& w = pr == FvPrecision::Fp64 ? row.ztr64 : row.ztr32;
                    w = std::max(w, dz);
                }
            }
        }

        // (a) inert settings: byte for byte the cell without the module (FP32, 10 steps).
        auto mean5 = [&](const FvState& q) {
            return std::vector<double>(q.q.begin(), q.q.begin() + std::ptrdiff_t(kFvMeanVars) * nC);
        };
        if (chem && !tf && !eff) {
            FvModuleSet without = cell;
            without.reset(std::size_t(FvModule::ChemC0));
            without.reset(std::size_t(FvModule::ChemC1));
            without.reset(std::size_t(FvModule::ChemC2b));
            // Turb needs a flame: its rows go with the species, at u' 0 (its own inert gate).
            without.reset(std::size_t(FvModule::Turb));
            const FvConfig cw = fvm::matrixConfig(without, fm, bcA, m.tr);
            const FvConfig cp = fvm::matrixConfig(cell, fm, bcA, m.tr, {.passive = true, .turbAmp = 0.0});
            const FvState sp = cellState(cp, fm, nV);
            FvState sw = FvState::zeros(nC, nV - cp.chem.nY);
            std::copy_n(sp.q.begin(), std::size_t(kFvMeanVars) * std::size_t(nC), sw.q.begin());
            const Run a = march(fm, devOf(cp, FvPrecision::Fp32), sp, dt, 10, lts);
            const Run b = march(fm, devOf(cw, FvPrecision::Fp32), sw, dt, 10, lts);
            const bool ok = a.ok && b.ok && sameBytes(mean5(a.s), mean5(b.s));
            inertOk = inertOk && ok;
            if (std::ranges::find(inert, "passive species") == inert.end()) inert.push_back("passive species");
        }
        // C2b with C(xi) == 1 (ln C 0, no spline terms: phiRateLn is exactly 0) is C1's
        // stage byte for byte: the law's kernel and dt functor add nothing else.
        if (has(cell, FvModule::ChemC2b)) {
            FvModuleSet one = cell;
            one.reset(std::size_t(FvModule::ChemC2b));
            one.set(std::size_t(FvModule::ChemC1));
            one.reset(std::size_t(FvModule::Turb)); // refused with C1: compared at u' 0 rows
            const FvConfig c1c = fvm::matrixConfig(one, fm, bcA, m.tr);
            FvConfig cz = fvm::matrixConfig(cell, fm, bcA, m.tr, {.turbAmp = 0.0});
            std::ranges::fill(cz.chem.law.lnC, 0.0);
            std::ranges::fill(cz.chem.law.m2, 0.0);
            for (const bool comp : {false, true}) {
                const Run a = march(fm, devOf(cz, FvPrecision::Fp32, comp), s0, dt, 10, false);
                const Run b = march(fm, devOf(c1c, FvPrecision::Fp32, comp), s0, dt, 10, false);
                inertOk = inertOk && a.ok && b.ok && sameBytes(a.s.q, b.s.q);
            }
            auto d1 = orDie(DeviceFvComp::make(fm, devOf(cz, FvPrecision::Fp64)), "C = 1");
            auto d2 = orDie(DeviceFvComp::make(fm, devOf(c1c, FvPrecision::Fp64)), "C1");
            orDie(d1.upload(s0), "upload");
            orDie(d2.upload(s0), "upload");
            inertOk = inertOk && orDie(d1.chemDt(0.5), "chemDt") == orDie(d2.chemDt(0.5), "chemDt");
            if (std::ranges::find(inert, "C(xi) 1 = C1") == inert.end()) inert.push_back("C(xi) 1 = C1");
        }
        // TFLES F 1 is the DNS only without LES: with it, Omega > 0 still switches the SGS
        // scalar flux off (docs/fvcomp.md "Species LES and TFLES").
        if (tf && !eff && cfg.les.cs == 0.0) {
            FvModuleSet without = cell;
            without.reset(std::size_t(FvModule::TflesUniform));
            without.reset(std::size_t(FvModule::TflesDynamic));
            const FvConfig cw = fvm::matrixConfig(without, fm, bcA, m.tr);
            const FvConfig c1f = fvm::matrixConfig(cell, fm, bcA, m.tr, {.tflesF = 1.0});
            const Run a = march(fm, devOf(c1f, FvPrecision::Fp32), s0, dt, 10, false);
            const Run b = march(fm, devOf(cw, FvPrecision::Fp32), s0, dt, 10, false);
            const bool ok = a.ok && b.ok && sameBytes(a.s.q, b.s.q);
            inertOk = inertOk && ok;
            if (std::ranges::find(inert, "TFLES F 1") == inert.end()) inert.push_back("TFLES F 1");
        }
        // Turb (M5): rows of u' 0 are their patches' rows: the cell without Turb byte for byte.
        if (has(cell, FvModule::Turb)) {
            FvModuleSet without = cell;
            without.reset(std::size_t(FvModule::Turb));
            const FvConfig cw = fvm::matrixConfig(without, fm, bcA, m.tr);
            const FvConfig cz = fvm::matrixConfig(cell, fm, bcA, m.tr, {.turbAmp = 0.0});
            for (const bool comp : {false, true}) {
                const Run a = march(fm, devOf(cz, FvPrecision::Fp32, comp), s0, dt, 10, false);
                const Run b = march(fm, devOf(cw, FvPrecision::Fp32, comp), s0, dt, 10, false);
                inertOk = inertOk && a.ok && b.ok && sameBytes(a.s.q, b.s.q);
            }
            if (std::ranges::find(inert, "turb u' 0") == inert.end()) inert.push_back("turb u' 0");
        }
        if (rans && !lts) { // LTS: the closure's own dt bound moves lts
            FvModuleSet without = cell;
            without.reset(std::size_t(FvModule::RansSa));
            without.reset(std::size_t(FvModule::RansSst));
            const FvConfig cw = fvm::matrixConfig(without, fm, bcA, m.tr);
            const FvConfig cz = fvm::matrixConfig(cell, fm, bcA, m.tr, {.ransZero = true});
            FvState sz = cellState(cz, fm, nV);
            const int iT = kFvMeanVars;
            for (std::size_t c = 0; c < std::size_t(nC); ++c) {
                sz.var(iT)[c] = 0.0;
                if (ransSst(cz.rans.model)) sz.var(iT + 1)[c] = sz.var(0)[c];
            }
            FvState sw = FvState::zeros(nC, kFvMeanVars);
            std::copy_n(sz.q.begin(), std::size_t(kFvMeanVars) * std::size_t(nC), sw.q.begin());
            const Run a = march(fm, devOf(cz, FvPrecision::Fp32), sz, dt, 10, lts);
            const Run b = march(fm, devOf(cw, FvPrecision::Fp32), sw, dt, 10, lts);
            const bool ok = a.ok && b.ok && sameBytes(mean5(a.s), mean5(b.s));
            inertOk = inertOk && ok;
            if (std::ranges::find(inert, "RANS scalars 0") == inert.end()) inert.push_back("RANS scalars 0");
        }
    }
    if (!inert.empty()) {
        row.inert.clear();
        for (const auto& s : inert) row.inert += (row.inert.empty() ? "" : ", ") + s;
        row.inert += inertOk ? ": bitwise" : ": DIFFERS";
    }
    const bool zok = !c1 || (row.ztr64 <= 1e-13 && row.ztr32 <= 4 * std::ldexp(1.0, -23) * std::sqrt(3.0 * 20));
    row.pass = row.pass && inertOk && row.oracle <= 1e-13 && row.f32 <= 2e-5 && row.orders <= 1e-13 && row.rest64 <= 1e-13 &&
               row.rest32 <= 1e-5 && row.open64 <= 1e-13 && row.open32 <= 1e-5 && row.mass <= 1e-13 && row.energy <= 1e-13 && row.pow2 && row.rerun && zok;
    return row;
}

std::string sci(double v) { return v < 0 ? std::string("n/a") : std::format("{:.1e}", v); }

// The TFLES dynamic sensor on a TIED state (cellState without untie): every numbering vs
// Source, 3 FP64 steps, and FP32 vs FP64 over 3 steps. Measured, not gated: the T-gate is a
// switch, an exact tie flips under rounding.
void tiedSensor(const std::vector<MeshCase>& ms)
{
    const FvModuleSet cell = fvModuleSet({FvModule::TflesDynamic, FvModule::ChemC0});
    for (const MeshCase& m : ms) {
        double eo = 0.0, ef = 0.0;
        std::vector<double> base;
        for (std::size_t o = 0; o < kOrders.size(); ++o) {
            const FaceMesh& fo = m.fm[o];
            const FvConfig co = fvm::matrixConfig(cell, fo, m.open[o], m.tr);
            const FvRef ref = orDie(FvRef::make(fo, co), "tied");
            const FvState s = cellState(co, fo, ref.nV(), false);
            const double dt = 0.25 * orDie(ref.stableDt(s), "dt");
            const Run r = march(fo, devOf(co, FvPrecision::Fp64), s, dt, 3, false);
            const Run f = march(fo, devOf(co, FvPrecision::Fp32), s, dt, 3, false);
            ef = std::max(ef, relDiff(f.s.q, r.s.q, ref.nV(), fo.nC));
            std::vector<double> bySrc(r.s.q.size());
            const std::size_t nC = std::size_t(fo.nC);
            for (std::size_t v = 0; v < std::size_t(ref.nV()); ++v)
                for (std::size_t c = 0; c < nC; ++c) bySrc[v * nC + std::size_t(fo.srcCell[c])] = r.s.q[v * nC + c];
            if (o == 0) {
                base = std::move(bySrc);
                continue;
            }
            eo = std::max(eo, relDiff(bySrc, base, ref.nV(), fo.nC));
        }
        std::printf("  tied state, TFLES dynamic + C0, %s: numberings 3 steps %.2e, FP32 vs FP64 3 steps %.2e\n",
                    m.name.c_str(), eo, ef);
    }
}

void matrix(bool markdown)
{
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<MeshCase> ms = meshes();
    std::vector<Row> rows;
    for (const FvModuleSet& cell : fvm::matrixCells(true)) {
        Row r = runCell(cell, ms);
        if (!markdown)
            std::printf("  %-44s %d meshes | oracle %s f32 %s | orders %s%s | rest %s/%s open %s/%s | mass %s energy %s | pow2 %s | "
                        "Z %s/%s | inert %s | rerun %s\n",
                        r.cell.c_str(), r.meshes, sci(r.oracle).c_str(), sci(r.f32).c_str(), sci(r.orders).c_str(),
                        r.ordersBitwise ? " (bitwise)" : "", sci(r.rest64).c_str(), sci(r.rest32).c_str(),
                        sci(r.open64).c_str(), sci(r.open32).c_str(), sci(r.mass).c_str(), sci(r.energy).c_str(), !r.pow2Run ? "n/a" : r.pow2 ? "exact" : "BROKEN", sci(r.ztr64).c_str(),
                        sci(r.ztr32).c_str(), r.inert.c_str(), r.rerun ? "identical" : "DIFFERS");
        check(r.pass, std::format("cell {}: every generic gate", r.cell));
        rows.push_back(std::move(r));
    }
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  %zu live cells, %.1f s\n", rows.size(), sec);
    tiedSensor(ms);
    if (!markdown) return;
    std::printf("%s", fvCombinationsMarkdown().c_str());
    std::printf("\n<!-- generated: test_gpu_fv_matrix markdown (tests/test_gpu_fv_matrix.cpp) -->\n");
    std::printf("Harness rows (`gpu_fv_matrix`, %zu live cells on the meshes box 8x6x5 periodic, box_cyclic, box_refined, "
                "cyl2d_snappy; gravity on the open set through the hydrostatic ambient's rows). oracle: double vs FvRef, max relative over every "
                "pass, 3 steps; orders: Source / Morton / ColourMajor, 3 FP64 steps; rest: |rhs - source| / (p sum|S| / V), "
                "FP64 / FP32 on the closed set, open rest on the open set at rest (gravity, cyl2d_snappy: Far, Outflow, Inflow faces); mass, energy: 10 FP64 steps; Z: max|Z(Y) - tracer|, FP64 / FP32, 20 steps. Steady LTS "
                "cells: the oracle is the rhs and stableDt (FvRef has no LTS), mass n/a (LTS drifts it). Energy n/a with "
                "the drive (its work).\n\n",
                rows.size());
    std::printf("| Cell | oracle | f32 vs f64 | orders | rest FP64 / FP32 | open rest | mass | energy | pow2 Y | Z == tracer | inert | rerun |\n");
    std::printf("|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (const Row& r : rows)
        std::printf("| %s | %s | %s | %s%s | %s / %s | %s | %s | %s | %s | %s | %s | %s |\n", r.cell.c_str(), sci(r.oracle).c_str(),
                    sci(r.f32).c_str(), sci(r.orders).c_str(), r.ordersBitwise ? " bitwise" : "", sci(r.rest64).c_str(),
                    sci(r.rest32).c_str(), r.open64 < 0 ? "n/a" : std::format("{:.1e} / {:.1e}", r.open64, r.open32).c_str(),
                    sci(r.mass).c_str(), sci(r.energy).c_str(), !r.pow2Run ? "n/a" : r.pow2 ? "exact" : "BROKEN",
                    r.ztr64 < 0 ? "n/a" : std::format("{:.1e} / {:.1e}", r.ztr64, r.ztr32).c_str(), r.inert.c_str(),
                    r.rerun ? "byte-identical" : "DIFFERS");
}

// --- physics (`test_gpu_fv_matrix physics`): one sanity number per pair turned on in MC.

// A 2D box, every side a resting adiabatic wall (closed: gravity's rule).
FaceMesh wallBox(int n, double L)
{
    const Grid g = Grid::make(n, n, 1, L, L, L / n);
    const StructuredSpec sp{.lo = {0, 0, 0},
                            .hi = {n - 1, n - 1, 0},
                            .name = {"a", "b", "c", "d", "e", "f"},
                            .type = {FoamPatchType::Wall, FoamPatchType::Wall, FoamPatchType::Wall, FoamPatchType::Wall,
                                     FoamPatchType::Empty, FoamPatchType::Empty}};
    return orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Morton), "wall box");
}

// P1 species + gravity, P2 + LES: a hot product pocket (a passive marker Y_1, T 1.5 T0 at
// the ambient's p) in a hydrostatic isothermal column. The marker's centroid rises under g
// and stays put without (the pocket sits at the box centre: up-down symmetric to rounding).
// Early-time acceleration vs the 2D added-mass estimate g (rho_a - rho_p)/(rho_a + rho_p)
// at the pocket centre. LES: mu_t 0 at rest; rise LES / DNS.
void pocket()
{
    std::printf("== P1 species + gravity, P2 + LES: buoyant product pocket, 64 x 64 closed box, FP64\n");
    const FaceMesh fm = wallBox(64, 1.0);
    const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return wall(); });
    const Transport tr = Transport::constant(kGamma, 0.002);
    const double g = -0.4, R = tr.Rgas, T0 = 1.0 / R;
    FvChem chem = fvm::kC0;
    chem.lnA = -std::numeric_limits<double>::infinity();
    chem.q = 0.0;
    auto ic = [&](double gy) {
        FvState s = FvState::zeros(fm.nC, kFvMeanVars + chem.nY);
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            const double x = fm.C[0][c], y = fm.C[1][c];
            const double p = std::exp(gy * y / (R * T0));
            const double m = std::exp(-(std::pow(x - 0.5, 2) + std::pow(y - 0.5, 2)) / (0.08 * 0.08));
            const double T = T0 * (1.0 + 0.5 * m), rho = p / (R * T);
            s.var(0)[c] = rho;
            s.var(4)[c] = p / (kGamma - 1.0);
            s.var(6)[c] = rho * m;
        }
        return s;
    };
    auto centroid = [&](const FvState& s) {
        double a = 0.0, b = 0.0;
        for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
            a += fm.V[c] * s.var(6)[c] * fm.C[1][c];
            b += fm.V[c] * s.var(6)[c];
        }
        return a / b;
    };
    struct Out
    {
        double rise = 0, a = 0, mutMax = 0, mutRest = -1;
        int steps = 0;
    };
    const double tEnd = 1.0, tA = 0.1;
    auto run = [&](double gy, double cs) {
        FvCompConfig c{.tr = tr, .bc = bc, .precision = FvPrecision::Fp64, .csLes = cs, .chem = chem, .gravity = {0.0, gy, 0.0}};
        auto d = orDie(DeviceFvComp::make(fm, c), "pocket");
        Out o;
        if (cs > 0) {
            // mu_t at the hydrostatic rest (no pocket): exactly 0.
            FvState rest = ic(gy);
            for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
                const double p = (kGamma - 1.0) * rest.var(4)[q];
                rest.var(0)[q] = p / (R * T0);
                rest.var(6)[q] = 0.0;
            }
            orDie(d.upload(rest), "rest");
            o.mutRest = maxAbs(orDie(d.eddyViscosity(), "nut"));
        }
        const FvState s0 = ic(gy);
        orDie(d.upload(s0), "pocket");
        const double y0 = centroid(s0);
        const double dt = 0.5 * orDie(d.stableDt({.cfl = 0.4}), "dt");
        double t = 0.0;
        FvState s;
        bool tookA = false;
        while (t < tEnd - 1e-12) {
            const double h = std::min(dt, tEnd - t);
            d.step(h);
            t += h;
            ++o.steps;
            if (!tookA && t >= tA - 1e-12) {
                orDie(d.download(s), "pocket");
                o.a = 2.0 * (centroid(s) - y0) / (t * t);
                tookA = true;
            }
            if (cs > 0 && o.steps % 50 == 0) o.mutMax = std::max(o.mutMax, maxAbs(orDie(d.eddyViscosity(), "nut")));
        }
        orDie(d.download(s), "pocket");
        o.rise = centroid(s) - y0;
        return o;
    };
    const Out dnsG = run(g, 0.0), dns0 = run(0.0, 0.0), les = run(g, 0.17);
    const double rhoA = 1.0, rhoP = 1.0 / 1.5, aEst = -g * (rhoA - rhoP) / (rhoA + rhoP);
    std::printf("  DNS g %.2f: rise %.5f at t %.1f (%d steps), early acceleration %.4f vs added-mass estimate %.4f (ratio %.3f)\n",
                g, dnsG.rise, tEnd, dnsG.steps, dnsG.a, aEst, dnsG.a / aEst);
    std::printf("  DNS g 0: rise %.2e\n", dns0.rise);
    std::printf("  LES smag 0.17: mu_t at rest %.1e, max mu_t / mu %.3f, rise %.5f = %.4f x DNS\n", les.mutRest,
                les.mutMax / tr.muRef, les.rise, les.rise / dnsG.rise);
    check(dnsG.rise > 0.0 && std::abs(dns0.rise) < 1e-10 * std::abs(dnsG.rise) + 1e-14,
          "the hot pocket rises under g, stays without");
    check(dnsG.a > 0.3 * aEst && dnsG.a < 1.5 * aEst, "early acceleration within the added-mass estimate's bracket [0.3, 1.5]");
    check(les.mutRest == 0.0, "LES under gravity: mu_t exactly 0 at the hydrostatic rest");
    check(les.rise > 0.0 && les.rise <= dnsG.rise * 1.05, "LES rise: positive, not above DNS + 5%");
}

// P3 gravity + bodyForce, P4 + passive species: plane Poiseuille across gravity. x cyclic,
// y walls, f per volume: u = f y (H - y) / (2 mu) whatever rho(y) does (no convection,
// constant mu). FP64, from the hydrostatic rest to steady.
void poiseuille()
{
    std::printf("== P3 gravity + bodyForce, P4 + species: Poiseuille across g, FP64\n");
    const double mu = 0.05, f = 0.004, H = 1.0, g = -0.4;
    double prev = 0.0;
    for (const int ny : {16, 32}) {
        const Grid gr = Grid::make(4, ny, 1, 4.0 / ny, H, 1.0 / ny);
        const StructuredSpec sp{.lo = {0, 0, 0},
                                .hi = {3, ny - 1, 0},
                                .name = {"xl", "xh", "lo", "hi", "e", "f"},
                                .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall, FoamPatchType::Wall,
                                         FoamPatchType::Empty, FoamPatchType::Empty}};
        const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::uniform(gr), sp, CellOrder::Morton), "channel");
        const std::vector<FvBc> bc = bcFor(fm, [](const MeshPatch&) { return wall(); });
        const Transport tr = Transport::constant(kGamma, mu);
        const double R = tr.Rgas, T0 = 1.0 / R;
        double err[2] = {0, 0};
        std::vector<double> u[2];
        for (const bool species : {false, true}) {
            FvChem chem = fvm::kC0;
            chem.lnA = -std::numeric_limits<double>::infinity();
            chem.q = 0.0;
            FvCompConfig c{.tr = tr, .bc = bc, .precision = FvPrecision::Fp64, .bodyForce = {f, 0.0, 0.0},
                           .gravity = {0.0, g, 0.0}};
            if (species) c.chem = chem;
            auto d = orDie(DeviceFvComp::make(fm, c), "channel");
            FvState s = FvState::zeros(fm.nC, kFvMeanVars + c.chem.nY);
            for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
                const double p = std::exp(g * fm.C[1][q] / (R * T0)), rho = p / (R * T0);
                s.var(0)[q] = rho;
                s.var(4)[q] = p / (kGamma - 1.0);
                if (species) {
                    s.var(5)[q] = 0.25 * rho;
                    s.var(6)[q] = 0.5 * rho;
                }
            }
            orDie(d.upload(s), "channel");
            const double tEnd = 2.0 * H * H / mu; // the slowest mode, nu (pi/H)^2: e^-19.7
            double t = 0.0, dt = 0.0;
            int n = 0;
            while (t < tEnd - 1e-12) {
                if (n % 64 == 0) dt = 0.9 * orDie(d.stableDt({.cfl = 0.4}), "dt");
                const double h = std::min(dt, tEnd - t);
                d.step(h);
                t += h;
                ++n;
            }
            orDie(d.download(s), "channel");
            const double uc = f * H * H / (8.0 * mu);
            for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
                const double y = fm.C[1][q], ue = f * y * (H - y) / (2.0 * mu);
                u[species].push_back(s.var(1)[q] / s.var(0)[q]);
                err[species] = std::max(err[species], std::abs(u[species].back() - ue) / uc);
            }
            if (species) {
                bool exact = true;
                for (std::size_t q = 0; q < std::size_t(fm.nC); ++q)
                    exact = exact && s.var(5)[q] == 0.25 * s.var(0)[q] && s.var(6)[q] == 0.5 * s.var(0)[q];
                check(exact, std::format("ny {}: uniform Y exact in FP64 after {} steps", ny, n));
            }
            if (!species) std::printf("  ny %d: %d steps to t %.0f, max|u - u_exact| / u_c %.3e (u_c %.4f, Mach %.3f)\n", ny, n,
                                      tEnd, err[0], uc, uc / std::sqrt(kGamma));
        }
        check(sameBytes(u[0], u[1]), std::format("ny {}: a passive species leaves the driven channel's u bitwise", ny));
        if (prev > 0) std::printf("  order %.2f\n", std::log2(prev / err[0]));
        check(err[0] < (ny == 16 ? 5e-3 : 1.5e-3), std::format("ny {}: Poiseuille across gravity", ny));
        prev = err[0];
    }
}

// P5 two reactants + LES: Z == the carried tracer under the eddy species flux (one limiter,
// one diffusivity for every species), a sheared periodic layer, FP32; the eddy flux's
// effect: the Z variance falls faster than DNS's.
void mixing()
{
    std::printf("== P5 two reactants + LES: sheared layer, 96 x 96 periodic, FP32\n");
    const Grid gr = Grid::make(96, 96, 1, 1.0, 1.0, 1.0 / 96);
    const FaceMesh fm = orDie(FaceMesh::fromCurvMesh(CurvMesh::sinusoidal(gr, 0.02), periodicSpec(gr), CellOrder::Morton), "mix");
    const Transport tr = Transport::constant(kGamma, 2e-4);
    struct Out
    {
        double dz = 0, var = 0;
        long steps = 0;
    };
    auto run = [&](double cs) {
        FvCompConfig c{.tr = tr, .bc = bcFor(fm, [](const MeshPatch&) { return wall(); }), .precision = FvPrecision::Fp32,
                       .csLes = cs, .chem = fvm::kC1};
        auto d = orDie(DeviceFvComp::make(fm, c), "mix");
        FvState s = FvState::zeros(fm.nC, kFvMeanVars + 3);
        for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
            const double x = fm.C[0][q], y = fm.C[1][q];
            // Two periodic sheets at y 1/4, 3/4, each where the shear peaks.
            const double tp = 2 * std::numbers::pi;
            const double u = 0.3 * std::cos(tp * y), v = 0.03 * std::sin(tp * x) * std::sin(2 * tp * y);
            const double zf = 0.5 * (1.0 + std::tanh(std::cos(tp * y) / 0.15)), yf = 0.9 * zf, yo = 0.9 * (1.0 - zf);
            setPrim(s, q, 1.0, std::array<double, 3>{u, v, 0.0}.data(), 1.0 / kGamma);
            s.var(4)[q] += 0.5 / (kGamma - 1.0) * (1.0 - std::abs(2.0 * zf - 1.0)); // hot sheet: it burns
            s.var(5)[q] = yf;
            s.var(6)[q] = yo;
            s.var(7)[q] = zOf(yf, yo);
        }
        orDie(d.upload(s), "mix");
        Out o;
        const double tEnd = 2.0;
        double t = 0.0, dt = 0.0;
        while (t < tEnd - 1e-12) {
            if (o.steps % 16 == 0)
                dt = 0.9 * std::min(orDie(d.stableDt({.cfl = 0.4}), "dt"), orDie(d.chemDt(0.2), "chem"));
            const double h = std::min(dt, tEnd - t);
            d.step(h);
            t += h;
            ++o.steps;
        }
        orDie(d.download(s), "mix");
        double zm = 0.0, vol = 0.0;
        for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
            const double rho = s.var(0)[q], z = zOf(s.var(5)[q] / rho, s.var(6)[q] / rho);
            o.dz = std::max(o.dz, std::abs(z - s.var(7)[q] / rho));
            zm += fm.V[q] * z;
            vol += fm.V[q];
        }
        zm /= vol;
        for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
            const double rho = s.var(0)[q], z = zOf(s.var(5)[q] / rho, s.var(6)[q] / rho);
            o.var += fm.V[q] * (z - zm) * (z - zm) / vol;
        }
        return o;
    };
    const Out dnsR = run(0.0), les = run(0.17);
    const auto walk = [](long n) { return 4 * std::ldexp(1.0, -23) * std::sqrt(3.0 * double(n)); };
    std::printf("  DNS: %ld steps, max|Z(Y) - tracer| %.3e (walk %.3e), Z variance %.6f\n", dnsR.steps, dnsR.dz, walk(dnsR.steps),
                dnsR.var);
    std::printf("  LES smag 0.17: %ld steps, max|Z(Y) - tracer| %.3e (walk %.3e), Z variance %.6f (%.4f x DNS)\n", les.steps, les.dz,
                walk(les.steps), les.var, les.var / dnsR.var);
    check(dnsR.dz < walk(dnsR.steps) && les.dz < walk(les.steps), "Z(Y) equals the tracer within the FP32 walk, DNS and LES");
    check(les.var < dnsR.var, "the eddy species flux mixes: LES Z variance below DNS's");
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "physics") {
        pocket();
        poiseuille();
        mixing();
        return dnstest::finish();
    }
    std::printf("== MC generic gate matrix (docs/fvcomp.md \"Combinations\")\n");
    matrix(mode == "markdown");
    return dnstest::finish();
}
