// The composition harness' cell -> configuration map (docs/fvcomp.md "Combinations",
// dns/fvmodules): one FvConfig per module set, each module at fixed test strengths, so
// the host gate (test_fv_comp) and the GPU matrix (test_gpu_fv_matrix) run the SAME
// cells. Host-only.
#pragma once

#include "dns/facemesh.hpp"
#include "dns/fvmodules.hpp"
#include "dns/fvref.hpp"
#include "dns/turb_ic.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dnstest::fvm {

using namespace dns;

inline bool has(const FvModuleSet& s, FvModule m) { return s.test(std::size_t(m)); }

// Module strengths. Premixed: a hot non-stiff step (omega ~ 0.6 rho Y, test_gpu_fv_flame_les's
// kChem); two reactants: F + 2 O, a passive tracer row 2 (Z == tracer).
inline constexpr FvChem kC0{.nY = 2, .fuel = 0, .lnA = 1.6, .Ta = 2.0, .q = 0.3, .invPr = 1.0 / 0.72};
inline constexpr FvChem kC1{.nY = 3, .fuel = 0, .lnA = 1.0, .Ta = 2.0, .q = 0.3, .invPr = 1.0 / 0.72, .ox = 1, .stoich = 2.0};
// C2b: kC1 with an A(xi) law over the cell states' xi range (s 2, Y_O2 1: xi == Z of
// zOf), spline terms on (any second derivatives are a law; the oracle evaluates the same).
inline FvChem c2b()
{
    FvChem c = kC1;
    PhiRateLaw<double>& L = c.law;
    L.n = 5;
    L.s = kC1.stoich;
    L.yO2 = 1.0;
    L.invDen = 1.0 / (L.s + L.yO2);
    constexpr double xi[] = {0.05, 0.25, 0.45, 0.65, 0.85}, lnC[] = {1.2, 0.6, 0.0, -0.4, -0.7}, m2[] = {0, 3, -2, 1, 0};
    for (int k = 0; k < L.n; ++k) {
        L.xi[k] = xi[k];
        L.lnC[k] = lnC[k];
        L.m2[k] = m2[k];
    }
    return c;
}
inline constexpr double kGy = -0.4;
inline constexpr double kForceX = 0.05;
inline constexpr double kCsSmag = 0.3, kCsSigma = 1.35;

inline TflesSetup tfles(bool uniform, bool charlette, double F = 3.0)
{
    return {.F = F,
            .uniform = uniform,
            .charlette = charlette,
            .deltaL = 0.2,
            .nuU = 0.05,
            .lnAs = 0.5,
            .TaS = 1.0,
            .sensorBeta = 2.0,
            .dilate = 3};
}

// Distance from each cell centre to the nearest Wall-kind boundary face centre (1 if
// none): an input of the closure, a function of position only, so every cell order
// gets the same numbers.
inline std::vector<double> wallDistance(const FaceMesh& fm, const std::vector<FvBc>& bc)
{
    std::vector<std::size_t> walls;
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        if (bc[p].kind == FvBcKind::Wall)
            for (int i = 0; i < fm.patches[p].count; ++i) walls.push_back(std::size_t(fm.nI + fm.patches[p].first + i));
    std::vector<double> d(std::size_t(fm.nC), 1.0);
    if (walls.empty()) return d;
    for (std::size_t c = 0; c < d.size(); ++c) {
        double m = std::numeric_limits<double>::infinity();
        for (const std::size_t f : walls)
            m = std::min(m, std::hypot(fm.C[0][c] - fm.Cf[0][f], fm.C[1][c] - fm.Cf[1][f], fm.C[2][c] - fm.Cf[2][f]));
        d[c] = std::max(m, 1e-3);
    }
    return d;
}

// The matrix' hydrostatic column: p0 1 at the origin, isothermal T0 = 1/R.
inline FvAmbient ambient(const Transport& tr) { return {.p0 = 1.0, .T0 = 1.0 / tr.Rgas}; }

struct CellKnobs
{
    bool passive = false; // species: no reaction, no heat (lnA -inf, q 0)
    double tflesF = 3.0;
    bool ransZero = false; // freestream scalars 0 (the laminar limit)
    double turbAmp = 0.05; // Turb: u' of the rows (0: the inert setting)
};

// Turb (M5): every face of a Far or Inflow patch gets its own row (bcRow), its patch's
// row plus the turb_ic box at the face centre (u' amp, l_t = the mesh's shortest
// extent / 6, an 8^3 lattice over the bounding box; w 0 on a 2D mesh): Far m += rho du,
// E += rho (u0.du + du.du/2); Inflow uIn += du. amp 0: every row equals its patch's bit
// for bit. Position only: every order gets the same rows.
inline void turbRows(const FaceMesh& fm, FvConfig& c, double amp)
{
    std::array<double, 3> lo, hi;
    for (std::size_t a = 0; a < 3; ++a) {
        const auto [l, h] = std::ranges::minmax(fm.Cf[a]);
        lo[a] = l;
        hi[a] = h;
    }
    TurbSpec ts;
    ts.urms = amp > 0 ? amp : 1.0;
    ts.Lx = std::max(hi[0] - lo[0], 1e-3);
    ts.Ly = std::max(hi[1] - lo[1], 1e-3);
    ts.Lz = std::max(hi[2] - lo[2], 1e-3);
    ts.lt = std::min({ts.Lx, ts.Ly, fm.emptyAxis == 2 ? ts.Ly : ts.Lz}) / 6.0;
    ts.nx = ts.ny = ts.nz = 8;
    ts.seed = 11;
    const std::vector<TurbMode> modes = turbModes(ts);
    c.bcRow.resize(std::size_t(fm.nB));
    for (std::size_t p = 0; p < fm.patches.size(); ++p)
        for (int i = 0; i < fm.patches[p].count; ++i) c.bcRow[std::size_t(fm.patches[p].first + i)] = int(p);
    const std::size_t nPatch = c.bc.size();
    for (std::size_t p = 0; p < nPatch; ++p) {
        const FvBc base = c.bc[p];
        if (base.kind != FvBcKind::Far && base.kind != FvBcKind::Inflow) continue;
        for (int i = 0; i < fm.patches[p].count; ++i) {
            const int b = fm.patches[p].first + i;
            const std::size_t f = std::size_t(fm.nI + b);
            std::array<double, 3> du{};
            if (amp > 0) {
                du = turbPoint(modes, ts, fm.Cf[0][f] - lo[0], fm.Cf[1][f] - lo[1], fm.Cf[2][f] - lo[2]);
                if (fm.emptyAxis >= 0) du[std::size_t(fm.emptyAxis)] = 0.0;
            }
            FvBc r = base;
            if (base.kind == FvBcKind::Far) {
                const double rho = base.qFar[0];
                double work = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    work += base.qFar[1 + k] * du[k] + 0.5 * rho * du[k] * du[k];
                    r.qFar[1 + k] = base.qFar[1 + k] + rho * du[k];
                }
                r.qFar[4] = base.qFar[4] + work;
            } else {
                for (std::size_t k = 0; k < 3; ++k) r.uIn[k] = base.uIn[k] + du[k];
            }
            c.bcRow[std::size_t(b)] = int(c.bc.size());
            c.bc.push_back(r);
        }
    }
}

// Pre: s is a closure (fvModulesClosure); its exclusive axes hold one choice.
inline FvConfig matrixConfig(const FvModuleSet& s, const FaceMesh& fm, const std::vector<FvBc>& bc, const Transport& tr,
                             CellKnobs k = {})
{
    FvConfig c{.tr = tr, .bc = bc};
    if (has(s, FvModule::ChemC0)) c.chem = kC0;
    if (has(s, FvModule::ChemC1)) c.chem = kC1;
    if (has(s, FvModule::ChemC2b)) c.chem = c2b();
    if (k.passive && c.chem.nY > 0) {
        c.chem.lnA = -std::numeric_limits<double>::infinity();
        c.chem.q = 0.0;
    }
    if (has(s, FvModule::LesSmag)) c.les = {.cs = kCsSmag, .sgs = SgsModel::Smagorinsky};
    if (has(s, FvModule::LesSigma)) c.les = {.cs = kCsSigma, .sgs = SgsModel::Sigma};
    const bool eff = has(s, FvModule::TflesEfficiency);
    if (has(s, FvModule::TflesUniform)) c.tfles = tfles(true, eff, k.tflesF);
    if (has(s, FvModule::TflesDynamic) || (eff && !c.tfles)) c.tfles = tfles(false, eff, k.tflesF);
    if (has(s, FvModule::RansSa) || has(s, FvModule::RansSst)) {
        const bool sst = has(s, FvModule::RansSst);
        c.rans.model = sst ? RansModel::Sst2003m : RansModel::SaNeg;
        c.rans.wallDist = wallDistance(fm, bc);
        c.rans.phiInf = k.ransZero ? std::array<double, 2>{0.0, sst ? 1.0 : 0.0}
                                   : (sst ? std::array<double, 2>{1e-3, 1.0} : std::array<double, 2>{3.0 * tr.muRef, 0.0});
    }
    if (has(s, FvModule::Gravity)) {
        c.gravity = {0.0, kGy, 0.0};
        // Open patches: the hydrostatic ambient's rows (p0 1 at the origin, T0 = 1/R: restState's
        // column), docs/fvcomp.md "Hydrostatic ambient".
        const bool open = std::ranges::any_of(bc, [](const FvBc& b) {
            return b.kind == FvBcKind::Far || b.kind == FvBcKind::Outflow;
        });
        if (open) {
            c.ambient = ambient(tr);
            FvBcTable t = fvAmbientRows(fm, {.bc = bc, .faceRow = {}}, *c.ambient, c.gravity, tr.gamma, tr.Rgas);
            c.bc = std::move(t.bc);
            c.bcRow = std::move(t.faceRow);
        }
    }
    if (has(s, FvModule::BodyForce)) c.bodyForce = {kForceX, 0.0, 0.0};
    if (has(s, FvModule::Turb)) {
        c.turb = true;
        turbRows(fm, c, k.turbAmp);
    }
    if (has(s, FvModule::Shock)) c.shock = {.mode = ShockMode::Hll, .eps4 = kFvShockEps4};
    return c;
}

// Every harness cell: each module alone and each pair, closed under the requirements,
// precision and the steady axis' device modules included as cells of their own only
// where they are not every cell's (precision runs inside every cell). Deterministic
// (ascending bit pattern), deduplicated; `live` keeps the sets the inventory passes.
inline std::vector<FvModuleSet> matrixCells(bool live)
{
    std::vector<FvModuleSet> v;
    constexpr std::size_t nModel = std::size_t(FvModule::Compensated); // the precision axis runs inside each cell
    auto add = [&](FvModuleSet s) {
        s = fvModulesClosure(s);
        if (fvModulesCheck(s).has_value() != live) return;
        if (std::ranges::find(v, s) == v.end()) v.push_back(s);
    };
    add({});
    for (std::size_t a = 0; a < nModel; ++a) {
        FvModuleSet one;
        one.set(a);
        add(one);
        for (std::size_t b = a + 1; b < nModel; ++b) {
            FvModuleSet two = one;
            two.set(b);
            add(two);
        }
    }
    FvModuleSet shock; // past the precision axis in the enum, a model module
    shock.set(std::size_t(FvModule::Shock));
    add(shock);
    for (std::size_t a = 0; a < nModel; ++a) { // its pairs (live: C0, masterplan_combnozzle P3)
        FvModuleSet two = shock;
        two.set(a);
        add(two);
    }
    std::ranges::sort(v, {}, [](const FvModuleSet& s) { return s.to_ullong(); });
    return v;
}

} // namespace dnstest::fvm
