// The thrust ledger's host gates (masterplan_combnozzle P2, docs/nozzle.md "Thrust"): ctest fvref_ledger.
//   G2.1  sum_c V R(c) + sum_bnd F = sources, every row, random states on six meshes; control: a patch dropped.
//   L1    FvRef's ledger over a march: P(t) - P(0) + sum I = q Omega [E] - Omega [fuel] to rounding, inert and
//         reacting, on the one-row nozzle.
//   L2    roles, the one-row exit area, the flux inversion fvExitMach, the burnt station fvStation.
// Host only. Included by tests/test_fvref.cpp.
#pragma once

#include "check.hpp"
#include "fvref_nozzle.hpp"

#include "dns/facemesh.hpp"
#include "dns/fvnozzle.hpp"
#include "dns/fvref.hpp"

#include <cmath>
#include <cstdio>
#include <format>
#include <random>
#include <string>
#include <vector>

namespace dnstest::led {

using namespace dns;
using dnstest::check;
using dnstest::noz::must;
namespace fs = std::filesystem;

inline constexpr double kG = 1.4;

// A BC row per patch by mesh type and name: empty, wedge (Slip side), symmetry (Slip), wall (no-slip
// Wall), inlet Total, outlet riemann Outflow, any other patch Far.
inline std::vector<FvBc> anyBc(const FaceMesh& fm)
{
    std::vector<FvBc> v;
    for (const MeshPatch& m : fm.patches) {
        FvBc b;
        switch (m.type) {
        case FoamPatchType::Empty: b.kind = FvBcKind::Empty; break;
        case FoamPatchType::Wedge:
            b.kind = FvBcKind::Slip;
            b.wedge = true;
            break;
        case FoamPatchType::Symmetry:
        case FoamPatchType::SymmetryPlane: b.kind = FvBcKind::Slip; break;
        case FoamPatchType::Wall: b.kind = FvBcKind::Wall; break;
        default:
            if (m.name == "inlet") {
                b.kind = FvBcKind::Total;
                b.pTotal = 1.0 / kG;
                b.tTotal = 1.0 / kG;
            } else if (m.name == "outlet") {
                b.kind = FvBcKind::Outflow;
                b.exit = FvExit::Riemann;
                b.pOut = 0.5 / kG;
                b.tBack = 1.0 / kG;
            } else {
                b.kind = FvBcKind::Far;
                const double rho = 1.0, u = 0.3, p = 1.0 / kG;
                b.qFar[0] = rho;
                b.qFar[1] = rho * u;
                b.qFar[4] = p / (kG - 1.0) + 0.5 * rho * u * u;
            }
        }
        v.push_back(b);
    }
    return v;
}

// rho, u, p random per cell (w 0 on a 2D mesh), Y in [0, 1] on species rows.
inline FvState randomState(const FaceMesh& fm, int nV, unsigned seed)
{
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<double> r(0.5, 1.5), u(-0.3, 0.3), y(0.0, 1.0);
    FvState s = FvState::zeros(fm.nC, nV);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const double rho = r(gen), U[3] = {u(gen), u(gen), fm.dim == 2 ? 0.0 : u(gen)}, p = r(gen) / kG;
        s.var(0)[c] = rho;
        for (int k = 0; k < 3; ++k) s.var(1 + k)[c] = rho * U[k];
        s.var(4)[c] = p / (kG - 1.0) + 0.5 * rho * (U[0] * U[0] + U[1] * U[1] + U[2] * U[2]);
        for (int v = kFvMeanVars; v < nV; ++v) s.var(v)[c] = rho * y(gen);
    }
    return s;
}

struct Mesh
{
    std::string name;
    FaceMesh fm;
};

inline FaceMesh polyFm(const fs::path& d, bool wedge = false)
{
    const PolyMesh pm = must(readPolyMesh(d / "constant/polyMesh"), d.string());
    return must(FaceMesh::fromPolyMesh(pm, CellOrder::Morton, 0, {.wedge = wedge}), d.string());
}

inline std::vector<Mesh> meshes()
{
    const fs::path m = fs::path(DNS_MESH_DATA), fv = m / ".." / "fv";
    std::vector<Mesh> v;
    v.push_back({"box_cyclic", polyFm(m / "box_cyclic")});
    v.push_back({"box_refined", polyFm(m / "box_refined")});
    v.push_back({"cyl2d_snappy", polyFm(m / "cyl2d_snappy")});
    v.push_back({"nozzle_A", polyFm(fv / "nozzle_A")});
    v.push_back({"nozzle_W", polyFm(fv / "nozzle_W", true)});
    v.push_back({"q1d 200", must(FaceMesh::fromPolyMesh(noz::q1dParabola(200), CellOrder::Morton), "q1d")});
    return v;
}

// C0, slow: omega ~ 0.05 at T ~ 1/gamma (400 steps burn ~10% without a runaway).
inline FvChem chemC0() { return {.nY = 1, .fuel = 0, .lnA = std::log(2.0), .Ta = 2.0, .q = 3.0, .invPr = 1.0 / 0.72}; }

// G2.1: the gather telescopes. Returns the worst relative residual; ctrl: the same with the first
// Wall / non-wedge Slip patch's faces left out of the boundary sum.
inline void telescope(const Mesh& M, bool species)
{
    FvConfig c;
    c.tr = Transport::air(1e-2, 1.0, 0.0, 0.72, kG, 1.0, false);
    c.bc = anyBc(M.fm);
    if (species) {
        c.chem = chemC0();
        for (FvBc& b : c.bc) b.y[0] = 1.0;
    }
    const FvRef ref = must(FvRef::make(M.fm, c), M.name);
    const int nV = ref.nV();
    const FvState s = randomState(M.fm, nV, 17);
    FvState out;
    FvEval ev;
    ref.evaluate(s, ev, out);
    const FaceMesh& fm = M.fm;
    int drop = -1;
    for (std::size_t p = 0; p < fm.patches.size() && drop < 0; ++p)
        if (c.bc[p].kind == FvBcKind::Wall || (c.bc[p].kind == FvBcKind::Slip && !c.bc[p].wedge)) drop = int(p);
    // An all-cyclic mesh has no boundary: its control leaves cell 0 out of the cell sum.
    const bool cellCtrl = drop < 0;
    double worst = 0.0, ctrl = 0.0;
    for (int v = 0; v < nV; ++v) {
        const std::vector<double>& F = v < kFvMeanVars ? ev.F[std::size_t(v)] : ev.Fy[std::size_t(v - kFvMeanVars)];
        double cells = 0.0, bnd = 0.0, dropped = 0.0, scale = 0.0, src = 0.0;
        for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
            cells += fm.V[q] * out.var(v)[q];
            if (species && v == 4) src += c.chem.q * fm.V[q] * ev.omega[q];
            if (species && v == kFvMeanVars) src -= fm.V[q] * ev.omega[q];
        }
        for (std::size_t f = 0; f < std::size_t(fm.nI); ++f) scale += std::abs(F[f]);
        for (int b = 0; b < fm.nB; ++b) {
            const std::size_t f = std::size_t(fm.nI + b);
            if (!ref.faceGeom().active[f]) continue;
            scale += std::abs(F[f]);
            bnd += F[f];
            const bool inDrop = drop >= 0 && b >= fm.patches[std::size_t(drop)].first &&
                                b < fm.patches[std::size_t(drop)].first + fm.patches[std::size_t(drop)].count;
            if (!inDrop) dropped += F[f];
        }
        worst = std::isfinite(cells + bnd - src) ? std::max(worst, std::abs(cells + bnd - src) / scale) : HUGE_VAL;
        const double c0 = cellCtrl ? fm.V[0] * out.var(v)[0] : 0.0;
        ctrl = std::max(ctrl, std::abs(cells - c0 + dropped - src) / scale);
    }
    const std::string what = cellCtrl ? std::string("cell 0 dropped (no boundary)") : "patch `" + fm.patches[std::size_t(drop)].name + "` dropped";
    std::printf("  %-13s%s %6d cells, %d rows: max |sum V R + sum F - src| / sum |F| %.2e; control (%s) %.2e\n", M.name.c_str(),
                species ? " +C0" : "    ", fm.nC, nV, worst, what.c_str(), ctrl);
    check(worst <= 1e-14, std::format("G2.1 {}{}: {:.2e} <= 1e-14", M.name, species ? " C0" : "", worst));
    check(ctrl >= 1e-6, std::format("G2.1 {} control ({}): {:.2e} >= 1e-6", M.name, what, ctrl));
}

// The ledger identity over an FvRef march on the one-row parabola (Total in, riemann out, slip wall,
// symmetry axis), from rest at half the inlet's p0: the inflow transient.
inline void marchIdentity(bool species)
{
    const FaceMesh fm = must(FaceMesh::fromPolyMesh(noz::q1dParabola(200), CellOrder::Morton), "q1d");
    FvConfig c;
    c.tr = Transport::air(0.0, 1.0, 0.0, 0.72, kG, 1.0, false);
    c.bc = anyBc(fm);
    for (FvBc& b : c.bc)
        if (b.kind == FvBcKind::Wall) b.kind = FvBcKind::Slip;
    if (species) {
        c.chem = chemC0();
        for (FvBc& b : c.bc) b.y[0] = 1.0;
    }
    // The inert march under --shock hll (the face pass in the ledger); species ride KG alone (Shock x C0 is P3's).
    if (!species) c.shock = {.mode = ShockMode::Hll, .eps4 = kFvShockEps4};
    const FvRef ref = must(FvRef::make(fm, c), "q1d");
    const int nV = ref.nV(), fuel = species ? kFvMeanVars : -1;
    // Inert: rest at half the inlet's p0 (a strong start, under hll). C0 rides KG alone: a weak start at
    // 0.9 p0 against an exit at 0.85 p0.
    const double p0 = species ? 0.9 : 0.5;
    if (species)
        for (FvBc& b : c.bc)
            if (b.kind == FvBcKind::Outflow) b.pOut = 0.85 / kG;
    FvState s = FvState::zeros(fm.nC, nV);
    for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
        s.var(0)[q] = p0;
        s.var(4)[q] = p0 / kG / (kG - 1.0);
        if (species) s.var(kFvMeanVars)[q] = 0.5 * p0;
    }
    const std::vector<FvRole> roles = fvRoles(c.bc, fm);
    const FvRow6 P0 = fvTotals6(fm, s, fuel);
    FvLedger led = FvLedger::zeros(int(fm.patches.size()), kFvMeanVars + c.chem.nY);
    // dt per step (the ledger takes any dt sequence): the half stable bound and the chemistry's.
    for (int n = 0; n < 400; ++n) ref.march(s, std::min(0.5 * must(ref.stableDt(s), "dt"), ref.chemDt(s, 0.5)), 1, nullptr, &led);
    const FvRow6 P = fvTotals6(fm, s, fuel);
    const FvRoleSums rs = fvRoleSums(led, roles, fuel);
    double worst = 0.0;
    for (std::size_t k = 0; k < 6; ++k) {
        if (k == 2 || k == 3 || (k == 5 && !species)) continue;
        double res = P[k] - P0[k], S = 0.0;
        for (const FvRow6& r : rs.I) {
            res += r[k];
            S += std::abs(r[k]);
        }
        if (k == 4) res -= c.chem.q * rs.omega;
        if (k == 5) res += rs.omega;
        const int row = k == 5 ? kFvMeanVars : int(k);
        for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) S += fm.V[q] * std::abs(s.var(row)[q]);
        worst = std::isfinite(res / S) ? std::max(worst, std::abs(res) / S) : HUGE_VAL;
    }
    const double moved = std::abs(P[0] - P0[0]) / P0[0];
    std::printf("  FvRef march 400 steps%s: max |res| / S %.2e; mass moved %.3f, exit mass %.4g, Omega %.4g\n", species ? " +C0" : "",
                worst, moved, rs.I[std::size_t(FvRole::Exit)][0], rs.omega);
    check(moved >= (species ? 0.005 : 0.05), std::format("L1: the march moved the mass {:.3g} (a live transient)", moved));
    check(worst <= 1e-12, std::format("L1{}: ledger identity {:.2e} <= 1e-12 of S", species ? " C0" : "", worst));
    if (species) check(rs.omega > 1e-3, "L1 C0: the reaction ran (Omega > 0)");
}

// Roles, exit area, fvExitMach, fvStation.
inline void units()
{
    const FaceMesh fm = must(FaceMesh::fromPolyMesh(noz::q1dParabola(200), CellOrder::Morton), "q1d");
    const std::vector<FvBc> bc = anyBc(fm);
    const std::vector<FvRole> r = fvRoles(bc, fm);
    const FvRole want[] = {FvRole::In, FvRole::Exit, FvRole::Wall, FvRole::Side, FvRole::None, FvRole::None};
    check(std::ranges::equal(r, want), "L2 roles: inlet in, outlet exit, wall wall, axis side, empty none");
    const FaceMesh w = polyFm(fs::path(DNS_MESH_DATA) / ".." / "fv" / "nozzle_W", true);
    const std::vector<FvRole> rw = fvRoles(anyBc(w), w);
    long sides = std::ranges::count(rw, FvRole::Side);
    check(sides == 2, std::format("L2 roles: the wedge's two sides are side ({})", sides));
    const double ae = fvExitArea(fm, r, {1, 0, 0});
    std::printf("  exit area %.17g (y_e depth = 0.14)\n", ae);
    check(std::abs(ae - 0.14) <= 1e-15, "L2 exit area = y(4) x depth");
    // fvExitMach: a state's three fluxes give its Mach back, both branches.
    double worst = 0.0;
    for (double M : {0.2, 0.7, 1.3, 2.5, 4.0}) {
        const double rho = 0.8, p = 0.6, c = std::sqrt(kG * p / rho), u = M * c;
        const double f1 = rho * u, f2 = rho * u * u + p, f3 = u * (p / (kG - 1.0) + 0.5 * rho * u * u + p);
        worst = std::max(worst, std::abs(fvExitMach(f1, f2, f3, kG, M > 1.0) - M) / M);
    }
    std::printf("  fvExitMach round trip %.2e\n", worst);
    check(worst <= 1e-12, "L2 fvExitMach: Mach from the fluxes, both roots, 1e-12");
    // fvStation: a uniform stream: p0, T0 exact, one face crossing on one row.
    FvState s = FvState::zeros(fm.nC);
    const double rho = 1.2, u = 0.4, p = 0.9, R = 1.0;
    for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) {
        s.var(0)[q] = rho;
        s.var(1)[q] = rho * u;
        s.var(4)[q] = p / (kG - 1.0) + 0.5 * rho * u * u;
    }
    const FvStation st = fvStation(fm, s, {1, 0, 0}, 0.013, kG, R);
    const double T = p / (rho * R), T0 = T + u * u / (2.0 * kG * R / (kG - 1.0)), p0 = p * std::pow(T0 / T, kG / (kG - 1.0));
    std::printf("  fvStation: %d face(s), p0 %.3e T0 %.3e rel\n", st.faces, st.p0 / p0 - 1, st.T0 / T0 - 1);
    check(st.faces == 1 && std::abs(st.p0 / p0 - 1) <= 1e-14 && std::abs(st.T0 / T0 - 1) <= 1e-14, "L2 fvStation: uniform stream exact");
}

// L3: R1's geometric premise of v == 0.0 on one row: S_y exactly 0 on the x faces (i lines vertical),
// the wall's S_y and the axis's exactly opposite per cell. Printed by face.
inline void oneRowGeometry()
{
    for (int nx : {200, 400, 800}) {
        const FaceMesh fm = must(FaceMesh::fromPolyMesh(noz::q1dParabola(nx), CellOrder::Morton), "q1d");
        double sy = 0, syIo = 0, pair = 0;
        long nonzero = 0;
        for (int f = 0; f < fm.nI; ++f) {
            sy = std::max(sy, std::abs(fm.S[1][std::size_t(f)]));
            nonzero += fm.S[1][std::size_t(f)] != 0.0;
        }
        std::vector<double> sum(std::size_t(fm.nC), 0.0);
        for (const MeshPatch& m : fm.patches) {
            for (int b = m.first; b < m.first + m.count; ++b) {
                const std::size_t f = std::size_t(fm.nI + b);
                if (m.name == "inlet" || m.name == "outlet") syIo = std::max(syIo, std::abs(fm.S[1][f]));
                if (m.name == "wall" || m.name == "axis") sum[std::size_t(fm.own[f])] += fm.S[1][f];
            }
        }
        for (double v : sum) pair = std::max(pair, std::abs(v));
        std::printf("  q1d %d: x faces max |S_y| %.2e (%ld nonzero), inlet/outlet %.2e; max |S_y,wall + S_y,axis| per cell %.2e\n", nx, sy, nonzero, syIo, pair);
    }
}

inline int run()
{
    std::printf("G2.1 the gather telescopes, every row, random states\n");
    for (const Mesh& m : meshes()) telescope(m, false);
    telescope(meshes().back(), true);
    std::printf("L1 FvRef's ledger over a march\n");
    marchIdentity(false);
    marchIdentity(true);
    std::printf("L2 roles, exit area, flux inversion, station\n");
    units();
    std::printf("L3 the one-row geometry (R1's premise of v == 0.0)\n");
    oneRowGeometry();
    return dnstest::finish();
}

} // namespace dnstest::led
