// The reacting quasi-1D nozzle's host gates (masterplan_combnozzle P4, docs/nozzle.md "Reacting"; ctest fvref_react):
//   R1  fvReactingNozzleDesign vs tools/fv/nozzle_react.py's twin: every number <= 1e-12 relative (gamma 1.4, 1.2)
//   R2  the T0b convention: T0b == T0u + (sigma - 1) Tu (q on the static Tu) to 1e-12; control: A3's sigma T0u >= 1e-4 off
//   R3  the one-row react mesh (nozzle_react.py's rig, 15 cells / delta_th): nBin == nx, the chamber bins == nx / 11,
//       A_ref == the chamber's eps_c A* to 1e-12, every chamber slab constant-area to 1e-12; off a [nozzle] run the
//       probe keeps vol/L and every bin (the shipped probe)
//   R4  the IC: fresh inlet (rho 1, u = u_u of the design), Y in [0, 1], the nozzle half isentropic at (p0b, T0b)
//       (T0 and s per cell to 1e-12), M < 1 to the throat slab and > 1 past it
//   R5  the chamber on the Rayleigh line (masterplan_combclose 2.1): fvRayleighBurnt at q (1 - 0) == d.b bitwise; every
//       chamber cell rho u == d.m (1e-13), p + rho u^2, cp T + u^2/2 + q Y conserved (1e-12); burnt cells at (T0b,
//       p0b), T0 continuous into the contraction (1e-12); control: premixedPoint's branch, T0 +1.71e-3 (>= 1e-3)
#pragma once

#include "fvref_nozzle.hpp"
#include "check.hpp"

#include "dns/chem.hpp"
#include "dns/flame1d.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvnozzle.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <functional>
#include <numbers>

namespace dnstest::react {

using namespace dns;
using dnstest::check;
using dnstest::noz::must;
using dnstest::noz::q1dPoly;

// nozzle_react.py inverse (target u_u = S_L,num) + design at M 0.042, eps_e 2 (the rig's): gamma, S_L,num, eps_c, p0, t0 ratios, the inverse's
// M_u, then the design's M_u, u_u, T0b, p0b, M_b, M_e, CFvac (Python's repr).
struct PyRow
{
    double g, sl, epsC, p0, t0, MuInv, Mu, uu, T0b, p0b, Mb, Me, CF;
};
inline constexpr PyRow kPy[2] = {
    {1.4, 0.9945059, 6.172901800721939, 1.0012218018140673, 1.0003489340123555, 0.041769247800000006, 0.041769247800000006,
     0.9945059, 2024.7606632703096, 403.42545723099414, 0.09424960517372154, 2.1971981216521863, 1.457600096948547},
    {1.2, 0.9982783, 6.294448557148318, 1.0010552222997162, 1.0001757931071338, 0.041927688600000006, 0.041927688599999985,
     0.9982782999999995, 2362.138980114859, 470.90270791489337, 0.09451821470649938, 2.0550955011468544, 1.465747108603498},
};
inline constexpr double kMach = 0.042, kSigma = 5.0, kEpsE = 2.0, kDth = 1.601651;
inline constexpr double kLc = 1.0, kLcon = 6.0, kLdiv = 4.0; // nozzle_react.py's rig, y* units

inline double rel(double a, double b) { return std::abs(a - b) / std::abs(b); }

inline FvReactingDesign designOf(const PyRow& r)
{
    const double Tu = 1.0 / (r.g * kMach * kMach), cp = r.g / (r.g - 1.0);
    return must(fvReactingNozzleDesign(r.g, 1.0, cp, cp * (kSigma - 1) * Tu, kSigma, r.epsC, kEpsE, r.p0 * Tu, r.t0 * Tu),
                "design");
}

inline void designGates()
{
    for (const PyRow& r : kPy) {
        const FvReactingDesign d = designOf(r);
        const double e = std::max({rel(d.Mu, r.Mu), rel(d.uu, r.uu), rel(d.b.T0b, r.T0b), rel(d.b.p0b, r.p0b), rel(d.b.Mb, r.Mb),
                                   rel(d.Me, r.Me), rel(d.CFvac, r.CF)});
        std::printf("R1 gamma %.1f: M_u %.15g u_u %.15g T0b %.15g p0b %.15g CFvac %.15g; max rel vs Python %.2e\n", r.g, d.Mu, d.uu,
                    d.b.T0b, d.b.p0b, d.CFvac, e);
        check(e <= 1e-12, std::format("R1 gamma {}: design == nozzle_react.py to 1e-12 ({:.2e})", r.g, e));
        check(rel(d.Mu, r.MuInv) <= 1e-12 && rel(d.uu, r.sl) <= 1e-12,
              std::format("R1 gamma {}: the design returns the inverse's M_u and u_u = S_L,num", r.g));
        const double Tu = 1.0 / (r.g * kMach * kMach), T0u = r.t0 * Tu;
        const double ok = rel(d.b.T0b, T0u + (kSigma - 1) * Tu), a3 = rel(kSigma * T0u, d.b.T0b);
        std::printf("R2 gamma %.1f: T0b vs T0u + (sigma - 1) Tu %.2e; A3's sigma T0u off by %.3e\n", r.g, ok, a3);
        check(ok <= 1e-12, std::format("R2 gamma {}: T0b = T0u + (sigma - 1) Tu", r.g));
        check(a3 >= 1e-4, std::format("R2 gamma {} control: sigma T0u off by >= 1e-4", r.g));
    }
    const PyRow& r = kPy[0];
    const double Tu = 1.0 / (r.g * kMach * kMach), cp = r.g / (r.g - 1.0);
    auto bad = fvReactingNozzleDesign(r.g, 1.0, cp, cp * (kSigma - 1) * Tu, kSigma, 1.0, kEpsE, r.p0 * Tu, r.t0 * Tu);
    check(!bad, "R1: eps_c 1 (no throat) is an error: " + (bad ? std::string("none") : bad.error()));
}

// The react contour (nozzle_mesh.py reactContour) at scale s.
inline double reactY(double x, double ec, double ee, double s)
{
    const double u = x / s, pi = std::numbers::pi;
    if (u <= kLc) return s * ec;
    if (u <= kLc + kLcon) return s * (1 + (ec - 1) * (1 + std::cos(pi * (u - kLc) / kLcon)) / 2);
    return s * (1 + (ee - 1) * (1 - std::cos(pi * std::min(u - kLc - kLcon, kLdiv) / kLdiv)) / 2);
}

// The chamber's errors against the design's Rayleigh line, per cell (rho, u, E of the cell): mass rho u vs d.m,
// momentum p + rho u^2 vs d.pu + d.m d.uu, energy cp T + u^2/2 + q Y vs cp d.Tu + d.uu^2/2 + d.q; on the burnt cells
// (Y <= 1e-12) T0, p0 vs d.b; T0 of the last chamber cell vs the first contraction cell.
struct IcErr
{
    double mass = 0, mom = 0, energy = 0, t0b = 0, p0b = 0, seam = 0, seamH = 0; // seamH: cp T0 + q Y across the seam
    int burnt = 0;
};

inline IcErr icErr(const FaceMesh& fm, const FvFlameProbe& p, const FvReactingDesign& d,
                   const std::function<std::array<double, 4>(int)>& cell) // c -> rho, u, E, Y
{
    const double g = d.g, cp = d.cp;
    const auto prim = [&](int c) {
        const auto [rho, u, E, Y] = cell(c);
        const double pr = (g - 1) * (E - 0.5 * rho * u * u), T = pr / (d.R * rho), T0 = T + u * u / (2 * cp);
        return std::array<double, 6>{rho, u, pr, T, T0, pr * std::pow(T0 / T, g / (g - 1))};
    };
    IcErr e;
    const int nCh = p.chamberBins();
    double t0Last = NAN, t0First = NAN, hLast = NAN;
    for (int c = 0; c < fm.nC; ++c) {
        const int b = p.binOf(c);
        const auto [rho, u, pr, T, T0, p0] = prim(c);
        const double Y = cell(c)[3];
        if (b == nCh - 1) {
            t0Last = T0;
            hLast = cp * T0 + d.q * Y;
        }
        if (b == nCh) t0First = T0;
        if (b >= nCh) continue;
        e.mass = std::max(e.mass, rel(rho * u, d.m));
        e.mom = std::max(e.mom, rel(pr + rho * u * u, d.pu + d.m * d.uu));
        e.energy = std::max(e.energy, rel(cp * T + u * u / 2 + d.q * Y, cp * d.Tu + d.uu * d.uu / 2 + d.q));
        if (Y > 1e-12) continue;
        ++e.burnt;
        e.t0b = std::max(e.t0b, std::abs(T0 / d.b.T0b - 1));
        e.p0b = std::max(e.p0b, std::abs(p0 / d.b.p0b - 1));
    }
    e.seam = std::abs(t0Last / t0First - 1);
    e.seamH = std::abs(hLast / (cp * t0First) - 1);
    return e;
}

// R5 (masterplan_combclose K2): the chamber IC on the Rayleigh line; control: the shipped premixedPoint branch.
inline void icGates(const FaceMesh& fm, const FvFlameProbe& p, const FvReactingDesign& d, const FvState& s, const PremixedEigen& eig,
                    double astar)
{
    const FvBurnt b0 = fvRayleighBurnt(d.g, d.R, d.cp, d.q * (1 - 0.0), d.Tu, d.pu, d.uu);
    check(std::memcmp(&b0, &d.b, sizeof(FvBurnt)) == 0, "R5a: fvRayleighBurnt at q (1 - 0) == the design's burnt state, bit for bit");
    check(d.g == kPy[0].g && d.R == 1.0 && d.cp == kPy[0].g / (kPy[0].g - 1.0), "R5: the design echoes gamma, R, cp");
    const auto stateOf = [](const FvState& x, int c) {
        const std::size_t i = std::size_t(c);
        const double rho = x.var(0)[i];
        return std::array<double, 4>{rho, x.var(1)[i] / rho, x.var(4)[i], x.var(kFvMeanVars)[i] / rho};
    };
    const IcErr e = icErr(fm, p, d, [&](int c) { return stateOf(s, c); });
    // The oracle table ends at sigma - 4e-9 (Y 1e-9 by the shooting's construction), so the shipped profile has no cell
    // with Y <= 1e-12: (c), (d) run on the same IC with the profile at sigma past the table's end (masterplan_combclose
    // B1), the shipped one on cp T0 + q Y across the seam.
    const double xEnd = eig.tab.x.back(), yEnd = (kSigma - eig.tab.T.back()) / (kSigma - 1);
    const auto burnt = [&](double x) { return x >= xEnd ? kSigma : flameProfile(eig.tab, x); };
    const FvState s1 = fvReactingNozzleIC(fm, p, d, burnt, 12.0 * kDth, astar);
    const IcErr e1 = icErr(fm, p, d, [&](int c) { return stateOf(s1, c); });
    // The control: the shipped chamber branch, premixedPoint(gamma, Tu, sigma, d.m, t) on the same cells (the profile
    // at sigma past the table's end), the contraction as the IC's.
    const FvFlameSpec& sp = p.spec();
    const IcErr k = icErr(fm, p, d, [&](int c) {
        if (p.binOf(c) >= p.chamberBins()) return stateOf(s1, c);
        const std::size_t i = std::size_t(c);
        const double x = fm.C[0][i] * sp.dir[0] + fm.C[1][i] * sp.dir[1] + fm.C[2][i] * sp.dir[2] - p.lo();
        const PremixedPoint q = premixedPoint(d.g, sp.Tu, sp.sigma, d.m, burnt(x - 12.0 * kDth));
        return std::array<double, 4>{q.rho, q.u, q.E, q.Y};
    });
    std::printf("R5 chamber on the Rayleigh line: mass %.2e, momentum %.2e, energy %.2e; seam cp T0 + q Y %.2e (T0 %.3e = q Y_end/(cp T0b) "
                "%.3e: the oracle table's burnt end x %.6g, Y_end %.3e)\n",
                e.mass, e.mom, e.energy, e.seamH, e.seam, d.q * yEnd / (d.cp * d.b.T0b), xEnd, yEnd);
    std::printf("R5 profile at sigma past the table: %d burnt cells T0 vs T0b %.2e, p0 vs p0b %.2e; seam T0 %.2e; energy %.2e\n", e1.burnt,
                e1.t0b, e1.p0b, e1.seam, e1.energy);
    std::printf("R5 control (premixedPoint): mass %.2e, momentum %.2e, energy %.4e; burnt T0 vs T0b %+.4e, p0 %.3e; seam %.4e\n", k.mass,
                k.mom, k.energy, k.t0b, k.p0b, k.seam);
    check(e.mass <= 1e-13 && e1.mass <= 1e-13, std::format("R5b: rho u == d.m on every chamber cell ({:.2e} <= 1e-13)", e.mass));
    check(e.mom <= 1e-12 && e.energy <= 1e-12 && e1.mom <= 1e-12 && e1.energy <= 1e-12,
          std::format("R5b: p + rho u^2 and cp T + u^2/2 + q Y conserved ({:.2e}, {:.2e} <= 1e-12)", e.mom, e.energy));
    check(e1.burnt > 0 && e1.t0b <= 1e-12 && e1.p0b <= 1e-12,
          std::format("R5c: {} burnt chamber cells at (T0b, p0b) ({:.2e}, {:.2e} <= 1e-12)", e1.burnt, e1.t0b, e1.p0b));
    check(e1.seam <= 1e-12 && e.seamH <= 1e-12,
          std::format("R5d: T0 continuous at the chamber's end ({:.2e}; shipped profile cp T0 + q Y {:.2e} <= 1e-12)", e1.seam, e.seamH));
    check(k.t0b >= 1e-3, std::format("R5 control: premixedPoint's burnt T0 off T0b by {:.4e} >= 1e-3", k.t0b));
    check(k.energy >= 1e-3, std::format("R5 control: premixedPoint's energy off the line by {:.4e} >= 1e-3", k.energy));
}

inline void meshGates()
{
    const PyRow& r = kPy[0];
    const double h = kDth / 15.0, L = kLc + kLcon + kLdiv;
    const int k = int(std::lround(L / kLc)), nx = k * int(std::lround(L * 50.0 / (k * h)));
    const double S = nx * h / L;
    const PolyMesh pm = q1dPoly(nx, [&](double x) { return reactY(x, r.epsC, kEpsE, S); }, 0.0, S * L);
    const FaceMesh fm = must(FaceMesh::fromPolyMesh(pm, CellOrder::Morton), "react mesh");
    const double Tu = 1.0 / (r.g * kMach * kMach), cp = r.g / (r.g - 1.0);
    const FvFlameSpec fl{.dir = {1, 0, 0}, .Tu = Tu, .sigma = kSigma, .dth = kDth, .gamma = r.g, .Rgas = 1.0,
                         .q = cp * (kSigma - 1) * Tu, .fuel = 0, .chamber = true};
    const FvFlameProbe p = must(FvFlameProbe::make(fm, fl), "chamber probe");
    FvFlameSpec off = fl;
    off.chamber = false;
    const FvFlameProbe q = must(FvFlameProbe::make(fm, off), "probe");
    const double astar = S * 0.1, aC = r.epsC * astar;
    double slabs = 0;
    for (int c = 0; c < fm.nC; ++c)
        if (p.binOf(c) < p.chamberBins()) slabs = std::max(slabs, rel(fm.V[std::size_t(c)] / p.slab(), p.area()));
    std::printf("R3 one row nx %d (scale %.10g): nBin %d, chamber %d slabs, A_ref %.15g vs eps_c A* %.15g (%.2e), max chamber slab "
                "|V/h - A_ref| / A_ref %.2e; vol/L %.10g (A_ref / vol/L %.6f); off: area %.10g, bins %d of %d\n",
                nx, S, p.bins(), p.chamberBins(), p.area(), aC, rel(p.area(), aC), slabs, p.meanArea(), p.area() / p.meanArea(),
                q.area(), q.chamberBins(), q.bins());
    check(p.bins() == nx, "R3: nBin == nx on the one-row mesh");
    check(p.chamberBins() == nx / k, std::format("R3: the chamber bins == nx / {} (the chamber's cells)", k));
    check(rel(p.area(), aC) <= 1e-12 && slabs <= 1e-12, "R3: A_ref == eps_c A*, every chamber slab constant-area, to 1e-12");
    check(q.area() == q.meanArea() && q.chamberBins() == q.bins() && q.length() > p.length(),
          "R3: off a [nozzle] run the probe keeps vol/L and every bin");
    // R4: the IC.
    const FvReactingDesign d = designOf(r);
    const PremixedEigen eig = must(premixedEigen({.sigma = kSigma, .beta = 8.0}), "oracle");
    const FvState s = fvReactingNozzleIC(fm, p, d, [&](double x) { return flameProfile(eig.tab, x); }, 12.0 * kDth, astar);
    double yMin = 1, yMax = 0, t0e = 0, se = 0, inl = 0;
    bool machOk = true;
    int throat = p.chamberBins();
    std::vector<double> A(std::size_t(p.bins()), 0.0);
    for (int c = 0; c < fm.nC; ++c) A[std::size_t(p.binOf(c))] += fm.V[std::size_t(c)] / p.slab();
    for (int b = p.chamberBins(); b < p.bins(); ++b)
        if (A[std::size_t(b)] < A[std::size_t(throat)]) throat = b;
    const double g = r.g, sRef = std::log(d.b.p0b) - g / (g - 1) * std::log(d.b.T0b);
    for (int c = 0; c < fm.nC; ++c) {
        const std::size_t i = std::size_t(c);
        const FvPrim<double> P = fvPrimOf(s.var(0)[i], s.var(1)[i], s.var(2)[i], s.var(3)[i], s.var(4)[i], g, 1.0);
        const double Y = s.var(kFvMeanVars)[i] / P.rho;
        yMin = std::min(yMin, Y);
        yMax = std::max(yMax, Y);
        const int b = p.binOf(c);
        if (b == 0) inl = std::max(rel(P.rho, 1.0), rel(P.u[0], d.uu));
        if (b < p.chamberBins()) continue;
        const double M = P.u[0] / std::sqrt(g * P.temp), T0 = P.temp * (1 + (g - 1) / 2 * M * M);
        t0e = std::max(t0e, rel(T0, d.b.T0b));
        se = std::max(se, std::abs(std::log(P.p) - g / (g - 1) * std::log(P.temp) - sRef));
        machOk = machOk && (b <= throat ? M < 1.0 : M > 1.0);
    }
    std::printf("R4 IC: inlet cell vs (1, u_u) %.2e; Y in [%.3g, %.3g]; nozzle half T0 vs T0b %.2e, entropy %.2e; throat slab %d of "
                "%d, M < 1 before, > 1 past: %s\n",
                inl, yMin, yMax, t0e, se, throat, p.bins(), machOk ? "yes" : "NO");
    // The oracle profile's preheat tail at 12 delta_th: T/Tu - 1 ~ 4e-7 in the inlet cell.
    check(inl <= 1e-6, "R4: the inlet cell is the design's fresh state (rho 1, u_u) to 1e-6 (the profile's tail)");
    check(yMin >= 0.0 && yMax <= 1.0, "R4: Y in [0, 1]");
    check(t0e <= 1e-12 && se <= 1e-12, "R4: the nozzle half is isentropic at (p0b, T0b) to 1e-12");
    check(machOk, "R4: subsonic to the throat slab, supersonic past it");
    icGates(fm, p, d, s, eig, astar);
}

inline int run()
{
    designGates();
    meshGates();
    return dnstest::finish();
}

} // namespace dnstest::react
