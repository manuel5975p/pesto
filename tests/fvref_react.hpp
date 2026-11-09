// The reacting quasi-1D nozzle's host gates (masterplan_combnozzle P4, docs/nozzle.md "Reacting"; ctest fvref_react):
//   R1  fvReactingNozzleDesign vs tools/fv/nozzle_react.py's twin: every number <= 1e-12 relative (gamma 1.4, 1.2)
//   R2  the T0b convention: T0b == T0u + (sigma - 1) Tu (q on the static Tu) to 1e-12; control: A3's sigma T0u >= 1e-4 off
//   R3  the one-row react mesh (nozzle_react.py's rig, 15 cells / delta_th): nBin == nx, the chamber bins == nx / 11,
//       A_ref == the chamber's eps_c A* to 1e-12, every chamber slab constant-area to 1e-12; off a [nozzle] run the
//       probe keeps vol/L and every bin (the shipped probe)
//   R4  the IC: fresh inlet (rho 1, u = u_u of the design), Y in [0, 1], the nozzle half isentropic at (p0b, T0b)
//       (T0 and s per cell to 1e-12), M < 1 to the throat slab and > 1 past it
#pragma once

#include "fvref_nozzle.hpp"
#include "check.hpp"

#include "dns/chem.hpp"
#include "dns/flame1d.hpp"
#include "dns/fvflame.hpp"
#include "dns/fvnozzle.hpp"

#include <cmath>
#include <format>
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
}

inline int run()
{
    designGates();
    meshGates();
    return dnstest::finish();
}

} // namespace dnstest::react
