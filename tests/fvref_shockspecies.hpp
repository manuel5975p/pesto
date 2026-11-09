// Shock x C0 host gates (masterplan_combnozzle P3, docs/nozzle.md "Reacting"): ctest fvref_shockspecies.
//   G3.2  FvRef, one row of 200 cells, Euler + a passive C0 pair (Y, 1 - Y), 200 SSP-RK3 steps at the Sigma-form
//         cfl 0.8, `force` sigma 1 and hll: a Sod tube (p 10, rho 8) with a Y step at the contact, a Mach 3 piston
//         (the post-shock state through a supersonic Far face) through a Y ramp. Y in [-1e-14, 1 + 1e-14] every
//         step; the Sod contact (Y = 1/2) within 2 cells of the exact Riemann contact.
//         Control: the structured host form D Y_f(sign m_KG) (tests/shock_ref.hpp Species::Reuse, sigma 1 on
//         every face) on the same ICs: min Y < -1e-2 on the piston (counted); the Sod step keeps it in [0, 1]
//         (min Y 0.0, reported: D and m_KG share their sign at a contact at rest in pressure).
//   G3.1 control (host mutation of the face fluxes, one forward-Euler update of a uniform Y = 1/4 on a sheared
//         state, hll eps 1/16, sensor silent): the route's F_Y = Y_f F_final[0] keeps rho Y == rho / 4 bitwise;
//         the stated mutation "D3 on rho Y itself" (Y_f F_KG[0] + l d(rho Y)) and the replacement "species on the
//         KG mass flux" (Y_f F_KG[0]) are measured. The first is dead by power-of-two arithmetic (d(rho Y) ==
//         d(rho) / 4 exactly): replaced once by the second (amendment A14).
// Host only. Included by tests/test_fvref.cpp.
#pragma once

#include "check.hpp"
#include "shock_ref.hpp"

#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvgrad.hpp"
#include "dns/fvref.hpp"
#include "dns/fvshock.hpp"
#include "dns/shockflux.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

namespace dnstest::shsp {

using namespace dns;
using dnstest::check;

inline constexpr double kG = 1.4;
inline constexpr int kN = 200, kSteps = 200;

template <class T>
T must(std::expected<T, std::string> e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL: %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    return std::move(*e);
}

// One row of n cells on [0, 1]: x ends and y sides Wall (Slip rows), z empty.
inline FaceMesh row(int n)
{
    const double h = 1.0 / n;
    const Grid g = Grid::make(n, 1, 1, 1.0, h, h);
    const StructuredSpec sp{.lo = {0, 0, 0},
                            .hi = {n - 1, 0, 0},
                            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            .type = {FoamPatchType::Wall, FoamPatchType::Wall, FoamPatchType::Wall, FoamPatchType::Wall,
                                     FoamPatchType::Empty, FoamPatchType::Empty}};
    return must(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Source), "row");
}

// The passive C0 pair: no rate, no heat, no diffusion (inviscid).
inline FvChem passive()
{
    return {.nY = 2, .fuel = 0, .lnA = -std::numeric_limits<double>::infinity(), .Ta = 0.0, .q = 0.0, .invPr = 1.0};
}

struct Prim
{
    double r, u, p, y;
};

inline void put(FvState& s, int c, const Prim& q)
{
    s.var(0)[c] = q.r;
    s.var(1)[c] = q.r * q.u;
    s.var(2)[c] = 0.0;
    s.var(3)[c] = 0.0;
    s.var(4)[c] = q.p / (kG - 1.0) + 0.5 * q.r * q.u * q.u;
    s.var(5)[c] = q.r * q.y;
    s.var(6)[c] = q.r * (1.0 - q.y);
}

struct Tube
{
    const char* name;
    Prim l, r;
    double x0;
    bool ramp; // Y = clamp((x - x0)/0.2) on the right (the piston), else the left's Y | the right's
    bool farLeft; // the left x face a Far row at the left state (supersonic inflow)
};

inline Tube sod() { return {"Sod p 10 rho 8", {1.0, 0.0, 1.0, 1.0}, {0.125, 0.0, 0.1, 0.0}, 0.5, false, false}; }

inline Tube piston()
{
    const shockref::RH a = shockref::rankine(3.0); // pre-shock rho 1, p 1 (c sqrt(1.4))
    return {"Mach 3 piston", {a.r2, a.u2, a.p2, 0.0}, {1.0, 0.0, 1.0, 0.0}, 0.3, true, true};
}

inline double yAt(const Tube& t, double x)
{
    if (!t.ramp) return x < t.x0 ? t.l.y : t.r.y;
    return std::clamp((x - t.x0) / 0.2, 0.0, 1.0);
}

struct TubeOut
{
    double yMin = HUGE_VAL, yMax = -HUGE_VAL, t = 0, contact = NAN, exact = NAN;
    long sensed = 0;
    bool ok = true;
};

// FvRef on the row: kSteps steps, dt = stableDt(cfl 0.8) every step.
inline TubeOut fvTube(const Tube& tb, bool force)
{
    const FaceMesh fm = row(kN);
    std::vector<FvBc> bc;
    for (const MeshPatch& p : fm.patches) {
        FvBc b;
        b.kind = p.type == FoamPatchType::Empty ? FvBcKind::Empty : FvBcKind::Slip;
        if (tb.farLeft && p.name == "xlo") {
            b.kind = FvBcKind::Far;
            b.qFar[0] = tb.l.r;
            b.qFar[1] = tb.l.r * tb.l.u;
            b.qFar[4] = tb.l.p / (kG - 1.0) + 0.5 * tb.l.r * tb.l.u * tb.l.u;
            b.y[0] = tb.l.y;
            b.y[1] = 1.0 - tb.l.y;
        }
        bc.push_back(b);
    }
    FvConfig cfg{.tr = Transport::constant(kG, 0.0), .bc = bc, .chem = passive()};
    cfg.shock = {.mode = ShockMode::Hll, .eps4 = kFvShockEps4, .force = force};
    const FvRef ref = must(FvRef::make(fm, cfg), "FvRef");
    FvState s = FvState::zeros(fm.nC, ref.nV());
    for (int c = 0; c < fm.nC; ++c) {
        const double x = fm.C[0][std::size_t(c)];
        Prim q = x < tb.x0 ? tb.l : tb.r;
        q.y = yAt(tb, x);
        put(s, c, q);
    }
    TubeOut o;
    auto bounds = [&] {
        for (int c = 0; c < fm.nC; ++c) {
            const double y0 = s.var(5)[c] / s.var(0)[c], y1 = s.var(6)[c] / s.var(0)[c];
            if (!std::isfinite(y0) || !std::isfinite(y1)) o.ok = false;
            o.yMin = std::min({o.yMin, y0, y1});
            o.yMax = std::max({o.yMax, y0, y1});
        }
    };
    bounds();
    for (int n = 0; n < kSteps && o.ok; ++n) {
        auto dt = ref.stableDt(s, {.cfl = 0.8});
        if (!dt) {
            o.ok = false;
            break;
        }
        if (!force) {
            FvEval ev;
            FvState r;
            ref.evaluate(s, ev, r);
            o.sensed += std::ranges::count_if(ev.sigma, [](std::uint8_t b) { return b != 0; });
        }
        ref.step(s, *dt);
        o.t += *dt;
        bounds();
    }
    if (!tb.ramp) {
        // The contact: Y = 1/2, linear between cell centres, the last crossing from the left.
        for (int c = 0; c + 1 < fm.nC; ++c) {
            const double a = s.var(5)[c] / s.var(0)[c], b = s.var(5)[c + 1] / s.var(0)[c + 1];
            if (a >= 0.5 && b < 0.5) {
                const double xa = fm.C[0][std::size_t(c)], xb = fm.C[0][std::size_t(c + 1)];
                o.contact = xa + (a - 0.5) / (a - b) * (xb - xa);
            }
        }
        shockref::Riemann ex{tb.l.r, tb.l.u, tb.l.p, tb.r.r, tb.r.u, tb.r.p};
        ex.solve();
        o.exact = tb.x0 + ex.us * o.t;
    }
    return o;
}

// The structured host form D Y_f(sign m_KG) on the same IC and end time (sigma 1 everywhere, held rows 4).
inline double reuseMinY(const Tube& tb, double tEnd)
{
    std::vector<double> r(kN), u(kN), p(kN), y(kN);
    for (int i = 0; i < kN; ++i) {
        const double x = (i + 0.5) / kN;
        const Prim q = x < tb.x0 ? tb.l : tb.r;
        r[std::size_t(i)] = q.r;
        u[std::size_t(i)] = q.u;
        p[std::size_t(i)] = q.p;
        y[std::size_t(i)] = yAt(tb, x);
    }
    shockref::State<double> U = shockref::make<double>(r, u, p, &y);
    const shockref::State<double> U0 = U;
    shockref::Cfg<double> c;
    c.h = 1.0 / kN;
    c.species = shockref::Species::Reuse;
    c.testForce = true;
    double yMin = HUGE_VAL;
    c.hook = [&](shockref::State<double>& V) {
        for (std::size_t v = 0; v < 4; ++v)
            for (int i = 0; i < 4; ++i) {
                V.q[v][std::size_t(i)] = U0.q[v][std::size_t(i)];
                V.q[v][std::size_t(kN - 1 - i)] = U0.q[v][std::size_t(kN - 1 - i)];
            }
    };
    c.mon = [&](const shockref::State<double>& V, double, long) {
        for (int i = 0; i < kN; ++i) yMin = std::min(yMin, V.q[3][std::size_t(i)] / V.q[0][std::size_t(i)]);
    };
    const shockref::Report rep = shockref::run(U, tEnd, c);
    if (!rep.ok) std::printf("    reuse %s: %s\n", tb.name, rep.why.c_str());
    return yMin;
}

inline void g32()
{
    std::printf("G3.2: FvRef one row %d cells, %d steps, cfl 0.8, passive C0 pair under force sigma 1 and hll\n", kN, kSteps);
    const double lo = -1e-14, hi = 1.0 + 1e-14;
    for (const Tube& tb : {sod(), piston()}) {
        double tEnd = 0;
        for (const bool force : {true, false}) {
            const TubeOut o = fvTube(tb, force);
            tEnd = o.t;
            std::printf("  %-15s %-5s t %.4f: Y in [%.3e, 1%+.3e]%s", tb.name, force ? "force" : "hll", o.t, o.yMin, o.yMax - 1.0,
                        force ? "" : std::format(", sensed cell-steps {}", o.sensed).c_str());
            if (!tb.ramp) std::printf(", contact %.5f vs exact %.5f (%.2f cells)", o.contact, o.exact, (o.contact - o.exact) * kN);
            std::printf("\n");
            check(o.ok, std::format("{} {}: ran {} steps", tb.name, force ? "force" : "hll", kSteps));
            check(o.yMin >= lo && o.yMax <= hi, std::format("G3.2 {} {}: Y in [-1e-14, 1 + 1e-14]", tb.name, force ? "force" : "hll"));
            if (!tb.ramp)
                check(std::abs(o.contact - o.exact) <= 2.0 / kN, std::format("G3.2 {} {}: contact within 2 cells", tb.name, force ? "force" : "hll"));
            if (!force) check(o.sensed > 0, std::format("G3.2 {} hll: the sensor fired (a live hybrid run)", tb.name));
        }
        const double m = reuseMinY(tb, tEnd);
        std::printf("  control %-15s D Y_f(sign m_KG), sigma 1: min Y %.4e\n", tb.name, m);
        // Live on the piston (the ramp meets the shock: D and m_KG differ in sign); on the Sod step the
        // reuse form stays in [0, 1] (min Y 0.0): reported, the gate's control is the piston's (A14).
        if (tb.ramp) check(m < -1e-2, std::format("G3.2 control {}: the structured reuse form leaves [0, 1] (min Y < -1e-2)", tb.name));
    }
}

// G3.1's control: uniform Y = 1/4 on a sheared periodic-x row of 32 cells (the sensor silent), one forward-Euler
// update of rho and rho Y from a gather of mutated face fluxes. Returns whether rho Y' == rho' / 4 bitwise.
inline void g31Control()
{
    std::printf("G3.1 control: uniform Y 1/4, one forward-Euler update from the face fluxes (hll eps 1/16, silent)\n");
    const int n = 32;
    const Grid g = Grid::make(n, 1, 1, 1.0, 1.0 / n, 1.0 / n);
    const StructuredSpec sp{.lo = {0, 0, 0},
                            .hi = {n - 1, 0, 0},
                            .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                            .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Wall, FoamPatchType::Wall,
                                     FoamPatchType::Empty, FoamPatchType::Empty}};
    const FaceMesh fm = must(FaceMesh::fromCurvMesh(CurvMesh::uniform(g), sp, CellOrder::Source), "periodic row");
    std::vector<FvBc> bc;
    for (const MeshPatch& p : fm.patches) {
        FvBc b;
        b.kind = p.type == FoamPatchType::Empty ? FvBcKind::Empty : FvBcKind::Slip;
        bc.push_back(b);
    }
    FvConfig on{.tr = Transport::constant(kG, 1e-3), .bc = bc, .chem = passive()}, off = on;
    on.shock = {.mode = ShockMode::Hll, .eps4 = kFvShockEps4};
    const FvRef rOn = must(FvRef::make(fm, on), "on"), rOff = must(FvRef::make(fm, off), "off");
    FvState s = FvState::zeros(fm.nC, rOn.nV());
    for (int c = 0; c < fm.nC; ++c) {
        const double x = fm.C[0][std::size_t(c)];
        const double rho = 1.0 + 0.2 * std::sin(2 * std::numbers::pi * x) + 0.05 * std::cos(6 * std::numbers::pi * x);
        put(s, c, {rho, 0.3 + 0.1 * std::cos(2 * std::numbers::pi * x), 1.0 / kG, 0.25});
    }
    FvEval eOn, eOff;
    FvState rr;
    rOn.evaluate(s, eOn, rr);
    rOff.evaluate(s, eOff, rr);
    const bool silent = std::ranges::all_of(eOn.sigma, [](std::uint8_t b) { return b == 0; });
    const FvShockLines lines = fvShockLines(fm, must(fvFaceGeom(fm, FvFaceGradRule::AlongD, 1.0), "geom"));
    const std::size_t nI = std::size_t(fm.nI), nC = std::size_t(fm.nC);
    std::vector<double> dRhoF(nI), dRyF(nI); // D3's increment on rho and on rho Y, as kFvShockFace forms it
    double d3max = 0.0;
    for (std::size_t f = 0; f < nI; ++f) {
        const int pm = lines.faceLine[2 * f], np = lines.faceLine[2 * f + 1];
        if (pm < 0 || np < 0) continue;
        const std::size_t o = std::size_t(fm.own[f]), m = std::size_t(fm.nei[f]);
        FvPrim<double> A{}, B{};
        for (std::size_t k = 0; k < 3; ++k) {
            A.u[k] = eOn.prim[1 + k][o];
            B.u[k] = eOn.prim[1 + k][m];
        }
        const double S[3] = {fm.S[0][f], fm.S[1][f], fm.S[2][f]};
        const double l = fvD4Rate(kFvShockEps4, 0.0, A, B, S);
        const auto q = [&](int v, std::size_t c) { return s.var(v)[c]; };
        dRhoF[f] = l * fvD4Row(q(0, std::size_t(pm)), q(0, o), q(0, m), q(0, std::size_t(np)));
        dRyF[f] = l * fvD4Row(q(5, std::size_t(pm)), q(5, o), q(5, m), q(5, std::size_t(np)));
        d3max = std::max(d3max, std::abs(dRhoF[f]));
    }
    // A forward-Euler update from face fluxes Fr (rho) and Fy (rho Y): every cell's gather in CSR order.
    auto exact = [&](const std::vector<double>& Fr, const std::vector<double>& Fy) {
        const double dt = 1e-3;
        for (std::size_t c = 0; c < nC; ++c) {
            double gr = 0.0, gy = 0.0;
            for (int k = fm.cellStart[c]; k < fm.cellStart[c + 1]; ++k) {
                const std::size_t f = std::size_t(fm.cellFace[std::size_t(k)]);
                const double sg = f < nI && std::size_t(fm.nei[f]) == c ? -1.0 : 1.0;
                gr += sg * Fr[f];
                gy += sg * Fy[f];
            }
            const double rho = s.var(0)[c] - dt * gr / fm.V[c], ry = s.var(5)[c] - dt * gy / fm.V[c];
            if (ry != 0.25 * rho) return false;
        }
        return true;
    };
    const std::size_t nF = std::size_t(fm.nI + fm.nB);
    std::vector<double> Fr(eOn.F[0].begin(), eOn.F[0].begin() + std::ptrdiff_t(nF)), Fy = eOn.Fy[0];
    std::vector<double> FyA(nF), FyB(nF);
    for (std::size_t f = 0; f < nF; ++f) {
        const double kg = eOff.F[0][f];
        FyB[f] = 0.25 * kg;
        FyA[f] = f < nI && dRyF[f] != 0.0 ? 0.25 * kg + dRyF[f] : 0.25 * kg;
    }
    const bool route = exact(Fr, Fy), mutA = exact(Fr, FyA), mutB = exact(Fr, FyB);
    std::printf("  silent %s, max |D3 rho| %.3e; route F_Y = Y_f F_final: %s; D3 on rho Y itself: %s; species on F_KG: %s\n",
                silent ? "yes" : "NO", d3max, route ? "bitwise" : "BROKEN", mutA ? "bitwise (dead control)" : "broken",
                mutB ? "bitwise" : "broken");
    check(silent && d3max > 1e-8, "G3.1 control rig: sensor silent, D3 live on rho");
    check(route, "G3.1: the route's species flux keeps uniform Y bitwise");
    check(mutA, "G3.1 stated control (D3 on rho Y itself) is DEAD on a power-of-two Y: recorded, replaced (A14)");
    check(!mutB, "G3.1 replacement control (species on the KG mass flux) breaks uniform Y");
}

inline int run()
{
    g32();
    g31Control();
    return dnstest::finish();
}

} // namespace dnstest::shsp
