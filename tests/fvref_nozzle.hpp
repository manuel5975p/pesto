// The nozzle's host gates (docs/nozzle.md, masterplan_cdnozzle): the rows, the wedge, the T1
// residual and the hoop bound on FvRef. Included by tests/test_fvref.cpp, one mode each
// (ctest fvref_total, fvref_exit, fvref_bits, fvref_wedge, fvref_wedgew, fvref_t1r,
// fvref_hoop). Host only, no OpenFOAM: references live in tests/data/fv/nozzle_{A,B,W}.
#pragma once

#include "check.hpp"
#include "dns/facemesh.hpp"
#include "dns/foamio.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvref.hpp"
#include "dns/fvshock.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <limits>
#include <numbers>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <vector>

namespace dnstest::noz {

using namespace dns;
using dnstest::check;
namespace fs = std::filesystem;

inline constexpr double kP0 = 1e5, kT0 = 300.0, kGam = 1.4;

template <class T>
T must(std::expected<T, std::string> e, const std::string& what)
{
    if (!e) {
        std::printf("FATAL: %s: %s\n", what.c_str(), e.error().c_str());
        std::exit(1);
    }
    return std::move(*e);
}

inline fs::path noz(const char* c) { return fs::path(DNS_MESH_DATA) / ".." / "fv" / c; }

// A reference case: the mesh (Source order, so cells are foam's), foam's last write.
struct Ref
{
    PolyMesh pm;
    FaceMesh fm;
    std::vector<double> p, T, U, rho;
    double Rgas = 0;
    fs::path dir;
};

inline std::vector<double> field(const fs::path& f, int nc, int n) { return must(readFoamInternalField(f, nc, n), f.string()); }

inline fs::path lastTime(const fs::path& d)
{
    fs::path best;
    double t = -1;
    for (const auto& e : fs::directory_iterator(d / "ref")) {
        const double v = std::stod(e.path().filename().string());
        if (v > t) {
            t = v;
            best = e.path();
        }
    }
    return best;
}

inline Ref loadRef(const char* name, CellOrder order = CellOrder::Source)
{
    Ref r;
    r.dir = noz(name);
    r.pm = must(readPolyMesh(r.dir / "constant/polyMesh"), name);
    r.fm = must(FaceMesh::fromPolyMesh(r.pm, order, 0, {.wedge = true}), name);
    const fs::path t = lastTime(r.dir);
    r.p = field(t / "p", 1, r.fm.nC);
    r.T = field(t / "T", 1, r.fm.nC);
    r.U = field(t / "U", 3, r.fm.nC);
    r.rho = field(t / "rho", 1, r.fm.nC);
    std::vector<double> q(r.p.size());
    std::ranges::transform(std::views::iota(std::size_t(0), q.size()), q.begin(),
                           [&](std::size_t c) { return r.p[c] / (r.rho[c] * r.T[c]); });
    std::ranges::nth_element(q, q.begin() + std::ptrdiff_t(q.size() / 2));
    r.Rgas = q[q.size() / 2];
    return r;
}

// Conserved state of cell-order (source) fields p, T, U on fm, gas R.
inline FvState stateOf(const FaceMesh& fm, const std::vector<double>& p, const std::vector<double>& T, const std::vector<double>& U,
                       double R)
{
    FvState s = FvState::zeros(fm.nC);
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
        const std::size_t o = std::size_t(fm.srcCell[c]);
        const double rho = p[o] / (R * T[o]), u = U[3 * o], v = U[3 * o + 1], w = U[3 * o + 2];
        s.var(0)[c] = rho;
        s.var(1)[c] = rho * u;
        s.var(2)[c] = rho * v;
        s.var(3)[c] = rho * w;
        s.var(4)[c] = p[o] / (kGam - 1.0) + 0.5 * rho * (u * u + v * v + w * w);
    }
    return s;
}

struct Rows
{
    double pb = 0;      // > 0: the exit's p, else zeroGradient
    bool slip = true;   // wall
    double T0 = kT0, p0 = kP0;
    FvExit exit = FvExit::Fixed; // pb > 0
    bool characteristic = false; // pb > 0
};

// The table of a nozzle mesh by patch name.
inline std::vector<FvBc> nozzleBc(const FaceMesh& fm, const Rows& k)
{
    std::vector<FvBc> v;
    for (const MeshPatch& m : fm.patches) {
        FvBc b;
        if (m.type == FoamPatchType::Empty) b.kind = FvBcKind::Empty;
        else if (m.type == FoamPatchType::Wedge) {
            b.kind = FvBcKind::Slip;
            b.wedge = true;
        } else if (m.name == "inlet") {
            b.kind = FvBcKind::Total;
            b.pTotal = k.p0;
            b.tTotal = k.T0;
        } else if (m.name == "outlet") {
            b.kind = FvBcKind::Outflow;
            if (k.pb > 0) {
                b.pOut = k.pb;
                b.tBack = k.T0;
                b.exit = k.exit;
                b.characteristic = k.characteristic;
            } else {
                b.exit = FvExit::Extrapolate;
            }
        } else if (m.name == "wall") {
            b.kind = k.slip ? FvBcKind::Slip : FvBcKind::Wall;
        } else {
            b.kind = FvBcKind::Slip;
        }
        v.push_back(b);
    }
    return v;
}

inline FvConfig nozzleCfg(const FaceMesh& fm, double R, const Rows& k, double mu = 0.0)
{
    FvConfig c;
    c.tr = Transport::air(mu, 1.0, 0.0, 0.72, kGam, R, false);
    c.bc = nozzleBc(fm, k);
    return c;
}

// --- G1, the Total row ---------------------------------------------------------

inline int total()
{
    std::printf("G1 Total row\n");
    const double g = kGam, R = 287.1, cp = g * R / (g - 1.0);
    FvBc bc;
    bc.kind = FvBcKind::Total;
    bc.pTotal = kP0;
    bc.tTotal = kT0;
    const double S[3] = {-0.6, -0.8, 0.0}, Cf[3] = {0, 0, 0}, n[3] = {-0.6, -0.8, 0.0};
    FvPrim<double> P = fvPrimOf(1.2, 0.0, 0.0, 0.0, 5e4 / (g - 1.0), g, R);
    FvPrim<double> B = fvBoundaryState(bc, P, S, Cf, g, R);
    check(B.p == kP0 && B.temp == kT0 && B.u[0] == 0.0 && B.u[1] == 0.0 && B.u[2] == 0.0, "rest returns p0, t0 exactly");
    double worstT = 0, worstP = 0, worstF = 0;
    std::mt19937_64 gen(7);
    std::uniform_real_distribution<double> mach(1e-3, 0.95), ang(0, 2 * std::numbers::pi);
    int in = 0, outN = 0;
    for (int i = 0; i < 2000; ++i) {
        const double p = 4e4 + 5e4 * double(i) / 2000.0, T = 250.0, rho = p / (R * T), c = std::sqrt(g * R * T);
        const double m = mach(gen) * c, a = ang(gen);
        const double ux = m * std::cos(a), uy = m * std::sin(a);
        FvPrim<double> Q = fvPrimOf(rho, rho * ux, rho * uy, 0.0, p / (g - 1.0) + 0.5 * rho * m * m, g, R);
        const double un = ux * n[0] + uy * n[1];
        B = fvBoundaryState(bc, Q, S, Cf, g, R);
        if (un >= 0.0) {
            ++outN;
            worstP = std::max({worstP, std::abs(B.p - kP0) / kP0, std::abs(B.temp - kT0) / kT0, std::abs(B.u[0] - ux) / c, std::abs(B.u[1] - uy) / c});
            continue;
        }
        ++in;
        const double q2 = B.u[0] * B.u[0] + B.u[1] * B.u[1] + B.u[2] * B.u[2];
        worstT = std::max(worstT, std::abs(B.temp + q2 / (2 * cp) - kT0) / kT0);
        worstP = std::max(worstP, std::abs(B.p / (kP0 * std::pow(B.temp / kT0, g / (g - 1.0))) - 1.0));
        worstF = std::max({worstF, std::abs(B.u[0] - un * n[0]) / c, std::abs(B.u[1] - un * n[1]) / c, std::abs(B.u[2]) / c});
    }
    std::printf("  worst T0 %.2e, p %.2e, u %.2e; %d in, %d out\n", worstT, worstP, worstF, in, outN);
    check(worstT <= 1e-14, "T + u^2/(2 cp) = t0 to 1e-14");
    check(worstP <= 1e-14, "isentropic p and the outgoing branch to 1e-14");
    check(worstF <= 1e-15, "entering: the normal velocity of P");
    check(in > 500 && outN > 500, "both branches hit");
    const double gu[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, gT[3] = {1, 2, 3};
    const FvFlux<double> a = fvBoundaryFlux(bc, P, B, S, gu, gT, 1.0, 1.0), e = fvEulerFlux(B, S);
    check(a.rho == e.rho && a.m[0] == e.m[0] && a.E == e.E, "no viscous flux: the Euler flux of B");
    bc.kind = FvBcKind::Total;
    bc.y[0] = 0.25;
    check(fvBoundaryY(bc, 0, 0.5, -1.0) == 0.25 && fvBoundaryY(bc, 0, 0.5, 1.0) == 0.5, "open to species");
    return dnstest::finish();
}

// --- G2, the half-Riemann exit -------------------------------------------------

struct Exact
{
    double rho, un, p;
};

// Toro's exact sampler at x/t = 0, the right state (a / (R tb), u*, a): p* found by bisection.
inline Exact exactExit(double rho, double un, double p, double a, double tb, double g, double R)
{
    const double cL = std::sqrt(g * p / rho);
    auto fL = [&](double q) {
        if (q > p) return (q - p) * std::sqrt(2.0 / ((g + 1) * rho) / (q + (g - 1) / (g + 1) * p));
        return 2 * cL / (g - 1) * (std::pow(q / p, (g - 1) / (2 * g)) - 1);
    };
    const double uR = un - fL(a), rhoR = a / (R * tb), cR = std::sqrt(g * a / rhoR);
    auto fR = [&](double q) {
        if (q > a) return (q - a) * std::sqrt(2.0 / ((g + 1) * rhoR) / (q + (g - 1) / (g + 1) * a));
        return 2 * cR / (g - 1) * (std::pow(q / a, (g - 1) / (2 * g)) - 1);
    };
    double lo = 1e-9 * std::min(p, a), hi = 1e9 * std::max(p, a);
    for (int i = 0; i < 400; ++i) {
        const double m = std::sqrt(lo * hi);
        (fL(m) + fR(m) + uR - un > 0 ? hi : lo) = m;
    }
    const double ps = std::sqrt(lo * hi), us = 0.5 * (un + uR + fR(ps) - fL(ps));
    if (us < 0.0) return {rhoR, us, a};
    if (ps > p) {
        const double S = un - cL * std::sqrt((g + 1) / (2 * g) * ps / p + (g - 1) / (2 * g));
        if (S >= 0.0) return {rho, un, p};
        const double m = (g - 1) / (g + 1), r = ps / p;
        return {rho * (r + m) / (m * r + 1), us, ps};
    }
    if (un - cL >= 0.0) return {rho, un, p};
    const double cs = cL * std::pow(ps / p, (g - 1) / (2 * g));
    if (us - cs <= 0.0) return {rho * std::pow(ps / p, 1 / g), us, ps};
    const double k = 2 / (g + 1) + (g - 1) / ((g + 1) * cL) * un;
    return {rho * std::pow(k, 2 / (g - 1)), 2 / (g + 1) * (cL + (g - 1) / 2 * un), p * std::pow(k, 2 * g / (g - 1))};
}

inline int exitRow()
{
    std::printf("G2 half-Riemann exit vs Toro's exact sampler\n");
    const double g = kGam, R = 287.1, tb = 280.0;
    FvBc bc;
    bc.kind = FvBcKind::Outflow;
    bc.exit = FvExit::Riemann;
    bc.tBack = tb;
    const double S[3] = {0.0, 1.0, 0.0}, Cf[3] = {0, 0, 0};
    int hits[6] = {};
    double worst = 0, jump = 0;
    const double rho = 1.1, p = 1e5, c = std::sqrt(g * p / rho);
    for (double M = -5.0; M <= 5.0001; M += 0.0725) {
        for (double lr = -2.0; lr <= 2.0001; lr += 0.043) {
            const double un = M * c, a = p * std::pow(10.0, lr);
            bc.pOut = a;
            const FvPrim<double> P = fvPrimOf(rho, rho * 7.0, rho * un, 0.0, p / (g - 1.0) + 0.5 * rho * (49.0 + un * un), g, R);
            const FvPrim<double> B = fvBoundaryState(bc, P, S, Cf, g, R);
            const Exact e = exactExit(rho, un, p, a, tb, g, R);
            const double scale = std::abs(e.un) + c;
            worst = std::max({worst, std::abs(B.rho - e.rho) / e.rho, std::abs(B.p - e.p) / e.p, std::abs(B.u[1] - e.un) / scale});
            // Tangential: the owner's unless backflow.
            const bool back = e.un < 0.0 && B.u[0] == 0.0;
            if (!back) worst = std::max(worst, std::abs(B.u[0] - 7.0) / scale);
            // The branch.
            const double us = e.un;
            if (a >= p) {
                const double Ss = un - c * std::sqrt((g + 1) / (2 * g) * a / p + (g - 1) / (2 * g));
                ++hits[Ss >= 0 ? 0 : us >= 0 ? 1 : 5];
            } else {
                const double k = 2 * c / (g - 1) * (1 - std::pow(a / p, (g - 1) / (2 * g)));
                const double ust = un + k, cst = c * std::pow(a / p, (g - 1) / (2 * g));
                ++hits[un - c >= 0 ? 2 : ust < 0 ? 5 : ust - cst <= 0 ? 3 : 4];
            }
        }
    }
    // Continuity: across S = 0 (a shock that just stays out) and M 1 on the fan.
    auto at = [&](double un, double a) {
        bc.pOut = a;
        const FvPrim<double> P = fvPrimOf(rho, 0.0, rho * un, 0.0, p / (g - 1.0) + 0.5 * rho * un * un, g, R);
        return fvBoundaryState(bc, P, S, Cf, g, R);
    };
    const double aS = p * (2 * g / (g + 1) * 4.0 - (g - 1) / (g + 1)), dx = 1e-7;
    const FvPrim<double> s0 = at(2.0 * c * (1 + dx), aS), s1 = at(2.0 * c * (1 - dx), aS);
    const FvPrim<double> f0 = at(c * (1 + dx), 0.5 * p), f1 = at(c * (1 - dx), 0.5 * p);
    // Across S = 0 the STATE jumps by the shock itself (the face sits on it); Rankine-Hugoniot
    // is what continuity is there: the flux (masterplan_cdnozzle A1).
    const FvFlux<double> fs0 = fvEulerFlux(s0, S), fs1 = fvEulerFlux(s1, S);
    const double fm = std::abs(fs1.rho) * c;
    const double jumpS = std::max({std::abs(fs0.rho - fs1.rho) / std::abs(fs1.rho), std::abs(fs0.m[1] - fs1.m[1]) / fm,
                                   std::abs(fs0.E - fs1.E) / (fm * c)});
    jump = std::max({jumpS, std::abs(f0.p - f1.p) / p, std::abs(f0.rho - f1.rho) / rho});
    std::printf("  S = 0: state jump p %.3f (the shock), flux jump %.2e\n", std::abs(s0.p - s1.p) / p, jumpS);
    std::printf("  branches: stays %d, enters %d, supersonic %d, subsonic %d, transonic %d, backflow %d; worst %.2e, jump %.2e\n", hits[0],
                hits[1], hits[2], hits[3], hits[4], hits[5], worst, jump);
    check(worst <= 1e-12, "every branch equals the exact sampler to 1e-12");
    check(jump <= 4e-6, "state continuous at M 1, flux continuous at S = 0 (a 1e-7 step moves them <= 4e-6)");
    check(std::ranges::all_of(hits, [](int h) { return h > 0; }), "each branch hit");
    return dnstest::finish();
}

// --- G3, off rows are the shipped bytes ------------------------------------------

// fvBoundaryState's Inflow, Outflow, Far as at 644e596, verbatim: the frozen row.
template <class T, class G>
inline FvPrim<T> shippedState(const FvBc& bc, const FvPrim<T>& P, const G S[3], double gamma, double Rgas)
{
    using std::pow;
    using std::sqrt;
    const double S2 = double(S[0]) * double(S[0]) + double(S[1]) * double(S[1]) + double(S[2]) * double(S[2]);
    const double magS = sqrt(S2);
    const T n[3] = {T(double(S[0]) / magS), T(double(S[1]) / magS), T(double(S[2]) / magS)};
    auto un = [&](const T u[3]) { return u[0] * n[0] + u[1] * n[1] + u[2] * n[2]; };
    auto ke = [](const T u[3]) { return T(0.5) * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]); };
    FvPrim<T> b = P;
    switch (bc.kind) {
    case FvBcKind::Inflow: {
        const T cP = sqrt(T(gamma) * P.p / P.rho);
        const T uin[3] = {T(bc.uIn[0]), T(bc.uIn[1]), T(bc.uIn[2])};
        b.p = P.p + P.rho * cP * (un(P.u) - un(uin));
        b.temp = T(bc.tIn);
        b.rho = b.p / (T(Rgas) * b.temp);
        for (int k = 0; k < 3; ++k) b.u[k] = uin[k];
        b.e = b.p / (T(gamma - 1.0) * b.rho) + ke(b.u);
        return b;
    }
    case FvBcKind::Outflow: {
        if (bc.characteristic && value(un(P.u)) >= 0.0) {
            const T r = pow(T(bc.pOut) / P.p, 1.0 / gamma), cP = sqrt(T(gamma) * P.p / P.rho);
            const T dun = T(2.0 / (gamma - 1.0)) * cP * (T(1.0) - pow(r, 0.5 * (gamma - 1.0)));
            b.p = T(bc.pOut);
            b.rho = P.rho * r;
            b.temp = b.p / (b.rho * T(Rgas));
            for (int k = 0; k < 3; ++k) b.u[k] = P.u[k] + dun * n[k];
            b.e = b.p / (T(gamma - 1.0) * b.rho) + ke(b.u);
            return b;
        }
        if (value(un(P.u)) < 0.0) {
            const T unP = un(P.u);
            for (int k = 0; k < 3; ++k) b.u[k] = unP * n[k];
            b.temp = T(bc.tBack);
        }
        b.p = T(bc.pOut);
        b.rho = b.p / (T(Rgas) * b.temp);
        b.e = b.p / (T(gamma - 1.0) * b.rho) + ke(b.u);
        return b;
    }
    default: return b;
    }
}

template <class T>
inline bool sameBits(const FvPrim<T>& a, const FvPrim<T>& b)
{
    const auto same = [](T x, T y) { return std::bit_cast<std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>>(x) ==
                                            std::bit_cast<std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>>(y); };
    return same(a.rho, b.rho) && same(a.u[0], b.u[0]) && same(a.u[1], b.u[1]) && same(a.u[2], b.u[2]) && same(a.e, b.e) &&
           same(a.p, b.p) && same(a.temp, b.temp);
}

template <class T>
inline void bitsFor(const char* tag)
{
    std::mt19937_64 gen(11);
    std::uniform_real_distribution<double> d(0.2, 3.0), sgn(-2.0, 2.0);
    const double g = kGam, R = 1.0;
    long diff = 0, n = 0;
    for (int i = 0; i < 40000; ++i) {
        const T rho = T(d(gen)), mx = T(sgn(gen)), my = T(sgn(gen)), mz = T(sgn(gen)), E = T(4.0 + d(gen));
        const FvPrim<T> P = fvPrimOf(rho, mx, my, mz, E, g, R);
        const double S[3] = {sgn(gen), sgn(gen), sgn(gen)}, Cf[3] = {0, 0, 0};
        FvBc bc;
        const int kind = i % 3;
        bc.kind = kind == 0 ? FvBcKind::Inflow : FvBcKind::Outflow;
        bc.uIn[0] = 1.0;
        bc.uIn[1] = -0.5;
        bc.tIn = 1.1;
        bc.pOut = d(gen);
        bc.tBack = 1.2;
        bc.characteristic = kind == 2;
        const FvPrim<T> a = fvBoundaryState(bc, P, S, Cf, g, R), b = shippedState<T>(bc, P, S, g, R);
        diff += sameBits(a, b) ? 0 : 1;
        ++n;
    }
    check(diff == 0, std::format("{}: {} random rows equal the shipped bytes ({} differ)", tag, n, diff));
}

inline int bits()
{
    std::printf("G3 unset rows are the shipped bytes\n");
    bitsFor<double>("FP64");
    bitsFor<float>("FP32");
    return dnstest::finish();
}

// --- G4 is test_facemesh. G5, G6: the wedge ---------------------------------------

// z of a wedge PolyMesh scaled from half angle h0 to h1 (the y = r form's exact gauge). cos: the
// cos form instead, (x, r cos h, +-r sin h).
inline PolyMesh rewedge(PolyMesh pm, double h0, double h1, bool cosForm = false)
{
    for (std::size_t i = 0; i < pm.px.size(); ++i) {
        const double r = pm.py[i], sg = pm.pz[i] > 0 ? 1.0 : pm.pz[i] < 0 ? -1.0 : 0.0;
        (void)h0;
        if (cosForm) {
            pm.py[i] = r * std::cos(h1);
            pm.pz[i] = sg * r * std::sin(h1);
        } else {
            pm.pz[i] = sg * r * std::tan(h1);
        }
    }
    return pm;
}

inline FvState march(const FaceMesh& fm, const FvConfig& cfg, FvState s, double dt, int steps)
{
    const FvRef ref = must(FvRef::make(fm, cfg), "FvRef");
    ref.march(s, dt, steps);
    return s;
}

// Pressure per source cell.
inline std::vector<double> pressure(const FaceMesh& fm, const FvState& s)
{
    std::vector<double> p(std::size_t(fm.nC));
    for (std::size_t c = 0; c < p.size(); ++c)
        p[std::size_t(fm.srcCell[c])] = fvPrimOf(s.var(0)[c], s.var(1)[c], s.var(2)[c], s.var(3)[c], s.var(4)[c], kGam, 287.1).p;
    return p;
}

inline int wedge()
{
    std::printf("G5 wedge rest and gauge\n");
    const Ref w = loadRef("nozzle_W");
    const double R = w.Rgas, h20 = w.fm.wedgeHalf;
    std::printf("  half angle %.15g deg, R %.15g\n", h20 * 180 / std::numbers::pi, R);
    check(std::abs(h20 * 180 / std::numbers::pi - 10.0) < 1e-9, "half angle read off the sides");
    // Uniform rest: Slip p_P S on sides and walls, Sum S = 0 on a closed cell. An open row at
    // the rest state itself.
    {
        FvConfig c = nozzleCfg(w.fm, R, {.pb = 7e4});
        for (FvBc& b : c.bc)
            if (b.kind == FvBcKind::Total) {
                b.pTotal = 7e4;
                b.tTotal = 300.0;
            }
        const FvRef ref = must(FvRef::make(w.fm, c), "FvRef rest");
        FvState s = FvState::zeros(w.fm.nC), o;
        const double rho = 7e4 / (R * 300.0);
        std::ranges::fill_n(s.var(0), w.fm.nC, rho);
        std::ranges::fill_n(s.var(4), w.fm.nC, 7e4 / (kGam - 1.0));
        ref.rhs(s, o);
        double mx = 0, w0 = 0;
        for (std::size_t q = 0; q < o.q.size(); ++q) mx = std::max(mx, std::abs(o.q[q]));
        for (int c2 = 0; c2 < w.fm.nC; ++c2) w0 = std::max(w0, std::abs(o.var(3)[c2]));
        const double scale = 7e4 / std::cbrt(*std::ranges::min_element(w.fm.V));
        std::printf("  rest: max |rhs| %.3e, scale p/h %.3e, ratio %.3e, |w row| %.3e\n", mx, scale, mx / scale, w0);
        check(mx / scale <= 1e-13, "uniform rest: rhs <= 1e-13 of p/h");
        check(w0 == 0.0, "uniform rest: w row exactly 0.0");
    }
    // The gauge: the same march on 5 and 20 deg. Foam's 0/ field, hybrid at eps 1/16.
    const fs::path d0 = w.dir / "0";
    const std::vector<double> p0 = field(d0 / "p", 1, w.fm.nC), T0v = field(d0 / "T", 1, w.fm.nC), U0 = field(d0 / "U", 3, w.fm.nC);
    auto run = [&](const PolyMesh& pm) {
        const FaceMesh fm = must(FaceMesh::fromPolyMesh(pm, CellOrder::Source, 0, {.wedge = true}), "rewedge");
        FvConfig c = nozzleCfg(fm, R, {});
        c.shock = {.mode = ShockMode::Hll, .eps4 = kFvShockEps4};
        return pressure(fm, march(fm, c, stateOf(fm, p0, T0v, U0, R), 2e-4, 200));
    };
    const double h5 = 2.5 * std::numbers::pi / 180;
    const std::vector<double> pa = run(w.pm), pb = run(rewedge(w.pm, h20, h5));
    const std::vector<double> pc = run(rewedge(w.pm, h20, h5, true)), pd = run(rewedge(w.pm, h20, h20, true));
    double e = 0, ec = 0, moved = 0;
    for (std::size_t c = 0; c < pa.size(); ++c) {
        e = std::max(e, std::abs(pa[c] / pb[c] - 1.0));
        ec = std::max(ec, std::abs(pc[c] / pd[c] - 1.0));
        moved = std::max(moved, std::abs(pa[c] / p0[c] - 1.0));
    }
    std::printf("  max dp/p: y = r 20 vs 5 deg %.3e, cos form %.3e; the run moved p %.3e\n", e, ec, moved);
    check(moved >= 1e-3, "the march moved the field");
    check(e <= 1e-10, "y = r: 5 vs 20 deg max dp/p <= 1e-10");
    check(ec >= 1e-2, "control: the cos form differs >= 1e-2");
    return dnstest::finish();
}

inline int wedgeW()
{
    std::printf("G6 wedge w stays 0.0, viscous, 1000 steps\n");
    const Ref w = loadRef("nozzle_W");
    const double R = w.Rgas, mu = 1e-3;
    const std::vector<double> p0 = field(w.dir / "0/p", 1, w.fm.nC), T0v = field(w.dir / "0/T", 1, w.fm.nC), U0 = field(w.dir / "0/U", 3, w.fm.nC);
    auto wMax = [&](const FaceMesh& fm) {
        FvConfig c = nozzleCfg(fm, R, {.slip = false}, mu);
        c.shock = {.mode = ShockMode::Hll, .eps4 = kFvShockEps4};
        const FvRef ref = must(FvRef::make(fm, c), "FvRef");
        FvState s = stateOf(fm, p0, T0v, U0, R);
        const double dt = 0.5 * must(ref.stableDt(s), "dt");
        std::printf("  dt %.3e\n", dt);
        ref.march(s, dt, 1000);
        double mw = 0;
        for (int q = 0; q < fm.nC; ++q) mw = std::max(mw, std::abs(s.var(3)[q]));
        return mw;
    };
    const double m0 = wMax(w.fm);
    FaceMesh bad = w.fm; // remove the mirror pin on one side face, one ulp
    const MeshPatch& back = *std::ranges::find(bad.patches, std::string("back"), &MeshPatch::name);
    const std::size_t f = std::size_t(bad.nI + back.first + back.count / 2);
    bad.S[0][f] = std::nextafter(bad.S[0][f], 1e300);
    bad.S[2][f] = std::nextafter(bad.S[2][f], 1e300);
    const double m1 = wMax(bad);
    std::printf("  max |rho w|: pinned %.3e, one ulp off the mirror %.3e\n", m0, m1);
    check(m0 == 0.0, "w == 0.0 bitwise after 1000 viscous FP64 steps");
    check(m1 != 0.0, "control: an unpinned side leaves w != 0");
    return dnstest::finish();
}

// --- G10, T1 at foam's field --------------------------------------------------------

struct Resid
{
    double r[5];
    double max() const { return *std::max_element(r, r + 5); }
};

// max |V dq/dt| per row, each row over its flux scale. Fg: the FvRef configured.
inline Resid residual(const Ref& w, const FvConfig& c, double Rstate)
{
    const FvRef ref = must(FvRef::make(w.fm, c), "FvRef T1R");
    const FvState s = stateOf(w.fm, w.p, w.T, w.U, Rstate);
    FvState o;
    FvEval ev;
    ref.evaluate(s, ev, o);
    double fm = 0, Um = 0, em = 0;
    for (int f = 0; f < w.fm.nI; ++f) fm = std::max(fm, std::abs(ev.F[0][std::size_t(f)]));
    for (int q = 0; q < w.fm.nC; ++q) {
        Um = std::max(Um, std::abs(ev.prim[1][std::size_t(q)]));
        Um = std::max(Um, std::abs(ev.prim[2][std::size_t(q)]));
        em = std::max(em, ev.prim[4][std::size_t(q)]);
    }
    Resid r{};
    for (int v = 0; v < 5; ++v) {
        const double scale = v == 0 ? fm : v == 4 ? fm * em : fm * Um;
        for (int q = 0; q < w.fm.nC; ++q) r.r[v] = std::max(r.r[v], std::abs(o.var(v)[q]) * w.fm.V[std::size_t(q)] / scale);
    }
    return r;
}

inline int t1r()
{
    std::printf("G10 T1 residual: FvRef force at rhoCentralFoam's field, upwind Kurganov\n");
    struct C
    {
        const char* name;
        double pb, band;
    } cases[] = {{"nozzle_A", 0.0, 1e-13}, {"nozzle_B", 6e4, 1e-13}, {"nozzle_W", 0.0, 1e-11}};
    for (const C& k : cases) {
        const Ref w = loadRef(k.name);
        FvConfig c = nozzleCfg(w.fm, w.Rgas, {.pb = k.pb});
        c.shock = {.mode = ShockMode::Hll, .eps4 = 0.0, .force = true};
        const Resid r = residual(w, c, w.Rgas);
        FvConfig kg = nozzleCfg(w.fm, w.Rgas, {.pb = k.pb});
        const Resid q = residual(w, kg, w.Rgas);
        FvConfig wrongR = nozzleCfg(w.fm, w.Rgas * (1 + 1e-6), {.pb = k.pb});
        wrongR.shock = c.shock;
        const Resid q2 = residual(w, wrongR, w.Rgas * (1 + 1e-6));
        FvConfig wrongIn = c; // the inlet at a Riemann-free fixed state: p0, T0, u_P
        for (FvBc& b : wrongIn.bc)
            if (b.kind == FvBcKind::Total) {
                const double uIn = 0.0;
                b.kind = FvBcKind::Inflow;
                b.tIn = 300.0;
                b.uIn[0] = uIn;
            }
        const Resid q3 = residual(w, wrongIn, w.Rgas);
        std::printf("  %s: R %.8f, rows %.2e %.2e %.2e %.2e %.2e; KG control %.3e; R + 1e-6 %.3e; wrong inlet %.3e\n", k.name, w.Rgas, r.r[0],
                    r.r[1], r.r[2], r.r[3], r.r[4], q.max(), q2.max(), q3.max());
        check(r.max() <= k.band, std::format("{}: max |R| / flux scale {:.2e} <= {:.0e}", k.name, r.max(), k.band));
        check(q.max() >= 0.05, std::format("{}: control: the KG flux {:.3f} >= 0.05", k.name, q.max()));
        check(q2.max() > 10 * r.max(), std::format("{}: control: gas R + 1e-6 moves it ({:.2e})", k.name, q2.max()));
        check(q3.max() > 1e-3, std::format("{}: control: a wrong inlet fails ({:.2e})", k.name, q3.max()));
    }
    return dnstest::finish();
}

// --- G24, the hoop bound -------------------------------------------------------------

// A flat wedge column: nx slabs of nr cells, radii r[0] = 0 .. r[nr], half angle h, patches
// inlet, outlet, wall, back, front. The y = r form, as nozzle_mesh.py writes it.
inline PolyMesh wedgeColumn(int nx, const std::vector<double>& r, double dx, double h)
{
    const int nr = int(r.size()) - 1, nxp = nx + 1;
    const double t = std::tan(h);
    PolyMesh pm;
    auto nid = [&](int i, int j, int k) { return j == 0 ? i : nxp + (i + nxp * (j - 1)) * 2 + k; };
    for (int i = 0; i < nxp; ++i) {
        pm.px.push_back(i * dx);
        pm.py.push_back(0.0);
        pm.pz.push_back(0.0);
    }
    for (int j = 1; j <= nr; ++j)
        for (int i = 0; i < nxp; ++i)
            for (int k = 0; k < 2; ++k) {
                pm.px.push_back(i * dx);
                pm.py.push_back(r[std::size_t(j)]);
                pm.pz.push_back((k == 0 ? -1.0 : 1.0) * r[std::size_t(j)] * t);
            }
    pm.faceStart.push_back(0);
    auto add = [&](std::vector<int> f, int o, int n = -1) {
        std::vector<int> u;
        for (int a : f)
            if (std::ranges::find(u, a) == u.end()) u.push_back(a);
        if (u.size() < 3) return;
        pm.faceNodes.insert(pm.faceNodes.end(), u.begin(), u.end());
        pm.faceStart.push_back(std::int32_t(pm.faceNodes.size()));
        pm.owner.push_back(o);
        if (n >= 0) pm.neighbour.push_back(n);
    };
    auto cell = [&](int i, int j) { return i + nx * j; };
    for (int j = 0; j < nr; ++j)
        for (int i = 0; i < nx; ++i) {
            if (i < nx - 1) add({nid(i + 1, j, 0), nid(i + 1, j + 1, 0), nid(i + 1, j + 1, 1), nid(i + 1, j, 1)}, cell(i, j), cell(i + 1, j));
            if (j < nr - 1) add({nid(i, j + 1, 0), nid(i, j + 1, 1), nid(i + 1, j + 1, 1), nid(i + 1, j + 1, 0)}, cell(i, j), cell(i, j + 1));
        }
    const int nI = int(pm.neighbour.size());
    auto patch = [&](const char* name, FoamPatchType ty, auto gen) {
        const int st = int(pm.owner.size());
        gen();
        pm.man.patches.push_back({name, ty, std::int64_t(pm.owner.size()) - st, st, "", std::nullopt});
    };
    patch("inlet", FoamPatchType::Patch, [&] {
        for (int j = 0; j < nr; ++j) add({nid(0, j, 0), nid(0, j, 1), nid(0, j + 1, 1), nid(0, j + 1, 0)}, cell(0, j));
    });
    patch("outlet", FoamPatchType::Patch, [&] {
        for (int j = 0; j < nr; ++j) add({nid(nx, j, 0), nid(nx, j + 1, 0), nid(nx, j + 1, 1), nid(nx, j, 1)}, cell(nx - 1, j));
    });
    patch("wall", FoamPatchType::Wall, [&] {
        for (int i = 0; i < nx; ++i) add({nid(i, nr, 0), nid(i, nr, 1), nid(i + 1, nr, 1), nid(i + 1, nr, 0)}, cell(i, nr - 1));
    });
    patch("back", FoamPatchType::Wedge, [&] {
        for (int j = 0; j < nr; ++j)
            for (int i = 0; i < nx; ++i) add({nid(i, j, 0), nid(i, j + 1, 0), nid(i + 1, j + 1, 0), nid(i + 1, j, 0)}, cell(i, j));
    });
    patch("front", FoamPatchType::Wedge, [&] {
        for (int j = 0; j < nr; ++j)
            for (int i = 0; i < nx; ++i) add({nid(i, j, 1), nid(i + 1, j, 1), nid(i + 1, j + 1, 1), nid(i, j + 1, 1)}, cell(i, j));
    });
    pm.man.nPoints = std::int64_t(pm.px.size());
    pm.man.nCells = nx * nr;
    pm.man.nFaces = std::int64_t(pm.owner.size());
    pm.man.nInternalFaces = nI;
    return pm;
}

// Largest |lambda| of the viscous part of the operator on the w = 0 subspace, FD on the
// conserved rows. w's side mode is outside the dt by design (~1/t^2): G6 keeps w at 0.0.
inline double viscousLambda(const FvRef& mu, const FvRef& inv, const FvState& s0, int nC)
{
    FvState x = FvState::zeros(nC), r0, r1, a0, a1, y = s0, z = FvState::zeros(nC);
    std::mt19937_64 gen(3);
    std::normal_distribution<double> nd;
    for (int v : {1, 2, 4})
        for (int c = 0; c < nC; ++c) x.var(v)[c] = nd(gen);
    mu.rhs(s0, r0);
    inv.rhs(s0, a0);
    double lam = 0;
    for (int it = 0; it < 600; ++it) {
        double n = 0;
        for (double v : x.q) n += v * v;
        n = std::sqrt(n);
        for (double& v : x.q) v /= n;
        const double e = 1e-7;
        for (std::size_t i = 0; i < y.q.size(); ++i) y.q[i] = s0.q[i] + e * x.q[i];
        mu.rhs(y, r1);
        inv.rhs(y, a1);
        for (std::size_t i = 0; i < x.q.size(); ++i) z.q[i] = ((r1.q[i] - r0.q[i]) - (a1.q[i] - a0.q[i])) / e;
        double l = 0;
        for (double v : z.q) l += v * v;
        lam = std::sqrt(l);
        x = z;
        std::ranges::fill_n(x.var(3), nC, 0.0);
    }
    return lam;
}

inline int hoop()
{
    std::printf("G24 hoop: lambda dt <= 2 diff on the wedge viscous operator\n");
    const double deg[] = {5.0, 20.0};
    const double beta[] = {0.0, 2.5};
    for (double d : deg)
        for (double b : beta)
            for (int nr : {20, 40}) {
                const double h = 0.5 * d * std::numbers::pi / 180;
                std::vector<double> r(std::size_t(nr) + 1);
                for (int j = 0; j <= nr; ++j)
                    r[std::size_t(j)] = b > 0 ? std::tanh(b * j / nr) / std::tanh(b) : double(j) / nr;
                // dx long: the bound is the radial column's and the side term's, hoop.py's rig (x
                // neighbours identical); a short dx lets x faces hide the side term.
                const PolyMesh pm = wedgeColumn(4, r, 1.0, h);
                const FaceMesh fm = must(FaceMesh::fromPolyMesh(pm, CellOrder::Source, 0, {.wedge = true}), "column");
                // Pr 2: max(4/3, gamma/Pr) = 4/3, the momentum factor: no energy margin on the bound.
                auto cfg = [&](double mu) {
                    FvConfig c;
                    c.tr = Transport::air(mu, 1.0, 0.0, 2.0, kGam, 1.0, false);
                    for (const MeshPatch& m : fm.patches) {
                        FvBc q;
                        q.kind = FvBcKind::Slip;
                        q.wedge = m.type == FoamPatchType::Wedge;
                        c.bc.push_back(q);
                    }
                    return c;
                };
                const double mu = 1e-2;
                const FvRef ref = must(FvRef::make(fm, cfg(mu)), "viscous"), inv = must(FvRef::make(fm, cfg(0.0)), "inviscid");
                FvState s = FvState::zeros(fm.nC);
                std::ranges::fill_n(s.var(0), fm.nC, 1.0);
                std::ranges::fill_n(s.var(4), fm.nC, 1.0 / (kGam * (kGam - 1.0)));
                const double lam = viscousLambda(ref, inv, s, fm.nC);
                const double dt = must(ref.stableDt(s, {.cfl = 1e12, .diff = 1.0}), "dt");
                // The control: the same bound with no side term.
                double dtBare = HUGE_VAL;
                const double f = fvDiffusivityFactor(kGam, 2.0), sh = std::sin(h);
                for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) {
                    double side = 0;
                    for (int q = fm.cellStart[c]; q < fm.cellStart[c + 1]; ++q) {
                        const std::size_t fc = std::size_t(fm.cellFace[std::size_t(q)]);
                        if (fc < std::size_t(fm.nI)) continue;
                        if (ref.config().bc[std::size_t(ref.bcOf()[fc - std::size_t(fm.nI)])].wedge)
                            side += fm.V[c] * sh * sh / (fm.d[2][fc] * fm.d[2][fc]);
                    }
                    dtBare = std::min(dtBare, fm.V[c] / (mu * f * (ref.diffCoef()[c] - side)));
                }
                std::printf("  %g deg beta %g nr %d: lambda %.4g, lambda dt / 2 %.3f, no side term %.3f\n", d, b, nr, lam, lam * dt / 2, lam * dtBare / 2);
                check(lam * dt <= 2.0, std::format("{:g} deg beta {:g} nr {}: lambda dt <= 2", d, b, nr));
                if (b == 0.0) check(lam * dtBare / 2 >= 1.2, "control: no side term is over the bound");
            }
    return dnstest::finish();
}

} // namespace dnstest::noz
