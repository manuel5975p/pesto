// The 0D adiabatic reactor of model C0 (masterplan_combnozzle P0, docs/chem.md "P0: the 0D reactor"):
// chemistry alone, at rest, rho fixed. Shared by test_fvref (`fvref_reactor`, host) and
// test_gpu_fv_flame (`gpu_fv_flame_reactor`, device).
//   dY/dt = -e^(lnA - Ta/T) Y,  T(Y) = T0 + dT (Y0 - Y)
// dT = q/cv = gamma (sigma - 1) Tu at constant volume (the FV box); q/cp = (sigma - 1) Tu is the
// constant-pressure reactor (the gate's control oracle; chemFoam's `constantProperty pressure`).
// The closed oracle: t(Y) = int_{ln Y}^{ln Y0} ds / k(T(e^s)) by adaptive Gauss-Kronrod G7K15
// (relative 1e-14 per panel, positive integrand: the sum is relative 1e-14), Y(t) by bisection
// in s = ln Y. References in kRk3* are the exact-arithmetic SSP-RK3 recurrences of the same ODE
// (40 digits, `tools/chem/chemfoam_reactor.py rk3ref`).
#pragma once

#include "dns/chem.hpp"
#include "dns/curvmesh.hpp"
#include "dns/facemesh.hpp"
#include "dns/fvflux.hpp"
#include "dns/fvref.hpp"

#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <string>
#include <utility>
#include <vector>

namespace dnstest::reactor {

using namespace dns;

inline constexpr double kGam = 1.4, kMach = 0.01, kSigma = 5.0, kBeta = 8.0, kPr = 0.72;
// premixedEigen(sigma 5, beta 8) Lambda, cached by tools/chem/foam_flame.py
// (runs/flames/foam/oracle_s5_b8.npz); the host test re-derives it.
inline constexpr double kLambda = 3656401.0984806987;
inline constexpr double kT0OverTu = 2.5;

// The ODE in any units. Pre: Y0 > 0, T(Y) > 0 on [0, Y0].
struct Reactor
{
    double lnA = 0, Ta = 0, T0 = 0, dT = 0, Y0 = 1;
    double T(double Y) const { return T0 + dT * (Y0 - Y); }
    // 1/k at s = ln Y: the integrand of t(Y).
    double invRate(double s) const { return std::exp(Ta / T(std::exp(s)) - lnA); }
};

// G7K15 on [a, b]: the Kronrod value; err = |K15 - G7|.
template <class F>
double gk15(const F& f, double a, double b, double& err)
{
    static constexpr std::array<double, 8> xk{0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                              0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                              0.586087235467691130294144845693013, 0.405845151377397166906606412076961,
                                              0.207784955007898467600689403773245, 0.0};
    static constexpr std::array<double, 8> wk{0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                              0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                              0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                              0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static constexpr std::array<double, 4> wg{0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                                              0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
    const double c = 0.5 * (a + b), h = 0.5 * (b - a);
    const double f0 = f(c);
    double k = wk[7] * f0, g = wg[3] * f0;
    for (int i = 0; i < 7; ++i) {
        const double s = f(c - h * xk[std::size_t(i)]) + f(c + h * xk[std::size_t(i)]);
        k += wk[std::size_t(i)] * s;
        if (i % 2 == 1) g += wg[std::size_t(i / 2)] * s; // Gauss nodes are xk[1], xk[3], xk[5]
    }
    err = std::abs(h * (k - g));
    return h * k;
}

// Adaptive G7K15: a panel is accepted when |K - G| <= rel |K| (bisected otherwise, depth <= 50).
// Pre: f finite on [a, b]; a positive integrand (the reactor's) makes the sum relative rel.
template <class F>
double integrate(const F& f, double a, double b, double rel, int depth = 0)
{
    assert(depth <= 50 && "integrate: no convergence");
    double err = 0.0;
    const double v = gk15(f, a, b, err);
    if (err <= rel * std::abs(v) || depth == 50) return v;
    const double m = 0.5 * (a + b);
    return integrate(f, a, m, rel, depth + 1) + integrate(f, m, b, rel, depth + 1);
}

// t(Y): time from Y0 to Y. Pre: 0 < Y <= Y0.
inline double timeTo(const Reactor& r, double Y)
{
    return integrate([&](double s) { return r.invRate(s); }, std::log(Y), std::log(r.Y0), 1e-14);
}

// Y(t) by bisection in s = ln Y over [ln Y0 - 700, ln Y0], to |ds| <= 1e-15 (Y relative 1e-15).
// Pre: t >= 0. Past t(e^-700 Y0) it returns that bound's Y.
inline double yAt(const Reactor& r, double t)
{
    double hi = std::log(r.Y0), lo = hi - 700.0;
    while (hi - lo > 1e-15) {
        const double m = 0.5 * (lo + hi);
        if (m == lo || m == hi) break;
        (timeTo(r, std::exp(m)) > t ? lo : hi) = m;
    }
    return std::exp(0.5 * (lo + hi));
}

// The box-unit reactor of the FV case: Tu = 1/(gamma M^2), T0 = 2.5 Tu, Ta = theta Tu, A = Lambda.
// cv = true: dT = q/cv (the FV box); false: q/cp (the control oracle).
inline Reactor boxReactor(bool cv)
{
    const double Tu = 1.0 / (kGam * kMach * kMach);
    PremixedModel md;
    md.sigma = kSigma;
    md.beta = kBeta;
    const double dT = (cv ? kGam : 1.0) * (kSigma - 1.0) * Tu;
    return {.lnA = std::log(kLambda), .Ta = md.theta() * Tu, .T0 = kT0OverTu * Tu, .dT = dT, .Y0 = 1.0};
}

// Exact-arithmetic Y(t_e) - 1/2 of SSP-RK3 (Shu-Osher) at dt = t_e/N, and Kutta's RK3 at N 2000
// (the control: another third-order scheme). Independent of Lambda (only Lambda dt enters).
inline constexpr std::array<std::pair<int, double>, 6> kRk3Shu{{{2000, 0.00985789999890865},
                                                                 {4000, 0.00214102827171547},
                                                                 {8000, 0.00035739739022347},
                                                                 {16000, 5.13835450750308e-5},
                                                                 {32000, 6.86698883115692e-6},
                                                                 {64000, 8.86600817517595e-7}}};
inline constexpr double kRk3Kutta2000 = 0.00345131607375642;
// t(1/2) at constant volume and pressure (40 digits, mpmath quad).
inline constexpr double kTeV = 3.416547072079102420614175, kTeP = 4.843940899982749326036909;
// The constant-pressure oracle's Y at kTeV (the control's value).
inline constexpr double kYcpAtTe = 0.9514579947756939028;

// The FV case: a uniform periodic 4 x 4 x 1 box (h = 1: every coordinate an integer, so a
// uniform state's face sums cancel exactly), at rest, rho 1, Y 1, T0 = 2.5 Tu, model C0
// sigma 5 beta 8 at M 0.01 units.
struct ReactorCase
{
    FaceMesh fm;
    std::vector<FvBc> bc;
    Transport tr;
    FvChem chem;
    FvState ic;
    Reactor ode;
    double Tu = 0;
};

inline std::expected<ReactorCase, std::string> reactorCase()
{
    const Grid g = Grid::make(4, 4, 1, 4.0, 4.0, 1.0);
    const StructuredSpec spec{.lo = {0, 0, 0},
                              .hi = {3, 3, 0},
                              .name = {"xlo", "xhi", "ylo", "yhi", "back", "front"},
                              .type = {FoamPatchType::Cyclic, FoamPatchType::Cyclic, FoamPatchType::Cyclic,
                                       FoamPatchType::Cyclic, FoamPatchType::Empty, FoamPatchType::Empty}};
    auto fm = FaceMesh::fromCurvMesh(CurvMesh::uniform(g), spec, CellOrder::Morton);
    if (!fm) return std::unexpected(fm.error());
    ReactorCase rc{.fm = std::move(*fm), .bc = {}, .tr = {}, .chem = {}, .ic = {}, .ode = boxReactor(true)};
    for (const MeshPatch& p : rc.fm.patches) {
        FvBc b;
        if (p.type == FoamPatchType::Empty) b.kind = FvBcKind::Empty;
        rc.bc.push_back(b);
    }
    rc.Tu = 1.0 / (kGam * kMach * kMach);
    rc.tr = Transport::air(kPr, rc.Tu, 288.15, kPr, kGam, 1.0, false);
    const double q = rc.tr.cp() * (kSigma - 1.0) * rc.Tu;
    rc.chem = {.nY = 1, .fuel = 0, .lnA = rc.ode.lnA, .Ta = rc.ode.Ta, .q = q, .invPr = 1.0 / kPr};
    rc.ic = FvState::zeros(rc.fm.nC, kFvMeanVars + 1);
    for (std::size_t c = 0; c < std::size_t(rc.fm.nC); ++c) {
        rc.ic.var(0)[c] = 1.0;
        rc.ic.var(4)[c] = rc.ode.T0 / (kGam - 1.0);
        rc.ic.var(5)[c] = 1.0;
    }
    return rc;
}

// A plan gate recorded UNMET (masterplan_combnozzle A1): printed, never counted, never a pass.
// The regression power is the counted gate beside it (G0.2c: the exact recurrence).
inline void recordUnmet(bool ok, const std::string& what)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "UNMET (recorded, masterplan_combnozzle A1)", what.c_str());
}

// Cell c's T (R = 1, at rest: (gamma - 1) (E - |m|^2/2 rho)/rho) and Y.
inline double cellT(const FvState& s, std::size_t c)
{
    const double r = s.var(0)[c], m2 = s.var(1)[c] * s.var(1)[c] + s.var(2)[c] * s.var(2)[c] + s.var(3)[c] * s.var(3)[c];
    return (kGam - 1.0) * (s.var(4)[c] - 0.5 * m2 / r) / r;
}
inline double cellY(const FvState& s, std::size_t c) { return s.var(5)[c] / s.var(0)[c]; }

// Every row of every cell bitwise equal to cell 0's, momentum exactly 0.0.
inline bool uniformAtRest(const FvState& s)
{
    for (int v = 0; v < s.nV; ++v)
        for (std::size_t c = 0; c < std::size_t(s.nC); ++c) {
            const double x = s.var(v)[c];
            if (std::bit_cast<std::uint64_t>(x) != std::bit_cast<std::uint64_t>(s.var(v)[0])) return false;
            if (v >= 1 && v <= 3 && x != 0.0) return false;
        }
    return true;
}

// Sum V (E + q rho Y): the coupled energy, conserved by the step.
inline double coupledEnergy(const FvState& s, const FaceMesh& fm, double q)
{
    double e = 0.0;
    for (std::size_t c = 0; c < std::size_t(s.nC); ++c) e += fm.V[c] * (s.var(4)[c] + q * s.var(5)[c]);
    return e;
}

} // namespace dnstest::reactor
