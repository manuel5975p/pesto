// The wedge on the flame route (masterplan_combscale S6a, docs/nozzle.md "Wedge flame"): ctest fvref_wedgeflame.
//   W1  nozzle_W, C0 + hll + mu 1e-3, a Y ramp, 1000 FP64 steps: max |rho w| == 0.0. Control: a side face one ulp
//       off the mirror leaves w != 0.
//   W2  host: passive Y = 1/4, 40 steps: rho Y == rho/4 in every cell, every wedge side's F_Y == 0.0, bitwise.
//       Control: Y = 0.3 is off by >= 1 ulp somewhere.
//   W4  FvRef's ledger over 400 reacting steps, C0 + hll: |res_k| / S_k <= 1e-12 on every row (the z row exactly
//       0.0). Control: Omega dropped from the rho Y row >= 0.1.
// The rig (cfgW, icW, chemW) is shared with the device gates (test_gpu_fv_nozzle wedgeflame: W2 device, W3).
// Host only. Included by tests/test_fvref.cpp.
#pragma once

#include "check.hpp"
#include "fvref_nozzle.hpp"

#include "dns/facemesh.hpp"
#include "dns/fvnozzle.hpp"
#include "dns/fvref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <format>
#include <limits>
#include <string>
#include <vector>

namespace dnstest::wfl {

using namespace dns;
using dnstest::check;
using dnstest::noz::must;
using dnstest::noz::Ref;

inline constexpr int kFuel = kFvMeanVars;
// SI one-step chemistry on nozzle_W's air (T 170..300 K): rate A exp(-Ta/T) ~ 50 /s at 300 K (400 steps of 1.9e-5 s
// burn a third), a 30 K burn.
inline constexpr double kLnA = 5.9, kTa = 600.0, kQ = 3e4;

// C0 on the nozzle's gas. react false: lnA -inf, omega exactly 0 (the passive species).
inline FvChem chemW(bool react)
{
    return {.nY = 1, .fuel = 0, .lnA = react ? kLnA : -std::numeric_limits<double>::infinity(), .Ta = kTa, .q = kQ, .invPr = 1.0 / 0.72};
}

// nozzle_W's rows (Total in, zeroGradient out, a no-slip wall, the wedge sides), every row's y = yIn.
inline std::vector<FvBc> bcW(const FaceMesh& fm, double yIn)
{
    std::vector<FvBc> bc = noz::nozzleBc(fm, {.slip = false});
    for (FvBc& b : bc) b.y[0] = yIn;
    return bc;
}

// The viscous hll C0 rig of fm (mu 1e-3, D3 eps 1/16).
inline FvConfig cfgW(const FaceMesh& fm, double R, bool react, double yIn)
{
    FvConfig c;
    c.tr = Transport::air(1e-3, 1.0, 0.0, 0.72, noz::kGam, R, false);
    c.bc = bcW(fm, yIn);
    c.chem = chemW(react);
    c.shock = {.mode = ShockMode::Hll, .eps4 = kFvShockEps4};
    return c;
}

// Foam's 0/ field of w on fm (any ordering of w's mesh) with Y(x) of the cell centre's x.
template <class YOf>
FvState icW(const Ref& w, const FaceMesh& fm, YOf&& Y)
{
    const std::filesystem::path d0 = w.dir / "0";
    const FvState m = noz::stateOf(fm, noz::field(d0 / "p", 1, fm.nC), noz::field(d0 / "T", 1, fm.nC), noz::field(d0 / "U", 3, fm.nC), w.Rgas);
    FvState s = FvState::zeros(fm.nC, kFvMeanVars + 1);
    std::copy(m.q.begin(), m.q.end(), s.q.begin());
    for (std::size_t c = 0; c < std::size_t(fm.nC); ++c) s.var(kFuel)[c] = s.var(0)[c] * Y(fm.C[0][c]);
    return s;
}

// 1 at the low-x end, 0 at the high-x end, linear in x.
inline auto rampOf(const FaceMesh& fm)
{
    const auto [lo, hi] = std::ranges::minmax(fm.C[0]);
    return [lo = lo, hi = hi](double x) { return (hi - x) / (hi - lo); };
}

inline double maxAbs(const FvState& s, int v)
{
    double m = 0;
    for (int c = 0; c < s.nC; ++c) m = std::max(m, std::abs(s.var(v)[c]));
    return m;
}

// Cells whose rho Y is not rho * y bit for bit.
inline long offCells(const FvState& s, double y)
{
    long n = 0;
    for (int c = 0; c < s.nC; ++c) n += s.var(kFuel)[c] != s.var(0)[c] * y;
    return n;
}

// max |F_Y| over the wedge sides' faces of an evaluation.
inline double sideFy(const FaceMesh& fm, const FvEval& ev)
{
    double m = 0;
    for (const MeshPatch& p : fm.patches)
        if (p.type == FoamPatchType::Wedge)
            for (int b = p.first; b < p.first + p.count; ++b) m = std::max(m, std::abs(ev.Fy[0][std::size_t(fm.nI + b)]));
    return m;
}

inline void w1(const Ref& w)
{
    std::printf("W1 w stays 0.0: C0 + hll, viscous, a Y ramp, 1000 FP64 steps\n");
    double burnt = 0;
    auto wMax = [&](const FaceMesh& fm) {
        const FvRef ref = must(FvRef::make(fm, cfgW(fm, w.Rgas, true, 1.0)), "FvRef W1");
        FvState s = icW(w, fm, rampOf(fm));
        const double fuel0 = fvTotals6(fm, s, kFuel)[5];
        const double dt = std::min(0.5 * must(ref.stableDt(s), "dt"), ref.chemDt(s, 0.5));
        FvLedger led = FvLedger::zeros(int(fm.patches.size()), kFvMeanVars + 1);
        ref.march(s, dt, 1000, nullptr, &led);
        burnt = led.omegaSum() / fuel0;
        std::printf("  dt %.3e, Omega / fuel(0) %.3f, max |rho v| %.3e\n", dt, burnt, maxAbs(s, 2));
        return maxAbs(s, 3);
    };
    const double m0 = wMax(w.fm);
    check(burnt >= 0.05, std::format("W1: the reaction ran (Omega / fuel(0) {:.3f} >= 0.05)", burnt));
    FaceMesh bad = w.fm; // one side face one ulp off the mirror
    const MeshPatch& back = *std::ranges::find(bad.patches, std::string("back"), &MeshPatch::name);
    const std::size_t f = std::size_t(bad.nI + back.first + back.count / 2);
    bad.S[0][f] = std::nextafter(bad.S[0][f], 1e300);
    bad.S[2][f] = std::nextafter(bad.S[2][f], 1e300);
    const double m1 = wMax(bad);
    std::printf("  max |rho w|: pinned %.3e, one ulp off the mirror %.3e\n", m0, m1);
    check(m0 == 0.0, "W1: w == 0.0 bitwise after 1000 reacting viscous FP64 steps");
    check(m1 != 0.0, "W1 control: a side face one ulp off the mirror leaves w != 0");
}

inline void w2(const Ref& w)
{
    std::printf("W2 host: a passive Y = 1/4 through the wedge, 40 steps\n");
    auto run = [&](double y, double& fy) {
        const FvRef ref = must(FvRef::make(w.fm, cfgW(w.fm, w.Rgas, false, y)), "FvRef W2");
        FvState s = icW(w, w.fm, [y](double) { return y; });
        const FvState s0 = s;
        ref.march(s, 0.5 * must(ref.stableDt(s), "dt"), 40);
        FvEval ev;
        FvState out;
        ref.evaluate(s, ev, out);
        fy = sideFy(w.fm, ev);
        double moved = 0;
        for (int c = 0; c < s.nC; ++c) moved = std::max(moved, std::abs(s.var(0)[c] / s0.var(0)[c] - 1.0));
        check(moved >= 1e-4, std::format("W2: the march moved rho ({:.2e})", moved));
        return offCells(s, y);
    };
    double fy = 0, fyc = 0;
    const long off = run(0.25, fy), offc = run(0.3, fyc);
    std::printf("  Y 1/4: %ld cells off rho/4, max side |F_Y| %.3e; control Y 0.3: %ld cells off\n", off, fy, offc);
    check(off == 0, "W2: rho Y == rho/4 bitwise in every cell");
    check(fy == 0.0, "W2: the wedge sides' species flux == 0.0");
    check(fyc == 0.0, "W2: the sides' species flux == 0.0 at Y 0.3 too");
    check(offc >= 1, "W2 control: Y = 0.3 is off by >= 1 ulp somewhere");
}

inline void w4(const Ref& w)
{
    std::printf("W4 the ledger over a reacting wedge march: 400 steps, C0 + hll, viscous\n");
    const FaceMesh& fm = w.fm;
    const FvConfig c = cfgW(fm, w.Rgas, true, 1.0);
    const FvRef ref = must(FvRef::make(fm, c), "FvRef W4");
    FvState s = icW(w, fm, [](double) { return 1.0; });
    const std::vector<FvRole> roles = fvRoles(c.bc, fm);
    const FvRow6 P0 = fvTotals6(fm, s, kFuel);
    FvLedger led = FvLedger::zeros(int(fm.patches.size()), kFvMeanVars + 1);
    for (int n = 0; n < 400; ++n) ref.march(s, std::min(0.5 * must(ref.stableDt(s), "dt"), ref.chemDt(s, 0.5)), 1, nullptr, &led);
    const FvRow6 P = fvTotals6(fm, s, kFuel);
    const FvRoleSums rs = fvRoleSums(led, roles, kFuel);
    double worst = 0, ctrl = 0;
    for (std::size_t k = 0; k < 6; ++k) {
        double res = P[k] - P0[k], S = 0;
        for (const FvRow6& r : rs.I) {
            res += r[k];
            S += std::abs(r[k]);
        }
        const double noOmega = res;
        if (k == 4) res -= c.chem.q * rs.omega;
        if (k == 5) res += rs.omega;
        const int row = k == 5 ? kFuel : int(k);
        for (std::size_t q = 0; q < std::size_t(fm.nC); ++q) S += fm.V[q] * std::abs(s.var(row)[q]);
        // The z row: S 0, the identity must hold as 0.0.
        const double e = S > 0.0 ? std::abs(res) / S : (res == 0.0 ? 0.0 : HUGE_VAL);
        worst = std::isfinite(e) ? std::max(worst, e) : HUGE_VAL;
        if (k == 5) ctrl = std::abs(noOmega) / S;
        std::printf("  row %zu: res / S %.2e (S %.4g)\n", k, e, S);
    }
    std::printf("  max |res_k| / S_k %.2e; Omega %.4g of fuel(0) %.4g; control (Omega dropped from rho Y) %.3f; max |rho w| %.1e\n", worst,
                rs.omega, P0[5], ctrl, maxAbs(s, 3));
    check(worst <= 1e-12, std::format("W4: ledger identity {:.2e} <= 1e-12 of S on every row", worst));
    check(ctrl >= 0.1, std::format("W4 control: Omega dropped from the rho Y row {:.3f} >= 0.1", ctrl));
}

inline int run()
{
    const Ref w = noz::loadRef("nozzle_W");
    w1(w);
    w2(w);
    w4(w);
    return dnstest::finish();
}

} // namespace dnstest::wfl
