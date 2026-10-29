// dns flame --mode candle --wick, host side (docs/chem.md, "Wick candle"):
// the body SDF and wall temperature, the region map (solid = sdf < 0, the
// fuel source band beside the WICK only), the soot model's Damkohler
// normalisation, the defaults / dry-run / label, the IC (hydrostatic p on
// every cell, solids species-free, a closed sheet round the wick), the
// flame / soot measurement on drawn fields, --soot off, and every rejection
// BY NAME.
#include "dns/chem.hpp"
#include "dns/config.hpp"
#include "dns/diffusion.hpp"
#include "dns/flame1d.hpp"
#include "dns/wick.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using dnstest::check;
using dnstest::runCli;

void rejects(std::vector<std::string> args, int code, const std::string& flag,
             const std::string& what)
{
    auto c = runCli(std::move(args));
    const bool ok = !c && c.error().code == code &&
                    (c.error().field == flag ||
                     c.error().message.find(flag) != std::string::npos);
    if (!ok && !c)
        std::printf("    got code %d field '%s': %s\n", c.error().code,
                    c.error().field.c_str(), c.error().message.c_str());
    if (c) std::printf("    accepted\n");
    check(ok, what);
}

std::vector<std::string> wick(std::vector<std::string> extra)
{
    std::vector<std::string> a{"flame", "--mode", "candle", "--wick"};
    a.insert(a.end(), extra.begin(), extra.end());
    return a;
}

double pressure(const dns::CompState& q, std::size_t c)
{
    const double u = q.mx[c] / q.rho[c], v = q.my[c] / q.rho[c];
    return 0.4 * (q.E[c] - 0.5 * q.rho[c] * (u * u + v * v));
}

} // namespace

int main()
{
    using namespace dns;
    // --- geometry ------------------------------------------------------------------
    {
        WickSpec s;
        s.yc = 10;
        s.xBase = 2;
        s.xTop = 6;
        s.wCandle = 6;
        s.wWick = 1;
        s.hWick = 2;
        s.tWick = 2;
        check(wickSdf(s, 4, 10) < 0 && wickSdf(s, 7, 10) < 0 &&
                  wickSdf(s, 7, 12) > 0 && wickSdf(s, 1, 10) > 0,
              "sdf: body and wick solid, air fluid");
        check(std::abs(wickSdf(s, 4, 15) - 2.0) < 1e-15 &&
                  std::abs(wickSdf(s, 10, 10) - 2.0) < 1e-15 &&
                  std::abs(wickSdf(s, 7, 11.5) - 1.0) < 1e-15 &&
                  std::abs(wickSdf(s, 9, 11.5) - std::hypot(1.0, 1.0)) < 1e-15,
              "sdf: exact distance outside (side, tip, wick side, corner)");
        check(std::abs(wickSdf(s, 7, 10) + 0.5) < 1e-15 &&
                  std::abs(wickSdf(s, 4, 10) + 2.0) < 1e-15,
              "sdf: depth inside (wick half-width, body to its top/base)");
        check(wickWallT(s, 7, 10.5) == 2 && wickWallT(s, 6, 9) == 1 &&
                  wickWallT(s, 4, 13) == 1,
              "wall T: the wick above the wax at tWick, the body at T0");
    }
    // --- defaults, dry-run, label -------------------------------------------------
    {
        auto c = runCli(wick({}));
        check(c.has_value(), "the wick candle defaults validate");
        if (!c) return dnstest::finish();
        const Grid& g = c->grid;
        const WickSpec& s = c->wick;
        std::printf("  defaults: %d x %d, h %.4g, body [%g, %g] x %g, wick %g x "
                    "%g at %g T0, Q %g, Da %g, soot %g / %g\n",
                    g.nx, g.ny, g.hx, s.xBase, s.xTop, s.wCandle, s.wWick,
                    s.hWick, s.tWick, s.Q, c->diff.Da, c->soot.Daf,
                    c->soot.Daox);
        check(c->flameWick && c->flameCandle && c->flameDiffusion &&
                  c->wickSoot,
              "wick = the candle's model + the body + soot");
        check(g.nx == 288 && g.ny == 200 && g.hx == 0.125,
              "box 288 x 200 at 8 cells/l (the slot candle's)");
        check(s.wWick == 0.75 && s.hWick == 2 && s.tWick == 2 && s.Q == 1 &&
                  s.wCandle == 6 && s.xTop - s.xBase == 4 && s.xBase == 3 &&
                  s.yc == 12.5 && s.tCandle == 1 && s.yF == 1,
              "wick defaults (0.75 x 2 l at 2 T0, Q 1, body 6 x 4 l from x 3)");
        check(c->diff.Da == kWickDaDefault && c->soot.Daf == kSootDafDefault &&
                  c->soot.Daox == kSootDaoxDefault && c->flameGravity == 14 &&
                  c->mach == 0.005,
              "Da 5e4, soot 10 / 2e5, g 14, M 0.005");
        const std::string d = c->describe(), lb = wickRunLabel(*c);
        check(d.find("WICK CANDLE") != std::string::npos &&
                  d.find(lb) != std::string::npos &&
                  d.find("FIXED") != std::string::npos,
              "dry-run: the mode, the fixed fuel rate, the label");
        check(lb.find("DNS 2D") != std::string::npos &&
                  lb.find("INFLATED") != std::string::npos &&
                  lb.find("TUNED FOR APPEARANCE") != std::string::npos &&
                  lb.find("no radiation") != std::string::npos &&
                  lb.find("NOT body-fitted") != std::string::npos,
              "label: 2D, inflated Mach, appearance-tuned soot, no radiation, "
              "the staircase wall");
        for (const auto& w : c->warnings) std::printf("  warning: %s\n", w.c_str());

        // Regions: solid = sdf < 0 at the centre; the source band = fluid
        // cells face-adjacent to a WICK solid cell (above the wax top);
        // mirror-symmetric about the axis.
        const auto reg = wickRegions(g, s);
        int nSolid = 0, nSrc = 0, badSrc = 0, asym = 0;
        auto at = [&](int i, int j) { return reg[g.idx(i, j, 0)]; };
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double x = (i + 0.5) * g.hx, y = (j + 0.5) * g.hy;
                const auto r = at(i, j);
                if ((r == std::uint8_t(WickRegion::Solid)) != (wickSdf(s, x, y) < 0))
                    ++badSrc;
                nSolid += r == std::uint8_t(WickRegion::Solid);
                if (r == std::uint8_t(WickRegion::Source)) {
                    ++nSrc;
                    // Within one cell of the wick box above the wax.
                    const double dx = std::max({s.xTop - x, x - s.xTip(), 0.0});
                    const double dy =
                        std::max(std::abs(y - s.yc) - 0.5 * s.wWick, 0.0);
                    if (!(x > s.xTop && std::hypot(dx, dy) < g.hx)) ++badSrc;
                }
                if (r != at(i, g.ny - 1 - j)) ++asym;
            }
        std::printf("  regions: %d solid, %d source, %d misplaced, %d "
                    "asymmetric\n",
                    nSolid, nSrc, badSrc, asym);
        check(badSrc == 0 && nSrc > 0 && asym == 0,
              "regions: solid = sdf < 0, the source band beside the wick "
              "only, symmetric");
        // 6 wick columns x 16 rows above the wax: 2 x 16 side cells + 6 on
        // top.
        check(nSrc == 2 * 16 + 6, "source band = the wick's 38 exposed faces' "
                                  "cells");

        // IC: p = p_a(x) on every cell, solids species-free, a closed sheet.
        CompState q = CompState::zeros(g);
        std::vector<double> ry[3];
        wickIC(*c, q, ry);
        const double T0 = c->flameTu;
        double worst = 0, tMax = 0, ySolid = 0, yfMax = 0, bound = 0;
        bool rest = true;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t cc = g.idx(i, j, 0);
                const double pa = candleAmbientP(T0, 14, T0, (i + 0.5) * g.hx);
                worst = std::max(worst, std::abs(pressure(q, cc) / pa - 1));
                const double t = pressure(q, cc) / q.rho[cc] / T0;
                tMax = std::max(tMax, t);
                rest = rest && q.mx[cc] == 0 && q.my[cc] == 0;
                if (at(i, j) == std::uint8_t(WickRegion::Solid)) {
                    ySolid = std::max({ySolid, std::abs(ry[0][cc]),
                                       std::abs(ry[1][cc]),
                                       std::abs(ry[2][cc])});
                    continue;
                }
                const double yf = ry[0][cc] / q.rho[cc], yo = ry[1][cc] / q.rho[cc];
                yfMax = std::max(yfMax, yf);
                if (yf < 0 || yf > 1 || yo < 0 || yo > 0.233 + 1e-15 ||
                    ry[2][cc] != 0)
                    ++bound;
            }
        std::printf("  IC: max |p/p_a - 1| %.3g, max T/T0 %.4f, max Y_F %.4f\n",
                    worst, tMax, yfMax);
        check(worst < 1e-13 && rest, "IC: at rest, p = p_a(x) on every cell");
        check(ySolid == 0, "IC: every species zero in the solid cells");
        check(bound == 0, "IC: Y_F, Y_O in bounds, no soot");
        check(tMax > 4.9 && tMax <= 5 + 1e-12 && yfMax > 0.5,
              "IC: a burning sheet (T_ad) round a fuel-rich wick");
        const std::size_t far = g.idx(g.nx - 20, 3, 0);
        check(ry[0][far] == 0 &&
                  std::abs(ry[1][far] / q.rho[far] - 0.233) < 1e-15,
              "IC away from the wick: air");

        // The measurement on drawn fields: Z = zst at x = xTip + 5 on the
        // axis (linear in x), half-width 1 at mid-flame, soot up to x = 12.
        const int nx = g.nx, ny = g.ny;
        const double h = g.hx;
        std::vector<float> YF(g.size()), YO(g.size()), YS(g.size()),
            W(g.size());
        std::vector<std::uint8_t> sol(g.size());
        const DiffusionModel& m = c->diff;
        const double xt = s.xTip() + 5.0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t cc = std::size_t(j) * nx + i;
                const double x = (i + 0.5) * h, y = (j + 0.5) * h;
                sol[cc] = at(i, j) == std::uint8_t(WickRegion::Solid);
                // Z = zst (1 + (xt - x)/5) on the axis, falling with |y - yc|
                // so that Z = zst at |y - yc| = 1 at mid-flame.
                const double xm = 0.5 * (s.xTip() + xt);
                const double zAxis = m.zst() * (1 + (xt - x) / 5.0);
                const double zAxisMid = m.zst() * (1 + (xt - xm) / 5.0);
                // Flat across the two axis columns (the measurement averages
                // them), then linear: Z = zst at |y - yc| = 1 at mid-flame.
                const double slope = (zAxisMid - m.zst()) / (1.0 - 0.5 * h);
                const double Z = std::max(
                    0.0, zAxis - slope * std::max(std::abs(y - s.yc) - 0.5 * h,
                                                  0.0));
                const SheetState st = burkeSchumann(m, std::min(Z, 1.0));
                YF[cc] = float(st.YF);
                YO[cc] = float(st.YO);
                YS[cc] = x < 12.0 && std::abs(y - s.yc) < 1 ? 1e-3f : 0.0f;
                W[cc] = std::abs(Z - m.zst()) < 0.01 && x > s.xTop + 1 ? 1.0f
                                                                       : 0.0f;
            }
        const WickMeasure wm =
            measureWick(m, s, YF, YO, YS, W, sol, nx, ny, h);
        std::printf("  measure: tip %.5f (drawn %.5f), height %.5f, half "
                    "%.5f, soot top %.4g, base %.4g, hits %g\n",
                    wm.xTip, xt, wm.height, wm.half, wm.xSoot, wm.xBase,
                    wm.hits);
        check(std::abs(wm.xTip - xt) < 0.02 * h && std::abs(wm.height - 5) <
                  0.02 * h,
              "measureWick: the drawn tip and height");
        check(std::abs(wm.half - 1.0) < 0.05, "measureWick: the half-width");
        check(wm.xSoot == 12.0 && wm.sootMax == float(1e-3) && wm.hits == 0,
              "measureWick: the soot zone's top, max, no hits");
    }
    // --- the soot model's Damkohler normalisation ----------------------------------
    {
        SootModel sm;
        const double T0 = 2e4, sig = 5;
        const double wf = arrheniusRate(sm.lnAf(sig), sm.thetaF * T0, 1.0, 1.0,
                                        sig * T0);
        const double wx = arrheniusRate2(sm.lnAox(sig), sm.thetaOx * T0, 1.0,
                                         1.0, 1.0, sig * T0);
        check(std::abs(wf / sm.Daf - 1) < 1e-12 && std::abs(wx / sm.Daox - 1) <
                  1e-12,
              "soot rates at T_ad, rho0, Y = 1 are the Damkohler numbers");
        const double cold = arrheniusRate(sm.lnAf(sig), sm.thetaF * T0, 1.0,
                                          1.0, 2 * T0);
        check(cold / wf < 1e-5, "formation is frozen at 2 T0 (Ta_f = 50 T0)");
        check(arrheniusRate(sm.lnAf(sig), sm.thetaF * T0, 1.0, -0.1, sig * T0) ==
                      0 &&
                  arrheniusRate2(sm.lnAox(sig), sm.thetaOx * T0, 1.0, 0.1,
                                 -1e-3, sig * T0) == 0,
              "the soot rates floor their Y");
    }
    // --- --soot off ----------------------------------------------------------------
    {
        auto c = runCli(wick({"--soot", "off"}));
        check(c.has_value() && !c->wickSoot, "--soot off validates");
        if (c) {
            CompState q = CompState::zeros(c->grid);
            std::vector<double> ry[3];
            wickIC(*c, q, ry);
            check(ry[2].empty() && ry[0].size() == c->grid.size(),
                  "--soot off: two species");
            check(wickRunLabel(*c).find("no soot") != std::string::npos,
                  "--soot off named in the label");
        }
        auto d = runCli(wick({"--damkohler", "5000", "--cells-per-unit", "16"}));
        check(d.has_value() && d->diff.Da == 5000 && d->grid.hx == 0.0625,
              "--damkohler and --cells-per-unit apply with --wick");
    }
    // --- rejections ----------------------------------------------------------------
    for (const char* f : {"--wick-width", "--wick-height", "--wick-temp",
                          "--fuel-rate", "--candle-width", "--candle-height",
                          "--soot-form", "--soot-ox"}) {
        rejects({"flame", "--mode", "candle", f, "1"}, 2, f,
                std::string(f) + " rejected by name without --wick");
        rejects({"flame", "--mode", "diffusion", f, "1"}, 2, f,
                std::string(f) + " rejected by name with --mode diffusion");
        rejects({"flame", f, "1"}, 2, f,
                std::string(f) + " rejected by name with --mode premixed");
        rejects({"flame", "--mode", "chamber", f, "1"}, 2, f,
                std::string(f) + " rejected by name with --mode chamber");
    }
    rejects({"flame", "--mode", "candle", "--soot", "on"}, 2, "--soot",
            "--soot rejected by name without --wick");
    for (const char* m : {"premixed", "bunsen", "diffusion", "chamber"})
        rejects({"flame", "--mode", m, "--wick"}, 2, "--wick",
                std::string("--wick rejected by name with --mode ") + m);
    for (const char* f : {"--slot-width", "--slot-edge", "--pilot-width",
                          "--pilot-speed", "--pulse", "--ignition"})
        rejects(wick({f, "1"}), 2, f,
                std::string(f) + " rejected by name with --wick");
    rejects(wick({"--pulse-freq", "1"}), 2, "--pulse-freq",
            "--pulse-freq rejected by name with --wick");
    rejects(wick({"--quiescent"}), 2, "--quiescent",
            "--quiescent rejected by name with --wick");
    rejects(wick({"--soot", "off", "--soot-form", "3"}), 2, "--soot-form",
            "--soot-form rejected with --soot off");
    rejects(wick({"--soot", "off", "--soot-ox", "3"}), 2, "--soot-ox",
            "--soot-ox rejected with --soot off");
    rejects(wick({"--wick-width", "0.25"}), 2, "--wick-width",
            "a wick under 4 cells rejected (two mirror cells per side)");
    rejects(wick({"--wick-height", "0.1"}), 2, "--wick-height",
            "a wick under 2 cells tall rejected");
    rejects(wick({"--wick-temp", "6"}), 2, "--wick-temp",
            "--wick-temp above sigma rejected");
    rejects(wick({"--wick-temp", "0.5"}), 2, "--wick-temp",
            "--wick-temp below T0 rejected");
    rejects(wick({"--fuel-rate", "0"}), 2, "--fuel-rate", "--fuel-rate 0 rejected");
    rejects(wick({"--fuel-rate", "nan"}), 1, "--fuel-rate",
            "--fuel-rate nan rejected (CLI)");
    rejects(wick({"--candle-width", "1"}), 2, "--candle-width",
            "a body no wider than the wick + 4 cells rejected");
    rejects(wick({"--candle-height", "0.25"}), 2, "--candle-height",
            "a body under 4 cells tall rejected");
    rejects(wick({"--length", "12"}), 2, "--length",
            "a box too short for body + wick + flame rejected");
    rejects(wick({"--soot-form", "0"}), 2, "--soot-form", "--soot-form 0 rejected");
    rejects(wick({"--soot-ox", "-1"}), 2, "--soot-ox", "--soot-ox < 0 rejected");
    rejects(wick({"--coflow-width", "1"}), 2, "--coflow-width",
            "--coflow-width under 2 l rejected");
    return dnstest::finish();
}
