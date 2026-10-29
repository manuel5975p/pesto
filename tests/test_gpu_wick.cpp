// dns flame --mode candle --wick end to end (docs/chem.md, "Wick candle")
// through the built binary.
//
//   inert  --damkohler 1e-30 --soot off, tt 1: no reaction, the fuel source
//          on, the walled two-reactant kernel. Fuel inventory = IC fuel +
//          injected - what leaves the box, integrated from the stats rows:
//          the walls (zero species flux, solid cells never move) leak
//          nothing. Band 1e-3 of the inventory (FP32 absorption of the
//          small source increment, the chamber's C5 precedent).
//   flame  the defaults, tt 6, window t in [4, 6]: anchored (base within
//          1.5 l of the wax, Tmax above the mid temperature), a closed tip
//          3-10 l above the wick, the fuel balance closing on its time
//          integral after the start (t > 2) to 2e-3 of the fuel injected,
//          the soot balance likewise to 1e-2 of the soot formed, soot formed
//          (1-20% of the fuel) and burned out at the tip (oxidised/formed >
//          0.9, out of the top/formed < 1e-2: a NON-smoking candle), the
//          flame heating the wick (qWick > 0, < 20% of Q q).
//
// Usage: test_gpu_wick inert|flame. No HIP device -> skip (77).
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

// Cumulative trapezoid integral of y(t) at every row.
std::vector<double> cumtrapz(const std::vector<double>& t,
                             const std::vector<double>& y)
{
    std::vector<double> out(t.size(), 0.0);
    for (std::size_t i = 1; i < t.size(); ++i)
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (t[i] - t[i - 1]);
    return out;
}

std::size_t firstAt(const std::vector<double>& t, double t0)
{
    std::size_t i = 0;
    while (i + 1 < t.size() && t[i] < t0) ++i;
    return i;
}

} // namespace

int main(int argc, char** argv)
{
    using dnstest::check;
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "flame";
    const std::string bin = "\"" DNS_BIN "\" flame --mode candle --wick ";
    const std::string prefix = "/tmp/dns_test_wick_" + mode;
    if (mode == "inert") {
        const int rc = valtest::runBinary(
            bin + "--damkohler 1e-30 --soot off --tt 1 --emit stats --quiet "
                  "--out " + prefix);
        check(rc == 0, "the inert wick runs to tt 1 (exit 0)");
        const auto tb = valtest::loadCsv(prefix + "_stats.csv");
        check(tb.has_value(), "stats readable");
        if (rc != 0 || !tb) return dnstest::finish();
        const auto& t = tb->col("t");
        const auto out = cumtrapz(t, tb->col("fuelOut"));
        const auto burn = cumtrapz(t, tb->col("burn"));
        const auto& fuel = tb->col("fuel");
        const auto& inj = tb->col("fuelInjCum");
        double worst = 0;
        for (std::size_t i = 0; i < t.size(); ++i)
            worst = std::max(worst, std::abs(fuel[i] - fuel[0] - inj[i] +
                                             out[i] + burn[i]));
        const double scale = fuel[0] + inj.back();
        std::printf("  inert: %zu rows, fuel %.6g -> %.6g, injected %.6g, out "
                    "%.3g, burnt %.3g, max |imbalance| %.3g (%.3g of the "
                    "inventory)\n",
                    t.size(), fuel[0], fuel.back(), inj.back(), out.back(),
                    burn.back(), worst, worst / scale);
        check(burn.back() < 1e-12 * scale, "inert: nothing burns");
        check(worst < 1e-3 * scale,
              "inert: fuel = IC + injected - out (no leak through the wall)");
        return dnstest::finish();
    }
    const int rc = valtest::runBinary(bin + "--tt 6 --emit stats --quiet --out " +
                                      prefix);
    check(rc == 0, "the default wick candle runs to tt 6 (exit 0)");
    const auto tb = valtest::loadCsv(prefix + "_stats.csv");
    check(tb.has_value(), "stats readable");
    if (rc != 0 || !tb) return dnstest::finish();
    const auto& t = tb->col("t");
    const double xTop = 7.0, T0 = 1.0 / (1.4 * 0.005 * 0.005), Q = 1.0;
    const auto burnI = cumtrapz(t, tb->col("burn"));
    const auto formI = cumtrapz(t, tb->col("sootForm"));
    const auto oxI = cumtrapz(t, tb->col("sootOx"));
    const auto outI = cumtrapz(t, tb->col("fuelOut"));
    const auto sOutI = cumtrapz(t, tb->col("sootOut"));
    const auto& fuel = tb->col("fuel");
    const auto& soot = tb->col("soot");
    const auto& inj = tb->col("fuelInjCum");
    // Integral balances after the start transient (the IC sheet's fast burn
    // is under-sampled by the 200-step stats cadence): t in [2, 6].
    const std::size_t a = firstAt(t, 2.0), b = t.size() - 1;
    auto fres = [&](std::size_t i) {
        return fuel[i] - inj[i] + burnI[i] + formI[i] + outI[i];
    };
    auto sres = [&](std::size_t i) {
        return soot[i] - formI[i] + oxI[i] + sOutI[i];
    };
    const double dFuel = fres(b) - fres(a), dInj = inj[b] - inj[a];
    const double dSoot = sres(b) - sres(a), dForm = formI[b] - formI[a];
    // Window t in [4, 6].
    const std::size_t w0 = firstAt(t, 4.0);
    double n = 0, ox = 0, sout = 0, form = 0, h = 0, hMin = HUGE_VAL,
           hMax = -HUGE_VAL, baseMax = 0, tMin = HUGE_VAL, qw = 0, qr = 0,
           fv = 0, hits = 0, dtS = HUGE_VAL, dt = 0;
    bool closed = true;
    for (std::size_t i = w0; i <= b; ++i) {
        n += 1;
        ox += tb->col("sootOx")[i];
        sout += tb->col("sootOut")[i];
        form += tb->col("sootForm")[i];
        const double hh = tb->col("height")[i];
        closed = closed && std::isfinite(hh);
        h += hh;
        hMin = std::min(hMin, hh);
        hMax = std::max(hMax, hh);
        baseMax = std::max(baseMax, tb->col("xBase")[i] - xTop);
        tMin = std::min(tMin, tb->col("Tmax")[i] / T0);
        qw += tb->col("qWick")[i];
        qr += tb->col("qWickRatio")[i];
        fv = std::max(fv, tb->col("fvMax")[i]);
        hits = std::max(hits, tb->col("hits")[i]);
        dtS = std::min(dtS, tb->col("dtSoot")[i]);
        dt = std::max(dt, tb->col("dt")[i]);
    }
    std::printf("  flame: %zu rows; window [4, 6]: height %.4f l (%.4f .. "
                "%.4f), base %.3f l above the wax, min Tmax %.4f T0\n",
                t.size(), h / n, hMin, hMax, baseMax, tMin);
    std::printf("  fuel balance t in [2, 6]: %.4g of the %.4g injected; soot "
                "balance %.4g of the %.4g formed\n",
                dFuel / dInj, dInj, dSoot / dForm, dForm);
    std::printf("  soot: formed %.4g of the fuel, oxidised/formed %.5f, "
                "out/formed %.3g, peak f_v %.3g ppm; qWick %.4g (%.4f of Q q); "
                "hits %g; min dtSoot %.3g vs dt %.3g\n",
                form / n / Q, ox / form, sout / form, fv, qw / n, qr / n, hits,
                dtS, dt);
    check(closed && h / n > 3 && h / n < 10,
          "a closed tip 3-10 l above the wick");
    check(baseMax < 1.5 && tMin > 3.0,
          "anchored: base within 1.5 l of the wax, Tmax above (1 + sigma)/2");
    check(std::abs(dFuel) < 2e-3 * dInj,
          "fuel: inventory change = injected - burnt - into soot - out");
    check(std::abs(dSoot) < 1e-2 * dForm,
          "soot: inventory change = formed - oxidised - out");
    check(form / n / Q > 0.01 && form / n / Q < 0.2,
          "soot formed: 1-20% of the fuel");
    check(ox / form > 0.9 && sout / form < 1e-2,
          "soot burned out at the tip (a non-smoking candle)");
    check(qw > 0 && qr / n < 0.2, "the flame heats the wick (< 20% of Q q)");
    return dnstest::finish();
}
