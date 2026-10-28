// dns flame --mode chamber end to end (docs/chem.md, "Chamber"), through the
// binary and its stats CSV. Two runs at the defaults (4 cells per delta_th,
// M 0.02):
//
// (a) INERT (--reaction off, tt 8): the walls are impermeable for the
//     species. fuel(t) - fuel(0) == injCum (the FP64 time integral of the
//     injection) while nothing reaches the domain faces (outDom == 0
//     exactly). Measured 2026-10-28 (gfx1100): the residual grows steadily
//     to 3.3e-5 of fuel0 + injected at t = 8 -- the FP32 ABSORPTION bias of the
//     injector's constant per-stage increment (~1.5e-3 into rho Y ~ 1: one
//     dt recompute keeps the increment, hence the same rounding, for 16
//     steps), not a leak: a diffusive wall flux against an empty ghost would
//     move ~1e2 per unit time here, 1e4 x the band. Band 1e-4.
//     Hot IC gas only ever compresses: Tmax <= Tb (p/p0)^(2/7) (measured max
//     ratio 1.0013; band 1.005).
// (b) REACTING (tt 40, means over t in [25, 40]): anchored (exit 0: no
//     flash-back, no blow-off), and the global balances, measured
//     burn/inj 0.9943, mdotNoz/mdotIn 0.9949, q burn / (h0Noz - mdot cp Tu)
//     1.0018 (bands 3%, 2%, 2%), no fuel through the exit, no burn in the
//     injector. The nozzle against the 1D isentropic estimate from the
//     measured mdot, T0 and pAmb over the exit height He: exit Mach 0.3844
//     vs 0.3808 (band 3%); the measured total-pressure rise is 1.449 x the
//     isentropic one -- a discharge coefficient Cd = mdot / mdot_ideal(p0Noz)
//     of 0.832 for this Re_e ~ 100 nozzle (band [0.75, 0.95]).
#include "dns/chamber.hpp"
#include "dns/config.hpp"
#include "dns/transport.hpp"
#include "check.hpp"
#include "cli_test_util.hpp"
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

using dnstest::check;

std::optional<valtest::CsvTable> run(const std::string& name,
                                     const std::string& args)
{
    const std::string prefix = "/tmp/dns_test_chamber_" + name;
    const std::string cmd = "\"" DNS_BIN "\" flame --mode chamber " + args +
                            " --emit stats --quiet --out " + prefix;
    const int rc = valtest::runBinary(cmd);
    check(rc == 0, name + ": run exits 0 (rc " + std::to_string(rc) + ")");
    if (rc != 0) return std::nullopt;
    auto t = valtest::loadCsv(prefix + "_stats.csv");
    check(t.has_value(), name + ": stats CSV loads");
    return t;
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    auto cs = dnstest::runCli({"flame", "--mode", "chamber"});
    if (!cs) {
        check(false, "the default chamber case validates");
        return dnstest::finish();
    }
    const double Tu = cs->flameTu, Tb = 5 * Tu, He = cs->chamber.He;
    const double gam = dns::kGammaAir, cp = gam / (gam - 1);

    // (a) inert ---------------------------------------------------------------
    if (auto tb = run("inert", "--reaction off --tt 8")) {
        const auto& t = tb->col("t");
        const auto& fuel = tb->col("fuel");
        const auto& inj = tb->col("injCum");
        const auto& od = tb->col("outDom");
        const auto& burn = tb->col("burn");
        const auto& tmax = tb->col("Tmax");
        const auto& pch = tb->col("pCh");
        const auto& p0n = tb->col("p0Noz");
        double res = 0, odMax = 0, bMax = 0, tRatio = 0;
        for (std::size_t i = 0; i < t.size(); ++i) {
            res = std::max(res, std::abs(fuel[i] - fuel[0] - inj[i]) /
                                    (fuel[0] + inj[i]));
            odMax = std::max(odMax, std::abs(od[i]));
            bMax = std::max(bMax, burn[i]);
            const double p = std::max(pch[i], p0n[i]);
            tRatio = std::max(tRatio, tmax[i] / (Tb * std::pow(p / Tu, 2.0 / 7)));
        }
        std::printf("  inert: max |fuel - fuel0 - injCum| / (fuel0 + injCum) "
                    "%.3e, injected %.6g, outDom max %.3e, Tmax / (Tb "
                    "(p/p0)^(2/7)) max %.6f\n",
                    res, inj.back(), odMax, tRatio);
        check(inj.back() > 500, "inert: the injector delivered > 500");
        check(bMax == 0, "inert: nothing burns (--reaction off)");
        check(odMax < 1e-20, "inert: no fuel reaches a domain face");
        check(res < 1e-4, "inert: fuel = fuel0 + injected to 1e-4 (walls "
                          "impermeable; FP32 absorption bias 3.3e-5)");
        check(tRatio < 1.005, "inert: hot gas only compresses: Tmax <= Tb "
                              "(p/p0)^(2/7) (+0.5%)");
    }

    // (b) reacting --------------------------------------------------------------
    if (auto tb = run("react", "--tt 40")) {
        const auto& t = tb->col("t");
        constexpr double t0 = 25, t1 = 40;
        auto mean = [&](const char* k) {
            return valtest::windowMean(t, tb->col(k), t0, t1);
        };
        const double inj = mean("inj"), burn = mean("burn");
        const double mIn = mean("mdotIn"), mNoz = mean("mdotNoz");
        const double h0 = mean("h0Noz"), burnCav = mean("burnCav");
        const double pa = mean("pAmb"), p0n = mean("p0Noz"), pch = mean("pCh");
        const double Me = mean("Me");
        const double q = cp * 4 * Tu;
        const double eRatio = q * burnCav / (h0 - mIn * cp * Tu);
        const double T0 = h0 / (mNoz * cp);
        const auto iso = dns::isentropicNozzle(mNoz, T0, He, pa, gam, 1.0);
        check(iso.has_value(), "react: the isentropic estimate exists");
        const double MeIso = iso ? iso->Me : NAN;
        const double dpRatio = iso ? (p0n - pa) / (iso->p0 - pa) : NAN;
        // Ideal flux at the MEASURED total pressure.
        const double Mi = std::sqrt(2 / (gam - 1) *
                                    (std::pow(p0n / pa, (gam - 1) / gam) - 1));
        const double mIdeal =
            pa * Mi * std::sqrt(gam * (1 + 0.5 * (gam - 1) * Mi * Mi) / T0) * He;
        const double Cd = mNoz / mIdeal;
        double outMax = 0, injMax = 0;
        const auto& on = tb->col("outNoz");
        const auto& oi = tb->col("omegaInj");
        for (std::size_t i = 0; i < t.size(); ++i)
            if (t[i] >= t0) {
                outMax = std::max(outMax, on[i]);
                injMax = std::max(injMax, oi[i]);
            }
        std::printf("  react [%g, %g]: burn/inj %.6f  mdotNoz/mdotIn %.6f  "
                    "energy %.6f  T0/Tb %.6f\n"
                    "    pCh/pAmb %.6f  p0Noz/pAmb %.6f  isentropic p0/pa "
                    "%.6f  dp ratio %.4f  Cd %.4f\n"
                    "    Me %.5f vs isentropic %.5f  outNoz max %.2e  "
                    "omegaInj max %.2e\n",
                    t0, t1, burn / inj, mNoz / mIn, eRatio, T0 / Tb, pch / pa,
                    p0n / pa, iso ? iso->p0 / pa : NAN, dpRatio, Cd, Me, MeIso,
                    outMax, injMax);
        check(std::abs(burn / inj - 1) < 0.03, "react: burn / inj within 3%");
        check(std::abs(mNoz / mIn - 1) < 0.02, "react: nozzle mdot = injected "
                                               "within 2%");
        check(std::abs(eRatio - 1) < 0.02, "react: heat release = exit "
                                           "enthalpy rise within 2%");
        check(outMax < 1e-6 * inj, "react: no fuel leaves through the nozzle");
        check(injMax < 1e-6, "react: no burn in the injector band");
        check(pch / pa > 1.1, "react: the chamber is pressurized (> 1.1 pAmb)");
        check(std::abs(Me / MeIso - 1) < 0.03,
              "react: exit Mach = isentropic estimate within 3%");
        check(Cd > 0.75 && Cd < 0.95, "react: discharge coefficient in "
                                      "[0.75, 0.95] (measured 0.832)");
    }
    return dnstest::finish();
}
