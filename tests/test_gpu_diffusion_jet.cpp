// dns flame --mode diffusion end to end (docs/chem.md): the 2D slot-jet
// diffusion flame through the built binary.
//
//   jet    the defaults (W 16 l, s 1, sigma 5, beta 5, Da 1000, M 0.05, edge
//          3 l, outflow sigma 4), tt 80, window t in [60, 80]:
//            fuel balance: burn / (fuel through the first interior face)
//              within 1% (measured 0.999-1.003), nothing leaves the outflow
//              (closed tip: fuelOut/fuelIn < 1e-3);
//            steady: the tip's window halves agree to 1%;
//            attached: the base on the first rows (<= 2 h);
//            flame sheet: Tmax past the inflow band / T_ad in [0.95, 1.005]
//              (measured 0.986: the finite-rate deficit);
//            box isobaric: mean p / p0 within 1%;
//            bounds: Z, Y_F, Y_O in [-1e-6, 1 + 1e-6] (hits 0);
//            tip 51.1 l: a REGRESSION pin (+-5%), no reference -- the
//              constant-density Burke-Schumann 70.3 l is a scale only.
//   guard  beta 8 at Da 1000 extinguishes the IC sheet: exit 8 (measured
//          t = 1.8), within tt 10.
//
// Usage: test_gpu_diffusion_jet jet|guard. No HIP device -> skip (77).
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

int main(int argc, char** argv)
{
    using dnstest::check;
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "jet";
    const std::string prefix = "/tmp/dns_test_diffusion_" + mode;
    if (mode == "guard") {
        const int rc = valtest::runBinary(
            "\"" DNS_BIN "\" flame --mode diffusion --zeldovich 8 --tt 10 "
            "--emit stats --quiet --out " + prefix);
        std::printf("  beta 8, Da 1000: rc %d\n", rc);
        check(rc == 8, "the extinguished sheet exits 8");
        return dnstest::finish();
    }
    const int rc = valtest::runBinary("\"" DNS_BIN "\" flame --mode diffusion "
                                      "--tt 80 --emit stats --quiet --out " +
                                      prefix);
    check(rc == 0, "the default jet runs to tt 80 (exit 0)");
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    check(tbl.has_value(), "stats readable");
    if (rc != 0 || !tbl) return dnstest::finish();
    const auto& t = tbl->col("t");
    auto mean = [&](const char* k, double a, double b) {
        const auto& v = tbl->col(k);
        double s = 0;
        int n = 0;
        for (std::size_t i = 0; i < t.size(); ++i)
            if (t[i] >= a && t[i] <= b && std::isfinite(v[i])) {
                s += v[i];
                ++n;
            }
        return n ? s / n : NAN;
    };
    auto maxOf = [&](const char* k, double a) {
        double m = -HUGE_VAL;
        for (std::size_t i = 0; i < t.size(); ++i)
            if (t[i] >= a) m = std::max(m, tbl->col(k)[i]);
        return m;
    };
    const double T0 = 1 / (1.4 * 0.05 * 0.05), sigma = 5;
    const double burn = mean("burnRatio", 60, 80);
    const double out = mean("fuelOut", 60, 80) / mean("fuelIn", 60, 80);
    const double tipA = mean("xTip", 60, 70), tipB = mean("xTip", 70, 80);
    const double tip = mean("xTip", 60, 80), half = mean("half", 60, 80);
    const double tmax = mean("TmaxIn", 60, 80) / (sigma * T0);
    const double p = mean("pMean", 60, 80);
    const double base = maxOf("xBase", 60);
    const double hits = maxOf("hits", 0);
    double zmin = HUGE_VAL, zmax = -HUGE_VAL;
    for (std::size_t i = 0; i < t.size(); ++i) {
        zmin = std::min(zmin, tbl->col("zMin")[i]);
        zmax = std::max(zmax, tbl->col("zMax")[i]);
    }
    std::printf("  window [60, 80]: burn/in %.5f, out/in %.2e, tip %.4f "
                "(halves %.4f / %.4f), half-width %.4f, Tmax/T_ad %.5f, p/p0 "
                "%.5f, base %.3f, Z [%.3g, 1%+.2e], hits %g\n",
                burn, out, tip, tipA, tipB, half, tmax, p, base, zmin,
                zmax - 1, hits);
    check(std::abs(burn - 1) < 0.01, "fuel balance: burn/in within 1%");
    check(out < 1e-3, "closed tip: fuelOut/fuelIn < 1e-3");
    check(std::abs(tipA / tipB - 1) < 0.01, "steady: tip halves agree to 1%");
    check(base <= 0.5, "attached: base within the first two rows");
    check(tmax > 0.95 && tmax < 1.005, "Tmax / T_ad in [0.95, 1.005]");
    check(std::abs(p - 1) < 0.01, "isobaric: mean p/p0 within 1%");
    check(hits == 0 && zmin > -1e-6 && zmax < 1 + 1e-6,
          "Z, Y_F, Y_O in bounds (no hits)");
    check(std::abs(tip / 51.1 - 1) < 0.05, "tip 51.1 l +-5% (regression pin)");
    return dnstest::finish();
}
