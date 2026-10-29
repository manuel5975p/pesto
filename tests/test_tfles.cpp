// The thickened-flame model's host formulas (dns/tfles.hpp; docs/chem.md,
// "LES species and TFLES"):
//   Charlette 2002 efficiency: E(u' = 0) = 1, E(F = 1) = 1, E(Re -> 0) = 1,
//     monotone non-decreasing in u' and in Delta, 1 <= E <= F^beta; the C++
//     (double) equals the independent Python transcription
//     (tools/chem/tfles_eff.py -> tests/data/chem/tfles_charlette.csv) to
//     1e-12 relative on its 50 points; the FP32 instantiation (the kernel's)
//     within 2e-5 of the double one;
//   Colin's scaling: u'_Delta = uh (Delta/h)^(1/3), 0 -> 0;
//   the sensor normalisation tflesSensorLogMax against a dense FP64 scan.
#include "dns/tfles.hpp"
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef DNS_CHEM_DATA
#error "DNS_CHEM_DATA must be defined (tests/data/chem)"
#endif

using dns::charletteEfficiency;
using dnstest::check;

namespace {

void limits()
{
    std::printf("charlette limits\n");
    check(charletteEfficiency(4.0, 0.0, 10.0, 0.5) == 1.0, "E(u' = 0) == 1");
    check(charletteEfficiency(1.0, 5.0, 10.0, 0.5) == 1.0, "E(F = 1) == 1");
    check(charletteEfficiency(4.0, 5.0, 0.0, 0.5) == 1.0, "E(Re = 0) == 1");
    check(charletteEfficiency(4.0f, 0.0f, 10.0f, 0.5f) == 1.0f,
          "E(u' = 0) == 1 in FP32 (the kernel's planar laminar flame)");
    check(charletteEfficiency(4.0, 3.0, 1e-3, 0.5) == 1.0,
          "E(Re 1e-3) == 1 (f_Re underflows: Gamma 0, no NaN)");
    check(charletteEfficiency(4.0f, 3.0f, 1e-3f, 0.5f) == 1.0f,
          "E(Re 1e-3) == 1 in FP32");
    // Monotone in u' and in Delta, bounded by F^beta.
    bool monoU = true, monoD = true, bound = true;
    double worst = 0;
    for (double F : {1.5, 2.0, 3.0, 4.0, 6.0, 10.0}) {
        double prev = 0;
        for (int k = 0; k <= 200; ++k) {
            const double u = 0.05 * k;
            const double e = charletteEfficiency(F, u, u * F * 1.6 / 0.72, 0.5);
            monoU = monoU && e >= prev - 1e-14;
            bound = bound && e >= 1.0 && e <= std::pow(F, 0.5) * (1 + 1e-14);
            prev = e;
        }
    }
    for (double u : {0.1, 0.5, 2.0, 8.0}) {
        double prev = 0;
        for (int k = 0; k <= 200; ++k) {
            const double F = 1.0 + 0.05 * k;
            const double e = charletteEfficiency(F, u, u * F * 1.6 / 0.72, 0.5);
            monoD = monoD && e >= prev - 1e-14;
            worst = std::max(worst, e / std::pow(F, 0.5));
            prev = e;
        }
    }
    check(monoU, "E non-decreasing in u'_Delta (F 1.5 - 10, Re on nu_u)");
    check(monoD, "E non-decreasing in Delta/delta_L (u' 0.1 - 8)");
    check(bound, "1 <= E <= F^beta");
    std::printf("  max E / F^beta over the Delta scan: %.6f\n", worst);
}

void table()
{
    std::printf("charlette vs tools/chem/tfles_eff.py\n");
    std::ifstream in(std::string(DNS_CHEM_DATA) + "/tfles_charlette.csv");
    check(bool(in), "tfles_charlette.csv opened");
    std::string line;
    std::getline(in, line);
    int n = 0;
    double err = 0, err32 = 0;
    while (std::getline(in, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream s(line);
        double r, u, re, beta, xi;
        if (!(s >> r >> u >> re >> beta >> xi)) continue;
        const double e = charletteEfficiency(r, u, re, beta);
        err = std::max(err, std::abs(e - xi) / xi);
        const float e32 = charletteEfficiency(float(r), float(u), float(re),
                                              float(beta));
        err32 = std::max(err32, std::abs(double(e32) - e) / e);
        ++n;
    }
    std::printf("  %d points: max rel |C++ - Python| %.3e, FP32 vs FP64 %.3e\n",
                n, err, err32);
    check(n == 50, "50 table points");
    check(err <= 1e-12, "C++ == the Python transcription to 1e-12");
    check(err32 <= 2e-5, "the FP32 instantiation within 2e-5 of FP64");
}

void colin()
{
    std::printf("colin scaling\n");
    check(dns::colinUprime(0.0, 4.0, 1.0) == 0.0, "u'_Delta(0) == 0");
    check(std::abs(dns::colinUprime(2.0, 8.0, 1.0) - 4.0) < 1e-15,
          "u'_Delta = uh (Delta/h)^(1/3)");
}

void sensorNorm()
{
    std::printf("sensor normalisation\n");
    // sigma 5, theta 50: the sensor's reduced activation theta/2 = 25.
    for (double sig : {3.0, 5.0, 8.0})
        for (double ths : {12.5, 25.0, 40.0}) {
            const double lm = dns::tflesSensorLogMax(sig, ths);
            double best = -HUGE_VAL;
            const int N = 2000000;
            for (int k = 0; k < N; ++k) {
                const double t = 1.0 + (sig - 1.0) * (k + 0.5) / N;
                best = std::max(best, -ths / t - std::log(t) +
                                          std::log((sig - t) / (sig - 1.0)));
            }
            const bool ok = lm >= best - 1e-12 && lm - best < 1e-9;
            std::printf("  sigma %g thetaS %g: ln max %.12f (scan %.12f)\n", sig,
                        ths, lm, best);
            check(ok, "tflesSensorLogMax == the dense scan's max");
        }
}

} // namespace

int main()
{
    limits();
    table();
    colin();
    sensorNorm();
    return dnstest::finish();
}
