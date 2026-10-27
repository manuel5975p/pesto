// GPU twin of hit_mini: `dns hit` (the FP32 GPU path, through the built
// binary) at 64^3, h = 1.5 eta, 6 + 6 eddy turnovers, seed 7 -- the CPU
// test's case. The same bands: the fixed-eps controller gives a stationary
// <eps_visc>/eps_f and u' in [0.8, 1.2], and the compensated spectrum
// C = <E(k) k^{5/3} eps^{-2/3}> over k in [4, 10] lands in [0.7, 1.6] (the
// band's derivation is in test_hit_mini.cpp). The divergence band is FP32:
// measured max 8.5e-6 over every row (2026-10-26, seed 1), banded 1e-4.
// eps_f and T_e are recomputed here in box units, as the CPU test does, so a
// binary that mis-derives its own forcing target fails too. ~2 s.
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <numbers>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using dnstest::check;

int main()
{
    const std::string prefix = "gpu_hit_mini";
    const int rc = valtest::runBinary(
        "\"" DNS_BIN "\" hit --n 64 --spin 6 --stats 6 --seed 7 --backend gpu "
        "--quiet --emit stats --emit spectrum --out " + prefix);
    if (rc == 5) dnstest::skip("no GPU device");
    check(rc == 0, "dns hit --n 64 --spin 6 --stats 6: exits 0");
    if (rc != 0) return dnstest::finish();

    // Box units: L = 2 pi, u'_target = 1 (dns/units.hpp HitDesign).
    const double epsF = 1.0 / (0.19 * 2 * std::numbers::pi);
    const double tEddy = 1.5 / epsF, tSpin = 6.0 * tEddy, tEnd = 12.0 * tEddy;

    const auto st = valtest::loadCsv(prefix + "_stats.csv");
    check(st.has_value(), "stats CSV readable");
    if (!st) return dnstest::finish();
    const auto& t = st->col("t");
    const double eps = valtest::windowMean(t, st->col("epsVisc"), tSpin, tEnd);
    const double k = valtest::windowMean(t, st->col("kPrime"), tSpin, tEnd);
    const double uPrime = std::sqrt(2.0 * k / 3.0);
    double maxDiv = 0;
    for (double d : st->col("maxDiv")) maxDiv = std::max(maxDiv, d);

    std::vector<double> spec;
    int samples = 0;
    for (;; ++samples) {
        const auto s = valtest::loadCsv(prefix + "_spec_" + std::to_string(samples) +
                                        ".csv");
        if (!s) break;
        const auto& kk = s->col("k");
        const auto& E = s->col("E");
        if (spec.empty()) spec.assign(kk.size() + 1, 0.0);
        for (std::size_t r = 0; r < kk.size() && std::size_t(kk[r]) < spec.size(); ++r)
            spec[std::size_t(kk[r])] += E[r];
    }
    double C = 0;
    int nk = 0;
    for (int kk = 4; kk <= 10 && std::size_t(kk) < spec.size(); ++kk, ++nk)
        C += spec[std::size_t(kk)] / samples * std::pow(double(kk), 5.0 / 3.0) *
             std::pow(eps, -2.0 / 3.0);
    if (nk > 0) C /= nk;

    std::printf("rows %zu  eps/epsF %.4f  u'/target %.4f  maxDiv %.3e  C %.3f "
                "(%d spectra, k in [4,10])\n",
                st->rows(), eps / epsF, uPrime, maxDiv, C, samples);
    check(eps / epsF > 0.8 && eps / epsF < 1.2, "eps_visc/eps_f in [0.8, 1.2]");
    check(uPrime > 0.8 && uPrime < 1.2, "u'/target in [0.8, 1.2]");
    check(maxDiv < 1e-4, "FP32 maxDiv < 1e-4 on every row");
    check(samples >= 16 && nk == 7, "spectra written over the stats window");
    check(C > 0.7 && C < 1.6, "compensated spectrum C in [0.7, 1.6]");

    std::error_code ec;
    for (const char* s : {".json", "_stats.csv"})
        std::filesystem::remove(prefix + s, ec);
    for (int i = 0; i < samples; ++i)
        std::filesystem::remove(prefix + "_spec_" + std::to_string(i) + ".csv", ec);
    return dnstest::finish();
}
