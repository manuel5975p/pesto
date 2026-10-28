// dns flame --mode candle end to end (docs/chem.md, candle section): the
// diffusion flame under gravity through the built binary.
//
//   rest    --quiescent (no fuel, every inflow row at rest, the hydrostatic
//           IC), tt 10 (~37 free-fall times sqrt(W/g), ~55 acoustic box
//           crossings): max |u| over every stats row < 5e-3 U (measured
//           1.21e-3; 1.8e-4 of the free-fall scale 28.4 U), box-mean p/p_a
//           within 1e-4, no out-of-bounds cells. Without the NSCBC
//           hydrostatic waves the same box reached |u| 0.67 U (measured
//           2026-10-28 by zeroing them: the band has teeth).
//   g0      --mode candle --gravity 0 against --mode diffusion with the same
//           flags (W 4, 4 cells/l, ignition 4, tt 1): every one of the jet's
//           21 stats columns byte-identical (the candle's two extra columns
//           ignored).
//   height  the flame height (axis Z_st crossing) at W 0.75 and 1 (fuel
//           flow Q = W at U = 1, g 14: Fr 0.095 / 0.071), tt 30, window t in
//           [20, 30]: both closed and steady (window halves within 2%), the
//           exponent ln(H1/H0.75)/ln(4/3) in [1.15, 1.45] -- Roper's
//           buoyancy-controlled slot law H ~ Q^(4/3) within ~13%, the
//           momentum-controlled slot (H ~ Q^2 at fixed U) excluded. Measured
//           1.30 (H 7.4218 / 10.7853 l).
//
// Usage: test_gpu_candle rest|g0|height. No HIP device -> skip (77).
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

// The first n comma-separated fields of every line of a CSV, as TEXT.
std::vector<std::string> csvFields(const std::string& path, int n)
{
    std::ifstream f(path);
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string cell, row;
        for (int k = 0; k < n && std::getline(ss, cell, ','); ++k)
            row += cell + ",";
        out.push_back(row);
    }
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    using dnstest::check;
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "rest";
    const std::string bin = "\"" DNS_BIN "\" flame ";
    const std::string prefix = "/tmp/dns_test_candle_" + mode;
    if (mode == "rest") {
        const int rc = valtest::runBinary(bin + "--mode candle --quiescent "
                                                "--tt 10 --emit stats --quiet "
                                                "--out " + prefix);
        check(rc == 0, "the quiescent box runs to tt 10 (exit 0)");
        const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
        check(tbl.has_value(), "stats readable");
        if (rc != 0 || !tbl) return dnstest::finish();
        double umax = 0, pdev = 0, hits = 0;
        for (std::size_t i = 0; i < tbl->col("t").size(); ++i) {
            umax = std::max(umax, tbl->col("uMax")[i]);
            pdev = std::max(pdev, std::abs(tbl->col("pMean")[i] - 1));
            hits = std::max(hits, tbl->col("hits")[i]);
        }
        std::printf("  rest: max |u| %.4g U over %zu rows, max |p/p_a - 1| "
                    "%.3g, hits %g\n",
                    umax, tbl->col("t").size(), pdev, hits);
        check(umax < 5e-3, "hydrostatic box at rest: max |u| < 5e-3 U");
        check(pdev < 1e-4, "box-mean p on the ambient column within 1e-4");
        check(hits == 0, "no out-of-bounds cells");
        return dnstest::finish();
    }
    if (mode == "g0") {
        const std::string shared =
            " --slot-width 4 --cells-per-unit 4 --slot-edge 1 --length 16 "
            "--coflow-width 4 --ignition 4 --stoich 3.4 --ox-y 0.233 "
            "--damkohler 5000 --mach 0.05 --tt 1 --emit stats --quiet --out ";
        const int ra = valtest::runBinary(bin + "--mode candle --gravity 0" +
                                          shared + prefix + "_c");
        const int rb = valtest::runBinary(bin + "--mode diffusion" + shared +
                                          prefix + "_d");
        check(ra == 0 && rb == 0, "both runs exit 0");
        const auto a = csvFields(prefix + "_c_stats.csv", 21);
        const auto b = csvFields(prefix + "_d_stats.csv", 21);
        std::size_t diff = 0;
        for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i)
            diff += a[i] != b[i];
        std::printf("  g0: %zu / %zu rows, %zu differ\n", a.size(), b.size(),
                    diff);
        check(a.size() == b.size() && a.size() > 10 && diff == 0,
              "g = 0: the jet's 21 stats columns byte-identical");
        return dnstest::finish();
    }
    // height
    double tip[2] = {0, 0};
    const char* widths[2] = {"0.75", "1"};
    for (int w = 0; w < 2; ++w) {
        const std::string p = prefix + "_" + widths[w];
        const int rc = valtest::runBinary(bin + "--mode candle --slot-width " +
                                          widths[w] +
                                          " --tt 30 --emit stats --quiet "
                                          "--out " + p);
        check(rc == 0, std::string("W ") + widths[w] + " runs (exit 0)");
        const auto tbl = valtest::loadCsv(p + "_stats.csv");
        if (rc != 0 || !tbl) return dnstest::finish();
        const auto& t = tbl->col("t");
        const auto& x = tbl->col("xTip");
        double s[2] = {0, 0}, n[2] = {0, 0};
        bool closed = true;
        for (std::size_t i = 0; i < t.size(); ++i) {
            if (t[i] < 20) continue;
            closed = closed && std::isfinite(x[i]);
            const int k = t[i] < 25 ? 0 : 1;
            s[k] += x[i];
            n[k] += 1;
        }
        const double a = s[0] / n[0], b = s[1] / n[1];
        tip[w] = (s[0] + s[1]) / (n[0] + n[1]);
        std::printf("  W %s: tip %.4f l (halves %.4f / %.4f)\n", widths[w],
                    tip[w], a, b);
        check(closed, std::string("W ") + widths[w] + ": closed tip");
        check(std::abs(a / b - 1) < 0.02,
              std::string("W ") + widths[w] + ": steady (halves within 2%)");
    }
    const double e = std::log(tip[1] / tip[0]) / std::log(1.0 / 0.75);
    std::printf("  height exponent H ~ Q^%.4f (Roper buoyant slot 4/3, "
                "momentum slot 2)\n",
                e);
    check(e > 1.15 && e < 1.45,
          "H ~ Q^e with e in [1.15, 1.45]: buoyancy-controlled (4/3)");
    return dnstest::finish();
}
