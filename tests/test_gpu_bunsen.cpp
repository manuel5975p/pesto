// dns flame --mode bunsen end to end (docs/chem.md): the 2D slot Bunsen cone
// against the flame it is built from. Driven through the built binary.
//
//   cone    U_jet 3 and 5, 8 cells/delta_th, M 0.02, tt 30, window t in
//           [20, 30] (steady: the halves agree). Anchored on the burnt
//           coflow (base on the inflow row, exit 0); the flank (middle half
//           of the cone) measured three ways:
//             sinRatio = sin(alpha) U_jet / S_L, the textbook cone angle.
//               It reads HIGH by the fresh flow's deceleration ahead of the
//               flank (measured +2.8% / +4.7% at U 3 / 5, 8 cells; +2.1% at
//               U 3, 16 cells: a flow effect, not a grid error). Band [1, 1.07].
//             SLflank = omega over the flank rows / flank length (the
//               consumption speed): 0.979 / 1.007; 16 cells U 3: 0.978.
//               Band 3%.
//             mFlank = rho u.n at the Tmid isotherm (the kinematic mass
//               burning flux): 0.989 / 1.008. Band 2%.
//           Fuel balance burn / (inflow rho u Y) within 1%; the cone is
//           symmetric (tanL == tanR to 1e-4; FP32 gives ~1e-5); Y in
//           [-1e-6, 1 + 1e-5].
//   anchor  the measured anchoring map at U_jet 5: a cold inert coflow
//           blows off (exit 8 within tt 5); a cold coflow with 3 delta_th
//           burnt strips at the jet speed anchors (exit 0 over tt 30, base on
//           the inflow row, flank mass flux within 5%, fuel within 3%).
//
// Usage: test_gpu_bunsen cone|anchor. No HIP device -> skip (77).
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

using dnstest::check;

struct Cone
{
    int rc = -1;
    bool ok = false;
    double sinRatio = 0, sinA = 0, sinB = 0, SLflank = 0, mFlank = 0;
    double burn = 0, xBase = 0, h = 0, tanL = 0, tanR = 0;
    double yMin = 0, yMax = 0;
};

// Window means of the stats rows with t in [t0, t1], and the halves of
// sinRatio.
Cone runCase(const std::string& name, const std::string& args, double t0,
             double t1)
{
    Cone c;
    const std::string prefix = "/tmp/dns_test_bunsen_" + name;
    const std::string cmd = "\"" DNS_BIN "\" flame --mode bunsen --mach 0.02 " +
                            args + " --emit stats --quiet --out " + prefix;
    c.rc = valtest::runBinary(cmd);
    if (c.rc != 0) return c;
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    if (!tbl) return c;
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
    const double tm = 0.5 * (t0 + t1);
    c.sinRatio = mean("sinRatio", t0, t1);
    c.sinA = mean("sinRatio", t0, tm);
    c.sinB = mean("sinRatio", tm, t1);
    c.SLflank = mean("SLflank", t0, t1);
    c.mFlank = mean("mFlank", t0, t1);
    c.burn = mean("burnRatio", t0, t1);
    c.xBase = tbl->col("xBase").back();
    c.tanL = tbl->col("tanL").back();
    c.tanR = tbl->col("tanR").back();
    c.yMin = *std::min_element(tbl->col("yMin").begin(), tbl->col("yMin").end());
    c.yMax = *std::max_element(tbl->col("yMax").begin(), tbl->col("yMax").end());
    c.ok = std::isfinite(c.sinRatio) && std::isfinite(c.SLflank) &&
           std::isfinite(c.mFlank);
    std::printf("  %-6s sin(a)U/SL %.5f (halves %.5f / %.5f)  SLflank %.5f  "
                "mFlank %.5f  burn/in %.5f  base %.4f  tanL %.6f tanR %.6f  "
                "Y [%.3g, 1%+.2e]\n",
                name.c_str(), c.sinRatio, c.sinA, c.sinB, c.SLflank, c.mFlank,
                c.burn, c.xBase, c.tanL, c.tanR, c.yMin, c.yMax - 1);
    return c;
}

} // namespace

int main(int argc, char** argv)
{
    dns::test::requireDeviceOrSkip();
    const std::string mode = argc > 1 ? argv[1] : "cone";
    const double h8 = 1.60165 / 8;
    if (mode == "cone") {
        for (const char* U : {"3", "5"}) {
            const std::string n = std::string("U") + U;
            const Cone c = runCase(n, std::string("--tt 30 --jet-speed ") + U,
                                   20, 30);
            check(c.rc == 0 && c.ok, n + ": anchored run exits 0, stats finite");
            check(c.xBase < h8, n + ": base on the inflow row (anchored)");
            check(std::abs(c.sinA - c.sinB) < 5e-3,
                  n + ": steady (sinRatio window halves agree to 5e-3)");
            check(c.sinRatio >= 1 && c.sinRatio < 1.07,
                  n + ": sin(alpha) U_jet / S_L in [1, 1.07] (flow divergence "
                      "reads high)");
            check(std::abs(c.SLflank - 1) < 0.03,
                  n + ": flank consumption speed within 3% of S_L");
            check(std::abs(c.mFlank - 1) < 0.02,
                  n + ": flank mass burning flux within 2% of rho_u S_L");
            check(std::abs(c.burn - 1) < 0.01,
                  n + ": burn / fuel inflow within 1% (complete combustion)");
            // FP32 rounding breaks the mirror symmetry slightly: measured
            // |tanL/tanR - 1| = 1.1e-5 (U 3) and 4.7e-6 (U 5) at t = 30.
            check(std::abs(c.tanL - c.tanR) <= 1e-4 * std::abs(c.tanL),
                  n + ": symmetric cone (tanL == tanR to 1e-4)");
            check(c.yMin >= -1e-6 && c.yMax <= 1 + 1e-5,
                  n + ": Y in [-1e-6, 1 + 1e-5]");
        }
    } else if (mode == "anchor") {
        const Cone cold = runCase("cold", "--tt 5 --coflow-temp 1", 0, 5);
        check(cold.rc == 8, "cold inert coflow: not anchored, exit 8");
        const Cone p = runCase("strip",
                               "--tt 30 --coflow-temp 1 --pilot-width 3 "
                               "--pilot-speed 5",
                               25, 30);
        check(p.rc == 0 && p.ok, "fast burnt strips + cold coflow: exit 0");
        check(p.xBase < h8, "strips: base on the inflow row (anchored)");
        check(std::abs(p.mFlank - 1) < 0.05,
              "strips: flank mass burning flux within 5%");
        check(std::abs(p.burn - 1) < 0.03, "strips: burn / fuel inflow within 3%");
    } else {
        std::fprintf(stderr, "usage: test_gpu_bunsen cone|anchor\n");
        return 2;
    }
    return dnstest::finish();
}
