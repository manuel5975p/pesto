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

std::vector<char> readAll(const std::string& path)
{
    std::vector<char> b;
    if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
        char buf[1 << 16];
        std::size_t n;
        while ((n = std::fread(buf, 1, sizeof buf, f)) > 0)
            b.insert(b.end(), buf, buf + n);
        std::fclose(f);
    }
    return b;
}

// extrude3d: the 2D slot Bunsen and the same burner at --nz 8 (z periodic,
// hz = h, the IC extruded, no --turb). A z-invariant state stays z-invariant
// bit for bit (every z difference is exactly 0), the 3D dt is the 2D one
// (stableDt is the max over axes), so the 3D run's mid-span xy frames equal
// the 2D frames byte for byte, and its span-mean cone (the mean of nz equal
// floats) the 2D cone: xTip, xBase, tanA/L/R, rms, sinRatio, mFlank exactly.
// Twice: the DNS, and TFLES F 2 where the 3D run adds LES sigma 1.35 (the
// sigma operator is exactly 0 on a 2D field) with --efficiency none.
void extrude3d()
{
    const std::string base =
        "\"" DNS_BIN "\" flame --mode bunsen --mach 0.02 --cells-per-delta 4 "
        "--slot-width 6 --coflow-width 3 --jet-speed 3 --tt 2 --emit stats "
        "--emit bin --fdt 0.5 --quiet ";
    struct Variant
    {
        const char *name, *both, *only3d;
    };
    const Variant vs[] = {
        {"dns", "", ""},
        {"tfles", "--tfles 2 ",
         "--les 1.35 --sgs sigma --efficiency none "}};
    for (const Variant& v : vs) {
        const std::string p2 = std::string("/tmp/dns_test_bunsen_x2_") + v.name;
        const std::string p3 = std::string("/tmp/dns_test_bunsen_x3_") + v.name;
        const int r2 = valtest::runBinary(base + v.both + "--out " + p2);
        const int r3 = valtest::runBinary(base + v.both + v.only3d +
                                          "--nz 8 --out " + p3);
        check(r2 == 0 && r3 == 0, std::string(v.name) + ": 2D and 3D runs exit 0");
        if (r2 != 0 || r3 != 0) continue;
        for (const char* f : {"temp", "fuel", "omega"}) {
            const auto a = readAll(p2 + "_" + f + ".bin");
            const auto b = readAll(p3 + "_" + f + ".bin");
            check(!a.empty() && a == b,
                  std::string(v.name) + ": " + f +
                      " frames (mid-span xy plane) byte-identical to 2D (" +
                      std::to_string(a.size()) + " / " +
                      std::to_string(b.size()) + " bytes)");
        }
        const auto t2 = valtest::loadCsv(p2 + "_stats.csv");
        const auto t3 = valtest::loadCsv(p3 + "_stats.csv");
        check(t2 && t3, std::string(v.name) + ": stats readable");
        if (!t2 || !t3) continue;
        for (const char* k : {"t", "dt", "xTip", "xBase", "tanA", "tanL",
                              "tanR", "rms", "sinRatio", "mFlank"}) {
            const auto& a = t2->col(k);
            const auto& b = t3->col(k);
            bool same = a.size() == b.size() && !a.empty();
            for (std::size_t i = 0; same && i < a.size(); ++i)
                same = (a[i] == b[i]) || (std::isnan(a[i]) && std::isnan(b[i]));
            check(same, std::string(v.name) + ": span-mean " + k +
                            " equals the 2D column exactly");
        }
        const double s2 = t2->col("SLflank").back(), s3 = t3->col("SLflank").back();
        check(std::abs(s3 / s2 - 1) < 1e-6,
              std::string(v.name) + ": SLflank per unit span equals 2D to 1e-6");
        std::printf("  %s: %zu stats rows, SLflank 2D %.9g 3D %.9g, ST %.6g, "
                    "area %.6g, I0 %.6g\n",
                    v.name, t2->col("t").size(), s2, s3, t3->col("ST").back(),
                    t3->col("area").back(), t3->col("I0").back());
    }
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
    } else if (mode == "extrude3d") {
        extrude3d();
    } else {
        std::fprintf(stderr, "usage: test_gpu_bunsen cone|anchor|extrude3d\n");
        return 2;
    }
    return dnstest::finish();
}
