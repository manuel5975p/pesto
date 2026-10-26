// The full revolution end to end through the binary (docs/rans.md "The full
// revolution"), small grids, ~40 s:
//   (a) laminar Re 100, 32 x 24: --nz 36 against the 10-deg wedge (the same
//       discretization, Delta phi = 2 pi/36): both certify, cd, theta_s and
//       L_r agree to the stop's tolerance; the surface CSV has nTheta x N
//       rows, cl == cs == 0.0 on every row (exact pi-symmetry), the stats
//       carry lr
//   (b) laminar Re 250, 64 x 48 x 32: untripped, the exactly symmetric state
//       (cl == 0.0 on every row); with --ogrid-trip 0.05, the steady m = 1
//       wake of the first bifurcation (Johnson & Patel 1999 Re 250: cd ~
//       0.70, cl ~ 0.062): certified, |cl| at its measured value, the lift
//       in the trip's plane (cs ~ 0)
#include "validation_util.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using dnstest::check;

namespace {

const std::string kBase =
    "\"" DNS_BIN "\" flow --physics compressible --body sphere --ogrid "
    "--mach 0.2 --ogrid-far 20 --ogrid-beta 3 --steady 1e-4 --emit stats "
    "--emit surface ";

// First cf sign change from the front on meridian k (linear interpolation).
double sepAngle(const valtest::CsvTable& s, int k)
{
    const auto& kk = s.col("k");
    const auto& th = s.col("theta_deg");
    const auto& cf = s.col("cf");
    std::vector<std::pair<double, double>> line;
    for (std::size_t r = 0; r < kk.size(); ++r)
        if (int(kk[r]) == k) line.emplace_back(th[r], cf[r]);
    std::ranges::sort(line);
    for (std::size_t i = 1; i < line.size(); ++i)
        if (line[i - 1].second > 0 && line[i].second <= 0)
            return line[i - 1].first + (line[i].first - line[i - 1].first) *
                                           line[i - 1].second /
                                           (line[i - 1].second - line[i].second);
    return std::nan("");
}

struct Run
{
    int rc = -1;
    double cd = 0, lr = 0, theta = 0, iter = 0;
    double clMax = 0, csMax = 0; // max |cl|, |cs| over every stats row
    double cl = 0, cs = 0;       // last row
    std::size_t surfRows = 0;
};

Run run(const std::string& name, const std::string& args, int meridian)
{
    Run r;
    r.rc = valtest::runBinary(kBase + args + " --out " + name);
    const auto st = valtest::loadCsv(name + "_stats.csv");
    const auto sf = valtest::loadCsv(name + "_surface.csv");
    if (st && sf && st->rows() > 0) {
        r.cd = st->col("cd").back();
        r.lr = st->col("lr").back();
        r.iter = st->col("iter").back();
        r.cl = st->col("cl").back();
        r.cs = st->col("cs").back();
        for (double v : st->col("cl")) r.clMax = std::max(r.clMax, std::abs(v));
        for (double v : st->col("cs")) r.csMax = std::max(r.csMax, std::abs(v));
        r.theta = sepAngle(*sf, meridian);
        r.surfRows = sf->rows();
    }
    std::printf("%s: rc %d, %.0f iterations, cd %.5f, cl %+.4e, cs %+.2e, "
                "theta_s %.2f, L_r %.4f, %zu surface rows\n",
                name.c_str(), r.rc, r.iter, r.cd, r.cl, r.cs, r.theta, r.lr,
                r.surfRows);
    std::error_code ec;
    for (const char* s : {"_stats.csv", "_surface.csv", ".json", "_mask.bin"})
        std::filesystem::remove(name + s, ec);
    return r;
}

void wedgeEquivalence()
{
    const std::string g = "--n 32 --ny 24 --re 100 --steady-steps 80000";
    const Run w = run("ax3r_w10", g + " --nz 1 --ogrid-wedge 10", 1);
    const Run f = run("ax3r_n36", g + " --nz 36", 0);
    check(w.rc == 0 && f.rc == 0, "(a) the 10-deg wedge and N 36 both certify");
    // Measured at TOL 1e-5 (240000 iterations): cd 1.08139 / 1.08138,
    // theta_s 125.61 / 125.61, L_r 0.9224 / 0.9224; at TOL 1e-4 the stops
    // land on different iterations (22200 / 77000): cd 1.08139 / 1.08141.
    check(std::abs(f.cd / w.cd - 1) < 1e-4,
          "(a) N 36 cd == the 10-deg wedge's (1e-4, the TOL 1e-4 stops)");
    check(std::abs(f.theta - w.theta) < 0.02 && std::abs(f.lr / w.lr - 1) < 1e-3,
          "(a) theta_s (0.02 deg) and L_r (0.1%) == the wedge's");
    check(f.surfRows == 32u * 36u, "(a) the surface CSV: nTheta x N real faces");
    check(f.clMax == 0.0 && f.csMax == 0.0,
          "(a) cl == cs == 0.0 on every stats row (exact pi-symmetry)");
}

void firstBifurcation()
{
    const std::string g = "--n 64 --ny 48 --nz 32 --re 250 --steady-steps 300000";
    const Run s = run("ax3r_sym", g, 0);
    const Run t = run("ax3r_trip", g + " --ogrid-trip 0.05", 0);
    check(s.rc == 0 && s.clMax == 0.0 && s.csMax == 0.0,
          "(b) untripped Re 250: certifies the exactly symmetric state (cl == "
          "cs == 0.0 on every row)");
    // Measured: untripped cd 0.69099; tripped cd 0.70690, cl -0.0647, cs
    // -6.0e-6, 108400 iterations.
    check(std::abs(s.cd / 0.69099 - 1) < 1e-3, "(b) untripped cd at its "
                                               "measured 0.69099 (0.1%)");
    check(t.rc == 0, "(b) tripped Re 250: the asymmetric steady state certifies");
    check(std::abs(std::abs(t.cl) / 0.0647 - 1) < 0.05 &&
              std::abs(t.cd / 0.70690 - 1) < 2e-3,
          "(b) tripped: the steady m = 1 wake, |cl| 0.0647 (5%), cd 0.70690 "
          "(0.2%) -- JP99 ~0.062 / ~0.70");
    check(std::abs(t.cs) < 1e-3 * std::abs(t.cl),
          "(b) the lift lies in the trip's plane (|cs| < 1e-3 |cl|)");
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    wedgeEquivalence();
    firstBifurcation();
    return dnstest::finish();
}
