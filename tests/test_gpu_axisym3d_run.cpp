// The full revolution end to end through the binary (docs/rans.md "The full
// revolution"), small grids, ~50 s:
//   (a) laminar Re 100, 32 x 24: --nz 36 against the 10-deg wedge (the same
//       discretization, Delta phi = 2 pi/36). With the DEFAULT trip it
//       decays: cd, theta_s and L_r equal the wedge's to the stop's
//       tolerance, |cl| -> rounding; with --ogrid-trip 0 the run is exactly
//       pi-symmetric (cl == cs == 0.0 on every row). The surface CSV has
//       nTheta x N rows, the stats carry lr
//   (b) laminar Re 250, 64 x 48 x 32: --ogrid-trip 0 certifies the exactly
//       symmetric (m = 1-unstable) state; the DEFAULT trip reaches the steady
//       m = 1 wake of the first bifurcation (Johnson & Patel 1999 Re 250:
//       cd ~ 0.70, cl ~ 0.062), the lift in the trip's plane
//   (c) --init-from: N 36 started from the converged 10-deg wedge certifies
//       the wedge's cd in fewer iterations than the cold start; a wedge of
//       another --ny, another --mach, another closure, a missing file are
//       input errors (exit 3) named in the message
#include "validation_util.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using dnstest::check;

namespace {

const std::string kBase =
    "\"" DNS_BIN "\" flow --physics compressible --body sphere --ogrid "
    "--ogrid-far 20 --steady 1e-4 --emit stats "
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
    std::string out;             // stdout + stderr
};

// keep: leave the artefacts (a later --init-from reads them).
Run run(const std::string& name, const std::string& args, int meridian,
        bool keep = false)
{
    Run r;
    const std::string log = name + ".log";
    const int rc = std::system((kBase + args + " --out " + name + " > " + log +
                                " 2>&1")
                                   .c_str());
    r.rc = rc == -1 || !WIFEXITED(rc) ? -1 : WEXITSTATUS(rc);
    {
        std::ifstream f(log);
        std::stringstream ss;
        ss << f.rdbuf();
        r.out = ss.str();
    }
    const auto st = valtest::loadCsv(name + "_stats.csv");
    const auto sf = valtest::loadCsv(name + "_surface.csv");
    if (r.rc == 0 && st && sf && st->rows() > 0) {
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
    if (!keep)
        for (const char* s : {"_stats.csv", "_surface.csv", ".json", "_mask.bin",
                              "_mesh.bin", "_final.bin"})
            std::filesystem::remove(name + s, ec);
    std::filesystem::remove(log, ec);
    return r;
}

const std::string kSmall =
    "--n 32 --ny 24 --re 100 --mach 0.2 --ogrid-beta 3 --steady-steps 80000";

void wedgeEquivalence()
{
    const Run w = run("ax3r_w10", kSmall + " --nz 1 --ogrid-wedge 10 --emit mesh",
                      1, true);
    const Run f = run("ax3r_n36", kSmall + " --nz 36", 0);
    const Run z = run("ax3r_n36t0", kSmall + " --nz 36 --ogrid-trip 0", 0);
    check(w.rc == 0 && f.rc == 0 && z.rc == 0,
          "(a) the 10-deg wedge and N 36 (default trip, and 0) certify");
    // Measured: wedge cd 1.08141 (77000 it); N 36 untripped 1.08139 (22200);
    // at TOL 1e-5 both 1.08139 / 1.08138, theta_s 125.61, L_r 0.9224.
    check(std::abs(f.cd / w.cd - 1) < 1e-4 && std::abs(z.cd / w.cd - 1) < 1e-4,
          "(a) N 36 cd == the 10-deg wedge's, tripped and untripped (1e-4)");
    check(std::abs(f.theta - w.theta) < 0.02 && std::abs(f.lr / w.lr - 1) < 1e-3,
          "(a) theta_s (0.02 deg) and L_r (0.1%) == the wedge's");
    check(std::abs(f.cl) < 1e-4 && std::abs(f.cs) < 1e-4,
          "(a) the default trip decays at Re 100 (|cl|, |cs| < 1e-4 at the "
          "stop)");
    check(f.surfRows == 32u * 36u, "(a) the surface CSV: nTheta x N real faces");
    check(z.clMax == 0.0 && z.csMax == 0.0,
          "(a) --ogrid-trip 0: cl == cs == 0.0 on every stats row (exact "
          "pi-symmetry)");
    check(f.out.find("[full revolution 36: odd-mode trip 0.05 U]") !=
                  std::string::npos &&
              z.out.find("[full revolution 36: untripped, exactly "
                         "pi-symmetric]") != std::string::npos,
          "(a) the run log names the resolved trip");
}

void firstBifurcation()
{
    const std::string g = "--n 64 --ny 48 --nz 32 --re 250 --mach 0.2 "
                          "--ogrid-beta 3 --steady-steps 300000";
    const Run s = run("ax3r_sym", g + " --ogrid-trip 0", 0);
    const Run t = run("ax3r_trip", g, 0);
    check(s.rc == 0 && s.clMax == 0.0 && s.csMax == 0.0,
          "(b) --ogrid-trip 0 at Re 250: certifies the exactly symmetric "
          "state (cl == cs == 0.0 on every row)");
    // Measured: untripped cd 0.69099 (36200 it); default trip cd 0.70690, cl
    // -0.0647, cs -6.0e-6 (108400 it).
    check(std::abs(s.cd / 0.69099 - 1) < 1e-3, "(b) untripped cd at its "
                                               "measured 0.69099 (0.1%)");
    check(t.rc == 0, "(b) default trip at Re 250: the asymmetric steady state "
                     "certifies");
    check(std::abs(std::abs(t.cl) / 0.0647 - 1) < 0.05 &&
              std::abs(t.cd / 0.70690 - 1) < 2e-3,
          "(b) default trip: the steady m = 1 wake, |cl| 0.0647 (5%), cd "
          "0.70690 (0.2%) -- JP99 ~0.062 / ~0.70");
    check(std::abs(t.cs) < 1e-3 * std::abs(t.cl),
          "(b) the lift lies in the trip's plane (|cs| < 1e-3 |cl|)");
}

void initFrom()
{
    const Run cold = run("ax3r_cold", kSmall + " --nz 36", 0);
    const Run i = run("ax3r_init", kSmall + " --nz 36 --init-from ax3r_w10", 0);
    check(i.rc == 0 && i.out.find("# initial state: the wedge ax3r_w10 "
                                  "revolved over 36 planes") !=
                           std::string::npos &&
              i.out.find("started from the wedge ax3r_w10]") != std::string::npos,
          "(c) --init-from: exit 0, the start reported and labelled");
    check(std::abs(i.cd / cold.cd - 1) < 1e-4,
          "(c) the wedge start certifies the cold start's cd (1e-4)");
    check(i.iter < cold.iter, "(c) ... in fewer iterations than the cold "
                              "start");
    struct Bad
    {
        std::string args, why, what;
    };
    const Bad bad[] = {
        {"--n 32 --ny 26 --re 100 --mach 0.2 --ogrid-beta 3 --nz 36",
         "--n and --ny must match",
         "another --ny"},
        {"--n 32 --ny 24 --re 100 --mach 0.1 --ogrid-beta 3 --nz 36",
         "--mach must match", "another --mach"},
        {kSmall + " --nz 36 --rans sa-neg", "--rans must match",
         "another closure"},
        {"--n 32 --ny 24 --re 100 --mach 0.2 --nz 36 --ogrid-beta 2.5",
         "--ogrid-beta, --ogrid-far and the body must match",
         "another --ogrid-beta"},
    };
    for (const auto& b : bad) {
        const Run r = run("ax3r_bad", b.args + " --init-from ax3r_w10", 0);
        check(r.rc == 3 && r.out.find(b.why) != std::string::npos,
              "(c) --init-from a wedge of " + b.what + ": exit 3, '" + b.why +
                  "'");
    }
    const Run m = run("ax3r_bad", kSmall + " --nz 36 --init-from ax3r_none", 0);
    check(m.rc == 3 && m.out.find("cannot read ax3r_none_mesh.bin") !=
                           std::string::npos,
          "(c) --init-from a missing run: exit 3, 'cannot read'");
    std::error_code ec;
    for (const char* s : {"_stats.csv", "_surface.csv", ".json", "_mask.bin",
                          "_mesh.bin", "_final.bin"})
        std::filesystem::remove(std::string("ax3r_w10") + s, ec);
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    wedgeEquivalence();
    firstBifurcation();
    initFrom();
    return dnstest::finish();
}
