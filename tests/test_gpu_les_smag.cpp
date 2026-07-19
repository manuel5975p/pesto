// Coverage for the app-local Smagorinsky LES path in cylinder_tunnel_gpu
// (--les Cs, 2D only). The kernels (kSmagNut/kEddyDiffuse) live in the app's
// anonymous namespace, so we exercise them by driving the built binary and
// reading its stats CSV (columns: t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters).
//
// Checks:
//   1. --les 0 stats == no --les stats (default-off is a no-op).
//   2. --les 0.2 completes (exit 0), final K finite and STRICTLY LOWER than
//      the --les 0 final K at identical settings (dissipation has the right
//      sign).
//   3. --les with --nz > 1 exits nonzero (2D-only guard).
//
// No HIP device -> skip (exit 77). The app itself needs a device, so device
// presence is probed here up front via the same predicate the app uses.

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>

#ifndef CYLINDER_BIN
#error "CYLINDER_BIN must be defined (path to cylinder_tunnel_gpu binary)"
#endif

namespace {

constexpr int skipExitCode = 77;

// Run the app with the given extra args and output prefix; return its exit
// status (WEXITSTATUS), or -1 if it did not exit normally. Output silenced.
int runApp(const std::string& extraArgs, const std::string& prefix)
{
    // Re 5000 on a D=8-cell body: an under-resolved wake with real small-scale
    // energy, so eddy viscosity produces an unambiguous drop in total K (at the
    // laminar default Re=60 the domain K is dominated by the fixed freestream
    // and the LES effect is FP-noise). Still only a few-second run at n=64.
    std::string cmd = "\"" CYLINDER_BIN "\" 64 --nz 1 --re 5000 --ft 1.0 --fdt 100 ";
    cmd += extraArgs;
    cmd += " --out \"" + prefix + "\" > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    if (rc == -1 || !WIFEXITED(rc)) return -1;
    return WEXITSTATUS(rc);
}

std::string slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Final K value (3rd CSV column) from the last data row; NaN if unreadable.
double finalK(const std::string& prefix)
{
    std::ifstream f(prefix + "_stats.csv");
    std::string line, last;
    std::getline(f, line); // header
    while (std::getline(f, line))
        if (!line.empty()) last = line;
    if (last.empty()) return std::nan("");
    std::stringstream ls(last);
    std::string cell;
    for (int col = 0; col < 3; ++col)
        if (!std::getline(ls, cell, ',')) return std::nan("");
    return std::strtod(cell.c_str(), nullptr);
}

void cleanup(const std::string& prefix)
{
    for (const char* suf : {"_stats.csv", ".json", "_vort.bin", "_speed.bin",
                            "_mask.bin"})
        std::remove((prefix + suf).c_str());
}

bool pass = true;
void check(bool ok, const char* what)
{
    std::printf("%-40s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) pass = false;
}

} // namespace

int main()
{
    if (!dns::gpu::deviceAvailable())
    {
        std::printf("no HIP device, skipping\n");
        return skipExitCode;
    }

    const std::string pOff = "les_test_off", pNoflag = "les_test_noflag",
                      pOn = "les_test_on", p3d = "les_test_3d";

    const int rcNoflag = runApp("", pNoflag);
    const int rcOff = runApp("--les 0", pOff);
    const int rcOn = runApp("--les 0.2", pOn);
    const int rc3d = runApp("--les 0.2 --nz 4", p3d);

    check(rcNoflag == 0, "no-flag run exits 0");
    check(rcOff == 0, "--les 0 run exits 0");
    check(rcOn == 0, "--les 0.2 run exits 0");

    // (1) default-off is a bit-for-bit no-op.
    const std::string sOff = slurp(pOff + "_stats.csv");
    const std::string sNoflag = slurp(pNoflag + "_stats.csv");
    check(!sOff.empty() && sOff == sNoflag,
          "--les 0 stats == no-flag stats");

    // (2) LES dissipates: finite, lower final K.
    const double kOff = finalK(pOff), kOn = finalK(pOn);
    std::printf("final K: off=%.6g on=%.6g\n", kOff, kOn);
    check(std::isfinite(kOn), "--les 0.2 final K finite");
    check(std::isfinite(kOff) && std::isfinite(kOn) && kOn < kOff,
          "--les 0.2 final K < --les 0 final K");

    // (3) 2D-only guard.
    check(rc3d != 0, "--les with --nz 4 exits nonzero");

    for (const auto& p : {pOff, pNoflag, pOn, p3d}) cleanup(p);

    std::printf("%s\n", pass ? "ALL OK" : "FAILURES");
    return pass ? 0 : 1;
}
