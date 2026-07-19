// Coverage for the --open (fully periodic far field + fringe sponge) path in
// cylinder_tunnel_gpu. The kernels live in the app's anonymous namespace, so
// we drive the built binary and read its stats CSV (columns:
// t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters).
//
// Checks (Re 60, n=512, dr 16 -> 16D x 16D domain, D = 32 cells, native 2D;
// dt is diffusion-limited, run ~2 min wall). The domain must be this large:
// the fringe clamps u toward uInf ~1.6D inside each boundary, and at 8D that
// clamp confines HARDER than the free-slip tunnel (measured cd ~2.0 at dr 8):
//   1. --open run exits 0.
//   2. Tail-window (last 25% of rows) mean cd finite and inside the generous
//      open-domain band [1.1, 1.8]; unbounded literature Re60 cd ~ 1.4. The
//      confined tunnel carries a +26% blockage excess (repo-calibrated), so
//      additionally cd < 1.75 shows the excess is gone.
//   3. Tail max of the maxDiv column < 1e-2 (periodic projection healthy).
//
// No HIP device -> skip (exit 77), probed via the app's own predicate.

#include "dns/gpu/device_util.hpp"

#include <algorithm>
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

// Run the app; return its exit status (WEXITSTATUS), -1 on abnormal exit.
int runApp(const std::string& args)
{
    const std::string cmd = "\"" CYLINDER_BIN "\" " + args + " > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    if (rc == -1 || !WIFEXITED(rc)) return -1;
    return WEXITSTATUS(rc);
}

struct Row
{
    double t, cd, maxDiv;
};

// t (col 0), cd (col 3), maxDiv (col 7) from every complete data row.
std::vector<Row> readStats(const std::string& prefix)
{
    std::vector<Row> rows;
    std::ifstream f(prefix + "_stats.csv");
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line))
    {
        if (line.empty()) continue;
        std::stringstream ls(line);
        std::string cell;
        double col[8]{};
        bool ok = true;
        for (int q = 0; q < 8 && ok; ++q)
        {
            ok = bool(std::getline(ls, cell, ','));
            if (ok) col[q] = std::strtod(cell.c_str(), nullptr);
        }
        if (ok) rows.push_back({col[0], col[3], col[7]});
    }
    return rows;
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
    std::printf("%-48s %s\n", what, ok ? "ok" : "FAIL");
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

    // 16D x 16D open domain, D = 32 cells; t_end ~ 38 = ~96 D-times, past the
    // ~55 D-times the confined anchor needed to settle. --fdt huge = no
    // frames past t=0.
    const std::string prefix = "open_incomp_test";
    const int rc = runApp("512 --nz 1 --re 60 --dr 16 --open --ft 6 --fdt 1e9 "
                          "--out " + prefix);
    check(rc == 0, "--open run exits 0");

    const auto rows = readStats(prefix);
    check(rows.size() > 20, "stats CSV has data rows");
    if (!rows.empty())
    {
        const double tLast = rows.back().t;
        double sum = 0.0, maxDiv = 0.0;
        int nTail = 0;
        for (const auto& r : rows)
        {
            if (r.t < 0.75 * tLast) continue;
            sum += r.cd;
            maxDiv = std::max(maxDiv, r.maxDiv);
            ++nTail;
        }
        const double cd = nTail > 0 ? sum / nTail : std::nan("");
        std::printf("tail (t >= %.2f, %d rows): mean cd=%.4f maxDiv=%.3e\n",
                    0.75 * tLast, nTail, cd, maxDiv);
        check(std::isfinite(cd), "tail mean cd finite");
        check(cd > 1.1 && cd < 1.8, "cd in open-domain band [1.1, 1.8]");
        check(cd < 1.75, "cd below confined-tunnel level (blockage gone)");
        check(maxDiv < 1e-2, "tail maxDiv < 1e-2");
    }

    cleanup(prefix);
    std::printf("%s\n", pass ? "ALL OK" : "FAILURES");
    return pass ? 0 : 1;
}
