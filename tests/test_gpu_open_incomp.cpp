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
#include "gpu_test_util.hpp"
#include "validation_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/wait.h>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;


// Drives the real binary through validation_util::runBinary -- this used to
// be a private copy of it.
int runApp(const std::string& args)
{
    return valtest::runBinary("\"" DNS_BIN "\" flow --body cylinder " + args);
}

struct Row
{
    double t, cd, maxDiv;
};

// Rows of <prefix>_stats.csv, parsed BY COLUMN NAME. The hardcoded col[0] /
// col[3] / col[7] indices this replaces were a live hazard: _stats.csv has
// three different schemas (incompressible flow / compressible flow /
// channel), so an index that is `cd` in one is something else in another.
std::vector<Row> readStats(const std::string& prefix)
{
    std::vector<Row> rows;
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    if (!tbl) return rows;
    const auto& t = tbl->col("t");
    const auto& cd = tbl->col("cd");
    const auto& mdiv = tbl->col("maxDiv");
    for (std::size_t i = 0; i < t.size(); ++i)
        rows.push_back({t[i], cd[i], mdiv[i]});
    return rows;
}

void cleanup(const std::string& prefix)
{
    for (const char* suf : {"_stats.csv", ".json", "_vort.bin", "_speed.bin",
                            "_mask.bin"})
        std::remove((prefix + suf).c_str());
}


} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();

    // 16D x 16D open domain, D = 32 cells; t_end ~ 38 = ~96 D-times, past the
    // ~55 D-times the confined anchor needed to settle. --fdt huge = no
    // frames past t=0.
    const std::string prefix = "open_incomp_test";
    const int rc = runApp("--far fringe --n 512 --nz 1 --re 60 --dr 16 --ft 6 --fdt 1e9 "
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
    std::printf("%s\n", dnstest::gFailures == 0 ? "ALL OK" : "FAILURES");
    return dnstest::finish();
}
