// Smoke coverage for the three time-loop runners.
//
// This closes the coverage hole masterplan_drift §1.14 found: no file under
// tests/ referenced runner_incomp, runner_comp or runner_channel -- they were
// the only public headers in the tree no test included. In particular
//
//   * `--far tunnel` WITH walls, the incompressible DEFAULT far field, was
//     exercised end-to-end by nothing (gpu_open_incomp drives --far fringe,
//     gpu_tunnel_open drives --far tunnel --open);
//   * runner_comp had no end-to-end coverage at all -- val_comp_cyl_re60 exits
//     77 unconditionally, every one of its checks being anchor-gated and every
//     anchor commented out.
//
// These are SMOKE tests by construction and carry no literature band, so
// nothing here is quarantine-shaped: exit 0, finite coefficients, a healthy
// projection where there is one, and enough stats rows to prove the loop ran.
// A band belongs in the val_* tier, behind the regeneration procedure.
#include "validation_util.hpp"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

using dnstest::check;


int runDns(const std::string& args)
{
    return valtest::runBinary("\"" DNS_BIN "\" " + args);
}

void cleanup(const std::string& prefix)
{
    std::error_code ec;
    for (const char* suf : {".json", "_stats.csv", "_vort.bin", "_speed.bin",
                            "_mask.bin", "_curl.bin", "_profiles.csv"})
        std::filesystem::remove(prefix + suf, ec);
}

// Rows of any _stats.csv, by COLUMN NAME -- the three schemas share only `t`.
std::size_t rowCount(const std::string& prefix)
{
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    return tbl ? tbl->col("t").size() : 0;
}

bool allFinite(const std::string& prefix, const std::string& column)
{
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    if (!tbl) return false;
    for (double v : tbl->col(column))
        if (!std::isfinite(v)) return false;
    return true;
}

// Max of `column` over rows with t > tMin. tMin > 0 skips the impulsive-start
// row, which is a genuine transient and not a health signal: at t = 0 the
// 128^2 cylinder reports cd 55.7 and maxDiv 1.1e-3, and by t = 0.8 it has
// settled to cd 2.03 and maxDiv 6.7e-5. gpu_tunnel_open windows the same way.
double maxOf(const std::string& prefix, const std::string& column,
             double tMin = -1)
{
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    if (!tbl) return -1;
    const auto& t = tbl->col("t");
    const auto& v = tbl->col(column);
    double m = -1;
    for (std::size_t i = 0; i < v.size() && i < t.size(); ++i)
        if (t[i] > tMin) m = std::max(m, v[i]);
    return m;
}

// runner_incomp on the DEFAULT far field: --far tunnel, walls, no --open.
void testIncompTunnel()
{
    const std::string prefix = "runner_incomp_tunnel";
    const int rc = runDns("flow --body cylinder --far tunnel --n 128 --nz 1 "
                          "--re 60 --dr 8 --ft 1.5 --fdt 1e9 --out " + prefix);
    check(rc == 0, "runner_incomp: --far tunnel (walls) exits 0");
    if (rc == 0)
    {
        check(rowCount(prefix) >= 5, "runner_incomp: stats rows written");
        check(allFinite(prefix, "cd"), "runner_incomp: cd finite throughout");
        check(allFinite(prefix, "cl"), "runner_incomp: cl finite throughout");
        // Past the start row the plateau is ~6.5e-5; 1e-3 is a wide band
        // around that, not a threshold tuned to the measurement.
        const double d = maxOf(prefix, "maxDiv", 0.0);
        std::printf("  incomp tunnel: max maxDiv (t > 0) = %.3e\n", d);
        check(d >= 0 && d < 1e-3, "runner_incomp: projection healthy");
    }
    cleanup(prefix);
}

// runner_comp on the compressible DEFAULT far field: --far nscbc.
void testCompNscbc()
{
    const std::string prefix = "runner_comp_nscbc";
    const int rc = runDns("flow --physics compressible --body cylinder "
                          "--far nscbc --n 128 --nz 1 --re 60 --mach 0.2 "
                          "--ft 1.5 --fdt 1e9 --out " + prefix);
    check(rc == 0, "runner_comp: --far nscbc exits 0");
    if (rc == 0)
    {
        check(rowCount(prefix) >= 5, "runner_comp: stats rows written");
        check(allFinite(prefix, "cd"), "runner_comp: cd finite throughout");
        check(allFinite(prefix, "cs"), "runner_comp: cs finite throughout");
        // wmax is the acoustic dt diagnostic; a diverged run makes it NaN or
        // enormous long before the loop notices.
        const double w = maxOf(prefix, "wmax");
        std::printf("  comp nscbc: max wmax = %.4g\n", w);
        check(std::isfinite(w) && w > 0, "runner_comp: wave speed sane");
    }
    cleanup(prefix);
}

// runner_channel, the third loop: a few steps of the KMM box, shrunk.
void testChannel()
{
    const std::string prefix = "runner_channel_smoke";
    const int rc = runDns("channel --n 32 --ny 32 --nz 32 --re-tau 180 "
                          "--tt 0.02 --profile-dt 0 --fdt 1e9 --out " + prefix);
    check(rc == 0, "runner_channel: short KMM-shaped run exits 0");
    if (rc == 0)
    {
        check(rowCount(prefix) >= 2, "runner_channel: stats rows written");
        check(allFinite(prefix, "ubulk"), "runner_channel: ubulk finite");
        check(allFinite(prefix, "retau"), "runner_channel: retau finite");
    }
    cleanup(prefix);
}

} // namespace

int main()
{
    // Phase 3a probes the GPU before any of this; without a device every run
    // would exit 5 and the checks would be meaningless.
    if (runDns("flow --body cylinder --n 32 --nz 1 --ft 0.001 --emit none "
               "--out runner_probe") == 5)
    dnstest::skip("no HIP device");
    cleanup("runner_probe");

    testIncompTunnel();
    testCompNscbc();
    testChannel();
    return dnstest::finish();
}
