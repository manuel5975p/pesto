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

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
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

// runner_channel's EXPLICIT steady RANS march and its stop rule
// (dns/steady_stop): exit 0 at TOL 1e-3, with ub where the rule claims.
// Measured 2026-10-25: stops at 177800 iterations (~28 s), ub 15.8196. ub
// ALONE would certify at 3400 iterations, 2.8% low (its dip); the companion
// integrals are what hold the stop off, so the iteration count is checked as
// well as ub. The FP64 oracle (rans_channel1d, Re_tau 180, ny 64) is 15.8300;
// the march's own fixed point sits +3.3e-4 above it -- MASS DRIFT: local time
// stepping conserves sum V rho / lts, not sum V rho, while v != 0, and the
// steady equations are a one-parameter family in mass (docs/rans.md) -- so
// the band is TOL plus that offset, rounded up. The scheme is spelled out so
// a future default flip cannot silently retarget this test.
void testChannelSteady()
{
    const std::string prefix = "runner_channel_steady";
    const int rc = runDns("channel --n 4 --nz 1 --ny 64 --rans sa-neg "
                          "--steady 1e-3 --steady-scheme explicit --out " +
                          prefix);
    check(rc == 0, "runner_channel --steady 1e-3: converges, exits 0");
    if (rc == 0)
    {
        const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
        check(tbl.has_value(), "runner_channel --steady: stats readable");
        if (tbl)
        {
            const double it = tbl->col("iter").back(),
                         ub = tbl->col("ubulk").back(),
                         rem = tbl->col("remaining").back();
            std::printf("  channel steady: iter %.0f ub %.6g remaining %.3g\n",
                        it, ub, rem);
            check(rem < 1e-3, "runner_channel --steady: last row certifies");
            check(it > 1e5, "runner_channel --steady: not stopped at ub's "
                            "early dip");
            check(std::abs(ub / 15.8300 - 1) < 1.5e-3,
                  "runner_channel --steady: ub within TOL of the oracle "
                  "(+ the fixed point's offset)");
        }
    }
    cleanup(prefix);
}

// The IMPLICIT steady march (--steady-scheme implicit, docs/rans.md): the
// Newton line solve, mass pinned, so it lands ON the oracle (15.8299695),
// not on the explicit march's drifted fixed point. Measured 2026-10-25: 27
// iterations, ub 15.8299691, 0.1 s.
void testChannelSteadyImplicit()
{
    const std::string prefix = "runner_channel_steady_implicit";
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = runDns("channel --n 4 --nz 1 --ny 64 --rans sa-neg "
                          "--steady 1e-4 --steady-scheme implicit --out " +
                          prefix);
    const double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
    check(rc == 0, "runner_channel --steady-scheme implicit: converges, exits 0");
    if (rc == 0)
    {
        const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
        check(tbl.has_value(), "runner_channel implicit: stats readable");
        if (tbl)
        {
            const double it = tbl->col("iter").back(),
                         ub = tbl->col("ubulk").back(),
                         rem = tbl->col("remaining").back();
            std::printf("  channel steady implicit: iter %.0f ub %.6g remaining "
                        "%.3g, %.2f s\n",
                        it, ub, rem, secs);
            check(rem < 1e-4, "runner_channel implicit: last row certifies");
            check(it <= 60, "runner_channel implicit: <= 60 iterations");
            check(std::abs(ub / 15.82997 - 1) < 1e-5,
                  "runner_channel implicit: ub within 1e-5 of the oracle");
            check(secs < 2.0, "runner_channel implicit: < 2 s");
        }
    }
    cleanup(prefix);
}

// runner_ogrid with RANS (docs/rans.md, "dns flow"): a short SA URANS run on
// the coarse O-grid cylinder writes the RANS columns, finite, 0 guard hits.
void testOGridRansUrans()
{
    const std::string prefix = "runner_ogrid_sa_urans";
    const int rc = runDns("flow --physics compressible --body cylinder --ogrid "
                          "--n 256 --ny 96 --re 40 --rans sa-neg --ramp 0 "
                          "--ft 0.0006 --emit stats --out " + prefix);
    check(rc == 0, "runner_ogrid --rans sa-neg URANS: exits 0");
    if (rc == 0)
    {
        check(rowCount(prefix) >= 1, "runner_ogrid RANS: stats rows written");
        for (const char* c : {"cd", "cdp", "cdf", "mutMax", "yplusMax"})
            check(allFinite(prefix, c),
                  std::string("runner_ogrid RANS: ") + c + " finite");
        check(maxOf(prefix, "floorHits") == 0, "runner_ogrid RANS: 0 hits");
    }
    cleanup(prefix);
}

// runner_ogrid's explicit steady march (SA, Re 40). Measured 2026-10-25
// (gfx1100): the integrals reach their FP32 plateau within ~5e4 iterations
// but jitter there by ~2e-6 relative (pressure-integrated cd amplifies the
// FP32 residual floor by p/q ~ 36 at M 0.2), above kSteadyFlat = 2^-20, so
// the stop never certifies and the march ends at its cap with exit 8 -- the
// honest non-convergence report (masterplan_flowrans U1/U2, docs/rans.md).
// The smoke checks the march, the columns and the cap's exit code.
void testOGridRansSteadyCap()
{
    const std::string prefix = "runner_ogrid_sa_steady";
    const int rc = runDns("flow --physics compressible --body cylinder --ogrid "
                          "--n 256 --ny 96 --re 40 --rans sa-neg --steady 1e-4 "
                          "--steady-steps 2000 --out " + prefix);
    check(rc == 8, "runner_ogrid --steady at a 2000-iteration cap: exit 8");
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    check(tbl.has_value(), "runner_ogrid --steady: stats readable");
    if (tbl)
    {
        check(tbl->col("iter").size() == 10 && tbl->col("iter").back() == 2000,
              "runner_ogrid --steady: a row every 200 iterations to the cap");
        for (const char* c : {"cd", "cdf", "resid", "resU", "resT"})
            check(allFinite(prefix, c),
                  std::string("runner_ogrid --steady: ") + c + " finite");
        check(maxOf(prefix, "floorHits") == 0, "runner_ogrid --steady: 0 hits");
    }
    cleanup(prefix);
}

// The revolved sphere's axisymmetric wedge through the binary: the steady
// march to a short cap (exit 8 by the G1b finding), one surface row per real
// wall face under the wedge label, and the --emit mesh artefacts at their
// exact sizes (dns/meshio.hpp: 84-byte header + 3 n nodes + nx nz d1;
// 28 + 16 names + n per field).
void testWedgeArtefacts()
{
    const std::string prefix = "runner_ogrid_wedge";
    const int rc = runDns("flow --physics compressible --body sphere --ogrid "
                          "--nz 1 --n 64 --ny 48 --re 100 --steady 1e-4 "
                          "--steady-steps 1000 --emit stats --emit surface "
                          "--emit mesh --out " + prefix);
    check(rc == 8, "wedge --steady at a 1000-iteration cap: exit 8");
    std::size_t rows = 0;
    std::string head;
    {
        std::ifstream f(prefix + "_surface.csv");
        std::getline(f, head);
        for (std::string l; std::getline(f, l);) ++rows;
    }
    check(head.find("[axisymmetric wedge 5 deg") != std::string::npos,
          "wedge surface CSV: the label comment carries the wedge suffix");
    check(rows == 65, "wedge surface CSV: header + one row per real wall face "
                      "(64)");
    const std::size_t n = 66 * 48 * 3;
    std::error_code ec;
    check(std::filesystem::file_size(prefix + "_mesh.bin", ec) ==
              84 + 24 * n + 8 * 66 * 3,
          "wedge _mesh.bin at its exact size");
    check(std::filesystem::file_size(prefix + "_final.bin", ec) ==
              28 + 8 * 16 + 8 * 8 * n,
          "wedge _final.bin (8 laminar fields) at its exact size");
    const double cd = maxOf(prefix, "cd", 0.0);
    check(std::isfinite(cd) && cd > 0, "wedge: cd finite and positive");
    for (const char* s : {"_surface.csv", "_mesh.bin", "_final.bin"})
        std::filesystem::remove(prefix + s, ec);
    cleanup(prefix);
}

// The flow Newton-Krylov march through the binary (cylinder Re 40 laminar,
// 256 x 96): a short cap (exit 8 -- gate B0 failed, the stop cannot certify,
// docs/rans.md), one row per accepted Newton step with the NK columns, the
// residual falling. Measured (gfx1100): 12 steps take ~24 s, cd 1.5975 at
// step 12 on its way to the plateau 1.56656.
void testOGridNk()
{
    const std::string prefix = "runner_ogrid_nk";
    const int rc = runDns("flow --physics compressible --body cylinder --ogrid "
                          "--n 256 --ny 96 --re 40 --steady 1e-4 "
                          "--steady-scheme implicit --steady-steps 12 --out " +
                          prefix);
    check(rc == 8, "runner_ogrid --steady-scheme implicit at a 12-step cap: "
                   "exit 8");
    const auto tbl = valtest::loadCsv(prefix + "_stats.csv");
    check(tbl.has_value(), "runner_ogrid NK: stats readable");
    if (tbl)
    {
        const auto& it = tbl->col("iter");
        const auto& res = tbl->col("resid");
        check(it.size() >= 10 && it.back() == 12,
              "runner_ogrid NK: a row per accepted Newton step");
        check(!res.empty() && res.back() < 0.05 * res.front(),
              "runner_ogrid NK: the residual falls 20x in 12 steps");
        for (const char* c : {"cd", "cfl", "alpha", "krylov"})
            check(allFinite(prefix, c),
                  std::string("runner_ogrid NK: ") + c + " finite");
    }
    cleanup(prefix);
}

// runner_comp with --rans on the immersed ghost wall (P5): exit 0, the RANS
// columns finite, SA 0 hits.
void testImmersedRansUrans()
{
    const std::string prefix = "runner_comp_sa_urans";
    const int rc = runDns("flow --physics compressible --body cylinder --nz 1 "
                          "--n 192 --re 1000 --rans sa-neg --ramp 0 --ft 0.05 "
                          "--emit stats --out " + prefix);
    check(rc == 0, "runner_comp --rans sa-neg (immersed) URANS: exits 0");
    if (rc == 0)
    {
        check(rowCount(prefix) >= 1, "runner_comp RANS: stats rows written");
        for (const char* c : {"cd", "mutMax", "yplusMax", "yplusMean"})
            check(allFinite(prefix, c),
                  std::string("runner_comp RANS: ") + c + " finite");
        check(maxOf(prefix, "floorHits") == 0, "runner_comp RANS: 0 hits");
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
    testChannelSteady();
    testChannelSteadyImplicit();
    testOGridRansUrans();
    testOGridRansSteadyCap();
    testWedgeArtefacts();
    testOGridNk();
    testImmersedRansUrans();
    return dnstest::finish();
}
