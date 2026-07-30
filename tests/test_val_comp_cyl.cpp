// Validation A5/B5: compressible cylinder tunnel (Kennedy-Gruber, M 0.2).
// argv[1] = case: re60 (Tier A) | re800 (Tier B heavy). ANCHOR-ONLY test: 2D
// Re800 is a model (the real wake is 3D), and the Re60 M0.2 cd has no clean
// literature value -- truth is the repo's own converged numbers. A case whose
// checks are ALL anchor-gated skips (rc 77): re60 is anchored (2026-07-29),
// re800 is not.
// Comp app geometry: D = pi/4 fixed, h = D/64 * (512/n), L = n*h = 8D = 2pi
// at every n; --ft counts L, and the run adds the --ramp 2 quiet start on top.
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

int main(int argc, char** argv)
{
    using std::numbers::pi;
    const std::string what = argc > 1 ? argv[1] : "re60";
    if (what != "re60" && what != "re800")
    {
        std::printf("usage: %s re60|re800\n", argv[0]);
        return 1;
    }
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const auto anchors = valtest::loadAnchors(DNS_ANCHORS_CSV);
    const double D = pi / 4.0; // fixed in the comp app

    if (what == "re60")
    {
        const auto aCd = valtest::findAnchor(anchors, "comp_re60_n256_cd");
        const auto aInc = valtest::findAnchor(anchors, "cyl_re60_n256_cd");
        if (!aCd && !aInc) valtest::skip("all anchor checks deferred");

        const std::string prefix = std::string(DNS_VAL_SCRATCH) + "/comp_re60";
        // Anchor window, not the run end: the run reaches t = 127.7 (2 + 20L),
        // and cd is steady from t ~ 20 on -- the mean over [30, tWin] and over
        // the last flow-throughs agree to 6 digits (measured 2026-07-29).
        const double tWin = 20.0 * pi;
        const int rc = valtest::runBinary(
            std::string(DNS_BIN)
            + " flow --physics compressible --body cylinder --far sponge"
              " --n 256 --re 60 --mach 0.2 --ft 20 --fdt 1e9 --out " + prefix);
        valtest::check(rc == 0, "dns flow compressible cylinder Re60 exits 0");

        auto tab = valtest::loadCsv(prefix + "_stats.csv");
        if (!tab) { std::printf("FAIL: no stats csv\n"); return 1; }
        const double cdMean =
            valtest::windowMean(tab->col("t"), tab->col("cd"), 30.0, tWin);
        std::printf("cd=%.4f (window [30, %.1f])\n", cdMean, tWin);

        if (aCd)
            valtest::check(std::abs(cdMean - aCd->value) <= aCd->tolRel * aCd->value,
                           "cd matches anchor comp_re60_n256_cd");
        if (aInc)
        {
            // Cross-branch offset at matched Re / box / D-in-cells, NOT a
            // Mach correction: the compressible ghost-mirror wall drags 6.9%
            // more than the incompressible cut-cell wall. Measured 2026-07-29,
            // +0.1255. The M^2 reading this check used to assert is REFUTED --
            // cd FALLS with Mach (1.9441 at M 0.2, 1.9666 at 0.1, 2.0016 at
            // 0.05), so the incompressible limit of this branch sits further
            // above 1.8187, not closer; and the gap does not close with
            // resolution (n512 gives 1.9659). Band = measured +-50%: it pins
            // the two branches to each other, which is the one thing neither
            // single-branch anchor can catch.
            const double d = cdMean - aInc->value;
            std::printf("cd_comp - cd_incomp = %.4f\n", d);
            valtest::check(d > 0.06 && d < 0.20,
                           "cd_comp - cd_incomp in [0.06, 0.20] (cross-branch wall offset)");
        }
        for (const char* s : {"_stats.csv", ".json", "_vort.bin", "_speed.bin",
                              "_mask.bin"})
            std::remove((prefix + s).c_str());
        return valtest::finish();
    }

    // re800 (heavy): grid-converged repo numbers cd 1.71, St 0.2445 (2026-07-17).
    const auto aCd = valtest::findAnchor(anchors, "comp_re800_n512_cd");
    const auto aSt = valtest::findAnchor(anchors, "comp_re800_n512_st");
    if (!aCd && !aSt) valtest::skip("all anchor checks deferred");

    const std::string prefix = std::string(DNS_VAL_SCRATCH) + "/comp_re800";
    const double tEnd = 15.0 * 2 * pi; // n=512 -> L=2pi
    const int rc = valtest::runBinary(
        std::string(DNS_BIN)
        + " flow --physics compressible --body cylinder --far sponge"
          " --n 512 --re 800 --mach 0.2 --ft 15 --fdt 1e9 --out " + prefix);
    valtest::check(rc == 0, "dns flow compressible cylinder Re800 exits 0");

    auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab) { std::printf("FAIL: no stats csv\n"); return 1; }
    const auto& t = tab->col("t");
    const double t0 = tEnd - 5.0 * 2 * pi; // last 5 flow-throughs
    const double cdMean = valtest::windowMean(t, tab->col("cd"), t0, tEnd);
    const auto st = valtest::strouhal(t, tab->col("cl"), t0, tEnd);
    const double stD = st.freq * D;
    std::printf("cd=%.4f St=%.4f (periods=%d)\n", cdMean, stD, st.periods);

    valtest::check(st.periods >= 5, "St window holds >= 5 periods");
    if (aCd)
        valtest::check(std::abs(cdMean - aCd->value) <= aCd->tolRel * aCd->value,
                       "cd matches anchor comp_re800_n512_cd");
    if (aSt)
        valtest::check(std::abs(stD - aSt->value) <= aSt->tolRel * aSt->value,
                       "St matches anchor comp_re800_n512_st");

    for (const char* s : {"_stats.csv", ".json", "_vort.bin", "_speed.bin", "_mask.bin"})
        std::remove((prefix + s).c_str());
    return valtest::finish();
}
