// Validation A5/B5: compressible cylinder tunnel (Kennedy-Gruber, M 0.2).
// argv[1] = case: re60 (Tier A) | re800 (Tier B heavy). ANCHOR-ONLY test: 2D
// Re800 is a model (the real wake is 3D), and the Re60 M0.2 cd has no clean
// literature value -- truth is the repo's own converged numbers. All checks
// are anchor-gated, so the test SKIPS (rc 77) while anchors stay deferred.
// Comp app geometry: D = pi/4 fixed, h = D/64, L = n*h (n=256 -> L=pi).
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
        const double tEnd = 20.0 * pi; // n=256 -> L=pi, --ft 20 -> t ~ 62.8
        const int rc = valtest::runBinary(
            std::string(COMP_CYLINDER_BIN)
            + " 256 --re 60 --mach 0.2 --ft 20 --fdt 1e9 --out " + prefix);
        valtest::check(rc == 0, "comp_cylinder_tunnel_gpu Re60 exits 0");

        auto tab = valtest::loadCsv(prefix + "_stats.csv");
        if (!tab) { std::printf("FAIL: no stats csv\n"); return 1; }
        const double cdMean =
            valtest::windowMean(tab->col("t"), tab->col("cd"), 30.0, tEnd);
        std::printf("cd=%.4f (window [30, %.1f])\n", cdMean, tEnd);

        if (aCd)
            valtest::check(std::abs(cdMean - aCd->value) <= aCd->tolRel * aCd->value,
                           "cd matches anchor comp_re60_n256_cd");
        if (aInc)
        {
            // Mach-squared correction: small, strictly positive.
            const double d = cdMean - aInc->value;
            std::printf("cd_comp - cd_incomp = %.4f\n", d);
            valtest::check(d > 0.005 && d < 0.06,
                           "cd_comp - cd_incomp in [0.005, 0.06] (M^2 sign check)");
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
        std::string(COMP_CYLINDER_BIN)
        + " 512 --re 800 --mach 0.2 --ft 15 --fdt 1e9 --out " + prefix);
    valtest::check(rc == 0, "comp_cylinder_tunnel_gpu Re800 exits 0");

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
