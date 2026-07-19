// Validation B3 (heavy): sphere tunnel at Re 300, 160^3, D = L/8 (20 cells).
// literature: cd vs Johnson & Patel (JFM 1999) 0.656, band [0.657, 0.77]
//   (free-slip 8D confinement only adds drag; ceiling from the repo's measured
//   blockage at this D ratio).
// literature: |cl| in [0.01, 0.10]. J&P report 0.069 IN the wake symmetry
//   plane; the app tallies only the y-projection and the plane orientation
//   drifts despite the y-trip (measured cl_y = -0.027 at t = 75), so the
//   floor asserts only the planar-symmetric NON-AXISYMMETRIC wake (an
//   axisymmetric wake, Re < ~210 physics, gives |cl| ~ 0).
// literature (gated): hairpin-shedding St in [0.12, 0.15] when the cl
//   oscillation is established (cl_rms > 0.01; J&P 0.137). At Re 300 the
//   oscillation is weak near onset (J&P dCl ~ 0.016) -- measured rms 0.006,
//   so the St band is typically not applicable at this confinement/grid.
// anchor: cd vs sphere_re300_n160_cd +-3% (deferred).
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

int main()
{
    using std::numbers::pi;
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const std::string prefix = std::string(DNS_VAL_SCRATCH) + "/sphere_re300";
    const double tEnd = 12.0 * 2 * pi; // ~75.4
    const int rc = valtest::runBinary(
        std::string(DNS_BIN)
        + " flow --body sphere --far tunnel --n 160 --re 300 --dr 8"
          " --ft 12 --fdt 1e9 --out " + prefix);
    valtest::check(rc == 0, "dns flow sphere Re300 exits 0");

    auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab) { std::printf("FAIL: no stats csv\n"); return 1; }
    const auto& t = tab->col("t");
    const auto& cd = tab->col("cd");
    const auto& cl = tab->col("cl");
    const double t0 = tEnd - 5.0 * 2 * pi; // last 5 flow-throughs

    const double cdMean = valtest::windowMean(t, cd, t0, tEnd);
    const double clMean = valtest::windowMean(t, cl, t0, tEnd);
    const double clRms = valtest::windowRms(t, cl, t0, tEnd);
    const auto st = valtest::strouhal(t, cl, t0, tEnd);
    const double stD = st.freq * (2 * pi / 8.0);
    std::printf("cd=%.4f cl=%.4f cl_rms=%.4f St=%.4f (periods=%d)\n", cdMean,
                clMean, clRms, stD, st.periods);

    valtest::check(cdMean > 0.657 && cdMean < 0.77,
                   "cd in [0.657, 0.77] (Johnson & Patel 0.656 + blockage)");
    valtest::check(std::abs(clMean) > 0.01 && std::abs(clMean) < 0.10,
                   "|cl| in [0.01, 0.10] (non-axisymmetric wake; J&P 0.069 in-plane)");
    if (st.periods >= 5 && clRms > 0.01)
        valtest::check(stD > 0.12 && stD < 0.15,
                       "St in [0.12, 0.15] (J&P 0.137)");
    else
        std::printf("  [NOTE] cl oscillation weak (periods=%d rms=%.4f); St check"
                    " not applicable\n", st.periods, clRms);

    const auto anchors = valtest::loadAnchors(DNS_ANCHORS_CSV);
    if (auto a = valtest::findAnchor(anchors, "sphere_re300_n160_cd"))
        valtest::check(std::abs(cdMean - a->value) <= a->tolRel * a->value,
                       "cd matches anchor sphere_re300_n160_cd");

    for (const char* s : {"_stats.csv", ".json", "_vort.bin", "_speed.bin", "_mask.bin"})
        std::remove((prefix + s).c_str());
    return valtest::finish();
}
