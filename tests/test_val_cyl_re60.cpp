// Validation A2: cylinder tunnel at Re 60, 256^2, D = L/8 (32 cells).
// literature: St from cl zero-crossings vs Williamson (1988) St(60) = 0.1358,
//   band [0.130, 0.156] (+15/-4%). Measured 0.152 (+12%): the free-slip
//   periodic-image confinement at beta = 12.5% raises St as it raises cd
//   (repo-measured cd excess +26% at Re60; St excess +28% at Re800) -- see
//   docs/validation.md for the calibration argument. D = 32 cells adds ~2-3%.
// literature guard: cd in [1.42, 1.90] (floor = unbounded Tritton/Henderson
//   1.42; free-slip periodic-image blockage only adds drag).
// anchor: windowed-mean cd vs cyl_re60_n256_cd (deferred until regenerated).
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

int main()
{
    using std::numbers::pi;
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const std::string prefix = std::string(DNS_VAL_SCRATCH) + "/cyl_re60";
    const double tEnd = 12.0 * 2 * pi; // --ft 12 => ~7.8 shedding periods in window
    const int rc = valtest::runBinary(
        std::string(CYLINDER_BIN)
        + " 256 --nz 1 --re 60 --dr 8 --ft 12 --fdt 1e9 --out " + prefix);
    valtest::check(rc == 0, "cylinder_tunnel_gpu Re60 exits 0");

    auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab) { std::printf("FAIL: no stats csv\n"); return 1; }
    const auto& t = tab->col("t");
    const auto& cd = tab->col("cd");
    const auto& cl = tab->col("cl");
    const double t0 = 30.0, t1 = tEnd;

    const double cdMean = valtest::windowMean(t, cd, t0, t1);
    const double clRms = valtest::windowRms(t, cl, t0, t1);
    const auto st = valtest::strouhal(t, cl, t0, t1);
    const double stD = st.freq * (2 * pi / 8.0); // St = f D / uInf
    std::printf("cd=%.4f cl_rms=%.4f St=%.4f (periods=%d scatter=%.3f)\n", cdMean,
                clRms, stD, st.periods, st.periodScatter);

    valtest::check(clRms > 0.01, "shedding established (cl_rms > 0.01)");
    valtest::check(st.periods >= 5, "St window holds >= 5 periods");
    valtest::check(stD > 0.130 && stD < 0.156,
                   "St in [0.130, 0.156] (Williamson 1988: 0.1358 + confinement)");
    valtest::check(cdMean > 1.42 && cdMean < 1.90,
                   "cd in [1.42, 1.90] (unbounded 1.42 + blockage)");

    const auto anchors = valtest::loadAnchors(DNS_ANCHORS_CSV);
    if (auto a = valtest::findAnchor(anchors, "cyl_re60_n256_cd"))
        valtest::check(std::abs(cdMean - a->value) <= a->tolRel * a->value,
                       "cd matches anchor cyl_re60_n256_cd");

    for (const char* s : {"_stats.csv", ".json", "_vort.bin", "_speed.bin", "_mask.bin"})
        std::remove((prefix + s).c_str());
    return valtest::finish();
}
