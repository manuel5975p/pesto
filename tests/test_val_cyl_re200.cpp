// Validation B2 (heavy): cylinder tunnel at Re 200, 512^2, D = L/8 (64 cells).
// literature: St vs the 2D-DNS consensus 0.196-0.201 (Henderson 1997 / 2D
//   computations -- the experimental wake is three-dimensional at Re 200, so
//   2D numbers, not wind-tunnel St ~ 0.181, are the truth class here), band
//   [0.183, 0.235]. Measured 0.2237 (+13% over 0.197): the beta = 12.5%
//   free-slip periodic-image confinement (repo-measured St excess +12% @Re60,
//   +20% @Re100, +30% @Re800; docs/validation.md).
// literature: cd in [1.32, 1.78] (2D-DNS ~ 1.34 + measured confinement excess
//   +26..29% at this beta; measured 1.696).
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

int main()
{
    using std::numbers::pi;
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const std::string prefix = std::string(DNS_VAL_SCRATCH) + "/cyl_re200";
    const double tEnd = 20.0 * 2 * pi; // ~125.7; window [50,125] ~ 18 periods
    const int rc = valtest::runBinary(
        std::string(DNS_BIN)
        + " flow --body cylinder --far tunnel --n 512 --nz 1 --re 200 --dr 8 --ft 20 --fdt 1e9 --out " + prefix);
    valtest::check(rc == 0, "dns flow cylinder Re200 exits 0");

    auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab) { std::printf("FAIL: no stats csv\n"); return 1; }
    const auto& t = tab->col("t");
    const auto& cd = tab->col("cd");
    const auto& cl = tab->col("cl");
    const double t0 = 50.0, t1 = tEnd;

    const double cdMean = valtest::windowMean(t, cd, t0, t1);
    const double clRms = valtest::windowRms(t, cl, t0, t1);
    const auto st = valtest::strouhal(t, cl, t0, t1);
    const double stD = st.freq * (2 * pi / 8.0);
    std::printf("cd=%.4f cl_rms=%.4f St=%.4f (periods=%d scatter=%.3f)\n", cdMean,
                clRms, stD, st.periods, st.periodScatter);

    valtest::check(clRms > 0.01, "shedding established (cl_rms > 0.01)");
    valtest::check(st.periods >= 5, "St window holds >= 5 periods");
    valtest::check(stD > 0.183 && stD < 0.235,
                   "St in [0.183, 0.235] (2D-DNS consensus 0.196-0.201 + confinement)");
    valtest::check(cdMean > 1.32 && cdMean < 1.78, "cd in [1.32, 1.78]");

    for (const char* s : {"_stats.csv", ".json", "_vort.bin", "_speed.bin", "_mask.bin"})
        std::remove((prefix + s).c_str());
    return valtest::finish();
}
