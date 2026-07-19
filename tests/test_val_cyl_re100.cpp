// Validation A3: cylinder tunnel at Re 100, 256^2, D = L/8 (32 cells).
// literature: St vs Williamson (1988) 0.164, band [0.157, 0.205]. Measured
//   0.196 (+20%): the beta = 12.5% free-slip periodic-image confinement excess
//   grows with Re (repo-measured St +12% @Re60, +20% @Re100, +30% @Re800) --
//   calibration argument in docs/validation.md.
// literature: cd vs Henderson/Williamson 1.33-1.35, band [1.33, 1.82].
//   Measured 1.730 (+29%); repo-measured confinement excess is +26% at Re60
//   and grows with Re (free-slip periodic images confine much harder than
//   open-jet Maskell +9%). The tight cd claim lives in val_cyl_blockage at
//   dr = 16.
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>

int main()
{
    using std::numbers::pi;
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const std::string prefix = std::string(DNS_VAL_SCRATCH) + "/cyl_re100";
    const double tEnd = 16.0 * 2 * pi; // ~100.5; window [40,100] ~ 12 periods
    const int rc = valtest::runBinary(
        std::string(CYLINDER_BIN)
        + " 256 --nz 1 --re 100 --dr 8 --ft 16 --fdt 1e9 --out " + prefix);
    valtest::check(rc == 0, "cylinder_tunnel_gpu Re100 exits 0");

    auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab) { std::printf("FAIL: no stats csv\n"); return 1; }
    const auto& t = tab->col("t");
    const auto& cd = tab->col("cd");
    const auto& cl = tab->col("cl");
    const double t0 = 40.0, t1 = tEnd;

    const double cdMean = valtest::windowMean(t, cd, t0, t1);
    const double clRms = valtest::windowRms(t, cl, t0, t1);
    const auto st = valtest::strouhal(t, cl, t0, t1);
    const double stD = st.freq * (2 * pi / 8.0);
    std::printf("cd=%.4f cl_rms=%.4f St=%.4f (periods=%d scatter=%.3f)\n", cdMean,
                clRms, stD, st.periods, st.periodScatter);

    valtest::check(clRms > 0.01, "shedding established (cl_rms > 0.01)");
    valtest::check(st.periods >= 5, "St window holds >= 5 periods");
    valtest::check(stD > 0.157 && stD < 0.205,
                   "St in [0.157, 0.205] (Williamson 1988: 0.164 + confinement)");
    valtest::check(cdMean > 1.33 && cdMean < 1.82,
                   "cd in [1.33, 1.82] (Henderson 1.33-1.35 + blockage)");

    for (const char* s : {"_stats.csv", ".json", "_vort.bin", "_speed.bin", "_mask.bin"})
        std::remove((prefix + s).c_str());
    return valtest::finish();
}
