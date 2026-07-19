// Validation B1 (heavy): free-slip-wall blockage study, Re 100 at 512^2 with
// dr = 8 (beta = 12.5%, D = 64 cells) and dr = 16 (beta = 6.25%, D = 32 cells,
// BL ~ 3.2 cells -- at the 3-cell DNS rule).
// literature (dr16, weak confinement): cd in [1.33, 1.55], St in
//   [0.156, 0.180] (Williamson 0.164 - 5% floor; ceiling +10%: the residual
//   beta = 6.25% free-slip confinement still raises St, measured +6.6% --
//   halving beta roughly halves the +20% excess seen at dr = 8).
// direction (Maskell trend): confinement only ADDS drag and raises St, so
//   cd(dr16) < cd(dr8) and St(dr16) <= St(dr8) * 1.01.
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>

namespace {

struct RunStats {
    double cd, clRms, st;
    int periods;
};

RunStats runCase(int dr, const std::string& prefix, double tEnd)
{
    const int rc = valtest::runBinary(
        std::string(DNS_BIN)
        + " flow --body cylinder --far tunnel --n 512 --nz 1 --re 100 --dr "
        + std::to_string(dr) + " --ft 16 --fdt 1e9 --out " + prefix);
    valtest::check(rc == 0, "dr" + std::to_string(dr) + " run exits 0");
    auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab) { std::printf("FAIL: no stats csv (dr=%d)\n", dr); std::exit(1); }
    const auto& t = tab->col("t");
    const auto& cd = tab->col("cd");
    const auto& cl = tab->col("cl");
    const double t0 = 40.0;
    const auto st = valtest::strouhal(t, cl, t0, tEnd);
    RunStats r{valtest::windowMean(t, cd, t0, tEnd),
               valtest::windowRms(t, cl, t0, tEnd),
               st.freq * (2 * std::numbers::pi / dr), st.periods};
    std::printf("dr=%d: cd=%.4f cl_rms=%.4f St=%.4f periods=%d\n", dr, r.cd,
                r.clRms, r.st, r.periods);
    for (const char* s : {"_stats.csv", ".json", "_vort.bin", "_speed.bin", "_mask.bin"})
        std::remove((prefix + s).c_str());
    return r;
}

} // namespace

int main()
{
    using std::numbers::pi;
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const double tEnd = 16.0 * 2 * pi;
    const auto r8 = runCase(8, std::string(DNS_VAL_SCRATCH) + "/cyl_blk8", tEnd);
    const auto r16 = runCase(16, std::string(DNS_VAL_SCRATCH) + "/cyl_blk16", tEnd);

    valtest::check(r8.clRms > 0.01 && r16.clRms > 0.01, "both cases shed");
    valtest::check(r8.periods >= 5 && r16.periods >= 5, ">= 5 periods each");
    valtest::check(r16.cd > 1.33 && r16.cd < 1.55,
                   "dr16 cd in [1.33, 1.55] (Henderson 1.35 + weak blockage)");
    valtest::check(r16.st > 0.156 && r16.st < 0.180,
                   "dr16 St in [0.156, 0.180] (Williamson 0.164 + weak confinement)");
    valtest::check(r16.cd < r8.cd, "cd(dr16) < cd(dr8) (Maskell trend)");
    valtest::check(r16.st <= r8.st * 1.01, "St(dr16) <= St(dr8) + 1%");

    return valtest::finish();
}
