// Validation: the body-fitted O-grid cylinder against the 2D literature --
// the default tier's only O-grid literature test (masterplan_verify W12).
// Two cases, both `dns flow --physics compressible --body cylinder --ogrid`,
// M 0.2, far row at 20 D (the O-grid default):
//   (a) Re 100 laminar shedding, 192x96, --ogrid-trip 0.05: St, mean cd and
//       the cl amplitude over 6 periods after saturation, against Williamson
//       (St 0.164-0.166) and the 2D DNS spread (cd 1.33-1.40, cl' 0.32-0.34).
//   (b) Re 40 steady, 256x96, explicit --steady 1e-4: cd against Dennis &
//       Chang 1.522 after removing the MEASURED M 0.2 / 20 D offset
//       (docs/validation.md "O-grid cylinder"), and pinned to its own value.
// Units: box D = pi/4, U = 1, so St = f D and tEnd = ft * 40 D (+ ramp 2).
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <vector>

namespace {

// Half the peak-to-peak of y over t in [t0, t1].
double halfSwing(const std::vector<double>& t, const std::vector<double>& y,
                 double t0, double t1)
{
    double lo = 1e300, hi = -1e300;
    for (std::size_t i = 0; i < t.size(); ++i)
    {
        if (t[i] < t0 || t[i] > t1) continue;
        lo = std::min(lo, y[i]);
        hi = std::max(hi, y[i]);
    }
    return hi >= lo ? 0.5 * (hi - lo) : std::nan("");
}

void removeOutputs(const std::string& prefix)
{
    for (const char* s : {"_stats.csv", ".json", "_mask.bin"})
        std::remove((prefix + s).c_str());
}

const std::string kBase =
    std::string(DNS_BIN) +
    " flow --physics compressible --body cylinder --ogrid --mach 0.2"
    " --emit stats --quiet";

// (a) Re 100 shedding. Measured 2026-10-27 (gfx1100, 21 s): St 0.1679, cd
// 1.3802, cl' 0.3331; the envelope is within 0.5% of its limit from t ~ 80
// (99% at 71.6, docs/rans.md "The full revolution"), so the window [82, 121.4] is 8.4 periods
// (>= 7 full crossing-to-crossing periods) of the saturated cycle. Bands
// are the masterplan's, set before measuring.
void shedding()
{
    using std::numbers::pi;
    const double D = pi / 4.0;
    const std::string prefix = std::string(DNS_VAL_SCRATCH) + "/ogrid_re100";
    const int rc = valtest::runBinary(
        kBase + " --n 192 --ny 96 --re 100 --ogrid-trip 0.05 --ft 3.8 --out " +
        prefix);
    valtest::check(rc == 0, "Re 100 O-grid shedding run exits 0");
    const auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab)
    {
        valtest::check(false, "Re 100 stats csv readable");
        return;
    }
    const auto& t = tab->col("t");
    const auto& cl = tab->col("cl");
    const double t1 = t.back(), t0 = 82.0;
    const auto st = valtest::strouhal(t, cl, t0, t1);
    const double stD = st.freq * D;
    const double cd = valtest::windowMean(t, tab->col("cd"), t0, t1);
    const double amp = halfSwing(t, cl, t0, t1);
    // Saturation: the swing of the window's first and last period agree.
    const double period = 1.0 / st.freq;
    const double ampFirst = halfSwing(t, cl, t0, t0 + period);
    const double ampLast = halfSwing(t, cl, t1 - period, t1);
    std::printf("Re100: window [%.1f, %.1f] periods %d scatter %.2e  St %.4f"
                "  cd %.4f  cl' %.4f  (first/last period cl' %.4f / %.4f)\n",
                t0, t1, st.periods, st.periodScatter, stD, cd, amp, ampFirst,
                ampLast);
    valtest::check(st.periods >= 6, "Re 100 window holds >= 6 periods");
    valtest::check(std::abs(ampLast / ampFirst - 1.0) < 0.02,
                   "Re 100 cl envelope saturated (first/last period < 2%)");
    valtest::check(stD >= 0.160 && stD <= 0.172, "Re 100 St in [0.160, 0.172]");
    valtest::check(cd >= 1.33 && cd <= 1.42, "Re 100 mean cd in [1.33, 1.42]");
    valtest::check(amp >= 0.30 && amp <= 0.36, "Re 100 cl' in [0.30, 0.36]");
    removeOutputs(prefix);
}

// (b) Re 40 steady. Measured 2026-10-27 (3.2 s, 31200 iterations): cd
// 1.56656. Offsets, all steady at TOL 1e-4 (docs/validation.md): grid
// 192x72 / 256x96 / 384x144 1.5663 / 1.5666 / 1.5668 (converged); Mach 0.2
// / 0.1 / 0.05 1.5666 / 1.5580 / 1.5579; far row 20 / 40 / 80 D at matched
// h1 1.5666 / 1.5354 / 1.5240, Richardson (order 1.45) 1.5174 at R -> inf.
// So cd(20 D, M 0.2) = 1.0382 cd(R -> inf, M -> 0) = 1.0382 * 1.5090.
void steady40()
{
    const std::string prefix = std::string(DNS_VAL_SCRATCH) + "/ogrid_re40";
    const int rc = valtest::runBinary(
        kBase + " --n 256 --ny 96 --re 40 --steady 1e-4 --out " + prefix);
    valtest::check(rc == 0, "Re 40 O-grid steady run certifies (exit 0)");
    const auto tab = valtest::loadCsv(prefix + "_stats.csv");
    if (!tab)
    {
        valtest::check(false, "Re 40 stats csv readable");
        return;
    }
    const double cd = tab->col("cd").back();
    const double kOffset = 1.0382; // measured 20 D / M 0.2 factor, above
    const double cdLimit = cd / kOffset;
    std::printf("Re40: cd %.5f  offset-corrected %.4f  vs Dennis-Chang 1.522"
                " (%+.2f%%)\n",
                cd, cdLimit, 100.0 * (cdLimit / 1.522 - 1.0));
    // 2%: the literature spread (Dennis & Chang 1.522, Fornberg 1.498).
    valtest::check(std::abs(cdLimit / 1.522 - 1.0) < 0.02,
                   "Re 40 offset-corrected cd within 2% of Dennis-Chang 1.522");
    // A grid-converged steady state: pin it (FP32 jitter floor ~4e-5).
    valtest::check(std::abs(cd / 1.56656 - 1.0) < 1e-3,
                   "Re 40 cd within 1e-3 of its measured 1.56656");
    removeOutputs(prefix);
}

} // namespace

int main()
{
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");
    steady40();
    shedding();
    return valtest::finish();
}
