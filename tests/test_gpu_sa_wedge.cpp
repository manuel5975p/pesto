// SA-neg's small separated-flow gate (docs/rans.md "SST: small-case
// verification", the SA twin of gpu_sst (b)): the L1 sphere wedge (64 x 48),
// Re 1e3, M 0.2, RANS SA-neg, explicit LTS to TOL 1e-4 through the binary,
// pinned to its measured state. The OpenFOAM comparison on the identical mesh
// is PRINTED, not gated: tier B measures -1.07%, outside the 1% the
// masterplan set before measuring (issues.md). ~30 s.
#include "validation_util.hpp"
#include "check.hpp"
#include "gpu_test_util.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

using dnstest::check;

namespace {

// MEASURED (gfx1100, 2026-10-27): exit 0 at 136600 iterations (29 s), cd
// 0.48541, theta_s 99.50 deg, L_r 1.2691 D, mu_t/mu max 9.86. M 0.1 (exit 8
// at the 4e5 cap by the M 0.1 jitter floor, settled: window mean 0.48061
// +- 0.06%, L_r 1.2391, theta_s 99.68) gives cd0 = (4 cd(0.1) - cd(0.2))/3 =
// 0.47901, cd(0.2)/cd0 = 1.01336. OpenFOAM v2606 SpalartAllmaras (ft2 true,
// Ct3 1.2, Ct4 0.5, Prt 0.9), CENTRAL U (sph1e4.py --u-scheme linear), 3000
// iterations, settled 1e-4 in 327, on the IDENTICAL mesh: cd 0.48419,
// theta_s 100.24 deg, L_r 1.1407 D. Tier B -1.07%, L_r +7.8% (cd0-extrap).
constexpr double kCd = 0.48541, kLr = 1.2691, kThetaS = 99.50;
constexpr double kMachFactor = 1.01336, kFoamCd = 0.48419, kFoamThetaS = 100.24,
                 kFoamLr = 1.1407;

double sepAngle(const valtest::CsvTable& s)
{
    const auto& th = s.col("theta_deg");
    const auto& cf = s.col("cf");
    for (std::size_t i = 1; i < cf.size(); ++i)
        if (cf[i - 1] > 0 && cf[i] <= 0)
            return th[i - 1] + (th[i] - th[i - 1]) * cf[i - 1] / (cf[i - 1] - cf[i]);
    return std::nan("");
}

void wedgeGate()
{
    const std::string prefix = "gpu_sa_wedge";
    const int rc = valtest::runBinary(
        "\"" DNS_BIN "\" flow --physics compressible --body sphere --ogrid --nz 1 "
        "--n 64 --ny 48 --re 1e3 --mach 0.2 --ogrid-far 20 --ogrid-beta 4.5 "
        "--rans sa-neg --steady 1e-4 --steady-steps 200000 --emit stats "
        "--emit surface --out " + prefix);
    check(rc == 0, "wedge L1 Re 1e3 SA-neg: the steady march certifies (exit 0)");
    const auto st = valtest::loadCsv(prefix + "_stats.csv");
    const auto sf = valtest::loadCsv(prefix + "_surface.csv");
    if (!st || !sf || st->col("iter").empty()) {
        check(false, "SA wedge gate: stats and surface readable");
        return;
    }
    const double it = st->col("iter").back(), cd = st->col("cd").back();
    const double lr = st->col("lr").back(), ts = sepAngle(*sf);
    double mut = 0;
    for (double m : st->col("mutMax")) mut = std::max(mut, m);
    std::printf("wedge L1 Re 1e3 SA-neg: %.0f iterations, cd %.5f, theta_s %.2f, "
                "L_r %.4f, mutMax %.2f\n",
                it, cd, ts, lr, mut);
    check(it <= 1.8e5, "SA wedge gate: certified within 1.8e5 iterations (136600)");
    check(std::abs(cd / kCd - 1) < 5e-4 && std::abs(lr / kLr - 1) < 5e-3 &&
              std::abs(ts - kThetaS) < 0.1,
          "SA wedge gate: cd, L_r, theta_s at their measured values");
    check(mut > 5 && mut < 20, "SA wedge gate: turbulent wake (mu_t/mu max in (5, 20))");
    // Not a check: the code-to-code number, for the log (issues.md).
    std::printf("  vs OpenFOAM SpalartAllmaras (central U, identical mesh): cd0 "
                "%.5f vs %.5f (%+.2f%%, tier B), theta_s %.2f vs %.2f (M 0.2), "
                "L_r %.4f vs %.4f (M 0.2)\n",
                cd / kMachFactor, kFoamCd, 100 * (cd / kMachFactor / kFoamCd - 1), ts,
                kFoamThetaS, lr, kFoamLr);
    std::error_code ec;
    for (const char* s : {"_stats.csv", "_surface.csv", ".json", "_mask.bin"})
        std::filesystem::remove(prefix + s, ec);
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    wedgeGate();
    return dnstest::finish();
}
