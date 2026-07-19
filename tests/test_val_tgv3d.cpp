// Validation A4/B4: 3D Taylor-Green vortex at Re 1600 (tgv3d n 1600 20).
// argv[1] = grid size (64 = Tier A, 128 = Tier B heavy).
// anchor: max-rel-diff vs the repo's own curve data/validation/ours_tgvN.csv
//   (same binary => near-bitwise; slack for compiler/ROCm drift): K <= 2e-3,
//   epsVisc <= 2e-2.
// literature: peak dissipation vs the 512^3 spectral reference
//   data/spectral_Re1600_512.gdiag (0.012858 at t = 8.97). Convergence to the
//   spectral peak is OSCILLATORY, not monotone from below (repo curves: 64^3
//   -8.1%, 128^3 +5.6%, 256^3 -4.3%, 512^3 +1.2%): near the peak the
//   KE-conserving scheme piles enstrophy at the grid scale and can overshoot.
//   64^3: trend check [-40%, +5%], peak time [8.0, 10.0]; 128^3: [-10%, +10%],
//   peak time 9.0 +- 0.7.
#include "validation_util.hpp"

#include "dns/gpu/device_util.hpp"

#include <cmath>
#include <cstdio>
#include <string>

int main(int argc, char** argv)
{
    const int n = argc > 1 ? std::atoi(argv[1]) : 64;
    if (n != 64 && n != 128) { std::printf("usage: %s 64|128\n", argv[0]); return 1; }
    if (!dns::gpu::deviceAvailable()) valtest::skip("no HIP device");

    const std::string out =
        std::string(DNS_VAL_SCRATCH) + "/tgv3d_" + std::to_string(n) + ".csv";
    const int rc = valtest::runBinaryTo(
        std::string(DNS_BIN) + " tgv --n " + std::to_string(n) + " --re 1600 --ft 20", out);
    valtest::check(rc == 0, "dns tgv exits 0");

    auto sim = valtest::loadCsv(out);
    if (!sim) { std::printf("FAIL: no output csv\n"); return 1; }
    const auto& t = sim->col("t");
    const auto& K = sim->col("K");
    const auto& eps = sim->col("epsVisc");

    // anchor: repo's own converged curve (file present, not deferred).
    const std::string refPath =
        std::string(DNS_VALIDATION_DIR) + "/ours_tgv" + std::to_string(n) + ".csv";
    auto ref = valtest::loadCsv(refPath);
    valtest::check(bool(ref), "reference curve " + refPath + " loads");
    if (ref)
    {
        const double dK = valtest::curveMaxRelDiff(ref->col("t"), ref->col("K"),
                                                   t, K, 0.0, 20.0);
        const double dE = valtest::curveMaxRelDiff(ref->col("t"), ref->col("epsVisc"),
                                                   t, eps, 0.0, 20.0);
        std::printf("anchor curve diff: K=%.2e epsVisc=%.2e\n", dK, dE);
        valtest::check(dK <= 2e-3, "K curve within 2e-3 of ours_tgv anchor");
        valtest::check(dE <= 2e-2, "epsVisc curve within 2e-2 of ours_tgv anchor");
    }

    // literature: spectral 512^3 dissipation peak (columns t, E, -dE/dt, ens).
    auto spec = valtest::loadCsv(DNS_SPECTRAL_REF);
    valtest::check(bool(spec), "spectral reference loads");
    if (spec)
    {
        const auto pRef = valtest::curvePeak(spec->col("c0"), spec->col("c2"));
        const auto pSim = valtest::curvePeak(t, eps);
        const double rel = pSim.value / pRef.value - 1.0;
        std::printf("dissipation peak: sim %.5f @ t=%.3f, spectral %.5f @ t=%.3f "
                    "(rel %.3f)\n", pSim.value, pSim.t, pRef.value, pRef.t, rel);
        if (n == 64)
        {
            valtest::check(pSim.t > 8.0 && pSim.t < 10.0, "peak time in [8.0, 10.0]");
            valtest::check(rel > -0.40 && rel < 0.05,
                           "peak value within [-40%, +5%] of spectral (64^3 trend)");
        }
        else
        {
            valtest::check(pSim.t > 8.3 && pSim.t < 9.7, "peak time 9.0 +- 0.7");
            valtest::check(rel > -0.10 && rel < 0.10,
                           "peak value within [-10%, +10%] of spectral");
        }
    }

    std::remove(out.c_str());
    return valtest::finish();
}
