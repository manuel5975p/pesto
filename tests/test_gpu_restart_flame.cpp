// dns flame checkpoint / restart end to end (dns/checkpoint.hpp, docs/chem.md
// "Checkpoint and restart"; masterplan real_bunsen S2). The byte rule: a run
// stopped by `--wall-limit 0` at each checkpoint and resumed with --restart
// (1 + 3 restarts) equals the uninterrupted run BYTE FOR BYTE in every output
// but the .json (stats, profiles, frames, volumes, slices, PNG).
//
// Cases (each straight vs chained, 4 jobs):
//   pre1d     premixed 1D, profiles + bin frames; between jobs 2 and 3 junk
//             is appended to every output (a job killed after its
//             checkpoint): the restart truncates it away
//   rich1d    --equivalence 1.3 (two species, the C2b --sl-law, zMin/zMax)
//   bunsen2d  slot Bunsen, per-row inflow with --pulse on, bin + PNG frames
//   round3d   --nozzle round, nz 32, volumes
//   tfles3d   premixed 3D --tfles 2 (F E, Omega recomputed per step)
//   candle    the diffusion runner under gravity
//   turbhold  premixed 3D --turb --hold-flame (sweep + controller state),
//             slices (FFTW builds only)
// and
//   grow      pre1d run to --tt 1.2 (complete: rows past its last checkpoint
//             at ~1.0, a forced end row), then --restart to --tt 2: equals
//             the straight tt 2 run (truncation + a larger --tt)
//   reject    on a checkpoint: a changed flag (exit 3, named: "--inflow-speed
//             (absent) vs 0.9"), a changed --emit set (exit 3), --tt not past
//             the checkpoint (exit 2), --restart != --out (exit 2), a
//             missing checkpoint (exit 3); none touches the outputs
//   sigint    SIGINT mid-run writes a checkpoint at the step boundary; the
//             restart equals the straight run (skipped, not failed, when the
//             signal lands before the time loop)
//
// Usage: test_gpu_restart_flame. No HIP device -> skip (77).
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "restart_test_util.hpp"

#include <cstdio>
#include <filesystem>
#include <format>
#include <string>

namespace {

namespace fs = std::filesystem;
using dnstest::check;
using namespace dnstest::restart;

const std::string kRoot = "/tmp/dns_test_restart_flame";

template <class Between>
int chained(const std::string& name, const std::string& args, double every,
            int wantJobs, Between between)
{
    return dnstest::restart::chained(kRoot, name, args, every, wantJobs, between);
}

int chained(const std::string& name, const std::string& args, double every,
            int wantJobs)
{
    return dnstest::restart::chained(kRoot, name, args, every, wantJobs);
}

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    std::error_code ec;
    fs::remove_all(kRoot, ec);
    fs::create_directories(kRoot);

    const std::string pre1d =
        "flame --tt 2 --profile-dt 0.5 --emit stats --emit bin --fdt 0.25";
    // pre1d with junk past the checkpoint before the third job.
    chained("pre1d", pre1d, 0.5, 4, [&](int job) {
        if (job == 2) appendJunk(kRoot + "/pre1d/chain");
    });
    chained("rich1d",
            "flame --equivalence 1.3 --stoich 3.99 --ox-y 0.233 --expansion "
            "6.88 --visc sutherland --tinf 298 --cells-per-delta 7.5 --tt 2 "
            "--profile-dt 1 --emit stats --emit bin --fdt 0.5",
            0.5, 4);
    chained("bunsen2d",
            "flame --mode bunsen --pulse 0.1 --pulse-freq 1 --tt 1 --emit "
            "stats --emit bin --emit png --fdt 0.25",
            0.25, 4);
    chained("round3d",
            "flame --mode bunsen --nozzle round --nz 32 --cells-per-delta 4 "
            "--slot-width 4 --coflow-width 2 --tt 0.5 --emit stats --emit "
            "volume --fdt 0.125",
            0.125, 4);
    chained("tfles3d",
            "flame --nz 8 --ny 8 --cells-per-delta 6 --tfles 2 --mach 0.02 "
            "--tt 0.3 --emit stats --emit volume --fdt 0.1",
            0.075, 4);
    chained("candle", "flame --mode candle --tt 1 --emit stats --emit bin --fdt 0.25",
            0.25, 4);
#if DNS_WITH_FFTW
    chained("turbhold",
            "flame --nz 24 --ny 24 --cells-per-delta 4 --turb 2 --turb-length 1 "
            "--hold-flame --tt 0.4 --emit stats --emit slices --fdt 0.1",
            0.1, 4);
#endif

    // grow: a complete tt 1.2 run continued to tt 2 from its last checkpoint.
    {
        const std::string d = kRoot + "/grow";
        fs::create_directories(d);
        const std::string ck = " --checkpoint-dt 0.5 --quiet --out " + d + "/r";
        const std::string p = "flame --profile-dt 0.5 --emit stats --emit bin --fdt 0.25";
        const int rc1 = runDns(p + " --tt 1.2" + ck, d + "/job0.log");
        const int rc2 = runDns(p + " --tt 2" + ck + " --restart " + d + "/r", d + "/job1.log");
        check(rc1 == 0 && rc2 == 0, "grow: tt 1.2 then --restart --tt 2 exit 0");
        // The straight tt 2 run of pre1d is the reference.
        const std::string ref = kRoot + "/grow_ref";
        fs::create_directories(ref);
        fs::copy(kRoot + "/pre1d/straight", ref, fs::copy_options::recursive, ec);
        check(rc1 == 0 && rc2 == 0 && sameOutputs(ref, d, "grow"),
              "grow: equals the straight tt 2 run (rows past the checkpoint "
              "and the forced tt 1.2 row truncated)");
    }

    // reject: on a fresh one-job checkpoint (t ~ 0.5).
    {
        const std::string d = kRoot + "/reject";
        fs::create_directories(d);
        const std::string ck = " --checkpoint-dt 0.5 --wall-limit 0";
        const std::string out = " --quiet --out " + d + "/r";
        const int rc0 = runDns(pre1d + ck + out, d + "/job0.log");
        check(rc0 == 0, "reject: first job exits 0");
        const auto before = artefacts(d);
        std::string snap;
        for (const auto& f : before) snap += slurp(d + "/" + f);
        const std::string rs = " --restart " + d + "/r";
        int rc = runDns(pre1d + " --inflow-speed 0.9" + ck + rs + out, d + "/r1.log");
        check(rc == 3 && slurp(d + "/r1.log").find(
                             "case differs from the checkpoint: --inflow-speed "
                             "(absent) vs 0.9") != std::string::npos,
              "reject: a changed flag is exit 3 naming it");
        rc = runDns(pre1d + " --emit png" + ck + rs + out, d + "/r2.log");
        check(rc == 3 && slurp(d + "/r2.log").find("--emit stats,bin vs "
                                                   "stats,bin,png") !=
                             std::string::npos,
              "reject: a changed --emit set is exit 3 naming it");
        rc = runDns("flame --tt 0.4 --profile-dt 0.5 --emit stats --emit bin --fdt 0.25" +
                     ck + rs + out,
                 d + "/r3.log");
        check(rc == 2 && slurp(d + "/r3.log").find("--tt") != std::string::npos,
              "reject: --tt not past the checkpoint is exit 2");
        rc = runDns(pre1d + ck + rs + " --quiet --out " + d + "/other", d + "/r4.log");
        check(rc == 2 && slurp(d + "/r4.log").find("must equal --out") !=
                             std::string::npos,
              "reject: --restart != --out is exit 2");
        rc = runDns(pre1d + ck + " --restart " + d + "/none --quiet --out " + d + "/none",
                 d + "/r5.log");
        check(rc == 3, "reject: a missing checkpoint is exit 3");
        std::string after;
        for (const auto& f : artefacts(d)) after += slurp(d + "/" + f);
        check(artefacts(d) == before && after == snap,
              "reject: no rejected restart touched the outputs");
    }

    // sigint: SIGINT ~1 s into a run (GPU set-up included), then resume.
    {
        const std::string d = kRoot + "/sigint";
        fs::create_directories(d);
        const std::string out = " --quiet --out " + d + "/r";
        const std::string cmd = "timeout --preserve-status -s INT 1.0 \"" DNS_BIN "\" " + pre1d +
                                " --checkpoint-dt 0.5" + out + " > \"" + d +
                                "/job0.log\" 2>&1";
        const int raw = std::system(cmd.c_str());
        const int rc0 = raw == -1 || !WIFEXITED(raw) ? -1 : WEXITSTATUS(raw);
        if (!fs::exists(d + "/r_ckpt.bin")) {
            std::printf("  sigint: no checkpoint (rc %d: the signal landed "
                        "before the time loop) -- not checked\n",
                        rc0);
        } else {
            const bool stopped = jsonHas(d + "/r", "\"complete\":false");
            const int rc1 = runDns(pre1d + " --checkpoint-dt 0.5 --restart " + d + "/r" + out,
                                d + "/job1.log");
            check(rc0 == 0 && rc1 == 0, "sigint: the stopped and the resumed job exit 0");
            std::printf("  sigint: first job %s\n",
                        stopped ? "stopped by the signal" : "completed before it");
            check(sameOutputs(kRoot + "/pre1d/straight", d, "sigint"),
                  "sigint: resumed outputs byte-identical to the straight run");
        }
    }
    return dnstest::finish();
}
