// dns flame --physics lowmach checkpoint / restart (masterplan real_bunsen
// S4; S2's byte rule on runLowMach): a run stopped by `--wall-limit 0` at each
// checkpoint and resumed with --restart (1 + 3 restarts) equals the straight
// run BYTE FOR BYTE in every output but the .json.
//
//   lm1d    premixed 1D, profiles + bin frames; junk appended past the second
//           checkpoint (truncated away by the restart)
//   lmrich  --equivalence 1.3 (two species, the C2b law, dzMin/dzMax)
//   lm2d    a wrinkled 2D flame (ny 32), bin frames
//   lm3d    a 3D planar flame (ny = nz = 16), volumes
//   lmb2d   (S5) the low-Mach 2D slot Bunsen, bin frames
//   lmbrich (S5) the partially premixed slot (--equivalence, pilots, air)
//   lmbround (S5) the partially premixed round nozzle (nz 40), volumes
//   lmbopen (S6) the partially premixed slot with --laterals open
//   lmbjet  (S6) the inert open jet (--reaction off), jet profiles + frames
//   lmbropen (S6) the round nozzle, y and z open, volumes
//   lmbgrav (S7) the open partially premixed slot under --gravity 0.5
//   lmbrgrav (S7) the open round nozzle under --gravity 0.5, volumes
//   lmbrim  (S8) the 2D rich slot on its immersed tube rim (open), bin frames
//           (the PCG warm starts are checkpointed: without them the resumed
//           run's first solves would start from 0, other iterates)
//   lmbrrim (S8) the round tube, open, --gravity 0.5, volumes
//   gravdiff (S7) a gravity checkpoint restarted without --gravity or with
//           another g: exit 3 naming --gravity
//   solver  a compressible checkpoint restarted with --physics lowmach (and
//           back) is exit 3 naming both solvers
//
// Usage: test_gpu_lowmach_restart. No HIP device -> skip (77).
#include "dns/gpu/device_util.hpp"
#include "gpu_test_util.hpp"
#include "restart_test_util.hpp"

#include <cstdio>
#include <filesystem>
#include <string>

namespace {

namespace fs = std::filesystem;
using dnstest::check;
using namespace dnstest::restart;

const std::string kRoot = "/tmp/dns_test_lowmach_restart";

} // namespace

int main()
{
    dns::test::requireDeviceOrSkip();
    std::error_code ec;
    fs::remove_all(kRoot, ec);
    fs::create_directories(kRoot);

    const std::string lm = "flame --physics lowmach ";
    chained(kRoot, "lm1d",
            lm + "--cells-per-delta 8 --tt 2 --profile-dt 0.5 --emit stats "
                 "--emit bin --fdt 0.25",
            0.5, 4, [&](int job) {
                if (job == 2) appendJunk(kRoot + "/lm1d/chain");
            });
    chained(kRoot, "lmrich",
            lm + "--equivalence 1.3 --expansion 6.88 --visc sutherland --tinf "
                 "298 --cells-per-delta 6 --tt 2 --profile-dt 1 --emit stats "
                 "--emit bin --fdt 0.5",
            0.5, 4);
    chained(kRoot, "lm2d",
            lm + "--ny 32 --cells-per-delta 6 --wrinkle 1 --wrinkle-modes 2 "
                 "--tt 1 --emit stats --emit bin --fdt 0.25",
            0.25, 4);
    chained(kRoot, "lm3d",
            lm + "--ny 16 --nz 16 --cells-per-delta 6 --tt 0.5 --emit stats "
                 "--emit volume --fdt 0.125",
            0.125, 4);
    // S5: the low-Mach Bunsen (per-row inflow, cone stats, speed frames), the
    // partially premixed burner (findex, oxid, the C2 columns) and the round
    // nozzle (volumes).
    chained(kRoot, "lmb2d",
            lm + "--mode bunsen --cells-per-delta 4 --slot-width 6 --coflow-width "
                 "3 --jet-speed 3 --tt 1 --emit stats --emit bin --fdt 0.25",
            0.25, 4);
    chained(kRoot, "lmbrich",
            lm + "--mode bunsen --equivalence 1.3 --expansion 6.88 --visc "
                 "sutherland --tinf 298 --prandtl 0.52 --cells-per-delta 4 "
                 "--slot-width 4 --jet-speed 3 --pilot-width 1.5 --pilot-speed 3 "
                 "--coflow-width 3 --tt 1 --emit stats --emit bin --fdt 0.25",
            0.25, 4);
    chained(kRoot, "lmbround",
            lm + "--mode bunsen --nozzle round --nz 40 --equivalence 1.3 "
                 "--expansion 6.88 --visc sutherland --tinf 298 --prandtl 0.52 "
                 "--cells-per-delta 4 --slot-width 4 --pilot-width 1 "
                 "--pilot-speed 3 --coflow-width 2 --jet-speed 3 --tt 0.5 "
                 "--emit stats --emit volume --fdt 0.125",
            0.125, 4);
    // S6: open laterals (the hi-face mass fluxes, the species tallies, the
    // open stats columns), the inert jet with its jet profiles, and the open
    // round nozzle (y and z open).
    chained(kRoot, "lmbopen",
            lm + "--mode bunsen --laterals open --equivalence 1.3 --expansion "
                 "6.88 --visc sutherland --tinf 298 --prandtl 0.52 "
                 "--cells-per-delta 4 --slot-width 4 --jet-speed 3 --pilot-width "
                 "1.5 --pilot-speed 3 --coflow-width 3 --tt 1 --emit stats "
                 "--emit bin --fdt 0.25",
            0.25, 4);
    chained(kRoot, "lmbjet",
            lm + "--mode bunsen --laterals open --reaction off --coflow-temp 1 "
                 "--coflow-speed 0 --cells-per-delta 4 --slot-width 4 "
                 "--jet-speed 3 --coflow-width 6 --length 20 --tt 4 "
                 "--profile-dt 0.5 --emit stats --emit bin --fdt 1",
            1.0, 4);
    chained(kRoot, "lmbropen",
            lm + "--mode bunsen --laterals open --nozzle round --nz 40 "
                 "--equivalence 1.3 --expansion 6.88 --visc sutherland --tinf "
                 "298 --prandtl 0.52 --cells-per-delta 4 --slot-width 4 "
                 "--pilot-width 1 --pilot-speed 3 --coflow-width 2 --jet-speed 3 "
                 "--tt 0.5 --emit stats --emit volume --fdt 0.125",
            0.125, 4);
    // S7: gravity (the Grav momentum instantiation; --gravity in the case
    // signature) on the open partially premixed slot and the open round
    // nozzle.
    chained(kRoot, "lmbgrav",
            lm + "--mode bunsen --laterals open --gravity 0.5 --equivalence 1.3 "
                 "--expansion 6.88 --visc sutherland --tinf 298 --prandtl 0.52 "
                 "--cells-per-delta 4 --slot-width 4 --jet-speed 3 --pilot-width "
                 "1.5 --pilot-speed 3 --coflow-width 3 --tt 1 --emit stats "
                 "--emit bin --fdt 0.25",
            0.25, 4);
    chained(kRoot, "lmbrgrav",
            lm + "--mode bunsen --laterals open --gravity 0.5 --nozzle round --nz 40 "
                 "--equivalence 1.3 --expansion 6.88 --visc sutherland --tinf "
                 "298 --prandtl 0.52 --cells-per-delta 4 --slot-width 4 "
                 "--pilot-width 1 --pilot-speed 3 --coflow-width 2 --jet-speed 3 "
                 "--tt 0.5 --emit stats --emit volume --fdt 0.125",
            0.125, 4);
    // S8: the immersed tube (the Wall instantiations, the apertured PCG and
    // its per-stage warm starts "state.phi0..2" in the checkpoint): the 2D
    // rich slot on its rim and the 3D round tube under gravity, open.
    chained(kRoot, "lmbrim",
            lm + "--mode bunsen --rim 1 --laterals open --equivalence 1.3 "
                 "--expansion 6.88 --visc sutherland --tinf 298 --prandtl 0.41 "
                 "--cells-per-delta 4 --slot-width 6 --jet-speed 4 --tube-length 3 "
                 "--coflow-width 3 --tt 1 --emit stats --emit bin --fdt 0.25",
            0.25, 4);
    chained(kRoot, "lmbrrim",
            lm + "--mode bunsen --rim 1 --laterals open --gravity 0.5 --nozzle round "
                 "--nz 40 --equivalence 1.3 --expansion 6.88 --visc sutherland "
                 "--tinf 298 --prandtl 0.41 --cells-per-delta 4 --slot-width 4 "
                 "--jet-speed 3 --tube-length 3 --coflow-width 2 --chem-dt 0.5 "
                 "--tt 0.5 --emit stats --emit volume --fdt 0.125",
            0.125, 4);
    // A checkpoint written with gravity, restarted without (and with
    // another g): exit 3 naming --gravity.
    {
        const std::string d = kRoot + "/gravdiff";
        fs::create_directories(d);
        const std::string base = lm + "--mode bunsen --laterals open --cells-per-delta "
                                      "4 --slot-width 6 --coflow-width 3 --jet-speed 3 "
                                      "--tt 1 --emit stats --checkpoint-dt 0.25 "
                                      "--wall-limit 0";
        const int rc0 =
            runDns(base + " --gravity 0.5 --quiet --out " + d + "/r", d + "/r0.log");
        const int rc1 = runDns(base + " --restart " + d + "/r --quiet --out " + d + "/r",
                               d + "/r1.log");
        const int rc2 = runDns(base + " --gravity 0.25 --restart " + d +
                                   "/r --quiet --out " + d + "/r",
                               d + "/r2.log");
        check(rc0 == 0 && rc1 == 3 && rc2 == 3 &&
                  slurp(d + "/r1.log").find("--gravity") != std::string::npos &&
                  slurp(d + "/r2.log").find("--gravity 0.5 vs 0.25") != std::string::npos,
              "gravdiff: restarting a gravity checkpoint without --gravity or with "
              "another g is exit 3 naming --gravity");
    }

    // solver: the checkpoint's solver name against --physics (phase 1.5).
    {
        const std::string d = kRoot + "/solver";
        fs::create_directories(d);
        const std::string ck = " --checkpoint-dt 0.5 --wall-limit 0";
        const std::string comp = "flame --tt 2 --emit stats";
        const int rc0 = runDns(comp + ck + " --quiet --out " + d + "/c", d + "/c0.log");
        int rc = runDns(lm + "--tt 2 --emit stats" + ck + " --restart " + d +
                            "/c --quiet --out " + d + "/c",
                        d + "/c1.log");
        check(rc0 == 0 && rc == 3 &&
                  slurp(d + "/c1.log").find("is a 'flame-compressible' "
                                            "checkpoint; this run is "
                                            "flame-lowmach") != std::string::npos,
              "solver: a compressible checkpoint under --physics lowmach is "
              "exit 3 naming both");
        const int rc2 = runDns(lm + "--tt 2 --emit stats" + ck + " --quiet --out " +
                                   d + "/l",
                               d + "/l0.log");
        rc = runDns(comp + ck + " --restart " + d + "/l --quiet --out " + d + "/l",
                    d + "/l1.log");
        check(rc2 == 0 && rc == 3 &&
                  slurp(d + "/l1.log").find("is a 'flame-lowmach' checkpoint; "
                                            "this run is flame-compressible") !=
                      std::string::npos,
              "solver: a low-Mach checkpoint under the compressible solver is "
              "exit 3 naming both");
    }
    return dnstest::finish();
}
