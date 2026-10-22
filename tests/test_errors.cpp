// The exit-code contract of the `dns` binary (masterplan §5):
//   0 ok · 1 CLI · 2 config · 3 input (mesh) · 4 output · 5 no GPU ·
//   6 resource · 7 numerical failure · 8 solver non-convergence · 77 skipped.
//
// Each row below drives one error path to its code through the real binary.
// The point is that these are ERRORS at all: before the recomposition an
// unrecognised token silently became the grid size, a bad number silently
// became the default, and a run whose output directory could not be written
// discovered it only after the GPU work.
//
// Every case here is reachable without a GPU (phases 1-3a), so this test does
// not skip on a device-less machine.

#include "check.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

#include <csignal>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {


int run(const std::string& args)
{
    const std::string cmd =
        "\"" DNS_BIN "\" " + args + " > /dev/null 2>&1";
    const int rc = std::system(cmd.c_str());
    if (rc == -1 || !WIFEXITED(rc)) return -1;
    return WEXITSTATUS(rc);
}

void expect(const std::string& args, int want, const char* what)
{
    const int got = run(args);
    const bool ok = got == want;
    std::printf("%-56s want %d got %d  %s\n", what, want, got, ok ? "ok" : "FAIL");
    if (!ok) ++dnstest::gFailures;
}

} // namespace

int main()
{
    // --- 0: the happy paths -------------------------------------------------
    expect("--help", 0, "--help");
    expect("flow --help", 0, "flow --help");
    expect("flow --body cylinder --dry-run", 0, "a valid case, --dry-run");

    // --- 1: CLI -------------------------------------------------------------
    expect("flow --oepn --dry-run", 1, "unknown option");
    expect("", 1, "no subcommand");
    expect("wobble --dry-run", 1, "unknown subcommand");
    expect("flow --n --dry-run", 1, "option with a missing value");
    expect("flow --body cylinder --n banana --dry-run", 1, "unparseable number");
    expect("flow --body cylinder --n 64x --dry-run", 1,
           "trailing garbage after a number");
    expect("flow --body cylinder --re nan --dry-run", 1, "a NaN number");
    expect("flow --body cylinder --physics compressible --mach inf --dry-run", 1,
           "an infinite number");

    // --- 2: config ----------------------------------------------------------
    expect("flow --dry-run", 2, "--body is required");
    expect("flow --body wombat --dry-run", 2, "unrecognised body");
    expect("flow --body cylinder --re -1 --dry-run", 2, "--re must be > 0");
    expect("flow --body cylinder --cfl 0 --dry-run", 2, "--cfl must be in (0,1]");
    expect("flow --body cylinder --nz 2 --dry-run", 2, "--nz must be 1 or >= 4");
    expect("flow --body sphere --nz 1 --dry-run", 2,
           "a 3D body with --nz 1 is rejected");
    expect("flow --body cylinder --les 0.1 --nz 8 --dry-run", 0,
           "incompressible --les is 3D-capable");
    expect("flow --body cylinder --les 1.35 --sgs sigma --nz 8 --dry-run", 0,
           "--sgs sigma with --les and nz > 1 is accepted");
    expect("flow --body cylinder --sgs sigma --nz 8 --dry-run", 2,
           "--sgs without --les is rejected");
    expect("flow --body cylinder --les 1.35 --sgs sigma --nz 1 --dry-run", 2,
           "--sgs sigma at nz == 1 is rejected (D_sigma == 0 in 2D)");
    expect("flow --body cylinder --les 0.17 --sgs wale --nz 8 --dry-run", 1,
           "unknown --sgs value is a CLI error");
    expect("flow --body cylinder --mach 0.2 --dry-run", 2,
           "--mach is compressible-only");
    expect("flow --physics compressible --body cylinder --far tunnel --dry-run", 2,
           "--far tunnel is incompressible-only");
    expect("flow --body cylinder --far tunnel --open --dry-run", 0,
           "--open turns the tunnel laterals into open faces");
    expect("flow --body cylinder --far fringe --open --dry-run", 2,
           "--open is rejected, not ignored, with fringe");
    expect("flow --physics compressible --body cylinder --far nscbc "
           "--nscbc-beta 1.0 --dry-run", 2,
           "--nscbc-beta 1.0 is rejected (measured divergence)");
    expect("flow --physics compressible --body cylinder --far nscbc "
           "--re 2000 --dry-run", 0,
           "above Re 800 the DEFAULT sink band satisfies the strip rule");
    expect("flow --physics compressible --body cylinder --far nscbc "
           "--re 2000 --nscbc-sink 0 --dry-run", 2,
           "bare NSCBC above Re 800 (sink explicitly off) is rejected");
    expect("flow --physics compressible --body cylinder --far nscbc "
           "--nscbc-abc modal --dry-run", 0,
           "--nscbc-abc modal validates on the shipped walled-lateral case");
    expect("flow --physics compressible --body cylinder --far sponge "
           "--nscbc-abc modal --dry-run", 2,
           "--nscbc-abc requires --far nscbc");
    expect("flow --physics compressible --body cylinder --far nscbc --open "
           "--nscbc-abc modal --dry-run", 2,
           "--nscbc-abc modal is rejected, not ignored, with open laterals");
    // adaptive forms the same ratio inside the NSCBC kernel, so it needs no
    // separable plane and IS legal there -- which is the all-open box a static
    // rotor runs in.
    expect("flow --physics compressible --body cylinder --far nscbc --open "
           "--nscbc-abc adaptive --dry-run", 0,
           "--nscbc-abc adaptive works on open lateral faces");
    expect("flow --physics compressible --body floor --far nscbc "
           "--nscbc-abc modal --dry-run", 2,
           "--nscbc-abc is rejected where the outflow plane is part solid");
    expect("flow --physics compressible --body cylinder --far sponge --pot "
           "--dry-run", 2,
           "--pot requires an open far field (sponge rejected)");
    expect("flow --body cylinder --far tunnel --open --pot --dry-run", 0,
           "--pot works with the incompressible open tunnel");
    expect("flow --physics compressible --body cylinder --dr 3 --dry-run", 2,
           "compressible --dr below 4 is rejected");
    expect("flow --physics compressible --body cylinder --far nscbc --dr 16 "
           "--dry-run", 0,
           "compressible --dr widens the box");
    expect("flow --body cylinder --out '' --dry-run", 2, "an empty --out");
    expect("flow --body cylinder --dr 1e9 --dry-run", 2,
           "a body under 4 cells across");
    expect("flow --physics compressible --body none --far sponge --stretch 2 "
           "--noslip --n 128 --ny 128 --dry-run", 0,
           "body-free stretched no-slip wall case validates");
    expect("flow --physics compressible --body none --stretch 2 --n 128 "
           "--dry-run", 0,
           "--stretch with the default nscbc far field validates");
    expect("flow --physics compressible --body none --far periodic --stretch 2 "
           "--n 128 --dry-run", 2,
           "--stretch with a wall-less far field is rejected");
    expect("flow --body none --n 128 --dry-run", 2,
           "--body none is compressible-only");
    expect("flow --body cylinder --noslip --dry-run", 2,
           "--noslip is compressible-only");
    expect("flow --body fence --dr 10 --re 2000 --n 1024 --dry-run", 0,
           "the fence shear-layer rig validates in native 2D");
    expect("flow --body fence --physics compressible --n 512 --dry-run", 2,
           "fence is incompressible-only");
    expect("flow --body floor --dr 10 --re 2000 --n 1024 --dry-run", 0,
           "the raised-floor wall-layer rig validates in native 2D");
    expect("flow --body floor --far fringe --n 512 --dry-run", 2,
           "floor needs the tunnel wall to stand on");
    expect("tgv --nz 2 --dry-run", 2, "tgv --nz must be 1 or >= 4");
    expect("hit --nz 1 --dry-run", 2, "hit needs nz >= 4");
    expect("bench --steps 0 --dry-run", 2, "--steps must be >= 1");
    // bench builds an n^3 box and used to skip the grid guards every other
    // branch applies: `--n 1300` overflowed the int indexing, and `--n 1` made
    // nz == 1 with hz == 2pi, violating the enforced `nz == 1 => hz == 1`.
    expect("bench --n 1 --dry-run", 2, "bench --n 1 is rejected (hz == 1 rule)");
    expect("bench --n 2 --dry-run", 2, "bench --n below 4 is rejected");
    expect("bench --n 1300 --dry-run", 2, "bench --n 1300 exceeds INT_MAX cells");
    expect("bench --n 64 --steps 1 --dry-run", 0, "a sane bench still validates");

    // --- 3: input -----------------------------------------------------------
    expect("flow --body /nonexistent/model.stl --n 32 --nz 32", 3,
           "a mesh file that does not exist");
    {
        const auto tmp = std::filesystem::temp_directory_path() / "dns_bad.stl";
        std::ofstream(tmp) << "this is not a mesh";
        expect("flow --body " + tmp.string() + " --n 32 --nz 32", 3,
               "a file that is not a mesh");
        std::filesystem::remove(tmp);
    }
    {
        // Every triangle a sliver: MeshSdf's precondition, checked on input.
        const auto tmp = std::filesystem::temp_directory_path() / "dns_sliver.stl";
        std::ofstream(tmp) << "solid s\n facet normal 0 0 0\n  outer loop\n"
                              "   vertex 0 0 0\n   vertex 1 0 0\n   vertex 2 0 0\n"
                              "  endloop\n endfacet\nendsolid s\n";
        expect("flow --body " + tmp.string() + " --n 32 --nz 32", 3,
               "a mesh with no triangle of nonzero area");
        std::filesystem::remove(tmp);
    }
    {
        // A NaN vertex in a binary STL: rejected by the parser itself.
        const auto tmp = std::filesystem::temp_directory_path() / "dns_nan.stl";
        {
            std::ofstream f(tmp, std::ios::binary);
            const char hdr[80] = {};
            f.write(hdr, 80);
            const unsigned n = 1;
            f.write(reinterpret_cast<const char*>(&n), 4);
            const float nan = std::numeric_limits<float>::quiet_NaN();
            const float rec[12] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, nan, 0};
            f.write(reinterpret_cast<const char*>(rec), sizeof rec);
            const char attr[2] = {};
            f.write(attr, 2);
        }
        expect("flow --body " + tmp.string() + " --n 32 --nz 32", 3,
               "a binary STL with a NaN vertex");
        std::filesystem::remove(tmp);
    }

    // --- 4: output ----------------------------------------------------------
    // Phase 3c creates and truncates EVERY output before the first hipMalloc,
    // so an unwritable destination is exit 4 in the first second rather than a
    // surprise hours into a run. This is the only case here that needs a
    // device: `dns flow` probes the GPU (phase 3a) before it opens files, so
    // without one the run exits 5 first and the check is skipped rather than
    // failed.
    {
        const int rc = run("flow --body cylinder --n 32 --nz 1 --ft 0.01 "
                           "--out /proc/nonexistent/x");
        if (rc == 5)
            std::printf("SKIP  exit 4 (unwritable --out): no GPU, exits 5 first\n");
        else if (rc != 4)
        {
            std::printf("FAIL  exit 4 (unwritable --out): got %d\n", rc);
            ++dnstest::gFailures;
        }
        else
            std::printf("ok    exit 4 (unwritable --out)\n");
    }
    // A write that fails MID-RUN (a full disk) must not end in exit 0: the
    // run continues, and finish() turns the recorded failure into exit 4.
    if (std::filesystem::exists("/dev/full"))
    {
        const auto dir = std::filesystem::temp_directory_path() / "dns_full_run";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        std::filesystem::create_symlink("/dev/full", dir / "x_stats.csv");
        const int rc = run("conv --n 16 --ny 8 --nz 1 --tt 0.05 --emit stats "
                           "--out " + (dir / "x").string());
        if (rc == 5)
            std::printf("SKIP  exit 4 (full disk mid-run): no GPU\n");
        else if (rc != 4)
        {
            std::printf("FAIL  exit 4 (full disk mid-run): got %d\n", rc);
            ++dnstest::gFailures;
        }
        else
            std::printf("ok    exit 4 (full disk mid-run)\n");
        std::filesystem::remove_all(dir);
    }

    // A stop signal ends a periodic run like reaching tEnd does (exit 0). The
    // tgv/hit loops used to ignore it, so this `hit` -- far too long to finish
    // -- ran until killed.
    {
        const std::string prefix =
            (std::filesystem::temp_directory_path() / "dns_hit_sigint").string();
        const pid_t pid = fork();
        if (pid == 0)
        {
            const int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0)
            {
                dup2(devnull, 1);
                dup2(devnull, 2);
            }
            execl(DNS_BIN, DNS_BIN, "hit", "--n", "32", "--spin", "1e4",
                  "--stats", "1e4", "--emit", "none", "--out", prefix.c_str(),
                  static_cast<char*>(nullptr));
            _exit(127);
        }
        // Let it reach its time loop, then ask it to stop; give it a bounded
        // time to unwind before calling it a hang.
        usleep(3000000);
        kill(pid, SIGINT);
        int status = 0, waited = 0;
        while (waitpid(pid, &status, WNOHANG) == 0 && waited < 600)
        {
            usleep(100000);
            ++waited;
        }
        if (waited >= 600)
        {
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            std::printf("FAIL  SIGINT stops dns hit: still running after 60 s\n");
            ++dnstest::gFailures;
        }
        else
        {
            const int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
            if (rc == 5)
                std::printf("SKIP  SIGINT stops dns hit: no GPU\n");
            else if (rc != 0)
            {
                std::printf("FAIL  SIGINT stops dns hit: exit %d\n", rc);
                ++dnstest::gFailures;
            }
            else
                std::printf("ok    SIGINT stops dns hit (exit 0)\n");
        }
    }

    std::printf("%s\n", dnstest::gFailures == 0 ? "ALL OK" : "FAILURES");
    return dnstest::finish();
}
