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

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <sys/wait.h>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace {

bool pass = true;

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
    if (!ok) pass = false;
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
           "--re 2000 --dry-run", 2,
           "pure NSCBC above Re 800 is rejected");
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

    std::printf("%s\n", pass ? "ALL OK" : "FAILURES");
    return pass ? 0 : 1;
}
