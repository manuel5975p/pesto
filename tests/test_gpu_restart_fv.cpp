// dns flame checkpoint / restart on the FV routes (masterplan_multiphysics M5;
// dns/checkpoint.hpp, docs/chem.md "Checkpoint and restart"): runFvComp and
// runFvLowMach. The byte rule: a run stopped by `--wall-limit 0` at each
// checkpoint and resumed with --restart equals the uninterrupted run BYTE FOR
// BYTE in every output but the .json (stats, profiles, bin frames); a straight
// run that writes checkpoints (no wall limit) equals it too, and its LAST
// checkpoint (state words, loop, output resume point) equals the chain's byte
// for byte but the restart count. Teeth: a carried record ZEROED in a
// checkpoint (c2b's compensated low words; lm_tfles's PCG warm start p and
// face mass flux phi) makes the resumed outputs differ -- the records are state
// and the byte gate sees them.
//
// Group `comp` (the compressible FV flame, runFvComp):
//   les3d   premixed 3D box, LES sigma + TFLES 2 (dynamic sensor, Charlette):
//           eddy and TFLES fields of each step's input state, not carried
//   c2b     --equivalence 1.3 (model C2b), --compensated: the low words
//   fp64    C0 at --fv-precision fp64 (F64 words), profiles; junk appended to
//           every output between jobs 2 and 3 (a job killed after its
//           checkpoint): the restart truncates it
//   fp64c   C0 at --fv-precision fp64 --compensated (masterplan_combclose R6): F64 q and
//           lo words; teeth on state.fv.lo (the final state words move: the printed
//           outputs do not resolve an FP64 low word over 1 tau); a restart without
//           --compensated is exit 3
//   jet     --mode diffusion (the jet's per-face rows, characteristic outflow)
//   turb    --turb 4 (3D box 24^2): the inflow rows of the box swept to X = sum U dt,
//           X carried in the loop record "sweep" (restarting at X 0 replays the
//           start column)
//   candle  --mode candle: the jet under gravity, the hydrostatic ambient's
//           per-face rows (rebuilt from the flags, not carried), Far top
//   poly    --polymesh: a 30-degree sheared box, Far / burnt Outflow
//   reject  on poly's checkpoint: the polyMesh content changed, the --bc
//           file changed (exit 3, "the mesh differs"); an FV checkpoint into
//           the structured flame and back (exit 3, the solver by name); none
//           touches the outputs
// Group `lm` (the low-Mach FV flame, runFvLowMach):
//   burner  --mode bunsen --laterals open --rim 4 --gravity 1: per-face rows,
//           the open faces' rule, the rim wall, the face force; the PCG warm
//           start p and the face mass flux phi are checkpointed
//   les3d   planar 3D box, LES sigma (refreshEddy: mu_t of the current state)
//   tfles   planar, TFLES 2, profiles
//   poly    --polymesh: the sheared box at --fv-precision fp64 (F64 words)
//   reject  the polyMesh content changed (exit 3)
//
// Group `thrust` (combscale S1: the reacting nozzle's [thrust] impulse ledger, runFvComp):
//   host    a loop struct with a std::array<double, 6> and a bool through saveLoop/loadLoop
//           (before the device check); a mis-sized array record is an error by name
//   thrust64, thrust64c  tools/fv/nozzle_react.py design --cells 15 (Total inlet, riemann exit,
//           [nozzle], [thrust] station), --shock hll, FP64 plain and compensated: straight ==
//           4-job chain == checkpointing straight, _stats/_thrust/_profiles.csv BYTES, last
//           checkpoints equal but the restart count
//   teeth   on thrust64c, one at a time: state.fv.ledger.I, loop.thrust.P0, loop.thrust.prevExit
//           zeroed (each non-zero before) moves _thrust.csv
//   reject  on a one-job thrust64 checkpoint (exit 3, outputs untouched): [thrust] pa changed,
//           [thrust] removed, [nozzle] astar one ulp off (each named by its section, before the
//           --bc sha), a comment line appended (the mesh identity), [nozzle] cd = 1 added
//           (combscale S5: named by [nozzle]); a thrust64c checkpoint into thrust64 (--compensated by name)
//   loop.shockHot (--shock hll's sticky dt cadence) has no CLI-reachable teeth: the nozzle's
//   sensor marks 0 cells over these runs, so it stays false; carried all the same.
//
// Usage: test_gpu_restart_fv comp|lm|fp64c|thrust. No HIP device -> skip (77).
#include "dns/checkpoint.hpp"
#include "dns/gpu/device_util.hpp"
#include "dns/gpu/internal/checkpoint.hpp"
#include "gpu_test_util.hpp"
#include "restart_test_util.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using dnstest::check;
using namespace dnstest::restart;

// The group's root (main: /tmp/dns_test_restart_fv/GROUP, cleared first).
std::string kRoot;

// One ASCII polyMesh list file (tools/chem/fv_flame_mesh.py's layout).
void writeList(const fs::path& p, const char* obj, const char* cls, const std::vector<std::string>& items,
               const std::string& note = "")
{
    std::ofstream f(p);
    f << "FoamFile\n{\n    version     2.0;\n    format      ascii;\n    arch        \"LSB;label=32;scalar=64\";\n"
      << note << "    class       " << cls << ";\n    location    \"constant/polyMesh\";\n    object      " << obj << ";\n}\n\n"
      << items.size() << "\n(\n";
    for (const std::string& s : items) f << s << "\n";
    f << ")\n";
}

// The planar flame's polyMesh, tools/chem/fv_flame_mesh.py --shear 30 ported
// (a 180 x 8 box [0, 36.0371] x [0, 1.60165], y' = y + x tan 30 deg: every face
// non-orthogonal; x-lo, x-hi open, y cyclic translational, one empty-extruded
// cell) and its --bc file.
void writeFlameMesh(const std::string& dir, bool lowMach)
{
    const int nx = 180, ny = 8;
    const double lx = 36.0371, ly = 1.60165, hx = lx / nx, hy = ly / ny, ts = std::tan(std::numbers::pi / 6);
    auto pid = [&](int i, int j, int k) { return (i + (nx + 1) * j) * 2 + k; };
    auto cid = [&](int i, int j) { return i + nx * j; };
    auto q = [](int a, int b, int c, int d) { return std::format("4({} {} {} {})", a, b, c, d); };
    std::vector<std::string> pts, faces, own, nei;
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i)
            for (int k = 0; k < 2; ++k) pts.push_back(std::format("({:.17g} {:.17g} {})", i * hx, j * hy + i * hx * ts, k));
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            if (i + 1 < nx) {
                faces.push_back(q(pid(i + 1, j, 0), pid(i + 1, j + 1, 0), pid(i + 1, j + 1, 1), pid(i + 1, j, 1)));
                own.push_back(std::to_string(cid(i, j)));
                nei.push_back(std::to_string(cid(i + 1, j)));
            }
            if (j + 1 < ny) {
                faces.push_back(q(pid(i, j + 1, 0), pid(i, j + 1, 1), pid(i + 1, j + 1, 1), pid(i + 1, j + 1, 0)));
                own.push_back(std::to_string(cid(i, j)));
                nei.push_back(std::to_string(cid(i, j + 1)));
            }
        }
    std::vector<std::string> bnd;
    auto patch = [&](const std::string& name, const std::string& body, const std::vector<std::pair<std::string, int>>& fl) {
        bnd.push_back(std::format("    {}\n    {{\n        {}\n        nFaces {};\n        startFace {};\n    }}", name, body,
                                  fl.size(), faces.size()));
        for (const auto& [f, c] : fl) {
            faces.push_back(f);
            own.push_back(std::to_string(c));
        }
    };
    std::vector<std::pair<std::string, int>> fl;
    for (int j = 0; j < ny; ++j) fl.emplace_back(q(pid(0, j, 0), pid(0, j, 1), pid(0, j + 1, 1), pid(0, j + 1, 0)), cid(0, j));
    patch("xlo", "type patch;", fl);
    fl.clear();
    for (int j = 0; j < ny; ++j)
        fl.emplace_back(q(pid(nx, j, 0), pid(nx, j + 1, 0), pid(nx, j + 1, 1), pid(nx, j, 1)), cid(nx - 1, j));
    patch("xhi", "type patch;", fl);
    fl.clear();
    for (int i = 0; i < nx; ++i) fl.emplace_back(q(pid(i, 0, 0), pid(i + 1, 0, 0), pid(i + 1, 0, 1), pid(i, 0, 1)), cid(i, 0));
    patch("ylo", std::format("type cyclic; neighbourPatch yhi; transform translational; separationVector (0 {:.17g} 0);", ly),
          fl);
    fl.clear();
    for (int i = 0; i < nx; ++i)
        fl.emplace_back(q(pid(i, ny, 0), pid(i, ny, 1), pid(i + 1, ny, 1), pid(i + 1, ny, 0)), cid(i, ny - 1));
    patch("yhi", std::format("type cyclic; neighbourPatch ylo; transform translational; separationVector (0 {:.17g} 0);", -ly),
          fl);
    fl.clear();
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            fl.emplace_back(q(pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0)), cid(i, j));
            fl.emplace_back(q(pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1), pid(i, j + 1, 1)), cid(i, j));
        }
    patch("frontAndBack", "type empty;", fl);
    const fs::path d = fs::path(dir) / "constant" / "polyMesh";
    fs::create_directories(d);
    writeList(d / "points", "points", "vectorField", pts);
    writeList(d / "faces", "faces", "faceList", faces);
    const std::string note = std::format("    note        \"nPoints:{} nCells:{} nFaces:{} nInternalFaces:{}\";\n",
                                         pts.size(), nx * ny, faces.size(), nei.size());
    writeList(d / "owner", "owner", "labelList", own, note);
    writeList(d / "neighbour", "neighbour", "labelList", nei, note);
    writeList(d / "boundary", "boundary", "polyBoundaryMesh", bnd);
    std::ofstream bc(dir + "/bc.toml");
    bc << "[reference]\ndirection = [1.0, 0.0, 0.0]\nlift = [0.0, 1.0, 0.0]\n";
    if (lowMach) bc << "[patch.xlo]\ntype = \"inflow\"\ny = [1.0]\n[patch.xhi]\ntype = \"outflow\"\n";
    else bc << "[patch.xlo]\ntype = \"far\"\ny = [1.0]\n[patch.xhi]\ntype = \"outflow\"\nburnt = true\n";
}

// Every record of a's checkpoint byte-identical to b's, the restart count aside:
// both rewritten with a's count, the files compared.
bool sameCheckpoint(const std::string& a, const std::string& b, const std::string& name)
{
    auto ca = dns::Checkpoint::read(a), cb = dns::Checkpoint::read(b);
    if (!ca || !cb) {
        std::printf("    %s: cannot read %s\n", name.c_str(), (!ca ? a : b).c_str());
        return false;
    }
    const auto rs = ca->i64Scalar("restarts");
    if (!rs) return false;
    cb->putI64("restarts", *rs);
    const std::string ta = a + ".cmp", tb = b + ".cmp";
    if (!ca->write(ta) || !cb->write(tb)) return false;
    const bool same = slurp(ta) == slurp(tb) && !slurp(ta).empty();
    std::printf("    %s: last checkpoint (%zu state records) %s\n", name.c_str(), ca->names("state.").size(),
                same ? "identical" : "DIFFERS");
    fs::remove(ta);
    fs::remove(tb);
    return same;
}

// The straight run vs the chain (restart_test_util's chained), then a straight
// run writing checkpoints every `every`: its outputs equal the straight run's,
// its last checkpoint the chain's.
template <class Between>
void fvCase(const std::string& name, const std::string& args, double every, int wantJobs, Between between)
{
    if (chained(kRoot, name, args, every, wantJobs, between) == 0) return;
    const std::string d = kRoot + "/" + name;
    const std::string ck = d + "/ck";
    fs::create_directories(ck);
    const int rc = runDns(args + std::format(" --checkpoint-dt {} --quiet --out {}/r", every, ck), ck + "/run.log");
    check(rc == 0, name + ": the checkpointing straight run exits 0");
    check(rc == 0 && sameOutputs(d + "/straight", ck, name + " (checkpointing)"),
          name + ": writing checkpoints leaves every output byte-identical");
    check(rc == 0 && sameCheckpoint(ck + "/r_ckpt.bin", d + "/chain/r_ckpt.bin", name),
          name + ": the chain's last checkpoint equals the straight run's (state, loop, outputs)");
}

void fvCase(const std::string& name, const std::string& args, double every, int wantJobs)
{
    fvCase(name, args, every, wantJobs, [](int) {});
}

// Teeth: the case's first checkpoint with F32 record `rec` (n = grid[gi] *
// mult words) ZEROED, then resumed to the end: the outputs must DIFFER from the
// straight run's (the record is carried state, and the byte comparison sees it).
void teeth(const std::string& name, const std::string& args, double every, const std::string& rec, int gi, int mult)
{
    const std::string d = kRoot + "/" + name + "_teeth_" + rec;
    fs::create_directories(d);
    const std::string out = " --quiet --out " + d + "/r";
    const std::string ck = std::format(" --checkpoint-dt {}", every);
    int rc = runDns(args + ck + " --wall-limit 0" + out, d + "/job0.log");
    auto c = dns::Checkpoint::read(d + "/r_ckpt.bin");
    std::vector<std::int64_t> grid;
    if (c)
        if (auto g = c->i64("grid")) grid = *g;
    const bool ok = rc == 0 && c && grid.size() == 4;
    check(ok, name + " teeth: first job and its checkpoint");
    if (!ok) return;
    const std::vector<float> zero(std::size_t(grid[std::size_t(gi)] * mult), 0.0f);
    std::vector<float> was(zero.size());
    const bool had = c->f32Into(rec, was.data(), was.size()).has_value();
    c->putF32(rec, zero);
    const bool wrote = c->write(d + "/r_ckpt.bin").has_value();
    rc = runDns(args + ck + " --restart " + d + "/r" + out, d + "/job1.log");
    const bool nonzero = std::ranges::any_of(was, [](float v) { return v != 0.0f; });
    std::printf("    %s teeth: %s %s\n", name.c_str(), rec.c_str(), nonzero ? "was nonzero" : "was ALL ZERO");
    check(had && wrote && rc == 0 && nonzero && !sameOutputs(kRoot + "/" + name + "/straight", d, name + " teeth"),
          name + ": zeroing " + rec + " in the checkpoint changes the outputs (carried state, seen by the byte gate)");
}

// teeth on an F64 record (the FP64 words): the whole record zeroed, resumed to the end; the
// final checkpoint's state.fv.q must differ from the straight run's.
void teeth64(const std::string& name, const std::string& args, double every, const std::string& rec)
{
    const std::string d = kRoot + "/" + name + "_teeth_" + rec;
    fs::create_directories(d);
    const std::string out = " --quiet --out " + d + "/r";
    const std::string ck = std::format(" --checkpoint-dt {}", every);
    int rc = runDns(args + ck + " --wall-limit 0" + out, d + "/job0.log");
    auto c = dns::Checkpoint::read(d + "/r_ckpt.bin");
    const bool ok = rc == 0 && c.has_value();
    check(ok, name + " teeth: first job and its checkpoint");
    if (!ok) return;
    const auto was = c->f64(rec);
    const bool nonzero = was && std::ranges::any_of(*was, [](double v) { return v != 0.0; });
    if (was) c->putF64(rec, std::vector<double>(was->size(), 0.0));
    const bool wrote = c->write(d + "/r_ckpt.bin").has_value();
    rc = runDns(args + ck + " --restart " + d + "/r" + out, d + "/job1.log");
    std::printf("    %s teeth: %s %s\n", name.c_str(), rec.c_str(), nonzero ? "was nonzero" : "was ALL ZERO");
    // An FP64 low word is <= ulp(hi)/2: the printed outputs (stats, profiles, FP32 frames) do not
    // resolve it over a short run. The carried state does: the final checkpoint's q words.
    auto mine = dns::Checkpoint::read(d + "/r_ckpt.bin"), ref = dns::Checkpoint::read(kRoot + "/" + name + "/ck/r_ckpt.bin");
    std::vector<double> qa, qb;
    if (mine)
        if (auto v = mine->f64("state.fv.q")) qa = std::move(*v);
    if (ref)
        if (auto v = ref->f64("state.fv.q")) qb = std::move(*v);
    std::size_t moved = 0;
    if (!qa.empty() && qa.size() == qb.size())
        for (std::size_t i = 0; i < qa.size(); ++i) moved += qa[i] != qb[i];
    std::printf("    %s teeth: %zu of %zu final state.fv.q words moved\n", name.c_str(), moved, qa.size());
    check(was.has_value() && wrote && rc == 0 && nonzero && moved > 0,
          name + ": zeroing " + rec + " in the checkpoint moves the final state words (carried state)");
}

// Rejections on the one-job checkpoint of `args` (prefix d/r): each a
// (modification, restart args, exit code, text) row; the outputs stay as they were.
struct Reject
{
    std::string what, args;
    int rc;
    std::string text;
};

template <class Mutate>
void rejects(const std::string& d, const std::string& args, double every, const std::vector<Reject>& rows,
             Mutate mutate)
{
    fs::create_directories(d);
    const std::string ck = std::format(" --checkpoint-dt {} --wall-limit 0", every);
    const std::string out = " --quiet --out " + d + "/r";
    const int rc0 = runDns(args + ck + out, d + "/job0.log");
    check(rc0 == 0, d + ": first job exits 0");
    const auto before = artefacts(d);
    std::string snap;
    for (const auto& f : before) snap += slurp(d + "/" + f);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const Reject& r = rows[i];
        mutate(int(i), true);
        const std::string log = d + std::format("/r{}.log", i);
        const int rc = runDns(r.args + ck + " --restart " + d + "/r" + out, log);
        mutate(int(i), false);
        const bool named = slurp(log).find(r.text) != std::string::npos;
        if (rc != r.rc || !named) std::printf("    rc %d, log: %s", rc, slurp(log).c_str());
        check(rc == r.rc && named, std::format("reject: {} is exit {} naming \"{}\"", r.what, r.rc, r.text));
    }
    std::string after;
    for (const auto& f : artefacts(d)) after += slurp(d + "/" + f);
    check(artefacts(d) == before && after == snap, "reject: no rejected restart touched the outputs");
}

// on: appends a `comment` line to a text file (its content changed, its meaning
// not); off: restores it.
void touch(const std::string& path, bool on, std::string& keep, const char* comment)
{
    if (on) {
        keep = slurp(path);
        std::ofstream(path, std::ios::app) << comment << " changed\n";
    } else {
        std::ofstream(path, std::ios::trunc) << keep;
    }
}

// masterplan_combclose R6: the compensated FP64 state (F64 q and lo words); its checkpoint
// carries state.fv.lo, zeroed it changes the outputs; a restart without --compensated is
// exit 3 by name. Also alone: `test_gpu_restart_fv fp64c`.
void compFp64()
{
    const std::string fp64c = "flame --solver fv --fv-precision fp64 --compensated --cells-per-delta 8 --tt 1 --profile-dt 0.5 "
                              "--emit stats --emit bin --fdt 0.25";
    fvCase("fp64c", fp64c, 0.25, 4);
    teeth64("fp64c", fp64c, 0.25, "state.fv.lo");
    rejects(kRoot + "/reject_fp64c", fp64c, 0.25,
            {{"a compensated FP64 checkpoint into a plain FP64 run",
              "flame --solver fv --fv-precision fp64 --cells-per-delta 8 --tt 1 --profile-dt 0.5 --emit stats --emit bin --fdt 0.25", 3,
              "--compensated"}},
            [](int, bool) {});
}

void comp()
{
    fvCase("les3d",
           "flame --solver fv --nz 8 --ny 8 --cells-per-delta 4 --les 1.35 --sgs sigma --tfles 2 --tt 0.2 "
           "--emit stats --emit bin --fdt 0.05",
           0.05, 4);
    const std::string c2b =
        "flame --solver fv --equivalence 1.3 --stoich 3.99 --ox-y 0.233 --expansion 6.88 --visc sutherland "
        "--tinf 298 --cells-per-delta 4 --compensated --tt 1 --profile-dt 0.5 --emit stats --emit bin --fdt 0.25";
    fvCase("c2b", c2b, 0.25, 4);
    teeth("c2b", c2b, 0.25, "state.fv.lo", 0, 7); // the low words of rows rho .. E, Y_F, Y_O
    fvCase("fp64",
           "flame --solver fv --fv-precision fp64 --cells-per-delta 8 --tt 1 --profile-dt 0.5 --emit stats "
           "--emit bin --fdt 0.25",
           0.25, 4, [](int job) {
               if (job == 2) appendJunk(kRoot + "/fp64/chain");
           });
    compFp64();
    fvCase("jet", "flame --mode diffusion --solver fv --tt 1 --emit stats --emit bin --fdt 0.25", 0.25, 4);
    fvCase("turb",
           "flame --solver fv --turb 4 --turb-length 1 --cells-per-delta 4 --ny 24 --nz 24 --upstream 6 "
           "--downstream 4 --mach 0.02 --tt 0.2 --emit stats --emit bin --fdt 0.05",
           0.05, 4);
    fvCase("candle", "flame --mode candle --solver fv --tt 1 --emit stats --emit bin --fdt 0.25", 0.25, 4);
    const std::string mesh = kRoot + "/mesh_comp";
    writeFlameMesh(mesh, false);
    const std::string poly = "flame --polymesh " + mesh + " --bc " + mesh + "/bc.toml --upstream 12 --tt 1 "
                             "--profile-dt 0.5 --emit stats --emit bin --fdt 0.25";
    fvCase("poly", poly, 0.25, 4);

    // reject: on a one-job polyMesh checkpoint.
    const std::string points = mesh + "/constant/polyMesh/points", bcFile = mesh + "/bc.toml";
    std::string keep;
    const std::string box = "flame --cells-per-delta 8 --tt 1 --emit stats --emit bin --fdt 0.25";
    rejects(kRoot + "/reject", poly, 0.25,
            {{"a changed polyMesh file", poly, 3, "the mesh differs from the checkpoint's"},
             {"a changed --bc file", poly, 3, "the mesh differs from the checkpoint's"},
             {"an FV checkpoint into the structured flame", box, 3, "is a 'flame-fv-compressible' checkpoint"}},
            [&](int i, bool on) {
                if (i == 0) touch(points, on, keep, "//");
                if (i == 1) touch(bcFile, on, keep, "#");
            });
    // ... and a structured checkpoint into the FV flame.
    rejects(kRoot + "/reject_box", box, 0.25,
            {{"a structured checkpoint into the FV flame", "flame --solver fv --cells-per-delta 8 --tt 1 --emit "
                                                           "stats --emit bin --fdt 0.25",
              3, "is a 'flame-compressible' checkpoint; this run is flame-fv-compressible"}},
            [](int, bool) {});
}

void lowMach()
{
    fvCase("lm_burner",
           "flame --physics lowmach --solver fv --mode bunsen --laterals open --rim 4 --gravity 1 "
           "--cells-per-delta 4 --tt 0.5 --emit stats --emit bin --fdt 0.125",
           0.125, 4);
    fvCase("lm_les3d",
           "flame --physics lowmach --solver fv --nz 8 --ny 8 --cells-per-delta 6 --les 1.35 --sgs sigma --tt 0.3 "
           "--emit stats --emit bin --fdt 0.1",
           0.075, 4);
    const std::string tfles = "flame --physics lowmach --solver fv --tfles 2 --cells-per-delta 6 --tt 1 --profile-dt 0.5 "
                              "--emit stats --emit bin --fdt 0.25";
    fvCase("lm_tfles", tfles, 0.25, 4);
    // The projection's carried words: the PCG warm start p (nC), the face mass flux phi (nI).
    teeth("lm_tfles", tfles, 0.25, "state.fvlm.p", 0, 1);
    teeth("lm_tfles", tfles, 0.25, "state.fvlm.phi", 1, 1);
    const std::string mesh = kRoot + "/mesh_lm";
    writeFlameMesh(mesh, true);
    const std::string poly = "flame --physics lowmach --polymesh " + mesh + " --bc " + mesh +
                             "/bc.toml --upstream 12 --tt 1 --fv-precision fp64 --profile-dt 0.5 --emit stats "
                             "--emit bin --fdt 0.25";
    fvCase("lm_poly", poly, 0.25, 4);
    std::string keep;
    rejects(kRoot + "/lm_reject", poly, 0.25,
            {{"a changed polyMesh file", poly, 3, "the mesh differs from the checkpoint's"},
             {"an FV low-Mach checkpoint into the structured low-Mach flame",
              "flame --physics lowmach --cells-per-delta 8 --tt 1 --emit stats", 3,
              "is a 'flame-fv-lowmach' checkpoint; this run is flame-lowmach"}},
            [&](int i, bool on) {
                if (i == 0) touch(mesh + "/constant/polyMesh/owner", on, keep, "//");
            });
}

// combscale S1, host: an array record and a bool through saveLoop/loadLoop, bit for bit; a
// mis-sized array record is an error by name.
struct ArrayLoop
{
    std::array<double, 6> a{};
    bool b = false;
    double x = 0;
    template <class F>
    void fields(F&& f)
    {
        f("a", a);
        f("b", b);
        f("x", x);
    }
};

void hostLoopRoundTrip()
{
    ArrayLoop s{{0.1, -2.5e-300, 1.0 / 3.0, std::nextafter(1.0, 2.0), -0.0, 6.02e23}, true, std::numbers::pi};
    dns::Checkpoint ck;
    dns::gpu::internal::saveLoop(ck, s);
    ArrayLoop r;
    const auto e = dns::gpu::internal::loadLoop(ck, r);
    bool same = e.has_value() && r.b && std::memcmp(&r.x, &s.x, sizeof s.x) == 0;
    for (std::size_t i = 0; i < 6; ++i) same = same && std::memcmp(&r.a[i], &s.a[i], sizeof(double)) == 0;
    check(same, "host: std::array<double, 6> + bool + double round-trip saveLoop/loadLoop bit for bit");
    const auto a = ck.f64("loop.a");
    check(a && a->size() == 6, "host: loop.a is one F64 record of 6 values");
    ck.putF64("loop.a", std::vector<double>(5, 0.0));
    const auto bad = dns::gpu::internal::loadLoop(ck, r);
    check(!bad && bad.error() == "checkpoint 'loop.a' holds 5 values, the run needs 6",
          "host: a mis-sized array record is an error by name");
    if (!bad.has_value()) std::printf("    %s\n", bad.error().c_str());
}

// teeth on any F64 record (a device word or a loop value): zeroed in the first checkpoint, resumed to
// the end, _thrust.csv must differ from the straight run's.
void teethThrust(const std::string& name, const std::string& args, double every, const std::string& rec)
{
    const std::string d = kRoot + "/" + name + "_teeth_" + rec;
    fs::create_directories(d);
    const std::string out = " --quiet --out " + d + "/r";
    const std::string ck = std::format(" --checkpoint-dt {}", every);
    int rc = runDns(args + ck + " --wall-limit 0" + out, d + "/job0.log");
    auto c = dns::Checkpoint::read(d + "/r_ckpt.bin");
    const bool ok = rc == 0 && c.has_value();
    check(ok, name + " teeth " + rec + ": first job and its checkpoint");
    if (!ok) return;
    const auto was = c->f64(rec);
    const bool nonzero = was && std::ranges::any_of(*was, [](double v) { return v != 0.0; });
    if (was) c->putF64(rec, std::vector<double>(was->size(), 0.0));
    const bool wrote = c->write(d + "/r_ckpt.bin").has_value();
    rc = runDns(args + ck + " --restart " + d + "/r" + out, d + "/job1.log");
    const std::string mine = slurp(d + "/r_thrust.csv"), ref = slurp(kRoot + "/" + name + "/straight/r_thrust.csv");
    std::printf("    %s teeth: %s %s (%zu values); _thrust.csv %s\n", name.c_str(), rec.c_str(),
                nonzero ? "was nonzero" : "was ALL ZERO", was ? was->size() : 0, mine == ref ? "UNCHANGED" : "changed");
    check(was.has_value() && wrote && rc == 0 && nonzero && !mine.empty() && !ref.empty() && mine != ref,
          name + ": zeroing " + rec + " in the checkpoint changes _thrust.csv (carried state)");
}

void thrust()
{
    const std::string D = kRoot + "/react15";
    const int g = std::system(std::format("python3 {}/tools/fv/nozzle_react.py design --cells 15 --mach 0.042 --sl 0.9945059 "
                                          "--out {} > {}/design.log 2>&1",
                                          DNS_SOURCE_DIR, D, kRoot)
                                  .c_str());
    check(g == 0, "thrust: nozzle_react.py design --cells 15");
    if (g != 0) return;
    const std::string t64 = "flame --mach 0.042 --expansion 5 --zeldovich 8 --prandtl 0.72 --polymesh " + D + " --bc " + D +
                            "/bc.toml --shock hll --fv-precision fp64 --upstream 12 --tt 1 --emit stats";
    const std::string t64c = t64 + " --compensated";
    fvCase("thrust64", t64, 0.25, 4);
    fvCase("thrust64c", t64c, 0.25, 4);
    check(!slurp(kRoot + "/thrust64/straight/r_thrust.csv").empty(), "thrust64: _thrust.csv written (compared above)");
    for (const char* rec : {"state.fv.ledger.I", "loop.thrust.P0", "loop.thrust.prevExit"}) teethThrust("thrust64c", t64c, 0.25, rec);

    const std::string bcFile = D + "/bc.toml";
    const std::string orig = slurp(bcFile);
    auto edited = [&](std::string_view from, std::string_view to) {
        std::string s = orig;
        const auto at = s.find(from);
        check(at != std::string::npos, std::format("thrust reject: bc.toml holds '{}'", from));
        if (at != std::string::npos) s.replace(at, from.size(), to);
        return s;
    };
    const std::string thrustBlock = "[thrust]\npa = 0.01\nstation = 46.768209200000001\n";
    // combscale S5: [nozzle] cd joins the section line (a cd added is named by [nozzle]).
    const std::string variants[] = {edited("pa = 0.01", "pa = 0.02"), edited(thrustBlock, ""),
                                    edited("astar = 4.9971511200000007", "astar = 4.9971511200000016"),
                                    orig + "# a comment\n",
                                    edited("astar = 4.9971511200000007\n", "astar = 4.9971511200000007\ncd = 1\n")};
    rejects(kRoot + "/reject_thrust", t64, 0.25,
            {{"[thrust] pa 0.01 -> 0.02", t64, 3,
              "the --bc file's [thrust] differs from the checkpoint's: pa 0.01 station 46.7682092 vs pa 0.02 station 46.7682092"},
             {"[thrust] removed", t64, 3, "the --bc file's [thrust] differs from the checkpoint's: pa 0.01 station 46.7682092 vs (absent)"},
             {"[nozzle] astar one ulp off (its 17th digit)", t64, 3, "the --bc file's [nozzle] differs from the checkpoint's: astar 4.997151120000001"},
             {"a comment line appended to the --bc file", t64, 3, "the mesh differs from the checkpoint's"},
             {"[nozzle] cd = 1 added", t64, 3,
              "the --bc file's [nozzle] differs from the checkpoint's: astar 4.997151120000001 axis 1 0 0 vs astar 4.997151120000001 axis 1 0 0 cd 1"}},
            [&](int i, bool on) { std::ofstream(bcFile, std::ios::trunc) << (on ? variants[i] : orig); });
    rejects(kRoot + "/reject_thrust_c", t64c, 0.25,
            {{"a compensated FP64 [thrust] checkpoint into a plain FP64 run", t64, 3, "--compensated"}}, [](int, bool) {});
}

} // namespace

int main(int argc, char** argv)
{
    const std::string_view group = argc > 1 ? argv[1] : "";
    if (group != "comp" && group != "lm" && group != "fp64c" && group != "thrust") {
        std::fprintf(stderr, "usage: test_gpu_restart_fv comp|lm|fp64c|thrust\n");
        return 1;
    }
    if (group == "thrust") hostLoopRoundTrip();
    dns::test::requireDeviceOrSkip();
    std::error_code ec;
    kRoot = "/tmp/dns_test_restart_fv/" + std::string(group);
    fs::remove_all(kRoot, ec);
    fs::create_directories(kRoot);
    if (group == "comp") comp();
    else if (group == "fp64c") compFp64();
    else if (group == "thrust") thrust();
    else lowMach();
    return dnstest::finish();
}
