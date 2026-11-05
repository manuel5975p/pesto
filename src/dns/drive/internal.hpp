// The drivers' shared internals (src/dns/drive/*.cpp only): exit codes, the
// message-and-code return, the sink config, the GPU probe, and every driver one
// file calls in another. Not a public interface: drive.hpp is.
#pragma once

#include "dns/checkpoint.hpp"
#include "dns/config.hpp"
#include "dns/drive/drive.hpp"
#include "dns/fvincomp.hpp"
#include "dns/output.hpp"

#include <cstdint>
#include <cstdio>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace dns {
struct CaseGeometry;
struct CompState;
struct CurvMesh;
struct FaceMesh;
struct VectorField;
} // namespace dns

namespace dns::drive {

// Exit codes: 0 ok, 1 CLI, 2 config, 3 input, 4 output, 5 no GPU, 6 resource,
// 7 numerical failure, 8 solver non-convergence.
constexpr int exitInput = 3, exitOutput = 4, exitNoGpu = 5, exitResource = 6, exitNumerical = 7;

// Box units of both compressible body runs (Cartesian and O-grid): rho = u =
// R = 1 and p from the Mach number.
constexpr double kRhoInf = 1.0;

// Print "dns: msg" to stderr and return code: every driver's error tail.
int die(int code, const std::string& msg);

// The end of every run: the sink's checked close (a write failure anywhere in
// the run is exit 4 here, never a silent 0), then the line every run ends with.
// Returns the process exit code, so a runner's tail is `return reportDone(...)`.
template <class Stats>
int reportDone(const Stats& st, RunOutput& out)
{
    if (auto e = out.finish(); !e) return die(exitOutput, e.error());
    std::printf("done: %ld steps, %d frames, %.2f ms/step\n", st.steps, out.frames(), st.msPerStep);
    return 0;
}

// Case -> sink config: the cadences and PNG coarsening of every run; only the
// `sim` metadata differs. suppress: no artefacts (a sequenced coarse level).
RunOutput::Config outputConfig(const Case& c, const std::string& simJson, bool suppress = false);

// A usable device for this case (false on --backend cpu or a build without GPU).
bool haveGpu(const Case& c);

// A frame mask over n cells: 0 where map[q] == solid, else 1 (map shorter than n, empty
// included: the rest fluid).
std::vector<float> frameMask(std::size_t n, std::span<const std::uint8_t> map, std::uint8_t solid);

// Trilinear prolongation of a MAC velocity field onto a finer grid, by world
// position. Both grids carry one ghost cell, so the interior origin is at
// x = h and coordinates are measured from there: a u-face at index i sits at
// X = (i-1)*h, a cell centre at X = (j-0.5)*h.
//
// Deliberately NOT conservative and NOT divergence-free: this only has to be
// close. The level that receives it masks, projects and re-converges, so the
// fixed point it reaches is that level's own -- prolongation accuracy buys
// transient length, never correctness.
int runFlowIncompressible(const Case& c, CaseGeometry& geo, const VectorField* ic0 = nullptr, VectorField* finalOut = nullptr, bool suppressOutput = false);

// The body IS the mesh's inner boundary, so this shares nothing with the
// immersed-body path but the case: no cut geometry, no ghost wall, no far-field
// box. The mesh is built from the SAME sdf the immersed path would have used,
// so a body is described once.
int runFlowOGrid(const Case& c, CaseGeometry& geo);

int runFlowCompressible(const Case& c, CaseGeometry& geo);

// Grid sequencing: converge on n/2^(L-1), prolong, repeat, finish on n. Each
// level is a normally validated Case -- the ladder reuses validate() rather
// than hand-editing a Case, so no level can skip a check. Only the finest
// level writes output and only its result is reported. `fine` is the finest
// level's geometry, already built by runFlow; only the coarse ones are built here.
int runFlowSequenced(const Case& c, const RawConfig& raw, const Scene* scene, CaseGeometry& fine);

// dns flow --polymesh CASE (docs/fv.md): Acquire = the OpenFOAM case (exit 3
// before any device), then the GPU SIMPLE run (gpu/runner_fv).
int runFvFlow(const Case& c);

// DIR, or the polyMesh of the case DIR.
std::filesystem::path fvPolyDir(const std::string& dir);

// Phase 1.5 of --polymesh + --bc: the manifest bound against the --bc file, and
// the 2D facts config could not see. Errors as an exit code + message.
std::expected<bool, std::pair<int, std::string>> fvPolyCheck(const Case& c);

// dns flow --physics compressible --polymesh DIR --bc FILE.
int runFvCompPoly(const Case& c);

std::expected<FvIncompBcs, std::string> readFvIncompBcs(const Case& c);

// Phase 1.5 (--dry-run): the manifest and the --bc table, no bulk mesh data.
int runFvIncompManifest(const Case& c);

// flow --polymesh DIR --bc FILE: Acquire = mesh and table (exit 3 before any device).
int runFvIncompPoly(Case c);

// flow --solver fv --ogrid: the structured O-grid through fromCurvMesh.
int runFvIncompOGrid(const Case& c, const Scene* scene);

// tgv --solver fv: the periodic box as a FaceMesh.
int runFvIncompTgv(const Case& c);

// dns flow --solver simple (docs/fvstructured.md): Acquire = the mesh and the
// generated case on disk, then the --polymesh path reads that directory. With
// --export the directory is the whole output: no device is ever touched.
int runFvStructured(const Case& c);

int runFlow(const Case& c, const RawConfig& raw, const Scene* scene);

int runChannel(const Case& c);

// Phase 1.5 of dns conv --polymesh: the manifest against the --bc table.
int runConvFvManifest(const Case& c);

int runConv(const Case& c);

// The checkpoint policy of a flame run (both solvers), and its --restart
// checkpoint read whole (phase 3: the run's INPUT, exit 3) with its grid
// and species count against the case.
std::expected<std::optional<Checkpoint>, int> readFlameRestart(const Case& c, int nY);

// The FV routes' --restart checkpoint read whole (phase 3, exit 3): its FaceMesh
// counts (cells, internal faces, boundary faces) and species against the mesh
// built. Phase 1.5 already matched the mesh identity; this is the second gate.
std::expected<std::optional<Checkpoint>, int> readFvRestart(const Case& c, const FaceMesh& fm, int nY);

CheckpointPolicy flamePolicy(const Case& c, const CheckpointContext& ckx, const std::optional<Checkpoint>& restart);

// The flame routes taking --solver fv or --polymesh (runFvFlame, runFvLowMachCase).
bool fvFlameRoute(const RawConfig& r);

// The checkpoint "solver" record a flame invocation reads and writes.
const char* flameSolverName(const RawConfig& r);

// An FV flame's mesh identity (CheckpointPolicy::mesh): "flags" for the box the
// case flags define; on --polymesh the sha256 over every regular file of the
// polyMesh directory (name, its sha256; ascending names) and the --bc file's.
std::expected<std::string, std::string> fvMeshIdentity(const RawConfig& r);

int runFvFlame(const Case& c, const CheckpointContext& ckx);

// dns flame --physics lowmach --solver fv | --polymesh (docs/fvincomp.md "Low-Mach"):
// the planar flame on the FV low-Mach solver. Box units Tu = P0 = R = rho_u = 1. The
// box lays gpu/lowmach's IC cell for cell; a polyMesh the oracle along reference.direction.

int runFvLowMachCase(const Case& c, const CheckpointContext& ckx);

int runFlame(const Case& c, const CheckpointContext& ckx);

int runTgv(const Case& c);

int runHit(const Case& c);

int runBench(const Case& c);

// Host-only: Acquire = read the two input CSVs (exit 3) and create the output
// CSV (exit 4), then Run. No GPU is ever probed.
int runProp(const Case& cs);

// --dry-run: the manifest only (owner note, boundary, zone names); no bulk
// data. A mesh that cannot be read is an input error (3).
int runMeshManifest(const Case& cs);

// Host-only: Acquire = read the polyMesh and build the FaceMesh (exit 3 on
// any rejection), Run = the quality report. Never probes a GPU.
int runMesh(const Case& cs);

} // namespace dns::drive
