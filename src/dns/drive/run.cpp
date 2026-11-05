// The drivers' entry points (drive.hpp): phase 1.5 input files, --dry-run
// manifests, and the subcommand dispatch of phases 3 and 4.
#include "dns/drive/drive.hpp"
#include "dns/drive/internal.hpp"

#include "dns/fvcase.hpp"
#include "dns/fvcomp_config.hpp"
#include "dns/fvlowmach.hpp"
#include "dns/fvsimple.hpp"
#include "dns/fvstructured.hpp"
#include "dns/scene.hpp"
#include "jobs/sha256.hpp"

#include <format>
#include <fstream>
#include <iterator>

namespace dns::drive {

namespace {

// --bc for the compressible and low-Mach FV routes: an input file whose
// reference length validate() reads, so it is parsed before validate (exit 3).
// --physics picks the parser: the incompressible file is read by that path's
// Acquire, against the mesh's patches.
std::expected<void, int> readFvBcFile(RawConfig& raw)
{
    const bool fvRoute = raw.sub == Subcommand::Flow || raw.sub == Subcommand::Flame || raw.sub == Subcommand::Conv;
    const bool parsesHere = raw.physics == Physics::Compressible || raw.sub == Subcommand::Flame;
    if (!fvRoute || !parsesHere || !raw.given.contains("--polymesh") || !raw.given.contains("--bc")) return {};
    std::ifstream f(raw.bc);
    if (!f) return std::unexpected(die(exitInput, "--bc: cannot read " + raw.bc));
    const std::string text{std::istreambuf_iterator<char>(f), {}};
    auto b = parseFvBc(text);
    if (!b) return std::unexpected(die(exitInput, "--bc " + raw.bc + ": " + b.error()));
    if (raw.physics == Physics::LowMach)
        if (auto l = fvLowMachBcs(*b, 1.0, 1.0); !l) return std::unexpected(die(exitInput, raw.bc + ": " + l.error()));
    raw.fvBc = std::move(*b);
    return {};
}

// dns flame checkpoints: the binary's fingerprint, the FV routes' mesh identity,
// and --restart's checkpoint HEADER (no fields) against this invocation --
// solver, binary, every case-defining flag, mesh: a mismatch is exit 3 by name.
// Its t feeds validate()'s --tt rule.
std::expected<void, int> readFlameCheckpoint(RawConfig& raw, CheckpointContext& ckx)
{
    if (raw.sub != Subcommand::Flame) return {};
    if (raw.given.contains("--checkpoint-dt") || raw.given.contains("--restart")) {
        auto h = jobs::sha256File("/proc/self/exe");
        if (!h) return std::unexpected(die(exitInput, "checkpoint: cannot fingerprint the dns binary: " + h.error()));
        ckx.binary = *h;
        // (--polymesh without --bc: validate's exit 2 names it.)
        if (fvFlameRoute(raw) && (!raw.given.contains("--polymesh") || raw.given.contains("--bc"))) {
            auto m = fvMeshIdentity(raw);
            if (!m) return std::unexpected(die(exitInput, "checkpoint: cannot fingerprint the mesh: " + m.error()));
            ckx.mesh = *m;
        }
    }
    if (!raw.given.contains("--restart")) return {};
    const std::string path = checkpointPath(raw.restart);
    auto ck = Checkpoint::read(path, true);
    if (!ck) return std::unexpected(die(exitInput, "--restart: " + ck.error()));
    auto f = checkpointFacts(*ck);
    if (!f) return std::unexpected(die(exitInput, "--restart: " + path + ": " + f.error()));
    const char* solver = flameSolverName(raw);
    if (f->solver != solver)
        return std::unexpected(
            die(exitInput, std::format("--restart: {} is a '{}' checkpoint; this run is {}", path, f->solver, solver)));
    if (f->binary != ckx.binary)
        return std::unexpected(die(exitInput, std::format("--restart: {} was written by a different dns binary (sha256 "
                                                          "{} vs this {}): a restart reproduces a run only on the "
                                                          "binary that started it",
                                                          path, f->binary, ckx.binary)));
    if (auto d = caseFlagsDiff(f->caseFlags, raw.caseFlags))
        return std::unexpected(die(exitInput, "--restart: case differs from the checkpoint: " + *d));
    if (f->mesh != ckx.mesh)
        return std::unexpected(die(exitInput, std::format("--restart: {}: the mesh differs from the checkpoint's: {} vs {}",
                                                          path, f->mesh.empty() ? "(none)" : f->mesh,
                                                          ckx.mesh.empty() ? "(none)" : ckx.mesh)));
    raw.restartT = f->t;
    ckx.restarts = f->restarts + 1;
    return {};
}

} // namespace

std::expected<void, int> readInputs(RawConfig& raw, std::optional<Scene>& scene, CheckpointContext& ckx)
{
    // --scene: its model block feeds the validator. A bad scene file is an
    // input error (3); a model conflict is a config error (2).
    if (!raw.scene.empty()) {
        auto s = loadScene(raw.scene);
        if (!s) return std::unexpected(die(exitInput, s.error()));
        if (auto m = applySceneModel(*s, raw); !m) {
            std::fputs(format(m.error()).c_str(), stderr);
            return std::unexpected(m.error().code);
        }
        scene = std::move(*s);
    }
    if (auto e = readFvBcFile(raw); !e) return e;
    return readFlameCheckpoint(raw, ckx);
}

int dryRun(const Case& c, const Scene* scene)
{
    if (c.fvs.active) {
        auto fc = fvStructuredResolved(c.fvs);
        if (!fc) return die(exitInput, fc.error());
        fc->dir = c.fvCase;
        std::printf("--solver simple: %s\n%s", fvStructuredLabel(c.fvs).c_str(), describeFoamCase(*fc).c_str());
        return 0;
    }
    std::fputs(c.describe().c_str(), stdout);
    if (scene) std::fputs(scene->describe().c_str(), stdout);
    // `dns mesh`: the manifest names the input file's facts, read without bulk data.
    if (c.sub == Subcommand::Mesh) return runMeshManifest(c);
    if ((c.sub == Subcommand::Flow || c.sub == Subcommand::Flame) && c.fvMode == FvCompMode::PolyMesh) {
        auto two = fvPolyCheck(c);
        if (!two) return die(two.error().first, two.error().second);
        std::printf("  label      %s\n", (c.sub == Subcommand::Flame ? fvFlameLabel(c) : fvCompLabel(c, *two)).c_str());
        return 0;
    }
    if (c.fvConv && !c.fvMeshDir.empty()) return runConvFvManifest(c);
    if (c.fvIncomp && !c.fvMeshDir.empty()) return runFvIncompManifest(c);
    if (c.sub == Subcommand::Flow && !c.fvCase.empty()) {
        auto fc = loadFoamCaseManifest(c.fvCase);
        if (!fc) return die(exitInput, fc.error());
        std::fputs(describeFoamCase(*fc).c_str(), stdout);
    }
    return 0;
}

int run(const Case& c, const RawConfig& raw, const Scene* scene, const CheckpointContext& ckx)
{
    switch (c.sub) {
    case Subcommand::Flow:
        if (c.fvMode == FvCompMode::PolyMesh) return runFvCompPoly(c);
        if (c.fvIncomp) return c.fvMeshDir.empty() ? runFvIncompOGrid(c, scene) : runFvIncompPoly(c);
        if (c.fvs.active) return runFvStructured(c);
        if (!c.fvCase.empty()) return runFvFlow(c);
        return runFlow(c, raw, scene);
    case Subcommand::Tgv: return runTgv(c);
    case Subcommand::Hit: return runHit(c);
    case Subcommand::Bench: return runBench(c);
    case Subcommand::Channel: return runChannel(c);
    case Subcommand::Conv: return runConv(c);
    case Subcommand::Prop: return runProp(c);
    case Subcommand::Flame: return runFlame(c, ckx);
    case Subcommand::Mesh: return runMesh(c);
    }
    return 1;
}

} // namespace dns::drive
