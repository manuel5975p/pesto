// The case drivers: a validated Case onto the library's meshes, initial
// conditions, runners and output sink, one per subcommand (phases 1.5, 3 and 4
// of `dns`; the executable is only argv, parse, validate and these calls).
#pragma once

#include "dns/config.hpp"

#include <chrono>
#include <expected>
#include <optional>
#include <string>

namespace dns {
struct Scene;
}

namespace dns::drive {

// Checkpoint / restart facts gathered in phase 1.5 for the flame runs
// (dns/checkpoint.hpp): the binary's sha256 (only when a checkpoint flag is
// given), the job's start (--wall-limit counts from it), the restart count, and
// the FV routes' mesh identity.
struct CheckpointContext
{
    std::string binary;
    std::chrono::steady_clock::time_point start;
    long restarts = 0;
    std::string mesh;
};

// Phase 1.5: the INPUT files whose facts the validator reads -- --scene (its
// model block into raw), the compressible / low-Mach --bc file, the --restart
// checkpoint header against this invocation. Error: the exit code, message printed.
std::expected<void, int> readInputs(RawConfig& raw, std::optional<Scene>& scene, CheckpointContext& ckx);

// --dry-run after validate(): the resolved case, plus each route's input-file
// manifest (mesh, --bc table, OpenFOAM case). Returns the exit code.
int dryRun(const Case& c, const Scene* scene);

// Phases 3 and 4 of a validated, non-dry-run case. Returns the exit code.
int run(const Case& c, const RawConfig& raw, const Scene* scene, const CheckpointContext& ckx);

} // namespace dns::drive
