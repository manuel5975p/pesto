// dns flame checkpoint/restart facts shared by every flame route (phase 1.5 and Acquire).
#include "dns/drive/internal.hpp"

#include "dns/checkpoint.hpp"
#include "dns/config.hpp"
#include "dns/facemesh.hpp"
#include "dns/lowmach.hpp"
#include "jobs/sha256.hpp"

#if DNS_WITH_GPU
#endif

#include <algorithm>
#include <chrono>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

namespace dns::drive {

using std::numbers::pi;

std::expected<std::optional<Checkpoint>, int>
readFlameRestart(const Case& c, int nY)
{
    if (c.restartPrefix.empty()) return std::optional<Checkpoint>{};
    const auto& g = c.grid;
    const std::string path = checkpointPath(c.restartPrefix);
    auto ck = Checkpoint::read(path);
    if (!ck) return std::unexpected(die(exitInput, "--restart: " + ck.error()));
    auto f = checkpointFacts(*ck);
    if (!f)
        return std::unexpected(
            die(exitInput, "--restart: " + path + ": " + f.error()));
    if (f->nx != g.nx || f->ny != g.ny || f->nz != g.nz || f->nY != nY)
        return std::unexpected(die(
            exitInput, std::format("--restart: {} holds a {} x {} x {} grid "
                                   "with {} species, the case {} x {} x {} "
                                   "with {}",
                                   path, f->nx, f->ny, f->nz, f->nY, g.nx,
                                   g.ny, g.nz, nY)));
    return std::optional<Checkpoint>(std::move(*ck));
}

std::expected<std::optional<Checkpoint>, int> readFvRestart(const Case& c, const FaceMesh& fm, int nY)
{
    if (c.restartPrefix.empty()) return std::optional<Checkpoint>{};
    const std::string path = checkpointPath(c.restartPrefix);
    auto ck = Checkpoint::read(path);
    if (!ck) return std::unexpected(die(exitInput, "--restart: " + ck.error()));
    auto f = checkpointFacts(*ck);
    if (!f) return std::unexpected(die(exitInput, "--restart: " + path + ": " + f.error()));
    if (f->nx != fm.nC || f->ny != fm.nI || f->nz != fm.nB || f->nY != nY)
        return std::unexpected(die(exitInput, std::format("--restart: {} holds a FaceMesh of {} cells, {} internal and {} "
                                                          "boundary faces with {} species, the case {}, {}, {} with {}",
                                                          path, f->nx, f->ny, f->nz, f->nY, fm.nC, fm.nI, fm.nB, nY)));
    return std::optional<Checkpoint>(std::move(*ck));
}

CheckpointPolicy flamePolicy(const Case& c, const CheckpointContext& ckx,
                             const std::optional<Checkpoint>& restart)
{
    CheckpointPolicy policy;
    policy.every = c.checkpointDt;
    policy.wallLimit = c.wallLimit;
    policy.wallStart = ckx.start;
    policy.prefix = c.outPrefix;
    policy.binary = ckx.binary;
    policy.caseText = caseFlagsText(c.caseFlags);
    policy.mesh = ckx.mesh;
    policy.restart = restart ? &*restart : nullptr;
    policy.restarts = ckx.restarts;
    return policy;
}

bool fvFlameRoute(const RawConfig& r)
{
    return r.sub == Subcommand::Flame && (r.solver == int(SolverKind::Fv) || r.given.contains("--polymesh"));
}

const char* flameSolverName(const RawConfig& r)
{
    const bool lm = r.physics == Physics::LowMach;
    if (fvFlameRoute(r)) return lm ? kFvLowMachSolverName : kFvFlameCompSolverName;
    return lm ? kLowMachSolverName : kFlameCompSolverName;
}

std::expected<std::string, std::string> fvMeshIdentity(const RawConfig& r)
{
    if (!r.given.contains("--polymesh")) return std::string("flags");
    namespace fs = std::filesystem;
    const fs::path dir = fvPolyDir(r.polymesh);
    std::error_code ec;
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir, ec))
        if (e.is_regular_file()) files.push_back(e.path());
    if (ec) return std::unexpected(std::format("cannot list '{}': {}", dir.string(), ec.message()));
    std::ranges::sort(files);
    jobs::Sha256 all;
    for (const fs::path& p : files) {
        auto h = jobs::sha256File(p.string());
        if (!h) return std::unexpected(h.error());
        const std::string line = p.filename().string() + '\0' + *h + '\n';
        all.update(std::as_bytes(std::span(line)));
    }
    auto bc = jobs::sha256File(r.bc);
    if (!bc) return std::unexpected(bc.error());
    const auto d = all.finish();
    return std::format("polyMesh sha256:{} bc sha256:{}", jobs::toHex(d), *bc);
}


} // namespace dns::drive
