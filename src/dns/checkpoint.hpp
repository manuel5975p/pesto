// Checkpoint / restart of a time-loop run (docs/chem.md "Checkpoint and
// restart"; masterplan real_bunsen S2). Host-only: the file format, its
// atomic write, the phase-1.5 header read, and the case signature a restart
// is checked against. The runner side (device state, loop state) is
// gpu/internal/checkpoint.hpp.
//
// FILE `<prefix>_ckpt.bin` (little-endian, native widths):
//   "DNSCKPT\0"  u32 version (= kCheckpointVersion)  u32 record count
//   record*      u32 name length, name bytes, u32 type, u64 element count,
//                payload (count elements: text = bytes, F64 / I64 = 8 bytes,
//                F32 = 4 bytes, the raw bit pattern -- doubles and device
//                floats round-trip exactly)
//   "ENDCKPT\0"  (a file cut short is rejected, never half-read)
// Records are NAMED, looked up by name, order-free: a solver adds its own
// (the low-Mach runner's "state.p") without a version bump; a missing name is
// an error by name. Standard records: "binary" (sha256 of the dns binary),
// "solver", "case" (caseFlagsText), "grid" (nx, ny, nz, nY), "t", "steps",
// "restarts"; "mesh" (the FV solvers' mesh identity, CheckpointPolicy::mesh;
// their "grid" holds nC, nI, nB, nY); "loop.*" (the runner's loop-state
// struct), "out.*" (the RunOutput resume point), "state.*" (raw device fields).
#pragma once

#include "dns/output.hpp"

#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dns {

inline constexpr std::uint32_t kCheckpointVersion = 1;

// `<prefix>_ckpt.bin`: the one spelling of the checkpoint's path.
std::string checkpointPath(const std::string& prefix);

// The case-defining flags the user gave, (name, canonical value), sorted by
// name: cli::parse fills it from the FINAL parsed values (a number in its
// shortest round-trip form, an enum by its name, --emit as its sorted set),
// so spelling and order do not matter. Excludes what may change between the
// jobs of one run: --tt, --checkpoint-dt, --wall-limit, --restart, --out,
// --quiet, --dry-run, --backend.
using CaseFlags = std::vector<std::pair<std::string, std::string>>;

// One "name value" line per flag (a bare flag: its name alone).
std::string caseFlagsText(const CaseFlags&);
// Inverse of caseFlagsText. Pre: text came from caseFlagsText.
CaseFlags parseCaseFlagsText(std::string_view text);
// The first flag (by name) whose value differs, as "--jet-speed 4 vs 5"
// (checkpoint first; "(absent)" for a flag only one side gave); nullopt when
// the two are equal.
std::optional<std::string> caseFlagsDiff(const CaseFlags& ckpt,
                                         const CaseFlags& run);

// The named records of one checkpoint, in memory.
class Checkpoint
{
public:
    enum class Type : std::uint32_t { Text = 1, F64 = 2, I64 = 3, F32 = 4 };

    // Adds or replaces a record.
    void putText(std::string name, std::string_view v);
    void putF64(std::string name, std::span<const double> v);
    void putI64(std::string name, std::span<const std::int64_t> v);
    void putF32(std::string name, std::span<const float> v);
    void putF64(std::string name, double v) { putF64(std::move(name), {&v, 1}); }
    void putI64(std::string name, std::int64_t v) { putI64(std::move(name), {&v, 1}); }

    bool has(std::string_view name) const;
    // A missing record or a type/size mismatch is an error naming it.
    std::expected<std::string, std::string> text(std::string_view name) const;
    std::expected<std::vector<double>, std::string> f64(std::string_view name) const;
    std::expected<std::vector<std::int64_t>, std::string> i64(std::string_view name) const;
    std::expected<double, std::string> f64Scalar(std::string_view name) const;
    std::expected<std::int64_t, std::string> i64Scalar(std::string_view name) const;
    // F32 payload copied into `out` (exact size n). Pre: out holds n floats.
    std::expected<void, std::string> f32Into(std::string_view name, float* out,
                                             std::size_t n) const;
    // Names of the records starting with `prefix`, ascending.
    std::vector<std::string> names(std::string_view prefix) const;

    // Writes `path.tmp`, fsyncs it, renames it over `path` and fsyncs the
    // directory: a job killed mid-write keeps the previous checkpoint.
    std::expected<void, std::string> write(const std::string& path) const;
    // Reads a checkpoint; headerOnly skips the F32 payloads (the phase-1.5
    // read: the facts without the fields). Errors name the path.
    static std::expected<Checkpoint, std::string> read(const std::string& path,
                                                       bool headerOnly = false);

private:
    struct Record
    {
        std::string name;
        Type type;
        std::uint64_t count = 0;
        std::vector<std::byte> bytes;
    };
    const Record* find(std::string_view name) const;
    Record& slot(std::string name, Type t);
    std::vector<Record> recs_;
};

// The header facts a restart is validated against (phase 1.5).
struct CheckpointFacts
{
    std::string binary, solver;
    std::string mesh; // the "mesh" record; empty when absent (a structured solver)
    std::string bcSections; // the "bc.sections" record ([nozzle]/[thrust] lines); empty when absent
    CaseFlags caseFlags;
    int nx = 0, ny = 0, nz = 0, nY = 0;
    double t = 0;
    long steps = 0, restarts = 0;
};
std::expected<CheckpointFacts, std::string> checkpointFacts(const Checkpoint&);

// The RunOutput resume point <-> "out.*" records.
void putOutputResume(Checkpoint&, const OutputResume&);
std::expected<OutputResume, std::string> outputResume(const Checkpoint&);

// What a runner needs to checkpoint and to resume (the drivers fill it, dns/drive).
struct CheckpointPolicy
{
    double every = 0;          // --checkpoint-dt (model time); 0 = off
    double wallLimit = -1;     // --wall-limit seconds; < 0 = none
    // The job's start (main's first line): --wall-limit counts from here.
    std::chrono::steady_clock::time_point wallStart{};
    std::string prefix;        // the run's --out
    std::string binary;        // sha256 of the dns binary
    std::string caseText;      // caseFlagsText of the run
    // The FV solvers' mesh identity (the "mesh" record, compared by text in
    // phase 1.5): "polyMesh sha256:<hex> bc sha256:<hex>" over the polyMesh
    // files and the --bc file, "flags" for a box the case flags define. Empty:
    // a structured grid (no record).
    std::string mesh;
    // The --bc file's [nozzle]/[thrust] lines (fvBcCheckpointSections): the "bc.sections" text
    // record, compared section by section before the mesh. Empty: no record.
    std::string bcSections;
    const Checkpoint* restart = nullptr; // --restart: the loaded checkpoint
    long restarts = 0;         // this job's restart count (ckpt's + 1)
    bool active() const { return every > 0 || restart != nullptr; }
};

} // namespace dns
