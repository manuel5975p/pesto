// What a job IS: the submitted spec (strictly validated, hard limits), its
// lifecycle state and the on-disk record. Shared by simjobd, its in-scope
// runner and the simjob client so all three agree on one schema.
#pragma once

#include "jobs/json.hpp"

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jobs {

// Hard caps. A spec over either is rejected at submit, never clamped.
inline constexpr int kMaxRuntimeS = 1800;      // 30 min
inline constexpr std::int64_t kMaxMemMb = 16384; // 16 GiB
inline constexpr std::int64_t kMinMemMb = 16;    // below this the runner itself cannot live
inline constexpr std::size_t kMaxSteps = 64;
// SIGTERM at the deadline, SIGKILL this much later.
inline constexpr int kKillGraceS = 10;

enum class Resource { cpu, gpu };

struct FileCheck
{
    std::string path, sha256;
};

// One invocation. exe is absolute and pinned by sha256; `verify` pins further
// files the invocation depends on (e.g. the solver mpirun launches).
struct Step
{
    std::string exe, sha256;
    std::vector<std::string> args;
    std::vector<FileCheck> verify;
};

struct JobSpec
{
    std::string label;
    Resource resource = Resource::cpu;
    bool benchmark = false; // runs alone: nothing else starts or runs beside it
    int maxRuntimeS = 0;    // whole job, all steps
    std::int64_t maxMemMb = 0; // whole job cgroup, swap 0
    std::string cwd;           // absolute
    std::vector<std::pair<std::string, std::string>> env; // overrides on the server's env
    std::vector<Step> steps;   // run in order; the first failure ends the job
};

// Shape and limit validation only (no filesystem). Unknown keys are errors.
std::expected<JobSpec, std::string> parseSpec(const json::Value& v);
json::Value toJson(const JobSpec& s);

// Filesystem checks at submit: cwd is a directory, every exe is an absolute
// executable regular file, every pinned hash matches NOW.
std::expected<void, std::string> checkFiles(const JobSpec& s);

enum class State { queued, running, done, failed, timeout, oom, cancelled, rejected, lost };
std::string_view name(State s);
std::optional<State> stateFrom(std::string_view s);
inline bool finished(State s) { return s != State::queued && s != State::running; }

// Exit code `simjob wait` mirrors: 0 done, the failing step's rc (1..255) on
// failed, 124 timeout, 137 oom, 130 cancelled, 3 rejected, 75 lost.
int exitCode(State s, int rc);

// Per-step outcome the runner writes to <jobdir>/steps.json.
struct StepResult
{
    int step = 0;
    State state = State::failed; // done | failed | oom | rejected
    int rc = 0;                  // exit status, or 128 + signal
    double wallS = 0;
    std::string error;
};
json::Value toJson(const StepResult& r);
std::expected<StepResult, std::string> parseStepResult(const json::Value& v);

// The job record, <jobdir>/job.json; times are unix seconds, 0 = not yet.
struct Job
{
    std::int64_t id = 0;
    JobSpec spec;
    State state = State::queued;
    double submitted = 0, started = 0, ended = 0;
    int rc = 0;
    std::string detail;
    std::string dir;
    std::vector<StepResult> steps;
};
json::Value toJson(const Job& j);
std::expected<Job, std::string> parseJob(const json::Value& v);

} // namespace jobs
