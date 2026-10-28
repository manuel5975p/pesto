// The in-scope step runner (`simjobd --run-job DIR`). simjobd starts it as the
// command of the job's transient systemd scope, so every step shares the job's
// cgroup: one memory cap, swap 0, OOMPolicy=continue (an OOM kill ends the step,
// not the runner, so the runner can report it).
#pragma once

#include <string>

namespace jobs {

// Run <dir>/job.json's steps in order. Per step: open the exe, hash the OPEN
// fd, fexecve that same fd -- the bytes checked are the bytes run. Writes
// <dir>/steps.json after every step and appends all output to <dir>/output.log.
// SIGTERM is forwarded by the cgroup kill; the runner then stops after the
// current step. Returns the process exit code (0 unless the runner itself
// could not work).
int runJob(const std::string& dir);

} // namespace jobs
