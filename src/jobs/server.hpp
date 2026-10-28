// simjobd: a single-threaded epoll loop that queues jobs posted on a unix
// socket, admits them by the `admit` rule, runs each in its own transient
// systemd scope (memory cap, swap 0) and stops it at its deadline.
#pragma once

#include <string>

namespace jobs {

struct ServerOptions
{
    std::string socketPath;
    std::string stateDir;
    std::string gpuLock; // flock(2) file shared with other GPU users; empty = none
    std::string selfExe; // this binary, re-entered as `--run-job DIR` inside the scope
};

// Runs until SIGINT/SIGTERM: then stops accepting, terminates running jobs
// (state cancelled, "server shutdown") and returns 0. Queued jobs stay queued
// on disk and are re-admitted by the next start. Returns non-zero on a setup
// failure (socket in use, state dir unwritable, ...).
int serve(const ServerOptions& o);

} // namespace jobs
