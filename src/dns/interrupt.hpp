// Clean shutdown on SIGINT / SIGTERM.
//
// Without this, Ctrl-C or a `kill` terminates the process mid-step and NO
// destructor runs: the stats CSV loses its tail, and -- since --emit shm -- a
// shared-memory segment survives its own run, still flagged `running`, with a
// pid that the OS is free to hand to something else. A viewer then sees a live
// run that does not exist and refuses to start a new one.
//
// So the signal only sets a flag. The runners check it once per step and return
// normally, which unwinds through every destructor exactly as reaching the end
// time does. The handler itself does nothing that is not async-signal-safe.
#pragma once

#include <atomic>
#include <csignal>

namespace dns {

// The flag the handler sets. `volatile sig_atomic_t` is the only thing a
// handler may portably write; the atomic wrapper is for the readers.
inline volatile std::sig_atomic_t g_stopRequested = 0;

// True once a stop signal has arrived. Cheap enough to test every step.
inline bool stopRequested() { return g_stopRequested != 0; }

// Installs the handler for SIGINT and SIGTERM. Idempotent. Call once, early.
void installInterruptHandler();

// Why the run ended, for the log line.
inline const char* stopReason()
{
    return g_stopRequested == SIGINT ? "interrupted (SIGINT)"
                                     : "stopped (SIGTERM)";
}

} // namespace dns
