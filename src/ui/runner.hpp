// Running the solver from the editor.
//
// dns-ui still solves nothing itself: it SPAWNS `dns` as a child process and
// reads its output. That is the whole reason this file exists rather than a
// link against dns_gpu -- the editor keeps linking dns_core only, it starts on
// a machine with no GPU, and the thing that runs is the same binary, with the
// same arguments, that the export panel prints. If the run misbehaves you can
// reproduce it by pasting that command into a shell.
//
// Output is captured line by line off a reader thread; the UI drains it once
// per frame. Nothing here touches Doc.
#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ui {

class Runner
{
public:
    ~Runner();

    Runner() = default;
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;

    // Spawns argv[0] with argv, cwd `dir`. Returns "" on success or the reason
    // it could not start. Refuses while a run is already in flight.
    std::string start(const std::vector<std::string>& argv, const std::string& dir);

    // SIGTERM, then SIGKILL if it does not go. Safe to call when idle.
    void stop();

    // Call once per frame: reaps the child and returns true when a run just
    // finished (so the caller can report the exit code once).
    bool poll();

    bool running() const { return running_; }
    int exitCode() const { return exit_; }
    // Wall-clock seconds since start(), frozen at the end of the run.
    double seconds() const;

    // The captured output, oldest first, capped at kMaxLines.
    std::vector<std::string> lines() const;
    std::size_t lineCount() const;
    void clear();

    static constexpr std::size_t kMaxLines = 2000;

private:
    void reap(int status);

    mutable std::mutex mu_;
    std::vector<std::string> lines_;
    std::atomic<bool> running_{false};
    std::atomic<bool> finished_{false};   // child exited, not yet reported
    int pid_ = -1;
    int fd_ = -1;
    int exit_ = 0;
    double t0_ = 0, t1_ = 0;
    std::thread reader_;
};

} // namespace ui
