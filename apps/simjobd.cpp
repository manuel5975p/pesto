// simjobd -- the simulation job server. See docs/jobs.md.

#include "jobs/io.hpp"
#include "jobs/runner.hpp"
#include "jobs/server.hpp"

#include <climits>
#include <cstdio>
#include <cstring>
#include <string>

#include <unistd.h>

namespace {

constexpr const char* kUsage = R"(usage: simjobd [--socket PATH] [--state DIR] [--gpu-lock FILE]

Queues jobs posted on a unix socket and runs them one gpu job and one cpu job
at a time (benchmark jobs alone), each in a transient systemd scope with its
own memory cap (swap 0) and a hard deadline. Clients: `simjob`.

  --socket PATH    listen here [$SIMJOBS_SOCKET, else $XDG_RUNTIME_DIR/simjobs.sock]
  --state DIR      job records and logs [$XDG_STATE_HOME/simjobs]
  --gpu-lock FILE  also hold flock(FILE) while a gpu or benchmark job runs, so
                   GPU users outside the server (flock wrappers) serialize with it
)";

} // namespace

int main(int argc, char** argv)
{
    jobs::ServerOptions o{jobs::defaultSocketPath(), jobs::defaultStateDir(), "", ""};
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool hasValue = i + 1 < argc;
        if (a == "--help" || a == "-h") {
            std::fputs(kUsage, stdout);
            return 0;
        }
        if (a == "--run-job" && hasValue) return jobs::runJob(argv[++i]);
        if (a == "--socket" && hasValue)
            o.socketPath = argv[++i];
        else if (a == "--state" && hasValue)
            o.stateDir = argv[++i];
        else if (a == "--gpu-lock" && hasValue)
            o.gpuLock = argv[++i];
        else {
            std::fprintf(stderr, "simjobd: bad argument \"%s\"\n%s", a.c_str(), kUsage);
            return 1;
        }
    }
    char self[PATH_MAX];
    const ssize_t n = ::readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0) {
        std::fprintf(stderr, "simjobd: readlink /proc/self/exe: %s\n", std::strerror(errno));
        return 1;
    }
    self[n] = '\0';
    o.selfExe = self;
    return jobs::serve(o);
}
