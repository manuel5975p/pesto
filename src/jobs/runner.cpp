#include "jobs/runner.hpp"

#include "jobs/io.hpp"
#include "jobs/sha256.hpp"
#include "jobs/spec.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <format>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace jobs {

namespace {

volatile std::sig_atomic_t gTerminated = 0;

void onTerm(int) { gTerminated = 1; }

// oom_kill counter of this process's own cgroup (cgroup v2), 0 if unreadable.
long oomKills()
{
    auto cg = readFile("/proc/self/cgroup");
    if (!cg) return 0;
    const auto p = cg->find("0::");
    if (p == std::string::npos) return 0;
    std::string path = cg->substr(p + 3);
    path = path.substr(0, path.find('\n'));
    auto ev = readFile("/sys/fs/cgroup" + path + "/memory.events");
    if (!ev) return 0;
    const auto k = ev->find("oom_kill ");
    if (k == std::string::npos) return 0;
    return std::strtol(ev->c_str() + k + 9, nullptr, 10);
}

void logLine(int fd, const std::string& s)
{
    const std::string line = s + "\n";
    [[maybe_unused]] const ssize_t n = ::write(fd, line.data(), line.size());
}

std::vector<std::string> buildEnv(const JobSpec& spec)
{
    std::vector<std::string> env;
    for (char** e = environ; *e; ++e) {
        const std::string kv = *e;
        const std::string k = kv.substr(0, kv.find('='));
        bool overridden = false;
        for (const auto& [ok, _] : spec.env) overridden |= ok == k;
        if (!overridden) env.push_back(kv);
    }
    for (const auto& [k, v] : spec.env) env.push_back(k + "=" + v);
    return env;
}

std::vector<char*> cstrs(std::vector<std::string>& v)
{
    std::vector<char*> out;
    for (auto& s : v) out.push_back(s.data());
    out.push_back(nullptr);
    return out;
}

// One step; fills r. Returns false when the job must stop after it.
bool runStep(const JobSpec& spec, std::size_t n, int logFd, std::vector<char*>& envp, StepResult& r)
{
    const Step& step = spec.steps[n];
    r.step = int(n);
    std::string cmd = step.exe;
    for (const auto& a : step.args) cmd += " " + a;
    logLine(logFd, std::format("=== step {}: {}", n, cmd));

    const int fd = ::open(step.exe.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        r.state = State::rejected;
        r.error = std::format("open {}: {}", step.exe, std::strerror(errno));
        logLine(logFd, "=== rejected: " + r.error);
        return false;
    }
    std::vector<std::string> stale;
    const auto h = sha256Fd(fd);
    if (!h || *h != step.sha256) stale.push_back(step.exe);
    for (const FileCheck& c : step.verify) {
        const auto hv = sha256File(c.path);
        if (!hv || *hv != c.sha256) stale.push_back(c.path);
    }
    if (!stale.empty()) {
        ::close(fd);
        r.state = State::rejected;
        r.error = "checksum mismatch at launch (stale or rebuilt):";
        for (const auto& s : stale) r.error += " " + s;
        logLine(logFd, "=== rejected: " + r.error);
        return false;
    }

    std::vector<std::string> argv{step.exe};
    argv.insert(argv.end(), step.args.begin(), step.args.end());
    std::vector<char*> av = cstrs(argv);
    const long oom0 = oomKills();
    const auto t0 = std::chrono::steady_clock::now();
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(fd);
        r.state = State::failed;
        r.rc = 127;
        r.error = std::format("fork: {}", std::strerror(errno));
        return false;
    }
    if (pid == 0) {
        std::signal(SIGTERM, SIG_DFL);
        const int nul = ::open("/dev/null", O_RDONLY);
        if (nul >= 0) ::dup2(nul, 0);
        ::dup2(logFd, 1);
        ::dup2(logFd, 2);
        if (::chdir(spec.cwd.c_str()) != 0) {
            std::fprintf(stderr, "simjob: chdir %s: %s\n", spec.cwd.c_str(), std::strerror(errno));
            ::_exit(127);
        }
        ::fcntl(fd, F_SETFD, 0); // a #! script re-opens /dev/fd/N
        ::fexecve(fd, av.data(), envp.data());
        std::fprintf(stderr, "simjob: exec %s: %s\n", step.exe.c_str(), std::strerror(errno));
        ::_exit(127);
    }
    ::close(fd);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {}
    r.wallS = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    r.rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
    if (oomKills() > oom0)
        r.state = State::oom;
    else
        r.state = r.rc == 0 ? State::done : State::failed;
    logLine(logFd, std::format("=== step {} {} rc={} wall={:.3f}s", n, name(r.state), r.rc, r.wallS));
    return r.state == State::done && !gTerminated;
}

} // namespace

int runJob(const std::string& dir)
{
    // A handler, not SIG_IGN: the disposition must not survive into the steps.
    struct sigaction sa{};
    sa.sa_handler = onTerm;
    ::sigaction(SIGTERM, &sa, nullptr);

    auto text = readFile(dir + "/job.json");
    if (!text) {
        std::fprintf(stderr, "simjobd --run-job: %s\n", text.error().c_str());
        return 2;
    }
    auto v = json::parse(*text);
    if (!v) {
        std::fprintf(stderr, "simjobd --run-job: %s\n", v.error().c_str());
        return 2;
    }
    auto job = parseJob(*v);
    if (!job) {
        std::fprintf(stderr, "simjobd --run-job: %s\n", job.error().c_str());
        return 2;
    }
    const int logFd = ::open((dir + "/output.log").c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (logFd < 0) {
        std::fprintf(stderr, "simjobd --run-job: output.log: %s\n", std::strerror(errno));
        return 2;
    }
    std::vector<std::string> env = buildEnv(job->spec);
    std::vector<char*> envp = cstrs(env);
    json::Array done;
    for (std::size_t n = 0; n < job->spec.steps.size(); ++n) {
        StepResult r;
        const bool more = runStep(job->spec, n, logFd, envp, r);
        done.push_back(toJson(r));
        if (auto w = writeFileAtomic(dir + "/steps.json", json::dump(done)); !w) {
            std::fprintf(stderr, "simjobd --run-job: %s\n", w.error().c_str());
            return 2;
        }
        if (!more) break;
    }
    ::close(logFd);
    return 0;
}

} // namespace jobs
