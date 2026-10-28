#include "jobs/server.hpp"

#include "jobs/io.hpp"
#include "jobs/sched.hpp"
#include "jobs/spec.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/epoll.h>
#include <sys/file.h>
#include <sys/syscall.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

namespace jobs {

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kMaxRequestBytes = 1 << 20;
constexpr std::size_t kListFinished = 50;

enum Tag : std::uint64_t { kListen = 1, kSignal = 2, kClient = 3, kChild = 4 };
std::uint64_t tag(Tag t, std::uint64_t v) { return (std::uint64_t(t) << 56) | v; }

void say(const char* fmt, ...)
{
    char ts[32];
    const std::time_t t = std::time(nullptr);
    std::strftime(ts, sizeof ts, "%F %T", std::localtime(&t));
    std::fprintf(stderr, "simjobd %s ", ts);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fputc('\n', stderr);
}

json::Value fail(const std::string& e) { return json::Object{{"ok", false}, {"error", e}}; }

struct Running
{
    pid_t pid = -1;
    int pidfd = -1;
    double deadline = 0;
    double killAt = 0; // SIGKILL time once SIGTERM went out, 0 = not yet
    bool timedOut = false, cancelled = false;
    std::string why;
};

class Server
{
public:
    explicit Server(ServerOptions o) : o_(std::move(o)) {}

    int run()
    {
        if (!setup()) return 1;
        say("listening on %s, state %s%s", o_.socketPath.c_str(), o_.stateDir.c_str(),
            o_.gpuLock.empty() ? "" : (", gpu lock " + o_.gpuLock).c_str());
        while (!(stopping_ && running_.empty())) {
            admitJobs();
            epoll_event ev[32];
            const int n = ::epoll_wait(ep_, ev, 32, timeoutMs());
            if (n < 0 && errno != EINTR) {
                say("epoll_wait: %s", std::strerror(errno));
                return 1;
            }
            for (int i = 0; i < n; ++i) dispatch(ev[i].data.u64);
            timers();
        }
        ::unlink(o_.socketPath.c_str());
        say("stopped");
        return 0;
    }

private:
    // ---- setup ----------------------------------------------------------

    bool setup()
    {
        std::error_code ec;
        fs::create_directories(o_.stateDir, ec);
        if (ec) {
            say("state dir %s: %s", o_.stateDir.c_str(), ec.message().c_str());
            return false;
        }
        if (!o_.gpuLock.empty()) {
            gpuLockFd_ = ::open(o_.gpuLock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
            if (gpuLockFd_ < 0) {
                say("gpu lock %s: %s", o_.gpuLock.c_str(), std::strerror(errno));
                return false;
            }
        }
        recover();
        ep_ = ::epoll_create1(EPOLL_CLOEXEC);
        if (ep_ < 0) return false;
        sigset_t mask;
        sigemptyset(&mask);
        sigaddset(&mask, SIGINT);
        sigaddset(&mask, SIGTERM);
        ::sigprocmask(SIG_BLOCK, &mask, nullptr);
        std::signal(SIGPIPE, SIG_IGN);
        sigFd_ = ::signalfd(-1, &mask, SFD_CLOEXEC | SFD_NONBLOCK);
        if (sigFd_ < 0) return false;
        add(sigFd_, tag(kSignal, 0));
        return listenSocket();
    }

    bool listenSocket()
    {
        sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        if (o_.socketPath.size() >= sizeof addr.sun_path) {
            say("socket path too long");
            return false;
        }
        std::memcpy(addr.sun_path, o_.socketPath.c_str(), o_.socketPath.size() + 1);
        if (request(o_.socketPath, json::Object{{"op", "ping"}})) {
            say("another simjobd already serves %s", o_.socketPath.c_str());
            return false;
        }
        ::unlink(o_.socketPath.c_str());
        lfd_ = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
        const mode_t old = ::umask(0177); // socket born 0600: same user only
        const int b = ::bind(lfd_, reinterpret_cast<const sockaddr*>(&addr), sizeof addr);
        ::umask(old);
        if (b != 0 || ::listen(lfd_, 64) != 0) {
            say("bind %s: %s", o_.socketPath.c_str(), std::strerror(errno));
            return false;
        }
        add(lfd_, tag(kListen, 0));
        return true;
    }

    // Jobs on disk from an earlier server: queued ones are re-admitted in id
    // order, running ones cannot be re-attached and are marked lost.
    void recover()
    {
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(o_.stateDir, ec)) {
            const std::string nm = e.path().filename().string();
            if (nm.empty() || !std::ranges::all_of(nm, [](char c) { return c >= '0' && c <= '9'; })) continue;
            auto text = readFile(e.path().string() + "/job.json");
            if (!text) continue;
            auto v = json::parse(*text);
            if (!v) continue;
            auto j = parseJob(*v);
            if (!j) continue;
            nextId_ = std::max(nextId_, j->id + 1);
            if (j->state == State::running) {
                j->state = State::lost;
                j->detail = "simjobd restarted while the job ran";
                j->ended = nowUnix();
                save(*j);
            }
            jobs_[j->id] = std::move(*j);
        }
        for (const auto& [id, j] : jobs_)
            if (j.state == State::queued) queue_.push_back(id);
        if (!queue_.empty()) say("recovered %zu queued job(s)", queue_.size());
    }

    void add(int fd, std::uint64_t data)
    {
        epoll_event ev{};
        ev.events = EPOLLIN;
        ev.data.u64 = data;
        ::epoll_ctl(ep_, EPOLL_CTL_ADD, fd, &ev);
    }

    void save(const Job& j)
    {
        if (auto w = writeFileAtomic(j.dir + "/job.json", json::dump(toJson(j))); !w)
            say("job %lld: %s", (long long)j.id, w.error().c_str());
    }

    // ---- events ---------------------------------------------------------

    void dispatch(std::uint64_t d)
    {
        const auto t = Tag(d >> 56);
        const std::uint64_t v = d & ((std::uint64_t(1) << 56) - 1);
        switch (t) {
        case kListen: accept(); break;
        case kSignal: onSignal(); break;
        case kClient: onClient(int(v)); break;
        case kChild: onChildExit(std::int64_t(v)); break;
        }
    }

    void onSignal()
    {
        signalfd_siginfo si;
        while (::read(sigFd_, &si, sizeof si) == sizeof si) {}
        if (stopping_) return;
        say("shutting down: %zu running job(s) terminated, %zu queued stay queued", running_.size(), queue_.size());
        stopping_ = true;
        ::epoll_ctl(ep_, EPOLL_CTL_DEL, lfd_, nullptr);
        ::close(lfd_);
        for (auto& [id, r] : running_) terminate(id, r, false, "simjobd shutdown");
    }

    void accept()
    {
        for (;;) {
            const int c = ::accept4(lfd_, nullptr, nullptr, SOCK_CLOEXEC | SOCK_NONBLOCK);
            if (c < 0) return;
            ucred cr{};
            socklen_t len = sizeof cr;
            if (::getsockopt(c, SOL_SOCKET, SO_PEERCRED, &cr, &len) != 0 || cr.uid != ::getuid()) {
                ::close(c);
                continue;
            }
            buf_[c].clear();
            add(c, tag(kClient, std::uint64_t(c)));
        }
    }

    void onClient(int c)
    {
        char tmp[1 << 16];
        bool eof = false;
        for (;;) {
            const ssize_t n = ::recv(c, tmp, sizeof tmp, 0);
            if (n > 0) {
                buf_[c].append(tmp, std::size_t(n));
                if (buf_[c].size() > kMaxRequestBytes) return reply(c, fail("request too large"));
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            eof = n == 0 || errno != EAGAIN;
            break;
        }
        const auto nl = buf_[c].find('\n');
        if (nl == std::string::npos) {
            if (eof) drop(c); // closed before a full line
            return;
        }
        const auto req = json::parse(std::string_view(buf_[c]).substr(0, nl));
        if (!req) return reply(c, fail(req.error()));
        handle(c, *req);
    }

    void drop(int c)
    {
        ::epoll_ctl(ep_, EPOLL_CTL_DEL, c, nullptr);
        ::close(c);
        buf_.erase(c);
    }

    void reply(int c, const json::Value& v)
    {
        const std::string line = json::dump(v) + "\n";
        const int fl = ::fcntl(c, F_GETFL);
        ::fcntl(c, F_SETFL, fl & ~O_NONBLOCK);
        timeval tv{2, 0};
        ::setsockopt(c, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
        std::size_t off = 0;
        while (off < line.size()) {
            const ssize_t n = ::send(c, line.data() + off, line.size() - off, MSG_NOSIGNAL);
            if (n <= 0) break;
            off += std::size_t(n);
        }
        drop(c);
    }

    // ---- requests -------------------------------------------------------

    void handle(int c, const json::Value& req)
    {
        const json::Value* op = req.find("op");
        if (!op || !op->string()) return reply(c, fail("missing \"op\""));
        const std::string& o = *op->string();
        if (o == "ping") return reply(c, json::Object{{"ok", true}, {"pid", int(::getpid())}});
        if (o == "submit") return reply(c, submit(req));
        if (o == "list") return reply(c, list());
        const json::Value* idv = req.find("id");
        if (!idv || !idv->integer()) return reply(c, fail("missing integer \"id\""));
        const auto it = jobs_.find(*idv->integer());
        if (it == jobs_.end()) return reply(c, fail(std::format("no job {}", *idv->integer())));
        Job& j = it->second;
        if (o == "status") return reply(c, json::Object{{"ok", true}, {"job", toJson(j)}});
        if (o == "wait") {
            if (finished(j.state)) return reply(c, json::Object{{"ok", true}, {"job", toJson(j)}});
            ::epoll_ctl(ep_, EPOLL_CTL_DEL, c, nullptr); // parked until the job ends
            waiters_[j.id].push_back(c);
            return;
        }
        if (o == "cancel") return reply(c, cancel(j));
        reply(c, fail("unknown op \"" + o + "\""));
    }

    json::Value submit(const json::Value& req)
    {
        if (stopping_) return fail("simjobd is shutting down");
        const json::Value* jv = req.find("job");
        if (!jv) return fail("missing \"job\"");
        auto spec = parseSpec(*jv);
        if (!spec) return fail(spec.error());
        if (auto f = checkFiles(*spec); !f) return fail(f.error());
        Job j;
        j.id = nextId_++;
        j.spec = std::move(*spec);
        j.submitted = nowUnix();
        j.dir = o_.stateDir + "/" + std::to_string(j.id);
        std::error_code ec;
        fs::create_directories(j.dir, ec);
        if (ec) return fail("job dir: " + ec.message());
        save(j);
        say("job %lld queued: %s %s%s, %d s, %lld MB, %zu step(s)", (long long)j.id, j.spec.label.c_str(),
            j.spec.resource == Resource::gpu ? "gpu" : "cpu", j.spec.benchmark ? " BENCHMARK" : "",
            j.spec.maxRuntimeS, (long long)j.spec.maxMemMb, j.spec.steps.size());
        queue_.push_back(j.id);
        const auto id = j.id;
        jobs_[id] = std::move(j);
        return json::Object{{"ok", true}, {"id", (long long)id}};
    }

    json::Value list() const
    {
        json::Array out;
        std::size_t finishedShown = 0;
        for (auto it = jobs_.rbegin(); it != jobs_.rend(); ++it) {
            const Job& j = it->second;
            if (finished(j.state) && finishedShown++ >= kListFinished) continue;
            json::Object o{{"id", (long long)j.id},
                           {"state", std::string(name(j.state))},
                           {"label", j.spec.label},
                           {"resource", j.spec.resource == Resource::gpu ? "gpu" : "cpu"},
                           {"benchmark", j.spec.benchmark},
                           {"submitted", j.submitted},
                           {"started", j.started},
                           {"ended", j.ended},
                           {"rc", j.rc}};
            out.insert(out.begin(), json::Value(std::move(o)));
        }
        return json::Object{{"ok", true}, {"jobs", std::move(out)}};
    }

    json::Value cancel(Job& j)
    {
        if (j.state == State::queued) {
            std::erase(queue_, j.id);
            j.state = State::cancelled;
            j.detail = "cancelled while queued";
            j.ended = nowUnix();
            save(j);
            notify(j);
            return json::Object{{"ok", true}};
        }
        const auto r = running_.find(j.id);
        if (r == running_.end()) return fail(std::format("job {} already {}", j.id, name(j.state)));
        terminate(j.id, r->second, false, "cancelled");
        return json::Object{{"ok", true}};
    }

    void notify(const Job& j)
    {
        const auto it = waiters_.find(j.id);
        if (it == waiters_.end()) return;
        for (int c : it->second) reply(c, json::Object{{"ok", true}, {"job", toJson(j)}});
        waiters_.erase(it);
    }

    // ---- scheduling and processes --------------------------------------

    bool gpuFree()
    {
        if (gpuLockFd_ < 0 || gpuLockHeld_) return true;
        if (::flock(gpuLockFd_, LOCK_EX | LOCK_NB) != 0) return false;
        gpuLockHeld_ = true;
        return true;
    }

    void releaseGpuLockIfIdle()
    {
        if (!gpuLockHeld_) return;
        for (const auto& [id, r] : running_) {
            const JobSpec& s = jobs_.at(id).spec;
            if (s.resource == Resource::gpu || s.benchmark) return;
        }
        ::flock(gpuLockFd_, LOCK_UN);
        gpuLockHeld_ = false;
    }

    Occupancy occupancy() const
    {
        Occupancy o;
        for (const auto& [id, r] : running_) {
            const JobSpec& s = jobs_.at(id).spec;
            (s.resource == Resource::gpu ? o.gpu : o.cpu) = true;
            o.benchmark |= s.benchmark;
        }
        return o;
    }

    void admitJobs()
    {
        if (stopping_ || queue_.empty()) return;
        std::vector<Pending> pend;
        bool wantsGpu = false;
        for (std::int64_t id : queue_) {
            const JobSpec& s = jobs_.at(id).spec;
            pend.push_back({id, s.resource, s.benchmark});
            wantsGpu |= s.resource == Resource::gpu || s.benchmark;
        }
        const bool free = !wantsGpu || gpuFree();
        for (std::int64_t id : admit(pend, occupancy(), free)) launch(jobs_.at(id));
        releaseGpuLockIfIdle();
    }

    void launch(Job& j)
    {
        std::erase(queue_, j.id);
        const std::string unit = std::format("simjob-{}-{}", j.id, ::getpid());
        std::vector<std::string> argv{"systemd-run", "--user", "--scope", "--quiet", "--collect",
                                      "--unit=" + unit,
                                      "-p", std::format("MemoryMax={}M", j.spec.maxMemMb),
                                      "-p", "MemorySwapMax=0",
                                      "-p", "OOMPolicy=continue",
                                      "--", o_.selfExe, "--run-job", j.dir};
        j.state = State::running;
        j.started = nowUnix();
        save(j);
        const pid_t pid = ::fork();
        if (pid == 0) {
            sigset_t none;
            sigemptyset(&none);
            ::sigprocmask(SIG_SETMASK, &none, nullptr);
            const int log = ::open((j.dir + "/runner.log").c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
            if (log >= 0) {
                ::dup2(log, 1);
                ::dup2(log, 2);
            }
            std::vector<char*> av;
            for (auto& s : argv) av.push_back(s.data());
            av.push_back(nullptr);
            ::execvp(av[0], av.data());
            std::fprintf(stderr, "exec systemd-run: %s\n", std::strerror(errno));
            ::_exit(127);
        }
        Running r;
        r.pid = pid;
        // Raw syscall: glibc's pidfd_open declaration lacks C linkage in C++.
        r.pidfd = pid > 0 ? int(::syscall(SYS_pidfd_open, pid, 0)) : -1;
        r.deadline = j.started + j.spec.maxRuntimeS;
        if (pid < 0 || r.pidfd < 0) {
            j.state = State::failed;
            j.detail = std::format("could not start the runner: {}", std::strerror(errno));
            j.ended = nowUnix();
            save(j);
            notify(j);
            return;
        }
        add(r.pidfd, tag(kChild, std::uint64_t(j.id)));
        running_[j.id] = r;
        say("job %lld started (%s)", (long long)j.id, unit.c_str());
    }

    // The cgroup the runner lives in, once systemd-run has moved it.
    std::optional<std::string> scopeOf(pid_t pid) const
    {
        auto cg = readFile(std::format("/proc/{}/cgroup", pid));
        if (!cg) return std::nullopt;
        const auto p = cg->find("0::");
        if (p == std::string::npos) return std::nullopt;
        std::string path = cg->substr(p + 3);
        path = path.substr(0, path.find('\n'));
        if (path.find("/simjob-") == std::string::npos) return std::nullopt;
        return "/sys/fs/cgroup" + path;
    }

    void signalJob(const Running& r, int sig) const
    {
        if (const auto cg = scopeOf(r.pid)) {
            if (sig == SIGKILL) {
                if (writeFileAtomicNoRename(*cg + "/cgroup.kill", "1")) return;
            }
            if (auto procs = readFile(*cg + "/cgroup.procs")) {
                for (const char* s = procs->c_str(); *s;) {
                    char* end = nullptr;
                    const long p = std::strtol(s, &end, 10);
                    if (end == s) break;
                    if (p > 0) ::kill(pid_t(p), sig);
                    s = end;
                    while (*s == '\n') ++s;
                }
                return;
            }
        }
        ::kill(r.pid, sig);
    }

    static bool writeFileAtomicNoRename(const std::string& path, const char* s)
    {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
        if (fd < 0) return false;
        const bool ok = ::write(fd, s, std::strlen(s)) == ssize_t(std::strlen(s));
        ::close(fd);
        return ok;
    }

    void terminate(std::int64_t id, Running& r, bool timeout, const std::string& why)
    {
        if (r.killAt != 0) return;
        r.timedOut = timeout;
        r.cancelled = !timeout;
        r.why = why;
        r.killAt = nowUnix() + kKillGraceS;
        say("job %lld: SIGTERM (%s)", (long long)id, why.c_str());
        signalJob(r, SIGTERM);
    }

    int timeoutMs() const
    {
        double next = 1e300;
        for (const auto& [id, r] : running_) next = std::min(next, r.killAt != 0 ? r.killAt : r.deadline);
        if (next > 1e299) return -1;
        return std::clamp(int(std::ceil((next - nowUnix()) * 1000)), 0, 60000);
    }

    void timers()
    {
        const double now = nowUnix();
        for (auto& [id, r] : running_) {
            if (r.killAt == 0 && now >= r.deadline)
                terminate(id, r, true, std::format("max_runtime_s {} reached", jobs_.at(id).spec.maxRuntimeS));
            else if (r.killAt != 0 && now >= r.killAt) {
                say("job %lld: SIGKILL after %d s grace", (long long)id, kKillGraceS);
                signalJob(r, SIGKILL);
                r.killAt = now + 3600; // once
            }
        }
    }

    void onChildExit(std::int64_t id)
    {
        const auto it = running_.find(id);
        if (it == running_.end()) return;
        Running r = it->second;
        int status = 0;
        if (::waitpid(r.pid, &status, WNOHANG) == 0) return; // spurious
        ::epoll_ctl(ep_, EPOLL_CTL_DEL, r.pidfd, nullptr);
        ::close(r.pidfd);
        running_.erase(it);
        Job& j = jobs_.at(id);
        j.ended = nowUnix();
        j.steps.clear();
        if (auto t = readFile(j.dir + "/steps.json"))
            if (auto v = json::parse(*t); v && v->array())
                for (const json::Value& x : *v->array())
                    if (auto s = parseStepResult(x)) j.steps.push_back(*s);
        const int runnerRc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
        if (r.timedOut) {
            j.state = State::timeout;
            j.detail = r.why;
        } else if (r.cancelled) {
            j.state = State::cancelled;
            j.detail = r.why;
        } else if (j.steps.empty()) {
            j.state = State::failed;
            j.detail = std::format("runner exited {} before any step (see {}/runner.log)", runnerRc, j.dir);
        } else {
            const StepResult& last = j.steps.back();
            const bool all = last.state == State::done && j.steps.size() == j.spec.steps.size();
            j.state = all ? State::done : last.state == State::done ? State::failed : last.state;
            if (!all && last.state == State::done) j.detail = std::format("runner exited {} after step {}", runnerRc, last.step);
            if (!last.error.empty()) j.detail = last.error;
        }
        if (!j.steps.empty()) j.rc = j.steps.back().rc;
        save(j);
        say("job %lld %s rc=%d wall=%.1fs", (long long)id, std::string(name(j.state)).c_str(), j.rc, j.ended - j.started);
        notify(j);
        releaseGpuLockIfIdle();
    }

    ServerOptions o_;
    int ep_ = -1, lfd_ = -1, sigFd_ = -1, gpuLockFd_ = -1;
    bool gpuLockHeld_ = false, stopping_ = false;
    std::int64_t nextId_ = 1;
    std::map<std::int64_t, Job> jobs_;
    std::vector<std::int64_t> queue_;
    std::map<std::int64_t, Running> running_;
    std::map<std::int64_t, std::vector<int>> waiters_;
    std::map<int, std::string> buf_;
};

} // namespace

int serve(const ServerOptions& o)
{
    Server s(o);
    return s.run();
}

} // namespace jobs
