#include "ui/runner.hpp"

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace ui {
namespace {

double nowSeconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

Runner::~Runner()
{
    stop();
    if (reader_.joinable()) reader_.join();
}

std::string Runner::start(const std::vector<std::string>& argv,
                          const std::string& dir)
{
    if (running_) return "a run is already in flight";
    if (argv.empty()) return "empty command";
    if (reader_.joinable()) reader_.join();

    // Everything the child needs is built BEFORE fork: between fork and exec
    // only async-signal-safe calls are allowed, and a vector allocation (or a
    // std::string copy) in a child forked from a threaded process can deadlock
    // on an allocator lock another thread held at the fork.
    std::vector<char*> cargv;
    cargv.reserve(argv.size() + 1);
    for (const auto& a : argv) cargv.push_back(const_cast<char*>(a.c_str()));
    cargv.push_back(nullptr);
    const char* const cdir = dir.empty() ? nullptr : dir.c_str();

    int fds[2];
    if (pipe(fds) != 0) return std::string("pipe: ") + std::strerror(errno);

    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return std::string("fork: ") + std::strerror(errno);
    }

    if (pid == 0) {
        // Child. Only async-signal-safe work between fork and exec.
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        // Its own process group, so stop() can signal the whole run and not
        // the editor.
        setpgid(0, 0);
        if (cdir && chdir(cdir) != 0) _exit(127);
        execv(cargv[0], cargv.data());
        execvp(cargv[0], cargv.data());
        _exit(127);   // exec failed; the parent sees 127
    }

    // The parent sets the group too: whichever of the two runs first, the
    // group exists before stop() can signal it -- a stop() racing the child's
    // own setpgid would otherwise signal a group that does not exist yet and
    // leave the child running. (EACCES after the child's exec is harmless: it
    // had already done it.)
    setpgid(pid, pid);
    close(fds[1]);
    {
        std::lock_guard lk(mu_);
        lines_.clear();
    }
    pid_ = pid;
    fd_ = fds[0];
    exit_ = 0;
    t0_ = nowSeconds();
    t1_ = 0;
    running_ = true;
    finished_ = false;

    reader_ = std::thread([this, fd = fds[0]] {
        std::string partial;
        char buf[4096];
        for (;;) {
            const ssize_t n = read(fd, buf, sizeof buf);
            if (n <= 0) break;
            for (ssize_t i = 0; i < n; ++i) {
                if (buf[i] == '\n') {
                    std::lock_guard lk(mu_);
                    // A ring in spirit: a long run prints thousands of stats
                    // rows and the dock only ever shows the tail.
                    if (lines_.size() >= kMaxLines) lines_.erase(lines_.begin());
                    lines_.push_back(partial);
                    partial.clear();
                } else if (buf[i] != '\r') {
                    partial += buf[i];
                }
            }
        }
        if (!partial.empty()) {
            std::lock_guard lk(mu_);
            lines_.push_back(partial);
        }
        close(fd);
        finished_ = true;   // the pipe closed: the child is done writing
    });
    return {};
}

void Runner::stop()
{
    if (!running_ || pid_ <= 0) return;
    // The whole process group: the child made itself a leader in start().
    kill(-pid_, SIGTERM);
    for (int i = 0; i < 200; ++i) {
        int status = 0;
        const pid_t r = waitpid(pid_, &status, WNOHANG);
        if (r == pid_) {
            reap(status);
            return;
        }
        usleep(10000);
    }
    kill(-pid_, SIGKILL);
    int status = 0;
    if (waitpid(pid_, &status, 0) == pid_) reap(status);
}

void Runner::reap(int status)
{
    exit_ = WIFEXITED(status) ? WEXITSTATUS(status)
                              : (WIFSIGNALED(status) ? 128 + WTERMSIG(status) : -1);
    t1_ = nowSeconds();
    running_ = false;
    pid_ = -1;
    fd_ = -1;
}

bool Runner::poll()
{
    if (!running_ || pid_ <= 0) return false;
    int status = 0;
    const pid_t r = waitpid(pid_, &status, WNOHANG);
    if (r != pid_) return false;
    reap(status);
    if (reader_.joinable()) reader_.join();
    return true;
}

double Runner::seconds() const
{
    if (t0_ == 0) return 0;
    return (t1_ > 0 ? t1_ : nowSeconds()) - t0_;
}

std::vector<std::string> Runner::lines() const
{
    std::lock_guard lk(mu_);
    return lines_;
}

std::size_t Runner::lineCount() const
{
    std::lock_guard lk(mu_);
    return lines_.size();
}

void Runner::clear()
{
    std::lock_guard lk(mu_);
    lines_.clear();
}

} // namespace ui
