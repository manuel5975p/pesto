#include "jobs/io.hpp"

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace jobs {

namespace {

std::unexpected<std::string> errnoErr(const std::string& what)
{
    return std::unexpected(what + ": " + std::strerror(errno));
}

} // namespace

std::expected<std::string, std::string> readFile(const std::string& path)
{
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return errnoErr("open " + path);
    std::string out;
    char buf[1 << 16];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof buf);
        if (n < 0) {
            if (errno == EINTR) continue;
            const auto e = errnoErr("read " + path);
            ::close(fd);
            return e;
        }
        if (n == 0) break;
        out.append(buf, std::size_t(n));
    }
    ::close(fd);
    return out;
}

std::expected<void, std::string> writeFileAtomic(const std::string& path, const std::string& data)
{
    const std::string tmp = path + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return errnoErr("open " + tmp);
    std::size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::write(fd, data.data() + off, data.size() - off);
        if (n < 0) {
            if (errno == EINTR) continue;
            const auto e = errnoErr("write " + tmp);
            ::close(fd);
            return e;
        }
        off += std::size_t(n);
    }
    ::fsync(fd);
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str()) != 0) return errnoErr("rename " + tmp);
    return {};
}

std::string defaultSocketPath()
{
    if (const char* s = std::getenv("SIMJOBS_SOCKET"); s && *s) return s;
    if (const char* r = std::getenv("XDG_RUNTIME_DIR"); r && *r) return std::string(r) + "/simjobs.sock";
    return "/tmp/simjobs-" + std::to_string(::getuid()) + ".sock";
}

std::string defaultStateDir()
{
    if (const char* s = std::getenv("XDG_STATE_HOME"); s && *s) return std::string(s) + "/simjobs";
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "/tmp") + "/.local/state/simjobs";
}

double nowUnix()
{
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

std::expected<json::Value, std::string> request(const std::string& socketPath, const json::Value& req)
{
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socketPath.size() >= sizeof addr.sun_path) return std::unexpected("socket path too long: " + socketPath);
    std::memcpy(addr.sun_path, socketPath.c_str(), socketPath.size() + 1);
    const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return errnoErr("socket");
    if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof addr) != 0) {
        const auto e = errnoErr("connect " + socketPath + " (is simjobd running?)");
        ::close(fd);
        return e;
    }
    const std::string line = json::dump(req) + "\n";
    std::size_t off = 0;
    while (off < line.size()) {
        const ssize_t n = ::send(fd, line.data() + off, line.size() - off, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            const auto e = errnoErr("send");
            ::close(fd);
            return e;
        }
        off += std::size_t(n);
    }
    std::string resp;
    char buf[1 << 16];
    for (;;) {
        const ssize_t n = ::recv(fd, buf, sizeof buf, 0);
        if (n < 0) {
            if (errno == EINTR) continue;
            const auto e = errnoErr("recv");
            ::close(fd);
            return e;
        }
        if (n == 0) break;
        resp.append(buf, std::size_t(n));
        if (resp.back() == '\n') break;
    }
    ::close(fd);
    if (resp.empty()) return std::unexpected("server closed the connection without an answer");
    return json::parse(resp);
}

} // namespace jobs
