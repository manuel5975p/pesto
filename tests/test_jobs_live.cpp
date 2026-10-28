// Job server end to end: the real simjobd on a private socket and state dir,
// driven over the wire and once through the simjob CLI. Covers the outcomes
// (done, failed, timeout, oom, rejected at submit and at launch, cancelled),
// multi-step stop-at-first-failure, the hard caps, and the admission rule as
// observed in start/end times. Skips (77) without a systemd user session.

#include "check.hpp"

#include "jobs/io.hpp"
#include "jobs/sha256.hpp"
#include "jobs/spec.hpp"

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#if !defined(SIMJOBD_BIN) || !defined(SIMJOB_BIN) || !defined(JOBS_HOG_BIN)
#error "SIMJOBD_BIN, SIMJOB_BIN and JOBS_HOG_BIN must be defined"
#endif

using dnstest::check;
namespace fs = std::filesystem;
using jobs::json::Array;
using jobs::json::Object;
using jobs::json::Value;

namespace {

std::string gSock;
std::string gCwd;

struct Cmd
{
    std::string exe;
    std::vector<std::string> args;
};

Value step(const Cmd& c)
{
    Array args(c.args.begin(), c.args.end());
    return Object{{"exe", c.exe}, {"sha256", *jobs::sha256File(c.exe)}, {"args", std::move(args)}};
}

Value job(const std::string& res, std::vector<Cmd> steps, int rt = 30, int mem = 256, bool bench = false)
{
    Array st;
    for (const auto& c : steps) st.push_back(step(c));
    return Object{{"label", "t"}, {"resource", res}, {"benchmark", bench}, {"max_runtime_s", rt},
                  {"max_mem_mb", mem}, {"cwd", gCwd}, {"steps", std::move(st)}};
}

Value send(const Value& req)
{
    auto r = jobs::request(gSock, req);
    return r ? *r : Value(Object{{"ok", false}, {"error", r.error()}});
}

bool ok(const Value& r) { return r.find("ok") && r.find("ok")->boolean() && *r.find("ok")->boolean(); }

long long submit(const Value& spec)
{
    const Value r = send(Object{{"op", "submit"}, {"job", spec}});
    if (!ok(r)) {
        std::printf("  submit refused: %s\n", jobs::json::dump(r).c_str());
        return -1;
    }
    return *r.find("id")->integer();
}

Value wait(long long id)
{
    const Value r = send(Object{{"op", "wait"}, {"id", id}});
    return r.find("job") ? *r.find("job") : r;
}

std::string state(const Value& j) { return j.find("state") && j.find("state")->string() ? *j.find("state")->string() : "?"; }
double num(const Value& j, const char* k) { return j.find(k) && j.find(k)->number() ? *j.find(k)->number() : -1; }

bool systemdUserScope()
{
    return std::system("systemd-run --user --scope --quiet -p MemoryMax=64M true > /dev/null 2>&1") == 0;
}

} // namespace

int main()
{
    if (!systemdUserScope()) dnstest::skip("no systemd user session (systemd-run --user --scope)");
    const fs::path root = fs::temp_directory_path() / std::format("simjobs_test_{}", ::getpid());
    fs::create_directories(root / "state");
    gSock = (root / "s.sock").string();
    gCwd = root.string();

    const pid_t server = ::fork();
    if (server == 0) {
        const std::string log = (root / "server.log").string();
        std::freopen(log.c_str(), "w", stderr);
        ::execl(SIMJOBD_BIN, SIMJOBD_BIN, "--socket", gSock.c_str(), "--state", (root / "state").c_str(), nullptr);
        ::_exit(127);
    }
    bool up = false;
    for (int i = 0; i < 100 && !up; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        up = ok(send(Object{{"op", "ping"}}));
    }
    check(up, "simjobd answers ping");
    if (!up) {
        ::kill(server, SIGKILL);
        return dnstest::finish();
    }

    const Cmd tru{"/usr/bin/true", {}}, fals{"/usr/bin/false", {}};
    const Cmd sleep1{"/usr/bin/sleep", {"1"}};

    std::printf("outcomes\n");
    {
        const Value j = wait(submit(job("cpu", {tru})));
        check(state(j) == "done" && num(j, "rc") == 0, "true: done rc 0");
        const Value f = wait(submit(job("cpu", {tru, fals, tru})));
        check(state(f) == "failed" && num(f, "rc") == 1, "true,false,true: failed rc 1");
        check(f.find("steps")->array()->size() == 2, "... stops after the failing step");
        const auto t0 = std::chrono::steady_clock::now();
        const Value t = wait(submit(job("cpu", {{"/usr/bin/sleep", {"30"}}}, 1)));
        const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        check(state(t) == "timeout", std::format("sleep 30 under 1 s: timeout ({:.1f} s)", waited));
        check(waited < 5, "... stopped promptly by SIGTERM");
        const Value m = wait(submit(job("cpu", {{JOBS_HOG_BIN, {"512"}}}, 30, 64)));
        check(state(m) == "oom", "512 MB under a 64 MB cap: oom, got " + state(m));
    }

    std::printf("refusals\n");
    {
        check(submit(job("cpu", {tru}, 1801)) < 0, "1801 s refused");
        check(submit(job("cpu", {tru}, 10, 16385)) < 0, "16385 MB refused");
        Value wrong = job("cpu", {tru});
        auto& st = std::get<Array>(std::get<Object>(wrong.v)[6].second.v)[0];
        std::get<Object>(st.v)[1].second = std::string(64, '0');
        check(submit(wrong) < 0, "wrong sha256 refused at submit");
    }

    std::printf("stale binary at launch\n");
    {
        const fs::path bin = root / "mytool";
        fs::copy_file("/usr/bin/true", bin);
        const long long blocker = submit(job("cpu", {sleep1}));
        const long long victim = submit(job("cpu", {{bin.string(), {}}}));
        fs::remove(bin); // a rebuild replaces the file while the job queues
        fs::copy_file("/usr/bin/false", bin);
        const Value v = wait(victim);
        check(state(v) == "rejected", "rebuilt between submit and launch: rejected, got " + state(v));
        wait(blocker);
    }

    std::printf("admission\n");
    {
        const long long g1 = submit(job("gpu", {sleep1}));
        const long long g2 = submit(job("gpu", {tru}));
        const long long c1 = submit(job("cpu", {sleep1}));
        const Value a = wait(g1), b = wait(g2), c = wait(c1);
        check(num(b, "started") >= num(a, "ended") - 0.01, "two gpu jobs never overlap");
        check(num(c, "started") < num(a, "ended"), "a cpu job runs beside a gpu job");

        const long long d = submit(job("cpu", {sleep1}));
        const long long e = submit(job("gpu", {tru}, 30, 256, true));
        const long long f = submit(job("gpu", {tru}));
        const Value jd = wait(d), je = wait(e), jf = wait(f);
        check(num(je, "started") >= num(jd, "ended") - 0.01, "benchmark waits for the running job");
        check(num(jf, "started") >= num(je, "ended") - 0.01, "nothing behind a benchmark overtakes it");
    }

    std::printf("cancel\n");
    {
        const long long hold = submit(job("cpu", {sleep1}));
        const long long q = submit(job("cpu", {tru}));
        check(ok(send(Object{{"op", "cancel"}, {"id", q}})), "cancel a queued job");
        check(state(wait(q)) == "cancelled", "... it ends cancelled");
        const long long r = submit(job("cpu", {{"/usr/bin/sleep", {"30"}}}));
        wait(hold);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        check(ok(send(Object{{"op", "cancel"}, {"id", r}})), "cancel a running job");
        check(state(wait(r)) == "cancelled", "... it ends cancelled");
    }

    std::printf("simjob CLI\n");
    {
        const std::string cmd = std::format("cd {} && {} --socket {} submit --cpu -t 10 -m 64 --wait -- true > /dev/null 2>&1",
                                            gCwd, SIMJOB_BIN, gSock);
        check(std::system(cmd.c_str()) == 0, "simjob submit --wait -- true exits 0");
        const std::string fcmd = std::format("cd {} && {} --socket {} submit --cpu -t 10 -m 64 --wait -- sh -c 'exit 7' > /dev/null 2>&1",
                                             gCwd, SIMJOB_BIN, gSock);
        const int rc = std::system(fcmd.c_str());
        check(WIFEXITED(rc) && WEXITSTATUS(rc) == 7, "simjob --wait mirrors a failing step's rc 7");
    }

    ::kill(server, SIGTERM);
    int status = 0;
    ::waitpid(server, &status, 0);
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "simjobd exits 0 on SIGTERM");
    check(!fs::exists(gSock), "socket removed on exit");
    if (dnstest::gFailures == 0) fs::remove_all(root);
    else std::printf("kept %s for inspection\n", root.c_str());
    return dnstest::finish();
}
