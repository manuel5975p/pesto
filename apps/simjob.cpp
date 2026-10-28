// simjob -- client of simjobd. See docs/jobs.md.

#include "jobs/io.hpp"
#include "jobs/sha256.hpp"
#include "jobs/spec.hpp"

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

extern char** environ;

namespace {

namespace fs = std::filesystem;
using jobs::json::Value;

constexpr const char* kUsage = R"(usage: simjob [--socket PATH] COMMAND ...

  submit (--gpu|--cpu) -t SEC -m MB [--bench] [--label L] [--cwd DIR]
         [--env K=V]... [--inherit-env] [--verify FILE]... [--wait] -- EXE [ARG...]
        one-step job. EXE without a slash is looked up in PATH. sha256 of EXE
        (and of every --verify FILE) is taken NOW; the server re-checks it at
        submit and again at launch, so a rebuilt binary is rejected, never run.
        -t <= 1800 s, -m <= 16384 MB. --bench: runs alone on the machine.
        --inherit-env sends this shell's whole environment (OpenFOAM needs it).
  submit --spec FILE.json [--inherit-env] [--wait]
        multi-step job (e.g. blockMesh, snappyHexMesh, decomposePar, mpirun).
        Missing "sha256" fields and "cwd" are filled in here. Schema: docs/jobs.md.
  status ID | wait ID | cancel ID | log ID | list | ping

  wait / submit --wait exit with: 0 done, the step's rc on failure, 124
  timeout, 137 oom, 130 cancelled, 3 rejected (stale binary), 75 lost.
)";

int usage(const std::string& why)
{
    std::fprintf(stderr, "simjob: %s\n%s", why.c_str(), kUsage);
    return 2;
}

std::optional<std::string> resolveExe(const std::string& exe)
{
    if (exe.find('/') != std::string::npos) return fs::absolute(exe).lexically_normal().string();
    const char* path = std::getenv("PATH");
    std::string p = path ? path : "";
    for (std::size_t b = 0; b <= p.size();) {
        const std::size_t e = std::min(p.find(':', b), p.size());
        const std::string cand = p.substr(b, e - b) + "/" + exe;
        if (e > b && ::access(cand.c_str(), X_OK) == 0) return fs::absolute(cand).string();
        b = e + 1;
    }
    return std::nullopt;
}

std::optional<long> number(const char* s)
{
    long v = 0;
    const char* end = s + std::strlen(s);
    const auto r = std::from_chars(s, end, v);
    if (r.ec != std::errc() || r.ptr != end) return std::nullopt;
    return v;
}

void printJob(const Value& job)
{
    auto num = [&](const char* k) {
        const Value* v = job.find(k);
        return v && v->number() ? *v->number() : 0.0;
    };
    auto str = [&](const char* k) {
        const Value* v = job.find(k);
        return v && v->string() ? *v->string() : std::string();
    };
    const double wall = num("ended") > 0 && num("started") > 0 ? num("ended") - num("started") : 0;
    std::printf("job %lld %s rc=%d wall=%.1fs dir=%s\n", (long long)num("id"), str("state").c_str(),
                int(num("rc")), wall, str("dir").c_str());
    if (const std::string d = str("detail"); !d.empty()) std::printf("  %s\n", d.c_str());
}

int finalExit(const Value& job)
{
    const Value* s = job.find("state");
    const Value* rc = job.find("rc");
    const auto st = s && s->string() ? jobs::stateFrom(*s->string()) : std::nullopt;
    if (!st) return 1;
    return jobs::exitCode(*st, rc && rc->integer() ? int(*rc->integer()) : 1);
}

// Send, print a refusal, hand back the payload.
std::optional<Value> call(const std::string& sock, const Value& req)
{
    auto r = jobs::request(sock, req);
    if (!r) {
        std::fprintf(stderr, "simjob: %s\n", r.error().c_str());
        return std::nullopt;
    }
    const Value* ok = r->find("ok");
    if (!ok || !ok->boolean() || !*ok->boolean()) {
        const Value* e = r->find("error");
        std::fprintf(stderr, "simjob: refused: %s\n", e && e->string() ? e->string()->c_str() : jobs::json::dump(*r).c_str());
        return std::nullopt;
    }
    return *r;
}

int waitFor(const std::string& sock, long long id)
{
    auto r = call(sock, jobs::json::Object{{"op", "wait"}, {"id", id}});
    if (!r) return 1;
    const Value* job = r->find("job");
    printJob(*job);
    return finalExit(*job);
}

// Fill sha256 (and cwd) a spec leaves out; hashes are taken here, now.
std::optional<Value> completeSpec(Value spec)
{
    auto* o = std::get_if<jobs::json::Object>(&spec.v);
    if (!o) return std::nullopt;
    if (!spec.find("cwd")) o->emplace_back("cwd", fs::current_path().string());
    for (auto& [k, v] : *o) {
        if (k != "steps") continue;
        auto* steps = std::get_if<jobs::json::Array>(&v.v);
        if (!steps) return std::nullopt;
        for (Value& st : *steps) {
            auto* so = std::get_if<jobs::json::Object>(&st.v);
            if (!so) return std::nullopt;
            const Value* exe = st.find("exe");
            if (!exe || !exe->string()) return std::nullopt;
            const auto abs = resolveExe(*exe->string());
            if (!abs) {
                std::fprintf(stderr, "simjob: %s not found\n", exe->string()->c_str());
                return std::nullopt;
            }
            for (auto& [sk, sv] : *so)
                if (sk == "exe") sv = *abs;
            if (!st.find("sha256")) {
                auto h = jobs::sha256File(*abs);
                if (!h) {
                    std::fprintf(stderr, "simjob: %s\n", h.error().c_str());
                    return std::nullopt;
                }
                so->emplace_back("sha256", *h);
            }
            for (auto& [sk, sv] : *so) {
                if (sk != "verify") continue;
                auto* va = std::get_if<jobs::json::Array>(&sv.v);
                if (!va) return std::nullopt;
                for (Value& c : *va) {
                    auto* co = std::get_if<jobs::json::Object>(&c.v);
                    const Value* p = c.find("path");
                    if (!co || !p || !p->string() || c.find("sha256")) continue;
                    auto h = jobs::sha256File(*p->string());
                    if (!h) {
                        std::fprintf(stderr, "simjob: %s\n", h.error().c_str());
                        return std::nullopt;
                    }
                    co->emplace_back("sha256", *h);
                }
            }
        }
    }
    return spec;
}

// This shell's environment with `extra` on top (extra wins).
jobs::json::Object inheritedEnv(const jobs::json::Object& extra)
{
    jobs::json::Object all;
    for (char** e = environ; *e; ++e) {
        const std::string kv = *e;
        const auto eq = kv.find('=');
        if (eq != std::string::npos && eq > 0) all.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    }
    for (const auto& kv : extra) {
        std::erase_if(all, [&](const auto& x) { return x.first == kv.first; });
        all.push_back(kv);
    }
    return all;
}

int submit(const std::string& sock, int argc, char** argv, int i)
{
    std::optional<std::string> res, specFile, label, cwd;
    std::optional<long> t, m;
    bool bench = false, wait = false, inherit = false;
    jobs::json::Object env;
    std::vector<std::string> verify;
    for (; i < argc; ++i) {
        const std::string a = argv[i];
        const bool v = i + 1 < argc;
        if (a == "--") {
            ++i;
            break;
        }
        if (a == "--gpu" || a == "--cpu") res = a.substr(2);
        else if (a == "--bench") bench = true;
        else if (a == "--wait") wait = true;
        else if (a == "--inherit-env") inherit = true;
        else if (a == "-t" && v) t = number(argv[++i]);
        else if (a == "-m" && v) m = number(argv[++i]);
        else if (a == "--label" && v) label = argv[++i];
        else if (a == "--cwd" && v) cwd = fs::absolute(argv[++i]).string();
        else if (a == "--spec" && v) specFile = argv[++i];
        else if (a == "--verify" && v) verify.push_back(fs::absolute(argv[++i]).string());
        else if (a == "--env" && v) {
            const std::string kv = argv[++i];
            const auto eq = kv.find('=');
            if (eq == std::string::npos || eq == 0) return usage("--env wants K=V");
            env.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
        } else
            return usage("bad submit argument \"" + a + "\"");
    }
    Value spec;
    if (specFile) {
        if (i < argc || res || t || m) return usage("--spec takes the whole job from the file");
        auto text = jobs::readFile(*specFile);
        if (!text) return usage(text.error());
        auto v = jobs::json::parse(*text);
        if (!v) return usage(*specFile + ": " + v.error());
        auto full = completeSpec(std::move(*v));
        if (!full) return usage(*specFile + ": malformed job (see docs/jobs.md)");
        spec = std::move(*full);
        if (inherit) {
            auto& o = std::get<jobs::json::Object>(spec.v);
            jobs::json::Object own;
            for (const auto& [k, val] : o)
                if (k == "env" && val.object()) own = *val.object();
            std::erase_if(o, [](const auto& kv) { return kv.first == "env"; });
            o.emplace_back("env", inheritedEnv(own));
        }
    } else {
        if (!res) return usage("submit needs --gpu or --cpu");
        if (!t || !m) return usage("submit needs -t SEC and -m MB");
        if (i >= argc) return usage("submit needs -- EXE [ARG...]");
        const auto exe = resolveExe(argv[i]);
        if (!exe) return usage(std::string(argv[i]) + " not found");
        auto h = jobs::sha256File(*exe);
        if (!h) return usage(h.error());
        jobs::json::Array args;
        for (int k = i + 1; k < argc; ++k) args.emplace_back(argv[k]);
        jobs::json::Object step{{"exe", *exe}, {"sha256", *h}, {"args", std::move(args)}};
        if (!verify.empty()) {
            jobs::json::Array va;
            for (const auto& p : verify) {
                auto hv = jobs::sha256File(p);
                if (!hv) return usage(hv.error());
                va.push_back(jobs::json::Object{{"path", p}, {"sha256", *hv}});
            }
            step.emplace_back("verify", std::move(va));
        }
        if (inherit) env = inheritedEnv(env);
        jobs::json::Object o{{"label", label.value_or(fs::path(*exe).filename().string())},
                             {"resource", *res},
                             {"benchmark", bench},
                             {"max_runtime_s", *t},
                             {"max_mem_mb", *m},
                             {"cwd", cwd.value_or(fs::current_path().string())}};
        if (!env.empty()) o.emplace_back("env", std::move(env));
        o.emplace_back("steps", jobs::json::Array{std::move(step)});
        spec = std::move(o);
    }
    auto r = call(sock, jobs::json::Object{{"op", "submit"}, {"job", std::move(spec)}});
    if (!r) return 1;
    const long long id = *r->find("id")->integer();
    if (!wait) {
        std::printf("%lld\n", id);
        return 0;
    }
    std::fprintf(stderr, "simjob: job %lld queued, waiting\n", id);
    return waitFor(sock, id);
}

} // namespace

int main(int argc, char** argv)
{
    std::string sock = jobs::defaultSocketPath();
    int i = 1;
    if (i + 1 < argc && std::string(argv[i]) == "--socket") {
        sock = argv[i + 1];
        i += 2;
    }
    if (i >= argc) return usage("missing command");
    const std::string cmd = argv[i++];
    if (cmd == "--help" || cmd == "-h" || cmd == "help") {
        std::fputs(kUsage, stdout);
        return 0;
    }
    if (cmd == "submit") return submit(sock, argc, argv, i);
    if (cmd == "ping") return call(sock, jobs::json::Object{{"op", "ping"}}) ? 0 : 1;
    if (cmd == "list") {
        auto r = call(sock, jobs::json::Object{{"op", "list"}});
        if (!r) return 1;
        std::printf("%6s  %-9s %-4s %-5s %5s %8s  %s\n", "id", "state", "res", "bench", "rc", "wall", "label");
        for (const Value& j : *r->find("jobs")->array()) {
            auto num = [&](const char* k) { return *j.find(k)->number(); };
            const double wall = num("ended") > 0 ? num("ended") - num("started")
                                : num("started") > 0 ? jobs::nowUnix() - num("started") : 0;
            std::printf("%6lld  %-9s %-4s %-5s %5d %7.1fs  %s\n", (long long)num("id"), j.find("state")->string()->c_str(),
                        j.find("resource")->string()->c_str(), *j.find("benchmark")->boolean() ? "yes" : "",
                        int(num("rc")), wall, j.find("label")->string()->c_str());
        }
        return 0;
    }
    if (i >= argc) return usage(cmd + " needs a job id");
    const auto id = number(argv[i]);
    if (!id) return usage("bad job id");
    if (cmd == "wait") return waitFor(sock, *id);
    if (cmd == "cancel") return call(sock, jobs::json::Object{{"op", "cancel"}, {"id", (long long)*id}}) ? 0 : 1;
    if (cmd == "status" || cmd == "log") {
        auto r = call(sock, jobs::json::Object{{"op", "status"}, {"id", (long long)*id}});
        if (!r) return 1;
        const Value* job = r->find("job");
        if (cmd == "status") {
            std::printf("%s\n", jobs::json::dump(*job).c_str());
            return 0;
        }
        auto text = jobs::readFile(*job->find("dir")->string() + "/output.log");
        if (!text) {
            std::fprintf(stderr, "simjob: %s\n", text.error().c_str());
            return 1;
        }
        std::fwrite(text->data(), 1, text->size(), stdout);
        return 0;
    }
    return usage("unknown command \"" + cmd + "\"");
}
