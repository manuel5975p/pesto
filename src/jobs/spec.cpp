#include "jobs/spec.hpp"

#include "jobs/sha256.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <format>

#include <sys/stat.h>
#include <unistd.h>

namespace jobs {

namespace {

using json::Value;

std::unexpected<std::string> err(std::string s) { return std::unexpected(std::move(s)); }

// Strict member access on one JSON object: every key read is marked, and
// `unknown()` names the first key nobody asked for.
class Fields
{
public:
    Fields(const Value& v, std::string_view what) : what_(what)
    {
        o_ = v.object();
    }
    bool ok() const { return o_ != nullptr; }

    const Value* get(std::string_view k)
    {
        seen_.emplace_back(k);
        return o_ ? jsonFind(k) : nullptr;
    }

    std::expected<std::string, std::string> str(std::string_view k, bool required)
    {
        const Value* v = get(k);
        if (!v) {
            if (required) return err(std::format("{}: missing \"{}\"", what_, k));
            return std::string();
        }
        if (!v->string()) return err(std::format("{}: \"{}\" must be a string", what_, k));
        return *v->string();
    }

    std::expected<std::int64_t, std::string> integer(std::string_view k)
    {
        const Value* v = get(k);
        if (!v) return err(std::format("{}: missing \"{}\"", what_, k));
        const auto i = v->integer();
        if (!i) return err(std::format("{}: \"{}\" must be an integer", what_, k));
        return *i;
    }

    std::expected<std::string, std::string> unknown() const
    {
        for (const auto& [k, _] : *o_)
            if (std::ranges::find(seen_, k) == seen_.end())
                return err(std::format("{}: unknown key \"{}\"", what_, k));
        return std::string();
    }

private:
    const Value* jsonFind(std::string_view k) const
    {
        for (const auto& [key, val] : *o_)
            if (key == k) return &val;
        return nullptr;
    }

    const json::Object* o_ = nullptr;
    std::string what_;
    std::vector<std::string> seen_;
};

bool isHex64(const std::string& s)
{
    return s.size() == 64 &&
           std::ranges::all_of(s, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

std::expected<std::vector<std::string>, std::string> stringList(const Value* v, std::string_view what)
{
    std::vector<std::string> out;
    if (!v) return out;
    const json::Array* a = v->array();
    if (!a) return err(std::format("{} must be an array of strings", what));
    for (const Value& x : *a) {
        if (!x.string()) return err(std::format("{} must be an array of strings", what));
        out.push_back(*x.string());
    }
    return out;
}

std::expected<FileCheck, std::string> parseCheck(const Value& v, std::string_view what)
{
    Fields f(v, what);
    if (!f.ok()) return err(std::format("{} must be an object", what));
    auto path = f.str("path", true);
    if (!path) return err(path.error());
    auto sha = f.str("sha256", true);
    if (!sha) return err(sha.error());
    if (auto u = f.unknown(); !u) return err(u.error());
    if (path->empty() || (*path)[0] != '/') return err(std::format("{}: path must be absolute", what));
    if (!isHex64(*sha)) return err(std::format("{}: sha256 must be 64 lower-case hex digits", what));
    return FileCheck{*path, *sha};
}

std::expected<Step, std::string> parseStep(const Value& v, std::size_t n)
{
    const std::string what = std::format("steps[{}]", n);
    Fields f(v, what);
    if (!f.ok()) return err(what + " must be an object");
    Step s;
    auto exe = f.str("exe", true);
    if (!exe) return err(exe.error());
    auto sha = f.str("sha256", true);
    if (!sha) return err(sha.error());
    auto args = stringList(f.get("args"), what + ".args");
    if (!args) return err(args.error());
    s.exe = *exe;
    s.sha256 = *sha;
    s.args = std::move(*args);
    if (const Value* ver = f.get("verify")) {
        const json::Array* a = ver->array();
        if (!a) return err(what + ".verify must be an array");
        for (std::size_t i = 0; i < a->size(); ++i) {
            auto c = parseCheck((*a)[i], std::format("{}.verify[{}]", what, i));
            if (!c) return err(c.error());
            s.verify.push_back(std::move(*c));
        }
    }
    if (auto u = f.unknown(); !u) return err(u.error());
    if (s.exe.empty() || s.exe[0] != '/') return err(what + ": exe must be an absolute path");
    if (!isHex64(s.sha256)) return err(what + ": sha256 must be 64 lower-case hex digits");
    return s;
}

std::string errnoText(const std::string& what) { return what + ": " + std::strerror(errno); }

constexpr std::array<std::pair<State, std::string_view>, 9> kStates = {{
    {State::queued, "queued"},     {State::running, "running"}, {State::done, "done"},
    {State::failed, "failed"},     {State::timeout, "timeout"}, {State::oom, "oom"},
    {State::cancelled, "cancelled"}, {State::rejected, "rejected"}, {State::lost, "lost"},
}};

} // namespace

std::expected<JobSpec, std::string> parseSpec(const Value& v)
{
    Fields f(v, "job");
    if (!f.ok()) return err("job must be an object");
    JobSpec s;
    auto label = f.str("label", false);
    if (!label) return err(label.error());
    s.label = *label;
    auto res = f.str("resource", true);
    if (!res) return err(res.error());
    if (*res == "gpu")
        s.resource = Resource::gpu;
    else if (*res == "cpu")
        s.resource = Resource::cpu;
    else
        return err("job: resource must be \"gpu\" or \"cpu\"");
    if (const Value* b = f.get("benchmark")) {
        if (!b->boolean()) return err("job: benchmark must be true or false");
        s.benchmark = *b->boolean();
    }
    auto rt = f.integer("max_runtime_s");
    if (!rt) return err(rt.error());
    if (*rt < 1 || *rt > kMaxRuntimeS)
        return err(std::format("job: max_runtime_s must be in [1, {}] (30 min), got {}", kMaxRuntimeS, *rt));
    s.maxRuntimeS = int(*rt);
    auto mem = f.integer("max_mem_mb");
    if (!mem) return err(mem.error());
    if (*mem < kMinMemMb || *mem > kMaxMemMb)
        return err(std::format("job: max_mem_mb must be in [{}, {}] (16 GiB), got {}", kMinMemMb, kMaxMemMb, *mem));
    s.maxMemMb = *mem;
    auto cwd = f.str("cwd", true);
    if (!cwd) return err(cwd.error());
    if (cwd->empty() || (*cwd)[0] != '/') return err("job: cwd must be an absolute path");
    s.cwd = *cwd;
    if (const Value* e = f.get("env")) {
        const json::Object* o = e->object();
        if (!o) return err("job: env must be an object of strings");
        for (const auto& [k, val] : *o) {
            if (!val.string()) return err(std::format("job: env.{} must be a string", k));
            if (k.empty() || k.find('=') != std::string::npos)
                return err(std::format("job: bad env name \"{}\"", k));
            s.env.emplace_back(k, *val.string());
        }
    }
    const Value* st = f.get("steps");
    if (!st || !st->array() || st->array()->empty())
        return err("job: steps must be a non-empty array");
    if (st->array()->size() > kMaxSteps) return err(std::format("job: at most {} steps", kMaxSteps));
    for (std::size_t i = 0; i < st->array()->size(); ++i) {
        auto step = parseStep((*st->array())[i], i);
        if (!step) return err(step.error());
        s.steps.push_back(std::move(*step));
    }
    if (auto u = f.unknown(); !u) return err(u.error());
    return s;
}

json::Value toJson(const JobSpec& s)
{
    json::Object env;
    for (const auto& [k, v] : s.env) env.emplace_back(k, v);
    json::Array steps;
    for (const Step& st : s.steps) {
        json::Array args(st.args.begin(), st.args.end());
        json::Array ver;
        for (const FileCheck& c : st.verify) ver.push_back(json::Object{{"path", c.path}, {"sha256", c.sha256}});
        json::Object o{{"exe", st.exe}, {"sha256", st.sha256}, {"args", std::move(args)}};
        if (!ver.empty()) o.emplace_back("verify", std::move(ver));
        steps.push_back(std::move(o));
    }
    json::Object o{{"label", s.label},
                   {"resource", s.resource == Resource::gpu ? "gpu" : "cpu"},
                   {"benchmark", s.benchmark},
                   {"max_runtime_s", s.maxRuntimeS},
                   {"max_mem_mb", (long long)s.maxMemMb},
                   {"cwd", s.cwd}};
    if (!env.empty()) o.emplace_back("env", std::move(env));
    o.emplace_back("steps", std::move(steps));
    return o;
}

std::expected<void, std::string> checkFiles(const JobSpec& s)
{
    struct stat st{};
    if (::stat(s.cwd.c_str(), &st) != 0) return err(errnoText("cwd " + s.cwd));
    if (!S_ISDIR(st.st_mode)) return err("cwd " + s.cwd + " is not a directory");
    for (std::size_t i = 0; i < s.steps.size(); ++i) {
        const Step& step = s.steps[i];
        if (::stat(step.exe.c_str(), &st) != 0) return err(errnoText(std::format("steps[{}] exe {}", i, step.exe)));
        if (!S_ISREG(st.st_mode) || ::access(step.exe.c_str(), X_OK) != 0)
            return err(std::format("steps[{}] exe {} is not an executable file", i, step.exe));
        auto h = sha256File(step.exe);
        if (!h) return err(h.error());
        if (*h != step.sha256)
            return err(std::format("steps[{}] exe {}: checksum mismatch (stale or rebuilt binary): have {}, spec {}",
                                   i, step.exe, *h, step.sha256));
        for (const FileCheck& c : step.verify) {
            auto hv = sha256File(c.path);
            if (!hv) return err(hv.error());
            if (*hv != c.sha256)
                return err(std::format("steps[{}] verify {}: checksum mismatch: have {}, spec {}", i, c.path, *hv, c.sha256));
        }
    }
    return {};
}

std::string_view name(State s)
{
    for (const auto& [st, n] : kStates)
        if (st == s) return n;
    return "?";
}

std::optional<State> stateFrom(std::string_view s)
{
    for (const auto& [st, n] : kStates)
        if (n == s) return st;
    return std::nullopt;
}

int exitCode(State s, int rc)
{
    switch (s) {
    case State::done: return 0;
    case State::failed: return std::clamp(rc, 1, 255);
    case State::timeout: return 124;
    case State::oom: return 137;
    case State::cancelled: return 130;
    case State::rejected: return 3;
    case State::lost: return 75;
    case State::queued:
    case State::running: break;
    }
    return 1;
}

json::Value toJson(const StepResult& r)
{
    json::Object o{{"step", r.step}, {"state", std::string(name(r.state))}, {"rc", r.rc}, {"wall_s", r.wallS}};
    if (!r.error.empty()) o.emplace_back("error", r.error);
    return o;
}

std::expected<StepResult, std::string> parseStepResult(const Value& v)
{
    StepResult r;
    const Value* step = v.find("step");
    const Value* state = v.find("state");
    if (!step || !step->integer() || !state || !state->string()) return err("bad step result");
    r.step = int(*step->integer());
    const auto s = stateFrom(*state->string());
    if (!s) return err("bad step state");
    r.state = *s;
    if (const Value* rc = v.find("rc"); rc && rc->integer()) r.rc = int(*rc->integer());
    if (const Value* w = v.find("wall_s"); w && w->number()) r.wallS = *w->number();
    if (const Value* e = v.find("error"); e && e->string()) r.error = *e->string();
    return r;
}

json::Value toJson(const Job& j)
{
    json::Array steps;
    for (const StepResult& r : j.steps) steps.push_back(toJson(r));
    return json::Object{{"id", (long long)j.id},
                        {"state", std::string(name(j.state))},
                        {"submitted", j.submitted},
                        {"started", j.started},
                        {"ended", j.ended},
                        {"rc", j.rc},
                        {"detail", j.detail},
                        {"dir", j.dir},
                        {"steps", std::move(steps)},
                        {"spec", toJson(j.spec)}};
}

std::expected<Job, std::string> parseJob(const Value& v)
{
    Job j;
    const Value* id = v.find("id");
    const Value* state = v.find("state");
    const Value* spec = v.find("spec");
    if (!id || !id->integer() || !state || !state->string() || !spec) return err("bad job record");
    j.id = *id->integer();
    const auto s = stateFrom(*state->string());
    if (!s) return err("bad job state");
    j.state = *s;
    auto sp = parseSpec(*spec);
    if (!sp) return err(sp.error());
    j.spec = std::move(*sp);
    auto num = [&](const char* k) {
        const Value* x = v.find(k);
        return x && x->number() ? *x->number() : 0.0;
    };
    j.submitted = num("submitted");
    j.started = num("started");
    j.ended = num("ended");
    j.rc = int(num("rc"));
    if (const Value* d = v.find("detail"); d && d->string()) j.detail = *d->string();
    if (const Value* d = v.find("dir"); d && d->string()) j.dir = *d->string();
    if (const Value* st = v.find("steps"); st && st->array())
        for (const Value& x : *st->array())
            if (auto r = parseStepResult(x)) j.steps.push_back(*r);
    return j;
}

} // namespace jobs
