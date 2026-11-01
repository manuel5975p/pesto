// The checkpoint/restart byte rule driven through the dns binary (S2's
// gpu_restart_flame and S4's gpu_lowmach_restart): run, compare every
// artefact but the .json, chain a run through --wall-limit 0 checkpoints.
// Needs DNS_BIN (the dns binary's path) defined before inclusion.
#pragma once

#include "check.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <sys/wait.h>

#ifndef DNS_BIN
#error "DNS_BIN must be defined (path to the dns binary)"
#endif

namespace dnstest::restart {

namespace fs = std::filesystem;

inline std::string slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream s;
    s << f.rdbuf();
    return s.str();
}

// Runs `dns ARGS`, stdout + stderr into `log`; the exit code.
inline int runDns(const std::string& args, const std::string& log)
{
    const std::string cmd = "\"" DNS_BIN "\" " + args + " > \"" + log + "\" 2>&1";
    const int rc = std::system(cmd.c_str());
    if (rc == -1 || !WIFEXITED(rc)) return -1;
    return WEXITSTATUS(rc);
}

// The artefacts of prefix `r` in dir: every file but the .json, the
// checkpoint and the logs.
inline std::set<std::string> artefacts(const std::string& dir)
{
    std::set<std::string> s;
    for (const auto& e : fs::recursive_directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string rel = fs::relative(e.path(), dir).string();
        if (rel.ends_with(".json") || rel.ends_with(".log") ||
            rel.find("_ckpt.bin") != std::string::npos)
            continue;
        s.insert(rel);
    }
    return s;
}

// Every artefact of a byte-identical to b's, same set.
inline bool sameOutputs(const std::string& a, const std::string& b,
                        const std::string& name)
{
    const auto fa = artefacts(a), fb = artefacts(b);
    bool ok = fa == fb && !fa.empty();
    if (fa != fb)
        std::printf("    %s: file sets differ (%zu vs %zu)\n", name.c_str(),
                    fa.size(), fb.size());
    int n = 0;
    for (const auto& f : fa) {
        if (!fb.contains(f)) continue;
        if (slurp(a + "/" + f) != slurp(b + "/" + f)) {
            std::printf("    %s: DIFF %s\n", name.c_str(), f.c_str());
            ok = false;
        }
        ++n;
    }
    std::printf("    %s: %d files compared\n", name.c_str(), n);
    return ok;
}

inline bool jsonHas(const std::string& prefix, const std::string& what)
{
    return slurp(prefix + ".json").find(what) != std::string::npos;
}

// Appends junk to every series and frame file of dir (a killed job's rows
// past its checkpoint).
inline void appendJunk(const std::string& dir)
{
    for (const auto& f : artefacts(dir)) {
        if (!(f.ends_with(".csv") || f.ends_with(".bin")) || f.ends_with("_mask.bin"))
            continue;
        std::ofstream o(dir + "/" + f, std::ios::binary | std::ios::app);
        o << "9.99,junk,rows\n";
    }
}

// Straight run vs the chained one (--wall-limit 0 at every checkpoint) under
// root/name. `between(job)` runs before restart `job` (1..). Returns the
// chain's job count (0 on a failed job).
template <class Between>
int chained(const std::string& root, const std::string& name,
            const std::string& args, double every, int wantJobs, Between between)
{
    const std::string a = root + "/" + name + "/straight";
    const std::string b = root + "/" + name + "/chain";
    fs::create_directories(a);
    fs::create_directories(b);
    const int rc0 = runDns(args + " --quiet --out " + a + "/r", a + "/run.log");
    check(rc0 == 0, name + ": straight run exits 0");
    const std::string ck = std::format(" --checkpoint-dt {} --wall-limit 0", every);
    int jobs = 0;
    for (; jobs < wantJobs + 2; ++jobs) {
        if (jobs > 0) between(jobs);
        const std::string rs = jobs > 0 ? " --restart " + b + "/r" : "";
        const int rc = runDns(args + ck + rs + " --quiet --out " + b + "/r",
                              b + std::format("/job{}.log", jobs));
        if (rc != 0) {
            check(false, std::format("{}: job {} exits 0 (rc {})", name, jobs, rc));
            return 0;
        }
        if (jsonHas(b + "/r", "\"complete\":true")) {
            ++jobs;
            break;
        }
    }
    check(jobs == wantJobs,
          std::format("{}: {} jobs (1 + {} restarts), got {}", name, wantJobs,
                      wantJobs - 1, jobs));
    check(jsonHas(b + "/r", std::format("\"complete\":true,\"restarts\":{}",
                                        wantJobs - 1)),
          name + ": .json complete true, restarts " + std::to_string(wantJobs - 1));
    check(rc0 == 0 && sameOutputs(a, b, name),
          name + ": chained outputs byte-identical to the straight run");
    return jobs;
}

inline int chained(const std::string& root, const std::string& name,
                   const std::string& args, double every, int wantJobs)
{
    return chained(root, name, args, every, wantJobs, [](int) {});
}

} // namespace dnstest::restart
