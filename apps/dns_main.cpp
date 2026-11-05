// dns -- the single executable. Four phases, and the phase order is the
// mechanism (masterplan §5):
//   1. parse    pure string -> RawConfig. No filesystem, no GPU.
//   1.5 inputs  the input files the validator reads (drive::readInputs).
//   2. validate pure RawConfig -> Case.   No filesystem, no GPU.
//   3. acquire  probe GPU, load and validate the mesh, then CREATE AND TRUNCATE
//               every output file, then allocate device memory.
//   4. run      the time loop, which by construction can only fail 7 or 8.
//
// This file owns argv and the phase order only: every case is the drivers'
// (dns/drive), every physics the library's.
#include "dns/cli.hpp"
#include "dns/config.hpp"
#include "dns/drive/drive.hpp"
#include "dns/interrupt.hpp"
#include "dns/scene.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

int die(int code, const std::string& msg)
{
    std::fprintf(stderr, "dns: %s\n", msg.c_str());
    return code;
}

// --config FILE.toml is an alternative SPELLING of the command line: it expands
// to the exact tokens parse() takes, spliced in before the explicit flags so the
// command line wins where an option repeats. Reading the file here keeps parse()
// pure. store owns the strings ptrs points into. Error: the exit code.
std::expected<void, int> expandConfig(int& argc, char**& argv, std::vector<std::string>& store, std::vector<char*>& ptrs)
{
    int cfg = -1;
    for (int i = 1; i < argc; ++i)
        if (std::string_view(argv[i]) == "--config") {
            cfg = i;
            break;
        }
    if (cfg < 0) return {};
    if (cfg + 1 >= argc) return std::unexpected(die(1, "--config requires a file path"));
    const std::string path = argv[cfg + 1];
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::unexpected(die(3, "cannot read config file '" + path + "'"));
    std::ostringstream ss;
    ss << f.rdbuf();
    auto ct = dns::cli::configTokens(ss.str());
    if (!ct) return std::unexpected(die(3, ct.error() + " ('" + path + "')"));
    const bool subFromCli = cfg != 1;
    const std::string sub = subFromCli ? argv[1] : ct->subcommand;
    if (sub.empty())
        return std::unexpected(die(1, "no subcommand: give one on the command line or set `subcommand` in the config file"));
    if (subFromCli && !ct->subcommand.empty() && ct->subcommand != sub)
        return std::unexpected(die(1, "config file subcommand '" + ct->subcommand + "' conflicts with '" + sub + "'"));
    store.emplace_back(argv[0]);
    store.push_back(sub);
    for (auto& a : ct->args) store.push_back(std::move(a));
    for (int i = subFromCli ? 2 : 1; i < argc; ++i)
        if (i != cfg && i != cfg + 1) store.emplace_back(argv[i]);
    ptrs.reserve(store.size());
    for (auto& s : store) ptrs.push_back(s.data());
    argc = int(ptrs.size());
    argv = ptrs.data();
    return {};
}

} // namespace

int main(int argc, char** argv)
{
    // --wall-limit counts the whole job, from here.
    dns::drive::CheckpointContext ckx;
    ckx.start = std::chrono::steady_clock::now();
    // Before anything allocates or opens a file: a stop signal must unwind
    // through the destructors, not terminate mid-step.
    dns::installInterruptHandler();
    std::vector<std::string> store;
    std::vector<char*> ptrs;
    if (auto e = expandConfig(argc, argv, store, ptrs); !e) return e.error();

    // Phase 1: parse.
    auto raw = dns::cli::parse(argc, argv);
    if (!raw) {
        std::fputs(raw.error().message.c_str(), raw.error().helpRequested ? stdout : stderr);
        return raw.error().helpRequested ? 0 : 1;
    }

    // Phase 1.5: the input files whose facts the validator reads.
    std::optional<dns::Scene> scene;
    if (auto e = dns::drive::readInputs(*raw, scene, ckx); !e) return e.error();

    // Phase 2: validate (pure: it collects advisories rather than printing them).
    auto c = dns::validate(*raw);
    if (!c) {
        std::fputs(format(c.error()).c_str(), stderr);
        return c.error().code;
    }
    for (const auto& w : c->warnings) std::fprintf(stderr, "dns: warning: %s\n", w.c_str());
    const dns::Scene* sc = scene ? &*scene : nullptr;
    if (c->dryRun) return dns::drive::dryRun(*c, sc);

    // A stop that arrived during parse/validate is honoured HERE, before phase 3
    // opens a file or touches the GPU (one during a runner's acquisition is seen
    // when its time loop starts -- docs/ui.md).
    if (dns::stopRequested()) {
        std::fprintf(stderr, "dns: %s before the run started\n", dns::stopReason());
        return 0;
    }

    // Phases 3 and 4.
    return dns::drive::run(*c, *raw, sc, ckx);
}
