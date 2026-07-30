// Driving the real CLI from a test: build an argv, run the real parse, then
// the real validate. Three config tests each carried a copy.
//
// The point of going through cli::parse rather than filling a RawConfig by
// hand is that `given` -- which flags the user actually typed -- is what half
// the validator's rules read, and only the parser sets it.
#pragma once

#include "dns/cli.hpp"
#include "dns/config.hpp"

#include "check.hpp"

#include <expected>
#include <string>
#include <utility>
#include <vector>

namespace dnstest {

// argv[0] is supplied; pass the subcommand as the first entry of `args`.
inline std::expected<dns::Case, dns::ConfigError>
runCli(std::vector<std::string> args)
{
    std::vector<char*> argv;
    argv.push_back(const_cast<char*>("dns"));
    for (auto& a : args) argv.push_back(a.data());
    auto raw = dns::cli::parse(int(argv.size()), argv.data());
    if (!raw)
        return std::unexpected(dns::ConfigError{1, "cli", raw.error().message});
    return dns::validate(*raw);
}

// Asserts that the case is REJECTED with `code`, and reports it as `what`.
// Returns the verdict so a caller can skip a follow-up check.
inline bool rejected(std::vector<std::string> args, int code,
                     const std::string& what)
{
    auto c = runCli(std::move(args));
    const bool ok = !c && c.error().code == code;
    check(ok, what);
    return ok;
}

} // namespace dnstest
