// One table-driven parser for every subcommand. Unknown flags, missing values
// and unparseable numbers are ERRORS -- never a silent default.
#pragma once

#include "dns/config.hpp"

#include <expected>
#include <string>
#include <vector>

namespace dns::cli {

// Already formatted for stderr; `helpRequested` distinguishes `--help` (exit 0,
// message on stdout) from a real parse failure (exit 1).
struct CliError
{
    std::string message;
    bool helpRequested = false;
};

std::expected<RawConfig, CliError> parse(int argc, char** argv);

// A run config in TOML, expanded to the exact tokens parse() takes -- the file
// is an alternative SPELLING of the command line, not a second config system:
//   subcommand = "channel"      ->  the subcommand (optional here)
//   n = 512                     ->  --n 512
//   re_tau = 180                ->  --re-tau 180   (underscores become dashes)
//   noslip = true               ->  --noslip       (false is an error: flags
//                                                   cannot be negated)
//   emit = ["stats", "png"]     ->  --emit stats --emit png
// Strictness is inherited: unknown keys surface as unknown flags in parse().
// Pure (string -> tokens); main reads the file (phase 1.5, like --scene).
struct ConfigTokens
{
    std::string subcommand;          // "" when the file does not set one
    std::vector<std::string> args;   // flag/value tokens, sorted key order
};
std::expected<ConfigTokens, std::string> configTokens(std::string_view toml);

std::string help(Subcommand);
std::string helpTop();

} // namespace dns::cli
