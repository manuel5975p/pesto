// One table-driven parser for every subcommand. Unknown flags, missing values
// and unparseable numbers are ERRORS -- never a silent default.
#pragma once

#include "dns/config.hpp"

#include <expected>
#include <string>

namespace dns::cli {

// Already formatted for stderr; `helpRequested` distinguishes `--help` (exit 0,
// message on stdout) from a real parse failure (exit 1).
struct CliError
{
    std::string message;
    bool helpRequested = false;
};

std::expected<RawConfig, CliError> parse(int argc, char** argv);

std::string help(Subcommand);
std::string helpTop();

} // namespace dns::cli
