// The strict numeric CSV both prop input files use: one exact header line,
// then rows of exactly `ncol` comma-separated finite numbers. Blank lines are
// skipped; anything else is an error naming its line. Internal to dns/prop.
#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace dns::prop {

struct CsvRow
{
    int line = 0;              // 1-based line number in the file
    std::vector<double> v;     // exactly ncol values
};

// Pre: ncol >= 1. Whitespace around fields and a trailing '\r' are ignored.
std::expected<std::vector<CsvRow>, std::string>
csvRows(std::string_view text, std::string_view header, int ncol);

} // namespace dns::prop
