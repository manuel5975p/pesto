#include "dns/prop/csv.hpp"

#include "dns/numparse.hpp"

#include <format>
#include <ranges>

namespace dns::prop {
namespace {

std::string_view trim(std::string_view s)
{
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string_view::npos) return {};
    const auto e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

// The header with whitespace dropped around each name.
std::string normalizedHeader(std::string_view line)
{
    std::string out;
    for (auto part : line | std::views::split(','))
    {
        if (!out.empty()) out += ',';
        out += trim(std::string_view(part.begin(), part.end()));
    }
    return out;
}

} // namespace

std::expected<std::vector<CsvRow>, std::string>
csvRows(std::string_view text, std::string_view header, int ncol)
{
    std::vector<CsvRow> rows;
    bool seenHeader = false;
    int lineNo = 0;
    for (auto lineRange : text | std::views::split('\n'))
    {
        ++lineNo;
        const std::string_view line = trim(std::string_view(lineRange.begin(), lineRange.end()));
        if (line.empty()) continue;
        if (!seenHeader)
        {
            if (normalizedHeader(line) != header)
                return std::unexpected(std::format(
                    "line {}: header must be '{}', got '{}'", lineNo, header, line));
            seenHeader = true;
            continue;
        }
        CsvRow r{.line = lineNo, .v = {}};
        for (auto part : line | std::views::split(','))
        {
            const std::string_view f = trim(std::string_view(part.begin(), part.end()));
            double d = 0;
            if (!parseDouble(f, d))
                return std::unexpected(std::format(
                    "line {}: '{}' is not a finite number", lineNo, f));
            r.v.push_back(d);
        }
        if (int(r.v.size()) != ncol)
            return std::unexpected(std::format("line {}: expected {} columns, got {}",
                                               lineNo, ncol, r.v.size()));
        rows.push_back(std::move(r));
    }
    if (!seenHeader)
        return std::unexpected(std::format("empty file (need the header '{}')", header));
    return rows;
}

} // namespace dns::prop
