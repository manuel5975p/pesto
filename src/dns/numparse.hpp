// The one number parser every user-typed value goes through (CLI flags and
// the --focal box list).
#pragma once

#include <charconv>
#include <cmath>
#include <string_view>
#include <system_error>

namespace dns {

// Full-consumption from_chars: "3x", "", "+1" and " 1" are errors, never a
// silent default, and so are "nan" and "inf" -- no input takes a non-finite
// value, and a NaN slips through every `x < lo` range check downstream.
// Post: v is written only on success.
inline bool parseDouble(std::string_view s, double& v)
{
    if (s.empty()) return false;
    double d{};
    const auto [q, ec] = std::from_chars(s.data(), s.data() + s.size(), d);
    if (ec != std::errc{} || q != s.data() + s.size() || !std::isfinite(d))
        return false;
    v = d;
    return true;
}

} // namespace dns
