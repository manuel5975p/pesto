// Minimal strict JSON for the jobserver wire protocol and job records: RFC 8259
// values, objects keep insertion order (deterministic output), duplicate keys
// and trailing garbage are errors. Numbers are doubles (ids and limits are far
// below 2^53).
#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace jobs::json {

struct Value;
using Array = std::vector<Value>;
using Object = std::vector<std::pair<std::string, Value>>;

struct Value
{
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> v = nullptr;

    Value() = default;
    Value(std::nullptr_t) {}
    Value(bool b) : v(b) {}
    Value(double d) : v(d) {}
    Value(int i) : v(double(i)) {}
    Value(long i) : v(double(i)) {}
    Value(long long i) : v(double(i)) {}
    Value(const char* s) : v(std::string(s)) {}
    Value(std::string s) : v(std::move(s)) {}
    Value(Array a) : v(std::move(a)) {}
    Value(Object o) : v(std::move(o)) {}

    const Object* object() const { return std::get_if<Object>(&v); }
    const Array* array() const { return std::get_if<Array>(&v); }
    const std::string* string() const { return std::get_if<std::string>(&v); }
    const double* number() const { return std::get_if<double>(&v); }
    const bool* boolean() const { return std::get_if<bool>(&v); }
    bool isNull() const { return v.index() == 0; }

    // Member lookup on an object; null on a non-object or a missing key.
    const Value* find(std::string_view key) const;
    // Integral value of a number, if it is one exactly.
    std::optional<std::int64_t> integer() const;
};

// Parse one complete document. Error text names the byte offset.
std::expected<Value, std::string> parse(std::string_view text);

// Compact single-line serialization (no newline: the wire format is one
// document per line).
std::string dump(const Value& v);

} // namespace jobs::json
