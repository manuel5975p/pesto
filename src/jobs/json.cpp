#include "jobs/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>

namespace jobs::json {

namespace {

constexpr int kMaxDepth = 64;

class Parser
{
public:
    explicit Parser(std::string_view s) : s_(s) {}

    std::expected<Value, std::string> document()
    {
        auto v = value(0);
        if (!v) return v;
        ws();
        if (i_ != s_.size()) return fail("trailing characters");
        return v;
    }

private:
    std::unexpected<std::string> fail(const std::string& what) const
    {
        return std::unexpected("json: " + what + " at byte " + std::to_string(i_));
    }

    void ws()
    {
        while (i_ < s_.size() &&
               (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r'))
            ++i_;
    }

    bool literal(std::string_view w)
    {
        if (s_.substr(i_, w.size()) != w) return false;
        i_ += w.size();
        return true;
    }

    std::expected<Value, std::string> value(int depth)
    {
        if (depth > kMaxDepth) return fail("nesting too deep");
        ws();
        if (i_ >= s_.size()) return fail("unexpected end");
        const char c = s_[i_];
        if (c == '{') return object(depth);
        if (c == '[') return array(depth);
        if (c == '"') {
            auto s = string();
            if (!s) return std::unexpected(s.error());
            return Value(std::move(*s));
        }
        if (literal("true")) return Value(true);
        if (literal("false")) return Value(false);
        if (literal("null")) return Value(nullptr);
        if (c == '-' || (c >= '0' && c <= '9')) return number();
        return fail("unexpected character");
    }

    std::expected<Value, std::string> number()
    {
        // RFC 8259 grammar: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?
        const std::size_t b = i_;
        auto digits = [&] {
            const std::size_t d = i_;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
            return i_ > d;
        };
        if (s_[i_] == '-') ++i_;
        if (i_ < s_.size() && s_[i_] == '0')
            ++i_;
        else if (!digits())
            return fail("bad number");
        if (i_ < s_.size() && s_[i_] == '.') {
            ++i_;
            if (!digits()) return fail("bad fraction");
        }
        if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
            ++i_;
            if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
            if (!digits()) return fail("bad exponent");
        }
        double d = 0;
        const auto r = std::from_chars(s_.data() + b, s_.data() + i_, d);
        if (r.ec != std::errc() || !std::isfinite(d)) return fail("number out of range");
        return Value(d);
    }

    static void utf8(std::string& out, std::uint32_t cp)
    {
        if (cp < 0x80) {
            out += char(cp);
        } else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }

    std::expected<std::uint32_t, std::string> hex4()
    {
        if (i_ + 4 > s_.size()) return fail("short \\u escape");
        std::uint32_t cp = 0;
        const auto r = std::from_chars(s_.data() + i_, s_.data() + i_ + 4, cp, 16);
        if (r.ec != std::errc() || r.ptr != s_.data() + i_ + 4) return fail("bad \\u escape");
        i_ += 4;
        return cp;
    }

    std::expected<std::string, std::string> string()
    {
        ++i_; // opening quote
        std::string out;
        while (i_ < s_.size()) {
            const char c = s_[i_++];
            if (c == '"') return out;
            if (static_cast<unsigned char>(c) < 0x20) return fail("control character in string");
            if (c != '\\') {
                out += c;
                continue;
            }
            if (i_ >= s_.size()) break;
            const char e = s_[i_++];
            switch (e) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                auto hi = hex4();
                if (!hi) return std::unexpected(hi.error());
                std::uint32_t cp = *hi;
                if (cp >= 0xD800 && cp < 0xDC00) {
                    if (!literal("\\u")) return fail("lone high surrogate");
                    auto lo = hex4();
                    if (!lo) return std::unexpected(lo.error());
                    if (*lo < 0xDC00 || *lo >= 0xE000) return fail("bad low surrogate");
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (*lo - 0xDC00);
                } else if (cp >= 0xDC00 && cp < 0xE000) {
                    return fail("lone low surrogate");
                }
                utf8(out, cp);
                break;
            }
            default: return fail("bad escape");
            }
        }
        return fail("unterminated string");
    }

    std::expected<Value, std::string> array(int depth)
    {
        ++i_;
        Array a;
        ws();
        if (i_ < s_.size() && s_[i_] == ']') {
            ++i_;
            return Value(std::move(a));
        }
        for (;;) {
            auto v = value(depth + 1);
            if (!v) return v;
            a.push_back(std::move(*v));
            ws();
            if (i_ < s_.size() && s_[i_] == ',') {
                ++i_;
                continue;
            }
            if (i_ < s_.size() && s_[i_] == ']') {
                ++i_;
                return Value(std::move(a));
            }
            return fail("expected , or ]");
        }
    }

    std::expected<Value, std::string> object(int depth)
    {
        ++i_;
        Object o;
        ws();
        if (i_ < s_.size() && s_[i_] == '}') {
            ++i_;
            return Value(std::move(o));
        }
        for (;;) {
            ws();
            if (i_ >= s_.size() || s_[i_] != '"') return fail("expected key");
            auto k = string();
            if (!k) return std::unexpected(k.error());
            for (const auto& [key, _] : o)
                if (key == *k) return fail("duplicate key \"" + *k + "\"");
            ws();
            if (i_ >= s_.size() || s_[i_] != ':') return fail("expected :");
            ++i_;
            auto v = value(depth + 1);
            if (!v) return v;
            o.emplace_back(std::move(*k), std::move(*v));
            ws();
            if (i_ < s_.size() && s_[i_] == ',') {
                ++i_;
                continue;
            }
            if (i_ < s_.size() && s_[i_] == '}') {
                ++i_;
                return Value(std::move(o));
            }
            return fail("expected , or }");
        }
    }

    std::string_view s_;
    std::size_t i_ = 0;
};

void dumpString(std::string& out, const std::string& s)
{
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                out += b;
            } else {
                out += char(c);
            }
        }
    }
    out += '"';
}

void dumpTo(std::string& out, const Value& v)
{
    if (v.isNull()) {
        out += "null";
    } else if (const bool* b = v.boolean()) {
        out += *b ? "true" : "false";
    } else if (const double* d = v.number()) {
        if (const auto i = v.integer()) {
            out += std::to_string(*i);
        } else {
            char b[32];
            const auto r = std::to_chars(b, b + sizeof b, *d);
            out.append(b, r.ptr);
        }
    } else if (const std::string* s = v.string()) {
        dumpString(out, *s);
    } else if (const Array* a = v.array()) {
        out += '[';
        for (std::size_t i = 0; i < a->size(); ++i) {
            if (i) out += ',';
            dumpTo(out, (*a)[i]);
        }
        out += ']';
    } else if (const Object* o = v.object()) {
        out += '{';
        for (std::size_t i = 0; i < o->size(); ++i) {
            if (i) out += ',';
            dumpString(out, (*o)[i].first);
            out += ':';
            dumpTo(out, (*o)[i].second);
        }
        out += '}';
    }
}

} // namespace

const Value* Value::find(std::string_view key) const
{
    const Object* o = object();
    if (!o) return nullptr;
    for (const auto& [k, val] : *o)
        if (k == key) return &val;
    return nullptr;
}

std::optional<std::int64_t> Value::integer() const
{
    const double* d = number();
    if (!d || std::trunc(*d) != *d || std::fabs(*d) > 9007199254740992.0) return std::nullopt;
    return std::int64_t(*d);
}

std::expected<Value, std::string> parse(std::string_view text)
{
    return Parser(text).document();
}

std::string dump(const Value& v)
{
    std::string out;
    dumpTo(out, v);
    return out;
}

} // namespace jobs::json
