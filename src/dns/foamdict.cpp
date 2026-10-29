#include "dns/foamdict.hpp"

#include <charconv>
#include <cmath>
#include <format>

namespace dns::foam {
namespace {

bool isWs(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
bool isPunct(char c) { return c == '(' || c == ')' || c == '{' || c == '}' || c == ';' || c == '[' || c == ']'; }

struct Lexer
{
    std::string_view t;
    std::string file;
    std::size_t p = 0;
    int line = 1;

    std::string err(int ln, const std::string& what) const { return std::format("{}:{}: {}", file, ln, what); }

    // Skips whitespace and comments. Error: an unterminated block comment.
    std::expected<void, std::string> skip()
    {
        while (p < t.size()) {
            const char c = t[p];
            if (c == '\n') { ++line; ++p; continue; }
            if (isWs(c)) { ++p; continue; }
            if (c == '/' && p + 1 < t.size() && t[p + 1] == '/') {
                while (p < t.size() && t[p] != '\n') ++p;
                continue;
            }
            if (c == '/' && p + 1 < t.size() && t[p + 1] == '*') {
                const int l0 = line;
                p += 2;
                while (p + 1 < t.size() && !(t[p] == '*' && t[p + 1] == '/')) {
                    if (t[p] == '\n') ++line;
                    ++p;
                }
                if (p + 1 >= t.size()) return std::unexpected(err(l0, "unterminated /* comment"));
                p += 2;
                continue;
            }
            break;
        }
        return {};
    }

    // Next token, or nullopt at the end of the text.
    std::expected<std::optional<Token>, std::string> next()
    {
        if (auto s = skip(); !s) return std::unexpected(s.error());
        if (p >= t.size()) return std::optional<Token>{};
        Token k;
        k.line = line;
        const char c = t[p];
        if (isPunct(c)) {
            k.kind = Token::Punct;
            k.text = t.substr(p++, 1);
            return k;
        }
        if (c == '"') {
            const std::size_t b = ++p;
            while (p < t.size() && t[p] != '"') {
                if (t[p] == '\\' && p + 1 < t.size()) ++p;
                if (t[p] == '\n') ++line;
                ++p;
            }
            if (p >= t.size()) return std::unexpected(err(k.line, "unterminated string"));
            k.kind = Token::String;
            k.text = t.substr(b, p - b);
            ++p;
            return k;
        }
        // A word: balanced parentheses inside it belong to it (OpenFOAM's
        // readWordToken), an unbalanced ')' or any other punctuation ends it.
        const std::size_t b = p;
        int depth = 0;
        while (p < t.size()) {
            const char d = t[p];
            if (isWs(d) || d == '"') break;
            if (d == '(') { ++depth; ++p; continue; }
            if (d == ')') {
                if (depth == 0) break;
                --depth;
                ++p;
                continue;
            }
            if (isPunct(d)) break;
            ++p;
        }
        k.text = t.substr(b, p - b);
        const char f = k.text.front();
        if ((f >= '0' && f <= '9') || f == '-' || f == '+' || f == '.') {
            double v{};
            const char* s0 = k.text.data();
            if (f == '+') ++s0;
            const auto r = std::from_chars(s0, k.text.data() + k.text.size(), v);
            if (r.ec == std::errc{} && r.ptr == k.text.data() + k.text.size() && std::isfinite(v)) {
                k.kind = Token::Number;
                k.num = v;
                return k;
            }
        }
        k.kind = Token::Word;
        return k;
    }
};

std::expected<void, std::string> parseBody(Lexer& L, Dict& d, Dict& out, bool top, int openLine)
{
    while (true) {
        auto tk = L.next();
        if (!tk) return std::unexpected(tk.error());
        if (!*tk) {
            if (top) return {};
            return std::unexpected(L.err(openLine, "unclosed '{'"));
        }
        Token k = **tk;
        if (k.is('}')) {
            if (top) return std::unexpected(L.err(k.line, "unexpected '}'"));
            return {};
        }
        if (k.is(';')) continue; // a stray ';' (OpenFOAM tolerates it)
        if (k.kind == Token::Punct)
            return std::unexpected(L.err(k.line, std::format("expected a keyword, got '{}'", k.text)));
        Entry e;
        e.line = k.line;
        e.key = std::string(k.text);
        e.quotedKey = k.kind == Token::String;
        if (k.kind == Token::Word && k.text.front() == '#') {
            // Directive: its arguments run to the end of the line.
            e.directive = true;
            while (true) {
                const std::size_t save = L.p;
                const int saveLine = L.line;
                auto a = L.next();
                if (!a) return std::unexpected(a.error());
                if (!*a || (*a)->line != e.line) {
                    L.p = save;
                    L.line = saveLine;
                    break;
                }
                e.value.push_back(**a);
            }
            out.entries.push_back(std::move(e));
            continue;
        }
        auto v = L.next();
        if (!v) return std::unexpected(v.error());
        if (!*v) return std::unexpected(L.err(e.line, "keyword '" + e.key + "' has no value"));
        if ((*v)->is(';')) { // `key;`: an entry without a value (e.g. cache { grad(U); })
            out.entries.push_back(std::move(e));
            continue;
        }
        if ((*v)->is('{')) {
            auto sub = std::make_shared<Dict>();
            sub->src = d.src;
            sub->file = d.file;
            if (auto r = parseBody(L, d, *sub, false, (*v)->line); !r) return std::unexpected(r.error());
            e.dict = std::move(sub);
            out.entries.push_back(std::move(e));
            continue;
        }
        // Value tokens up to the ';' at bracket depth 0.
        int depth = 0;
        Token cur = **v;
        while (true) {
            if (cur.is('(') || cur.is('[')) ++depth;
            if (cur.is(')') || cur.is(']')) {
                if (--depth < 0) return std::unexpected(L.err(cur.line, "unbalanced '" + std::string(cur.text) + "'"));
            }
            if (cur.is('{') || cur.is('}'))
                return std::unexpected(L.err(cur.line, "keyword '" + e.key + "': missing ';' before '" +
                                                       std::string(cur.text) + "'"));
            if (cur.is(';') && depth == 0) break;
            e.value.push_back(cur);
            auto n = L.next();
            if (!n) return std::unexpected(n.error());
            if (!*n) return std::unexpected(L.err(e.line, "keyword '" + e.key + "': missing ';'"));
            cur = **n;
        }
        out.entries.push_back(std::move(e));
    }
}

std::string keyErr(const Dict& d, const Entry& e, const std::string& what)
{
    return d.where(e.line) + ": '" + e.key + "' " + what;
}

} // namespace

const Entry* Dict::find(std::string_view key) const
{
    const Entry* hit = nullptr;
    for (const Entry& e : entries)
        if (!e.quotedKey && !e.directive && e.key == key) hit = &e;
    return hit;
}

std::string Dict::where(int line) const { return std::format("{}:{}", file, line); }

std::expected<Dict, std::string> parseDict(std::string text, std::string file)
{
    Dict d;
    d.src = std::make_shared<const std::string>(std::move(text));
    d.file = std::move(file);
    Lexer L{*d.src, d.file};
    if (auto r = parseBody(L, d, d, true, 0); !r) return std::unexpected(r.error());
    return d;
}

std::expected<double, std::string> scalarOf(const Dict& d, const Entry& e)
{
    if (e.dict || e.value.size() != 1 || e.value[0].kind != Token::Number)
        return std::unexpected(keyErr(d, e, "expects one number, got '" + joined(e) + "'"));
    return e.value[0].num;
}

std::expected<std::string, std::string> wordOf(const Dict& d, const Entry& e)
{
    if (e.dict || e.value.size() != 1 || (e.value[0].kind != Token::Word && e.value[0].kind != Token::String))
        return std::unexpected(keyErr(d, e, "expects one word, got '" + joined(e) + "'"));
    return std::string(e.value[0].text);
}

std::expected<bool, std::string> switchOf(const Dict& d, const Entry& e)
{
    auto w = wordOf(d, e);
    if (!w) return std::unexpected(w.error());
    if (*w == "yes" || *w == "on" || *w == "true" || *w == "y" || *w == "t") return true;
    if (*w == "no" || *w == "off" || *w == "false" || *w == "n" || *w == "f" || *w == "none") return false;
    return std::unexpected(keyErr(d, e, "expects a switch (yes/no/on/off/true/false), got '" + *w + "'"));
}

std::expected<std::vector<double>, std::string> vectorOf(const Dict& d, const std::vector<Token>& t,
                                                         std::size_t& pos, int line, std::string_view key)
{
    auto fail = [&] {
        return std::unexpected(d.where(line) + ": '" + std::string(key) + "' expects a vector (x y z)");
    };
    if (pos + 5 > t.size() || !t[pos].is('(') || !t[pos + 4].is(')')) return fail();
    std::vector<double> v;
    for (int i = 1; i <= 3; ++i) {
        if (t[pos + std::size_t(i)].kind != Token::Number) return fail();
        v.push_back(t[pos + std::size_t(i)].num);
    }
    pos += 5;
    return v;
}

std::string joined(const Entry& e)
{
    std::string s;
    for (const Token& t : e.value) {
        if (!s.empty()) s += ' ';
        s += t.text;
    }
    return s;
}

} // namespace dns::foam
