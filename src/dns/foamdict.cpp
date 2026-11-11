#include "dns/foamdict.hpp"

#include <charconv>
#include <cmath>
#include <format>
#include <regex>

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
            if (d == '{' && p == b + 1 && t[b] == '$') { // ${name}: one word up to its '}'
                while (p < t.size() && t[p] != '}' && t[p] != '\n') ++p;
                if (p >= t.size() || t[p] != '}') return std::unexpected(err(k.line, "unterminated ${"));
                ++p;
                break;
            }
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

// The variable name of a `$name` / `${name}` word, or an error naming the
// form: scoped names ($:a.b, $..a, $a.b, $/a/b), ${{expr}}, ${a:-b} and
// nested expansions are not expanded.
std::expected<std::string, std::string> macroName(const Lexer& L, const Token& k)
{
    std::string_view v = k.text.substr(1);
    if (!v.empty() && v.front() == '{') v = v.substr(1, v.size() - 2);
    const bool plain = !v.empty() && v.find_first_of(".:/${}+-*^ \t") == std::string_view::npos;
    if (!plain)
        return std::unexpected(L.err(k.line, std::format("macro '{}' is not supported (only $name / ${{name}}: an "
                                                         "entry of the same or an enclosing dictionary; expand "
                                                         "the file: foamDictionary -expand)",
                                                         k.text)));
    return std::string(v);
}

bool macroRegexMatch(const std::string& pattern, const std::string& name)
{
    try {
        return std::regex_match(name, std::regex(pattern, std::regex::ECMAScript));
    } catch (const std::regex_error&) {
        return false;
    }
}

// OpenFOAM's recursive lookup of `name` from the innermost dictionary of
// scope outwards, among the entries read so far: the last literal key, then
// (patterns: the dictionary-splice form) the last matching quoted key.
Entry* macroFind(const std::vector<Dict*>& scope, const std::string& name, bool patterns)
{
    for (auto it = scope.rbegin(); it != scope.rend(); ++it) {
        Entry* hit = nullptr;
        for (Entry& e : (*it)->entries)
            if (!e.quotedKey && !e.directive && e.key == name) hit = &e;
        if (!hit && patterns)
            for (Entry& e : (*it)->entries)
                if (e.quotedKey && macroRegexMatch(e.key, name)) hit = &e;
        if (hit) return hit;
    }
    return nullptr;
}

std::expected<void, std::string> parseBody(Lexer& L, Dict& d, Dict& out, bool top, int openLine,
                                           std::vector<Dict*>& scope)
{
    struct Scope
    {
        std::vector<Dict*>& s;
        Scope(std::vector<Dict*>& v, Dict* x) : s(v) { s.push_back(x); }
        ~Scope() { s.pop_back(); }
    } inScope(scope, &out);
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
        if (k.kind == Token::Word && k.text.front() == '$') {
            // `$name;` (dictionary::substituteScopedKeyword, merge mode): the
            // entries of dictionary `name` are added here; a later entry of
            // the same key replaces an earlier one (Dict::find: the last).
            auto name = macroName(L, k);
            if (!name) return std::unexpected(name.error());
            if (!(*v)->is(';'))
                return std::unexpected(L.err(e.line, std::format("macro '{}' as a keyword must be '{};' (a "
                                                                 "dictionary splice)", k.text, k.text)));
            Entry* src = macroFind(scope, *name, true);
            if (!src)
                return std::unexpected(L.err(e.line, std::format("macro '{}': no entry '{}' in this or an enclosing "
                                                                 "dictionary (environment variables are not "
                                                                 "expanded)", k.text, *name)));
            if (!src->dict)
                return std::unexpected(L.err(e.line, std::format("macro '{};': '{}' is not a dictionary", k.text,
                                                                 *name)));
            src->referenced = true;
            const std::vector<Entry> add = src->dict->entries; // a copy: src may live in `out`
            for (const Entry& a : add) {
                if (a.dict && out.find(a.key) && out.find(a.key)->dict)
                    return std::unexpected(L.err(e.line, std::format("macro '{};': sub-dictionary '{}' is already "
                                                                     "defined here (OpenFOAM merges the two: not "
                                                                     "supported)", k.text, a.key)));
                out.entries.push_back(a);
            }
            continue;
        }
        if ((*v)->is(';')) { // `key;`: an entry without a value (e.g. cache { grad(U); })
            out.entries.push_back(std::move(e));
            continue;
        }
        if ((*v)->is('{')) {
            auto sub = std::make_shared<Dict>();
            sub->src = d.src;
            sub->file = d.file;
            if (auto r = parseBody(L, d, *sub, false, (*v)->line, scope); !r) return std::unexpected(r.error());
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
            if (cur.kind == Token::Word && cur.text.front() == '$') {
                // primitiveEntry::expandVariable: the tokens of the literal
                // entry `name` (no pattern keys), innermost dictionary first.
                auto name = macroName(L, cur);
                if (!name) return std::unexpected(name.error());
                Entry* src = macroFind(scope, *name, false);
                if (!src)
                    return std::unexpected(L.err(cur.line, std::format("macro '{}': no entry '{}' in this or an "
                                                                       "enclosing dictionary (environment variables "
                                                                       "are not expanded)", cur.text, *name)));
                if (src->dict)
                    return std::unexpected(L.err(cur.line, std::format("macro '{}' names a dictionary inside a "
                                                                       "value (not supported)", cur.text)));
                src->referenced = true;
                for (Token t : src->value) {
                    t.line = cur.line;
                    e.value.push_back(t);
                }
            } else
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
    std::vector<Dict*> scope;
    if (auto r = parseBody(L, d, d, true, 0, scope); !r) return std::unexpected(r.error());
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
