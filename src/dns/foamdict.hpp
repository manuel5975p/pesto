// OpenFOAM dictionary reader (the subset the incompressible FV path reads from
// an OpenFOAM case: 0/U, 0/p, system/{controlDict,fvSchemes,fvSolution},
// constant/{transportProperties,MRFProperties,momentumTransport}; docs/fv.md).
// Host only, pure (text in, tree out): the case loader (dns/fvcase) reads the
// files. Tokens follow OpenFOAM's ISstream: a word may carry balanced
// parentheses (`div((nuEff*dev2(T(grad(U)))))` is ONE word), `(` opening a
// value is a list, `//` and `/* */` are comments, `"..."` a string (a quoted
// KEY is a regular expression in OpenFOAM: `"(U|e)"`). Directives (`#...`)
// are kept as entries; the consumer accepts or rejects them BY NAME.
// `$name` / `${name}` macros are expanded while reading, as OpenFOAM does,
// against the entries read SO FAR of the same dictionary, then of each
// enclosing one (docs/fv.md "Macros"):
//  - inside a value (primitiveEntry::expandVariable): the tokens of the
//    LITERAL entry `name` (pattern keys do not match); a dictionary there is
//    an error;
//  - `$name;` as an entry (dictionary::substituteScopedKeyword): the entries
//    of dictionary `name` (literal key, else the last matching pattern key)
//    are added in place -- a later entry of the same key wins (find: the
//    last); a sub-dictionary key already present is an error (OpenFOAM merges).
// Every other form ($:a.b, $..a, $a.b, ${a:-b}, ${{expr}}, an environment
// variable) is an error naming it.
#pragma once

#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dns::foam {

struct Token
{
    enum Kind : unsigned char { Word, String, Number, Punct } kind = Word;
    std::string_view text; // into the Dict's source (Punct: one char)
    double num = 0;        // Number only
    int line = 0;
    bool is(char c) const { return kind == Punct && text.size() == 1 && text[0] == c; }
};

struct Dict;

// One entry: `key value... ;` (tokens), `key { ... }` (sub-dictionary) or a
// directive `#name args` (key holds "#name", value its argument tokens).
struct Entry
{
    std::string key;
    bool quotedKey = false;  // "..." key: an OpenFOAM regular expression
    bool directive = false;
    bool referenced = false;    // a `$key` macro read this entry (a variable definition)
    std::shared_ptr<Dict> dict; // non-null for a sub-dictionary
    std::vector<Token> value;
    int line = 0;
};

struct Dict
{
    std::shared_ptr<const std::string> src; // keeps every token's text alive
    std::string file;                        // for messages
    std::vector<Entry> entries;              // file order

    // The LAST entry with this exact (unquoted) key (OpenFOAM: a repeated
    // key overrides), or null.
    const Entry* find(std::string_view key) const;
    // "file:line: what" for messages.
    std::string where(int line) const;
};

// Parses a whole file's text. Errors name the file and line: unbalanced
// brackets, a missing `;`, a key without a value, an unterminated string or
// comment.
std::expected<Dict, std::string> parseDict(std::string text, std::string file);

// Value accessors on an entry's tokens. Each error names the file, line and key.
std::expected<double, std::string> scalarOf(const Dict& d, const Entry& e);
std::expected<std::string, std::string> wordOf(const Dict& d, const Entry& e);
std::expected<bool, std::string> switchOf(const Dict& d, const Entry& e); // yes/no/on/off/true/false
// `(x y z)` -> 3 numbers. Pre: none (errors on anything else).
std::expected<std::vector<double>, std::string> vectorOf(const Dict& d, const std::vector<Token>& t,
                                                         std::size_t& pos, int line, std::string_view key);

// The tokens of e joined by single spaces (diagnostics, scheme strings).
std::string joined(const Entry& e);

} // namespace dns::foam
