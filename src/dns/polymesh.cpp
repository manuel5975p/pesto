#include "dns/polymesh.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <format>
#include <fstream>
#include <limits>
#include <map>
#include <span>
#include <sstream>
#include <string_view>
#include <tuple>

namespace dns {
namespace {

namespace fs = std::filesystem;
using std::string;
using std::string_view;

static_assert(std::endian::native == std::endian::little,
              "binary polyMesh I/O assumes a little-endian host (OpenFOAM LSB)");

constexpr const char* kTypeNames[] = {"wall",  "patch",  "symmetry", "symmetryPlane",
                                      "empty", "cyclic", "wedge"};

// One file's text and a cursor; every error names the file.
struct Src
{
    string path;  // for messages
    string text;
    std::size_t p = 0;

    string err(const string& what) const { return path + ": " + what; }
};

std::expected<Src, string> load(const fs::path& dir, const char* name)
{
    Src s;
    s.path = (dir / name).string();
    std::error_code ec;
    if (fs::exists(dir / (string(name) + ".gz"), ec))
        return std::unexpected(s.path + ".gz: compressed polyMesh file is not supported "
                                        "(rewrite it with writeCompression off, "
                                        "e.g. foamFormatConvert -constant)");
    std::ifstream f(dir / name, std::ios::binary);
    if (!f) return std::unexpected(s.path + ": cannot read (missing polyMesh file)");
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad()) return std::unexpected(s.path + ": read error");
    s.text = std::move(ss).str();
    return s;
}

bool isWs(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

void skipWs(Src& s)
{
    const string& t = s.text;
    while (s.p < t.size()) {
        const char c = t[s.p];
        if (isWs(c)) {
            ++s.p;
            continue;
        }
        if (c == '/' && s.p + 1 < t.size() && t[s.p + 1] == '/') {
            const auto e = t.find('\n', s.p);
            s.p = e == string::npos ? t.size() : e + 1;
            continue;
        }
        if (c == '/' && s.p + 1 < t.size() && t[s.p + 1] == '*') {
            const auto e = t.find("*/", s.p + 2);
            s.p = e == string::npos ? t.size() : e + 2;
            continue;
        }
        break;
    }
}

bool isPunct(char c)
{
    return c == '(' || c == ')' || c == '{' || c == '}' || c == ';' || c == '[' || c == ']';
}

struct Tok
{
    enum Kind { End, Word, Punct, Str } kind = End;
    string_view v;
};

Tok next(Src& s)
{
    skipWs(s);
    const string& t = s.text;
    if (s.p >= t.size()) return {};
    const char c = t[s.p];
    if (isPunct(c)) return {Tok::Punct, string_view(t).substr(s.p++, 1)};
    if (c == '"') {
        const auto e = t.find('"', s.p + 1);
        const std::size_t b = s.p + 1;
        s.p = e == string::npos ? t.size() : e + 1;
        return {Tok::Str, string_view(t).substr(b, (e == string::npos ? t.size() : e) - b)};
    }
    const std::size_t b = s.p;
    while (s.p < t.size() && !isWs(t[s.p]) && !isPunct(t[s.p]) && t[s.p] != '"') ++s.p;
    return {Tok::Word, string_view(t).substr(b, s.p - b)};
}

string show(const Tok& t)
{
    return t.kind == Tok::End ? string("end of file") : "'" + string(t.v) + "'";
}

std::expected<void, string> expectPunct(Src& s, char c)
{
    const Tok t = next(s);
    if (t.kind != Tok::Punct || t.v[0] != c)
        return std::unexpected(s.err(std::format("expected '{}', got {}", c, show(t))));
    return {};
}

std::expected<void, string> noDirective(const Src& s, const Tok& t)
{
    if (t.kind == Tok::Word && !t.v.empty() && t.v[0] == '#')
        return std::unexpected(s.err("directive '" + string(t.v)
                                     + "' is not supported (expand the file first: "
                                       "foamDictionary -expand)"));
    return {};
}

bool toI64(string_view w, std::int64_t& v)
{
    const auto r = std::from_chars(w.data(), w.data() + w.size(), v);
    return r.ec == std::errc{} && r.ptr == w.data() + w.size();
}

bool toF64(string_view w, double& v)
{
    const auto r = std::from_chars(w.data(), w.data() + w.size(), v);
    return r.ec == std::errc{} && r.ptr == w.data() + w.size() && std::isfinite(v);
}

// --- header ---------------------------------------------------------------

struct Header
{
    bool binary = false;
    int labelBits = 32;
    string cls, note;
};

std::expected<Header, string> readHeader(Src& s)
{
    Tok t = next(s);
    if (t.kind != Tok::Word || t.v != "FoamFile")
        return std::unexpected(s.err("no FoamFile header (got " + show(t) + ")"));
    if (auto e = expectPunct(s, '{'); !e) return std::unexpected(e.error());
    std::map<string, string> kv;
    for (;;) {
        t = next(s);
        if (t.kind == Tok::Punct && t.v[0] == '}') break;
        if (auto e = noDirective(s, t); !e) return std::unexpected(e.error());
        if (t.kind != Tok::Word)
            return std::unexpected(s.err("malformed FoamFile header at " + show(t)));
        const string key(t.v);
        string val;
        Tok v = next(s);
        if (v.kind == Tok::Punct && v.v[0] == '{') {  // e.g. `meta { ... }`
            for (int depth = 1; depth > 0;) {
                v = next(s);
                if (v.kind == Tok::End)
                    return std::unexpected(s.err("unterminated '" + key + "' in header"));
                if (v.kind == Tok::Punct && v.v[0] == '{') ++depth;
                if (v.kind == Tok::Punct && v.v[0] == '}') --depth;
            }
            continue;
        }
        while (!(v.kind == Tok::Punct && v.v[0] == ';')) {
            if (v.kind == Tok::End || (v.kind == Tok::Punct && v.v[0] == '}'))
                return std::unexpected(s.err("header entry '" + key + "' lacks ';'"));
            if (auto e = noDirective(s, v); !e) return std::unexpected(e.error());
            val += (val.empty() ? "" : " ") + string(v.v);
            v = next(s);
        }
        kv[key] = val;
    }
    Header h;
    const string fmt = kv.contains("format") ? kv["format"] : "ascii";
    if (fmt != "ascii" && fmt != "binary")
        return std::unexpected(s.err("format '" + fmt + "' (ascii or binary)"));
    h.binary = fmt == "binary";
    if (kv.contains("arch")) {
        const string& a = kv["arch"];
        if (a.find("MSB") != string::npos)
            return std::unexpected(s.err("arch '" + a + "': big-endian data is not supported"));
        auto bits = [&](const char* key, int dflt) {
            const auto at = a.find(key);
            if (at == string::npos) return dflt;
            std::int64_t v = 0;
            const char* b = a.data() + at + std::strlen(key);
            std::from_chars(b, a.data() + a.size(), v);
            return int(v);
        };
        h.labelBits = bits("label=", 32);
        const int sb = bits("scalar=", 64);
        if (h.labelBits != 32 && h.labelBits != 64)
            return std::unexpected(s.err(std::format("arch label={} (32 or 64)", h.labelBits)));
        if (sb != 64)
            return std::unexpected(s.err(std::format("arch scalar={}: only 64-bit (double) "
                                                     "meshes are supported",
                                                     sb)));
    }
    h.cls = kv.contains("class") ? kv["class"] : "";
    h.note = kv.contains("note") ? kv["note"] : "";
    return h;
}

// --- lists ----------------------------------------------------------------

// Reads `N (` and returns N; afterwards s.p is just past '('.
std::expected<std::int64_t, string> listOpen(Src& s, const char* what)
{
    const Tok t = next(s);
    std::int64_t n = 0;
    if (t.kind != Tok::Word || !toI64(t.v, n) || n < 0)
        return std::unexpected(s.err(std::format("expected the size of {}, got {}", what, show(t))));
    skipWs(s);
    if (s.p >= s.text.size()) return std::unexpected(s.err(std::format("truncated {}", what)));
    if (s.text[s.p] == '{')
        return std::unexpected(s.err(std::format("uniform list N{{...}} for {} is not supported", what)));
    if (s.text[s.p] != '(')
        return std::unexpected(s.err(std::format("expected '(' after the size of {}", what)));
    ++s.p;
    return n;
}

std::expected<void, string> listClose(Src& s, bool binary, const char* what)
{
    if (!binary) skipWs(s);
    if (s.p >= s.text.size() || s.text[s.p] != ')')
        return std::unexpected(s.err(std::format("truncated or malformed {} (no closing ')')", what)));
    ++s.p;
    return {};
}

std::expected<std::int32_t, string> narrow(const Src& s, std::int64_t v, const char* what)
{
    if (v < 0 || v > std::numeric_limits<std::int32_t>::max())
        return std::unexpected(s.err(std::format("{} label {} out of range (int32 index)", what, v)));
    return std::int32_t(v);
}

std::expected<std::int64_t, string> asciiLabel(Src& s, const char* what)
{
    skipWs(s);
    std::int64_t v = 0;
    const char* b = s.text.data() + s.p;
    const auto r = std::from_chars(b, s.text.data() + s.text.size(), v);
    if (r.ec != std::errc{})
        return std::unexpected(s.err(std::format("truncated or malformed {} (expected a label)", what)));
    s.p += std::size_t(r.ptr - b);
    return v;
}

std::expected<double, string> asciiScalar(Src& s, const char* what)
{
    skipWs(s);
    double v = 0;
    const char* b = s.text.data() + s.p;
    const auto r = std::from_chars(b, s.text.data() + s.text.size(), v);
    if (r.ec != std::errc{} || !std::isfinite(v))
        return std::unexpected(s.err(std::format("truncated or malformed {} (expected a finite scalar)", what)));
    s.p += std::size_t(r.ptr - b);
    return v;
}

// n labels after listOpen, narrowed to int32 and range-checked.
std::expected<std::vector<std::int32_t>, string> labelBody(Src& s, const Header& h, std::int64_t n,
                                                           const char* what)
{
    std::vector<std::int32_t> out;
    out.reserve(std::size_t(n));
    if (h.binary) {
        const std::size_t w = h.labelBits == 64 ? 8 : 4;
        if (s.text.size() - s.p < std::size_t(n) * w)
            return std::unexpected(s.err(std::format("truncated binary {}", what)));
        for (std::int64_t i = 0; i < n; ++i) {
            std::int64_t v = 0;
            if (w == 8)
                std::memcpy(&v, s.text.data() + s.p, 8);
            else {
                std::int32_t v32 = 0;
                std::memcpy(&v32, s.text.data() + s.p, 4);
                v = v32;
            }
            s.p += w;
            auto l = narrow(s, v, what);
            if (!l) return std::unexpected(l.error());
            out.push_back(*l);
        }
    } else {
        for (std::int64_t i = 0; i < n; ++i) {
            auto v = asciiLabel(s, what);
            if (!v) return std::unexpected(v.error());
            auto l = narrow(s, *v, what);
            if (!l) return std::unexpected(l.error());
            out.push_back(*l);
        }
    }
    if (auto e = listClose(s, h.binary, what); !e) return std::unexpected(e.error());
    return out;
}

std::expected<std::vector<std::int32_t>, string> labelList(Src& s, const Header& h, const char* what)
{
    auto n = listOpen(s, what);
    if (!n) return std::unexpected(n.error());
    return labelBody(s, h, *n, what);
}

// `nPoints:60  nCells:24  nFaces:98  nInternalFaces:46`
std::expected<void, string> parseNote(const Src& s, const string& note, PolyManifest& m)
{
    std::istringstream is(note);
    string kv;
    int got = 0;
    while (is >> kv) {
        const auto c = kv.find(':');
        if (c == string::npos) continue;
        std::int64_t v = 0;
        if (!toI64(string_view(kv).substr(c + 1), v) || v < 0)
            return std::unexpected(s.err("malformed note entry '" + kv + "'"));
        const string k = kv.substr(0, c);
        if (k == "nPoints") m.nPoints = v, got |= 1;
        if (k == "nCells") m.nCells = v, got |= 2;
        if (k == "nFaces") m.nFaces = v, got |= 4;
        if (k == "nInternalFaces") m.nInternalFaces = v, got |= 8;
    }
    if (got != 15)
        return std::unexpected(s.err("header note lacks nPoints/nCells/nFaces/nInternalFaces "
                                     "(not an OpenFOAM-written owner file?)"));
    if (m.nInternalFaces > m.nFaces)
        return std::unexpected(s.err("note: nInternalFaces > nFaces"));
    for (const std::int64_t v : {m.nPoints, m.nCells, m.nFaces})
        if (v > std::numeric_limits<std::int32_t>::max())
            return std::unexpected(s.err(std::format("note: count {} exceeds int32 indexing", v)));
    return {};
}

// --- dictionaries (boundary, cellZones) -------------------------------------

// The value tokens of one entry up to its ';' (depth 0). Sub-dictionaries
// and lists are kept as tokens.
std::expected<std::vector<string>, string> entryValue(Src& s, const string& key)
{
    std::vector<string> out;
    int depth = 0;
    for (;;) {
        const Tok t = next(s);
        if (t.kind == Tok::End) return std::unexpected(s.err("entry '" + key + "' lacks ';'"));
        if (auto e = noDirective(s, t); !e) return std::unexpected(e.error());
        if (t.kind == Tok::Punct) {
            const char c = t.v[0];
            if (c == ';' && depth == 0) return out;
            if (c == '(' || c == '{' || c == '[') ++depth;
            if (c == ')' || c == '}' || c == ']') {
                if (--depth < 0) return std::unexpected(s.err("entry '" + key + "' lacks ';'"));
            }
        }
        out.emplace_back(t.v);
    }
}

std::expected<FoamPatchType, string> patchType(const Src& s, const string& patch, const string& t)
{
    for (std::size_t i = 0; i < std::size(kTypeNames); ++i)
        if (t == kTypeNames[i]) return FoamPatchType(i);
    if (t.starts_with("processor"))
        return std::unexpected(s.err("patch '" + patch + "' is type " + t
                                     + ": the mesh is decomposed (run reconstructParMesh)"));
    if (t == "cyclicAMI" || t == "cyclicACMI" || t.starts_with("nonConformal"))
        return std::unexpected(s.err("patch '" + patch + "' is type " + t
                                     + ": non-conformal coupling is not supported"));
    if (t.starts_with("mapped"))
        return std::unexpected(s.err("patch '" + patch + "' is type " + t
                                     + ": mapped patches are not supported"));
    return std::unexpected(s.err("patch '" + patch + "' has unsupported type '" + t
                                 + "' (wall, patch, symmetry, symmetryPlane, empty, cyclic, wedge)"));
}

std::expected<std::vector<FoamPatch>, string> readBoundary(const fs::path& dir)
{
    auto s = load(dir, "boundary");
    if (!s) return std::unexpected(s.error());
    auto h = readHeader(*s);
    if (!h) return std::unexpected(h.error());
    if (h->binary)
        return std::unexpected(s->err("binary-format boundary dictionary is not supported"));
    auto n = listOpen(*s, "the patch list");
    if (!n) return std::unexpected(n.error());
    std::vector<FoamPatch> out;
    for (std::int64_t i = 0; i < *n; ++i) {
        Tok t = next(*s);
        if (auto e = noDirective(*s, t); !e) return std::unexpected(e.error());
        if (t.kind != Tok::Word) return std::unexpected(s->err("expected a patch name, got " + show(t)));
        FoamPatch p;
        p.name = string(t.v);
        if (auto e = expectPunct(*s, '{'); !e) return std::unexpected(e.error());
        std::map<string, std::vector<string>> kv;
        for (;;) {
            t = next(*s);
            if (t.kind == Tok::Punct && t.v[0] == '}') break;
            if (auto e = noDirective(*s, t); !e) return std::unexpected(e.error());
            if (t.kind != Tok::Word)
                return std::unexpected(s->err("patch '" + p.name + "': malformed entry at " + show(t)));
            const string key(t.v);
            auto v = entryValue(*s, key);
            if (!v) return std::unexpected(v.error());
            kv[key] = std::move(*v);
        }
        auto one = [&](const char* key) -> std::expected<string, string> {
            if (!kv.contains(key) || kv[key].size() != 1)
                return std::unexpected(s->err("patch '" + p.name + "': missing or malformed '" + key + "'"));
            return kv[key][0];
        };
        auto ty = one("type");
        if (!ty) return std::unexpected(ty.error());
        auto pt = patchType(*s, p.name, *ty);
        if (!pt) return std::unexpected(pt.error());
        p.type = *pt;
        for (auto [key, dst] : {std::pair{"nFaces", &p.nFaces}, {"startFace", &p.startFace}}) {
            auto w = one(key);
            if (!w) return std::unexpected(w.error());
            if (!toI64(*w, *dst) || *dst < 0)
                return std::unexpected(s->err("patch '" + p.name + "': " + key + " '" + *w + "'"));
        }
        if (p.type == FoamPatchType::Cyclic) {
            auto nb = one("neighbourPatch");
            if (!nb) return std::unexpected(nb.error());
            p.neighbourPatch = *nb;
            const string tr = kv.contains("transform") && kv["transform"].size() == 1
                                  ? kv["transform"][0]
                                  : string("unknown");
            if (tr == "rotational")
                return std::unexpected(s->err("cyclic '" + p.name
                                              + "': transform rotational is not supported "
                                                "(translational only)"));
            if (tr != "translational" && tr != "unknown" && tr != "coincidentFullMatch"
                && tr != "noOrdering")
                return std::unexpected(s->err("cyclic '" + p.name + "': unknown transform '" + tr + "'"));
            if (kv.contains("separationVector")) {
                const auto& v = kv["separationVector"];
                std::array<double, 3> sep{};
                if (v.size() != 5 || v[0] != "(" || v[4] != ")" || !toF64(v[1], sep[0])
                    || !toF64(v[2], sep[1]) || !toF64(v[3], sep[2]))
                    return std::unexpected(s->err("cyclic '" + p.name + "': malformed separationVector"));
                p.separation = sep;
            } else if (tr == "translational")
                return std::unexpected(s->err("cyclic '" + p.name
                                              + "': transform translational without separationVector"));
        }
        out.push_back(std::move(p));
    }
    if (auto e = listClose(*s, false, "the patch list"); !e) return std::unexpected(e.error());

    // Cyclic pairing: each names the other, same size.
    for (const auto& p : out) {
        if (p.type != FoamPatchType::Cyclic) continue;
        const auto q = std::ranges::find(out, p.neighbourPatch, &FoamPatch::name);
        if (q == out.end() || q->type != FoamPatchType::Cyclic || q->neighbourPatch != p.name
            || q->name == p.name)
            return std::unexpected(s->err("cyclic '" + p.name + "': neighbourPatch '" + p.neighbourPatch
                                          + "' is not a cyclic naming it back"));
        if (q->nFaces != p.nFaces)
            return std::unexpected(s->err("cyclic '" + p.name + "' and '" + q->name
                                          + "' have different face counts"));
    }
    return out;
}

// cellZones: names, and (keep) the cell lists. Absent file = no zones.
std::expected<void, string> readZones(const fs::path& dir, bool keep, std::vector<string>& names,
                                      std::vector<std::vector<std::int32_t>>& cells)
{
    std::error_code ec;
    if (fs::exists(dir / "cellZones.gz", ec))
        return std::unexpected((dir / "cellZones.gz").string()
                               + ": compressed polyMesh file is not supported");
    if (!fs::exists(dir / "cellZones", ec)) return {};
    auto s = load(dir, "cellZones");
    if (!s) return std::unexpected(s.error());
    auto h = readHeader(*s);
    if (!h) return std::unexpected(h.error());
    auto n = listOpen(*s, "the cellZone list");
    if (!n) return std::unexpected(n.error());
    for (std::int64_t z = 0; z < *n; ++z) {
        Tok t = next(*s);
        if (auto e = noDirective(*s, t); !e) return std::unexpected(e.error());
        if (t.kind != Tok::Word) return std::unexpected(s->err("expected a cellZone name, got " + show(t)));
        const string name(t.v);
        if (auto e = expectPunct(*s, '{'); !e) return std::unexpected(e.error());
        std::vector<std::int32_t> labels;
        bool haveLabels = false;
        for (;;) {
            t = next(*s);
            if (t.kind == Tok::Punct && t.v[0] == '}') break;
            if (auto e = noDirective(*s, t); !e) return std::unexpected(e.error());
            if (t.kind != Tok::Word)
                return std::unexpected(s->err("cellZone '" + name + "': malformed entry at " + show(t)));
            const string key(t.v);
            if (key != "cellLabels") {
                if (auto v = entryValue(*s, key); !v) return std::unexpected(v.error());
                continue;
            }
            // cellLabels [List<label>] N ( ... ) ;
            const std::size_t save = s->p;
            t = next(*s);
            if (!(t.kind == Tok::Word && t.v.starts_with("List<"))) s->p = save;
            auto l = labelList(*s, *h, "cellLabels");
            if (!l) return std::unexpected(l.error());
            labels = std::move(*l);
            haveLabels = true;
            if (auto e = expectPunct(*s, ';'); !e) return std::unexpected(e.error());
        }
        if (!haveLabels) return std::unexpected(s->err("cellZone '" + name + "' has no cellLabels"));
        names.push_back(name);
        if (keep) cells.push_back(std::move(labels));
    }
    if (auto e = listClose(*s, false, "the cellZone list"); !e) return std::unexpected(e.error());
    return {};
}

std::expected<void, string> checkPatchesContiguous(const string& where, const PolyManifest& m)
{
    std::int64_t at = m.nInternalFaces;
    for (const auto& p : m.patches) {
        if (p.startFace != at)
            return std::unexpected(std::format("{}: patch '{}' starts at face {}, expected {} (patches "
                                               "must tile [nInternalFaces, nFaces) in order)",
                                               where, p.name, p.startFace, at));
        at += p.nFaces;
    }
    if (at != m.nFaces)
        return std::unexpected(std::format("{}: patches end at face {}, nFaces is {}", where, at, m.nFaces));
    return {};
}

// --- writing ------------------------------------------------------------------

string header(const char* cls, const char* object, bool binary, int labelBits, const string& note)
{
    string h = std::format(
        "/*--------------------------------*- C++ -*----------------------------------*\\\n"
        "  written by dns (dns/polymesh)\n"
        "\\*---------------------------------------------------------------------------*/\n"
        "FoamFile\n{{\n"
        "    version     2.0;\n"
        "    format      {};\n"
        "    arch        \"LSB;label={};scalar=64\";\n",
        binary ? "binary" : "ascii", labelBits);
    if (!note.empty()) h += "    note        \"" + note + "\";\n";
    h += std::format("    class       {};\n    location    \"constant/polyMesh\";\n"
                     "    object      {};\n}}\n"
                     "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n",
                     cls, object);
    return h;
}

void putLabels(string& o, std::span<const std::int32_t> v, bool binary, int labelBits)
{
    o += std::format("{}\n(", v.size());
    if (binary) {
        if (labelBits == 64)
            for (const std::int32_t x : v) {
                const std::int64_t w = x;
                o.append(reinterpret_cast<const char*>(&w), 8);
            }
        else
            o.append(reinterpret_cast<const char*>(v.data()), v.size() * 4);
        o += ")\n";
        return;
    }
    o += "\n";
    for (const std::int32_t x : v) o += std::format("{}\n", x);
    o += ")\n";
}

std::expected<void, string> putFile(const fs::path& dir, const char* name, const string& body)
{
    std::ofstream f(dir / name, std::ios::binary | std::ios::trunc);
    if (!f) return std::unexpected((dir / name).string() + ": cannot create");
    f << body << "\n\n// ************************************************************************* //\n";
    f.close();
    if (!f) return std::unexpected((dir / name).string() + ": write failed");
    return {};
}

} // namespace

const char* foamPatchTypeName(FoamPatchType t)
{
    return kTypeNames[std::size_t(t)];
}

std::expected<PolyManifest, std::string> readPolyManifest(const std::filesystem::path& dir)
{
    std::error_code ec;
    if (!fs::is_directory(dir, ec))
        return std::unexpected(dir.string() + ": not a directory (expected constant/polyMesh)");
    PolyManifest m;
    {
        auto s = load(dir, "owner");
        if (!s) return std::unexpected(s.error());
        auto h = readHeader(*s);
        if (!h) return std::unexpected(h.error());
        if (auto e = parseNote(*s, h->note, m); !e) return std::unexpected(e.error());
        m.binary = h->binary;
        m.labelBits = h->labelBits;
    }
    auto b = readBoundary(dir);
    if (!b) return std::unexpected(b.error());
    m.patches = std::move(*b);
    if (auto e = checkPatchesContiguous((dir / "boundary").string(), m); !e)
        return std::unexpected(e.error());
    std::vector<std::vector<std::int32_t>> none;
    if (auto e = readZones(dir, false, m.cellZones, none); !e) return std::unexpected(e.error());
    return m;
}

std::expected<PolyMesh, std::string> readPolyMesh(const std::filesystem::path& dir)
{
    auto man = readPolyManifest(dir);
    if (!man) return std::unexpected(man.error());
    PolyMesh m;
    m.man = std::move(*man);
    const std::int64_t nP = m.man.nPoints, nF = m.man.nFaces, nI = m.man.nInternalFaces,
                       nC = m.man.nCells;

    {   // points
        auto s = load(dir, "points");
        if (!s) return std::unexpected(s.error());
        auto h = readHeader(*s);
        if (!h) return std::unexpected(h.error());
        auto n = listOpen(*s, "points");
        if (!n) return std::unexpected(n.error());
        if (*n != nP)
            return std::unexpected(s->err(std::format("{} points, owner note says {}", *n, nP)));
        m.px.resize(std::size_t(nP));
        m.py.resize(std::size_t(nP));
        m.pz.resize(std::size_t(nP));
        if (h->binary) {
            if (s->text.size() - s->p < std::size_t(nP) * 24)
                return std::unexpected(s->err("truncated binary points"));
            for (std::int64_t i = 0; i < nP; ++i) {
                double v[3];
                std::memcpy(v, s->text.data() + s->p, 24);
                s->p += 24;
                if (!std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]))
                    return std::unexpected(s->err(std::format("point {} is not finite", i)));
                m.px[std::size_t(i)] = v[0];
                m.py[std::size_t(i)] = v[1];
                m.pz[std::size_t(i)] = v[2];
            }
        } else {
            for (std::int64_t i = 0; i < nP; ++i) {
                if (auto e = expectPunct(*s, '('); !e) return std::unexpected(e.error());
                double v[3];
                for (double& x : v) {
                    auto d = asciiScalar(*s, "points");
                    if (!d) return std::unexpected(d.error());
                    x = *d;
                }
                if (auto e = expectPunct(*s, ')'); !e) return std::unexpected(e.error());
                m.px[std::size_t(i)] = v[0];
                m.py[std::size_t(i)] = v[1];
                m.pz[std::size_t(i)] = v[2];
            }
        }
        if (auto e = listClose(*s, h->binary, "points"); !e) return std::unexpected(e.error());
    }

    {   // faces
        auto s = load(dir, "faces");
        if (!s) return std::unexpected(s.error());
        auto h = readHeader(*s);
        if (!h) return std::unexpected(h.error());
        if (h->cls == "faceCompactList") {
            auto off = labelList(*s, *h, "face offsets");
            if (!off) return std::unexpected(off.error());
            auto lab = labelList(*s, *h, "face labels");
            if (!lab) return std::unexpected(lab.error());
            m.faceStart = std::move(*off);
            m.faceNodes = std::move(*lab);
        } else if (h->cls == "faceList") {
            if (h->binary)
                return std::unexpected(s->err("binary faceList (non-compact) is not supported"));
            auto n = listOpen(*s, "faces");
            if (!n) return std::unexpected(n.error());
            m.faceStart.reserve(std::size_t(*n) + 1);
            m.faceStart.push_back(0);
            m.faceNodes.reserve(std::size_t(*n) * 4);
            for (std::int64_t f = 0; f < *n; ++f) {
                auto k = asciiLabel(*s, "faces");
                if (!k) return std::unexpected(k.error());
                skipWs(*s);
                if (s->p >= s->text.size() || s->text[s->p] != '(')
                    return std::unexpected(s->err(std::format("face {}: expected '('", f)));
                ++s->p;
                for (std::int64_t q = 0; q < *k; ++q) {
                    auto v = asciiLabel(*s, "faces");
                    if (!v) return std::unexpected(v.error());
                    auto l = narrow(*s, *v, "face");
                    if (!l) return std::unexpected(l.error());
                    m.faceNodes.push_back(*l);
                }
                if (auto e = listClose(*s, false, "face"); !e) return std::unexpected(e.error());
                if (m.faceNodes.size() > std::size_t(std::numeric_limits<std::int32_t>::max()))
                    return std::unexpected(s->err("face labels exceed int32 offsets"));
                m.faceStart.push_back(std::int32_t(m.faceNodes.size()));
            }
            if (auto e = listClose(*s, false, "faces"); !e) return std::unexpected(e.error());
        } else
            return std::unexpected(s->err("class '" + h->cls + "' (faceList or faceCompactList)"));
        if (std::int64_t(m.faceStart.size()) != nF + 1)
            return std::unexpected(s->err(std::format("{} faces, owner note says {}",
                                                      std::int64_t(m.faceStart.size()) - 1, nF)));
        if (m.faceStart[0] != 0 || std::size_t(m.faceStart.back()) != m.faceNodes.size())
            return std::unexpected(s->err("face offsets do not span the label list"));
        for (std::int64_t f = 0; f < nF; ++f) {
            const auto a = m.faceStart[std::size_t(f)], b = m.faceStart[std::size_t(f) + 1];
            if (b - a < 3)
                return std::unexpected(s->err(std::format("face {} has {} nodes (< 3)", f, b - a)));
        }
        for (const std::int32_t v : m.faceNodes)
            if (v >= nP)
                return std::unexpected(s->err(std::format("point label {} out of range (nPoints {})", v, nP)));
    }

    for (auto [name, dst, want] : {std::tuple{"owner", &m.owner, nF}, {"neighbour", &m.neighbour, nI}}) {
        auto s = load(dir, name);
        if (!s) return std::unexpected(s.error());
        auto h = readHeader(*s);
        if (!h) return std::unexpected(h.error());
        auto l = labelList(*s, *h, name);
        if (!l) return std::unexpected(l.error());
        if (std::int64_t(l->size()) != want)
            return std::unexpected(s->err(std::format("{} entries, expected {}", l->size(), want)));
        for (const std::int32_t c : *l)
            if (c >= nC)
                return std::unexpected(s->err(std::format("cell label {} out of range (nCells {})", c, nC)));
        *dst = std::move(*l);
    }
    for (std::int64_t f = 0; f < nI; ++f)
        if (m.owner[std::size_t(f)] >= m.neighbour[std::size_t(f)])
            return std::unexpected((dir / "neighbour").string()
                                   + std::format(": owner >= neighbour on internal face {} ({} >= {})", f,
                                                 m.owner[std::size_t(f)], m.neighbour[std::size_t(f)]));

    {   // zones
        std::vector<string> names;
        if (auto e = readZones(dir, true, names, m.zoneCells); !e) return std::unexpected(e.error());
        std::vector<int> zoneOf(std::size_t(nC), -1);
        for (std::size_t z = 0; z < m.zoneCells.size(); ++z) {
            auto& cz = m.zoneCells[z];
            std::ranges::sort(cz);
            for (const std::int32_t c : cz) {
                if (c >= nC)
                    return std::unexpected((dir / "cellZones").string()
                                           + std::format(": cellZone '{}' cell {} out of range", names[z], c));
                if (zoneOf[std::size_t(c)] == int(z))
                    return std::unexpected((dir / "cellZones").string()
                                           + std::format(": cellZone '{}' lists cell {} twice", names[z], c));
                if (zoneOf[std::size_t(c)] >= 0)
                    return std::unexpected((dir / "cellZones").string()
                                           + std::format(": cell {} is in cellZones '{}' and '{}' "
                                                         "(overlapping zones)",
                                                         c, names[std::size_t(zoneOf[std::size_t(c)])],
                                                         names[z]));
                zoneOf[std::size_t(c)] = int(z);
            }
        }
    }
    return m;
}

std::expected<void, std::string> writePolyMesh(const std::filesystem::path& dir, const PolyMesh& m,
                                               bool binary, int labelBits)
{
    if (labelBits != 32 && labelBits != 64) return std::unexpected("writePolyMesh: labelBits 32 or 64");
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return std::unexpected(dir.string() + ": cannot create (" + ec.message() + ")");
    const auto& man = m.man;
    const string note = std::format("nPoints:{}  nCells:{}  nFaces:{}  nInternalFaces:{}", man.nPoints,
                                    man.nCells, man.nFaces, man.nInternalFaces);

    {   // points
        string o = header("vectorField", "points", binary, labelBits, "");
        o += std::format("{}\n(", m.px.size());
        if (binary)
            for (std::size_t i = 0; i < m.px.size(); ++i) {
                const double v[3] = {m.px[i], m.py[i], m.pz[i]};
                o.append(reinterpret_cast<const char*>(v), 24);
            }
        else {
            o += "\n";
            for (std::size_t i = 0; i < m.px.size(); ++i)
                o += std::format("({} {} {})\n", m.px[i], m.py[i], m.pz[i]);
        }
        o += ")\n";
        if (auto e = putFile(dir, "points", o); !e) return e;
    }
    {   // faces
        string o = header(binary ? "faceCompactList" : "faceList", "faces", binary, labelBits, "");
        if (binary) {
            putLabels(o, m.faceStart, true, labelBits);
            o += "\n";
            putLabels(o, m.faceNodes, true, labelBits);
        } else {
            const std::size_t nf = m.faceStart.empty() ? 0 : m.faceStart.size() - 1;
            o += std::format("{}\n(\n", nf);
            for (std::size_t f = 0; f < nf; ++f) {
                o += std::format("{}(", m.faceStart[f + 1] - m.faceStart[f]);
                for (auto q = m.faceStart[f]; q < m.faceStart[f + 1]; ++q)
                    o += std::format("{}{}", q == m.faceStart[f] ? "" : " ", m.faceNodes[std::size_t(q)]);
                o += ")\n";
            }
            o += ")\n";
        }
        if (auto e = putFile(dir, "faces", o); !e) return e;
    }
    for (auto [name, v] : {std::pair{"owner", &m.owner}, {"neighbour", &m.neighbour}}) {
        string o = header("labelList", name, binary, labelBits, note);
        putLabels(o, *v, binary, labelBits);
        if (auto e = putFile(dir, name, o); !e) return e;
    }
    {   // boundary (a dictionary: always ASCII, as OpenFOAM writes it)
        string o = header("polyBoundaryMesh", "boundary", false, labelBits, "");
        o += std::format("{}\n(\n", man.patches.size());
        for (const auto& p : man.patches) {
            o += std::format("    {}\n    {{\n        type            {};\n"
                             "        nFaces          {};\n        startFace       {};\n",
                             p.name, foamPatchTypeName(p.type), p.nFaces, p.startFace);
            if (p.type == FoamPatchType::Cyclic) {
                o += "        matchTolerance  0.0001;\n";
                o += std::format("        transform       {};\n", p.separation ? "translational" : "unknown");
                o += std::format("        neighbourPatch  {};\n", p.neighbourPatch);
                if (p.separation)
                    o += std::format("        separationVector ({} {} {});\n", (*p.separation)[0],
                                     (*p.separation)[1], (*p.separation)[2]);
            }
            o += "    }\n";
        }
        o += ")\n";
        if (auto e = putFile(dir, "boundary", o); !e) return e;
    }
    if (!man.cellZones.empty()) {
        if (man.cellZones.size() != m.zoneCells.size())
            return std::unexpected("writePolyMesh: cellZones names and cell lists differ in count");
        string o = header("regIOobject", "cellZones", binary, labelBits, "");
        o += std::format("{}\n(\n", man.cellZones.size());
        for (std::size_t z = 0; z < man.cellZones.size(); ++z) {
            o += std::format("{}\n{{\n    type            cellZone;\n    cellLabels      List<label> ",
                             man.cellZones[z]);
            putLabels(o, m.zoneCells[z], binary, labelBits);
            o += ";\n}\n";
        }
        o += ")\n";
        if (auto e = putFile(dir, "cellZones", o); !e) return e;
    } else {
        fs::remove(dir / "cellZones", ec);
    }
    return {};
}

} // namespace dns
