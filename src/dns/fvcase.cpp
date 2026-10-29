#include "dns/fvcase.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <fstream>
#include <initializer_list>
#include <regex>
#include <set>
#include <sstream>
#include <system_error>

namespace dns {
namespace fs = std::filesystem;
using foam::Dict;
using foam::Entry;
using foam::Token;
using std::string;

namespace {

using Err = std::unexpected<string>;

string at(const Dict& d, const Entry& e) { return d.where(e.line) + ": "; }

// Every key of d must be in `allowed` (FoamFile always is). Directives other
// than `#includeEtc "caseDicts/setConstraintTypes"` are rejected.
std::expected<void, string> onlyKeys(const Dict& d, std::initializer_list<const char*> allowed,
                                     const string& what, bool allowRegex = false)
{
    for (const Entry& e : d.entries) {
        if (e.directive)
            return Err(at(d, e) + "directive '" + e.key + "' is not supported in " + what +
                       " (expand it: foamDictionary -expand)");
        if (e.key == "FoamFile") continue;
        if (e.quotedKey && allowRegex) continue;
        if (std::ranges::none_of(allowed, [&](const char* k) { return e.key == k; }))
            return Err(at(d, e) + "'" + e.key + "' is not supported in " + what +
                       " (the FV path reproduces simpleFoam exactly; remove the entry)");
    }
    return {};
}

std::expected<const Dict*, string> subDict(const Dict& d, const char* key, bool required, const string& what)
{
    const Entry* e = d.find(key);
    if (!e) {
        if (!required) return nullptr;
        return Err(d.file + ": missing '" + key + "' (" + what + ")");
    }
    if (!e->dict) return Err(at(d, *e) + "'" + key + "' must be a dictionary");
    return e->dict.get();
}

std::expected<double, string> needScalar(const Dict& d, const char* key)
{
    const Entry* e = d.find(key);
    if (!e) return Err(d.file + ": missing '" + key + "'");
    return foam::scalarOf(d, *e);
}

std::expected<string, string> needWord(const Dict& d, const char* key)
{
    const Entry* e = d.find(key);
    if (!e) return Err(d.file + ": missing '" + key + "'");
    return foam::wordOf(d, *e);
}

// `dimensions [a b c d e f g];` must equal want.
std::expected<void, string> checkDims(const Dict& d, const std::array<int, 7>& want, const char* field,
                                      const char* hint)
{
    const Entry* e = d.find("dimensions");
    if (!e) return Err(d.file + ": missing 'dimensions'");
    const auto& v = e->value;
    bool ok = v.size() == 9 && v[0].is('[') && v[8].is(']');
    for (int i = 0; ok && i < 7; ++i)
        ok = v[std::size_t(i + 1)].kind == Token::Number && v[std::size_t(i + 1)].num == want[std::size_t(i)];
    if (!ok)
        return Err(at(d, *e) + std::format("{} has dimensions '{}', the FV path needs {}", field, foam::joined(*e),
                                           hint));
    return {};
}

// `uniform X` | `nonuniform List<T> N (...)` with X a number (nc 1) or a vector (nc 3).
std::expected<FvValue, string> valueOf(const Dict& d, const Entry& e, int nc)
{
    const auto& t = e.value;
    FvValue out;
    auto bad = [&](const string& why) { return Err(at(d, e) + "'" + e.key + "': " + why); };
    if (t.empty()) return bad("no value");
    std::size_t pos = 1;
    auto one = [&](std::size_t& p) -> std::expected<void, string> {
        if (nc == 1) {
            if (p >= t.size() || t[p].kind != Token::Number) return bad("expects a number");
            out.v.push_back(t[p++].num);
            return {};
        }
        auto v = foam::vectorOf(d, t, p, e.line, e.key);
        if (!v) return Err(v.error());
        out.v.insert(out.v.end(), v->begin(), v->end());
        return {};
    };
    if (t[0].text == "uniform") {
        if (auto r = one(pos); !r) return Err(r.error());
        if (pos != t.size()) return bad("trailing tokens after the uniform value");
        return out;
    }
    if (t[0].text != "nonuniform") return bad("expects 'uniform' or 'nonuniform' (got '" + string(t[0].text) + "')");
    out.uniform = false;
    const char* type = nc == 1 ? "List<scalar>" : "List<vector>";
    if (t.size() < 2 || t[1].text != type) return bad(string("nonuniform needs ") + type);
    if (t.size() >= 3 && t[2].kind == Token::Word && t[2].text == "0()") {
        if (t.size() != 3) return bad("trailing tokens");
        return out;
    }
    if (t.size() < 5 || t[2].kind != Token::Number || !t[3].is('(')) return bad("malformed nonuniform list");
    const auto n = std::int64_t(t[2].num);
    pos = 4;
    for (std::int64_t i = 0; i < n; ++i)
        if (auto r = one(pos); !r) return Err(r.error());
    if (pos + 1 != t.size() || !t[pos].is(')')) return bad("nonuniform list length does not match its count");
    return out;
}

bool regexMatch(const string& pattern, const string& name)
{
    try {
        return std::regex_match(name, std::regex(pattern, std::regex::ECMAScript));
    } catch (const std::regex_error&) {
        return false;
    }
}

bool isConstraint(FoamPatchType t)
{
    return t == FoamPatchType::Empty || t == FoamPatchType::Symmetry || t == FoamPatchType::SymmetryPlane ||
           t == FoamPatchType::Cyclic || t == FoamPatchType::Wedge;
}

// The boundaryField entry of patch p (OpenFOAM precedence: exact name, then
// its group -- `wall` for wall patches, the constraint type name -- then the
// LAST matching regular expression), or the implicit constraint entry.
const Entry* bcEntry(const Dict& bf, const FoamPatch& p)
{
    if (const Entry* e = bf.find(p.name)) return e;
    if (p.type == FoamPatchType::Wall)
        if (const Entry* e = bf.find("wall")) return e;
    if (isConstraint(p.type))
        if (const Entry* e = bf.find(foamPatchTypeName(p.type))) return e;
    const Entry* hit = nullptr;
    for (const Entry& e : bf.entries)
        if (e.quotedKey && regexMatch(e.key, p.name)) hit = &e;
    return hit;
}

// One velocity BC.
std::expected<void, string> readUBc(const Dict& bf, const Entry& e, FvPatchBc& b)
{
    if (!e.dict) return Err(at(bf, e) + "'" + e.key + "' must be a dictionary");
    const Dict& d = *e.dict;
    auto type = needWord(d, "type");
    if (!type) return Err(type.error());
    b.uType = *type;
    const string where = "0/U patch '" + b.name + "' (" + *type + ")";
    auto value = [&](const char* key, bool required) -> std::expected<FvValue, string> {
        const Entry* v = d.find(key);
        if (!v) {
            if (required) return Err(d.file + ": " + where + " needs '" + key + "'");
            return FvValue{{}, true};
        }
        return valueOf(d, *v, 3);
    };
    const string& t = *type;
    if (isConstraint(b.meshType)) {
        const char* want = foamPatchTypeName(b.meshType);
        if (t != want) return Err(at(bf, e) + where + ": a " + want + " patch takes type " + want);
    }
    if (t == "fixedValue" || t == "movingWallVelocity") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        auto v = value("value", true);
        if (!v) return Err(v.error());
        b.u = FvUBc::Fixed;
        b.uValue = *v;
    } else if (t == "noSlip") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.u = FvUBc::Fixed;
        b.uValue = FvValue{{0, 0, 0}, true};
    } else if (t == "zeroGradient") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.u = FvUBc::ZeroGrad;
    } else if (t == "inletOutlet" || t == "freestream") {
        if (auto r = onlyKeys(d, {"type", "value", "inletValue", "freestreamValue", "phi"}, where); !r) return r;
        if (const Entry* ph = d.find("phi")) {
            auto w = foam::wordOf(d, *ph);
            if (!w) return Err(w.error());
            if (*w != "phi") return Err(at(d, *ph) + where + ": phi must be 'phi'");
        }
        auto v = value(t == "inletOutlet" ? "inletValue" : "freestreamValue", true);
        if (!v) return Err(v.error());
        auto i = value("value", false);
        if (!i) return Err(i.error());
        b.u = FvUBc::InletOutlet;
        b.uValue = *v;
        b.uInit = *i;
    } else if (t == "freestreamVelocity") {
        if (auto r = onlyKeys(d, {"type", "value", "freestreamValue"}, where); !r) return r;
        auto v = value("freestreamValue", true);
        if (!v) return Err(v.error());
        auto i = value("value", false);
        if (!i) return Err(i.error());
        b.u = FvUBc::Freestream;
        b.uValue = *v;
        b.uInit = *i;
    } else if (t == "slip" || t == "symmetry" || t == "symmetryPlane") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.u = FvUBc::Symmetry;
    } else if (t == "empty") {
        if (b.meshType != FoamPatchType::Empty) return Err(d.file + ": " + where + ": empty on a non-empty patch");
        b.u = FvUBc::Empty;
    } else if (t == "cyclic") {
        if (b.meshType != FoamPatchType::Cyclic) return Err(d.file + ": " + where + ": cyclic on a non-cyclic patch");
        b.u = FvUBc::Empty; // consumed: the FaceMesh merges cyclic pairs into internal faces
    } else {
        return Err(d.file + ": " + where + ": velocity BC '" + t +
                   "' is not supported (fixedValue, noSlip, movingWallVelocity, zeroGradient, inletOutlet, "
                   "freestream, freestreamVelocity, slip, symmetry, symmetryPlane, empty, cyclic)");
    }
    return {};
}

std::expected<void, string> readPBc(const Dict& bf, const Entry& e, FvPatchBc& b)
{
    if (!e.dict) return Err(at(bf, e) + "'" + e.key + "' must be a dictionary");
    const Dict& d = *e.dict;
    auto type = needWord(d, "type");
    if (!type) return Err(type.error());
    b.pType = *type;
    const string where = "0/p patch '" + b.name + "' (" + *type + ")";
    const string& t = *type;
    if (isConstraint(b.meshType)) {
        const char* want = foamPatchTypeName(b.meshType);
        if (t != want) return Err(d.file + ": " + where + ": a " + want + " patch takes type " + want);
    }
    if (t == "fixedValue") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        const Entry* v = d.find("value");
        if (!v) return Err(d.file + ": " + where + " needs 'value'");
        auto x = valueOf(d, *v, 1);
        if (!x) return Err(x.error());
        b.p = FvPBc::Fixed;
        b.pValue = *x;
    } else if (t == "zeroGradient" || t == "slip" || t == "symmetry" || t == "symmetryPlane") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.p = FvPBc::ZeroGrad;
    } else if (t == "freestreamPressure") {
        if (auto r = onlyKeys(d, {"type", "value", "freestreamValue", "supersonic", "U"}, where); !r) return r;
        if (const Entry* s = d.find("supersonic")) {
            auto sw = foam::switchOf(d, *s);
            if (!sw) return Err(sw.error());
            if (*sw) return Err(at(d, *s) + where + ": supersonic yes is not supported");
        }
        if (const Entry* u = d.find("U")) {
            auto w = foam::wordOf(d, *u);
            if (!w) return Err(w.error());
            if (*w != "U") return Err(at(d, *u) + where + ": U must be 'U'");
        }
        const Entry* v = d.find("freestreamValue");
        if (!v) return Err(d.file + ": " + where + " needs 'freestreamValue'");
        auto x = valueOf(d, *v, 1);
        if (!x) return Err(x.error());
        b.p = FvPBc::Freestream;
        b.pValue = *x;
        if (const Entry* i = d.find("value")) {
            auto y = valueOf(d, *i, 1);
            if (!y) return Err(y.error());
            b.pInit = *y;
        }
    } else if (t == "empty") {
        if (b.meshType != FoamPatchType::Empty) return Err(d.file + ": " + where + ": empty on a non-empty patch");
        b.p = FvPBc::Empty;
    } else if (t == "cyclic") {
        if (b.meshType != FoamPatchType::Cyclic) return Err(d.file + ": " + where + ": cyclic on a non-cyclic patch");
        b.p = FvPBc::Empty;
    } else {
        return Err(d.file + ": " + where + ": pressure BC '" + t +
                   "' is not supported (fixedValue, zeroGradient, freestreamPressure, slip, symmetry, "
                   "symmetryPlane, empty, cyclic)");
    }
    return {};
}

std::expected<Dict, string> parseText(const FoamCaseTexts& t, const string& rel, bool required, bool& present)
{
    const auto it = t.find(rel);
    present = it != t.end();
    if (!present) {
        if (required) return Err(rel + ": missing (required case file)");
        return Dict{};
    }
    return foam::parseDict(it->second, rel);
}

std::expected<void, string> readField(const Dict& d, const PolyManifest& man, bool velocity, FoamCase& c)
{
    const string fname = velocity ? "U" : "p";
    if (auto r = onlyKeys(d, {"dimensions", "internalField", "boundaryField", "referenceLevel"}, "0/" + fname);
        !r)
        return r;
    if (velocity) {
        if (auto r = checkDims(d, {0, 1, -1, 0, 0, 0, 0}, "U", "m/s [0 1 -1 0 0 0 0]"); !r) return r;
    } else {
        if (auto r = checkDims(d, {0, 2, -2, 0, 0, 0, 0}, "p",
                               "KINEMATIC pressure [0 2 -2 0 0 0 0] (incompressible simpleFoam; a "
                               "compressible case is not a P5b case)");
            !r)
            return r;
    }
    if (d.find("referenceLevel")) return Err(d.file + ": 'referenceLevel' is not supported");
    const Entry* in = d.find("internalField");
    if (!in) return Err(d.file + ": missing 'internalField'");
    auto iv = valueOf(d, *in, velocity ? 3 : 1);
    if (!iv) return Err(iv.error());
    if (!iv->uniform && std::int64_t(iv->v.size()) != man.nCells * (velocity ? 3 : 1))
        return Err(d.file + ": internalField has " + std::to_string(iv->v.size() / (velocity ? 3 : 1)) +
                   " values, the mesh " + std::to_string(man.nCells) + " cells");
    (velocity ? c.uInternal : c.pInternal) = *iv;
    auto bf = subDict(d, "boundaryField", true, "0/" + fname);
    if (!bf) return Err(bf.error());
    for (const Entry& e : (*bf)->entries) {
        if (!e.directive) continue;
        const bool constraints = e.key == "#includeEtc" && e.value.size() == 1 &&
                                 e.value[0].kind == Token::String &&
                                 e.value[0].text == "caseDicts/setConstraintTypes";
        if (!constraints)
            return Err(at(**bf, e) + "directive '" + e.key + " " + foam::joined(e) +
                       "' is not supported (only #includeEtc \"caseDicts/setConstraintTypes\")");
    }
    for (std::size_t i = 0; i < man.patches.size(); ++i) {
        const FoamPatch& p = man.patches[i];
        FvPatchBc& b = c.patches[i];
        const Entry* e = bcEntry(**bf, p);
        if (!e) {
            if (isConstraint(p.type)) { // the constraint type implied (setConstraintTypes)
                if (velocity) { b.u = FvUBc::Empty; b.uType = foamPatchTypeName(p.type); }
                else { b.p = FvPBc::Empty; b.pType = foamPatchTypeName(p.type); }
                if (p.type == FoamPatchType::Symmetry || p.type == FoamPatchType::SymmetryPlane) {
                    if (velocity) b.u = FvUBc::Symmetry;
                    else b.p = FvPBc::ZeroGrad;
                }
                continue;
            }
            return Err(d.file + ": no boundaryField entry for patch '" + p.name + "'");
        }
        auto r = velocity ? readUBc(**bf, *e, b) : readPBc(**bf, *e, b);
        if (!r) return r;
        const FvValue& v = velocity ? b.uValue : b.pValue;
        if (!v.uniform && std::int64_t(v.v.size()) != p.nFaces * (velocity ? 3 : 1))
            return Err(d.file + ": patch '" + p.name + "' nonuniform value has the wrong length");
    }
    return {};
}

// wallFunctionCoefficients keys (Cmu, kappa, E) of a wall-function BC.
std::expected<void, string> readWallCoeffs(const Dict& d, FvWallCoeffs& w)
{
    for (auto [key, dst] : {std::pair{"Cmu", &w.Cmu}, {"kappa", &w.kappa}, {"E", &w.E}})
        if (const Entry* x = d.find(key)) {
            auto v = foam::scalarOf(d, *x);
            if (!v) return Err(v.error());
            *dst = *v;
        }
    if (!(w.kappa > 0) || !(w.E > 0)) return Err(d.file + ": wall function kappa and E must be > 0");
    return {};
}

// wallFunctionBlenders: `blending` and `n` (defaults per BC type).
std::expected<void, string> readBlending(const Dict& d, FvBlend& b, double& n)
{
    if (const Entry* x = d.find("blending")) {
        auto w = foam::wordOf(d, *x);
        if (!w) return Err(w.error());
        static constexpr std::pair<const char*, FvBlend> names[] = {{"stepwise", FvBlend::Stepwise},
                                                                    {"max", FvBlend::Max},
                                                                    {"binomial", FvBlend::Binomial},
                                                                    {"exponential", FvBlend::Exponential},
                                                                    {"tanh", FvBlend::Tanh}};
        const auto it = std::ranges::find_if(names, [&](const auto& p) { return *w == p.first; });
        if (it == std::end(names))
            return Err(at(d, *x) + "blending '" + *w + "' (stepwise, max, binomial, exponential, tanh)");
        b = it->second;
    }
    if (const Entry* x = d.find("n")) {
        auto v = foam::scalarOf(d, *x);
        if (!v) return Err(v.error());
        if (!(*v >= 0)) return Err(at(d, *x) + "blending exponent n must be >= 0");
        n = *v;
    }
    return {};
}

// One turbulence-scalar BC (k / omega / nuTilda) on patch b.
std::expected<void, string> readScalarBc(const Dict& bf, const Entry& e, const FoamPatch& patch, const string& field,
                                         FvScalarBc& b)
{
    if (!e.dict) return Err(at(bf, e) + "'" + e.key + "' must be a dictionary");
    const Dict& d = *e.dict;
    auto type = needWord(d, "type");
    if (!type) return Err(type.error());
    b.type = *type;
    const string where = "0/" + field + " patch '" + patch.name + "' (" + *type + ")";
    const string& t = *type;
    auto value = [&](const char* key, bool required) -> std::expected<FvValue, string> {
        const Entry* v = d.find(key);
        if (!v) {
            if (required) return Err(d.file + ": " + where + " needs '" + key + "'");
            return FvValue{{}, true};
        }
        return valueOf(d, *v, 1);
    };
    if (isConstraint(patch.type)) {
        const char* want = foamPatchTypeName(patch.type);
        if (t != want) return Err(at(bf, e) + where + ": a " + want + " patch takes type " + want);
    }
    if (t == "fixedValue") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        auto v = value("value", true);
        if (!v) return Err(v.error());
        b.kind = FvSBc::Fixed;
        b.value = *v;
    } else if (t == "zeroGradient" || t == "kqRWallFunction") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        auto v = value("value", false);
        if (!v) return Err(v.error());
        b.kind = FvSBc::ZeroGrad;
        b.init = *v;
    } else if (t == "inletOutlet" || t == "freestream") {
        if (auto r = onlyKeys(d, {"type", "value", "inletValue", "freestreamValue", "phi"}, where); !r) return r;
        if (const Entry* ph = d.find("phi")) {
            auto w = foam::wordOf(d, *ph);
            if (!w) return Err(w.error());
            if (*w != "phi") return Err(at(d, *ph) + where + ": phi must be 'phi'");
        }
        auto v = value(t == "inletOutlet" ? "inletValue" : "freestreamValue", true);
        if (!v) return Err(v.error());
        auto i = value("value", false);
        if (!i) return Err(i.error());
        b.kind = FvSBc::InletOutlet;
        b.value = *v;
        b.init = *i;
    } else if (t == "omegaWallFunction") {
        if (field != "omega") return Err(d.file + ": " + where + ": omegaWallFunction is an omega BC");
        if (patch.type != FoamPatchType::Wall)
            return Err(d.file + ": " + where + ": omegaWallFunction needs a wall patch");
        if (auto r = onlyKeys(d, {"type", "value", "blending", "n", "beta1", "Cmu", "kappa", "E"}, where); !r)
            return r;
        auto v = value("value", true); // fixedValue parent: required (then extrapolateInternal)
        if (!v) return Err(v.error());
        b.kind = FvSBc::OmegaWall;
        b.init = *v;
        b.blend = FvBlend::Binomial; // wallFunctionBlenders(dict, BINOMIAL, 2)
        b.n = 2;
        if (auto r = readBlending(d, b.blend, b.n); !r) return r;
        if (const Entry* x = d.find("beta1")) {
            auto s = foam::scalarOf(d, *x);
            if (!s) return Err(s.error());
            b.beta1 = *s;
        }
        if (auto r = readWallCoeffs(d, b.wc); !r) return r;
    } else if (t == "slip" || t == "symmetry" || t == "symmetryPlane") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.kind = FvSBc::ZeroGrad;
    } else if (t == "empty" || t == "cyclic") {
        if (t == "empty" && patch.type != FoamPatchType::Empty)
            return Err(d.file + ": " + where + ": empty on a non-empty patch");
        if (t == "cyclic" && patch.type != FoamPatchType::Cyclic)
            return Err(d.file + ": " + where + ": cyclic on a non-cyclic patch");
        b.kind = FvSBc::Empty;
    } else {
        return Err(d.file + ": " + where + ": " + field + " BC '" + t +
                   "' is not supported (fixedValue, zeroGradient, kqRWallFunction, inletOutlet, freestream, "
                   "omegaWallFunction, slip, symmetry, symmetryPlane, empty, cyclic)");
    }
    const FvValue& v = b.kind == FvSBc::Fixed || b.kind == FvSBc::InletOutlet ? b.value : b.init;
    if (!v.uniform && !v.v.empty() && std::int64_t(v.v.size()) != std::int64_t(patch.nFaces))
        return Err(d.file + ": patch '" + patch.name + "' nonuniform value has the wrong length");
    return {};
}

std::expected<void, string> readNutBc(const Dict& bf, const Entry& e, const FoamPatch& patch, FvNutPatch& b)
{
    if (!e.dict) return Err(at(bf, e) + "'" + e.key + "' must be a dictionary");
    const Dict& d = *e.dict;
    auto type = needWord(d, "type");
    if (!type) return Err(type.error());
    b.type = *type;
    const string where = "0/nut patch '" + patch.name + "' (" + *type + ")";
    const string& t = *type;
    if (isConstraint(patch.type)) {
        const char* want = foamPatchTypeName(patch.type);
        if (t != want) return Err(at(bf, e) + where + ": a " + want + " patch takes type " + want);
    }
    auto value = [&](bool required) -> std::expected<FvValue, string> {
        const Entry* v = d.find("value");
        if (!v) {
            if (required) return Err(d.file + ": " + where + " needs 'value'");
            return FvValue{{}, true};
        }
        return valueOf(d, *v, 1);
    };
    const bool wallFn = t == "nutLowReWallFunction" || t == "nutUSpaldingWallFunction" || t == "nutkWallFunction";
    if (wallFn && patch.type != FoamPatchType::Wall)
        return Err(d.file + ": " + where + ": a nut wall function needs a wall patch (OpenFOAM checkType)");
    if (t == "calculated") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.kind = FvNutBc::Calculated;
    } else if (t == "fixedValue") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.kind = FvNutBc::Fixed;
    } else if (t == "zeroGradient" || t == "slip" || t == "symmetry" || t == "symmetryPlane") {
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.kind = FvNutBc::Cell;
    } else if (t == "nutLowReWallFunction") {
        if (auto r = onlyKeys(d, {"type", "value", "Cmu", "kappa", "E", "U"}, where); !r) return r;
        b.kind = FvNutBc::LowRe;
    } else if (t == "nutUSpaldingWallFunction") {
        if (auto r = onlyKeys(d, {"type", "value", "Cmu", "kappa", "E", "maxIter", "tolerance", "U"}, where); !r)
            return r;
        b.kind = FvNutBc::Spalding;
        if (const Entry* x = d.find("maxIter")) {
            auto v = foam::scalarOf(d, *x);
            if (!v) return Err(v.error());
            if (*v < 1 || *v != std::floor(*v)) return Err(at(d, *x) + where + ": maxIter must be an integer >= 1");
            b.maxIter = int(*v);
        }
        if (const Entry* x = d.find("tolerance")) {
            auto v = foam::scalarOf(d, *x);
            if (!v) return Err(v.error());
            // A tolerance other than 0.01 takes calcNut's second pass (keep
            // the old value where err < tolerance): not reproduced.
            if (std::abs(*v - 0.01) > 1e-300)
                return Err(at(d, *x) + where + ": tolerance other than 0.01 is not supported");
            b.tolerance = *v;
        }
    } else if (t == "nutkWallFunction") {
        if (auto r = onlyKeys(d, {"type", "value", "Cmu", "kappa", "E", "blending", "n", "U"}, where); !r) return r;
        b.kind = FvNutBc::KWall;
        b.blend = FvBlend::Stepwise; // wallFunctionBlenders(dict, STEPWISE, 4)
        b.n = 4;
        if (auto r = readBlending(d, b.blend, b.n); !r) return r;
    } else if (t == "empty" || t == "cyclic") {
        b.kind = FvNutBc::Empty;
    } else {
        return Err(d.file + ": " + where + ": nut BC '" + t +
                   "' is not supported (calculated, fixedValue, zeroGradient, nutLowReWallFunction, "
                   "nutUSpaldingWallFunction, nutkWallFunction, slip, symmetry, symmetryPlane, empty, cyclic)");
    }
    if (const Entry* u = d.find("U")) {
        auto w = foam::wordOf(d, *u);
        if (!w) return Err(w.error());
        if (*w != "U") return Err(at(d, *u) + where + ": U must be 'U'");
    }
    if (wallFn)
        if (auto r = readWallCoeffs(d, b.wc); !r) return r;
    if (b.kind != FvNutBc::Empty && b.kind != FvNutBc::Cell) {
        auto v = value(true); // fixedValue-derived and calculated patches read `value`
        if (!v) return Err(v.error());
        b.value = *v;
        if (!b.value.uniform && std::int64_t(b.value.v.size()) != std::int64_t(patch.nFaces))
            return Err(d.file + ": patch '" + patch.name + "' nonuniform value has the wrong length");
    }
    return {};
}

// 0/<field> of a turbulence scalar: internalField and one BC per patch.
std::expected<void, string> readTurbField(const Dict& d, const PolyManifest& man, const string& field,
                                          const std::array<int, 7>& dims, const char* dimsText, FoamCase& c)
{
    if (auto r = onlyKeys(d, {"dimensions", "internalField", "boundaryField"}, "0/" + field); !r) return r;
    if (auto r = checkDims(d, dims, field.c_str(), dimsText); !r) return r;
    const Entry* in = d.find("internalField");
    if (!in) return Err(d.file + ": missing 'internalField'");
    auto iv = valueOf(d, *in, 1);
    if (!iv) return Err(iv.error());
    if (!iv->uniform && std::int64_t(iv->v.size()) != man.nCells)
        return Err(d.file + ": internalField has " + std::to_string(iv->v.size()) + " values, the mesh " +
                   std::to_string(man.nCells) + " cells");
    FvRas& r = c.ras;
    (field == "k" ? r.kInt : field == "omega" ? r.omegaInt : field == "nut" ? r.nutInt : r.nuTildaInt) = *iv;
    auto bf = subDict(d, "boundaryField", true, "0/" + field);
    if (!bf) return Err(bf.error());
    for (const Entry& e : (*bf)->entries) {
        if (!e.directive) continue;
        const bool constraints = e.key == "#includeEtc" && e.value.size() == 1 &&
                                 e.value[0].kind == Token::String &&
                                 e.value[0].text == "caseDicts/setConstraintTypes";
        if (!constraints)
            return Err(at(**bf, e) + "directive '" + e.key + " " + foam::joined(e) +
                       "' is not supported (only #includeEtc \"caseDicts/setConstraintTypes\")");
    }
    if (field == "nut") r.nut.assign(man.patches.size(), FvNutPatch{});
    else (field == "k" ? r.k : field == "omega" ? r.omega : r.nuTilda).assign(man.patches.size(), FvScalarBc{});
    for (std::size_t i = 0; i < man.patches.size(); ++i) {
        const FoamPatch& p = man.patches[i];
        const Entry* e = bcEntry(**bf, p);
        if (!e) {
            if (!isConstraint(p.type)) return Err(d.file + ": no boundaryField entry for patch '" + p.name + "'");
            const bool sym = p.type == FoamPatchType::Symmetry || p.type == FoamPatchType::SymmetryPlane;
            if (field == "nut") {
                r.nut[i].kind = sym ? FvNutBc::Cell : FvNutBc::Empty;
                r.nut[i].type = foamPatchTypeName(p.type);
            } else {
                auto& b = (field == "k" ? r.k : field == "omega" ? r.omega : r.nuTilda)[i];
                b.kind = sym ? FvSBc::ZeroGrad : FvSBc::Empty;
                b.type = foamPatchTypeName(p.type);
            }
            continue;
        }
        if (field == "nut") {
            if (auto x = readNutBc(**bf, *e, p, r.nut[i]); !x) return x;
        } else {
            auto& b = (field == "k" ? r.k : field == "omega" ? r.omega : r.nuTilda)[i];
            if (auto x = readScalarBc(**bf, *e, p, field, b); !x) return x;
        }
    }
    return {};
}

// turbulenceProperties `RAS` dictionary.
std::expected<void, string> readRasDict(const Dict& d, const Entry& rasE, FoamCase& c)
{
    if (!rasE.dict) return Err(at(d, rasE) + "'RAS' must be a dictionary");
    const Dict& r = *rasE.dict;
    if (auto x = onlyKeys(r, {"RASModel", "model", "turbulence", "printCoeffs", "kOmegaSSTCoeffs",
                              "SpalartAllmarasCoeffs", "kMin", "omegaMin", "epsilonMin"},
                          "RAS");
        !x)
        return x;
    const Entry* m = r.find("RASModel");
    if (!m) m = r.find("model");
    if (!m) return Err(r.file + ": RAS needs 'RASModel'");
    auto name = foam::wordOf(r, *m);
    if (!name) return Err(name.error());
    FvRas& ras = c.ras;
    if (*name == "kOmegaSST") ras.model = FvTurbModel::KOmegaSST;
    else if (*name == "SpalartAllmaras") ras.model = FvTurbModel::SpalartAllmaras;
    else
        return Err(at(r, *m) + "RASModel '" + *name + "' is not supported (kOmegaSST, SpalartAllmaras)");
    if (const Entry* t = r.find("turbulence")) {
        auto v = foam::switchOf(r, *t);
        if (!v) return Err(v.error());
        if (!*v) return Err(at(r, *t) + "turbulence off is not supported (the model would freeze nut)");
    }
    for (auto [key, dst] : {std::pair{"kMin", &ras.kMin}, {"omegaMin", &ras.omegaMin}})
        if (const Entry* x = r.find(key)) {
            auto v = foam::scalarOf(r, *x);
            if (!v) return Err(v.error());
            *dst = *v;
        }
    const bool sst = ras.model == FvTurbModel::KOmegaSST;
    if (const Entry* cd = r.find(sst ? "kOmegaSSTCoeffs" : "SpalartAllmarasCoeffs")) {
        if (!cd->dict) return Err(at(r, *cd) + "coefficients must be a dictionary");
        const Dict& k = *cd->dict;
        if (sst) {
            if (auto x = onlyKeys(k, {"alphaK1", "alphaK2", "alphaOmega1", "alphaOmega2", "gamma1", "gamma2", "beta1",
                                      "beta2", "betaStar", "a1", "b1", "c1", "F3", "decayControl"},
                                  "kOmegaSSTCoeffs");
                !x)
                return x;
            const std::pair<const char*, double*> sc[] = {
                {"alphaK1", &ras.alphaK1}, {"alphaK2", &ras.alphaK2}, {"alphaOmega1", &ras.alphaOmega1},
                {"alphaOmega2", &ras.alphaOmega2}, {"gamma1", &ras.gamma1}, {"gamma2", &ras.gamma2},
                {"beta1", &ras.beta1}, {"beta2", &ras.beta2}, {"betaStar", &ras.betaStar}, {"a1", &ras.a1},
                {"b1", &ras.b1}, {"c1", &ras.c1}};
            for (auto [key, dst] : sc)
                if (const Entry* x = k.find(key)) {
                    auto v = foam::scalarOf(k, *x);
                    if (!v) return Err(v.error());
                    *dst = *v;
                }
            if (const Entry* x = k.find("F3")) {
                auto v = foam::switchOf(k, *x);
                if (!v) return Err(v.error());
                ras.F3 = *v;
            }
            if (const Entry* x = k.find("decayControl")) {
                auto v = foam::switchOf(k, *x);
                if (!v) return Err(v.error());
                if (*v) return Err(at(k, *x) + "decayControl yes is not supported");
            }
        } else {
            if (auto x = onlyKeys(k, {"sigmaNut", "kappa", "Cb1", "Cb2", "Cw2", "Cw3", "Cv1", "Cs", "ck", "ft2", "Ct3",
                                      "Ct4"},
                                  "SpalartAllmarasCoeffs");
                !x)
                return x;
            const std::pair<const char*, double*> sc[] = {
                {"sigmaNut", &ras.sigmaNut}, {"kappa", &ras.kappa}, {"Cb1", &ras.Cb1}, {"Cb2", &ras.Cb2},
                {"Cw2", &ras.Cw2}, {"Cw3", &ras.Cw3}, {"Cv1", &ras.Cv1}, {"Cs", &ras.Cs}, {"Ct3", &ras.Ct3},
                {"Ct4", &ras.Ct4}};
            for (auto [key, dst] : sc)
                if (const Entry* x = k.find(key)) {
                    auto v = foam::scalarOf(k, *x);
                    if (!v) return Err(v.error());
                    *dst = *v;
                }
            if (const Entry* x = k.find("ft2")) {
                auto v = foam::switchOf(k, *x);
                if (!v) return Err(v.error());
                ras.ft2 = *v;
            }
        }
    }
    return {};
}

// A turbulence scalar's div scheme: [bounded] Gauss linear | upwind |
// linearUpwind grad(<field>) | limitedLinear k.
std::expected<FvConvScheme, string> convScheme(const Dict& d, const string& field, const string& s)
{
    FvConvScheme c;
    string rest = s;
    c.bounded = rest.starts_with("bounded ");
    if (c.bounded) rest = rest.substr(8);
    if (rest == "Gauss linear") c.kind = FvSDiv::Linear;
    else if (rest == "Gauss upwind") c.kind = FvSDiv::Upwind;
    else if (rest == "Gauss linearUpwind grad(" + field + ")") c.kind = FvSDiv::LinearUpwind;
    else if (rest.starts_with("Gauss limitedLinear ")) {
        double k = -1;
        const string num = rest.substr(20);
        const auto r = std::from_chars(num.data(), num.data() + num.size(), k);
        if (r.ec != std::errc{} || r.ptr != num.data() + num.size() || !(k >= 0 && k <= 1))
            return Err(d.file + ": div(phi," + field + ") '" + s + "': limitedLinear needs a coefficient in [0, 1]");
        c.kind = FvSDiv::LimitedLinear;
        c.k = k;
    } else
        return Err(d.file + ": div(phi," + field + ") '" + s + "' is not supported ([bounded] Gauss linear | upwind | "
                   "linearUpwind grad(" + field + ") | limitedLinear K)");
    return c;
}

std::expected<FvGradScheme, string> gradScheme(const Dict& d, const Entry& e)
{
    const string s = foam::joined(e);
    if (s == "Gauss linear") return FvGradScheme{};
    const auto& t = e.value;
    if (t.size() == 4 && t[0].text == "cellLimited" && t[1].text == "Gauss" && t[2].text == "linear" &&
        t[3].kind == Token::Number && t[3].num >= 0 && t[3].num <= 1)
        return FvGradScheme{FvGrad::CellLimited, t[3].num};
    return Err(at(d, e) + "gradient scheme '" + s + "' is not supported (Gauss linear, cellLimited Gauss linear K)");
}

// An snGrad scheme in tokens t[i..]: corrected | uncorrected | limited
// [corrected] psi (limitedSnGrad, psi in [0, 1]). {corrected, limit (< 0: none)}.
struct SnGradSpec
{
    bool corrected = true;
    double limit = -1;
    bool operator==(const SnGradSpec&) const = default;
};

std::optional<SnGradSpec> snGradSpec(const std::vector<Token>& t, std::size_t i)
{
    const std::size_t n = t.size() - i;
    if (n == 1 && t[i].text == "corrected") return SnGradSpec{true, -1};
    if (n == 1 && t[i].text == "uncorrected") return SnGradSpec{false, -1};
    if (t[i].text != "limited" || (n != 2 && n != 3)) return std::nullopt;
    if (n == 3 && t[i + 1].text != "corrected") return std::nullopt;
    const Token& x = t.back();
    if (x.kind != Token::Number || !(x.num >= 0 && x.num <= 1)) return std::nullopt;
    return SnGradSpec{true, x.num};
}

std::string snGradName(bool corrected, double limit)
{
    if (limit >= 0) return std::format("limited corrected {:g}", limit);
    return corrected ? "corrected" : "uncorrected";
}

std::expected<void, string> readSchemes(const Dict& d, FoamCase& c)
{
    if (auto r = onlyKeys(d, {"ddtSchemes", "gradSchemes", "divSchemes", "laplacianSchemes", "interpolationSchemes",
                              "snGradSchemes", "wallDist", "fluxRequired"},
                          "fvSchemes");
        !r)
        return r;
    auto ddt = subDict(d, "ddtSchemes", true, "fvSchemes");
    if (!ddt) return Err(ddt.error());
    {
        const Entry* e = (*ddt)->find("default");
        if (!e || foam::joined(*e) != "steadyState")
            return Err(d.file + ": ddtSchemes default must be steadyState (SIMPLE; transient PISO/PIMPLE is not built)");
    }
    auto gr = subDict(d, "gradSchemes", true, "fvSchemes");
    if (!gr) return Err(gr.error());
    std::optional<FvGradScheme> def;
    if (const Entry* e = (*gr)->find("default"); e && foam::joined(*e) != "none") {
        auto g = gradScheme(**gr, *e);
        if (!g) return Err(g.error());
        def = *g;
    }
    for (const char* key : {"grad(U)", "grad(p)"}) {
        const Entry* e = (*gr)->find(key);
        std::optional<FvGradScheme> g = def;
        if (e) {
            auto x = gradScheme(**gr, *e);
            if (!x) return Err(x.error());
            g = *x;
        }
        if (!g) return Err(d.file + ": gradSchemes needs '" + key + "' or a default");
        (string(key) == "grad(U)" ? c.schemes.gradU : c.schemes.gradP) = *g;
    }
    // RAS: the turbulence scalars' gradient (linearUpwind / limitedLinear /
    // CDkOmega / SA's Cb2 term), convection, and the wall distance.
    FvRas& ras = c.ras;
    std::vector<string> turbFields;
    if (ras.model == FvTurbModel::KOmegaSST) turbFields = {"k", "omega"};
    if (ras.model == FvTurbModel::SpalartAllmaras) turbFields = {"nuTilda"};
    for (const string& f : turbFields) {
        const Entry* e = (*gr)->find("grad(" + f + ")");
        std::optional<FvGradScheme> g = def;
        if (e) {
            auto x = gradScheme(**gr, *e);
            if (!x) return Err(x.error());
            g = *x;
        }
        if (!g) return Err(d.file + ": gradSchemes needs 'grad(" + f + ")' or a default");
        (f == "k" ? ras.gradK : f == "omega" ? ras.gradOmega : ras.gradNuTilda) = *g;
    }
    auto dv = subDict(d, "divSchemes", true, "fvSchemes");
    if (!dv) return Err(dv.error());
    {
        const Entry* def2 = (*dv)->find("default");
        const string defs = def2 ? foam::joined(*def2) : "none";
        const Entry* e = (*dv)->find("div(phi,U)");
        const string s = e ? foam::joined(*e) : defs;
        if (s == "none") return Err(d.file + ": divSchemes needs 'div(phi,U)'");
        string rest = s;
        c.schemes.bounded = rest.starts_with("bounded ");
        if (c.schemes.bounded) rest = rest.substr(8);
        if (rest == "Gauss linear") c.schemes.divU = FvDiv::Linear;
        else if (rest == "Gauss linearUpwind grad(U)") c.schemes.divU = FvDiv::LinearUpwind;
        else if (rest == "Gauss upwind") c.schemes.divU = FvDiv::Upwind;
        else
            return Err(d.file + ": div(phi,U) '" + s +
                       "' is not supported ([bounded] Gauss linear | linearUpwind grad(U) | upwind)");
        const Entry* t = (*dv)->find("div((nuEff*dev2(T(grad(U)))))");
        const string ts = t ? foam::joined(*t) : defs;
        if (ts != "Gauss linear")
            return Err(d.file + ": div((nuEff*dev2(T(grad(U))))) must be 'Gauss linear' (got '" + ts + "')");
        for (const string& f : turbFields) {
            const Entry* x = (*dv)->find("div(phi," + f + ")");
            const string xs = x ? foam::joined(*x) : defs;
            if (xs == "none") return Err(d.file + ": divSchemes needs 'div(phi," + f + ")'");
            auto cs = convScheme(**dv, f, xs);
            if (!cs) return Err(cs.error());
            (f == "k" ? ras.divK : f == "omega" ? ras.divOmega : ras.divNuTilda) = *cs;
        }
    }
    if (ras.model != FvTurbModel::Laminar) {
        auto wd = subDict(d, "wallDist", true, "fvSchemes (RAS needs the wall distance method)");
        if (!wd) return Err(wd.error());
        if (auto r = onlyKeys(**wd, {"method", "correctWalls", "nRequired", "updateInterval"}, "wallDist"); !r)
            return r;
        auto m = needWord(**wd, "method");
        if (!m) return Err(m.error());
        if (*m == "meshWave") ras.exactDist = false;
        else if (*m == "exactDistance") ras.exactDist = true;
        else return Err(d.file + ": wallDist method '" + *m + "' is not supported (meshWave, exactDistance)");
        if (const Entry* x = (*wd)->find("correctWalls")) {
            auto v = foam::switchOf(**wd, *x);
            if (!v) return Err(v.error());
            ras.correctWalls = *v;
        }
        if (const Entry* x = (*wd)->find("nRequired")) {
            auto v = foam::switchOf(**wd, *x);
            if (!v) return Err(v.error());
            if (*v) return Err(at(**wd, *x) + "wallDist nRequired yes is not supported");
        }
    }
    auto lap = subDict(d, "laplacianSchemes", true, "fvSchemes");
    if (!lap) return Err(lap.error());
    {
        std::optional<SnGradSpec> corr;
        for (const Entry& e : (*lap)->entries) {
            if (e.key == "FoamFile") continue;
            const string s = foam::joined(e);
            const auto& t = e.value;
            std::optional<SnGradSpec> cc;
            if (t.size() >= 3 && t[0].text == "Gauss" && t[1].text == "linear") cc = snGradSpec(t, 2);
            if (!cc)
                return Err(at(**lap, e) + "laplacian scheme '" + s +
                           "' is not supported (Gauss linear corrected | uncorrected | limited [corrected] psi)");
            if (corr && *corr != *cc)
                return Err(at(**lap, e) + "laplacian schemes must agree (one correction for U and p)");
            corr = cc;
        }
        if (!corr) return Err(d.file + ": laplacianSchemes needs a default");
        c.schemes.lapCorrected = corr->corrected;
        c.schemes.lapLimit = corr->limit;
    }
    auto ip = subDict(d, "interpolationSchemes", true, "fvSchemes");
    if (!ip) return Err(ip.error());
    for (const Entry& e : (*ip)->entries)
        if (e.key != "FoamFile" && foam::joined(e) != "linear")
            return Err(at(**ip, e) + "interpolation scheme '" + foam::joined(e) + "' is not supported (linear)");
    auto sn = subDict(d, "snGradSchemes", true, "fvSchemes");
    if (!sn) return Err(sn.error());
    {
        const Entry* e = (*sn)->find("default");
        const string s = e ? foam::joined(*e) : "";
        const std::optional<SnGradSpec> g = e && !e->value.empty() ? snGradSpec(e->value, 0) : std::nullopt;
        if (!g)
            return Err(d.file + ": snGradSchemes default must be corrected | uncorrected | limited [corrected] psi "
                                "(got '" + s + "')");
        c.schemes.snGradCorrected = g->corrected;
        c.schemes.snGradLimit = g->limit;
        for (const Entry& x : (*sn)->entries)
            if (x.key != "FoamFile" && x.key != "default")
                return Err(at(**sn, x) + "per-field snGrad scheme '" + x.key + "' is not supported");
    }
    return {};
}

// The solver dict for field `name` (exact key, else the last matching regex).
const Entry* solverEntry(const Dict& s, const string& name)
{
    if (const Entry* e = s.find(name)) return e;
    const Entry* hit = nullptr;
    for (const Entry& e : s.entries)
        if (e.quotedKey && regexMatch(e.key, name)) hit = &e;
    return hit;
}

std::expected<FvSolverCtl, string> readSolver(const Dict& s, const string& name, bool pressure)
{
    const Entry* e = solverEntry(s, name);
    if (!e) return Err(s.file + ": solvers has no entry for '" + name + "'");
    if (!e->dict) return Err(at(s, *e) + "solver '" + name + "' must be a dictionary");
    const Dict& d = *e->dict;
    const string where = "solvers/" + name;
    if (auto r = onlyKeys(d, {"solver", "smoother", "preconditioner", "tolerance", "relTol", "maxIter", "minIter",
                              "nSweeps", "nCellsInCoarsestLevel", "cacheAgglomeration", "agglomerator",
                              "mergeLevels", "nPreSweeps", "nPostSweeps", "nFinestSweeps", "directSolveCoarsest",
                              "nVcycles", "scaleCorrection", "maxPreSweeps", "maxPostSweeps",
                              "preSweepsLevelMultiplier", "postSweepsLevelMultiplier", "interpolateCorrection",
                              "processorAgglomerator", "log"},
                          where);
        !r)
        return Err(r.error());
    FvSolverCtl c;
    auto sv = needWord(d, "solver");
    if (!sv) return Err(sv.error());
    c.solver = *sv;
    const bool ok = pressure ? (c.solver == "GAMG" || c.solver == "PCG" || c.solver == "PPCG" || c.solver == "PPCR")
                             : (c.solver == "smoothSolver" || c.solver == "PBiCGStab" || c.solver == "PBiCG" ||
                                c.solver == "GAMG");
    if (!ok)
        return Err(d.file + ": " + where + ": solver '" + c.solver + "' is not supported (" +
                   (pressure ? "GAMG, PCG, PPCG, PPCR" : "smoothSolver, PBiCGStab, PBiCG, GAMG") + ")");
    if (const Entry* x = d.find("tolerance")) {
        auto v = foam::scalarOf(d, *x);
        if (!v) return Err(v.error());
        c.tolerance = *v;
    }
    if (const Entry* x = d.find("relTol")) {
        auto v = foam::scalarOf(d, *x);
        if (!v) return Err(v.error());
        c.relTol = *v;
    }
    if (const Entry* x = d.find("maxIter")) {
        auto v = foam::scalarOf(d, *x);
        if (!v) return Err(v.error());
        c.maxIter = int(*v);
    }
    if (const Entry* x = d.find("minIter")) {
        auto v = foam::scalarOf(d, *x);
        if (!v) return Err(v.error());
        c.minIter = int(*v);
    }
    if (c.tolerance < 0 || c.relTol < 0 || c.maxIter < 0 || c.minIter < 0)
        return Err(d.file + ": " + where + ": negative tolerance/relTol/maxIter/minIter");
    return c;
}

// The relaxation factor of `name` in relaxationFactors/<sub> (exact, else regex).
std::expected<std::optional<double>, string> relaxOf(const Dict& r, const char* sub, const string& name)
{
    auto sd = subDict(r, sub, false, "relaxationFactors");
    if (!sd) return Err(sd.error());
    if (!*sd) return std::optional<double>{};
    const Entry* e = solverEntry(**sd, name);
    if (!e) return std::optional<double>{};
    auto v = foam::scalarOf(**sd, *e);
    if (!v) return Err(v.error());
    if (!(*v > 0 && *v <= 1)) return Err(at(**sd, *e) + "relaxation factor must lie in (0, 1]");
    return std::optional<double>{*v};
}

std::expected<void, string> readSolution(const Dict& d, const PolyManifest& man, FoamCase& c)
{
    if (auto r = onlyKeys(d, {"solvers", "SIMPLE", "relaxationFactors", "cache", "PIMPLE", "PISO",
                              "potentialFlow"},
                          "fvSolution");
        !r)
        return r;
    auto sv = subDict(d, "solvers", true, "fvSolution");
    if (!sv) return Err(sv.error());
    auto su = readSolver(**sv, "U", false);
    if (!su) return Err(su.error());
    c.solverU = *su;
    auto sp = readSolver(**sv, "p", true);
    if (!sp) return Err(sp.error());
    c.solverP = *sp;
    FvRas& ras = c.ras;
    const bool sst = ras.model == FvTurbModel::KOmegaSST, sa = ras.model == FvTurbModel::SpalartAllmaras;
    if (sst) {
        auto k = readSolver(**sv, "k", false);
        if (!k) return Err(k.error());
        ras.solverK = *k;
        auto o = readSolver(**sv, "omega", false);
        if (!o) return Err(o.error());
        ras.solverOmega = *o;
    }
    if (sa) {
        auto n = readSolver(**sv, "nuTilda", false);
        if (!n) return Err(n.error());
        ras.solverNuTilda = *n;
    }
    auto si = subDict(d, "SIMPLE", true, "fvSolution");
    if (!si) return Err(si.error());
    const Dict& s = **si;
    if (auto r = onlyKeys(s, {"nNonOrthogonalCorrectors", "consistent", "momentumPredictor", "pRefCell",
                              "pRefValue", "residualControl", "transonic"},
                          "SIMPLE");
        !r)
        return r;
    if (const Entry* e = s.find("nNonOrthogonalCorrectors")) {
        auto v = foam::scalarOf(s, *e);
        if (!v) return Err(v.error());
        if (*v < 0 || *v != std::floor(*v)) return Err(at(s, *e) + "nNonOrthogonalCorrectors must be an integer >= 0");
        c.simple.nNonOrthCorr = int(*v);
    }
    if (const Entry* e = s.find("consistent")) {
        auto v = foam::switchOf(s, *e);
        if (!v) return Err(v.error());
        c.simple.consistent = *v;
    }
    if (const Entry* e = s.find("momentumPredictor")) {
        auto v = foam::switchOf(s, *e);
        if (!v) return Err(v.error());
        c.simple.momentumPredictor = *v;
    }
    if (const Entry* e = s.find("transonic")) {
        auto v = foam::switchOf(s, *e);
        if (!v) return Err(v.error());
        if (*v) return Err(at(s, *e) + "transonic yes is a compressible (rhoSimpleFoam) setting");
    }
    if (const Entry* e = s.find("pRefCell")) {
        auto v = foam::scalarOf(s, *e);
        if (!v) return Err(v.error());
        if (*v < 0 || *v >= double(man.nCells) || *v != std::floor(*v))
            return Err(at(s, *e) + "pRefCell out of range [0, nCells)");
        c.simple.pRefCell = std::int64_t(*v);
        c.simple.pRefGiven = true;
    }
    if (const Entry* e = s.find("pRefValue")) {
        auto v = foam::scalarOf(s, *e);
        if (!v) return Err(v.error());
        c.simple.pRefValue = *v;
    }
    if (const Entry* e = s.find("residualControl")) {
        if (!e->dict) return Err(at(s, *e) + "residualControl must be a dictionary");
        for (const Entry& x : e->dict->entries) {
            if (x.key == "FoamFile") continue;
            auto v = foam::scalarOf(*e->dict, x);
            if (!v) return Err(v.error());
            if (x.key == "p" || (x.quotedKey && regexMatch(x.key, "p"))) c.simple.resP = *v;
            if (x.key == "U" || (x.quotedKey && regexMatch(x.key, "U"))) c.simple.resU = *v;
            for (auto [name, dst] : {std::pair{"k", &ras.resK}, {"omega", &ras.resOmega}, {"nuTilda", &ras.resNuTilda}})
                if (x.key == name || (x.quotedKey && regexMatch(x.key, name))) *dst = *v;
        }
        if (!sst) ras.resK = ras.resOmega = std::nullopt;
        if (!sa) ras.resNuTilda = std::nullopt;
    }
    auto rf = subDict(d, "relaxationFactors", false, "fvSolution");
    if (!rf) return Err(rf.error());
    if (*rf) {
        if (auto r = onlyKeys(**rf, {"fields", "equations"}, "relaxationFactors (use fields {} / equations {})");
            !r)
            return r;
        auto rp = relaxOf(**rf, "fields", "p");
        if (!rp) return Err(rp.error());
        if (*rp) {
            c.relaxP = **rp;
            c.relaxPGiven = true;
        }
        auto ru = relaxOf(**rf, "equations", "U");
        if (!ru) return Err(ru.error());
        c.relaxU = *ru;
        auto rpe = relaxOf(**rf, "equations", "p");
        if (!rpe) return Err(rpe.error());
        if (*rpe) return Err(d.file + ": relaxationFactors/equations/p (equation relaxation of p) is not supported");
        auto rfu = relaxOf(**rf, "fields", "U");
        if (!rfu) return Err(rfu.error());
        if (*rfu) return Err(d.file + ": relaxationFactors/fields/U (field relaxation of U) is not supported");
        std::vector<std::pair<const char*, std::optional<double>*>> tf;
        if (sst) tf = {{"k", &ras.relaxK}, {"omega", &ras.relaxOmega}};
        if (sa) tf = {{"nuTilda", &ras.relaxNuTilda}};
        for (auto [name, dst] : tf) {
            auto re = relaxOf(**rf, "equations", name);
            if (!re) return Err(re.error());
            *dst = *re;
            auto rfx = relaxOf(**rf, "fields", name);
            if (!rfx) return Err(rfx.error());
            if (*rfx)
                return Err(d.file + ": relaxationFactors/fields/" + name + " (field relaxation) is not supported");
        }
    }
    if (d.find("PIMPLE") || d.find("PISO")) c.notes.push_back("fvSolution PIMPLE/PISO dictionaries ignored (SIMPLE run)");
    return {};
}

std::expected<void, string> readForces(const Dict& d, const string& name, const PolyManifest& man, FoamCase& c)
{
    if (auto r = onlyKeys(d, {"type", "libs", "writeControl", "writeInterval", "executeControl", "executeInterval",
                              "patches", "rho", "rhoInf", "CofR", "origin", "pRef", "log", "porosity", "enabled",
                              "e1", "e3"},
                          "functions/" + name);
        !r)
        return r;
    FvForces f;
    f.name = name;
    const Entry* p = d.find("patches");
    if (!p) return Err(d.file + ": functions/" + name + " needs 'patches'");
    for (const Token& t : p->value) {
        if (t.kind == Token::Punct) continue;
        string w(t.text);
        const auto lp = w.find('(');
        if (lp != string::npos) w = w.substr(lp + 1); // "1(propeller)" style
        while (!w.empty() && w.back() == ')') w.pop_back();
        if (w.empty()) continue;
        bool hit = false;
        for (const FoamPatch& mp : man.patches)
            if (mp.name == w || ((w.front() == '"' || w.find_first_of("(|.*") != string::npos) && regexMatch(w, mp.name))) {
                if (std::ranges::find(f.patches, mp.name) == f.patches.end()) f.patches.push_back(mp.name);
                hit = true;
            }
        if (!hit) return Err(at(d, *p) + "functions/" + name + ": patch '" + w + "' is not in the mesh");
    }
    if (const Entry* r = d.find("rho")) {
        auto w = foam::wordOf(d, *r);
        if (!w) return Err(w.error());
        if (*w != "rhoInf") return Err(at(d, *r) + "functions/" + name + ": rho must be rhoInf (incompressible)");
    }
    if (const Entry* r = d.find("rhoInf")) {
        auto v = foam::scalarOf(d, *r);
        if (!v) return Err(v.error());
        f.rhoInf = *v;
    }
    // forces::setCoordinateSystem: CofR wins (axes fixed to x, y, z; e1/e3/origin unread); without it
    // coordSystem::cartesian(dict) reads origin AND two axes and writes force/moment in that rotated frame
    // (origin alone or nothing: simpleFoam v2606 stops). So CofR is required: every other form is either
    // the rotated frame (not supported) or an OpenFOAM error.
    const Entry* cofr = d.find("CofR");
    if (!cofr) {
        for (const char* k : {"e1", "e3"})
            if (const Entry* r = d.find(k))
                return Err(at(d, *r) + "functions/" + name + ": '" + k +
                           "' without CofR (forces in a rotated coordinate frame) is not supported");
        return Err(d.file + ": functions/" + name +
                   " needs 'CofR' (without it OpenFOAM reads origin + e1/e3 and stops when they are missing)");
    }
    {
        std::size_t pos = 0;
        auto v = foam::vectorOf(d, cofr->value, pos, cofr->line, "CofR");
        if (!v) return Err(v.error());
        f.CofR = {(*v)[0], (*v)[1], (*v)[2]};
    }
    if (const Entry* r = d.find("pRef")) {
        auto v = foam::scalarOf(d, *r);
        if (!v) return Err(v.error());
        f.pRef = *v;
    }
    if (const Entry* r = d.find("porosity")) {
        auto v = foam::switchOf(d, *r);
        if (!v) return Err(v.error());
        if (*v) return Err(at(d, *r) + "porosity forces are not supported");
    }
    c.forces = f;
    return {};
}

std::expected<void, string> readControl(const Dict& d, const PolyManifest& man, const string& timeDirs,
                                        FoamCase& c)
{
    if (auto r = onlyKeys(d, {"application", "startFrom", "startTime", "stopAt", "endTime", "deltaT",
                              "writeControl", "writeInterval", "purgeWrite", "writeFormat", "writePrecision",
                              "writeCompression", "timeFormat", "timePrecision", "runTimeModifiable", "libs",
                              "functions", "graphFormat", "adjustTimeStep"},
                          "controlDict");
        !r)
        return r;
    auto app = needWord(d, "application");
    if (!app) return Err(app.error());
    if (*app != "simpleFoam")
        return Err(d.file + ": application '" + *app + "': the FV path reproduces simpleFoam only (P5b)");
    double start = 0;
    if (const Entry* e = d.find("startFrom")) {
        auto w = foam::wordOf(d, *e);
        if (!w) return Err(w.error());
        if (*w == "startTime") {
            auto s = needScalar(d, "startTime");
            if (!s) return Err(s.error());
            start = *s;
        } else if (*w == "latestTime" || *w == "firstTime") {
            std::istringstream is(timeDirs);
            double t = 0, latest = 0;
            while (is >> t) latest = std::max(latest, t);
            start = *w == "latestTime" ? latest : 0;
        } else return Err(at(d, *e) + "startFrom '" + *w + "' is not supported");
    }
    if (start != 0)
        return Err(d.file + ": the run must start from time 0 (startFrom resolves to " + std::format("{:g}", start) +
                   "; restarts are not supported)");
    if (const Entry* e = d.find("stopAt")) {
        auto w = foam::wordOf(d, *e);
        if (!w) return Err(w.error());
        if (*w != "endTime") return Err(at(d, *e) + "stopAt must be endTime");
    }
    auto end = needScalar(d, "endTime");
    if (!end) return Err(end.error());
    auto dt = needScalar(d, "deltaT");
    if (!dt) return Err(dt.error());
    if (!(*dt > 0)) return Err(d.file + ": deltaT must be > 0");
    c.endIter = int(std::llround(*end / *dt));
    c.deltaT = *dt;
    if (c.endIter < 1) return Err(d.file + ": endTime/deltaT must be >= 1 iteration");
    if (const Entry* e = d.find("writeControl")) {
        auto w = foam::wordOf(d, *e);
        if (!w) return Err(w.error());
        auto wi = needScalar(d, "writeInterval");
        if (!wi) return Err(wi.error());
        if (*w == "timeStep") c.writeInterval = int(std::llround(*wi));
        else if (*w == "runTime") c.writeInterval = int(std::llround(*wi / *dt));
        else return Err(at(d, *e) + "writeControl '" + *w + "' is not supported (timeStep, runTime)");
    }
    if (const Entry* e = d.find("writePrecision")) {
        auto v = foam::scalarOf(d, *e);
        if (!v) return Err(v.error());
        c.writePrecision = std::clamp(int(*v), 1, 17);
    }
    if (const Entry* e = d.find("writeCompression")) {
        auto v = foam::switchOf(d, *e);
        if (!v) return Err(v.error());
        if (*v) c.notes.push_back("writeCompression on ignored: fields are written uncompressed ASCII");
    }
    if (const Entry* e = d.find("adjustTimeStep")) {
        auto v = foam::switchOf(d, *e);
        if (!v) return Err(v.error());
        if (*v) return Err(at(d, *e) + "adjustTimeStep yes is a transient setting");
    }
    if (const Entry* e = d.find("functions")) {
        if (!e->dict) return Err(at(d, *e) + "functions must be a dictionary");
        for (const Entry& f : e->dict->entries) {
            if (f.directive)
                return Err(at(*e->dict, f) + "directive '" + f.key + "' in functions is not supported");
            if (!f.dict) return Err(at(*e->dict, f) + "function object '" + f.key + "' must be a dictionary");
            auto type = needWord(*f.dict, "type");
            if (!type) return Err(type.error());
            if (*type != "forces")
                return Err(at(*e->dict, f) + "function object '" + f.key + "' of type '" + *type +
                           "' is not supported (forces)");
            if (c.forces) return Err(at(*e->dict, f) + "only one forces function object is supported");
            if (auto r = readForces(*f.dict, f.key, man, c); !r) return r;
        }
    }
    return {};
}

std::expected<void, string> readMrf(const Dict& d, const PolyManifest& man, FoamCase& c)
{
    for (const Entry& z : d.entries) {
        if (z.key == "FoamFile") continue;
        if (z.directive) return Err(at(d, z) + "directive '" + z.key + "' is not supported in MRFProperties");
        if (!z.dict) return Err(at(d, z) + "MRF zone '" + z.key + "' must be a dictionary");
        const Dict& s = *z.dict;
        const string where = "MRFProperties/" + z.key;
        if (auto r = onlyKeys(s, {"cellZone", "active", "nonRotatingPatches", "origin", "axis", "omega"}, where);
            !r)
            return r;
        FvMrfZone m;
        m.name = z.key;
        if (const Entry* e = s.find("active")) {
            auto v = foam::switchOf(s, *e);
            if (!v) return Err(v.error());
            m.active = *v;
        }
        if (!m.active) { // MRFZone::read stops here: nothing else is read
            c.mrf.push_back(m);
            continue;
        }
        auto cz = needWord(s, "cellZone");
        if (!cz) return Err(cz.error());
        m.cellZone = *cz;
        if (std::ranges::find(man.cellZones, m.cellZone) == man.cellZones.end())
            return Err(s.file + ": " + where + ": cellZone '" + m.cellZone + "' is not in the mesh");
        if (const Entry* e = s.find("nonRotatingPatches")) {
            // wordRes: a word is a literal patch name, a quoted string a
            // regular expression. A selector matching no patch is an error
            // here (OpenFOAM ignores it: a typo would silently rotate the wall).
            std::vector<bool> hit(man.patches.size(), false);
            for (const Token& t : e->value) {
                if (t.kind == Token::Punct) continue;
                if (t.kind != Token::Word && t.kind != Token::String)
                    return Err(at(s, *e) + where + ": nonRotatingPatches expects a list of patch names");
                const string w(t.text);
                bool any = false;
                for (std::size_t i = 0; i < man.patches.size(); ++i)
                    if (t.kind == Token::Word ? man.patches[i].name == w : regexMatch(w, man.patches[i].name))
                        hit[i] = any = true;
                if (!any) return Err(at(s, *e) + where + ": nonRotatingPatches '" + w + "' matches no mesh patch");
            }
            for (std::size_t i = 0; i < man.patches.size(); ++i)
                if (hit[i]) m.nonRotatingPatches.push_back(man.patches[i].name);
        }
        for (const char* k : {"origin", "axis"}) {
            const Entry* e = s.find(k);
            if (!e) return Err(s.file + ": " + where + " needs '" + k + "'");
            std::size_t pos = 0;
            auto v = foam::vectorOf(s, e->value, pos, e->line, k);
            if (!v) return Err(v.error());
            if (pos != e->value.size()) return Err(at(s, *e) + where + ": '" + k + "' expects one vector");
            FvVec3 x{(*v)[0], (*v)[1], (*v)[2]};
            (string(k) == "origin" ? m.origin : m.axis) = x;
        }
        // vector::normalise(): v / mag(v) (v unchanged when mag is ~0).
        const double am = std::sqrt(m.axis[0] * m.axis[0] + m.axis[1] * m.axis[1] + m.axis[2] * m.axis[2]);
        if (!(am > 0)) return Err(s.file + ": " + where + ": axis must be non-zero");
        for (double& a : m.axis) a /= am;
        const Entry* om = s.find("omega");
        if (!om) return Err(s.file + ": " + where + " needs 'omega' (rad/s; rpm is not supported)");
        if (om->value.size() == 2 && om->value[0].text == "constant" && om->value[1].kind == Token::Number)
            m.omega = om->value[1].num;
        else {
            auto v = foam::scalarOf(s, *om);
            if (!v) return Err(v.error());
            m.omega = *v;
        }
        c.mrf.push_back(m);
    }
    return {};
}

} // namespace

const char* fvUBcName(FvUBc b)
{
    static constexpr const char* n[] = {"fixedValue", "zeroGradient", "inletOutlet", "basicSymmetry",
                                        "freestreamVelocity", "empty"};
    return n[std::size_t(b)];
}

const char* fvPBcName(FvPBc b)
{
    static constexpr const char* n[] = {"fixedValue", "zeroGradient", "freestreamPressure", "empty"};
    return n[std::size_t(b)];
}

std::expected<FoamCase, string> parseFoamCase(const FoamCaseTexts& t, const PolyManifest& man)
{
    // fv::options (fvOptions.C reads constant/ then system/): sources/constraints simpleFoam applies.
    for (const char* f : {"constant/fvOptions", "system/fvOptions"})
        if (t.contains(f)) return Err(string(f) + ": fv::options (sources / constraints) are not supported");
    FoamCase c;
    c.patches.resize(man.patches.size());
    for (std::size_t i = 0; i < man.patches.size(); ++i) {
        c.patches[i].name = man.patches[i].name;
        c.patches[i].meshType = man.patches[i].type;
    }
    bool have = false;
    auto U = parseText(t, "0/U", true, have);
    if (!U) return Err(U.error());
    if (auto r = readField(*U, man, true, c); !r) return Err(r.error());
    auto p = parseText(t, "0/p", true, have);
    if (!p) return Err(p.error());
    if (auto r = readField(*p, man, false, c); !r) return Err(r.error());

    auto tp = parseText(t, "constant/transportProperties", true, have);
    if (!tp) return Err(tp.error());
    if (auto r = onlyKeys(*tp, {"transportModel", "nu"}, "transportProperties"); !r) return Err(r.error());
    if (const Entry* e = tp->find("transportModel")) {
        auto w = foam::wordOf(*tp, *e);
        if (!w) return Err(w.error());
        if (*w != "Newtonian") return Err(at(*tp, *e) + "transportModel '" + *w + "' is not supported (Newtonian)");
    }
    {
        const Entry* e = tp->find("nu");
        if (!e) return Err(tp->file + ": missing 'nu'");
        const auto& v = e->value;
        // nu V; | nu [dims] V; | nu nu [dims] V;
        if (v.empty() || v.back().kind != Token::Number)
            return Err(at(*tp, *e) + "'nu' expects a number (optionally after its dimensions)");
        c.nu = v.back().num;
        if (!(c.nu > 0)) return Err(at(*tp, *e) + "nu must be > 0");
    }

    bool haveMt = false, haveTp = false;
    auto mt = parseText(t, "constant/momentumTransport", false, haveMt);
    if (!mt) return Err(mt.error());
    auto tu = parseText(t, "constant/turbulenceProperties", false, haveTp);
    if (!tu) return Err(tu.error());
    // turbulenceProperties is v2606's; a momentumTransport beside it must agree.
    std::optional<string> simType;
    for (const Dict* d : {haveMt ? &*mt : nullptr, haveTp ? &*tu : nullptr}) {
        if (!d) continue;
        if (auto r = onlyKeys(*d, {"simulationType", "laminar", "RAS"}, d->file); !r) return Err(r.error());
        auto st = needWord(*d, "simulationType");
        if (!st) return Err(st.error());
        if (*st != "laminar" && *st != "RAS")
            return Err(d->file + ": simulationType '" + *st + "' is not supported (laminar, RAS)");
        if (simType && *simType != *st)
            return Err(d->file + ": simulationType '" + *st + "' disagrees with constant/momentumTransport ('" +
                       *simType + "'); v2606 reads turbulenceProperties: make them agree");
        simType = *st;
        if (*st == "RAS") {
            const Entry* re = d->find("RAS");
            if (!re) return Err(d->file + ": simulationType RAS needs a 'RAS' dictionary");
            FvRas keep = c.ras;
            if (auto r = readRasDict(*d, *re, c); !r) return Err(r.error());
            if (keep.model != FvTurbModel::Laminar && keep.model != c.ras.model)
                return Err(d->file + ": RASModel disagrees with constant/momentumTransport");
            continue;
        }
        if (const Entry* l = d->find("laminar")) {
            if (!l->dict) return Err(at(*d, *l) + "'laminar' must be a dictionary");
            if (auto r = onlyKeys(*l->dict, {"model", "laminarModel"}, "laminar"); !r) return Err(r.error());
            for (const char* k : {"model", "laminarModel"})
                if (const Entry* m = l->dict->find(k)) {
                    auto w = foam::wordOf(*l->dict, *m);
                    if (!w) return Err(w.error());
                    if (*w != "Stokes") return Err(at(*l->dict, *m) + "laminar model '" + *w + "' is not supported (Stokes)");
                }
        }
    }
    if (c.ras.model != FvTurbModel::Laminar) {
        struct TF
        {
            const char* field;
            std::array<int, 7> dims;
            const char* text;
        };
        std::vector<TF> fs = {{"nut", {0, 2, -1, 0, 0, 0, 0}, "[0 2 -1 0 0 0 0]"}};
        if (c.ras.model == FvTurbModel::KOmegaSST) {
            fs.push_back({"k", {0, 2, -2, 0, 0, 0, 0}, "[0 2 -2 0 0 0 0]"});
            fs.push_back({"omega", {0, 0, -1, 0, 0, 0, 0}, "[0 0 -1 0 0 0 0]"});
        } else fs.push_back({"nuTilda", {0, 2, -1, 0, 0, 0, 0}, "[0 2 -1 0 0 0 0]"});
        for (const TF& f : fs) {
            auto fd = parseText(t, string("0/") + f.field, true, have);
            if (!fd) return Err(fd.error() + " (simulationType RAS)");
            if (auto r = readTurbField(*fd, man, f.field, f.dims, f.text, c); !r) return Err(r.error());
        }
    }

    auto sch = parseText(t, "system/fvSchemes", true, have);
    if (!sch) return Err(sch.error());
    if (auto r = readSchemes(*sch, c); !r) return Err(r.error());
    auto sol = parseText(t, "system/fvSolution", true, have);
    if (!sol) return Err(sol.error());
    if (auto r = readSolution(*sol, man, c); !r) return Err(r.error());
    auto cd = parseText(t, "system/controlDict", true, have);
    if (!cd) return Err(cd.error());
    const auto td = t.find("@timedirs");
    if (auto r = readControl(*cd, man, td == t.end() ? string() : td->second, c); !r) return Err(r.error());
    bool haveMrf = false;
    auto mrf = parseText(t, "constant/MRFProperties", false, haveMrf);
    if (!mrf) return Err(mrf.error());
    if (haveMrf)
        if (auto r = readMrf(*mrf, man, c); !r) return Err(r.error());
    if (c.solverP.solver == "GAMG")
        c.notes.push_back("p: OpenFOAM GAMG -> gpu/la AMG-PCG; OpenFOAM's tolerance/relTol stop reproduced (FP32: "
                          "at max(tolerance, floor)), its iterates are not");
    if (c.solverU.solver == "smoothSolver")
        c.notes.push_back("U: OpenFOAM smoothSolver -> gpu/la BiCGStab + multicolour GS; stop reproduced (FP32: at max(tolerance, floor))");
    if (c.ras.model != FvTurbModel::Laminar)
        c.notes.push_back(std::string("RAS scalars: OpenFOAM ") +
                          (c.ras.model == FvTurbModel::KOmegaSST ? c.ras.solverK.solver : c.ras.solverNuTilda.solver) +
                          " -> gpu/la BiCGStab + multicolour GS; stop reproduced (FP32: at max(tolerance, floor))");
    return c;
}

double FvWallCoeffs::yPlusLam() const
{
    double ypl = 11;
    for (int iter = 0; iter < 10; ++iter) ypl = std::log(std::max(E * ypl, 1.0)) / kappa;
    return ypl;
}

const char* fvBlendName(FvBlend b)
{
    static constexpr const char* n[] = {"stepwise", "max", "binomial", "exponential", "tanh"};
    return n[std::size_t(b)];
}

const char* fvTurbModelName(FvTurbModel m)
{
    static constexpr const char* n[] = {"laminar", "kOmegaSST", "SpalartAllmaras"};
    return n[std::size_t(m)];
}

std::expected<FoamCaseTexts, string> readFoamCaseTexts(const fs::path& dir)
{
    FoamCaseTexts t;
    const char* files[] = {"0/U",
                           "0/p",
                           "system/controlDict",
                           "system/fvSchemes",
                           "system/fvSolution",
                           "constant/transportProperties",
                           "constant/MRFProperties",
                           "constant/momentumTransport",
                           "constant/turbulenceProperties",
                           "0/nut",
                           "0/k",
                           "0/omega",
                           "0/nuTilda",
                           "constant/fvOptions",
                           "system/fvOptions"};
    for (const char* f : files) {
        std::error_code ec;
        const fs::path p = dir / f;
        if (!fs::exists(p, ec)) {
            if (fs::exists(fs::path(p.string() + ".orig"), ec) && (string(f) == "0/U" || string(f) == "0/p"))
                return Err(p.string() + ": missing (copy the .orig file first)");
            continue;
        }
        std::ifstream in(p, std::ios::binary);
        if (!in) return Err(p.string() + ": cannot read");
        std::ostringstream ss;
        ss << in.rdbuf();
        t[f] = std::move(ss).str();
    }
    // Numeric time directories (startFrom latestTime): only those holding U.
    string dirs;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(dir, ec)) {
        if (!de.is_directory()) continue;
        const string n = de.path().filename().string();
        double v = 0;
        const auto r = std::from_chars(n.data(), n.data() + n.size(), v);
        if (r.ec != std::errc{} || r.ptr != n.data() + n.size()) continue;
        if (!fs::exists(de.path() / "U", ec)) continue;
        dirs += n + " ";
    }
    t["@timedirs"] = dirs;
    return t;
}

std::string describeFoamCase(const FoamCase& c)
{
    string s = std::format("  case        {}\n  nu          {:g}\n  iterations  {} (write every {})\n", c.dir, c.nu,
                           c.endIter, c.writeInterval);
    for (const FvPatchBc& b : c.patches)
        s += std::format("  patch       {:<20} U {:<20} p {}\n", b.name, b.uType, b.pType);
    const auto& sc = c.schemes;
    auto g = [](const FvGradScheme& x) {
        return x.kind == FvGrad::GaussLinear ? string("Gauss linear") : std::format("cellLimited Gauss linear {:g}", x.k);
    };
    s += std::format("  schemes     grad(U) {}, grad(p) {}, div(phi,U) {}Gauss {}, laplacian Gauss linear {}, "
                     "snGrad {}\n",
                     g(sc.gradU), g(sc.gradP), sc.bounded ? "bounded " : "",
                     sc.divU == FvDiv::Linear ? "linear" : sc.divU == FvDiv::LinearUpwind ? "linearUpwind grad(U)" : "upwind",
                     snGradName(sc.lapCorrected, sc.lapLimit), snGradName(sc.snGradCorrected, sc.snGradLimit));
    s += std::format("  SIMPLE      nNonOrthogonalCorrectors {}, consistent {}, momentumPredictor {}, pRefCell {} "
                     "pRefValue {:g}\n",
                     c.simple.nNonOrthCorr, c.simple.consistent ? "yes" : "no", c.simple.momentumPredictor ? "yes" : "no",
                     c.simple.pRefCell, c.simple.pRefValue);
    s += std::format("  relax       p {} U {}\n", c.relaxPGiven ? std::format("{:g}", c.relaxP) : string("-"),
                     c.relaxU ? std::format("{:g}", *c.relaxU) : string("-"));
    s += std::format("  solvers     U {} tol {:g} relTol {:g}; p {} tol {:g} relTol {:g}\n", c.solverU.solver,
                     c.solverU.tolerance, c.solverU.relTol, c.solverP.solver, c.solverP.tolerance, c.solverP.relTol);
    if (const FvRas& r = c.ras; r.model != FvTurbModel::Laminar) {
        const bool sst = r.model == FvTurbModel::KOmegaSST;
        auto cv = [](const FvConvScheme& x, const char* f) {
            static constexpr const char* n[] = {"linear", "upwind", "linearUpwind", "limitedLinear"};
            string s = std::format("{}Gauss {}", x.bounded ? "bounded " : "", n[std::size_t(x.kind)]);
            if (x.kind == FvSDiv::LinearUpwind) s += std::format(" grad({})", f);
            if (x.kind == FvSDiv::LimitedLinear) s += std::format(" {:g}", x.k);
            return s;
        };
        auto g = [](const FvGradScheme& x) {
            return x.kind == FvGrad::GaussLinear ? string("Gauss linear") : std::format("cellLimited Gauss linear {:g}", x.k);
        };
        if (sst)
            s += std::format("  RAS         kOmegaSST (OpenFOAM v2606 kOmegaSSTBase: alphaK {:g}/{:g} alphaOmega {:g}/{:g} "
                             "gamma {:g}/{:g} beta {:g}/{:g} betaStar {:g} a1 {:g} b1 {:g} c1 {:g} F3 {})\n",
                             r.alphaK1, r.alphaK2, r.alphaOmega1, r.alphaOmega2, r.gamma1, r.gamma2, r.beta1, r.beta2,
                             r.betaStar, r.a1, r.b1, r.c1, r.F3 ? "on" : "off");
        else
            s += std::format("  RAS         SpalartAllmaras (OpenFOAM v2606: sigmaNut {:g} kappa {:g} Cb1 {:g} Cb2 {:g} "
                             "Cw2 {:g} Cw3 {:g} Cv1 {:g} Cs {:g}, ft2 {} Ct3 {:g} Ct4 {:g})\n",
                             r.sigmaNut, r.kappa, r.Cb1, r.Cb2, r.Cw2, r.Cw3, r.Cv1, r.Cs, r.ft2 ? "on" : "off", r.Ct3,
                             r.Ct4);
        s += std::format("  wallDist    {}{}\n", r.exactDist ? "exactDistance (exact nearest point on the wall faces)"
                                                              : "meshWave (FaceCellWave, propagationTol 0.01)",
                         r.exactDist ? "" : (r.correctWalls ? ", correctWalls" : ", correctWalls no"));
        if (sst)
            s += std::format("  RAS schemes div(phi,k) {}, div(phi,omega) {}; grad(k) {}, grad(omega) {}\n",
                             cv(r.divK, "k"), cv(r.divOmega, "omega"), g(r.gradK), g(r.gradOmega));
        else
            s += std::format("  RAS schemes div(phi,nuTilda) {}; grad(nuTilda) {}\n", cv(r.divNuTilda, "nuTilda"),
                             g(r.gradNuTilda));
        for (std::size_t i = 0; i < c.patches.size(); ++i) {
            string bc = std::format("nut {}", r.nut[i].type);
            if (r.nut[i].kind == FvNutBc::KWall) bc += std::format(" (blending {})", fvBlendName(r.nut[i].blend));
            for (const auto* f : sst ? std::vector{&r.k, &r.omega} : std::vector{&r.nuTilda}) {
                const FvScalarBc& b = (*f)[i];
                bc += std::format(", {} {}", f == &r.k ? "k" : f == &r.omega ? "omega" : "nuTilda", b.type);
                if (b.kind == FvSBc::OmegaWall) bc += std::format(" (blending {})", fvBlendName(b.blend));
            }
            s += std::format("  RAS patch   {:<20} {}\n", c.patches[i].name, bc);
        }
        auto rx = [](const std::optional<double>& x) { return x ? std::format("{:g}", *x) : string("-"); };
        if (sst)
            s += std::format("  RAS solve   k {} tol {:g} relTol {:g} relax {}; omega {} tol {:g} relTol {:g} relax {}\n",
                             r.solverK.solver, r.solverK.tolerance, r.solverK.relTol, rx(r.relaxK),
                             r.solverOmega.solver, r.solverOmega.tolerance, r.solverOmega.relTol, rx(r.relaxOmega));
        else
            s += std::format("  RAS solve   nuTilda {} tol {:g} relTol {:g} relax {}\n", r.solverNuTilda.solver,
                             r.solverNuTilda.tolerance, r.solverNuTilda.relTol, rx(r.relaxNuTilda));
    }
    for (const FvMrfZone& m : c.mrf) {
        if (!m.active) {
            s += std::format("  MRF         {} INACTIVE\n", m.name);
            continue;
        }
        string nr;
        for (const auto& p : m.nonRotatingPatches) nr += (nr.empty() ? "" : " ") + p;
        s += std::format("  MRF         {} cellZone {} omega {:g} axis ({:g} {:g} {:g}) origin ({:g} {:g} {:g}) "
                         "nonRotatingPatches ({}) [MRF frozen rotor]\n",
                         m.name, m.cellZone, m.omega, m.axis[0], m.axis[1], m.axis[2], m.origin[0], m.origin[1],
                         m.origin[2], nr);
    }
    if (c.forces) {
        string ps;
        for (const auto& p : c.forces->patches) ps += (ps.empty() ? "" : " ") + p;
        s += std::format("  forces      {} on ({}) rhoInf {:g} CofR ({:g} {:g} {:g})\n", c.forces->name, ps,
                         c.forces->rhoInf, c.forces->CofR[0], c.forces->CofR[1], c.forces->CofR[2]);
    }
    for (const auto& n : c.notes) s += "  note        " + n + "\n";
    return s;
}

} // namespace dns
