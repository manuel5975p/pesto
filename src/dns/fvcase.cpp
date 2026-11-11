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
        if (e.referenced) continue; // a `$key` macro's definition: OpenFOAM reads no such entry here
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
           t == FoamPatchType::Cyclic || t == FoamPatchType::Wedge || t == FoamPatchType::CyclicAMI;
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
        b.movingWall = t == "movingWallVelocity";
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
    } else if (t == "cyclicAMI") {
        if (b.meshType != FoamPatchType::CyclicAMI)
            return Err(d.file + ": " + where + ": cyclicAMI on a non-cyclicAMI patch");
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.u = FvUBc::Ami;
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
    } else if (t == "cyclicAMI") {
        if (b.meshType != FoamPatchType::CyclicAMI)
            return Err(d.file + ": " + where + ": cyclicAMI on a non-cyclicAMI patch");
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.p = FvPBc::Ami;
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
                if (p.type == FoamPatchType::CyclicAMI) {
                    if (velocity) b.u = FvUBc::Ami;
                    else b.p = FvPBc::Ami;
                }
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
    } else if (t == "cyclicAMI") {
        if (patch.type != FoamPatchType::CyclicAMI)
            return Err(d.file + ": " + where + ": cyclicAMI on a non-cyclicAMI patch");
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.kind = FvSBc::Ami;
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
    } else if (t == "cyclicAMI") {
        if (patch.type != FoamPatchType::CyclicAMI)
            return Err(d.file + ": " + where + ": cyclicAMI on a non-cyclicAMI patch");
        if (auto r = onlyKeys(d, {"type", "value"}, where); !r) return r;
        b.kind = FvNutBc::Ami;
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
    if (b.kind != FvNutBc::Empty && b.kind != FvNutBc::Cell && b.kind != FvNutBc::Ami) {
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
                if (p.type == FoamPatchType::CyclicAMI) r.nut[i].kind = FvNutBc::Ami;
                r.nut[i].type = foamPatchTypeName(p.type);
            } else {
                auto& b = (field == "k" ? r.k : field == "omega" ? r.omega : r.nuTilda)[i];
                b.kind = sym ? FvSBc::ZeroGrad : FvSBc::Empty;
                if (p.type == FoamPatchType::CyclicAMI) b.kind = FvSBc::Ami;
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
        if (c.pimple.on) {
            // pimpleFoam: Euler alone (fvm::ddt and fvc::ddtCorr are EulerDdtScheme's; `Euler X` would set
            // ddtPhiCoeff, a per-field entry another scheme).
            if (!e || foam::joined(*e) != "Euler")
                return Err(d.file + ": ddtSchemes default must be Euler (pimpleFoam; got '" +
                           (e ? foam::joined(*e) : string()) + "': backward, CrankNicolson, localEuler, steadyState and "
                           "an explicit ddtPhiCoeff are not built)");
            for (const Entry& x : (*ddt)->entries)
                if (x.key != "FoamFile" && x.key != "default")
                    return Err(at(**ddt, x) + "per-field ddt scheme '" + x.key + "' is not supported (default Euler)");
        } else if (!e || foam::joined(*e) != "steadyState")
            return Err(d.file + ": ddtSchemes default must be steadyState (simpleFoam; a transient case needs "
                                "`application pimpleFoam`)");
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
    if (c.motion.on) { // CorrectPhi's fvm::laplacian(rAUf, pcorr) correction reads grad(pcorr)
        const Entry* e = (*gr)->find("grad(pcorr)");
        std::optional<FvGradScheme> g = def;
        if (e) {
            auto x = gradScheme(**gr, *e);
            if (!x) return Err(x.error());
            g = *x;
        }
        if (!g) return Err(d.file + ": gradSchemes needs 'grad(pcorr)' or a default (moving mesh: CorrectPhi)");
        c.schemes.gradPcorr = *g;
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
            if (c.pimple.on && cs->bounded)
                return Err(d.file + ": div(phi," + f + ") '" + xs + "': `bounded` (the steady -Sp(div(phi)) term) with "
                           "pimpleFoam is not built");
        }
        if (c.pimple.on && c.schemes.bounded)
            return Err(d.file + ": div(phi,U) '" + s + "': `bounded` (the steady -Sp(div(phi)) term) with pimpleFoam is "
                       "not built");
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

// The tested PIMPLE loop counts (docs/fvpimple.md): each count's range is
// what the parity gates exercise; outside it is rejected by name.
constexpr int kPimpleMaxOuter = 3, kPimpleMaxCorr = 2, kPimpleMaxNonOrth = 2;

// pimpleFoam's fvSolution: PIMPLE (pimpleControl::read + solutionControl::read
// of v2606), the solver and relaxation entries by the names the loop selects
// (FvPimpleCtl). Everything not built is rejected by name.
std::expected<void, string> readPimpleSolution(const Dict& d, const PolyManifest& man, FoamCase& c)
{
    auto pdE = subDict(d, "PIMPLE", true, "fvSolution (application pimpleFoam)");
    if (!pdE) return Err(pdE.error());
    const Dict& s = **pdE;
    if (auto r = onlyKeys(s, {"nOuterCorrectors", "nCorrectors", "nNonOrthogonalCorrectors", "momentumPredictor",
                              "pRefCell", "pRefValue", "turbOnFinalIterOnly", "turbulenceOnFinalIterOnly",
                              "correctPhi", "consistent", "transonic", "ddtCorr", "finalOnLastPimpleIterOnly",
                              "residualControl", "checkMeshCourantNo", "moveMeshOuterCorrectors"},
                          "PIMPLE");
        !r)
        return r;
    FvPimpleCtl& pc = c.pimple;
    auto count = [&](const char* key, int lo, int hi, int def) -> std::expected<int, string> {
        const Entry* e = s.find(key);
        if (!e) return def;
        auto v = foam::scalarOf(s, *e);
        if (!v) return Err(v.error());
        if (*v != std::floor(*v)) return Err(at(s, *e) + key + " must be an integer");
        if (*v < lo || *v > hi)
            return Err(at(s, *e) + std::format("{} {:g}: pimpleFoam is built and gated for {} in [{}, {}] "
                                               "(docs/fvpimple.md)", key, *v, key, lo, hi));
        return int(*v);
    };
    auto no = count("nOuterCorrectors", 1, kPimpleMaxOuter, 1);
    if (!no) return Err(no.error());
    auto nc = count("nCorrectors", 1, kPimpleMaxCorr, 1);
    if (!nc) return Err(nc.error());
    auto nn = count("nNonOrthogonalCorrectors", 0, kPimpleMaxNonOrth, 0);
    if (!nn) return Err(nn.error());
    pc.nOuter = *no;
    pc.nCorr = *nc;
    c.simple.nNonOrthCorr = *nn;
    // A switch that must keep its v2606 default (`def`): the other value is not built.
    auto fixedSwitch = [&](const char* key, bool def, const char* why) -> std::expected<void, string> {
        const Entry* e = s.find(key);
        if (!e) return {};
        auto v = foam::switchOf(s, *e);
        if (!v) return Err(v.error());
        if (*v != def) return Err(at(s, *e) + "PIMPLE " + key + " " + (*v ? "yes" : "no") + ": " + why);
        return {};
    };
    if (auto r = fixedSwitch("momentumPredictor", true, "the predictor-free loop is not built"); !r) return r;
    if (auto r = fixedSwitch("consistent", false, "PIMPLEC (rAtU = 1/max(1/rAU - H1, 0.1/rAU)) is not built"); !r)
        return r;
    if (auto r = fixedSwitch("transonic", false, "a compressible setting"); !r) return r;
    if (auto r = fixedSwitch("ddtCorr", true, "the loop without fvc::ddtCorr is not built"); !r) return r;
    if (auto r = fixedSwitch("finalOnLastPimpleIterOnly", false, "not built (pFinal on every outer corrector's last "
                                                                  "inner iteration is)");
        !r)
        return r;
    if (const Entry* e = s.find("residualControl"))
        return Err(at(s, *e) + "PIMPLE residualControl (outer-corrector convergence control) is not built");
    if (const Entry* e = s.find("turbOnFinalIterOnly")) {
        auto v = foam::switchOf(s, *e);
        if (!v) return Err(v.error());
        pc.turbOnFinalIterOnly = *v;
    }
    if (s.find("turbulenceOnFinalIterOnly"))
        c.notes.push_back(std::format("PIMPLE turbulenceOnFinalIterOnly is read by NOTHING in OpenFOAM v2606 (pimpleControl "
                                      "reads `turbOnFinalIterOnly`, default yes): turbulence is corrected {}",
                                      pc.turbOnFinalIterOnly ? "on the final outer corrector only"
                                                             : "on every outer corrector (turbOnFinalIterOnly no)"));
    // createDyMControls.H: correctPhi (default mesh.dynamic()), checkMeshCourantNo (no),
    // moveMeshOuterCorrectors (no). Used only when the mesh changes.
    if (auto r = fixedSwitch("moveMeshOuterCorrectors", false,
                             "the mesh is moved once per time step, at the first outer corrector");
        !r)
        return r;
    if (c.motion.on) {
        pc.correctPhi = true;
        for (const char* key : {"correctPhi", "checkMeshCourantNo"})
            if (const Entry* e = s.find(key)) {
                auto v = foam::switchOf(s, *e);
                if (!v) return Err(v.error());
                (string(key) == "correctPhi" ? pc.correctPhi : pc.checkMeshCourantNo) = *v;
            }
    } else {
        if (s.find("correctPhi"))
            c.notes.push_back("PIMPLE correctPhi is read by createDyMControls and used only when the mesh changes: inert "
                              "on a static mesh");
        if (s.find("checkMeshCourantNo"))
            c.notes.push_back("PIMPLE checkMeshCourantNo is used only when the mesh changes: inert on a static mesh");
    }
    c.simple.consistent = false;
    c.simple.momentumPredictor = true;
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

    // Solvers: the entries the loop looks up, by name. The non-final ones exist
    // only when a non-final iteration does.
    auto sv = subDict(d, "solvers", true, "fvSolution");
    if (!sv) return Err(sv.error());
    FvRas& ras = c.ras;
    const bool sst = ras.model == FvTurbModel::KOmegaSST, sa = ras.model == FvTurbModel::SpalartAllmaras;
    const bool outerNonFinal = pc.nOuter > 1;
    const bool innerNonFinal = pc.nCorr > 1 || c.simple.nNonOrthCorr > 0;
    const bool turbNonFinal = outerNonFinal && !pc.turbOnFinalIterOnly;
    struct Want
    {
        const char* name;
        bool pressure, nonFinal;
        FvSolverCtl *dst, *fin;
    };
    std::vector<Want> want = {{"U", false, outerNonFinal, &c.solverU, &pc.solverUFinal},
                              {"p", true, innerNonFinal, &c.solverP, &pc.solverPFinal}};
    if (sst) {
        want.push_back({"k", false, turbNonFinal, &ras.solverK, &pc.solverKFinal});
        want.push_back({"omega", false, turbNonFinal, &ras.solverOmega, &pc.solverOmegaFinal});
    }
    if (sa) want.push_back({"nuTilda", false, turbNonFinal, &ras.solverNuTilda, &pc.solverNuTildaFinal});
    for (const Want& w : want) {
        auto f = readSolver(**sv, string(w.name) + "Final", w.pressure);
        if (!f) return Err(f.error());
        *w.fin = *f;
        *w.dst = *f; // never looked up by OpenFOAM when no non-final iteration exists
        if (!w.nonFinal) continue;
        auto n = readSolver(**sv, w.name, w.pressure);
        if (!n) return Err(n.error());
        *w.dst = *n;
    }
    if (c.motion.on && pc.correctPhi) {
        // CorrectPhi: pcorr.select(pimple.finalNonOrthogonalIter()) -> pcorrFinal | pcorr
        auto f = readSolver(**sv, "pcorrFinal", true);
        if (!f) return Err(f.error() + " (moving mesh, correctPhi: CorrectPhi's pressure-corrector solve)");
        pc.solverPcorrFinal = pc.solverPcorr = *f;
        if (c.simple.nNonOrthCorr > 0) {
            auto n = readSolver(**sv, "pcorr", true);
            if (!n) return Err(n.error() + " (moving mesh, correctPhi, nNonOrthogonalCorrectors > 0)");
            pc.solverPcorr = *n;
        }
    }

    auto rf = subDict(d, "relaxationFactors", false, "fvSolution");
    if (!rf) return Err(rf.error());
    if (*rf) {
        if (auto r = onlyKeys(**rf, {"fields", "equations"}, "relaxationFactors (use fields {} / equations {})"); !r)
            return r;
        for (const char* sub : {"fields", "equations"}) {
            auto sd = subDict(**rf, sub, false, "relaxationFactors");
            if (!sd) return Err(sd.error());
            if (*sd && (*sd)->find("default"))
                return Err(d.file + ": relaxationFactors/" + sub + "/default is not supported (name the fields)");
        }
        auto get = [&](const char* sub, const string& name) { return relaxOf(**rf, sub, name); };
        auto rp = get("fields", "p");
        if (!rp) return Err(rp.error());
        auto rpf = get("fields", "pFinal");
        if (!rpf) return Err(rpf.error());
        if (*rp) {
            c.relaxP = **rp;
            c.relaxPGiven = true;
        }
        pc.relaxPFinal = *rpf;
        // solutionControl::storePrevIter stores p's previous iteration only when relaxField("p") is found;
        // p.relax() on the final corrector looks up "pFinal" and then needs it: OpenFOAM stops.
        if (*rpf && !*rp)
            return Err(d.file + ": relaxationFactors/fields has a factor for pFinal but none for p: OpenFOAM stores "
                                "p's previous iteration only when `p` is found and stops in p.relax()");
        for (const char* name : {"U", "UFinal", "k", "kFinal", "omega", "omegaFinal", "nuTilda", "nuTildaFinal"}) {
            auto x = get("fields", name);
            if (!x) return Err(x.error());
            if (*x) return Err(d.file + ": relaxationFactors/fields/" + name + " (field relaxation) is not supported");
        }
        auto ru = get("equations", "U");
        if (!ru) return Err(ru.error());
        c.relaxU = *ru;
        auto ruf = get("equations", "UFinal");
        if (!ruf) return Err(ruf.error());
        pc.relaxUFinal = *ruf;
        for (const char* name : {"p", "pFinal"}) {
            auto x = get("equations", name);
            if (!x) return Err(x.error());
            if (*x)
                c.notes.push_back(std::format("relaxationFactors/equations: the factor {:g} matching {} is inert "
                                              "(pimpleFoam never calls pEqn.relax(); p is relaxed by fields/p, pFinal)",
                                              **x, name));
        }
        struct TR
        {
            const char* name;
            std::optional<double> *dst, *fin;
        };
        std::vector<TR> tr;
        if (sst) tr = {{"k", &ras.relaxK, &pc.relaxKFinal}, {"omega", &ras.relaxOmega, &pc.relaxOmegaFinal}};
        if (sa) tr = {{"nuTilda", &ras.relaxNuTilda, &pc.relaxNuTildaFinal}};
        for (const TR& t : tr) {
            auto a = get("equations", t.name);
            if (!a) return Err(a.error());
            *t.dst = *a;
            auto b = get("equations", string(t.name) + "Final");
            if (!b) return Err(b.error());
            *t.fin = *b;
        }
    }
    if (d.find("SIMPLE") || d.find("PISO")) c.notes.push_back("fvSolution SIMPLE/PISO dictionaries ignored (PIMPLE run)");
    return {};
}

std::expected<void, string> readSolution(const Dict& d, const PolyManifest& man, FoamCase& c)
{
    if (auto r = onlyKeys(d, {"solvers", "SIMPLE", "relaxationFactors", "cache", "PIMPLE", "PISO",
                              "potentialFlow"},
                          "fvSolution");
        !r)
        return r;
    if (c.pimple.on) return readPimpleSolution(d, man, c);
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
        // simpleFoam's pEqn.H never calls pEqn.relax() (v2606: p.relax() only, the FIELD factor), so an
        // equation factor matching p is read by nothing.
        if (*rpe)
            c.notes.push_back(std::format("relaxationFactors/equations: the factor {:g} matching p is inert "
                                          "(simpleFoam never calls pEqn.relax(); p is relaxed by fields/p only)",
                                          **rpe));
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

double dot3(const FvVec3& a, const FvVec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
FvVec3 cross3(const FvVec3& a, const FvVec3& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// forceCoeffs: setCoordinateSystem(dict, "liftDir", "dragDir") =
// coordSystem::cartesian(CofR, e3 = liftDir, e1 = dragDir), i.e.
// coordinateRotations::axes::rotation(liftDir, dragDir, E3_E1) verbatim.
std::expected<void, string> readForceCoeffs(const Dict& d, const string& where, FvForces& f)
{
    FvForceCoeffs k;
    FvVec3 lift{}, drag{};
    for (auto [key, dst] : {std::pair{"liftDir", &lift}, {"dragDir", &drag}}) {
        const Entry* e = d.find(key);
        if (!e) return Err(d.file + ": " + where + " needs '" + key + "'");
        std::size_t pos = 0;
        auto v = foam::vectorOf(d, e->value, pos, e->line, key);
        if (!v) return Err(v.error());
        if (pos != e->value.size()) return Err(at(d, *e) + where + ": '" + key + "' expects one vector");
        *dst = {(*v)[0], (*v)[1], (*v)[2]};
    }
    const double mag1 = std::sqrt(dot3(lift, lift));
    if (mag1 < 1e-150) return Err(d.file + ": " + where + ": liftDir has zero length");
    const FvVec3 ax1{lift[0] / mag1, lift[1] / mag1, lift[2] / mag1};
    FvVec3 ax2 = drag;
    if (std::sqrt(dot3(ax2, ax2)) < 1e-150)
        return Err(d.file + ": " + where + ": dragDir has zero length (OpenFOAM guesses an axis: not supported)");
    const double col = dot3(ax2, ax1); // removeCollinear
    for (int i = 0; i < 3; ++i) ax2[std::size_t(i)] -= col * ax1[std::size_t(i)];
    const double mag2 = std::sqrt(dot3(ax2, ax2));
    if (mag2 < 1e-15)
        return Err(d.file + ": " + where + ": liftDir and dragDir are co-linear (OpenFOAM guesses an axis: not "
                                           "supported)");
    for (double& x : ax2) x /= mag2;
    k.e1 = ax2;
    k.e2 = cross3(ax1, ax2);
    k.e3 = ax1;
    for (auto [key, dst] : {std::pair{"magUInf", &k.magUInf}, {"lRef", &k.lRef}, {"Aref", &k.Aref}}) {
        auto v = needScalar(d, key);
        if (!v) return Err(v.error() + " (" + where + ")");
        *dst = *v;
    }
    if (std::min(k.magUInf, f.rhoInf) < 1e-15 || std::min(k.lRef, k.Aref) < 1e-15)
        return Err(d.file + ": " + where + ": Non-zero values are required for reference scales (magUInf, rhoInf, "
                                           "lRef, Aref)");
    if (const Entry* e = d.find("coefficients"))
        return Err(at(d, *e) + where + ": 'coefficients' (a subset of the columns) is not supported: every "
                                       "coefficient is written");
    f.coeffs = k;
    return {};
}

std::expected<void, string> readForces(const Dict& d, const string& name, bool coeffs, const PolyManifest& man,
                                       FoamCase& c)
{
    if (auto r = coeffs ? onlyKeys(d, {"type", "libs", "writeControl", "writeInterval", "executeControl",
                                       "executeInterval", "patches", "rho", "rhoInf", "CofR", "pRef", "log",
                                       "porosity", "enabled", "liftDir", "dragDir", "pitchAxis", "magUInf", "lRef",
                                       "Aref", "coefficients"},
                                   "functions/" + name)
                        : onlyKeys(d, {"type", "libs", "writeControl", "writeInterval", "executeControl",
                                       "executeInterval", "patches", "rho", "rhoInf", "CofR", "origin", "pRef", "log",
                                       "porosity", "enabled", "e1", "e3"},
                                   "functions/" + name);
        !r)
        return r;
    if (std::ranges::any_of(c.forces, [&](const FvForces& x) { return x.name == name; }))
        return Err(d.file + ": functions/" + name + ": two function objects of that name");
    if (int(c.forces.size()) == FoamCase::kMaxForces)
        return Err(d.file + ": functions/" + name + std::format(": more than {} forces / forceCoeffs function objects "
                                                                "are not supported", FoamCase::kMaxForces));
    for (const char* k : {"writeControl", "executeControl"})
        if (const Entry* e = d.find(k)) {
            auto w = foam::wordOf(d, *e);
            if (!w) return Err(w.error());
            const char* iv = string(k) == "writeControl" ? "writeInterval" : "executeInterval";
            double n = 1;
            if (const Entry* x = d.find(iv)) {
                auto v = foam::scalarOf(d, *x);
                if (!v) return Err(v.error());
                n = *v;
            }
            if (*w != "timeStep" || n != 1)
                c.notes.push_back(std::format("functions/{}: {} {} {:g} ignored: the files get one row per iteration",
                                              name, k, *w, n));
        }
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
    if (coeffs) {
        if (auto r = readForceCoeffs(d, "functions/" + name, f); !r) return r;
        if (d.find("pitchAxis"))
            c.notes.push_back("functions/" + name + ": pitchAxis is read by nothing in v2606 forceCoeffs (the axes "
                                                    "are liftDir, dragDir and their cross product)");
    }
    c.forces.push_back(std::move(f));
    return {};
}

std::expected<void, string> readControl(const Dict& d, const PolyManifest& man, const string& timeDirs,
                                        FoamCase& c)
{
    if (auto r = onlyKeys(d, {"application", "startFrom", "startTime", "stopAt", "endTime", "deltaT",
                              "writeControl", "writeInterval", "purgeWrite", "writeFormat", "writePrecision",
                              "writeCompression", "timeFormat", "timePrecision", "runTimeModifiable", "libs",
                              "functions", "graphFormat", "adjustTimeStep", "maxCo", "maxDeltaT"},
                          "controlDict");
        !r)
        return r;
    auto app = needWord(d, "application");
    if (!app) return Err(app.error());
    if (*app != "simpleFoam" && *app != "pimpleFoam")
        return Err(d.file + ": application '" + *app + "': the FV path reproduces simpleFoam and pimpleFoam (static "
                            "mesh) only");
    const bool pimple = *app == "pimpleFoam";
    if (!pimple)
        for (const char* k : {"maxCo", "maxDeltaT"})
            if (const Entry* e = d.find(k)) return Err(at(d, *e) + "'" + k + "' is a transient (pimpleFoam) setting");
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
    c.deltaT = *dt;
    if (pimple) {
        // Time's controls as pimpleFoam reads them (createTimeControls.H, Time::readDict).
        FvTimeCtl& tc = c.pimple.time;
        tc.endTime = *end;
        tc.deltaT = *dt;
        if (!(*end > 0)) return Err(d.file + ": endTime must be > 0");
        const Entry* wc = d.find("writeControl");
        if (!wc) return Err(d.file + ": missing 'writeControl' (pimpleFoam: timeStep | runTime | adjustableRunTime)");
        auto w = foam::wordOf(d, *wc);
        if (!w) return Err(w.error());
        auto wi = needScalar(d, "writeInterval");
        if (!wi) return Err(wi.error());
        if (*w == "timeStep") tc.writeControl = FvWriteControl::TimeStep;
        else if (*w == "runTime") tc.writeControl = FvWriteControl::RunTime;
        else if (*w == "adjustableRunTime") tc.writeControl = FvWriteControl::AdjustableRunTime;
        else
            return Err(at(d, *wc) + "writeControl '" + *w + "' is not supported (timeStep, runTime, adjustableRunTime)");
        tc.writeInterval = *wi;
        if (tc.writeControl == FvWriteControl::TimeStep ? (*wi < 1 || *wi != std::floor(*wi)) : !(*wi > 0))
            return Err(d.file + ": writeInterval must be an integer >= 1 (timeStep) or > 0 (runTime)");
        if (const Entry* e = d.find("adjustTimeStep")) {
            auto v = foam::switchOf(d, *e);
            if (!v) return Err(v.error());
            tc.adjust = *v;
        }
        for (auto [key, dst] : {std::pair{"maxCo", &tc.maxCo}, {"maxDeltaT", &tc.maxDeltaT}})
            if (const Entry* e = d.find(key)) {
                auto v = foam::scalarOf(d, *e);
                if (!v) return Err(v.error());
                if (!(*v > 0)) return Err(at(d, *e) + key + " must be > 0");
                *dst = *v;
            }
        if (const Entry* e = d.find("timePrecision")) {
            auto v = foam::scalarOf(d, *e);
            if (!v) return Err(v.error());
            tc.timePrecision = std::clamp(int(*v), 1, 17);
        }
        if (const Entry* e = d.find("timeFormat")) {
            auto v = foam::wordOf(d, *e);
            if (!v) return Err(v.error());
            if (*v != "general") return Err(at(d, *e) + "timeFormat '" + *v + "' is not supported (general)");
        }
    } else {
        c.endIter = int(std::llround(*end / *dt));
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
        if (*v && !pimple) return Err(at(d, *e) + "adjustTimeStep yes is a transient setting");
    }
    if (const Entry* e = d.find("functions")) {
        if (!e->dict) return Err(at(d, *e) + "functions must be a dictionary");
        for (const Entry& f : e->dict->entries) {
            if (f.directive)
                return Err(at(*e->dict, f) + "directive '" + f.key + "' in functions is not supported");
            if (!f.dict) return Err(at(*e->dict, f) + "function object '" + f.key + "' must be a dictionary");
            auto type = needWord(*f.dict, "type");
            if (!type) return Err(type.error());
            if (*type == "yPlus") {
                // Read-only post-processing (yPlus.dat, the yPlus field): no effect on the solution.
                c.notes.push_back("functions/" + f.key + " (yPlus) accepted and SKIPPED: no yPlus.dat / yPlus field "
                                                         "is written");
                continue;
            }
            if (*type != "forces" && *type != "forceCoeffs")
                return Err(at(*e->dict, f) + "function object '" + f.key + "' of type '" + *type +
                           "' is not supported (forces, forceCoeffs; yPlus is skipped)");
            if (auto r = readForces(*f.dict, f.key, *type == "forceCoeffs", man, c); !r) return r;
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

// A constant Function1<scalar>: `key X;` | `key constant X;`.
std::expected<double, string> constScalar(const Dict& s, const string& where, const char* key)
{
    const Entry* e = s.find(key);
    if (!e) return Err(s.file + ": " + where + " needs '" + key + "'");
    const auto& t = e->value;
    if (!e->dict && t.size() == 1 && t[0].kind == Token::Number) return t[0].num;
    if (!e->dict && t.size() == 2 && t[0].text == "constant" && t[1].kind == Token::Number) return t[1].num;
    return Err(at(s, *e) + where + ": '" + key + "' must be a constant (`" + key + " X;` | `" + key +
               " constant X;`): a Function1 of the monitored speed is not supported");
}

std::expected<FvVec3, string> constVector(const Dict& s, const string& where, const char* key, bool fn1);

// d.optionalSubDict(key): the sub-dictionary when `key` names one, else d itself.
const Dict* optionalSubDict(const Dict& d, const char* key)
{
    const Entry* e = d.find(key);
    return e && e->dict ? e->dict.get() : &d;
}

// One motion solver dictionary (motionSolver::New + solidBodyMotionSolver +
// solidBodyMotionFunction::New + rotatingMotion of v2606): `motionSolver` (or
// the older `solver`) solidBody; its coefficients in optionalSubDict(
// "solidBodyCoeffs"): solidBodyMotionFunction rotatingMotion, whose origin /
// axis / omega sit in optionalSubDict("rotatingMotionCoeffs"). zoneInCoeffs:
// the solver's own cellZone selection (zoneMotion) is read (single solver) or
// rejected (a multi-motion entry: the mesh's `cellZone` selects the points).
std::expected<FvSolidBody, string> readSolidBody(const Dict& d, const string& where, const PolyManifest& man,
                                                 bool zoneInCoeffs)
{
    FvSolidBody b;
    const Entry* se = d.find("motionSolver"); // getCompat("motionSolver", {{"solver", -1612}})
    if (!se) se = d.find("solver");
    if (!se) return Err(d.file + ": " + where + " needs 'motionSolver' (or 'solver')");
    auto sn = foam::wordOf(d, *se);
    if (!sn) return Err(sn.error());
    if (*sn != "solidBody")
        return Err(at(d, *se) + where + ": motion solver '" + *sn +
                   "' is not built (solidBody; multiSolidBodyMotionSolver, displacementLaplacian, velocityLaplacian, "
                   "sixDoFRigidBodyMotion, ... are not)");
    const Dict& cd = *optionalSubDict(d, "solidBodyCoeffs");
    const string cw = where + (&cd == &d ? "" : "/solidBodyCoeffs");
    if (const Entry* e = cd.find("cellSet")) {
        auto w = foam::wordOf(cd, *e);
        if (!w || *w != "none") return Err(at(cd, *e) + cw + ": a cellSet motion selection is not built (cellZone)");
    }
    if (const Entry* e = cd.find("cellZone"); e && &cd != &d) {
        // (the multi-motion entry's own `cellZone` lives in d and is the caller's)
        if (!zoneInCoeffs)
            return Err(at(cd, *e) + cw + ": a cellZone inside a dynamicMultiMotionSolverFvMesh entry's solidBodyCoeffs "
                                         "(the solver would move a subset of the entry's points) is not built");
    }
    if (zoneInCoeffs)
        if (const Entry* e = cd.find("cellZone")) {
            if (e->quotedKey || e->value.size() != 1 || e->value[0].kind != Token::Word)
                return Err(at(cd, *e) + cw + ": cellZone must be one zone name (a regular expression / list is not built)");
            const string z(e->value[0].text);
            if (z != "none") { // zoneMotion: `none` is a placeholder
                if (std::ranges::find(man.cellZones, z) == man.cellZones.end())
                    return Err(at(cd, *e) + cw + ": No matching cellZones: " + z);
                b.cellZone = z;
            }
        }
    const Entry* fe = cd.find("solidBodyMotionFunction");
    if (!fe) return Err(cd.file + ": " + cw + " needs 'solidBodyMotionFunction'");
    auto fn = foam::wordOf(cd, *fe);
    if (!fn) return Err(fn.error());
    if (*fn != "rotatingMotion")
        return Err(at(cd, *fe) + cw + ": solidBodyMotionFunction '" + *fn +
                   "' is not built (rotatingMotion; linearMotion, oscillatingRotatingMotion, axisRotationMotion, "
                   "multiMotion, SDA, tabulated6DoFMotion, ... are not)");
    const Dict& rd = *optionalSubDict(cd, "rotatingMotionCoeffs");
    const string rw = cw + (&rd == &cd ? "" : "/rotatingMotionCoeffs");
    for (const Dict* x : {&cd, &rd}) { // every key OpenFOAM reads here is known; nothing else may hide in them
        if (x == &d) continue;         // the flat spelling: the top-level check is the caller's
        if (auto r = onlyKeys(*x, {"cellZone", "cellSet", "solidBodyMotionFunction", "rotatingMotionCoeffs", "origin",
                                   "axis", "omega"},
                              x == &cd ? cw : rw);
            !r)
            return Err(r.error());
    }
    auto o = constVector(rd, rw, "origin", false);
    if (!o) return Err(o.error());
    auto a = constVector(rd, rw, "axis", false);
    if (!a) return Err(a.error());
    if (!((*a)[0] != 0 || (*a)[1] != 0 || (*a)[2] != 0)) return Err(rd.file + ": " + rw + ": axis must be non-zero");
    const Entry* om = rd.find("omega");
    if (!om) return Err(rd.file + ": " + rw + " needs 'omega' (rad/s)");
    const auto& tv = om->value;
    if (!om->dict && tv.size() == 1 && tv[0].kind == Token::Number) b.omega = tv[0].num;
    else if (!om->dict && tv.size() == 2 && tv[0].text == "constant" && tv[1].kind == Token::Number) b.omega = tv[1].num;
    else
        return Err(at(rd, *om) + rw + ": omega must be a constant (`omega X;` | `omega constant X;`): a Function1 of time "
                                      "(table, sine, ...) is not built");
    b.origin = *o;
    b.axis = *a;
    return b;
}

// constant/dynamicMeshDict (dynamicFvMesh::New, dynamicMotionSolverFvMesh,
// dynamicMultiMotionSolverFvMesh of v2606) into c.motion; staticFvMesh: off.
std::expected<void, string> readDynamicMesh(const Dict& d, const PolyManifest& man, FoamCase& c)
{
    for (const Entry& e : d.entries)
        if (e.directive) return Err(at(d, e) + "directive '" + e.key + "' is not supported in dynamicMeshDict");
    auto type = needWord(d, "dynamicFvMesh");
    if (!type) return Err(type.error());
    if (*type == "staticFvMesh") {
        if (auto r = onlyKeys(d, {"dynamicFvMesh"}, "dynamicMeshDict"); !r) return Err(r.error());
        return {};
    }
    for (const char* k : {"updateControl", "updateInterval"})
        if (const Entry* e = d.find(k))
            return Err(at(d, *e) + "dynamicMeshDict " + k + " (a mesh update that skips time steps) is not built");
    FvDynamicMesh& m = c.motion;
    if (*type == "dynamicMotionSolverFvMesh") {
        if (auto r = onlyKeys(d, {"dynamicFvMesh", "motionSolverLibs", "motionSolver", "solver", "solidBodyCoeffs",
                                  "cellZone", "cellSet", "solidBodyMotionFunction", "rotatingMotionCoeffs", "origin",
                                  "axis", "omega"},
                              "dynamicMeshDict");
            !r)
            return Err(r.error());
        auto b = readSolidBody(d, "dynamicMeshDict", man, true);
        if (!b) return Err(b.error());
        m.bodies.push_back(*b);
    } else if (*type == "dynamicMultiMotionSolverFvMesh") {
        if (auto r = onlyKeys(d, {"dynamicFvMesh", "motionSolverLibs", "dynamicMultiMotionSolverFvMeshCoeffs"},
                              "dynamicMeshDict");
            !r)
            return Err(r.error());
        auto cd = subDict(d, "dynamicMultiMotionSolverFvMeshCoeffs", true, "dynamicMeshDict");
        if (!cd) return Err(cd.error());
        m.multi = true;
        for (const Entry& e : (*cd)->entries) {
            if (e.directive) return Err(at(**cd, e) + "directive '" + e.key + "' is not supported in dynamicMeshDict");
            if (!e.dict) continue; // `for (const entry& e : coeffs) if (e.dictPtr())`
            const Dict& s = *e.dict;
            const string where = "dynamicMultiMotionSolverFvMeshCoeffs/" + e.key;
            if (auto r = onlyKeys(s, {"motionSolver", "solver", "cellZone", "solidBodyCoeffs", "motionSolverLibs",
                                      "solidBodyMotionFunction", "rotatingMotionCoeffs", "origin", "axis", "omega"},
                                  where);
                !r)
                return Err(r.error());
            const Entry* ze = s.find("cellZone");
            if (!ze) return Err(s.file + ": " + where + " needs 'cellZone'");
            if (ze->value.size() != 1 || ze->value[0].kind != Token::Word)
                return Err(at(s, *ze) + where + ": cellZone must be one zone name (a regular expression / group is not built)");
            const string z(ze->value[0].text);
            if (std::ranges::find(man.cellZones, z) == man.cellZones.end())
                return Err(at(s, *ze) + where + ": No matching cellZones: " + z);
            // (flat spelling, no solidBodyCoeffs: the solver's own zoneMotion reads this same cellZone -- the same points)
            auto b = readSolidBody(s, where, man, false);
            if (!b) return Err(b.error());
            b->name = e.key;
            b->cellZone = z;
            if (std::ranges::any_of(m.bodies, [&](const FvSolidBody& x) { return x.cellZone == z; }))
                return Err(at(s, *ze) + where + ": cellZone '" + z + "' is moved by an earlier entry too");
            m.bodies.push_back(*b);
        }
        if (m.bodies.empty()) return Err(d.file + ": dynamicMultiMotionSolverFvMeshCoeffs has no motion entry");
    } else
        return Err(d.file + ": dynamicFvMesh '" + *type +
                   "' is not built (staticFvMesh, dynamicMotionSolverFvMesh, dynamicMultiMotionSolverFvMesh; "
                   "dynamicRefineFvMesh, dynamicOversetFvMesh, dynamicInkJetFvMesh, ... are not)");
    m.on = true;
    return {};
}

// A constant vector: `key (x y z);` | (fn1) `key constant (x y z);`.
std::expected<FvVec3, string> constVector(const Dict& s, const string& where, const char* key, bool fn1)
{
    const Entry* e = s.find(key);
    if (!e) return Err(s.file + ": " + where + " needs '" + key + "'");
    std::size_t pos = !e->dict && fn1 && !e->value.empty() && e->value[0].text == "constant" ? 1 : 0;
    if (e->dict || e->value.size() != pos + 5 || !e->value[pos].is('('))
        return Err(at(s, *e) + where + ": '" + key + "' must be a constant vector" +
                   (fn1 ? " (`(x y z)` | `constant (x y z)`): a Function1 of time is not supported" : " (x y z)"));
    auto v = foam::vectorOf(s, e->value, pos, e->line, key);
    if (!v) return Err(v.error());
    return FvVec3{(*v)[0], (*v)[1], (*v)[2]};
}

// fvOptions actuationDiskSource (OpenFOAM v2606 actuationDiskSource.C +
// cellSetOption + writeFile): variant Froude, selectionMode cellZone, one
// cellZone, monitorMethod points; the constructor's checks and
// calcFroudeMethod's "Cp and Ct must be greater than zero" at read time.
std::expected<void, string> readDisk(const Dict& s, const string& name, const string& file, const PolyManifest& man,
                                     FoamCase& c)
{
    const string where = file + "/" + name + " (actuationDiskSource)";
    if (auto r = onlyKeys(s, {"type", "active", "log", "variant", "selectionMode", "cellZone", "diskArea", "diskDir",
                              "sink", "Cp", "Ct", "monitorMethod", "monitorCoeffs", "upstreamPoint", "writeToFile",
                              "writeFileStart", "writeFileEnd", "writePrecision", "updateHeader", "useUserTime"},
                          where);
        !r)
        return r;
    auto word = [&](const char* key, const char* dflt) -> std::expected<string, string> {
        const Entry* e = s.find(key);
        if (!e) return string(dflt);
        return foam::wordOf(s, *e);
    };
    auto sw = [&](const char* key, bool dflt) -> std::expected<bool, string> {
        const Entry* e = s.find(key);
        if (!e) return dflt;
        return foam::switchOf(s, *e);
    };
    auto active = sw("active", true);
    if (!active) return Err(active.error());
    if (!*active) return Err(s.file + ": " + where + ": an inactive source (remove it)");
    auto variant = word("variant", "Froude");
    if (!variant) return Err(variant.error());
    if (*variant != "Froude")
        return Err(s.file + ": " + where + ": variant '" + *variant + "' is not supported (Froude)");
    auto mode = needWord(s, "selectionMode");
    if (!mode) return Err(mode.error());
    if (*mode != "cellZone")
        return Err(s.file + ": " + where + ": selectionMode '" + *mode + "' is not supported (cellZone)");
    FvDisk k;
    k.name = name;
    auto cz = needWord(s, "cellZone");
    if (!cz) return Err(cz.error() + " (" + where + "; a `cellZones` list is not supported)");
    k.cellZone = *cz;
    if (std::ranges::find(man.cellZones, k.cellZone) == man.cellZones.end())
        return Err(s.file + ": " + where + ": cellZone '" + k.cellZone + "' is not in the mesh");
    if (std::ranges::any_of(c.disks, [&](const FvDisk& x) { return x.cellZone == k.cellZone; }))
        return Err(s.file + ": " + where + ": a second actuationDiskSource on cellZone '" + k.cellZone +
                   "' is not supported");
    auto method = word("monitorMethod", "points");
    if (!method) return Err(method.error());
    if (*method != "points")
        return Err(s.file + ": " + where + ": monitorMethod '" + *method + "' is not supported (points)");
    auto sink = sw("sink", true);
    if (!sink) return Err(sink.error());
    const double sign = *sink ? 1.0 : -1.0;
    auto area = needScalar(s, "diskArea");
    if (!area) return Err(area.error());
    if (!(*area >= 1e-300)) return Err(s.file + ": " + where + ": diskArea must be > 0");
    k.diskArea = *area;
    auto dir = constVector(s, where, "diskDir", true);
    if (!dir) return Err(dir.error());
    const double magDir = std::sqrt(dot3(*dir, *dir));
    if (magDir < 1e-15)
        return Err(s.file + ": " + where + ": Actuator disk surface-normal vector is zero (diskDir)");
    k.dir = {(*dir)[0] / magDir, (*dir)[1] / magDir, (*dir)[2] / magDir}; // dir/magDir
    auto cp = constScalar(s, where, "Cp");
    if (!cp) return Err(cp.error());
    auto ct = constScalar(s, where, "Ct");
    if (!ct) return Err(ct.error());
    k.Cp = sign * *cp;
    k.Ct = sign * *ct;
    if (k.Cp <= 1e-300 || k.Ct <= 1e-300)
        return Err(s.file + ": " + where +
                   std::format(": Cp and Ct must be greater than zero (after the sink sign {:+g}: Cp = {:g}, Ct = {:g})",
                               sign, k.Cp, k.Ct));
    if (const Entry* mc = s.find("monitorCoeffs")) {
        if (!mc->dict) return Err(at(s, *mc) + where + ": monitorCoeffs must be a dictionary");
        if (auto r = onlyKeys(*mc->dict, {"points"}, where + "/monitorCoeffs"); !r) return r;
        const Entry* p = mc->dict->find("points");
        if (!p) return Err(mc->dict->file + ": " + where + ": monitorCoeffs needs 'points'");
        const auto& t = p->value;
        // [N] ( (x y z) ... )
        std::size_t pos = !t.empty() && t[0].kind == Token::Number ? 1 : 0;
        if (pos >= t.size() || !t[pos].is('(') || !t.back().is(')'))
            return Err(at(*mc->dict, *p) + where + ": points must be a list of points ((x y z) ...)");
        for (++pos; pos + 1 < t.size();) {
            auto v = foam::vectorOf(*mc->dict, t, pos, p->line, "points");
            if (!v) return Err(v.error());
            k.monitor.push_back({(*v)[0], (*v)[1], (*v)[2]});
        }
    } else {
        auto up = constVector(s, where, "upstreamPoint", false);
        if (!up) return Err(up.error() + " (or monitorCoeffs { points (...); })");
        k.monitor.push_back(*up);
    }
    if (k.monitor.empty())
        return Err(s.file + ": " + where + ": No cell is available for incoming velocity monitoring (no monitor point)");
    auto wf = sw("writeToFile", true);
    if (!wf) return Err(wf.error());
    k.writeToFile = *wf;
    for (auto [key, dst] : {std::pair{"writeFileStart", &k.writeStart}, {"writeFileEnd", &k.writeEnd}})
        if (const Entry* e = s.find(key)) {
            auto v = foam::scalarOf(s, *e);
            if (!v) return Err(v.error());
            *dst = *v;
        }
    if (const Entry* e = s.find("writePrecision")) {
        auto v = foam::scalarOf(s, *e);
        if (!v) return Err(v.error());
        if (*v < 0 || *v != std::floor(*v)) return Err(at(s, *e) + where + ": writePrecision must be an integer >= 0");
        k.writePrecision = std::min(int(*v), 17);
    }
    if (int(c.disks.size()) == FoamCase::kMaxDisks)
        return Err(s.file + ": " + where + std::format(": more than {} actuationDiskSource entries are not supported",
                                                       FoamCase::kMaxDisks));
    c.disks.push_back(std::move(k));
    return {};
}

// fv::options: ONE active constant specific momentum source on every cell
// (explicit part only), or actuationDiskSource entries (readDisk). Everything
// else simpleFoam would apply is rejected.
std::expected<void, string> readFvOptions(const Dict& d, const string& file, const PolyManifest& man, FoamCase& c)
{
    const string pre = file + ": fv::options (sources / constraints) are not supported";
    for (const Entry& e : d.entries) {
        if (e.key == "FoamFile") continue;
        if (e.directive) return Err(pre + ": directive '" + e.key + "'");
        if (!e.dict) return Err(pre + ": entry '" + e.key + "' is not a source dictionary");
        const Dict& s = *e.dict;
        auto type = needWord(s, "type");
        if (!type) return Err(type.error());
        if (*type == "actuationDiskSource") {
            if (auto r = readDisk(s, e.key, file, man, c); !r) return r;
            continue;
        }
        if (*type != "vectorSemiImplicitSource")
            return Err(pre + ": type '" + *type + "' (vectorSemiImplicitSource, actuationDiskSource)");
        if (auto r = onlyKeys(s, {"type", "active", "selectionMode", "volumeMode", "sources", "injectionRateSuSp"},
                              file + "/" + e.key);
            !r)
            return Err(pre + ": " + r.error());
        if (const Entry* a = s.find("active")) {
            auto v = foam::switchOf(s, *a);
            if (!v) return Err(v.error());
            if (!*v) return Err(pre + ": an inactive source (remove it)");
        }
        for (const auto& [key, want] : {std::pair{"selectionMode", "all"}, {"volumeMode", "specific"}}) {
            auto w = needWord(s, key);
            if (!w) return Err(w.error());
            if (*w != want) return Err(pre + ": " + key + " '" + *w + "' (" + want + " only)");
        }
        const Entry* src = s.find("sources");
        if (!src) src = s.find("injectionRateSuSp");
        if (!src || !src->dict) return Err(pre + ": '" + e.key + "' needs a 'sources' dictionary");
        const Dict& q = *src->dict;
        if (auto r = onlyKeys(q, {"U"}, file + "/" + e.key + "/sources"); !r) return Err(pre + ": " + r.error());
        const Entry* u = q.find("U");
        if (!u) return Err(pre + ": '" + e.key + "' has no U source");
        // ((Sx Sy Sz) Sp)
        const auto& t = u->value;
        std::size_t pos = 1;
        if (t.size() < 2 || !t[0].is('(')) return Err(pre + ": U source must be ((Sx Sy Sz) Sp)");
        auto v = foam::vectorOf(q, t, pos, u->line, "U");
        if (!v) return Err(v.error());
        if (pos + 2 != t.size() || t[pos].kind != Token::Number || !t[pos + 1].is(')'))
            return Err(pre + ": U source must be ((Sx Sy Sz) Sp)");
        if (t[pos].num != 0) return Err(pre + ": an implicit coefficient Sp != 0");
        if (c.uSource) return Err(pre + ": a second U source");
        c.uSource = FvVec3{(*v)[0], (*v)[1], (*v)[2]};
    }
    // fv::optionList::reset counts the dictionary entries: none = an empty
    // list, simpleFoam runs source-free (v2606 fvOptionList.C).
    if (!c.uSource && c.disks.empty())
        c.notes.push_back(file + ": no source in the file: fv::optionList is empty, the run is source-free (as simpleFoam)");
    if (c.uSource && !c.disks.empty())
        return Err(pre + ": vectorSemiImplicitSource together with actuationDiskSource");
    return {};
}

} // namespace

const char* fvUBcName(FvUBc b)
{
    static constexpr const char* n[] = {"fixedValue", "zeroGradient", "inletOutlet", "basicSymmetry",
                                        "freestreamVelocity", "empty", "cyclicAMI"};
    return n[std::size_t(b)];
}

const char* fvPBcName(FvPBc b)
{
    static constexpr const char* n[] = {"fixedValue", "zeroGradient", "freestreamPressure", "empty", "cyclicAMI"};
    return n[std::size_t(b)];
}

std::expected<FoamCase, string> parseFoamCase(const FoamCaseTexts& t, const PolyManifest& man)
{
    FoamCase c;
    c.patches.resize(man.patches.size());
    for (std::size_t i = 0; i < man.patches.size(); ++i) {
        c.patches[i].name = man.patches[i].name;
        c.patches[i].meshType = man.patches[i].type;
    }
    bool have = false;
    // The application picks the algorithm (and with it what fvSchemes / fvSolution must hold): read first.
    {
        auto cd = parseText(t, "system/controlDict", true, have);
        if (!cd) return Err(cd.error());
        if (const Entry* e = cd->find("application")) {
            auto w = foam::wordOf(*cd, *e);
            if (!w) return Err(w.error());
            c.pimple.on = *w == "pimpleFoam";
        }
    }
    if (!c.pimple.on &&
        std::ranges::any_of(man.patches, [](const FoamPatch& p) { return p.type == FoamPatchType::CyclicAMI; }))
        return Err("constant/polyMesh/boundary: cyclicAMI patches with simpleFoam are not built (application "
                   "pimpleFoam only)");
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

    if (c.pimple.on) { // before fvSchemes / fvSolution: mesh.dynamic() picks grad(pcorr), correctPhi, the pcorr solvers
        bool haveDyn = false;
        auto dm = parseText(t, "constant/dynamicMeshDict", false, haveDyn);
        if (!dm) return Err(dm.error());
        if (haveDyn)
            if (auto r = readDynamicMesh(*dm, man, c); !r) return Err(r.error());
        if (c.motion.on &&
            std::ranges::any_of(man.patches, [](const FoamPatch& p) { return p.type == FoamPatchType::Cyclic; }))
            return Err("constant/dynamicMeshDict: mesh motion with cyclic patches is not built (a moving cyclic pair's "
                       "transform, meshWave across it)");
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
    // fv::options: sources/constraints simpleFoam applies. fvOptions.C createIOobject reads
    // constant/fvOptions, and system/fvOptions ONLY when that one has no valid header.
    if (t.contains("constant/fvOptions") && t.contains("system/fvOptions"))
        return Err("constant/fvOptions and system/fvOptions both exist: OpenFOAM reads constant/fvOptions only "
                   "(system/fvOptions when constant's header is invalid); remove one");
    for (const char* f : {"constant/fvOptions", "system/fvOptions"}) {
        bool haveOpt = false;
        auto o = parseText(t, f, false, haveOpt);
        if (!o) return Err(o.error());
        if (haveOpt)
            if (auto r = readFvOptions(*o, f, man, c); !r) return Err(r.error());
    }
    if (c.pimple.on) {
        // pimpleFoam: no MRF, no fv::option source (docs/fvpimple.md); the mesh motion was read above.
        if (c.motion.on) {
            if (std::ranges::none_of(c.patches, [](const FvPatchBc& b) { return b.movingWall; }) &&
                std::ranges::any_of(c.patches, [](const FvPatchBc& b) { return b.meshType == FoamPatchType::Wall; }))
                c.notes.push_back("moving mesh: no wall patch is movingWallVelocity -- every fixedValue / noSlip wall "
                                  "keeps its LAB-frame value while its faces move (OpenFOAM does the same)");
        }
        if (std::ranges::any_of(c.mrf, [](const FvMrfZone& z) { return z.active; }))
            return Err("constant/MRFProperties: an active MRF zone with pimpleFoam is not built");
        if (c.uSource || !c.disks.empty())
            return Err("fvOptions: a momentum source with pimpleFoam is not built");
        if (c.ras.model == FvTurbModel::SpalartAllmaras)
            return Err("RASModel SpalartAllmaras with pimpleFoam is not built (kOmegaSST, laminar)");
    }
    // cyclicAMI (docs/fvpimple.md "cyclicAMI"): pimpleFoam only, the gated scheme set.
    if (std::ranges::any_of(man.patches, [](const FoamPatch& p) { return p.type == FoamPatchType::CyclicAMI; })) {
        c.ami = true; // (simpleFoam: refused before the fields are read)
        if (std::ranges::any_of(man.patches, [](const FoamPatch& p) { return p.type == FoamPatchType::Cyclic; }))
            return Err("constant/polyMesh/boundary: cyclic and cyclicAMI patches in one mesh are not built");
        if (!c.schemes.lapCorrected || !c.schemes.snGradCorrected)
            return Err("system/fvSchemes: cyclicAMI needs corrected (or limited corrected) laplacian and snGrad "
                       "schemes (uncorrected is not built on a coupled patch)");
        if (c.ras.model != FvTurbModel::Laminar && c.ras.exactDist)
            return Err("system/fvSchemes: wallDist exactDistance with cyclicAMI is not built (meshWave)");
        if (c.ras.model == FvTurbModel::KOmegaSST &&
            (c.ras.divK.kind == FvSDiv::LimitedLinear || c.ras.divOmega.kind == FvSDiv::LimitedLinear))
            return Err("system/fvSchemes: limitedLinear convection of k / omega with cyclicAMI is not built (upwind, "
                       "linear, linearUpwind)");
        c.notes.push_back("cyclicAMI: the pressure matrix is not symmetric (each side's own weights and "
                          "coefficients, as OpenFOAM's interface update); p / pcorr keep the flexible AMG-PCG");
    }
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

std::array<double, 12> fvForceCoeffRow(const FvForces& f, const FvVec3& fp, const FvVec3& fv, const FvVec3& mp,
                                       const FvVec3& mv)
{
    const FvForceCoeffs& k = *f.coeffs;
    const double pDyn = 0.5 * f.rhoInf * (k.magUInf * k.magUInf);
    const double fs = 1.0 / (k.Aref * pDyn + 1e-15), ms = 1.0 / (k.Aref * pDyn * k.lRef + 1e-15);
    // coordSys.localVector(v) = R^T v, the local axes the columns of R
    const auto local = [&](const FvVec3& v) { return FvVec3{dot3(k.e1, v), dot3(k.e2, v), dot3(k.e3, v)}; };
    const FvVec3 lfp = local(fp), lfv = local(fv), lmp = local(mp), lmv = local(mv);
    // forceComponents [c] = (pressure, viscous, internal = 0) of local component c
    struct C3
    {
        double p, v, i;
    };
    const auto F = [&](int c) { return C3{fs * lfp[std::size_t(c)], fs * lfv[std::size_t(c)], fs * 0.0}; };
    const auto M = [&](int c) { return C3{ms * lmp[std::size_t(c)], ms * lmv[std::size_t(c)], ms * 0.0}; };
    const auto sum = [](const C3& x) { return x.p + x.v + x.i; };
    const auto front = [&](int c0, int c1) {
        const C3 a = F(c0), b = M(c1);
        return sum(C3{0.5 * a.p + b.p, 0.5 * a.v + b.v, 0.5 * a.i + b.i});
    };
    const auto rear = [&](int c0, int c1) {
        const C3 a = F(c0), b = M(c1);
        return sum(C3{0.5 * a.p - b.p, 0.5 * a.v - b.v, 0.5 * a.i - b.i});
    };
    // Cd (0, 0), Cs (1, 2), Cl (2, 1); CmRoll 0, CmPitch 1, CmYaw 2
    return {sum(F(0)), front(0, 0), rear(0, 0), sum(F(2)), front(2, 1), rear(2, 1),
            sum(M(1)), sum(M(0)),   sum(M(2)),  sum(F(1)), front(1, 2), rear(1, 2)};
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
                           "system/fvOptions",
                           "constant/dynamicMeshDict"};
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
    string s = std::format("  case        {}\n  nu          {:g}\n", c.dir, c.nu);
    if (c.pimple.on) {
        const FvTimeCtl& t = c.pimple.time;
        static constexpr const char* wc[] = {"timeStep", "runTime", "adjustableRunTime"};
        s += std::format("  time        pimpleFoam: endTime {:g}, deltaT {:g}, {}; write {} {:g}\n", t.endTime, t.deltaT,
                         t.adjust ? std::format("adjustTimeStep maxCo {:g} maxDeltaT {:g}", t.maxCo, t.maxDeltaT)
                                  : string("fixed"),
                         wc[std::size_t(t.writeControl)], t.writeInterval);
    } else s += std::format("  iterations  {} (write every {})\n", c.endIter, c.writeInterval);
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
    auto rxo = [](const std::optional<double>& x) { return x ? std::format("{:g}", *x) : string("-"); };
    if (c.pimple.on) {
        const FvPimpleCtl& pc = c.pimple;
        s += std::format("  PIMPLE      nOuterCorrectors {}, nCorrectors {}, nNonOrthogonalCorrectors {}, "
                         "turbOnFinalIterOnly {}, ddtCorr yes, pRefCell {} pRefValue {:g}\n",
                         pc.nOuter, pc.nCorr, c.simple.nNonOrthCorr, pc.turbOnFinalIterOnly ? "yes" : "no",
                         c.simple.pRefCell, c.simple.pRefValue);
        s += std::format("  relax       p {} pFinal {} U {} UFinal {}\n",
                         c.relaxPGiven ? std::format("{:g}", c.relaxP) : string("-"), rxo(pc.relaxPFinal), rxo(c.relaxU),
                         rxo(pc.relaxUFinal));
        auto so = [](const FvSolverCtl& x) { return std::format("{} tol {:g} relTol {:g}", x.solver, x.tolerance, x.relTol); };
        s += std::format("  solvers     U {}; UFinal {}; p {}; pFinal {}\n", so(c.solverU), so(pc.solverUFinal),
                         so(c.solverP), so(pc.solverPFinal));
        if (c.motion.on) {
            for (const FvSolidBody& b : c.motion.bodies)
                s += std::format("  motion      {}solidBody rotatingMotion on {}: origin ({:g} {:g} {:g}) axis ({:g} {:g} "
                                 "{:g}) omega {:g} rad/s\n",
                                 b.name.empty() ? string() : b.name + ": ",
                                 b.cellZone.empty() ? string("the whole mesh") : "cellZone " + b.cellZone, b.origin[0],
                                 b.origin[1], b.origin[2], b.axis[0], b.axis[1], b.axis[2], b.omega);
            s += std::format("  moving mesh {}, correctPhi {}{}, checkMeshCourantNo {}\n",
                             c.motion.multi ? "dynamicMultiMotionSolverFvMesh" : "dynamicMotionSolverFvMesh",
                             pc.correctPhi ? "yes" : "no",
                             pc.correctPhi ? std::format(" (pcorr {}; pcorrFinal {})", so(pc.solverPcorr),
                                                         so(pc.solverPcorrFinal))
                                           : string(),
                             pc.checkMeshCourantNo ? "yes" : "no");
        }
    } else {
    s += std::format("  SIMPLE      nNonOrthogonalCorrectors {}, consistent {}, momentumPredictor {}, pRefCell {} "
                     "pRefValue {:g}\n",
                     c.simple.nNonOrthCorr, c.simple.consistent ? "yes" : "no", c.simple.momentumPredictor ? "yes" : "no",
                     c.simple.pRefCell, c.simple.pRefValue);
    s += std::format("  relax       p {} U {}\n", c.relaxPGiven ? std::format("{:g}", c.relaxP) : string("-"),
                     c.relaxU ? std::format("{:g}", *c.relaxU) : string("-"));
    s += std::format("  solvers     U {} tol {:g} relTol {:g}; p {} tol {:g} relTol {:g}\n", c.solverU.solver,
                     c.solverU.tolerance, c.solverU.relTol, c.solverP.solver, c.solverP.tolerance, c.solverP.relTol);
    }
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
        if (const FvPimpleCtl& pc = c.pimple; pc.on) {
            auto fin = [&](const char* n, const FvSolverCtl& x, const std::optional<double>& rl) {
                return std::format("{}Final {} tol {:g} relTol {:g} relax {}", n, x.solver, x.tolerance, x.relTol, rx(rl));
            };
            s += "  RAS final   " +
                 (sst ? fin("k", pc.solverKFinal, pc.relaxKFinal) + "; " + fin("omega", pc.solverOmegaFinal, pc.relaxOmegaFinal)
                      : fin("nuTilda", pc.solverNuTildaFinal, pc.relaxNuTildaFinal)) +
                 "\n";
        }
    }
    if (c.uSource)
        s += std::format("  source      U ({:g} {:g} {:g}) per unit mass on every cell (fvOptions "
                         "vectorSemiImplicitSource)\n",
                         (*c.uSource)[0], (*c.uSource)[1], (*c.uSource)[2]);
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
    for (const FvDisk& k : c.disks) {
        string mp;
        for (const FvVec3& p : k.monitor) mp += std::format("{}({:g} {:g} {:g})", mp.empty() ? "" : " ", p[0], p[1], p[2]);
        s += std::format("  source      {} actuationDiskSource Froude on cellZone {}: diskArea {:g} diskDir ({:g} {:g} "
                         "{:g}) Cp {:g} Ct {:g} (a = 1 - Cp/Ct = {:g}), monitor points {}{}\n",
                         k.name, k.cellZone, k.diskArea, k.dir[0], k.dir[1], k.dir[2], k.Cp, k.Ct, 1.0 - k.Cp / k.Ct, mp,
                         k.writeToFile ? "" : ", writeToFile no");
    }
    for (const FvForces& f : c.forces) {
        string ps;
        for (const auto& p : f.patches) ps += (ps.empty() ? "" : " ") + p;
        s += std::format("  forces      {} on ({}) rhoInf {:g} CofR ({:g} {:g} {:g})\n", f.name, ps, f.rhoInf, f.CofR[0],
                         f.CofR[1], f.CofR[2]);
        if (f.coeffs)
            s += std::format("  forceCoeffs {} dragDir ({:g} {:g} {:g}) liftDir ({:g} {:g} {:g}) magUInf {:g} lRef {:g} "
                             "Aref {:g}\n",
                             f.name, f.coeffs->e1[0], f.coeffs->e1[1], f.coeffs->e1[2], f.coeffs->e3[0], f.coeffs->e3[1],
                             f.coeffs->e3[2], f.coeffs->magUInf, f.coeffs->lRef, f.coeffs->Aref);
    }
    for (const auto& n : c.notes) s += "  note        " + n + "\n";
    return s;
}

} // namespace dns
