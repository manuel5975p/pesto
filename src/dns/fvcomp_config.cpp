#include "dns/fvcomp_config.hpp"

#include <tomlplusplus/toml.hpp>

#include <algorithm>
#include <cmath>
#include <format>
#include <sstream>

namespace dns {
namespace {

using Err = std::unexpected<std::string>;

bool number(const toml::node& n, double& out)
{
    if (const auto d = n.value_exact<double>()) out = *d;
    else if (const auto i = n.value_exact<std::int64_t>()) out = double(*i);
    else return false;
    return std::isfinite(out);
}

std::expected<std::array<double, 3>, std::string> vec3(const toml::node& n, const std::string& where)
{
    const toml::array* a = n.as_array();
    std::array<double, 3> v{};
    if (!a || a->size() != 3) return Err(where + ": needs an array of three numbers");
    for (std::size_t i = 0; i < 3; ++i)
        if (!number(*a->get(i), v[i])) return Err(where + ": needs an array of three finite numbers");
    return v;
}

std::expected<FvPatchKind, std::string> kindOf(std::string_view s)
{
    if (s == "wall") return FvPatchKind::Wall;
    if (s == "slip") return FvPatchKind::Slip;
    if (s == "inflow") return FvPatchKind::Inflow;
    if (s == "outflow") return FvPatchKind::Outflow;
    if (s == "far") return FvPatchKind::Far;
    if (s == "empty") return FvPatchKind::Empty;
    return Err(std::format("unknown patch type '{}'; one of: wall, slip, inflow, outflow, far, empty", s));
}

const char* kindName(FvPatchKind k)
{
    static constexpr const char* n[] = {"wall", "slip", "inflow", "outflow", "far", "empty"};
    return n[std::size_t(k)];
}

std::expected<void, std::string> needPositive(double v, const std::string& where)
{
    if (!(v > 0.0)) return Err(where + ": must be > 0");
    return {};
}

// The keys a kind reads; anything else in its table is an error.
std::vector<std::string_view> keysOf(FvPatchKind k)
{
    switch (k) {
    case FvPatchKind::Wall: return {"type", "thermal", "twall", "u", "omega", "origin", "force"};
    case FvPatchKind::Slip: return {"type", "force"};
    case FvPatchKind::Inflow: return {"type", "u", "t", "y", "jet"};
    case FvPatchKind::Outflow: return {"type", "p", "tback", "y", "burnt"};
    case FvPatchKind::Far: return {"type", "y"};
    case FvPatchKind::Empty: return {"type"};
    }
    return {"type"};
}

std::expected<FvPatchSpec, std::string> parsePatch(const std::string& name, const toml::table& t)
{
    const std::string w = "patch." + name;
    if (t.get("kind"))
        return Err(w + ": key 'kind' is the incompressible --bc schema; --physics compressible spells it type");
    const toml::node* ty = t.get("type");
    if (!ty || !ty->is_string()) return Err(w + ": needs type = \"wall|slip|inflow|outflow|far|empty\"");
    auto kind = kindOf(ty->value_or(std::string()));
    if (!kind) return Err(w + ": " + kind.error());
    FvPatchSpec p{.name = name, .kind = *kind};
    const auto allowed = keysOf(p.kind);
    for (const auto& [k, v] : t)
        if (std::ranges::find(allowed, k.str()) == allowed.end())
            return Err(std::format("{}: key '{}' does not belong to type {}", w, k.str(), kindName(p.kind)));
    auto num = [&](const char* key, double& out) -> std::expected<void, std::string> {
        const toml::node* n = t.get(key);
        if (!n) return {};
        if (!number(*n, out)) return Err(std::format("{}.{}: must be a finite number", w, key));
        return needPositive(out, w + "." + key);
    };
    auto vec = [&](const char* key, std::array<double, 3>& out) -> std::expected<void, std::string> {
        const toml::node* n = t.get(key);
        if (!n) return {};
        auto v = vec3(*n, w + "." + key);
        if (!v) return Err(v.error());
        out = *v;
        return {};
    };
    if (p.kind == FvPatchKind::Wall) {
        p.tallied = true;
        if (const toml::node* n = t.get("thermal")) {
            const std::string s = n->value_or(std::string());
            if (s != "adiabatic" && s != "isothermal")
                return Err(w + ".thermal: must be \"adiabatic\" or \"isothermal\"");
            p.isothermal = s == "isothermal";
        }
        if (t.contains("twall") && !p.isothermal) return Err(w + ".twall: only with thermal = \"isothermal\"");
        if (p.isothermal && !t.contains("twall")) return Err(w + ": an isothermal wall needs twall (T_w / T_inf)");
        if (auto r = num("twall", p.tWall); !r) return Err(r.error());
        if (auto r = vec("u", p.uWall); !r) return Err(r.error());
        if (auto r = vec("omega", p.omega); !r) return Err(r.error());
        if (auto r = vec("origin", p.origin); !r) return Err(r.error());
    }
    if (p.kind == FvPatchKind::Inflow) {
        p.hasU = t.contains("u");
        if (auto r = vec("u", p.uIn); !r) return Err(r.error());
        if (auto r = num("t", p.tIn); !r) return Err(r.error());
    }
    if (p.kind == FvPatchKind::Outflow) {
        if (auto r = num("p", p.pOut); !r) return Err(r.error());
        if (auto r = num("tback", p.tBack); !r) return Err(r.error());
    }
    if (const toml::node* n = t.get("y")) {
        const toml::array* a = n->as_array();
        if (!a || a->empty() || a->size() > p.y.size())
            return Err(std::format("{}.y: needs an array of 1 to {} mass fractions", w, p.y.size()));
        for (std::size_t i = 0; i < a->size(); ++i)
            if (!number(*a->get(i), p.y[i]) || p.y[i] < 0.0 || p.y[i] > 1.0)
                return Err(w + ".y: mass fractions are finite and in [0, 1]");
        p.hasY = true;
    }
    if (const toml::node* n = t.get("burnt")) {
        const auto b = n->value<bool>();
        if (!b) return Err(w + ".burnt: must be true or false");
        if (*b && (t.contains("p") || t.contains("tback")))
            return Err(w + ": burnt = true sets p and tback: give neither");
        p.burnt = *b;
    }
    if (const toml::node* n = t.get("jet")) {
        const auto b = n->value<bool>();
        if (!b) return Err(w + ".jet: must be true or false");
        if (*b && (t.contains("y") || t.contains("u") || t.contains("t")))
            return Err(w + ": jet = true sets u, t and y per face: give none of them");
        p.jet = *b;
    }
    if (const toml::node* n = t.get("force")) {
        const auto b = n->value<bool>();
        if (!b) return Err(w + ".force: must be true or false");
        p.tallied = *b;
    }
    return p;
}

} // namespace

std::expected<FvBcSpec, std::string> parseFvBc(std::string_view text)
{
    const auto res = toml::parse(text);
    if (!res) {
        std::ostringstream s;
        s << res.error().description() << " (line " << res.error().source().begin.line << ")";
        return Err(s.str());
    }
    const toml::table& root = res.table();
    FvBcSpec sp;
    for (const auto& [k, v] : root) {
        const std::string key(k.str());
        if (key == "reference") {
            const toml::table* t = v.as_table();
            if (!t) return Err("[reference] must be a table");
            for (const auto& [rk, rv] : *t) {
                const std::string r(rk.str());
                if (r == "length" || r == "area") {
                    double x = 0;
                    if (!number(rv, x) || !(x > 0)) return Err("reference." + r + ": must be a finite number > 0");
                    (r == "length" ? sp.length : sp.area) = x;
                } else if (r == "origin") {
                    auto a = vec3(rv, "reference.origin");
                    if (!a) return Err(a.error());
                    sp.origin = *a;
                    sp.hasOrigin = true;
                } else if (r == "direction" || r == "lift") {
                    auto a = vec3(rv, "reference." + r);
                    if (!a) return Err(a.error());
                    const double m = std::hypot((*a)[0], (*a)[1], (*a)[2]);
                    if (!(m > 0)) return Err("reference." + r + ": must not be zero");
                    auto& o = r == "direction" ? sp.dir : sp.lift;
                    for (std::size_t i = 0; i < 3; ++i) o[i] = (*a)[i] / m;
                } else
                    return Err("[reference]: unknown key '" + r + "' (length, area, direction, lift, origin)");
            }
        } else if (key == "patch") {
            const toml::table* t = v.as_table();
            if (!t) return Err("[patch.NAME] tables expected");
            for (const auto& [pk, pv] : *t) {
                const toml::table* pt = pv.as_table();
                if (!pt) return Err(std::format("patch.{}: must be a table", pk.str()));
                auto p = parsePatch(std::string(pk.str()), *pt);
                if (!p) return Err(p.error());
                sp.patches.push_back(std::move(*p));
            }
        } else
            return Err("unknown top-level key '" + key + "' (reference, patch)");
    }
    if (sp.patches.empty()) return Err("no [patch.NAME] table");
    const double dl = sp.dir[0] * sp.lift[0] + sp.dir[1] * sp.lift[1] + sp.dir[2] * sp.lift[2];
    if (std::abs(dl) > 1e-12) return Err("reference.lift must be normal to reference.direction");
    return sp;
}

std::string FvBcSpec::describe() const
{
    std::string s = std::format("  reference  L {:g}, area {}, direction ({:g} {:g} {:g}), lift ({:g} {:g} {:g})\n",
                                length, area > 0 ? std::format("{:g}", area) : std::string("L x depth"), dir[0],
                                dir[1], dir[2], lift[0], lift[1], lift[2]);
    for (const FvPatchSpec& p : patches) {
        s += std::format("  patch      {}: {}", p.name, kindName(p.kind));
        if (p.kind == FvPatchKind::Wall)
            s += p.isothermal ? std::format(", isothermal T_w/T_inf {:g}", p.tWall) : std::string(", adiabatic");
        if (p.tallied) s += ", in the force tally";
        s += "\n";
    }
    return s;
}

std::expected<void, std::string> checkFvBc(const FvBcSpec& spec, std::span<const FvMeshPatch> mesh)
{
    auto bound = [&](const std::string& n) {
        return std::ranges::find(spec.patches, n, &FvPatchSpec::name);
    };
    bool tally = false;
    for (const FvMeshPatch& m : mesh) {
        const auto it = bound(m.name);
        if (m.type == "wedge") return Err(std::format("patch `{}`: wedge patches are not supported (FaceMesh P8)", m.name));
        if (m.type == "cyclic") {
            if (it != spec.patches.end())
                return Err(std::format("patch `{}`: a cyclic patch is an internal face: do not bind it", m.name));
            continue;
        }
        if (m.type == "empty") {
            if (it != spec.patches.end() && it->kind != FvPatchKind::Empty)
                return Err(std::format("patch `{}`: an `empty` patch takes type = \"empty\" only", m.name));
            continue;
        }
        if (it == spec.patches.end())
            return Err(std::format("patch `{}` ({}) has no [patch.{}] table in the --bc file", m.name, m.type, m.name));
        if (it->kind == FvPatchKind::Empty)
            return Err(std::format("patch `{}`: type = \"empty\" needs an `empty` mesh patch, it is {}", m.name, m.type));
        tally = tally || it->tallied;
    }
    for (const FvPatchSpec& p : spec.patches)
        if (std::ranges::find(mesh, p.name, &FvMeshPatch::name) == mesh.end())
            return Err(std::format("--bc binds patch `{}`, which the mesh does not have", p.name));
    if (tally && !fvMeshIs2D(mesh) && !(spec.area > 0))
        return Err("a 3D mesh with a force tally needs reference.area in the --bc file");
    return {};
}

std::string fvBcFlameKey(const FvBcSpec& spec)
{
    if (spec.hasOrigin) return "reference.origin";
    for (const FvPatchSpec& p : spec.patches) {
        if (p.hasY) return "patch." + p.name + ".y";
        if (p.burnt) return "patch." + p.name + ".burnt";
        if (p.jet) return "patch." + p.name + ".jet";
    }
    return {};
}

bool fvMeshIs2D(std::span<const FvMeshPatch> mesh)
{
    return std::ranges::any_of(mesh, [](const FvMeshPatch& m) { return m.type == "empty"; });
}

} // namespace dns
