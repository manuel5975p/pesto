// Shared validator idioms (internal.hpp), the enum name tables, error formatting, --scene model.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"

#include "dns/fvimodules.hpp"

#include "dns/scene.hpp"
#include "dns/shockflux.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace dns {
using namespace cfg;
} // namespace dns

namespace dns::cfg {

bool finitePos(double v) { return std::isfinite(v) && v > 0; }

bool hasSuffix(std::string_view s, std::string_view suf)
{
    return s.size() > suf.size() && s.substr(s.size() - suf.size()) == suf;
}

} // namespace dns::cfg

namespace dns::cfg {
namespace {

// Table lookups behind every name()/FromName(): index == the enum value.
const char* nameAt(std::span<const char* const> n, std::size_t i)
{
    return i < n.size() ? n[i] : "?";
}

template <class E>
std::optional<E> fromName(std::span<const char* const> n, std::string_view s)
{
    const auto it = std::ranges::find(n, s);
    if (it == n.end()) return std::nullopt;
    return E(it - n.begin());
}

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

std::unexpected<ConfigError> bad(int code, const char* field, std::string msg)
{
    return std::unexpected(ConfigError{code, field, std::move(msg)});
}

bool remapsMesh(const RawConfig& r) { return r.stretch > 0 || !r.focal.empty(); }

ConfigError err(int code, const char* field, std::string msg)
{
    return ConfigError{code, field, std::move(msg)};
}

std::string bodyKeywords()
{
    std::string s;
    for (const char* n : bodyKindNames())
    {
        const std::string_view v(n);
        if (v == "mesh" || v == "scene") continue;
        if (!s.empty()) s += ", ";
        s += v;
    }
    return s;
}

std::optional<ConfigError> rejectGiven(const RawConfig& r,
                                       std::initializer_list<const char*> flags,
                                       std::string_view reason)
{
    for (const char* f : flags)
        if (r.given.contains(f))
            return ConfigError{2, f, std::string(f) + " " + std::string(reason)};
    return std::nullopt;
}

std::optional<ConfigError> fvIncompModules(const RawConfig& r, bool thermal)
{
    using M = FviModule;
    FviModuleSet s;
    if (r.les > 0) s.set(std::size_t(r.sgs == SgsModel::Sigma ? M::LesSigma : M::LesSmag));
    const bool rans = r.given.contains("--rans") && r.rans != int(RansModel::None);
    if (rans) s.set(std::size_t(r.rans == int(RansModel::SaNeg) ? M::RansSa : M::RansSst));
    if (thermal) {
        s.set(std::size_t(M::Temperature));
        s.set(std::size_t(M::Gravity));
    }
    auto ok = fviModulesCheck(s);
    if (ok) return std::nullopt;
    return ConfigError{2, rans ? "--rans" : "--les", ok.error() + " (docs/fvincomp.md \"Incompressible combinations\")"};
}

std::optional<ConfigError> validateFrameOutputs(const RawConfig& r, Case& c)
{
    c.fdt = r.fdt;
    // pdt/wdt default to the .bin cadence, so a run that asks only for --fdt
    // gets all three kinds on one clock.
    c.pdt = r.pdt > 0 ? r.pdt : c.fdt;
    c.wdt = r.wdt > 0 ? r.wdt : c.fdt;
    for (auto [v, f] : {std::pair{c.fdt, "--fdt"}, {c.pdt, "--pdt"},
                        {c.wdt, "--wdt"}})
        if (!finitePos(v)) return ConfigError{2, f, "must be finite and > 0"};
    c.planes = r.planes;
    if (c.planes != 1 && c.planes != 3)
        return ConfigError{2, "--planes",
                           "must be 1 (centre only) or 3 (centre +/- n/16)"};
    c.pngDown = r.pngDown;
    if (c.pngDown < 1)
        return ConfigError{2, "--png-down",
                           "must be >= 1 (1 = full-resolution PNGs)"};
    if (c.pngDown > 1 && (c.grid.nx % c.pngDown || c.grid.ny % c.pngDown))
        return ConfigError{2, "--png-down",
                           "must divide both nx and ny evenly for box-averaging"};
    return std::nullopt;
}

std::optional<ConfigError> checkMach(double mach, bool shock)
{
    if (finitePos(mach) && mach < 1) return std::nullopt;
    if (shock) return ConfigError{2, "--mach", "must be in (0,1): the characteristic boundaries are subsonic"};
    return ConfigError{2, "--mach", "must be in (0,1) -- the flux is "
                                    "Kennedy-Gruber central with no shock "
                                    "capturing"};
}

std::optional<ConfigError> shockRouteScope(const RawConfig& r)
{
    if (r.shock == ShockMode::None) return std::nullopt;
    // dns flow | flame --physics compressible --polymesh: the FV route (docs/nozzle.md; the flame's since
    // masterplan_combnozzle P3, Shock x C0).
    const bool fvPoly = (r.sub == Subcommand::Flow || r.sub == Subcommand::Flame) && r.physics == Physics::Compressible &&
                        r.given.contains("--polymesh");
    const std::initializer_list<const char*> always = {"--ogrid", "--stretch", "--focal", "--rans", "--steady",
                                                       "--precond", "--rotor-mach", "--rotor-frame", "--thin-seal", "--penalize",
                                                       "--nscbc-strip", "--pot"};
    if (auto e = rejectGiven(r, always,
                             "is not built with --shock (uniform Cartesian mesh, structured compressible; "
                             "docs/shock.md)");
        e)
        return e;
    if (!fvPoly)
        if (auto e = rejectGiven(r, {"--polymesh", "--bc"},
                                 "is not built with --shock (uniform Cartesian mesh, structured compressible; "
                                 "docs/shock.md)");
            e)
            return e;
    if (r.physics != Physics::Compressible) return err(2, "--shock", "is compressible only");
    if (!fvPoly && r.solver != int(SolverKind::Structured))
        return err(2, "--shock",
                   std::format("is not built on --solver {}", r.solver == int(SolverKind::Fv) ? "fv" : "simple"));
    return std::nullopt;
}

std::optional<ConfigError> shockFlagScope(const RawConfig& r, const Case& c)
{
    if (r.shock == ShockMode::None) return std::nullopt;
    if (c.sub == Subcommand::Flow && !r.given.contains("--polymesh")) {
        if (c.far != FarField::Nscbc) return err(2, "--shock", "needs --far nscbc");
        if (c.bodyKind == BodyKind::Floor) return err(2, "--shock", "is not built with --body floor");
    }
    if (c.sub == Subcommand::Flame) {
        const char* const modes[] = {"premixed", "bunsen", "diffusion", "candle", "chamber"};
        if (r.flameMode != 0 && r.flameMode != kFlameModeChamber)
            return err(2, "--shock",
                       std::format("is not built for --mode {}: premixed and chamber", modes[r.flameMode]));
        if (r.given.contains("--equivalence"))
            return err(2, "--shock", "is not built with --equivalence (two reactants)");
    }
    if (r.shock != ShockMode::Hll) return std::nullopt;
    const std::string noGate = " is not built with --shock hll (no gate yet; --shock sense measures it)";
    if (c.sub == Subcommand::Flame) {
        for (const char* f : {"--tfles", "--turb", "--les"})
            if (r.given.contains(f)) return err(2, f, f + noGate);
        if (c.mach > kShockMachMax)
            return err(2, "--mach",
                       std::format("{:g} above {:g}: the flame trips the sensor (bracket in dns/shockflux.hpp)", c.mach,
                                   kShockMachMax));
    }
    if (c.sub == Subcommand::Bench) return std::nullopt; // a fixed dt, no --cfl
    // The FV flame (polyMesh): the Sigma form is the axis sum, cap kShockCflAxisSum 1.2 (the nozzle
    // route's); the flame's own (0, 1] is inside it, not the structured per-axis 0.6.
    if (c.sub == Subcommand::Flame && r.given.contains("--polymesh")) return std::nullopt;
    const double cap = kShockCflAxisSum / (c.grid.nz > 1 ? 3 : 2);
    if (c.cfl > cap + 1e-12)
        return err(2, "--cfl",
                   std::format("must be <= {:g} with --shock hll: the axis sum {:g} is measured stable, 1.4 diverges",
                               cap, kShockCflAxisSum));
    return std::nullopt;
}

void shockCadence(const RawConfig& r, Case& c)
{
    c.shockDtAuto = !r.given.contains("--dt-recompute");
    if (!c.shockDtAuto && c.dtRecompute > 1)
        c.warnings.push_back(std::format("--dt-recompute {} under --shock hll: a dt held over steps diverged a 3D Sod "
                                         "tube from rest at 16 (docs/shock.md)",
                                         c.dtRecompute));
}

std::optional<ConfigError> checkNz(int nz, const char* belowOne)
{
    if (nz < 1) return ConfigError{2, "--nz", belowOne};
    if (nz != 1 && nz < 4)
        return ConfigError{2, "--nz", "must be 1 (native 2D) or >= 4"};
    return std::nullopt;
}

std::optional<ConfigError> checkCells(std::size_t cells)
{
    if (cells > std::size_t(2147483647))
        return ConfigError{2, "--n", "grid exceeds INT_MAX cells"};
    return std::nullopt;
}

std::expected<Grid, ConfigError> makeBox(int nx, int ny, int nz, double lx,
                                         double ly, double lz)
{
    if (auto e = checkCells(std::size_t(nx) * ny * nz); e)
        return std::unexpected(*e);
    return Grid::make(nx, ny, nz, lx, ly, nz == 1 ? 1.0 : lz);
}

std::optional<ConfigError> resolveCfl(const RawConfig& r, Case& c,
                                      double fallback)
{
    c.cfl = r.given.contains("--cfl") ? r.cfl : fallback;
    if (!(c.cfl > 0 && c.cfl <= 1))
        return ConfigError{2, "--cfl", "must be in (0,1]"};
    return std::nullopt;
}

std::optional<ConfigError> resolveDtRecompute(const RawConfig& r, Case& c,
                                              int fallback)
{
    c.dtRecompute = r.given.contains("--dt-recompute") ? r.dtRecompute : fallback;
    if (c.dtRecompute < 1)
        return ConfigError{2, "--dt-recompute", "must be >= 1"};
    return std::nullopt;
}

std::optional<ConfigError> checkProfileDt(const RawConfig& r, Case& c)
{
    c.profileDt = r.profileDt;
    if (!(std::isfinite(c.profileDt) && c.profileDt >= 0))
        return ConfigError{2, "--profile-dt", "must be finite and >= 0 (0 = off)"};
    return std::nullopt;
}

std::optional<ConfigError> resolveEmit(Case& c,
                                       std::initializer_list<Emit> fallback)
{
    if (c.emit.empty()) c.emit.insert(fallback);
    if (c.emit.contains(Emit::None) && c.emit.size() > 1)
        return ConfigError{2, "--emit",
                           "--emit none cannot be combined with other kinds"};
    return std::nullopt;
}

std::optional<ConfigError> checkWallGrid(const RawConfig& r)
{
    if (r.n < 4) return ConfigError{2, "--n", "must be >= 4"};
    if (r.ny < 8)
        return ConfigError{2, "--ny", "must be >= 8 fluid cells (wall-normal "
                                      "resolution; two ghost rows are added)"};
    return checkNz(r.nz, "must be >= 1 (interior cells; 1 = native 2D)");
}

std::optional<ConfigError> validateWallRun(const RawConfig& r, Case& c,
                                           double cflFallback)
{
    if (auto e = resolveCfl(r, c, cflFallback); e) return e;
    c.kick = r.kick;
    if (!(std::isfinite(c.kick) && c.kick >= 0))
        return ConfigError{2, "--kick", "must be finite and >= 0"};
    if (!finitePos(r.tt)) return ConfigError{2, "--tt", "must be finite and > 0"};
    c.ft = r.tt;
    if (auto e = checkProfileDt(r, c); e) return e;
    if (auto e = resolveDtRecompute(r, c, 16); e) return e;
    if (auto e = validateFrameOutputs(r, c); e) return e;
    return resolveEmit(c, {Emit::Stats});
}

std::optional<BodyKind> bodyKindFromName(std::string_view s)
{
    return fromName<BodyKind>(bodyKindNames(), s);
}

} // namespace dns::cfg

namespace dns {

// Index == the enum value; a static_assert on the size would not catch a
// TRANSPOSED pair, so the tables are written in enum order and read only
// through name()/*FromName(). The order is also the order every "one of:"
// message lists them in.
std::span<const char* const> bodyKindNames()
{
    static constexpr const char* kNames[] = {"none",  "cylinder", "sphere",
                                             "cube",  "lens",     "fence",
                                             "floor", "mesh",     "scene"};
    return kNames;
}

std::span<const char* const> farFieldNames()
{
    static constexpr const char* kNames[] = {"tunnel", "fringe", "sponge",
                                             "nscbc", "periodic"};
    return kNames;
}

std::span<const char* const> physicsNames()
{
    static constexpr const char* kNames[] = {"incompressible", "compressible",
                                             "lowmach"};
    return kNames;
}

std::span<const char* const> sgsModelNames()
{
    static constexpr const char* kNames[] = {"smag", "sigma"};
    return kNames;
}

std::span<const char* const> nscbcAbcNames()
{
    static constexpr const char* kNames[] = {"yooim", "adaptive", "modal"};
    return kNames;
}

std::span<const char* const> shockNames()
{
    static constexpr const char* kNames[] = {"none", "hll", "sense"};
    return kNames;
}

std::span<const char* const> subcommandNames()
{
    static constexpr const char* kNames[] = {"flow",    "tgv",  "hit",
                                             "bench",   "channel", "conv",
                                             "prop",    "flame", "mesh"};
    return kNames;
}

std::span<const char* const> steadySchemeNames()
{
    static constexpr const char* kNames[] = {"explicit", "implicit"};
    return kNames;
}

const char* name(BodyKind b) { return nameAt(bodyKindNames(), std::size_t(b)); }

const char* name(FarField f) { return nameAt(farFieldNames(), std::size_t(f)); }

const char* name(Physics p) { return nameAt(physicsNames(), std::size_t(p)); }

const char* name(SgsModel m) { return nameAt(sgsModelNames(), std::size_t(m)); }

const char* name(NscbcAbc a) { return nameAt(nscbcAbcNames(), std::size_t(a)); }

const char* name(ShockMode m) { return nameAt(shockNames(), std::size_t(m)); }

const char* name(Subcommand s)
{
    return nameAt(subcommandNames(), std::size_t(s));
}

const char* name(SteadyScheme s)
{
    return nameAt(steadySchemeNames(), std::size_t(s));
}

std::optional<Subcommand> subcommandFromName(std::string_view s)
{
    return fromName<Subcommand>(subcommandNames(), s);
}

std::optional<SgsModel> sgsModelFromName(std::string_view s)
{
    return fromName<SgsModel>(sgsModelNames(), s);
}

std::string lesRunLabel(SgsModel m, double cs, bool twoD)
{
    assert(cs > 0);
    return std::format("LES {} {}={:g} (filtered: grid- and Cs-dependent){}",
                       name(m), m == SgsModel::Sigma ? "Csigma" : "Cs", cs,
                       twoD ? " (2D: qualitative only, no vortex stretching)"
                            : "");
}

FarField defaultFar(Physics p)
{
    return p == Physics::Compressible ? FarField::Nscbc : FarField::Tunnel;
}

std::span<const FarField> farChoices(Physics p)
{
    static constexpr FarField kComp[] = {FarField::Nscbc, FarField::Sponge,
                                         FarField::Periodic};
    static constexpr FarField kIncomp[] = {FarField::Tunnel, FarField::Fringe,
                                           FarField::Periodic};
    return p == Physics::Compressible ? std::span<const FarField>(kComp)
                                      : std::span<const FarField>(kIncomp);
}

bool farAllowed(Physics p, FarField f)
{
    return std::ranges::find(farChoices(p), f) != farChoices(p).end();
}

BodyOptions bodyOptions(BodyKind b)
{
    switch (b)
    {
    case BodyKind::Cylinder: return {.dimples = true};   // 2D disks
    case BodyKind::Sphere: return {.dimples = true};     // 3D caps
    case BodyKind::Lens: return {.aspect = true};
    case BodyKind::Floor: return {.bumps = true};        // trip spikes
    case BodyKind::Mesh: return {.mesh = true};          // scale, robust
    default: return {};                                  // none/cube/fence/scene
    }
}

std::string format(const ConfigError& e)
{
    std::string s = "dns: ";
    if (!e.field.empty()) s += e.field + ": ";
    return s + e.message + "\n";
}

// ---------------------------------------------------------------------------

std::expected<void, ConfigError> applySceneModel(const Scene& s, RawConfig& r)
{
    const bool userLes = r.given.contains("--les");
    if (s.model == SceneModel::Les)
    {
        if (userLes && r.les != s.cs)
            return std::unexpected(ConfigError{
                2, "--les",
                "the scene selects model les with cs=" + std::to_string(s.cs)
                    + "; a conflicting --les was given (drop it or change the "
                      "scene)"});
        if (r.given.contains("--sgs") && r.sgs != s.sgs)
            return std::unexpected(ConfigError{
                2, "--sgs",
                std::string("the scene selects sgs = ") + name(s.sgs)
                    + "; a conflicting --sgs was given (drop it or change the "
                      "scene)"});
        r.les = s.cs;
        r.sgs = s.sgs;
    }
    else if (userLes && r.les > 0)
        return std::unexpected(ConfigError{
            2, "--les", "the scene selects model dns; drop --les or set the "
                        "scene model to les"});
    return {};
}

} // namespace dns
