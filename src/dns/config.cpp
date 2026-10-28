#include "dns/config.hpp"
#include "dns/numparse.hpp"

#include "dns/ogrid.hpp"

#include "dns/precond.hpp"
#include "dns/pseudo_time.hpp"
#include "dns/rans_label.hpp"
#include "dns/remap.hpp"
#include "dns/scene.hpp"
#include "dns/steady_stop.hpp"
#include "dns/transport.hpp"
#include "dns/units.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

namespace dns {
namespace {

using std::numbers::pi;

// The --dry-run tail of a --compensated time march: it changes the bits, so
// the resolved case says so.
constexpr const char* kCompensatedNote =
    " [compensated state: FP32 hi + lo per field]";

bool finitePos(double v) { return std::isfinite(v) && v > 0; }

bool hasSuffix(std::string_view s, std::string_view suf)
{
    return s.size() > suf.size() && s.substr(s.size() - suf.size()) == suf;
}

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

std::unexpected<ConfigError> bad(int code, const char* field, std::string msg)
{
    return std::unexpected(ConfigError{code, field, std::move(msg)});
}

// The run asks for a remapped (curvilinear) Cartesian mesh: the ONE spelling
// of "not the uniform mesh" that the sink default, the explicit-sink rejection
// and the modal closure all test. The default and the rejections stay separate
// decisions (a default that does not apply is absence; an explicit flag that
// does not apply is an error) -- only the predicate is shared.
bool remapsMesh(const RawConfig& r) { return r.stretch > 0 || !r.focal.empty(); }

// The same error as a validator stage's result (std::optional<ConfigError>).
ConfigError err(int code, const char* field, std::string msg)
{
    return ConfigError{code, field, std::move(msg)};
}

// The --body keywords, in one comma-separated string built from the shared
// table -- so a body kind added to the enum lists itself in both --body error
// messages. Mesh and Scene are excluded: they are kinds the validator assigns
// from a path or from --scene, not words the user may type.
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

// Rejects whichever of `flags` the user actually GAVE -- a flag left at its
// default is not an error, only one that was typed and cannot apply. The
// message is the flag's own name followed by `reason`, which is the shape
// every such rejection in this file uses.
//
// Returns the first offender, or nullopt when none was given:
//   if (auto e = rejectGiven(r, {"--mach"}, "is compressible-only"); e)
//       return std::unexpected(*e);
std::optional<ConfigError> rejectGiven(const RawConfig& r,
                                       std::initializer_list<const char*> flags,
                                       std::string_view reason)
{
    for (const char* f : flags)
        if (r.given.contains(f))
            return ConfigError{2, f, std::string(f) + " " + std::string(reason)};
    return std::nullopt;
}

// Frame cadences and the two PNG shaping flags. Every subcommand that writes
// frames takes exactly these rules; they used to be spelled out once per
// branch, and the copies had already drifted apart in their wording.
// Pre: c.grid is resolved (--png-down must divide it).
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

// --- validator idioms ---------------------------------------------------------
// Each rule below was spelled out per subcommand; one definition each now, with
// the message text unchanged. Every helper returns the error rather than
// throwing it, so the caller's check ORDER -- first error wins -- is untouched.

// The compressible Mach range.
std::optional<ConfigError> checkMach(double mach)
{
    if (!(finitePos(mach) && mach < 1))
        return ConfigError{2, "--mach", "must be in (0,1) -- the flux is "
                                        "Kennedy-Gruber central with no shock "
                                        "capturing"};
    return std::nullopt;
}

// The z cell count: 1 is the exact 2D restriction, otherwise >= 4.
// `belowOne` is the subcommand's own wording for nz < 1.
std::optional<ConfigError> checkNz(int nz, const char* belowOne)
{
    if (nz < 1) return ConfigError{2, "--nz", belowOne};
    if (nz != 1 && nz < 4)
        return ConfigError{2, "--nz", "must be 1 (native 2D) or >= 4"};
    return std::nullopt;
}

// The int indexing the whole tree assumes.
std::optional<ConfigError> checkCells(std::size_t cells)
{
    if (cells > std::size_t(2147483647))
        return ConfigError{2, "--n", "grid exceeds INT_MAX cells"};
    return std::nullopt;
}

// The one constructor of a validated box: rejects a grid over INT_MAX cells and
// is the ONE enforcement point of `nz == 1 => hz == 1` (lz is pinned to 1).
std::expected<Grid, ConfigError> makeBox(int nx, int ny, int nz, double lx,
                                         double ly, double lz)
{
    if (auto e = checkCells(std::size_t(nx) * ny * nz); e)
        return std::unexpected(*e);
    return Grid::make(nx, ny, nz, lx, ly, nz == 1 ? 1.0 : lz);
}

// --cfl, or the subcommand's own default when it was not given.
std::optional<ConfigError> resolveCfl(const RawConfig& r, Case& c,
                                      double fallback)
{
    c.cfl = r.given.contains("--cfl") ? r.cfl : fallback;
    if (!(c.cfl > 0 && c.cfl <= 1))
        return ConfigError{2, "--cfl", "must be in (0,1]"};
    return std::nullopt;
}

// --dt-recompute, or the subcommand's own default when it was not given.
std::optional<ConfigError> resolveDtRecompute(const RawConfig& r, Case& c,
                                              int fallback)
{
    c.dtRecompute = r.given.contains("--dt-recompute") ? r.dtRecompute : fallback;
    if (c.dtRecompute < 1)
        return ConfigError{2, "--dt-recompute", "must be >= 1"};
    return std::nullopt;
}

// --profile-dt: a dump interval, 0 = off.
std::optional<ConfigError> checkProfileDt(const RawConfig& r, Case& c)
{
    c.profileDt = r.profileDt;
    if (!(std::isfinite(c.profileDt) && c.profileDt >= 0))
        return ConfigError{2, "--profile-dt", "must be finite and >= 0 (0 = off)"};
    return std::nullopt;
}

// An empty --emit set takes the subcommand's own default; `none` is exclusive.
std::optional<ConfigError> resolveEmit(Case& c,
                                       std::initializer_list<Emit> fallback)
{
    if (c.emit.empty()) c.emit.insert(fallback);
    if (c.emit.contains(Emit::None) && c.emit.size() > 1)
        return ConfigError{2, "--emit",
                           "--emit none cannot be combined with other kinds"};
    return std::nullopt;
}

// The wall-bounded boxes (channel, conv) share their grid-count prologue and
// their run epilogue rule for rule and word for word; the physics between the
// two is theirs alone.
//
// Prologue: the lateral, wall-normal and spanwise FLUID cell counts.
std::optional<ConfigError> checkWallGrid(const RawConfig& r)
{
    if (r.n < 4) return ConfigError{2, "--n", "must be >= 4"};
    if (r.ny < 8)
        return ConfigError{2, "--ny", "must be >= 8 fluid cells (wall-normal "
                                      "resolution; two ghost rows are added)"};
    return checkNz(r.nz, "must be >= 1 (interior cells; 1 = native 2D)");
}

// Epilogue: --cfl, --kick, --tt (c.ft; the caller sets c.tEnd from it),
// --profile-dt, --dt-recompute (default 16), the frame cadences and --emit
// (default stats). Pre: c.grid is resolved.
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

// The inverse of name(BodyKind); nullopt when `s` is not a body keyword (a
// mesh path, say, which the caller recognises by suffix instead).
std::optional<BodyKind> bodyKindFromName(std::string_view s)
{
    return fromName<BodyKind>(bodyKindNames(), s);
}

} // namespace

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
    static constexpr const char* kNames[] = {"incompressible", "compressible"};
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

std::span<const char* const> subcommandNames()
{
    static constexpr const char* kNames[] = {"flow",    "tgv",  "hit",
                                             "bench",   "channel", "conv",
                                             "prop",    "flame"};
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

namespace {

// The per-subcommand halves of validate(). Each takes the Case the shared
// prologue has already filled in and finishes it, so the split falls exactly
// where the rules stop being common: no subcommand's rules are visible to
// another's, and the shared prologue is the only thing every case runs.

// --- bench ------------------------------------------------------------------
std::expected<Case, ConfigError> validateBench(const RawConfig& r, Case c)
{
    if (r.steps < 1) return bad(2, "--steps", "must be >= 1");
    c.steps = r.steps;
    // bench builds an n^3 box, so it needs the grid guards every other
    // branch applies and this one was missing: `--n 1300` overflowed the
    // int indexing the tree assumes, and `--n 1` produced nz == 1 with
    // hz == 2pi, violating the validator-enforced `nz == 1 => hz == 1`.
    // n >= 4 forecloses both the degenerate-z case (bench has no 2D mode:
    // the box is always n^3) and any n too small to time meaningfully.
    if (r.n < 4) return bad(2, "--n", "must be >= 4 (bench is an n^3 box)");
    auto box = makeBox(r.n, r.n, r.n, 2 * pi, 2 * pi, 2 * pi);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    c.nzInterior = r.n;
    // bench times the SOLVER the user names, so it needs the physics
    // parameters the solver reads: viscosity, Mach (compressible pressure
    // level) and the SGS constant. Before, --physics compressible parsed
    // and then ran the incompressible stepper anyway.
    c.nu = r.nu > 0 ? r.nu : 1.0 / r.re;
    if (!std::isfinite(c.nu) || c.nu < 0)
        return bad(2, "--nu", "must be finite and >= 0");
    c.mach = r.mach;
    if (r.physics == Physics::Compressible) {
        if (auto e = checkMach(c.mach); e) return std::unexpected(*e);
    } else if (r.given.contains("--mach")) {
        return bad(2, "--mach", "--mach is compressible-only");
    }
    c.cs = r.les;
    if (!(std::isfinite(c.cs) && c.cs >= 0))
        return bad(2, "--les", "must be finite and >= 0");
    if (c.cs > 0 && r.physics != Physics::Compressible)
        return bad(2, "--les", "bench --les times the compressible SGS path; "
                               "add --physics compressible (the incompressible "
                               "eddy viscosity is a runner-level substep, not "
                               "part of the periodic stepper this benchmarks)");
    c.sgsModel = r.sgs;
    if (r.given.contains("--sgs") && c.cs == 0)
        return bad(2, "--sgs", "selects the SGS model for --les; give --les CS "
                               "(> 0) or drop --sgs");
    c.lesLabel = c.cs > 0;
    c.meta = {.body = "none",
              .farField = "periodic",
              .physics = name(c.physics),
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

// --- tgv / hit --------------------------------------------------------------
std::expected<Case, ConfigError> validatePeriodic(const RawConfig& r, Case c)
{
    const int nz = r.given.contains("--nz") ? r.nz : r.n;
    if (auto e = checkNz(nz, "must be >= 1"); e) return std::unexpected(*e);
    if (r.sub == Subcommand::Hit && nz < 4)
        return bad(2, "--nz", "hit needs nz >= 4 (spectral initial condition)");
    auto box = makeBox(r.n, r.n, nz, 2 * pi, 2 * pi, 2 * pi);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    c.nzInterior = nz;
    if (auto e = resolveCfl(r, c, 0.9); e) return std::unexpected(*e);
    c.ft = r.ft;
    if (r.sub == Subcommand::Tgv && !finitePos(c.ft))
        return bad(2, "--ft", "must be finite and > 0");
    c.tEnd = c.ft;
    c.nu = r.nu > 0 ? r.nu : 1.0 / r.re;
    if (!std::isfinite(c.nu) || c.nu < 0)
        return bad(2, "--nu", "must be finite and >= 0");
    c.urms = r.urms;
    c.spin = r.spin;
    c.stats = r.stats;
    if (r.sub == Subcommand::Hit)
    {
        if (!finitePos(r.urms)) return bad(2, "--urms", "must be finite and > 0");
        if (!(r.spin >= 0 && r.stats > 0))
            return bad(2, "--spin", "--spin >= 0 and --stats > 0 required");
        c.nu = r.given.contains("--nu") ? r.nu : 1.5e-5;
        // The box is sized from nu (h = 1.5 eta): nu = 0 has no design.
        if (!finitePos(c.nu))
            return bad(2, "--nu", "hit sizes the box from nu (h = 1.5 eta): "
                                  "must be finite and > 0");
    }
    if (c.emit.contains(Emit::Error) && nz != 1)
        return bad(2, "--emit", "--emit error compares against the 2D analytic "
                                "Taylor-Green solution and requires --nz 1");
    if (c.emit.empty()) c.emit.insert(Emit::Stats);
    c.fdt = r.fdt;
    c.meta = {.re = r.re,
              .frameDt = c.fdt,
              .body = "none",
              .farField = "periodic",
              .physics = name(c.physics),
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

// --- channel ----------------------------------------------------------------

// The channel's RANS knobs (docs/rans.md). Every rule rejects, never
// ignores; --rans none with no --steady leaves the case untouched.
// Pre: the grid, wall and run epilogue are resolved (c.kick from --kick).
std::optional<ConfigError> channelRans(const RawConfig& r, Case& c)
{
    c.rans = RansModel(r.rans);
    const bool rans = c.rans != RansModel::None;
    if (rans)
    {
        if (r.given.contains("--kick") && r.kick > 0)
            return err(2, "--kick", "a RANS field is an ensemble mean; there "
                                    "is nothing to trip");
        c.kick = 0;
    }
    c.steadyTol = r.steady;
    if (!(std::isfinite(c.steadyTol) && c.steadyTol >= 0))
        return err(2, "--steady", "must be finite and >= 0 (0 = off)");
    // A relative change: >= 1 certifies nothing, and below kSteadyTolMin the
    // FP32 integrals' sampling noise decides (docs/rans.md).
    if (c.steadyTol > 0 && !(c.steadyTol >= kSteadyTolMin && c.steadyTol < 1))
        return err(2, "--steady",
                   std::format("is a relative change of the integrals: must "
                               "be in [{:g}, 1)",
                               kSteadyTolMin));
    c.steadySteps = r.steadySteps;
    if (r.given.contains("--steady-steps"))
    {
        if (c.steadyTol <= 0)
            return err(2, "--steady-steps", "caps the --steady iteration; "
                                            "requires --steady TOL");
        if (r.steadySteps < 1) return err(2, "--steady-steps", "must be >= 1");
    }
    c.steadyScheme = SteadyScheme(r.steadyScheme);
    if (r.given.contains("--steady-scheme") && c.steadyTol <= 0)
        return err(2, "--steady-scheme", "selects the --steady pseudo-time "
                                         "scheme; requires --steady TOL");
    if (c.steadyScheme == SteadyScheme::Implicit)
    {
        if (auto e = rejectGiven(r, {"--cfl", "--dt-recompute"},
                                 "with --steady-scheme implicit: explicit-"
                                 "march knobs; the implicit march ramps its "
                                 "own pseudo-time step");
            e)
            return e;
        // Measured stops 25-29 iterations: the explicit march's 1000000 would
        // let a stalled implicit run spin for hours before exit 8.
        if (!r.given.contains("--steady-steps"))
            c.steadySteps = kImplicitSteadySteps;
    }
    if (c.steadyTol > 0)
    {
        if (!rans)
            return err(2, "--steady", "a DNS channel has no steady state "
                                      "(it needs --rans)");
        if (!c.wallIso)
            return err(2, "--steady", "--wall adiabatic has no heat sink: no "
                                      "steady state");
        if (auto e = rejectGiven(r, {"--tt", "--profile-dt", "--fdt", "--pdt",
                                     "--wdt"},
                                 "with --steady: pseudo-time is not time; "
                                 "frames/profiles are time-keyed (profiles "
                                 "are written once, at convergence or at the "
                                 "cap)");
            e)
            return e;
        for (const Emit k : {Emit::Png, Emit::Web, Emit::Bin, Emit::Shm,
                             Emit::Volume, Emit::Slices, Emit::Vtk})
            if (c.emit.contains(k))
                return err(2, "--emit", "frames with --steady: pseudo-time is "
                                        "not time; frames/profiles are "
                                        "time-keyed");
    }
    // --compensated: the explicit steady march's two-word state on a time
    // march (on --steady it already is the state). The implicit march hands
    // the kernels a plain state and writes its iterate as a plain-state edit.
    c.compensated = r.compensated && c.steadyTol <= 0;
    if (r.compensated && c.steadyScheme == SteadyScheme::Implicit)
        return err(2, "--compensated", "with --steady-scheme implicit: the "
                                       "Newton march keeps its own FP64 "
                                       "iterate and a plain device state");
    if (!rans) return std::nullopt;
    if (std::size_t(c.grid.nx) * c.grid.nz > 4)
        c.warnings.emplace_back(
            "the RANS mean is homogeneous in x and z; --n 4 --nz 1 is the same "
            "answer");
    if (c.chanH1 * c.reTau > 1.0)
        c.warnings.push_back(std::format(
            "wall-resolved RANS with first-cell y+ {:.3g} > 1",
            c.chanH1 * c.reTau));
    return std::nullopt;
}

// The channel's LES knobs (docs/channel.md, "LES"): an SGS stress on the time
// march. Runs BEFORE channelRans so --les with --rans / --steady is named as
// the LES conflict. --les 0 / absent leaves the case untouched (cs 0, DNS).
// Pre: the grid is resolved.
std::optional<ConfigError> channelLes(const RawConfig& r, Case& c)
{
    c.cs = r.les;
    if (!(std::isfinite(c.cs) && c.cs >= 0))
        return err(2, "--les", "must be finite and >= 0 (0 = DNS)");
    c.sgsModel = r.sgs;
    if (r.given.contains("--sgs") && c.cs == 0)
        return err(2, "--sgs", "selects the SGS model for --les; give --les CS "
                               "(> 0) or drop --sgs");
    c.lesLabel = c.cs > 0;
    if (!c.lesLabel) return std::nullopt;
    if (r.rans != int(RansModel::None))
        return err(2, "--les", "a RANS field is an ensemble mean, an LES field "
                               "a filtered one; give --rans or --les, not both");
    if (r.steady > 0)
        return err(2, "--les", "LES is time-accurate; --steady is the RANS "
                               "pseudo-time solve");
    if (c.sgsModel == SgsModel::Sigma && c.grid.nz == 1)
        return err(2, "--sgs", "sigma: D_sigma is identically ZERO for any 2D "
                               "flow (that is the model's design); at nz == 1 "
                               "it is a silent no-op. Use nz > 1 or smag");
    return std::nullopt;
}

// Wall units: delta = 1, u_tau = 1, f = 1, nu = 1/reTau. The box momentum
// balance (f * V = 2 tau_w * A_wall with Ly = 2 delta) pins tau_w = 1, so
// reTau is the run's friction Reynolds number by construction.
std::expected<Case, ConfigError> validateChannel(const RawConfig& r, Case c)
{
    if (r.given.contains("--re"))
        return bad(2, "--re", "channel is configured in wall units by "
                              "--re-tau, not a body Reynolds number");
    if (r.given.contains("--ft"))
        return bad(2, "--ft", "channel run length is --tt (eddy turnovers "
                              "of delta/u_tau = 1)");
    if (auto e = checkWallGrid(r); e) return std::unexpected(*e);
    const int nyI = r.ny;
    const int nz = r.nz;
    if (!finitePos(r.reTau))
        return bad(2, "--re-tau", "must be finite and > 0");
    c.reTau = r.reTau;
    c.re = r.reTau;
    c.nu = 1.0 / r.reTau;
    c.mach = r.mach;
    // The implicit steady RANS channel is the one place a supersonic bulk Mach
    // is shock-free by construction: its iterate is ONE column broadcast
    // (homogeneous IC, --kick rejected), so v = 0 at the fixed point and the
    // flux never sees a shock (Coleman, Kim & Moser 1995 run M_b 1.5).
    const bool supersonicOk = r.rans != 0 && r.steady > 0 &&
                              SteadyScheme(r.steadyScheme) == SteadyScheme::Implicit;
    if (supersonicOk)
    {
        if (!finitePos(c.mach)) return bad(2, "--mach", "must be finite and > 0");
    }
    else if (auto e = checkMach(c.mach); e)
        return std::unexpected(*e);
    c.stretchBeta = r.stretch;
    if (!(std::isfinite(c.stretchBeta) && c.stretchBeta >= 0
          && c.stretchBeta <= 4))
        return bad(2, "--stretch", "must be in [0,4] (tanh clustering "
                                   "strength)");
    c.wallIso = r.wallIso == 1;

    const double delta = 1.0;
    const int nyT = nyI + 2; // + the two wall-mirror ghost rows
    const double h1 = ChannelStretchY::firstCell(nyT, c.stretchBeta, delta);
    const double ly = 2 * delta + 2 * h1; // period fixed by the ghost rows
    if (r.given.contains("--lx") && !finitePos(r.lx))
        return bad(2, "--lx", "must be finite and > 0 (0/omitted = 4 pi)");
    if (r.given.contains("--lz") && !finitePos(r.lzBox))
        return bad(2, "--lz", "must be finite and > 0 (0/omitted = 4 pi/3)");
    c.lxBox = r.lx > 0 ? r.lx : 4 * pi;
    c.lzBox = nz == 1 ? 1.0 : (r.lzBox > 0 ? r.lzBox : 4 * pi / 3);
    auto box = makeBox(r.n, nyT, nz, c.lxBox, ly, c.lzBox);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    c.nzInterior = nz;
    c.h = c.grid.hy;
    c.L = c.lxBox;
    c.chanH1 = h1;
    c.stretchH1 = h1;
    const ChannelStretchY R{ly, nyT, c.stretchBeta, delta};
    c.stretchGrowth = std::max(
        c.stretchGrowth,
        maxAdjacentGrowth([&](double y) { return R.mapY(y); }, c.grid.hy, 2,
                          nyT - 1, h1));

    // Bulk-velocity estimate (log law) fixes the sound speed for the
    // requested BULK Mach: c_s = ubEst/Mach, p = rho c_s^2/gamma,
    // Twall = p (rho = R = 1). The IC and the isothermal wall share it.
    c.ubEst = channelUbEstimate(c.reTau);
    c.pRef = channelPRef(c.reTau, c.mach);

    // --cfl has no channel-specific default: the parser's 0.4 stands.
    if (auto e = validateWallRun(r, c, r.cfl); e) return std::unexpected(*e);
    c.tEnd = r.tt;
    if (auto e = channelLes(r, c); e) return std::unexpected(*e);
    if (auto e = channelRans(r, c); e) return std::unexpected(*e);
    // PNG is allowed at nz > 1 here: it renders the centre z-plane,
    // which for a channel is exactly the picture you want.
    c.physics = Physics::Compressible;
    c.far = FarField::Periodic;
    c.bodyKind = BodyKind::None;
    c.meta = {.D = 2 * delta,
              .re = c.reTau,
              .mach = c.mach,
              .frameDt = c.fdt,
              .cs = c.cs,
              .les = c.lesLabel,
              .body = "channel",
              .farField = "periodic",
              .physics = "compressible",
              .kick = c.kick,
              .sgs = c.lesLabel ? name(c.sgsModel) : "",
              .rans = c.rans == RansModel::None ? std::string()
                                                : std::string(ransName(c.rans)),
              .steady = c.steadyScheme == SteadyScheme::Implicit
                            ? std::string(name(c.steadyScheme))
                            : std::string()};
    if (c.compensated) c.meta.state = "compensated";
    c.meta.setGrid(c.grid);
    return c;
}

// --- conv -------------------------------------------------------------------
// Rayleigh-Benard: heated floor, cooled lid, gravity in -y, periodic x/z.
//
// Box units fix everything: layer height H = 1, mid-layer temperature
// Tref = 1, Rgas = 1 and rho_ref = 1, hence p_ref = 1 and the reference
// sound speed is sqrt(gamma). The three inputs then close the case:
//
//   --dtemp eps   the imposed drop, Tbot/Ttop = 1 +/- eps/2
//   --mach M      M = Uf0/c with Uf0 = sqrt(g eps H) -- this is what SETS
//                 gravity, g = M^2 gamma / eps, and with it how far the
//                 run is from the incompressible limit
//   --rayleigh Ra on the SUPERADIABATIC drop, which is what fixes nu
//
// The stratification is not a detail: gravity itself carries an adiabatic
// temperature gradient g/cp, so only dTsuper = dT - (g/cp) H drives
// convection (Schwarzschild). A subadiabatic layer is stably stratified
// and simply will not convect, which is a configuration error and not a
// result -- so it is rejected here, by name, with the two knobs that fix
// it. The adiabatic fraction is M^2 (gamma-1) / eps^2, so it shrinks
// quadratically as the Mach number drops toward the Boussinesq limit.
std::expected<Case, ConfigError> validateConv(const RawConfig& r, Case c)
{
    if (r.given.contains("--re"))
        return bad(2, "--re", "convection is configured by --rayleigh and "
                              "--prandtl, not a body Reynolds number "
                              "(there is no body and no free stream)");
    if (r.given.contains("--ft"))
        return bad(2, "--ft", "convection run length is --tt (free-fall "
                              "times H/Uf)");
    if (auto e = checkWallGrid(r); e) return std::unexpected(*e);
    const int nyI = r.ny;
    const int nz = r.nz;

    if (!finitePos(r.rayleigh))
        return bad(2, "--rayleigh", "must be finite and > 0");
    c.rayleigh = r.rayleigh;
    if (!(finitePos(r.dtemp) && r.dtemp < 2))
        return bad(2, "--dtemp", "must be in (0,2): the imposed drop as a "
                                 "fraction of the mid-layer temperature, "
                                 "so the lid sits at 1 - dtemp/2 > 0");
    c.dtemp = r.dtemp;
    c.mach = r.mach;
    if (auto e = checkMach(c.mach); e) return std::unexpected(*e);
    c.prandtl = r.prandtl;
    if (!finitePos(c.prandtl))
        return bad(2, "--prandtl", "must be finite and > 0");
    c.sutherland = r.visc == 1;
    c.tinfK = r.tinf;
    if (c.sutherland && !finitePos(c.tinfK))
        return bad(2, "--tinf", "must be finite and > 0 (it is the "
                                "dimensional anchor Sutherland's 110.4 K "
                                "needs)");
    if (c.physics == Physics::Incompressible)
    {
        // Boussinesq: the M -> 0 limit taken analytically. There is no
        // sound speed, so the knobs that only exist to place the run
        // relative to one are not silently ignored -- they are rejected.
        if (r.given.contains("--mach"))
            return bad(2, "--mach", "the Boussinesq limit has no sound "
                                    "speed: --mach only means something "
                                    "for --physics compressible, where it "
                                    "sets gravity");
        if (r.given.contains("--dtemp"))
            return bad(2, "--dtemp", "the Boussinesq equations depend on "
                                     "the imposed drop only through Ra, so "
                                     "the drop is scaled to 1 and --dtemp "
                                     "would be a second, conflicting name "
                                     "for --rayleigh");
        if (c.sutherland)
            return bad(2, "--visc", "Sutherland's mu(T) is inconsistent "
                                    "with a divergence-free velocity "
                                    "field: constant transport properties "
                                    "are part of the Boussinesq "
                                    "approximation. Use --physics "
                                    "compressible for mu(T)");
        if (r.given.contains("--tinf"))
            return bad(2, "--tinf", "nothing in the Boussinesq branch is "
                                    "dimensional (it exists only to anchor "
                                    "Sutherland's 110.4 K)");
    }
    if (!finitePos(r.convAspect))
        return bad(2, "--aspect", "must be finite and > 0 (box width and "
                                  "depth in layer heights)");
    c.convAspect = r.convAspect;
    if (!(finitePos(r.hotPatch) && r.hotPatch <= 1))
        return bad(2, "--hot-patch", "must be in (0,1]: the heated side "
                                     "fraction of the floor (1 = the whole "
                                     "plate)");
    c.hotPatch = r.hotPatch;
    // --sides closed turns the lateral faces into no-slip adiabatic walls,
    // carried by the same one-cell ghost ring as the plates. At nz == 1
    // only x closes: 2D is exact periodic wrap in z, not a code path, so a
    // z ghost ring there would be a third dimension the case does not have.
    c.closedSides = r.sides == 1;

    const double gamma = kGammaAir, Rgas = kRgasBox, H = 1.0, Tref = 1.0;
    c.convH = H;
    if (c.physics == Physics::Incompressible)
    {
        // Free-fall scaling: H = dT = g*alpha = 1, hence Uff = t_ff = 1 and
        // Ra alone fixes both diffusivities. No stratification exists here,
        // so the superadiabatic drop IS the imposed one and Ra and Nu are
        // measured against the same thing.
        c.tBot = 0.5;
        c.tTop = -0.5;
        c.dtemp = 1.0;
        c.dtempSuper = 1.0;
        c.gravity = 1.0;   // g*alpha, the buoyancy coefficient
        c.uFreeFall = 1.0;
        c.mach = 0;        // there is no sound speed to be a fraction of
        c.nu = std::sqrt(c.prandtl / c.rayleigh);
        c.kappaDiff = 1.0 / std::sqrt(c.rayleigh * c.prandtl);
        c.re = c.uFreeFall * H / c.nu;   // = sqrt(Ra/Pr)
    }
    else {
    c.tBot = Tref * (1 + 0.5 * c.dtemp);
    c.tTop = Tref * (1 - 0.5 * c.dtemp);
    // g from the Mach number: M = sqrt(g eps H) / sqrt(gamma R Tref).
    c.gravity = c.mach * c.mach * gamma * Rgas * Tref / (c.dtemp * H);
    const double cp = gamma * Rgas / (gamma - 1.0);
    const double dTad = c.gravity / cp * H;
    c.dtempSuper = c.dtemp * Tref - dTad;
    if (!(c.dtempSuper > 0))
        return bad(2, "--mach",
                   std::format("the layer is subadiabatic and cannot "
                               "convect: gravity's own adiabatic drop "
                               "(g/cp) H = {:.4g} already exceeds the "
                               "imposed {:.4g}. Lower --mach (the "
                               "adiabatic fraction is M^2 (gamma-1) / "
                               "dtemp^2 = {:.3g}) or raise --dtemp",
                               dTad, c.dtemp * Tref,
                               c.mach * c.mach * (gamma - 1)
                                   / (c.dtemp * c.dtemp)));
    // Free-fall velocity on the superadiabatic excess -- the velocity the
    // run actually reaches, and the unit --tt counts in.
    c.uFreeFall = std::sqrt(c.gravity * c.dtempSuper / Tref * H);
    // Ra = g (dTsuper/Tref) H^3 / (nu kappa) with kappa = nu/Pr.
    c.nu = std::sqrt(c.gravity * (c.dtempSuper / Tref) * H * H * H
                     * c.prandtl / c.rayleigh);
    c.kappaDiff = c.nu / c.prandtl;
    c.re = c.uFreeFall * H / c.nu;   // free-fall Reynolds number
    }

    const int nyT = nyI + 2;         // + the two wall-mirror ghost rows
    // The ghost rows are geometric mirrors of the first fluid rows, so the
    // wall faces sit half a cell outside them: the mapped extent is
    // H + 2*(H/nyI) and the wall-to-wall height stays exactly H.
    const double hy = H / nyI;
    const double ly = H + 2 * hy;
    c.lxBox = c.convAspect * H;
    c.lzBox = nz == 1 ? 1.0 : c.convAspect * H;
    // A closed lateral axis gets the plates' treatment exactly: the fluid
    // extent stays lxBox/lzBox and the two ghost columns extend the mapped
    // box by one spacing each side, so the wall faces sit half a cell
    // outside them and the wall-to-wall width is exactly lxBox.
    const int nxT = c.closedSides ? r.n + 2 : r.n;
    const int nzT = c.closedSides && nz > 1 ? nz + 2 : nz;
    const double hx = c.lxBox / r.n, hz = c.lzBox / nz;
    const double lx = c.closedSides ? c.lxBox + 2 * hx : c.lxBox;
    const double lz = c.closedSides && nz > 1 ? c.lzBox + 2 * hz : c.lzBox;
    auto box = makeBox(nxT, nyT, nzT, lx, ly, lz);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    c.nxInterior = r.n;
    c.nzInterior = nz;
    c.h = c.grid.hy;
    c.L = c.lxBox;
    c.pRef = Rgas * Tref;   // rho_ref = 1 at mid-layer

    // The compressible branch is acoustically limited, the Boussinesq one
    // advectively; 0.4 and 0.8 are those two solvers' own defaults.
    if (auto e = validateWallRun(
            r, c, c.physics == Physics::Incompressible ? 0.8 : 0.4);
        e)
        return std::unexpected(*e);
    c.tEnd = r.tt * H / c.uFreeFall;   // free-fall times -> box time
    // Below onset the layer just conducts; that is a legitimate run (the
    // subcritical control), but it is not what the user usually wants, so
    // it is an advisory and not an error.
    if (c.rayleigh < 1708.0 && c.hotPatch >= 1.0)
        c.warnings.push_back(std::format(
            "Ra = {:.4g} is below the onset of convection (~1708 for "
            "rigid isothermal plates): the layer will stay quiescent and "
            "report Nu = 1", c.rayleigh));
    if (c.hotPatch < 1.0)
        c.warnings.push_back(std::format(
            "--hot-patch {:.4g}: only that side fraction of the floor is "
            "heated, so this is a localised plume and not classical "
            "Rayleigh-Benard -- the 1708 onset does not apply and nuBot is "
            "a net floor-averaged flux, not a Nusselt number",
            c.hotPatch));
    if (nz == 1)
        c.warnings.push_back("nz == 1: 2D convection has no vortex "
                             "stretching -- roll structure and Nu are "
                             "qualitative");
    // A closed cavity is a different experiment from the unbounded layer,
    // and the two numbers people quote against it -- the 1708 onset and the
    // Ra-Nu scaling -- are both the unbounded ones. Say so rather than let
    // a cavity result be read as classical Rayleigh-Benard.
    if (c.closedSides)
        c.warnings.push_back(std::format(
            "--sides closed: the lateral faces are no-slip adiabatic "
            "walls, so the roll count is fixed by --aspect {:.4g} and the "
            "corner regions are viscously braked -- Nu is a cavity number "
            "below the periodic one at the same Ra, and the ~1708 onset "
            "(and the Ra-Nu scaling) are the unbounded layer's",
            c.convAspect));
    c.far = FarField::Periodic;
    c.bodyKind = BodyKind::None;
    c.meta = {.D = H,
              .re = c.re,
              .mach = c.mach,
              .frameDt = c.fdt,
              .body = "convection",
              .farField = c.closedSides ? "closed-sides" : "periodic",
              .physics = name(c.physics),
              .kick = c.kick,
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

// --- flame --mode chamber ----------------------------------------------------
// The closed 2D combustion chamber (docs/chem.md, "Chamber"; dns/chamber.hpp).

// Which flags belong to which flame mode: a flag of the other mode that was
// TYPED is an error by name, never ignored.
std::optional<ConfigError> chamberFlagScope(const RawConfig& r, bool chamber)
{
    if (chamber)
        return rejectGiven(r, {"--ny", "--upstream", "--inflow-speed",
                               "--profile-dt", "--wrinkle", "--wrinkle-modes"},
                           "does not apply to --mode chamber (the grid is "
                           "derived from the chamber; fresh gas enters "
                           "through the injector, --inject-speed)");
    return rejectGiven(r, {"--inject-speed", "--coflow", "--chamber-height",
                           "--chamber-length", "--nozzle-ratio",
                           "--holder-height", "--clearance", "--inject-ramp",
                           "--reaction"},
                       "only applies to --mode chamber");
}

std::expected<Case, ConfigError> finishChamber(const RawConfig& r, Case c)
{
    const auto& f = r.chamber;
    const double cells =
        r.given.contains("--cells-per-delta") ? r.cellsPerDelta : 4.0;
    const double down = r.given.contains("--downstream") ? r.downstream : 40.0;
    if (!(down >= 10))
        return bad(2, "--downstream", "must be >= 10 thermal thicknesses "
                                      "(the jet region past the nozzle)");
    if (!(finitePos(f.injectSpeed) && f.injectSpeed <= 20))
        return bad(2, "--inject-speed", "must be in (0, 20] (x S_L)");
    if (!(finitePos(f.coflow) && f.coflow <= 20))
        return bad(2, "--coflow", "must be in (0, 20] (x S_L): a static "
                                  "ambient makes the NSCBC inflow an acoustic "
                                  "mirror");
    if (!(std::isfinite(f.height) && f.height >= 8))
        return bad(2, "--chamber-height", "must be >= 8 thermal thicknesses");
    if (!(std::isfinite(f.length) && f.length >= 10))
        return bad(2, "--chamber-length", "must be >= 10 thermal thicknesses");
    if (!(f.nozzleRatio >= 1.5 && f.nozzleRatio <= 6))
        return bad(2, "--nozzle-ratio", "must be in [1.5, 6] (Hc / He)");
    if (!(f.holder >= 1 && f.holder <= f.height - 4))
        return bad(2, "--holder-height", "must be in [1, chamber height - 4] "
                                         "thermal thicknesses");
    if (!(std::isfinite(f.clearance) && f.clearance >= 4))
        return bad(2, "--clearance", "must be >= 4 thermal thicknesses");
    if (!(std::isfinite(f.ramp) && f.ramp >= 0))
        return bad(2, "--inject-ramp", "must be finite and >= 0");

    const double dth = c.flameEig.dthOverDD;
    const double h = dth / cells;
    const int nb = c.nscbcSink;
    ChamberSpec s;
    s.tw = 1.5 * dth;
    s.Hc = f.height * dth;
    s.Lc = f.length * dth;
    s.He = s.Hc / f.nozzleRatio;
    s.Ln = 0.5 * s.Hc;
    s.Hg = f.holder * dth;
    s.Lg = 0.75 * s.Hg;
    s.wInj = 1.25 * dth;
    s.x0 = nb * h + 4 * dth;
    s.xg = s.xb() + s.wInj + 2.5 * dth;
    if (s.xBase() > s.xn() - 2 * dth)
        return bad(2, "--chamber-length", "too short for the injector, the "
                                          "holder and 2 thermal thicknesses "
                                          "behind it");
    if (s.He / h < 8)
        return bad(2, "--nozzle-ratio", std::format(
                       "leaves the nozzle exit {:.3g} cells tall (< 8): raise "
                       "--cells-per-delta or --chamber-height",
                       s.He / h));
    const int nx = int(std::ceil((s.xe() + down * dth) / h - 1e-9)) + nb;
    const int ny =
        int(std::ceil((s.Hc + 2 * s.tw + 2 * f.clearance * dth) / h - 1e-9));
    auto box = makeBox(nx, ny, 1, nx * h, ny * h, 1.0);
    if (!box) return std::unexpected(box.error());
    s.yc = 0.5 * ny * h;
    c.grid = *box;
    c.n = nx;
    c.h = h;
    c.L = nx * h;
    c.flameCells = cells;
    c.flameDown = down;
    c.flameTu = 1.0 / (kGammaAir * c.mach * c.mach);
    c.chamberMode = true;
    c.chamber = s;
    c.chamberMdot = f.injectSpeed * s.Hc; // rho_u = 1
    c.chamberCoflow = f.coflow;
    c.chamberRamp = f.ramp;
    c.chamberReaction = f.reaction == 1;
    c.flameUIn = f.injectSpeed;
    c.uInf = f.coflow;
    c.re = 1.0 / c.prandtl;
    c.nu = c.prandtl;
    c.far = FarField::Nscbc;
    c.bodyKind = BodyKind::None;

    // The nozzle at the burnt total temperature, exit at the ambient p0: a
    // choked estimate has no subsonic solution and the central flux has no
    // shock capturing.
    const double Tu = c.flameTu, sig = c.flameModel.sigma;
    auto nz = isentropicNozzle(c.chamberMdot, sig * Tu, s.He, Tu, kGammaAir,
                               kRgasBox);
    if (!nz)
        return bad(2, "--inject-speed", std::format(
                       "chokes the nozzle ({}): lower --inject-speed or "
                       "--nozzle-ratio, or raise --mach",
                       nz.error()));
    if (nz->Me > 0.7)
        c.warnings.push_back(std::format(
            "the isentropic exit Mach is {:.3g} (> 0.7): the central flux has "
            "no shock capturing", nz->Me));

    if (auto e = resolveCfl(r, c, 0.4); e) return std::unexpected(*e);
    if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
    c.ft = r.given.contains("--tt") ? r.tt : 60.0;
    c.tEnd = c.ft;
    if (auto e = resolveDtRecompute(r, c, 16); e) return std::unexpected(*e);
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    if (auto e = resolveEmit(c, {Emit::Stats}); e) return std::unexpected(*e);
    if (c.mach > 0.02)
        c.warnings.push_back(std::format(
            "--mach {:.4g} is above the measured ladder top 0.02", c.mach));
    if (nb == 0)
        c.warnings.push_back("--nscbc-sink 0: the bare characteristic "
                             "boundary (nothing drains mesh-scale content)");
    if (!c.chamberReaction)
        c.warnings.push_back("--reaction off: inert species transport (the "
                             "wall-impermeability check), no flame");
    c.meta = {.D = s.Hc,
              .re = c.re,
              .mach = c.mach,
              .frameDt = c.fdt,
              .body = "combustion chamber",
              .farField = "nscbc",
              .physics = name(c.physics),
              .kick = 0.0,
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

// --- flame ------------------------------------------------------------------
// --mode bunsen's default resolution: 8 cells per delta_th (the demo grid;
// the 1D ladder puts S_L +1.7% high at 7.5 cells, docs/chem.md).
constexpr double kBunsenCellsDefault = 8;

// The slot burner (dns/bunsen.hpp): validates the burner flags and sets the
// box, c.bunsen (box units) and c.uInf. Lengths arrive in delta_th.
// Pre: c.flameEig, c.flameCells, c.flameModel set; nb = the sink band.
std::expected<void, ConfigError> bunsenGeometry(const RawConfig& r, Case& c,
                                                int nb)
{
    const double dth = c.flameEig.dthOverDD, cells = c.flameCells;
    const double h = dth / cells, sig = c.flameModel.sigma;
    if (!(std::isfinite(r.slotWidth) && r.slotWidth >= 4))
        return bad(2, "--slot-width", "must be >= 4 thermal thicknesses (a "
                                      "cone needs a straight flank)");
    if (!(std::isfinite(r.jetSpeed) && r.jetSpeed >= 1.5 && r.jetSpeed <= 20))
        return bad(2, "--jet-speed", "must be in [1.5, 20] S_L (a cone "
                                     "needs U > S_L; above 20 it outgrows "
                                     "any affordable box)");
    if (r.jetSpeed * c.mach > 0.3)
        return bad(2, "--jet-speed", std::format(
                                         "jet Mach {:.3g} = U M exceeds 0.3: "
                                         "lower --mach or the jet speed",
                                         r.jetSpeed * c.mach));
    if (!(std::isfinite(r.coflowSpeed) && r.coflowSpeed >= 0 &&
          r.coflowSpeed <= r.jetSpeed))
        return bad(2, "--coflow-speed", "must be in [0, --jet-speed] S_L");
    const double wco = r.coflowWidth < 0 ? r.slotWidth : r.coflowWidth;
    if (!(std::isfinite(wco) && wco >= 2))
        return bad(2, "--coflow-width", "must be >= 2 thermal thicknesses per "
                                        "side (y is periodic: the coflow "
                                        "separates the jet from its image)");
    const double tco = r.coflowTemp < 0 && !r.given.contains("--coflow-temp")
                           ? sig
                           : r.coflowTemp;
    if (!(std::isfinite(tco) && tco >= 1 && tco <= sig))
        return bad(2, "--coflow-temp", std::format(
                                           "must be in [1, --expansion = {:g}] "
                                           "(T/Tu: 1 cold inert, sigma the "
                                           "burnt pilot)",
                                           sig));
    if (!(std::isfinite(r.pilotWidth) && r.pilotWidth >= 0))
        return bad(2, "--pilot-width", "must be finite and >= 0 thermal "
                                       "thicknesses");
    if (!(r.pilotWidth > 0))
        if (auto e = rejectGiven(r, {"--pilot-temp", "--pilot-speed"},
                                 "needs --pilot-width > 0");
            e)
            return std::unexpected(*e);
    const double tp = r.pilotTemp < 0 && !r.given.contains("--pilot-temp")
                          ? sig
                          : r.pilotTemp;
    if (!(std::isfinite(tp) && tp >= 1 && tp <= sig))
        return bad(2, "--pilot-temp", std::format(
                                          "must be in [1, --expansion = {:g}] "
                                          "(T/Tu)",
                                          sig));
    const double up = r.pilotSpeed < 0 && !r.given.contains("--pilot-speed")
                          ? r.coflowSpeed
                          : r.pilotSpeed;
    if (!(std::isfinite(up) && up >= 0 && up <= r.jetSpeed))
        return bad(2, "--pilot-speed", "must be in [0, --jet-speed] S_L");
    if (!(std::isfinite(r.slotEdge) && r.slotEdge * cells >= 1))
        return bad(2, "--slot-edge", std::format(
                                         "must be >= one cell ({:.4g} "
                                         "thermal thicknesses at "
                                         "--cells-per-delta {:g})",
                                         1.0 / cells, cells));
    const double coneH = 0.5 * r.slotWidth *
                         std::sqrt(r.jetSpeed * r.jetSpeed - 1.0); // delta_th
    const double len = r.flameLength < 0 ? std::ceil(1.25 * coneH + 8)
                                         : r.flameLength;
    if (!(std::isfinite(len) && len >= coneH + 4))
        return bad(2, "--length", std::format(
                                      "must be >= the ideal cone height + 4 = "
                                      "{:.4g} thermal thicknesses (the tip "
                                      "and its burnt gas must fit)",
                                      coneH + 4));
    if (!(std::isfinite(r.pulse) && r.pulse >= 0 && r.pulse <= 0.5))
        return bad(2, "--pulse", "must be in [0, 0.5] (fraction of the jet "
                                 "speed)");
    if (r.pulse > 0 && !(std::isfinite(r.pulseFreq) && r.pulseFreq > 0))
        return bad(2, "--pulse-freq", "must be finite and > 0 with --pulse "
                                      "(1/tau)");
    if (r.given.contains("--pulse-freq") && !(r.pulse > 0))
        return bad(2, "--pulse-freq", "needs --pulse > 0");

    const int half = int(std::ceil((0.5 * r.slotWidth + r.pilotWidth + wco) *
                                   cells - 1e-9));
    const int ny = 2 * half;
    const int nx = int(std::ceil(len * cells - 1e-9)) + 2 * nb;
    auto box = makeBox(nx, ny, 1, nx * h, ny * h, 1.0);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    BunsenGeom& b = c.bunsen;
    b.W = r.slotWidth * dth;
    b.uJet = r.jetSpeed;
    b.uCo = r.coflowSpeed;
    b.wCo = wco * dth;
    b.tCo = tco;
    b.pilot = r.pilotWidth * dth;
    b.tPilot = tp;
    b.uPilot = up;
    b.edge = r.slotEdge * dth;
    b.yc = half * h;
    b.coneH = coneH * dth;
    b.pulse = r.pulse;
    b.pulseFreq = r.pulseFreq;
    c.flameBunsen = true;
    c.uInf = r.jetSpeed;
    return {};
}

// --mode diffusion defaults that differ from the shared flame flags' (those
// are premixed numbers in delta_th and S_L): the jet Mach, the slot width
// (the jet Peclet number), the edge half-width, all in l = D0/U_jet.
constexpr double kDiffMachDefault = 0.05;
constexpr double kDiffSlotDefault = 16;
// Edge 3: measured (2026-10-28, W 16, s 1, beta 5, Da 1000, M 0.05, 4
// cells/l, tt 40) edge 1 exits 8 at t = 0.48 (the thin IC sheet goes out);
// edge 2 and edge 1 + 2 l burnt Z_st pilots anchor; edge 3 is steady (tip
// 51.2 l, TmaxIn/T_ad 0.986). docs/chem.md, anchoring table.
constexpr double kDiffEdgeDefault = 3.0;
// beta 5: measured (same case, edge 3, outflow sigma 4, the inflow pin) beta 8
// at Da 1000 extinguishes the IC sheet by t = 1.8 (exit 8); beta 5 anchors
// (tip 51.2 l, Tmax/T_ad 0.986, steady by t ~ 60).
constexpr double kDiffBetaDefault = 5.0;

// --mode candle defaults (docs/chem.md, candle section), in l = D0/U_fuel and
// U_fuel. Paraffin/air: s ~3.4 (mass, with O2), Y_O2 0.233 -> Z_st 0.064.
// Fr = U^2/(g W) = 1/(g W) = 0.071 at the defaults (buoyancy-controlled).
// Measured (2026-10-28, scouts): in the 2D buoyancy regime the flame's
// aspect follows the fuel Peclet number W (H/w ~ Q/D): W 2 at g 7 burned a
// needle (tip 24 l, half-width 2 l), W 1 at g 14 a teardrop (tip ~9.5 l,
// half-width 1.4 l) -- hence the low-Peclet wick on a finer grid.
constexpr double kCandleGravityDefault = 14;
constexpr double kCandleSlotDefault = 1;
constexpr double kCandleStoichDefault = 3.4;
constexpr double kCandleOxYDefault = 0.233;
constexpr double kCandleEdgeDefault = 0.5;
constexpr double kCandleCellsDefault = 8;
constexpr double kCandleCoflowWidthDefault = 12;
constexpr double kCandleLengthDefault = 32;
// Da 5000: at 1000 the W 1 candle's ignition kernel went out (exit 8 at t
// 0.56, Tmax past the band 2.99 T0).
constexpr double kCandleDaDefault = 5000;
// The IC lights a short column (a closed Burke-Schumann sheet 4 l tall), not
// the whole box: the extruded column of the jet IC is a box-tall thermal that
// rolled up into vortices descending from the top (scout, W 4: outflow
// backflow -44 U, diverged at t 10.7).
constexpr double kCandleIgnitionDefault = 4;
// M = U_fuel/c0: the buoyant flow runs ~35 U_fuel (measured |u|max), so the
// fuel Mach is inflated far less than the jet's: buoyant Mach ~0.18.
constexpr double kCandleMachDefault = 0.005;

// The 2D slot-jet diffusion flame (docs/chem.md, dns/diffusion.hpp): model
// C1, the burner on the bunsen flags (lengths in l, speeds in U_jet), the
// box, the label inputs. Every premixed-only flag is rejected by name.
// --mode candle is the same model and burner under gravity (-x) with the
// candle defaults above.
std::expected<Case, ConfigError> validateDiffusion(const RawConfig& r, Case c)
{
    const bool candle = r.flameMode == 3;
    if (auto e = chamberFlagScope(r, false); e) return std::unexpected(*e);
    if (auto e = candle ? rejectGiven(r, {"--pilot-width", "--pilot-speed"},
                                      "does not apply to --mode candle (the "
                                      "flame anchors on its inflow sheet: no "
                                      "pilot)")
                        : rejectGiven(r, {"--gravity", "--quiescent"},
                                      "only applies to --mode candle");
        e)
        return std::unexpected(*e);
    if (candle && r.quiescent)
        if (auto e = rejectGiven(r,
                                 {"--slot-width", "--slot-edge",
                                  "--coflow-speed", "--pulse", "--pulse-freq"},
                                 "does not apply with --quiescent (no fuel, "
                                 "every inflow row at rest)");
            e)
            return std::unexpected(*e);
    // A candle default where the shared flag (a premixed / jet number) was
    // not given.
    auto pick = [&](const char* flag, double given, double dflt) {
        return candle && !r.given.contains(flag) ? dflt : given;
    };
    if (auto e = rejectGiven(r, {"--re"}, "does not apply: the jet is set by "
                                          "--slot-width (its Peclet number "
                                          "U W / D0) and --prandtl");
        e)
        return std::unexpected(*e);
    if (auto e = rejectGiven(r, {"--ft"}, "does not apply: the run length is "
                                          "--tt (in l/U_jet)");
        e)
        return std::unexpected(*e);
    if (auto e = rejectGiven(r, {"--n"}, "does not apply: the grid is "
                                         "--cells-per-unit over the burner "
                                         "and --length");
        e)
        return std::unexpected(*e);
    if (c.physics != Physics::Compressible)
        return bad(2, "--physics", "a flame is compressible only (no low-Mach "
                                   "branch until P7; docs/chem.md)");
    if (auto e = rejectGiven(r,
                             {"--upstream", "--downstream", "--inflow-speed",
                              "--wrinkle", "--wrinkle-modes", "--ny",
                              "--profile-dt", "--cells-per-delta",
                              "--jet-speed", "--coflow-temp", "--pilot-temp"},
                             "does not apply to --mode diffusion (the jet "
                             "speed is the unit, the inflow state is the "
                             "Burke-Schumann sheet of its mixture fraction, "
                             "the grid is --cells-per-unit)");
        e)
        return std::unexpected(*e);
    if (r.nz != 1)
        return bad(2, "--nz", "must be 1: multi-D flames are P2 (docs/chem.md)");
    c.mach = r.given.contains("--mach")
                 ? r.mach
                 : (candle ? kCandleMachDefault : kDiffMachDefault);
    if (!(finitePos(c.mach) && c.mach <= 0.2))
        return bad(2, "--mach", "must be in (0, 0.2] for --mode diffusion "
                                "and candle: the jet Mach number U_jet/c0");
    DiffusionModel md;
    md.s = pick("--stoich", r.stoich, kCandleStoichDefault);
    if (!(std::isfinite(md.s) && md.s > 0 && md.s <= 100))
        return bad(2, "--stoich", "must be in (0, 100] (oxidizer mass per "
                                  "unit fuel mass)");
    md.yF1 = r.fuelY;
    if (!(md.yF1 > 0 && md.yF1 <= 1))
        return bad(2, "--fuel-y", "must be in (0, 1] (the fuel stream's Y_F; "
                                  "balance inert)");
    md.yO2 = pick("--ox-y", r.oxY, kCandleOxYDefault);
    if (!(md.yO2 > 0 && md.yO2 <= 1))
        return bad(2, "--ox-y", "must be in (0, 1] (the oxidizer stream's "
                                "Y_O; balance inert)");
    md.sigma = r.expansion;
    if (!(md.sigma > 1 && md.sigma <= 10))
        return bad(2, "--expansion", "must be in (1, 10] (sigma = T_ad(Z_st) "
                                     "/ T0, the flame-sheet temperature)");
    md.beta = r.given.contains("--zeldovich") ? r.zeldovich : kDiffBetaDefault;
    if (!(md.beta >= 2 && md.beta <= 20))
        return bad(2, "--zeldovich", "must be in [2, 20]");
    md.Da = pick("--damkohler", r.damkohler, kCandleDaDefault);
    if (!(finitePos(md.Da) && md.Da <= 1e6))
        return bad(2, "--damkohler", "must be in (0, 1e6] (A rho0 "
                                     "e^(-Ta/T_ad) l/U_jet)");
    c.prandtl = r.prandtl;
    if (!finitePos(c.prandtl)) return bad(2, "--prandtl", "must be finite and > 0");
    c.sutherland = r.visc == 1;
    c.tinfK = r.tinf;
    if (c.sutherland && !(r.given.contains("--tinf") && finitePos(c.tinfK)))
        return bad(2, "--tinf", "is required with --visc sutherland: T0 in "
                                "kelvin (Sutherland's 110.4 K needs a "
                                "dimensional anchor)");
    if (!c.sutherland && r.given.contains("--tinf"))
        return bad(2, "--tinf", "only anchors --visc sutherland");
    const double cells =
        pick("--cells-per-unit", r.cellsPerUnit, kCandleCellsDefault);
    if (!(std::isfinite(cells) && cells >= 2 && cells <= 64))
        return bad(2, "--cells-per-unit", "must be in [2, 64] (cells per l = "
                                          "D0/U_jet)");
    const double h = 1.0 / cells;
    const double W = r.given.contains("--slot-width")
                         ? r.slotWidth
                         : (candle ? kCandleSlotDefault : kDiffSlotDefault);
    // The candle's wick is a low-Peclet slot (the default Pe is 1).
    const double wMin = candle ? 0.5 : 4;
    if (!(std::isfinite(W) && W >= wMin && W <= 200))
        return bad(2, "--slot-width", std::format("must be in [{:g}, 200] l "
                                                  "(the jet Peclet number U W "
                                                  "/ D0)",
                                                  wMin));
    if (!(std::isfinite(r.coflowSpeed) && r.coflowSpeed >= 0 &&
          r.coflowSpeed <= 2))
        return bad(2, "--coflow-speed", "must be in [0, 2] U_jet");
    const double wco = r.coflowWidth < 0
                           ? (candle ? kCandleCoflowWidthDefault : W)
                           : r.coflowWidth;
    if (!(std::isfinite(wco) && wco >= 2))
        return bad(2, "--coflow-width", "must be >= 2 l per side (y is "
                                        "periodic: the coflow separates the "
                                        "jet from its image)");
    if (!(std::isfinite(r.pilotWidth) && r.pilotWidth >= 0))
        return bad(2, "--pilot-width", "must be finite and >= 0 l");
    if (!(r.pilotWidth > 0))
        if (auto e = rejectGiven(r, {"--pilot-speed"}, "needs --pilot-width > 0");
            e)
            return std::unexpected(*e);
    const double up = r.pilotSpeed < 0 && !r.given.contains("--pilot-speed")
                          ? r.coflowSpeed
                          : r.pilotSpeed;
    if (!(std::isfinite(up) && up >= 0 && up <= 2))
        return bad(2, "--pilot-speed", "must be in [0, 2] U_jet");
    const double edge = r.given.contains("--slot-edge")
                            ? r.slotEdge
                            : (candle ? kCandleEdgeDefault : kDiffEdgeDefault);
    if (!(std::isfinite(edge) && edge * cells >= 1))
        return bad(2, "--slot-edge", std::format("must be >= one cell ({:.4g} "
                                                 "l at --cells-per-unit {:g})",
                                                 1.0 / cells, cells));
    const double tipCold = sheetTipConstantDensity(W, md.zst());
    const double len =
        r.flameLength >= 0
            ? r.flameLength
            : (candle ? kCandleLengthDefault
                      : std::ceil(std::min(1.25 * tipCold + 8, 20.0 * W)));
    if (!(std::isfinite(len) && len >= W))
        return bad(2, "--length", "must be >= the slot width (l)");
    const double ign = r.given.contains("--ignition")
                           ? r.ignition
                           : (candle ? kCandleIgnitionDefault : 0.0);
    if (r.given.contains("--ignition") &&
        !(std::isfinite(ign) && ign >= 1 && ign <= len))
        return bad(2, "--ignition", "must be in [1, --length] l (the IC's "
                                    "burning column)");
    if (candle && r.quiescent && r.given.contains("--ignition"))
        return bad(2, "--ignition", "does not apply with --quiescent (no "
                                    "fuel: the IC is the ambient at rest)");
    const double grav =
        candle ? (r.given.contains("--gravity") ? r.gravity
                                                : kCandleGravityDefault)
               : 0.0;
    if (!(std::isfinite(grav) && grav >= 0 && grav <= 1000))
        return bad(2, "--gravity", "must be in [0, 1000] U_jet^2/l (g along "
                                   "-x; 0 = the diffusion jet bit for bit)");
    if (!(std::isfinite(r.pulse) && r.pulse >= 0 && r.pulse <= 0.5))
        return bad(2, "--pulse", "must be in [0, 0.5] (fraction of the jet "
                                 "speed)");
    if (r.pulse > 0 && !(std::isfinite(r.pulseFreq) && r.pulseFreq > 0))
        return bad(2, "--pulse-freq", "must be finite and > 0 with --pulse "
                                      "(U_jet/l)");
    if (r.given.contains("--pulse-freq") && !(r.pulse > 0))
        return bad(2, "--pulse-freq", "needs --pulse > 0");
    const double uMax = std::max({1.0, r.coflowSpeed, up}) * (1 + r.pulse);
    if (uMax * c.mach > 0.3)
        return bad(2, "--mach", std::format("the fastest inflow stream's Mach "
                                            "{:.3g} exceeds 0.3",
                                            uMax * c.mach));
    if (candle && grav > 0) {
        const CandleScales cs = candleScales(md, grav, W, len, c.mach);
        if (cs.machBuoyant > 0.3)
            return bad(2, "--mach", std::format(
                                        "the buoyant Mach M sqrt(2 g L (sigma "
                                        "- 1)/sigma) = {:.3g} exceeds 0.3: "
                                        "lower --mach, --gravity or --length",
                                        cs.machBuoyant));
        if (cs.machBuoyant > 0.15)
            c.warnings.push_back(std::format(
                "buoyant Mach {:.3g} (free-fall speed {:.3g} U): O(M^2) "
                "compressibility in the plume",
                cs.machBuoyant, cs.uFreeFall));
        if (!r.quiescent && cs.froude > 0.3)
            c.warnings.push_back(std::format(
                "Froude number U^2/(g W) = {:.3g}: the fuel jet's momentum is "
                "not negligible against buoyancy (not a candle)",
                cs.froude));
        if (cs.adiabatic > 0.02)
            c.warnings.push_back(std::format(
                "adiabatic fraction (g/cp) L/(T_ad - T0) = {:.3g}: the "
                "inflated Mach makes the column's lapse rate visible",
                cs.adiabatic));
    }
    c.nscbcEtaIn = r.nscbcEtaIn;
    if (!finitePos(c.nscbcEtaIn))
        return bad(2, "--nscbc-etain", "must be finite and > 0");
    c.nscbcThermalInflow = r.nscbcTinflow;
    c.nscbcSink = r.given.contains("--nscbc-sink") ? r.nscbcSink : 16;
    if (c.nscbcSink < 0) return bad(2, "--nscbc-sink", "must be >= 0 (0 = off)");
    c.nscbcSinkAmp = r.nscbcSinkAmp;
    if (!(c.nscbcSinkAmp > 0 && c.nscbcSinkAmp <= 2))
        return bad(2, "--nscbc-sink-amp", "must be in (0, 2] (the SSP-RK3 "
                                          "real-axis bound)");

    const int nb = c.nscbcSink;
    const int half = int(std::ceil((0.5 * W + r.pilotWidth + wco) * cells - 1e-9));
    const int ny = 2 * half;
    const int nx = int(std::ceil(len * cells - 1e-9)) + 2 * nb;
    auto box = makeBox(nx, ny, 1, nx * h, ny * h, 1.0);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    BunsenGeom& b = c.bunsen;
    b.W = W;
    b.uJet = 1;
    b.uCo = r.coflowSpeed;
    b.wCo = wco;
    b.pilot = r.pilotWidth;
    b.uPilot = up;
    b.tCo = b.tPilot = 1; // unused: the inflow is the Burke-Schumann state
    b.edge = edge;
    b.yc = half * h;
    b.coneH = 0;
    b.pulse = r.pulse;
    b.pulseFreq = r.pulseFreq;
    c.diff = md;
    c.flameDiffusion = true;
    c.flameCandle = candle;
    c.flameGravity = grav;
    c.flameQuiescent = candle && r.quiescent;
    c.flameIgnition = c.flameQuiescent ? 0.0 : ign;
    c.flameTu = 1.0 / (kGammaAir * c.mach * c.mach);
    c.flameCells = cells;
    c.n = c.grid.nx;
    c.h = h;
    c.L = c.grid.nx * h;
    c.re = W / c.prandtl; // U W rho0 / mu, mu = Pr
    c.nu = c.prandtl;
    c.uInf = 1.0;
    c.far = FarField::Nscbc;
    c.bodyKind = BodyKind::None;

    if (auto e = resolveCfl(r, c, 0.4); e) return std::unexpected(*e);
    if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
    c.ft = r.tt;
    c.tEnd = r.tt;
    if (auto e = resolveDtRecompute(r, c, 16); e) return std::unexpected(*e);
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    if (auto e = resolveEmit(c, {Emit::Stats}); e) return std::unexpected(*e);

    if (c.mach > 0.1)
        c.warnings.push_back(std::format(
            "--mach {:.4g}: the heat release's dilatation carries an O(M^2) "
            "compressibility error (gamma M^2 (sigma - 1) = {:.3g})",
            c.mach, kGammaAir * c.mach * c.mach * (md.sigma - 1)));
    if (nb == 0)
        c.warnings.push_back("--nscbc-sink 0: the bare characteristic "
                             "boundary (nothing drains mesh-scale content)");
    c.meta = {.D = 1.0,
              .re = c.re,
              .mach = c.mach,
              .frameDt = c.fdt,
              .body = candle ? (c.flameQuiescent ? "candle box at rest"
                                                 : "candle diffusion flame")
                             : "slot jet diffusion flame",
              .farField = "nscbc",
              .physics = name(c.physics),
              .kick = 0.0,
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

// The 1D freely propagating premixed flame (docs/chem.md): model C0, the
// oracle eigenvalue resolved HERE (pure FP64, ~0.1 s) so the grid is counted
// in the oracle's own thermal thickness and --dry-run prints Lambda.
std::expected<Case, ConfigError> validateFlame(const RawConfig& r, Case c)
{
    if (r.flameMode == 2 || r.flameMode == 3)
        return validateDiffusion(r, std::move(c));
    if (auto e = rejectGiven(r, {"--re"}, "does not apply: a flame is set by "
                                          "--expansion, --zeldovich and "
                                          "--prandtl (Re = 1/Pr on delta_D)");
        e)
        return std::unexpected(*e);
    if (auto e = rejectGiven(r, {"--ft"}, "does not apply: the run length is "
                                          "--tt (flame times delta_D/S_L)");
        e)
        return std::unexpected(*e);
    if (auto e = rejectGiven(r, {"--n"}, "does not apply: the grid is "
                                         "--cells-per-delta over --upstream "
                                         "+ --downstream thermal thicknesses");
        e)
        return std::unexpected(*e);
    if (c.physics != Physics::Compressible)
        return bad(2, "--physics", "a flame is compressible only (no low-Mach "
                                   "branch until P7; docs/chem.md)");
    const bool bunsen = r.flameMode == 1;
    const bool chamber = r.flameMode == kFlameModeChamber;
    if (r.flameMode != 0 && !bunsen && !chamber)
        return bad(2, "--mode", "must be premixed, bunsen, diffusion, candle "
                                "or chamber");
    if (auto e = chamberFlagScope(r, chamber); e) return std::unexpected(*e);
    if (auto e = rejectGiven(r,
                             {"--stoich", "--fuel-y", "--ox-y", "--damkohler",
                              "--cells-per-unit", "--ignition"},
                             "only applies to --mode diffusion and candle");
        e)
        return std::unexpected(*e);
    if (auto e = rejectGiven(r, {"--gravity", "--quiescent"},
                             "only applies to --mode candle");
        e)
        return std::unexpected(*e);
    // Each mode's own flags, rejected by name in the other.
    if (auto e = bunsen ? rejectGiven(r,
                                      {"--upstream", "--downstream",
                                       "--inflow-speed", "--wrinkle",
                                       "--wrinkle-modes", "--ny",
                                       "--profile-dt"},
                                      "does not apply to --mode bunsen (the "
                                      "slot burner sets the box and the IC; "
                                      "x-profiles are a 1D diagnostic)")
                        : rejectGiven(r,
                                      {"--slot-width", "--jet-speed",
                                       "--coflow-speed", "--coflow-width",
                                       "--coflow-temp", "--pilot-width",
                                       "--pilot-temp", "--pilot-speed",
                                       "--slot-edge", "--length", "--pulse",
                                       "--pulse-freq"},
                                      "only applies to --mode bunsen");
        e)
        return std::unexpected(*e);
    if (r.nz != 1)
        return bad(2, "--nz", "must be 1: multi-D flames are P2 (docs/chem.md)");
    if (!bunsen && r.ny < 4) return bad(2, "--ny", "must be >= 4 (periodic y)");
    c.mach = chamber && !r.given.contains("--mach") ? 0.02 : r.mach;
    if (!(finitePos(c.mach) && c.mach <= 0.05))
        return bad(2, "--mach", "must be in (0, 0.05]: the flame Mach number "
                                "S_L/c_u, inflated above a real flame's "
                                "~1e-3 to keep the acoustic dt affordable");
    if (!(r.expansion > 1 && r.expansion <= 10))
        return bad(2, "--expansion", "must be in (1, 10] (sigma = Tb/Tu)");
    if (!(r.zeldovich >= 2 && r.zeldovich <= 20))
        return bad(2, "--zeldovich", "must be in [2, 20]");
    c.prandtl = r.prandtl;
    if (!finitePos(c.prandtl)) return bad(2, "--prandtl", "must be finite and > 0");
    c.sutherland = r.visc == 1;
    c.tinfK = r.tinf;
    if (c.sutherland && !(r.given.contains("--tinf") && finitePos(c.tinfK)))
        return bad(2, "--tinf", "is required with --visc sutherland: the "
                                "unburnt temperature in kelvin (Sutherland's "
                                "110.4 K needs a dimensional anchor)");
    if (!c.sutherland && r.given.contains("--tinf"))
        return bad(2, "--tinf", "only anchors --visc sutherland");
    if (!(std::isfinite(r.cellsPerDelta) && r.cellsPerDelta >= 4))
        return bad(2, "--cells-per-delta", "must be >= 4");
    if (!(std::isfinite(r.upstream) && r.upstream >= 6))
        return bad(2, "--upstream", "must be >= 6 thermal thicknesses");
    if (!(std::isfinite(r.downstream) && r.downstream >= 4))
        return bad(2, "--downstream", "must be >= 4 thermal thicknesses");
    if (!finitePos(r.inflowSpeed))
        return bad(2, "--inflow-speed", "must be finite and > 0 (x S_L)");
    if (!(std::isfinite(r.wrinkle) && r.wrinkle >= 0))
        return bad(2, "--wrinkle", "must be finite and >= 0 (thermal thicknesses)");
    if (r.given.contains("--wrinkle-modes") && !r.given.contains("--wrinkle"))
        return bad(2, "--wrinkle-modes", "needs --wrinkle");
    if (r.wrinkle > 0 && (r.wrinkleModes < 1 || 8 * r.wrinkleModes > r.ny))
        return bad(2, "--wrinkle-modes", "must be in [1, ny/8] (8 cells per "
                                         "shortest seeded wavelength)");
    if (r.wrinkle > 0 && r.wrinkle * std::sqrt(double(r.wrinkleModes)) >
                             std::min(r.upstream, r.downstream) / 2)
        return bad(2, "--wrinkle", "peak displacement A sqrt(N) must stay "
                                   "within half of --upstream and --downstream");
    c.nscbcEtaIn = r.nscbcEtaIn;
    if (!finitePos(c.nscbcEtaIn))
        return bad(2, "--nscbc-etain", "must be finite and > 0");
    c.nscbcThermalInflow = r.nscbcTinflow;
    c.nscbcSink = r.given.contains("--nscbc-sink") ? r.nscbcSink : 16;
    if (c.nscbcSink < 0) return bad(2, "--nscbc-sink", "must be >= 0 (0 = off)");
    c.nscbcSinkAmp = r.nscbcSinkAmp;
    if (!(c.nscbcSinkAmp > 0 && c.nscbcSinkAmp <= 2))
        return bad(2, "--nscbc-sink-amp", "must be in (0, 2] (the SSP-RK3 "
                                          "real-axis bound)");

    PremixedModel md;
    md.sigma = r.expansion;
    md.beta = r.zeldovich;
    md.Pr = c.prandtl;
    md.sutherland = c.sutherland;
    md.sOverTu = c.sutherland ? Transport::sutherlandK / c.tinfK : 0.0;
    auto eig = premixedEigen(md);
    if (!eig) return bad(2, "--zeldovich", eig.error());
    c.flameModel = md;
    c.flameEig = std::move(*eig);
    if (chamber) return finishChamber(r, std::move(c));
    c.flameCells = r.cellsPerDelta;
    c.flameUp = r.upstream;
    c.flameDown = r.downstream;
    c.flameUIn = r.inflowSpeed;
    c.flameWrinkle = r.wrinkle;
    c.flameWrinkleModes = r.wrinkleModes;
    c.flameTu = 1.0 / (kGammaAir * c.mach * c.mach);

    const double dth = c.flameEig.dthOverDD;
    if (bunsen && !r.given.contains("--cells-per-delta"))
        c.flameCells = kBunsenCellsDefault;
    const double h = dth / c.flameCells;
    const int nb = c.nscbcSink;
    if (bunsen) {
        auto e = bunsenGeometry(r, c, nb);
        if (!e) return std::unexpected(e.error());
    } else {
        const int nx = int(std::ceil((c.flameUp + c.flameDown) * c.flameCells -
                                     1e-9)) + 2 * nb;
        auto box = makeBox(nx, r.ny, 1, nx * h, r.ny * h, 1.0);
        if (!box) return std::unexpected(box.error());
        c.grid = *box;
        c.flameXf0 = nb * h + c.flameUp * dth;
        c.uInf = c.flameUIn;
    }
    c.n = c.grid.nx;
    c.h = h;
    c.L = c.grid.nx * h;
    c.re = 1.0 / c.prandtl; // rho_u S_L delta_D / mu
    c.nu = c.prandtl;       // mu (rho_u = 1): lambda_u = cp
    c.far = FarField::Nscbc;
    c.bodyKind = BodyKind::None;

    if (auto e = resolveCfl(r, c, 0.4); e) return std::unexpected(*e);
    if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
    c.ft = r.tt;
    c.tEnd = r.tt; // tau = delta_D / S_L = 1
    if (auto e = checkProfileDt(r, c); e) return std::unexpected(*e);
    if (auto e = resolveDtRecompute(r, c, 16); e) return std::unexpected(*e);
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    if (auto e = resolveEmit(c, {Emit::Stats}); e) return std::unexpected(*e);

    if (c.mach > 0.02)
        c.warnings.push_back(std::format(
            "--mach {:.4g} is above the measured ladder top 0.02: the "
            "compressible flame speed carries an O(M^2) offset "
            "(gamma M^2 (sigma - 1) = {:.3g})",
            c.mach, kGammaAir * c.mach * c.mach * (md.sigma - 1)));
    if (nb == 0)
        c.warnings.push_back("--nscbc-sink 0: the bare characteristic "
                             "boundary (nothing drains mesh-scale content)");
    c.meta = {.D = 1.0,
              .re = c.re,
              .mach = c.mach,
              .frameDt = c.fdt,
              .body = c.flameBunsen ? "slot Bunsen flame" : "premixed flame",
              .farField = "nscbc",
              .physics = name(c.physics),
              .kick = 0.0,
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

// --- flow -------------------------------------------------------------------
// validateFlow() below runs these stages in order; each one returns the first
// error of its own checks.

// Stage: the body (--body/--scene), the far field it may have, and the
// wall-mounted bodies' rules.
std::optional<ConfigError> flowBody(const RawConfig& r, Case& c)
{
    // A scene file replaces --body: main loads it (phase 1.5) and hands the
    // parsed Scene to the geometry build; the validator only records the path
    // and applies the body-independent checks.
    if (!r.scene.empty())
    {
        if (!r.body.empty())
            return err(2, "--scene", "--scene and --body are mutually exclusive "
                                     "(the scene lists its own objects)");
        c.bodyKind = BodyKind::Scene;
        c.scenePath = r.scene;
    }
    else if (r.body.empty())
        return err(2, "--body", "required: one of " + bodyKeywords()
                                + ", a .stl/.gltf/.glb file, or "
                                  "--scene FILE.toml");
    // Mesh and Scene are kinds the validator ASSIGNS, never keywords the user
    // types: `--body mesh` falls through to the suffix test below and is then
    // rejected by name, as it always was.
    else if (const auto k = bodyKindFromName(r.body);
             k && *k != BodyKind::Mesh && *k != BodyKind::Scene)
        c.bodyKind = *k;
    else if (hasSuffix(r.body, ".stl") || hasSuffix(r.body, ".gltf")
             || hasSuffix(r.body, ".glb"))
    {
        c.bodyKind = BodyKind::Mesh;
        c.meshPath = r.body;
    }
    else
        return err(2, "--body", "unrecognised body '" + r.body + "'; expected "
                                + bodyKeywords()
                                + ", or a path ending in .stl/.gltf/.glb");

    // Body-free box (wall boundary layers, channel-like cases): the
    // incompressible runner requires solid cells, so this is compressible-only.
    if (c.bodyKind == BodyKind::None && c.physics != Physics::Compressible)
        return err(2, "--body", "--body none is compressible-only (the "
                                "incompressible tunnel needs a body)");

    // Far field defaults follow the physics when the user did not choose one.
    // Sponge is NEVER a default (any physics, any mode): it is an escape hatch that
    // eats the near field in a relaxation band. The compressible default is
    // the physical characteristic boundary; --far sponge must be asked for.
    if (!r.given.contains("--far")) c.far = defaultFar(c.physics);

    const bool comp = c.physics == Physics::Compressible;
    if (!farAllowed(c.physics, c.far))
        return err(2, "--far", std::string("--far ") + name(c.far)
                   + (comp ? " is an incompressible far field; compressible "
                             "runs use sponge, nscbc or periodic"
                           : " is a compressible far field; incompressible "
                             "runs use tunnel, fringe or periodic"));

    c.open = r.open;
    if (c.open && c.far != FarField::Sponge && c.far != FarField::Nscbc
        && c.far != FarField::Tunnel)
        return err(2, "--open", std::string("--open has no meaning with --far ")
                   + name(c.far)
                   + " (tunnel, sponge and nscbc have lateral faces)");

    // Floor-mounted bodies (fence blocker, raised floor): they sit ON the
    // tunnel floor (sealed through the ghost row), so they need the
    // incompressible tunnel's solid lateral wall. Compressible wall rows are
    // far-field mirror rows; a ghost body intersecting them is unvalidated --
    // rejected, not silently degraded.
    if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
    {
        const char* b = c.bodyKind == BodyKind::Fence ? "fence" : "floor";
        if (comp && c.bodyKind == BodyKind::Fence)
            return err(2, "--body", "fence is incompressible-only for now (a "
                                    "mid-plate blocker on the compressible "
                                    "mirror wall rows is unvalidated)");
        // Compressible floor: NSCBC/sponge x planes skip the fully-solid
        // rows (the ghost wall owns them); periodic has no walls at all.
        if (comp && c.far == FarField::Periodic)
            return err(2, "--far", "floor needs closed y walls; use --far "
                                   "nscbc (default) or an explicit sponge");
        if (!comp && c.far != FarField::Tunnel)
            return err(2, "--far", std::string(b) +
                                   " sits on the tunnel floor; incompressible "
                                   "runs need --far tunnel (periodic/fringe "
                                   "have no wall for it to stand on)");
        if (c.open)
            return err(2, "--open", std::string("--open removes the lateral "
                                    "walls; the ") + b + " needs the floor wall");
    }
    return std::nullopt;
}

// Stage: the grid: the two physics' box conventions and the body's minimum resolution.
std::optional<ConfigError> flowGrid(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Grid: two separately validated conventions, dispatched on --physics.
    const int ny = r.given.contains("--ny") ? r.ny : r.n;
    if (ny < 1) return err(2, "--ny", "must be >= 1");
    // --nz defaults to 1 (native 2D) for the z-invariant bodies and to --n for
    // the genuinely 3D ones: a sphere or a mesh in a one-cell-thick box has no
    // solid cells at all, which used to surface as a silent no-body run.
    const bool body3d = c.bodyKind == BodyKind::Sphere || c.bodyKind == BodyKind::Mesh;
    const int nzI = r.given.contains("--nz") ? r.nz : (body3d ? r.n : 1);
    if (auto e = checkNz(nzI, "must be >= 1 (interior cells; 1 = native 2D)"); e)
        return e;
    // The revolved O-grid reads --nz 1 as its axisymmetric wedge.
    const bool revolved = r.ogrid && c.bodyKind == BodyKind::Sphere;
    if (body3d && nzI == 1 && !revolved)
        return err(2, "--nz", std::string("--body ") + name(c.bodyKind)
                   + " is a 3D body; --nz 1 leaves it with no solid cells");
    if (!finitePos(r.dr)) return err(2, "--dr", "must be finite and > 0");
    c.dr = r.dr;
    c.nzInterior = nzI;

    c.uInf = 1.0;
    if (r.ogrid)
    {
        // Body-fitted O-grid (dns/ogrid.hpp): the domain is the DISC of radius
        // rFar around the body, not a box, and the wall is a mesh face -- so
        // every immersed-body and box knob is meaningless here and rejected by
        // name rather than silently ignored. --n is the angular cell count,
        // --ny the radial one INCLUDING the two ghost rows.
        if (!comp)
            return err(2, "--ogrid", "the body-fitted O-grid is compressible-"
                                     "only (the curvilinear path is)");
        if (c.bodyKind == BodyKind::None)
            return err(2, "--ogrid", "wraps a body; there is none");
        if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
            return err(2, "--ogrid", "wall-mounted bodies are not star-shaped "
                                     "about a centre and have no annulus "
                                     "around them");
        if (auto e = rejectGiven(
                r, {"--focal", "--stretch", "--cut-flux", "--penalize", "--open",
                    "--far", "--dr", "--seq"},
                "belongs to the Cartesian box with an immersed body; --ogrid "
                "replaces both (the wall IS a mesh face, and the far field is "
                "the outer circle)");
            e)
            return e;
        if (r.n < 16)
            return err(2, "--n", "is the ANGULAR cell count on an O-grid; "
                                 "below 16 the contour is not resolved");
        if (ny < 6)
            return err(2, "--ny", "is the RADIAL cell count on an O-grid, two "
                                  "of which are ghost rows; >= 6 needed");
        if (!(std::isfinite(r.ogridFar) && r.ogridFar > 1.0))
            return err(2, "--ogrid-far", "far-boundary radius in DIAMETERS; "
                                         "must be finite and > 1");
        if (!(std::isfinite(r.ogridBeta) && r.ogridBeta >= 0))
            return err(2, "--ogrid-beta", "radial clustering; must be finite "
                                          "and >= 0 (0 = uniform)");
        const double D = pi / 4.0;
        c.ogrid = true;
        c.ogridRFar = r.ogridFar * D;
        c.ogridBeta = r.ogridBeta;
        if (!revolved && r.given.contains("--ogrid-wedge"))
            return err(2, "--ogrid-wedge", "is the revolved (--body sphere) "
                                           "O-grid's axisymmetric wedge angle");
        // The O-grid start is uniform (or at rest) and never reads --kick;
        // its trip is --ogrid-trip.
        if (r.given.contains("--kick"))
            return err(2, "--kick", "is the Cartesian box's wake trip; the "
                                    "O-grid start never applies it -- use "
                                    "--ogrid-trip A");
        // The axisymmetric wedge has no odd modes to seed, and the planar
        // vortex is not solenoidal in the meridional plane (v/r).
        if (revolved && nzI <= 1 && r.given.contains("--ogrid-trip"))
            return err(2, "--ogrid-trip", "seeds a solenoidal vortex behind "
                                          "the body; the axisymmetric wedge "
                                          "(--nz 1) has no mode it could "
                                          "seed and no planar vortex is "
                                          "solenoidal there");
        if (!(std::isfinite(r.ogridTrip) && r.ogridTrip >= 0 &&
              r.ogridTrip <= 0.5))
            return err(2, "--ogrid-trip", "the trip amplitude, / U: must "
                                          "be in [0, 0.5]");
        // Default 0 (bit for bit the untripped start) except on the full
        // revolution, whose pi-symmetry locks out the m = 1 mode.
        c.ogridTrip = r.given.contains("--ogrid-trip") ? r.ogridTrip
                      : revolved && nzI > 1            ? kOgridTripDefault
                                                       : 0.0;
        if (!(revolved && nzI > 1) && r.given.contains("--init-from"))
            return err(2, "--init-from", "starts the full-revolution sphere "
                                         "O-grid (--body sphere --nz N > 1) "
                                         "from a converged wedge only");
        if (revolved && nzI > 1)
        {
            // The FULL revolution: N azimuthal cells, Cyclic k, the axis
            // ghost columns copied from k + N/2 (gpu/axis_bc.hpp). N even so
            // every cell has its partner across the axis; N >= 4 so that
            // partner is not a k neighbour (buildRevolvedOGrid's own floor).
            if (r.given.contains("--ogrid-wedge"))
                return err(2, "--ogrid-wedge", "is the axisymmetric wedge's "
                                               "angle (--nz 1); --nz N > 1 is "
                                               "the full revolution");
            if (nzI % 2 != 0 || nzI < 4)
                return err(2, "--nz", std::format(
                                          "is the full revolution's azimuthal "
                                          "cell count: it must be even and "
                                          ">= 4 (every cell has its partner "
                                          "across the axis, k + N/2); got {}",
                                          nzI));
            c.ogridRevolved = true;
            c.ogridNphi = nzI;
            if (r.given.contains("--init-from")) {
                // The file is read by main (phase 3); here only its shape.
                if (r.initFrom.empty() ||
                    r.initFrom.find_first_of("\"\\\n\r\t") != std::string::npos)
                    return err(2, "--init-from", "a wedge run's --out prefix "
                                                 "(its _final.bin and "
                                                 "_mesh.bin); not empty, no "
                                                 "quotes or control "
                                                 "characters");
                if (r.given.contains("--ramp"))
                    return err(2, "--ramp", "with --init-from: the run "
                                            "starts from the wedge's state, "
                                            "not from rest");
                c.ogridInitFrom = r.initFrom; // flowRans zeroes the ramp
            }
            c.ogridWedgeDeg = 0;
            c.ogridSpan = 1.0;
            c.grid = ogridRevolvedGrid(
                r.n, ny,
                RevolvedSpec{.wedge = false, .dphi = 2.0 * pi / nzI,
                             .nphi = nzI},
                0.5 * D, c.ogridRFar);
            c.nzInterior = nzI;
            // The polygonal body: N planar facets about the axis, frontal
            // area sin(dphi)/dphi of the circle's (qRef stays pi D^2/4).
            const double dphi = 2.0 * pi / nzI;
            const double facet = 1.0 - std::sin(dphi) / dphi;
            if (facet > 1e-2)
                c.warnings.push_back(std::format(
                    "--nz {}: the revolved body is a {}-sided polygon about "
                    "the axis, its frontal area {:.2f}% below the circle's "
                    "(qRef is the circle's); --nz 32 brings it to 0.64%, 64 "
                    "to 0.16%",
                    nzI, nzI, 100.0 * facet));
        }
        else if (revolved)
        {
            // A body of revolution: the O-grid revolved about x. Axisymmetry
            // is a mesh plus a ghost hook (dns/ogrid.hpp, gpu/axis_bc.hpp).
            if (!(std::isfinite(r.ogridWedge) && r.ogridWedge > 0 &&
                  r.ogridWedge <= 10))
                return err(2, "--ogrid-wedge", "the wedge angle in degrees, "
                                               "in (0, 10]");
            c.ogridRevolved = true;
            c.ogridWedgeDeg = r.ogridWedge;
            c.ogridSpan = 1.0;
            c.grid = ogridRevolvedGrid(
                r.n, ny,
                RevolvedSpec{.wedge = true, .dphi = r.ogridWedge * pi / 180.0},
                0.5 * D, c.ogridRFar);
            c.nzInterior = 1;
        }
        else
        {
            // Spanwise extent: pi D, the usual cylinder-DNS span. nz == 1
            // keeps the unit depth the 2D convention requires.
            c.ogridSpan = nzI > 1 ? pi * D : 1.0;
            c.grid = ogridGrid(r.n, ny, nzI, 0.5 * D, c.ogridRFar, c.ogridSpan);
        }
        c.ogridH1 = ogridNominalH1(ny, 0.5 * D, c.ogridRFar, c.ogridBeta);
        if (auto e = checkCells(c.grid.size()); e) return e;
        c.L = 2.0 * c.ogridRFar;
        c.h = c.grid.hy; // nominal radial spacing; the mesh reports the real h1
        // The body sits at the origin: the mesh is built about its centre.
        c.placement = {0.0, 0.0, 0.0, D};
    }
    else if (r.given.contains("--ogrid-wedge"))
        return err(2, "--ogrid-wedge", "is the revolved (--body sphere) "
                                       "O-grid's axisymmetric wedge angle");
    else if (r.given.contains("--ogrid-trip"))
        return err(2, "--ogrid-trip", "seeds the odd azimuthal modes of the "
                                      "full-revolution sphere O-grid (--ogrid "
                                      "--body sphere --nz N > 1) only");
    else if (r.given.contains("--init-from"))
        return err(2, "--init-from", "starts the full-revolution sphere O-grid "
                                     "(--ogrid --body sphere --nz N > 1) from "
                                     "a converged wedge only");
    else if (comp)
    {
        // D = pi/4 fixed, h scales with ny AND --dr: box = dr*D in y
        // (dr*(n/ny)*D in x), D spans ny/dr cells. The dr = 8 default keeps
        // the historical 8D x 8D box BIT-IDENTICAL (the *(dr/8) factor is an
        // exact *1.0 and the 2.1107 placement literal is preserved). No ghost
        // ring: the compressible far fields act on interior rows.
        if (c.dr < 4)
            return err(2, "--dr", "compressible box is dr*D wide; below 4 the "
                                  "body sits under 1D from the inflow row");
        const double D = pi / 4.0;
        c.h = D / 64.0 * (512.0 / double(ny)) * (c.dr / 8.0);
        const double lx = r.n * c.h, ly = ny * c.h;
        const double lz = nzI > 1 ? nzI * c.h : 1.0;
        auto box = makeBox(r.n, ny, nzI, lx, ly, lz);
        if (!box) return box.error();
        c.grid = *box;
        c.L = lx;
        // The floor is pinned to the box bottom (no ghost ring here): the
        // slab spans [0, D], cy = centre of the blocked strip.
        c.placement = {c.dr == 8.0 ? 2.1107 : lx / 3.0 + 0.0163,
                       c.bodyKind == BodyKind::Floor ? D / 2.0
                                                     : ly / 2.0 + 0.0146,
                       nzI > 1 ? lz / 2.0 : 0.5, D};
    }
    else
    {
        // L = 2 pi, h = L/n, one ghost cell on every side; --nz is interior.
        c.L = 2 * pi;
        c.h = c.L / double(r.n);
        const int nxT = r.n + 2, nyT = ny + 2;
        const int nzT = nzI <= 1 ? 1 : nzI + 2;
        const double lz = nzT == 1 ? 1.0 : nzT * c.h;
        auto box = makeBox(nxT, nyT, nzT, nxT * c.h, nyT * c.h, lz);
        if (!box) return box.error();
        c.grid = *box;
        const double D = c.L / c.dr;
        // Per-body placement offsets: each deliberately puts the body off-node
        // so the shedding instability trips. Carried verbatim (masterplan §6).
        // The fence is pinned to the floor instead: the wall face is at y = h
        // (top of the ghost row) and the blocker spans [wall, wall + D], so
        // cy = centre of the blocked strip. Its top edge is off-node through
        // the 0.31h x-offset alone.
        // y (and a sphere's z) is centred on the INTERIOR, cells*h long, as
        // the compressible box centres on ly. The square-box form L/2 is kept
        // verbatim when the axis has n cells -- every ny == n run is
        // bit-identical -- and is wrong otherwise: --ny 32 at --n 64 put the
        // body centre outside the box.
        const auto mid = [&](int cells) {
            return cells == r.n ? c.L / 2.0 : 0.5 * double(cells) * c.h;
        };
        const double cx = c.h + c.L / 3.0 + 0.31 * c.h;
        const double cy = c.bodyKind == BodyKind::Fence
                                  || c.bodyKind == BodyKind::Floor
                              ? c.h + D / 2.0
                              : c.h + mid(ny) + 0.17 * c.h;
        const double cz = c.bodyKind == BodyKind::Sphere
                              ? c.h + mid(nzT == 1 ? r.n : nzI) + 0.23 * c.h
                              : (nzT == 1 ? 0.5 * c.h : c.h + nzI * c.h / 2.0);
        c.placement = {cx, cy, cz, D};
    }

    // (An O-grid's resolution is its angular --n >= 16, and its nominal
    // radial hy measures nothing about D.)
    if (!c.ogrid && c.placement.D < 4 * c.h)
        return err(2, "--dr", "body diameter is under 4 cells; refine --n or lower "
                              "--dr");
    // The fence is D/8 thick; below 32 cells of D the blocker itself is under
    // 4 cells and the sampled cut degenerates.
    if (c.bodyKind == BodyKind::Fence && c.placement.D < 32 * c.h)
        return err(2, "--dr", "fence thickness is D/8; D under 32 cells makes it "
                              "thinner than 4 cells -- refine --n or lower --dr");
    if (c.bodyKind == BodyKind::Floor && c.dr < 2)
        return err(2, "--dr", "floor solidifies the bottom 1/dr of the channel; "
                              "--dr under 2 blocks more than half of it");
    return std::nullopt;
}

// Stage: Mach, viscosity, CFL.
std::optional<ConfigError> flowPhysics(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Physics.
    c.mach = r.mach;
    if (comp)
    {
        if (auto e = checkMach(c.mach); e) return e;
        if (r.given.contains("--tol"))
            return err(2, "--tol", "compressible runs are fully explicit; there is "
                                   "no pressure solve to converge");
    }
    else
    {
        if (auto e = rejectGiven(r, {"--mach", "--kick"},
                                 "is compressible-only");
            e)
            return e;
    }

    c.nu = c.uInf * c.placement.D / c.re;
    if (auto e = resolveCfl(r, c, comp ? 0.4 : 0.8); e) return e;
    return std::nullopt;
}

// Stage: the compressible thermal knobs.
std::optional<ConfigError> flowThermal(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Thermal model. Every knob here is compressible-only and OFF by default:
    // a constant-viscosity adiabatic-wall run is bit-for-bit what it was, so
    // no measured anchor in the repo moves until a run asks for temperature.
    c.tinfK = r.tinf;
    c.prandtl = r.prandtl;
    c.sutherland = r.visc == 1;
    c.twallRatio = r.twall;
    c.bodyIsothermal = r.given.contains("--wall") && r.wallIso == 1;
    c.nscbcThermalInflow = r.nscbcTinflow;
    if (!comp)
        if (auto e = rejectGiven(r,
                                 {"--tinf", "--prandtl", "--visc", "--wall",
                                  "--twall", "--nscbc-tinflow"},
                                 "is compressible-only (the incompressible "
                                 "solver carries no energy equation)");
            e)
            return e;
    if (!finitePos(c.tinfK))
        return err(2, "--tinf", "must be finite and > 0 (kelvin)");
    if (!finitePos(c.prandtl))
        return err(2, "--prandtl", "must be finite and > 0");
    if (!finitePos(c.twallRatio))
        return err(2, "--twall", "must be finite and > 0 (a ratio to Tinf)");
    if (r.given.contains("--twall") && !c.bodyIsothermal)
        return err(2, "--twall", "sets the temperature of an ISOTHERMAL wall; "
                                 "add --wall isothermal (the default body wall "
                                 "is adiabatic)");
    if (c.bodyIsothermal && c.bodyKind == BodyKind::None)
        return err(2, "--wall", "an isothermal wall needs a body; there is none");
    if (c.nscbcThermalInflow && c.far != FarField::Nscbc)
        return err(2, "--nscbc-tinflow", "is the NSCBC entropy inflow target; "
                                         "use --far nscbc");
    return std::nullopt;
}

// Stage: SGS model, the aperture/Brinkman options, the DNS/LES label.
std::optional<ConfigError> flowLes(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    c.cs = r.les;
    if (!(std::isfinite(c.cs) && c.cs >= 0))
        return err(2, "--les", "must be finite and >= 0");
    c.sgsModel = r.sgs;
    if (r.given.contains("--sgs") && c.cs == 0)
        return err(2, "--sgs", "selects the SGS model for --les; give --les CS "
                               "(> 0) or drop --sgs");
    c.cutFlux = r.cutFlux;
    if (c.cutFlux && c.bodyKind == BodyKind::None)
        return err(2, "--cut-flux", "apertures the body's momentum fluxes; "
                                    "there is no body");
    if (c.cutFlux && r.given.contains("--stretch"))
        return err(2, "--cut-flux", "apertures are uniform-mesh geometry; the "
                                    "curvilinear path has none. Drop --stretch");
    if (c.cutFlux && comp)
        return err(2, "--cut-flux", "compressible cut-flux DIVERGES near the "
                                    "body (measured twice, 2026-07-23: the KG "
                                    "split flux loses its KE property under "
                                    "per-face area weighting). Incompressible "
                                    "only until redesigned");
    c.penalize = r.penalize;
    if (!(std::isfinite(c.penalize) && c.penalize >= 0))
        return err(2, "--penalize", "must be finite and >= 0");
    if (c.penalize > 0 && c.bodyKind == BodyKind::None)
        return err(2, "--penalize", "decays the near-body band; there is no body");
    if (c.penalize > 0 && r.given.contains("--stretch"))
        return err(2, "--penalize", "needs the uniform-mesh fluid fractions; "
                                    "drop --stretch");
    if (c.sgsModel == SgsModel::Sigma && c.cs > 0 && c.grid.nz == 1)
        return err(2, "--sgs", "sigma: D_sigma is identically ZERO for any 2D "
                               "flow (that is the model's design); at nz == 1 "
                               "it is a silent no-op. Use nz > 1 or smag");
    // The wall face itself takes the image of nu_t (molecular stress only);
    // the first fluid cell does not, and |S| is the wall shear there.
    if (c.cs > 0 && c.sgsModel == SgsModel::Smagorinsky &&
        c.bodyKind != BodyKind::None && c.grid.nz > 1)
        c.warnings.push_back(std::format(
            "--les {:g} --sgs smag next to a wall: |S| does not vanish in "
            "laminar shear, so the first fluid cell carries nu_t/nu ~ "
            "Cs^2 Delta+^2 = {:.3g} Delta+^2 (no wall damping). --sgs sigma "
            "vanishes in pure shear and is the wall-consistent choice",
            c.cs, c.cs * c.cs));
    c.lesLabel = c.cs > 0;
    return std::nullopt;
}

// Stage: the far-field parameters: NSCBC closure, sink, strip, --pot.
std::optional<ConfigError> flowNscbc(const RawConfig& r, Case& c)
{
    // Far-field parameters.
    c.nscbcBeta = r.nscbcBeta;
    c.nscbcEtaIn = r.given.contains("--nscbc-etain") ? r.nscbcEtaIn : 100.0;
    c.nscbcStrip = r.nscbcStrip;
    c.nscbcEtaInShear = r.nscbcEtaInShear;
    c.nscbcEtaLat = r.nscbcEtaLat;
    c.nscbcSink = r.nscbcSink;
    c.nscbcSinkAmp = r.nscbcSinkAmp;
    // The O-grid's far boundary is its own characteristic row (gpu/ogrid_bc);
    // runFlowOGrid reads none of the NSCBC knobs, so each one is rejected by
    // name rather than accepted and ignored -- including the strip that used
    // to "satisfy" the Re-800 rule below without doing anything.
    if (r.ogrid)
    {
        if (auto e = rejectGiven(
                r, {"--nscbc-strip", "--nscbc-etain", "--nscbc-beta",
                    "--nscbc-abc", "--nscbc-etain-shear", "--nscbc-etalat",
                    "--nscbc-sink", "--nscbc-sink-amp", "--nscbc-tinflow",
                    "--pot"},
                "the O-grid's far boundary is its own characteristic row "
                "(gpu/ogrid_bc), not NSCBC; the flag would be ignored");
            e)
            return e;
        c.nscbcSink = 0;
        if (c.re > 800)
            c.warnings.emplace_back(
                "o-grid far row is unmeasured above Re 800 (the NSCBC Re-800 "
                "rule does not apply to it)");
        return std::nullopt;
    }
    // The sink is ON by default for --far nscbc wherever it is defined
    // (uniform mesh, whole faces, band fits twice on every open axis) --
    // measured free where it does not absorb: cd moved +0.003% on the Re60
    // nscbc cylinder, and it is what carries a run past Re 800 without the
    // strip's near-field cost. An explicit --nscbc-sink (any value, 0 = off)
    // always wins; incompatible cases fall back to 0 rather than rejecting,
    // because a DEFAULT that does not apply is absence, not an error.
    if (!r.given.contains("--nscbc-sink") && c.far == FarField::Nscbc)
    {
        const bool uniform = !remapsMesh(r) && !r.ogrid;
        const bool fits = 32 < c.grid.nx &&
                          (!c.open || (32 < c.grid.ny &&
                                       (c.grid.nz == 1 || 32 < c.grid.nz)));
        c.nscbcSink =
            uniform && fits && c.bodyKind != BodyKind::Floor ? 16 : 0;
    }
    if (c.far == FarField::Nscbc)
    {
        if (!finitePos(c.nscbcEtaIn))
            return err(2, "--nscbc-etain", "must be > 0");
        // Empirically bracketed stability window (masterplan §6.3): 0 pumps a
        // growing global mode, 1.0 diverges at t = 35.
        if (!(std::isfinite(c.nscbcBeta) && c.nscbcBeta >= 0 && c.nscbcBeta < 1))
            return err(2, "--nscbc-beta", "must be in [0,1); 1.0 is measured to diverge "
                                          "at t = 35 (full transverse cancellation "
                                          "is ill-posed at low Mach)");
        if (c.nscbcStrip < 0 || c.nscbcStrip >= c.grid.nx / 4)
            return err(2, "--nscbc-strip", "must be >= 0 and < nx/4");
        if (c.re > 800 && c.nscbcStrip == 0 && c.nscbcSink == 0)
            return err(2, "--nscbc-strip", "pure NSCBC is validated only up to "
                                           "Re 800 (it diverges at Re 2000, 2048^2); "
                                           "pass --nscbc-sink 16 (scale-selective, "
                                           "preferred) or --nscbc-strip ~ny/64 "
                                           "(freestream sponge; docs/nscbc.md)");
        if (c.nscbcSink < 0 || 2 * c.nscbcSink >= c.grid.nx ||
            (c.open && c.nscbcSink > 0 && 2 * c.nscbcSink >= c.grid.ny))
            return err(2, "--nscbc-sink", "band must be >= 0 and fit twice on "
                                          "every open axis");
        if (!(std::isfinite(c.nscbcSinkAmp) && c.nscbcSinkAmp > 0 &&
              c.nscbcSinkAmp <= 2.0))
            return err(2, "--nscbc-sink-amp",
                       "must be in (0, 2]; sigma dt <= 2 * cfl * axes is the "
                       "SSP-RK3 real-axis bound at cfl 0.4");
        // The sink's cutoff is quoted in CELLS on a uniform mesh; under a
        // remap the physical cutoff varies across the band, which needs its
        // own measurement before it is allowed. Rejected, never ignored.
        if (c.nscbcSink > 0 && remapsMesh(r))
            return err(2, "--nscbc-sink", "is measured on the uniform mesh; "
                                          "--stretch/--focal remap the band "
                                          "(unmeasured -- not yet allowed)");
        if (c.nscbcSink > 0 && c.bodyKind == BodyKind::Floor)
            return err(2, "--nscbc-sink", "band rows overlap the solid floor "
                                          "rows the ghost wall owns "
                                          "(--body floor)");
        if (c.nscbcSink > 0 && r.ogrid)
            return err(2, "--nscbc-sink", "is a Cartesian face band; the "
                                          "body-fitted O-grid has no such "
                                          "faces");
    }
    else if (r.given.contains("--nscbc-strip") && c.nscbcStrip > 0)
        return err(2, "--nscbc-strip", "requires --far nscbc");
    if (c.far != FarField::Nscbc && r.given.contains("--nscbc-sink") &&
        c.nscbcSink > 0)
        return err(2, "--nscbc-sink", "requires --far nscbc");
    if (r.given.contains("--nscbc-sink-amp") && c.nscbcSink == 0)
        return err(2, "--nscbc-sink-amp", "sets the strength of the "
                                          "--nscbc-sink band; pass a width");
    if (c.far != FarField::Nscbc && r.given.contains("--nscbc-etain"))
        return err(2, "--nscbc-etain", "requires --far nscbc");
    if (r.given.contains("--nscbc-etain-shear"))
    {
        if (c.far != FarField::Nscbc)
            return err(2, "--nscbc-etain-shear", "requires --far nscbc");
        if (!finitePos(c.nscbcEtaInShear))
            return err(2, "--nscbc-etain-shear", "must be > 0");
    }
    if (r.given.contains("--nscbc-etalat"))
    {
        if (c.far != FarField::Nscbc || !c.open)
            return err(2, "--nscbc-etalat", "is the OPEN lateral faces' relax "
                                            "rate; needs --far nscbc --open "
                                            "(walled laterals never read it)");
        if (!finitePos(c.nscbcEtaLat))
            return err(2, "--nscbc-etalat", "must be > 0");
        if (c.nscbcEtaLat > 0.5)
            c.warnings.emplace_back(
                "--nscbc-etalat > 0.5: the lateral relax writes vorticity onto "
                "the face in proportion to this rate (a stiff clamp is a wall "
                "in disguise; measured 2.16x the incident scale at 1.0)");
    }
    // The derived outflow coefficients. `adaptive` is formed inside the NSCBC
    // kernel from quantities it already has, so it constrains nothing beyond
    // needing the closure at all. `modal` owns ONE plane -- the uniform x-hi
    // face of a laterally walled box -- and each rejection below is a case
    // where that plane is not what the transform assumes. Rejected, never
    // silently ignored.
    c.nscbcAbc = r.nscbcAbc;
    if (c.nscbcAbc != NscbcAbc::YooIm && c.far != FarField::Nscbc)
        return err(2, "--nscbc-abc", "is the NSCBC outflow's transverse "
                                     "coefficient; use --far nscbc");
    if (c.nscbcAbc == NscbcAbc::Modal)
    {
        if (c.open)
            return err(2, "--nscbc-abc modal",
                       "needs walled lateral faces: with --open the far-field "
                       "problem is not separable in the transverse directions, "
                       "so the modal basis it transforms in does not exist "
                       "(--nscbc-abc adaptive works there)");
        if (remapsMesh(r))
            return err(2, "--nscbc-abc modal",
                       "is derived on the UNIFORM outflow plane; --stretch and "
                       "--focal remap it (--nscbc-abc adaptive works there)");
        if (r.ogrid)
            return err(2, "--nscbc-abc modal",
                       "is a Cartesian x-face closure; the body-fitted O-grid "
                       "has no such face");
        if (c.bodyKind == BodyKind::Floor)
            return err(2, "--nscbc-abc modal",
                       "needs a whole outflow plane; --body floor makes its "
                       "lower rows solid");
        if (c.grid.nz > 1)
            c.warnings.emplace_back(
                "--nscbc-abc modal transforms the outflow plane along y only; "
                "z is periodic, not mirror-symmetric, so a spanwise-oblique "
                "wave still sees a per-z-line coefficient");
    }
    c.farPot = r.farPot;
    if (c.farPot && c.bodyKind == BodyKind::None)
        return err(2, "--pot", "models the body's wake source and displacement "
                               "doublet; meaningless with --body none");
    if (c.farPot && c.far != FarField::Nscbc
        && !(c.far == FarField::Tunnel && c.open))
        return err(2, "--pot", "requires --far nscbc (compressible) or "
                               "--far tunnel --open (incompressible) -- the "
                               "correction retargets an open far field");
    return std::nullopt;
}

// Stage: sponge/fringe band widths.
std::optional<ConfigError> flowBands(const RawConfig& r, Case& c)
{
    // Sponge / fringe band width. clamp(0.1n, 8, n/2-1) is UB when n < 18.
    // Band width is measured along x for BOTH physics. (This was once a
    // `comp ? nx : nx` ternary -- a physics distinction someone intended and
    // abandoned. Restoring one needs a measurement, not a guess.)
    const int bandBase = c.grid.nx;
    if ((c.far == FarField::Fringe || c.far == FarField::Sponge
         || c.far == FarField::Nscbc)
        && bandBase < 18)
        return err(2, "--n", "sponge/fringe far fields need at least 18 cells along x");
    c.spongeCells = r.spongeCells > 0
                        ? r.spongeCells
                        : std::clamp(int(0.1 * bandBase), 8, bandBase / 2 - 1);
    if (c.spongeCells < 1 || c.spongeCells >= bandBase / 2)
        return err(2, "--sponge-cells", "must be >= 1 and < nx/2");

    // Auto: band z only when the box is thick enough to hold the band. A thin
    // quasi-2D run gets x,y only -- a z band there would sponge the whole wake.
    const bool zBandFits = c.grid.nz >= 2 * c.spongeCells + 2;
    c.fringeAxes = r.fringeAxes != 0 ? r.fringeAxes : (zBandFits ? 3 : 2);
    if (c.fringeAxes != 2 && c.fringeAxes != 3)
        return err(2, "--fringe-axes", "must be 2 or 3");
    if (c.far == FarField::Fringe && c.fringeAxes == 3 && !zBandFits)
        return err(2, "--fringe-axes", "a z fringe band needs nz >= 2*cells + 2; "
                                       "pass --fringe-axes 2 for a z-invariant run");
    return std::nullopt;
}

// Stage: the tanh y-stretch.
std::optional<ConfigError> flowStretch(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Curvilinear y-stretch + wall type. Scope is deliberate: the interior
    // scheme (curv finite-volume path) is general and NSCBC evaluates the
    // PointRemap on device (physical distances in its one-sided and
    // transverse terms), but the immersed ghost wall still assumes uniform
    // spacing and the curvilinear solver has no LES -- each combination is
    // rejected, not silently degraded. Sponge stays explicit per the ground
    // rule: the validator never swaps --far for the user.
    c.stretchBeta = r.stretch;
    c.noslipY = r.noslip;
    if (!(std::isfinite(c.stretchBeta) && c.stretchBeta >= 0 && c.stretchBeta <= 4))
        return err(2, "--stretch", "must be in [0,4] (tanh clustering strength; "
                                   "4 already compresses the wall cell ~190x)");
    if (c.stretchBeta > 0)
    {
        if (!comp)
            return err(2, "--stretch", "the curvilinear solver is compressible-"
                                       "only; --stretch needs --physics "
                                       "compressible");
        if (c.bodyKind != BodyKind::None)
            return err(2, "--stretch", "the ghost wall samples the body SDF at "
                                       "uniform cell positions; stretched runs "
                                       "need --body none for now");
        if (c.open)
            return err(2, "--open", "--stretch clusters cells toward the closed "
                                    "y walls; open lateral faces defeat it");
        if (c.far != FarField::Nscbc && c.far != FarField::Sponge)
            return err(2, "--far", "stretched-y runs need --far nscbc (the "
                                   "characteristic terms evaluate the PointRemap) "
                                   "or an explicit --far sponge");
        // Resolved wall resolution: first INTERIOR cell (row 1; rows 0 and
        // ny-1 are the wall-mirror ghost rows) and max growth ratio. These
        // set the acoustic dt, so --dry-run must surface them.
        const TanhStretchY R{double(c.grid.ny) * c.grid.hy, c.stretchBeta};
        const double hy = c.grid.hy;
        c.stretchH1 = R.mapY(2 * hy) - R.mapY(hy);
        c.stretchGrowth = std::max(
            c.stretchGrowth,
            maxAdjacentGrowth([&](double y) { return R.mapY(y); }, hy, 1,
                              c.grid.ny, R.mapY(hy) - R.mapY(0.0)));
    }
    return std::nullopt;
}

// Stage: the --focal plateau.
std::optional<ConfigError> flowFocal(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Focal clustering: a fine plateau box (typically the body plus its wake)
    // with a coarse far field, one independent Focal1D per axis. Same scope
    // as --stretch (compressible curvilinear, NSCBC or explicit sponge; LES is
    // allowed, the SGS operators read the mesh metrics and the per-cell
    // Delta), but bodies ARE supported here: the ghost wall classifies
    // against the mapped cell centres. The blend support must stay clear of
    // the domain faces, so the map is the exact identity there and the
    // periodic wrap stays C^2.
    if (r.given.contains("--focal"))
    {
        if (!comp)
            return err(2, "--focal", "the curvilinear solver is compressible-"
                                     "only; --focal needs --physics "
                                     "compressible");
        if (c.stretchBeta > 0)
            return err(2, "--focal", "--stretch and --focal are different "
                                     "curvilinear meshes; pick one");
        // Bodies ARE supported (the ghost wall classifies against the mapped
        // cell centres), but only the analytic ones: their AABB follows from
        // the placement alone, and validate() may not touch the filesystem to
        // read a mesh.
        // A ROTOR is the exception, and for a reason that is not a loophole:
        // its body sweeps a shaft-aligned cylinder of radius D/2 and half-
        // height --rotor-span * D, so the extent that matters is the SWEPT one
        // and that follows from the placement and two flags. --rotor-span is
        // capped at 0.5 below, so the placement cube already contains it and
        // the plateau derivation needs no change at all.
        const bool sweptBox = r.rotorMach != 0;
        if ((c.bodyKind == BodyKind::Mesh || c.bodyKind == BodyKind::Scene)
            && !sweptBox)
            return err(2, "--focal", "a mesh/scene body's bounding box is only "
                                     "known after the file is loaded, and "
                                     "validation touches no filesystem; give "
                                     "the plateau explicitly for now (a rotor "
                                     "is exempt: its swept extent follows from "
                                     "--dr and --rotor-span)");
        if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
            return err(2, "--focal", "the wall-mounted bodies span the full "
                                     "domain width; there is no compact "
                                     "plateau to cluster on");
        if (c.cutFlux || c.penalize > 0)
            return err(2, "--focal", "apertures and the Brinkman band are "
                                     "uniform-mesh geometry; the curvilinear "
                                     "path has none");
        if (c.open)
            return err(2, "--open", "--focal needs the closed lateral faces "
                                    "its blend margin is measured against");
        if (c.far != FarField::Nscbc && c.far != FarField::Sponge)
            return err(2, "--far", "focal runs need --far nscbc (the "
                                   "characteristic terms evaluate the "
                                   "PointRemap) or an explicit --far sponge");

        c.focalRatio = r.focalRatio;
        if (!(std::isfinite(c.focalRatio) && c.focalRatio > 1
              && c.focalRatio <= 16))
            return err(2, "--focal-ratio", "must be in (1,16] (coarse/fine "
                                           "cell-size ratio; the fine spacing "
                                           "sets the acoustic dt)");
        const double blendFrac = r.focalBlend;
        if (!(finitePos(blendFrac) && blendFrac <= 2))
            return err(2, "--focal-blend", "must be in (0,2] (blend width per "
                                           "side as a fraction of the plateau)");

        // The body's axis-aligned bounds, when there is a body: the
        // containment rule below keeps the ghost wall's neighbourhood inside
        // the plateau, so it always sees a locally UNIFORM (anisotropic) grid
        // -- the geometry the Cartesian construction was validated on. The
        // cylinder is z-invariant, so its z extent is the whole box.
        const char* axName[3] = {"x", "y", "z"};
        const bool hasBody = c.bodyKind != BodyKind::None;
        const bool zSpanning = c.bodyKind == BodyKind::Cylinder;
        const double bLo[3] = {c.placement.cx - 0.5 * c.placement.D,
                               c.placement.cy - 0.5 * c.placement.D,
                               c.placement.cz - 0.5 * c.placement.D};
        const double bHi[3] = {c.placement.cx + 0.5 * c.placement.D,
                               c.placement.cy + 0.5 * c.placement.D,
                               c.placement.cz + 0.5 * c.placement.D};

        // Every per-axis rule below, auto and explicit alike, is evaluated on
        // the map actually built from the PHYSICAL box (Focal1D::
        // fromPhysical): the body is placed in physical coordinates, so the
        // plateau, the guard and the containment test must be too. The face
        // margin is a CELL count and is therefore tested in reference space,
        // where cells are uniform.
        const int nAx[3] = {c.grid.nx, c.grid.ny, c.grid.nz};
        const double hAx[3] = {c.grid.hx, c.grid.hy, c.grid.hz};
        auto axisMap = [&](int a, double lo, double hi) {
            return Focal1D::fromPhysical(nAx[a] * hAx[a], lo, hi,
                                         blendFrac * (hi - lo), c.focalRatio);
        };
        // Identity margin at both faces: the blend support must end at least
        // 4 cells short of each domain face (x also clears the NSCBC rows and
        // any absorbing strip), so the far grid is exactly uniform where the
        // boundary treatment reads it.
        auto marginCells = [&](int a) {
            return a == 0 ? std::max(4, c.nscbcStrip + 2) : 4;
        };
        auto clearsFaces = [&](int a, const Focal1D& f) {
            return f.active && f.clears(marginCells(a) * hAx[a]);
        };
        auto fineOf = [&](int a, const Focal1D& f) {
            return f.c * f.r * hAx[a];
        };
        auto holdsBody = [&](int a, double lo, double hi, double guard) {
            return lo <= bLo[a] - guard && hi >= bHi[a] + guard;
        };

        double box[6];
        if (r.focal == "auto")
        {
            if (!hasBody)
                return err(2, "--focal", "auto needs a body; --body none has "
                                         "no AABB to focus on");
            if (!(std::isfinite(r.focalWake) && r.focalWake >= 0))
                return err(2, "--focal-wake", "must be finite and >= 0");
            if (!(std::isfinite(r.focalPad) && r.focalPad >= 0))
                return err(2, "--focal-pad", "must be finite and >= 0");
            const double D = c.placement.D, pad = r.focalPad * D;
            for (int a = 0; a < 3; ++a)
            {
                // Axes the body does not localize stay uniform.
                if (a == 2 && (zSpanning || c.grid.nz == 1))
                {
                    box[4] = box[5] = 0.0;
                    continue;
                }
                // Smallest plateau: the body plus FOUR fine cells of that
                // plateau's own spacing -- one more than the containment rule
                // below asks, so rounding cannot reject auto's own box. The
                // fine spacing grows with the box (through c), hence the
                // fixed point; it is monotone from the c = 1 lower bound and
                // contracts by ~8 h/L per step.
                double guard = 4.0 * hAx[a] / c.focalRatio;
                for (int it = 0; it < 64; ++it)
                {
                    const double g2 = 4.0 * fineOf(a, axisMap(a, bLo[a] - guard,
                                                              bHi[a] + guard));
                    if (g2 == guard) break;
                    guard = g2;
                }
                const double need[2] = {bLo[a] - guard, bHi[a] + guard};
                // Desired plateau, never smaller than the smallest one. It
                // shrinks toward `need` until it fits; the resolved box is what
                // --dry-run prints, so the shrink is never silent.
                const double want[2] = {
                    std::min(need[0], bLo[a] - pad),
                    std::max(need[1],
                             bHi[a] + pad + (a == 0 ? r.focalWake * D : 0.0))};
                auto fits = [&](double lo, double hi) {
                    const Focal1D f = axisMap(a, lo, hi);
                    return clearsFaces(a, f)
                           && holdsBody(a, lo, hi, 3.0 * fineOf(a, f));
                };
                if (!fits(need[0], need[1]))
                    return err(2, "--focal", std::string(axName[a])
                                   + " has no plateau that both contains the "
                                     "body and clears the domain faces: the "
                                     "fine cells cost --focal-ratio x the "
                                     "plateau width out of the grid, which "
                                     "coarsens the far cells until the face "
                                     "margin no longer fits. Raise --n, or "
                                     "lower --focal-ratio or --focal-blend");
                double t = 1.0;
                if (!fits(want[0], want[1]))
                {
                    double lo = 0.0, hi = 1.0; // largest fitting t
                    for (int it = 0; it < 60; ++it)
                    {
                        const double mid = 0.5 * (lo + hi);
                        const double a0 = need[0] + mid * (want[0] - need[0]);
                        const double a1 = need[1] + mid * (want[1] - need[1]);
                        (fits(a0, a1) ? lo : hi) = mid;
                    }
                    t = lo;
                }
                box[2 * a] = need[0] + t * (want[0] - need[0]);
                box[2 * a + 1] = need[1] + t * (want[1] - need[1]);
            }
        }
        // "x0,x1,y0,y1,z0,z1"; a degenerate pair leaves that axis uniform.
        else {
            std::string_view s(r.focal);
            for (int q = 0; q < 6; ++q)
            {
                const std::size_t comma = s.find(',');
                if (!parseDouble(s.substr(0, comma), box[q]))
                    return err(2, "--focal", "expects six numbers "
                                             "x0,x1,y0,y1,z0,z1 (world units) "
                                             "or 'auto'");
                if (comma == std::string_view::npos)
                {
                    if (q != 5)
                        return err(2, "--focal", "expects six comma-separated "
                                                 "numbers x0,x1,y0,y1,z0,z1");
                    s = {};
                }
                else
                    s = s.substr(comma + 1);
            }
            if (!s.empty())
                return err(2, "--focal", "expects exactly six numbers "
                                         "x0,x1,y0,y1,z0,z1");
        }

        for (int a = 0; a < 3; ++a)
        {
            const double lo = box[2 * a], hi = box[2 * a + 1];
            if (hi <= lo)
            {
                // Degenerate pair = leave this axis uniform.
                if (hi < lo)
                    return err(2, "--focal", std::string(axName[a])
                                                 + " plateau has hi < lo");
                c.focalHFine[a] = c.focalHCoarse[a] = hAx[a];
                continue;
            }
            if (a == 2 && c.grid.nz == 1)
                return err(2, "--focal", "nz == 1 IS the exact 2D restriction; "
                                         "a z plateau there is meaningless -- "
                                         "pass z0,z1 equal");
            const Focal1D f = axisMap(a, lo, hi);
            if (!clearsFaces(a, f))
                return err(2, "--focal",
                           std::string(axName[a]) + " plateau plus its blend "
                           "must clear both domain faces by " +
                           std::to_string(marginCells(a)) + " (coarse) cells "
                           "(the map must be the exact identity there). The "
                           "fine cells cost --focal-ratio x the plateau width "
                           "out of the grid, which coarsens the far cells; "
                           "shrink the box, --focal-blend or --focal-ratio, "
                           "or raise --n");
            const double fine = fineOf(a, f);
            if (hi - lo < 4 * fine)
                return err(2, "--focal", std::string(axName[a])
                                             + " plateau spans under 4 fine "
                                               "cells; widen it");
            // Containment: the body plus the ghost wall's 3-cell mirror
            // neighbourhood must sit inside the PLATEAU, never in the blend.
            // The wall stencil is then built on locally uniform cells, which
            // is the geometry it was validated on.
            if (hasBody && !(a == 2 && zSpanning))
            {
                const double guard = 3.0 * fine;
                if (!holdsBody(a, lo, hi, guard))
                    return err(2, "--focal",
                               std::string(axName[a]) + " plateau must contain "
                               "the body plus a 3-fine-cell guard band "
                               "([" + std::to_string(bLo[a] - guard) + "," +
                               std::to_string(bHi[a] + guard) + "]); the ghost "
                               "wall's mirror stencil must not straddle the "
                               "coarse/fine blend");
            }
            // The case carries the PHYSICAL plateau and blend; focalRemap()
            // turns them into this same map for every consumer.
            c.focalActive = true;
            c.focalLo[a] = lo;
            c.focalHi[a] = hi;
            c.focalW[a] = blendFrac * (hi - lo);
            c.focalHFine[a] = fine;
            c.focalHCoarse[a] = f.c * hAx[a];
            // Discrete adjacent growth: the KG flux carries no dissipation,
            // so a fast coarse->fine transition is a dispersive reflector.
            c.focalGrowth = std::max(
                c.focalGrowth,
                maxAdjacentGrowth([&](double X) { return f.map(X); }, hAx[a], 1,
                                  nAx[a], f.map(hAx[a]) - f.map(0.0)));
        }
        if (!c.focalActive)
            return err(2, "--focal", "every axis pair is degenerate; the mesh "
                                     "would be uniform -- drop --focal");
        if (c.focalGrowth > 1.25)
            return err(2, "--focal-blend", "max adjacent cell growth " +
                       std::to_string(c.focalGrowth) + " exceeds 1.25; the "
                       "central flux has no dissipation and reflects off a "
                       "sharp transition. Widen --focal-blend or lower "
                       "--focal-ratio");
        c.focalDtFactor = hFine(c) / hMin(c.grid);
    }
    else
    {
        if (auto e = rejectGiven(r,
                                 {"--focal-ratio", "--focal-blend",
                                  "--focal-wake", "--focal-pad"},
                                 "requires --focal");
            e)
            return e;
    }
    // Honest labelling: the boundary layer scales as D/sqrt(Re), and it is
    // resolved by the spacing AT the body -- the plateau's under --focal, which
    // the containment rule above guarantees the body sits on. Hence here, once
    // the mesh is known, and not with the LES label.
    // On the O-grid the spacing at the wall is the CLUSTERED radial
    // distribution's, not the nominal hy: count the cells D/sqrt(Re) spans
    // through it (exact for a cylinder, the nominal circle otherwise).
    const double hBody = c.focalActive ? hFine(c) : c.h;
    const double blLen = c.placement.D / std::sqrt(c.re);
    const double blCells =
        c.ogrid ? ogridRadialCells(blLen, c.grid.ny, 0.5 * c.placement.D,
                                   c.ogridRFar, c.ogridBeta)
                : blLen / hBody;
    if (blCells < 3.0 && !c.lesLabel && c.bodyKind != BodyKind::None)
    {
        c.warnings.push_back(
            std::format("boundary layer is {:.1f} cells (D/sqrt(Re)/h); this "
                        "is under-resolved for DNS", blCells));
    }
    return std::nullopt;
}

// Stage: the spinning body.
std::optional<ConfigError> flowRotor(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // --- rotor: the body spins ------------------------------------------------
    //
    // rotorMach is the ONE switch. At 0 nothing below runs, c.rotor stays
    // inactive, and every geometry/wall/tally path downstream takes the static
    // arithmetic it always took -- that is what keeps the measured cd/cl
    // anchors valid.
    {
        if (r.rotorMach == 0)
        {
            if (auto e = rejectGiven(r,
                                     {"--rotor-axis", "--rotor-bins",
                                      "--rotor-span", "--blade-thicken",
                                      "--rotor-frame"},
                                     "configures a spinning body; give "
                                     "--rotor-mach to have one");
                e)
                return e;
        }
        else
        {
            if (!comp)
                return err(2, "--rotor-mach", "the moving wall is the "
                               "compressible ghost wall; --rotor-mach needs "
                               "--physics compressible");
            // The O-grid takes the frozen-rotor frame only: the wall is a
            // mesh face row, so MOVING geometry would spin the mesh itself.
            // --rotor-frame mrf keeps the mesh static (rotating-frame
            // equations + a wall moving in its own plane) and is exact for a
            // circular section.
            if (c.ogrid && r.rotorFrame != 1)
                return err(2, "--rotor-mach", "an O-grid IS the body surface: "
                               "spinning the body would spin the mesh. "
                               "--rotor-frame mrf solves the rotating frame "
                               "on the static O-grid instead; the moving-"
                               "geometry rotor needs the immersed box");
            if (c.ogrid && r.rotorAxis != 2)
                return err(2, "--rotor-axis", "the O-grid is z-extruded, so "
                               "only the z shaft keeps the wall motion in the "
                               "mesh plane");
            if (c.ogrid && r.given.contains("--blade-thicken"))
                return err(2, "--blade-thicken", "exists for the immersed "
                               "ghost wall's mirror band; the O-grid wall is "
                               "a mesh face row and needs no dilation");
            if (c.ogrid && r.given.contains("--rotor-span"))
                return err(2, "--rotor-span", "sizes the swept box of an "
                               "immersed rotor; the O-grid's extent is the "
                               "mesh itself");
            if (c.bodyKind == BodyKind::None)
                return err(2, "--rotor-mach", "there is no body to spin");
            if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
                return err(2, "--rotor-mach", "the wall-mounted bodies are "
                               "sealed through the domain floor; they have no "
                               "shaft and cannot turn");
            if (c.cutFlux || c.penalize > 0)
                return err(2, "--rotor-mach", "apertures and the Brinkman band "
                               "are built once from a static body; the moving "
                               "wall is the ghost wall alone");
            if (r.seq > 1)
                return err(2, "--rotor-mach", "a spinning body has no per-"
                               "level restart; drop --seq");
            // The frozen rotor (--rotor-frame mrf) HAS a fixed point -- the
            // steady-in-the-blade-frame flow -- so it takes --steady
            // (flowRans checks the rest); moving geometry does not.
            if (r.steady > 0 && r.rotorFrame != 1)
                return err(2, "--steady", "moving geometry has no steady "
                               "state; --rotor-frame mrf is the frozen rotor");
            if (!finitePos(r.rotorMach))
                return err(2, "--rotor-mach", "must be finite and > 0 (the TIP "
                                              "Mach number)");
            // The KG central flux carries no shock capturing, so the FASTEST
            // material point in the run has to stay subsonic -- that is the
            // advancing tip, not the freestream.
            if (r.rotorMach + c.mach >= 1.0)
                return err(2, "--rotor-mach", "tip Mach + freestream Mach must "
                               "stay under 1: the advancing tip is the fastest "
                               "point in the flow and the flux has no shock "
                               "capturing");
            if (!(finitePos(r.rotorSpan) && r.rotorSpan <= 0.5))
                return err(2, "--rotor-span", "blade half-extent along the "
                               "shaft in diameters, in (0, 0.5]: beyond 0.5 the "
                               "swept volume leaves the placement cube that "
                               "bounds every extent check");
            if (!(std::isfinite(r.bladeThicken) && r.bladeThicken >= 0))
                return err(2, "--blade-thicken", "must be finite and >= 0 "
                                                 "(fine cells of dilation)");
            // A rotor's mesh is placed and sized about its SHAFT, and its own
            // origin IS that shaft (casegeom). Baking an attitude into it would
            // tilt the model out of the axis it spins about, so the two
            // deprecated angles are rejected outright rather than silently
            // producing a wobbling body.
            if (auto e = rejectGiven(
                    r, {"--pitch", "--yaw"},
                    "tilts the body out of the shaft it spins about; a rotor's "
                    "mesh is placed about its own origin. Bake attitude into "
                    "the mesh, or drop --rotor-mach");
                e)
                return e;
            if (c.grid.nz == 1 && r.rotorAxis != 2)
                return err(2, "--rotor-axis", "nz == 1 is the exact 2D "
                               "restriction: only a z shaft keeps the motion in "
                               "the plane. An x or y shaft sweeps the blade out "
                               "of it, which 2D cannot represent");

            // Fine spacing: the plateau's under --focal, the grid's otherwise.
            // It sets both the natural bin count and what a "cell" of
            // --blade-thicken means, so the two always agree.
            const double hFine = dns::hFine(c);

            c.rotor.axis = r.rotorAxis == 0   ? RotorAxis::X
                           : r.rotorAxis == 1 ? RotorAxis::Y
                                              : RotorAxis::Z;
            c.rotor.shaft = {c.placement.cx, c.placement.cy, c.placement.cz};
            c.rotor.radius = 0.5 * c.placement.D;
            // c = uInf / mach: pInf is chosen as rho uInf^2/(gamma M^2), so
            // the freestream sound speed is exactly that. One definition, and
            // it must not be restated anywhere else.
            c.rotor.omega = rotorOmegaFromTipMach(r.rotorMach, c.uInf / c.mach,
                                                  c.rotor.radius);
            c.rotor.thicken = r.bladeThicken * hFine;

            // --rotor-frame mrf: static geometry, rotating-frame zone. The
            // azimuth table does not exist, so sizing it is a contradiction
            // and is rejected by name rather than ignored.
            c.rotorMrf = r.rotorFrame == 1;
            if (c.rotorMrf && r.given.contains("--rotor-bins"))
                return err(2, "--rotor-bins", "--rotor-frame mrf freezes the "
                               "geometry at azimuth 0; there is no azimuth "
                               "table to size");

            const int natural = rotorNaturalBins(c.rotor, hFine);
            if (c.rotorMrf)
                // active() needs bins > 0; one frozen azimuth is exactly what
                // MRF is.
                c.rotor.bins = 1;
            else if (r.given.contains("--rotor-bins"))
            {
                if (r.rotorBins < 8)
                    return err(2, "--rotor-bins", "under 8 bins the body jumps "
                                   "by more than a blade width per step change; "
                                   "0 selects the natural count");
                if (r.rotorBins > 1 << 16)
                    return err(2, "--rotor-bins", "over 65536 bins; the table "
                                   "grows linearly and buys nothing past the "
                                   "natural count");
                c.rotor.bins = r.rotorBins;
                if (r.rotorBins > natural)
                    c.warnings.push_back(std::format(
                        "--rotor-bins {} exceeds the natural count {} "
                        "(2 pi R / h_fine): occupancy cannot change faster "
                        "than that even at the tip, so the extra bins are "
                        "duplicate geometry",
                        r.rotorBins, natural));
                else if (2 * r.rotorBins < natural)
                    c.warnings.push_back(std::format(
                        "--rotor-bins {} is under half the natural count {}: "
                        "the tip crosses more than two cells per bin, so the "
                        "motion is visibly quantized and injects an impulse at "
                        "bins*rev/s",
                        r.rotorBins, natural));
            }
            else
                c.rotor.bins = natural;

            // The dilated body must still be a body in this box: a dilation
            // comparable to the radius turns a blade into a disc.
            if (c.rotor.thicken > 0.25 * c.rotor.radius)
                return err(2, "--blade-thicken", "dilation exceeds a quarter of "
                               "the tip radius; that is a different body, not a "
                               "resolvable one");

            c.rotorTipMach = r.rotorMach;
            c.rotorSpanD = r.rotorSpan;
            c.bladeThickenCells = r.bladeThicken;

            // The blade has to be thick enough for the ghost wall's mirror
            // band on BOTH sides. We cannot measure the mesh here (validation
            // touches no filesystem), so this is the advisory; the run reports
            // the measured deep-solid count, which is the real check. The
            // O-grid wall is a mesh face row -- no mirror band exists there.
            if (!c.ogrid)
            c.warnings.push_back(std::format(
                "rotor: the ghost wall needs a mirror band >= 2 cells thick "
                "per side, so the blade must be >= 4 fine cells thick "
                "(h_fine {:.4g}, D = {:.1f} fine cells). The run reports the "
                "measured deep-solid cell count; 0 means the two blade "
                "surfaces read through each other",
                hFine, c.placement.D / hFine));
        }
    }
    return std::nullopt;
}

// Stage: the no-slip y walls.
std::optional<ConfigError> flowWalls(const RawConfig&, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    if (c.noslipY)
    {
        if (!comp)
            return err(2, "--noslip", "the y mirror walls belong to the "
                                      "compressible far field; --noslip needs "
                                      "--physics compressible");
        if (c.open)
            return err(2, "--noslip", "--open removes the y walls; there is "
                                      "nothing to make no-slip");
        if (c.far != FarField::Sponge && c.far != FarField::Nscbc)
            return err(2, "--noslip", "requires a far field with closed y walls "
                                      "(--far sponge or --far nscbc)");
    }
    return std::nullopt;
}

// Stage: quiet start and trip.
std::optional<ConfigError> flowStartup(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Start-up. The raw default (2.0) is the COMPRESSIBLE default; the
    // incompressible ramp is opt-in so existing impulsive-start trajectories
    // and anchors do not silently move.
    c.ramp = comp ? r.ramp : (r.given.contains("--ramp") ? r.ramp : 0.0);
    if (!comp && c.ramp > 0 && (r.seq > 1 || r.steady > 0))
        return err(2, "--ramp", "the quiet start would re-ramp the developed "
                                "field each --seq level / fight --steady; "
                                "use an impulsive start there");
    // The O-grid never applies the kick (flowGrid rejects a given one): the
    // case carries the value the run applies.
    c.kick = c.ogrid ? 0.0 : r.kick;
    if (!(std::isfinite(c.ramp) && c.ramp >= 0)) return err(2, "--ramp", "must be >= 0");
    if (!std::isfinite(c.kick)) return err(2, "--kick", "must be finite");
    return std::nullopt;
}

// Stage: the pressure solve and the dt recompute.
std::optional<ConfigError> flowSolver(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // Solver.
    c.tol = r.tol;
    if (!finitePos(c.tol)) return err(2, "--tol", "must be > 0");
    c.maxIters = r.maxIters;
    if (c.maxIters < 1) return err(2, "--max-iters", "must be >= 1");
    // Default per current app: 16 for the compressible cylinder (a validated
    // perf fix, masterplan §6.7), 1 everywhere else.
    if (auto e = resolveDtRecompute(
            r, c, comp && c.bodyKind == BodyKind::Cylinder ? 16 : 1);
        e)
        return e;
    return std::nullopt;
}

// Stage: the per-body options.
std::optional<ConfigError> flowGeometry(const RawConfig& r, Case& c)
{
    // Geometry options.
    // Which body takes which option is bodyOptions()'s, the table the editor
    // reads too.
    const BodyOptions bo = bodyOptions(c.bodyKind);
    c.aspect = r.aspect;
    if (bo.aspect && !(c.aspect > 1.0))
        return err(2, "--aspect", "must be > 1 (the lens needs w > hh)");
    c.dimples = r.dimples;
    if (c.dimples < 0) return err(2, "--dimples", "must be >= 0");
    if (c.dimples > 0 && !bo.dimples)
        return err(2, "--dimples", "only cylinder and sphere carry dimples");
    if (c.dimples > 0 && c.placement.D < 16 * c.h)
        return err(2, "--dimples", "dimple pits are 5h wide; D must span at least 16 "
                                   "cells or they consume the body");
    c.bumps = r.bumps;
    if (c.bumps < 0) return err(2, "--bumps", "must be >= 0");
    if (c.bumps > 0 && !bo.bumps)
        return err(2, "--bumps", "the surface ripple belongs to --body floor");
    c.bumpH = r.bumpH;
    if (c.bumps > 0)
    {
        if (!(std::isfinite(c.bumpH) && c.bumpH >= 4 && c.bumpH <= 48))
            return err(2, "--bump-h", "must be in [4,48] cells (base = height, "
                                      "63-degree flanks; under 4 the tip is "
                                      "sub-cell, over 48 it is a fence, not "
                                      "roughness)");
        if (c.bumps > int(0.65 * c.n / (2.0 * c.bumpH)))
            return err(2, "--bumps", "spike pitch is under twice the tooth "
                                     "base; the teeth merge -- fewer bumps, "
                                     "shorter --bump-h, or more cells");
        if (c.placement.D < 4 * c.bumpH * c.h)
            return err(2, "--bump-h", "spikes taller than D/4 are a fence "
                                      "field, not surface roughness -- lower "
                                      "--bump-h or --dr");
    }
    c.scale = r.scale;
    if (!(std::isfinite(c.scale) && c.scale >= 0))
        return err(2, "--scale", "must be finite and >= 0 (0 = fit D)");
    c.pitch = r.pitch;
    c.yaw = r.yaw;
    c.robust = r.robust;
    if (!bo.mesh)
        if (auto e = rejectGiven(r, {"--scale", "--pitch", "--yaw", "--robust"},
                                 "applies to mesh bodies only");
            e)
            return e;
    return std::nullopt;
}

// Stage: run length, stationary mode, sequencing, frames, metadata.
std::optional<ConfigError> flowOutput(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    // A compressible --steady run marches in PSEUDO-time: it has no run
    // length (flowRans rejects --ft there) and stops on its integrals.
    c.compSteady = comp && r.steady > 0;
    // Run length + output cadences.
    c.ft = r.ft;
    if (!c.compSteady && !finitePos(c.ft))
        return err(2, "--ft", "must be finite and > 0");
    // One flow-through is L/uInf in both physics (c.L already carries each
    // physics' own box convention). Formerly a `comp ? X : X` ternary.
    c.tEnd = c.compSteady ? 0.0 : c.ft * c.L / c.uInf;

    // Stationary mode. Incompressible: tEnd stops being the run length and
    // becomes the give-up cap -- not converging by then is exit 8, never a
    // silent answer. Compressible: the pseudo-time march (flowRans).
    c.steadyTol = r.steady;
    if (c.steadyTol < 0 || !std::isfinite(c.steadyTol))
        return err(2, "--steady", "must be finite and >= 0 (0 = time-march)");
    if (c.steadyTol > 0 && c.cs > 0)
        return err(2, "--steady",
                   "with --les is meaningless: a filtered field has no "
                   "stationary state to converge to");

    // Grid sequencing. Only the initial guess comes from the coarse level, so
    // the fine-grid fixed point is unchanged; the ladder just skips paying the
    // startup transient at full resolution.
    c.seqLevels = r.seq;
    if (c.seqLevels < 1 || c.seqLevels > 6)
        return err(2, "--seq", "must be in [1,6] (1 = off)");
    if (c.seqLevels > 1)
    {
        if (c.steadyTol <= 0)
            return err(2, "--seq", "requires --steady: without a convergence "
                                   "test there is nothing to advance a level on");
        // The coarsest level still has to resolve the body: same D >= 4h rule.
        const int nCoarse = r.n >> (c.seqLevels - 1);
        if (nCoarse < 16 || double(nCoarse) < 4.0 * c.dr)
            return err(2, "--seq", "coarsest level would not resolve the body "
                                   "(needs n/2^(levels-1) >= max(16, 4*dr)); "
                                   "use fewer levels or a larger --n");
    }
    if (auto e = validateFrameOutputs(r, c); e) return e;
    c.vortMax = r.vortMax;
    if (!(std::isfinite(c.vortMax) && c.vortMax >= 0))
        return err(2, "--vort-max", "must be finite and >= 0 (0 = auto scale)");
    c.vortAsinh = r.vortMap == 1;
    if (c.vortAsinh && !(c.vortMax > 0))
        return err(2, "--vort-map", "asinh needs an explicit full-scale clip; "
                                    "set --vort-max (~ the peak |vorticity| in "
                                    "uInf/D, e.g. sqrt(Re))");
    c.speedDelta = r.speedMap == 1;
    c.tempFrames = r.tempFrames;
    if (c.tempFrames && !comp)
        return err(2, "--temp", "the incompressible solver carries no energy "
                                "equation and no temperature field; use "
                                "--physics compressible");
    c.vortSigned = r.vortSigned;
    if (c.vortSigned && !comp)
        return err(2, "--vort-signed", "incompressible frames are already "
                                       "signed z-vorticity; the flag selects "
                                       "it for 3D compressible runs");

    // Pseudo-time has no frame clock: a compressible steady run defaults to
    // the stats alone (flowRans rejects time-keyed kinds that are asked for).
    if (auto e = c.compSteady ? resolveEmit(c, {Emit::Stats})
                              : resolveEmit(c, {Emit::Stats, Emit::Bin});
        e)
        return e;
    // PNG at nz > 1 renders the centre z-plane (same machinery the channel
    // already uses); the old flow-only nz == 1 restriction was conservative,
    // not load-bearing.

    // NOTE: qRef is deliberately NOT resolved here. It is 0.5 rho uInf^2 A
    // with A the frontal area MEASURED from the cut apertures, and validate()
    // touches no filesystem and builds no geometry, so it cannot know A. The
    // field that used to live here held 0.5 uInf^2 D span -- a placeholder
    // that --dry-run printed as if it were the number the run would divide by.
    // The real value comes from dns/geometry.hpp's referenceQ*(), once.

    c.meta = {.D = c.placement.D,
              .re = c.re,
              .mach = comp ? c.mach : 0.0,
              .frameDt = c.fdt,
              .cs = c.cs,
              .les = c.lesLabel,
              .body = c.bodyKind == BodyKind::Mesh    ? c.meshPath
                      : c.bodyKind == BodyKind::Scene ? "scene:" + c.scenePath
                                                      : name(c.bodyKind),
              // The O-grid's far field is its own characteristic outer row;
              // c.far is the unused Cartesian default there.
              .farField = c.ogrid ? std::string("ogrid") : name(c.far),
              .physics = name(c.physics),
              .kick = comp ? c.kick : 0.0,
              .sgs = c.lesLabel ? name(c.sgsModel) : "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return std::nullopt;
}

// The freestream turbulence as the describe line and RunMeta print it: the
// flags as given, and omega in U/D. Pre: c.rans != None, ransPhiInf resolved.
std::string ransInflowText(const Case& c)
{
    if (!ransSst(c.rans))
        return std::format("nu~inf/nu = {:g} (TMR)", c.ransNutInf);
    const double tu = std::sqrt(2.0 * c.ransKInf / 3.0);
    return std::format("k = {:.3g} U^2 (Tu {:.3g}%), mu_t/mu = {:g} -> "
                       "omega = {:.3g} U/D",
                       c.ransKInf, 100.0 * tu, c.ransMutInf,
                       c.ransPhiInf[1] * c.placement.D / c.uInf);
}

// The first-cell wall distance the y+ estimate uses: the O-grid's h1, or half
// the fine spacing of an immersed mesh (the mean centre offset of a
// staircase wall).
double ransFirstCellDistance(const Case& c)
{
    return c.ogrid ? c.ogridH1 : 0.5 * hFine(c);
}

// The first-cell y+ estimate the validator can quote without a run: the
// bluff-body PEAK friction velocity u_tau ~ 0.38 Re^-0.1 U, calibrated on the
// MEASURED 0.137 U of the OpenFOAM sphere at Re 1e4 (masterplan_flowrans
// section 3.6). Pre: h1, D, re > 0.
double ransYplusEstimate(double h1, double D, double re)
{
    return h1 / D * re * 0.38 * std::pow(re, -0.1);
}

// Stage: RANS and the compressible steady march (docs/rans.md, "dns flow").
// LAST, so every geometry/physics error wins first; every rule rejects by
// name, never ignores. RANS off and no compressible --steady leave the case
// exactly as the earlier stages built it.
std::optional<ConfigError> flowRans(const RawConfig& r, Case& c)
{
    const bool comp = c.physics == Physics::Compressible;
    c.rans = RansModel(r.rans);
    const bool rans = c.rans != RansModel::None;
    const bool sa = c.rans == RansModel::SaNeg, sst = ransSst(c.rans);
    // --thin-seal: the compressible immersed ghost wall's sealed faces
    // (CompGhostTables::sealed). Every path it is not built for rejects it.
    c.thinSeal = r.thinSeal;
    if (c.thinSeal)
    {
        if (!comp)
            return err(2, "--thin-seal", "seals faces of the compressible "
                                         "immersed ghost wall; add --physics "
                                         "compressible");
        if (c.ogrid)
            return err(2, "--thin-seal", "the O-grid wall is a mesh face row: "
                                         "it has no staircase to leak");
        if (c.bodyKind == BodyKind::None)
            return err(2, "--thin-seal", "--body none has no wall to seal");
        if (rans)
            return err(2, "--thin-seal", "the RANS scalar wall images (ghost "
                                         "k, nu~, omega, mu_t clamp) know no "
                                         "sealed faces");
        if (c.rotor.active() && !c.rotorMrf)
            return err(2, "--thin-seal", "the moving rotor's azimuth tables "
                                         "are not sealed; use --rotor-frame "
                                         "mrf");
    }
    if (c.ogridRevolved && c.ogridNphi == 0 && c.cs > 0)
        return err(2, "--les", "an axisymmetric wedge has no resolved 3D "
                               "turbulence to filter");
    if (c.ogridRevolved && c.ogridNphi > 0 && c.cs > 0)
        return err(2, "--les", "on the full-revolution O-grid is not built: "
                               "the filter width of the axis cells is "
                               "unverified");
    if (r.given.contains("--rans-nutinf") && !sa)
        return err(2, "--rans-nutinf", "is the SA freestream nu~/nu; it needs "
                                       "--rans sa-neg");
    if (auto e = rejectGiven(r, {"--rans-kinf", "--rans-mutinf"},
                             "is an SST freestream value; it needs --rans "
                             "sst-2003m|sst-vm");
        e && !sst)
        return e;
    if (c.emit.contains(Emit::Surface) && !c.ogrid && !comp)
        return err(2, "--emit", "surface is the per-face wall tally of the "
                                "body-fitted O-grid (--ogrid) or of the "
                                "compressible immersed ghost wall; the "
                                "incompressible cut-cell wall has none -- add "
                                "--physics compressible");
    if (c.emit.contains(Emit::Surface) && c.bodyKind == BodyKind::None)
        return err(2, "--emit", "surface tallies the body's wall faces; "
                                "--body none has none");
    if (c.emit.contains(Emit::Mesh) && !c.ogrid)
        return err(2, "--emit", "mesh exports the body-fitted O-grid; it "
                                "needs --ogrid");
    if (rans)
    {
        if (!comp)
            return err(2, "--rans", "there is no RANS on the incompressible "
                                    "projection solver; add --physics "
                                    "compressible");
        if (c.cs > 0 || r.given.contains("--sgs"))
            return err(2, "--rans", "and --les/--sgs would put two eddy "
                                    "viscosities in one slot; drop --les/--sgs");
        if (c.rotor.active())
            return err(2, "--rans", "with a rotor is not built: moving geometry "
                                    "changes d per azimuth bin, and MRF + RANS "
                                    "needs a rotation-correction decision");
        if (c.bodyKind == BodyKind::Scene)
            return err(2, "--rans", "with --scene is not built: the scene model "
                                    "block is dns|les");
        if (c.penalize > 0)
            return err(2, "--rans", "with --penalize: the Brinkman band is a "
                                    "state edit the closure has no image for");
        if (c.farPot)
            return err(2, "--rans", "with --pot: the potential correction is an "
                                    "unmeasured wake source under RANS");
        if (r.seq > 1)
            return err(2, "--rans", "with --seq: grid sequencing is the "
                                    "incompressible steady solve's");
        if (c.noslipY || c.bodyKind == BodyKind::None ||
            c.bodyKind == BodyKind::Floor || c.bodyKind == BodyKind::Fence)
            return err(2, "--rans", "needs a body (immersed, or wrapped by "
                                    "--ogrid); box walls (--noslip, --body "
                                    "none|floor|fence) need the ghost-row "
                                    "adapter, which is not built");
    }
    c.compSteady = comp && c.steadyTol > 0;
    if (!c.compSteady && r.given.contains("--precond"))
        return err(2, "--precond", "preconditioning changes the transient; it "
                                   "is a pseudo-time device only (requires "
                                   "the compressible --steady TOL)");
    if (!c.compSteady)
    {
        // Both knobs belong to the compressible pseudo-time march only.
        const char* why = comp ? "requires --steady TOL (the compressible "
                                 "pseudo-time march)"
                               : "is compressible steady only (the "
                                 "incompressible --steady has its own stop)";
        if (auto e = rejectGiven(r,
                                 {"--steady-steps", "--steady-scheme",
                                  "--steady-sample", "--lts-max"},
                                 why);
            e)
            return e;
    }
    else
    {
        if (!c.ogrid && !rans && !c.rotorMrf)
            return err(2, "--steady", "a compressible DNS has no steady pseudo-"
                                      "time solve on an immersed body; add "
                                      "--rans or use --ogrid");
        // The frozen rotor's steady march (masterplan_levers M1): immersed,
        // laminar, on a focal mesh. MRF + RANS is rejected above.
        if (c.rotorMrf && c.ogrid)
            return err(2, "--steady", "with an O-grid rotor is not built: its "
                                      "steady stop has no ct/cq series");
        if (c.rotorMrf && !c.focalActive)
            return err(2, "--steady", "the frozen-rotor steady march needs "
                                      "--focal: local time stepping has no "
                                      "leverage on a uniform mesh (issues.md "
                                      "6)");
        // A state relaxation in the stage hook (the sponge, the NSCBC strip)
        // is applied per stage, not per unit of local dt, so its fixed point
        // would move with the lts ratios.
        if (c.rotorMrf && (c.far == FarField::Sponge || c.nscbcStrip > 0))
            return err(2, "--steady", "with --far sponge or --nscbc-strip: a "
                                      "per-stage state relaxation is not "
                                      "local-time-step consistent, so the "
                                      "frozen rotor's fixed point would "
                                      "depend on the lts ratios");
        c.steadySample = r.steadySample;
        if (!(c.steadySample >= 1 && c.steadySample <= 100000))
            return err(2, "--steady-sample", "must be in [1, 100000] "
                                             "iterations");
        c.ltsMax = r.ltsMax;
        if (!(std::isfinite(c.ltsMax) && (c.ltsMax == 0 || c.ltsMax >= 1)))
            return err(2, "--lts-max", "must be >= 1 (a multiple of the "
                                       "smallest local dt), or 0 = unlimited");
        if (r.given.contains("--precond"))
        {
            c.precondK = r.precond;
            if (!(std::isfinite(c.precondK) && c.precondK > 0 &&
                  c.precondK <= kPrecondKMax))
                return err(2, "--precond",
                           std::format("K must be in (0, {:g}] (the U_r floor "
                                       "K U_inf; absent = off)",
                                       kPrecondKMax));
            if (rans)
                return err(2, "--precond", "with --rans is not built: the "
                                           "turbulence scalars' point-implicit "
                                           "stage has no preconditioned form");
        }
        // A relative change: >= 1 certifies nothing, and below kSteadyTolMin
        // the FP32 integrals' sampling noise decides (docs/rans.md).
        if (!(c.steadyTol >= kSteadyTolMin && c.steadyTol < 1))
            return err(2, "--steady",
                       std::format("is a relative change of the integrals: "
                                   "must be in [{:g}, 1)",
                                   kSteadyTolMin));
        if (auto e = rejectGiven(r, {"--ramp", "--kick", "--ft", "--seq"},
                                 "with --steady: pseudo-time is not time; "
                                 "--steady-steps is the cap");
            e)
            return e;
        for (const Emit k : {Emit::Png, Emit::Web, Emit::Bin, Emit::Shm,
                             Emit::Volume, Emit::Slices, Emit::Vtk})
            if (c.emit.contains(k))
                return err(2, "--emit", "frames with --steady: pseudo-time is "
                                        "not time; frames are time-keyed");
        c.ramp = 0;   // the march starts from the uniform freestream
        c.kick = 0;
        c.steadySteps = r.steadySteps;
        if (r.given.contains("--steady-steps") && r.steadySteps < 1)
            return err(2, "--steady-steps", "must be >= 1");
        c.steadyScheme = SteadyScheme(r.steadyScheme);
        if (c.steadyScheme == SteadyScheme::Implicit)
        {
            if (auto e = rejectGiven(r, {"--cfl", "--dt-recompute"},
                                     "with --steady-scheme implicit: explicit-"
                                     "march knobs; the implicit march ramps "
                                     "its own pseudo-time step");
                e)
                return e;
            if (auto e = rejectGiven(r,
                                     {"--steady-sample", "--lts-max",
                                      "--precond"},
                                     "with --steady-scheme implicit: it tunes "
                                     "the explicit march only (the Newton "
                                     "march samples every armed step and "
                                     "has no lts ratios)");
                e)
                return e;
            if (!c.ogrid)
                return err(2, "--steady-scheme",
                           "implicit (Newton-Krylov) needs --ogrid: the "
                           "immersed body has no line structure (not built)");
            if (c.grid.nz > 1 && !c.ogridRevolved)
                return err(2, "--steady-scheme",
                           "implicit (Newton-Krylov) on an extruded 3D O-grid "
                           "is not built yet (P4); use --nz 1 or explicit");
            if (c.ogridRevolved && c.ogridNphi > 0)
                return err(2, "--steady-scheme",
                           "implicit (Newton-Krylov) on the full-revolution "
                           "O-grid is not built (3D line ordering); use the "
                           "explicit march or the wedge (--nz 1)");
            // Measured stops are tens of Newton iterations: the explicit
            // march's 1000000 would let a stalled run spin before exit 8.
            if (!r.given.contains("--steady-steps"))
                c.steadySteps = kImplicitSteadySteps;
        }
        else if (!r.given.contains("--dt-recompute"))
        {
            // localDt every 16 iterations unless asked (the O-grid default).
            c.dtRecompute = 16;
        }
    }
    // --compensated: the explicit steady march's two-word state on a time
    // march (on an explicit --steady it already is the state). Every state
    // overwriter keeps the low words except the two plain-state edits named
    // below; the moving-rotor ghost swap is covered because the ghost pass
    // zeroes its cells' low words every stage.
    if (r.compensated)
    {
        if (!comp)
            return err(2, "--compensated", "is the compressible state's low "
                                           "word; the incompressible solver "
                                           "has no such state");
        if (c.compSteady && c.steadyScheme == SteadyScheme::Implicit)
            return err(2, "--compensated", "with --steady-scheme implicit: the "
                                           "Newton march keeps its own FP64 "
                                           "iterate and a plain device state");
        if (c.penalize > 0)
            return err(2, "--compensated", "with --penalize: the Brinkman band "
                                           "is a plain-state edit with no "
                                           "low-word form");
        c.compensated = !c.compSteady;
        if (c.compensated) c.meta.state = "compensated";
    }

    // Freestream turbulence, in freestream units (decision 4).
    c.ransNutInf = r.ransNutInf;
    c.ransKInf = r.ransKInf;
    c.ransMutInf = r.ransMutInf;
    if (sa && !finitePos(c.ransNutInf))
        return err(2, "--rans-nutinf", "must be finite and > 0");
    if (sst && !finitePos(c.ransKInf))
        return err(2, "--rans-kinf", "must be finite and > 0");
    if (sst && !finitePos(c.ransMutInf))
        return err(2, "--rans-mutinf", "must be finite and > 0");
    if (sa)
        c.ransPhiInf[0] = nuTildaInf(c.ransNutInf, c.nu);
    if (sst)
    {
        c.ransPhiInf[0] = kInf(c.ransKInf, c.uInf);
        c.ransPhiInf[1] = omegaInf(c.ransPhiInf[0], c.ransMutInf, c.nu);
    }

    if (rans)
    {
        if (sa && (c.ransNutInf < 3 || c.ransNutInf > 5))
            c.warnings.push_back(std::format(
                "--rans-nutinf {:g} is outside TMR's [3, 5]", c.ransNutInf));
        if (sst && (c.ransMutInf < 1e-5 || c.ransMutInf > 1e-2))
            c.warnings.push_back(std::format(
                "--rans-mutinf {:g} is outside Spalart-Rumsey's [1e-5, 1e-2]",
                c.ransMutInf));
        if (c.ogrid && c.grid.nz > 1 && !c.ogridRevolved)
            c.warnings.emplace_back(
                "the RANS mean of a z-invariant body is z-invariant; --nz 1 is "
                "the same answer");
        const double yp =
            ransYplusEstimate(ransFirstCellDistance(c), c.placement.D, c.re);
        if (yp > 1.0)
            c.warnings.push_back(std::format(
                "wall-resolved RANS with first-cell y+ ~ {:.3g} > 1 (estimate, "
                "bluff-body peak u_tau); {}",
                yp, c.ogrid ? "refine --ny or raise --ogrid-beta"
                            : "refine --n or cluster with --focal; an "
                              "under-resolved immersed run is not a "
                              "code-to-code number"));
        // The immersed d is |sdf|, exact only where the sdf is a distance.
        if (!c.ogrid && c.bodyKind == BodyKind::Lens)
            c.warnings.emplace_back(
                "--body lens: its sdf is not a true distance, so the wall "
                "distance d = |sdf| the closure reads is approximate");
    }

    // The laminar sphere wake leaves axisymmetry at Re ~ 210 through the m = 1
    // mode, which the exactly pi-symmetric full revolution cannot grow from
    // rounding (measured: cl == cs == 0.0 and the k / k + N/2 states equal
    // bit for bit). The trip is on by default; an explicit 0 is warned where
    // the symmetric state has been measured unstable.
    if (c.ogridNphi > 0 && c.ogridTrip == 0 && (rans || c.re >= 200))
        c.warnings.emplace_back(
            "--ogrid-trip 0: the full-revolution O-grid and its axisymmetric "
            "start are exactly symmetric under a rotation by pi about the "
            "axis, so the odd azimuthal modes (the m = 1 lift mode) cannot "
            "arise and an m = 1-unstable symmetric state can certify (measured: "
            "laminar Re 250, RANS SA-neg Re 1e3)");
    // A wedge start is a state, not a rest: no quiet-start ramp.
    if (!c.ogridInitFrom.empty()) c.ramp = 0;
    if (c.ogridRevolved && c.ogridNphi == 0) c.meta.wedgeDeg = c.ogridWedgeDeg;
    if (!rans && !c.compSteady) return std::nullopt;
    c.meta.rans = rans ? std::string(ransName(c.rans)) : std::string();
    c.meta.steady = "";
    c.meta.ransMode = !c.compSteady ? "urans"
                      : c.steadyScheme == SteadyScheme::Implicit
                          ? "steady-implicit-nk"
                          : "steady-explicit";
    c.meta.ransInflow = rans ? ransInflowText(c) : std::string();
    return std::nullopt;
}

// The `dns flow` validator, as ordered stages over one Case. ORDER IS
// LOAD-BEARING: the first error wins, so a stage may only move together with
// every check before it. Each stage reads what the earlier ones resolved.
std::expected<Case, ConfigError> validateFlow(const RawConfig& r, Case c)
{
    using Stage = std::optional<ConfigError> (*)(const RawConfig&, Case&);
    static constexpr Stage kStages[] = {
        flowBody, flowGrid, flowPhysics, flowThermal,
        flowLes, flowNscbc, flowBands, flowStretch,
        flowFocal, flowRotor, flowWalls, flowStartup,
        flowSolver, flowGeometry, flowOutput, flowRans
    };
    for (const Stage s : kStages)
        if (auto e = s(r, c); e) return std::unexpected(*e);
    return c;
}

// --- prop -------------------------------------------------------------------

// Python's round(x, 12): the correctly rounded 12-decimal value.
double round12(double x)
{
    double v = x;
    (void)parseDouble(std::format("{:.12f}", x), v);
    return v;
}

// --j: "a,b,c" or "start:stop:step" (stop inclusive) ->
// sorted, de-duplicated, every J >= 0. Python's parse_j, made strict: an
// empty list item is an error, not skipped.
std::expected<std::vector<double>, ConfigError> parseAdvance(std::string_view s)
{
    constexpr std::size_t kMaxPoints = 100000;
    auto fail = [&](std::string why) {
        return std::unexpected(ConfigError{2, "--j", std::move(why)});
    };
    std::vector<double> js;
    if (s.find(':') != std::string_view::npos)
    {
        double v[3];
        std::size_t start = 0;
        for (int k = 0; k < 3; ++k)
        {
            const std::size_t end = k < 2 ? s.find(':', start) : s.size();
            if (end == std::string_view::npos
                || !parseDouble(s.substr(start, end - start), v[k]))
                return fail("expected 'a,b,c' or 'start:stop:step', got '" + std::string(s) + "'");
            start = end + 1;
        }
        const double lo = v[0], hi = v[1], st = v[2];
        if (!(st > 0) || hi < lo)
            return fail("start:stop:step needs step > 0 and stop >= start");
        const double count = std::floor((hi - lo) / st + 1e-9);
        if (!(count < double(kMaxPoints)))
            return fail(std::format("more than {} advance ratios", kMaxPoints));
        for (int k = 0; k <= int(count); ++k) js.push_back(round12(lo + k * st));
    }
    else
    {
        std::size_t start = 0;
        while (start <= s.size())
        {
            const std::size_t end = std::min(s.find(',', start), s.size());
            double v = 0;
            if (!parseDouble(s.substr(start, end - start), v))
                return fail("expected 'a,b,c' or 'start:stop:step', got '" + std::string(s) + "'");
            js.push_back(v);
            start = end + 1;
        }
    }
    if (js.empty() || std::ranges::min(js) < 0.0)
        return fail("need J >= 0 (reverse flow through the disk is not modelled)");
    std::ranges::sort(js);
    js.erase(std::unique(js.begin(), js.end()), js.end());
    return js;
}

// The flags only one method reads, rejected (never ignored) with the other.
constexpr std::initializer_list<const char*> kBemtOnly = {"--stations-bemt"};
constexpr std::initializer_list<const char*> kLiftOnly = {
    "--panels",    "--wake-length", "--align-iters", "--align-tol",  "--seg-deg",
    "--core",      "--hub",         "--wake-pitch",  "--stall-visc", "--selftest"};

std::expected<Case, ConfigError> validateProp(const RawConfig& r, Case c)
{
    prop::Case& p = c.prop;
    p.method = prop::Method(r.propMethod);
    p.quiet = r.quiet;
    const bool ll = p.method == prop::Method::Liftline;
    if (auto e = rejectGiven(r, ll ? kBemtOnly : kLiftOnly,
                             ll ? "is a --method bemt option"
                                : "is a --method liftline option");
        e)
        return std::unexpected(*e);

    if (r.selftest)
    {
        // Fixed analytic cases: every input flag would be silently unused.
        if (auto e = rejectGiven(
                r, {"--planform", "--polar", "--j", "--out", "--dump-span", "--blades",
                    "--hub-radius", "--diameter", "--rpm", "--rho", "--nu", "--sound",
                    "--thin-cd0", "--thin-alpha0", "--panels", "--wake-length",
                    "--align-iters", "--align-tol", "--seg-deg", "--core", "--hub",
                    "--wake-pitch", "--stall-visc"},
                "has no meaning with --selftest (it runs fixed analytic cases)");
            e)
            return std::unexpected(*e);
        p.selftest = true;
        return c;
    }

    if (r.planform.empty())
        return bad(2, "--planform", "is required: an r_R,c_R,beta_deg CSV, hub -> tip "
                                    "(tools/propeller.py exports it)");
    if (r.polar.empty())
        return bad(2, "--polar", "is required: a re,alpha_deg,cl,cd CSV over -180..180 "
                                 "deg (tools/polars.py exports it), or `thin`");
    p.planformPath = r.planform;
    const bool thin = r.polar == "thin";
    if (!thin) p.polarPath = r.polar;
    if (!thin)
        if (auto e = rejectGiven(r, {"--thin-cd0", "--thin-alpha0"},
                                 "shapes the built-in polar; it needs --polar thin");
            e)
            return std::unexpected(*e);
    if (!(r.thinCd0 > 0 && r.thinCd0 < 1))
        return bad(2, "--thin-cd0", "must be in (0, 1)");
    if (!(r.thinAlpha0 > -10.0 && r.thinAlpha0 < 10.0))
        return bad(2, "--thin-alpha0", "must be in (-10, 10) deg: the linear range "
                                       "alpha0 +- 10 deg has to contain alpha = 0");
    p.thinCd0 = r.thinCd0;
    p.thinAlpha0Deg = r.thinAlpha0;

    if (r.blades < 1) return bad(2, "--blades", "must be >= 1");
    p.blades = r.blades;
    if (r.given.contains("--hub-radius") && !(r.hubRR > 0 && r.hubRR < 1))
        return bad(2, "--hub-radius", "must be in (0, 1) (a fraction of the tip radius)");
    p.hubRR = r.given.contains("--hub-radius") ? r.hubRR : 0.0;

    for (auto [v, f] : {std::pair{r.diameter, "--diameter"}, {r.rpm, "--rpm"},
                        {r.rho, "--rho"}, {r.nu, "--nu"}, {r.sound, "--sound"}})
        if (!finitePos(v)) return bad(2, f, "must be finite and > 0");
    p.op = {.diameter = r.diameter, .rpm = r.rpm, .rho = r.rho, .nu = r.nu,
            .sound = r.sound};

    auto js = parseAdvance(r.given.contains("--j") ? std::string_view(r.advance)
                                                   : (ll ? "0:1.0:0.1" : "0:1:0.05"));
    if (!js) return std::unexpected(js.error());
    p.js = std::move(*js);
    if (r.given.contains("--dump-span"))
    {
        if (!(r.dumpSpan >= 0)) return bad(2, "--dump-span", "need J >= 0");
        p.dumpSpan = r.dumpSpan;
    }

    if (r.stationsBemt < 4) return bad(2, "--stations-bemt", "must be >= 4");
    p.stations = r.stationsBemt;
    if (r.panels < 4) return bad(2, "--panels", "must be >= 4");
    if (r.alignIters < 0) return bad(2, "--align-iters", "must be >= 0");
    if (!finitePos(r.wakeLength)) return bad(2, "--wake-length", "must be > 0");
    if (!(r.segDeg > 0 && r.segDeg <= 30)) return bad(2, "--seg-deg", "must be in (0, 30]");
    if (!finitePos(r.alignTol)) return bad(2, "--align-tol", "must be > 0");
    if (!(r.core >= 0)) return bad(2, "--core", "must be >= 0 (0 = automatic)");
    if (!(r.stallVisc >= 0)) return bad(2, "--stall-visc", "must be >= 0 (0 = off)");
    p.ll = {.panels = r.panels, .wakeLength = r.wakeLength, .alignIters = r.alignIters,
            .alignTol = r.alignTol, .segDeg = r.segDeg, .core = r.core,
            .hub = prop::HubModel(r.liftHub), .wakePitch = prop::WakePitch(r.wakePitch),
            .stallVisc = r.stallVisc};

    if (r.given.contains("--out") && r.out.empty())
        return bad(2, "--out", "must not be empty (omit it to write no CSV)");
    p.outCsv = r.out;
    return c;
}

// The resolved prop case for --dry-run. Names the method's model content the
// way the DNS/LES label is named: what a curve came from must be visible.
std::string describeProp(const prop::Case& p)
{
    using prop::Method;
    if (p.selftest)
        return "case:\n"
               "  mode       lifting-line self-test (Biot-Savart closed forms, "
               "actuator-disk cylinder, Betz/Prandtl limits)\n"
               "  host-only  no GPU, no input files\n";
    const bool ll = p.method == Method::Liftline;
    const double n = p.op.revPerSec();
    const double tipMach = std::numbers::pi * n * p.op.diameter / p.op.sound;
    std::string s = std::format(
        "case:\n"
        "  mode       propeller thrust curve, {}\n"
        "  blade      {} blades, planform {}, hub r/R {}\n"
        "  polar      {}\n"
        "  operating  D={:.6g} m  {:.6g} rpm ({:.6g} rev/s)  rho={:.6g}  nu={:.6g}  "
        "c={:.6g}  tip Mach {:.3f}\n"
        "  J          {} points, {:.6g} .. {:.6g}{}\n",
        ll ? "LIFTING LINE (B blades, prescribed helical wake, Biot-Savart; no "
             "tip-loss factor)"
           : "BEMT (Ning residual, Prandtl tip + hub loss, Buhl branch)",
        p.blades, p.planformPath,
        p.hubRR > 0 ? std::format("{:.6g}", p.hubRR) : std::string("= the planform's first r/R"),
        p.polarPath.empty()
            ? std::format("built-in thin airfoil: 2 pi (alpha - {:.6g} deg), cd0 {:.6g}, "
                          "Viterna past +-10 deg",
                          p.thinAlpha0Deg, p.thinCd0)
            : "table " + p.polarPath,
        p.op.diameter, p.op.rpm, n, p.op.rho, p.op.nu, p.op.sound, tipMach, p.js.size(),
        p.js.front(), p.js.back(),
        p.dumpSpan ? std::format("  (span dump at J = {:g})", *p.dumpSpan) : std::string());
    if (ll)
        s += std::format(
            "  liftline   {} panels, wake {:.6g} D, {} realignments (tol {:.3g}), "
            "segment {:.6g} deg,\n"
            "             core {}, hub {}, wake pitch {}, stall viscosity {:.3g}\n",
            p.ll.panels, p.ll.wakeLength, p.ll.alignIters, p.ll.alignTol, p.ll.segDeg,
            p.ll.core > 0 ? std::format("{:.6g} tip radii", p.ll.core)
                          : std::string("auto (5% of the closest cp-node gap)"),
            prop::hubModelNames()[std::size_t(p.ll.hub)],
            prop::wakePitchNames()[std::size_t(p.ll.wakePitch)], p.ll.stallVisc);
    else
        s += std::format("  bemt       {} elements, cosine-clustered to root and tip\n",
                         p.stations);
    s += std::format("  out        {}\n  host-only  no GPU\n",
                     p.outCsv.empty() ? std::string("(table on stdout only)") : p.outCsv);
    return s;
}

// dns hit: the SI design (h = 1.5 eta) and its box-unit run, as runHit
// derives them. Pre: c validated by validatePeriodic for Subcommand::Hit.
std::string describeHit(const Case& c)
{
    const auto d = HitDesign::make(c.grid.nx, c.nu, c.urms);
    std::string emits;
    for (const auto& [k, n] : {std::pair{Emit::Stats, "stats"},
                               {Emit::Spectrum, "spectrum"},
                               {Emit::Snapshot, "snapshot"},
                               {Emit::Slices, "slices"}})
        if (c.emit.contains(k)) emits += emits.empty() ? n : std::string(",") + n;
    return std::format(
        "case:\n"
        "  mode       forced homogeneous isotropic turbulence (incompressible, "
        "periodic box,\n"
        "             linear forcing at fixed eps, Carroll-Blanquart)\n"
        "  grid       {} x {} x {}  h={:.6g}  L=2 pi (box units)\n"
        "  SI design  u'={:.6g} m/s  nu={:.6g} m^2/s  L={:.4e} m  eps={:.4e} "
        "W/kg\n"
        "             eta={:.4e} m (h = 1.5 eta)  T_e={:.4e} s  "
        "Re_lambda={:.1f} (design)\n"
        "  box units  u'=1  nu={:.6e}  eps_f={:.6e}  T_e={:.4f}\n"
        "  run        spin {:g} T_e + stats {:g} T_e (tEnd={:.6g} box)  "
        "seed={}\n"
        "  label      DNS\n"
        "  emit       {}\n"
        "  out        {}\n",
        c.grid.nx, c.grid.ny, c.grid.nz, c.grid.hx, c.urms, c.nu, d.lBoxSI,
        d.epsSI, d.etaSI, d.tEddySI, d.reLambda, d.nuBox(), d.epsBox(),
        d.tEddyBox(), c.spin, c.stats, (c.spin + c.stats) * d.tEddyBox(),
        c.seed, emits.empty() ? std::string("none") : emits, c.outPrefix);
}

} // namespace

std::expected<Case, ConfigError> validate(const RawConfig& r)
{
    Case c;
    c.sub = r.sub;
    c.backend = r.backend;
    c.physics = r.physics;
    c.far = r.far;
    c.seed = r.seed;
    c.quiet = r.quiet;
    c.dryRun = r.dryRun;
    c.outPrefix = r.out;
    c.emit = r.emit;

    // prop shares none of the grid prologue: its --out is a CSV path, and
    // optional.
    if (r.sub == Subcommand::Prop) return validateProp(r, std::move(c));

    if (c.outPrefix.empty())
        return bad(2, "--out", "output prefix must not be empty");
    // webframes.cpp concatenates the prefix into a JSON manifest unescaped.
    if (c.outPrefix.find_first_of("\"\\\n\r\t") != std::string::npos)
        return bad(2, "--out", "output prefix must not contain quotes, backslashes "
                               "or control characters (it is embedded in JSON)");

    if (c.emit.contains(Emit::Surface) && r.sub != Subcommand::Flow)
        return bad(2, "--emit", "surface is the per-face wall tally of a dns "
                                "flow body");
    if (c.emit.contains(Emit::Mesh) && r.sub != Subcommand::Flow)
        return bad(2, "--emit", "mesh exports the body-fitted O-grid of dns "
                                "flow --ogrid");
    if (r.n < 1) return bad(2, "--n", "must be >= 1");
    if (!finitePos(r.re)) return bad(2, "--re", "must be finite and > 0");
    c.re = r.re;
    c.n = r.n;

    if (r.sub == Subcommand::Bench) return validateBench(r, std::move(c));
    if (r.sub == Subcommand::Tgv || r.sub == Subcommand::Hit)
        return validatePeriodic(r, std::move(c));
    if (r.sub == Subcommand::Channel) return validateChannel(r, std::move(c));
    if (r.sub == Subcommand::Conv) return validateConv(r, std::move(c));
    if (r.sub == Subcommand::Flame) return validateFlame(r, std::move(c));
    return validateFlow(r, std::move(c));
}

double channelUbEstimate(double reTau)
{
    return 2.5 * std::log(std::max(reTau, 3.0)) + 2.6;
}

double channelPRef(double reTau, double mach)
{
    const double ub = channelUbEstimate(reTau);
    return (ub / mach) * (ub / mach) / kGammaAir;
}

std::string flowRunLabel(const Case& c)
{
    const bool rans = c.rans != RansModel::None;
    // The wedge's suffix says what it cannot resolve; the full revolution's
    // says its resolved odd-mode trip (and wedge start), which change what a
    // run can converge to, so they are never silent.
    const bool wedge = c.ogridRevolved && c.ogridNphi == 0;
    const bool full = c.ogridRevolved && c.ogridNphi > 0;
    std::string geo;
    if (wedge) geo = ransWedgeSuffix(c.ogridWedgeDeg, rans);
    if (full)
        geo = std::format(
            " [full revolution {}: {}{}]", c.ogridNphi,
            c.ogridTrip > 0
                ? std::format("odd-mode trip {:g} U", c.ogridTrip)
                : std::string("untripped, exactly pi-symmetric"),
            c.ogridInitFrom.empty() ? std::string()
                                    : ", started from the wedge " +
                                          c.ogridInitFrom);
    // A tripped cylinder O-grid starts off the symmetric state: never silent.
    if (c.ogrid && !c.ogridRevolved && c.ogridTrip > 0)
        geo = std::format(" [O-grid trip {:g} U]", c.ogridTrip);
    if (rans || c.compSteady)
    {
        const RansMode mode =
            !c.compSteady ? RansMode::Urans
            : c.steadyScheme == SteadyScheme::Implicit
                ? RansMode::SteadyImplicitNk
                : RansMode::SteadyExplicit;
        // The frozen rotor's steady march (immersed, laminar): the MRF note
        // ahead of the wall's.
        const std::string where =
            rans && !c.ogrid ? std::string(kImmersedSuffix)
            : c.rotorMrf && !c.ogrid
                ? std::string(kMrfSteadySuffix) + std::string(kImmersedSuffix)
                : geo;
        std::string l = ransRunLabel(c.rans, mode, 0, where);
        if (wedge && !c.compSteady) l += kAxisymUnsteady;
        if (c.thinSeal) l += kThinSealSuffix;
        if (c.precondK > 0) l += precondSuffix(c.precondK);
        return l;
    }
    if (wedge) return "DNS" + geo + std::string(kAxisymUnsteady);
    if (!geo.empty()) return "DNS" + geo;
    return {};
}

namespace {

// " | 2D slot Bunsen: ..." -- the burner and its anchoring, named.
std::string bunsenLabel(const BunsenGeom& b, double sigma)
{
    auto kind = [&](double t) {
        return t >= sigma ? "burnt" : t > 1 ? "warm inert" : "cold inert";
    };
    return std::format(
        " | 2D slot Bunsen: U_jet {:g} S_L, coflow {} (T/Tu {:g}) at {:g} "
        "S_L{}",
        b.uJet, kind(b.tCo), b.tCo, b.uCo,
        b.pilot > 0 ? std::format(", {} pilot strips (T/Tu {:g}) at {:g} S_L",
                                  kind(b.tPilot), b.tPilot, b.uPilot)
                    : std::string());
}

} // namespace

std::string flameRunLabel(const Case& c)
{
    assert(c.sub == Subcommand::Flame);
    if (c.flameDiffusion) {
        const DiffusionModel& m = c.diff;
        const BunsenGeom& b = c.bunsen;
        return std::format(
            "DNS [diffusion single-step F + {:g} O -> {:g} P, calorically "
            "perfect, "
            "Le=1, Y_F1={:g}, Y_O2={:g}, Z_st={:.4g}, T_ad/T0={:g}, Da={:g}, "
            "M_jet={:g}] species: MC-limited MUSCL, ONE limiter shared by the "
            "species (coupling functions exact), on the central mass flux "
            "(mean flow central) | 2D slot jet: Pe=W={:g}, coflow {:g} U_jet{}; "
            "inflow = the Burke-Schumann sheet of its mixture fraction",
            m.s, 1 + m.s, m.yF1, m.yO2, m.zst(), m.sigma, m.Da, c.mach, b.W,
            b.uCo,
            b.pilot > 0 ? std::format(", burnt Z_st pilot strips {:g} at {:g} "
                                      "U_jet",
                                      b.pilot, b.uPilot)
                        : std::string()) +
               (c.flameCandle
                    ? std::format(" | candle: gravity g={:g} U^2/l along -x "
                                  "(Fr=U^2/(gW)={:.3g}), hydrostatic "
                                  "ambient column{}",
                                  c.flameGravity,
                                  c.flameGravity > 0
                                      ? 1.0 / (c.flameGravity * b.W)
                                      : HUGE_VAL,
                                  c.flameQuiescent ? "; QUIESCENT (no fuel, "
                                                     "inflow at rest)"
                                                   : "")
                    : std::string());
    }
    return std::format(
        "DNS [premixed single-step, calorically perfect, Le=1, "
        "M_flame={:g} (inflated; CH4/air 1.15e-3)] species: MC-limited MUSCL "
        "on the central mass flux (bounded scalar transport; mean flow "
        "central){}",
        c.mach,
        c.flameBunsen ? bunsenLabel(c.bunsen, c.flameModel.sigma)
                      : std::string());
}

std::string chamberRunLabel(const Case& c)
{
    assert(c.chamberMode);
    return flameRunLabel(c) +
           " chamber: immersed adiabatic ghost wall [first-order wall "
           "position, NOT body-fitted; species flux zero through wall faces], "
           "injector = volumetric mass-flow source band" +
           (c.chamberReaction ? std::string()
                              : std::string(" [--reaction off: inert]"));
}

namespace {

std::string describeChamber(const Case& c)
{
    const ChamberSpec& s = c.chamber;
    const double sig = c.flameModel.sigma, dth = c.flameEig.dthOverDD;
    const double Tu = c.flameTu;
    const auto nz = isentropicNozzle(c.chamberMdot, sig * Tu, s.He, Tu,
                                     kGammaAir, kRgasBox);
    const double cb = std::sqrt(sig) / c.mach;
    const double ue = nz ? nz->Me * cb : 0.0;
    return std::format(
        "case:\n"
        "  mode       closed 2D combustion chamber (dns flame --mode chamber)\n"
        "  model      single step F -> P, calorically perfect (gamma 1.4), "
        "equal W, Le = 1\n"
        "             sigma = {:.6g}  beta = {:.6g}  Pr = {:.4g}  Lambda = "
        "{:.10g}  delta_th/delta_D = {:.6g}\n"
        "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = p0 = {:.6g}\n"
        "  grid       {} x {} x 1  h = {:.6g} = delta_th/{:.4g}  box {:.6g} x "
        "{:.6g}  [native 2D, y periodic]\n"
        "  chamber    inner {:.4g} x {:.4g} (back face x = {:.4g}), wall "
        "{:.4g}, nozzle {:.4g} long to exit {:.4g} at x = {:.4g} (ratio "
        "{:.3g})\n"
        "  holder     wedge apex x = {:.4g}, base {:.4g} tall at x = {:.4g}\n"
        "  injector   band x < {:.4g}: premixed Y = 1 at Tu, {:.4g} S_L bulk "
        "(mdot {:.4g} per unit depth), ramp {:.4g}\n"
        "  ambient    cold inert gas (Tu, Y = 0), coflow {:.4g} S_L, x NSCBC "
        "(etaIn {:.4g}), p = p0; sink {} cells, amp {:.3g}\n"
        "  IC         cavity fresh up to the holder base, burnt behind it "
        "(oracle profile), at rest, p0\n"
        "  estimate   isentropic exit (T0 = Tb, p_e = p0): Me = {:.4g}, u_e = "
        "{:.4g}, p0/p_amb = {:.6g}\n"
        "  run        tt = {:.6g} flame times  cfl = {:.3g}\n"
        "  label      {}\n"
        "  out        {}\n",
        sig, c.flameModel.beta, c.prandtl, c.flameEig.Lambda, dth, Tu,
        c.grid.nx, c.grid.ny, c.h, c.flameCells, c.grid.nx * c.h,
        c.grid.ny * c.h, s.Lc + s.Ln, s.Hc, s.xb(), s.tw, s.Ln, s.He, s.xe(),
        s.Hc / s.He, s.xg, s.Hg, s.xBase(), s.xb() + s.wInj, c.flameUIn,
        c.chamberMdot, c.chamberRamp, c.chamberCoflow, c.nscbcEtaIn,
        c.nscbcSink, c.nscbcSinkAmp, nz ? nz->Me : 0.0, ue,
        nz ? nz->p0 / Tu : 0.0, c.ft, c.cfl, chamberRunLabel(c), c.outPrefix);
}

} // namespace

double hFine(const Case& c)
{
    return c.focalActive ? hMin(c.grid.nz, c.focalHFine[0], c.focalHFine[1],
                                c.focalHFine[2])
                         : hMin(c.grid);
}

// ---------------------------------------------------------------------------

std::string Case::describe() const
{
    if (sub == Subcommand::Prop) return describeProp(prop);
    if (sub == Subcommand::Hit) return describeHit(*this);
    if (sub == Subcommand::Flame && flameDiffusion)
    {
        const DiffusionModel& m = diff;
        const BunsenGeom& b = bunsen;
        const double sig = m.sigma;
        double uMax = std::max({1.0, b.uCo, b.uPilot}) * (1 + b.pulse);
        std::string mode = "2D slot jet diffusion flame (dns flame --mode "
                           "diffusion)";
        std::string grav;
        if (flameCandle) {
            mode = "2D candle: the diffusion flame under gravity, pointing "
                   "+x (dns flame --mode candle)";
            const double Lb = grid.nx * h;
            if (flameGravity > 0) {
                const CandleScales cs =
                    candleScales(m, flameGravity, b.W, Lb, mach);
                uMax = std::max(uMax, cs.uFreeFall);
                grav = std::format(
                    "  gravity    g = {:.6g} U^2/l along -x; ambient column "
                    "isothermal T0, p_a(x) = p0 exp(-g x/(R T0)), p_a(L)/p0 = "
                    "{:.6g}\n"
                    "             Fr = U^2/(g W) = {:.4g}; free-fall speed "
                    "sqrt(2 g L (sigma-1)/sigma) = {:.4g} U (buoyant Mach "
                    "{:.3g})\n"
                    "             adiabatic fraction (g/cp) L/(T_ad - T0) = "
                    "{:.3g}; flicker scale 0.479 sqrt(g/W) = {:.4g} U/l\n",
                    flameGravity, 1 - cs.hydroDrop, cs.froude, cs.uFreeFall,
                    cs.machBuoyant, cs.adiabatic, cs.flicker);
            } else
                grav = "  gravity    g = 0: the diffusion jet bit for bit\n";
            if (flameQuiescent)
                grav += "  quiescent  NO fuel, every inflow row at rest (Z = "
                        "0, u = 0, T0): the hydrostatic rest check\n";
        }
        const double dtAc = cfl * h / (std::sqrt(sig) / mach + sig * uMax);
        const double dtTh = 0.25 * h * h * (1.0 / sig) /
                            (prandtl * std::max(1.0, kGammaAir / prandtl));
        return std::format(
            "case:\n"
            "  mode       {}\n"
            "{}"
            "  model      single step F + s O -> (1+s) P, omega = A rho^2 Y_F "
            "Y_O e^(-Ta/T); calorically perfect\n"
            "             (gamma 1.4), equal W, Le = 1; s = {:g}, Y_F1 = {:g}, "
            "Y_O2 = {:g} -> Z_st = {:.6g}\n"
            "             T_ad/T0 = {:.6g}, q/(cp T0) = {:.6g} per unit fuel, "
            "beta = {:.6g}, Ta/T0 = {:.6g}, Da = {:.6g} (ln A = {:.6g})\n"
            "  units      rho0 = R = U_jet = 1, l = D0/U_jet = 1 (mu = Pr = "
            "{:.4g}), T0 = p0 = 1/(gamma M^2) = {:.6g}\n"
            "  transport  {}\n"
            "  grid       {} x {} x 1  h = {:.6g} = l/{:.4g}  box {:.6g} x "
            "{:.6g}  [x jet axis, y periodic]\n"
            "  burner     slot W = {:.4g} l (Pe = Re Pr = {:.4g}, Re = {:.4g}) "
            "centred at y = {:.6g}, fuel Z = 1, u = 1 (jet Mach {:.3g})\n"
            "             oxidizer coflow {:.4g} l per side, Z = 0, u = {:.4g}"
            "{}\n"
            "             edges tanh half-width {:.4g} l ({:.3g} cells){}\n"
            "             inflow state: the Burke-Schumann sheet of Z(y) (the "
            "jet edge burns at the exit plane)\n"
            "  sheet      constant-density Burke-Schumann tip W^2/(16 "
            "erfinv(Z_st)^2) = {:.4g} l (scale only; D(T) shortens it); box "
            "length {:.4g} l past the bands\n"
            "  boundary   x-lo NSCBC per-row inflow (u, rho), etaIn {:.4g}, "
            "species pinned to the sheet; x-hi outflow p = {} (anchor sigma "
            "{:g}), backflow carries the cell's own Y\n"
            "             sink {} cells, amp {:.3g}\n"
            "  run        tt = {:.6g} l/U_jet  cfl = {:.3g}  dt est {:.3e} "
            "(acoustic {:.3e}, thermal {:.3e}; chemistry at run time)\n"
            "  label      {}\n"
            "  out        {}\n",
            mode, grav,
            m.s, m.yF1, m.yO2, m.zst(), sig, m.qOverCpT0(), m.beta, m.theta(),
            m.Da, m.lnA(), prandtl, flameTu,
            sutherland ? std::format("Sutherland mu(T), T0 = {:.6g} K", tinfK)
                       : std::string("constant mu = Pr, rho D = mu/Pr"),
            grid.nx, grid.ny, h, flameCells, grid.nx * h, grid.ny * h, b.W,
            b.W, b.W / prandtl, b.yc, mach, b.wCo, b.uCo,
            b.pilot > 0 ? std::format("\n             pilot strips {:.4g} l "
                                      "per side: burnt gas at Z_st (T_ad), u "
                                      "= {:.4g}",
                                      b.pilot, b.uPilot)
                        : std::string(),
            b.edge, b.edge / h,
            b.pulse > 0 ? std::format("\n             jet pulse u_jet (1 + "
                                      "{:.3g} sin(2 pi {:.4g} t))",
                                      b.pulse, b.pulseFreq)
                        : std::string(),
            sheetTipConstantDensity(b.W, m.zst()),
            grid.nx * h - 2.0 * nscbcSink * h, nscbcEtaIn,
            flameCandle && flameGravity > 0
                ? "p_a(x_out), both faces' incoming acoustic waves carrying "
                  "the ambient hydrostatic gradient; air drawn DOWN through "
                  "the top relaxes to the ambient rho_a, v = 0"
                : "p0",
            kDiffOutflowSigma,
            nscbcSink, nscbcSinkAmp, tEnd, cfl, std::min(dtAc, dtTh), dtAc, dtTh,
            flameRunLabel(*this), outPrefix);
    }
    if (sub == Subcommand::Flame && flameBunsen)
    {
        const double sig = flameModel.sigma, dth = flameEig.dthOverDD;
        const BunsenGeom& b = bunsen;
        const double dtAc = cfl * h / (std::sqrt(sig) / mach + sig * b.uJet);
        const double dtTh = 0.25 * h * h * (1.0 / sig) /
                            (prandtl * std::max(1.0, kGammaAir / prandtl));
        const double alpha =
            std::asin(1.0 / b.uJet) * 180.0 / std::numbers::pi;
        return std::format(
            "case:\n"
            "  mode       2D slot Bunsen flame (dns flame --mode bunsen)\n"
            "  model      single step F -> P, calorically perfect (gamma 1.4), "
            "equal W, Le = 1\n"
            "             sigma = Tb/Tu = {:.6g}  beta = {:.6g}  theta = Ta/Tu "
            "= {:.6g}  Pr = {:.4g}\n"
            "  oracle     Lambda = {:.10g}  delta_th/delta_D = {:.6g}\n"
            "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = p0 = {:.6g}\n"
            "  transport  {}\n"
            "  grid       {} x {} x 1  h = {:.6g} = delta_th/{:.4g}  box "
            "{:.6g} x {:.6g}  [x jet axis, y periodic]\n"
            "  burner     slot W = {:.4g} delta_th centred at y = {:.6g}; jet "
            "Y = 1, T = Tu, u = {:.4g} S_L (jet Mach {:.3g})\n"
            "             coflow {:.4g} delta_th per side, Y = 0, T = {:.4g} Tu, "
            "u = {:.4g} S_L{}\n"
            "             edges tanh half-width {:.4g} delta_th ({:.3g} cells)"
            "{}\n"
            "  cone       ideal sin(alpha) = S_L/U: alpha = {:.4g} deg, height "
            "{:.4g} delta_th; box length {:.4g} delta_th past the bands\n"
            "  boundary   x-lo NSCBC per-row inflow (u, rho, Y), etaIn {:.4g}; "
            "x-hi outflow p = p0; q omega re-added on the open rows\n"
            "             sink {} cells, amp {:.3g}\n"
            "  run        tt = {:.6g} flame times  cfl = {:.3g}  dt est {:.3e} "
            "(acoustic {:.3e}, thermal {:.3e})\n"
            "  label      {}\n"
            "  out        {}\n",
            sig, flameModel.beta, flameModel.theta(), prandtl, flameEig.Lambda,
            dth, flameTu,
            sutherland ? std::format("Sutherland lambda(T), S/Tu = {:.6g} "
                                     "(Tu = {:.6g} K)",
                                     flameModel.sOverTu, tinfK)
                       : std::string("constant mu = Pr, lambda = cp"),
            grid.nx, grid.ny, h, flameCells, grid.nx * h, grid.ny * h,
            b.W / dth, b.yc, b.uJet, b.uJet * mach, b.wCo / dth, b.tCo, b.uCo,
            b.pilot > 0 ? std::format("\n             pilot strips {:.4g} "
                                      "delta_th per side: Y = 0, T = {:.4g} "
                                      "Tu, u = {:.4g} S_L",
                                      b.pilot / dth, b.tPilot, b.uPilot)
                        : std::string(),
            b.edge / dth, b.edge / h,
            b.pulse > 0 ? std::format("\n             jet pulse u_jet (1 + "
                                      "{:.3g} sin(2 pi {:.4g} t))",
                                      b.pulse, b.pulseFreq)
                        : std::string(),
            alpha, b.coneH / dth, grid.nx * h / dth - 2.0 * nscbcSink / flameCells,
            nscbcEtaIn, nscbcSink, nscbcSinkAmp, tEnd, cfl,
            std::min(dtAc, dtTh), dtAc, dtTh, flameRunLabel(*this), outPrefix);
    }
    if (sub == Subcommand::Flame && chamberMode) return describeChamber(*this);
    if (sub == Subcommand::Flame)
    {
        const double sig = flameModel.sigma, dth = flameEig.dthOverDD;
        // dt estimate: the burnt acoustic bound and the thermal one (the
        // runner's stableDt; the chemistry bound is printed at run time).
        const double dtAc = cfl * h / (std::sqrt(sig) / mach + sig * flameUIn);
        const double dtTh = 0.25 * h * h * (1.0 / sig) /
                            (prandtl * std::max(1.0, kGammaAir / prandtl));
        return std::format(
            "case:\n"
            "  mode       1D freely propagating premixed flame (dns flame "
            "--mode premixed)\n"
            "  model      single step F -> P, calorically perfect (gamma 1.4), "
            "equal W, Le = 1\n"
            "             sigma = Tb/Tu = {:.6g}  beta = {:.6g}  theta = Ta/Tu "
            "= {:.6g}  Pr = {:.4g}\n"
            "  oracle     Lambda = {:.10g} (FP64 phase plane; A burns at S_L = "
            "1)  delta_th/delta_D = {:.6g}\n"
            "  units      rho_u = S_L = R = 1, delta_D = 1, Tu = p0 = "
            "1/(gamma M^2) = {:.6g}, q = cp (sigma-1) Tu = {:.6g}\n"
            "  transport  {}\n"
            "  grid       {} x {} x 1  h = {:.6g} = delta_th/{:.4g}  box "
            "{:.6g} x {:.6g}  [native 2D, y periodic]\n"
            "  flame      x_f0 = {:.6g} ({:.4g} delta_th fresh + {} sink "
            "cells), {:.4g} delta_th burnt{}\n"
            "  boundary   x NSCBC: inflow u = {:.4g} S_L (etaIn {:.4g}), "
            "outflow p = p_b = p0 - {:.4g}; species carried in at Y = 1\n"
            "             sink {} cells, amp {:.3g}\n"
            "  run        tt = {:.6g} flame times (tEnd = {:.6g})  cfl = {:.3g}  "
            "dt est {:.3e} (acoustic {:.3e}, thermal {:.3e})\n"
            "  label      {}\n"
            "  out        {}\n",
            sig, flameModel.beta, flameModel.theta(), prandtl,
            flameEig.Lambda, dth, flameTu,
            kGammaAir / (kGammaAir - 1) * (sig - 1) * flameTu,
            sutherland ? std::format("Sutherland lambda(T), S/Tu = {:.6g} "
                                     "(Tu = {:.6g} K)",
                                     flameModel.sOverTu, tinfK)
                       : std::string("constant mu = Pr, lambda = cp"),
            grid.nx, grid.ny, h, flameCells, grid.nx * h, grid.ny * h,
            flameXf0, flameUp, nscbcSink, flameDown,
            flameWrinkle > 0
                ? std::format("\n             wrinkle {:.4g} delta_th over "
                              "modes 1..{} (shortest {:.4g} delta_th)",
                              flameWrinkle, flameWrinkleModes,
                              grid.ny * h / flameWrinkleModes / dth)
                : std::string(),
            flameUIn, nscbcEtaIn,
            flameUIn * flameUIn * (sig - 1), nscbcSink, nscbcSinkAmp, ft,
            tEnd, cfl, std::min(dtAc, dtTh), dtAc, dtTh,
            flameRunLabel(*this), outPrefix);
    }
    // Shared by both conv branches.
    const std::string patchLabel =
        hotPatch >= 1.0
            ? std::string("uniformly heated")
            : std::format("heated on a centred patch of side fraction {:.4g}"
                          " ({:.1f}% of the area); the rest sits at the lid"
                          " temperature", hotPatch,
                          100.0 * (grid.nz == 1 ? hotPatch
                                                : hotPatch * hotPatch));
    // The lateral faces, in the terms the run actually has: which axes closed
    // and what a closed one is (a no-slip adiabatic wall, ghost ring and all).
    const std::string sidesLabel =
        !closedSides
            ? std::string("periodic (unbounded layer sampled over one box)")
            : std::format("CLOSED: no-slip adiabatic walls on {} (+1 ghost "
                          "column each side); fluid extent unchanged",
                          grid.nz == 1 ? "x (z stays the exact 2D wrap)"
                                       : "x and z");
    if (sub == Subcommand::Conv && physics == Physics::Incompressible)
    {
        return std::format(
            "case:\n"
            "  mode       Rayleigh-Benard convection, BOUSSINESQ "
            "(incompressible)\n"
            "  grid       {} x {} x {} ({} fluid rows + 2 ghost)  "
            "box {:.6g} x {:.6g} x {:.6g}{}\n"
            "  walls      no-slip isothermal plates at T = {:+.3g} / {:+.3g} "
            "(drop 1 over H=1)\n"
            "  sides      {}\n"
            "  floor      {}\n"
            "  driving    Ra={:.6g}  Pr={:.4g}   (free-fall scaling: "
            "H = dT = g*alpha = 1)\n"
            "  derived    nu={:.6g}  kappa={:.6g}  Uff=1  Re_ff={:.6g}  t_ff=1  "
            "cfl={:.3g}\n"
            "  limit      no sound speed, no acoustic dt, no adiabatic "
            "gradient: Ra and Nu share the full drop\n"
            "  run        tt={:.6g} free-fall times (tEnd={:.6g})  kick={:.3g}  "
            "profile-dt={:.3g}\n"
            "  label      DNS{}\n"
            "  out        {}\n",
            grid.nx, grid.ny, grid.nz, grid.ny - 2,
            lxBox, convH, lzBox, grid.nz == 1 ? "  [native 2D]" : "",
            tBot, tTop, sidesLabel, patchLabel,
            rayleigh, prandtl, nu, kappaDiff, re, cfl,
            ft, tEnd, kick, profileDt,
            grid.nz == 1 ? " (2D: qualitative only, no vortex stretching)"
                         : "",
            outPrefix);
    }
    if (sub == Subcommand::Conv)
    {
        return std::format(
            "case:\n"
            "  mode       Rayleigh-Benard convection (no-slip isothermal y "
            "walls)\n"
            "  grid       {} x {} x {} ({} fluid rows + 2 ghost)  "
            "box {:.6g} x {:.6g} x {:.6g}{}\n"
            "  walls      floor T={:.6g}, lid T={:.6g} (imposed drop {:.6g} over "
            "H={:.6g})\n"
            "  sides      {}\n"
            "  floor      {}\n"
            "  gravity    g={:.6g} in -y; adiabatic drop (g/cp)H={:.6g} "
            "({:.1f}% of the imposed)\n"
            "  driving    Ra={:.6g} on the superadiabatic drop {:.6g}  Pr={:.4g}  "
            "Mach={:.3g}\n"
            "  derived    nu={:.6g}  kappa={:.6g}  Uff={:.6g}  Re_ff={:.6g}  "
            "t_ff={:.6g}  cfl={:.3g}\n"
            "  transport  {}\n"
            "  run        tt={:.6g} free-fall times (tEnd={:.6g})  kick={:.3g}  "
            "profile-dt={:.3g}\n"
            "  label      DNS{}\n"
            "  out        {}\n",
            grid.nx, grid.ny, grid.nz, grid.ny - 2,
            lxBox, convH, lzBox, grid.nz == 1 ? "  [native 2D]" : "",
            tBot, tTop, dtemp, convH, sidesLabel, patchLabel,
            gravity, dtemp - dtempSuper,
            dtemp > 0 ? 100.0 * (dtemp - dtempSuper) / dtemp : 0.0,
            rayleigh, dtempSuper, prandtl, mach,
            nu, prandtl > 0 ? nu / prandtl : 0.0, uFreeFall, re,
            uFreeFall > 0 ? convH / uFreeFall : 0.0, cfl,
            sutherland
                ? "Sutherland mu(T), kappa(T) = mu cp / Pr"
                : "constant mu, kappa",
            ft, tEnd, kick, profileDt,
            grid.nz == 1 ? " (2D: qualitative only, no vortex stretching)"
                         : "",
            outPrefix);
    }
    if (sub == Subcommand::Channel)
    {
        // RANS: the label names the closure and the mode, and a RANS mean
        // really IS 1D -- it never carries the 2D "qualitative" suffix.
        const bool ransOn = rans != RansModel::None;
        const bool implicitSteady =
            steadyTol > 0 && steadyScheme == SteadyScheme::Implicit;
        // The ramp's three constants, spelled as the help text spells them.
        static_assert(kImplicitCfl0 == 100 && kImplicitCflGrowth == 2 &&
                      kImplicitCflMax == 1e8);
        const std::string runLine =
            steadyTol > 0
                ? std::format("steady TOL={:.6g} cap {} iterations{}",
                              steadyTol, steadySteps,
                              implicitSteady ? ", implicit Newton (CFL 100 x2 "
                                               "-> 1e8, total mass pinned)"
                                             : "")
                : std::format("tt={:.6g} eddy turnovers (tEnd={:.6g})  "
                              "kick={:.3g}  profile-dt={:.3g}{}",
                              ft, tEnd, kick, profileDt,
                              compensated ? kCompensatedNote : "");
        const RansMode mode = steadyTol <= 0   ? RansMode::Urans
                              : implicitSteady ? RansMode::SteadyImplicitLine
                                               : RansMode::SteadyExplicit;
        const std::string label =
            ransOn     ? ransRunLabel(rans, mode)
            : lesLabel ? lesRunLabel(sgsModel, cs, grid.nz == 1)
                       : std::string("DNS") +
                             (grid.nz == 1
                                  ? " (2D: qualitative only, no vortex "
                                    "stretching)"
                                  : "");
        const std::string wallScalars =
            ransSst(rans) ? "k_wall = 0 (antimirrored ghost), omega_wall = 60 "
                            "nu/(beta1 d1^2) (TMR; Dirichlet image),\n"
                            "             "
                          : "nu~_wall = 0 (antimirrored ghost), ";
        const std::string turb =
            ransOn ? std::format("  turbulence {}mu_t = 0 on the wall face, "
                                 "Pr_t = {:.3g}, wall-resolved (y+_1 = {:.3g} "
                                 "est.)\n",
                                 wallScalars, kPrandtlTurb, chanH1 * reTau)
            : lesLabel
                ? std::format("  sgs        mu_t = rho (C Delta)^2 op(grad u), "
                              "Delta = vol^(1/3) per cell (2D: area^(1/2));\n"
                              "             odd image on both ghost rows: "
                              "mu_t = 0 on the wall face{}\n",
                              sgsModel == SgsModel::Smagorinsky
                                  ? "; smag: no wall damping"
                                  : "")
                : std::string();
        return std::format(
            "case:\n"
            "  mode       compressible turbulent channel (periodic x/z, "
            "no-slip y walls)\n"
            "  grid       {} x {} x {} ({} fluid rows + 2 ghost)  "
            "box {:.6g} x {:.6g} x {:.6g}{}\n"
            "  mesh       y tanh-stretch beta={:.4g}: first cell {:.3e} "
            "(y+ = {:.3g}), growth <= {:.4f}\n"
            "  physics    Re_tau={:.6g}  nu={:.6g}  Mach_bulk={:.3g}  "
            "(u_bulk est {:.3g}, p={:.6g})  cfl={:.3g}\n"
            "  walls      {} no-slip mirrors; force f=1 (constant), "
            "u_tau target 1\n"
            "{}"
            "  run        {}\n"
            "  label      {}\n"
            "  out        {}\n",
            grid.nx, grid.ny, grid.nz, grid.ny - 2,
            lxBox, 2.0, lzBox, grid.nz == 1 ? "  [native 2D]" : "",
            stretchBeta, chanH1, chanH1 * reTau, stretchGrowth,
            reTau, nu, mach, ubEst, pRef, cfl,
            wallIso ? "isothermal" : "ADIABATIC (no heat sink -- short "
                                     "transients only)",
            turb, runLine, label, outPrefix);
    }
    // A mesh or scene path can be any length, which is why this is std::format
    // and not a fixed buffer: snprintf TRUNCATED the report of exactly the runs
    // whose body most needed naming.
    const std::string bodyPath = bodyKind == BodyKind::Scene ? " " + scenePath
                                 : meshPath.empty()          ? std::string()
                                                             : " " + meshPath;
    std::string s = std::format(
        "case:\n"
        "  physics    {}\n"
        "  body       {}{}{}\n"
        "  far field  {}{}\n"
        "  grid       {} x {} x {}  h={:.6g}  L={:.6g}{}\n"
        "  placement  cx={:.6g} cy={:.6g} cz={:.6g} D={:.6g} ({:.1f} cells)\n"
        "  Re={:.6g}  nu={:.6g}  Mach={:.6g}  cfl={:.6g}  Cs={:.6g} ({})\n"
        "  ft={:.6g}  tEnd={:.6g}  fdt={:.6g}\n"
        "  mode       {}\n"
        "  out        {}\n",
        name(physics), name(bodyKind), bodyPath,
        cutFlux ? " [cut-flux]" : "",
        // c.far stays Nscbc internally on the O-grid (nothing on that path
        // reads it); the report names the boundary the run actually has.
        ogrid ? "o-grid outer row (characteristic LODI, gpu/ogrid_bc)"
              : name(far),
        open ?(farPot ? " (lateral open, potential-corrected)"
                       : " (lateral open)")
             : (farPot ? " (potential-corrected)" : ""),
        grid.nx, grid.ny, grid.nz, h, L, grid.nz == 1 ? "  [native 2D]" : "",
        placement.cx, placement.cy, placement.cz, placement.D,
        h > 0 ? placement.D / h : 0.0,
        re, nu, mach, cfl, cs,
        lesLabel ? (sgsModel == SgsModel::Sigma ? "LES sigma" : "LES smag")
        : rans != RansModel::None ? "RANS"
                                  : "DNS",
        ft, tEnd, fdt,
        compSteady      ? "steady pseudo-time (stop on the integrals, "
                          "--steady-steps cap)"
        : steadyTol > 0 ? "stationary (stop at steady residual, tEnd = give-up cap)"
        : compensated   ? std::string("time-march to tEnd") + kCompensatedNote
                        : "time-march to tEnd",
        outPrefix);
    // Derived outflow coefficient: printed only when asked for, so a default
    // case's --dry-run reads exactly as it always did. It CHANGES the boundary
    // arithmetic, so like the DNS/LES label it must be visible.
    if (nscbcAbc != NscbcAbc::YooIm)
        s += std::string("  nscbc      outflow transverse coefficient derived ")
             + (nscbcAbc == NscbcAbc::Modal ? "per transverse DCT mode"
                                            : "per face cell")
             + " (--nscbc-abc; --nscbc-beta caps it)\n";
    // The sink is an explicit energy sink and is reported like the DNS/LES
    // label: a run that carries one says so in its resolved case.
    if (nscbcSink > 0)
        s += std::format(
            "  nscbc      scale-selective sink: {}-cell band, sin^4 cutoff "
            "(~86% at 4dx, ~1% at 16dx), sigma_max = {:.3g} c/h\n",
            nscbcSink, nscbcSinkAmp);
    // Thermal model: printed only when a run actually asked for one, so a
    // mechanical case's --dry-run reads exactly as it always did.
    if (physics == Physics::Compressible &&
        (sutherland || bodyIsothermal || nscbcThermalInflow))
    {
        const double Tinf = mach > 0 ? 1.0 / (kGammaAir * mach * mach) : 0.0;
        s += std::format(
            "  thermal    visc={}  Pr={:.6g}  Tinf={:.6g} box ({:.6g} K)\n"
            "             body wall {}{}  nscbc inflow holds {}\n",
            sutherland ? "sutherland" : "constant", prandtl, Tinf, tinfK,
            bodyIsothermal ? "isothermal" : "adiabatic",
            bodyIsothermal ? " Twall/Tinf=" + std::to_string(twallRatio)
                           : std::string(),
            nscbcThermalInflow ? "T" : "rho");
    }
    if (ogrid && !ogridRevolved)
    {
        s += std::format(
            "  o-grid     BODY-FITTED: {} angular x {} radial (2 ghost rows)"
            " x {} spanwise\n"
            "             rFar={:.6g} ({:.6g} D)  beta={:.6g}  span={:.6g}  "
            "h1≈{:.3e} (nominal circle, beta={:.6g})\n"
            "             the wall is a MESH FACE -- no cut cells, no immersed"
            " body,\n"
            "             forces are an exact surface integral\n",
            grid.nx, grid.ny, grid.nz, ogridRFar,
            placement.D > 0 ? ogridRFar / placement.D : 0.0, ogridBeta,
            ogridSpan, ogridH1, ogridBeta);
        if (ogridTrip > 0)
            s += std::format("             trip {:g} U (solenoidal vortex, 1 D "
                             "behind the body, D/4 off the axis{})\n",
                             ogridTrip,
                             grid.nz > 1 ? ", centred in the span" : "");
    }
    if (ogridRevolved && ogridNphi > 0)
    {
        const double dphi = 2.0 * std::numbers::pi / ogridNphi;
        s += std::format(
            "  o-grid     REVOLVED about x: {} meridional (theta 0..pi, + 2 "
            "ghost columns past the axis) x {} radial\n"
            "             (2 ghost rows), FULL revolution {} azimuthal cells "
            "({:.4g} deg; the axis ghosts\n"
            "             copy k + {}); polygonal body, frontal area "
            "{:.4f} of the circle's\n"
            "             axis faces degenerate (S = 0, limit metric); qRef on "
            "pi D^2/4\n"
            "             odd-mode trip {}\n"
            "             initial state {}\n",
            grid.nx - 2, grid.ny, ogridNphi, dphi * 180.0 / std::numbers::pi,
            ogridNphi / 2, std::sin(dphi) / dphi,
            ogridTrip > 0 ? std::format("{:g} U (solenoidal, 1 D behind the "
                                        "body{})",
                                        ogridTrip,
                                        ogridTrip == kOgridTripDefault
                                            ? "; the default"
                                            : "")
                          : std::string("none (exactly pi-symmetric: no odd "
                                        "azimuthal modes)"),
            ogridInitFrom.empty()
                ? std::string(ramp > 0 ? "rest (quiet-start ramp)"
                                       : "the uniform freestream")
                : std::format("the converged wedge {}_final.bin revolved over "
                              "the {} planes (mesh checked against "
                              "{}_mesh.bin), then the trip",
                              ogridInitFrom, ogridNphi, ogridInitFrom));
    }
    else if (ogridRevolved)
    {
        s += std::format(
            "  o-grid     REVOLVED about x: {} meridional (theta 0..pi, + 2 "
            "ghost columns past the axis) x {} radial\n"
            "             (2 ghost rows), axisymmetric WEDGE {:g} deg (1 real "
            "plane + 2 ghost planes)\n"
            "             axis faces degenerate (S = 0, limit metric); qRef on "
            "pi D^2/4\n",
            grid.nx - 2, grid.ny, ogridWedgeDeg);
    }
    // RANS / compressible steady / the wedge: the label and what the closure
    // resolved (RANS-off Cartesian or z-extruded time marches print none of
    // these lines).
    if (const std::string label = flowRunLabel(*this); !label.empty())
        s += std::format("  label      {}\n", label);
    if (rans != RansModel::None || compSteady)
    {
        if (rans != RansModel::None)
        {
            s += std::format("  turbulence {}, Pr_t = {:.3g}\n",
                             ransInflowText(*this), kPrandtlTurb);
            if (!ogrid)
                s += ransSst(rans)
                         ? "  wall       immersed ghost wall: k_g = a^2 k_s "
                           "(even, quadratic), omega_g = (1+a)\n"
                           "             omega_w - a omega_s, omega_w = 60 "
                           "nu/(beta1 d_s^2) (an UNVERIFIED sub-cell choice),\n"
                           "             mu_t_g = -min(a mu_t_s, fluid face "
                           "neighbours' mu_t) (no face below mu)\n"
                         : "  wall       immersed ghost wall, the velocity's "
                           "linear image: nu~_g = -a nu~_s,\n"
                           "             mu_t_g = -min(a mu_t_s, fluid face "
                           "neighbours' mu_t) (no face below mu)\n";
            else
                s += ransSst(rans)
                         ? "  wall       k_wall = 0 (antimirrored ghost), "
                           "omega_wall = 60 nu/(beta1 d1^2) per wall face\n"
                           "             (TMR; Dirichlet image), mu_t = 0 on "
                           "the wall face\n"
                         : "  wall       nu~_wall = 0 (antimirrored ghost), "
                           "mu_t = 0 on the wall face\n";
            s += std::format(
                "  y+         first cell ~{:.3g} (estimate: bluff-body peak "
                "u_tau = 0.38 Re^-0.1 U; the run measures it)\n",
                ransYplusEstimate(ransFirstCellDistance(*this), placement.D,
                                  re));
        }
        // The ramp's constants, spelled as the help text spells them.
        static_assert(kNkCfl0 == 10 && kNkCflGrowth == 2 && kNkCflMax == 1e8);
        s += !compSteady ? std::string()
             : steadyScheme == SteadyScheme::Implicit
                 ? std::format("  run        steady TOL={:.6g} cap {} "
                               "iterations, implicit Newton-Krylov (CFL 10 "
                               "x2 -> 1e8, GMRES({}) + line-SGS on a "
                               "first-order Rusanov P)\n",
                               steadyTol, steadySteps, kNkRestart)
                 : std::format("  run        steady TOL={:.6g} cap {} "
                               "iterations, local time stepping cfl={:.3g}, "
                               "localDt every {}{}\n",
                               steadyTol, steadySteps, cfl, dtRecompute,
                               (steadySample == 200
                                    ? std::string()
                                    : std::format(", stop sample every {}",
                                                  steadySample)) +
                                   (ltsMax > 0
                                        ? std::format(", lts ratio <= {:g}",
                                                      ltsMax)
                                        : std::string()) +
                                   (precondK > 0
                                        ? std::format(", Weiss-Smith low-Mach "
                                                      "preconditioner K={:g}",
                                                      precondK)
                                        : std::string()));
        s += compSteady ? std::string()
                 : std::string("  run        URANS: SSP-RK3 on the global "
                               "dt (acoustic + per-cell RANS diffusive "
                               "bound)\n");
    }
    if (stretchBeta > 0)
    {
        s += std::format(
            "  mesh       y tanh-stretch beta={:.4g}: first interior cell "
            "{:.3e} (hy/{:.1f}), growth <= {:.4f}\n"
            "             (acoustic dt shrinks with the wall cell)\n",
            stretchBeta, stretchH1,
            stretchH1 > 0 ? grid.hy / stretchH1 : 0.0, stretchGrowth);
    }
    if (focalActive)
    {
        const char* ax[3] = {"x", "y", "z"};
        std::string per;
        for (int a = 0; a < 3; ++a)
        {
            if (focalHi[a] <= focalLo[a]) continue;
            per += std::format(
                "             {} physical [{:.4g},{:.4g}] blend {:.4g}: fine "
                "{:.3e} coarse {:.3e}\n",
                ax[a], focalLo[a], focalHi[a], focalW[a], focalHFine[a],
                focalHCoarse[a]);
        }
        s += std::format("  mesh       focal clustering ratio {:.4g}, growth "
                         "<= {:.4f}, dt factor {:.3f}\n",
                         focalRatio, focalGrowth, focalDtFactor);
        s += per;
    }
    if (rotor.active())
    {
        const char* ax[3] = {"x", "y", "z"};
        const double hFine = dns::hFine(*this);
        const double tRev = 2.0 * std::numbers::pi / std::abs(rotor.omega);
        const char* shaft = ax[rotorAxisIndex(rotor.axis)];
        s += std::format(
            "  rotor      SPINNING body: shaft along {} through the placement "
            "centre\n"
            "             tip Mach {:.4g} (freestream {:.4g}, sum {:.4g} < 1)  "
            "omega={:.6g}  R={:.6g}\n"
            "             t_rev={:.6g} -> {:.2f} revolutions in tEnd={:.6g}\n",
            shaft, rotorTipMach, mach, rotorTipMach + mach, rotor.omega,
            rotor.radius, tRev, tRev > 0 ? tEnd / tRev : 0.0, tEnd);
        if (rotorMrf)
            s += std::format(
                "             frame: MRF (frozen rotor, an APPROXIMATION): "
                "geometry static at\n"
                "             azimuth 0, wall moves at omega x r, rotating-"
                "frame equations in a\n"
                "             blended cylinder around the swept volume (zone "
                "printed at run time).\n"
                "             Exact for an axisymmetric body; time-accurate "
                "blade passage needs\n"
                "             --rotor-frame moving\n");
        else
            s += std::format(
                "             azimuth {} bins (natural {} = 2 pi R / h_fine); "
                "geometry is\n"
                "             precomputed per bin and replayed -- the time "
                "loop never rebuilds it\n",
                rotor.bins, rotorNaturalBins(rotor, hFine));
        s += std::format(
            "             wall is MOVING no-slip (omega x r, tip speed "
            "{:.6g}); run reports\n"
            "             thrust and shaft torque about {}\n",
            std::abs(rotor.omega) * rotor.radius, shaft);
        if (!ogrid)
            s += std::format(
                "             blade must be >= 4 fine cells thick "
                "(h_fine={:.3e}, D/h_fine={:.1f}):\n"
                "             >= {:.4g} in world units, or the two surfaces' "
                "mirror bands overlap\n",
                hFine, hFine > 0 ? placement.D / hFine : 0.0, 4.0 * hFine);
        else
            s += "             wall is a mesh face row: no mirror band, no "
                 "thickness gate\n";
        if (rotor.thicken > 0)
            s += std::format(
                "  blade      THICKENED: surface dilated outward by {:.4g} "
                "fine cells ({:.4g} world).\n"
                "             This is a DIFFERENT BODY -- chord grows by "
                "{:.4g}, edges round.\n"
                "             Report it with every result.\n",
                bladeThickenCells, rotor.thicken, 2.0 * rotor.thicken);
    }
    if (thinSeal)
        s += "  wall       THIN SECTIONS SEALED: a fluid/fluid face whose "
             "centre segment crosses\n"
             "             the body is a wall face of both cells (zero-"
             "thickness sheet; body\n"
             "             unchanged). Label: [thin sections sealed: sub-cell "
             "sheets imaged as\n"
             "             zero-thickness walls]\n";
    if (noslipY) s += "  walls      y walls no-slip (adiabatic mirror)\n";
    return s;
}

} // namespace dns
