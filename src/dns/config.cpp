#include "dns/config.hpp"

#include "dns/ogrid.hpp"

#include "dns/remap.hpp"
#include "dns/scene.hpp"

#include <algorithm>
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

bool finitePos(double v) { return std::isfinite(v) && v > 0; }

bool hasSuffix(std::string_view s, std::string_view suf)
{
    return s.size() > suf.size() && s.substr(s.size() - suf.size()) == suf;
}

const char* name(Physics p)
{
    return p == Physics::Compressible ? "compressible" : "incompressible";
}

const char* name(SgsModel m)
{
    return m == SgsModel::Sigma ? "sigma" : "smag";
}

std::unexpected<ConfigError> bad(int code, const char* field, std::string msg)
{
    return std::unexpected(ConfigError{code, field, std::move(msg)});
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

} // namespace

// Index == the enum value; a static_assert on the size would not catch a
// TRANSPOSED pair, so the tables are written in enum order and read only
// through name()/bodyKindFromName().
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

const char* name(BodyKind b)
{
    const auto n = bodyKindNames();
    const auto i = std::size_t(b);
    return i < n.size() ? n[i] : "?";
}

const char* name(FarField f)
{
    const auto n = farFieldNames();
    const auto i = std::size_t(f);
    return i < n.size() ? n[i] : "?";
}

std::optional<BodyKind> bodyKindFromName(std::string_view s)
{
    const auto n = bodyKindNames();
    const auto it = std::ranges::find(n, s);
    if (it == n.end()) return std::nullopt;
    return BodyKind(it - n.begin());
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
                std::string("the scene selects sgs = ")
                    + (s.sgs == SgsModel::Sigma ? "sigma" : "smag")
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
    const std::size_t benchCells = std::size_t(r.n) * r.n * r.n;
    if (benchCells > std::size_t(2147483647))
        return bad(2, "--n", "grid exceeds INT_MAX cells");
    c.grid = Grid::make(r.n, r.n, r.n, 2 * pi, 2 * pi, 2 * pi);
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
        if (!(finitePos(c.mach) && c.mach < 1))
            return bad(2, "--mach", "must be in (0,1) -- the flux is "
                                    "Kennedy-Gruber central with no shock "
                                    "capturing");
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
    c.meta = {r.n, r.n, r.n, c.grid.hx, c.grid.hy, c.grid.hz,
              0, 0, 0, 0, 0, false, "none", "periodic", name(c.physics), 0,
              ""};
    return c;
}

// --- tgv / hit --------------------------------------------------------------
std::expected<Case, ConfigError> validatePeriodic(const RawConfig& r, Case c)
{
    const int nz = r.given.contains("--nz") ? r.nz : r.n;
    if (nz < 1) return bad(2, "--nz", "must be >= 1");
    if (nz != 1 && nz < 4) return bad(2, "--nz", "must be 1 (native 2D) or >= 4");
    if (r.sub == Subcommand::Hit && nz < 4)
        return bad(2, "--nz", "hit needs nz >= 4 (spectral initial condition)");
    const std::size_t cells = std::size_t(r.n) * r.n * nz;
    if (cells > std::size_t(2147483647))
        return bad(2, "--n", "grid exceeds INT_MAX cells");
    const double lz = nz == 1 ? 1.0 : 2 * pi;
    c.grid = Grid::make(r.n, r.n, nz, 2 * pi, 2 * pi, lz);
    c.nzInterior = nz;
    c.cfl = r.given.contains("--cfl") ? r.cfl : 0.9;
    if (!(c.cfl > 0 && c.cfl <= 1)) return bad(2, "--cfl", "must be in (0,1]");
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
    }
    if (c.emit.contains(Emit::Error) && nz != 1)
        return bad(2, "--emit", "--emit error compares against the 2D analytic "
                                "Taylor-Green solution and requires --nz 1");
    if (c.emit.empty()) c.emit.insert(Emit::Stats);
    c.fdt = r.fdt;
    c.meta = {r.n, r.n, nz, c.grid.hx, c.grid.hy, c.grid.hz,
              0, r.re, 0, c.fdt, 0, false, "none", "periodic",
              name(c.physics), 0, ""};
    return c;
}

// --- channel ----------------------------------------------------------------
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
    if (r.n < 4) return bad(2, "--n", "must be >= 4");
    const int nyI = r.ny;
    if (nyI < 8)
        return bad(2, "--ny", "must be >= 8 fluid cells (wall-normal "
                              "resolution; two ghost rows are added)");
    const int nz = r.nz;
    if (nz < 1) return bad(2, "--nz", "must be >= 1 (interior cells; 1 = "
                                      "native 2D)");
    if (nz != 1 && nz < 4)
        return bad(2, "--nz", "must be 1 (native 2D) or >= 4");
    if (!finitePos(r.reTau))
        return bad(2, "--re-tau", "must be finite and > 0");
    c.reTau = r.reTau;
    c.re = r.reTau;
    c.nu = 1.0 / r.reTau;
    c.mach = r.mach;
    if (!(finitePos(c.mach) && c.mach < 1))
        return bad(2, "--mach", "must be in (0,1) -- the flux is "
                                "Kennedy-Gruber central with no shock "
                                "capturing");
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
    const std::size_t cells = std::size_t(r.n) * nyT * nz;
    if (cells > std::size_t(2147483647))
        return bad(2, "--n", "grid exceeds INT_MAX cells");
    c.grid = Grid::make(r.n, nyT, nz, c.lxBox, ly, c.lzBox);
    c.nzInterior = nz;
    c.h = c.grid.hy;
    c.L = c.lxBox;
    c.chanH1 = h1;
    c.stretchH1 = h1;
    const ChannelStretchY R{ly, nyT, c.stretchBeta, delta};
    double prev = h1;
    for (int j = 2; j < nyT - 1; ++j)
    {
        const double dy = R.mapY((j + 1) * c.grid.hy) - R.mapY(j * c.grid.hy);
        c.stretchGrowth = std::max(c.stretchGrowth,
                                   std::max(dy / prev, prev / dy));
        prev = dy;
    }

    // Bulk-velocity estimate (log law) fixes the sound speed for the
    // requested BULK Mach: c_s = ubEst/Mach, p = rho c_s^2/gamma,
    // Twall = p (rho = R = 1). The IC and the isothermal wall share it.
    c.ubEst = 2.5 * std::log(std::max(c.reTau, 3.0)) + 2.6;
    c.pRef = (c.ubEst / c.mach) * (c.ubEst / c.mach) / 1.4;

    c.cfl = r.cfl;
    if (!(c.cfl > 0 && c.cfl <= 1)) return bad(2, "--cfl", "must be in (0,1]");
    c.kick = r.kick;
    if (!(std::isfinite(c.kick) && c.kick >= 0))
        return bad(2, "--kick", "must be finite and >= 0");
    if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
    c.ft = r.tt;
    c.tEnd = r.tt;
    c.profileDt = r.profileDt;
    if (!(std::isfinite(c.profileDt) && c.profileDt >= 0))
        return bad(2, "--profile-dt", "must be finite and >= 0 (0 = off)");
    c.dtRecompute = r.given.contains("--dt-recompute") ? r.dtRecompute : 16;
    if (c.dtRecompute < 1) return bad(2, "--dt-recompute", "must be >= 1");
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    if (auto e = resolveEmit(c, {Emit::Stats}); e) return std::unexpected(*e);
    // PNG is allowed at nz > 1 here: it renders the centre z-plane,
    // which for a channel is exactly the picture you want.
    c.physics = Physics::Compressible;
    c.far = FarField::Periodic;
    c.bodyKind = BodyKind::None;
    c.meta = {c.grid.nx, c.grid.ny, c.grid.nz, c.grid.hx, c.grid.hy,
              c.grid.hz, 2 * delta, c.reTau, c.mach, c.fdt, 0, false,
              "channel", "periodic", "compressible", c.kick, ""};
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
    if (r.n < 4) return bad(2, "--n", "must be >= 4");
    const int nyI = r.ny;
    if (nyI < 8)
        return bad(2, "--ny", "must be >= 8 fluid cells (wall-normal "
                              "resolution; two ghost rows are added)");
    const int nz = r.nz;
    if (nz < 1) return bad(2, "--nz", "must be >= 1 (interior cells; 1 = "
                                      "native 2D)");
    if (nz != 1 && nz < 4)
        return bad(2, "--nz", "must be 1 (native 2D) or >= 4");

    if (!finitePos(r.rayleigh))
        return bad(2, "--rayleigh", "must be finite and > 0");
    c.rayleigh = r.rayleigh;
    if (!(finitePos(r.dtemp) && r.dtemp < 2))
        return bad(2, "--dtemp", "must be in (0,2): the imposed drop as a "
                                 "fraction of the mid-layer temperature, "
                                 "so the lid sits at 1 - dtemp/2 > 0");
    c.dtemp = r.dtemp;
    c.mach = r.mach;
    if (!(finitePos(c.mach) && c.mach < 1))
        return bad(2, "--mach", "must be in (0,1) -- the flux is "
                                "Kennedy-Gruber central with no shock "
                                "capturing");
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

    const double gamma = 1.4, Rgas = 1.0, H = 1.0, Tref = 1.0;
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
    const std::size_t cells = std::size_t(nxT) * nyT * nzT;
    if (cells > std::size_t(2147483647))
        return bad(2, "--n", "grid exceeds INT_MAX cells");
    c.grid = Grid::make(nxT, nyT, nzT, lx, ly, lz);
    c.nxInterior = r.n;
    c.nzInterior = nz;
    c.h = c.grid.hy;
    c.L = c.lxBox;
    c.pRef = Rgas * Tref;   // rho_ref = 1 at mid-layer

    // The compressible branch is acoustically limited, the Boussinesq one
    // advectively; 0.4 and 0.8 are those two solvers' own defaults.
    c.cfl = r.given.contains("--cfl")
                ? r.cfl
                : (c.physics == Physics::Incompressible ? 0.8 : 0.4);
    if (!(c.cfl > 0 && c.cfl <= 1)) return bad(2, "--cfl", "must be in (0,1]");
    c.kick = r.kick;
    if (!(std::isfinite(c.kick) && c.kick >= 0))
        return bad(2, "--kick", "must be finite and >= 0");
    if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
    c.ft = r.tt;
    c.tEnd = r.tt * H / c.uFreeFall;   // free-fall times -> box time
    c.profileDt = r.profileDt;
    if (!(std::isfinite(c.profileDt) && c.profileDt >= 0))
        return bad(2, "--profile-dt", "must be finite and >= 0 (0 = off)");
    c.dtRecompute = r.given.contains("--dt-recompute") ? r.dtRecompute : 16;
    if (c.dtRecompute < 1) return bad(2, "--dt-recompute", "must be >= 1");
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    if (auto e = resolveEmit(c, {Emit::Stats}); e) return std::unexpected(*e);
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
    c.meta = {c.grid.nx, c.grid.ny, c.grid.nz, c.grid.hx, c.grid.hy,
              c.grid.hz, H, c.re, c.mach, c.fdt, 0, false,
              "convection", c.closedSides ? "closed-sides" : "periodic",
              name(c.physics), c.kick, ""};
    return c;
}

// --- flow -------------------------------------------------------------------
std::expected<Case, ConfigError> validateFlow(const RawConfig& r, Case c)
{
    // A scene file replaces --body: main loads it (phase 1.5) and hands the
    // parsed Scene to the geometry build; the validator only records the path
    // and applies the body-independent checks.
    if (!r.scene.empty())
    {
        if (!r.body.empty())
            return bad(2, "--scene", "--scene and --body are mutually exclusive "
                                     "(the scene lists its own objects)");
        c.bodyKind = BodyKind::Scene;
        c.scenePath = r.scene;
    }
    else if (r.body.empty())
        return bad(2, "--body", "required: one of " + bodyKeywords()
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
        return bad(2, "--body", "unrecognised body '" + r.body + "'; expected "
                                + bodyKeywords()
                                + ", or a path ending in .stl/.gltf/.glb");

    // Body-free box (wall boundary layers, channel-like cases): the
    // incompressible runner requires solid cells, so this is compressible-only.
    if (c.bodyKind == BodyKind::None && c.physics != Physics::Compressible)
        return bad(2, "--body", "--body none is compressible-only (the "
                                "incompressible tunnel needs a body)");

    // Far field defaults follow the physics when the user did not choose one.
    // Sponge is NEVER a default (any physics, any mode): it is an escape hatch that
    // eats the near field in a relaxation band. The compressible default is
    // the physical characteristic boundary; --far sponge must be asked for.
    if (!r.given.contains("--far"))
        c.far = c.physics == Physics::Compressible ? FarField::Nscbc
                                                   : FarField::Tunnel;

    const bool comp = c.physics == Physics::Compressible;
    if (comp && (c.far == FarField::Tunnel || c.far == FarField::Fringe))
        return bad(2, "--far", std::string("--far ") + name(c.far)
                   + " is an incompressible far field; compressible runs use "
                     "sponge, nscbc or periodic");
    if (!comp && (c.far == FarField::Sponge || c.far == FarField::Nscbc))
        return bad(2, "--far", std::string("--far ") + name(c.far)
                   + " is a compressible far field; incompressible runs use "
                     "tunnel, fringe or periodic");

    c.open = r.open;
    if (c.open && c.far != FarField::Sponge && c.far != FarField::Nscbc
        && c.far != FarField::Tunnel)
        return bad(2, "--open", std::string("--open has no meaning with --far ")
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
            return bad(2, "--body", "fence is incompressible-only for now (a "
                                    "mid-plate blocker on the compressible "
                                    "mirror wall rows is unvalidated)");
        // Compressible floor: NSCBC/sponge x planes skip the fully-solid
        // rows (the ghost wall owns them); periodic has no walls at all.
        if (comp && c.far == FarField::Periodic)
            return bad(2, "--far", "floor needs closed y walls; use --far "
                                   "nscbc (default) or an explicit sponge");
        if (!comp && c.far != FarField::Tunnel)
            return bad(2, "--far", std::string(b) +
                                   " sits on the tunnel floor; incompressible "
                                   "runs need --far tunnel (periodic/fringe "
                                   "have no wall for it to stand on)");
        if (c.open)
            return bad(2, "--open", std::string("--open removes the lateral "
                                    "walls; the ") + b + " needs the floor wall");
    }

    // Grid: two separately validated conventions, dispatched on --physics.
    const int ny = r.given.contains("--ny") ? r.ny : r.n;
    if (ny < 1) return bad(2, "--ny", "must be >= 1");
    // --nz defaults to 1 (native 2D) for the z-invariant bodies and to --n for
    // the genuinely 3D ones: a sphere or a mesh in a one-cell-thick box has no
    // solid cells at all, which used to surface as a silent no-body run.
    const bool body3d = c.bodyKind == BodyKind::Sphere || c.bodyKind == BodyKind::Mesh;
    const int nzI = r.given.contains("--nz") ? r.nz : (body3d ? r.n : 1);
    if (nzI < 1) return bad(2, "--nz", "must be >= 1 (interior cells; 1 = native 2D)");
    if (nzI != 1 && nzI < 4)
        return bad(2, "--nz", "must be 1 (native 2D) or >= 4");
    if (body3d && nzI == 1)
        return bad(2, "--nz", std::string("--body ") + name(c.bodyKind)
                   + " is a 3D body; --nz 1 leaves it with no solid cells");
    if (!finitePos(r.dr)) return bad(2, "--dr", "must be finite and > 0");
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
            return bad(2, "--ogrid", "the body-fitted O-grid is compressible-"
                                     "only (the curvilinear path is)");
        if (c.bodyKind == BodyKind::None)
            return bad(2, "--ogrid", "wraps a body; there is none");
        // A 3D O-grid around a closed surface has polar singularities (cells
        // collapse on the axis, det -> 0), which is a different mesh topology,
        // not a parameter. Only bodies with a z-invariant cross-section wrap.
        if (c.bodyKind == BodyKind::Sphere)
            return bad(2, "--ogrid", "a sphere needs a 3D O-grid, whose polar "
                                     "axis is a mesh singularity (collapsed "
                                     "cells, det = 0). Bodies with a "
                                     "z-invariant cross-section only: cylinder, "
                                     "lens, cube, or a z-invariant mesh");
        if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
            return bad(2, "--ogrid", "wall-mounted bodies are not star-shaped "
                                     "about a centre and have no annulus "
                                     "around them");
        if (auto e = rejectGiven(
                r, {"--focal", "--stretch", "--cut-flux", "--penalize", "--open",
                    "--far", "--dr", "--seq"},
                "belongs to the Cartesian box with an immersed body; --ogrid "
                "replaces both (the wall IS a mesh face, and the far field is "
                "the outer circle)");
            e)
            return std::unexpected(*e);
        if (r.n < 16)
            return bad(2, "--n", "is the ANGULAR cell count on an O-grid; "
                                 "below 16 the contour is not resolved");
        if (ny < 6)
            return bad(2, "--ny", "is the RADIAL cell count on an O-grid, two "
                                  "of which are ghost rows; >= 6 needed");
        if (!(std::isfinite(r.ogridFar) && r.ogridFar > 1.0))
            return bad(2, "--ogrid-far", "far-boundary radius in DIAMETERS; "
                                         "must be finite and > 1");
        if (!(std::isfinite(r.ogridBeta) && r.ogridBeta >= 0))
            return bad(2, "--ogrid-beta", "radial clustering; must be finite "
                                          "and >= 0 (0 = uniform)");
        const double D = pi / 4.0;
        c.ogrid = true;
        c.ogridRFar = r.ogridFar * D;
        c.ogridBeta = r.ogridBeta;
        // Spanwise extent: pi D, the usual cylinder-DNS span. nz == 1 keeps
        // the unit depth the 2D convention requires.
        c.ogridSpan = nzI > 1 ? pi * D : 1.0;
        c.grid = ogridGrid(r.n, ny, nzI, 0.5 * D, c.ogridRFar, c.ogridSpan);
        c.L = 2.0 * c.ogridRFar;
        c.h = c.grid.hy; // nominal radial spacing; the mesh reports the real h1
        // The body sits at the origin: the mesh is built about its centre.
        c.placement = {0.0, 0.0, 0.0, D};
    }
    else if (comp)
    {
        // D = pi/4 fixed, h scales with ny AND --dr: box = dr*D in y
        // (dr*(n/ny)*D in x), D spans ny/dr cells. The dr = 8 default keeps
        // the historical 8D x 8D box BIT-IDENTICAL (the *(dr/8) factor is an
        // exact *1.0 and the 2.1107 placement literal is preserved). No ghost
        // ring: the compressible far fields act on interior rows.
        if (c.dr < 4)
            return bad(2, "--dr", "compressible box is dr*D wide; below 4 the "
                                  "body sits under 1D from the inflow row");
        const double D = pi / 4.0;
        c.h = D / 64.0 * (512.0 / double(ny)) * (c.dr / 8.0);
        const double lx = r.n * c.h, ly = ny * c.h;
        const double lz = nzI > 1 ? nzI * c.h : 1.0;
        c.grid = Grid::make(r.n, ny, nzI, lx, ly, lz);
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
        c.grid = Grid::make(nxT, nyT, nzT, nxT * c.h, nyT * c.h, lz);
        const double D = c.L / c.dr;
        // Per-body placement offsets: each deliberately puts the body off-node
        // so the shedding instability trips. Carried verbatim (masterplan §6).
        // The fence is pinned to the floor instead: the wall face is at y = h
        // (top of the ghost row) and the blocker spans [wall, wall + D], so
        // cy = centre of the blocked strip. Its top edge is off-node through
        // the 0.31h x-offset alone.
        const double cx = c.h + c.L / 3.0 + 0.31 * c.h;
        const double cy = c.bodyKind == BodyKind::Fence
                                  || c.bodyKind == BodyKind::Floor
                              ? c.h + D / 2.0
                              : c.h + c.L / 2.0 + 0.17 * c.h;
        const double cz = c.bodyKind == BodyKind::Sphere
                              ? c.h + c.L / 2.0 + 0.23 * c.h
                              : (nzT == 1 ? 0.5 * c.h : c.h + nzI * c.h / 2.0);
        c.placement = {cx, cy, cz, D};
    }

    const std::size_t cells = std::size_t(c.grid.nx) * c.grid.ny * c.grid.nz;
    if (cells > std::size_t(2147483647))
        return bad(2, "--n", "grid exceeds INT_MAX cells");
    if (c.placement.D < 4 * c.h)
        return bad(2, "--dr", "body diameter is under 4 cells; refine --n or lower "
                              "--dr");
    // The fence is D/8 thick; below 32 cells of D the blocker itself is under
    // 4 cells and the sampled cut degenerates.
    if (c.bodyKind == BodyKind::Fence && c.placement.D < 32 * c.h)
        return bad(2, "--dr", "fence thickness is D/8; D under 32 cells makes it "
                              "thinner than 4 cells -- refine --n or lower --dr");
    if (c.bodyKind == BodyKind::Floor && c.dr < 2)
        return bad(2, "--dr", "floor solidifies the bottom 1/dr of the channel; "
                              "--dr under 2 blocks more than half of it");

    // Physics.
    c.mach = r.mach;
    if (comp)
    {
        if (!(finitePos(c.mach) && c.mach < 1))
            return bad(2, "--mach", "must be in (0,1) -- the flux is Kennedy-Gruber "
                                    "central with no shock capturing");
        if (r.given.contains("--tol"))
            return bad(2, "--tol", "compressible runs are fully explicit; there is "
                                   "no pressure solve to converge");
    }
    else
    {
        if (auto e = rejectGiven(r, {"--mach", "--kick"},
                                 "is compressible-only");
            e)
            return std::unexpected(*e);
    }

    c.nu = c.uInf * c.placement.D / c.re;
    c.cfl = r.given.contains("--cfl") ? r.cfl : (comp ? 0.4 : 0.8);
    if (!(c.cfl > 0 && c.cfl <= 1)) return bad(2, "--cfl", "must be in (0,1]");

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
            return std::unexpected(*e);
    if (!finitePos(c.tinfK))
        return bad(2, "--tinf", "must be finite and > 0 (kelvin)");
    if (!finitePos(c.prandtl))
        return bad(2, "--prandtl", "must be finite and > 0");
    if (!finitePos(c.twallRatio))
        return bad(2, "--twall", "must be finite and > 0 (a ratio to Tinf)");
    if (r.given.contains("--twall") && !c.bodyIsothermal)
        return bad(2, "--twall", "sets the temperature of an ISOTHERMAL wall; "
                                 "add --wall isothermal (the default body wall "
                                 "is adiabatic)");
    if (c.bodyIsothermal && c.bodyKind == BodyKind::None)
        return bad(2, "--wall", "an isothermal wall needs a body; there is none");
    if (c.nscbcThermalInflow && c.far != FarField::Nscbc)
        return bad(2, "--nscbc-tinflow", "is the NSCBC entropy inflow target; "
                                         "use --far nscbc");

    c.cs = r.les;
    if (!(std::isfinite(c.cs) && c.cs >= 0))
        return bad(2, "--les", "must be finite and >= 0");
    c.sgsModel = r.sgs;
    if (r.given.contains("--sgs") && c.cs == 0)
        return bad(2, "--sgs", "selects the SGS model for --les; give --les CS "
                               "(> 0) or drop --sgs");
    c.cutFlux = r.cutFlux;
    if (c.cutFlux && c.bodyKind == BodyKind::None)
        return bad(2, "--cut-flux", "apertures the body's momentum fluxes; "
                                    "there is no body");
    if (c.cutFlux && r.given.contains("--stretch"))
        return bad(2, "--cut-flux", "apertures are uniform-mesh geometry; the "
                                    "curvilinear path has none. Drop --stretch");
    if (c.cutFlux && comp)
        return bad(2, "--cut-flux", "compressible cut-flux DIVERGES near the "
                                    "body (measured twice, 2026-07-23: the KG "
                                    "split flux loses its KE property under "
                                    "per-face area weighting). Incompressible "
                                    "only until redesigned");
    c.penalize = r.penalize;
    if (!(std::isfinite(c.penalize) && c.penalize >= 0))
        return bad(2, "--penalize", "must be finite and >= 0");
    if (c.penalize > 0 && c.bodyKind == BodyKind::None)
        return bad(2, "--penalize", "decays the near-body band; there is no body");
    if (c.penalize > 0 && r.given.contains("--stretch"))
        return bad(2, "--penalize", "needs the uniform-mesh fluid fractions; "
                                    "drop --stretch");
    if (c.sgsModel == SgsModel::Sigma && c.cs > 0 && c.grid.nz == 1)
        return bad(2, "--sgs", "sigma: D_sigma is identically ZERO for any 2D "
                               "flow (that is the model's design); at nz == 1 "
                               "it is a silent no-op. Use nz > 1 or smag");
    // Honest labelling: the boundary layer scales as D/sqrt(Re).
    const double blCells = c.placement.D / std::sqrt(c.re) / c.h;
    c.lesLabel = c.cs > 0;
    if (blCells < 3.0 && !c.lesLabel && c.bodyKind != BodyKind::None)
    {
        c.warnings.push_back(
            std::format("boundary layer is {:.1f} cells (D/sqrt(Re)/h); this "
                        "is under-resolved for DNS", blCells));
    }

    // Far-field parameters.
    c.nscbcBeta = r.nscbcBeta;
    c.nscbcEtaIn = r.given.contains("--nscbc-etain") ? r.nscbcEtaIn : 100.0;
    c.nscbcStrip = r.nscbcStrip;
    c.nscbcEtaLat = r.nscbcEtaLat;
    c.nscbcSink = r.nscbcSink;
    c.nscbcSinkAmp = r.nscbcSinkAmp;
    if (c.far == FarField::Nscbc)
    {
        if (!finitePos(c.nscbcEtaIn))
            return bad(2, "--nscbc-etain", "must be > 0");
        // Empirically bracketed stability window (masterplan §6.3): 0 pumps a
        // growing global mode, 1.0 diverges at t = 35.
        if (!(std::isfinite(c.nscbcBeta) && c.nscbcBeta >= 0 && c.nscbcBeta < 1))
            return bad(2, "--nscbc-beta", "must be in [0,1); 1.0 is measured to diverge "
                                          "at t = 35 (full transverse cancellation "
                                          "is ill-posed at low Mach)");
        if (c.nscbcStrip < 0 || c.nscbcStrip >= c.grid.nx / 4)
            return bad(2, "--nscbc-strip", "must be >= 0 and < nx/4");
        if (c.re > 800 && c.nscbcStrip == 0)
            return bad(2, "--nscbc-strip", "pure NSCBC is validated only up to "
                                           "Re 800 (it diverges at Re 2000, 2048^2); "
                                           "pass --nscbc-strip ~ny/64 (or see "
                                           "--nscbc-sink, docs/nscbc.md)");
        if (c.nscbcSink < 0 || 2 * c.nscbcSink >= c.grid.nx ||
            (c.open && c.nscbcSink > 0 && 2 * c.nscbcSink >= c.grid.ny))
            return bad(2, "--nscbc-sink", "band must be >= 0 and fit twice on "
                                          "every open axis");
        if (!(std::isfinite(c.nscbcSinkAmp) && c.nscbcSinkAmp > 0 &&
              c.nscbcSinkAmp <= 2.0))
            return bad(2, "--nscbc-sink-amp",
                       "must be in (0, 2]; sigma dt <= 2 * cfl * axes is the "
                       "SSP-RK3 real-axis bound at cfl 0.4");
        // The sink's cutoff is quoted in CELLS on a uniform mesh; under a
        // remap the physical cutoff varies across the band, which needs its
        // own measurement before it is allowed. Rejected, never ignored.
        if (c.nscbcSink > 0 && (r.stretch > 0 || !r.focal.empty()))
            return bad(2, "--nscbc-sink", "is measured on the uniform mesh; "
                                          "--stretch/--focal remap the band "
                                          "(unmeasured -- not yet allowed)");
        if (c.nscbcSink > 0 && c.bodyKind == BodyKind::Floor)
            return bad(2, "--nscbc-sink", "band rows overlap the solid floor "
                                          "rows the ghost wall owns "
                                          "(--body floor)");
        if (c.nscbcSink > 0 && r.ogrid)
            return bad(2, "--nscbc-sink", "is a Cartesian face band; the "
                                          "body-fitted O-grid has no such "
                                          "faces");
    }
    else if (r.given.contains("--nscbc-strip") && c.nscbcStrip > 0)
        return bad(2, "--nscbc-strip", "requires --far nscbc");
    if (c.far != FarField::Nscbc && r.given.contains("--nscbc-sink") &&
        c.nscbcSink > 0)
        return bad(2, "--nscbc-sink", "requires --far nscbc");
    if (r.given.contains("--nscbc-sink-amp") && c.nscbcSink == 0)
        return bad(2, "--nscbc-sink-amp", "sets the strength of the "
                                          "--nscbc-sink band; pass a width");
    if (c.far != FarField::Nscbc && r.given.contains("--nscbc-etain"))
        return bad(2, "--nscbc-etain", "requires --far nscbc");
    if (r.given.contains("--nscbc-etalat"))
    {
        if (c.far != FarField::Nscbc || !c.open)
            return bad(2, "--nscbc-etalat", "is the OPEN lateral faces' relax "
                                            "rate; needs --far nscbc --open "
                                            "(walled laterals never read it)");
        if (!finitePos(c.nscbcEtaLat))
            return bad(2, "--nscbc-etalat", "must be > 0");
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
        return bad(2, "--nscbc-abc", "is the NSCBC outflow's transverse "
                                     "coefficient; use --far nscbc");
    if (c.nscbcAbc == NscbcAbc::Modal)
    {
        if (c.open)
            return bad(2, "--nscbc-abc modal",
                       "needs walled lateral faces: with --open the far-field "
                       "problem is not separable in the transverse directions, "
                       "so the modal basis it transforms in does not exist "
                       "(--nscbc-abc adaptive works there)");
        if (r.stretch > 0 || !r.focal.empty())
            return bad(2, "--nscbc-abc modal",
                       "is derived on the UNIFORM outflow plane; --stretch and "
                       "--focal remap it (--nscbc-abc adaptive works there)");
        if (r.ogrid)
            return bad(2, "--nscbc-abc modal",
                       "is a Cartesian x-face closure; the body-fitted O-grid "
                       "has no such face");
        if (c.bodyKind == BodyKind::Floor)
            return bad(2, "--nscbc-abc modal",
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
        return bad(2, "--pot", "models the body's wake source and displacement "
                               "doublet; meaningless with --body none");
    if (c.farPot && c.far != FarField::Nscbc
        && !(c.far == FarField::Tunnel && c.open))
        return bad(2, "--pot", "requires --far nscbc (compressible) or "
                               "--far tunnel --open (incompressible) -- the "
                               "correction retargets an open far field");

    // Sponge / fringe band width. clamp(0.1n, 8, n/2-1) is UB when n < 18.
    // Band width is measured along x for BOTH physics. (This was once a
    // `comp ? nx : nx` ternary -- a physics distinction someone intended and
    // abandoned. Restoring one needs a measurement, not a guess.)
    const int bandBase = c.grid.nx;
    if ((c.far == FarField::Fringe || c.far == FarField::Sponge
         || c.far == FarField::Nscbc)
        && bandBase < 18)
        return bad(2, "--n", "sponge/fringe far fields need at least 18 cells along x");
    c.spongeCells = r.spongeCells > 0
                        ? r.spongeCells
                        : std::clamp(int(0.1 * bandBase), 8, bandBase / 2 - 1);
    if (c.spongeCells < 1 || c.spongeCells >= bandBase / 2)
        return bad(2, "--sponge-cells", "must be >= 1 and < nx/2");

    // Auto: band z only when the box is thick enough to hold the band. A thin
    // quasi-2D run gets x,y only -- a z band there would sponge the whole wake.
    const bool zBandFits = c.grid.nz >= 2 * c.spongeCells + 2;
    c.fringeAxes = r.fringeAxes != 0 ? r.fringeAxes : (zBandFits ? 3 : 2);
    if (c.fringeAxes != 2 && c.fringeAxes != 3)
        return bad(2, "--fringe-axes", "must be 2 or 3");
    if (c.far == FarField::Fringe && c.fringeAxes == 3 && !zBandFits)
        return bad(2, "--fringe-axes", "a z fringe band needs nz >= 2*cells + 2; "
                                       "pass --fringe-axes 2 for a z-invariant run");

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
        return bad(2, "--stretch", "must be in [0,4] (tanh clustering strength; "
                                   "4 already compresses the wall cell ~190x)");
    if (c.stretchBeta > 0)
    {
        if (!comp)
            return bad(2, "--stretch", "the curvilinear solver is compressible-"
                                       "only; --stretch needs --physics "
                                       "compressible");
        if (c.bodyKind != BodyKind::None)
            return bad(2, "--stretch", "the ghost wall samples the body SDF at "
                                       "uniform cell positions; stretched runs "
                                       "need --body none for now");
        if (c.open)
            return bad(2, "--open", "--stretch clusters cells toward the closed "
                                    "y walls; open lateral faces defeat it");
        if (c.far != FarField::Nscbc && c.far != FarField::Sponge)
            return bad(2, "--far", "stretched-y runs need --far nscbc (the "
                                   "characteristic terms evaluate the PointRemap) "
                                   "or an explicit --far sponge");
        // Resolved wall resolution: first INTERIOR cell (row 1; rows 0 and
        // ny-1 are the wall-mirror ghost rows) and max growth ratio. These
        // set the acoustic dt, so --dry-run must surface them.
        const TanhStretchY R{double(c.grid.ny) * c.grid.hy, c.stretchBeta};
        const double hy = c.grid.hy;
        c.stretchH1 = R.mapY(2 * hy) - R.mapY(hy);
        double prev = R.mapY(hy) - R.mapY(0.0);
        for (int j = 1; j < c.grid.ny; ++j)
        {
            const double dy = R.mapY((j + 1) * hy) - R.mapY(j * hy);
            c.stretchGrowth = std::max(c.stretchGrowth,
                                       std::max(dy / prev, prev / dy));
            prev = dy;
        }
    }
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
            return bad(2, "--focal", "the curvilinear solver is compressible-"
                                     "only; --focal needs --physics "
                                     "compressible");
        if (c.stretchBeta > 0)
            return bad(2, "--focal", "--stretch and --focal are different "
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
            return bad(2, "--focal", "a mesh/scene body's bounding box is only "
                                     "known after the file is loaded, and "
                                     "validation touches no filesystem; give "
                                     "the plateau explicitly for now (a rotor "
                                     "is exempt: its swept extent follows from "
                                     "--dr and --rotor-span)");
        if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
            return bad(2, "--focal", "the wall-mounted bodies span the full "
                                     "domain width; there is no compact "
                                     "plateau to cluster on");
        if (c.cutFlux || c.penalize > 0)
            return bad(2, "--focal", "apertures and the Brinkman band are "
                                     "uniform-mesh geometry; the curvilinear "
                                     "path has none");
        if (c.open)
            return bad(2, "--open", "--focal needs the closed lateral faces "
                                    "its blend margin is measured against");
        if (c.far != FarField::Nscbc && c.far != FarField::Sponge)
            return bad(2, "--far", "focal runs need --far nscbc (the "
                                   "characteristic terms evaluate the "
                                   "PointRemap) or an explicit --far sponge");

        c.focalRatio = r.focalRatio;
        if (!(std::isfinite(c.focalRatio) && c.focalRatio > 1
              && c.focalRatio <= 16))
            return bad(2, "--focal-ratio", "must be in (1,16] (coarse/fine "
                                           "cell-size ratio; the fine spacing "
                                           "sets the acoustic dt)");
        const double blendFrac = r.focalBlend;
        if (!(finitePos(blendFrac) && blendFrac <= 2))
            return bad(2, "--focal-blend", "must be in (0,2] (blend width per "
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

        double box[6];
        if (r.focal == "auto")
        {
            if (!hasBody)
                return bad(2, "--focal", "auto needs a body; --body none has "
                                         "no AABB to focus on");
            if (!(std::isfinite(r.focalWake) && r.focalWake >= 0))
                return bad(2, "--focal-wake", "must be finite and >= 0");
            if (!(std::isfinite(r.focalPad) && r.focalPad >= 0))
                return bad(2, "--focal-pad", "must be finite and >= 0");
            const double D = c.placement.D, pad = r.focalPad * D;
            const int nA[3] = {c.grid.nx, c.grid.ny, c.grid.nz};
            const double hA[3] = {c.grid.hx, c.grid.hy, c.grid.hz};
            for (int a = 0; a < 3; ++a)
            {
                // Axes the body does not localize stay uniform.
                if (a == 2 && (zSpanning || c.grid.nz == 1))
                {
                    box[4] = box[5] = 0.0;
                    continue;
                }
                const double want[2] = {
                    bLo[a] - pad,
                    bHi[a] + pad + (a == 0 ? r.focalWake * D : 0.0)};
                // Smallest plateau that still satisfies the containment rule
                // below; the desired box shrinks toward it until its blend
                // clears the domain faces. The resolved box is what
                // --dry-run prints, so the shrink is never silent.
                const double guard = 3.0 * hA[a] / c.focalRatio;
                const double need[2] = {bLo[a] - guard, bHi[a] + guard};
                const double Lax = nA[a] * hA[a];
                const double m =
                    (a == 0 ? std::max(4, c.nscbcStrip + 2) : 4) * hA[a];
                auto fits = [&](double lo, double hi) {
                    const double w = blendFrac * (hi - lo);
                    return hi > lo && lo - w >= m && hi + w <= Lax - m;
                };
                if (!fits(std::min(need[0], want[0]),
                          std::max(need[1], want[1])) &&
                    !fits(need[0], need[1]))
                    return bad(2, "--focal", std::string(axName[a])
                                   + " has no plateau that both contains the "
                                     "body and clears the domain faces; the "
                                     "box is too small for --focal-blend");
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
                const std::string tok(s.substr(0, comma));
                char* end = nullptr;
                box[q] = std::strtod(tok.c_str(), &end);
                if (tok.empty() || end != tok.c_str() + tok.size()
                    || !std::isfinite(box[q]))
                    return bad(2, "--focal", "expects six numbers "
                                             "x0,x1,y0,y1,z0,z1 (world units) "
                                             "or 'auto'");
                if (comma == std::string_view::npos)
                {
                    if (q != 5)
                        return bad(2, "--focal", "expects six comma-separated "
                                                 "numbers x0,x1,y0,y1,z0,z1");
                    s = {};
                }
                else
                    s = s.substr(comma + 1);
            }
            if (!s.empty())
                return bad(2, "--focal", "expects exactly six numbers "
                                         "x0,x1,y0,y1,z0,z1");
        }

        const int nAx[3] = {c.grid.nx, c.grid.ny, c.grid.nz};
        const double hAx[3] = {c.grid.hx, c.grid.hy, c.grid.hz};
        for (int a = 0; a < 3; ++a)
        {
            const double lo = box[2 * a], hi = box[2 * a + 1];
            const double L = nAx[a] * hAx[a];
            if (hi <= lo)
            {
                // Degenerate pair = leave this axis uniform.
                if (hi < lo)
                    return bad(2, "--focal", std::string(axName[a])
                                                 + " plateau has hi < lo");
                c.focalHFine[a] = c.focalHCoarse[a] = hAx[a];
                continue;
            }
            if (a == 2 && c.grid.nz == 1)
                return bad(2, "--focal", "nz == 1 IS the exact 2D restriction; "
                                         "a z plateau there is meaningless -- "
                                         "pass z0,z1 equal");
            const double w = blendFrac * (hi - lo);
            // Identity margin at both faces: the blend support must end at
            // least 4 cells short of each domain face (x also clears the
            // NSCBC rows and any absorbing strip), so the far grid is
            // exactly uniform where the boundary treatment reads it.
            const int mCells = a == 0 ? std::max(4, c.nscbcStrip + 2) : 4;
            const double margin = mCells * hAx[a];
            if (!(lo - w >= margin && hi + w <= L - margin))
                return bad(2, "--focal",
                           std::string(axName[a]) + " plateau plus its blend "
                           "must clear both domain faces by " +
                           std::to_string(mCells) + " cells (the map must be "
                           "the exact identity there); shrink the box or "
                           "--focal-blend");
            const double fine = hAx[a] / c.focalRatio;
            if (hi - lo < 4 * fine)
                return bad(2, "--focal", std::string(axName[a])
                                             + " plateau spans under 4 fine "
                                               "cells; widen it");
            // Containment: the body plus the ghost wall's 3-cell mirror
            // neighbourhood must sit inside the PLATEAU, never in the blend.
            // The wall stencil is then built on locally uniform cells, which
            // is the geometry it was validated on.
            if (hasBody && !(a == 2 && zSpanning))
            {
                const double guard = 3.0 * fine;
                if (!(lo <= bLo[a] - guard && hi >= bHi[a] + guard))
                    return bad(2, "--focal",
                               std::string(axName[a]) + " plateau must contain "
                               "the body plus a 3-fine-cell guard band "
                               "([" + std::to_string(bLo[a] - guard) + "," +
                               std::to_string(bHi[a] + guard) + "]); the ghost "
                               "wall's mirror stencil must not straddle the "
                               "coarse/fine blend");
            }
            c.focalActive = true;
            c.focalLo[a] = lo;
            c.focalHi[a] = hi;
            c.focalW[a] = w;
            const Focal1D f = Focal1D::make(L, lo, hi, w, c.focalRatio);
            c.focalHFine[a] = f.c * f.r * hAx[a];
            c.focalHCoarse[a] = f.c * hAx[a];
            // Discrete adjacent growth: the KG flux carries no dissipation,
            // so a fast coarse->fine transition is a dispersive reflector.
            double prev = f.map(hAx[a]) - f.map(0.0);
            for (int i = 1; i < nAx[a]; ++i)
            {
                const double d = f.map((i + 1) * hAx[a]) - f.map(i * hAx[a]);
                c.focalGrowth =
                    std::max(c.focalGrowth, std::max(d / prev, prev / d));
                prev = d;
            }
        }
        if (!c.focalActive)
            return bad(2, "--focal", "every axis pair is degenerate; the mesh "
                                     "would be uniform -- drop --focal");
        if (c.focalGrowth > 1.25)
            return bad(2, "--focal-blend", "max adjacent cell growth " +
                       std::to_string(c.focalGrowth) + " exceeds 1.25; the "
                       "central flux has no dissipation and reflects off a "
                       "sharp transition. Widen --focal-blend or lower "
                       "--focal-ratio");
        const double hUni = c.grid.nz == 1
                                ? std::min(c.grid.hx, c.grid.hy)
                                : std::min({c.grid.hx, c.grid.hy, c.grid.hz});
        const double hFoc = c.grid.nz == 1
                                ? std::min(c.focalHFine[0], c.focalHFine[1])
                                : std::min({c.focalHFine[0], c.focalHFine[1],
                                            c.focalHFine[2]});
        c.focalDtFactor = hFoc / hUni;
    }
    else
    {
        if (auto e = rejectGiven(r,
                                 {"--focal-ratio", "--focal-blend",
                                  "--focal-wake", "--focal-pad"},
                                 "requires --focal");
            e)
            return std::unexpected(*e);
    }

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
                                      "--rotor-span", "--blade-thicken"},
                                     "configures a spinning body; give "
                                     "--rotor-mach to have one");
                e)
                return std::unexpected(*e);
        }
        else
        {
            if (c.sub != Subcommand::Flow)
                return bad(2, "--rotor-mach", "only `dns flow` immerses a body "
                                              "that could spin");
            if (!comp)
                return bad(2, "--rotor-mach", "the moving wall is the "
                               "compressible ghost wall; --rotor-mach needs "
                               "--physics compressible");
            if (c.ogrid)
                return bad(2, "--rotor-mach", "an O-grid IS the body surface: "
                               "spinning the body would spin the mesh, which "
                               "is a different solver. Immerse it in the box");
            if (c.bodyKind == BodyKind::None)
                return bad(2, "--rotor-mach", "there is no body to spin");
            if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
                return bad(2, "--rotor-mach", "the wall-mounted bodies are "
                               "sealed through the domain floor; they have no "
                               "shaft and cannot turn");
            if (c.cutFlux || c.penalize > 0)
                return bad(2, "--rotor-mach", "apertures and the Brinkman band "
                               "are built once from a static body; the moving "
                               "wall is the ghost wall alone");
            if (r.seq > 1 || r.steady > 0)
                return bad(2, "--rotor-mach", "a spinning body has no steady "
                               "state and no per-level restart; drop --seq / "
                               "--steady");
            if (!finitePos(r.rotorMach))
                return bad(2, "--rotor-mach", "must be finite and > 0 (the TIP "
                                              "Mach number)");
            // The KG central flux carries no shock capturing, so the FASTEST
            // material point in the run has to stay subsonic -- that is the
            // advancing tip, not the freestream.
            if (r.rotorMach + c.mach >= 1.0)
                return bad(2, "--rotor-mach", "tip Mach + freestream Mach must "
                               "stay under 1: the advancing tip is the fastest "
                               "point in the flow and the flux has no shock "
                               "capturing");
            if (!(finitePos(r.rotorSpan) && r.rotorSpan <= 0.5))
                return bad(2, "--rotor-span", "blade half-extent along the "
                               "shaft in diameters, in (0, 0.5]: beyond 0.5 the "
                               "swept volume leaves the placement cube that "
                               "bounds every extent check");
            if (!(std::isfinite(r.bladeThicken) && r.bladeThicken >= 0))
                return bad(2, "--blade-thicken", "must be finite and >= 0 "
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
                return std::unexpected(*e);
            if (c.grid.nz == 1 && r.rotorAxis != 2)
                return bad(2, "--rotor-axis", "nz == 1 is the exact 2D "
                               "restriction: only a z shaft keeps the motion in "
                               "the plane. An x or y shaft sweeps the blade out "
                               "of it, which 2D cannot represent");

            // Fine spacing: the plateau's under --focal, the grid's otherwise.
            // It sets both the natural bin count and what a "cell" of
            // --blade-thicken means, so the two always agree.
            const double hFine =
                c.focalActive
                    ? (c.grid.nz == 1
                           ? std::min(c.focalHFine[0], c.focalHFine[1])
                           : std::min({c.focalHFine[0], c.focalHFine[1],
                                       c.focalHFine[2]}))
                    : (c.grid.nz == 1 ? std::min(c.grid.hx, c.grid.hy)
                                      : std::min({c.grid.hx, c.grid.hy,
                                                  c.grid.hz}));

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

            const int natural = rotorNaturalBins(c.rotor, hFine);
            if (r.given.contains("--rotor-bins"))
            {
                if (r.rotorBins < 8)
                    return bad(2, "--rotor-bins", "under 8 bins the body jumps "
                                   "by more than a blade width per step change; "
                                   "0 selects the natural count");
                if (r.rotorBins > 1 << 16)
                    return bad(2, "--rotor-bins", "over 65536 bins; the table "
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
                return bad(2, "--blade-thicken", "dilation exceeds a quarter of "
                               "the tip radius; that is a different body, not a "
                               "resolvable one");

            c.rotorTipMach = r.rotorMach;
            c.rotorSpanD = r.rotorSpan;
            c.bladeThickenCells = r.bladeThicken;

            // The blade has to be thick enough for the ghost wall's mirror
            // band on BOTH sides. We cannot measure the mesh here (validation
            // touches no filesystem), so this is the advisory; the run reports
            // the measured deep-solid count, which is the real check.
            c.warnings.push_back(std::format(
                "rotor: the ghost wall needs a mirror band >= 2 cells thick "
                "per side, so the blade must be >= 4 fine cells thick "
                "(h_fine {:.4g}, D = {:.1f} fine cells). The run reports the "
                "measured deep-solid cell count; 0 means the two blade "
                "surfaces read through each other",
                hFine, c.placement.D / hFine));
        }
    }

    if (c.noslipY)
    {
        if (!comp)
            return bad(2, "--noslip", "the y mirror walls belong to the "
                                      "compressible far field; --noslip needs "
                                      "--physics compressible");
        if (c.open)
            return bad(2, "--noslip", "--open removes the y walls; there is "
                                      "nothing to make no-slip");
        if (c.far != FarField::Sponge && c.far != FarField::Nscbc)
            return bad(2, "--noslip", "requires a far field with closed y walls "
                                      "(--far sponge or --far nscbc)");
    }

    // Start-up. The raw default (2.0) is the COMPRESSIBLE default; the
    // incompressible ramp is opt-in so existing impulsive-start trajectories
    // and anchors do not silently move.
    c.ramp = comp ? r.ramp : (r.given.contains("--ramp") ? r.ramp : 0.0);
    if (!comp && c.ramp > 0 && (r.seq > 1 || r.steady > 0))
        return bad(2, "--ramp", "the quiet start would re-ramp the developed "
                                "field each --seq level / fight --steady; "
                                "use an impulsive start there");
    c.kick = r.kick;
    if (!(std::isfinite(c.ramp) && c.ramp >= 0)) return bad(2, "--ramp", "must be >= 0");
    if (!std::isfinite(c.kick)) return bad(2, "--kick", "must be finite");

    // Solver.
    c.tol = r.tol;
    if (!finitePos(c.tol)) return bad(2, "--tol", "must be > 0");
    c.maxIters = r.maxIters;
    if (c.maxIters < 1) return bad(2, "--max-iters", "must be >= 1");
    // Default per current app: 16 for the compressible cylinder (a validated
    // perf fix, masterplan §6.7), 1 everywhere else.
    c.dtRecompute = r.given.contains("--dt-recompute")
                        ? r.dtRecompute
                        : (comp && c.bodyKind == BodyKind::Cylinder ? 16 : 1);
    if (c.dtRecompute < 1) return bad(2, "--dt-recompute", "must be >= 1");

    // Geometry options.
    c.aspect = r.aspect;
    if (c.bodyKind == BodyKind::Lens && !(c.aspect > 1.0))
        return bad(2, "--aspect", "must be > 1 (the lens needs w > hh)");
    c.dimples = r.dimples;
    if (c.dimples < 0) return bad(2, "--dimples", "must be >= 0");
    if (c.dimples > 0 && c.bodyKind != BodyKind::Cylinder
        && c.bodyKind != BodyKind::Sphere)
        return bad(2, "--dimples", "only cylinder and sphere carry dimples");
    if (c.dimples > 0 && c.placement.D < 16 * c.h)
        return bad(2, "--dimples", "dimple pits are 5h wide; D must span at least 16 "
                                   "cells or they consume the body");
    c.bumps = r.bumps;
    if (c.bumps < 0) return bad(2, "--bumps", "must be >= 0");
    if (c.bumps > 0 && c.bodyKind != BodyKind::Floor)
        return bad(2, "--bumps", "the surface ripple belongs to --body floor");
    c.bumpH = r.bumpH;
    if (c.bumps > 0)
    {
        if (!(std::isfinite(c.bumpH) && c.bumpH >= 4 && c.bumpH <= 48))
            return bad(2, "--bump-h", "must be in [4,48] cells (base = height, "
                                      "63-degree flanks; under 4 the tip is "
                                      "sub-cell, over 48 it is a fence, not "
                                      "roughness)");
        if (c.bumps > int(0.65 * c.n / (2.0 * c.bumpH)))
            return bad(2, "--bumps", "spike pitch is under twice the tooth "
                                     "base; the teeth merge -- fewer bumps, "
                                     "shorter --bump-h, or more cells");
        if (c.placement.D < 4 * c.bumpH * c.h)
            return bad(2, "--bump-h", "spikes taller than D/4 are a fence "
                                      "field, not surface roughness -- lower "
                                      "--bump-h or --dr");
    }
    c.scale = r.scale;
    if (!(std::isfinite(c.scale) && c.scale >= 0))
        return bad(2, "--scale", "must be finite and >= 0 (0 = fit D)");
    c.pitch = r.pitch;
    c.yaw = r.yaw;
    c.robust = r.robust;
    if (c.bodyKind != BodyKind::Mesh)
        if (auto e = rejectGiven(r, {"--scale", "--pitch", "--yaw", "--robust"},
                                 "applies to mesh bodies only");
            e)
            return std::unexpected(*e);

    // Run length + output cadences.
    c.ft = r.ft;
    if (!finitePos(c.ft)) return bad(2, "--ft", "must be finite and > 0");
    // One flow-through is L/uInf in both physics (c.L already carries each
    // physics' own box convention). Formerly a `comp ? X : X` ternary.
    c.tEnd = c.ft * c.L / c.uInf;

    // Stationary mode. tEnd stops being the run length and becomes the
    // give-up cap: not converging by then is exit 8, never a silent answer.
    c.steadyTol = r.steady;
    if (c.steadyTol < 0 || !std::isfinite(c.steadyTol))
        return bad(2, "--steady", "must be finite and >= 0 (0 = time-march)");
    if (c.steadyTol > 0 && comp)
        return bad(2, "--steady",
                   "is incompressible-only; the compressible loop has acoustic "
                   "transients that a velocity residual cannot certify");
    if (c.steadyTol > 0 && c.cs > 0)
        return bad(2, "--steady",
                   "with --les is meaningless: a filtered field has no "
                   "stationary state to converge to");

    // Grid sequencing. Only the initial guess comes from the coarse level, so
    // the fine-grid fixed point is unchanged; the ladder just skips paying the
    // startup transient at full resolution.
    c.seqLevels = r.seq;
    if (c.seqLevels < 1 || c.seqLevels > 6)
        return bad(2, "--seq", "must be in [1,6] (1 = off)");
    if (c.seqLevels > 1)
    {
        if (c.steadyTol <= 0)
            return bad(2, "--seq", "requires --steady: without a convergence "
                                   "test there is nothing to advance a level on");
        // The coarsest level still has to resolve the body: same D >= 4h rule.
        const int nCoarse = r.n >> (c.seqLevels - 1);
        if (nCoarse < 16 || double(nCoarse) < 4.0 * c.dr)
            return bad(2, "--seq", "coarsest level would not resolve the body "
                                   "(needs n/2^(levels-1) >= max(16, 4*dr)); "
                                   "use fewer levels or a larger --n");
    }
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    c.vortMax = r.vortMax;
    if (!(std::isfinite(c.vortMax) && c.vortMax >= 0))
        return bad(2, "--vort-max", "must be finite and >= 0 (0 = auto scale)");
    c.vortAsinh = r.vortMap == 1;
    if (c.vortAsinh && !(c.vortMax > 0))
        return bad(2, "--vort-map", "asinh needs an explicit full-scale clip; "
                                    "set --vort-max (~ the peak |vorticity| in "
                                    "uInf/D, e.g. sqrt(Re))");
    c.speedDelta = r.speedMap == 1;
    c.tempFrames = r.tempFrames;
    if (c.tempFrames && !comp)
        return bad(2, "--temp", "the incompressible solver carries no energy "
                                "equation and no temperature field; use "
                                "--physics compressible");
    c.vortSigned = r.vortSigned;
    if (c.vortSigned && !comp)
        return bad(2, "--vort-signed", "incompressible frames are already "
                                       "signed z-vorticity; the flag selects "
                                       "it for 3D compressible runs");

    if (auto e = resolveEmit(c, {Emit::Stats, Emit::Bin}); e)
        return std::unexpected(*e);
    // PNG at nz > 1 renders the centre z-plane (same machinery the channel
    // already uses); the old flow-only nz == 1 restriction was conservative,
    // not load-bearing.

    // NOTE: qRef is deliberately NOT resolved here. It is 0.5 rho uInf^2 A
    // with A the frontal area MEASURED from the cut apertures, and validate()
    // touches no filesystem and builds no geometry, so it cannot know A. The
    // field that used to live here held 0.5 uInf^2 D span -- a placeholder
    // that --dry-run printed as if it were the number the run would divide by.
    // The real value comes from dns/geometry.hpp's referenceQ*(), once.

    c.meta = {c.grid.nx, c.grid.ny, c.grid.nz, c.grid.hx, c.grid.hy, c.grid.hz,
              c.placement.D, c.re, comp ? c.mach : 0.0, c.fdt, c.cs, c.lesLabel,
              c.bodyKind == BodyKind::Mesh    ? c.meshPath
              : c.bodyKind == BodyKind::Scene ? "scene:" + c.scenePath
                                              : name(c.bodyKind),
              name(c.far), name(c.physics), comp ? c.kick : 0.0,
              c.lesLabel ? name(c.sgsModel) : ""};
    return c;
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

    if (c.outPrefix.empty())
        return bad(2, "--out", "output prefix must not be empty");
    // webframes.cpp concatenates the prefix into a JSON manifest unescaped.
    if (c.outPrefix.find_first_of("\"\\\n\r\t") != std::string::npos)
        return bad(2, "--out", "output prefix must not contain quotes, backslashes "
                               "or control characters (it is embedded in JSON)");

    if (r.n < 1) return bad(2, "--n", "must be >= 1");
    if (!finitePos(r.re)) return bad(2, "--re", "must be finite and > 0");
    c.re = r.re;
    c.n = r.n;

    if (r.sub == Subcommand::Bench) return validateBench(r, std::move(c));
    if (r.sub == Subcommand::Tgv || r.sub == Subcommand::Hit)
        return validatePeriodic(r, std::move(c));
    if (r.sub == Subcommand::Channel) return validateChannel(r, std::move(c));
    if (r.sub == Subcommand::Conv) return validateConv(r, std::move(c));
    return validateFlow(r, std::move(c));
}

// ---------------------------------------------------------------------------

std::string Case::describe() const
{
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
            "  run        tt={:.6g} eddy turnovers (tEnd={:.6g})  kick={:.3g}  "
            "profile-dt={:.3g}\n"
            "  label      DNS{}\n"
            "  out        {}\n",
            grid.nx, grid.ny, grid.nz, grid.ny - 2,
            lxBox, 2.0, lzBox, grid.nz == 1 ? "  [native 2D]" : "",
            stretchBeta, chanH1, chanH1 * reTau, stretchGrowth,
            reTau, nu, mach, ubEst, pRef, cfl,
            wallIso ? "isothermal" : "ADIABATIC (no heat sink -- short "
                                     "transients only)",
            ft, tEnd, kick, profileDt,
            grid.nz == 1 ? " (2D: qualitative only, no vortex stretching)"
                         : "",
            outPrefix);
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
        name(far),
        open ? (farPot ? " (lateral open, potential-corrected)"
                       : " (lateral open)")
             : (farPot ? " (potential-corrected)" : ""),
        grid.nx, grid.ny, grid.nz, h, L, grid.nz == 1 ? "  [native 2D]" : "",
        placement.cx, placement.cy, placement.cz, placement.D,
        h > 0 ? placement.D / h : 0.0,
        re, nu, mach, cfl, cs,
        lesLabel ? (sgsModel == SgsModel::Sigma ? "LES sigma" : "LES smag")
                 : "DNS",
        ft, tEnd, fdt,
        steadyTol > 0 ? "stationary (stop at steady residual, tEnd = give-up cap)"
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
        const double Tinf = mach > 0 ? 1.0 / (1.4 * mach * mach) : 0.0;
        s += std::format(
            "  thermal    visc={}  Pr={:.6g}  Tinf={:.6g} box ({:.6g} K)\n"
            "             body wall {}{}  nscbc inflow holds {}\n",
            sutherland ? "sutherland" : "constant", prandtl, Tinf, tinfK,
            bodyIsothermal ? "isothermal" : "adiabatic",
            bodyIsothermal ? " Twall/Tinf=" + std::to_string(twallRatio)
                           : std::string(),
            nscbcThermalInflow ? "T" : "rho");
    }
    if (ogrid)
    {
        s += std::format(
            "  o-grid     BODY-FITTED: {} angular x {} radial (2 ghost rows)"
            " x {} spanwise\n"
            "             rFar={:.6g} ({:.6g} D)  beta={:.6g}  span={:.6g}\n"
            "             the wall is a MESH FACE -- no cut cells, no immersed"
            " body,\n"
            "             forces are an exact surface integral\n",
            grid.nx, grid.ny, grid.nz, ogridRFar,
            placement.D > 0 ? ogridRFar / placement.D : 0.0, ogridBeta,
            ogridSpan);
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
                "             {} [{:.4g},{:.4g}] blend {:.4g}: fine {:.3e} "
                "coarse {:.3e}\n",
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
        const double hFine = focalActive
                                 ? (grid.nz == 1
                                        ? std::min(focalHFine[0], focalHFine[1])
                                        : std::min({focalHFine[0], focalHFine[1],
                                                    focalHFine[2]}))
                                 : (grid.nz == 1 ? std::min(grid.hx, grid.hy)
                                                 : std::min({grid.hx, grid.hy,
                                                             grid.hz}));
        const double tRev = 2.0 * std::numbers::pi / std::abs(rotor.omega);
        const char* shaft = ax[rotorAxisIndex(rotor.axis)];
        s += std::format(
            "  rotor      SPINNING body: shaft along {} through the placement "
            "centre\n"
            "             tip Mach {:.4g} (freestream {:.4g}, sum {:.4g} < 1)  "
            "omega={:.6g}  R={:.6g}\n"
            "             t_rev={:.6g} -> {:.2f} revolutions in tEnd={:.6g}\n"
            "             azimuth {} bins (natural {} = 2 pi R / h_fine); "
            "geometry is\n"
            "             precomputed per bin and replayed -- the time loop "
            "never rebuilds it\n"
            "             wall is MOVING no-slip (omega x r, tip speed "
            "{:.6g}); run reports\n"
            "             thrust and shaft torque about {}\n"
            "             blade must be >= 4 fine cells thick (h_fine={:.3e}, "
            "D/h_fine={:.1f}):\n"
            "             >= {:.4g} in world units, or the two surfaces' "
            "mirror bands overlap\n",
            shaft, rotorTipMach, mach, rotorTipMach + mach, rotor.omega,
            rotor.radius, tRev, tRev > 0 ? tEnd / tRev : 0.0, tEnd, rotor.bins,
            rotorNaturalBins(rotor, hFine),
            std::abs(rotor.omega) * rotor.radius, shaft, hFine,
            hFine > 0 ? placement.D / hFine : 0.0, 4.0 * hFine);
        if (rotor.thicken > 0)
            s += std::format(
                "  blade      THICKENED: surface dilated outward by {:.4g} "
                "fine cells ({:.4g} world).\n"
                "             This is a DIFFERENT BODY -- chord grows by "
                "{:.4g}, edges round.\n"
                "             Report it with every result.\n",
                bladeThickenCells, rotor.thicken, 2.0 * rotor.thicken);
    }
    if (noslipY) s += "  walls      y walls no-slip (adiabatic mirror)\n";
    return s;
}

} // namespace dns
