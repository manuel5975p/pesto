#include "dns/cli.hpp"

#define TOML_EXCEPTIONS 0
#include <tomlplusplus/toml.hpp>

#include <algorithm>
#include <charconv>
#include <cstring>
#include <format>
#include <initializer_list>
#include <string_view>
#include <variant>
#include <vector>

namespace dns::cli {
namespace {

using std::string;
using std::string_view;

// --- scalar parsing ---------------------------------------------------------
// Full-consumption from_chars: "3x" and "" are errors, never a silent default.
// (The old apps used parseOr/atof, which returned the default on garbage.)
bool toDouble(string_view s, double& v)
{
    if (s.empty()) return false;
    auto [q, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    return ec == std::errc{} && q == s.data() + s.size();
}

bool toInt(string_view s, int& v)
{
    double d{};
    if (!toDouble(s, d)) return false;
    if (!(d >= -2147483648.0 && d <= 2147483647.0) || d != double(int(d))) return false;
    v = int(d);
    return true;
}

// --- option table -----------------------------------------------------------
struct Opt
{
    const char* name;
    // Target is one of: double*, int*, bool* (flag), string*, or a custom sink.
    std::variant<double*, int*, bool*, unsigned*, string*> target;
};

struct EnumOpt
{
    const char* name;
    std::vector<std::pair<const char*, int>> values;
    int* target;
};

string joinNames(const std::vector<string>& v)
{
    string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ", "; s += v[i]; }
    return s;
}

// Cheap common-prefix suggestion (no Levenshtein, by design).
string suggest(string_view bad, const std::vector<string>& known)
{
    size_t best = 0;
    const string* hit = nullptr;
    for (const auto& k : known)
    {
        size_t p = 0;
        while (p < bad.size() && p < k.size() && bad[p] == k[p]) ++p;
        if (p > best && p >= 4) { best = p; hit = &k; }
    }
    return hit ? " (did you mean '" + *hit + "'?)" : string{};
}

Emit emitOf(string_view s, bool& ok)
{
    ok = true;
    if (s == "stats") return Emit::Stats;
    if (s == "bin") return Emit::Bin;
    if (s == "png") return Emit::Png;
    if (s == "web") return Emit::Web;
    if (s == "shm") return Emit::Shm;
    if (s == "volume") return Emit::Volume;
    if (s == "slices") return Emit::Slices;
    if (s == "vtk") return Emit::Vtk;
    if (s == "error") return Emit::Error;
    if (s == "spectrum") return Emit::Spectrum;
    if (s == "snapshot") return Emit::Snapshot;
    if (s == "none") return Emit::None;
    ok = false;
    return Emit::None;
}

} // namespace

// ---------------------------------------------------------------------------

std::expected<RawConfig, CliError> parse(int argc, char** argv)
{
    RawConfig c;

    // No subcommand at all is a USAGE ERROR, not a help request: `dns` alone
    // used to print the help and exit 0, so a script that lost its arguments
    // reported success without running anything.
    if (argc < 2)
        return std::unexpected(CliError{
            helpTop() + "\ndns: a subcommand is required\n", false});
    const string_view sub(argv[1]);
    if (sub == "--help" || sub == "-h" || sub == "help")
        return std::unexpected(CliError{helpTop(), true});

    if (sub == "flow") c.sub = Subcommand::Flow;
    else if (sub == "tgv") c.sub = Subcommand::Tgv;
    else if (sub == "hit") c.sub = Subcommand::Hit;
    else if (sub == "bench") c.sub = Subcommand::Bench;
    else if (sub == "channel") c.sub = Subcommand::Channel;
    else
        return std::unexpected(CliError{
            "dns: unknown subcommand '" + string(sub) + "'\n"
            "dns: try 'dns --help'\n"});

    // Per-subcommand defaults that differ from RawConfig's.
    switch (c.sub)
    {
    case Subcommand::Flow: c.out = "flow"; c.ft = 20.0; break;
    case Subcommand::Tgv:  c.out = "tgv";  c.ft = 20.0; c.n = 64; c.nz = 0;
                           c.re = 1600.0; c.cfl = 0.9; break;
    case Subcommand::Hit:  c.out = "hit";  c.n = 256; c.nz = 0; break;
    case Subcommand::Bench: c.out = "bench"; c.n = 256; c.nz = 0; break;
    // KMM Re_tau 180 grid; interior wall-normal cells, spanwise 160.
    case Subcommand::Channel: c.out = "channel"; c.n = 192; c.ny = 128;
                              c.nz = 160; c.physics = Physics::Compressible;
                              c.stretch = 2.0; c.kick = 0.2; c.cfl = 0.4;
                              break;
    }

    // Numeric / string options valid in this subcommand.
    std::vector<Opt> opts{
        {"--out", &c.out},
        {"--n", &c.n},
        {"--nz", &c.nz},
        {"--re", &c.re},
        {"--ft", &c.ft},
        {"--seed", &c.seed},
    };
    auto add = [&](std::initializer_list<Opt> l) { opts.insert(opts.end(), l); };

    if (c.sub == Subcommand::Flow)
        add({{"--ny", &c.ny}, {"--dr", &c.dr}, {"--body", &c.body},
             {"--scene", &c.scene},
             {"--aspect", &c.aspect}, {"--dimples", &c.dimples},
             {"--bumps", &c.bumps}, {"--bump-h", &c.bumpH},
             {"--scale", &c.scale}, {"--pitch", &c.pitch}, {"--yaw", &c.yaw},
             {"--mach", &c.mach}, {"--les", &c.les}, {"--cfl", &c.cfl},
             {"--penalize", &c.penalize},
             {"--sponge-cells", &c.spongeCells}, {"--fringe-axes", &c.fringeAxes},
             {"--nscbc-beta", &c.nscbcBeta}, {"--nscbc-strip", &c.nscbcStrip},
             {"--stretch", &c.stretch}, {"--focal", &c.focal},
             {"--focal-ratio", &c.focalRatio},
             {"--focal-blend", &c.focalBlend},
             {"--focal-wake", &c.focalWake}, {"--focal-pad", &c.focalPad},
             {"--ramp", &c.ramp}, {"--kick", &c.kick}, {"--tol", &c.tol},
             {"--max-iters", &c.maxIters}, {"--dt-recompute", &c.dtRecompute},
             {"--steady", &c.steady}, {"--seq", &c.seq},
             {"--fdt", &c.fdt}, {"--pdt", &c.pdt}, {"--wdt", &c.wdt},
             {"--planes", &c.planes}, {"--png-down", &c.pngDown},
             {"--vort-max", &c.vortMax}});
    if (c.sub == Subcommand::Tgv)
        add({{"--nu", &c.nu}, {"--cfl", &c.cfl}, {"--fdt", &c.fdt}});
    if (c.sub == Subcommand::Hit)
        add({{"--urms", &c.urms}, {"--nu", &c.nu}, {"--spin", &c.spin},
             {"--stats", &c.stats}, {"--fdt", &c.fdt}});
    if (c.sub == Subcommand::Bench)
        add({{"--steps", &c.steps}});
    if (c.sub == Subcommand::Channel)
        add({{"--ny", &c.ny}, {"--re-tau", &c.reTau}, {"--mach", &c.mach},
             {"--stretch", &c.stretch}, {"--kick", &c.kick},
             {"--tt", &c.tt}, {"--lx", &c.lx}, {"--lz", &c.lzBox},
             {"--cfl", &c.cfl}, {"--dt-recompute", &c.dtRecompute},
             {"--profile-dt", &c.profileDt},
             {"--fdt", &c.fdt}, {"--pdt", &c.pdt}, {"--wdt", &c.wdt},
             {"--planes", &c.planes}, {"--png-down", &c.pngDown}});

    int physicsSel = int(c.physics), farSel = int(c.far), backendSel = int(c.backend),
        sgsSel = int(c.sgs);
    std::vector<EnumOpt> enums{
        {"--backend", {{"cpu", int(Backend::Cpu)}, {"gpu", int(Backend::Gpu)},
                       {"auto", int(Backend::Auto)}}, &backendSel},
    };
    if (c.sub == Subcommand::Flow || c.sub == Subcommand::Bench)
        enums.push_back({"--physics",
                         {{"incompressible", int(Physics::Incompressible)},
                          {"compressible", int(Physics::Compressible)}},
                         &physicsSel});
    if (c.sub == Subcommand::Flow)
    {
        enums.push_back({"--far",
                         {{"tunnel", int(FarField::Tunnel)},
                          {"fringe", int(FarField::Fringe)},
                          {"sponge", int(FarField::Sponge)},
                          {"nscbc", int(FarField::Nscbc)},
                          {"periodic", int(FarField::Periodic)}},
                         &farSel});
        enums.push_back({"--sgs",
                         {{"smag", int(SgsModel::Smagorinsky)},
                          {"sigma", int(SgsModel::Sigma)}},
                         &sgsSel});
        enums.push_back({"--vort-map",
                         {{"linear", 0}, {"asinh", 1}}, &c.vortMap});
        enums.push_back({"--speed-map",
                         {{"abs", 0}, {"delta", 1}}, &c.speedMap});
    }
    if (c.sub == Subcommand::Channel)
        enums.push_back({"--wall", {{"isothermal", 1}, {"adiabatic", 0}},
                         &c.wallIso});

    // Boolean flags (no value).
    std::vector<std::pair<const char*, bool*>> flags{
        {"--quiet", &c.quiet}, {"--dry-run", &c.dryRun}};
    if (c.sub == Subcommand::Flow)
    {
        flags.push_back({"--robust", &c.robust});
        flags.push_back({"--vort-signed", &c.vortSigned});
        flags.push_back({"--open", &c.open});
        flags.push_back({"--pot", &c.farPot});
        flags.push_back({"--cut-flux", &c.cutFlux});
        flags.push_back({"--noslip", &c.noslip});
    }

    std::vector<string> known;
    for (const auto& o : opts) known.emplace_back(o.name);
    for (const auto& e : enums) known.emplace_back(e.name);
    for (const auto& f : flags) known.emplace_back(f.first);
    known.emplace_back("--emit");
    known.emplace_back("--help");

    auto err = [&](const string& m) {
        return std::unexpected(CliError{"dns: " + m + "\ndns: try 'dns " + string(sub)
                                        + " --help'\n"});
    };

    for (int a = 2; a < argc; ++a)
    {
        const string_view s(argv[a]);
        if (s == "--help" || s == "-h")
            return std::unexpected(CliError{help(c.sub), true});

        auto fl = std::ranges::find_if(flags, [&](auto& p) { return s == p.first; });
        if (fl != flags.end()) { *fl->second = true; c.given.insert(string(s)); continue; }

        if (s == "--emit")
        {
            if (a + 1 >= argc) return err("option '--emit' requires a value");
            bool ok = false;
            const Emit e = emitOf(argv[++a], ok);
            if (!ok)
                return err("unknown --emit kind '" + string(argv[a])
                           + "' (stats, bin, png, web, volume, slices, vtk, error, "
                             "spectrum, snapshot, none)");
            c.emit.insert(e);
            c.given.insert("--emit");
            continue;
        }

        auto en = std::ranges::find_if(enums, [&](auto& e) { return s == e.name; });
        if (en != enums.end())
        {
            if (a + 1 >= argc) return err("option '" + string(s) + "' requires a value");
            const string_view v(argv[++a]);
            auto hit = std::ranges::find_if(en->values,
                                            [&](auto& p) { return v == p.first; });
            if (hit == en->values.end())
            {
                std::vector<string> names;
                for (auto& p : en->values) names.emplace_back(p.first);
                return err("invalid value '" + string(v) + "' for " + string(s)
                           + " (one of: " + joinNames(names) + ")");
            }
            *en->target = hit->second;
            c.given.insert(string(s));
            continue;
        }

        auto op = std::ranges::find_if(opts, [&](auto& o) { return s == o.name; });
        if (op != opts.end())
        {
            if (a + 1 >= argc) return err("option '" + string(s) + "' requires a value");
            const string_view v(argv[++a]);
            bool ok = true;
            std::visit([&](auto* t) {
                using T = std::remove_pointer_t<decltype(t)>;
                if constexpr (std::is_same_v<T, string>) *t = string(v);
                else if constexpr (std::is_same_v<T, double>) ok = toDouble(v, *t);
                else if constexpr (std::is_same_v<T, int>) ok = toInt(v, *t);
                else if constexpr (std::is_same_v<T, unsigned>)
                {
                    int q{};
                    ok = toInt(v, q) && q >= 0;
                    if (ok) *t = unsigned(q);
                }
                else ok = false;
            }, op->target);
            if (!ok)
                return err("option '" + string(s) + "' expects a number, got '"
                           + string(v) + "'");
            c.given.insert(string(s));
            continue;
        }

        if (s.starts_with("-"))
            return err("unknown option '" + string(s) + "'" + suggest(s, known));
        return err("unexpected argument '" + string(s)
                   + "' (all inputs are named options; the grid size is --n)");
    }

    c.backend = Backend(backendSel);
    c.physics = Physics(physicsSel);
    c.far = FarField(farSel);
    c.sgs = SgsModel(sgsSel);
    if (!c.given.contains("--out")) c.out = string(sub);
    return c;
}

// ---------------------------------------------------------------------------

std::expected<ConfigTokens, std::string> configTokens(std::string_view toml)
{
    const auto res = toml::parse(toml);
    if (!res)
    {
        const auto& e = res.error();
        return std::unexpected("config: TOML parse error: "
                               + std::string(e.description()) + " (line "
                               + std::to_string(e.source().begin.line) + ")");
    }

    ConfigTokens out;
    // One scalar to one CLI token; the strict full-consumption parse() then
    // re-reads it, so the formatting only has to round-trip.
    auto scalar = [](const toml::node& n) -> std::expected<string, string> {
        if (const auto s = n.value<string>()) return *s;
        if (n.is_integer()) return std::format("{}", *n.value<int64_t>());
        if (n.is_floating_point()) return std::format("{}", *n.value<double>());
        return std::unexpected("must be a string, number, or bool");
    };

    for (const auto& [key, v] : res.table())
    {
        string k(key.str());
        std::ranges::replace(k, '_', '-');
        if (k == "subcommand")
        {
            const auto s = v.value<string>();
            if (!s || s->empty())
                return std::unexpected("config: `subcommand` must be a "
                                       "non-empty string");
            out.subcommand = *s;
            continue;
        }
        const string flag = "--" + k;
        if (v.is_boolean())
        {
            // Flags can only be asserted; `x = false` looks like it disables
            // something and silently would not -- reject it instead.
            if (!v.value_or(false))
                return std::unexpected("config: `" + k + " = false` has no CLI "
                                       "form (flags cannot be negated); drop "
                                       "the key");
            out.args.push_back(flag);
            continue;
        }
        if (const toml::array* a = v.as_array())
        {
            for (const auto& item : *a)
            {
                auto s = scalar(item);
                if (!s)
                    return std::unexpected("config: `" + k + "` entries "
                                           + s.error());
                out.args.push_back(flag);
                out.args.push_back(std::move(*s));
            }
            continue;
        }
        auto s = scalar(v);
        if (!s) return std::unexpected("config: `" + k + "` " + s.error());
        out.args.push_back(flag);
        out.args.push_back(std::move(*s));
    }
    return out;
}

// ---------------------------------------------------------------------------

std::string helpTop()
{
    return
R"(dns — incompressible & compressible DNS/LES on a MAC grid (CPU + HIP)

usage: dns <subcommand> [options]
       dns <subcommand> --help
       dns [<subcommand>] --config FILE.toml [options]
           (TOML run config: `n = 512` = `--n 512`, `re_tau` = `--re-tau`,
            `emit = ["stats","png"]` repeats the flag, `noslip = true` asserts
            it, `subcommand = "channel"` may replace the positional; explicit
            CLI flags override the file)

subcommands:
  flow    a body in a uniform free stream: drag/lift, wake, slices
  tgv     Taylor-Green vortex — 2D analytic check or 3D Re1600 benchmark
  hit     forced homogeneous isotropic turbulence (SI-configured)
  channel wall-resolved turbulent channel flow (compressible, forced)
  bench   step-time and throughput benchmark

common options (every subcommand):
  --backend cpu|gpu|auto   compute backend                        [auto]
  --out PREFIX             output file/dir prefix           [<subcommand>]
  --seed N                 RNG seed for any stochastic input          [1]
  --quiet                  suppress the periodic progress line
  --dry-run                validate, print the resolved case, exit 0
  --help                   this text, or the subcommand's

conventions:
  * nz == 1 IS the 2D restriction — there is no separate 2D code path.
    Every z stencil cancels identically and w stays exactly 0.
  * --nz always means INTERIOR cells in z, in every subcommand. Ghost
    rings, where a boundary treatment needs them, are added internally.
  * Box units: uInf = 1, rho_inf = 1. Re and Mach are the physical inputs.
  * Exit codes: 0 ok, 1 CLI, 2 config, 3 input, 4 output, 5 no GPU,
    6 resource, 7 numerical failure, 8 solver non-convergence, 77 skipped.
)";
}

std::string help(Subcommand s)
{
    switch (s)
    {
    case Subcommand::Flow:
        return
R"(dns flow — a body in a uniform free stream

usage: dns flow (--body BODY | --scene FILE.toml) [grid] [physics] [far field]
                [output]

GRID
  --n N              cells along x (streamwise)                    [512]
  --ny N             cells along y                                  [=n]
  --nz N             cells along z; 1 = native 2D                    [1]
  --dr RATIO         domain height / body diameter D                 [8]

  Grid convention differs by --physics and is stated here once:
    incompressible: L = 2*pi, h = L/n, D = L/--dr  (box scales with n)
    compressible:   D = pi/4, box = --dr * D in y ((n/ny)*dr*D in x),
                    D spans ny/dr cells (dr 8, n 512 = the historical
                    8D x 8D box, bit-identical); --dr >= 4
  These are two genuinely different, separately validated conventions.
  Merging them relocates the body and rescales Re — a real past bug
  (measured cd 3.19). They are NOT unified by this recomposition.

BODY (exactly one; required)
  --body cylinder        z-aligned circle                    (2D or quasi-2D)
  --body sphere          ball                                          (3D)
  --body cube            45-deg rotated square prism              (2D or 3D)
  --body lens            two-disk intersection, sharp tips              (2D)
  --body fence           wall-mounted blocker: rectangle sealed to the
                         tunnel floor, top edge at D above it (thickness
                         D/8) -- --dr 10 blocks the bottom 10%. The 2D
                         shear-layer rig (incompressible --far tunnel)  (2D)
  --body floor           raised solid floor: slab over the FULL streamwise
                         extent, surface at D above the wall -- --dr 10
                         solidifies the bottom 10%; a no-slip flat plate
                         with its leading edge in the inflow plane. The
                         wall-layer rig (incompressible --far tunnel)   (2D)
  --body FILE.stl|.gltf|.glb    triangle mesh, sdf via BVH       (2D or 3D)
  --body none            body-free box (wall boundary layers; compressible
                         only -- pair with --noslip and/or --stretch)
  --scene FILE.toml      generalized scene instead of --body: several
                         objects ([[objects]] tables: cylinder/sphere/cube/
                         lens/mesh) placed in units of D about the standard
                         placement, plus an explicit [model] block
                         (dns | les + cs). Uses the generic sampled cut;
                         per-object cd/cl/cdf land in <out>_forces.csv
                         (incompressible). Spec + examples: docs/scene.md,
                         examples/*.toml

  --aspect A         lens width / height, A > 1                    [3.5]
  --dimples N        golf dimples, 0 = smooth                        [0]
                     (cylinder -> 2D disks, sphere -> 3D caps)
  --bumps N          floor trip spikes: N triangular teeth (height
                     --bump-h, base = height), evenly pitched over
                     10%..75% of the plate (smooth development length
                     ahead, smooth recovery length before the
                     outlet); 0 = smooth. --body floor only.          [0]
  --bump-h H         spike height in cells, [4,48]. The layer trips
                     when the teeth poke through the local boundary
                     layer (k/delta ~ 1); 8h buried under a
                     quiet-start BL stays laminar (measured).         [8]
  --scale S          mesh scale; 0 = fit max extent to D             [0]
  --robust           mesh sign by winding number (leaky meshes)
  --cut-flux         apertured momentum fluxes: the wall enters
                     advection/diffusion at its sub-cell position
                     instead of the binary staircase mask, removing
                     the stationary 2-grid vorticity the staircase
                     sheds upstream. Both physics, needs a body.
                     Intended default after validation.
  --penalize ETA     Brinkman solid-band decay: near-wall velocity
                     relaxes to rest on time scale ETA/(1-lambda),
                     SMOOTH along the surface (lambda = exact fluid
                     fraction) -- damps the jagged staircase slip
                     that --cut-flux's flux weighting cannot reach.
                     Dissipative and local; the drained momentum is
                     NOT in the cd budget -- measured, it HALVES cdf.
                     Never use it when friction drag is the physics
                     you are after.
                     Composable with --cut-flux. Try ETA ~ h/uInf. [0]
  --pitch DEG        DEPRECATED: rotates RAW MODEL AXES, not aircraft
  --yaw DEG          angles. Bake attitude with build/orient.py and
                     leave these at 0.

PHYSICS
  --physics incompressible|compressible          [incompressible]
  --re RE            Reynolds number on D                           [60]
  --mach M           Mach number (compressible only)               [0.2]
  --les CS           SGS model constant; 0 = DNS                     [0]
                     full 3D gradients in both physics; nz==1 reduces
                     exactly to the 2D model (see CLAUDE.md caveats)
  --sgs smag|sigma   SGS model, used with --les                   [smag]
                     smag:  nut = (Cs D)^2 |S|, Cs ~ 0.17. Nonzero in
                            laminar shear -- eats boundary layers.
                     sigma: Nicoud 2011, Csigma ~ 1.35. Vanishes in
                            2D/pure-shear/rotation states, y^3 wall
                            decay: the transition-capable choice.
  --cfl C            Courant target             [0.8 incomp / 0.4 comp]

FAR FIELD (--far, exactly one)
  --far tunnel       Dirichlet inflow + convective outlet + free-slip
                     lateral walls. Incompressible default. Pressure =
                     DCT-preconditioned CG on the confined box.
  --far fringe       fully periodic + cosine sponge on the boundaries
                     relaxing u -> uInf. FFT pressure. (was: --open)
  --far sponge       compressible streamwise freestream sponge +
                     free-slip lateral walls. Escape hatch / explicit fallback
                     ONLY -- never a default; it eats the near field in
                     a relaxation band. Ask for it by name.
  --far nscbc        Poinsot-Lele characteristic boundaries.
                     Compressible default.
  --far periodic     no far field (plain periodic box).

  --open             make the LATERAL faces open too.
                       --far tunnel  -> zero-gradient open faces; the
                                        lateral flux is free and enters
                                        the projection (sponge-free)
                       --far sponge  -> lateral rho/p-relax bands
                       --far nscbc   -> six-face characteristic
                     Rejected (not ignored) with fringe/periodic.
  --sponge-cells N   sponge / fringe band width; 0 = 10% of n         [0]
  --fringe-axes 2|3  fringe on x,y only, or x,y,z; 0 = auto
                     (2 when --nz 1, else 3)                          [0]
  --nscbc-beta B     Yoo-Im transverse correction, in [0,1)         [0.5]
                     Measured: 0 pumps a growing global mode;
                     1.0 diverges at t=35. Values >= 1 are rejected.
  --nscbc-strip N    absorbing strip inside the NSCBC rows; 0 = pure
                     characteristic. Required above Re 800; use ~ny/64
                     (pure NSCBC diverges at Re 2000, 2048^2).         [0]
  --pot              potential-corrected far field: targets/inlet are
                     freestream + wake source (strength from the
                     measured drag, EMA) + displacement doublet instead
                     of uniform uInf. Removes the finite-domain bias of
                     pinning u = uInf a few D from the body (measured:
                     8D corrected == 32D plain, comp cyl Re100).
                       compressible   --far nscbc: all targets
                       incompressible --far tunnel --open: the inlet
                     Rejected with any other far field.

MESH & WALLS (compressible)
  --stretch B        tanh wall-normal clustering along y (curvilinear
                     finite-volume path; dns/remap.hpp TanhStretchY).
                     B in (0,4]: cells shrink toward both y walls with
                     bounded growth; 0 = uniform. Requires --body none
                     and --far nscbc (default; the characteristic terms
                     evaluate the remap) or an explicit --far sponge.
                     --les works here (per-cell Delta). --dry-run
                     prints the resolved first-cell height + growth
                     ratio (the acoustic dt shrinks with the wall
                     cell).                                            [0]
  --focal BOX        focal clustering: a FINE plateau box, coarse far
                     field, one independent per-axis map (dns/remap.hpp
                     Focal1D / AxisRemap3). BOX = x0,x1,y0,y1,z0,z1 in
                     world units; a degenerate pair (x0 == x1) leaves
                     that axis uniform, so a tube around a body and its
                     wake is x = body..wake, y/z = body diameter. The
                     map is the EXACT identity within 4 cells of every
                     domain face (x also clears the NSCBC strip), so the
                     boundary treatment always sees a uniform grid.
                     BOX = auto derives the plateau from the body: its
                     AABB inflated by --focal-pad, extended downstream
                     by --focal-wake, shrunk if the box cannot hold it
                     (--dry-run prints what was resolved). Analytic
                     bodies only -- a mesh/scene AABB is not known
                     until the file is loaded, and validation touches no
                     filesystem. With a body the plateau must contain it
                     plus a 3-fine-cell guard, so the ghost wall's
                     mirror stencil never straddles the blend.
                     Compressible, --far nscbc or explicit sponge;
                     mutually exclusive with --stretch. --les is
                     allowed: the SGS filter width is the cell's OWN
                     Delta = vol^(1/3), so the fine plateau filters at
                     the fine spacing and the coarse far field at the
                     coarse one.
                     --dry-run prints the per-axis fine/coarse spacings,
                     the max adjacent growth ratio and the dt penalty.
  --focal-ratio R    coarse/fine cell-size ratio, (1,16].            [4]
  --focal-blend F    blend width per side as a fraction of the plateau
                     extent, (0,2]. Wider = gentler transition; the
                     validator rejects a max adjacent growth over 1.25
                     (the central flux has no dissipation and reflects
                     off a sharp coarse/fine step).                 [0.5]
  --focal-wake W     --focal auto: plateau extension downstream of the
                     body, in diameters.                             [5]
  --focal-pad P      --focal auto: AABB inflation, in diameters.   [0.25]
  --noslip           closed y walls become adiabatic NO-SLIP mirrors
                     (boundary layers grow on them) instead of
                     free-slip. Rejected with --open.

START-UP
  --ramp T           quiet start: IC at rest, freestream raised
                     smoothly over [0,T] (compressible: uniform
                     Galilean boosts; incompressible: ramped inlet /
                     fringe target). 0 = impulsive start. An impulsive
                     start rings up trapped dispersive standing waves
                     in the stagnation region that decay only on the
                     viscous time scale -- the quiet start never
                     creates them.               [2.0 comp / 0 incomp]
  --kick A           deterministic solenoidal Gaussian trip 1 D behind
                     the body. Divergence-free, hence acoustically
                     silent; the quiet start is too symmetric to shed
                     without it.                                   [0.05]

SOLVER
  --tol RMS          CG per-cell RMS residual target (incomp)      [1e-4]
  --max-iters N      CG iteration cap                               [500]
  --dt-recompute N   steps between stableDt recomputes (comp)        [16]

RUN LENGTH
  --ft N             flow-throughs to integrate                      [20]
  --steady TOL       stationary solve: march until the field stops
                     changing, then stop. TOL is on the residual
                     R = max|du/dt| * D / uInf^2 (dt- and grid-free);
                     1e-6 is tight, 1e-4 loose. --ft becomes the
                     give-up cap -- reaching it is exit 8, not a
                     silent answer. Incompressible, --les 0. Only
                     converges below the shedding onset (cyl Re<47).
                     0 = plain time-march.                            [0]
  --seq LEVELS       grid sequencing (requires --steady): converge on a
                     ladder of grids n/2^(LEVELS-1) .. n, prolonging each
                     solution as the next level's initial guess. Only the
                     initial condition changes, never the scheme, so the
                     converged fine-grid answer is the same one --seq 1
                     reaches -- just without paying the full transient at
                     the finest resolution. 1 = off.                  [1]

OUTPUT (--emit is repeatable; default = stats,bin)
  --emit stats       <out>_stats.csv -- the schema depends on --physics:
                       incompressible: t,dt,K,cd,cl,cdf,cfl,maxDiv,
                                       pcgIters,resid
                       compressible:   t,dt,cd,cl,cs,wmax
                     (dns channel writes a third schema; see its help.)
                     Parse by COLUMN NAME from the header row, never
                     by index.
  --emit bin         <out>_{vort,speed,mask}.bin float32 mid-plane
  --emit png         png_<out>/{vort,speed}_%05d.png  (nz > 1: the centre
                     z-plane)
  --emit web         web_<out>/ live browser frames + planes.json
  --emit shm         live frames in POSIX SHARED MEMORY (/dev/shm/dns_<out>),
                     nothing written to disk. Same planes as --emit web plus
                     the 3D field when nz > 1, published under a seqlock so a
                     slow or dead viewer can never stall the solver. This is
                     what `dns-ui` attaches to; pair it with --emit none for a
                     run that leaves no artefacts at all.
  --emit volume      <out>_{vol,lam}.bin full 3D field per frame
  --emit slices      three centre planes instead of the volume
  --emit vtk         <out>_%05d.vtk legacy structured points
  --emit none        diagnostics only

  --fdt DT           .bin / volume frame interval                  [0.15]
  --pdt DT           png interval;  0 = follow --fdt                  [0]
  --wdt DT           web interval;  0 = follow --fdt                  [0]
  --planes N         web planes per axis: 1 = centre, 3 = +/- n/16    [3]
  --png-down S       box-average PNG frames by S (must divide nx, ny);
                     1 = full res. e.g. 2048x1024 --png-down 2 -> 1024x512 [1]
  --vort-max V       vorticity/|curl| colour clip in uInf/D units; 0 = auto
                     (8 signed / 15 magnitude, calibrated at Re 60-200 --
                     high-Re runs saturate them; BL vorticity scales as
                     sqrt(Re) uInf/D)                                  [0]
  --vort-signed      3D compressible frames: signed z-vorticity on the
                     diverging map (neutral background, matches the
                     incompressible look) instead of |curl| sequential
  --vort-map M       png vorticity mapping: linear (clip at --vort-max) or
                     asinh (linear toe = vort-max/32, log above: BL and
                     weak wake filaments both read at high Re). Display
                     only -- bin/web data stay raw.             [linear]
  --speed-map M      abs: |u| on viridis, 0..1.6 uInf. delta: frames carry
                     dspeed = |u| - uInf on the diverging map (freestream
                     = background, overspeed red, deficit blue); bin
                     series becomes <out>_dspeed.bin.              [abs]
)";
    case Subcommand::Tgv:
        return
R"(dns tgv — Taylor-Green vortex
  --n N        cells per side                                       [64]
  --nz N       cells in z; 1 = the 2D analytic case                 [=n]
  --re RE      Reynolds number (3D benchmark uses 1600)           [1600]
  --nu NU      viscosity; overrides --re when > 0                    [0]
  --ft T       end time (TGV reports t, not flow-throughs)          [20]
  --cfl C      Courant target                                      [0.9]
  --emit stats t,K,dKdt,epsVisc CSV                                 [on]
  --emit vtk   final field as legacy VTK
  --emit error max-norm error vs the 2D analytic solution + KE ratio
               (requires --nz 1)
  --emit slices mid-plane speed/vorticity/pressure frames
)";
    case Subcommand::Hit:
        return
R"(dns hit — forced homogeneous isotropic turbulence
  Configured in SI (air viscosity, target u'), sized so h = 1.5 eta;
  runs in box units and reports both. Requires nz >= 4 (spectral IC).
  --n N            cells per side                                  [256]
  --urms U         target one-component rms velocity [m/s]          [60]
  --nu NU          kinematic viscosity [m^2/s]                 [1.5e-5]
  --spin TE        eddy turnovers of spin-up before statistics      [10]
  --stats TE       eddy turnovers of statistics                     [10]
  --emit stats | spectrum | snapshot | slices
)";
    case Subcommand::Channel:
        return
R"(dns channel — wall-resolved turbulent channel flow

Compressible DNS of a periodic plane channel driven by a constant streamwise
body force, on the curvilinear wall-clustered mesh. Wall units throughout:
half-height delta = 1, u_tau = 1, force density f = 1, nu = 1/Re_tau -- the
box momentum balance pins tau_wall = 1, so --re-tau IS the run's friction
Reynolds number (in the mean). Validation anchor: Kim-Moin-Moser Re_tau 180.

GRID (defaults = the KMM 192 x 128 x 160 box)
  --n N              streamwise cells (x, periodic)                 [192]
  --ny N             wall-normal FLUID cells (two mirror ghost rows
                     are added internally)                          [128]
  --nz N             spanwise cells (z, periodic); 1 = native 2D
                     (qualitative only -- no vortex stretching)     [160]
  --lx L             box length; 0 = 4 pi delta (KMM)                 [0]
  --lz L             box span;   0 = 4 pi/3 delta (KMM)               [0]
  --stretch B        tanh wall-normal clustering, [0,4]. --dry-run
                     prints the resolved first-cell y+ and growth
                     ratio (they set the acoustic dt).              [2.0]

PHYSICS
  --re-tau RE        friction Reynolds number (nu = 1/RE)           [180]
  --mach M           BULK Mach number: sets the background pressure
                     from the log-law bulk-velocity estimate. Above
                     ~0.3 compressibility contaminates comparison
                     with incompressible references.                [0.2]
  --wall isothermal|adiabatic                              [isothermal]
                     isothermal walls conduct the body-force power
                     out; adiabatic has NO heat sink (T climbs until
                     dt collapses) -- short transients only.
  --cfl C            Courant target                                 [0.4]

START-UP
  --kick A           deterministic solenoidal trip amplitude (times
                     the bulk-velocity estimate) on the laminar
                     Poiseuille IC; 0 = untripped (stays laminar)   [0.2]

RUN LENGTH
  --tt N             eddy turnovers (delta/u_tau = 1) to integrate   [30]
  --dt-recompute N   steps between stableDt recomputes               [16]

OUTPUT
  --emit stats       <out>_stats.csv (t,dt,ubulk,utauBot,utauTop,
                     retau,KE)                                       [on]
  --profile-dt DT    <out>_profiles.csv y-profile dump interval
                     (t,y,yplus,uplus,urms,vrms,wrms,uv,rho,T),
                     Favre-averaged over x,z; 0 = off               [1.0]
  --emit png|web|bin mid-plane frames (png renders the centre z-plane,
                     2D or 3D)
  --fdt/--pdt/--wdt  frame cadences as in dns flow
)";
    case Subcommand::Bench:
        return
R"(dns bench — step-time and throughput benchmark
  --n N            cells per side                                  [256]
  --steps N        timed steps after 3 warm-up steps                [20]
  --physics incompressible|compressible          [incompressible]
  Reports ms/step, Mcell-updates/s, VRAM in use.
)";
    }
    return {};
}

} // namespace dns::cli
