#include "dns/cli.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
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
             {"--aspect", &c.aspect}, {"--dimples", &c.dimples},
             {"--scale", &c.scale}, {"--pitch", &c.pitch}, {"--yaw", &c.yaw},
             {"--mach", &c.mach}, {"--les", &c.les}, {"--cfl", &c.cfl},
             {"--sponge-cells", &c.spongeCells}, {"--fringe-axes", &c.fringeAxes},
             {"--nscbc-beta", &c.nscbcBeta}, {"--nscbc-strip", &c.nscbcStrip},
             {"--ramp", &c.ramp}, {"--kick", &c.kick}, {"--tol", &c.tol},
             {"--max-iters", &c.maxIters}, {"--dt-recompute", &c.dtRecompute},
             {"--steady", &c.steady},
             {"--fdt", &c.fdt}, {"--pdt", &c.pdt}, {"--wdt", &c.wdt},
             {"--planes", &c.planes}});
    if (c.sub == Subcommand::Tgv)
        add({{"--nu", &c.nu}, {"--cfl", &c.cfl}, {"--fdt", &c.fdt}});
    if (c.sub == Subcommand::Hit)
        add({{"--urms", &c.urms}, {"--nu", &c.nu}, {"--spin", &c.spin},
             {"--stats", &c.stats}, {"--fdt", &c.fdt}});
    if (c.sub == Subcommand::Bench)
        add({{"--steps", &c.steps}});

    int physicsSel = int(c.physics), farSel = int(c.far), backendSel = int(c.backend),
        poissonSel = int(c.poisson);
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
        enums.push_back({"--poisson",
                         {{"auto", int(PoissonKind::Auto)}, {"fft", int(PoissonKind::Fft)},
                          {"dct", int(PoissonKind::Dct)}, {"cg", int(PoissonKind::Cg)}},
                         &poissonSel});
    }

    // Boolean flags (no value).
    std::vector<std::pair<const char*, bool*>> flags{
        {"--quiet", &c.quiet}, {"--dry-run", &c.dryRun}};
    if (c.sub == Subcommand::Flow)
    {
        flags.push_back({"--robust", &c.robust});
        flags.push_back({"--open", &c.open});
        flags.push_back({"--pot", &c.farPot});
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
    c.poisson = PoissonKind(poissonSel);
    if (!c.given.contains("--out")) c.out = string(sub);
    return c;
}

// ---------------------------------------------------------------------------

std::string helpTop()
{
    return
R"(dns — incompressible & compressible DNS/LES on a MAC grid (CPU + HIP)

usage: dns <subcommand> [options]
       dns <subcommand> --help

subcommands:
  flow    a body in a uniform free stream: drag/lift, wake, slices
  tgv     Taylor-Green vortex — 2D analytic check or 3D Re1600 benchmark
  hit     forced homogeneous isotropic turbulence (SI-configured)
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

usage: dns flow --body BODY [grid] [physics] [far field] [output]

GRID
  --n N              cells along x (streamwise)                    [512]
  --ny N             cells along y                                  [=n]
  --nz N             cells along z; 1 = native 2D                    [1]
  --dr RATIO         domain height / body diameter D                 [8]

  Grid convention differs by --physics and is stated here once:
    incompressible: L = 2*pi, h = L/n, D = L/--dr  (box scales with n)
    compressible:   D = pi/4, box = --dr * D in y ((n/ny)*dr*D in x),
                    D spans ny/dr cells (dr 8, n 512 = the historical
                    8D x 8D box, bit-identical); --dr >= 6
  These are two genuinely different, separately validated conventions.
  Merging them relocates the body and rescales Re — a real past bug
  (measured cd 3.19). They are NOT unified by this recomposition.

BODY (exactly one; required)
  --body cylinder        z-aligned circle                    (2D or quasi-2D)
  --body sphere          ball                                          (3D)
  --body cube            45-deg rotated square prism              (2D or 3D)
  --body lens            two-disk intersection, sharp tips              (2D)
  --body FILE.stl|.gltf|.glb    triangle mesh, sdf via BVH       (2D or 3D)

  --aspect A         lens width / height, A > 1                    [3.5]
  --dimples N        golf dimples, 0 = smooth                        [0]
                     (cylinder -> 2D disks, sphere -> 3D caps)
  --scale S          mesh scale; 0 = fit max extent to D             [0]
  --robust           mesh sign by winding number (leaky meshes)
  --pitch DEG        DEPRECATED: rotates RAW MODEL AXES, not aircraft
  --yaw DEG          angles. Bake attitude with build/orient.py and
                     leave these at 0.

PHYSICS
  --physics incompressible|compressible          [incompressible]
  --re RE            Reynolds number on D                           [60]
  --mach M           Mach number (compressible only)               [0.2]
  --les CS           Smagorinsky constant; 0 = DNS                   [0]
                     full 3D strain in both physics; nz==1 reduces
                     exactly to the 2D model (see CLAUDE.md caveats)
  --cfl C            Courant target             [0.8 incomp / 0.4 comp]

FAR FIELD (--far, exactly one)
  --far tunnel       Dirichlet inflow + convective outlet + free-slip
                     lateral walls. Incompressible default. Pressure =
                     DCT-preconditioned CG on the confined box.
  --far fringe       fully periodic + cosine sponge on the boundaries
                     relaxing u -> uInf. FFT pressure. (was: --open)
  --far sponge       compressible streamwise freestream sponge +
                     free-slip lateral walls. Compressible default.
  --far nscbc        Poinsot-Lele characteristic boundaries.
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
                     1.0 blows up at t=35. Values >= 1 are rejected.
  --nscbc-strip N    absorbing strip inside the NSCBC rows; 0 = pure
                     characteristic. Required above Re 800; use ~ny/64
                     (pure NSCBC blows up at Re 2000, 2048^2).         [0]
  --pot              potential-corrected far field: targets/inlet are
                     freestream + wake source (strength from the
                     measured drag, EMA) + displacement doublet instead
                     of uniform uInf. Removes the finite-domain bias of
                     pinning u = uInf a few D from the body (measured:
                     8D corrected == 32D plain, comp cyl Re100).
                       compressible   --far nscbc: all targets
                       incompressible --far tunnel --open: the inlet
                     Rejected with any other far field.

START-UP (compressible)
  --ramp T           quiet start: IC at rest, freestream raised by
                     uniform Galilean boosts over [0,T]. 0 = impulsive
                     (legacy taper IC).                             [2.0]
  --kick A           deterministic solenoidal Gaussian trip 1 D behind
                     the body. Divergence-free, hence acoustically
                     silent; the quiet start is too symmetric to shed
                     without it.                                   [0.05]

SOLVER
  --poisson auto|fft|dct|cg    pressure solve; auto follows --far  [auto]
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

OUTPUT (--emit is repeatable; default = stats,bin)
  --emit stats       <out>_stats.csv (t,dt,K,cd,cl,cdf,cfl,maxDiv,
                     pcgIters,resid)
  --emit bin         <out>_{vort,speed,mask}.bin float32 mid-plane
  --emit png         png_<out>/{vort,speed}_%05d.png  (requires --nz 1)
  --emit web         web_<out>/ live browser frames + planes.json
  --emit volume      <out>_{vol,lam}.bin full 3D field per frame
  --emit slices      three centre planes instead of the volume
  --emit vtk         <out>_%05d.vtk legacy structured points
  --emit none        diagnostics only

  --fdt DT           .bin / volume frame interval                  [0.15]
  --pdt DT           png interval;  0 = follow --fdt                  [0]
  --wdt DT           web interval;  0 = follow --fdt                  [0]
  --planes N         web planes per axis: 1 = centre, 3 = +/- n/16    [3]
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
