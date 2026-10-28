#include "dns/cli.hpp"
#include "dns/numparse.hpp"
#include "dns/rans_label.hpp"

#include <tomlplusplus/toml.hpp>

#include <algorithm>
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
// dns/numparse.hpp: full consumption, finite only. (The old apps used
// parseOr/atof, which returned the default on garbage.)
bool toDouble(string_view s, double& v) { return parseDouble(s, v); }

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

// An enum's EnumOpt values straight from its name table (index == value), so
// the CLI accepts exactly the spellings config.hpp defines, in its order.
std::vector<std::pair<const char*, int>>
enumValues(std::span<const char* const> names)
{
    std::vector<std::pair<const char*, int>> v;
    for (std::size_t i = 0; i < names.size(); ++i) v.emplace_back(names[i], int(i));
    return v;
}

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
    if (s == "surface") return Emit::Surface;
    if (s == "mesh") return Emit::Mesh;
    if (s == "none") return Emit::None;
    ok = false;
    return Emit::None;
}

// The --help texts; defined at the end of this file.
std::string help(Subcommand);
std::string helpTop();

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

    const auto sc = subcommandFromName(sub);
    if (!sc)
        return std::unexpected(CliError{
            "dns: unknown subcommand '" + string(sub) + "'\n"
            "dns: try 'dns --help'\n"});
    c.sub = *sc;

    // Per-subcommand defaults that differ from RawConfig's.
    // The --out default is the subcommand name, set after the flag loop.
    switch (c.sub)
    {
    case Subcommand::Flow: c.ft = 20.0; break;
    case Subcommand::Tgv:  c.ft = 20.0; c.n = 64; c.nz = 0;
                           c.re = 1600.0; c.cfl = 0.9; break;
    case Subcommand::Hit:  c.n = 256; c.nz = 0; break;
    case Subcommand::Bench: c.n = 256; c.nz = 0; break;
    // KMM Re_tau 180 grid; interior wall-normal cells, spanwise 160.
    case Subcommand::Channel: c.n = 192; c.ny = 128;
                              c.nz = 160; c.physics = Physics::Compressible;
                              c.stretch = 2.0; c.kick = 0.2; c.cfl = 0.4;
                              break;
    // A cube of cells over a layer twice as wide as it is tall, and a trip
    // small enough to leave the conduction state visible before it grows.
    case Subcommand::Conv: c.n = 128; c.ny = 64; c.nz = 128;
                           c.physics = Physics::Compressible; c.kick = 0.02;
                           c.cfl = 0.4; c.mach = 0.05; c.tt = 30;
                           c.profileDt = 0.0; break;
    // Air at 15 C.
    case Subcommand::Prop: c.nu = 1.5e-5; break;
    // The P1 flame: M 0.01 (inflated; docs/chem.md), 10 flame times.
    case Subcommand::Flame: c.ny = 4; c.physics = Physics::Compressible;
                            c.mach = 0.01; c.cfl = 0.4; c.tt = 10;
                            c.profileDt = 0.0; break;
    }

    // Numeric / string options valid in this subcommand. prop is host-only
    // and has no grid, so it takes none of the grid/run-length/seed flags.
    std::vector<Opt> opts{{"--out", &c.out}};
    if (c.sub != Subcommand::Prop)
        opts.insert(opts.end(), {{"--n", &c.n},
                                 {"--nz", &c.nz},
                                 {"--re", &c.re},
                                 {"--ft", &c.ft},
                                 {"--seed", &c.seed}});
    auto add = [&](std::initializer_list<Opt> l) { opts.insert(opts.end(), l); };

    if (c.sub == Subcommand::Flow)
        add({{"--ny", &c.ny}, {"--dr", &c.dr}, {"--body", &c.body},
             {"--scene", &c.scene},
             {"--aspect", &c.aspect}, {"--dimples", &c.dimples},
             {"--bumps", &c.bumps}, {"--bump-h", &c.bumpH},
             {"--scale", &c.scale}, {"--pitch", &c.pitch}, {"--yaw", &c.yaw},
             {"--mach", &c.mach}, {"--les", &c.les}, {"--cfl", &c.cfl},
             {"--tinf", &c.tinf}, {"--prandtl", &c.prandtl},
             {"--twall", &c.twall},
             {"--penalize", &c.penalize},
             {"--sponge-cells", &c.spongeCells}, {"--fringe-axes", &c.fringeAxes},
             {"--nscbc-beta", &c.nscbcBeta}, {"--nscbc-strip", &c.nscbcStrip},
             {"--nscbc-etain", &c.nscbcEtaIn},
             {"--nscbc-etain-shear", &c.nscbcEtaInShear},
             {"--nscbc-etalat", &c.nscbcEtaLat},
             {"--nscbc-sink", &c.nscbcSink},
             {"--nscbc-sink-amp", &c.nscbcSinkAmp},
             {"--stretch", &c.stretch}, {"--focal", &c.focal},
             {"--focal-ratio", &c.focalRatio},
             {"--focal-blend", &c.focalBlend},
             {"--focal-wake", &c.focalWake}, {"--focal-pad", &c.focalPad},
             {"--rotor-mach", &c.rotorMach}, {"--rotor-bins", &c.rotorBins},
             {"--rotor-span", &c.rotorSpan},
             {"--blade-thicken", &c.bladeThicken},
             {"--ogrid-far", &c.ogridFar}, {"--ogrid-beta", &c.ogridBeta},
             {"--ogrid-wedge", &c.ogridWedge}, {"--ogrid-trip", &c.ogridTrip},
             {"--init-from", &c.initFrom},
             {"--ramp", &c.ramp}, {"--kick", &c.kick}, {"--tol", &c.tol},
             {"--max-iters", &c.maxIters}, {"--dt-recompute", &c.dtRecompute},
             {"--steady", &c.steady}, {"--seq", &c.seq},
             {"--steady-steps", &c.steadySteps},
             {"--rans-nutinf", &c.ransNutInf}, {"--rans-kinf", &c.ransKInf},
             {"--rans-mutinf", &c.ransMutInf},
             {"--fdt", &c.fdt}, {"--pdt", &c.pdt}, {"--wdt", &c.wdt},
             {"--planes", &c.planes}, {"--png-down", &c.pngDown},
             {"--vort-max", &c.vortMax}});
    if (c.sub == Subcommand::Tgv)
        add({{"--nu", &c.nu}, {"--cfl", &c.cfl}, {"--fdt", &c.fdt}});
    if (c.sub == Subcommand::Hit)
        add({{"--urms", &c.urms}, {"--nu", &c.nu}, {"--spin", &c.spin},
             {"--stats", &c.stats}, {"--fdt", &c.fdt}});
    if (c.sub == Subcommand::Bench)
        add({{"--steps", &c.steps}, {"--nu", &c.nu}, {"--mach", &c.mach},
             {"--les", &c.les}});
    if (c.sub == Subcommand::Channel)
        add({{"--ny", &c.ny}, {"--re-tau", &c.reTau}, {"--mach", &c.mach},
             {"--stretch", &c.stretch}, {"--kick", &c.kick},
             {"--tt", &c.tt}, {"--lx", &c.lx}, {"--lz", &c.lzBox},
             {"--cfl", &c.cfl}, {"--dt-recompute", &c.dtRecompute},
             {"--profile-dt", &c.profileDt},
             {"--fdt", &c.fdt}, {"--pdt", &c.pdt}, {"--wdt", &c.wdt},
             {"--planes", &c.planes}, {"--png-down", &c.pngDown},
             {"--steady", &c.steady}, {"--steady-steps", &c.steadySteps},
             {"--les", &c.les}});
    if (c.sub == Subcommand::Conv)
        add({{"--ny", &c.ny}, {"--rayleigh", &c.rayleigh},
             {"--dtemp", &c.dtemp}, {"--aspect", &c.convAspect},
             {"--hot-patch", &c.hotPatch},
             {"--prandtl", &c.prandtl}, {"--tinf", &c.tinf},
             {"--mach", &c.mach}, {"--kick", &c.kick},
             {"--tt", &c.tt}, {"--cfl", &c.cfl},
             {"--dt-recompute", &c.dtRecompute},
             {"--profile-dt", &c.profileDt},
             {"--fdt", &c.fdt}, {"--pdt", &c.pdt}, {"--wdt", &c.wdt},
             {"--planes", &c.planes}, {"--png-down", &c.pngDown}});
    if (c.sub == Subcommand::Flame)
        add({{"--ny", &c.ny}, {"--mach", &c.mach},
             {"--expansion", &c.expansion}, {"--zeldovich", &c.zeldovich},
             {"--prandtl", &c.prandtl}, {"--tinf", &c.tinf},
             {"--cells-per-delta", &c.cellsPerDelta},
             {"--upstream", &c.upstream}, {"--downstream", &c.downstream},
             {"--inflow-speed", &c.inflowSpeed},
             {"--wrinkle", &c.wrinkle}, {"--wrinkle-modes", &c.wrinkleModes},
             {"--slot-width", &c.slotWidth}, {"--jet-speed", &c.jetSpeed},
             {"--coflow-speed", &c.coflowSpeed},
             {"--coflow-width", &c.coflowWidth},
             {"--coflow-temp", &c.coflowTemp},
             {"--pilot-width", &c.pilotWidth}, {"--pilot-temp", &c.pilotTemp},
             {"--pilot-speed", &c.pilotSpeed},
             {"--slot-edge", &c.slotEdge}, {"--length", &c.flameLength},
             {"--pulse", &c.pulse}, {"--pulse-freq", &c.pulseFreq},
             {"--stoich", &c.stoich}, {"--fuel-y", &c.fuelY},
             {"--ox-y", &c.oxY}, {"--damkohler", &c.damkohler},
             {"--cells-per-unit", &c.cellsPerUnit},
             {"--gravity", &c.gravity}, {"--ignition", &c.ignition},
             {"--nscbc-etain", &c.nscbcEtaIn}, {"--nscbc-sink", &c.nscbcSink},
             {"--nscbc-sink-amp", &c.nscbcSinkAmp},
             {"--tt", &c.tt}, {"--cfl", &c.cfl},
             {"--dt-recompute", &c.dtRecompute},
             {"--profile-dt", &c.profileDt},
             {"--fdt", &c.fdt}, {"--pdt", &c.pdt}, {"--wdt", &c.wdt},
             {"--planes", &c.planes}, {"--png-down", &c.pngDown}});
    if (c.sub == Subcommand::Flame) // --mode chamber (validator scopes them)
        add({{"--inject-speed", &c.chamber.injectSpeed},
             {"--coflow", &c.chamber.coflow},
             {"--chamber-height", &c.chamber.height},
             {"--chamber-length", &c.chamber.length},
             {"--nozzle-ratio", &c.chamber.nozzleRatio},
             {"--holder-height", &c.chamber.holder},
             {"--clearance", &c.chamber.clearance},
             {"--inject-ramp", &c.chamber.ramp}});
    if (c.sub == Subcommand::Prop)
        add({{"--planform", &c.planform}, {"--polar", &c.polar},
             {"--thin-cd0", &c.thinCd0}, {"--thin-alpha0", &c.thinAlpha0},
             {"--blades", &c.blades}, {"--hub-radius", &c.hubRR},
             {"--diameter", &c.diameter}, {"--rpm", &c.rpm}, {"--rho", &c.rho},
             {"--nu", &c.nu}, {"--sound", &c.sound}, {"--j", &c.advance},
             {"--dump-span", &c.dumpSpan}, {"--stations-bemt", &c.stationsBemt},
             {"--panels", &c.panels}, {"--wake-length", &c.wakeLength},
             {"--align-iters", &c.alignIters}, {"--align-tol", &c.alignTol},
             {"--seg-deg", &c.segDeg}, {"--core", &c.core},
             {"--stall-visc", &c.stallVisc}});

    int physicsSel = int(c.physics), farSel = int(c.far), backendSel = int(c.backend),
        sgsSel = int(c.sgs);
    int abcSel = int(c.nscbcAbc);
    std::vector<EnumOpt> enums;
    // prop never touches a GPU, so it has no backend to choose.
    if (c.sub != Subcommand::Prop)
        enums.push_back({"--backend", {{"cpu", int(Backend::Cpu)}, {"gpu", int(Backend::Gpu)},
                                       {"auto", int(Backend::Auto)}}, &backendSel});
    if (c.sub == Subcommand::Prop)
    {
        enums.push_back({"--method", enumValues(prop::methodNames()), &c.propMethod});
        enums.push_back({"--hub", enumValues(prop::hubModelNames()), &c.liftHub});
        enums.push_back({"--wake-pitch", enumValues(prop::wakePitchNames()), &c.wakePitch});
    }
    if (c.sub == Subcommand::Flow || c.sub == Subcommand::Bench
        || c.sub == Subcommand::Conv || c.sub == Subcommand::Flame)
        enums.push_back({"--physics", enumValues(physicsNames()), &physicsSel});
    if (c.sub == Subcommand::Flame)
    {
        enums.push_back({"--mode",
                         {{"premixed", 0}, {"bunsen", 1}, {"diffusion", 2},
                          {"candle", 3}, {"chamber", kFlameModeChamber}},
                         &c.flameMode});
        enums.push_back({"--reaction", {{"on", 1}, {"off", 0}},
                         &c.chamber.reaction});
    }
    if (c.sub == Subcommand::Flow || c.sub == Subcommand::Bench
        || c.sub == Subcommand::Channel)
        enums.push_back({"--sgs", enumValues(sgsModelNames()), &sgsSel});
    if (c.sub == Subcommand::Flow)
    {
        enums.push_back({"--far", enumValues(farFieldNames()), &farSel});
        enums.push_back({"--nscbc-abc", enumValues(nscbcAbcNames()), &abcSel});
        enums.push_back({"--rotor-axis",
                         {{"x", 0}, {"y", 1}, {"z", 2}}, &c.rotorAxis});
        enums.push_back({"--rotor-frame",
                         {{"moving", 0}, {"mrf", 1}}, &c.rotorFrame});
        enums.push_back({"--vort-map",
                         {{"linear", 0}, {"asinh", 1}}, &c.vortMap});
        enums.push_back({"--speed-map",
                         {{"abs", 0}, {"delta", 1}}, &c.speedMap});
    }
    if (c.sub == Subcommand::Flow || c.sub == Subcommand::Channel)
        enums.push_back({"--wall", {{"isothermal", 1}, {"adiabatic", 0}},
                         &c.wallIso});
    // RANS closures: only the BUILT ones are registered, so --help and the
    // "one of:" message stay truthful (docs/rans.md, phases).
    if (!ransBuilt(c.sub).empty())
    {
        std::vector<std::pair<const char*, int>> v;
        for (const RansModel m : ransBuilt(c.sub))
            v.emplace_back(ransCliNames()[std::size_t(m)], int(m));
        enums.push_back({"--rans", std::move(v), &c.rans});
        enums.push_back({"--steady-scheme", enumValues(steadySchemeNames()),
                         &c.steadyScheme});
    }
    if (c.sub == Subcommand::Flow || c.sub == Subcommand::Conv
        || c.sub == Subcommand::Flame)
        enums.push_back({"--visc", {{"constant", 0}, {"sutherland", 1}},
                         &c.visc});
    if (c.sub == Subcommand::Conv)
        enums.push_back({"--sides", {{"periodic", 0}, {"closed", 1}},
                         &c.sides});

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
        flags.push_back({"--nscbc-tinflow", &c.nscbcTinflow});
        flags.push_back({"--ogrid", &c.ogrid});
        flags.push_back({"--temp", &c.tempFrames});
    }
    if (c.sub == Subcommand::Flow || c.sub == Subcommand::Channel)
        flags.push_back({"--compensated", &c.compensated});
    if (c.sub == Subcommand::Flame) {
        flags.push_back({"--nscbc-tinflow", &c.nscbcTinflow});
        flags.push_back({"--quiescent", &c.quiescent});
    }
    if (c.sub == Subcommand::Prop) flags.push_back({"--selftest", &c.selftest});

    std::vector<string> known;
    for (const auto& o : opts) known.emplace_back(o.name);
    for (const auto& e : enums) known.emplace_back(e.name);
    for (const auto& f : flags) known.emplace_back(f.first);
    if (c.sub != Subcommand::Prop) known.emplace_back("--emit");
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

        if (s == "--emit" && c.sub != Subcommand::Prop)
        {
            if (a + 1 >= argc) return err("option '--emit' requires a value");
            bool ok = false;
            const Emit e = emitOf(argv[++a], ok);
            if (!ok)
                return err("unknown --emit kind '" + string(argv[a])
                           + "' (stats, bin, png, web, volume, slices, vtk, error, "
                             "spectrum, snapshot, surface, mesh, none)");
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
    c.nscbcAbc = NscbcAbc(abcSel);
    // prop's --out is an optional CSV path: absent = no file.
    if (!c.given.contains("--out") && c.sub != Subcommand::Prop) c.out = string(sub);
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

namespace {

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
  conv    Rayleigh-Bénard convection: heated floor, cooled lid, gravity
  flame   single-step flames: 1D premixed, 2D slot Bunsen, 2D jet diffusion
  bench   step-time and throughput benchmark
  prop    propeller thrust curve by BEMT or lifting line (host-only, no GPU)

common options (every subcommand; prop takes no --backend/--seed):
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
  --yaw DEG          angles. Bake attitude into the mesh and leave
                     these at 0.

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
  --nscbc-abc M      where the outflow's transverse coefficient comes
                     from. The exact value for a wave at incidence t is
                     cos t/(1+cos t); a fixed --nscbc-beta is its t = 0
                     value 1/2, and these derive it from the data.
                     Normal incidence is unchanged bit-for-bit. [yooim]
                       yooim     the fixed --nscbc-beta (default)
                       adaptive  per face cell, from the outgoing /
                                 transverse amplitude ratio; every open
                                 face, any mesh
                       modal     per transverse DCT mode of the whole
                                 outflow plane (the free-slip laterals
                                 are that basis exactly); x-hi only,
                                 uniform mesh, walled laterals
                     Measured (docs/nscbc.md): matches the best hand-
                     tuned constant beta to within 6% out to 60 deg
                     without being told the angle -- but the prize is
                     small, and in an all-open box --nscbc-etain is
                     worth ~280x what this is.
  --nscbc-etain E    inflow relaxation stiffness, > 0                 [100]
                     100 resists a body's blockage sag (etaIn 2 biased
                     cd 30% low at the Re60 cylinder) but IS an acoustic
                     mirror: measured 81% amplitude returned on a normal
                     wave, and 11% of a radial pulse's energy kept in an
                     open box against 0.04% at 0.5. With no freestream
                     to sag (a static rotor) there is nothing to buy.
  --nscbc-strip N    absorbing strip inside the NSCBC rows; 0 = pure
                     characteristic. Required above Re 800; use ~ny/64
                     (pure NSCBC diverges at Re 2000, 2048^2).         [0]
  --nscbc-etain-shear E
                     inflow rate for the lambda = u family (entropy +
                     shear) alone, > 0; default: follow --nscbc-etain.
                     The 100 default is the L5 sag guard -- an argument
                     about the NORMAL velocity anchor -- and clamping
                     entropy and shear at the same rate makes the
                     inflow a mirror for turbulence too. Soften this
                     one to make the inflow vortically transparent
                     while the freestream stays pinned.
  --nscbc-etalat E   lateral (open-face) incoming relax rate, > 0     [0.05]
                     The relax writes tangential velocity onto the face
                     under acoustic load -- a vorticity source that
                     SCALES with E (measured 0.20/0.80/2.16 of the
                     incident scale at 0.05/0.25/1.0). Soft is vortical
                     transparency; stiff is a wall in disguise. Needs
                     --open (walled laterals never read it).
  --nscbc-sink N     scale-selective damping band of N cells inside the
                     open faces (x always; y/z with --open); 0 = off.
                     Damps only sub-6dx content -- the waves whose group
                     velocity c cos(kh) is <= 0, which never reach a
                     face and which no characteristic closure can act
                     on (measured spectrum: docs/nscbc.md). Resolved
                     scales pass (< 2 permille at 16dx); there is no
                     freestream reference, so nothing sags and nothing
                     reflects. 16 is the measured default width.       [0]
  --nscbc-sink-amp A sink strength: sigma_max = A c/h. (0, 2]; 2 is the
                     SSP-RK3 stability cap at cfl 0.4.               [1.0]
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
                     PHYSICAL (world) units, the frame the body is
                     placed in: that region gets the fine spacing. The
                     fine cells are paid for out of --n, so a plateau
                     costs --focal-ratio x its width in cells and a box
                     the grid cannot afford is rejected. A degenerate
                     pair (x0 == x1) leaves
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
  --focal-blend F    physical blend width per side as a fraction of the
                     physical plateau extent, (0,2]. Wider = gentler
                     transition; the
                     validator rejects a max adjacent growth over 1.25
                     (the central flux has no dissipation and reflects
                     off a sharp coarse/fine step).                 [0.5]
  --focal-wake W     --focal auto: plateau extension downstream of the
                     body, in diameters.                             [5]
  --focal-pad P      --focal auto: AABB inflation, in diameters.   [0.25]
  --noslip           closed y walls become adiabatic NO-SLIP mirrors
                     (boundary layers grow on them) instead of
                     free-slip. Rejected with --open.

ROTOR (compressible; the body SPINS about a fixed shaft)
  --rotor-mach M     TIP Mach number: the blade tip travels at M times the
                     freestream sound speed, so omega = M*c/(D/2). This is
                     the ONE switch -- at 0 the run is bit-for-bit the
                     static solver, geometry is built once and no azimuth
                     table exists. Compressible --body/--scene runs only.
                     The wall is a MOVING no-slip surface (velocity
                     omega x r), and the run reports thrust and shaft
                     torque alongside cd/cl.                            [0]
  --rotor-axis A     shaft axis, x|y|z. The blade sweeps the plane normal
                     to it and thrust acts along it; x is the freestream
                     direction, i.e. an axial-flow rotor.               [x]
  --rotor-frame F    how the rotation reaches the solver.          [moving]
                       moving  the geometry itself turns: azimuth table,
                               bin replay, time-accurate blade passage.
                       mrf     MULTIPLE REFERENCE FRAME, the frozen-rotor
                               APPROXIMATION: geometry static at azimuth
                               0, wall velocity omega x r, and inside a
                               blended cylinder around the swept volume
                               the equations become the rotating-frame
                               ones (relative transport + Coriolis
                               source, absolute-velocity form). No
                               azimuth table (rejects --rotor-bins), no
                               bin swaps, and the blade never moves in
                               the frames -- the flow it reports is the
                               steady-in-the-blade-frame one, EXACT for
                               an axisymmetric body (pinned in gpu_mrf)
                               and an approximation for a real blade.
                               Thrust/torque columns are unchanged.
  --rotor-bins N     azimuth bins. Geometry is precomputed at N evenly
                     spaced angles and replayed, so the time loop never
                     rebuilds it. 0 = the NATURAL count 2 pi R / h_fine:
                     occupancy can only change when something crosses a
                     cell boundary and the tip is fastest, so more bins
                     than that only cost memory. Fewer bins quantize the
                     motion and inject an impulse at (bins * rev/s).     [0]
  --rotor-span S     blade half-extent ALONG the shaft, in diameters.
                     With the tip radius this bounds the SWEPT volume
                     from flags alone, which is what lets --focal auto
                     place a plateau around a rotor whose mesh the
                     validator may not read.                         [0.25]
  --blade-thicken C  dilate the body's surface outward by C FINE cells
                     (sdf - C*h, exactly a Minkowski sum with a ball, so
                     the field stays a true distance and convex edges
                     round rather than distort). This CHANGES THE BODY
                     and is reported as such. It exists because the ghost
                     wall needs a mirror band at least two cells thick on
                     each side: a blade under 4 cells thick has its two
                     surfaces reading through each other. The run reports
                     the deep-solid cell count -- 0 means the blade is
                     nowhere thicker than the band.                     [0]

BODY-FITTED O-GRID (compressible)
  --ogrid            wrap the body in an ANNULAR body-fitted mesh instead
                     of immersing it in a Cartesian box. The inner mesh
                     boundary IS the body surface, so there are no cut
                     cells and no ghost staircase: the wall is resolved
                     exactly, forces and the wall heat flux are exact
                     surface integrals, and the sub-cell boundary layer
                     that forces the upstream ripples is gone.
                     The domain becomes the DISC of radius --ogrid-far
                     around the body, so --n is the ANGULAR cell count
                     and --ny the RADIAL one (two of which are ghost
                     rows). Rejects --focal/--stretch/--cut-flux/
                     --penalize/--open/--far/--dr/--seq: those all
                     describe a box with a body immersed in it. The
                     far boundary is the O-grid's own characteristic
                     row, not NSCBC: every --nscbc-* flag and --pot is
                     rejected (they would be ignored), and Re > 800 only
                     warns (that far row is unmeasured there).
                     Bodies with a z-invariant cross-section (cylinder,
                     lens, cube, z-invariant mesh) are extruded in z.
                     `--body sphere` REVOLVES the mesh about x: --n is
                     then the MERIDIONAL cell count over theta in [0, pi]
                     (>= 16), and --nz 1 is the AXISYMMETRIC WEDGE (one
                     real azimuthal plane of --ogrid-wedge degrees; the
                     axis faces are degenerate, the cells beside them
                     read the across-axis continuation). Labelled
                     "[axisymmetric wedge ...]": no swirl, no 3D wake
                     modes. --nz N > 1 is the FULL REVOLUTION: N
                     azimuthal cells (even, >= 4; the axis ghosts copy
                     the cell across the axis, k + N/2), a polygonal
                     body of frontal area sin(2pi/N)/(2pi/N) of the
                     circle's (warned when 1% or more short), explicit
                     steady or time-accurate (no --steady-scheme
                     implicit, no --les). Default --nz for a sphere is
                     --n.
  --ogrid-far R      far-boundary radius, in diameters.               [20]
  --ogrid-beta B     radial tanh clustering toward the wall; 0 =
                     uniform. Higher = thinner first cell.             [3]
  --ogrid-wedge DEG  the revolved sphere's wedge angle, (0, 10].      [5]
  --ogrid-trip A     every O-grid but the axisymmetric wedge: a solenoidal
                     Gaussian vortex trip (strength A U, width D/4, 1 D
                     behind the body, D/4 off the axis; extruded: centred
                     in the span) in the initial state -- with --steady,
                     in the initial guess. The O-grid's own --kick: the
                     uniform start is exactly up-down symmetric, so a
                     cylinder sheds only once rounding has grown the
                     mode; this starts it. The full-revolution mesh and
                     an axisymmetric start are EXACTLY symmetric under a
                     rotation by pi about the axis, so the odd azimuthal
                     modes (the m = 1 lift mode) cannot grow from
                     rounding at all; this seeds them. A stable steady
                     state is unchanged. 0 = none, exactly the untripped
                     march (on the full revolution warned at Re >= 200
                     or with --rans). Printed in the label.
                                     [0.05 full revolution / 0 otherwise]
  --init-from PREFIX full revolution only: start from the converged
                     WEDGE run PREFIX (its PREFIX_final.bin, from --emit
                     mesh), revolved over the N planes (velocity rotated
                     per plane), then the trip. PREFIX_mesh.bin must
                     carry the same meridional mesh (--n, --ny,
                     --ogrid-beta, --ogrid-far) and the file the same
                     closure scalars and freestream pressure (--mach):
                     a mismatch is an input error (exit 3) by name.
                     Implies --ramp 0. Labelled "started from the
                     wedge PREFIX".                                [none]

RANS (compressible; an ensemble-mean MODEL, never a DNS; labelled
"RANS <variant> -- <mode>" everywhere, docs/rans.md "dns flow")
  --rans none|sa-neg|sst-2003m|sst-vm
                     turbulence closure, on the body-fitted O-grid
                     (--ogrid) or on the IMMERSED ghost wall (uniform or
                     --focal box; labelled "[immersed ghost wall:
                     first-order wall position, NOT body-fitted]" -- a
                     staircase wall, never a code-to-code number when
                     under-resolved). The wall is resolved (wall
                     functions do not exist here): the run warns when
                     the first-cell y+ estimate is above 1. Without
                     --steady: URANS (explicit SSP-RK3, time-accurate).
                     Rejects the incompressible solver, --les/--sgs (one
                     eddy viscosity), any --rotor-*, --scene,
                     --penalize, --cut-flux, --pot, --seq, --stretch,
                     --noslip and wall-mounted bodies.             [none]
  --rans-nutinf R    SA freestream nu~/nu (TMR: 3 to 5).                [3]
  --rans-kinf K      SST freestream k/U^2 (TMR's 9e-9 c^2 at M 0.2: Tu
                     0.0387%), in FREESTREAM units so a Mach study keeps
                     the inflow turbulence.                     [2.25e-7]
  --rans-mutinf R    SST freestream mu_t/mu; sets omega = rho k/(R mu)
                     (TMR).                                       [0.009]

THERMAL (compressible only; every knob is off by default, and a run
that names none of them is bit-for-bit the constant-viscosity,
adiabatic-wall solver every published anchor was measured on)
  --visc constant|sutherland
                     viscosity law. `sutherland` makes mu and the
                     conductivity kappa = mu cp / Pr functions of the
                     local temperature (air, S = 110.4 K); the
                     face value uses the face-average T, so the scheme
                     stays exactly conservative.               [constant]
  --tinf K           freestream STATIC temperature in kelvin. The
                     solver works in box units where T carries the
                     scale 1/(gamma M^2); this is the one dimensional
                     anchor Sutherland's law needs, and nothing else
                     reads it.                                  [288.15]
  --prandtl P        molecular Prandtl number.                    [0.72]
  --wall adiabatic|isothermal
                     the BODY surface condition. Adiabatic is the
                     historical wall (zero wall-normal dT/dn).
                                                              [adiabatic]
  --twall RATIO      isothermal wall temperature, as a ratio Twall/Tinf.
                     Requires --wall isothermal.                     [1]
  --nscbc-tinflow    NSCBC entropy inflow holds the TEMPERATURE
                     (rho -> p/(R Tinf) at the local pressure) instead
                     of the density. The two agree at the fixed point;
                     the density form lets the inflow temperature drift
                     with the incoming pressure, which is wrong once
                     transport depends on T or a wall exchanges heat.

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
                     without it. Rejected with --ogrid (never applied
                     there): use --ogrid-trip.                     [0.05]

SOLVER
  --tol RMS          CG per-cell RMS residual target (incomp)      [1e-4]
  --max-iters N      CG iteration cap                               [500]
  --dt-recompute N   steps between stableDt recomputes. The dt reduction
                     is one mandatory device reduction and host sync per
                     step; N > 1 amortizes it, with a safety margin on
                     the reused dt. Default 16 on the compressible
                     cylinder (a validated perf fix), 1 everywhere else --
                     raising it CHANGES THE TRAJECTORY (not the
                     statistics).                                 [16 / 1]

RUN LENGTH
  --ft N             flow-throughs to integrate                      [20]
  --steady TOL       stationary solve; TWO meanings by --physics.
                     INCOMPRESSIBLE: march until the field stops
                     changing, then stop. TOL is on the residual
                     R = max|du/dt| * D / uInf^2 (dt- and grid-free);
                     1e-6 is tight, 1e-4 loose. --ft becomes the
                     give-up cap -- reaching it is exit 8, not a
                     silent answer. --les 0. Only converges below the
                     shedding onset (cyl Re<47).
                     COMPRESSIBLE (--ogrid with or without --rans, or
                     immersed with --rans): PSEUDO-time, stopped on the
                     integrals (cd, cdf, wake deficit, turbulence; the
                     immersed wall: cd and turbulence) -- TOL is their
                     relative remaining change, in [1e-6, 1). The FP32
                     state jitters the integrals (3e-7..5e-5 relative,
                     measured), so the stop is JITTER-FLOORED: the trend
                     of block means < TOL, cd/cdf with sqrt(2) x their
                     jitter added, the other series' jitter <= 2^-10;
                     the run prints each series' trend and jitter
                     (docs/rans.md). Not time-accurate: rejects --ft,
                     --ramp, --kick, --seq and time-keyed frames;
                     --steady-steps is the cap.
                     0 = plain time-march.                            [0]
  --steady-steps N   compressible --steady iteration cap; reaching it is
                     exit 8.                                     [1000000]
  --steady-scheme explicit|implicit
                     compressible --steady pseudo-time scheme: explicit
                     local time stepping, or implicit Newton-Krylov
                     (--ogrid at --nz 1 or the revolved wedge only, not
                     the full revolution: JFNK
                     GMRES(40) with a line Gauss-Seidel first-order
                     Rusanov preconditioner, CFL 10 x2 -> 1e8; cap
                     default 500). Same fixed point; certifies the
                     cylinder (laminar/SA/SST) in 70-180 Newton steps,
                     slower than explicit, and does not converge from a
                     cold start on the Re 1e4 wedge (docs/rans.md).
                                                               [explicit]
  --compensated      compressible time march on the compensated state
                     (FP32 hi + lo per field, the explicit --steady
                     march's), so an update below ulp(q)/2 is kept
                     instead of rounded away. Measured: the plain URANS
                     dt ladder DIVERGES as dt -> 0 (FP32 stagnation),
                     the compensated one converges. Changes the bits;
                     ~1.15-1.2x per step. A no-op on explicit --steady;
                     rejects --steady-scheme implicit and --penalize.
                     (docs/rans.md, "Time accuracy")             [off]
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
                     (--ogrid adds heat; --rans appends cdp,cdf,mutMax,
                     yplusMax,yplusMean,floorHits -- immersed: mutMax,
                     yplusMax,yplusMean,floorHits; compressible --steady
                     appends iter,resid,resU,resE,resT,remaining, and
                     --steady-scheme implicit cfl,alpha,rejects,krylov.)
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
                     what `dns-ui` attaches to.
  --emit volume      <out>_{vol,lam}.bin full 3D field per frame
  --emit slices      three centre planes instead of the volume
  --emit vtk         <out>_%05d.vtk legacy structured points
  --emit mesh        <out>_mesh.bin (the O-grid's nodes and d1, written
                     before the run) and <out>_final.bin (FP64 rho, u, v,
                     w, p, T, the RANS scalars, mu_t, d at the end);
                     --ogrid only (docs/viewer-format.md). The OpenFOAM
                     parity tools build an identical polyMesh from it.
  --emit surface     <out>_surface.csv once at the end. --ogrid: one
                     row per wall face, i,k,x,y,z,theta_deg,area,cp,cf,
                     tau_x,tau_y,tau_z,yplus,d1,qw; theta from the
                     upstream axis, cf signed along the downstream
                     surface tangent. Compressible immersed body: one
                     row per ghost-wall interface face, x,y,z,axis,sign,
                     area,s,p,f,fp,fc,fv (area vector into the body,
                     gauge pressure, total / pressure / convective /
                     viscous force, solver units). Header comment: the
                     run label (docs/viewer-format.md).
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
  --temp             add a THIRD frame channel, temp = T - Tinf, on the
                     diverging map (freestream = neutral background, hot red,
                     cold blue). Compressible only. Off by default because it
                     changes what every artefact of the run contains: one more
                     png series, one more web/shm plane. The clip is the
                     larger of the isothermal wall's offset and the stagnation
                     rise (gamma-1)/2 M^2 Tinf, so an adiabatic body still
                     reads.
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
                     with incompressible references. In (0,1),
                     except --rans --steady-scheme implicit: any
                     M > 0 (one broadcast column, v = 0: no shock
                     can form).                                     [0.2]
  --wall isothermal|adiabatic                              [isothermal]
                     isothermal walls conduct the body-force power
                     out; adiabatic has NO heat sink (T climbs until
                     dt collapses) -- short transients only.
  --cfl C            Courant target                                 [0.4]

START-UP
  --kick A           deterministic solenoidal trip amplitude (times
                     the bulk-velocity estimate) on the laminar
                     Poiseuille IC; 0 = untripped (stays laminar)   [0.2]

RANS (an ensemble-mean MODEL, never a DNS; labelled "RANS <variant>"
everywhere, docs/rans.md)
  --rans none|sa-neg|sst-2003m|sst-vm
                     turbulence closure. sa-neg = TMR's SA-neg (standard
                     SA with ft2 for nu~ >= 0, the negative branch
                     below); sst-2003m / sst-vm = Menter SST, 2003
                     constants with S / 1994 with vorticity, both with
                     2/3 rho k dropped from the stress (TMR "m"). k and
                     omega carry a COUNTED positivity guard (stats
                     floorHits; never silent). The mean is homogeneous
                     in x and z, so --n 4 --nz 1 is the same answer.
                     A trip is meaningless: --kick defaults to 0 and a
                     given --kick > 0 is rejected.                 [none]
  --steady TOL       STEADY solve in local-time-stepping pseudo-time
                     (not time-accurate) until the integrals (ub,
                     bulk T, the turbulence scalars) are within TOL,
                     RELATIVE, of their extrapolated fixed point;
                     TOL in [1e-6, 1). The residual is reported, not
                     used: its FP32 floor hides the slow mode
                     (docs/rans.md). Requires --rans and isothermal
                     walls; rejects --tt/--profile-dt and frame
                     output (pseudo-time is not time). Profiles are
                     written once, at convergence or at the cap.
                     Two schemes (--steady-scheme): explicit Re_tau
                     180, ny 64: TOL 1e-4 ~3.2e5 iterations, 1e-5
                     ~4.5e5; implicit ~26 iterations.      [0 = URANS]
  --steady-steps N   iteration cap; reaching it is exit 8
                     [1000000 explicit, 500 implicit]
  --steady-scheme explicit|implicit
                     pseudo-time scheme of --steady. explicit = local-
                     time-stepping SSP-RK3 (Re_tau 180 ny 64 TOL 1e-4:
                     ~3.1e5 iterations). implicit = coupled Newton line
                     solve, CFL ramp 100 x2 -> 1e8, total mass pinned
                     (~26 iterations, Re_tau 180-1e4); rejects --cfl
                     and --dt-recompute; --steady-steps defaults to 500.
                     Requires --steady.                       [explicit]
  --compensated      time march on the compensated state (FP32 hi + lo
                     per field, the explicit --steady march's), so an
                     update below ulp(q)/2 is kept instead of rounded
                     away. Changes the bits; ~1.1x per step. A no-op
                     on explicit --steady; rejects --steady-scheme
                     implicit. (docs/rans.md, "Time accuracy")   [off]

LES (a FILTERED field: statistics depend on grid AND Cs; labelled
"LES <model> Cs=..." everywhere, docs/channel.md "LES")
  --les CS           SGS model constant; 0 = DNS, bit for bit. mu_t is
                     imaged ODD across both wall ghost rows, so it is
                     exactly 0 on the wall faces. Time-accurate only:
                     rejects --rans and --steady. Stats append nutMax
                     (max mu_t/mu, fluid rows) and nutWall (wall-face
                     mean mu_t/mu, 0 by construction), profiles
                     nutRatio.                                       [0]
  --sgs smag|sigma   SGS model, used with --les. smag carries
                     nu_t/nu ~ Cs^2 Delta+^2 into the first cell (no
                     wall damping); sigma (C ~ 1.35) vanishes in pure
                     shear; rejected at --nz 1 (zero in 2D)       [smag]

RUN LENGTH
  --tt N             eddy turnovers (delta/u_tau = 1) to integrate   [30]
  --dt-recompute N   steps between stableDt recomputes               [16]

OUTPUT
  --emit stats       <out>_stats.csv (t,dt,ubulk,utauBot,utauTop,
                     retau,KE; RANS appends mutMax, SST floorHits,
                     and --steady iter,resid,resU,resE,resT,
                     Tbulk,remaining; implicit also cfl,alpha,
                     rejects, one row per iteration, t and dt in
                     pseudo-time; LES appends nutMax,nutWall and
                     opens with a "# LES ..." label line)        [on]
  --profile-dt DT    <out>_profiles.csv y-profile dump interval
                     (t,y,yplus,uplus,urms,vrms,wrms,uv,rho,T; RANS
                     appends mutRatio and SA nutilde | SST kplus,
                     omegaplus; LES appends nutRatio),
                     Favre-averaged over x,z; 0 = off               [1.0]
  --emit png|web|bin mid-plane frames (png renders the centre z-plane,
                     2D or 3D)
  --fdt/--pdt/--wdt  frame cadences as in dns flow
)";
    case Subcommand::Conv:
        return
R"(dns conv — Rayleigh-Bénard convection

Compressible DNS of a fluid layer heated from below: no-slip isothermal walls
(hot floor, cold lid), periodic or closed sides, gravity in -y. There is no body, no
free stream and no forcing besides the heated floor -- buoyancy is gravity
acting on the density the equation of state already gives, so nothing here is
Boussinesq-approximated.

Box units: layer height H = 1, mid-layer temperature Tref = 1, R = rho_ref = 1
(hence p_ref = 1 and the reference sound speed is sqrt(gamma) = 1.183).

STRATIFICATION -- the one thing that is not textbook Rayleigh-Bénard. Gravity
carries its own adiabatic temperature gradient g/cp, so only the SUPERADIABATIC
excess dTsuper = dtemp - (g/cp) H drives convection (Schwarzschild). --rayleigh
is defined on dTsuper, so its critical value is the textbook ~1708; the
reported Nusselt numbers are normalised on the FULL imposed drop, so quiescent
conduction reads exactly Nu = 1. The adiabatic fraction is M^2 (gamma-1)/dtemp^2
-- it vanishes quadratically as --mach drops toward the Boussinesq limit, and a
subadiabatic (stably stratified) layer is REJECTED, not silently run.

GRID
  --n N              lateral FLUID cells in x                       [128]
  --ny N             wall-normal FLUID cells (two mirror ghost rows
                     are added internally)                           [64]
  --nz N             spanwise FLUID cells (z); 1 = native 2D
                     (qualitative only -- no vortex stretching)     [128]
  --aspect A         box width and depth in layer heights            [2]
  --sides periodic|closed                                    [periodic]
                     the LATERAL faces. periodic: an unbounded layer
                     sampled over one box, the classical Rayleigh-Benard
                     setup. closed: a sealed cavity -- x (and z when
                     nz > 1) become no-slip ADIABATIC walls, carried by
                     the same one-cell ghost ring as the plates, so the
                     grid gains two cells per closed axis while the fluid
                     extent stays --aspect * H. Rolls then have a fixed
                     count set by the aspect ratio and the corner regions
                     are viscously braked, so Nu comes out BELOW the
                     periodic value at the same Ra: an aspect-ratio
                     dependent cavity number, not the textbook one, and
                     neither the ~1708 onset nor the Ra-Nu scaling of the
                     unbounded layer applies to it. The deficit shrinks as
                     the box widens -- measured 8-15% at --aspect 2.
                     Both physics carry it; at nz == 1 only x closes,
                     because the 2D restriction is exact periodic wrap
                     in z.
  --hot-patch F      heat only a CENTRED fraction F of the floor, in
                     (0,1]; outside it the floor sits at the lid
                     temperature. 1 = the uniformly heated plate, and a
                     run omitting the flag is bit-for-bit that. In 3D the
                     fraction applies to both x and z (a centred square
                     patch of side F*L); at nz == 1 it is the middle
                     strip in x, which is the 2D reading. F = 1/3 heats
                     the middle third.
                     With F < 1 the layer is no longer classical
                     Rayleigh-Benard: there is no 1D conduction state, the
                     onset Ra ~ 1708 does not apply, and the reported
                     nuBot becomes the NET floor-area-averaged heat flux
                     (the cold surround absorbs heat) rather than a
                     Nusselt number.                                   [1]

PHYSICS
  --physics compressible|incompressible                 [compressible]
                     compressible: density is a state variable, the time
                     step is acoustic, and the layer is stratified (see
                     above). incompressible: the BOUSSINESQ limit, taken
                     analytically -- divergence-free velocity, density kept
                     only in the buoyancy term. No sound speed, hence no
                     acoustic dt (the same case runs ~40x fewer steps), no
                     stratification and no adiabatic gradient, so Ra and Nu
                     are measured against the same full drop. Free-fall
                     scaling pins H = dT = g*alpha = 1, so
                     nu = sqrt(Pr/Ra), kappa = 1/sqrt(Ra Pr) and the
                     free-fall time is exactly 1.
                     These are two MODELS, not two solvers for one problem:
                     --mach, --dtemp, --visc sutherland and --tinf only mean
                     something against a sound speed and are REJECTED on the
                     incompressible branch rather than ignored. --sides,
                     --hot-patch and the grid are shared by both.
  --rayleigh RA      Rayleigh number on the superadiabatic drop    [1e6]
                     (below ~1708 the layer only conducts)
  --dtemp E          imposed (Tfloor - Tlid)/Tref; the walls sit at
                     1 +/- E/2                                     [0.1]
  --prandtl P        molecular Prandtl number                     [0.72]
  --mach M           M = sqrt(g dtemp H)/c: this is what SETS gravity,
                     g = M^2 gamma / dtemp. Lower = closer to the
                     incompressible limit and more steps per
                     free-fall time.                               [0.05]
  --visc constant|sutherland   viscosity law; sutherland also makes
                     kappa = mu cp / Pr temperature-dependent  [constant]
  --tinf K           dimensional anchor for Sutherland's 110.4 K [288.15]
  --cfl C            Courant target                                [0.4]

START-UP
  --kick A           deterministic solenoidal trip on the hydrostatic
                     conduction state, as a fraction of the free-fall
                     velocity; 0 = untripped (a perfectly symmetric
                     layer takes far longer to break)              [0.02]

RUN LENGTH
  --tt N             free-fall times H/Uff to integrate              [30]
  --dt-recompute N   steps between stableDt recomputes               [16]

OUTPUT
  --emit stats       <out>_stats.csv (t,dt,nuBot,nuTop,nuFlux,urms,KE;
                     incompressible adds maxDiv). nuBot/nuTop are
                     wall-gradient Nusselt numbers, nuFlux the interior
                     heat-flux one -- all three normalised on the FULL
                     imposed drop, so quiescent conduction reads 1  [on]
  --profile-dt DT    <out>_profiles.csv y-profile dump interval
                     (t,y,T,rho,v,vT,nu); 0 = off                     [0]
  --emit png|web|bin mid-plane frames: temp (T - Tref, diverging), speed
                     and vorticity. png renders the centre z-plane.
  --fdt/--pdt/--wdt  frame cadences as in dns flow
)";
    case Subcommand::Bench:
        return
R"(dns bench — step-time and throughput benchmark
  --n N            cells per side                                  [256]
  --steps N        timed steps after 3 warm-up steps                [20]
  --physics incompressible|compressible          [incompressible]
  --re RE          sets the viscosity nu = 1/RE (--nu overrides)     [60]
  --mach M         compressible only: uniform pressure 1/(gamma M^2) [0.2]
  --les CS         compressible only: time the SGS path too          [off]
  --sgs smag|sigma which nut(grad u) --les uses                    [smag]
  Reports ms/step, Mcell-updates/s and the VRAM this run allocates.
)";
    case Subcommand::Flame:
        return
R"(dns flame — single-step flames: 1D planar premixed, a 2D slot Bunsen, or a
2D slot-jet diffusion flame (docs/chem.md)

Compressible DNS of a laminar flame held in an open x box: fresh gas enters
through a characteristic (NSCBC) inflow, burns in a single step F -> P, and
leaves through the outflow. The model is C0: calorically perfect (gamma 1.4
fixed), equal molecular weights, Le = 1, one species carried (Y = Y_F).

Box units: rho_u = S_L = R = 1 and delta_D = lambda_u/(rho_u cp S_L) = 1, so
mu = Pr and the pre-exponential A = Lambda -- the FP64 phase-plane eigenvalue
of the model, computed at validation and printed by --dry-run. The oracle
flame burns at S_L = 1 exactly; the run measures its own S_L against that.
Tu = p0 = 1/(gamma M^2): --mach sets the sound speed, not the flame.

MACH INFLATION. A real CH4/air flame has S_L/c_u ~ 1.15e-3; the acoustic dt
scales with it, so the run inflates it (default 0.01). The compressible flame
speed carries an O(M^2) offset (gamma M^2 (sigma - 1)): measured below 0.5%
over M 0.005-0.02. The label states the M used, always.

SPECIES. rho Y rides EXACTLY the central mass flux of the mean flow with an
MC-limited MUSCL face value: deliberate, bounded scalar dissipation on the
species alone (the RANS precedent). The mean flow stays central.

  --mode premixed|bunsen|diffusion|candle|chamber
                       1D planar flame | 2D slot Bunsen | 2D jet diffusion
                       flame | 2D candle | closed 2D combustion chamber
                       (own sections below)                   [premixed]
  --mach M             flame Mach number S_L/c_u, (0, 0.05]       [0.01]
                       (warns above 0.02, the measured ladder top)
  --expansion SIGMA    Tb/Tu, (1, 10]                                [5]
  --zeldovich BETA     Zeldovich number, [2, 20]                     [8]
  --prandtl PR         molecular Prandtl number (Le = 1)          [0.72]
  --visc constant|sutherland                                 [constant]
  --tinf K             Tu in kelvin; required with sutherland
  --cells-per-delta N  cells per oracle thermal thickness, >= 4
                                                     [15; bunsen 8]
  --upstream N         fresh extent in thermal thicknesses, >= 6    [12]
  --downstream N       burnt extent in thermal thicknesses, >= 4    [10]
  --ny N               periodic y cells, >= 4                        [4]
  --nz 1               the only value (multi-D flames are P2)
  --inflow-speed U     inflow velocity in S_L (1 = a stationary flame) [1]
  --wrinkle A          IC front displacement, delta_th: modes 1..N of the
                       box height, A/sqrt(N) each, phases k^2 golden;
                       0 = planar (seeds Darrieus-Landau in 2D)       [0]
  --wrinkle-modes N    number of seeded modes, [1, ny/8]              [8]
  --nscbc-etain E      inflow relax stiffness, as dns flow         [100]
  --nscbc-sink N       sink band per open face, cells; 0 = off      [16]
  --nscbc-sink-amp A   sink strength, (0, 2]                         [1]
  --tt T               run length in flame times delta_D/S_L        [10]
  --cfl C              acoustic CFL                                [0.4]
  --dt-recompute N     steps between dt recomputations             [16]
  --profile-dt DT      <out>_profiles.csv x-profile dump interval
                       (t,x,T,rho,u,p,Y,omega); 0 = off               [0]
  --emit png|web|bin|shm  frames: temp (T - Tu), fuel (Y), omega
  --fdt/--pdt/--wdt    frame cadences as in dns flow

  Rejected by name: --re, --ft (use --tt), --n (use --cells-per-delta),
  --physics incompressible (no low-Mach branch yet).

BUNSEN (--mode bunsen). x is the jet axis: the x-lo face is the burner exit,
a per-row NSCBC inflow of a slot jet of fresh gas (Y = 1, T = Tu) between
two coflows of inert gas (Y = 0); y is periodic (the coflow separates the
jet from its image). Edges are tanh blends. The flame ANCHORS ON A BURNT
PILOT: by default the whole coflow is burnt gas (T = sigma Tu). Measured at
U_jet 5, 8 cells: burnt coflow anchors; cold coflow, T/Tu 3, a cold u = 0
rim and slow (1 S_L) burnt strips of 0.75-1.5 delta_th all blow off.
Lengths in delta_th, speeds in S_L, temperatures as T/Tu.
  --slot-width W       jet width, >= 4                               [12]
  --jet-speed U        fresh jet speed, [1.5, 20]; U M <= 0.3          [5]
  --coflow-speed U     coflow speed, [0, U_jet]                       [1]
  --coflow-width W     coflow per side, >= 2              [slot width]
  --coflow-temp T      coflow T/Tu, [1, sigma]               [sigma: pilot]
  --pilot-width W      strip per side between jet and coflow, Y = 0  [0]
  --pilot-temp T       its T/Tu, [1, sigma]                       [sigma]
  --pilot-speed U      its speed, [0, U_jet]               [coflow speed]
                       (a cold coflow anchors on a FAST burnt strip:
                       --pilot-width 3 --pilot-speed U_jet measured)
  --slot-edge E        tanh half-width of the edges, >= 1 cell     [0.25]
  --length L           x extent past the sink bands, >= cone + 4
                                         [ceil(1.25 cone height + 8)]
  --pulse A            jet speed modulation u_jet (1 + A sin 2 pi f t),
                       [0, 0.5]                                       [0]
  --pulse-freq F       its frequency f, 1/tau (needs --pulse)
  Rejected by name with bunsen: --upstream, --downstream, --inflow-speed,
  --wrinkle, --wrinkle-modes, --ny (the burner sets the box), --profile-dt;
  and the bunsen flags without it.

STATS (<out>_stats.csv): t, dt, dtChem, SL (consumption speed, the integral
of omega over rho_u Y_u), SLdisp (u_in - dx_f/dt), xf (mid-temperature
crossing), dth ((Tb - Tu)/max dT/dx), Tmax, mass, energy (E + q rho Y),
omegaEdge (max omega in the sink bands / max omega), yHits (cells with Y
outside [-1e-6, 1 + 1e-6], counted, never clamped), yMin, yMax.
Exit 8 when the flame drifts within band + 2 delta_th of either face.
Bunsen: t, dt, dtChem, xTip, xBase (the Tmid isotherm on the axis / the
lowest burning flank row), tanA, tanL, tanR, rms (least squares of the flank
over the middle half of the cone), sinRatio (sin(alpha) U_jet / S_L; 1 =
the ideal cone), SLflank (omega over the flank rows per flank length),
mFlank (rho u.n at the flank isotherm), burn, fuelIn (the inflow face's
rho u Y), fuelNom, burnRatio, Tmax, mass, energy, yHits, yMin, yMax, uJet.
Exit 8 when the cone is not anchored: flash-back, blow-through, the base
lifting past half the tip height, or fewer than 4 burning flank rows.

DIFFUSION (--mode diffusion). Non-premixed: F + s O -> (1+s) P, omega =
A rho^2 Y_F Y_O e^(-Ta/T), Y_F and Y_O carried with ONE MC limiter shared by
the species (the coupling function Y_F - Y_O/s and the mixture fraction Z
are then transported exactly as passive scalars). Box units: rho0 = R =
U_jet = 1, length l = D0/U_jet (mu = Pr; the slot width IS the jet Peclet
number), T0 = p0 = 1/(gamma M^2), M = U_jet/c0. x is the jet axis: a fuel
slot (Z = 1) between oxidizer coflows (Z = 0), y periodic. The inflow row is
the Burke-Schumann flame sheet of its mixture fraction Z(y) (the jet edge
burns at the exit plane: the anchor), species pinned there; the IC is that
row extruded. Lengths in l, speeds in U_jet.
  --stoich S           oxidizer mass per fuel mass, (0, 100]          [1]
  --fuel-y Y           fuel stream Y_F (balance inert), (0, 1]        [1]
  --ox-y Y             oxidizer stream Y_O (balance inert), (0, 1]    [1]
                       Z_st = Y_O2 / (S Y_F1 + Y_O2)
  --expansion SIGMA    flame-sheet temperature T_ad(Z_st)/T0, (1, 10]  [5]
  --zeldovich BETA     Zeldovich number at T_ad, [2, 20]            [5]
                       (beta 8 extinguishes the default sheet, measured)
  --damkohler DA       A rho0 e^(-Ta/T_ad) l/U_jet, (0, 1e6]       [1000]
  --mach M             jet Mach U_jet/c0, (0, 0.2]                 [0.05]
  --cells-per-unit N   cells per l, [2, 64]                           [4]
  --slot-width W       fuel slot width = Peclet number, [4, 200]     [16]
  --coflow-speed U     oxidizer coflow speed, [0, 2]                  [1]
  --coflow-width W     coflow per side, >= 2                  [slot width]
  --pilot-width W      burnt Z_st strips per side (T_ad, no reactant) [0]
  --pilot-speed U      their speed, [0, 2]                 [coflow speed]
  --slot-edge E        tanh half-width of the inflow edges, >= 1 cell  [3]
                       (edge 1 extinguishes at the exit, measured)
  --length L           x extent past the sink bands, >= W
                     [ceil(min(1.25 x constant-density sheet tip + 8, 20 W))]
  --pulse A, --pulse-freq F   jet speed u_jet (1 + A sin 2 pi f t)
  --tt T               run length in l/U_jet                         [10]
  Rejected by name with diffusion: --upstream, --downstream, --inflow-speed,
  --wrinkle, --wrinkle-modes, --ny, --profile-dt, --cells-per-delta,
  --jet-speed, --coflow-temp, --pilot-temp; and the diffusion flags
  (--stoich, --fuel-y, --ox-y, --damkohler, --cells-per-unit) elsewhere.
  Stats: t, dt, dtChem, xTip (the axis Z_st crossing; NaN = open tip),
  xBase (lowest row with omega >= 0.1 max), half (Z_st half-width at tip/2),
  Tmax, TmaxIn (past the inflow band), burn, fuelIn (fuel through the first
  interior face: convective + diffusive), fuelOut (rho u Y_F out), burnRatio,
  mass, energy (E + q rho Y_F), pMean (box-mean p/p0), zMin, zMax, yfMin,
  yoMin, hits, uJet. Frames: temp, fuel, oxid, omega. Exit 8 when TmaxIn
  falls below (1 + sigma) T0/2 (extinction) or the base lifts past half the
  box.
  --ignition L         the IC burns a closed Burke-Schumann column L tall
                       (Z(y) w(x), tanh edge), coflow air above; absent =
                       the inflow row extruded over the whole box

CANDLE (--mode candle). The diffusion model above under GRAVITY g along -x:
the flame points UP (+x), fuel leaves a low-Peclet slot (the wick) at the
bottom into a slow air coflow. Buoyancy, not the fuel jet, sets the flow:
Fr = U^2/(g W) printed (0.071 at the defaults). The ambient column is
isothermal at T0, p_a(x) = p0 exp(-g x/(R T0)): the IC pressure, the inflow
density, the outflow target p_a(x_out), and BOTH faces' incoming acoustic
waves carry its hydrostatic gradient (without them the boundary rows sag at
g/2: a quiescent box reached |u| 0.67 U, with them 1.2e-3 U). Air drawn DOWN
through the top (the plume's entrainment in the periodic box) is ambient
air: rho_a, v = 0, Y_O2. Gravity is THE source of dns conv (one kernel).
Defaults: paraffin/air s 3.4, Y_O2 0.233 (Z_st 0.064), W 1, g 14, 8 cells/l,
edge 0.5, Da 5000, M 0.005, coflow 1 U x 12 l per side, length 32,
ignition 4. g = 0 is --mode diffusion bit for bit.
  --gravity G          g along -x in U^2/l, [0, 1000]                [14]
  --quiescent          no fuel, every inflow row at rest: the hydrostatic
                       rest check (rejects the slot and pulse flags)
  --mach M             U_fuel/c0; buoyant Mach M sqrt(2 g L (sigma-1)/sigma)
                       must stay <= 0.3 (warns above 0.15)       [0.005]
  Rejected by name with candle: --pilot-width, --pilot-speed; --gravity and
  --quiescent elsewhere. Stats: the diffusion columns + uMax (max |u|),
  uOutMin (min u on the top row; < 0 = air drawn in); pMean over the ambient
  column. Measured: H ~ Q^1.19 over W 0.5-1 (local 1.30 at 0.75-1; Roper's
  buoyant slot 4/3); W >= 1.25 oscillates slowly (period ~13 l/U).

--mode chamber: a closed 2D combustion chamber (docs/chem.md, "Chamber").
An immersed ADIABATIC ghost-wall body -- back wall, straight section,
linear converging nozzle -- with a wedge flame holder inside; premixed
reactants (Y = 1, Tu) enter through an injector band across the whole back
wall (a volumetric mass-flow source: the chamber pressure rises until the
nozzle passes it). Outside: cold inert ambient (Tu, Y = 0) in a coflow, x
NSCBC + sink, y periodic. IC: cavity fresh up to the holder base, burnt
behind it (oracle profile), all at rest at p0. Lengths in delta_th.
  --inject-speed U     fresh bulk speed past the injector, S_L        [3]
  --coflow U           ambient coflow, S_L                            [2]
  --chamber-height H   inner height, >= 8                            [15]
  --chamber-length L   straight section, >= 10                       [30]
  --nozzle-ratio R     Hc / He, [1.5, 6]                              [3]
  --holder-height H    wedge base height, [1, H - 4]                  [4]
  --clearance C        ambient above and below, >= 4                 [10]
  --downstream N       jet region past the nozzle, >= 10             [40]
  --inject-ramp T      smooth injection ramp, flame times             [1]
  --reaction on|off    off = inert species (wall-impermeability check) [on]
  chamber defaults: --mach 0.02, --cells-per-delta 4, --tt 60.
  Rejected with chamber: --ny, --upstream, --inflow-speed, --profile-dt;
  the chamber flags are rejected without it.
  STATS: t, dt, inj (fuel injection rate), injCum (its FP64 time integral),
  fuel (fluid inventory of rho Y), burn (integral of omega), burnCav (in the
  cavity), outNoz (fuel flux through the exit plane), outDom (fuel leaving
  the domain), mdotNoz, Me (mass-weighted exit Mach), pCh (fresh plenum),
  p0Noz (total p ahead of the contraction), pAmb (ambient beside the lips),
  h0Noz (exit total-enthalpy flux), Tmax, mass, energy (E + q rho Y),
  yHits, omegaInj (burn in the injector band / burn), fx (fluid -> body
  force, x). Exit 8 on flash-back (injector-band burn > 1e-3 of the total
  after ramp + 2) or blow-off (fuel through the exit > 0.5 of the injected
  after ramp + 2 Lc/U); 7 on divergence.
)";
    case Subcommand::Prop:
        return
R"(dns prop — propeller thrust curve (CT, CP, CQ, eta vs J), host-only

The cheap references a resolved rotor run is checked against, on the SAME
blade: blade-element momentum and a lifting line. Never acquires a
GPU. Conventions: shaft +x, freestream +x, +omega about +x. PROPULSIVE signs:
T > 0 pulls upstream, Q > 0 is the motor torque, P = 2 pi n Q;
CT = T/(rho n^2 D^4), CP = P/(rho n^3 D^5), CQ = CP/(2 pi), eta = J CT/CP
(nan where CP <= 0 or J = 0).

usage: dns prop --planform FILE --polar FILE|thin [--method bemt|liftline]
                [operating point] [method options] [--out FILE.csv]
       dns prop --method liftline --selftest

METHOD
  --method bemt|liftline                                           [bemt]
    bemt      blade-element momentum: axial + tangential induction, Prandtl
              tip AND hub loss, Buhl's turbulent-wake branch; inflow angle by
              Ning's (2014) bracketed residual, so a root is guaranteed
              wherever a bracket changes sign. J = 0 is the same equation.
              Section Re frozen per solve, refreshed from the solved W (3
              passes; 1 for the Re-free thin polar).
    liftline  lifting line: one horseshoe per panel, constant-pitch helical
              trailing wake realigned from the solution, Kerwin image hub,
              Vatistas cores, Biot-Savart over all B blades -- tip loss and
              blade interference come out of the vortex system, so a
              disagreement with BEMT is information. Gamma by Newton (MINPACK
              hybrd and pseudo-transient continuation behind it), Chattot
              stall viscosity + homotopy from the saturated linear polar.
              A J that does not converge is FLAGGED (converged = 0), never
              smoothed over, and reported on stderr; the exit code stays 0.

INPUTS (strict CSV: exact header, every row all finite numbers)
  --planform FILE    r_R,c_R,beta_deg rows hub -> tip: r/R (tip radii),
                     chord / tip radius, geometric pitch angle in deg. Linear
                     in r/R, held constant past either end.  >= 2 rows,
                     0 < r_R <= 1 strictly increasing, c_R > 0, |beta| < 90.
                     tools/propeller.py exports it from its blade law.
  --polar FILE       re,alpha_deg,cl,cd rows grouped by re ascending, alpha
                     ascending over exactly -180..180 deg in each group
                     (XFOIL + Viterna, tools/polars.py exports it).
                     Interpolated linearly in alpha and in log Re, Re clamped
                     to the table (never extrapolated).
  --polar thin       built-in thin-airfoil polar: 2 pi (alpha - alpha0),
                     constant cd0 on alpha0 +- 10 deg, Viterna past it
                     (flat-plate cd 1.11 + 0.018 AR from the planform).
                     Re-independent.
  --thin-cd0 CD      --polar thin: profile drag, (0,1)              [0.02]
  --thin-alpha0 DEG  --polar thin: zero-lift angle, (-10,10) deg; NACA
                     4412 is -4.154                                    [0]

BLADE AND OPERATING POINT (SI)
  --blades N         blade count                                       [3]
  --hub-radius F     hub radius / tip radius; the blade is integrated from
                     the hub SURFACE            [the planform's first r/R]
  --diameter D       m                                             [0.254]
  --rpm RPM                                                         [6000]
  --rho RHO          kg/m^3                                        [1.225]
  --nu NU            kinematic viscosity, m^2/s                   [1.5e-5]
  --sound C          speed of sound, m/s (station/tip Mach, reported) [340]
  --j LIST           advance ratios J = V/(n D): 'a,b,c' or
                     'start:stop:step' (stop inclusive), sorted, >= 0
                                          [bemt 0:1:0.05, liftline 0:1.0:0.1]

BEMT
  --stations-bemt N  radial elements, cosine-clustered to root and tip,
                     >= 4                                             [40]

LIFTING LINE
  --panels N         panels per blade, >= 4                            [30]
  --wake-length L    helix length in diameters                         [10]
  --align-iters N    wake realignments after the seed wake              [3]
  --align-tol T      converged when the wake pitch moves less than this,
                     relative, at every node                         [0.01]
  --seg-deg A        helix segment angle, (0,30]; half on the first turn,
                     double past one diameter                          [10]
  --core R           vortex core radius in tip radii; 0 = 5% of the
                     smallest control-point/node gap                    [0]
  --hub image|none   image: Kerwin hub images (the hub is a wall, the root
                     keeps its load); none: a free root end         [image]
  --wake-pitch mean|local
                     helix pitch from the circumferential-mean disk
                     velocity of the vortex system, or the blade-local
                     velocity (collapses behind a stalled root)      [mean]
  --stall-visc K     Chattot artificial viscosity past stall, in units of
                     the sawtooth stability bound; 0 = off              [2]
  --selftest         Biot-Savart vs closed forms, the actuator-disk
                     cylinder, Betz/Prandtl limits; exit 0 iff all pass

OUTPUT
  --out FILE.csv     J,CT,CP,CQ,eta,T_N,Q_Nm,P_W,Re75 (liftline adds
                     converged). Omitted: the table on stdout only.
  --dump-span J      also print the spanwise solution at this J
  --quiet            no table on stdout (warnings still on stderr)
  --dry-run          validate, print the resolved case; reads no file
)";
    }
    return {};
}

} // namespace

} // namespace dns::cli
