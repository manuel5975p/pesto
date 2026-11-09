// validate(): the shared prologue and the per-subcommand dispatch.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"
#include "dns/shockflux.hpp"



#include <algorithm>
#include <cassert>
#include <format>
#include <optional>
#include <string>

namespace dns {
using namespace cfg;
} // namespace dns

namespace dns {

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
    c.shock = r.shock;

    // The low-Mach solver is the flame's alone (masterplan real_bunsen S4):
    // every other subcommand rejects it here, before its own validator reads
    // c.physics as "not compressible".
    if (r.physics == Physics::LowMach && r.sub != Subcommand::Flame)
        return bad(2, "--physics", "--physics lowmach exists for dns flame "
                                   "only (the zero-Mach variable-density "
                                   "flame solver)");
    if (r.given.contains("--chem-dt") &&
        !(r.sub == Subcommand::Flame && r.physics == Physics::LowMach))
        return bad(2, "--chem-dt", "only applies to dns flame --physics "
                                   "lowmach (the compressible chemistry dt is "
                                   "kChemDt 0.5 / J)");
    if (r.given.contains("--dt-rule") &&
        !(r.sub == Subcommand::Flame && r.physics == Physics::LowMach))
        return bad(2, "--dt-rule", "only applies to dns flame --physics "
                                   "lowmach (the compressible dt is acoustic)");

    if (auto e = shockRouteScope(r)) return std::unexpected(*e);
    if (r.sub == Subcommand::Flame && r.given.contains("--shock-d4")) {
        if (!r.given.contains("--polymesh"))
            return bad(2, "--shock-d4", "is the FV polyMesh route's (the structured solvers carry no background term)");
        if (r.shock != ShockMode::Hll) return bad(2, "--shock-d4", "needs --shock hll");
        if (!(r.shockD4 >= 0.0 && r.shockD4 <= kFvShockEps4Max)) return bad(2, "--shock-d4", "must be in [0, 0.125]");
    }
    if (r.sub == Subcommand::Flow && !r.given.contains("--polymesh")) {
        if (r.given.contains("--shock-d4"))
            return bad(2, "--shock-d4", "is the FV polyMesh route's (the structured solvers carry no background term)");
        if (r.inviscid) return bad(2, "--inviscid", "needs --shock hll");
        if (r.given.contains("--foam-write")) return bad(2, "--foam-write", "needs --polymesh");
        if (r.given.contains("--gamma")) return bad(2, "--gamma", "is the FV polyMesh route's (gamma 1.4 fixed elsewhere)");
    }

    // prop shares none of the grid prologue: its --out is a CSV path, and
    // optional.
    if (r.sub == Subcommand::Prop) return validateProp(r, std::move(c));
    // mesh: host-only, no grid and no output; the parser requires --polymesh.
    if (r.sub == Subcommand::Mesh)
    {
        c.polyMeshDir = r.polymesh;
        c.meshOrder = r.meshOrder;
        return c;
    }
    // flow's routes. --polymesh CASE alone is simpleFoam, its whole case the
    // OpenFOAM dictionaries; --polymesh DIR --bc FILE and --solver fv are the FV
    // solver of --physics; --solver simple takes the structured mesh flags.
    if (r.sub == Subcommand::Flow) {
        const bool poly = r.given.contains("--polymesh");
        const bool comp = r.physics == Physics::Compressible;
        const bool fv = r.solver == int(SolverKind::Fv);
        if (poly && r.given.contains("--solver") && !fv)
            return bad(2, "--solver", "--polymesh names its own path: only --solver fv goes with it");
        if (r.solver == int(SolverKind::Simple)) {
            if (!finitePos(r.re)) return bad(2, "--re", "must be finite and > 0");
            c.re = r.re;
            c.n = r.n;
            return validateFvStructured(r, std::move(c));
        }
        if (poly && comp) return validateFvCompMesh(r, std::move(c));
        if (poly && (fv || r.given.contains("--bc"))) return validateFvIncompFlow(r, std::move(c));
        if (poly) {
            for (const char* f : {"--tt", "--frame-box", "--frame-n"})
                if (r.given.contains(f))
                    return bad(2, f, "belongs to the compressible FV path: add --physics compressible and --bc");
            return validateFvFlow(r, std::move(c));
        }
        if (fv && !comp) return validateFvIncompFlow(r, std::move(c));
        if (r.given.contains("--bc")) return bad(2, "--bc", "binds the patches of --polymesh DIR");
        for (const char* f : {"--export", "--re-tau"})
            if (r.given.contains(f)) return bad(2, f, "only with --solver simple");
        if (r.given.contains("--tt"))
            return bad(2, "--tt", "belongs to the compressible FV path on a polyMesh (--polymesh DIR)");
    }
    // Past here the only FV routes left: flow --solver fv compressible, tgv --solver fv.
    const bool fvStructured = r.sub == Subcommand::Flow && r.solver == int(SolverKind::Fv);
    const bool fvTgv = r.sub == Subcommand::Tgv && r.solver == int(SolverKind::Fv);
    const bool fvConv = r.sub == Subcommand::Conv && (r.solver == int(SolverKind::Fv) || r.given.contains("--polymesh"));
    if (r.given.contains("--fv-precision") && !fvStructured && !fvTgv && !fvConv && r.sub != Subcommand::Flame)
        return bad(2, "--fv-precision", "selects the arithmetic of the FV path: only with --polymesh, --solver "
                                        "simple or --solver fv");
    if (r.sub == Subcommand::Flow && !fvStructured)
        for (const char* f : {"--frame-box", "--frame-n"})
            if (r.given.contains(f)) return bad(2, f, "the FV path's frame plane: only with --polymesh or --solver fv");
    if (fvStructured)
        if (auto e = checkFvStructured(r)) return std::unexpected(*e);
    for (const char* f : {"--fv-coupled", "--fv-coupled-eta", "--fv-coupled-alpha", "--fv-coupled-check"})
        if (r.given.contains(f)) return bad(2, f, "the coupled prototype of the FV path: only with --polymesh");

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

    if (r.sub == Subcommand::Bench) {
        auto b = validateBench(r, std::move(c));
        if (b)
            if (auto e = shockFlagScope(r, *b)) return std::unexpected(*e);
        return b;
    }
    if (r.sub == Subcommand::Tgv || r.sub == Subcommand::Hit)
        return validatePeriodic(r, std::move(c));
    if (r.sub == Subcommand::Channel) return validateChannel(r, std::move(c));
    if (r.sub == Subcommand::Conv) return validateConv(r, std::move(c));
    if (r.sub == Subcommand::Flame) {
        auto f = validateFlame(r, std::move(c));
        if (f)
            if (auto e = shockFlagScope(r, *f); e) return std::unexpected(*e);
        if (f && r.shock != ShockMode::None) f->meta.shock = name(r.shock);
        if (f && r.shock == ShockMode::Hll) shockCadence(r, *f);
        // The FV flame's background term (masterplan_combnozzle P3); the structured flame has none.
        if (f && r.shock == ShockMode::Hll && r.given.contains("--polymesh")) f->shockEps4 = r.shockD4;
        if (f)
            if (auto e = resolveFrameStart(r, *f); e) return std::unexpected(*e);
        if (f)
            if (auto e = resolveCheckpoint(r, *f); e) return std::unexpected(*e);
        if (f && (r.solver == int(SolverKind::Fv) || r.given.contains("--polymesh"))) {
            f->fvMode = r.given.contains("--polymesh") ? FvCompMode::PolyMesh : FvCompMode::Structured;
            if (r.given.contains("--polymesh")) {
                if (!r.fvBc) return bad(2, "--bc", "--polymesh needs --bc FILE.toml");
                f->fvMeshDir = r.polymesh;
                f->fvBc = *r.fvBc;
                if (r.shock == ShockMode::Hll && f->physics == Physics::Compressible) {
                    // masterplan_combnozzle P3: the reacting nozzle's tables on the hll flame route.
                    if (auto e = fvFlameNozzleScope(f->fvBc)) return std::unexpected(*e);
                    // masterplan_combnozzle P4: the throat (the composition design) sets u_u.
                    if (f->fvBc.nozzle && r.given.contains("--inflow-speed"))
                        return bad(2, "--inflow-speed", "the throat sets the fresh velocity on a [nozzle] run");
                } else if (const std::string k = fvBcNozzleKey(f->fvBc); !k.empty()) {
                    return bad(2, "--bc", k + " belongs to the nozzle route (dns flow --shock hll --polymesh): the "
                                              "compressible flame's open faces are far/outflow");
                }
                if (f->physics != Physics::LowMach)
                    for (const FvPatchSpec& p : f->fvBc.patches)
                        if (p.kind == FvPatchKind::Open)
                            return bad(2, "--bc", std::format("patch.{}: type = \"open\" is the low-Mach route's face "
                                                              "(--physics lowmach); the compressible flame's open "
                                                              "faces are far/outflow",
                                                              p.name));
                // Compressible --equivalence: the flags set every open patch's Y (the fresh
                // premix in, the burnt one out); a y in the file would be a second source.
                if (f->physics != Physics::LowMach && f->flamePremix2)
                    for (const FvPatchSpec& p : f->fvBc.patches)
                        if (p.hasY)
                            return bad(2, "--bc", std::format("patch.{}.y: --equivalence sets the mass fractions of "
                                                              "the compressible flame's open patches (far / inflow: "
                                                              "the fresh premix's Y_F1, Y_O1; outflow: the burnt "
                                                              "premix's); give no y",
                                                              p.name));
                const bool jetBc = std::ranges::any_of(f->fvBc.patches, &FvPatchSpec::jet);
                // The low-Mach burner's rows ride the one inflow patch with jet = true.
                const bool lmBurner = f->physics == Physics::LowMach && f->flameBunsen;
                const bool wantJet = f->flameDiffusion || lmBurner;
                if (wantJet != jetBc)
                    return bad(2, "--bc", wantJet ? std::string(f->flameDiffusion ? "--mode diffusion" : "--mode bunsen") +
                                                        " needs a patch with jet = true"
                                                  : "jet = true belongs to --mode diffusion and the low-Mach --mode "
                                                    "bunsen");
                if (lmBurner && std::ranges::count_if(f->fvBc.patches, &FvPatchSpec::jet) != 1)
                    return bad(2, "--bc", "--mode bunsen: exactly one patch carries jet = true (the burner's rows)");
                if (lmBurner && (f->fvBc.dir != std::array<double, 3>{1, 0, 0} ||
                                 f->fvBc.lift != std::array<double, 3>{0, 1, 0}))
                    return bad(2, "--bc", "--mode bunsen on a polyMesh needs reference.direction = [1, 0, 0] and "
                                          "lift = [0, 1, 0] (the measurement plane is the mesh's x-y)");
                // No sink band on a mesh: --upstream is the plain distance from the origin.
                if (!f->flameDiffusion) f->flameXf0 = r.upstream * f->flameEig.dthOverDD;
            }
            if (auto e = fvCompCommon(r, *f)) return std::unexpected(*e);
            if (f->physics == Physics::LowMach) {
                if (!finitePos(r.tol)) return bad(2, "--tol", "must be finite and > 0");
                f->fvTol = r.given.contains("--tol") ? r.tol : 1e-5;
            }
        }
        return f;
    }
    auto f = validateFlow(r, std::move(c));
    if (f)
        if (auto e = shockFlagScope(r, *f); e) return std::unexpected(*e);
    if (f && r.shock != ShockMode::None) f->meta.shock = name(r.shock);
    if (f && r.shock == ShockMode::Hll) shockCadence(r, *f);
    if (f && fvStructured) {
        f->fvMode = FvCompMode::Structured;
        if (auto e = fvCompCommon(r, *f)) return std::unexpected(*e);
    }
    return f;
}

} // namespace dns
