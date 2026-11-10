// validate(): dns flame (premixed, bunsen, low-Mach) and its FaceMesh route allow-list (checkFvFlame).
#include "dns/config.hpp"
#include "dns/config/internal.hpp"

#include "dns/fvmodules.hpp"
#include "dns/lowmach.hpp"

#include "dns/tfles.hpp"
#include "dns/transport.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace dns::cfg {
namespace {

// combscale S5: --wrinkle reaches a [nozzle] flame polyMesh (compressible, --shock hll), its chamber's lowest
// transverse mode; everywhere else on --polymesh it is refused as before.
bool nozzleWrinkle(const RawConfig& r)
{
    return r.given.contains("--polymesh") && r.physics != Physics::LowMach && r.shock == ShockMode::Hll && r.fvBc &&
           r.fvBc->nozzle.has_value();
}

// The 1D freely propagating premixed flame (docs/chem.md): model C0, the
// oracle eigenvalue resolved HERE (pure FP64, ~0.1 s) so the grid is counted
// in the oracle's own thermal thickness and --dry-run prints Lambda.
// The FV route of dns flame (docs/fvcomp.md "Species"): only what is built on it
// passes, everything else by name. Off it, its own flags are rejected.
std::optional<ConfigError> checkFvFlame(const RawConfig& r)
{
    const bool poly = r.given.contains("--polymesh");
    if (!poly && r.solver != int(SolverKind::Fv)) {
        if (r.physics == Physics::LowMach && r.given.contains("--tol"))
            return ConfigError{2, "--tol", "belongs to dns flame --physics lowmach --solver fv (the structured solve is direct)"};
        return rejectGiven(r, {"--bc", "--fv-precision", "--compensated", "--frame-box", "--frame-n"},
                           "belongs to dns flame --solver fv");
    }
    const bool lm = r.physics == Physics::LowMach;
    static const char* const allowed[] = {
        "--solver", "--polymesh", "--bc", "--physics", "--mode", "--mach", "--expansion", "--zeldovich", "--prandtl",
        "--visc", "--tinf", "--cells-per-delta", "--upstream", "--downstream", "--inflow-speed", "--ny", "--nz",
        "--wrinkle", "--wrinkle-modes", "--tt", "--cfl", "--dt-recompute", "--profile-dt", "--out", "--emit", "--fdt",
        "--pdt", "--wdt", "--png-down", "--frame-box", "--frame-n", "--frame-start", "--backend", "--quiet", "--dry-run",
        "--fv-precision", "--compensated",
        // M5: runFvComp's and runFvLowMach's checkpoints (gpu_restart_fv: every route).
        "--checkpoint-dt", "--wall-limit", "--restart",
        // gammaScope below keeps it to the compressible premixed C0 polyMesh flame.
        "--gamma",
        // masterplan_combnozzle P3 (Shock x C0): shockRouteScope keeps them to the compressible polyMesh flame.
        "--shock", "--shock-d4"};
    static const char* const allowedJet[] = {"--stoich", "--fuel-y", "--ox-y", "--damkohler", "--cells-per-unit",
                                             "--slot-width", "--slot-edge", "--coflow-speed", "--coflow-width",
                                             "--pilot-width", "--pilot-speed", "--length", "--ignition"};
    // Species LES and the thickened flame (docs/fvcomp.md "Species LES and TFLES"):
    // the compressible premixed flame only; flameLesTfles applies the structured rules.
    static const char* const allowedLes[] = {"--les", "--sgs", "--tfles", "--tfles-sensor", "--efficiency"};
    // M5: --turb on the flags' box (premixed C0, C2b; both physics), dns/fvmodules Turb.
    static const char* const allowedTurb[] = {"--turb", "--turb-length", "--seed"};
    const bool jet = r.flameMode == 2;
    // Low-Mach FV: model C2 / C2b (two reactants, --equivalence) on faces, the 2D slot
    // burner (--mode bunsen: per-face inflow rows, --dt-rule cell).
    static const char* const allowedLm[] = {"--chem-dt", "--tol",   "--equivalence", "--sl-law",
                                            "--stoich",  "--ox-y", "--mix-phi"};
    // M3 steps 3-5: open laterals (the `open` face kind), gravity (the face force),
    // the rim (the tube's staircase as an isothermal Wall patch).
    static const char* const allowedLmBunsen[] = {
        "--dt-rule",     "--slot-width",  "--jet-speed",  "--coflow-speed", "--coflow-width", "--coflow-temp",
        "--pilot-width", "--pilot-temp",  "--pilot-speed", "--slot-edge",   "--length",       "--reaction",
        "--laterals",    "--gravity",     "--rim",         "--rim-temp",    "--tube-length"};
    const bool lmBunsen = lm && r.flameMode == 1;
    // Compressible FV --equivalence (masterplan_multiphysics M2): models C2 / C2b on the
    // two-reactant face kernels (dns/fvmodules C1 / C2b), premixed mode; LES with it.
    static const char* const allowedRich[] = {"--equivalence", "--sl-law", "--stoich", "--ox-y", "--mix-phi"};
    const bool rich = !lm && r.flameMode == 0 && r.given.contains("--equivalence");
    // --mode candle: the jet under gravity, its open patches on the hydrostatic ambient's
    // rows (docs/fvcomp.md "Hydrostatic ambient"); --wick (an immersed ghost wall) is not
    // in any list: refused by name below.
    const bool candle = !lm && r.flameMode == 3;
    static const char* const allowedCandle[] = {"--gravity", "--quiescent"};
    if (r.given.contains("--turb") || r.given.contains("--turb-length")) {
        if (poly)
            return ConfigError{2, "--turb", "with --polymesh: not built (the synthetic box's periods and lattice are "
                                            "the flags' box: --solver fv on --ny, --nz; issues.md M5c)"};
        if (jet || candle) return ConfigError{2, "--turb", std::string(fvPair(FvModule::Turb, FvModule::ChemC1).message)};
        if (!lm && rich && r.slLaw == 1)
            return ConfigError{2, "--turb", std::string(fvPair(FvModule::Turb, FvModule::ChemC1).message)};
    }
    if (r.given.contains("--hold-flame"))
        return ConfigError{2, "--hold-flame", "is not built on the FV flame path (the brush's inflow control is the "
                                              "structured compressible solver's)"};
    for (const auto& g : r.given) {
        if (!poly && !jet && !lmBunsen && std::ranges::any_of(allowedTurb, [&](const char* a) { return g == a; }))
            continue;
        if (lm && std::ranges::any_of(allowedLm, [&](const char* a) { return g == a; })) continue;
        if (lmBunsen && std::ranges::any_of(allowedLmBunsen, [&](const char* a) { return g == a; })) continue;
        if (rich && std::ranges::any_of(allowedRich, [&](const char* a) { return g == a; })) continue;
        // M4: the low-Mach FV flame takes them too (lowMachModules decides the pairs).
        if (!jet && !candle && std::ranges::any_of(allowedLes,[&](const char* a) { return g == a; })) continue;
        if (candle && std::ranges::any_of(allowedCandle, [&](const char* a) { return g == a; })) continue;
        if (std::ranges::none_of(allowed, [&](const char* a) { return g == a; }) &&
            !((jet || candle) && !lm && std::ranges::any_of(allowedJet, [&](const char* a) { return g == a; })))
            return ConfigError{2, "--solver",
                               g + " is not built on the FV flame path (docs/fvcomp.md \"Species\", docs/fvincomp.md "
                                   "\"Low-Mach\")"};
    }
    if (rich)
        for (const char* f : {"--tfles", "--tfles-sensor", "--efficiency"})
            if (r.given.contains(f))
                return ConfigError{2, f, std::string(fvPair(FvModule::TflesDynamic, FvModule::ChemC2b).message)};
    if (r.flameMode != 0 && !jet && !candle && !lmBunsen)
        return ConfigError{2, "--mode", "--solver fv is built for --mode premixed, diffusion and candle, and bunsen "
                                        "with --physics lowmach (chamber: not built, docs/fvcomp.md)"};
    if (lmBunsen && r.given.contains("--nz") && r.nz != 1)
        return ConfigError{2, "--nz", "--mode bunsen on the low-Mach FV path is the 2D slot (the 3D slot and the round "
                                      "nozzle are not built there, docs/fvincomp.md \"Low-Mach burner\")"};
    if (lmBunsen && r.given.contains("--profile-dt"))
        return ConfigError{2, "--profile-dt", "the burner's jet profiles are not built on the low-Mach FV path"};
    if (jet && lm)
        return ConfigError{2, "--mode", "--mode diffusion on --solver fv is the compressible solver's (two reactants "
                                        "are not built on the low-Mach FV path, docs/fvincomp.md \"Low-Mach\")"};
    if (lm && r.given.contains("--compensated"))
        return ConfigError{2, "--compensated", "the low-Mach FV state's rho, rho Y are compensated in FP32 always"};
    if (poly != r.given.contains("--bc"))
        return ConfigError{2, "--bc", "--polymesh DIR and --bc FILE.toml go together on dns flame"};
    if (poly && r.given.contains("--solver") && r.solver != int(SolverKind::Fv))
        return ConfigError{2, "--solver", "--polymesh is the FV path: only --solver fv goes with it"};
    // The burner on a polyMesh: --cells-per-delta and --length keep their meaning, the
    // flags' slot box (its centre yc, the measurement plane's pixel); the mesh is the flow.
    if (poly && lmBunsen) {
        if (auto e = rejectGiven(r, {"--laterals", "--rim", "--rim-temp", "--tube-length"},
                                 "does not apply with --polymesh: the mesh and --bc are the case (an `open` patch, "
                                 "an isothermal wall patch)"))
            return e;
        return rejectGiven(r, {"--downstream", "--ny", "--nz", "--wrinkle", "--wrinkle-modes"},
                           "does not apply with --polymesh: the mesh is the case");
    }
    // combscale S5: --wrinkle A seeds a multi-row [nozzle] chamber's lowest transverse mode (the hll route's).
    if (poly && nozzleWrinkle(r)) {
        if (r.given.contains("--wrinkle-modes"))
            return ConfigError{2, "--wrinkle-modes", "a [nozzle] chamber seeds its lowest transverse mode only"};
        return rejectGiven(r, {"--cells-per-delta", "--downstream", "--ny", "--nz", "--cells-per-unit", "--length"},
                           "does not apply with --polymesh: the mesh is the case");
    }
    if (poly)
        return rejectGiven(r, {"--cells-per-delta", "--downstream", "--ny", "--nz", "--wrinkle", "--wrinkle-modes",
                               "--cells-per-unit", "--length"},
                           "does not apply with --polymesh: the mesh is the case");
    return std::nullopt;
}

// --gamma (masterplan_combnozzle R7): one constant gamma for fresh AND burnt gas, built
// on the compressible premixed C0 flame on a polyMesh only; everything else by name.
std::optional<ConfigError> gammaScope(const RawConfig& r)
{
    if (!r.given.contains("--gamma")) return std::nullopt;
    const bool poly = r.given.contains("--polymesh");
    if (r.physics == Physics::LowMach || (!poly && r.solver != int(SolverKind::Fv)))
        return ConfigError{2, "--gamma",
                           "is the compressible FV polyMesh routes' (structured and low-Mach flames are gamma 1.4)"};
    if (!poly) return ConfigError{2, "--gamma", "needs --polymesh"};
    for (const char* f : {"--equivalence", "--les", "--tfles"})
        if (r.given.contains(f))
            return ConfigError{2, "--gamma", std::format("with {}: not built (the premixed C0 flame's; "
                                                         "masterplan_combnozzle R7)",
                                                         f)};
    if (r.flameMode != 0)
        return ConfigError{2, "--gamma", "with --mode other than premixed: not built (the premixed C0 flame's; "
                                         "masterplan_combnozzle R7)"};
    if (!(r.gamma > 1.0 && r.gamma <= 1.67)) return ConfigError{2, "--gamma", "must be in (1, 1.67]"};
    return std::nullopt;
}

} // namespace

std::optional<ConfigError> fvFlameNozzleScope(const FvBcSpec& bc)
{
    for (const FvPatchSpec& p : bc.patches) {
        if (p.kind == FvPatchKind::Wedge)
            return ConfigError{2, "--bc", std::format("patch `{}`: a wedge on the flame route: not built (next cycle)", p.name)};
        if (p.kind == FvPatchKind::Total && !p.hasY)
            return ConfigError{2, "--bc", std::format("patch.{}: total needs y on a flame (the fresh mass fractions)", p.name)};
    }
    if (bc.nozzle && (bc.nozzle->hasP || bc.nozzle->hasT))
        return ConfigError{2, "--bc", "[nozzle].p, t: the flame route starts from the composition IC (docs/nozzle.md \"Reacting\")"};
    if (bc.thrust && !bc.thrust->station)
        return ConfigError{2, "--bc", "[thrust]: needs station on dns flame (the burnt stagnation plane, masterplan_combnozzle R2)"};
    return std::nullopt;
}

} // namespace dns::cfg

namespace dns::cfg {

std::expected<Case, ConfigError> validateFlame(const RawConfig& r, Case c)
{
    if (auto e = gammaScope(r)) return std::unexpected(*e);
    if (auto e = checkFvFlame(r)) return std::unexpected(*e);
    // --equivalence: the rich two-reactant premixed jet (model C2): the
    // premixed 1D/3D flame (both solvers) and the low-Mach bunsen (masterplan
    // real_bunsen S5: the partially premixed burner).
    const bool premix2 = r.given.contains("--equivalence");
    // --physics lowmach (masterplan real_bunsen S4): the zero-Mach solver's
    // 1D/2D/3D premixed flame (C0 and the --equivalence jet); everything it
    // does not have yet is rejected by name.
    const bool lm = r.physics == Physics::LowMach;
    if (lm) {
        if (r.flameMode != 0 && r.flameMode != 1)
            return bad(2, "--mode",
                       "with --physics lowmach only --mode premixed and bunsen "
                       "are built (diffusion, candle and chamber are the "
                       "compressible solver's)");
        if (r.flameMode == 1)
            if (auto e = rejectGiven(r, {"--pulse", "--pulse-freq"},
                                     "is not built for --physics lowmach (the "
                                     "low-Mach inflow face is steady)");
                e)
                return std::unexpected(*e);
        if (auto e = rejectGiven(r, {"--mach"},
                                 "does not apply to --physics lowmach: no "
                                 "acoustic scale (the low-Mach model is the "
                                 "M -> 0 limit, P0 constant)");
            e)
            return std::unexpected(*e);
        if (auto e = rejectGiven(r,
                                 {"--nscbc-etain", "--nscbc-sink",
                                  "--nscbc-sink-amp", "--nscbc-tinflow"},
                                 "does not apply to --physics lowmach: no "
                                 "NSCBC and no sink (the low-Mach boundaries "
                                 "are face conditions: fixed inflow face "
                                 "values, p = 0 outflow)");
            e)
            return std::unexpected(*e);
        // M4: LES and TFLES are built (lowMachModules below). M5: --turb on the
        // planar flame (both low-Mach solvers); the brush's inflow control is not.
        if (auto e = rejectGiven(r, {"--hold-flame"},
                                 "is not built for --physics lowmach (the "
                                 "brush's inflow control is the structured "
                                 "compressible solver's)");
            e)
            return std::unexpected(*e);
        if (r.turb > 0 && r.flameMode == 1)
            return bad(2, "--turb",
                       "on the low-Mach burner (--mode bunsen): not built (the "
                       "low-Mach --turb is the planar flame box's; the "
                       "turbulent slot is the compressible solver's)");
        if (r.turb > 0 && (r.given.contains("--checkpoint-dt") ||
                           r.given.contains("--restart") ||
                           r.given.contains("--wall-limit")))
            return bad(2, "--turb",
                       "with --checkpoint-dt / --restart / --wall-limit on "
                       "--physics lowmach: not built (no restart byte gate "
                       "for the swept inflow; issues.md M5c)");
    }
    if (premix2 && r.flameMode != 0 && !(r.flameMode == 1 && lm))
        return bad(2, "--equivalence",
                   r.flameMode == 1
                       ? "with --mode bunsen needs --physics lowmach (the "
                         "partially premixed burner is the low-Mach "
                         "solver's)"
                       : "only applies to --mode premixed (a rich premixed "
                         "jet on the two-reactant step) and --mode bunsen "
                         "--physics lowmach");
    if (premix2 && r.flameMode == 1)
        if (auto e = rejectGiven(r, {"--mix-phi"},
                                 "only applies to --mode premixed (the 1D/3D "
                                 "flame of a diluted jet; the burner's jet is "
                                 "--equivalence)");
            e)
            return std::unexpected(*e);
    if (!premix2)
        if (auto e = rejectGiven(r, {"--sl-law", "--mix-phi"},
                                 "only applies with --equivalence (the "
                                 "two-reactant premixed jet)");
            e)
            return std::unexpected(*e);
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
    if (c.physics == Physics::Incompressible)
        return bad(2, "--physics", "a flame is compressible or lowmach (the "
                                   "zero-Mach variable-density solver), never "
                                   "incompressible");
    const bool bunsen = r.flameMode == 1;
    const bool chamber = r.flameMode == kFlameModeChamber;
    if (r.flameMode != 0 && !bunsen && !chamber)
        return bad(2, "--mode", "must be premixed, bunsen, diffusion, candle "
                                "or chamber");
    if (auto e = chamberFlagScope(r, chamber, lm && bunsen); e)
        return std::unexpected(*e);
    if (auto e = wickFlagScope(r, false); e) return std::unexpected(*e);
    // --laterals (masterplan real_bunsen S6, D6): open lateral faces exist on
    // the low-Mach burner only.
    if (r.given.contains("--laterals") && !(lm && bunsen))
        return bad(2, "--laterals",
                   lm ? "only applies to --mode bunsen (the planar flame's y, z "
                        "are its periodic span)"
                      : "only with --mode bunsen --physics lowmach (open "
                        "laterals are built on the low-Mach solver only: the "
                        "compressible species and NSCBC faces are x-only)");
    // --rim (masterplan real_bunsen S8, D6): the immersed isothermal tube
    // exists on the low-Mach burner only; with it the rim anchors the flame.
    const bool rim = r.given.contains("--rim");
    if (rim && !(lm && bunsen))
        return bad(2, "--rim",
                   lm ? "only applies to --mode bunsen (the burner's tube)"
                      : "only with --mode bunsen --physics lowmach (the immersed "
                        "tube is built on the low-Mach solver only)");
    if (!rim)
        if (auto e = rejectGiven(r, {"--tube-length", "--rim-temp"}, "needs --rim");
            e)
            return std::unexpected(*e);
    if (rim)
        if (auto e = rejectGiven(r,
                                 {"--pilot-width", "--pilot-temp", "--pilot-speed",
                                  "--coflow-temp", "--slot-edge", "--pulse",
                                  "--pulse-freq"},
                                 "does not apply with --rim: the rim anchors the "
                                 "flame (no pilot, the room air is cold) and the "
                                 "tube shapes the profile (Poiseuille)");
            e)
            return std::unexpected(*e);
    if (auto e = premix2
                     ? rejectGiven(r,
                                   {"--fuel-y", "--damkohler",
                                    "--cells-per-unit", "--ignition"},
                                   "only applies to --mode diffusion and "
                                   "candle (the --equivalence jet is set by "
                                   "--equivalence, --stoich, --ox-y)")
                     : rejectGiven(r,
                                   {"--stoich", "--fuel-y", "--ox-y",
                                    "--damkohler", "--cells-per-unit",
                                    "--ignition"},
                                   "only applies to --mode diffusion and "
                                   "candle");
        e)
        return std::unexpected(*e);
    // Two-reactant LES: live on the low-Mach solvers (M4, dns/lmmodules: TFLES refused
    // there) and on the compressible FV route (M2: checkFvFlame refused --tfles by the
    // inventory's text, --turb by its allow-list); refused on the structured compressible one.
    const bool fvRich = premix2 && !lm &&
                        (r.given.contains("--polymesh") || r.solver == int(SolverKind::Fv));
    if (premix2 && !lm && !fvRich)
        if (auto e = rejectGiven(r, {"--les", "--tfles", "--turb"},
                                 "with --equivalence: two-reactant LES/TFLES "
                                 "species and the turbulent premixed "
                                 "two-reactant inflow are not built on the "
                                 "structured solver (LES: --solver fv)");
            e)
            return std::unexpected(*e);
    // --gravity: the candle's, and the low-Mach burner's (masterplan
    // real_bunsen S7, D6: compressible bunsen rejected by name).
    if (auto e = rejectGiven(r, {"--quiescent"}, "only applies to --mode candle");
        e)
        return std::unexpected(*e);
    if (r.given.contains("--gravity") && !(lm && bunsen))
        return bad(2, "--gravity",
                   bunsen ? "with --mode bunsen needs --physics lowmach (gravity "
                            "on the burner is built on the low-Mach solver "
                            "only)"
                          : "only applies to --mode candle and --mode bunsen "
                            "--physics lowmach");
    // Each mode's own flags, rejected by name in the other. The low-Mach
    // burner takes --profile-dt: its x profiles of the jet (S6).
    if (bunsen && !lm)
        if (auto e = rejectGiven(r, {"--profile-dt"},
                                 "does not apply to --mode bunsen (x-profiles "
                                 "are a 1D diagnostic; the low-Mach burner's "
                                 "jet profiles need --physics lowmach)");
            e)
            return std::unexpected(*e);
    if (auto e = bunsen ? rejectGiven(r,
                                      {"--upstream", "--downstream",
                                       "--inflow-speed", "--wrinkle",
                                       "--wrinkle-modes", "--ny"},
                                      "does not apply to --mode bunsen (the "
                                      "slot burner sets the box and the IC)")
                        : rejectGiven(r,
                                      {"--slot-width", "--jet-speed",
                                       "--coflow-speed", "--coflow-width",
                                       "--coflow-temp", "--pilot-width",
                                       "--pilot-temp", "--pilot-speed",
                                       "--slot-edge", "--length", "--pulse",
                                       "--pulse-freq", "--nozzle"},
                                      "only applies to --mode bunsen");
        e)
        return std::unexpected(*e);
    const bool premixed = !bunsen && !chamber;
    if (auto e = turbFlagScope(r, premixed, bunsen); e)
        return std::unexpected(*e);
    if (auto e = lesTflesFlagScope(r, !chamber); e) return std::unexpected(*e);
    if (chamber && r.nz != 1)
        return bad(2, "--nz", "must be 1: 3D --mode chamber is not built");
    if (r.nz != 1 && r.nz < 8)
        return bad(2, "--nz", bunsen ? "must be 1 or >= 8 (a 3D slot Bunsen: "
                                       "z periodic, hz = h)"
                                     : "must be 1 or >= 8 (a 3D premixed "
                                       "flame: z periodic, hz = h)");
    if (premixed || bunsen) {
        if (!(std::isfinite(r.turb) && r.turb >= 0 && r.turb <= 20))
            return bad(2, "--turb", "must be in [0, 20] (u'/S_L; 0 = off)");
        const bool turb = r.turb > 0;
        if (turb && r.nz == 1)
            return bad(2, "--turb", "needs --nz >= 8: 2D turbulence is not "
                                    "turbulence");
#if !DNS_WITH_FFTW
        if (turb)
            return bad(2, "--turb", "needs FFTW: this binary was built "
                                    "without it (DNS_WITH_FFTW)");
#endif
        if (!turb)
            if (auto e = rejectGiven(r, {"--turb-length", "--hold-flame"},
                                     "needs --turb > 0");
                e)
                return std::unexpected(*e);
        if (!turb)
            if (auto e = rejectGiven(r, {"--seed"},
                                     "only seeds the --turb box in dns flame");
                e)
                return std::unexpected(*e);
    }
    if (!chamber)
        if (auto e = flameLesTfles(r, c, bunsen); e) return std::unexpected(*e);
    if (lm)
        if (auto e = lowMachModules(r, c, premix2, rim); e) return std::unexpected(*e);

    if (!bunsen && r.ny < 4) return bad(2, "--ny", "must be >= 4 (periodic y)");
    c.mach = chamber && !r.given.contains("--mach") ? 0.02 : r.mach;
    if (lm) {
        c.mach = 0; // the M -> 0 limit: no acoustic scale
        if (!(std::isfinite(r.chemDt) && r.chemDt > 0 && r.chemDt <= 2.5))
            return bad(2, "--chem-dt", "must be in (0, 2.5] (the chemistry dt "
                                       "is K / max J; SSP-RK3's real-axis "
                                       "stability bound is 2.51)");
        c.lmChemDt = r.chemDt;
        c.lmDtCell = r.given.contains("--dt-rule") ? r.dtRule == 1 : bunsen;
    } else if (!(finitePos(c.mach) && c.mach <= 0.05))
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
    // TFLES: the THICKENED flame is what the grid resolves (F delta_th).
    if (c.flameTfles > 0) {
        if (!(std::isfinite(r.cellsPerDelta) && r.cellsPerDelta > 0 &&
              c.flameTfles * r.cellsPerDelta >= 4))
            return bad(2, "--cells-per-delta",
                       std::format("must be > 0 with --tfles F * "
                                   "--cells-per-delta >= 4 (cells across the "
                                   "thickened flame; F = {:g})",
                                   c.flameTfles));
    } else if (!(std::isfinite(r.cellsPerDelta) && r.cellsPerDelta >= 4))
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
    // combscale S5: a [nozzle] chamber's seed is one mode on the mesh's rows (the run checks it has two).
    const bool oneMode = nozzleWrinkle(r);
    if (!oneMode && r.wrinkle > 0 && (r.wrinkleModes < 1 || 8 * r.wrinkleModes > r.ny))
        return bad(2, "--wrinkle-modes", "must be in [1, ny/8] (8 cells per "
                                         "shortest seeded wavelength)");
    if (r.wrinkle > 0 && r.wrinkle * std::sqrt(double(oneMode ? 1 : r.wrinkleModes)) >
                             std::min(r.upstream, r.downstream) / 2)
        return bad(2, "--wrinkle", "peak displacement A sqrt(N) must stay "
                                   "within half of --upstream and --downstream");
    c.nscbcEtaIn = r.nscbcEtaIn;
    if (!finitePos(c.nscbcEtaIn))
        return bad(2, "--nscbc-etain", "must be finite and > 0");
    c.nscbcThermalInflow = r.nscbcTinflow;
    // Low-Mach: no NSCBC, no sink band (face conditions; the flags were
    // rejected above).
    c.nscbcSink = lm                              ? 0
                  : r.given.contains("--nscbc-sink") ? r.nscbcSink
                                                     : 16;
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
    if (premix2) {
        if (!(std::isfinite(r.equivalence) && r.equivalence >= 1.05 &&
              r.equivalence <= 1.6))
            return bad(2, "--equivalence",
                       "must be in [1.05, 1.6] (rich premix only: a lean or "
                       "stoichiometric jet has no outer cone; 1.6 below "
                       "CH4's rich limit 1.67)");
        PartialPremixModel pm;
        pm.phi = r.equivalence;
        pm.s = r.given.contains("--stoich") ? r.stoich : 3.99;
        pm.yO2 = r.given.contains("--ox-y") ? r.oxY : 0.233;
        if (!(finitePos(pm.s) && pm.s <= 100))
            return bad(2, "--stoich", "must be in (0, 100] (oxidizer mass per "
                                      "fuel mass)");
        if (!(finitePos(pm.yO2) && pm.yO2 <= 1))
            return bad(2, "--ox-y", "must be in (0, 1] (the air's oxidizer "
                                    "mass fraction)");
        pm.sigma = md.sigma;
        pm.beta = md.beta;
        pm.Pr = md.Pr;
        pm.sutherland = md.sutherland;
        pm.sOverTu = md.sOverTu;
        // Model C2b: GRI-3.0 is CH4/air, so the law fixes s and Y_O2.
        const bool law = r.slLaw == 0;
        if (law && (pm.s != 3.99 || pm.yO2 != 0.233))
            return bad(2, pm.s != 3.99 ? "--stoich" : "--ox-y",
                       "with --sl-law gri30 (the --equivalence default) must "
                       "be CH4/air's (--stoich 3.99, --ox-y 0.233): the A(phi) "
                       "law is calibrated on GRI-3.0 CH4/air; another fuel "
                       "needs --sl-law onestep");
        const bool mix = r.given.contains("--mix-phi");
        if (mix && !(std::isfinite(r.mixPhi) && r.mixPhi >= 0.6 &&
                     r.mixPhi <= pm.phi && std::abs(r.mixPhi - 1) >= 0.01))
            return bad(2, "--mix-phi",
                       std::format("must be in [0.6, {:g}] (the jet diluted "
                                   "with air: 0.6 the leanest calibrated "
                                   "mixture, at most the jet's phi) and "
                                   "|PHI - 1| >= 0.01 (the oracle's hot-end "
                                   "start needs a leftover reactant)",
                                   pm.phi));
        if (law) {
            auto L = phiRateLaw(pm);
            if (!L) return bad(2, "--sl-law", L.error());
            c.premixLaw = *L;
        }
        auto eig = mix ? premixedEigen2At(pm, r.mixPhi,
                                          law ? &c.premixLaw : nullptr)
                       : premixedEigen2(pm);
        if (!eig) return bad(2, mix ? "--mix-phi" : "--equivalence",
                             eig.error());
        c.flamePremix2 = true;
        c.premix = pm;
        c.premixMixPhi = mix ? r.mixPhi : 0.0;
        c.flameEig = std::move(*eig);
    } else {
        auto eig = premixedEigen(md);
        if (!eig) return bad(2, "--zeldovich", eig.error());
        c.flameEig = std::move(*eig);
    }
    c.flameModel = md;
    if (chamber) return finishChamber(r, std::move(c));
    c.flameCells = r.cellsPerDelta;
    c.flameUp = r.upstream;
    c.flameDown = r.downstream;
    c.flameUIn = r.inflowSpeed;
    c.flameWrinkle = r.wrinkle;
    c.flameWrinkleModes = r.wrinkleModes;
    // Low-Mach box units: Tu = P0 = R = rho_u = 1 (no acoustic scale). gammaScope
    // left r.gamma at kGammaAir everywhere but the premixed polyMesh flame.
    c.fvGamma = r.gamma;
    c.flameTu = lm ? 1.0 : 1.0 / (c.fvGamma * c.mach * c.mach);

    const double dth = c.flameEig.dthOverDD;
    if (bunsen && !r.given.contains("--cells-per-delta"))
        c.flameCells = kBunsenCellsDefault;
    const double h = dth / c.flameCells;
    const int nb = c.nscbcSink;
    if (lm) {
        // The pressure solve's FFT sizes (K11): y, z periodic need 7-smooth
        // sizes (an error naming the nearest), x's DCT-IV a 7-smooth nx/2
        // (nx derived: rounded up, the cells added downstream).
        for (auto [flag, n] : {std::pair{"--ny", r.ny}, std::pair{"--nz", r.nz}})
            if (n > 1 && !lmFftSmooth(n))
                return bad(2, flag,
                           std::format("{} is not FFT-friendly for the low-Mach "
                                       "pressure solve (prime factors <= 7 "
                                       "only): nearest {} or {}",
                                       n, lmPrevSmooth(n), lmNextSmooth(n)));
    }
    if (!(bunsen && lm))
        if (auto e = rejectGiven(r, {"--si-delta-d", "--si-sl"},
                                 "only echoes the low-Mach burner (--mode bunsen "
                                 "--physics lowmach) in SI units");
            e)
            return std::unexpected(*e);
    if (bunsen) {
        auto e = bunsenGeometry(r, c, nb);
        if (!e) return std::unexpected(e.error());
        if (lm) {
            // S6: open laterals (y; z too around the round nozzle -- the
            // slot's z is its periodic span) and the inert jet.
            c.lmOpenY = r.laterals == 1;
            c.lmOpenZ = c.lmOpenY && c.bunsen.round;
            c.lmInert = r.chamber.reaction == 0;
            // S7: gravity along -x (absent = 0: the S6 solver bit for bit).
            c.lmGravity = r.given.contains("--gravity") ? r.gravity : 0.0;
            if (!(std::isfinite(c.lmGravity) && c.lmGravity >= 0 &&
                  c.lmGravity <= 100))
                return bad(2, "--gravity",
                           std::format("must be in [0, 100] S_L^2/delta_D along "
                                       "-x (the jet axis points up; real g, "
                                       "CH4/air phi 1.3 on this model's "
                                       "delta_th/delta_D {:.4g}: {:.4g})",
                                       c.flameEig.dthOverDD,
                                       lmRealGravity(c.flameEig.dthOverDD)));
            if (c.lmGravity > 0 && !c.lmOpenY && !r.given.contains("--polymesh"))
                c.warnings.push_back(
                    "--gravity with periodic laterals: a periodic box under "
                    "gravity is a lattice of plumes (the candle's chimney; the "
                    "buoyant gas cannot entrain room air) -- --laterals open is "
                    "the burner in a room");
            // S8: the immersed isothermal tube (absent = the S7 solver bit for
            // bit).
            c.lmRim = rim;
            if (rim && !c.lmOpenY)
                c.warnings.push_back(
                    "--rim with periodic laterals: a lattice of burners (the "
                    "room draft is the only air supply) -- --laterals open is "
                    "the burner in a room");
            // S9: the SI echo (label, dry-run, .json "si"; no physics).
            const bool siD = r.given.contains("--si-delta-d");
            if (siD != r.given.contains("--si-sl"))
                return bad(2, siD ? "--si-sl" : "--si-delta-d",
                           "is required with the other SI flag (--si-delta-d MM "
                           "and --si-sl M_PER_S come together)");
            if (siD) {
                if (!(finitePos(r.siDeltaD) && r.siDeltaD < 10))
                    return bad(2, "--si-delta-d",
                               "must be in (0, 10) mm (the model's delta_D: "
                               "the real delta_th over the model's "
                               "delta_th/delta_D)");
                if (!(finitePos(r.siSL) && r.siSL < 100))
                    return bad(2, "--si-sl", "must be in (0, 100) m/s (S_L)");
                c.siDeltaD = r.siDeltaD * 1e-3; // mm -> m
                c.siSL = r.siSL;
            }
        }
        if (premix2 && !c.lmInert && !c.lmRim && !(c.bunsen.pilot > 0) &&
            c.bunsen.tCo < c.flameModel.sigma)
            c.warnings.push_back(
                "--equivalence --mode bunsen without --pilot-width: nothing "
                "hot anchors the rich cone against the cold air coflow (the "
                "compressible burner blew off on a cold coflow; the rim is "
                "S8) -- watch xBase, exit 8 on blow-off");
    } else {
        int nx = int(std::ceil((c.flameUp + c.flameDown) * c.flameCells -
                               1e-9)) + 2 * nb;
        if (lm) {
            const int nx0 = nx;
            nx = 2 * lmNextSmooth((nx + 1) / 2);
            c.lmNxPad = nx - nx0;
        }
        // 3D: z periodic at the same spacing (hz = h); nz == 1 keeps hz = 1.
        auto box = makeBox(nx, r.ny, r.nz, nx * h, r.ny * h,
                           r.nz == 1 ? 1.0 : r.nz * h);
        if (!box) return std::unexpected(box.error());
        c.grid = *box;
        c.flameXf0 = nb * h + c.flameUp * dth;
        c.uInf = c.flameUIn;
        if (r.turb > 0) {
            if (auto e = flameTurbSetup(r, c, c.flameUIn, "U_in"); e)
                return std::unexpected(*e);
            c.flameHold = r.holdFlame;
        }
    }
    if (c.flameTfles > 0) {
        // The sensor's max filter: kTflesDilate thickened thermal
        // thicknesses, in cells.
        c.flameTflesDilate =
            int(std::ceil(kTflesDilate * c.flameTfles * c.flameCells - 1e-9));
        // --polymesh: the runner measures the mesh (cells, radius, the >= 4 rule).
        const double thick = c.flameTfles * c.flameCells;
        if (thick < 6 && !r.given.contains("--polymesh"))
            c.warnings.push_back(std::format(
                "--tfles {:g} at --cells-per-delta {:g}: {:.3g} cells across "
                "the thickened flame (below 6: the laminar flame speed carries "
                "the grid's error; docs/chem.md)",
                c.flameTfles, c.flameCells, thick));
    }
    c.n = c.grid.nx;
    c.h = h;
    c.L = c.grid.nx * h;
    c.re = 1.0 / c.prandtl; // rho_u S_L delta_D / mu
    c.nu = c.prandtl;       // mu (rho_u = 1): lambda_u = cp
    c.far = FarField::Nscbc;
    c.bodyKind = BodyKind::None;

    // Low-Mach: the convective CFL of the central MAC SSP-RK3 (no acoustics).
    if (auto e = resolveCfl(r, c, lm ? kLmCflDefault : 0.4); e)
        return std::unexpected(*e);
    if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
    c.ft = r.tt;
    c.tEnd = r.tt; // tau = delta_D / S_L = 1
    if (auto e = checkProfileDt(r, c); e) return std::unexpected(*e);
    // The low-Mach burner recomputes dt EVERY step by default (S5, measured
    // 2026-10-31): its IC is not the projected flow -- the first projection
    // sets velocities ~2x the IC's in the products, and a dt held over 16
    // steps diverged the 4-cell partially premixed slot within 50 steps.
    if (auto e = resolveDtRecompute(r, c, lm && bunsen ? 1 : 16); e)
        return std::unexpected(*e);
    if (auto e = validateFrameOutputs(r, c); e) return std::unexpected(*e);
    if (auto e = resolveEmit(c, {Emit::Stats}); e) return std::unexpected(*e);

    if (c.mach > 0.02)
        c.warnings.push_back(std::format(
            "--mach {:.4g} is above the measured ladder top 0.02: the "
            "compressible flame speed carries an O(M^2) offset "
            "(gamma M^2 (sigma - 1) = {:.3g})",
            c.mach, c.fvGamma * c.mach * c.mach * (md.sigma - 1)));
    if (nb == 0 && !lm)
        c.warnings.push_back("--nscbc-sink 0: the bare characteristic "
                             "boundary (nothing drains mesh-scale content)");
    c.meta = {.D = 1.0,
              .re = c.re,
              .mach = c.mach,
              .frameDt = c.fdt,
              // Flame LES: the .json carries Cs and the model like every
              // other LES run (DNS: 0 / false, the pre-LES bytes).
              .cs = c.lesLabel ? c.cs : 0.0,
              .les = c.lesLabel,
              .body = c.flameBunsen ? "slot Bunsen flame" : "premixed flame",
              .farField = lm ? (c.lmOpenY ? "lowmach-open" : "lowmach-faces")
                             : "nscbc",
              .physics = name(c.physics),
              .kick = 0.0,
              .sgs = c.lesLabel ? name(c.sgsModel) : "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    return c;
}

} // namespace dns::cfg
