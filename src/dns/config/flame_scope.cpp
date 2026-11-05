// validate(): dns flame flag scopes and the resolutions every flame mode shares (turb, LES/TFLES, low-Mach modules, frames, checkpoints).
#include "dns/config.hpp"
#include "dns/config/internal.hpp"

#include "dns/lmmodules.hpp"

#include "dns/turb_ic.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace dns::cfg {

std::optional<ConfigError> chamberFlagScope(const RawConfig& r, bool chamber,
                                            bool lmBurner)
{
    if (chamber)
        return rejectGiven(r, {"--ny", "--upstream", "--inflow-speed",
                               "--profile-dt", "--wrinkle", "--wrinkle-modes"},
                           "does not apply to --mode chamber (the grid is "
                           "derived from the chamber; fresh gas enters "
                           "through the injector, --inject-speed)");
    if (auto e = rejectGiven(r, {"--inject-speed", "--coflow", "--chamber-height",
                                 "--chamber-length", "--nozzle-ratio",
                                 "--holder-height", "--clearance",
                                 "--inject-ramp"},
                             "only applies to --mode chamber");
        e)
        return e;
    if (lmBurner) return std::nullopt;
    return rejectGiven(r, {"--reaction"},
                       "only applies to --mode chamber and --mode bunsen "
                       "--physics lowmach");
}

std::optional<ConfigError> turbFlagScope(const RawConfig& r, bool premixed,
                                         bool bunsen)
{
    if (premixed) return std::nullopt;
    if (bunsen)
        return rejectGiven(r, {"--hold-flame"},
                           "only applies to --mode premixed (the slot "
                           "burner's jet speed is fixed)");
    return rejectGiven(r, {"--turb", "--turb-length", "--hold-flame", "--seed"},
                       "only applies to --mode premixed and bunsen (3D, --nz "
                       ">= 8)");
}

std::optional<ConfigError> flameTurbSetup(const RawConfig& r, Case& c,
                                          double uMax, const char* uName)
{
    assert(r.turb > 0);
    const double dth = c.flameEig.dthOverDD, h = dth / c.flameCells;
    const double lt = r.turbLength * dth;
    if (!(std::isfinite(r.turbLength) && lt >= 4 * h))
        return err(2, "--turb-length",
                   std::format("must be >= 4 cells ({:.4g} thermal "
                               "thicknesses at --cells-per-delta {:g})",
                               4.0 / c.flameCells, c.flameCells));
    // kp/k0 = 0.40 L/l_t: 6 l_t puts the spectral peak on the 2.4th
    // lattice shell of the shortest periodic axis.
    const int ny = c.grid.ny, nz = c.grid.nz;
    if (ny * h < 6 * lt || nz * h < 6 * lt)
        return err(2, "--turb-length",
                   std::format("must be <= L/6 of both periodic axes "
                               "({:.4g} thermal thicknesses: --ny {} "
                               "--nz {} at --cells-per-delta {:g})",
                               std::min(ny, nz) * h / (6 * dth), ny, nz,
                               c.flameCells));
    const double uM = uMax + 3 * r.turb;
    if (uM * c.mach > 0.3)
        return err(2, "--turb", std::format("the fresh gas Mach M ({} + 3 "
                                            "u') = {:.3g} exceeds 0.3",
                                            uName, uM * c.mach));
    c.flameTurb = r.turb;
    c.flameTurbL = lt;
    c.flameTurbNbx = nz; // L_bx = L_z
    const TurbNumbers tn = turbNumbers(r.turb, lt, c.prandtl, dth);
    if (h / tn.eta > 2 && !c.lesLabel)
        c.warnings.push_back(std::format(
            "h/eta = {:.3g} at the inflow (Re_t = {:.3g}): the "
            "velocity field is under-resolved -- not a DNS",
            h / tn.eta, tn.ReT));
    return std::nullopt;
}

std::optional<ConfigError> lesTflesFlagScope(const RawConfig& r, bool premixed)
{
    if (premixed) return std::nullopt;
    return rejectGiven(r, {"--les", "--sgs", "--tfles", "--tfles-sensor",
                           "--efficiency"},
                       "only applies to --mode premixed and bunsen (LES "
                       "species and the thickened flame are premixed-only)");
}

std::optional<ConfigError> flameLesTfles(const RawConfig& r, Case& c,
                                         bool bunsen)
{
    if (!(std::isfinite(r.les) && r.les >= 0))
        return err(2, "--les", "must be finite and >= 0 (0 = DNS)");
    if (r.given.contains("--sgs") && r.les == 0)
        return err(2, "--sgs", "selects the SGS model for --les; give --les CS "
                               "(> 0) or drop --sgs");
    if (r.les > 0) {
        // --polymesh (the FV path) has no --nz: the run refuses a 2D mesh (exit 3).
        if (r.nz == 1 && !r.given.contains("--polymesh"))
            return err(2, "--les", bunsen
                                       ? "needs --nz >= 8 (a 3D slot Bunsen): "
                                         "2D LES is regularized 2D Euler, not a "
                                         "filtered turbulent field (--tfles "
                                         "alone thickens the 2D cone)"
                                       : "needs --nz >= 8 (a 3D premixed "
                                         "flame): 2D LES is regularized 2D "
                                         "Euler, not a filtered turbulent "
                                         "field");
        if (!r.given.contains("--sgs"))
            return err(2, "--les", "needs an explicit --sgs in dns flame: "
                                   "--sgs sigma --les 1.35 recommended -- "
                                   "D_sigma is exactly 0 on a laminar flame's "
                                   "1D dilatation, Smagorinsky's |S| is not "
                                   "(it diffuses a laminar flame)");
        c.cs = r.les;
        c.sgsModel = r.sgs;
        c.lesLabel = true;
    }
    if (!(std::isfinite(r.tfles) && (r.tfles == 0 || (r.tfles >= 1 &&
                                                       r.tfles <= 20))))
        return err(2, "--tfles", "must be 0 (off) or in [1, 20] (the "
                                 "thickening factor F)");
    if (r.tfles == 0) {
        if (auto e = rejectGiven(r, {"--tfles-sensor", "--efficiency"},
                                 "needs --tfles F (>= 1)");
            e)
            return e;
        return std::nullopt;
    }
    const bool charlette = r.given.contains("--efficiency")
                               ? r.efficiency == 1
                               : c.lesLabel;
    if (charlette && !c.lesLabel)
        return err(2, "--efficiency", "charlette needs --les (the sub-grid "
                                      "velocity of a filtered field; without "
                                      "LES give --efficiency none)");
    c.flameTfles = r.tfles;
    c.flameTflesUniform = r.tflesSensor == 1;
    c.flameCharlette = charlette;
    return std::nullopt;
}

std::optional<ConfigError> lowMachModules(const RawConfig& r, const Case& c, bool premix2,
                                          bool rim)
{
    using M = LmModule;
    LmModuleSet s;
    if (c.lesLabel) s.set(std::size_t(c.sgsModel == SgsModel::Sigma ? M::LesSigma : M::LesSmag));
    if (c.flameTfles > 0) s.set(std::size_t(c.flameTflesUniform ? M::TflesUniform : M::TflesDynamic));
    if (c.flameTfles > 0 && c.flameCharlette) s.set(std::size_t(M::TflesEfficiency));
    s.set(std::size_t(!premix2 ? M::ChemC0 : r.slLaw == 0 ? M::ChemC2b : M::ChemC2));
    if (r.given.contains("--gravity") && r.gravity > 0) s.set(std::size_t(M::Gravity));
    if (r.given.contains("--laterals") && r.laterals == 1) s.set(std::size_t(M::Open));
    if (rim) s.set(std::size_t(M::Rim));
    if (r.turb > 0) s.set(std::size_t(M::Turb));
    auto ok = lmModulesCheck(s);
    if (ok) return std::nullopt;
    const bool turbPair = r.turb > 0 && lmModulesCheck(LmModuleSet(s).reset(std::size_t(M::Turb))).has_value();
    const char* flag = turbPair ? "--turb"
                       : c.flameTfles > 0 && !(s.test(std::size_t(M::Rim)) && c.lesLabel) ? "--tfles"
                                                                                          : "--les";
    return ConfigError{2, flag, ok.error() + " (docs/fvincomp.md \"Low-Mach combinations\")"};
}

std::optional<ConfigError> resolveFrameStart(const RawConfig& r, Case& c)
{
    if (!r.given.contains("--frame-start")) return std::nullopt;
    if (!(std::isfinite(r.frameStart) && r.frameStart >= 0 &&
          r.frameStart < c.tEnd))
        return err(2, "--frame-start", "must be in [0, --tt)");
    const bool frames = std::ranges::any_of(c.emit, [](Emit e) {
        return e == Emit::Bin || e == Emit::Volume || e == Emit::Slices ||
               e == Emit::Png || e == Emit::Web || e == Emit::Shm;
    });
    if (!frames)
        return err(2, "--frame-start", "needs a frame --emit (bin, volume, "
                                       "slices, png, web or shm)");
    c.frameStart = r.frameStart;
    return std::nullopt;
}

std::optional<ConfigError> resolveCheckpoint(const RawConfig& r, Case& c)
{
    c.caseFlags = r.caseFlags;
    const bool ck = r.given.contains("--checkpoint-dt");
    const bool wl = r.given.contains("--wall-limit");
    const bool rs = r.given.contains("--restart");
    if (!ck && !wl && !rs) return std::nullopt;
    const char* first = ck ? "--checkpoint-dt" : wl ? "--wall-limit" : "--restart";
    if (c.flameWick || c.chamberMode)
        return err(2, first, "not wired for --mode candle --wick or --mode "
                             "chamber (checkpoint/restart: dns flame premixed, "
                             "bunsen, diffusion, candle)");
    if (ck && !(std::isfinite(r.checkpointDt) && r.checkpointDt > 0))
        return err(2, "--checkpoint-dt", "must be > 0 (model time between "
                                         "checkpoints; omit it for none)");
    if (wl && !ck)
        return err(2, "--wall-limit", "needs --checkpoint-dt (a wall-limit "
                                      "stop is a checkpoint stop)");
    if (wl && !(std::isfinite(r.wallLimit) && r.wallLimit >= 0))
        return err(2, "--wall-limit", "must be >= 0 seconds");
    if (rs && r.restart != c.outPrefix)
        return err(2, "--restart",
                   std::format("must equal --out (a restart continues that "
                               "run's outputs): --restart {} vs --out {}",
                               r.restart, c.outPrefix));
    for (const Emit e : c.emit)
        if (e == Emit::Web || e == Emit::Vtk)
            return err(2, "--emit",
                       std::format("{} cannot resume a checkpointed run (its "
                                   "manifest is rebuilt per process): with "
                                   "{} use stats, bin, volume, slices, png, "
                                   "shm",
                                   e == Emit::Web ? "web" : "vtk", first));
    if (rs && r.restartT >= 0 && !(c.tEnd > r.restartT))
        return err(2, "--tt",
                   std::format("{} is not past the checkpoint's t = {} "
                               "(--restart continues a run to a later --tt)",
                               c.tEnd, r.restartT));
    c.checkpointDt = ck ? r.checkpointDt : 0;
    c.wallLimit = wl ? r.wallLimit : -1;
    c.restartPrefix = rs ? r.restart : std::string();
    return std::nullopt;
}

} // namespace dns::cfg
