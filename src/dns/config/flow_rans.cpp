// validate(): dns flow --rans.
#include "dns/config.hpp"
#include "dns/config/internal.hpp"


#include "dns/precond.hpp"
#include "dns/pseudo_time.hpp"
#include "dns/steady_stop.hpp"

#include <cassert>
#include <cmath>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

using std::numbers::pi;

namespace dns::cfg {

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

double ransFirstCellDistance(const Case& c)
{
    return c.ogrid ? c.ogridH1 : 0.5 * hFine(c);
}

double ransYplusEstimate(double h1, double D, double re)
{
    return h1 / D * re * 0.38 * std::pow(re, -0.1);
}

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
        const bool fvWall = r.solver == int(SolverKind::Fv) && c.noslipY;
        if (!fvWall && r.solver == int(SolverKind::Fv) && c.bodyKind == BodyKind::None)
            return err(2, "--rans", "needs a wall: --ogrid, or --noslip on the "
                                    "FV box");
        if (!fvWall && (c.noslipY || c.bodyKind == BodyKind::None ||
            c.bodyKind == BodyKind::Floor || c.bodyKind == BodyKind::Fence))
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

} // namespace dns::cfg
