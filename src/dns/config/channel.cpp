// validate(): dns channel (DNS, LES, RANS).
#include "dns/config.hpp"
#include "dns/config/internal.hpp"


#include "dns/pseudo_time.hpp"
#include "dns/remap.hpp"
#include "dns/steady_stop.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

using std::numbers::pi;

namespace dns::cfg {
namespace {

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

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

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

} // namespace dns::cfg
