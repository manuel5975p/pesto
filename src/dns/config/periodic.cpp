// validate(): dns tgv, dns hit (+ the hit SI design for --dry-run).
#include "dns/config.hpp"
#include "dns/config/internal.hpp"


#include "dns/units.hpp"

#include <cassert>
#include <cmath>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

using std::numbers::pi;

namespace dns::cfg {
namespace {

// --- tgv / hit --------------------------------------------------------------
// dns tgv --solver fv (docs/fvincomp.md). Without it every flag of the FV solver
// is rejected by name.
std::optional<ConfigError> resolveFvTgv(const RawConfig& r, Case& c)
{
    if (r.sub != Subcommand::Tgv) return std::nullopt;
    if (r.solver != int(SolverKind::Fv))
        return rejectGiven(r, {"--mesh-perturb", "--mesh-perturb-kind", "--les", "--sgs", "--tol", "--max-iters"},
                           "belongs to dns tgv --solver fv");
    if (r.given.contains("--emit") && (c.emit.contains(Emit::Slices) || c.emit.contains(Emit::Vtk)))
        return err(2, "--emit", "dns tgv --solver fv writes stats and error only");
    if (!finitePos(r.tol)) return err(2, "--tol", "must be finite and > 0");
    if (r.maxIters < 1) return err(2, "--max-iters", "must be >= 1");
    const bool given = r.given.contains("--mesh-perturb");
    if (r.given.contains("--mesh-perturb-kind") && !given)
        return err(2, "--mesh-perturb-kind", "needs --mesh-perturb A");
    if (given) {
        const bool sine = r.meshPerturbKind == 0;
        if (!(std::isfinite(r.meshPerturb) && r.meshPerturb > 0 && r.meshPerturb <= (sine ? 0.06 : 0.45)))
            return err(2, "--mesh-perturb", sine ? "sine: must be in (0, 0.06]" : "random: must be in (0, 0.45]");
    }
    if (r.les < 0 || !std::isfinite(r.les)) return err(2, "--les", "must be finite and >= 0");
    if (r.given.contains("--sgs") && r.les == 0) return err(2, "--sgs", "selects the SGS model for --les");
    if (r.les > 0 && r.sgs == SgsModel::Sigma && c.grid.nz == 1)
        return err(2, "--sgs", "sigma: D_sigma is identically ZERO for any 2D state");
    c.fvIncomp = true;
    c.fvPerturb = given ? r.meshPerturbKind + 1 : 0;
    c.fvPerturbAmp = given ? r.meshPerturb : 0;
    c.fvPrecision = FvPrecision(r.fvPrecision);
    c.fvTol = r.given.contains("--tol") ? r.tol : 1e-5;
    c.fvMaxPcg = r.given.contains("--max-iters") ? r.maxIters : 200;
    c.cs = r.les;
    c.sgsModel = r.sgs;
    return std::nullopt;
}

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

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
    if (auto e = resolveFvTgv(r, c); e) return std::unexpected(*e);
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

} // namespace dns::cfg
