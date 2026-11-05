// validate(): dns conv, structured and FaceMesh (--solver fv, --polymesh).
#include "dns/config.hpp"
#include "dns/config/internal.hpp"


#include "dns/transport.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <optional>
#include <string>
#include <string_view>

namespace dns::cfg {

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
    if (auto e = checkWallGrid(r); e) return std::unexpected(*e);
    const int nyI = r.ny;
    const int nz = r.nz;

    if (!finitePos(r.rayleigh))
        return bad(2, "--rayleigh", "must be finite and > 0");
    c.rayleigh = r.rayleigh;
    if (!(finitePos(r.dtemp) && r.dtemp < 2))
        return bad(2, "--dtemp", "must be in (0,2): the imposed drop as a "
                                 "fraction of the mid-layer temperature, "
                                 "so the lid sits at 1 - dtemp/2 > 0");
    c.dtemp = r.dtemp;
    c.mach = r.mach;
    if (auto e = checkMach(c.mach); e) return std::unexpected(*e);
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

    const double gamma = kGammaAir, Rgas = kRgasBox, H = 1.0, Tref = 1.0;
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
    auto box = makeBox(nxT, nyT, nzT, lx, ly, lz);
    if (!box) return std::unexpected(box.error());
    c.grid = *box;
    c.nxInterior = r.n;
    c.nzInterior = nz;
    c.h = c.grid.hy;
    c.L = c.lxBox;
    c.pRef = Rgas * Tref;   // rho_ref = 1 at mid-layer

    // The compressible branch is acoustically limited, the Boussinesq one
    // advectively; 0.4 and 0.8 are those two solvers' own defaults.
    if (auto e = validateWallRun(
            r, c, c.physics == Physics::Incompressible ? 0.8 : 0.4);
        e)
        return std::unexpected(*e);
    c.tEnd = r.tt * H / c.uFreeFall;   // free-fall times -> box time
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
    c.meta = {.D = H,
              .re = c.re,
              .mach = c.mach,
              .frameDt = c.fdt,
              .body = "convection",
              .farField = c.closedSides ? "closed-sides" : "periodic",
              .physics = name(c.physics),
              .kick = c.kick,
              .sgs = "",
              .rans = "",
              .steady = ""};
    c.meta.setGrid(c.grid);
    if (auto e = resolveFvConv(r, c); e) return std::unexpected(*e);
    return c;
}

std::optional<ConfigError> resolveFvConv(const RawConfig& r, Case& c)
{
    const bool poly = r.given.contains("--polymesh");
    if (poly && r.given.contains("--solver") && r.solver != int(SolverKind::Fv))
        return ConfigError{2, "--solver", "--polymesh is the FV path: only --solver fv goes with it"};
    const bool comp = c.physics == Physics::Compressible;
    if (r.solver != int(SolverKind::Fv) && !poly) {
        if (auto e = rejectGiven(r, {"--les", "--sgs"},
                                 "is built on dns conv --solver fv --physics incompressible only (the structured "
                                 "conv solvers carry no SGS model)");
            e)
            return e;
        return rejectGiven(r, {"--bc", "--fv-precision", "--compensated", "--tol", "--max-iters", "--frame-box", "--frame-n",
                               "--mesh-perturb", "--mesh-perturb-kind"},
                           "belongs to dns conv --solver fv");
    }
    static const char* const common[] = {"--solver", "--physics", "--rayleigh", "--prandtl", "--ny",
                                         "--aspect", "--sides",   "--tt",       "--cfl",     "--kick",
                                         "--out",    "--emit",    "--frame-box", "--frame-n", "--fv-precision",
                                         "--dt-recompute", "--profile-dt", "--fdt", "--pdt", "--wdt",
                                         "--png-down", "--backend", "--quiet",  "--dry-run"};
    static const char* const box[] = {"--n", "--nz", "--mesh-perturb", "--mesh-perturb-kind"};
    static const char* const mesh[] = {"--polymesh", "--bc"};
    static const char* const compOnly[] = {"--mach", "--dtemp", "--visc", "--tinf", "--compensated"};
    static const char* const incompOnly[] = {"--tol", "--max-iters", "--les", "--sgs"};
    auto in = [](std::span<const char* const> l, const std::string& f) {
        return std::ranges::any_of(l, [&](const char* x) { return f == x; });
    };
    for (const std::string& g : r.given) {
        const bool ok = in(common, g) || (poly ? in(mesh, g) : in(box, g)) || (comp ? in(compOnly, g) : in(incompOnly, g));
        if (ok) continue;
        return ConfigError{2, g, g + (in(box, g) ? " belongs to the mesh on --polymesh"
                                  : in(mesh, g)  ? " needs --polymesh DIR"
                                                 : " is not built on dns conv --solver fv (docs/fvcomp.md)")};
    }
    if (poly) {
        if (r.polymesh.empty()) return ConfigError{2, "--polymesh", "needs a mesh directory"};
        if (!r.given.contains("--bc") || r.bc.empty()) return ConfigError{2, "--bc", "--polymesh needs --bc FILE.toml"};
        if (!r.given.contains("--ny"))
            return ConfigError{2, "--ny", "required on --polymesh: the height bins of the plane statistics"};
        if (comp && !r.fvBc) return ConfigError{2, "--bc", "--physics compressible --polymesh needs --bc FILE.toml"};
        if (comp)
            if (const std::string k = fvBcFlameKey(*r.fvBc); !k.empty())
                return ConfigError{2, "--bc", k + " belongs to a dns flame file: dns conv carries no species"};
    }
    if (auto e = fvCompCommon(r, c)) return e;
    if (!comp && r.given.contains("--compensated"))
        return ConfigError{2, "--compensated", "is the compressible state's low word"};
    // FV plain FP32 drops the updates of its 2x smaller dt: Nu -2% at 192 x 96, Ra 5e4
    // (compensated == FP64 to 3e-4; docs/fvcomp.md "Gravity"). Same cost.
    if (comp && c.fvPrecision == FvPrecision::Fp32) c.compensated = true;
    if (!finitePos(r.tol)) return ConfigError{2, "--tol", "must be finite and > 0"};
    if (r.maxIters < 1) return ConfigError{2, "--max-iters", "must be >= 1"};
    if (r.given.contains("--mesh-perturb-kind") && !r.given.contains("--mesh-perturb"))
        return ConfigError{2, "--mesh-perturb-kind", "needs --mesh-perturb A"};
    if (r.given.contains("--mesh-perturb")) {
        const bool sine = r.meshPerturbKind == 0;
        if (!(std::isfinite(r.meshPerturb) && r.meshPerturb > 0 && r.meshPerturb <= (sine ? 0.12 : 0.45)))
            return ConfigError{2, "--mesh-perturb", sine ? "sine: must be in (0, 0.12]" : "random: must be in (0, 0.45]"};
        c.fvPerturb = r.meshPerturbKind + 1;
        c.fvPerturbAmp = r.meshPerturb;
    }
    if (!comp) {
        if (!(std::isfinite(r.les) && r.les >= 0)) return ConfigError{2, "--les", "must be finite and >= 0"};
        if (r.given.contains("--sgs") && r.les == 0) return ConfigError{2, "--sgs", "selects the SGS model for --les"};
        if (r.les > 0 && r.sgs == SgsModel::Sigma && !poly && c.grid.nz == 1)
            return ConfigError{2, "--sgs", "sigma: D_sigma is identically ZERO for any 2D state"};
        if (auto e = fvIncompModules(r, true); e) return e;
        c.cs = r.les;
        c.sgsModel = r.sgs;
        c.lesLabel = r.les > 0;
        if (c.lesLabel) c.meta.sgs = std::string(name(c.sgsModel));
    }
    c.fvConv = true;
    c.fvConvBins = r.ny;
    c.fvTol = r.given.contains("--tol") ? r.tol : 1e-6;
    c.fvMaxPcg = r.given.contains("--max-iters") ? r.maxIters : 200;
    if (poly) {
        c.fvMeshDir = r.polymesh;
        c.fvBcFile = r.bc;
        if (comp) c.fvBc = *r.fvBc;
    }
    return std::nullopt;
}

} // namespace dns::cfg
