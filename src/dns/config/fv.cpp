// validate(): the FaceMesh routes of dns flow (--polymesh, --solver fv|simple, --bc).
#include "dns/config.hpp"
#include "dns/config/internal.hpp"

#include "dns/fvmodules.hpp"
#include "dns/ogrid.hpp"

#include "dns/remap.hpp"
#include "dns/steady_stop.hpp"

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>

using std::numbers::pi;

namespace dns::cfg {

std::expected<Case, ConfigError> validateFvFlow(const RawConfig& r, Case c)
{
    static const char* const allowed[] = {"--polymesh",     "--physics",          "--out",
                                          "--max-iters",    "--emit",             "--fv-precision",
                                          "--backend",      "--quiet",            "--dry-run",
                                          "--fv-coupled",   "--fv-coupled-eta",   "--fv-coupled-alpha",
                                          "--fv-coupled-check"};
    for (const auto& g : r.given)
        if (std::ranges::none_of(allowed, [&](const char* a) { return g == a; }))
            return bad(2, "--polymesh",
                       g + " is not a flag of the FV path: the OpenFOAM case dictionaries (0/, system/, "
                           "constant/) define the mesh, physics and numerics (docs/fv.md)");
    if (r.polymesh.empty()) return bad(2, "--polymesh", "needs an OpenFOAM case directory");
    if (c.outPrefix.empty()) return bad(2, "--out", "output prefix must not be empty");
    for (const Emit e : c.emit)
        if (e != Emit::Stats && e != Emit::None)
            return bad(2, "--emit", "the FV path writes OpenFOAM time directories and stats only "
                                    "(--emit stats | none)");
    if (r.given.contains("--max-iters") && r.maxIters < 1) return bad(2, "--max-iters", "must be >= 1");
    c.fvCase = r.polymesh;
    c.fvMaxIters = r.given.contains("--max-iters") ? r.maxIters : 0;
    c.fvPrecision = FvPrecision(r.fvPrecision);
    // the EXPERIMENTAL coupled prototype (plan_perf_5 P0b)
    const bool cpl = r.given.contains("--fv-coupled");
    for (const char* f : {"--fv-coupled-eta", "--fv-coupled-alpha", "--fv-coupled-check"})
        if (r.given.contains(f) && !cpl) return bad(2, f, "tunes the coupled prototype: only with --fv-coupled");
    if (cpl) {
        if (r.fvCoupled < 0) return bad(2, "--fv-coupled", "SIMPLE iterations before the switch: must be >= 0");
        if (c.fvPrecision != FvPrecision::Fp64)
            return bad(2, "--fv-coupled", "the P0b prototype is FP64 only: add --fv-precision fp64");
        if (!(r.fvCoupledEta > 0 && r.fvCoupledEta < 1)) return bad(2, "--fv-coupled-eta", "must be in (0, 1)");
        if (!(r.fvCoupledAlpha > 0 && r.fvCoupledAlpha <= 1))
            return bad(2, "--fv-coupled-alpha", "must be in (0, 1]");
        c.fvCoupled = r.fvCoupled;
        c.fvCoupledEta = r.fvCoupledEta;
        c.fvCoupledAlpha = r.fvCoupledAlpha;
        c.fvCoupledCheck = r.fvCoupledCheck;
    }
    return c;
}

} // namespace dns::cfg

namespace dns::cfg {
namespace {

// "a,b,c,d" of --frame-box.
std::expected<std::array<double, 4>, ConfigError> frameBoxOf(const RawConfig& r)
{
    std::array<double, 4> b{};
    if (!r.given.contains("--frame-box")) return b;
    std::size_t at = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const std::size_t e = r.frameBox.find(',', at);
        if ((e == std::string::npos) != (i == 3)) return bad(2, "--frame-box", "needs lo0,hi0,lo1,hi1");
        const std::string_view tok = std::string_view(r.frameBox).substr(at, e == std::string::npos ? e : e - at);
        const auto [ptr, ec] = std::from_chars(tok.data(), tok.data() + tok.size(), b[i]);
        if (ec != std::errc{} || ptr != tok.data() + tok.size() || !std::isfinite(b[i]))
            return bad(2, "--frame-box", "needs four finite numbers lo0,hi0,lo1,hi1");
        at = e + 1;
    }
    if (!(b[1] > b[0]) || !(b[3] > b[2])) return bad(2, "--frame-box", "needs lo < hi on both axes");
    return b;
}

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

std::optional<ConfigError> fvCompCommon(const RawConfig& r, Case& c)
{
    for (const Emit e : c.emit)
        if (e != Emit::Stats && e != Emit::None && e != Emit::Bin && e != Emit::Png && e != Emit::Web && e != Emit::Shm)
            return ConfigError{2, "--emit",
                               "the compressible FV path writes stats, bin, png, web and shm (a uniform plane "
                               "through the mesh); volume, slices, vtk, surface, mesh, spectrum, snapshot and "
                               "error are not built"};
    auto box = frameBoxOf(r);
    if (!box) return box.error();
    if (r.frameN < 16) return ConfigError{2, "--frame-n", "must be >= 16"};
    c.fvFrameBox = *box;
    c.fvFrameN = r.frameN;
    c.fvPrecision = FvPrecision(r.fvPrecision);
    // The inventory's pairs a flag pair spells (dns/fvmodules, docs/fvcomp.md "Combinations").
    if (r.compensated && c.fvPrecision == FvPrecision::Fp64)
        return ConfigError{2, "--compensated", std::string(fvPair(FvModule::Compensated, FvModule::Fp64).message)};
    if (r.les > 0.0 && r.given.contains("--steady"))
        return ConfigError{2, "--steady", std::string(fvPair(FvModule::LesSmag, FvModule::Steady).message)};
    c.compensated = r.compensated;
    return std::nullopt;
}

} // namespace dns::cfg

namespace dns::cfg {
namespace {

// --rans on the polyMesh route (flowRans owns the structured-mesh routes): the
// closure's own flags, rejected by name.
std::optional<ConfigError> fvRansCheck(const RawConfig& r)
{
    const RansModel m = RansModel(r.rans);
    const bool sa = m == RansModel::SaNeg, sst = ransSst(m);
    if (m != RansModel::None && (r.les > 0.0 || r.given.contains("--sgs")))
        return ConfigError{2, "--rans",
                           std::string(fvPair(FvModule::LesSmag, FvModule::RansSa).message) + "; drop --les/--sgs"};
    if (r.given.contains("--rans-nutinf") && !sa)
        return ConfigError{2, "--rans-nutinf", "is the SA freestream nu~/nu; it needs --rans sa-neg"};
    for (const char* f : {"--rans-kinf", "--rans-mutinf"})
        if (r.given.contains(f) && !sst)
            return ConfigError{2, f, "is an SST freestream value; it needs --rans sst-2003m|sst-vm"};
    if (sa && !finitePos(r.ransNutInf)) return ConfigError{2, "--rans-nutinf", "must be finite and > 0"};
    if (sst && !finitePos(r.ransKInf)) return ConfigError{2, "--rans-kinf", "must be finite and > 0"};
    if (sst && !finitePos(r.ransMutInf)) return ConfigError{2, "--rans-mutinf", "must be finite and > 0"};
    return std::nullopt;
}

// Pre: fvRansCheck passed; c.nu, c.uInf set.
void fvRansResolve(const RawConfig& r, Case& c)
{
    c.rans = RansModel(r.rans);
    c.ransNutInf = r.ransNutInf;
    c.ransKInf = r.ransKInf;
    c.ransMutInf = r.ransMutInf;
    if (c.rans == RansModel::SaNeg) c.ransPhiInf[0] = nuTildaInf(c.ransNutInf, c.nu);
    if (ransSst(c.rans)) {
        c.ransPhiInf[0] = kInf(c.ransKInf, c.uInf);
        c.ransPhiInf[1] = omegaInf(c.ransPhiInf[0], c.ransMutInf, c.nu);
    }
}

} // namespace
} // namespace dns::cfg

namespace dns::cfg {

namespace {

// The nozzle route's flags and --bc tables on `dns flow --physics compressible --polymesh`
// (docs/nozzle.md), by name. Pre: r.fvBc set.
std::optional<ConfigError> fvNozzleScope(const RawConfig& r)
{
    const bool hll = r.shock == ShockMode::Hll;
    const FvBcSpec& bc = *r.fvBc;
    const bool hasTotal = std::ranges::any_of(bc.patches, [](const FvPatchSpec& p) { return p.kind == FvPatchKind::Total; });
    if (r.given.contains("--shock-d4") && !hll) return ConfigError{2, "--shock-d4", "needs --shock hll"};
    if (r.given.contains("--shock-d4") && !(r.shockD4 >= 0.0 && r.shockD4 <= kFvShockEps4Max))
        return ConfigError{2, "--shock-d4", "must be in [0, 0.125]"};
    if (r.inviscid) {
        if (!hll) return ConfigError{2, "--inviscid", "needs --shock hll"};
        if (r.given.contains("--re")) return ConfigError{2, "--inviscid", "--re is the viscous number: give one of them"};
        for (const char* f : {"--visc", "--prandtl", "--tinf"})
            if (r.given.contains(f)) return ConfigError{2, "--inviscid", "no viscous flag with it (--visc, --prandtl, --tinf)"};
        for (const FvPatchSpec& p : bc.patches)
            if (p.kind == FvPatchKind::Wall)
                return ConfigError{2, "--bc", std::format("patch `{}`: --inviscid: a no-slip wall needs viscosity, use type = \"slip\"", p.name)};
    }
    if (bc.nozzle && !hll) return ConfigError{2, "--bc", "[nozzle] needs --shock hll"};
    if (hasTotal && !bc.nozzle) return ConfigError{2, "--bc", "a total patch needs [nozzle] (the rest state)"};
    if (r.given.contains("--foam-write") && !(r.foamWrite > 0.0 && std::isfinite(r.foamWrite)))
        return ConfigError{2, "--foam-write", "must be finite and > 0"};
    if (r.given.contains("--gamma") && !(r.gamma > 1.0 && r.gamma <= 1.67)) return ConfigError{2, "--gamma", "must be in (1, 1.67]"};
    if (hll && r.given.contains("--cfl") && r.cfl > kShockCflAxisSum + 1e-12)
        return ConfigError{2, "--cfl", "must be <= 1.2 on the FV path: the Sigma form is already the axis sum (docs/shock.md)"};
    return std::nullopt;
}

} // namespace

std::expected<Case, ConfigError> validateFvCompMesh(const RawConfig& r, Case c)
{
    static const char* const allowed[] = {"--polymesh", "--bc",      "--physics",    "--re",        "--mach",
                                          "--tt",       "--ramp",    "--cfl",        "--les",       "--sgs",
                                          "--visc",     "--tinf",    "--prandtl",    "--fv-precision", "--out",
                                          "--emit",     "--fdt",     "--pdt",        "--wdt",       "--png-down",
                                          "--vort-max", "--vort-map", "--speed-map", "--dt-recompute", "--frame-box",
                                          "--frame-n",  "--backend", "--quiet",      "--dry-run",   "--compensated",
                                          "--solver",   "--rans",    "--rans-nutinf", "--rans-kinf", "--rans-mutinf",
                                          "--steady",   "--steady-steps", "--shock", "--shock-d4", "--inviscid",
                                          "--foam-write", "--gamma"};
    for (const auto& g : r.given)
        if (std::ranges::none_of(allowed, [&](const char* a) { return g == a; }))
            return bad(2, "--polymesh", g + " is not a flag of the compressible FV path on a polyMesh (docs/fvcomp.md)");
    if (!r.fvBc) return bad(2, "--bc", "--physics compressible --polymesh needs --bc FILE.toml");
    if (const std::string k = fvBcFlameKey(*r.fvBc); !k.empty())
        return bad(2, "--bc", k + " belongs to a dns flame file: dns flow carries no species");
    if (c.outPrefix.empty()) return bad(2, "--out", "output prefix must not be empty");
    if (c.outPrefix.find_first_of("\"\\\n\r\t") != std::string::npos)
        return bad(2, "--out", "output prefix must not contain quotes, backslashes or control characters");
    const bool hll = r.shock == ShockMode::Hll;
    if (auto e = fvNozzleScope(r)) return std::unexpected(*e);
    if (!r.inviscid && (!r.given.contains("--re") || !finitePos(r.re))) return bad(2, "--re", "required, finite and > 0");
    if (hll && (!r.given.contains("--mach") || !finitePos(r.mach))) return bad(2, "--mach", "must be finite and > 0");
    if (!hll && (!r.given.contains("--mach") || !finitePos(r.mach) || r.mach >= 1.0))
        return bad(2, "--mach", "required, in (0, 1): the KG flux has no shock capturing");
    const bool steady = r.given.contains("--steady");
    if (steady) {
        for (const char* f : {"--tt", "--ramp", "--fdt", "--pdt", "--wdt"})
            if (r.given.contains(f)) return bad(2, f, "with --steady: pseudo-time is not time; --steady-steps is the cap");
        if (r.given.contains("--emit") && !c.emit.contains(Emit::Stats) && !c.emit.contains(Emit::None))
            return bad(2, "--emit", "frames with --steady: pseudo-time is not time; frames are time-keyed");
        if (!(r.steady >= kSteadyTolMin && r.steady < 1))
            return bad(2, "--steady", std::format("is a relative change of the integrals: must be in [{:g}, 1)",
                                                  kSteadyTolMin));
        if (r.given.contains("--steady-steps") && r.steadySteps < 1) return bad(2, "--steady-steps", "must be >= 1");
    } else if (r.given.contains("--steady-steps")) {
        return bad(2, "--steady-steps", "requires --steady TOL");
    }
    if (!steady && (!r.given.contains("--tt") || !finitePos(r.ttFv)))
        return bad(2, "--tt", "required, finite and > 0 (L/U)");
    if (r.given.contains("--ramp") && !(r.ramp >= 0.0)) return bad(2, "--ramp", "must be >= 0");
    if (!(r.prandtl > 0.0)) return bad(2, "--prandtl", "must be > 0");
    if (r.given.contains("--sgs") && !(r.les > 0.0)) return bad(2, "--sgs", "selects the --les model: needs --les");
    if (!(r.les >= 0.0)) return bad(2, "--les", "must be >= 0");
    if (auto e = fvRansCheck(r)) return std::unexpected(*e);
    c.fvMode = FvCompMode::PolyMesh;
    c.fvMeshDir = r.polymesh;
    c.fvBc = *r.fvBc;
    c.re = r.inviscid ? 0.0 : r.re;
    c.inviscid = r.inviscid;
    c.foamWrite = r.foamWrite;
    c.fvGamma = r.gamma;
    c.shockEps4 = hll ? r.shockD4 : 0.0;
    if (r.shock != ShockMode::None) c.meta.shock = name(r.shock);
    if (hll) shockCadence(r, c);
    c.mach = r.mach;
    c.uInf = 1.0;
    c.tEnd = r.ttFv * r.fvBc->length / c.uInf;
    c.ramp = r.given.contains("--ramp") ? r.ramp : 0.0;
    if (steady) {
        c.steadyTol = r.steady;
        c.compSteady = true;
        c.steadySteps = r.steadySteps;
    }
    c.cfl = r.given.contains("--cfl") ? r.cfl : 0.4;
    c.dtRecompute = r.given.contains("--dt-recompute") ? r.dtRecompute : 16;
    c.cs = r.les;
    c.sgsModel = r.sgs;
    c.lesLabel = c.cs > 0;
    c.sutherland = r.visc == 1;
    c.tinfK = r.tinf;
    c.prandtl = r.prandtl;
    c.fdt = r.fdt;
    c.pdt = r.pdt > 0 ? r.pdt : c.fdt;
    c.wdt = r.wdt > 0 ? r.wdt : c.fdt;
    c.pngDown = r.pngDown;
    c.vortMax = r.vortMax;
    c.vortAsinh = r.vortMap == 1;
    c.speedDelta = r.speedMap == 1;
    c.nu = r.inviscid ? 0.0 : c.uInf * r.fvBc->length / c.re;
    fvRansResolve(r, c);
    c.meta = {.D = r.fvBc->length, .re = c.re, .mach = c.mach, .frameDt = c.fdt, .cs = c.cs, .les = c.lesLabel,
              .body = "polymesh", .farField = "fv", .physics = "compressible", .kick = 0,
              .sgs = c.lesLabel ? std::string(name(c.sgsModel)) : std::string(),
              .rans = c.rans == RansModel::None ? std::string() : std::string(ransName(c.rans)), .steady = {}};
    if (r.shock != ShockMode::None) c.meta.shock = name(r.shock);
    if (auto e = fvCompCommon(r, c)) return std::unexpected(*e);
    return c;
}

std::optional<ConfigError> checkFvStructured(const RawConfig& r)
{
    static const char* const allowed[] = {
        "--solver",  "--body",      "--n",        "--ny",          "--nz",         "--dr",       "--re",
        "--mach",    "--physics",   "--ogrid",    "--ogrid-far",   "--ogrid-beta", "--ogrid-trip", "--aspect",
        "--stretch", "--focal",     "--focal-ratio", "--focal-blend", "--focal-wake", "--focal-pad", "--far",
        "--noslip",  "--wall",      "--twall",    "--visc",        "--tinf",       "--prandtl",  "--les",
        "--sgs",     "--cfl",       "--ft",       "--ramp",        "--kick",       "--dt-recompute", "--out",
        "--emit",    "--fdt",       "--pdt",      "--wdt",         "--png-down",   "--vort-max", "--vort-map",
        "--speed-map", "--quiet",   "--dry-run",  "--backend",     "--fv-precision", "--frame-box", "--frame-n",
        "--compensated", "--rans",  "--rans-nutinf", "--rans-kinf", "--rans-mutinf", "--steady", "--steady-steps"};
    for (const auto& g : r.given)
        if (std::ranges::none_of(allowed, [&](const char* a) { return g == a; }))
            return ConfigError{2, "--solver", g + " is not built on the FV path (--solver fv, docs/fvcomp.md)"};
    if (!r.scene.empty()) return ConfigError{2, "--scene", "a scene is immersed: FV has no immersed wall"};
    const bool none = r.body == "none";
    if (!r.ogrid && !none)
        return ConfigError{2, "--solver", "fv has no immersed wall: a body needs --ogrid (body-fitted) or a polyMesh "
                                          "(--polymesh DIR --bc FILE); --body none runs the empty box"};
    if (r.ogrid && r.body == "sphere")
        return ConfigError{2, "--ogrid", "the revolved O-grid (wedge, axis faces, full revolution) is not on FaceMesh "
                                         "(masterplan P8); --solver fv takes z-extruded O-grids"};
    return std::nullopt;
}

std::expected<Case, ConfigError> validateFvIncompFlow(const RawConfig& r, Case c)
{
    const bool poly = r.given.contains("--polymesh");
    static const char* const common[] = {"--solver", "--physics", "--re",           "--ft",     "--cfl",
                                         "--les",    "--sgs",     "--tol",          "--out",    "--emit",
                                         "--backend", "--quiet",  "--dry-run",      "--max-iters",
                                         "--fv-precision", "--seed"};
    static const char* const polyOnly[] = {"--polymesh", "--bc"};
    static const char* const ogridOnly[] = {"--ogrid", "--ogrid-far", "--ogrid-beta", "--ogrid-trip", "--body",
                                            "--n",     "--ny",        "--nz"};
    const auto listed = [](std::span<const char* const> l, const std::string& g) {
        return std::ranges::any_of(l, [&](const char* a) { return g == a; });
    };
    // The inventory's refusals first (dns/fvimodules): --rans is a module, refused by name.
    if (auto e = fvIncompModules(r, false); e) return std::unexpected(*e);
    for (const auto& g : r.given)
        if (!listed(common, g) && !(poly ? listed(polyOnly, g) : listed(ogridOnly, g)))
            return bad(2, "--solver", g + " is not a flag of the time-accurate FV solver (docs/fvincomp.md)");
    if (poly && !r.given.contains("--bc"))
        return bad(2, "--bc", "--polymesh DIR with --solver fv needs --bc FILE.toml (--polymesh CASE alone is "
                              "simpleFoam)");
    if (!poly && r.given.contains("--bc")) return bad(2, "--bc", "binds the patches of --polymesh DIR");
    if (!poly && !r.ogrid)
        return bad(2, "--solver", "--solver fv has no immersed wall: use --ogrid (a body-fitted O-grid) or "
                                  "--polymesh DIR --bc FILE.toml");
    if (!finitePos(r.tol)) return bad(2, "--tol", "must be finite and > 0");
    if (r.given.contains("--max-iters") && r.maxIters < 1) return bad(2, "--max-iters", "must be >= 1");
    if (r.les < 0 || !std::isfinite(r.les)) return bad(2, "--les", "must be finite and >= 0");
    if (r.given.contains("--sgs") && r.les == 0) return bad(2, "--sgs", "selects the SGS model for --les");
    if (poly) {
        if (r.polymesh.empty()) return bad(2, "--polymesh", "needs a mesh directory");
        if (r.bc.empty()) return bad(2, "--bc", "needs a file");
        if (c.outPrefix.empty()) return bad(2, "--out", "output prefix must not be empty");
        if (!finitePos(r.re)) return bad(2, "--re", "must be finite and > 0");
        if (!finitePos(r.ft)) return bad(2, "--ft", "must be finite and > 0");
        c.re = r.re;
        c.ft = r.ft;
        c.fvMeshDir = r.polymesh;
        c.fvBcFile = r.bc;
        c.cfl = r.given.contains("--cfl") ? r.cfl : 0.8;
        if (!finitePos(c.cfl)) return bad(2, "--cfl", "must be finite and > 0");
    } else {
        if (r.n < 1) return bad(2, "--n", "must be >= 1");
        if (!finitePos(r.re)) return bad(2, "--re", "must be finite and > 0");
        c.re = r.re;
        c.n = r.n;
        auto f = validateFlow(r, std::move(c));
        if (!f) return f;
        c = std::move(*f);
        if (c.ogridRevolved)
            return bad(2, "--ogrid", "--solver fv: a revolved O-grid is an axis mesh (no FaceMesh yet)");
        c.cfl = r.given.contains("--cfl") ? r.cfl : 0.8;
    }
    if (r.given.contains("--emit")) {
        for (const Emit e : r.emit)
            if (e != Emit::Stats && e != Emit::None)
                return bad(2, "--emit", "the FV solver writes stats (--emit stats | none)");
        c.emit = r.emit;
    } else
        c.emit = {Emit::Stats};
    if (r.les > 0 && r.sgs == SgsModel::Sigma && !poly && c.grid.nz == 1)
        return bad(2, "--sgs", "sigma: D_sigma is identically ZERO for any 2D state");
    c.fvIncomp = true;
    c.fvPrecision = FvPrecision(r.fvPrecision);
    c.fvTol = r.given.contains("--tol") ? r.tol : 1e-5;
    c.fvMaxPcg = r.given.contains("--max-iters") ? r.maxIters : 200;
    c.cs = r.les;
    c.sgsModel = r.sgs;
    return c;
}

std::expected<Case, ConfigError> validateFvStructured(const RawConfig& r0, Case c)
{
    static const char* const allowed[] = {
        "--solver",       "--export",       "--out",          "--emit",        "--backend",     "--quiet",
        "--dry-run",      "--fv-precision", "--physics",      "--max-iters",   "--steady",      "--re",
        "--re-tau",       "--n",            "--ny",           "--nz",          "--dr",          "--body",
        "--aspect",       "--far",          "--open",         "--noslip",      "--stretch",     "--focal",
        "--focal-ratio",  "--focal-blend",  "--focal-wake",   "--focal-pad",   "--ogrid",       "--ogrid-far",
        "--ogrid-beta",   "--ogrid-wedge",  "--rans",         "--rans-nutinf", "--rans-kinf",   "--rans-mutinf",
        "--rotor-mach",   "--rotor-frame",  "--rotor-axis",   "--mach"};
    for (const auto& g : r0.given)
        if (std::ranges::none_of(allowed, [&](const char* a) { return g == a; }))
            return bad(2, g.c_str(), "has no simpleFoam meaning: --solver simple solves the steady incompressible case "
                             "the mesh flags name (docs/fvstructured.md)");
    if (r0.physics != Physics::Incompressible)
        return bad(2, "--physics", "--solver simple is incompressible (simpleFoam)");
    const bool exporting = r0.given.contains("--export");
    if (exporting) {
        if (r0.exportCase.empty()) return bad(2, "--export", "needs a directory");
        for (const char* f : {"--out", "--emit", "--fv-precision", "--backend"})
            if (r0.given.contains(f))
                return bad(2, f, "belongs to a run; --export only writes the case (dns flow --polymesh CASE runs it)");
    }
    const RansModel model = RansModel(r0.rans);
    if (model == RansModel::SstVm)
        return bad(2, "--rans", "sst-vm: OpenFOAM's kOmegaSST has no vorticity-production variant (sa-neg | "
                                "sst-2003m, docs/fvstructured.md)");
    if (model == RansModel::None)
        for (const char* f : {"--rans-nutinf", "--rans-kinf", "--rans-mutinf"})
            if (r0.given.contains(f)) return bad(2, f, "sets the RANS freestream: give --rans");
    if (r0.rotorMach == 0 && r0.given.contains("--mach"))
        return bad(2, "--mach", "has no simpleFoam meaning alone; with --rotor-mach the ratio is tip speed / U");
    if (r0.given.contains("--rotor-mach") && !r0.ogrid)
        return bad(2, "--rotor-mach", "--solver simple spins a body-fitted wall: needs --ogrid");
    if (r0.given.contains("--far") && r0.far != FarField::Tunnel && r0.far != FarField::Nscbc &&
        r0.far != FarField::Periodic)
        return bad(2, "--far", std::string(name(r0.far)) + " has no simpleFoam meaning (tunnel | nscbc: inlet and "
                                                           "outlet; periodic: the --re-tau channel)");
    const bool periodic = r0.given.contains("--far") && r0.far == FarField::Periodic;
    const bool channel = r0.given.contains("--re-tau");
    if (channel != periodic)
        return bad(2, channel ? "--re-tau" : "--far",
                   "the plane channel is --re-tau R --far periodic --noslip --body none: a periodic box has no "
                   "driver without the channel's body force");
    if (channel && !r0.noslip) return bad(2, "--noslip", "the plane channel has no-slip walls: give --noslip");
    if (channel)
        for (const char* f : {"--re", "--dr", "--open"})
            if (r0.given.contains(f)) return bad(2, f, "is not a channel flag: --re-tau sets nu, delta = 1");

    // The stages read the compressible box conventions; --far is settled above.
    RawConfig r = r0;
    r.physics = Physics::Compressible;
    c.physics = Physics::Compressible;
    if (!r.ogrid) r.given.erase("--far");
    r.far = FarField::Nscbc;
    c.far = FarField::Nscbc;
    if (auto e = flowBody(r, c); e) return std::unexpected(*e);
    if (c.bodyKind != BodyKind::None && !r.ogrid)
        return bad(2, "--body", "an immersed body has no body-fitted faces on a Cartesian mesh: use --ogrid, or "
                                "--body none (a staircase is not exported)");

    FvStructured s;
    s.active = true;
    s.uInf = 1;
    if (channel) {
        if (!finitePos(r.reTau)) return bad(2, "--re-tau", "must be finite and > 0");
        if (r.n < 2) return bad(2, "--n", "the channel's cyclic x needs >= 2 cells");
        const int ny = r.given.contains("--ny") ? r.ny : r.n;
        if (ny < 8) return bad(2, "--ny", "the channel's fluid rows: >= 8");
        if (r.nz < 1 || r.nz == 2) return bad(2, "--nz", "1 (2D) or >= 3");
        if (!(std::isfinite(r.stretch) && r.stretch >= 0 && r.stretch <= 4))
            return bad(2, "--stretch", "must be in [0,4] (tanh clustering strength)");
        c.stretchBeta = r.stretch;
        const double lx = 2 * pi, lz = r.nz == 1 ? 1.0 : pi;
        // channelY's grid (ghost rows included), or the bare box under --focal.
        const bool focal = r.given.contains("--focal");
        if (focal && r.stretch > 0)
            return bad(2, "--focal", "--stretch and --focal are different curvilinear meshes; pick one");
        const int nyT = focal ? ny : ny + 2;
        const double h1 = ChannelStretchY::firstCell(ny + 2, c.stretchBeta, 1.0);
        auto box = makeBox(r.n, nyT, r.nz, lx, focal ? 2.0 : 2.0 + 2 * h1, lz);
        if (!box) return std::unexpected(box.error());
        c.grid = *box;
        c.nzInterior = r.nz;
        c.h = c.grid.hy;
        c.L = lx;
        c.placement = {0.0, 0.0, 0.0, 1.0};
        c.re = r.reTau;
        c.reTau = r.reTau;
        c.nu = 1.0 / r.reTau;
        if (focal) {
            if (auto e = flowFocal(r, c); e) return std::unexpected(*e);
        } else if (auto e = rejectGiven(r, {"--focal-ratio", "--focal-blend", "--focal-wake", "--focal-pad"},
                                        "requires --focal");
                   e)
            return std::unexpected(*e);
        s.mesh = FvStructMesh::Channel;
        s.reTau = r.reTau;
    } else {
        if (auto e = flowGrid(r, c); e) return std::unexpected(*e);
        if (auto e = flowPhysics(r, c); e) return std::unexpected(*e);
        if (auto e = flowStretch(r, c); e) return std::unexpected(*e);
        if (auto e = flowFocal(r, c); e) return std::unexpected(*e);
        if (auto e = flowRotor(r, c); e) return std::unexpected(*e);
        s.mesh = r.ogrid ? FvStructMesh::OGrid : FvStructMesh::Box;
        s.lateral = r.noslip ? FvStructLateral::Wall : r.open ? FvStructLateral::Open : FvStructLateral::Symmetry;
        if (r.noslip && r.open) return bad(2, "--open", "--noslip walls are not open faces");
    }
    if (r.ogrid && r.noslip) return bad(2, "--noslip", "is the box's y walls; the O-grid has none");
    s.grid = c.grid;
    s.nu = c.nu;
    s.stretchBeta = c.stretchBeta;
    s.focal = c.focalActive;
    s.focalRatio = c.focalRatio;
    for (std::size_t a = 0; a < 3; ++a) {
        s.focalLo[a] = c.focalLo[a];
        s.focalHi[a] = c.focalHi[a];
        s.focalW[a] = c.focalW[a];
    }
    if (c.ogrid) {
        const double D = c.placement.D;
        s.ogrid = {.cx = c.placement.cx, .cy = c.placement.cy, .cz = c.placement.cz, .rFar = c.ogridRFar,
                   .beta = c.ogridBeta, .span = c.ogridSpan};
        s.revolved = c.ogridRevolved;
        if (s.revolved) {
            s.rev = c.ogridNphi > 0 ? RevolvedSpec{.wedge = false, .dphi = 2.0 * pi / c.ogridNphi, .nphi = c.ogridNphi}
                                    : RevolvedSpec{.wedge = true, .dphi = c.ogridWedgeDeg * pi / 180.0};
            s.refArea = 0.25 * pi * D * D * (s.rev.wedge ? s.rev.dphi / (2.0 * pi) : 1.0);
        } else {
            s.refArea = D * c.ogridSpan;
        }
        if (c.rotor.active()) {
            if (s.revolved)
                return bad(2, "--rotor-mach", "the revolved O-grid's axis is x, the MRF zone's shaft z: the frame "
                                              "is not built there");
            if (c.grid.nz > 1)
                return bad(2, "--rotor-mach", "--nz > 1: every zone cell touches the cyclic z faces, which OpenFOAM's "
                                              "MRF treats per side (rejected by the FV path); run --nz 1");
            s.omega = c.rotor.omega;
            s.mrfRadius = 2.0 * 0.5 * D; // inside the far circle: rFar > D
            if (s.mrfRadius >= 0.5 * c.ogridRFar)
                return bad(2, "--ogrid-far", "the MRF zone (r < D) must stay inside half the far radius");
        }
    }
    s.model = model == RansModel::SaNeg ? FvTurbModel::SpalartAllmaras
              : model == RansModel::Sst2003m ? FvTurbModel::KOmegaSST
                                             : FvTurbModel::Laminar;
    if (!finitePos(r.ransNutInf) || !finitePos(r.ransKInf) || !finitePos(r.ransMutInf))
        return bad(2, "--rans-nutinf", "the RANS freestream values must be finite and > 0");
    s.nutRatio = r.ransNutInf;
    s.kInf = r.ransKInf;
    s.mutRatio = r.ransMutInf;
    if (r.given.contains("--max-iters") && r.maxIters < 1) return bad(2, "--max-iters", "must be >= 1");
    s.iters = r.given.contains("--max-iters") ? r.maxIters : 2000;
    if (r.given.contains("--steady")) {
        if (!finitePos(r.steady)) return bad(2, "--steady", "the residualControl of every field: must be > 0");
        if (channel)
            return bad(2, "--steady", "the channel's Uy is rounding normalised by rounding (residual ~0.2 forever): "
                                      "residualControl never fires; give --max-iters");
        s.residual = r.steady;
        if (r.steady < 2e-6 && FvPrecision(r.fvPrecision) == FvPrecision::Fp32)
            c.warnings.push_back(std::format("--steady {:g} sits on the FP32 residual noise (U 7.9e-7, p 6.8e-7): it "
                                             "may never certify; give --fv-precision fp64",
                                             r.steady));
    }

    if (exporting) {
        c.fvExport = r.exportCase;
        c.fvCase = r.exportCase;
    } else {
        if (c.outPrefix.empty()) return bad(2, "--out", "output prefix must not be empty");
        for (const Emit e : c.emit)
            if (e != Emit::Stats && e != Emit::None)
                return bad(2, "--emit", "the FV path writes OpenFOAM time directories and stats only "
                                        "(--emit stats | none)");
        c.fvCase = c.outPrefix;
    }
    c.fvPrecision = FvPrecision(r.fvPrecision);
    c.fvs = s;
    c.physics = Physics::Incompressible;
    return c;
}

} // namespace dns::cfg
