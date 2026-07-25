#include "dns/config.hpp"

#include "dns/remap.hpp"
#include "dns/scene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string_view>

namespace dns {
namespace {

using std::numbers::pi;

bool finitePos(double v) { return std::isfinite(v) && v > 0; }

bool hasSuffix(std::string_view s, std::string_view suf)
{
    return s.size() > suf.size() && s.substr(s.size() - suf.size()) == suf;
}

const char* name(Physics p)
{
    return p == Physics::Compressible ? "compressible" : "incompressible";
}

const char* name(SgsModel m)
{
    return m == SgsModel::Sigma ? "sigma" : "smag";
}

const char* name(FarField f)
{
    switch (f)
    {
    case FarField::Tunnel: return "tunnel";
    case FarField::Fringe: return "fringe";
    case FarField::Sponge: return "sponge";
    case FarField::Nscbc: return "nscbc";
    case FarField::Periodic: return "periodic";
    }
    return "?";
}

const char* name(BodyKind b)
{
    switch (b)
    {
    case BodyKind::None: return "none";
    case BodyKind::Cylinder: return "cylinder";
    case BodyKind::Sphere: return "sphere";
    case BodyKind::Cube: return "cube";
    case BodyKind::Lens: return "lens";
    case BodyKind::Fence: return "fence";
    case BodyKind::Floor: return "floor";
    case BodyKind::Mesh: return "mesh";
    case BodyKind::Scene: return "scene";
    }
    return "?";
}

std::unexpected<ConfigError> bad(int code, const char* field, std::string msg)
{
    return std::unexpected(ConfigError{code, field, std::move(msg)});
}

} // namespace

std::string format(const ConfigError& e)
{
    std::string s = "dns: ";
    if (!e.field.empty()) s += e.field + ": ";
    return s + e.message + "\n";
}

// ---------------------------------------------------------------------------

std::expected<void, ConfigError> applySceneModel(const Scene& s, RawConfig& r)
{
    const bool userLes = r.given.contains("--les");
    if (s.model == SceneModel::Les)
    {
        if (userLes && r.les != s.cs)
            return std::unexpected(ConfigError{
                2, "--les",
                "the scene selects model les with cs=" + std::to_string(s.cs)
                    + "; a conflicting --les was given (drop it or change the "
                      "scene)"});
        if (r.given.contains("--sgs") && r.sgs != s.sgs)
            return std::unexpected(ConfigError{
                2, "--sgs",
                std::string("the scene selects sgs = ")
                    + (s.sgs == SgsModel::Sigma ? "sigma" : "smag")
                    + "; a conflicting --sgs was given (drop it or change the "
                      "scene)"});
        r.les = s.cs;
        r.sgs = s.sgs;
    }
    else if (userLes && r.les > 0)
        return std::unexpected(ConfigError{
            2, "--les", "the scene selects model dns; drop --les or set the "
                        "scene model to les"});
    return {};
}

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

    if (c.outPrefix.empty())
        return bad(2, "--out", "output prefix must not be empty");
    // webframes.cpp concatenates the prefix into a JSON manifest unescaped.
    if (c.outPrefix.find_first_of("\"\\\n\r\t") != std::string::npos)
        return bad(2, "--out", "output prefix must not contain quotes, backslashes "
                               "or control characters (it is embedded in JSON)");

    if (r.n < 1) return bad(2, "--n", "must be >= 1");
    if (!finitePos(r.re)) return bad(2, "--re", "must be finite and > 0");
    c.re = r.re;
    c.n = r.n;

    // --- bench / tgv / hit ---------------------------------------------------
    if (r.sub == Subcommand::Bench)
    {
        if (r.steps < 1) return bad(2, "--steps", "must be >= 1");
        c.steps = r.steps;
        // bench builds an n^3 box, so it needs the grid guards every other
        // branch applies and this one was missing: `--n 1300` overflowed the
        // int indexing the tree assumes, and `--n 1` produced nz == 1 with
        // hz == 2pi, violating the validator-enforced `nz == 1 => hz == 1`.
        // n >= 4 forecloses both the degenerate-z case (bench has no 2D mode:
        // the box is always n^3) and any n too small to time meaningfully.
        if (r.n < 4) return bad(2, "--n", "must be >= 4 (bench is an n^3 box)");
        const std::size_t benchCells = std::size_t(r.n) * r.n * r.n;
        if (benchCells > std::size_t(2147483647))
            return bad(2, "--n", "grid exceeds INT_MAX cells");
        c.grid = Grid::make(r.n, r.n, r.n, 2 * pi, 2 * pi, 2 * pi);
        c.nzInterior = r.n;
        c.meta = {r.n, r.n, r.n, c.grid.hx, c.grid.hy, c.grid.hz,
                  0, 0, 0, 0, 0, false, "none", "periodic", name(c.physics), 0,
                  ""};
        return c;
    }

    if (r.sub == Subcommand::Tgv || r.sub == Subcommand::Hit)
    {
        const int nz = r.given.contains("--nz") ? r.nz : r.n;
        if (nz < 1) return bad(2, "--nz", "must be >= 1");
        if (nz != 1 && nz < 4) return bad(2, "--nz", "must be 1 (native 2D) or >= 4");
        if (r.sub == Subcommand::Hit && nz < 4)
            return bad(2, "--nz", "hit needs nz >= 4 (spectral initial condition)");
        const std::size_t cells = std::size_t(r.n) * r.n * nz;
        if (cells > std::size_t(2147483647))
            return bad(2, "--n", "grid exceeds INT_MAX cells");
        const double lz = nz == 1 ? 1.0 : 2 * pi;
        c.grid = Grid::make(r.n, r.n, nz, 2 * pi, 2 * pi, lz);
        c.nzInterior = nz;
        c.cfl = r.given.contains("--cfl") ? r.cfl : 0.9;
        if (!(c.cfl > 0 && c.cfl <= 1)) return bad(2, "--cfl", "must be in (0,1]");
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
        }
        if (c.emit.contains(Emit::Error) && nz != 1)
            return bad(2, "--emit", "--emit error compares against the 2D analytic "
                                    "Taylor-Green solution and requires --nz 1");
        if (c.emit.empty()) c.emit.insert(Emit::Stats);
        c.fdt = r.fdt;
        c.meta = {r.n, r.n, nz, c.grid.hx, c.grid.hy, c.grid.hz,
                  0, r.re, 0, c.fdt, 0, false, "none", "periodic",
                  name(c.physics), 0, ""};
        return c;
    }

    // --- channel ------------------------------------------------------------
    // Wall units: delta = 1, u_tau = 1, f = 1, nu = 1/reTau. The box momentum
    // balance (f * V = 2 tau_w * A_wall with Ly = 2 delta) pins tau_w = 1, so
    // reTau is the run's friction Reynolds number by construction.
    if (r.sub == Subcommand::Channel)
    {
        if (r.given.contains("--re"))
            return bad(2, "--re", "channel is configured in wall units by "
                                  "--re-tau, not a body Reynolds number");
        if (r.given.contains("--ft"))
            return bad(2, "--ft", "channel run length is --tt (eddy turnovers "
                                  "of delta/u_tau = 1)");
        if (r.n < 4) return bad(2, "--n", "must be >= 4");
        const int nyI = r.ny;
        if (nyI < 8)
            return bad(2, "--ny", "must be >= 8 fluid cells (wall-normal "
                                  "resolution; two ghost rows are added)");
        const int nz = r.nz;
        if (nz < 1) return bad(2, "--nz", "must be >= 1 (interior cells; 1 = "
                                          "native 2D)");
        if (nz != 1 && nz < 4)
            return bad(2, "--nz", "must be 1 (native 2D) or >= 4");
        if (!finitePos(r.reTau))
            return bad(2, "--re-tau", "must be finite and > 0");
        c.reTau = r.reTau;
        c.re = r.reTau;
        c.nu = 1.0 / r.reTau;
        c.mach = r.mach;
        if (!(std::isfinite(c.mach) && c.mach > 0 && c.mach < 1))
            return bad(2, "--mach", "must be in (0,1) -- the flux is "
                                    "Kennedy-Gruber central with no shock "
                                    "capturing");
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
        const std::size_t cells = std::size_t(r.n) * nyT * nz;
        if (cells > std::size_t(2147483647))
            return bad(2, "--n", "grid exceeds INT_MAX cells");
        c.grid = Grid::make(r.n, nyT, nz, c.lxBox, ly, c.lzBox);
        c.nzInterior = nz;
        c.h = c.grid.hy;
        c.L = c.lxBox;
        c.chanH1 = h1;
        c.stretchH1 = h1;
        const ChannelStretchY R{ly, nyT, c.stretchBeta, delta};
        double prev = h1;
        for (int j = 2; j < nyT - 1; ++j)
        {
            const double dy = R.mapY((j + 1) * c.grid.hy) - R.mapY(j * c.grid.hy);
            c.stretchGrowth = std::max(c.stretchGrowth,
                                       std::max(dy / prev, prev / dy));
            prev = dy;
        }

        // Bulk-velocity estimate (log law) fixes the sound speed for the
        // requested BULK Mach: c_s = ubEst/Mach, p = rho c_s^2/gamma,
        // Twall = p (rho = R = 1). The IC and the isothermal wall share it.
        c.ubEst = 2.5 * std::log(std::max(c.reTau, 3.0)) + 2.6;
        c.pRef = (c.ubEst / c.mach) * (c.ubEst / c.mach) / 1.4;

        c.cfl = r.cfl;
        if (!(c.cfl > 0 && c.cfl <= 1)) return bad(2, "--cfl", "must be in (0,1]");
        c.kick = r.kick;
        if (!(std::isfinite(c.kick) && c.kick >= 0))
            return bad(2, "--kick", "must be finite and >= 0");
        if (!finitePos(r.tt)) return bad(2, "--tt", "must be finite and > 0");
        c.ft = r.tt;
        c.tEnd = r.tt;
        c.profileDt = r.profileDt;
        if (!(std::isfinite(c.profileDt) && c.profileDt >= 0))
            return bad(2, "--profile-dt", "must be finite and >= 0 (0 = off)");
        c.dtRecompute = r.given.contains("--dt-recompute") ? r.dtRecompute : 16;
        if (c.dtRecompute < 1) return bad(2, "--dt-recompute", "must be >= 1");
        c.fdt = r.fdt;
        c.pdt = r.pdt > 0 ? r.pdt : c.fdt;
        c.wdt = r.wdt > 0 ? r.wdt : c.fdt;
        for (auto [v, f] : {std::pair{c.fdt, "--fdt"}, {c.pdt, "--pdt"},
                            {c.wdt, "--wdt"}})
            if (!finitePos(v)) return bad(2, f, "must be finite and > 0");
        c.planes = r.planes;
        if (c.planes != 1 && c.planes != 3)
            return bad(2, "--planes", "must be 1 (centre only) or 3");
        c.pngDown = r.pngDown;
        if (c.pngDown < 1)
            return bad(2, "--png-down", "must be >= 1 (1 = full resolution)");
        if (c.pngDown > 1 && (c.grid.nx % c.pngDown || c.grid.ny % c.pngDown))
            return bad(2, "--png-down", "must divide both nx and ny evenly");
        if (c.emit.empty()) c.emit.insert(Emit::Stats);
        if (c.emit.contains(Emit::None) && c.emit.size() > 1)
            return bad(2, "--emit", "--emit none cannot be combined with "
                                    "other kinds");
        // PNG is allowed at nz > 1 here: it renders the centre z-plane,
        // which for a channel is exactly the picture you want.
        c.physics = Physics::Compressible;
        c.far = FarField::Periodic;
        c.bodyKind = BodyKind::None;
        c.meta = {c.grid.nx, c.grid.ny, c.grid.nz, c.grid.hx, c.grid.hy,
                  c.grid.hz, 2 * delta, c.reTau, c.mach, c.fdt, 0, false,
                  "channel", "periodic", "compressible", c.kick, ""};
        return c;
    }

    // --- flow ---------------------------------------------------------------
    // A scene file replaces --body: main loads it (phase 1.5) and hands the
    // parsed Scene to the geometry build; the validator only records the path
    // and applies the body-independent checks.
    if (!r.scene.empty())
    {
        if (!r.body.empty())
            return bad(2, "--scene", "--scene and --body are mutually exclusive "
                                     "(the scene lists its own objects)");
        c.bodyKind = BodyKind::Scene;
        c.scenePath = r.scene;
    }
    else if (r.body.empty())
        return bad(2, "--body", "required: one of cylinder, sphere, cube, lens, "
                                "fence, floor, none, a .stl/.gltf/.glb file, or "
                                "--scene FILE.toml");

    else if (r.body == "none") c.bodyKind = BodyKind::None;
    else if (r.body == "cylinder") c.bodyKind = BodyKind::Cylinder;
    else if (r.body == "sphere") c.bodyKind = BodyKind::Sphere;
    else if (r.body == "cube") c.bodyKind = BodyKind::Cube;
    else if (r.body == "lens") c.bodyKind = BodyKind::Lens;
    else if (r.body == "fence") c.bodyKind = BodyKind::Fence;
    else if (r.body == "floor") c.bodyKind = BodyKind::Floor;
    else if (hasSuffix(r.body, ".stl") || hasSuffix(r.body, ".gltf")
             || hasSuffix(r.body, ".glb"))
    {
        c.bodyKind = BodyKind::Mesh;
        c.meshPath = r.body;
    }
    else
        return bad(2, "--body", "unrecognised body '" + r.body
                                + "'; expected cylinder, sphere, cube, lens, fence, "
                                  "floor, none, or a path ending in .stl/.gltf/.glb");

    // Body-free box (wall boundary layers, channel-like cases): the
    // incompressible runner requires solid cells, so this is compressible-only.
    if (c.bodyKind == BodyKind::None && c.physics != Physics::Compressible)
        return bad(2, "--body", "--body none is compressible-only (the "
                                "incompressible tunnel needs a body)");

    // Far field defaults follow the physics when the user did not choose one.
    // Sponge is NEVER a default (any physics, any mode): it is an escape hatch that
    // eats the near field in a relaxation band. The compressible default is
    // the physical characteristic boundary; --far sponge must be asked for.
    if (!r.given.contains("--far"))
        c.far = c.physics == Physics::Compressible ? FarField::Nscbc
                                                   : FarField::Tunnel;

    const bool comp = c.physics == Physics::Compressible;
    if (comp && (c.far == FarField::Tunnel || c.far == FarField::Fringe))
        return bad(2, "--far", std::string("--far ") + name(c.far)
                   + " is an incompressible far field; compressible runs use "
                     "sponge, nscbc or periodic");
    if (!comp && (c.far == FarField::Sponge || c.far == FarField::Nscbc))
        return bad(2, "--far", std::string("--far ") + name(c.far)
                   + " is a compressible far field; incompressible runs use "
                     "tunnel, fringe or periodic");

    c.open = r.open;
    if (c.open && c.far != FarField::Sponge && c.far != FarField::Nscbc
        && c.far != FarField::Tunnel)
        return bad(2, "--open", std::string("--open has no meaning with --far ")
                   + name(c.far)
                   + " (tunnel, sponge and nscbc have lateral faces)");

    // Floor-mounted bodies (fence blocker, raised floor): they sit ON the
    // tunnel floor (sealed through the ghost row), so they need the
    // incompressible tunnel's solid lateral wall. Compressible wall rows are
    // far-field mirror rows; a ghost body intersecting them is unvalidated --
    // rejected, not silently degraded.
    if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
    {
        const char* b = c.bodyKind == BodyKind::Fence ? "fence" : "floor";
        if (comp && c.bodyKind == BodyKind::Fence)
            return bad(2, "--body", "fence is incompressible-only for now (a "
                                    "mid-plate blocker on the compressible "
                                    "mirror wall rows is unvalidated)");
        // Compressible floor: NSCBC/sponge x planes skip the fully-solid
        // rows (the ghost wall owns them); periodic has no walls at all.
        if (comp && c.far == FarField::Periodic)
            return bad(2, "--far", "floor needs closed y walls; use --far "
                                   "nscbc (default) or an explicit sponge");
        if (!comp && c.far != FarField::Tunnel)
            return bad(2, "--far", std::string(b) +
                                   " sits on the tunnel floor; incompressible "
                                   "runs need --far tunnel (periodic/fringe "
                                   "have no wall for it to stand on)");
        if (c.open)
            return bad(2, "--open", std::string("--open removes the lateral "
                                    "walls; the ") + b + " needs the floor wall");
    }

    // Grid: two separately validated conventions, dispatched on --physics.
    const int ny = r.given.contains("--ny") ? r.ny : r.n;
    if (ny < 1) return bad(2, "--ny", "must be >= 1");
    // --nz defaults to 1 (native 2D) for the z-invariant bodies and to --n for
    // the genuinely 3D ones: a sphere or a mesh in a one-cell-thick box has no
    // solid cells at all, which used to surface as a silent no-body run.
    const bool body3d = c.bodyKind == BodyKind::Sphere || c.bodyKind == BodyKind::Mesh;
    const int nzI = r.given.contains("--nz") ? r.nz : (body3d ? r.n : 1);
    if (nzI < 1) return bad(2, "--nz", "must be >= 1 (interior cells; 1 = native 2D)");
    if (nzI != 1 && nzI < 4)
        return bad(2, "--nz", "must be 1 (native 2D) or >= 4");
    if (body3d && nzI == 1)
        return bad(2, "--nz", std::string("--body ") + name(c.bodyKind)
                   + " is a 3D body; --nz 1 leaves it with no solid cells");
    if (!finitePos(r.dr)) return bad(2, "--dr", "must be finite and > 0");
    c.dr = r.dr;
    c.nzInterior = nzI;

    c.uInf = 1.0;
    if (comp)
    {
        // D = pi/4 fixed, h scales with ny AND --dr: box = dr*D in y
        // (dr*(n/ny)*D in x), D spans ny/dr cells. The dr = 8 default keeps
        // the historical 8D x 8D box BIT-IDENTICAL (the *(dr/8) factor is an
        // exact *1.0 and the 2.1107 placement literal is preserved). No ghost
        // ring: the compressible far fields act on interior rows.
        if (c.dr < 4)
            return bad(2, "--dr", "compressible box is dr*D wide; below 4 the "
                                  "body sits under 1D from the inflow row");
        const double D = pi / 4.0;
        c.h = D / 64.0 * (512.0 / double(ny)) * (c.dr / 8.0);
        const double lx = r.n * c.h, ly = ny * c.h;
        const double lz = nzI > 1 ? nzI * c.h : 1.0;
        c.grid = Grid::make(r.n, ny, nzI, lx, ly, lz);
        c.L = lx;
        // The floor is pinned to the box bottom (no ghost ring here): the
        // slab spans [0, D], cy = centre of the blocked strip.
        c.placement = {c.dr == 8.0 ? 2.1107 : lx / 3.0 + 0.0163,
                       c.bodyKind == BodyKind::Floor ? D / 2.0
                                                     : ly / 2.0 + 0.0146,
                       nzI > 1 ? lz / 2.0 : 0.5, D};
    }
    else
    {
        // L = 2 pi, h = L/n, one ghost cell on every side; --nz is interior.
        c.L = 2 * pi;
        c.h = c.L / double(r.n);
        const int nxT = r.n + 2, nyT = ny + 2;
        const int nzT = nzI <= 1 ? 1 : nzI + 2;
        const double lz = nzT == 1 ? 1.0 : nzT * c.h;
        c.grid = Grid::make(nxT, nyT, nzT, nxT * c.h, nyT * c.h, lz);
        const double D = c.L / c.dr;
        // Per-body placement offsets: each deliberately puts the body off-node
        // so the shedding instability trips. Carried verbatim (masterplan §6).
        // The fence is pinned to the floor instead: the wall face is at y = h
        // (top of the ghost row) and the blocker spans [wall, wall + D], so
        // cy = centre of the blocked strip. Its top edge is off-node through
        // the 0.31h x-offset alone.
        const double cx = c.h + c.L / 3.0 + 0.31 * c.h;
        const double cy = c.bodyKind == BodyKind::Fence
                                  || c.bodyKind == BodyKind::Floor
                              ? c.h + D / 2.0
                              : c.h + c.L / 2.0 + 0.17 * c.h;
        const double cz = c.bodyKind == BodyKind::Sphere
                              ? c.h + c.L / 2.0 + 0.23 * c.h
                              : (nzT == 1 ? 0.5 * c.h : c.h + nzI * c.h / 2.0);
        c.placement = {cx, cy, cz, D};
    }

    const std::size_t cells = std::size_t(c.grid.nx) * c.grid.ny * c.grid.nz;
    if (cells > std::size_t(2147483647))
        return bad(2, "--n", "grid exceeds INT_MAX cells");
    if (c.placement.D < 4 * c.h)
        return bad(2, "--dr", "body diameter is under 4 cells; refine --n or lower "
                              "--dr");
    // The fence is D/8 thick; below 32 cells of D the blocker itself is under
    // 4 cells and the sampled cut degenerates.
    if (c.bodyKind == BodyKind::Fence && c.placement.D < 32 * c.h)
        return bad(2, "--dr", "fence thickness is D/8; D under 32 cells makes it "
                              "thinner than 4 cells -- refine --n or lower --dr");
    if (c.bodyKind == BodyKind::Floor && c.dr < 2)
        return bad(2, "--dr", "floor solidifies the bottom 1/dr of the channel; "
                              "--dr under 2 blocks more than half of it");

    // Physics.
    c.mach = r.mach;
    if (comp)
    {
        if (!(std::isfinite(c.mach) && c.mach > 0 && c.mach < 1))
            return bad(2, "--mach", "must be in (0,1) -- the flux is Kennedy-Gruber "
                                    "central with no shock capturing");
        if (r.given.contains("--tol"))
            return bad(2, "--tol", "compressible runs are fully explicit; there is "
                                   "no pressure solve to converge");
    }
    else
    {
        for (const char* f : {"--mach", "--kick"})
            if (r.given.contains(f))
                return bad(2, f, std::string(f) + " is compressible-only");
    }

    c.nu = c.uInf * c.placement.D / c.re;
    c.cfl = r.given.contains("--cfl") ? r.cfl : (comp ? 0.4 : 0.8);
    if (!(c.cfl > 0 && c.cfl <= 1)) return bad(2, "--cfl", "must be in (0,1]");

    c.cs = r.les;
    if (!(std::isfinite(c.cs) && c.cs >= 0))
        return bad(2, "--les", "must be finite and >= 0");
    c.sgsModel = r.sgs;
    if (r.given.contains("--sgs") && c.cs == 0)
        return bad(2, "--sgs", "selects the SGS model for --les; give --les CS "
                               "(> 0) or drop --sgs");
    c.cutFlux = r.cutFlux;
    if (c.cutFlux && c.bodyKind == BodyKind::None)
        return bad(2, "--cut-flux", "apertures the body's momentum fluxes; "
                                    "there is no body");
    if (c.cutFlux && r.given.contains("--stretch"))
        return bad(2, "--cut-flux", "apertures are uniform-mesh geometry; the "
                                    "curvilinear path has none. Drop --stretch");
    if (c.cutFlux && comp)
        return bad(2, "--cut-flux", "compressible cut-flux DIVERGES near the "
                                    "body (measured twice, 2026-07-23: the KG "
                                    "split flux loses its KE property under "
                                    "per-face area weighting). Incompressible "
                                    "only until redesigned");
    c.penalize = r.penalize;
    if (!(std::isfinite(c.penalize) && c.penalize >= 0))
        return bad(2, "--penalize", "must be finite and >= 0");
    if (c.penalize > 0 && c.bodyKind == BodyKind::None)
        return bad(2, "--penalize", "decays the near-body band; there is no body");
    if (c.penalize > 0 && r.given.contains("--stretch"))
        return bad(2, "--penalize", "needs the uniform-mesh fluid fractions; "
                                    "drop --stretch");
    if (c.sgsModel == SgsModel::Sigma && c.cs > 0 && c.grid.nz == 1)
        return bad(2, "--sgs", "sigma: D_sigma is identically ZERO for any 2D "
                               "flow (that is the model's design); at nz == 1 "
                               "it is a silent no-op. Use nz > 1 or smag");
    // Honest labelling: the boundary layer scales as D/sqrt(Re).
    const double blCells = c.placement.D / std::sqrt(c.re) / c.h;
    c.lesLabel = c.cs > 0;
    if (blCells < 3.0 && !c.lesLabel && c.bodyKind != BodyKind::None)
    {
        char w[192];
        std::snprintf(w, sizeof w,
                      "boundary layer is %.1f cells (D/sqrt(Re)/h); this is "
                      "under-resolved for DNS", blCells);
        c.warnings.emplace_back(w);
    }

    // Far-field parameters.
    c.nscbcBeta = r.nscbcBeta;
    c.nscbcStrip = r.nscbcStrip;
    if (c.far == FarField::Nscbc)
    {
        // Empirically bracketed stability window (masterplan §6.3): 0 pumps a
        // growing global mode, 1.0 diverges at t = 35.
        if (!(std::isfinite(c.nscbcBeta) && c.nscbcBeta >= 0 && c.nscbcBeta < 1))
            return bad(2, "--nscbc-beta", "must be in [0,1); 1.0 is measured to diverge "
                                          "at t = 35 (full transverse cancellation "
                                          "is ill-posed at low Mach)");
        if (c.nscbcStrip < 0 || c.nscbcStrip >= c.grid.nx / 4)
            return bad(2, "--nscbc-strip", "must be >= 0 and < nx/4");
        if (c.re > 800 && c.nscbcStrip == 0)
            return bad(2, "--nscbc-strip", "pure NSCBC is validated only up to "
                                           "Re 800 (it diverges at Re 2000, 2048^2); "
                                           "pass --nscbc-strip ~ny/64");
    }
    else if (r.given.contains("--nscbc-strip") && c.nscbcStrip > 0)
        return bad(2, "--nscbc-strip", "requires --far nscbc");
    c.farPot = r.farPot;
    if (c.farPot && c.bodyKind == BodyKind::None)
        return bad(2, "--pot", "models the body's wake source and displacement "
                               "doublet; meaningless with --body none");
    if (c.farPot && c.far != FarField::Nscbc
        && !(c.far == FarField::Tunnel && c.open))
        return bad(2, "--pot", "requires --far nscbc (compressible) or "
                               "--far tunnel --open (incompressible) -- the "
                               "correction retargets an open far field");

    // Sponge / fringe band width. clamp(0.1n, 8, n/2-1) is UB when n < 18.
    // Band width is measured along x for BOTH physics. (This was once a
    // `comp ? nx : nx` ternary -- a physics distinction someone intended and
    // abandoned. Restoring one needs a measurement, not a guess.)
    const int bandBase = c.grid.nx;
    if ((c.far == FarField::Fringe || c.far == FarField::Sponge
         || c.far == FarField::Nscbc)
        && bandBase < 18)
        return bad(2, "--n", "sponge/fringe far fields need at least 18 cells along x");
    c.spongeCells = r.spongeCells > 0
                        ? r.spongeCells
                        : std::clamp(int(0.1 * bandBase), 8, bandBase / 2 - 1);
    if (c.spongeCells < 1 || c.spongeCells >= bandBase / 2)
        return bad(2, "--sponge-cells", "must be >= 1 and < nx/2");

    // Auto: band z only when the box is thick enough to hold the band. A thin
    // quasi-2D run gets x,y only -- a z band there would sponge the whole wake.
    const bool zBandFits = c.grid.nz >= 2 * c.spongeCells + 2;
    c.fringeAxes = r.fringeAxes != 0 ? r.fringeAxes : (zBandFits ? 3 : 2);
    if (c.fringeAxes != 2 && c.fringeAxes != 3)
        return bad(2, "--fringe-axes", "must be 2 or 3");
    if (c.far == FarField::Fringe && c.fringeAxes == 3 && !zBandFits)
        return bad(2, "--fringe-axes", "a z fringe band needs nz >= 2*cells + 2; "
                                       "pass --fringe-axes 2 for a z-invariant run");

    // Curvilinear y-stretch + wall type. Scope is deliberate: the interior
    // scheme (curv finite-volume path) is general and NSCBC evaluates the
    // PointRemap on device (physical distances in its one-sided and
    // transverse terms), but the immersed ghost wall still assumes uniform
    // spacing and the curvilinear solver has no LES -- each combination is
    // rejected, not silently degraded. Sponge stays explicit per the ground
    // rule: the validator never swaps --far for the user.
    c.stretchBeta = r.stretch;
    c.noslipY = r.noslip;
    if (!(std::isfinite(c.stretchBeta) && c.stretchBeta >= 0 && c.stretchBeta <= 4))
        return bad(2, "--stretch", "must be in [0,4] (tanh clustering strength; "
                                   "4 already compresses the wall cell ~190x)");
    if (c.stretchBeta > 0)
    {
        if (!comp)
            return bad(2, "--stretch", "the curvilinear solver is compressible-"
                                       "only; --stretch needs --physics "
                                       "compressible");
        if (c.bodyKind != BodyKind::None)
            return bad(2, "--stretch", "the ghost wall samples the body SDF at "
                                       "uniform cell positions; stretched runs "
                                       "need --body none for now");
        if (c.open)
            return bad(2, "--open", "--stretch clusters cells toward the closed "
                                    "y walls; open lateral faces defeat it");
        if (c.far != FarField::Nscbc && c.far != FarField::Sponge)
            return bad(2, "--far", "stretched-y runs need --far nscbc (the "
                                   "characteristic terms evaluate the PointRemap) "
                                   "or an explicit --far sponge");
        // Resolved wall resolution: first INTERIOR cell (row 1; rows 0 and
        // ny-1 are the wall-mirror ghost rows) and max growth ratio. These
        // set the acoustic dt, so --dry-run must surface them.
        const TanhStretchY R{double(c.grid.ny) * c.grid.hy, c.stretchBeta};
        const double hy = c.grid.hy;
        c.stretchH1 = R.mapY(2 * hy) - R.mapY(hy);
        double prev = R.mapY(hy) - R.mapY(0.0);
        for (int j = 1; j < c.grid.ny; ++j)
        {
            const double dy = R.mapY((j + 1) * hy) - R.mapY(j * hy);
            c.stretchGrowth = std::max(c.stretchGrowth,
                                       std::max(dy / prev, prev / dy));
            prev = dy;
        }
    }
    // Focal clustering: a fine plateau box (typically the body plus its wake)
    // with a coarse far field, one independent Focal1D per axis. Same scope
    // as --stretch (compressible curvilinear, NSCBC or explicit sponge; LES is
    // allowed, the SGS operators read the mesh metrics and the per-cell
    // Delta), but bodies ARE supported here: the ghost wall classifies
    // against the mapped cell centres. The blend support must stay clear of
    // the domain faces, so the map is the exact identity there and the
    // periodic wrap stays C^2.
    if (r.given.contains("--focal"))
    {
        if (!comp)
            return bad(2, "--focal", "the curvilinear solver is compressible-"
                                     "only; --focal needs --physics "
                                     "compressible");
        if (c.stretchBeta > 0)
            return bad(2, "--focal", "--stretch and --focal are different "
                                     "curvilinear meshes; pick one");
        // Bodies ARE supported (the ghost wall classifies against the mapped
        // cell centres), but only the analytic ones: their AABB follows from
        // the placement alone, and validate() may not touch the filesystem to
        // read a mesh.
        if (c.bodyKind == BodyKind::Mesh || c.bodyKind == BodyKind::Scene)
            return bad(2, "--focal", "a mesh/scene body's bounding box is only "
                                     "known after the file is loaded, and "
                                     "validation touches no filesystem; give "
                                     "the plateau explicitly for now");
        if (c.bodyKind == BodyKind::Fence || c.bodyKind == BodyKind::Floor)
            return bad(2, "--focal", "the wall-mounted bodies span the full "
                                     "domain width; there is no compact "
                                     "plateau to cluster on");
        if (c.cutFlux || c.penalize > 0)
            return bad(2, "--focal", "apertures and the Brinkman band are "
                                     "uniform-mesh geometry; the curvilinear "
                                     "path has none");
        if (c.open)
            return bad(2, "--open", "--focal needs the closed lateral faces "
                                    "its blend margin is measured against");
        if (c.far != FarField::Nscbc && c.far != FarField::Sponge)
            return bad(2, "--far", "focal runs need --far nscbc (the "
                                   "characteristic terms evaluate the "
                                   "PointRemap) or an explicit --far sponge");

        c.focalRatio = r.focalRatio;
        if (!(std::isfinite(c.focalRatio) && c.focalRatio > 1
              && c.focalRatio <= 16))
            return bad(2, "--focal-ratio", "must be in (1,16] (coarse/fine "
                                           "cell-size ratio; the fine spacing "
                                           "sets the acoustic dt)");
        const double blendFrac = r.focalBlend;
        if (!(std::isfinite(blendFrac) && blendFrac > 0 && blendFrac <= 2))
            return bad(2, "--focal-blend", "must be in (0,2] (blend width per "
                                           "side as a fraction of the plateau)");

        // The body's axis-aligned bounds, when there is a body: the
        // containment rule below keeps the ghost wall's neighbourhood inside
        // the plateau, so it always sees a locally UNIFORM (anisotropic) grid
        // -- the geometry the Cartesian construction was validated on. The
        // cylinder is z-invariant, so its z extent is the whole box.
        const char* axName[3] = {"x", "y", "z"};
        const bool hasBody = c.bodyKind != BodyKind::None;
        const bool zSpanning = c.bodyKind == BodyKind::Cylinder;
        const double bLo[3] = {c.placement.cx - 0.5 * c.placement.D,
                               c.placement.cy - 0.5 * c.placement.D,
                               c.placement.cz - 0.5 * c.placement.D};
        const double bHi[3] = {c.placement.cx + 0.5 * c.placement.D,
                               c.placement.cy + 0.5 * c.placement.D,
                               c.placement.cz + 0.5 * c.placement.D};

        double box[6];
        if (r.focal == "auto")
        {
            if (!hasBody)
                return bad(2, "--focal", "auto needs a body; --body none has "
                                         "no AABB to focus on");
            if (!(std::isfinite(r.focalWake) && r.focalWake >= 0))
                return bad(2, "--focal-wake", "must be finite and >= 0");
            if (!(std::isfinite(r.focalPad) && r.focalPad >= 0))
                return bad(2, "--focal-pad", "must be finite and >= 0");
            const double D = c.placement.D, pad = r.focalPad * D;
            const int nA[3] = {c.grid.nx, c.grid.ny, c.grid.nz};
            const double hA[3] = {c.grid.hx, c.grid.hy, c.grid.hz};
            for (int a = 0; a < 3; ++a)
            {
                // Axes the body does not localize stay uniform.
                if (a == 2 && (zSpanning || c.grid.nz == 1))
                {
                    box[4] = box[5] = 0.0;
                    continue;
                }
                const double want[2] = {
                    bLo[a] - pad,
                    bHi[a] + pad + (a == 0 ? r.focalWake * D : 0.0)};
                // Smallest plateau that still satisfies the containment rule
                // below; the desired box shrinks toward it until its blend
                // clears the domain faces. The resolved box is what
                // --dry-run prints, so the shrink is never silent.
                const double guard = 3.0 * hA[a] / c.focalRatio;
                const double need[2] = {bLo[a] - guard, bHi[a] + guard};
                const double Lax = nA[a] * hA[a];
                const double m =
                    (a == 0 ? std::max(4, c.nscbcStrip + 2) : 4) * hA[a];
                auto fits = [&](double lo, double hi) {
                    const double w = blendFrac * (hi - lo);
                    return hi > lo && lo - w >= m && hi + w <= Lax - m;
                };
                if (!fits(std::min(need[0], want[0]),
                          std::max(need[1], want[1])) &&
                    !fits(need[0], need[1]))
                    return bad(2, "--focal", std::string(axName[a])
                                   + " has no plateau that both contains the "
                                     "body and clears the domain faces; the "
                                     "box is too small for --focal-blend");
                double t = 1.0;
                if (!fits(want[0], want[1]))
                {
                    double lo = 0.0, hi = 1.0; // largest fitting t
                    for (int it = 0; it < 60; ++it)
                    {
                        const double mid = 0.5 * (lo + hi);
                        const double a0 = need[0] + mid * (want[0] - need[0]);
                        const double a1 = need[1] + mid * (want[1] - need[1]);
                        (fits(a0, a1) ? lo : hi) = mid;
                    }
                    t = lo;
                }
                box[2 * a] = need[0] + t * (want[0] - need[0]);
                box[2 * a + 1] = need[1] + t * (want[1] - need[1]);
            }
        }
        // "x0,x1,y0,y1,z0,z1"; a degenerate pair leaves that axis uniform.
        else {
            std::string_view s(r.focal);
            for (int q = 0; q < 6; ++q)
            {
                const std::size_t comma = s.find(',');
                const std::string tok(s.substr(0, comma));
                char* end = nullptr;
                box[q] = std::strtod(tok.c_str(), &end);
                if (tok.empty() || end != tok.c_str() + tok.size()
                    || !std::isfinite(box[q]))
                    return bad(2, "--focal", "expects six numbers "
                                             "x0,x1,y0,y1,z0,z1 (world units) "
                                             "or 'auto'");
                if (comma == std::string_view::npos)
                {
                    if (q != 5)
                        return bad(2, "--focal", "expects six comma-separated "
                                                 "numbers x0,x1,y0,y1,z0,z1");
                    s = {};
                }
                else
                    s = s.substr(comma + 1);
            }
            if (!s.empty())
                return bad(2, "--focal", "expects exactly six numbers "
                                         "x0,x1,y0,y1,z0,z1");
        }

        const int nAx[3] = {c.grid.nx, c.grid.ny, c.grid.nz};
        const double hAx[3] = {c.grid.hx, c.grid.hy, c.grid.hz};
        for (int a = 0; a < 3; ++a)
        {
            const double lo = box[2 * a], hi = box[2 * a + 1];
            const double L = nAx[a] * hAx[a];
            if (hi <= lo)
            {
                // Degenerate pair = leave this axis uniform.
                if (hi < lo)
                    return bad(2, "--focal", std::string(axName[a])
                                                 + " plateau has hi < lo");
                c.focalHFine[a] = c.focalHCoarse[a] = hAx[a];
                continue;
            }
            if (a == 2 && c.grid.nz == 1)
                return bad(2, "--focal", "nz == 1 IS the exact 2D restriction; "
                                         "a z plateau there is meaningless -- "
                                         "pass z0,z1 equal");
            const double w = blendFrac * (hi - lo);
            // Identity margin at both faces: the blend support must end at
            // least 4 cells short of each domain face (x also clears the
            // NSCBC rows and any absorbing strip), so the far grid is
            // exactly uniform where the boundary treatment reads it.
            const int mCells = a == 0 ? std::max(4, c.nscbcStrip + 2) : 4;
            const double margin = mCells * hAx[a];
            if (!(lo - w >= margin && hi + w <= L - margin))
                return bad(2, "--focal",
                           std::string(axName[a]) + " plateau plus its blend "
                           "must clear both domain faces by " +
                           std::to_string(mCells) + " cells (the map must be "
                           "the exact identity there); shrink the box or "
                           "--focal-blend");
            const double fine = hAx[a] / c.focalRatio;
            if (hi - lo < 4 * fine)
                return bad(2, "--focal", std::string(axName[a])
                                             + " plateau spans under 4 fine "
                                               "cells; widen it");
            // Containment: the body plus the ghost wall's 3-cell mirror
            // neighbourhood must sit inside the PLATEAU, never in the blend.
            // The wall stencil is then built on locally uniform cells, which
            // is the geometry it was validated on.
            if (hasBody && !(a == 2 && zSpanning))
            {
                const double guard = 3.0 * fine;
                if (!(lo <= bLo[a] - guard && hi >= bHi[a] + guard))
                    return bad(2, "--focal",
                               std::string(axName[a]) + " plateau must contain "
                               "the body plus a 3-fine-cell guard band "
                               "([" + std::to_string(bLo[a] - guard) + "," +
                               std::to_string(bHi[a] + guard) + "]); the ghost "
                               "wall's mirror stencil must not straddle the "
                               "coarse/fine blend");
            }
            c.focalActive = true;
            c.focalLo[a] = lo;
            c.focalHi[a] = hi;
            c.focalW[a] = w;
            const Focal1D f = Focal1D::make(L, lo, hi, w, c.focalRatio);
            c.focalHFine[a] = f.c * f.r * hAx[a];
            c.focalHCoarse[a] = f.c * hAx[a];
            // Discrete adjacent growth: the KG flux carries no dissipation,
            // so a fast coarse->fine transition is a dispersive reflector.
            double prev = f.map(hAx[a]) - f.map(0.0);
            for (int i = 1; i < nAx[a]; ++i)
            {
                const double d = f.map((i + 1) * hAx[a]) - f.map(i * hAx[a]);
                c.focalGrowth =
                    std::max(c.focalGrowth, std::max(d / prev, prev / d));
                prev = d;
            }
        }
        if (!c.focalActive)
            return bad(2, "--focal", "every axis pair is degenerate; the mesh "
                                     "would be uniform -- drop --focal");
        if (c.focalGrowth > 1.25)
            return bad(2, "--focal-blend", "max adjacent cell growth " +
                       std::to_string(c.focalGrowth) + " exceeds 1.25; the "
                       "central flux has no dissipation and reflects off a "
                       "sharp transition. Widen --focal-blend or lower "
                       "--focal-ratio");
        const double hUni = c.grid.nz == 1
                                ? std::min(c.grid.hx, c.grid.hy)
                                : std::min({c.grid.hx, c.grid.hy, c.grid.hz});
        const double hFoc = c.grid.nz == 1
                                ? std::min(c.focalHFine[0], c.focalHFine[1])
                                : std::min({c.focalHFine[0], c.focalHFine[1],
                                            c.focalHFine[2]});
        c.focalDtFactor = hFoc / hUni;
    }
    else
    {
        for (const char* f : {"--focal-ratio", "--focal-blend", "--focal-wake",
                              "--focal-pad"})
            if (r.given.contains(f))
                return bad(2, f, std::string(f) + " requires --focal");
    }

    if (c.noslipY)
    {
        if (!comp)
            return bad(2, "--noslip", "the y mirror walls belong to the "
                                      "compressible far field; --noslip needs "
                                      "--physics compressible");
        if (c.open)
            return bad(2, "--noslip", "--open removes the y walls; there is "
                                      "nothing to make no-slip");
        if (c.far != FarField::Sponge && c.far != FarField::Nscbc)
            return bad(2, "--noslip", "requires a far field with closed y walls "
                                      "(--far sponge or --far nscbc)");
    }

    // Start-up. The raw default (2.0) is the COMPRESSIBLE default; the
    // incompressible ramp is opt-in so existing impulsive-start trajectories
    // and anchors do not silently move.
    c.ramp = comp ? r.ramp : (r.given.contains("--ramp") ? r.ramp : 0.0);
    if (!comp && c.ramp > 0 && (r.seq > 1 || r.steady > 0))
        return bad(2, "--ramp", "the quiet start would re-ramp the developed "
                                "field each --seq level / fight --steady; "
                                "use an impulsive start there");
    c.kick = r.kick;
    if (!(std::isfinite(c.ramp) && c.ramp >= 0)) return bad(2, "--ramp", "must be >= 0");
    if (!std::isfinite(c.kick)) return bad(2, "--kick", "must be finite");

    // Solver.
    c.tol = r.tol;
    if (!(std::isfinite(c.tol) && c.tol > 0)) return bad(2, "--tol", "must be > 0");
    c.maxIters = r.maxIters;
    if (c.maxIters < 1) return bad(2, "--max-iters", "must be >= 1");
    // Default per current app: 16 for the compressible cylinder (a validated
    // perf fix, masterplan §6.7), 1 everywhere else.
    c.dtRecompute = r.given.contains("--dt-recompute")
                        ? r.dtRecompute
                        : (comp && c.bodyKind == BodyKind::Cylinder ? 16 : 1);
    if (c.dtRecompute < 1) return bad(2, "--dt-recompute", "must be >= 1");

    // Geometry options.
    c.aspect = r.aspect;
    if (c.bodyKind == BodyKind::Lens && !(c.aspect > 1.0))
        return bad(2, "--aspect", "must be > 1 (the lens needs w > hh)");
    c.dimples = r.dimples;
    if (c.dimples < 0) return bad(2, "--dimples", "must be >= 0");
    if (c.dimples > 0 && c.bodyKind != BodyKind::Cylinder
        && c.bodyKind != BodyKind::Sphere)
        return bad(2, "--dimples", "only cylinder and sphere carry dimples");
    if (c.dimples > 0 && c.placement.D < 16 * c.h)
        return bad(2, "--dimples", "dimple pits are 5h wide; D must span at least 16 "
                                   "cells or they consume the body");
    c.bumps = r.bumps;
    if (c.bumps < 0) return bad(2, "--bumps", "must be >= 0");
    if (c.bumps > 0 && c.bodyKind != BodyKind::Floor)
        return bad(2, "--bumps", "the surface ripple belongs to --body floor");
    c.bumpH = r.bumpH;
    if (c.bumps > 0)
    {
        if (!(std::isfinite(c.bumpH) && c.bumpH >= 4 && c.bumpH <= 48))
            return bad(2, "--bump-h", "must be in [4,48] cells (base = height, "
                                      "63-degree flanks; under 4 the tip is "
                                      "sub-cell, over 48 it is a fence, not "
                                      "roughness)");
        if (c.bumps > int(0.65 * c.n / (2.0 * c.bumpH)))
            return bad(2, "--bumps", "spike pitch is under twice the tooth "
                                     "base; the teeth merge -- fewer bumps, "
                                     "shorter --bump-h, or more cells");
        if (c.placement.D < 4 * c.bumpH * c.h)
            return bad(2, "--bump-h", "spikes taller than D/4 are a fence "
                                      "field, not surface roughness -- lower "
                                      "--bump-h or --dr");
    }
    c.scale = r.scale;
    if (!(std::isfinite(c.scale) && c.scale >= 0))
        return bad(2, "--scale", "must be finite and >= 0 (0 = fit D)");
    c.pitch = r.pitch;
    c.yaw = r.yaw;
    c.robust = r.robust;
    if (c.bodyKind != BodyKind::Mesh)
        for (const char* f : {"--scale", "--pitch", "--yaw", "--robust"})
            if (r.given.contains(f))
                return bad(2, f, std::string(f) + " applies to mesh bodies only");

    // Run length + output cadences.
    c.ft = r.ft;
    if (!finitePos(c.ft)) return bad(2, "--ft", "must be finite and > 0");
    // One flow-through is L/uInf in both physics (c.L already carries each
    // physics' own box convention). Formerly a `comp ? X : X` ternary.
    c.tEnd = c.ft * c.L / c.uInf;

    // Stationary mode. tEnd stops being the run length and becomes the
    // give-up cap: not converging by then is exit 8, never a silent answer.
    c.steadyTol = r.steady;
    if (c.steadyTol < 0 || !std::isfinite(c.steadyTol))
        return bad(2, "--steady", "must be finite and >= 0 (0 = time-march)");
    if (c.steadyTol > 0 && comp)
        return bad(2, "--steady",
                   "is incompressible-only; the compressible loop has acoustic "
                   "transients that a velocity residual cannot certify");
    if (c.steadyTol > 0 && c.cs > 0)
        return bad(2, "--steady",
                   "with --les is meaningless: a filtered field has no "
                   "stationary state to converge to");

    // Grid sequencing. Only the initial guess comes from the coarse level, so
    // the fine-grid fixed point is unchanged; the ladder just skips paying the
    // startup transient at full resolution.
    c.seqLevels = r.seq;
    if (c.seqLevels < 1 || c.seqLevels > 6)
        return bad(2, "--seq", "must be in [1,6] (1 = off)");
    if (c.seqLevels > 1)
    {
        if (c.steadyTol <= 0)
            return bad(2, "--seq", "requires --steady: without a convergence "
                                   "test there is nothing to advance a level on");
        // The coarsest level still has to resolve the body: same D >= 4h rule.
        const int nCoarse = r.n >> (c.seqLevels - 1);
        if (nCoarse < 16 || double(nCoarse) < 4.0 * c.dr)
            return bad(2, "--seq", "coarsest level would not resolve the body "
                                   "(needs n/2^(levels-1) >= max(16, 4*dr)); "
                                   "use fewer levels or a larger --n");
    }
    c.fdt = r.fdt;
    c.pdt = r.pdt > 0 ? r.pdt : c.fdt;
    c.wdt = r.wdt > 0 ? r.wdt : c.fdt;
    for (auto [v, f] : {std::pair{c.fdt, "--fdt"}, {c.pdt, "--pdt"}, {c.wdt, "--wdt"}})
        if (!finitePos(v)) return bad(2, f, "must be finite and > 0");
    c.planes = r.planes;
    if (c.planes != 1 && c.planes != 3)
        return bad(2, "--planes", "must be 1 (centre only) or 3 (centre +/- n/16)");
    c.pngDown = r.pngDown;
    if (c.pngDown < 1)
        return bad(2, "--png-down", "must be >= 1 (1 = full-resolution PNGs)");
    if (c.pngDown > 1 && (c.grid.nx % c.pngDown || c.grid.ny % c.pngDown))
        return bad(2, "--png-down",
                   "must divide both nx and ny evenly for box-averaging");
    c.vortMax = r.vortMax;
    if (!(std::isfinite(c.vortMax) && c.vortMax >= 0))
        return bad(2, "--vort-max", "must be finite and >= 0 (0 = auto scale)");
    c.vortAsinh = r.vortMap == 1;
    if (c.vortAsinh && !(c.vortMax > 0))
        return bad(2, "--vort-map", "asinh needs an explicit full-scale clip; "
                                    "set --vort-max (~ the peak |vorticity| in "
                                    "uInf/D, e.g. sqrt(Re))");
    c.speedDelta = r.speedMap == 1;
    c.vortSigned = r.vortSigned;
    if (c.vortSigned && !comp)
        return bad(2, "--vort-signed", "incompressible frames are already "
                                       "signed z-vorticity; the flag selects "
                                       "it for 3D compressible runs");

    if (c.emit.empty()) { c.emit.insert(Emit::Stats); c.emit.insert(Emit::Bin); }
    if (c.emit.contains(Emit::None) && c.emit.size() > 1)
        return bad(2, "--emit", "--emit none cannot be combined with other kinds");
    // PNG at nz > 1 renders the centre z-plane (same machinery the channel
    // already uses); the old flow-only nz == 1 restriction was conservative,
    // not load-bearing.

    // NOTE: qRef is deliberately NOT resolved here. It is 0.5 rho uInf^2 A
    // with A the frontal area MEASURED from the cut apertures, and validate()
    // touches no filesystem and builds no geometry, so it cannot know A. The
    // field that used to live here held 0.5 uInf^2 D span -- a placeholder
    // that --dry-run printed as if it were the number the run would divide by.
    // The real value comes from dns/geometry.hpp's referenceQ*(), once.

    c.meta = {c.grid.nx, c.grid.ny, c.grid.nz, c.grid.hx, c.grid.hy, c.grid.hz,
              c.placement.D, c.re, comp ? c.mach : 0.0, c.fdt, c.cs, c.lesLabel,
              c.bodyKind == BodyKind::Mesh    ? c.meshPath
              : c.bodyKind == BodyKind::Scene ? "scene:" + c.scenePath
                                              : name(c.bodyKind),
              name(c.far), name(c.physics), comp ? c.kick : 0.0,
              c.lesLabel ? name(c.sgsModel) : ""};
    return c;
}

// ---------------------------------------------------------------------------

std::string Case::describe() const
{
    if (sub == Subcommand::Channel)
    {
        char buf[2048];
        std::snprintf(buf, sizeof buf,
            "case:\n"
            "  mode       compressible turbulent channel (periodic x/z, "
            "no-slip y walls)\n"
            "  grid       %d x %d x %d (%d fluid rows + 2 ghost)  "
            "box %.6g x %.6g x %.6g%s\n"
            "  mesh       y tanh-stretch beta=%.4g: first cell %.3e "
            "(y+ = %.3g), growth <= %.4f\n"
            "  physics    Re_tau=%.6g  nu=%.6g  Mach_bulk=%.3g  "
            "(u_bulk est %.3g, p=%.6g)  cfl=%.3g\n"
            "  walls      %s no-slip mirrors; force f=1 (constant), "
            "u_tau target 1\n"
            "  run        tt=%.6g eddy turnovers (tEnd=%.6g)  kick=%.3g  "
            "profile-dt=%.3g\n"
            "  label      DNS%s\n"
            "  out        %s\n",
            grid.nx, grid.ny, grid.nz, grid.ny - 2,
            lxBox, 2.0, lzBox, grid.nz == 1 ? "  [native 2D]" : "",
            stretchBeta, chanH1, chanH1 * reTau, stretchGrowth,
            reTau, nu, mach, ubEst, pRef, cfl,
            wallIso ? "isothermal" : "ADIABATIC (no heat sink -- short "
                                     "transients only)",
            ft, tEnd, kick, profileDt,
            grid.nz == 1 ? " (2D: qualitative only, no vortex stretching)"
                         : "",
            outPrefix.c_str());
        return buf;
    }
    char buf[2048];
    std::snprintf(buf, sizeof buf,
        "case:\n"
        "  physics    %s\n"
        "  body       %s%s%s\n"
        "  far field  %s%s\n"
        "  grid       %d x %d x %d  h=%.6g  L=%.6g%s\n"
        "  placement  cx=%.6g cy=%.6g cz=%.6g D=%.6g (%.1f cells)\n"
        "  Re=%.6g  nu=%.6g  Mach=%.6g  cfl=%.6g  Cs=%.6g (%s)\n"
        "  ft=%.6g  tEnd=%.6g  fdt=%.6g\n"
        "  mode       %s\n"
        "  out        %s\n",
        name(physics), name(bodyKind),
        bodyKind == BodyKind::Scene
            ? (" " + scenePath).c_str()
            : meshPath.empty() ? "" : (" " + meshPath).c_str(),
        cutFlux ? " [cut-flux]" : "",
        name(far),
        open ? (farPot ? " (lateral open, potential-corrected)"
                       : " (lateral open)")
             : (farPot ? " (potential-corrected)" : ""),
        grid.nx, grid.ny, grid.nz, h, L, grid.nz == 1 ? "  [native 2D]" : "",
        placement.cx, placement.cy, placement.cz, placement.D,
        h > 0 ? placement.D / h : 0.0,
        re, nu, mach, cfl, cs,
        lesLabel ? (sgsModel == SgsModel::Sigma ? "LES sigma" : "LES smag")
                 : "DNS",
        ft, tEnd, fdt,
        steadyTol > 0 ? "stationary (stop at steady residual, tEnd = give-up cap)"
                      : "time-march to tEnd",
        outPrefix.c_str());
    std::string s = buf;
    if (stretchBeta > 0)
    {
        char m[256];
        std::snprintf(m, sizeof m,
                      "  mesh       y tanh-stretch beta=%.4g: first interior cell "
                      "%.3e (hy/%.1f), growth <= %.4f\n"
                      "             (acoustic dt shrinks with the wall cell)\n",
                      stretchBeta, stretchH1,
                      stretchH1 > 0 ? grid.hy / stretchH1 : 0.0, stretchGrowth);
        s += m;
    }
    if (focalActive)
    {
        char m[512];
        const char* ax[3] = {"x", "y", "z"};
        std::string per;
        for (int a = 0; a < 3; ++a)
        {
            if (focalHi[a] <= focalLo[a]) continue;
            char l[192];
            std::snprintf(l, sizeof l,
                          "             %s [%.4g,%.4g] blend %.4g: fine %.3e "
                          "coarse %.3e\n",
                          ax[a], focalLo[a], focalHi[a], focalW[a],
                          focalHFine[a], focalHCoarse[a]);
            per += l;
        }
        std::snprintf(m, sizeof m,
                      "  mesh       focal clustering ratio %.4g, growth <= "
                      "%.4f, dt factor %.3f\n",
                      focalRatio, focalGrowth, focalDtFactor);
        s += m;
        s += per;
    }
    if (noslipY) s += "  walls      y walls no-slip (adiabatic mirror)\n";
    return s;
}

} // namespace dns
