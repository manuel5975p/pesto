#include "dns/config.hpp"

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
    case BodyKind::Mesh: return "mesh";
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
        c.grid = Grid::make(r.n, r.n, r.n, 2 * pi, 2 * pi, 2 * pi);
        c.meta = {r.n, r.n, r.n, c.grid.hx, c.grid.hy, c.grid.hz,
                  0, 0, 0, 0, 0, false, "none", "periodic", name(c.physics), 0};
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
                  name(c.physics), 0};
        return c;
    }

    // --- flow ---------------------------------------------------------------
    if (r.body.empty())
        return bad(2, "--body", "required: one of cylinder, sphere, cube, lens, "
                                "or a .stl/.gltf/.glb file");

    if (r.body == "cylinder") c.bodyKind = BodyKind::Cylinder;
    else if (r.body == "sphere") c.bodyKind = BodyKind::Sphere;
    else if (r.body == "cube") c.bodyKind = BodyKind::Cube;
    else if (r.body == "lens") c.bodyKind = BodyKind::Lens;
    else if (hasSuffix(r.body, ".stl") || hasSuffix(r.body, ".gltf")
             || hasSuffix(r.body, ".glb"))
    {
        c.bodyKind = BodyKind::Mesh;
        c.meshPath = r.body;
    }
    else
        return bad(2, "--body", "unrecognised body '" + r.body
                                + "'; expected cylinder, sphere, cube, lens, or a "
                                  "path ending in .stl/.gltf/.glb");

    // Far field defaults follow the physics when the user did not choose one.
    if (!r.given.contains("--far"))
        c.far = c.physics == Physics::Compressible ? FarField::Sponge
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
        if (c.dr < 6)
            return bad(2, "--dr", "compressible box is dr*D wide; below 6 the "
                                  "body sits under 1.5D from the inflow row");
        const double D = pi / 4.0;
        c.h = D / 64.0 * (512.0 / double(ny)) * (c.dr / 8.0);
        const double lx = r.n * c.h, ly = ny * c.h;
        const double lz = nzI > 1 ? nzI * c.h : 1.0;
        c.grid = Grid::make(r.n, ny, nzI, lx, ly, lz);
        c.L = lx;
        c.placement = {c.dr == 8.0 ? 2.1107 : lx / 3.0 + 0.0163,
                       ly / 2.0 + 0.0146, nzI > 1 ? lz / 2.0 : 0.5, D};
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
        const double cx = c.h + c.L / 3.0 + 0.31 * c.h;
        const double cy = c.h + c.L / 2.0 + 0.17 * c.h;
        const double cz = c.bodyKind == BodyKind::Sphere
                              ? c.h + c.L / 2.0 + 0.23 * c.h
                              : (nzT == 1 ? 0.5 * c.h : c.h + nzI * c.h / 2.0);
        c.placement = {cx, cy, cz, D};
    }
    c.body = c.placement;

    const std::size_t cells = std::size_t(c.grid.nx) * c.grid.ny * c.grid.nz;
    if (cells > std::size_t(2147483647))
        return bad(2, "--n", "grid exceeds INT_MAX cells");
    if (c.placement.D < 4 * c.h)
        return bad(2, "--dr", "body diameter is under 4 cells; refine --n or lower "
                              "--dr");

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
        if (r.given.contains("--poisson"))
            return bad(2, "--poisson", "compressible runs have no Poisson solve");
    }
    else
    {
        for (const char* f : {"--mach", "--ramp", "--kick"})
            if (r.given.contains(f))
                return bad(2, f, std::string(f) + " is compressible-only");
    }

    c.nu = c.uInf * c.placement.D / c.re;
    c.cfl = r.given.contains("--cfl") ? r.cfl : (comp ? 0.4 : 0.8);
    if (!(c.cfl > 0 && c.cfl <= 1)) return bad(2, "--cfl", "must be in (0,1]");

    c.cs = r.les;
    if (!(std::isfinite(c.cs) && c.cs >= 0))
        return bad(2, "--les", "must be finite and >= 0");
    // Honest labelling: the boundary layer scales as D/sqrt(Re).
    const double blCells = c.placement.D / std::sqrt(c.re) / c.h;
    c.lesLabel = c.cs > 0;
    if (blCells < 3.0 && !c.lesLabel)
        std::fprintf(stderr, "dns: warning: boundary layer is %.1f cells "
                             "(D/sqrt(Re)/h); this is under-resolved for DNS\n",
                     blCells);

    // Far-field parameters.
    c.nscbcBeta = r.nscbcBeta;
    c.nscbcStrip = r.nscbcStrip;
    if (c.far == FarField::Nscbc)
    {
        // Empirically bracketed stability window (masterplan §6.3): 0 pumps a
        // growing global mode, 1.0 blows up at t = 35.
        if (!(std::isfinite(c.nscbcBeta) && c.nscbcBeta >= 0 && c.nscbcBeta < 1))
            return bad(2, "--nscbc-beta", "must be in [0,1); 1.0 is measured to blow "
                                          "up at t = 35 (full transverse cancellation "
                                          "is ill-posed at low Mach)");
        if (c.nscbcStrip < 0 || c.nscbcStrip >= c.grid.nx / 4)
            return bad(2, "--nscbc-strip", "must be >= 0 and < nx/4");
        if (c.re > 800 && c.nscbcStrip == 0)
            return bad(2, "--nscbc-strip", "pure NSCBC is validated only up to "
                                           "Re 800 (it blows up at Re 2000, 2048^2); "
                                           "pass --nscbc-strip ~ny/64");
    }
    else if (r.given.contains("--nscbc-strip") && c.nscbcStrip > 0)
        return bad(2, "--nscbc-strip", "requires --far nscbc");
    c.farPot = r.farPot;
    if (c.farPot && c.far != FarField::Nscbc
        && !(c.far == FarField::Tunnel && c.open))
        return bad(2, "--pot", "requires --far nscbc (compressible) or "
                               "--far tunnel --open (incompressible) -- the "
                               "correction retargets an open far field");

    // Sponge / fringe band width. clamp(0.1n, 8, n/2-1) is UB when n < 18.
    const int bandBase = comp ? c.grid.nx : c.grid.nx;
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

    // Start-up.
    c.ramp = r.ramp;
    c.kick = r.kick;
    if (!(std::isfinite(c.ramp) && c.ramp >= 0)) return bad(2, "--ramp", "must be >= 0");
    if (!std::isfinite(c.kick)) return bad(2, "--kick", "must be finite");

    // Solver.
    c.poisson = r.poisson;
    if (c.poisson == PoissonKind::Auto)
        c.poisson = comp ? PoissonKind::Auto
                         : (c.far == FarField::Tunnel ? PoissonKind::Dct
                                                      : PoissonKind::Fft);
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
    c.tEnd = comp ? c.ft * c.L / c.uInf : c.ft * c.L / c.uInf;

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
    c.fdt = r.fdt;
    c.pdt = r.pdt > 0 ? r.pdt : c.fdt;
    c.wdt = r.wdt > 0 ? r.wdt : c.fdt;
    for (auto [v, f] : {std::pair{c.fdt, "--fdt"}, {c.pdt, "--pdt"}, {c.wdt, "--wdt"}})
        if (!finitePos(v)) return bad(2, f, "must be finite and > 0");
    c.planes = r.planes;
    if (c.planes != 1 && c.planes != 3)
        return bad(2, "--planes", "must be 1 (centre only) or 3 (centre +/- n/16)");

    if (c.emit.empty()) { c.emit.insert(Emit::Stats); c.emit.insert(Emit::Bin); }
    if (c.emit.contains(Emit::None) && c.emit.size() > 1)
        return bad(2, "--emit", "--emit none cannot be combined with other kinds");
    if (c.emit.contains(Emit::Png) && c.grid.nz != 1)
        return bad(2, "--emit", "--emit png requires --nz 1");

    // Reference dynamic pressure x frontal area.
    const double span = comp ? (c.grid.nz > 1 ? c.grid.nz * c.h : c.grid.hz)
                             : (c.grid.nz == 1 ? c.h : c.nzInterior * c.h);
    c.qRef = 0.5 * c.uInf * c.uInf * c.placement.D * span;

    c.meta = {c.grid.nx, c.grid.ny, c.grid.nz, c.grid.hx, c.grid.hy, c.grid.hz,
              c.placement.D, c.re, comp ? c.mach : 0.0, c.fdt, c.cs, c.lesLabel,
              c.bodyKind == BodyKind::Mesh ? c.meshPath : name(c.bodyKind),
              name(c.far), name(c.physics), comp ? c.kick : 0.0};
    return c;
}

// ---------------------------------------------------------------------------

std::string Case::describe() const
{
    char buf[2048];
    std::snprintf(buf, sizeof buf,
        "case:\n"
        "  physics    %s\n"
        "  body       %s%s\n"
        "  far field  %s%s\n"
        "  grid       %d x %d x %d  h=%.6g  L=%.6g%s\n"
        "  placement  cx=%.6g cy=%.6g cz=%.6g D=%.6g (%.1f cells)\n"
        "  Re=%.6g  nu=%.6g  Mach=%.6g  cfl=%.6g  Cs=%.6g (%s)\n"
        "  ft=%.6g  tEnd=%.6g  fdt=%.6g  qRef=%.6g\n"
        "  mode       %s\n"
        "  out        %s\n",
        name(physics), name(bodyKind), meshPath.empty() ? "" : (" " + meshPath).c_str(),
        name(far),
        open ? (farPot ? " (lateral open, potential-corrected)"
                       : " (lateral open)")
             : (farPot ? " (potential-corrected)" : ""),
        grid.nx, grid.ny, grid.nz, h, L, grid.nz == 1 ? "  [native 2D]" : "",
        placement.cx, placement.cy, placement.cz, placement.D,
        h > 0 ? placement.D / h : 0.0,
        re, nu, mach, cfl, cs, lesLabel ? "LES" : "DNS",
        ft, tEnd, fdt, qRef,
        steadyTol > 0 ? "stationary (stop at steady residual, tEnd = give-up cap)"
                      : "time-march to tEnd",
        outPrefix.c_str());
    return buf;
}

} // namespace dns
