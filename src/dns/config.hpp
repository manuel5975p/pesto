// RawConfig: what the parser produced, still in user units. Case: what the
// solver may see -- fully resolved, every cross-field combination checked.
// The ONLY way to obtain a Case is validate(); its constructor is private.
#pragma once

#include "dns/grid.hpp"
#include "dns/sgs.hpp"

#include <expected>
#include <set>
#include <string>

namespace dns {

enum class Subcommand { Flow, Tgv, Hit, Bench, Channel };
enum class Physics { Incompressible, Compressible };
enum class FarField { Tunnel, Fringe, Sponge, Nscbc, Periodic };
enum class Backend { Cpu, Gpu, Auto };
enum class PoissonKind { Auto, Fft, Dct, Cg };
enum class BodyKind { None, Cylinder, Sphere, Cube, Lens, Fence, Floor, Mesh, Scene };
enum class Emit { Stats, Bin, Png, Web, Volume, Slices, Vtk, Error, Spectrum, Snapshot, None };

// One field per CLI flag. `given` records which flags the user actually typed,
// so the validator can distinguish "left at the default" from "set to the
// default" (--ny follows --n, --pdt follows --fdt, ...).
struct RawConfig
{
    Subcommand sub = Subcommand::Flow;
    std::set<std::string> given;

    // common
    Backend backend = Backend::Auto;
    std::string out;
    unsigned seed = 1;
    bool quiet = false;
    bool dryRun = false;

    // grid
    int n = 512, ny = 0, nz = 1;
    double dr = 8.0;

    // body
    std::string body;   // keyword or mesh path
    std::string scene;  // scene JSON path (--scene); exclusive with --body
    double aspect = 3.5;
    int dimples = 0;
    int bumps = 0;      // floor trip spikes along the plate; 0 = smooth
    double bumpH = 8;   // spike height in CELLS (base = height, 63-deg flanks)
    double scale = 0.0, pitch = 0.0, yaw = 0.0;
    bool robust = false;

    // physics
    Physics physics = Physics::Incompressible;
    double re = 60.0, mach = 0.2, les = 0.0, cfl = 0.0, nu = 0.0;
    SgsModel sgs = SgsModel::Smagorinsky;   // which nut(grad u), used iff les > 0

    // far field
    FarField far = FarField::Tunnel;
    bool open = false;
    int spongeCells = 0, fringeAxes = 0;
    double nscbcBeta = 0.5;
    int nscbcStrip = 0;
    bool farPot = false;

    // body discretization
    bool cutFlux = false;  // apertured momentum fluxes (sharp cut-cell wall)
    double penalize = 0;   // Brinkman band decay time eta; 0 = off

    // mesh (compressible curvilinear)
    double stretch = 0;    // tanh wall-normal clustering along y; 0 = uniform
    bool noslip = false;   // closed y walls no-slip instead of free-slip
    // Focal clustering: a fine plateau box, coarse far field. "auto" derives
    // the box from the body; otherwise "x0,x1,y0,y1,z0,z1" in world units
    // (a degenerate pair leaves that axis uniform).
    std::string focal;
    double focalRatio = 4.0;  // coarse/fine cell-size ratio R
    double focalBlend = 0.5;  // blend width per side, as a fraction of the
                              // plateau extent on that axis
    double focalWake = 5.0;   // auto: extra plateau downstream, in diameters
    double focalPad = 0.25;   // auto: AABB inflation, in diameters

    // start-up (compressible)
    double ramp = 2.0, kick = 0.05;

    // solver
    PoissonKind poisson = PoissonKind::Auto;
    double tol = 1e-4;
    int maxIters = 500, dtRecompute = 0;

    // run length
    double ft = 0.0;
    double steady = 0.0;   // >0: stop at this steady residual instead of tEnd
    int seq = 1;           // grid-sequencing levels; 1 = solve on --n directly

    // output
    std::set<Emit> emit;
    double fdt = 0.15, pdt = 0.0, wdt = 0.0;
    int planes = 3;
    int pngDown = 1;   // PNG output box-average coarsening factor; 1 = full res
    double vortMax = 0;      // vorticity/|curl| colour clip in uInf/D; 0 = auto
    bool vortSigned = false; // 3D compressible: signed wz + diverging map
                             // instead of |curl| + sequential
    int vortMap = 0;         // 0 linear clip, 1 asinh (png display only)
    int speedMap = 0;        // 0 |u| viridis, 1 delta: |u|-uInf diverging

    // hit
    double urms = 60.0, spin = 10.0, stats = 10.0;

    // channel (wall units: delta = 1, uTau = 1, f = 1, nu = 1/reTau)
    double reTau = 180;    // friction Reynolds number
    double tt = 30;        // eddy turnovers (delta/uTau = 1) to integrate
    double lx = 0;         // box length; 0 = 4 pi delta (KMM)
    double lzBox = 0;      // box span;   0 = 4 pi/3 delta (KMM)
    double profileDt = 1.0; // y-profile dump interval; 0 = off
    int wallIso = 1;       // 1 isothermal (default), 0 adiabatic

    // bench
    int steps = 20;
};

// code: the process exit code this error maps to (1 CLI, 2 config, ...).
struct ConfigError
{
    int code = 2;
    std::string field, message;
};

std::string format(const ConfigError&);

// Metadata carried into the run's .json / manifest.
struct RunMeta
{
    int nx = 0, ny = 0, nz = 0;
    double hx = 0, hy = 0, hz = 0;
    double D = 0, re = 0, mach = 0, frameDt = 0, cs = 0;
    bool les = false;
    std::string body, farField, physics;
    double kick = 0;
    std::string sgs;   // "smag" | "sigma" when les, "" for DNS
};

// Body centre and diameter in grid (world) coordinates, ghost offset included.
struct Placement
{
    double cx = 0, cy = 0, cz = 0, D = 0;
};

// Fully resolved case. Every derived quantity is computed here, so downstream
// asserts are legitimate BECAUSE validate() ran.
class Case
{
public:
    Subcommand sub = Subcommand::Flow;
    Backend backend = Backend::Auto;
    Physics physics = Physics::Incompressible;
    FarField far = FarField::Tunnel;
    BodyKind bodyKind = BodyKind::None;
    PoissonKind poisson = PoissonKind::Auto;

    Grid grid{1, 1, 1, 1, 1, 1};
    Placement body;
    Placement placement;   // alias kept for readability at call sites
    std::string meshPath;
    std::string scenePath;   // BodyKind::Scene: the scene JSON (loaded by main)

    int n = 0, nzInterior = 1;
    double h = 0, L = 0, uInf = 1.0;
    double re = 0, mach = 0, nu = 0, cfl = 0, tol = 0, ft = 0, tEnd = 0;
    double steadyTol = 0;   // 0 = time-march to tEnd; >0 = stationary solve
    int seqLevels = 1;      // grid-sequencing levels (1 = off)
    double cs = 0, qRef = 0, dr = 0, aspect = 0;
    SgsModel sgsModel = SgsModel::Smagorinsky;
    double penalize = 0;
    double ramp = 0, kick = 0, nscbcBeta = 0;
    double scale = 0, pitch = 0, yaw = 0;
    int dimples = 0, bumps = 0, maxIters = 0, dtRecompute = 1;
    double bumpH = 8;   // spike height in cells
    int spongeCells = 0, fringeAxes = 0, nscbcStrip = 0, planes = 3;
    int steps = 0;
    unsigned seed = 1;
    bool open = false, robust = false, lesLabel = false, quiet = false, dryRun = false;
    bool farPot = false, cutFlux = false;

    // Curvilinear y-stretch (compressible only). stretchH1/stretchGrowth are
    // the resolved first-interior-cell height and max cell-to-cell growth
    // ratio, surfaced by --dry-run because they set the acoustic dt.
    double stretchBeta = 0, stretchH1 = 0, stretchGrowth = 1;
    bool noslipY = false;

    // Focal clustering (compressible only, mutually exclusive with --stretch).
    // focalActive: at least one axis clusters. The resolved plateau [lo,hi],
    // blend width and ratio per axis are what runner_comp turns into the
    // AxisRemap3; focalHFine/focalGrowth are the diagnostics --dry-run prints
    // because the fine spacing sets the acoustic dt.
    bool focalActive = false;
    double focalLo[3] = {0, 0, 0}, focalHi[3] = {0, 0, 0}, focalW[3] = {0, 0, 0};
    double focalRatio = 1;
    double focalHFine[3] = {0, 0, 0}, focalHCoarse[3] = {0, 0, 0};
    double focalGrowth = 1;   // max adjacent cell-to-cell growth ratio
    double focalDtFactor = 1; // hmin_focal / hmin_uniform (the dt penalty)

    // hit (SI inputs)
    double urms = 0, spin = 0, stats = 0;

    // channel (wall units; see the validate() channel branch)
    double reTau = 0;      // friction Reynolds number (nu = 1/reTau)
    double ubEst = 0;      // log-law bulk-velocity estimate (sets p via Mach)
    double pRef = 0;       // background pressure; Twall = pRef (rho = R = 1)
    double chanH1 = 0;     // resolved first fluid cell height
    double profileDt = 0;  // y-profile dump interval; 0 = off
    double lxBox = 0, lzBox = 0;
    bool wallIso = true;   // isothermal walls (adiabatic heats unboundedly)

    std::string outPrefix;
    std::set<Emit> emit;
    double fdt = 0, pdt = 0, wdt = 0;
    int pngDown = 1;   // PNG output box-average coarsening factor
    double vortMax = 0; // vorticity/|curl| colour clip, uInf/D units; 0 = auto
                        // (8 signed / 15 magnitude -- calibrated at Re 60-200,
                        // far too low for high-Re runs: everything clips)
    bool vortSigned = false; // 3D comp frames: signed wz (diverging, neutral
                             // background) instead of |curl| (sequential)
    bool vortAsinh = false;  // png vorticity: asinh(v/toe), toe = vortMax/32
                             // (display only; bin/web stay raw)
    bool speedDelta = false; // frames carry dspeed = |u| - uInf (diverging,
                             // freestream = background) instead of |u|

    RunMeta meta;

    // Human-readable dump for --dry-run.
    std::string describe() const;

private:
    Case() = default;
    friend std::expected<Case, ConfigError> validate(const RawConfig&);
};

// Pure: no filesystem, no HIP, no allocation beyond the Case.
std::expected<Case, ConfigError> validate(const RawConfig&);

} // namespace dns
