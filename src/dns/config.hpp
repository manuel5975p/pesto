// RawConfig: what the parser produced, still in user units. Case: what the
// solver may see -- fully resolved, every cross-field combination checked.
// The ONLY way to obtain a Case is validate(); its constructor is private.
#pragma once

#include "dns/grid.hpp"

#include <expected>
#include <set>
#include <string>

namespace dns {

enum class Subcommand { Flow, Tgv, Hit, Bench };
enum class Physics { Incompressible, Compressible };
enum class FarField { Tunnel, Fringe, Sponge, Nscbc, Periodic };
enum class Backend { Cpu, Gpu, Auto };
enum class PoissonKind { Auto, Fft, Dct, Cg };
enum class BodyKind { None, Cylinder, Sphere, Cube, Lens, Mesh };
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
    double aspect = 3.5;
    int dimples = 0;
    double scale = 0.0, pitch = 0.0, yaw = 0.0;
    bool robust = false;

    // physics
    Physics physics = Physics::Incompressible;
    double re = 60.0, mach = 0.2, les = 0.0, cfl = 0.0, nu = 0.0;

    // far field
    FarField far = FarField::Tunnel;
    bool open = false;
    int spongeCells = 0, fringeAxes = 0;
    double nscbcBeta = 0.5;
    int nscbcStrip = 0;

    // start-up (compressible)
    double ramp = 2.0, kick = 0.05;

    // solver
    PoissonKind poisson = PoissonKind::Auto;
    double tol = 1e-4;
    int maxIters = 500, dtRecompute = 0;

    // run length
    double ft = 0.0;

    // output
    std::set<Emit> emit;
    double fdt = 0.15, pdt = 0.0, wdt = 0.0;
    int planes = 3;

    // hit
    double urms = 60.0, spin = 10.0, stats = 10.0;

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

    int n = 0, nzInterior = 1;
    double h = 0, L = 0, uInf = 1.0;
    double re = 0, mach = 0, nu = 0, cfl = 0, tol = 0, ft = 0, tEnd = 0;
    double cs = 0, qRef = 0, dr = 0, aspect = 0;
    double ramp = 0, kick = 0, nscbcBeta = 0;
    double scale = 0, pitch = 0, yaw = 0;
    int dimples = 0, maxIters = 0, dtRecompute = 1;
    int spongeCells = 0, fringeAxes = 0, nscbcStrip = 0, planes = 3;
    int steps = 0;
    unsigned seed = 1;
    bool open = false, robust = false, lesLabel = false, quiet = false, dryRun = false;

    // hit (SI inputs)
    double urms = 0, spin = 0, stats = 0;

    std::string outPrefix;
    std::set<Emit> emit;
    double fdt = 0, pdt = 0, wdt = 0;

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
