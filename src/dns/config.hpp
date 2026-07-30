// RawConfig: what the parser produced, still in user units. Case: what the
// solver may see -- fully resolved, every cross-field combination checked.
// The ONLY way to obtain a Case is validate(); its constructor is private.
#pragma once

#include "dns/grid.hpp"
#include "dns/rotor.hpp"
#include "dns/runmeta.hpp"
#include "dns/sgs.hpp"

#include <expected>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dns {

enum class Subcommand { Flow, Tgv, Hit, Bench, Channel, Conv };
enum class Physics { Incompressible, Compressible };
enum class FarField { Tunnel, Fringe, Sponge, Nscbc, Periodic };
enum class Backend { Cpu, Gpu, Auto };
enum class BodyKind { None, Cylinder, Sphere, Cube, Lens, Fence, Floor, Mesh, Scene };
// Which coefficient multiplies the Yoo-Im transverse term in the NSCBC
// outflow's incoming acoustic amplitude. YooIm = the fixed --nscbc-beta (the
// theta = 0 value of the exact coefficient); the other two derive it from the
// data, per face cell or per transverse DCT mode. See docs/nscbc.md.
enum class NscbcAbc { YooIm, Adaptive, Modal };

// The ONE spelling of each of these enums, indexed BY the enum value. The CLI
// parser, the resolved-case printer, the --body error text and the editor's
// pickers all read these rather than restating them -- a body kind added to
// the enum and the table is then complete everywhere, which four hand-kept
// copies (one of them a hardcoded count) were not.
std::span<const char* const> bodyKindNames();
std::span<const char* const> farFieldNames();
const char* name(BodyKind);
const char* name(FarField);
// The inverse of name(BodyKind); nullopt when `s` is not a body keyword (a
// mesh path, say, which the caller recognises by suffix instead).
std::optional<BodyKind> bodyKindFromName(std::string_view s);

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

    // thermal (compressible). tinf is the DIMENSIONAL freestream static
    // temperature; it exists only to give Sutherland's 110.4 K a scale, so a
    // constant-viscosity run never reads it. visc: 0 constant, 1 Sutherland.
    double tinf = 288.15;
    int visc = 0;
    double prandtl = 0.72;
    double twall = 1.0;        // body wall temperature, as a ratio to tinf
    bool nscbcTinflow = false; // NSCBC entropy inflow holds T, not rho

    // far field
    FarField far = FarField::Tunnel;
    bool open = false;
    int spongeCells = 0, fringeAxes = 0;
    double nscbcBeta = 0.5;
    // Where that coefficient comes from: fixed, per face cell, or per
    // transverse DCT mode of the outflow plane. The derived forms read the
    // face's own incidence angle off Tp/Lout and scale nscbcBeta by
    // 2/(2 + Tp/Lout), which reproduces the exact incoming amplitude at that
    // angle. Normal incidence is bit-for-bit the constant form.
    NscbcAbc nscbcAbc = NscbcAbc::YooIm;
    // Inflow relaxation stiffness. 100 is sized against a body's blockage sag
    // and is an ACOUSTIC MIRROR (measured: it returns 81% of a normal wave's
    // amplitude, and 11% of a radial pulse's energy in an open box against
    // 0.04% at 0.5). A run with no freestream to sag -- a static rotor, an
    // acoustic study -- wants it soft.
    double nscbcEtaIn = 100.0;
    int nscbcStrip = 0;
    // Scale-selective absorbing band inside the open faces (gpu/comp_sink):
    // an explicit sink for the sub-6dx content the central flux cannot
    // transport to a face (docs/nscbc.md, "Small scales and the strip
    // hand-off"). Unlike the strip it carries no freestream reference, so
    // resolved scales and the displacement flow pass through. 0 = off,
    // bit-for-bit the bare characteristic boundary.
    int nscbcSink = 0;
    double nscbcSinkAmp = 1.0; // sigma_max = amp c/h; SSP-RK3 caps amp at 2
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

    // Rotor (compressible flow): the body spins about a fixed shaft through
    // the placement centre. rotorMach is the TIP Mach and is the ONE switch --
    // 0 leaves every path bit-for-bit the static solver. rotorSpan is the
    // blade's half-extent along the shaft in diameters, which with the tip
    // radius bounds the SWEPT volume without reading the mesh. bladeThicken
    // dilates the body's sdf, in FINE cells; it changes the geometry and is
    // reported as such.
    double rotorMach = 0;
    int rotorAxis = 0;        // 0 x, 1 y, 2 z
    int rotorBins = 0;        // 0 = the natural count, 2 pi R / h_fine
    double rotorSpan = 0.25;  // half-extent along the shaft, in diameters
    double bladeThicken = 0;  // sdf dilation, in fine cells

    // Body-fitted O-grid (compressible): the mesh's inner boundary IS the
    // body surface, so there is no immersed body and no cut cell. --n is the
    // angular cell count, --ny the radial one (two of which are ghost rows).
    bool ogrid = false;
    double ogridFar = 20.0;  // far-boundary radius, in diameters
    double ogridBeta = 3.0;  // radial tanh clustering toward the wall

    // start-up (compressible)
    double ramp = 2.0, kick = 0.05;

    // solver
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
    bool tempFrames = false; // --temp: add a T - Tinf frame channel

    // hit
    double urms = 60.0, spin = 10.0, stats = 10.0;

    // channel (wall units: delta = 1, uTau = 1, f = 1, nu = 1/reTau)
    double reTau = 180;    // friction Reynolds number
    double tt = 30;        // eddy turnovers (delta/uTau = 1) to integrate
    double lx = 0;         // box length; 0 = 4 pi delta (KMM)
    double lzBox = 0;      // box span;   0 = 4 pi/3 delta (KMM)
    double profileDt = 1.0; // y-profile dump interval; 0 = off
    int wallIso = 1;       // 1 isothermal (default), 0 adiabatic

    // conv (Rayleigh-Benard: heated floor, cooled lid, gravity in -y)
    double rayleigh = 1e6;  // Rayleigh number on the SUPERADIABATIC drop
    double dtemp = 0.1;     // imposed (Tbot - Ttop) / Tref
    double convAspect = 2;  // Lx / H = Lz / H
    double hotPatch = 1.0;  // heated side fraction of the floor; 1 = all of it
    int sides = 0;          // 0 periodic lateral faces, 1 closed (--sides)

    // bench
    int steps = 20;
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

    Grid grid{1, 1, 1, 1, 1, 1};
    Placement placement;   // body centre + diameter, ghost offset included
    std::string meshPath;
    std::string scenePath;   // BodyKind::Scene: the scene JSON (loaded by main)

    int n = 0, nzInterior = 1;
    double h = 0, L = 0, uInf = 1.0;
    double re = 0, mach = 0, nu = 0, cfl = 0, tol = 0, ft = 0, tEnd = 0;
    double steadyTol = 0;   // 0 = time-march to tEnd; >0 = stationary solve
    int seqLevels = 1;      // grid-sequencing levels (1 = off)
    double cs = 0, dr = 0, aspect = 0;
    SgsModel sgsModel = SgsModel::Smagorinsky;

    // Thermal model (compressible). tinfK anchors Sutherland's dimensional
    // constant; sutherland selects mu(T), kappa(T) = mu(T) cp / Pr. twallRatio
    // is Twall / Tinf and is read only when bodyIsothermal. See
    // dns/transport.hpp for what box-unit temperature means here.
    double tinfK = 288.15, prandtl = 0.72, twallRatio = 1.0;
    bool sutherland = false, bodyIsothermal = false, nscbcThermalInflow = false;
    double penalize = 0;
    double ramp = 0, kick = 0, nscbcBeta = 0, nscbcEtaIn = 0, nscbcSinkAmp = 0;
    NscbcAbc nscbcAbc = NscbcAbc::YooIm;
    double scale = 0, pitch = 0, yaw = 0;
    int dimples = 0, bumps = 0, maxIters = 0, dtRecompute = 1;
    double bumpH = 8;   // spike height in cells
    int spongeCells = 0, fringeAxes = 0, nscbcStrip = 0, nscbcSink = 0, planes = 3;
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

    // Rotor motion (compressible flow). `rotor.active()` is the ONE test for
    // "this run has moving geometry"; it is false unless --rotor-mach was
    // given, and every geometry, wall and tally path then takes the static
    // arithmetic it always took. rotorTipMach and rotorSpanD are carried for
    // reporting; the resolved rate, bins, tip radius and dilation live in the
    // Rotor itself.
    Rotor rotor;
    double rotorTipMach = 0;
    double rotorSpanD = 0;     // blade half-extent along the shaft, in D
    double bladeThickenCells = 0;

    // Body-fitted O-grid (dns/ogrid.hpp). ogridRFar is resolved to WORLD
    // units; the grid is (n angular) x (ny radial, incl. 2 ghost rows) x nz,
    // and its reference box comes from ogridGrid(). Mutually exclusive with
    // every immersed-body knob -- the wall is a mesh face here.
    bool ogrid = false;
    double ogridRFar = 0, ogridBeta = 0, ogridSpan = 0;

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

    // conv (Rayleigh-Benard; see the validate() conv branch for the full
    // derivation). Box units: H = 1, Tref = Rgas = rho_ref = 1, so p_ref = 1
    // and the sound speed at the reference state is sqrt(gamma).
    //
    // The layer is stratified, so gravity itself carries part of the imposed
    // temperature drop: only the SUPERADIABATIC excess dTsuper = dT - (g/cp) H
    // is available to drive convection (Schwarzschild). `rayleigh` is defined
    // on dTsuper -- so its critical value is the textbook 1708 -- while the
    // Nusselt numbers the runner reports are normalised on the FULL dT, which
    // is what makes quiescent conduction read exactly Nu = 1.
    double rayleigh = 0;   // Ra on dTsuper
    double dtemp = 0;      // imposed dT / Tref
    double dtempSuper = 0; // dT - (g/cp) H, > 0 (validator-enforced)
    double gravity = 0;    // magnitude; the vector is (0, -gravity, 0)
    double convH = 1;      // layer height, wall face to wall face
    double tBot = 0, tTop = 0;  // isothermal wall temperatures
    double uFreeFall = 0;  // sqrt(gravity * dTsuper / Tref * H)
    double convAspect = 0; // Lx / H = Lz / H
    // Side fraction of a CENTRED hot patch on the floor, in (0,1]. Outside it
    // the floor sits at the lid temperature. 1 = the uniformly heated plate
    // and is bit-for-bit the case without the flag. Below 1 the reported
    // nuBot is a net floor-area-averaged flux, not a classical Nusselt number:
    // the cold surround absorbs heat, so the Ra-Nu scaling does not apply.
    double hotPatch = 1.0;
    // --sides closed: the lateral faces are no-slip ADIABATIC walls instead of
    // periodic, carried by the same one-cell ghost ring as the y plates. So
    // the grid gains two cells in x (and in z when nz > 1, never at nz == 1 --
    // the 2D restriction keeps z periodic) while the FLUID extent stays
    // convAspect * H, and `nxInterior`/`nzInterior` are the fluid counts.
    // Both physics carry it; false is bit-for-bit the periodic solver.
    bool closedSides = false;
    int nxInterior = 0;    // fluid cells in x (grid.nx minus the ghost ring)
    // Boussinesq branch (--physics incompressible) only. Free-fall scaling
    // pins H = dT = g*alpha = 1, so nu = sqrt(Pr/Ra) and kappa = 1/sqrt(Ra Pr)
    // and the free-fall time is exactly 1. There is no Mach number and no
    // stratification here, so dtempSuper == dtemp and gravity is not a knob.
    double kappaDiff = 0;  // thermal diffusivity

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
    bool tempFrames = false; // --temp: frames carry a third channel, T - Tinf
                             // on the diverging map. Off by default: it
                             // changes what every artefact of the run holds.

    RunMeta meta;

    // Non-fatal advisories raised by validate(). CARRIED, not printed:
    // validate() is specified pure (no filesystem, no GPU), and a library
    // function writing to stderr is exactly the kind of hidden side effect
    // that makes it untestable. main() prints these after phase 2.
    std::vector<std::string> warnings;

    // Human-readable dump for --dry-run.
    std::string describe() const;

private:
    Case() = default;
    friend std::expected<Case, ConfigError> validate(const RawConfig&);
};

struct Scene;   // dns/scene.hpp

// Merges the scene's model selection into the RawConfig before validate():
//   model les + no --les       -> raw.les = cs
//   model les + agreeing --les -> keep
//   any conflict               -> config error (strict, never silently
//                                 resolved)
// Config-layer, not scene-layer: this MUTATES the parse layer, so it belongs
// on the same side of the boundary as RawConfig.
std::expected<void, ConfigError> applySceneModel(const Scene&, RawConfig&);

// Pure: no filesystem, no HIP, no allocation beyond the Case.
std::expected<Case, ConfigError> validate(const RawConfig&);

} // namespace dns
