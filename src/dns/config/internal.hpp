// validate()'s internals, shared by the src/dns/config/*.cpp files and by
// nothing else: the validator idioms, the per-subcommand halves of validate()
// and the --dry-run helpers one file defines and another reads. Each
// per-subcommand half takes the Case the shared prologue (validate.cpp) has
// filled in and finishes it, so no subcommand's rules are visible to another's.
// Never included outside src/dns/config/.
#pragma once

#include "dns/config.hpp"

#include <cstddef>
#include <expected>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>

namespace dns::cfg {

// --- common.cpp --------------------------------------------------------------

// The --dry-run tail of a --compensated time march: it changes the bits, so
// the resolved case says so.
inline constexpr const char* kCompensatedNote =
    " [compensated state: FP32 hi + lo per field]";

bool finitePos(double v);

bool hasSuffix(std::string_view s, std::string_view suf);

std::unexpected<ConfigError> bad(int code, const char* field, std::string msg);

// The run asks for a remapped (curvilinear) Cartesian mesh: the ONE spelling
// of "not the uniform mesh" that the sink default, the explicit-sink rejection
// and the modal closure all test. The default and the rejections stay separate
// decisions (a default that does not apply is absence; an explicit flag that
// does not apply is an error) -- only the predicate is shared.
bool remapsMesh(const RawConfig& r);

// The same error as a validator stage's result (std::optional<ConfigError>).
ConfigError err(int code, const char* field, std::string msg);

// The --body keywords, in one comma-separated string built from the shared
// table -- so a body kind added to the enum lists itself in both --body error
// messages. Mesh and Scene are excluded: they are kinds the validator assigns
// from a path or from --scene, not words the user may type.
std::string bodyKeywords();

// Rejects whichever of `flags` the user actually GAVE -- a flag left at its
// default is not an error, only one that was typed and cannot apply. The
// message is the flag's own name followed by `reason`, which is the shape
// every such rejection in this file uses.
//
// Returns the first offender, or nullopt when none was given:
//   if (auto e = rejectGiven(r, {"--mach"}, "is compressible-only"); e)
//       return std::unexpected(*e);
std::optional<ConfigError> rejectGiven(const RawConfig& r, std::initializer_list<const char*> flags, std::string_view reason);

// The incompressible FV stage's module set (mci, dns/fvimodules: ONE decision per pair)
// from the flags, checked: --les/--sgs, --rans, and T + gravity when thermal (dns conv).
// The refusal names the flag that brought its module. Pre: r.les finite >= 0.
std::optional<ConfigError> fvIncompModules(const RawConfig& r, bool thermal);

// Frame cadences and the two PNG shaping flags. Every subcommand that writes
// frames takes exactly these rules; they used to be spelled out once per
// branch, and the copies had already drifted apart in their wording.
// Pre: c.grid is resolved (--png-down must divide it).
std::optional<ConfigError> validateFrameOutputs(const RawConfig& r, Case& c);

// Each rule below was spelled out per subcommand; one definition each now, with
// the message text unchanged. Every helper returns the error rather than
// throwing it, so the caller's check ORDER -- first error wins -- is untouched.
// The compressible Mach range. shock: --shock hll's text; the limit is the same.
std::optional<ConfigError> checkMach(double mach, bool shock = false);

// --shock's scope, by name, exit 2. shockRouteScope: BEFORE the routes, whose
// allow-lists would otherwise answer first: the meshes and closures it is not
// built with, physics, --solver. shockFlagScope: on the route's finished Case
// (far, body, flame mode).
std::optional<ConfigError> shockRouteScope(const RawConfig& r);
std::optional<ConfigError> shockFlagScope(const RawConfig& r, const Case& c);

// --shock hll's dt cadence: the sticky rule unless --dt-recompute was given; a
// given N > 1 warns. Pre: c.dtRecompute resolved.
void shockCadence(const RawConfig& r, Case& c);

// The z cell count: 1 is the exact 2D restriction, otherwise >= 4.
// `belowOne` is the subcommand's own wording for nz < 1.
std::optional<ConfigError> checkNz(int nz, const char* belowOne);

// The int indexing the whole tree assumes.
std::optional<ConfigError> checkCells(std::size_t cells);

// The one constructor of a validated box: rejects a grid over INT_MAX cells and
// is the ONE enforcement point of `nz == 1 => hz == 1` (lz is pinned to 1).
std::expected<Grid, ConfigError> makeBox(int nx, int ny, int nz, double lx, double ly, double lz);

// --cfl, or the subcommand's own default when it was not given.
std::optional<ConfigError> resolveCfl(const RawConfig& r, Case& c, double fallback);

// --dt-recompute, or the subcommand's own default when it was not given.
std::optional<ConfigError> resolveDtRecompute(const RawConfig& r, Case& c, int fallback);

// --profile-dt: a dump interval, 0 = off.
std::optional<ConfigError> checkProfileDt(const RawConfig& r, Case& c);

// An empty --emit set takes the subcommand's own default; `none` is exclusive.
std::optional<ConfigError> resolveEmit(Case& c, std::initializer_list<Emit> fallback);

// The wall-bounded boxes (channel, conv) share their grid-count prologue and
// their run epilogue rule for rule and word for word; the physics between the
// two is theirs alone.
//
// Prologue: the lateral, wall-normal and spanwise FLUID cell counts.
std::optional<ConfigError> checkWallGrid(const RawConfig& r);

// Epilogue: --cfl, --kick, --tt (c.ft; the caller sets c.tEnd from it),
// --profile-dt, --dt-recompute (default 16), the frame cadences and --emit
// (default stats). Pre: c.grid is resolved.
std::optional<ConfigError> validateWallRun(const RawConfig& r, Case& c, double cflFallback);

// The inverse of name(BodyKind); nullopt when `s` is not a body keyword (a
// mesh path, say, which the caller recognises by suffix instead).
std::optional<BodyKind> bodyKindFromName(std::string_view s);

// --- bench.cpp ---------------------------------------------------------------

std::expected<Case, ConfigError> validateBench(const RawConfig& r, Case c);

// --- periodic.cpp ------------------------------------------------------------

std::expected<Case, ConfigError> validatePeriodic(const RawConfig& r, Case c);

// dns hit: the SI design (h = 1.5 eta) and its box-unit run, as runHit
// derives them. Pre: c validated by validatePeriodic for Subcommand::Hit.
std::string describeHit(const Case& c);

// --- channel.cpp -------------------------------------------------------------

// Wall units: delta = 1, u_tau = 1, f = 1, nu = 1/reTau. The box momentum
// balance (f * V = 2 tau_w * A_wall with Ly = 2 delta) pins tau_w = 1, so
// reTau is the run's friction Reynolds number by construction.
std::expected<Case, ConfigError> validateChannel(const RawConfig& r, Case c);

// --- conv.cpp ----------------------------------------------------------------

std::expected<Case, ConfigError> validateConv(const RawConfig& r, Case c);

// dns conv --solver fv | --polymesh (docs/fvcomp.md, docs/fvincomp.md): the
// allow-list of the route, the frame plane, the arithmetic. Off: its flags are
// rejected by name. Pre: the structured conv case is resolved.
std::optional<ConfigError> resolveFvConv(const RawConfig& r, Case& c);

// --- flame_scope.cpp ---------------------------------------------------------

// The closed 2D combustion chamber (docs/chem.md, "Chamber"; dns/chamber.hpp).
// Which flags belong to which flame mode: a flag of the other mode that was
// TYPED is an error by name, never ignored.
// lmBurner: --mode bunsen --physics lowmach, which takes --reaction too (the
// inert jet: masterplan real_bunsen S6's Bickley case).
std::optional<ConfigError> chamberFlagScope(const RawConfig& r, bool chamber, bool lmBurner = false);

// The 3D-turbulence flags exist for --mode premixed and bunsen (docs/chem.md,
// "3D and turbulence", "3D turbulent slot Bunsen"); every other mode rejects
// them by name. --hold-flame (the planar brush's inflow control) is
// premixed-only: the slot burner's jet speed is the experiment.
std::optional<ConfigError> turbFlagScope(const RawConfig& r, bool premixed, bool bunsen);

// --turb on a validated flame box (c.grid, c.flameCells set): the l_t bounds
// (>= 4 cells, <= L/6 of both periodic axes), the Mach bound on uMax + 3 u'
// (uName names uMax in the message), then c.flameTurb, flameTurbL,
// flameTurbNbx (= nz: L_bx = L_z) and the h/eta warning. Pre: r.turb > 0.
std::optional<ConfigError> flameTurbSetup(const RawConfig& r, Case& c, double uMax, const char* uName);

// LES species and the thickened flame (docs/chem.md, "LES species and
// TFLES") exist for --mode premixed and bunsen; diffusion, candle and
// chamber reject every flag of it by name.
std::optional<ConfigError> lesTflesFlagScope(const RawConfig& r, bool premixed);

// Flame LES (--les, --sgs) and TFLES (--tfles, --tfles-sensor,
// --efficiency) into c.cs/sgsModel/lesLabel and c.flameTfles*. Runs after the
// --nz rules; the resolution rule (F --cells-per-delta >= 4) and the
// dilation radius are the caller's (they need the grid). Pre: premixed or
// bunsen.
std::optional<ConfigError> flameLesTfles(const RawConfig& r, Case& c, bool bunsen);

// The low-Mach flame's module set (M4, dns/lmmodules: ONE decision per pair) from the
// flags, checked; the refusal names the flag that brought its first module. Runs after
// flameLesTfles (c.lesLabel, c.flameTfles, c.flameCharlette resolved). Pre: lowmach.
std::optional<ConfigError> lowMachModules(const RawConfig& r, const Case& c, bool premix2, bool rim);

// --frame-start (every flame mode): frames begin at T; it needs a frame kind
// to delay. Pre: c.emit and c.tEnd resolved.
std::optional<ConfigError> resolveFrameStart(const RawConfig& r, Case& c);

// --checkpoint-dt / --wall-limit / --restart (dns/checkpoint.hpp): wired in
// runFlame (premixed, bunsen, diffusion, candle), runFvComp and runFvLowMach
// (the FV routes, --solver fv | --polymesh: each proven by gpu_restart_fv);
// the frame kinds that can resume (stats, bin, volume, slices, png, shm: the
// web manifest is rebuilt per process). r.restartT is the checkpoint's t
// (main, phase 1.5).
std::optional<ConfigError> resolveCheckpoint(const RawConfig& r, Case& c);

// --- flame_modes.cpp ---------------------------------------------------------

std::expected<Case, ConfigError> finishChamber(const RawConfig& r, Case c);

// --mode bunsen's default resolution: 8 cells per delta_th (the demo grid;
// the 1D ladder puts S_L +1.7% high at 7.5 cells, docs/chem.md).
inline constexpr double kBunsenCellsDefault = 8;

// The slot burner (dns/bunsen.hpp): validates the burner flags and sets the
// box, c.bunsen (box units) and c.uInf. Lengths arrive in delta_th.
// Pre: c.flameEig, c.flameCells, c.flameModel set; nb = the sink band.
std::expected<void, ConfigError> bunsenGeometry(const RawConfig& r, Case& c, int nb);

// Which flags belong to the wick candle: typed elsewhere = an error by name.
std::optional<ConfigError> wickFlagScope(const RawConfig& r, bool candle);

// The 2D slot-jet diffusion flame (docs/chem.md, dns/diffusion.hpp): model
// C1, the burner on the bunsen flags (lengths in l, speeds in U_jet), the
// box, the label inputs. Every premixed-only flag is rejected by name.
// --mode candle is the same model and burner under gravity (-x) with the
// candle defaults above.
std::expected<Case, ConfigError> validateDiffusion(const RawConfig& r, Case c);

// --- flame.cpp ---------------------------------------------------------------

std::expected<Case, ConfigError> validateFlame(const RawConfig& r, Case c);

// --- flow.cpp ----------------------------------------------------------------

// validateFlow() below runs these stages in order; each one returns the first
// error of its own checks.
// Stage: the body (--body/--scene), the far field it may have, and the
// wall-mounted bodies' rules.
std::optional<ConfigError> flowBody(const RawConfig& r, Case& c);

// Stage: the grid: the two physics' box conventions and the body's minimum resolution.
std::optional<ConfigError> flowGrid(const RawConfig& r, Case& c);

// Stage: Mach, viscosity, CFL.
std::optional<ConfigError> flowPhysics(const RawConfig& r, Case& c);

// Stage: the tanh y-stretch.
std::optional<ConfigError> flowStretch(const RawConfig& r, Case& c);

// The `dns flow` validator, as ordered stages over one Case. ORDER IS
// LOAD-BEARING: the first error wins, so a stage may only move together with
// every check before it. Each stage reads what the earlier ones resolved.
std::expected<Case, ConfigError> validateFlow(const RawConfig& r, Case c);

// --- flow_mesh.cpp -----------------------------------------------------------

// Stage: the --focal plateau.
std::optional<ConfigError> flowFocal(const RawConfig& r, Case& c);

// Stage: the spinning body.
std::optional<ConfigError> flowRotor(const RawConfig& r, Case& c);

// --- flow_rans.cpp -----------------------------------------------------------

// The freestream turbulence as the describe line and RunMeta print it: the
// flags as given, and omega in U/D. Pre: c.rans != None, ransPhiInf resolved.
std::string ransInflowText(const Case& c);

// The first-cell wall distance the y+ estimate uses: the O-grid's h1, or half
// the fine spacing of an immersed mesh (the mean centre offset of a
// staircase wall).
double ransFirstCellDistance(const Case& c);

// The first-cell y+ estimate the validator can quote without a run: the
// bluff-body PEAK friction velocity u_tau ~ 0.38 Re^-0.1 U, calibrated on the
// MEASURED 0.137 U of the OpenFOAM sphere at Re 1e4 (masterplan_flowrans
// section 3.6). Pre: h1, D, re > 0.
double ransYplusEstimate(double h1, double D, double re);

// Stage: RANS and the compressible steady march (docs/rans.md, "dns flow").
// LAST, so every geometry/physics error wins first; every rule rejects by
// name, never ignores. RANS off and no compressible --steady leave the case
// exactly as the earlier stages built it.
std::optional<ConfigError> flowRans(const RawConfig& r, Case& c);

// --- prop.cpp ----------------------------------------------------------------

std::expected<Case, ConfigError> validateProp(const RawConfig& r, Case c);

// The resolved prop case for --dry-run. Names the method's model content the
// way the DNS/LES label is named: what a curve came from must be visible.
std::string describeProp(const prop::Case& p);

// --- fv.cpp ------------------------------------------------------------------

// dns flow --polymesh CASE (docs/fv.md): the case IS the OpenFOAM case
// (phase 1.5 / 3 read it); here only the flags. Allowed: --polymesh,
// --physics incompressible, --out, --max-iters, --emit stats|none,
// --fv-precision, --backend, --quiet, --dry-run. Every other flag is
// rejected by name.
std::expected<Case, ConfigError> validateFvFlow(const RawConfig& r, Case c);

// The FV family's frame window and arithmetic, and the --emit kinds it writes.
std::optional<ConfigError> fvCompCommon(const RawConfig& r, Case& c);

// dns flow --physics compressible --polymesh DIR --bc FILE (docs/fvcomp.md): the
// mesh and its patches are the case; every flag not listed is rejected by name.
std::expected<Case, ConfigError> validateFvCompMesh(const RawConfig& r, Case c);

// --solver fv --physics compressible on a structured case, BEFORE validateFlow:
// the mesh routes through FaceMesh, so only what FV is built for passes; rejected by name, never ignored.
std::optional<ConfigError> checkFvStructured(const RawConfig& r);

// The time-accurate incompressible FV solver (docs/fvincomp.md). An allowlist:
// every other flag is rejected by name.
std::expected<Case, ConfigError> validateFvIncompFlow(const RawConfig& r, Case c);

// dns flow --solver simple (docs/fvstructured.md): the structured mesh stages
// on their compressible box conventions (the curvilinear meshes live there),
// then the generated case. Only flags with a simpleFoam meaning pass.
std::expected<Case, ConfigError> validateFvStructured(const RawConfig& r0, Case c);

// --- labels.cpp --------------------------------------------------------------

std::string describeWick(const Case& c);

std::string describeChamber(const Case& c);

// --mode bunsen --physics lowmach (masterplan real_bunsen S5).
std::string describeLowMachBunsen(const Case& c);

} // namespace dns::cfg
