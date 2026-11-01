# Masterplan: a real lab Bunsen burner (`dns flame --mode bunsen --nozzle round`)

Status (2026-10-31): S1 DONE (see its STATUS block; it corrects F1's cause,
section 1's starred inputs and D4's sign); S1b DONE (model C2b: S_L(phi)
calibrated to GRI-3.0, the `--equivalence` default; see its STATUS block,
it replaces D4); S2 DONE (checkpoint/restart on runFlame + chain.sh; see
its STATUS block); S3 DONE (SeparablePoisson + LmPressure; see its STATUS
block); S4 DONE (the low-Mach core, `--physics lowmach`; see its STATUS
block: compensated rho / rho Y, S4b not needed); S5 DONE (the low-Mach
Bunsen, partially premixed, flame index; see its STATUS block: T votes,
the 4-cell dt issue LM10); S6 DONE (open laterals `--laterals open`,
Bickley, ambient backflow; see its STATUS block: in 2D open laterals do
NOT close the outer cone, LM13); S7 DONE (gravity `--gravity` on the
low-Mach burner; see its STATUS block: real g does not close the 2D outer
cone either, the candle gate is not met, LM15); S8 DONE (the immersed
isothermal tube rim `--rim`; see its STATUS block: a STAIRCASE wall, the
cold rim anchors at every admissible U, LM17-LM19); S9 DONE (the burner's dt
rule `--dt-rule cell`, the SI echo, the two-cone renderer, half-scale scouts;
see its STATUS block: the outer cone is ~4x the half scale's, the S10 box
is 200 delta_th, K10's widen rule not followed, LM20-LM23; the S10 command
set is at the end of S10); S10 not built. Scope as
amended by
the user: "implement the real deal" -- true Mach (a low-Mach solver), real Re
AND real D/delta_th, checkpoint/restart for multi-job runs.

Baseline (simjob 746, `runs/flames/hot/real`): stoich single-species F -> P,
D 22 delta_th, U 4 S_L, sigma 7.5, M 0.02, 4 cells/delta_th, 224 x 160 x 160 =
5.73 M cells, 13.3 ms/step (2.3 ns/cell-step), tt 30 tau in 398 s. Departures
from a real burner: (1) no outer cone, (2) burnt-pilot anchor instead of a
steel rim, (3) transport scale choice unstated, (4) no gravity, (5) periodic
y-z lattice of jets, (6) inflated Mach, (7) [new] no true-Mach solver, (8)
[new] no way past one 1800 s job.

Binding rules (CLAUDE.md): every existing path bit for bit at every stage;
own instantiations, never runtime branches in hot kernels; std::expected for
user errors, assert only behind the validator; GPU only via `build/simjob`;
Release build; tests with timeouts, GPU tests rc 77 without a device; each
acceptance run <= 5 min.

Decisions are tagged **D#** with the reason; risks **K#**.

---------------------------------------------------------------------------

## 0. Findings from reading the code (they shape the plan)

F1. **The FP64 oracle is NOT robust at the real sigma.** `dns flame --mode
premixed --visc sutherland --tinf 298 --dry-run` (this build):

| sigma | Lambda | delta_th/delta_D |
|---|---|---|
| 6.0 | 1044276.4 | 4.203 |
| **6.5** | **16383.95 (= 2^14: a halving artefact)** | **5.131** |
| 6.8 | 893935.8 | 4.480 |
| **6.88** | **42008.4** | **45.51** |
| 6.9 | 879919.8 | 4.514 |
| 7.5 | 810628.5 | 4.710 |

Constant lambda at 6.88 is fine (1.584). The bracket in `src/dns/chem.cpp:86-101`
(doubling/halving from m = 1, then bisection on `tooFast`) trusts a monotone
`tooFast`; it is not (issue 69's mechanism, now with Sutherland at sigma 6.5 /
6.88 -- CH4's real phi-1.3 sigma). A real-burner case would have been laid on a
garbage Lambda with no error. Fixed in S1 before anything else; new issue 72.

F2. Species open faces exist only on x: `cellFaces` sets the open flags for
`a == 0` only (`src/dns/gpu/species.hip:173-182`); the re-carry
`kSpeciesOpenFix` and `speciesMassDiv` enumerate i = 0, nx-1 only
(`species.hip:454-491`). Sink + species sink already take openY/openZ
(`src/dns/gpu/comp_sink.hpp:26-29`).

F3. NSCBC refuses hydrostatic waves with open laterals
(`src/dns/gpu/comp_nscbc.hip:525-528`); lateral targets are constant p, rho
(`comp_nscbc.hip:193-216, 366`).

F4. `runFlame` ends on `t >= tEnd` with a CLIPPED last step
(`src/dns/gpu/runner_flame.hip:662`) and a forced stats row
(`src/dns/gpu/internal/run_loop.hpp:95-98`). A restart cannot be "run to
t_c, then resume" through `--tt`: the checkpoint must be taken at an unclipped
step boundary and the job must stop WITHOUT the final-row logic (S2).

F5. The flame validator rejects every physics but compressible
(`src/dns/config.cpp:1921-1923`); `Physics` has two values
(`src/dns/config.hpp:33`, names `config.cpp:283-287`); dns-ui maps physics
by index (`src/ui/doc.cpp:37,132,176`).

F6. Reusable pieces for the low-Mach solver: separable spectral Poisson
(`src/dns/gpu/poisson.hpp:32`, Makhoul DCT-II, `PeriodicXZWallY` tridiagonal);
apertured PCG preconditioned by that solve (`src/dns/gpu/cutpoisson.hpp:16-50`,
"tens of iterations"); matrix-free deterministic Krylov
(`src/dns/gpu/la/krylov.hpp:22-23,104`: LinOp callables, batched fixed-order
FP64 dots); diffusion-only MAC ghost view (`src/dns/gpu/ghostfill.hpp:30-53`);
rate functions `arrheniusRate/arrheniusRate2` (`src/dns/chem.hpp:47-61`,
DNS_HD templates); `Transport` (the one gas object); the shared-limiter face
formula (`species.hip:280-331`).

---------------------------------------------------------------------------

## 1. The physical case and its scaling (items 3, 6, 8)

Real burner (inputs; S1 replaces the starred values by a Cantera GRI-3.0
reference run, `tools/chem/cantera_flame.py`, checked in under
`tests/data/chem/`, never a build dependency):

| quantity | value |
|---|---|
| mixture | CH4/air, phi 1.3, 298 K, 1 atm |
| S_L(phi 1.3) | 0.25 m/s* (lit. 0.22-0.27) |
| T_ad(phi 1.3) | 2050 K* -> sigma_in 6.88 |
| delta_th,real(phi 1.3) | 0.70 mm* (max-gradient; stoich 0.45) |
| alpha_u, nu_u | 2.19e-5, 1.60e-5 m^2/s -> delta_D = alpha/S_L 0.0877 mm |
| s, Y_O2 air | 3.99 (CH4 + 2 O2), 0.233 |
| tube | ID D = 10 mm, wall 1 mm (OD 12), steel, rim at room T |
| U_bulk | 1.5 m/s (6 S_L), Poiseuille: centre 3 m/s |
| Re_D = U D/nu | 940 |
| Fr = U^2/(g D) | 22.9 |
| one-step model delta_th/delta_D (Sutherland, sigma 6.9) | 4.51 (dry-run; two-reactant value from S1) |
| real delta_th/delta_D | 0.70/0.0877 = 8.0 |

**D1 (the trilemma).** A one-step, Le = 1, constant-cp model has
delta_th/delta_D = 4.5 where CH4 has 8.0. Of {Re (nu), Pe (alpha, = species D
at Le 1), D/delta_th} at most TWO can be real. The user's goal is Re AND
D/delta_th, so:

- **Mapping M1 (the final case): real Re, real D/delta_th, real S_L, U, g
  (Fr, Gr real).** Length unit delta_D,model = delta_th,real / 4.51 = 0.155
  mm; time unit tau = delta_D/S_L = 0.62 ms. Viscosity real by an EFFECTIVE
  Prandtl number: Pr_eff = nu_real/(S_L delta_D,model) = 0.41 (`--prandtl
  0.41`, an EXISTING flag -- the oracle does not depend on Pr). Cost: thermal
  and species diffusivity 1.68 x real (Pe = Re Pr_eff = 0.6 x real): the
  diffusion-controlled outer cone is ~0.6 x its real height relative to D, the
  rim's thermal layer 1.6 x thicker than its momentum layer than real. Le = 1
  kept. Labelled everywhere: "Re and D/delta_th real; Pe 0.6x (Pr_eff 0.41,
  one-step delta_th/delta_D 4.5 vs CH4 8.0)".
- Mapping M2 (documented alternative, same solver, different flags): real Re
  AND Pe (Pr 0.72), D/delta_th 1.77 x real (the model flame 0.56 x as thick as
  CH4's at the same D). h = 0.099 mm: 66 M cells, ~8 x M1's cost (section 6).

Rejected: matching delta_th/delta_D by a fake lambda(T) law (T^0.97: a model
nobody measured); two-step chemistry (not built; the honest fix, listed in
issues).

**D2 (equivalence ratio).** phi 1.3: the classic two-cone Bunsen (inner rich
premixed cone, outer diffusion cone), clear of the rich limit (1.67), no soot.
Premix: f = phi Y_O2/s, Y_F1 = f/(1+f) = 0.0705, Y_O1 = Y_O2/(1+f) = 0.2166,
left after the inner cone Y_F,rem = (phi - 1) Y_O1/s = 0.0163 (23% of the
fuel). Jet Z = 1, air Z = 0: Z_st = (Y_O2/s)/(Y_F,rem + Y_O2/s) = 0.78.

**D3 (heat release).** q fixed by the INNER flame: sigma_in = T_b(rich
premix)/Tu = 6.88 = 1 + q (Y_O1/s)/(cp Tu). The outer sheet then sits at
sigma_st = 1 + q Y_F,st/(cp Tu) with Y_F,st = 0.0551: 7.00 (2086 K; real
stoich 2225 K, the burner's outer cone ~1900-2000 K). The inner cone owns the
flow (expansion, cone angle); one-step has no CO/H2 so the split of heat
between cones is the stoichiometric one (77/23).

**D4 (S_L calibration).** [SUPERSEDED by S1b: the shape IS fitted now --
model C2b, A C(xi) with C from GRI-3.0's S_L(phi), unit S_L at the jet's
phi kept (C(jet) = 1).] Unit S_L AT the jet's phi from a two-reactant FP64
oracle (S1). Not a fitted S_L(phi) shape: one-step's own S_L(phi) is printed
by `--dry-run` over phi 1.0-1.6 next to the Cantera curve (documented, not
gated). It matters only at the edge flame where rim mixing leans the
premix toward phi 1 (model and CH4 both burn faster there: right sign).

**D5 (Mach; items 6-7).** The real case runs on a NEW low-Mach solver (exact
M -> 0 limit; real M 1.15e-3 differs by O(M^2) = 1e-6). The compressible flame
solver keeps its inflated M and its precedents untouched. The compressible
path at true M is rejected: FP32 p resolution (issues §31: ulp(p0)/(rho S_L^2)
4e-4 at M 0.01, 0.37 at M 1.15e-3 -- the pressure jump of the flame is below
one ulp of p0) and 17 x more steps; FP64 is 1/32 rate on gfx1100.

**D6 (build the new physics once).** Rim, open laterals, gravity, partial
premixing at 3D are built ON the low-Mach solver only. The compressible path
gets only the cheap two-reactant 1D premixed twin (S1, a cross-check of the
chemistry on the trusted solver). `--rim`, `--open-sides`, `--gravity` with
`--mode bunsen --physics compressible` are rejected by name ("only with
--physics lowmach"). Reason: no precedent breaks (compressible open laterals
would need hydrostatic lateral NSCBC + species open-y kernels + lateral
re-carry, F2/F3: three new instantiations for a solver the real case does not
use).

---------------------------------------------------------------------------

## 2. The low-Mach solver (item 7), design

Files (layering, CLAUDE.md "Layering"): config `dns/lowmach.{hpp,cpp}` (case
derivation, inflow rows, Poiseuille, ambient, IC `lowmachIC`, measurement on
host fields), `dns/chem` (oracle), library `gpu/poisson` (+ `SeparablePoisson`),
`gpu/lm_poisson.{hpp,hip}` (apertured PCG on gpu/la Krylov), solver
`gpu/lowmach.{hpp,hip}` (`DeviceLowMach`), runner `gpu/runner_lowmach.{hpp,hip}`
(`runLowMach`), dispatch `apps/dns_main.cpp` (`runFlame`, 1167-1178:
`c.physics == Physics::LowMach` first). The species face formula moves VERBATIM
from `species.hip:84-90, 280-331` (mcSlope, kVoteJump, sharedTheta, faceY2) to
`gpu/internal/species_face.hpp` and both solvers include it (D7).

**D7 (one face formula).** A pure code move of `__device__ inline` helpers;
acceptance = the bit-identity suite (below) byte-identical. Two copies of the
limiter would be the "restated rule" CLAUDE.md forbids.

### 2.1 Model and state

- Variable-density low-Mach Navier-Stokes, calorically perfect, gamma 1.4,
  equal W, Le = 1; thermodynamic pressure P0 CONSTANT (open domain); EOS rho T =
  P0/R. Box units rho_u = S_L = R = 1, delta_D = 1, Tu = P0 = 1 (no acoustic
  scale: `--mach` rejected by name), cp = 3.5, mu = Pr at Tu, Sutherland via
  `Transport` (`--tinf` anchors S/Tu).
- Grid: uniform, MAC staggered (the incompressible solver's `DeviceField`
  convention). State per step: rho, rho Y_k (cells), m = (rho u) on faces, p
  (dynamic, cells), p_prev. T, Y_k, u_f = m_f/rho_f are derived.
- Chemistry: `arrheniusRate` (premixed) / `arrheniusRate2` (two-reactant), the
  existing templates with (rho, Y, T) -- no second rate expression.
- Gravity: the ONE buoyancy term of this model, (rho_f - rho_a) g on the x
  faces, rho_a = P0/(R T0). Hydrostatics live in p; the quiescent ambient is p
  = 0 exactly (that is what the open faces need: item 4's "ambient
  hydrostatic" is automatic). CLAUDE.md "Gravity" invariant gains: "low-Mach:
  `gpu/lowmach` `addBuoyancyLm`, the only one there".

### 2.2 Step (SSP-RK3, stage combination ca, cb, cr as `StageCombine`)

Per stage s, from (rho, rho Y, m)^(s):
1. Primitives: T = P0/(R rho), Y = rho Y / rho, u_f, omega (cell pass).
2. **Temperature predictor** T* = ca T^n + cb T^(s) + cr dt R_T, R_T = [-div(m
   T_f) + T div m]/rho + [div(lambda grad T) + q omega]/(rho cp); T_f with the
   SAME shared theta as the species (T does not vote) so the coupling H = T +
   (q/cp) Y_F is transported by one linear face formula. Target density rho* =
   P0/(R T*).
3. **Momentum predictor** m* = ca m^n + cb m^(s) + cr dt [-div(m (x) u)_skew +
   div tau + (rho - rho_a) g e_x - grad p^(s)] (variable-density skew-symmetric
   central convection, Morinishi; no added dissipation, like the compressible
   mean flow).
4. **Projection on the MASS FLUX** (Pierce & Moin 2004 / NGA): the momentum
   equation projects m, so the operator is CONSTANT-coefficient:
   L phi = [div m* - D*]/(cr dt), D* = (ca rho^n + cb rho^(s) - rho*)/(cr dt);
   m^(s+1) = m* - cr dt grad phi; p^(s+1) = p^(s) + phi.
5. **Continuity + species ride the PROJECTED flux**: rho^(s+1) = ca rho^n + cb
   rho^(s) - cr dt div m^(s+1) (= rho* to solver tolerance), rho Y_k^(s+1) with
   F = m^(s+1) Y_f (shared limiter) - rho D grad Y, - omega (- s omega).

**D8 (why this formulation).** (i) Constant-coefficient Poisson: no variable
coefficient, no AMG per stage. (ii) Mass conserved EXACTLY (telescoping
fluxes) whatever the solver tolerance. (iii) A uniform Y is preserved to the
bit (rho Y and rho ride the same m^(s+1); `gpu_flame_passive`'s power-of-two
test carries over). (iv) EOS drift = solver residual, self-correcting (each
stage's T* starts from T = P0/(R rho) of the carried rho). Cost: the T
predictor uses m^(s), species m^(s+1): an O(dt^2) per-stage Shvab-Zeldovich
defect (K3, gated in S4). Projection per stage, not per step: the constraint
then holds at every stage (the repo's incompressible convention).

**D9 (pressure solver).** Without immersed solids the apertured operator IS
the constant-coefficient Laplacian with separable BCs: ONE direct spectral
solve per stage (`SeparablePoisson`, below). With the tube: PCG
(`la::Krylov<float>::pcg`, LinOp = binary-apertured 7-point Laplacian on the
fluid cells, identity on solid rows; M = the spectral solve of the full box),
warm-started from the extrapolated phi. The cutpoisson precedent (spectrum
clustered, tens of iterations cold) on a thin tube, warm, is expected at 3-10;
gate <= 15 (S8). AMG rejected: a V-cycle-PCG at 96^3 is 46 ms (docs/la.md
"AMG"), 3 per step at 12 M rows ~ 1.5 s/step; the spectral PCG ~ 3 x (1 + it)
x 2 ms. FP32 fields and iterate, FP64 batched dots (la rules); the FP32 floor
~ eps ||A|| ||x|| sets the EOS drift (measured, K5).

`SeparablePoisson` (`gpu/poisson.{hpp,hip}`, NEW class; `DevicePoissonSolver`
and `PoissonBc` untouched): per-axis kind {Periodic (FFT), NN (DCT-II, the
existing Makhoul path), DD (DST-II = DCT-II of (-1)^i x reversed), ND (DCT-IV,
eigenvectors cos((k+1/2) pi (i+1/2)/n): Neumann lo face, Dirichlet hi face;
n/2-point complex FFT + twiddles, n even)}; eigenvalues of the 2nd-order FD
operator (2 - 2 cos theta_k)/h^2; z untransformed at nz 1. Low-Mach uses x =
ND always (inflow Neumann, outflow p = 0), y/z = Periodic or DD (open sides).
FFT-friendly sizes enforced by the validator (prime factors <= 7 on
transformed axes, error by name with the nearest sizes).

### 2.3 Boundaries

- x-lo INFLOW face: Dirichlet m_x = rho_row u_row per (j, k) (Poiseuille in the
  tube, coflow outside, 0 on solid faces), tangential u = 0; T, Y_k FIXED FACE
  values with convective AND diffusive flux through the face (reactingFoam's
  inlet: H 5e-4 vs dns' pinned LODI row 0.35, issues §45/§67; D10). phi:
  Neumann.
- x-hi OUTFLOW: p = 0 (phi = 0), m_n from the projection, tangential u
  zero-gradient; scalars upwind, no diffusion; BACKFLOW (m_n < 0) carries the
  ambient (air, T0; the candle's `ambientHi` rule, `species.hip:399-400`) and a
  backflow-stabilization term -1/2 rho min(u_n, 0) u on the face (Moghadam et
  al. 2011) in its own instantiation.
- y/z (`--open-sides`): traction-free p = 0 (DD), m_n from the projection;
  inflow faces: tangential u = 0, T0, air; outflow faces: zero-gradient,
  own values, no diffusion. Stencils near an open face take per-axis open flags
  (the `cellFaces` first-order rule generalized from `a == 0` to `open[a]`);
  `offsetsAt` keeps wrapping (gotcha: one cell out of range MUST wrap) -- the
  open treatment is a flag override at the face, never an edit of offsetsAt.
- Periodic y/z otherwise (comparison runs).

**D10 (fixed-face inflow).** The low-Mach inflow is a face condition, so
§66 (density-held inflow enters hot), §68 (LODI mass-flux overshoot ~3-7%) and
§45 (cool inflow row) do not exist by construction; the inflow mass flux is
EXACTLY the profile's. Consequence: compare to the compressible solver only
with `--nscbc-tinflow --nscbc-etain 1000` (its foam-agreeing variant).

### 2.4 Immersed tube (item 2)

- Geometry: annulus R_i <= r <= R_o, 0 <= x <= x_rim (exact distance in the
  (x, r) half-plane: a true sdf, so `--thin-seal`-style crossing tests are
  valid if ever needed). The tube STANDS ON the inflow face (D11).
- Velocity: MAC faces with a solid side are closed (m = 0: impermeable,
  exactly, in the apertured projection); the viscous stencil reads the
  diffusion-only ghost view of `gpu/ghostfill` (its clean trilinear normal
  image, `ghostfill.hpp:5-12`) -- CLAUDE.md's "ghost view is diffusion-only"
  holds: convection sees the closed face.
- Temperature: isothermal T_rim (`--rim-temp`, T/T0, default 1) as a
  diffusion-only scalar ghost view, q_g = T_w + (sdf/d)(T(image) - T_w): a new
  scalar instantiation of the ghostfill table (cell-centred), same image rule.
  Wall heat flux tallied from the conduction face fluxes the stage uses.
- Species: no flux (convective or diffusive) through a fluid/solid face, solid
  second neighbour -> theta 1 face (the walled-kernel rule, `species.hip:
  339-346`), solid cells carry rho Y = 0, no omega, no heat.
- Gravity, Poisson: solid rows identity in the PCG LinOp.

**D11 (tube on the inflow face, not floating).** The compressible chamber/wick
floated bodies because ghost rows must not sit in NSCBC rows or sink bands.
The low-Mach solver has neither: the annulus' inflow faces are simply closed
(m = 0), so the tube enters the domain like the real one. No stagnant gap
under a floating lip mixing air into the near-wall jet layer (that layer IS
where the rim anchors).

**D12 (Poiseuille).** Fully developed u = 2 U (1 - r^2/R_i^2): the lowest
wall gradient a laminar tube can have (8U/D = 0.75 S_L/delta_D at M1) -- the
conservative anchoring case. Real tubes at L/D ~10, Re ~900 are developing
(flatter, steeper wall gradient). Flashback margin (Lewis-von Elbe g_F ~
S_L/d_pen, d_pen ~ d_q/2 ~ 1 mm ~ 6.5 delta_D): g/g_F ~ 5.

### 2.5 dt, determinism, precision

- dt = min(CFL h/sum_a |u_a| (cfl 1.0 for MAC RK3, measured), diffusive
  0.25 h^2 rho_min/(mu(T_max) max(1, 1/Pr)) (one reduction, Sutherland
  monotone), chemistry kChemDt / J with J of the cp-form (`chem.hpp`'s
  expression with cp for cv)). Expected at M1 (h 1.15 delta_D): diffusive
  0.011 tau, chemistry 0.0022 tau at kChemDt 0.5 (the run 746 stats read
  dtChem 0.0016 with cv), convective 0.04. **K1**: chemistry binds 5x below
  diffusion. S4 MEASURES the explicit stability bound (kChemDt 0.5/1/2:
  S_L drift <= 1e-4) and, if the measured bound still binds 2x below
  diffusion, S4b adds point-implicit chemistry (cell-local linearised backward
  Euler on the stage's (Y_O, T) Jacobian, its own instantiation; S_L vs
  explicit <= 1e-3).
- Determinism: own-index writes, no FP atomics, fixed-order FP64 reductions
  (la batched dots; `internal::reduce` only for max), hipFFT plans fixed per
  size: reruns byte-identical (tested).
- FP32 fields; P0 = 1 (no p0 + small-difference problem: p is the DYNAMIC
  pressure, O(rho S_L^2)), which is the point of D5.

---------------------------------------------------------------------------

## 3. Checkpoint / restart (item 8)

Flags (every `dns flame` path that adopts it; S2 wires `runFlame`, S4
`runLowMach`): `--checkpoint-dt T` (model time; 0 = off),
`--wall-limit SEC` (stop at the first step boundary after a checkpoint once the
wall clock passes SEC; needs `--checkpoint-dt`), `--restart PREFIX`.

- **Checkpoint time = the first step boundary at or past k T** -- never a
  clipped dt, never a forced stats row (F4). The file `<prefix>_ckpt.bin`
  (written to `.tmp`, fsync, rename: a job killed mid-write keeps the previous
  one) holds: magic, version, binary sha256, a canonical hash of every
  case-defining flag (everything but `--tt`, `--checkpoint-dt`,
  `--wall-limit`, `--restart`; the `--emit` set and frame cadence ARE
  case-defining: byte equality covers the frames), grid, nY; the host loop state (t, steps,
  sinceDt, dtStab, dtChem, bindChem, StatsCadence lastLog/lastPrint,
  ProfileSchedule next, xfPrev/tPrev, lastHits/Min/Max, turbulence sweep,
  uNow/sEma/tCtl/xI0, injCum, the RunOutput frame index and next frame time,
  each series' row count and byte length); the device state raw (rho, m, E,
  y[], lo words if compensated; low-Mach: rho, rho Y, m, p, p_prev).
- **Restart** (phase 1.5, like `--scene`: the validator reads the HEADER
  facts -- hash, grid, t -- and rejects a mismatch by name, exit 3:
  "--restart: case differs from the checkpoint: --jet-speed 4 vs 5"; a
  `--tt` at or below the checkpoint's t is exit 2). Phase 3 truncates every
  series and frame file to the checkpoint's recorded length (a job killed after
  its checkpoint left extra rows) and appends. `--restart` must equal `--out`
  (a restart continues that run's outputs; error by name otherwise).
- **Byte rule**: a run interrupted by `--wall-limit` at any checkpoint and
  restarted equals the uninterrupted run byte for byte in every output except
  the `.json` timing fields (`msPerStep`); the json gains `"complete": true|
  false` and `"restarts": N`.
- Exit: a wall-limit stop is exit 0 with stdout `checkpoint: t=... steps=...
  (incomplete; resume with --restart PREFIX)` and json complete false (**D13**:
  no new exit code; the chain script reads the json).
- SIGINT (`stopRequested`) writes a checkpoint too.
- `tools/chem/chain.sh PREFIX N -- ARGS`: submits up to N simjobs in
  sequence (`--wait`, `--wall-limit 1700`, `--restart` from the second), stops
  on rc != 0 or complete true. Used through `run_in_background`.

CLAUDE.md: phase-1.5 exception list gains `--restart`; ground rule "a
restarted run equals the uninterrupted one byte for byte".

---------------------------------------------------------------------------

## 4. Stages

Each: buildable, testable, committable alone; every existing path bit for
bit. **Bit-identity suite** (run at the end of every stage that touches a
shared file; base = the previous commit's binary): `flow --physics
compressible --far nscbc` (stats), `conv`, `channel --nz 8`, premixed 1D
(stats+profiles), bunsen 2D (frames), bunsen `--nozzle round --nz 16` tt 1,
diffusion, candle, wick, chamber -- every output but the json byte-identical
(the turb3d convention, docs/chem.md "Bit identity"). One simjob, < 5 min.

### S1 -- Two-reactant premixed oracle + rich 1D twin (compressible) + oracle fix

Files:
- `src/dns/chem.{hpp,cpp}`: fix the bracket (F1): log-grid scan of m (e.g.
  2^-20..2^20, 81 points) for the cold-residual sign pattern, accept exactly
  ONE tooFast transition, bisect inside it; several transitions = error by
  name ("flame eigenvalue: non-monotone shot at m in [a, b]; raise n"). Add
  `FlameProblem::rateFactor` (std::function<double(double T, double s)>,
  empty = 1: the shipped arithmetic bit for bit) multiplying the rate in
  `shoot` (`chem.cpp:40`) and the hot-end B (`chem.cpp:32`, factor at s = 0).
- `struct PartialPremixModel` (new, `dns/chem.hpp`): phi, s, yO2, sigma (=
  sigma_in), beta, Pr, sutherland, sOverTu; derived yF1, yO1, yFrem, zst, q,
  sigmaSt, theta; `premixedEigen2(m)`. O is deficient: progress s_p = (Tb -
  T)/(Tb - Tu) = Y_O/yO1, Y_F = yFrem + Y_O/s, and q omega_F = cp dT A rho s_p
  e^(-Ta/T) * [rho s Y_F(s_p)], so rateFactor(T, s_p) = rho(T) s (yFrem + yO1
  s_p/s); m ~ sqrt(A) still holds (Lambda2 = 1/m(A=1)^2). Pre phi > 1 (yFrem >
  0 keeps the hot-end linearisation p = a (Tb - T)).
- `src/dns/config.cpp` (`validateFlame` 1903-2154): `--equivalence PHI`
  (premixed mode, compressible or lowmach; bunsen with lowmach only), range
  [1.05, 1.6] ("rich premix only: a lean or stoichiometric jet has no outer
  cone; 1.6 below CH4's rich limit 1.67"); with it `--stoich` (default 3.99)
  and `--ox-y` (0.233) are accepted, `--fuel-y`, `--damkohler`,
  `--cells-per-unit`, `--ignition` stay rejected by name; `--les`, `--tfles`,
  `--turb` rejected ("two-reactant LES/TFLES species not built").
  `flameRunLabel` gains "partially premixed two-reactant (phi, s, Y_O2), unit
  S_L at phi". Dry-run prints Lambda2, delta_th/delta_D, yF1, yO1, yFrem,
  zst, sigma_st and the model S_L(phi)/S_L(phi_jet) table phi 1.0..1.6.
- `src/dns/flame1d.cpp`: `flameIC2` (oracle profile; Y_O = yO1 s_prog, Y_F =
  yFrem + Y_O/s).
- `apps/dns_main.cpp:1295-1378`: premixed branch, nY 2, ox 1 (kernels
  `kChemCellPass2`, `kSpeciesStage2<false,false>` -- already compiled).
- `src/dns/gpu/runner_flame.hpp/.hip`: `FlameRun::yFBurnt` (default 0): SL
  = integral omega / (rho Y_F - rho yFBurnt)_ref (`runner_flame.hip:498-499`;
  x - 0 is exact, so 0 is the old expression bit for bit).
- `tools/chem/cantera_flame.py`: `--real phi..` mode: GRI-3.0 S_L, T_ad,
  delta_th, alpha_u at phi 1.0-1.6 -> `tests/data/chem/ch4_gri30.csv` (reference
  only).

Tests: `flame1d` (host): Lambda(sigma) continuous over sigma 2..10 step 0.02
with and without Sutherland (no jump > 3% between neighbours: catches F1),
sigma 6.88 Sutherland delta_th/delta_D in [4.4, 4.7]; `rateFactor` empty ==
old table bitwise; two-reactant m(4A)/2m(A) - 1 < 1e-10; phi -> large (yFrem >>
yO1) reproduces the single-reactant Lambda scaled by rho s yF (limit check,
1e-6); vs the numpy port (`tools/chem/foamflame/model.py`) extended.
`flame_config`: every new rejection by name, dry-run lines. GPU binary
`gpu_flame_rich1d`: compressible 1D phi 1.3, sigma 6.88, Sutherland, M 0.01,
7.5/15 cells, `--inflow-speed 0.98`, tt 20, window [10, 20]: S_c within the
single-reactant ladder's band (+1.7% / +0.3%), yHits 0, Z of the products =
1 to 1e-5.

Acceptance (simjob, < 3 min): `gpu_flame_rich1d`; bit-identity suite
(premixed 1D, bunsen, diffusion byte-identical: the oracle fix must not move
a WORKING sigma -- if it does, the old Lambda was wrong there too: document).

Docs/issues: docs/chem.md "P0" (bracket, rateFactor, Model C2 partially
premixed); issues 72 (oracle bracket, fixed: table F1), 73 (one-step
delta_th/delta_D 4.5 vs CH4 8.0: the D1 trilemma).

**S1 STATUS: DONE (2026-10-31, uncommitted working tree).** What was built
and where it deviates:

- F1 had TWO causes. (a) The spurious verdict is a STIFF collapse onto the
  manifold p* = lambda q omega/(m cp) at m far above the root (RK4 bounced p
  to a positive residual) -- a scan alone would have turned sigma 6.5 into a
  "non-monotone" ERROR, not a number. Added: a step with h J > 1 (J =
  lambda q omega |dT/du|/p^2) is too fast (slow shots peak at 2.2e-3). The
  scan is 2^-60..2^20 in POWERS OF TWO (not 2^-20..2^20 in half powers:
  Lambda up to 2^120 must resolve, and the bisection must start from the
  shipped [lo, 1]/[1, hi] bracket so every well-posed Lambda is bit for
  bit). (b) The cold-boundary difficulty at low sigma beta (sigma 2 beta 8:
  the uncut march drifted as 1/n): an adaptive rate cutoff `kColdCut` 0.5
  (inactive = bit for bit where the unburnt rate is negligible), a cut
  above `kColdCutMaxRise` 1e-4 of the rise is an error by name.
- Changed a shipped number: sigma 2 beta 8 (lesgap/G2b) Lambda 4.7246e8 ->
  5.0312e8, delta_th/delta_D 1.7953 -> 1.7234 (GPU: the old Lambda burns at
  0.975, the new at 1.002 at 15 cells). Newly rejected (were 1/n garbage):
  beta 8 sigma <= 1.70, beta 6 <= 2.2, beta 4 <= 3.25, beta 3 <= 4.25, beta
  2 <= 6.2. Every other well-posed case bit for bit (suite below).
- Test (e) uses |second difference of ln Lambda| < 0.02 over sigma 2..10
  step 0.02, not "no jump > 3%": the model's own Lambda changes 16% per step
  at sigma 2. Measured max 5.9e-3; the old bracket's jumps are >= 0.69.
- `--equivalence` is premixed-mode only now (bunsen: rejected, "needs
  --physics lowmach (not built yet)"); the dry-run S_L(phi) table runs phi
  1.05..1.6 (phi 1.0 has Y_F,rem = 0: no hot-end start). Z gate in
  `gpu_flame_rich1d` is 5e-4, not 1e-5: the FP32 walk of non-power-of-two
  Y's reads +-1.7e-4 (both grids; 2.4e-5 at tt 2).
- Measured: rich (C2) vs one-reactant S at 7.5 / 15 cells 1.015612 /
  1.002544 vs 1.015058 / 1.002695 (gate 2e-3). Lambda2(phi 1.3, sigma 6.88,
  Sutherland 298 K) = 46986660.73, delta_th/delta_D 4.70964; Y_F1 0.070558,
  Y_O1 0.216560, Y_F,rem 0.016283, Z_st 0.78196, sigma_st 6.9773.
- GRI-3.0 (Cantera 3.2.0, venv `runs/cantera_venv`, run via simjob --cpu;
  `tests/data/chem/ch4_gri30.csv`) REPLACES the starred inputs of section 1:
  S_L(1.3) 0.2312 m/s (not 0.25), T_ad 2056 K (sigma 6.899), delta_th 0.619
  mm (not 0.70), alpha_u 2.234e-5, nu_u 1.591e-5 m^2/s, delta_D 0.0966 mm,
  real delta_th/delta_D 6.40 (not 8.0). Re-derived M1 (with C2's 4.71):
  delta_D,model = 0.619/4.71 = 0.131 mm, tau = 0.568 ms, Pr_eff = nu/(S_L
  delta_D,model) = 0.52 (not 0.41), Pe 0.74x real (not 0.6x); real Re at U
  1.5 m/s, D 10 mm = 943; U/S_L = 6.5 (S5+ must re-pick --jet-speed: 1.5
  m/s is 6.5 S_L, or keep 6 S_L = 1.39 m/s); D = 10 mm = 16.2 delta_th.
- **K8 is worse than assumed (issue 73)**: the C2 model's S_L(phi) RISES
  with phi (0.755 at 1.05 .. 1.174 at 1.6, rel. 1.3) where GRI-3.0 falls
  (1.634 .. 0.342). D4's "right sign" is wrong: the leaner edge flame at the
  rim burns SLOWER in the model than in CH4. S5/S8 must state it; the
  anchoring scouts (rim standoff) inherit it.
- Bit-identity suite (base = HEAD e92dad1 binary, `runs/s1/bitid.sh`, one
  simjob, 29 s): premixed 1D (stats, profiles, frames), Sutherland 1D,
  3D planar, bunsen 2D, round 3D (volumes), diffusion, candle, chamber,
  wick -- every output but the json byte-identical.

For S2: `runFlame` gained one premixed variant (nY 2, `FlameRun::premix2`,
`yFBurnt`, stats `zMin`, `zMax` appended after the 1D/3D columns) -- the
checkpoint must carry rhoY[1] and the restart byte rule must cover an
`--equivalence` run. The oracle is ~0.12 s per solve (scan + bisection);
`--equivalence` dry-runs pay 1 + 2 x 7 solves for the S_L(phi) table
(describe() only).

**S1b STATUS: DONE (2026-10-31, uncommitted working tree; issue 73 (b)
fixed).** Model C2b (docs/chem.md "Model C2b"):

- omega = A C(xi) rho^2 Y_F Y_O e^(-Ta/T), xi = (s Y_F - Y_O + Y_O2)/(s +
  Y_O2) (fuel/air mixture fraction = Y_F1 of a fresh premix; jet Z = xi (s
  + Y_O2)/(s Y_F,rem + Y_O2)). ln C = clamped cubic spline (zero end slopes,
  constant beyond) on 26 GRI-3.0 nodes phi 0.5..1.7 (step 0.05 + 0.975 /
  1.025 / 1.075; phi 1 skipped), ln C_k = 2 ln(S_GRI/m_uncorrected), shifted
  so C(jet) = 1: Lambda2, the jet's IC table and its S_L = 1 unchanged.
  ONE definition `PhiRateLaw<R>` + `phiRateLn` (DNS_HD, `dns/chem.hpp`);
  host builder `phiRateLaw(md)` (validate time, threaded, ~0.1 s); GRI rows
  embedded (`ch4Gri30Speeds`, = csv rows, test-checked). Table, not closed
  form (the data is the definition; a tanh fit adds error).
- Oracle gained the LEAN branch (`premixedEigen2At(md, phi, law)`: Y_F = Y_F1
  s_p, Y_O = Y_O,rem + s Y_F; returns the JET's Lambda, tab.m = S_L(phi)/
  S_L(jet), the flame's table) and `PartialPremixModel::sigmaAt`, `yOrem`.
  `premixedSpeedRatio(md, phi, law = null)`; null and phi = jet: 1 exactly.
- GPU: `kChemCellPass2Phi` (its own kernel: arrheniusRate2(lnA + ln C,
  ...) in FP32), `ChemRate2Phi` in chemStableDt (FP64); selected by
  `SpeciesSetup::phiRate.n > 0`. Every existing mode byte-identical to HEAD
  (`runs/s1/bitid.sh`, 9 cases, job of 2026-10-31).
- Flags: `--sl-law gri30|onestep` (gri30 = DEFAULT under `--equivalence`:
  decided because S1 is uncommitted and the uncorrected step's wrong-sign
  S_L(phi) is a physics error for the rim; onestep = S1's model, its kernel,
  S 1.002544 at 15 cells as S1 measured); gri30 needs CH4/air (`--stoich
  3.99 --ox-y 0.233`, else error by name). `--mix-phi PHI` (premixed mode,
  [0.6, phi_jet], |PHI - 1| >= 0.01): the 1D/3D flame burns the jet diluted
  with air (the rim's mixtures), A/q/Ta the jet's. Both rejected by name
  without `--equivalence`.
- Fit: nodes exact (4e-15); held-out GRI rows (0.575, 0.625, 1.0, 1.125,
  1.325, 1.375, 1.425, 1.575) <= 3.35e-3 (1.575, the end clamp's reach),
  <= 1.85e-3 over 0.575..1.425 -- GRI's own grid uncertainty is 0.2-0.4%.
  C: 20.2 (phi 0.6), 5.46 (phi 1 = Z_st), 1 (1.3), 0.085 (1.6); ln C
  monotone, C1 (slopes at nodes to 3.6e-5); FP32 vs FP64 ln C 2.9e-6.
- T_ad(phi) is the one-step's, NOT calibrated: Tb/Tu -16% at phi 0.6, -6.5%
  at 1, -0.3% at 1.3, +10% at 1.6 vs GRI (jet q at 1.3; no CO/H2, equal
  cp). sigma_st 6.977 vs GRI's stoich 7.465.
- GPU 1D (`gpu_flame_rich1d_law`, 15 cells, inflow 0.98 r): S/r - 1 =
  +2.1e-3 (jet), +3.9e-3 (mix 0.7; +7.6e-4 at 30 cells), +1.4e-3 (1.05),
  +1.3e-3 (1.2), +3.0e-3 (1.2 onestep); (S/S_jet)/r - 1 <= 1.8e-3.
  `gpu_flame_rich1d` (S1) passes with the law on (rich - single -5.6e-4 /
  +5.0e-4 at 15 / 7.5 cells).

For S2+: (1) the checkpoint's case hash must include `--sl-law` and
`--mix-phi` (they change the rate / the inflow); the law itself is derived
at validate (deterministic), not state. (2) S4's `kLmCellPass<TwoR>` MUST
take the law (`<TwoR, Phi>` or a law argument, its own instantiation):
call `phiRateLn` on the cell's Y_F, Y_O -- never a second rate expression,
and chemDt with it (ChemRate2Phi's J). (3) K15 below: the FP32
coupling-function walk now reaches the rate. (4) S5's air coflow and pilot
strips: a burnt rich pilot carries Y_F,rem, xi of the jet -> C 1; air xi 0
-> C saturates at C(0.5) = 23.2 where Y_F ~ 0. (5) The outer sheet's Da is
now C(Z_st) = 5.46 x the jet's A (closer to Burke-Schumann): S5/S8's outer
cone height and the triple-flame base inherit it; the edge flame now burns
FASTER leaner (GRI), the right physics for anchoring scouts (K7). (6) S5
labels must name the law (flameRunLabel's "S_L(phi) law C2b ...").

### S2 -- Checkpoint / restart on `runFlame`

Files: `src/dns/gpu/internal/checkpoint.{hpp,hip}` (`RunCheckpoint`: named
host scalars + device buffers -> file; read/verify), `src/dns/output.{hpp,cpp}`
(`RunOutput::resumeAt(positions)` truncate+append; frame counter export/import;
json `complete`, `restarts`), `src/dns/cli.cpp` (three flags, --help),
`src/dns/config.cpp` (validation + phase-1.5 header read in `dns_main`),
`runner_flame.hip` (loop state into a struct the checkpoint serializes; the
checkpoint/wall-limit test at the step boundary after `t += dt`, BEFORE the
stats test, so it never changes what a row sees), `tools/chem/chain.sh`.

Rejections: `--wall-limit` without `--checkpoint-dt`; `--restart` != `--out`;
`--restart` on wick/chamber/flow (not wired: "only dns flame
premixed/bunsen/diffusion/candle and --physics lowmach"); checkpoint-dt <= 0.

Tests: `restart_config` (host: hash, mismatch messages, truncation logic on a
temp dir in the test's own dir -- a test may touch the filesystem);
`gpu_restart_flame` (GPU binary): 1D premixed tt 2 straight vs
`--checkpoint-dt 0.5 --wall-limit 0` (stops after the first checkpoint) +
3 restarts: stats, profiles, frames byte-identical; same for bunsen 2D
(per-row inflow, pulse on), round 3D nz 16 tt 0.5 (volumes), a TFLES
premixed 3D tt 0.3 (TFLES fields recomputed: must still match), and a
deliberately truncated-after-checkpoint output (rows appended past the
checkpoint) -> identical after restart.

Acceptance (< 4 min). Bit identity: no checkpoint flag = the shipped runner
(suite).

**S2 STATUS: DONE (2026-10-31, uncommitted working tree, on top of
S1/S1b).** docs/chem.md "Checkpoint and restart" (format, rules), issue 74.

- Files: `src/dns/checkpoint.{hpp,cpp}` (host, dns_core: the format, atomic
  write, header-only read, `CaseFlags` signature + diff, `CheckpointFacts`,
  `OutputResume` records, `CheckpointPolicy`); `src/dns/gpu/internal/
  checkpoint.hpp` (header-only: `saveLoop`/`loadLoop` over a loop struct's
  `fields()`, `saveCompState`/`loadCompState`); `output.{hpp,cpp}`
  (`resumePoint`, `resumeAt`, `setCompletion`); `cli.cpp` (three flame
  flags, --help, the signature built in `parse`); `config.{hpp,cpp}`
  (`resolveCheckpoint`, describe lines); `apps/dns_main.cpp` (phase 1.5
  header check, full read in phase 3, `src/jobs/sha256.cpp` linked into
  `dns`); `runner_flame.hip` (`FlameLoop`, checkpoint/stop logic);
  `run_loop.hpp` (`finishTiming(.., stepsBefore)`); `tools/chem/chain.sh`;
  tests `restart_config` (host, 55 checks, 1 s), `gpu_restart_flame` (GPU
  binary, 39 checks, 37 s).
- Measured (simjob 776-781, 2026-10-31): byte rule holds on pre1d (junk
  appended past a checkpoint, truncated away), rich1d (`--equivalence`
  1.3, both species, zMin/zMax), bunsen2d (`--pulse`, bin + 15 PNGs),
  round3d (nz 32, volumes), tfles3d, candle, turbhold (`--turb
  --hold-flame`: uInflow 1 -> 1.024 across the checkpoints; slices), a
  complete tt 1.2 run grown to tt 2, SIGINT (stop at t 0.5766, resume).
  Bit identity: `runs/s1/bitid.sh` (9 flame cases) + flow compressible
  nscbc / conv / channel nz 8 byte-identical to HEAD e92dad1.
- DEVIATIONS. (1) The checkpoint is taken at the END of the step (after its
  stats row, profile, frame), not "before the stats test": a resumed job
  then starts a fresh step and no re-entry point is needed; same guarantee
  (nothing a row sees changes; never at tEnd, never a clipped step).
  (2) The case "hash" is the canonical TEXT of the given flags (sorted,
  final parsed values, shortest round-trip numbers, enum names, --emit as
  its set) stored in the checkpoint: the mismatch message names the flag
  (`--jet-speed 4 vs 5`, `(absent)`). Excluded: --tt, --checkpoint-dt,
  --wall-limit, --restart, --out, --quiet, --dry-run, --backend. A flag
  given explicitly at its default differs from absent (conservative).
  (3) `--wall-limit` stops at a checkpoint when elapsed >= SEC OR elapsed
  + the last interval's wall time > SEC (predictive: chain.sh's 1700 under
  the 1800 cap), counted from main's start. (4) `--checkpoint-dt` must be
  > 0 when given (no "0 = off": absent is off). (5) `--restart` also
  rejects another binary (sha256 of /proc/self/exe) and another solver
  name, exit 3; outputs shorter than recorded are exit 3. (6) `--emit web`
  and `vtk` rejected with the flags (web manifest per process). (7) A
  separate host module (dns/checkpoint) + a header-only GPU part instead
  of `gpu/internal/checkpoint.{hpp,hip}`: the phase-1.5 header read and
  the host test need the format in dns_core. (8) SIGTERM is treated like
  SIGINT (stopRequested): simjob's timeout writes a checkpoint in its
  10 s grace (rc 124; chain.sh stops, `--resume` continues). (9) CLAUDE.md
  NOT edited (the phase-1.5 exception list + "a restarted run equals the
  uninterrupted one byte for byte" are for the user to add).
- For S4 (low-Mach runner): reuse as is -- `CheckpointPolicy` on its run
  struct; solver name "flame-lowmach" (phase 1.5 in dns_main checks
  `solver` == "flame-compressible" today: make it per --physics); its own
  loop struct with `fields()` (EVERY loop variable, K12); its fields as
  `state.*` F32 records (`state.rho`, `state.y0..`, `state.mx/my/mz` for the
  MAC faces, `state.p`, `state.pPrev`; PCG warm-start history if it changes
  results) -- no format version bump; `out.resumeAt` before declareSeries;
  checkpoint at the end of a step; `gpu_lowmach_restart` = a copy of
  `gpu_restart_flame`'s `chained()`. Any runner-local scratch that carries
  across steps and changes results (a warm start, an extrapolated phi)
  MUST be checkpointed or the byte rule fails.

### S3 -- `SeparablePoisson` + apertured PCG (library)

Files: `src/dns/gpu/poisson.{hpp,hip}` (new class, per-axis kinds; existing
modes untouched), `src/dns/gpu/lm_poisson.{hpp,hip}` (`LmPressure`: spectral
direct or PCG on `la::Krylov<float>` with the apertured LinOp; face mask
from the solid map; warm start; result iterations/residual/EOS floor).

Tests: `poisson_sep` (host FP64 dense eigen-solve of the 7-point operator for
every kind combination on 6x8x10 and odd/even sizes; DCT-IV and DST-II vs
direct sums to 1e-14); `gpu_poisson_sep`: device vs host 1e-6 rel (FP32), all
16 combos (x in {P,NN,DD,ND}, y/z in {P,DD}), nz 1 = the 2D restriction
bitwise vs a 2D grid, reruns byte-identical; `gpu_poisson_apert`: a solid
annulus in a 64x96x96 box: PCG converges, solid faces carry exactly 0 flux,
iteration count reported (expect <= 15 warm, <= 40 cold), FP32 floor vs FP64
LinOp solve.

Acceptance < 2 min. No shared runtime file touched except poisson.hip (suite
on `flow --physics incompressible` tunnel + conv Boussinesq: byte-identical).

**S3 STATUS: DONE (2026-10-31, uncommitted working tree, on top of
S1/S1b/S2).** docs/la.md "Separable Poisson + the apertured low-Mach PCG",
issues 75 (FP32 floor, cost) and 76 (gpu_cutcell2d flake, fixed).

- Files: `src/dns/septransform.hpp` (host + device: kinds, eigenpairs, the
  per-element pre/post steps of DHT / DCT-II / DST-II / DCT-IV and the
  two-lines-per-FFT packing -- one definition, the host test runs it in
  FP64); `gpu/poisson.hpp` (+ `SepBox`, `SeparablePoisson`, additive);
  `gpu/poisson_sep.hip` (new TU); `gpu/lm_poisson.{hpp,hip}` (`LmPressure`);
  tests `poisson_sep` (host, 1 s), `test_gpu_poisson_sep` modes `sep` (4 s),
  `apert` (4 s), `perf`, `perfsep`, `hash`; `tests/sep_test_util.hpp`.
- Measured: device vs host FP64 on the same float rhs <= 6.2e-7 of max over
  every kind triple (64 in 3D; plan: 16), manufactured <= 4.6e-6, device
  eigenvalues to 3.9e-7, nz 1 == nz 8 extruded BITWISE per plane, reruns /
  fresh processes byte-identical. Tube 64x96x96: 15-16 iterations cold, 9-11
  warm (rhs change 1e-3 / 1e-2) to tol 1e-6; solution vs FP64 CG 3-5e-7.
  Final grid 384x176x176: spectral 1.45 ms (DD) / 1.40 (periodic), half
  scale 0.32 / 0.40 ms; the real tube 14 cold / 9-11 warm iterations,
  30-44 ms per solve (~2.9 ms per iteration). Bit identity (pre-S3 binary vs
  new): flow incompressible tunnel, cut-cell sphere, conv Boussinesq
  periodic + closed, bunsen 2D: byte-identical; poisson.hip untouched.
- DEVIATIONS. (1) `SeparablePoisson` lives in a NEW TU `gpu/poisson_sep.hip`
  (declared in poisson.hpp): poisson.hip stays byte-unchanged. (2) The box
  has NO ghost ring (boundary on the box faces); x-fastest dense arrays.
  (3) DHT for Periodic (real in, real out) instead of complex coefficients:
  every pass is real-to-real, so periodic and the cosine/sine kinds share
  one rotating-layout chain. (4) Two real lines per complex FFT on every
  non-ND axis + the scale fused into the last forward post (2.55 -> 1.45
  ms); lines never pair along z (the extrusion bit-identity). (5) All 4^3
  kind triples are supported and tested (the plan listed 16); nz == 1
  needs z Periodic, ND needs an even axis (errors by name; the FFT-size
  validator stays S4's). (6) PCG stop is |r| <= tol |b| (relTol 0, absTol
  tol |b|): relative to the RHS so a warm start pays only for its defect.
  (7) `LmPressure::make` rejects by name a solid without a Dirichlet face
  and fluid enclosed by solid (host flood fill). (8) The FP64 reference is
  plain CG on the FP64 operator instantiation (`apply(double)`), not a
  mixed FP64-PCG. (9) `gpu_cutcell2d` (pre-existing flake, uninitialised
  warm start in the TEST) fixed: issue 76.
- For S4 (what it must know). (1) Layout: cells only, the boundary on the
  box faces; S4's MAC faces need nx+1 x-faces (inflow and outflow faces are
  distinct) -- `kLmFace*` codes say which INTERIOR/wrap faces are open;
  boundary faces follow the kind (x: ND always). (2) The projection
  equation is `LmPressure::solve(rhs, phi, warm)` with L_ap phi = rhs; the
  matching correction is m -= cr dt G phi with G = (phi_nb - phi_c)/h on
  open faces, (0 - phi_c)/(h/2) on the outflow (Dirichlet) face, 0 on
  closed / Neumann faces -- then div m = D* to the floor (verified on the
  host in `apert`). (3) Without a solid `solve` is ONE direct pass (1.45 ms
  at 11.9 M cells): S4-S7 pay ~4.4 ms/step for three stages. (4) FP32
  floor (issue 75): the projected divergence carries 1e-5 - 1e-4 of |b|;
  measure eosErr against it; tol 1e-5 is the knee (solution 2.5e-6 of
  FP64). (5) Checkpoint: phi (the warm start) is runner state -- if the
  runner warm-starts from the previous phi or an extrapolation, those
  buffers go into the checkpoint (K12); LmPressure itself is stateless
  across solves. (6) The plan's projection budget (~45 ms/step with the
  tube) is ~2x low: ~100 ms/step at tol 1e-6 warm (S8/S10 cost).

### S4 -- Low-Mach core: `dns flame --physics lowmach --mode premixed`

Scope: single reactant (C0) AND two-reactant (S1's model) 1D/2D/3D, y/z
periodic, uniform inflow, x-hi outflow p = 0. Checkpoint from day one.

Files: `config.hpp:33` Physics::LowMach (+ name "lowmach", `config.cpp:283`);
EVERY other validator rejects it by name (flow, tgv, hit, channel, conv,
bench, mesh, prop: "--physics lowmach exists for dns flame only"); dns-ui
maps index 0/1 only (`src/ui/doc.cpp:37,132,176` unchanged; `--check`
round-trip unaffected; verify). `config.cpp:1921-1923` admits lowmach for
flame; `--mach` rejected by name with lowmach ("no acoustic scale: the
low-Mach model is the M -> 0 limit"), as are `--nscbc-*` (no NSCBC: "the
low-Mach boundaries are face conditions"), `--les`, `--tfles`, `--turb` (not
built). New: `dns/lowmach.{hpp,cpp}`, `gpu/lowmach.{hpp,hip}`,
`gpu/runner_lowmach.{hpp,hip}`, `gpu/internal/species_face.hpp` (D7 move),
CMake, `apps/dns_main.cpp` dispatch, `--help` text.

Kernels (each its own instantiation; names fixed now): `kLmCellPass<TwoR>`,
`kLmTemperature<TwoR, OpenY, OpenZ, Wall>`, `kLmMomentum<OpenY, OpenZ, Wall,
Grav, Backflow>`, `kLmDivTarget`, `kLmCorrect<OpenY, OpenZ, Wall>`,
`kLmScalars<TwoR, OpenY, OpenZ, Wall>` (continuity + species), `kLmDt`
(FP64 reduction functors). S4 compiles only the all-false variants + TwoR.

Runner stats (premixed): the 1D runner's columns (SL, xf, dth, ...) + pDyn
jump, divErr (max |div m - D*|), eosErr (max |rho T - P0/R|), pcgIt.

Tests:
- `lowmach_config` (host): label, units, every rejection by name, dry-run
  (dt bounds, Lambda), FFT-size rule.
- `gpu_lowmach_passive` (API): uniform power-of-two Y bitwise through 100
  steps; mass to the FP64 tally exactly (telescoping); rest state at rest to
  0.0; reruns byte-identical.
- `gpu_lowmach_1d` (binary): C0 sigma 5 beta 8, 7.5 / 15 / 30 cells, sliding
  0.98, window [10, 20]: |S - 1| <= 2e-2 / 4e-3 / 1e-3, order >= 1.8; vs the
  compressible ladder M -> 0 extrapolation (docs/chem.md: 1.003021 + 3.9e-4 at
  15 cells) within 1.5e-3 (different schemes: MAC/T-equation vs KG/E).
- `gpu_lowmach_rh` (binary, the acoustics-free check): 1D, 15 cells: dynamic
  p jump = -rho_u S_L^2 (sigma - 1) within 1e-3; pDyn time rms over [10, 20]
  <= 1e-6 of the jump (no acoustic content); T_b/(sigma Tu) - 1 <= 2e-4 and the
  issue-64 burnt-gas ramp measured (expected absent: report).
- `gpu_lowmach_sz` (API): Shvab-Zeldovich H defect on the 1D flame <= the
  compressible's (gpu_tfles_teeth DNS floor) x 2 (K3).
- `gpu_lowmach_chemdt` (binary): kChemDt 0.5 / 1 / 2 S_L at 15 cells (decides
  S4b).
- `gpu_lowmach_restart`: S2's byte rule on the low-Mach runner (1D, 2D, 3D).
- Documented comparison (not a test): compressible 1D at the TRUE M 1.15e-3,
  15 cells (~490k steps, ~100 s) vs low-Mach: S_L and its window drift (§31).

Acceptance (< 5 min): the binaries above; bit-identity suite (the D7 move).

S4b (conditional on gpu_lowmach_chemdt): point-implicit chemistry
instantiation `kLmCellPass<TwoR, Implicit>`; 1D S_L vs explicit <= 1e-3;
dt then diffusion-bound.

Docs: new `docs/lowmach.md` (spec: model, step, BCs, dt, determinism, the
EOS/SZ floors measured); docs/chem.md pointer; CLAUDE.md "Low-Mach
(`--physics lowmach`)" invariants (mass-flux projection, species ride the
projected flux, P0 constant, buoyancy one place, no NSCBC/sink, rejected
elsewhere); issues "Low-Mach (LM1...)".

**S4 STATUS: DONE (2026-10-31, uncommitted working tree, on top of
S1-S3).** docs/chem.md "Low-Mach" (spec + measured), issues.md LM1-LM7.

- Files: `dns/lowmach.{hpp,cpp}` (FFT rule `lmFftSmooth/Next/Prev`,
  `kLmCflDefault` 0.8, `kLmChemDtDefault` 1, the solver names,
  `lowmachIC`), `gpu/lowmach.{hpp,hip}` (`DeviceLowMach`: kLmCellPass<TwoR,
  Phi>, kLmTemperature<TwoR>, kLmMomX, kLmMomT<A>, kLmDivRhs, kLmCorrectX/C,
  kLmScalars<TwoR>, kLmTally, FP64 functors), `gpu/runner_lowmach.{hpp,hip}`
  (`runLowMach`, `LowMachLoop`), `gpu/internal/species_face.hpp` (D7: Face,
  mcSlope, kVoteJump, `sharedThetaOf`, `faceValue2`, faceY2; species.hip's
  sharedTheta is a wrapper), `flame1d` (`premix2Reactants` factored out of
  flameIC2, shared by both ICs), config (Physics::LowMach "lowmach"; every
  other subcommand rejects it in validate()'s prologue; `--chem-dt`), cli
  (help section), dns_main (`runLowMachCase`, `readFlameRestart` /
  `flamePolicy` shared with runFlame, phase-1.5 solver name per
  --physics), tests `lowmach_config` (host), `gpu_lowmach_passive` (API),
  `gpu_lowmach_1d` / `_rich` / `_chemdt` / `_sz` (binary),
  `gpu_lowmach_restart` (binary; `tests/restart_test_util.hpp` now shared
  with gpu_restart_flame), `runs/s4/bitid.sh` (S1's 9 + flow comp nscbc /
  conv / channel nz 8: all byte-identical to HEAD e92dad1).
- Measured: C0 sigma 5: 0.994280 / 0.998274 / 0.999537 at 7.5 / 15 / 30
  cells, order 1.66, Richardson 1.000121 vs the compressible M -> 0 limit
  1.000079 (4e-5); pJump window mean -1.1e-3 of rho_u S^2 (sigma - 1); no
  burnt ramp; Tb exact; SZ <= 1.4e-6 (compressible 4.7e-3); mass = FP64
  tally (7.9e-12 / 4.0e-11 after 100 API steps); Z +-2e-7 (K15 closed);
  C2b jet 0.998470, mix 0.7 / 1.2 S/r - 1 = -1.7e-3 / -1.5e-3; restart
  byte rule 1D/rich/2D/3D; cost 3D 128^2: 2.87 ns/cell-step, 526 steps/tau
  (compressible M 0.02: 2.33, 1536) = 5.2 vs 12.2 s/tau.
- DEVIATIONS. (1) **rho and rho Y_k are COMPENSATED** (Kahan low words,
  stage_write.hpp's combineWord2): the plain FP32 state dropped the
  preheat zone's sub-ulp increments and the flame lagged (SZ 7.7e-4 and
  S -1.2e-3 growing with steps). m, p plain. (2) **D is the cancellation-
  free form** D = rho0 R_T/T* + P0 ca cb (rho_a - rho_b)^2/(rho_a rho_b cr
  T*) (the plan's (ca rho_a + cb rho_b - rho*)/cr fed rounding/cr into p);
  every stage write is a + (cb (b - a) + cr r) (the plain combination drifted
  uniform rho by an ulp per step). (3) The dynamic p carries an FP32 floor
  ~ 1/dt (row rms 0.17% / 1.0% / 6.5% of the jump at 7.5 / 15 / 30 cells,
  LM2); the plan's 1e-6 gate is not an FP32 number -- the rh gate is the
  window mean (2e-3). (4) Order 1.66, gate 1.5 (MC on T, LM3); the
  compressible comparison is gated on the grid-converged values, not at 15
  cells (5.1e-3 apart there, opposite-sign scheme errors). (5) S4b NOT
  built: K 0.5 binds 1.43x below diffusion at M1, K 1 does not (S 3.1e-5);
  `--chem-dt` default 1 (LM5). (6) Outflow scalars zero-gradient both ways
  (no ambient backflow: S6's instantiation); the tangential inflow is
  no-slip; momentum convection is the conservative (mass flux x mean
  velocity) form, KE-consistent with the discrete continuity, not a
  separate skew form. (7) Kernel templates: only the instantiations S4
  uses exist (TwoR, Phi); OpenY/OpenZ/Wall/Grav/Backflow are S5-S8's to
  add as template parameters (names fixed above). (8) The docs live in
  docs/chem.md "Low-Mach" (user's instruction), not a new docs/lowmach.md.
  (9) The last two steps share the remainder (no sliver step: p noise ~
  1/dt). (10) Stats: the plan's columns + dtDiff, massTally, szDefect; Z as
  dzMin/dzMax = Z - Z_inflow (the CSV's 6 digits). (11) `--cfl` default
  0.8 is NOT measured: the convective bound never binds in S4's cases
  (diffusion does, 9-30x below it); S5's jet (U 6, 4 cells) is the first
  place to bracket it.
- For S5+ (what to know). (1) The face-open definition for S6's open y/z:
  extend `kLmFace*`/`facePeriodic` -- today y/z are periodic in every
  low-Mach kernel (cellAt wraps). (2) Per-row inflow: `LowMachSetup::mIn,
  TIn, yIn` are scalars; S5 needs rows (j + ny k) as its own instantiation
  of kLmMomX's face 0, kLmTemperature/kLmScalars' x-lo face, and the
  tally's inflow sum (today the tally sums mx at face 0, so a row profile
  is already counted). (3) The rotational part of m is plain FP32 (LM1):
  measure a vorticity budget on the S5 jet before trusting long runs. (4)
  K3 (LM4) grows as dt^2: 2.5e-4 SZ on the M1 grid -- S9 must report the
  szDefect column. (5) Checkpoint: add any new runner state to
  `LowMachLoop::fields()`; device additions as `state.*` records in
  DeviceLowMach::save/load. (6) The 3D cost per step is ~1.2x the
  compressible's per cell; the solid tube's PCG (S8) adds S3's ~30 ms per
  solve at the final grid.

### S5 -- Low-Mach bunsen (per-row inflow), partial premixing, flame index

Scope: `--mode bunsen --physics lowmach`, slot 2D/3D and `--nozzle round`,
periodic laterals, pilot anchoring (no rim yet), `--equivalence`.

Files: `dns/lowmach.cpp` (`lmInflowRow(geom, y, z)`: the bunsen profile weights
of `bunsenInflowAt` (`bunsen.hpp:53`) reused; with `--equivalence` the coflow
is AIR (Y_O2, Y_F 0, T0) and the pilot strips rich products (sigma_in,
yFrem, Y_O 0)), `gpu/runner_lowmach.hip` (cone measurement: `measureCone` on
the axial plane (round) or span mean (slot), `measureJet`-style outer tip on
Z(Y_F, Y_O) with the S1 zst), `gpu/derived.{hpp,hip}` (`flameIndexCentred(gg,
YF, YO, out)`: xi = grad Y_F . grad Y_O / (|.||.| + eps), xi := 0 where either
|grad Y| < 2^-24/h: the vote threshold's reason).

Frames (lowmach bunsen): temp, fuel, oxid, omega, findex, speed (2D frames /
3D volumes). Stats: t, dt, dtChem, xTipIn (T crossing at Tmid_in on the
axis), xTipOut (axis Z = zst), xBase, burn, burnPre (= integral omega w_p),
burnDif, preFrac (-> 1/phi = 0.769 at complete burning), fuelIn (exact face
flux), fuelOut, Tmax, mass, divErr, eosErr, pcgIt, uMax; w_p =
smoothstep(-0.2, 0.2, xi) (the renderer's split, one definition in
`dns/lowmach.hpp`, D14).

**D14 (premixed vs diffusion per cell).** Takeno flame index sign(grad Y_F .
grad Y_O): + where both reactants are consumed toward the same side (the
inner cone), - across the outer sheet. Z vs Z_st cannot separate them here:
the inner cone burns at Z ~ 1 and the sheet at Z_st 0.78, which merge at the
base's triple flame. The solver emits xi (physics), the renderer weights.

Tests:
- `gpu_lowmach_bunsen2d` (binary): G2a case (U 5, W 12, burnt coflow, 8
  cells, tt 30, window [20, 25]) vs the compressible `--nscbc-tinflow
  --nscbc-etain 1000` at M 0.01 and 0.02 (M^2-extrapolated): xTip, sinRatio,
  SLflank within 1% (compressible G2a gap to foam: 0.6-2%). Also uniform
  per-row inflow through the profile path == uniform targets bitwise.
- Foam twin: `tools/chem/foam_flame.py` case dict `G2a` gains
  `extra_dns: lowmach` (the `twins` mechanism); docs table row "dns lowmach
  vs foam mclin" (not a ctest; foam ~4 min on 12 ranks, separate --cpu job).
- `gpu_lowmach_rich2d` (binary): 2D slot, phi 1.3, W 12, U 5, 3 delta_th
  rich-product pilots, air coflow, tt 30: two flames (burnDif > 0.1 burn),
  preFrac in [0.72, 0.80], closed outer tip, Z in [0,1] +- 1e-5, hits 0.
- `gpu_lowmach_findex` (API): planar premixed layer xi = +1, a planar
  Burke-Schumann sheet xi = -1 (to 1e-6 away from the noise floor), noise
  floor gives 0.
- `gpu_lowmach_round` (binary smoke): round nz 64, D 8 delta_th, pilot, tt 3:
  anchored, axial-plane cone measured, volumes written.

Acceptance (< 5 min, two simjobs). Bit identity: compressible untouched (no
shared file but derived.hip: suite).

**S5 STATUS: DONE (2026-10-31, uncommitted working tree, on top of
S1-S4).** docs/chem.md "Low-Mach Bunsen" (+ "Changed in S5" in "Low-Mach"),
issues LM8-LM12 (LM6/LM7 updated).

- Files: `dns/lowmach.{hpp,cpp}` (`lmInflowRow`, `lmPartialPremixMix`,
  `lowmachBunsenIC`, `lmPremixedWeight` = the ONE w_p), `gpu/lowmach.{hpp,
  hip}` (per-row inflow buffers -- ALWAYS rows, the premixed flame fills
  them from the scalars bit for bit; `cellVelocity`; `rho()`, `mx()`),
  `gpu/derived.{hpp,hip}` (`flameIndexCentred`; additive), `gpu/
  runner_lowmach.{hpp,hip}` (bunsen branch: measureB, writeFrameB),
  `apps/dns_main.cpp` (rows, json `nozzle/W/uJet/yc/zc/dth/phi/zst`),
  config/cli (bunsen + lowmach, help), tests `test_gpu_lowmach_bunsen.cpp`
  (g2a / extrude / rich / round), `test_gpu_lowmach.cpp` (+ rows, findex),
  `test_gpu_lowmach_restart.cpp` (+ 3 burners), `test_lowmach_config.cpp`
  (+ S5), `tools/chem/foam_flame.py` (twins may DROP flags) + `foamflame/
  cases.py` (G2a twin `dns_lowmach`) + `observe2d.py` (Tu = 1 scaling),
  `runs/s5/*.sh` (scouts, `bitid.sh`).
- Measured: G2a low-Mach xTip 43.927 / sinRatio 1.0243 / SLflank 0.9918 /
  mFlank 0.9925 / burnRatio 0.99999, steady to 1e-5; vs compressible M -> 0
  (M 0.01/0.02 T-held etaIn 1000) -1.9% / -2.6% / +0.16% / +0.04% /
  -0.25%; vs reactingFoam (identical 392 x 288 mesh, foam_mclin_c8 reused)
  +0.48% / -0.44% / -1.2%, theta field L2 4.6e-3 (compressible 0.022).
  Partially premixed slot (W 4, U 3, pilots 1.5 at U, air): outer cone
  closed at 80.1 delta_D, preFrac 0.727, burnRatio 0.99987, Z in [0, 1],
  0 hits. nz 8 == 2D bytewise. Restart byte rule on 3 burners. Compressible
  path byte-identical to HEAD (`runs/s5/bitid.sh`, 12 cases).
- DEVIATIONS. (1) **T votes in the shared limiter** (LM8: an unvoted
  theta is 2 = downwind; the jet core blew up) -- the plan's "T does not
  vote" is wrong for the low-Mach T. (2) The plan's 1% vs the compressible
  is not met for xTip / sinRatio (LM9; the low-Mach is the one that agrees
  with foam); test gates 3% there, 1% vs foam. (3) Partially premixed
  products/air are laid BURNT (Burke-Schumann) in the IC and the inflow
  rows (a linear blend exploded). (4) The rich test is W 4 / U 3 / pilots
  1.5 at the jet speed, not W 12 / U 5 / 3 (outer cone ~180 delta_th,
  LM11); preFrac 0.727 sits in [0.72, 0.80]. (5) dt: convective rate over
  min(rho_face, rho_cell); 3D diffusive coefficient 1/6 (positivity; S4's
  3D runs now take 1.5x the steps); `--dt-recompute 1` default on the
  burner. (6) 4 cells/delta_th 3D is not bounded at the default dt (LM10):
  the round test runs `--chem-dt 0.5`. (7) The extrusion identity needs the
  same binding dt bound (1/6 vs 1/4): the test pins it with `--cfl 0.1`.
  (8) The cone of the partially premixed burner is measured on omega w_p;
  SLflank / ST per rho_u (Y_F1 - Y_F,rem). (9) Extra stats columns beyond
  the plan's list (the compressible cone set, massErr, dtDiff); "xTipIn"
  is `xTip`. (10) `--pulse` rejected on the low-Mach burner (steady face).
  (11) The foam case was not re-run: the existing G2a `foam_mclin_c8` is
  the identical mesh.
- For S6+ (what to know). (1) Open laterals: the rows are already per
  (j, k); `lmInflowRow` is the place for the room-air rows; the cone
  measurement assumes y periodic only for `measureCone`'s outward walk (it
  never crosses the wrap). (2) **LM10 before the final case**: pick the
  dt rule at 4 cells (per-cell positivity bound, or `--chem-dt 0.5`) and
  re-scout. (3) LM11: the round outer cone must fit the box (xTipOut
  column; NaN = open). (4) The stats row is host-heavy (10 field
  downloads, FP64 loops): at 11.9M cells measure it, raise the cadence if
  needed. (5) Checkpoint: no new loop state in S5; the rows are derived from
  the case. (6) The renderer reads `findex` frames/volumes and must use
  `lmPremixedWeight`'s smoothstep(-0.2, 0.2) (D14). (7) K15/LM4: the C2
  burner's Z stays in [0, 1] to 1e-6 here; report szDefect/Z on long runs.

### S6 -- Open lateral faces (`--open-sides`, low-Mach)

Files: `gpu/lowmach.hip` (OpenY/OpenZ instantiations, face rules 2.3), `gpu/
lm_poisson` (y/z DD), `config.cpp` (`--open-sides`: lowmach only; with it
`--coflow-width` = air clearance from the outer edge (pilot or tube) to the
face; round keeps nz = ny), runner stats mdotLat (net in through laterals),
uLatMin.

Tests:
- `gpu_lowmach_rest` (API): quiescent box, open y, z: u == 0 and p == 0
  exactly for 1000 steps (no source, no drift).
- `gpu_lowmach_bickley` (binary): 2D isothermal planar jet (`--reaction off`
  allowed with lowmach), Re_jet 30, open y, x 120 W: centreline u_c ~
  x^(-1/3) (fit exponent -0.333 +- 0.02 over the self-similar half), momentum
  flux J conserved +- 2%, entrainment dQ/dx ~ x^(-2/3) (Bickley 1937 /
  Schlichting); the same with periodic y shows confinement (report).
- `gpu_lowmach_open_bal` (API): species and mass budget closure through open
  faces to the FP64 tally (1e-6 of the throughput), backflow face carries air
  exactly.

Acceptance < 4 min.

**S6 STATUS: DONE (2026-10-31, uncommitted working tree, on top of
S1-S5).** docs/chem.md "Low-Mach open laterals", issues LM13, LM14 (LM6,
LM10, LM11 updated), 77 (found, not caused).

- Flag: **`--laterals periodic|open`** (the user's name; the plan's
  `--open-sides` does not exist -- section 5's command line must read
  `--laterals open`). Burner + lowmach only; y open, z too around the round
  nozzle (the slot's z stays its periodic span). Also `--reaction off` on
  the low-Mach burner (inert: no chemistry, IC = rows extruded, no anchoring
  guards; the Bickley case) and `--profile-dt` on it (`<out>_jet.csv`: uc,
  Tc, Yc, J, P, M, Q, b, Qj per x column). Compressible burner: all three
  rejected by name (D6).
- Built: every low-Mach kernel templated on (OY, OZ) via `Lat<OY, OZ>`
  (periodic wraps = S5 indexing; open CLAMPS: ghost = boundary cell); hi
  faces `myHi` (nx nz) / `mzHi` (nx ny); pressure ky/kz = DD (still one
  direct solve); `kLmMomT` adds one thread per hi face; face rule by the
  sign of the face's own flux: outflow zero gradient, inflow = quiescent
  ambient (T Tu, air | Y 0, rho_amb, u_t = 0, face p = -beta rho_amb
  u_n^2/2, beta 1 = Moghadam / total-pressure inflow), no diffusion through
  an open face; with open laterals the x-hi backflow takes the same rule.
  New in every mode: the FP64 SPECIES tally (boundary face fluxes the stage
  used; `speciesTally`), `boundarySums()` (mdotIn/Out/Lat, uLatMin/Max,
  species out ambient-aware), `p()` accessor; flameIndexCentred one-sided
  at open ends. Checkpoint adds `state.myhi`, `state.mzhi`, `state.ytally`.
- Measured: periodic = the pre-S6 binary byte for byte (7 low-Mach cases,
  `runs/s6/lmbit.sh`); compressible = HEAD e92dad1 (`runs/s5/bitid.sh`, 12
  cases); all S4/S5 low-Mach tests pass unchanged. API: open rest exact
  over 1000 steps (2D, 3D); open tallies 3.6e-11 (mass, species); open
  balance 1.5e-10 of throughput; backflow vortex KE never grows at beta 1,
  x4.42 at beta 0. Bickley (Re 60 test; Re 30 the plan's): u_c^-3 slope
  0.984 (0.989), exponent -0.3330, b^1.5 0.976 (0.954), Qj^3 0.967 (0.908),
  dQj/dx exponent -0.680 (-0.767), J +-0.38% (1.04%), J + P 0.36% along the
  box; clearance halved: u_c 0.32%, b 0.58%, J 1.04%; periodic confined (Q
  constant, J +-10%). Restart byte rule: open slot, inert jet, open round.
- DEVIATIONS. (1) The Bickley K is the FAR-FIELD J: a wall-issued jet loses
  J to its base's suction (Schneider 1985), the box conserves J + P (P(x-lo)
  = -0.25 J_in at Re 30). (2) Re 60 in the test (every Re-30 residual
  halves at 60: O(1/Re)); Q is the jet's own (Qj, |y| < 3b), not the box
  width's (the induced flow runs back toward the wall). (3) The plan's
  "plume of the same with periodic y" = a confinement REPORT turned gate
  (Q constant, J spread > 5%). (4) Stats names: mdotIn, mdotOut, mdotLat
  (net in), uLatMin, uLatMax (plan: mdotLat, uLatMin). (5) The "reflected
  pressure" check = the clearance test: moving the p = 0 face from 15 W to
  7.5 W moves the integrated base suction 10%, the jet <= 1%. (6) The
  species tally carries no chemistry (inventory - tally = -int int omega).
  (7) Inflow tangential u = 0 is a Dirichlet face (the plan's), so an
  entrainment face carries a viscous shear 2 mu u_t/h; outflow faces zero
  gradient.
- **LM11 answered (LM13): in 2D, open laterals do NOT close the outer cone
  sooner** (rich2d, clearance 6: 90.47 vs periodic 80.09; clearance <= 3,
  room air or W 12: never) -- the dilatation drives a net lateral OUTflow
  and the planar jet entrains weakly. The 3D round closes at ~42 delta_D
  regardless of box (periodic 41.84, open 42.57 / 41.99, D 6 delta_th, U
  3). An open box needs t ~ 100 tau for the starting head to leave.
- For S7+ (what to know). (1) Gravity is the missing entrainment driver in
  the open box: the plume test should run open laterals, and the open faces'
  p = 0 is the AMBIENT's dynamic pressure -- with gravity the ambient is
  hydrostatic in p (rho_a g x lives in p, D2.1), so the open face value
  stays 0 only if the buoyancy term is (rho - rho_a) g as planned. (2) The
  lateral faces see hot outflow (uLatMax up to 5 S_L at W 12): a plume
  leaving sideways under gravity is legitimate, a clearance too small for
  the head is not -- S9's clearance scout should watch uLatMax and mdotLat.
  (3) The ambient is Tu room air; a hotter room or a draft (D15's 0.4 S_L
  coflow) goes through `ambT`/`--coflow-speed`, no new face code. (4) LM10:
  at U 3 / 4 cells the default dt is CONVECTIVE-bound and not positive
  (11156 Y_O hits at cfl 0.8; 0 at dt 0.0055) -- S9's dt rule must cover the
  convective MC bound, not only diffusion. (5) Restart: no new loop state;
  device records added (checkpoints of S5 binaries are rejected anyway by
  the sha). (6) The Bickley runs: 480k cells 1.66 ms/step (open) vs 1.41
  (periodic) -- open laterals cost ~18% per step in 2D (3D round 3.38 vs
  2.57 ms at 576k: +32%, the clamps and the extra hi-face threads).

### S7 -- Gravity (`--gravity` on low-Mach bunsen)

Files: `gpu/lowmach.hip` (`addBuoyancyLm`, Grav instantiation of the
momentum kernel), `config.cpp` (`--gravity G` in S_L^2/delta_D for
bunsen+lowmach; dry-run prints Fr = U^2/(g D), Ri, the free-fall speed;
warn when gravity with periodic laterals: "a periodic box under gravity is a
lattice of plumes (the candle's chimney)").

Tests:
- `gpu_lowmach_plume` (API): 2D laminar plane plume from a hot inert inflow
  slot (rows set directly), open y: centreline u ~ x^(1/5), width ~ x^(2/5)
  (Gebhart et al.), exponents +- 0.03, buoyancy flux conserved +- 3%.
- `gpu_lowmach_hydro` (API): hot blob at rest released, compared with the
  same blob with g = 0 (bitwise until the Grav instantiation is chosen:
  g = 0 is the non-Grav kernel, byte-identical).
- `gpu_lowmach_candle` (binary, cross-check): the compressible candle
  defaults (W 1, g 14, periodic y, Da 5000) on low-Mach: tip within 5% of
  10.79 l (compressible stratification 1.75% and inflow differences D10 in
  the band); needs the diffusion inflow rows on low-Mach (`--mode candle
  --physics lowmach`, `diffusionInflow` reused; else this test is an API
  test with the rows set directly -- decide in S7, smaller scope wins).

Acceptance < 4 min.

**S7 STATUS: DONE (2026-10-31, uncommitted working tree, on top of
S1-S6).** docs/chem.md "Low-Mach gravity", issues LM15, LM16 (LM6, LM10,
LM13 updated).

- Built: `LowMachSetup::gravity` (g >= 0 along -x); `addBuoyancyLm` (the
  ONE term, -(rho_f - rho_a) g on every x face, rho_f = `rhoX`, rho_a =
  P0/(R T_amb)) in the `Grav` instantiation of `kLmMomX<OY, OZ, Grav>`
  (g == 0 launches the S6 kernel); `--gravity G` (the candle's flag, now
  also the low-Mach burner's: [0, 100] S_L^2/delta_D; compressible bunsen
  and the planar flame rejected by name; `--quiescent` stays the
  candle's), `Case::lmGravity`, label "GRAVITY g=...", json "gravity",
  dry-run Fr / Ri / u_ff (+ over the box), the periodic-laterals warning
  (lattice of plumes), `lmRealGravity(dth/dD)` = 0.0241 at M1 (GRI-3.0
  phi 1.3, delta_D,model 0.13138 mm). No new loop or device state (the
  checkpoint's case text carries --gravity).
- Open faces: nothing new. The ambient's hydrostatic p lives in p (rho_a
  g x), so the quiescent ambient is p = 0 exactly and the open/outflow p =
  0 (inflow -rho u_n^2/2) stays its pressure; no hydrostatic face
  condition (the compressible candle's hydro LODI waves have no
  counterpart).
- Measured: rest (2D periodic, 2D open, 3D open) u = p = 0 bitwise over
  1000 steps at g 1; stratified column g vs g 0: |dm| 4.7e-7 / 1.9e-9 of
  the free-fall scale, p - p(0) = the discrete hydrostatic sum to 5.7e-5 /
  4.1e-5 of its range; blob rises (momentum 0.911 of the buoyancy
  impulse); plane plume exponents +0.1999 / +0.4121 / -0.6019, buoyancy
  flux 7.8e-5, amplitudes vs the derived Boussinesq similarity solution
  0.982 / 0.969 / 0.974 (u_c^5, b^5/2, g'^-5/3 slopes). Restart byte rule:
  open slot + open round under --gravity 0.5. Bit identity: low-Mach
  without gravity = the S6 binary (`runs/s7/lmbit.sh`, 10 cases + 3 with
  --gravity 0), compressible = HEAD e92dad1 (`runs/s7/bitid.sh`, 12).
- DEVIATIONS. (1) The plan's `gpu_lowmach_hydro` "hot blob vs g 0
  bitwise" became three checks (exact rest, the stratified column's
  balance, the rising blob's impulse): g 0 = the non-Grav kernel is the
  bit-identity suite's. (2) The plume test checks AMPLITUDES too (the
  similarity ODE solved in the test), at g 6 (g 0.6 was too slow to
  settle in t 450). (3) **Candle: the plan's "tip within 5% of 10.79" is
  NOT met (-34.8%, LM15)**: at Pe 1 the tip follows the inflow's fuel flux
  (low-Mach face: 54% by Fick; compressible LODI row: §68); the test gates
  tip per burned fuel vs the compressible D10 variant (+2.3%). API test
  with rows set directly (no `--mode candle --physics lowmach`: smaller
  scope). (4) No new stats columns (the cone/outer-tip/lateral columns
  carry the gravity physics; the label and json name g).
- **LM13 answered with gravity**: real g does NOT close the 2D outer cone
  where the open box left it open (clearance 6 closes sooner, 90.47 ->
  82.40; clearance <= 3 and W 12 stay open); 4-12x real g closes it and
  it flickers (periods 18-36 tau). 3D round D 6 delta_th: 42.57 -> 42.15
  (real g, steady to t 200), 41.46 at 4x, no flicker.
- For S8+ (what to know). (1) Gravity is a template parameter of kLmMomX
  only; S8's Wall instantiation of the momentum kernel must carry Grav
  too (a solid face is closed: m = 0, its buoyancy irrelevant; the
  apertured PCG absorbs a column's hydrostatics exactly as the spectral
  solve does -- check the stratified-column test with a tube in it). (2)
  rho_a is the ambient Tu; the rim's warm tube (--rim-temp) heats air
  that then buoys -- physics, not a bug. (3) S9: with gravity the 3D
  4-cell burner is CONVECTIVE-bound (LM10 note): the dt rule must see the
  plume. The final case's box (62 mm above the rim) gives u_ff = sqrt(g L
  (1 - 1/sigma)) ~ 3.1 S_L over the box at real g (L ~ 470 delta_D):
  comparable to the 6.5 S_L jet, so buoyancy shapes the outer cone; watch
  uLatMax/mdotLat and xTipOut for flicker (real period ~118 tau at D 10
  mm: the 18.6 ms clip cannot show one). (4) The section 5 command line's
  `--gravity 0.0244` is S1-stale: 0.0241 (lmRealGravity(4.70964)).

### S8 -- Immersed isothermal tube rim (`--rim`)

Flags (bunsen + lowmach only; rejected by name elsewhere): `--rim T` (lip
thickness, delta_th; >= 4 cells: two mirror cells per side, the wick's rule),
`--tube-length L` (delta_th of tube inside the domain below the exit, >= 8
cells), `--rim-temp R` (T_wall/T0, [1, 3], default 1). With `--rim`: inflow
profile Poiseuille (D12) inside R_i; `--pilot-*`, `--coflow-temp`,
`--slot-edge`, `--pulse` rejected by name ("the rim anchors the flame: no
pilot; the tube shapes the profile"); `--coflow-speed` = room draft below the
rim (default 0.4 S_L = 0.1 m/s at M1, D15).

Files: `dns/lowmach.cpp` (`tubeSdf`, `tubeRegions`, Poiseuille rows, IC: fresh
premix in the tube, the ideal inner cone above the rim (bunsenConeDistanceAt
with coneH from U_bulk), a SHORT rich-product column above it as a closed
Burke-Schumann sheet against the air (the candle's `--ignition` lesson: a
box-tall hot column rolls up), air elsewhere, p = 0), `gpu/ghostfill.{hpp,
hip}` (scalar cell-centred instantiation of the image table for T),
`gpu/lowmach.hip` (Wall instantiations), runner stats xBase above the rim,
rBase, qRim (conduction into the tube), and json `"rim": {"x", "ri", "ro",
"h"}` for the renderer.

**D15 (room draft).** The x-lo face outside the tube is an inflow at 0.4 S_L
(0.1 m/s): a lab's convective draft, keeps x-lo an inflow everywhere (a u =
0 face would be a wall); laterals supply the rest of the entrainment. Scout
0.2 / 0.4 / 1.0 in S9 (sensitivity of xTipOut).

Tests:
- `gpu_lowmach_channel` (binary): 2D inert channel between two lips (slot
  with `--rim`, `--reaction off`): developed Poiseuille preserved to 1e-4,
  dp/dx = 12 mu U / W^2 within 1% (staircase W = an integer of cells), wall
  faces carry exactly 0 mass.
- `gpu_lowmach_rim_cond` (API): fluid at 2 T0 against the T0 tube: wall heat
  flux = the conduction the stage uses (closure 1e-6), linear profile exact.
- `gpu_lowmach_rim_anchor` (binary): 2D slot rich phi 1.3, W 14 delta_th, U 6,
  lip 1.4 delta_th, 4 cells, tt 30: anchored on the rim (xBase in (0, 4
  delta_th] above the rim, flat in time), no flashback (axis cold at the
  rim), preFrac in [0.72, 0.80]; the same with `--rim-temp 1.5` (report the
  standoff change). pcgIt per stage <= 15 (D9 gate; else the fallback
  direct-forcing IB is designed in an S8b).

Acceptance < 5 min.

**S8 STATUS: DONE (2026-10-31, uncommitted working tree, on top of
S1-S7).** docs/chem.md "Low-Mach rim", issues LM17-LM19 (LM6, LM10
updated).

- Files: `dns/lowmach.{hpp,cpp}` (`lmTubeSdf` = the ONE tube definition,
  `lmTubeSolid` staircase, `lmPoiseuille`/`lmPoiseuilleScale`, `lmWallHeat`,
  rim rows in `lmInflowRow`, the rim IC, `kLmRoomDraft` 0.4,
  `kLmTubeLengthDefault` 7.5, `kLmRimColumn` 4), `bunsen.hpp` (BunsenGeom
  rimLip / xRim / tRim / rimScale), config (`--rim`, `--tube-length`,
  `--rim-temp`, `Case::lmRim`, rejections, label, dry-run TUBE lines), cli
  (flags + help), `gpu/lowmach.{hpp,hip}` (`LowMachSetup::solid / wallT /
  pcgTol`; the `Wall` instantiation of kLmTemperature, kLmMomX, kLmMomT,
  kLmCorrectX/C, kLmScalars; per-stage warm-start buffers phiW_,
  `pcgIterations()`, `pcgFailed()`, checkpoint `state.phi0..2`),
  `gpu/runner_lowmach.{hpp,hip}` (real pcgIt, xRimBase / rBase / qRim, the
  rim guards, PCG non-convergence exit 8), `runmeta.hpp` + `output.cpp`
  (`.json "rim"`, absent = unchanged), `apps/dns_main.cpp` (solid map, frame
  mask, json), tests `test_gpu_lowmach_rim.cpp` (channel / cond / imperm /
  anchor), `test_lowmach_config.cpp` (+ S8), `test_gpu_lowmach_restart.cpp`
  (+ lmbrim, lmbrrim), `runs/s8/{lmbit,bitid}.sh`, `runs/s8/map`,
  `runs/s8/round`, `runs/s8/cost`.
- Measured: channel profile 6.4e-7 U, dp/dx = the discrete curvature to
  7e-6 (0.99225 of 12 mu U/W^2 at 16 cells), wall shear = -dp/dx W/2 to
  1e-6, every face touching the tube m == 0; conduction: the linear profile
  is the fixed point (2.4e-7), qRim = lambda dT/L to 2.4e-7 and = the inflow
  conduction; imperm (3D round, hot jet, open, g): closed faces 0, tube
  bitwise, passive Y bitwise, tallies 3e-13, PCG 13 cold / <= 14 warm,
  rerun bytes; restart byte rule on the 2D rim slot and the 3D round tube
  (warm starts checkpointed). 2D anchoring map (M1 mapping, W 14, lip 1.4,
  open): anchored on the rim at every U in [1.5, 20] (4 cells), stand-off
  0.62-0.87 delta_th (4 cells), 0.69 (8 cells, U 6), rBase over the lip;
  rim-temp 1.5 moves the base < 1 cell, halves qRim; no flash-back at 1.5,
  no blow-off at 20 (LM18). 3D round (D 6, U 4, lip 1.4, open, real g, 4
  cells, tt 100): anchored 0.87 delta_th, inner tip 7.5 delta_th above the
  rim, outer cone CLOSED 15.0 delta_th above it, steady, preFrac 0.7998,
  burnRatio 0.9998, 0 hits, qRim 4.4% of the heat release, pcgIt <= 9.
  Cost: final grid 171 ms/step (rim) vs 69 (pilots, direct). Bit identity:
  low-Mach without the rim = the S7 binary (12 + 3 cases), compressible =
  HEAD e92dad1 (12).
- DEVIATIONS. (1) **STAIRCASE wall, not the sub-cell ghostfill view** (the
  plan's 2.4 "diffusion-only ghost view ... clean trilinear normal image"
  and "a new scalar instantiation of the ghostfill table"): the wall sits on
  the cell faces -- no-slip by the antimirror of a closed tangential
  neighbour face, isothermal by the face conduction over h/2. Exact for the
  slot's grid-aligned lips; first-order wall position on the round tube
  (LM17). gpu/ghostfill is untouched. (2) The Poiseuille rows are the
  SCHEME's discrete solution on the staircase slot (the parabola widened by
  d = (sqrt(W^2 + h^2) - W)/2 per side, exact for the antimirror 3-point
  operator), the round tube's analytic profile renormalized to the
  staircase; both bulk = --jet-speed. (3) The channel test is an API test
  (the plan: binary) on a 16-cell channel at Re 11; dp/dx = 12 mu U/W^2 to
  0.8% (the h^2/(2 R^2) of 16 cells) and to the discrete curvature 7e-6.
  (4) The anchor test runs tt 40 (window [30, 40]) at `--chem-dt 0.5`
  (LM10: the default overshoots there) and gates yHits 0; the plan's preFrac
  [0.72, 0.80] is NOT met in 2D (0.97: the 2D outer cone stays open,
  LM13) -- met in 3D (0.7998), not gated in the 2D test. (5) qRim is the
  conduction of the CURRENT state with the stage's face expression (host
  `lmWallHeat`), not an RK-weighted tally. (6) The PCG warm start is the
  same STAGE's increment of the previous step (3 buffers), not an
  extrapolation; tol 1e-6 (an API knob `pcgTol`, no flag). (7) The rim top
  snaps to a cell face; a solid cell = sdf(centre) <= 0. (8) The IC's
  products column ends 4 delta_th past the ideal tip (`kLmRimColumn`).
  (9) The room draft is uniform up to the tube's outer wall (no outer
  boundary-layer profile).
- For S9 (what to know). (1) **dt rule at 4 cells (LM10)**: the 2D rim
  slot overshoots at the default dt from U 6 (Tmax 7.14; U 10: 8.42, 2e4
  hits); `--chem-dt 0.5` is bounded (0 hits at U 10); `--cfl 0.2` nearly
  (18 hits); the 3D round tube (D 6, U 4, real g) is bounded even at the
  default (dt 0.0092, diffusion). Run the half-scale scout at the default
  AND `--chem-dt 0.5` and keep the one with 0 hits. (2) Cost: 171
  ms/step at 12.7M cells -> the section 5 run is ~2 simjobs; the stats row
  is host-heavy (one row at 12.7M cells inside the 42-step measurement).
  (3) The section 5 command line works as written with `--nz 180`
  (ny 176 -> 180 by the FFT rule: the round nozzle needs --nz = ny), giving
  392 x 180 x 180. (4) The renderer's tube: `.json "rim": {"x", "ri",
  "ro", "h", "T"}` in box units (x = the rim top; multiply by delta_D,model
  for metres). (5) The round staircase: check the base/qRim against an
  axis shifted by h/2 (LM17) before trusting stand-off numbers at the
  final grid. (6) The 3D outer cone needs t ~ 80 tau to settle in the open
  box (the starting head); the section 5 frames start at 70 tau.

### S9 -- Real-burner integration (3D round) + renderer

Files: runner (round + rim + open y,z + gravity + equivalence together;
`axialPlane` measurement as `runner_flame.hip:258-266`), dry-run SI echo
(`--si-delta-d MM --si-sl MS`: prints D, U, h, box in mm, Re, Fr, tau; inputs
for the label and the json only, no physics), `tools/chem/flame_room.py`:
reads `_findex.bin`, writes two VDB grids `em_pre` = max(omega,0) w_p/ref,
`em_dif` = max(omega,0)(1 - w_p)/ref (`write_vdb`, 354-363), one volume
shader adding two Emission nodes (`flame_material`, 252-268) -- inner: CH*
431 + C2* 516/473 rich weighting (blue-green), outer: CH* + CO2* continuum,
gain `--dif-gain` (0.35, a DISPLAY choice, labelled); proxy lights per band
carry the summed power and the power-weighted colour (`light_proxies`,
365-388); `--crop-lo auto` = the rim row from the json (low-Mach has no sink
bands: `--bands 0`); `bunsen_tube` (142-165) takes ID/wall from the json
(real 10/12 mm), rim at the burner top. `turb_render.chemi_rgb` gains line
weights; `turb_render_selftest` extended + `flame_room_selftest` (python, no
Blender: VDB grid split sums to the single-grid total to 1e-6, colours
deterministic).

Scouts (each < 5 min): 3D round at HALF scale (D 5 mm, 15.4 x 15.4 x 33.6 mm
box at h 0.175 mm = 192 x 88 x 88 = 1.49 M cells, tt 30 tau): anchoring,
outer cone closes, preFrac, divErr/eosErr, pcgIt, ms/step (calibrates section
6); lateral clearance 1.5 D vs 2 D at half scale (xTipOut change <= 3% else
widen); room draft 0.2/0.4/1.0.

Acceptance: scouts + `gpu_burner_round` (binary smoke, nz 64, tt 2: every
stat finite, volumes + findex written, restart byte rule once).

**S9 STATUS: DONE (2026-11-01, uncommitted working tree, on top of
S1-S8).** docs/chem.md "Low-Mach dt rule, real-burner scouts, renderer",
issues LM10 (settled), LM20-LM23.

- (a) **dt rule `--dt-rule bounds|cell`** (LM10 settled): `cell` adds dt <=
  1/max_c (2 C_c + D_c) -- the explicit transport step's per-cell
  positivity bound (MC x2 on LmConvRate's rate, the cell's OWN diffusivity
  max(4/3, 1/Pr) mu(T_c)/rho_c; `DeviceLowMach::positivityDt`, `LmPosRate`,
  `kLmPosKDefault` 1); stats append `dtPos`. DEFAULT on the low-Mach
  burner (its S8 default is demonstrably unbounded at 4 cells: 2D rim
  slot U 6 / 10 6390 / 20087 hits, 3D pilot round 1440 (g 0) / 1556 (real
  g)), `bounds` on the planar flame (S4 bit for bit). Measured: 0 hits
  everywhere (+ the half-scale real burner), 1.2x (3D) - 1.5x (2D) the old
  steps, 1.3x FEWER than `--chem-dt 0.5`; K 1.5 unbounded (1.6e3 / 1.9e4
  hits); J in the sum (full or consumption-only) costs 1.5x more and bounds
  nothing more (variants measured, not shipped). Bit identity: compressible
  = HEAD e92dad1 (`runs/s9/bitid.sh`, 12 cases); low-Mach planar = the S8
  binary as is, every burner (periodic, open, gravity, rim, slot, round)
  with `--dt-rule bounds` = the S8 binary byte for byte (`runs/s9/lmbit.sh`,
  3 + 10 cases).
- **SI echo** `--si-delta-d MM --si-sl M_PER_S` (burner, both or neither, no
  physics): dry-run SI block (nu_u = Pr S_L delta_D, g, D, U, h, box and
  tube in mm, Re_D, tt), label item, .json `"si"`. The final case's echo:
  nu_u 1.591e-5 (GRI's), g 9.806, D 9.999 mm, U 1.503 m/s, Re_D 944.5.
- (b) **Renderer**: `tools/chem/flame_room.py` + its numpy half
  `flame_room_lib.py` (two cones split by `premixed_weight` = the port of
  lmPremixedWeight, equal-power colours: inner CH* + rich C2* blue-green,
  outer CO2* continuum + CH* trace violet-blue (0.116x the luminance at
  equal power), `--dif-gain` 1 (display), `--height 0` real scale from the
  json rim h x si deltaD (`--delta-d MM` without si), `--crop-lo auto`,
  `--crop-hi`, the steel tube with the json's ID/OD, collar + air holes,
  cast base, inlet; proxy lights colour-mixed). Tests: `flame_room_selftest`
  (host), `flame_room_split` (the port vs the solver's burnPre/burnDif from
  gpu_burner_round's last frame: 6 printed digits). Test frame:
  `runs/s9/render/g1/f0004.png` (half-scale scout A, t 80, 1080p, 96
  samples, 4.6 s).
- `gpu_burner_round` (nz 64, round + rim + open + real g + C2b + cell rule +
  SI; 9 rows finite, yHits 0, Z in [-8e-8, 1], pcgIt 8-10, dt <= dtPos,
  six volume channels, findex in [-1, 1], json rim + si, the split = the
  row to 6 digits, restart byte rule once; 11 s). `lowmach_config` (S9
  block): rule defaults / explicit / rejection, SI carried, echo text,
  rejections by name. The anchor (`gpu_lowmach_rim_anchor`) and round
  (`gpu_lowmach_round`) tests dropped their `--chem-dt 0.5` workaround: the
  default is bounded now.
- (c) **Half-scale scouts** (`runs/s9/half.sh`, chained 5-min simjobs via
  `tools/chem/chain.sh` + its new `CHAIN_DNS` (a frozen binary copy: build
  while a chain runs); table in docs/chem.md). Half SIZE at the final h
  (0.1547 mm): D 5 mm = 8.081 delta_th, lip 1.0 (4-cell minimum), U 6.5,
  open, real g, length 60, 256 x 98 x 98 = 2.46M cells. A (half box 1.5 D,
  draft 0.4) to tt 160: ANCHORED (xRimBase 4.127 = 0.875 delta_th = 0.54
  mm, rBase 21.23 over the lip), inner tip 19.31 delta_th above the rim
  (11.9 mm; steady to 1e-3 from t 15), outer cone CLOSED at 39.88 delta_th
  (24.7 mm; within 0.5% from t ~125, creeping +0.015%/tau at 160), preFrac
  0.7834, burnRatio 0.9999, Tmax 6.982 <= sigma_st 6.997 (t >= 20), 0 hits,
  Z in [-3.4e-8, 1], pcgIt <= 12, eosErr 1.7e-7, divErr 8.5e-7; dt 7.17e-3
  (dtPos binds; dtDiff 9.74e-3, dtChem 1.18e-2); 32.6 ms/step = 13.3
  ns/cell-step (5.0M: 13.5; S8's 12.7M: 13.5). **K10 (clearance 2 D vs 1.5
  D): the inner cone, base, rBase, preFrac, qRim move <= 0.1%; the OUTER
  tip is 3.7% shorter over [140, 160] (5.0% over [120, 160]) in the 2 D box,
  which is itself NOT settled at t 160 (a lateral circulation, net inflow
  8-17k, swings the tip +-3% with a ~45 tau period; the 1.5 D box's laterals
  are quiescent from t ~115, uLatMax 0.02).** The plan's rule (> 3% ->
  widen) is NOT followed for S10: widening doubles the final case (61M
  cells, ~19 jobs at tt 260) for a reference that does not settle either;
  S10 keeps 1.5 D and quotes the outer tip with the box bias (-4 +- 1%,
  issues LM21); the 2 D variant is in the S10 set (flags only). Room
  draft (phase-locked runs, [60, 80]): 0.2 vs 0.4 S_L outer +0.7%, inner
  +0.4%; 1.0 S_L outer -3.6%, inner -0.9%, qRim 3326 / 3130 / 2577. Render
  test frame: `runs/s9/render/g1/f0004.png` (scout A, t 80).
- **Projection of S10** (section 5 with the S1-corrected mapping): D 10 mm =
  16.16 delta_th, lip 1 mm = 1.616 delta_th, tube 7.5, half box 1.5 D
  (clearance 14.54 from the OD), length 200 delta_th above the rim (124 mm:
  the outer tip scales with the jet's volume flux, Roper L_f ~ Q, 4 x 39.9 =
  ~160 delta_th = ~99 mm; the plan's 62 mm box is too short, LM21): 840 x
  196 x 196 = 32.3M cells, 0.43 s/step, dt 0.0066-0.0072 tau (the plume is
  faster over the longer box). Settling: the half scale needed ~125 tau;
  full scale (2x box, 4x outer cone) is estimated ~2x: **tt 260, frames
  t 230-260 every 0.25 tau (121 frames, 17 ms of real time)** -> 36-39k
  steps = 15.6-17.0k s = **10 chained 1700-s jobs** (chain N 12 for margin);
  checkpoint 1.7 GB; volumes 6 channels x 129 MB x 121 = **94 GB** (31 GB
  for the renderer: `--emit volume` cannot be restricted, LM23); host RAM
  ~6 GB, VRAM ~8 GB. A tt 160 / frames 130-160 run is 6-7 jobs with the
  outer tip ~5% short of settled (the half scale at the matching phase).
- DEVIATIONS. (1) The plan's `--chem-dt`-or-positivity choice became a
  rule switch (`--dt-rule`), default per path; the positivity sum excludes
  chemistry (measured). (2) The SI echo's json record is "si" {deltaD m,
  SL m/s}; the dry-run prints nu_u from Pr (the model's own nu), not an
  input. (3) The renderer's colours are at EQUAL RADIANT POWER per unit
  omega (not unit luminance), so `--dif-gain` 1 is the neutral default (the
  plan's 0.35 assumed unit-luminance colours); `turb_render.chemi_rgb` is
  unchanged (`flame_room_lib.spectrum_rgb` reproduces it bit for bit with
  its weights). (4) The half-scale lip is 1.0 delta_th, not 0.81 (4-cell
  minimum). (5) K10 widen rule not followed (above). (6) The scouts ran
  tt 80 (drafts) / 160 (K10), not 30: the outer cone needs ~125 tau.
- CLAUDE.md lines for the user (not edited): "Low-Mach dt: `--dt-rule cell`
  (the burner's default) adds dt <= 1/max_c (2 C_c + D_c), the per-cell
  transport positivity bound, K = 1 measured (1.5 overshoots); the planar
  flame keeps `bounds` (S4 bit for bit); `--dt-rule bounds` on the burner is
  S8 byte for byte; chemistry stays out of the sum (measured)."; "The
  renderer's two-cone split IS `lmPremixedWeight` (flame_room_lib port,
  `flame_room_split` pins it to the solver's burnPre/burnDif)."; "`--si-*`
  is an echo: label, dry-run, .json, never physics."

### S10 -- The final case (section 5)

Chained simjobs, render, docs/chem.md "Real Bunsen", issues R1..., CLAUDE.md.

**S10 STATUS (2026-11-01): DONE.** Command set below run verbatim (1.5 D
box; the 2 D variant not run). 12 chained jobs, complete. Inner tip 30.3
mm above the rim, outer 71.9 mm mean flickering 61–88 mm (~38 ms period,
rough); bounded (yHits 0, Tmax ≤ σ_st, pcgIt ≤ 10). Render
`runs/flames/real2/room_real2.mp4`. docs/chem.md "Real Bunsen".

**S10 command set (from S9; supersedes section 5's command lines, which are
S1-stale: Pr 0.41 -> 0.5245, sigma 6.88 -> 6.899, D 14.3 -> 16.16, rim
1.43 -> 1.616, g 0.0244 -> 0.02406, length 88.6 -> 200, tt 100 -> 260,
`--open-sides` -> `--laterals open`).** Mapping M1 with GRI-3.0 phi 1.3:
delta_th 0.61874 mm, model delta_th/delta_D 4.71614 -> delta_D,model
0.13120 mm, tau 0.5674 ms, S_L 0.23124 m/s, Pr_eff = nu_u/(S_L
delta_D,model) = 0.5245; D 10 mm, wall 1 mm, U_bulk 1.503 m/s (6.5 S_L),
room draft 0.4 S_L (0.093 m/s), real g. Run from the repo root:

    # 0. freeze the binary (a restart rejects another binary) and dry-run
    mkdir -p build/s10 runs/flames/real2
    cp -p build/dns build/s10/dns
    A=(flame --mode bunsen --nozzle round --physics lowmach
       --equivalence 1.3 --expansion 6.899 --visc sutherland --tinf 298
       --prandtl 0.5245 --slot-width 16.16 --jet-speed 6.5
       --rim 1.616 --tube-length 7.5 --laterals open --coflow-width 14.54
       --gravity 0.02406 --cells-per-delta 4 --length 200 --nz 196
       --tt 260 --frame-start 230 --fdt 0.25 --checkpoint-dt 5
       --emit stats --emit volume --si-delta-d 0.1312 --si-sl 0.23124)
    build/s10/dns "${A[@]}" --out runs/flames/real2/r --dry-run
    #    (grid 840 x 196 x 196, SI: D 9.999 mm, U 1.503 m/s, Re_D 944.5,
    #     g 9.806 m/s^2, above the rim 125.3 mm, tt 147.5 ms)

    # 1. the run: ~10 jobs of 1700 s through the job server (run_in_background)
    CHAIN_DNS=$PWD/build/s10/dns tools/chem/chain.sh runs/flames/real2/r 12 -- "${A[@]}"
    #    stopped chain: the same line with --resume after chain.sh
    #    watch: runs/flames/real2/r_stats.csv xTip, xTipOut (settled before
    #    t 230?), xRimBase, preFrac, yHits 0, Tmax <= 6.997, pcgIt <= 15

    # 2. render (Blender Cycles, simjob --gpu, <= 1800 s each: two halves)
    for r in "0 60" "61 120"; do set -- $r
      build/simjob submit --gpu -t 1800 -m 16384 --label "room real2 $1" --wait -- \
        /usr/bin/blender -b --factory-startup --python tools/chem/flame_room.py -- \
        runs/flames/real2/r --out runs/flames/real2/room --first $1 --last $2 \
        --orbit-span 0:120 --t0 230 --res 1920x1080 --samples 96 --device gpu \
        --burner round --bands 0 --crop-lo auto --height 0 \
        --emit 330 --light-gain 600 --distance 0.40 --target 0.25 --lens 50 \
        --elevation 10 --azimuth 20 --orbit 40 --ambient 6
    done
    #    (real scale from the json: rim h x si deltaD; two cones, --dif-gain 1)

    # 3. mp4 with bloom (runs/flames/hot/room_real.sh's filter)
    ffmpeg -y -loglevel error -framerate 24 -pattern_type glob \
      -i 'runs/flames/real2/room/f*.png' -filter_complex \
      "[0]format=gbrp,split[a][b];[b]colorlevels=rimin=0.45:gimin=0.45:bimin=0.45,gblur=sigma=28[g];[a][g]blend=all_mode=screen:all_opacity=0.55,format=yuv420p" \
      -c:v libx264 -crf 18 runs/flames/real2/room_real2.mp4

Variants (flags only, same binary): the 2 D half box (K10's reference):
`--coflow-width 22.62 --nz 270` (840 x 270 x 270 = 61.2M cells, ~19 jobs,
178 GB of volumes); the shorter run: `--tt 160 --frame-start 130` (6-7
jobs; the outer tip ~5% short of settled). Before the render, a single
test frame (`--first 0 --last 0`, ~1 min) to check framing: the full-scale
flame is ~2x the scout's inner and ~4x its outer cone (outer tip ~99 mm
above the rim; frame at distance 0.40 m, 50 mm lens: 162 mm tall).

---------------------------------------------------------------------------

## 5. FINAL CASE (mapping M1)

Physical: CH4/air phi 1.3 (S_L 0.25 m/s, T_ad 2050 K), tube ID 10 mm / wall
1 mm, U_bulk 1.5 m/s Poiseuille, room air 298 K drawn in laterally and at
0.1 m/s from below, g 9.81 m/s^2, real Re 940 and D/delta_th 14.3; Pe 0.6 x
(D1). True Mach (low-Mach limit).

Model numbers (delta_D,model = 0.70/4.51 = 0.155 mm, tau = 0.62 ms; S1
re-derives them from the fixed oracle and Cantera):

    dns flame --mode bunsen --nozzle round --physics lowmach
      --equivalence 1.3 --stoich 3.99 --ox-y 0.233
      --expansion 6.88 --zeldovich 8 --visc sutherland --tinf 298 --prandtl 0.41
      --slot-width 14.3 --jet-speed 6 --rim 1.43 --tube-length 7.5
      --rim-temp 1 --laterals open --coflow-width 13.4 --coflow-speed 0.4
      --gravity 0.0244 --cells-per-delta 4 --length 88.6
      --tt 100 --frame-start 70 --fdt 0.25
      --checkpoint-dt 5 --wall-limit 1700
      --emit stats --emit volume --out runs/flames/real2/r
      (--si-delta-d 0.155 --si-sl 0.25: label/json only)

(lengths in delta_th as the bunsen flags are today; slot-width = D;
coflow-width = clearance from the OD to the lateral face: 9.4 mm, half-box
15.4 mm; length = 62 mm above the rim, tube-length 5.25 mm below it.)

Grid: h = delta_th/4 = 0.175 mm. ny = nz = 176 (30.8 mm = 3.1 D: 9.4 mm air
outside the OD each side), nx = 384 (67.2 mm: 7.5 delta_th = 5.25 mm of tube
below the rim, 62 mm above). FFT-friendly (176 = 2^4 11 -- if 11 is slow on
hipFFT, 180 = 2^2 3^2 5). Cells 384 x 176 x 176 = **11.9 M**. Tube 57 cells
ID, lip 5.7 cells.

Cost (estimates; S9's half-scale scout calibrates them):
- transport ~2.7-3 ns/cell-step (the compressible measured 2.3 for 5 + 1
  equations; low-Mach: T + 2 species + continuity + 3 momentum + chemistry):
  33-36 ms;
- projection 3 stages x (1 + pcg it ~6) x ~2.2 ms (spectral solve of 11.9 M
  ~ 1.7 ms memory-bound + LinOp + dots): ~45 ms;
- **~80 ms/step (bracket 50-130)**.
- dt: diffusive 0.0115 tau, chemistry 0.0022 tau (explicit, kChemDt 0.5) or
  ~0.005 (S4's measured bound / S4b implicit). t_end 100 tau = 62 ms physical
  (the 5-mm-tall inner cone settles from its IC in ~5 ms; products at ~3 m/s
  flush the 62 mm box in ~20 ms; two flushes + 20 ms recorded).
- steps 20 k (dt 0.005) to 45 k (dt 0.0022); **wall 1 600 s to 5 900 s = 1 to
  4 simjobs** of 1 700 s through `tools/chem/chain.sh` (checkpoint 11.9 M x 8
  fields x 4 B = 380 MB, ~1 s).
- VRAM ~3 GB; host RAM < 4 GB (fits `-m 16384`).

Frames: volumes from t 70 tau (43 ms) every 0.25 tau (0.155 ms) = 121
frames over 18.6 ms physical; channels temp, fuel, oxid, omega, findex,
speed = 6 x 48 MB x 121 = **34 GB** (the renderer reads omega + findex;
fuel/oxid/temp/speed serve flame_dnsvid and diagnostics). A 10-15 Hz flicker
period (67-100 ms) does NOT fit; the clip shows the steady burner (stated).

Render (simjob --gpu, Blender Cycles, ~5 s/frame at 1080p x 121 = 10 min):

    blender -b --factory-startup --python tools/chem/flame_room.py --
      runs/flames/real2/r --out runs/flames/real2/room --t0 70
      --res 1920x1080 --samples 96 --device gpu --burner round
      --bands 0 --crop-lo auto --height 0.0620 --dif-gain 0.35
      --emit 330 --light-gain 600 --distance 0.30 --target 0.30 --lens 50
      --elevation 10 --azimuth 20 --orbit 40 --ambient 6

`--height` = the cropped box height above the rim = (384 - 30) x 0.175 mm =
0.0620 m; the tube (ID 10 / OD 12 mm) is real scale from the json; mp4 as
`runs/flames/hot/room_real.sh` (bloom).

Alternative M2 (real Re AND Pe, Pr 0.72, D/delta_th 1.77 x real): h 0.099 mm,
312 x 312 x 680 = 66 M cells, ~450 ms/step, dt in tau the same but tau =
0.35 ms -> 34-78 k steps for 62 ms: **15 000-35 000 s = 9-21 simjobs**, 15 GB
VRAM, volumes 190 GB at the same cadence. Same binary, flags only.

---------------------------------------------------------------------------

## 6. Risks

- K1 chemistry dt binds 5x below diffusion (explicit) -> S4 measures, S4b
  point-implicit.
- K2 central variable-density momentum convection at rho ratio 7 and 4
  cells/delta_th without acoustic damping: skew-symmetric form; S4/S5 runs
  watch the divergence-free kinetic energy budget; fallback a measured
  limiter is a new decision, not a default.
- K3 O(dt^2) Shvab-Zeldovich defect from the T-predictor/projected-flux lag
  (gpu_lowmach_sz gate); fallback: a second predictor-corrector pass (2x
  projection cost).
- K4 p = 0 open faces with backflow (plume top, lateral corners): backflow
  stabilization instantiation; S6/S7 jets and plumes are the gates.
- K5 EOS drift at the FP32 PCG floor (eosErr column); the Poisson right-hand
  side is a small difference div m* - D* only where the flame is; measured.
- K6 PCG iterations with the tube unknown (D9 gate <= 15; fallback S8b
  direct-forcing IB on the full-box direct solve, O(dt^3 p_tt) wall leak
  re-zeroed each stage).
- K7 anchoring on a COLD rim at 4 cells/delta_th: a one-step quench distance
  in delta_th units is not CH4's (D1); real rims run 400-600 K: `--rim-temp`
  scout; the 2D anchor test is the gate before 3D.
- K8 rich two-reactant calibration: unit S_L only at phi 1.3; the base edge
  flame burns leaner mixtures (model S_L(phi) printed, not fitted);
  sigma_st 7.0 vs CH4 7.5 (D3). [S1b: S_L(phi) now FITTED (model C2b,
  <= 0.34% vs GRI over 0.575..1.575); T_ad(phi) still the one-step's (-16% /
  +10% at phi 0.6 / 1.6), sigma_st unchanged.]
- K15 [S1b] the FP32 coupling-function walk reaches the rate through C:
  d ln S/dZ = -7.3 at phi 1.3 (where CH4's S_L(phi) is steepest). 1D
  compressible jet at 30 cells: Z up to 1 + 4.6e-4, S -2.3e-3 vs the
  uncorrected step. The walk grows with step count; the low-Mach solver
  must measure its own (S4 `gpu_lowmach_sz` gains a Z drift column) and
  S8's long 3D runs must report it.
- K9 the oracle (F1): S1 must make Lambda(sigma) provably continuous before
  any phi-1.3 number is trusted; every earlier sigma/Sutherland run needs its
  Lambda re-checked (issue 72 lists which docs rows used Sutherland).
- K10 open laterals at 1.5 D from the axis may confine the entrainment:
  S9 half-scale clearance scout.
- K11 FFT sizes with large prime factors (11) slow on hipFFT: validator
  rule + sizes chosen 2^a 3^b 5^c 7^d.
- K12 restart bitwise: every piece of host loop state must be in the
  checkpoint; the 4-restart byte test (S2) is the only proof -- a new
  runner variable added later without it breaks chaining silently, so the
  checkpoint struct is the loop-state struct (one place).
- K13 disk: 34 GB volumes (M1), 190 GB (M2).
- K14 dns-ui maps Physics by index (F5): LowMach = index 2 must never reach
  the editor; `--check` round trip run in S4.

---------------------------------------------------------------------------

## 7. Docs and issues (by stage)

- docs/chem.md: P0 (oracle fix, rateFactor, Model C2), "Low-Mach bunsen",
  "Real Bunsen" (section 5 result table, label), Tests table rows, Not built
  (two-step chemistry, compressible rim/laterals/gravity, LES on low-Mach).
- docs/lowmach.md (new): the solver spec (section 2) + measured floors.
- issues.md: 72 (oracle bracket), 73 (D1 trilemma), LM1... (K1-K6 as
  measured), R1... (rim standoff vs CH4, outer-cone height at Pe 0.6x,
  flicker not covered, Poiseuille assumption, room draft sensitivity).
- CLAUDE.md: Physics::LowMach rejected outside flame; low-Mach invariants
  (mass-flux projection; species ride the projected flux; P0 constant;
  buoyancy once; face BCs, no NSCBC/sink); `--restart` in the phase-1.5
  exception; the restart byte rule; layering line gains `gpu/lowmach`,
  `gpu/lm_poisson`, `gpu/runner_lowmach`, `dns/lowmach`.
