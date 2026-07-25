# masterplan_drift — repairing the post-recompose drift

Reviewer/masterplan step of the Simplification TPRI, 2026-07-25. Supersedes
`analysis_drift_{1,2,3}.md`. **This document is self-contained**: the
implementer does not need to read the analyses.

Scope was the whole tree, analysis-only, working tree as-is (uncommitted
curvilinear/framemap work included). Nothing in this plan has been implemented.

---

## IMPLEMENTATION ADDENDUM (2026-07-25) — read this before acting on §1.1

Implementation began on branch `drift-repair`. Four claims in this plan did not
survive contact with the code. They are corrected here; the sections below are
left as written so the reasoning stays auditable.

**§1.1 is WRONG. `--focal` does NOT report cd/cl scaled by R.** The mechanism
described is real -- `sampleCut()` works in reference coordinates, the ghost
wall classifies against mapped centres -- but the plan stopped one step short.
The reference body is SMALLER in physical space than the solved body by the
plateau slope `c·r` (~1/R), and the reference spacings are LARGER by exactly
the same factor. They cancel. Measured (cylinder, n = 256, plateau centred on
the body, true D = 0.785398):

| R | A, reference spacings | A, physical spacings |
|---|---|---|
| 1 | 0.785228 | 0.785228 |
| 2 | 0.785228 | 0.523485 |
| 4 | 0.785228 | 0.314091 |

Reference spacings give the solved body's true physical frontal area at every
R. Fix (a) -- "weight each row by `fy.deriv(y)·fz.deriv(z)`" -- would have made
every focal coefficient LOW by that slope. **R2 is void: no published focal
number is superseded.** The measured table now lives in `dns/geometry.hpp` and
is pinned by `tests/test_geometry.cpp`. §1.4 (three definitions of one number,
two live and inconsistent) was real and IS fixed: one `referenceQ*()` in the
geometry layer, `Case::qRef` deleted.

The plan marked the magnitude **[A]** and said "measure it before quoting it".
That instruction was the load-bearing part.

**R5 is resolved GREEN, not red.** `gpu_cutcell2d` passes at HEAD 6aaf912 on
this gfx1201 machine, as does the whole 54-test sweep. 2D parity IS usable as
evidence.

**§1.4's "`Case::qRef` is read by nothing" is wrong** -- `--dry-run` printed
it (`config.cpp:1099`), which made it worse than dead: `validate()` builds no
geometry, so it cannot know the frontal area, and the value printed as `qRef`
was a `0.5 uInf² D span` placeholder that no run ever used. The field and the
`--dry-run` column are both gone.

**§1.11's "`nzInterior` is read only inside `config.cpp`" is wrong** --
`tests/test_cli.cpp:265` asserts on it. It stays.

**Wave 5 (§1.5) was deliberately NOT completed as specified.** `runner_channel`
was not merged into a parameterised `runCompressible`. What landed instead is
`gpu/internal/run_loop.hpp`, holding the three pieces that are shared AND carry
no ordering: `fail()` (three identical copies), the periodic stable-dt
recompute with its collapse guard, and the wall-clock epilogue. Each runner
keeps its own loop body verbatim.

The reason is R1, taken at its word. The plan says the channel "is not a
different integrator", and structurally that is true -- but the loops differ in
exactly the places R1 lists, and every one of those differences is physics:
comp boosts the frame BETWEEN `dev->step` and `applyBc`, the channel applies
the wall after `step` and never boosts, the incompressible force tally reads
the FIRST stage's `phi`, and the dt-collapse thresholds differ per physics
(1e-6 / 1e-9 / 1e-4) because the dt scales do. A merged loop that standardised
any of them would be a physics change wearing a refactor's clothes, and none of
the remaining duplication is worth that. The reasoning is recorded at the top
of `run_loop.hpp` so the next reader does not re-derive it.

The residual duplication is the ~120 channel-specific lines the plan already
counted as genuinely channel-specific, plus the per-runner `writeFrame` lambdas
(which differ in fields, scales and remap). If the full merge is still wanted,
it should be its own change with before/after trajectories recorded -- not
folded into a cleanup.

**§1.6 landed in a second pass**, after the line count was challenged: the
first pass had deleted the dead surface the plan named but left the DOMINANT
item undone, so the tree barely shrank. `apps/dns_main.cpp` is 1313 -> 833 and
src/+apps is -444 net code. New: `dns/casegeom` (Case -> sdf/cut, with the
bodies' measured constants), `dns/flow_ic` (the three ICs),
`gpu/runner_periodic` (the tgv and hit loops, so all five time loops live in
`gpu/runner_*`), `gpu/periodic_frames`, `framemap::rasterFor`,
`operators::prolongMac`. `hit` no longer writes behind RunOutput's back.

The plan's "under ~400 lines" was NOT reached and is not being chased: what
remains is dispatch, GPU acquisition and Case -> RunSpec plumbing across five
subcommands. Cutting further means pushing CLI vocabulary into the runner
structs, which are deliberately Case-free so tests can construct them.

Verified by running the master binary and the branch binary side by side:
every artefact of flow/tunnel, flow/nscbc, flow/focal, flow/sphere and channel
is bit-identical, and tgv/hit stdout is identical. Two bugs the comparison
caught in my own extraction are recorded in the commit message.

**§1.14's smoke tests DID land** (`tests/test_gpu_runners.cpp`, ctest
`gpu_runners`): one per runner, asserting only exit 0, finite coefficients, a
healthy projection and enough stats rows. No literature band, so nothing is
quarantine-shaped. This closes the hole where `--far tunnel` WITH walls -- the
incompressible default -- and `runner_comp` were exercised by nothing at all.

**Two deliberate deviations from the letter of the plan:**
- §1.9's `CutPoissonSolver` did become a static `make()` returning `expected`,
  but `CutPoissonSolver::solve` and the CPU CG were NOT converted to
  `expected<SolveReport>`: neither `SolveReport` nor `PoissonError` exists in
  the tree (the plan says so itself), and the GPU CG already signals divergence
  by returning `maxIters`. The CPU CG now matches that contract exactly, which
  is the actual §1.10 defect, at zero call-site churn.
- §1.11's `bench` guards: the plan's own test table wants `dns bench --n 1`
  REJECTED, which is incompatible with admitting `nz == 1` there. `bench` has
  no 2D mode (the box is always n³), so the guard is `n >= 4`, which forecloses
  the degenerate-z case rather than handling it.

---

**Verification status.** Every claim below carries one of two marks. **[V]** =
the reviewer re-read the cited source and confirmed it. **[A]** = reported by an
analysis agent with a citation, not independently re-read by the reviewer —
check it before acting. Do not treat an **[A]** line count or magnitude as
measured.

---

## 0. What the TPRI found

The recomposition of 2026-07-19 held. Verified at HEAD: one binary, **zero
`__global__` kernels in `apps/`**, `comp_bc` gone, the five copies of the tunnel
BC / body force / LES kernels unified, the FP64 deterministic body tally in
place, NSCBC a genuine `setRhsHook` RHS overwrite, the ghost view still
diffusion-only, laterals still free-slip, and sponge nowhere a default in code.
No CLAUDE.md invariant is violated by the solver today.

Six days of feature work — `--steady`, `--seq`, `--pot` + open tunnel +
curvilinear, focal clustering + channel + scene + sigma SGS + cut-flux, plus the
uncommitted framemap work — re-accumulated in a specific, repairable shape:

1. **One live physics-reporting defect.** `--focal` with a body reports
   `cd`/`cl` wrong (§1.1). This is the only finding that changes a number a user
   has already read.
2. **Three dead surfaces still carried in the build**: the compressible
   `--cut-flux` path (validator-blocked, ~150 lines), the `--poisson` flag
   (parsed→validated→resolved→never read), and a scatter of dead `Case` fields,
   dead runner parameters and two tautological ternaries.
3. **Re-copied infrastructure the recompose masterplan had explicitly asked to
   unify**: four `reduce()` drivers, two `uploadNarrow`s, three `CurvView`
   builders, four SGS operator copies (three uncommitted), four frame writers,
   a third runner built by copying the second.
4. **`apps/dns_main.cpp` regrew to 1363 lines against a planned ~250, with
   physics back in it** — two divergent frontal-area integrals, a MAC
   prolongation operator, three ICs carrying measured physical constants.
5. **Documentation that contradicts the code, including a binding invariant.**
   README calls `--far sponge` the compressible default in two places; the
   validation docs give a regeneration recipe in binaries that no longer exist.
6. **A coverage hole at the riskiest layer**: no test includes any of the three
   runner headers, and `--far tunnel` with walls — the incompressible default —
   is exercised end-to-end by nothing.

### Binding constraints on everything below

- **No CLAUDE.md invariant may be relaxed.** Nothing in this plan requires it.
  Specifically preserved: ghost-view-diffusion-only, free-slip laterals,
  post-projection `refreshOnly`, NSCBC-as-RHS-overwrite, sponge-never-a-default,
  `2D is nz == 1, not a code path`, the anisotropic deep-solid `maxSpacing`
  threshold, and every "X off is bit-identical" pin.
- **Pure code motion must move numbers by exactly zero.** Copy expressions
  verbatim; do not "clean up" an associativity while moving it.
- **Anything that cannot be bit-identical is a separately measured commit**,
  with before/after recorded — not folded into a refactor.
- **`w stays zero` (`gpu_cutcell2d`) is believed RED at clean HEAD on this
  machine (gfx1201).** Establish its current status *before* using "the 2D
  parity tests still pass" as evidence for anything.

---

## 1. Verified findings, ranked

Rank = (value of the fix) × (confidence). Risk is the risk of *making the
change*, not of the status quo.

### 1.1 `--focal` with a body reports `cd`/`cl` scaled wrong — **the one live defect**
**Value: highest · Confidence: mechanism verified, magnitude inferred · Risk of fix: medium**

**[V]** `sampleCut()` evaluates the sdf at **reference** coordinates —
`geometry.cpp:82` `const double x0 = i * g.hx, y0 = j * g.hy`, and every face
sample at `:87-107` — so `geo.cut` is indexed and measured in reference space.

**[V]** The compressible frontal area is then integrated off that cut with
**reference** spacings (`apps/dns_main.cpp:530`):
`aFront += (1.0 - amin) * g.hy * g.hz;`, and `qRef` follows at `:535-537`. That
`qRef` goes to the runner unchanged and divides the force tally.

**[V]** The validator *deliberately allows* this combination:
`config.cpp:630-632` — "Bodies ARE supported (the ghost wall classifies against
the mapped cell centres), but only the analytic ones". So the ghost wall (the
solve) uses **physical** positions while `qRef` (the report) uses **reference**
ones.

**[V]** `qRef` is computed at `dns_main.cpp:535`, *before* the focal remap is
even constructed at `:554-565`. There is no later correction.

**[A]** Magnitude: a body of physical diameter `D` inside the fine plateau spans
`D·R/h` reference cells, so `aFront` is high by ≈ R and every reported `cd`,
`cl`, `cs` is **low by ≈ R** (`--focal-ratio`, default 4). This factor is an
inference from `Focal1D`'s definition, **not a measurement** — measure it before
quoting it.

CLAUDE.md's focal-clustering entry records exactly such a run (`--focal` Re100
cylinder). Any `cd` recorded from a `--focal` body run is suspect.

**Fix (in order of preference).**
(a) Integrate the frontal area over **physical** cell extents: weight each row
by `fy.deriv(y)·fz.deriv(z)`. (b) Better, and subsumes §1.4: one
`double referenceQ(const Case&, const CutGeometry&, const AxisRemap3&)` in the
geometry layer, called by both runners, replacing all three current definitions.
(c) Stopgap if neither lands this session: have `validate()` reject `--focal`
with a body and say why.

**Acceptance test:** one `--focal auto` cylinder against the same case without
`--focal`. The two `cd` values must agree to the discretisation difference — not
to a factor of 4.

### 1.2 The compressible `--cut-flux` path is unreachable — delete it
**Value: high · Confidence: verified · Risk: very low**

**[V]** `config.cpp:491-496` rejects `--cut-flux` with compressible physics
outright ("compressible cut-flux DIVERGES near the body (measured twice,
2026-07-23)"). **[V]** `CompCutView` and `kCompRhsCutOverwrite`
(`comp.hip:112,127`) exist in that one TU and nowhere else; `attachCutFlux` is
declared at `comp.hpp:90` and called only from `runner_comp.hip:108`. No test
references any of it.

**[A]** Live unreachable surface ≈ 150 lines: `comp.hpp:80-91` and `:151-153`
(six device buffers in **every** `DeviceComp`), `comp.hip:112-161` (kernel +
view), `:593-600` (dispatch), `:629-659` (`attachCutFlux`), `:261,274`
(`WavePerDt::lam` dt amplification), `runner_comp.hip:93-110` (solid-centre
classification loop), `runner_comp.hpp:33-34`, and the `comp.hip:634` assert
that exists only to guard it.

The measurement that killed this path is recorded in `config.cpp:491-496` and
CLAUDE.md, so the knowledge survives the deletion. Do not leave a dead branch in
the hot RHS dispatch as a memorial.

### 1.3 `--poisson` is parsed, documented, validated, resolved, and read by nothing
**Value: high · Confidence: verified · Risk: none**

**[V]** `cli.cpp:185,323` and `config.hpp:86,158` carry it; the validator
rejects it under compressible physics and resolves `Auto → Dct|Fft`. A grep for
reads of `c.poisson` outside `config.cpp`/`cli.cpp` returns nothing.
**[A]** The actual preconditioner comes from a local `fringe` bool at
`dns_main.cpp:333`. So `dns flow --far tunnel --poisson fft` silently runs DCT.

Either wire `Case::poisson` into the `bc` selection (~5 lines) or delete flag,
enum, field and help block (~15 lines out). **Deleting is the recommendation** —
the flag has no demonstrated use and a resolved-then-discarded field is exactly
the class of lie this TPRI is here to remove.

### 1.4 `qRef` has three definitions; two frontal-area integrals disagree
**Value: high · Confidence: verified · Risk: low**

**[A]** `config.cpp:1029-1032` computes `Case::qRef` — read by nothing.
**[V]** `dns_main.cpp:518-537` (compressible) re-derives it on a locally
sanitized cut over `k ∈ [0,nz-1], j ∈ [0,ny-1], i ∈ [1,nx-1]`, normalising by
`span`. **[A]** `dns_main.cpp:363-374` (incompressible) re-derives it again over
`k ∈ [1,nz-2], j ∈ [1,ny-2], i ∈ [2,nx-2]` with no `span` division, and contains
the no-op `g.hy * (g.nz == 1 ? g.hz : g.hz)` at `:370`.

Three definitions of one number: one dead, two live and mutually inconsistent in
both index range and normalisation, and both wrong under `--focal` (§1.1).
Collapse into the single `referenceQ()` of §1.1(b) and delete `Case::qRef`.

**Caution:** merge the *code path*, keep the two **normalisations** distinct and
dispatched on `--physics`. The compressible `span` handling is not decoration —
merging it away silently rescales compressible `cd`.

### 1.5 A third runner was created by copying the second
**Value: high · Confidence: verified (file), reported (block-level) · Risk: medium**

**[V]** `runner_channel.hip` is 249 lines against `runner_comp.hip`'s 352.
**[A]** Verbatim-or-near-verbatim from `runner_comp`: `fail()` (also in
`runner_incomp`), the dt-recompute + collapse-guard block (differing only in
threshold `1e-6` vs `1e-9`), the `writeFrame` lambda, the stats-row + every-10th
printf, and the `hipDeviceSynchronize`/`msPerStep` epilogue. Genuinely
channel-specific: ~120 of 249 lines (`ChannelStretchY`, `DeviceChannelWall`,
the body-force RHS hook, host row geometry, `measure()`/`wallGradient`,
`emitProfiles`).

The channel is not a different integrator — it is `DeviceComp` on a curvilinear
mesh with a stage hook and an RHS hook, which `runCompressible` already
supports. **This is the riskiest item in the plan and is sequenced last.**

### 1.6 `apps/dns_main.cpp`: 1363 lines, physics back in the app layer
**Value: high · Confidence: verified in part · Risk: low per item**

**[V]** The file is 1363 lines against a planned ~250, and `main` itself is
still a clean Parse → Scene → Validate → dispatch with `--dry-run` returning
before acquisition. Below it, **[A]**:

- two divergent frontal-area integrals (§1.4);
- two copies of the floor-row measurement (`:339-355` with `floorFrac`,
  `:705-714` without) implementing the identical sliver-row promotion rule, with
  the measured justification present in only one;
- `prolongMac` (`:281-323`) — a 43-line MAC-aware trilinear prolongation
  operator, i.e. a numerical operator, in the app layer;
- **[V]** the focal/stretch remap built at `:554-565` and rebuilt identically at
  `runner_comp.hip:56-67`, with a third axis-dispatch of the same knowledge in
  the raster lambdas at `:604-624`;
- named physical constants with measured brackets (dimple depth `:113,132`,
  fence thickness `:172`, trip-spike geometry with a 14-line measured-failure
  comment `:189-205`, the wake trip `:391-394`, the compressible taper +
  solenoidal kick `:546-584`, the channel IC streamfunction `:824-869`) —
  CLAUDE.md says constants live at their definitions, which are `bodies`/
  `geometry` and an IC module, not `main`. **Move the comments with the code.**
- the `hit` path writes `_stats.csv`, `_spec_N.csv` and `_snap_N.raw` directly
  (`:1122,1169,1186-1214`), behind `RunOutput`'s back and after phase 3c has
  closed — so a full disk surfaces hours into a run, not in the first second.
  `PeriodicFrames` (`:918-973`) is a fourth copy of the frame writer.

Realistic target: **under ~400 lines**.

### 1.7 Re-copied GPU infrastructure the recompose masterplan asked to unify
**Value: high · Confidence: verified · Risk: very low**

**[V]** Four `reduce()` drivers — `reductions.hip:53`, `cutcell.hip` (~:82),
`cutpoisson.hip:110`, `comp.hip` (~:359) — same rocPRIM transform-reduce with
the same scratch-regrowth block, plus a fifth `reduceInto` variant at
`cutpoisson.hip:137`. The masterplan explicitly asked for one shared helper
("kills 3 duplicated UB sites"); there are now more copies than when it was
written. **[V]** `uploadNarrow` is duplicated at `cutcell.hip:127` and
`comp.hip:102`.

Two copies carry the comment "kept local so hipCUB/rocPRIM stays in TUs" — a
constraint fully satisfied by a `gpu/internal/reduce.hpp` included only from
`.hip` TUs, exactly as `comp_flux.hpp` and `derived_remap.hpp` already do.
**[A]** ≈ 90 lines out; identical bodies, so per-TU codegen is unchanged.

Also unfinished from the recompose masterplan: **[A]** `comp_flux.hpp` never
moved under `internal/` (and `derived_remap.hpp` joined it outside);
`TimeStepper::reduceScratch()` (`timestepper.hpp:34`) survives with zero
callers; `DeviceComp::stableDt` still returns a bare `double` while the CPU
sibling's contract was specified to change too (§1.9).

### 1.8 Three `CurvView` builders, four SGS operator copies — fix inside the in-flight work
**Value: high (timing) · Confidence: reported · Risk: low**

**[A]** `CurvView` is assembled by hand identically at `comp.hip:519-528`,
`comp_ghost.hip:375-380`, `comp_nscbc.hip:354-365`. One
`CurvView DeviceCurv::view() const` kills two.

**[A]** The SGS operator (`sigma ? sigmaDsigma(gd) : the 6-line s2 loop`) is
written out four times in `comp.hip` — `kCompSmagNut:59-69`,
`kCompSmagNutCurv:183-193` *(uncommitted)*, `SmagMax:311-318`,
`SmagMaxCurv:347-354` *(uncommitted)* — all four taking an identical `gd[3][3]`.
One `__host__ __device__ inline double sgsNu(int sigma, const T gd[3][3])` in
`dns/sgs.hpp` (where `sigmaDsigma` already lives) serves all four.

**Three of these copies are uncommitted.** This is the cheapest fix in the plan
and it expires: land it *before* the curvilinear-LES work is committed, or it
becomes a refactor of committed physics instead of a tidy-up of in-flight work.

### 1.9 `assert`-as-input-guard survives where the recompose masterplan named it by line
**Value: high · Confidence: reported · Risk: medium (signature changes)**

CLAUDE.md: "`std::expected` for anything a USER can cause… Release is the build
that runs — never assert-as-input-guard." Recompose step 11 fixed
`diagnostics`, `trimesh`, `webframes`, `fieldpng` and the GPU side. **[A]** It
left:

| site | condition | user path |
|---|---|---|
| `comp.cpp:31,163` | `rho > 0` in `primitives()`/`pressure()` | any diverging compressible field |
| `comp.cpp:278,280` | `rho > 0`, `p > 0` in `stableDt()` | ditto — in Release becomes `sqrt(gamma*p/r)` with `p < 0` → NaN dt → silent non-termination, the exact exit-7 failure the masterplan named |
| `cutcell.cpp:503` | `nFluid_ > 0` | an all-solid geometry; in Release `projectMean` divides by 0 and poisons every solve |
| `hit_ic.cpp:143` | `kPrime > 0` | a degenerate HIT IC |
| `grid.hpp:24-25` | `n* > 0`, `l* > 0` | every `Grid::make`; the validator covers the CLI paths **except `bench`** (§1.11) |

`comp.cpp:146,153` (`gamma > 1 && mu >= 0`) are **legitimate** — the validator
makes them unreachable. Minimum fix: `Comp::stableDt` →
`std::expected<double, std::string>` (the GPU sibling already has this contract;
the CPU one was missed, so the two now disagree) and `CutPoissonSolver` → a
static `make()` returning `expected`.

### 1.10 The CPU CG reports a perfect solve on a NaN residual
**Value: high · Confidence: verified · Risk: none**

**[V]** `cutcell.cpp:557-562`: `double rr = fluidDot(r_, r_); … for (; it <
maxIters_ && rr > target; ++it)`. `NaN > target` is `false`, so a blown-up field
exits at iteration 0 and `solve()` returns `0` — indistinguishable from a
perfect warm-started solve. The recompose masterplan listed this verbatim; the
GPU CG got the `isfinite` guard, the CPU one did not.

One-line fix, and it pairs naturally with the `int → expected<SolveReport>`
change §1.9 wants anyway (`cutcell.hpp:148`, `poisson.hpp:19` — **[A]** neither
`SolveReport` nor `PoissonError` exists in the tree at all).

### 1.11 Dead fields, dead parameters, dead ternaries, and a validator hole in `bench`
**Value: medium-high · Confidence: verified · Risk: none**

**[V]** Two tautological ternaries with byte-identical branches:
`config.cpp:548` `const int bandBase = comp ? c.grid.nx : c.grid.nx;` and
`config.cpp:959` `c.tEnd = comp ? c.ft * c.L / c.uInf : c.ft * c.L / c.uInf;`.
Each marks a physics distinction somebody intended and abandoned — decide which,
or delete the ternary.

**[V]** `CompRun::mach` (`runner_comp.hpp:24`) is assigned at
`dns_main.cpp:668` and never read by `runner_comp.hip`; Mach reaches the solver
through `pInf` (`dns_main.cpp:515`). A parameter plumbed to the GPU and dropped.

**[A]** `Case::body` is a byte-copy of `Case::placement` (`config.cpp:436`),
read by nothing. `Case::lesLabel`, `nzInterior`, `L`, `dr`, `ft` are read only
inside `config.cpp`. `ChannelRun::fBody` is never assigned, so it is permanently
`1.0` (correct by the wall-units design — then make it a documented constant,
not a configurable-looking field). `TimeStepper::reduceScratch()` has zero
callers.

**[V] The `bench` validator hole, including a named invariant.**
`config.cpp:104-114` builds `Grid::make(n,n,n,2π,2π,2π)` after checking only
`n >= 1` and `steps >= 1`. Missing relative to every other branch: the
`cells > INT_MAX` check (so `dns bench --n 1300` overflows the `int` indexing
the tree assumes), the `nz == 1 or >= 4` guard, and **`nz == 1 ⇒ hz == 1`** —
CLAUDE.md states `hz = 1` is validator-enforced, and `dns bench --n 1` produces
`nz == 1` with `hz == 2π`. Four lines, reusing the guards two branches above.

### 1.12 Flags accepted, validated, and silently ignored
**Value: medium-high · Confidence: reported · Risk: low (user-visible)**

The unknown-flag half of the recomposition's strictness holds. The
*inapplicable*-flag half has eroded. **[A]**:

- `bench` accepts `--nz`, `--re`, `--ft`; `--re` is even validated
  (`config.cpp:100`) and discarded.
- `--emit` kinds are not scoped: `dns flow --emit spectrum`,
  `dns channel --emit snapshot`, `dns hit --emit png` all validate and do
  nothing. Only `Emit::Error` is scoped.
- Far-field parameters are not scoped to their far field: `--nscbc-beta` with
  `--far tunnel`, `--sponge-cells`/`--fringe-axes` under any far field.
  `--nscbc-strip 0 --far tunnel` passes silently.

Each is a one-line `if (r.given.contains(...)) return bad(...)` following the
pattern `config.cpp:460-471` already uses. **Land as one announced commit** —
this turns silent no-ops into exit 1 and will break scripts.

### 1.13 Documentation contradicts the code, including a binding invariant
**Value: medium-high · Confidence: verified · Risk: none**

**[V] Two** README claims that `--far sponge` is the compressible default:
`README.md:254` (`### --far sponge (compressible default) and --open lateral
bands`) and `README.md:186` (`dns flow --physics compressible --body cylinder`
(default `--far sponge`)). Agent 3 found the first; the second is a reviewer
addition. **[V]** `config.cpp:321-322` sets the compressible default to
`FarField::Nscbc`, and CLAUDE.md:29 forbids sponge-as-default in capitals. These
are the two most quotable lines in the section and both say the opposite of the
invariant.

**[V] `docs/validation.md` is unexecutable.** The Tier-C regeneration recipe —
the documented way out of the `val_*` quarantine — is written in deleted
binaries: `build/cylinder_tunnel_gpu` (`:140`),
`build/comp_cylinder_tunnel_gpu` (`:141,142`), `build/sphere_tunnel_gpu`
(`:143`), plus `:155,157`. **[V]** `grep BASELINE docs/` returns nothing —
`DNS_VALIDATION_BASELINE`, the mechanism that *is* the quarantine, is
undocumented outside `CMakeLists.txt` and CLAUDE.md.

**[V] Every `config` field in `data/validation/anchors.csv` (rows 7-11) names a
deleted binary**, though the file's own header says "run the exact 'config'
command". These are comment lines: fixing them cannot move a number.
`masterplan_recompose.md:1151-1161` already contains the replacement command
block. **Do not uncomment or delete anchor rows** — the quarantine is a
decision.

**[V] `docs/nscbc.md`** names `comp_cylinder_tunnel_gpu`,
`comp_mesh_tunnel_gpu`, `--nsbeta`, `--nsstrip` at `:4,88,128,129,137`.

**[A]** Further mismatches worth one pass: `_stats.csv` is documented as one
schema and is three (incomp 10 cols / comp 6 / channel 7 — `cli.cpp:667` and
`README.md:702` state only the incompressible one); README omits `--config`,
`--bumps`, `--bump-h`, `--png-down`, `--noslip`, `--focal-blend`,
`--tt/--lx/--lz` and the bodies `fence|floor|none`; the gfx1100 claims
(`README:28,44`, `CLAUDE.md:13`) outlive the arch auto-detection and this
machine is gfx1201; the layering diagram (`CLAUDE.md:19`, `README:138-143`) omits
six post-recompose modules, so an invariant that says "Preserve it" does not name
half of what it governs; CLAUDE.md says "eight" quarantined `val_*` tests where
CMake guards nine registrations.

### 1.14 The runners have no test; the incompressible default far field has none either
**Value: medium-high · Confidence: verified · Risk: medium (this ADDS coverage)**

**[V]** No file under `tests/` references `runner_incomp`, `runner_comp` or
`runner_channel` — they are the only public headers in the tree no test
includes. **[A]** `val_spindown`, the stated canary, builds its own time loop
from `kernels.hpp`/`cutpoisson.hpp` primitives and never enters a runner.
End-to-end runner coverage in the default build is `gpu_open_incomp`
(`--far fringe`) and `gpu_tunnel_open` (`--far tunnel --open`) — so **`--far
tunnel` with walls, the incompressible default, is exercised by nothing**, and
`runner_comp` by nothing at all (`val_comp_cyl_re60` **[A]** exits 77
unconditionally because every check is anchor-gated and every anchor is
commented out).

Fix: one unquarantined smoke test per runner — `--ft 0.5 --n 128`, asserting
only exit 0, finite `cd`, small `maxDiv`, ≥ N stats rows. No literature band, so
nothing is quarantine-shaped. `tests/test_gpu_tunnel_open.cpp:184-208` is
already this shape and is the template.

### 1.15 Test-layer duplication
**Value: medium · Confidence: reported · Risk: low-medium**

**[A]** ~25 test TUs each define a private `check(bool, const char*)` differing
only in `printf` field width, split across **two incompatible protocols**
(`int failures; ++failures` in 14 files, `bool pass; pass = false` in 7), each
with its own `main` epilogue. `validation_util.hpp:24-35` already provides the
counting version, but only the ten `val_*` files use it because that header also
drags in CSV parsing, anchors and `sys/wait.h`. Split three functions into
`tests/check.hpp`; ≈ 120 lines out, and it removes the possibility of a test
that counts failures then returns 0.

**[A]** `test_gpu_open_incomp.cpp` and `test_gpu_tunnel_open.cpp` contain
byte-identical `runApp`, `readStats` and `cleanup`, which `validation_util.hpp`
already provides — and provides *better*, parsing the stats CSV by column name
rather than the hardcoded `col[3]`/`col[7]` these two use (the hazard §1.13's
three-schema finding makes concrete). Their stated justification ("the kernels
live in the app's anonymous namespace", echoed at `CMakeLists.txt:182`) stopped
being true at recompose steps 5 and 7.

**[A]** `CMakeLists.txt:131-345`: the two `foreach` loops cover 47 of 54
registered tests in 18 lines; everything after `:146` is 14 hand-written target
blocks repeating the same six calls, with the quarantine marker comment
copy-pasted verbatim **eight times** (24 lines). One `dns_add_test()` helper
collapses ~196 lines to ~60. Verify by diffing `ctest -N` before and after.

### 1.16 Smaller items, all verified-cheap
**[A]** unless marked.

- `bodies.hpp:7` includes `cutcell.hpp` — the **inverse** of what the recompose
  masterplan specified — so anything wanting an `Sdf` transitively pulls in
  `CutPoissonSolver` and its CG workspace. The 30-line move is still available.
- `scene.cpp:304-330` `applySceneModel(const Scene&, RawConfig&)` has the
  geometry layer **mutating the parse layer**, inverting the documented order.
- `config.hpp` has become the hub header (`output.hpp`, `scene.hpp`, `cli.hpp`
  all include it for 2-3 symbols each); splitting `Emit`/`RunMeta`/`Placement`
  into a 40-line `dns/runmeta.hpp` kills the one backwards arrow in the layering.
- `validate()` is **[A]** 964 lines with two embedded algorithms: a 60-iteration
  bisection for `--focal auto` plateau derivation and a hand-rolled `strtod` CSV
  parser for `--focal BOX` that re-implements the strict full-consumption parse
  `cli.cpp:24-29` already owns. Five idioms duplicated 2-4× (output cadences,
  `--emit` defaults, `cfl` range, `nz` rule, `INT_MAX` check, and the discrete
  growth-ratio scan written three times over three map types).
- `validate()` is **no longer pure**: `config.cpp:513` `fprintf(stderr)`, same at
  `scene.cpp:399-402`. Both warnings are justified — *carry* them in
  `Case::warnings` and let main print them, rather than removing them.
- `RunMeta` is brace-initialised **positionally at four sites** with 17 members
  including a `bool` between doubles. Use designated initialisers.
- `RunOutput` does not create every output up front as promised: `.bin` streams
  open lazily at the first frame with no error check (`output.cpp:93-100`).
- `bodyforce.hip:129-155`: two `bodyBox` overloads, one exactly the other on an
  expanded AABB (−13 lines, arithmetically identical).
- `runner_comp.hip:332-339` carries two **contradictory** comments in sequence
  about frame suppression during the ramp; the code implements the second.
- `runner_comp.hip:319-320`: the `wmax` stats column is `wmax_true / dtMargin`
  (5 % low at the default), is not a wave speed at all when the run is
  viscous-limited, and multiplies by `min(hx,hy,hz)` — reference spacings on a
  curvilinear mesh. Diagnostic-only, but it is one of the `min(h)`-as-"the"-cell-
  size sites CLAUDE.md asks to audit.
- **[V]** `src/dns/framemap.{cpp,hpp}` and `tests/test_framemap.cpp` are
  **untracked** while `CMakeLists.txt:23,134` already reference them: a clean
  checkout does not build.
- `*.log` is not gitignored (`reconfig.log`, `viewer_server.log` untracked since
  21 Jul).
- Exit codes 4-8 are untested although `test_errors.cpp:1-3` claims the whole
  contract; **code 4 is GPU-free and one line to test** and is exactly what
  phase 3c exists to produce.
- `Scene::zInvariant()` excludes `Cube`, but a scene cube spans the full box and
  *is* z-invariant (`scene.hpp:54` vs `scene.cpp:198-204,362`). Comment or
  predicate is wrong; costs only sampling speed.

---

## 2. Implementation plan, in dependency order

Five waves. Each is independently landable and independently revertable. **Do
not reorder** — wave 1 fixes the documents later waves cite, and wave 2 deletes
code later waves would otherwise refactor.

### Wave 1 — documentation and provenance (zero code risk, do first)
1. `README.md:186` **and** `:254`: remove both "compressible default"
   attributions from `--far sponge`; retitle §7 to
   `### --far sponge (escape hatch; NOT a default)`.
2. `docs/validation.md`: replace the Tier-C recipe with the `dns`-subcommand
   block from `masterplan_recompose.md:1151-1161`; add a quarantine subsection
   naming `DNS_VALIDATION_BASELINE`; correct the Tier-A row count and the
   "runs in every ctest sweep" claim to what is actually registered.
3. `data/validation/anchors.csv` rows 7-11: rewrite the `config` strings as
   `dns` invocations. Comments only — do not touch values, do not uncomment.
4. `docs/nscbc.md`: `comp_cylinder_tunnel_gpu` → `dns flow --physics
   compressible`, `--nsbeta`/`--nsstrip` → `--nscbc-beta`/`--nscbc-strip`.
5. `CLAUDE.md` + `README.md`: gfx1100 → auto-detected (this machine is gfx1201);
   layering diagram gains `gpu/cutflux`, `gpu/comp_channel`,
   `gpu/runner_channel`, `dns/scene`, `dns/curvmesh`, `dns/remap`, `dns/sgs`,
   `dns/framemap`; "eight" quarantined → nine; add the `gpu_cutcell2d`
   known-red-on-gfx1201 status to "Known gotchas".
6. `cli.cpp` help fixes: `--penalize`'s "cd reads slightly low" → the measured
   "HALVES cdf" (CLAUDE.md is authoritative); move the stranded
   cylinder/sphere dimple note from `--bump-h` to `--dimples`; document the
   three `_stats.csv` schemas rather than one; drop the stale
   `output.cpp:138-141` comment.
7. `git add src/dns/framemap.{cpp,hpp} tests/test_framemap.cpp`; add `*.log` to
   `.gitignore`.

### Wave 2 — deletions (subtractive, near-zero risk)
8. Delete the compressible `--cut-flux` path (§1.2), including the
   `comp.hip:634` assert and the six `DeviceComp` buffers.
9. Delete `--poisson`: flag, enum, `Case` field, help block (§1.3).
10. Delete `Case::body`, `Case::qRef`, `CompRun::mach`,
    `TimeStepper::reduceScratch()`; make `ChannelRun::fBody` a documented
    constant; resolve the two tautological ternaries (§1.11).
11. Delete the far-field band-width fallbacks re-derived in both runners
    (`runner_incomp.hip:113-115`, `runner_comp.hip:126-127`) — the validator
    already resolves this — and reject a non-positive `cells` at the API
    instead. Same for `runner_channel.hip:92`'s re-derived `ubEst`.

### Wave 3 — correctness fixes (each with its own test)
12. **§1.10** CPU CG `isfinite(rr)` guard. One line, no signature change.
13. **§1.11** the four missing `bench` validator guards, including
    `nz == 1 ⇒ hz == 1`.
14. **§1.9** `Comp::stableDt` → `std::expected<double, std::string>`;
    `CutPoissonSolver` → static `make()` returning `expected`; migrate the
    `comp.cpp` ρ/p asserts and `cutcell.cpp:503`. Mechanical but touches
    `test_comp`, `test_comp_curv`, `test_cutcell`, `test_ke_conservation`,
    `test_projection` and `dns_main` in the same commit.
15. **§1.1 + §1.4** the `--focal` frontal-area fix. **This is the one commit in
    the plan that deliberately moves a published number.** Land the arithmetic
    fix, the single `referenceQ()`, the acceptance test and the CLAUDE.md /
    memory note together. Do not quietly re-baseline.

### Wave 4 — mechanical lifts (pure code motion; numbers must not move)
16. **§1.8 first, before the curvilinear-LES work is committed**: `sgsNu()` into
    `dns/sgs.hpp`, `DeviceCurv::view()` replacing the three hand-rolled builders.
17. **§1.7** `gpu/internal/reduce.hpp` (one `reduce` + `reduceInto`), one
    `uploadNarrow`; move `comp_flux.hpp` and `derived_remap.hpp` under
    `gpu/internal/`.
18. **§1.6** app → library: one `frontalArea()`, one floor-row measurement,
    `prolongMac` into the operators layer, the three ICs and their measured
    constants into `bodies`/`geometry`/an IC module, the focal remap built once
    in `validate()` into `Case`. Route the `hit` writes through `RunOutput`.
19. **§1.16** `bodies.hpp` ⟷ `cutcell.hpp` dependency inversion;
    `applySceneModel` into `config.cpp` taking a `const Scene&`;
    `dns/runmeta.hpp` split; designated initialisers for `RunMeta`;
    `bodyBox` forwarding; delete the contradictory ramp comment.
20. **§1.16** `validate()` decomposition: `maxGrowth()` template into
    `remap.hpp`, `validateOutput()` shared by flow and channel, `planFocal()`
    into its own TU (255 → ~30 lines in the largest block), `Case::warnings`
    replacing the two `fprintf(stderr)` sites.
21. **§1.15** `tests/check.hpp`; fold the two shell-out tests onto
    `validation_util.hpp`'s helpers (and onto **by-name** CSV parsing);
    `dns_add_test()` CMake helper.

### Wave 5 — the risky one, last and alone
22. **§1.5** merge `runner_channel` into a parameterised `runCompressible`.
    Only after waves 2-4 have shrunk the surface. See R1.

### Not in this plan — deliberately deferred
- **§1.12 flag scoping**, and the `cli.cpp` table rewrite that would make it
  fall out by construction. Correct, but it is a user-visible breaking change to
  the CLI contract and deserves its own decision, not a slot in a cleanup plan.
- **§1.14's smoke tests** are in the plan as *tests* (below), but the deeper
  question — whether `val_comp_cyl_re60`'s anchor-free M² physics check should
  be ungated — is a validation-policy call for the user.

---

## 3. New and extended tests

| test | covers | wave |
|---|---|---|
| `--focal` cylinder `cd` vs the same case without `--focal` | §1.1 — the acceptance test for the defect | 3 |
| CPU CG on a NaN-seeded RHS returns an error, not 0 | §1.10 | 3 |
| `dns bench --n 1` rejected; `--n 1300` rejected | §1.11, and the `hz == 1` invariant | 3 |
| `Comp::stableDt` on a negative-pressure state returns `unexpected` | §1.9 | 3 |
| smoke: `--far tunnel` walls, `--ft 0.5 --n 128` — exit 0, finite `cd`, small `maxDiv` | §1.14, `runner_incomp` default path | 3 |
| smoke: `--physics compressible --far nscbc`, same shape | §1.14, `runner_comp` — currently zero coverage | 3 |
| exit code 4: `--out` into an uncreatable directory | §1.16, phase-3c contract, GPU-free | 3 |
| `--cut-flux` + `--physics compressible` rejected | §1.2 — pins the invariant the deletion relies on | 2 |
| `--penalize` scope rules (three constraints, `config.cpp:498-503`) | currently unpinned | 2 |

Every added GPU test carries `TIMEOUT` and `SKIP_RETURN_CODE 77` like the rest
of the tier.

---

## 4. Explicitly rejected — do not do these

- **Collapsing the six per-axis convection/Laplacian functions**
  (`kernels.hip:56-114`, `cutflux.hip:63-175,190-254`, ~230 lines — the largest
  remaining duplication in the tree). `cutflux.hip:31-39` documents that
  `fluxDiff`'s associativity is chosen precisely so the `nz == 1` wrap cancels
  exactly under FMA contraction, "which is precisely how a compiler/arch change
  can break the 'w stays exactly 0' invariant"; `:256-264` documents that the
  plain kernel is launched first and overwritten because two TUs do not contract
  identically ("they don't -- measured"). A templated axis rewrite reassociates
  by construction.
- **Replacing `SmagMax`'s full gradient recomputation with `maxAbs(nut_)`.**
  Cheaper, but it uses a one-stage-stale νt and moves dt, hence trajectories.
- **Merging `comp.cpp`'s `rhsCurv` and `Comp::rhs`** (~80 lines of duplicated
  validated physics). Cannot be bit-identical — `(hi-lo)/h` vs `(hi·S-lo·S)/vol`
  differ in roundoff even on a uniform mesh. If the divergence risk becomes real,
  do it as a separately measured commit; otherwise add a cross-reference comment
  in each.
- **Unifying `TanhStretchY` and `ChannelStretchY`.** The reflection-ghost
  extension and `firstCell()` are load-bearing for the wall-mirror rows.
- **Merging `gpu_cutcell2d` into `gpu_cutcell`.** The whole point is the same
  operators at `nz == 1` and `nz > 1`.
- **Renaming `DeviceCompFar`** because it reads as sponge-ish. With `--far
  nscbc` and no strip it sets `sigMax = 0` and the kernel returns before
  touching a cell — confusing name, correct behaviour, cosmetic churn.
- **Undoing the `val_*` quarantine, or deleting/uncommenting anchor rows.**
  CLAUDE.md: a decision, not a failure.
- **Guarding the eight quarantined `test_val_*` `add_executable` calls behind
  the option.** They currently link for zero registered tests, which looks
  wasteful and is not: it keeps quarantined sources compiling against the
  current library. Document it; do not "fix" it.
- **Relaxing any bit-identity pin to a tolerance.** Every "X off is
  bit-identical" claim in CLAUDE.md has a corresponding `check(... == 0.0, ...)`;
  these are the tree's best asset.

---

## 5. Risks

**R1 — the runner merge (§1.5) moves trajectories if ordering is not carried
verbatim.** Load-bearing and *not* interchangeable: comp boosts *between*
`dev->step` and `applyBc` (`runner_comp.hip:282-298`); the channel applies the
wall *after* `step` (`:203`); the incompressible force tally reads the **first**
stage's `phi` (`runner_incomp.hip:263-271`); `dtRecompute = 16` with
`dtMargin = 0.95` is a deliberate trajectory-changing perf default. A merged
runner that standardises any of these is a physics change wearing a refactor's
clothes. This is why it is wave 5, alone.

**R2 — the `--focal` fix (§1.1) supersedes published numbers.** Certain, and the
point. Any `cd` recorded from a focal body run — including the one in CLAUDE.md's
focal-clustering memory — must be re-measured or marked superseded in the same
commit.

**R3 — extracting `planFocal` (§1.16/wave 4) is where numbers move by
accident.** The plateau derivation ends in `Focal1D::make(L, lo, hi, w, ratio)`;
those five doubles fully determine the mesh, hence `hmin`, hence dt, hence every
trajectory. Copy the expressions verbatim and gate on `--dry-run` producing
byte-identical output across `--focal auto` and explicit-BOX cases for every
body kind.

**R4 — the `check.hpp` conversion (§1.15) touches ~25 `main()` epilogues.** A
file that counts into `int failures` but returns `pass ? 0 : 1` would silently
become a test that always passes. Convert mechanically and diff every `return`.

**R5 — `gpu_cutcell2d` is believed red at clean HEAD on gfx1201.** No step in
this plan may be validated by "the 2D parity tests still pass" until its current
status is established on this machine. This belongs in CLAUDE.md's "Known
gotchas" whichever way it resolves.

**R6 — two latent traps found but not triggered today.** Neither is a bug now;
both are one validator relaxation away from being one, and both should get an
explicit comment or API guard rather than continuing to rely on a distant rule:
- `DeviceCompGhost::make` ignores `TanhStretchY` (`runner_comp.hip:166-167`
  passes only the focal `AxisRemap3`), safe **only** because `config.cpp`
  forbids a body with `--stretch`. Relax that and the ghost wall silently puts
  the body in the wrong place.
- `runner_incomp.hip:49` `const double h = g.hx` feeds the diffusive dt limit
  `6ν/h²`, which should use the *smallest* spacing. True today only because
  incompressible grids are built isotropic and `--focal` is compressible-only.
  If clustering ever reaches the incompressible path, dt is over-estimated and
  the run goes unstable — the same family as the focal drag inversion CLAUDE.md
  already records.

**R7 — nothing in this plan touches a CLAUDE.md invariant.** Checked explicitly
across all three lanes: ghost-view/convection separation, free-slip laterals,
post-projection `refreshOnly`, NSCBC-as-RHS-overwrite, sponge-never-a-default,
`nz == 1` never branched, the anisotropic `maxSpacing` deep-band threshold, the
two grid conventions, and every bit-identity pin are all preserved as-is.

---

## 6. Build and test commands for the implementer

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDNS_WITH_HIP=ON
cmake --build build -j
ctest --test-dir build --output-on-failure --timeout 900
build/dns flow --dry-run <flags>          # before every real run
ctest -N --test-dir build                 # diff before/after the CMake helper
```

Wave 4 items are pure code motion: the acceptance criterion is that
`ctest` output and every pinned bit-identity check are unchanged. Wave 3 item 15
is the only commit expected to change a reported number, and it must say so.
