# analysis_drift_2 — GPU layer and the single binary

TPRI agent 2 of 3 (GPU layer `src/dns/gpu/**` + `apps/dns_main.cpp`). **No code was modified**; this is a read of the working tree as of **2026-07-25**, including the uncommitted modifications to `comp.hip`, `comp.hpp`, `comp_flux.hpp`, `runner_comp.{hip,hpp}`, `remap.hpp` and `apps/dns_main.cpp`. Every line reference is to that tree. Agent 1 owns `src/dns/` non-GPU, agent 3 owns `tests/`, CMake and docs; items landing in their lane are marked **[xref]**.

---

## 0. Executive summary

The recomposition held where it was finished. There is exactly one binary, no `__global__` anywhere in `apps/` (grep-verified), the five copies of the tunnel BC / body force / LES kernels are gone, `tunnel.hip` carries the unified diagonal-corner `kBcTunnel` the masterplan specified, `comp_ghost` uses a deterministic FP64 tally, and `comp_bc` is dead and buried. NSCBC is a genuine `setRhsHook` overwrite, the ghost view still feeds only diffusion, the laterals are still free-slip, and sponge is nowhere a default.

Drift since 2026-07-19 comes from four additions — channel mode, focal/curvilinear clustering, `--cut-flux`, scenes/framemap — and has three shapes:

1. **A third runner was created by copy.** `runner_channel.hip` (249 lines) repeats `fail()`, the `dtRecompute` block, the frame lambda and the `msPerStep` epilogue from `runner_comp.hip` verbatim. The masterplan's architecture had exactly two runners; the channel is a compressible run with different walls and different statistics, not a third integrator.
2. **`apps/dns_main.cpp` grew from the planned ~250 lines to 1363, and physics came back with it.** A MAC prolongation operator, two divergent copies of the frontal-area integral, two copies of the floor-row measurement, three ICs with named physical constants (dimple depth, trip spikes with measured brackets, the channel trip streamfunction), a duplicate construction of the focal/stretch remap that `runner_comp.hip` builds again 20 lines later, and a `hit` path that writes files behind `RunOutput`'s back after phase 3c has closed.
3. **Infrastructure the masterplan said to unify was re-copied instead.** Four byte-identical `reduce()` drivers, two `uploadNarrow()`s, three hand-rolled `CurvView` builders, four copies of the SGS `|S|`/`D_sigma` operator (three of them uncommitted), two `bodyBox` overloads with a duplicated `span` lambda.

Plus one clean deletion: the **compressible `--cut-flux` path is unreachable**. The validator hard-rejects `--cut-flux` with `--physics compressible` (`src/dns/config.cpp:491-496`), no test references it, and ~150 lines of kernel, state and runner branch remain live in `comp.{hpp,hip}` and `runner_comp.hip`.

No CLAUDE.md invariant is violated today. Two latent traps are noted in §4.

---

## 1. Inventory

### 1.1 File census (`wc -l`, 2026-07-25)

| group | files | lines |
|---|---|---|
| compressible core | `comp.hip` 740, `comp.hpp` 167, `comp_flux.hpp` 221 | 1128 |
| compressible boundaries | `comp_ghost` 399+95, `comp_nscbc` 385+118, `comp_far` 143+50, `comp_channel` 138+61 | 1389 |
| incompressible core | `kernels` 309+47, `cutcell` 199+56, `cutpoisson` 363+88, `cutflux` 319+43, `poisson` 426+48, `ghostfill` 184+67 | 2149 |
| shared services | `reductions` 267+59, `derived` 137+51, `derived_remap.hpp` 55, `device_*` 47+41+85, `index.hpp` 63, `farpot.hpp` 39, `les` 156+72, `bodyforce` 219+77, `tunnel` 319+132, `timestepper` 79+53, `projection.hpp` 30 | 1971 |
| runners | `runner_incomp` 377+117, `runner_comp` 352+103, `runner_channel` 249+44 | 1242 |
| **`src/dns/gpu/` total** | 49 files | **7889** |
| `apps/dns_main.cpp` | 1 file | **1363** |

`__global__` kernels: **62**, all inside `src/dns/gpu/`. Zero in `apps/`. The masterplan's headline defect ("twenty-one `__global__` kernels carrying validated physics live in anonymous namespaces in `apps/*.hip`") is fully repaid.

### 1.2 What the masterplan asked for and what actually happened

| masterplan item | state today |
|---|---|
| `gpu/tunnel.{hpp,hip}` absorbing 5×`kBcTunnel` etc. | **done**, with the specified diagonal corner + `nz == 1` guard (`tunnel.hip:28-29`) and the Class-A `kInLo/kInHi` outlet form (`tunnel.hip:131,157,174`) |
| `gpu/bodyforce`, `gpu/les`, `gpu/derived`, `gpu/cutpoisson` split | **done** |
| `comp_bc` deleted | **done** (no file, no reference) |
| FP64 deterministic body tally | **done** — `comp_ghost.hip:80-84` stages per face, `:391-396` reduces in FP64 |
| `GpuGrid::kInterior()` | **done** as `kInLo()/kInHi()/kInCount()` |
| `sumScalar`/`maxAbsScalar` moved to `reductions.hpp` | **done** |
| one shared reduction-scratch helper ("kills 3 duplicated UB sites") | **NOT done** — now 4 copies + a 5th variant (§2.3/F4) |
| `comp_flux.hpp` → `gpu/internal/comp_flux.hpp` | **NOT done**; and `derived_remap.hpp` (new) is a second GPU-internal header outside `internal/` |
| `gpu/timestepper.hpp`: `grid()`/`reduceScratch()` accessors removed | **half done** — `grid()` gone, `reduceScratch()` remains at `timestepper.hpp:34` with **zero** callers |
| `apps/dns_main.cpp` ~250 lines, zero physics | **1363 lines, physics present** (F3) |
| two runners (`runner_incomp`, `runner_comp`) | **three** (F2) |
| `DeviceComp::stableDt` → `std::expected` | **not done** — still a bare `double`; the runners check `isfinite` at the call site (`runner_comp.hip:271`, `runner_channel.hip:192`), a correct-by-convention rather than by-type contract |

### 1.3 New since the masterplan (all post-2026-07-19)

`cutflux.{hpp,hip}`, `comp_channel.{hpp,hip}`, `runner_channel.{hpp,hip}`, `derived_remap.hpp`, `farpot.hpp`, the curvilinear half of `comp.hip` (`kCompRhsCurv`, `kCompSmagNutCurv`, `SmagMaxCurv`, `WavePerDtCurv`, `DeviceCurv`), the `Remap` template parameter of `kNscbcCells`, the focal-aware `DeviceCompGhost::make` overload, and `--pot`/floor/scene support threaded through `tunnel.hip` and `bodyforce.hip`.

---

## 2. Findings, ranked

Rank = value × confidence. Risk is the risk of the *change*, not of the status quo.

### F1. The compressible `--cut-flux` path is unreachable code — delete it
**Value: high · Confidence: high · Risk: very low**

`src/dns/config.cpp:491-496` rejects `--cut-flux` with compressible physics outright ("compressible cut-flux DIVERGES near the body (measured twice, 2026-07-23)"), which CLAUDE.md also records. Nothing else can set the flag: `CompRun::cutFlux` is written only at `apps/dns_main.cpp:676` from `c.cutFlux`. `grep -rln "CompCutView|kCompRhsCutOverwrite"` over the whole repo returns **`src/dns/gpu/comp.hip` only** — no test, no other TU.

Live, unreachable surface:

- `comp.hpp:80-91` (`attachCutFlux` declaration + 12-line contract comment), `comp.hpp:151-153` (`cutFlux_`, `cutAx_/cutAy_/cutAz_/cutLam_`, `cutFlag_`, `cutSolid_` — six device buffers in every `DeviceComp`).
- `comp.hip:112-161` (`CompCutView` + `kCompRhsCutOverwrite`, 50 lines), `comp.hip:593-600` (the dispatch), `comp.hip:629-659` (`attachCutFlux`, 31 lines), `comp.hip:261,274` (`WavePerDt::lam`, the 1/λ̃ dt amplification), `comp.hip:706`.
- `runner_comp.hip:93-110` — the solid-centre classification triple loop and `clearBoundaryFlag` call, plus `runner_comp.hpp:33-34`.
- `comp.hip:634` `assert(!curv_ && …)` exists only to guard this path.

Total ≈ 150 lines. **What breaks:** nothing reachable. `DeviceComp` shrinks by six `DeviceScalar` members. If the intent is to keep the experiment alive, the honest form is a note in `docs/archive/` plus removal from the build — not a live branch in the hot RHS dispatch.

### F2. `runner_channel.hip` is a third copy of the compressible time loop
**Value: high · Confidence: high · Risk: medium**

| block | runner_comp | runner_channel | difference |
|---|---|---|---|
| `fail()` helper | `:26-29` | `:24-27` | none (also `runner_incomp.hip:26-29`) |
| dt recompute + collapse guard | `:268-279` | `:189-200` | collapse threshold `1e-6` vs `1e-9` |
| `writeFrame` lambda | `:218-252` | `:97-115` | channel has no raster/asinh/`speedDelta`; both do `primitiveVelocity` → vort/curl → `speedCentred` → 2×`download` → float convert → 2×`FrameField` |
| stats-row + every-10th printf | `:302-331` | `:207-236` | different columns |
| `hipDeviceSynchronize` + `msPerStep` + `setMsPerStep` | `:343-349` | `:240-246` | none |

Genuinely channel-specific: `ChannelStretchY` mesh (`:43-48`), `DeviceChannelWall` (`:53-57`), the body-force rhs hook (`:60-62`), host row geometry `yc/dy` (`:69-74`), `measure()`/`wallGradient` (`:31-34`, `:124-150`), `emitProfiles` (`:152-177`) — ~120 of 249 lines. The other ~130 are the compressible loop. The channel is not a different integrator: it is `DeviceComp` on a curvilinear mesh with a stage hook and an rhs hook, exactly the shape `runCompressible` already supports. The clean form is one `runCompressible` parameterised by (a) the BC objects installed on the two hooks, (b) the stats series + row callback, (c) the frame field list.

**What breaks:** trajectories, if the ordering is not carried verbatim — see R1. Pinned by `tests/test_gpu_channel.cpp` and the comp validation set. Do this only after F3/F4 shrink the surface.

### F3. Physics and duplicated measurement in `apps/dns_main.cpp`
**Value: high · Confidence: high · Risk: low per item**

The file is a thin `main` (`:1273-1363` is exactly Parse → Scene → Validate → dispatch, and `--dry-run` returns before any acquisition). Below that, physics leaked back:

**(a) Two divergent frontal-area integrals.** Incompressible `:363-374` sweeps `k ∈ [1, nz-2]`, `j ∈ [1, ny-2]`, `i ∈ [2, nx-2]` forming `aFront += (1-min_i ax)·hy·hz`, then `qRef = 0.5·uInf²·aFront`. Compressible `:520-537` sweeps `k ∈ [0,nz-1]`, `j ∈ [0,ny-1]`, `i ∈ [1,nx-1]` on a *local sanitized copy* of the cut, then divides by the span. Same measurement, three different index ranges, two normalisations, and the incompressible copy contains the no-op `g.hy * (g.nz == 1 ? g.hz : g.hz)` at `:370`. One `frontalArea(const Grid&, const CutGeometry&)` removes ~25 lines and the question of which range is right. **[xref agent 1]** for the home.

**(b) Two copies of the floor-row measurement.** `:339-355` (with `floorFrac`) and `:705-714` (without) implement the identical "advance `fr` while `ax == 0`, then promote a `< 0.5` sliver row" rule, including the identical `k0 = g.nz == 1 ? 0 : 1`. The measured justification ("measured ratchet at 2048²") appears only in the first.

**(c) Duplicate remap construction.** `:554-565` builds `TanhStretchY yRemap` + `AxisRemap3 fRemap` from the focal parameters; the identical loop is rebuilt at `runner_comp.hip:56-67`; a *third* axis-dispatch of the same knowledge (which axis goes through `yRemap` first) appears in the raster `fwd`/`inverse` lambdas at `:604-624`. These are trivially copyable value types — `validate()` should build them once into `Case`. Newest code, most smeared.

**(d) `prolongMac` (`:281-323`)** — a 43-line MAC-aware trilinear prolongation operator, i.e. a numerical operator, in the app layer. **[xref agent 1]**

**(e) Named physical constants with measured brackets living in the app.** Dimple footprint/depth `8h/4h` and `5h/2.5h` (`:113`, `:132`); fence thickness `D/16` (`:172`); trip-spike geometry `f.x0 = hx + 0.10·(nx-2)·hx`, `f.pitch = 0.65·(nx-2)·hx/bumps` with a 14-line measured-failure comment (`:189-205`); the wake trip box and `v = 0.05 uInf` (`:391-394`); the compressible taper + solenoidal kick (`:546-584`); the whole channel IC streamfunction (`:824-869`). CLAUDE.md: constants live at their definitions — which are `bodies`/`geometry` and an IC module, not `main`. The comments are excellent; move them with the code.

**(f) `hit` writes behind `RunOutput`.** `:1122` hand-builds a `_stats.csv` path, `:1169` `appendStatsCsv`, `:1186-1193` opens `_spec_N.csv` inside the loop, `:1195-1214` opens `_snap_N.raw` + `.json` inside the loop. That defeats phase 3c — a full disk is found after hours of GPU, not in the first second. `PeriodicFrames` (`:918-973`) is a fourth copy of the frame writer.

Realistic target: `dns_main.cpp` under ~400 lines.

### F4. Four byte-identical `reduce()` drivers (the masterplan asked for one)
**Value: high · Confidence: high · Risk: very low**

`reductions.hip:52-76`, `cutcell.hip:82-106`, `cutpoisson.hip:107-131`, `comp.hip:359-379` are the same 25-line rocPRIM transform-reduce with the same scratch-regrowth block and the same "assert() is a comment in Release" comment; `cutpoisson.hip:136-158` adds `reduceInto` (same body, result left on device). Two carry the comment "kept local so hipCUB/rocPRIM stays in TUs" (`cutcell.hip:81`, `cutpoisson.hip:108`) — a constraint satisfied by a `gpu/internal/reduce.hpp` included only from `.hip` TUs, exactly as `comp_flux.hpp` and `derived_remap.hpp` already do. `uploadNarrow` is likewise duplicated at `cutcell.hip:127-138` and `comp.hip:102-110` (the latter's comment admits it). Removes ~90 lines; identical bodies, so per-TU codegen is unchanged.

### F5. Three hand-rolled `CurvView` builders and four copies of the SGS operator
**Value: medium-high · Confidence: high · Risk: low** *(mostly uncommitted code)*

`CurvView` from `DeviceCurv` is assembled by hand identically in `comp.hip:519-528` (private `curvView()`), `comp_ghost.hip:375-380`, `comp_nscbc.hip:354-365`. One `CurvView DeviceCurv::view() const` kills two, and the `assert(t && …)` + `if (!t) return` pairing at `comp_ghost.hip:373-374` / `comp_nscbc.hip:357-358` collapses with it.

The SGS operator (`if (sigma) → sigmaDsigma(gd)`, else the 6-line `s2` loop) is written out four times in `comp.hip` — `kCompSmagNut:59-69`, `kCompSmagNutCurv:183-193` *(uncommitted)*, `SmagMax:311-318`, `SmagMaxCurv:347-354` *(uncommitted)* — plus a fifth MAC-staggered variant in `les.hip:50-64`. The first four take an identical `gd[3][3]`; one `__host__ __device__ inline double sgsNu(int sigma, const T gd[3][3])` in `dns/sgs.hpp` (where `sigmaDsigma` already lives) serves all four. Cheapest fix in the in-flight curvilinear-LES work; land it before the fourth copy is committed.

Related note, not a proposal: `SmagMax`/`SmagMaxCurv` recompute the full gradient tensor over the whole grid purely to bound νt for the viscous dt, duplicating what `nut_` already holds. Replacing it with `maxAbs(nut_)` would use a one-stage-stale νt and **would move dt, hence trajectories**.

### F6. Runner boilerplate: `rampFrac`, `fail`, the frame lambda, the epilogue
**Value: medium · Confidence: high · Risk: low**

`rampFrac` — `runner_comp.hip:32-37` and `runner_incomp.hip:33-38`, identical smoothstep, the second's comment saying so ("the runner_comp schedule, verbatim"). `fail()` — three copies. `msPerStep` epilogue — three copies. The frame lambda — four copies counting `PeriodicFrames::write` (`apps/dns_main.cpp:951-966`): each does `download` ×2 → float convert with optional `uInf` offset → two `FrameField`s → `out.frame`; comp adds the raster resample (`runner_comp.hip:237-243`), incomp the asinh toe, channel neither. A `gpu/internal/framewriter.hpp` removes ~100 lines and puts the raster hook in one place. Note the resample runs **on the host** per frame into a grid up to `--focal-ratio`× larger per axis (`framemap.hpp:47-55`); `derived.hpp` exists to do that work on device.

### F7. Six per-axis convection/Laplacian functions where two would do
**Value: medium · Confidence: high · Risk: HIGH — do not do casually**

`kernels.hip:56-114` `convX/convY/convZ` (≈19 lines each), `cutflux.hip:63-175` `convXCut/convYCut/convZCut` (≈36 each), `cutflux.hip:190-254` `lapCutU/lapCutV/lapCutW` (≈24 each): the same expression under a cyclic index permutation, ~230 lines.

Largest remaining duplication and the one I am **not** recommending. `cutflux.hip:31-39` documents that `fluxDiff` is written in a specific associativity precisely so the `nz == 1` wrap cancels exactly under FMA contraction — "which is precisely how a compiler/arch change can break the 'w stays exactly 0' invariant"; `cutflux.hip:256-264` documents that the plain kernel is launched first and overwritten because "two TUs [do not] contract the same expressions identically (they don't -- measured)". A templated axis rewrite reassociates by construction. Blocked in practice by the "2D is `nz == 1`, not a code path" invariant's parity tests — which memory records as already red at HEAD on gfx1201, so a new break would be indistinguishable from the old one.

### F8. The far-field band width is re-derived in the GPU layer
**Value: medium · Confidence: high · Risk: low**

`runner_incomp.hip:113-115` and `runner_comp.hip:126-127` both recompute `std::clamp(int(0.1 * g.nx), 8, g.nx / 2 - 1)` when the passed cell count is 0. `validate()` already resolves this (`config.cpp:553-555`) and `dns_main` passes the resolved value (`:449`, `:683`), so both fallbacks are dead in the binary — but live for direct callers (tests), and `std::clamp` with `hi < lo` is UB for `nx < 18`. The validator's own `nx >= 18` guard (`config.cpp:549-552`) is conditioned on `far ∈ {Fringe, Sponge, Nscbc}`, so the *validator's* copy is still reachable with `--far tunnel --n 16` **[xref agent 1 — that is the live UB; also `config.cpp:548`'s `comp ? c.grid.nx : c.grid.nx` is a tautological ternary]**. In my lane the fix is subtractive: delete both fallbacks and reject a non-positive `cells`/`ns` (`DeviceFringe::make` already rejects `cells < 1`, `tunnel.hip:301`). Same shape, smaller: `runner_channel.hip:92` recomputes the log-law bulk estimate `2.5·log(max(reTau,3)) + 2.6` that `config.cpp:230` already put in `Case::ubEst`, here only to scale a colour map.

### F9. Dead parameters and dead accessors
**Value: medium · Confidence: high · Risk: nil**

- **`CompRun::mach`** (`runner_comp.hpp:24`) is assigned at `apps/dns_main.cpp:668` and **never read** in `runner_comp.hip` (`grep -c "r\.mach"` → 0); Mach reaches the solver through `pInf` (`dns_main.cpp:515`). A flag plumbed to the GPU and dropped.
- **`TimeStepper::reduceScratch()`** (`timestepper.hpp:34`) has zero callers anywhere. The masterplan asked for both accessors to go; only `grid()` went.
- **`ChannelRun::fBody`** (`runner_channel.hpp:24`) is never assigned by `runChannel` (`dns_main.cpp:891-903`), so it is permanently `1.0`. Correct by the wall-units design (header line 3: "f = 1") — then it should be a documented constant, not a field no path configures.
- `gpu::convection`, `rkCombine`, `addForcing`, `addScaledLaplacian`, `subtractGradient` (`kernels.hip:229-270`) have no caller outside `tests/test_gpu_operators.cpp`: library API kept alive solely by its own unit test. Real CPU/GPU-mirror coverage, but nobody should count them as used. **[xref agent 3]**
- `derived.hip:48-56` `kVorticityZCentred` is exactly `kVorticityZRemap<IdentityRemap>` (`derived_remap.hpp:22-44` says so), but the divisors differ in rounding (`2*g.hx` in `Real` vs a `double` map difference) — not a free lift; leave unless the frame path is reworked anyway.

### F10. `bodyBox` overloads and a stale contradictory comment
**Value: low-medium · Confidence: high · Risk: nil**

`bodyforce.hip:129-142` and `:144-155` are the same function with the same nested `span` lambda and the same `nz == 1` early return; the centre/radius overload is exactly the AABB overload on `{cx-r, cy-r, cz-r, cx+r, cy+r, cz+r}`. One forwards to the other: −13 lines, arithmetically identical.

`runner_comp.hip:332-339` carries two contradictory comments in sequence: "No frames during the quiet-start ramp…" immediately followed by "The output schedule is independent of the quiet-start ramp… The ramp gate that used to suppress them hid the first `ramp` time units of every run for no numerical reason". The code implements the second. Delete the first paragraph.

### F11. `wmax` in the compressible stats row is not `wmax`
**Value: low-medium · Confidence: high · Risk: low (diagnostic only)**

`runner_comp.hip:319-320`: `const double wmax = r.cfl / dtStab * std::min({g.hx, g.hy, g.hz});`. Since `dtStab = r.dtMargin * stableDt(...)` (`:274`), the reported value is `wmax_true / dtMargin` — 5 % low at the default `dtMargin = 0.95` — and when the run is viscous-limited (`comp.hip:724-736`) it is not a wave speed at all. On a curvilinear mesh `g.hx/hy/hz` are *reference* spacings, so the factor is not a physical length either. Nothing in the solver reads the column, but it is a published diagnostic and one of the `min(hx,hy,hz)`-as-"the"-cell-size sites CLAUDE.md asks to audit. Fix: have `stableDt` return the reduced `wmax` it already computes, or drop the column.

### F12. assert-vs-`std::expected` discipline
**Value: low-medium · Confidence: medium-high · Risk: nil**

28 asserts in the GPU layer; acquisition paths return `std::expected<…, hipError_t>`. Reviewed individually:

- `comp.hip:482` `assert(gamma > 1.0 && mu >= 0.0 && csLes >= 0.0)` — all three derive from user flags, but the validator enforces them, so this is legitimate *because the validator ran*. It should say so; today a reader cannot tell it from an input guard.
- `comp.hip:634` `assert(!curv_ && …)` guards a flag *combination* (`--cut-flux` + `--focal`), rejected at `config.cpp:643-646` — legitimate, and it disappears with F1.
- `cutcell.hip:151` `assert(nFluid > 0)` is the masterplan's named case ("else `kFluidMean` divides 0/0 and poisons every `projectMean`") and is still an assert in the build that ships. `dns_main.cpp:372` catches the *inverse* error (no solid cells, exit 3); there is no matching check for an all-solid domain. `DeviceCutGeometry::make` already returns `std::expected` — returning `hipErrorInvalidValue` costs one line and closes the last user-reachable assert in the incompressible geometry path.
- `poisson.hip:266,303`, `timestepper.hip:13-14,63`, `cutpoisson.hip:271,355`, `kernels.hip:303`, `device_util.hip:17,24` — validator-guaranteed or programmer-only. Leave.

Discarded statuses remain widespread (49 `(void)` in `src/dns/gpu/*.hip`), including `comp.hip:673` `(void)ptrs_.upload(h)` inside the RK3 hot path and every `(void)dVort->download(...)` in the runners. Masterplan step 11 unfinished. Low priority in the runners (a failed download shows as a garbage frame); higher for `ptrs_.upload`, which silently produces a wrong axpy pointer table.

---

## 3. Verified correct / leave alone

1. **Ghost view is diffusion-only.** `runner_incomp.hip:141-148`: `dGhost->fillFrom(in, *dG)` then `fusedRhsStage(..., in, *dG, ...)` — convection reads `in`, diffusion reads `*dG`. `cutflux.hip:277-282` keeps the split in the apertured overwrite. The force tally deliberately re-fills from `*dU0` (`runner_incomp.hip:269`) so friction is read through the view the viscous term saw. Intact.
2. **Free-slip laterals, no Dirichlet far field.** `tunnel.hip:39,58-60,77-79`; warning preserved at `tunnel.hpp:21-22`. No `u = uInf` lateral row exists.
3. **Open laterals leave normal velocity free, slaved once per stage.** `tunnel.hip:46-57` with the `slave` flag; `runner_incomp.hip:161` passes `slave = 1` pre-projection, `:169` `refreshOnly = true` after — exactly the "re-slaving broke div u by ~150×" rule.
4. **NSCBC is an rhs overwrite.** `runner_comp.hip:183-187` installs via `setRhsHook`; `kNscbcCells` writes only `krho/kmx/kmy/kmz/kE` (`comp_nscbc.hip:273-279`), never the state. `etaIn`/`betaTrans` brackets live in the Config. The channel's body force follows the same discipline (`runner_channel.hip:60-62`, comment citing it).
5. **Sponge is never a default.** `CompRun::far = FarField::Nscbc` (`runner_comp.hpp:57`); `config.cpp:321-322` defaults comp to Nscbc and `:325-330,355` reject everything but {Sponge, Nscbc}. With `--far nscbc` and no strip, `runner_comp.hip:135` sets `fc.sigMax = 0` and `kFarSponge` returns before touching any cell (`comp_far.hip:36`) — `DeviceCompFar` is then purely the free-slip/no-slip wall module. Confusing name, correct behaviour; renaming is cosmetic churn.
6. **`nz == 1` is not a code path.** z terms always present and cancelling by wrap: `bodyforce.hip:30-32,66-70`, `tunnel.hip:29,68-69`, `les.hip:15-17,70`, `comp_flux.hpp:103-104,142-146`, `cutflux.hip:31-39`. The degenerate-axis switches that exist (`comp.hip:705`, `WavePerDtCurv`'s null `az`, `ndir = nz == 1 ? 4 : 6` at `comp_ghost.hip:294`) are the documented legitimate ones.
7. **The anisotropic deep-solid threshold.** `comp_ghost.hip:151-161,217-231` uses `maxSpacing(dloc)` with the measured `cd 1.49 → −7.6` bracket in place, and `hCoarse` (not fine) bounds the bbox short-circuit at `:162-181`. I audited every other `min(h)` in the GPU layer: `ghostfill.hip:57` (FD step only), `comp.hip:731` (viscous dt — minimum is the conservative choice), `comp_nscbc.hip:100` (per-axis array), `runner_comp.hip:320` (diagnostic — F11). No second instance of the focal-drag bug.
8. **`--cut-flux` incompressible: overwrite-after-plain-kernel.** `cutflux.hip:300-317` launches the stock `fusedRhsStage` first so interior cells keep its exact bits by construction (same trick at `comp.hip:588-600`). Ugly, deliberate, documented, measured — keep.
9. **`comp_nscbc`'s `Remap` template parameter.** `comp_nscbc.hip:46-56` with `constexpr bool remapped = !is_same_v<Remap, IdentityRemap>` keeps the uniform instantiation textually unchanged; `launchNscbc` (`:284-312`) funnels all three overloads. This is how curvilinear work *should* be done and is the model for the smear in F3(c).
10. **`kPotInlet` as a separate kernel** — `tunnel.hip:84-88` records that folding it into `kBcTunnel` "perturbed its codegen and broke the pinned bit-identity". Leave separate.
11. **`kBcTunnel`'s unified corner treatment** (`tunnel.hip:28-29`) and the outlet trio's `kInLo()/kInHi()` form — masterplan step 5's decision, implemented and commented. Do not reintroduce a mode switch.
12. **`DeviceCompGhost`'s `bbox` culling** (`comp_ghost.hip:183-192`) — keep-list item 8, still bit-identical and still a large setup win.

---

## 4. Risks

**R1 — F2 (runner merge) moves trajectories if ordering is not carried verbatim.** Load-bearing: comp boosts *between* `dev->step` and `applyBc` (`runner_comp.hip:282-298`); the channel applies the wall *after* `step` (`:203`); the incompressible force tally reads the **first** stage's `phi` (`runner_incomp.hip:263-271`); `dtRecompute = 16` with `dtMargin = 0.95` is a deliberate trajectory-changing perf choice (keep-list 7) and must stay a default. A merged runner that standardises any of these is a physics change wearing a refactor's clothes.

**R2 — F7 (axis templating) reassociates floating point.** `cutflux.hip:31-39` and `:256-264` both record measured evidence that this breaks either the exact-zero `w` invariant or interior bit-identity. Compounded by `cutcell2d` already being red at HEAD on gfx1201.

**R3 — latent: `DeviceCompGhost::make` ignores `TanhStretchY`.** `runner_comp.hip:166-167` passes only the focal `AxisRemap3`; on the `--stretch` mesh the ghost wall would classify against reference coordinates. Safe **only** because `config.cpp` forbids a body with `--stretch`. If that is relaxed, the ghost wall silently puts the body in the wrong place. Either take a general `PointRemap` there or make the impossibility explicit at the API.

**R4 — latent: the incompressible runner assumes an isotropic grid.** `runner_incomp.hip:49` `const double h = g.hx` feeds both the outlet convection speed (`:161`, correct — that axis is x) and the diffusive dt limit `6ν/h²` (`:234`, which should use the *smallest* spacing). True today because `config.cpp:395-401` builds incompressible grids with `hx == hy == hz`, and `--focal` is compressible-only. Nothing asserts it. If clustering ever reaches the incompressible path, dt is over-estimated and the run goes unstable — same family as the focal drag inversion.

**R5 — F1 is irreversible in the working tree.** The measurement that killed compressible cut-flux is recorded in `config.cpp:491-496` and CLAUDE.md, so the knowledge survives deletion; the code does not. If a redesign is imminent, drop the file from `CMakeLists` instead — but do not leave a live branch in `DeviceComp::rhs`.

**R6 — F3 moves files across the app/library boundary agent 1 owns.** Frontal area, `prolongMac`, the ICs and the body constants all want homes in `src/dns/`; sequencing with agent 1's proposals matters more than the individual moves.
