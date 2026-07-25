# analysis_drift_1 — host library (src/dns, non-GPU)

TPRI agent 1 of 3. Focus: **the CPU/host library — `src/dns/*.{cpp,hpp}` excluding `src/dns/gpu/`.** No code was modified. All line references are to the working tree as of 2026-07-25, **including uncommitted modifications and untracked files** (`framemap.{cpp,hpp}` are untracked; `cli.cpp`, `config.cpp`, `remap.hpp` carry uncommitted edits — flagged inline where relevant).

Agent 2 owns `src/dns/gpu/` + `apps/dns_main.cpp`; agent 3 owns `tests/`, `CMakeLists.txt` and docs. Where a finding's root cause sits in my lane but its symptom lives in theirs, it is marked **[x-ref agent 2]**.

---

## 0. Executive summary

The recomposition landed a clean host library on 2026-07-19. Six days of feature work (`e65ea13` --steady, `de7e10a` --seq, `f9f16d6` --pot + open tunnel + curvilinear, `01e4f65` focal + channel + scene + sigma + cut-flux, plus the uncommitted curvilinear-frames work) has re-accumulated in three specific places, and has left three live correctness defects.

**The shape of the drift.**

1. **`validate()` grew a fourth subcommand and a curvilinear mesh generator without growing any structure.** `config.cpp` is 1149 lines, of which `validate()` is 964 (`:79-1042`). The `--focal` block alone is **255 lines** (`:613-867`) — a bisection search, a hand-rolled CSV parser, three per-axis loops and a discrete growth-ratio scan — and the channel branch is another 111 (`:163-273`) that re-derives, line for line, the output-cadence and emit validation the flow branch already does (`:246-263` vs `:993-1024`). Five idioms are now duplicated 2-4× each. The masterplan's `validate()` was one flat sequence of guards; it is now a 964-line function with two large embedded algorithms.
2. **The CLI table is still a table, but it is *assembled imperatively*.** `cli.cpp:138-232` builds three parallel vectors (`opts`, `enums`, `flags`) through nine per-subcommand `if` blocks, with `--emit` special-cased in the scan loop (`:255-267`). The lookup is still generic; the *registration* is not. That is exactly why `--nz`, `--re` and `--ft` reach `bench` and are silently ignored, and why `channel` can emit PNGs but cannot set `--vort-max`. The 350-line `help()` (`:439-790`) is hand-synchronised with those tables and has already drifted (§2 F16).
3. **Three concepts that should be one are three.** `Case::body` is a byte-copy of `Case::placement` (`config.cpp:436`, admitted at `config.hpp:161-162`) and nothing reads it. `Case::qRef` is computed (`config.cpp:1029-1032`) and never read — both runners recompute it from the cut geometry instead **[x-ref agent 2]**. `Case::poisson` is parsed, documented and resolved (`config.cpp:896-900`) and never read by anything.

**The three live defects**, in descending severity:

- **`--focal` + a body reports the wrong `cd`/`cl`.** The frontal area that sets `qRef` is measured from `geo.cut`, which `sampleCut()` built on the *reference* grid, while the physical cell size inside the fine plateau is smaller by `--focal-ratio`. `qRef` is therefore too large by ~R (default 4) and every reported coefficient too small by the same factor. §2 F1.
- **`--poisson` does nothing.** A documented flag (`cli.cpp:643`), validated and resolved, with zero consumers. §2 F2.
- **`assert`-as-input-guard survives in the compressible CPU reference.** `comp.cpp:31, 163, 278, 280` assert `rho > 0` and `p > 0` inside `primitives()`, `pressure()` and `stableDt()`. Masterplan §5 named these by file and line as *the* migration targets; step 11 (`c58b31c`) did not touch them. Release is the build that runs, so a diverging field divides by zero and takes `sqrt` of a negative, silently. Same class: `cutcell.cpp:503` `assert(nFluid_ > 0)`, `hit_ic.cpp:143`, `grid.hpp:24-25`. §2 F9.

**What is genuinely healthy:** `sgs.hpp` is exemplary (clamped, documented, device-compilable, zero drift). `remap.hpp`'s `Focal1D` identity-by-branch discipline is the right pattern and is followed. `sampleCut()` is still the one sampler. `PhysicalRaster` is display-only and inactive-is-a-copy. No `min(hx,hy,hz)`-as-"the"-cell-size abuse exists anywhere in the host library (§3). The sponge-is-never-a-default rule is correctly implemented (`config.cpp:320-322`).

---

## 1. Inventory — what masterplan_recompose specified vs what is there now

`src/dns` (non-GPU) is **8362 lines** over 51 files. The masterplan's §1.1 table listed 23 files; there are now 30 distinct concepts. New since 2026-07-19, none of them in the masterplan:

| file | LOC | landed in | what it is |
|---|---|---|---|
| `remap.hpp` | 396 | `01e4f65` (+ uncommitted `invMapY`) | `PointRemap` concept + 5 concrete maps (Identity, Sinusoidal, TanhStretchY, ChannelStretchY, Focal1D/AxisRemap3) |
| `curvmesh.{hpp,cpp}` | 87 + 223 | `f9f16d6`/`01e4f65` | `CurvMesh` nodes + `CurvMetrics` FV tables |
| `scene.{hpp,cpp}` | 102 + 443 | `01e4f65` | scene TOML parse + union-sdf mesh builder |
| `framemap.{hpp,cpp}` | 78 + 131 | **untracked** | `PhysicalRaster` display resampling |
| `sgs.hpp` | 77 | `01e4f65` | `SgsModel` enum + `sigmaDsigma()` |
| `hd.hpp` | 10 | `01e4f65` | `DNS_HD` host/device macro |

Against the masterplan's per-file spec, the deviations are:

| masterplan §1.1 / §3 said | actual state |
|---|---|
| `cutcell.hpp`: shapes **move out** to `bodies.hpp`; `cutcell.hpp` includes `bodies.hpp` | **Not done, and inverted.** `RotatedCube`/`Sphere`/`CylinderZ` are still at `cutcell.hpp:28-45`; `bodies.hpp:7` includes `cutcell.hpp` and its header comment (`:2-4`) documents the inversion as intended. §2 F11 |
| `poisson.hpp`: `solve()` → `std::expected<SolveReport, PoissonError>` | **Not done.** `poisson.hpp:19` still `virtual void solve(...)`. Neither `SolveReport` nor `PoissonError` exists anywhere in the tree (grep) |
| `gpu/cutpoisson.hpp`: `int solve` → `expected<SolveReport,...>` | CPU sibling `CutPoissonSolver::solve` still returns `int` (`cutcell.hpp:148`) |
| §5: migrate every assert guarding a user-reachable condition — `grid.hpp:22`, `cutcell` `nFluid > 0`, `comp.cpp` ρ/p positivity, `hit_ic.cpp:143` | **None of the four migrated** (§2 F9) |
| §5/step 11: "CG `isfinite(rr)` (a blown-up field reports a perfect solve at iteration 0)" | Fixed on the GPU side; the **CPU** `CutPoissonSolver` loop (`cutcell.cpp:558-562`) still has no finiteness test. §2 F10 |
| `config.hpp`: `Case` is "fully resolved, no optionals" | Holds ~12 fields no consumer reads (§2 F3/F4), one field that is an exact duplicate of another, and one (`poisson`) that is resolved and discarded |
| `config.hpp:238`: "Pure: no filesystem, no HIP" | `validate()` writes to stderr (`config.cpp:513`). §2 F6 |
| §1.1 `output.hpp` / `RunOutput`: "creates every output before the first `hipMalloc`" | Only `<prefix>.json` and `_mask.bin` are created in `make()`; `.bin` streams open lazily at the first frame (`output.cpp:93-100`). §2 F19 |
| CLI: 4 subcommands | 5 (`channel`), plus `--scene` and `--config` as parallel input channels |

Everything else in §1.1 is present and in the right file. `diagnostics.cpp:60-61` carries the §5 `nShells < 1` fix; `output.cpp:37-43` carries the `closeChecked` fix; `trimesh`/`meshsdf` return `std::expected`; `projection.cpp` and `poisson_jacobi.cpp` are merged away as specified.

---

## 2. Findings, ranked by (value × confidence)

### F1 — `--focal` with a body reports `cd`/`cl` low by ~`--focal-ratio` · risk: **high value, medium-risk fix**

`sampleCut()` (`geometry.cpp:27-124`) evaluates the sdf at `(i*g.hx, j*g.hy, k*g.hz)` — **reference** coordinates — and returns apertures indexed in reference space. That is correct for a uniform mesh and is what every incompressible run has. Under `--focal` the physical position of cell `(i,j,k)` is `AxisRemap3::map` of that point, and the physical spacing inside the fine plateau is `hAx[a]/focalRatio` (`config.cpp:830`, `remap.hpp:212-241`).

The compressible frontal area is then measured off that reference-space cut and weighted with reference spacings:

```
apps/dns_main.cpp:531:   aFront += (1.0 - amin) * g.hy * g.hz;
apps/dns_main.cpp:535:   const double qRef = aFront > 0 ? 0.5*rhoInf*uInf^2*(aFront/span) : ...
```

and `qRef` is handed to the runner unchanged (`dns_main.cpp:670`), which divides by it (`gpu/runner_comp.hip:306-308`). A body of physical diameter `D` sitting in the plateau spans `D·R/h` reference cells, so `aFront ≈ D·R·hz` instead of `D·hz`: **`qRef` is high by ≈R and every reported `cd`/`cl`/`cs` is low by ≈R** (R defaults to 4). `qRef` is computed **before** the remap is even constructed (`dns_main.cpp:535` vs `:557`), so there is no chance for a later correction.

Confidence: the code path is certain (read end to end). The factor-R magnitude is an inference from `Focal1D`'s definition, not a measurement — verify with one `--focal auto` cylinder run against the same case without `--focal`.

The validator *does* permit this: `config.cpp:634-642` rejects only mesh/scene and fence/floor bodies with `--focal`; cylinder/sphere/cube/lens are allowed, and CLAUDE.md's focal-clustering memory records exactly such a run.

**Fix, cheapest first.** (a) Compute the frontal area from the *physical* cell extents: multiply each row's contribution by `fy.deriv(y)·fz.deriv(z)` at the row. (b) Better: delete the two ad-hoc `aFront` loops and make `Case::qRef` (§F3) the single definition, computed in `validate()` where the focal map is already resolved — it knows `focalHFine[]` (`config.cpp:830`) and could integrate the map exactly. (c) Cheapest stopgap: have `validate()` reject `--focal` with a body until (a) or (b) lands, and say why.

**Breaks:** any recorded `cd` from a `--focal` body run is superseded; say so in CLAUDE.md rather than silently re-baselining. **Does not** touch any CLAUDE.md invariant — this is a reporting quantity, not a scheme.

### F2 — `--poisson` is parsed, documented, validated, resolved, and dead · risk: **none to fix**

`cli.cpp:212-215` registers the enum; `cli.cpp:643` documents it; the validator rejects it under compressible physics (`config.cpp:463-464`) and resolves `Auto → Dct | Fft` from the far field (`config.cpp:896-900`). Grep for `c.poisson` outside `config.cpp`/`cli.cpp`: **zero hits.** The actual preconditioner is chosen in `dns_main.cpp:333` from a local `fringe = far == Fringe || far == Periodic` bool and passed to `DeviceCutPoissonSolver` as a `bc` argument (`gpu/cutpoisson.hpp:26`).

So `dns flow --far tunnel --poisson fft` runs the DCT preconditioner and says nothing. Either wire `Case::poisson` into the `bc` selection (and keep the `auto` resolution as the default), or delete the flag, the enum, the `Case` field and the help block. Deleting is 15 lines out and closes a lie; wiring is ~5 lines in `dns_main.cpp` **[x-ref agent 2]**.

### F3 — `qRef` is defined twice in the library and computed twice more in main · risk: **low**

`config.cpp:1029-1032` computes `c.qRef = 0.5·uInf²·D·span` under the comment "Reference dynamic pressure x frontal area" — and no consumer reads `Case::qRef`. Meanwhile `dns_main.cpp:363-374` (incompressible) and `:518-536` (compressible) each re-derive it from the sanitized cut, with *different* fallbacks and different `span` handling. Three definitions of one number, one of them dead, and the two live ones are the ones F1 gets wrong.

Collapse to one: `validate()` cannot see the cut (it touches no geometry), so the honest split is a single `double referenceQ(const Case&, const CutGeometry&, const AxisRemap3&)` in the geometry layer that both runners call. Delete `Case::qRef`. **Breaks:** nothing today (the field is unread); after the merge, compressible and incompressible `qRef` become the same formula — check whether the incompressible `(1 - amin)*hy*(nz==1 ? hz : hz)` at `dns_main.cpp:370` (both branches identical — itself a dead ternary) matches the compressible `aFront/span` normalisation before merging.

### F4 — dead and duplicate `Case` members · risk: **none**

- `Case::body` (`config.hpp:161`) is assigned `= c.placement` at `config.cpp:436` and read by **nothing** in `apps/`, `src/dns/gpu/` or `tests/`. `config.hpp:162` calls it an "alias kept for readability"; it is a second copy of a struct that can silently diverge. Delete.
- `Case::scenePath` is read only to build `meta.body` (`config.cpp:1037`); main loads the scene from the `RawConfig` in phase 1.5, not from the `Case` (`config.hpp:164`'s comment says otherwise). Either make main read it or drop the comment.
- `Case::lesLabel`, `nzInterior`, `L`, `dr`, `ft` are read only inside `config.cpp` — legitimate intermediates promoted to public API. Making them private (or local) shrinks the `Case` surface by five fields at zero risk.
- Two dead ternaries, both with byte-identical branches: `config.cpp:548` `const int bandBase = comp ? c.grid.nx : c.grid.nx;` and `config.cpp:959` `c.tEnd = comp ? c.ft*c.L/c.uInf : c.ft*c.L/c.uInf;`. Each is a place where somebody intended a physics distinction and left the scaffolding.

### F5 — `validate()` re-accumulated: 964 lines, 5 duplicated idioms, 2 embedded algorithms · risk: **low-medium**

Verified duplicate blocks (both copies read):

| idiom | copies | lines |
|---|---|---|
| output cadences + `--planes` + `--png-down` (fdt/pdt/wdt default-follow, finitePos loop, 1-or-3, divides-nx-ny) | 2 | `:246-259` (channel) vs `:993-1006` (flow) — identical modulo two message strings |
| `--emit` empty-default + `none`-exclusivity | 2 | `:260-263` vs `:1022-1024` |
| `cfl in (0,1]` | 3 | `:130`, `:234`, `:475` |
| `nz == 1 or >= 4` | 3 | `:120`, `:179-180`, `:377-378` |
| `cells > INT_MAX` | 3 | `:123-125`, `:208-210`, `:438-440` — **and missing from the bench branch** (§F8) |
| discrete max-growth-ratio scan over a 1D map | 3 | `:219-225` (`ChannelStretchY`), `:605-611` (`TanhStretchY`), `:834-841` (`Focal1D`) — three copies of the same loop over three map types |

Two algorithms are embedded in the validator that are not validation:

- `--focal auto` plateau derivation, including a **60-iteration bisection** (`:725-737`) that shrinks the desired box until its blend clears the domain faces.
- a hand-rolled comma-separated float parser for `--focal BOX` (`:743-769`), which re-implements the strict full-consumption parse `cli.cpp:24-29` already owns, using `strtod` — the one numeric-parsing idiom the recomposition removed everywhere else.

**Proposal.** (a) Hoist the three growth-ratio loops into one `template <class Map> double maxGrowth(const Map&, int n, double h)` in `remap.hpp` — all three maps expose `mapY`/`map`, so it is a 6-line template and removes 21 lines. (b) Hoist the output-cadence block into `std::expected<void, ConfigError> validateOutput(const RawConfig&, Case&)` — one caller from each of flow/channel; removes 28 lines and the message drift. (c) Move the `--focal` plateau derivation and BOX parsing into `remap.hpp`/a new `focal_plan.cpp` behind `std::expected<FocalPlan, std::string> planFocal(const Grid&, const Placement&, FocalOpts)`, leaving `validate()` with ~30 lines of guards. That is 255 → ~30 in the single largest block. All three are pure code motion; **no numbers move** if the expressions are copied verbatim (masterplan §8's rule).

**Breaks:** nothing, provided the error `field` strings and messages are carried across — `tests/test_cli.cpp` and `test_channel_config.cpp` assert on them **[x-ref agent 3]**.

### F6 — `validate()` is no longer pure · risk: **none to fix**

`config.hpp:238` and `config.cpp`'s own header comment promise "Pure: no filesystem, no HIP, no allocation beyond the Case." `config.cpp:512-515` writes the under-resolved-boundary-layer warning straight to `stderr` with `fprintf`. The same pattern appears in the geometry layer at `scene.cpp:399-402` (non-watertight mesh warning).

This is small but it is the one contract that makes `--dry-run` trustworthy and `validate()` unit-testable without capturing streams. Add `std::vector<std::string> warnings;` to `Case` and have main print them after the resolved-case dump. Masterplan §5 also states "No error message is ever a warning the run continues past" — these two are exactly that, and both are justified (they are honesty warnings, not errors), so the fix is to *carry* them, not to remove them.

### F7 — flags accepted, validated, and silently ignored · risk: **low**

The recomposition's headline property was "unknown flags are a hard error… today every unrecognised token silently becomes the grid size". The unknown-flag half holds. The *inapplicable*-flag half has eroded:

- **`bench`**: `--nz`, `--re` and `--ft` are in the base option table (`cli.cpp:138-145`) and reach the bench branch (`config.cpp:105-114`), which reads only `--n` and `--steps`. `--re` is even validated (`:100`) and then discarded. `dns bench --nz 8 --ft 40` runs a 256³ 20-step benchmark.
- **`--emit` kinds are not scoped to a subcommand.** `emitOf` (`cli.cpp:76-92`) accepts all eleven for all five subcommands. `dns flow --emit spectrum`, `dns channel --emit snapshot`, `dns hit --emit png` all validate and do nothing (the consumers are at `dns_main.cpp:1183`/`:1195`, hit-only). Only `Emit::Error` is scoped (`config.cpp:148-150`).
- **Far-field parameters are not scoped to their far field.** `--nscbc-beta` with `--far tunnel` is accepted and ignored; `--sponge-cells` and `--fringe-axes` are resolved for every far field (`config.cpp:553-567`) and read only by fringe/sponge. Only `--nscbc-strip` is scoped, and only when `> 0` (`:535-536`), so `--nscbc-strip 0 --far tunnel` passes silently.

Each is a one-line `if (r.given.contains(...)) return bad(...)` in the branch that ignores it — the pattern `config.cpp:460-471` already uses for `--tol`/`--poisson`/`--mach`/`--kick`. Or, better, scope the tables (§F16's fix) so the parser rejects them by construction.

**Breaks:** any script passing a harmless-but-ignored flag starts failing. That is the point, but it should land in one commit with a note.

### F8 — validator gaps in the `bench` branch, including an `hz == 1` invariant hole · risk: **none to fix**

`config.cpp:105-114` builds `Grid::make(n, n, n, 2π, 2π, 2π)` after checking only `n >= 1` and `steps >= 1`. Missing, relative to every other branch:

- **no `cells > INT_MAX` check** — `dns bench --n 1300` allocates 2.2e9 cells and overflows the `int` indexing the rest of the tree assumes.
- **no `nz == 1 ⇒ hz == 1`.** CLAUDE.md: "`hz = 1` is validator-enforced." `dns bench --n 1` gives `nz == 1` with `hz == 2π`. Every other branch applies the rule explicitly (`:126`, `:400`, `:417`, `:207`).
- **no `nz == 1 or >= 4`, no minimum for confined axes.** `dns bench --n 2` produces a 2×2×2 grid.

Low impact (bench is a throughput toy) but it is a live, named invariant violation and the fix is four lines reusing the guards two branches above.

### F9 — `assert` still guards user-causable conditions in the compressible reference and the CPU cut solver · risk: **medium (changes signatures)**

CLAUDE.md: "`std::expected` for anything a USER can cause; `assert` only for what a PROGRAMMER can cause… Release is the build that runs — never assert-as-input-guard." Masterplan §5 named the exact sites. Step 11 (`c58b31c`) fixed `diagnostics`, `trimesh`, `webframes`, `fieldpng` and the GPU side, and left these:

| site | condition | user path |
|---|---|---|
| `comp.cpp:31` (`primitives`) | `s.rho[i] > 0` | any diverging compressible field |
| `comp.cpp:163` (`pressure`) | `rho > 0` | ditto |
| `comp.cpp:278, 280` (`stableDt`) | `rho > 0`, `p > 0` | ditto — in Release this becomes `sqrt(gamma*p/r)` with `p < 0` → NaN `dt` → an infinite silent loop, the exact failure mode masterplan §5 called out ("dt not finite" is exit code 7) |
| `comp.cpp:146, 153` (`Comp::make`) | `gamma > 1 && mu >= 0` | validator covers it — these are *legitimate* |
| `cutcell.cpp:503` | `nFluid_ > 0` | an all-solid geometry (bad `--dr`/`--scale`); in Release `projectMean` then divides by 0 and poisons every solve — masterplan §5 named this by name |
| `hit_ic.cpp:143` | `kPrime > 0` | a degenerate HIT IC |
| `grid.hpp:24-25` | `n* > 0`, `l* > 0` | every `Grid::make` call site; the validator covers the CLI paths but `bench` (§F8) does not check `n` bounds |

`comp.cpp` is the CPU reference the GPU compressible path is tested against (`tests/test_comp.cpp`, `test_comp_curv.cpp`), so it runs in CI in Release. Minimum fix: `Comp::stableDt` → `std::expected<double, std::string>` (the masterplan already specified exactly this for `gpu/comp.hpp`, and the GPU side did it — the CPU sibling was missed, so the two now disagree on their contract), and `CutPoissonSolver`'s constructor → a static `make()` returning `expected`.

**Breaks:** every caller of `Comp::stableDt` and `CutPoissonSolver`'s ctor — a handful in `tests/` and `dns_main.cpp`. Purely mechanical.

### F10 — the CPU CG reports convergence on a NaN residual · risk: **none to fix**

`cutcell.cpp:558-562`:

```
double rr = fluidDot(r_, r_);
const double target = tol_ * tol_ * double(nFluid_);
int it = 0;
for (; it < maxIters_ && rr > target; ++it)
```

`NaN > target` is `false`, so a blown-up field exits at iteration 0 and `solve()` returns `0` — indistinguishable from a perfect warm-started solve. Masterplan step 11 listed this verbatim ("CG `isfinite(rr)` (NaN compares false, so today a blown-up field reports a perfect solve at iteration 0)"); the GPU CG got it, the CPU one did not. Add `!std::isfinite(rr)` → error. Pairs naturally with the `int → expected<SolveReport>` change the masterplan §3 specified for both solvers and neither got (`cutcell.hpp:148`, `poisson.hpp:19`).

### F11 — `bodies.hpp` depends on `cutcell.hpp` instead of the reverse · risk: **low**

Masterplan §3 "Changed existing headers": *"`cutcell.hpp`: `RotatedCube` / `Sphere` / `CylinderZ` **moved** to `bodies.hpp` (`cutcell.hpp` includes it for the `CutGeometry::make` overloads)."* The shapes are still at `cutcell.hpp:28-45`, and `bodies.hpp:7` includes `cutcell.hpp` — with `bodies.hpp:2-4` documenting the inversion as if it were the design.

Consequence: anything that wants a `Lens` or an `Sdf` of a sphere — `scene.cpp`, `geometry.cpp`, `dns_main.cpp` — transitively pulls in `CutPoissonSolver`, `CutProjectionScratch` and `<vector>`-heavy CG workspace declarations. The 30-line move the masterplan specified is still available and still free; it is a header-only change plus three `#include` edits.

### F12 — the geometry layer writes the parse layer · risk: **low**

`scene.cpp:304-330` `applySceneModel(const Scene&, RawConfig&)` takes a **mutable `RawConfig`** and writes `r.les` / `r.sgs` into it, returning a `ConfigError`. `scene.hpp:15` therefore includes `dns/config.hpp`.

The documented order is `cli/config (pure parse+validate) → bodies/geometry`. A geometry module that mutates the parser's output inverts it, and it means the scene's model selection is applied *outside* `validate()`, so `--dry-run`'s purity claim depends on main calling the two in the right order (`dns_main.cpp:1325`). Move `applySceneModel` into `config.cpp` (it is pure config reconciliation — no geometry, no filesystem) and have it take `const Scene&`. `Scene`'s model fields are three POD members; `config.cpp` can take a small `SceneModelView { bool les; double cs; SgsModel sgs; }` and `scene.hpp` stops including `config.hpp` for anything but `Placement`.

### F13 — `config.hpp` has become the tree's hub header · risk: **low**

`output.hpp:10`, `scene.hpp:15` and `cli.hpp:5` all include `config.hpp`. `output.hpp` needs exactly two things from it (`Emit`, `RunMeta`); `scene.hpp` needs three (`Placement`, `ConfigError`, `SgsModel`). What they get is the 241-line `Case` with its 60 fields, five enums and `<set>`. Every touch of the `--focal` fields recompiles the output layer.

Split `Emit`, `RunMeta` and `Placement` into a 40-line `dns/runmeta.hpp` that `config.hpp` includes. Zero behaviour change; kills the `output → config` edge, which is the one arrow that runs backwards along the documented layering (`cli/config → … → output`).

### F14 — two parallel compressible RHS implementations · risk: **medium**

`comp.cpp:45-135` `rhsCurv` and `comp.cpp:167-245` `Comp::rhs` are the same algorithm written twice: same `faceFlux` lambda shape, same normal-compact/tangential-averaged face gradients, same `tau = mu(du_cd + du_dc) - 2/3 mu divu delta`, same three-axis accumulation loop. The Cartesian version is the curvilinear one with `S[a] = (A/h)·e_a` and a diagonal `Ji` — which is precisely what `CurvMesh::uniform` + `CurvMetrics::build` produce. Roughly 80 lines of duplicated, validated physics with two places to fix any future bug (and the `--pot`/curvilinear work already had to touch both).

I am **not** recommending the merge outright. Against it: it is a CPU reference, so the Cartesian fast path costs nothing to keep, and merging is not bit-identical (`(hi-lo)/h` per axis vs `(hi·S - lo·S)/vol`), which moves `test_comp` and `test_ke_conservation` numbers by roundoff. In favour: the divergence risk is real and `tests/test_comp_curv.cpp` already exists as the parity net. If it is done, do it as the masterplan §8 prescribes — motion without editing expressions is impossible here, so it must be a separately measured commit with the before/after recorded.

### F15 — `remap.hpp`: two two-sided tanh maps, three growth loops · risk: **low**

`TanhStretchY` (`remap.hpp:81-123`) and `ChannelStretchY` (`:137-198`) are both "two-sided tanh clustering along y". They are **not** redundant — `ChannelStretchY` adds the reflection-ghost extension (`:171-184`) and a `firstCell()` that fixes the box period, and its fluid map spans `[hy, (ny-1)hy] → [0, 2δ]` rather than `[0,ly] → [0,ly]`. Keep both; the shared part is only `tanh(β s)/tanh(β)`.

What *is* duplicated is the discrete growth-ratio scan, three times over three map types (§F5). One template in `remap.hpp` covers all three.

Also worth noting: `SinusoidalRemap` (`:47-72`) has exactly one consumer, `CurvMesh::sinusoidal`, which exists for the curvilinear parity tests **[x-ref agent 3]**. That is a legitimate test-only abstraction; leave it.

### F16 — help text has drifted from the tables it documents · risk: **none**

The 350-line `help()` (`cli.cpp:439-790`) is hand-maintained against three imperatively-built tables. Verified drift:

- `cli.cpp:502` — "`(cylinder -> 2D disks, sphere -> 3D caps)`" is stranded at the end of `--bump-h`'s paragraph; it belongs to `--dimples` (`:492`).
- `cli.cpp:511-518` — `--penalize`'s help says "drained momentum is not yet in the cd budget (**cd reads slightly low**)". CLAUDE.md records the measurement as "`--penalize` ~15% (**HALVES cdf** — never for friction physics)". The help understates a documented, measured factor-of-two on a friction quantity. Whichever is right, they must agree; CLAUDE.md is authoritative.
- `cli.cpp:770-778` documents `--emit png|web|bin` for `channel`, but the channel option table (`cli.cpp:175-182`) omits `--vort-max`, `--vort-signed`, `--vort-map` and `--speed-map` (all Flow-only, `:166-167`, `:208-211`, `:227`), and `validate()`'s channel branch never sets `c.vortMax` etc. A channel PNG cannot be colour-scaled at all — which is exactly the high-Re-saturation problem CLAUDE.md's memory records for flow runs.
- `output.cpp:138-141` — comment claims "flow still rejects it in validate()" for PNG at `nz > 1`; `config.cpp:1025-1027` explicitly removed that restriction. Stale in the opposite direction.
- `helpTop()` documents `--config FILE.toml` (`cli.cpp:407-411`), but `--config` is not in `parse()`'s `known` list (`cli.cpp:234-239`) because main strips it in phase 0.5 (`dns_main.cpp:1275-1301`). Harmless today; `dns flow --confg x.toml` gets no suggestion.

**Structural fix (recommended, one pass):** replace the three vectors + nine `if` blocks with one static `constexpr` table whose rows carry a subcommand bitmask, and generate the option lines of `help()` from it. That single change closes F7 (scoping falls out), the channel `--vort-max` gap, and the drift class permanently. It is ~120 lines of `cli.cpp` rewritten; the prose paragraphs (conventions, physics rationale) stay hand-written.

### F17 — `Scene::zInvariant()` disagrees with its own documentation · risk: **none**

`scene.hpp:54-55`: "True when every object is z-invariant (**no sphere, no mesh**)". `scene.cpp:198-204` also excludes `Cube` — but a scene cube is built as `RotatedCube{cx, cy, r, 0.0, lzFull}` (`scene.cpp:362`), i.e. spanning the whole box, and *is* z-invariant. The cost is only a slower `sampleCut` (all layers instead of one) for cube-only scenes; the output is correct either way. Fix the comment or the predicate — they cannot both be right.

### F18 — `RunMeta` is initialised positionally at four sites · risk: **none**

`config.cpp:110-112`, `:153-155`, `:269-271`, `:1034-1040` each brace-initialise all **17** `RunMeta` members by position, including three trailing strings and a `bool les` in the middle. Adding or reordering a field silently mis-assigns at every site (`double kick` sits between `physics` and `sgs`). Use designated initialisers, or a `RunMeta metaOf(const Case&)` helper — the four sites already share 12 of 17 values.

### F19 — `RunOutput` does not create every output up front · risk: **low**

`output.hpp:76-77` and masterplan §1.1 promise "creates and truncates every file and directory the run will ever write and fails here if any cannot be created — before any GPU allocation". `RunOutput::make` (`output.cpp:133-192`) probes `<prefix>.json`, writes `_mask.bin`, and creates the PNG directory. The `.bin` field streams open lazily on first use in `Impl::binFor` (`output.cpp:93-100`) with **no error check**, and `writeJson()` (`:87-91`) never tests the stream. A read-only or full destination for the `.bin` series surfaces at the first frame — potentially minutes in — instead of in the first second. The field names are not known at `make()` time, which is the real obstacle; either pass the expected field names into `Config`, or at minimum check `binFor`'s stream and propagate.

### F20 — housekeeping: `framemap.{cpp,hpp}` are untracked · risk: **none**

`CMakeLists.txt:23` already lists `src/dns/framemap.cpp` and `:134` registers the `framemap` test, and `apps/dns_main.cpp:13`, `gpu/runner_comp.hpp:10` include the header — but both source files and the test are untracked (`git status`), as is the `CMakeLists.txt` edit. A clean checkout does not build. Commit them **[x-ref agent 3]**.

---

## 3. Verified correct / leave alone

Read and deliberately *not* flagged:

- **`sgs.hpp`** (77 lines). The `σ3 <= 1e-6 σ1 → 0` floor (`:69-71`), the double-precision Cardano path with the "float cancels catastrophically" rationale, and the three `fmax` clamps that keep `νt >= 0` are all documented with their reason. The `nz == 1` rejection lives in the validator (`config.cpp:505-508`) with the right message. Exemplary; do not touch.
- **The anisotropy gotcha is not present in the host library.** Every `min(hx,hy,…)` in `src/dns` non-GPU is a *stability* minimum, which is the correct direction: `timestepper.cpp:28` (diffusive dt), `comp.cpp:303` (viscous dt), `config.cpp:852-858` (the `--focal` dt-penalty diagnostic). `sampleCut` correctly uses the **half cell diagonal** `0.5·sqrt(hx²+hy²+hz²)` for its band (`geometry.cpp:40`), not a min. The CLAUDE.md audit request is satisfied here; the deep-solid threshold it warns about is GPU-side **[x-ref agent 2]**.
- **`Focal1D`'s identity-by-branch discipline** (`remap.hpp:212-241, 277-289`): `active == false` returns `X` and `1.0` exactly, mirroring the `TanhStretchY β == 0` pattern, so the degenerate limit stays bit-pinnable. The compact-support quintic that makes the far grid *exactly* uniform is the right construction for the identity margin the validator enforces.
- **`PhysicalRaster`** (`framemap.{hpp,cpp}`): display-only, inactive-is-a-copy (`framemap.cpp:92-95`), degenerate axis never resampled (`:40`), and a hard cell cap that degrades detail rather than killing the run (`:52-61`). Exactly the right shape for a display path. The header comment states the invariant ("Nothing here touches the state, the fluxes or the statistics") and the code honours it.
- **`sampleCut`'s Lipschitz block prefilter and z-invariant replication** (`geometry.cpp:52-72, 111-120`): still the single sampler the masterplan specified, with the per-body budgets carried through as `SampleOpts` exactly as §step 4 required. `exactLensCut` is still separate because the lens tips are sub-cell — the documented reason.
- **The two grid conventions stay separate** (`config.cpp:387-435`), with the compressible one extended by `--dr` in a way that keeps `dr == 8` bit-identical (`:394-398`, the `*(dr/8)` factor and the preserved `2.1107` literal). This is masterplan §6.12 respected under pressure, which is the hard case.
- **Sponge is never a default** (`config.cpp:320-322`: compressible defaults to `Nscbc`), and the incompressible/compressible far-field cross-rejections (`:325-332`) are explicit. The `--pot` scoping (`:537-545`) and the `--nscbc-beta ∈ [0,1)` / `re > 800 ⇒ --nscbc-strip` rules (`:520-533`) all carry their measured justification inline. Leave.
- **Strict TOML in both parsers.** `scene.cpp:67-120` and `:144-170` iterate every key rather than looking up expected ones, so unknown keys are hard errors; `cli.cpp:369-376` rejects `flag = false` rather than silently ignoring it. Both inherit the recomposition's strictness correctly.
- **`closeChecked`** (`output.cpp:37-43`) and the `declareSeries` reopen guard (`:200-204`) — both are fixes for real, subtle failure modes, correctly commented.
- **`ChannelStretchY::firstCell` fixing the box period** (`remap.hpp:146-151` consumed at `config.cpp:200-201`): the ghost-row geometry constrains `ly`, and the validator derives it rather than asking the user. Correct and well-explained.
- **`CutPoissonSolver` (CPU) alongside the GPU solver** — masterplan §6.13, the documented reference/mirror pairing. Not accidental duplication; leave.

---

## 4. Risks

Ranked by (probability × blast radius) for the changes proposed above.

**R1 — F1's `qRef` fix moves every `--focal` body number (CERTAIN / HIGH).** That is the point, but it means any `cd` recorded from a focal run is superseded. Do the arithmetic fix and the CLAUDE.md/memory note in one commit, and re-run one focal cylinder against its uniform twin as the acceptance test (the two should agree on `cd` to the discretisation difference, not to a factor of 4). Do **not** quietly re-baseline.

**R2 — F5's `--focal` extraction is where numbers can move by accident (MEDIUM / HIGH).** The plateau derivation ends in `Focal1D::make(L, lo, hi, w, ratio)`; those five doubles fully determine the mesh. Copy the expressions verbatim, and gate the change on `--dry-run` producing byte-identical output for a set of `--focal auto` and explicit-BOX cases across body kinds. Anything that changes `focalLo/Hi/W` changes the mesh, hence `hmin`, hence `dt`, hence every trajectory.

**R3 — F7's flag scoping is user-visible and breaks scripts (CERTAIN / LOW).** Rejecting `--nz` on `bench` or `--emit spectrum` on `flow` turns silent no-ops into exit 1. Correct, but it must land as one announced commit, not scattered.

**R4 — F9's signature changes ripple into tests (CERTAIN / LOW).** `Comp::stableDt → expected<double,...>` and `CutPoissonSolver::make` touch `test_comp`, `test_comp_curv`, `test_cutcell`, `test_ke_conservation`, `test_projection` and `dns_main` **[x-ref agent 3, agent 2]**. Mechanical, but it is a same-commit rewrite.

**R5 — F14 (merging the two compressible RHS) cannot be bit-identical (MEDIUM / MEDIUM).** `(hi - lo)/h` per axis versus `(hi·S - lo·S)/vol` differ in roundoff even on a uniform mesh. `test_comp`'s tolerances and `test_ke_conservation` would need re-measuring. This is the one proposal here that is genuinely optional; if in doubt, keep both and add a comment in each pointing at the other.

**R6 — F13's header split touches every TU (CERTAIN / NIL).** Pure include surgery; the compiler is the test.

**R7 — nothing proposed here touches a CLAUDE.md invariant.** Specifically checked and *not* proposed: unifying `TanhStretchY`/`ChannelStretchY` (§F15 — the reflection extension is load-bearing for the wall-mirror ghost rows); branching any z-term (the `flat` paths in `geometry.cpp:36-59` and `config.cpp:126, 400, 417` are degenerate-axis *sizing*, not stencil branches, which masterplan §6.6 explicitly permits); relaxing the `--far sponge` never-a-default rule; touching the free-slip lateral treatment, NSCBC's `betaTrans` bracket, `etaIn`, the ghost-view/convection separation, or the `--pot` off ⇒ bit-identical property. One temptation was explicitly rejected: **merging the incompressible and compressible `qRef`/frontal-area formulas into one** (§F3) is blocked in spirit by masterplan §6.12 / CLAUDE.md's two-grid-conventions rule — merge the *code path*, but keep the two normalisations distinct and dispatched on `--physics`, or the compressible `span` handling silently rescales `cd`.
