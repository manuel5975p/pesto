# analysis_recompose_3 — safety, hacks, error handling

Agent 3 of the Simplification TPRI. Focus: no hacks; every function maximally safe and always returns something reasonable. No code was modified.

Scope read: all 16 `apps/*.{cpp,hip}`, all of `src/dns/`, all of `src/dns/gpu/`.

Headline: the **library** is mostly disciplined (`DeviceBuffer`/`DeviceField` are exemplary `std::expected` factories, `writeVtk` returns `std::expected<void,string>`). The **apps** are where the rot is: 4 mutually incompatible ad-hoc CLI parsers, none of which can detect an unknown flag, all of which convert user error into a silently-wrong physics run. The second worst band is *release-mode invariant loss*: ~30 preconditions that exist only as `assert`, i.e. only in Debug, while every production run is Release (per CLAUDE.md).

## 1. Ranked defect list

Severity: **S1** = silently produces wrong physics or corrupts memory in a Release run; **S2** = silent failure / unreportable error; **S3** = hygiene.

### S1 — silently wrong runs

| # | Site | What breaks | Fix |
|---|---|---|---|
| 1.1 | `apps/cylinder_tunnel_gpu_main.hip:479`, `apps/mesh_tunnel_gpu_main.hip:255`, `apps/comp_mesh_tunnel_gpu_main.hip:118`, `apps/eye_tunnel_gpu_main.hip:437`, `apps/movie_main.cpp:132`, `apps/cube_cut_main.cpp:96` | **Unknown-flag fallthrough.** The final `else n = int(parseOr(argv[a], n));` catches *everything* unrecognised. `--reynolds 500` → flag silently ignored, then `500` eaten as the **grid size**. `--Les 0.1` → `n = int(0.1) = 0`. A typo does not error; it reconfigures the grid. | Unified parser: unknown token starting with `-` is a hard error. Bare positional accepted only in a declared positional slot. |
| 1.2 | `apps/comp_cylinder_tunnel_gpu_main.hip:66` | `const int n = argc > 1 ? std::atoi(argv[1]) : 512;` — `--re 800` as first args gives `argv[1]=="--re"`, `atoi` → **n=0**. `Grid::make(0,…)` guarded only by `assert` (`grid.hpp:22`), so Release proceeds with a zero-size grid. | Check `argv[1][0] != '-'` (as `hit_main.cpp:48` already does correctly) and validate `n >= 1`. |
| 1.3 | `apps/comp_cylinder_tunnel_gpu_main.hip:37-41` (`arg()`) | `std::atof` returns **0.0** for non-numeric. `--re abc` → `reD=0` → `mu = rhoInf*uInf*D/0 = inf` (line 96). `--mach abc` → `pInf = inf`. No diagnostic. | `from_chars` with full-consumption check; error on failure, never a fallback. |
| 1.4 | `parseOr` (`hit_main.cpp:25`, `eye:47`, `movie:106`, `cube_cut_main.cpp:27`) | Correct `from_chars` but **silently substitutes the default** on garbage. `--cfl 0,8` → default CFL; the run looks fine and is not the requested run. | Parse failure is a user error: report and abort. |
| 1.5 | `apps/comp_cylinder_tunnel_gpu_main.hip:107`, `apps/cylinder_tunnel_gpu_main.hip:590`, `apps/mesh_tunnel_gpu_main.hip:409` | `std::clamp(int(0.1*n), 8, n/2 - 1)` — for `n < 18` hi < lo, which is **UB** (precondition violation). Reachable from the CLI (`… 16`). | Validate `n >= 18`, or derive `ns` with explicit `max(lo, min(hi,·))`. |
| 1.6 | `apps/comp_cylinder_tunnel_gpu_main.hip:39`, `:44` | Loop bound `i + 1 < argc`: a **trailing flag with no value is silently ignored** — `--re` last on the line gives default Re, no warning. | Missing value is an error. |
| 1.7 | `src/dns/gpu/comp.hip:365` | `stableDt` = `cfl/wmax`; `wmax==0` (quiescent/rest IC — which the `--ramp` quiet start *deliberately creates*) → **`dt = +inf`**. `rhomin <= 0` (`:376`) → negative `dtv` → negative dt. Never clamped, no NaN check. | Return `std::expected<double,…>`; reject non-finite/non-positive; clamp to configured `dtMax`. |
| 1.8 | `src/dns/gpu/cutcell.hip:443` | CG loop `rr > target`: **NaN is false**, so the loop exits at iteration 0 and `solve` returns `0` = "converged immediately". A blown-up field reports a perfect solve. | Explicit `std::isfinite(rr)` → `unexpected(NotFinite)`. |
| 1.9 | `src/dns/gpu/cutcell.hip:271`, `:383`; `src/dns/cutcell.cpp:452,489` | `nFluid > 0` is **assert-only**; `kFluidMean` then does `sum[0]/nFluid` = 0/0 = NaN, poisoning every `projectMean`. All-solid geometry (bad `--scale`, mesh larger than box) is plausible user input. | Validate at geometry construction; `unexpected(EmptyFluidDomain)`. |
| 1.10 | `src/dns/gpu/poisson.hip:254,291` | `assert(g.nx>=4 && g.ny>=4 && (g.nz==1 || g.nz>=4))` — the **NeumannBox n>=4 / degenerate-axis invariant is Release-invisible**. Then `mz = nz-2` is 1 or negative and `hipfftPlan1d` gets a nonsense length. | Return `unexpected(hipErrorInvalidValue)`; also enforce in the CLI validator (§3). |
| 1.11 | `src/dns/diagnostics.cpp:108` | `nShells = min({nx,ny,nz})/2` is **0 when nz==1**, then `shellE[0] = 0.0` is an **OOB write** on an empty vector. Hits every 2D run asking for a spectrum. | Guard `nShells > 0`; compute 2D shell count for `nz==1`. |
| 1.12 | `src/dns/hit_ic.cpp:55,74,143` | Same `nz==1 ⇒ nShells==0`: every mode zeroed, `fluctuationEnergy` = 0, `assert(kPrime>0)` is the only guard before the division at `:144` → Release gives an **all-`inf` field**. | Reject `nz==1` for HIT in the validator, or handle 2D shells. |
| 1.13 | `src/dns/trimesh.cpp:179-181` | glTF index from `cgltf_accessor_read_index` used **unbounded** as an index into `id` — OOB read from a corrupt mesh. (The float read at `:170` *is* checked; indices are not.) | Bounds-check against `pos->count`, return `unexpected`. |
| 1.14 | `src/dns/trimesh.cpp:43-44` | `f.tellg()` failure returns `-1` → `std::vector<char> buf(size_t(-1))` → ~2^64 allocation. | Check `tellg() >= 0`. |
| 1.15 | `src/dns/meshsdf.cpp:248` | `tris_[h.tri]` with `h.tri` default `-1` (`:244`); only guard is Release-disabled `assert(!tris_.empty())` at `:112`. A mesh whose triangles are all degenerate-filtered (`:110`) gives `tris_[-1]`. | Make "no triangles survived" a construction-time `unexpected`. |
| 1.16 | `src/dns/gpu/cutcell.hip:161-165`, `reductions.hip:34-38`, `comp.hip:175-179` | Reduction scratch regrowth: `assert(grown)` is a no-op in Release, then `*grown` **dereferences a valueless `expected`** (UB) and rocprim gets undersized scratch. Three copies of one bug. | Single shared helper returning `std::expected`. |

### S2 — silent failure / unreportable errors

| # | Site | What breaks | Fix |
|---|---|---|---|
| 2.1 | `src/dns/poisson_jacobi.cpp:17,44`; `poisson.hpp:19` | `solve()` returns **`void`**. `resMax` computed and thrown away; exhausting `maxIters_` is indistinguishable from converging. | `std::expected<SolveReport,PoissonError>` carrying `{iters, residual}`. |
| 2.2 | `gpu/cutcell.hip:443,459-461,480`; `cutcell.cpp:511-537` | CG returns only an iteration count; convergence and the `dq<=0` breakdown share one exit; `projectCut` (`cutcell.cpp:537`) **discards the return entirely**. Unprojected velocity, no signal. | Same `SolveReport`; `projectCut` propagates. |
| 2.3 | `gpu/poisson.hip:343-344,354-355,378-379,401-402` | All four `hipfftExec*` statuses are `[[maybe_unused]] … assert(…)` — **discarded in Release**. A failed transform leaves stale memory feeding the CG preconditioner. | Check and return. |
| 2.4 | `gpu/poisson.hip:279-284,319-322` | Plan-creation errors collapsed by short-circuit `&&` into a bare `hipErrorUnknown`; the real hipFFT status is lost. | Per-call status propagation. |
| 2.5 | `poisson_fft.cpp:20-25,43,61`; `hit_ic.cpp:35-38`; `diagnostics.cpp:62-64` | `fftw_alloc_*` may return null and `fftw_plan_*` may return a **null plan**; none checked; `fftw_execute(nullptr)` is UB. Constructors have no failure path. | Convert to `make()` factories returning `std::expected`. |
| 2.6 | `comp_cylinder_tunnel_gpu_main.hip:200,202,210,214,215,343`; `comp_mesh_…:325,328,350,365,479,497,521`; `movie_main.cpp:38,193`; `hit_main.cpp:66,75` | **Every app output stream is opened and written unchecked.** A read-only cwd or full disk burns hours of GPU, writes nothing, reports success. | Open-and-check all outputs up front, before any GPU allocation (§2 phase 3c). |
| 2.7 | `webframes.cpp:56`, `fieldpng.cpp:67`, `vtk_writer.cpp:33` | `return bool(f)` **before close**: buffered ENOSPC surfaces only in the destructor → success reported on a truncated file. | Explicit `f.close()` then test. |
| 2.8 | `webframes.cpp:26-27` | `create_directories` error_code captured and **discarded**; all later `writeFrame` calls fail silently. | Factory returning `std::expected`. |
| 2.9 | `webframes.cpp:88-113` | Manifest JSON built by concatenation with **no escaping** of `f.name`/`simJson_`; `--out 'a"b'` emits malformed JSON, viewer silently shows nothing. | Escape, or reject non-portable `--out` in the validator. |
| 2.10 | `webframes.cpp:75` | `snprintf` truncation return ignored → silently colliding output paths for long `dir_`. | Check, or use `std::format`/`std::filesystem::path`. |
| 2.11 | `gpu/comp_ghost.hip:212-239` | `DeviceCompGhost out;` populated field-by-field; the `up` lambda (`:221`) returns `hipSuccess` **without assigning** when `src.empty()`, leaving null buffers the `:245/251/263` count-only guards miss. | Build all buffers first, construct last (`device_buffer.hpp:71` pattern). |
| 2.12 | `gpu/comp_bc.hip:89-98` | `make()` returns `std::expected` but **validates nothing** — no `gamma>1`, no `nx>=3` although `kBcX` reads `i=2` and `nx-2`. Contrast `comp_far.hip:102-105` / `comp_nscbc.hip:203-207`, which do validate. | Add checks — or delete the module (§4). |
| 2.13 | `gpu/cutcell.hip:322-344` | `DeviceCutSolver::make` validates **no parameter**: `maxIters<=0` → `solve` returns 0 with `p` zeroed; `tol` NaN → instant false convergence. | Validate `maxIters>=1`, `tol>0 && isfinite`. |
| 2.14 | `gpu/cutcell.hpp:19-25,87` | `DeviceCutGeometry` carries **no grid dims**, so nothing can assert `cut` matches `g`; `cut_` is a raw non-owning pointer in a movable class → dangling after move, lifetime rule only in a comment (`:57`). | Store dims + assert on bind; hold a `const&` or document non-movability. |
| 2.15 | `gpu/timestepper.hip:46`; `timestepper.cpp:27-31` | `lim > 0 ? min(dtMax, cfl/lim) : dtMax` — a **NaN `lim` makes both comparisons false**, so the stepper proceeds at `dtMax` with a NaN field. No blowup detection in either time loop. | `isfinite` → `unexpected(FieldNotFinite)`; periodic NaN sentinel in the step loop. |
| 2.16 | `comp.cpp:39,55,134,165-166`; `gpu/comp.hip:113-116`; `gpu/comp_nscbc.hip:76,165,180` | Positivity of `rho`/`p` is **assert-only**; Release divides by zero and `sqrt`s negatives, NaN propagates for the whole run. `comp_nscbc.hip:165` also divides by `kLatBand*cs`, `:180` by `2*r*cs` — NaN at `cs==0`. | State-admissibility check at a configurable cadence; hard stop with a clear message. |
| 2.17 | `meshsdf.cpp:34,150,155` | `normalized()` returns `{0,0,0}` for a zero-length vector — **failure indistinguishable from a valid normal**. Consumed at `:252` where `dot(...)<0` is then always false, so the SDF reports "outside" everywhere near a degenerate feature (silently inverted body). | Return `std::optional`; treat as a mesh-validation failure. |
| 2.18 | `meshsdf.cpp:123` | `watertight_` computed and **never used, asserted, or reported**; a non-watertight mesh silently yields a wrong-sign SDF in the non-`--robust` path. | Surface it: refuse non-watertight meshes unless `--robust`. |
| 2.19 | `meshsdf.cpp:68` | `denom = 1.0/(va+vb+vc)` with no zero check → inf/NaN closest point for a degenerate triangle that slipped the `:110` filter. | Guard. |
| 2.20 | `trimesh.cpp:108-118,199` | `bounds()` on an empty mesh silently returns the **inverted sentinel** `{1e300,-1e300}` (the "Pre: mesh non-empty" at `trimesh.hpp:25` is unenforced); `transform()` returns `void` and does not validate `scale` — `--scale 0` collapses the body to a point. | Assert non-empty; validate `scale > 0 && isfinite` in the validator. |
| 2.21 | `gpu/comp_ghost.hip:80-82` | Body-force tally accumulates in **FP32 `atomicAdd`** while every other reduction is FP64 (`reductions.hip:25`, `cutcell.hip:152`). Cancelling ± contributions lose precision and atomic ordering makes reported cd **irreproducible run-to-run** — the number the anchors rest on. | FP64 accumulation or a deterministic two-pass reduction. |
| 2.22 | `gpu/poisson.hip:320` | `int(total / n[p])` batch count: `total` is `size_t`; a large interior box wraps the batch argument negative. `assert(n <= INT_MAX)` in the reducers (`cutcell.hip:154`, `reductions.hip:27`, `comp.hip:170`) is likewise assert-only → **fails only in Release**, the only build that runs. | Checked narrowing helper returning `unexpected`. |
| 2.23 | `poisson.hip:341`; `cutcell.hip:419,422,435,459,169`; `reductions.hip:42`; `comp.hip:183,338`; `comp_ghost.hip:266,275`; all `rocprim::reduce` statuses; `device_buffer.hpp:41` | ~20 `hipError_t` discarded via `(void)`. `cutcell.hip:459` is the memcpy of `[rr,dq]` — **the value driving the CG loop**. | `DNS_HIP_TRY` on every call; `(void)` only in destructors, with a comment. |
| 2.24 | `cutcell.cpp:306` | `while (changed)` pocket-closing fixed point has **no iteration cap and no diagnostic**, while `cutcell.hpp:82` claims convergence. | Cap + report; exceeding it is a geometry error. |
| 2.25 | `trimesh.cpp:129,51-67,35` | Binary-STL detection requires `n>0`, so a zero-triangle binary STL falls into the **ASCII parser scanning binary floats for the token `vertex`** and can accept garbage. Coordinates never validated (NaN/inf reach the BVH). Degenerate triangles dropped silently with no count. | Explicit format detection on the 80-byte header + size arithmetic; validate finiteness; report drop count. |
| 2.26 | `diagnostics.cpp:37` | Throwing `std::filesystem::exists` overload inside a `bool`-returning function, in a codebase that abstains from exceptions. | `error_code` overload. |
| 2.27 | `gpu/reductions.hip:155-156,162,136` | Max-reductions seeded `init = 0.0` — an empty set returns 0 rather than being rejected; `meanVelocity` divides by `g.size()` with no zero guard. | Reject empty; seed with `-inf`. |

### S3 — hygiene / magic numbers

- `cutcell.cpp:139` recursion cap `24`, `:180` tol `1e-12` — hitting the cap silently returns a low-accuracy volume while `cutcell.hpp:56` promises "abs err ~1e-12".
- `cutcell.hpp:65` `openTol = 1e-9` **absolute, not scaled to h**; `:85` `lambdaMin = 1e-2`, `apertureMin = 1e-2` undocumented.
- `cutcell.cpp:90,164` `std::array` sized to exactly the max `bp[nb++]` writes (6 and 18) with **no `assert(nb < bp.size())`** — correct today, silent stack corruption the day a break point is added.
- **Two different undocumented cutoffs for the same "image buried" condition**: `comp_ghost.hip:180` `wsum < 0.05` vs `ghostfill.hip:128` `wsum < 0.25`. One is wrong, or they need distinct names and justifications.
- `comp_ghost.hip:141` `1e-3*h` FD step, `:148`/`ghostfill.hip:96` `1e-12`, `:156` `0.75*h` (this one *is* physics-justified, §4).
- `comp.hip:275` `DeviceBuffer<const Real*>::make(20)` vs 20 pointers laid out by hand at `:333-337` — **no static assertion tying the two**.
- `1e300` as an infinity sentinel in five places (`trimesh.cpp:110`, `meshsdf.cpp:96,172-173`, `comp.hip:377`).
- Unnamed constants: `comp.cpp:33` `0.72`/`1.0`, `:179` `0.25`; `comp.hpp:87-90` `Pr_`,`Rgas_`,`prt_`; `timestepper.cpp:27` `6.0`, `:31`/`timestepper.hip:48` `0.2/forcing_.A`; `comp_nscbc.hip:17` `kLatBand = 0.01` (comment concedes it should be config); `device_util.hpp:22` `blockSize = 256`; `poisson.hip:82-83,204` tile sizes; `units.hpp:47-52` `0.19`/`1.5`/`15.0`.
- `poisson_fft.cpp:56` `lam == 0.0` exact float equality for the DC mode.
- `gpu/poisson.hpp:39` requires `rhs` and `p` distinct; nothing checks it.

## 2. Error-handling policy for the unified binary

### The rule

> `std::expected` for anything a **user** can cause. `assert` for anything only a **programmer** can cause. Nothing in between, and no third category.

**`std::expected<T, Error>`** — CLI parsing, config validation, file open/read/write/close, mesh loading and validation, every GPU resource acquisition (`hipMalloc`, plan creation), every `hipError_t`-returning call, solver convergence, and state admissibility (NaN/negative density). All are reachable from a command line or a filesystem, therefore all are recoverable errors and none may be an `assert`.

**`assert`** — only invariants that hold by construction given a validated config: index bounds inside kernels, buffer sizes matching `g.size()`, the `nb < bp.size()` counts, "geometry dims match grid dims", "`nFluid_ > 0`" *after* the validator has rejected all-solid geometry. An `assert` is legitimate **only if** the §3 validator provably makes it unreachable. Every assert today guarding a user-reachable condition (`grid.hpp:22`, `poisson.hip:254,291`, `cutcell.hip:271`, `comp.cpp:39,55`, `hit_ic.cpp:143`, all `n <= INT_MAX`) migrates to the validator or to a returned error. This single migration kills most of the S1 band, because the project builds Release.

**Never**: silent defaulting on parse failure (`parseOr`), `atof`/`atoi`, discarding a status via `(void)` or `[[maybe_unused]] + assert`, or a sentinel return a caller cannot distinguish from success.

**`(void)` exemption** — destructors only (`device_buffer.hpp:41`, `poisson.hip:240-243`), where there is no one to report to. Must carry a comment.

**HIP status discipline** — one macro on every HIP/hipFFT/rocprim call:

```cpp
#define DNS_HIP_TRY(call) \
    if (auto e_ = (call); e_ != hipSuccess) \
        return std::unexpected(Error::hip(e_, #call, __FILE__, __LINE__))
```

Preserves the *actual* status and the call site, fixing 2.3, 2.4 and 2.23 in one pass. Kernel launches keep `checkLastKernel` (`device_util.hpp:30`) — a bad launch config genuinely is a bug — but it should also `hipDeviceSynchronize` in Debug so the error is attributed to the right kernel.

**CLI errors** — `stderr`, one line, offending token quoted, nearest known flag (Levenshtein over the flag table):

```
dns: unknown option '--reynolds' (did you mean '--re'?)
dns: try 'dns --help' or 'dns <case> --help'
```

Config errors name the constraint and both values:

```
dns: --nz 1 is incompatible with case 'hit' (spectral IC needs nz >= 4)
dns: --les 0.1 requires nz == 1 (Smagorinsky path is 2D-only)
```

No error message is ever a warning the run continues past.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | run completed, all output flushed and closed successfully |
| 1 | CLI error: unknown flag, missing value, unparseable number |
| 2 | config-validation error: a §3 constraint failed |
| 3 | input error: mesh missing / unreadable / malformed / non-watertight |
| 4 | output error: cannot create dir, open, write, or close an output |
| 5 | no usable GPU device (today's `deviceAvailable()` path, which returns 1) |
| 6 | resource acquisition failed: `hipMalloc`, FFT plan, host allocation |
| 7 | runtime numerical failure: non-finite state, negative density, dt not finite |
| 8 | solver failed to converge within its cap (CG / Jacobi) |
| 77 | skipped (preserves the ctest no-device convention) |

Distinguishing 5/6/7/8 matters for the validation suite: a Tier-C anchor run dying at 7 is a physics regression; the same run dying at 6 is a machine problem. Today both print "GPU alloc failed" and return 1.

### Failing before any allocation or GPU work

`main` is a fixed four-phase pipeline; **phase order is the mechanism**:

1. **Parse** — pure string→`RawConfig`. No filesystem, no GPU. `expected<RawConfig, CliError>`.
2. **Validate** — pure function of `RawConfig` (§3). No filesystem, no GPU. `expected<Case, ConfigError>`.
3. **Acquire** — in order: (a) probe GPU; (b) load and validate mesh; (c) **create and truncate every output file and directory the run will ever write, checking every one**; only then (d) `hipMalloc` / FFT plans / geometry.
4. **Run** — the time loop, which by construction can now only fail on 7 or 8.

Phase 3c fixes 2.6: today a full disk is discovered after the run. Because phases 1–2 touch neither filesystem nor HIP, *every* bad flag combination in §3 is caught before a byte is allocated — the stated requirement — and `--dry-run` (validate, print resolved `Case`, exit 0) falls out free and becomes the natural regression test for the validator.

## 3. Validation layer

```cpp
struct Case { /* fully resolved, all derived quantities computed, no optionals */ };

std::expected<Case, ConfigError>
validate(const RawConfig& raw);   // pure: no I/O, no GPU, no allocation beyond Case
```

`Case` is the **only** thing the solver sees. Its invariant: every field is in range and every cross-field combination is consistent, so downstream `assert`s are legitimate. Constructing a `Case` other than through `validate` is prevented by a private constructor + friend.

### Constraints, exhaustively

**Grid**
1. `nx,ny,nz >= 1`; no overflow in `nx*ny*nz` (checked in `size_t`, and `<= INT_MAX` — kills 2.22).
2. Only `z` may be degenerate: `nz == 1` OK; `nx == 1 || ny == 1` reject.
3. `nz == 1 || nz >= 4` (the `closeBoxBoundary` / NeumannBox / DCT invariant).
4. Confined axes need `n >= 4` each (NeumannBox); periodic axes `n >= 2`.
5. `nz == 1 ⇒ hz == 1` (unit depth), enforced not assumed.
6. `Lx,Ly,Lz > 0` and finite; `h` uniform per axis and > 0.
7. Total cell count within a memory budget computed from the selected solver's per-cell footprint — reject before `hipMalloc` (6).
8. Ghost-ring convention consistent: apps adding `+2` per axis must do so uniformly; `nz == 1` gets **no** z ghosts.

**Solver selection**
9. Exactly one of {incompressible, compressible}.
10. Poisson backend ∈ {FFT-periodic, DCT-NeumannBox, CG-cutcell}, compatible with the boundary set: NeumannBox is GPU-only; FFT-periodic requires all axes periodic; cut-cell requires a body.
11. CG: `tol > 0 && isfinite`, `maxIters >= 1` (kills 2.13).
12. Compressible ⇒ no Poisson requested; incompressible ⇒ no `--mach`.

**Physics**
13. `Re > 0` and finite (kills 1.3).
14. `0 < Mach < 1` for compressible (no shock capturing); warn/reject above ~0.3 for KG-central validity.
15. `gamma > 1`; `Pr > 0`; `Pr_t > 0`; `rhoInf > 0`; `pInf > 0`; `uInf` finite.
16. `0 < cfl <= 1`.
17. `ft, fdt, pdt, wdt, voldt > 0` and finite; each cadence `<=` total run time (else the run writes 0 frames — silently, today).
18. `ramp >= 0`; `kick` finite.
19. `Cs >= 0`; **`Cs > 0` ⇒ `nz == 1` for the app-local incompressible Smagorinsky path** — today checked, but only at `cylinder_tunnel_gpu_main.hip:516`, *after* geometry construction and the `sanitizeCut` print. The shared compressible-core LES is 3D-valid and carries no such restriction; the validator must distinguish the two.
20. Resolution sanity: warn (do not reject) when `D/sqrt(Re)` is under ~3 cells — the DNS-vs-LES honesty rule. Record as a flag in `Case` so output metadata says "LES" vs "DNS" automatically rather than by human discipline.

**Geometry / body**
21. Body kind ∈ {none, cylinder-z, sphere, rotated cube, mesh}; mesh kind requires a recognised extension (`hasMeshExt`), and a recognised extension requires mesh kind.
22. `scale > 0 && isfinite` (kills 2.20); auto-scale requires `ext0 > 0` (the mesh apps already check at `mesh_tunnel_gpu_main.hip:292` — keep, hoist).
23. `dRatio > 0`; `D >= 4h` (a body under 4 cells across is not a body); `D < min(Lx,Ly,Lz)`.
24. Body AABB after scale/rotation/placement lies strictly inside the domain with the ghost/taper band clear — prevents the all-solid `nFluid == 0` disaster (1.9) and the geometry-broken "body 67% downstream" case CLAUDE.md records as a real past bug.
25. `golf > 0 ⇒` 2D cylinder case, and the `8h` footprint / `4h` depth must fit: `D/h` large enough that dimples do not consume the body.
26. Mesh watertight unless `--robust` (surfaces 2.18); triangle count > 0; all coordinates finite.

**Boundaries**
27. Free-slip walls and `--open` mutually exclusive per axis.
28. `--nscbc` with `--open`: allowed for cylinder; for the mesh case `--open` no longer exists — reject the flag rather than ignoring it. (CLAUDE.md records the earlier "warn+ignore" behaviour — exactly the silence this policy forbids.)
29. `nsstrip >= 0` and `< nx/4`; `nsstrip > 0` requires `--nscbc`.
30. `nsbeta ∈ [0,1)`; `>= 1` rejected outright — CLAUDE.md records `beta = 1.0` **blows up at t=35** (full cancellation ill-posed at low M). A measured instability boundary belongs in the validator.
31. `--nscbc` alone with `Re > 800` ⇒ require `--nsstrip` (pure NSCBC blows up at Re2000/2048²; rule is `--nsstrip ~ ny/64` beyond Re800). Reject, or accept with a loudly recorded metadata warning — never silently.
32. Sponge width `ns = clamp(0.1*nx, 8, nx/2-1)` derived **only after** validating `nx >= 18`, else the bounds invert (1.5).
33. `etaIn > 0`; CLAUDE.md records `etaIn = 2` sagging inflow to 0.79 uInf and under-predicting cd by 30% — floor it (e.g. `>= 50`) or warn explicitly.

**Output**
34. `--out` non-empty, no path separators unless intended, JSON-safe (kills 2.9); directory creatable and writable (checked in phase 3c — `validate` stays pure).
35. Requested outputs producible by the selected case: PNG frames require `nz == 1`; web slices require the fields the case computes; spectrum output requires `min(nx,ny,nz)/2 >= 1` (kills 1.11/1.12).
36. No two requested outputs collide on the same path.

`ConfigError` is `{ code, field, message }` — one struct, formatted at the top level, so every constraint produces a uniform, greppable diagnostic.

## 4. Hacks

### Remove

- **All four CLI parsers** (`arg`/`argS`/`argB` at `comp_cylinder_tunnel_gpu_main.hip:37-56`; the `parseOr` + fallthrough loop in five apps; the `atoi(argv[1])` positional at `:66`; `hit_main.cpp:48`'s variant). One table-driven parser. Largest safety win available, and exactly what the recomposition is for.
- **The unknown-token fallthrough** `else n = int(parseOr(argv[a], n))` (1.1) — not a convenience, a silent misconfiguration engine.
- **`(void)`-discarded HIP statuses** and the `[[maybe_unused]] + assert` idiom on `hipfftExec*` (2.3, 2.23).
- **`assert` used as a user-input guard** (1.9, 1.10, 2.16, `grid.hpp:22`) — must become validator constraints or returned errors, because Release is the build that runs.
- **`1e300` infinity sentinels** (five sites) → `numeric_limits::infinity()`.
- **Three duplicated reduction-scratch regrowth blocks** sharing one valueless-`expected` UB (1.16) → one helper.
- **`src/dns/gpu/comp_bc.*` entirely.** CLAUDE.md: passes unit tests but *regresses real flows* (cd σ 10×); both comp apps now use `comp_far`; its `make()` validates nothing (2.12). Dead, known-bad code with a footgun API is precisely what a recomposition deletes. Keep the unit tests only if the LODI-replacement work is actually queued.
- **The unused `watertight_` computation** (2.18) — surface it or drop the cost. Computing a correctness signal and discarding it is the worst of both.
- **The two divergent `wsum` cutoffs** (0.05 vs 0.25) — reconcile to one named constant, or name them separately with a comment explaining why the compressible and incompressible ghost fills differ.

### Keep — validated physics, not mess

- **The ghost view never feeds convection** (`ghostfill.*`, `comp2d_ghost.*`). Mirror values in the skew-symmetric form *inject energy* — measured, tested. This *looks* like an inconsistency (why does diffusion see a different field than convection?) and a cleanup pass would "unify" it and silently break energy stability. Document harder, don't remove. Ideally the type system enforces it: a distinct `GhostView` type the convection kernels physically cannot accept.
- **`comp_ghost.hip:156` sampling `>= 0.75h` beyond the wall**, not plain `2|sdf|` mirroring. Barely-inside cells lose their stencil and the force tally's closed surface breaks. Keep the constant, name it, cite the reason at the definition.
- **`betaTrans = 0.5` default and the `< 1` bound.** Measured: 0 pumps a growing global mode, 1.0 blows up at t=35. Not a magic number — an empirically bracketed stability window. Promote to a named constant *with the bracket recorded*, enforce via constraint 30.
- **`etaIn = 100` stiff inflow.** Measured: `etaIn = 2` → cd 30% low. Same treatment.
- **Free-slip (not Dirichlet) lateral tunnel walls.** CLAUDE.md explicitly warns not to reintroduce the Dirichlet far field. A simplification pass noticing the mirror/antimirror asymmetry and "regularising" it would undo a validated fix. Add a regression test that fails if a Dirichlet lateral row reappears.
- **`nz == 1` as the exact 2D restriction with no separate code path.** The z-stencil cancellation is exact by address arithmetic. Do **not** "clean up" by branching z-terms out — the one refactor guaranteed to break the `*2d` parity tests. The validator enforces the *constraints* around it (only z degenerate, `hz == 1`) rather than the code special-casing more.
- **`lamz = {0}` / `hipfftPlan2d` degenerate-axis switches** — legitimate, documented special cases.
- **Recomputing `stableDt` every 16 steps with a 0.95 margin.** A perf fix that changes trajectories but not statistics, deliberately. Keep, name both constants, record the rationale — otherwise someone "fixes" it back to every-step for determinism.
- **The `bbox` culling parameter** in `DeviceCompGhost::make` and the IC taper. Reads like premature optimisation; is a ~20× setup win (26s → 1.3s at 256³), bit-identical. `bbox = nullptr` stays as the analytic/test path.
- **The deterministic solenoidal `--kick` trip.** Looks like a physics-perturbing hack; is div-free (silent) and deterministic, and the quiet start is genuinely too symmetric to shed without it. Keep, and have the validator record it in output metadata so no anchor is ever compared across different kick settings by accident.

### Judgement call

`comp_ghost.hip:80-82`'s FP32 atomic force tally (2.21) *looks* like a keeper because the anchors were measured with it — but non-determinism in the primary validated quantity is not a physics decision, it is a bug the anchors have absorbed. Recommend fixing it deliberately, with an anchor regeneration pass, rather than inheriting it into the unified binary.
