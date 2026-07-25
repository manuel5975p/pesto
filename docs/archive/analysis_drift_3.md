# analysis_drift_3 — tests, build, and documentation coherence

TPRI agent 3 of 3 (post-recompose drift). Focus: `tests/`, `CMakeLists.txt`,
and documentation coherence (`README.md`, `CLAUDE.md`, `docs/*.md`).
**No code was modified** — this is a read-only analysis. All line references
are to the working tree as of **2026-07-25**, including uncommitted
modifications (`CMakeLists.txt`, `README.md`, `tests/test_cli.cpp`,
`tests/test_gpu_comp_curv.cpp`) and untracked files (`src/dns/framemap.*`,
`tests/test_framemap.cpp`). `src/dns/` non-GPU is agent 1's lane;
`src/dns/gpu/` + `apps/dns_main.cpp` is agent 2's — cross-references are marked.

---

## 0. Executive summary

The recomposition landed a genuinely good test suite and then a week of feature
work landed on top of it without touching the surrounding paperwork. The result
is not test rot — the *unit* layer is in better shape than the library — it is
**three coupled coverage/documentation failures**:

1. **The verification layer's headline number is a lie by omission.**
   `docs/validation.md` still documents Tier A as "runs in every ctest sweep,
   ~2.8 min GPU total" (`docs/validation.md:20`) and lists six Tier-A rows
   (`:56-62`), of which four are unregistered by default
   (`CMakeLists.txt:229,248,284`) and a fifth (`val_comp_cyl_re60`) *always*
   exits 77 because every check is anchor-gated and every anchor row in
   `data/validation/anchors.csv` is commented out
   (`tests/test_val_comp_cyl.cpp:34`). **Exactly one validation test actually
   runs: `val_spindown`.** That is the documented design; what is not documented
   is that it is now the *whole* tier.

2. **The two time loops — the riskiest code the recomposition produced — have
   no library-level test and, for the default incompressible far field, no
   end-to-end test either.** `gpu/runner_incomp.hpp`, `runner_comp.hpp` and
   `runner_channel.hpp` are the only public headers in the tree no test
   includes. `val_spindown`, the stated canary, builds its own time loop out of
   `kernels.hpp`/`cutpoisson.hpp` primitives (`tests/test_val_spindown.cpp`
   includes, lines 2-8) and never enters a runner. End-to-end runner coverage in
   the default build is two tests: `gpu_open_incomp` (`--far fringe`) and
   `gpu_tunnel_open` (`--far tunnel --open`). **`--far tunnel` with walls — the
   incompressible default — is exercised by nothing**; its coverage was
   `val_cyl_re60`, which is quarantined.

3. **`docs/` still describes the pre-recomposition 15-binary CLI.** The Tier-C
   anchor-regeneration recipe — the procedure CLAUDE.md:43 and README §15 both
   point at as the way out of the quarantine — is a list of commands that no
   longer exist (`docs/validation.md:140-143,155,157`). So does every `config`
   field in `anchors.csv`. `docs/nscbc.md` names `comp_cylinder_tunnel_gpu`,
   `comp_mesh_tunnel_gpu`, `--nsbeta` and `--nsstrip` in nine places. The
   masterplan (`masterplan_recompose.md:1144-1167`) specified the replacement
   text; it was never applied.

Plus one active contradiction of a binding invariant: **README §7's heading
calls `--far sponge` the compressible default** (`README.md:254`), which
`CLAUDE.md:29` forbids in capitals and `src/dns/config.cpp:321` disproves.

Mechanical duplication is real but secondary: **24 near-identical
`check(bool, const char*)` definitions**, two competing pass/fail protocols,
three isentropic-vortex constructors, and ~200 lines of hand-written CMake
target boilerplate that one function collapses.

---

## 1. Inventory

### 1.1 Census

| | files | lines |
|---|---|---|
| `tests/` total | 66 (60 `.cpp`, 2 `.hip`, 4 `.hpp`) | 11 167 |
| — shared headers (`*_util.hpp`) | 4 | 570 |
| — `test_gpu_*` | 27 | 6 232 |
| — `test_val_*` | 10 | 875 |
| — CPU/library tests | 21 | ~3 490 |
| `src/dns/` non-GPU | — | 10 550 |
| `src/dns/gpu/` | — | 7 889 |
| `apps/dns_main.cpp` | 1 | 1 363 |
| `CMakeLists.txt` | 1 | 346 |

Registered ctest names in a default `-DDNS_WITH_HIP=ON` build: **54**
(20 core incl. 2 FFTW-gated, 27 `gpu_*`, 2 HIP-TU `gpu_*`, `errors`,
`gpu_open_incomp`, `gpu_tunnel_open`, `val_spindown`, `val_comp_cyl_re60`).
Executables *built* but registering **zero** tests by default: **8**
(`test_val_cyl_re60/re100/re200/blockage`, `test_val_tgv3d`,
`test_val_sphere_re300`, `test_val_mesh_sphere`, `test_val_comp_mesh_sphere` —
`CMakeLists.txt:222-228,241-247,266-283`).

Every `tests/test_*.{cpp,hip}` is referenced by `CMakeLists.txt`; no stale
target references a deleted source; no orphan test file exists. Both
uncommitted additions are wired (`CMakeLists.txt:23` `framemap.cpp`,
`:134` `framemap` test).

### 1.2 Coverage map

**Heavily tested (good).** Cut-cell geometry and operators (`cutcell`,
`cutcell_sphere`, `gpu_cutcell`, `gpu_cutcell2d`, `gpu_cutcell_sphere`,
`geometry`, `meshsdf`); NSCBC (`test_gpu_comp_nscbc.cpp`, 729 lines — the
largest test in the tree, covering six-face, curvilinear β = 0 equivalence, and
the `--pot` neutrality pin at `:587,721`); the boundary-kernel identity oracles
(`test_gpu_tunnel_bc.hip` 507 lines, `test_gpu_bodyforce.hip` 244, both keeping
the *pre*-recomposition kernels as oracles — the right shape); curvilinear/focal
(`comp_curv` 409, `gpu_comp_curv` 524, `gpu_comp_ghost_focal` 316); CLI
validation (`test_cli.cpp` 280 lines, ~150 accept/reject rows).

**Bit-identity invariants: pinned, and correctly.** `csLes == 0`
(`test_gpu_comp_les.cpp:117`), neutral `--pot` in both physics
(`test_gpu_comp_nscbc.cpp:721`, `test_gpu_tunnel_bc.hip:440`), inactive focal
map (`test_gpu_comp_ghost_focal.cpp:94`), all-open `--cut-flux`
(`test_gpu_cutflux.cpp:127`), `nz == 1` unified tunnel kernel vs the Class-A 2D
variant (`test_gpu_tunnel_bc.hip:142`), sigma at `nz == 1` on a distorted mesh
(`test_gpu_comp_curv.cpp:412`, uncommitted). The `*2d` parity tests CLAUDE.md
demands stay green **do exist and do test the stated thing**:
`test_gpu_cutcell2d.cpp:130` asserts `w` stays zero and `:139-140` compares 2D
against every z-slice of a z-invariant 3D run; `test_gpu_poisson2d.cpp` covers
the degenerate-axis solver switches.

**Untested.**

| thing | why it matters | evidence |
|---|---|---|
| `gpu/runner_incomp.hpp`, `runner_comp.hpp`, `runner_channel.hpp` | the three time loops; masterplan step 9 was the "MOVES (risk)" step | only public headers no test includes |
| `--far tunnel` (walls) end-to-end | the **incompressible default** | `gpu_open_incomp` = fringe, `gpu_tunnel_open` = `--open`; `val_cyl_re60` quarantined |
| compressible flow end-to-end | the default physics per user memory | `val_comp_cyl_re60` always rc 77 (`test_val_comp_cyl.cpp:34`) |
| exit codes 4, 5, 6, 7, 8 | `tests/test_errors.cpp:1-3` claims the whole contract | file stops at code 3 (`:135-144`) |
| `--config FILE.toml` | an entire alternative input surface, advertised in the authoritative help (`src/dns/cli.cpp:407-411`) | no test anywhere |
| `--cut-flux` + compressible rejection | a **CLAUDE.md invariant** ("comp variant diverges and is validator-blocked") | rule at `src/dns/config.cpp:491`; no row in `test_cli.cpp` or `test_errors.cpp` |
| `--penalize` scope rules | `src/dns/config.cpp:498-503` (three constraints) | no CLI test row |
| `PhysicalRaster` mask resampling | `README.md:611` and user memory both say the mask must be resampled too | `tests/test_framemap.cpp` pins field round-trip only; mask path lives in `apps/dns_main.cpp:595-627` (agent 2) |
| `vtk_writer.*` | `--emit vtk` is a documented output | no test includes `dns/vtk_writer.hpp` |
| a "no Dirichlet lateral row" regression test | masterplan §6.5 explicitly asked for one | `test_gpu_tunnel_bc.hip` pins kernel identity, not the absence of a Dirichlet row |

---

## 2. Findings, ranked

Ranking = (value of the simplification) × (confidence). Risk stated per item;
"coverage lost" stated for anything that deletes or merges a test.

### F1 — `docs/validation.md` is unexecutable and silent about the quarantine
**Value: high · Confidence: certain · Risk: none (docs only)**

`docs/validation.md` is the document `CLAUDE.md:43` and `README.md:749` both
name as the authority for un-quarantining. It:

- never mentions `DNS_VALIDATION_BASELINE` (`grep BASELINE docs/` is empty);
- states Tier A "runs in every ctest sweep… ~2.8 min GPU total" (`:20`) and
  lists six Tier-A rows (`:56-62`), four unregistered by default, one always
  skipping;
- gives a Tier-C regeneration recipe in deleted commands:
  `build/cylinder_tunnel_gpu 256 --nz 1 …` (`:140`),
  `build/comp_cylinder_tunnel_gpu` (`:141,142`), `build/sphere_tunnel_gpu`
  (`:143`), plus `:155,157`;
- omits `val_comp_mesh_sphere` from the per-test table, although
  `CMakeLists.txt:288` registers it as Tier A.

`masterplan_recompose.md:1151-1161` already contains the exact replacement
command block. **Fix: paste it in, add a quarantine subsection, add the missing
row.** Nothing here is a judgement call.

### F2 — `data/validation/anchors.csv` `config` fields name deleted binaries
**Value: high · Confidence: certain · Risk: none**

All five commented rows carry `config` strings like
`cylinder_tunnel_gpu 256 --nz 1 --re 60 --dr 8 --ft 12 --fdt 1e9` and
`sphere_tunnel_gpu 160 …`. The file's own header says "run the exact 'config'
command". `masterplan_recompose.md:1164-1167` says regeneration must set
"value + date + **the new `config` string**". Update the strings now (they are
comments; changing them cannot move a number) so regeneration is a one-step
task. Do **not** uncomment or delete rows — `masterplan_recompose.md:1191`
forbids it and `CLAUDE.md:43` makes the quarantine a decision.

### F3 — README §7 contradicts a binding CLAUDE.md invariant
**Value: high · Confidence: certain · Risk: none**

`README.md:254`: `### --far sponge (compressible default) and --open lateral bands`.

`CLAUDE.md:29`: "**Sponge is NEVER a default — anywhere, any physics, any
mode.** … The compressible default is `--far nscbc`."
`src/dns/config.cpp:321`: `c.far = comp ? FarField::Nscbc : FarField::Tunnel;`
`src/dns/cli.cpp:544-549`: sponge = "Escape hatch / explicit fallback ONLY —
never a default"; nscbc = "Compressible default."

The heading is the most quotable line in that section and says the opposite of
the invariant. Retitle to `### --far sponge (escape hatch; NOT a default)`.
Note that `tests/test_val_comp_cyl.cpp:39,76` runs the compressible anchor with
`--far sponge` — legitimate (named explicitly), but it means regenerated anchors
will pin a non-default configuration; say so in `docs/validation.md` when F1 is
fixed.

### F4 — the runners have no library-level test; the default far field has no test at all
**Value: high · Confidence: high · Risk: medium (this *adds* coverage)**

Established in §1.2. Concrete consequence: on a default `ctest --test-dir build`
today, no test drives `runner_incomp` with `--far tunnel` (closed walls), and no
test drives `runner_comp` at all — the one that would (`val_comp_cyl_re60`)
exits 77 at `tests/test_val_comp_cyl.cpp:34` before spawning the binary.

Cheapest honest fix, in order of cost:

1. Register an **unquarantined smoke test** per runner: a `--ft 0.5 --n 128` run
   of `--far tunnel` and of `--physics compressible --far nscbc` asserting only
   exit 0, finite `cd`, small `maxDiv`, ≥ N stats rows. No literature band, so
   nothing is quarantine-shaped; it is a "did the loop run and stay finite"
   test, which is what is missing. `tests/test_gpu_tunnel_open.cpp:184-208` is
   already exactly this shape and is the template.
2. Make `val_comp_cyl_re60` **not** skip wholesale: the
   `cd_comp − cd_incomp ∈ [0.005, 0.06]` M²-sign check (`:56-58`) is a *physics*
   assertion needing no compressible anchor; today it is gated behind the same
   `if (!aCd && !aInc) skip(...)` at `:34`.

Risk: added runtime in the default sweep. Both proposals are ≪ 60 s at
`--n 128 --ft 0.5`.

### F5 — 24 copies of `check()`, two incompatible pass/fail protocols
**Value: high · Confidence: certain · Risk: low**

Twenty-four test TUs each define a private `void check(bool, const char*)`
differing only in `printf` field width (`%-48s`/`%-52s`/`%-56s`/`%-58s`) or in
whether it prints on success: `test_channel_config.cpp:16`, `test_cli.cpp:15`,
`test_comp.cpp:18`, `test_comp_curv.cpp:20`, `test_fieldpng.cpp:20`,
`test_framemap.cpp:25`, `test_geometry.cpp:22`, `test_gpu_channel.cpp:25`,
`test_gpu_comp.cpp:19`, `test_gpu_comp_curv.cpp:25`, `test_gpu_comp_far.cpp:24`,
`test_gpu_comp_ghost.cpp:20`, `test_gpu_comp_ghost3d.cpp:24`,
`test_gpu_comp_ghost_focal.cpp:24`, `test_gpu_comp_les.cpp:22`,
`test_gpu_comp_mesh.cpp:27`, `test_gpu_comp_nscbc.cpp:30`,
`test_gpu_cutflux.cpp:43`, `test_gpu_derived.cpp:21`,
`test_gpu_les_smag.cpp:39`, `test_gpu_open_incomp.cpp:87`,
`test_gpu_tunnel_open.cpp:40`, `test_output.cpp`, `test_scene.cpp`,
`test_webframes.cpp:19`, `test_gpu_bodyforce.hip:34`, `test_gpu_tunnel_bc.hip`
— plus `test_errors.cpp:39` (`expect`) and `test_meshsdf.cpp:36` (a
tolerance-taking variant).

They split into **two protocols**: `int failures; … ++failures` (14 files) and
`bool pass; … pass = false` (7 files), each with its own `main` epilogue.
`tests/validation_util.hpp:24-35` already provides `check`/`finish` with the
counting protocol — but only the ten `test_val_*` files use it, because that
header also drags in CSV parsing, anchors and `sys/wait.h`.

**Fix: split the three lines out into `tests/check.hpp`** (`check`, `finish`,
`skip`), have `validation_util.hpp` and `gpu_test_util.hpp` include it, and
convert the 24 TUs. Removes ~120 lines and, more usefully, removes the
possibility of a test that counts failures and then returns 0.

Coverage lost: none — a pure rename of an assertion helper. Risk is that a
conversion silently flips a `pass`-protocol file's epilogue; do it mechanically
and diff each `main`'s return statement.

### F6 — `test_errors` re-asserts ~20 constraints `test_cli` already covers, via 49 process spawns
**Value: high · Confidence: high · Risk: medium**

`tests/test_errors.cpp` runs 49 `fork`+`exec` of the `dns` binary (`:30-37`).
Its *unique* value is the **mapping** from a failure class to an exit code, plus
code 3 (mesh input), which needs the filesystem. But `:65-133` re-drives ~20
constraints that `tests/test_cli.cpp` already asserts in-process against
`validate()`:

| constraint | `test_errors.cpp` | `test_cli.cpp` |
|---|---|---|
| `--nz 2` | `:70` | `:125` |
| `--cfl 0` | `:69` | `:132` |
| `--mach` incompressible | `:83` | `:120` |
| `--far tunnel` compressible | `:85` | `:114` |
| `--open` + fringe | `:89` | `:118` |
| `--nscbc-beta 1.0` | `:91` | `:121` |
| pure NSCBC above Re 800 | `:94` | `:123` |
| `--pot` + sponge | `:97` | `:94` |
| `--dr 3` compressible | `:102` | `:96` |
| `--body none` incompressible | `:119` | `:150` |
| `--noslip` incompressible | `:121` | `:160` |
| fence compressible | `:125` | `:166` |
| floor + fringe | `:129` | `:168` |
| unknown flag / missing value / bad number | `:57,60,61` | `:107,108,109` |

**Fix: keep one representative row per exit code** (one code-1, one code-2, the
two code-3 rows, the happy paths) and move the rest of the constraint matrix
into `test_cli`, where it costs a function call instead of a process.

Coverage lost, plainly: the assertion that *each individual* config constraint
maps to exit 2 rather than some other code. That mapping is a single shared
`bad(2, …)` in `src/dns/config.cpp` and a single `die(2, …)` in
`apps/dns_main.cpp` (agent 2's lane), so one representative genuinely covers it
— but if either grows a second code-2 path, this consolidation hides it.
**Medium risk; take F8 first if you want the safe half.**

### F7 — the exit-code contract is half-tested, and the untested half includes the cheap one
**Value: high · Confidence: certain · Risk: low (adds coverage)**

`tests/test_errors.cpp:1-3` states the ten-code contract; `:11-12` claims "every
case here is reachable without a GPU (phases 1-3a)". It tests 0, 1, 2, 3.
**Code 4 (output) is exactly the case masterplan §4 step 3c exists to create** —
`RunOutput::make` failing on an uncreatable directory, before any `hipMalloc` —
and is GPU-free and one line to test (`--out /proc/nope/x --ft 0.01`, or a
`chmod 500` tmpdir). Codes 5/6/7/8 need a device or a divergence and belong in
the GPU tier; say so in the header comment instead of claiming the whole
contract.

### F8 — CMake: 196 lines and 14 hand-written targets that one function collapses
**Value: high · Confidence: certain · Risk: low**

`CMakeLists.txt:131-345`. The two `foreach` loops (`:138-143`, `:159-171`) are
exactly right and cover 47 of the 54 registered tests in 18 lines. Everything
after `:146` is hand-written and every block repeats the same calls
(`add_executable` / `target_link_libraries` / `target_compile_definitions(DNS_BIN …)`
/ `add_dependencies(… dns)` / `add_test` /
`set_tests_properties(TIMEOUT … SKIP_RETURN_CODE 77 LABELS …)`): at `:148-153`
(errors), `:183-190` (open_incomp), `:194-201` (tunnel_open), `:216-220`
(spindown), `:222-239` (four cyl), `:241-255` (tgv3d), `:257-264` (comp_cyl),
`:266-270` (sphere), `:272-276` (mesh_sphere), `:279-291` (comp_mesh_sphere),
`:293-343` (heavy tier).

Worse, the **quarantine marker comment is copy-pasted verbatim eight times**
(`:230-232, 249-251, 285-287, 295-297, 304-306, 313-315, 326-328, 335-337`) —
24 lines of identical text guarding nine `add_test` calls.

**Fix:** one helper,

```cmake
function(dns_add_test name)   # SOURCE, LINK, NEEDS_BIN, TIMEOUT, LABELS, BASELINE, HEAVY
```

collapsing `:146-343` to roughly 60 lines, with the quarantine reason stated
**once** at the `option()` and referenced by a `BASELINE` keyword. Zero
behaviour change; verify by diffing `ctest -N` before and after.

### F9 — eight `test_val_*` executables compile and link in every default build for zero registered tests
**Value: medium · Confidence: certain · Risk: low**

`CMakeLists.txt:222-228` builds `test_val_cyl_re60/re100/re200/blockage` (and
adds a dependency on `dns`) unconditionally; `:241-247`, `:266-270`, `:272-276`,
`:279-283` do the same for `tgv3d`, `sphere_re300`, `mesh_sphere`,
`comp_mesh_sphere`. With `DNS_VALIDATION_BASELINE=OFF` and
`DNS_HEAVY_VALIDATION=OFF` — the default — **none registers a test.**

This is defensible (it stops quarantined sources bit-rotting: they still must
compile against the current library, real value given they will be
un-quarantined) but undocumented, and it is the largest slice of default-build
link time in the test tier. **Recommendation: keep the behaviour, state it** —
one comment at `:222`. Do *not* guard the `add_executable` calls with the
option: that would let the quarantined tests drift out of compilability, a far
worse failure mode than 8 link steps.

### F10 — two shell-out tests duplicate each other verbatim, and their stated justification is stale
**Value: medium · Confidence: certain · Risk: low**

`tests/test_gpu_open_incomp.cpp` and `tests/test_gpu_tunnel_open.cpp` contain
byte-identical `runApp` (`:41-48` / `:107-114`), `struct Row` + `readStats`
(`:50-77` / `:116-142`) and `cleanup` (`:79-84` / `:144-149`), plus their own
`check`. `tests/validation_util.hpp` already provides `runBinary` (`:230`),
`loadCsv` (`:81`) and `windowMean` (`:120`) — which parse the stats CSV **by
column name** rather than by the hardcoded index `col[3]`/`col[7]` these two
use, so the consolidation is also more robust.

The stale part: `test_gpu_open_incomp.cpp:2-3` justifies driving the binary with
"The kernels live in the app's anonymous namespace", and `CMakeLists.txt:182`
repeats it ("Open far-field path is app-local; drive the built binary"). Both
stopped being true at recompose steps 5 and 7 — the fringe kernels are in
`src/dns/gpu/tunnel.hip`, and `test_gpu_tunnel_open.cpp:19` proves the point by
linking `gpu/tunnel.hpp` directly for its first two checks.
`test_gpu_open_incomp` also still prints `"--open run exits 0"` (`:109`) for
what is now `--far fringe` (`:107`).

**Fix:** have both include `validation_util.hpp` for the CSV/exec helpers,
delete the six duplicated functions, correct the two comments and the check
string. Keep both tests: they cover different far fields and their end-to-end
half is legitimately binary-driving (masterplan step 10 kept these).
Coverage lost: none.

### F11 — README documents a shrinking fraction of the CLI
**Value: medium · Confidence: certain · Risk: none**

`CLAUDE.md:12` says `dns --help` is authoritative, and it very nearly is
(`src/dns/cli.cpp:400-790` is 390 lines of well-maintained help matching the
parser table). README §3 has fallen behind it. Flags in the parser table
(`src/dns/cli.cpp:139-231`) and in `--help`, absent from README:

`--bumps`, `--bump-h` (`cli.cpp:152`, help `:493-501`), `--png-down` (`:166`,
help `:682`), `--noslip` (`:231`, help `:624`), `--focal-blend` (`:160`, help
`:616`), `--tt`/`--lx`/`--lz` (`:178`, help `:743,744,767`),
`--config FILE.toml` (help `:407-411`), and the bodies `fence`, `floor`, `none`
(help `:470-481`) — `README.md:66` still lists only
`cylinder|sphere|cube|lens|FILE.stl`.

`--config` is the notable one: an entire alternative input format (TOML → argv
expansion) with no README paragraph and no test (§1.2). **Cross-reference for
agent 2:** it is handled in "phase 0.5" (`apps/dns_main.cpp:1275-1287`), i.e. it
reads a file *before* the parse phase, which sits awkwardly with `README.md:147`'s
"Phases 1-2 touch NEITHER filesystem NOR GPU".

### F12 — the `_stats.csv` schema is documented as one thing and is three
**Value: medium · Confidence: certain · Risk: none**

| runner | columns | site |
|---|---|---|
| incompressible | `t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters,resid` | `src/dns/gpu/runner_incomp.hip:171-172` |
| compressible | `t,dt,cd,cl,cs,wmax` | `src/dns/gpu/runner_comp.hip:203` |
| channel | `t,dt,ubulk,utauBot,utauTop,retau,KE` | `src/dns/gpu/runner_channel.hip:85-86` |

`src/dns/cli.cpp:667-668` and `README.md:702-703` state the incompressible set
unconditionally, under `dns flow --help`, where both physics live. Only
`docs/channel.md:60` gets its case right. Consumers that index by position
(`test_gpu_open_incomp.cpp:74`, `test_gpu_tunnel_open.cpp:139` — both read
`col[3]` as `cd`, `col[7]` as `maxDiv`) are incompressible-only today and
therefore correct, but are one copy-paste from silently reading `cs` as
`maxDiv` on a compressible run. Fixing F10 (parse by name via `loadCsv`)
removes that hazard too.

### F13 — `gfx1100` claims outlive the auto-detection
**Value: low · Confidence: certain · Risk: none**

`CMakeLists.txt:64-88` auto-detects via `amdgpu-arch` and falls back to a fat
`gfx1100;gfx1201` binary. `README.md:44` still says "`CMAKE_HIP_ARCHITECTURES`
defaults to gfx1100"; `README.md:28` and `CLAUDE.md:13` still label the GPU
backend "(gfx1100)". Per user memory this machine is gfx1201 and a wrong arch
produces a HIP `getStatFunc` assertion — so the stale claim costs a debugging
session.

### F14 — the layering diagram omits six post-recompose modules
**Value: low · Confidence: certain · Risk: none**

`CLAUDE.md:19` and `README.md:138-143` both give
`cli/config → bodies/geometry → gpu/{tunnel,bodyforce,les,cutpoisson,derived} →
gpu/runner_{incomp,comp} → output`, and CLAUDE.md adds "Preserve it". Since
2026-07-19 the tree also has `gpu/cutflux`, `gpu/comp_channel`,
`gpu/runner_channel`, `dns/scene`, `dns/curvmesh` + `dns/remap` + `dns/sgs`,
and (uncommitted) `dns/framemap`. A layering invariant that does not name half
the layer cannot be checked. One line each.

### F15 — three isentropic-vortex constructors, three `randomField` variants
**Value: low · Confidence: certain · Risk: low**

`tests/test_comp.cpp:26` `vortexState(Comp&, x0,y0,beta,uInf,vInf)`,
`tests/test_gpu_comp.cpp:25` `vortexState(Comp&, beta)`, and
`tests/curv_test_util.hpp:74` `vortexState(CurvMesh&, beta, gamma)`. The third
is the general one — on an identity `CurvMesh` it reproduces the second — so the
consolidation is "delete two, call the third". Similarly
`tests/gpu_test_util.hpp:28` `randomField(g, seed)` vs
`test_gpu_conservation.cpp:26` (adds `amp`), `test_gpu_cutflux.cpp:49` (adds
`with_w`), `test_gpu_cutcell2d.cpp:21` (`randomField2d`). One
`randomField(g, seed, amp = 1.0, withW = true)` covers all four.

Coverage lost: none, but the vortex merge changes *which* FP expression builds
the IC (centroid averaging vs direct cell centres). Analytically identical on an
identity map, not necessarily bitwise. **Do the vortex merge only if the
tolerances still hold; the `randomField` merge is unconditionally safe.**

### F16 — `*.log` is not gitignored
**Value: low · Confidence: certain · Risk: none**

`reconfig.log` (163 B) and `viewer_server.log` (554 KB) have been untracked in
the repo root since 21 Jul. `.gitignore` covers `build/`, `*.bin`, `*.png`,
`web_*/`, `png_*/` but not `*.log`. One line.

### F17 — HIP test TUs get no warning flags
**Value: low · Confidence: high · Risk: none**

`CMakeLists.txt:12` applies `-Wall -Wextra -Wpedantic` to
`COMPILE_LANGUAGE:CXX` only; `:122-123` adds `-Wall -Wextra` to `dns_gpu`'s HIP
sources. The two HIP *test* TUs (`test_gpu_tunnel_bc.hip`,
`test_gpu_bodyforce.hip`, 751 lines carrying the pre-recomposition kernel
oracles) are compiled with neither. Folding them into F8's `dns_add_test` helper
is the natural place to fix it.

---

## 3. Doc/code mismatches

Each row verified against the tree; both citations given.

| # | claim | doc citation | code citation |
|---|---|---|---|
| 1 | "`--far sponge` (compressible default)" | `README.md:254` | `src/dns/config.cpp:321`; `src/dns/cli.cpp:544-549` |
| 2 | Tier A "runs in every ctest sweep", six rows, ~2.8 min | `docs/validation.md:20,56-62` | `CMakeLists.txt:229,248,284` + `tests/test_val_comp_cyl.cpp:34` |
| 3 | Tier-C regeneration commands | `docs/validation.md:140-143,155,157` | binaries deleted; only executable is `apps/dns_main.cpp` (`CMakeLists.txt:56`) |
| 4 | anchor `config` strings name `cylinder_tunnel_gpu` / `comp_cylinder_tunnel_gpu` / `sphere_tunnel_gpu` | `data/validation/anchors.csv` rows 7-11 | as above |
| 5 | "flag `--nscbc` in `comp_cylinder_tunnel_gpu`"; `--nsbeta`; `--nsstrip`; `comp_mesh_tunnel_gpu` | `docs/nscbc.md:4,43,88,95,100,128,129,130,137` | `src/dns/cli.cpp:197-203`, `:157` |
| 6 | `_stats.csv` = `t,dt,K,cd,cl,cdf,cfl,maxDiv,pcgIters,resid` | `src/dns/cli.cpp:667-668`; `README.md:702-703` | `src/dns/gpu/runner_comp.hip:203`; `runner_channel.hip:85-86` |
| 7 | "`CMAKE_HIP_ARCHITECTURES` defaults to gfx1100"; backend "(gfx1100)" | `README.md:44,28`; `CLAUDE.md:13` | `CMakeLists.txt:64-88` |
| 8 | bodies are `cylinder\|sphere\|cube\|lens\|FILE.stl` | `README.md:66` | `src/dns/cli.cpp:470-481` adds `fence`, `floor`, `none` |
| 9 | flags absent from README entirely | `README.md:64-95` | `src/dns/cli.cpp:152,166,231,160,178,407` |
| 10 | layering "Preserve it" | `CLAUDE.md:19`; `README.md:138-143` | `CMakeLists.txt:23,34,103,112` + `src/dns/{remap,sgs,curvmesh}` |
| 11 | "eight literature-band `val_*` tests are quarantined" | `CLAUDE.md:43`; `README.md:744` | `CMakeLists.txt` guards **nine** registrations (`val_tgv3d_128`, `:316`, is the ninth); two (`val_mesh_sphere` `:338`, `val_comp_mesh_sphere` `:288`) are labelled `analytic`, not `literature` |
| 12 | quarantine dated 2026-07-20 | `README.md:744` | `CMakeLists.txt:230` et al. say 2026-07-19; `masterplan_recompose.md:3` is 2026-07-19 |
| 13 | "Phases 1-2 touch NEITHER filesystem NOR GPU" | `README.md:147`; `CLAUDE.md:15` | `apps/dns_main.cpp:1275-1287` reads `--config FILE.toml` in "phase 0.5" — **cross-ref agent 2** |
| 14 | fringe kernels "live in the app's anonymous namespace" / "Open far-field path is app-local" | `tests/test_gpu_open_incomp.cpp:2-3`; `CMakeLists.txt:182` | `src/dns/gpu/tunnel.hip`; `tests/test_gpu_tunnel_open.cpp:19` |
| 15 | "Keep the `*2d` parity tests green" | `CLAUDE.md:22` | tests correct (`tests/test_gpu_cutcell2d.cpp:130,139`), but per user memory `"w stays zero"` currently **fails at clean HEAD on gfx1201**; the known-red status lives only in the user's memory file, not in CLAUDE.md's "Known gotchas" |

---

## 4. Verified correct / leave alone

- **The quarantine mechanism is clean.** It guards *registrations* only, never
  sources and never `anchors.csv`, exactly as
  `masterplan_recompose.md:1130-1142` specified. No `val_*` source was edited to
  weaken a band. Do not propose undoing it (`CLAUDE.md:43`: decision, not
  failure). Defects are cosmetic (F8's eight-fold comment) and documentary
  (F1, mismatch row 11).
- **`val_spindown` is correctly wired as the canary.** `CMakeLists.txt:216-220`
  registers it unconditionally, outside both options, `LABELS
  "validation;analytic"`; `tests/test_val_spindown.cpp` skips only on a missing
  device (`:95`), consults no anchor, and makes hard analytic assertions
  (`:112,117,133,232,268`) including a self-check of its own Bessel series. It
  is the only validation test that cannot silently pass by skipping.
- **`tests/validation_util.hpp`** (287 lines) is good shared infrastructure:
  tolerant CSV, trapezoidal `windowMean`, hysteresis `strouhal` with an explicit
  justification for not using an FFT (`:149-150`), and a deferred-anchor
  registry that prints `[DEFER]` rather than failing (`:275-285`). Leave it;
  extend its reach (F5, F10) rather than replacing it.
- **`gpu_test_util.hpp` / `mesh_test_util.hpp` / `curv_test_util.hpp`** are
  genuine shared fixtures with no duplication between them.
- **The two HIP-TU tests keeping pre-recomposition kernels as oracles**
  (`test_gpu_tunnel_bc.hip`, `test_gpu_bodyforce.hip`) are the right design, and
  the CMake comment explaining why they are `.hip` (`CMakeLists.txt:173-174`) is
  accurate.
- **The `*2d` parity tests exist and test what CLAUDE.md says.** Do not merge
  `gpu_cutcell2d` into `gpu_cutcell`; the whole point is the same operators at
  `nz == 1` and `nz > 1`.
- **The bit-identity pins** listed in §1.2 are the tree's best asset — every
  "X off is bit-identical" claim in CLAUDE.md/README has a corresponding
  `check(... == 0.0, ...)`. Do not relax any to a tolerance.
- **`docs/channel.md`** is accurate: `:60-63` matches
  `src/dns/gpu/runner_channel.hip:85-90` column for column, and its flag names
  match `src/dns/cli.cpp:729-779`. `docs/scene.md`, `docs/viewer-format.md`: no
  stale-binary references found.
- **No dead docs.** Every file in `docs/` (excluding `docs/archive/`) describes a
  module that still exists; every spec `CLAUDE.md:5` lists is present. The rot is
  *inside* `docs/nscbc.md` and `docs/validation.md`, not whole files.
- **No stale CMake targets.** Every source named exists; every `tests/test_*`
  file is registered; no option is unused (`DNS_WITH_FFTW`, `DNS_WITH_HIP`,
  `BUILD_TESTING`, `DNS_HEAVY_VALIDATION`, `DNS_VALIDATION_BASELINE` all have
  live consumers).
- **`tests/test_cli.cpp`** (uncommitted changes included) is the best-shaped test
  in the tree: ~150 accept/reject rows driving `validate()` in-process, each with
  its rationale in a comment, including a pin that the compressible default far
  field *is* nscbc (`:146-147`) — the CLAUDE.md sponge-is-never-a-default
  invariant made executable. Extend it (F6, and the missing rules in §1.2); do
  not shrink it.

---

## 5. Risks

- **F6 is the only proposal that removes assertions.** Deleting the duplicated
  exit-2 rows loses per-constraint code attribution. Safe *today* because there
  is one code-2 emission path; unsafe the moment there are two. If in doubt, do
  F8 (pure CMake) and skip F6.
- **F5's conversion touches 24 `main()` epilogues.** A file that counts into
  `int failures` but returns `pass ? 0 : 1` (or vice versa) would silently become
  a test that always passes. Convert mechanically and diff every `return`.
- **F15's vortex merge can move numbers.** `curv_test_util.hpp:74` builds the IC
  from cell **centroids** (an 8-node average); `test_gpu_comp.cpp:25` uses cell
  centres directly. Identical analytically on an identity map, not necessarily
  bitwise; `test_gpu_comp` asserts FP32-vs-FP64 parity at a fixed tolerance.
  Verify before merging. The `randomField` half has no such hazard.
- **F4 adds GPU runtime to the default sweep** (two ~30 s smoke runs). That is
  the price of the two time loops having any coverage; worth paying, but it
  should be deliberate, and the tests must carry `TIMEOUT` and
  `SKIP_RETURN_CODE 77` like every other GPU test.
- **Fixing F1/F2/F3 changes documents other agents' proposals will cite.** Land
  the documentation corrections in one commit, before any test or CMake change,
  so the reviewer's masterplan quotes a coherent baseline.
- **`CLAUDE.md:22` says keep the `*2d` parity tests green, and one is believed
  red on this machine.** Any proposal in this TPRI validated by "the `*2d` tests
  still pass" must establish the *current* status first, on this hardware. This
  belongs in CLAUDE.md's "Known gotchas" whichever way it resolves.
- **Nothing proposed here touches a CLAUDE.md invariant.** No test pinning
  ghost-view/convection separation, free-slip laterals, the sponge default, the
  `nz == 1` restriction, NSCBC-as-RHS-overwrite, the anisotropic deep-band
  threshold, or any bit-identity claim is proposed for deletion or merging.
