# analysis_recompose_2 — internal defragmentation

Agent 2 of the Simplification TPRI. Focus: duplication census, target file layout,
dead code, refactor order. No code was modified.

Totals as measured: `apps/` = 5188 lines over 16 mains; `src/dns/` = 2565 lines
(CPU) + 2549 (gpu). **The app layer is as big as the library it drives, and
~2250 of its lines are copies.**

Similarity numbers below are `difflib.SequenceMatcher` ratios over
comment-stripped, blank-stripped code lines of the extracted function bodies.

---

## 1. Duplication census

### 1.1 Verbatim clones (sim = 1.000) — mechanical lifts

| block | copies | sites | lines each | removable |
|---|---|---|---|---|
| `parseOr` | 13 | tgv3d_main.cpp:21, hit_main.cpp:25, movie_main.cpp:106, cube_main.hip:171, cube_cut_main.cpp:27, cube_cut_gpu_main.hip:33, sphere_cut_gpu_main.hip:30, sphere_tunnel_gpu_main.hip:36, cube_tunnel_gpu_main.hip:34, mesh_tunnel_gpu_main.hip:43, cylinder_tunnel_gpu_main.hip:54, eye_tunnel_gpu_main.hip:47, comp_mesh_tunnel_gpu_main.hip:54 | 7 | 84 |
| `downloadSlab` | 9 | movie_main.cpp:27, cube_main.hip:90, cube_cut_gpu_main.hip:64, sphere_cut_gpu_main.hip:61, sphere_tunnel_gpu_main.hip:275, cube_tunnel_gpu_main.hip:143, mesh_tunnel_gpu_main.hip:215, cylinder_tunnel_gpu_main.hip:354, eye_tunnel_gpu_main.hip:317 | 8 | 64 |
| `kBodyPressure` | 5 | sphere_tunnel:126, cube_tunnel:126, mesh_tunnel:163, cylinder_tunnel:175, eye_tunnel:138 | 15 | 60 |
| `kBcTunnel` | 5 | sphere_tunnel:51, cube_tunnel:52, mesh_tunnel:66, cylinder_tunnel:75, eye_tunnel:62 | 30–34 | ~130 |
| `kOutletTally` | 5 | sphere:87, cube:87, mesh:124, cylinder:135, eye:98 | 10 | 40 |
| `kOutletCorrect` | 5 | sphere:99, cube:99, mesh:136, cylinder:147, eye:110 | 9–10 | 39 |
| `kInOutFluxRhs` | 5 | sphere:111, cube:111, mesh:148, cylinder:160, eye:123 | 11 | 44 |
| `kBodyFriction` | 4 | sphere:147, mesh:182, cylinder:196, eye:159 | 31–32 | ~95 |
| `kSmagNut` | 2 | cylinder:245, eye:208 | 27 | 27 |
| `kEddyDiffuse` | 2 | cylinder:273, eye:236 | 29 | 29 |
| `kBodySgsFriction` | 2 | cylinder:307, eye:270 | 46 | 46 |
| `kFringeCut` | 2 | cube_cut_gpu:44, sphere_cut_gpu:41 | 19 | 19 |
| `snapNode` | 3 | cube_cut_main.cpp:25, cube_cut_gpu:31, cube_tunnel:32 | 1 | 2 |
| `kInLo`/`kInHi` z-range helpers | 4 | cylinder:49-50, eye, mesh, comp apps inline | 2 | 6 |

Subtotal: **~685 lines** of byte-identical or near-identical code.

Notes on the sub-1.0 pairs — all the *same* block with a local tweak, not a
different algorithm:

- `kBcTunnel` cylinder/eye (sim 0.963) differ from sphere/cube/mesh (sim 1.000
  among themselves, 0.778 vs cylinder) only in the native-2D `nz == 1` guard
  (`cylinder_tunnel_gpu_main.hip:81`, `km` mirror row) and the `w` z-wall clause.
  One kernel with an `nz == 1` branch — already the project's stated 2D policy —
  covers all five.
- `kOutletTally`/`kOutletCorrect`/`kInOutFluxRhs` differ only by `kInLo/kInHi`
  vs a hardcoded `k >= 1 && k <= nz-2`. Same fix.
- `kBodyFriction` sphere/mesh (32 lines) tally x/y/z links; cylinder/eye
  (29 lines) drop the z links because free-slip z ghosts carry no viscous flux.
  One kernel + a `bool zLinks` parameter.

### 1.2 Structural clones — same algorithm, retyped

**Supersampled sdf -> CutGeometry.** Four independent implementations of
"classify by band, otherwise supersample the sdf":

- `src/dns/meshsdf.cpp:271-330` `makeMeshCut` — the good one (`frac` lambda at
  :277, `classify` at :288, plus an 8^3 Lipschitz block prefilter at :295 that
  the others lack).
- `apps/sphere_tunnel_gpu_main.hip:228-273` `makeGolfCut` — identical `frac`
  (:238) + `classify` (:255) with 4^3/4^2 samples, no prefilter.
- `apps/cylinder_tunnel_gpu_main.hip:402-450` `makeGolfCut` — the 2D restriction
  of the same, 8x8/8 samples, z-replicated (:441-448).
- `apps/eye_tunnel_gpu_main.hip:339-410` `makeLensCut` — exact chords for ax/ay
  plus a 64^2 supersampled lambda (:392-399); the sampling half is the same idea.

~160 lines that collapse into one free function.

**Golf-dimple sdf.** `GolfBall` (sphere_tunnel:184-220, 3D, golden-angle
placement) and `GolfDisk` (cylinder_tunnel:368-400, 2D, irrational phase) are
the same construction `max(base, -dimples)` in different dimension. ~70 lines
-> one `dimpled()` over an arbitrary base sdf.

**Frame writers.** `movie_main.cpp:36-104` (`SliceWriter` + `writeFrame`) and
`cube_main.hip:99-169` are sim 0.85/0.79 — same vorticity+speed slab-to-float
conversion and file append. The tunnel apps then re-derive the *same* code
inline: `cylinder_tunnel_gpu_main.hip:679-724`,
`comp_cylinder_tunnel_gpu_main.hip:236-292` (identical in structure,
primitive-derived instead of MAC), plus `eye_tunnel`, `mesh_tunnel`,
`sphere_tunnel`, `cube_tunnel`. Roughly **6 copies x ~50 lines = 300 lines**.

**3D plane/volume dump.** `comp_mesh_tunnel_gpu_main.hip:358-378`
(`writeLamPlanes`), `:399-430` (`vortSpeed`/`downloadPrims`), `:502-511`
(`writeVolume`), `:518-543` (`writeSlices`), `:441-486` (the 9-plane `addWeb`
+ `planes.json`) is ~190 lines of output machinery only this app has, but which
`mesh_tunnel_gpu_main.hip` and `sphere_tunnel_gpu_main.hip` partially
reimplement with their own `_lam.bin`/`_vort.bin`/`_speed.bin` loops.

**Run JSON + stats CSV.** Every app opens `prefix + ".json"` twice (once before
the loop with `frames:0`, once after with `frames`/`msPerStep`) and a
`prefix + "_stats.csv"` with a hand-written header. 11 apps x ~15 lines ~= 165
lines, all trivially schema-driven. `comp_mesh_tunnel_gpu_main.hip:325-346`
already factored it into a local `writeJson` lambda — right shape, wrong scope.

**Blowup forensics.** `cylinder_tunnel_gpu_main.hip:735-758` (24 lines:
downloads the field, argmaxes each component, prints `BLOWUP`) is duplicated in
`sphere_tunnel_gpu_main.hip` and `mesh_tunnel_gpu_main.hip`; the comp apps have
the shorter `dt collapsed at t=... -- blowup` variant
(`comp_cylinder_tunnel_gpu_main.hip:305`, `comp_mesh_tunnel_gpu_main.hip:551`).
~60 lines.

**RK3 + projection driver.** The incompressible tunnel `stage` lambda —
`fusedRhsStage` -> optional LES -> `maskSolidFaces` -> fringe *or* (bc + flux
correct) -> `divergenceCut` -> `kInOutFluxRhs` -> `poisson->solve` ->
`subtractGradientCut` -> bc refresh — appears 5x
(`cylinder_tunnel_gpu_main.hip:603-635` and equivalents in sphere/cube/mesh/eye).
~35 lines x 5. The surrounding `while (t < tEnd)` loop (dt from
`maxCourantPerDt`, three `stage` calls, force tally every 20 steps, frame every
`frameDt`, stats every 20, log every 200) is another ~90 lines x 5.

Measured main-body overlap (`awk '/^int main/,0'`, `diff -y --suppress-common-lines`):

```
main() sizes: cube 291, sphere 320, eye 386, cylinder 406, mesh 445  (1848 total)
sphere vs mesh    : 189 differing lines  (58% shared)
sphere vs cube    : 130 differing lines  (59% shared)
cylinder vs eye   :  81 differing lines  (80% shared)
sphere vs cylinder: 167 differing lines  (54% shared)
```

Of the 1848 lines of incompressible-tunnel `main`, **~1100 are shared machinery**;
only ~700 are genuinely per-case (geometry, IC, argument names).

### 1.3 Bottom line

| category | duplicated lines |
|---|---|
| verbatim kernels/helpers (1.1) | ~685 |
| supersamplers + dimple sdfs | ~230 |
| frame/volume/JSON/CSV writers | ~650 |
| stage lambda + time loop | ~625 |
| blowup forensics | ~60 |
| **total** | **~2250 of 5188 app lines (43%)** |

Target: `apps/` becomes a single `main.cpp` of ~250 lines (argument table ->
`Case`, dispatch, call `run`); everything else moves into named modules.

### 1.4 Proposed single homes

```cpp
// src/dns/cli.hpp                          — replaces 13 parseOr + 3 arg/argS/argB
namespace dns::cli {
struct Args {                                   // argv wrapper, order-free
    Args(int argc, char** argv);
    double num(std::string_view flag, double dflt) const;
    int    intg(std::string_view flag, int dflt) const;
    std::string str(std::string_view flag, std::string_view dflt) const;
    bool   flag(std::string_view name) const;
    std::optional<double> positional(int i) const;
    std::vector<std::string> unknown() const;   // typo guard; today typos are silent
};
}

// src/dns/geometry.hpp                      — replaces 4 supersamplers + 2 golf sdfs
namespace dns {
using Sdf = std::function<double(double, double, double)>;   // >0 fluid
struct SampleOpts { int nCell = 4, nFace = 4; double bandLo = 0, bandHi = 0; };
// Cut geometry by band-classified supersampling of an sdf; Lipschitz block
// prefilter (meshsdf.cpp:295) applies to every source. nz == 1 replicates in z.
CutGeometry sampleCut(const Grid& g, const Sdf& sdf, SampleOpts o = {});
// Solid = base minus spherical dimples: sdf = max(base, -dimples).
Sdf dimpled(const Sdf& base, std::span<const std::array<double,3>> centres, double rd);
Sdf sphereSdf(const Sphere&); Sdf cylinderZSdf(const CylinderZ&);
Sdf rotatedCubeSdf(const RotatedCube&);
Sdf lensSdf(double cx, double cy, double w, double hh);
}

// src/dns/gpu/tunnel_bc.hpp                 — replaces kBcTunnel/kOutlet*/kInOutFluxRhs x5
namespace dns::gpu {
struct TunnelBc {                               // channel: Dirichlet in, convective out,
    struct Config { double uInf = 1;            // free-slip lateral, exact mass balance
                    bool zWalls = true; };      // false => native-2D / periodic z
    static std::expected<TunnelBc, hipError_t> make(const GpuGrid&, Config);
    void apply(const DeviceField& in, DeviceField& out, double cdt) const;
    void correctOutletFlux(DeviceField& out, DeviceScalar& scratch, ReduceScratch&) const;
    void addFluxRhs(const DeviceField& u, DeviceScalar& rhs, Real scale) const;
};
struct FringeBc {                               // --open: cosine sponge to freestream
    struct Config { int ns; double sigMax = 0.5, uInf = 1; bool zBand = false; };
    static std::expected<FringeBc, hipError_t> make(const GpuGrid&, Config);
    void apply(DeviceField& out) const;         // absorbs kFringe x3 + kFringeCut x2
};
}

// src/dns/gpu/bodyforce.hpp                 — replaces kBodyPressure/kBodyFriction/
namespace dns::gpu {                          //          kBodySgsFriction x(5,4,2)
struct BodyBox { int i0,i1,j0,j1,k0,k1; };
BodyBox bodyBox(const GpuGrid&, double cx, double cy, double cz, double r, int margin = 2);
struct Force { double fx, fy, fz; };
// Aperture-pressure surface integral + masked-link skin friction (+ optional SGS).
Force bodyForce(const GpuGrid&, const DeviceCutGeometry&, const DeviceScalar& phi,
                const DeviceField& ghostView, Real nu, const DeviceScalar* nut,
                BodyBox, DeviceScalar& fx, DeviceScalar& fy, ReduceScratch&);
}

// src/dns/gpu/les.hpp                       — replaces kSmagNut/kEddyDiffuse x2 (app-local)
namespace dns::gpu {
class SmagorinskyIncomp {                       // 2D today; 3D-general is a follow-up
public: static std::expected<SmagorinskyIncomp, hipError_t> make(const GpuGrid&, double cs);
        void addEddyDiffusion(const DeviceField& ghostView, const DeviceCutGeometry&,
                              Real scale, DeviceField& out);
        const DeviceScalar& nut() const;
};
}

// src/dns/output.hpp                        — replaces every writeFrame/writeVolume/
namespace dns {                               //          writeJson/stats CSV/BLOWUP
struct RunMeta { int nx, ny, nz; double frameDt, D, reD; std::string extraJson; };
class RunOutput {                               // owns .json, _stats.csv, _mask.bin,
public:                                         // _vort/_speed/_curl .bin, PNGs, WebFrames
    RunOutput(std::string prefix, RunMeta, const Scalar& lambda, const Grid&);
    void addSeries(std::string_view name, std::span<const std::string> cols);
    void row(std::string_view series, std::span<const double>);
    void plane2d(double t, std::span<const float> vort, std::span<const float> speed);
    void volume3d(double t, std::span<const float> curl, std::span<const float> speed);
    void planes9(double t, const VolumeSampler&);   // comp_mesh's 9-plane web set
    ~RunOutput();                                   // rewrites .json with frames/msPerStep
};
void reportBlowup(const Grid&, const VectorField& host, const Scalar& lambda, double t);
}

// src/dns/gpu/derived.hpp                   — replaces vortSpeed/downloadPrims/
namespace dns::gpu {                          //          the 6 inline slab-to-float loops
void vorticityMagnitude(const GpuGrid&, const DeviceField&, DeviceScalar& out);
void speed(const GpuGrid&, const DeviceField&, DeviceScalar& out);
void vorticityZ2d(const GpuGrid&, const DeviceField&, DeviceScalar& out);
// compressible overloads taking DeviceCompState
}
```

---

## 2. Proposed final layout

Axes as stated: 2D/3D (no split — `nz == 1` stays the exact restriction, per
CLAUDE.md), compressible/incompressible, FFT/CG solver, preconditioner,
geometry source, boundary treatment, output writer.

### src/dns/ (CPU reference + host-side infrastructure)

| file | fate |
|---|---|
| `grid.hpp`, `field.hpp` | keep, unchanged |
| `units.hpp` | **merge** into `case.hpp` (63 lines, 2 apps + `hit_mini`) |
| `operators.{hpp,cpp}` | keep (CPU reference; 14 tests depend) |
| `poisson.hpp` + `poisson_fft.cpp` + `poisson_jacobi.cpp` | **merge** to `poisson.{hpp,cpp}`; jacobi is the no-FFTW fallback, keep behind the existing `DNS_WITH_FFTW` guard but in one TU |
| `projection.{hpp,cpp}` | **merge** into `timestepper.cpp` (19 + 22 lines, one function) |
| `timestepper.{hpp,cpp}` | keep |
| `cutcell.{hpp,cpp}` | **split**: keeps `CutGeometry`/`sanitizeCut`/`closeBoxBoundary`/operators; the exact shape constructors (`RotatedCube`/`Sphere`/`CylinderZ`) move to `geometry.{hpp,cpp}` |
| **new** `geometry.{hpp,cpp}` | all geometry sources: analytic sdfs, exact cut constructors, `sampleCut` (from `meshsdf.cpp:271`), `dimpled`, lens |
| `trimesh.{hpp,cpp}`, `meshsdf.{hpp,cpp}` | keep as the mesh backend; `makeMeshCut` becomes `sampleCut(g, MeshSdf)` in `geometry.cpp` |
| `comp.{hpp,cpp}` | keep (CPU compressible reference) |
| `diagnostics.*`, `forcing.hpp`, `hit_ic.*`, `taylor_green.hpp` | keep |
| `fieldpng.*`, `webframes.*`, `vtk_writer.*` | **merge** under `output.{hpp,cpp}` (the `RunOutput` facade); keep `fieldpng`/`webframes` as its private backends |
| **new** `cli.hpp` | argument parsing |
| **new** `case.hpp` | the `Case` struct: grid/units/Re/Mach/geometry/BC-policy/output selection, populated from `cli::Args` |
| **new** `runner_incomp.{hpp,hip}` | RK3 + projection stage and time loop, one copy |
| **new** `runner_comp.{hpp,hip}` | SSP-RK3 comp loop, one copy |

### src/dns/gpu/

| file | fate |
|---|---|
| `device_buffer.hpp`, `device_util.*`, `index.hpp` | keep |
| `kernels.*`, `reductions.*`, `projection.*` | **merge** `projection.hip` (18 lines) into `kernels.hip`; keep the rest |
| `poisson.{hpp,hip}` | keep — the FFT/DCT solver axis |
| `cutcell.{hpp,hip}` | **split**: `cutcell` (apertured operators) and `cutpoisson` (the PCG + its FFT/DCT preconditioner — the CG-solver axis; 486 lines currently mixing both) |
| `ghostfill.{hpp,hip}` | keep (incompressible sharp wall) |
| `timestepper.{hpp,hip}` | keep (periodic TGV/HIT path) |
| `comp.{hpp,hip}`, `comp_flux.hpp` | keep |
| `comp_ghost.{hpp,hip}` | keep (compressible sharp wall) |
| `comp_far.{hpp,hip}`, `comp_nscbc.{hpp,hip}` | keep — the compressible boundary-policy modules |
| `comp_bc.{hpp,hip}` | **delete** — see §3 |
| **new** `tunnel_bc.{hpp,hip}` | `TunnelBc` + `FringeBc` — the incompressible boundary-policy module, mirroring `comp_far`/`comp_nscbc` |
| **new** `bodyforce.{hpp,hip}` | pressure + friction + SGS-friction tallies |
| **new** `les.{hpp,hip}` | `kSmagNut`/`kEddyDiffuse` promoted out of the two apps |
| **new** `derived.{hpp,hip}` | vorticity/speed/curl-magnitude on device (kills the host slab loops and the ~1 GB PCIe pulls at `comp_mesh_tunnel_gpu_main.hip:410-430`) |

### apps/

`apps/dns_main.cpp` only. Presets to preserve (each pinned by a test or anchor):
`tgv`, `tgv3d`, `hit`, `bench`, `movie`, `cube`, `cube_cut`, `cube_cut_gpu`,
`sphere_cut_gpu`, `cube_tunnel`, `sphere_tunnel`, `cylinder_tunnel`,
`eye_tunnel`, `mesh_tunnel`, `comp_cylinder_tunnel`, `comp_mesh_tunnel`.

**Preserve the invocation strings.** 9 tests bake `$<TARGET_FILE:...>` paths
(`CMakeLists.txt:161,175,199,222,231` and the `val_*` block). Either keep thin
`add_executable` shims that exec the single binary with a preset name, or update
every `target_compile_definitions` in the same commit. The shim route keeps
ctest green with zero test edits and is recommended.

---

## 3. Dead code / superseded / redundant

**Delete outright:**

- `src/dns/gpu/comp_bc.{hpp,hip}` (47 + 122 = 169 lines). Referenced only by
  `tests/test_gpu_comp_bc.cpp` — zero app, zero library use (grep confirms).
  CLAUDE.md: "passes unit tests but REGRESSES real flows (cd sigma 10x) ... Do
  not cite BC-run numbers." Both comp apps use `comp_far`; `comp_nscbc` is the
  correct-by-construction successor and already implements the documented fix
  (LODI/characteristic rows with time dynamics). **Delete the module and its
  test.** Keeping it costs a build target and invites misuse.
- `apps/cube_cut_gpu_main.hip` (296) and `apps/sphere_cut_gpu_main.hip` (280) —
  fringe-sponge-in-a-periodic-box precursors of the tunnel apps; their only
  unique content is `kFringeCut` (identical in both), superseded by the
  cylinder/mesh `--open` path. Nothing in `tests/` drives them.
- `apps/cube_main.hip` (303) — periodic staircase-mask cube, predates cut cells
  (`CubeMask`/`kMaskCube` at :31-61 exist nowhere else). Superseded by
  `cube_cut_gpu`; no test drives it.

**Superseded but load-bearing — keep with a note:**

- `poisson_jacobi.cpp` (66 lines): only reachable with `DNS_WITH_FFTW=OFF`, but
  it is the only non-FFT CPU solver and `test_poisson` covers it. Keep inside
  the merged `poisson.cpp` under the existing guard.
- `vtk_writer.{hpp,cpp}` (17 + 38): used by `tgv_main.cpp` only, no test. Keep
  as an `output.cpp` backend or drop with the `tgv` preset's VTK flag.

**Redundant abstractions:**

- `CutPoissonSolver` (CPU, `cutcell.hpp:112-140`) vs `DeviceCutPoissonSolver`
  (`gpu/cutcell.hpp:53-99`): both real, both tested (`test_cutcell`,
  `test_gpu_cutcell`) — keep; this is the documented CPU-reference/GPU-mirror
  pairing, not accidental duplication.
- `sumScalar`/`maxAbsScalar` are declared in `gpu/cutcell.hpp:43,46` but are
  generic reductions — move to `gpu/reductions.hpp` where the other six live.
  Today every tunnel app includes `cutcell.hpp` *and* `reductions.hpp` just to
  get both halves of one API.
- `gpu/projection.hip` is 18 lines with one function; `src/dns/projection.cpp`
  is 19. Two files, two TUs, one concept each. Merge into their neighbours.
- `kInLo`/`kInHi` (`cylinder_tunnel_gpu_main.hip:49-50` + 3 copies) encode the
  "z ghost ring or not" rule that `GpuGrid` should answer itself. Add
  `GpuGrid::kInterior()` returning a pair; delete four copies.
- The two-pass `.json` write (open with `frames:0`, reopen at the end) exists in
  11 apps because nobody owns the file. `RunOutput`'s destructor should.

---

## 4. Refactor order (build + ctest green at every step)

Each step is independently committable and testable. Pinning tests named per step.

**Step 0 — baseline.** Record `ctest` output and the current
`data/validation/anchors.csv` numbers. Tier A ~= 3 min GPU. Everything below
must reproduce them.

**Step 1 — delete dead weight.** Remove `gpu/comp_bc.{hpp,hip}` +
`tests/test_gpu_comp_bc.cpp` + the `gpu_comp_bc` entry at `CMakeLists.txt:150`.
Remove `apps/cube_main.hip`, `apps/cube_cut_gpu_main.hip`,
`apps/sphere_cut_gpu_main.hip` and their targets. *Pinned by:* nothing — that is
the point; verify by grep before deleting. Net -1017 lines, zero behaviour risk.

**Step 2 — `cli.hpp`.** Add the module, convert all 16 mains. Pure host-side, no
kernel touched. *Pinned by:* every `val_*` and `gpu_*` test that passes flags on
the command line (`val_cyl_re60`, `gpu_open_incomp`, `gpu_les_smag`,
`val_comp_cyl`, `val_sphere_re300`, `val_comp_mesh_sphere`) — a mis-parsed flag
fails immediately, a good canary. Note `arg()` in
`comp_cylinder_tunnel_gpu_main.hip:37` uses `atof` while `parseOr` uses
`from_chars`; unify on `from_chars` and confirm `--nsbeta -1` (negative default
sentinel) still parses.

**Step 3 — `output.hpp` / `RunOutput`.** Move `writeFrame`, the `.json`
double-write, the stats CSV, `_mask.bin`, and `reportBlowup` out of the apps.
2D path first (`cylinder`, `comp_cylinder`, `eye`), then the 3D volume/9-plane
path (`comp_mesh`, `mesh`, `sphere`). *Pinned by:* `fieldpng`, `webframes`
(writer unit tests) and the `val_*` tests which parse `_stats.csv` columns —
column order and header text must not move. Biggest line win (~650) at the
lowest physics risk.

**Step 4 — `geometry.hpp` / `sampleCut`.** Move `makeMeshCut` from
`meshsdf.cpp:271` to `geometry.cpp`, generalise to any `Sdf`, then re-express
both `makeGolfCut`, `makeLensCut`, and the analytic `CutGeometry::make`
overloads against it. *Pinned by:* `cutcell`, `cutcell_sphere`,
`gpu_cutcell_sphere`, `meshsdf` (icosphere/cube oracle), `gpu_comp_mesh`
(MeshSdf vs analytic sphere, 0.3%), `val_mesh_sphere`. Danger: the golf/lens
cuts use *different sample counts* (4^3 vs 8x8 vs 64^2) — carry them through as
`SampleOpts`, do not "unify" them, or the golf runs shift.

**Step 5 — `gpu/bodyforce.hpp`.** Extract `kBodyPressure` (verbatim, 5 copies),
then `kBodyFriction` (4 copies, add the `zLinks` switch), then
`kBodySgsFriction` (2 copies). *Pinned by:* `val_cyl_re60` (cd anchor 1.803 /
cdf 0.553 per CLAUDE.md), `val_cyl_re100`, `val_cyl_blockage`,
`val_sphere_re300` (cd 0.681 +/- 0.006), `gpu_ghostfill`. These are the anchors
that move if the link tally changes; do this step alone and diff cd to the last
digit.

**Step 6 — `gpu/tunnel_bc.hpp`.** Extract `kBcTunnel` + `kOutletTally` +
`kOutletCorrect` + `kInOutFluxRhs` into `TunnelBc`; `kFringe`/`kFringeCut` into
`FringeBc`. The 2D/3D difference becomes the `zWalls` parameter. *Pinned by:*
`gpu_open_incomp` (fringe path, maxDiv 1.5e-4; CLAUDE.md's "tunnel mode
bit-identical" is the acceptance test here), `val_cyl_*`, `gpu_cutcell2d`. Do
the three apps that are already sim-1.000 (`sphere`, `cube`, `mesh`) first, then
`cylinder`/`eye` with the `nz == 1` branch — the branch is where a regression
would hide.

**Step 7 — `gpu/les.hpp`.** Promote `kSmagNut`/`kEddyDiffuse` out of
`cylinder_tunnel` and `eye_tunnel` (sim 1.000, so this is a lift). *Pinned by:*
`gpu_les_smag` — which currently drives the *binary* precisely because the code
is app-local (`CMakeLists.txt:168-176` says so). Once the kernels live in
`dns_gpu`, that test can link the library directly; converting it is part of
this step and is the proof the move was faithful.

**Step 8 — `runner_incomp` / `runner_comp`.** Last and riskiest: the `stage`
lambda and the time loop. Only after 5-7 have shrunk each `main` to geometry +
IC + a `Case`. *Pinned by:* everything, especially `gpu_conservation`,
`gpu_energy_stability`, `val_spindown` (analytic, KE 2.7%), `val_tgv3d_64`,
`val_comp_cyl_re60`. Ordering details that carry physics: the force tally reads
the **first** stage's `phi` (`cylinder_tunnel_gpu_main.hip:768-771` comment);
the comp apps recompute `stableDt` **every 16 steps with a 0.95 margin**
(`comp_cylinder_tunnel_gpu_main.hip:298-302`) while the incompressible ones do
it every step; the boost/ramp must fire *after* `dev->step` and *before*
`applyBc`. Any of these silently changes trajectories.

**Step 9 — one binary.** Collapse `apps/` to `dns_main.cpp` + a preset table.
Keep per-preset `add_executable` shims initially so the 9 tests that bake
`$<TARGET_FILE:...>` keep working; retire the shims in a follow-up once the
tests take a preset argument.

**Ordering rationale:** steps 1-4 touch no GPU kernel and no anchored number.
Step 5 is the first that can move a validated cd. Steps 6-8 escalate. Anything
that changes a `data/validation/anchors.csv` number must stop and follow the
Tier-C regeneration procedure in `docs/validation.md` — a moved anchor during a
pure refactor is a bug, not a new measurement.
