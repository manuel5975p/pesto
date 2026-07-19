# masterplan_recompose — the recomposition of the DNS tree

Reviewer/masterplan step of the Simplification TPRI, 2026-07-19. Supersedes
`analysis_recompose_{1,2,3}.md`. **This document is self-contained**: the
implementer does not need to read the analyses.

Every source claim below was re-verified against the tree on 2026-07-19.

---

## 0. What this is and the one binding constraint

`apps/` is 6149 lines over 16 executables; `src/dns/` + `src/dns/gpu/` is 7073.
The app layer is as large as the library it drives, and ~43% of it is copy-paste.
Twenty-one `__global__` kernels carrying validated physics live in anonymous
namespaces in `apps/*.hip`. Two ctest targets exist only to shell out to a built
binary because the physics has no library home (`CMakeLists.txt:159-177`).

**Goal:** one executable `dns`, one clean library, split along the axes
2D/3D (which is *not* a code split — see §6), compressible/incompressible,
FFT solver, CG solver, preconditioners, geometry sources, boundary treatments,
output writers.

**BINDING USER DECISION — numeric anchors are set aside.**
The recomposition is **not** gated on bit-identity or on reproducing validated
numbers. Where two variants of a kernel exist, pick the **physically correct**
one and state plainly which runs move. Fix the FP32 atomic tally. Quarantine
(never delete) the anchor machinery. See §7 for regeneration.

This does **not** license touching §6, the keep-list: those items concern
stability and correctness, not numbers.

---

## 1. Target architecture

### 1.1 `src/dns/` — CPU reference, host infrastructure, pure logic

| file | what lives there |
|---|---|
| `grid.hpp` | `Grid`: MAC staggering, index arithmetic, periodic wrap. Unchanged. |
| `field.hpp` | `Scalar` / `VectorField` aliases + `zeros()`. Unchanged. |
| `units.hpp` | SI↔box unit conversion for the `hit` case. Unchanged. |
| `sdf.hpp` | **new.** `using Sdf = std::function<double(double,double,double)>` — the single project-wide signed-distance type (today declared twice, incompatibly, in `gpu/ghostfill.hpp:24` and `gpu/comp_ghost.hpp:20`). |
| `bodies.hpp` / `bodies.cpp` | **new.** Analytic solid shapes + their `Sdf`: `Sphere`, `CylinderZ`, `RotatedCube` (moved out of `cutcell.hpp`), `Lens` (from `eye_tunnel:329`), `dimpled()` (unifies `GolfBall`/`GolfDisk`). Body placement offsets as named per-body constants. |
| `geometry.hpp` / `geometry.cpp` | **new.** Geometry sources → `CutGeometry`: exact constructors (dispatching to `cutcell.cpp`), and `sampleCut()` — the one band-classified supersampler with the Lipschitz block prefilter, generalised from `meshsdf.cpp:271-330`. Absorbs `makeMeshCut`, both `makeGolfCut`, `makeLensCut`. |
| `cutcell.hpp` / `cutcell.cpp` | Cut-cell **discretisation only**: `CutGeometry` struct, `sanitizeCut`, `closeBoxBoundary`, apertured div/grad/mask, CPU `CutPoissonSolver`. Shapes move out to `bodies.hpp`. |
| `trimesh.hpp` / `trimesh.cpp` | STL + glTF/GLB parsing (vendored `cgltf`). Mesh validation hardened (§5). |
| `meshsdf.hpp` / `meshsdf.cpp` | `MeshSdf`: BVH closest-triangle, pseudonormal sign, winding-number fallback. `makeMeshCut` moves to `geometry.cpp`. |
| `operators.hpp` / `operators.cpp` | CPU reference operators (skew advection, diffusion, divergence). 14 tests depend. Unchanged. |
| `poisson.hpp` / `poisson.cpp` | **merged** (`poisson_fft.cpp` + `poisson_jacobi.cpp` → one TU). `PoissonSolver` interface + `makeFftSolver`; `makeJacobiSolver` demoted to `poisson_internal.hpp`. `solve()` returns `std::expected<SolveReport, PoissonError>`. |
| `timestepper.hpp` / `timestepper.cpp` | CPU RK3 + projection. Absorbs the 19-line `projection.cpp`. |
| `comp.hpp` / `comp.cpp` | CPU compressible reference (Kennedy-Gruber, SSP-RK3). Unchanged physics. |
| `diagnostics.hpp` / `diagnostics.cpp` | KE, dissipation, spectra. 2D shell-count bug fixed (§5). |
| `forcing.hpp`, `hit_ic.hpp/.cpp`, `taylor_green.hpp` | Forcing, HIT initial condition, TGV analytic. Unchanged. |
| `output.hpp` / `output.cpp` | **new.** `RunOutput` — the single sink for `.json`, `_stats.csv`, `_mask.bin`, `_vort/_speed/_curl/_lam.bin`, PNG, WebFrames, VTK, volume, 9-plane sets, and `reportBlowup`. |
| `fieldpng.hpp/.cpp`, `webframes.hpp/.cpp`, `vtk_writer.hpp/.cpp` | Private backends of `RunOutput`. Kept as files; no app includes them directly. |
| `config.hpp` / `config.cpp` | **new.** `RawConfig` (parsed), `Case` (validated, fully resolved), `validate()`. Pure: no I/O, no GPU. |
| `cli.hpp` / `cli.cpp` | **new.** One table-driven parser + `--help` rendering. Replaces 13 `parseOr` copies and the `arg/argS/argB` trio. |

### 1.2 `src/dns/gpu/` — HIP backend

| file | what lives there |
|---|---|
| `index.hpp`, `device_buffer.hpp`, `device_util.hpp/.hip` | `GpuGrid`, `DeviceBuffer`, launch helpers, `DNS_HIP_TRY`. `GpuGrid` gains `kInterior()` (kills 4 copies of `kInLo/kInHi`). |
| `reductions.hpp/.hip` | All reductions. Gains `sumScalar` / `maxAbsScalar` (moved out of `gpu/cutcell.hpp:43,46`). One shared scratch-regrowth helper (kills 3 duplicated UB sites). |
| `kernels.hpp/.hip` | Core incompressible kernels. Absorbs the 18-line `projection.hip`. |
| `poisson.hpp/.hip` | **FFT-solver axis + preconditioners.** Periodic FFT (hipFFT) and NeumannBox DCT-II via Makhoul. |
| `cutcell.hpp/.hip` | **Cut-cell operators axis.** Apertured div/grad/mask, `DeviceCutGeometry`. |
| `cutpoisson.hpp/.hip` | **new (split from `cutcell.hip`). CG-solver axis.** `DeviceCutPoissonSolver` (PCG on the fluid subspace) + `projectCut`, using `poisson.hpp` as preconditioner. |
| `ghostfill.hpp/.hip` | Incompressible sharp viscous wall (mirror ghosts). See §6. |
| `tunnel.hpp/.hip` | **new. Incompressible boundary-treatment axis.** `DeviceTunnelBc` (Dirichlet inflow, convective outlet, exact mass balance, free-slip lateral walls) + `DeviceFringe` (cosine sponge). Absorbs 5×`kBcTunnel`, 5×`kOutletTally`, 5×`kOutletCorrect`, 5×`kInOutFluxRhs`, 2×`kFringe`, 2×`kFringeCut`. |
| `bodyforce.hpp/.hip` | **new.** `DeviceBodyForce`: aperture pressure integral + closed-link skin friction + SGS friction. Absorbs 5×`kBodyPressure`, 4×`kBodyFriction`, 2×`kBodySgsFriction`. |
| `les.hpp/.hip` | **new.** `DeviceSmagorinsky` (incompressible, 2D). Absorbs `kSmagNut`/`kEddyDiffuse` out of two apps, kills the CMake binary-driving hack. |
| `derived.hpp/.hip` | **new.** Device-side vorticity magnitude / speed / curl / λ2 → kills 9 host slab loops and the ~1 GB PCIe pulls at `comp_mesh_tunnel_gpu_main.hip:410-430`. |
| `timestepper.hpp/.hip` | Periodic GPU RK3 (TGV/HIT/bench path). `grid()`/`reduceScratch()` accessors removed. |
| `comp.hpp/.hip`, `internal/comp_flux.hpp` | Compressible core (3D; `nz==1` exact). `comp_flux.hpp` moves under `internal/` so "HIP TUs only" is mechanical, not advisory. |
| `comp_ghost.hpp/.hip` | Compressible sharp ghost wall + momentum-flux force tally. **FP64 tally** after step 6. |
| `comp_far.hpp/.hip` | Compressible sponge far field + free-slip walls + `boostX` quiet start. |
| `comp_nscbc.hpp/.hip` | Six-face NSCBC/LODI characteristic boundaries. |
| `comp_bc.hpp/.hip` | **DELETED** (step 1). |
| `runner_incomp.hpp/.hip` | **new.** The one incompressible RK3+projection stage and time loop. |
| `runner_comp.hpp/.hip` | **new.** The one compressible SSP-RK3 time loop. |

### 1.3 `apps/`

`apps/dns_main.cpp` — **only file.** ~250 lines: parse → validate → acquire →
dispatch to `run_flow` / `run_tgv` / `run_hit` / `run_bench`. Zero physics,
zero `__global__`.

All 16 existing `apps/*` files are deleted by the end of step 10.

---

## 2. The final CLI

Binary: **`dns`**. Four subcommands. Long flags only. **Unknown flags are a
hard error** (today every unrecognised token silently becomes the grid size —
`cylinder_tunnel_gpu --oepn` runs at n=512 without a word).

### 2.1 `dns --help`

```
dns — incompressible & compressible DNS/LES on a MAC grid (CPU + HIP)

usage: dns <subcommand> [options]
       dns <subcommand> --help

subcommands:
  flow    a body in a uniform free stream: drag/lift, wake, slices
  tgv     Taylor-Green vortex — 2D analytic check or 3D Re1600 benchmark
  hit     forced homogeneous isotropic turbulence (SI-configured)
  bench   step-time and throughput benchmark

common options (every subcommand):
  --backend cpu|gpu|auto   compute backend                        [auto]
  --out PREFIX             output file/dir prefix           [<subcommand>]
  --seed N                 RNG seed for any stochastic input          [1]
  --quiet                  suppress the periodic progress line
  --dry-run                validate, print the resolved case, exit 0
  --help                   this text, or the subcommand's

conventions:
  * nz == 1 IS the 2D restriction — there is no separate 2D code path.
    Every z stencil cancels identically and w stays exactly 0.
  * --nz always means INTERIOR cells in z, in every subcommand. Ghost
    rings, where a boundary treatment needs them, are added internally.
  * Box units: uInf = 1, rho_inf = 1. Re and Mach are the physical inputs.
  * Exit codes: 0 ok, 1 CLI, 2 config, 3 input, 4 output, 5 no GPU,
    6 resource, 7 numerical failure, 8 solver non-convergence, 77 skipped.
```

### 2.2 `dns flow --help`

```
dns flow — a body in a uniform free stream

usage: dns flow --body BODY [grid] [physics] [far field] [output]

GRID
  --n N              cells along x (streamwise)                    [512]
  --ny N             cells along y                                  [=n]
  --nz N             cells along z; 1 = native 2D                    [1]
  --dr RATIO         domain height / body diameter D                 [8]

  Grid convention differs by --physics and is stated here once:
    incompressible: L = 2*pi, h = L/n, D = L/--dr  (box scales with n)
    compressible:   D = pi/4, h = D/64 * (512/--ny), box 8D x 8D fixed
                    (resolution scales with --ny; the box does not)
  These are two genuinely different, separately validated conventions.
  Merging them relocates the body and rescales Re — a real past bug
  (measured cd 3.19). They are NOT unified by this recomposition.

BODY (exactly one; required)
  --body cylinder        z-aligned circle                    (2D or quasi-2D)
  --body sphere          ball                                          (3D)
  --body cube            45-deg rotated square prism              (2D or 3D)
  --body lens            two-disk intersection, sharp tips              (2D)
  --body FILE.stl|.gltf|.glb    triangle mesh, sdf via BVH       (2D or 3D)

  --aspect A         lens width / height, A > 1                    [3.5]
  --dimples N        golf dimples, 0 = smooth                        [0]
                     (cylinder -> 2D disks, sphere -> 3D caps)
  --scale S          mesh scale; 0 = fit max extent to D             [0]
  --robust           mesh sign by winding number (leaky meshes)
  --pitch DEG        DEPRECATED: rotates RAW MODEL AXES, not aircraft
  --yaw DEG          angles. Bake attitude with build/orient.py and
                     leave these at 0.

PHYSICS
  --physics incompressible|compressible          [incompressible]
  --re RE            Reynolds number on D                           [60]
  --mach M           Mach number (compressible only)               [0.2]
  --les CS           Smagorinsky constant; 0 = DNS                   [0]
                     incompressible: requires --nz 1 (2D-only path)
                     compressible:   shared 3D core, no restriction
  --cfl C            Courant target             [0.8 incomp / 0.4 comp]

FAR FIELD (--far, exactly one)
  --far tunnel       Dirichlet inflow + convective outlet + free-slip
                     lateral walls. Incompressible default. Pressure =
                     DCT-preconditioned CG on the confined box.
  --far fringe       fully periodic + cosine sponge on the boundaries
                     relaxing u -> uInf. FFT pressure. (was: --open)
  --far sponge       compressible streamwise freestream sponge +
                     free-slip lateral walls. Compressible default.
  --far nscbc        Poinsot-Lele characteristic boundaries.
  --far periodic     no far field (plain periodic box).

  --open             make the LATERAL faces open too.
                       --far sponge  -> lateral rho/p-relax bands
                       --far nscbc   -> six-face characteristic
                     Rejected (not ignored) with tunnel/fringe/periodic.
  --sponge-cells N   sponge / fringe band width; 0 = 10% of n         [0]
  --fringe-axes 2|3  fringe on x,y only, or x,y,z; 0 = auto
                     (2 when --nz 1, else 3)                          [0]
  --nscbc-beta B     Yoo-Im transverse correction, in [0,1)         [0.5]
                     Measured: 0 pumps a growing global mode;
                     1.0 blows up at t=35. Values >= 1 are rejected.
  --nscbc-strip N    absorbing strip inside the NSCBC rows; 0 = pure
                     characteristic. Required above Re 800; use ~ny/64
                     (pure NSCBC blows up at Re 2000, 2048^2).         [0]

START-UP (compressible)
  --ramp T           quiet start: IC at rest, freestream raised by
                     uniform Galilean boosts over [0,T]. 0 = impulsive
                     (legacy taper IC).                             [2.0]
  --kick A           deterministic solenoidal Gaussian trip 1 D behind
                     the body. Divergence-free, hence acoustically
                     silent; the quiet start is too symmetric to shed
                     without it.                                   [0.05]

SOLVER
  --poisson auto|fft|dct|cg    pressure solve; auto follows --far  [auto]
  --tol RMS          CG per-cell RMS residual target (incomp)      [1e-4]
  --max-iters N      CG iteration cap                               [500]
  --dt-recompute N   steps between stableDt recomputes (comp)        [16]

RUN LENGTH
  --ft N             flow-throughs to integrate                      [20]

OUTPUT (--emit is repeatable; default = stats,bin)
  --emit stats       <out>_stats.csv (t,dt,K,cd,cl,cdf,cfl,maxDiv,iters)
  --emit bin         <out>_{vort,speed,mask}.bin float32 mid-plane
  --emit png         png_<out>/{vort,speed}_%05d.png  (requires --nz 1)
  --emit web         web_<out>/ live browser frames + planes.json
  --emit volume      <out>_{vol,lam}.bin full 3D field per frame
  --emit slices      three centre planes instead of the volume
  --emit vtk         <out>_%05d.vtk legacy structured points
  --emit none        diagnostics only

  --fdt DT           .bin / volume frame interval                  [0.15]
  --pdt DT           png interval;  0 = follow --fdt                  [0]
  --wdt DT           web interval;  0 = follow --fdt                  [0]
  --planes N         web planes per axis: 1 = centre, 3 = +/- n/16    [3]
```

### 2.3 `dns tgv` / `dns hit` / `dns bench --help`

```
dns tgv — Taylor-Green vortex
  --n N        cells per side                                       [64]
  --nz N       cells in z; 1 = the 2D analytic case                 [=n]
  --re RE      Reynolds number (3D benchmark uses 1600)           [1600]
  --nu NU      viscosity; overrides --re when > 0                    [0]
  --ft T       end time (TGV reports t, not flow-throughs)          [20]
  --cfl C      Courant target                                      [0.9]
  --emit stats t,K,dKdt,epsVisc CSV                                 [on]
  --emit vtk   final field as legacy VTK
  --emit error max-norm error vs the 2D analytic solution + KE ratio
               (requires --nz 1)
  --emit slices mid-plane speed/vorticity/pressure frames

dns hit — forced homogeneous isotropic turbulence
  Configured in SI (air viscosity, target u'), sized so h = 1.5 eta;
  runs in box units and reports both. Requires nz >= 4 (spectral IC).
  --n N            cells per side                                  [256]
  --urms U         target one-component rms velocity [m/s]          [60]
  --nu NU          kinematic viscosity [m^2/s]                 [1.5e-5]
  --spin TE        eddy turnovers of spin-up before statistics      [10]
  --stats TE       eddy turnovers of statistics                     [10]
  --emit stats | spectrum | snapshot | slices

dns bench — step-time and throughput benchmark
  --n N            cells per side                                  [256]
  --steps N        timed steps after 3 warm-up steps                [20]
  --physics incompressible|compressible          [incompressible]
  Reports ms/step, Mcell-updates/s, VRAM in use.
```

### 2.4 Migration table

| old invocation | new |
|---|---|
| `tgv 64 0.05 1.0 --vtk o.vtk` | `dns tgv --n 64 --nz 1 --nu 0.05 --ft 1.0 --emit error --emit vtk --out o` |
| `tgv3d 128 1600 20` | `dns tgv --n 128 --re 1600 --ft 20` |
| `bench 256 20` | `dns bench --n 256 --steps 20` |
| `hit 256 --urms 60 --spin 10 --stats 10 --snap --out h` | `dns hit --n 256 --urms 60 --spin 10 --stats 10 --emit snapshot --out h` |
| `movie tgv 256 --out m` | `dns tgv --n 256 --ft 20 --emit slices --fdt 0.1 --out m` |
| `movie hit 256 --out m` | `dns hit --n 256 --emit slices --spin 5 --stats 3 --out m` |
| `cube 256 --re 300 --ft 6` | **removed** — direct-forcing staircase IB deleted (§8) |
| `cube_cut 128 --re 300 --flat 8` | `dns flow --backend cpu --body cube --far fringe --n 128 --nz 8 --re 300` |
| `cube_cut_gpu 256 --re 300 --flat 8` | `dns flow --body cube --far fringe --n 256 --nz 8 --re 300` |
| `sphere_cut_gpu 128 --re 300` | `dns flow --body sphere --far fringe --n 128 --re 300` |
| `cube_tunnel_gpu 256 --re 300 --flat 8` | `dns flow --body cube --far tunnel --n 256 --nz 8 --re 300` |
| `sphere_tunnel_gpu 128 --re 300 --golf 300` | `dns flow --body sphere --far tunnel --n 128 --re 300 --dimples 300` |
| `cylinder_tunnel_gpu 1024 --nz 1 --re 60 --dr 8` | `dns flow --body cylinder --far tunnel --n 1024 --nz 1 --re 60 --dr 8` |
| `cylinder_tunnel_gpu 512 --nz 8 --re 60` | `dns flow --body cylinder --far tunnel --n 512 --nz 8 --re 60` |
| `cylinder_tunnel_gpu 512 --open --re 60` | `dns flow --body cylinder --far fringe --n 512 --re 60` |
| `cylinder_tunnel_gpu 512 --les 0.17 --nz 1` | `dns flow --body cylinder --far tunnel --n 512 --nz 1 --les 0.17` |
| `cylinder_tunnel_gpu 1024 --golf 40` | `dns flow --body cylinder --far tunnel --n 1024 --dimples 40` |
| `eye_tunnel_gpu 1024 --ar 3.5 --re 60` | `dns flow --body lens --aspect 3.5 --far tunnel --n 1024 --re 60` |
| `mesh_tunnel_gpu m.stl 96 --re 300 --robust --vol --voldt 0.75` | `dns flow --body m.stl --far tunnel --n 96 --re 300 --robust --emit volume --fdt 0.75` |
| `mesh_tunnel_gpu m.stl 96 --open` | `dns flow --body m.stl --far fringe --n 96` |
| `comp_cylinder_tunnel_gpu 512 --re 800 --mach 0.2` | `dns flow --physics compressible --body cylinder --far sponge --n 512 --re 800 --mach 0.2` |
| `comp_cylinder_tunnel_gpu 512 --nscbc --nsbeta 0.5` | `dns flow --physics compressible --body cylinder --far nscbc --nscbc-beta 0.5 --n 512` |
| `comp_cylinder_tunnel_gpu 512 --nscbc --open` | `... --far nscbc --open ...` |
| `comp_cylinder_tunnel_gpu 1024 --nscbc --nsstrip 12` | `... --far nscbc --nscbc-strip 12 --n 1024` |
| `comp_mesh_tunnel_gpu m.stl 96 --les 0.17 --nz 96` | `dns flow --physics compressible --body m.stl --far nscbc --open --n 96 --nz 96 --les 0.17` |
| `comp_mesh_tunnel_gpu m.stl 128 --slices` | `... --far nscbc --open --n 128 --emit slices` |

Renames: `--open`(incomp) → `--far fringe`; `--nscbc` → `--far nscbc`;
`--flat` → `--nz`; `--golf` → `--dimples`; `--ar` → `--aspect`;
`--nsbeta/--nsstrip` → `--nscbc-beta/--nscbc-strip`;
`--vol/--voldt/--snap/--slices` → `--emit …`; positional `n` → `--n`.

**Semantic change, deliberate and user-visible:** `--nz` is interior cells
everywhere. `cylinder_tunnel_gpu --nz 8` allocated `nz = 10` (8 + ghost ring);
`dns flow --nz 8` allocates the ring internally and the user still says 8.

---

## 3. Public API — exact header contents

Only new or changed headers are given. Anything not listed keeps its current
contents.

### `src/dns/sdf.hpp`
```cpp
// Signed distance to a solid in world coordinates: negative inside, and
// approximately a true distance near the wall (|grad sdf| ~ 1 there).
// Host-side, build time only — never called inside a stepping loop.
#pragma once
#include <functional>
namespace dns {
using Sdf = std::function<double(double x, double y, double z)>;
}
```

### `src/dns/bodies.hpp`
```cpp
// Analytic solid shapes in world units and their signed-distance functions.
// Placement offsets are per-body constants: each deliberately puts the body
// off-node so the shedding instability trips. They select which attractor a
// run lands on — carry them verbatim, do not normalise.
#pragma once
#include "dns/grid.hpp"
#include "dns/sdf.hpp"
#include <array>
#include <span>
#include <vector>

namespace dns {

struct RotatedCube { double cx, cy, r, z0, z1; };   // moved from cutcell.hpp
struct Sphere      { double cx, cy, cz, r; };       // moved from cutcell.hpp
struct CylinderZ   { double cx, cy, r; };           // moved from cutcell.hpp
// Two overlapping disks of half-width w and half-height hh (w > hh): a lens
// with sharp leading/trailing tips. Interior: both disks.
struct Lens        { double cx, cy, w, hh; };

Sdf sdfOf(const Sphere&);
Sdf sdfOf(const CylinderZ&);
Sdf sdfOf(const RotatedCube&);
Sdf sdfOf(const Lens&);

// base minus spherical-cap pits: sdf = max(base, -pits). Pit footprint radius
// rd and depth are in WORLD units; the caller sizes them in grid units so they
// stay resolvable (5h / 2.5h in the validated golf runs — a real ball's 0.01D
// dimples vanish on any affordable grid).
// Pre: rd > 0, depth > 0, centres non-empty.
Sdf dimpled(Sdf base, std::span<const std::array<double,3>> centres,
            double rd, double depth);

// Deterministic dimple lattices. 3D: golden-angle Fibonacci on the sphere.
// 2D: irrational phase around the circle, z-replicated.
std::vector<std::array<double,3>> fibonacciCentres(const Sphere&, int n);
std::vector<std::array<double,3>> ringCentres(const CylinderZ&, int n);

// Body centre for a tunnel/fringe run, offset off-node to trip shedding.
// physics selects the grid convention (see `dns flow --help`).
struct Placement { double cx, cy, cz, D; };
Placement placeBody(const Grid&, bool compressible);

} // namespace dns
```

### `src/dns/geometry.hpp`
```cpp
// Geometry sources: any Sdf (analytic, dimpled, or MeshSdf) -> CutGeometry.
// Exact closed-form geometry is used where one exists; otherwise the cut is
// built by band-classified supersampling with a Lipschitz block prefilter.
#pragma once
#include "dns/cutcell.hpp"
#include "dns/grid.hpp"
#include "dns/sdf.hpp"
#include <expected>
#include <string>

namespace dns {

// Sampling budget. The validated bodies use DIFFERENT budgets (mesh 8^3/8^2,
// golf sphere 4^3/4^2, golf disk 8x8/8, lens 64^2 lambda with exact chord
// apertures). They are carried through as explicit options, NOT unified — a
// changed budget shifts the golf/lens cuts.
struct SampleOpts { int nCell = 8; int nFace = 8; };

// lambda / ax / ay / az by supersampling `sdf` (>0 fluid). Cells whose
// 8-corner Lipschitz bound proves them wholly fluid or wholly solid skip the
// inner samples. nz == 1: the z pass is skipped and az is replicated.
// Pre: grid dims > 0. Post: every value in [0,1].
std::expected<CutGeometry, std::string>
sampleCut(const Grid&, const Sdf&, SampleOpts = {});

// Exact chord apertures + supersampled lambda for the lens (its tips are
// sub-cell; pure sampling loses them).
std::expected<CutGeometry, std::string>
exactLensCut(const Grid&, const Lens&, int lambdaSamples = 64);

} // namespace dns
```

### `src/dns/gpu/tunnel.hpp`
```cpp
// Incompressible far-field boundary treatments on the ghost ring.
// GPU-only header.
#pragma once
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/index.hpp"
#include "dns/gpu/reductions.hpp"
#include <expected>

namespace dns::gpu {

// Confined channel: Dirichlet inflow, convective outflow with exact global
// mass balance, FREE-SLIP (Neumann) lateral walls.
//   inlet  (i<=1)    u = uInf; v,w antimirrored so they vanish on x = hx
//   outlet (i==nx-1) convective advection out of the domain
//   y,z rings        tangential components mirrored (the wall generates NO
//                    vorticity), normal component 0 on the wall face and
//                    antimirrored in the ghost. Pressure is already Neumann
//                    there (closed apertures + DCT).
// A Dirichlet u = uInf far field clamps the velocity against the sheared
// interior flow and paints a vortex sheet along the wall. Never reintroduce it.
//
// Ghost corners are mirrored DIAGONALLY: an edge cell takes the single
// (jm,km) value, and the k==0 w row is explicitly antimirrored. At nz == 1
// km == k identically, so this reduces exactly to the y-only treatment — one
// kernel covers native 2D and full 3D with no mode switch.
class DeviceTunnelBc
{
public:
    struct Config { double uInf = 1.0; };
    static std::expected<DeviceTunnelBc, hipError_t> make(const GpuGrid&, Config);

    // cdt = uInf*dt/hx drives the outlet from `in`. cdt == 0 with in aliased
    // to out is a pure ghost refresh. Pre: same grid, same size.
    void apply(const DeviceField& in, DeviceField& out, double cdt) const;

    // Restore exact global mass balance on the open outlet plane.
    void correctOutletFlux(DeviceField&, ReduceScratch&, DeviceScalar& tmp) const;

    // Inlet/outlet faces carry prescribed flux at zero aperture, so
    // divergenceCut misses them; add the missing RHS terms. scale must match
    // the one passed to divergenceCut.
    void addFluxRhs(const DeviceField&, double scale, DeviceScalar& rhs) const;
};

// Cosine-ramped fringe relaxing u -> (uInf,0,0) on the domain boundaries of a
// fully periodic box (the FFT-Poisson far field).
//   sig = sigMax * (1 - cos(pi r)) / 2, r = max over banded axes of the
//   normalised band depth (0 at the interior edge, 1 on the boundary).
// axes == 2 leaves z plain periodic: for a z-invariant (quasi-2D) run a z band
// would sponge the whole wake. axes == 3 bands all three.
// Pre: cells >= 1; axes == 2, or axes == 3 with nz >= 2*cells + 2.
class DeviceFringe
{
public:
    struct Config { double uInf = 1.0, sigMax = 0.5; int cells = 0; int axes = 3; };
    static std::expected<DeviceFringe, hipError_t> make(const GpuGrid&, Config);
    void apply(DeviceField&) const;
};

} // namespace dns::gpu
```

### `src/dns/gpu/bodyforce.hpp`
```cpp
// Discrete surface integrals over an embedded body inside an index box.
// GPU-only header.
#pragma once
#include "dns/gpu/cutcell.hpp"
#include "dns/gpu/reductions.hpp"
#include <expected>

namespace dns::gpu {

// Index bounding box of the body plus a margin, so the tallies scan O(body)
// not O(grid).
struct BodyBox { int i0, i1, j0, j1, k0, k1; };
BodyBox bodyBox(const GpuGrid&, double cx, double cy, double cz, double r,
                int margin = 2);

class DeviceBodyForce
{
public:
    // cd, cl: total force / qRef.  cdf: the FRICTION part only, and always
    // nu-only — the SGS eddy viscosity never enters cdf (see CLAUDE.md's LES
    // stance). cs: z-component, 0 in native 2D.
    struct Coeffs { double cd, cl, cs, cdf; };

    static std::expected<DeviceBodyForce, hipError_t> make(const GpuGrid&, BodyBox);

    // Pressure: F = -sum phi_c (a_xp - a_c) A_face — the exact discrete
    //   aperture identity for the surface integral of phi.
    // Friction: F = nu sum over links from an open face to a CLOSED neighbour
    //   of (q - q_ghost) A/d — the Laplacian flux the momentum update applies,
    //   with the closed neighbour holding the ghost-fill mirror value.
    //   All three link directions are tallied. At nz == 1 the z neighbours
    //   wrap to the cell itself, whose aperture is open, so the z links
    //   contribute exactly zero: the 3D form IS the 2D form.
    // ghosted = the ghost-filled view the viscous term actually saw.
    // nut != nullptr adds the SGS wall drain to cd/cl but NOT to cdf.
    // Pre: geometry sanitized; ghosted is the ghost view, not the state.
    Coeffs measure(const DeviceCutGeometry&, const DeviceScalar& phi,
                   const DeviceField& ghosted, double nu, double qRef,
                   const DeviceScalar* nut, ReduceScratch&);
};

} // namespace dns::gpu
```

### `src/dns/gpu/les.hpp`
```cpp
// Optional incompressible Smagorinsky LES (2D only). nut = (Cs*Delta)^2 |S|
// scaled by the fluid fraction lambda; the eddy-diffusion substep adds
// b*dt*div(nut grad u) per RK stage in conservative flux form.
// NOTE gradient form div(nut grad u), NOT the symmetric stress div(2 nut S).
// Output is a FILTERED field: statistics depend on grid AND Cs by
// construction. Label such runs LES and state Cs. 2D caveat: Smagorinsky
// models a 3D forward cascade; 2D cascades energy up — qualitative only.
// GPU-only header.
#pragma once
#include "dns/gpu/cutcell.hpp"
#include <expected>

namespace dns::gpu {

class DeviceSmagorinsky
{
public:
    // Pre: cs > 0, g.nz == 1.
    static std::expected<DeviceSmagorinsky, hipError_t> make(const GpuGrid&, double cs);
    void computeNut(const DeviceField& U, const DeviceCutGeometry&);
    void addEddyDiffusion(const DeviceField& U, const DeviceCutGeometry&,
                          Real scale, DeviceField& out) const;
    const DeviceScalar& nut() const;
};

} // namespace dns::gpu
```

### `src/dns/output.hpp`
```cpp
// The single sink for every run artefact. Unselected kinds allocate nothing.
// Owns the run .json (written at construction with frames:0, REWRITTEN in the
// destructor with the final frame count and ms/step) and _stats.csv.
#pragma once
#include "dns/grid.hpp"
#include <expected>
#include <set>
#include <span>
#include <string>
#include <string_view>

namespace dns {

enum class Emit { Stats, Bin, Png, Web, Volume, Slices, Vtk };

struct RunMeta {
    int nx, ny, nz; double hx, hy, hz;
    double D, re, mach, frameDt, cs;
    bool les;                       // set by the validator, not by hand
    std::string body, farField, physics, kick;
};

class RunOutput
{
public:
    struct Config {
        std::string prefix;
        std::set<Emit> kinds;
        double fdt = 0.15, pdt = 0, wdt = 0;
        int planesPerAxis = 3;
    };
    // Creates and truncates every file and directory the run will ever write,
    // and fails here if any cannot be created — before any GPU allocation.
    static std::expected<RunOutput, std::string> make(Config, RunMeta, const Grid&);
    ~RunOutput();                                    // rewrites .json; flushes

    void declareSeries(std::string_view name, std::span<const std::string> cols);
    [[nodiscard]] std::expected<void, std::string>
        row(std::string_view series, std::span<const double>);
    [[nodiscard]] std::expected<void, std::string>
        frame(double t, const FrameViews&);          // writes whatever kinds are on
    [[nodiscard]] std::expected<void, std::string> flush();
};

// Downloads the field, argmaxes each component, prints the offending cells.
// Called on a non-finite state before exiting 7.
void reportBlowup(const Grid&, const VectorField& host, const Scalar& lambda, double t);

} // namespace dns
```

### `src/dns/config.hpp`
```cpp
// RawConfig: what the parser produced. Case: what the solver may see.
// The ONLY way to obtain a Case is validate(); its constructor is private.
#pragma once
#include "dns/bodies.hpp"
#include "dns/grid.hpp"
#include "dns/output.hpp"
#include <expected>
#include <string>
#include <variant>

namespace dns {

enum class Subcommand { Flow, Tgv, Hit, Bench };
enum class Physics    { Incompressible, Compressible };
enum class FarField   { Tunnel, Fringe, Sponge, Nscbc, Periodic };
enum class Backend    { Cpu, Gpu, Auto };
enum class PoissonKind{ Auto, Fft, Dct, Cg };

struct RawConfig { /* one field per CLI flag, all still in user units */ };

struct ConfigError { int code; std::string field, message; };

// Fully resolved: every derived quantity computed, no optionals, every
// cross-field combination checked. Downstream asserts are legitimate BECAUSE
// this ran (see §5).
class Case
{
public:
    Subcommand sub; Backend backend; Physics physics; FarField far;
    Grid grid; Placement body; Sdf sdf;
    double re, mach, nu, cfl, tol, ft, cs, qRef;
    bool open, lesLabel;
    RunOutput::Config out; RunMeta meta;
    /* ... one field per resolved quantity ... */
private:
    Case() = default;
    friend std::expected<Case, ConfigError> validate(const RawConfig&);
};

// Pure: no filesystem, no HIP, no allocation beyond the Case.
std::expected<Case, ConfigError> validate(const RawConfig&);

} // namespace dns
```

### `src/dns/cli.hpp`
```cpp
// One table-driven parser for every subcommand. Unknown flags, missing
// values and unparseable numbers are ERRORS — never a silent default.
#pragma once
#include "dns/config.hpp"
#include <expected>
#include <string>

namespace dns::cli {

struct CliError { std::string message; };   // already formatted for stderr

std::expected<RawConfig, CliError> parse(int argc, char** argv);
std::string help(Subcommand);               // exact text of §2
std::string helpTop();

} // namespace dns::cli
```

### Changed existing headers

- `poisson.hpp`: `struct SolveReport { int iters; double residual; bool converged; };`
  `virtual std::expected<SolveReport, PoissonError> solve(const Scalar&, Scalar&) = 0;`
  `makeJacobiSolver` moves to `poisson_internal.hpp`.
- `gpu/cutcell.hpp`: `sumScalar` / `maxAbsScalar` **removed** (now in
  `gpu/reductions.hpp`). `DeviceCutPoissonSolver` / `projectCut` **removed**
  (now in `gpu/cutpoisson.hpp`). `DeviceCutGeometry` gains `int nx, ny, nz;`
  so bindings can be asserted against the grid.
- `gpu/cutpoisson.hpp`: `int solve(...)` → `std::expected<SolveReport, hipError_t>`;
  `make()` validates `maxIters >= 1`, `tol > 0 && isfinite`.
- `gpu/ghostfill.hpp`, `gpu/comp_ghost.hpp`: local `Sdf` typedefs **deleted**,
  both include `dns/sdf.hpp`.
- `gpu/comp.hpp`: `stableDt` → `std::expected<double, CompError>`; rejects
  non-finite / non-positive dt and non-positive rho. Private members pimpl'd.
- `gpu/timestepper.hpp`: `grid()` and `reduceScratch()` accessors **removed**.
- `cutcell.hpp`: `RotatedCube` / `Sphere` / `CylinderZ` **moved** to `bodies.hpp`
  (`cutcell.hpp` includes it for the `CutGeometry::make` overloads).
- `gpu/comp_flux.hpp` → `gpu/internal/comp_flux.hpp`.

---

## 4. Implementation sequence

**Rules for every step:** the build is green and `ctest -LE validation` is green
at the end of the step. Steps that move numbers are marked **MOVES**; they must
record the before/after values in the step's commit message so §7 can be
executed later. Do not "fix" anything in §6.

### Step 0 — baseline capture
Record `ctest --output-on-failure` results and, for each of the eight
configurations below, a `--ft 0.2` `_stats.csv`, stored under
`data/validation/precompose/`. These are **not** a gate — they are the
before-picture for §7's regeneration write-up.
```
cylinder tunnel nz=1 | cylinder tunnel nz=8 | cylinder fringe | cylinder les
sphere tunnel        | cube tunnel          | mesh tunnel     | comp_cyl sponge
```
Verify: none. Moves numbers: no.

### Step 1 — delete dead weight
Remove `src/dns/gpu/comp_bc.{hpp,hip}`, `tests/test_gpu_comp_bc.cpp`, and their
CMake entries. Reason: CLAUDE.md records it passes unit tests but *regresses
real flows* (cd σ 10×, "do not cite BC-run numbers"); no app includes it
(grep-verified: only its own test and `CMakeLists.txt`); and its `make()`
validates nothing while `comp_far`/`comp_nscbc` both do. `comp_nscbc` is the
correct-by-construction successor.

Remove `apps/cube_main.hip` (periodic staircase direct-forcing cube — the only
user of `kMaskCube`, predates cut cells, no test drives it),
`apps/cube_cut_gpu_main.hip` and `apps/sphere_cut_gpu_main.hip` (fringe-in-a-
periodic-box precursors; their only unique content, `kFringeCut`, is identical
in both and superseded by the `--open` fringe path; no test drives them).
Their capability survives as `dns flow --far fringe`; **direct forcing does not
survive** (§8).

Verify: full `ctest` unchanged minus `gpu_comp_bc`. Moves numbers: no.

### Step 2 — `cli.hpp` + `config.hpp` (parse/validate split)
Add the table-driven parser and the validator (§5). Convert all 12 remaining
mains to use them, keeping each main's current behaviour otherwise. Unify on
`std::from_chars` with a full-consumption check — note `comp_cylinder_tunnel_gpu`
uses `atof`/`atoi` today; confirm `--nscbc-beta -1` (negative auto sentinel)
still parses. Add `--dry-run`.

Verify: `ctest` full; every `val_*` and `gpu_*` test that passes flags on the
command line is the canary (`val_cyl_re60`, `gpu_open_incomp`, `gpu_les_smag`,
`val_comp_cyl`, `val_sphere_re300`, `val_comp_mesh_sphere`). Add a new
`test_cli` unit test driving `--dry-run` over the whole migration table of §2.4.
Moves numbers: no.

### Step 3 — `output.hpp` / `RunOutput`
Move `writeFrame`, the two-pass `.json`, the stats CSV, `_mask.bin`,
`reportBlowup`, `downloadSlab` (9 copies), the 9-plane WebFrames block and the
volume/slice writers out of the apps. 2D path first (cylinder, comp_cylinder,
eye), then 3D (comp_mesh, mesh, sphere, cube_tunnel).
**Column order and header text of `_stats.csv` must not change** — the `val_*`
tests parse it by column.
Also fix here: `webframes.cpp:56` / `fieldpng.cpp:67` / `vtk_writer.cpp:33`
return `bool(f)` *before* close, so a buffered ENOSPC reports success on a
truncated file — close explicitly, then test.

Verify: `fieldpng`, `webframes`, plus every `val_*`. Moves numbers: no.

### Step 4 — `sdf.hpp`, `bodies.hpp`, `geometry.hpp`
Hoist the one `dns::Sdf`. Move `RotatedCube`/`Sphere`/`CylinderZ` out of
`cutcell.hpp` into `bodies.hpp`; add `Lens` (from `eye_tunnel:329`) and
`dimpled()` (unifying `GolfBall`/`GolfDisk`). Move `makeMeshCut`
(`meshsdf.cpp:271-330`, the good one — it has the 8³ Lipschitz prefilter the
others lack) into `geometry.cpp` as `sampleCut()`, then re-express
`makeGolfCut`×2 and `makeLensCut` against it.

**Carry the per-body sample budgets through as `SampleOpts`** — mesh 8³/8²,
golf sphere 4³/4², golf disk 8×8/8, lens 64² lambda with exact chords. Unifying
them shifts the golf and lens cuts.

Verify: `cutcell`, `cutcell_sphere`, `gpu_cutcell_sphere`, `meshsdf`,
`gpu_comp_mesh`, `val_mesh_sphere`. Moves numbers: no, if budgets are carried.

### Step 5 — `gpu/tunnel.hpp` — **MOVES**
Extract `kBcTunnel`, `kOutletTally`, `kOutletCorrect`, `kInOutFluxRhs` into
`DeviceTunnelBc`, and `kFringe`/`kFringeCut` into `DeviceFringe`.

There are **two real variants** of `kBcTunnel` in the tree, verified byte-level:

*Class A (cylinder:75, eye:62)* — `km = nz==1 ? k : …`; a ghost corner gets the
y-mirror value only (`jm != j` then `km != k` as separate branches); `v[j==0]`
mirrors across `k` not `km`; `w = 0` whenever `nz==1 || k<=1 || k==nz-1`.

*Class B (sphere:51, cube_tunnel:52, mesh:66)* — no `nz==1` guard on `km`; a
corner gets the single diagonal `(jm,km)` value; `v[j==0] = -v[i,2,km]`; an
explicit `w[k==0] = -w[i,jm,2]` antimirror row.

**Decision: adopt Class B, plus the Class A `nz==1` guard on `km`.**
Justification: for a box with free-slip walls on both y and z, an edge ghost
cell must be mirrored across *both* walls; the sequential form applies only one
and is simply wrong for `nz > 1`. With `km = nz==1 ? k : (…)` restored, at
`nz == 1` we have `km == k` identically, so `jm != j || km != k` collapses to
`jm != j`, `-v[i,2,km]` becomes `-v[i,2,k]`, and Class B's `k == nz-1` clause
fires at `k == 0` giving `w = 0`. **The unified kernel is bit-identical to
Class A at `nz == 1` and is the correct generalisation above it.** No
`CornerMode` enum, no per-body switch — that would freeze a known-wrong corner
treatment into the API.

Runs that move: **`cylinder`/`lens` with `--nz > 1` only.** Native 2D
(`--nz 1`, which is the cylinder default and every cylinder anchor's
configuration) is unchanged. Sphere/cube/mesh are unchanged.

For the outlet trio the direction is the opposite: Class A uses
`kInLo(nz)/kInHi(nz)`, Class B hardcodes `k in [1, nz-2]`, which is an **empty
range at nz == 1**. Adopt Class A's form unconditionally (`GpuGrid::kInterior()`);
it is identical for `nz > 1` and is the only correct 2D form. Adopting the
hardcoded form would silently disable outlet flux correction in every 2D run.

`DeviceFringe`: cylinder bands x,y only (a z band would sponge the z-invariant
wake); mesh bands x,y,z. Keep both via `Config::axes`, defaulted to
`nz == 1 ? 2 : 3`, and let the validator **reject** `axes == 3` when
`nz < 2*cells + 2` with a message naming `--fringe-axes` (today
`cylinder --nz 8 --open` would silently get a nonsense z band).

Verify: `gpu_open_incomp` ("tunnel mode bit-identical" is the acceptance test),
`gpu_cutcell2d`, `val_cyl_re60` (2D — must be unchanged), and a new
`test_gpu_tunnel_bc` unit test asserting the 2D/3D reduction identity above.

### Step 6 — `gpu/bodyforce.hpp` — **MOVES**
Extract `kBodyPressure` (5 identical copies — a pure lift) and `kBodyFriction`
(4 copies, 2 variants: cylinder/eye tally x/y links, sphere/mesh tally x/y/z).

**Decision: tally all three link directions unconditionally.** Justification:
at `nz == 1` the z neighbours `gidx(g,i,j,k±1)` wrap to the cell itself, whose
aperture passed the `> openTol` guard that gated the call — so the z links
contribute exactly zero. For a z-invariant cylinder at `nz > 1` the z
neighbours of a body face are fluid, so they are skipped as well. The 3D form
is therefore expected to be a **no-op** for cylinder and lens; verify this by
measurement (run cylinder `--nz 1` and `--nz 8`, compare `cdf` to full
precision) rather than assuming it. No `Links` enum.

Also extract `kBodySgsFriction` (2 identical copies). `cdf` stays ν-only.

`cube_tunnel_gpu` has **no** `kBodyFriction` at all (grep-verified) — it reports
pressure drag only. The unified path always tallies friction, so **cube's cd
gains its cdf contribution and moves upward.** This is a correctness fix; state
it in the commit and in §7.

Add `fz`/`cs` to the tally (3D bodies currently report x/y only), and extend
`Coeffs` accordingly. `cs` is 0 in native 2D.

**Fix the FP32 atomic tally (`gpu/comp_ghost.hip:80-82`) in this step.**
Today the compressible body force accumulates via `atomicAdd(&acc[i], float(...))`
while every other reduction in the tree is FP64 (`reductions.hip:25`,
`cutcell.hip:152`). Cancelling ± contributions lose precision and atomic
ordering makes the reported `cd` irreproducible run to run — non-determinism in
the primary validated quantity. Replace with a per-cell FP64 staging buffer +
the existing `sumScalar` deterministic reduction. **This moves every
compressible `cd`/`cl`** (by an amount that is, by construction, the size of the
FP32 tally error) and makes the number reproducible.

Verify: `gpu_ghostfill`, `gpu_comp_ghost`, `gpu_comp_ghost3d`, plus a new
determinism test: run the same compressible ghost tally 5× and require
bit-identical `cd`. Record before/after `cd` for cylinder Re60 and comp Re60.

### Step 7 — `gpu/les.hpp`
Promote `kSmagNut`/`kEddyDiffuse`/`kBodySgsFriction` out of `cylinder_tunnel`
and `eye_tunnel` (byte-identical copies — a pure lift). Then convert
`test_gpu_les_smag` to link `dns_gpu` directly instead of shelling out to
`$<TARGET_FILE:cylinder_tunnel_gpu>`, and delete the CMake comment at
`CMakeLists.txt:168-176` that documents the hack. That conversion is the proof
the move was faithful.

Verify: `gpu_les_smag` (converted), `val_cyl_re60`. Moves numbers: no.

### Step 8 — `gpu/cutpoisson.hpp` split + `gpu/derived.hpp`
Split `gpu/cutcell.hip` (486 lines, two concepts) into `cutcell` (apertured
operators, the geometry axis) and `cutpoisson` (PCG + its FFT/DCT
preconditioner, the CG-solver axis). Move `sumScalar`/`maxAbsScalar` to
`reductions.hpp`. Merge `gpu/projection.hip` (18 lines) into `kernels.hip` and
`src/dns/projection.cpp` (19 lines) into `timestepper.cpp`. Merge
`poisson_fft.cpp` + `poisson_jacobi.cpp` into `poisson.cpp`. Add
`gpu/derived.hip` (device-side vorticity/speed/curl) and repoint `RunOutput`'s
callers at it, deleting the host slab loops.

Verify: `poisson`, `projection`, `gpu_projection`, `gpu_poisson*`,
`gpu_cutcell*`, `webframes`, `fieldpng`. Moves numbers: no (device-side
vorticity is computed in the same precision as the host loop it replaces —
confirm on one frame before deleting the host path).

### Step 9 — `runner_incomp` / `runner_comp` — **MOVES (risk)**
Last and riskiest: the RK3 `stage` lambda (5 copies) and the surrounding time
loop (5 copies, ~90 lines each). Only attempt after steps 3-8 have shrunk each
main to geometry + IC + a `Case`.

Ordering details that carry physics and must be preserved exactly:
- the incompressible stage order is `fusedRhsStage` → optional LES →
  `maskSolidFaces` → fringe **or** (bc + outlet flux correct) → `divergenceCut`
  → `addFluxRhs` → `poisson->solve` → `subtractGradientCut` → bc refresh;
- the force tally reads the **first** stage's `phi`;
- the compressible boost/ramp fires **after** `dev->step` and **before**
  `applyBc`;
- `stableDt` cadence: comp_cylinder recomputes every 16 steps with a 0.95
  margin, comp_mesh and the incompressible apps every step. Expose as
  `--dt-recompute`, defaulted **per current app** (16 comp_cylinder, 1
  otherwise). Do not standardise; see §6.

Verify: `gpu_conservation`, `gpu_energy_stability`, `gpu_tgv`, `val_spindown`
(analytic), `val_tgv3d_64`, `val_comp_mesh_sphere`. Moves numbers: only through
FP reassociation from retranslating units — move code without editing
expressions, and do any algebraic cleanup in a separate follow-up.

### Step 10 — one binary
Collapse `apps/` to `dns_main.cpp` + the four subcommand dispatchers. Delete
all remaining `apps/*` files and their `add_executable` entries. Rewrite the
nine tests that bake `$<TARGET_FILE:…>` (`test_val_cyl_re60/100/200`,
`val_cyl_blockage`, `val_sphere_re300`, `val_mesh_sphere`, `val_comp_cyl`,
`val_comp_mesh_sphere`, `val_tgv3d`, `gpu_open_incomp`) to invoke
`$<TARGET_FILE:dns>` with the migrated argument strings from §2.4, in the same
commit. Do **not** add per-preset shim executables — they would preserve
exactly the 15-binary surface this pass exists to remove.

Verify: full `ctest`, both with and without `-DDNS_HEAVY_VALIDATION=ON`.

### Step 11 — error-handling sweep
Apply §5 across the tree: `DNS_HIP_TRY` on every HIP/hipFFT/rocprim call (~20
statuses are currently `(void)`-discarded, including `cutcell.hip:459`, the
memcpy of `[rr,dq]` that drives the CG loop); the four `hipfftExec*` statuses
at `poisson.hip:343-402` that are `[[maybe_unused]] + assert`, i.e. discarded in
the only build that runs; the three duplicated reduction-scratch regrowth blocks
that dereference a valueless `expected` in Release; CG `isfinite(rr)` (NaN
compares false, so today a blown-up field reports a perfect solve at iteration
0); `nShells == 0` at `diagnostics.cpp:108` (OOB write on every 2D spectrum);
the `trimesh`/`meshsdf` input-validation set.

Verify: full `ctest`; add `test_errors` driving each error path to its exit code.
Moves numbers: no.

---

## 5. Error-handling + validation spec

Adopted from analysis 3, trimmed where noted.

### The rule
> `std::expected` for anything a **user** can cause.
> `assert` for anything only a **programmer** can cause.
> Nothing in between, and no third category.

`std::expected` covers: CLI parsing, config validation, every file
open/read/write/**close**, mesh loading and validation, every GPU resource
acquisition, every `hipError_t`-returning call, solver convergence, and state
admissibility (non-finite field, non-positive density).

`assert` is legitimate **only if** the validator provably makes it unreachable.
Every assert today guarding a user-reachable condition migrates: `grid.hpp:22`,
`poisson.hip:254,291` (the NeumannBox `n >= 4` invariant), `cutcell.hip:271,383`
(`nFluid > 0`, else `kFluidMean` divides 0/0 and poisons every `projectMean`),
`comp.cpp:39,55` (ρ,p positivity), `hit_ic.cpp:143`, and every
`n <= INT_MAX`. The project builds Release; an assert there is a comment.

Never: silent defaulting on parse failure, `atof`/`atoi`, discarding a status
via `(void)` or `[[maybe_unused]] + assert`, or a sentinel a caller cannot
distinguish from success. `(void)` is exempt **in destructors only**, with a
comment.

### HIP discipline
```cpp
#define DNS_HIP_TRY(call) \
    if (auto e_ = (call); e_ != hipSuccess) \
        return std::unexpected(Error::hip(e_, #call, __FILE__, __LINE__))
```
Kernel launches keep `checkLastKernel`; it should also `hipDeviceSynchronize`
in Debug so failures attribute to the right kernel.

### Four-phase `main` — phase order is the mechanism
1. **Parse** — pure string → `RawConfig`. No filesystem, no GPU.
2. **Validate** — pure `RawConfig` → `Case`. No filesystem, no GPU.
3. **Acquire** — in order: (a) probe GPU; (b) load and validate mesh;
   (c) **create and truncate every output file and directory the run will ever
   write, checking each**; then (d) `hipMalloc` / FFT plans / geometry.
4. **Run** — the time loop, which by construction can now only fail 7 or 8.

Phase 3c is why a full disk is discovered in the first second, not after hours
of GPU. `--dry-run` (phases 1-2, print the resolved `Case`, exit 0) falls out
free and is the validator's regression test.

### Exit codes
`0` ok · `1` CLI · `2` config · `3` input (mesh) · `4` output · `5` no GPU ·
`6` resource acquisition · `7` numerical failure (non-finite state, negative
density, dt not finite) · `8` solver non-convergence · `77` skipped (preserves
the ctest convention).

Distinguishing 6/7 matters: a validation run dying at 7 is a physics
regression, at 6 a machine problem. Today both print "GPU alloc failed" and
return 1.

### Messages
```
dns: unknown option '--reynolds' (did you mean '--re'?)
dns: try 'dns --help' or 'dns flow --help'
dns: --les 0.1 requires --nz 1 (the incompressible Smagorinsky path is 2D-only)
```
`ConfigError` is `{code, field, message}`, formatted once at the top level.
No error message is ever a warning the run continues past.
*(Trimmed from analysis 3: the "did you mean" suggestion uses a cheap common-
prefix match, not Levenshtein.)*

### Validator constraints
**Grid** — `nx,ny,nz >= 1`; `nx*ny*nz` computed in `size_t` and `<= INT_MAX`;
only `z` may be degenerate; `nz == 1 || nz >= 4`; confined axes `>= 4`;
`nz == 1 ⇒ hz == 1`; all lengths finite and positive; total footprint within a
memory budget derived from the selected solver.

**Solver** — exactly one physics; Poisson backend compatible with the boundary
set (NeumannBox is GPU-only; FFT-periodic needs all axes periodic; cut-cell
needs a body); `tol > 0 && isfinite`; `maxIters >= 1`; compressible rejects
`--tol`/`--poisson`, incompressible rejects `--mach`/`--ramp`/`--kick`.

**Physics** — `re > 0` finite; `0 < mach < 1`, warn above 0.3 (no shock
capturing, KG central); `gamma > 1`; `0 < cfl <= 1`; `ft, fdt, pdt, wdt > 0`
and each cadence `<= ft` (else the run silently writes zero frames);
`ramp >= 0`; `kick` finite; `cs >= 0`, and `cs > 0` with incompressible physics
requires `nz == 1` — checked **before** geometry construction, unlike today
(`cylinder_tunnel_gpu_main.hip:516`, after the `sanitizeCut` print). The shared
compressible LES core is 3D-valid and carries no such restriction; the
validator must distinguish the two.

**DNS-vs-LES honesty** — warn (do not reject) when `D/sqrt(re)` is under ~3
cells, and record `lesLabel` in `Case` so `RunMeta` says "LES" vs "DNS"
automatically instead of by human discipline.

**Geometry** — mesh kind ⟺ recognised extension; `scale > 0 && isfinite`;
auto-scale requires a positive extent; `dr > 0`; `D >= 4h`; `D < min(L)`;
the body AABB after scale/rotation/placement lies strictly inside the domain
with the ghost/taper band clear (prevents both the all-solid `nFluid == 0`
disaster and the "body 67% downstream" geometry break CLAUDE.md records);
mesh watertight unless `--robust` (`meshsdf.cpp:123` computes `watertight_`
today and never uses it); triangle count > 0; all coordinates finite;
`--dimples > 0` requires enough cells across D that the 5h/2.5h pits do not
consume the body.

**Boundaries** — `--open` rejected (not ignored) with tunnel/fringe/periodic;
`nscbc-strip >= 0` and `< nx/4`, and `> 0` requires `--far nscbc`;
`nscbc-beta ∈ [0,1)` — `>= 1` rejected outright, because 1.0 is *measured* to
blow up at t=35 (full cancellation is ill-posed at low M);
`--far nscbc` with `re > 800` and `nscbc-strip == 0` is rejected with the
`~ny/64` guidance (pure NSCBC blows up at Re2000/2048²);
sponge width `ns = clamp(0.1*nx, 8, nx/2-1)` derived only after checking
`nx >= 18`, else `hi < lo` and `std::clamp` is UB — reachable today from
`… 16`; `etaIn >= 50` (etaIn = 2 measured to sag the inflow to 0.79 uInf and
under-predict cd by 30%).

**Output** — `--out` non-empty and JSON-safe (`webframes.cpp:88-113`
concatenates it into the manifest unescaped); PNG requires `nz == 1`; spectrum
output requires `min(nx,ny,nz)/2 >= 1`; directory creatable and writable
(phase 3c, so `validate` stays pure).

*(Trimmed from analysis 3: the "no two outputs collide on the same path" check
— the prefix scheme makes collisions structurally impossible.)*

---

## 6. Keep-list — validated physics, do not "simplify"

Each of these looks like mess and is not. They concern **stability and
correctness**, so setting the anchors aside does not license touching them.

1. **The ghost view never feeds convection** (`ghostfill.*`, `comp2d_ghost.*`).
   Mirror values in the skew-symmetric form *inject energy* — measured and
   tested. It looks like an inconsistency (diffusion sees a different field
   than convection) and "unifying" it silently breaks energy stability.
   Preferred hardening: a distinct `GhostView` type the convection kernels
   cannot accept.
2. **`comp_ghost.hip:156` samples ≥ 0.75h beyond the wall**, never plain
   `2|sdf|` mirroring. Barely-inside cells otherwise lose their stencil and the
   force tally's closed surface breaks. Name the constant, cite the reason.
3. **`betaTrans = 0.5` default and the `< 1` bound.** An empirically bracketed
   stability window: 0 pumps a growing global mode (cl 2→13), 1.0 blows up at
   t=35. Not a magic number.
4. **`etaIn = 100` stiff inflow.** etaIn = 2 measured cd 30% low.
5. **Free-slip (not Dirichlet) lateral tunnel walls.** The Dirichlet u = uInf
   far field clamped velocity against sheared flow and painted a vortex sheet.
   A pass "regularising" the mirror/antimirror asymmetry would undo a validated
   fix. Add a regression test that fails if a Dirichlet lateral row reappears.
6. **`nz == 1` as the exact 2D restriction, with no separate code path.**
   The z-stencil cancellation is exact by address arithmetic. Never branch
   z-terms out; the `*2d` parity tests exist to catch exactly that. The
   validator constrains *around* it (only z degenerate, `hz == 1`); the code
   does not special-case more. The degenerate-axis switches that do exist
   (`lamz = {0}`, `hipfftPlan2d`, the skipped DCT pass, `ghostOff`) are
   legitimate and documented.
7. **`stableDt` every 16 steps with a 0.95 margin** in the compressible
   cylinder path. A perf fix that changes trajectories but not statistics,
   deliberately. Keep as a default, name both constants — otherwise someone
   "fixes" it back to every-step for determinism.
8. **The `bbox` culling parameter** in `DeviceCompGhost::make` and the IC
   taper. Reads as premature optimisation; is a ~20× setup win (26 s → 1.3 s at
   256³) and bit-identical. `bbox = nullptr` stays the analytic/test path.
9. **The deterministic solenoidal `--kick` trip.** Divergence-free hence
   acoustically silent, and deterministic; the quiet start is genuinely too
   symmetric to shed without it. Record its value in output metadata so no two
   runs are compared across different kick settings by accident.
10. **The quiet start (`--ramp`, `boostX`).** IC at rest is exactly consistent
    with the no-slip body; measured startup |p′|/p∞ 12.7% → 0.56%.
11. **Per-body sample budgets** in the cut constructors (§ step 4) and the
    per-body placement offsets (§ `bodies.hpp`). Both select geometry and
    attractor, not style.
12. **The two grid conventions** (incompressible `L = 2π`, compressible
    `h = D/64·(512/ny)` with a fixed 8D×8D box). Both separately validated;
    merging them relocates the body and rescales Re.
13. **`CutPoissonSolver` (CPU) alongside `DeviceCutPoissonSolver` (GPU).**
    The documented CPU-reference / GPU-mirror pairing, both tested. Not
    accidental duplication.

---

## 7. Anchor regeneration

### Current state (verified)
Every row in `data/validation/anchors.csv` is **already commented out** —
`cyl_re60_n256_cd`, `comp_re60_n256_cd`, `comp_re800_n512_cd`,
`comp_re800_n512_st`, `sphere_re300_n160_cd` are all `TODO`, deferred by
decision on 2026-07-17. Anchor-gated checks therefore already print
`[DEFER]` and their tests already skip (rc 77). **There is nothing to
quarantine there.** Do not delete the file or the rows.

What *does* assert live numbers is the **literature-band** half of the `val_*`
tests, which is not anchor-gated:

| test | live assertion | at risk from |
|---|---|---|
| `val_cyl_re60` | St ∈ [0.130,0.156], cd ∈ [1.42,1.90], cl_rms > 0.01 | step 6 (friction links — expected no-op, verify) |
| `val_cyl_re100` | St ∈ [0.157,0.205], cd ∈ [1.33,1.82] | as above |
| `val_cyl_re200` (heavy) | St ∈ [0.183,0.235], cd ∈ [1.32,1.78] | as above |
| `val_cyl_blockage` (heavy) | dr16 cd/St bands + monotonicity | as above |
| `val_sphere_re300` (heavy) | cd ∈ [0.657,0.77], \|cl\| ∈ [0.01,0.10] | step 6 (fz added) |
| `val_mesh_sphere` (heavy) | cd within 5% of the analytic-sphere pipeline | step 4 (sample budgets) |
| `val_comp_mesh_sphere` | analytic-class sphere drag | step 6 (**FP64 tally — will move**) |
| `val_tgv3d_64` | curve match + peak band | step 9 (FP reassociation only) |

### Quarantine procedure (step 1, one CMake edit)
Add `option(DNS_VALIDATION_BASELINE "Register validation tests whose bands
predate the 2026-07-19 recomposition" OFF)` and guard the registration (not the
sources, not the CSV) of the eight tests above with it. Every guarded block
carries the marker comment:
```cmake
# QUARANTINED 2026-07-19 (masterplan_recompose §7): bands measured before the
# kBcTunnel corner unification, the 3-direction friction tally, and the FP64
# force tally. Re-enable after regenerating per docs/validation.md Tier C.
```
`test_val_*.cpp` sources and `data/validation/anchors.csv` **stay on disk,
unmodified**. `val_spindown` (analytic truth, no body force) is **not**
quarantined and must stay green throughout — it is the physics canary.

### What must be re-measured, and how
After step 10, with the unified binary, re-run each configuration below from the
repo root and recompute the statistic with the estimators in
`tests/validation_util.hpp` (trapezoidal `windowMean` for cd, hysteresis
zero-crossing `strouhal` for St; compressible St = f·π/4), exactly as
`docs/validation.md` "Tier C — anchor regeneration" prescribes:

```sh
build/dns flow --body cylinder --far tunnel --n 256  --nz 1 --re 60  --dr 8 --ft 12 --fdt 1e9 --out /tmp/a60
build/dns flow --body cylinder --far tunnel --n 512  --nz 1 --re 60  --dr 8 --ft 12 --fdt 1e9 --out /tmp/a60b
build/dns flow --body cylinder --far tunnel --n 1024 --nz 1 --re 60  --dr 8 --ft 12 --fdt 1e9 --out /tmp/a60c
build/dns flow --body cylinder --far tunnel --n 512  --nz 8 --re 60  --dr 8 --ft 12 --fdt 1e9 --out /tmp/a60z
build/dns flow --body cube     --far tunnel --n 256  --nz 8 --re 300        --ft  6 --fdt 1e9 --out /tmp/cube
build/dns flow --body sphere   --far tunnel --n 160  --re 300 --dr 8 --ft 12 --fdt 1e9 --out /tmp/s300
build/dns flow --body sphere.stl --far tunnel --n 96 --re 300 --dr 8 --ft 12 --fdt 1e9 --out /tmp/ms
build/dns flow --physics compressible --body cylinder --far sponge --n 256 --re 60  --mach 0.2 --ft 20 --fdt 1e9 --out /tmp/c60
build/dns flow --physics compressible --body cylinder --far sponge --n 512 --re 800 --mach 0.2 --ft 15 --fdt 1e9 --out /tmp/c800
build/dns flow --physics compressible --body sphere.stl --far nscbc --open --n 96 --nz 96 --re 100 --mach 0.2 --ft 12 --out /tmp/cms
```

Anchors to fill in `data/validation/anchors.csv` (uncomment, set value + date +
the new `config` string):
`cyl_re60_n256_cd`, `comp_re60_n256_cd`, `comp_re800_n512_cd`,
`comp_re800_n512_st`, `sphere_re300_n160_cd`.

Anchors to **add** (they did not exist and the recomposition creates the need):
- `cyl_re60_n512_nz8_cd` — pins the new diagonal corner treatment (step 5).
- `cube_re300_n256_cd` / `_cdf` — cube gains a friction contribution (step 6).
- `comp_re60_n256_cd_fp64` — the deterministic FP64 tally (step 6); its
  acceptance test is bit-identical repeats, so `tol_rel` may be 1e-6 rather
  than the 0.01 the FP32 era needed.

Then re-measure the literature bands in `docs/validation.md`'s per-test table,
re-derive the confinement-excess table (Re60/100/800 cd and St vs unbounded)
from the new numbers, un-quarantine the eight tests, and record in CLAUDE.md
that the 2026-07-16..18 numbers are superseded, naming the three physics changes
(corner unification, 3-direction friction, FP64 tally) as the cause.

Regeneration is a **separate task**, not part of this recomposition.

---

## 8. Non-goals / out of scope

- **Bit-identity with the pre-recomposition tree.** Explicitly set aside by the
  user. Do not design around it; do not add compatibility modes or per-body
  behaviour enums to preserve numbers.
- **Deleting `data/validation/anchors.csv` or any `tests/test_val_*.cpp`.**
  Quarantine only.
- **Re-measuring the anchors.** §7 specifies what and how; performing it is a
  follow-up task.
- **Unifying the two grid conventions** (incompressible `L = 2π` vs
  compressible fixed 8D×8D). Both stay, dispatched on `--physics`.
- **Direct-forcing staircase immersed boundary.** `apps/cube_main.hip`'s path
  is deleted, not ported. No `--forcing` flag exists in the new CLI: cut-cell
  is the only body representation. (Deliberate scope reduction — the path is
  untested, superseded by cut cells, and its survival would keep an entire
  second geometry axis alive.)
- **Staircase (non-ghost) wall mode.** Likewise not exposed; the sharp
  ghost-fill wall is the only wall. No `--wall` flag.
- **3D incompressible Smagorinsky.** `--les` stays 2D-only on the
  incompressible side (the compressible core's LES is already 3D-valid).
  Generalising it is a physics project, not a recomposition.
- **Algebraic/numerical cleanup of kernel bodies.** Move code without editing
  expressions; FP reassociation on gfx1100 is a separate, separately-verified
  pass.
- **Replacing FFTW/hipFFT, changing the Poisson algorithms, or touching the
  NSCBC/LODI formulation.** Structure only.
- **New physics of any kind**, including the deferred items in CLAUDE.md
  (FP64 gauge-p differencing, wall-aware transverse stencil, LES νt at boundary
  rows, betaLat split, LODI-corrected `comp_bc` replacement).
- **Performance work**, beyond the incidental wins that fall out of
  `gpu/derived.hip` replacing host slab pulls.
