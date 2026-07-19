# analysis_recompose_1 — minimal public API + one unified executable

TPRI agent 1 of 3. Focus: **public API trim + single binary with a redesigned CLI.**
No code was modified. All line references are to the tree as of 2026-07-19.

---

## 0. Executive summary

`apps/` is 6149 lines across 16 files (15 executables) of which, by inspection,
**~3400 lines are verbatim or near-verbatim duplication**. The duplication is not
incidental — it is the *entire* incompressible-body pipeline (BC kernels, force
tallies, LES, frame writers, RK3 stage lambda, blowup forensics, JSON/CSV
emission) copy-pasted five times, and the compressible pipeline copy-pasted
twice.

Three structural consequences drive the design below:

1. **The app files contain physics.** `kBodyPressure`, `kBodyFriction`,
   `kSmagNut`, `kEddyDiffuse`, `kBodySgsFriction`, `kBcTunnel`, `kFringe`,
   `kInOutFluxRhs` are `__global__` kernels living in anonymous namespaces in
   `apps/*.hip`. They are validated, anchored physics with no library home.
   CMakeLists.txt:169-177 and :159-167 openly admit this — two tests exist *only*
   to drive a built binary because "the LES Smagorinsky path is app-local
   (anonymous namespace in cylinder_tunnel_gpu)". That is the smell to fix.
2. **The CLI is inconsistent per app.** Three different parser idioms coexist:
   `parseOr` + `else if` chain (11 apps), `arg/argS/argB` scanning
   (`comp_cylinder_tunnel_gpu_main.hip:37-56`), and positional-only (`tgv`,
   `tgv3d`, `bench`). Flags overlap in name but differ in meaning (`--nz` is
   *interior z cells + ghosts* in cylinder, *raw nz* in comp_cylinder;
   `--dr` is domain/diameter ratio everywhere but the resulting `h` is computed
   two different ways). There is no `--help` anywhere.
3. **Only two of the five "identical" BC kernels are actually identical.** See
   §1.3 — this is the single biggest bit-identity trap in the recomposition.

---

## 1. Inventory

### 1.1 App table

| # | Executable | src | LOC | Unique capability | Flags | CMake |
|---|---|---|---|---|---|---|
| 1 | `tgv` | `apps/tgv_main.cpp` | 65 | 2D Taylor-Green, CPU, error vs analytic + KE ratio, optional VTK | positional `n nu T`, `--vtk` | :46 |
| 2 | `tgv3d` | `apps/tgv3d_main.cpp` | 118 | 3D TGV Re1600 dissipation CSV; **auto CPU/GPU** (`#if DNS_WITH_HIP` :78) | positional `n re tEnd` | :49,:124 |
| 3 | `bench` | `apps/bench_main.cpp` | 69 | ms/step + VRAM at fixed dt; exercises forcing path | positional `n steps` | :87 |
| 4 | `hit` | `apps/hit_main.cpp` | 207 | Forced HIT in SI units (`HitDesign`), spectra, snapshots, stationary averages | `--urms --nu --spin --stats --seed --out --snap` | :90 |
| 5 | `movie` | `apps/movie_main.cpp` | 198 | Mid-plane speed/vort/**pressure** slices for tgv or hit | `tgv|hit` subcmd, `n`, `--out` | :93 |
| 6 | `cube` | `apps/cube_main.hip` | 303 | **Direct-forcing staircase IB** on the periodic solver (only app using this) + x-only fringe | `--re --dr --ft --out` | :96 |
| 7 | `cube_cut` | `apps/cube_cut_main.cpp` | 263 | **CPU** cut-cell, drag from fringe momentum injection | `--re --dr --ft --flat --fdt --out` | :52 |
| 8 | `cube_cut_gpu` | `apps/cube_cut_gpu_main.hip` | 296 | GPU cut-cell, periodic + `kFringeCut`, FFT-preconditioned CG | `--re --dr --ft --flat --fdt --tol --out` | :99 |
| 9 | `sphere_cut_gpu` | `apps/sphere_cut_gpu_main.hip` | 280 | as #8, non-grid-aligned sphere (exercises `sanitizeCut`) | `--re --dr --ft --fdt --tol --out` | :102 |
| 10 | `cube_tunnel_gpu` | `apps/cube_tunnel_gpu_main.hip` | 456 | Tunnel + rotated cube; `downloadSideSlab` (xz slice, :153) | `--re --dr --ft --flat --fdt --tol --out` | :108 |
| 11 | `sphere_tunnel_gpu` | `apps/sphere_tunnel_gpu_main.hip` | 605 | Tunnel + sphere; **`GolfBall`** 3D dimple sdf (:184) + `makeGolfCut` (:228) | `--re --dr --ft --golf --fdt --tol --out` | :105 |
| 12 | `cylinder_tunnel_gpu` | `apps/cylinder_tunnel_gpu_main.hip` | 857 | Tunnel/`--open` + cylinder; **`GolfDisk`** 2D dimples (:368); **app-local LES** (:245-352); PNG output | `--re --nz --dr --ft --cfl --les --open --golf --fdt --tol --out` | :111 |
| 13 | `eye_tunnel_gpu` | `apps/eye_tunnel_gpu_main.hip` | 797 | Tunnel + **`Lens`** body (:329) w/ exact chord apertures (:339) | `--re --nz --dr --ar --ft --cfl --les --fdt --tol --out` | :115 |
| 14 | `mesh_tunnel_gpu` | `apps/mesh_tunnel_gpu_main.hip` | 670 | Tunnel/`--open` + `MeshSdf`; `--vol/--voldt` 3D volume dump | `--re --dr --scale --pitch --yaw --robust --open --vol --voldt --ft --fdt --tol --out` | :117 |
| 15 | `comp_cylinder_tunnel_gpu` | `apps/comp_cylinder_tunnel_gpu_main.hip` | 349 | Compressible cylinder; sponge/NSCBC/open; ramp+kick; PNG + WebFrames | `--ny --nz --re --mach --ft --fdt --pdt --cfl --open --nscbc --nsstrip --nsbeta --ramp --kick --out` | :113 |
| 16 | `comp_mesh_tunnel_gpu` | `apps/comp_mesh_tunnel_gpu_main.hip` | 616 | Compressible 3D mesh; six-face NSCBC; 3D LES; 9-plane WebFrames (:452-480) | `--ny --nz --re --mach --scale --dr --pitch --yaw --robust --les --nsbeta --nsstrip --ramp --kick --slices --ft --fdt --wdt --cfl --out` | :119 |

### 1.2 Verbatim duplication catalogue

Measured by extracting each function and diffing. "Identical" = byte-identical
after stripping comments.

| Symbol | Copies | Locations | Status |
|---|---|---|---|
| `parseOr(const char*, double)` | 8 | cyl:54, eye:47, mesh:43, sphere:36, cube_tunnel:34, cube_cut_gpu:33, sphere_cut:30, comp_mesh:54 | **identical** |
| `downloadSlab` | 6 | cyl:354, eye:317, mesh:215, sphere:275, cube_tunnel:143, movie:27 | **identical** |
| `kBcTunnel` | 5 | cyl:75, eye:62, sphere:51, cube_tunnel:52, mesh:66 | **2 distinct variants** — see §1.3 |
| `kOutletTally` | 5 | cyl:135, eye:98, sphere:87, cube_tunnel:87, mesh:124 | 2 variants (`kInLo/kInHi` vs hard `1..nz-2`) |
| `kOutletCorrect` | 5 | cyl:147, eye:110, sphere:99, cube_tunnel:99, mesh:136 | same 2-variant split |
| `kInOutFluxRhs` | 5 | cyl:160, eye:123, sphere:111, cube_tunnel:111, mesh:148 | same 2-variant split |
| `kBodyPressure` | 5 | cyl:175, eye:138, sphere:126, cube_tunnel:126, mesh:163 | **identical** (all 5) |
| `kBodyFriction` | 4 | cyl:196, eye:159, sphere:147, mesh:182 | 2 variants: x/y links (cyl,eye) vs x/y/z links (sphere,mesh) |
| `kSmagNut` | 2 | cyl:245, eye:208 | **identical** |
| `kEddyDiffuse` | 2 | cyl:273, eye:236 | **identical** |
| `kBodySgsFriction` | 2 | cyl:307, eye:270 | **identical** |
| `kFringe` | 2 | cyl:114, mesh:102 | 1 line differs (2-axis vs 3-axis `max`) |
| `kFringeCut` | 2 | cube_cut_gpu:44, sphere_cut:41 | **identical** |
| `snapNode` | 3 | cube_cut:25, cube_cut_gpu:31, cube_tunnel:32 | **identical** |
| `kInLo/kInHi` | 2 | cyl:51-52, eye:44-45 | **identical** |
| `hasMeshExt` | 2 | mesh:51, comp_mesh:62 | **identical** |
| `rampFrac` | 2 | comp_mesh:71, comp_cyl:191 (lambda) | **identical** logic |
| Golf dimple sdf + supersampled cut | 2 | cyl `GolfDisk`:368+`makeGolfCut`:402, sphere `GolfBall`:184+`makeGolfCut`:228 | 2D/3D siblings, same algorithm |
| `writeFrame` (vort+speed slice) | 5 | cyl:682, eye:627, mesh:475, sphere, cube_tunnel | near-identical (PNG only in cyl/eye) |
| Blowup forensics block | 5 | cyl:741-761, eye:686-706, mesh:569-589, sphere, cube_tunnel | **identical** |
| RK3 `stage` lambda | 5 | cyl:603-636, eye:555-581, mesh:422-444, sphere, cube_tunnel | identical modulo `open`/LES branches |
| `vortSpeed` + `downloadPrims` | 1½ | comp_mesh:379-402; comp_cyl:241-287 inlines the same | near-identical |
| Solenoidal Gaussian trip ("kick") | 2 | comp_mesh:293-311, comp_cyl:160-167 | **identical** formula |
| stats CSV header + emit loop | 5 (incomp) + 2 (comp) | all tunnel apps | identical schema |
| `.json` metadata write (twice: pre/post) | 7 | every tunnel app | same keys, ad-hoc string concat |

**cylinder vs eye is the extreme case**: `apps/eye_tunnel_gpu_main.hip` is
`cylinder_tunnel_gpu_main.hip` with `CylinderZ`→`Lens`, `makeGolfCut`→`makeLensCut`,
and `--golf`→`--ar`. Lines 42-325 of eye are byte-identical to lines 49-352 of
cylinder modulo comment text. That is **284 duplicated lines in one pair.**

### 1.3 THE bit-identity trap: `kBcTunnel` has two real variants

Diffing the five copies (comments stripped) yields exactly two equivalence classes:

**Class A — quasi-2D** (`cylinder:75`, `eye:62`; byte-identical to each other):
```
const int km = nz == 1 ? k : (k == 0 ? 1 : k == nz - 1 ? nz - 2 : k);
...
else if (jm != j) u[c] = u[gidx(g, i, jm, k)];      // y mirror only
else if (km != k) u[c] = u[gidx(g, i, j, km)];      // then z mirror
...
else if (j == 0)  v[c] = -v[gidx(g, i, 2, k)];      // k, not km
else if (nz == 1 || k <= 1 || k == nz - 1) w[c] = 0;
```

**Class B — full 3D** (`sphere:51`, `cube_tunnel:52`, `mesh:66`; byte-identical
to each other):
```
const int km = k == 0 ? 1 : k == nz - 1 ? nz - 2 : k;   // no nz==1 guard
...
else if (jm != j || km != k) u[c] = u[gidx(g, i, jm, km)];  // DIAGONAL corner mirror
...
else if (j == 0)  v[c] = -v[gidx(g, i, 2, km)];             // km, not k
else if (k == 1 || k == nz - 1) w[c] = 0;
else if (k == 0)  w[c] = -w[gidx(g, i, jm, 2)];             // z antimirror row
```

The classes agree when `nz == 1`, and disagree on **edge/ghost-corner cells**
whenever `nz > 1`. Class A gives a corner cell the y-mirror value only; Class B
gives it the diagonal `(jm, km)` value. Class B additionally has an explicit z
antimirror row (`w[k==0] = -w[...,2]`) that Class A folds into `w = 0`.

Consequence: **`cylinder_tunnel_gpu --nz 8` (the quasi-2D default!) is not
reproducible by the 3D kernel.** Any unification that picks one kernel silently
moves the `val_cyl_re60 / re100 / re200 / blockage` anchors. The design below
keeps both as an explicit `CornerMode` enum, defaulted per-scenario to preserve
current behaviour. Same argument applies to:

- `kBodyFriction`: Class A tallies **x/y links only** (cyl:213-222, eye) because
  the free-slip z ghosts carry no viscous flux; Class B adds the z links
  (mesh:207-208). Picking wrong changes `cdf` and hence `cd` directly.
- `kOutletTally/Correct/kInOutFluxRhs`: Class A uses `kInLo(nz)/kInHi(nz)`,
  Class B hardcodes `1 .. nz-2`. Identical when `nz > 1`; **differ when `nz == 1`**
  (Class B would iterate an empty k-range). So the 2D path *requires* Class A.

---

## 2. Proposed single executable

### 2.1 Name and shape

**Binary: `dns`.** One executable, four subcommands, orthogonal flag groups.

```
dns <subcommand> [options]

  flow    body in a free stream (replaces the 10 tunnel/cut/cube apps)
  tgv     Taylor-Green vortex (2D or 3D)
  hit     forced homogeneous isotropic turbulence
  bench   step-time / throughput benchmark
```

Design rules applied:

- **Orthogonality.** Every axis named in the brief becomes exactly one flag
  group, and no flag encodes two axes. `--body` never implies `--far`;
  `--physics` never implies `--dim`.
- **No boolean-that-means-a-mode.** Today `--open` means "fringe far field"
  in the incompressible apps and "lateral sponge bands" in the compressible one —
  two different physics behind one spelling. Both become explicit `--far` values.
- **Dimension is derived, never doubled.** `--nz 1` *is* 2D (project invariant:
  "there is NO 2D code path"). There is no `--2d` flag. `--nz` always means
  *interior cells in z*, in every subcommand — fixing the current
  cylinder-vs-comp_cylinder disagreement.
- **Units and grid convention are stated once**, in `--help`, and are the
  `comp_cylinder` (fixed 2026-07-18) convention: domain fixed, resolution scales
  with `--ny`.
- Long flags only. No single-letter aliases (they invite collision as the
  matrix grows).

### 2.2 Proposed `--help` text

Top level:

```
dns — incompressible & compressible DNS/LES on a MAC grid (CPU + HIP)

usage: dns <subcommand> [options]
       dns <subcommand> --help

subcommands:
  flow    a body in a uniform free stream: drag/lift, wake, slices
  tgv     Taylor-Green vortex — 2D analytic check or 3D Re1600 benchmark
  hit     forced homogeneous isotropic turbulence (SI-configured)
  bench   step-time and throughput benchmark

common options (accepted by every subcommand):
  --backend cpu|gpu|auto    compute backend                     [auto]
  --out PREFIX              output file/dir prefix              [<subcommand>]
  --seed N                  RNG seed for any stochastic input   [1]
  --quiet                   suppress the per-200-step progress line
  --help                    this text, or the subcommand's

conventions:
  * nz == 1 IS the 2D restriction — there is no separate 2D code path.
    Every z stencil cancels identically and w stays exactly 0.
  * All runs are in box units: uInf = 1, rho_inf = 1, L = 2*pi unless the
    subcommand states otherwise. Re and Mach are the only physical inputs.
  * Time is measured in flow-throughs (--ft), converted internally to t.
```

`dns flow --help`:

```
dns flow — a body in a uniform free stream

usage: dns flow --body BODY [grid] [physics] [far field] [output]

GRID
  --n N                     cells along x (streamwise)          [512]
  --ny N                    cells along y; N = --n              [=n]
  --nz N                    cells along z; 1 = native 2D        [1]
  --dr RATIO                domain height / body diameter       [8]
                            fixes h = D/64 * (512/ny): the box is
                            constant, resolution scales with --ny.

BODY (exactly one; --body is required)
  --body cylinder           z-aligned circle            (2D or quasi-2D)
  --body sphere             ball                        (3D)
  --body cube               45-deg rotated square prism (2D or 3D)
  --body lens               two-disk intersection, sharp tips (2D)
  --body FILE.stl|.gltf|.glb   triangle mesh, sdf via BVH   (2D or 3D)

  body shape modifiers:
  --aspect A                lens width / height, A > 1          [3.5]
  --dimples N               golf-ball dimples, 0 = smooth       [0]
                            (cylinder -> 2D disks, sphere -> 3D caps)
  --scale S                 mesh scale; 0 = fit max extent to D [0]
  --robust                  mesh sign by winding number (leaky meshes)
  --mesh-sdf-band CELLS     supersampling band half-width       [0.87]

  attitude: bake it into the mesh with build/orient.py and leave the
  mesh unrotated. The legacy --pitch/--yaw rotate RAW MODEL AXES and are
  NOT aircraft angles; they are accepted but deprecated.

PHYSICS
  --physics incompressible|compressible                          [incompressible]
  --re RE                   Reynolds number on the body diameter [60]
  --mach M                  Mach number (compressible only)      [0.2]
  --les CS                  Smagorinsky constant; 0 = DNS        [0]
                            incompressible: 2D only. compressible: 3D-valid.
  --cfl C                   Courant target                       [0.8 / 0.4]

BODY REPRESENTATION
  --wall sharp|staircase    sub-cell ghost-fill no-slip, or the
                            legacy masked staircase               [sharp]
  --forcing cutcell|direct  aperture cut-cell projection, or the
                            direct-forcing immersed boundary      [cutcell]

FAR FIELD (--far, exactly one)
  --far tunnel              Dirichlet inflow + convective outflow +
                            free-slip lateral walls. Incompressible
                            default; pressure = DCT-preconditioned CG.
  --far fringe              fully periodic + cosine sponge on every
                            boundary relaxing u -> uInf. FFT pressure.
  --far sponge              compressible streamwise freestream sponge +
                            free-slip lateral walls (the validated
                            compressible default).
  --far nscbc               Poinsot-Lele characteristic boundaries.
  --far periodic            no far field at all (fringe-free periodic box).

  --open                    make the LATERAL faces open too. With
                            --far sponge: lateral rho/p-relax bands.
                            With --far nscbc: six-face characteristic.
                            Ignored by tunnel/periodic.
  --sponge-cells N          sponge/fringe band width; 0 = 10% of n   [0]
  --nscbc-beta B            Yoo-Im transverse correction; <0 = Mach   [0.5]
  --nscbc-strip N           absorbing strip inside the NSCBC rows;
                            0 = pure characteristic. Use ~ny/64 above
                            Re 800 (pure NSCBC blows up at Re 2000).  [0]

START-UP (compressible)
  --ramp T                  quiet start: IC at rest, freestream raised
                            by Galilean boosts over [0,T]. 0 = impulsive. [2.0]
  --kick A                  solenoidal Gaussian trip one D behind the
                            body; needed to break the quiet-start
                            symmetry. Divergence-free, so silent.       [0.05]

SOLVER
  --poisson auto|fft|dct|cg   pressure solve; auto picks from --far    [auto]
  --tol RMS                 CG tolerance (incompressible)              [1e-4]
  --max-iters N             CG iteration cap                           [500]

RUN LENGTH
  --ft N                    flow-throughs to integrate                 [20]

OUTPUT (repeatable; default = stats,bin)
  --emit stats              <out>_stats.csv (t,dt,K,cd,cl,cdf,cfl,maxDiv,iters)
  --emit bin                <out>_{vort,speed,mask}.bin float32 slices
  --emit png                png_<out>/{vort,speed}_%05d.png (2D only)
  --emit web                web_<out>/ live browser frames + planes.json
  --emit volume             <out>_{vol,lam}.bin full 3D field per frame
  --emit slices             the three centre planes instead of the volume
  --emit vtk                <out>_%05d.vtk legacy structured points
  --emit none               diagnostics only

  --fdt DT                  .bin / volume frame interval                [0.15]
  --pdt DT                  png interval; 0 = follow --fdt              [0]
  --wdt DT                  web frame interval; 0 = follow --fdt        [0]
  --planes N                web planes per axis: 1 = centre, 3 = +/- n/16 [3]
```

`dns tgv --help`:

```
dns tgv — Taylor-Green vortex

usage: dns tgv [options]

  --n N          cells per side                                  [64]
  --nz N         cells in z; 1 = the 2D analytic case            [=n]
  --re RE        Reynolds number (3D benchmark uses 1600)        [1600]
  --nu NU        viscosity; overrides --re when > 0              [0]
  --ft T         end time (TGV is reported in t, not flow-throughs) [20]
  --cfl C        Courant target                                  [0.9]
  --emit stats   t,K,dKdt,epsVisc CSV on stdout                  [on]
  --emit vtk     final field as legacy VTK
  --emit error   max-norm error vs the 2D analytic solution + KE ratio
                 (2D only: nz == 1)
```

`dns hit --help`:

```
dns hit — forced homogeneous isotropic turbulence

Configured in SI (air viscosity, target u'), sized so h = 1.5 eta; runs
internally in box units and reports both.

  --n N              cells per side                              [256]
  --urms U           target one-component rms velocity [m/s]     [60]
  --nu NU            kinematic viscosity [m^2/s]                 [1.5e-5]
  --spin TE          eddy turnovers of spin-up before statistics [10]
  --stats TE         eddy turnovers of statistics                [10]
  --emit stats       <out>_stats.csv
  --emit spectrum    <out>_spec_%d.csv every 0.25 Te (needs FFTW)
  --emit snapshot    <out>_snap_%d.{raw,json} every Te
  --emit slices      mid-plane speed/vorticity/pressure frames
```

`dns bench --help`:

```
dns bench — step-time and throughput benchmark

  --n N          cells per side                                  [256]
  --steps N      timed steps after 3 warm-up steps               [20]
  --physics incompressible|compressible                          [incompressible]
  Reports ms/step, Mcell-updates/s and VRAM in use.
```

### 2.3 Migration table (every current invocation)

| Current | New |
|---|---|
| `tgv 64 0.05 1.0 --vtk o.vtk` | `dns tgv --n 64 --nz 1 --nu 0.05 --ft 1.0 --emit error --emit vtk --out o` |
| `tgv3d 128 1600 20` | `dns tgv --n 128 --re 1600 --ft 20` |
| `bench 256 20` | `dns bench --n 256 --steps 20` |
| `hit 256 --urms 60 --nu 1.5e-5 --spin 10 --stats 10 --snap --out h` | `dns hit --n 256 --urms 60 --nu 1.5e-5 --spin 10 --stats 10 --emit snapshot --out h` |
| `movie tgv 256 --out m` | `dns tgv --n 256 --re 1600 --ft 20 --emit slices --fdt 0.1 --out m` |
| `movie hit 256 --out m` | `dns hit --n 256 --emit slices --spin 5 --stats 3 --out m` |
| `cube 256 --re 300 --ft 6` | `dns flow --body cube --forcing direct --far fringe --n 256 --re 300 --ft 6` |
| `cube_cut 128 --re 300 --flat 8` | `dns flow --backend cpu --body cube --far fringe --n 128 --nz 8 --re 300` |
| `cube_cut_gpu 256 --re 300 --flat 8 --tol 1e-4` | `dns flow --body cube --far fringe --n 256 --nz 8 --re 300 --tol 1e-4` |
| `sphere_cut_gpu 128 --re 300` | `dns flow --body sphere --far fringe --n 128 --re 300` |
| `cube_tunnel_gpu 256 --re 300 --flat 8` | `dns flow --body cube --far tunnel --n 256 --nz 8 --re 300` |
| `sphere_tunnel_gpu 128 --re 300 --golf 300` | `dns flow --body sphere --far tunnel --n 128 --re 300 --dimples 300` |
| `cylinder_tunnel_gpu 1024 --nz 1 --re 60 --dr 8` | `dns flow --body cylinder --far tunnel --n 1024 --nz 1 --re 60 --dr 8` |
| `cylinder_tunnel_gpu 512 --nz 8 --re 60` | `dns flow --body cylinder --far tunnel --n 512 --nz 8 --re 60` |
| `cylinder_tunnel_gpu 512 --open --re 60` | `dns flow --body cylinder --far fringe --n 512 --re 60` |
| `cylinder_tunnel_gpu 512 --les 0.17 --nz 1` | `dns flow --body cylinder --far tunnel --n 512 --nz 1 --les 0.17` |
| `cylinder_tunnel_gpu 1024 --golf 40` | `dns flow --body cylinder --n 1024 --dimples 40 --far tunnel` |
| `eye_tunnel_gpu 1024 --ar 3.5 --re 60` | `dns flow --body lens --aspect 3.5 --far tunnel --n 1024 --re 60` |
| `mesh_tunnel_gpu m.stl 96 --re 300 --robust --vol --voldt 0.75` | `dns flow --body m.stl --far tunnel --n 96 --re 300 --robust --emit volume --fdt 0.75` |
| `mesh_tunnel_gpu m.stl 96 --open` | `dns flow --body m.stl --far fringe --n 96` |
| `comp_cylinder_tunnel_gpu 512 --re 800 --mach 0.2` | `dns flow --physics compressible --body cylinder --far sponge --n 512 --re 800 --mach 0.2` |
| `comp_cylinder_tunnel_gpu 512 --nscbc --nsbeta 0.5` | `dns flow --physics compressible --body cylinder --far nscbc --nscbc-beta 0.5 --n 512` |
| `comp_cylinder_tunnel_gpu 512 --nscbc --open` | `dns flow --physics compressible --body cylinder --far nscbc --open --n 512` |
| `comp_cylinder_tunnel_gpu 1024 --nscbc --nsstrip 12` | `dns flow --physics compressible --body cylinder --far nscbc --nscbc-strip 12 --n 1024` |
| `comp_cylinder_tunnel_gpu 512 --ramp 0` | `... --ramp 0` (unchanged) |
| `comp_mesh_tunnel_gpu m.stl 96 --les 0.17 --nz 96` | `dns flow --physics compressible --body m.stl --far nscbc --open --n 96 --nz 96 --les 0.17` |
| `comp_mesh_tunnel_gpu m.stl 128 --slices` | `dns flow --physics compressible --body m.stl --far nscbc --open --n 128 --emit slices` |

**Renames** (old → new, all deliberate de-overloading):

| old | new | why |
|---|---|---|
| `--open` (incompressible) | `--far fringe` | it selected a far-field *model*, not a boolean |
| `--open` (compressible) | `--open` (kept) | genuinely a lateral-face modifier of `--far` |
| `--nscbc` | `--far nscbc` | ditto |
| `--flat nz` | `--nz` | one spelling for one concept |
| `--golf N` | `--dimples N` | says what it does |
| `--ar` | `--aspect` | |
| `--nsbeta / --nsstrip` | `--nscbc-beta / --nscbc-strip` | grouped prefix |
| `--vol / --voldt` | `--emit volume` + `--fdt` | output selection is one mechanism |
| `--snap` / `--slices` | `--emit snapshot` / `--emit slices` | ditto |
| positional `n` | `--n` | positional-vs-flag ambiguity today lets a stray token silently become `n` (`else n = int(parseOr(argv[a], n))`, cyl:479) |

The last one matters: today **every unrecognised argument silently becomes the
grid size.** `cylinder_tunnel_gpu --oepn` sets n = 512 (the parse fails, fallback
returned) and runs the wrong configuration without a word. The new parser must
**reject unknown flags with a nonzero exit**.

---

## 3. Proposed minimal public library API

### 3.1 Principle

Public headers describe *what the solver can do*; everything an app currently
open-codes moves behind one of them. Layering stays:

```
grid/field  ->  operators/cutcell/comp  ->  gpu/*  ->  scenario  ->  cli
                (no gpu include in a CPU header; no scenario include in a solver header)
```

New layer: `src/dns/scenario/` — the recomposed pipelines. It may include `gpu/`;
nothing in `dns/` or `dns/gpu/` may include it. That keeps the dependency graph
acyclic while giving the app kernels a home.

### 3.2 Header-by-header verdict

#### Keep as-is (already minimal, self-contained, no impl leakage)

| Header | Verdict |
|---|---|
| `grid.hpp` | Keep. 46 lines, header-only, exemplary. |
| `field.hpp` | Keep. |
| `units.hpp` | Keep. |
| `forcing.hpp` | Keep. |
| `taylor_green.hpp` | Keep. |
| `hit_ic.hpp` | Keep. |
| `vtk_writer.hpp` | Keep. |
| `fieldpng.hpp` | Keep. |
| `trimesh.hpp` | Keep. |
| `gpu/index.hpp` | Keep. |
| `gpu/device_buffer.hpp` | Keep. |
| `gpu/device_util.hpp` | Keep. |
| `gpu/reductions.hpp` | Keep. |
| `gpu/comp_flux.hpp` | Keep — already correctly marked "GPU-internal header (HIP TUs only)". Move to `src/dns/gpu/internal/` to make that mechanical rather than advisory. |

#### Trim (public surface is wider than needed)

**`poisson.hpp`** — `makeJacobiSolver` (`:23`) is a reference/teaching solver
used only by `test_poisson`. Demote: keep the `PoissonSolver` interface and
`makeDefaultSolver`; move `makeJacobiSolver` to an internal header the test
includes directly. Removes a public API nobody should call.

**`cutcell.hpp`** (153 lines, the widest CPU header) — `CutPoissonSolver`'s
private section (`:129-139`) exposes `apply/fluidDot/projectMean` and the CG
workspace `r_, d_, q_`. Private members are not API, but they force `<vector>`
weight and recompilation on every user. Pimpl it as `gpu::DevicePoissonSolver`
already is (`gpu/poisson.hpp:42-45`). Also: `RotatedCube`, `Sphere`, `CylinderZ`
(`:26-43`) are three analytic bodies in the *cut-cell* header — they belong in a
new `dns/bodies.hpp` (§3.3) so `cutcell.hpp` is purely the discretisation.

**`gpu/comp.hpp`** — the `DeviceComp` private section (`:81-97`) names
`k_, q1_, q2_, u_..T_, nut_, ptrs_`. Same pimpl argument. Note the public
`primitives()/u()/v()/...` accessors (`:53-59`) exist solely so app BC kernels
can reach the scratch — once the BC modules are library code (§3.3), these can
become `friend`-scoped or move to an internal view struct.

**`gpu/cutcell.hpp`** — `sumScalar` (`:43`) and `maxAbsScalar` (`:46`) are
generic reductions living in the cut-cell header purely because the tunnel apps
needed them for force sums. Move to `gpu/reductions.hpp` where they belong.

**`gpu/ghostfill.hpp`** — `using Sdf = std::function<double(double,double,double)>`
(`:24`) and `DeviceCompGhost::Sdf` (`gpu/comp_ghost.hpp:20`) are the same type
declared twice. Hoist one `dns::Sdf` into a new `dns/sdf.hpp` and have both use
it. Currently they are incompatible-by-name for no reason.

#### Delete / demote

**`gpu/comp_bc.hpp` + `comp_bc.hip`** — **delete from the public API.** Per
CLAUDE.md this path "passes unit tests but REGRESSES real flows (cd sigma 10x)"
and "Do not cite BC-run numbers"; no app includes it (grep: only
`tests/test_gpu_comp_bc.cpp` and `CMakeLists.txt:67`). A publicly reachable
header that must never be used in production is a trap. Options, best first:
(a) delete both, and delete `test_gpu_comp_bc`; (b) move under
`src/dns/gpu/internal/` and keep the test as a documented negative control. Do
not leave it as a peer of `comp_far.hpp` / `comp_nscbc.hpp`.

**`gpu/timestepper.hpp` `grid()` / `reduceScratch()` (`:33-34`)** — accessors
that exist only so `hit_main.cpp:125-126`, `tgv3d_main.cpp:89-90` and
`movie_main.cpp:158-159` can reach into the stepper's internals to run their own
reductions. With `dns hit` / `dns tgv` as library scenarios, the scenario owns
its own `ReduceScratch` and these two accessors disappear.

**`poisson_jacobi.cpp`** stays as a source file; only its declaration moves.

### 3.3 New public headers (the actual recomposition)

Six new headers absorb the ~3400 duplicated app lines. Each is one axis from
the brief.

```
src/dns/sdf.hpp                 // dns::Sdf typedef + analytic sdf helpers
src/dns/bodies.hpp              // Sphere, CylinderZ, RotatedCube, Lens, Dimpled<B>
                                //   + CutGeometry builders (incl. supersampled)
src/dns/gpu/tunnel.hpp          // DeviceTunnelBc, DeviceFringe
src/dns/gpu/bodyforce.hpp       // DeviceBodyForce (pressure + friction + SGS tallies)
src/dns/gpu/les.hpp             // DeviceSmagorinsky (incompressible 2D)
src/dns/scenario/flow.hpp       // FlowConfig + run(); the whole body pipeline
src/dns/scenario/canonical.hpp  // tgv / hit / bench scenarios
src/dns/output.hpp              // Emitter: stats/bin/png/web/vtk/volume/slices
```

**`dns/bodies.hpp`** — kills `GolfDisk` (cyl:368), `GolfBall` (sphere:184),
`Lens` (eye:329), both `makeGolfCut`s (cyl:402, sphere:228), `makeLensCut`
(eye:339), `snapNode` x3. Sketch:

```cpp
// Analytic solid shapes in world units; sdf < 0 inside. Self-contained.
struct Lens { double cx, cy, w, hh; };            // w > hh
struct Dimpled { Sdf base; std::vector<...> pits; double rd, depth; };

double sdf(const Sphere&,      double x, double y, double z);
double sdf(const CylinderZ&,   double x, double y, double z);
double sdf(const RotatedCube&, double x, double y, double z);
double sdf(const Lens&,        double x, double y, double z);

// Exact cut geometry where a closed form exists; supersampled otherwise.
// samples: cells per axis for lambda, segments per face for apertures.
CutGeometry makeCut(const Grid&, const Sdf&, int samples = 8);
```

**`dns/gpu/tunnel.hpp`** — kills 5x`kBcTunnel`, 5x`kOutletTally/Correct`,
5x`kInOutFluxRhs`, 2x`kFringe`, 2x`kFringeCut`. **This is where the §1.3 trap is
made explicit**:

```cpp
// Ghost-ring far field for the confined channel: Dirichlet inflow, convective
// outflow with exact mass-flux correction, free-slip lateral walls.
class DeviceTunnelBc
{
public:
    // Ghost-corner treatment on the lateral rings. The two variants agree at
    // nz == 1 and differ on y/z edge cells otherwise -- they are NOT
    // interchangeable, they anchor different validated runs.
    //   Sequential: y mirror, else z mirror   (cylinder/eye, quasi-2D)
    //   Diagonal:   single (jm,km) mirror + explicit z antimirror row
    //                                        (sphere/cube/mesh, full 3D)
    enum class Corners { Sequential, Diagonal };

    struct Config { double uInf = 1.0; Corners corners = Corners::Sequential; };

    static std::expected<DeviceTunnelBc, hipError_t> make(const Grid&, Config);

    // cdt = uInf*dt/hx drives the outlet from `in`; cdt == 0 with in aliased
    // to out is a pure ghost refresh.
    void apply(const DeviceField& in, DeviceField& out, double cdt) const;

    // Restore exact global mass balance on the outlet plane.
    void correctOutletFlux(DeviceField& out, ReduceScratch&, DeviceScalar& tmp) const;

    // Inlet/outlet faces carry prescribed flux at zero aperture, so
    // divergenceCut misses them; add the missing RHS terms. Pre: same scale.
    void addFluxRhs(const DeviceField& U, double scale, DeviceScalar& rhs) const;
};

// Cosine-ramped fringe relaxing u -> uInf on the domain boundaries.
// axes = 2 (x,y only: z stays plain periodic for z-invariant flow) or 3.
class DeviceFringe
{
public:
    struct Config { double uInf = 1.0, sigMax = 0.5; int cells = 0; int axes = 3; };
    static std::expected<DeviceFringe, hipError_t> make(const Grid&, Config);
    void apply(DeviceField& U) const;
};
```

**`dns/gpu/bodyforce.hpp`** — kills 5x`kBodyPressure`, 4x`kBodyFriction`,
2x`kBodySgsFriction`, and the identical `cd/cl/cdf` reduction blocks:

```cpp
// Discrete surface integral over an embedded body inside an index bbox:
//   pressure  F = -sum phi_c (a_xp - a_c) A_face   (exact aperture identity)
//   friction  F = nu sum over links open->closed of (q - q_ghost) A/d
// Links: XY tallies x/y only (free-slip z ghosts carry no viscous flux --
// the quasi-2D convention); XYZ adds the z links (fully 3D bodies).
class DeviceBodyForce
{
public:
    enum class Links { XY, XYZ };
    struct Box { int i0, i1, j0, j1, k0, k1; };
    struct Coeffs { double cd, cl, cdf; };   // cdf is nu-only, never nut

    static std::expected<DeviceBodyForce, hipError_t> make(const GpuGrid&, Box, Links);

    // ghosted = the ghost-filled view the viscous term actually saw.
    // nut = optional SGS field; its wall drain folds into cd/cl, not cdf.
    Coeffs measure(const DeviceCutGeometry&, const DeviceScalar& phi,
                   const DeviceField& ghosted, double nu, double qRef,
                   const DeviceScalar* nut, ReduceScratch&);
};
```

**`dns/gpu/les.hpp`** — kills the app-local Smagorinsky and, with it, the CMake
hack at `CMakeLists.txt:169-177`. The test can then link the library instead of
shelling out to a binary.

**`dns/output.hpp` (`Emitter`)** — kills 5x`writeFrame`, 6x`downloadSlab`,
7x hand-rolled `.json`, 7x stats CSV, 2x `vortSpeed/downloadPrims`, and the
9-plane WebFrames block (comp_mesh:413-493):

```cpp
// One sink for every run artefact. Selected by --emit; unselected kinds cost
// nothing (no buffers allocated).
class Emitter
{
public:
    enum class Kind { Stats, Bin, Png, Web, Volume, Slices, Vtk };
    struct Config { std::string prefix; std::set<Kind> kinds;
                    double fdt, pdt, wdt; int planesPerAxis = 3; };
    // Metadata written once at start and rewritten at end (frames, msPerStep).
    void meta(const RunMeta&);
    void step(double t, const StepStats&);       // stats row
    void frame(double t, const FieldViews&);     // whatever kinds are on
};
```

**`dns/scenario/flow.hpp`** — the single pipeline. `FlowConfig` is a plain
aggregate that maps 1:1 onto the `dns flow` flag groups, so the CLI layer is a
pure parser with no physics knowledge:

```cpp
struct FlowConfig
{
    Grid grid;
    Body body;                  // variant<Sphere, CylinderZ, RotatedCube, Lens, MeshRef>
    Physics physics;            // Incompressible | Compressible
    FarField far;               // Tunnel | Fringe | Sponge | Nscbc | Periodic (+ open)
    WallModel wall;             // Sharp | Staircase
    Forcing forcing;            // CutCell | Direct
    double re, mach, les, cfl, tol, ft;
    Output out;
};
// Pre: config validated by validate() below. Returns exit status.
std::expected<void, std::string> validate(const FlowConfig&);
int run(const FlowConfig&, Emitter&);
```

`validate()` is where the current silent misconfigurations become loud:
`--les > 0 && nz != 1 && physics == Incompressible` (today: a runtime
`fprintf` + `return 1` duplicated at cyl:512 and eye:472), `--aspect <= 1`
(eye:456), `--far nscbc && re > 800 && nscbc-strip == 0` (today: silent blow-up
at t=63.6), `--body mesh` with degenerate bounds (mesh:293), zero frontal area
(mesh:330).

### 3.4 Net effect

| | before | after (est.) |
|---|---|---|
| executables | 15 | 1 |
| `apps/` LOC | 6149 | ~350 (CLI parse + dispatch only) |
| public headers | 34 | 33 (−`comp_bc`, −8 moved internal, +8 new) |
| distinct `parseOr` impls | 8 | 1 |
| `__global__` kernels in `apps/` | 21 | 0 |
| CMake "drive the binary" test hacks | 2 | 0 |

---

## 4. Risks — what could move, and how to keep it bit-identical

Ranked by (probability x blast radius).

### R1 — `kBcTunnel` corner unification (HIGH / HIGH)

Anchors at risk: `val_cyl_re60`, `val_cyl_re100`, `val_cyl_re200`,
`val_cyl_blockage`, `val_sphere_re300`, `val_mesh_sphere`, and the memory-note
numbers (cyl Re60 cd 1.803/1.785/1.785; steady tail 1.813/cdf 0.553; sphere
Re300 cd 0.710 vs 0.685).

Mitigation: `Corners::Sequential` for cylinder/lens, `Corners::Diagonal` for
sphere/cube/mesh — chosen by `--body`, never by the user. **Do not "fix" the
Sequential variant** even though Diagonal is arguably the more principled
corner treatment; that is a physics change dressed as a cleanup.

Verification: before touching anything, capture `--ft 0.2` runs of all six
configurations and byte-compare `_stats.csv` after the refactor. A single
differing digit in `cd` means the corner class flipped.

### R2 — `kBodyFriction` link set (HIGH / HIGH)

`cdf` feeds directly into `cd` (`cd += cdf`, cyl:791). XY vs XYZ changes it.
Bind `Links::XY` to cylinder/lens (quasi-2D, free-slip z) and `Links::XYZ` to
sphere/mesh. Note `cube_tunnel_gpu` has **no** `kBodyFriction` at all — it
reports pressure drag only. Preserve that asymmetry or explicitly declare the
anchor regenerated.

### R3 — `kInLo/kInHi` vs hardcoded `1..nz-2` (MEDIUM / HIGH)

The 3D-family outlet kernels iterate `k in [1, nz-2]`, which is **empty at
nz == 1**. Unifying on the 3D form silently disables outlet flux correction in
every native-2D run — maxDiv would jump and the anchors would drift without an
obvious crash. The unified kernel must use `kInLo/kInHi` unconditionally; it is
a strict generalisation (identical for nz > 1).

### R4 — floating-point reassociation (MEDIUM / MEDIUM)

Moving kernels between translation units changes inlining and can change FP32
reassociation on gfx1100. Merging the two `fusedRhsStage` call sites, or
folding the LES `nut` multiply into the main flux, will not be bit-identical
even when algebraically identical.

Mitigation: move code **without editing expressions** in the first pass. Keep
kernel bodies byte-identical; change only their file and namespace. Do algebraic
cleanup in a separate, separately-verified pass. Compile the moved kernels with
the same flags (`-Wall -Wextra` at `CMakeLists.txt:84-85`, no fast-math today —
keep it that way).

### R5 — `stableDt` cadence and CFL margin (MEDIUM / MEDIUM)

`comp_cylinder_tunnel_gpu` recomputes every 16 steps with a 0.95 margin
(:296-302); `comp_mesh_tunnel_gpu` recomputes every step (:560); the
incompressible apps recompute every step (cyl:734). Trajectories are already
not comparable across these. The unified `flow` must expose the cadence as a
config field defaulted **per current app**, not standardised — CLAUDE.md
records that the comp_cylinder change already broke bit-identity with
pre-2026-07-18 runs, and statistics-only equivalence was accepted there.
Do not silently extend that acceptance to the incompressible anchors.

### R6 — grid convention divergence (MEDIUM / HIGH if missed)

`comp_cylinder` uses `h = D/64 * (512/ny)` with a fixed 8Dx8D box (:89);
every incompressible app uses `L = 2*pi`, `h = L/n`, `D = L/dRatio` (cyl:489-498).
These are genuinely different conventions and both are anchored. The unified
`--dr` must dispatch on `--physics`, and `--help` must say so. Merging them into
one formula silently relocates the body and rescales Re — exactly the class of
bug CLAUDE.md records as "geometry-broken, measured cd 3.19".

### R7 — body placement offsets (LOW / HIGH)

Every app places the body at a deliberately off-node point to trip the shedding
instability, and the offsets differ: cylinder `(h + L/3 + 0.31h, h + L/2 + 0.17h)`
(:498), mesh adds a z term `+0.23h` (:281-282), comp_cylinder uses absolute
`(2.1107, Ly/2 + 0.0146)` (:93). These are not arbitrary — they select which
attractor the run lands on. Carry each forward verbatim as a per-body constant;
do not normalise.

### R8 — `--nz` semantic change (LOW / MEDIUM, but user-visible)

Today `cylinder --nz 8` means 8 *interior* cells and allocates `nz = 10` with a
ghost ring (:491); `comp_cylinder --nz 8` means 8 cells, no ring. The new
`--nz` = interior cells everywhere. The incompressible path keeps its ghost ring
internally. This is the one intentional user-visible semantic change; it must be
called out in the release note, and the migration table above already reflects it.

### R9 — losing `cube`'s direct-forcing path (LOW / LOW)

`apps/cube_main.hip` is the only user of staircase direct forcing
(`kMaskCube` :36, bulk-velocity pin, drag from the momentum balance of that pin).
It is not covered by any validation test. Either carry it as
`--forcing direct` (proposed) or delete it deliberately and say so — do not
let it evaporate in the merge.

### R10 — test harness coupling (CERTAIN / LOW)

Eight tests hardcode binary names and argument strings:
`tests/test_val_cyl_re60.cpp:26`, `re100:27`, `re200:26`, `blockage:29`,
`test_val_sphere_re300.cpp:31`, `test_val_mesh_sphere.cpp:47,50`,
`test_val_comp_cyl.cpp:39,74`, `test_val_comp_mesh_sphere.cpp:26`,
`test_val_tgv3d.cpp:30`, `test_gpu_les_smag.cpp:44`,
`test_gpu_open_incomp.cpp:43`, plus `CMakeLists.txt:159-250`.
All must be rewritten in the same commit as the CLI change. Two of them
(`gpu_les_smag`, `gpu_open_incomp`) should stop shelling out entirely once LES
and the fringe are library code.

### Recommended verification gate

Before merging, for each of these nine configurations, run `--ft 0.2` old vs new
and require **byte-identical** `_stats.csv`:

```
cylinder tunnel nz=1 | cylinder tunnel nz=8 | cylinder fringe | cylinder les
sphere tunnel | cube tunnel | mesh tunnel | comp_cyl sponge | comp_cyl nscbc
```

Anything that cannot be made byte-identical must be declared, measured over a
full window, and the anchor in `data/validation/anchors.csv` regenerated per
`docs/validation.md` — not quietly re-baselined.
