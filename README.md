# dns — incompressible & compressible DNS/LES/RANS on a MAC grid (CPU + GPU)

One solver binary, `build/dns <flow|tgv|hit|channel|conv|bench|prop>`, plus one
front end, `build/dns-ui`. `dns --help` and `dns <sub> --help` are
authoritative for every flag and default — this file does not restate them.
Open problems: **`issues.md`**. Cheap runnable recipes with measured numbers:
`docs/scenarios.md`. Design history: `docs/archive/`.

- Incompressible: staggered MAC grid, 2nd-order central, skew-symmetric
  (KE-conserving) advection, SSP-RK3 with a pressure projection per stage.
- Compressible: conservative (ρ, ρu, ρv, ρw, E), Kennedy–Gruber KE-consistent
  central fluxes, SSP-RK3, fully explicit, **no shock capturing** (smooth
  subsonic flow only).
- Bodies: cut cells + sharp ghost-fill wall (incompressible), sharp ghost-mirror
  wall (compressible), analytic shapes, STL/glTF meshes, multi-object scenes.
- Meshes: uniform Cartesian, tanh wall stretching, focal clustering
  (`--focal`), body-fitted O-grids (2D/extruded, axisymmetric wedge, full
  revolution).
- Physics: DNS, LES (Smagorinsky, σ-model), RANS (SA-neg, SST-2003m, SST-Vm;
  steady or URANS), Sutherland viscosity and thermal walls, buoyant convection
  (compressible and Boussinesq), spinning rotors (moving wall or MRF).
- Far fields: free-slip tunnel, fringe, NSCBC (six-face, scale-selective sink,
  angle-adaptive closure), potential-corrected targets.
- Live output over shared memory to `dns-ui`; PNG, web, VTK, binary frames.

---

## Build and test

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDNS_WITH_HIP=ON   # AMD
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DDNS_WITH_CUDA=ON  # NVIDIA
ninja -C build
ctest --test-dir build --timeout 600   # GPU tests skip (rc 77) without a device
```

Release always. Both backend options OFF is the CPU-only build; both ON is a
configure error.

- **HIP** (`DNS_WITH_HIP`): ROCm on AMD, `CMAKE_HIP_ARCHITECTURES`
  auto-detected (gfx1100 and gfx1201 exercised). A wrong arch shows up as a
  HIP `getStatFunc` assertion.
- **CUDA** (`DNS_WITH_CUDA`): the SAME `src/dns/gpu/*.hip` sources compiled as
  CUDA through the header shim in `src/compat/cuda` (see its `README.md`).
  `CMAKE_CUDA_ARCHITECTURES` auto-detected (`native`, else a fat
  75/86/89/120 binary). Needs an nvcc accepting `-std=c++23`, which configure
  checks by compiling `<expected>`.
- Other options: `DNS_WITH_FFTW` (ON), `BUILD_TESTING` (master test switch),
  `DNS_HEAVY_VALIDATION` (Tier B, ~25 min, RUN_SERIAL),
  `DNS_VALIDATION_BASELINE` (quarantined literature-band `val_*`, OFF),
  `DNS_WITH_UI` (builds `dns-ui`).

Backends: CPU (`dns_core`) is serial FP64, the correctness oracle. GPU
(`dns_gpu`) is FP32 fields with FP64 reductions, hipFFT/cuFFT, a fused
RHS + RK-stage kernel; it mirrors the CPU API and is parity-tested. No
GPU-runtime header leaks into a public header. Throughput: 512³ ~86 ms/step
on an RX 7900 XTX (~7 GiB); compressible 256³ periodic, gfx1201:
DNS 13.6 ms/step (1234 Mcell/s), Smagorinsky 15.6, σ 16.0, 1.46 GiB.

## The two binaries

| sub | purpose | spec |
|---|---|---|
| `flow` | a body (or scene) in a uniform free stream: forces, wake, frames; or an OpenFOAM simpleFoam case (`--polymesh CASE [--fv-precision fp32\|fp64]`) | below, `docs/scene.md`, `docs/fv.md` |
| `tgv` | Taylor–Green vortex: 2D analytic check or 3D Re 1600 benchmark | `dns tgv --help` |
| `hit` | forced homogeneous isotropic turbulence, SI-configured, h = 1.5 η | `dns hit --help` |
| `channel` | wall-resolved compressible channel; `--les`, `--rans` | `docs/channel.md`, `docs/rans.md` |
| `conv` | Rayleigh–Bénard convection, compressible or Boussinesq | below |
| `flame` | single-step flames: premixed against its FP64 oracle (1D; 3D at `--nz >= 8`, in synthetic inflow turbulence with `--turb`, as LES with `--les --sgs` and a thickened flame `--tfles`; or a slot Bunsen cone, `--mode bunsen`, 2D or 3D with the same turbulence/LES/TFLES flags); a 2D slot-jet diffusion flame, F + s O → P, against the Burke–Schumann sheet (`--mode diffusion`); the same flame under gravity from a wick, a 2D candle (`--mode candle`), with an immersed wick, fuel from its surface and soot (`--mode candle --wick`); a closed 2D combustion chamber with a nozzle jet (`--mode chamber`) | `docs/chem.md` |
| `bench` | step time, Mcell/s and the VRAM this process allocates | `dns bench --help` |
| `prop` | propeller thrust curve by BEMT or lifting line, host-only | below |
| `mesh` | read an OpenFOAM polyMesh into a FaceMesh, report counts and checkMesh-equivalent quality, host-only | `docs/mesh.md` |

Common to all: `--quiet`, `--dry-run`, `--help`, `--out` (except `mesh`), and
(except `prop`, `mesh`) `--backend`, `--seed`. `--config FILE.toml` supplies any flag from a TOML file
(`n = 512` is `--n 512`); explicit CLI flags override it.

Parsing is strict: an unknown flag, a missing or non-numeric value, or a flag
that does not apply to the resolved case is a hard error, never ignored. Exit
codes: `0` ok, `1` CLI, `2` config, `3` input, `4` output, `5` no GPU,
`6` resource, `7` numerical failure, `8` non-convergence, `77` skipped.
`--dry-run` validates and prints the resolved case (including the
DNS/LES/RANS label) without touching GPU or filesystem — use it before every
real run.

**`dns-ui`** — scene editor, run cockpit and live viewer
(`-DDNS_WITH_UI=ON`, fetches SDL3 + HarfBuzz; needs Vulkan). It solves
nothing: it links `dns_core` only, never `dns_gpu`, and starts on a machine
without a GPU. Edits drive the solver's own `validate()` /
`buildCaseGeometry()` / `sanitizeCut()`, so the grid, cut cells and the
`(D/h)/√Re` resolution verdict it shows are what the solver would get; RUN
spawns `dns` as a child process and watches it by mapping the shared-memory
segment the run publishes (`--emit shm`). `build/dns-ui --scene FILE.toml`
opens a scene; `build/dns-ui --check` is the headless self-check (ctest
`ui_check`). Full description: `docs/ui.md`.

## Quick start

```sh
# incompressible cylinder, Re 100, free-slip tunnel, PNG frames
build/dns flow --body cylinder --re 100 --n 256 --emit stats --emit png

# body-fitted O-grid cylinder, compressible M 0.2 (the val_ogrid_cyl case)
build/dns flow --physics compressible --body cylinder --re 100 --mach 0.2 \
               --ogrid --n 192 --ny 96 --ogrid-far 20

# steady RANS channel, Re_tau 180, implicit Newton (~0.1 s)
build/dns channel --rans sa-neg --steady 1e-4 --steady-scheme implicit \
                  --n 4 --nz 1 --ny 64 --kick 0

# 2D Rayleigh-Benard, Ra 5e4
build/dns conv --rayleigh 5e4 --n 96 --ny 48 --nz 1

# 1D premixed flame, 15 cells per thermal thickness, M 0.01 (~11 s)
build/dns flame --inflow-speed 0.98 --tt 20 --emit stats

# 3D premixed flame in synthetic inflow turbulence (u'/S_L 4, l_t 2.5 delta_th,
# frozen hash-seeded box swept in at u_in), the fresh mass held by the inflow
# controller; 224 x 90 x 90 at 6 cells per delta_th, ~95 s (docs/chem.md, 3D;
# NOT statistically stationary at tt 20, issues.md T4); window statistics:
# tools/chem/turb_stats.py PREFIX --t0 10 --t1 20 --sl 1.025126 --mach 0.02
build/dns flame --nz 90 --ny 90 --cells-per-delta 6 --upstream 16 \
  --downstream 16 --turb 4 --hold-flame --inflow-speed 1.2 --mach 0.02 --tt 20

# The same flame as LES + thickened flame (TFLES F 3 at 2 cells per delta_th,
# 96 x 30 x 30, ~4 s; sigma-model SGS, species SGS flux Sc_t = Pr_t, dynamic
# sensor, Charlette efficiency). Labelled LES everywhere. It reads S_T/S_L
# 1.04 against the DNS's 1.70: thickening suppresses the resolved wrinkling
# and Charlette's efficiency is ~1 at these Re_Delta (docs/chem.md, issues.md
# L1). --tfles alone (any nz, also --mode bunsen) thickens a laminar flame.
build/dns flame --nz 30 --ny 30 --cells-per-delta 2 --tfles 3 --les 1.35 \
  --sgs sigma --nscbc-sink 5 --upstream 16 --downstream 16 --turb 4 \
  --hold-flame --inflow-speed 1.2 --mach 0.02 --tt 20

# 2D slot Bunsen cone on a burnt coflow, U_jet 5 S_L, M 0.02 (~20 s); video:
# tools/chem/fire_video.py PREFIX --out cone.mp4 (needs --emit bin)
build/dns flame --mode bunsen --mach 0.02 --tt 30 --emit stats

# 3D turbulent slot Bunsen, LES sigma + TFLES F 3 + Charlette, synthetic jet
# turbulence u'/S_L 3 (4.74 M cells, ~10 min GPU, 41 GB of volume frames);
# measured span-mean S_T/S_L 1.59, tip 57 delta_th (docs/chem.md). Video:
# tools/chem/turb_render.py PREFIX --out turb_flame.mp4 (runs/flames/turb3d/render.sh)
build/dns flame --mode bunsen --jet-speed 8 --slot-width 24 --coflow-width 16 \
  --mach 0.01 --les 1.35 --sgs sigma --turb 3 --turb-length 6 \
  --cells-per-delta 2.5 --tfles 3 --nz 120 --length 100 --slot-edge 0.5 \
  --tt 36 --frame-start 18 --fdt 0.04 --emit stats --emit volume --emit slices

# 2D slot-jet diffusion flame (fuel slot, oxidizer coflow), steady by tt ~60
# (~15 s for tt 80); same video tool
build/dns flame --mode diffusion --tt 80 --emit stats

# 2D candle: the diffusion flame under gravity from a low-Peclet wick, steady
# by tt ~20 (~100 s for tt 30); fire_video.py --gamma 3 --smooth 4
build/dns flame --mode candle --tt 30 --emit stats

# 2D WICK candle: an immersed isothermal wick + candle body (ghost wall), fuel
# vapour from the wick surface, soot (one-equation, appearance-tuned) formed
# inside and burned out at the tip; steady by tt ~5 (~45 s for tt 10); the
# physically coloured render (soot Planck x CIE -> sRGB, CH* blue)
build/dns flame --mode candle --wick --tt 10 --emit stats --emit bin --fdt 0.05 --out w
python3 tools/chem/candle_render.py w --out candle.mp4 --png candle.png

# closed 2D combustion chamber: premixed injector, wedge flame holder,
# nozzle jet into a cold coflow; 4 cells per thermal thickness (~37 s), video
build/dns flame --mode chamber --tt 80 --emit stats --emit bin --fdt 0.5 --out c
python3 tools/chem/chamber_video.py c --out chamber.mp4 --png chamber.png

# a multi-object scene
build/dns flow --scene examples/tandem_cylinders.toml
```

Append `--dry-run` to any of them first. More, with wall times and expected
numbers: `docs/scenarios.md`. TOML run configs: `examples/configs/`.

## Code layout

```
cli/config (parse + validate, pure)  [+ dns/scene, dns/remap, dns/sgs, dns/rans]
  → bodies/geometry (sdf + cut sampling)  [+ dns/curvmesh, dns/framemap, dns/walldist,
                                           dns/polymesh, dns/facemesh, dns/la/graph,
                                           dns/fvflux.hpp, dns/fvgrad, dns/fvref (FP64 FV oracle),
                                           dns/la/{ell,lines,structured,amg},
                                           dns/foamdict, dns/fvcase, dns/fvsimple (OpenFOAM case layer)]
    → gpu/{tunnel,bodyforce,les,rans,cutpoisson,cutflux,derived,comp_channel,comp_conv,boussinesq}
      [+ gpu/la/{blockell,smooth,krylov,amg}: the sparse solver library (P3, P5)]
      [+ gpu/fv_simple: incompressible FV SIMPLE on gpu/la (P5b)]
      → gpu/runner_{incomp,comp,channel,conv,rb,ogrid,fv}   (the time / iteration loops)
        → output (RunOutput, the one artefact sink; the FV path writes
          OpenFOAM time directories through dns/foamio)
```

`apps/dns_main.cpp` is **Parse → Validate → Acquire → Run**: phases 1–2 touch
neither filesystem nor GPU; phase 3 creates every output before the first GPU
allocation. `std::expected` for anything a user can cause, `assert` only for
what the validator makes unreachable.

## Solvers

**Box units**: `uInf = 1`, `rho_inf = 1`; Re and Mach are the inputs. Grid
conventions differ by physics on purpose: incompressible `L = 2π`, `h = L/n`,
`D = L/dr`; compressible `D = π/4`, box `dr·D` in y (`(n/ny)·dr·D` in x), D
spans `ny/dr` cells, `dr ≥ 6`.

**2D is `nz == 1`, not a code path.** The periodic wrap makes every z-stencil
term cancel identically and w stays exactly 0. Only the solvers special-case
the degenerate axis (`hipfftPlan2d`, skipped DCT pass, open box faces). Only z
may be degenerate; `hz = 1` is validator-enforced; NeumannBox needs n ≥ 4 per
confined axis. The `*2d` parity tests guard this.

**Incompressible.** Poisson = exact spectral inversion of the discrete
Laplacian: FFT when periodic, per-axis DCT-II (Makhoul same-length FFT) in a
confined box, FFT/DCT-preconditioned CG on the fluid subspace with cut cells.
The pressure BC follows `--far`. `--steady TOL` marches until
`max|du/dt|·D/U² < TOL` (only below the shedding onset); `--seq` converges on a
grid ladder, changing only the initial guess.

**Incompressible FV on an OpenFOAM mesh (`dns flow --polymesh CASE`,
`docs/fv.md`).** OpenFOAM v2606 simpleFoam (SIMPLE / SIMPLEC, laminar, MRF)
on the IDENTICAL polyMesh on the GPU (FP32 by default, `--fv-precision fp64`
the parity mode; below), the linear solves on `gpu/la`
(U: BiCGStab + multicolour GS; p: AMG-PCG) with OpenFOAM's normalised-residual
stop. The case is the OpenFOAM case directory (0/, system/, constant/);
anything the solver does not reproduce exactly is rejected by name. Fields
match simpleFoam to 1e-13 iteration by iteration on blockMesh cases with
tight solves; the snappy sphere Re 100 cd to 9e-13; the DTMB 4119 prop + MRF
thrust / torque to 5e-6 / 2e-9 (G1, 989k cells) in 37.7 s vs 296.7 s for
simpleFoam np 12. MRF is OpenFOAM's `MRFZoneList` (several zones, Coriolis,
`makeRelative` incl. the double frame flux on a zone/zone face, rotating
walls, `nonRotatingPatches`, forces moment about `CofR`; label `[MRF frozen
rotor]`): the laminar rotor2 L1/L2 (115k / 145k cells) CT and CQ equal
simpleFoam + MRF to 1e-11 after 1500 iterations, settling in the same
iteration count; wall 13.2 / 15.8 s vs 32 / 44 s for simpleFoam np 6
(2.4–2.8×; these small meshes are launch-bound). `limited corrected
psi` laplacian / snGrad (the snappy-mesh default) is reproduced. RAS
(`simulationType RAS`): OpenFOAM's kOmegaSST and SpalartAllmaras with
omegaWallFunction (all blenders), nutLowRe / nutUSpalding / nutk wall
functions, kqR, bounded upwind / linearUpwind / limitedLinear scalar
transport, `bound()`, and the wall distance as OpenFOAM computes it (meshWave
in its serial FaceCellWave order, or exactDistance — the knob is in the
label): blockMesh cases equal simpleFoam to ≤ 1.6e-10 iteration by
iteration; the TMR-like flat plate cf within 2.6e-4; rotor2 L1 re-meshed with
snappy prism layers + kOmegaSST + MRF: CT 2.3e-8, CQ 1.3e-9 of serial
simpleFoam, 12.1 s vs 39 s (np 6, `--bench`). Final prop table vs simpleFoam
np 12 (cold, both `--bench`; docs/fv.md "Final prop parity"): rotor2 L1 / L2
laminar MRF 13.2 / 15.8 s vs 41 / 55 s, rotor2 L1 SST + MRF 12.1 s vs 42 s,
DTMB F1 laminar MRF (708k) 40.6 s vs 345 s, CT and CQ equal to ≤ 2.7e-9
(laminar) / 2e-6 (SST: foam's np 12 decomposition; 2.3e-8 of serial foam).
Those parity numbers are the FP64 parity mode (`--fv-precision fp64`, bit for
bit the performance-round solver). The DEFAULT is `--fv-precision fp32`
(docs/fv.md "Precision"): fields, matrices, Krylov and AMG in float, every
reduction in FP64, each solve stopped at max(fvSolution tolerance, a measured
FP32 floor: U / p 1e-6, turbulence 1e-7; logged `floor-limited`): rotor2 L1 /
L2 / L1 SST / F1 2.63 / 2.30 / 3.79 / 8.18 ms/it (FP64 3.98 / 4.99 / 7.19 /
21.88), 4.2 / 3.8 / 3.4 / 11.3 s whole run (8.1× / 12.8× / 12.9× / 29.4×
simpleFoam np 6 per iteration), CT within 1.7e-6 of FP64 laminar and 1.25e-5
SST (inside its 3e-5 end jitter), CQ ≤ 8.8e-7, same settle iteration.

**Compressible.** 3D throughout. `stableDt` is recomputed every
`--dt-recompute` steps with a 0.95 margin. The diffusive limit is thermal:
`0.25 h² ρ_min / (μ_max · max(1, γ/Pr))` — the energy equation diffuses at
`γν/Pr`, 1.94× the momentum ν for air. Quiet start: `--ramp T` starts at rest
and raises the freestream by uniform Galilean boosts (start-up |p′|/p∞ 0.56%
vs 12.7% impulsive); the quiet start is too symmetric to shed, so `--kick`
adds a solenoidal (acoustically silent) vortex trip one D behind the body.
`--steady TOL` on the compressible paths is pseudo-time stopped on the
integrals, not the residual (`docs/rans.md`; `--steady-sample N` = iterations
per stop sample, default 200); `--compensated` runs a time march
on the Kahan-compensated state (use it for any dt-convergence claim).

**Temperature.** Air, calorically perfect, γ = 1.4 fixed. `dns::Transport`
(`dns/transport.hpp`) is the one gas object (γ, R, Pr, viscosity law) and the
only route by which μ and κ reach a kernel. Every thermal knob is off by
default and a run naming none is bit-for-bit the pre-thermal solver:
`--visc sutherland` (evaluated once per FACE on the face-average T, so the
scheme stays exactly conservative), `--tinf K` (the one dimensional anchor,
needed only for Sutherland), `--prandtl`, `--wall adiabatic|isothermal` +
`--twall` (the BODY surface; the box y walls are adiabatic mirrors),
`--nscbc-tinflow`. Wall heat flux rides the force tally (`qw`, conduction
only, so an adiabatic wall reports exactly 0). `--temp` adds a T − T∞ frame
channel.

### DNS, LES, RANS — which runs mean what

The core scheme's only dissipation is physical ν: it is DNS iff the grid
resolves the body boundary layer (~D/√Re, ≥ ~3 cells). Beyond that the shear
layer piles enstrophy at the grid scale and the run diverges; no CFL setting
fixes zero dissipation.

- `--les CS` adds an eddy viscosity, `--sgs smag|sigma` picks it (both
  physics, curvilinear included, Δ² = vol^(2/3)). σ-model (Nicoud 2011,
  Cσ ≈ 1.35 — not a Smagorinsky Cs) vanishes in pure shear, rotation and any
  2D field, so it is the transition-capable choice and is rejected at
  `nz == 1`. Smagorinsky has no wall damping (νt/ν ≈ 0.03 Δ+² in the first
  cell): prefer σ for wall-bounded LES. `--les 0` is bit-for-bit DNS.
- LES μt takes the RANS wall image (odd at walls, even on mirror rows), so a
  wall face carries molecular stress only; every tally passes the solver's own
  μt, so forces are the budget of the solved equations (`gpu_les_wall`,
  `gpu_les_tally3d`). The incompressible `cdf` column is ν-only.
- An LES field is FILTERED: statistics depend on grid AND Cs by construction.
  Label it `LES <model> Cs=<CS>` everywhere. 2D + `--les` is regularized 2D
  Euler, qualitative only.
- RANS (`dns channel --rans`, `dns flow --physics compressible --rans`) is an
  ensemble-mean model output, never a grounder. Labels (`RANS SA-neg -- steady
  (...)` / `-- URANS (time-accurate)`, plus geometry suffixes) are printed by
  `--dry-run` and carried into every artefact. Spec: `docs/rans.md`.

## Far-field boundaries (`dns flow --far`)

| mode | physics | what it is |
|---|---|---|
| `tunnel` | incompressible default | Dirichlet inflow, convective outlet, FREE-SLIP lateral walls (symmetry planes) |
| `tunnel --open` | incompressible | lateral normal velocity zero-gradient, its flux enters the projection: sponge-free open tunnel |
| `fringe` | incompressible | fully periodic FFT pressure + cosine fringe relaxing toward the freestream |
| `nscbc` | compressible default | Poinsot–Lele characteristic faces as an RHS overwrite (`docs/nscbc.md`) |
| `sponge` | compressible | freestream sponge band + free-slip walls: an explicit escape hatch, never a default |
| `periodic` | both | no far field |

- **Tunnel walls are free-slip**, never Dirichlet `u = uInf` (that clamps
  sheared flow and paints a vortex sheet on the wall). Free-slip walls make
  the run a periodic array of bodies: the confinement excess over unbounded
  literature grows with Re (cd +26–29%, St +12–30% at dr 8), which is why the
  validation bands are repo-calibrated (`docs/validation.md`). Fringe roughly
  halves that excess (Re 60: cd 1.654 vs 1.803 walled, unbounded 1.42).
- **Open faces leave the normal velocity free** — relaxing v → 0 or u → uInf
  laterally is a wall in disguise. In the open tunnel the lateral faces are
  slaved once per stage BEFORE projection; the post-projection refresh must not
  re-slave them (that broke div u ~150×). The freestream is an exact fixed
  point of the full BC → div → CG → gradient cycle.
- **NSCBC** (`gpu/comp_nscbc.*`): outflow needs the Yoo–Im transverse term
  (`--nscbc-beta`, 0.5; ≥ 1 rejected); inflow relax stays stiff
  (`--nscbc-etain` 100) wherever there is a freestream to sag — a soft inflow
  sagged to 0.79 uInf and biased cd 30% low — but with no freestream (a static
  rotor) etaIn is worth a factor 280 in kept pulse energy. `--open` makes the
  laterals characteristic too (six-face, corners summed per axis).
  `--nscbc-abc adaptive|modal` reads the incidence angle off the face instead
  of the constant β; `modal` needs free-slip laterals on a uniform mesh.
  `--nscbc-sink N` (scale-selective D4 damping in flux form, exactly zero on
  linear fields, no freestream reference) is ON by default at 16 wherever it is
  defined (uniform mesh); `--nscbc-sink 0` is the bare boundary bit-for-bit.
  `--nscbc-strip N` is a freestream sponge strip inside the NSCBC rows.
  `--far nscbc` alone is accepted only to Re 800; beyond, the validator wants
  the sink or a strip of ~ny/64. Measurements behind each knob:
  `docs/nscbc.md`.
- **Sponge** relaxes the freestream in a band that eats the near field. With
  `--open` it relaxes ρ/p only (velocity free). Its residual open-lateral
  blockage is topological (periodic images through the band): shrink it with
  domain size, not BC tweaks.
- **`--pot`** (compressible `--far nscbc`, incompressible `--far tunnel
  --open`): far-field targets = freestream + wake source (`Q = F_drag/(ρ∞u∞)`,
  EMA over 2D/u∞, fed only past the ramp) + displacement doublet, Bernoulli
  pressure. `Q = 0 ∧ a = 0` or the flag off is bit-identical. Measured,
  cylinder Re 100, D = 32 cells: compressible 8D box cd 1.681 → 1.365 with
  `--pot` (plain 32D box 1.377); incompressible 8D 1.736 → 1.409 vs 32D truth
  1.403, St exact. Lift amplitude stays box-sensitive. Only 2D is validated.
  The incompressible correction is its own kernel `kPotInlet` after
  `kBcTunnel` (folding it in moved pinned bits).

## Walls and bodies

**Bodies.** `--body cylinder|sphere|cube|lens|fence|floor|none|FILE` or
`--scene FILE.toml` (`docs/scene.md`, `examples/*.toml`; per-object force
columns in `<out>_forces.csv`). `cube` is a cube whose square cross-section is
rotated 45° about z (D = the diagonal frontal width, side D/√2); at `nz == 1`
it is the z-invariant prism.

**Incompressible ghost-fill wall** (`gpu/ghostfill.*`). Per stage the
DIFFUSION term reads a ghost-filled copy (`q_ghost = (sdf/d)·q(image)`, the
image out along the sdf normal) — no-slip at the true sub-cell surface.
Convection keeps the masked zeros: mirror values in the skew form inject
energy, so the ghost view must never reach convection or the state. Ghost faces
have zero aperture. The image is read by a clean trilinear cell (it steps out
along the normal until no corner is closed); imposed-field friction is then
first order (−4.5/−2.4/−1.3% at R/h 16/32/64), the solver's own steady
friction −0.17/−0.048/−0.018% (`gpu_tally_couette_torque`). Unresolved spots
take a capped fallback (`fallbackCount()`, 0 on resolved convex bodies).

**Cut cells** (`cutcell.*`, `sampleCut()` in `geometry.*`): λ + exact face
apertures, `sanitizeCut`, pressure by preconditioned CG on the fluid subspace.
`--cut-flux` (incompressible only) weights convective and viscous momentum
fluxes by the apertures over λ-floored volumes, staying exactly skew-symmetric;
interior cells are bit-identical by construction. The compressible variant
diverges (KG loses KE consistency under aperture weighting) and is rejected.
`--penalize ETA` is a λ-smooth Brinkman decay; it HALVES cdf — never use it
when friction matters.

**Upstream ripples** near bodies are trapped 6–12h dispersive standing waves of
the central scheme, continuously forced by the SUB-CELL boundary layer — not
staircase noise, not a start transient. Measured: `--cut-flux` −20%,
`--penalize` −15%, quiet start ~0%. The only cure is resolving the layer
(D/√Re ≥ 3 cells); no filter or limiter is applied anywhere.

**Compressible ghost-mirror wall** (`gpu/comp_ghost.*`). Per-stage mirror
ghosts sampled ≥ 0.75h beyond the wall, any `Sdf`, 3D-general. Drag is the
exact momentum flux through fluid↔ghost faces, which are IMPERMEABLE (nothing
advected relative to the wall; pressure, viscous and conduction fluxes kept).
The tallied surface is closed by construction: any solid cell in a fluid
cell's flux stencil is a ghost, imaged from its fluid neighbours where its
normal image is unusable. The wall pressure image is the wall's momentum
balance (`grad p = ρ(f − a_wall)`: the zero-gradient copy at rest). Under ~2
cells of section, a face whose fluid cell sits on the other surface reads its
own side's image. The deep-solid threshold is a physical depth set by the
WIDEST local spacing (set from the narrowest, a 1.7:1 focal cell inverted cd).
Every `DeviceComp` stepper with a ghost wall must call `applyRhs` in its rhs
hook.

**Mesh bodies** (`src/dns/trimesh.*`: STL parser + glTF/GLB via
`third_party/cgltf.h`; `src/dns/meshsdf.*`: BVH closest triangle,
pseudonormal sign, `--robust` winding-number sign for leaky or overlapping
shells). Host-side, build time only; the ghost classification takes the body
AABB and skips cells provably outside the band (~20× setup). qRef comes from
the aperture-measured frontal area. `val_mesh_sphere`: mesh vs analytic sphere
cd 0.6798 / 0.6727. Orientation: world +X is the freestream; `--pitch/--yaw`
are deprecated (they rotate raw model axes, not aircraft angles) — bake the
attitude into the mesh and leave them at 0.

## Meshes (compressible)

**Curvilinear** (`src/dns/curvmesh.*`): an optional path inside `Comp` /
`DeviceComp`. `CurvMetrics::build` is the only validator (positive volumes,
invertible face Jacobians, planar layer at `nz == 1`). Finite volume: KG flux
with the exact bilinear face area vector, viscous gradients through per-face
inverse Jacobians; cells tile the box exactly, so freestream preservation and
conservation are exact (FP32 freestream drift 0.0), and the uniform mesh
reproduces the Cartesian rhs to 5e-15. Isentropic vortex: order 4.01 ratio.
LES filters at the cell's own Δ (isotropic `vol^(1/3)`, deliberately not
anisotropy-corrected). The channel's tanh stretching is `CurvMesh::channelY`.
The host phase runs on all host threads (`dns/parallel.hpp`: row blocks or an
atomic row queue, own-index writes, one sdf copy per worker), bit-identical to
serial at any thread count (`host_threads`): `CurvMetrics::build`,
`sampleCut`, the compressible body IC (one sdf call per cell) and the MRF
ghost classification. DTMB 4119 host phase: P1 (6.1M cells) 50.7 → 8.0 s, P3
(20.7M) 156.0 → 20.8 s.

**`--focal`** clusters a fine plateau box (body + wake, or `auto` from the body
AABB) with one `Focal1D` map per axis: an integrated quintic smoothstep with
compact support, so the far grid is exactly uniform, the wrap exactly C² and
the ratio exactly `--focal-ratio`. The box and blend are PHYSICAL coordinates
(`Focal1D::fromPhysical`). Validator: blend ≥ 4 cells off every face, plateau
≥ 4 fine cells, adjacent growth ≤ 1.25 (a sharp step reflects in a
dissipation-free scheme; widen `--focal-blend`), body + 3 h_fine guard inside
the plateau, analytic bodies or rotors only (a mesh AABB needs the
filesystem). Fine cells cost R× their width out of `--n`. Frames are resampled
onto a uniform PHYSICAL raster at the finest pitch (`dns/framemap.hpp`;
display only, identity map = bit-exact copy).

**`--ogrid`** makes the inner mesh boundary the body surface: no cut cells,
no staircase, forces and wall heat as exact surface integrals
(`src/dns/ogrid.*`, `gpu/ogrid_bc.*`, `gpu/runner_ogrid.*`).

| axis | meaning | wrap | cells |
|---|---|---|---|
| i | angular, CLOCKWISE (`θ_i = −2πi/nx`) | cyclic | `--n` |
| j | radial; cell 0 inner ghost, ny−1 outer ghost | extend | `--ny` incl. 2 ghosts |
| k | spanwise, span πD | translate | `--nz` |

θ runs clockwise because (θ, r, z) counter-clockwise is left-handed and every
volume would be negative: anything computing an angle or tangent must use that
sign or cd/cl flip. Nodes are a transfinite blend of the wall-normal ray and
the radial ray (`P = C + D(t)[(1−w)Ln n + w(F−C)]`, `w = 3t²−2t³`, tanh radial
distribution `--ogrid-beta`): the first cell leaves along the normal, the
outer boundary is an exact circle, a circular body gives the exact polar
annulus. Cell volumes sum to the POLYGONAL annulus, not π(r²−r²). Scope:
z-invariant sections extruded in z, and `--body sphere` revolved about x
(`--nz 1` = axisymmetric wedge, `--nz N` = full revolution, tripped by default
via `--ogrid-trip`; `docs/rans.md`). The far row is its own characteristic
condition: `--far`, every `--nscbc-*`, `--pot`, `--focal`, `--stretch`,
`--open`, `--kick` and friends are rejected. Measured Re 100 / Re 40:
`docs/validation.md` "O-grid cylinder".

**FaceMesh** (`src/dns/{polymesh,facemesh}.*`, `src/dns/la/graph.*`,
`docs/mesh.md`; masterplan_meshsolver phase 1, host only — no solver runs on
it yet): the one face-based mesh of the coming FV path. Built from an
OpenFOAM polyMesh (`readPolyMesh`: ASCII/binary, label 32/64; snappy,
blockMesh, extrudeMesh, refineMesh output; everything else rejected by name)
or from any structured `CurvMesh` (`fromCurvMesh` writes a PolyMesh and
routes through the same builder). Geometry is OpenFOAM v2606's own
definition: the checked-in snappy cylinder and the 831k-cell DTMB `mB` mesh
reproduce checkMesh's volume, non-orthogonality, skewness and aspect ratio
to every printed digit (`test_facemesh`); the uniform box is exact, the 2D
O-grid matches `CurvMetrics` to 4e-16 (S) / 2.4e-15 (V). Cyclic pairs merge
into internal faces, `empty` one-cell extrusions are the 2D restriction
(S_e, d_e exactly 0), cells in Source / Morton / colour-major order with
bitwise-identical geometry, threaded and byte-identical at any thread count.
`la::CellGraph` is the matrix pattern with per-face slots. `dns mesh
--polymesh DIR` prints the report (mB: 0.7 s).

**FV face physics + CPU FP64 oracle** (`src/dns/fvflux.hpp`,
`src/dns/fvgrad.*`, `src/dns/fvref.*`; phase P2, host only — the GPU FV
path is P4): the Kennedy–Gruber mean flux and face viscous terms restated
for an arbitrary face (DNS_HD, templated on double / float / `Dual<N>`),
least-squares (pinned) or Green–Gauss cell gradients, the corrected face
gradient (`AlongD`, pinned; α-damping available), every boundary kind as a
face flux (wall adiabatic/isothermal with `u_w(x)`, slip, inflow, outflow,
characteristic far field), the Σ-form dt and SSP-RK3. On a uniform periodic
box it equals `dns::Comp` to 6e-15; second order on skewed meshes (MMS
2.00); an adiabatic wall leaks exactly 0.0; threaded and byte-identical at
any thread count (`docs/mesh.md`, `test_fvref`).

**GPU sparse solver library** (`src/dns/gpu/la/`, `src/dns/la/{ell,lines,
structured,amg}`; phases P3 + P5; its first customer is the incompressible FV
path `dns flow --polymesh`, `docs/fv.md`):
sliced-ELL block matrices on a `CellGraph` for B ∈ {1, 3, 5, 6, 7, 9, 11}
(dense or 5 × 5 + scalar off-diagonals; float or double), SpMV, block Jacobi,
multicolour block GS, batched block-tridiagonal line solves, FGMRES / PCG
(classic or flexible) / BiCGStab whose inner products are batched,
fixed-order FP64 reductions (byte-identical reruns), and aggregation AMG
(Notay pairwise aggregation on the host once per mesh, coarse values
re-gathered on the device in fixed order, V / scaled-V / K-cycle, B = 1 and
every block shape). Own kernels, no vendor library, HIP and the CUDA shim
alike. B5 SpMV 1.16 ns/cell at 1M cells; 8192 × 128 B5 lines 0.64 ms;
AMG-PCG on Poisson 64² → 1024²: 12 / 16 / 15 iterations; on the 989k-cell
prop mesh OpenFOAM's own pressure matrix (fields equal to 1e-14): 6
iterations (~36 ms) to OpenFOAM's normalised residual at relTol 0.01 vs GAMG
8 V-cycles / 0.70 s serial
(`docs/la.md`, `test_la`, `test_amg`, `test_gpu_la`, `test_gpu_amg`).

## Rotors (`dns flow --rotor-mach`)

`--rotor-mach M` spins the immersed body about a shaft through the placement
centre (`--rotor-axis`, z only at `nz == 1`) so its NOMINAL tip `D/2` runs at
Mach M. It is the one switch: 0 is bit-for-bit the static solver, and every
other rotor flag is rejected without it. Compressible `flow` only.

- **Moving wall**: the ghost image about `u_wall = ω × r`,
  `u_ghost = (1+a) u_wall − a u_sample` — the same image the isothermal wall
  uses; without it a blade is a rotating hole. `comp_ghost` owns no motion
  model (`GhostWallMotion` callable).
- **Moving geometry** (`--rotor-frame moving`, default): occupancy is
  precomputed at `--rotor-bins` azimuths (default `2πR/h_fine`) in host memory
  and replayed into one set of device buffers — a bin change is a copy, never
  an allocation. Azimuth is COMPUTED (`θ = ωt`, integer bin mod bins), never
  accumulated, so revolutions close exactly. Geometry is fixed per step, not
  per stage (the tip moves ~0.1 cell per step). A freed cell enters on the
  wall profile; the swap ledger (`ct_c cq_c ct_sw cq_sw mdot_sw`) reports what
  the swaps carry.
- **Stats** add `thrust, torque, ct, cq` (`ρn²D⁴`, `ρn²D⁵`), torque on the same
  FP64 face values as drag.
- **Resolution is measured**: a blade under 4 fine cells thick has its two
  surfaces reading through each other; the run reports the deep-solid count
  and warns at 0. `--blade-thicken C` dilates by C fine cells (exact Minkowski
  sum) — it CHANGES THE BODY and must be reported with every result.
- **`--thin-seal`** (compressible immersed ghost wall, static or `--rotor-frame
  mrf`; off by default): a section thinner than the staircase — only a cell
  CENTRE in solid makes a ghost — lets flow through between two fluid cells.
  With the flag, every fluid/fluid face whose centre-to-centre segment
  crosses the body (sphere-traced on the sdf) is a wall face of BOTH cells,
  each imaged about its own surface: a zero-thickness sheet. The body is not
  changed; the run prints the sealed count and labels `[thin sections sealed:
  sub-cell sheets imaged as zero-thickness walls]`. 0 sealed faces is
  bit-identical to the flag off. Rejected with `--ogrid`, `--rans`,
  incompressible and `--rotor-frame moving`. Measured (DTMB 4119 P1, D/h 114,
  `plans/verify/impl_propperf.md`): 235 faces sealed; tip bin vs OpenFOAM G1
  −74.2% → −0.5%, but total CT −33% → **+18.0%** and CQ −1.1% → **+2.9%**
  (window 2.58–3.18 D/U, still rising): it overshoots; not a parity setting.
- **Tip Mach**: the run measures the real tip radius and stops if a body
  reaching past `D/2` exceeds the budget. Flow accelerating over a bluff
  spinning body goes sonic far below tip Mach 1 (a 3.5:1 lens at real tip 0.53
  + freestream 0.2 diverges): treat it as a budget.
- **`--rotor-frame mrf`** — frozen rotor, an APPROXIMATION: static azimuth-0
  geometry, the same moving-wall image, rotating-frame equations in a blended
  cylinder sized off the swept bounds (rejected if it comes within band + 4
  cells of a face). The transport correction comes from a corner stream
  potential, so it telescopes to zero discretely; the tally uses the same
  flux; `stableDt` uses `u_rel ± c`. Exact for axisymmetric outer conditions
  (`gpu_mrf`: spinning cylinder vs the direct moving-wall run, torque 1.8e-5,
  banded 1e-3); a crossflow rotor co-rotates its own far field. Also runs on
  the O-grid (`--ogrid --rotor-frame mrf`, z shaft, per-node stream potential;
  torque vs Cartesian direct 0.22%).
- **`--rotor-frame mrf --steady TOL`** (immersed, laminar, `--focal` only):
  the frozen rotor's fixed point in local-time-stepping pseudo-time, stopped
  on ct, cq and the axial jet momentum ∫(ρu_a − ρ∞U_a)dV; label `DNS --
  steady (...) [MRF frozen rotor: an APPROXIMATION, blade fixed at azimuth
  0] [immersed ghost wall ...]`. `--steady-sample N` sets the stop cadence
  (the stop needs ≥ 48 samples: 9600 iterations at the default 200);
  `--lts-max R` caps dt_c/dtRef. **Measured, not a speed-up**
  (`plans/verify/impl_M1.md`): the fixed point is lts-independent (rotor2 L1
  ct 0.037694 / cq −0.074022 at R = 1, 2, 4, ∞; +0.23% / −0.01% vs the time
  march's 3.0–4.5 D/U mean), but local time stepping SLOWS it (ct inside 1%
  for good after 8100 / 19600 / 23850 / 25550 iterations at R = 1 / 2 / 4 /
  ∞, vs the 8570-step time march) and nothing certifies TOL 2e-3 inside 60000
  iterations at M 0.082 (ct jitter 1.8e-3, jet 8e-3). DTMB P1 (M 0.109,
  ratio 10): ct sample sd 65% unlimited, 7% at R = 1 after 13500 iterations.
- **`--precond K`** (any explicit laminar `--steady`; label adds `[low-Mach
  preconditioned pseudo-time, K=...]`): Weiss–Smith preconditioning, the
  pressure rate of each cell's total residual scaled by ε = U_r²/c², U_r =
  max(K U∞, |u_rel|, ν/h) ≤ c, local dt on u′ + c′. Same fixed point
  (`plans/verify/impl_M2.md`): rotor2 L1 `--precond 2` reaches ct 0.037697 /
  cq −0.074022, ct inside 1e-3 of it after 15500 iterations (unlimited LTS
  59900, `--lts-max 1` 29700), cq inside 3e-4 after 10750 (52500 / 29500); ct
  jitter at 30000 it 3.2e-4 (5.7e-3 / 1.9e-3), +20% per iteration. Still
  exit 8 (the jet monitor) and still slower than the 8570-step time march.
  `--precond 1` diverges on rotor2 (U_r floor under the tip speed); K 3 is
  stable but jitters more. DTMB P1 K 2: ct sd 7% (unlimited 65%), window
  mean flat at −0.0168. No gain on the O-grid cylinder (diffusive-bound).
- **Test propeller**: `python3 tools/propeller.py prop.stl --blades 3
  --pitch 0.8` writes a parametric STL (shaft +x) and prints the `--scale`
  to pass (auto-fit uses the static extent, which for odd blade counts
  understates the swept diameter). Pass `--robust` (hub and blades are
  overlapping shells). The default thickness needs ~370 cells across D:
  raise `--thickness` or use `--blade-thicken`, and say which.
- Parity against OpenFOAM: `docs/scenarios.md` §7 (rotor2).

## Convection (`dns conv`)

A layer between no-slip isothermal plates (hot floor, cold lid), gravity in
−y, lateral faces `--sides periodic|closed`. Box units: H = 1, T_ref = 1,
R = ρ_ref = 1.

**`--physics compressible`** (default, `gpu/comp_conv` + `runner_conv`).
Buoyancy is ρ(T)·g through the EOS — one source term (`addGravity`) on the
RHS hook; there is no Boussinesq approximation in this branch. The IC is the
hydrostatic conduction state. `--mach` SETS gravity (`g = M²γ/dtemp`). Gravity
carries its adiabatic gradient g/c_p, so only the superadiabatic excess drives
convection: `--rayleigh` is on that excess (critical ≈ 1708), the reported
Nusselt numbers on the FULL drop (conduction = exactly 1). The adiabatic
fraction `M²(γ−1)/dtemp²` is 10% at the defaults; a subadiabatic layer is
rejected by name. Measured 2D, Pr 0.72, coarse (`gpu_conv`): Ra 1000 → Nu
1.00 at rest, 2e4 → 3.00, 5e4 → 3.81; wall and interior Nu agree to < 5% (the
heat balance closing).

**`--physics incompressible`** (Boussinesq, `gpu/boussinesq` + `runner_rb`):
divergence-free, no acoustic dt (~2k steps where the compressible case takes
87k), free-fall scaling (H = ΔT = gα = 1, ν = √(Pr/Ra), t_ff = 1). Sound-speed
knobs (`--mach`, `--dtemp`, `--visc sutherland`, `--tinf`) are rejected.
Pressure: `PoissonBc::PeriodicXZWallY` (FFT in x/z, exact tridiagonal in y,
`gpu_poisson_wally`). `rbIC`'s `cos(πy/H)` factor breaks the exact up-down
symmetry; without it the two wall Nusselt numbers agree to six digits forever
(`gpu_rb` checks the means agree AND the instants differ).

**Cross-validation** — the two branches share the case derivation and the
output sink and no physics: Ra 5e4 2D, compressible Nu 3.79 vs Boussinesq 3.97
(4.6%, banded 15% in `gpu_rb`). If you touch buoyancy, a wall or temperature
transport, that agreement tells you which branch broke.

**`--sides closed`**: a sealed cavity — x (and z when `nz > 1`) become no-slip
ADIABATIC walls on the plates' own ghost ring; the grid gains two cells per
closed axis, `--n`/`--nz` stay fluid cells; the Boussinesq pressure becomes
`NeumannBox`. A different experiment: rolls fixed by `--aspect`, Nu below the
periodic value, neither 1708 nor Ra–Nu scaling applies. Subcritical stays at
rest (Nu 1.00); the heat balance closes to < 5%. Measured, Ra 5e4, aspect 2:

| case | periodic | closed |
| --- | --- | --- |
| 2D Boussinesq (96×48) | Nu 3.97 | 3.65 |
| 2D compressible (96×48) | Nu 3.81 | 3.38 |
| 3D Boussinesq (32×24×32) | Nu 4.20 | 3.58 |

**`--hot-patch F`** heats a centred fraction of the floor (both branches;
surround at lid temperature; in 3D a side fraction, area F²). F = 1 is
bit-for-bit the flagless run. Below 1 it is not classical RB: `nuBot` is a net
floor-area flux, not a Nusselt number, and the temperature map centres on the
area-mean wall temperature.

## `dns prop`

Host-only thrust curves (links `dns_prop`, never a GPU, never exit 5): BEMT
(Ning residual, Prandtl tip + hub loss, Buhl branch) or a lifting line
(realigned helical wake, image hub, Newton → MINPACK hybrd → pseudo-transient
continuation). Inputs: strict CSVs from `tools/propeller.py` (planform) and
`tools/polars.py` (polar), or `--polar thin`. A non-converged lifting-line J is
flagged (`converged` 0), not an error. Matches the Python reference to ≤ 3e-15
(BEMT) / ≤ 3e-14 (lifting line) at ~50× the speed (`test_prop`).
`--method liftline --selftest` runs the Biot–Savart / actuator-disk / Betz
checks.

## Output

`--emit` is repeatable (default `stats,bin`); each kind's files are listed in
`dns flow --help`. Formats: `docs/viewer-format.md`.

- `stats` — `<out>_stats.csv`. The schema depends on subcommand, physics and
  options: **parse by column name from the header row, never by index.**
- `shm` — live frames in POSIX shared memory (`/dev/shm/dns_<out>`) under a
  seqlock: writer bumps odd, copies, bumps even; a reader keeps a copy only
  between two equal even reads. The solver never blocks on a viewer. 3D
  volumes are capped at 512 MB, over budget planes only. This is what
  `dns-ui` attaches to.
- `png` — `png_<out>/{vort,speed}_%05d.png` (centre z-plane in 3D), row 0 =
  image bottom; display knobs `--vort-max`, `--vort-map asinh`,
  `--speed-map delta`, `--vort-signed`, `--png-down`, `--temp`.
- `web` — `web_<out>/` (atomic `manifest.json` + u16 frames; 3D: 9 planes
  listed in `planes.json`). View with `python3 -m http.server <port>` at the
  repo root and open `site/live.html?run=../build/web_<out>` (live tail,
  scrubbable). Ports 8000 and 8080 are taken on this machine.
- `bin`, `volume`, `slices`, `vtk`, `mesh` / `surface` (O-grid and immersed
  wall tables), `none`.

The moving-rotor mask reaches the PNG and shm frames; `<prefix>_mask.bin` and
the web static mask hold the azimuth-0 body (`docs/viewer-format.md`).

## Validation

Tiers, labels, tolerances, the quarantine and anchor regeneration:
`docs/validation.md`. Scalar anchors: `data/validation/anchors.csv`
(commented-out rows are deferred, their checks skip with rc 77).

- Plain `ctest` runs verification (parity, order, conservation) plus Tier A.
  `gpu_runners` is the unquarantined smoke test of the time loops (exit 0,
  finite coefficients, healthy projection, stats rows; no literature band).
- The nine literature-band `val_*` tests are quarantined behind
  `-DDNS_VALIDATION_BASELINE=ON` by decision, not failure: re-measure before
  citing a band. `val_spindown` is the green canary.
- TGV convergence to the spectral dissipation peak is oscillatory, not
  monotone (64³ −8.1%, 128³ +5.6%, 256³ −4.3%, 512³ +1.2%; HiOCFD peak
  0.012858 at t = 8.97).

## Documentation

| file | covers |
|---|---|
| `issues.md` | what is broken, unproven or undecided at this commit |
| `CLAUDE.md` | working notes: invariants that must not break, known gotchas |
| `docs/scenarios.md` | cheap runnable recipes: exact argv, wall time, label, measured numbers |
| `docs/validation.md` | validation tiers, per-test tables, tolerance rationale, anchor regeneration |
| `docs/nscbc.md` | NSCBC/LODI design and every boundary absorption measurement |
| `docs/channel.md` | `dns channel`: units, mesh, walls, forcing, statistics, RANS and LES |
| `docs/rans.md` | SA-neg / SST: equations, discretization, steady stops, implicit and NK, O-grid, wedge, full revolution, immersed RANS |
| `docs/scene.md` | the `--scene` TOML format and per-object integrals |
| `docs/ui.md` | `dns-ui`: editor, run cockpit, live viewer |
| `docs/viewer-format.md` | web frame format, run `.json`, `_surface.csv`, `--emit mesh` |
| `docs/jobs.md` | `simjobd` / `simjob`: the job server every GPU/CPU run goes through |
| `docs/mesh.md` | FaceMesh: the polyMesh subset accepted and rejected, the geometry definition, cyclic/2D/ordering rules, `CellGraph`, `dns mesh`, test meshes; the FV face physics (flux, gradients, boundary kinds, Σ-form dt) and its FP64 oracle `FvRef` with the measured gates |
| `docs/fv.md` | the incompressible FV path (`dns flow --polymesh`): simpleFoam's SIMPLE/SIMPLEC + MRF + RAS (kOmegaSST, SpalartAllmaras, wall functions, meshWave / exact wall distance) verbatim, BCs and schemes accepted, constraint-patch semantics, OpenFOAM's solver stop, outputs, prism-layer meshing, the IF1/IF2/IF3/IF-T/BL parity and wall-time numbers |
| `docs/la.md` | `gpu/la`: block shapes, the sliced-ELL layout, smoothers, line sets, Krylov solvers and batched FP64 dots, the FP32-iterate floor, the measured LA1/LA2/LA-perf gates |
| `docs/chem.md` | `dns flame`: model C0, the phase-plane oracle vs Cantera, species scheme, hook order, measured S_L; the 2D slot Bunsen (per-row inflow, pilot anchoring, measured cone); the slot-jet diffusion flame; the 2D candle (gravity, hydrostatic NSCBC, height vs fuel flow); the wick candle (immersed wick, fuel source, soot, the physical render); the `--mode chamber` combustion chamber and its balances; 3D premixed flames in synthetic turbulence (generator, Taylor-swept inflow, hold controller, S_T/area/I0); LES species (SGS flux, Sc_t = Pr_t) and the thickened flame (TFLES: sensor, Charlette efficiency, laminar and LES-vs-DNS measurements); the 3D turbulent slot Bunsen LES (span-mean cone, ST/area/I0, the volume renderer) |
| `src/compat/cuda/README.md` | the CUDA header shim |
| `docs/archive/` | historical plans and analyses |
