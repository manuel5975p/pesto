# dns — incompressible & compressible DNS/LES on a MAC grid (CPU + HIP)

One binary: `build/dns <flow|tgv|hit|channel|bench> [flags]`. The 15 old apps are gone
(recomposed 2026-07-20); every capability is reachable from `dns`. `dns --help`
and `dns <sub> --help` are authoritative and kept in sync with behaviour — read
them rather than guessing flags. Historical plan/analysis documents live in
`docs/archive/`.

---

## 1. Overview + numerics

Staggered MAC grid, 2nd-order central differences, skew-symmetric
(kinetic-energy-conserving) advection, SSP-RK3.

* **Incompressible**: RK3 + pressure projection per stage. Poisson = exact
  spectral FD inversion of the discrete Laplacian (modified wavenumbers).
  Periodic ⇒ FFT (FFTW / hipFFT); confined box (NeumannBox) ⇒ per-axis DCT-II
  via Makhoul same-length FFT (GPU only). Cut-cell bodies ⇒ FFT/DCT-
  preconditioned CG on the fluid subspace.
* **Compressible** (`src/dns/comp.*`, `src/dns/gpu/comp.*`): conservative
  (ρ, ρu, ρv, E), Kennedy-Gruber KE-consistent central fluxes, **no shock
  capturing** (smooth Mach only), SSP-RK3, fully explicit (no Poisson).
  ~0.2 ms/step @512², but acoustic CFL ⇒ ~5× more steps at M 0.2.

Backends:
* **CPU** (`dns_core`): serial, double precision — the correctness oracle.
* **GPU** (`dns_gpu`): HIP/ROCm (arch auto-detected; gfx1100 and gfx1201 both
  exercised), FP32 fields with FP64 reductions,
  hipFFT Poisson, fused RHS+RK-stage kernel; mirrors the CPU API 1:1 and is
  tested for parity. ~86 ms/step at 512³ on an RX 7900 XTX (~7 GiB VRAM).
  No HIP header leaks into a public header.

Box units: `uInf = 1`, `rho_inf = 1`. Re and Mach are the physical inputs.

## 2. Build and test

```sh
cmake -B build -G Ninja -DDNS_WITH_HIP=ON   # OFF for CPU-only
ninja -C build
ctest --test-dir build                      # GPU tests skip (rc 77) without a device
```

Options: `DNS_WITH_FFTW` (ON), `DNS_WITH_HIP` (OFF; `CMAKE_HIP_ARCHITECTURES`
is AUTO-DETECTED from the installed device — set it explicitly only to
cross-compile; a wrong arch shows up as a HIP `getStatFunc` assertion),
`BUILD_TESTING` (master test switch),
`DNS_HEAVY_VALIDATION` (heavy tier, ~25 min, RUN_SERIAL),
`DNS_VALIDATION_BASELINE` (quarantined literature-band `val_*`, OFF — see §15),
`DNS_WITH_UI` (OFF; builds `dns-ui`, below).
Build type: Release.

### `dns-ui` — obstacle editor and grid viewer

```sh
cmake -B build -G Ninja -DDNS_WITH_UI=ON    # fetches SDL3 3.4.0 + HarfBuzz 14.2.1
ninja -C build dns-ui
build/dns-ui                                # interactive
build/dns-ui --check                        # headless self-check (ctest: ui_check)
```

A second binary, and the only exception to the one-binary rule: it solves
nothing. It is a pre-run inspector — place bodies, choose a resolution, and see
the grid and the **cut cells** (fluid fraction λ, face apertures, post-
`sanitizeCut`) the solver would actually receive, plus the resolution verdict
`(D/h)/√Re` that decides DNS from under-resolved. It links `dns_core` only and
drives the same `validate()` / `buildCaseGeometry()` / `sanitizeCut()` the
runners do, so it cannot disagree with them; a rejected edit shows the
validator's own message and exit code. Dependencies are SDL3, Vulkan and
HarfBuzz and nothing else — no FreeType (glyph outlines come from `hb-draw`),
no widget toolkit. Full description: `docs/ui.md`.

## 3. CLI

### Subcommands

| sub | purpose |
|---|---|
| `flow` | a body in a uniform free stream: drag/lift, wake, slices |
| `tgv` | Taylor-Green vortex — 2D analytic check or 3D Re1600 benchmark |
| `hit` | forced homogeneous isotropic turbulence (SI-configured) |
| `channel` | wall-resolved turbulent channel flow (compressible, forced; spec: `docs/channel.md`) |
| `bench` | step-time and throughput benchmark |

Common to every subcommand: `--backend cpu|gpu|auto` (auto), `--out PREFIX`
(= subcommand name), `--seed N` (1), `--quiet`, `--dry-run`, `--help`.

`--config FILE.toml` supplies any of the above from a TOML run config —
`n = 512` is `--n 512`, `re_tau` is `--re-tau`, `emit = ["stats","png"]`
repeats the flag, `noslip = true` asserts it, and `subcommand = "channel"` may
replace the positional. Explicit CLI flags override the file.

### `flow` — orthogonal flag groups

* **Body** `--body cylinder|sphere|cube|lens|fence|floor|none|FILE.stl|.gltf|.glb`
  (exactly one, required). `cylinder` = z-aligned circle, `cube` = 45° rotated
  square prism, `lens` = two-disk intersection with sharp tips (2D),
  `fence` = wall-mounted blocker sealed to the tunnel floor, top edge at D
  above it, thickness D/8 — the 2D shear-layer rig; `floor` = raised solid slab
  over the full streamwise extent with its surface at D above the wall, i.e. a
  no-slip flat plate with its leading edge in the inflow plane — the wall-layer
  rig (both incompressible `--far tunnel`, both 2D; `--dr 10` blocks the bottom
  10%); `none` = body-free box for wall boundary layers (compressible only,
  pair with `--noslip` and/or `--stretch`). Modifiers:
  `--aspect A` (lens width/height, 3.5), `--dimples N` (golf dimples, 0 = smooth;
  cylinder → 2D disks, sphere → 3D caps), `--bumps N` + `--bump-h H`
  (floor trip spikes: N triangular teeth of height H cells, [4,48], default 8,
  pitched over 10–75% of the plate; the layer trips when the teeth poke through
  the local BL, k/δ ~ 1 — measured, teeth buried under a quiet-start BL stay
  laminar; `--body floor` only), `--scale S` (mesh, 0 = fit max extent
  to D), `--robust` (winding-number sign for leaky meshes),
  `--pitch/--yaw` DEPRECATED (§10).
* **Scene** `--scene FILE.toml` instead of `--body`: the generalized
  environment — several `[[objects]]` (cylinder/sphere/cube/lens/mesh) placed
  in units of D about the standard placement, an explicit `[model]` block
  (`dns` | `les` + Cs) merged before validation, one union-SDF sampled cut,
  and per-object `cd_/cl_/cdf_` columns in `<out>_forces.csv` (incompressible,
  ≥ 2 objects; the columns sum exactly to the stats totals). Parsed with the
  vendored toml++ (`third_party/tomlplusplus`). Spec: `docs/scene.md`;
  examples: `examples/*.toml`.
* **Physics** `--physics incompressible|compressible` (incompressible), `--re`
  (60), `--mach` (0.2, compressible only), `--les CS` (0 = DNS),
  `--sgs smag|sigma` (smag; sigma = transition-capable σ-model, Cσ ≈ 1.35,
  nz > 1 only), `--cfl` (0.8 incomp / 0.4 comp).
* **Far field** `--far tunnel|fringe|sponge|nscbc|periodic` + `--open` for the
  lateral faces (§7). Tuning: `--sponge-cells N` (0 = 10% of n),
  `--fringe-axes 2|3` (0 = auto: 2 when `--nz 1`), `--nscbc-beta` (0.5),
  `--nscbc-strip` (0), `--pot`.
* **Solver** `--tol` (1e-4, CG per-cell RMS residual), `--max-iters` (500),
  `--dt-recompute` (16, comp). The pressure solve is NOT selectable: it follows
  `--far` (periodic FFT for `fringe`/`periodic`, Neumann-box DCT otherwise).
* **Walls** `--noslip` — the closed y walls become adiabatic NO-SLIP mirrors
  (boundary layers grow on them) instead of free-slip. Rejected with `--open`.
* **Emit** (repeatable) `stats,bin,png,web,volume,slices,vtk,none`
  (default `stats,bin`), with `--fdt` 0.15, `--pdt` 0, `--wdt` 0, `--planes` 3,
  `--png-down S` (box-average PNG frames by S, which must divide nx and ny;
  1 = full res, e.g. 2048×1024 `--png-down 2` → 1024×512).
* **Grid** `--n` 512, `--ny` (=n), `--nz` 1, `--dr` 8. **Run length** `--ft` 20,
  `--steady TOL` 0, `--seq LEVELS` 1. **Compressible start-up** `--ramp` 2.0,
  `--kick` 0.05.

Grid conventions differ by physics and are deliberately **not** unified:
* incompressible: `L = 2π`, `h = L/n`, `D = L/--dr` (box scales with n).
* compressible: `D = π/4`, box = `dr·D` in y (`(n/ny)·dr·D` in x), D spans
  `ny/dr` cells, `dr ≥ 6`; `h = D/64 · (512/ny) · (dr/8)`. dr 8, n 512 = the
  historical 8D×8D box, bit-identical.

Merging them relocates the body and rescales Re — a real past bug (the old app
hardcoded `h = D/64` so n grew only the DOMAIN: body 67% downstream, sponge 0.4D
behind it, measured cd 3.19).

### `channel`

Spec: `docs/channel.md`. Wall units throughout (δ = 1, u_τ = 1, force density
f = 1, ν = 1/Re_τ), so `--re-tau` IS the run's friction Reynolds number in the
mean. Grid defaults are the KMM 192×128×160 box: `--n` 192 (streamwise),
`--ny` 128 (wall-normal FLUID cells; two mirror ghost rows are added
internally), `--nz` 160, `--lx` 0 (= 4πδ, KMM), `--lz` 0 (= 4πδ/3, KMM),
`--stretch` 2.0 (tanh wall clustering, [0,4]). Physics: `--re-tau` 180,
`--mach` 0.2 (BULK Mach, from the log-law bulk-velocity estimate; above ~0.3
compressibility contaminates comparison with incompressible references),
`--wall isothermal|adiabatic` (isothermal — adiabatic has no heat sink, so T
climbs until dt collapses: short transients only), `--cfl` 0.4. Start-up
`--kick` 0.2 (solenoidal trip on the laminar Poiseuille IC; 0 stays laminar).
Run length `--tt` 30 eddy turnovers (δ/u_τ = 1), `--dt-recompute` 16. Output
adds `--profile-dt` 1.0 (`<out>_profiles.csv`, Favre-averaged over x and z).

### `tgv`, `hit`, `bench`

* `tgv`: `--n` 64, `--nz` (=n), `--re` 1600, `--nu` (overrides `--re` when > 0),
  `--ft` 20 (end time t, not flow-throughs), `--cfl` 0.9. Emits
  `stats` (t, K, dKdt, epsVisc CSV), `vtk`, `error` (max-norm vs the 2D analytic
  solution + KE ratio; requires `--nz 1`), `slices`.
* `hit`: configured in SI (air viscosity, target u′), sized so `h = 1.5 η`
  (honest DNS) — which fixes Re_λ by N alone (256³ → 68, 512³ → 107, at any u′);
  runs internally in box units (L = 2π, u′ = 1) for FP32 health and reports both.
  Requires `nz ≥ 4` (spectral IC). `--n` 256, `--urms` 60 m/s, `--nu` 1.5e-5
  m²/s, `--spin` 10 eddy turnovers, `--stats` 10. Emits
  `stats | spectrum | snapshot | slices`. Linear forcing with a fixed-ε
  controller (Carroll-Blanquart): stationary ⟨ε_visc⟩ = ε_f.
* `bench`: `--n` 256, `--steps` 20 (after 3 warm-up), `--physics`. Reports
  ms/step, Mcell-updates/s, VRAM in use.

### Strictness, exit codes, `--dry-run`

Parsing is strict: unknown flag, missing value or non-numeric = hard error with
a suggestion, never a silent default. `--open` with `tunnel/sponge/nscbc` is
meaningful, with `fringe/periodic` it is REJECTED (not ignored); likewise `--pot`
outside compressible-nscbc / incompressible-tunnel-open, and `--nscbc-beta ≥ 1`.

Exit codes: `0` ok, `1` CLI, `2` config, `3` input, `4` output, `5` no GPU,
`6` resource, `7` numerical failure, `8` solver non-convergence, `77` skipped.

`--dry-run` validates, prints the resolved case (including the DNS/LES label)
and exits 0 — use it to check a config without touching the GPU.

## 4. Library layering

```
cli/config (parse + validate, pure)  [+ dns/scene, dns/remap, dns/sgs]
  → bodies/geometry (sdf + cut sampling)  [+ dns/curvmesh, dns/framemap]
    → gpu/{tunnel,bodyforce,les,cutpoisson,cutflux,derived,comp_channel}
      → gpu/runner_{incomp,comp,channel}   (the three time loops)
        → output (RunOutput, the one artefact sink)
```

Post-recompose additions, placed by layer: `dns/scene` (scene-file model, feeds
config), `dns/remap` + `dns/sgs` (pure headers: axis remaps and the SGS
operator), `dns/curvmesh` + `dns/framemap` (mesh mapping and the
physical↔index frame raster), `gpu/cutflux` (incompressible cut-cell flux),
`gpu/comp_channel` + `gpu/runner_channel` (the channel path).

`apps/` is `dns_main.cpp` alone. `main` is four phases:
**Parse → Validate → Acquire → Run**. Phases 1-2 touch NEITHER filesystem NOR
GPU, so every bad flag combination is caught before a byte is allocated; phase 3
creates/truncates every output BEFORE any `hipMalloc` (a full disk used to be
discovered after hours of GPU).

Policy: `std::expected` for anything a USER can cause, `assert` only for what a
PROGRAMMER can cause — an assert is legitimate only if the validator makes it
unreachable. Do not reintroduce assert-as-input-guard: Release is the build that
runs. Validated physics constants live at their definitions with the measured
bracket recorded.

Recomposition deltas worth knowing: forces unmoved (cyl Re60 n=256 cd 1.956 /
cdf 0.616 identical before + after); the compressible force tally is FP64 now
(was FP32 atomics, cd shift 1.7e-8 — the predicted non-determinism was NOT
observed at test size, so this is hardening, not a cured bug); derived fields
differ from the old host loops by FP32-vs-FP64 rounding only (max |Δvort| 7.6e-6
vs field max 78.7). FEATURE DROP: the old `movie` app's pressure channel is gone
— `tgv/hit --emit slices` writes speed + vorticity only. Deleted: direct-forcing
IB and the staircase wall (untested, superseded); `gpu/comp_bc.*`. Full record in
`docs/archive/` (`masterplan_recompose.md`, `analysis_recompose_[1-3].md`); git
branch `recompose-single-binary`, pre-recomposition tree at `master`.

## 5. Incompressible solver

RK3 + Chorin projection per stage; exact spectral inversion of the discrete
Laplacian (FFT periodic, DCT-II confined), preconditioned CG on the fluid
subspace with cut cells.

`--steady TOL` marches until `R = max|du/dt|·D/uInf² < TOL` (dt- and grid-free;
1e-6 tight, 1e-4 loose). `--ft` becomes the give-up cap — reaching it is exit 8,
not a silent answer. Incompressible, `--les 0`; only converges below the shedding
onset (cyl Re < 47). `--seq LEVELS` (requires `--steady`) converges on a ladder
`n/2^(LEVELS-1) .. n`, prolonging each solution as the next level's initial
guess: only the initial condition changes, never the scheme, so the converged
fine-grid answer is the one `--seq 1` reaches — without paying the full transient
at the finest resolution.

## 6. Compressible solver

`dns flow --physics compressible --body cylinder` (default `--far nscbc`;
`--far sponge` is an explicit escape hatch only — see §7).
The core was already 3D (`DeviceCompState` carries mz; `comp_flux.hpp::faceFlux`
loops axes 0/1/2, nz==1 cancels by wrap).

Sharp ghost wall (`gpu/comp2d_ghost.*`): per-stage mirror ghosts via
`setStageHook`, sampled ≥0.75h beyond the wall — never plain 2|sdf| mirroring
(barely-inside cells lose their stencil and the force tally's closed surface
breaks). Drag = exact momentum flux through fluid↔ghost faces.
`DeviceCompGhost` takes an arbitrary `Sdf = function<double(x,y,z)>` and is
3D-general (ndir = 6, 3D normals, ±z faces; only nz==1 flips nv[2]=0 / ndir=4).

Measured: Re60 M0.2 cd **1.822 / 1.830 / 1.832** (256/512/1024², converged;
incompressible sharp 1.785 + M² correction). Re800 M0.2: cd 1.71, St 0.2445
grid-converged — 4.8% below incompressible St 0.257, a real Mach effect (the
staircase had masked it). Tests: `comp`, `gpu_comp`, `gpu_comp_ghost`.

`stableDt` is recomputed every 16 steps (`--dt-recompute`) with a 0.95 margin
(was every step — two full-grid reductions + host syncs per step); trajectories
are not bit-identical to pre-2026-07-18 runs, statistics unaffected.

**Quiet start** (`gpu/comp_far.*`): `--ramp T` (default 2.0) starts the IC at
REST — exactly consistent with the no-slip body, replacing the 3h taper — and
raises the freestream by per-step uniform Galilean boosts (`boost()`) toward
`uInf·smoothstep(t/T)`; the sponge target tracks u(t). Measured startup
|p′|/p∞: impulsive 12.7%, quiet 0.56% (22×). The quiet start is TOO symmetric to
shed, hence `--kick A` (default 0.05): a deterministic solenoidal Gaussian-vortex
trip 1D behind the body (divergence-free ⇒ acoustically silent). Re800 trips
fast; Re60 onset needs `t > 100` or `--kick 0.3` (steady-symmetric cd 1.96 @512
walled vs the shedding-era anchor 1.82 — a different attractor, not a bug).
`--ramp 0` = legacy impulsive taper IC. `boostX(GpuGrid, DeviceCompState&, du)`
is a free function in `comp.hpp` so sponge-free setups keep the quiet start.

## 7. Boundary conditions / far field

### `--far tunnel` (incompressible default)
Lateral walls are FREE-SLIP/Neumann: tangential mirrored (zero wall-generated
vorticity), normal 0 + antimirrored ghost, pressure Neumann via DCT. Walls =
symmetry planes (periodic-image confinement). The earlier Dirichlet `u = uInf`
far field clamped velocity against sheared flow and painted a vortex sheet along
the walls — do not reintroduce it.

### `--far tunnel --open` (sponge-free open tunnel, 2026-07-20)
Tangential ghosts keep the free-slip/zero-gradient mirror (same code path); the
NORMAL velocity on lateral boundary faces is zero-gradient from the nearest
interior face (flux free, vents the displacement field — relaxing it is a wall in
disguise), and those faces join inlet/outlet as prescribed-flux zero-aperture
faces: `kInOutFluxRhs` adds them to the projection RHS, `correctOutletFlux`
closes `uInf·Ain = Qout + Qlat` at the outlet plane, pressure stays Neumann/DCT.
2D and 3D (y, and z when nz > 1).

ORDER MATTERS: lateral faces are slaved once per stage BEFORE the projection; the
post-projection ghost refresh is `apply(..., refreshOnly=true)` and must NOT
re-slave them — re-slaving after the gradient update broke div u in the boundary
cells (measured maxDiv 1e-2 persistent vs 6e-5). `openLat=false` is bit-identical
to the wall tunnel. Freestream is an EXACT fixed point through the full
BC→div→CG→gradient cycle (drift 0.0, 0 CG iters, 2D + 3D). Measured cyl Re60 dr8
256² t=31: cd 1.59 and falling vs walls 1.803, fringe 1.654 (lit unbounded 1.42)
— best open far field in the tree; maxDiv 6e-5. Long-window anchors not yet
regenerated. Tests `gpu_tunnel_bc §7-9`, `gpu_tunnel_open`.

### `--far fringe` (incompressible; was `--open`)
Fully periodic FFT cut-Poisson (skips `closeBoxBoundary`, `kBcTunnel` and the
outlet machinery) + all-side cosine fringe relaxing u → (uInf, 0, 0),
`ns = clamp(0.1nx, 8, nx/2−1)`, σmax 0.5/stage. Measured cyl Re60 dr8: cd 1.654
vs confined 1.803 (lit unbounded 1.42) — fringe is a softer image system, roughly
halving the blockage excess; maxDiv 1.5e-4. Tunnel mode bit-identical.
Test `gpu_open_incomp`.

### `--far sponge` (escape hatch; NOT a default) and `--open` lateral bands
Sponge is never a default anywhere — the compressible default is `--far nscbc`
(`config.cpp`). It relaxes the freestream in a boundary band that eats the near
field; ask for it by name or not at all.

`gpu/comp_far.*` (`DeviceCompFar`) = x-band freestream sponge + lateral free-slip
mirror walls, or `--open` lateral bands that relax rho/p ONLY (velocity free —
the displacement/dipole field is not clamped; relaxing v→0 or u→uInf laterally is
a wall in disguise). Measured cyl Re60 512 quiet: full-freestream lateral sponge
cd 1.981 ≈ walls 1.962, rho/p-only 1.914. Residual open-lateral blockage is
TOPOLOGICAL (periodic translation images through the free-velocity band) — shrink
it with domain size, not BC tweaks; a local-BC fix would be a
potential-source-corrected fringe target (not implemented).
Test `gpu_comp_far`.

### `--far nscbc` — six-face NSCBC / LODI
`gpu/comp_nscbc.*`, test `gpu_comp_nscbc`, writeup `docs/nscbc.md`.
Poinsot-Lele characteristic boundaries as a RHS OVERWRITE, not a state overwrite:
`DeviceComp::setRhsHook` (called after `kCompRhs`, stage primitives live) lets the
module rewrite rows i = 0 / nx−1 of k with the LODI d/dt plus the exact transverse
y/z `faceFlux` divergence; the rows then ride through SSP-RK3 via the normal axpy
(no dt plumbing needed — this is why it avoids `comp_bc`'s acoustic transients).
`kCompRhs` stays pure-periodic: the wrap-contaminated x-fluxes at those rows are
overwritten and no interior cell reads the wrap face. x-viscous terms on boundary
rows are dropped.

* Outflow: `L1 = σ(1−M²)c/Lx (p − pInf)`, σ 0.25.
* Inflow: relax (u, v, w, ρ) with `etaIn 100` — MUST be stiff: etaIn 2 let the
  blockage pressure gradient sag the inflow row to 0.79 uInf (cd 1.30, 30% low);
  100 pins it to 1.000 with sag ~1/etaIn.
* Outflow REQUIRES the Yoo-Im transverse correction
  `L1 −= betaTrans (Tp − ρc·Tu)`, `--nscbc-beta` default 0.5, ≥ 1 rejected:
  measured Re800 scan — β 0 pumps a growing global mode (wake cores crossing the
  outflow radiate uncanceled incoming waves, cl 2→13), β = M = 0.2 still drifts,
  0.5 clean, 1.0 diverges at t = 35 (full cancellation is ill-posed at low M).
  Unit vortex-exit residual: 13% (β 0) → 2.5% (0.5).

**Six-face** (2026-07-18): generalized from x-rows to all open faces — Config
`openY/openZ` (+ `sigLat` 0.25, `etaLat` 0.25) turn the lateral extremes into
characteristic far-field faces. Incoming acoustic = soft pressure anchor (normal
velocity FREE — the displacement field vents, no wall in disguise); entropy/shear
waves upwinded on `sign(outward un)` with gentle freestream relax when inflowing
(switch tanh-blended over `kLatBand = 0.01·cs`; the x-outflow branch is
deliberately untouched, it holds the Re60/Re800 anchors); edge/corner cells sum
per-axis LODI contributions (Lodato-style), non-boundary axes keep the exact
`faceFlux` divergence, transverse Yoo-Im derivatives go one-sided at edges of open
axes. One kernel `kNscbcCells`, disjoint slab enumeration; `openY/Z = false` is
bit-equivalent-in-math to the old x-only rows. 3D mesh bodies always use all-face
NSCBC (sponge + walls removed). Cylinder: `--far nscbc --open` = laterally open;
walls remain the default.

Measured @512 (quiet + kick, window [62.8, 94.3]): Re60 cd 1.813 / St 0.145±;
Re800 cd 1.776±0.180 / clAmp 1.33 / St_fft 0.2499 vs same-era sponge control
1.894±0.176 / 1.85 / 0.2503 — identical σ and St, cd 6% lower = the wake exits
with its real momentum deficit (less streamwise confinement). NB the
quiet-start-era sponge itself no longer reproduces the impulsive-era anchor
(1.894 vs 1.71; St 0.250 vs 0.2445) — Re800 anchors are start-condition-sensitive
and need regeneration either way. Sphere Re100 M0.2 dr6: 64³ cd 1.10, 96³
cd 1.146 (cl, cs ~1e-3) vs the walled-sponge-era 96³ anchor 1.226, literature
unbounded ~1.09 — blockage excess gone. Open-BC sphere anchors: Re300 128³
cd 0.681±0.006 (lit ~0.66), Re500 192³ cd 0.589±0.010 (lit ~0.55). Unit
residuals: radial pulse all faces 2.4%, lateral-skimming vortex 2.9%,
vortex-on-face 2.6-2.8%; 3D freestream fixed point exact. NOT yet validated:
3D/mesh app, long-window saturated Re60 shedding; anchors remain sponge-era.
Deferred: FP64 gauge-p differencing, wall-aware transverse stencil, LES nut at
boundary rows, betaLat split, long-window open validation campaign.

### `--nscbc-strip N` — the Re2000 limit and the production hybrid
Pure NSCBC diverges at Re2000 2048² (t = 63.6) — residual outflow reflection pumps
box modes between the stiff inflow and the walls; final frames show vorticity
sheets ON the free-slip walls and shed FROM the inflow row (a stiff inflow is a
wall for transverse sloshing). At 1024² it survives but is contaminated
(cd σ 0.58, clAmp 4.8). Fix = a thin sponge strip (sigMax 0.5, N ≈ 1-1.5% of nx)
inside the NSCBC rows. 1024² triplet [50, 96]: pure 1.846±0.582 / clAmp 4.8,
strip12 1.902±0.341 / 2.4 (calmest), sponge 2.087±0.370 / 2.8; the cd ordering is
the confinement ordering. 2048² strip32 ran clean to t ≈ 25 (user-stopped, not
validated long). Rule: `--far nscbc` alone ≤ Re800; add `--nscbc-strip ~ny/64`
beyond (the validator enforces this). `N = 0` is byte-identical to pure six-face.

### `--pot` — potential-corrected far field
Compressible `--far nscbc`, incompressible `--far tunnel --open`; rejected
elsewhere. Shared field evaluator `gpu/farpot.hpp`; tests in `gpu_comp_nscbc` and
`gpu_tunnel_bc` pot sections.

The finite-domain bias of uniform-freestream targets is a REAL measured effect:
pinning u = uInf at an inflow row ~2.7D upstream ignores the body's upstream
influence (source + doublet ≈ −7% u there at dr8), and anchoring p = pInf at 4D
laterally is a soft wall. Fix: far-field targets = freestream + wake source
(2D `Q/2πr`, 3D `Q/4πr²`; `Q = F_drag/(ρ∞u∞)`, EMA over `2D/u∞` in the runner so
shedding does not feed back, fed only past the ramp) + displacement doublet
(a = D/2, strength tracks the ramped u∞), Bernoulli pressure. Applied to the
inflow relax targets, the lateral p anchor + entropy/shear targets, and the
outflow p∞. `Q = 0 ∧ a = 0` (or the flag off) is bit-identical (test-pinned).

Measured domain scan, cyl Re100 M0.2 D = 32 cells, open NSCBC, window [90, 153]:
plain 8D cd 1.681 / St 0.190 → plain 16D 1.469 / 0.175 → plain 32D 1.377 / 0.168;
**pot-corrected 8D: cd 1.365 / St 0.165** = the 32D answer at 1/16 cost, in the
unbounded-literature band (incompressible lit ~1.33-1.35 / 0.164). Bias halves per
box doubling without it. Not default (pre-existing anchors are uniform-target).
3D forms are implemented but only 2D is validated.

Incompressible `--pot` corrects the open tunnel's Dirichlet INLET only
(u = uInf + source + doublet per face, v/w ghosts antimirror about the target;
laterals stay free — zero-gradient already vents adaptively), with pot-aware flux
accounting (projection RHS + outlet balance read the actual inlet faces,
`corr = −Qnet/A`). Q EMA fed from cd after `t > 2D/u∞` (the impulsive-start spike
is not drag). Measured cyl Re100 dr8 D = 32 cells (window ≥ 90 or scaled):
plain 8D cd 1.736 / St 0.192 → **pot 8D 1.409 / 0.168** vs truth 32D (n = 1024)
1.403 / 0.168 — cd within 0.4%, St exact; lit ~1.33-1.35 / 0.164. clAmp stays
box-sensitive (0.59 vs 0.34 at 32D) — lift amplitude feels the uncorrected
laterals; know this before citing clAmp from small boxes.

Implementation note: the incompressible correction is a separate sequenced kernel
`kPotInlet` AFTER `kBcTunnel` — folding it into `kBcTunnel` perturbed codegen and
broke the pinned bit-identity with the historical 2D kernel via a pre-existing
benign inlet-ghost-corner race (excluded in test 10a).

`--dr` also works compressible: box = `dr·D` in y (`dr·(n/ny)·D` in x), D spans
`ny/dr` cells, `dr ≥ 6`; dr 8 is bit-identical (h and the 2.1107 placement
literal preserved); otherwise the body sits at lx/3.

## 8. Ghost-fill sharp viscous wall

`gpu/ghostfill.*`, GPU backend only. Per RK stage the **diffusion** term reads a
ghost-filled copy of the field (solid faces get mirror values `−q(image)` from the
body sdf, no-slip at the true sub-cell surface); convection keeps the masked zeros
— mirror values in the skew form inject energy (tested), so NEVER feed the ghost
view to convection or leak it into the state. Ghost faces have zero aperture:
projection/continuity never see them. `kBodyFriction` / `kBodySgsFriction` tally
per-link `(q − q_ghost)` drains through the ghost view.

Measured (cylinder Re60, matched t ≈ 21.7): ghost cd 256² = 1.803, 512² = 1.785,
1024² = 1.785 (grid-converged); the old staircase 512² was 1.719 — 0.066 below the
converged value (staircase pinned no-slip up to h inside the body, under-predicting
shear). Steady tail 512²: cd 1.813 / cdf 0.553 (old staircase run 1.738 / 0.514).
`data/` regression anchors were regenerated with the sharp wall on 2026-07-16 —
pre-ghostfill numbers survive only in memory notes.

## 9. Cut cells

`cutcell.*`: lambda + face apertures, exact geometry (rotated cube / sphere /
z-cylinder), `sanitizeCut` then optional `closeBoxBoundary`. Pressure =
FFT/DCT-preconditioned CG on the fluid subspace. `makeMeshCut` / `makeGolfCut` /
`makeLensCut` are now the single `sampleCut()` in `geometry.*` (bit-identical,
verified against all three).

### `--cut-flux` — sharp apertured momentum (incompressible, off by default)

Momentum advection/diffusion classically see the body as a binary staircase
(`maskSolidFaces` zeros); `--cut-flux` instead weights the convective and
viscous fluxes by the exact face apertures the projection already uses, over
λ-floored control volumes (`gpu/cutflux.*`, λ_floor = 0.5 on the time term
only). Advecting fluxes are averages of the stored face mass fluxes and the
skew correction uses the APERTURED divergence, so the operator stays exactly
skew-symmetric in the λ̃-weighted inner product — no hidden dissipation
(pinned by `test_gpu_cutflux`: inviscid KE bounded, exact 2D degeneracy).
Fluid↔ghost links keep full conductance (the mirror owns the sub-cell wall;
`kBodyFriction` unchanged); the flag band is cleared near the domain boundary
(the far-field rows own it). Interior cells are bit-identical BY CONSTRUCTION:
the plain kernel runs first, a band-only kernel overwrites flagged cells (two
TUs do NOT contract identical expressions identically — measured).
Compressible `--cut-flux` is validator-blocked: the KG split flux loses its
KE-consistency under per-face area weighting and the run diverges near the
body (measured 2026-07-23, twice).

### `--penalize ETA` — λ-smooth Brinkman band decay (experimental)

Near-wall velocity decays by `exp(-dt·(1-λ_face)/ETA)` per stage, smooth along
the surface, unconditionally stable, both physics (compressible: momentum
only, drained KE becomes internal energy). CAVEAT measured: it HALVES cdf
(the ghost mirror reads a pre-damped profile) — never use it when friction
physics matters. No cd-budget tally yet.

### Upstream "ripples" — what they are and what actually helps

The stationary vorticity filaments upstream of bodies (visible under asinh
maps at 256³+, ABSENT at 128³) are broadband 6–12h dispersive standing waves
of the central scheme: continuously seeded at the shoulder by the SUB-CELL
boundary layer (D/√Re < 1 cell at these Re), trapped in the low-speed
stagnation region where their upstream group velocity holds. Every clean
lever was measured (2026-07-23/24, 256³ σ-sphere probes): `--cut-flux` ~20%,
`--penalize` ~15%, quiet start (`--ramp`, now incompressible too, opt-in)
~0% — the saturated level is forced, not a start transient. They are the
honest footprint of an unresolved BL under a zero-dissipation scheme:
schemes that suppress them (Smagorinsky |S| in the laminar BL, filters,
penalization) do it by injecting unphysical dissipation. The only true cure
is resolving the BL (D/√Re ≥ 3 cells). Amplitude is ~20% of local wmax at
0.25D, decaying upstream; bounded, stationary, does not destabilize.
No filter/limiter is applied anywhere: dissipation stays ν + explicit SGS.

## 10. Mesh bodies + orientation convention

`src/dns/trimesh.*` (STL hand parser + glTF/GLB via vendored
`third_party/cgltf.h`) and `src/dns/meshsdf.*` (`MeshSdf`: BVH closest-triangle,
angle-weighted pseudonormal sign, winding-number `--robust` fallback; sdf < 0 in
solid; `makeMeshCut` = golf-ball supersampler). CPU-only, build-time only
(~7 s @96³) — ghostfill precomputes, the sdf never enters the stepping loop.
qRef from the aperture-measured frontal area (`makeMeshCut` + `sanitizeCut`; the
cut is used for area ONLY, never the solve).

Mesh insertion into the compressible solver = feed `MeshSdf` to the ghost hook —
no solver rewrite. `DeviceCompGhost::make(..., bbox)` and the IC taper take the
body AABB and skip the mesh sdf for any cell provably beyond the near-wall band
(2h + Δ) / 3h taper — O(bbox), not O(grid) sdf evaluations (both loops were
unculled O(16.8M) at 256³ ⇒ a ~26 s "hang"; bbox ⇒ ~1.3 s, bit-identical,
cd 1.47066 unchanged). Keep the bbox parameter — it is a ~20× setup win, not dead
weight. `bbox = nullptr` = evaluate everywhere (analytic sdfs, tests).

Validated: sphere STL 96³ Re100 M0.2 dr6 cd 1.226 (Afront 0.856 ≈ πR², cl/cs
~1e-3, 2230 steps stable). `val_mesh_sphere`: sphere STL vs analytic sphere Re300
cd 0.710 vs 0.685 — the gap is cut discretization at D = 12 cells, proven
mesh-resolution-independent. Tests: `meshsdf` (icosphere/cube oracle),
`gpu_comp_mesh` (icosphere `MeshSdf` vs analytic sphere sdf, 0.3%),
`gpu_comp_ghost3d` (sphere quiescent + drag + symmetry), `val_comp_mesh_sphere`.

**Attitude convention (wake viz).** The built-in `--pitch`/`--yaw` are NOT
aircraft angles — `transform()` pitches about model-x and yaws about model-z,
applied to raw model axes, so they neither zero at nose-on nor rotate about flight
axes. Do not use them to set AoA. Instead BAKE attitude into the mesh with
`build/orient.py pitch yaw roll` (needs a venv with trimesh) and leave
`--pitch 0 --yaw 0`. Convention: DNS world +X = freestream, air comes from −X.
(0,0,0) = nose points upstream (−X), wings level (span +Y), up +Z (fuselage along
X, wings edge-on). pitch > 0 = nose UP (about +Y), yaw > 0 = nose to +Y (about
+Z), roll > 0 = right-wing-down (about −X). VERIFIED model axes of
`b777_solid.stl` (from orthographic silhouettes — do NOT trust extent magnitudes,
they are near-symmetric): model-X = fuselage (nose at −X, tail/fin at +X),
model-Y = height (up), model-Z = wingspan. `orient.py` bakes
`P = [[1,0,0],[0,0,-1],[0,1,0]]` then `Rz(yaw)Ry(pitch)Rx(−roll)`. ALWAYS
re-verify a new mesh by rendering the world side-silhouette (nose high and
upstream at +pitch) before trusting it. VIEWER: the three.js up-vector must be +Z
(`camera.up.set(0,0,1)`) — aircraft up is Z, not the three.js default Y, or pitch
renders as a sideways swing. Body and flow are marching-cubed from the same
λ/velocity grid, so they align.

## 11. DNS vs LES — which runs mean what

The core scheme is DNS: KE-conserving skew-symmetric advection, the ONLY
dissipation is the physical ν. Valid iff the grid resolves the smallest active
scale (body BL ~ D/√Re). Cylinder 512², D = 64 cells: honest up to Re ≈ 500
(BL ≈ 3 cells). Beyond that the separated shear layer piles enstrophy at the grid
scale and the run diverges (~t = 3 at Re 3e5) — no CFL/tol setting fixes zero
dissipation.

`--les Cs` (default 0 = off) switches to LES: adds per-stage `div(νt grad u)`,
`νt = (Cs·Δ)²|S|` (`gpu/les.*`, `DeviceSmagorinsky`; shared kernels untouched).
Full 3D since 2026-07-20: complete 3D strain + w diffusion,
`Δ² = (hx·hy·hz)^⅔` at nz > 1; at nz == 1 the z terms are exactly zero and
`Δ² = hx·hy` is kept verbatim, so 2D LES trajectories are bit-identical to the
historical 2D-only kernel (the old `--nz 1` rejection is gone). Compressible LES:
`DeviceComp::make(g, γ, μ, csLes)` adds a full-3D Smagorinsky stress via an
optional per-cell eddy viscosity `μ_t = ρ(CsΔ)²|S|` face-averaged into
`faceFlux`'s μ (and `μ_t·cp/Pr_t`, Pr_t = 0.9, into κ); `Δ² = (hxhyhz)^⅔`
(hxhy at nz == 1); `stableDt` folds max μ_t into the viscous limit (SmagMax
reduction). `csLes == 0` is bit-for-bit molecular DNS; the ghost force tally calls
`faceFlux` with `nut = nullptr` so `cdf` stays ν-only.

`--sgs smag|sigma` (default smag) selects the νt operator, same plumbing in both
physics. `sigma` is the σ-model (Nicoud et al., Phys. Fluids 23, 085106, 2011):
`νt = (Cσ·Δ)²·Dσ`, `Dσ = σ3(σ1−σ2)(σ2−σ3)/σ1²` from the singular values of
∇u (`dns/sgs.hpp`, shared by all three νt sites; CPU-pinned in `test_sgs_op`).
Use `Cσ ≈ 1.35` — the constant slot is `--les`, it is NOT a Smagorinsky Cs. Dσ
vanishes identically in pure shear, solid rotation, axisymmetric strain and any
2D flow, and decays as y³ at walls — so laminar/transitional boundary layers see
zero eddy viscosity and transition/separation stay physical (the drag-crisis /
tripped-BL use case, where Smagorinsky's |S| relaminarizes the very BL under
study). Consequences: `--sgs sigma` requires nz > 1 (validator: Dσ ≡ 0 in 2D
would be a silent no-op), and a σ-run's nut says nothing until the field is
genuinely 3D. Smagorinsky stays the default; all pinned trajectories are on it.

Consequences — never blur these:
* LES output is a FILTERED field; statistics depend on grid AND Cs by
  construction (measured pre-ghostfill: cd 1.70±0.34 @512² vs 1.88±0.44 @1024²,
  νt ∝ Δ²). This is not a discretization error; do not present LES numbers as
  converged predictions.
* Dissipation at the body is model-supplied, so cd/St/separation carry model
  error. The `cdf` friction column uses ν only (no νt in the surface stress).
* 2D caveat: Smagorinsky models a 3D forward ENERGY cascade; 2D cascades energy
  up and enstrophy down (no vortex stretching), so 2D + `--les` = stabilized /
  regularized 2D Euler, qualitative only.

**Rule**: if the run needed `--les` to survive, label it LES everywhere (reports,
artifacts, memory) and state Cs; use the DNS label only when `--les 0` and
BL ≥ ~3 cells. `--dry-run` prints the resolved DNS/LES label.
Tests: `gpu_comp_les` (Cs = 0 bit-identical, conserves mass/momentum/E,
KE < DNS in 3D).

## 12. Curvilinear (distorted regular) meshes — compressible

`src/dns/curvmesh.*`, tests `comp_curv` + `gpu_comp_curv`. Periodic structured
curvilinear support as an OPTIONAL path inside `Comp` / `DeviceComp`:
`CurvMesh` (one node per cell = its low corner, periodic images offset by the
axis-aligned box) → `CurvMetrics::build` (the only validator: sizes, positive
volumes, invertible face Jacobians, planar layer at nz == 1) →
`Comp::make(mesh, γ, μ)` / `DeviceComp::make(metrics, γ, μ)`.

Scheme = finite volume: KG flux with the exact bilinear-quad face area vector S
(`Un = u_avg·S`), viscous gradients in index space through per-face inverse
Jacobians, divergence / cell volume (volumes via consistently-diagonalized face
triangles ⇒ cells tile the box EXACTLY). Measured properties: per-cell face
closure exact ⇒ freestream preserved (double 6e-14; FP32 50-step 0.0),
conservation exact-telescoping (1e-15 double / 1e-6 FP32 over 100 steps),
uniform-mesh degeneracy = Cartesian rhs to 5e-15, isentropic-vortex convergence
ratio 4.01 (2nd order), nz == 1 keeps mz EXACTLY 0 on distorted meshes
(prismatic z ⇒ Ji z-column exact zeros). `stableDt`: `(|u| + c)·aov` per axis
(|u| ≥ |u·n̂| ⇒ conservative), viscous limit from hmin.
`CurvMesh::sinusoidal(g, amp)` = a fixed smooth mapping (skewness survives
refinement), |amp| ≤ 0.06 valid.

LES IS wired on curvilinear (`make(metrics, gamma, mu, csLes, sgs)`): the same
two operators, the same eddy conductivity `nut·cp/Pr_t`, with the velocity
gradient assembled from the mesh metrics (`curvVelGradCentre`: index-space
central differences contracted with the cell-centre inverse Jacobian, averaged
from the two faces normal to each axis) and the filter width the cell's OWN
`Delta² = vol^(2/3)` (`vol/hz` at `nz == 1`), so a clustered mesh filters at
the local spacing instead of a box average. Measured degeneracy: on the
identity map both operators match the Cartesian LES path to 1e-6 rel Linf over
10 steps, and the LES viscous dt limit matches to 1e-4 (the ACOUSTIC halves of
the two `stableDt`s are not comparable by construction — see the `|u|` vs
`|u·n̂|` note above — so the dt pin exaggerates C until the viscous limit
binds). At `nz == 1` on a distorted mesh the sigma model is still identically
zero (the Ji z-column is exactly zero ⇒ the gradient tensor stays 2-block ⇒
σ3 = 0), so a sigma run there is BIT-identical to molecular DNS — test-pinned.
NOT wired: sponge on curvilinear. The Cartesian path is untouched — dispatch
is one branch on the optional metrics. The compressible periodic curvilinear path IS driven by
`dns channel` (`docs/channel.md`): `CurvMesh::channelY` + `DeviceChannelWall`
+ body-force RHS hook.

### 12.1 Focal clustering — `--focal` (bodies on a curvilinear mesh)

`--focal x0,x1,y0,y1,z0,z1` (or `auto`) puts a FINE plateau box where the
physics is — a body plus its wake — and coarsens the far field, via one
independent per-axis `Focal1D` map composed into `AxisRemap3` (`dns/remap.hpp`).
`Focal1D` is an integrated quintic smoothstep: shape `s(u)=u³(10−15u+6u²)`,
weight `phi` = 1 on `[a,b]`, blended to 0 over `w` each side, and
`x(X) = c(X − (1−r)∫phi)`, `dx/dX = c(1 − (1−r)phi) ∈ [c·r, c]`, `r = 1/R`,
`c = L/(L − (1−r)((b−a)+w))`. The COMPACT support is why polynomial beats
tanh here: outside the blend the slope is exactly `c`, so the far grid is
exactly uniform, the periodic wrap is exactly C², the plateau spacing is
exactly `c·r·h` and the cell-size ratio is exactly `R` (all test-pinned).
Inactive is the exact identity by branch, so an all-off `AxisRemap3` gives
`CurvMesh::fromRemap` nodes bit-identical to `uniform` and the device path
reproduces the Cartesian one.

ONE concrete type, not a template product: NSCBC and the vorticity frame each
get exactly one extra instantiation and the coefficients travel as data, so
any combination of clustered axes costs no further kernels.

Validator (compressible, `--far nscbc` or explicit sponge, mutually
exclusive with `--stretch`; `--les` is allowed and filters at the per-cell
Delta): the blend must clear each domain face by ≥ 4
cells (x also `nscbc-strip + 2`) so the boundary treatment always reads an
exactly uniform grid; the plateau must span ≥ 4 fine cells; `nz == 1` forbids
a z plateau; and the max ADJACENT cell growth is capped at 1.25 — the KG
central flux has no dissipation, so a sharp coarse/fine step is a dispersive
reflector; widen `--focal-blend F` (blend width per side as a fraction of the
plateau extent, (0,2], default 0.5) to get under that cap. `--dry-run` prints
per-axis fine/coarse spacings, the exact ratio,
the growth and the dt penalty factor. With a body the plateau must contain it
plus a 3-fine-cell guard, so the ghost wall's mirror stencil never straddles
the blend; `auto` derives the box from the body AABB (`--focal-pad`,
`--focal-wake`), shrinking it to fit and reporting what it resolved.
Analytic bodies only — a mesh/scene AABB is not known until the file loads and
validation touches no filesystem.

Display raster (`dns/framemap.hpp`, `PhysicalRaster`): the solver's arrays are
indexed in REFERENCE space, so a pixel-per-cell frame is a picture of the INDEX
space — on a clustered mesh the plateau is blown up and the far field squashed,
and the body comes out the wrong shape. Every frame (and the body mask) is
resampled onto a uniform PHYSICAL grid before it reaches `RunOutput`, which is
constructed with that render grid. The raster pitch is the FINEST spacing on
each axis, so clustering never costs displayed detail: the render grid is
LARGER than the solver grid by up to `--focal-ratio` per clustered axis
(measured: 128³ focal auto ratio 4 → 326×382×382, PNGs 326×382). Guards: per-
axis growth ≤ 8, and a 128M total render-cell cap past which all grown axes
shrink by one common factor — the geometry stays true, only the detail is
capped. The identity map is INACTIVE: render grid == solver grid and `apply()`
is a bit-exact copy, so uniform-mesh runs are unchanged. Test-pinned by round
trip (a known function of physical position, stored at the reference cells,
comes back as that function on the raster, to the linear-interpolation floor of
the coarse cells) and by a step edge landing at its physical position.
Display only — nothing here touches state, fluxes or statistics.

Ghost wall on curvilinear (`comp_ghost`): classification is entirely host-side,
so the map's inverse never reaches a kernel. Cell centres, normals, wall
distances and band tests are PHYSICAL (`R.map` of the reference centres); the
sample point is taken back through the per-axis inverse so the trilinear
weights stay the Cartesian index-space ones (a separable monotone map keeps
the centre lattice a valid interpolation lattice). The force tally switches to
`faceFluxCurv`, which carries the exact face area vector, so the `hy·hz`
multiply is dropped.

Measured (Re100 cylinder, M0.2, NSCBC, 2D, mean over t > 60):

| mesh | cells | mean cd | St |
|---|---|---|---|
| uniform 512 (reference) | 262k | 1.7983 | 0.2532 |
| uniform 256 | 65k | 1.7876 | 0.2544 |
| focal 256 `auto`, R = 2 | 65k | 1.8029 | 0.2546 |

Read this as "focal costs nothing", NOT as "focal wins": Re100 at 256² is
already grid-converged (uniform 256 is within 0.6% of uniform 512), so the
case cannot discriminate. It does pin that clustering, its blend and the
ported ghost wall leave cd and St intact at 1/4 the reference's cells.

To measure a BENEFIT the coarse uniform mesh has to be genuinely
under-resolved while the reference is not — the body BL spans D/√Re, so on
the default 8D box that window is Re ~ 200-400. At Re300 the BL is 3.7 cells
on uniform 512, 1.9 on uniform 256 and 3.2 on the focal mesh:

| mesh (Re300) | cells | mean cd | err | St | err |
|---|---|---|---|---|---|
| uniform 512 (reference) | 262k | 1.7366 | — | 0.3012 | — |
| uniform 256 | 65k | 1.7060 | −1.76% | 0.3053 | +1.36% |
| focal 256 `auto`, R = 2 | 65k | 1.7240 | −0.73% | 0.3026 | +0.46% |

At the SAME cell count focal more than halves the drag error and cuts the
Strouhal error ~3×, recovering most of the gap to a 4× finer uniform mesh.
The margin is modest because 2D Re300 is itself near convergence; the gap
this closes grows with Re, which is the regime the feature is for.

One trap worth stating, because it cost a sign-flipped drag: the deep-solid
threshold that decides "quiescent" vs "mirrored" is a physical depth, and on
ANISOTROPIC cells it must follow the WIDEST local spacing. Set from the
smallest, the mirror band is under two cells thick along the wide axis, the
interior stencil reaches quiescent deep cells, and cd went 1.49 → −7.6 (a
thrust). Isotropic cells make the two identical, which is why the Cartesian
path never saw it. Pinned structurally in `test_gpu_comp_ghost_focal`.

## 13. 2D == `nz == 1`, no separate code

There is NO 2D code path. `nz == 1` IS the exact 2D restriction: the periodic wrap
maps k±1 → k, every z-stencil term cancels identically (same address minus
itself), and w stays exactly 0. Kernels, cut-cell operators, CG and reductions are
all shared and untouched. Only the solvers special-case degenerate axes:

* Periodic: `hipfftPlan2d` when nz == 1; `lamz = {0}` (`poisson.hip`).
* NeumannBox: the degenerate axis has no ghost ring (m = 1) and the DCT pass is
  skipped — rotation across a size-1 axis is a memory no-op; pack/unpack ghost
  offset is 0 there (`ghostOff`).
* `closeBoxBoundary` leaves degenerate-axis faces open/periodic.

Constraints: only z may be degenerate (nx, ny > 1); NeumannBox needs n ≥ 4 per
confined axis. Use `hz = 1` (unit depth) — since 2026-07-20 this is ENFORCED by
the validator, not assumed, which rescales the native-2D incompressible `K`
(kinetic-energy) stats column by 1/h vs pre-recomposition runs (~40.8 at n = 256).
Forces are unaffected: hz cancels between the frontal area and the surface
integrals. When touching kernels/solvers: keep the cancellation exact (never
branch z-terms out) and keep the `*2d` tests (2D vs z-invariant 3D slice parity)
green.

## 14. Output and viewers

`--emit` is repeatable; default `stats,bin`.

* `stats` — `<out>_stats.csv`. **Three schemas, one filename** — parse by
  COLUMN NAME from the header row, never by index:
  * incompressible `flow`: `t, dt, K, cd, cl, cdf, cfl, maxDiv, pcgIters, resid`
  * compressible `flow`: `t, dt, cd, cl, cs, wmax`
  * `channel`: `t, dt, ubulk, utauBot, utauTop, retau, KE`
* `bin` — `<out>_{vort,speed,mask}.bin`, float32 mid-plane, one frame per `--fdt`.
* `png` — live PNG frames (`src/dns/fieldpng.*`, test `fieldpng`); at `--nz`
  > 1 the centre z-plane is rendered (channel precedent). High-Re framing:
  `--vort-max V` (clip in uInf/D; auto 8/15 saturates ~15× at Re 1e5),
  `--vort-map asinh` (display-only compression, bin/web stay raw),
  `--speed-map delta` (|u|−uInf diverging, freestream = background),
  `--vort-signed` (3D compressible: signed wz instead of |curl|):
  `png_<out>/vort_%05d.png` (diverging cool-warm, clip ±8 uInf/D, body
  dark) and `speed_%05d.png` (viridis, 0..1.6 uInf), one pair per `--pdt` frame
  (0 = follow `--fdt`), alongside the `.bin` frames. Writer = hand-rolled zlib PNG
  (`dns_core` links ZLIB), row 0 = image bottom.
* `web` — live browser viewer (spec `docs/viewer-format.md`, writer
  `src/dns/webframes.*`, test `webframes`, viewer `site/live.html`). Writes
  `web_<out>/` = `manifest.json` (atomic rename = commit) + u16-quantized per-field
  frames. 2D bodies: vort/speed at `--wdt` cadence. 3D: |curl| / |u| on 9 planes
  (3 per axis, centre ± n/16 cells), dirs `web_<out>[_xy|_xz|_yz][_m|_p]`,
  discovery via `web_<out>/planes.json` `{suf, axis, off-in-D}`; the viewer shows
  axis + offset tabs and falls back to probing the 3 centre planes for old runs.
  `--planes 1` = centre only. Post-run: `build/volume_to_web.py <prefix>` slices
  the dumped 3D volumes (`--emit volume`) into the same 9-plane set under
  `web_<prefix>_vol` (completed, scrubbable). View: `python3 -m http.server <port>`
  at repo root, then open `site/live.html?run=../build/web_<out>` — it polls the
  manifest, follows the live tail, and is scrubbable. Ports 8000 (old wake viewer)
  and 8080 (SearXNG) are taken on this machine.
* `volume` — `<out>_{vol,lam}.bin`, the full 3D field per frame; 3D mesh runs also
  write `_curl.bin` / `_speed.bin` / `_lam.bin` + `.json` for the wake viewer.
* `slices` — three centre planes instead of the volume. `tgv`/`hit` slices are
  speed + vorticity only (the old `movie` pressure channel is gone).
* `vtk` — `<out>_%05d.vtk`, legacy structured points.
* `none` — diagnostics only.

## 15. Validation suite + quarantine status

Smoke coverage for the three time loops is separate and unquarantined:
`gpu_runners` (`tests/test_gpu_runners.cpp`) drives `--far tunnel` with walls
(the incompressible default), `--physics compressible --far nscbc`, and a short
`dns channel`, asserting only exit 0, finite coefficients, a healthy projection
and that stats rows were written. No literature band, so nothing there is
quarantine-shaped.

`test_val_*` + `tests/validation_util.hpp`; labels = truth class
(`analytic` / `literature` / `anchor`) + `validation`. Heavy tier behind
`-DDNS_HEAVY_VALIDATION=ON` (~25 min, RUN_SERIAL). Routine: plain `ctest`
(Tier A ≈ 3 min on GPU). Tolerance rationale and the Tier C anchor-regeneration
procedure: `docs/validation.md`; scalar anchors in `data/validation/anchors.csv`
(deferred anchors ⇒ rc 77 skip).

**QUARANTINE (2026-07-20, recomposition)**: the nine literature-band `val_*`
registrations sit behind `-DDNS_VALIDATION_BASELINE=ON` (default OFF) — sources and
`anchors.csv` are untouched on disk. They were set aside BY DECISION so the
recomposition was not gated on bit-identity, NOT because they failed.
`val_spindown` (analytic, no body force) stayed green throughout as the canary.
Re-measure and un-quarantine per `docs/archive/masterplan_recompose.md` §7 +
`docs/validation.md` before citing any band again. NB no pre-recomposition baseline
CSVs were captured, so regeneration starts from measurement, not comparison.

Key measured: spindown KE 2.7% / profile 0.8% vs the Bessel series;
free-slip-image confinement excess vs unbounded literature GROWS with Re
(St +12 / +20 / +13% at Re60/100/200 @dr8; cd +26-29%) — the bands are
repo-calibrated, do not "fix" them to raw Williamson. TGV convergence to the
spectral peak is oscillatory (64³ −8.1%, 128³ +5.6%, 256³ −4.3%, 512³ +1.2%), not
monotone; at Re = 1600 the HiOCFD spectral reference is peak dissipation 0.012858
at t = 8.97.

## 16. Measured anchors at a glance

All numbers verbatim from the sections above; read the surrounding section for the
conditions before citing any of them.

| case | number |
|---|---|
| cyl Re60 incomp sharp wall, matched t≈21.7 | cd 1.803 (256²) / 1.785 (512²) / 1.785 (1024²) |
| cyl Re60 incomp steady tail 512² | cd 1.813 / cdf 0.553 (old staircase 1.738 / 0.514) |
| cyl Re60 n=256 recomposition parity | cd 1.956 / cdf 0.616 (unchanged) |
| cyl Re60 dr8 far-field comparison | walls 1.803, fringe 1.654, open tunnel 1.59 (t=31), lit unbounded 1.42 |
| cyl Re60 M0.2 compressible | cd 1.822 / 1.830 / 1.832 (256/512/1024²) |
| cyl Re800 M0.2 (impulsive era) | cd 1.71, St 0.2445 (incomp St 0.257) |
| cyl Re800 @512 quiet+kick, [62.8,94.3] | nscbc 1.776±0.180 / clAmp 1.33 / St 0.2499; sponge 1.894±0.176 / 1.85 / 0.2503 |
| cyl Re60 @512 nscbc | cd 1.813 / St 0.145± |
| cyl Re100 M0.2, D=32 cells, [90,153] | 8D 1.681/0.190, 16D 1.469/0.175, 32D 1.377/0.168, **pot 8D 1.365/0.165** |
| cyl Re100 incomp dr8, D=32 cells | plain 8D 1.736/0.192, **pot 8D 1.409/0.168**, truth 32D 1.403/0.168 |
| cyl Re2000-class 1024², [50,96] | pure 1.846±0.582 (clAmp 4.8), strip12 1.902±0.341 (2.4), sponge 2.087±0.370 (2.8) |
| sphere Re100 M0.2 dr6, nscbc | 64³ cd 1.10, 96³ cd 1.146 (walled-sponge era 1.226; lit ~1.09) |
| sphere Re300 128³ / Re500 192³, open BC | cd 0.681±0.006 (lit ~0.66) / cd 0.589±0.010 (lit ~0.55) |
| sphere STL 96³ Re100 M0.2 dr6 | cd 1.226, Afront 0.856 ≈ πR² |
| sphere STL vs analytic, Re300, D=12 cells | cd 0.710 vs 0.685 |
| TGV Re1600 peak dissipation vs spectral | 64³ −8.1%, 128³ +5.6%, 256³ −4.3%, 512³ +1.2% |
