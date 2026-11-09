# The compressible FV family

Explicit compressible DNS, LES and RANS on any `FaceMesh`, on the GPU. First phase
of "every solver on every mesh": no per-mesh fork, one solver, one loop.

| route | command | mesh |
|---|---|---|
| polyMesh | `dns flow --physics compressible --polymesh DIR --bc FILE.toml --re R --mach M --tt T` | any polyMesh `FaceMesh::fromPolyMesh` reads (`docs/mesh.md`) |
| O-grid | `dns flow --physics compressible --solver fv --ogrid --body BODY ...` | `fromCurvMesh` of the structured solver's own nodes, fluid rows only |
| box | `dns flow --physics compressible --solver fv --body none [--stretch B\|--focal BOX] ...` | `fromCurvMesh` of the uniform, stretched or focal mesh |

`dns flow --polymesh CASE` alone stays OpenFOAM simpleFoam (`docs/fv.md`),
byte for byte.

A flame: `dns flame --solver fv` and `dns flame --polymesh DIR --bc FILE`, "Species", "Diffusion flame".

Source: `src/dns/gpu/fv_comp.{hpp,hip}` (solver),
`src/dns/gpu/runner_fvcomp.{hpp,hip}` (loop), `src/dns/fvcomp_config.*` (the
`--bc` file, pure), `src/dns/fvcomp_case.*` (FvBc table, freestream),
`src/dns/fvraster.*` (frame plane). Physics: `src/dns/fvflux.hpp`.

## Scheme

The FvRef scheme (`docs/mesh.md`, "The FV face physics"), device instantiation
of the same templates, the same pass order:

    primitives -> boundary states -> cell gradients of u, v, w, T -> [mu_t]
      -> face fluxes F[5][nI + nB] (materialized) -> gather + stage

- **R in {float, double} from one source** (`FvCompImpl<R>`, `--fv-precision`,
  FP32 default). Tables are narrowed once; dt is an exact min; tallies and
  budgets are downloaded and summed in ascending index in FP64. Own-index writes,
  no FP atomics: reruns are byte-identical.
- **SSP-RK3**, fused: `out = ca a + cb b + cr R(b)`, `ca = 1 - cb` (Sterbenz).
  Three state buffers. Stage 3 writes `q1` and swaps, so an output never aliases
  `a`: a later rhs hook may re-read it, and no caller holds a raw device pointer
  across a step.
- **dt**: FvRef's Sigma forms per cell (`2 cfl V / Sum(|u.S| + c|S|)`,
  `diff rho V / (mu max(4/3, gamma/Pr) Sum corr.S)`), with `mu_t` at its own factor
  `max(4/3, gamma/Pr_t)`. Recomputed every `--dt-recompute` steps (16), margin
  0.95, as the structured loops. A bad cell is named: `FV state diverged: rho R p P
  at cell C` (exit 7).
- **Empty faces are compacted out** of the face tables and the CSR, which stays
  ascending. Device CSR entries carry the gather code `f << 1 | side`, the gradient
  operand (`other` cell or `~b`) and weights.
- **`--compensated`** (FP32): the structured two-word state, hi + lo per state
  word, `internal::combineWord2`, a compile-time kernel instantiation. See
  "FP32 on a steady flow". FP64 rejects it (exit 2).

**Quiet start.** `--ramp`: the IC at rest, the freestream raised by exact
Galilean boosts over `[0, ramp]` with the Far/Inflow rows of the table following,
as `runner_ogrid`. Default as the structured flow on `--solver fv`, 0 on a
polyMesh (name it: a mesh with walls wants it).

**LES.** `--les C --sgs smag|sigma`: `mu_t = rho C^2 Delta^2 op(grad u)`
(`sgsRate`, the structured operators), `Delta^2 = V^(2/3)`, 2D: cell area
(`curvDelta2`'s rule). Face mean into `mu`, `mu_t cp / Pr_t` into `kappa`, Pr_t
0.9 (the structured `prt_`). `grad u` is the FV cell gradient. A Wall face carries
the molecular stress only (`mu_t` odd across the wall), a Slip face the owner's
`mu_t`. Sigma is rejected on a 2D mesh by name (exit 2): `nz == 1`, or an `empty`
patch in the manifest. Label: `LES <model> C=...`.

**Forces.** Wall (and `force = true` Slip) faces, from the state at the row:
gauge pressure force `(p_P - p_inf) S`, viscous `-tau.S`, heat into the body.
FP64 sums. cd/cl/cs on `direction`/`lift`, `refArea` = `L x depth` on a 2D mesh,
`D x span` on an O-grid, `reference.area` on a 3D polyMesh (required with a
tally). Stats: `t,dt,cd,cl,cs,heat,machMax,cdf,clf,mass,energy,kinetic`.

**Frames.** An unstructured mesh has no raster, so every FV frame on any mesh is
a uniform plane (`FvRaster`): nearest centroid as a first guess, then a walk
across faces to the containing cell; value per pixel, no blending; mask 0 outside
the mesh, so a body is a hole. 2D: the mesh plane. 3D: the z mid-plane.
`--frame-box a,b,c,d`, `--frame-n N`. Channels: `vort` (2D) or `curl`, `speed` or
`dspeed`. `--emit stats|bin|png|web|shm`. Rejected by name: `volume`, `slices`,
`vtk`, `surface`, `mesh`, `spectrum`, `snapshot`, `error`.

## The `--bc` file

Read in phase 1.5 (the `--scene` precedent), strict: an unknown table, key or
kind, a wrong type, exit 3 by name. `--dry-run` reads it and the manifest, no GPU.

```toml
[reference]                # all optional
length = 1.0               # L in Re = rho U L / mu, the time unit of --tt
area = 0.5                 # force reference; default L x depth (2D); 3D: required with a tally
direction = [1, 0, 0]      # freestream
lift = [0, 1, 0]           # normal to direction

[patch.NAME]               # one per non-cyclic, non-empty mesh patch
type = "wall"              # wall | slip | inflow | outflow | far | empty | total | wedge
thermal = "isothermal"     # wall: adiabatic (default) | isothermal
twall = 1.1                # T_w / T_inf
u = [0, 0, 0]              # wall velocity / U
omega = [0, 0, 0]          # wall rotation
origin = [0, 0, 0]
force = true               # wall default true, slip default false
# inflow: u (default the freestream), t (T / T_inf)
# outflow: p (p / p_inf), tback (T / T_inf at backflow), exit = fixed | extrapolate | riemann
# total: p0, t0 (ratios), --shock hll only (docs/nozzle.md); dns flow needs [nozzle], dns flame needs y
```

Bound against the patch list: an unbound patch, a binding with no patch, a bound
cyclic (an internal face), `wedge` off the `--shock` route: exit 3 by name.
`examples/fvcomp_cyl2d_snappy.toml`. `[nozzle]` (`p`, `t`, `astar`, `axis`), `total`,
`wedge` and `exit` are the nozzle route's, `docs/nozzle.md`.

Box units: rho 1, U 1, `p = 1/(gamma M^2)`, `T = p`, `mu = L/Re`.

## CLI: what is rejected

`--solver fv`: allow-list in `src/dns/config/fv.cpp`, every other flag exit 2 by name: all
`--nscbc-*`, `--pot`, `--rotor-*`, `--thin-seal`,
`--penalize`, `--cut-flux`, `--temp`, `--scene`. An immersed body exit 2 (FV has no
immersed wall). The revolved O-grid exit 2 (P8). `--solver fv` dispatches on
`--physics`: `compressible` is this path, the default is `docs/fvincomp.md`. On
the polyMesh route likewise; `--re --mach --tt --bc` required, `--solver fv`
implied. `--physics` picks the `--bc` parser: an incompressible file's `kind`,
`[initial]` or `reference.u` is exit 3 by name. `--bc`, `--tt` off the polyMesh route, `--fv-precision`, `--frame-*`
off the FV routes: exit 2. Box faces are `Far`, `--noslip` y `Wall`, `--far
periodic` cyclic: the structured far-field machinery does not exist here, and
`--dry-run` says so in its `patches` line.

## RANS

`--rans sa-neg|sst-2003m|sst-vm` on every route with a wall: `--solver fv --ogrid`,
`--solver fv --body none --noslip`, `--polymesh DIR --bc FILE`. The closures are
`dns/rans.hpp`, the ONE home of every constant (SA-neg with ft2, SST-2003m,
SST-Vm); the algebra on a face is `dns/fvrans.hpp`, shared by FvRef and the device.
OpenFOAM's RAS of `gpu/fv_simple` is a separate model and is untouched. Label:
`RANS SA-neg -- URANS (time-accurate)` or `-- steady (local time stepping:
pseudo-time, not time-accurate)`, plus `[exact wall distance, scalars first-order
upwind]`. `--rans none`: no scalar row, no pass, no buffer.

**Passes.** State rows 5.. hold rho nu~ | rho k, rho omega. Per stage, after the
velocity and T gradients: specific scalars (SA: nu~, rho; SST: k, omega) -> their
boundary values -> their LSQ gradients -> `mu_t`, F1, cell terms (`kFvRansCell`)
-> the mean-flow faces (the LES instantiation: `mu_t` rides `nut`, Pr_t 0.9) ->
mean gather -> scalar faces -> scalar gather, point-implicit (`pointImplicit`,
`pointImplicitDelta` on the two-word state).

**Faces.** Convection first-order upwind of the specific scalar on `F[0]`, the
mean flow's own mass flux, so a uniform specific scalar 2^-13 stays exactly uniform
(R4). Coefficients are formed on the face: SA `rho_f (nu_f + nu~_f fn(chi_f))`,
SST `mu_f + sigma(F1_f) mu_t,f` with F1, `mu_t` face means. Gradient: the FV
corrected face gradient. Cell terms read FV cell gradients: `Omega`, `S` from the
velocity LSQ gradient; SA's cb2 and grad rho terms, SST's `grad k . grad omega`.

**Boundaries are face fluxes, as everywhere on this path.**

| kind | gradient's face value | scalar flux |
|---|---|---|
| Wall | nu~ 0, k 0, `omega_w = 60 mu_w/(beta1 rho_w d1^2)`; SA rho: `rho_w` | no convection; `-(mu_w/sigma) grad nu~.S`, `-mu_w grad(k, omega).S`: `mu_t` EXACTLY 0 on the face |
| Slip | the owner's | 0 |
| Inflow, Outflow, Far | upwind on the boundary velocity: the owner's leaving, `phi_inf` entering | `F[0] phi_b`, no diffusion |

`rho_w`: the EOS at `T_w`, isothermal; the owner's, adiabatic (the mean wall state
holds `rho_P` either way, so this is the scalars' own). `mu_w`: `bndViscous`'s.
`d1`: the owner cell's wall distance. **The wall face's mean-flow flux is the laminar
one, bit for bit, whatever `mu_t`** (R5).

**Wall distance.** `fvRansWallDistance` (`dns/fvcomp_case`): exact, nearest point on
the faces of the Wall-KIND patches (`--bc` decides, not the polyMesh type word),
FaceMesh centroid, `dns/fvwalldist`'s BVH. meshWave is not used: it refuses cyclics.
Every route the same. On an O-grid this is not `ogridWallDistance` (node-mean
centroid, discrete polyline): a truncation-order difference.

**dt.** FvRef's Sigma bound with `mu_t` at `max(4/3, gamma/Pr_t)`, SA also `rho
max(nu, K)/sigma` (`saDtDiffusivity`). SST's `mu_t` is the cell's own, not the
structured upper bound. A non-finite `mu_t` is a diverged cell.

**Guard.** SST `k >= 0`, `omega >= floor rho`: counted (`floorHits`, stats column),
never silent. `nu~` is never clipped. Stats append `mutMax`, SST `floorHits`.

**`--steady TOL`.** Local time stepping: `DeviceFvComp::localDt` makes `lts[c] =
dt_c/dtRef`, every stage writer (mean gather, force, scalar gather) reads `cr
lts[c]`; a compile-time kernel instantiation, off = the kernels bit for bit. FP32
marches the two-word state, as the structured solver. Stops on `steadyFloor`: cd,
cdf observables, kinetic energy and the scalars' integrals monitors. Exit 8 past
`--steady-steps`. Needs a tallied wall. LTS conserves `Sum V rho / lts`: the
channel's mass drifts (+1.4e-5 at ny 32 from the oracle, shrinking with h), and the
channel's steady equations are a one-parameter family in mass, so the oracle is
solved AT the march's mass. A certified `--steady` stop on an external flow is NOT
measured: below.

**The drive.** `FvCompConfig::bodyForce`: constant force per volume, `m + f`, `E +
f.u`, the oracle's channel force; a pass after the mean gather, none at 0. Library
only: no CLI route to a body force on this path.

### RANS verification

Agreement proves the port, not the physics. `gpu_fv_rans`
(`tests/test_gpu_fv_rans.cpp`), `tools/fv/rans_ladder.sh`:

| solver pair | mesh | quantity | expected relation | measured |
|---|---|---|---|---|
| R1 FV(FP64) vs FvRef, 3 closures | box_cyclic (adiabatic, isothermal moving walls); cyl2d_snappy (spinning isothermal wall, inflow, outflow, slip, far, Sutherland); SA both signs, force on | every RANS pass, mean fluxes, rhs, 3 steps, dt | rounding <= 1e-13 | rhs 1.5e-15, boundary values 7.1e-16, scalar gradients 8.4e-16, mu_t 4.6e-16, F1 1.5e-15, cell terms 4.2e-16, scalar fluxes 3.0e-16, mean fluxes 1.4e-15, steps 9.4e-16, dt 1.1e-16 |
| R2 FV(FP32) vs FP64 | same | rhs, 3 steps | <= 1e-3, 2e-5 | 1.2e-6, 4.7e-7 |
| R3 nu~ 0 / k 0 vs laminar FV | same, 3 closures, FP32, FP32 compensated, FP64 | mean rows after 20 steps, dt | bitwise | memcmp equal |
| R4 uniform specific scalar 2^-13, transport | same, FP32, FP64 | phi after 40 steps | bitwise | 0 |
| R5 RANS vs laminar wall face | same, 3 closures | wall F, all 5 rows; mass | bitwise; 0 | equal; 0 |
| R6 rerun | same, 3 closures, both precisions | 20 steps; floor hits | bytes equal; 0 | equal; 0 |
| R7 compensated vs FP64 | cyl2d_snappy, dt 2% of stable, 2000 steps | state | <= 0.2 x plain | 5.5e-7 / 5.4e-7 / 5.4e-7 (plain 1.3e-4 / 1.4e-4 / 1.0e-4) |
| FV(FP64) vs `rans_channel1d`, SA-neg | channel Re_tau 180, M 0.2, beta 2, ny 32 / 64 / 128, 4 x ny on a cyclic + empty FaceMesh | ub at the oracle at the march's mass | order 2 | -4.226e-3 / -1.049e-3 / -2.634e-4 (2.01, 1.99) |
| same, SST-2003m | same | ub | order 2 | -3.764e-3 / -9.048e-4 / -2.307e-4 (2.06, 1.97) |
| same, SST-Vm | same | ub | order 2 | -3.778e-3 / -9.079e-4 / -2.315e-4 (2.06, 1.97) |
| same, u+ profile, 3 closures | same | max over rows, of u_c | order 2 | SA 4.25e-3 / 1.06e-3 / 2.65e-4; 2003m 3.88e-3 / 9.35e-4 / 2.37e-4; Vm 3.90e-3 / 9.38e-4 / 2.38e-4 |
| same, scalar profile | same | nu~ (SA), k (SST), max of scalar | order 2 | SA 2.98e-3 / 6.64e-4 / 1.77e-4; 2003m 1.32e-2 / 3.25e-3 / 8.08e-4; Vm 1.33e-2 / 3.27e-3 / 8.11e-4 |
| same, wall stress | same | tau_w vs 1 | fixed point | 1.00000004 (the stop's tail, ub within 2e-8): the force balance, not a free number |
| same, SA-neg | Re_tau 1000, beta 3, ny 64 / 128 / 192 | ub | order 2 | -1.959e-3 / -4.886e-4 / -2.175e-4 (2.00, 2.00) |
| same, SST-2003m | Re_tau 1000, beta 3, ny 64 / 128 | ub | order 2 | -1.392e-3 / -3.292e-4 (2.08). ny 192 NOT MEASURED: the ORACLE stops at resid 3.1e-11 > tol 1e-11 in 400 iterations (exit 1): `test_gpu_fv_rans channel sst-2003m 1000 192 3` |
| R8 (`gpu_fv_rans_channel`) | Re_tau 180, ny 24 / 48, 3 closures | ub | order >= 1.5 | 1.98 / 1.98 / 2.04, hits 0 |
| FV vs structured O-grid, SA-neg, both compensated, URANS | Re 1e4, M 0.2, far 20 D, beta 4, 128x50 / 256x98 / 512x194, y+ max 1.8 / 0.84 / 0.41 | cd at t 8.283 (impulsive transient, not shedding) | order 2 | 0.863346, 0.867796 / 0.898745, 0.898583 / 0.905894, 0.905847: +4.45e-3 / -1.62e-4 / -4.7e-5. Sign change on rung 1, then 1.79. cd itself still moves 0.8% between the last two |
| same | same | cdf | none claimed | 0.0222309, 0.0222456 / 0.024095, 0.024106 / 0.0243343, 0.0243419: +1.5e-5 / +1.1e-5 / +7.6e-6: shrinks at ~0.5, order NOT established |
| same | same | mu_t max / mu | none claimed | 230.7, 228.8 / 276.6, 276.2 / 304.0, 303.9 |
| same, SST-2003m | same | cd at t 8.283 | order 2 | 0.802099, 0.880578 / 0.825073, 0.823702 / 0.799611, 0.799144: +7.85e-2 / -1.37e-3 / -4.67e-4. Rung 1 is off the asymptote (y+ max 1.7, the two disagree 10%); last pair order 1.55. cd itself non-monotone: not asymptotic |
| FV vs structured O-grid `--steady 1e-3`, SA-neg | Re 1e3, M 0.2, 128x50, beta 4 | stop; cd, cdf mean, last half of 1e6 iterations | converged limit | NEITHER certifies: both exit 8, cd jitter 3-5%, no steady wake (the known LTS sawtooth). Means, of a limit cycle, not a limit: structured 1.0831, 0.09662; FV 1.0461, 0.09394 |
| `--rans none` FV and every other path vs baseline binary | `comp_bit_identity.sh`: 10 runs, stats and fields | bytes | bitwise | identical, files non-empty (0 failures) |

Order 1.5 in R8 is a gate set AFTER the ladder (X4's value), not a measured margin.
The channel starts FROM the oracle: it pins the fixed point, not a cold start. FV's
ub sits below the oracle at every rung: on a stretched mesh LSQ and central cell
gradients differ, and the wall cell's LSQ gradient is one-sided where the oracle's
ghost is an image. That difference converges at 2, so it is truncation, not a gap.
FP64 only: an FP32 channel is not on this table.


Compared with care: `gpu_sst`, `gpu_sa_wedge`, `gpu_precond` failed at baseline
(issues 78(h)) and were not read as oracles.

## Combinations

Every module of the stage (turbulence, thickening, chemistry, gravity, the drive, local time
stepping, precision) is a set of ADDITIVE face / cell terms or a stage-writer switch on the one
`gatherStage` (mean rows, gravity, drive, species, scalars). `dns/fvmodules` is the inventory:
each module as data, ONE decision per pair (live | refused + error text + reason), the
requirements (TFLES needs C0, Charlette needs LES and TFLES). `FvRef::make` (`fvModulesOf`),
`DeviceFvComp::make` (the precision axis), `DeviceFvComp::localDt` (the time axis) and the CLI's
flag pairs (`fvCompCommon`: `--compensated` + `fp64`, `--les` + `--steady`; `fvRansCheck`:
`--rans` + `--les`) read it; nothing restates a pair. A pair goes live when the generic gate
matrix passes its cell, in the same commit as its physics number.

- **Generic gates** (`gpu_fv_matrix`, `tests/test_gpu_fv_matrix.cpp`; host part `test_fv_comp`
  `inventoryGate`: every model pair FvRef::make can be configured with is decided exactly as the
  inventory, with its text; every live FvRef cell byte-identical at 1 and 4 threads). A cell is a
  module, or a pair, closed under the requirements (`tests/fv_matrix_util.hpp`: one FvConfig per
  cell, fixed strengths); precision runs inside every cell. Meshes: uniform periodic box 8x6x5,
  box_cyclic (y walls), box_refined (split faces, Sutherland), cyl2d_snappy (wall, inflow,
  outflow, slip, far; gravity runs it with the hydrostatic ambient's rows, and its rest gate on the
  same set at rest: "Hydrostatic ambient").
  (a) inert = without, bitwise: passive species, TFLES F 1 (no LES: with it Omega still switches
  the SGS scalar flux off), RANS scalars 0; (b) rest (hydrostatic) rhs = the drive to rounding,
  uniform power-of-two Y exact (FP32, compensated, FP64), mass and E + q rho Y_F + rho Phi
  telescope, Z == tracer, C2b with C(xi) == 1 = C1 byte for byte; (c) FP64 vs FvRef 1e-13 every pass and 3 steps, FP32 vs FP64; (d)
  Source / Morton / ColourMajor, 3 FP64 steps, to summation order (test_fvref E4: the CSR order
  moves with the numbering); (e) rerun byte-identical, plain and compensated. 48 cells, 4
  meshes, ~15 s.
- **Matrix states are UNTIED** (1e-3 hashed noise on E). The TFLES dynamic sensor's T-gate is a
  switch, and mirror-symmetric fields on a uniform patch tie neighbouring T exactly. Measured on
  the TIED state (`tiedSensor`, TFLES dynamic + C0, 3 steps): numberings at rounding on every
  mesh (<= 4.6e-16); FP32 vs FP64 2-5e-7 on three meshes, 1.8e-4 on box_refined (a tie flipped
  by FP32 rounding). Untied, every cell's FP32 vs FP64 is <= 5.3e-6.
- **Found and fixed by (d): the TFLES walk depended on the cell numbering on split faces.**
  `fvTflesWalk` took the first of exactly tied sub-faces in CSR order: box_refined ColourMajor vs
  Source, TFLES dynamic, 2.3e-3 of the state in 3 steps. Ties now go to the lowest `srcFace`:
  every numbering <= 8.6e-16 over all 43 cells (uniform meshes have no tie: no number moved
  there; X1, L5 of `gpu_fv_flame_les` reproduce their table rows).

Turned on in MC (each with its physics number, `gpu_fv_matrix_phys`, FP64 unless stated):

| Pair | Physics | Measured |
|---|---|---|
| species + gravity (C0, C1, TFLES) | buoyancy is rho(T) g through the EOS: no Boussinesq term; closed patches only | hot product pocket (T 1.5 T0, passive marker) in a hydrostatic 64x64 walled box, g 0.4: marker centroid rise 8.94e-3 at t 1 (925 steps); g 0: 1.6e-15; early acceleration 0.0689 = 0.861 x the 2D added-mass estimate g (rho_a - rho_p)/(rho_a + rho_p) |
| gravity + LES (smag, sigma) | resolved buoyancy is resolved; the SGS model carries no buoyancy correction (Lilly / Richardson): a stated model gap, the usual LES-of-convection closure | mu_t at the hydrostatic rest exactly 0; same pocket, Smagorinsky 0.17 (sigma: the matrix only): max mu_t / mu 0.002, rise 0.9994 x DNS |
| gravity + bodyForce | two additive momentum sources | plane Poiseuille across g (x cyclic, y walls, g -0.4, f 0.004, mu 0.05, M 0.008) from the hydrostatic rest to t 2 H^2/mu: max abs(u - f y (H - y)/(2 mu)) / u_c 3.9e-3 / 9.8e-4 at ny 16 / 32 (order 2.00) |
| species + bodyForce | the drive is per volume, species ride the mass flux | the same channel with a passive species: u bitwise the species-free run, uniform Y 0.25, 0.5 exact (FP64) |
| C1 + LES (smag, sigma) | `kFvSpFace<R, true, true, Eddy>`: one rho D + nut_f/Sc_t and one limiter for every species, so Z(Y) is linear in Y | two periodic sheared Z sheets on a wavy 96x96 periodic mesh, burning, FP32, t 2 (1316 steps): max abs(Z(Y) - tracer) DNS 9.5e-6, Smagorinsky 0.17 9.1e-6, both inside the FP32 walk 3.0e-5; Z variance LES / DNS 0.9982 (the eddy flux mixes) |

Turned on in M2 ("Air-methane"): C2b (`FvChem::law`) with LES smag / sigma (3D laminar LES vs
DNS: sigma 0, Smagorinsky 0.17 +5.3e-4 in S_L), gravity and bodyForce (harness only: the law is a
rate factor on C1's transport); C2b + TFLES refused (premixed thickening only).

Kept refused, by physics: species + RANS (a mean reaction rate needs a closure: Arrhenius of the
mean state is not the mean rate), RANS + gravity (no buoyancy production in the closures: an
unstable stratification makes no k or nu~, a RANS convection cell would decay toward laminar),
LES + `--steady` (LES is time-accurate; the structured validator refuses the same pair;
before MC the polyMesh route ACCEPTED `--les` with `--steady` and ran it: now exit 2; the structured-mesh FV route already refused it in `validateFlow`), TFLES with RANS or C1.
By code: species or gravity under LTS (`kFvChem`, `kFvGravity` take the bare cr). By route
(no CLI spelling yet; the case surface is M6): gravity + LES and gravity + bodyForce on `dns
conv`, two-reactant LES on the 2D jet (`--nz 1`, LES needs `--nz >= 8`). `dns flame --gravity`
is the candle's (M5a, "Hydrostatic ambient"): gravity x open patches and gravity x C1 live.

<!-- generated: dns::fvCombinationsMarkdown (src/dns/fvmodules.cpp); test_gpu_fv_matrix markdown -->
| Module | Axis | Spelling |
|---|---|---|
| LES smag | turbulence | `--les Cs [--sgs smag]` |
| LES sigma | turbulence | `--les C --sgs sigma` |
| TFLES uniform | flame | `--tfles F --tfles-sensor uniform` |
| TFLES dynamic | flame | `--tfles F [--tfles-sensor dynamic]` |
| Charlette | efficiency | `--efficiency charlette` |
| RANS SA | turbulence | `--rans sa-neg` |
| RANS SST | turbulence | `--rans sst-2003m, --rans sst-vm` |
| C0 | chemistry | `dns flame --mode premixed` |
| C1 | chemistry | `dns flame --mode diffusion; --equivalence PHI --sl-law onestep (C2)` |
| C2b | chemistry | `dns flame --equivalence PHI [--sl-law gri30] [--mix-phi PHI]` |
| gravity | gravity | `dns conv; dns flame --mode candle (FvConfig::gravity; open patches: FvConfig::ambient rows)` |
| bodyForce | drive | `the channel's drive (FvConfig::bodyForce)` |
| turb | inflow | `dns flame --turb U --solver fv (FvConfig::turb)` |
| steady LTS | time | `--steady TOL (DeviceFvComp::localDt)` |
| compensated | precision | `--compensated` |
| FP64 | precision | `--fv-precision fp64` |
| shock | capture | `--shock hll|sense --polymesh` |

Pair matrix: L live, R refused, x one axis (exclusive).

|   | LES smag | LES sigma | TFLES uniform | TFLES dynamic | Charlette | RANS SA | RANS SST | C0 | C1 | C2b | gravity | bodyForce | turb | steady LTS | compensated | FP64 | shock |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LES smag | . | x | L | L | L | x | x | L | L | L | L | L | L | R | L | L | R |
| LES sigma | x | . | L | L | L | x | x | L | L | L | L | L | L | R | L | L | R |
| TFLES uniform | L | L | . | x | L | R | R | L | R | R | L | L | L | R | L | L | R |
| TFLES dynamic | L | L | x | . | L | R | R | L | R | R | L | L | L | R | L | L | R |
| Charlette | L | L | L | L | . | R | R | L | R | R | L | L | L | R | L | L | R |
| RANS SA | x | x | R | R | R | . | x | R | R | R | R | L | R | L | L | L | R |
| RANS SST | x | x | R | R | R | x | . | R | R | R | R | L | R | L | L | L | R |
| C0 | L | L | L | L | L | R | R | . | x | x | L | L | L | R | L | L | L |
| C1 | L | L | R | R | R | R | R | x | . | x | L | L | R | R | L | L | R |
| C2b | L | L | R | R | R | R | R | x | x | . | L | L | L | R | L | L | R |
| gravity | L | L | L | L | L | R | R | L | L | L | . | L | R | R | L | L | R |
| bodyForce | L | L | L | L | L | L | L | L | L | L | L | . | R | L | L | L | R |
| turb | L | L | L | L | L | R | R | L | R | L | R | R | . | R | L | L | R |
| steady LTS | R | R | R | R | R | L | L | R | R | R | R | L | R | . | L | L | R |
| compensated | L | L | L | L | L | L | L | L | L | L | L | L | L | L | . | x | L |
| FP64 | L | L | L | L | L | L | L | L | L | L | L | L | L | L | x | . | L |
| shock | R | R | R | R | R | R | R | L | R | R | R | R | R | R | L | L | . |

| Pair | Decision | Error text | Why |
|---|---|---|---|
| LES smag + RANS SA | exclusive | `FV: LES and RANS put two eddy viscosities in one slot` | mu_t has one slot (nut); one model owns it |
| LES smag + RANS SST | exclusive | `FV: LES and RANS put two eddy viscosities in one slot` | mu_t has one slot (nut); one model owns it |
| LES smag + steady LTS | refused | `FV LES with local time stepping: LES is time-accurate (pseudo-time has no filtered field)` | an LES statistic is a time average of a filtered unsteady field; the structured validator refuses the same pair |
| LES smag + shock | refused | `FV --shock with LES: not built (no gate)` | no nozzle rung runs LES; the background term has no filter width |
| LES sigma + RANS SA | exclusive | `FV: LES and RANS put two eddy viscosities in one slot` | mu_t has one slot (nut); one model owns it |
| LES sigma + RANS SST | exclusive | `FV: LES and RANS put two eddy viscosities in one slot` | mu_t has one slot (nut); one model owns it |
| LES sigma + steady LTS | refused | `FV LES with local time stepping: LES is time-accurate (pseudo-time has no filtered field)` | an LES statistic is a time average of a filtered unsteady field; the structured validator refuses the same pair |
| LES sigma + shock | refused | `FV --shock with LES: not built (no gate)` | no nozzle rung runs LES; the background term has no filter width |
| TFLES uniform + RANS SA | refused | `FV TFLES with RANS: the thickened flame is a resolved (DNS / LES) flame` | TFLES thickens a flame the grid carries; a RANS mean flame has none |
| TFLES uniform + RANS SST | refused | `FV TFLES with RANS: the thickened flame is a resolved (DNS / LES) flame` | TFLES thickens a flame the grid carries; a RANS mean flame has none |
| TFLES uniform + C1 | refused | `FV TFLES: the thickened flame is premixed species only (FvChem ox < 0)` | the thickening and its sensor are the premixed step's (kChemCellPass<true>) |
| TFLES uniform + C2b | refused | `FV TFLES with --equivalence: the thickened flame is premixed species only (FvChem ox < 0)` | the thickening, its sensor (the one-step rate at Ta/2) and Charlette's E are the premixed step's; a two-reactant TFLES would thicken the partially premixed jet's outer diffusion sheet too, which no model here defines (the structured --equivalence refuses --tfles alike) |
| TFLES uniform + steady LTS | refused | `FV local time stepping with species: not built` | kFvChem takes the bare cr: under LTS the fixed point would move with lts |
| TFLES uniform + shock | refused | `FV --shock with TFLES: not built (no gate)` | no gate thickens a flame under the hybrid flux (Shock x C0 is gated on the laminar flame) |
| TFLES dynamic + RANS SA | refused | `FV TFLES with RANS: the thickened flame is a resolved (DNS / LES) flame` | TFLES thickens a flame the grid carries; a RANS mean flame has none |
| TFLES dynamic + RANS SST | refused | `FV TFLES with RANS: the thickened flame is a resolved (DNS / LES) flame` | TFLES thickens a flame the grid carries; a RANS mean flame has none |
| TFLES dynamic + C1 | refused | `FV TFLES: the thickened flame is premixed species only (FvChem ox < 0)` | the thickening and its sensor are the premixed step's (kChemCellPass<true>) |
| TFLES dynamic + C2b | refused | `FV TFLES with --equivalence: the thickened flame is premixed species only (FvChem ox < 0)` | the thickening, its sensor (the one-step rate at Ta/2) and Charlette's E are the premixed step's; a two-reactant TFLES would thicken the partially premixed jet's outer diffusion sheet too, which no model here defines (the structured --equivalence refuses --tfles alike) |
| TFLES dynamic + steady LTS | refused | `FV local time stepping with species: not built` | kFvChem takes the bare cr: under LTS the fixed point would move with lts |
| TFLES dynamic + shock | refused | `FV --shock with TFLES: not built (no gate)` | no gate thickens a flame under the hybrid flux (Shock x C0 is gated on the laminar flame) |
| Charlette + RANS SA | refused | `FV TFLES efficiency (Charlette) with RANS: it reads an LES sub-grid velocity` | Charlette's u' is the filtered field's (Colin's operator), RANS has none |
| Charlette + RANS SST | refused | `FV TFLES efficiency (Charlette) with RANS: it reads an LES sub-grid velocity` | Charlette's u' is the filtered field's (Colin's operator), RANS has none |
| Charlette + C1 | refused | `FV TFLES: the thickened flame is premixed species only (FvChem ox < 0)` | the thickening and its sensor are the premixed step's (kChemCellPass<true>) |
| Charlette + C2b | refused | `FV TFLES with --equivalence: the thickened flame is premixed species only (FvChem ox < 0)` | the thickening, its sensor (the one-step rate at Ta/2) and Charlette's E are the premixed step's; a two-reactant TFLES would thicken the partially premixed jet's outer diffusion sheet too, which no model here defines (the structured --equivalence refuses --tfles alike) |
| Charlette + steady LTS | refused | `FV local time stepping with species: not built` | kFvChem takes the bare cr: under LTS the fixed point would move with lts |
| Charlette + shock | refused | `FV --shock with TFLES: not built (no gate)` | no gate thickens a flame under the hybrid flux (Shock x C0 is gated on the laminar flame) |
| RANS SA + C0 | refused | `FV species with RANS: not built (no turbulent scalar flux, two users of rows 5..)` | a RANS flame needs a mean reaction-rate closure (Arrhenius of the mean state is not the mean rate) and a turbulent scalar flux; neither is built |
| RANS SA + C1 | refused | `FV species with RANS: not built (no turbulent scalar flux, two users of rows 5..)` | a RANS flame needs a mean reaction-rate closure (Arrhenius of the mean state is not the mean rate) and a turbulent scalar flux; neither is built |
| RANS SA + C2b | refused | `FV species with RANS: not built (no turbulent scalar flux, two users of rows 5..)` | a RANS flame needs a mean reaction-rate closure (Arrhenius of the mean state is not the mean rate) and a turbulent scalar flux; neither is built |
| RANS SA + gravity | refused | `FV RANS with gravity: not built (no buoyancy term in the closures)` | the closures carry no buoyancy production: an unstable stratification produces no k or nu~, so a RANS convection cell (gravity's customer, dns conv) would decay toward laminar. Wrong physics, not a small gap |
| RANS SA + turb | refused | `FV --turb with RANS: synthetic fluctuations on an ensemble mean` | a RANS field is the ensemble mean: resolved fluctuations at its inflow are no RANS input (that is k and omega / nu~ at the patch), and the RANS flame is refused anyway |
| RANS SA + shock | refused | `FV --shock with RANS: not built (no gate; the scalars ride F[0])` | F[0] would carry the hybrid mass flux under the scalars: ungated |
| RANS SST + C0 | refused | `FV species with RANS: not built (no turbulent scalar flux, two users of rows 5..)` | a RANS flame needs a mean reaction-rate closure (Arrhenius of the mean state is not the mean rate) and a turbulent scalar flux; neither is built |
| RANS SST + C1 | refused | `FV species with RANS: not built (no turbulent scalar flux, two users of rows 5..)` | a RANS flame needs a mean reaction-rate closure (Arrhenius of the mean state is not the mean rate) and a turbulent scalar flux; neither is built |
| RANS SST + C2b | refused | `FV species with RANS: not built (no turbulent scalar flux, two users of rows 5..)` | a RANS flame needs a mean reaction-rate closure (Arrhenius of the mean state is not the mean rate) and a turbulent scalar flux; neither is built |
| RANS SST + gravity | refused | `FV RANS with gravity: not built (no buoyancy term in the closures)` | the closures carry no buoyancy production: an unstable stratification produces no k or nu~, so a RANS convection cell (gravity's customer, dns conv) would decay toward laminar. Wrong physics, not a small gap |
| RANS SST + turb | refused | `FV --turb with RANS: synthetic fluctuations on an ensemble mean` | a RANS field is the ensemble mean: resolved fluctuations at its inflow are no RANS input (that is k and omega / nu~ at the patch), and the RANS flame is refused anyway |
| RANS SST + shock | refused | `FV --shock with RANS: not built (no gate; the scalars ride F[0])` | F[0] would carry the hybrid mass flux under the scalars: ungated |
| C0 + steady LTS | refused | `FV local time stepping with species: not built` | kFvChem takes the bare cr: under LTS the fixed point would move with lts |
| C1 + turb | refused | `FV --turb with model C1 / C2 (--mode diffusion, --sl-law onestep): not built (the turbulent box is measured on C0 and C2b)` | the slot jet takes no --turb on any solver; the one-step C2 premix is C1's stage and has no turbulent number |
| C1 + steady LTS | refused | `FV local time stepping with species: not built` | kFvChem takes the bare cr: under LTS the fixed point would move with lts |
| C1 + shock | refused | `FV --shock with two reactants: not built (no gate; docs/nozzle.md "Reacting")` | Shock x C0 is gated (masterplan_combnozzle G3.1-G3.3); the two-reactant step's shared limiter under the hybrid mass flux is not (P8) |
| C2b + steady LTS | refused | `FV local time stepping with species: not built` | kFvChem takes the bare cr: under LTS the fixed point would move with lts |
| C2b + shock | refused | `FV --shock with --equivalence (C2b): not built (no gate; docs/nozzle.md "Reacting")` | Shock x C0 is gated (masterplan_combnozzle G3.1-G3.3); the two-reactant step's shared limiter under the hybrid mass flux is not (P8) |
| gravity + turb | refused | `FV --turb with gravity: not built (the box's inflow is an open patch)` | gravity needs a closed domain (no hydrostatic ambient on open patches); --turb needs an open inflow |
| gravity + steady LTS | refused | `FV local time stepping with gravity: not built` | kFvGravity takes the bare cr: under LTS the fixed point would move with lts |
| gravity + shock | refused | `FV --shock with gravity: not built (no consumer)` | no rung has both; the hydrostatic balance is unproven under D3 |
| bodyForce + turb | refused | `FV --turb with a body force: not built (the drive's channel has no inflow)` | the box is swept through an inflow patch; the driven channel is closed |
| bodyForce + shock | refused | `FV --shock with a body force: not built (no consumer)` | the driven channel is subsonic: nothing to capture |
| turb + steady LTS | refused | `FV --turb with local time stepping: the swept inflow is time-accurate` | the rows follow X = sum U dt; pseudo-time has no physical sweep |
| turb + shock | refused | `FV --shock with --turb: not built (no consumer)` | --turb is the flags' flame box's (--solver fv); --shock is the polyMesh route's |
| steady LTS + shock | refused | `FV --shock with local time stepping: not built (the sensor flickers; the structured steady march refuses it too)` | a frozen or live sensor both left the host LTS residual flat at 3.7e-3 on a Total inlet (issues.md Z7); the time march is the route |
| compensated + FP64 | exclusive | `--compensated is the FP32 state's low word: FP64 has no update floor to lift` | the two-word state lifts FP32's update floor; FP64 has none |
| turb without C0 or C2b | requirement | `FV --turb: the premixed flame box (C0, C2b; dns flame --mode premixed --solver fv)` | |
| TFLES uniform without C0 | requirement | `FV TFLES: the thickened flame is premixed species only (FvChem ox < 0)` | |
| TFLES dynamic without C0 | requirement | `FV TFLES: the thickened flame is premixed species only (FvChem ox < 0)` | |
| Charlette without TFLES uniform or TFLES dynamic | requirement | `FV TFLES efficiency (Charlette) is an option of the thickened flame: needs TFLES` | |
| Charlette without LES smag or LES sigma | requirement | `FV TFLES efficiency (Charlette) needs LES: the sub-grid velocity of a filtered field` | |

| Live pair | Hand gates besides gpu_fv_matrix |
|---|---|
| LES smag + TFLES uniform | gpu_fv_flame_les L1-L5, X1 |
| LES smag + TFLES dynamic | gpu_fv_flame_les L1-L5, X1 |
| LES smag + Charlette | gpu_fv_flame_les L1-L5, X1 |
| LES smag + C0 | gpu_fv_flame_les L1-L5, X1 |
| LES smag + C2b | gpu_fv_flame_rich_les (3D laminar LES vs DNS) |
| LES smag + turb | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| LES smag + FP64 | gpu_fv_cross X2 |
| LES sigma + TFLES uniform | gpu_fv_flame_les L1-L5, X1 |
| LES sigma + TFLES dynamic | gpu_fv_flame_les L1-L5, X1 |
| LES sigma + Charlette | gpu_fv_flame_les L1-L5, X1 |
| LES sigma + C0 | gpu_fv_flame_les L1-L5, X1 |
| LES sigma + C2b | gpu_fv_flame_rich_les (3D laminar LES vs DNS) |
| LES sigma + turb | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| LES sigma + FP64 | gpu_fv_cross X2 |
| TFLES uniform + Charlette | gpu_fv_flame_les L1-L5, X1 |
| TFLES uniform + C0 | gpu_fv_flame_les L1-L5, X1 |
| TFLES uniform + turb | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| TFLES dynamic + Charlette | gpu_fv_flame_les L1-L5, X1 |
| TFLES dynamic + C0 | gpu_fv_flame_les L1-L5, X1 |
| TFLES dynamic + turb | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| Charlette + C0 | gpu_fv_flame_les L1-L5, X1 |
| Charlette + turb | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| RANS SA + bodyForce | gpu_fv_rans R8 (the channel) |
| RANS SA + steady LTS | gpu_fv_rans |
| RANS SA + compensated | gpu_fv_rans |
| RANS SA + FP64 | gpu_fv_rans |
| RANS SST + bodyForce | gpu_fv_rans R8 (the channel) |
| RANS SST + steady LTS | gpu_fv_rans |
| RANS SST + compensated | gpu_fv_rans |
| RANS SST + FP64 | gpu_fv_rans |
| C0 + turb | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| C0 + compensated | gpu_fv_flame S3, S4 |
| C0 + FP64 | gpu_fv_flame S3, S4 |
| C0 + shock | gpu_fv_matrix, gpu_fv_matrix_phys (G3.3), fvref_shockspecies (G3.2), gpu_fv_flame_shock |
| C1 + gravity | gpu_fv_candle_rest, _g0, _height (the candle vs structured) |
| C1 + compensated | gpu_fv_flame_jet |
| C1 + FP64 | gpu_fv_flame_jet |
| C2b + turb | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| C2b + compensated | gpu_fv_flame_rich R1, R2, gpu_fv_flame_rich1d |
| C2b + FP64 | gpu_fv_flame_rich R1, R2, gpu_fv_flame_rich1d |
| gravity + compensated | gpu_fv_conv |
| gravity + FP64 | gpu_fv_conv |
| bodyForce + steady LTS | gpu_fv_rans |
| turb + compensated | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| turb + FP64 | gpu_fv_turb (IC, rows, trajectory X), tools/chem/turb_fv.sh comp, pairs (S_c, area vs structured) |
| compensated + shock | gpu_fv_nozzle |
| FP64 + shock | gpu_fv_nozzle |

<!-- generated: test_gpu_fv_matrix markdown (tests/test_gpu_fv_matrix.cpp) -->
Harness rows (`gpu_fv_matrix`, 57 live cells on the meshes box 8x6x5 periodic, box_cyclic, box_refined, cyl2d_snappy; gravity on the open set through the hydrostatic ambient's rows). oracle: double vs FvRef, max relative over every pass, 3 steps; orders: Source / Morton / ColourMajor, 3 FP64 steps; rest: |rhs - source| / (p sum|S| / V), FP64 / FP32 on the closed set, open rest on the open set at rest (gravity, cyl2d_snappy: Far, Outflow, Inflow faces); mass, energy: 10 FP64 steps; Z: max|Z(Y) - tracer|, FP64 / FP32, 20 steps. Steady LTS cells: the oracle is the rhs and stableDt (FvRef has no LTS), mass n/a (LTS drifts it). Energy n/a with the drive (its work).

| Cell | oracle | f32 vs f64 | orders | rest FP64 / FP32 | open rest | mass | energy | pow2 Y | Z == tracer | inert | rerun |
|---|---|---|---|---|---|---|---|---|---|---|---|
| laminar | 4.7e-15 | 4.3e-06 | 4.2e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.6e-15 | 1.3e-15 | n/a | n/a | n/a | byte-identical |
| LES smag | 4.4e-15 | 4.3e-06 | 4.7e-16 | 6.7e-17 / 3.3e-08 | n/a | 7.8e-16 | 6.7e-16 | n/a | n/a | n/a | byte-identical |
| LES sigma | 1.3e-14 | 5.3e-06 | 6.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 8.9e-16 | 1.1e-15 | n/a | n/a | n/a | byte-identical |
| RANS SA | 3.8e-15 | 3.6e-06 | 3.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.4e-15 | 4.4e-16 | n/a | n/a | RANS scalars 0: bitwise | byte-identical |
| RANS SST | 3.3e-15 | 4.7e-06 | 4.7e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.6e-15 | 6.7e-16 | n/a | n/a | RANS scalars 0: bitwise | byte-identical |
| C0 | 4.8e-15 | 2.7e-06 | 4.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | 7.8e-16 | exact | n/a | passive species: bitwise | byte-identical |
| LES smag + C0 | 4.7e-15 | 3.2e-06 | 4.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 6.7e-16 | 7.8e-16 | exact | n/a | passive species: bitwise | byte-identical |
| LES sigma + C0 | 1.3e-14 | 2.7e-06 | 4.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | 8.9e-16 | exact | n/a | passive species: bitwise | byte-identical |
| TFLES uniform + C0 | 4.8e-15 | 2.7e-06 | 3.0e-16 | 6.7e-17 / 3.3e-08 | n/a | 2.0e-15 | 8.9e-16 | exact | n/a | TFLES F 1: bitwise | byte-identical |
| LES smag + TFLES uniform + C0 | 4.7e-15 | 3.2e-06 | 3.4e-16 | 6.7e-17 / 3.3e-08 | n/a | 6.7e-16 | 4.4e-16 | exact | n/a | n/a | byte-identical |
| LES sigma + TFLES uniform + C0 | 1.3e-14 | 2.7e-06 | 4.6e-16 | 6.7e-17 / 3.3e-08 | n/a | 2.0e-15 | 8.9e-16 | exact | n/a | n/a | byte-identical |
| TFLES dynamic + C0 | 4.8e-15 | 2.7e-06 | 3.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 4.4e-16 | 8.9e-16 | exact | n/a | TFLES F 1: bitwise | byte-identical |
| LES smag + TFLES dynamic + C0 | 4.7e-15 | 3.2e-06 | 3.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 4.4e-16 | 1.7e-15 | exact | n/a | n/a | byte-identical |
| LES sigma + TFLES dynamic + C0 | 1.3e-14 | 2.7e-06 | 4.6e-16 | 6.7e-17 / 3.3e-08 | n/a | 4.4e-16 | 1.0e-15 | exact | n/a | n/a | byte-identical |
| LES smag + TFLES uniform + Charlette + C0 | 4.7e-15 | 3.2e-06 | 3.4e-16 | 6.7e-17 / 3.3e-08 | n/a | 8.9e-16 | 1.3e-15 | exact | n/a | n/a | byte-identical |
| LES sigma + TFLES uniform + Charlette + C0 | 1.3e-14 | 2.7e-06 | 3.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 2.0e-15 | 1.6e-15 | exact | n/a | n/a | byte-identical |
| LES smag + TFLES dynamic + Charlette + C0 | 4.7e-15 | 3.2e-06 | 3.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 6.7e-16 | 1.7e-15 | exact | n/a | n/a | byte-identical |
| C1 | 4.8e-15 | 2.7e-06 | 4.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 6.7e-16 | 1.8e-15 | exact | 1.4e-15 / 6.9e-07 | passive species: bitwise | byte-identical |
| LES smag + C1 | 4.7e-15 | 3.2e-06 | 3.2e-16 | 6.7e-17 / 3.3e-08 | n/a | 3.1e-15 | 2.2e-15 | exact | 1.3e-15 / 7.0e-07 | passive species: bitwise | byte-identical |
| LES sigma + C1 | 1.3e-14 | 2.7e-06 | 4.5e-16 | 6.7e-17 / 3.3e-08 | n/a | 4.4e-16 | 1.8e-15 | exact | 1.4e-15 / 6.9e-07 | passive species: bitwise | byte-identical |
| C2b | 4.8e-15 | 2.7e-06 | 3.2e-16 | 6.7e-17 / 3.3e-08 | n/a | 2.2e-16 | 8.9e-16 | exact | 1.4e-15 / 7.4e-07 | passive species, C(xi) 1 = C1: bitwise | byte-identical |
| LES smag + C2b | 4.7e-15 | 3.2e-06 | 4.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.0e-15 | 1.3e-15 | exact | 1.1e-15 / 8.4e-07 | passive species, C(xi) 1 = C1: bitwise | byte-identical |
| LES sigma + C2b | 1.3e-14 | 2.7e-06 | 3.2e-16 | 6.7e-17 / 3.3e-08 | n/a | 6.7e-16 | 8.9e-16 | exact | 1.4e-15 / 7.4e-07 | passive species, C(xi) 1 = C1: bitwise | byte-identical |
| gravity | 4.7e-15 | 4.3e-06 | 3.2e-16 | 3.4e-16 / 1.5e-07 | 3.4e-16 / 1.5e-07 | 1.4e-15 | 4.4e-16 | n/a | n/a | n/a | byte-identical |
| LES smag + gravity | 4.4e-15 | 4.3e-06 | 3.2e-16 | 3.4e-16 / 1.5e-07 | 3.4e-16 / 1.5e-07 | 8.9e-16 | 1.1e-15 | n/a | n/a | n/a | byte-identical |
| LES sigma + gravity | 1.3e-14 | 5.3e-06 | 3.7e-16 | 3.4e-16 / 1.5e-07 | 3.4e-16 / 1.5e-07 | 1.4e-15 | 4.4e-16 | n/a | n/a | n/a | byte-identical |
| C0 + gravity | 4.8e-15 | 2.4e-06 | 4.0e-16 | 3.4e-16 / 1.5e-07 | 3.4e-16 / 1.5e-07 | 1.1e-15 | 4.4e-16 | exact | n/a | passive species: bitwise | byte-identical |
| TFLES uniform + C0 + gravity | 4.8e-15 | 2.4e-06 | 3.3e-16 | 1.0e-15 / 4.5e-07 | 1.0e-15 / 4.5e-07 | 4.4e-16 | 8.9e-16 | exact | n/a | TFLES F 1: bitwise | byte-identical |
| TFLES dynamic + C0 + gravity | 4.8e-15 | 2.4e-06 | 3.2e-16 | 5.4e-16 / 2.4e-07 | 5.4e-16 / 2.4e-07 | 1.7e-15 | 2.0e-15 | exact | n/a | TFLES F 1: bitwise | byte-identical |
| LES smag + TFLES uniform + Charlette + C0 + gravity | 4.4e-15 | 2.8e-06 | 3.1e-16 | 1.0e-15 / 4.5e-07 | 1.0e-15 / 4.5e-07 | 8.9e-16 | 4.4e-16 | exact | n/a | n/a | byte-identical |
| C1 + gravity | 4.8e-15 | 2.4e-06 | 4.5e-16 | 3.4e-16 / 1.5e-07 | 3.4e-16 / 1.5e-07 | 4.4e-16 | 8.9e-16 | exact | 1.7e-15 / 6.9e-07 | passive species: bitwise | byte-identical |
| C2b + gravity | 4.8e-15 | 2.4e-06 | 4.5e-16 | 3.4e-16 / 1.5e-07 | 3.4e-16 / 1.5e-07 | 8.9e-16 | 2.4e-15 | exact | 1.2e-15 / 8.0e-07 | passive species, C(xi) 1 = C1: bitwise | byte-identical |
| bodyForce | 4.7e-15 | 4.3e-06 | 3.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.3e-15 | n/a | n/a | n/a | n/a | byte-identical |
| LES smag + bodyForce | 4.4e-15 | 4.3e-06 | 3.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | n/a | n/a | n/a | n/a | byte-identical |
| LES sigma + bodyForce | 1.3e-14 | 5.3e-06 | 4.2e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.3e-15 | n/a | n/a | n/a | n/a | byte-identical |
| RANS SA + bodyForce | 3.8e-15 | 3.6e-06 | 3.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 6.7e-16 | n/a | n/a | n/a | RANS scalars 0: bitwise | byte-identical |
| RANS SST + bodyForce | 3.5e-15 | 4.7e-06 | 4.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | n/a | n/a | n/a | RANS scalars 0: bitwise | byte-identical |
| C0 + bodyForce | 4.8e-15 | 2.7e-06 | 3.4e-16 | 6.7e-17 / 3.3e-08 | n/a | 4.4e-16 | n/a | exact | n/a | passive species: bitwise | byte-identical |
| TFLES uniform + C0 + bodyForce | 4.8e-15 | 2.7e-06 | 3.0e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | n/a | exact | n/a | TFLES F 1: bitwise | byte-identical |
| TFLES dynamic + C0 + bodyForce | 4.8e-15 | 2.7e-06 | 4.6e-16 | 6.7e-17 / 3.3e-08 | n/a | 6.7e-16 | n/a | exact | n/a | TFLES F 1: bitwise | byte-identical |
| LES smag + TFLES uniform + Charlette + C0 + bodyForce | 4.7e-15 | 3.2e-06 | 4.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | n/a | exact | n/a | n/a | byte-identical |
| C1 + bodyForce | 4.8e-15 | 2.7e-06 | 3.2e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | n/a | exact | 1.4e-15 / 5.8e-07 | passive species: bitwise | byte-identical |
| C2b + bodyForce | 4.8e-15 | 2.7e-06 | 3.8e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.3e-15 | n/a | exact | 1.6e-15 / 7.5e-07 | passive species, C(xi) 1 = C1: bitwise | byte-identical |
| gravity + bodyForce | 4.7e-15 | 4.3e-06 | 3.4e-16 | 3.4e-16 / 1.5e-07 | 3.4e-16 / 1.5e-07 | 4.4e-16 | n/a | n/a | n/a | n/a | byte-identical |
| C0 + turb | 4.8e-15 | 2.7e-06 | 4.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | 7.8e-16 | exact | n/a | passive species, turb u' 0: bitwise | byte-identical |
| LES smag + C0 + turb | 4.7e-15 | 3.2e-06 | 4.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 6.7e-16 | 7.8e-16 | exact | n/a | passive species, turb u' 0: bitwise | byte-identical |
| LES sigma + C0 + turb | 1.3e-14 | 2.7e-06 | 4.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.1e-15 | 8.9e-16 | exact | n/a | passive species, turb u' 0: bitwise | byte-identical |
| TFLES uniform + C0 + turb | 4.8e-15 | 2.7e-06 | 3.0e-16 | 6.7e-17 / 3.3e-08 | n/a | 2.0e-15 | 8.9e-16 | exact | n/a | TFLES F 1, turb u' 0: bitwise | byte-identical |
| TFLES dynamic + C0 + turb | 4.8e-15 | 2.7e-06 | 3.3e-16 | 6.7e-17 / 3.3e-08 | n/a | 4.4e-16 | 8.9e-16 | exact | n/a | TFLES F 1, turb u' 0: bitwise | byte-identical |
| LES smag + TFLES uniform + Charlette + C0 + turb | 4.7e-15 | 3.2e-06 | 3.1e-16 | 6.7e-17 / 3.3e-08 | n/a | 8.9e-16 | 1.3e-15 | exact | n/a | turb u' 0: bitwise | byte-identical |
| C2b + turb | 4.8e-15 | 2.7e-06 | 3.2e-16 | 6.7e-17 / 3.3e-08 | n/a | 2.2e-16 | 8.9e-16 | exact | 1.4e-15 / 7.4e-07 | passive species, C(xi) 1 = C1, turb u' 0: bitwise | byte-identical |
| steady LTS | 4.7e-15 | 4.3e-06 | 5.2e-16 | 6.7e-17 / 3.3e-08 | n/a | n/a | n/a | n/a | n/a | n/a | byte-identical |
| RANS SA + steady LTS | 3.8e-15 | 3.6e-06 | 5.4e-16 | 6.7e-17 / 3.3e-08 | n/a | n/a | n/a | n/a | n/a | n/a | byte-identical |
| RANS SST + steady LTS | 3.3e-15 | 4.7e-06 | 5.0e-16 | 6.7e-17 / 3.3e-08 | n/a | n/a | n/a | n/a | n/a | n/a | byte-identical |
| bodyForce + steady LTS | 4.7e-15 | 4.3e-06 | 8.6e-16 | 6.7e-17 / 3.3e-08 | n/a | n/a | n/a | n/a | n/a | n/a | byte-identical |
| shock | 3.0e-15 | 2.8e-06 | 4.2e-16 | 6.7e-17 / 3.3e-08 | n/a | 1.3e-15 | 1.7e-15 | n/a | n/a | n/a | byte-identical |
| C0 + shock | 4.5e-15 | 3.3e-06 | 3.5e-16 | 6.7e-17 / 3.3e-08 | n/a | 2.2e-16 | 6.7e-16 | exact | n/a | passive species: bitwise | byte-identical |

Regenerate: `test_gpu_fv_matrix markdown` (via simjob) prints everything from the first
`<!-- generated` marker to here.

## Cross-verification

OpenFOAM is absent here (the nozzle's rows at the end of the table are the exception: rhoCentralFoam, docs/nozzle.md): the structured solver is the oracle. **This table is
the template every later phase fills in.** Relation: *bitwise*, *rounding* (the
schemes are pinned equal, growth is a bug), *order p* (the discretizations differ
legitimately, the difference must shrink at p), *converged limit* (two different
models, compared at convergence, never bitwise).

| solver pair | mesh | quantity | expected relation | measured |
|---|---|---|---|---|
| `gpu_fv_comp` FV(FP64) vs FvRef | box_cyclic, box_refined, cyl2d_snappy | rhs, every pass, 3 steps | rounding | ≤ 2.2e-15 |
| FV(FP64) vs `Comp::rhs` | uniform periodic 8x6x5 | rhs | rounding | 3.0e-15 |
| FV(FP64) vs `Comp` (X1) | TGV 32³, M 0.3, Re 200, same dt | state, step 1 / 100 / 400 | rounding, no growth | 9.7e-14 / 1.6e-14 / 6.1e-14 |
| FV(FP32) vs `DeviceComp` (X1) | same | state, step 1 / 100 / 400 | rounding: ≤ 3 x structured FP32 vs FP64 | 5.2e-5 / 4.9e-6 / 8.6e-6 (own 3.8e-5 / 3.9e-6 / 7.0e-6) |
| FV(FP32) vs `DeviceComp` LES smag Cs 0.17 (X2) | TGV 32³, Re 2000 | state, step 400; mu_t | rounding: ≤ 3 x FV FP32 vs FP64 | 1.6e-5 (own 1.6e-5); mu_t 1.4e-5 of max |
| same, sigma Csigma 1.35 (X2) | same | state, step 400; mu_t | rounding | 3.2e-5 (own 2.5e-5); mu_t 1.7e-4 of max |
| FV(FP64) vs `Comp` curvilinear (X3) | sinusoidal periodic 2D, N 32 / 64 / 128 | rhs; state at t 0.1 | order 2 | rhs 3.8e-4 / 9.5e-5 / 2.4e-5 (2.00); state 1.5e-5 / 3.9e-6 / 9.8e-7 (1.99) |
| FV vs structured O-grid, both compensated (X4) | Re 40, M 0.2, 64x32 / 128x64 / 256x128 | cd at t 12.6, before the wake reaches the far row | order 2 | 2.98e-3 / 8.0e-4 / 2.5e-4 (1.90, 1.68) |
| same (`comp_ladder`) | Re 40, 96x48 / 192x96 / 384x192, far 20 D | cd at t 10 | order 2 | 9.8e-4 / 4.0e-4 / 1.2e-4 (1.29, 1.74) |
| same (X5, `gpu_fv_far`) | Re 40, M 0.2, 96x48 / 192x96 / 384x192, far 20 D | settled cd, window means flat to 3e-6 | order 2 | structured 1.565929, 1.566620, 1.56685; FV 1.564501, 1.566207, 1.56669: -1.43e-3 / -4.13e-4 / -1.6e-4 (1.79, 1.37). 384x192 stopped by the 290 s cap at t 59.46, both read at that t, FV still moving 2e-5 per 7 t |
| same | Re 40, M 0.1, 96x48 / 192x96, far 20 D | settled cd | order 2 | -1.43e-3 / -4.13e-4 (1.79), same as M 0.2 |
| same | Re 40, M 0.2, 192x96, far 10 / 20 / 40 / 80 D | settled cd difference | h error | -6.9e-4 / -4.1e-4 / -5.5e-4 / -1.18e-3 (was +1.35e-2 / +5.8e-3 / +2.3e-3 / +1.5e-4). The rise past 20 D is the wall cell: at fixed ny, h1 grows with R |
| same | Re 40, M 0.1, far 80 D, 128x64 / 192x96 | settled cd | order 2 | 1.51104, 1.51112; 1.50826, 1.50995: -2.8e-3 / -1.2e-3 (2.1) |
| same | Re 40, M 0.1, far 160 D, 128x64 / 192x96 | settled cd | order 2 | 1.51051, 1.50800; 1.49988, 1.50405: -1.06e-2 / -4.0e-3 (2.4) |
| same | Re 100, trip 0.05, 48x24 / 96x48 / 192x96, far 20 D | mean cd, t 200-316 | order 2 | 1.471937, 1.469549 / 1.386270, 1.385530 / 1.378701, 1.378607: -2.39e-3 / -7.4e-4 / -9.4e-5 (1.69, 2.98) |
| same | same | St | order 2 | 0.19551, 0.19612 / 0.17509, 0.17521 / 0.16769, 0.16772: 6.1e-4 / 1.2e-4 / 3e-5 (2.3, 2.0). St itself moves 4% between the last two rungs: not asymptotic |
| same | same | cl rms, t 200-316 | none claimed | 0.271402, 0.270738 / 0.203835, 0.203496 / 0.235742, 0.235656: 6.6e-4 / 3.4e-4 / 8.6e-5. cl rms itself non-monotone in h, window not shown to be a limit cycle |
| immersed `DeviceComp` ghost wall vs FV body-fitted | Re 40: 40 D box NSCBC vs 40 D diameter disc, not the same domain | cd | converged limit, two walls | immersed 12.8 cells/D 1.5714 (flat t 64-80, cl 0.03), 25.6 cells/D 1.5791 at t 51 and still falling. FV 384x192 1.5729. CONVERGENCE NOT MEASURED |
| plain FP32 vs FP64 (FV) | Re 40, 128x64 | cd, settled | "FP32 on a steady flow" | 1.5665-1.5764 vs 1.571630 |
| FV(FP64) `force` vs rhoCentralFoam Kurganov + upwind (T1, `gpu_fv_nozzle`) | N-C 80x16: A, B (shock), W (20 deg wedge) | every cell p, T, U at the fixed point | rounding | 1.2e-14, 3.4e-13, 1.5e-14 |
| FV(FP64) hybrid + D3 vs rhoCentralFoam vanLeer (T2) | N-C A 160x32, 320x64 | time means: vol L1 p; wall p; Cd; Cf | converging, order >= 1 | 1.61e-3, 1.07e-3 (0.59); 1.38e-3, 9.6e-4 (0.53); 1.2e-4, 5.0e-5; 3.5e-4, 2.4e-4. UNMET on order: see docs/nozzle.md |
| same (T2) | N-C B, p_b 0.6 | vol L1 p; wall p; x_s; Cf | same | 3.84e-3, 2.37e-3 (0.70); 3.32e-3, 1.99e-3 (0.74); 0.18 h, 0.41 h; 3.6e-5, 2.6e-4 |
| FV(FP64) vs exact source flow | 15 deg sector, 64, 128, 256 | L1 p | order 2 | 4.97e-4, 1.28e-4, 3.25e-5 (1.98) |
| FV(FP64) vs Sauer | parabola R 20, 280x32 | dM/dx, sonic x, Cd | converged limit | -2.34%, -1.20%, -3.6e-5 |

Orders above rest on 3 levels and nothing finer. Finer rungs: "not measured",
see "Reproducing the long numbers".


### What differs between the schemes, and what must not

Equal on a uniform periodic box: everything. LSQ gradients are central
differences, `AlongD` is the compact normal gradient (G1, X1).

Different on a curvilinear mesh: FV cell gradients are least squares, structured
ones come through metrics; a FaceMesh volume is OpenFOAM's pyramids, not
`CurvMetrics::vol` (warp order); `C` is the pyramid centroid. Truncation order, so
the difference is O(h^2) (X3).

Different at a wall: FV's wall face is a flux of the boundary state, `p_P`, the
face gradient corrected along `d` from the owner's LSQ gradient. The structured
wall is a ghost row with an antimirrored image, a face mean, tangential derivatives
exactly 0. Both second order on cd (X4).

Different at the far boundary: FV `Far` is the 1D characteristic state on the
face, nonlinear, sharp upwind switch, no viscous flux. The structured O-grid far
row is the linearized LODI state as a reflected ghost row, tanh blend, viscous
flux through the face. Equal to first order on an entropy wave since the fix
below; the differences left are measured there and do not reach cd.

### Far gap

**Closed: it was a bug in `Far`.** The settled cd used to plateau under h (+4.8e-3,
+5.8e-3, +6.0e-3 at 96 / 192 / 384, far 20 D). The outgoing invariant read `c_P`,
but `u_n + 2c/(gamma-1)` is invariant only at constant entropy. The wake is warm
(`dT/T ~ (gamma-1) M^2 du/U`), so the row read the entropy wave as an acoustic
wave: `2/(gamma-1) dc = M du` against the true `du`, a spurious pressure and
normal velocity across the wake, relative size M. The term now reads the
ISENTROPIC sound speed of the pressure, `cF (p_P/p_F)^((gamma-1)/(2 gamma))`, and
the boundary p comes from `cb/cF`, density from the upwind entropy. A pure entropy
wave leaves p and u_n untouched (`test_fvref`, exact).

Each row of the old `Far` swapped for the structured one, FV 96x48, Re 40, M 0.2,
far 20 D, structured 1.565929:

| FV far row | cd | FV - structured |
|---|---|---|
| old `Far` | 1.570760 | +4.83e-3 |
| + viscous flux through the far face | 1.57082 | +4.89e-3 |
| structured LODI state, Euler flux | 1.564491 | -1.44e-3 |
| + tanh band 1e-2 -> 1e-4 | 1.564492 | -1.44e-3 |
| + viscous flux | 1.56457 | -1.36e-3 |
| + reflected ghost, KG flux (the structured row entire) | 1.56457 | -1.36e-3 |
| `Far`, isentropic `c` (the fix) | 1.564501 | -1.43e-3 |

The whole structured row and the one-term fix agree to 7e-5; viscous flux, band and
ghost form are worth 8e-5 at most. The prediction "relative size M" held: 192x96
old gap +5.8e-3 at M 0.2, +3.0e-3 at M 0.1; fixed gap -4.13e-4 at both. Fixed gap
vs h and R: table above. The old 80 D +1.5e-4 was a sign crossing, not agreement.

What stays: FV and the structured row are still two forms of one characteristic
model (nonlinear vs linear, face vs ghost). Neither answers the source term of a
drag wake, so cd carries a D/R blockage on both, the same size now: 1.6545, 1.5662,
1.5348, 1.5230 at 10, 20, 40, 80 D (FV, 192x96, M 0.2). No R -> infinity limit is
extrapolated. Past 20 D at fixed `--ny` the wall cell grows with R (h1 7.5e-3,
3.0e-2, 6.1e-2 at 20, 80, 160 D, n 128), so an R ladder there is also an h ladder.


### FP32 on a steady flow

A plain FP32 state drops every update below `ulp(q)/2`: `E ~ 45` at M 0.2, `ulp`
3.8e-6, `dt` 4e-4, so a residual under 5e-3 is not applied. On a steady cylinder
that is not a floor but a slow limit cycle:

| O-grid 128 x 64, Re 40, M 0.2 | cd |
|---|---|
| FV FP32 plain | wanders 1.5665-1.5764 with period ~250 (t units), total mass pulsing 2.7e-4, to t 900 |
| FV FP64 | 1.571630, 6 digits flat from t 55 to t 222, cl 1e-12 |
| structured FP32 plain | 1.5650-1.5665 over t 150-944 |
| structured `--compensated` | 1.56624 at t 96 |

A number off a settled FV flow in plain FP32 is not a number. Use `--compensated`
(gate O11: at dt 2% of stable, 4000 steps, 8.2e-7 vs FP64 where plain sits at
1.8e-4) or `--fv-precision fp64`.

## Species

`dns flame --solver fv` (the flags' uniform box) and `dns flame --polymesh DIR --bc
FILE` (any polyMesh): the premixed one-step flame of model C0 (`docs/chem.md`) on
FaceMesh. One solver, one loop: `runFvComp` with `FvCompRun::flame`. `FvChem` in
`dns/fvflux.hpp`; `nY == 0` launches no species kernel and is the solver byte for byte.

Passes, FvRef's and the device's:

    primitives -> boundary states -> Y, omega -> boundary Y -> [u v w T gradients]
      -> mean fluxes -> grad Y -> species fluxes on F[0] -> gather every row -> the step

- **State** `[5 + nY][nC]`, up to 4 species. Flux rows `5 + nY` wide. The mean-flow
  kernels are the ones they were; they stride by `nV`.
- **Convection**: the species rides the face's own `F[0]`, read out of the flux row,
  so a uniform Y gives `F_Y = Y F_rho` and the mass flux is one number.
- **Face value** (`fvFaceY`): upwind on `mdot`, `Y_up + MC(d1, d2)/2`, `d2 = Y_down -
  Y_up`, `d1` the VIRTUAL upwind difference `2 grad Y_up . r - d2`, `r` up to down
  (Darwish-Moukalled). LSQ's gradient on a uniform mesh is central, so `d1 = Y_up -
  Y_upup`: the structured `faceY` stencil, to rounding (`fv_comp`, 1e-15). An
  unstructured mesh has no second neighbour; this is the same limiter with the only
  second difference it has.
- **Diffusion**: `rho D = mu / Pr` at the face's own mu, the corrected face gradient
  of Y. Le = 1.
- **The step**: ONE omega per stage (`arrheniusRate`, the structured expression), read
  by the fuel row and by `E`. The rate floors Y, the state is never clamped, `yHits`
  counts `Y` outside `[-1e-6, 1 + 1e-6]`.
- **Boundaries** (`fvBoundaryY`): an open face's flux is `mdot Y_b`, `Y_b` the row's
  `y[s]` where `mdot` comes in, the owner's where it leaves, no diffusion (the
  structured `ScalarBoundary` rule). Wall, Slip: `mdot` 0, zero gradient, no flux.
- **The gather**: species rows go through the same `kFvGather` source as the mean rows
  (`First`, `Count` template arguments), then `kFvChem` adds `-cr omega` and `+cr q
  omega`. Two kernels of different shape fused `cb X + cr R` into different FMAs:
  a compensated uniform Y walked off rho's bits, or a compensated passive species moved
  the mean flow (S3, S4 caught both). Same source, the rows scale bit for bit.
- **dt**: `min(stableDt, 0.5 / max J)`, `J` the structured `chemStableDt`, FP64 from
  the state words. Species diffuse under the thermal factor `max(4/3, gamma/Pr)`
  already.

**Open faces of the flame.** No NSCBC and no sink on a FaceMesh. x-lo is `Far` at the
fresh state (absorbs acoustics), x-hi `Outflow` at the burnt pressure `p_b = p0 - m^2
(sigma - 1)`. An `Inflow` + `Outflow` pair would be a closed organ pipe.

**`--bc` for a flame.** The file of "The `--bc` file", plus:

```toml
[reference]
direction = [0.94, 0.34, 0]  # the flame normal, burnt side +; the freestream
origin = [0, 0, 0]           # s = (C - origin) . direction; the flame sits at s = --upstream delta_th
[patch.in]
type = "far"                 # far, inflow, outflow
y = [1.0]                    # mass fractions an incoming face carries
[patch.out]
type = "outflow"
burnt = true                 # p = p_b, T_back = T_b; excludes p and tback
```

A `dns flow` file with any of `y`, `burnt`, `reference.origin` is exit 2 by name.

**Stats.** The structured runner's columns, same names, then `burn`, `fuel`:
`FvFlameProbe` (`dns/fvflame`) bins cells by centroid into slabs normal to `direction`,
width the mean `|d . dir|`, so on a Cartesian mesh a slab IS a structured column and
the x_f, delta_th (parabolic peak), S_L formulas are `runner_flame`'s. `energy` is
`Sum V (E + q rho Y)`. `--profile-dt` writes slab means. Frames: `temp`, `fuel`,
`omega`. Exit 8 when x_f comes within 2 delta_th of an end.
On a `[nozzle]` run (masterplan_combnozzle R9, `FvFlameSpec::chamber`): S_L divides by the
reference slab's own area `A_ref = V_ref/h` (not `vol/L`, which is the mean over the nozzle:
A_ref/(vol/L) = 2.02 on the reacting rig), and x_f, delta_th, Tmax, omegaEdge and the drift
guard read the CHAMBER bins only, contiguous from the inlet while `|V_slab/h - A_ref| <= 1e-9
A_ref` (the nozzle's own T drop crosses T_mid). Off a `[nozzle]` run: `vol/L` and every bin, bit
for bit (`fvref_react` R3).

Rejected by name on this route: every `--nscbc-*`, `--turb`,
`--hold-flame`, `--mode` other than premixed and diffusion (`--equivalence`: "Air-methane"); the mesh
flags with `--polymesh`. `--les`, `--sgs`, `--tfles`, `--tfles-sensor`, `--efficiency`:
"Species LES and TFLES" below (premixed only). `--physics lowmach` on this route is `docs/fvincomp.md`
"Low-Mach". `--gamma G` on `--polymesh` (premixed C0 only): one gamma for fresh and burnt gas
(docs/chem.md "FV path"). Off it: `--bc`, `--fv-precision`,
`--compensated`, `--frame-*`.

### Species cross-verification

Same table rules as "Cross-verification". S_L is the mean over [7, 14], delta_th the
oracle's 1.60165 at sigma 5, beta 8, Pr 0.72, M 0.01 unless stated, u_in 0.98.

| solver pair | mesh | quantity | expected relation | measured |
|---|---|---|---|---|
| FV(FP64) vs FvRef (S1) | box_cyclic, box_refined, cyl2d_snappy | rhs, Y, omega, grad Y, boundary Y, fluxes, 3 steps, chem dt | rounding, 1e-13 | rhs 5.5e-16 / 3.3e-15 / 4.0e-15; species flux 1.6e-16 / 1.6e-16 / 3.8e-16; 3 steps 3.8e-16 / 5.6e-16 / 5.7e-16; grad Y 0 / 5.9e-16 / 2.1e-16; chem dt 0 |
| FV(FP32) vs FV(FP64) (S2) | same | rhs; 3 steps | O2's bounds 2e-5, 2e-6 | 1.1e-6 / 4.9e-6 / 3.7e-6; 2.3e-7 / 2.8e-7 / 3.7e-7 |
| uniform Y 0.25, 0.5 (S3) | same, every boundary kind, 40 steps | rho Y vs Y rho | bitwise | bitwise: FP32, FP32 compensated, FP64 |
| passive species vs nY 0 (S4) | same, 40 steps | mean rows | bitwise | bitwise, all three |
| burning box (S5) | uniform periodic, box_cyclic walls, box_refined; 16-56% of the fuel burnt | mass, E + q rho Y, passive rho Y; Sum V R_Y + Sum V omega | rounding, 1e-13 | mass 2.9e-15; E + q rho Y 2.2e-15 / 3.6e-15 / 5.1e-15; passive 1.6e-15 / 2.9e-15 / 2.0e-15; budget 9.8e-16 / 2.2e-16 / 1.1e-15 |
| FV(FP32) vs structured `kSpeciesStage` (S8) | uniform periodic 128x32, burning, M 0.05, same dt, 400 steps | full state | rounding: <= 3 x FV FP32 vs FP64 | 3.3e-5 (own 4.1e-5); structured vs FV FP64 4.6e-5 |
| FV(FP64) vs oracle | 1D box, 7.5 / 15 / 30 cells per delta_th | S_L - 1 | order 2 with an O(M^2) offset | +1.69e-2, +3.47e-3, +4.37e-4 (2.28, 2.99); about the offset `1 - gamma M^2 (sigma - 1)` (-5.6e-4): 1.75e-2, 4.03e-3, 9.97e-4 (2.12, 2.01). Spans in the window 2.9e-4, 8.0e-5, 2.0e-5 |
| same | same | delta_th - oracle | order 2 | -4.84e-2, -1.05e-2, -1.79e-3 (2.20, 2.55) |
| FV(FP64) vs structured (FP32) | same mesh, M 0.01 | S_L | not rounding: FP32 floor and the fresh state, below | 1.016873 vs 1.016933, 1.003472 vs 1.003104, 1.000437 vs 1.000054: -6.0e-5, +3.7e-4, +3.8e-4 |
| same | same | delta_th | same | 1.55325 vs 1.55488, 1.59115 vs 1.59111, 1.59986 vs 1.59982: -1.6e-3, +3.6e-5, +3.9e-5 |
| FV `--compensated` vs FV(FP64) | 15 cells | S_L, delta_th, 20 flame times | rounding | 1.003447, 1.003447; 1.59072, 1.59073 |
| polyMesh reader vs fromCurvMesh | same box through `tools/chem/fv_flame_mesh.py`, FP64 | S_L, delta_th, x_f | bitwise to the printed digits | 1.003447, 1.59073, 20.4096, equal |
| FV(FP64) polyMesh vs the box row above (G1.1) | poly1d (`fv_flame_mesh.py` ny 4), 7.5 / 15 / 30, [7, 14] | S_L - (1 - gamma M^2 (sigma - 1)) | order 2; the box within its spans | +1.743e-2, +4.032e-3, +9.966e-4 (2.112, 2.017); vs the box +1.0e-8, +4.3e-7, -4.1e-7 |
| FV(FP64) vs reactingFoam (G1.2, masterplan_combnozzle P1) | the SAME poly1d files, foam mclin (docs/chem.md "P1") | S_L | converged limit: gap below both oracle errors, falling | +1.384e-3, +4.592e-4, +2.341e-4 (own errors 1.74e-2, 4.03e-3, 9.97e-4); fresh-state term 1.7e-5 |
| FV(FP64) at M 0.042 (G1.5) | poly1d, 7.5 / 15 / 30 | S_L self-convergence | order 2 | 1.007853, 0.994837, 0.991887: order 2.141, Richardson 0.991021 (+9.0e-4 off the O(M^2) formula, reported) |
| FV(FP64) gamma 1.2 vs 1.4 (G1.6, G1.7) | poly1d, 15 / 30 | T_b/(sigma Tu) - 1 difference over [x_f + 3, x_f + 5] delta_th | +4.8e-5 [D] +- 1.2e-5 | +4.809e-5, +4.751e-5; S_L order 2.074, 1.865 about 1 - 1.2 M^2 (sigma - 1) |
| `--shock hll` vs `--shock-d4 0` (G3.3, masterplan_combnozzle P3) | poly1d, 7.5 / 15 / 30, M 0.01 and 0.042 | Delta = S(hll) - S(d4 0) | <= \|S(d4 0) - S_ref\|/3, order >= 1.8; SZ defect <= 4/3 x d4 0; sensed 0 | M 0.01 -2.982e-3, -3.491e-4, -4.204e-5 (order 3.09, 3.05); M 0.042 -2.809e-3, -3.314e-4, -4.084e-5 (3.08, 3.02); bands 5.8e-3 .. 3.3e-4; SZ ratio 0.82 .. 1.00; shockCells 0 on every row |
| FV(FP64) `--shock hll` vs reactingFoam at M 0.042 (G1.8, masterplan_combclose K1) | the SAME poly1d files, dns 7.5 / 15 / 30, foam mclin 15 / 30 (docs/chem.md "P1") | S_L | converged limit: gap below both errors vs the dns Richardson S∞, falling | **UNMET**: +2.620e-3, +2.593e-3 (not falling) vs 3.56e-3, 1.70e-3; S∞ 0.9909473, foam Richardson 0.988374: a converged O(M^2) gap, 0.26 gamma M^2 (sigma - 1) (M 0.01: 0.29; issues.md N21) |
| `--shock hll --shock-d4 0` vs no `--shock` (G3.4) | poly1d 15 (both Machs), the box and poly1d at 7.5 | stats columns, profiles, state | bytes | identical (781 / 229 rows; gpu_fv_matrix_phys state bytes; gpu_fv_flame_shock) |
| rotated 30 deg vs planar | 7.5 / 15 / 30, ny 16 / 32 / 64, FP64 | S_L; delta_th | rounding of the geometry | -5.1e-8, 0, -1.3e-8; +1.2e-5, -1.8e-6, +1.7e-7 |
| sheared 30 deg (every face non-orthogonal) vs planar | same | S_L | order 2 | -3.50e-3, -9.40e-4, -2.38e-4 (1.89, 1.98) |
| same | same | delta_th | order 2 | -2.12e-2, -5.38e-3, -1.25e-3 (1.98, 2.11) |
| wavy, flame crossing i lines (A 2 h) vs planar | same | S_L | order >= 2 | +1.18e-3, +3.21e-5, +6.0e-7 (5.2, 5.7) |
| same | same | delta_th (slab read across curved i lines) | order 2 | +2.10e-2, +4.25e-3, +9.48e-4 (2.30, 2.16) |
| shear + wavy + rotate 20 deg vs planar | same | S_L | order 2 | -1.99e-3, -9.05e-4, -2.37e-4 (1.14, 1.94): the first rung is two errors of opposite sign |
| same | same | delta_th | order 2 | +4.4e-4, -1.31e-3, -3.04e-4: sign change, last 2.11 |

All rows: 0 cells outside `[-1e-6, 1 + 1e-6]`. Orders rest on 3 levels.

**FV vs structured is not rounding end to end, and why.** The interior is (S8). Three
things are not:

1. **Plain FP32 at M 0.01 cannot resolve S_L below ~1e-3.** FV FP32 reads 1.017040,
   1.003167, 1.001623 at its dt, 1.016802, 1.003884, 1.000168 at `--cfl 0.8` (the
   structured dt), FP64 1.016873, 1.003472, 1.000437: non-monotone in dt and h, spread
   7e-4 at one mesh, and the window span grows to 1.6e-3 at 30 cells. `E ~ 1.8e4`,
   `ulp 2e-3`, and the state lands on quantized fresh temperatures (TFresh 7143.36 vs
   7142.91). The structured flame has no FP64 and no compensated species: its numbers carry
   this (structured S_L spans 4e-4 to 1e-3).
   `--compensated` or FP64 on the FV path has no such floor.
2. **The boundary sets the fresh state.** At M 0.05, where FP32 is clean on FV (FV32
   vs FV64 ≤ 1.1e-4), structured NSCBC holds rho, FV `Far` holds entropy: TFresh
   286.146 vs 285.786 at 7.5 cells. `--nscbc-tinflow` moves the structured flame to
   285.634. S_L follows T_u at 3.5e-3 per unit T (`dln S/dln T ~ 1.0`), measured on the
   structured pair, stable across h and u_in.
3. **Structured S_L at the finest rung is noisy.** Read off at FV's fresh T (linear
   between the two structured inflows):

   | M 0.05 | FV(FP64) - structured at FV's T_u | structured window span |
   |---|---|---|
   | u_in 0.98, 7.5 / 15 / 30 cells | -4e-5, -9e-6, +8e-4 | 6.6e-4, 1.9e-4, 9.3e-4 |
   | u_in 0.90, 7.5 / 15 / 30 cells | +3.4e-4, +7.8e-5, -5.3e-4 | 7.5e-4, 7.2e-4, 1.0e-3 |

   The 30-cell gap changes sign with u_in and sits inside the span. It moves 3e-4 with
   `--nscbc-sink 0|16|32` (0.987973, 0.987782, 0.987665) and a 21-flame-time window moves
   it -5e-4. FV spans 8.7e-5 there. p differs too (structured 286.03 vs FV 285.84 at 30
   cells): FV forced to the structured p moves S_L +1.1e-4. Stated gap: at a matched
   fresh state FV and structured agree to the structured run's own window span, ≤ 1e-3.
   No order is claimed on this difference. Better is not measurable with a plain FP32
   NSCBC structured flame.

The FV ladder at M 0.05 (`--fv-precision fp64`): S_L 1.004004, 0.991129, 0.988211,
delta_th 1.58090, 1.61690, 1.62581. S_L heads for `1 - gamma M^2 (sigma - 1)` = 0.986,
the O(M^2) offset of `docs/chem.md`.

Not measured (dropped by the 3-level rule): a 60-cell rung of any row:

    tools/chem/fv_flame_ladder.py --cells 60 --tt 14 --meshes rot,shear,wavy,all

## Species LES and TFLES

`dns flame --solver fv` and `--polymesh DIR --bc FILE` take the structured flame's
`--les CS --sgs smag|sigma`, `--tfles F`, `--tfles-sensor`, `--efficiency` (docs/chem.md
"LES species and the thickened flame"), premixed C0 only; same validator rules (`--sgs`
explicit; `--les` needs `--nz >= 8` on the box, a 3D mesh on `--polymesh`: exit 3 at the
run; Charlette needs `--les`; F x cells per delta_th >= 4, measured off the probe's slab
on a mesh). Equations, constants, label and IC are the structured ones; the FV terms
restate the `Eddy` / `Thick` instantiations of `kSpeciesStage` and `gpu/tfles` on faces:

- **Eddy** (`fvEddyRhoD`): every internal face rho D + nut_f / Sc_t, nut_f the mean flux's
  own face mean of mu_t (the FV LES slot, `kFvNut`), Sc_t = Pr_t = `prt` (ONE constant,
  0.9). Boundary faces: no SGS flux (the structured open rows: none either). The
  structured flame copies mu_t onto its open x rows (its SGS gradient wraps there); the
  FV gradient reads the boundary face, so no copy.
- **Thick** (`fvThickRhoD`, `fvThickEnergyFlux`): rho D = FE_f mu_f/Pr + (1 - Om_f)
  nut_f/Sc_t; F_E += -(FE_f - 1) kappa_f g_T.S + Om_f nut_f (cp/Pr_t) g_T.S, an ADDITIVE
  face flux in the species face kernel (`kFvSpFace<.., Thick>`), g_T the corrected face
  gradient: the structured compact dT/dn on a uniform mesh. omega -> (E/F) omega in the
  cell pass, read by the fuel row and q omega (one buffer). Boundary faces unthickened.
- **Frozen per step**: `step()` forms F E, E/F, Omega from its INPUT state; `rhs`,
  `evaluate`, `stableDt`, `speciesFields`, `tflesStats` from the current one.
- **Sensor**: `tanh(beta w_s)` (rate at Ta/2), then the T-gated max filter, one pass per
  walked global axis (x, y, z; the empty axis skipped). The filter walks FACES
  (`fvTflesWalk`, `dns/fvles`): from a cell, `dilate` steps along +-e_a, each to the
  neighbour across the face whose outward normal is most aligned (ties: first in CSR
  order); a boundary face stops the walk, a cyclic face is internal. On a uniform box
  that is the structured line: x clamped at Far/Outflow, y and z wrapped (L5 below). On
  a mesh rotated or sheared by less than 45 deg it walks the mesh lines.
- **Efficiency**: Colin's u' off the LSQ curl and the compact Laplacian (1/V) sum (phi_N
  - phi_P) corr.S, h = sqrt Delta^2 (`fvLesDelta2`: V^(1/3); 2D the cell's side), E = 1
  on the cells within `kTflesEdgeRows` face layers of an active boundary face (the
  structured edge rows on the flame box; walls too).
- **dt**: the Sigma form with mu max(4/3, FE gamma/Pr) + mu_t max(4/3, gamma/Pr_t), FE the
  cell's. The structured `tflesStableDt` is a separate 0.25 h^2 bound.
- **Stats**: `Ebar` (sum V Omega E / sum V Omega; structured: unweighted), `FEmax`; S_L
  is then the consumption speed of (E/F) omega.

Off: no `--les`, no `--tfles` launches no new kernel (the Molecular instantiations,
byte for byte); `--tfles 1` (uniform or dynamic, no LES) is the DNS byte for byte (L2).
Two reactants + LES (MC, "Combinations"): the `Eddy` instantiation of the `Two` face
kernel (`kFvSpFace<R, true, true, Eddy>`), one diffusivity `rho D + nut_f/Sc_t` and one
limiter for every species, so Z(Y) == a carried tracer holds (`gpu_fv_matrix`,
`gpu_fv_matrix_phys` P5). No CLI spelling for the jet (2D, `--nz 1`; LES takes `--nz >= 8`); the CLI spelling is
`--equivalence` ("Air-methane").
Not built: TFLES of two reactants (the inventory refuses it), `--turb`, a thickened
boundary face, the structured open-row mu_t copy (not needed), LES on a 2D mesh.

### Species LES and TFLES verification

`gpu_fv_flame_les` (`tests/test_gpu_fv_flame_les.cpp`) and `gpu_fv_tfles1d`. FvRef carries
the same terms (`FvConfig::les`, `FvConfig::tfles`, `lesCells`, `tfles*`).

| pair | mesh | quantity | expected | measured |
|---|---|---|---|---|
| FV(FP64) vs FvRef (L1): LES; TFLES uniform; dynamic; LES + dynamic + Charlette | box_cyclic, box_refined (Sutherland), cyl2d_snappy (every kind), 3D periodic 12x10x8 (sigma) | rhs; mean fluxes (thickened E included); species fluxes; omega; mu_t; F E, E/F, Omega; stableDt; 3 steps | rounding, 1e-13 | <= 5.3e-15; <= 1.1e-15; <= 7.6e-16; <= 4.0e-16; <= 6.0e-14 (sigma's conditioning, gpu_fv_comp O9); <= 3.4e-16; 0; <= 9.4e-16 |
| FV(FP32) vs FV(FP64) (L1) | same | rhs; 3 steps | O2's 2e-5, 2e-6 | <= 4.5e-6; <= 4.2e-7 |
| `--tfles 1` vs DNS (L2) | same 4 meshes, 40 steps, FP32, compensated, FP64 | state | bitwise | bitwise, uniform and dynamic |
| passive species + LES vs LES nY 0 (L2) | same | mean rows | bitwise | bitwise, all three |
| uniform Y 0.25, 0.5 (L2) | same, LES and LES + TFLES | rho Y vs Y rho | bitwise | bitwise, all three |
| burning box, LES + TFLES + Charlette (L3) | box_cyclic, box_refined, 3D periodic; 9-39% burnt | mass; E + q rho Y; passive | rounding, 1e-13 | 2.4e-15 / 2.9e-15 / 2.0e-15; 8.9e-16 / 1.3e-15 / 8.9e-16; 2.7e-15 / 2.2e-15 / 2.0e-15 |
| rerun (L4) | 4 meshes, 40 steps | state | bitwise | bitwise, both precisions |
| FV face walk vs `DeviceTfles` (L5) | uniform 64x16x16, open x (Far, Outflow), y z cyclic, one burning state, FP32, F 2, radius 12 | Omega; F E (dynamic; + Charlette, E up to 1.222) | rounding (the raw sensor, LSQ vs central curl) | 8.3e-7; 5.3e-7, 9.4e-7 of F E |
| FV(FP32) vs structured `kSpeciesStage<Eddy/Thick>` (X1) | uniform periodic 96x16x16, burning slab + vortical patch u' 2 S_L, M 0.05, same dt, 300 steps | full state | rounding: <= 3 x FV FP32 vs FP64 | Smagorinsky 0.17: 2.2e-5 (own 3.4e-5), mu_t/mu 0.013; sigma 1.35: 2.2e-5 (3.1e-5); Smagorinsky 1.5 (test-only, mu_t/mu 0.78): 2.3e-5 (3.5e-5); TFLES F 2 uniform: 2.9e-5 (4.5e-5); dynamic: 6.1e-5 (2.9e-5); sigma + dynamic + Charlette: 4.0e-5 (2.6e-5); Smagorinsky 1.5 + dynamic + Charlette: 3.6e-5 (2.7e-5) |
| structured vs FV(FP64) (X1) | same | full state | same bound | 3.3e-5, 3.2e-5, 3.0e-5, 5.6e-5, 5.4e-5, 3.7e-5, 3.4e-5 |
| TFLES F 2 dynamic, FV(FP64) vs oracle | box, 3.75 / 7.5 / 15 cells per delta_th (7.5 / 15 / 30 thickened), M 0.01, --upstream 16 --downstream 12, tt 28, window [14, 28] | S_L - 1 | order 2 | 1.016790, 1.003352, 1.000304: +1.68e-2, +3.35e-3, +3.04e-4 (2.33, 3.46). FV DNS at twice the cells ("Species" table, another box and window [7, 14]): 1.016873, 1.003472, 1.000437 |
| same | same | delta_th / 2 | -> oracle | 1.55285, 1.59129, 1.59999 |
| FV(FP32) vs structured (FP32), same flags | same | S_L | not rounding: the M 0.01 FP32 floor and the boundaries ("Species" items 1-3) | -1.44e-4, +7.9e-6, +2.07e-3 (FV FP32 at 15: 1.001771 vs FP64 1.000304, item 1's scatter) |
| rotated 30 deg vs planar | same, ny 8 / 16 / 32, FP64 | S_L; delta_th | rounding of the geometry | -1.08e-5, < 1e-9, < 1e-9; +4.2e-6, +3.4e-6, -6.3e-7. The 3.75 rung with the UNIFORM sensor: -9.2e-6, so not the walk; h = delta_th/3.75 is coarser than any DNS rung (DNS rotated at 7.5 cells: -5.1e-8). Cause not isolated |
| sheared 30 deg (every face non-orthogonal) vs planar | same | S_L; delta_th | order 2 | -3.51e-3, -9.40e-4, -2.36e-4 (1.90, 1.99); -4.19e-2, -1.06e-2, -2.50e-3 (1.98, 2.08): the DNS row's numbers at twice the cells (-3.50e-3, -9.40e-4, -2.38e-4) |
| 3D laminar LES + TFLES F 2 + Charlette vs 2D TFLES F 2 | box 3.75 cells, nz 8 ny 8 vs ny 8, FP64, window [14, 28] | S_L; Ebar | D_sigma 0 and curl 0 on a planar flame (E = 1): equal up to the different step sequence | sigma 1.35: +5.1e-7; Smagorinsky 0.17: +7.8e-7 (its stress stays; Omega 1 switches its scalar flux off); Ebar 1.000000, FEmax 2 both |
| binary route (`gpu_fv_tfles1d`) | box, F 2 dynamic at 3.75 cells/delta_th (7.5 thickened), FP64, t 20, window [10, 20] | S_L; delta_th; FEmax; hits | S_L 2%; 2 delta_th 5%; F; 0 | 1.016984; 3.10744 (2 x 1.55372); 2.000000; 0 |

The dynamic-sensor rows of X1 sit 1.3-2.1 x their own FP32 distance (uniform sensor and LES
alone: 0.6-1.2 x). Within the bound; the excess is not isolated (a candidate, not measured:
the gate `T_n >= T_c` is a switch a near tie can flip between two FP32 evaluations).

Ladder rows (3 levels; a 30-cell rung, 60 thickened, not measured):

    tools/chem/fv_flame_ladder.py --cells 3.75,7.5,15 --tt 28 --tfles 2 --box-up 16 --box-down 12 --meshes rot,shear

## Diffusion flame

`dns flame --mode diffusion --solver fv` (the structured box as a FaceMesh) and `dns flame
--mode diffusion --polymesh DIR --bc FILE` (any polyMesh): model C1 of `docs/chem.md`, F +
s O -> (1 + s) P, on the species passes above. One solver, one loop: `runFvComp` with
`FvCompRun::jet`. Chamber, bunsen, wick, `--pulse`: exit 2 by name; the candle: "Hydrostatic ambient".

- **`FvChem::ox >= 0`**: omega = A rho^2 Y_F Y_O e^(-Ta/T) (`arrheniusRate2`, floors both Y),
  oxidizer `-s omega`, E `+q omega`, ONE omega. `kFvSpCell`, `kFvSpFace`, `kFvSpBndFace`,
  `kFvChem` take a `Two` argument: `ox < 0` is the premixed instantiation, bit for bit.
- **ONE limiter per face** (`fvThetaVote`, `fvFaceYShared`): theta = min over the species
  with |d2| >= 2^-24 of MC(d1, d2)/d2, in [0, 2], d1 the virtual upwind difference of
  `fvFaceY`. The face value `Y_up + theta/2 (Y_down - Y_up)` is linear in Y, and so are
  the LSQ gradient, Fick and the boundary value, so every affine combination rides the
  same formula. The structured kernel's "boundary cell: theta 1" has no FV twin: `d1`
  exists next to a boundary too.
- **Per-face rows** (the FvBc table's design): `FvConfig::bcRow`, one row of `bc` per
  boundary face, extra rows after the patch rows, same kind as the patch (`make` refuses
  anything else). `BndRow::bc` always held a row, the kernels did not change.
  `fvJetRows`: a row per face of a `jet = true` Inflow patch, `diffusionInflow` at the
  face centre: u, T, Y_F, Y_O of the Burke-Schumann sheet.
- **The inflow is a fixed FACE** (`FvBc::diffusive`): u, T and Y on the face whatever
  the sign of mdot, Fick, Fourier and stress through it (`fvBoundaryFlux`,
  `kFvSpBndFace`). That is foam's G3a inflow (`docs/chem.md`), not the structured pinned
  first cell: the first cell is interior, it burns (`xBase` 0). What each alternative
  did, FP64 (4 cells/l, the last row 4.5, t 120-160):

  | inflow | fuelIn | tip | |
  |---|---|---|---|
  | `Far` rows (the premixed recipe) | 7.95 | 42.97 | 6% under the nominal rho u Y: u_b sags under the flame's pressure, a soft inflow does not hold a jet |
  | `Inflow`, no face diffusion | 8.51 | 46.1 | nominal rho u Y 8.4667: the 5% foam takes by Fick never enters |
  | `Inflow`, `diffusive` | 8.951 | 47.24 | foam mclin 8.959, 47.217 (4 cells/l); structured etaIn 1000 8.986, 47.59 |

- **Outflow**: `Outflow` at p0, T_back T0, `ownY` (a backflow face carries the owner's
  Y, docs/chem.md item 4), `characteristic`: while the flow leaves, p_out with the
  owner's entropy and outgoing invariant, so a cell below p0 slows its own efflux. The plain
  row drains it at p0's density instead: at 30 deg shear, cells 3 from the outflow fell to
  rho 6e-4, p 0.5 at t 23 and dt collapsed; rotated and planar never reached it. Inflow +
  Outflow is an organ pipe: the start rings p/p0 +-5%, it decays with an e-fold of ~30 l/U,
  so read a jet past t 100, not at the structured window. No growth at these Da.
- **`--bc`**: `[patch.in] type = "inflow"`, `jet = true` (excludes u, t, y);
  `reference.direction` the jet, `reference.lift` lateral, `reference.origin` the slot's
  centre on the inflow plane. `jet = true` with no `--mode diffusion`, or the mode with no
  jet patch: exit 2. `tools/chem/fv_flame_mesh.py --jet`.
- **Probe** (`FvJetProbe`, `dns/fvflame`): `measureJet`'s numbers on slabs along `dir` (a
  Cartesian mesh's rows, `test_fv_comp` pins tip, half, base to `measureJet`), the axis the
  cells within the mean |d.lift| of y = yc, y wrapped to the lateral period (a sheared
  cyclic mesh leaves the slot's y). `fuelIn` the open patches' species flux, Fick included;
  `fuelOut` the Outflow's. Columns: the structured jet's.

### Diffusion flame verification

| pair | mesh | quantity | expected | measured |
|---|---|---|---|---|
| FV(FP64) vs FvRef (D1) | box_cyclic, box_refined, cyl2d_snappy, a row per open face (inflow `diffusive`, outflow `ownY`, `characteristic`) | rhs; omega; grad Y; species flux; 3 steps; chem dt | rounding, 1e-13 | 5.5e-16 / 3.3e-15 / 4.0e-15; 1.4e-16 / 2.6e-16 / 2.6e-16; 0 / 5.4e-16 / 2.0e-16; 1.7e-16 / 1.8e-16 / 2.7e-16; 5.0e-16 / 8.0e-16 / 7.5e-16; 0 |
| FV(FP32) vs FV(FP64) (D1) | same | rhs; 3 steps | O2's 2e-5, 2e-6 | 1.1e-6 / 4.9e-6 / 3.7e-6; 2.1e-7 / 2.9e-7 / 3.7e-7 |
| uniform Y, passive (D2 = S3, S4, ox 1) | same, 40 steps, FP32, compensated, FP64 | rho Y vs Y rho; mean rows vs nY 0 | bitwise | bitwise, all three |
| burning box (D3) | uniform periodic, box_cyclic walls, box_refined; 4.8-20% burnt | mass; E + q rho Y_F; rho (Y_F - Y_O/s) of gross; tracer; Sum V R_F + Sum V omega; Sum V R_O + s Sum V omega | rounding, 1e-13 | 4.0e-15 / 2.2e-15 / 2.7e-15; 3.6e-15 / 4.0e-15 / 3.8e-15; 1.1e-15 / 3.7e-16 / 7.4e-16; 3.8e-15 / 2.4e-15 / 1.3e-15; 9.7e-15 / 4.3e-16 / 2.0e-16; 3.3e-15 / 4.3e-16 / 4.0e-16 |
| Z(Y) vs tracer (`gpu_fv_flame_mix`, `gpu_diffusion_mix`'s layer) | uniform 256^2, FP32 | max difference: reacting; passive; per-species limiter | FP32 walk 6.4e-5, 5.5e-5; contrast > 30 x | 8.9e-6; 2.8e-5; 1.226e-3 (43 x; structured 8.3e-6, 1.4e-5, 1.226e-3) |
| same | sinusoidal 0.02 | same | same | 9.3e-6; 1.7e-5; 1.92e-3 (115 x) |
| Z vs the analytic erf | uniform; sinusoidal | max | 3e-3 | 1.46e-3 (structured 1.460e-3); 1.89e-3. Z in [8.88e-2, 1 - 8.83e-2], no bound broken |
| FV(FP64) vs structured etaIn 1000, jet, same mesh | 2, 3, 4.5 cells/l, t 160, last quarter | tip | first order: the structured row is a pinned CELL at h/2 | 47.958 vs 47.233, 47.729 vs 47.239, 47.593 vs 47.242: +0.725, +0.490, +0.352 (order 0.97, 0.81); linear limit +0.05 |
| same | same | half-width | shrinks | 7.7145 vs 7.6544, 7.6880 vs 7.6782, 7.6701 vs 7.6634: +0.060, +0.010, +0.007 (FV's own values are not monotone: the tip/2 slab) |
| same | same | TmaxIn / T_ad; fuelIn | none claimed (a stated boundary gap) | 0.9853 vs 0.9846, 0.9854 vs 0.9844, 0.9854 vs 0.9843; 8.994 vs 8.978, 8.989 vs 8.960, 8.986 vs 8.951 |
| FV(FP64) vs structured default | same | tip; fuelIn | NOT the same inflow: the etaIn 100 row overshoots (`docs/chem.md` G3a) | 51.449, 51.236, 51.163; +4.22, +4.00, +3.92 l; 9.635, 9.621, 9.618 |
| FV vs foam mclin (from `docs/chem.md`, not rerun; foam 4 cells/l, FV 4.5) | one mesh each | tip; half; fuelIn; TmaxIn/T_ad; burn/in | same inflow model | 47.24 vs 47.217; 7.663 vs 7.669; 8.951 vs 8.959; 0.9843 vs 0.9849; 0.99992 vs 0.99839 |
| FV alone | 2, 3, 4.5 cells/l | tip; TmaxIn/T_ad; burn/in; hits | converging | 47.233, 47.239, 47.242; 0.98464, 0.98441, 0.98433; 1.00004, 0.99999, 0.99992; 0. span of tip in the window 0.15, 0.12, 0.10: the ring |
| rotated 30 deg vs planar | 2, 3, 4.5 cells/l | tip; half; TmaxIn | rounding of the geometry | 0, -4.8e-6, -6.5e-6; 0, 4.1e-7, 4.6e-7; 0, -4.9e-8 rel, 0 |
| sheared 30 deg vs planar | 2, 3 cells/l | tip; half; TmaxIn rel | order 2 | -0.700, -0.342 (1.77); +8.1e-3, -1.9e-2; +3.5e-3, +1.7e-3 (1.8). **4.5 DIVERGED**, below |
| wavy 2 h vs planar | 3 cells/l, one rung | tip; half; TmaxIn rel | small | +4.1e-6; +4.4e-4; +6.4e-6 |
| polyMesh reader vs fromCurvMesh | 3 cells/l, FP64 | tip, half, fuelIn | bitwise to the digits | 47.2391, 7.6782, 8.9596, equal |
| FV `--compensated`, plain FP32 vs FP64 | 3 cells/l | tip; TmaxIn | rounding | 47.2391, 47.2373 vs 47.2391; 1406.305, 1406.240 vs 1406.305 |
| frames | 2 cells/l, `--emit bin` | temp, fuel, oxid, omega | 4 channels | written |

Stated gaps. The structured row pins a CELL, this one a FACE: burn base 0 vs 0.25, a first-order
tip offset. No sink: the ring above. The interior is not compared to rounding: no
two-reactant S8 against the jet (the premixed S8 covers the passes they share; the
interior of the two-reactant stage is rounding-equal to `kSpeciesStage2`: "Air-methane" R1).

**Sheared jets are NOT stable to the end.** 30 deg, FP64: 3 cells/l diverges at t 308.7, 4.5
cells/l at t 155.9 (dt falls 3x per 200 steps), 2 cells/l survives t 160, not run past it.
Blow-up time ~ h^1.7, so a rate of a diffusive kind; `dtChem` falls for ~70 l/U first.
At 4.5, t 155.6: a sawtooth, cell to cell p 3.7 vs 569, rho 4.5e-3 vs 0.68, at x 6 on the lean
side of the sheet, Z ~ 0.25, away from any boundary. Not the outflow evacuation (fixed: that
one reached t 23). The Fourier symbols of the corrected Laplacian and of the stress operator
have no growing mode at 0-45 deg with constant coefficients, so it needs the variable state.
Cause NOT isolated (issues.md 78c). Rotated, wavy: no sign of it. Numbers on a sheared jet are
read before the blow-up, and the 4.5 rung is not in the table.

Not measured (dropped by the 3-level rule and the 295 s cap): a finer rung:

    tools/chem/fv_jet_ladder.py --cells 6 --tt 160 --meshes rot,shear

    dns flame --mode diffusion --polymesh SHEAR_4.5 --bc SHEAR_4.5/bc.toml --tt 160 --fv-precision fp64   # diverges at t 155.9

## Air-methane

`dns flame --equivalence PHI [--sl-law gri30|onestep] [--stoich S --ox-y Y] [--mix-phi PHI]
--solver fv` (the flags' box) and `--polymesh DIR --bc FILE` (any polyMesh): models C2 and C2b
of `docs/chem.md` ("Model C2", "Model C2b"), the rich CH4/air premix on the two-reactant stage
of "Diffusion flame" (masterplan_multiphysics M2). One solver, one loop: `runFvComp` with
`FvCompRun::flame`, the C0 flame's probe, stats, frames.

- **C2 is C1's stage** (`FvChem::ox >= 0`, `law.n == 0`): F + s O, ONE omega, one shared
  limiter; only the case differs (premix in, oracle `premixedEigen2`, q = cp Tu (sigma - 1)
  s / Y_O1, `flameIC2`). **C2b = `FvChem::law`** (`PhiRateLaw`, module `C2b`):
  `kFvSpCellPhi` puts `arrheniusRate2(lnA + phiRateLn(law, Y_F, Y_O), ...)` in the cell pass
  (the structured `kChemCellPass2Phi`), `ChemJ2Phi` the same ln C in FP64 into the chemistry
  dt (`ChemRate2Phi`). Their own kernel and functor: `law.n == 0` launches the C1 ones bit for
  bit, and a law with C(xi) == 1 equals C1 byte for byte (`gpu_fv_matrix` inert column).
  FvRef: the same expression per cell, `chemDt` likewise.
- **Patches.** The box: x-lo `Far` at the fresh premix (Y_F1, Y_O1 of `--mix-phi`'s phi when
  given), x-hi burnt `Outflow` (rich: Y_F,rem, 0; lean: 0, Y_O,rem). A polyMesh: the same
  rule on every `far` / `inflow` and every `outflow` patch; a `y` in the file is exit 2 by
  name (the flags are the one source). `tools/chem/fv_flame_mesh.py --premix` writes the
  y-free file, `--nz N` a 3D box (z cyclic).
- **Stats**: the C0 flame's columns with S_L on rho (Y_F - Y_F,rem) (`FvFlameSpec::yBurnt`,
  the structured `FlameRun::yFBurnt`), then `zMin`, `zMax` = `mixtureFraction(Y_F, Y_O)` over
  the cells (runner_flame's). `.json` appends `model` (C2 | C2b), `phi`, `zst`.
- **LES** (`--les CS --sgs smag|sigma`, `--nz >= 8` on the box, a 3D mesh on `--polymesh`):
  the `Eddy` instantiation of the two-reactant face kernel, one rho D + nut_f/Sc_t for every
  species (MC's C1 + LES), the law in the cell pass. **TFLES refused** by the inventory's text
  (`--tfles`, `--tfles-sensor`, `--efficiency`, exit 2): the thickening, its sensor and
  Charlette's E are the premixed step's; a two-reactant TFLES would thicken the partially
  premixed jet's outer sheet too, which no model here defines. The structured `--equivalence`
  keeps refusing `--les`.

Pairs (inventory, "Combinations"): C2b + LES smag / sigma live (harness + `gpu_fv_flame_rich_les`);
C2b + gravity, + bodyForce live on the harness (the law is a rate factor: transport, buoyancy
and the drive are C1's; the CLI keeps `--gravity` off the flame: open patches); C2b + TFLES
(uniform, dynamic, Charlette) refused; C2b + RANS refused (species + RANS); C2b + steady LTS
refused (`kFvChem` bare cr); compensated and FP64 live.

### Air-methane verification

`gpu_fv_flame_rich` (R1, R2), `gpu_fv_flame_rich1d`, `gpu_fv_flame_rich_les`
(`tests/test_gpu_fv_flame_rich.cpp`), the generic cells C2b, LES smag / sigma + C2b,
C2b + gravity, C2b + bodyForce of `gpu_fv_matrix`. Ladder: sigma 6.88, beta 8, Sutherland 298 K,
M 0.01, phi 1.3, C2b, u_in 0.98, tt 20, window [10, 20] (gpu_flame_rich1d's protocol).

| pair | mesh | quantity | expected | measured |
|---|---|---|---|---|
| FV(FP64) vs FvRef, generic gates (`gpu_fv_matrix`) | 4 meshes | every pass, 3 steps; numberings; rest; pow2 Y; Z == tracer; C(xi) 1 vs C1 | 1e-13; bitwise | oracle 4.8e-15 (LES sigma 1.3e-14); orders 3.2e-16; Z 1.3e-15 FP64, 7.4e-7 FP32; pow2 exact; C(xi) 1 = C1 bytes, chem dt equal |
| device omega vs `phiRateLn` + `arrheniusRate2` restated (R2) | uniform 128 x 32, stratified premix | omega, FP64 | 1e-13 | 7.4e-15; C(xi) 1.004..2.98, C(jet) - 1 = 6.7e-16 |
| FV(FP32) vs structured `kSpeciesStage2` + `kChemCellPass2(Phi)` (R1) | uniform periodic 128 x 32, burning slab, premix phi 1.17..1.3, A / 10, M 0.05, same dt, 400 steps | full state | rounding: <= 3 x FV FP32 vs FP64 | C2: 3.6e-5 (own 9.3e-5), structured vs FV FP64 9.8e-5; C2b: 4.8e-5 (own 5.5e-5), 7.2e-5. At the flame's A the non-equilibrium slab explodes (T 2 Tb, ox < 0) in all three: chaotic, not a gate |
| FV(FP64) vs oracle | 1D box, 7.5 / 15 / 30 cells per delta_th | S_L - 1 | order 2 about the O(M^2) offset (-8.2e-4) | +1.53e-2, +2.98e-3, +1.23e-4; about the offset 1.61e-2, 3.80e-3, 9.5e-4 (2.08, 2.00). Window spans 7.2e-4, 2.2e-4, 6.0e-5 |
| same | same | delta_th | -> 4.70964 | 4.59387, 4.68800, 4.70774 |
| structured (FP32) | same box | S_L | not rounding: NSCBC vs `Far`, and the FP32 Z walk through C (d ln S/dZ -7.3, docs/chem.md "Model C2b") | 1.015555, 1.002132, 0.996512: the 30-cell rung reads below 1, the walk's sign |
| FV(FP32) vs structured (FP32) | same | S_L | same, both walk | -4.9e-4, +1.6e-4, +2.7e-5 (FV FP32 0.996539 at 30: the same walk) |
| FV(FP64) | same | Z = mixtureFraction over the box and run | exact | 1 to the csv digits at every rung (structured FP32: +-1.7e-4) |
| FV(FP64) `--mix-phi 1.2` vs oracle ratio r 1.417470 (rich1d) | 15 cells | S/r - 1; (S_mix/S_jet)/r - 1 | 5e-3; 3e-3 (gpu_flame_rich1d_law's) | +2.37e-3; -6.1e-4 (structured FP32: +1.3e-3, -8.1e-4) |
| rotated 30 deg vs planar | 7.5 / 15 / 30, ny 16 / 32 / 64, FP64 | S_L; delta_th | rounding of the geometry | -2.7e-7, 0, 0; +2.1e-5, +3.4e-6, -1.6e-6 |
| sheared 30 deg vs planar | same | S_L; delta_th | order 2 | -1.56e-3, -4.42e-4, -1.13e-4 (1.82, 1.96); -4.74e-2, -1.20e-2, -2.86e-3 |
| 3D laminar LES vs DNS (`gpu_fv_flame_rich_les`) | box 7.5 cells, ny 8 nz 8, FP64 | S_L | sigma: D_sigma 0 on a planar flame; Smagorinsky: mu_t of the dilatation | sigma 1.015320 = DNS 1.015320 (0 to the csv digits); Smagorinsky 0.17 +5.3e-4 |
| README example: LES sigma + C2b on a rotated (20 deg) 3D polyMesh | 165 x 8 x 8, 7.5 cells, `--compensated` | S_L, delta_th, Z, hits | = the planar FP64 box | 1.015320, 4.59413, Z 1, 0 hits; 66545 steps, 17 s |

All rows: 0 cells outside `[-1e-6, 1 + 1e-6]`.

Not measured (3 levels, 295 s): a 60-cell rung, a wavy mesh, `--mix-phi` below 1 on FV, a
turbulent LES flame (no synthetic inflow on the route: `--turb` exit 2):

    tools/chem/fv_flame_ladder.py --cells 60 --tt 20 --dth 4.70964 --meshes rot,shear,wavy \
        --extra "--equivalence 1.3 --expansion 6.88 --visc sutherland --tinf 298"

## Turbulence (`--turb`, masterplan_multiphysics M5)

`dns flame --solver fv --turb U [--turb-length L --seed S]` on the flags' box, premixed C0
and C2b (`--equivalence`). The generator is `dns/turb_ic`'s (hash-seeded per wavevector,
the reference set, `flameTurbSpec`: the structured box, nbx = nz columns); the FV route
reads it as its Fourier sum at points (`dns/fvturb`):

```
IC     every centroid: m += rho w u'(C), E += the kinetic-energy change (p kept),
       w = turbWindow(C_x, x_f0 - 1.5 l_t, l_t/4): flame1d addTurbulence's arithmetic per cell
       (turbAt: turbPoint per point, threaded, bit-identical at any thread count; w == 0 skips the sum)
inflow the xlo Far patch gets one FvBc row per face (FvCompConfig::bcRow, rows after the patch rows);
       before every step the box swept to X = sum U dt (FP64 host) is sampled at the face centres
       (TurbSweep: the sum grouped by mx, B_p(mx) built once) and each row rewritten as fvBcOf's Far
       row with velocity U dir + du (fvFarTurbRow), one upload (DeviceFvComp::setBcRows); held over
       the step's three stages
```

No device generator, no stage term: the module `Turb` (dns/fvmodules, axis `inflow`) is
live with C0, C2b, LES (smag, sigma), TFLES (uniform, dynamic, Charlette) and both
precision modules; refused by name with RANS (fluctuations on an ensemble mean), C1/C2 (the
jet and the one-step C2 premix: no turbulent number), gravity (no open patch under it),
the body force (a closed channel), local time stepping (the sweep is time-accurate); it
requires C0 or C2b. Refused on the route: `--polymesh` (the box's periods and lattice are
the flags'; issues.md M5c1), `--hold-flame`, `--mode diffusion`, `--sl-law onestep`. The
3D stats gain `area` (sum V |grad Y_F| / A, the structured column's co-area integral; the
least-squares gradient is central on the box) and `I0` = SL/area.

Verification (gfx1201, 2026-11-03; `tests/test_gpu_fv_turb.cpp`, ctest `gpu_fv_turb` 3 s):

| gate | case | measured | relation |
|---|---|---|---|
| off | `tools/fv/comp_bit_identity.sh` (36 cases, + `flameturb`, `lm3d`, `fvlm3d`) vs the pre-M5 binary | every artefact | byte-identical (0 failures) |
| T1 IC | the CLI's box, 4 cells/delta_th, 24^2 cross-section (41472 cells, 12144 modes) | FV point sum vs the structured `addTurbulence(turbBox)`: m 3.7e-14 (u' 1), E 2.0e-16 of max E | rounding (direct FP64 sum vs FFTW c2r) |
| T2 zero amplitude | the same box, xlo rows rewritten every step with u' 0, 60 steps | FP32, compensated, FP64 | byte-identical to the laminar table |
| T3 rows reach the flow | u' 1 rows, FP64, 60 steps | first cell layer max \|v\|, \|w\| 0.54 u'; mass vs laminar 7.1e-7 | plane-mean u' 0 |
| X3 FV vs structured | burning slab + turbBox velocity, 48 x 16 x 16 periodic, M 0.05, 200 steps, same dt | FV32 vs structured 1.77e-5, structured vs FV64 1.93e-5, FV32 vs FV64 1.66e-5 (transverse KE decays to 1.3%: Re_t ~ 1) | rounding, <= 3 x own |
| matrix | `gpu_fv_matrix`: 7 Turb cells x 4 meshes (C0, C2b, + LES smag / sigma, TFLES uniform / dynamic, Charlette) | oracle <= 1.3e-14, orders <= 4e-16, mass, energy, pow2 Y, Z, rerun; inert: rows of u' 0 = the cell without Turb bitwise (FP32 plain and compensated) | every generic gate |

The inflow itself is a STATED GAP, not rounding: the structured flame relaxes its NSCBC
inflow CELL (x = h/2) to the box's lerped column, the FV route sets the Far FACE (x = 0) to
the exact sum. The IC's box sits on the flame at t = 0 and the inflow's needs ~13 flame
times to arrive (6 delta_th + the 16 sink cells at u_in 1), so the physics window below is
the IC's.

Physics number (`tools/chem/turb_fv.sh`, M 0.02, u'/S_L 4, l_t 2 delta_th (Re_t 17.8), L_y =
L_z = 6 l_t, 6 fresh + 4 burnt delta_th, means over t in [0.5, 2]; structured = the
NSCBC + sink solver, FV = this route on the same mesh, FP32):

| cells/delta_th (mesh) | S_c structured | S_c FV | FV - st | area structured | area FV | laminar S_c st / FV (same window, 8^2) |
|---|---|---|---|---|---|---|
| 4 (72 x 48 x 48) | 1.09880 | 1.09638 | -0.22% | 1.03683 | 1.03632 | 1.03957 / 1.03289 (-0.64%) |
| 6 (92 x 72 x 72) | 1.06125 | 1.05230 | -0.84% | 1.02815 | 1.02633 | 1.03358 / 1.02873 (-0.47%) |
| 8 (112 x 96 x 96) | 1.05237 | 1.04039 | -1.14% | 1.03422 | 1.03135 | 1.02003 / 1.01566 (-0.43%) |

FV `--compensated` moves S_c by < 1.1e-4 (1.09644, 1.05239, 1.04050): not the FP32 update
floor. The gap is NOT rounding and does not shrink over the three levels (area -0.05%,
-0.18%, -0.28%): the window is the start transient of an M 0.02 box, and the laminar flame
already differs by 0.4-0.6% there between the NSCBC + sink structured boundaries and the FV
Far / Outflow faces; the low-Mach pair, whose boundaries are the same face conditions on both
solvers, agrees to 3e-4 on the same turbulent case (docs/chem.md "Low-Mach turbulence"). Read
as a stated boundary gap (issues.md M5c2), 3 levels by decision, no limit claimed. FV cost: 1.2
ms/step at 41k cells (the rows' sample + upload included).

The pairs, 4 cells/delta_th (72 x 48 x 48), the same case and window:

| cell | S_c structured | S_c FV | area structured | area FV | S_c / laminar S_c, st / FV |
|---|---|---|---|---|---|
| C0 + turb (DNS) | 1.09880 | 1.09638 | 1.03683 | 1.03632 | 1.0570 / 1.0615 |
| + LES smag 0.17 | 1.10044 | 1.09778 | 1.03636 | 1.03585 | |
| + LES sigma 1.35 | 1.09881 | 1.09627 | 1.03632 | 1.03582 | |
| + TFLES F 2 (dynamic) | 1.15741 | 1.04206 | 1.03387 | 1.03401 | 1.0222 / 1.0238 (laminar TFLES 1.13224 / 1.01787) |
| + TFLES F 2 + sigma + Charlette | 1.15999 | 1.04187 | 1.03218 | 1.03231 | 1.0245 / 1.0236 |
| C2b phi 1.3 + turb | (not built) | 1.10562 | | 1.03449 | low-Mach partner: MAC 1.04023, FV 1.04078 |

LES and the turbulent pairs keep the C0 gap (-0.24%, -0.23% S_c, -0.05% area); TFLES and
Charlette agree on the flame surface to 1.4e-4 and on S_T/S_L to 0.16%, but the LAMINAR
thickened flame already differs by 10% in this window between the two compressible solvers
(1.132 vs 1.018; low-Mach 1.106): a pre-existing TFLES gap of the 3D M 0.02 start, no `--turb`
in it (issues.md M5c6). C2b has no structured compressible --turb; its compressible FV S_c
(1.106) carries the M 0.02 start transient the low-Mach pair (1.040, agreeing to 5e-4) has
not.

`issues.md` M5c.

## Gates

`fv_comp` (host, `tests/test_fv_comp.cpp`): the `--bc` parser, the binding, the
FvBc table, the CLI routes and rejections, the raster, the flame route, `fvFaceY` vs
the structured stencil (1e-15), `FvFlameProbe`, FvRef species at 1 vs 4 threads,
`FvJetProbe` vs `measureJet` (1e-5, FP32 fields), the shared limiter's linearity and vote, the
characteristic Outflow's entropy and invariant.

`fvref` adds W1, W2 ("Gravity"), W3 ("Hydrostatic ambient"); `gpu_fv_candle_rest` (56 s), `_g0` (2 s), `_height` (102 s) the candle.

`gpu_fv_comp` (device vs FP64 oracle, `tests/test_gpu_fv_comp.cpp`):

| gate | measured | bound |
|---|---|---|
| O1 double vs FvRef, box_cyclic (adiabatic; isothermal moving wall) | rhs 3.5e-16, gradients 0, fluxes 1.5e-16, 3 RK3 steps 4.2e-16, dt 0 | 1e-13 |
| O1 box_refined (split faces, Sutherland) | rhs 2.2e-15, gradients 1.2e-15, fluxes 2.5e-16, steps 5.8e-16, dt 0 | 1e-13 |
| O1 cyl2d_snappy (spinning isothermal wall, inflow, outflow, slip, far) | rhs 1.1e-15, gradients 3.2e-16, fluxes 2.5e-16, steps 6.4e-16, dt 1.1e-16 | 1e-13 |
| O2 float vs double, same meshes | rhs 9.1e-7 / 3.3e-6 / 1.3e-6, 3-step state 2.5e-7 / 3.2e-7 / 2.9e-7 | 2e-5, 2e-6 |
| O3 device (double) vs `Comp::rhs`, uniform periodic | 3.0e-15 | 1e-13 |
| O4 freestream, cyl2d_snappy, every patch Far | FP64 max\|rhs\| 2.0e-14, 100 steps 1.4e-15; FP32 1.2e-5, 100 steps 6.9e-7 (Sum S rounds, p/V amplifies; noise, stated) | 1e-12; 5e-5, 2e-6 |
| O5 rerun, 40 steps, 3 meshes, both precisions | memcmp equal | exact |
| O6 adiabatic wall, cyl2d_snappy, both precisions | 32 faces, mass and energy flux 0 | exactly 0 |
| O7 2D, both precisions | z flux, rhs, d/dz, rho w after 20 steps 0 | exactly 0 |
| O8 inviscid KE, box_refined | KE rate - pressure work 1.4e-16 (uniform p), 6.7e-17 (random p); 20 steps: mass 6.7e-16, energy 4.4e-16, momentum 6.9e-18 | 1e-13; 1e-14, 1e-13 |
| O9 LES vs host restatement on FvRef's gradients, curvilinear box | smag mu_t 3.4e-16, rhs 4.4e-15; sigma mu_t 2.0e-13, rhs 4.8e-15. The bound on mu_t is 3 x the host operator's own response to 1 ulp of grad u (3.4e-16, 2.0e-13): sigma's conditioning, measured in the gate | see left; 1e-12 |
| O10 tally, uniform p on a closed wall | force 0, 5.6e-17; viscous 0 | 1e-12 |
| O11 compensated vs FP64, dt 2% of stable, 4000 steps | 8.2e-7 (plain 1.8e-4), rerun identical | 0.2 x plain |


`gpu_fv_flame` (S1-S8, table in "Species cross-verification"; D1-D3, "Diffusion flame";
`gpu_fv_flame1d`: the binary at 7.5 cells, S_L 1.01681, delta_th 1.56669, 0 hits at t 12,
bounds 2%, 3%; `gpu_fv_flame_mix`; `gpu_fv_flame_jet`: the binary at defaults, FP64, t 80,
window [60, 80]: tip 47.23, TmaxIn/T_ad 0.98475, burn/in 0.9975, out/in 3.5e-11, p/p0 1.0017,
bands gpu_diffusion_jet's, no tip pin: the ring is still in that window):
`tests/test_gpu_fv_flame.cpp`.

`gpu_fv_flame_rich` (R1, R2), `gpu_fv_flame_rich1d`, `gpu_fv_flame_rich_les`: "Air-methane
verification".

`gpu_fv_flame_les` (L1-L5, X1) and `gpu_fv_tfles1d` (the binary, F 2): table in "Species
LES and TFLES verification", `tests/test_gpu_fv_flame_les.cpp`.

`gpu_fv_cross` (X1-X3), `gpu_fv_ladder` (X4, every route's smoke run), `gpu_fv_far`
(X5, settled gap, 95 s: 192x96 -4.1e-4 <= 6e-4, order 1.80 >= 1.5):
`tests/test_gpu_fv_cross.cpp`.

## Gravity: `dns conv --solver fv`

```
dns conv --solver fv [--physics compressible] --n N --ny M [--nz K] [--sides periodic|closed]
         --rayleigh RA [--mach M] [--dtemp E] [--tt T] [--mesh-perturb A [--mesh-perturb-kind sine|random]]
         [--fv-precision fp32|fp64] [--frame-n N] [--frame-box ...]
dns conv --polymesh DIR --bc FILE.toml --ny BINS [same physics flags]
```

Same physics as `gpu/comp_conv`, no Boussinesq: buoyancy is rho(T) g through the EOS.
`FvCompConfig::gravity` is two additive passes, `faceFlux` and `kFvGather` untouched:

- `kFvGravFace`, after the fluxes, on every internal face (`fvGravityFace`): `(g.d/2)
  rho_f S` on momentum, `(g.d/2) mdot` on E, d = C_nei - C_own, mdot = `F[0]`.
- `kFvGravity`, after the gather: ascending CSR faces, BOTH cells add the term, unsigned
  (a source, not a flux), over V.
- `rho_f` (`fvHydroRho`): `L(p_P, p_N) / (R L(T_P, T_N))`, L the log mean, series near
  equal arguments (`fvLogFactor`). FvRef carries all of it.
- Isothermal plates are FvBc rows, `fvBoundaryFlux`, no ghost. Side walls are
  FaceMesh walls, adiabatic to exactly 0.
- IC: `convHydroAt`, `convTripAt` at the cell centres. The box is a FaceMesh with its
  floor at y = 0 (`dns/fvconv`); `--polymesh` is the same route, y up, plates flat at
  y = 0 and 1, wall `twall` equal to the flags' (floor `1 + dtemp/2`, lid
  `1 - dtemp/2`), else exit 3 by name. wall, slip, empty, cyclic only: an open face has
  no hydrostatic ambient. Moving walls, `[reference]`: rejected. `--ny` on a polyMesh =
  height bins of the plane statistics, required; `--n`, `--nz` belong to the mesh.
- Rejected by name: `--hot-patch` (not built), `--planes`, every flag off the allow-list.

Stats: `nuBot, nuTop, nuFlux, urms, KE` are `runner_conv`'s on height bins (a bin is a
row on the box), `nuBotF, nuTopF` the conservative wall-face conduction, plus
`machMax, mass, energy`.

**FP32 is compensated by default here.** `--cfl` means 2x less dt than the structured
solver (the Sigma form), so a plain state drops 2x the updates: Nu 3.671 / 3.698 plain vs
3.745 / 3.728 compensated vs 3.745 / 3.729 FP64, 192 x 96. `--compensated` costs nothing
measured (50.6 vs 50.8 s). 96 x 48: 3.7685 plain, 3.7782 compensated, 3.7781 FP64.

**Well balanced, on any mesh.** The KG pressure sum of a closed cell is `Sum (p_N - p_P)/2
S`: that closure is the whole scheme's gradient, and on a random mesh its operator `T =
(1/V) Sum d S/2` is not the identity (zeroth order). A cell source `rho g` cannot balance
`T grad p` (it was 1.1% Uff, flat). So gravity takes the pressure's own form: the same sum
over the same faces, `T rho g`, and the rest residual is `Sum (p_N - p_P - rho_f g.d)/2 S`.
What is left is one face relation, and it must be path independent, else no field solves
it on a loop of cells. A mean of two rho is not. With log means: T = T0 + a h(x),
`ln(p_N/p_P) = g.d/(R L(T))` integrates dp/dh = -rho g EXACTLY for T linear in the height
h, a = 0 included, on any pair of points; `p_N - p_P = L(p) ln(p_N/p_P)`. So:

- **The conduction state at cell centroids is an exact fixed point of the discrete
  equations on ANY mesh**, FP64 to rounding (W1, C8). Conduction closes too: a linear T has
  an exact LSQ gradient, an exact corrected face gradient, Sum S = 0 (constant kappa). Not
  exact: Sutherland (kappa(T): a nonlinear conduction T), a curved adiabatic wall (a
  linear T is not adiabatic there), any non-linear T(h): truncation there, order 2 in
  the face's height step, not a fixed forcing.
- **A Wall or Slip face adds nothing.** Its hydrostatic image (`dp = rho_P g.r`, the
  structured plates') and that face's share of the term are one number, opposite: the wall
  cell's normal balance is `-(p_1 - p_0)/(2h) + rho_f g/2`, which the old pair spelled with
  rho_P. So there is no image pass; the force tally reads p_P as before.
- **`Sum V (E + rho phi)` telescopes** on a closed box, phi = -g.x (W2): mass flux `F[0]`
  times the same `g.d/2`. Not `g.m` of a cell. Kinetic energy and that energy term are
  consistent to O(h^2), as the KG pressure work.
- **The buoyancy of a perturbation carries T too**: on a random mesh T(rho' g - grad p'),
  not the true force. It was already T grad p'. Effect at Ra 5e4: the table.
- Uniform mesh: `(rho_f,j+1/2 + rho_f,j-1/2)/2 g` against the structured `rho_j g`,
  O(h^2) apart; Nu below. g = 0 launches neither pass: the solver byte for byte.

FP32 floor: the state is two-word but the fluxes are FP32, so `p Sum S` and the sum's own
rounding leave `eps p/h`, a frozen forcing, 1.2e-6 to 2.2e-6 urms; FP64 rounding, 2.5e-15.

| pair | mesh | quantity | expected | measured |
|---|---|---|---|---|
| FV(FP64) vs FvRef + gravity (C1) | sine 0.08 24 x 12 | rhs, 3 steps | rounding, gate 1e-13 | 6.7e-15 of 3.5, 1.8e-15 of 2.6 |
| conduction state, FvRef (W1) | uniform, sine 0.08, random ±0.3 h, 24 x 12, T linear and isothermal | max\|rhs\| | rounding of p/h | 2.7e-14 to 3.7e-14, bound 9.6e-14 |
| `Sum V (E + rho phi)`, mass, FvRef (W2) | random ±0.3 h, closed, adiabatic, 200 steps of a sloshing state | relative change | rounding, gate 1e-13 | 1.8e-15, 8.9e-16 |
| rest (old face mean -> this form), urms / Uff (Uff 0.0561), kick 0, Ra 1000, t 20 x H/Uff, FP32 compensated | uniform 64 x 32 | urms | rounding-level | 3.8e-6 -> 2.7e-6; structured 5.3e-6 |
| same | sine 0.08, 32, 64, 128 | urms | floor | 6.4e-4, 1.5e-4, 6.7e-5 -> 3.2e-5, 3.2e-5, 3.9e-5 (the FP32 floor, no h trend) |
| same | random ±0.3 h, 64 x 32 | urms; machMax; nuFlux; nuBotF, nuTopF | floor | 1.1e-2 -> 2.1e-5; 2.4e-3 -> 2.6e-6; 0.613 -> 1.00001; 0.998, 1.002 -> 0.999997, 1.000000 |
| same, FP64 | random ±0.3 h, 64 x 32 | urms; machMax; nuFlux | rounding | 1.1e-2 -> 4.4e-14; 2.4e-3 -> 2.0e-14; 1.000000 |
| same, FP64, 3D (C8's sibling) | random ±0.3 h, 32 x 16 x 16, t 2 | urms | rounding | 2.4e-3 absolute -> 3.8e-15 absolute |
| subcritical | uniform 64 x 32, Ra 1000, kick 0.02 (C5) | Nu, urms end | Nu 1 to 1% | 1.0000 / 1.0000, 1.9e-6 absolute at t 40 |
| FV vs structured, Ra 5e4 2D | 48 x 24, 96 x 48, 192 x 96, t 40 | tail nuBot | shrinks | FV 3.926, 3.777, 3.745 (old form 3.914, 3.778, 3.745); structured 3.940, 3.774, 3.736; diff -0.35%, +0.09%, +0.25%. Not monotone: 3 rungs of a time mean, structured plain FP32 |
| anchor | 96 x 48 | nuBot | 3.79 (issue); gate 2% vs structured | 3.777 FV (was 3.778); 3.774 structured |
| heat balance | same 3 rungs | spread of nuBot, nuTop, nuFlux / nuBot | < 5% | 1.7%, 0.4%, 0.4% |
| face vs gradient Nu | same | nuBotF vs nuBot | first order at the wall | 3.857, 3.769, 3.744 vs 3.926, 3.777, 3.745 |
| FV compressible vs FV Boussinesq | 96 x 48 | nuBot | 15% (`gpu_rb`) | 3.777 vs 3.960: 4.7% |
| closed sides | 96 x 48 aspect 2 | nuBot, nuTop, nuFlux | structured | FV 3.380, 3.277, 3.322 (was 3.381, 3.278, 3.323); structured 3.386, 3.273, 3.325 |
| non-orthogonal Ra 5e4 | sine 0.05, 96 x 48 | nuBot, nuTop, nuFlux | uniform's | 3.775, 3.785, 3.732; face 3.764, 3.746 |
| critical Ra | 64 x 32, periodic, kick 0.02, t 80 | urms trend | ~1708 bracket | Ra 1600 2.8e-4, 1.6e-4, 9.2e-5 decays; Ra 1900 1.2e-3, 2.6e-3, 3.7e-3 grows |
| random Ra 5e4, settled (old -> this form) | random ±0.3 h, 96 x 48, t 120 H/Uff, last 60% | nuBotF, nuTopF; nuFlux | uniform same run 3.733, 3.733; 3.733 | 3.714, 3.798 -> 3.752, 3.757 (+0.6%); **-4.67 -> 3.763**. nuBot, nuTop read 3.93, 3.94: bins smear (81e) |
| same | random ±0.3 h, 192 x 96 | same | uniform 192 at t 120: not measured | 3.605, 3.624; 1.44 -> 3.613, 3.609; 3.600. New run cut by the 295 s cap at t 2003 of 2138 (106 of 120 H/Uff). 3.5% below the 96 x 48 random run; urms 1.32e-2 vs 1.51e-2: a different roll state is suspected, not shown |
| settling | random ±0.3 h, 96 x 48, same run | nuBotF, nuTopF, windows of H/Uff | none claimed | this form 3.26, 3.20 over t 16-40, 3.72, 3.75 over t 40-60, flat after; the old form 3.63, 3.71, then 3.72, 3.79: read a random mesh past t 60 |
| `--polymesh` = box (C6) | sine 0.05, 32 x 16, binary polyMesh | stats rows | bytes | identical |
| byte | `dns.fvbase2` vs this tree: g = 0 paths | `tools/fv/comp_bit_identity.sh` | identical | 24 cases identical, 0 failures, FV flame at nY 1 included; g != 0 FV results move on purpose |

Max non-orthogonality at 64 x 32: sine 0.05 26.2 deg, sine 0.08 40.6 deg, random 0.3 60.0 deg. 3D ran as smokes only:
48 x 24 x 24 closed t 4 finite, onset not reached. No 3D Nu is quoted.

## Hydrostatic ambient: open patches under gravity, the candle

`FvConfig::ambient` (`FvAmbient`: p0, T0, x0) is the isothermal column `p_a(x) =
candleAmbientP(p0, |g|, R T0, h)`, `h = (x - x0).up`, `up = -g/|g|`: the structured candle's ONE
definition (`dns/diffusion`), the same numbers when g is along -x and x0 on the inflow plane.
`fvAmbientRows` gives every face of a `Far` / `Outflow` row a row of its own (`FvConfig::bcRow`,
no device path, no kernel edit): Far's `qFar` the ambient's conserved state at the OWNER'S
CENTROID moving with the row's velocity (`fvAmbientConserved`), Outflow's `pOut` p_a there.
`Inflow` needs no row: its p is the owner's (`p_P + rho c (u_n,P - u_n,in)`).

- **Why the owner's centroid, not the face centre.** A boundary face carries no gravity share
  (`fvGravityFace` is internal), so its pressure must be the owner height's, exactly as a Wall
  face's is p_P ("A Wall or Slip face adds nothing" above): the missing `rho g.r` between
  centroid and face is that half cell's absent share. With the row at the centroid the rest
  state crosses an open face as `P == F` (Far returns F exactly, the characteristic Outflow
  `pOut == p_P`): the column is a fixed point through Inflow, Outflow and Far faces to the same
  rounding as the closed box, on any mesh, and the incoming characteristic of a moving state
  compares the owner with the ambient AT ITS HEIGHT (the structured candle's hydrostatic NSCBC
  waves, `Config::hydroLo/Hi`, in face-flux form). A face-centre row would need a boundary
  gravity term and a hydrostatic shift inside `fvBoundaryState`: more device code, rest no
  longer exact.
- **Checked, not trusted** (`FvRef::make`, so `DeviceFvComp::make` too): gravity with a Far or
  Outflow patch and no `ambient`: exit by name (`needs the hydrostatic ambient`); every such
  face's row p (Far: and T) within 1e-12 of the column at its owner: else by name (`not the
  hydrostatic ambient's row`); `ambient` without gravity: refused. Moving walls stay refused.
  `dns conv` keeps its closed layer (an open patch there is refused by name: conv's experiment).
- **IC** `fvAmbientState` (rest) and `fvJetState` under gravity (`p = p_a(C)`, `rho =
  p_a/(R T)`: the structured candle IC) read `fvAmbientP`: the rows' bits.

`dns flame --mode candle --solver fv` (and `--polymesh DIR --bc FILE`, g along `-direction`,
x0 = `origin`): the FV jet ("Diffusion flame") + `FvCompConfig::gravity = -g dir` + the ambient
rows on its Outflow (Far) patches (`fvJetRows`, `fvJetAmbient`). Under gravity an Outflow
backflow is the ambient's air at T0 (`y = {0, Y_O2}`, not `ownY`): air drawn down through the
top, the structured `ambientHi`. `--quiescent`: every inflow face and the IC at rest, no
extinction guard. Stats: the jet's 21 columns + `uMax`, `uOutMin` (the structured candle's);
`pMean` against `Sum V p_a`. g 0 is the FV jet byte for byte (`gpu_fv_candle_g0`). `--wick`,
`--les`, the structured-only knobs: exit 2 by name.

**The candle's top is a `Far` row, the jet's `Outflow` only at g 0.** Measured first with the
jet's fixed-p characteristic Outflow at the top (8 cells/l, M 0.01, defaults): an organ pipe the
flame pumps. p/p0 swung +-0.7% sustained to t 30 (the g 0 jet's ring decays to +-0.2%), air
drawn down at uOutMin -11 to -23 U (structured -7 to -11), 3600-4000 cells with Z below -1e-6
(Y_O 3e-5 above Y_O2), tip 6.48 l; at 6 cells/l tip 6.3 -> 4.6 inside the window, at 4 cells/l
extinction (exit 8, t 25). A `Far` top (characteristic, the ambient's air at T0 moving with the
coflow, `--quiescent` at rest) radiates the mode: steady, no hits, tip h-independent. The
polyMesh route takes either (`type = "outflow"` keeps the rows of `fvAmbientRows` too);
`tools/chem/fv_flame_mesh.py --jet` + `[patch.xhi] type = "far"` reproduces the box to the
printed digits (tip 7.1333 both).

### Hydrostatic ambient verification

| pair | mesh | quantity | expected | measured |
|---|---|---|---|---|
| column at rest, FvRef (W3) | 24 x 14 closed + open box (Wall floor, Inflow at rest, characteristic Outflow lid, Far side): uniform, oblique g 30 deg (the rotated mesh's frame), random +-0.3 h | max\|rhs\|; max\|u\| after 200 FP64 steps | rounding of p/h | 9.3e-15, 1.6e-14, 2.0e-14 (bound 9.5e-14 - 1.2e-13); 8.6e-16, 7.5e-16, 4.7e-16 |
| refusals (W3) | same | open patch without ambient; ambient without g; a Far row 1e-6 off | exit by name | all three |
| FV(FP64) vs FvRef, ambient rows (`gpu_fv_candle_rest`) | 48 x 28, the three meshes, a moving state | rhs, 3 steps | 1e-13 | 6.4e-15, 6.4e-15, 4.6e-15 |
| FP32 rest, 4000 steps | same | max\|u\|/c, first / second half | flat, FP32 floor | 20000 steps (18-25 acoustic crossings): 1.11e-6 / 1.12e-6, 1.01e-6 / 1.22e-6, 4.7e-7 / 5.1e-7 |
| matrix (`gpu_fv_matrix`) | cyl2d_snappy open set (inflow, outflow, far, slip, wall) at rest, every gravity cell | open rest FP64 / FP32 | rounding | 3.4e-16 / 1.5e-7 (C0, C1, C2b, LES, TFLES, bodyForce cells: <= 1.0e-15 / 4.5e-7) = the closed set's |
| quiescent candle, binary (`gpu_fv_candle_rest`) | 288 x 200 (8 cells/l), M 0.005, tt 10, the structured `gpu_candle_rest` case | max\|u\| / U; max\|p/p_a - 1\|; hits | structured 1.21e-3 (NSCBC hydrostatic waves; 0.67 without) | Far top: 6.3e-4 (halves 6.3e-4 / 6.3e-4), 0, 0; FP64 tt 2 8.0e-13 (Outflow top before: 1.16e-3 FP32) |
| same, `--compensated`; FP64 (Outflow top, before the Far top) | same | max\|u\| / U, halves | rounding | 6.6e-5 / 6.4e-5; 1.6e-12 / 1.7e-12: the FP32 number is p's rounding (p = T0 = 28571), the scheme's rest is exact |
| g 0 = FV jet (`gpu_fv_candle_g0`) | W 4, 4 cells/l, tt 1 | 21 stats columns | bytes | 17 rows, 0 differ (tt 3) |
| FV candle alone | 4, 6, 8 cells/l, M 0.01, defaults, window t 20-30 | tip; TmaxIn/T_ad; fuelIn; half | converging | 7.154, 7.133, 7.133; 0.9756, 0.9749, 0.9748; 0.8862, 0.8833, 0.8792; 1.340, 1.342, 1.345; halves within 0.6%, hits 0 |
| FV vs structured etaIn 1000, same mesh | 4, 6, 8 cells/l | tip; TmaxIn/T_ad; fuelIn | the inflow model's gap (FV fixed face vs a pinned cell, "Diffusion flame") | structured 7.07 (flapping, halves 7.96 / 6.18), 7.88 (8.20 / 7.55), 7.94: FV +1.2%, -9.5%, -10.2%; TmaxIn 0.9702, 0.9729, 0.9716: +0.6%, +0.2%, +0.3%; fuelIn 0.958, 0.939, 0.929: FV -7.5%, -5.9%, -5.4%. Roper's buoyant H ~ Q^(4/3) explains -7.4% of the -10.2%: residual ~3%, no h trend in it established (structured is unsteady below 8) |
| FV vs structured default (etaIn 100) | 4, 8 cells/l | tip; fuelIn | NOT the same inflow (the etaIn 100 row overshoots, 39% more fuel at W 1) | 8.69 (unsteady), 10.19; 1.22 at 8 cells/l |
| g 0, FV vs structured default | 8 cells/l | tip; fuelIn | the momentum jet, same inflow gap | 10.42 vs 21.44; 0.717 vs 1.025 |

## Performance

O-grid 256x128 (32 640 fluid cells), Re 40, gfx1201, ms/step: FV plain 0.182, FV
`--compensated` 0.186, structured plain 0.107: FV costs 1.70x the structured O-grid
per step, compensation 1.02x. 384x192 compensated: FV 0.197, structured 0.120.
Launch-bound at this size (six launches per stage). Cell order is Morton, not
compared with Source / colour. FP64 operations in the float kernels
(`fvPrimOf`'s `T(gamma - 1.0)`) not profiled.

## Reproducing the long numbers

    simjob submit --gpu -t 280 -m 8192 --wait -- build/test_gpu_fv_cross far
    tools/fv/comp_ladder.py --re 40 --ft 3 --levels 96:50,192:98,384:194 --compensated
    tools/fv/comp_ladder.py --re 40 --ft 3 --levels 192:98 --far 10 --compensated
    tools/fv/comp_ladder.py --re 40 --ft 5 --levels 192:98 --far 40 --compensated
    tools/fv/comp_ladder.py --re 40 --ft 7 --levels 192:98 --far 80 --compensated
    tools/fv/comp_ladder.py --re 100 --ft 10 --trip 0.05 --st-from 200 --levels 48:26,96:50,192:98 --compensated
    dns flow --physics compressible --body cylinder --n 512 --ny 512 --dr 40 --re 40 --mach 0.2 --ft 2.5 --compensated --emit stats --out /tmp/dns/imm/b512
    dns flow --physics compressible --body cylinder --n 1024 --ny 1024 --dr 40 --re 40 --mach 0.2 --ft 1.6 --compensated --emit stats --out /tmp/dns/imm/b1024
    tools/fv/comp_bit_identity.sh BASE_BINARY build/dns

The settled rows stop on a flat cd, not on `--ft`: `tools/fv/flat_stop.py CSV TOL SPAN
TMIN -- dns flow ...` under simjob sends SIGTERM when two adjacent SPAN windows' mean
cd agree to TOL (3e-6, 20, 40 at M 0.2; 1e-5 at M 0.1).

Not measured (dropped to keep every GPU job under 5 minutes; the orders quoted
rest on the three levels above):

    tools/fv/comp_ladder.py --re 40 --ft 3 --levels 384:194 --compensated --limit 1800   # to a flat cd
    tools/fv/comp_ladder.py --re 40 --ft 1.5 --levels 768:386 --compensated --limit 1800
    tools/fv/comp_ladder.py --re 100 --ft 10 --trip 0.05 --st-from 200 --levels 384:194 --compensated --limit 1800
    dns flow --physics compressible --body cylinder --n 1024 --ny 1024 --dr 16 --re 40 --mach 0.2 --ft 5 --compensated --emit stats --out /tmp/dns/imm/n1024

## Hooks for later phases

Not built here, shaped for:

- **Scalars**: the species pattern: rows after the mean flow, own kernels, `F[0]`,
  `kFvGather<.., First, Count>`.
- **Per-face boundary data**: `FvConfig::bcRow` (built, "Diffusion flame"): any patch
  profile is rows of the patch's kind; a time-dependent row is `setBc` on that row.
- **A frame flux (MRF)**: `fvMeanFlux`'s `wS`, 0 today.
- **Implicit**: `kFvDt` is per cell.

## Not built

`wedge` / axisymmetric meshes (P8), MRF, `--hot-patch` (per-face wall T), a non-isothermal
ambient column (stratified T), every flame mode but premixed C0, diffusion C1 and the candle
(bunsen, chamber, wick), the jet's `--pulse`, LES / TFLES of the two-reactant jet,
turbulent inflow on a polyMesh (`--turb` is the box's), moving meshes, immersed bodies on a FaceMesh, a
checkpoint of `dns flow` (the flame's are built, M5: docs/chem.md "Checkpoint and restart"), `--temp`,
frames other than one plane, the CUDA run (compiled through the shim only on a
machine with nvcc). `issues.md` item 78.
