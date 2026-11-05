# Time-accurate incompressible FV on a FaceMesh

The MAC solver's time scheme (SSP-RK3, one pressure projection per stage) on
any FaceMesh, collocated. `gpu/fv_incomp` (kernels), `gpu/runner_fvincomp` (time
loop), `dns/fvincomp` (host tables, `--bc` table, structured meshes). DNS and
LES (Smagorinsky, σ). Not simpleFoam: `dns flow --polymesh CASE` alone stays
`gpu/fv_simple`, byte for byte.

```
dns tgv --solver fv [--nz 1] [--mesh-perturb A [--mesh-perturb-kind sine|random]]
dns flow --solver fv --ogrid --body cylinder --n N --ny M [--nz K] --re R --ft F
dns flow --polymesh DIR --bc FILE.toml --re R --ft T
         [--cfl C] [--les CS [--sgs smag|sigma]] [--tol T] [--max-iters N]
         [--fv-precision fp32|fp64] [--out PREFIX] [--emit stats|none]
```

This is the default physics of `--solver fv` and `--polymesh DIR --bc FILE`;
`--physics compressible` is `docs/fvcomp.md`, with its own `--bc` schema (its
`type`, `reference.direction` are errors here by name); the flame's `--physics
lowmach` is "Low-Mach" below. `--polymesh CASE` alone
stays simpleFoam. Every other flag is rejected by name (`--compensated`, `--tt`,
`--frame-box`, `--frame-n` among them). `--solver fv` has no immersed wall: a
`dns flow` body needs `--ogrid` (`fromCurvMesh`) or `--polymesh`. `DIR` is a case
or a polyMesh directory. Output: `PREFIX_stats.csv` (`t,dt,K,cd,cl,cdf,cfl,maxDiv,
pcgIters,resid,epsVisc,rmsDiv`); with `--polymesh`, `PREFIX/<t>/{U,p}` in polyMesh
order; `dns tgv` keeps its stdout series `t,K,dKdt,epsVisc`.

`tools/fv/cyl_tunnel_mesh.py` writes the MAC tunnel as a body-fitted polyMesh
(same box, same h, same off-node circle; the lattice with an O-ring in a square
hole) and its `bc.toml`.

## `--bc FILE.toml`

```toml
[reference]             # Re = u length / nu, --ft counts length / u, area for cd / cl
u = 1.0
length = 1.0
area = 0.1
[initial]
u = [1, 0, 0]
[patch.inlet]
kind = "inflow"         # u fixed
u = [1, 0, 0]
[patch.outlet]
kind = "outflow"        # p fixed (p = 0.0), u zero gradient, backflow: normal part only
[patch.lateral]
kind = "slip"           # impermeable, no shear
[patch.body]
kind = "wall"           # no-slip, impermeable: u + omega x (Cf - origin)
omega = [0, 0, 1]
origin = [0, 0, 0]
force = true            # enters cd / cl
[patch.exit]
kind = "convective"     # du/dt + uc du/dn = 0, p zero gradient, net flux closed here
uc = 1.0
[patch.farfield]
kind = "far"            # per face: inflow where u.S < 0, else outflow at p
u = [1, 0, 0]
```

Every non-empty, non-cyclic mesh patch needs an entry; an entry naming no patch,
an unknown key or kind, a key that does not belong to the kind: exit 3 by name.
`cyclic` patches are internal faces, `empty` is 2D. A body's `omega` on a 2D mesh
points along the empty axis.

## Scheme

State: cell velocity `u`, face flux `phi` (own → nei), boundary values.

**Convection: skew-symmetric central.** `C_P = (1/V) (Σ_f ±φ_f ū_f − ½ u_P Σ_f ±φ_f)
= (1/2V) Σ_f ±φ_f u_N`, `ū = ½(u_P + u_N)`. The operator is a skew matrix for ANY
antisymmetric φ, so `Σ V u·C = 0` to rounding whatever the projection did. The ½
is forced: for any compact interpolation with weights `w, 1 − w` the skew part of
`φ(w u_P + (1 − w) u_N)` is `½ φ (u_N − u_P)`; the weights drop out. Hence the
ORDER LIMIT on a non-smooth mesh (below): that ½ has truncation O(1) where
adjacent cells differ by a fixed ratio; a KE-preserving compact scheme has no
other choice.

**Viscosity.** `ν g_f·S`, `g_f` = `AlongD`: `ḡ + (Δu − ḡ·d) d/|d|²` with
least-squares cell gradients (the FV path's pinned scheme, `dns/fvgrad`, docs/mesh.md).
On an orthogonal mesh the gradient pass drops out. A boundary face that fixes u:
`g_P + (u_b − u_P − g_P·r) r/|r|²`. LES adds `ν_t,f (g_f + g_fᵀ)·S`, `ν_t,f` the
mean of the cells' (0 at walls, the structured wall image's face value).

**Wall stress.** A wall face carries OpenFOAM's whole face stress, not the
Laplacian's half: `ν (g_b + dev2(g_bᵀ))·S`, `g_b` = the cell gradient with its normal
part replaced by the one-sided `(u_b − u_P)/dn` (`wallStressRest`). Measured, not
derived: same mesh, same BCs as simpleFoam ("Same mesh", below), the Laplacian alone
left tunnel N 128 cd 1.4% high (2.0756 vs 2.0462) with its momentum budget open. The
wall term is −2.7e-2 of that +2.9e-2; simpleFoam's interior `div(ν dev2(T(grad U)))`
−1.7e-3, not taken: a gradient pass on every face of every mesh. Mean-of-cells vs
linear convective weight +1e-3, the Laplacian's `S.d/d²` vs `|S|²/S.d` −3.9e-3. A
mesh with a wall runs the cell gradient every stage, orthogonal or not; a periodic box
takes no new term.

**Stage** (`a` = 0, ¾, ⅓; `cb = 1 − a`, `ca = 1 − cb`, `c = cb dt`; the same
coefficients as `runner_incomp`):

```
u*  = ca u0 + cb b + c R(b, phi)        R: convection + viscosity, no pressure
phi*= S . (w u*_o + (1 - w) u*_n)       OpenFOAM's linear weights
A p = -div(phi*)/c + div(kg)            compact |S|²/(S.d), k = S - a d, fixed matrix
fp  = a (p_n - p_o) + kg                kg = k . linear(least-squares grad p)
phi = phi* - c fp
u   = u* - c G,   G = (1/V) Σ_f ± r_f fp_f
```

`p` is warm-started across stages; because the solve is scaled by `c`, it is the
pressure at every stage.

**Rhie-Chow equivalent.** Mahesh's form: `phi*` interpolates the PREDICTED velocity
(no pressure in it), the projection subtracts the compact gradient at faces and
its reconstruction at cells. A checkerboard p is not in the null space of `A`. At
a steady state `phi = linear(u) + c (linear(G p) − fp)`: the fixed point depends
on c, stated and measured: cavity N 24 max|U − simpleFoam| 6.1e-2, 7.8e-2, 9.2e-2,
1.0e-1 at cfl 0.6, 0.3, 0.15, 0.075 (before the wall stress), simpleFoam's `rAU`
being a different c. Where p is smooth that is O(c h² D³p); at a lid corner p is
singular and no c makes it small.

**A step remembers the last c.** `phi` of a steady state holds `c_prev D`, D =
`linear(G p) − fp`, and the next predictor forgets it. So a step of `c` moves
`p` by `(1 − c_prev/c) A⁻¹ div D` and `u` by `(c_prev − c) G A⁻¹ div D`: bounded by
c_prev however short the step, while p grows as 1/f. Cavity N 24, steady to 2e-16
under equal steps: a step of 0.5, 0.1, 1e-3, 1e-6 dt moves U 1.9e-2, 3.6e-2, 4.07e-2,
4.07e-2, gone 400 steps later (T7); 1e-12 dt puts 8e10 in the warm start and the
next steps diverge. This was issue 79c: `march` clamped its last step to the end
time. **Reach an end time in equal steps** (`dt = rem / ceil(rem / dt_stable)`,
runner and test): a step of f dt moves U by (1 − f) of that, T7 1.5e-4 at f =
0.9968. It is a precondition of `DeviceFvIncomp::step`.

**Forces.** Wall p is the cell's, the value the momentum equation applies: on a
no-flux face `fp = 0` in G IS `p_b = p_P`. `p_P + G·r` read half a normal gradient
carrying the stage's c: +1.21e-2 on tunnel N 128 cd at cfl 0.8, +2.01e-2 at cfl 0.2,
the cell-p number moving 8e-4. The viscous part is the wall stress above. Check: the
budget, inlet minus outlet momentum (`cd(cv)` of the `fixedpoint` mode), agrees with
the tally to 4e-4 at N 128, 3e-4 at N 192.

**G, the Perot reconstruction.** `(1/V) Σ_f ±r_f fp_f` is exact for linear p on any
mesh (the divergence theorem: `Σ_f r_f ⊗ S_f = V I`). On an orthogonal uniform mesh
it is the central gradient. On a wall face fp = 0: the cell's normal correction is
half the top face's, the linear profile a no-slip wall demands.

**kg uses a least-squares gradient of p, not G.** A gradient built from fp carries
kg, which carries the gradient: a recursion `G ← T G` with long-wave eigenvalues
`−s² ± s √(1 + s²)` on a parallelogram of shear s, past −1 at 30°. Measured: the
recursive form diverged in ~7 steps at 41° max non-orthogonality (at every
resolution, at 35° it held). The least-squares gradient has no memory.

**Non-orthogonal correctors.** `nCorr` solves (auto: 1 orthogonal, 2 otherwise),
each with `kg` of the previous p; `fp` always carries the last `kg`, so `phi`
closes to solver tolerance after ANY number of correctors.

**Pressure.** `A` is geometric: built, uploaded and AMG-factored once; a stage
costs one RHS. Scaled V, 2 Jacobi sweeps, flexible PCG (`gpu/la`), FP64 dots,
`constantNullSpace` when no face fixes p (RHS mean removed, FP64 sum).
Stop: `|r|₂ ≤ max(tol |div φ*/c|₂, 4 ε_R |gross flux/c|₂)`, the second term the
arithmetic's noise (below it an FP32 estimate keeps falling on nothing, CLAUDE.md).

**dt**: `cfl / max_c max(Σ|φ|/2V, Σ(ν + 2ν_t,f) corr·S / V)`: on a uniform mesh the
MAC runner's rule (`Σ|u_a|/h_a`, `6ν/h²` in 3D). Explicit viscosity: `dt ~ h²/ν`
on the thinnest wall cell. A wall-clustered mesh pays that in steps; it is why
the tunnel ladder uses an unclustered ring. `2 ν_t`: the eddy normal stress.

**Convective outlet.** The face's own RK stage, the MAC tunnel's model; the net
boundary flux is closed on its faces by a uniform `dU` (FP64 sum). All-Neumann p.

**2D.** One-cell-thick mesh, no code path: S, d, k, r and every gradient weight are
exactly 0 along the empty axis, so `w` stays exactly 0 (T4).

## Precision

One source, `R` in float | double (`--fv-precision`, default fp32). Reductions are
FP64 in both.

**The projection does not need FP64 for a time-accurate run.** FP32 stops at the
noise floor: `|r|/|b|` 1.68e-4 asked for 1e-5 or 1e-8 alike (2D 32² sine, T3), true
max|div φ| 5.1e-6 (FP64 at tol 1e-5: 1.4e-6), rms 1.4e-6. That is ε |φ|/V, the
level the MAC solver lives at. 3D TGV N 64, FP32 vs FP64: peak −dK/dt 0.0104295 vs
0.0104294; K(20) 3.05754e-2 vs 3.05750e-2; epsVisc within 0.32% of peak, tol 1e-3
and 1e-4 the same. FP64 is for energy budgets, order tests, `tol` below ~2e-4:
gates T1, T2, T6 run FP64.

## Temperature: `dns conv --solver fv --physics incompressible`

The Boussinesq model of `gpu/boussinesq` on any FaceMesh, free-fall scaling (H = dT =
g alpha = 1, nu = sqrt(Pr/Ra), kappa = 1/sqrt(Ra Pr)). `dns conv --polymesh DIR --bc
FILE --ny BINS`: `[patch.X] kind = "wall"`, `t = 0.5` the fixed T (floor +0.5, lid
-0.5, exit 3 otherwise), no `t` = adiabatic, `slip`. `t` under `dns flow`: exit 3 by
name. `--tol` defaults to 1e-6 here. See `docs/fvcomp.md` "Gravity" for the route.
`--les C [--sgs smag|sigma]`: LES of the layer, the eddy heat flux `nu_t,f / Pr_t` on the
internal T faces ("Incompressible combinations" below).

- **T** rides the stage on the same phi: `kFaceT`, `kBndT`, `kGatherT`, the scalar of
  `kFace`, `kBnd`, `kGather`, skew-symmetric central + compact diffusion with AlongD.
  `sum V T C(T)` is 0 to rounding (`gpu_fv_conv` C3: 1e-18 of 3e-2). A wall without `t`
  carries exactly 0.
- **The force is a face force** (Mencinger; OpenFOAM's `phig` form). `phi* += c B_f`,
  `B_f = (T_f - Tref) betaG.S`, and the reconstruction reads `fp - B_f`. At rest `fp =
  B_f` and G is 0 face by face. A wall stays Neumann with no net force; `B_b` in `phi*`
  is not allowed, its sum over the plates is the T difference, not 0 (pure Neumann
  compatibility). In the wall cell this is what a cell-centred force gets wrong: half a
  hydrostatic gradient per stage.
- **kg's wall image**: `kGradP`'s Neumann difference is `dpB = (T_b - Tref) betaG.r`. With
  0 the least-squares gradient lost the hydrostatic part at a wall: rest urms 1.38e-3
  vs 9.3e-5 (sine 0.08, 64 x 32, t 4).
- dt: `max(nu, kappa)` in the diffusive rate, the free-fall bound of `runner_rb`.
- IC: `rbPointAt`, `rbIC`'s streamfunction at the cell centre, with its `cos(pi y/H)`.

Stats: `runner_rb`'s columns on height bins, `nuBotF, nuTopF`, `maxDiv, pcgIters`; every
50 steps.

| pair | mesh | quantity | expected | measured |
|---|---|---|---|---|
| exact identity (C3) | uniform, random 0.3, 32 x 16 | Σ V T C(T) / gross | 0, gate 1e-13 | 3.1e-17, 3.6e-18 |
| rest | uniform 64 x 32, Ra 1000, kick 0, t 20 | urms | rounding | 1.7e-9 FP32 |
| rest, non-orthogonal | sine 0.08, max non-orth. 40.6 deg: 32, 64, 128 | urms | shrinks | 5.7e-4, 1.4e-4, 2.3e-5: order 2.1, 2.6; Nu 1.0000 face |
| rest, non-smooth | random ±0.3 h, max non-orth. 60.0 deg, 64 x 32 | urms | stated | 2.1e-5 |
| subcritical | uniform 64 x 32, Ra 1000, kick 0.02, t 40 | Nu, urms end | Nu 1 to 1% | 1.0000 / 1.0000 / 1.0000, 4.1e-6 |
| FV vs MAC, Ra 5e4 2D | 48 x 24, 96 x 48, 192 x 96, t 40 | tail nuBot | shrinks | FV 4.233, 3.960, 3.851; MAC 4.263, 3.968, 3.847; diff -0.7%, -0.19%, +0.10% |
| anchor | 96 x 48 | nuBot | 3.97 | 3.960; FP64 3.960 same 4 digits |
| heat balance | same | spread / nuBot | < 5% | 1.0%, 1.4%, 2.1% |
| symmetry | 96 x 48 | max instantaneous gap, whole run; nuBot - nuTop tail mean | broken, mean equal | 0.66; 3e-4 |
| closed sides | 96 x 48 | nuBot, nuTop, nuFlux | MAC | FV 3.645, 3.517, 3.555; MAC 3.651, 3.517, 3.555 |
| non-orthogonal Ra 5e4 | sine 0.05, max non-orth. 26.2 deg, 96 x 48 | nuBot, nuTop, nuFlux; face | uniform's | 4.021, 4.048, 3.978; 3.978, 3.969: +0.7% on the face Nu |
| critical Ra | 64 x 32, kick 0.02, t 80 | urms trend | ~1708 bracket | Ra 1600 decays (3.6e-3, 2.3e-3, 1.6e-3); Ra 1900 grows (3.4e-2, 5.3e-2, 6.6e-2) |
| projection | 96 x 48, Ra 5e4 | max \|div\| | 1e-4 | FP64 3.8e-7; **FP32 1.4e-4, 192 x 96 4.6e-4**: the floor `4 eps gross`, the flux carries c B_f. FP32 Nu equals FP64 to 4 digits. The gate runs FP64 |

## Incompressible combinations

`dns/fvimodules` (mci; the shape of the compressible `dns/fvmodules` and the low-Mach
`dns/lmmodules`): the modules of `DeviceFvIncomp`'s stage and ONE decision per pair, read by
`validate()` (`fvIncompModules`: `dns flow` / `dns conv` on this route; the refusal names its
flag, exit 2), `DeviceFvIncomp::make` (`fviModulesOf` of the controls: a refused set is an
error by its text) and `gpu_fvi_matrix`. A module that is not built on this family is refused
ALONE (the module's own row), not per pair.

- **LES** (Smagorinsky, sigma): momentum `nu_t,f (g_f + g_fᵀ)·S` (Scheme above), built before
  mci. **LES + T** (mci): the eddy heat flux `(kappa + nu_t,f / Pr_t)` on the internal T faces,
  `nu_t,f` the cells' mean (the momentum's), `Pr_t = kLesPrt` (0.9, the one constant of the
  compressible and low-Mach LES), boundary faces molecular (the wall's `nu_t,f` is 0):
  `kFaceT<…, Les>`. `nu_t = 0` gives `kappa + 0 = kappa`: the molecular T bit for bit. dt: the
  rule's `2 nu_t,f` already bounds `nu_t,f / Pr_t`.
- **T** (the scalar row) and **gravity** (the Boussinesq face force, requires T). T without
  gravity (`betaG = 0`) is a PASSIVE TRACER: the force passes (`kBuoyF`, `kBuoyB`, the
  `fp - B_f` reconstruction, `kGradP`'s wall image) do not run, so momentum is the run without T
  bit for bit (gate (a)). No CLI spelling: `dns conv` always carries gravity.
- **Refused alone**: RANS (SA, SST; simpleFoam owns RAS on a FaceMesh, issues.md FI2) and species
  (C0, C1, C2b; decided on evidence, issues.md FI1: no oracle, no consumer, the passive tracer
  exists as T).

`dns conv --solver fv --physics incompressible --les C [--sgs smag|sigma]` is the LES route of
the Boussinesq layer (sigma needs nz > 1; `--les` on the structured conv and on the compressible
FV conv: exit 2 by name). Label `LES Smagorinsky Cs C, eddy heat flux Pr_t 0.9 [FV path, …]`.
The wall Nu columns (`nuBot`, `nuTop`, `nuBotF`, `nuTopF`) are exact under LES (walls carry no
`nu_t`); `nuFlux` is resolved + molecular, without the SGS heat flux (issues.md FI3).

**`gpu_fvi_matrix`** (`test_gpu_fvi_matrix`, 15 s): every live set (each module alone and each
pair, closed under the requirements: 9 sets) on four 3D meshes, each in Source / Morton /
ColourMajor numbering: a uniform 8 x 6 x 4 walled box (conv's: periodic x, z, plates at y 0
and 1 with T 0.5 / -0.5), the same box rotated rigidly by 0.7 rad about (1, 2, 3) (polyOf, points
and cyclic separations rotated, gravity rotated with it), a random ±0.25 h box, a cylinder O-grid
16 x 8 x 4 (open: hot wall T 0.5 + far u (1, 0, 0); closed: adiabatic wall + slip circle).
(a) inert: passive T leaves momentum bitwise (FP32 and FP64, every mesh); sigma on a 2D mesh
(D_sigma = 0) is the molecular run bitwise, momentum and T, on the non-orthogonal 2D meshes
(random ±0.25 h, sine 0.08; the 2D O-grid is orthogonal there: its molecular run skips the
gradient, another instantiation, n/a). (b) on the closed set rest stays rest: exactly 0 without
gravity; with gravity the hydrostatic state (boxes: the conduction profile; O-grid: uniform T
0.25 != Tref, a linear p) to 1e-12 on uniform, rotated, O-grid (measured 7e-17, 1.4e-16, 1.7e-14),
random 0.25 h a STATED residual 1.6e-3 of the free-fall velocity after 10 steps (gate 5e-3; the
non-orthogonal balance of "Temperature" above); the divergence floor of a vortical state after
10 steps, max|div φ|/u_max ≤ 1.0e-10 FP64 (gate 1e-9), ≤ 1.2e-5 FP32 (gate 1e-4); `Σ V u·C(u)`
and `Σ V T C(T)` ≤ 2e-16 of their gross (gate 1e-13). (c) FP32 vs FP64 |ΔU|/U ≤ 3.4e-6 over 10
steps (gate 1e-4). (d) numbering: FP64 tol 1e-12, 3 steps, by srcCell: ≤ 1.4e-13 (gate 1e-10),
summation order (CSR and dot order move with the numbering), not bitwise. (e) rerun byte-
identical, FP32 and FP64. Make refuses gravity without T by the inventory's text.

**`gpu_fvi_matrix_phys`** (the live pairs' physics; same grid, LES vs DNS: gate 5 has no
structured counterpart, the structured conv solvers carry no SGS model, issues.md FI3):

| case | grid | DNS nuBot | sigma 1.35 | Smagorinsky 0.17 |
|---|---|---|---|---|
| Ra 1e5 3D, aspect 2, t 60 (window t 24-60) | 32 x 16 x 32 | 5.3968 | 5.5803 (+3.4%) | 5.8679 (+8.7%) |
| | 48 x 24 x 48 | 4.9395 | 5.0942 (+3.1%) | 5.2141 (+5.6%) |
| | 64 x 32 x 64 | 4.8450 | not measured | not measured |
| Ra 5e4 2D (qualitative), t 40 | 48 x 24 | 4.2328 | (2D: 0) | 4.2997 (+1.6%) |

Both LES RAISE Nu: no wall damping, so the first interior faces carry `nu_t,f / Pr_t` inside the
thermal layer, where the DNS conducts. Sigma moves it less (D_sigma vanishes in pure shear, the
near-wall state; Smagorinsky's |S| does not), and the shift shrinks with h (Delta ~ h: the
filtered solution tends to the DNS). Walls balance to < 3% in every LES run. The DNS itself is not
converged at these grids (5.40, 4.94, 4.85): LES is not a substitute for resolution here, it is
the filtered model of the same grid. Gates: walls 5%, DNS < sigma < Smagorinsky at both grids,
LES within 12% of the DNS, the shift shrinking 32 → 48; 2D Smagorinsky within 5%.

Generated (`test_gpu_fvi_matrix markdown`):

<!-- generated: dns::fviCombinationsMarkdown (src/dns/fvimodules.cpp); test_gpu_fvi_matrix markdown -->
| Module | Axis | Spelling | Built |
|---|---|---|---|
| LES smag | turbulence | `--les Cs [--sgs smag]` | yes |
| LES sigma | turbulence | `--les C --sgs sigma (3D)` | yes |
| RANS SA | turbulence | `--rans sa-neg` | refused |
| RANS SST | turbulence | `--rans sst-2003m, sst-vm` | refused |
| T | scalar | `dns conv --physics incompressible --solver fv; DeviceFvIncomp::make(T0)` | yes |
| gravity | force | `dns conv (betaG = g alpha, free-fall scaling)` | yes |
| C0 | chemistry | `(premixed one-step)` | refused |
| C1 | chemistry | `(two reactants)` | refused |
| C2b | chemistry | `(--equivalence, GRI-3.0 A(phi))` | refused |

Pair matrix: L live, R refused, x one axis (exclusive).

|   | LES smag | LES sigma | RANS SA | RANS SST | T | gravity | C0 | C1 | C2b |
|---|---|---|---|---|---|---|---|---|---|
| LES smag | . | x | R | R | L | L | R | R | R |
| LES sigma | x | . | R | R | L | L | R | R | R |
| RANS SA | R | R | . | R | R | R | R | R | R |
| RANS SST | R | R | R | . | R | R | R | R | R |
| T | L | L | R | R | . | L | R | R | R |
| gravity | L | L | R | R | L | . | R | R | R |
| C0 | R | R | R | R | R | R | . | R | R |
| C1 | R | R | R | R | R | R | R | . | R |
| C2b | R | R | R | R | R | R | R | R | . |

| Refused | Error text | Why |
|---|---|---|
| RANS SA | `--rans on the time-accurate incompressible FV solver: not built (steady RANS on a FaceMesh is simpleFoam: dns flow --polymesh CASE or --solver simple)` | the family's RAS is gpu/fv_simple + gpu/fv_ras, OpenFOAM v2606 kOmegaSST / SpalartAllmaras verbatim and gated to 1e-9 (gpu_fv_simple IF-T); URANS on the projection stage would be a second implementation of the closures with its own wall treatment and no consumer |
| RANS SST | `--rans on the time-accurate incompressible FV solver: not built (steady RANS on a FaceMesh is simpleFoam: dns flow --polymesh CASE or --solver simple)` | the family's RAS is gpu/fv_simple + gpu/fv_ras, OpenFOAM v2606 kOmegaSST / SpalartAllmaras verbatim and gated to 1e-9 (gpu_fv_simple IF-T); URANS on the projection stage would be a second implementation of the closures with its own wall treatment and no consumer |
| C0 | `species on the incompressible FV solver: not built (constant-density reacting scalars; the flame on a FaceMesh is dns flame --physics lowmach --solver fv)` | decided on evidence (issues.md FI1): no oracle (neither structured incompressible solver, gpu/incomp nor gpu/boussinesq, carries a scalar beyond T, so gate 5 has no counterpart); no consumer (every flame model, C0, C1, C2b, is calibrated by its heat release: S_L, T_ad; constant density removes it, DeviceFvLowMach is the variable-density flame on this mesh view); a passive tracer is the T row without gravity (gpu_fvi_matrix: momentum bitwise, conv T energy 0), so a scalar row would add no kernel |
| C1 | `species on the incompressible FV solver: not built (constant-density reacting scalars; the flame on a FaceMesh is dns flame --physics lowmach --solver fv)` | decided on evidence (issues.md FI1): no oracle (neither structured incompressible solver, gpu/incomp nor gpu/boussinesq, carries a scalar beyond T, so gate 5 has no counterpart); no consumer (every flame model, C0, C1, C2b, is calibrated by its heat release: S_L, T_ad; constant density removes it, DeviceFvLowMach is the variable-density flame on this mesh view); a passive tracer is the T row without gravity (gpu_fvi_matrix: momentum bitwise, conv T energy 0), so a scalar row would add no kernel |
| C2b | `species on the incompressible FV solver: not built (constant-density reacting scalars; the flame on a FaceMesh is dns flame --physics lowmach --solver fv)` | decided on evidence (issues.md FI1): no oracle (neither structured incompressible solver, gpu/incomp nor gpu/boussinesq, carries a scalar beyond T, so gate 5 has no counterpart); no consumer (every flame model, C0, C1, C2b, is calibrated by its heat release: S_L, T_ad; constant density removes it, DeviceFvLowMach is the variable-density flame on this mesh view); a passive tracer is the T row without gravity (gpu_fvi_matrix: momentum bitwise, conv T energy 0), so a scalar row would add no kernel |
| gravity without T (requirement) | `the Boussinesq force needs the temperature row (dns conv)` | |

| Live pair | Hand gates |
|---|---|
| LES smag + T | gpu_fvi_matrix_phys P2 (Ra 1e5 3D: Nu +8.7%, +5.6% of the same-grid DNS at 32, 48 cells, shrinking with h) |
| LES smag + gravity | gpu_fvi_matrix_phys P1 (2D Ra 5e4: Nu +1.6%), P2 |
| LES sigma + T | gpu_fvi_matrix (a: sigma on a 2D mesh = molecular bitwise, momentum and T); gpu_fvi_matrix_phys P2 (Nu +3.4%, +3.1%) |
| LES sigma + gravity | gpu_fvi_matrix_phys P2 (DNS < sigma < Smagorinsky Nu at both grids) |
| T + gravity | gpu_fv_conv C3-C8, long (FV vs MAC Ra 5e4) |

Matrix rows (`test_gpu_fvi_matrix`, the run of this commit; (b) columns: rest max |u|, div = max|div φ|/u_max
FP64 (FP32), KE and T = the convective rates over their gross; (c) max |U32 − U64|/max|U|; (d) max
difference over the numberings / max|U|):

| Set | Mesh | (a) inert | (b) rest, div, KE | (c) fp32 | (d) numbering | (e) rerun |
|---|---|---|---|---|---|---|
| laminar | uniform box 8x6x4 | n/a (off = absence) | rest max u 0.0e+00, div/u 2.0e-11 (fp32 6.2e-06), KE 9.9e-17 | 6.7e-07 | 4.3e-16 | = |
| laminar | rotated box (0.7 rad about (1,2,3)) | n/a (off = absence) | rest max u 0.0e+00, div/u 2.2e-11 (fp32 3.0e-06), KE 5.6e-17 | 2.6e-07 | 3.6e-14 | = |
| laminar | random 0.25 h box | n/a (off = absence) | rest max u 0.0e+00, div/u 2.5e-11 (fp32 5.3e-06), KE 2.0e-17 | 7.6e-07 | 1.3e-13 | = |
| laminar | O-grid 16x8x4 | n/a (off = absence) | rest max u 0.0e+00, div/u 3.9e-11 (fp32 1.2e-05), KE 5.7e-17 | 1.4e-06 | 2.2e-14 | = |
| LES smag | uniform box 8x6x4 | n/a (off = absence) | rest max u 0.0e+00, div/u 2.2e-11 (fp32 6.0e-06), KE 3.2e-17 | 6.1e-07 | 4.5e-16 | = |
| LES smag | rotated box (0.7 rad about (1,2,3)) | n/a (off = absence) | rest max u 0.0e+00, div/u 1.2e-12 (fp32 3.4e-06), KE 1.6e-17 | 4.2e-07 | 3.8e-14 | = |
| LES smag | random 0.25 h box | n/a (off = absence) | rest max u 0.0e+00, div/u 1.8e-11 (fp32 4.7e-06), KE 1.8e-17 | 6.3e-07 | 1.4e-13 | = |
| LES smag | O-grid 16x8x4 | n/a (off = absence) | rest max u 0.0e+00, div/u 4.0e-11 (fp32 1.1e-05), KE 1.0e-17 | 1.1e-06 | 2.1e-14 | = |
| LES sigma | uniform box 8x6x4 | ; sigma 2D random 0.25 h box 2D: bitwise; sigma 2D sine 0.08 box 2D: bitwise; O-grid 2D: n/a (orthogonal) | rest max u 0.0e+00, div/u 2.2e-11 (fp32 6.5e-06), KE 6.8e-18 | 6.8e-07 | 4.3e-16 | = |
| LES sigma | rotated box (0.7 rad about (1,2,3)) | n/a (off = absence) | rest max u 0.0e+00, div/u 2.0e-11 (fp32 2.7e-06), KE 2.4e-17 | 3.7e-07 | 3.0e-14 | = |
| LES sigma | random 0.25 h box | n/a (off = absence) | rest max u 0.0e+00, div/u 2.6e-11 (fp32 5.0e-06), KE 1.8e-17 | 7.0e-07 | 1.3e-13 | = |
| LES sigma | O-grid 16x8x4 | n/a (off = absence) | rest max u 0.0e+00, div/u 3.9e-11 (fp32 1.2e-05), KE 0.0e+00 | 2.2e-06 | 2.2e-14 | = |
| T | uniform box 8x6x4 | T passive: momentum bitwise | rest max u 0.0e+00, div/u 2.0e-11 (fp32 6.2e-06), KE 9.9e-17, T 9.9e-18 | 6.7e-07 | 2.5e-16 | = |
| T | rotated box (0.7 rad about (1,2,3)) | T passive: momentum bitwise | rest max u 0.0e+00, div/u 2.2e-11 (fp32 3.0e-06), KE 5.6e-17, T 7.4e-18 | 2.6e-07 | 1.4e-14 | = |
| T | random 0.25 h box | T passive: momentum bitwise | rest max u 0.0e+00, div/u 2.5e-11 (fp32 5.3e-06), KE 2.0e-17, T 4.4e-18 | 7.6e-07 | 6.7e-14 | = |
| T | O-grid 16x8x4 | T passive: momentum bitwise | rest max u 0.0e+00, div/u 3.9e-11 (fp32 1.2e-05), KE 5.7e-17, T 5.9e-17 | 1.4e-06 | 2.2e-14 | = |
| LES smag + T | uniform box 8x6x4 | T passive: momentum bitwise | rest max u 0.0e+00, div/u 2.2e-11 (fp32 6.0e-06), KE 3.2e-17, T 1.1e-17 | 6.1e-07 | 1.9e-16 | = |
| LES smag + T | rotated box (0.7 rad about (1,2,3)) | T passive: momentum bitwise | rest max u 0.0e+00, div/u 1.2e-12 (fp32 3.4e-06), KE 1.6e-17, T 2.8e-18 | 4.2e-07 | 1.4e-14 | = |
| LES smag + T | random 0.25 h box | T passive: momentum bitwise | rest max u 0.0e+00, div/u 1.8e-11 (fp32 4.7e-06), KE 1.8e-17, T 3.9e-17 | 6.3e-07 | 7.0e-14 | = |
| LES smag + T | O-grid 16x8x4 | T passive: momentum bitwise | rest max u 0.0e+00, div/u 4.0e-11 (fp32 1.1e-05), KE 1.0e-17, T 6.3e-17 | 1.1e-06 | 2.1e-14 | = |
| LES sigma + T | uniform box 8x6x4 | T passive: momentum bitwise; sigma 2D random 0.25 h box 2D: bitwise; sigma 2D sine 0.08 box 2D: bitwise; O-grid 2D: n/a (orthogonal) | rest max u 0.0e+00, div/u 2.2e-11 (fp32 6.5e-06), KE 6.8e-18, T 2.0e-17 | 6.8e-07 | 2.5e-16 | = |
| LES sigma + T | rotated box (0.7 rad about (1,2,3)) | T passive: momentum bitwise | rest max u 0.0e+00, div/u 2.0e-11 (fp32 2.7e-06), KE 2.4e-17, T 2.5e-17 | 3.7e-07 | 1.1e-14 | = |
| LES sigma + T | random 0.25 h box | T passive: momentum bitwise | rest max u 0.0e+00, div/u 2.6e-11 (fp32 5.0e-06), KE 1.8e-17, T 4.5e-18 | 7.0e-07 | 6.6e-14 | = |
| LES sigma + T | O-grid 16x8x4 | T passive: momentum bitwise | rest max u 0.0e+00, div/u 3.9e-11 (fp32 1.2e-05), KE 0.0e+00, T 1.5e-17 | 2.2e-06 | 2.2e-14 | = |
| T + gravity | uniform box 8x6x4 | T passive: momentum bitwise | rest max u 7.1e-17, div/u 1.4e-11 (fp32 1.2e-06), KE 1.1e-17, T 2.4e-17 | 2.5e-07 | 2.5e-16 | = |
| T + gravity | rotated box (0.7 rad about (1,2,3)) | T passive: momentum bitwise | rest max u 1.4e-16, div/u 9.9e-11 (fp32 1.1e-05), KE 5.5e-18, T 4.8e-17 | 8.9e-07 | 3.0e-14 | = |
| T + gravity | random 0.25 h box | T passive: momentum bitwise | rest max u 1.6e-03 (stated), div/u 7.6e-11 (fp32 9.7e-06), KE 4.3e-17, T 4.3e-17 | 1.2e-06 | 5.4e-14 | = |
| T + gravity | O-grid 16x8x4 | T passive: momentum bitwise | rest max u 1.7e-14, div/u 4.1e-11 (fp32 1.2e-05), KE 1.9e-17, T 1.1e-16 | 1.3e-06 | 2.1e-14 | = |
| LES smag + T + gravity | uniform box 8x6x4 | T passive: momentum bitwise | rest max u 7.1e-17, div/u 1.2e-11 (fp32 1.4e-06), KE 4.0e-17, T 5.3e-18 | 3.7e-07 | 2.5e-16 | = |
| LES smag + T + gravity | rotated box (0.7 rad about (1,2,3)) | T passive: momentum bitwise | rest max u 1.4e-16, div/u 6.1e-12 (fp32 1.1e-05), KE 2.7e-17, T 1.6e-17 | 1.1e-06 | 2.8e-14 | = |
| LES smag + T + gravity | random 0.25 h box | T passive: momentum bitwise | rest max u 1.6e-03 (stated), div/u 9.6e-11 (fp32 6.9e-06), KE 2.8e-17, T 3.8e-17 | 1.0e-06 | 5.6e-14 | = |
| LES smag + T + gravity | O-grid 16x8x4 | T passive: momentum bitwise | rest max u 1.7e-14, div/u 3.4e-11 (fp32 1.0e-05), KE 2.1e-17, T 2.0e-16 | 1.1e-06 | 2.2e-14 | = |
| LES sigma + T + gravity | uniform box 8x6x4 | T passive: momentum bitwise; sigma 2D random 0.25 h box 2D: bitwise; sigma 2D sine 0.08 box 2D: bitwise; O-grid 2D: n/a (orthogonal) | rest max u 7.1e-17, div/u 1.3e-11 (fp32 1.6e-06), KE 4.8e-17, T 0.0e+00 | 3.0e-07 | 2.5e-16 | = |
| LES sigma + T + gravity | rotated box (0.7 rad about (1,2,3)) | T passive: momentum bitwise | rest max u 1.4e-16, div/u 1.0e-10 (fp32 1.2e-05), KE 5.7e-18, T 7.7e-17 | 9.5e-07 | 2.8e-14 | = |
| LES sigma + T + gravity | random 0.25 h box | T passive: momentum bitwise | rest max u 1.6e-03 (stated), div/u 7.7e-11 (fp32 8.4e-06), KE 8.9e-18, T 3.5e-17 | 1.2e-06 | 5.4e-14 | = |
| LES sigma + T + gravity | O-grid 16x8x4 | T passive: momentum bitwise | rest max u 1.7e-14, div/u 4.1e-11 (fp32 1.1e-05), KE 3.9e-17, T 1.6e-16 | 3.4e-06 | 2.0e-14 | = |

FV LES vs the structured MAC LES (gpu/les) is two models (gradient form `div(nu_t grad u)` with
the fluid-fraction damping vs the symmetric stress) with no shared route: not measured
(issues.md FI3).

## Verification

OpenFOAM is not installed; existing solvers and exact identities are the oracle.
Collocated vs staggered differ by design: a FV − MAC number is expected to SHRINK,
never to vanish.

**Every order below rests on 3 grid levels (user decision: at most 3 levels per
ladder, no job over 5 minutes). None is a converged order.** Rows measured before
the trim are marked; they do not enter an order.

| pair | mesh | quantity | expected | measured |
|---|---|---|---|---|
| FV vs analytic | 2D uniform 32, 64 | L2 u error, TGV ν 0.05, t 1, cfl 0.1 | order 2 | 2.31e-4, 4.71e-5: **2.29**; gate T1 > 1.8. Before the end-time rule 2.10e-4, 4.59e-5, 2.19: the clamped last step was re-projecting |
| FV vs analytic | 2D sine 0.04: 32, 64 | same | order 2 | **2.00**; gate T1 > 1.8. Sine 0.06: 2.07e-2, 5.18e-3, 1.30e-3, 3.24e-4, order 2.00 |
| FV vs analytic | 2D random ±0.1 h, ±0.25 h, 32, 64, 128 | same | order 2 asked; 1 by the ½ argument | **1.19, 1.00** (±0.1: 6.53e-3, 2.87e-3, 1.43e-3); ±0.25: 1.91e-2, 9.63e-3, 6.39e-3 (0.99, 0.59). NOT order 2. Viscous-dominated (ν 5, t 0.02, ±0.25): 1.56e-2, 9.97e-3, 7.71e-3, order 0.65, 0.37: AlongD on a non-smooth mesh also limits |
| exact identity | uniform, random 0.25 (2D), sine 0.05, random 0.15 (3D), `box_refined` | Σ V u·C / Σ\|V u·C\| | 0 to rounding | 9.2e-17, 6.5e-17, 1.6e-17, 3.5e-17, 3.7e-17; gate 1e-13 |
| KE, inviscid total | same | K drift over 8 steps | pressure work, not 0: collocated | −3.6e-3, **+2.3e-3**, −1.7e-3, −3.5e-3, −1.3e-2. Over t 30 (cfl 0.4): 2D uniform 32 −3.6%, sine −2.9%, random ±0.25 N 32 **−48%**, N 64 −35%, 3D random ±0.15 N 24 −68%: the projection is DISSIPATIVE on a random mesh. Never blew up |
| projection | sine 0.04, 2D 32, FP64 | \|r\|/\|b\| vs tol 1e-3, 1e-5, 1e-8 | ≤ tol | 1.00e-3, 9.1e-6, 9.4e-9 |
| rerun | `cyl2d_snappy` | U, p, φ bytes | identical | identical |
| end to end | `--solver fv --ogrid` Re 40, 96 x 48, far 20 D, beta 3, `--ft 4` | exit 0, finite | runs | cd 1.5438, cdf 0.5356, cl 2e-4 at t 125.7, 89505 steps, 1.09 ms/step; no ladder |
| 2D | `cyl2d_snappy`, every BC kind but outflow-only (inflow, convective, slip, far, moving wall) | w | exactly 0 | exactly 0 |
| LES | uniform 16³ | K(3): DNS, Smagorinsky 0.17, σ 1.35 | Smag < σ < DNS | 26.51 < 26.79 < 29.59 |
| LES | 2D random | σ vs molecular | bitwise | bitwise |
| FV vs MAC | uniform 3D TGV Re 1600, N 96, 128, 192 (MAC `dns tgv`, cfl 0.4) | rms(epsVisc), rms(−dK/dt) of FV − MAC over t | shrinks, order ~2 | epsVisc 7.46e-4, 5.84e-4, 1.90e-4: **order 0.85, 2.77, 1.97 over 96→192**. −dK/dt 2.55e-4, 4.85e-4, 1.87e-4: NON-MONOTONE, order 0.44 over 96→192. Peak −dK/dt diff −5.0e-4, +1.3e-4, +2.3e-4 (no order: sign changes; "TGV convergence is oscillatory", and MAC's own peak moves 0.0137, 0.0136, 0.0120). Rests on 3 levels. Before the trim: N 32 1.93e-3, N 64 1.39e-3 (rms eps) |
| FV alone | same | peak −dK/dt vs peak epsVisc, N 192 | FV gap = projection dissipation | 0.012215 vs 0.011871: 2.8% numerical; MAC 0.011983 vs 0.011990 |
| FV vs MAC | tunnel L/D 8, Re 40, 128, 192, 256 (`cyl_tunnel_mesh.py` vs `--far tunnel`) | steady cd | difference shrinks | FV 2.04109, 2.04452, 2.0432 (N 192 falling 2e-4 per 3.5 t; N 256 at t 8 falling 1e-3 per 2 t, not flat, ~2.042); MAC 2.08833, 2.08556, 2.08091; diff −4.72e-2, −4.10e-2, −3.77e-2: **shrinks, order 0.35, 0.29, and it is MAC moving**: FV spans 3.4e-3, MAC 7.4e-3 and still going (next row). N 128 from rest: flat 2.04108 to 1e-6. Before the wall stress and the tally, FV 2.07612, 2.06360, 2.05669 and the "plateau" −1.22e-2, −2.20e-2, −2.42e-2 |
| MAC alone | same, N 128, 256, 512 | steady cd | converges | 2.08833, 2.08091, **2.0654** (t 16.3, cfl 0.7, flat to 6e-5 per 1.4 t): −7.4e-3, then −1.56e-2. NOT asymptotic on 3 levels: first-order cut wall, the off-node offset moves with N. Limit ≤ ~2.05; N 1024 not measured |
| outlet | FV N 128, 192, 256: convective vs `p = 0` outflow | cd | a model difference | −1.05e-2, −1.10e-2, ~−1.1e-2: flat in h. The MAC tunnel is the convective one |
| FV vs MAC | tunnel, Re 100, 128, 192, 256 | time-mean cd, t ∈ [60, 100] | shrinks | FV 1.77551, 1.75882, 1.73923; MAC 1.75565, 1.75063, 1.73562; diff +1.99e-2, +8.2e-3, +3.6e-3: **order 2.18, 2.84**; 3 levels and WINDOW-SENSITIVE: [50, 70] gives +1.98e-2, +7.2e-3, +4e-6 |
| FV vs simpleFoam | `cavity` Re 100, stored fixed point, central, 24² (gate T6) | outside 2 cells of the lid corners, max, rms \|ΔU\|/U | < 5e-2, unchanged bound | **2.03e-2, 2.55e-3, passes.** All cells 1.01e-1 (the corner cell), rms 8.3e-3: not gated, see "Lid corner" |
| FV vs simpleFoam | `channel`, stored fixed point, linearUpwind there | max, rms | scheme difference | 3.58e-2, 3.12e-3; passes |
| steady | `cavity` from rest, equal steps | U change | flat | 2e-16 per 200 steps (T7). Issue 79c closed: the 3.2e-2 was the short last step of a `march`, reproduced at will |
| byte | `dns.f892cc6` vs this tree: incompressible flow, LES, compressible, tgv, 4 `--polymesh` cases, 3 dry-runs | 56 files (timing fields masked) | identical | identical |
| byte | `dns.merged` vs the wall stress + tally + end-time rule: `dns flow --body cylinder --physics incompressible --far tunnel` N 128, `dns flow --polymesh cavity` FP64 100 it | stats (ms masked), .bin, json (masked), U, p, phi | identical | identical. The changes sit in `fv_incomp`, `runner_fvincomp` only |

### Same mesh, same BCs: FV vs simpleFoam

`--solver simple` on the case, then the march started ON its fixed point: one mesh,
one BC table, the time scheme the only intended difference. That isolates the
projection and the face flux from the mesh and from MAC. O-grid: nonOrth 0, far 20 D,
beta 2, far split at x = 0 into fixed inflow and fixed p, on both sides (`--far`'s
freestreamVelocity is a smooth blend, not this). Tunnel: `cyl_tunnel_mesh.py`,
nonOrth 43°, `p = 0` outflow on both. Both central. simpleFoam on these meshes moves
5e-4 under U relaxation 0.97, 0.9, 0.7, 0.5 (2.0463, 2.0462, 2.0458), so its own c
is not the story.

| mesh | | O 64, T 128 | O 128, T 192 | N 256 |
|---|---|---|---|---|
| O-grid | simpleFoam cd | 1.53862 | 1.54598 | 1.54653 |
| | FV cd | 1.52974 | 1.54254 | 1.54467 |
| | FV − simple | −8.88e-3 | −3.44e-3 | −1.86e-3 |
| | rms, max \|ΔU\|/U | 1.10e-3, 1.07e-2 | 3.25e-4, 2.79e-3 | 1.32e-4, 1.07e-3 |
| | same, Laplacian wall, cell-p tally: FV − simple | +6.96e-3 | −1.76e-3 | −1.81e-3 |
| | rms, max | 2.49e-3, 3.05e-2 | 5.22e-4, 8.03e-3 | 1.51e-4, 2.10e-3 |
| tunnel | simpleFoam cd (N 512: 2.05302) | 2.04620 | 2.05559 | 2.05544 |
| | FV cd | 2.05154 | 2.05549 | 2.05351 |
| | FV − simple | +5.3e-3 | −1.0e-4 | −1.9e-3 |
| | rms, max | 1.43e-3, 1.96e-2 | 1.29e-3, 1.70e-2 | 1.45e-3, 1.58e-2 |
| | tally − budget | +3.9e-4 | +3.0e-4 | +1.0e-3 |
| | Laplacian wall: FV − simple | +2.94e-2 | not measured | not measured |

O-grid: the fields converge, rms order 1.76, 1.30, max 1.94, 1.38 (N 256 at t 12,
still moving 2.5e-4, so its last rung is a ceiling); cd difference order 1.37, 0.89.
**The projection and face-flux algebra hold: the difference is a scheme difference
that shrinks.** Tunnel: cd agrees to 1.9e-3 past N 128 with no trend; the
field difference does NOT shrink (rms 1.4e-3, max 1.6e-2, at the ring's corner cells,
nonOrth 43°, volume ratio 1.67 at every N, an O(h) mesh: Limits). simpleFoam alone
on the tunnel: 2.04620, 2.05559, 2.05544, 2.05302, a ±1e-3 band past N 192. T256 runs
FP64 tol 1e-5, t 8, cd rising 1.2e-4 per 2 t. FP32 there reads 1.5e-3 low (2.05199):
cd rungs are FP64.

**So the Re 40 gap, in order:** (1) the tally, +1.2e-2 to +2.0e-2, dt-dependent,
fixed; (2) the missing wall stress, +2.7e-2 at N 128 and O(h), fixed: FV is flat to
3.4e-3 over the ladder; (3) the MAC side: not converged at N 256, falling; (4) the
outlet, −1.05e-2, flat, a model and the same in both when both are convective. Domain
equal by construction (same box, same h, same off-node circle). FV's wall and MAC's
cut wall stay different walls; a limit gap is not excluded: FV ~2.042, simple −
outlet 2.053 − 0.011 = 2.042, MAC 2.0654 at N 512 with −1.56e-2 on the last doubling.

### Lid corner

The lid meets a resting wall: velocity jumps, p ~ 1/r. A cell there is a scheme
number, not a solution. FV vs simpleFoam at their own fixed points, FP64, cfl 0.6,
`dns flow --polymesh` + `test_gpu_fv_incomp cavity`, excluded region FIXED in
physical size:

| excluded | N 24 | N 48 | N 96 | order |
|---|---|---|---|---|
| nothing: max | 1.01e-1 | 9.41e-2 | 9.10e-2 | 0.10, 0.05 |
| nothing: rms | 8.25e-3 | 3.82e-3 | 1.85e-3 | 1.11, 1.05 |
| h_24: max | 4.84e-2 | 1.86e-2 | 5.09e-3 | 1.38, 1.87 |
| h_24: rms | 4.82e-3 | 1.16e-3 | 1.28e-4 | 2.05, 3.18 |
| 2 h_24: max | 2.03e-2 | 6.17e-3 | 4.99e-4 | 1.72, 3.63 |
| 2 h_24: rms | 2.55e-3 | 3.11e-4 | 4.09e-5 | 3.03, 2.93 |

The all-cell max does not shrink, one cell closer to the corner each level; anything
a fixed distance off it does. That is the singularity, not a bug. Before the wall
stress: all cells 6.07e-2, 5.35e-2, 5.12e-2; 2 h_24 3.58e-2, 1.06e-2, 2.91e-3 (the
wall stress trades the corner cell for everything else). The 8.6e-2 once in this
table was that plus the short last step. So T6 gates the cells outside 2 cells of a
lid corner on the SAME 5e-2 bound, and prints the rest. The corner cell also moves
with c: Laplacian wall 6.1e-2, 7.8e-2, 9.2e-2, 1.0e-1 at cfl 0.6, 0.3, 0.15, 0.075;
this wall 1.0e-1, 1.3e-1, 1.4e-1, 1.5e-1 at cfl 0.6, 1.0, 1.2, 1.4, stable.

### Random meshes after these fixes

Unchanged: neither fix touches a periodic box's interior (`hasWall` false; the
end-time rule moves the last digits). ±0.1 h, ν 0.05: 6.28e-3, 2.73e-3, 1.41e-3,
order 1.20, 0.95 (was 6.53e-3, 2.87e-3, 1.43e-3, 1.19, 1.00). Inviscid 32² ±0.25 h
to t 29: K −47.9% (was −48%). No scheme redesign was started; Limits stand.

Not measured (the trim):

```
# TGV N 256 (~20 min FV):
dns tgv --n 256 --ft 20 --cfl 0.4 --out /tmp/dns/incomp/m256
dns tgv --solver fv --n 256 --ft 20 --cfl 0.4 --fv-precision fp32 --out /tmp/dns/incomp/f256
# MAC N 1024: ~40 min at cfl 0.7 (the limit of the "MAC alone" row):
dns flow --body cylinder --physics incompressible --far tunnel --n 1024 --nz 1 --re 40 --dr 8 --ft 2.6 --fdt 1e9 --cfl 0.7
# simpleFoam tunnel N 1024: 3000 iterations > 5 min, cut at 2081, not converged:
dns flow --polymesh T1024 --fv-precision fp64 --max-iters 3000
# FV tunnel N 256 to a flat cd (tol 1e-7: ~100 ms/step, t 4.5 in 295 s), N 512 likewise:
test_gpu_fv_incomp fixedpoint T256 T256/bc_conv.toml S256/3000/U 0.019634954084936207 30 0.8
# Re 100 tunnel row: FV numbers are before the wall stress and the tally, not re-run.
# The interior dev2(T(grad U)) term: 1.7e-3 of cd at N 128, not built.
# Re 40, Re 100 N 512:
tools/fv/cyl_tunnel_mesh.py --n 512 --out T512
dns flow --polymesh T512 --bc T512/bc.toml --re 40 --ft 30 --out F512
dns flow --body cylinder --physics incompressible --far tunnel --n 512 --nz 1 --re 40 --dr 8 --ft 6 --fdt 1e9 --out M512
```

Tried, measured, not taken, on tunnel N 128 vs simpleFoam 2.0462 (Laplacian wall
2.0756): the linear weight in the convective mean 2.0766; `|S|²/S.d` for the viscous
coefficient 2.0717; both 2.0727. An O-grid-shaped smooth mesh of the tunnel box
(log-polar blended onto the box) left SIMPLE in a limit cycle, cell Péclet 5 in the
wake: no reference, dropped.

Tried and dropped: closing the Perot sum at a wall on the cell's own gradient
(`g0 + r (S.g0)/(V − S.r)`, the hypothesis that half a normal gradient at the wall
explains the Re 40 gap): cd −1.4e-2, gap WIDER (−3.65e-2 at N 256), cavity corner
1.6e-1. Reverted.

### Commands

```
tests:   simjob submit --gpu -t 100 -m 8192 --wait -- build/test_gpu_fv_incomp [t1..t7]
same mesh: tools/fv/incomp_cases.py cavity N C | tunnel N C | ogrid NTH NR C
         dns flow --polymesh C --fv-precision fp64 --max-iters I --out S
         test_gpu_fv_incomp cavity C S/I/U T CFL
         test_gpu_fv_incomp fixedpoint C C/bc.toml S/I/U NU T CFL [TOL]   (bc_conv.toml: convective)
ladder:  tools/fv/incomp_ladder.py tgv FV_96 MAC_96 FV_128 MAC_128 ...
         tools/fv/incomp_ladder.py cd T0 T1 FV_stats.csv MAC_stats.csv ...
TGV:     dns tgv --n N --ft 20 --cfl 0.4
         dns tgv --solver fv --n N --ft 20 --cfl 0.4 --fv-precision fp32
2D:      dns tgv --solver fv --nz 1 --n N --nu 0.05 --ft 1 --cfl 0.1 --emit error \
             --fv-precision fp64 --tol 1e-9 [--mesh-perturb A [--mesh-perturb-kind random]]
tunnel:  tools/fv/cyl_tunnel_mesh.py --n N --out T
         dns flow --polymesh T --bc T/bc.toml --re RE --ft 8F --fv-precision fp32
         dns flow --body cylinder --physics incompressible --far tunnel --n N --nz 1 \
             --re RE --dr 8 --ft F --fdt 1e9      (F: 6 Re 40, 16 Re 100)
```

Cost: 3D TGV 13, 61, 195 ms/step at N 64, 128, 192 FP32 (MAC is ~40× cheaper at
N 192, 415 s vs 12 s: FFT pressure vs AMG-PCG, launch bound); tunnel 7–12 ms/step.

## Limits

- **Order on a non-smooth mesh is ≤ 1.** Smooth map: 2. Random nodes: 1, viscous-
  dominated 0.4–0.65. So a snappy or O-ring/lattice junction (volume ratio 1.67 at
  every N) is not an order-2 mesh.
- **KE: the convective rate is exact, the total is not.** The projection dissipates:
  2.8% of the peak on the 192³ TGV, heavily on a random mesh.
- **`dt ~ h²/ν`** on the thinnest wall cell: the tunnel mesh is unclustered for this.
- **The fixed point carries c.** Steady numbers depend on dt through Rhie-Chow; a
  step of a different length re-projects (`step`'s precondition, T7).
- **Wall stress is one-sided**: cd on a body-fitted wall stays first order in dn in
  principle; measured flat to 3.4e-3 on the tunnel ladder.

## Low-Mach: `dns flame --physics lowmach --solver fv`

The reacting low-Mach model of `gpu/lowmach` (docs/chem.md "Low-Mach") on any FaceMesh.
Planar premixed flame, model C0, one species plus an optional passive one in the device
API. Same family as the rest of this page: `DeviceFvLowMach` lives in `gpu/fv_incomp`
beside `DeviceFvIncomp` and holds one: mesh view, least-squares gradients, `project()`,
AMG. `dns/fvlowmach` (host tables), `gpu/runner_fvlowmach` (time loop). Structured low-Mach
and `fv_incomp` without low-Mach are byte for byte ("Verification", low-Mach table).

```
dns flame --physics lowmach --solver fv --cells-per-delta C [--ny N] [--nz K] [--wrinkle A]
          [--upstream U] [--downstream D] [--inflow-speed S] [--tt T] [--cfl C] [--chem-dt K]
          [--tol T] [--fv-precision fp32|fp64] [--frame-n N] [--frame-box ...] [--profile-dt D]
dns flame --physics lowmach --polymesh DIR --bc FILE.toml --upstream U [same physics flags]
```

`--bc` is `docs/fvcomp.md`'s file (`type`, `reference.direction`, `reference.origin`,
ratios), with a low-Mach table: `inflow` fixes m = rho_b u_b, T, `y` (required); `outflow`
is p = 0, scalars zero gradient both ways; `open` (`t`, `y`: the ambient, default Tu and 0;
one ambient per run, two different ones exit 3 by name) the sign-of-flux face below;
`wall` adiabatic or isothermal, species-tight, no-slip; `slip`. `far` (no characteristic state), outflow `p`, `tback`, `burnt`, `y`, no
outflow at all (a closed box needs P0(t)): exit 3 by name, `--dry-run` included.
`tools/chem/fv_flame_mesh.py --lowmach` writes that file.

Also built (M3): `--equivalence PHI [--sl-law gri30|onestep] [--stoich] [--ox-y] [--mix-phi]`
(model C2 / C2b, "Two reactants" below) and `--mode bunsen` (the 2D slot burner, "Burner"
below, with `--dt-rule cell|bounds` and the burner flags `--slot-width`, `--jet-speed`,
`--coflow-*`, `--pilot-*`, `--slot-edge`, `--length`, `--reaction`), and on the burner
`--laterals open`, `--gravity G`, `--rim LIP [--rim-temp R] [--tube-length L]` ("Open faces,
gravity, rim" below; on a polyMesh: an `open` patch, `--gravity` along -`reference.direction`,
an isothermal wall patch).

Rejected by name: `--mach`, `--compensated` (always on in FP32), `--mode` diffusion, candle,
chamber, `--nozzle`, `--nz > 1` with `--mode bunsen`, the burner's `--profile-dt`, `--turb`
(LES and TFLES are built, M4: "Low-Mach LES" below; their pairs "Low-Mach combinations"), `--gravity` off the burner, `--laterals`, `--rim`,
`--rim-temp`, `--tube-length` with `--polymesh` (the mesh and `--bc` are the case).
`--dt-rule` only with `--mode bunsen`. Off the route `--tol`, `--bc`, `--fv-precision`;
`type = "open"` off it (dns flow, conv: a flame key; the compressible flame: by name).

**State.** rho, rho Y_k (cells; FP32: hi + Kahan lo, `combineWord2`, gpu/lowmach's words;
FP64 plain), rho u (cells), m (faces), p. T = P0/(R rho), Y = rho Y / rho derived.

**Stage** (`a` = 0, ¾, ⅓ as gpu/lowmach: cb = 0, ¼, ⅔, cr = cb dt or dt, ca = 1 − cb):

```
1 cell pass on b: T, Y, omega, u = (rho u)/rho
2 T* = T_a + (cb (T_b - T_a) + cr R_T), R_T = [-sum_f m (T_f - T_c)/V + div(rho D grad T) + q/cp omega] / rho_b
  V D = V (rho0 R_T / T* + P ca cb (rho_a - rho_b)^2 / (rho_a rho_b cr T*))    the structured cancellation-free target
3 (rho u)* = a + (cb (b - a) + cr R), R = -(1/V) sum_f (m (u_o + u_n)/2 - tau . S)      no pressure
4 m* = S . linear((rho u)*); A p = (V D - div m*)/cr + div(kg); m = m* - cr fp; rho u -= cr G
5 rho = a + (cb (b - a) - cr div m), rho Y likewise with F = m Y_f - rho D grad Y . S, sink -omega    on the PROJECTED m
```

Every line but 3 and 4 is `gpu/lowmach`'s stage on faces instead of MAC rows, operand for
operand: `lmfPut` is `lmPut`, the D expression is `kLmTemperature`'s, `kRhs` carries V D inside
the bracket so a difference of near-equal terms is read once before the scale by 1/cr.
The temperature predictor reads the stage's INPUT m, continuity the OUTPUT m, as structured.

**The pressure matrix is the geometric one, 1/rho never appears.** The momentum equation is
written for rho u and the projection acts on the MASS flux: `(rho u)_t = ... - grad p`, `m = m*
- cr fp`, so A is exactly `DeviceFvIncomp`'s compact `|S|^2/(S.d)` Laplacian (LmPressure's
constant coefficient, structured). Built, uploaded and AMG-factored once per run; no rebuild per
stage, no refresh. Scaled V, 2 Jacobi, flexible PCG, FP64 dots. Non-orthogonal meshes: `nCorr`
= 2 with kg, as `DeviceFvIncomp`. A pure Neumann case does not exist here (`DeviceFvLowMach::make`
refuses it): `Sum V D` is not 0 in a closed box. `p` is warm-started from the last stage.

**Momentum.** Collocated Mahesh form: u* carries no pressure, m* interpolates the predicted
rho u, the projection subtracts the compact gradient at faces and its Perot reconstruction at
cells. Conservative central flux `m (u_o + u_n)/2`: kinetic-energy consistent with continuity on
the same m. `DeviceFvIncomp`'s skew half `u_P div(phi)/2` is gone: div m is not 0 here and that term
is what the density's rate pays for. Stress: `fvTraction` of the AlongD face gradient, mu at the mean
face T. The 1D flame never reads this (m follows continuity); the 2D rows below do.

**One limiter.** theta = min over Y_k and T of `fvMcSlope(d1, d2)/d2` (`fvThetaVote`,
`sharedThetaOf` for any T; a field votes iff |d2| >= 2^-24), d1 the virtual upwind difference of
`fvFaceY`: on a uniform mesh `Y_up - Y_upup`. Face value `Y_up + theta/2 d2` (`fvFaceYShared`):
linear in the fields, so T + (q/cp) Y keeps one face formula, the Shvab-Zeldovich defect is the
structured one. A face whose upwind cell has no second neighbour goes central (theta 1):
`FvLowMachTables::first`, a boundary face behind the cell within 60 deg of the face's direction,
`sharedThetaOf`'s `firstL`, `firstR`. Tangential faces of a boundary cell keep the limiter.

**Boundaries.** Inflow: convective + diffusive face flux, half-cell gradient (`bCs`), T and Y
fixed, m fixed. Outflow: p = 0 at `bA`, u, T, Y zero gradient both ways, no diffusion. Wall:
m 0, species flux EXACTLY 0, heat EXACTLY 0 unless isothermal. Boundary kinds are
`FvIncompFaceBc`: nothing new in `buildFvIncompSetup`.

**dt.** min of `cfl / max_c sum_f |m| / (2 V min(rho_f, rho_c))`, `1 / max_c mu(T_c) max(4/3, 1/Pr)
sum_f (|S|^2/S.d) / (rho_c V)`, `--chem-dt / max J`. The diffusive line is the structured
`0.25 h^2 rho/mu` on a uniform 2D mesh. Its sum is the pressure matrix's `a`, not the `corr.S` of
the compressible rule: equal on an orthogonal mesh, larger past it, where the explicit gradient
correction adds stiffness. `corr.S` diverged in ~10 steps on the combined mesh (67 deg), stable
at half that dt; the compressible flame never sees it, its acoustic dt is smaller. Recomputed every `--dt-recompute` steps; equal steps to tEnd.

**Stats.** The structured runner's premixed columns, same names, then `burn`, `fuel`, `eDefect`;
`pcgIt` = the step's iterations. `FvFlameProbe::measureLowMach`: slabs normal to
`reference.direction`, a Cartesian mesh's slabs ARE the columns. `pFresh`, `pJump` the dynamic p.

**Energy.** rho T = P0/R pointwise, so `Sum V rho T` is constant to the EOS defect and the balance
is a statement about fluxes: boundary enthalpy in + (q/cp) Sum V omega = 0. Semi-discretely it is an
identity: `Sum_c V (rho R_T + T R_rho)` telescopes, every internal face's `m T_f - rho D g . S`
cancels (`budget()`, L3). The march does not telescope: R_T carries the input m, R_rho the output
m, a lag of one stage. So `eDefect` closes at ORDER 1 in dt and vanishes at a steady flame.
That is `gpu/lowmach`'s algebra verbatim, not an FV defect (L7 is rounding): the structured solver
has no energy tally to show it. Making it exact means a T_f-weighted pressure equation, not this model.

**Cost.** One hierarchy per run, no refresh: the whole process, mesh read to first step, 0.6 s at
43 008 cells. A step: 512 x 4 box structured 0.31 to 0.44 ms, FV FP32 1.8 ms (3 PCG iterations a
step, the noise floor stops it), FV FP64 4.3 ms (`--tol 1e-3`, 16 to 30 iterations a step). The FV
step is launch-bound. A 30-cell rung in FP64 takes
tt 8 inside 295 s only at `--tol 1e-3`; `--tol` moves S_L 1e-6 at 15 cells (1e-3 vs 1e-5).

### Two reactants (model C2, C2b; `--equivalence`)

`FvLowMachModel::ox = 1` (nY 2): gpu/lowmach's two-reactant algebra on faces, its own
instantiations, `ox < 0` the one-reactant kernels bit for bit. `kLmfCell<R, TwoR, Phi>`:
omega = `arrheniusRate2(lnA (+ phiRateLn(law, Y_F, Y_O)), ...)`, kLmCellPass<true, Phi>'s
call; `kLmfGatherS<R, Comp, TwoR>`: the oxidizer row's sink `- s omega` (kLmScalars' `div -
s w`); `kLmfRate<R, TwoR, Phi>`: LmChemJ's `k rho (Y_O + s Y_F + q/cp Y_F Y_O Ta/T^2)`. The
limiter was already one theta over Y_F, Y_O and T. Box IC = `lowmachIC` (flameIC2's rule)
cell for cell; a polyMesh lays the oracle s_p along `reference.direction` and
`premix2Reactants` turns it into Y_F, Y_O. Inflow `y = [Y_F1, Y_O1]` in the `--bc` file
(`tools/chem/fv_flame_mesh.py --lowmach --inflow-y A,B`); the route does not check the
values against the premix (a stratified inflow is legal). Stats append `dzMin`, `dzMax`
(Z - Z_in, gpu/runner_lowmach's); SL reads `rho (Y_F - Y_F,rem)` at the reference slab
(`FvFlameSpec::yBurnt`, 0 = C0's expression).

### Burner (`--mode bunsen`, 2D slot)

The structured burner's inflow rows are a TABLE on faces: `fvLowMachRows` writes each
face's `bU = (m T/P0) e`, `rUb = m e`, `bT`, `bY` (e the patch's unit velocity) before the
upload; the kernels read per-face tables already, so no device path is new. Box: the rows
of the owner's row j (lmInflowRow at (j + 1/2) hy, the structured `sp.rowM/rowT/rowY`
exactly), IC `lowmachBunsenIC` cell for cell with rho u = the row's m. polyMesh: the one
`jet = true` inflow patch takes lmInflowRow at the face centre, every cell
`lmBunsenPoint` (the structured IC's ONE rule, factored out of lowmachBunsenIC) at its
centroid, x = (C - origin).dir, y = (C - origin).lift + yc; dir = x, lift = y, 2D only
(exit 3 by name). `--cells-per-delta`, `--length` keep the slot box's meaning there (yc,
the measurement pixel).

`--dt-rule cell` (the burner default, structured S9): dt also <= posK / max_c (2 conv_c +
diff_c), `kLmfPos` on `kLmfRate`'s two rates. The FV conv rate is `sum_f |m| / (2 V min
rho)`, the MEAN of a cell's faces per axis where LmConvRate takes the larger face: a
stated rule difference, so a binary run's dt is not the structured run's.

Stats: gpu/runner_lowmach's burner columns, same names, closed laterals. The cone
(`measureCone`, `flankMassFlux`) on a uniform plane through the mesh (`FvRaster`, pixel h
= the flags' cell, lower corner the inflow plane at y = yc - origin): on the box pixel =
cell, bitwise the cell values. Integrals exact on cells per unit depth; fuelIn, fuelOut the
current state's boundary face fluxes (`DeviceFvLowMach::faces()`); the flame index on the
least-squares gradients (`flameIndex()`, gpu/derived's expression; interior equal to the
central difference on the box, one-sided ends differ). The structured exit-8 anchoring
guards (no rim).

### Open faces, gravity, rim (M3 steps 3-5)

**Open faces** (`type = "open"`, the box's `--laterals open`: `ylo`, `yhi`; `FvLowMachTables::bOpen`,
2 on an open patch, 1 on every outflow while one exists: the structured x-hi backflow rule).
p = 0 as an outflow's (the AMG matrix unchanged), and ONE rule per face by the sign of the
face's own flux, gpu/lowmach's: OUT zero gradient (u, T, Y the cell's), IN the quiescent
ambient (T_a, Y_a, rho_a = P0/(R T_a), the face velocity phi_b/(|S| rho_a) along n, tangential
0) with the face pressure `bP = -beta rho_a u_n^2/2` (`kLmfOpenP`, openFaceP's operands; beta =
`FvLowMachModel::backflow`, 1). No diffusion either way. Which flux decides: the temperature
predictor, momentum and the face p read the stage INPUT's (structured: latT, kLmMomT read M of
b), the species flux the PROJECTED one it carries (`kLmfBndY`, kLmScalars'): upwinding species
by the input's sign carried the ambient's Y_O out of a face whose projected flux had turned
outward, Y_O -2.6e-5 (O3, measured; structured 3.7e-25). Viscous: the normal stress is the
CELL's (gpu/lowmach's zero normal-stress gradient), the tangential the one-sided full stress
toward u_b (the structured edge stresses: zero gradient out, still air in). Traction-free
(tau_nn = 0) is another boundary condition: it left the boundary rows O(1) off MAC and the l1
distance growing with h (O4 before the fix: 2.7e-2, 3.1e-2, 3.3e-2). Absent (`bOpen` empty):
none of it launches, the solver bit for bit.

**Gravity** (`FvLowMachModel::gravity`, the vector: the box's `--gravity G` is (-G, 0, 0), a
polyMesh's -G `reference.direction`). `addBuoyancyLm` on faces: every internal face's m* gains
`cr (rho_f - rho_a) g . S` (`kLmfBuoyF`, rho_f the linear face density, the two-cell mean on a
uniform mesh), and the reconstruction reads `fp - B` (`IncompImpl::buoyant`): a hydrostatic
state cancels face by face, `DeviceFvIncomp`'s Boussinesq face force with the low-Mach density.
A p-fixed face (outflow, open) carries the force of its upwind density (the cell's; on an open
face's inflow the ambient's: 0), the structured x-hi face; Neumann faces none, their hydrostatic
image `dpB = (rho_b - rho_a) g . r` feeds the deferred `kGradP`. Hydrostatics live in p: the
quiescent ambient is p = 0 exactly. Absent (g = 0): none of it launches.

**Rim** (`--rim` on the box): the tube's staircase (`lmTubeSolid`, the ONE tube definition)
cells REMOVED from the box (`fvLowMachBox(g, openY, solid)`: polyOf, fluid cells kept in
ascending order, a fluid/solid face a `rim` face owned by its fluid cell, S out of the fluid;
srcCell stays Grid::idx), the `rim` patch an isothermal Wall at `--rim-temp`. Nothing new in a
kernel: a Wall face is impermeable (phi_b == 0), species-tight (flux EXACTLY 0), no-slip, and
conducts `rho D(T_w) (T_w - T_P) |S|/(h/2)` (the structured wall face's form); the limiter's
first-side rule is the structured "second neighbour solid: central". The structured staircase
is the oracle on these cells; a body-fitted tube (a polyMesh with its own wall patch) has no
structured twin. IC, rows (Poiseuille, the room draft), stats (`xRimBase`, `rBase`, `qRim` =
cp x the rim faces' conduction, per unit span) and the anchoring guards (flash-back below the
rim, blow-off above half the cone over it) are gpu/runner_lowmach's. A polyMesh burner takes
no `--rim`: its wall patch is the tube.

Gates (`test_gpu_fv_lowmach burner`, ctest `gpu_fv_lowmach_burner`):

| gate | case | measured | bound |
|---|---|---|---|
| O1/G1 rest | quiescent open box (rho = rho_a, Y the ambient's, m_in 0), 32 x 16, 1000 steps of dt 0.05, g 0 and 1, both precisions | rho u, m, p exactly 0 | exact |
| O2 backflow | a vortex (mu 0.01) 3 cells inside the open y-hi face, 64 x 48, 2000 steps of dt 0.01, KE every 100 from the first step's (projected) field | beta 1: KE max/KE0 1.000000 (end 0.457); beta 0: 4.47 (structured 4.42) | <= 1 + 1e-6; > 1.5 |
| O3 tallies | the hot blob through open y (64 x 32, 100 steps), uniform Y 0.5 / 0.25 at that ambient; a pure-fuel box under an air ambient | 0 cells off; mass, species - tally of the throughput: FP64 <= 8.4e-14, FP32 <= 2.6e-9; air: Y_O in [3.6e-25, 0.0496] (structured 3.7e-25), lateral inflow 0.72 | bitwise; FP64 1e-13, FP32 1e-7; [-1e-6, Y_O2] |
| O4 vs structured | the blob, open y, swirl vanishing on the open faces, FV64 vs structured(FP32), dt 0.03 h^2, t 0.25, 32 x 16 to 128 x 64 | l1 rho 7.62e-5, 7.04e-5, 3.76e-5 (0.11, 0.90); l1 rho u 1.37e-2, 8.52e-3, 4.58e-3 (0.68, 0.90) of 1.25; max rho u 0.141, 0.296, 0.171 (the boundary row); lateral outflow -1.38e-2, -2.00e-2, -1.07e-2 of 0.13-0.21 | l1 shrinks |
| G3 vs structured | the same, g 0.5 | l1 rho 7.51e-5, 7.02e-5, 3.74e-5 (0.10, 0.91); l1 rho u 1.36e-2, 8.46e-3, 4.53e-3 (0.69, 0.90) | l1 shrinks |
| G2 hydrostatic | stratified column (T 1 -> 1.5 up, x-lo m 0, p = 0 on top), g 1 vs g 0, same dt 0.01, 1000 steps, FP64 | |rho u(g) - rho u(0)| / sqrt(g L): box 4.3e-16, sheared 21 deg 5.2e-5, random 0.3 h 3.1e-6 | 1e-4 (gpu/lowmach's; applied to sheared, random after their first measurement) |
| R2 rim conduction | x-lo T 2 m 0, a slab across y T_w 1 at x 12, open y, 500 steps | max |T - T_lin| FP32 4.0e-7, FP64 2.2e-16; wall heat = lambda dT/L L_y = the inflow face's (0.333333, over cp) | FP32 2e-6, FP64 1e-12; 1e-5 |
| R1 vs structured staircase | a slot tube (lips 2 f cells, x < 3, T_w 1.5), jet m 1, draft 0.2, passive Y 0.5, open y, t 0.125, 48 x 32 to 192 x 128 | uniform Y bitwise; rim faces phi_b == 0, species flux == 0 exactly; mass, species tallies <= 5.2e-15; l1 rho 4.00e-5, 2.20e-5, 8.74e-6 (0.87, 1.33); l1 rho u 7.33e-3, 2.87e-3, 1.11e-3 (1.35, 1.37) of 4.1; 0.5 off the tube's corners 5.40e-2, 2.41e-2, 1.03e-2 (1.17, 1.23); the max (inflow corner) 0.141, 0.098, 0.098 (flat: a corner cell, LF8) | bitwise; exact; 1e-13; l1 and off the corners shrink |

The ctest takes 174 s on a shared GPU.

### Low-Mach LES (M4: `--les`, `--sgs`, `--tfles`)

gpu/lowmach's LES and TFLES (docs/chem.md "Low-Mach LES") on faces, operand for operand
(`lmfEffD` is `lmEffD`, species.hip's effRhoD operands): `kLmfNut` mu_t = rho (Cs Delta)^2
rate of the least-squares gradient (I.G, the cell velocity (rho u)/rho), Delta^2 =
`FvIncompSetup::delta2`, per stage after the gradients, riding `I.nut`; `kLmfFaceM<R, true>`
mu + mu_t,f; `kLmfFaceT`, `kLmfFaceY<R, Corr, Diff>` rho D + mu_t,f/Sc_t (Eddy) or F E_f rho D +
(1 - Omega_f) mu_t,f/Sc_t (Thick) on internal faces, boundary faces molecular; `kLmfCell<.., Eof>`
omega E/F; `kLmfRate<.., Eddy>` the cell's mu max(4/3, FE/Pr) + mu_t max(4/3, 1/Sc_t). TFLES:
gpu/tfles' sensor on dns/fvles' face walk (`fvLowMachTfles`: FvLowMachTables::walk, the edge
layers), Colin's u' on the compact cS Laplacian, the Charlette combine; a uniform sensor
without the efficiency is formed once at make. Off: none of it launches (the M3 kernels).

Same mesh + same dt vs structured (docs/chem.md table): TFLES is rounding (X1: the T, Y
algebra never reads the velocity); Smagorinsky is a scheme gap of mu_t (X2: the cell
velocity is (rho u)/rho here, the mean of the face m/rho_f there; order 2, the march's rho gap
reaches the L7 rounding at h/4); sigma on a planar flame is 0 in both (L1, bitwise).
`--les` on a 2D polyMesh: exit 3 (the compressible FV flame's rule).

### Low-Mach turbulence (M5: `--turb`)

The planar box only (docs/chem.md "Low-Mach turbulence"): the IC adds rho w u' to every
centroid's rho u (`fvLowMachAddTurbulence`), the xlo faces' `bU`, `rUb` are rewritten before
every step from the box swept past their centres (`TurbSweep`, `fvLowMachTurbRows`,
`DeviceFvLowMach::setInflow`: a host table upload, no device path); only inflow faces read
them, so the kernels are untouched. u' 0 = the tables bit for bit (L1). Same mesh + same dt
vs structured is a scheme gap of order 2 (L2: collocated vs MAC, and the structured tangential
rows sit on the y / z edges, the FV ones on the face centres). `--turb` on a polyMesh: exit 2.

### Low-Mach combinations

`dns/lmmodules` (the compressible family's `dns/fvmodules` shape): the low-Mach stage's
modules, one decision per pair, read by `validate()` (`lowMachModules`: the refusal names its
flag), both solvers' `make()` guards and `gpu_lm_matrix` (`test_gpu_lowmach_les matrix`:
every live set holding an M4 module, singles and pairs closed under the requirements, on a 3D
box, a sine-perturbed 3D box and the open 3D box: (a) TFLES F 1 = the cell without TFLES
byte for byte (without LES; with it Omega switches the SGS scalar flux off, so F 1 is not
LES alone), (b) passive uniform Y 0.25 bitwise / Z == 1 kept (C2, C2b: measured FP64 <= 7.8e-16,
FP32 <= 3.2e-8), mass and fuel tallies (bounds FP64 1e-13, FP32 1e-7; measured <= 4.5e-15,
<= 7.8e-11), (c) rerun bitwise both precisions, (d) Source / Morton / ColourMajor FP64 to
summation order (<= 3.0e-12); 22 sets, 42 rows, 386 checks, 51 s on a shared GPU; M5: 29 sets, 56 rows with the Turb sets, 81 s; a Turb cell adds (a) rows of u' 0 = the cell without Turb bitwise and runs (b)-(d) with swept xlo rows). The M3 pairs (C0/C2/C2b with gravity, open, rim) keep their own gates
(`gpu_fv_lowmach*`). Generated below (`test_gpu_lowmach_les markdown`):

<!-- generated: dns::lmCombinationsMarkdown (src/dns/lmmodules.cpp); test_gpu_lm_matrix markdown -->
| Module | Axis | Spelling |
|---|---|---|
| LES smag | turbulence | `--les Cs --sgs smag` |
| LES sigma | turbulence | `--les C --sgs sigma` |
| TFLES uniform | flame | `--tfles F --tfles-sensor uniform` |
| TFLES dynamic | flame | `--tfles F [--tfles-sensor dynamic]` |
| Charlette | efficiency | `--efficiency charlette` |
| C0 | chemistry | `dns flame --physics lowmach (premixed, bunsen)` |
| C2 | chemistry | `--equivalence PHI --sl-law onestep` |
| C2b | chemistry | `--equivalence PHI [--sl-law gri30]` |
| gravity | gravity | `--gravity G (--mode bunsen)` |
| open | open | `--laterals open; type = open in --bc` |
| rim | rim | `--rim LIP, an isothermal wall patch` |
| turb | inflow | `--turb U (the planar flame, --nz >= 8)` |

Pair matrix: L live, R refused, x one axis (exclusive).

|   | LES smag | LES sigma | TFLES uniform | TFLES dynamic | Charlette | C0 | C2 | C2b | gravity | open | rim | turb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LES smag | . | x | L | L | L | L | L | L | L | L | R | L |
| LES sigma | x | . | L | L | L | L | L | L | L | L | R | L |
| TFLES uniform | L | L | . | x | L | L | R | R | L | R | R | L |
| TFLES dynamic | L | L | x | . | L | L | R | R | L | R | R | L |
| Charlette | L | L | L | L | . | L | R | R | L | R | R | L |
| C0 | L | L | L | L | L | . | x | x | L | L | L | L |
| C2 | L | L | R | R | R | x | . | x | L | L | L | R |
| C2b | L | L | R | R | R | x | x | . | L | L | L | L |
| gravity | L | L | L | L | L | L | L | L | . | L | L | R |
| open | L | L | R | R | R | L | L | L | L | . | L | R |
| rim | R | R | R | R | R | L | L | L | L | L | . | R |
| turb | L | L | L | L | L | L | R | L | R | R | R | . |

| Pair | Decision | Error text | Why |
|---|---|---|---|
| LES smag + rim | refused | `--les with --rim: not built (no wall model for mu_t at the immersed staircase)` | the cell-centred velocity differences read the solid cells' zero velocity: mu_t peaks at the lip with no damping (the incompressible cut-cell LES caveat) |
| LES sigma + rim | refused | `--les with --rim: not built (no wall model for mu_t at the immersed staircase)` | the cell-centred velocity differences read the solid cells' zero velocity: mu_t peaks at the lip with no damping (the incompressible cut-cell LES caveat) |
| TFLES uniform + C2 | refused | `--tfles with --equivalence: the thickened flame is premixed C0 only (low-Mach TFLES of two reactants is refused)` | F keeps one S_L and E wrinkles one (S_L, delta_L) pair: the C2 burner carries an outer Burke-Schumann diffusion sheet whose structure thickening does not preserve, and the sensor's normalisation is the one-step C0 manifold's (tflesSensorLogMax) |
| TFLES uniform + C2b | refused | `--tfles with --equivalence: the thickened flame is premixed C0 only (low-Mach TFLES of two reactants is refused)` | F keeps one S_L and E wrinkles one (S_L, delta_L) pair: the C2 burner carries an outer Burke-Schumann diffusion sheet whose structure thickening does not preserve, and the sensor's normalisation is the one-step C0 manifold's (tflesSensorLogMax) |
| TFLES uniform + open | refused | `--tfles with --laterals open: the sensor's T-gated max filter wraps y and z periodic (gpu/tfles)` | an open lateral is no wrap: the dilation would read across the box |
| TFLES uniform + rim | refused | `--tfles with --rim: not built (the sensor dilates through the tube's solid cells)` | the max filter has no solid gate |
| TFLES dynamic + C2 | refused | `--tfles with --equivalence: the thickened flame is premixed C0 only (low-Mach TFLES of two reactants is refused)` | F keeps one S_L and E wrinkles one (S_L, delta_L) pair: the C2 burner carries an outer Burke-Schumann diffusion sheet whose structure thickening does not preserve, and the sensor's normalisation is the one-step C0 manifold's (tflesSensorLogMax) |
| TFLES dynamic + C2b | refused | `--tfles with --equivalence: the thickened flame is premixed C0 only (low-Mach TFLES of two reactants is refused)` | F keeps one S_L and E wrinkles one (S_L, delta_L) pair: the C2 burner carries an outer Burke-Schumann diffusion sheet whose structure thickening does not preserve, and the sensor's normalisation is the one-step C0 manifold's (tflesSensorLogMax) |
| TFLES dynamic + open | refused | `--tfles with --laterals open: the sensor's T-gated max filter wraps y and z periodic (gpu/tfles)` | an open lateral is no wrap: the dilation would read across the box |
| TFLES dynamic + rim | refused | `--tfles with --rim: not built (the sensor dilates through the tube's solid cells)` | the max filter has no solid gate |
| Charlette + C2 | refused | `--tfles with --equivalence: the thickened flame is premixed C0 only (low-Mach TFLES of two reactants is refused)` | F keeps one S_L and E wrinkles one (S_L, delta_L) pair: the C2 burner carries an outer Burke-Schumann diffusion sheet whose structure thickening does not preserve, and the sensor's normalisation is the one-step C0 manifold's (tflesSensorLogMax) |
| Charlette + C2b | refused | `--tfles with --equivalence: the thickened flame is premixed C0 only (low-Mach TFLES of two reactants is refused)` | F keeps one S_L and E wrinkles one (S_L, delta_L) pair: the C2 burner carries an outer Burke-Schumann diffusion sheet whose structure thickening does not preserve, and the sensor's normalisation is the one-step C0 manifold's (tflesSensorLogMax) |
| Charlette + open | refused | `--tfles with --laterals open: the sensor's T-gated max filter wraps y and z periodic (gpu/tfles)` | an open lateral is no wrap: the dilation would read across the box |
| Charlette + rim | refused | `--tfles with --rim: not built (the sensor dilates through the tube's solid cells)` | the max filter has no solid gate |
| C2 + turb | refused | `--turb with --sl-law onestep (model C2): not built (no turbulent number)` | the turbulent box is measured on C0 and C2b; the uncorrected one-step C2 is the S1 diagnostic step |
| gravity + turb | refused | `--turb on the low-Mach burner (--rim, --gravity): not built (the low-Mach --turb is the planar flame box's)` | the burner's inflow is the jet's per-row profile (lmInflowRow); a turbulent jet weight J(y) is the compressible slot's and has no low-Mach counterpart |
| open + turb | refused | `--turb with --laterals open: not built (the synthetic box is periodic in y and z)` | an open lateral is no period of the box; its rows would wrap across it |
| rim + turb | refused | `--turb on the low-Mach burner (--rim, --gravity): not built (the low-Mach --turb is the planar flame box's)` | the burner's inflow is the jet's per-row profile (lmInflowRow); a turbulent jet weight J(y) is the compressible slot's and has no low-Mach counterpart |
| Charlette without LES smag or LES sigma | requirement | `--efficiency charlette needs --les (the sub-grid velocity of a filtered field)` | |
| Charlette without TFLES uniform or TFLES dynamic | requirement | `--efficiency needs --tfles F (>= 1)` | |
| TFLES uniform without C0 | requirement | `--tfles: the premixed C0 flame` | |
| TFLES dynamic without C0 | requirement | `--tfles: the premixed C0 flame` | |

| Live pair | Hand gates |
|---|---|
| LES smag + C0 | gpu_lowmach_les X2 (FV vs structured: the scheme gap shrinks) |
| LES smag + C2b | gpu_lm_matrix (b): Z == 1 kept under the one-Sc_t eddy flux |
| LES smag + turb | gpu_lm_turb (IC, rows, structured vs FV), tools/chem/turb_fv.sh lm, lmpairs (S_c vs structured) |
| LES sigma + C0 | gpu_lowmach_les L1 (sigma on a planar flame = DNS bitwise, both solvers) |
| LES sigma + C2b | gpu_lm_matrix (b): Z == 1 kept under the one-Sc_t eddy flux |
| LES sigma + turb | gpu_lm_turb (IC, rows, structured vs FV), tools/chem/turb_fv.sh lm, lmpairs (S_c vs structured) |
| TFLES uniform + C0 | gpu_lowmach_les X1 (FV vs structured: rounding), S_L of F |
| TFLES uniform + turb | gpu_lm_turb (IC, rows, structured vs FV), tools/chem/turb_fv.sh lm, lmpairs (S_c vs structured) |
| TFLES dynamic + C0 | gpu_lowmach_les X1 (FV vs structured: rounding) |
| TFLES dynamic + turb | gpu_lm_turb (IC, rows, structured vs FV), tools/chem/turb_fv.sh lm, lmpairs (S_c vs structured) |
| Charlette + turb | gpu_lm_turb (IC, rows, structured vs FV), tools/chem/turb_fv.sh lm, lmpairs (S_c vs structured) |
| C0 + turb | gpu_lm_turb (IC, rows, structured vs FV), tools/chem/turb_fv.sh lm, lmpairs (S_c vs structured) |
| C2b + turb | gpu_lm_turb (IC, rows, structured vs FV), tools/chem/turb_fv.sh lm, lmpairs (S_c vs structured) |

### Low-Mach verification

Same table rules as `docs/fvcomp.md` "Cross-verification". Agreement proves the port, not
the physics. Flame rows: sigma 5, beta 8, Pr 0.72, u_in 0.98, S_L and delta_th means over
[tt/2, tt], `tools/chem/fv_flame_ladder.py --physics lowmach`. Orders rest on 3 levels.

| solver pair | mesh | quantity | expected relation | measured |
|---|---|---|---|---|
| uniform Y 0.5, 0.25 (L1) | box, sine, snappy; FP32 compensated, FP64; burning and inert, 40 steps | rho Y vs Y rho | bitwise | 0 cells off |
| mass (L2) | same, burning, 40 steps | inventory vs FP64 tally | rounding | FP64 1.3e-15 / 1.1e-15 / 1.1e-15; FP32 1.7e-11 / 1.0e-11 / 3.0e-11 |
| species (L2) | same | y + burn - tally; passive | rounding | fuel 3.4e-15 / 8.6e-16 / 1.7e-15, passive as mass; FP32 fuel 1.0e-11 / 1.5e-11 / 4.3e-11 |
| energy, semi-discrete (L3) | same, IC and t 40 steps | Sum V (rho R_T + T R_rho) - boundary flux - q/cp Sum V omega, of gross | rounding | FP64 2.0e-16 / 1.1e-16 / 5.5e-17; FP32 2.5e-7 / 2.9e-8 / 7.0e-9 |
| energy, the march (L3) | box, burning front, 200 steps, dt, dt/2, dt/4 | Sum V rho T - tally, relative | order 1: m lags one stage | 7.7e-5, 3.9e-5, 1.9e-5: 1.00, 1.00. Flame, 15 cells, t 1 to 7.6: `eDefect` rate 1.9e-7 against 3.4 of heat release, the flame sliding at 0.018 |
| EOS, projection (L2) | same, FP64 | max |rho T*/P - 1|; max |div m - D| | solver tolerance | 6.7e-16 / 4.0e-15 / 9.8e-15; 1.1e-13 / 4.1e-12 / 8.6e-11 |
| rerun (L4) | same, both precisions | every field | bitwise | equal |
| 2D (L5) | same, both precisions | rho w | exactly 0 | exactly 0 |
| freestream (L6) | sine, 20 steps | rho u, rho | rounding | FP64 0, 0; FP32 1.9e-8, 2.4e-9 |
| FV vs `DeviceLowMach`, same dt, 400 steps (L7) | 128 x 4 box, burning | rho, rho Y | rounding, <= 3 x FV32 vs FV64 | FV32 1.9e-7, own 1.9e-7; FV64 vs structured(FP32) 1.8e-8 |
| FV(FP64) vs structured(FP32) | 7.5, 15, 30 cells, 1D box | S_L | rounding: <= FV32 vs FV64 | 0.9942915 vs 0.9942912, 0.9982715 vs 0.9982716, 0.9995362 vs 0.9995364: +2.4e-7, -6.0e-8, -1.6e-7 (FV32 - FV64 +8.3e-6, +4.7e-6, +1.4e-6) |
| same | same | delta_th | same | 1.616601 vs 1.616599, 1.601806 vs 1.601806, 1.601634 vs 1.601634: +2.4e-6, +2.0e-7, +9.0e-7 (own -1.7e-5, -8.7e-6, -4.3e-6) |
| same | same | Shvab-Zeldovich defect, last row | rounding | FV 9.9e-7, 3.4e-7, 1.9e-7; structured 8.6e-7, 5.8e-7, 4.8e-7. Y hits 0 |
| FV(FP64) vs oracle | same | S_L - 1 | order <= 2, MC limits T | -5.71e-3, -1.73e-3, -4.64e-4 (1.72, 1.90); window spans 4.1e-5, 4.0e-6, 1.0e-6 |
| FV vs MAC, wrinkled front (L8) | 64 x 16, 128 x 32, 256 x 64, FP64 vs FP32, same dt | max |rho v|, peak 0.248 | order 2: the schemes differ, collocated vs staggered | 1.08e-2, 2.69e-3, 6.72e-4 (2.01, 2.00); rho u 7.1e-3, 1.8e-3, 4.6e-4; rho 1.4e-4, 3.4e-5, 8.4e-6 |
| FV(FP32) vs structured, wrinkled flame, A 0.3, mode 1, t 1 | 7.5, 15, 30 cells, ny 16, 32, 64 | S_L | not read by momentum at this A | +5e-6, +4e-6, +1e-6 at the CSV's 6 digits; x_f equal to 1e-4 |
| rotated 30 deg vs planar | 7.5, tight (`--tol 1e-5`, ny 8 both), FP64 | S_L; delta_th | rounding of the geometry | -3.5e-8; +3.4e-7 |
| same, `--tol 1e-3` | 7.5, 15, 30, ny 8 mesh, ny 4 planar | S_L | the tolerance's | -7.8e-6, -5.5e-7, +4.5e-8: at this tol the two solve different iterates, not a mesh effect |
| sheared 30 deg vs planar | 7.5, 15, 30, ny 8, FP64 | S_L | order 2 | -3.92e-3, -9.75e-4, -2.41e-4 (2.01, 2.02); compressible -3.50e-3, -9.40e-4, -2.38e-4 |
| same | same | delta_th | order 2 | -3.86e-2, -4.79e-3, -1.23e-3 (3.01, 1.96) |
| wavy (A 2 h) vs planar | 7.5, 15, 30, ny 16, FP64 | S_L | order 2 | +1.21e-3, +3.11e-4, +7.89e-5 (1.96, 1.98). The 30-cell rung ran on the `corr.S` dt, the final rule timed out (dt only; the other two rungs moved 2e-6 at most) |
| same | same | delta_th | order 2 | +1.05e-2, +6.33e-3, +1.63e-3 (0.73, 1.96): slab read across curved i lines |
| shear + wavy + rotate 20 deg vs planar | 7.5, 15, ny 16, FP64, 67 deg max | S_L | order 2 | -2.51e-3, -6.47e-4 (1.96). 30: not measured |
| same | same | delta_th | order 2 | -7.91e-3, +1.13e-3: sign change, no order |
| two reactants, Z (R1) | box, sine, snappy; C2 and C2b; 40 steps | max |Z - 1| | rounding | FP64 6.7e-16 / 1.4e-15 / 8.9e-16 (C2b 6.7e-16 / 1.7e-15 / 8.9e-16); FP32 2.9e-8 / 3.5e-8 / 2.7e-8 (C2b 3.1e-8 / 3.0e-8 / 3.5e-8) |
| two reactants, tallies (R2, R3) | same | fuel + burn, oxidizer + s burn, mass vs FP64 tallies; energy rate | rounding | FP64 <= 1.9e-15 every row, energy <= 2.7e-16; FP32 <= 8.9e-11, energy <= 1.1e-7 |
| FV vs `DeviceLowMach` two-reactant C2b, same dt, 400 steps (R7) | 128 x 4 box, burning, phi 1.3 | rho, rho Y_F, rho Y_O | rounding, <= 3 x FV32 vs FV64 | FV32 3.16e-7, own 3.24e-7; FV64 vs structured(FP32) 5.2e-8 |
| FV(FP64) vs structured(FP32), C2b, Sutherland 298 K, sigma 6.88 | 7.5, 15 cells, 1D box, [10, 20] | S_L | rounding: <= FV32 vs FV64 | 0.9945958 vs 0.9945961, 0.9984696 vs 0.9984699: -3.0e-7, -3.0e-7 (FV32 - FV64 -8.0e-6, +4.4e-6). 30 cells: not measured (FV64 ~800 s) |
| same | same | delta_th; Z - Z_in | rounding | 4.700089 vs 4.700087, 4.711926 vs 4.711924; FV64 Z within 1.3e-14 / 5.6e-14 of Z_in, FV32 1.1e-7 / 1.2e-7, structured 1.7e-7 / 1.8e-7 |
| C2b sheared 30 deg (polyMesh) vs planar, FP64 | 7.5, 15 cells, ny 8, 16 | S_L | order 2 | 0.9926079, 0.9980001: -1.99e-3, -4.70e-4 (2.08); Z within 8.4e-15 / 4.8e-14 |
| per-face rows (B1, B2) | 48 x 32 box, a hot slot in a fresh coflow, burning, 40 steps | inflow phi_b vs - m_row |S|; mass, fuel, passive vs tallies | rounding | FP64 0, 5.3e-15, 1.2e-16, 5.3e-15; FP32 5.7e-8, 2.5e-11, 7.3e-11, 2.5e-11 |
| FV(FP32) - structured(FP32), G2a slot burner (C0, sigma 5, W 12, U 5, burnt coflow, `--length 49`), each its own dt | 4, 6, 8 cells, window [20, 25] | xTip | scheme gap (collocated vs MAC momentum, cell u (rho u)/rho vs face-mean m/rho_f): shrinks | 44.3072 vs 44.3040, 44.0420 vs 44.0405, 43.9278 vs 43.9267: +3.18e-3, +1.45e-3, +1.07e-3 (1.94, 1.04) |
| same | same | tanA; sinRatio | same | -8.07e-5, -4.50e-5, -2.79e-5 (1.44, 1.66); -3.80e-4, -2.10e-4, -1.30e-4 (1.46, 1.66) |
| same | same | SLflank; mFlank | same | +2.86e-5, +2.05e-5, +1.40e-5 (0.82, 1.33); -1.93e-3, -2.12e-3, -1.34e-3 (no order 4 to 6, 1.60 6 to 8) |
| same | same | burnRatio; massErr | rounding; the tally | both 0.99999 to 1.2e-6; FV 3e-11 to 4.4e-10, structured -4.7e-8, +2.6e-8 (its IC adjustment, chem.md) |
| FV(FP32) - structured(FP32), rich2d partially premixed slot (C2b phi 1.3, sigma 6.88, Sutherland 298 K, Pr 0.52, W 4, U 3, pilots 1.5 at 3, air coflow 6, `--length 50`) | 4, 6, 8 cells, window [30, 40] | inner xTip; outer xTipOut | scheme gap: shrinks | 17.7787 vs 17.7868, 17.7450 vs 17.7497, 17.7106 vs 17.7143: -8.03e-3, -4.65e-3, -3.74e-3 (1.35, 0.76); 79.8441 vs 79.8243, 80.0348 vs 80.0269, 80.0899 vs 80.0848: +1.99e-2, +7.83e-3, +5.05e-3 (2.30, 1.52) |
| same | same | tanA; sinRatio | same | -9.82e-4, -4.42e-4, -2.14e-4 (1.97, 2.52); -2.42e-3, -1.09e-3, -5.25e-4 (1.98, 2.53) |
| same | same | SLflank; mFlank | same | +3.01e-4, +9.04e-5, +6.79e-5 (2.97, 0.99); +2.03e-2, +7.32e-3, +3.79e-3 (2.52, 2.29) |
| same | same | preFrac; burnRatio; Z, yHits | same; rounding | -8.9e-6, -2.7e-4, -6.5e-5 (no order); both 0.99987 to 4e-6; FV Z in [2.3e-8, 1 + 1e-7] at 8 cells, 0 hits both |
| polyMesh `--wavy 2` (h) of the G2a box vs the box, FV32 | 6 cells, one rung | xTip, tanA, sinRatio, SLflank, mFlank | the mesh's truncation | +1.2e-3, -7.6e-5, -3.6e-4, +1.7e-4, -1.4e-4 (44.0432, 0.208502, 1.02056, 0.989016, 0.990589); burnRatio 0.99999, massErr -4.8e-10, 0 hits. 4, 8 cells: not measured |
| FV(FP64, `--tol 1e-3`) - FV(FP32), G2a | 6 cells | xTip, tanA, mFlank | the precision's (and tol) | +1.1e-4, -8.5e-7, -2.3e-6: below the gap above |
| FV(FP32) - structured(FP32), G2a `--laterals open` (the open face kind on ylo, yhi), each its own dt | 4, 6, 8 cells, window [20, 25] | xTip | scheme gap: shrinks | 44.2862 vs 44.2823, 44.0217 vs 44.0192, 43.9049 vs 43.9026: +3.92e-3, +2.47e-3, +2.25e-3 (1.14, 0.32) |
| same | same | tanA; sinRatio | same | -1.51e-4, -1.02e-4, -8.62e-5 (0.97, 0.57); -7.10e-4, -4.79e-4, -4.06e-4 (0.97, 0.58) |
| same | same | SLflank; mFlank | same | +6.7e-6, +2.22e-5, +1.52e-5 (no order 4 to 6, 1.31); -2.02e-3, -2.19e-3, -1.41e-3 (none, 1.54) |
| same | same | mdotOut; mdotLat (net in); mdotIn | the outflow's partition: PLATEAU (issues LF6) | 63.718 vs 63.808, 63.823 vs 63.923, 63.832 vs 63.946: -9.0e-2, -1.01e-1, -1.13e-1; mdotLat -9.2e-2, -1.03e-1, -1.15e-1 (0.25-0.31% of it, grows); mdotIn 100.692 both (the rows) |
| same | same | uLatMin; uLatMax | pointwise extremes | -4.4e-3, -2.5e-3, -1.4e-3 (1.35, 2.01); +1.96e-2, +1.89e-2, +1.87e-2 (1%, flat) |
| same | same | burnRatio; massErr; yHits | rounding; the tally | 1.00025 vs 1.00022 to 1.00022 vs 1.00020 (both); FV max massErr 1.1e-9, 0 hits |
| polyMesh `--wavy 2 --open-y` (`tools/chem/fv_flame_mesh.py`: ylo, yhi `type = "open"` patches) vs the box `--laterals open`, FV32 | 6 cells, one rung | xTip, tanA, sinRatio, SLflank, mFlank; mdotLat | the mesh's truncation | +1.2e-3, -9.0e-5, -4.2e-4, +1.7e-4, -9.4e-5 (44.0229, 0.201216, 0.986311, 0.990362, 0.99345); mdotLat -1.5e-2 of 36.9; burnRatio equal to 2e-6, massErr 4.5e-10, 0 hits. 4, 8 cells: not measured |
| FV(FP32) - structured(FP32), the same + `--gravity 0.0964` (4x real g) | 4, 6, 8 cells, [20, 25] | xTip | scheme gap: shrinks | 44.2873 vs 44.2841, 44.0207 vs 44.0193, 43.9044 vs 43.9034: +3.17e-3, +1.35e-3, +9.68e-4 (2.10, 1.16) |
| same | same | tanA; sinRatio | same | -1.52e-4, -1.09e-4, -8.56e-5 (0.83, 0.83); -7.14e-4, -5.12e-4, -4.03e-4 (0.82, 0.83) |
| same | same | SLflank; mFlank | same | +9.1e-6, +1.88e-5, +1.13e-5 (none, 1.78); -2.06e-3, -2.23e-3, -1.43e-3 (none, 1.54) |
| same | same | mdotOut; mdotLat (net in) | same | +5.42e-1, +3.31e-1, +2.41e-1 (1.21, 1.11); +5.70e-1, +3.51e-1, +2.28e-1 (1.20, 1.49) of 8.6-9.3 |
| same | same | uLatMin; uLatMax | pointwise extremes | +6.6e-3, +8.7e-3, +2.48e-2 (grows); +2.56e-2, +2.80e-2, +2.87e-2 (3.5% of 0.80, flat) |
| same | same | burnRatio; massErr; yHits | rounding; the tally | 0.99995 to 0.99997 both, within 4e-6; FV max massErr 1.6e-9, 0 hits |
| FV(FP32) - structured(FP32), the rim anchor (gpu_lowmach_rim_anchor's case: rich phi 1.3 C2b, sigma 6.88, Sutherland 298 K, Pr 0.41, W 14, U 6, lip 1.4, open laterals at 6, T_w 1, tt 40); the tube's staircase = the FV mesh's `rim` patch (grid-aligned lips: the staircase IS the body-fitted wall) | 4, 6, 8 cells, window [30, 40] | anchored (exit 0) | both | both exit 0 at every rung (no flash-back, no blow-off) |
| same | same | xRimBase; rBase | the base ROW (a cell centre): equal or one row apart | 4: 4.12093 both, rBase 37.0997 vs 37.1050; 6: FV one row lower until t 39.5 (2.747 vs 3.532 = 3.5 h vs 4.5 h, rBase 39.35 vs 37.76 follows the row), then the same row (37.80 vs 37.82); 8: 3.23788 both (0.69 delta_th), rBase 38.598 vs 38.612 |
| same | same | qRim; preFrac; burnRatio | scheme gap: shrinks | 85.314 vs 85.801, 83.187 vs 83.487, 84.340 vs 84.528: -0.57%, -0.36%, -0.22% (1.19, 1.64); +6.9e-4, +4.3e-4, -8.2e-5 (1.16, sign change); +1.24e-3, +7.2e-4, +2.9e-4 (1.34, 3.14) |
| same | same | inner xTip; mdotLat (net in); Tmax | | -2.9e-2, -8.7e-3, -2.6e-2 of 244-248 (1e-4, no order); +2.01, +1.21, -0.45 of 340-404 (sign change); -5.1e-4, +1.7e-4, +9.0e-4. 0 hits both, FV max massErr 5.2e-8 |
| low-Mach FV(FP64) - compressible FV(FP64), M 0.01 | 7.5, 15, 30, same box | S_L | h part order 2, limit `+gamma M^2 (sigma - 1)` = +5.6e-4, the O(M^2) offset | 1.017309, 1.003584, 1.000469 vs low-Mach: -2.30e-2, -5.31e-3, -9.32e-4 (2.12, 2.51). Richardson on the last two at order 2: +5.3e-4 |
| same | same | delta_th | shrinks | 1.55562, 1.59063, 1.59976 vs low-Mach: +6.1e-2, +1.12e-2, +1.87e-3 |

The h part of that last row is a stated model and scheme gap: the compressible flame carries
central energy flux, the low-Mach one the limited T_f. It shrinks; its limit is the known O(M^2)
offset.

M3 off = bit for bit: `tools/fv/comp_bit_identity.sh` (pre-M3 binary vs this tree, 29 cases
with the low-Mach FV planar box FP32 / FP64, a sheared polyMesh, the structured low-Mach C0 and
C2 burner through `lmBunsenPoint`'s cell rule): 29 byte-identical, 0 failures (2026-11-03).
Steps 3-5 off = bit for bit: the same script, faad136 vs this tree, 32 byte-identical, 0
failures; and the steps 1-2 binary vs this tree on the low-Mach FV routes (C2, C2b gri30 FP64,
the G2a burner FP32 / FP64, the rich burner, the planar box, the sheared polyMesh): 7
byte-identical, 0 failures (2026-11-03). No open patch, g = 0, no rim: no new launch.

Not measured (M3, each past 295 s or past the 3-level budget): the C2b 1D 30-cell rung in FP64
(~800 s), the G2a burner at 8 cells in FP64, the wavy polyMesh burner at 4 and 8 cells.

Not measured (each past 295 s on the final dt rule):

    tools/chem/fv_flame_ladder.py --physics lowmach --cells 30 --tt 6 --tol 1e-3 --tags fv64 --meshes all --ny-mesh 16
    tools/chem/fv_flame_ladder.py --physics lowmach --cells 30 --tt 6 --tol 1e-3 --tags fv64 --meshes wavy --ny-mesh 16

**Gates.** `fv_lowmach` (host, `tests/test_fv_lowmach.cpp`): route, rejections, `--bc`, the first
table, `fvThetaVote` vs the structured stencil, `measureLowMach`. `gpu_fv_lowmach` (L1-L7,
`tests/test_gpu_fv_lowmach.cpp`, with R1-R3, R7, B1-B2 55 s), `gpu_fv_lowmach_momentum` (L8, 147 s),
`gpu_fv_lowmach1d` (the binary at 7.5 cells: S_L 0.994256, delta_th 1.61176, 0 hits at tt 12),
`gpu_fv_lowmach_bunsen` (the binary's G2a burner at 4 cells, 32 s: xTip 44.3072, burnRatio
0.999990, max |massErr| 1.2e-9, 0 hits), `gpu_fv_lowmach_burner` (O1-O4, G1-G3, R1-R2: "Open
faces, gravity, rim", 174 s). The host gate adds the `open` kind's parse and errors, the route's
new flags, and the box minus a solid strip (`rim` faces, S out of the fluid, `bOpen` 1 / 2).

| gate | bound |
|---|---|
| L2 mass, fuel, passive | FP64 1e-13, FP32 1e-7 |
| L3 energy rate | FP64 1e-13, FP32 1e-5 |
| L3 march order | >= 0.95. Set after the result: first written >= 1.8 on the expectation of the RK order, measured 1.00; see "Energy" |
| L6 freestream | FP64 1e-12, FP32 2e-4 (set before the result) |
| L7 | FV32 vs structured <= 3 x FV32 vs FV64 |
| L8 | order >= 1.5 per rung |
| flame1d | S_L within 2e-3 of the structured 7.5-cell value, 0 hits |
| R1 Z kept | FP64 1e-12, FP32 1e-6 (set before the result) |
| R2 mass, fuel, oxidizer; R3 energy rate | L2's, L3's |
| R7 | L7's |
| B1 inflow m vs row; B2 tallies | FP64 1e-14, FP32 1e-6; L2's |
| bunsen | xTip within 1e-2 of the structured 4-cell 44.304 (set after the result: +3.2e-3, a scheme gap, issues LF5), burnRatio within 1e-4 of 1, |massErr| <= 1e-7, 0 hits, exit 0 (anchored) |

## Rejected / not built

- `dns flow --solver fv` on a Cartesian box with an immersed body: no immersed
  wall in the FV path (`--ogrid` or `--polymesh`).
- Revolved O-grids (`--ogrid --body sphere`): axis prisms are not in FaceMesh.
- `--ramp`, `--steady`, `--seq`, `--rans`, frames: not flags of this path. MRF,
  sliding meshes, wall models: not built.
- Implicit viscosity: `dt ~ h²/ν` on wall-clustered meshes (open, issues.md 79).
- Restart: not built.
- `dns conv --solver fv`: `--hot-patch` (per-face wall T), inflow / outflow under
  buoyancy: not built.
- Low-Mach FV: not built, rejected by name: the 3D slot and the round nozzle, the
  burner's `--profile-dt`, `--turb` on a polyMesh or the burner (M5: the planar box only), a closed
  box (P0(t)), gravity on the planar flame, `--rim` on a polyMesh (its wall patch is the
  tube). Built in M3: two reactants, the 2D slot burner, open faces (`type = "open"`,
  `--laterals open`), gravity (the face force), the rim (the staircase as a Wall patch);
  in M4: LES and TFLES ("Low-Mach LES", refusals in "Low-Mach combinations").

See `issues.md` item 79.
