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
is p = 0, scalars zero gradient both ways; `wall` adiabatic or isothermal, species-tight,
no-slip; `slip`. `far` (no characteristic state), outflow `p`, `tback`, `burnt`, `y`, no
outflow at all (a closed box needs P0(t)): exit 3 by name, `--dry-run` included.
`tools/chem/fv_flame_mesh.py --lowmach` writes that file.

Rejected by name: `--mach`, `--compensated` (always on in FP32), `--equivalence`, `--mode`
other than premixed, `--gravity`, `--rim`, `--laterals`, every LES, TFLES and turbulence
flag, `--dt-rule`, checkpoints. Off the route `--tol`, `--bc`, `--fv-precision`.

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
| low-Mach FV(FP64) - compressible FV(FP64), M 0.01 | 7.5, 15, 30, same box | S_L | h part order 2, limit `+gamma M^2 (sigma - 1)` = +5.6e-4, the O(M^2) offset | 1.017309, 1.003584, 1.000469 vs low-Mach: -2.30e-2, -5.31e-3, -9.32e-4 (2.12, 2.51). Richardson on the last two at order 2: +5.3e-4 |
| same | same | delta_th | shrinks | 1.55562, 1.59063, 1.59976 vs low-Mach: +6.1e-2, +1.12e-2, +1.87e-3 |

The h part of that last row is a stated model and scheme gap: the compressible flame carries
central energy flux, the low-Mach one the limited T_f. It shrinks; its limit is the known O(M^2)
offset.

Not measured (each past 295 s on the final dt rule):

    tools/chem/fv_flame_ladder.py --physics lowmach --cells 30 --tt 6 --tol 1e-3 --tags fv64 --meshes all --ny-mesh 16
    tools/chem/fv_flame_ladder.py --physics lowmach --cells 30 --tt 6 --tol 1e-3 --tags fv64 --meshes wavy --ny-mesh 16

**Gates.** `fv_lowmach` (host, `tests/test_fv_lowmach.cpp`): route, rejections, `--bc`, the first
table, `fvThetaVote` vs the structured stencil, `measureLowMach`. `gpu_fv_lowmach` (L1-L7,
`tests/test_gpu_fv_lowmach.cpp`, 43 s), `gpu_fv_lowmach_momentum` (L8, 147 s), `gpu_fv_lowmach1d`
(the binary at 7.5 cells: S_L 0.994256, delta_th 1.61176, 0 hits at tt 12).

| gate | bound |
|---|---|
| L2 mass, fuel, passive | FP64 1e-13, FP32 1e-7 |
| L3 energy rate | FP64 1e-13, FP32 1e-5 |
| L3 march order | >= 0.95. Set after the result: first written >= 1.8 on the expectation of the RK order, measured 1.00; see "Energy" |
| L6 freestream | FP64 1e-12, FP32 2e-4 (set before the result) |
| L7 | FV32 vs structured <= 3 x FV32 vs FV64 |
| L8 | order >= 1.5 per rung |
| flame1d | S_L within 2e-3 of the structured 7.5-cell value, 0 hits |

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
- Low-Mach FV: not built, rejected by name: the Bunsen burner (per-face inflow rows,
  cone stats), open laterals, `--rim`, `--gravity`, `--equivalence` and two reactants,
  `--les`, `--tfles`, `--turb`, `--dt-rule cell`, checkpoint and restart, a closed box
  (P0(t)). The tables are per face (`bU`, `bT`, `bY`): a burner's rows need no new
  device path, the cost is its stats and its gates.

See `issues.md` item 79.
