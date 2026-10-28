# RANS — SA-neg and SST on the compressible channel and `dns flow`

A RANS field is an ensemble-mean MODEL output: never a grounder, never DNS.
Every artefact says which closure and which mode (`--dry-run` label, RunMeta
`"rans"`, the stats header, the progress line). `dns/rans_label` is the ONE
producer of every label string (`ransRunLabel`, `ransModelLabel`):

| label | closure |
|---|---|
| `RANS SA-neg` | TMR SA-neg: standard SA **with ft2** for ν̃ ≥ 0, the negative branch below |
| `RANS SST-2003m` | Menter SST, 2003 constants, μt with S, ⅔ρk dropped |
| `RANS SST-Vm` | Menter 1994 SST with vorticity, ⅔ρk dropped |

Mode suffix: `-- steady (local time stepping: pseudo-time, not time-accurate)`
with `--steady` (explicit scheme), `-- steady (implicit Newton pseudo-time:
not time-accurate)` with `--steady --steady-scheme implicit` on the channel,
`-- steady (implicit Newton-Krylov pseudo-time: not time-accurate)` with it on
`dns flow`, `-- URANS (time-accurate)` without. A RANS 1D/2D run never
carries the DNS label's "2D: qualitative only" suffix — the mean really is
1D. Geometry suffixes: `[axisymmetric wedge DEG deg: ensemble mean, no
swirl, no 3D wake modes]` on the revolved wedge (the full revolution
carries none: it resolves swirl and 3D wake modes), `[immersed ghost wall:
first-order wall position, NOT body-fitted]` without `--ogrid`.

Built: `dns channel --rans none|sa-neg|sst-2003m|sst-vm`, plus
`--steady TOL` / `--steady-steps N` / `--steady-scheme explicit|implicit`;
`dns flow --physics compressible --rans ...` on the body-fitted O-grid
(2D/z-extruded, the revolved sphere wedge and its full revolution `--nz N`)
and on the immersed ghost wall (uniform or `--focal`), URANS and the explicit
local-time-stepping steady march (stopped on external flows by the
jitter-floored rule), plus the laminar compressible `--steady` on `--ogrid`,
and the flow Newton-Krylov march (`--steady-scheme implicit`, O-grid at `--nz
1` or the wedge), which certifies the same plateaus on the cylinder but is
slower than explicit everywhere it converges. A laminar steady run is
labelled `DNS -- steady (local time stepping: pseudo-time, not
time-accurate)`. Not built: Newton-Krylov on the full revolution, dual-time
URANS, wall models, the incompressible branch, MUSCL scalar convection.
`--rans` is registered only where it is built (`ransBuilt(Subcommand)`),
only with the closures that exist.

Sources: TMR `spalart.html`, `sst.html`; MKM chan180 / chan395 `.means`
(Re_τ 178.12 / 392.24; ub⁺ by trapezoid over the published profile: 15.679 /
17.545 — the headers carry no Re_b).

## Equations as implemented

### SA-neg (`dns/rans.hpp`, `namespace sa`)

Conservative variable ρν̃. TMR conservative compressible form

    ∂t(ρν̃) + ∇·(ρuν̃) = ρ cb1(1−ft2) S̃ ν̃ − ρ(cw1 fw − cb1 ft2/κ²)(ν̃/d)²
        + (1/σ)[∇·(ρ K ∇ν̃) + ρ cb2 |∇ν̃|²] − (1/σ) K ∇ρ·∇ν̃,   K = ν + ν̃ fn

fn = 1 for ν̃ ≥ 0 (so K = ν + ν̃, TMR's positive form). For ν̃ < 0: production
cb1(1−ct3)Ων̃, destruction **+**cw1(ν̃/d)², fn = (cn1+χ³)/(cn1−χ³), μt = 0. S̃
carries TMR note 1(c)'s limiter (c2 0.7, c3 0.9); r = min(ν̃/(S̃κ²d²), 10),
10 at S̃ = 0. Ω = |curl u| from the cell-centre gradient. ν̃ is NEVER clipped.

The ∇ρ term's coefficient for ν̃ < 0 is **unverified**: TMR gives no
conservative form for the negative branch. Implemented as K = ν + ν̃fn — the
coefficient that moving ρ into the divergence produces for the SAME diffusion
operator, so the form stays a consistent rewrite on both branches.

### SST (`namespace sst`)

ρk, ρω; F1 = tanh(arg1⁴) (arg1 clamped at 10 before the power: exact),
F2 = tanh(arg2²), CDkω = max(2ρσω2∇k·∇ω/ω, floor), blended σk, σω, β, γ.

| | SST-2003m | SST-Vm |
|---|---|---|
| μt | ρa1k/max(a1ω, S F2) | ρa1k/max(a1ω, Ω F2) |
| P | μt S² | μt Ω² |
| k-eq limiter | min(P, 10β*ρωk) | min(P, 20β*ρωk) |
| ω production | γ/νt·P̃ (limited) | γ/νt·P = γρΩ² |
| γ1, γ2 | 5/9, 0.44 | β_i/β* − σωiκ²/√β* (0.5532, 0.4404) |
| CDkω floor | 1e-10 | 1e-20 |

The ω equation carries +2(1−F1)ρσω2∇k·∇ω/ω. The "m": ⅔ρk is dropped from
τ_ij in momentum and energy (TMR), which is what lets μt ride the existing
eddy-viscosity slot of `faceFluxCurv` with `kPrim`/`faceFlux` untouched.
k ≥ 0 and ω ≥ floor·ρ are a **counted** guard (stats `floorHits`); every
pinned test requires zero hits.

### SST audit: TMR vs dns vs OpenFOAM v2606 `kOmegaSST`

OpenFOAM from
`src/TurbulenceModels/turbulenceModels/Base/kOmegaSST/kOmegaSSTBase.C`. The
pointwise dns algebra is pinned against an independent FP64 restatement of
TMR (test `rans`, `sstRestatement`: 4e4 random states per closure plus
states placed ON every switch -- F1 ≈ 0.42, P/limiter ∈ [½, 2], CDkω within
2× of its floor of either sign, a1ω/(R F2) ∈ [½, 2]): FP64 worst 7e-16
(F1, F2, μt, both sources), FP32 within 1e-4 of the term scale.

| term | TMR SST-2003m | dns `sst-2003m` | OpenFOAM `kOmegaSST` (v2606 defaults) |
|---|---|---|---|
| F1 | tanh(arg1⁴), arg1 = min(max(√k/(β*ωd), 500ν/(d²ω)), 4ρσω2k/(CDkω d²)) | same; arg1 clamped at 10 (exact) | same; clamp 10; CDkω+ per unit ρ (incompressible) |
| CDkω | max(2ρσω2∇k·∇ω/ω, 1e-10) | same (Vm: 1e-20, TMR's 1994 value) | same, 1e-10 |
| F2 | tanh(arg2²), arg2 = max(2√k/(β*ωd), 500ν/(d²ω)) | same, clamp 10 | same, clamp 100 (equivalent) |
| F3 | -- | -- | `F3 false` (default; matches) |
| μt | ρa1k/max(a1ω, S F2), S = √(2SᵢⱼSᵢⱼ) | same | a1k/max(a1ω, b1F2√S2), b1 = 1 (same) |
| P (k eq.) | min(P, 10β*ρωk); "m": P ≈ μt S² | min(μt S², 10β*ρωk) | min(νt ∇U:dev(2S), c1β*kω), c1 = 10 = νt(S² − ⅔(∇·u)²), plus −⅔k∇·u (SuSp); ∇·u ≈ 0 in simpleFoam |
| ω production | (γ/νt) P̃ (TMR: the paper's αρS² was a typo) | γρ min(S², lim/μt) ≡ (γ/νt)P̃ | γ min(∇U:dev(2S), (c1/a1)β*ω max(a1ω, b1F2√S2)) ≡ (γ/νt)P̃, plus −⅔γω∇·u |
| γ1, γ2 | 5/9, 0.44 | same | same |
| β1, β2, β*, σk1, σk2, σω1, σω2, a1 | 0.075, 0.0828, 0.09, 0.85, 1, 0.5, 0.856, 0.31 | same | same |
| cross diffusion | 2(1−F1)ρσω2∇k·∇ω/ω | same, unfloored; its negative part point-implicit | same, unfloored; SuSp (implicit when negative) |
| ⅔ρk in τᵢⱼ | dropped ("m") | dropped | absorbed in the kinematic pressure (incompressible) |
| wall ω | 60ν/(β1 d1²) | face Dirichlet image per wall face, d1 = first cell-centre distance | parity case: `fixedValue` per face from dns's own d1 (NOT `omegaWallFunction`) |
| wall k, μt | 0 | antimirror (face 0) / odd (face 0.0) | `fixedValue 0` / `fixedValue 0` |
| freestream | U∞/L < ω < 10U∞/L, μt∞/μ 1e-5..1e-2 | k∞ 2.25e-7 U², μt∞/μ 0.009 (ω∞ 0.25 U/D at Re 1e4) | `inletOutlet` at the same values |
| decay control | only in SST-sust | none | `decayControl false` (default; matches) |
| positivity | -- | counted guard (0 hits in every pinned run) | `bound(k, kMin)`, `bound(ω, ωMin)`: SILENT clipping (central-U runs print `bounding k, min: 0` every iteration) |
| Pr_t | 0.9 (air) | 0.9 | `Prt 0.9` set in the parity case (default 1.0) |
| scalar convection | -- | first-order upwind | `bounded Gauss upwind` (matches) |
| mean-flow convection | -- | KG central, no added dissipation | parity reference: central `linear` |
| wall distance | exact | exact to the wall polyline at the node-mean centroid | `exactDistance` |

SST-Vm vs TMR SST-V / SSTm: μt with ΩF2, P = μt Ω², k limiter 20β*ρωk
only, ω production γρΩ² unlimited, γᵢ derived (0.5532, 0.4404), CDkω floor
1e-20 -- all as TMR; OpenFOAM has no Vm switch (its G and μt are strain-based
by construction), so Vm has no matched OpenFOAM reference.

**No model mismatch.** The one material difference a parity case can
introduce is OpenFOAM's U scheme: `linearUpwind` adds numerical dissipation
dns does not have, and moves OpenFOAM's own SST cd +3.7% (L1 Re 1e3, 0.47129
→ 0.48872), +2.8% (L2 Re 1e4, 0.39496 → 0.40589), +7.6% (L1 Re 1e4, 0.39964 →
0.43000) and L_r −28% / −14% / −28% when U goes central; SA L2 Re 1e4 +1.5%
(0.41285 → 0.41911). The parity reference is central U (docs/validation.md).

## Discretization (`gpu/rans.{hpp,hip}`, oracle `dns/rans_channel1d`)

- **State** (D1): the scalars live in `DeviceCompState::t[2]` (`nT` 0/1/2) —
  one owner, so the stage choreography (output never aliases `cmb.a`, handle
  swap) covers them by construction.
- **Mean flow** (D2): μt enters through the LES `nut` slot (μ += μt_f,
  κ += μt_f cp/Pr_t, Pr_t 0.9). The only hot-kernel edit is `cmb.cr` →
  `cmb.crAt(c)`. ν̃ ≡ 0 (SA) or k ≡ 0 (SST) is bit-identical to RANS off
  (`muf + 0.0f == muf`), test-pinned.
- **Convection** (D3): first-order upwind of the specific scalar on the mean
  mass flux `rm·Ua`, the VERBATIM `faceFluxCurv` expression (MRF included).
  A uniform specific scalar 2^-13 stays EXACTLY uniform (0 ulp, `gpu_rans`
  G3): any drift between the two copies of that expression would break it.
- **Face coefficients** (D14): formed ON the face from face-averaged
  primitives — SA ρ_f(ν_f + ν̃_f fn(χ_f)); SST μ_f + σ(F1_f) μt_f with
  F1_f, μt_f face averages. Never the average of per-cell coefficients.
- **Cell terms**: sources plus SA's cb2 and ∇ρ terms from cell-centre
  gradients (`curvScalarGradCentre`, the scalar twin of the SGS gradient).
- **Point-implicit** (D13): `out = (comb + crAt(R + J q))/(1 + crAt J)`,
  J ≥ 0 the frozen-coefficient destruction rate (SA: 2(cw1fw − cb1ft2/κ²)ν̃/d²,
  or the negative branch's; SST: β*ω, 2βω + the negative cross-diffusion).
  At J = 0 it is `comb + cr R`, the mean flow's own combination. It is FIRST
  order in dt for the stiff sources even inside SSP-RK3 (`gpu_sst` (a): the
  closed-form freestream decay converges at order 1.00 (ω) / 0.91–0.98 (k));
  steady fixed points are unaffected (issues.md 9).
- **Walls** (D15, D16): μt is computed everywhere, then written ODD across the
  ghost rows (`kMutMirrorY`) — the wall-face average is exactly 0.0, so the
  wall stress the u_τ fit and the tally read stays molecular. Never computed
  from ghost scalars. SST's F1 mirrors EVEN. Ghost scalars mirror the
  SPECIFIC value, rebuilt with the ghost's own density (the isothermal ghost
  has ρ_g ≠ ρ_1): ν̃_g = −ν̃_1, k_g = −k_1, and ω a Dirichlet image
  ω_g = 2ω_w − ω_1 with TMR's ω_w = 60ν_w/(β1 d1²), ν_w on the wall-face
  density, d1 = the first cell CENTRE's distance (TMR does not define d1 for a
  cell-centred code — **unverified**; see the measured grid order below).
- **Wall distance** (D18): `dns/walldist` — the channel's exact plate
  distance at the node-mean centroid; ghost rows take their mirrored row's d.
- **Stage cell cache**: the μt kernel (`kPrimSaMutCurv`, `kSstMutCurv`)
  writes `rans::RansCellCache` (the specific scalars φ = t/ρ, SST also R and
  ∇k·∇ω of `sstCellGrads`); the scalar stage reads it instead of dividing t/ρ
  at every face and neighbour or recomputing the gradients. Same expressions,
  so bit-identical. `DeviceComp::stage` calls `primitivesAndMut(s)` itself
  right before the stage and asserts that the cache is of `s`. The SST stage
  carries `__launch_bounds__(blockSize)`: without it the 1024-thread default
  capped it at 192 VGPRs and spilled 9 (with it: 202, no spill; SA 117).

### Launch structure (small 2D grids)

The small 2D grids are launch- and latency-bound (an empty launch costs
2.7–2.8 µs). Four bit-identical structural rules:

- **Real-plane stage on the wedge.** `DeviceComp::setStageRange` restricts
  `kCompStageCurv` and the scalar stage to a contiguous cell run; the O-grid
  runner sets plane k = 1 on the wedge. Legal because planes 0 and 2 are
  `kWedgePlanes`' rotated copies of plane 1, every field and low word
  rewritten by the stage hook before any read: the hook's row passes read a
  ghost plane but their writes there are overwritten, the residual norms and
  the NK kernels are fluid-masked, and every download comes after the hook.
  Primitives, μt, hooks, `localDt` and `stableDt` stay whole-grid.
- **No redundant entry hook.** `setStageHook(h, Idempotent::yes)`: `step()`
  skips re-applying the hook to its input. Legal only when the hooked state
  depends on no cell the hook writes AND the caller hands `step()` only
  hooked states. The O-grid (rows + axis) and channel runners qualify; the
  immersed path does not claim it (issues.md 26).
- **Fused boundary rows.** `kOGridRows` (wall, far, turbulence, ghost-lo) and
  `kChannelRows` (wall, ghost-lo, turbulence). Legal because each thread
  reads only real cells (rows 1 and ny−2, which no pass writes) and the ghost
  cells its own earlier pass wrote. The axis columns and planes stay two
  launches: the planes read the columns another thread wrote, and the axis
  μt copies read the row images likewise.
- **SA primitives + μt fused** (`kPrimSaMutCurv`): pointwise, each cell reads
  only its own T. SST keeps two launches (its μt reads a stencil of
  primitives).

Measured (gfx1100, `--steady`, median of 3 interleaved `msPerStep` runs):
launches per iteration wedge SA 27.6, O-grid 256×96 SA 15.5, channel 4×66 SA
18.4; ms/iteration:

| case | ms/it |
|---|---|
| channel 4×66 SA / SST | 0.121 / 0.124 |
| channel 64³ SA / SST | 0.498 / 0.599 |
| O-grid 256×96 SA / SST | 0.149 / 0.177 |
| wedge 130×98×3 SA | 0.192 |
| full revolution 66×50×16 SA | 0.299 |
| immersed 384² SA / SST | 0.640 / 0.750 |

What remains on the small SA grids is per-face work (the ν_f, χ_f divisions)
and the pointwise source, neither a per-cell value. `gpu_axisym` (h) pins the
wedge pieces directly.

## Pseudo-time (`--steady`)

Two schemes (`--steady-scheme`): `explicit` (the default; this section and
the two after it) and `implicit` (the Newton line march, "Implicit
pseudo-time" below; ~27 iterations instead of ~3e5).

- **LTS** (D8, D9): `StageCombine::lts` is a per-cell RATIO dt_c/dtRef ≥ 1 on
  top of `cr`; `crAt(c)` is the ONLY way any kernel reads `cr` (stage
  kernels, body force, gravity, sink, NSCBC, the scalar kernels). Additive
  hooks take the whole `StageCombine`, so ignoring lts is a compile error —
  ignoring it converges to lts·R + f = 0, the wrong steady state (G7 pins the
  right one with random ratios in [1, 10]). dt_c = min(cfl/wave_c, the
  per-cell diffusive bound); dtRef = min dt_c; recomputed on `--dt-recompute`.
- **RK3 coefficient sum**: every stepper forms ca = 1 − cb, an exact sum in
  FP32, so a steady state is a fixed point of the update.
- **Residual** (D10): one extra stage per stats row with ca = cb = 0, cr = 1,
  no lts, boundary hooks included, the scalar kernel in explicit mode — the
  pure rhs, never state differencing.
- **Norms** (D11), wall units L = δ = 1, U = u_τ = 1, Ue = ub estimate:
  resU = rms(|R_m|/ρ), resE = rms(R_E/ρ)/Ue, resT = SA rms(R_ν̃/ρ), SST
  max(rms(R_k/ρ), rms(R_ω/(ρω))) (ω spans six decades: its rate of ln ω).
  `resid = max(resU, resE, resT)`, reported in the stats, NOT the stop
  (below). `--steady-steps` (1000000 explicit, 500 implicit) is exit 8.

### The FP32 update floor — why the residual cannot certify

`gpu_rans` G5/G9, Re_τ 180, ny 64, M 0.2: the residual of the FP64 oracle's
discrete solution, uploaded to the FP32 device, is 0.025 (static FP32 floor
0.042: the spread over ±1-ulp roundings of the same state). The march does
NOT reach it. Once |R_c|·dt_c < ulp(q_c)/2 the FP32 state stops changing, so
the march stalls on a DYNAMIC floor ulp(q)/(2dt_c): resid 1–2, dominated by
resE (E ≈ p/(γ−1) ≈ 1.1e4 at M 0.2, ulp 1e-3, dt 2.4e-5 in the wall cell) —
reached within ~1e3 iterations even when started AT the oracle. And the slow
mode (ub) carries a residual below that floor long before it has converged (a
cold start sits at resid ~1.3 with ub 1.6% low): no TOL separates converged
from unconverged, hence the integral stop below. The solution itself DOES
converge onto the oracle: `gpu_rans` G9 pins the PLAIN march on ny 32,
stretch 2.6 (ub-mode e-folding ~2.8e4 iterations): −0.053% after 1.2e5
iterations (15 s), residual 2.7 against the oracle state's 0.05.

**The compensated state.** `--steady` marches a Kahan pair per field
(`DeviceCompState::lo`, `DeviceComp::enableCompensation`): the kernels read
hi, every stage write (`gpu/internal/stage_write.hpp`) accumulates into hi +
lo in the increment form a + cb (b − a) + cr R, exact because ca == 1 − cb.
Fluxes stay FP32. Plain/compensated is a COMPILE-TIME kernel switch (a
runtime branch at the write moved the NSCBC rows' FMA contraction); the plain
instantiation is bit-for-bit the pre-compensation solver. The channel wall
zeroes its ghost cells' low words; boostX has a pair kernel (`kBoostXPair`);
penalizeMomentum / channelSetRows assert a plain state, which the validator
makes unreachable. Measured (gfx1100):

| | plain | compensated |
|---|---|---|
| u drift, 2e4 LTS steps from the oracle (G10) | 6.3e-5 | 3.2e-8 |
| resid plateau from the oracle (G10, max of tail) | 2.03 | 0.56 |
| ub after 2.2e5 it from the perturbed state | −0.160% | −0.104% |
| cold start 4e5 it, resid tail median (SA / 2003m / Vm) | 1.27 / 1.25 / 1.27 | 0.35 / 0.36 / 0.36 |
| cold start final ub vs oracle | +0.057 / +0.019 / −0.003% | +0.032 / +0.030 / +0.030% |
| ub drift over the last 1e5 it | +0.027 / +0.022 / +0.015% | +0.009 / +0.008 / +0.007% |
| ms/iteration 4×66 (SA / SST) | 0.133 / 0.146 | 0.159 / 0.179 |
| ms/iteration 64³ | 0.531 / 0.738 | 0.600 / 0.849 (1.13× / 1.15×) |

The plain march's ub never stops creeping; the compensated one decays onto a
fixed point, but its plateau is still ~10× above the static floor (0.042), so
the residual still certifies nothing.

**The cold starts' +0.03% above the oracle is MASS DRIFT.** Local time
stepping conserves Σ V ρ / lts, not Σ V ρ, while v ≠ 0 in the transient, and
the steady equations are a one-parameter family in total mass (Σ V R_ρ ≡ 0:
the wall faces carry no mass flux), so the march lands on the member whose
mass it drifted to. `--n 4 --nz 1 --ny 64 --rans sa-neg --steady 1e-4`
(305800 iterations): final fluid mass 1.998440, −7.80e-4 against the IC's 2;
the oracle solved AT that mass gives ub 15.83529 against the march's 15.8338
(−9e-5, inside TOL), while against mass 2 the march is +2.4e-4. Sensitivity
dub/ub per dM/M = −0.430. The implicit scheme pins the mass (below); a mass
projection on the explicit march is a pending decision, not built.

### Time accuracy: the same floor on a time march (`--compensated`)

On a TIME march the per-step update of a cell is |R| dt, and once |R| dt <
ulp(q)/2 it rounds away: the plain FP32 state loses every rate below
**ulp(q)/(2 dt)**, a floor that RISES as dt → 0. A dt ladder therefore stops
converging and then diverges. `--compensated` (`dns flow --physics
compressible`, Cartesian / focal / O-grid / MRF / rotor, and `dns channel`)
puts the time march on the compensated state; off by default, so every
default run is bit-for-bit what it was (issues.md 20). It rejects
`--steady-scheme implicit` (the Newton march keeps its own FP64 iterate) and
`--penalize` (a plain-state edit); on an explicit `--steady` it is a no-op.
The resolved case prints `[compensated state: FP32 hi + lo per field]` and the
run's .json carries `"state":"compensated"`.

Measured: `flow --physics compressible --body cylinder --ogrid --n 256 --ny 96
--ogrid-beta 4 --ogrid-far 20 --re 1e4 --rans sst-2003m --ramp 1 --ft 0.04`
(tEnd 2.26, impulsive-start transient, still symmetric), cfl 0.4 / 0.2 / 0.1
/ 0.05, each run interpolated onto 2000 points, RMS over t ∈ [0.05, 0.6] of
the difference between adjacent rungs:

| | d(0.4−0.2) | d(0.2−0.1) | d(0.1−0.05) | ratios |
|---|---|---|---|---|
| cd plain | 6.49e-4 | 2.96e-3 | 1.03e-2 | 0.22 / 0.29 |
| cd compensated | 2.91e-4 | 7.77e-5 | 2.35e-5 | 3.75 / 3.30 |
| cdf plain | 2.77e-5 | 3.47e-5 | 1.16e-4 | 0.80 / 0.30 |
| cdf compensated | 2.44e-5 | 1.21e-5 | 6.11e-6 | 2.01 / 1.99 |

The plain ladder DIVERGES; the compensated one converges, cd at observed
order ~1.8 and cdf at order 1.0 (the first-order source quotient). Final cd
at t 2.26, cfl 0.4 → 0.05: plain 1.7261 / 1.7295 / 1.7421 / 1.7288,
compensated 1.7255 / 1.7255 / 1.7254 / 1.7254. The windows [0.6, 1.3] and
[1.3, 2.2] read the same. The compensated run keeps the symmetric start
symmetric (max |cl| 1.9e-6 → 6.5e-7 over cfl 0.4 → 0.05, against the plain
run's 1.0e-3 → 4.7e-3: the plain rounding is the asymmetric seed). Cost
1.16–1.21× per step on this case, 1.07–1.3× on short Cartesian / channel /
LES / rotor smoke runs. `comp_time_order` pins it on the isentropic vortex:
order 2.85 / 3.00 like the plain run, plain vs compensated at dt/4 2e-2 of the
time error, and the FP32 rounding vs the FP64 host 2.8e-7 against the plain
run's 1.5e-6.

Rule: a plain FP32 URANS (or DNS) dt study is not a convergence study once
|R| dt nears ulp(q)/2 in the cells that matter -- at M 0.2 the energy is
E ≈ p/(γ−1) ≈ 45 in box units, so a rate below ~2e-6/dt is lost. Use
`--compensated` for any dt-ladder or time-convergence claim.

### Reporting a URANS result

A URANS number is a WINDOW MEAN ± SE, never an instant or an unbounded
average. Protocol (`tools/parity/ocyl1e4.py` `stats()`, the same code reads
dns and OpenFOAM): times in D/U; periods between upward cl crossings (0.2 rms
dead band, as `tests/validation_util.hpp` `strouhal`); the window starts at
the first period after which the per-period cl swing changes < 2% between
consecutive periods for good, and needs ≥ 4 periods, else the run is
reported "not saturated" and nothing is averaged; cd, cl_rms, swing and the
period are batch means over whole periods, SE = std/√n (St = 1/T̄, SE_St =
SE_T/T̄²). The SE of a few correlated periods is an estimate; quote the window
and n with it. ≥ 4 periods is the rule's floor, not enough to cite: on the 2D
O-grid cylinder SST Re 1e4 (256×96, β 4, 20 D) a 27.7–62.3 D/U window still
carried the start-up overshoot (cl_rms 1.2164) against the saturated 75–128
D/U window's 1.2061 ± 0.0006 (12 periods). Cite ~10 saturated periods
(docs/validation.md "O-grid cylinder URANS shedding vs OpenFOAM").

**What is first order in time** (the interior is SSP-RK3; a URANS dt claim
must name these):
- the turbulence sources: the point-implicit stage quotient (D13);
- the O-grid boundary rows: wall and far row are state overwrites once per
  stage, not ridden through the RK combination (`gpu/ogrid_bc.hpp`
  `DeviceOGridBc::apply`);
- the quiet-start ramp: the Galilean boost (`boostX`) and the far-row
  retarget are applied AFTER each step, a Lie splitting of the ramp
  forcing -- first order over the ramp only (t ≤ `--ramp`).
Observed: the cd ladder above converges at ~1.8, and from cfl 0.4 to 0.8 the
saturated shedding statistics move ≤ 0.2%.

### Stopping on the integrals (the channel)

`--steady TOL` stops when every monitored integral is within TOL, RELATIVE,
of its extrapolated fixed point: ub, bulk T (not T − Tw: the 0.6% excess is a
difference of FP32-resolved numbers, relative noise 5e-6 against 3e-8), and
∫ρν̃ dy | ∫ρk dy, ∫ρω dy. `dns/steady_stop`, test `steady_stop`. Per series,
on the regular stats rows (every 200 iterations; the cap row is off the
cadence and not sampled):

- window = the second half of the history, four samples at spacing n/6;
- every row in the window monotone to kSteadyFlat = 2^-20 (else +inf): four
  samples alone alias an oscillation into a contracting sequence — without
  this an undamped 1e-4 wave certified at TOL 1e-5 and a damped one stopped
  8.7× early;
- both difference ratios in [0, 1) → remaining = |d3| r/(1 − r), r the larger
  (exact for one geometric mode); otherwise +inf, except a window flat to
  min(TOL/4, 2^-20) (converged to sampling noise) → max|d|.

TOL ∈ [1e-6, 1) (validator): below 1e-6 the sampling noise decides. ONE
series is not enough: approaching a turning point looks geometric, and the
channel's ub dips first (fast mode) and then rises (slow mode) — ub alone
certifies TOL 1e-3 at 3400 iterations, 2.8% low. The companions are not at
their turning point then, which holds the stop off; a transient that turns
every one of them at once would still fool it.

Measured (gfx1100, Re_τ 180, ny 64, M 0.2, cold start; error = vs the march's
own fixed point extrapolated from 6e5 iterations):

| TOL | SA-neg | SST-2003m | SST-Vm |
|---|---|---|---|
| 1e-3 | 177800 it, ub −9.9e-4 | 210200, −4.9e-4 | 206200, −4.6e-4 |
| 1e-4 | 305800 it, ub −9.6e-5 | 326600, −5.8e-5 | 318400, −5.9e-5 |
| 1e-5 (offline) | 433000 | 453000 | 443200 |

Every stop is exit 0, ~0.16–0.18 ms/iteration (49 s at TOL 1e-4 SA). The
slow mode's e-fold is ~5.6e4 iterations at Re_τ 180 and ~1.1–1.5e5 at Re_τ
395 (ny 96), where bulk T is the slowest series: TOL 1e-4 needs ~5e5
iterations and 1e-5 is past 6e5. The stop certifies the MARCH's fixed point;
its +0.03% offset from the oracle is the mass drift above, not the stop.

## Implicit pseudo-time (`--steady-scheme implicit`, the channel)

`dns/channel_line`, `dns/blockband`, `dns/pseudo_time`, and the implicit
branch of `gpu/runner_channel.hip`. Backward-Euler pseudo-time with defect
correction:

    (diag(1/Δτ_j) − J) Δq = R_gpu(q),   q ← P_M0(q + α Δq)

- **The right-hand side is always the GPU's**: the residual-mode stage
  (`DeviceComp::residual`, hooks in), gathered from column (0, ·, 0)
  (`channelGatherColumn`). J only preconditions: a drift between the line
  operator and the GPU costs iterations, never the answer. `gpu_rans_implicit`
  I1/I2 pin the drift.
- **One column, on the host, broadcast.** The RANS mean is x/z-homogeneous
  by construction (homogeneous IC, `--kick` rejected), so the host holds ONE
  FP64 column and the device holds its FP32 rounding in every column
  (`channelSetRows`, then the wall). The device state is column-uniform bit
  for bit at every iteration (I4), the y-line Jacobian is exact for it, and
  the iterate on nx 6 × nz 2 equals nx 4 × nz 1 bit for bit (I6). Exact ONLY
  for this channel: a non-homogeneous RANS case needs a new design
  (line-Jacobi plus off-line coupling).
- **Unknowns** B = 4 + nT per row: (ρ, ρu, ρv, E, ρν̃ | ρk, ρω). At w = 0,
  R_mz is linear in w and every other row even in it, so ρw decouples
  exactly and stays exactly 0 (I4).
- **Jacobian**: (2K+1)-colour CENTRAL FD in FP64 of the host line operator,
  step 1e-7 × max(|q|, floor) (ρ and E their own magnitude, the rest
  max(|q|, 1e-3 ρ)), the wall reapplied after each perturbation so the ghost
  chain rule is in. `dns/channel_line` restates the GPU's y-line in conserved
  variables — KG + face-viscous flux with μt, scalar upwind on the verbatim
  mass flux, face diffusion with face-formed coefficients, cell terms,
  isothermal and scalar ghosts, μt odd / F1 even, the body force — and CALLS
  the closure algebra of `dns/rans.hpp` / `dns/sgs.hpp`. K = 1 for SA (its
  operator is genuinely tridiagonal), K = 2 for SST (μt and F1 carry
  cell-centre gradients); the band is pinned EXACT (`channel_line` (b)).
  The line's residual vanishes at the oracle's solution to 2e-13 (momentum),
  5e-10 / 1e-9 (energy, SA / SST; at Re_τ 1e4 its FP64 rounding floor, 1.2e-6,
  is above that) and 1.7e-15 relative (SST ω).
  CENTRAL, not forward: a forward step's per-entry error reached 1.6e-5 near
  the centreline (Ω = |du/dy| → 0 is a curvature spike in the SA source and
  the SST rate) and is one-sided across the upwind kink max(ṁ, 0), on which
  every face sits at v = 0; J·δ vs a central FD of the line is 8e-5 (SA) /
  2.2e-4 (SST) forward, 1.5e-7 central, and forward-FD pure Newton stalled on
  SST Re_τ 1e4 ny 256 (ub +19.9% after 100 iterations) where central
  certifies in 9. Not by FP32 FD of the GPU residual either: its working step
  is a non-monotone, case-dependent window (2^-10 never certifies SST Re_τ
  1e4 ny 128, 2^-12 fails ny 256, 2^-14 passes). SST needs K = 2: K = 1
  (frozen R, F2, F1) converges with a 0.70-per-iteration tail and stops at
  −4.4e-5 (Re_τ 180); a truncated exact K = 1 diverges at CFL ≥ 300.
- **Solve**: block LU without inter-block pivoting, partial pivoting inside
  each diagonal block (`dns/blockband`), FP64 on the host. Host cost per
  Jacobian + solve: SA 0.26 / 0.52 / 1.04 ms, SST 0.35 / 0.70 / 1.40 ms at
  ny 128 / 256 / 512 — no threading needed.
- **Mass**: every accepted update is scaled to the IC's total mass M0 =
  Σ ρ vol/S (= 2, the oracle's `totalMass`); specific quantities move by 2
  ulp. This selects the oracle's member of the mass family.
- **Ramp, limiter, rejection** (`dns/pseudo_time.hpp`, the one home of the
  constants): Δτ_j = CFL / ((|u| + c) max_a aov_a) — a continuation scale,
  not a stability bound; CFL 100, ×2 per accepted undamped step, cap 1e8; the
  step LENGTH α (whole update scaled, never clipped per cell) keeps ρ and E
  within 20%, SST ρω above −50% and ρk above −90%; α < 1 holds the CFL. A
  non-finite residual or an inadmissible state restores the last accepted
  state and cuts the CFL ×0.1; below 1 it is exit 7. A singular block pivot
  is exit 7.
- **Stop**: `SteadyStop` unchanged, one sample per iteration, but ARMED:
  samples enter the history only after an undamped step taken AT the cap;
  any damped step, rejection or sub-cap step clears it. Unarmed it certified
  ub 1.1% off after rejections, and a damped stall (host test (e): α forced
  to 0.5 at the cap certified at iteration 15 with ub +6.8e-5; armed, never).
- **Surface**: `--steady-steps` defaults to 500 (explicit keeps 1000000);
  `--cfl` and `--dt-recompute` are rejected. Stats: the explicit columns plus
  `cfl, alpha, rejects` (of the step that produced the row's state; 0 on the
  first row), one row per iteration; `t`/`dt` are pseudo-time (the smallest
  Δτ). RunMeta json gains `"steady":"implicit"`. The device state is plain
  (no Kahan words: the host holds the iterate). The progress line prints ub
  to 7 decimals (the stats CSV carries 6 significant digits) and the run ends
  with the host / GPU time split.

| constant | value | bracket |
|---|---|---|
| CFL₀ | 100 | host line with the central J: 27–29 iterations from 100, 7–11 from 1e8 (36 cases) |
| growth | ×2 | ×1.5 needed 23–28 iterations to 1e-4 vs 15–18 at ×2 |
| cap | 1e8 | a 1e5 cap leaves a slow tail (SA Re_τ 1e4 +5.9e-5 / +7.7e-5) |
| CFL min | 1 | rejection never fired in any measured case |
| step bounds | ρ, E 20%; ρω −50%; ρk −90% | fired only at iteration 0 on SST |
| cap (iterations) | 500 | measured 27–28 |
| FD step | 1e-7, central | above |

**Measured** (gfx1100, M 0.2, TOL 1e-4, `--n 4 --nz 1`; oracle
`solveChannel1d` tol 1e-8 in process; wall = process start to exit, of which
~0.05 s is start-up; host = Jacobian, solve, update; GPU = upload, wall,
residual, gather, stats):

| case | iterations | ub | oracle | error | wall | host / GPU |
|---|---|---|---|---|---|---|
| Re_τ 180, ny 64, β 2, RANS SA-neg | 27 | 15.8299691 | 15.8299695 | −2.7e-8 | 0.10 s | 3 / 37 ms |
| — RANS SST-2003m | 28 | 15.6677693 | 15.6677686 | +4.5e-8 | 0.10 s | 6 / 40 ms |
| — RANS SST-Vm | 28 | 15.6330125 | 15.6330113 | +7.9e-8 | 0.10 s | 5 / 40 ms |
| Re_τ 1e4, ny 128, β 4, RANS SA-neg | 27 | 25.3453147 | 25.3453151 | −1.6e-8 | 0.10 s | 8 / 38 ms |
| — RANS SST-2003m | 27 | 25.9915408 | 25.9915400 | +3.1e-8 | 0.11 s | 11 / 44 ms |
| Re_τ 1e4, ny 256, β 4, RANS SA-neg | 27 | 25.4021882 | 25.4021884 | −7.7e-9 | 0.11 s | 14 / 44 ms |
| — RANS SST-2003m | 28 | 25.6762395 | 25.6762386 | +3.7e-8 | 0.13 s | 23 / 48 ms |
| — RANS SST-Vm | 28 | 25.5959025 | 25.5959014 | +4.3e-8 | 0.13 s | 20 / 47 ms |

All exit 0, mode `steady (implicit Newton pseudo-time: not time-accurate)`,
0 rejections, 0 SST guard hits, α = 1 after iteration 0. TOL 1e-6 (Re_τ 180
SA) certifies in the same 27. The residual at the stop is the FP32 floor
(0.02–0.03 at Re_τ 180, 0.9–8.7 at Re_τ 1e4): the stop is on the integrals,
as for the explicit march. The explicit march needs ~3.1e5 iterations / ~50 s
for Re_τ 180 TOL 1e-4, and its default cap (1e6) cannot reach TOL 1e-4 at
Re_τ 1e4 (it needs ~1.3e6).

## `dns flow` on the body-fitted O-grid

`dns flow --physics compressible --ogrid --rans sa-neg|sst-2003m|sst-vm
[--steady TOL]`, cylinder / lens / cube / z-invariant mesh. Wall-resolved
only: the validator warns when the first-cell y⁺ estimate
(h1/D)·Re·0.38·Re^-0.1 (a bluff-body PEAK u_τ, calibrated on OpenFOAM's
0.137 U for the sphere at Re 1e4) is above 1; the run measures y⁺ (stats
`yplusMax`, `yplusMean`). Rejected by name: incompressible, `--les`/`--sgs`,
any `--rotor-*`, `--scene`, `--penalize`, `--cut-flux`, `--pot`, `--seq`,
`--noslip`, wall-mounted bodies, and immersed (non-`--ogrid`) bodies.

**Plumbing.** `RansSetup` carries a `mutHook` (the mesh family's μt/F1 image
rows, run after μt is formed on every cell and before any flux reads it) and
a `fluidMask` (1 = real fluid cell: the SST guard counts only these,
`residual()` norms only these) in place of the channel's y-mirror flags;
`testSource` (TEST ONLY, compile-time) enters R before the point-implicit
quotient. `residualInto(s, out)` writes the pure rhs into a caller buffer.
`enableSteady(mask)` gives the laminar march its local-dt buffers
(`rans::LaminarDiffDt`, μt = 0, a separate functor so the RANS functor's code
is untouched). The ghost scalar algebra is `gpu/internal/rans_wall.hpp`
(shared with the channel); the explicit steady driver is
`gpu/internal/steady_march.hpp`, shared by the channel.

**Wall row** (j = 0): ν̃_g = −ρ_g ν̃₁/ρ₁ (SA) and k the same; SST ω_g the
Dirichlet image of ω_w = 60 μ(T_f)/(β1 d1²)/ρ_f **per wall face** (i, k): the
face table holds 60 μ_ref/(β1 d1²) and Sutherland scales it by μ(T_f)/μ_ref
on the face temperature (the channel's constant-μ ω_w under Sutherland is a
recorded channel limitation). μt odd (the wall face is exactly 0.0, so the
molecular-only tally is the full wall stress and κ_t = 0 there), F1 even.
**Far row** (j = ny−1): φ_g = w φ₁ + (1 − w) φ∞ with the mean-flow row's own
tanh weight w — a convex combination (positive), and the upwind flux of an
inflow face carries φ∞ exactly; NOT a reflection. μt and F1 an EVEN copy of
row ny−2 (SST's centre gradient on row ny−1 wraps to row 0). Both rows zero
their cells' Kahan low words on a compensated state.

**Wall distance**: exact distance to the DISCRETE wall polyline (node row
j = 1) at the node-mean centroid, `ogridWallDistance` (threaded, bitwise
thread-count independent); ghosts take their source row's d. On a circle it
sits within the polyline's sagitta R(1 − cos(π/nx)) of r − R.

**Freestream**, in FREESTREAM units so a Mach study keeps the inflow
turbulence: SA `--rans-nutinf` ν̃∞/ν∞ (3, TMR); SST `--rans-kinf` k∞/U∞²
(2.25e-7: TMR's 9e-9 c∞² at M 0.2, Tu 0.0387%) and `--rans-mutinf` μt∞/μ∞
(0.009) → ω∞ = ρ∞k∞/(R μ∞) (0.25 U/D at Re 1e4).

**Stats** append `cdp, cdf, mutMax, yplusMax, yplusMean, floorHits` (RANS) and
`iter, resid, resU, resE, resT, remaining` (compressible steady); residual
norms in freestream units (ρ∞, U∞, D: momentum D/U², energy D/U³, SA
1/U², SST k D/U³ and ln ω D/U). `--emit surface` writes `<out>_surface.csv`
once at the end (`docs/viewer-format.md`). RunMeta adds `"ransMode"`,
`"ransInflow"` on flow RANS / compressible steady runs only.

**Verification**: `gpu_rans_ogrid` (a)–(g) and the residual MMS
`gpu_rans_mms`: on the annulus (32×24, 64×48, 128×96, β 1) the scalar
residual converges at order 1.85 (SA, u = 0: diffusion, sources, cb2, ∇ρ),
1.88/1.85 (SST k/ω, u = 0, F1 = F2 = 1 branch), 0.91 / 0.97 (SA full /
convection-isolated: the first-order upwind) and 0.92/0.90 (SST full); the
FP64 reference's own error is ≤ 1e-7 of the finest.

**URANS** (cylinder Re 1e4, `--n 384 --ny 192 --ogrid-beta 4 --ogrid-far
30`, quiet start, 5 D/U): finite, 0 guard hits, y⁺max 0.65–0.66; 0.117
ms/step laminar, 0.275 SA-neg (2.3×), 0.325 SST-2003m (2.8×); μt/μ max 178
(SA) / 6.2 (SST) at t = 5 D/U.

### Stopping on a jitter floor (`dns flow --steady`)

The channel's geometric rule never certifies an external flow: the integrals
settle, then JITTER at the FP32 floor -- the residual floor moves the state by
~1 ulp per cell every iteration, and a pressure-integrated coefficient
amplifies p's ulp by p/q∞ = 2/(γM²) ≈ 36 at M 0.2. Measured jitter (rms about
the trend, relative, 200-iteration samples): cd 3e-7 / cdf 4e-7 / W 5e-7 on
the cylinder Re 40; cd 1.5e-6 / W 2e-5 on the laminar sphere wedge L3; cd
5.4e-5 / cdf 1.5e-5 / W 2.6e-4 / ∫ρν̃ 3e-7 on the SA sphere wedge Re 1e4 L2 --
WHITE (lag-1 autocorrelation −0.1..0.3, block means of 100 fall as 1/√100),
all above kSteadyFlat = 2⁻²⁰. A pseudo-time LIMIT CYCLE looks different: SST
sphere Re 1e4 L2 swings cd by 4.3e-3 (lag-1 acf −0.90, W 3.2e-2).

`dns/steady_stop` `steadyFloor` (flow runners: O-grid explicit and NK,
immersed explicit; the channel keeps `steadyRemaining`, byte for byte), per
series over the second half of the history (three blocks of b ≥ 8
samples, twelve sub-blocks):

- **jitter J** = rms of the samples about each block's own least-squares
  line (relative) -- REPORTED per series at the end of every run
  (`# steady stop (jitter-floored, relative): cd trend ... jitter ...`);
- **trend** from the block means m1, m2, m3 (which average the jitter
  down), against their detection limit δ = 3√2 σ_m, σ_m from the second
  differences of the sub-block means (blind to a linear trend, so a decaying
  mode cannot inflate it; it sees correlation up to b/4 samples):
  |d1|, |d2| ≤ δ → trend = max|d| + δ; same sign and |d2| + δ < |d1| − δ →
  geometric with r = (|d2| + δ)/(|d1| − δ), trend = (|d2| + δ) r/(1 − r);
  otherwise +inf. A window flat to kSteadyFlat (block means and their
  reversal) is trend = max|d|;
- **monotone to the noise**: the sub-block means' reversal must stay within
  3√2 of their own σ (without it the test's damped swing certified TOL 1e-3
  at sample 51 with the last sample 5.1e-3 off, instead of at 784);
- **verdict**: an OBSERVABLE (cd, cdf: reported values) is certified when
  trend + √2 J < TOL -- the LAST sample, jitter included, is inside TOL of
  the fixed point; a MONITOR (W, the turbulence integrals: slow-mode
  detectors) when trend < TOL and J ≤ **kSteadyJitterMax = 2⁻¹⁰** (9.8e-4:
  above the largest FP32 jitter measured, 4.2e-4 for W on the Re 1e4 wedge;
  below the SST limit cycle's 4e-3 / 3e-2).

The full revolution adds two observables, `cd+cl` and `cd+cs` (section "The
full revolution": cd is blind to a growing m = 1 mode).

What it does not do: an oscillation of a MONITOR below 2⁻¹⁰ is not told
from FP32 jitter; an observable's undamped swing of amplitude A is
certified iff trend + A < TOL (every sample is then inside TOL); a single
series decelerating into a turning point beyond its window is fooled, as
under the geometric rule -- the reason for several series. The window is
half the history, so a run certifies at about twice its settling time.
Test `steady_stop` and `gpu_runners` (SA cylinder Re 40 certifies).

Measured (gfx1100, M 0.2, cap far above the stop):

| run | stop | iterations | wall | cd | jitter cd / W |
|---|---|---|---|---|---|
| cylinder Re 40 laminar 256×96, TOL 1e-4 | exit 0 | 31200 | 3.0 s | 1.56656 | -- |
| — NK (`--steady-scheme implicit`) | exit 0 | 78 Newton | 28.4 s | 1.56656 | -- |
| — RANS SA-neg, NK | exit 0 | 73 | 30.1 s | 1.57012 | -- |
| — RANS SST-2003m, NK | exit 0 | 174 (13 rejections) | 55.4 s | 1.56656 | 4.1e-7 / 2.5e-6 |
| sphere wedge laminar Re 100 L3 256×192 | exit 0 | 100000 | 26.1 s | 1.09543 | 8.7e-6 / 8.7e-5 |
| — L2, wedge 2.5° | exit 0 | 244000 | 35.3 s | 1.09558 | 1.6e-5 / 8.4e-5 |
| sphere wedge RANS SA-neg Re 1e4 L2 128×96 | exit 0 | 606000 | 184 s | 0.42480 | 6.4e-5 / 3.1e-4 |
| — RANS SST-2003m | exit 8 (limit cycle) | 1e6 cap | 340 s | 0.4088–0.4136 | 4.3e-3 / 3.3e-2 |

## Axisymmetric wedge and the revolved O-grid

`dns flow --physics compressible --body sphere --ogrid --nz 1 [--ogrid-wedge
DEG]`: the O-grid REVOLVED about x (`buildRevolvedOGrid`), i = θ from the
UPSTREAM axis (`--n` meridional cells over [0, π], grid nx = n + 2), j = r
(two ghost rows), k = φ: the WEDGE of DEG (default 5) degrees, one real cell
plane between two ghost planes (internal nz 3). Labelled `[axisymmetric wedge
DEG deg: ensemble mean, no swirl, no 3D wake modes]` (a DNS run: no "ensemble
mean"; a time-accurate one adds `; axisymmetric unsteadiness only -- NOT the
sphere's physical shedding`). The full revolution (`--nz N`): section "The
full revolution" below.

**Axisymmetry is a mesh plus a ghost hook, never a kernel branch.**

- **Axis faces are exactly degenerate**: the axis node columns sit at ρ = 0
  to the bit (θ evaluated on the half nearer 0 and reflected), so the +i
  face S of columns 0 and nTheta is (0, 0, 0) and NO flux of any kind
  crosses the axis (`faceFluxCurv` multiplies every term by S).
  `CurvMetrics::build` requires it and stores the analytic LIMIT metric there:
  rows j, k of dξ/dx are 0 (read only times S = 0), row i is (t_j × e_φ)/(c_i ·
  (t_j × e_φ)), c_i = ±2ρ_F ρ̂_F the vector from the fluid cell's centroid to
  its across-axis continuation — ŷ/(2ρ_F) on the real plane (test `ogrid`,
  2e-16). The axis cell's centre gradient is then the across-axis central
  difference (the Mohseni–Colonius r → −r treatment) on a finite metric.
- **Ghost columns hold the across-axis CONTINUATION** (`gpu/axis_bc`, after
  the O-grid rows): (i_g, j, 1) = R_x(π) of (i_s, j, 1), every row; then the
  ghost planes (i, j, 0/2) = R_x(∓Δφ) of plane 1, every column (rotations
  about x commute). Bit-exact against a host recomputation (explicit fmas).
- **The ghost-column GEOMETRY is a placeholder.** A point reflection
  node(0) = 2 node(1) − node(2) has the WRONG handedness (an across-axis
  continuation runs its azimuth backwards in index space, so the volume and
  the j/k face Jacobians come out negative). The nodes are kept, and
  `build()` gives the placeholder columns their real neighbour's volume and
  j-/k-face tables; no real cell reads them.
- **`AxisWrap::MirrorZ`**: node planes 0..2 at φ = −3/2, −1/2, +1/2 Δφ are
  written as exact z-mirrors, node(3) is the mirror of node(0), so the two
  side faces of the real plane are exact mirrors (bitwise).
- **The wedge's azimuthal CFL width: ρ for the time march, ρΔφ for the
  steady local dt.** The ghost planes are slaved rotations, so the side faces
  carry only hoop terms whose rates go as 1/ρ: the time march's `aov[2]`
  keeps ×Δφ (time-accurate wedge runs are byte-identical to 75f39db). The
  steady march's local dt reads `CurvMetrics::aovSteadyZ`, the side faces'
  real spacing 1/(ρΔφ): with ρ there the cold-start RANS transient falls into
  a pseudo-time LIMIT CYCLE (sphere Re 1e4 SA L2: cd −6.5..7.0, resid ~800),
  in every partial variant tried (axis columns only, off them only, wave
  bound only, diffusive bound only). The ρΔφ width puts the laminar L3
  (256×192×5°) and 128×96×2.5° axis dtRef at 6.7e-7 -- legitimately below
  the absolute 1e-6 collapse guard -- so the revolved steady march uses a
  RELATIVE guard, dtRef < 1e-3 × the first dtRef
  (`SteadyMarchParams::collapseRel`); every other runner keeps its absolute
  guard bit for bit.
- **Far row**: the same LODI algebra on each far face's own 3D normal (from
  the metrics), the tangential deviation blended by the same tanh weight.
- **Tallies** over the real faces only (i ∈ [1, nTheta], plane 1) times
  2π/Δφ; the lateral forces and the torque are 0 by symmetry. qRef on πD²/4.
  Archimedes on a linear pressure: the scaled tally within 0.5% (test
  `gpu_axisym` (g); the residual is the first-cell face pressure).
- **Wall distance**: exact, unsigned BVH distance to the real wall quads (4
  triangles each through the centroid; the axis quads' degenerate triangle
  dropped), `revolvedWallDistance`.
- **Stats** add `lr` (the recirculation length on the downstream axis
  column); the run prints θ_s (first cf sign change from the front) and L_r
  at the end.

Laminar sphere Re 100, M 0.2, wedge 5°, rFar 20 D, β 3:

| level (n × ny) | cd | θ_s | L_r / D | ms/iteration |
|---|---|---|---|---|
| L1 64×48 | 1.09280 | 126.27° | 0.898 | 0.120 |
| L2 128×96 | 1.09492 | 126.43° | 0.890 | 0.145 |
| L2, wedge 2.5° | 1.09556 (+5.8e-4) | 126.38° | 0.892 | 0.148 |
| L3 256×192 | 1.09543 | 126.47° | 0.889 | 0.266 |

Literature context only (Johnson & Patel 1999): cd ≈ 1.09, θ_s ≈ 127°, L_r
≈ 0.88 D. Δφ 2.5° vs 5°: cd within 1e-3 (+5.8e-4). OpenFOAM parity on the
identical mesh (`sph100`): tier B −0.49% / −0.095% / −0.013% at L1 / L2 / L3,
θ_s within 0.2°, L_r within 0.3% at L3 (docs/validation.md).

**RANS on the wedge, sphere Re 1e4** (explicit LTS, 5°, rFar 20 D, β 4.5):

| run | ms/it | outcome |
|---|---|---|
| SA-neg L2 128×96, M 0.2, TOL 1e-4 | 0.30 | **exit 0 at 606000 iterations, 184 s**: cd 0.42480, cdf 0.03496, θ_s 87.40°, L_r 1.2206 D, y⁺max 0.33, μt/μ max 108, 0 hits (cd within 1e-3 of it from 2.6e5 iterations, 79 s) |
| — M 0.1 | 0.31 | exit 8 at 2.5e6 (766 s), settled: cd 0.41943 (to 1e-3 from 5.0e5), θ_s 87.64°, L_r 1.2065 D. Its jitter is above TOL 1e-4 (cd 1.6e-4) and its W jitter above the monitor floor (4.7e-3 > 2⁻¹⁰): the p/q∞ amplification is 4× M 0.2's (issues.md 4) |
| SST-2003m L2, M 0.2 | 0.34 | exit 8 at 1e6: a LIMIT CYCLE in the near wake (θ 125–145°, j 38–50; u swings 0.28 in 100 iterations), cd 0.4088–0.4136 -- the time-accurate URANS is unsteady in the same band: no stable steady state (issues.md 2) |
| SST-Vm L2, M 0.2 | -- | steady: cd 0.41882, jitter 6.5e-5, θ_s 87.29°, L_r 1.1786 D; remaining 1.1e-4 at the 3e5 cap |
| SA-neg L1 64×48, M 0.2 / 0.1 | 0.22 | exit 8 at 2.5e6: no steady state -- cd 0.4197 / 0.4264 with a 1.1% / 0.8% jitter, θ_s 12.9°: a spurious front recirculation (cp 0.58 at the stagnation point) |
| SST-2003m L1 | -- | exit 7 at 4.8e4: the closure-independent L1 front failure of the LTS march (issues.md 1) |

`sph1e4` against OpenFOAM (central U): SA tier B at L2 −0.35%
(docs/validation.md).

### SST: small-case verification

The SST closure is verified pointwise (test `rans` `sstRestatement`), as an
ODE (`gpu_sst` (a), the freestream decay), on the channel against the FP64
oracle (G9) and by residual MMS. What fails on the sphere wedge is the mean
flow's steady state (explicit LTS, M 0.2, `--ogrid-beta 4.5`, rFar 20 D, 2e5
cap; 1e5 for the laminar/SA/Vm columns):

| mesh, Re | SST-2003m | laminar | SA-neg | SST-Vm |
|---|---|---|---|---|
| 32×24, 100 | exit 0, cd 1.07749 | | | |
| 32×24, 300 | cd 0.67777 settled (remaining inf: W jitter 6.4e-4) | | | |
| 32×24, 1e3 | cd jitter 18%, resid 4.4 | jitter 20%, resid 6.5 | jitter 21%, resid 4.8 | jitter 11%, resid 1.8 |
| 32×24, 3e3 | resid 210, θ_s 10° | resid 150 | exit 7 at 2.8e4 | |
| 32×24, 1e4 | exit 7 at 9.6e4 | | | |
| 64×48, 100 | exit 0, cd 1.09080 | | | |
| 64×48, 300 | settled, resid 5.9e-3 (exit 8) | | | |
| 64×48, 1e3 | **exit 0, 114600 it, 28 s, cd 0.49851** | | **exit 0, 136600 it, 29 s, cd 0.48541** | |
| 64×48, 3e3 | remaining 1.7e-4 at 2e5, cd 0.44261 | resid 53 | resid 0.025 | resid 0.035 |
| 64×48, 1e4 | exit 7 at 4.8e4 (13.6 s) | resid 261 | resid 0.46 | resid 30 |
| 128×96, 1e4 | limit cycle, cd 0.409–0.414 | | exit 0 (606000 it) | steady, cd 0.41882 (remaining 1.1e-4 at 3e5) |

Two mechanisms, neither in the closure:

- **No stable steady state of the discrete model.** At L2 the SST-2003m
  TIME-ACCURATE URANS is unsteady in the band the LTS march cycles in (cd
  0.404–0.424, mean 0.4115 over t = 8–14.6 D_box/U, no decay; the LTS cycle
  0.4088–0.4136); the cycle sits at the separated shear layer's edge (θ
  125–145°, j 38–50, k ±40% there, μt/μ 4–40 against SA's 13–80 at the same
  cells), survives `--dt-recompute 1` and gets worse at `--cfl 0.2`. SST-Vm
  (vorticity production and μt limiter, the 1994 production limiter) is
  steady on the same mesh, SA too; which of the three differences carries
  the stability is not isolated. An explicit march cannot converge onto an
  unstable fixed point; OpenFOAM's SIMPLE does not need stability.
- **LTS focuses the radiation of an unsteady wake into the front.** Where
  the wake has no steady state (laminar from Re ~1e3 on the coarse meshes,
  every closure at L1 Re 1e4), the front stagnation region breaks into a
  period-2 θ sawtooth from the axis outward, for every closure -- laminar
  worst (L1 Re 1e4 wall cp 0.33, 1.48, 2.07, 1.09, −2.48 on i = 1..5).
  Capping the LTS ratio dt_c/dtRef at 1 (a global dt) keeps the front clean
  (laminar L1 Re 3e3: cp 1.21, 1.19, 1.17, 1.12, ...), at 8 it is already
  broken, and smoothing dt_c by a neighbour minimum does not fix it -- the dt
  CONTRAST between the coarse outer cells and the fine front cells is what
  matters. The time-accurate laminar L1 Re 3e3 run has a clean front (cp
  1.15, 1.13, 1.08, ...) and a shedding wake. SST-2003m at L1 Re 1e4
  diverges through this route (its URANS is unsteady: cd 0.29–0.47).

The small-case parity agrees once the reference uses the matched mean-flow
scheme: L1 Re 1e3, the smallest separated case that certifies, tier B cd0
0.49070 against OpenFOAM kOmegaSST with central U 0.48872 (**+0.41%**), θ_s
100.51° vs 100.77°, L_r 1.150 vs 1.067 D (gate `gpu_sst` (b)). SA-neg on the
same case (gate `gpu_sa_wedge`): exit 0 at 136600 iterations (29 s), cd
0.48541, θ_s 99.50°, L_r 1.2691 D, μt/μ max 9.86; tier B −1.07% at L1,
−0.125% at L2 (docs/validation.md). The Newton–Krylov march reaches the SST
plateau (cd 0.49852, resid 2.3e-3 against the explicit 0.17) but does not
certify within its 500-step cap (issues.md 3).

## The full revolution (`--nz N`)

`dns flow --physics compressible --body sphere --ogrid --nz N` (N > 1, even,
>= 4; default N = `--n`): the revolved O-grid of the wedge section with
k = φ CYCLIC over N cells, φ_k = 2πk/N, the node planes k and k + N/2 exact
(y, z) negations (computed as ± the same cos/sin), so the across-axis
partner of a cell is a cell to the bit. Everything else is the wedge's
machinery, unchanged and unbranched:

- **Axis**: the axis faces are degenerate on every k (S == 0 exactly) with
  the limit metric along each cell's own ρ̂_F (test `ogrid`: 4e-16 against
  ±ρ̂_F/(2ρ_F)); the ghost columns copy `(i_s, j, (k + N/2) mod N)`, every
  field, Cartesian components UNROTATED -- it is the same physical point, so
  the "correct vector rotation" is the identity (`kAxisColumnsFull`, μt/F1 by
  `kAxisMut`). Ghost-column geometry stays the placeholder.
- **Wall distance**: the exact BVH distance to all N·nθ real wall quads
  (`revolvedWallDistance(.., wedge = false)`). Equal to the wedge's at the
  same Δφ to 7e-15 and bracketed by r − R ≤ d ≤ r − R + sagitta (test
  `ogrid`).
- **No wedge widths**: the azimuth is a resolved direction, `aov[2]` is the
  true 1/(ρΔφ) for both marches and `aovSteadyZ` is empty; the relative
  dt-collapse guard is kept (the axis cells are small).
- **Tallies** over every real wall face (i ∈ [1, nθ], all k), scale 1, the
  lateral forces REPORTED (cl, cs: they are 0 only by the physics); L_r on
  the k-mean of the downstream axis ring; the end line prints θ_s per
  meridian (mean and range) and the largest azimuthal spread of u over the
  real (i, j) rings (`# revolved wall (full revolution, N meridians): ...`).
- **Labels**: no `[axisymmetric wedge ...]` suffix, no
  axisymmetric-unsteadiness tail; instead ` [full revolution N: odd-mode
  trip A U]` (or `untripped, exactly pi-symmetric`; plus `, started from the
  wedge PREFIX` with `--init-from`) on every run, the DNS time march
  included (`DNS [full revolution ...]`). RunMeta has no `wedgeDeg`. `--emit
  mesh` / `surface` work unchanged (the mesh file's flags have bit 3 without
  bit 2, wrap[2] = Cyclic, dphi = 2π/N: no version bump);
  `tools/parity/dnsmesh.py` reads the wedge only.
- **Rejected by name**: odd N or N < 4 (`--nz`), `--ogrid-wedge`, `--les`
  (the axis cells' filter width is unverified), `--steady-scheme implicit`
  (the 3D line ordering is not built).
- **The polygonal body.** The N facets about the axis give a frontal area
  sin(Δφ)/Δφ of the circle's (−2.55% at N 16, −0.64% at 32, −0.16% at 64);
  qRef stays πD²/4. Warned above 1%. cd converges in Δφ at second order,
  and the 5° wedge is the same discretization at Δφ = 5°.

**Exact π-symmetry, and the trip.** A rotation by π about the axis maps the
mesh onto itself to the bit, and the FP32 kernels are sign-symmetric, so an
axisymmetric start stays EXACTLY π-symmetric for ever (planes k and k + N/2
bitwise equal after 1.2e3 time steps, 8e3 steady iterations and 200 URANS
steps; cl == cs == 0.0). Rotations by 2π/N are NOT exact, so the EVEN
azimuthal modes do see rounding; the ODD ones -- the m = 1 lift mode the
laminar sphere wake bifurcates through at Re ≈ 210 (Johnson & Patel 1999) --
cannot arise at all. `--ogrid-trip A` (≤ 0.5; valid on every O-grid except
the axisymmetric wedge, where no planar vortex is solenoidal and there is no
odd mode to seed; **default `kOgridTripDefault` = 0.05 on the full
revolution**, 0 -- bit for bit the untripped start -- on the cylinder
O-grids) adds a solenoidal Gaussian vortex (axis z, strength A U, width D/4,
at (cx + D, cy + D/4, cz)) to the initial state -- with `--steady`, to the
initial guess: a stable fixed point is unchanged, an m = 1-unstable one is
left. `--ogrid-trip 0` is the untripped march bit for bit; it is warned at
Re ≥ 200 and with `--rans`. The resolved trip is in the describe block and
the label. Bracket (64×48×32, TOL 1e-4): at laminar Re 250 the trips 0.005,
0.01, 0.02 and 0.05 all reach the same m = 1 wake (cd 0.70690, cl −0.0647),
|cl| passing 1e-2 at 29400 / 27000 / 24600 / 21400 iterations, against the
untripped run's certification of the symmetric state at 36200 (measured
before the stop's lateral observables; the larger seed leaves the larger
margin). Where m = 1 is stable the trip decays: laminar Re 100 cd 1.08930 with
and without (|cl| 1.3e-6 at the stop), Re 200 |cl| 4e-5 (cd 0.77170),
SST-2003m Re 1e3 cd 0.49511 both ways.

On the 2D cylinder O-grid the same vortex (the cell centres sit on cz; an
extruded O-grid centres it in the span) breaks the up-down symmetry that
leaves shedding to rounding (`--kick` is rejected with `--ogrid`; `meta.kick`
and `meta.far = "ogrid"` record what the run does). Laminar Re 100, 192×96,
M 0.2, quiet start: every trip reaches the same limit cycle (|cl| envelope
0.333, cd 1.380); 90% of saturation at t 95.8 untripped vs 52.7 / 41.2 / 35.1
at A = 0.05 / 0.2 / 0.5 (99%: 114.6 vs 71.6 / 62.3 / 53.9). A trip shifts the
linear growth phase only; the ~40 time units of nonlinear approach stay.

Measured (M 0.2, rFar 20 D, explicit LTS, TOL 1e-4; tables in
docs/validation.md "The full revolution vs the wedge"):

- **Small invariants** (`gpu_axisym3d`, 32×24, 6 s): the freestream through
  the axis at the FP32 floor along, 30° to and across the axis; the 3D N 16
  march == the 22.5° wedge's (cd 4e-5, state 1.4e-5); conservation across
  the axis and the seam to FP32 rounding; a cross-flow sphere regular at
  the axis. 3D N 36 vs the 10° wedge through the binary (32×24, TOL 1e-5):
  cd 1.08139 vs 1.08138, θ_s 125.61° and L_r 0.9224 D both.
- **Δφ convergence** (laminar Re 100): 3D vs the 5° wedge −0.32% at N 32,
  −0.022% at N 64; second order in Δφ, the Δφ → 0 limit at L2 (1.09579)
  equals the wedge's own (1.09577).
- **The m = 1 mode**: tripped, the laminar Re 250 sphere takes the steady
  planar-symmetric wake (cd 0.70690, cl −0.0647; Johnson & Patel 1999,
  context: cd ≈ 0.70, cl ≈ 0.062) and SA-neg Re 1e3 leaves its axisymmetric
  state likewise (cd 0.49196, cl −0.0585); SST-2003m Re 1e3 is stable (the
  trip decays). Untripped, every closure certifies the π-symmetric state.
- **L_r is the k-mean on the downstream axis ring**, from the FIRST
  reversed cell along j (the wedge keeps demanding it of cell 1, verbatim:
  in 3D the first cell's k-mean sits within jitter of 0 at the rear
  stagnation point and flipped SA-neg Re 1e3 N 64 between 1.269 D and 0).
  An off-axis (m = 1) recirculation reads short or 0 there; θ_s is reported
  per meridian for the same reason.
- **The DNS time march pays the axis width**: the viscous dt of laminar Re
  100 is set by the axis cells' azimuthal width ½ R Δθ Δφ -- a true limit,
  not a bound artefact (issues.md 12). The steady march's local dt is
  unaffected.

**The stop watches the lateral forces** (full revolution only). cd is even
in the m = 1 mode, so a growing m = 1 oscillation leaves cd flat (G1 from the
wedge certified at 46600 iterations with cl swinging ±1e-3, and cl kept
growing to 1e-2 by 1.5e5). cl and cs have no scale of their own (0 at a
symmetric state), so they ride on cd's: two more OBSERVABLES, `cd+cl` and
`cd+cs` (trend + √2 J < TOL), in the `# steady stop` line. An untripped run
(cl ≡ 0) reads them as copies of cd: its stop iteration is unchanged. The
decaying trip transient costs where m = 1 is stable: SST-2003m Re 1e3
64×48×32 146600 iterations (95000 without the observables), laminar Re 100
unchanged (23400); the steady m = 1 wakes certify at laminar Re 250 108800,
SA-neg Re 1e3 128400.

**Starting from the wedge** (`--init-from PREFIX`, full revolution only):
the converged wedge run's `PREFIX_final.bin` (`--emit mesh`), its real plane
revolved over the N planes ((v, w) rotated to each cell's centroid azimuth
2π(k + ½)/N; ρ, u, p, the specific scalars copied), then the trip.
`PREFIX_mesh.bin` must carry this run's meridional nodes to 1e-9 of the
domain (so --n, --ny, --ogrid-beta, --ogrid-far and the body match), the
file this run's closure scalars, and its far-row pressure this run's p∞ to
2% (--mach); every mismatch, and a missing file, is exit 3 by name
(`dns/meshio` `revolveWedgeState`). Implies `--ramp 0`; labelled `, started
from the wedge PREFIX`. Measured (M 0.2, TOL 1e-4):

| case | cold start | from the wedge, trip 0.05 | from the wedge, untripped |
|---|---|---|---|
| laminar Re 100 64×48×64 | 47800 it, cd 1.09257 | 30800 it, 1.09256 | 10800 it (the stop's minimum history), 1.09256 |
| laminar Re 100 128×96×64 | 62000 it / 131 s, 1.09470 | 34400 it / 48 s, 1.09469 | -- |
| RANS SA-neg Re 1e4 G1 128×96×96 | exit 8 at 170000 / 581 s (m = 1 growing) | exit 8 at 170000 / 584 s: cd 0.42520, cl ±1e-3 → 8.7e-3, residual 0.13 → 4.0 | exit 0 at 38800 it / 135 s, cd 0.42520 |

An UNTRIPPED wedge start reproduces the axisymmetric state; it is not
evidence of 3D stability (G1: issues.md 11; an untripped cold G1 is the 3.75°
wedge to rounding, ~6e5 iterations, ~35 min).

## Newton–Krylov pseudo-time on `dns flow` (slower than explicit)

`--steady TOL --steady-scheme implicit` on `--ogrid` at `--nz 1` or the
revolved wedge (`gpu/nk`, `gpu/internal/nk_march.hpp`). Per Newton step
`(diag(1/Δτ_c) − ∂R/∂q) Δq = R(q)`, Δτ_c = CFL dt_c:

- an FP64 master iterate; the kernels see its FP32 rounding, hooked. R is
  ALWAYS `residualInto` — the central KG stage, no added dissipation — so
  the fixed point is the explicit march's.
- JFNK: J v = [R(q + ε D v) − R(q)]/ε on the scaled unknowns (ρ∞, ρ∞(U+c),
  ρ∞c², SA ρ∞ max(ν̃∞, ν), SST ρ∞U², ρω per cell). GMRES(40), CGS2 with FP64
  masked dots, forcing 0.1, ≤ 200 Krylov per step.
- P = diag(1/Δτ) − J₁, J₁ the FP64 dual-number derivative (`dns/dual.hpp`,
  `dns/nk_precond.hpp`) of a first-order Rusanov + compact-viscous flux, with
  the wall mirror and the wedge rotations folded into the diagonal; the
  scalars block-diagonal (upwind on the frozen KG mass flux, compact
  diffusion, the point-implicit source rate). Applied by a symmetric
  LEXICOGRAPHIC line Gauss–Seidel along the wall-normal lines in FP32
  (red-black lines ran the full 200 Krylov at every capped step): ONE launch
  per application (a 32-lane workgroup walks the lines i ascending then
  descending), the line factors stored line-major with each row's entries
  contiguous, each chain's rows staged into LDS by unrolled flat loads (a
  load-wait-use loop over global memory serializes), every lane carrying the
  whole running vector (no shuffles). 13.4 ms per application on the
  cylinder 256×96 (26 µs per line); the serial chain (2 × 94 rows per line, 2
  × 256 lines) is the floor of this algorithm.
- Continuation: CFL 10, ×2 per undamped step to 1e8, rejection (residual
  growth > 3×, non-finite, singular line pivot, GMRES breakdown) restores and
  cuts ×0.1, exit 7 below 0.1; the step length keeps ρ and E within 20%, SST
  ρω above −50% and ρk above −90% (whole-step scaling). The stop hears the
  integrals ONLY on an armed step (undamped, at the cap).

| constant | value | bracket |
|---|---|---|
| `kNkCfl0`, growth, cap, min | 10, ×2, 1e8, 0.1 | prototype |
| `kNkRejectGrowth` | 3 | plan |
| `kNkRestart`, `kNkMaxKrylov`, `kNkForcing` | 40, 200, 0.1 | forcing 1e-2 stalled at the cap, 0.1 converged |
| `kNkFdStep` | **2⁻¹⁴** | N1 below (2⁻¹⁰ fails) |
| `kNkStallKrylov` | 40 (one restart cycle) | at the CFL cap after a solve that hit the 200 cap unconverged: the FP32 floor |

Gates (gfx1100; test `gpu_nk` for N1–N3):

- **N1 (J v)**: forward ε vs central 2ε, per field, 16×12 cylinder Re 40.
  2⁻¹⁰: mean flow ≤ 4.6e-3 but SA's ν̃ 2.3e-2 (mid-transient) / 7.2e-2 (near
  the plateau) — FAIL; 2⁻¹² 3.0e-3 / 2.8e-2; **2⁻¹⁴ 1.1e-3 / 5.7e-3 (mean flow
  ≤ 3.4e-3) — PASS**.
- **N2 (assembly)**: the dual blocks equal a 4th-order central difference of
  F_P to 5e-12; the GPU's FP32-stored blocks equal a host FP64 restatement
  built from those central differences to 5.7e-8 of the block scale, every
  real cell, laminar and SA.
- **N3 (line-SGS)**: the GPU application equals a host dense block
  Gauss–Seidel on the downloaded P, same order, to 1.4e-7 (laminar) / 9.5e-8
  (SA).
- 2D: ρw stays exactly 0 through Newton iterations.
- **N5 (fixed point)**: NK and explicit plateaus agree -- cd 1.56656 (both,
  laminar), 1.57012 (SA), 1.56656 (SST-2003m), wedge laminar 1.092797 vs
  1.09280, wedge SA Re 1e4 L2 0.42482–0.42483 vs 0.42480.
- **Converges, slower than explicit**: laminar cylinder 78 steps / 28.4 s
  (explicit 31200 iterations / 3.0 s); SA 73 / 30.1 s; SST-2003m 174 / 55.4 s
  with 13 rejections. Does not converge from a cold start on the Re 1e4
  wedge (issues.md 3).

## Immersed-wall RANS (labelled)

`dns flow --physics compressible --rans ...` WITHOUT `--ogrid`: the body is
the compressible sharp ghost wall on the uniform or `--focal` box. Labelled
`[immersed ghost wall: first-order wall position, NOT body-fitted]`
everywhere — a staircase wall, never a code-to-code number when
under-resolved (the validator's y⁺ estimate uses h_fine/2).

- The closure lives on the curvilinear path: without `--focal` the solver
  runs on `CurvMesh::uniform` (RANS only; RANS-off keeps the Cartesian
  solver bit for bit). Cylinder Re 1000, 384², gfx1100: RANS-off Cartesian
  0.37 ms/step, SA 0.70, SST 0.75 (1.9–2.0×, curvilinear + RANS).
- d = |sdf| at the physical cell centre (`immersedWallDistance`; exact for
  sphere/cylinder/cube/mesh, the lens warns).
- Ghost images through the velocity's own image stencil (`DeviceCompGhost::
  applyTurb`, after `apply`): ν̃_g = −a Σw ν̃_s (odd: ν̃ ∝ d at the wall);
  **k_g = +a² Σw k_s** (EVEN and quadratic: k ∝ d² in the sublayer, wall
  value 0 and zero wall slope; an odd image is negative and fed the counted
  guard at convex staircase corners with two body neighbours); ω_g = (1 +
  a)ω_w − a Σw ω_s with ω_w = 60ν_w/(β1 d_s²), d_s = d_w/a, ν_w from the
  thermo sample — **an UNVERIFIED sub-cell choice**; deep cells ν̃ = k = 0,
  ω = ω∞; **μt_g = −min(a Σw μt_s, min μt of the ghost's fluid face
  neighbours)** and F1_g = Σw F1_s (`mirrorMut`); a compensated state's
  ghost and deep low words zeroed. The clamp keeps every staircase face at
  μ + ½(μt_g + μt_c) ≥ μ: the plain odd image extrapolates the wall-linear μt
  through a sub-cell wall, and wherever a ghost lies deeper than its fluid
  neighbour is far the face average goes negative. The neighbour list
  (`CompGhostTables::gfnb`, 6 per ghost, −1 padded) is built in
  `classifyCompGhost`, so it rides the rotor azimuth tables too. Unbound
  ghosts are the plain image bit for bit (strict <).
- **Wall-value convection** (`ScalarBoundary::solid`, the body's ghost and
  deep cells): a face between a fluid and a body cell lies at the wall to
  within the staircase, so when its mass flux points INTO the fluid ν̃ and
  k are carried at their wall value 0, not at the ghost's. ω keeps its wall
  image; diffusion still reads the ghost (the Dirichlet image).
- Open domain faces (x always, y/z with `--open`): `ScalarBoundary`, a
  compile-time branch of the scalar kernels — mdot = ρ_c u_c·S of the
  boundary cell, φ∞ in, φ_c out, no diffusion. The sponge/strip relaxes the
  scalars to ρφ∞ with the density's strength; the free-slip mirror rows copy
  scalars, μt and F1 even. NSCBC and the sink stay mean-flow only. After
  the NSCBC rhs hook, every open-face cell's scalars are re-carried by the
  density the mean flow actually produced (`rans::openScalarConsistency`,
  the φ R_ρ form): φ = t/ρ_s with ρ_s the density the scalar's own fluxes
  imply, then t = φ ρ_out (residual mode: R_t += φ_c (R_ρ − M)). NSCBC
  overwrites continuity with its characteristic form, which no scalar FLUX
  can match; the specific scalar is therefore advanced non-conservatively
  there, exactly as the characteristic continuity itself is.
- The body tally carries μt through the solver's own curvilinear flux; stats
  add `mutMax, yplusMax, yplusMean, floorHits` (y⁺ the linear-sublayer
  estimate sqrt(|u_c| d_c/ν_c) per interface face); immersed `--steady` (with
  `--rans` only) runs the explicit LTS march on the compensated state,
  stopped on cd and the turbulence integrals.

Gates (`gpu_rans_immersed`, gfx1100):

- **G2-flow**: focal + ghost wall + NSCBC, SA ν̃ ≡ 0 and SST k ≡ 0: mean
  flow AND body tally bit for bit the RANS-off run's over 50 steps.
- **Wall image**: a plane tilted 30° at a 0.37-cell offset: ν̃ = μt = c·d
  reproduced on every ghost to 2.6e-7 (c h); k = c·d² reproduced to the
  sample's bilinear interpolation error (≤ 0.25 c h², the image ≥ 0
  everywhere); the ω image is ω_w-anchored. μt_g = c·d on the 60 of 101
  ghosts where the clamp does not bind and −min(neighbour) on the 41 where it
  does (both to 1e-5 c h).
- **Bounded μt image (leg (e))**: 128² cylinder (D = 32 cells), Re 1e5, y⁺
  30–80, a seeded law-of-the-wall layer (ν_t = κu_τd e^(−d/2δ), δ = D/20; SA
  ν̃ = ν_t, SST the log-layer k/ω pair), 200 URANS steps. The plain odd image
  (recomputed on the host from the same tables) drives 51 of 128 staircase
  faces NEGATIVE for SA (down to −39 μ) and 21 for SST (−16.5 μ); unclamped
  SST's dt fell to 2e-9 by step 200. Clamped: 0 faces below μ (min (μ +
  μt)/μ = 1.0000), ghosts = −min(image, neighbours) to 1.2e-7, dt 2.6e-3 for
  both. Past 200 steps this Re still collapses dt, clamp or not (issues.md
  19).
- **G3-flow**: a uniform 2⁻¹³ specific scalar rides a non-uniform flow
  EXACTLY (0 ulp) in the interior AND in the NSCBC columns (the φ R_ρ form
  above is what closes it).
- **SST guard**: 0 counted hits on the 128² cylinder (200 steps) and on 384²
  Re 1000 over 7.5e4 steps. The even quadratic k image is what takes the
  staircase-corner hits to 0; the wall-value convection alone took the 128²
  case to 0 but left 3.6e4 hits at 384²; both are kept.
- Immersed cylinder Re 40 SA `--steady 1e-4` (384²) does not certify
  (issues.md 6).

## dt with RANS (D12)

RANS-only per-cell diffusive bound (`RansDiffDt`): 0.25/(D_c amax_c²),
D_c = max((μ+μt)/ρ, γ(μ/Pr + μt/Pr_t)/ρ, SA max(ν, ν + ν̃ f_n(χ))/σ); SST
uses the pointwise upper bound μt ≤ ρk/ω. The SA term is ν̃'s own diffusion
coefficient (`dns::saDtDiffusivity`): for χ ≥ 0 it is ν + max(ν̃, 0), for
χ ∈ [−2.52, 0) exactly ν, and below χ = −c_n1^(1/3) ≈ −2.52 f_n < 0 makes it
exceed ν (4.87 ν at χ = −5). The bound uses amax² where the compact face
Laplacian's stiffest mode needs Σ_a a_a²: unstable on isotropic 3D cells
where it binds (issues.md 15). The DNS/LES `stableDt` is unchanged by RANS;
its LES global-extreme bound (μ_eff,max/(h_min² ρ_min)) is over-conservative
by construction on a clustered mesh but has not been seen to bind
(docs/validation.md "LES known answers"); the incompressible dt ignores ν_t
and `kSmagNut` zeroes the domain ring (LES-correct).

## The FP64 oracle (`dns/rans_channel1d`)

The steady 1D restriction of the SAME discrete operator on
`CurvMesh::channelY` (same `CurvMetrics` in double, face formulas, ghost
algebra, body force), unknowns per row (u, T/Tw − 1, SA ν̃·Re_τ | SST ln k,
ln ω) plus the uniform p fixed by the total mass; Newton with
pseudo-transient continuation (block-Thomas for SA, dense LU for SST, whose
μt/F1 reach j±2). Temperatures enter as T − Tw: differencing T ~ Tw itself
floors the energy residual at eps·Tw/dy² (1.4e-10 measured). Converges to
resid < 1e-11 in 19–88 iterations, milliseconds.

Measured (`rans_channel1d`; the test's grids are pinned there, the
CONTINUUM column is the Richardson limit of ny 192/384/768 at stretch 2):

| | Re_τ 178.12: ny 64 | continuum | Re_τ 392.24: ny 96 | continuum |
|---|---|---|---|---|
| MKM ub⁺ | 15.679 | | 17.545 | |
| SA-neg | 15.8069 (+0.82%) | 15.8403 (**+1.03%**, order 2.17) | 17.5596 (+0.08%) | 17.6226 (**+0.44%**, order 2.06) |
| SST-2003m | 15.6332 (−0.29%) | 15.1289 (**−3.51%**, order 1.06) | 17.8881 (+1.96%) | 17.2213 (**−1.84%**, order 0.95) |
| SST-Vm | 15.5986 (−0.51%) | 15.0941 (**−3.73%**, order 1.07) | 17.8455 (+1.71%) | 17.1785 (**−2.09%**, order 0.95) |

Cite the continuum values: the test-grid SST numbers are mostly the wall-ω
discretization error, which has the opposite sign to the model's and is
larger than it at ny 96.

Grid convergence at Re_τ 178.12 (stretch 2): SA-neg 2nd order (ub 15.8346 /
15.8390 / 15.8400 at ny 192/384/768, apparent order 2.17; ny 48/96/192 is
pre-asymptotic, 1.23). SST-2003m FIRST order: the TMR wall value
ω_w = 60ν/(β1 d1²) moves with the first cell, an O(d1) boundary error. The
ladder (pinned in `rans_channel1d` to 1e-6) is ub 15.7764 / 15.4640 /
15.2920 / 15.2070 / 15.1662 at ny 48/96/192/384/768 (y⁺_1 0.59 → 0.034),
apparent order 0.86 / 1.02 / 1.06, Richardson limit 15.124 (96/192/384) /
15.129 (192/384/768). With d1 = the first cell HEIGHT instead of its centre
the order and the limit are the same (15.365 → 15.144 over ny 48…768), only
the coarse-grid error is smaller. A 6/β1 (cell) wall value converges at
about second order with 3–10× smaller error (a user decision, masterplan §5
item 4 -- the TMR default stays).

**Code to code: SA-neg vs TMR's CFL3D 2DFDC** (test `rans_channel1d` O5).
TMR's 2D fully-developed channel verification case, CFL3D SA, file
`sa_u+y+KM_cfl3d.dat` ("CFL3D high Re channel, SA"): u⁺ at y⁺ 30 / 100 /
300 / 1000 (linear in y⁺ between file points) 13.2582 / 16.1767 / 18.8002 /
21.6989. The file's own sublayer slope u⁺/y⁺ is 0.988 (0.9877–0.9883 over
y⁺ 0.6–1.7, where it must be 1), i.e. its wall units carry a ~0.6% u_τ
offset; the comparison is reported raw and sublayer-normalized (u⁺·0.988).
Oracle, Re_τ 1e5, ny 1024, stretch 5 (y⁺_1 0.090):

| y⁺ | oracle u⁺ | CFL3D | raw | normalized |
|---|---|---|---|---|
| 30 | 13.36065 | 13.2582 | +0.773% | −0.437% |
| 100 | 16.29580 | 16.1767 | +0.736% | −0.473% |
| 300 | 18.93785 | 18.8002 | +0.732% | −0.477% |
| 1000 | 21.85925 | 21.6989 | +0.739% | −0.470% |

Flat over the log layer (spread 0.04%) and grid-independent (ny 512 / 2048:
normalized −0.49..−0.52% / −0.42..−0.47%). Gated at |normalized| ≤ 0.75%,
|raw| ≤ 1.2%, spread ≤ 0.1%. The binary IS the oracle there:
`dns channel --n 4 --nz 1 --ny 2048 --stretch 4 --re-tau 1e5 --rans sa-neg
--steady 1e-6 --steady-scheme implicit` ub 31.0237053 vs the oracle's
31.0237061 on its grid (−2.5e-8, 27 iterations, 0.23 s; `gpu_rans_implicit`
I5b). SA-neg's log-layer slope over 30 < y⁺ < 0.1Re_τ (Re_τ 2000): 2.447 vs
1/κ = 2.439 (+0.32%); over 30 < y⁺ < 0.3Re_τ it is +3.2%, an outer-layer
(shear-stress) effect that does not refine away. Laminar limit: 2nd order to
the exact compressible Poiseuille (u and T).

### Scenario: the supersonic isothermal channel (Coleman, Kim & Moser 1995)

A scenario, not a test. CKM's DNS: M_b = U_b/c_w = 1.5, Re_b = ρ_b U_b δ/μ_w
= 3000 (Re_τ = 222 on wall values), isothermal walls; centre temperature
Tc/Tw ≈ 1.38, and the van Driest transformed profile collapses onto the
incompressible one. `--mach` is ub_est/c_w (log-law estimate) and `--re-tau`
is 1/μ with ρ_b = 1, so the CKM point is the fixed point `--re-tau 192.63
--mach 1.517` (oracle, solved for M_b 1.5 and Re_b 3000 on ny 128). A
supersonic `--mach` is accepted ONLY with `--rans --steady-scheme implicit`:
the iterate is one broadcast column, v = 0, no shock can form; everything
else keeps the (0, 1) bound (issues.md 23).

    dns channel --n 4 --nz 1 --ny 128 --re-tau 192.63 --mach 1.517 \
        --rans sa-neg --steady 1e-6 --steady-scheme implicit --out ckm

Exit 0 in 27 iterations, < 1 s (gfx1100). Label `RANS SA-neg -- steady
(implicit Newton pseudo-time: not time-accurate)`, constant viscosity (CKM
use μ ∝ T^0.7; `dns channel` has no viscosity law), Pr 0.72, Pr_t 0.9.

| | binary ny 128 | oracle ny 64 / 128 / 256 | CKM 1995 |
|---|---|---|---|
| ub (ρ_b = 1 units) | 15.5737 | 15.5538 / 15.5737 / 15.5822 | |
| Tc/Tw | 1.3816 | 1.3821 / 1.3817 / 1.3816 | ≈ 1.38 |
| ρ_w (ρ_b = 1) | | 1.3565 / 1.3584 / 1.3589 | |
| Re_τ on wall values √ρ_w/μ | | 224.6 / 224.5 / 224.4 | 222 |
| van Driest slope, 30 < y⁺ < 0.3 Re_τ | | 3.045 / 3.069 / 3.060 | |
| same at M_b 0.2, Re_b 3000 | | 3.038 / 3.039 / 3.041 | |
| raw u⁺ slope at M_b 1.5 | | 3.553 / 3.582 / 3.573 | |

y⁺ and u⁺ here are on wall values (u_τ = √(τ_w/ρ_w), τ_w = 1 discretely, ρ_w
extrapolated from the first two rows). The van Driest transform brings the
M_b 1.5 slope to within 0.2–1.0% of the M_b 0.2 one, against +17% raw --
CKM's collapse. The slope itself is 3.04, not 1/κ = 2.44: at Re_τ ~200 the
band 30 < y⁺ < 0.3 Re_τ is outer-layer dominated; there is no log layer to
read κ from. Tc/Tw 1.382 against CKM's ≈ 1.38 is within their figure's
reading, with constant μ against their power law. The nominal flags
(`--re-tau 220 --mach 1.5`) are a different point: M_b 1.480, Re_b 3491,
Re_τ (wall) 255, Tc/Tw 1.3686 (binary and oracle agree, ub 15.8693).

## Tests

| test | what |
|---|---|
| `rans` | SA/SST closed forms: fv1, fw, cw1, r cap, S̃ limiter, the log-layer identity (cw1; γ1 derived vs 5/9: 0 vs 0.43%), SA-neg branch continuity, J ≥ 0, positive branch == standard SA, the dt diffusivity `saDtDiffusivity` (ν + max(ν̃, 0) bit for bit at χ ≥ −2.52, ν + ν̃ f_n > ν below), point-implicit fixed point (and its increment form, exactly 0 there), float/double, F1/F2 limits, limiters, μt branches, wall distance (channel; O-grid polyline within the sagitta, ghosts, thread-count bitwise), the `ransRunLabel` strings (the channel's exactly) and `ransBuilt`; `sstRestatement`: the whole pointwise SST-2003m / SST-Vm pipeline (sstCd → sstBlend → sstMut → sstSource, the rate from the gradient tensor) vs an independent FP64 TMR restatement on 4e4 random states per closure and on every switch (F1 knee, production limiter, CDkω floor of both signs, μt limiter): FP64 ≤ 1e-12 (measured 7e-16), FP32 ≤ 1e-4 of the term scale |
| `gpu_sst` | (a) freestream decay on a periodic uniform box: k, ω vs the closed form, both closures, F1 = 0 and 1, TMR freestream and a stiff ω0 = 1e3; exactly uniform, 0 hits, first order in dt (ω 1.00–1.01, k 0.91–0.98), finest errors ≤ 1.25× measured; (b) the L1 sphere wedge Re 1e3 SST-2003m `--steady 1e-4` through the binary: exit 0 within 1.5e5 iterations (114600), cd / L_r / θ_s at their measured values (5e-4 / 5e-3 / 0.1°), cd0 within 1% and θ_s within 1° of OpenFOAM kOmegaSST (central U) on the identical mesh, 0 hits |
| `gpu_sa_wedge` | the same L1 wedge Re 1e3 with `--rans sa-neg` (~30 s): exit 0 within 1.8e5 iterations (136600), cd / L_r / θ_s at their measured values (5e-4 / 5e-3 / 0.1°), μt/μ max in (5, 20); the OpenFOAM SpalartAllmaras (central U) comparison is printed, NOT gated (tier B −1.07% at L1, the L1 discretization: −0.125% at L2) |
| `rans_channel1d` | the oracle: laminar order, MKM band (5% provisional ceiling, values pinned to 1e-6), log slope (1%, measured +0.32%), ub order (SA 2nd), the SST-2003m wall-ω ladder ny 48–384 (pinned 1e-6, order 96/192/384 in [0.8, 1.3], measured 1.02; Richardson limit 15.13 ± 0.02, measured 15.124), O5 SA-neg vs TMR CFL3D 2DFDC at Re_τ 1e5 ny 1024 (sublayer-normalized ≤ 0.75%, raw ≤ 1.2%, spread ≤ 0.1%; measured 0.48 / 0.77 / 0.04%), Newton < 2 s; 0.4 s |
| `gpu_rans` | G1 lts = 1.0f bitwise; G2 ν̃ ≡ 0 bitwise to RANS off; G3 uniform scalar exact; G4 wall face μt == 0, ghost algebra; G5 residual at the oracle at the FP32 floor; G6/G7 oracle fixed point (uniform / random lts); G8 SA-neg ν̃ < 0 patch; G9 the PLAIN march onto the oracle from the 5% perturbation on ny 32 stretch 2.6 (1.2e5 iterations, ub −0.053%; its residual ends ≥ 10× the oracle state's); G10 compensated state (TwoSum exact, pair upload/download, plateau ≤ ½ plain, oracle fixed point u and ν̃ < 1e-6 (measured 3e-8 / 8e-8), perturbed 2.2e5 march, ny 64) |
| `gpu_rans_sst` | `test_gpu_rans sst`: SST k ≡ 0 bitwise to RANS off; wall face and ghost algebra (ω_w on the face); residual at the oracle; G6 fixed point (2003m u 1e-6 / k 2e-4, measured 8e-8 / 1e-5; Vm u 1e-4 / k 1e-3, measured 3.9e-5 / 8e-5, the plain dynamic floor); SST-2003m G9 2.2e5 march onto the oracle; rough-field positivity, 0 guard hits |
| `channel_config` | every `--rans`/`--steady`/`--steady-scheme` rule (TOL ∈ [1e-6, 1); implicit rejects `--cfl`/`--dt-recompute`, cap 500, a given `--steady-steps` honoured), labels and describe lines of both schemes, explicit given == defaulted byte for byte, json `"steady"` only on implicit, RANS-off describe unchanged; `--mach 1.5` accepted only with `--rans --steady-scheme implicit` (rejected on DNS, URANS and the explicit march; `--mach 0` still rejected) |
| `steady_stop` | the channel's geometric rule: geometric tail exact, never early; growth / drift / short / zero-scale never certified; undamped oscillation never, damped only once flat; 3e-8 noise certified; a single series fooled at a turning point, a companion holds it off. The flow's jitter-floored rule (`steadyFloor`): white jitter 3e-7 / 5e-5 read back, certifies TOL 1e-3 and never a TOL below it; geometric tails clean and under 3e-7 / 1e-6 jitter, never early; linear drift and a 1e-6 drift under 1e-6 jitter never, a 1e-7 drift never once its window resolves it; undamped 1.5 TOL swings (3 frequencies) never; a period-2 limit cycle never; a monitor swing above 2⁻¹⁰ never; damped swings clean / jittered never early; the turning-point pair; SteadyStop(n) == steadyRemaining and SteadyStop(n, k) == steadyFloor bit for bit |
| `blockband` | block LU vs a dense partial-pivot LU: random B 5/6 × K 1/2 × n 8/64/254 and the line matrix at CFL 100 (backward error ≤ 1e-12, agreement ≤ 1e-10); at CFL 1e8 backward errors only (the matrix is singular to ~1e-8 by Σ V R_ρ ≡ 0, so ANY two solvers differ by ~cond eps); singular / NaN / Inf → error |
| `channel_line` | (a) R at the oracle (tol 1e-11) ≤ 1e-10 momentum/k, ω 1e-13 relative, E ≤ max(1e-7, its 1-ulp floor); (b) band EXACTLY K, ghosts included; (c) J δ vs central FD ≤ 1e-6 (measured 1.5e-7); (d) host Newton from Reichardt: SA/2003m/Vm Re_τ 180 ny 64, SA/2003m Re_τ 1e4 ny 128 in 27–28 iterations, ub ≤ 7e-10 off, mass 1e-12; (e) a damped march never certified (an unarmed stop does); (f) projection: 2 ulp, mass 1e-15; host timings |
| `pseudo_time` | CFL 100·2^n to the cap at n = 20; α < 1 holds and disarms; rejection ×0.1, false below 1; armed only after an undamped step at the cap; ramp/reject/regrow never arms below the cap; each step bound binds alone, SA has none |
| `gpu_rans_implicit` | I1 GPU vs line residual per equation (10× measured: ρ 2e-6, mx 3e-4, my 1e-3, E 2e-4, t 5e-4 / 7e-5); I2 line J vs GPU FP32 central FD ≤ 2e-2 (measured ≤ 1.8e-2); I3 one iteration at the oracle: dub ≤ 1.2e-10, mass exact, resid 0.016–0.031; I4 Re_τ 180 cold starts via the binary (27–28 it, ub ≤ 8e-8 off, 0 rejects / hits) + R_mz, w ≡ 0 and bitwise column-uniform state every iteration; I5 Re_τ 1e4 ny 128 (27 it, ≤ 3e-8, 0.1 s); I5b SA Re_τ 1e5 ny 2048 `--stretch 4` TOL 1e-6 (ub ≤ 1e-6 of the oracle, measured −2.5e-8, 27 it, 0.23 s); I6 IC residual column-uniform, nx 6 nz 2 iterate == nx 4 nz 1 bitwise; I7 TOL 1e-6 |
| `gpu_runners` | `--steady 1e-3 --steady-scheme explicit` SA-neg end to end on ny 32 stretch 2.6: exit 0, > 3.2e4 iterations (stops at 64000; ub alone would certify at its dip, 1600), ub within 1.5e-3 of the in-process FP64 oracle 15.6920 (TOL + the mass-drift offset; measured −0.042%); the implicit runner path is `gpu_rans_implicit` I4/I5/I7; O-grid SA URANS (RANS columns finite, 0 hits), the O-grid SA explicit steady march to a 2000-iteration cap (exit 8, a row every 200, finite, 0 hits) and certified at TOL 1e-4 (exit 0 within 2–6e4 iterations, cd within 1e-4 of the plateau 1.5701190); immersed SST 384² Re 1000 URANS, 0 guard hits |
| `gpu_rans_mask` | the plumbing on the channel: masked residual == row-range residual bitwise (every norm), `residualInto` == `residualField`, the channel mutHook odd μt / even F1, the SST guard counts only masked cells (0 vs 120 with every cell), `testSource` enters R as one FP32 add, laminar `localDt` == SST's at k ≡ 0 bitwise and the nT = 0 compensated LTS march; M6 an SA-neg χ = −5 cell takes `RansDiffDt` = 0.25σ/(K a²), K = ν + ν̃ f_n; M7 the `RansDiffDt` stability probe (16² / 16³ periodic box, uniform μt, T checkerboard, 2000 steps at the bound): 2D decays (pinned), 3D grows (printed; issues.md 15) |
| `gpu_rans_ogrid` | (a) RANS tables on a RANS-off solver change no bit; (b) ν̃ ≡ 0 / k ≡ 0 mean flow == RANS off bitwise over 50 fixed-dt steps; (c) far row φ∞ to 0.4/0.8 ulp, outer rows after 100 steps; (d) wall-face μt == 0.0, far ghost even copy, F1 even, ω_w per face to 1e-5; (e) a uniform 2^-13 scalar rides ρ exactly (0 ulp) everywhere; (f) SA/SST 200-step URANS on 256×96 finite, 0 hits; (g) compensated LTS: rows 0 and ny−1 low words 0 after every apply |
| `gpu_rans_mms` | residual MMS on the O-grid annulus, 3 levels: SA u = 0 order 1.85, SST u = 0 1.88/1.85, SA full 0.91, SA convection 0.97, SST full 0.92/0.90 (bands [p − 0.25, p + 0.5]); FP64 reference error ≤ 1e-3 of the finest; the revolved WEDGE ladder with axisymmetric fields (hoop terms): SA u = 0 1.94, SA full 0.97, SST u = 0 1.92/1.93, SST full 0.97/0.89 |
| `gpu_axisym` | the revolved wedge: (a) uniform x-freestream fixed point (3.4 ulp, outer rows 3.0); (b) side-face mass flux of a random axisymmetric state 0.02 ulp·ρvA; (c) ghost-column R_x(π) / plane R_x(±Δφ) bit-exact vs host; (d) w ≡ 0 kept (2e-10 U over 200 steps); (e) laminar Re 100 smoke; (f) SA/SST 200-step smoke, 0 hits, μt 0 on the wall faces; (g) Archimedes on a linear pressure within 0.5% (×2π/Δφ); (h) real-plane stage range + skipped entry hook byte-equal to the whole-grid march over 120 steps, every field and low word (SA plain URANS, SST compensated LTS) |
| `gpu_axisym3d` | the full revolution, 32×24 (6 s): (a) uniform freestream along, 30° to and ACROSS the axis (the mesh rotated): residual 4.0–4.9 ulp of the largest face flux on every real cell, axis rings 0.7–3.4; the outer third after 100 steps 4.5–4.9 ulp (5° wedge baseline 6.0); (b) axisymmetric start, laminar Re 100, N 16: the azimuthal spread random-walks (×2.1 from 300 to 1200 steps; ρ 37, u 54, v_r 62, p 18 ulp, swirl 5e-6 U) and saturates in an 8000-iteration steady march (×1.7 from 2000; swirl 1.3e-5 U); exactly π-symmetric bitwise; (c) N 16 vs the 22.5° wedge, same march: cd ≤ 4.3e-5 apart at every sample, every plane == the wedge's rotated to 1.4e-5; (d) cross-flow sphere (axis ⟂ flow), N 32: cd +0.97% of the axial-flow cd (64×48×64: +0.14%), lateral force 5e-6 cd, a quadratic fit extrapolates ONTO the axis ring better than one ring out (2.7% vs 5.0% of the range); (e) a bump across the axis and the seam: Σ V ΔR telescopes to 2.6e-6 of the flux activity (3.0e-7 at 4× the amplitude: FP32 cancellation of p∞\|S\| in the small axis cells, not a leak; off both 1.2e-8); (f) SA/SST 200-step URANS, 0 hits, μt 0 on every wall face, π-symmetric; (g) the tally on a linear pressure along / across the axis == −Σ p₁ S (1.1e-6 / 2.1e-8), Archimedes of the faceted body within 1.4%, no off-gradient force |
| `gpu_axisym3d_run` | the full revolution through the binary (~40 s; the tripped Re 250 run certifies at TOL 1e-3, 89200 iterations, same cd/cl digits as 1e-4; (c)'s cold reference is (a)'s N 36 run): (a) laminar Re 100 32×24, `--nz 36` vs the 10° wedge at TOL 1e-4: the DEFAULT trip decays (cd within 1e-4 of the wedge, θ_s 0.02°, L_r 0.1%, \|cl\|, \|cs\| < 1e-4 at the stop), `--ogrid-trip 0` keeps cl == cs == 0.0 on every row, 1152 surface rows, the log names the resolved trip; (b) laminar Re 250 64×48×32: `--ogrid-trip 0` certifies the symmetric state (cd 0.69099, cl == 0.0 on every row), the default trip the steady m = 1 wake (cd 0.70690 within 0.2%, \|cl\| 0.0647 within 5%, \|cs\| < 1e-3 \|cl\|); (c) `--init-from` the 10° wedge: exit 0 at the cold start's cd (1e-4) in fewer iterations, the start reported and labelled; a wedge of another `--ny` / `--mach` / closure / `--ogrid-beta` and a missing prefix are exit 3 with the named reason |
| `flow_rans_config` | every `dns flow --rans` / compressible `--steady` accept, reject (by flag, exit 2) and warning (the full revolution's `--nz`, `--ogrid-wedge`, `--les`, implicit, `--ogrid-trip` (default 0.05, an explicit 0 warned at Re ≥ 200 / RANS) and `--init-from` rules, the facet warning, the full-revolution label suffix); the O-grid start: `--kick` rejected, `meta.kick` 0 and `meta.far` "ogrid", `--ogrid-trip` on the 2D / extruded cylinder (default 0, labelled `[O-grid trip A U]`, rejected on the wedge), no Cartesian D < 4h guard (`--n 128 --ny 64`); the label/turbulence/wall/y⁺/run describe lines; RANS-off describes byte-identical; explicit given == defaulted; RunMeta keys only on flow RANS/steady; `--compensated` accepted on O-grid URANS / Cartesian / MRF / channel (state key + describe tail), a no-op on explicit `--steady`, rejected with implicit / incompressible / `--penalize`, unknown on conv / hit |
| `comp_time_order` | + the compensated leg: same order band [2.7, 3.3] on the vortex, plain − compensated at dt/4 < 1/10 of the time error, compensated FP32 rounding < 1/2 of the plain run's |
| `gpu_determinism` | + the O-grid SA URANS run twice, stats byte-identical |
| `gpu_nk` | the flow Newton–Krylov pieces: dual blocks == 4th-order FD of F_P (5e-12); N2 GPU P blocks == host FD restatement (5.7e-8, FP32 storage); N3 the single-launch FP32 line-SGS == host dense block GS (≤ 1e-5; measured ~1e-7); N1 forward vs central J v per field at 2⁻¹⁴ (the 2⁻¹⁰/2⁻¹² sweep printed); ρw ≡ 0 and the residual falls through 6 Newton steps (laminar, SA) |
| `gpu_rans_immersed` | (a) G2-flow: focal + ghost + NSCBC, ν̃ ≡ 0 / k ≡ 0 mean flow and tally == RANS off bitwise; (b) G3-flow: a uniform scalar exact (0 ulp) in the interior AND ≤ 2 ulp in the NSCBC band (measured 0); (c) the 30° wall image: ν̃, μt = c·d exact to FP32 (μt: where the clamp does not bind, −min(neighbour) where it does), k = c·d² to its interpolation error with k_g ≥ 0, the ω_w-anchored ω image; (d) SA/SST 200-step immersed URANS on 128² finite, μt 0 on deep cells, 0 guard hits with and without the wall-value convection; (e) the clamped μt image on a seeded Re 1e5 wall layer: unclamped image 51 / 21 negative faces (SA / SST), solver 0 below μ, finite with a healthy dt (unclamped SST: 2e-9) |
