# RANS — SA-neg and SST on the compressible channel

A RANS field is an ensemble-mean MODEL output: never a grounder, never DNS.
Every artefact says which closure and which mode (`--dry-run` label, RunMeta
`"rans"`, the stats header, the progress line):

| label | closure |
|---|---|
| `RANS SA-neg` | TMR SA-neg: standard SA **with ft2** for ν̃ ≥ 0, the negative branch below |
| `RANS SST-2003m` | Menter SST, 2003 constants, μt with S, ⅔ρk dropped |
| `RANS SST-Vm` | Menter 1994 SST with vorticity, ⅔ρk dropped |

Mode suffix: `-- steady (local time stepping: pseudo-time, not time-accurate)`
with `--steady`, `-- URANS (time-accurate)` without. A RANS 1D/2D run never
carries the DNS label's "2D: qualitative only" suffix — the mean really is 1D.

Built (this round): `dns channel --rans none|sa-neg|sst-2003m|sst-vm`, plus
`--steady TOL` / `--steady-steps N`. Not built: `dns flow` (O-grid, plate,
immersed bodies), wall models, the incompressible branch, MUSCL scalar
convection. `--rans` is registered only on `dns channel`, only with the
closures that exist.

Sources: TMR `spalart.html`, `sst.html` (read 2026-10-24); MKM chan180 /
chan395 `.means` (Re_τ 178.12 / 392.24; ub⁺ by trapezoid over the published
profile: 15.679 / 17.545 — the headers carry no Re_b).

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
  At J = 0 it is `comb + cr R`, the mean flow's own combination.
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

## Pseudo-time (`--steady`)

- **LTS** (D8, D9): `StageCombine::lts` is a per-cell RATIO dt_c/dtRef ≥ 1 on
  top of `cr`; `crAt(c)` is the ONLY way any kernel reads `cr` (stage
  kernels, body force, gravity, sink, NSCBC, the scalar kernels). Additive
  hooks take the whole `StageCombine`, so ignoring lts is a compile error —
  ignoring it converges to lts·R + f = 0, the wrong steady state (G7 pins the
  right one with random ratios in [1, 10]). dt_c = min(cfl/wave_c, the
  per-cell diffusive bound); dtRef = min dt_c; recomputed on `--dt-recompute`.
- **RK3 coefficient sum**: every stepper forms ca = 1 − cb, an exact sum in
  FP32 (fixed 2026-10-24 for all modes; `Real(1/3) + Real(2/3)` was 1 + 2^-25
  and grew every conserved total by 3e-8 per step — Σρ·vol drifted 1.9e-6 in
  100 vortex steps). A steady state is therefore a fixed point of the update.
- **Residual** (D10): one extra stage per stats row with ca = cb = 0, cr = 1,
  no lts, boundary hooks included, the scalar kernel in explicit mode — the
  pure rhs, never state differencing.
- **Norms** (D11), wall units L = δ = 1, U = u_τ = 1, Ue = ub estimate:
  resU = rms(|R_m|/ρ), resE = rms(R_E/ρ)/Ue, resT = SA rms(R_ν̃/ρ), SST
  max(rms(R_k/ρ), rms(R_ω/(ρω))) (ω spans six decades: its rate of ln ω).
  `resid = max(resU, resE, resT)`; `--steady TOL` stops on resid < TOL,
  `--steady-steps` (200000) is exit 8.

### The FP32 update floor — what the residual can and cannot certify

MEASURED 2026-10-24 (`gpu_rans` G5/G9, Re_τ 180, ny 64, M 0.2): the residual
of the FP64 oracle's discrete solution, uploaded to the FP32 device, is
0.025 (static FP32 floor 0.042: the spread over ±1-ulp roundings of the same
state). The march does NOT reach it. Once |R_c|·dt_c < ulp(q_c)/2 the FP32
state stops changing, so the march stalls on a DYNAMIC floor ulp(q)/(2dt_c):
resid 1–2, dominated by resE (E ≈ p/(γ−1) ≈ 1.1e4 at M 0.2, ulp 1e-3, dt 2.4e-5
in the wall cell) — reached within ~1e3 iterations even when started AT the
oracle. The plan's TOL = 10 × static floor is therefore unreachable, and
worse: the slow mode (ub) carries a residual below that floor long before it
has converged (a cold start sits at resid ~1.3 with ub 1.6% low). The
residual cannot certify convergence of the mean here; `--steady TOL` stops on
it as specified, but no TOL separates converged from unconverged on this
case. The solution itself DOES converge onto the oracle (G9: −0.16% ub after
2.2e5 iterations from a perturbed state, 30 s). **Open user decision**
(alongside U1): certify differently (an FP64 or compensated state for the
steady mode, a residual normalised by the update resolution, or larger
pseudo-time steps); none is built.

## dt with RANS (D12)

RANS-only per-cell diffusive bound (`RansDiffDt`): 0.25/(D_c amax_c²),
D_c = max((μ+μt)/ρ, γ(μ/Pr + μt/Pr_t)/ρ, SA (ν + max(ν̃,0))/σ); SST uses the
pointwise upper bound μt ≤ ρk/ω. The DNS/LES `stableDt` is byte-for-byte
unchanged. Recorded, NOT fixed: the LES global-extreme bound
(μ_eff,max/(h_min² ρ_min)) is over-conservative on a clustered mesh (never
unstable); the incompressible dt ignores ν_t and `kSmagNut` zeroes the
domain ring (LES-correct; RANS there is phase F).

## The FP64 oracle (`dns/rans_channel1d`)

The steady 1D restriction of the SAME discrete operator on
`CurvMesh::channelY` (same `CurvMetrics` in double, face formulas, ghost
algebra, body force), unknowns per row (u, T/Tw − 1, SA ν̃·Re_τ | SST ln k,
ln ω) plus the uniform p fixed by the total mass; Newton with
pseudo-transient continuation (block-Thomas for SA, dense LU for SST, whose
μt/F1 reach j±2). Temperatures enter as T − Tw: differencing T ~ Tw itself
floors the energy residual at eps·Tw/dy² (1.4e-10 measured). Converges to
resid < 1e-11 in 19–88 iterations, milliseconds.

Measured (`rans_channel1d`):

| | Re_τ 178.12, ny 64 | Re_τ 392.24, ny 96 |
|---|---|---|
| MKM ub⁺ | 15.679 | 17.545 |
| SA-neg | 15.8069 (+0.82%) | 17.5596 (+0.08%) |
| SST-2003m | 15.6332 (−0.29%) | 17.8881 (+1.96%) |
| SST-Vm | 15.5986 (−0.51%) | 17.8455 (+1.71%) |

Grid convergence at Re_τ 178.12 (stretch 2): SA-neg 2nd order (ub 15.8346 /
15.8390 / 15.8400 at ny 192/384/768, apparent order 2.17; ny 48/96/192 is
pre-asymptotic, 1.23). SST-2003m FIRST order (15.2920 / 15.2070 / 15.1662,
order 1.06, Richardson limit 15.129 = −3.5% vs MKM): the TMR wall value
ω_w ∝ 1/d1² moves with the first cell, an O(d1) boundary error. With d1 = the
first cell HEIGHT instead of its centre the order and the limit are the same
(15.365 → 15.144 over ny 48…768), only the coarse-grid error is smaller — so
the SST ny-64 MKM agreement above is partly that error; cite the limit.
SA-neg's log-layer slope over 30 < y⁺ < 0.1Re_τ (Re_τ 2000): 2.447 vs
1/κ = 2.439 (+0.32%); over the plan's 30 < y⁺ < 0.3Re_τ it is +3.2%, an
outer-layer (shear-stress) effect that does not refine away.
Laminar limit: 2nd order to the exact compressible Poiseuille (u and T).

## Tests

| test | what |
|---|---|
| `rans` | SA/SST closed forms: fv1, fw, cw1, r cap, S̃ limiter, the log-layer identity (cw1; γ1 derived vs 5/9: 0 vs 0.43%), SA-neg branch continuity, J ≥ 0, positive branch == standard SA, point-implicit fixed point, float/double, F1/F2 limits, limiters, μt branches, wall distance |
| `rans_channel1d` | the oracle: laminar order, MKM band (5% provisional ceiling, values pinned to 1e-6), log slope, ub order (SA 2nd, SST 1st), Newton < 2 s |
| `gpu_rans` | G1 lts = 1.0f bitwise; G2 ν̃ ≡ 0 (and SST k ≡ 0) bitwise to RANS off; G3 uniform scalar exact; G4 wall face μt == 0, ghost algebra (SST: ω_w on the face); G5 residual at the oracle at the FP32 floor; G6/G7 oracle fixed point (uniform / random lts); G8 SA-neg ν̃ < 0 patch; G9 (SA, SST-2003m) march onto the oracle; SST rough-field positivity, 0 guard hits |
| `channel_config` | every `--rans`/`--steady` rule, labels, describe lines, RANS-off describe unchanged |

Manual cold starts (not in ctest — they cannot exit 0, see the floor above),
`dns channel --n 4 --nz 1 --ny 64 --rans <m> --steady 1e-6 --steady-steps N`,
Re_τ 180, measured 2026-10-24 (gfx1100):

| closure | oracle ub | ub −0.3% reached | final ub (iterations) | resid plateau | wall |
|---|---|---|---|---|---|
| SA-neg | 15.8300 | ~1.2e5 it (16 s) | 15.8391, +0.06% (4e5) | 0.7–1.7 | 53 s, 0.133 ms/it |
| SST-2003m | 15.6678 | ~1.2e5 it | 15.6672, −0.004% (3e5) | 0.9–1.7 | 44 s, 0.147 ms/it |
| SST-Vm | 15.6330 | ~1.15e5 it | 15.6302, −0.02% (3e5) | 0.9–1.7 | 44 s |

All exit 8 (cap) by design; SST 0 guard hits. The u_τ fit reads Re_τ
183.6–183.7 (+2%) on every closure: the 2-point one-sided fit's own error at
y⁺_1 0.88 (the discrete wall stress is exactly 1, the oracle's τ_w to 1e-9).
