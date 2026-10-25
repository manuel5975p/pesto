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
with `--steady` (explicit scheme), `-- steady (implicit Newton pseudo-time:
not time-accurate)` with `--steady --steady-scheme implicit`, `-- URANS
(time-accurate)` without. A RANS 1D/2D run never carries the DNS label's "2D:
qualitative only" suffix — the mean really is 1D.

Built (this round): `dns channel --rans none|sa-neg|sst-2003m|sst-vm`, plus
`--steady TOL` / `--steady-steps N` / `--steady-scheme explicit|implicit`.
Not built: `dns flow` (O-grid, plate,
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
  FP32 (fixed 2026-10-24 for all modes; `Real(1/3) + Real(2/3)` was 1 + 2^-25
  and grew every conserved total by 3e-8 per step — Σρ·vol drifted 1.9e-6 in
  100 vortex steps). A steady state is therefore a fixed point of the update.
- **Residual** (D10): one extra stage per stats row with ca = cb = 0, cr = 1,
  no lts, boundary hooks included, the scalar kernel in explicit mode — the
  pure rhs, never state differencing.
- **Norms** (D11), wall units L = δ = 1, U = u_τ = 1, Ue = ub estimate:
  resU = rms(|R_m|/ρ), resE = rms(R_E/ρ)/Ue, resT = SA rms(R_ν̃/ρ), SST
  max(rms(R_k/ρ), rms(R_ω/(ρω))) (ω spans six decades: its rate of ln ω).
  `resid = max(resU, resE, resT)`, reported in the stats, NOT the stop
  (below). `--steady-steps` (1000000 explicit, 500 implicit) is exit 8.

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
residual cannot certify convergence of the mean here: no TOL separates
converged from unconverged on this case (hence the integral stop below). The solution itself DOES converge onto the oracle (G9: −0.16% ub after
2.2e5 iterations from a perturbed state, 30 s).

**The compensated state (built 2026-10-25).** `--steady` now marches a
Kahan pair per field (`DeviceCompState::lo`, `DeviceComp::enableCompensation`):
the kernels read hi, every stage write (`gpu/internal/stage_write.hpp`)
accumulates into hi + lo in the increment form a + cb (b − a) + cr R, exact
because ca == 1 − cb. Fluxes stay FP32. Plain/compensated is a COMPILE-TIME
kernel switch: a runtime branch at the write moved the NSCBC rows' FMA
contraction off the old bits; the plain instantiation is bit-for-bit the
pre-compensation solver (7 cases compared byte for byte: comp NSCBC + sink,
sponge, O-grid, conv gravity, DNS / SA / SST channel). The channel wall zeroes
its ghost cells' low words; boostX / penalizeMomentum assert a plain state.
Measured (gfx1100):

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

The plain march's ub never stops creeping (it crossed the oracle and kept
going); the compensated one decays onto a fixed point. What it does NOT do is
reach the static floor (0.042): the plateau is ~10× above it, so the
residual still certifies nothing.

**The cold starts' +0.03% above the oracle is MASS DRIFT** (explained
2026-10-25; it used to be recorded as unexplained). Local time stepping
conserves Σ V ρ / lts, not Σ V ρ, while v ≠ 0 in the transient, and the
steady equations are a one-parameter family in total mass (Σ V R_ρ ≡ 0: the
wall faces carry no mass flux), so the march lands on the member of the
family whose mass it drifted to. Measured on the device (TPRI analyses a1,
a2, reproduced 2026-10-25 from the final profile; `--n 4 --nz 1 --ny 64
--rans sa-neg --steady 1e-4`, 305800 iterations): final fluid mass 1.998440,
−7.80e-4 against the IC's 2; the oracle solved AT that mass gives ub
15.83529 against the march's 15.8338 (−9e-5, inside TOL), while against mass
2 the march is +2.4e-4. Sensitivity dub/ub per dM/M = −0.430. The oracle-start
drift is 3e-8 because a start at the fixed point has no transient v. The
implicit scheme pins the mass (below); the explicit march is unchanged — a
projection there (offline replica: 298,000 vs 305,800 iterations, ub
−9.8e-5 instead of +2.4e-4) is a pending decision, not built.

### Stopping on the integrals (built 2026-10-25)

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
their turning point then, which holds the stop off. That is the reason for
monitoring several integrals; a transient that turns every one of them at
once would still fool it.

Measured (gfx1100, Re_τ 180, ny 64, M 0.2, cold start; error = vs the march's
own fixed point extrapolated from 6e5 iterations; these stops are exactly
those predicted offline from the recorded histories):

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

## Implicit pseudo-time (`--steady-scheme implicit`, built 2026-10-25)

`dns/channel_line`, `dns/blockband`, `dns/pseudo_time`, and the implicit
branch of `gpu/runner_channel.hip`. Backward-Euler pseudo-time with defect
correction:

    (diag(1/Δτ_j) − J) Δq = R_gpu(q),   q ← P_M0(q + α Δq)

- **The right-hand side is always the GPU's**: the existing residual-mode
  stage (`DeviceComp::residual`, hooks in), gathered from column (0, ·, 0)
  (`channelGatherColumn`). J only preconditions: a drift between the line
  operator and the GPU costs iterations, never the answer. `gpu_rans_implicit`
  I1/I2 pin the drift.
- **One column, on the host, broadcast.** The RANS mean is x/z-homogeneous
  by construction (homogeneous IC, `--kick` rejected), so the host holds ONE
  FP64 column and the device holds its FP32 rounding in every column
  (`channelSetRows`, then the wall). The device state is column-uniform bit
  for bit at every iteration (I4), the y-line Jacobian is exact for it, and
  the iterate on nx 6 × nz 2 equals nx 4 × nz 1 bit for bit (I6, same cells).
  Exact ONLY for this channel: a non-homogeneous RANS case needs a new design
  (line-Jacobi plus off-line coupling).
- **Unknowns** B = 4 + nT per row: (ρ, ρu, ρv, E, ρν̃ | ρk, ρω). At w = 0,
  R_mz is linear in w and every other row even in it, so ρw decouples
  exactly and stays exactly 0 (I4: gathered R_mz ≡ 0 and w ≡ 0 at every
  iteration).
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
  any damped step, rejection or sub-cap step clears it. Unarmed, a damped
  march certifies (host test (e): α forced to 0.5 at the cap certified at
  iteration 15 with ub +6.8e-5; armed, never).
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
| CFL₀ | 100 | prototype (forward-FD J, FP32 residual), Re_τ 1e4 ny 256 SA/SST: 29/29 it from 10, 25/26 from 100, 22/23 from 1000; from 1e8 SST stalled. Host line with the central J: 27–29 from 100, 7–11 from 1e8 (36 cases) |
| growth | ×2 | ×1.5 needed 23–28 iterations to 1e-4 vs 15–18 at ×2 |
| cap | 1e8 | a 1e5 cap leaves a slow tail (SA Re_τ 1e4 +5.9e-5 / +7.7e-5) |
| CFL min | 1 | rejection never fired in any measured case |
| step bounds | ρ, E 20%; ρω −50%; ρk −90% | fired only at iteration 0 on SST |
| cap (iterations) | 500 | measured 27–28 |
| FD step | 1e-7, central | see below |

**First measurement** (before any gate; 30 iterations from the uploaded
oracle, Re_τ 180 ny 64): the stop's integrals jitter (max − min)/mean
≤ 1.42e-7 over all samples (SA 1.36e-7, SST-2003m 1.42e-7) and ≤ 7.0e-8 over
the armed ones — below 2^-21 = 4.8e-7, so the flat branch (2^-20) certifies;
resid 0.008–0.048 (the static floor).

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
as for the explicit march. For comparison the explicit march needs ~3.1e5
iterations / ~50 s for Re_τ 180 TOL 1e-4, and its default cap (1e6) cannot
reach TOL 1e-4 at Re_τ 1e4 (it needs ~1.3e6).

**Why the Jacobian is CENTRAL FD** (a deviation from the masterplan, which
specified forward): a forward step's per-entry error reached 1.6e-5 near the
centreline (Ω = |du/dy| → 0 is a curvature spike in the SA source and the
SST rate) and it is one-sided across the upwind kink max(ṁ, 0), on which
every face sits at v = 0 (the cold start and the fixed point). J·δ against a
central FD of the line was 8e-5 (SA) / 2.2e-4 (SST) per equation with the
forward form, 1.5e-7 with the central one. With the forward form pure Newton
(CFL₀ 1e8) stalled damped on SST Re_τ 1e4 ny 256 (ub +19.9% after 100
iterations — the masterplan's R-a stall); with the central form it certifies
there in 9 iterations, and in 7–11 across SA/2003m/Vm × Re_τ 180/2000/1e4 ×
ny 64–512 (host line, FP64 residual). The CFL ramp is kept as the plan's
guard; the doubled line evaluations cost < 1.4 ms per iteration.

Rejected alternatives (masterplan, reviewer reruns of the prototype):

- **J by FP32 FD of the GPU residual**: the working step is a non-monotone,
  case-dependent window — 2^-10 never certifies SST Re_τ 1e4 ny 128 (period-2
  ub oscillation ±0.4%), 2^-12 fails ny 256, 2^-14 passes.
- **K = 1 for SST** (frozen R, F2, F1): converges with a LINEAR tail, 0.70
  per iteration at the cap, and stops at −4.4e-5 (Re_τ 180); a truncated
  exact K = 1 diverges at CFL ≥ 300.
- **A 1e5 cap**: a slow tail (SA Re_τ 1e4 stopped +5.9e-5 / +7.7e-5).
- **An unarmed stop**: certified ub 1.1% off after rejections (a1 F5), and a
  damped stall (a3).

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
| `rans` | SA/SST closed forms: fv1, fw, cw1, r cap, S̃ limiter, the log-layer identity (cw1; γ1 derived vs 5/9: 0 vs 0.43%), SA-neg branch continuity, J ≥ 0, positive branch == standard SA, point-implicit fixed point (and its increment form, exactly 0 there), float/double, F1/F2 limits, limiters, μt branches, wall distance |
| `rans_channel1d` | the oracle: laminar order, MKM band (5% provisional ceiling, values pinned to 1e-6), log slope, ub order (SA 2nd, SST 1st), Newton < 2 s |
| `gpu_rans` | G1 lts = 1.0f bitwise; G2 ν̃ ≡ 0 (and SST k ≡ 0) bitwise to RANS off; G3 uniform scalar exact; G4 wall face μt == 0, ghost algebra (SST: ω_w on the face); G5 residual at the oracle at the FP32 floor; G6/G7 oracle fixed point (uniform / random lts); G8 SA-neg ν̃ < 0 patch; G9 (SA, SST-2003m) march onto the oracle; G10 compensated state (TwoSum exact, pair upload/download, plateau ≤ ½ plain, oracle fixed point, perturbed march); SST rough-field positivity, 0 guard hits |
| `channel_config` | every `--rans`/`--steady`/`--steady-scheme` rule (TOL ∈ [1e-6, 1); implicit rejects `--cfl`/`--dt-recompute`, cap 500, a given `--steady-steps` honoured), labels and describe lines of both schemes, explicit given == defaulted byte for byte, json `"steady"` only on implicit, RANS-off describe unchanged |
| `steady_stop` | the stop rule: geometric tail exact, never early; growth / drift / short / zero-scale never certified; undamped oscillation never, damped only once flat; 3e-8 noise certified; a single series fooled at a turning point, a companion holds it off |
| `blockband` | block LU vs a dense partial-pivot LU: random B 5/6 × K 1/2 × n 8/64/254 and the line matrix at CFL 100 (backward error ≤ 1e-12, agreement ≤ 1e-10); at CFL 1e8 backward errors only (the matrix is singular to ~1e-8 by Σ V R_ρ ≡ 0, so ANY two solvers differ by ~cond eps); singular / NaN / Inf → error |
| `channel_line` | (a) R at the oracle (tol 1e-11) ≤ 1e-10 momentum/k, ω 1e-13 relative, E ≤ max(1e-7, its 1-ulp floor); (b) band EXACTLY K, ghosts included; (c) J δ vs central FD ≤ 1e-6 (measured 1.5e-7); (d) host Newton from Reichardt: SA/2003m/Vm Re_τ 180 ny 64, SA/2003m Re_τ 1e4 ny 128 in 27–28 iterations, ub ≤ 7e-10 off, mass 1e-12; (e) a damped march never certified (an unarmed stop does); (f) projection: 2 ulp, mass 1e-15; host timings |
| `pseudo_time` | CFL 100·2^n to the cap at n = 20; α < 1 holds and disarms; rejection ×0.1, false below 1; armed only after an undamped step at the cap; R-b ramp/reject/regrow never arms below the cap; each step bound binds alone, SA has none |
| `gpu_rans_implicit` | I1 GPU vs line residual per equation (10× measured: ρ 2e-6, mx 3e-4, my 1e-3, E 2e-4, t 5e-4 / 7e-5); I2 line J vs GPU FP32 central FD ≤ 2e-2 (measured ≤ 1.8e-2); I3 one iteration at the oracle: dub ≤ 1.2e-10, mass exact, resid 0.016–0.031; I4 Re_τ 180 cold starts via the binary (27–28 it, ub ≤ 8e-8 off, 0 rejects / hits) + R_mz, w ≡ 0 and bitwise column-uniform state every iteration; I5 Re_τ 1e4 ny 128 (27 it, ≤ 3e-8, 0.1 s); I6 IC residual column-uniform, nx 6 nz 2 iterate == nx 4 nz 1 bitwise; I7 TOL 1e-6 |
| `gpu_runners` | `--steady 1e-3 --steady-scheme explicit` SA-neg end to end: exit 0, > 1e5 iterations (not at ub's dip), ub within 1.5e-3 of the oracle (TOL + the mass-drift offset); `--steady 1e-4 --steady-scheme implicit`: exit 0, ≤ 60 iterations (27), ub within 1e-5 of the oracle, < 2 s |

Manual cold starts, `dns channel --n 4 --nz 1 --ny 64 --rans <m> --steady
1e-6 --steady-steps N`, Re_τ 180, measured 2026-10-24 (gfx1100), BEFORE the
compensated state and the integral stop (both tables above supersede it; the
residual stop then could not exit 0):

| closure | oracle ub | ub −0.3% reached | final ub (iterations) | resid plateau | wall |
|---|---|---|---|---|---|
| SA-neg | 15.8300 | ~1.2e5 it (16 s) | 15.8391, +0.06% (4e5) | 0.7–1.7 | 53 s, 0.133 ms/it |
| SST-2003m | 15.6678 | ~1.2e5 it | 15.6672, −0.004% (3e5) | 0.9–1.7 | 44 s, 0.147 ms/it |
| SST-Vm | 15.6330 | ~1.15e5 it | 15.6302, −0.02% (3e5) | 0.9–1.7 | 44 s |

All exit 8 (cap) by design; SST 0 guard hits. The u_τ fit reads Re_τ
183.6–183.7 (+2%) on every closure: the 2-point one-sided fit's own error at
y⁺_1 0.88 (the discrete wall stress is exactly 1, the oracle's τ_w to 1e-9).
