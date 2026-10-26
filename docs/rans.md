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
swirl, no 3D wake modes]` on the revolved O-grid, `[immersed ghost wall:
first-order wall position, NOT body-fitted]` without `--ogrid`.

Built: `dns channel --rans none|sa-neg|sst-2003m|sst-vm`, plus
`--steady TOL` / `--steady-steps N` / `--steady-scheme explicit|implicit`;
`dns flow --physics compressible --rans ...` on the body-fitted O-grid
(2D/z-extruded and the revolved sphere wedge) and on the immersed ghost wall
(uniform or `--focal`), URANS and the explicit local-time-stepping steady
march -- whose integral stop does NOT certify on external flows (section
"dns flow" below) -- plus the laminar compressible `--steady` on `--ogrid`,
and the flow Newton-Krylov march (`--steady-scheme implicit`, O-grid at
`--nz 1` or the wedge), which reaches the same plateau but failed its stop
gate B0 too. A laminar steady run is labelled `DNS -- steady (local time
stepping: pseudo-time, not time-accurate)`. Not built: the full-revolution
3D sphere, dual-time URANS, wall models, the incompressible branch, MUSCL
scalar convection. `--rans` is registered only where it is built
(`ransBuilt(Subcommand)`), only with the closures that exist.

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

## `dns flow` on the body-fitted O-grid (built 2026-10-25)

`dns flow --physics compressible --ogrid --rans sa-neg|sst-2003m|sst-vm
[--steady TOL]`, cylinder / lens / cube / z-invariant mesh (masterplan_flowrans
P1). Wall-resolved only: the validator warns when the first-cell y⁺ estimate
(h1/D)·Re·0.38·Re^-0.1 (a bluff-body PEAK u_τ, calibrated on OpenFOAM's
0.137 U for the sphere at Re 1e4) is above 1; the run measures y⁺ (stats
`yplusMax`, `yplusMean`). Rejected by name: incompressible, `--les`/`--sgs`,
any `--rotor-*`, `--scene`, `--penalize`, `--cut-flux`, `--pot`, `--seq`,
`--noslip`, wall-mounted bodies, and immersed (non-`--ogrid`) bodies.

**Plumbing (P0, bit-identical).** `RansSetup` carries a `mutHook` (the mesh
family's μt/F1 image rows, run after μt is formed on every cell and before
any flux reads it) and a `fluidMask` (1 = real fluid cell: the SST guard
counts only these, `residual()` norms only these) in place of the channel's
y-mirror flags; `testSource` (TEST ONLY, compile-time) enters R before the
point-implicit quotient. `residualInto(s, out)` writes the pure rhs into a
caller buffer. `enableSteady(mask)` gives the laminar march its local-dt
buffers (`rans::LaminarDiffDt`, μt = 0, a separate functor so the RANS
functor's code is untouched). The ghost scalar algebra is
`gpu/internal/rans_wall.hpp` (moved verbatim from the channel); the explicit
steady driver is `gpu/internal/steady_march.hpp`, the channel migrated onto it
— all byte-identical to 2eb06ba (the masterplan's 15-case protocol).

**Wall row** (j = 0): ν̃_g = −ρ_g ν̃₁/ρ₁ (SA) and k the same; SST ω_g the
Dirichlet image of ω_w = 60 μ(T_f)/(β1 d1²)/ρ_f **per wall face** (i, k): the
face table holds 60 μ_ref/(β1 d1²) and Sutherland scales it by μ(T_f)/μ_ref
on the face temperature (the channel's constant-μ ω_w under Sutherland is a
recorded channel limitation, not changed). μt odd (the wall face is exactly
0.0, so the molecular-only tally is the full wall stress and κ_t = 0 there),
F1 even. **Far row** (j = ny−1): φ_g = w φ₁ + (1 − w) φ∞ with the mean-flow
row's own tanh weight w — a convex combination (positive), and the upwind
flux of an inflow face carries φ∞ exactly; NOT a reflection. μt and F1 an
EVEN copy of row ny−2 (SST's centre gradient on row ny−1 wraps to row 0).
Both rows zero their cells' Kahan low words on a compensated state.

**Wall distance**: exact distance to the DISCRETE wall polyline (node row
j = 1) at the node-mean centroid, `ogridWallDistance` (threaded, bitwise
thread-count independent); ghosts take their source row's d. On a circle it
sits within the polyline's sagitta R(1 − cos(π/nx)) of r − R.

**Freestream** (decision 4), in FREESTREAM units so a Mach study keeps the
inflow turbulence: SA `--rans-nutinf` ν̃∞/ν∞ (3, TMR); SST `--rans-kinf`
k∞/U∞² (2.25e-7: TMR's 9e-9 c∞² at M 0.2, Tu 0.0387%) and `--rans-mutinf`
μt∞/μ∞ (0.009) → ω∞ = ρ∞k∞/(R μ∞) (0.25 U/D at Re 1e4).

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

### The explicit steady march does not certify on the O-grid (G1b, U1/U2)

Measured 2026-10-25 (gfx1100), cylinder Re 40, `--n 256 --ny 96`, M 0.2,
`--steady 1e-4`, cap 1e6:

| run | cd at the plateau | settles to 1e-4 / 1e-5 (all three series) | ms/iteration | stop |
|---|---|---|---|---|
| laminar (`DNS -- steady`) | 1.56656 | 18600 / 27400 it | 0.10 | never certifies (exit 8 at 1e6) |
| RANS SA-neg | 1.57012 | 18000 / 25400 it | 0.24 | never certifies |
| RANS SST-2003m | 1.56656 | 18600 / 27800 it | 0.25 | never certifies |
| RANS SA-neg, lens `--aspect 4` Re 1e5, 256×128 β 5.5 | 0.084–0.088 | — (cd p-p 7%, cl −0.013 to −0.017, resid ~18) | 0.25 | never: no steady state reached |

The cylinder integrals converge, then JITTER at the FP32 floor: cd, cdf and
the wake deficit W reverse by 2.1e-6, 2.5e-6 and 3.3e-6 RELATIVE inside the
stop's window (sd 3–6e-7) — above kSteadyFlat = 2^-20 = 9.5e-7, so the
window is never "monotone to the sampling noise" and the flat branch never
fires. The turbulence integrals are flat (1e-8). Cause: the FP32 residual
floor moves the compensated state by ~1 ulp per cell, and a pressure-
integrated coefficient amplifies p's ulp by p/q∞ = 2/(γM²) ≈ 36 at M 0.2
(the channel's ub and bulk T are not amplified: 1.4e-7 there). The proposed
U2 remedy measured NOT sufficient: the mean of the last 8 iterates jitters
the same (the noise is correlated over hundreds of iterations); a 200-
iterate mean certifies cd and cdf but W still reverses by 1.9e-6. The lens
at Re 1e5 does not reach a steady state at all (non-geometric oscillation).
Per the masterplan this is the G1b stop clause: reported, no constant
loosened, nothing flipped — a decision (U1/U2) for the user. Until then a
compressible `--steady` run on the O-grid ends at its cap with exit 8, the
honest non-convergence report, and the settled plateau is read from the
stats.

**URANS** (G1c, cylinder Re 1e4, `--n 384 --ny 192 --ogrid-beta 4
--ogrid-far 30`, quiet start, 5 D/U): finite, 0 guard hits, y⁺max 0.65–0.66;
0.117 ms/step laminar, 0.275 SA-neg (2.3×), 0.325 SST-2003m (2.8×); μt/μ max
178 (SA) / 6.2 (SST) at t = 5 D/U.

## Axisymmetric wedge and the revolved O-grid (built 2026-10-26)

`dns flow --physics compressible --body sphere --ogrid --nz 1 [--ogrid-wedge
DEG]` (masterplan_flowrans P2a/P2b): the O-grid REVOLVED about x
(`buildRevolvedOGrid`), i = θ from the UPSTREAM axis (`--n` meridional
cells over [0, π], grid nx = n + 2), j = r (two ghost rows), k = φ: the
WEDGE of DEG (default 5) degrees, one real cell plane between two ghost
planes (internal nz 3). Labelled `[axisymmetric wedge DEG deg: ensemble
mean, no swirl, no 3D wake modes]` (a DNS run: no "ensemble mean"; a
time-accurate one adds `; axisymmetric unsteadiness only -- NOT the
sphere's physical shedding`). The full revolution (`--nz N`) is P4, not built.

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
- **The ghost-column GEOMETRY is a placeholder.** The masterplan's point
  reflection node(0) = 2 node(1) − node(2) has the WRONG handedness — an
  across-axis continuation runs its azimuth backwards in index space, so the
  placeholder's volume and its j/k face Jacobians are negative (measured:
  `degenerate face Jacobian at cell (0, 0, 0) axis 1`). The nodes are kept,
  and `build()` gives the placeholder columns their real neighbour's volume
  and j-/k-face tables; no real cell reads them.
- **`AxisWrap::MirrorZ`**: node planes 0..2 at φ = −3/2, −1/2, +1/2 Δφ are
  written as exact z-mirrors, node(3) is the mirror of node(0), so the two
  side faces of the real plane are exact mirrors (bitwise).
- **The wedge's azimuthal CFL width is ρ, not ρΔφ** (a deviation from the
  masterplan's "no change"): the ghost planes are slaved rotations, no
  azimuthal mode exists, and the side faces carry only hoop terms whose
  rates go as 1/ρ. `build()` scales aov[2] by Δφ on a MirrorZ axis. With the
  1/(ρΔφ) width the axis cells' diffusive dt fell below the 1e-6 collapse
  guard at 256×192×5° and 128×96×2.5° (measured, exit 7 on the first
  iterations); with ρ the same runs march, and L1/L2 reach the same plateau
  as before (cd to 6 digits) in fewer iterations (cd settled to 1e-4: L1
  31400 → 16000, L2 45200 → 32200).
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

Laminar sphere Re 100, M 0.2, wedge 5°, rFar 20 D, β 3 (G2a.2, gfx1100; the
integral stop does not certify, so these are the settled plateaus at the cap;
cd settles to 1e-4 in ~16k / 32k iterations at L1 / L2):

| level (n × ny) | cd | θ_s | L_r / D | ms/iteration |
|---|---|---|---|---|
| L1 64×48 | 1.09280 | 126.27° | 0.898 | 0.120 |
| L2 128×96 | 1.09492 | 126.43° | 0.890 | 0.145 |
| L2, wedge 2.5° | 1.09556 (+5.8e-4) | 126.38° | 0.892 | 0.148 |

Literature context only (Johnson & Patel 1999): cd ≈ 1.09, θ_s ≈ 127°, L_r
≈ 0.88 D. G2a.3 (Δφ 2.5° vs 5° cd within 1e-3): +5.8e-4, pass. L3 (256×192):
cd 1.09543, θ_s 126.47°, L_r 0.889 D (0.266 ms/it, 47k iterations to 1e-4).

**OpenFOAM parity on the identical mesh (`sph100`, docs/validation.md)**:
`--emit mesh` → `tools/parity/dnsmesh.py` → `simpleFoam`; tier B (dns cd0 =
(4 cd(M 0.1) − cd(M 0.2))/3 vs the incompressible solver): −0.49% / −0.095% /
−0.013% at L1 / L2 / L3 — gate |ΔCd| ≤ 0.5% at L2 passes and the gap closes
monotonically; θ_s within 0.2°, L_r within 0.3% at L3.

**RANS on the wedge, sphere Re 1e4 (G2b, gfx1100): the explicit march does
not reach a steady state.** M 0.2, 5°, rFar 20 D, β 4.5, cap 2e6:

| run | ms/it | outcome |
|---|---|---|
| SA-neg L1 64×48 | 0.24 | exit 8: cd oscillates between −5.5 and +6.2 with a period ~1.5e6 iterations, resid ~800 (y⁺max 0.77, μt/μ max 108, 0 hits) |
| SST-2003m L1 | 0.45 | exit 7 at 4.5e5 (diverged; 0 guard hits before) |
| SA-neg L2 128×96 | 0.31 | stopped at 5.8e5: cd between −2.3 and +7.3 |

Before the azimuthal-CFL-width change above (the 1/(ρΔφ) width, i.e. a
smaller axis dt) the same march gave L1 SA garbage (θ_s 3.7°), L1 SST exit 7
at 4.8e4, but L2 SA settled near cd 0.4248 (θ_s 87.4°, L_r 1.22 D, resid
0.045) and L2 SST near 0.4098 (θ_s 86.7°, L_r 1.25 D, resid 1.08) — against
a3's OpenFOAM SA (a different 128-cell mesh) 0.4149. Whether the width change
or the pseudo-time path is responsible is not isolated; the masterplan's
answer to a non-geometric oscillation is "stop and report" (R3/U1), and the
Newton-Krylov march that was to fix it does not converge here either (next
section). The OpenFOAM half of `sph1e4` is built (`tools/parity/sph1e4.py
case <mesh> <dir> sa|sst 1e4`); the comparison waits for a dns steady state.

## Newton–Krylov pseudo-time on `dns flow` (built 2026-10-26; stop gate B0 FAILED)

`--steady TOL --steady-scheme implicit` on `--ogrid` at `--nz 1` or the
revolved wedge (`gpu/nk`, `gpu/internal/nk_march.hpp`; masterplan_flowrans
4.7). Per Newton step `(diag(1/Δτ_c) − ∂R/∂q) Δq = R(q)`, Δτ_c = CFL dt_c:

- an FP64 master iterate; the kernels see its FP32 rounding, hooked. R is
  ALWAYS `residualInto` — the central KG stage, no added dissipation — so
  the fixed point is the explicit march's (measured: the same cd to 6
  digits, below).
- JFNK: J v = [R(q + ε D v) − R(q)]/ε on the scaled unknowns (ρ∞, ρ∞(U+c),
  ρ∞c², SA ρ∞ max(ν̃∞, ν), SST ρ∞U², ρω per cell). GMRES(40), CGS2 with FP64
  masked dots, forcing 0.1, ≤ 200 Krylov per step.
- P = diag(1/Δτ) − J₁, J₁ the FP64 dual-number derivative (`dns/dual.hpp`,
  `dns/nk_precond.hpp`) of a first-order Rusanov + compact-viscous flux, with
  the wall mirror and the wedge rotations folded into the diagonal; the
  scalars block-diagonal (upwind on the frozen KG mass flux, compact
  diffusion, the point-implicit source rate). Applied by a symmetric
  LEXICOGRAPHIC line Gauss–Seidel along the wall-normal lines (one 32-lane
  workgroup per line: the block-Thomas chain per row is one 5×5 product,
  lane-parallel).
- Continuation: CFL 10, ×2 per undamped step to 1e8, rejection (residual
  growth > 3×, non-finite, singular line pivot, GMRES breakdown) restores and
  cuts ×0.1, exit 7 below 0.1; the step length keeps ρ and E within 20%, SST
  ρω above −50% and ρk above −90% (whole-step scaling). The stop hears the
  integrals ONLY on an armed step (undamped, at the cap).

| constant | value | bracket |
|---|---|---|
| `kNkCfl0`, growth, cap, min | 10, ×2, 1e8, 0.1 | a2 prototype |
| `kNkRejectGrowth` | 3 | plan |
| `kNkRestart`, `kNkMaxKrylov`, `kNkForcing` | 40, 200, 0.1 | a2: 1e-2 stalled at the cap, 0.1 converged |
| `kNkFdStep` | **2⁻¹⁴** (the plan's 2⁻¹⁰ FAILED N1) | N1 sweep below |

Gates (gfx1100; test `gpu_nk` for N1–N3):

- **N1 (J v)**: forward ε vs central 2ε, per field, 16×12 cylinder Re 40.
  2⁻¹⁰: mean flow ≤ 4.6e-3 but SA's ν̃ 2.3e-2 (mid-transient) / 7.2e-2 (near
  the plateau) — FAIL; 2⁻¹² 3.0e-3 / 2.8e-2; **2⁻¹⁴ 1.1e-3 / 5.7e-3 (mean flow
  ≤ 3.4e-3) — PASS**, adopted per the plan's sweep rule.
- **N2 (assembly)**: the dual blocks equal a 4th-order central difference of
  F_P to 5e-12; the GPU's FP32-stored blocks equal a host FP64 restatement
  built from those central differences to 5.7e-8 of the block scale, every
  real cell, laminar and SA (the plan's 1e-10 is unreachable with FP32 block
  storage, which the plan also prescribes — the duals are what is checked at
  1e-10).
- **N3 (line-SGS)**: the GPU application equals a host dense block
  Gauss–Seidel on the downloaded P, same order, to 1.4e-7 (laminar) / 9.5e-8
  (SA).
- 2D: ρw stays exactly 0 through Newton iterations.
- **B0 (stop jitter) — FAILED.** Cylinder Re 40, 256×96, β 3, M 0.2, the
  last 30 armed iterations at CFL 1e8, (max − min)/mean, gate 2⁻²¹ = 4.8e-7:

  | case | cd | cdf | wake W | ∫ρν̃ | Krylov/step at the cap |
  |---|---|---|---|---|---|
  | laminar | 7.4e-6 | 3.8e-6 | 4.2e-5 | — | 5–8 |
  | SA-neg | 9.9e-6 | 2.6e-6 | 2.9e-5 | 2.4e-7 | 5–7 |
  | wedge laminar Re 100 L1 | 8.0e-6 | 6.5e-6 | 1.6e-3 | — | ≤ 34 |

  The march sits at the FP32 residual floor (scaled RMS ~6e-5, the explicit
  compensated march's resid in freestream units, 2.7e-3 vs 2.1e-3), and a
  Newton step at CFL 1e8 is J⁻¹ of that floor: the integrals jitter at
  ~1e-5, as the explicit march's do (2–3e-6), so `--steady` never certifies
  (`remaining` stays inf). Per the plan: STOP, report, U2 (the stop on the
  mean of the last m armed iterates; measured on the explicit march as
  insufficient at m = 8, section "dns flow").
- **N5 (fixed point)**: NK and explicit plateaus agree — cd 1.565558 vs
  1.56656 (laminar), 1.570117 vs 1.57012 (SA), wedge 1.092797 vs 1.09280.
  Neither certifies at TOL 1e-5, so the gate's stop-to-stop form is moot.
- **Not converged**: SST-2003m on the same cylinder (60 iterations: 25
  rejections, CFL bouncing 27–5e3, cd 1.5696 vs explicit 1.56656, rn 7e-4),
  the wedge sphere Re 1e4 SA (80 iterations, rn ~1, cd 0.46 ± 36%) and SST
  (α → 0 from the first step: the ω step bound against the ω_w wall jump;
  rn 1e7). G3-scale and G3-time were therefore not reached.
- Cost: 2 s per Newton step at 256×96 (the lexicographic sweep is 2 × 256
  sequential line launches per Krylov iteration); the laminar cylinder
  reaches its plateau in ~21 steps (40 s) — the explicit LTS march settles cd
  to 1e-4 in ~2 s. Red-black lines (measured, then removed — the plan
  forbids them): every step at the cap ran the full 200 Krylov with GMRES
  stalled (rel 0.99), jitter 1.4e-4.

## Immersed-wall RANS (built 2026-10-26; labelled; two P5 gates FAILED)

`dns flow --physics compressible --rans ...` WITHOUT `--ogrid`: the body is
the compressible sharp ghost wall on the uniform or `--focal` box. Labelled
`[immersed ghost wall: first-order wall position, NOT body-fitted]`
everywhere — a staircase wall, never a code-to-code number when
under-resolved (the validator's y⁺ estimate uses h_fine/2).

- The closure lives on the curvilinear path: without `--focal` the solver
  runs on `CurvMesh::uniform` (RANS only; RANS-off keeps the Cartesian
  solver bit for bit). Measured cylinder Re 1000, 384², gfx1100: RANS-off
  Cartesian 0.37 ms/step, SA 0.70, SST 0.75 (1.9–2.0×, curvilinear + RANS).
- d = |sdf| at the physical cell centre (`immersedWallDistance`; exact for
  sphere/cylinder/cube/mesh, the lens warns).
- Ghost images through the velocity's own linear image (`DeviceCompGhost::
  applyTurb`, after `apply`): ν̃_g = −a Σw ν̃_s, k_g = −a Σw k_s, ω_g = (1 +
  a)ω_w − a Σw ω_s with ω_w = 60ν_w/(β1 d_s²), d_s = d_w/a, ν_w from the
  thermo sample — **an UNVERIFIED sub-cell choice**; deep cells ν̃ = k = 0,
  ω = ω∞; μt_g = −a Σw μt_s and F1_g = Σw F1_s (`mirrorMut`); a compensated
  state's ghost and deep low words zeroed.
- Open domain faces (x always, y/z with `--open`): `ScalarBoundary`, a
  compile-time branch of the scalar kernels — mdot = ρ_c u_c·S of the
  boundary cell, φ∞ in, φ_c out, no diffusion. The sponge/strip relaxes the
  scalars to ρφ∞ with the density's strength; the free-slip mirror rows copy
  scalars, μt and F1 even. NSCBC and the sink stay mean-flow only.
- The body tally carries μt through the solver's own curvilinear flux; stats
  add `mutMax, yplusMax, yplusMean, floorHits` (y⁺ the linear-sublayer
  estimate sqrt(|u_c| d_c/ν_c) per interface face); immersed `--steady` (with
  `--rans` only) runs the explicit LTS march on the compensated state,
  stopped on cd and the turbulence integrals.

Gates (`gpu_rans_immersed`, gfx1100):

- **G2-flow — PASS**: focal + ghost wall + NSCBC, SA ν̃ ≡ 0 and SST k ≡ 0:
  mean flow AND body tally bit for bit the RANS-off run's over 50 steps.
- **Wall-linear image — PASS**: a plane tilted 30° at a 0.37-cell offset,
  ν̃ = k = μt = c·d: every ghost reproduces c·sdf to 2.6e-7 (c h); the ω image
  is ω_w-anchored.
- **G3-flow — boundary clause FAILED**: a uniform 2⁻¹³ specific scalar rides
  a non-uniform flow EXACTLY (0 ulp) in the interior, but the NSCBC columns
  drift 1.36e4 ulp (8e-4 relative) in 100 steps (plan: ≤ 2 ulp) and the
  drift convects downstream. Cause: NSCBC overwrites the boundary cells'
  continuity rhs with the characteristic form, which no scalar FLUX — the
  plan's ρ_c u_c·S included — can match. A consistent form would put the
  scalar's boundary rhs on φ R_ρ(NSCBC) (non-conservative there); not built
  without a decision. Pinned in the test as a measured ceiling, not a pass.
- **SST guard — FAILED (open finding)**: SST trips the COUNTED k guard in a
  few fluid cells beside the staircase (710 hits in 200 steps on a 128²
  cylinder, 4 cells; 2.6e4 over 8600 steps at 384² Re 1000). The interface
  faces carry a nonzero mass flux (the face-average velocity is not the
  wall's), which upwinds the antimirrored ghost k into the fluid. SA-neg
  takes ν̃ < 0 by design: 0 hits. Reported in `floorHits`, never silent.
- Not run (the phase stopped): the immersed channel oracle (heavy tier), the
  uniform-curvilinear/focal/sinusoidal MMS levels, the momentum-budget tally
  test (only its ν̃ ≡ 0 bitwise form above).
- Measured: immersed cylinder Re 40 SA `--steady` (384²): exit 8 at 4e4
  iterations, cd 2.09 → 2.12 still drifting, resid 0.45 — the uniform mesh
  gives local time stepping no leverage.

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
| `rans` | SA/SST closed forms: fv1, fw, cw1, r cap, S̃ limiter, the log-layer identity (cw1; γ1 derived vs 5/9: 0 vs 0.43%), SA-neg branch continuity, J ≥ 0, positive branch == standard SA, point-implicit fixed point (and its increment form, exactly 0 there), float/double, F1/F2 limits, limiters, μt branches, wall distance (channel; O-grid polyline within the sagitta, ghosts, thread-count bitwise), the `ransRunLabel` strings (the channel's exactly) and `ransBuilt` |
| `rans_channel1d` | the oracle: laminar order, MKM band (5% provisional ceiling, values pinned to 1e-6), log slope, ub order (SA 2nd, SST 1st), Newton < 2 s |
| `gpu_rans` | G1 lts = 1.0f bitwise; G2 ν̃ ≡ 0 (and SST k ≡ 0) bitwise to RANS off; G3 uniform scalar exact; G4 wall face μt == 0, ghost algebra (SST: ω_w on the face); G5 residual at the oracle at the FP32 floor; G6/G7 oracle fixed point (uniform / random lts); G8 SA-neg ν̃ < 0 patch; G9 (SA, SST-2003m) march onto the oracle; G10 compensated state (TwoSum exact, pair upload/download, plateau ≤ ½ plain, oracle fixed point, perturbed march); SST rough-field positivity, 0 guard hits |
| `channel_config` | every `--rans`/`--steady`/`--steady-scheme` rule (TOL ∈ [1e-6, 1); implicit rejects `--cfl`/`--dt-recompute`, cap 500, a given `--steady-steps` honoured), labels and describe lines of both schemes, explicit given == defaulted byte for byte, json `"steady"` only on implicit, RANS-off describe unchanged |
| `steady_stop` | the stop rule: geometric tail exact, never early; growth / drift / short / zero-scale never certified; undamped oscillation never, damped only once flat; 3e-8 noise certified; a single series fooled at a turning point, a companion holds it off |
| `blockband` | block LU vs a dense partial-pivot LU: random B 5/6 × K 1/2 × n 8/64/254 and the line matrix at CFL 100 (backward error ≤ 1e-12, agreement ≤ 1e-10); at CFL 1e8 backward errors only (the matrix is singular to ~1e-8 by Σ V R_ρ ≡ 0, so ANY two solvers differ by ~cond eps); singular / NaN / Inf → error |
| `channel_line` | (a) R at the oracle (tol 1e-11) ≤ 1e-10 momentum/k, ω 1e-13 relative, E ≤ max(1e-7, its 1-ulp floor); (b) band EXACTLY K, ghosts included; (c) J δ vs central FD ≤ 1e-6 (measured 1.5e-7); (d) host Newton from Reichardt: SA/2003m/Vm Re_τ 180 ny 64, SA/2003m Re_τ 1e4 ny 128 in 27–28 iterations, ub ≤ 7e-10 off, mass 1e-12; (e) a damped march never certified (an unarmed stop does); (f) projection: 2 ulp, mass 1e-15; host timings |
| `pseudo_time` | CFL 100·2^n to the cap at n = 20; α < 1 holds and disarms; rejection ×0.1, false below 1; armed only after an undamped step at the cap; R-b ramp/reject/regrow never arms below the cap; each step bound binds alone, SA has none |
| `gpu_rans_implicit` | I1 GPU vs line residual per equation (10× measured: ρ 2e-6, mx 3e-4, my 1e-3, E 2e-4, t 5e-4 / 7e-5); I2 line J vs GPU FP32 central FD ≤ 2e-2 (measured ≤ 1.8e-2); I3 one iteration at the oracle: dub ≤ 1.2e-10, mass exact, resid 0.016–0.031; I4 Re_τ 180 cold starts via the binary (27–28 it, ub ≤ 8e-8 off, 0 rejects / hits) + R_mz, w ≡ 0 and bitwise column-uniform state every iteration; I5 Re_τ 1e4 ny 128 (27 it, ≤ 3e-8, 0.1 s); I6 IC residual column-uniform, nx 6 nz 2 iterate == nx 4 nz 1 bitwise; I7 TOL 1e-6 |
| `gpu_runners` | `--steady 1e-3 --steady-scheme explicit` SA-neg end to end: exit 0, > 1e5 iterations (not at ub's dip), ub within 1.5e-3 of the oracle (TOL + the mass-drift offset); `--steady 1e-4 --steady-scheme implicit`: exit 0, ≤ 60 iterations (27), ub within 1e-5 of the oracle, < 2 s; O-grid SA URANS (RANS columns finite, 0 hits) and the O-grid SA explicit steady march to a 2000-iteration cap (exit 8, a row every 200, finite, 0 hits) |
| `gpu_rans_mask` | the P0 plumbing on the channel: masked residual == row-range residual bitwise (every norm), `residualInto` == `residualField`, the channel mutHook odd μt / even F1, the SST guard counts only masked cells (0 vs 120 with every cell), `testSource` enters R as one FP32 add, laminar `localDt` == SST's at k ≡ 0 bitwise and the nT = 0 compensated LTS march |
| `gpu_rans_ogrid` | (a) RANS tables on a RANS-off solver change no bit; (b) ν̃ ≡ 0 / k ≡ 0 mean flow == RANS off bitwise over 50 fixed-dt steps; (c) far row φ∞ to 0.4/0.8 ulp, outer rows after 100 steps; (d) wall-face μt == 0.0, far ghost even copy, F1 even, ω_w per face to 1e-5; (e) a uniform 2^-13 scalar rides ρ exactly (0 ulp) everywhere; (f) SA/SST 200-step URANS on 256×96 finite, 0 hits; (g) compensated LTS: rows 0 and ny−1 low words 0 after every apply |
| `gpu_rans_mms` | residual MMS on the O-grid annulus, 3 levels: SA u = 0 order 1.85, SST u = 0 1.88/1.85, SA full 0.91, SA convection 0.97, SST full 0.92/0.90 (bands [p − 0.25, p + 0.5]); FP64 reference error ≤ 1e-3 of the finest; the revolved WEDGE ladder with axisymmetric fields (hoop terms): SA u = 0 1.94, SA full 0.97, SST u = 0 1.92/1.93, SST full 0.97/0.89 |
| `gpu_axisym` | the revolved wedge: (a) uniform x-freestream fixed point (3.4 ulp, outer rows 3.0); (b) side-face mass flux of a random axisymmetric state 0.02 ulp·ρvA; (c) ghost-column R_x(π) / plane R_x(±Δφ) bit-exact vs host; (d) w ≡ 0 kept (2e-10 U over 200 steps); (e) laminar Re 100 smoke; (f) SA/SST 200-step smoke, 0 hits, μt 0 on the wall faces; (g) Archimedes on a linear pressure within 0.5% (×2π/Δφ) |
| `flow_rans_config` | every `dns flow --rans` / compressible `--steady` accept, reject (by flag, exit 2) and warning; the label/turbulence/wall/y⁺/run describe lines; RANS-off describes byte-identical to HEAD's strings; explicit given == defaulted; RunMeta keys only on flow RANS/steady |
| `gpu_determinism` | + the O-grid SA URANS run twice, stats byte-identical |
| `gpu_nk` | the flow Newton–Krylov pieces: dual blocks == 4th-order FD of F_P (5e-12); N2 GPU P blocks == host FD restatement (5.7e-8, FP32 storage); N3 line-SGS == host dense block GS (1.4e-7); N1 forward vs central J v per field at 2⁻¹⁴ (the 2⁻¹⁰/2⁻¹² sweep printed); ρw ≡ 0 and the residual falls through 6 Newton steps (laminar, SA) |
| `gpu_rans_immersed` | (a) G2-flow: focal + ghost + NSCBC, ν̃ ≡ 0 / k ≡ 0 mean flow and tally == RANS off bitwise; (b) G3-flow: a uniform scalar exact (0 ulp) in the interior, the NSCBC band's drift pinned as a CEILING (the plan's 2 ulp FAILED); (c) the 30° wall-linear image exact to FP32 (ν̃, k, μt) and the ω_w-anchored ω image; (d) SA/SST 200-step immersed URANS finite, μt 0 on deep cells, SA 0 hits (SST's staircase hits counted: an open finding) |

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
