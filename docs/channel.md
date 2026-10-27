# dns channel — wall-resolved turbulent channel flow

Compressible DNS of a periodic plane channel driven by a constant streamwise
body force, on the curvilinear wall-clustered mesh. The validation anchor is
Kim–Moin–Moser (1987) Re_tau = 180; the default grid is the KMM
192 × 128 × 160 box (4πδ × 2δ × 4πδ/3).

## Nondimensionalization (wall units)

delta = 1 (half-height), rho_wall = 1, u_tau = 1, R = 1, gamma = 1.4.

- Force density `f = rho u_tau^2 / delta = 1`, constant. The box momentum
  balance `f * (Lx * 2delta * Lz) = 2 tau_w * (Lx * Lz)` pins
  `tau_w = f * delta = 1` identically, so `--re-tau` IS the run's friction
  Reynolds number (in the mean; the instantaneous bulk flux oscillates).
- `nu = u_tau delta / Re_tau = 1/Re_tau`, constant mu (no Sutherland).
- Background pressure from the BULK Mach: the log-law estimate
  `ub ~ 2.5 ln(Re_tau) + 2.6` fixes `c = ub / Mach`, `p = rho c^2 / gamma`,
  `Twall = p` (rho = R = 1). One eddy turnover is `delta/u_tau = 1`.

Constant-force forcing is v1 (deterministic, no controller, the KMM setup);
constant-mass-flux (PI controller on the bulk flux) is documented as v2.

## Mesh: the ghost-reflection channel map

`ChannelStretchY` (remap.hpp) + `CurvMesh::channelY`. The periodic array of
ny cells holds two mirror GHOST rows (0, ny-1) and ny-2 fluid rows spanning
[0, 2delta]; the wall faces coincide with nodes 1 and ny-1. The fluid
interior is a two-sided tanh; beyond each wall the map is the exact
REFLECTION about the wall face, so the ghost cells mirror the first fluid
cells geometrically and a value mirror places u = 0 (and T = Twall) ON the
face, second order. (Reusing `stretchedY` instead puts the wall one thinnest
cell inside the box and drops the wall to first order — do not.)

Periodicity fixes the box: the grid period must be `2 delta + 2 h1` with
`h1 = ChannelStretchY::firstCell(nyTotal, beta, delta)`; the validator sizes
the Grid accordingly. `--dry-run` prints the resolved first-cell y+ and
growth ratio (defaults: y+ = 0.43, growth 1.062).

## Walls: isothermal is mandatory for long runs

The body force injects power `f * ub * V` forever. The adiabatic mirror
(comp_far's kFarWallY) has no heat sink: T and c climb until dt collapses
(exit 7). The isothermal ghost (`DeviceChannelWall`, comp_channel.hip)
conducts that power out through the wall Fourier flux: T0 = 2 Twall - T1
(floored at Twall/20), p0 = p1, rho0 = p0/(R T0), momentum antimirrored,
E0 from (p0, rho0, |u1|). `--wall adiabatic` exists for short transient
studies only. `isothermal = false` reproduces kFarWallY bit-for-bit
(test-pinned).

## Forcing and statistics

- Body force is an RHS hook on the live stage primitives (`addBodyForceX`:
  k.mx += f, k.E += f u, fluid rows only) — it rides SSP-RK3 exactly, like
  the NSCBC rhs overwrite. Never a state edit.
- `channelPlaneStats`: deterministic FP64 per-y-plane moment sums (one
  thread per row, fixed order). Host forms Favre means, Reynolds stresses,
  u_tau from a one-sided 2-point quadratic wall fit (u(0) = 0 pinned),
  bulk velocity mass-weighted with the stretched row heights.
- `<out>_stats.csv`: t, dt, ubulk, utauBot, utauTop, retau, KE (RANS and
  LES append columns, below; an LES file opens with a `# LES ...` line).
- `<out>_profiles.csv` (long format, every `--profile-dt`): t, y, yplus,
  uplus, urms, vrms, wrms, uv, rho, T — instantaneous x,z-plane averages;
  time-average offline over the stationary window.
- `yplus`/`uplus` (and the stats `utauBot`/`utauTop`/`retau`) use the FITTED
  u_τ = √(ν du/dy|_w), a 2-point one-sided wall gradient with ν = 1/Re_τ
  (ρ_b = 1), NOT the exact discrete wall units (τ_w ≡ 1, u_τ = 1). Measured
  on converged RANS states (2026-10-27): fitted u_τ +0.13% (Re_τ 1e5, ny
  2048, `--stretch 4`), +0.52% (Re_τ 1e4, ny 512), +1.0% (Re_τ 193, ny 128),
  +2% (Re_τ 180, ny 64). For a profile comparison at the 1% level divide by
  the exact u_τ = 1 instead (u⁺ = u, y⁺ = y Re_τ). At high Mach the wall
  density also departs from ρ_b = 1 (ρ_w 1.36 at M_b 1.5): wall-value units
  then need u_τ = 1/√ρ_w as well (docs/rans.md, the CKM scenario).

## IC and transition

Laminar Poiseuille scaled to the log-law bulk target plus a deterministic
divergence-free trip u' = curl(psi), envelope `(y(2d-y)/d^2)^2` (value and
slope zero at the walls), fixed low modes and phases, no RNG. Amplitude
`--kick * ub` (default 0.2). nz == 1 keeps psi_x = 0 so w stays exactly 0.
Untripped (`--kick 0`) stays laminar — that is the test oracle, not a bug.

## Measured anchors

- Laminar oracle (test_gpu_channel): ny=32, beta=1 — profile matches the
  Poiseuille parabola to 1.6e-3 Linf; extracted tau_wall 1.043 (the bias is
  the wall-adjacent cell's truncation; halves with dy: 8.5% at ny=16).
- KMM Re_tau 180 campaign: NOT yet run/measured. dt ~ 2.4e-5 at Mach 0.2
  (acoustic limit in the wall cell) -> ~1.25M steps for 30 turnovers on the
  default grid. `--mach 0.3` cuts ~35% at the price of compressibility
  drift from the incompressible reference. Do not cite KMM agreement until
  a stationary-window average has been measured and recorded here.

## RANS (`--rans sa-neg|sst-2003m|sst-vm`, `--steady TOL`)

Spec and measurements: `docs/rans.md`. The channel is the first RANS path: an
exact reflected ghost wall, an exact wall distance, no far field, and
τ_w ≡ 1 by force balance. The mean is homogeneous in x and z, so `--n 4
--nz 1` is the same answer (the validator warns above nx·nz = 4); the IC is
the untripped Reichardt profile + the model's starting scalars, `--kick`
defaults to 0 and a given `--kick > 0` is rejected. `--steady` runs local
time stepping in PSEUDO-time (rejects `--tt`, `--profile-dt`, the frame
cadences and frame `--emit`; profiles are written once, at convergence or the
cap) and needs isothermal walls. Stats append `mutMax` (SST also
`floorHits`) and, with `--steady`, `iter, resid, resU, resE, resT, Tbulk,
remaining` (implicit also `cfl, alpha, rejects`); profiles
append `mutRatio` and SA `nutilde` | SST `kplus, omegaplus`. RANS off is the
DNS channel byte for byte.

Measured (FP64 oracle = the same discrete operator; MKM ub⁺ 15.679 at Re_τ
178.12): continuum (Richardson) SA-neg 15.840 (+1.03%), SST-2003m 15.129
(−3.5%), SST-Vm 15.094 (−3.7%). SST converges FIRST order in ny (the TMR
wall ω; ub 15.776 / 15.464 / 15.292 / 15.207 / 15.166 at ny 48–768), so its
ny-64 15.633 (−0.29%) is wall-BC error, not model agreement; at Re_τ 392.24
the continuum is −1.8% (2003m) / −2.1% (Vm) and SA-neg +0.44%
(docs/validation.md). The FP32 steady march lands on the oracle (ub within 0.06% after
3–4e5 iterations, 45–55 s at ny 64) but its residual floors at ~1 (the FP32
update floor) — it cannot certify convergence; see `docs/rans.md`.

`--steady-scheme explicit|implicit` picks the pseudo-time scheme (default
`explicit`, the local-time-stepping march above). `implicit` is a coupled
Newton line solve: the host holds ONE FP64 column (the mean is x/z-
homogeneous), the GPU evaluates the residual of its FP32 broadcast, an FP64
host line operator only preconditions, the CFL ramps 100 ×2 → 1e8, and every
update is projected onto the IC's total mass. Labelled `RANS <variant> --
steady (implicit Newton pseudo-time: not time-accurate)`; `--steady-steps`
defaults to 500; `--cfl`/`--dt-recompute` are rejected; stats append `cfl,
alpha, rejects`. Measured (gfx1100, M 0.2, TOL 1e-4, `--n 4 --nz 1`): Re_τ
180 ny 64 in 27–28 iterations, Re_τ 1e4 ny 128/256 (`--stretch 4`) in 27–28,
every closure within 8e-8 of the FP64 oracle, 0.10–0.13 s wall per run.
The implicit RANS channel is the ONLY path that accepts `--mach` ≥ 1 (one
broadcast column, v = 0: shock-free by construction); the supersonic
isothermal channel of Coleman, Kim & Moser (M_b 1.5) runs in 27 iterations,
Tc/Tw 1.382 (docs/rans.md, "Scenario: the supersonic isothermal channel").

The explicit march's cold starts settle +0.03% above the oracle because local
time stepping drifts the total mass (−7.8e-4 at TOL 1e-4) and the steady
equations are a one-parameter family in it; the implicit scheme pins the mass
and lands on the oracle. `docs/rans.md`, "Implicit pseudo-time".

## LES (`--les CS`, `--sgs smag|sigma`)

A FILTERED field: statistics depend on the grid AND the constant, and the run
is labelled `LES <model> Cs=<CS> (filtered: grid- and Cs-dependent)` (σ:
`Csigma=`; at `--nz 1` the 2D qualitative-only note is appended) in the
`--dry-run` label line, every progress line, a `# <label>` FIRST line of
`<out>_stats.csv` and `<out>_profiles.csv` (a DNS/RANS file has none), and the
`.json` (`les` 1, `cs`, `sgs`). μt = ρ(CΔ)² op(∇u) per cell (Δ = vol^(1/3) on
the stretched mesh, `kCompSmagNutCurv`), Pr_t 0.9, and the SAME odd image
across both ghost rows as the RANS μt (`mutMirrorYHook` through
`setMutHook`): the wall face carries the molecular stress only, μt = 0.0
exactly. Time march only: `--rans` and `--steady` are rejected, σ is rejected
at `--nz 1` (D_σ ≡ 0 in 2D), `--sgs` without `--les` is rejected. Stats
append `nutMax` (max μt/μ over the fluid rows) and `nutWall` (the larger
wall's face-mean μt/μ, 0 by construction: ghost and first row are summed in
the same order); profiles append `nutRatio` (plane-mean μt/μ). `--les 0` /
absent is the DNS channel byte for byte: no SGS make argument, no hook, no
columns (`gpu_channel_les` L4).

Smagorinsky has no wall damping: |S| at the wall is u_τ²/ν, so the first cell
carries ν_t/ν ≈ (Cs Δ⁺)². σ vanishes in pure shear (`gpu_channel_les` L3: μt
≡ 0 on the laminar Poiseuille state, smag 10.1 μ on the same state).

**Measured (2026-10-27, gfx1100, W11).** 32 × 48 × 32, `--stretch 1.6`
(y⁺₁ = 2.08, Δx⁺ 70.7, Δz⁺ 23.6), `--mach 0.3`, Re_τ 180, `--kick 0.1`,
`--tt 40` (≈ 5.0e5 steps, dt ≈ 8e-5); ub⁺ = ub (the exact u_τ = 1) over
t 25–40, ± batch SE of 5 × 3 δ/u_τ batches (an UNDERESTIMATE: 3 δ/u_τ is
not many ub correlation times), against MKM 15.68:

| model | ub⁺ (t 25–40) | vs MKM | trend over window | state | wall-face μt | first-cell ν_t/ν | max ν_t/ν |
|---|---|---|---|---|---|---|---|
| none (under-resolved DNS) | 16.647 ± 0.004 | +6.2% | −0.019 (4.7 SE) | turbulent, u_rms⁺ peak 3.31 | — | — | — |
| smag 0.17 | 16.47 ± 0.14 | (+5.0%) | +0.96 (7.0 SE) | **RELAMINARIZED** (u_rms⁺ peak 0.20, v_rms⁺ 0.002; ub climbing) | 0.0 | 3.85 (the profile's maximum, at y⁺ 1) | 4.0 |
| σ 1.35 | 18.10 ± 0.03 | +15.5% | +0.17 (6.8 SE) | turbulent, u_rms⁺ peak 3.44 | 0.0 | 1.4e-5 | 7.2 (mean-profile peak 0.62 at y 0.5) |

None of the three windows passes the stationarity rule (|linear trend| < 1
SE), so these are NOT converged LES statistics and carry no band; Δx⁺ 70 is
a wall-resolved LES grid in name only. Smagorinsky's undamped wall ν_t (its
largest value sits IN the first cell) kills the turbulence at this Δ — the
expected failure, now a number. σ keeps the near-wall layer molecular but
over-predicts ub⁺ further than no model on this grid; whether that is the
model or the 40 δ/u_τ transient is not resolved by these runs. Commands:
plans/verify/impl_W11.md.

## Cost notes

Explicit acoustics at low Mach are the dominant cost: dt = cfl * dy_wall /
(c + |u|), and c = ub/Mach. Deeper stretch (smaller y+1) shrinks dt
further — `--dry-run` surfaces both before launch.
