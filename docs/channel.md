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
- `<out>_stats.csv`: t, dt, ubulk, utauBot, utauTop, retau, KE (RANS
  appends columns, below).
- `<out>_profiles.csv` (long format, every `--profile-dt`): t, y, yplus,
  uplus, urms, vrms, wrms, uv, rho, T — instantaneous x,z-plane averages;
  time-average offline over the stationary window.

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
178.12): SA-neg ny 64 15.807 (+0.82%), SST-2003m 15.633 (−0.29%), SST-Vm
15.599 (−0.51%); SST converges FIRST order in ny (the TMR wall ω), to 15.13
(−3.5%). The FP32 steady march lands on the oracle (ub within 0.06% after
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

The explicit march's cold starts settle +0.03% above the oracle because local
time stepping drifts the total mass (−7.8e-4 at TOL 1e-4) and the steady
equations are a one-parameter family in it; the implicit scheme pins the mass
and lands on the oracle. `docs/rans.md`, "Implicit pseudo-time".

## Cost notes

Explicit acoustics at low Mach are the dominant cost: dt = cfl * dy_wall /
(c + |u|), and c = ub/Mach. Deeper stretch (smaller y+1) shrinks dt
further — `--dry-run` surfaces both before launch.
