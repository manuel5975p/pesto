# Chemistry: `dns flame` (P0 oracle + P1 premixed flame + 2D slot Bunsen + 2D jet diffusion flame + 2D candle)

Spec of the single-step premixed flame, of the two-reactant diffusion
flame (`--mode diffusion`) and of the candle (`--mode candle`, the diffusion
flame under gravity). Plan and measurements:
`plans/chem_plan.md`, `plans/impl_chem.md`. Open items: `issues.md` §30–53.

## Model C0

- One irreversible step F → P, Y = Y_F carried (Y_P = 1 − Y), unity Y_u.
- Calorically perfect, γ = 1.4 fixed, EQUAL molecular weights: R and cp do
  not depend on composition, so `Transport` and the mean-flow flux are
  untouched. Sensible enthalpies are equal, so species diffusion carries no
  enthalpy flux; the chemistry enters the energy equation only as `+q ω`.
- Le = 1: ρD = κ/cp = μ/Pr on the face, with the face's own Sutherland
  ratio (the one `faceFlux` uses).
- ω = exp(lnA − Ta/T) ρ max(Y, 0) (`dns::arrheniusRate`, one expression for
  kernel, oracle and host). The log form keeps A in FP32 range. The rate
  floors Y; the state is never clamped; out-of-bounds Y is COUNTED (`yHits`).
- Parameters: σ = Tb/Tu (`--expansion`, 5), β (`--zeldovich`, 8),
  θ = Ta/Tu = β σ²/(σ − 1) (50).

## Units (box)

ρ_u = S_L = R = 1, δ_D = λ_u/(ρ_u cp S_L) = 1, τ = δ_D/S_L = 1. Hence
μ = Pr (λ_u = cp = 3.5), A = Λ (the oracle eigenvalue), q = cp (σ − 1) Tu,
Tu = p0 = 1/(γ M²) with M = S_L/c_u (`--mach`, 0.01). The IC's burnt state
is the low-Mach Rankine–Hugoniot one: p_b = p0 − u_in²(σ − 1), u_b = σ u_in.

**Mach inflation.** A real CH4/air flame has M ≈ 1.15e-3; the acoustic dt
scales with it, so the run inflates M and says so in the label:

    DNS [premixed single-step, calorically perfect, Le=1, M_flame=0.01
    (inflated; CH4/air 1.15e-3)] species: MC-limited MUSCL on the central
    mass flux (bounded scalar transport; mean flow central)

## P0: the FP64 oracle (`dns/chem.{hpp,cpp}`, `test_flame1d`)

Phase plane p = λ dT/dx: dp/dT = m cp − λ q ω/p, shot from the hot end
(analytic start p = a(Tb − T), a² + m cp a = λ_b B) to the cold end, RK4 in
the LOGIT of s = (Tb − T)/(Tb − Tu) (steps cluster at both ends, where dx/dT
would otherwise be 1/s), bisection to round-off on the cold residual
p − m cp (T − Tu). m ∝ √A exactly, so Λ = 1/m(A = 1)². The profile T(x) is a
cubic Hermite on the (x_k, T_k, dT/dx_k) table (4th order), x = 0 at the mid
temperature.

| quantity | value |
|---|---|
| Λ (σ 5, β 8) | 3656401.0985 (N 2e4 vs 8e4: 3.3e-14) |
| δ_th/δ_D | 1.60165 |
| β 10 / β 6 | Λ 69343676.77 / 169306.6266, δ_th 1.49422 / 1.78079 |
| Sutherland, Tu 300 K | Λ 1416753.777, δ_th/δ_D 3.82628 |
| m(4A)/2m(A) − 1 | < 1e-10 |

**Against Cantera 3.2.0 on the same model** (`tools/chem/cantera_flame.py`,
reference only, never a build dependency; its output is checked in under
`tests/data/chem/`): constant-cp F/P of equal W (28.014), unity-Lewis
transport, A = 1e9 /s, Ta = θ Tu, Tu 300 K, 1 atm; the oracle reads
Cantera's own λ(T) table.

| | S_L [m/s] | oracle vs it |
|---|---|---|
| oracle | 0.128937 | — |
| Cantera finest (2423 points) | 0.128821 | +0.090% |
| Cantera Richardson O(h) | 0.128628 | +0.240% |

Cantera's ladder (178/331/631/1226/2423 points): 0.130884, 0.129893,
0.129408, 0.129013, 0.128821 — first order (upwind convection). Its default
refinement reads 0.6% off; the check uses the finest rung (gate 0.5%).

## P1 scheme (`gpu/species.{hpp,hip}`, `gpu/chem.{hpp,hip}`)

**State.** `DeviceCompState::y[kMaxY]` (ρY_k, kMaxY 4), `nY`; nY == 0 leaves
y[] empty and no kernel reads it. `StageCombine` is untouched (every hot
kernel takes it by value); species ride a separate `SpeciesCombine`.

**Stage order with species** (uniform branch of `DeviceComp::stage`):

1. `primitives(s)`
2. `kCompStage` — verbatim
3. cell pass: Y_k = ρY_k/ρ, ω from (ρ, Y_F, T) — ONE buffer per stage
4. species stage, which in the same launch adds `E += crAt q ω` on that
   SAME ω (fuel consumed and heat released are the same FP32 numbers)
5. the rhs hook

**Species flux.** ρY rides EXACTLY the mass flux of `faceFlux`
(`Real(0.5)(ρ_a + ρ_b) * Real(0.5)(u_a + u_b)`, same operand order), with the
face value MC-limited MUSCL upwind on the sign of mdot:
Y_f = Y_up + MC(Y_up − Y_upup, Y_down − Y_up)/2, and central Fick diffusion
−ρD (Y_b − Y_a)/h. A uniform Y gives F_Y = Y F_ρ; a power-of-two Y is exact
to the bit through every stage, hook and band (`gpu_flame_passive` (c)). The
limiter is deliberate, bounded dissipation on the species alone (the RANS
precedent); the mean flow stays central.

Why MC (host FD Newton, low-Mach, scouting): error S_h/S_L − 1 at 7.5 / 15 /
30 cells per δ_th, 3-grid order p:

| Y scheme | 7.5 | 15 | 30 | p |
|---|---|---|---|---|
| central (T and Y) | +5.10e-4 | +1.10e-4 | +2.70e-5 | 2.2 |
| first-order upwind | −3.88e-2 | −2.03e-2 | −1.03e-2 | 1.0 |
| van Leer | +1.036e-2 | +2.99e-3 | +7.99e-4 | 1.75 |
| MC | +1.576e-2 | +3.67e-3 | +8.83e-4 | 2.12 |

The error is the species/energy scheme mismatch (Shvab–Zeldovich violation),
not reaction-zone resolution.

**Open x faces.** The domain face flux is the boundary cell's own ρu,
carrying Y_in in and Y_c out, no diffusion (the RANS `ScalarBoundary` rule);
a face whose upwind cell is a boundary cell goes first order (its second
neighbour would wrap). y (and z at nz == 1) are periodic: no branch.

**Rhs hook order** (`runner_flame`): NSCBC → species re-carry → sink →
species sink. The NSCBC overwrites continuity on the open-face cells, so the
re-carry (`DeviceComp::speciesOpenConsistency`, the plain branch of the RANS
`kOpenScalarFix` copied onto y[]) rescales ρY there by the density the mean
flow just produced: ρY = (ρY_own/ρ_own) ρ_out. It must see that density
before the sink writes ρ.

**Species sink.** `DeviceCompSink::applySpecies` adds Y_k × the band's own
continuity term (a copy of `kCompSink`'s arithmetic on ρ): the band moves
species with the mass and never changes Y. The first version applied D4 to
ρY itself and was measured UNBOUNDED: a high-pass with mixed-sign weights,
it acts on the ratio ρY/ρ as an anti-diffusion and pushed the fresh Y = 1 to
1 + 2e-5 in the inflow band (7.5 cells, `--inflow-speed 0.98`: 52 cells out
of bounds; 0 with the band off, 0 after the change).

**dt.** `stableDt` is byte-identical; `chemStableDt` = kChemDt /
max_c A e^(−Ta/T)(1 + q Y Ta/(cv T²)), kChemDt 0.5, min-ed in by the runner.
Measured dtChem/dt = 8–32 over the P1 range: it never binds. Le = 1 puts the
species diffusion (ν/Pr) under the existing thermal bound (γν/Pr).

## Stats (`<out>_stats.csv`)

`t, dt, dtChem, SL, SLdisp, xf, dth, Tmax, mass, energy, omegaEdge, yHits,
uFresh, pFresh, TFresh, yMin, yMax` (the bunsen schema: below). SL is the consumption speed ∫ω dx / (ρY)_ref with
the fresh reference column 2 δ_th inside the inflow band; SLdisp =
u_ref − dx_f/dt; x_f the mid-temperature crossing; dth = (Tb − Tu)/max dT/dx
(parabolic peak); omegaEdge = max ω in the sink bands / max ω. Profiles
(`--profile-dt`): `t, x, T, rho, u, p, Y, omega`. Frames: `temp` (T − Tu),
`fuel` (Y), `omega` — names live in the manifest, so no viewer-format bump.
Exit 8 when x_f comes within band + 2 δ_th of either face.

`--wrinkle A --wrinkle-modes N` seeds a 2D front (`--ny` periodic): the IC
is laid row by row at x_f0 + Σ_{k=1..N} (A/√N) cos(2πky/L_y + φ_k), φ_k =
2π frac(k² · golden) (deterministic; a LINEAR phase in k sums the modes into
one bump). x momentum only, so the start relaxes acoustically. A = 0 is the
planar IC bit for bit. It seeds Darrieus–Landau; nothing about the wrinkled
front is validated (no linear-growth-rate test yet).

`yMin`, `yMax` (appended columns) give the overshoot MAGNITUDE the count
hides. Measured (2026-10-28, M 0.02, 8 cells, `--ny 256 --wrinkle 1
--wrinkle-modes 16`, tt 4, frames): Y − 1 ≤ 1.8e-6 (15 ulp of 1), never
below 0, and ONLY in the inflow row and its band (i ≤ 14): the inflow
row's Y drifts +9…+15 ulp in 2D (the 1D run's inflow row stays exactly 1;
its interior drifts −6…−8.5 ulp, below 1). Sink off: the same +12 ulp, so
not the sink twin; the limiter/RK3/diffusion are not involved (interior
excess ≤ 7 ulp). It is FP32 rounding of the open-row re-carry under the
transverse (y) fluxes, convected downstream — what lit up ~150k cells of a
long 2D run at the 1e-6 (8.4 ulp) threshold. Not material; not fixed;
counted, never clamped (issues.md §39).

## 2D slot Bunsen (`--mode bunsen`)

x is the jet axis; the x-lo face is the burner exit, y is periodic.
`dns/bunsen.{hpp,cpp}` (core: geometry, inflow profile, cone measurement),
`bunsenIC` (`flame1d`), runner branch in `runner_flame`.

**Per-row inflow.** `DeviceCompNscbc::Config::inflowU/inflowJet/inflowRho`
(one entry per (j, k) row) replace the uniform x-lo targets: u target
`inflowU + pulse · inflowJet`, entropy target the row density (or 1/(R T_row)
times the local p with `--nscbc-tinflow`). `SpeciesSetup::yInRow` replaces
`yIn` on the open face. Both are SEPARATE kernel instantiations (`Prof`,
`Row` template flags), so the uniform kernels are the shipped code; a
uniform profile through the profile kernels is bit for bit the uniform run
(`gpu_flame_passive` (f)). Profile (`bunsenInflow`): jet weight J =
½(1 − tanh((r − W/2)/e)), coflow C = ½(1 + tanh((r − W/2 − w_p)/e)), pilot
strip P = 1 − J − C; u = U_jet J + u_p P + U_co C, T/Tu = J + t_p P + t_co C,
Y = J. A burnt coflow without strips is Shvab–Zeldovich exact:
T/Tu = σ − (σ − 1) Y.

**Boundary heat release.** The NSCBC overwrite drops `crAt q ω` on the open
rows (issues.md §35); the cone burns in its inflow rows, so the bunsen hook
re-adds it right after the overwrite (`DeviceComp::speciesBoundaryHeat`):
NSCBC with a source term — LODI owns the hyperbolic part, the source rides
along. The 1D runner does not call it (keeps its bits; its boundary rate is
4.7e-16 of the peak).

**Anchoring: a burnt pilot** (measured, U_jet 5, M 0.02, 8 cells, tt 30):

| inflow beside the jet | result |
|---|---|
| coflow burnt (T/Tu = σ = 5) at 1 S_L — the default | anchored, steady |
| coflow T/Tu 4 | anchored, unsteady (SLflank 0.97–0.98) |
| coflow T/Tu 3 | blow-off at t = 1.8 |
| coflow cold inert (T/Tu 1) | blow-off at t = 0.9 |
| cold coflow + 2 δ_th cold u = 0 rim strips | blow-off at t = 0.9 |
| cold coflow + burnt strips 0.75 / 1.5 δ_th at 1 S_L | cone opens, blows through |
| cold coflow + burnt strips 2 δ_th at 5 S_L | anchored, unsteady |
| cold coflow + burnt strips 3 δ_th at 5 S_L | anchored (the demo) |

Without heat in the shear layer the jet edge has nothing that can hold an
edge flame against U_jet (the cold inflow face is no wall: the tanh edge's
velocity gradient is ~U/2e); a burnt pilot ignites the edge continuously at
the exit plane, the standard slot-DNS anchoring. An immersed ghost-wall lip
(heat loss + boundary-layer anchoring) is not built (issues.md §41).

**Drift guard (2D).** Exit 8 when: the slot axis is hot at the inflow row
(flash-back), never reaches Tmid (blow-through), the tip reaches band +
2 δ_th of the outflow, the base lifts past half the tip height, or fewer
than 4 BURNING flank rows remain (the cone opened). "Burning" = ω within
½ δ_th of the crossing ≥ 0.1 max ω — a hot pilot crosses Tmid without a
flame. The 1D guard is unchanged.

**Cone measurement** (`measureCone`, `flankMassFlux`): tip = first Tmid
crossing on the slot axis; per row the first Tmid crossings outward from
the axis; least squares of the half-width over the middle half of the cone
(base + ¼ … ¾ of the height) gives tan α (and each side). Three flank
speeds: `sinRatio` = sin α U_jet / S_L (the textbook cone angle),
`SLflank` = ∫ω over the fit rows / flank length (consumption), `mFlank` =
ρ u·n at the Tmid isotherm (kinematic mass burning flux; a steady flame
passes ρ_u S_L through every isotherm). `burnRatio` = ∫ω / the inflow face's
ρ u Y. Host test `bunsen`: the drawn cone back to 3e-6 in tan α at 8 cells,
mass flux of a uniform jet 1 to 3e-6.

**Measured** (burnt coflow at 1 S_L, W 12 δ_th, M 0.02, tt 30, mean over
t ∈ [20, 30]; window halves agree to ≤ 3e-3):

| U_jet | cells/δ_th | α (ideal) | sin α U_jet/S_L | SLflank | mFlank | burn/in |
|---|---|---|---|---|---|---|
| 3 | 8 | 20.04° (19.47°) | 1.02796 | 0.97861 | 0.98923 | 1.00173 |
| 3 | 16 | 19.90° | 1.02089 | 0.97795 | 0.98610 | 1.00047 |
| 5 | 8 | 12.08° (11.54°) | 1.04679 | 1.00698 | 1.00833 | 1.00003 |
| 5 | 12 | 12.00° | 1.03998 | 1.00365 | 1.00443 | 0.99963 |
| 5 | 16 | 11.98° | 1.03799 | 1.00291 | 1.00343 | 0.99941 |

The cone-angle relation reads 2–5% HIGH and converges to that (U 5:
+4.7/+4.0/+3.8% at 8/12/16 cells): the fresh gas decelerates ahead of the
flank (the cone's own displacement), so U_jet overstates the approach
speed. It is a flow effect, not a scheme error; the two direct flank speeds
sit within 0.3% (U 5, 16 cells) and 2.2% (U 3, both grids) of S_L. U 3's
−2.2% is grid-independent (stretch/curvature on the shorter cone, window
nearer tip and base; unattributed, issues.md §40). Cost: 0.41 ms/step at
392 × 288 (8 cells), 20 s for tt 30.

**Video.** `tools/chem/fire_video.py PREFIX --out X.mp4 --png X.png`: reads
`--emit bin` frames (`<prefix>_temp.bin`, raw float32 per frame, and
`<prefix>.json`), fire colormap on T − Tu normalised by the last frame's
burnt plateau (black → dark red → orange → yellow → near white), optional
blurred-ω glow, jet axis UP, ffmpeg libx264 yuv420p crf 18. numpy only.
Demo (cold coflow so the products glow against a dark background; anchored
on 3 δ_th burnt strips at the jet speed): `dns flame --mode bunsen --mach
0.02 --coflow-temp 1 --pilot-width 3 --pilot-speed 5 --length 70 --tt 50
--emit stats --emit bin --fdt 0.1` (592 × 336, 81591 steps, 0.49 ms/step,
40 s); at t = 49 sinRatio 0.993, SLflank 1.005, mFlank 1.003. Flicker clip:
the same with `--tt 40 --pulse 0.2 --pulse-freq 0.15` (anchored, 33 s).

## Measured (gfx1100, simjob, 2026-10-28)

Protocol: `--inflow-speed 0.98 --tt 20`, mean over t ∈ [10, 20]. The flame
SLIDES upstream at ~2% so its consumption speed averages over sub-cell
positions; held stationary (`--inflow-speed 1`) at 30 cells it PINS to the
grid (issues.md §30).

| cells/δ_th | S_L (M 0.01) | window halves |
|---|---|---|
| 7.5 | 1.016856 (+1.686%) | 1.016862 / 1.016850 |
| 15 | 1.003021 (+0.302%) | 1.003040 / 1.003003 |
| 30 | 1.000336 (+0.034%) | 1.000103 / 1.000570 |

Observed order p = log2((S7.5 − S15)/(S15 − S30)) = 2.365.

| M (15 cells) | S_L |
|---|---|
| 0.005 | 1.004144 |
| 0.01 | 1.003021 |
| 0.02 | 1.001864 (cfl 0.2: 1.001919, ΔS −5.5e-5) |

Spread 2.27e-3 (0.23%). M 0.01 → 0.02 is the O(M²) offset (k ≈ 3.9, i.e.
−3.9e-4 at M 0.01); M 0.005 sits +8.3e-4 above that fit — FP32-contaminated
(issues.md §31). Sutherland (Tu 300 K, 15 cells): +0.286%. Tmax/(σ Tu) − 1
≤ 1.5e-4, omegaEdge 4.7e-16 (the cold-boundary rate Λ e^−50), yHits 0, x_f
inside its zone, every run.

Conservation (periodic box, two flames, M 0.02): over 50 steps mass 1.0e-7,
E + qρY 2.1e-8, fuel lost vs the FP64 rate tally 5.2e-8 of the inventory;
over t = 2 the drifts are the FP32 stage-write rounding the non-reacting
twin shows at the same size (issues.md §32).

Bit identity: species off is the solver bit for bit — HEAD-vs-branch stats
(and profiles) byte-identical for `flow --physics compressible` (NSCBC +
sink + ghost wall), `conv`, `channel` 3D and `channel --rans sa-neg
--steady`; `dns bench --physics compressible --n 128` 2.34/2.34 ms/step base
vs 2.33/2.34 branch.

Cost: 15 cells, M 0.01, tt 10: 56346 steps, 0.20 ms/step, 11.3 s wall.

## 2D slot-jet diffusion flame (`--mode diffusion`)

`dns/diffusion.{hpp,cpp}` (core: model, Burke–Schumann state, inflow,
`measureJet`), `diffusionIC` (`flame1d`), the `r.diffusion` branch of
`runner_flame`, the two-reactant kernels in `gpu/species.hip`.

**Model C1.** F + s O → (1+s) P, single irreversible step, calorically
perfect, γ 1.4, equal W, Le = 1, both streams at T0. Y_F and Y_O carried
(nY 2, `SpeciesSetup::ox = 1`, `stoich = s`); products/inert are the
balance. ω = A ρ² Y_F Y_O e^(−Ta/T) (`dns::arrheniusRate2`, floors both Y);
the oxidizer loses s ω, E gains q ω (q per unit fuel), on the SAME ω.
Mixture fraction Z = (s Y_F − Y_O + Y_O2)/(s Y_F1 + Y_O2), affine in the
coupling function β = Y_F − Y_O/s (no source); Z_st = Y_O2/(s Y_F1 + Y_O2).

**Units.** ρ0 = R = U_jet = 1, length l = D0/U_jet with D0 = μ/(ρ0 Pr) (so
μ = Pr and the slot width in l IS the jet Péclet number U W/D0), T0 = p0 =
1/(γ M²), M = U_jet/c0. `--expansion σ` = T_ad(Z_st)/T0 (the flame sheet),
so q/(cp T0) = (σ − 1)/(Y_F1 Z_st); Ta = β σ²/(σ − 1) T0 with β the
Zeldovich number at T_ad; `--damkohler Da` = A ρ0 e^(−Ta/T_ad) (the rate at
the sheet, per l/U_jet), ln A = ln Da + Ta/T_ad.

**Scheme: ONE limiter for all species** (`kSpeciesStage2`, its own kernel;
`kSpeciesStage` is the premixed one, untouched). The face value is Y_up +
θ/2 (Y_down − Y_up) with θ = min over the voting species of the MC ratio
mcSlope(d1, d2)/d2 ∈ [0, 2]. It is LINEAR in Y across species, so β, Z and
any affine combination are transported by exactly that formula — a carried
Z tracer and Z(Y_F, Y_O) agree to rounding (`gpu_diffusion_mix`). Each voter
keeps its TVD bound (θ ≤ its own ratio; Sweby's region is star-shaped toward
0). A species votes iff |d2| ≥ 2^−24: a reactant consumed to ~1e−40 behind
the sheet carries sign-flipping noise whose "extrema" forced θ = 0 on every
species there (Z 5.0e−3 below the erf at the slab centre; with the
threshold the max error is 1.5e−3). A non-voter moves at most |d2| < 2^−24.
The face whose upwind cell is an open-face boundary cell takes θ = 1
(central, between the two cells).

**Inflow and anchor.** The bunsen per-row machinery carries the jet:
weights J, P, C of `bunsenInflow` (uJet = 1), mixture fraction Z = J +
Z_st P, the Burke–Schumann state of Z (Y_F = Y_F1 (Z − Z_st)/(1 − Z_st) rich,
Y_O = Y_O2 (1 − Z/Z_st) lean, T = T0 + (q/cp)(Y_F1 Z − Y_F)): the jet edge is
a burning sheet AT the exit plane — the anchor. Optional pilot strips are
burnt gas at Z_st. The IC is the inflow row extruded along x (p0, ρ = 1/t,
the row's u); the run closes it into its tip in about one flow-through.

Four boundary choices. The diagnostic is the Shvab–Zeldovich defect H =
T/T0 − 1 + (q/cp T0)(Y_F − Y_F1 Z), 0 for any mixing at Le 1 with both
streams at T0 (the first run of the jet showed TmaxIn 1.28 T_ad, H up to
+2.8 T0 at the inflow — the pressure, item 1):
1. **Outflow pressure anchor σ_out = 4** (`kDiffOutflowSigma`; 0.25
   elsewhere). The heat release is a steady volume source; at 0.25 (relax
   L/(σ c) ~ 20 l/U) the box sat at p/p0 1.67 (t = 14), every cold stream
   compressed to T/T0 1.6–1.7, the interior 1.28 T_ad. σ 4: p/p0 1.000 ±
   0.003 steady (stats column `pMean`).
2. **Species pinned on the inflow row** (`DeviceComp::speciesInflowPin`,
   ρY_k = Y_in,k ρ after the NSCBC overwrite). The stiff entropy relax holds
   that row's T; free species there mix laterally against a held T. A
   consistency choice, its isolated effect not measured (issues.md §45).
3. The row's T itself is smoothed by the transverse conduction the LODI row
   keeps against the entropy relax (etaIn c/L): at the default etaIn 100 the
   inflow row's sheet peak sits 0.35 T0 (7%) below T_BS(Z) (steady default
   jet: max |H| 0.351 at i = 0, 0.206 at i = 1, 0.074 at i = 4, 0.0095 past
   the sink band). Not fixed (issues.md §45).
4. **No boundary heat re-add, and an own-Y outflow.** The bunsen hook's
   `speciesBoundaryHeat` is NOT called: the pinned inflow sheet has ω = 0,
   and at the outflow row an open (pulsed or transient) tip burning in the
   LODI row ran away with it — T 4.3 T_ad at i = nx − 1, p/p0 swinging
   0.95–1.12, backflow (s 2, coflow 40 l). The dropped q ω there leaves
   with the gas (issues.md §35, §48). The two-reactant kernel's x-hi face
   carries the boundary cell's own Y both ways (the premixed kernel's
   backflow takes the inflow row's fractions — here the jet's fuel).

**Anchoring (defaults, M 0.05, 4 cells/l):** tt 40; "settling" = burn/in
not yet 1 at t = 40. Measured before item 4 (the outflow re-add still on;
the IC sheet reaches the outflow in every run's first flow-through), so the
map is indicative; the default case is re-measured after it
(`gpu_diffusion_jet`, below).

| edge | pilot | β | Da | result |
|---|---|---|---|---|
| 1 | — | 5 | 1000 | exit 8 at t = 0.48 (TmaxIn 2.07 T0: the thin IC sheet goes out) |
| 2 | — | 5 | 1000 | anchored; tip 66.2 l at t 40 (settling, burn/in 1.022) |
| 1 | 2 l burnt Z_st | 5 | 1000 | anchored; tip 100.6 l at t 40 (settling, burn/in 0.954), TmaxIn/T_ad 1.011 |
| **3** | — | **5** | **1000** | **anchored, steady (default): tip 51.2 l, TmaxIn/T_ad 0.986** |
| 3 | — | 6 | 1000 | anchored; tip 51.33 l, TmaxIn/T_ad 0.994 |
| 3 | — | 8 | 1000 | exit 8 at t = 1.76 (`gpu_diffusion_guard`) |
| 3 | — | 8 | 5000 | anchored; tip 51.32 l, TmaxIn/T_ad 1.0005; chemistry dt binds (3.8e−4 vs 2.0e−3) |

The edge (the exit mixing layer's tanh half-width) sets the initial
scalar dissipation: a thin sheet at the exit goes out before it thickens.
Hence the defaults edge 3 l and β 5 (`kDiffEdgeDefault`,
`kDiffBetaDefault`); β 8 needs Da ~5000 and a 5× smaller dt.

**Drift guard.** Exit 8 when TmaxIn (T past the sink bands, ≥ 4 cells in)
falls below (1 + σ) T0/2 — the inflow row holds its sheet T whether or not
the flame burns, so the global max cannot tell — or the base (lowest row
with ω ≥ 0.1 max ω) lifts past half the box.

**Stats** (`<out>_stats.csv`): `t, dt, dtChem, xTip, xBase, half, Tmax,
TmaxIn, burn, fuelIn, fuelOut, burnRatio, mass, energy, pMean, zMin, zMax,
yfMin, yoMin, hits, uJet`. `measureJet`: tip = first axis row (mean of the
two columns at yc ± h/2) where Z drops below Z_st, NaN = open tip; half =
the Z_st half-width at x_tip/2. fuelIn = the fuel through the first
interior face as the kernel forms it there (face mdot × central Y − ρD
∂Y/∂x): the base sits at the exit plane and x-diffusion carries ~3% of the
fuel in (the row's own ρuY read burn/in 1.032 steady). Frames: `temp`,
`fuel`, `oxid`, `omega`.

**Validation (gfx1100, simjob, 2026-10-28).**

*Mixing layer, coupling function* (`gpu_diffusion_mix`): a fuel slab on a
periodic 64 × 64 box at h 0.25, layer normal along (1, 1)/√2, advected at u =
(1, 0) (the shared limiter acts on both axes), isothermal (no heat release:
ρ, T, u stay uniform, the erf is exact), s 3 (Z_st 0.25), Da 50, IC the
Burke–Schumann state at diffusion age 2, run t = 20 (6071 steps); a third,
passive species carries Z.

| quantity | measured | band |
|---|---|---|
| max \|Z(Y_F, Y_O) − tracer\|, reacting | 8.285e−6 | FP32 walk 4 ulp(1) √(3 steps) = 6.4e−5 |
| same, passive, D 0.02 (cell Péclet 12.5) | 1.409e−5 | walk 4.0e−5 |
| same, per-species limiter (premixed kernel) | 1.226e−3 | > 30 × the shared (87 ×) |
| Z range | [8.882e−2, 1 − 8.828e−2] | [−1e−6, 1 + 1e−6] |
| max \|Z − analytic erf\| (row 0) | 1.460e−3 | 3e−3 |
| ω peak − analytic x_f(t), left / right | +0.264 / −0.009 cells | 1 cell |
| Z at the ω peaks | 0.25181 / 0.25153 | — |
| fuel burnt / ρ D Y_F1 \|∂Z/∂n\|/(1 − Z_st) | 1.00468 / 1.00429 | 2% |

x_f(t) solves Z = Z_st on the slab's erf pair (images ±3 periods); the
fuel burnt per flame against the Burke–Schumann diffusive flux into the
sheet — the flame-sheet position AND its consumption.

*Flame-sheet temperature* (`gpu_diffusion_heat`): a heated (σ 4, β 8, s 3)
1D layer on an open x box (NSCBC both faces, outflow σ 4: isobaric),
M 0.01, h 0.125, IC the sheet at age 4, t = 10.

| Da | steps | Tmax/T_ad | at p0 | Z at the ω peak | max \|p/p0 − 1\| | \|Z − tracer\| |
|---|---|---|---|---|---|---|
| 500 | 40611 | 0.94751 | 0.94924 | 0.25749 | 8.15e−3 | 1.82e−5 |
| 4000 | 41645 | 0.97525 | 0.97769 | 0.25170 | 1.07e−2 | 2.20e−5 |

Deficit ratio Da 500 / Da 4000 = 2.2750 (Da^(−1/3): 2); extrapolated
T(Da → ∞)/T_ad = 1.00614. Bands: Da 4000 deficit in (0, 4%), ratio in [1.6,
2.5], extrapolation within 1%, Z at the peak within 0.01 of Z_st, the tracer
within the rounding walk, Z and Y in bounds. Z, Y never left [0, 1].
The heated layer's own x_f(t) is not analytic (ρ D ∝ 1 but ρ²D varies:
no Howarth similarity), hence the isothermal mix case for the position.

The peak T is read at p0 (÷ (p/p0)^((γ−1)/γ) at that cell: the start's
box acoustic mode, ≤ 1.1%). The deficit falls as Da^(−1/3) (Liñán's
diffusion-flame regime: ratio 2 for 8× Da), and T(Da → ∞) = T_b + (T_b −
T_a) extrapolates to the flame sheet.

*The jet* (`gpu_diffusion_jet`, defaults W 16 l, s 1, σ 5, β 5, Da 1000,
M 0.05, 4 cells/l, 416 × 192, tt 80, window [60, 80]): burn/in 1.00008,
fuelOut/fuelIn 2.12e−11 (closed tip), tip 51.1713 l (halves 51.1647 /
51.1776), Z_st half-width at mid tip 7.7717 l, TmaxIn/T_ad 0.98633, p/p0
1.00162, base 0.25 (row 1: attached), hits 0. Bands: burn/in 1%, out/in
< 1e−3, tip halves 1%, base ≤ 2 h, TmaxIn/T_ad ∈ [0.95, 1.005], p/p0 1%,
no hits, tip 51.1 ± 5%. The tip is a regression pin
only: the constant-density Burke–Schumann tip W²/(16 erfinv(Z_st)²) =
70.3 l is a scale (D ∝ T shortens the heated jet). Cost 0.38 ms/step,
~15 s for tt 80.

**Video.** `tools/chem/fire_video.py` as for bunsen (it reads `temp` and
`omega`); `--gamma 3` (θ → θ³ before the colormap) darkens the warm
products so the sheet carries the light — at γ 1 a diffusion flame's plume
sits near T_ad and saturates. Demo (2026-10-28, `/home/manuel/Documents/
pesto/runs/flames/diffusion/`): `dns flame --mode diffusion --stoich 2
--length 170 --coflow-width 24 --tt 200 --emit stats --emit bin --fdt 1`
(Z_st 1/3, 712 × 256, 102703 steps, 0.49 ms/step); at t ≈ 187 tip 129.3 l,
half-width 11.58 l, TmaxIn/T_ad 0.994, burn/in 1.0006, p/p0 1.0014. Pulsed
clip: the same with `--tt 300 --fdt 0.5 --pulse 0.3 --pulse-freq 0.02`
(the tip swings 113–168 l and opens: fuel out/in up to 0.019 in a scouting
run). Rendered `fire_video.py PREFIX --crop-bands 16 --scale 2 --gamma 3
--glow 0.8` (pulsed `--first 200`). Raw frames deleted (~1.6 GB; ~2 min of
GPU to regenerate). The same case at coflow 16 put the tip at 147.4 l: the
periodic images confine the jet strongly (issues.md §47).

## 2D candle (`--mode candle`)

The diffusion model C1 and slot burner above, under gravity g along −x (x
is the burner axis: the flame points UP). Fuel leaves a low-Péclet slot at
the bottom (the "wick", Pe = W) into a slow air coflow; y periodic. Units
as the jet (U = the fuel speed = 1, l = D0/U, T0 = p0 = 1/(γM²)); g in
U²/l. `dns/diffusion.{hpp,cpp}` (`candleAmbientP`, `candleScales`,
`quiescentInflow`), `diffusionIC` (`flame1d`), the `r.candle` branch of
`runner_flame`, `kNscbcCells<…, Hydro>`, `kSpeciesStage2<Row, AmbHi>`.

**Choices (pick-and-justify).**
- *Wick = an inflow slot, no immersed body.* A species-aware ghost wall
  (impermeable species faces, Y image) is not built (§41); the slot's
  Burke–Schumann inflow sheet anchors the base (the jet's measured anchor,
  §43). The wick is the inflow plane.
- *Slow air coflow (1 U), not an open (pressure) bottom.* In a periodic box
  an open bottom is a chimney: the light column draws a box-wide draft
  ~sqrt(1.6 g L w_p/L_y) ≈ 9 U (estimate) — a candle in a flue. A fixed
  coflow keeps the inflow known; the plume's extra entrainment is drawn
  DOWN through the top (uOutMin −7…−12 U, every run), which the top
  treats as ambient air (below). Relative to the plume (|u|max 25–40 U) the
  coflow is 3–4%.
- *Lateral faces periodic*, 12 l per side (L_y = 25 l ≈ 17 flame widths).
  Open y faces need species on NSCBC openY (not built, §42/§47).

**Gravity.** THE source of `dns conv`, generalized: `addGravity(g, axis 0,
…)` adds `mx += cr(−g ρ)`, `E += cr(−g mx)` on every cell AFTER the NSCBC
overwrite (conv's axis-1 instantiation is the old `addGravityY` bit for
bit; conv stats byte-identical). It writes no ρ: the species re-carry still
sees the NSCBC density.

**Hydrostatic consistency.** The ambient column is isothermal at T0:
p_a(x) = p0 exp(−g x/(R T0)) (`candleAmbientP`, x from the inflow face).
- IC: p = p_a(x) on EVERY cell (horizontally uniform: only buoyancy starts
  the hot gas), ρ = p_a/(R T). Host test: |p/p_a − 1| ≤ 4.4e−16.
- Inflow density targets at p_a(h/2); outflow target p_a(x_out); the top's
  backflow density target ρ_a(x_out) (`Config::rhoInf`).
- NSCBC: LODI hands half of a hydrostatic dp/dx to each acoustic wave, so a
  rest state is NOT a LODI fixed point under gravity — both rows sag at g/2.
  Source-term LODI (Sutherland & Kennedy 2003): the incoming acoustic
  amplitude carries its hydrostatic value, L5 += (u + c) dp_a/dx at the
  inflow, L1 += (u − c) dp_a/dx at the outflow (`Config::hydroLo/Hi`, its
  own instantiation). The AMBIENT gradient −ρ_a g, not the local −ρ g: a
  plume crossing the top sits at the ambient pressure; a local-ρ term would
  offset its p by L (ρ_a − ρ) g/σ.
- The top with sustained backflow: the one-sided entropy/shear amplitudes
  un (…) are DOWNWIND differences when un < 0 (scouts diverged at t 12–14,
  T 1.3–1.4 T_ad on the outflow row, backflow −30 U). The Hydro top takes
  the lateral tanh blend instead (incoming: relax ρ → ρ_a, v → 0 at
  etaLat), and the two-reactant species face carries `yHi` = air in
  (`SpeciesSetup::ambientHi`) — the cell's own Y recirculated products.

**IC: a short burning column** (`--ignition 4`): Z(y) w(x), w = (1 −
tanh((x − 4)/edge))/2, as its Burke–Schumann state (a closed sheet), coflow
air above. The jet's extruded column was a box-tall thermal (W 4 scout:
vortices descending from the top, backflow −44 U, diverged at t 10.7). The
start's warm products flush in ~20 l/U (box mass 590 → 736 by t 20, flat
after): measure in t ∈ [20, 30]. `--ignition` also exists for
`--mode diffusion` (absent = the extruded row, bit for bit).

**Defaults (measured choices).** Paraffin/air s 3.4, Y_O2 0.233 (Z_st
0.0641), σ 5, β 5, Da 5000 (1000: the W 1 kernel went out, exit 8 at t
0.56), W 1, g 14 (Fr = U²/(g W) = 0.071), 8 cells/l, edge 0.5 l, coflow
1 U × 12 l per side, length 32 l, M 0.005. The aspect follows the fuel
Péclet number (2D buoyant scaling H/w ∝ Q/D): W 2 at g 7 burned a needle
(tip 24 l, half-width 2 l); W 1 at g 14 the teardrop.

**Mach.** M = U/c0 = 0.005, but the plume runs |u|max 34 U: buoyant Mach
0.17 (the dry-run's free-fall scale sqrt(2 g L (σ−1)/σ) = 28.4 U, M 0.14;
rejected above 0.3, warned above 0.15). Gravity's adiabatic fraction (g/cp)
L/(T_ad − T0) = g L (γ−1) M²/(σ−1) = 1.26e−3; the column's stratification
p_a(L)/p0 = 0.9825. A real candle's is ~1e−5: the inflated M is visible as
the 1.75% stratification only.

**Stats** (candle): the jet's 21 columns + `uMax` (max |u| over the box),
`uOutMin` (min u on the top row; < 0 = air drawn in); `pMean` is over the
ambient column's mean. Frames as the jet.

**Validation (gfx1100, simjob, 2026-10-28).**

| check | measured | band (test) |
|---|---|---|
| g = 0 vs `--mode diffusion` (W 4, 4 cells/l, ignition 4, M 0.05, tt 1) | 18 rows × 21 columns byte-identical (md5 64fa7e0d4b10) | identical (`gpu_candle_g0`) |
| quiescent box (`--quiescent`, defaults, tt 10 ≈ 37 free-fall times sqrt(W/g)) | max \|u\| 1.208e−3 U (211 rows), \|p/p_a − 1\| ≤ 3e−6, hits 0 | \|u\| < 5e−3 U, 1e−4, 0 (`gpu_candle_rest`) |
| same, the hydrostatic waves zeroed (temporary build) | \|u\| 0.666 U (t 0.33), 0.24 at t 3, p/p0 +0.79%, 26600 Z hits | — (the band's teeth) |
| flame height vs fuel flow Q = W (below) | H ∝ Q^1.19 over W 0.5–1; local 1.115 (0.5–0.75), 1.299 (0.75–1) | 0.75–1 exponent in [1.15, 1.45] (`gpu_candle_height`) |

Flame height = the axis Z_st crossing (`measureJet` tip), g 14, defaults,
tt 30, mean over t ∈ [20, 30]:

| W = Q | Fr | tip (halves) | burn/in | \|u\|max | uOutMin |
|---|---|---|---|---|---|
| 0.5 | 0.143 | 4.7222 (4.7200 / 4.7243) | 0.98642 | 24.61 | −7.59 |
| 0.75 | 0.095 | 7.4218 (7.4116 / 7.4321) | 1.01531 | 29.44 | −8.20 |
| 1 | 0.071 | 10.7853 (10.8462 / 10.7247) | 1.03118 | 34.23 | −9.15 |
| 1.25 | 0.057 | unsteady, 9.41–15.14 | 1.04015 | 37.28 | −10.46 |
| 1.5 | 0.048 | unsteady, 9.37–22.98 | 1.04332 | 39.81 | −9.73 |

Roper's correlations (laminar jet flames): circular ports H ∝ Q; SLOT
burners buoyancy-controlled H ∝ (Q/L)^(4/3), momentum-controlled H ∝ b
(Q/L)² / J (∝ Q² at fixed U). The steady 2D candle reads 1.19 overall and
its local exponent RISES toward 4/3 with the flame's height (1.12 → 1.30):
buoyancy-controlled, the far-field law approached from the near-source
(Pe ≤ 1, axial diffusion) side; momentum control (2) is excluded. (A
"Q^(2/3)" slot law does not exist in Roper; 4/3 is his slot exponent.)

**Unsteadiness / flicker** (report, not tuned). W ≤ 1: steady (tip std
0.11 / 0.27 l in t ∈ [15, 30]; no line near the flicker scale). W 1.5
(the second demo, tt 40): a SINUOUS flapping with tip pinch-off, periods
4.26 / 4.07 / 4.14 l/U (t 27–40), f ≈ 0.24 U/l. Cetegen & Ahmed's pool-fire
f = 1.5 D^(−1/2) Hz (D in m) is f = 0.479 sqrt(g/D) in any units (1.5 =
0.479 sqrt(9.81)): 1.46 U/l with D = W, 0.95 with D = the flame width ~3.5 l
— measured 0.165× / 0.25× of it. W 1.25 showed only one ~13 l/U cycle in
t ∈ [10, 30]. The periodic box's recirculation (air drawn down the top,
−8…−12 U) is not separated from the flame's own instability (§51).

**Demo** (`/home/manuel/Documents/pesto/runs/flames/candle/`):
`dns flame --mode candle --tt 30 --emit stats --emit bin --fdt 0.1` (288 ×
200, 293206 steps, 0.35 ms/step, 103 s); at t = 30 tip 10.95 l, half-width
1.454 l at tip/2, burn/in 1.033, fuel out/in 0.017, TmaxIn/T_ad 0.977,
p/p_a 0.9993, |u|max 34.4, uOutMin −9.1. Rendered `fire_video.py PREFIX
--height 184 --width 112 --pad-bottom 8 --smooth 4 --gamma 3 --glow 0.6`
(`candle.mp4`, `candle_final.png`: the ignition fireball rises out of frame,
the teardrop settles — dark fuel-rich core at the wick, bright sheet,
orange plume). Flapping clip: the same with `--slot-width 1.5 --tt 40`,
rendered `--first 100 --height 232 --width 128 --smooth 3`
(`candle_w15.mp4`). Raw frames deleted (~600 MB; ~4 min GPU to
regenerate). New `fire_video.py` options: `--height`, `--width`,
`--pad-bottom`, `--smooth` (ffmpeg bicubic upscale).

## Tests

| test | kind | content |
|---|---|---|
| `flame1d` | host | oracle convergence, √A similarity, profile, vs Cantera |
| `flame_config` | host | grid, IC, dry-run label/Λ, every rejection by name |
| `gpu_flame_passive` | GPU API | species off/on bitwise mean flow, uniform Y, passive copies, boundedness |
| `gpu_flame_conservation` | GPU API | mass, E + qρY, fuel vs rate tally |
| `gpu_flame1d_grid` / `_mach` / `_suth` | GPU binary | the tables above |
| `bunsen` | host | inflow profile, cone distance, measureCone / flankMassFlux on drawn cones (8, 16 cells; U 3, 5), the burning qualification |
| `gpu_flame_passive` (f) | GPU API | uniform per-row inflow (u, ρ, Y) bit for bit the uniform targets |
| `gpu_bunsen_cone` | GPU binary | U 3, 5 at 8 cells: anchored, steady, sinRatio ∈ [1, 1.07], SLflank 3%, mFlank 2%, burn/in 1%, symmetric |
| `gpu_bunsen_anchor` | GPU binary | cold coflow exits 8; fast burnt strips + cold coflow anchor |
| `diffusion` | host | Z_st, the Burke–Schumann sheet (Shvab–Zeldovich, bounds), erfcInv, the sheet tip, inflow/IC, defaults, dry-run label, measureJet on a drawn jet, every rejection by name |
| `gpu_diffusion_mix` | GPU API | coupling function to rounding (vs per-species 87 ×), bounds, Z vs erf, x_f(t) and the sheet's fuel flux |
| `gpu_diffusion_heat` | GPU API | Tmax vs T_ad at Da 500 / 4000, Da^(−1/3) deficit ratio, extrapolated T_ad, Z at the ω peak |
| `gpu_diffusion_jet` / `_guard` | GPU binary | the default jet: fuel balance, closed tip, steady, attached, T, p, bounds, tip pin; β 8 exits 8 |
| `candle` | host | the ambient column, the groups (Fr, free fall, adiabatic fraction, stratification, flicker scale), defaults, dry-run, label, the hydrostatic IC (p = p_a), the ignition column, quiescent IC, g = 0 IC == the diffusion IC bitwise, warnings, every rejection by name |
| `gpu_candle_rest` / `_g0` / `_height` | GPU binary | quiescent box at rest (\|u\| < 5e−3); g = 0 == `--mode diffusion` byte for byte; H(W 1)/H(W 0.75) exponent in [1.15, 1.45], both closed and steady |
| `chamber_config` | host | polygon SDF, regions (closed cavity, probes), IC, isentropic inverse, dry-run, every rejection by name |
| `gpu_chamber` | GPU binary | inert wall impermeability; reacting balances, nozzle vs isentropic, anchoring |

## Chamber (`--mode chamber`)

A closed 2D combustion chamber, model C0 as above, at M_flame 0.02
(inflated; the label says so). Files: `dns/chamber.{hpp,cpp}` (geometry,
regions, IC, isentropic nozzle), `gpu/runner_chamber.{hpp,hip}`, the walled
instantiation of `kSpeciesStage` (`SpeciesSetup::solid`), validation in
`config.cpp` (`finishChamber`, `chamberFlagScope`), tool
`tools/chem/chamber_video.py`.

**Geometry** (lengths in δ_th, `--chamber-height` 15, `--chamber-length`
30): back wall, straight section, linear converging nozzle (length Hc/2,
`--nozzle-ratio` Hc/He 3), walls 1.5 δ_th, one simple C-shaped polygon; a
solid wedge flame holder (`--holder-height` 4, length 0.75 of it, apex 3.75
δ_th past the inner back face). ONE immersed ADIABATIC ghost wall
(`DeviceCompGhost`, stage hook `apply`, rhs hook `applyRhs`: impermeable
faces). Ambient: cold inert gas (Tu, Y = 0) in a coflow (`--coflow` 2 S_L),
x NSCBC (inflow at the coflow, outflow anchored at p0) + sink, y periodic,
`--clearance` 10 δ_th above and below, `--downstream` 40 δ_th of jet.

**Injector = volumetric mass-flow source.** The task's cheap choice was the
whole back face; it is the whole back wall here, as a band 1.25 δ_th deep:
ρ += S, E += S cp Tu (the injected gas's enthalpy at rest), ρY += S, S
uniform per band cell, total ṁ = ρ_u `--inject-speed` Hc (3 S_L), sin² ramp
over `--inject-ramp` (1). Why not an NSCBC face: the chamber's back wall
then lies on the x-lo face with ghost rows inside the characteristic rows
and sink band, and the species open-face rule has ONE yIn per run (the inert
ambient needs 0) -- the source keeps the body away from every open face and
every shared kernel. Held for the whole step: an SSP-RK3 step adds exactly S
dt (weights sum to 1); `injCum` is its FP64 sum.

**Species at the wall** (`SpeciesSetup::solid`, a device mask, compiled
only into the `kSpeciesStage<true>` instantiation; null = the old kernel):
a fluid/solid face carries NO species flux (convective or diffusive -- the
impermeable, zero-gradient wall `applyRhs` makes for the mean flow), a solid
MUSCL second neighbour drops the face to first order, a solid cell gets no
species rhs and no heat release (ρY = 0 there from the IC on, so ω = 0).

**IC (ignition, no spark):** cavity fresh (Y = 1, Tu) up to the holder base,
burnt behind it (oracle profile across x = holder base), all at rest at p0;
ambient at the coflow. Rhs hook: NSCBC → species re-carry → sink → species
sink → wall faces → injector.

**Stop guard:** exit 8 on flash-back (injector-band burn > 1e-3 of the
total after ramp + 2) or blow-off (fuel through the exit > 0.5 of the
injected after ramp + 2 Lc/U); 7 on a non-finite state.

**Stats** (`<out>_stats.csv`, region sums over fluid cells,
`dns::chamberRegions`): t, dt, inj, injCum, fuel, burn, burnCav, outNoz
(fuel through the exit column), outDom (fuel through the domain faces),
mdotIn, mdotNoz, Me (mass-weighted exit Mach), pCh (fresh plenum), p0Noz
(p + ½ρ|u|² ahead of the contraction), pAmb (ambient beside the lips),
h0Noz (exit total-enthalpy flux), Tmax, mass, energy, yHits, omegaInj, fx
(fluid → body force; thrust = −fx minus the coflow drag, not separated).

### Measured (gfx1100, simjob, 2026-10-28)

Defaults (4 cells/δ_th: 364 × 152, 0.44 ms/step, tt 80 in 37 s) and the
demo (6 cells/δ_th: 530 × 228, 0.57 ms/step, tt 80 in 71 s). Anchored V
flame on the holder, complete combustion inside the chamber, every run.

| quantity | 4 cells, t ∈ [40, 80] | 6 cells, t ∈ [40, 80] |
|---|---|---|
| burn / injected | 0.9978 | 0.9986 |
| ṁ_exit / ṁ_in | 1.0010 | 0.9993 |
| q burn / (h0 flux − ṁ_in cp Tu) | 0.9989 | 0.9959 (definition unverified, issues.md C8) |
| T0_exit / Tb | 0.9982 | 0.9964 |
| p_chamber / p_amb (fresh plenum) | 1.1676 | 1.1697 |
| p0_nozzle / p_amb | 1.1521 | 1.1536 |
| p_amb / p0 | 1.0430 | 1.0431 |
| exit Mach, measured (mass-weighted) | 0.3841 | 0.3827 |
| exit Mach, isentropic from measured ṁ, T0, p_amb, He | 0.3805 | 0.3794 |
| Δp0 measured / Δp0 isentropic | 1.448 | 1.470 |
| fuel through the exit / injected | < 1e-16 | < 1e-16 |

(Simple means of the stats rows; `gpu_chamber` uses trapezoid window means
over [25, 40]: burn/inj 0.9943, ṁ 0.9949, energy 1.0018, Me 0.3844 vs
0.3808, Cd 0.832.) The 1D estimate at the validator's assumptions (T0 = Tb,
p_e = p0) gives Me 0.3963 and p0/p_amb 1.1143. The nozzle needs 1.45× the
isentropic total-pressure drop: a discharge coefficient Cd = ṁ/ṁ_ideal(p0)
of 0.83 for an exit Reynolds number u_e He/ν_b ~ 100 (thick boundary layers
in a 20-cell exit). The ambient beside the lips sits 4.3% above p0 (the
outflow's soft pressure anchor, issues.md C3).

**Inert** (`--reaction off`, tt 8): fuel − fuel0 − injCum stays within
3.3e-5 of fuel0 + injCum (grows steadily; attributed, not proven, to the
FP32 absorption bias of the injector's constant ~1.5e-3 increment into
ρY ~ 1, the same rounding error for every step of a dt-recompute block; a
diffusive wall leak
against the empty ghost would be ~1e2 per unit time, 1e4× that); nothing
reaches a domain face (outDom 1.8e-32); hot IC gas only compresses, Tmax ≤
Tb (p/p0)^(2/7) × 1.0013.

**yHits** (counted, never clamped): up to 162 (4 cells) / 178 (6 cells)
fluid cells per row, all FRESH gas with Y − 1 ≤ 4.3e-6 (demo frames: max
4.29e-6, min Y exactly 0) -- consistent with (not proven to be) the same
FP32 absorption of the injector increment into ρ and ρY, whose transport
rounds differently (KG continuity vs MUSCL species flux).

**Thermo-acoustic limit cycle.** After the start transient decays (burn
rms 0.3–0.4% at t ≈ 30), a self-excited oscillation grows: f = 2.95
(4 cells) / 2.93 (6 cells) per flame time -- grid-independent to 1% -- in
burn, exit Mach, fx and the nozzle total pressure together (the plenum
probe shows mostly its 2f harmonic). Growth rate of the burn rms (log fit
over t ∈ [40, 70]) 0.075 (4 cells, already bending into saturation) /
0.052 (6 cells) per flame time; at 4 cells it levels off near t = 70 at burn
rms 6.7%, plenum p rms 0.36% (rms over [75, 80]; still creeping up, issues.md
C9); at 6 cells it is still growing at t = 80
(burn rms 2.3%). The 1D two-temperature chamber (closed back,
fresh to the holder base, burnt to the exit, measured p and T) has
longitudinal modes 2.74 (open exit) and 3.14 (closed exit); a compact
nozzle at M_e 0.38 lies between the two, and so does the measured 2.94. The
hot-section transverse mode is 2.33. Mode not identified (issues.md C1).
With it, Tmax rises above Tb in the holder's base recirculation: 1.046 Tb
(4 cells, saturated) and 1.013 Tb (6 cells, t = 75) -- cause not isolated
(C2).

## Not built

Le ≠ 1 (`--lewis` unregistered until an oracle exists), curvilinear,
compensated, RANS/LES species (asserted), nz > 1 (`--nz` must be 1), an
immersed burner lip or wick (the chamber's walled species kernel is
premixed-only), open y faces for bunsen, the jet and the candle, an open
(pressure) inflow bottom, gravity in the premixed modes, unequal W, a
counterflow/strained-flame oracle, the low-Mach branch (`--physics
incompressible` rejected), multi-step chemistry, dns-ui (its editor covers
`flow` and `conv` only). Chamber: isothermal (cooled) walls, a
total-pressure inlet, non-premixed injection, a moving/2D-curved nozzle wall
beyond the polygon, 3D.
