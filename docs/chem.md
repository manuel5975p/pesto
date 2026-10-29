# Chemistry: `dns flame` (P0 oracle + P1 premixed flame + 2D slot Bunsen + 2D jet diffusion flame + 2D candle + 3D premixed in turbulence + LES species and TFLES)

Spec of the single-step premixed flame, of the two-reactant diffusion
flame (`--mode diffusion`) and of the candle (`--mode candle`, the diffusion
flame under gravity). Plan and measurements:
`plans/chem_plan.md`, `plans/impl_chem.md`. Open items: `issues.md` §30–53, T1–T7, L1–L6.

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

## 3D and turbulence (`--nz`, `--turb`; premixed only)

`--mode premixed` at `--nz N` (1, or ≥ 8; hz = h, z periodic) is the same
solver: every kernel already loops over three axes. Files: `dns/turb_ic`
(generator), `flame1d` (`flameTurbSpec`, `addTurbulence`),
`gpu/flame_turb` (the swept box), `kNscbcCells<…, Turb>`, `kTurbColumns`
(`gpu/chem`), the `d3` / `turb` branches of `runner_flame`.

**Bit identity.** nz == 1 and no `--turb` run the shipped instantiations:
base (pre-change tree, including the uncommitted FV work) vs branch md5 of
every output but the `.json` (whose msPerStep is a timing): premixed 1D
(stats + profiles), premixed 2D wrinkled with `--emit bin` (frames), bunsen,
diffusion, candle, chamber, `flow --physics compressible --far nscbc`,
`conv`, `channel` 3D, `channel --rans sa-neg --steady`, `hit` — 11/11
identical (2026-10-29). A 2D wrinkled flame extruded to nz 4 equals the 2D
field in every k plane bitwise after 200 fixed-dt steps, w ≡ 0
(`gpu_flame3d_extrude`). A planar 3D flame takes the 2D dt (stableDt is the
MAX over axes) and reproduces the 2D run exactly: 8 cells, M 0.02, ny 16,
nz 16 vs 1, tt 20: 30715 steps both, S_L 1.0131918 both, ΔS_L = 0
(`gpu_flame3d_planar`, gate 2e-4).

**Generator** (`dns/turb_ic.hpp`). The box is built from its Fourier modes:
mode m draws a complex Gaussian vector from SplitMix64(seed, m) + Box–Muller,
the canonical member of ±m is drawn and the other is its conjugate, then the
solenoidal projection. Shells (width k0 = min 2π/L_d) are rescaled to the
deterministic per-mode weights E(k) dk³/(4πk²) of E = `hitModelSpectrum`(k/kp)
(the `dns hit` shape, one definition), kp = √(2π)/l_t (L11 = l_t for the
continuous spectrum), summed over the REFERENCE set |k| ≤ 8 kp, my = mz = 0
excluded (every x-plane mean is then 0: no inflow mass-flux pumping); one
global factor sets u'. Serial FP64 sums, lexicographic order; FFTW c2r per
component with the cell-centre phase. A run grid holds the reference modes
with |m_d| < n_d/2: a coarser grid is the spectral truncation of a finer one.
Measured (`turb_ic`, 64³, L = 6 l_t): |k·û|/(|k||û|) ≤ 2.2e-15, û(−m) =
conj û(m) exactly, shell energies = targets to 3e-15, u' to 2e-14, plane
means ≤ 2.0e-15, FFT = direct mode sum to 7e-15; 32³ amplitudes bitwise the
64³ ones and its field the fine field at its centres to 5.9e-14 (16³: 1.2e-3,
truncated); lattice L11/l_t = 0.979 / 0.995 / 0.999 at L/l_t = 6 / 12 / 24.
The lattice value is the EXPECTATION over draws; one realization at L = 6
l_t holds few energetic modes (kp L/2π = 2.4) and scatters. The runs' box
(seed 1, 90³, h = δ_th/6; P1 verifier, numpy correlation of the dumped
`turbBox`): longitudinal L11 = 1.04 (u, x) / 0.81 (v, y) / 0.84 (w, z) l_t,
mean 0.90 l_t; component rms 1.06 / 0.98 / 0.96 u'. Seed 2: mean 0.78 l_t
(w, z 0.46). The l_t of the tables below is the spectrum's parameter, not
the realized integral scale.

**Inflow (Taylor).** The frozen box (`turbNbx` = nz cells of the run's h,
L_bx = L_z) is swept through the inflow face at the CURRENT u_in: the face
samples the box at x_b = −X(t), X = Σ u_in dt (FP64 host), linear in x between
two box columns (`turbLerp`, an explicit `fmaf`, host twin bitwise), held over
the step (as the pulse). `kNscbcCells<…, Turb>`: the u target gains du, the
inflow shear waves L3/L4 relax v, w toward dv, dw instead of 0. The IC's fresh
gas carries box(x_b = x) — the sweep's own continuation — windowed off ½(1 −
tanh((x − (x_f0 − 1.5 l_t))/(0.25 l_t))) ahead of the flame (m += ρu', E +=
the kinetic change, p kept). Inflow-row tracking of a held smooth solenoidal
row set (`gpu_flame_turb_inflow` (c), one wavelength per box): v, w residual
4.7% / 1.4% of the amplitude at 16 / 32 cells per wavelength (∝ k² — the
transverse fluxes the LODI row keeps against etaIn c/L_x); u (acoustic L5,
half that rate) 6.0% / 4.4%. A run's peak wavelength is ≥ 50 cells.

**`--hold-flame`.** At every stats row u_in ← S_ema − (x_I − x_I0)/τ_h with
x_I = ∫ρY dx/(ρ_u Y_u) the FRESH-MASS THICKNESS (stats `xFresh`; exactly
dx_I/dt = u_in − S_c, inflow minus consumption, whatever the brush's shape),
x_I0 its first-row value, S_ema the consumption speed's EMA (`kHoldEmaTau`
1), τ_h = `kHoldTau` 2, rate-limited to `kHoldMaxRate` 1 S_L/τ, clamped to
[0.5, 10] (`setFreestream`; the box sweep uses it). The consumption speed
∫ω/(ρ_u Y_u A) does not depend on the frame; the controller only keeps the
brush in the box. Measured (2026-10-29, u'_in 4, 6 cells): the first
controller held the plane-mean T crossing x_f — the brush skewed, x_f moved
DOWNSTREAM (16.9 → 19.1) while S_c rose 2.2 → 3, the controller cut u_in to
1.16 and the leading edge flashed back into the inflow band (S_c 8, I0 2.2,
omegaEdge 0.3). Hence x_I, and the 3D guard: exit 8 when the brush's leading
edge (`xLead`, the first plane-mean c ≥ 0.05 from the inflow) comes within
one δ_th of the inflow band.

**Stats at nz > 1** (appended; dth is then the plane-mean brush thickness):
`area` = ∫|∇c| dV/(L_y L_z) (co-area: the flame-surface ratio A_T/A; central
|∇Y|, 0 on the open rows; a planar flame reads 1), `I0` = SL/area
(Damköhler's first hypothesis: 1 at Le = 1), `uPrime` = √(mean component
variance) over x ∈ [x_f − 3 l_t, x_f − 1.5 l_t] (l = δ_th without
`--turb`), `uInflow`, `xFresh` (x_I above), `xLead`. Frames add `vort` (|∇×u|) and `speed` (|u|).
The `I0` COLUMN divides the raw SL (box units) by area; the tables below
first divide SL by the grid's laminar S_L (1.025 at 6 cells), so a column
mean reads ~2.5% above the tabled I0 (R1: 1.030 vs 1.006). `Tmax` is, as in
2D, the maximum of the PLANE-MEAN T over x — not the pointwise peak, which
in 3D is higher; the Tmax/Tb gate below is on the plane mean.
`--frame-start T` (every mode) delays every frame kind to T.

**Labels.** `DNS [premixed …, M_flame=…; 3D nz=N, synthetic HIT inflow
u'/S_L=…, l_t=… delta_th (frozen hash-seeded box, Taylor-swept, seed …;
Re_t=…, h/eta=… at the inflow); flame held by inflow control]`; h/η > 2 at
the inflow → `UNDER-RESOLVED DNS (h/eta=…)` and a warning. `--dry-run`
prints Re_t, η, h/η, τ_t, Ka = (u'/S_L)^1.5 (l_t/δ_th)^−0.5, Da =
(l_t/δ_th)/(u'/S_L) and the report-only Peters 1999 / Damköhler S_T/S_L.

**Measured** (gfx1100, simjob, 2026-10-29; `tools/chem/turb_stats.py`).
Cost: 5.90 ms/step at 176 × 120 × 120 (8 cells/δ_th, 2.53 M cells), 3.1 /
3.8 ms/step at 176 / 224 × 90 × 90 (6 cells). dt 6.3e-4 (8 cells, M 0.02),
8.4e-4 (6 cells, M 0.02), 4.3e-4 (M 0.01). The P1 plan's 8 cells/δ_th, 10 + 8
δ_th box did not hold the brush (below), so the runs that count are at 6
cells/δ_th (h/η ≤ 1.15 at the inflow; laminar S_L of that grid, 1D, tt 20,
window [10, 20]: 1.025126 at M 0.02, 1.026526 at M 0.01 — the normalization),
ny = nz = 90 (L_y = L_z = 15 δ_th = 6 l_t), l_t = 2.5 δ_th, downstream 16
δ_th, `--hold-flame --inflow-speed 1.2`, window t ∈ [10, 20], S_T = SL /
S_L(grid):

| run | u'_in, M | upstream | u'_flame | S_T/S_L | A_T/A | I0 | halves | Tmax/Tb | Peters | Damköhler |
|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 2, 0.02 | 8 δ_th, tt 25 | 0.917 | 1.478 | 1.469 | 1.006 | 1.306 / 1.644 | 1.0019 | 2.083 | 1.917 |
| U2 | 4, 0.02 | 16 δ_th, tt 20 | 0.879 | 1.703 | 1.692 | 1.006 | 1.421 / 1.976 | 1.0041 | 2.051 | 1.879 |
| U3 | 8, 0.01 | 16 δ_th, tt 20 | 2.492 | 3.049 | 2.923 | 1.043 | 3.307 / 2.792 | 1.0114 | 3.116 | 3.492 |

(Peters / Damköhler at the MEASURED u'_flame, l = l_t, l_F = δ_D; report-only.)
x_I held to −0.17/+0.09 (R1), −0.34/+0.11 (U2), −1.39/+0.83 δ_D (U3) of its
first value; the plane-mean x_f wandered −6.2/+5.6 δ_D in U3. R1 over its own
second half [12.5, 25]: S_T/S_L 1.790, I0 1.037, halves 1.512 / 2.060.

What holds: **Damköhler's first hypothesis, I0 = (S_T/S_L)/(A_T/A) = 1.006 /
1.006 / 1.043** (gate [0.8, 1.2]; per row 0.98–1.24 in U3), every run from the
first rows on — the consumption speed is the flame surface times S_L at Le =
1. S_T rises with u'_in (1.48, 1.70, 3.05) and sits at 0.71 / 0.83 / 0.98 of
Peters at the measured u'. What does NOT hold (issues.md T4): the runs are
NOT stationary — S_T still grows across the window at u'_flame ~0.9 (halves
+26% / +39%: a developing brush plus Darrieus–Landau at σ = 5 in a 15 δ_th box)
and falls in U3 (−16%); the window-halves gate (7%) fails in all three. S_T is
not strictly increasing with the MEASURED u'_flame: R1 and U2 reach u' 0.92 /
0.88 (the longer upstream of U2 decays its stronger inflow turbulence to the
same level) with S_T 1.48 / 1.70 — non-stationary noise at equal u', not a
trend. U3 burns unburnt pockets in the outflow band (omegaEdge up to 0.386)
and reaches Tmax/Tb 1.0114 (gate 1.01).

Failed configurations (kept for the record): (a) the plan's T2, 8 cells,
upstream 10 / downstream 8, tt 15, the first controller (x_f): window [7.5, 15]
S_T 1.92 (unnormalized), I0 1.027, u' 1.73, halves 1.58 / 2.26, brush 15–17 δ_D
past t 13.7 burning in a band (omegaEdge 0.51); (b) 6 cells, upstream 8, the
first controller: u'_in 4 and 8 flashed back into the inflow band (S_c 8, I0
2.2, omegaEdge 0.3–0.43); (c) the x_I controller, upstream 8: u'_in 4 and 8
exit 8 on the leading-edge guard at t 11.9 / 4.8 (leading edge 11–13 δ_D
ahead of x_f: the brush grows ~1–2.5 δ_D/τ) — hence upstream 16.

Regenerate: `dns flame --nz 90 --ny 90 --cells-per-delta 6 --upstream 16
--downstream 16 --hold-flame --inflow-speed 1.2 --tt 20 --turb 4 --mach 0.02`
(U2, 95 s), `--turb 8 --mach 0.01` (U3, 180 s); R1 `--upstream 8 --tt 25
--turb 2 --mach 0.02` (92 s); then `tools/chem/turb_stats.py PREFIX --t0 10
--t1 20 --sl S_L(grid) --mach M`.

## LES species and the thickened flame (`--les`, `--sgs`, `--tfles`)

Files: `dns/tfles.hpp` (constants, Charlette, Colin, sensor norm,
`TflesSetup`), `gpu/tfles.{hpp,hip}` (`DeviceTfles`, `mutCopyOpenXHook`),
the `Eddy` / `Thick` instantiations of `kSpeciesStage` and `kChemCellPass<true>`
(`gpu/species.hip`), `tflesStableDt` (`gpu/chem`), the runner and validator
branches; `tools/chem/tfles_eff.py` (the independent transcription).

**Equations** (F_max = `--tfles`, Ω ∈ [0, 1] the sensor, F = 1 + (F_max −
1) Ω and E per cell, μ_t the LES `nut` slot, Sc_t = Pr_t = 0.9 ONE constant):

    species  rho D_eff = F E mu/Pr + (1 - Omega) mu_t/Sc_t
    energy   kappa_eff = F E kappa + (1 - Omega) mu_t cp/Pr_t
    rate     (E/F) omega   -- the fuel sink and q omega, one buffer
    momentum unchanged (mu + mu_t; mu is not thickened)

The energy line is `faceFlux`'s κ + μ_t c_p/Pr_t plus an ADDITIVE face flux
−(FE_f − 1) κ_f ∂T/∂n + Ω_f μ̄_t (c_p/Pr_t) ∂T/∂n in the species launch
(κ_f with faceFlux's own Sutherland ratio, face values arithmetic means):
conservative, `faceFlux` untouched. LES without TFLES (`Eddy`): ρD + μ̄_t/Sc_t
on every interior face. Open x domain faces: no diffusion (as before).

**Sensor** (`--tfles-sensor dynamic`, the default): Ω_raw = tanh(50 ω_s/
ω_s,max), ω_s = `arrheniusRate` at Ta/2 (`kTflesTaFraction`, Legier et al.
2000), normalised on the low-Mach Shvab–Zeldovich manifold
(`tflesSensorLogMax`, FP64 bisection = a 2e6-point scan to 1e-12), then
spread TOWARD COLDER GAS by a T-gated max filter of radius ceil(2.5 F_max
δ_th/h) cells (`kTflesDilate`): Ω(c) = max of Ω_raw over the box cells n with
T_n ≥ T_c, three separable passes (x clamped at the open faces, y and z
periodic; the gate composes transitively). Why gated, measured on the
thickened oracle (host scan, F 2–4 at 1.875–3 cells): a SYMMETRIC filter that
covers c = 0.01 leaves Ω 0.03–0.6 at 2 F δ_th past c = 0.99; any symmetric
radius that keeps the burnt side clean leaves Ω < 0.05 at c 0.01–0.03. The
gated one at 2.5: Ω = 1.000000 on c ∈ [0.01, 0.99], ≤ 3.6e-5 beyond 2 F δ_th
on both sides (`gpu_tfles_sensor`; Ω ≥ 0.99 from c = 0.006 / 0.008 at F 4 /
F 2). `--tfles-sensor uniform`: Ω ≡ 1 (the laminar tests).

**Efficiency** (`--efficiency charlette`, the default with `--les`; needs
`--les`): Charlette, Meneveau & Veynante 2002, static β = 0.5, the saturated
form min[Δ/δ_L − 1, Γ u'_Δ/S_L] (Wang, Boileau & Veynante 2011), Δ = F δ_L,
δ_L = δ_th, Re_Δ = u'_Δ Δ/ν_u, C_k = 1.5; u'_Δ by Colin's operator 2 h³
|∇²(∇×u)| (Δ/h)^(1/3) (central, wrap-aware; the curl removes dilatation, so a
planar laminar flame has u' = 0 and E = 1 exactly). E = 1 on the 3 outermost
x rows (`kTflesEdgeRows`: the curl would wrap). The C++ (double) equals the
Python transcription on its 50-point table to 0.0 (same libm); the FP32
kernel instantiation within 8e-8 of FP64 (`tfles`).

**Frozen per step.** Ω, F, E, FE, E/F are formed from each step's INPUT
state (one primitives launch + 7 kernels) and held over the three stages. The
diffusive dt adds `tflesStableDt` = 0.25 h² ρ_min/(γ/Pr (μ(T_max) max FE +
max μ_t)); `stableDt` and `chemStableDt` stay byte-identical (E/F ≤ 1).

**Flame LES** (`--les CS --sgs smag|sigma`, nz ≥ 8, premixed): an explicit
`--sgs` is REQUIRED (σ is exactly 0 on a laminar flame's 1D dilatation,
Smagorinsky's |S| is not); μ_t on i = 0, nx − 1 is copied from i = 1, nx − 2
(`mutCopyOpenXHook`: the SGS gradient there differences inflow against
outflow across the wrap, §17 — measured on a u-jump box: Smagorinsky raw
0.082 vs interior max 0.017; σ 0.008, rank-1). `--tfles` without `--les`
(any nz, also `--mode bunsen` in 2D) is labelled `thickened flame TFLES [...,
no SGS model]`; the IC is the oracle profile at (x − x_f)/F_max. Stats with
`--tfles` append `Ebar` (Ω-weighted mean E) and `FEmax`; SL is then the
consumption speed of (E/F) ω = resolved area × E × S_L.

**Label** (always): `LES [sigma-model Csigma=1.35; species SGS flux mu_t/Sc_t,
Sc_t=Pr_t=0.9; mu_t on the open x rows copied from the neighbour row] + TFLES
[F=3, dynamic sensor (rate at Ta/2, tanh 50, spread 2.5 F delta_th toward
colder gas), Charlette 2002 efficiency beta=0.5 (delta_L=delta_th), u'_Delta
by Colin's operator] [premixed single-step, ..., M_flame=...]`.

**Bit identity** (gfx1100, 2026-10-29; base = the pre-P2 tree incl. the
uncommitted FV work, `build/turb_base_P2/dns`): flags absent, every output
but the `.json` byte-identical in 13 cases (premixed 1D stats + profiles,
2D wrinkled frames, 3D `--turb` volumes, bunsen / diffusion / chamber
frames, candle, `flow` compressible NSCBC, `flow` 3D LES σ, conv, channel 3D,
channel RANS, hit). `--tfles 1` (uniform AND dynamic: F = 1 + 0 Ω = 1) is the
DNS byte for byte, 1D and 3D `--turb 4` (FP32 frames/volumes + every stats
column; `gpu_tfles_bits`); a planar laminar 3D LES + TFLES 4 flame with
`--efficiency charlette` equals `none` byte for byte (curl u ≡ 0 → E ≡ 1).
Uniform ρY = ρ/2 stays bitwise under Eddy and Thick (60 steps, stirred
hot/cold box); mass / ρY / E drift equals the DNS's own FP32 drift (200
steps: 8.3e-7 / 8.4e-7 / 2.4e-6 vs 7.8e-7 / 7.7e-7 / 2.4e-6;
`gpu_flame_les_species`).

**Laminar TFLES** (M 0.01, `--inflow-speed 0.98`, `gpu_tfles_1d`): F 4 at
1.875 cells (7.5 thickened), box 30 + 20 δ_th, tt 80, window [40, 80] against
the DNS at 7.5 cells in the F-scaled box (7.5 + 5 δ_th, tt 20, window
[10, 20]; the same 126 cells):

| run | S_L | vs |
|---|---|---|
| L1a TFLES F 4, uniform | 1.016824 | L1c − 8.2e-4 (gate 2e-3) |
| L1b TFLES F 4, dynamic | 1.016945 | L1a + 1.2e-4 relative (gate 1%) |
| L1c DNS 7.5 cells | 1.017642 | (default box: 1.016856) |
| L2 3D nz 16, LES σ 1.35 + Charlette | 1.016945 | L1b + 0.0 (bitwise S_L; gate 2e-4) |
| L2' 3D, LES Smagorinsky 0.17 | 1.016992 | L1b + 4.7e-5 (+0.005%) |

The Smagorinsky shift is small because Ω = 1 across the flame switches its
SGS scalar flux off; its stress stays. Laminar S_L of the LES grids (1D,
M 0.02, tt 40, window [20, 40]): 4 cells 1.030619; F 1.5 at 4 cells
1.024656; F 2 at 3 cells 1.024326; F 3 at 2 cells 1.024106.

**Shvab–Zeldovich teeth** (`gpu_tfles_teeth`, non-reacting stirred hot/cold
box, 32 × 16 × 16, h 4 δ_D, two ABC modes, M 0.005, 300 steps, Smagorinsky
Cs 2 so mean μ_t/μ = 14): max |T/Tu + (σ−1)Y − σ| = 1.58e-2 for the
molecular DNS (the MUSCL-species / central-energy scheme floor), 1.14e-2
with the eddy fluxes at Sc_t = Pr_t, 0.197 at the TEST-ONLY Sc_t = 0.45
(×17.3); LES + TFLES F 2: 2.05e-2 / 4.89e-2 (the SGS flux is off where Ω =
1). The plan's absolute 1e-2 is NOT met (the floor alone exceeds it): the
test gates relative to the floor.

**LES vs DNS** (the U2 case of "3D and turbulence": u'_in 4, l_t 2.5 δ_th,
seed 1, hold, M 0.02, tt 20, window [10, 20]; the coarse boxes carry the
same Fourier modes (spectral truncation); sink bands scaled to the DNS's
physical width, 16 × h_DNS; S_T = SL / the grid's laminar S_L):

| run | cells | S_T/S_L | / DNS | halves | A_T/A | E·A_T/A_T,DNS | brush / DNS | u'_flame | Ebar | FEmax | ms/step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DNS 6 cells | 224 × 90² | 1.703 | 1 | 1.42 / 1.98 | 1.692 | 1 | 10.09 δ_D | 0.879 | — | — | 3.81 |
| DNS 4 cells | 160 × 60² | 1.835 | 1.078 | 1.52 / 2.14 | 1.797 | 1.062 | 1.021 | 0.898 | — | — | 1.12 |
| LES σ, no TFLES, 4 cells | 160 × 60² | 1.851 | 1.087 | 1.54 / 2.15 | 1.810 | 1.069 | 1.017 | 0.896 | — | — | 1.28 |
| LES σ + TFLES 1.5, Charlette, 4 cells | 160 × 60² | 1.186 | 0.696 | 1.13 / 1.24 | 1.190 | 0.703 | 0.707 | 0.815 | 1.00001 | 1.60 | 1.69 |
| LES2-C: TFLES 2, Charlette, 3 cells | 128 × 45² | 1.088 | 0.639 | 1.07 / 1.11 | 1.092 | 0.646 | 0.602 | 0.801 | 1.0004 | 2.81 | 0.86 |
| LES2-N: as LES2-C, E = 1 | | 1.089 | 0.639 | 1.07 / 1.11 | 1.092 | 0.645 | 0.602 | 0.801 | 1 | 2.00 | 0.82 |
| LES4-C: TFLES 3, Charlette, 2 cells | 96 × 30² | 1.042 | 0.612 | 1.03 / 1.05 | 1.039 | 0.629 | 0.551 | 0.779 | 1.024 | 5.20 | 0.47 |
| LES4-N: as LES4-C, E = 1 | | 1.042 | 0.612 | 1.03 / 1.05 | 1.040 | 0.614 | 0.552 | 0.779 | 1 | 3.00 | 0.47 |

(The tables' "LES2"/"LES4" are the plan's names; the DNS is P1's 6-cell
grid, so the LES grids are 1/2 and 1/3 of it with F 2 and 3, 6 thickened
cells each.) **The plan's LES-vs-DNS gates FAIL** (issues.md L1): S_T 0.64 /
0.61 of the DNS (gates within 15% / 25%), modelled area 0.65 / 0.63 (20% /
30%), brush 0.60 / 0.55 (30% / 40%); the efficiency closes nothing (C − N:
−0.0002 / +0.0003 in S_T/S_L). What the numbers say: resolution alone is not
the loss (4 cells DNS / LES-σ within 9% of the 6-cell DNS); THICKENING is
(F 1.5 already −30%), and the Charlette efficiency is ~1 here because Re_Δ =
u'_Δ Δ/ν ~ 2–5 sits on its viscous cutoff (f_Re) — while the DNS's S_T
grows through the window (halves 1.42 → 1.98), consistent with
Darrieus–Landau wrinkling (σ = 5, unstable band ∝ 1/(F δ)) that a
wrinkling-factor model does not represent (unproven: a u'_in 0.25 control
shows no DL growth in 20 τ_F, A_T/A 1.003; F 1.5 with the UNIFORM sensor
reads 1.184 vs 1.186 dynamic, so the sensor is not the loss; issues.md
L1). Internally the LES is
consistent: I0 = S_T/(A_T/A) = 0.996 / 1.002, resolved A_T ≤ A_T,DNS.

Regenerate: `runs/flames/turb/p2/les.sh` and `les2.sh` (2.1 + 1.3 min GPU),
then `tools/chem/turb_stats.py PREFIX --t0 10 --t1 20 --sl S_L(grid) --mach
0.02`.

## 3D turbulent slot Bunsen (`--mode bunsen --nz N`)

`--mode bunsen` at `--nz N` (1, or ≥ 8; hz = h, z periodic): one span
period of an infinite slot. Nothing new in a kernel: the per-row inflow
arrays are filled at j + ny k (the slot repeated over z), `bunsenIC` is
extruded over z, `--les`/`--sgs`/`--tfles` take the premixed 3D machinery
(`Eddy`/`Thick` species instantiations with `Row`, the open-row μ_t copy).
Files: `config` (`bunsenGeometry` nz, the shared `flameTurbSetup`),
`flame1d` (`addBunsenTurbulence`), `dns_main` (rows, the J-weighted box),
`runner_flame` (the `b3` branch), `tools/chem/turb_render.py`.

**Turbulent jet** (`--turb U --turb-length L --seed S`): the same
hash-seeded frozen box as the planar flame (`dns/turb_ic`, L_bx = L_z,
L_y and L_z ≥ 6 l_t, l_t ≥ 4 cells), multiplied row by row by the jet weight
J(y) on the device (the coflow stays laminar) and swept through the inflow at
U_jet (+ the pulse) — `kNscbcCells<…, Prof, Turb>`, compiled since P1. The
IC lays the box times the cell's fresh fraction Y (jet core and the unburnt
cone). The box times J is no longer plane-mean free (its z-mean per y row
is not zero): a small acoustic pumping, not measured. The IC cone and the
`--length` bound use Damköhler's S_T = S_L (1 + u'/S_L) — an UPPER estimate
(a slower-burning cone is taller), so it is a floor, warned at validation;
U_jet ≤ 1.25 S_T is rejected (no cone). M (U_jet + 3 u') ≤ 0.3.

**Measurement** (3D only; the 2D path is the shipped code): T, ρ, u, v, ω
downloaded and averaged over z (k ascending, FP64) onto the nx × ny plane
`measureCone` and `flankMassFlux` read; `Tmax` is the pointwise 3D peak,
`fuelIn` sums the 3D inflow face, `SLflank` is per unit span. The burning
window of the flank crossings is ½(2F − 1) δ_th (F = `--tfles`, 1 without):
the thickened reaction zone sits up to ~F − ½ thicknesses behind the Tmid
crossing — at F 3 and 2.5 cells/δ_th the 2D window (one cell) rejected every
flank row of the span-mean cone (measured, run killed at t = 16 by the
"cone opened" guard while burn/in read 0.977 and the base sat on the inflow
row). In 3D that guard (fewer than 4 burning flank rows) no longer stops the
run: a turbulent brush's span mean need not keep a straight flank; the fit
columns read NaN; flash-back, blow-through and base lift-off still exit 8.
Appended stats: `ST` = burn / A_cone (A_cone = 2 hypot(xTip − xBase, W/2) L_z,
the span-mean cone's flank area; ρ_u Y_u = 1), `area` = ∫|∇c| dV / A_cone
(co-area: the flame-surface ratio), `I0` = ST/area. Frames add `vort`, `speed`.

**Bit identity** (gfx1100, simjob 612, 2026-10-29; base = the pre-P3 tree
incl. the uncommitted FV work, `build/turb_base_P3/dns`): every output but
the `.json` byte-identical in 10 flame cases — premixed 1D (stats +
profiles), premixed 2D wrinkled frames, 3D `--turb` volumes, 3D LES σ + TFLES
3 `--turb` volumes, bunsen (frames), bunsen `--tfles 2` (frames), bunsen
`--pulse` (frames), diffusion (frames), candle, chamber (frames); 53 files
(612), 60 after the last runner edit (617, every one identical). Verifier
(simjob 618): the PRE-P1 base (`build/turb_base_P1/dns`) vs branch for
premixed 1D, bunsen, diffusion, `flow --physics compressible --far nscbc`,
`conv`, `hit`: 21/21 files identical — species off / nz 1 / LES off is the
pre-turbulence solver across P1–P3. The final run's first three stats rows
are reproduced byte for byte by the current binary (simjob 619). A flame
LES run's `.json` now carries `cs` and `les` (it read `cs 0, les 0` with
`sgs "sigma"` in run 615; DNS jsons unchanged).
`gpu_bunsen3d_extrude`: the burner (W 6, U 3, 4 cells, M 0.02, tt 2)
extruded to nz 8 equals the 2D run BYTE FOR BYTE — temp/fuel/omega mid-span
frames and the span-mean cone columns (t, dt, xTip, xBase, tanA/L/R, rms,
sinRatio, mFlank exactly; SLflank per unit span to 1e-6) — for the DNS and
for TFLES 2 where the 3D run adds LES σ 1.35 (σ ≡ 0 on a 2D field) with
`--efficiency none`. The 3D dt is the 2D dt (stableDt is the max over axes).

**The run** (gfx1100, simjob 615, 2026-10-29; LABELLED LES everywhere):

    dns flame --mode bunsen --jet-speed 8 --slot-width 24 --coflow-width 16
      --mach 0.01 --les 1.35 --sgs sigma --efficiency charlette --turb 3
      --turb-length 6 --cells-per-delta 2.5 --tfles 3 --nz 120 --length 100
      --slot-edge 0.5 --tt 36 --frame-start 18 --fdt 0.04
      --emit stats --emit volume --emit slices --out runs/flames/turb3d/final/b3

282 × 140 × 120 = 4.74 M cells (h = δ_th/2.5, 7.5 cells across the F 3
flame), burnt coflow at 1 S_L, span 2 W, Re_jet = U W/ν = 427, inflow Re_t =
40, h/η = 1.06 (the velocity field is near-DNS-resolved; the LES content is
the 3× thickened flame and a small σ SGS). 37361 steps, 16.34 ms/step (3.4
ns/cell-step with stats every 200 steps and 451 volume frames of 5 channels,
41 GB), 611 s wall. The plan's U_jet 20 was not run: the P2 measurement (TFLES
keeps S_T near laminar) put its cone past any affordable box; at U 8 even
the laminar cone (95 δ_th) fits. A first attempt exited 8 at t = 16.1 on the
"cone opened" guard with burn/in 0.977 and the base on the inflow row (the
1-cell burning window, above): fixed, rerun. Scout (U 10, 5.16 M cells):
16.6 ms/step.

Window t ∈ [18, 36] (94 stats rows, span means; halves [18, 27] / [27, 36]):

| quantity | mean (sd) | halves |
|---|---|---|
| xTip | 91.7 δ_D = 57.3 δ_th (7.0 δ_D) | 90.6 / 92.8 δ_D (+2.3%) |
| xBase | 0.32 δ_D (the first cell row: anchored) | – |
| burn / fuelIn | 0.991 (0.029; 0.916–1.052) | 0.989 / 0.993 |
| ST / S_L | 1.586 (0.123) | 1.602 / 1.571 |
| area A_T/A_cone | 1.424 (0.096) | 1.436 / 1.414 |
| I0 = ST/area | 1.113 (0.022) | 1.116 / 1.110 |
| SLflank | 1.229 | 1.233 / 1.226 |
| Ebar (Ω-weighted E) | 1.013 | 1.011 / 1.016 |

Kinematic consistency (NOT an independent check: ST is defined on the same
cone, ST = burn / (2 hypot(H, W/2) L_z) with burn ≈ fuelIn, so inverting it
returns H up to burn/in and the jet profile): a cone of S_T 1.586 at U 8 is
(W/2) √((U/S_T)² − 1) = 59.3 δ_th tall; measured 57.3. Likewise A_cone cancels
in I0 = burn / ∫|∇c| dV; the laminar extruded burner of
`gpu_bunsen3d_extrude` already reads I0 1.06 (DNS) / 1.13 (TFLES 2), so I0 >
1 is not a turbulence effect (issues.md B4). Damköhler (1 + u'_in) 4, Peters 1999 4.23 at the
INFLOW u' (u' at the flame not measured; the box decays in transit, ~14 τ_F
to the tip): no literature band is claimed. FEmax = 3 √3 every row: E
saturates at F^β = √3 somewhere (Charlette's Δ/δ_L − 1 branch); Ebar 1.013 —
the efficiency closes little (as in P2). yHits ≤ 13 per row, Y ∈ [0, 1 +
1.43e-6]. **Tmax** (pointwise 3D peak) / T_b: ≤ 1.043 up to t = 27, then
grows to 1.131 (t 35.7): ALL of the excess sits in the inflow band (rows i ≤
12, at the jet edges y ≈ yc ± W/2, where the base burns in the LODI rows and
the re-added boundary heat; 15393 cells > 1.03 T_b in the last frame), the
interior (i ∈ [16, 266)) peaks at 1.025 T_b (issues.md B2). With chemistry
binding the dt from t ≈ 29 (dtChem 7e-4–1.5e-3).

**Video** (`tools/chem/turb_render.py`, numpy + ffmpeg, simjob --cpu 616,
8 workers, 109 s): `runs/flames/turb3d/turb_flame.mp4` (1920 × 1080, 30 fps,
451 frames = 15.0 s for 18 τ_F, 3.1 MB) and the last frame
`turb_flame_poster.png`. Left: the rendered light — chemiluminescence j = ω
(the resolved, (E/F)-scaled heat release; optically thin), coloured by the
CH* 431.4 / C2* 516.5, 473.7 nm lines + a weak CO2* continuum through the
CIE 1931 functions (Wyman 2013 fit; CH* is out of the sRGB gamut, mapped in by
adding white), plus a faint steel-blue fog ∝ |∇×u| Y_F (cold-jet vorticity,
NOT light, small artificial absorption); orthographic orbit 15° → 40° about
the jet axis; each ray covers the z period with a Hann window of half-length
L_z cos θ (a partition of unity: every point of the periodic field counted
once at any angle, no seam); the sink bands are cropped. Middle: |∇×u|
(sqrt-mapped), right: |u|, both on the mid-span xy plane. Fixed exposure,
FFT bloom, ACES. `turb_render_selftest` (host ctest): a blob's image total
equals ∫ j dV to 1.0000 / 0.99996 / 0.9968 at θ 0 / 25 / 40°, the z seam to
2.6e-4, bytes deterministic. Raw frames (41 GB) are deleted after the
render; regenerate: the command above (simjob --gpu, ~10 min), then
`runs/flames/turb3d/render.sh` (simjob --cpu, ~2 min).

## Wick candle (`--mode candle --wick`)

The candle above with a REAL wick: `dns/wick.{hpp,cpp}` (geometry, regions,
soot constants, `measureWick`), `wickIC` (`flame1d`), `gpu/runner_wick`
(its own runner: the slot candle's `runner_flame` is untouched), the walled
two-reactant kernel `kSpeciesStage2<Row, AmbHi, Wall, Soot>` and the soot
cell pass `kChemCellPass2Soot` (`gpu/species.hip`), per-ghost wall
temperature (`GhostWallThermal::TwallAt`, `gpu/comp_ghost`), renderer
`tools/chem/candle_render.py`.

**Geometry (choices).** x up, gravity −x, y periodic. A candle body (box
`--candle-width` 6 × `--candle-height` 4 l, top = the wax surface) floats
1 l above the inflow sink band; the slow air coflow (1 U, the slot candle's
reason: a periodic box with an open bottom is a chimney) passes round it. A
wick (`--wick-width` 0.75 × `--wick-height` 2 l, sunk into the body) stands
on its top. ONE immersed ghost wall, isothermal: the wick at
`--wick-temp` 2 T0 (the fuel's vaporisation temperature, ~600 K at T0 300
K), the body at T0 — a per-ghost wall temperature sampled at each ghost's
wall point (`TwallAt`; empty = the uniform wall, tables and kernels byte for
byte). The body floats rather than standing on the inflow face: the chamber's
reason (no ghost rows inside the NSCBC rows and sink band); the render shows
only its top 2.5 l.

**Pressure image under gravity.** The wall's pressure image is the wall
momentum balance grad p = ρ(f − a_wall); with f = −g x̂ the image is the
hydrostatic one, obtained by handing the RESTING wall a `GhostWallMotion`
with u = 0, a = +g x̂ (`p_g = p_s + ρ_s g (x_s − x_g)`, the conv plates'
image). The impermeable faces then carry u_wall·n = 0: the resting wall's
fluxes.

**Fuel = a volumetric source band, FIXED rate.** The fluid cells
face-adjacent to a WICK solid cell (above the wax; 38 cells at the defaults)
get ρ += S, E += S cp T_wick (vapour at rest at the wick temperature),
ρY_F += S Y_F1, S uniform, total `--fuel-rate` Q = 1 per unit depth (the
slot candle's Q at W 1), held for the whole step (SSP-RK3 adds exactly S dt;
`fuelInjCum` is its FP64 sum). The rate is NOT tied to the heat flux into
the wick; that flux is measured (`qWick`, the Fourier face flux of every
fluid/wick face with κ(T_face), the solver's own expression) and reported
as `qWickRatio` = qWick/(Q q).

**Species at the wall.** The chamber's rule on the two-reactant kernel, its
own instantiation: no species flux (convective or diffusive) through a
fluid/solid face, a solid upwind second neighbour makes the face "first"
(shared θ = 1, central; soot first-order upwind), solid cells get no rhs and
no heat release and keep ρY = 0 from the IC. Rhs hook: NSCBC → gravity →
species re-carry → inflow pin → sink → species sink → wall faces
(`applyRhs`) → fuel source; stage hook: the ghost fill.

**Soot (model S1, one equation, semi-empirical).** ρY_s is species 2:

    formation  ω_f  = A_f ρ Y_F e^(−Ta_f/T)          fuel → soot, no heat
    oxidation  ω_ox = A_ox ρ² Y_s Y_O e^(−Ta_ox/T)   soot + s O → products, + q ω_ox

Soot is the fuel's carbon in condensed form: the fuel's stoichiometry s and
heat q, so T_ad and Z keep their meaning (the gas-phase Z excludes soot; the
tip and Z bounds read it). Ta_f = 50 T0, Ta_ox = 60 T0 (~15000 K / 18000 K at
T0 300 K: between Moss's surface growth 12100 K and Leung–Lindstedt's
nucleation 21100 K, and Lee's oxidation 19800 K). A_f, A_ox are set by
Damköhler numbers at T_ad (`--soot-form` 10, `--soot-ox` 2e5) — TUNED FOR
APPEARANCE (a sooting yellow zone burned out at the tip), not calibrated.
Soot rides its OWN MC limiter (it is in no coupling function; it never votes
in the shared θ), no molecular diffusion (Sc → ∞), no thermophoresis, no
radiation (the heat loss a real candle radiates, ~20–30%, is absent). First
order in soot mass = a fixed particle size. Its stiffness k_f + k_ox ρ(Y_O +
s Y_s) enters the step (`dtSoot`, kChemDt / max): never binding at the
defaults (3e-3 vs the acoustic 1.1e-4). `--soot off` = two species.

**Anchoring (measured, scouts, defaults otherwise, tt 4).** At the slot
candle's Da 5000 the IC sheet's base LIFTED OFF (0.75 → 6.9 l above the wax
by t 2) and the flame went out (exit 8 at t 3.0): the buoyant draft past the
wick base outran the edge flame (the slot candle never tested this: its
inflow sheet is attached by construction, §43). Da 2e4: base wandering
0.6–1.6 l; Da 5e4: base 0.75 l, steady (the default with `--wick`); Da 5000
with `--wick-temp 3`: base 0.125 l. dtChem 1.3e-3 never binds.

**Soot tuning (scouts, defaults, tt 8, window t > 5; Da_f / Da_ox).**

| Da_f | Da_ox | formed / Q | oxidised / formed | out of the top / formed | peak f_v |
|---|---|---|---|---|---|
| 2 | 2e3 | 0.0098 | 0.09 | 0.90 (smoking: a filament up the axis) | 2.7 ppm |
| 2 | 5e4 | 0.0098 | 0.92 | 0.094 | 0.26 ppm |
| 4 | 2e5 | 0.0195 | 1.009 | 3.4e-4 | 0.45 ppm |
| **10** | **2e5** | **0.047** | **1.009** | **3.3e-4** | **1.1 ppm** |
| 10 | 1e6 | 0.048 | 1.007 | 2.6e-5 | 0.96 ppm |

(oxidised/formed > 1: the start's soot inventory still draining.) At the
default the soot forms on the rich side just inside the sheet in the flame's
upper half, converges (the buoyant stream tube contracts) into an inverted V
and burns out through the tip: a non-smoking candle. Real candles reach
f_v ~ 1–10 ppm; 1 ppm here is in range, by tuning.

**Measured (gfx1100, simjob, 2026-10-28).** Defaults (8 cells/l, 288 ×
200, 0.45 ms/step, tt 24 in 102 s) and the demo (12 cells/l, 416 × 300,
0.62 ms/step, tt 12 in 107 s). The flame height drifts slowly (5.99 → 5.43 →
6.13 → 6.36 l over t 4–24: the slot candle's slow box cycle, §51); the
window halves of t ∈ [20, 24] agree to 1%.

| quantity | 8 cells, t ∈ [20, 24] | 8 cells, [4, 6] | 12 cells, [4, 6] | 12 cells, [10, 12] |
|---|---|---|---|---|
| flame height above the wick tip (axis Z_st) | 6.362 l (halves 6.334 / 6.390) | 5.988 | 5.973 | 5.351 |
| Z_st half-width at mid flame | 1.142 l | 1.110 | 1.100 | 1.041 |
| flame base above the wax (ω ≥ 0.1 max) | 0.75 l | 0.75 | 0.75 | 0.75 |
| top of the sooting zone above the wick tip | 6.86 l | 6.47 | 6.46 | 5.83 |
| Tmax / T0 (T_ad 5) | 4.375 | 4.360 | 4.354 | 4.332 |
| burn / Q | 0.940 | 0.949 | 0.951 | 0.949 |
| soot formed / Q | 0.057 | 0.052 | 0.051 | 0.043 |
| soot oxidised / formed | 0.998 | 1.005 | 1.008 | 0.999 |
| soot out of the top / formed | 8.2e−5 | 2.0e−4 | 2.2e−4 | 4.1e−4 |
| (burn + soot formed + fuel out) / Q | 0.9990 | 1.0039 | 1.0052 | 0.9985 |
| peak f_v | 1.34 ppm | 1.22 | 1.26 | 1.09 |
| qWick / (Q q) (conduction into the wick) | 0.0407 | 0.0438 | 0.0219 | 0.0281 |
| qBody / (Q q) (into the T0 body) | 0.125 | 0.122 | 0.143 | 0.139 |
| mean gas T in the source band / T0 | 2.027 | 2.029 | 2.001 | 2.002 |
| box-mean p / p_a | 0.99949 | 0.99784 | 0.99759 | 0.99845 |
| \|u\|max, min u on the top row | 26.6, −4.7 U | 26.6, −8.0 | 26.3, −7.3 | 25.7, −6.1 |
| min dtSoot, dtChem vs dt | 2.7e−3, 1.4e−3 vs 1.08e−4 | — | 3.0e−3, 1.3e−3 vs 7.2e−5 | — |

Fuel balance on its time INTEGRAL (inventory change = injected − burnt −
into soot − out, stats rows trapezoid, after the start t > 2): 7.0e−5 of the
22 injected (8 cells, t 2–24), 7.0e−5 (12 cells); soot: 2.3e−5 of the 1.13
formed. The windowed ratio's ±0.5% is the inventory still moving. The
start's under-sampled IC burn costs 7e−3 of the inventory before t 1. Inert
(`gpu_wick_inert`, no reaction, tt 1): fuel = IC + injected − out to 3.2e−5
of the inventory (the wall leaks nothing; the FP32 absorption of the source
increment, the chamber's C5).

Grid: height, half-width, Tmax, soot formed and f_v agree to 0.2–2% between
8 and 12 cells/l at equal t; the WALL heat fluxes do not (qWick 0.044 vs
0.022, qBody 0.122 vs 0.143): the staircase wall's conduction on a 6- vs
9-cell wick is first order and unconverged (W6). The wick heat flux is the
order of what vaporisation needs: qWick/(Q cp T0) = 2.7 (8 cells) — the
vapour's sensible heat from T0 to 2 T0 (1.0) plus ~1.7 cp T0 ≈ 0.5 MJ/kg at
T0 300 K, paraffin's latent heats' range — a consistency, not a validation.
`hits` (Y_O above Y_O2 by ≤ 2e−5) 5000–20000 cells: the FP32 walk of the
slot candle's start (W5). No flicker: the default flame is steady but for
the slow height cycle above.

**Demo** (`/home/manuel/Documents/pesto/runs/flames/candle2/`): `dns flame
--mode candle --wick --cells-per-unit 12 --tt 12 --emit stats --emit bin
--fdt 0.05` (171843 steps, 107 s), rendered `candle_render.py PREFIX
--cells-per-unit 12 --scale 3 --first 80` (t 4–12, 160 frames, 5.3 s at 30
fps): `candle_wick.mp4`, `candle_wick_final.png` (planar, the default);
`--revolve`: `candle_wick_revolved.mp4` / `_final.png` (the visualization
as if axisymmetric). Raw frames deleted (~690 MB; ~2 min of GPU to
regenerate).

**Render** (`tools/chem/candle_render.py`, numpy + ffmpeg). Colour from
physics: soot incandescence of optically thin Rayleigh particles, I_λ = 6π
E(m) f_v/λ · B_λ(T) · L (E(m) 0.26, L = one cell = `--cell-mm` 0.4 mm, f_v =
ρY_s ρ0/ρ_soot with ρ_soot/ρ0 = 1800/1.177, T = 300 K (1 + temp/T0)),
integrated over 380–780 nm against the CIE 1931 colour-matching functions
(Wyman–Sloan–Shirley multi-lobe fit) → XYZ → linear sRGB; a camera white
balance (von Kries, `--white-k` 1900 K: candlelight renders neutral);
chemiluminescence as a CH* 431 / C2* 516 + 473 nm line spectrum scaled by the
heat release ω (a DISPLAY stand-in: one-step chemistry has no CH*; its gain
puts the peak at 0.5 of the display range); exposure fixed per video from the
last frame's soot luminance; ACES filmic curve, sRGB OETF, mild blur and a
bloom halo. The body: wax (albedo 0.95/0.92/0.85) and wick (char, 0.03)
from the mask, lit by the soot emission through a 1/r kernel (FFT), a
translucent glow under the wax top. The wick sits at T_wick ~600 K in the
model: it does not incandesce and is drawn dark (no glowing tip). 2D: no
line of sight (`--revolve` renders the planar field AS IF axisymmetric — a
visualization, labelled so; not the default).

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
| `wick` | host | body SDF (exact outside), wall T, regions (solid = sdf < 0, the source band beside the wick only, symmetric), soot Damköhler normalisation and floors, defaults / dry-run / label (2D, inflated, appearance-tuned, no radiation), IC (p = p_a on every cell, solids species-free, a closed sheet), `measureWick` on drawn fields, `--soot off`, every rejection by name |
| `gpu_wick_inert` / `_flame` | GPU binary | no reaction: fuel = IC + injected − out (no wall leak); defaults tt 6: anchored, closed tip, fuel and soot integral balances, soot formed and burned out at the tip, the wick heated |
| `chamber_config` | host | polygon SDF, regions (closed cavity, probes), IC, isentropic inverse, dry-run, every rejection by name |
| `gpu_chamber` | GPU binary | inert wall impermeability; reacting balances, nozzle vs isentropic, anchoring |
| `turb_ic` | host | the turbulence box: solenoidal, Hermitian, shell energies, u', plane means, FFT vs direct sum, coarse = spectral truncation (n/2, n/4), seed determinism, lattice L11, Peters/Damköhler, turbNumbers |
| `flame_config` (3D) | host | `--nz` 1 / ≥ 8, hz = h, 3D dry-run and label, the T2 case, UNDER-RESOLVED label, every `--turb`/`--turb-length`/`--hold-flame`/`--seed`/`--frame-start` rejection by name |
| `flame_config` (3D bunsen) | host | `--mode bunsen --nz 16`, the P3 case (282 × 140 × 120, Damköhler IC cone, label: LES + TFLES + jet turbulence + Mach, dry-run Re_jet, the warning), rejections: `--nz 6`, chamber `--nz`, U_jet ≤ 1.25 S_T, L_z < 6 l_t, `--length` below the turbulent cone, M (U + 3u') > 0.3, `--les` without `--sgs` |
| `gpu_bunsen3d_extrude` | GPU binary | the burner at nz 8 equals 2D byte for byte (frames, span-mean cone), DNS and TFLES 2 + LES σ |
| `turb_render_selftest` | host (python) | the renderer: blob total vs ∫ j dV at θ 0/25/40°, z seam, determinism, colour |
| `gpu_flame3d_extrude` | GPU API | 2D wrinkled flame vs its nz 4 extrusion: every k plane bitwise, w ≡ 0 |
| `gpu_flame_turb_inflow` | GPU API | Turb with zero rows = off (uniform and Prof targets) bitwise; device rows = host `fmaf` lerp bitwise; rows relaxed onto the inflow row (n 16, 32) |
| `gpu_flame3d_planar` | GPU binary | nz 16 vs nz 1 planar flame: ΔS_L ≤ 2e-4, area = 1 |
| `tfles` | host | Charlette limits (E(u'=0) = E(F=1) = E(Re→0) = 1, monotone in u' and Δ, ≤ F^β), C++ vs `tools/chem/tfles_eff.py` 1e-12, FP32 vs FP64, Colin scaling, the sensor norm vs a dense scan |
| `flame_config` (LES/TFLES) | host | the LES + TFLES case, defaults (charlette with --les, dilation radius), labels (LES / Smagorinsky / thickened flame), dry-run lines, warning below 6 thickened cells, every rejection by name (--les without --sgs / at nz 1 / with bunsen, --sgs without --les, --tfles range, sensor/efficiency without --tfles, charlette without --les, F cells < 4, diffusion/candle/chamber) |
| `gpu_tfles_bits` | GPU binary | `--tfles 1` (uniform, dynamic) == no `--tfles` bytewise, 1D and 3D `--turb`; planar 3D charlette == none bytewise |
| `gpu_tfles_sensor` | GPU API | Ω coverage on the thickened oracle (F 4 / 1.875, F 2 / 3); the T-gated filter == host passes bitwise, == brute-force box max at uniform T |
| `gpu_flame_les_species` | GPU API | uniform Y exact under Eddy / Thick; conservation vs the DNS's drift; the open-row μ_t image (σ and Smagorinsky) |
| `gpu_tfles_teeth` | GPU API | Shvab–Zeldovich defect at Sc_t = Pr_t ≤ the DNS floor, ×10 at Sc_t 0.45; TFLES + LES within 2× the floor |
| `gpu_tfles_1d` | GPU binary | laminar TFLES S_L (L1a/b/c), 3D LES σ bitwise S_L, Smagorinsky shift (~75 s) |

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
compensated, RANS species (asserted), LES at nz == 1, LES/TFLES species for
the two-reactant and walled kernels (diffusion, candle, chamber), a separate
Sc_t, thickened viscosity, SGS/thickening inside the NSCBC rows (the LODI
transverse flux ignores μ_t), Colin's efficiency (needs a global Re_t), a
dynamic Charlette β, nz > 1 for diffusion/candle/chamber, a cold-coflow
or piloted 3D Bunsen (only the burnt coflow is run), `--hold-flame` for the
Bunsen, a GPU span-mean kernel (the 3D Bunsen averages on the host),
forced turbulence (the inflow box decays in transit), an inertial-range
spectrum, an
immersed burner lip for bunsen and the jet (the walled kernels exist:
premixed for the chamber, two-reactant for the wick candle; neither takes
LES/TFLES), nz > 1 for the wick candle, open y faces for bunsen, the jet and
the candle, an open
(pressure) inflow bottom, gravity in the premixed modes, unequal W, a
counterflow/strained-flame oracle, the low-Mach branch (`--physics
incompressible` rejected), multi-step chemistry, soot beyond the one-equation
appearance model (nucleation/growth/PAH, thermophoresis, radiation), a fuel
rate tied to the wick heat flux, a melting wax pool, dns-ui (its editor covers
`flow` and `conv` only). Chamber: isothermal (cooled) walls, a
total-pressure inlet, non-premixed injection, a moving/2D-curved nozzle wall
beyond the polygon, 3D.
