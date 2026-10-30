# Chemistry: `dns flame` (P0 oracle + P1 premixed flame + 2D slot Bunsen + 2D jet diffusion flame + 2D candle + 3D premixed in turbulence + LES species and TFLES)

Spec of the single-step premixed flame, of the two-reactant diffusion
flame (`--mode diffusion`) and of the candle (`--mode candle`, the diffusion
flame under gravity). Plan and measurements:
`plans/chem_plan.md`, `plans/impl_chem.md`. Open items: `issues.md` §30–53, §64–69, T1–T7, L1–L6.

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
planar IC bit for bit. It seeds Darrieus–Landau; the single-mode linear
growth rate is measured against reactingFoam and Matalon–Matkowsky (G2b,
"Against OpenFOAM reactingFoam"); no ctest pins it.

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

(These rows carry the default ρ-target inflow: under the cone's +0.56%
inflow overpressure the fresh jet enters 0.59% hot, which alone lifts
sinRatio by 1.8% and SLflank by 0.95% at U 5, 8 cells; `--nscbc-tinflow`
holds T_row — G2a below, issues.md §66.)

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

## Against OpenFOAM reactingFoam (G1: 1D laminar premixed, 2026-10-30)

The C0 model in OpenFOAM v2606 `reactingFoam` on the IDENTICAL cells,
both solvers generated from one case spec. Full record:
`plans/verify/impl_foamflame_G1.md`.

**Tooling** (`tools/chem/foam_flame.py`, package `tools/chem/foamflame/`;
numpy only, reference only — never a build dependency): `model.py` (a numpy
port of the P0 oracle — Λ 3656401.0984807 vs dns' 3656401.098, δ_th/δ_D
1.6016509 — and `si_map`, the ONE box → SI mapping), `writer.py` (the v2606
case dictionaries), `cases.py` (CASES: one dict per case; a geometry = five
functions: grid, IC, patch BCs, blockMesh, probes), `observe.py` (one
measurement code path for both solvers, the runner's own definitions). `setup
CASE` writes `runs/flames/foam/CASE/` (foam cases, dns argv); `run CASE`
dry-runs dns, then submits ONE GPU job (one step per rung) and ONE CPU job
(blockMesh per case, then the reactingFoam runs side by side, serial each,
`--np` at a time); `compare CASE` prints the table and writes
`compare.json`. A new case is a dict; a new geometry its five functions and
its `measure()`.

**SI mapping** (`si_map`; inputs Tu 300 K, p0 101325 Pa, W 28.014 for F and
P, M = S_L/c_u 0.01, δ_D 1e-4 m): R 296.797, cp = 3.5 R = 1038.788 J/kg/K,
c_u 353.065 m/s, S_L = M c_u = 3.53065 m/s, ρ_u 1.137984 kg/m³, κ = ρ_u cp
S_L δ_D = 0.417366, μ = Pr κ/cp = 2.89283e-4 Pa s (constant), τ = δ_D/S_L =
2.83234e-5 s, A = Λ/τ = 1.290946e11 /s, Ta = θ Tu = 15000 K, Hf_F = q = cp
(σ − 1) Tu = 1.246546e6 J/kg (Hf_P 0). Box quantity × unit = SI (lengths δ_D,
times τ, speeds S_L, densities ρ_u, p × p0/p0_box, T × Tu/Tu_box).

**Model, checked in the v2606 source:**

| item | OpenFOAM | file |
|---|---|---|
| rate | `irreversibleArrheniusReaction` "F = P", beta 0: kf = A e^(−Ta/T); ω_i = kf max(c_F, 0), RR_F = −W kf ρY/W = −A ρ Y e^(−Ta/T) | `specie/reaction/reactionRate/ArrheniusReactionRate/ArrheniusReactionRateI.H`, `chemistryModel/.../StandardChemistryModel.C` (`omega`, `calculate`) |
| heat release | Qdot = −Σ Hc_i RR_i = Hf_F ω | `StandardChemistryModel.C` (`Qdot`), `specie/thermo/hConst/hConstThermoI.H` (Hc = Hf, Hs = Cp (T − Tref)) |
| gas | `hePsiThermo`, `reactingMixture`, `const` transport, `hConst`, `sensibleEnthalpy`, `perfectGas`; equal W, Cp, μ, Pr ⇒ composition-free R, cp, μ, κ; T from h by Newton, exact for hConst (`thermo/thermoI.H`, tol 1e-4 T0: one exact step) | `basic/psiThermo/hePsiThermo.C`, `transport/const/constTransportI.H` (κ = Cp μ/Pr, alphah = μ/Pr) |
| rate evaluation | `integrateReactionRate false`: the instantaneous rate at the current PIMPLE iterate (`calculate()`), so converged outer correctors = implicit Euler on the whole system (no dt-dependent splitting error in a steady flame) | `combustionModels/laminar/laminar.C` |
| Le = 1 | **reactingFoam's YEqn diffuses Y with `muEff` (Sc = 1, Le = 1/Pr = 1.39)**, h with `alphaEff` = κ/cp; no thermophysicalTransport model in this solver. The case adds `+ fvm::laplacian(alphaEff − muEff, Y)` (coded `scalarCodedSource`, `system/fvOptions`) ⇒ ρD = κ/cp exactly | `applications/solvers/combustion/reactingFoam/YEqn.H`, `TurbulenceModels/compressible/ThermalDiffusivity/ThermalDiffusivity.H`, `basic/heThermo/heThermo.C` (`alphahe`) |
| Y scheme | `MUSCL` limiter = clamp(min(2r, (r+1)/2), 0, 2) = MC; r = 2 d·∇Y_P/(Y_N − Y_P) − 1 with Gauss-linear ∇Y = the classic r on a uniform 1D row; face = Y_up + ψ (Y_lin − Y_up) = dns' MC-MUSCL | `limitedSchemes/MUSCL/MUSCL.H`, `LimitedScheme/NVDTVD.H`, `limitedSurfaceInterpolationScheme.C` |
| per-field schemes | `multivariateSelection` and plain multivariate apply ONE limiter (the min over Y and h) to every field; `multivariateIndependent` gives each its own | `multivariateSchemes/*` |

**What matches exactly:** the cells (dns' h = δ_th/N, nx = (12 + 10) N +
2·16, x_f0, cubic cells, one row; dns y periodic ny 4 = foam one cell,
`empty`), the IC (dns' `premixedState` of the oracle at the cell centres:
p = p0 − m²(1/ρ0 − 1), u = m/ρ, m = 0.98), every model constant, Le = 1,
constant μ, γ 1.4, M 0.01 (both compressible: same O(M²) physics), the
Y face interpolant on interior faces (dns goes first order where the upwind
cell is a boundary cell; Y is uniform there), the observables (one code
path).

**What differs (not removable):**

- inflow: dns NSCBC (u relaxed to 0.98, T relaxed, etaIn 100, p free, Y
  carried in, no face diffusion) + a 16-cell sink band; foam fixedValue U,
  T, Y, zeroGradient p, Fick/Fourier through the face (zero: uniform fresh
  gas). dns' fresh u wanders (sd 5e-4 S_L), foam's is exact.
- outflow: dns NSCBC p → p_b = p0 − u_in²(σ − 1) + 16-cell sink band; foam
  `waveTransmissive` p (γ 1.4, p_inf = p_b, l_inf = box length), U, T, Y
  zeroGradient. foam's band cells are plain interior.
- mean flow: dns explicit central (KG face flux) on total E, FP32; foam
  pressure-based PIMPLE, `Gauss linear` U, K and h (variant `mclin`),
  Rhie–Chow (ddtCorr), sensible h with K and dpdt, FP64. foam's energy
  equation has NO viscous work div(τ·u) (O(M²); drops out between plateaus).
- time: dns SSP-RK3 at acoustic CFL 0.4 (dt 3.7e-4 / 1.9e-4 / 9.3e-5 τ);
  foam implicit Euler, fixed dt 1e-3 τ (convective Co 0.023 / 0.047 / 0.093,
  acoustic Co 1.0 / 2.1 / 4.2), 3 outer × 2 pressure correctors, every solve
  to 1e-14 (DIC/DILU are exact on a tridiagonal row: 1 iteration).

**Protocol** both: `--inflow-speed 0.98 --tt 20`, mean over t ∈ [10, 20]
(dns stats rows / foam `volIntegrate(Qdot)` every 10 steps over the probed
ρY at the reference column; profiles: dns every 0.1 τ, foam 0.5 τ).

**Measured** (S_L consumption; δ_th = (Tb − Tu)/max dT/dx vs the oracle
1.60165; T_b at 2.5–3.5 δ_th behind x_f, exact energy balance −9.6e-5; Δp
vs the measured-state Rankine–Hugoniot jump; L∞ of θ over |ξ| ≤ 5 δ_th vs
the oracle at the solver's own cells):

| N | solver (scheme) | S_L | S_L halves | δ_th/or − 1 | T_b/σTu − 1 | Δp/RH − 1 | L∞ θ | L∞ ω/ω_max |
|---|---|---|---|---|---|---|---|---|
| 7.5 | dns (central, Y MC) | 1.016856 | 1.016862 / 1.016850 | −2.956e-2 | −8.6e-5 | −1.28e-2 (§65) | 9.88e-3 | 5.50e-2 |
| 7.5 | foam mclin (U, h linear, Y MUSCL) | 1.015386 | 1.015414 / 1.015357 | −3.121e-2 | −9.0e-5 | −3.4e-5 | 1.00e-2 | 5.49e-2 |
| 7.5 | foam mcmc (U linear, h, Y MUSCL) | 0.993611 | 0.993584 / 0.993638 | −2.38e-3 | −9.5e-5 | +4.4e-6 | 3.58e-3 | 1.12e-2 |
| 15 | dns | 1.003021 | 1.003040 / 1.003003 | −6.515e-3 | −1.07e-4 | +2.0e-4 | 2.05e-3 | 1.24e-2 |
| 15 | foam mclin | 1.002978 | 1.002988 / 1.002969 | −6.809e-3 | −9.4e-5 | −7.3e-6 | 2.17e-3 | 1.25e-2 |
| 15 | foam mcmc | 0.997593 | 0.997595 / 0.997591 | +7.16e-4 | −9.5e-5 | −5e-8 | 9.67e-4 | 3.79e-3 |
| 30 | dns | 1.000336 | 1.000103 / 1.000570 | −1.359e-3 | −8.0e-5 | −2.3e-3 | 4.23e-4 | 2.72e-3 |
| 30 | foam mclin | 1.000189 | 1.000194 / 1.000185 | −1.173e-3 | −9.55e-5 | +1.3e-6 | 3.52e-4 | 1.97e-3 |
| 30 | foam mcmc | 0.998850 | 0.998852 / 0.998848 | +6.67e-4 | −9.6e-5 | +3.0e-6 | 3.35e-4 | 1.69e-3 |

| ladder | S_L order p | S_L Richardson (p) / (2) | δ_th p | δ_th Richardson | L∞ θ p |
|---|---|---|---|---|---|
| dns | 2.365 | 0.99969 / 0.99944 | 2.160 | 1.60186 | 2.27 |
| foam mclin | 2.153 | 0.99938 / 0.99926 | 2.114 | 1.60248 | 2.11 |
| foam mcmc | 1.664 | 0.99943 / 0.99927 | — (non-monotone) | 1.60269 (p 2) | 2.05 |

dns vs foam mclin at equal mesh, profiles aligned at x_f (foam cubic-
interpolated onto dns' offsets; the same interpolation of the oracle is the
floor): L∞ θ 7.7e-4 / 1.6e-4 / 2.0e-4, Y 9.7e-4 / 1.8e-4 / 2.7e-4, u (as
(u − u_in)/(σ − 1)) 7.1e-3 / 1.6e-3 / 1.2e-3 (dns' wandering inflow
velocity), ω 1.3e-2 / 8.4e-4 / 1.0e-3 (floor 1.0e-2 / 6.5e-4 / 1.4e-4) at
7.5 / 15 / 30. At 7.5 and 15 cells every dns − foam difference is below
both solvers' own oracle error (the L∞ columns above); at 30 cells θ and ω
are below, Y sits between (2.7e-4 vs 3.1e-4 dns / 2.2e-4 foam) and u at
dns' own (1.22e-3 vs 1.21e-3 / 0.84e-3: dns' wandering inflow velocity).
S_L dns − foam +1.47e-3 / +4.3e-5 / +1.47e-4 (the 30 cell one inside dns'
own window-half spread, issues §31).

Reading:

- **Converged S_L at M 0.01 = 0.9995 ± 0.0002 from three independent
  ladders** (Richardson 0.99926–0.99969): the O(M²) offset of the low-Mach
  oracle. The M-ladder estimate above (−3.9e-4 → 0.99961) sits inside dns'
  own range (0.99944 / 0.99969) but 2.3–3.5e-4 above foam's (0.99926–0.99943),
  i.e. the FP64 code does not pin the offset tighter than ±2e-4. foam's energy
  equation lacks the viscous work div(τ·u), an O(M²) term (~1e-4 of the heat
  flux in the flame, estimate) of a size that could move S_L by that much —
  not separated.
- **The +1.7% at 7.5 cells is the Y/energy scheme mismatch, not the dns**:
  foam with the same mismatch (Y MC, h central) reads +1.54%; foam with h on
  the SAME limiter (Shvab–Zeldovich consistent) reads −0.64%, δ_th within
  0.24%, but converges at order 1.66 (the limiter clips the h profile).
- **Time-converged foam**: dt/2 moves S_L +4e-6, 6 outer correctors +7e-6
  (15 cells).
- **Le matters**: stock reactingFoam (ρD = μ, Le 1.39) at 15 cells reads
  S_L 1.137843 (+13.4% vs Le 1), δ_th −12.3%.
- **foam conserves energy across the flame to 1.1e-5 / 3e-6 / 8e-7 q and
  its Δp is the Rankine–Hugoniot jump to 3.4e-5 / 7e-6 / 1e-6** (7.5 / 15 /
  30); T_b/σTu → −9.55e-5 (exact −9.6e-5).
  dns: the flame is energy-exact within its sd (1–2e-5 q), but the burnt gas
  heats by ~1e-4 q over the next 6 δ_th (issues §64) and its fresh pressure
  is not flat at 7.5 cells (issues §65). Neither moves S_L.

Cost (job server): dns GPU 14.8 / 23.0 / 45.5 s (0.20–0.26 ms/step; 56k /
113k / 225k acoustic steps); foam 1 core each, 20 000 implicit steps:
exec 26.0 / 39.7 / 45.2 s (mclin; + ~8 s of dynamicCode compile for the
Le = 1 fvOption), the 8 runs (two ladders + two checks) side by side in one
78 s CPU job.

### G2a: 2D slot Bunsen (2026-10-30)

Full record `plans/verify/impl_foamflame_G2.md`; tooling as G1 plus the
geometries `bunsen2d` / `wrinkle2d` (`foamflame/cases.py`) and
`foamflame/observe2d.py` (numpy port of `measureCone`, `flankMassFlux` and
the runner's SLflank/burn/fuelIn; both solvers through it).

**Case.** `--mode bunsen` defaults (W 12 δ_th, U_jet 5, burnt coflow T/Tu
= σ at 1 S_L, coflow 12 δ_th per side, edge 0.25 δ_th), M 0.02, 8 cells:
392 × 288 cells, h = δ_th/8, every dns cell a foam cell (the 16-cell sink
bands are plain interior cells in foam). foam IC = `bunsenIC` (ported;
dns' frame 0, one step in, agrees to median 6e-7 Tu, 99% < 7.5e-4 Tu, the
rest is the first step at the inflow row and the flame).

| item | dns | foam |
|---|---|---|
| inflow | per-row NSCBC: u target, ENTROPY target = the row's DENSITY ρ = 1/t (default) or T_row held (`--nscbc-tinflow`), etaIn 100, Y_row carried, no face diffusion | fixedValue U, T, Y per face = the same `bunsenInflow` rows; zeroGradient p; Fourier/Fick through the face |
| outflow | NSCBC p → p0, sink 16 | `waveTransmissive` p → p0 (lInf = box), zeroGradient U, T, Y |
| y | periodic | `cyclic` |
| mean flow | explicit central KG, FP32, SSP-RK3 at acoustic CFL 0.4 (dt 5.5e-4 τ) | PISO (nOuter 1, nCorr 2), Euler, dt 4e-3 τ (Co ≤ 0.25), FP64; U, h `linear`, Y MUSCL (`mclin`) |

nOuter 1 is a cost decision: nOuter 3 at dt 2e-3 τ is 0.135 s/step on 14
ranks (25 τ = 1690 s); nOuter 1 at dt 4e-3 τ is 0.042 s/step on 12. The
steady cone is the fixed point of either (the chemistry source lags one
step, dt·k_chem ≈ 0.7 at the peak rate); the time-step dependence of foam's
fixed point (Rhie–Chow ddtCorr) was NOT measured here.

**Window.** The cone rings with a ~10 τ convective period (dns: tip 38 → 45
→ 44.6 δ_D, sinRatio swing 0.05 / 0.015 / 0.003 at t ≈ 5 / 15 / 25), so
nothing is steady before t ≈ 20. Window [20, 25] (its dns means sit within
1.5e-3 of [20, 30], xTip 1.7e-3 relative; [15, 20] is 8.7e-3 off in
sinRatio); foam runs to t = 25.

**The port** reproduces dns' own stats row at t = 30 to the csv's 6 digits
(xTip −9.7e-7, tanA −1.5e-6, sinRatio +4.4e-6, SLflank +4e-9, burn
+3e-7 relative). mFlank and fuelIn cannot be read from dns' 2D frames (no ρ,
u, v): dns' values are its stats; the port is checked on a drawn cone with
ρ u = 5, v = 0 (mFlank = U sin α to 3e-15).

**Measured** (mean over [20, 25], halves in brackets; dns "frames" = the
port on dns' integer-τ frames, the same sampling as foam's dumps; hw0 = the
fitted flank half-width at x = 0, tipOff = xTip − hw0/tanA, both port-only):

| | dns (ρ-target inflow, default) | dns `--nscbc-tinflow` | foam mclin |
|---|---|---|---|
| xTip (δ_D) | 44.6455 (44.627 / 44.665) | 45.5378 (45.527 / 45.548) | 43.7155 (43.633 / 43.798) |
| tan α | 0.21441 | 0.21043 | 0.21025 |
| sinRatio | 1.04823 (1.05112 / 1.04520) | 1.02961 (1.03463 / 1.02483) | 1.02876 (1.02630 / 1.03121) |
| SLflank | 1.00695 | 0.99742 | 1.00392 |
| mFlank | 1.00868 | 0.99823 | 1.00304 |
| burnRatio | 0.99861 | 1.00080 | 1.00318 (face flux + Fick: 0.99966) |
| Tmax/T_ad | 1.00365 (at the coflow INFLOW row) | 1.00025 | 1.00002 |
| hw0 (δ_D) | 10.611 | 10.640 | 10.267 |
| tipOff (δ_D) | −4.85 | −4.96 | −5.12 |
| inflow-row T/T_row − 1 (jet / coflow) | +5.86e-3 / +3.79e-3 | +1.4e-6 / +2e-7 | 0 (fixed) |

dns [20, 30] (the docs' "Measured" window): sinRatio 1.04679, SLflank
1.00698, mFlank 1.00833, burnRatio 1.00003 — the table above in this file
reproduced digit for digit.

Reading:

- **The inflow pressure rises 0.56% (jet) / 0.33% (coflow) above p0 in
  BOTH solvers** (foam p at the inlet cells: +5.59e-3 / +3.29e-3 p0). dns'
  default entropy target is the row DENSITY, so its fresh jet enters 0.59%
  hot and the burnt coflow 0.38% hot (the box's Tmax sits there). The oracle
  with Tu × 1.0059 (ρ_u held) burns 1.1% faster. With `--nscbc-tinflow`
  (T held, as foam's fixedValue T) dns' flank angle equals foam's: sinRatio
  1.02961 vs 1.02876 (+8.3e-4), tan α 0.21043 vs 0.21025 — inside the
  window-half spreads (sinRatio halves differ by 9.8e-3 dns, 4.9e-3 foam),
  so "equal within the ringing", not to 8e-4. The default inflates
  sinRatio by +1.8%, SLflank by +0.95% (issues.md §66).
- **The remaining difference is the BASE, not the flank**: at equal α
  foam's straight flank meets the exit plane 0.37 δ_D (1.9 cells, 3.5%)
  closer to the axis (hw0 10.27 vs 10.64), hence a 4% shorter cone (xTip
  43.72 vs 45.54). The flank speeds differ by −0.64% (SLflank) / −0.48%
  (mFlank), dns `--nscbc-tinflow` below foam. The flame base sits on the
  inflow face, where the two boundaries differ in kind (foam conducts heat
  and diffuses fuel through the face — 0.22% of the fuel enters by Fick —
  dns' open face carries none); unattributed, issues.md §67.
- foam is not as settled: its sinRatio halves differ by 4.9e-3 (the
  ringing drifts out of phase with dns' after t ≈ 10).
- Fields at t = 25 (θ = (T − Tu)/(T_b − Tu); |u| from the z-extruded nz 8
  twin, whose temp frames equal the 2D run's BYTE FOR BYTE through t = 20):
  dns − foam L∞ θ 0.42 (L2 2.2e-2) at the TIP (x 44.75 δ_D on the axis: a
  tip-position shift, not a field error); |u| L∞ 2.26 S_L (L2 0.22) on the
  INFLOW ROW at the jet edge (x 0.30, y 19.52: dns' row-0 u sits +2.1–2.4
  S_L above its target there, issues.md §68), 2.24 at the tip past the
  bands; dns `--nscbc-tinflow` − foam L∞ θ 0.62 (L2 3.4e-2), at the tip.

Cost: dns 0.372 ms/step (2D, 51k steps, 18.4 s for tt 30), nz 8 twin 1.70
ms/step (69.5 s for tt 25); foam 6250 steps, 279 s wall on 12 MPI ranks
(exec 259 s).

### G2b: Darrieus–Landau linear growth (2026-10-30)

**Case.** σ 5, β 8, Pr 0.72, M 0.02, `--upstream 16 --downstream 16
--inflow-speed 1 --ny 240 --wrinkle 0.02 --wrinkle-modes 1` at 8 cells:
288 × 240 cells, λ = L_y = 30 δ_th = 48.05 δ_D (2.1 λ_c), initial front
amplitude 0.032 δ_D (kA = 0.004). foam: the same cells, `flameIC` ported
(frame-0 agreement median 5e-7 Tu, amplitude 0.0320330 vs 0.0320319),
inlet fixedValue U = S_L, T, Y; outlet waveTransmissive p → p0 − 4 (the
1D's), cyclic y; PISO nOuter 1, dt 4e-3 τ (Co ≤ 0.10), fields every 0.5 τ.
Amplitude a = mode 1 of x_I(y) = h Σ_i Y (`wrinkle_rate.py`'s) and of the
mid-T isoline x_T(y); s = least-squares slope of ln a; dns frame k at t =
k·0.5.

**Theory** (Matalon, Fluid Dyn. Res. 50 (2018) 051412, eq. (19) + appendix
= Matalon & Matkowsky 1982, constant transport): s = ω_DL S_L k − l_f B1
S_L k² (B2 multiplies Le_eff − 1 = 0; B3 = 0 for constant viscosity),
ω_DL = (√(σ³ + σ² − σ) − σ)/(σ + 1), B1 = (σ/2) [(σ + 1 + 2ω_DL) σ ln σ/(σ −
1) + σ − 1]/(σ + (σ + 1)ω_DL), l_f = D_th,u/S_L = δ_D. O(k l_f), large β.
σ 5: ω_DL 1.17360, B1 4.31688, λ_c = 2π B1/ω_DL = 23.11 δ_D (the lesgap
dispersion scan interpolated ≈ 21).

| run | s(x_I) [8, 20] | s(x_T) [8, 20] | s(x_I) [4, 16] | long fit |
|---|---|---|---|---|
| dns 8 cells | 0.08837 (rms ln 0.002) | 0.08342 | 0.08434 | [8, 32] 0.08832 |
| foam mclin 8 cells | 0.08733 (rms ln 0.002) | 0.08255 | 0.08320 | — (tt 20) |
| dns 6 cells (ny 180) | — | — | — | [16, 32] 0.08621 |
| Matalon–Matkowsky | 0.07965 | | | |
| inviscid Landau (upper bound) | 0.15347 | | | |

- **dns − foam −1.2% in the rate** (0.08837 vs 0.08733), local rates
  (per 2 τ) agree to ≤ 0.002 at every t; θ L∞ at t = 20 1.8e-2 (L2 2.2e-3).
  The dns 6 → 8 cell change is +2.2%: the solver difference sits inside one
  solver's own grid sensitivity.
- **Both are 10–11% ABOVE Matalon–Matkowsky** and 43% below inviscid
  Landau: k l_f = 0.13, β = 8 and the O(k²) truncation of the theory; the
  sign (finite-β, O(k³) terms) is not established here. The measured rate
  is a clean exponential from t ≈ 8 at 8 cells (local s 0.085–0.089 to t
  32); at 6 cells it swings 0.04–0.12 over t 0–16 before settling (the [4,
  16] fit of the lesgap tables is transient-limited at coarse grids).
- σ 2 (dns only, λ = 36 δ_th = 64.6 δ_D ≈ 2 λ_c, ny 288, tt 32): s(x_I)
  [8, 32] 0.01503, [16, 32] 0.01518 vs Matalon–Matkowsky 0.01904 (−21%),
  inviscid 0.03766 (λ_c 31.97 δ_D). foam not run (CPU budget).

Cost: dns 0.36 ms/step (32 τ, 8 cells); foam 5000 steps, 160 s wall on 8
ranks (exec 142 s).

### G3a: 2D slot-jet diffusion flame (2026-10-30)

Full record `plans/verify/impl_foamflame_G3.md`; tooling as G1/G2 plus the
geometries `diffusion2d` / `candle2d` (`foamflame/cases.py`: ports of
`diffusionInflow`, `burkeSchumann`, `diffusionIC`, `candleAmbientP`, the
config.cpp grid), `foamflame/observejet.py` (`measureJet` verbatim + the
runner's diffusion stats row), `model.si_map_diff`.

**Case.** `--mode diffusion` defaults (W 16 l, s 1, Y_F1 = Y_O2 = 1, Z_st
0.5, σ 5, β 5, Da 1000, M 0.05, coflow 16 l at 1 U, edge 3 l), 4 cells/l:
416 × 192, h = l/4. SI (T0 300 K, p0 101325 Pa, W 28.014, l = 1 mm): U
17.653 m/s, μ = Pr ρ0 U l = 1.44642e-2 Pa s, reaction `F + O^1 = 2P`
(the `^1` matters: OpenFOAM's order defaults to the stoichiometric
coefficient — `F + 3.4O^1` for the candle), A = A_box (U/l)/ρ0 × W
(kmol units) = 2.25115e11 m³/kmol/s, Ta 9375 K, Hf_F = q = 2.49309e6 J/kg.
foam carries a passive tracer Zt = 1e-3 Z (Le 1, inlet Z(y)): the carried
check of Z(Y_F, Y_O). Full amplitude is impossible: `multiComponentMixture`
divides every Y by ΣY at construction and the pure fuel stream has no inert
to displace (a Zt = Z run halved the fuel core, measured).

| item | dns | foam |
|---|---|---|
| inflow | per-row NSCBC on the first CELL (x = h/2): u → u_row, ρ → 1/t_row (etaIn 100), p free; ρY_k PINNED to the Burke–Schumann row after the overwrite (ω = 0 there: Y_F Y_O = 0); no face diffusion; row-0 T free (0.896 T_ad peak) | per-face fixedValue U, T, Y_F, Y_O, Zt = the same row at the FACE (x = 0), zeroGradient p; Fourier/Fick through the face; the first cell is interior (it burns) |
| outflow | NSCBC p → p0, anchor σ 4, sink 16 | waveTransmissive p → p0 (lInf = box), zeroGradient U, T, Y |
| species scheme | ONE MC limiter shared by the voters (≥ 2^-24), central E | `mclin`: multivariateIndependent (each species its own MUSCL, h linear); `mvsel`: multivariateSelection {F, O MUSCL; Zt, P, h `blended 2`} = one shared limiter, but applied to h too (foam's `div(phi,Yi_h)` is ONE weight field for Y and h: shared species limiter AND central h is not expressible) |
| time | SSP-RK3, acoustic CFL 0.4 (dt 2.0e-3 l/U), explicit chemistry | PISO nOuter 1 / nCorr 2, dt 0.025 l/U (Co 0.52; dt × J = 0.63 with dns' dtChem = 0.5/J = 0.020), lagged explicit source |

Protocol: both from `diffusionIC` (the row extruded), dns tt 80 (the
`gpu_diffusion_jet` run), foam tt 75; window [60, 75] (dns [60, 75] vs [60,
80]: tip 5.4e-4 apart). One measurement path: `measure_jet` + the runner's
row on dns' frames reproduces dns' own t = 80 row (tip −1.4e-8, half
−3.9e-7, burn −2.7e-7, TmaxIn +3.3e-6); fuelIn (face 0|1, central mdot ×
central Y − ρD ∂Y/∂x), fuelOut, pMean of dns are its stats (no ρ, u in 2D
frames). Tst = T on the Z_st isoline averaged over x ∈ [¼, ¾] tip.

| [60, 75] | dns (etaIn 100, default) | dns `--nscbc-etain 1000` | foam mclin | foam mvsel | Burke–Schumann |
|---|---|---|---|---|---|
| tip (l) | 51.199 (51.117 / 51.286) | 47.543 (47.427 / 47.653) | 47.217 (47.290 / 47.144) | 47.171 | 70.34 (constant ρ: a scale) |
| Z_st half-width at tip/2 | 7.7727 | 7.6686 | 7.6692 | 7.6589 | |
| tip / (2 half) | 3.2935 | 3.0999 | 3.0784 | 3.0795 | |
| fuelIn (face 0\|1) | 9.6168 | 8.9888 | 8.9590 | 8.9452 | |
| foam inflow face: ρuY + Fick | — | — | 8.5421 + 0.4447 | 8.5459 + 0.5183 | nominal ρ_row u Y 8.4667 |
| burn (∫ω) | 9.6221 | 8.9900 | 8.9446 | 9.0239 | |
| burn / fuelIn | 1.00055 | 1.00015 | 0.99839 (face 0.99530) | 1.00880 (face 0.99556) | 1 |
| tip / fuelIn | 5.3240 | 5.2891 | 5.2704 | 5.2733 | |
| TmaxIn / T_ad | 0.98625 | 0.98561 | 0.98488 | 0.98438 | 1 |
| T at Z_st (mid flame) / T_ad | 0.94755 | 0.94707 | 0.94655 | 0.94623 | 1 |
| same, base (4 l past the band) | 0.90047 | 0.90471 | 0.90532 | 0.90456 | 1 |
| Z at max ω | 0.4873 | 0.4975 | 0.4867 | 0.4869 | Z_st 0.5 |
| base (row) | 0.25 (row 1: row 0 pinned) | 0.25 | 0 | 0 | |
| pMean (box p/p0) | 1.00128 | 1.00122 | 1.00498 | 1.00536 | |
| max \|H\| rows i = 0 / 1 / 4 | 0.350 / 0.205 / 0.073 | 0.131 / 0.071 / 0.030 | 4.7e-4 / 7.4e-4 / 1.2e-3 | 1.1e-4 / 2.9e-4 / 6.4e-4 | 0 |
| max \|H\| past the bands | 0.0162 | 0.0194 | 0.0528 (flame x ≤ 64: ≤ 0.0034) | 0.0512 | 0 |
| max \|Z(Y_F, Y_O) − Zt\| (L2) | — | — | 4.2e-7 (5.9e-8) | 1.6e-10 (3.5e-11) | 0 |
| Z range / hits | [2.53e-5, 0.99478] / 0 | same / 0 | [3.46e-5, 0.99354] / 0 | [4.02e-5, 0.99326] / 0 | |

H = T/T0 − 1 + (q/cp T0)(Y_F − Y_F1 Z) (Shvab–Zeldovich, 0 at Le 1).
Fields at t = 75 vs foam mclin (past the bands): dns default Z L∞ 2.16e-2
(L2 1.17e-2), θ L∞ 4.28e-2 (L2 2.35e-2), both on the axis below the tip
(x 45 / 41); dns etaIn 1000 Z
L∞ 1.11e-2 (L2 2.80e-3), θ L∞ 2.99e-2 (L2 7.44e-3), both at the outflow
corner (x 99.9).

Reading:

- **The default dns jet is 8.4% longer than foam's because its inflow
  passes 7.3% more fuel** (fuelIn 9.617 vs 8.959; tip per fuel −1.0%). The
  per-row NSCBC at etaIn 100 lets the inflow mass flux overshoot its target:
  the convective part of dns' face-0|1 fuel flux is 9.27 against the
  nominal 8.467 of the row it relaxes to (and 8.385 for its own row-0 T at
  p0): the excess is in the row's u. It scales with the relax:
  `--nscbc-etain` 20 / 100 / 1000 → fuelIn 11.115 / 9.617 / 8.989, tip
  60.29 / 51.20 / 47.54; `--nscbc-tinflow` does not move it (9.665, 51.52).
  The bunsen rows overshoot the same way (G2a fuelIn/fuelNom 1.032 default,
  1.038 tinflow, 1.010 at etaIn 1000; foam 1.005). **Mechanism (verified
  2026-10-30, `plans/verify/verify_foamflame.md`)**: a STEADY LODI inflow
  cell has dp/dt = du/dt = 0, so its relaxed wave must cancel the row's own
  transverse sources: L5 = S_p + ρc S_u, i.e. u − u_row = (S_p + ρc S_u)
  L_x/(etaIn (1 − M²) ρc²). The row carries the Burke–Schumann T(y) (bunsen:
  the cold-jet/burnt-coflow T step, plus the re-added q ω): its y conduction
  alone, S_p = (γ − 1) κ ∂²T/∂y² from dns' row-0 frames, predicts the fuel
  excess 0.96 / 0.10 / 0.97 vs measured 0.89 / 0.10 / 0.90 (etaIn 100 /
  1000 / tinflow; 4.1 vs 2.7 at etaIn 20, nonlinear), bunsen 3.8 vs 3.3 /
  0.42 vs 0.44. The parameter is etaIn/L_x: `--length 160` (L_x 168) at
  etaIn 161.5 reproduces the default (fuelIn 9.623 vs 9.617, tip 51.53 vs
  51.20), at etaIn 100 the excess grows ×1.50 (fuelIn 9.983, tip 53.39).
  Direct: the bunsen nz 8 twin's row-0 |u| is +2.1–2.4 S_L above u_row at
  the jet edge (+50%), −1.6e-3 on the axis. issues.md §68.
- **At etaIn 1000 dns and foam agree to ≤ 0.7%** (one mesh, no ladder:
  not shown to be discretization error): tip −0.7%,
  half-width 0.00%, fuelIn −0.3%, TmaxIn −0.07%, sheet T −0.05%, Z field
  L2 2.8e-3. What remains differs in kind: dns' pinned row is a CELL that
  cannot burn (base row 1, x_ω,max 1.58 vs 1.38 l), foam's fixed FACE
  half a cell upstream conducts and diffuses through it (5% of foam's fuel
  enters by Fick, 0.445 of 8.987; dns' 0|1 diffusion 0.345 of 9.617);
  foam's box sits 0.4% above p0 (waveTransmissive vs the σ 4 anchor).
- **The flame sheet burns at 0.947 T_ad mid-flame, 0.985 T_ad at its
  hottest** in all four runs (Da 1000: the finite-rate deficit; `gpu_diffusion_heat`'s
  Da^(−1/3) law), with ω peaking at Z = 0.487–0.497 vs Z_st 0.5.
- **The scheme choice is invisible here**: mclin vs mvsel tip −0.10%, burn
  +0.9%. Independent MC limiters cost Z(Y) − Zt 4.2e-7 only: in FP64 a
  consumed reactant is EXACTLY 0 (clamp) and OpenFOAM's r of a uniform field
  is 1999 (limiter 2, no vote), so per-species limiters differ only inside
  the reaction zone; dns' FP32 noise reactants (~1e-40) are what forced its
  2^-24 vote threshold (§44). The shared limiter (mvsel) holds Zt to 1.6e-10.
- **Shvab–Zeldovich**: foam's fixed face is exact (H 5e-4 on the first
  row) where dns' pinned row carries 0.35 (§45; 0.13 at etaIn 1000, 0.85 at
  20). Past the bands foam's H grows to 0.023–0.053 over the last 30 l
  before its zeroGradient outflow (≤ 0.0034 in the flame); dns 0.016
  (window mean of the max, sitting on the first cell past the inflow band
  or the last before the outflow band; 0.014 without the last 10 l).
- **Settling**: dns rings (tip 55.0 → 50.9 → 52.3 → 51.3 by t 48, then
  ±0.15); foam creeps monotonically (62.4 at t 32 → 47.55 at 60 → 47.12 at
  68 → 47.14 at 72): its window halves differ by 3e-3.
- **Time-converged foam** (restarted from t = 60, compared on [64, 68]):
  dt/2 moves tip −9.3e-5, half +3.2e-5, TmaxIn −1.8e-5, burn/in −2.0e-4,
  pMean +1.2e-4; nOuter 3 tip +4.4e-5, half +7.5e-5, TmaxIn −1.2e-4, burn/in
  −1.0e-4 — the lagged chemistry at dt J 0.63 and PISO's ddtCorr are below
  2e-4 on every observable. foam's own fuel budget (burn / face inflow)
  closes to 0.9953 independent of dt and nOuter (not attributed). nOuter 3
  raises the tracer check max |Z(Y_F, Y_O) − Zt| 4.2e-7 → 4.8e-5 (dt/2:
  unchanged) — not attributed either.

Cost: dns 0.373 ms/step (2D, tt 80, 15.5 s); foam 3000 steps, 243 s wall
per variant on 7 ranks, the two variants concurrently (14 ranks, job 267
s); the checks (8 l/U each, concurrent) 53 / 75 s.

### G3b: candle (built, the flame not compared; 2026-10-30)

reactingFoam has no gravity; `rhoReactingBuoyantFoam` (same source tree)
solves p_rgh = p − ρ g·x with heRhoThermo but takes reactingFoam's YEqn.H
and EEqn.H, whose energy equation has NO ρ U·g (buoyantPimpleFoam's has;
dns' E gains −g m_x): the twin adds it as a coded fvOption on h (v2606
registers g as a `meshObjects::gravity` on Time — a mesh lookup fails — and
the buoyant solver needs `fluxRequired { p_rgh; }`). Hydrostatic faces:
inlet fixedValue U, T, Y + p_rgh `fixedFluxPressure`; top p_rgh
`prghPressure` at the ambient static p_a(L) (dns: NSCBC p → p_a(x_out) with
the hydrostatic incoming wave), U `pressureInletOutletVelocity` (normal
inflow, tangential 0: dns' v → 0), T, Y `inletOutlet` with ambient air
(dns: ρ → ρ_a, yHi = air); IC p = p_a(x). At rest (`--quiescent`, 4 cells/l,
tt 10): foam max |u| 1.35e-6 U, |p/p_a − 1| ≤ 2.3e-8 (p_rgh is
well-balanced; its inlet is a u = 0 wall); dns max |u| 8.7e-4 U.

The flame comparison was NOT run: foam at the documented 8 cells/l (57.6k
cells, dt ≤ 2e-3 l/U for the chemistry, 16.5k steps) is ~11 min CPU; at 4
cells/l (16k cells, 0.016 s/step, ~4 min) the dns candle is no longer the
documented steady flame (tip 10.29 mean, 7.79–11.31 over [20, 30], std 0.93
over [15, 30] vs 0.27 at 8 cells; Z hits up to 10025 at t 0.34), so window
means of two flapping flames compare phase. The case dict (`G3b`, ladder
[4]) runs as is; a budgeted run is `ladder [8]`.

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
the inflow → `UNDER-RESOLVED DNS (h/eta=…)` and a warning. Every premixed label
(1D–3D, Bunsen, chamber, wick) names `sigma=…` and `Zeldovich beta=…` after
`Le=1` ONLY when they differ from the defaults 5 and 8 (default labels byte
for byte as before; stdout only). `--dry-run`
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
+26% / +39%) and falls in U3 (−16%); the window-halves gate (7%) fails in all
three. Measured since (TPRI lesgap, "Why TFLES misses the U2 DNS" below): the
15 δ_th box is 24.0 δ_D ≈ 1.15 λ_c at σ 5, so its box mode is
Darrieus–Landau-UNSTABLE (+0.048 / +0.057 /τ at 4 / 6 cells, 2D single mode),
and the U2 S_T depends on the seed (4-cell mean over [20, 80): 1.912 seed 1,
3.325 seed 2, ×1.74), the box width (2× L_y, L_z: 3.090 vs 2.118 over [20,
40), ×1.46) and the 10 τ block (1.69–2.30, tt 80). S_T is
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
−0.0002 / +0.0003 in S_T/S_L). Resolution alone is not the loss (4 cells
DNS / LES-σ within 9% of the 6-cell DNS); THICKENING is (F 1.5 already
−30%); the sensor is not (F 1.5 with the UNIFORM sensor reads 1.184 vs 1.186
dynamic). Why: the next subsection (the box mode). Internally the LES is
consistent: I0 = S_T/(A_T/A) = 0.996 / 1.002, resolved A_T ≤ A_T,DNS.

Regenerate: `runs/flames/turb/p2/les.sh` and `les2.sh` (2.1 + 1.3 min GPU),
then `tools/chem/turb_stats.py PREFIX --t0 10 --t1 20 --sl S_L(grid) --mach
0.02`.

### Why TFLES misses the U2 DNS: the Darrieus–Landau box mode (TPRI lesgap, 2026-10-29)

Plan `plans/masterplan_lesgap.md`, report `plans/verify/impl_lesgap.md`;
gfx1100, simjob. Not a code defect, and not closable by an efficiency
function.

**TFLES stretches the DL dispersion relation by F**: s_F(k) = s_1(F k)/F,
so the neutral wavelength becomes F λ_c. Single 2D wrinkle mode (`--wrinkle
0.5 --wrinkle-modes 1`, upstream / downstream 16, `--inflow-speed 1`, M 0.02,
tt 20, frames every 0.5 from t = 0 (frame k = the first step with t ≥ k·0.5;
41 frames); s = slope of ln a over t ∈ [4, 16], a = mode 1 of
x_I(y) = h Σ_i Y; [2, 10] fit in brackets; `tools/chem/wrinkle_rate.py`),
σ 5, rates in 1/τ:

| λ (δ_D) | DNS 6 cells | DNS 4 cells | F 1.5 @ 4 | F 2 @ 3 | F 3 @ 2 |
|---|---|---|---|---|---|
| 12.0 (ny 30, all at 4 cells) | – | −0.62 (−0.85) | −0.52 | −0.60 | −0.57 |
| **24.0 (the U2 box, 15 δ_th)** | **+0.057 (+0.054)** | **+0.048 (+0.032)** | **−0.051** | **−0.19** | **−0.39** |
| 48.1 | – | +0.047 (+0.039) | +0.054 | +0.024 | −0.038 |

(λ 12 and F 3 @ λ 24 are fit-limited: a reaches ~0 within 6 τ. The verifier
re-timed the frames: the first lesgap tables put frame k at (k+1)·0.5, one
frame late, which moved these rates by up to 0.04 at the fit-limited entries
and ≤ 0.002 elsewhere; no conclusion changed.) Scaling: s_F2(λ 48) = +0.024
vs s_1(λ 24)/2 = +0.024 (4 cells) / +0.029 (6 cells); two estimates of s_1
at λ 16 (F 1.5 at λ 24: −0.077; F 3 at λ 48: −0.11) agree in sign and size.
λ_c ≈ 21 δ_D ≈ 13 δ_th at σ 5 (4–6 cells, interpolated), so the TFLES
cutoffs are ≈ 31 / 42 / 63 δ_D at F 1.5 / 2 / 3: the U2 box mode grows in
the DNS and decays under every thickening. Pinned by `gpu_tfles_dl`.

**The missing area is RESOLVED, mostly the box mode** (`tools/chem/
front_filter.py` on the U2 seed-1 volumes; Pope's Gaussian exp(−6x²/Δ²), Δ in
δ_th, Ξ_Δ = A/A_Δ; shell 1 = the box fundamental):

| t | DNS A | Ξ(1.5 δ_th) | Ξ(2) | Ξ(3) | shell-1 share | rms x_I (δ_D) |
|---|---|---|---|---|---|---|
| 10 | 1.197 | 1.009 | 1.015 | 1.032 | 0.76 | 2.18 |
| 14 | 1.600 | 1.015 | 1.026 | 1.058 | 0.86 | 4.35 |
| 20 | 2.186 | 1.024 | 1.042 | 1.093 | 0.84 | 6.51 |

At t = 20 the LES's resolved A against the DNS's A filtered at its Δ = F δ_th:
F 1.5 1.298 / 2.135 (0.61, Ebar 1.00001), F 2 1.143 / 2.097 (0.545, Ebar
1.0004), F 3 1.059 / 2.000 (0.53, Ebar 1.024); rms x_I 2.9 / 2.0 / 1.2 δ_D
against the DNS's 6.5. A ~40% deficit at resolved scales against ≤ 9%
sub-filter: a perfect efficiency (E = Ξ_Δ) lifts LES2-C from 0.64 to ≤ ~0.66
of the DNS.

**The σ 5 U2 reference is not a statistic** (`runs/flames/lesgap/a1`; S_T
normalized by the grid's laminar S_L, 10 τ block means):

| run | [10,20) | [20,30) | [30,40) | [40,50) | [50,60) | [60,70) | [70,80) | mean from t = 20 |
|---|---|---|---|---|---|---|---|---|
| DNS 6c s1 (tt 60) | 1.695 | 2.283 | 1.949 | 1.675 | 1.641 | | | 1.887 [20,60) |
| DNS 4c s1 | 1.823 | 2.304 | 1.931 | 1.831 | 1.691 | 2.022 | 1.694 | 1.912 |
| LES2-C s1 | 1.086 | 1.097 | 1.056 | 1.065 | 1.063 | 1.057 | 1.063 | 1.067 (0.558×) |
| DNS 4c s2 | 2.351 | 2.669 | 3.476 | 3.388 | 3.507 | 3.424 | 3.485 | 3.325 |
| LES2-C s2 | 1.179 | 1.108 | 1.204 | 1.126 | 1.198 | 1.145 | 1.186 | 1.161 (0.349×) |
| DNS 4c, 2× width (120²) | 2.386 | 3.246 | 2.934 | | | | | 3.090 [20,40) |
| LES2-C, 2× width (90²) | 1.188 | 1.575 | 2.354 | | | | | still growing |

The ratio does not converge with a longer window (0.558× / 0.349× per seed
over [20, 80)); the wide LES grows because its 48 δ_D box mode is
DL-unstable at F 2 (+0.024). I0 0.98–1.09 in every block.

**σ and l_t** (analyst 3, `a3`; laminar S_L at σ 2 below):

| case | window | DNS 4c | F 1.5 @ 4 | F 2 @ 3 | F 3 @ 2 |
|---|---|---|---|---|---|
| U2 σ 5, l_t 2.5 (P2) | [10, 20] | 1.835 | 0.646× | 0.593× | 0.568× |
| U2 σ 2, l_t 2.5 | [10, 20] | 1.358 (halves 1.24 / 1.47) | 0.882× | – | – |
| σ 5, l_t 6 (box 57.7 δ_D), tt 30 | [10, 30] | 3.073 (halves 2.71 / 3.43) | – | 0.785× | 0.565× (E = 1: 0.566×) |
| σ 5, l_t 6 | [5, 12] | 1.591 | – | 0.829× | 0.755× |
| σ 2, l_t 6, tt 12 | [5, 12] | 1.358 | – | 0.936× | 0.897× |

The l_t 6 ordering is DL again: 57.7 δ_D is above F 2's cutoff (unstable at
F 2: 0.785×) and at F 3's (≈ neutral: 0.565×).

**Why the u'_in 0.25 control (P2 verifier) saw no DL.** The box-mode rate is
only +0.05/τ (×2.7 in 20 τ from a tiny seed) and the area excess is second
order in the amplitude: ~1e-3 is expected. In the u' 4 runs the turbulence
seeds 1.4–2.2 δ_D of front amplitude by t = 8–10, and the box mode grows from
there.

**Laminar extruded Bunsen** (P3 burner flags, nz 8, tt 6, TFLES 3,
Charlette): FEmax = 5.196 = 3√3 and Ebar 1.0048; its E = 1 twin reads area
0.9447 vs 0.9442. Colin's operator saturates E on LAMINAR jet shear, so
"FEmax 3√3 every row" is not a turbulence signature (issues.md B1).

### LES vs DNS at σ* = 2 (the DL-stable U2 box; TPRI lesgap, 2026-10-29)

**σ* selection** (2D dispersion scan, `runs/flames/lesgap/mp/disp/disp.sh`,
λ = 15 δ_th(σ) = the U2 width, s over [4, 16], [2, 10] in brackets; δ_th/δ_D
= 1.64959 at σ 3, 1.7953 at σ 2):

| run | λ | s (1/τ) |
|---|---|---|
| σ 3 DNS 6 | 15 δ_th = 24.7 δ_D | +0.0098 (+0.0020) |
| σ 3 DNS 4 | 15 δ_th | −0.0007 (−0.0105) |
| σ 2 DNS 6 | 15 δ_th = 26.9 δ_D | −0.0159 (−0.0231) |
| σ 2 DNS 4 | 15 δ_th | −0.0194 (−0.0257) |
| σ 2 DNS 4 | 30 δ_th = 53.9 δ_D | +0.0105 (+0.0061) |
| σ 2 F 2 @ 3 | 15 δ_th | −0.1119 (−0.1230) |
| σ 2 F 2 @ 3 | 30 δ_th | −0.0137 (−0.0199) |
| σ 2 F 3 @ 2 | 15 δ_th | −0.2228 (−0.2251) |

(Frame k at t = k·0.5, verifier-corrected as above.) σ 3 is not DL-stable at
the U2 width (the 6-cell rate is +0.010), σ 2 is (both grids ≤ −0.01):
**σ* = 2**. Scaling at σ 2: s(F 2, λ 30) / (s(DNS 4, λ 15)/2) = 1.411 (gate
[0.6, 1.4]: FAILS, marginally; it read 1.397, a pass, with the frames one
late — at σ 2 the λ 30 and λ 15 rates are both small, so the ratio is
fragile; issues.md L8). Note that at σ 2 the thickened flame still damps the
box mode 5.8× (F 2) and 11.5× (F 3) faster than the DNS.

**Runs** (`runs/flames/lesgap/mp/camp/camp.sh`; U2 flags `--upstream 16
--downstream 16 --hold-flame --inflow-speed 1.2 --turb 4 --mach 0.02
--expansion 2`, grids as P2 (DNS6 90², DNS4 60² sink 11, LES2 = σ 1.35 + TFLES
2 @ 3 cells 45² sink 8, LES4 = TFLES 3 @ 2 cells 30² sink 5), window [20, 40],
S_T = SL / laminar S_L of the grid at σ 2 (M 0.02, tt 40, [20, 40]): 6 cells
0.995855, 4 cells 1.007845, F 2 @ 3 0.996470, F 3 @ 2 0.995677). Labels read
`DNS [premixed single-step, …, Le=1, sigma=2, M_flame=0.02 …; 3D nz=90,
synthetic HIT inflow u'/S_L=4, l_t=2.5 delta_th (… seed 1; Re_t=24.9,
h/eta=0.744 at the inflow) …]` and `LES [sigma-model Csigma=1.35; …] + TFLES
[F=2, dynamic sensor (…), Charlette 2002 efficiency beta=0.5 (delta_L=
delta_th), u'_Delta by Colin's operator|no efficiency (E=1)] [premixed …,
sigma=2, …; h/eta=1.49 …]` (F 3: h/eta 2.23).

| run | S_T/S_L | halves | A_T/A | I0 | brush (δ_D) | u'_flame | Ebar | FEmax |
|---|---|---|---|---|---|---|---|---|
| DNS6 s1 | 1.459 | 1.518 / 1.401 | 1.421 | 1.027 | 7.73 | 0.924 | – | – |
| DNS6 s2 | 2.761 | 2.856 / 2.668 | 2.542 | 1.086 | 12.48 | 1.409 | – | – |
| DNS4 s1 | 1.590 | 1.582 / 1.599 | 1.531 | 1.039 | 8.42 | 0.952 | – | – |
| DNS4 s2 | 2.759 | 2.938 / 2.579 | 2.502 | 1.102 | 12.78 | 1.421 | – | – |
| LES2-C s1 | 1.212 | 1.243 / 1.181 | 1.190 | 1.018 | 8.14 | 0.803 | 1.0157 | 2.83 |
| LES2-C s2 | 1.410 | 1.427 / 1.393 | 1.372 | 1.028 | 11.10 | 1.168 | 1.0367 | 2.83 |
| LES2-N s1 | 1.210 | 1.243 / 1.180 | 1.188 | 1.019 | 8.07 | 0.814 | 1 | 2.00 |
| LES4-C s1 | 1.107 | 1.130 / 1.087 | 1.091 | 1.015 | 7.54 | 0.788 | 1.0956 | 5.20 |
| LES4-C s2 | 1.155 | 1.194 / 1.118 | 1.118 | 1.033 | 8.22 | 1.111 | 1.1467 | 5.20 |
| LES4-N s1 | 1.109 | 1.132 / 1.089 | 1.094 | 1.014 | 7.54 | 0.787 | 1 | 3.00 |

Seed spread |s1 − s2|/mean: DNS6 0.62, DNS4 0.54, LES2-C 0.15, LES4-C 0.04 —
the seed dependence survives in a DL-STABLE box: it is the realization, not
DL alone. The seed-2 reruns with volumes reproduce every stats row byte for
byte.

**Gates** (paired per seed; a gate passes only if it holds for both):

| # | gate | seed 1 | seed 2 | verdict |
|---|---|---|---|---|
| G1 | DNS6 halves ≤ 7% | 8.0% | 6.8% | FAIL (s1) |
| G2 | DNS4/DNS6 ∈ [0.90, 1.10] | 1.090 | 0.999 | pass |
| G3 | LES2-C/DNS6 ≥ 0.85 | 0.831 | 0.511 | FAIL |
| G3e | LES2-C excess ≥ 0.64 | 0.461 | 0.233 | FAIL |
| G4 | LES4-C/DNS6 ≥ 0.75 | 0.759 | 0.418 | FAIL (s2) |
| G4e | LES4-C excess ≥ 0.39 | 0.234 | 0.088 | FAIL |
| G5 | Ebar A_T,LES/A_T,DNS: LES2 ±20%, LES4 ±30% | 0.851 / 0.841 | 0.559 / 0.504 | FAIL (s2) |
| G6 | brush/DNS: LES2 ±30%, LES4 ±40% | 1.053 / 0.976 | 0.890 / 0.659 | pass |
| G7 | \|LES4-C − DNS\| < \|LES4-N − DNS\| (s1) | 0.3515 vs 0.3496 | – | FAIL (expected); LES2: 0.2472 vs 0.2484, pass |
| G8 | I0 ∈ [0.8, 1.2], window means | 1.014–1.102 over all 10 runs | | pass (per row 0.971–1.223: DNS4 s2 rows exceed) |
| G9 | LES2-C/DNS4 [40, 80] − [20, 40] within ±0.05 (report) | 0.813 − 0.762 = +0.051 | – | FAIL (marginal) |

Verdict: seed 1 is INCONCLUSIVE (G1, reference non-stationary); seed 2 has G1
and G2 passing and fails G3, G3e, G4, G4e, G5 — **a non-DL gap remains** at
σ 2. The DL box mode explains the σ 5 numbers; it does not explain these.

**Sub-filter share at σ 2** (front_filter on the d6 volumes, t = 20 / 30 /
40, means): Ξ(2 δ_th) = 1.040 (s1) / 1.073 (s2), Ξ(3 δ_th) = 1.079 / 1.153.
The LES's resolved A over the DNS's A filtered at Δ = F δ_th: LES2 0.83 /
0.84 / 0.96 (s1), 0.63 / 0.58 / 0.53 (s2); LES4 0.79 / 0.81 / 0.95 (s1),
0.57 / 0.52 / 0.47 (s2). Shell-1 (box mode) share of the DNS front variance
0.70–0.86; rms x_I 2.0–4.0 (s1) / 6.3–9.0 δ_D (s2) against LES2 1.6–2.6 /
3.0–4.1. The most an efficiency could restore: S_T,LES Ξ_Δ / S_T,DNS = 0.863
/ 0.548 (LES2), 0.819 / 0.482 (LES4); with the plan's /Ebar: 0.850 / 0.529,
0.748 / 0.420. So the deficit is again at resolved scales, mostly the box
mode, which the DNS holds at 3–9 δ_D rms although it is linearly DL-stable
(−0.02/τ): turbulence forces it, and the thickened front relaxes it 5.8–11.5×
faster (s_F(k) = s_1(F k)/F). Second contributor, not separated: u'_flame is
13–15% (s1) / 17–21% (s2) lower on the LES grids (l_t = 7.5 / 5 cells; the
s2 DNS value is inflated by the brush's leading edge in the `uPrime` slab,
issues.md T2).

**The efficiency does not reach the consumption speed.** At σ 2 Ebar is
1.016 / 1.096 (F 2 / F 3, s1) and 1.037 / 1.147 (s2) — at or above the
a-priori Ξ_Δ at F 3 — yet C − N moves S_T by +0.0012 (F 2) and −0.0019
(F 3), I0 1.0155 vs 1.0144. Ebar is an Ω-weighted mean over the sensor
region, which extends 2.5 F δ_th into the turbulent fresh gas; a cell-local
E > 1 in the preheat zone with E ≈ 1 at the reaction layer would explain it
(not measured: E is not in the frames). Next TPRI (issues.md L9).

**TFLES-regime point** (Step 3b, `runs/flames/lesgap/mp/lt6/lt6.sh`: σ 2, l_t
6 δ_th, upstream 28, downstream 20, box 36 δ_th = 64.6 δ_D = 6 l_t, tt 30;
LES with σ 1.35 + Charlette): the DNS4 (214 × 144²) exits 8 at t
26.9 (flash-back guard: the brush reached the inflow band; 119 s), so the
window is [10, 26.5]: DNS4 2.471 (halves 1.90 / 3.01), F 2 @ 3 1.883 (0.762×, Ebar
1.009), F 3 @ 2 1.606 (0.650×, Ebar 1.072); [10, 20]: 1.989 / 0.841× /
0.740×; [5, 12]: 1.336 / 0.936× / 0.898× (a3's tt-12 ratios; a3's 1.358
above is the same run plus its final t = 12 stats row). The 64.6 δ_D
box is DL-unstable at σ 2 in the DNS (λ 53.9: +0.011) and stable at F 2 (λ
53.9: −0.014), so this point carries the same box-mode bias; the ratio falls
as the DNS's S_T grows. No 6-cell reference. Report only.

**`val_turbflame_les` is not registered** (G3–G5 fail at σ*; P2's decision
stands).

Regenerate: `runs/flames/lesgap/mp/{disp/disp.sh, lam/lam.sh, camp/camp.sh,
camp2/camp2.sh, lt6/lt6.sh}` (simjob --gpu; 0.3 + 0.2 + 5.4 + 2.0 + 3.2 min),
then `runs/flames/lesgap/mp/camp/gates.py`, `tools/chem/turb_stats.py PREFIX
--t0 20 --t1 40 --sl S_L(grid) --mach 0.02 --sigma 2 --lt 4.488`,
`tools/chem/wrinkle_rate.py PREFIX --ny N --cells C --dth 1.7953`,
`tools/chem/front_filter.py PREFIX --n N --cells C --dth 1.7953 --t-first 20`.

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
1 is not a turbulence effect (issues.md B4). At σ 5 the burner's span (2W
= 48 δ_th ≈ 77 δ_D) admits Darrieus–Landau modes that TFLES 3 suppresses
below ~63 δ_D ("Why TFLES misses the U2 DNS"), so this S_T is a TFLES number
with that bias; FEmax = 3√3 also occurs in the LAMINAR extruded burner
(Ebar 1.0048). Damköhler (1 + u'_in) 4, Peters 1999 4.23 at the
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
| `gpu_tfles_dl` | GPU binary | Darrieus–Landau under thickening, σ 5, one 2D wrinkle mode: the U2 width grows in the DNS at 4 cells (s ≥ +0.02/τ; +0.0479), decays at F 2 (≤ −0.10; −0.1909); twice the width at F 2 grows at half the DNS rate (ratio in [0.6, 1.4]; 1.00) (~10 s) |

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
