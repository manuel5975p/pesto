# Chemistry: `dns flame` (P0 oracle + P1 premixed flame)

Spec of the single-step premixed flame. Plan and measurements:
`plans/chem_plan.md`, `plans/impl_chem.md`. Open items: `issues.md` §30–38.

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
uFresh, pFresh, TFresh`. SL is the consumption speed ∫ω dx / (ρY)_ref with
the fresh reference column 2 δ_th inside the inflow band; SLdisp =
u_ref − dx_f/dt; x_f the mid-temperature crossing; dth = (Tb − Tu)/max dT/dx
(parabolic peak); omegaEdge = max ω in the sink bands / max ω. Profiles
(`--profile-dt`): `t, x, T, rho, u, p, Y, omega`. Frames: `temp` (T − Tu),
`fuel` (Y), `omega` — names live in the manifest, so no viewer-format bump.
Exit 8 when x_f comes within band + 2 δ_th of either face.

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

## Tests

| test | kind | content |
|---|---|---|
| `flame1d` | host | oracle convergence, √A similarity, profile, vs Cantera |
| `flame_config` | host | grid, IC, dry-run label/Λ, every rejection by name |
| `gpu_flame_passive` | GPU API | species off/on bitwise mean flow, uniform Y, passive copies, boundedness |
| `gpu_flame_conservation` | GPU API | mass, E + qρY, fuel vs rate tally |
| `gpu_flame1d_grid` / `_mach` / `_suth` | GPU binary | the tables above |
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
compensated, RANS/LES species (asserted), nz > 1 (`--nz` must be 1), the
low-Mach branch (`--physics incompressible` rejected), multi-step
chemistry, dns-ui (its editor covers `flow` and `conv` only). Chamber:
isothermal (cooled) walls, a total-pressure inlet, non-premixed injection,
a moving/2D-curved nozzle wall beyond the polygon, 3D.
