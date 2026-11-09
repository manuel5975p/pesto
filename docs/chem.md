# Chemistry: `dns flame` (P0 oracle + P1 premixed flame + 2D slot Bunsen + 2D jet diffusion flame + 2D candle + 3D premixed in turbulence + LES species and TFLES)

Spec of the single-step premixed flame, of the two-reactant diffusion
flame (`--mode diffusion`) and of the candle (`--mode candle`, the diffusion
flame under gravity). Plan and measurements:
`plans/chem_plan.md`, `plans/impl_chem.md`. Open items: `issues.md` §30–53, §64–73, T1–T7, L1–L6.

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

### The bracket and the cold boundary (issues.md 72, masterplan real_bunsen S1)

The shipped bracket (doubling/halving from m = 1, then bisection on a
"too fast" verdict) trusted a MONOTONE verdict. It is not: at m well above
the root the trajectory collapses onto the quasi-steady manifold p* = λ q
ω/(m cp), where RK4 (h J > 2.785, J = λ q ω |dT/du|/p²) can bounce p to a
spurious positive cold residual. One such shot on the halving path made Λ
a power of two: σ 6.5 Sutherland (Tu 298 K) printed Λ = 16383.95 (2^14,
δ_th/δ_D 5.13; correct 941442.56, 4.378), σ 6.88 42008.4 (δ_th/δ_D 45.5;
correct 882657.41, 4.507). A sweep found 9 (constant λ) + 6 (Sutherland)
such σ in 2..10 at β 8, more at β 4/6/10/14. Now:

- **Too fast** = p hits 0, OR a step goes stiff (h J > 1 at a base point:
  slow shots peak at h J ≈ 2.2e-3, measured σ 1.2–10, β 2–20), OR the cold
  residual is negative.
- **Scan**: m = 2^−60 … 2^20 (81 powers of two) must show EXACTLY one slow →
  fast transition (else an error by name: "non-monotone shot at m in [a,
  b]; raise n", "no slow/fast bracket"). Bisection then runs over the
  SHIPPED bracket, [2^j, 1] below 1 or [1, 2^(j+1)] above — what the old
  halving/doubling found when the verdict was monotone — so every
  well-posed problem keeps its bisection sequence: Λ bit for bit (`flame1d`
  (g): σ 5 β 6/8/10, σ 7.5 Sutherland, σ 3; sweep: 359 + 366 of 401 σ at β
  8 identical, every difference a spurious old value or a cold-cut case).
- **Cold cutoff** (`kColdCut` 0.5, `dns/chem.hpp`): the unburnt mixture
  reacts at e^(−β σ) of the peak rate. Where p*/(m cp (T − Tu)) climbs back
  above 0.5 on the cold side after the preheat zone dipped below it, the
  shot switches the rate OFF (the classic ignition-temperature cutoff,
  placed adaptively; `FlameEigen::tCut`, printed by `--dry-run`). The
  uncut march was not an eigenvalue there: σ 2 β 8 drifted as 1/n (Λ
  4.72e8 / 1.24e8 / 0.31e8 at n 2e4 / 8e4 / 3.2e5) — the captured fast
  shots end with a POSITIVE residual p*(T_end) once p* exceeds m cp (T_end
  − Tu). Cut: n-converged (4e-8 at σ 2), 4e-5 per decade of the bound (σ
  2: 5.0312 / 5.0313 / 5.0315 / 5.0317e8 at 0.5 / 0.1 / 0.01 / 0.001). The
  cut is inactive (bit for bit the uncut march) wherever p*/(m cp (T −
  Tu)) stays below 0.5 to the cold end: β 8 from σ 2.66 (constant λ) /
  2.58 (Sutherland). A cut more than `kColdCutMaxRise` = 1e-4 of the rise
  above Tu, or no preheat zone at all, is an ERROR by name ("cold-boundary
  difficulty"), not a number: there Λ is the cutoff's (bound 0.5 vs 0.1
  moves Λ 3.5e-5 at a cut 1.3e-5 above Tu, 0.4–0.6% at 1–2e-3, > 2% beyond
  5e-3). Rejected (σ step 0.05): β 8 σ ≤ 1.70, β 6 ≤ 2.20, β 4 ≤ 3.25, β 3
  ≤ 4.25, β 2 ≤ 6.20 (constant λ; Sutherland 0.05–0.45 lower). The old
  march printed a number there that was no eigenvalue: at each of these
  boundary points Λ(n 8e4)/Λ(n 2e4) = 1/4 (the 1/n drift; β 2 σ 2 printed
  Λ 0.09).
- **σ 2, β 8 (the lesgap / G2b normalisation) moves**: Λ 472463262.4 →
  503116277.2 (+6.5%), δ_th/δ_D 1.7953 → 1.72336 (−4.0%). Measured on the
  GPU (15 cells, M 0.01, sliding, window [10, 20]): old Λ burns at S 0.9747,
  new at 1.0023 (σ 5's 15-cell offset is +0.27%) — the new Λ is the
  solver's S_L = 1. Those campaigns normalised by their measured grid S_L
  (ratios survive); their δ_th-scaled lengths used 1.7953.
- `rateFactor(T, s)` (`FlameProblem`): an optional multiplier of the rate
  (and of the hot-end B at s = 0); empty = 1, the arithmetic bit for bit
  (`flame1d` (g): a factor ≡ 1 reproduces the empty hook's table bitwise).

The numpy port (`tools/chem/foamflame/model.py`, `Oracle`) carries the same
stiff test, cutoff and scan (σ 2: Λ 503116277.2, δ_th 1.72336, as dns).

### Model C2: the rich two-reactant premixed jet (`--equivalence`)

The partially premixed burner's jet (masterplan real_bunsen, D1–D4):
CH4/air at φ > 1 on model C1's step F + s O → (1+s) P (`arrheniusRate2`,
ω = A ρ² Y_F Y_O e^(−Ta/T)), Le = 1, equal W. Fuel per air f = φ Y_O2/s,
Y_F1 = f/(1+f), Y_O1 = Y_O2/(1+f); O is deficient, the inner flame leaves
Y_F,rem = (φ − 1) Y_O1/s, which burns against the air at Z_st = (Y_O2/s)/
(Y_F,rem + Y_O2/s). q per unit fuel is fixed by the INNER flame: σ = 1 + q
(Y_O1/s)/(cp Tu) (`--expansion`). The outer Burke–Schumann sheet then sits
at σ_st = 1 + q Z_st Y_F1/(cp Tu).

Oracle (`premixedEigen2`): progress s_p = (Tb − T)/(Tb − Tu) = Y_O/Y_O1,
Y_F = Y_F,rem + Y_O/s, so q ω_F = cp ΔT · A ρ s_p e^(−Ta/T) · [ρ (s Y_F,rem
+ Y_O1 s_p)] — the bracket is `rateFactor`; m ∝ √A still (Λ2 = 1/m(A =
1)², `flame1d` (i): m(4A)/2m(A) − 1 < 1e-10; φ → ∞ reproduces the
one-reactant eigenvalue × √(s Y_F,rem) to 1e-7).

| φ 1.3, s 3.99, Y_O2 0.233, σ 6.88, β 8, Sutherland Tu 298 K | |
|---|---|
| Y_F1 / Y_O1 / Y_F,rem | 0.0705584 / 0.216560 / 0.0162827 |
| Z_st, σ_st | 0.781963, 6.97733 |
| Λ2, δ_th/δ_D | 46986660.73, 4.70964 (one-reactant at σ 6.88: 882657.41, 4.50704) |

`dns flame --equivalence PHI` (premixed mode; `--stoich` 3.99, `--ox-y`
0.233): nY 2 (fuel 0, ox 1, `kChemCellPass2` / `kSpeciesStage2<false,
false>`), fresh Y_F1, Y_O1 carried in, q = cp Tu (σ − 1) s/Y_O1, IC
`flameIC2` (the oracle profile, Y_O = Y_O1 s_p, Y_F = Y_F,rem + Y_O/s: Z = 1
to 2e-16). SL = ∫ω dx / (ρ (Y_F − Y_F,rem))_ref (`FlameRun::yFBurnt`; 0 =
the shipped expression bit for bit); stats append zMin, zMax. Rejected by
name: `--mode bunsen` (needs `--physics lowmach`, not built), diffusion /
candle / chamber, `--fuel-y`, `--damkohler`, `--cells-per-unit`,
`--ignition`, `--les` (structured; the FV route takes it: docs/fvcomp.md "Air-methane"),
`--tfles`, `--turb`; φ outside [1.05, 1.6]. Label:
`DNS [partially premixed two-reactant F + 3.99 O -> 4.99 P (rich jet
phi=1.3, s=3.99, Y_O2=0.233: …), unit S_L at phi, …]`.

**Measured** (`gpu_flame_rich1d`, gfx1100, 2026-10-31; σ 6.88, Sutherland
298 K, M 0.01, sliding 0.98, window [10, 20]):

| cells/δ_th | S rich (C2) | S one-reactant (C0) | rich − single |
|---|---|---|---|
| 7.5 | 1.015612 | 1.015058 | +5.5e-4 |
| 15 | 1.002544 | 1.002695 | −1.5e-4 |

yHits 0, omegaEdge < 1e-20. Z (on Y_F − Y_O/s) stays within ±1.7e-4 of 1
(both grids; 1.2e-4 at M 0.02, 2.4e-5 at tt 2): the FP32 transport walk
of non-power-of-two mass fractions, amplified by s/(s Y_F,rem + Y_O2) = 13
(gate 5e-4; the plan's 1e-5 is below the walk).

**The model's own S_L(φ) has the WRONG trend** (`--dry-run` table at fixed
A, q, Ta; `flame1d` (i) prints it beside GRI-3.0): S_L(φ)/S_L(1.3) = 0.755 /
0.816 / 0.917 / 1 / 1.068 / 1.126 / 1.174 at φ 1.05 / 1.1 / 1.2 / 1.3 /
1.4 / 1.5 / 1.6; GRI-3.0: 1.634 / 1.618 / 1.418 / 1 / 0.590 / 0.431 /
0.342. One step's rate ∝ Y_F Y_O starves as Y_F,rem → 0 toward φ 1; real
CH4 burns FASTER there. The masterplan's D4 assumed the same sign (issues.md
73).

**GRI-3.0 reference** (`tools/chem/cantera_flame.py --real`, Cantera 3.2.0
in a venv under `runs/`, mixture-averaged, Tu 298 K, 1 atm;
`tests/data/chem/ch4_gri30.csv`, reference only):

| φ | S_L [m/s] | T_ad [K] | σ_ad | δ_th [mm] | α_u [m²/s] | δ_th/δ_D |
|---|---|---|---|---|---|---|
| 1.0 | 0.3709 | 2224.5 | 7.465 | 0.438 | 2.229e-5 | 7.29 |
| 1.05 | 0.3778 | 2231.3 | 7.488 | 0.431 | 2.230e-5 | 7.30 |
| 1.1 | 0.3740 | 2209.2 | 7.413 | 0.429 | 2.231e-5 | 7.19 |
| 1.2 | 0.3278 | 2135.2 | 7.165 | 0.464 | 2.232e-5 | 6.81 |
| 1.3 | 0.2312 | 2056.0 | 6.899 | 0.619 | 2.234e-5 | 6.40 |
| 1.4 | 0.1365 | 1978.4 | 6.639 | 1.003 | 2.235e-5 | 6.12 |
| 1.5 | 0.0997 | 1903.4 | 6.387 | 1.332 | 2.237e-5 | 5.94 |
| 1.6 | 0.0791 | 1831.0 | 6.144 | 1.638 | 2.238e-5 | 5.79 |

The finest rung (slope/curve 0.01) sits 0.2–0.4% below the previous one
(first-order upwind): S_L quoted to ~0.3%. ν_u = 1.591e-5 m²/s at φ 1.3.
Against the masterplan's starred inputs: S_L 0.231 (not 0.25), δ_th 0.619
mm (not 0.70), real δ_th/δ_D 6.40 (not 8.0); σ 6.90 (6.88 used). The D1
trilemma shrinks: model 4.71 (C2) vs CH4 6.40. S1b extended the csv to
φ 0.5–1.7 (34 rows: every 0.05, 0.975/1.025/1.075 at the peak, and the
held-out rows 0.575, 0.625, 1.125, 1.325, 1.375, 1.425, 1.575; `--real
--phis a,b,…` merges a subset, one simjob per ≤ 8 flames).

### Model C2b: S_L(φ) calibrated to GRI-3.0 (`--sl-law gri30`, the `--equivalence` default)

Fixes issues.md 73 (masterplan real_bunsen S1b). In the burner the rich jet
mixes with air at the rim, so the edge flame and the stratified premix burn
across φ_loc ∈ (0, φ_jet]; they must burn at CH4's S_L(φ), not the one-step's
(wrong sign rich, 4.5× slow at φ 0.6). Fernández-Galisteo/Sánchez/Liñán/
Williams-style: the pre-exponential becomes a function of the mixture,

ω = A C(ξ) ρ² Y_F Y_O e^(−Ta/T),  ξ = (s Y_F − Y_O + Y_O2)/(s + Y_O2),

ξ the fuel/air mixture fraction (= Y_F1 of a fresh premix; jet-relative Z =
ξ (s + Y_O2)/(s Y_F,rem + Y_O2)). ξ is the Shvab–Zeldovich coupling
function: conserved by the step, transported exactly by the shared limiter
— so a premixed flame sees ONE C across it, its eigenvalue scales as √C,
and every jet/air mixture is a fuel/air premix whose φ_loc is a function of
ξ alone (jet-independent). **One definition** (`PhiRateLaw<R>`,
`phiRateLn`, DNS_HD in `dns/chem.hpp`): the oracle evaluates it in FP64
(`premix2Problem`'s rateFactor), the cell pass in FP32
(`kChemCellPass2Phi`: `arrheniusRate2(lnA + ln C, …)`), `chemStableDt` in
FP64 (`ChemRate2Phi`). FP32 vs FP64 ln C: 2.9e-6 (`flame1d` (j)).

**Calibration** (`phiRateLaw`, at validate time, threaded, ~0.1 s): node k
at ξ_k = Y_F1(φ_k) of the GRI-3.0 rows (`ch4Gri30Speeds`, the csv's SL
column embedded; φ = 1 skipped: no hot-end start), ln C_k = 2 ln(S_GRI(φ_k)/
m_k) with m_k the UNCORRECTED oracle eigenvalue there (the jet's A = 1, q,
Ta; lean branch below), shifted so ln C(ξ_jet) = 0 — the jet keeps A = Λ2
and S_L = 1, its IC table bit for bit. ln C is a cubic spline in ξ CLAMPED
(zero slope) at the end nodes φ 0.5 and 1.7 and constant beyond: C1
everywhere, C2 inside, no wiggles (monotone on a 4000-point grid; one-sided
slopes at nodes agree to 3.6e-5). Table vs closed form: the table is the
data (26 nodes, 0.4 KB in FP32 by value as a kernel argument; a linear
scan finds the interval); a tanh closed form
(Franzelli PEA) would add its own fit error on top of nothing.

**Lean branch of the oracle** (`premixedEigen2At`, φ < 1): F deficient,
Y_F = Y_F1 s_p, Y_O = Y_O,rem + s Y_F, Y_O,rem = (1 − φ) Y_O1; rateFactor =
ρ (Y_O,rem + s Y_F1 s_p), Tb = 1 + q Y_F1/(cp Tu). The branches meet at φ 1
(m(1 − 1e-4)/m(1 + 1e-4) − 1 = −4.3e-4, slope 1.47 both sides: the one-step
S_L(φ) is C1 through stoichiometric, so C needs no kink there).

**Fit** (`flame1d` (j); σ 6.88, β 8, Sutherland 298 K, jet φ 1.3):

| φ | GRI S_L/S_L(1.3) | one-step | C2b | C | σ model / GRI |
|---|---|---|---|---|---|
| 0.6 | 0.4864 | 0.1083 | node | 20.19 | 4.667 / 5.585 |
| 0.7 | 0.8221 | 0.2150 | node | 14.62 | 5.255 / 6.165 |
| 0.8 | 1.1552 | 0.3594 | node | 10.33 | 5.835 / 6.696 |
| 0.9 | 1.4324 | 0.5257 | node | 7.43 | 6.409 / 7.158 |
| 1.0 | 1.6038 | 0.6864 | +9.2e-5 (held out) | 5.462 | 6.977 / 7.465 |
| 1.05 | 1.6336 | 0.7550 | node | 4.68 | 6.961 / 7.488 |
| 1.1 | 1.6176 | 0.8155 | node | 3.93 | 6.945 / 7.413 |
| 1.2 | 1.4175 | 0.9174 | node | 2.39 | 6.912 / 7.165 |
| 1.3 | 1 | 1 | node | 1 | 6.880 / 6.899 |
| 1.4 | 0.5902 | 1.0684 | node | 0.305 | 6.848 / 6.639 |
| 1.5 | 0.4312 | 1.1257 | node | 0.147 | 6.817 / 6.387 |
| 1.6 | 0.3423 | 1.1740 | node | 0.085 | 6.786 / 6.144 |

Nodes reproduce GRI's ratio to 4e-15. **Held-out error** (rows not in the
law): 0.575 +4.5e-4, 0.625 −1.4e-4, 1.0 +9.2e-5 (at 1 + 1e-4), 1.125
+5.6e-5, 1.325 +1.06e-3, 1.375 −1.85e-3, 1.425 −1.27e-3, 1.575 +3.35e-3
(the clamp's 1.6–1.7 interval reaching in): ≤ 0.34% over 0.575..1.575,
≈ GRI's own grid uncertainty (0.2–0.4%). Leaving every other node out
instead (spacing 0.1) read 2.1% at 1.4 (GRI's S_L bends there) and 3.3% at
1.6 — hence the 0.05 nodes.

**T_ad(φ) is NOT calibrated** (no variable cp; stated): Tb/Tu = 1 + q
Y_def/(cp Tu) on the jet's q is −16% at φ 0.6 (4.67 vs 5.59), −6.5% at 1
(6.98 vs 7.46), −0.3% at 1.3, +10% at 1.6 (6.79 vs 6.14). Lean low: the
jet's q (σ 6.88 at φ 1.3, where CH4 leaves CO/H2) is short of CH4's
complete lean heat; rich high: no CO/H2 dissociation. So C2b's S_L(φ) is
GRI's but the expansion ratio of a lean edge flame is 13–16% low, of a
φ 1.5 premix 7% high. The outer (Z_st) sheet is σ_st 6.977 vs GRI's
stoichiometric 7.465 (−6.5%).

**The diffusion (outer) flame**: φ_loc = 1 at Z_st by definition, so the
outer sheet burns at C(Z_st) = 5.46 × the jet's A — a Damköhler number
5.46× the uncorrected model's (closer to Burke–Schumann, reaction zone
~5.46^(−1/3) = 0.57× as thick). Smooth there: d ln C/dZ = −4.1 at Z_st
(±20% over Z_st ± 0.05). On the air side C saturates at C(φ 0.5) = 23.2,
where Y_F → 0 anyway. Right direction (CH4 near φ 1 is the fastest
chemistry), but the outer cone's Da is now the law's, not a free choice.

**The FP32 Z walk now reaches the rate**: d ln S/dZ = ½ d ln C/dZ = −7.3
at the jet (φ 1.3 is where CH4's S_L(φ) is steepest, d ln S_L/dφ = −4.6).
1D jet at 15 / 30 cells: C2b 1.002132 / 0.996512 vs uncorrected 1.002544 /
0.998796 (Z up to 1 + 1.5e-4 / 4.6e-4: −4e-4 / −2.3e-3), single-reactant
1.002695 / 0.999878. The walk grows with step count; the low-Mach burner
(S4+) must budget it (K10 in the masterplan).

**`--mix-phi PHI`** (premixed mode, with `--equivalence`; [0.6, φ_jet],
|PHI − 1| ≥ 0.01): the 1D/3D flame burns the jet diluted with air to PHI
— A, q, Ta stay the jet's, so S reads S_L(PHI)/S_L(jet). Validator:
`premixedEigen2At(jet, PHI, law)` (Λ the JET's Λ2 — the run's A —, tab.m
the ratio, the table that flame's, δ_th its own so `--cells-per-delta`
counts its cells); inflow Y_F1(PHI), Y_O1(PHI); IC `flameIC2` (lean:
Y_F = Y_F1 s_p, Y_O = Y_O,rem + s Y_F; Z = Z(PHI) to 3e-16); burnt σ =
`sigmaAt(PHI)` for the IC, `FlameRun::sigma`, pB; yFBurnt = Y_F,rem rich,
0 lean. `--sl-law onestep` keeps S1's model (no law, `kChemCellPass2`).
`--sl-law gri30` needs CH4/air (`--stoich 3.99 --ox-y 0.233`, else by
name); both flags rejected by name without `--equivalence`. Label: `S_L(phi)
law C2b: A C(xi) calibrated to GRI-3.0 CH4/air (26 nodes phi 0.5..1.7,
C(jet)=1, C(Z_st)=5.462)` or `uncorrected one-step S_L(phi) (…issues.md
73)`; `burning the jet diluted to phi=… (S_L … of the jet's)`.

**Measured** (`gpu_flame_rich1d_law`, gfx1100, 2026-10-31; jet φ 1.3, 15
cells, inflow 0.98 r, window [10, 20]):

| run | oracle r | S | S/r − 1 | (S/S_jet)/r − 1 |
|---|---|---|---|---|
| jet | 1 | 1.002132 | +2.1e-3 | — |
| --mix-phi 0.7 | 0.822112 | 0.825346 | +3.9e-3 (30 cells +7.6e-4) | +1.8e-3 |
| --mix-phi 1.05 | 1.633614 | 1.635922 | +1.4e-3 | −7.2e-4 |
| --mix-phi 1.2 | 1.417470 | 1.419340 | +1.3e-3 | −8.1e-4 |
| 1.2, `--sl-law onestep` | 0.917369 | 0.920161 | +3.0e-3 | +9.1e-4 |

Gates: |S/r − 1| < 5e-3, ratio < 3e-3, Z within 5e-4 of Z(PHI), yHits 0.
Bit identity: every existing mode (premixed 1D/3D, Sutherland, bunsen 2D,
round 3D, diffusion, candle, chamber, wick) byte-identical to HEAD
(`runs/s1/bitid.sh`); the law is its own kernel and functor.

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
`dns/config/flame_modes.cpp` grid), `foamflame/observejet.py` (`measureJet` verbatim + the
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

### G4a / G4b / G5: fast jets (2026-10-30)

G2a and G3a at higher jet speed and jet Mach (the model has no absolute
pressure: p0 only sets the SI map, so "high pressure" = high jet speed).
Same meshes per case, same protocol; dns twins run from the case dict
(`twins` / `extra_dns`, `foam_flame.py run CASE --dns`). simjob 728–735.

**Bunsen** (G2a burner, M 0.02, 8 cells; window [20, 25], foam tt 25, dns
tt 30): G4a U_jet 10 (jet Mach 0.2, 696 × 288, foam dt 4e-3 τ), G4b U_jet 15
(jet Mach 0.3 = the guard, 1000 × 288, foam dt 3e-3 τ). Ideal cone 59.7 /
89.8 δ_th = 95.6 / 143.8 δ_D.

| | dns default | dns `--nscbc-tinflow` | tinflow + etaIn 1000 | foam mclin |
|---|---|---|---|---|
| G4a xTip (δ_D) | 86.64 | 92.23 | 92.45 | 90.31 (92.40 / 88.23) |
| G4a sinRatio | 1.1049 | 1.0664 | 1.0684 | 1.0780 (1.108 / 1.048) |
| G4a SLflank | 1.0422 | 1.0149 | 1.0093 | 1.0017 |
| G4a burnRatio | 0.9955 | 1.0061 | 1.0106 | 0.9951 |
| G4b xTip (δ_D) | 125.91 | 143.48 | 145.52 | 142.67 (148.45 / 136.90) |
| G4b sinRatio | 1.1317 | 1.0153 | 0.9962 | 0.9803 |
| G4b SLflank | 1.1016 | 1.0450 | 1.0381 | 1.0440 |
| G4b Tmax/Tu | 31.32 | 30.24 | 30.15 | 30.14 |

- **T-held dns vs foam: tip +2.1% (U 10), +0.6% (U 15)**, flank angle and
  SLflank inside the window halves. NOT steady: the tip halves differ 2–5%
  (U 10) and up to 8% (foam U 15); flank rms 0.07 / 0.28 (G2a 0.03). The
  tip SHEDS fresh-gas pockets (videos at 0.09 τ, `foam_video.py render
  G4b|G4c`): it grows, pinches off a pocket and drops ~15 δ_D, a sawtooth
  of period ~3–5 τ, dns and foam in phase (G4c too). Equal "within the
  shedding" only; a mean needs several periods. A deterministic cycle, not
  amplified noise: the foam video rerun differs from the committed run by
  a FLAT 1–5e-7 (G4b) / 0.5–9e-8 (G4c) of field max over all 25 dumps
  (`foam_video.py check`), no growth across ~6 pinch-offs.
- **The default (density-held) inflow's error grows with the jet speed**:
  tip −2.0% (G2a U 5, vs the T-held twin), −6.1% (U 10), −12.2% (U 15);
  sinRatio +1.8% / +3.6% / +11.5%; the jet enters hot (U 15 Tmax 31.3 vs
  30.2 Tu). issues.md §66.
- foam's inflow pressure rises with the speed: +2.1% / +0.79% (jet /
  coflow) at U 10, +5.4% / +5.2% at U 15 (G2a +0.56%). foam burn/fuelIn at
  U 15 1.054: its fuel inventory not steady over the window.
- Cost: dns 0.47 / 0.58 ms/step (75 s / 100 s for the three runs); foam 453
  s / 922 s on 12 ranks.

**Diffusion jet** G5 = G3a at jet Mach 0.2 (G3a 0.05; the dns cap), 4
cells/l. The box's acoustic period 2L/c is ~42 l/U (G3a ~10): at tt 80
neither solver was steady (dns pMean halves 0.93 / 1.09 over [60, 75]; foam,
waveTransmissive lInf = box, pressurized to p/p0 2.03 at t 20, 1.57 at 75,
fuel leaving; `runs/flames/foam/G5_tt80`). Rerun: tt 240, window [200,
240], foam lInf = box/4 (`lInf_frac`).

| [200, 240] | dns (etaIn 100) | dns etaIn 1000 | foam mclin |
|---|---|---|---|
| tip (l) | 52.814 (52.91 / 52.72) | 47.785 | 50.626 (51.22 / 50.00) |
| Z_st half-width | 7.8135 | 7.6525 | 7.7175 |
| fuelIn | 9.8257 | 9.0334 | 9.4785 (face 9.083 + Fick 0.425) |
| tip / fuelIn | 5.3751 | 5.2898 | 5.3412 |
| burnRatio | 1.0030 | 1.0027 | 1.0027 |
| pMean | 1.0226 | 1.0180 | 1.0068 |
| foam inflow p/p0 − 1 (jet / coflow) | — | — | +0.073 / +0.047 |
| Tst / T_ad | 0.9664 | 0.9643 | 0.9406 |
| H rows i = 0 | 0.59 | 0.26 | 6.5e-4 |

Fields at t 240 vs foam: dns Z L∞ 1.34e-2 (L2 8.6e-3), θ L∞ 4.5e-2 (L2
3.2e-2); etaIn 1000 Z L∞ 1.46e-2 (L2 6.5e-3).

- **The tip per unit fuel agrees: dns +0.6% (default), −1.0% (etaIn
  1000)**; the tips differ by the fuel flux (+4.3% / −5.6%). Unlike G3a
  (foam = etaIn 1000 to 0.7%), foam's fuelIn sits between the two: its
  inlet holds T with zeroGradient p, and the box raises the inflow p 7.3%
  (∝ γM², 16× G3a's) → ρ_in and the fuel flux rise with it; dns' row holds
  ρ (the entropy target). Different inflow problems at M 0.2, issues.md §71.
- foam is still drifting (tip halves −2.4%); dns flat to 0.4%.
- Cost: dns 0.375 ms/step (93 s, two runs); foam 9600 steps, 354 s on 12
  ranks.
- G5 `--nscbc-tinflow` twin (job 740): tip 58.33 (+15.2% vs foam), fuelIn
  10.70 (+12.9%), tip/fuelIn 5.452 (+2.1%), pMean 1.022, Tst/T_ad 0.939;
  Z L∞ 3.95e-2. Holding T does not give foam's inflow (issues.md §71).

### G4c / G6: thin, fast nozzles (2026-10-31)

simjob 736–739. **G4c** = slot 6 δ_th at U_jet 20 (the `--jet-speed` cap),
M 0.015 (jet Mach 0.3, as G4b), 696 × 144, foam dt 2.25e-3 τ; ideal cone
59.9 δ_th = 95.9 δ_D, flow-through 3 τ. Window [20, 25].

| G4c | dns default | dns tinflow | tinflow + etaIn 1000 | foam mclin |
|---|---|---|---|---|
| xTip (δ_D) | 79.03 | 86.30 | 87.10 | 86.55 (86.32 / 86.77) |
| sinRatio | 1.1300 | 1.0556 | 1.0551 | 1.0274 |
| SLflank | 1.0629 | 1.0201 | 1.0068 | 1.0168 |
| burnRatio | 0.9907 | 1.0039 | 1.0038 | 1.0051 |
| Tmax/Tu | 16.34 | 15.99 | 16.01 | 15.96 |

- **T-held dns vs foam: tip −0.3% (+0.6% with etaIn 1000)**; foam's tip is
  steadier than G4b's (halves ±0.3%). sinRatio +2.7% over foam.
- Default inflow −8.4% vs T-held (G4b −12.2% at the same jet Mach): §66
  is not set by U or jet Mach alone.
- Cost: dns 0.38 ms/step; foam 333 s on 12 ranks.

**G6** = G5 at slot 8 l (jet Mach 0.2), 152 × 160 cells, tt 120, window
[80, 120], foam lInf = box/4, np 4.

| G6 [80, 120] | dns (etaIn 100) | dns etaIn 1000 | dns tinflow | foam mclin |
|---|---|---|---|---|
| tip (l) | 8.227 | 7.847 | 8.541 | 7.898 (7.88 / 7.92) |
| fuelIn | 2.691 | 2.591 | 2.787 | 2.638 |
| tip / fuelIn | 3.057 | 3.028 | 3.065 | 2.994 |
| pMean | 1.0078 | 1.0073 | 1.0045 | 1.0114 |
| Tst / T_ad | 0.909 | 0.912 | 0.900 | 0.908 |

- Tips vs foam +4.2% / −0.6% / +8.1%; per fuel +2.1% / +1.1% / +2.4%. As
  G3a, foam = dns etaIn 1000 (tip 0.6%, fuel 1.8%); the short box keeps the
  inflow pressure low (foam pIn +8.3% jet but pMean 1.011).
- Fields at t 120 vs foam: Z L∞ 1.4e-2 / 8.6e-3 / 2.5e-2.
- Cost: dns 0.26 ms/step; foam 71 s on 4 ranks.

### P0: the 0D reactor, FV and chemFoam against one quadrature (masterplan_combnozzle, 2026-11-09)

Chemistry alone. Model C0 (σ 5, β 8, Λ 3656401.0984807) in a closed adiabatic box at rest, ρ fixed,
Y 1, T0 = 2.5 Tu, M 0.01 units. ω/ρ = Λ Y e^(−θ Tu/T): ρ cancels, so

    dY/dt = −Λ Y e^(−θ Tu/T),   T(Y) = T0 + ΔT (1 − Y)
    ΔT = q/cv = γ (σ − 1) Tu   (constant volume: the FV box; T_end 8.1 Tu)
    ΔT = q/cp = (σ − 1) Tu     (constant pressure: the control oracle, chemFoam's `pressure`)
    t(Y) = ∫_{ln Y}^0 ds e^(θ Tu/T(e^s)) / Λ

Oracle: `tests/fvref_reactor.hpp` (adaptive G7K15 in s = ln Y, relative 1e-14 per panel; Y(t) by
bisection in s) and its Python twin in `tools/chem/chemfoam_reactor.py` (Gauss-Legendre 12 on
panels of 0.02 in s, Newton in the panel). t(1/2) = 3.4165470720791 (cv), 4.8439408999827
(cp): both match 40-digit mpmath to 7e-16 and 1.3e-15 (`chemfoam_reactor.py rk3ref` prints every
reference). FV case: the uniform periodic 4 × 4 × 1 FaceMesh, h = 1 (integer coordinates, so a
uniform state's face sums cancel exactly). Tests: `fvref_reactor` (host), `gpu_fv_flame_reactor`.

| gate | relation | measured | |
|---|---|---|---|
| G0.1 | 2000 steps of t_e/2000: every cell bitwise cell 0's, u v w 0.0 | FP32, FP32 compensated, FP64: exact every 50 steps | PASS |
| G0.2 | N 2000/4000/8000 (dt t_e/N): order ≥ 2.8, finest \|ΔY\| ≤ 1e-9 | orders 2.203, 2.583 (Y and T); \|ΔY\| 9.86e-3, 2.14e-3, 3.57e-4 | **UNMET** |
| G0.2 control | the cp oracle must miss | 0.451 | fails as required |
| G0.2b | N 16000/32000/64000: order ≥ 2.8; \|ΔY\| ≤ 1e-9 | 2.904, 2.961; 8.82e-7 | order PASS, 1e-9 **UNMET** |
| G0.2c | every rung vs the exact-arithmetic SSP-RK3 recurrence ≤ 1e-8 | 1.2e-10 (N 2000) … 4.85e-9 (N 64000); FvRef 6.7e-11 at 2000 | PASS |
| G0.2c control | Kutta's RK3 (also order 3) at N 2000 must miss | 6.41e-3 | fails as required |
| G0.3 | burn to 2 t_e at t_e/8000: Σ V (E + q ρY) per step ≤ 1e-14; T_end 1e-12 | per step 1.01e-15; T 3.19e-3 off, Y −4.617e-3 | energy PASS, T **UNMET** |
| G0.3b | the same at t_e/64000 | per step 1.01e-15; T 2.76e-13, Y 4.9e-324 | PASS |
| G0.4 | FvRef vs device, 100 steps from the IC ≤ 1e-13 | 2.60e-15 | PASS |
| G0.4 | across the runaway (steps 1900..2000), each step from the device state | 2.68e-16 (continuous: 5.90e-13; a 1e-14 ρY perturbation grows ×12.3 there) | PASS |
| G0.5 | chemFoam `volume` vs the cv quadrature ≤ 1e-5 every step | 0.495 (t_e/2000), 0.391 (t_e/8000) | **UNMET** |
| G0.5b | chemFoam `pressure` vs the cp quadrature ≤ 1e-5 every step | 3.33e-10; T_end 6.5 Tu exact | PASS |
| G0.5 control | Hf_P = 1% of Hf_F must miss by > 1e-3 | cp 0.459, cv 0.578 | fails as required |

Why the three UNMET rows are the rigs, not the solver:
- **G0.2**: Y(t_e) is in the thermal runaway (J dt 1.96 at t_e/2000, positive eigenvalue). The exact
  SSP-RK3 recurrence itself has orders 2.20, 2.58 there and 3.57e-4 at t_e/8000; it reaches 2.9
  only at N ≥ 16000, and 1e-9 needs N ~ 6e5, where the device's rounding gap (growing with N,
  4.85e-9 at 64000) is already larger. G0.2c is the sharp form: the device IS that recurrence.
- **G0.3**: t_e/8000 is 6.6× the solver's own chemistry bound `chemDt` (`kChemDt` 0.5) at T_end
  (J dt 3.3 > RK3's 2.51). The floored RK3 overshoots into Y −4.617e-3 (a scalar FP64 twin gives
  −4.61669e-3) and then never reacts. Inside the bound (G0.3b) the burn ends at T_end to 2.8e-13.
- **G0.5**: `StandardChemistryModel::derivatives` integrates T at CONSTANT PRESSURE inside every
  chemistry step (dT = −Σ h_a dc/(ρ c_p)); `constantProperty volume` then resets h = u0 + p_old/ρ +
  ∫Q̇. So chemFoam's constant-volume reactor is an O(dt) split: ignition-delay error 2.53e-3,
  7.06e-4 at t_e/2000, /8000 (order 0.92), max T error at the front 0.495, 0.391. Its
  constant-pressure reactor has no split and matches the cp quadrature to 3.3e-10 at every step:
  the janaf map, A, Ta and Hf are exact.

chemFoam rig (`tools/chem/chemfoam_reactor.py setup|run|compare [--ode S]`, cases in
`runs/flames/foam/P0/<S>/`, one CPU job):
- `readInitialConditions.H` casts the thermo to `reactingMixture<gasHThermoPhysics>`
  (sutherland + janaf): **hConst is refused**. Constant cp = janaf with a0 = 3.5 in both ranges,
  a1..a4 = a6 = 0, the offset a5 carrying the heat: h_a,F − h_a,P = R (a5_F − a5_P) = cp (σ − 1)
  Tu = q, independent of R and Tstd.
- **seulex aborts on the runaway** (all five cases, `Integration steps greater than maximum
  10000`, dxDid ~4e-12 s): `StandardChemistryModel::jacobian` leaves the T row zero (no dT'/dc,
  dT'/dT), and a linearly implicit extrapolation without the runaway's eigenvalue collapses its
  step. The gated runs use RKDP45 (explicit, absTol 1e-14, relTol 1e-10): the reaction is not
  stiff here (J ≤ 7624/τ).

### P1: the 1D flame on ONE polyMesh, dns FV (FP64) and reactingFoam (masterplan_combnozzle, 2026-11-09)

G1's box (12 + 10 δ_th + 2 × 16 cells, h = δ_th/N, u_in 0.98, σ 5, β 8, Pr 0.72, M 0.01) as ONE
`tools/chem/fv_flame_mesh.py` polyMesh per rung (ny 4, y cyclic, z empty, depth 1 δ_D:
`runs/flames/foam/poly1d_mesh/c{7.5,15,30}`). dns reads it (`--polymesh --bc`, `--fv-precision
fp64`, `--upstream 12 + 16/N`); reactingFoam reads the same faces/owner/neighbour, points and
cyclic separation × δ_D (checkMesh OK, no blockMesh), IC = the oracle at foam's own
`writeCellCentres`, BCs `foam_bcs_1d` unchanged (fixedValue U, T, Y in; waveTransmissive p out),
G1's `mclin` schemes, dt 1e-3 τ, 3 × 2 correctors. Window [7, 14] τ for both. Probes = the dns
`FvFlameProbe` restated (`observe.load_foam_poly`): slab = floor(C.x/h), reference slab
round(2 δ_th/h) from the low end, S_L = Σ V ω/(A ρY_ref), A = ly × depth. Tooling:
`foam_flame.py setup|run|compare fv1d|fv1d_stockle|fv1d_g12` (`run fv1d --foam --also
fv1d_stockle fv1d_g12`: the six foam runs in one CPU job), `foam_flame.py gates`
(`foamflame/p1gates.py`, `runs/flames/foam/fv1d/gates.txt`). Fresh-state sensitivity
`Oracle.sens()` (A, Ta, q, cp, λ fixed, ±1e-4 central): a_T = dln S/dln T_u = 2.33532, a_p =
−0.50000.

| gate | relation | measured | |
|---|---|---|---|
| G1.0 | foam on poly1d vs foam G1 blockMesh, 15 cells, G1's column, [7, 14] ≤ 2e-5 | 1.00301311 vs 1.00301311: −1.4e-9 | PASS |
| G1.0 control | G1's IC laid j fastest on the poly cells must miss | −1.003 (no planar flame) | fails as required |
| G1.1 | dns S_L − (1 − γM²(σ−1)) order ≥ 1.8; docs/fvcomp.md's box rows within the spans | +1.743e-2, +4.032e-3, +9.966e-4 (2.112, 2.017); vs the box +1.0e-8, +4.3e-7, −4.1e-7 | PASS |
| G1.2 | \|S_dns − S_foam\| ≤ max(own error vs S_ref) each rung, falling | +1.384e-3, +4.592e-4, +2.341e-4 vs 1.743e-2, 4.032e-3, 9.966e-4 | PASS |
| G1.2 control | foam stock Le 1.39 (ρD = μ) gap ≥ 0.1 | +0.1342 | fails as required |
| G1.3 | L∞ θ, Y at x_f, dns vs foam ≤ max own oracle L∞ | θ 1.79e-3 / 3.21e-4 / 1.10e-4 vs 1.04e-2 / 2.15e-3 / 3.63e-4; Y 1.95e-3 / 3.77e-4 / 1.35e-4 vs 5.71e-3 / 1.16e-3 / 2.66e-4 | PASS |
| G1.4 | δS_fresh = a_T δT_u/T_u + a_p δp_u/p_u ≤ band/4 | +2.8e-5, +1.7e-5, +1.7e-5 vs 4.4e-3, 1.0e-3, 2.5e-4: raw S_L compared | PASS |
| G1.5 | dns at M_n 0.042: self-convergence order ≥ 1.8 | 1.007853, 0.994837, 0.991887: ratio 4.41, order 2.141; Richardson 0.991021, +9.0e-4 off 1 − γM²(σ−1) (reported) | PASS |
| G1.6 | γ 1.2: order ≥ 1.8 about 1 − 1.2 M²(σ−1); dns − foam at 15 ≤ max own error | +1.708e-2, +4.057e-3, +1.114e-3 (2.074, 1.865); +2.53e-4 vs 4.06e-3 | PASS |
| G1.7 | D = [T_b/(σT_u) − 1]_1.2 − [·]_1.4, T_b over [x_f + 3, x_f + 5] δ_th: 4.8e-5 ± 1.2e-5 | +4.809e-5 (15), +4.751e-5 (30) | PASS |
| G1.7 control | q on cp(1.4) at γ 1.2 (flag form: σ' = 1 + 3.5·4/6, the same q/cp) | D −0.3333 | fails as required |

Readings:
- **Same mesh, both readers**: foam's 4-row cyclic poly1d run IS its 1-row blockMesh run (1.4e-9
  in the window mean), so every dns − foam number below is the solvers', not the meshes'.
- **dns − foam shrinks** (1.38e-3, 4.6e-4, 2.3e-4: ratios 3.0, 2.0, no order claimed) and sits 13×, 9×, 4× below the
  oracle error; the fresh states differ by 1.7e-5 in S_L (Far vs fixedValue: T_u 7142.91 vs
  7142.856, p_u equal to 2e-6). foam's converged S_L (G1) 0.99939, dns' Richardson 0.99955.
- **γ enters only Tu, q, the O(M²) offset and the burnt T offset**: γ 1.2 S_L errors equal γ 1.4's
  to 2% at 15 cells; the burnt T offset halves (−3.9e-5 / −4.5e-5 vs −8.7e-5 / −9.3e-5).
- **M_n 0.042 (P4's flame)**: S_L 0.994837 at 15 cells; the Richardson value sits 9.0e-4 above
  the O(M²) formula (formula error ~1e-3 at M 0.05, docs/fvcomp.md). T_b/(σTu) − 1 −1.67e-3 vs
  −1.69e-3 [D].
- Cost: dns 0.211 ms/step (FP64, 1448–5792 cells), 16.6 / 33.2 / 66.1 s at M 0.01, 4.3 / 9.8 /
  38.7 s at M_n; foam 55 / 104 / 196 s exec (1 core, 14 000 steps), the six side by side in a
  204 s CPU job.

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
`flame1d` (`addBunsenTurbulence`), `dns/drive/flame.cpp` (rows, the J-weighted box),
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

## Round-nozzle Bunsen (`--mode bunsen --nozzle round`)

`--nozzle round` (3D only, `--nz` = ny): a circular nozzle of DIAMETER
`--slot-width` on the axis (yc, zc) of a square periodic y-z box; the slot's
inflow profile, pilot and IC cone are evaluated at r = hypot(y − yc, z − zc)
(`bunsenInflowAt`, `bunsenConeDistanceAt`; the slot entry points are the old
functions bit for bit, `test_bunsen`). The cone is measured on the AXIAL
plane z = zc (mean of the planes k = nz/2 − 1, nz/2, FP64), not the span
mean (an axisymmetric cone's span mean is not a cone); ST = burn / (π R
hypot(H, R)), R = D/2; `SLflank` is NaN (per unit span). The periodic box
holds a lattice of jets one box apart (jet interaction not measured).
Files: `bunsen`, `config` (`bunsenGeometry`), `flame1d` (`bunsenRoundIC`),
`dns/drive/flame.cpp` (rows), `runner_flame` (`axialPlane`).

The round cone is SHORTER than the ideal cone (measured, laminar DNS):
U 10, D 6 δ_th, 4 cells/δ_th, cold
coflow + 3 δ_th pilot at the jet speed, nz 96 (1.84 M cells, simjob 725, 82
s): steady, burn/in 1.00, tip 17 δ_th = 2.8 D vs the ideal (D/2) √(U² − 1)
= 30, ST 1.63. U 5, D 12 (nz 120): tip 21.5 vs 29.4, ST 1.35. Not
separated: curvature vs resolution at the tip.

Renders: `tools/chem/flame_room.py` (inside Blender: `blender -b
--factory-startup --python tools/chem/flame_room.py -- PREFIX --out DIR
[...]`, `--help`): Cycles path tracing of the `--emit volume` ω field as an
emission-only VDB in a small lab (bench, burner `--burner slot|round`,
props), proxy lights per height band, AgX; ~4–5.5 s/frame at 1080p on
gfx1100. A low-Mach `--equivalence` run renders two cones (premixed /
diffusion split by the flame index) at real scale from its .json: "Low-Mach
dt rule, real-burner scouts, renderer" below. `tools/chem/flame_dnsvid.py`:
2×2 slice video (∫ω dz light, |∇×u|, |u|, T).

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

## Checkpoint and restart (`--checkpoint-dt`, `--wall-limit`, `--restart`)

Masterplan real_bunsen S2: a run longer than one 1800 s simjob. Wired in
`runFlame` (premixed 1D/3D incl. `--equivalence`, `--tfles`, `--les`,
`--turb --hold-flame`; bunsen slot/round 2D/3D; diffusion; candle); `--wick`
and `--mode chamber` are rejected by name. **No checkpoint flag = the runner
bit for bit** (`runs/s1/bitid.sh`, 9 flame cases + flow compressible /
conv / channel byte-identical to HEAD, 2026-10-31).

- **Byte rule**: a run stopped at a checkpoint and resumed equals the
  uninterrupted run byte for byte in every output but the `.json` (stats,
  profiles, `bin`/`volume`/`slices` frames, PNG). `gpu_restart_flame` is the
  proof, over every mode above (1 + 3 restarts each).
- **Where**: at the END of the first step whose t is at or past k
  `--checkpoint-dt` (after that step's stats row, profile and frame), never at
  `--tt` (the last step is clipped and forces a stats row: masterplan F4). A
  resumed job starts a fresh step. A stop (wall limit, SIGINT/SIGTERM with
  `--checkpoint-dt`) happens only there, so no output ever holds a partial
  step or a forced end row.
- **`--wall-limit SEC`**: counted from the process start; at a checkpoint the
  job stops when the elapsed wall time is ≥ SEC or the last checkpoint
  interval's wall time added to it would pass SEC (the next checkpoint would
  end past it). Exit 0, stdout `checkpoint: t=... steps=... (incomplete;
  resume with --restart PREFIX)`, `.json` `"complete": false`. `--wall-limit
  0` stops at the first checkpoint (the tests' chain).
- **`--restart PREFIX`** (must equal `--out`): phase 1.5 (the `--scene`
  precedent) reads the checkpoint HEADER and rejects, exit 3 by name: another
  solver, another dns binary (sha256 of `/proc/self/exe`), any case-defining
  flag (`--restart: case differs from the checkpoint: --jet-speed 4 vs 5`).
  The case signature is every GIVEN flag but `--tt`, `--checkpoint-dt`,
  `--wall-limit`, `--restart`, `--out`, `--quiet`, `--dry-run`, `--backend`,
  at its final parsed value in one spelling (shortest round-trip number, enum
  name, `--emit` as its set), sorted: order and spelling free; `--emit`, the
  frame cadences, `--equivalence`, `--sl-law`, `--mix-phi` are in it. `--tt`
  must pass the checkpoint's t (exit 2); it may grow. Phase 3 reads the whole
  checkpoint (exit 3), truncates every series and frame file to its
  checkpointed length (a job killed after its checkpoint left rows; a file
  SHORTER than recorded is exit 3) and appends; the frame counters and
  cadences continue. `.json` gains `"complete"` and `"restarts"` (only on a
  checkpointed or restarted run).
- `--emit web`/`vtk` are rejected with these flags (the web manifest is
  rebuilt per process); `shm` is live only (a new segment per job).
- **Chaining**: `tools/chem/chain.sh [--resume] PREFIX N -- flame ARGS` runs
  up to N simjobs (`--wait`, `--wall-limit $CHAIN_WALL` (1700), `--restart`
  from the second), stopping on rc ≠ 0 or `"complete": true`. simjob's
  timeout sends SIGTERM first: with `--checkpoint-dt` that writes a
  checkpoint at the next step boundary (simjob's grace before SIGKILL is
  10 s: one step + the write must fit), and `--resume` continues from it
  (the job itself reads rc 124).

**Format** (`dns/checkpoint.{hpp,cpp}`; `<prefix>_ckpt.bin`, written to
`.tmp`, fsync, rename, directory fsync: a job killed mid-write keeps the
previous one). Little-endian: `"DNSCKPT\0"`, u32 version 1, u32 record count,
records (u32 name length, name, u32 type, u64 element count, payload: text
bytes | f64 | i64 | f32 raw bit patterns), `"ENDCKPT\0"` (a file cut short
is rejected). Records are NAMED and order-free; a missing one is an error by
name. Standard records: `binary`, `solver` (`flame-compressible`), `case`
(the signature text), `grid` (nx, ny, nz, nY), `t`, `steps`, `restarts`;
`loop.*` (the runner's loop-state struct, `FlameLoop::fields()`: t, steps,
the stats/progress cadence, the profile schedule, dt state, x_f history, the
`--turb` sweep and the `--hold-flame` controller); `out.*` (`OutputResume`:
`out.file:<suffix>` byte lengths, frame count, PNG index, next frame/PNG/web
times); `state.*` (raw FP32 device fields: rho, mx, my, mz, E, y0.. (both
species of `--equivalence`), t0/1 and lo*/tlo* when a state carries them).
**Extended by the low-Mach runner (S4)**: solver `flame-lowmach` (phase 1.5
picks the expected name from `--physics`; a checkpoint of the other solver
is exit 3 naming both), its own loop struct `LowMachLoop`, and `state.rho`,
`state.y0..`, `state.mx` (nx + 1 faces per row), `state.my`, `state.mz`,
`state.p`, the compensated low words `state.rlo`, `state.ylo0..` and the
FP64 mass tally `state.tally`; no version bump (no warm start: the solve is
direct). Every loop variable must live in the loop struct (K12): one outside
it breaks the byte rule silently, and only the chained byte test catches it.

**Extended to the FV routes (masterplan_multiphysics M5)**: `--solver fv` and
`--polymesh DIR --bc FILE`, both physics, every mode the FV path builds
(premixed C0 / C2 / C2b, `--les`, `--tfles`, `--compensated`, `--fv-precision
fp64`; the diffusion jet; the low-Mach planar flame and burner with `--laterals
open`, `--rim`, `--gravity`). Solvers `flame-fv-compressible` (`runFvComp`) and
`flame-fv-lowmach` (`runFvLowMach`); phase 1.5 picks the expected name from
`--physics` AND the route, so a structured checkpoint into an FV run (or back)
is exit 3 naming both. Their `grid` record is the FaceMesh's (nC, nI, nB, nY),
compared with the mesh built (exit 3). **The mesh is part of the identity**: a
`mesh` record, `polyMesh sha256:<hex> bc sha256:<hex>` (every regular file of
the polyMesh directory, name + sha256 in name order, and the `--bc` file's
content) or `flags` (the box the case flags define); phase 1.5 compares it by
text: a changed mesh file or `--bc` file is exit 3 (`--restart: ...: the mesh
differs from the checkpoint's: ...`) even under the same path. Loop structs
`FvCompLoop`, `FvLowMachLoop`. State records are the device words raw, F32 at
FP32 and F64 at `--fv-precision fp64` (`gpu/internal/ckpt_words.hpp`):
`state.fv.q` (+ `state.fv.lo`, the compensated low words; `state.fv.hits`, the
SST guard's count, when allocated) for the compressible solver; for the
low-Mach one the state `state.fvlm.rho`, `rlo`, `ru`, `y<s>`, `ylo<s>`, the
projection's WARM START and memory `state.fvlm.p` (the PCG's x0 of the next
stage) and `phi`, `phiB` (the last stage's face mass flux: the next stage's
Rhie-Chow term of c), the FP64 tallies `state.fvlm.tally` and the uploaded
`state.fvlm.enthalpy0`. Not carried, by construction: LES mu_t and the TFLES
F E, E/F, Omega (of each step's input state; `refreshEddy` for dt and stats),
the burner's per-face rows (tables of the mesh), the AMG hierarchy
(geometric, built once). `gpu_restart_fv` (`comp`) and `gpu_restart_fvlm`
(`lm`) prove it: chained (1 + 3 restarts) = straight byte for byte, a
checkpointing straight run = straight, and its last checkpoint = the chain's
record for record but `restarts`. The low-Mach FV loop's equal steps to `--tt`
(`rem / ceil(rem / dt)`) make its trajectory a function of `--tt`: a restart
that GROWS `--tt` is a valid continuation but not the straight run to the
larger `--tt` (the structured runners' "grow" equality does not hold there).

## Low-Mach (`--physics lowmach`; masterplan real_bunsen S4)

The zero-Mach, variable-density, acoustically filtered flame solver: the
M → 0 limit the compressible solver approaches only through an inflated Mach
(issues §31: at the true CH4 M 1.15e-3 its FP32 pressure cannot resolve the
flame's jump). S4 builds the premixed 1D/2D/3D flame, model C0 and the
`--equivalence` jet (C2 / C2b, `--mix-phi`); bunsen, open laterals, gravity
and the rim come in S5–S8 (S5–S7 built: the sections below) and are
rejected by name until then. Files: `dns/lowmach.
{hpp,cpp}` (FFT-size rule, defaults, `lowmachIC`), `gpu/lowmach.{hpp,hip}`
(`DeviceLowMach`), `gpu/runner_lowmach.{hpp,hip}` (`runLowMach`),
`gpu/internal/species_face.hpp` (the D7 move: Face, mcSlope, the shared
limiter, `faceValue2`/`faceY2`, read by both solvers; the compressible
kernels byte-identical, `runs/s4/bitid.sh`).

**Model.** Calorically perfect, equal W, Le = 1, P0 constant, ρT = P0/R. Box
units ρ_u = S_L = R = 1, δ_D = 1, **Tu = P0 = 1** (no acoustic scale:
`--mach` rejected), cp = 3.5, μ = Pr at Tu (Sutherland through `Transport`),
λ/cp = ρD = μ/Pr. State: ρ, ρY_k (cells, **compensated**: hi + Kahan lo
word), the MAC mass fluxes m (x faces 0..nx per row: face 0 the inflow,
face nx the outflow; y/z faces periodic), the dynamic pressure p (cells).
T = P0/(Rρ), Y = ρY/ρ derived.

**Step** (SSP-RK3, the compressible StageCombine weights, every write in the
increment form a + (cb (b − a) + cr r)): per stage from (a, b):
1. cell pass on b (`kLmCellPass<TwoR, Phi>`): T, Y, ω (the shipped
   `arrheniusRate`/`arrheniusRate2`/`phiRateLn` templates), μ(T), div u;
2. temperature predictor (`kLmTemperature`): T* = T_a + cb (T_b − T_a) + cr
   R_T, R_T = [−Σ_f m_f (T_f − T_c) + div(λ/cp ∇T) + (q/cp) ω]/ρ_b, T_f
   the species' SHARED limiter's face value (T does not vote, so H = T +
   (q/cp) Y_F rides one linear face formula); the divergence target
   D = ρ0 R_T/T* + P0 ca cb (ρ_a − ρ_b)²/(ρ_a ρ_b cr T*) — the exact
   rewrite of (ρ0 − P0/T*)/cr without its cancellation (the difference form
   fed FP32 rounding/cr into the projection: p rows swung ±100 at 15
   cells, ±1.8e5 on the clipped last step);
3. momentum predictor (`kLmMomX`, `kLmMomT<A>`): central conservative
   mass-flux × mean-velocity convection (kinetic-energy consistent with the
   discrete continuity), full stress with −2/3 μ div u (cell μ for normal,
   the 4-cell mean at edges), −∇p_b; no added dissipation;
4. projection on the MASS FLUX: L φ = (div m* − D)/cr through `LmPressure`
   (one direct `SeparablePoisson` pass: x ND = inflow Neumann / outflow
   p = 0, y/z periodic), m −= cr G φ (S3's face rules), p = p_b + φ;
5. continuity and species on the projected m (`kLmScalars`, compensated
   writes through stage_write.hpp's `combineWord2`).

**Boundaries.** x-lo inflow FACE: m = u_in (ρ_u = 1), T = Tu, Y = the
premix, convective AND diffusive face flux (reactingFoam's inlet; no LODI
row, so issues §45/§66/§68 do not exist), tangential u = 0 (antimirror). x-hi
outflow: p = φ = 0 on the face, m from the projection, scalars zero-gradient
BOTH ways (backflow carries the cell's own value; with `--laterals open`
(S6) the backflow carries the AMBIENT, below), no diffusion, zero
normal-stress gradient. y, z periodic (the burner: or open, S6). **FFT sizes** (K11): ny, nz 7-smooth (error naming the nearest),
nx rounded up so nx/2 is (the x DCT-IV's complex length; the cells go
downstream, the dry-run says how many).

**dt** = min(convective cfl h/Σ|u| (`--cfl`, default 0.8), diffusive 0.25
h² ρ_min/(μ(T_max) max(4/3, 1/Pr)), chemistry `--chem-dt` K / max J (the
cp-form Jacobian, default K = 1)); the last two steps share the remainder
(no sliver step: the projection reads m*'s rounding over cr). The burner
(S9) adds the per-cell positivity bound by default (`--dt-rule cell`:
"Low-Mach dt rule, real-burner scouts, renderer" below).

**Why compensated** (measured 2026-10-31, before the low words): at 7.5 /
15 / 30 cells the preheat zone's per-step increments fell below half an ulp
and were dropped, so the state LAGGED: the Shvab–Zeldovich defect grew with
the step count (4.8e-5 / 3.3e-4 / 7.7e-4; 1.1e-3 at 5× smaller dt), mass
drifted from the tally (3.3e-5 / 1.0e-4 / 4.6e-4) and S_L carried it (−1.2e-3
grid-independent, −7e-4 more at 5× smaller dt). With the low words: SZ
7–14e-7, S_L dt-independent to 1e-6 (held, sliding and 5× smaller dt all
0.998274 at 15 cells), Z exact to 2e-7.

**Measured** (gfx1100, simjob, 2026-10-31; σ 5 β 8 C0, `--inflow-speed
0.98 --tt 20`, window [10, 20]; `gpu_lowmach_1d`):

| cells/δ_th | S_L | halves | SZ defect | p jump / (ρ_u S_d² (σ−1)) − 1 | p row rms / jump |
|---|---|---|---|---|---|
| 7.5 | 0.994280 | 0.994249 / 0.994310 | 1.4e-6 | −1.5e-3 | 0.17% |
| 15 | 0.998274 | 0.998274 / 0.998274 | 8.6e-7 | −1.1e-3 | 1.0% |
| 30 | 0.999537 | 0.999537 / 0.999537 | 7.2e-7 | (noise) | 6.5% |

Observed order 1.66 (the plan's 1.8 is not met: the MC limiter also sets T's
face value, which the heat release reads); Richardson limit **1.000121**.
The compressible ladder (M 0.01: 1.016856 / 1.003021 / 1.000336, order 2.37)
extrapolates to 0.999689, +3.9e-4 for M → 0: **1.000079 — the two solvers'
grid-converged flame speeds agree to 4e-5**. At 15 cells their truncation
errors have opposite signs (low-Mach −0.17%, compressible +0.34%): the plan's
"within 1.5e-3 at 15 cells" is not met and is the wrong comparison (MAC /
T-equation / MC-limited T vs KG / energy / central E). The compressible
solver at the TRUE M 1.15e-3 (15 cells, tt 10, 480354 steps, 95 s): S 1.00111
over [5, 10] with halves 1.00132 / 1.00091 (drift −4e-4 per 2.5 τ, FP32 p,
§31), T_b −5.3e-4 off σTu; low-Mach 0.998273, halves equal to 1e-6, ~18.5k
steps (5 s). Burnt gas T = σTu exactly (no issue-64 ramp), T_max/(σTu) − 1 =
0 (column means). Mass = the FP64 face-flux tally (CSV digits; the API test:
7.9e-12 2D, 4.0e-11 3D after 100 steps, the low words' own rounding). EOS
defect max |ρT* − 1| 1.0–1.4e-7; max |div m − D|/max |D| 7e-6 (2D), 1e-5
(3D).

**The pressure floor** (issues LM2): p is the projection's multiplier, and
the stage targets D carry the FP32 quantization of the T field read through
the diffusion term (k δT/h², δT ~ ulp/2); the multiplier reads it over cr.
The row-to-row rms of the jump scales as 1/dt (0.17% / 1.0% / 6.5% at 7.5 /
15 / 30 cells); it is white in time, not acoustic, and the window mean is
within 1.1e-3 of ρ_u S² (σ − 1). The plan's "rms ≤ 1e-6 of the jump" is not
an FP32 number. No effect on the flow found (1D: m is constraint-determined;
multi-D: the noise is a gradient, removed by the next projection).

**Model C2 / C2b** (15 cells, σ 6.88 Sutherland 298 K; `gpu_lowmach_rich`):
the jet 0.998470; `--mix-phi` 0.7 S/r − 1 = −1.7e-3, (S/S_jet)/r − 1 =
−1.9e-4; 1.2: −1.5e-3 / +1.4e-5 (r the oracle's S_L(φ)/S_L(jet)). The
mixture fraction stays within ±2.1e-7 of the inflow premix's (K15: the
compressible FP32 walk read ±1.7e-4 and reached the rate through C).

**Chemistry dt** (K1; `gpu_lowmach_chemdt`): at the M1 case (φ 1.3 C2b, σ
6.88, Sutherland, Pr 0.52, 4 cells/δ_th) K 0.5 binds 1.43× below diffusion
(dt 9.8e-3 vs 1.48e-2); K 1 and 2 leave diffusion binding and move S by
3.1e-5. The cp form (vs the compressible's cv) and the low-Mach dt being
diffusive, not acoustic, make explicit chemistry cheap: **S4b (point-
implicit chemistry) is not built**; the default K is 1. At that grid the
SZ defect is 2.5e-4 (K3: the O(dt²) predictor/projected-flux lag at dt
1.4e-2; 8.6e-7 at 15 cells, vs the compressible's 4.7e-3 there,
`gpu_lowmach_sz`).

**Cost** (3D planar, 8 cells/δ_th, ny = nz = 128: 210 × 128 × 128 low-Mach
(`--downstream 14`) vs 208 × 128 × 128 compressible at M 0.02): 9.89 vs
7.94 ms/step (2.87 vs 2.33 ns/cell-step: transport + 3 spectral solves), 526
vs 1536 steps per τ — **5.2 vs 12.2 s per τ**; at 64²: 2.54 vs 1.84 ms/step.
At the true Mach the compressible would take ~17× its M 0.02 step count.
3D planar = 2D (nz 16 vs nz 1, 8 cells: S, x_f, pJump identical).

**Determinism.** Own-index writes, no FP atomics, min/max rocPRIM reductions,
the mass tally a fixed-order single-block FP64 tree, the direct spectral
solve: reruns byte-identical (`gpu_lowmach_passive`); restarts byte-
identical (`gpu_lowmach_restart`: 1D + junk, `--equivalence`, wrinkled 2D,
3D volumes; S5: the 2D slot, the partially premixed slot, the round
nozzle).

**Changed in S5** (each measured on the burner; issues LM8, LM10):
(1) **T votes in the shared limiter.** A shared θ nobody votes for is 2 —
the DOWNWIND face value — harmless for a species below the vote threshold
but not for the T it carried: in the `--equivalence` jet core (Y_F 0.07,
ulp 7e-9 < 2^-24, nobody votes) an odd-even T wiggle grew from rounding to
T = 0.47 Tu and the run diverged at t 2.4. With T a voter H = T + (q/cp) Y_F
still rides one linear face formula (one θ), and T is TVD itself. The 1D
ladder is unchanged to the printed digits (0.994280 / 0.998274 /
0.999537); the rh window-mean p jump moved −1.1e-3 → −3.0e-3 (gate 4e-3:
the burnt gas' ulp noise now limits T at its extrema; LM2's floor).
(2) **Convective rate per cell over min(ρ_face, ρ_cell)** (a hot cell
beside a cold one exchanges m dt/(ρ_c h) of its own mass). (3) **3D
diffusive bound 1/6 h²/D** (the explicit Laplacian's positivity, Σ_a 2 D
dt/h² ≤ 1; 2D keeps 1/4): at 1/4 the 8-cell round burner overshot T to 8.6
(σ_st 6.98) with Y_O < 0 at the inner tip; at 1/6 bounded. 3D planar runs
take 1.5× more diffusion-bound steps than S4 measured (LM7's cost). (4)
`--mode bunsen` recomputes dt every step by default (`--dt-recompute 1`):
the burner IC is not the projected flow, and a dt held 16 steps diverged
the 4-cell partially premixed slot within 50 steps.

## Low-Mach Bunsen (`--mode bunsen --physics lowmach`; masterplan real_bunsen S5)

The slot (2D, 3D z-periodic) and round-nozzle burner on the low-Mach solver,
single-reactant (C0) or **partially premixed** (`--equivalence`, C2/C2b).
Files: `dns/lowmach.{hpp,cpp}` (`lmInflowRow`, `lmPartialPremixMix`,
`lowmachBunsenIC`, `lmPremixedWeight`), `gpu/lowmach` (per-row inflow,
`cellVelocity`), `gpu/derived` (`flameIndexCentred`), `gpu/runner_lowmach`
(the cone measurement, the C2 columns, frames), `dns/drive/flame.cpp` (rows),
config (bunsen admitted with lowmach; `--pulse` rejected; `--mix-phi`
rejected on the burner; ny widened to a 7-smooth size into the coflow).

**Inflow face per row** (j + ny k): m = u/t, T, Y_k of `lmInflowRow` (the
bunsen profile's weights J, P, C). The device ALWAYS reads per-row arrays;
the premixed flame fills them from the scalars, bit for bit
(`gpu_lowmach_passive` "rows"). No sink band: `--length` is the whole box.

**Partially premixed burner.** Jet = the fresh rich premix (Y_F1, Y_O1, Tu),
pilot strips = its burnt products (Y_F,rem, Y_O 0, `--pilot-temp`, default
σ_in), coflow = AIR (Y_O2, `--coflow-temp`, default 1). Where products meet
air (the pilot/air edge of the rows, the IC's column) the mixture is laid
**burnt to completion** (`lmPartialPremixMix`: Burke–Schumann, s Y_F − Y_O
and T + (q/cp) Y_F kept): laid as a linear blend, the IC's box-long column
exploded at t = 0 (`--length 90` diverged in its first steps). Z = J + P
exactly. Without `--pilot-width` the validator warns (nothing hot anchors the
cone on cold air; the rim is S8).

**Flame index** (D14): ξ = ∇Y_F·∇Y_O/(|∇Y_F||∇Y_O|) per cell, central, x
one-sided at the ends, 0 where either |∇Y| < 2^-24/h (the vote threshold).
The runner splits ω by w_p = smoothstep(−0.2, 0.2, ξ)
(`lmPremixedWeight`, the ONE definition the S9 renderer reads):
burnPre = ∫ω w_p, burnDif = ∫ω(1 − w_p), preFrac. The inner cone is
measured on ω w_p (the outer sheet must not qualify an inner flank);
SLflank and ST are per ρ_u(Y_F1 − Y_F,rem). xTipOut = the axis' first
downward Z = Z_st crossing (NaN: open). `gpu_lowmach_passive` findex: a
pure-x premixed layer/sheet reads exactly ±1, a wavy one 1.8e-7 / 2.4e-7
off (2D / 3D), 1-ulp noise 0 everywhere.

**Measurement.** The slot on SPAN-MEAN planes (FP64, exact for an extruded
field), every integral per unit span; the round nozzle's cone on the axial
plane z = zc, its integrals over the box. fuelIn is the inflow FACE's exact
flux (convective + diffusive, the kernel's expression in FP64).

**G2a vs the compressible solver and reactingFoam** (σ 5 β 8 C0, W 12, U 5,
burnt coflow, 8 cells, window [20, 25]; low-Mach `--length 49` = the
392 × 288 mesh; compressible `--nscbc-tinflow --nscbc-etain 1000`;
`gpu_lowmach_bunsen2d`, `tools/chem/foam_flame.py compare G2a`, twin
`dns_lowmach`, foam = the existing `foam_mclin_c8`, not re-run):

| | low-Mach | M 0.01 | M 0.02 | M → 0 (M²) | foam |
|---|---|---|---|---|---|
| xTip | 43.927 | 44.836 | 45.002 | 44.781 | 43.716 |
| tanA | 0.20929 | 0.21454 | 0.21276 | 0.21514 | 0.21025 |
| sinRatio | 1.02426 | 1.04886 | 1.04053 | 1.05163 | 1.02876 |
| SLflank | 0.99183 | 0.99006 | 0.98957 | 0.99022 | 1.00392 |
| mFlank | 0.99245 | 0.99172 | 0.99072 | 0.99206 | 1.00304 |
| burnRatio | 0.99999 | 1.00214 | 1.00122 | 1.00245 | 1.00318 |

Low-Mach vs M → 0: xTip −1.9%, sinRatio −2.6% (the plan's 1% NOT met),
SLflank +0.16%, mFlank +0.04%, burnRatio −0.25%. Low-Mach vs foam: xTip
+0.48%, sinRatio −0.44%, SLflank −1.2%; the temperature field at t 25:
θ Linf 0.076, L2 4.6e-3 (compressible default 0.42 / 0.022, T-held 0.62 /
0.034); jet T at x 20 +5.0e-3 (foam +4.6e-3, compressible +9.7e-3). The
compressible cone sits 2–3% off BOTH (sink band not the cause: `--nscbc-
sink 0` reads 44.69 / 1.0484) — issues LM9. The low-Mach cone is steady to
1e-5 (halves), burn = fuel inflow to 1e-5, Tmax = σ exactly; the IC's
unprojected flow undershoots Y_F to −8e-4 for t < 5 (clean from t 5.2);
mass vs the FP64 tally −4.7e-8 (constant from the first stats row: the
IC adjustment's FP32 divergence rounding). Cost 0.51–0.68 ms/step at 113k
cells, dt 1.9e-3 (diffusion), ~16k steps for 30 τ (8 s); the compressible
M 0.01 92k steps at 0.38 ms (35 s).

**Partially premixed slot** (φ 1.3 C2b, σ 6.88, Sutherland 298 K, Pr 0.52;
`gpu_lowmach_rich2d`). W 4 δ_th, U 3, 1.5 δ_th rich-product pilots at U,
air coflow 6 δ_th at 1 S_L, 8 cells, length 50 δ_th, window [30, 40]:
inner tip 17.71, **outer cone closed at 80.09 δ_D** (steady to 1e-4),
burn 3.8265 = premixed 2.7807 + diffusion 1.0456, **preFrac 0.7267** (1/φ =
0.769 if the inner cone burnt exactly Y_F1 − Y_F,rem: the triple-flame base
and the cone's lean-mixing edge carry ξ < 0), burnRatio 0.99987, Z in
[0, 1], yHits 0; dt 3.42e-3 diffusion-bound, dtChem 1.22e-2 (3.6×
headroom at K 1); 0.63 ms/step. The plan's W 12 / U 5 / 3 δ_th pilots
does not close its outer cone in an affordable box: measured open at 45
δ_th, the tip still moving at 410 δ_D of 424 at `--length 90` (mixing
length ~ (W/2 + pilot)² U/D at Z_st 0.78: ~180 δ_th).

**3D.** The nz 8 extrusion equals 2D BYTE FOR BYTE (mid-span frames, every
stats column but the box totals and dtDiff) when the same dt bound binds
— the 3D diffusive bound is 1/6 vs 1/4, so `gpu_lowmach_extrude` pins dt
by `--cfl 0.1` (the convective rate of an extruded field is the 2D one).
Round nozzle (`gpu_lowmach_round`: nz 64, D 8 δ_th, 4 cells, C2 + pilots):
the axial-plane cone (tip 72.6 δ_D at t 3), 2.7 ms/step at 573k cells
(4.7 ns/cell-step); at 8 cells (4.6M) 18.4 ms/step. **At 4 cells the
default dt is not bounded** (issues LM10): diffusion-bound dt 0.0091 lets
the inner tip overshoot T (7.6 vs σ_st 6.98) and undershoot Y (5e4 cells);
`--chem-dt 0.5` (dt 0.0055) or `--cfl 0.3` is bounded; 8 cells is bounded at
the defaults. The test runs `--chem-dt 0.5`.

## Low-Mach open laterals (`--laterals open`; masterplan real_bunsen S6)

The low-Mach burner's y faces (and z around `--nozzle round`; the slot's z
stays its periodic span) as OPEN faces entraining quiescent room air;
`--laterals periodic` (the default) is the S5 solver bit for bit
(`runs/s6/lmbit.sh`: 7 low-Mach cases, every output byte-identical to the
pre-S6 binary). The compressible burner rejects the flag by name (D6).

**Face rule** (ONE per face, decided by the sign of the face's own mass
flux). Traction-free: p = φ = 0 on the face (the pressure solve's DD kind:
one direct DST pass, no iteration), m_n from the projection (m −= cr G φ
with G = (φ_c − 0)/(h/2) on the boundary face), zero normal-stress
gradient. OUTFLOW: zero gradient — every cell-centred read across the face
is the boundary cell's (the kernels' `Lat<OY, OZ>` clamps instead of
wrapping), scalars leave with their own value. INFLOW: the quiescent
AMBIENT — T = Tu, Y = air (`--equivalence`) or Y = 0 (C0), ρ_amb = P0/(R
Tu) as the face density, tangential velocity 0 on the face (the ghost
antimirrors), and the face pressure −β ρ_amb u_n²/2 (β = 1): the
total-pressure inflow condition p + ρu_n²/2 = 0, i.e. Moghadam et al.
2011's backflow stabilization, which removes the incoming kinetic-energy
flux. No diffusion through an open face either way. The x-hi outflow's
BACKFLOW takes the same rule when the laterals are open (an ambient exists).
Storage: the y = Ly faces in `myHi` (nx nz), z = Lz in `mzHi` (nx ny), each
written by its one adjacent cell. Own instantiations: every kernel is
templated on (OY, OZ); (false, false) is the S5 code path.

**Why the stabilization** (`gpu_lowmach_passive` backflow): a nearly
inviscid vortex (μ 0.01, cell Re ~ 40) straddling the open y-hi face, 2000
steps: with β = 1 the kinetic energy never exceeds its initial value; with
β = 0 (API knob `LowMachSetup::backflow`) the open face injects energy, KE
grows 4.42× — a plain p = 0 face with backflow is unstable for the central
scheme.

**Tallies.** The FP64 mass tally adds the lateral faces; a new FP64
SPECIES tally (`speciesTally(k)`) sums the stage's exact boundary face
fluxes (x-lo convective + diffusive, x-hi and the open faces convective;
`bflux_`, written by `kLmScalars`), in every mode. It carries no chemistry:
inventory − tally = −∫∫ω. API (`gpu_lowmach_passive`): the three passive
cases with open faces keep a uniform Y bitwise, mass and species = their
tallies to 3.6–3.9e-11; a quiescent open box (2D open y, 3D open y and z)
stays at u = p = 0 exactly over 1000 steps; a pure-fuel box (hot blob, fuel
jet) under an AIR ambient: mass, fuel and oxygen = their tallies to
3.6e-11 – 1.9e-10 of the throughput, Y_O in [0, Y_O2] — every oxygen atom
in the box came in through an inflow face carrying the ambient.

**Stats** (open only, appended): mdotIn, mdotOut, mdotLat (net IN through
the laterals), uLatMin, uLatMax (outward normal velocity m_n/ρ_upwind on
the lateral faces), and a C0 fuelOut; fuelOut is x-hi + the laterals,
ambient-aware (`DeviceLowMach::boundarySums`, one fixed-order block); the
slot's per unit span. `--reaction off` (S6) makes the burner INERT: no
chemistry, the IC the inflow rows extruded, no anchoring guards.
`--profile-dt` on the low-Mach burner writes `<out>_jet.csv` per x column:
uc, Tc, Yc on the axis, J = ∫ρu², P = ∫p, M = ∫ρu, Q = ∫u, b (u = uc/2),
Qj (Q within 3 b) — slot per unit span on the span mean, round over the
cross-section.

**Bickley** (`gpu_lowmach_bickley`; inert 2D planar jet, W 4 δ_th =
6.4066 δ_D = 16 cells, U 3, x-lo a wall outside the slot, 60 W long, 15 W
clearance per side, tt 1200, window x ∈ [20, 50] W; Bickley 1937 with K =
the FAR-FIELD momentum flux, the window mean of J):

| | Re 60 (test) | Re 30 |
|---|---|---|
| u_c^-3 slope / 32ν/(3K²) | 0.984 | 0.989 |
| u_c exponent (u_c^-3 origin; free origin) | −0.3330 (−0.341) | −0.3316 (−0.374) |
| b^1.5 slope / Bickley | 0.976 | 0.954 |
| Qj³ slope / 36Kν tanh³(2.64) | 0.967 | 0.908 |
| dQj/dx exponent (−2/3) | −0.680 | −0.767 |
| J over the window | ±0.38% | ±1.04% |
| J + P along the box | 0.36% | 0.55% |
| wall suction P(x-lo cell) / J_in | −0.119 | −0.250 |
| u_c steady (last 50 τ) | 7e-5 | 3.6e-3 |

Every residual halves from Re 30 to 60: the O(1/Re) higher-order
boundary-layer terms of the similarity solution, not the boundary (the test
runs Re 60). **J is not conserved, J + P is**: a jet issuing from a wall
hands part of its momentum flux to the wall suction its entrainment flow
draws at the base (Schneider 1985, "Decay of momentum flux in submerged
jets"): P(x-lo) = −0.25 J_in at Re 30, −0.12 at Re 60, decaying to ~0 by 10
W; downstream J + P = J is constant to the quoted digits. The box's Q =
∫u dy over the full width is not the jet's (the induced flow runs back
toward the wall outside the jet: dQ/dx exponent −1.0); Qj is. Mass balance
(Re 60): in 19.22 + lateral 38.307 = out 57.527; uLatMin −0.099, every
lateral face an inflow. **No confinement**: halving the clearance (15 W →
7.5 W, the jet's 3b reaches the face at ~40 W) moves u_c ≤ 0.32%, b ≤ 0.58%,
J ≤ 1.04% over [5, 50] W; the integrated base suction moves 10% (−6.41 →
−5.80: where p = 0 is imposed moves the potential flow's pressure, not the
jet). **Periodic laterals confine**: the net Q is constant along the box
(19.22 = the jet's own flux: no lateral entrainment, the jet feeds on its
own recirculation through x-hi), J varies ±9.6–10%, the u_c^-3 slope reads
1.54× Bickley. 480k cells, 1.66 ms/step (open) / 1.41 (periodic), 21k
steps.

**The partially premixed outer cone with open laterals (issues LM11,
LM13).** rich2d's slot (φ 1.3 C2b, W 4, U 3, pilots 1.5 at U, air coflow 1
S_L, 8 cells, length 50), window [150, 200] (the starting-jet head needs t
~ 100 to leave the box: at t 40 the box was still draining mass at −115/τ,
every column unsteady):

| laterals, clearance | xTipOut | preFrac | mdotIn | mdotLat (net in) | uLat [min, max] |
|---|---|---|---|---|---|
| periodic 6 (S5) | 80.09 | 0.7267 | — | — | — |
| open 6 | 90.47 | 0.7268 | 116.17 | −32.07 | [−0.092, 0.40] |
| open 3 | open | 0.7538 | 85.56 | −30.16 | [+1.8e-4, 1.18] |
| open 2 | open | 0.8100 | 74.96 | −27.38 | [+9e-7, 1.80] |
| open 6, coflow 0 (room air) | open | 0.7780 | 57.48 | −6.09 | [−0.21, 0.30] |
| W 12, U 5, pilots 3, open 6 | open | 0.9325 | 348.4 | −237.8 | [+0.016, 5.1] |

**In 2D, open laterals do NOT close the outer cone sooner — the opposite.**
The flame's dilatation (σ 6.9) pushes gas out sideways; a planar jet
entrains weakly (dQ/dx ∝ x^-2/3); so the laterals are a net OUTFLOW and at a
clearance ≤ 3 δ_th every lateral face flows out — the coflow air the
periodic box kept beside the jet leaves the box, and the products never
dilute to Z_st. All steady to 1e-6, burnRatio 1.0000, Z ∈ [0, 1], 0 hits,
Tmax ≤ σ_st (`gpu_lowmach_openrich` pins the clearance-6 case). **3D round
closes regardless** (D 6 δ_th, U 3, pilots 1.5 at U, 4 cells, `--chem-dt
0.5`, length 40, window [45, 60]): xTipOut 41.84 periodic clearance 3,
42.57 open 3, 41.99 open 6 (radial dilution: a round jet's entrainment
grows ∝ x); laterals net outflow there too (the head transient still
draining the box at t 60). The real burner's lab plume rises by buoyancy
(S7), which neither box has.

**LM10 data point** (open round, D 6, U 3, 4 cells, default `--chem-dt
1`): dt 0.0071 is CONVECTIVE-bound (cfl 0.8; dtDiff 0.0097, dtChem 0.0116)
and Y_O undershoots in up to 11156 cells (T ≤ σ_st); `--chem-dt 0.5` (dt
0.0055) has 0 hits. The binding limit there is the MC positivity of the
convective step (~half the upwind CFL), not diffusion.

**Restart** (`gpu_lowmach_restart`): the open partially premixed slot,
the inert open jet with its jet profiles, and the open round nozzle (y, z)
restart byte for byte (checkpoint records `state.myhi`, `state.mzhi`,
`state.ytally`).

## Low-Mach gravity (`--gravity`; masterplan real_bunsen S7)

`--gravity G` on the low-Mach burner: g along −x in S_L²/δ_D (the jet axis
points UP), [0, 100]; absent (or 0) is the S6 solver bit for bit
(`runs/s7/lmbit.sh`: 10 low-Mach cases, periodic and open, + 3 with
`--gravity 0`, every output byte-identical to the pre-S7 binary; the
compressible path byte-identical to HEAD e92dad1, `runs/s7/bitid.sh`, 12
cases). The compressible burner rejects it by name (D6); `--gravity` stays
the candle's elsewhere, `--quiescent` the candle's only.

**The ONE buoyancy term** (`gpu/lowmach.hip` `addBuoyancyLm`, the `Grav`
instantiation of `kLmMomX`): every x face's momentum predictor adds
−(ρ_f − ρ_a) g, ρ_f = the density the face velocity reads (`rhoX`: the
two-cell mean; the outflow face the last cell's, the ambient's on
backflow), ρ_a = P0/(R T_amb) = the Tu room air. The ambient's hydrostatic
pressure ρ_a g x lives in p, so the quiescent ambient is p = 0 EXACTLY and
every open face's p = 0 (outflow, laterals; −ρ u_n²/2 on inflow) stays the
ambient's pressure: no hydrostatic face condition is needed (the
compressible candle needs hydrostatic LODI waves and an ambient column
p_a(x) for the same thing). No energy term: the low-Mach T equation has no
pressure work (P0 constant; the compressible's −g m_x is the work against
the stratification, O(M²)).

**Real g** (M1 mapping, GRI-3.0 φ 1.3: S_L 0.23124 m/s, δ_th 0.61874 mm,
the C2 model's δ_th/δ_D 4.70964 → δ_D,model 0.13138 mm): g = 9.80665 ×
δ_D/S_L² = **0.0241 S_L²/δ_D** (`lmRealGravity`; printed by the dry-run
and the flag's range error). The dry-run adds Fr = U²/(g D), Ri = g D
(1 − 1/σ)/U², the free-fall speed u_ff = sqrt(g D (1 − 1/σ)) and its value
over the box length; with periodic laterals a warning (a lattice of
plumes). Label: `| GRAVITY g=G S_L^2/delta_D along -x (...)`; json
`"gravity"`.

**Validation (API, `gpu_lowmach_hydro`, `gpu_lowmach_plume`).**

| check | measured | gate |
|---|---|---|
| quiescent isothermal box, g 1, 1000 steps (2D periodic, 2D open y, 3D open y z) | u = p = 0 bitwise | exact |
| stably stratified column (T 1 → 1.5 tanh, closed bottom, periodic laterals), g 1 vs g 0, 1000 steps | \|m(g) − m(0)\| 4.7e-7 (2D) / 1.9e-9 (3D) of the free-fall scale 4 | 1e-4 |
| same: p(g) − p(0) vs the discrete hydrostatic sum of the current ρ | 5.7e-5 / 4.1e-5 of its range 2.7 (the stage density's lag) | 1e-4 |
| hot blob (T 1 + e^(−r²/4)) at rest, open y, g 0.5, t 10 | heat centroid +0.858 δ_D, net x momentum 0.911 of the buoyancy impulse; g 0: 1e-5 | ≥ 0.5, [0.8, 1]; 1e-3 |
| laminar plane plume (hot slot T 1.2, u 0.25, 4 δ_D, g 6, 192 × 128, h 0.5, steady to 5e-4), window x ∈ [60, 150] | exponents u_c +0.1999, b +0.4121, θ_c −0.6019 (one virtual origin) | ±0.03 of 1/5, 2/5, −3/5 |
| same: buoyancy (= enthalpy) flux ∫(m(T − Ta) − ρD ∂T/∂x) dy | constant to 7.8e-5 | 3% |
| same vs the Boussinesq similarity solution (derived in the test, Pr 0.72: f'(0) 0.80961, h(0) 0.37728, η_half 2.24567) at B = g E | u_c⁵ slope 0.982, b^(5/2) 0.969, g'_c^(−5/3) 0.974 (u_c 0.4%, b 1.3%, g'_c 1.6%) | 3%, 5%, 5% |

The 1D column is the well-balanced check: a laterally uniform buoyancy is
a discrete gradient, the projection absorbs it, the flow is the g-free
conduction flow to the FP32 floor. (Measured on the way: a blob 6 radii
from the x-lo face drifts +0.12 δ_D in t 10 at g 0 — the inflow FACE at T
1 conducts heat out of its lower half; the test's box puts the wall 7.5
diffusion lengths away.)

**Candle cross-check** (`gpu_lowmach_candle`, API: the compressible
candle's defaults — W 1 = Pe 1, g 14 U²/l, Da 5000, periodic y, the same
288 × 200 grid, rows = `diffusionInflow` at the x-lo FACE, the ignition
column, `measureJet`'s tip, t ∈ [20, 30]; units 1:1, T0 = P0 = 1):

| | low-Mach | compressible default | compressible `--nscbc-tinflow --nscbc-etain 1000` (D10's comparable variant) |
|---|---|---|---|
| tip (l) | 7.029 (steady, 7.023–7.043) | 10.785 | 8.031 (unsteady 7.36–8.33) |
| burn ∫ω | 0.887 | — | 1.037 (fuelIn 0.924) |
| tip / burn | 7.92 | — | 7.74 (+2.3%) |
| g = 0 tip | 10.41 | 25.31 (fuelIn 1.105, burn 1.160) | — |

**The plan's gate (tip within 5% of 10.79) is NOT met (−34.8%).** At Pe =
1 the flame length follows the fuel the inflow passes, and the two inflows
pass different amounts: the low-Mach face's convective nominal is 0.405,
Fick through the face carries 54% of the 0.887 it burns (the sheet sits on
the exit plane); the compressible LODI row overshoots its target (issues
§68; default fuelIn 1.105 at g 0), its D10 variant less. Per unit burned
fuel the tips agree to 2.3% (the test's gate, 5%). The compressible candle
budget itself does not close (burnRatio 1.12 at etaIn 1000, 1.03 default)
and its zMax reads 0.852 constant in both variants, above the rows' max Z
0.772 (not investigated: compressible-side, issues LM15).

**LM13 with gravity** (`runs/s7/lm13.sh`, φ 1.3 C2b, σ 6.88,
Sutherland 298 K, Pr 0.52, open laterals; 2D: S6's rich2d slot W 4, U 3,
pilots 1.5 at U, air coflow 1 S_L, 8 cells, length 50, window [150, 200];
real g = 0.0241, Fr = U²/(g W) 19.8):

| case | g | xTipOut (range, period) | mdotLat (net in) | uLat [min, max] |
|---|---|---|---|---|
| clearance 6 | 0 (S6) | 90.47 | −32.07 | [−0.09, 0.40] |
| clearance 6 | 0.0241 | 82.40 (steady) | −11.80 | [−0.03, 0.22] |
| clearance 6 | 0.0964 (4×) | 73.80 (67.6–85.1, ~36 τ) | +83.26 | [−2.97, 6.00] |
| clearance 6 | 0.289 (12×) | 64.02 (49.8–100.4, ~27 τ) | +122.2 | [−4.84, 8.16] |
| clearance 3 | 0 (S6) | open | −30.16 | |
| clearance 3 | 0.0241 | open | −22.46 | [−0.13, 0.82] |
| clearance 3 | 0.289 | 62.26 (56.3–69.7, ~17.7 τ) | +101.4 | [−4.77, 8.83] |
| clearance 2 | 0.0241 | open | −24.34 | [−0.12, 1.47] |
| W 12, U 5, pilots 3, clearance 6, length 90 | 0.0241 | open | −168.6 (S6 −237.8) | |
| same | 0.0964 | open (preFrac 0.834 vs 0.932) | −33.06 | |

**Real gravity does not close the 2D outer cone where the open box left it
open**: it shortens the clearance-6 cone 90.47 → 82.40 (the periodic box's
80.09) and cuts the dilatation-driven lateral OUTflow ×2.7, but at
clearance ≤ 3 and on the W 12 burner the cone stays open at real g. At 4–12×
real g the laterals turn into a net INflow (the plume entrains), the cone
closes (62–74 δ_D) and FLICKERS (xTipOut swings 17–50 δ_D, periods 18–36
τ; the pool-fire scale 0.479 sqrt(g/W), W = 18.84 δ_D, gives 29 τ at 4×
and 17 τ at 12× — the 2D slot in a 6 δ_th clearance is no pool fire).
All bounded (yHits 0, Tmax ≤ σ_st; at 12× Z dips to −1.5e-4).

**3D round** (D 6 δ_th, U 3, pilots 1.5 at U, 4 cells, `--chem-dt 0.5`,
length 40, open clearance 3/6): xTipOut 42.57 (g 0, S6) → 42.15 at real g
(t 200: steady to 1e-4) → 41.46 at 4× (a ±0.003 ripple, ~17 τ); clearance
6 at real g 41.74. The inner tip moves ≤ 0.2%. **No flicker** at this
scale up to t 200 (a real 3.7 mm burner's flicker period would be ~70 τ).
Gravity turns the 3D box's lateral flux from net out to net in only at 4×
(−793 → +2930 at t ∈ [100, 200]). LM10 note: under gravity these 4-cell
runs are CONVECTIVE-bound below the `--chem-dt 0.5` dt (dt 0.0032 at real
g, 0.0038 at 4×, vs dtChem 0.0058): bounded (0 hits) — S9's dt rule must
see the buoyant plume's speed. Cost: 2D 0.71 ms/step with and without
gravity (the term is one load + one FMA per x face).

**Restart** (`gpu_lowmach_restart`): the open partially premixed slot and
the open round nozzle under `--gravity 0.5` restart byte for byte; a
gravity checkpoint resumed without `--gravity` or with another g is exit 3
naming `--gravity` (`--gravity 0.5 vs 0.25`).

## Low-Mach rim (`--rim`; masterplan real_bunsen S8)

`--rim LIP` on the low-Mach burner: the REAL tube rim -- an immersed,
no-slip, impermeable, isothermal tube wall of thickness LIP δ_th (≥ 4
cells) around the jet (inner diameter = `--slot-width`; the slot's two
lips), standing on the inflow face up to `--tube-length` (δ_th, ≥ 8 cells,
default 7.5; snapped to a cell face) inside the box; `--length` is the box
ABOVE the rim. `--rim-temp R` = T_w/Tu ∈ [1, 3] (default 1). With it the jet
rows are the developed Poiseuille profile (bulk `--jet-speed`; D12), the
tube's rows closed (m = 0), outside the room draft `--coflow-speed`
(default 0.4 S_L, D15) of cold room air (C2: air; C0: Y = 0); NO pilot:
`--pilot-*`, `--coflow-temp`, `--slot-edge`, `--pulse*` are rejected by
name; the compressible burner and the planar flame reject `--rim` (D6);
`--tube-length` / `--rim-temp` need `--rim`. Absent = the S7 solver bit for
bit (`runs/s8/lmbit.sh`: 12 low-Mach cases + 3 with `--gravity 0`, periodic
and open, with and without gravity, every output byte-identical to the
pre-S8 binary; the compressible path byte-identical to HEAD e92dad1,
`runs/s8/bitid.sh`, 12 cases).

**The ONE tube definition** (`dns/lowmach.hpp`): `lmTubeSdf` = the exact
distance to the wall R_i ≤ r ≤ R_o, x ≤ x_rim in the (x, r) half plane (r =
\|y − y_c\| slot, hypot round; a true sdf); `lmTubeSolid` = the STAIRCASE:
a cell is solid when sdf(centre) ≤ 0. The solver, the frames' mask (solid
cells greyed) and the run's `.json` `"rim": {"x","ri","ro","h","T"}` (box
units, for the renderer) all read it.

**The wall** (`gpu/lowmach.hip`, a `Wall` instantiation of every flux
kernel -- `kLmTemperature`, `kLmMomX`, `kLmMomT`, `kLmCorrectX/C`,
`kLmScalars`; no solid = the S7 kernels bit for bit):
- **`LmPressure`'s face codes are the face-open definition**: a MAC face with
  a solid cell on either side is CLOSED (for an interior face exactly the
  complement of the codes' face-open bit). Its m is 0 in every stage: the
  momentum predictor writes 0, the correction skips it (the codes'
  `kLmFaceXLo/YLo/ZLo` bits; an open boundary face: its cell), the
  apertured PCG sees no flux there. So the wall is impermeable to mass and,
  riding m, carries no convective momentum, enthalpy or species flux.
- **No-slip**: a closed TANGENTIAL neighbour face reads as the antimirror of
  the face itself (u = 0 on the wall: the inflow face's tangential rule);
  a closed face in the NORMAL direction reads its 0 (the wall's normal
  velocity).
- **Isothermal** at T_w: each wall face conducts ρD(T_w) (T_c − T_w)/(h/2)
  (the inflow face's form). Species-tight: zero diffusive flux too. A face
  whose upwind SECOND neighbour is solid goes central in the shared limiter
  (no second neighbour; the walled compressible kernel's rule).
- The tube's cells carry ρ = P0/(R T_w) and ρY = 0 unchanged (R_T = 0, no
  reaction, the words kept bitwise).
- **Pressure**: the apertured FP32 PCG (S3) preconditioned by the box's
  spectral solve, stop \|r\| ≤ 1e-6 \|b\|, warm-started per STAGE from that
  stage's increment of the previous step (`state.phi0..2` in the
  checkpoint); a solve that does not converge in 200 iterations is exit 8.
- **Staircase, not sub-cell** (a DEVIATION from the plan's diffusion-only
  ghost view): the wall sits on the cell faces. Exact for the grid-aligned
  slot lips (the channel test below); the round tube's circle is a
  staircase (first-order wall position; issues LM17).

**The Poiseuille rows** (`lmPoiseuille`): the discrete steady solution of
the solver's viscous operator, not the continuous parabola. Slot: c ((R_e
+ d)² − r²), R_e = the staircase wall face, d = (√(W_e² + h²) − W_e)/2 --
with the antimirror ghost the 3-point Laplacian of this parabola is
constant up to the wall rows (host-checked to 1e-12); round: 2 (1 − r²/
R_i²). Both scaled to bulk 1 over the tube's inflow cells.

**Stats** (bunsen, `--rim`): pcgIt = the most PCG iterations of the last
step's three stage solves (0 without a solid); + xRimBase (xBase − x_rim:
the stand-off above the rim), rBase (the base row's flank half-width on the
cone plane), qRim (`lmWallHeat`: the conduction into the tube through every
fluid/solid face, the stage's face expression with λ_w = cp μ(T_w)/Pr; per
unit span on the slot). Exit 8 adds: the base below the rim (flash-back
into the tube), blow-off = xRimBase past half the cone height above the
rim. The IC (`lowmachBunsenIC`): fresh premix in the tube, the ideal cone
standing on the rim, its products a SHORT column (r < R_i, up to 4 δ_th past
the tip) laid against the air as its Burke–Schumann sheet, air elsewhere.

**Validation (`test_gpu_lowmach_rim`).**

| test | case | measured | gate |
|---|---|---|---|
| `gpu_lowmach_channel` | 2D inert channel W 8 (16 cells) between 4-cell lips, tube 96/128 cells, Re 11, 3000 steps | profile in the tube (≥ 3.5 W from the exit) 6.4e-7 U, v 6.3e-7; dp/dx −0.133954 = the rows' discrete curvature μ D2u to 7e-6, 0.99225 of −12 μU/W² (the h²/(2R²) of a 16-cell channel); wall shear 2μu₁/h = −dp/dx W/2 to 1e-6 (0.99225 of 6μU/W); every face touching the tube m == 0; mass = tally | 1e-4, 1e-5, 1e-3, 1%, 1e-3, 1%, exact, 1e-6 |
| `gpu_lowmach_rim_cond` | 2D, open y, slab across y, x-lo face T 2 with m 0, constant μ, 500 steps | the LINEAR profile is the fixed point: \|T − T_lin\| 2.4e-7 (T_w 1) / 1.2e-7 (T_w 1.5), \|m\| ≤ 4.6e-7; qRim = λ ΔT/L L_y to 2.4e-7 / exact, = the inflow face's conduction (closure) | 2e-6 |
| `gpu_lowmach_rim_imperm` | 3D round tube (D 8 cells, lip 3), hot Poiseuille jet T 2, open y z, g 0.5, passive Y 0.5, 300 steps | every face touching the tube m == 0; the tube's ρ, ρY unchanged bitwise; ρY == 0.5 ρ bitwise in every fluid cell; mass − tally 2.9e-13, species − tally 3.1e-13; PCG 13 iterations cold, ≤ 14 warm; rerun byte-identical | exact, 1e-6, ≤ 15 |
| `gpu_lowmach_rim_anchor` | the plan's 2D case (below, U 6), tt 40, `--chem-dt 0.5`, T_w 1 and 1.5 | anchored on the rim (xRimBase 4.12 = 0.87 δ_th, rBase 37.1 / 35.1 over the lip, flat over [30, 40]), pcgIt ≤ 12, yHits 0 (Tmax 7.06) | (0, 4 δ_th], one cell, ≤ 15, 0 |
| `gpu_lowmach_restart` | the 2D rich slot on its rim (bin frames), the round tube open under `--gravity 0.5` (volumes) | byte rule, 1 + 3 restarts (the PCG warm starts are in the checkpoint) | bytes |

**Anchoring on the cold rim (2D; `runs/s8/map`)**: the rich φ 1.3 slot at
mapping M1 (C2b, σ 6.88, Sutherland 298 K, Pr 0.41), W 14 δ_th, lip 1.4,
open laterals at clearance 6, no gravity, tt 40, window [30, 40] (R_i 32.97,
R_o 39.56 δ_D):

| U/S_L | cells/δ_th | xRimBase δ_D (δ_th) | rBase | inner tip − x_rim | preFrac | Tmax | yHits max (yMin) |
|---|---|---|---|---|---|---|---|
| 1.5 | 4 | 4.12 (0.87) | 34.2 | 31.0 | 0.923 | 6.98 | 920 (−3e-7) |
| 2 | 4 | 2.94 (0.62) | 39.1 | 50.1 | 0.944 | 6.99 | 1840 (−2e-6) |
| 3 | 4 | 2.94 (0.62) | 39.3 | 89.1 | 0.966 | 7.02 | 3340 (−1e-5) |
| 4 | 4 | 2.94 (0.62) | 39.4 | 130.4 | 0.975 | 7.05 | 5100 (−4e-5) |
| 6 | 4 | 4.12 (0.87) | 37.1 | 211.8 | 0.970 | 7.14 | 6390 (−3e-4) |
| 8 | 4 | 4.12 (0.87) | 37.2 | 280.5 | 0.973 | 7.83 | 8000 (−5e-3) |
| 10 | 4 | 4.12 (0.87) | 37.3 | 341.5 | 0.977 | 8.42 | 20087 (−9e-3) |
| 12 | 4 | 4.12 (0.87) | 37.3 | 398.4 | 0.975 | 8.54 | 24793 (−1e-2) |
| 15 | 4 | 4.12 (0.87) | 37.2 | 503.0 | 0.976 | 7.37 | 32938 (−1e-3) |
| 20 | 4 | 4.12 (0.87) | 37.0 | 678.4 | 0.985 | 7.72 | 40651 (−7e-4) |
| 1.5 | 8 | 3.83 (0.81) | 34.7 | 30.8 | 0.923 | 6.97 | 0 |
| 6 | 8 | 3.24 (0.69) | 38.6 | 209.2 | 0.972 | 7.05 | 0 |

**The cold rim anchors the rich slot over the whole admissible range**: no
flash-back at U 1.5 (the validator's floor; the flame base stays 0.8 δ_th
above the rim, the fresh core never burns into the tube), no blow-off at
U 20 within t 40 (the cones above U 10 are still the IC's within one flow-
through; the base settles within ~15 τ at every U). The base sits OVER the
lip (rBase between R_i and R_o) a stand-off 0.6–0.9 δ_th above it at 4
cells (the base row is a cell centre: 2.5 or 3.5 h) and 0.69 δ_th at 8
cells (U 6) -- 0.42 mm at M1 (δ_D,model 0.131 mm), a real rim's ~0.5–1 mm.
A WARM rim (`--rim-temp 1.5`, U 6, 4 cells) moves the base by less than a
cell (xRimBase 4.12, rBase 35.0 vs 37.0) and halves the heat drawn (qRim 48
vs 86 per unit span; 0.6% / 0.95% of the heat release). The real burner's
operating point (U ≈ 6.5 S_L, φ 1.3) anchors here; the map's LIMITS are the
MODEL's -- its cold-wall quench is the one-step's at Pe 0.6× (D1, K7), not
CH4's, and no flash-back / blow-off was found inside the validator's U
range (issues LM18). preFrac 0.92–0.98 in 2D, not the
plan's [0.72, 0.80]: the 2D outer cone stays open (LM13; it closes in 3D,
below). The plan's anchoring gate is met; its preFrac gate is a 3D one.

**LM10 with the rim** (2D U 10, 4 cells): default dt (cfl 0.8, diffusion /
convection bound) Tmax 8.42, 2.0e4 hits, Y_F min −9e-3; `--cfl 0.4` 7.27,
1.9e4, −9.5e-6; `--cfl 0.2` 7.18, 18, −2.9e-7; `--chem-dt 0.5` 7.20, **0
hits**, −2.4e-7 (U 6: default 7.14 / 6390 hits / −2.9e-4, `--chem-dt 0.5` 7.06 / 0 / −2.9e-8: the S8 anchor test ran it; S9's `--dt-rule cell` default replaced the workaround, below). The over/undershoots sit in the products near the outflow
(i > 160 of 270), not at the rim. 3D round (below) at the DEFAULT dt: 0
hits, Tmax 6.973 ≤ σ_st 6.977 (diffusion-bound dt 0.0092 at real g).
8 cells: 0 hits at the defaults.

**3D round** (`runs/s8/round`): D 6 δ_th, U 4, lip 1.4, tube 3, clearance 3,
open y and z, real g 0.0241, 4 cells, `--chem-dt 0.5`, length 30 (108 × 60
× 60 = 389k cells, 6240 solid), tt 100, window [80, 100]: anchored on the
rim (xRimBase 4.12 = 0.87 δ_th, rBase 16.9: over the lip, R_i 14.13 R_o
20.72); inner tip 7.51 δ_th above the rim (49.50, steady ±0.005; the ideal
uniform-U cone 11.6 δ_th: the base sits over the lip and the 3D tip
burns ahead, ST 1.70); outer cone CLOSED at 15.0 δ_th above the rim
(84.91, ±0.1); preFrac 0.7998 (in the plan's [0.72, 0.80]); burnRatio
0.9998; Z ∈ [1.2e-5, 1]; 0 hits; Tmax 6.85 (t ≥ 80); laterals net out −274
(uLat ∈ [−0.023, 0.093]); qRim 2923 = 4.4% of the heat release; pcgIt ≤ 9.
The outer cone needs t ~ 80 to settle (84.4 at t 40: the open box's
starting head, S6).

**Cost**: 2D 32k cells 3.1 ms/step (the S7 pilot burner ~0.7); 3D 389k
7.4 ms/step (19 ns/cell-step: launch-bound, 3 × 6–9 PCG iterations); the
final grid (section 5: 392 × 180 × 180 = 12.7M cells, 33840 solid) **171
ms/step with the rim vs 69 ms/step for the same box with pilots** (direct
solve; 42 / 33 steps incl. one stats row, pcgIt 11 at t 0.3): the
apertured PCG is ~100 ms/step, S3's estimate. At dt ~0.005–0.009 τ the
section 5 case (100 τ) is 11k–20k steps = 1900–3400 s: 2 simjobs.

## Low-Mach dt rule, real-burner scouts, renderer (masterplan real_bunsen S9)

**The dt rule (`--dt-rule bounds|cell`; issues LM10 settled).** The S4–S8
dt was the minimum of three SEPARATE bounds, each its own maximum over the
box (convective cfl/C, diffusive 1/D, chemistry K/J), so each spent the
whole explicit budget alone; the hot, fast products near the outer flame
hold large convection AND large diffusion at once, and at 4 cells/δ_th the
step overshot T and Y there (2D rim slot U ≥ 6, the 3D pilot burner).
`--dt-rule cell` adds the per-cell POSITIVITY bound of the explicit transport
step, dt ≤ K / max_c (2 C_c + D_c) (`DeviceLowMach::positivityDt`, functor
`LmPosRate`): C_c the convective rate of `LmConvRate` (×2: the MC face
value reaches twice the upwind slope), D_c = 2 Σ_a μ(T_c) max(4/3, 1/Pr)/
(ρ_c h_a²) (the cell's OWN diffusivity; at the hottest cell exactly
1/dtDiff), K = 1 (`kLmPosKDefault`: forward Euler's positivity bound, which
SSP-RK3 inherits). The three old bounds stay; stats append `dtPos`. **Default
`cell` on the low-Mach burner, `bounds` on the planar flame** (the S4 ladder
bit for bit; it never overshot). The burner's change is deliberate: its old
default is demonstrably unbounded at 4 cells (below); with an explicit
`--dt-rule bounds` every burner is the S8 binary byte for byte
(`runs/s9/lmbit.sh`).

Measured (`runs/s9/dt.sh`, `dtvar.sh`; 4 cells/δ_th, φ 1.3 C2b, σ 6.88,
Sutherland 298 K, tt 40; hits = cells with Y_F or Y_O outside [−1e-6,
1 + 1e-6]; "late" = t ≥ 20):

| case | rule | dt (late) | hits | Tmax late (σ_st 6.977) |
|---|---|---|---|---|
| 2D rim slot U 6 (S8 map) | bounds (S8) | 0.0138 | 6390 | 6.96 |
| | `--chem-dt 0.5` (S8) | 0.0073 | 0 | 6.95 |
| | **cell** | 0.0102 | **0** | 6.95 |
| 2D rim slot U 10 | bounds (S8) | 0.0138 | 20087 | 6.99 |
| | `--cfl 0.2` (S8) | 0.0102 | 18 | 6.98 |
| | `--chem-dt 0.5` (S8) | 0.0073 | 0 | 6.97 |
| | **cell** | 0.0093 | **0** | 6.98 |
| | cell, K 1.5 | 0.0128 | 18790 | — |
| 3D round pilot (S6's LM10: D 6, U 3, pilots 1.5, open 3) g 0 | bounds | 0.0091 | 1440 | 6.97 |
| | **cell** | 0.0076 | **0** | 6.97 |
| same, real g 0.0241 | bounds | 0.0091 | 1556 | 6.97 |
| | **cell** | 0.0075 | **0** | 6.97 |
| | cell, K 1.5 | 0.0091 | 1556 | — |
| 3D round rim (S8: D 6, U 4, real g) | bounds | 0.0092 | 0 | 6.95 |
| | **cell** | 0.0076 | 0 | 6.95 |

The cell rule is bounded everywhere measured and costs 1.2× (3D) to 1.4–
1.5× (2D slot) the old default's steps -- 1.3× FEWER than `--chem-dt 0.5`
(S8's workaround, which bounded by accident: it shrank dt). K 1.5 is not
bounded (the theory's 1 is the edge, not a margin). Adding the chemistry to
the per-cell sum (2C + D + J, or only J's consumption part kρ(Y_O + sY_F))
cost 1.5× more in dt and bounded nothing the transport sum did not
(`dtvar.sh` variants 0/1: dt 0.0060 / 0.0063 at U 10, 0.0047 / 0.0049 on the
pilot burner): chemistry keeps its own K/J. Results are dt-converged: the
3D rim's inner tip, outer tip, base, preFrac and qRim agree with the old
default's to the printed digits (49.25, 82.80 vs 82.85, 4.121, 0.8008,
2976); the G2a burner (8 cells, steady) reproduces every printed digit of
its S5 table under the new rule (a steady state of the RK3 march does not
depend on dt). Under `cell` the convective `--cfl` binds only below 0.5
(dtPos ≤ 1/(2 max C) < 0.8/max C): the per-cell bound is the MC limiter's
CFL 1/2 with the diffusion added. **Not dt-dependent** (every rule, every
dt, including dt 0.0037):
the 2D rim slot's IC transient overshoots T to 7.2 and Y_O above the air's
(Z −1.2e-2) for t < 10–15 τ (the unprojected IC, issues LM12/LM20), then
returns inside the bounds.

**SI echo** (`--si-delta-d MM --si-sl M_PER_S`, burner only, both or
neither; no physics): the dry-run prints δ_D, S_L, τ, ν_u = Pr S_L δ_D, g in
m/s², D, U, h, the box and the tube in mm, Re_D and tt in ms; the label
gains `| SI (echo, no physics): ...`; the .json gains `"si": {"deltaD" (m),
"SL" (m/s)}` -- with the `"rim"` record (box units, h) the renderer's real
scale. The final case's echo: ν_u 1.591e-5 m²/s (GRI-3.0's), g 9.806 m/s²,
D 9.999 mm, U 1.503 m/s, Re_D 944.5.

**Renderer** (`tools/chem/flame_room.py` in Blender, its numpy half
`tools/chem/flame_room_lib.py`): a run that wrote `findex` (the low-Mach
`--equivalence` burner) renders TWO cones -- VDB grids `em_pre` = max(ω, 0)
w_p(ξ)/ref and `em_dif` = max(ω, 0)(1 − w_p)/ref, w_p =
`flame_room_lib.premixed_weight`, the port of `dns::lmPremixedWeight` (the
ONE split definition; `flame_room_selftest` pins the formula,
`flame_room_split` reproduces the solver's burnPre / burnDif from the last
frame of `gpu_burner_round` to the CSV's 6 digits: 412.154018 vs 412.154,
110.706646 vs 110.707), each its own Emission node:

- inner premixed cone: CH* 431.4 nm (1.0) + C2* Swan 516.5 / 473.7 / 563.5
  nm (1.2 / 0.4 / 0.25: C2*/CH* rises with φ, ~1.5 band-integrated at φ 1.3
  for CH4/air) over a weak CO2* continuum (0.2) -- blue-green;
- outer diffusion cone (the rich products' CO and H2 burning in air): the CO
  + O → CO2* continuum (peak ~400 nm, falling through the green) + a CH*
  trace (0.15) -- violet-blue;
- both at EQUAL radiant power per unit ω (`cone_colours`): the violet reads
  0.116× the inner's luminance at equal power (the eye), and its heat
  release is spread thinner (the scout's projected outer luminance above the
  inner tip is 0.5% of the inner peak); `--dif-gain` (default 1 = no further
  assumption; a DISPLAY knob) scales its emission. No soot (laminar φ 1.3:
  a blue flame), burnt gas not drawn (CO2/H2O radiate in the IR): the light
  is ω > 0 only.

**Half-scale scouts of the final case** (`runs/s9/half.sh`, chained 5-min
simjobs through `tools/chem/chain.sh` with `CHAIN_DNS` a frozen copy; summary
`runs/s9/scouts.py`). The section-5 case at HALF SIZE, the same h = δ_th/4 =
0.1547 mm: CH4/air φ 1.3 (C2b, σ 6.899 = GRI-3.0's T_ad 2056 K),
Sutherland 298 K, Pr_eff 0.5245 (M1: δ_D,model = 0.61874 mm / 4.71614 =
0.13120 mm, τ 0.5674 ms), tube ID 5 mm = 8.081 δ_th, lip 1.0 δ_th (0.62 mm:
the 4-cell minimum; the half-scaled 0.5 mm wall is 3.2 cells), U_bulk 6.5
S_L = 1.50 m/s Poiseuille (Re_D 470, Fr 46), open y/z, real g 0.02406,
--length 60 above the rim, tube 4 δ_th below, 256 × 98 × 98 = 2.46M cells
(1.5 D half box: clearance 7.08 δ_th from the OD), `--dt-rule cell`:

| run | half box (clearance) | draft | window | inner tip above rim δ_th (mm) | outer tip δ_th (mm) | xRimBase δ_th | rBase | preFrac | burnRatio | qRim | mdotLat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (`a2`) | 1.5 D (7.08) | 0.4 | [140, 160] | 19.31 (11.95) | 39.88 (24.68) ± 0.05 | 0.875 | 21.23 | 0.7834 | 0.9999 | 3131 | +533 |
| B (`b2d`) | 2 D (11.12) | 0.4 | [140, 160] | 19.30 | 38.41 ± 0.33 (unsettled) | 0.875 | 21.23 | 0.7827 | 1.0008 | 3122 | +9068 |
| A | 1.5 D | 0.4 | [60, 80] | 19.28 | 38.61 | 0.875 | 21.20 | 0.7834 | 0.9989 | 3130 | −4918 |
| `d02` | 1.5 D | 0.2 | [60, 80] | 19.35 | 38.88 | 0.875 | 21.30 | 0.7830 | 0.9988 | 3326 | −2137 |
| `d10` | 1.5 D | 1.0 | [60, 80] | 19.12 | 37.23 | 0.875 | 21.06 | 0.7844 | 0.9996 | 2577 | −10908 |

(R_i 19.06, R_o 23.77 δ_D: the base sits over the lip, 0.54 mm above it.)
Every run: anchored, 0 hits, Tmax ≤ 6.984 (σ_st 6.997) for t ≥ 20 (7.013
at t 1.7: the IC transient, LM20), Z ∈ [−8e-8, 1], pcgIt ≤ 13, eosErr
1.7e-7, divErr 8.5e-7; dt 7.1–7.2e-3 with dtPos binding (dtDiff 9.7e-3,
dtChem 1.2e-2). Cost 32.6 ms/step at 2.46M cells, 67.5 at 5.0M = 13.3–13.5
ns/cell-step (S8's 12.7M: 13.5) -- linear in the cells; 80 τ = 11.2k steps
= 6 min.

- **Settling**: the inner cone and the base are steady from t ~15; the
  OUTER tip rises at ~3 δ_D/τ, overshoots (190 at t 35), dips (168 at 48)
  and settles within 0.5% by t ~125 (+0.015%/τ at 160) in the 1.5 D box,
  phase-locked to a box-scale lateral mass-flux swing (mdotLat −5e4 → +1e4
  → ±3e3) that ends with the laterals quiescent (uLatMax 0.02, net inflow
  +500: buoyant entrainment). LM22.
- **K10 (the lateral clearance)**: inner cone, base, rBase, preFrac and
  qRim move ≤ 0.1% from 1.5 D to 2 D; the outer tip is 3.7% shorter over
  [140, 160] (5.0% over [120, 160]) in the 2 D box, which is itself NOT
  settled (a lateral circulation, net inflow 8–17k, swings its tip ±3%
  with a ~45 τ period). Above the plan's 3% gate; the final case keeps 1.5
  D and quotes the outer tip with this bias (masterplan S9 STATUS; LM21).
- **Room draft** (runs started from the same IC stay phase-locked, so their
  t 60–80 difference is the draft's): 0.2 vs 0.4 S_L outer +0.7%, inner
  +0.4%; 1.0 S_L outer −3.6%, inner −0.9% (a strong draft lifts more air
  into the products); qRim falls with the draft (3326 / 3130 / 2577).
- **Projection**: the inner cone scales with D at fixed U (2 × 19.3 = ~39
  δ_th = ~24 mm at D 10 mm), the outer with the jet's volume flux (Roper's
  L_f ~ Q: 4 × 39.9 = ~160 δ_th = ~99 mm) -- the plan's 62 mm (100 δ_th) box
  is too short; S10 uses 200 δ_th (124 mm; masterplan S10 command set).

Scale and burner: `--height 0` takes the real scale from the .json (rim h ×
si deltaD; `--delta-d MM` when the run has no si record); `--crop-lo auto`
crops below the rim top (`--bands 0`: the low-Mach run has no sink bands);
`--burner round` draws a steel tube with the json's real ID/OD (its top at
the rim, the dark bore), an air collar with two air holes, a cast base, the
brass gas inlet and hose; the proxy lights carry the power-weighted mix of
the two colours. Test frame (half-scale scout A, t 80, real scale, 1080p,
96 samples, 4.6 s on gfx1100): `runs/s9/render/g1/f0004.png`.

## Low-Mach LES (`--physics lowmach --les`, `--sgs`, `--tfles`; masterplan_multiphysics M4)

LES and the thickened flame on the low-Mach solvers, structured (`gpu/lowmach`,
`runner_lowmach`) and FV (`DeviceFvLowMach`, `runner_fvlowmach`) together, each the
other's oracle. Same flags, rules and labels as the compressible flame ("LES species and
the thickened flame"): `--les` needs an explicit `--sgs` and `--nz >= 8` (a 2D polyMesh:
exit 3), sigma is rejected at nz 1, `--efficiency charlette` needs `--les`, F x
`--cells-per-delta` >= 4, the IC is the oracle thickened by F. Off = the M3 kernels bit
for bit (`--les`/`--tfles` absent: no new launch, no new instantiation runs).

Model (one expression per term, both solvers; their own kernel instantiations):

```
mu_t     = rho_c (C Delta)^2 rate(g)          per cell, per RK stage (dns/sgs.hpp: Smagorinsky |S|, sigma D_sigma)
           Delta^2 = (hx hy hz)^(2/3) (hx hy at nz 1); FV: FvIncompSetup::delta2 (V^(2/3), area in 2D)
           g: structured = central differences of the cell-centred MAC velocity (kLmCellVelocity),
              an end cell's missing neighbour itself; FV = the least-squares gradient of (rho u)/rho
momentum mu + mu_t in the stress (structured: the cell mu array, so edges average it; FV: face mean)
T, Y_k   rho D + mu_t,f / Sc_t on INTERIOR faces, Sc_t = Pr_t = kLesPrt 0.9 (Le_t = 1: T + (q/cp) Y_F
           keeps one face formula); boundary faces (inflow, open, wall) molecular
TFLES    F E rho D + (1 - Omega) mu_t,f / Sc_t there, omega (E/F) (fuel and heat release); gpu/tfles'
           F, E, Omega frozen over a step from its input state; FV: the same sensor on dns/fvles' face walk
dt       structured: coef h^2 rho_min / (mu(T_max) max(4/3, FE_max/Pr) + mu_t,max max(4/3, 1/Sc_t));
         FV and --dt-rule cell: the cell's own mu, mu_t, F E in the per-cell rate. Both read the eddy
         fields of the CURRENT state (refreshEddy), so a restart is the uninterrupted run byte for byte.
```

Pairs (`dns/lmmodules`, one decision each, docs/fvincomp.md "Low-Mach combinations"): LES is
live with C0, C2, C2b (one Sc_t for both reactants: Z(Y) stays linear), gravity and open
laterals. Refused by name: TFLES with `--equivalence` (F keeps one S_L and E wrinkles one
(S_L, delta_L): the C2 burner's outer Burke-Schumann sheet is a diffusion flame thickening
does not preserve, and the sensor's normalisation is the C0 manifold's), TFLES with
`--laterals open` (the sensor's max filter wraps y and z periodic), LES and TFLES with `--rim`
(no wall model for mu_t at the staircase; the sensor reads the solid). `--turb`: "Low-Mach
turbulence" (M5).

Stats: both runners append `mutMax` (LES) and `FEmax`, `Ebar` (TFLES) after `dtPos`.

Measured (gfx1201, simjob, 2026-11-03; `test_gpu_lowmach_les`, ctest `gpu_lowmach_les`,
`gpu_lowmach_les_smag`, `gpu_lm_matrix`; planar flame sigma 5, beta 8, Pr 0.72, u_in 0.98,
S_L, delta_th means over [10, 20]):

| gate | case | measured | relation |
|---|---|---|---|
| off | `tools/fv/comp_bit_identity.sh` vs faad136 (33 cases); 16 low-Mach routes vs the pre-M4 binary (1D/3D, C2, bunsen 2D/3D, C2 burner, open + gravity, rim, frames; FV box FP32/FP64, 3D, C2, burner, open + gravity, rim) | every artefact | byte-identical (0 failures) |
| L1 laminar limit | 64 x 4 (x 4), 200 steps, both solvers, FV both precisions: TFLES F 1 uniform and dynamic; LES sigma on a planar 3D flame (rank-1 grad u: D_sigma 0); LES Smagorinsky on a uniform inert stream (grad u 0) | every state field of the DNS run | bitwise |
| X1 TFLES F 2, FV vs structured | 128 x 4 box, burning front, dt 1e-3, 400 steps | uniform: FV32 1.0e-7, FV64 2.5e-8 vs structured(FP32), FV32 vs FV64 9.4e-8; dynamic: 1.8e-7, 2.3e-8, own 1.8e-7; TFLES moved rho 4.6e-2 | rounding, <= 3 x own |
| X2 Smagorinsky Cs 0.17, FV vs structured | 1D flame, 2D device level, h 0.25, 0.125, 0.0625, dt 0.03 h^2, t 0.25 | mu_t of one state: 1.2e-2, 3.1e-3, 7.7e-4 of max mu_t (order 1.96, 1.99); rho after the march 1.0e-6, 7.2e-8, 2.4e-8 (the DNS gap 1.7e-8, 1.5e-8, 2.5e-8: rounding reached at h/4) | scheme gap (cell velocity: MAC face mean vs (rho u)/rho), order 2 |
| restart | TFLES dynamic 1D; Smagorinsky + TFLES + Charlette 3D (nz 8); sigma 3D slot Bunsen (nz 8); 4 cells, a checkpoint at 0.5, stop and resume | stats (and profiles) | byte-identical |
| S_L, structured 3D planar (ny = nz = 8) | 4, 6, 8 cells | DNS 1.0006462, 0.9925245, 0.9948435; sigma 1.35: the same 7 digits (mutMax 0: bitwise); Smagorinsky 0.17: 1.0008215, 0.9933300, 0.9952609 (+1.8e-4, +8.1e-4, +4.2e-4), mutMax 4.0e-3, 1.9e-3, 1.1e-3 (h^2, of mu_u 0.72) | the LES shift < 1e-3, not monotone in h (the DNS S_L is not either: no order claimed) |
| same, FV (FP32) | 4, 6 cells (8: not measured, 295 s) | DNS 1.0004300, 0.9925304; Smagorinsky 1.0006183, 0.9933655: the shift +1.9e-4, +8.4e-4 (structured +1.8e-4, +8.1e-4); sigma = DNS to 7 digits, mutMax 9.4e-9 (FP32 rows not bitwise equal on the Morton mesh: D_sigma of the noise) | the solvers' LES shifts agree to 3e-5 |
| S_L, TFLES F 2, 1D | 4, 6, 8 cells | structured uniform 0.9955989, 0.9981457, 0.9991839 (1 - S_L order 2.1, 2.2: F cells per thickened flame converge to the oracle), dth 3.21, 3.20, 3.20 (2 x DNS); dynamic 0.9948831, 0.9974743, 0.9985314; FV uniform 0.9955994, 0.9981518, 0.9991838; FV dynamic 0.9948844, 0.9974810 (8: not measured) | FV - structured <= 6.7e-6 (FV32's own spread, the L-table's) |
| 3D slot Bunsen LES (structured) | the default slot burner (G2a's C0, W 12, U 5), nz 8, 4 cells, t [20, 30] | DNS xTip 44.3041, SLflank 0.98289, mFlank 0.99598, ST 1.02351; sigma 1.35: identical (mutMax 0: the z-extruded slot is exactly 2D); Smagorinsky 0.17: xTip 44.2444 (-0.13%), SLflank 0.98432, mFlank 0.99733, ST 1.02483, mutMax 7.1e-3; 1.3 vs 1.5 ms/step | sanity: a laminar slot is no LES test (sigma is 0 on it, Smagorinsky diffuses it by 0.1%) |

Not measured: a turbulent low-Mach LES (no `--turb` on low-Mach), the round-nozzle 3D
Bunsen under LES (its 144 x 144 cross-section is past a 295 s job at 4 cells), LES vs DNS
statistics of a turbulent flame (none exists on these solvers yet).

## Low-Mach turbulence (`--physics lowmach --turb`; masterplan_multiphysics M5)

`--turb U [--turb-length L --seed S]` on the low-Mach PLANAR flame (`--nz >= 8`), both
solvers, the compressible flags' rules (`flameTurbSetup`: l_t >= 4 cells, L_y, L_z >= 6
l_t; the Mach bound is void at M 0). The ONE generator (`dns/turb_ic`, `flameTurbSpec`'s
box), read where each solver keeps its velocity:

```
structured IC   every MAC face m += rho_f w(x_f) u'_a(x_f): turbBoxAt (the FFT box at the
                face lattice: the face's axis unshifted, the others at the cell centre), column
                i mod nbx, rho_f the two cells' mean (inflow face P0/(R Tu), outflow the last
                cell's), w = turbWindow (addTurbulence's)
structured rows the inflow face per row j + ny k: m = mIn + rho_in u'(-X, (j+1/2) h, (k+1/2) h),
                v at (0, j h, (k+1/2) h), w at (0, (j+1/2) h, k h) (the tangential edges'
                positions; TurbSweep, FP64 host, X = sum u_in dt), uploaded before every step
                (DeviceLowMach::setInflowTurb) and read by the Turb instantiation of kLmMomT:
                the inflow edge's tangential velocity is the row (vl = 2 v_in - v_r) instead of
                the antimirror (u_t = 0); off = the kernel bit for bit
FV IC           rho u += rho w u' at the centroids (dns/fvturb fvLowMachAddTurbulence)
FV rows         the xlo faces' bU += du, rUb += rho_b du (rho_b = P0/(R T_b)) at the face
                centres (fvLowMachTurbRows; DeviceFvLowMach::setInflow), the make() tables
                elsewhere
```

Pairs (`dns/lmmodules` `Turb`, axis `inflow`): live with C0, C2b, LES (smag, sigma), TFLES
(uniform, dynamic, Charlette); refused by name with `--laterals open` (the box is periodic in
y, z), the rim and gravity (the burner's: `--turb` with `--mode bunsen` on low-Mach is exit 2),
C2 (`--sl-law onestep`: no turbulent number). Also refused: `--hold-flame`, `--checkpoint-dt` /
`--restart` / `--wall-limit` with `--turb` (the loop carries the sweep in its checkpoint
fields, but no restart byte gate was run: issues.md M5c3).

Verification (gfx1201, 2026-11-03; `tests/test_gpu_fv_turb.cpp lowmach`, ctest `gpu_lm_turb`
70 s; `gpu_lm_matrix` the Turb cells):

| gate | case | measured | relation |
|---|---|---|---|
| L1 zero amplitude | the CLI's planar box, 4 cells, 24^2, 40 steps | structured: the Turb instantiation with rows (mIn, 0, 0) every step; FV FP32 and FP64: u' 0 rows and the re-set make() tables | byte-identical to the solvers without |
| L2 structured vs FV | the turbulent planar flame (IC + rows), same dt, t 0.4, 4 and 8 cells/delta_th (24^2, 48^2) | max \|rho_FV - rho_MAC\| 7.70e-4, 1.98e-4 (order 1.96); the laminar march's gap 2.8e-8, 2.4e-8 (rounding) | scheme gap (MAC vs collocated velocity, staggered vs centred rows), order 2 |
| matrix | `gpu_lm_matrix`: the Turb cells on box3d, sine3d (xlo rows from a small box, u' 0.1) | (a) rows of u' 0 = the cell without Turb bitwise; (b) uniform Y / Z == 1, mass and fuel tallies; (c) rerun; (d) numbering | every generic gate |

Physics number (`tools/chem/turb_fv.sh lm`, the compressible table's case at M 0):

| cells/delta_th | S_c MAC | S_c FV (FP32) | FV - MAC | laminar S_c MAC / FV (same window, 8^2) |
|---|---|---|---|---|
| 4 (48^2) | 1.04828 | 1.04898 | +6.7e-4 | 1.00848 / 1.00829 |
| 6 (72^2) | 1.05635 | 1.05671 | +3.4e-4 | 0.99657 / 0.99657 |
| 8 (96^2) | 1.05781 | 1.05809 | +2.6e-4 | 0.99925 / 0.99925 |

S_T/S_L 1.039, 1.060, 1.059 (MAC; FV 1.040, 1.060, 1.059) over the IC's decay (u'/S_L 4 at t
0, Re_t 17.8, t in [0.5, 2]); the solvers' gap shrinks (order 1.7, 0.9) and sits under the
grid change of S_c itself (8e-3, 1.5e-3). The compressible box at M 0.02 reads S_c 1.052
(structured) at 8 cells: the M 0.02 start transient (its laminar S_c is 1.020 in the same
window).

The pairs at 4 cells/delta_th (48^2), the same window:

| cell | S_c MAC | S_c FV | FV - MAC |
|---|---|---|---|
| C0 + turb (DNS) | 1.04828 | 1.04898 | +6.7e-4 |
| + LES smag 0.17 | 1.05955 | 1.05008 | -9.0e-3: the Smagorinsky mu_t scheme gap (LF9), larger under turbulence |
| + LES sigma 1.35 | 1.05458 | 1.04878 | -5.5e-3 (sigma's gap, the same cell-velocity definitions) |
| + TFLES F 2 (dynamic) | 1.16021 | 1.16107 | +7.4e-4 (laminar TFLES MAC 1.10592 in this window) |
| + TFLES F 2 + sigma + Charlette | 1.16237 | 1.16119 | -1.0e-3 |
| C2b phi 1.3 + turb | 1.04023 | 1.04078 | +5.3e-4 |

## Real Bunsen (masterplan real_bunsen S10, 2026-11-01)

The masterplan's S10 command set verbatim (frozen binary `build/s10/dns`,
`tools/chem/chain.sh`, 12 simjobs 976 ≤ wall ≤ 1434 s each, ~4.6 h): rich
CH4/air φ 1.3 (C2b GRI-3.0 S_L(φ)), D 10 mm tube with a 1 mm isothermal lip
standing on the inflow face, Poiseuille bulk 1.503 m/s, room draft 0.4 S_L,
Sutherland, Re_D 944.5, real g (Fr 23.0), open laterals (box 3.03 D),
low-Mach, 840 × 196 × 196 = 32.3 M cells, h 0.155 mm, tt 260 τ = 147.5 ms;
outputs `runs/flames/real2/` (89 GB, 121 volume frames t 230–260).

| (SI above the rim top) | t 100–260 | t 230–260 (frames) |
|---|---|---|
| inner cone tip | 30.30 ± 0.07 mm | 30.27 ± 0.03 mm |
| outer cone tip (`xTipOut`) | 71.9 mm, range 61.2–88.3 | 71.3 mm, range 69.1–75.5 |

- Bounded throughout: yHits 0, Tmax ≤ 6.999 (σ_st 6.997), pcgIt 8–10,
  burnRatio 0.999 ± 0.01 (flicker), preFrac 0.77 ± 0.01, base on the lip
  (xRimBase constant).
- The inner cone is 30.3 mm: 0.97 of the ideal (D/2)√(U² − 1) = 51.9 δ_th =
  32.1 mm (the S9 projection's 24 mm assumed the half-scale aspect).
- The OUTER cone FLICKERS (buoyant, the Fr 23 plume): dominant period ~67 τ
  = 38 ms (26 Hz) from t ≥ 60 — only ~3 periods in the record, a rough
  number (the plume law 0.5 √(g/D) gives ~16 Hz for the 10 mm exit). Mean
  72 mm vs the S9 projection's 99 mm (Roper scaling from the half-scale
  scout); it carries the 1.5 D box's −4 ± 1% bias (LM21).
- Render: `flame_room.py` two cones at real scale (`--height 0 --crop-lo
  auto --bands 0`), 121 frames, orbit 40°: `runs/flames/real2/room_real2.mp4`.

## Tests

| test | kind | content |
|---|---|---|
| `flame1d` | host | oracle convergence, √A similarity, profile, vs Cantera; (e) ln Λ(σ) smooth over σ 2..10 step 0.02, constant and Sutherland (\|d2\| < 0.02; the old bracket: ≥ 0.69); (f) σ 6.5 / 6.88 Sutherland; (g) shipped Λ bit for bit, rateFactor ≡ 1 = empty; (h) the cold cutoff (σ 2 n-converged, β 2 σ 1.5 an error); (i) model C2 numbers, two-reactant √A, φ → ∞ limit; (j) model C2b: nodes = csv rows, C(jet) = 1, every csv row uncorrected + C2b vs GRI-3.0 (nodes 1e-9, held-out < 5e-3), issue 73's sign, T_ad within 17%, ln C monotone + C1, FP32 = FP64 to 1e-4, lean/rich branches meet at φ 1, premixedEigen2At(jet) bit for bit |
| `flame_config` | host | grid, IC, dry-run label/Λ, every rejection by name; `--equivalence`: C2 dry-run (Λ2, Y_F,rem, Z_st, σ_st, S_L(φ) table), label, IC Z = 1, every rejection by name; C2b: law default (26 nodes) / onestep, jet Λ2 and table bit for bit, C2b dry-run + label, `--mix-phi` 0.8 / 1.2 (jet's Λ2, GRI ratio, inflow, IC Z = Z_mix, leftover reactant), rejections (`--stoich`/`--ox-y` under gri30, `--sl-law`/`--mix-phi` without `--equivalence`, `--mix-phi` range) |
| `gpu_flame_rich1d` | GPU binary | model C2 φ 1.3 vs the one-reactant flame, σ 6.88 Sutherland, 7.5 / 15 cells: \|ΔS\| ≤ 2e-3, \|S(15) − 1\| < 5e-3, Z within 5e-4 (~30 s) |
| `gpu_flame_rich1d_law` | GPU binary | model C2b: jet + `--mix-phi` 0.7 / 1.05 / 1.2 (+ 1.2 onestep), 15 cells: \|S/r − 1\| < 5e-3, \|(S/S_jet)/r − 1\| < 3e-3, Z within 5e-4 of Z(φ) (~60 s) |
| `gpu_flame_passive` | GPU API | species off/on bitwise mean flow, uniform Y, passive copies, boundedness |
| `gpu_flame_conservation` | GPU API | mass, E + qρY, fuel vs rate tally |
| `fvref_reactor` | host | the 0D reactor's quadrature (G7K15, 40-digit t(1/2), round trip, the cp oracle, Λ) and FvRef's march: bitwise uniform, E + qρY per step, the exact SSP-RK3 recurrence (control Kutta) |
| `gpu_fv_flame_reactor` | GPU API | the 0D reactor G0.1–G0.4 (section "P0: the 0D reactor"), ~37 s |
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
| `restart_config` | host | checkpoint file: every record type bitwise (−0, denormal, NaN), header-only read, a file cut short / another file / a missing one rejected, atomic rewrite, facts + output resume records; case signature order- and spelling-free, excluded flags, differences named; every checkpoint rejection by name; RunOutput stopped, junk appended, resumed == straight byte for byte, a shorter output rejected (1 s) |
| `gpu_restart_flame` | GPU binary | straight vs `--wall-limit 0` + 3 restarts byte for byte: premixed 1D (junk past a checkpoint truncated), `--equivalence` 1D, bunsen 2D with `--pulse` (+ PNG), round 3D nz 32 (volumes), TFLES 3D, candle, `--turb --hold-flame` 3D (slices); a complete tt 1.2 run grown to tt 2; rejections (changed flag / `--emit` exit 3 named, `--tt` exit 2, `--restart` ≠ `--out`, no checkpoint) leave the outputs alone; SIGINT checkpoint + resume (~37 s) |
| `lowmach_config` | host | `--physics lowmach`: box units (Tu 1, no Mach, no sink), the FFT-size rule (nx 330 → 336, `--ny 11` / `--nz 22` named with the nearest sizes), label + dry-run, `--chem-dt` default = kLmChemDtDefault and range, every rejection by name (`--mach`, `--nscbc-*`, `--les`, `--tfles`, `--turb`, modes other than premixed, `flow`/`conv`/`bench --physics lowmach`), the IC (T = 1/ρ, H = σ, p = m²(T_last − T), C2 Z = 1) |
| `gpu_lowmach_passive` | GPU API | 2D/3D hot blob, no reaction, 100 steps: uniform Y (C0 0.5; Y_F 0.25, Y_O 0.5) exactly Y ρ in hi and hi + lo; mass = the FP64 tally to 1e-10; max \|div m − D\|/max \|D\| ≤ 1e-4, EOS ≤ 1e-6; rest bitwise at rest; reruns byte-identical (1 s) |
| `gpu_lowmach_1d` | GPU binary | C0 ladder 7.5 / 15 / 30 (\|S − 1\| ≤ 2e-2 / 4e-3 / 1e-3, order ≥ 1.5, Richardson within 1.5e-3 of the compressible M → 0 limit), flat, SZ ≤ 1e-5, mass = tally; rh: the window-mean p jump within 2e-3, no burnt ramp (1e-5) (~56 s) |
| `gpu_lowmach_rich` | GPU binary | `--equivalence` jet + `--mix-phi` 0.7 / 1.2: \|S/r − 1\| < 5e-3, \|(S/S_jet)/r − 1\| < 3e-3, \|Z − Z_in\| ≤ 1e-5 (~25 s) |
| `gpu_lowmach_chemdt` | GPU binary | the M1 case at 4 cells, `--chem-dt` 0.5 / 1 / 2: S within 1e-4, diffusion binds at the default (~3 s) |
| `gpu_lowmach_sz` | GPU binary | SZ defect low-Mach ≤ 2× the compressible's (15 cells; 8.6e-7 vs 4.7e-3) (~34 s) |
| `gpu_lowmach_restart` | GPU binary | `runLowMach` straight vs `--wall-limit 0` + 3 restarts byte for byte: 1D (+ junk), `--equivalence`, wrinkled 2D, 3D volumes, the low-Mach slot, the partially premixed slot, the round nozzle (volumes); a checkpoint of the other solver is exit 3 naming both (~24 s) |
| `gpu_lowmach_passive` (S5) | GPU API | uniform per-row inflow == the scalar targets bitwise (50 steps); a short profile is an error; findex: pure-x premixed layer / diffusion sheet exactly ±1, wavy ±1 to 1e-6, 1-ulp noise 0 (2D, 3D) |
| `gpu_lowmach_bunsen2d` | GPU binary | G2a low-Mach vs compressible M 0.01/0.02 (`--nscbc-tinflow --nscbc-etain 1000`, M² extrapolated) and foam constants: steady, burn = inflow 1e-3, SLflank/mFlank/burnRatio 1% of M → 0, xTip/sinRatio 3% of M → 0 and 1% of foam, SLflank 1.5% of foam, window Tmax = σ, yHits 0 (~70 s) |
| `gpu_lowmach_extrude` | GPU binary | slot nz 8 == 2D bytewise (frames, every column but mass/massErr/dtDiff), C0 and `--equivalence` (`--cfl 0.1` pins dt) (~3 s) |
| `gpu_lowmach_rich2d` | GPU binary | partially premixed slot (W 4, U 3, pilots 1.5): burnDif > 0.1 burn, preFrac ∈ [0.72, 0.80], outer cone closed and steady, burnRatio 1e-3, Z ∈ [0, 1] ± 1e-5, yHits 0 (~8 s) |
| `gpu_lowmach_round` | GPU binary | round nozzle nz 64 C2 + pilots (4 cells, `--chem-dt 0.5`): anchored, axial cone, T ≤ σ_st, yHits 0, six volume channels, findex ∈ [−1, 1] (~2 s) |
| `lowmach_config` (S5) | host | bunsen + lowmach: G2a grid 360 × 288, dt every step, dry-run lines, rows (m per face, m t = u), IC H = σ; `--equivalence` burner: air coflow, pilots at σ, 7-smooth ny/nx, label (air, pilots, flame index), Z = J + P on rows, no products/air overlap, IC Z ∈ [0, 1]; round IC reactants ≥ 0, T ≤ σ_st; rejections (`--equivalence` compressible bunsen, `--pulse`, `--mix-phi`, round nz ≠ ny); no-pilot warning |
| `lowmach_config` (S6) | host | `--laterals open`: slot y only, round y and z, same box, label/dry-run (periodic text unchanged), json far field; `--reaction off` inert + `--profile-dt` on the burner, the inert IC = rows extruded; rejections by name (`--laterals` compressible / planar flame, `--reaction` and `--profile-dt` off the low-Mach burner), `--laterals sideways` a CLI error |
| `gpu_lowmach_passive` (S6) | GPU API | species tally 1e-10 (every case); open faces: passive Y bitwise, mass/species tallies, rerun; quiescent open box u = p = 0 over 1000 steps (2D, 3D); fuel box + fuel jet under AIR: tallies to 1e-6 of the throughput, entrainment carries the ambient; backflow vortex: KE never grows at β 1, grows 4.4× at β 0 (~5 s) |
| `gpu_lowmach_bickley` | GPU binary | inert planar jet Re 60, open: u_c^-3 slope 3%, u_c ~ x^-1/3 ± 0.02, b^1.5 5%, Qj³ 6%, dQj/dx ~ x^-2/3 ± 0.05, J ± 2%, J + P 1%, mass balance; clearance halved (u_c 0.5%, b 1%, J 1.5%); periodic confines (Q constant, J spread > 5%) (~130 s) |
| `gpu_lowmach_openrich` | GPU binary | partially premixed slot, open clearance 6, tt 200: outer cone closed + steady, LONGER than periodic's 80.09, laterals net outflow, burnRatio 1e-3, preFrac, Z, hits, Tmax (~42 s) |
| `gpu_lowmach_restart` (S6) | GPU binary | + the open partially premixed slot, the inert open jet (jet profiles), the open round nozzle |
| `lowmach_config` (S7) | host | `--gravity`: absent / 0 / value, label and dry-run (no gravity text at 0; Fr, Ri, u_ff), lmRealGravity 0.0241, the periodic-laterals warning, rejections (compressible burner, planar flame, < 0, > 100), `--quiescent` stays the candle's |
| `gpu_lowmach_hydro` | GPU API | gravity: quiescent isothermal box (2D periodic, 2D open, 3D open) u = p = 0 bitwise over 1000 steps; stratified column g vs g 0 (\|dm\| 1e-4 of free fall, p − p(0) = the discrete hydrostatic sum 1e-4); hot blob rises (centroid, momentum vs buoyancy impulse), stays at g 0 (~4 s) |
| `gpu_lowmach_plume` | GPU API | laminar plane plume: steady, exponents u_c 1/5, b 2/5, θ_c −3/5 ± 0.03, buoyancy flux 3%, amplitudes vs the Boussinesq similarity solution 3/5/5%, laterals entrain (~6 s) |
| `gpu_lowmach_candle` | GPU API | the compressible candle's defaults on the low-Mach solver: steady, Z ∈ [0, 1], tip per burned fuel within 5% of the compressible D10 variant (~22 s) |
| `gpu_lowmach_restart` (S7) | GPU binary | + the open slot and the open round nozzle under `--gravity 0.5`; a gravity checkpoint without / with another `--gravity` is exit 3 naming it |
| `lowmach_config` (S8) | host | `--rim`: lip / rim top snapped (30 cells) / T_w / room draft / no pilot, nx = tube + length, `lmTubeSdf` exact distances, the staircase count, `lmPoiseuille` (mean 1, discrete Laplacian constant to 1e-11 incl. the wall rows, curvature 12/W² to 1e-3), rows (Poiseuille, closed wall, draft of air), label + dry-run, the IC (tube 1/T_w, ρY 0, faces 0), `lmWallHeat` on a slab; rejections by name (compressible burner, planar flame, `--tube-length`/`--rim-temp` without `--rim`, pilots/coflow-temp/slot-edge with it, lip < 4 cells, tube < 8 cells, T_w > 3) |
| `gpu_lowmach_channel` | GPU API | the developed channel between two lips: profile 1e-4, v 1e-5, dp/dx = μ D2u 1e-3 and 12 μU/W² 1%, wall shear balance 1e-3 and 6 μU/W 1%, closed faces 0 exactly, mass tally (~23 s) |
| `gpu_lowmach_rim_cond` | GPU API | the isothermal wall: the linear conduction profile is the fixed point (2e-6), qRim = λΔT/L and = the inflow face's conduction (2e-6), T_w 1 and 1.5 (~2.5 s) |
| `gpu_lowmach_rim_imperm` | GPU API | 3D round tube, hot jet, open, gravity: closed faces 0, the tube unchanged, passive Y bitwise, mass/species tallies, PCG ≤ 15 warm, rerun bytes (~4 s) |
| `gpu_lowmach_rim_anchor` | GPU binary | 2D rich slot on the cold rim (M1 mapping, W 14, U 6, lip 1.4, 4 cells, `--chem-dt 0.5`, tt 40), T_w 1 and 1.5: anchored, flat, over the lip, pcgIt ≤ 15, yHits 0, qRim > 0 (~45 s) |
| `gpu_lowmach_restart` (S8) | GPU binary | + the 2D rich slot on its rim, the round tube open under gravity (PCG warm starts checkpointed) |
| `lowmach_config` (S9) | host | `--dt-rule` defaults (cell on the burner, bounds on the planar flame), explicit values, the dry-run text (planar unchanged), rejection off the low-Mach flame, an unknown value a CLI error; `--si-delta-d` / `--si-sl` carried with the case unchanged, the SI dry-run block and label item (absent without them), rejections by name (one without the other, 0, planar flame, compressible burner) |
| `gpu_burner_round` | GPU binary | the real-burner integration smoke: round + rim + open + real g + C2b + the cell rule + SI, nz 64, tt 2: key stats finite in every row, dtPos last, dt ≤ dtPos, pcgIt ∈ [1, 15], yHits 0, Z ∈ [−1e-5, 1 + 1e-5], six volume channels (5 frames), findex ∈ [−1, 1], json rim + si, the last frame's `lmPremixedWeight` split = the row's burnPre/burnDif (6 digits), restart byte rule once (~11 s) |
| `flame_room_selftest` | host (python) | `flame_room_lib`: the lmPremixedWeight port (closed form bit for bit, clamps, monotone, w(ξ) + w(−ξ) = 1), em_pre + em_dif = the single grid, both ≥ 0 and 0 where ω ≤ 0, colours deterministic + chemi_rgb reproduced bit for bit + hues (inner blue-green, outer violet-blue, equal-power outer dimmer), proxy power = Σ gain × grid |
| `flame_room_split` | host (python, fixture of `gpu_burner_round`) | the port against the solver: the last frame's split = burnPre / burnDif to the CSV's 6 digits (77 without the files) |
| `gpu_lowmach_extrude`, `gpu_lowmach_bickley`, `gpu_lowmach_round`, `gpu_lowmach_rim_anchor` (S9) | GPU binary | extrude pins `--dt-rule bounds` (the cell rule's 3D diffusion sum differs from 2D by construction), bickley too (an inert isothermal jet; the cell rule took it 132 → 231 s of its 300 s timeout); round and anchor dropped `--chem-dt 0.5` (the default is bounded). Every other low-Mach test passes unchanged under the new burner default (G2a reproduces its printed digits) |

## Chamber (`--mode chamber`)

A closed 2D combustion chamber, model C0 as above, at M_flame 0.02
(inflated; the label says so). Files: `dns/chamber.{hpp,cpp}` (geometry,
regions, IC, isentropic nozzle), `gpu/runner_chamber.{hpp,hip}`, the walled
instantiation of `kSpeciesStage` (`SpeciesSetup::solid`), validation in
`dns/config/flame_modes.cpp` (`finishChamber`) and `flame_scope.cpp` (`chamberFlagScope`), tool
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

**A choking chamber** (`--shock hll`, `docs/shock.md`). Under `none` an isentropic
inverse that chokes is refused; under `hll` it is a warning and the throat is
captured. Long box, sigma 10, 4 cells: `--expansion 10 --downstream 80 --inject-ramp 4
--coflow 6`. At inject 16 the jet is underexpanded and the row passes M 2.0, exit 0
until the blow-off guard: choked for 5.13 flame times from t 26.9, p0/pa mean 5.73
past t 40 against 5.2 from ṁ at the burnt temperature, a Mach disk near 65 columns
past the exit (axis M 2.56 to 0.94). `MexitMax` reads the exit column only, and
the sonic line sits past it: the lip's stair contracts the jet, so at inject 8
p0Noz/pAmb 2.5 stands at exit Mach 0.82 with static p/pa 1.56 at the exit column and
`MJet` 1.38 in the ambient. No run reached a settled state: each ends on the
blow-off guard, exit 8, with half the fuel leaving through the nozzle (inject 12, t
27.4; 16, t 54.0; 20, t 20.1; a wider holder at 16, t 24.4). issues.md S17, S18.

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

## FV path (`--solver fv`, `--polymesh DIR --bc FILE`)

The premixed flame of model C0 on any FaceMesh. Spec, verification table and the
gaps against the structured flame: `docs/fvcomp.md` "Species". Same model, same units,
same IC, same stats columns. Face value restated on an unstructured mesh with a
virtual upwind difference, identical on a uniform mesh. No NSCBC, no sink: fresh `Far`,
burnt `Outflow`. Rotated 30 deg: S_L 5e-8 off planar; sheared 30 deg: order 2.
Plain FP32 at M 0.01 scatters S_L ~1e-3 on BOTH solvers: quote FV in FP64 or
`--compensated`.

`--gamma G` (`--polymesh` premixed C0 only; masterplan_combnozzle R7): one constant γ for fresh
AND burnt gas, calorically perfect, frozen; Tu = 1/(γM²), q = cp(γ)(σ − 1)Tu, Λ γ-free (the
oracle has no γ). Label `calorically perfect, gamma G for fresh AND burnt gas, frozen`; `.json`
`"gamma"`; absent or 1.4 = the 1.4 bytes. Refused by name: structured and low-Mach flames (`is
the compressible FV polyMesh routes'`), `--solver fv` (`needs --polymesh`), `--equivalence`,
`--les`, `--tfles`, `--mode` other than premixed. Gated in "P1" above (G1.6, G1.7).

LES species and TFLES (`--les --sgs`, `--tfles`, `--tfles-sensor`, `--efficiency`; this
section's "LES species and the thickened flame") on either route, premixed C0:
`docs/fvcomp.md` "Species LES and TFLES". Same equations, constants, label, IC; the eddy
and thickened face terms are the `Eddy` / `Thick` algebra on faces, the sensor's max filter
walks faces (on the uniform box: the structured filter, Omega 8e-7 off it in FP32), no
open-row mu_t copy (the FV gradient reads the boundary face). FV vs structured on one
periodic 3D box: rounding over 300 steps, LES, TFLES and both. `--turb` is not built there.

`--mode diffusion` (model C1) on any FaceMesh: `docs/fvcomp.md` "Diffusion flame":
one limiter shared per face, the jet's row per FACE (a fixed face, Fick and Fourier
through it: foam's G3a inflow, not the pinned first cell). Candle, chamber, bunsen,
wick exit 2 by name.

`--equivalence` (models C2, C2b, `--sl-law`, `--stoich`, `--ox-y`, `--mix-phi`) on either
compressible route: `docs/fvcomp.md` "Air-methane". The two-reactant face kernels of C1 with
the premix carried in (every far / inflow patch: the fresh premix, every outflow: the burnt
one; a `y` in a polyMesh's `--bc` is exit 2), C2b's `ln C(xi)` in the cell pass and the
chemistry dt (`kChemCellPass2Phi`'s and `ChemRate2Phi`'s expressions), the same oracle, IC,
label and `zMin`, `zMax` columns. LES (`--les --sgs`, 3D) works with it there; TFLES is refused
by name (premixed C0 only), as on the structured flame. Same mesh, same dt: rounding vs
`kSpeciesStage2` (R1); 1D FP64 S_L +3.0e-3 at 15 cells, Z exact in FP64.

`--physics lowmach` on either route: the low-Mach stage on faces, `docs/fvincomp.md`
"Low-Mach": same units, same IC on the box, same stats columns, the 1D flame equal to
the structured low-Mach flame on the same mesh. Built: planar C0, `--equivalence` (C2,
C2b: the two-reactant stage on faces, rounding vs `kLmCellPass<true, Phi>`), and the 2D
slot `--mode bunsen` (per-face inflow rows, `lowmachBunsenIC` / `lmBunsenPoint`, `--dt-rule
cell`, the cone columns on a pixel = cell plane), and on it `--laterals open` (the `open`
face kind: this page's "Low-Mach open laterals" face rule, one per face by its own flux's
sign), `--gravity` (`addBuoyancyLm` as a face force, hydrostatics in p) and `--rim` (the
staircase tube's cells removed, its faces an isothermal `rim` Wall patch: the structured
staircase on the same cells is the oracle), with the open and rim stats columns and the rim's
anchoring guards: `docs/fvincomp.md` "Open faces, gravity, rim". 3D and the round nozzle,
checkpoints exit 2 by name.

## Not built

Le ≠ 1 (`--lewis` unregistered until an oracle exists), curvilinear,
compensated, RANS species (asserted), LES at nz == 1, LES/TFLES species for
the two-reactant and walled kernels (diffusion, candle, chamber; the FV path takes LES with
`--equivalence`, docs/fvcomp.md "Air-methane"), a separate
Sc_t, thickened viscosity, SGS/thickening inside the NSCBC rows (the LODI
transverse flux ignores μ_t), Colin's efficiency (needs a global Re_t), a
dynamic Charlette β, nz > 1 for diffusion/candle/chamber, a cold-coflow
or piloted 3D Bunsen (only the burnt coflow is run), `--hold-flame` for the
Bunsen, a GPU span-mean kernel (the 3D Bunsen averages on the host),
forced turbulence (the inflow box decays in transit), an inertial-range
spectrum, an
immersed burner lip for the compressible bunsen and the jet (the low-Mach
burner has its tube rim: S8; the compressible walled kernels exist:
premixed for the chamber, two-reactant for the wick candle; neither takes
LES/TFLES), nz > 1 for the wick candle, open y faces for the compressible bunsen,
the jet and the candle (the low-Mach burner has them: S6), an open
(pressure) inflow bottom, gravity in the premixed modes and on the
compressible burner (the low-Mach burner has it: S7), unequal W, a
counterflow/strained-flame oracle, the low-Mach solver beyond the premixed
1D/2D/3D flame and the burner with open laterals, gravity and the
staircase tube rim (a sub-cell immersed wall for the round tube,
`--turb` (LES and TFLES are built: "Low-Mach LES"), point-implicit chemistry, a low-Mach `--mode candle`
(the S7 cross-check drives the solver through its API), the rim on the
planar flame or the compressible burner);
the planar flame's x-hi backflow still carries its own scalars;
`--physics incompressible` rejected), multi-step chemistry, soot beyond the one-equation
appearance model (nucleation/growth/PAH, thermophoresis, radiation), a fuel
rate tied to the wick heat flux, a melting wax pool, dns-ui (its editor covers
`flow` and `conv` only). Chamber: isothermal (cooled) walls, a
total-pressure inlet, non-premixed injection, a moving/2D-curved nozzle wall
beyond the polygon, 3D.
