# Shock capture (`--shock none|hll|sense`)

`dns flow`, `dns flame --mode premixed|chamber`, `dns bench`, compressible, structured,
uniform Cartesian mesh, `--far nscbc` on `flow`; `dns flow --polymesh` on the FV path, section
"FV". `none` is the default and launches
nothing. Files: `dns/shockflux.hpp` (the face arithmetic, host and device),
`gpu/shock.{hpp,hip}` (`DeviceShock`), `kNscbcCells<..., Super>` in `gpu/comp_nscbc.hip`,
`internal::shockDtEvery`, `shockOpenRow` in `gpu/internal/run_loop.hpp`. Plan:
`plans/masterplan_shockflux.md`. Open items: issues.md "Shock capturing", S1 to S18.

- `none`: the shipped solver, byte for byte.
- `sense`: the sensor runs and reports, D never does. The solver's bytes are unchanged:
  the stats differ only in their shock columns, appended last. Its row is the shipped
  row.
- `hll`: D runs where the sensor fires, and x-hi is the `Super` row.

Every result is labelled with the DNS/LES label and the shock suffix.

## The pass

`--shock` is an additive pass, `kShockFaces`, after the species stage and before the
rhs hook. No hot kernel gains a branch (`faceFlux`, `faceFluxOf`, `faceFluxCurv`,
`kCompStage`, `kCompStageCurv`, `kSpeciesStage`, `kSpeciesStage2`, `kCompSink`,
`kWallRhs`, `kSideRhs`, `species_face.hpp` are not edited; `kNscbcCells` only under
`Super`). A face's flux becomes

    F_KG + sigma_f (F_HLL - F_KG)

added as a difference D: the centred HLL form, one divide, exactly 0 on U_L = U_R and an
exact mirror. Both faces of a cell are evaluated, each cell writes its own index, and
`fp contract(off)` makes hi of c and lo of c + 1 the same bits (conservation is the
check). Species writes go through `stageAdd` like rho's, so uniform Y stays uniform.

**Sensor**, per cell, pressure only, stride 2, no gate:

    v = max_a |p[+2] - 2 p + p[-2]| / (p[+2] + 2 p + p[-2])

byte ramp 0 at th1 0.02, 255 at th2 0.06, `sigma_f` = max of the 4 bytes around the face.
Pressure sees a flame as M^2, velocity as M, density as 1: a live flame reads 8.98e-4
(sigma 5) and 2.14e-3 (sigma 10) at M 0.02, 4 cells, against 0.02. A solid at +-1 or
+-2 reads the centre on that side (a slanted wall loses no face). `sigma_f` is 0 on a
face with a solid side and on every face touching an open row.

**HLL** is first order with Davis speeds. MC and Einfeldt raised an aligned standing
shock's transverse noise x20 to x400.

**Species** ride the donor stream on D's own sign, `max(D,0) Y_L + min(D,0) Y_R`: bounded
(piston flip Y in [0, 1] exactly). `D Y_f(sign m_KG)` is unbounded, Y -0.068 on the host.

**Dt.** Default cadence until a cell is sensed, then every step, sticky (a 48^3 diagonal
Sod diverges at 16 by step 9). `--dt-recompute` given: the user's. `nDim cfl <= 1.2`
(`kShockCflAxisSum`): 1.2 stable, 1.4 diverges.

**Off where it is not built.** `--ogrid`, `--stretch`, `--focal`, `--rans`, `--steady`,
`--precond`, `--rotor-*`, `--thin-seal`, `--penalize`, `--nscbc-strip`, `--pot`, `--solver fv`,
low-Mach, `--body floor`, flame modes other than premixed and chamber, `--equivalence`, and
under `hll` `--les`, `--tfles`, `--turb` on a flame: refused by name. `--polymesh`, `--bc`: only
`dns flow --physics compressible`, the FV route below. `--mach` above `kShockMachMax` on a flame
trips the sensor: refused.

## FV (`--physics compressible --polymesh DIR --bc FILE`)

`docs/nozzle.md`. The structured pass does not transfer: F is materialized, so one face pass
writes F_KG + sigma_f (F_HLL - F_KG) + (1 - sigma_f) eps |u.S| D3 in place (`kFvShockFace`), one
write per face, conservative by construction. `shockFaceDelta` verbatim. `force` (test only, bytes
255): rhoCentralFoam Kurganov + upwind to rounding on every cell. The sensor is `shockJump` along mesh
lines (`dns/fvshock`), whole lines only: a blocked read as the centre fires on a smooth wall.
Thresholds `kShockTh1`, `kShockTh2` unchanged, once per step on the step's input. D3 runs on a silent
run: FV `hll` equals `none` at `--shock-d4 0` only, and is never labelled DNS. `--cfl` <= 1.2.
`sense`: `none` bytes, its two columns appended. Bridged with the structured sensor: on a uniform
periodic box the bytes are equal (`gpu_fv_cross` X6: 0 off; FP64 vs FvRef 3.3e-15).

## The outflow under `--shock hll`

`docs/nscbc.md` "Outflow past M 1 and in backflow". `Super` is x-hi only, its own
instantiation, on from the case's `hll` alone (no flag, no Config knob on the CLI):

- M <= 0.8 and un >= 0: the shipped arithmetic, no added operation; quiet `hll` keeps
  its bytes (gate 32).
- 0.8 < M < 1: L1 blends from `Lin` to the upwind amplitude; M >= 1: L1 = Lout1, no
  target, no transverse term. A supersonic row cannot feel the back pressure: at
  pInf/p 0.8 and 1.25 Super is unchanged to all printed digits, the shipped row 51 to
  484 x worse (gate 33). Nozzle exit plane at N 200, 400, 800: M 5.80e-5, 2.74e-5,
  1.06e-5, order 1.23, against the extrapolated row's 1.70e-3, 8.61e-4, 4.22e-4 (gate
  34). The shipped row is off by 1.85 to 2.66 in M.
- un < 0: L2 to L4 relax to ambient at `etaLat`. The shipped row's downwind difference
  runs a cell away at rate |un|/h. Rig, V 20, M 0.4 along the row: shipped T/Tu 0.011,
  `dt 0 at t=11.82 step 12480`; blend T/Tu 0.720 over 10 transits. Long box, inject 8:
  shipped row rho 72, T/Tu 0.012, failed at t 10.76; blend rho 1.21, T/Tu 1.05, exit 0.

A moving shock through the row with the sink on leaves 1.1e-2 (Ms 2) and 1.6e-2 (Ms 3)
of its jump behind (gate 35).

**Guard.** At the dt cadence, reading the conserved state: x-lo, the laterals and
inflow on x-hi at |u_n|/c >= 1 are exit 7, `supersonic flow on the open row at t=...
(|u_n|/c = ...): the NSCBC boundary is subsonic only (docs/shock.md)`. x-hi flowing out
may be supersonic.

**Ungated.** `Super` and its backflow blend are built on the prof, hydro and turb
instantiations of `kNscbcCells`, and are gated on the uniform-target instantiation
only: bunsen, candle and `--turb` are refused under `--shock`, so no case reaches
them. Each needs its own gate when `--shock` opens those modes.

## The chamber (`--mode chamber`)

Long box, DNS + `--shock hll`, sigma 10, M_flame 0.02 (inflated), 4 cells,
`--expansion 10 --downstream 80 --inject-ramp 4 --coflow 6`, 524 x 152. Stats columns
under `--shock`, all after the shipped ones: `shockCells shockMax rhoMin pMin openMach
shockCav shockCavMax shockLip shockCavRest MexitMax shockCavFar`, `shockBurnCells` (or
`shockBurnMax` under `sense`), `mdotLo mdotHi MThroat pThroat p0Throat MJet`.

| inject | end | exit | Me | MexitMax | MJet | p0Noz/pAmb | longest choked | sensed peak |
|---|---|---|---|---|---|---|---|---|
| 8 | t 20 | 0 | 0.791 | 0.824 | 1.39 | 2.54 | 0 | 0.28% |
| 12 | t 27.4 | 8 | 0.836 | 0.953 | 1.75 | 3.31 | 0 | 0.57% |
| 16 | t 34 | 0 | 0.872 | 1.048 | 2.45 | 5.33 | 5.13 from t 26.9 | 1.79% |
| 16 | t 54.0 | 8 | | 1.055 | 2.80 | 7.34 | 5.13 | 2.1% |
| 20 | t 20 | 0 | 0.871 | 1.055 | 2.27 | 4.71 | 1.93, 2.07 run on | 1.09% |

Choked = p0Noz/pAmb > 1.893 and `MexitMax` >= 0.98. **The sonic line sits past the
geometric throat** (the exit column, i 187, 20 cells): the lip's stair contracts the jet.
At inject 8, t 16.2, the throat reads M 0.820, isentropic p0/pa 2.37, static p/pa 1.56,
so the exit is above ambient and the jet expands past it, `MJet` 1.38; the axis is M
0.78 at the exit, 1 about 5 columns on. p0Noz/pAmb 2.5 at exit Mach 0.82 is that, not a
fault.

At inject 16, t >= 40: p0/pa mean 5.73, range 4.83 to 7.34, against 5.2 from mdot at the
burnt temperature. The jet is underexpanded: axis M 2.56, a Mach disk ~65 columns past
the exit (M to 0.94, p/pa 0.29 to 2.75), a second cell to M 2.11. Sensed cells mean
1.24% past t 40. The row carries M 2.0.

**Limits, measured.** No run settled: each ends on the blow-off guard with half the fuel
leaving through the nozzle (`dns: blow-off: 0.502 of the injected fuel leaves through
the nozzle at t=20.14 (lower --inject-speed or raise --holder-height)`); `--holder-height`
7 at inject 16 was earlier, t 24.4. Gate 29's burning and window clauses are UNMET: from
t ~13 unburnt gas burns on the contraction's last stairs, which fire the sensor on
their own pressure pattern (S17). D on a staircase is a real limit; the fix is a
body-fitted wall. The lip itself fires in subsonic flow, 4.07e-2 against 0.02, 2 cells,
mean sigma 0.305 past t 20, so a default chamber is not quiet: Me 0.929%, pCh/pAmb
0.290%, burn/inj 0.040%, pCh rms 7.74%, frequency 0.039% off `none` (gate 12b).

Mass budget: domain mass change against the integrals of inflow, injection and outflow,
-9.4e-4 of the injected mass (inject 8), -7.6e-4 (16, t 34). The end-face flux is `1.5
row - 0.5 inner`, what a one-sided LODI row telescopes to.

## Measured against the plan's gates

Device, FP32, DNS. `docs/validation.md` "Shock capture" has the ladder. Bench, 256^3:
none 20.92, sense 21.56, hll 22.64 ms/step (+8.2%), 1.46, 1.55, 1.49 GiB. Default
chamber 0.47, 0.52, 0.56 ms/step. A rig with 9.38% of the cells sensed is UNMET at
<= +40%: inviscid +82% to +131% (10.3 to 10.5 ms/step off, rerun noise), viscous +42.7%
(20.20 to 28.83): 8.6 ms/step, from `kShockFaces` over 3 stages. The plan's levers do not
reach it: the mask reads neighbours' bytes, so it cannot share the sensor's launch, and
the zero-sensed skip acts on a quiet run only.

## Proposed CLAUDE.md invariants

CLAUDE.md is not edited by this work; these are its proposed text.

- **`--shock` is an additive pass after the species stage, before the rhs hook; no hot
  kernel gains a branch.** Off = no launch; a silent sensor = the solver bit for bit,
  default flags included. `Super` is the one exception, its own instantiation.
- **`sigma_f` is 0 on a face with a solid side and on faces touching an open row.** The
  sensor reads no solid cell and no wrap. A blocked side reads the centre value: skipping
  the axis left 48 of 714 near-wall faces unsensed on a 30 deg staircase. A sharp
  staircase lip reads a true second difference and fires in subsonic flow (4.07e-2
  against 0.02, 2 cells, gate 12b); a default chamber is therefore not quiet.
- **The sensor reads pressure only, stride 2, no gate.** Pressure sees a flame as M^2,
  velocity as M, density as 1. Live flame 2.14e-3 at 4 cells, M 0.02, sigma 10, against
  0.02.
- **First order, Davis speeds**: MC and Einfeldt both raise an aligned standing shock's
  transverse noise x20 to x400.
- **Species correction is the donor stream on D's own sign.** `D Y_f(sign m_KG)` is
  unbounded (Y -0.068).
- **Sticky dt cadence**: default until a cell is sensed, then every step.
- **`nDim cfl <= 1.2`**: 1.2 stable, 1.4 diverges.
- **Label rule**: every `--shock` result carries the DNS/LES label and the shock suffix.
- **The open-row guard is exit 7 at the dt cadence**, on x-lo, the laterals and inflow
  on x-hi. x-hi flowing out may pass M 1 under `hll` (`kNscbcCells<..., Super>`):
  L1 = Lout1 at M >= 1, blended from M 0.8, no operation added below it.
- **`Super` carries the exception to "no outflow backflow model"**: at un < 0 the x-hi
  L2 to L4 are downwind differences, and sustained backflow runs a cell away at |un|/h
  (rho 72, T/Tu 0.012, issues.md S15). Its blend relaxes them to ambient at `etaLat`,
  one sided. The note at `comp_nscbc.hip:349` ("Deliberately NO backflow model", measured
  to 4 x freestream swirl) holds for the shipped row on a brief inflow, and is the
  chamber's failure on a sustained one.
- **`Super` and its backflow blend are ungated on the prof, hydro and turb
  instantiations**; they are built, unreachable from the CLI, and each needs its own
  gate before `--shock` opens that mode.
- **"Exact to the bit through every stage, hook and band" has an exception**: `rhoY == Y
  rho` on an open row is not bitwise under a strong transverse jump, with the blend or
  without, sink, shock and mu off too: max |Y - 1/2| 1.8e-6 to 2.2e-6 (`kSpeciesOpenFix`,
  issues.md S16, open). A solid side has its own, S10.
- **A chamber's `MexitMax` is not its sonic line**: the stair's vena contracta puts that
  past the geometric throat; read `MJet`, `pThroat`, `p0Throat`.
- **Mass through an end face is `1.5 row - 0.5 inner`** on a one-sided LODI row: the
  row's own rho u leaves the budget 1.8 x off.
