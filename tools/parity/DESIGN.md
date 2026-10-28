# dns vs OpenFOAM parity suite

Accuracy and cost, both solvers generated from one case spec. A cost number
compares like with like: dns' explicit compressible time march against
simpleFoam's steady incompressible solve is different equations in a
different mode, hence the tiers.

## Two tiers, one case spec

Every case is defined ONCE (`cases.toml`): geometry, Re, Mach, J, box, BCs,
start-up, observables, resolution ladder per solver, wall-time budget. Both
solvers are generated from that spec; nothing is hand-matched per run.

| tier | question | dns | OpenFOAM |
|---|---|---|---|
| A, same method | is dns' solver efficient at what it does? | compressible, time-accurate | `rhoPimpleFoam`, laminar, same Mach/Re/Pr/gamma, same IC + ramp, time-accurate |
| B, same question | which tool for this job? | its best mode for the case | its best mode: `simpleFoam` (steady, incompressible) |
| Bc, same question at the dns Mach | is the dns steady answer right? | tier A runs re-scored | `rhoSimpleFoam`, laminar, steady, tier A's gas and Mach |

Tier B scores dns' compressibility error against it: the job is the
low-speed answer. Where the compressibility increment is not small against
the comparison (rotor: tip Mach 0.41, Bc - B = +4.2% CT, +1.5% CQ), dns is
scored against Bc instead (`dns_scored_in = "Bc"`) and tier B stays as the
M -> 0 leg, listed for information.

## U convection (the parity rule)

A foam parity number names its U scheme. The reference is central
(`u_scheme = "linear"`: `Gauss linear` in tier A, `bounded Gauss linear` in
B/Bc), as in CLAUDE.md "OpenFOAM parity" -- linearUpwind is a numerical
diffusion dns does not have (rotor: linearUpwind put foam's L1 CT 22% above
its own reference, central 5%). Every report row names its scheme; the 2D
cases (cyl40, cyl100, sect) are still configured `linearUpwind`, so their rows
are not parity numbers. Tier Bc's energy convection (`bounded Gauss
linearUpwind grad(e)`, `bounded Gauss linear` for Ekp) is recorded in the
report; the rule covers U.

The laplacian / snGrad is unlimited (`limited = False`, `Gauss linear
corrected`, nNonOrthogonalCorrectors 1-2) on the rotor and section branches:
`limited corrected 0.33` cost +9.3% Cn on the 7° ellipse section at 53°
non-orthogonality (it does not refine away) and 1.2–2.3% |CT| on rotor2.

## Cases (cheap on purpose)

| id | what | observables | tiers |
|---|---|---|---|
| cyl40 | 2D cylinder Re 40, M 0.2, steady | cd, Lw/D | A, B |
| cyl100 | 2D cylinder Re 100, M 0.2, shedding | St, mean cd, cl rms | A |
| sect | 2D prop1c section, AoA 3.78, Re_c 93, M 0.318 | cd, cl | A, B |
| rotor | 3D single-blade rotor, J 0.63, Re_D 220, tip M 0.41, MRF | CT, CQ | A, B, Bc (dns scored in Bc) |
| rotor2 | 3D two-blade rotor (chord 0.40/0.30, pitch 1.0), same flow; the dns ladder that fits 300 s per run; parity verdict (`parity_bands`) | CT, CQ | A, B, Bc (dns scored in Bc) |

Identical-mesh cases live outside the suite, one script each, because their
mesh is dns's own (`--emit mesh` -> `dnsmesh.py`), never snappy-generated:
`sph1e4.py` (revolved wedge, steady RANS and laminar, simpleFoam),
`ocyl1e4.py` (2D cylinder O-grid, URANS shedding Re 1e4, rhoPimpleFoam;
masterplan_verify W12) and `othermal.py` (2D cylinder O-grid, Re 40,
isothermal wall heat transfer, rhoSimpleFoam; below).

## Thermal parity (`othermal.py`)

Laminar steady Re 40 cylinder, M 0.2, O-grid 128 x 64 far 20 D, identical
cells. dns `--wall isothermal --twall R --visc sutherland --tinf 300
--prandtl 0.690335 --steady 1e-5 --steady-scheme implicit` (exact argv in
the script); OpenFOAM rhoSimpleFoam, `sutherland` transport, U/e/Ekp
`bounded Gauss linear`, laplacian `Gauss linear corrected` (unlimited).
Observables: Nu (area mean, film, and at 0/45/90/135/180 deg), cd, T1.
Three traps, each commented in the script:

- **Viscous work.** rhoSimpleFoam/rhoPimpleFoam (v2606) carry no
  div(tau.u) in the energy equation. The case adds it as a coded fvOption
  (`scalarCodedSource` on e, `fvc::div((-devRhoReff) & U)`). Without it
  foam's Nu reads 4.4-4.6% high at Ec 0.16 (Tw/Tinf 1.1) and 0.55% at Ec
  0.016 (2.0). `--stock` omits the source to reproduce that gap.
- **Pr.** OpenFOAM's sutherland kappa is modified Eucken, mu cp/0.690335;
  dns must run `--prandtl 0.690335` (the script refuses another Pr with
  `--visc suth`). At 0.72 the comparison shows a false 4.3% kappa gap.
- **Sign.** dns qw is positive into the body, foam wallHeatFlux into the
  fluid: q_hot = -qw = +wallHeatFlux.

Verdict (exp_thermal.md, reproduced end to end by the script in
impl_batch.md): equivalent. Nu -0.34% (Tw/Tinf 1.1) and -0.14% (2.0), cd
+0.43% / +0.14%. The dns 2x-grid change is 0.12% in Nu. Open (issues.md 29):
the M 0.5 adiabatic recovery reference and RANS Prt heat transfer; neither is
codified.

`rotor` is NOT prop1c. prop1c (chord 0.109 D, t/c 0.55) needs c/h >= 10,
i.e. D/h >= 111 and ~1 h per point (L3: below c/h 10 the result depends on
where the body sits on the grid). The suite rotor is a wider blade
(propeller.py `--root-chord 0.30 --tip-chord 0.20 --thickness 0.35`) so
c/h >= 10 and t/h >= 4 hold at D/h ~48. prop1c stays a one-off
showcase, not a suite member.

`rotor` itself cannot be refined under the 300 s run cap (L1 at D/h_fine
59.5 is 131 s; L2/L3 ~550/2100 s), so `rotor2` carries the dns ladder: two
wider blades (c/h 10.3, t/h 3.6 at D/h_fine,x 39) put L1/L2/L3 at 35 / 113 /
260 s. Its L1 grid is the ctest pin `gpu_mrf_rotor2` (1 D/U; STL fixture
`tests/data/rotor2.stl`, written by `geom.write_rotor_stl` from the case's
`blade`). `report.py` prints the dns ladder, its Richardson value and the
PARITY / NOT AT PARITY verdict for any rotor case that sets `parity_bands`.

Attribution (W6): `split_dns.py` splits a rotor's CT / CQ into hub vs blade
and pressure / convective / viscous, for dns from `--emit surface` (the
immersed ghost-wall tally per interface face) and for foam from per-face
force fields it writes on a COPY of a converged case (`foam` mode runs
OpenFOAM: call it inside gpurun; `--reuse` never does). Both sides use the
same geometric sort and one hub slack T per case; JSONs saved under
`runs/parity/split/<case>/{dns_Lk,foam_<tier>_L3}.json` are rendered by
`report.py` with each part's share of the gap to the scored tier.

## Resolution ladders

Three levels per solver per case, refinement ratio sqrt 2, sized so the
finest ROUTINE level finishes in <= 10 min on this machine; whole suite <= 2 h.
Levels are fixed in `cases.toml` after a calibration pass (dry-run +
short timed run), not guessed.

Reference per case: one grid-converged run, cached in
`runs/parity/ref/`, regenerated only on request, excluded from the budget:

- tier A: rhoPimpleFoam fine body-fitted mesh (+ Richardson from its ladder);
- tier B: simpleFoam fine (+ Richardson);
- cyl40/cyl100 additionally checked against literature, M -> 0 corrected.

## One stop rule, applied by the harness to both

Neither solver's own stop is used: dns `--steady` (a compressible
local-time-stepping pseudo-time march with its own jitter-floor stop) and
foam's residual tolerances certify different things, and the suite's dns
runs are time-accurate marches. Each run goes to a generous cap; the
harness then reads the monitored history and defines

- steady: t_settle = first time after which |q - q_final| <= 1e-3 |q_final|
  for every observable, for good;
- periodic (cyl100): the same on St and the windowed cd mean, window = 3
  shedding periods.

Cost = wall time from process start to t_settle. Setup (meshing: snappy,
blockMesh, decompose; dns geometry/tables) is reported separately and also
summed.

Wall clock: the harness wrapper timestamps every stdout line
(monotonic); dns progress lines and foam `Time =` lines map simulation
time to wall time.

## Cost accounting

- wall time (setup, solve to t_settle, total);
- hardware: dns = RX 7900 XTX; foam = 6 MPI ranks on i5-14600KF P-cores
  (plus one 12-rank point per case to show scaling);
- GPU energy: `rocm-smi` board power sampled at 1 Hz, integrated. CPU energy
  needs RAPL (`/sys/class/powercap/intel-rapl:0/energy_uj`, root-only
  here): reported when readable, else marked n/a, never estimated;
- throughput: cell-updates/s (dns), cell-iterations/s (foam), for context
  only.

Exclusivity: ONE timed job on the machine at a time. dns takes the GPU lock;
foam takes the same lock (CPU contention skews both).

## Outputs

- `runs/parity/<tier>/<case>/<solver>_L<k>/` -- run dir, log with timestamps,
  `result.json` (observables, t_settle, costs, binary/OpenFOAM version, foam
  U scheme); a retired run is `<run>_stale` with a `stale` reason: listed,
  never scored;
- `runs/parity/report.md` + `report.csv`: per case, error vs reference
  against cost for both solvers (the error-vs-cost curve), and cost to reach
  1 % and 3 % error read off it;
- `runs/parity/split/<case>/*.json`: rotor CT/CQ splits (`split_dns.py`).

## Driver

`tools/parity/suite.py`:

```
suite.py list
suite.py calibrate CASE            # dry-run + short timing, proposes levels
suite.py run CASE --tier A|B|Bc [--solver dns|foam] [--level k] [--ref] [--timeout S]
suite.py run rotor --tier B --solver foam --level 2 --zone-r 0.555 --tag zr0.555
suite.py adopt CASE --tier A --level k --from DIR --dry FILE   # score an existing dns run
suite.py stale CASE --tier T --run NAME --reason TEXT           # retire a run, keep it listed
suite.py report
```

Each `run` prints the exact dns argv / foam case path, dry-runs dns first,
and refuses to start if the budget estimate exceeds the case's cap. The
placement the foam box is built around is cached
(`runs/parity/tmp_place/<case>/placement.json`, keyed by the dry-run argv
minus `--ft`), so foam runs on a known case do not need the dns binary.
Under a wrapper that already holds the lock (the session `gpurun` flocks the
same file) set `PARITY_LOCK_HELD=1`: a nested flock on the same file
deadlocks. `--timeout` bounds the solver; a timed-out run is scored on what
it wrote (rc 124). `--zone-r` (rotor, needs `--tag`, lands in
`runs/parity/calib/`) moves the sharp MRF zone while the refinement
cylinders keep the case's `zone_r` (`snappyHexMeshDict.zone`), so the study
varies the zone, not the mesh; the report tabulates it against the same
level at the case's zone.
