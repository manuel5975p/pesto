# pimpleFoam on the FV path (stage 3a: static mesh; 3b: solid-body mesh motion; 3c: cyclicAMI; 3d: the M17 case)

`dns flow --polymesh CASE` runs OpenFOAM v2606 **pimpleFoam** when the case's
`system/controlDict` says `application pimpleFoam;` (simpleFoam otherwise:
`docs/fv.md`, unchanged byte for byte). Same reader, same FaceMesh in SOURCE
order, same kernels, same `gpu/la` solves, same outputs; FP32 by default,
`--fv-precision fp64` the parity mode. Phase 3 of "propeller models on the
M17 airframe" (1 actuator discs, 2 MRF: `docs/fv.md`). Stage 3a is the
transient algorithm on a mesh that does not move; stage 3b ("Moving mesh"
below) is pimpleFoam's dynamic-mesh branch for ONE rigid rotation of the
WHOLE mesh (`constant/dynamicMeshDict`: `dynamicMotionSolverFvMesh` |
`dynamicMultiMotionSolverFvMesh`, `solidBody` + `rotatingMotion`), no
non-conformal interface. Stage 3c ("cyclicAMI" below) is the `cyclicAMI`
patch pair (`faceAreaWeightAMI`, no transform) in every operator of the
path, static or with a cellZone rotating behind it. Stage 3d ("The M17 case"
below) is the user's sliding-mesh propeller case (`m17_prop5_8000_ami`,
1 227 027 cells, two contra-rotating zones) run through it with ONE
dictionary line changed, compared step by step with pimpleFoam: measured, not
a parity gate at the case's own tolerances (a stated solver-quality gap).
Open points: `issues.md` item 65 of "FaceMesh / FV path".

Dispatch: `drive::runFvFlow` reads the case (`loadFvCase`), then
`FoamCase::pimple.on` picks `gpu::runFvPimple` | `gpu::runFvSimple`
(`gpu/runner_fv`). No new flag: the case IS the OpenFOAM case, its
`application` names the solver, as `simpleFoam` did. `--max-iters N` = at most
N time steps. `--fv-coupled` is simpleFoam's (exit 2).

Label: `[FV path, body-fitted, laminar | RANS kOmegaSST (OpenFOAM v2606), URANS
(time-accurate), FP32 | FP64] incompressible PIMPLE (OpenFOAM v2606 pimpleFoam
algorithm, static mesh | MOVING mesh: solid-body rotation of the whole mesh, N
points, omega W rad/s, correctPhi yes|no; phi written RELATIVE to the mesh,
Euler; nOuterCorrectors …)`.

## Measured (gfx1100, FP64, serial pimpleFoam, tight solves on both sides)

`tests/data/fv/p*` (`tools/fv/make_pimple_cases.py`, `run_foam.sh`,
`make_pimple_test_data.sh`), gated by `gpu_fv_pimple`. Fields: max-norm of
scale (p: of its range) at the first / last reference time. Residuals: every
"Initial residual" of every corrector of every time step, relative, where
above 1e-9.

| case | cells, steps | loop (nOuter, nCorr, nNonOrth) | U | p | phi | residuals | other |
|---|---|---|---|---|---|---|---|
| P1 `pcavity`: IF1a lid cavity, deltaT 0.002, PISO mode, Gauss linear, no relaxation | 576, 100 | 1, 2, 0 | 1.3e-15 / 9.9e-15 | 9.2e-15 / 3.5e-15 | 1.4e-15 / 1.0e-14 | 1.4e-11 (400 values) | Courant number 2.5e-15 |
| P2 `pcavity_skew`: 26.6° cavity, the M17 fvSolution shape (`pFinal { $p; }`, `"(U\|k\|omega)Final" { $U; }`, p 0.7 / pFinal 1, `"U.*"` 0.7, `turbulenceOnFinalIterOnly`, `correctPhi`) and schemes (cellLimited Gauss linear 1 default, linearUpwind, `limited corrected 0.33`) | 576, 100 | 3, 1, 1 | 1.8e-14 / 2.6e-14 | 7.8e-15 / 2.0e-14 | 1.6e-14 / 2.4e-14 | 6.0e-11 (1200) | force.dat, moment.dat 2.2e-13 of column scale, all 100 rows |
| P3 `pchannel`: IF1c graded channel (inletOutlet, fixedValue p, slip), `adjustTimeStep yes; maxCo 2; maxDeltaT 0.03`, `adjustableRunTime 0.5`, U 0.7 / UFinal 1 | 800, 89 | 2, 2, 1 | 8.2e-14 / 1.2e-13 | 2.1e-14 / 4.4e-14 | 2.9e-14 / 4.6e-14 | 2.3e-11 (712) | **deltaT of all 89 steps equal to pimpleFoam's 17 printed digits** (worst relative 0); Courant number 5.2e-15 |
| P3b `pduct`: IF1d 3D duct (freestreamVelocity / freestreamPressure, symmetryPlane, cyclic z), deltaT 0.02 | 512, 60 | 1, 2, 1 | 5.4e-15 / 5.6e-15 | 5.4e-15 / 5.9e-15 | (cyclic: not compared) | 8.2e-10 (360) | |
| P4 `psst_plate`: IF-T plate, kOmegaSST, kqR / omegaWallFunction / nutUSpalding, turbulence on the FINAL corrector only (v2606's default; what the M17 case runs with), `"(k\|omega).*"` 0.7 | 960, 100 | 3, 1, 0 | 7.2e-14 / 1.7e-12 | 3.0e-13 / 2.1e-12 | 3.2e-15 / 5.4e-14 | 1.1e-8 (1100) | k 1.4e-12 / 1.1e-12, omega 3.7e-13 / 1.1e-11, nut 1.8e-10 / 1.6e-9 |
| P4b `psst_skew`: IF-T skewed channel, kOmegaSST, nutk / nutLowRe, inletOutlet k / omega, `turbOnFinalIterOnly no`, k 0.7 / kFinal 1, exactDistance | 600, 100 | 2, 1, 1 | 4.5e-15 / 9.0e-14 | 2.5e-14 / 1.4e-13 | 2.1e-15 / 1.9e-14 | 2.1e-11 (1200) | k 1.8e-13 / 9.1e-13, omega 1.9e-13 / 1.4e-13, nut 8.9e-12 / 5.6e-12 |

**P4's level is the case's, not a difference between the codes.** Serial
pimpleFoam against ITSELF with only U's linear solver swapped (PBiCGStab →
smoothSolver, both to 1e-14) differs at t = 0.5 by U 5.4e-12, p 4.1e-12,
omega 3.6e-11, nut 5.0e-9 (t = 0.1: nut 1.8e-10) — more than dns differs from
it. The Spalding wall function on an impulsively started plate amplifies
rounding (the simpleFoam IF-T plate shows the same nut 1.6e-10). `gpu_fv_pimple`
gates P4's nut at 1e-7 and residuals at 1e-6 for that reason, every other
case at 1e-10 (U, p, phi) / 1e-9 (k, omega, nut) / 1e-8 (residuals).

FP32 (default floors) vs FP64 at the last reference time: `pcavity_skew` U
5.4e-6, p 1.8e-6, phi 5.3e-6; `psst_skew` U 1.1e-6, p 5.3e-5, nut 1.1e-5;
`psst_plate` U 2.5e-5, p 2.4e-4, nut 5.6e-2 — without floors 2.5e-4: the
turbulence floor (1e-7) sets the nut FIELD, as on the simpleFoam path
(`docs/fv.md` "Precision"). Quote field parity in FP64. The FP32 floors were
measured on SIMPLE tails; no transient floor study was made (issues.md 65).
Two runs are byte-identical (deltaT, Courant numbers, residuals: `pchannel`,
`psst_plate`). Wall time on these launch-bound meshes: 7–16 ms per time step
(pimpleFoam serial: 0.8–3.5 ms): nothing to quote before a large mesh.

## The time step (pimpleFoam.C, UEqn.H, pEqn.H of v2606)

`gpu/fv_pimple.hip` `SimpleImpl<R>::timeStep(deltaT)` composes the simpleFoam
path's stage methods (`assembleU`, `systemU`, `solveU`, `prepP`, `assembleP`,
`solve`, `fluxes`, `correctU`, `rasCorrect`) — no kernel is forked:

1. `++runTime`: the old-time fields U, U_b, phi, phi_b (RAS: k, omega) are the
   state at the call (`GeometricField::oldTime()` is the value when the time
   index changed; the first `oldTime()` call copies the current field).
2. For corr = 1 … nOuterCorrectors (`pimple.loop()`); final = (corr ==
   nOuterCorrectors) = `mesh.data().isFinalIteration()`:
   - `storePrevIterFields`: p, when `relaxationFactors/fields` has a factor
     for `p` (`solutionControl::storePrevIter` asks `relaxField("p")`).
   - **UEqn.H**: `fvm::ddt(U) + fvm::div(phi, U) + turbulence->divDevReff(U)`,
     `UEqn.relax()`, `solve(UEqn == -fvc::grad(p))`. Euler: diagonal
     `rDeltaT V`, source `(rDeltaT U_old) V` (`kURow<R, Multi, Ddt>`; `Ddt =
     false` is the simpleFoam kernel bit for bit). `fvMatrix::relax()` and
     `fvMatrix::solve()` take the entries of `U` or, on the final corrector,
     `UFinal`. No factor found: no relaxation (the dominance fix of
     `relax(alpha)` does not run either).
   - For piso = 1 … nCorrectors (`pimple.correct()`), **pEqn.H**: `rAU =
     1/UEqn.A()`, `HbyA = constrainHbyA(rAU UEqn.H(), U, p)`, `phiHbyA =
     flux(HbyA) + interpolate(rAU) fvc::ddtCorr(U, phi)`, `adjustPhi` (no p
     patch fixes the level), then nNonOrthogonalCorrectors + 1 times
     `fvm::laplacian(rAU, p) == fvc::div(phiHbyA)`, `setReference`, solved
     with the `pFinal` entry on the last non-orthogonal iteration of the
     last corrector (`pimple.finalInnerIter()`: EVERY outer corrector's, as
     `finalOnLastPimpleIterOnly` defaults to no) and `p` otherwise; `phi =
     phiHbyA − pEqn.flux()`; `continuityErrs.H` (× deltaT); `p.relax()` with
     the factor of `p` | `pFinal` (final corrector) when one is found — a
     factor 1 still evaluates `prev + 1 (p − prev)`, OpenFOAM's expression;
     `U = HbyA − rAU grad p`, `U.correctBoundaryConditions()`.
   - `turbulence->correct()` when `pimple.turbCorr()`: on the final corrector,
     or on every corrector with `turbOnFinalIterOnly no`. kOmegaSST is
     `kOmegaSSTBase::correct()` as on the simpleFoam path plus `fvm::ddt(omega)`
     and `fvm::ddt(k)` (their own pass `kSDdt` on the model's diagonal / source
     additions before each solve); `omega` | `omegaFinal`, `k` | `kFinal`
     solver and relaxation entries by `isFinalIteration()`.

**`fvc::ddtCorr(U, phi)`** (`EulerDdtScheme::fvcDdtPhiCorr`, `kDdtCorrF`,
`kDdtCorrB`): `phiCorr = phi_old − (Sf · interpolate(U_old))`, `ddtCorr =
(c rDeltaT) phiCorr`, `c = 1 − min(|phiCorr| / (|phi_old| + SMALL), 1)`
(`ddtScheme::fvcDdtPhiCoeff` with `ddtPhiCoeff` −1, the default of `default
Euler;`; `experimentalDdtCorr` off). Boundary faces: c = 0 where the U patch
`fixesValue()` (fixedValue and every mixed type; cyclicAMI too, stage 3c),
the same expression on the patch values elsewhere (zeroGradient, slip,
symmetry), rAU's patch value the cell's.

**The clock** (`dns/fvpimple` `FoamClock`, host, FP64, pure): pimpleFoam.C's
`CourantNo.H` + `setInitialDeltaT.H` before AND after `turbulence->validate()`,
then per step `CourantNo.H`, `setDeltaT.H` (`f = maxCo/(Co + SMALL)`, `deltaT =
min(min(min(f, 1 + 0.1 f), 1.2) deltaT, maxDeltaT)`), `Time::adjustDeltaT`
(`adjustableRunTime`: deltaT bent to land on the next write time, within ×2 /
÷5), `++runTime`, `Time::run()` (`t < endTime − 0.5 deltaT`), the write times of
`timeStep | runTime | adjustableRunTime`. Fed pimpleFoam's own printed Courant
numbers it reproduces its printed deltaT digit for digit (`fvcase` P3). The
Courant number is `0.5 max(Σ|phi| / V) deltaT` (`DeviceSimple::courant`: a
cell's internal faces then its boundary faces, block maxima, exact in any
order).

## Reading the case (what differs from simpleFoam)

- `controlDict`: `application pimpleFoam`; `endTime`, `deltaT`, `adjustTimeStep`,
  `maxCo` (default 1), `maxDeltaT` (GREAT), `writeControl timeStep | runTime |
  adjustableRunTime` + `writeInterval` (required), `timePrecision`, `timeFormat
  general`. Start from time 0 only.
- `fvSchemes`: `ddtSchemes { default Euler; }`. `bounded` convection is
  rejected (the steady `−Sp(div phi)` term).
- `fvSolution` `PIMPLE`: `nOuterCorrectors` 1–3, `nCorrectors` 1–2,
  `nNonOrthogonalCorrectors` 0–2 (the ranges the gates exercise: (1, 2, 0),
  (3, 1, 1), (2, 2, 1), (1, 2, 1), (3, 1, 0), (2, 1, 1); other combinations
  inside the ranges run the same loops, ungated), `turbOnFinalIterOnly`,
  `pRefCell`, `pRefValue`. Solver entries are read by the names the loop looks
  up: `UFinal`, `pFinal` (+ `kFinal`, `omegaFinal`) always; `U` only with
  nOuterCorrectors > 1; `p` only when a non-final inner iteration exists;
  `k`, `omega` only with `turbOnFinalIterOnly no`. Relaxation: `fields { p;
  pFinal; }`, `equations { U; UFinal; k; kFinal; omega; omegaFinal; }`, literal
  or regex keys.
- **`turbulenceOnFinalIterOnly` is not a v2606 key.** `pimpleControl::read`
  reads `turbOnFinalIterOnly` (default yes); the M17 case's
  `turbulenceOnFinalIterOnly no;` is read by nothing, so its pimpleFoam run
  corrects turbulence on the final outer corrector only. Accepted with a
  dry-run note saying so. `correctPhi` is read by `createDyMControls.H` and
  used only when the mesh changes: noted, inert here.
- An equations factor matching `p` / `pFinal`: inert (pimpleFoam never calls
  `pEqn.relax()`), noted.

Rejected by name (exit 3): every `constant/dynamicMeshDict` but the ones of
"Moving mesh" below and `dynamicFvMesh staticFvMesh` (simpleFoam ignores the
file, as OpenFOAM does); an active MRF zone; any fvOptions source; `RASModel SpalartAllmaras`; ddt schemes other than
`default Euler` (backward, CrankNicolson, localEuler, `Euler <ddtPhiCoeff>`,
per-field entries); `bounded`; `momentumPredictor no`; `consistent yes`
(PIMPLEC: `rAtU = 1/max(1/rAU − H1, 0.1/rAU)`); `ddtCorr no`;
`finalOnLastPimpleIterOnly yes`; PIMPLE `residualControl`; loop counts outside
the ranges; `moveMeshOuterCorrectors yes`; every other PIMPLE key (`SIMPLErho`,
`solveFlow`, …); a `pFinal` field factor without a `p` one (OpenFOAM stops in
`p.relax()`: no previous iteration stored); `default` relaxation factors; field
relaxation of U / k / omega; `writeControl` cpuTime / clockTime; `timeFormat`
other than general; `maxCo` / `maxDeltaT` / `adjustTimeStep yes` on a
simpleFoam case; `cyclicAMI` patches under simpleFoam, and every cyclicAMI
variant but the one of "cyclicAMI" below; `--fv-coupled` (exit 2). `movingWallVelocity` on a static mesh is a fixedValue (OpenFOAM's
`updateCoeffs` acts only when `mesh.moving()`), as on the simpleFoam path.

## Moving mesh (stage 3b: solid-body rotation of the whole mesh)

`constant/dynamicMeshDict` with a dynamic mesh class switches pimpleFoam.C's
dynamic branch on (`mesh.dynamic()`): `Uf`, `fvc::ddtCorr(U, Uf)`, the mesh
update at the first outer corrector, `correctPhi.H`, `fvc::makeRelative` /
`makeAbsolute`, `fvm::ddt` on `V0`, `movingWallVelocity`. This stage moves
ONE rigid body: every cell (no cellZone, or a cellZone holding every cell).

### Measured (gfx1100, FP64, serial pimpleFoam, tight solves; `gpu_fv_pimple`)

`tests/data/fv/m*` (`tools/fv/make_motion_cases.py`, `run_foam.sh`,
`make_pimple_test_data.sh`). Columns as the table above; **phi is the flux
RELATIVE to the mesh motion**, which is what pimpleFoam writes (`phi` after
`fvc::makeRelative`; the absolute flux is `phi + meshPhi`).

| case | cells, steps | loop | U | p | phi | residuals (pcorr included) | motion |
|---|---|---|---|---|---|---|---|
| M1 `mspin`: closed 2D box 16 x 16 rotating about z at omega 2, walls `movingWallVelocity`, fluid at rest in the lab frame (spin-up), deltaT 0.005, 2 rad; `dynamicMotionSolverFvMesh`, flat spelling; `checkMeshCourantNo` | 256, 200 | 1, 2, 0 | 2.1e-15 / 2.9e-15 | 3.9e-14 / 3.8e-14 | 2.1e-15 / 7.2e-15 | 3.6e-10 (1000 values) | points 0 (bitwise), meshPhi 0 (bitwise), Uf 1.5e-15 / 2.8e-15; mesh Courant number 0 (all 200 steps); moment.dat 3.8e-15 |
| M2 `mduct`: 3D duct 10 x 8 x 8 of 26.6° sheared section rotating about an OFF-CENTRE x axis at omega 4, inlet fixedValue / outlet inletOutlet + fixedValue p ON the rotating mesh, walls `movingWallVelocity`; the M17 relaxation set and schemes, `correctPhi yes`, `adjustTimeStep yes; maxCo 0.5; maxDeltaT 0.02`, `adjustableRunTime 0.25`; `solver solidBody; solidBodyCoeffs { cellZone rotor; … }` | 640, 35 | 3, 1, 1 | 3.2e-14 / 3.4e-14 | 4.9e-14 / 2.0e-14 | 2.9e-14 / 5.4e-14 | 2.8e-12 (595) | points 0, meshPhi 0, Uf 3.1e-14 / 3.3e-14; **deltaT of all 35 steps: worst relative 0**; force.dat 1.6e-13, moment.dat 5.8e-13 |
| M3 `msst`: M2's duct, kOmegaSST, kqRWallFunction / omegaWallFunction / nutkWallFunction on the rotating walls (the M17 wall set), the M17 fvSchemes verbatim (cellLimited, linearUpwind, upwind k / omega, `limited corrected 0.33`, meshWave), `maxCo 1`, omega −4; `dynamicMultiMotionSolverFvMesh` in the M17 dictionary's spelling | 640, 33 | 3, 1, 1 | 1.4e-12 / 1.1e-12 | 1.2e-12 / 1.5e-12 | 6.8e-13 / 9.0e-13 | 6.2e-12 (627) | points 0, meshPhi 0, Uf 1.1e-12 / 9.9e-13; k 2.0e-13 / 3.2e-13, omega 1.2e-13 / 2.0e-13, nut 1.4e-12 / 2.9e-12; deltaT worst relative 0 (33 steps); force.dat 3.0e-12, moment.dat 1.6e-12 |

M1 passes 2 rad (0.32 revolution), M2 and M3 2 rad each: more than a quarter
revolution. M1's net wall FORCE is a cancellation to 1e-12 against moments of
1e-2 and is not compared (below 1e-9 of the object's largest column); its
moment is.

**M3's level is the case's solver tolerance, not a difference between the
codes.** Serial pimpleFoam against ITSELF with only the PRESSURE solver
swapped (PCG + DIC → GAMG + GaussSeidel, both to 1e-13, relTol 0) differs at
t = 0.25 / 0.5 by U 2.3e-12 / 2.6e-12, p 1.9e-12 / 3.1e-12, k 2.5e-13 /
7.5e-13, nut 3.6e-12 / 7.3e-12 — more than dns differs from it. The same twin
of M2 gives U 2.9e-14 / 4.0e-14 (dns 3.2e-14 / 3.4e-14). Mechanism, measured
after one step (U 1.7e-13 at the wall cells, p 3.6e-12 absolute on a pressure
of 18): the impulsive start at deltaT 0.0024 makes p ~ U / deltaT, a p solved
to 1e-13 of that leaves `rAU grad p` ~ 2e-13 in U, and nu 1e-4 does not damp
it. (Swapping only the U / k / omega solver leaves the twin at 5e-14: the p
iterates are then the same.) M3 keeps the common gates (1e-10 / 1e-9).

FP32 (default floors) vs FP64 at t = 0.5: `mduct` U 1.8e-6, p 1.9e-6, phi
1.3e-6; `msst` U 1.1e-5, p 1.2e-5, nut 2.2e-5 (without floors 8.7e-6). Two
30-step `msst` runs are byte-identical. Wall time: 7.5 (M1) / 24 (M2) / 26
(M3) ms per step against serial pimpleFoam's 0.7 / 5.4 / 5.5 ms: launch bound
as stage 3a, plus the HOST mesh update (FaceMesh + setup + wall distance
rebuilt and re-uploaded every step: correctness first; see "open" in
issues.md 65).

### The time step with a moving mesh (pimpleFoam.C, correctPhi.H, pEqn.H)

The runner calls `FvMeshMotion::update(t_new, deltaT, …)` (host) after
`++runTime` and hands the result to `timeStep(deltaT, &moved)`, which applies
it where `mesh.controlledUpdate()` stands: at the first outer corrector
(`pimple.firstIter()`; `moveMeshOuterCorrectors yes` is rejected).

1. **`mesh.update()`** (`dynamicMotionSolverFvMesh::update` /
   `dynamicMultiMotionSolverFvMesh::update`, `fvMesh::movePoints`):
   - points(t) = `transformPoints(TR, points0)` with `TR = septernion(-origin)
     * quaternion(axis, omega t) * septernion(origin)` (`rotatingMotion::
     transformation`; angle = `omega_->integrate(0, t)` = `(t − 0) omega`):
     always from the INITIAL points, never an increment. `dns/fvmotion`
     `foamRotatingMotion` restates the septernion / quaternion products and
     `transformPoints`' two tests (`mag(trans) > VSMALL`: `p − trans`;
     `mag(R − I) > SMALL`: `R & p`) operand for operand: the moved points
     equal pimpleFoam's written `polyMesh/points` BIT FOR BIT (`fvcase`,
     `gpu_fv_pimple`).
   - `V0 = V` (`storeOldVol`), then the geometry on the new points through the
     ONE geometry definition: `FaceMesh::fromPolyMesh` → `buildSimpleSetup`
     (S, Cf, C, V, weights, deltaCoeffs, non-orthogonal correction vectors) and
     `buildRasSetup` (the wall distance: `wallDist::movePoints` recomputes
     meshWave at every mesh motion with its default `updateInterval 1`;
     `nearWallDist::correct()` runs in `turbulenceModel::correct()` when
     `mesh.changing()`), re-uploaded. No second geometry code; nothing is
     rotated on the device.
   - `meshPhi = face::sweptVol(oldPoints, newPoints) / deltaT` per face
     (`fvGeometryScheme::setMeshPhi`; `face::sweptVol` sums `triangle::sweptVol`
     over the fan about `face::centre`), internal and boundary: bitwise
     pimpleFoam's `meshPhi` file. Geometric conservation, measured: `|Σ meshPhi
     deltaT − (V − V0)| ≤ 2.8e-15 V` (`fvcase`).
   - `U.correctBoundaryConditions()`: `movingWallVelocity` = `Uwall()` = `Up +
     n (meshPhi_b / (|S| + VSMALL) − n·Up)`, `Up = (face::centre(points) −
     face::centre(oldPoints)) / deltaT`, n and |S| of the new geometry. First
     order in omega deltaT by construction (the chord's direction is the
     mid-step's, the normal the new time's: |Uwall| / (omega r) − 1 = 2.5e-3 at
     omega deltaT 0.01); `Uwall·n = meshPhi_b / |S|` to 2e-16. `face::centre`
     is the face CLASS's centroid (not `primitiveMesh`'s): a formula of
     `Uwall` and `sweptVol`, not mesh geometry.
2. **`correctPhi yes`** (default `mesh.dynamic()`): `phi = mesh.Sf() & Uf()`
   (the face velocity of the previous step on the NEW faces), then
   `CorrectPhi(U, phi, p, rAUf = 1, divU = 0, pimple)`: `correctUphiBCs`
   (the U patches that fix their value evaluate, `phi_b = U_b & Sf`); `pcorr =
   0` with fixedValue 0 where p `fixesValue()` (fixedValue, freestreamPressure)
   and zeroGradient elsewhere; when pcorr needs a reference `makeRelative`,
   `adjustPhi(phi, U, pcorr)`, `makeAbsolute`; nNonOrthogonalCorrectors + 1
   times `fvm::laplacian(1, pcorr) == fvc::div(phi)` with `setReference(0, 0)`
   (cell 0, value 0), solved with the `pcorr` entry and `pcorrFinal` on the last
   one (`pcorr.select(pimple.finalNonOrthogonalIter())`), the correction's
   gradient by the scheme of `grad(pcorr)` (the default); `phi −=
   pcorrEqn.flux()`; `continuityErrs.H` (printed, added to the cumulative
   error); `fvc::makeRelative(phi, U)`. The pcorr equation is the p equation's
   kernels on rAUf = 1 (`ones`: l (1 − 1) + 1 = 1 exactly).
3. `checkMeshCourantNo yes`: `meshCourantNo.H` on `meshPhi` (log only).
4. **UEqn.H**: `fvm::ddt(U)` = diag `rDeltaT V`, source `rDeltaT U_old V0`
   (`mesh.moving()`); `fvm::div(phi, U)` on the RELATIVE flux.
5. **pEqn.H**: `fvc::ddtCorr(U, phi, Uf)` takes the `Uf` overload on a dynamic
   mesh (`EulerDdtScheme::fvcDdtUfCorr`: `phiUf0 = Sf & Uf.oldTime()` on the
   CURRENT Sf, `phiCorr = phiUf0 − dotInterpolate(Sf, U.oldTime())`, coefficient
   `fvcDdtPhiCoeff(U.oldTime(), phiUf0, phiCorr)`); `adjustPhi` between
   `makeRelative` / `makeAbsolute` when p needs a reference; `phi = phiHbyA −
   pEqn.flux()` is ABSOLUTE through `continuityErrs.H`, `p.relax()`, `U`'s
   corrector and `U.correctBoundaryConditions()` (inletOutlet reads its sign
   there); then `fvc::correctUf(Uf, U, phi)` (`Uf = interpolate(U) + n (phi /
   |S| − n·interpolate(U))`) and `fvc::makeRelative(phi, U)`.
6. `turbulence->correct()`: kOmegaSST's `divU = fvc::div(fvc::absolute(phi,
   U))`; `fvm::ddt(k)`, `fvm::ddt(omega)` on `V0`; convection on the relative
   flux. CourantNo.H before the step uses the relative flux.

**Old-time fields.** `Uf.oldTime()` is Uf at the step's start. `U.oldTime()`:
from the second step on the state before the mesh update (`storeOldTimes()`
runs at the head of `mesh.update()`'s `U.correctBoundaryConditions()`); in the
FIRST step no old-time field exists until `fvm::ddt(U)` asks for it, AFTER the
mesh update rewrote the patch values — its boundary values are then the new
ones. Reproduced (`MotionState::steps`); it only shows on patches whose
value depends on the geometry and whose `ddtCouplingCoeff` is not 0 (slip /
symmetry on a rotating mesh: not gated).

### Reading the case

- `constant/dynamicMeshDict`: `dynamicFvMesh dynamicMotionSolverFvMesh`
  with `motionSolver` (or the older `solver`) `solidBody`, its coefficients
  at the top level or in `solidBodyCoeffs { }` (`optionalSubDict`):
  `solidBodyMotionFunction rotatingMotion`, `origin`, `axis`, `omega` there or
  in `rotatingMotionCoeffs { }`, an optional `cellZone` (`none`: ignored);
  `motionSolverLibs` accepted. `dynamicFvMesh dynamicMultiMotionSolverFvMesh`:
  `dynamicMultiMotionSolverFvMeshCoeffs { <name> { solver solidBody; cellZone
  Z; solidBodyCoeffs { solidBodyMotionFunction rotatingMotion;
  rotatingMotionCoeffs { origin …; axis …; omega …; } } } … }` — the M17 case's
  spelling, entries in dictionary order. `omega X;` | `omega constant X;`
  (rad/s). The axis is used as written (`quaternion(axis, angle)` normalises).
- `fvSolution`: `PIMPLE { correctPhi; checkMeshCourantNo; }`; with `correctPhi`
  the solver entries `pcorrFinal` and (nNonOrthogonalCorrectors > 0) `pcorr`
  (M17: `"pcorr.*"`). `fvSchemes`: `grad(pcorr)` or the default.
- `0/U`: `movingWallVelocity` on the walls that move with the mesh.

Rejected by name (exit 3): any other `dynamicFvMesh` (dynamicRefineFvMesh,
dynamicOversetFvMesh, …); any other motion solver (multiSolidBodyMotionSolver,
displacementLaplacian, velocityLaplacian, sixDoFRigidBodyMotion, …); any other
`solidBodyMotionFunction` (linearMotion, oscillatingRotatingMotion,
axisRotationMotion, multiMotion, SDA, tabulated6DoFMotion, …); a Function1
`omega` (table, sine, …); `cellSet`; a regular-expression / group cellZone;
a `cellZone` inside a multi-motion entry's `solidBodyCoeffs`; `updateControl` /
`updateInterval`; `moveMeshOuterCorrectors yes`; a cellZone the mesh lacks
(`No matching cellZones: …`); no `pcorr` solver entry with `correctPhi`; cyclic
patches on a moving mesh; MRF, fvOptions sources and SpalartAllmaras (as 3a);
at Acquire (`FvMeshMotion::make`): **a moving cellZone that borders non-moving
or differently moving cells through ordinary internal faces** (`… borders
non-moving cells through N ordinary internal faces …: a moving zone needs a
cyclicAMI interface or must be the whole mesh` — OpenFOAM runs such a case
and shears the cells); a point shared by two motions, or by moving and
non-moving cells (an interface made with createBaffles: split its points);
on a mesh WITHOUT cyclicAMI patches a motion that does not cover every cell
with one body (disconnected regions moving differently, or a static
remainder); a 2D (`empty`) mesh rotating about another axis than its empty
one.

### Output

Beside stage 3a's files a moving case writes, per time directory, what
pimpleFoam writes: `phi` (RELATIVE to the mesh), `meshPhi`, `Uf`
(surfaceVectorField) and `polyMesh/points` (identical to pimpleFoam's text);
the log gains the `pcorr` solves, their continuity errors and `Mesh Courant
Number mean: … max: …`. `FvState` carries `Uf`, `meshPhi`.

## cyclicAMI (stage 3c)

A `cyclicAMI` patch pair as OpenFOAM v2606 computes and uses it:
`AMIMethod faceAreaWeightAMI`, `transform noOrdering` (no transform), the
defaults `requireMatch true`, `reverseTarget false`, no `lowWeightCorrection`,
`restartUncoveredSourceFace 1` — the M17 spelling. Static, or with cellZones
rotating behind it (the 3b motion; the refusal "a moving zone borders other
cells" stays for ordinary faces). pimpleFoam only.

Label: `… MOVING mesh: solid-body rotation of N cellZone(s) behind cyclicAMI
(faceAreaWeightAMI, recomputed at every mesh update), …`. The log prints
OpenFOAM's `AMI: Patch source|target sum(weights) min: max: average:` lines at
every AMI calculation, and the count of faces with a weight sum below 0.2.

### Measured (gfx1100, FP64, serial pimpleFoam, tight solves; `gpu_fv_pimple`, `fvami`)

**Weights** (`dns/fvami`, `tests/test_fvami.cpp` against dumps of
`tools/fv/foamAmiDump`, a v2606 utility): srcAddress / tgtAddress (the same
lists in the same order), srcMagSf / tgtMagSf, the weight sums, the
normalised weights, `interpolateToSource` / `interpolateToTarget`, and
`cyclicAMIFvPatch`'s weights / delta / deltaCoeffs / nonOrthDeltaCoeffs /
nonOrthCorrectionVectors: **BIT FOR BIT**, 0 differing values in 598 checks:
planar 8 | 12, matching, a partial overlap (2 source / 3 target faces
uncovered), the cylinder (20-gon | 28-gon, rest + 8 positions incl. coinciding
vertices and a tilt that uncovers 16 / 34 faces), the 2D and 3D mixers (rest
+ 6 angles), and the M17 mesh (both pairs, rest + 4 rotor positions, 105
checks; its time-0 `sum(weights)` lines equal the user's log to the 8 printed
digits: AMIFwd source min 0.59878664 max 1.5409241 average 1.0072946, AMIAft
source min 0). No place where accumulation order makes bits differ was found:
the walk, the clipping and the sums are restated in OpenFOAM's order. Bit
identity relies on glibc's `acos` / `cos` (the quad's split diagonal, the 89°
neighbour filter).

**Flow** (`tests/data/fv/a*`, `tools/fv/make_ami_flow_cases.py`). Columns as
the tables above.

| case | cells, steps | loop | U | p | phi | residuals | other |
|---|---|---|---|---|---|---|---|
| A1 `aplanar`: 2D channel split by a planar cyclicAMI, 10 x 8 against 10 x 12 cells, the flow crosses it; deltaT 0.005, the M17 schemes | 200, 100 | 2, 2, 1 | 5.1e-15 / 5.4e-15 | 2.0e-13 / 2.3e-14 | 4.6e-15 / 5.1e-15 | 1.3e-10 (799 values) | Courant number 4.0e-15 |
| A1m `amatch`: 8 against 8, faces match point for point | 160, 100 | 2, 2, 1 | 6.9e-15 / 9.3e-15 | 2.7e-13 / 2.1e-13 | 7.0e-15 / 8.6e-15 | 5.5e-11 (798) | |
| A2 `acyl`: 3D static, cylindrical interface of flat quads (20-gon against 28-gon, non-matching in z; weight sums 1.0037 … 1.0044 before normalisation), non-orthogonal to 42.8°, the M17 relaxation set | 703, 100 | 3, 1, 1 | 1.5e-14 / 2.3e-14 | 1.1e-13 / 7.7e-14 | 6.7e-15 / 1.0e-14 | 1.2e-9 (1498) | |
| A3 `amix2d`: 2D mixer, the rotor disc (paddle: `movingWallVelocity`) at omega 4 behind the circular cyclicAMI in a static box; laminar, the M17 relaxation set, `correctPhi yes`, `adjustTimeStep` (maxCo 0.5), 1.8 rad | 528, 55 | 3, 1, 1 | 2.0e-13 / 4.4e-13 | 5.3e-13 / 4.6e-13 | 4.1e-14 / 4.6e-14 | 1.4e-11 (770, pcorr included) | points 0, meshPhi 0 (bitwise), Uf 2.0e-13 / 4.4e-13; **deltaT of all 55 steps: worst relative 0**; force.dat, moment.dat 5.8e-10 |
| A4 `amix3d`: the mixer in 3D (rotor 2, stator 3 cells in z), kOmegaSST, kqR / omegaWallFunction / nutkWallFunction on every wall, meshWave across the interface, the M17 fvSchemes verbatim, `dynamicMultiMotionSolverFvMesh` in the M17 spelling, omega −4, maxCo 1 | 1344, 52 | 3, 1, 1 | 3.6e-11 / 9.2e-11 | 1.7e-11 / 2.7e-11 | 2.2e-12 / 1.1e-11 | 1.3e-9 (986) | Uf 1.8e-11 / 7.3e-11; k 7.2e-13 / 4.7e-12, omega 1.7e-13 / 1.1e-12, nut 5.9e-11 / 2.2e-10; deltaT worst relative 0 (52 steps); forces 1.1e-9 |
| A1p `apartial` (stage 3d): static 2D, a partial overlap, 2 source / 3 target faces WITHOUT overlap, laminar; pimpleFoam run with `FOAM_SIGFPE=false` | | | 5.7e-14 / 1.2e-13 | 2.7e-14 / 2.0e-13 | | 1.65e-11 (799) | |
| A5 `amixtilt` (stage 3d): A4 with the rotor axis tilted to (0.3 0 1): weight sums fall 1 → 0 on both sides, faces without overlap from step 49 of 74; kOmegaSST, `correctPhi`; `FOAM_SIGFPE=false` | 74 steps | | 4.4e-14 / 2.4e-13 | 2.3e-13 / 9.0e-13 | | 7.5e-11 (1405, pcorr included) | k 6.3e-13, omega 1.1e-12, nut 1.8e-12; **deltaT of all 74 steps: worst relative 0**; forces 4.0e-12 |

A1p and A5 are gated at the COMMON bands (1e-10 fields, 1e-9 turbulence, 1e-8
residuals, 1e-9 forces); test data 188 K / 2.2 M. `gpu_fv_pimple`: 157 PASS,
0 FAIL with the final binary; A1–A4, P*, M* at their documented numbers.

**Faces without overlap: three defects found with these two cases, fixed**
(stage 3d; each acts only on a cyclicAMI face WITHOUT overlap, so A1–A4 never
saw them; pimpleFoam needs `FOAM_SIGFPE=false` to run such a case at all):

| # | defect | OpenFOAM | fix | measured (FP64, serial pimpleFoam, tight solves) |
|---|---|---|---|---|
| B1 | laminar `nuEff` on a coupled face was the constant `nu` | `interpolate(nuEff)` of the CELL field: `(1 − w) Σ_j w_j nu + w nu`, 0 on a face without overlap | `kAmiUBnd` | `apartial` before: U 14 % / 28 %, p 12 % / 14 % of scale at t = 0.1 / 0.5; after: U 9.1e-14 / 1.8e-13 (abs, scale 1.5), p 2.3e-13 / 7.3e-13 |
| B2 | CorrectPhi's `rAUf` on a coupled face was `lerp` of a cell field of ones (0 without overlap) | `dimensionedScalar rAUf 1`: a SURFACE field, 1 on every face — pcorr (neighbour value 0) carries flux through a face without overlap | `kAmiPBnd(rA = nullptr)`, the two `amiPBnd` calls of `correctPhi` | `amixtilt` before: U 9.5e-3 abs (scale 1.45) at t = 0.3; after: U 3.7e-13, p 2.3e-12, k 1.3e-14, omega 2.7e-11, nut 3.9e-15 (abs) at t = 0.15 / 0.3 / 0.45 |
| B3 | normFactor's row sum `A 1` lacked the share of a face without overlap | `lduMatrix::sumA` subtracts `boundaryCoeffs` whole | `kAmiSumA` + `amiSumA`, called in `solve()` and `solveU()`; only the reported / stop residual moves | A5's pcorr initial residuals were 29 % off (fields already 1e-13); after: 7.5e-11 over 1405 values |

B1 changes the laminar coupled-face nuEff at rounding on covered faces: A1–A3
are not byte-identical to stage 3c's binary, by design; they pass their gates
at the documented numbers.

Also checked, no defect (scratch): the pimpleFoam path on a snappy polyhedral
mesh WITHOUT cyclicAMI (65 859-cell sphere, the M17 schemes, (3, 1, 1), 3
steps): U 5.1e-12, p 1.4e-11 of scale. The wall distance on the M17 mesh moved
to step 1 against `checkMesh -writeFields wallDistance`: 3.4e-11 relative, all
1 227 027 cells.

**A4's level is the case's solver tolerance** (as M3, more on these coarse
wall cells). Serial pimpleFoam against ITSELF with only the p / pcorr solver
swapped (GAMG + GaussSeidel → PBiCGStab + DIC, both to 1e-13) differs at t =
0.15 / 0.45 by U 4.8e-11 / 1.2e-10, p 2.3e-11 / 6.4e-11, phi 2.9e-12 /
2.4e-11, k 8.8e-13 / 1.1e-11, omega 1.6e-13 / 2.5e-12, nut 7.8e-11 /
3.3e-10, forces 1.4e-9, residuals 1.9e-9 — more than dns differs from it in
every field. The same twins: A2 U 4.7e-14 / 4.6e-14, residuals 4.3e-9; A3 U
2.6e-14 / 2.6e-13, forces 9.7e-10. A4's gates are that spread's decade (U, p,
phi 1e-9, k, omega, nut 1e-8, forces 1e-8); A1–A3 keep the common gates. The
same mixer at 4 / 5 cells in z (2400 cells, scratch): U 3.7e-13 / 2.2e-13,
nut 4.5e-13 / 2.1e-13. A propeller-like scratch case (the cylinder's zone
rotating, freestreamVelocity / freestreamPressure, kOmegaSST): U 6.0e-15 /
6.9e-15, omega 1.9e-12. The mixer refined to 86 016 cells (`refineMesh` x 2):
runs, see "open" in issues.md 65.

**A matching cyclicAMI is NOT the conformal mesh, in OpenFOAM.** `amatch`
against the same geometry as one block (`amatch_int`, ordinary faces), cells
matched by centre: U 4.5e-6 / 3.3e-5, p 1.8e-5 / 7.7e-5 at t = 0.1 / 0.5 in
pimpleFoam, and the same digits in dns (4.614e-06, 3.726e-05 both). The
cause is `ddtCorr`: `ddtCouplingCoeff` is 0 on a cyclicAMI patch
(`ddtScheme::fvcDdtPhiCoeff`), non-zero on the internal faces. Proof:
pimpleFoam with `ddtCorr no` on both gives U 1.5e-14 / 4.8e-15.

**Conservation.** No code makes a face pair's fluxes equal and opposite: each
side's patch flux comes from its own Sf, coefficients and interpolated
neighbour values. Every cell's continuity still closes (its own faces), so on
a static mesh `continuityErrs` stays at rounding (A2: global 1.1e-18). On the
rotating closed mixers the pair's net flux shows as the global error: A3 sum
local = |global| = 3.91e-8 per step (cumulative 9.84e-6 after 55 steps), A4
5.62e-8 (cumulative 5.83e-5) — OpenFOAM's numbers, reproduced to the 6
printed digits (3.91321e-08, 5.61546e-08, 5.83367e-05).

Two 30-step `amix3d` runs are byte-identical. Wall time (launch bound + the
per-step host rebuild): A1 7.9, A2 17.9, A3 42, A4 53 ms per step against
serial pimpleFoam's 2.1 / 8.6 / 7.6 / 29 ms.

### What a cyclicAMI face is (verified in the v2606 source)

A COUPLED boundary face, never an internal face. For any cell field x its
patch-neighbour value is `pnf = Σ_j w_j x[N_j]`, j over the face's AMI
addresses in OpenFOAM's order (`AMIInterpolation::interpolateToSource` on the
owner patch — the lower patch index — and `interpolateToTarget` of the OWNER's
AMI on the other; `result = Zero; result += w_j x_j`). Weights are divided by
their own sum per face (`normaliseWeights`, conformal); a face without
overlap has no address and `pnf = 0` exactly (it does not abort:
`Unable to set target face` needs a zero-area face). Then, with the patch's
`w = dni / (di + dni)`, `di = |nf · (Cf − Cn)|`, `dni = |interpolate(nbr nf ·
(Cf − Cn))|`, `δ = (Cf − Cn) − interpolate(nbr (Cf − Cn))`, `dcNO = 1 /
max(nf · δ, 0.05 |δ|)`, `k = nf − δ dcNO` (`cyclicAMIFvPatch::makeWeights`,
`delta`; `basicFvGeometryScheme`):

| operator | on a coupled face |
|---|---|
| face value (`interpolate`, `flux`, Gauss gradient, `evaluate`) | `(1 − w) pnf + w x_P` |
| `cellLimited` | min / max take `pnf`, not the face value |
| snGrad, `limited corrected` | `dcNO (pnf − x_P)`; correction `k · lerp(grad)`, the limiter on those two |
| convection | internalCoeffs `phi wc`, boundaryCoeffs `−phi (1 − wc)`, wc the SCHEME's patch weight (linear: w; upwind, linearUpwind: `pos0(phi)`); linearUpwind's explicit part `(Cf − C_P) · grad_P` for phi > 0, else `(Cf − δ − C_P) · pnf(grad)` |
| laplacian(Γ) | Γ_f = lerp of the cell Γ; internalCoeffs `−Γ_f |S| dcNO`, boundaryCoeffs `−Γ_f |S| dcNO`; faceFluxCorrection `Γ_f |S| limiter (k · lerp(grad))` in the source and in `flux()` |
| laminar nuEff | the interpolated CELL field, as the turbulent one: `(1 − w) Σ_j w_j nu + w nu` — 0 on a face without overlap, never the constant `nu` (B1) |
| CorrectPhi's `rAUf` | a SURFACE field 1, on a face without overlap too (`dimensionedScalar rAUf 1`), not `lerp` of a cell field of ones: pcorr carries flux through such a face (B2) |
| `div(nuEff dev2(T(grad U)))` | `Sf · lerp(nuEff dev2(T(grad U)))` of the CELL tensors |
| matrix (`lduMatrix::Amul`'s interface update) | row P: `−boundaryCoeffs w_j` at column N_j |
| `fvMatrix::relax` | D += the internal coefficient itself (not its magnitude), sumOff += `|boundaryCoeffs|`, the same coefficient removed after the division |
| `H()` | `+ boundaryCoeffs pnf(U)` of the CURRENT U (`addBoundarySource`) |
| `flux()` | `internalCoeffs x_P − boundaryCoeffs pnf + faceFluxCorrection` |
| `fvc::ddtCorr` | 0 (`ddtCouplingCoeff = 0` on a cyclicAMI patch) |
| `adjustPhi`, `correctUphiBCs` | skipped (coupled; does not fix its value) |
| meshPhi | each side its own swept volume: the rotor side the motion's, the stator side exactly 0 |
| `setValues` (omega wall cells) | a fixed cell's row has no coupling; a row coupled TO a fixed cell keeps it |
| meshWave | `FaceCellWave::handleAMICyclicPatches` at the wave's head and after every cellToFace: face i takes the nearest origin over its addresses (`updateFace`, 1% tolerance), measured from the face centre at `owner patch start + i` whatever the receiving patch — OpenFOAM's own indexing, reproduced — then merged on the face's own centre |
| k, omega, nut | type `cyclicAMI`: the rows above; `bound()` averages `lerp` of `max(x, lb)` |

In `gpu/la` the couplings are ASSEMBLED: the matrix graph gains one entry
per (cell, addressed cell) (`buildFvGraph`), the same linear system
OpenFOAM's interface update applies. Two stated differences, both rounding:
the momentum solve's `source + boundaryCoeffs pnf − boundaryCoeffs pnf`
(`solveSegregated`) is taken as `source`; on a COVERED face a row sum
subtracts `boundaryCoeffs Σ w_j` where OpenFOAM's `sumA` subtracts
`boundaryCoeffs` (Σ w_j = 1 to rounding). On a face without overlap Σ w_j is
0 and that was a defect, not rounding: `amiSumA` subtracts the face's
`boundaryCoeffs` whole there, as `lduMatrix::sumA` (B3).

**The pressure matrix is not symmetric** (each side's own weights, Γ_f and
dcNO), in OpenFOAM too (its PCG / GAMG run on it). p and pcorr keep the
flexible AMG-PCG: it converges on these systems (1e-13; 86 016 cells: 58 – 86
iterations). BiCGStab around the scaled V-cycle (a variable preconditioner)
diverged at 86 016 cells and at M17 (`fv: Ux: bicgstab: non-finite residual`
one solve later): do not go back to it.

### Motion

`cyclicAMIPolyPatch::initMovePoints` marks the AMI stale and `AMI()`
recomputes it lazily: ONCE per `fvMesh::movePoints`, nothing per outer
corrector. Here: `FvMeshMotion::update` moves the zone's points, rebuilds the
FaceMesh, then `buildFvGraph` (the AMI of every pair on the moved points and
the graph with its couplings) and the setup; the device re-uploads the AMI
lists and REBUILDS the ELL pattern, both matrices, the multicolour GS and the
AMG hierarchy (`makeMatrices`): the pattern of the couplings changes every
step. Correctness first; cost in "The M17 case" below and issues.md 65. The
host part of that matrix rebuild (ELL pattern, colours, AMG hierarchy) runs on
a second thread beside the wall distance (`FvMeshMotion::update(…, beside)`,
`DeviceSimple::prepareMatrices`), the meshPhi / Uwall sweeps are `forBlocks`:
byte-identical (M17, 20 steps, FP32 and FP64, stats + 15 files each).

### The M17 case (stage 3d)

`m17_prop5_8000_ami`: 1 227 027 cells, two rotor zones (`rotorFwd`,
`rotorAft`, 161 224 of the points) behind two cyclicAMI pairs, 8000 rpm,
kOmegaSST, (3, 1, 1), adjustTimeStep. Label of every number here: `RANS
kOmegaSST (OpenFOAM v2606), meshWave, URANS (time-accurate)`, precision per
row. **One dictionary line differs from the user's case**:
`system/controlDict` `startFrom latestTime;` → `startFrom startTime;` (a
time-0 copy; restarts are not built). `--dry-run` exit 0. FP64 and FP32 both
run, 0 linear solves at maxIter; stage 3c's `fv: Ux: bicgstab: non-finite
residual` does not occur with p / pcorr on flexible AMG-PCG. A solve that
reaches maxIter is reported on stderr by name, its count in the end summary.

**pimpleFoam itself needs `FOAM_SIGFPE=false` on this case in serial** (the
user's np-14 log has no `trapFpe` line): with trapping it dies in
`kOmegaSSTBase::F2` at construction (division by the patch value 0 of omega
on cyclicAMI faces without overlap).

**20 steps from time 0, FP64** (re-run on the final binary, B3 included:
every written field byte-identical to the run before B3, every iteration count
equal; only reported initial residuals move — Ux ≤ 2.0e-9, k ≤ 4.0e-10
relative, pcorr in the 6th digit — so the table below holds as printed).
References: the user's np-14 log; serial pimpleFoam at the case's own
tolerances (328 s for 20 steps); a serial OpenFOAM-vs-OpenFOAM twin (U / k /
omega `PBiCGStab + DILU`, p `GaussSeidel`, `nCellsInCoarsestLevel 10`).

Equal to the printed digits on every step: deltaT (worst 7.4e-9 = the log's 8
digits), every `AMI: … sum(weights)` number of both pairs (0 difference at 8
digits), Courant mean (1.9e-8). Courant max: 3.1e-6.

dns FP64 vs SERIAL pimpleFoam, the user's tolerances (relative):

| step | time | T_fwd | Q_fwd | T_aft | Q_aft | airframe \|F\| | first-corrector U residuals | first p residual per corrector | omega / k residual |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 5.9952038e-06 | 6.3e-6 | 7.1e-6 | 2.4e-6 | 2.0e-6 | 1.3e-5 | 1.8e-3 | 2.8e-4 | 1.1e-6 / 3.6e-6 |
| 10 | 7.2076379e-05 | 4.0e-5 | 2.1e-5 | **3.2e-3** | **2.8e-3** | 8.7e-4 | 4.1e-3 | 3.0e-4 | 1.1e-4 / 2.1e-4 |
| 20 | 0.00014634826 | 6.9e-6 | 8.6e-6 | 1.4e-4 | 1.5e-4 | 5.3e-4 | 8.1e-3 | 2.3e-3 | 2.4e-4 / 5.4e-5 |
| worst of 20 | | 8.1e-5 (15) | 6.7e-5 (2) | 3.15e-3 (10) | 2.8e-3 (10) | 2.9e-3 (2) | 1.15e-2 (15) | 2.3e-3 (20) | 2.4e-4 / 4.1e-4 |

Against the user's np-14 run: T_fwd 1.4e-5 / 4.8e-5 / 1.7e-4, T_aft 1.1e-5 /
3.1e-3 / 3.3e-5 at steps 1 / 10 / 20 (run before B1–B3; the aft numbers did
not move with them). OpenFOAM against itself over the same 20 steps: np 14 vs
serial T_fwd ≤ 2.6e-4, T_aft ≤ 1.3e-4, airframe 1.9e-3 (step 2); twin vs
serial T_fwd ≤ 4.5e-5, T_aft ≤ 6.4e-5. Second non-orthogonal p residuals and
the continuity errors are not comparable (they measure how the first, relTol
0.01, solve stopped): up to 48 % apart. Iterations: GAMG 37–66 per step, dns
AMG-PCG 103–171. FP32 (pre-fix binary) vs serial: T_fwd ≤ 8.8e-5, T_aft ≤
2.3e-3, U residuals up to 12× (FP32 noise floor), continuity sum local 1e-10
against 1e-15; 125 `floor-limited` stops in 20 steps.

**The aft bump (steps 6–13, peak +0.058 N of 18.38 N at step 10) is NOT an
algebra difference; it is a stated solver-quality gap, not closed** (issues.md
65 (p)). Measured chain:
1. Tightening only `tolerance` (p 1e-11, pcorr 1e-10) changes neither code:
   dns +0.0587 N, pimpleFoam within 1e-5 N of its loose run.
2. With `relTol 0` on EVERY solve as well (12 steps): **dns FP64 vs serial
   pimpleFoam T_fwd ≤ 1.5e-6, Q_fwd ≤ 1.6e-6, T_aft ≤ 1.1e-6, Q_aft ≤ 1.3e-6**
   (step 10: T_aft 2.9e-7), Courant max 6.2e-8, airframe |F| 2.3e-5 typical,
   2.7e-4 at step 2.
3. So the difference lives in the solves the case stops at relTol 0.01 (first
   non-orthogonal p solve of each corrector) / 0.1 (non-final U): the second p
   solve then corrects with an explicit non-orthogonal term built on an
   unconverged p. OpenFOAM's GAMG at relTol 0.01 is within 6.1e-5 of its own
   converged run on T_aft; **dns's AMG-PCG at the same stop is 3.2e-3 off**:
   it leaves more error where the system is ill-conditioned.
4. Where: a dead-end cavity inside the tail, x 0.78–0.9986, r < 0.045
   (first-step p 6e5–1.3e6 against 612 of stagnation), fed only through the
   ring at r 0.04 next to the aft AMI's end cap, where all 24 faces without
   overlap sit (each in an `airframe` wall cell of the ROTATING zone: the
   user's `airframe` noSlip patch has faces inside `rotorAft`) and where one
   AMIAft2 face's weight sum falls 0.33 → 0.0002 over steps 3–15. One step in:
   p differs by a near-uniform 227 Pa in that cavity (2.7e-4 of its level), U
   by up to 0.56 m/s in hub cells; nut differs in 98 000 cells by up to 37 %
   of scale because `cellLimited` amplifies U differences of 1e-4 (k, omega
   agree to 1e-3 / 1e-4 there).

A parity run of this case needs `relTol 0`, or a p solver whose error at
relTol 0.01 is GAMG's (FGMRES + AMG is in `gpu/la`).

**720 steps, FP32 (the default), final binary** (sha256 `ed54fed3e95fff83…`),
the user's dictionaries with the one line above, against the user's np-14
pimpleFoam log at equal step numbers: t = 0.0052342239 (foam 0.0052342864),
0.698 of a revolution at 8000 rpm. Stable in FP32: 0 linear solves at maxIter,
3636 `floor-limited` stops. Relative differences:

| step | T_fwd | Q_fwd | T_aft | Q_aft | airframe \|F\| |
|---|---|---|---|---|---|
| 1 | 1.5e-5 | 2.3e-5 | 7.7e-7 | 9.7e-6 | 3.4e-5 |
| 10 | 7.0e-6 | 1.9e-6 | 1.7e-3 | 1.5e-3 | 1.1e-3 |
| 100 | 5.6e-4 | 5.1e-4 | 9.8e-4 | 1.3e-3 | 7.8e-4 |
| 300 | 1.4e-4 | 5.1e-5 | 7.6e-4 | 2.7e-4 | 5.2e-3 |
| 600 | 5.6e-4 | 2.7e-4 | 5.9e-3 | 4.6e-3 | 8.5e-3 |
| 720 | 1.2e-4 | 7.0e-5 | 2.5e-4 | 4.7e-4 | 8.6e-3 |
| worst of 720 | 1.3e-3 (655) | 8.8e-4 (101) | 6.6e-3 (603) | 5.3e-3 (603) | 1.0e-1 (655) |

Worst over 720 steps also: deltaT 2.4e-3 (step 628), Courant max 2.4e-3
(629), time 3.7e-5 (423): deltaT follows the Courant maximum once it is not
capped. The airframe force is ~6 N, a small difference of large terms (the
cavity's): 1.0e-1 at step 655. Values at step 720: T_fwd 25.25578 N (user)
25.25885 (dns); T_aft 18.74541 / 18.75008.

| mean of the last 100 common steps | user np 14 | dns FP32 | relative |
|---|---|---|---|
| T_fwd [N] | 25.39080 | 25.38008 | 4.2e-4 |
| Q_fwd | −1.69227 | −1.69191 | 2.1e-4 |
| T_aft [N] | 18.89050 | 18.88676 | 2.0e-4 |
| Q_aft | 1.32988 | 1.32981 | 5.8e-5 |

The aft excursions (steps ~10, ~603) are the relTol / cavity mechanism above:
measured for steps 6–13, inferred for step 603. This is a trajectory
comparison against a relTol 0.01 / 0.1 np-14 run, not a gate.

**Cost.** The 720-step run: 2174.44 ms per step = host mesh update 1785.2 +
device 217.2 + solves 172.0; wall 1569.9 s against the user's ClockTime 4170 s
at step 720 (**2.66×**; np 14). Device memory 2222 MiB, host peak RSS 5673
MiB. Per time step on 20-step runs (CPU otherwise idle unless noted):

| binary, precision | total ms | host mesh update | device mesh update | solves |
|---|---|---|---|---|
| start of 3d, FP64 | 2789 | not split | not split | not split |
| start of 3d + timers, FP32 (a serial pimpleFoam running beside it) | 2831 | 2030 | 624 | 177 |
| after the overlap change (the side thread), FP32 | 2236 | 1832 | 224 | 181 |
| after the overlap change, FP64 | 2414 | 1848 | 199 | 368 |
| final algebra (B1, B2), FP64 | 2182 | 1624 | 190 | 368 |

Device memory (free-memory delta): FP32 2222 MiB, FP64 3718 MiB; host peak
RSS 3.5–4.5 GiB on the 20-step runs. The user's np-14 pimpleFoam: 5.8 s per
step (ClockTime 583 s at step 100, 36 494 s for 6239 steps); serial 16.4 s.
Host update split (3 updates, before the change): 1.74–1.88 s = meshWave
0.88–0.91, AMI + graph 0.32–0.33, FaceMesh 0.20–0.22, setup 0.11–0.17, points
+ meshPhi + moves 0.22–0.25. Not done (not byte-safe or not cheap): meshWave
is serial and is half the step; only the rotor zones' points move, so FaceMesh
/ setup could be recomputed for their faces only (identical bits, a
restructure); the AMG aggregation is redone every step.

**Regression.** Final binary: `ctest -L fast` 66/66, `gpu_fv_pimple` 157
PASS, `gpu_fv_simple` 87 PASS. `bit_perf.sh` 14/14 byte-identical FP64 and
FP32, `pimple_bytes.sh` 9/9 FP64, FP32 8/9 + `mspin` rc 7 (pre-existing,
stage 3c: the pre-3d binary exits 7 there too), all against the pre-3d
binary's outputs.

### Reading the case

- `constant/polyMesh/boundary`: `type cyclicAMI; neighbourPatch P;` with
  `transform noOrdering` (or `unknown` / absent), `AMIMethod faceAreaWeightAMI`
  (or absent), `restartUncoveredSourceFace`, `requireMatch` true (or absent),
  `reverseTarget` false (or absent); `matchTolerance` (read by the rotational
  transform only), `inGroups`. Face counts of the two patches are free.
- `0/*`: `type cyclicAMI` on U, p, k, omega, nut (`"AMI.*"`, the patch name,
  or `setConstraintTypes`), `value` accepted.

Rejected by name (exit 3): `cyclicACMI`, `nonConformal*`; a cyclicAMI
`transform` other than noOrdering (rotational, translational); any other
`AMIMethod` (faceAreaWeightAMI2D, nearestFaceAMI, …); `lowWeightCorrection`;
`restartUncoveredSourceFace` / `requireMatch` off, `reverseTarget` on; every
other entry of the patch (`rotationAxis`, `periodicPatch`, `surface`,
`createAMIFaces`, `fixedPatch`, …); a neighbourPatch that is not a cyclicAMI
naming it back; a non-cyclicAMI field type on such a patch; cyclicAMI under
simpleFoam; cyclic and cyclicAMI patches in one mesh; uncorrected laplacian /
snGrad; `wallDist exactDistance`; limitedLinear convection of k / omega; a
cell pair joined both by a face and by an AMI address; cyclicAMI on every
other route (`--bc`, `dns mesh`, the compressible and time-accurate FV
solvers: the FaceMesh refuses it).

## Output

As `docs/fv.md` "Output and CLI": `PREFIX/<time>/{U,p,phi}` (+ k, omega, nut)
at OpenFOAM's write times and at the end, the time name `%.{timePrecision}g`;
`PREFIX/postProcessing/<forces>/0/{force,moment}.dat` / `coefficient.dat`, one
row per time step; `PREFIX_stats.csv` (`step, time, deltaT, coMean, coMax,
Ux0, Uy0, Uz0, p0, pFinal, contLocal, contGlobal, F, M, ms` + `omega0, k0`:
U*0 / p0 from the step's first outer corrector); the log in pimpleFoam's shape
(`Courant Number mean: … max: …`, `deltaT = …`, `Time = …`, `PIMPLE: iteration
k`, the solves, `time step continuity errors`).

Stated differences: the end state is always written (OpenFOAM writes at write
times only); `Time::operator++`'s growth of `timePrecision` for clashing
directory names is not built; a function object's `writeControl` is ignored
(one row per step, noted), as on the simpleFoam path.

## Files

| file | content |
|---|---|
| `dns/fvcase.{hpp,cpp}` | `FvPimpleCtl`, `FvTimeCtl`; `readPimpleSolution`, the pimpleFoam branch of `readControl` / `readSchemes`, the rejections |
| `dns/fvpimple.{hpp,cpp}` | `FoamClock`: OpenFOAM's Time as pimpleFoam drives it |
| `gpu/fv_pimple.hip` | `timeStep`, `courant`, `ddtCorr`, `rasDdt`, `pimpleMake`; kernels `kDdtCorrF/B`, `kSDdt`, `kCoCell`, `kMaxPart` |
| `gpu/internal/fv_device.hpp` | `PimpleState<R>` (old-time fields, both control sets), `Ctl::rDeltaT` |
| `gpu/fv_simple.hip` | `kURow<R, Multi, Ddt>`, the `ddtCorr` hook in `prepP`, `pimpleMake` in `make` |
| `gpu/fv_ras.hip` | the `rasDdt` hook before each scalar solve |
| `gpu/runner_fv.{hpp,hip}` | `runFvPimple`; the forces-file helpers shared with `runFvSimple` |
| `tools/fv/make_pimple_cases.py`, `run_foam.sh`, `make_pimple_test_data.sh` | the six cases, their serial pimpleFoam runs, the checked-in references |
| `tests/data/fv/{pcavity,pcavity_skew,pchannel,pduct,psst_plate,psst_skew}` (2.7 MB) | the cases, two reference times each, `ref/log.csv` (time, Courant number, deltaT, every initial residual per step), force files |
| `tests/test_gpu_fv_pimple.cpp` (`gpu_fv_pimple`), `tests/test_fvcase.cpp` P1–P3 | the parity gates; the reader, the rejections, `FoamClock` |
| **moving mesh** | |
| `dns/fvcase.{hpp,cpp}` | `FvDynamicMesh`, `FvSolidBody`, `FvPatchBc::movingWall`, `FvSchemes::gradPcorr`, `FvPimpleCtl::correctPhi / checkMeshCourantNo / solverPcorr*`; `readDynamicMesh`, `readSolidBody` |
| `dns/fvmotion.{hpp,cpp}` (compiled `-ffp-contract=off`) | `foamRotatingMotion` (septernion / quaternion / `transformPoints`), `foamFaceCentre`, `foamFaceSweptVol`, `FvMeshMotion::make` (the motion groups, the refusals) and `::update` (points → `fromPolyMesh` → `buildSimpleSetup` / `buildRasSetup`, meshPhi, Uwall) |
| `gpu/internal/fv_device.hpp` | `MotionState<R>` (meshPhi, Uf and its old time, V0, pcorr, phiAbs) |
| `gpu/fv_pimple.hip` | `meshUpdate`, `relFlux`, `correctUf`, `courantOf`, the moving branch of `timeStep` / `ddtCorr`; kernels `kDdtCorrUfF/B`, `kUfF/B`, `kFluxShift`, `kSetWall`; `kSDdt` on `V0` |
| `gpu/fv_simple.hip` | `correctPhi` (CorrectPhi on the p equation's kernels), `adjustPhi` (prepP's block, now shared), `kPhiUf`, `kPhiFixedB`, `kURow`'s `V0`; `state()` downloads Uf, meshPhi |
| `gpu/fv_ras.hip` | kOmegaSST's divU on `phiAbs()` |
| `gpu/runner_fv.hip`, `dns/foamio` | the per-step `FvMeshMotion::update`; `meshPhi`, `Uf`, `polyMesh/points` in the time directories (`writeFoamPoints`, surfaceVectorField) |
| `tools/fv/make_motion_cases.py`, `run_foam.sh`, `make_pimple_test_data.sh`, `pimple_bytes.sh` | the three cases, their serial pimpleFoam runs, the references; the byte gate of the pimpleFoam path against an earlier binary |
| `tests/data/fv/{mspin,mduct,msst}` (2.5 MB) | the cases, two reference times each (U, p, phi, Uf, meshPhi, `polyMesh/points`, RAS fields), `ref/log.csv`, `ref/meshco.csv`, force files |
| `gpu_fv_pimple` M1–M3, `fvcase` "moving mesh" | the parity gates; the reader, the rejections, points / meshPhi bitwise, geometric conservation, Uwall |
| **cyclicAMI** | |
| `dns/polymesh`, `dns/facemesh` | `FoamPatchType::CyclicAMI` (the reader's accepted spelling, every other entry by name), `FaceMeshOptions::ami` (boundary patches; refused on every other route) |
| `dns/fvami.{hpp,cpp}` (`-ffp-contract=off`) | `foamCyclicAmi`: OpenFOAM's `cyclicAMIPolyPatch::resetAMI` → `faceAreaWeightAMI::calculate` (triangulation, the advancing-front walk, `faceAreaIntersect`, the target octree, restartUncoveredSourceFace, normaliseWeights), `foamAmiInterpolate`, the log's sums |
| `dns/fvamisetup.cpp` (`-ffp-contract=off`), `dns/fvsimple.hpp` | `buildFvGraph` (the AMI pairs, the graph with the couplings), `bindFvAmi` → `FvAmiSetup` (addresses, weights, ELL positions, patch w / δ / dcNO / k) |
| `dns/la/graph` | `graphOf(fm, extra)`, `graphSlot` |
| `dns/fvwalldist` | `FvWaveAmi`: the wave across cyclicAMI |
| `dns/fvmotion` | zones behind cyclicAMI; `update` rebuilds the AMI and the graph (`FvGraph&`) |
| `gpu/fv_ami.hip` | `amiMake`, `amiEval`, `amiUBnd`, `amiOff` (records the coupling), `amiHsrc`, `amiPhiH`, `amiPBnd` (null rA: CorrectPhi's surface rAUf 1), `amiPhiB`, `amiSumA`; kernels `kAmi*` (`kAmiSumA`) |
| `gpu/internal/fv_device.hpp` | `Mesh` AMI view, `amiNbr`, `amiLerp`, `kGrad<…, Cpl>`, `gradLaunch`, `SimpleImpl::AmiDev`; `MatPrep`, `prep`, `AmiCpl`, `cplU`, `cplP` |
| `gpu/fv_simple.{hpp,hip}`, `gpu/fv_ras.hip`, `gpu/fv_pimple.hip` | `kURow<…, Cpl>`, `kPRow<…, Cpl>`, `kSRow<Cpl>`, `kBoundCells<Cpl>`, `kAmiSBnd`, `kAmiZero`, the hooks, `makeMatrices` (again after a mesh update) and its host part `matPrepOf` / `DeviceSimple::prepareMatrices`; `amiSumA` in `solve` / `solveU`; `meshUpdate` timed (`FvIterLog::msSetup`) |
| `dns/fvmotion`, `gpu/runner_fv.hip` (stage 3d) | `Beside`: the side thread of `FvMeshMotion::update`, `forBlocks` sweeps; the per-step `# mesh update: host / device / time step` line and the end summary (split, device memory, host peak RSS, maxIter stops by name on stderr); AMI sums printed with 8 digits |
| `tools/fv/make_ami_cases.py`, `make_ami_flow_cases.py`, `foamAmiDump/`, `run_ami_foam.sh`, `run_foam.sh` | the meshes, the flow cases (`apartial`, `amixtilt` run with `FOAM_SIGFPE=false`), the v2606 dump utility (wmake), the references |
| `tests/data/fv/ami` (1.0 MB), `tests/data/fv/{aplanar,amatch,acyl,amix2d,amix3d}` (3.8 MB), `tests/data/fv/{apartial,amixtilt}` (188 K, 2.2 M) | meshes + AMI dumps; the flow cases with their reference times |
| `fvami`, `gpu_fv_pimple` A1–A4, A1p, A5, `fvcase` "cyclicAMI", `polymesh` | weights bit for bit; the parity gates; the graph, the setup, the rejections |

## Invariants

- **A simpleFoam case is the pre-pimple binary byte for byte** (`bit_perf.sh`
  14/14 in FP32 and FP64, `structured_bytes.sh`): `PimpleState` is null, no ddt
  kernel instantiation launches, `kURow<…, false>` is the old kernel.
- **The Impl's `FoamCase` copy and `RasState::co` hold the CURRENT corrector's
  entries** (`fc.solverU`, `c.relaxU`, `relaxP`, `ras->co.solverK`, …):
  `timeStep` rewrites them per corrector from `PimpleState`'s two sets; the
  stage methods read them where simpleFoam's do.
- A plateau above rounding is a bug; a case that amplifies rounding is shown
  to by an OpenFOAM-vs-OpenFOAM twin (P4, M3), never by a wider band alone.
  The twin swaps the solver whose iterates differ between the codes: for a
  tolerance-limited case that is the PRESSURE solver (a U-solver swap left
  M3's twin at 5e-14 and proved nothing).
- **A static-mesh pimpleFoam case is the pre-motion binary byte for byte**
  (`tools/fv/pimple_bytes.sh`: the six stage-3a cases, stats + every file, FP32
  and FP64): `PimpleState::mov` is null, `kURow` / `kSDdt` read `V0 = m.V`, no
  motion kernel launches.
- **The moved mesh is recomputed, never transformed**: points(t) from the
  initial points, then `fromPolyMesh` + `buildSimpleSetup` + the wall distance.
  A device-side rotation of S, C, … would be a second geometry code (and is
  not bitwise OpenFOAM's: its geometry is recomputed from rotated points).
- **phi's state is positional**: relative from the end of pEqn.H (and of
  correctPhi.H) to the next one, absolute inside. A reader of phi (CourantNo,
  `fvm::div`, inletOutlet's sign, kOmegaSST's divU) takes the state OpenFOAM's
  has at that line.
- `makeRelative` / `makeAbsolute` are separate launches (`phi − meshPhi`, later
  `+ meshPhi`): never folded to a no-op, the round trip is not the identity.
- **A mesh without cyclicAMI is the pre-AMI binary byte for byte**
  (`bit_perf.sh` 14/14, `pimple_bytes.sh` 9/9, FP32 and FP64): `m.nA == 0`
  launches no `kAmi*` kernel and takes the `Cpl = false` instantiations.
- **The patch kernels never learn about cyclicAMI**: a coupled face carries
  the zeroGradient codes and every `ami*()` hook overwrites what they wrote
  there. A new patch kernel output read on boundary faces needs its hook.
- **A coupled face's neighbour value is computed where it is used** (`amiNbr`,
  the address order, no contraction): no cached patch-neighbour field to keep
  in step with the cell field.
- **The AMI is recomputed, never rotated**: `foamCyclicAmi` on the moved
  points once per mesh update, and the matrix pattern with it.
