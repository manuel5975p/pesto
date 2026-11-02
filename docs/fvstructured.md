# simpleFoam on the structured meshes — `dns flow --solver simple`

The structured flags name a `CurvMesh`. `FaceMesh::polyOf` turns it into a
polyMesh, `dns/fvstructured` writes a full OpenFOAM case, and the unchanged
loader reads that directory: `dns flow --solver simple --out P` =
`--export P` + `--polymesh P --out P`, byte for byte. No second solver, no
second geometry. `docs/fv.md` is the solver, `docs/mesh.md` the mesh.

```
dns flow --solver simple [mesh] [physics] [--max-iters N] [--steady TOL]
         [--fv-precision fp32|fp64] [--out PREFIX] [--emit stats|none] [--dry-run]
dns flow --solver simple [mesh] [physics] --export CASE     # writes the case, no GPU
```

`--export` lives on `dns flow`, not `dns mesh`: the case needs the same flags
and validator as the run, and `dns mesh` has no grid flags. `--dry-run` parses
the generated texts against a flags-only manifest (`FaceMesh::polyManifest`):
no file, no GPU, the case the run will read. A run writes the case into PREFIX/,
so a run directory is a complete OpenFOAM case with its results beside it.

## What the flags become

| flags | mesh | patches (U / p) |
|---|---|---|
| `--body none` | uniform box, compressible box conventions (D = pi/4, `--dr`) | `inlet` fixedValue / zeroGradient; `outlet` inletOutlet / fixedValue 0; `lower`, `upper`: symmetryPlane; `--noslip` wall; `--open` freestreamVelocity / freestreamPressure; z empty or cyclic |
| `+ --stretch B` | `CurvMesh::stretchedY` | same |
| `+ --focal x0,x1,y0,y1,z0,z1` | `fromRemap(focalRemap)` | same |
| `--body none --far periodic --noslip --re-tau R` | the plane channel: delta 1, u_tau 1, nu 1/R, cyclic x (2 pi), `--ny` fluid rows; `--stretch B` is `channelY`, the oracle's own y nodes; `--focal` the bare box | `lower`, `upper` noSlip; `constant/fvOptions` body force 1 |
| `--ogrid --body B` | `buildOGrid`, rows 1..ny-2 | `body` noSlip; `far` freestreamVelocity / freestreamPressure; z empty or cyclic |
| `--ogrid --body sphere --nz 1` | `buildRevolvedOGrid` wedge, plane k = 1 | `body`, `far`, `front`, `back`: `symmetry`; no axis patch |
| `--ogrid --body sphere --nz N` | full revolution | `body`, `far`; no axis patch |

Numerics in the generated case: `bounded Gauss linearUpwind grad(U)`,
`bounded Gauss upwind` for k, omega, nuTilda (the structured RANS scalars are
first-order upwind too), `Gauss linear corrected`, SIMPLEC, U 0.9, turbulence
0.7, p GAMG, 1 non-orthogonal corrector on an O-grid. Edit the exported
dictionaries for anything else; `dns flow --polymesh` reads them.

`--rans sa-neg` = OpenFOAM `SpalartAllmaras` with `ft2 true`; `--rans sst-2003m`
= `kOmegaSST`; `k`, `nuTilda` fixedValue 0 on walls, `omegaWallFunction`,
`nutLowReWallFunction`. `--rans sst-vm` is rejected: OpenFOAM has no such
variant. wallDist is meshWave, or exactDistance where a cyclic patch exists
(meshWave rejects cyclics, `docs/fv.md`), with `libdistributed.so` in controlDict.

`--rotor-mach T --mach M --rotor-frame mrf --rotor-axis z` on a 2D O-grid: a
`rotor` cellZone of cells with r < D, Omega from tip speed / U = T / M, the
wall moves as an MRF-included patch. The ratio is the only meaning those flags
keep here; `--mach` alone is rejected.

**Rejected by name** (exit 2): every flow flag not listed in `--help`'s block
(`--les`, `--cfl`, `--ramp`, ...); `--far sponge | fringe`; an immersed body
without `--ogrid` (no body-fitted faces; a staircase is not exported);
`--physics compressible`; `--far periodic` without `--re-tau` (nothing drives
it); `--rans sst-vm`; MRF on `--nz > 1` (zone cells on cyclic faces, `docs/fv.md`)
and on a revolved mesh (shaft axis); `--steady` on the channel (Uy is rounding
normalised by rounding, residual ~0.2 forever).

**Built vs not.** `axisFaces` and `MirrorZ` are built in `polyOf`: axis prisms,
no zero faces. OpenFOAM's `wedge` patch is NOT built: the wedge's sides are
`symmetry`, whose value is `cos(dphi/2)` of the rotated one, and `Uz` is solved
(rounding, ~1e-17). The wedge therefore never certifies a `--steady` stop
(measured: Uz residual 0.25 for 12000 iterations); use `--max-iters`. The same
holds on any 3D case with a zero component.

## Gates

`test_fvstructured` (host, 0.1 s), `gpu_fv_structured` (GPU, 81 s measured beside other jobs),
`tools/fv/structured_bytes.sh` (bytes). Ladders: `test_gpu_fv_structured MODE`,
FP64, `simjob --gpu`.

| pair | mesh | quantity | expected | measured |
|---|---|---|---|---|
| export vs read | 15 families (`test_fvstructured`) | `readPolyMesh`, every text, FaceMesh vs `fromPolyMesh(polyOf)`, flags-only manifest vs `polyOf` | bitwise | bitwise on all 15 |
| exported case vs `--solver simple` | O-grid, wedge, full revolution, SST, SA channel, focal `--noslip` | stats but `ms`, fields, force.dat, moment.dat (FP64) | bitwise | 6 of 6 identical (`structured_bytes.sh` b) |
| this binary vs f892cc6 | 9 `tests/data/fv` cases, FP32 and FP64; 3 structured flows | stats but `ms`, fields, forces; flow stats | bitwise | 21 of 21 identical (a) |
| `polyOf` vs `CurvMetrics` | focal box | V | rounding | 8.5e-15 of max |
| `gpu_fv_simple` | stored simpleFoam data | fields, residuals | unchanged gate | pass |
| periodic Poiseuille vs FV closed form `y(2-y)/(2 nu) + f h^2/(8 nu)` | uniform, ny 32 | u, every row | rounding | 1e-12 gate pass; wall stress 1 to 5e-12 |
| Poiseuille vs exact | `--stretch 2`, ny 64 / 128 / 256 / 512 | max err / peak | order 2 | 1.048e-3, 2.625e-4, 6.566e-5, 1.642e-5: order 2.000 |
| Poiseuille vs exact | `--focal`, ny 64 / 128 / 256 / 512 | max err / peak | order 2 | 4.900e-4, 1.225e-4, 3.062e-5, 7.631e-6: order 2.005 |
| freestream | focal box, 20 it | max \|U - 1\| | rounding | 1.3e-15 |
| MRF spinning cylinder in a fixed circle vs Couette | O-grid n 32 / 64 / 128 / 256, 4 D | L2 err / U_wall | order 2 | 2.58e-2, 6.46e-3, 1.52e-3, 3.64e-4: order 2.00, 2.09, 2.06 (max norm 1.67, 1.89, 1.96) |
| SA channel vs `rans_channel1d` M 0.05 | oracle's y mesh, Re_tau 180, beta 2, ny 48 / 96 / 192 / 384 | ub gap | converged limit, named model gap | +0.042, +0.014, +0.007, +0.005% (ny 384 at the 60000 it cap, monitor 1.9e-11); max \|du\|/u_c 7.1e-4, 2.1e-4, 6.5e-5, 3.4e-5 |
| SA channel | Re_tau 1000, beta 3, ny 96 / 192 / 384 | ub gap | same | +0.065, +0.020, +0.008% (ny 384 at the cap) |
| SST channel vs `rans_channel1d` | Re_tau 180, beta 2, ny 48 / 96 / 192 / 384 | ub gap | same | +0.452, +0.228, +0.084, +0.035% (ny 384 at the 60000 it cap, monitor 2.4e-11) |
| SST channel | Re_tau 1000, beta 3, ny 96 / 192 / 384 | ub gap | same | +0.256, +0.123, +0.037% (ny 384 at the cap, monitor 4.2e-9: NOT converged) |
| RAS wall stress | both channels | tau_w vs the body force | rounding at the fixed point | 1 to 3e-9 |
| wedge vs full revolution vs structured | sphere Re 100 | cd | converged limit | see below |
| O-grid cylinder Re 40 vs structured `--ogrid` M 0.1 | same mesh | cd | converged limit, named model gap | see below: far model; U clamped on the far circle, gap +8.1e-4 at 20 D |
| O-grid cylinder Re 40 vs incompressible cut cell | `--dr 16 --open` | cd | same | see below: different domain, gap not closed |

Ladder commands:

```
simjob submit --gpu -t 900 -m 8192 --wait -- build/test_gpu_fv_structured poiseuille
simjob submit --gpu -t 900 -m 8192 --wait -- build/test_gpu_fv_structured couette 32 64 128 256
simjob submit --gpu -t 900 -m 8192 --wait -- build/test_gpu_fv_structured ras sa-neg 180 2 48 96 192 384
simjob submit --gpu -t 900 -m 8192 --wait -- build/test_gpu_fv_structured cyl 20 40000 40 3 64 128 256 512
simjob submit --gpu -t 600 -m 8192 --wait -- /bin/bash tools/fv/structured_bytes.sh build/dns BASE /tmp/dns/simple/bytes
dns flow --ogrid --body cylinder --physics compressible --n N --ny N/2+2 --ogrid-far R --ogrid-beta 3 --re 40 --mach 0.1 --steady 1e-5
dns flow --body cylinder --n N --dr 16 --open --re 40 --ft 60 --steady 1e-5
```

## Reading the gaps

**RAS.** The fixed point is a body-force channel on the oracle's own nodes, so
every gap is the model or the discretization. Named differences, none an error
band: OpenFOAM `SpalartAllmaras` (Stilda floor `Cs Omega`, r capped at 10,
`nuTilda` bounded at 0, ft2 on) vs TMR SA-neg; OpenFOAM `kOmegaSST` fixes
omega in the wall CELL by `omegaWallFunction` (6 nu / (beta1 y^2) with binomial
blending) where SST-2003m images 60 nu / (beta1 d1^2) on a ghost row, and its
scalars are solved with k = 0 on the wall face; the oracle is compressible
(M 0.05, isothermal), the FV path is not. SST's ub itself moves 3% with ny on
both sides; the gap falls with it. SA's gap flattens near +0.005%; every ny 384
row stopped at the 60000 iteration cap, so that rung is not a certified fixed
point and no limit is claimed for either model.

**Cylinder Re 40, grid.** simple, beta 3, rFar 20 D (FP64, stops to 1e-12 of
Fx, `linearUpwind`):

| n x nf | cd | cdp | cdf |
|---|---|---|---|
| 64 x 32 | 1.56097 | 1.01862 | 0.54235 |
| 128 x 64 | 1.54371 | 1.00922 | 0.53449 |
| 256 x 128 | 1.53955 | 1.00816 | 0.53138 |
| 512 x 256 | 1.53929 | 1.00906 | 0.53023 |

central `bounded Gauss linear`: 1.53483, 1.53496, 1.53548, 1.53608. Not
asymptotic second order: cd from wall faces carries first-order wall terms
(zeroGradient p, one-sided stress), with opposite signs in cdp and cdf, and
beta 2 stays non-monotone to n 1024 (1.5392, 1.5392, 1.5408). The cd differences
fall 0.0173, 0.0042, 0.0003 at beta 3, an apparent order > 2 that is a sign
change, not convergence: no Richardson number is quoted.

**Cylinder Re 40, solvers.** Same mesh, beta 3, structured `--ogrid` M 0.1,
TOL 1e-5 (M 0.05 moves cd 1e-4 at n 128: not Mach):

| rFar | simple n 128 | structured n 128 | gap |
|---|---|---|---|
| 20 D | 1.54371 | 1.55781 | 0.0141 |
| 40 D | 1.52018 | 1.52319 | 0.0030 |
| 80 D | 1.51057 | 1.51104 | 0.0005 |
| 160 D | 1.50427 | 1.51051 | 0.0062 |

**The 20 D gap is the far model, measured.** Clamp `U` on the far circle
(`fixedValue` + zeroGradient p, edited into the exported case, central scheme) and
simple lands on the structured row, which is stiff on `u_n` at low M
(`dun_b = dp/(rho c)`):

| rFar 20 D, central | n 192 | n 256 |
|---|---|---|
| simple, freestreamVelocity / Pressure | | 1.53548 |
| simple, U clamped | 1.55917 | 1.55904 |
| structured M 0.1 | 1.55812 | 1.55823 |
| clamped - structured | +1.05e-3 | +8.1e-4 |

-0.0228 becomes +8.1e-4: the two boundaries are different models, and the gap is
that model difference, a stated gap, not a bug. The mixed patches let `u_n` move
with p; the characteristic row barely does. M is not it (M 0.05 moves structured
1.1e-4 at 20 D). Clamped `U` is not the characteristic row either (it also pins
`u_t` on the outflow half), so +8.1e-4 is not a zero; two rungs, no order. n 128 and
40 D clamped did not reach the stop (monitor 1e-4 to 1e-3 at 40000 it).

Tried and NOT decisive: `Far` replaced by the mixed patches' algebra in compressible
FV (vf = 0.5 -+ 0.5 cos) moves cd -4.6e-3 at 20 D but rings at +-4e-3 and leaves
FV 0.0174 above simple central, because simple's `phi_b` follows the pressure
equation, not `U_b.S`: the face algebra is not the same model. FV with `U` clamped
rings at +-0.025 (a rigid acoustic box, period 29), time mean 1.564-1.568, NO
steady number.

Central-scheme table, n 128, M 0.1 (the rows above are `linearUpwind`, a 0.006-0.009
offset of its own):

| rFar | simple central | FV | structured |
|---|---|---|---|
| 20 D | 1.53496 | 1.55697 | 1.55781 |
| 40 D | 1.51305 | 1.52197 | 1.52319 |
| 80 D | 1.50386 | 1.50826 | 1.51104 |
| 160 D | 1.49820 | 1.49988 | 1.51051 |

**160 D = 80 D at n 128 was a grid coincidence, not a stop artefact.** The mesh keeps
`--ny`, so the wall cell grows with R (h1 7.5e-3 at 20 D, 3.0e-2 at 80, 6.1e-2 at
160: under 2 cells per D/sqrt(Re)). Time march to a flat cd equals the steady stop:

| structured, 160 D, M 0.1 | n 128 | n 192 | n 256 |
|---|---|---|---|
| `--steady 1e-5` | 1.51051 | 1.50802 | 1.5072 |
| time march, flat | 1.51051 | 1.50800 | 1.50718 |

Differences 2.5e-3, 8.2e-4: order 2 on h^2 (h^2 slope 4.5e-3, 4.2e-3). At n 256:
80 D 1.51119, 160 D 1.5072, the -0.0040 simple shows there (-0.0039). Structured
minus simple `linearUpwind`, n 256: 0.0187, 0.0081, 0.0050, 0.0049 at 20, 40, 80,
160 D.

**Open: that last 0.005 moves with M at large R, not at 20 D.** M 0.1 -> 0.05,
structured: 20 D -1e-4; 160 D -3.1e-3 at n 128, 192 and 256 alike (1.50746, 1.50488,
1.50408), so not a grid effect. n 256 at 1e-5 exit 8 (jitter; cd flat to 5e-5 over
t 20-174). Cause NOT isolated. Do not quote a far-radius limit from M 0.1.

Grid ladder at rFar 80 D, where the boundary gap is smallest: simple n 64 / 128
/ 256: cd 1.52501, 1.51057, 1.50619, order 1.72 from those THREE LEVELS (a
fourth, n 512, 1.50509, gives 1.99 on the last three). Structured: n 64 1.5133
(exit 8, not certified: remaining change 1.6e-5 at the 1e6 cap, TOL 1e-5), n 128
1.51104. The solver gap is +0.0117 then -0.0005: two rungs, one uncertified, a
sign change. NO order is claimed for the difference, and the requirement that
it shrink at the expected order is NOT met on measured data.

**Cylinder Re 40, cut cell.** `--body cylinder --dr 16 --open`, n 128 / 256 /
512 (D 8 / 16 / 32 cells): cd 1.74485, 1.72160, 1.70031; cdf 0.694, 0.635, 0.611.
Differences 0.023, 0.021: first order or worse, still falling, 10% above the
O-grid. A 16 D by 16 D box with the body 5 D off the inflow, a first-order wall:
the domain differs and the gap is NOT closed; n 1024 did not finish inside a
job (cancelled at 890 s). It shows the sign of the trend only.

**Sphere Re 100** (`--ogrid --body sphere`, rFar 20 D, beta 3): wedge 5 deg,
n x nr 64 x 48 / 128 x 96 / 256 x 192: cd 1.0935, 1.0928, 1.0915 (on the circle's qRef; the wedge's planar sector is sin(dphi)/dphi = 0.13% smaller); full revolution
64 x 48 x 32: 1.0870 (its facets carry a frontal area 0.64% below the circle's,
which qRef keeps; the difference from the wedge is that 0.6%). Structured wedge
M 0.1, 64 / 128: 1.0874, 1.0892. Agree within 0.4%, far boundary not separated.

## Not measured

Dropped to keep ladders to 3 levels and jobs under 5 minutes (user decision,
2026-11-01). Earlier ladders with a measured fourth rung keep it; a job above
that limit ran before the decision. Run these for the missing rungs:

```
FV_DIV_LINEAR=1 + U clamped on the far circle (edit 0/U, 0/p of `--export`), n 128 and rFar 40 D, past 40000 it
dns flow --ogrid --body cylinder --physics compressible --n 512 --ny 258 --ogrid-far 20 --ogrid-beta 2 --re 40 --mach 0.1 --steady 1e-5
dns flow --body cylinder --n 1024 --dr 16 --open --re 40 --ft 60 --steady 1e-5
test_gpu_fv_structured ras sa-neg 180 2 768      # and sst-2003m; ny 384 needs > 60000 it
```
