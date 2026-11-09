# Converging-diverging nozzle (`--shock hll --polymesh`)

`dns flow --physics compressible --polymesh DIR --bc FILE --shock hll`: a body-fitted FV
nozzle, planar or wedge, shock capturing on `gpu/fv_comp`, parity with rhoCentralFoam on the
same mesh. Plan and gates: `plans/masterplan_cdnozzle.md`. Open items: issues.md
"Converging-diverging nozzle", Z1 to Z22. Off the route: nothing moved, byte for byte
(`tools/fv/comp_bit_identity.sh`).

```
python3 tools/fv/nozzle_mesh.py --contour nc --nx 320 --ny 64 --mode wedge --dphi 20 \
    --exit riemann --pb 0.05 --out run/n
build/dns flow --physics compressible --polymesh run/n --bc run/n/bc.toml --shock hll --inviscid \
    --mach 1 --ramp 40 --tt 260 --fv-precision fp64 --foam-write 100 --out run/n/dns
```

Units: `--mach 1` makes the reference the stagnation state: rho 1, U = c0, `p_inf = 1/gamma`,
`T_inf = p_inf/R`. `--bc` values are ratios to those, so `p0 = 1`, `t0 = 1`. t in y*/c0 = t*.
`--re` is rho0 c0 y* / mu. `--inviscid`: mu 0, no `--re`, walls `slip`.

`--bc` adds `[nozzle]` (`p`, `t` the rest state, `astar` the mesh's throat area, `axis`),
`type = "total"` (`p0`, `t0`), `exit` on `outflow`, `type = "wedge"` (optional: a `wedge` patch binds
itself). `slip` walls need `force = true` for `Fwall`. Stats append `shockCells byteMax` and, with
`[nozzle]`, `mdotIn mdotOut Fwall Fin Fexit Cd Cf`: `Cd = mdotOut / ideal(p0, T0, A*)`, `Cf = Fexit /
(p0 A*)`, `Fexit`, `Fin` outward patch fluxes at pRef 0, `Fwall` gauge at the rest p. A wedge's numbers
are per wedge, not 2 pi. `--foam-write DT`: `PREFIX_foam/<t>/{p,T,U,rho}`, box units, polyMesh cell order,
17 digits, patches from B. `[thrust]` (`pa`): the impulse ledger and `PREFIX_thrust.csv` ("Thrust").

## Scheme

On internal face f, owner P, neighbour N, `s` the stage state:

    sg = max(b[P-], b[P], b[N], b[N+])          missing ends omitted
    F_k += |S| D_k(sg/255)                      shockFaceDelta, sg > 0
    F_k += eps (1 - sigma) |u_f.S| d_k          P-, N+ exist, sg < 255
    d_k = (s_k[N+] - s_k[P-]) - 3 (s_k[N] - s_k[P])

One pass (`kFvShockFace`, `FvRef::shockFaces`) on the materialized F: one write per face, so
conservation holds by construction. Viscous terms stay beneath. Wall, Slip, wedge and open faces
carry neither term. `sense`: the bytes, no term. eps `kFvShockEps4` 1/16, `--shock-d4` in [0, 0.125].
Label: `Euler` or `laminar`, never DNS: D3 is numerical dissipation everywhere, on a silent run too.

**Sensor** (`dns/fvshock`): raw `shockJump` along mesh lines, once per step on the step's input.
The opposite of a CSR entry is the most antiparallel outward normal, cos <= -0.5, ties to the lowest
`srcFace`. A line counts only if its 5 cells exist: an axis prism has no radial line, rows 0, 1 read their
x line. The structured rule (a blocked read is the centre) reads a first difference at a wall.

**Mesh rule** (`fvUnlinedCells`, `fvShockMeshCheck`; masterplan_combnozzle R8). A cell is unlined when an active face
has no opposite (no other face's outward normal at cos <= -0.5); on a wedge FaceMesh the 5-face axis prisms are exempt.
`--shock hll|sense` on a polyMesh (`dns flow`, `dns flame`) refuses any unlined cell at Acquire, exit 3, before the GPU:
`mesh: --shock needs a mesh line through every face: N of M cells have a face with no opposite (tets, triangles; ...)`.
The library (FvRef, DeviceFvComp) stays capable. Counts (`fvshock_unlined`; analyst scratch meshes by argv):

| mesh | cells | unlined | note |
|---|---|---|---|
| nozzle_A, nozzle_B, nozzle_QA | 1280, 1280, 200 | 0 | hex |
| nozzle_W | 1280 | 0 | 80 axis prisms exempt (80 without the exemption) |
| box_cyclic, box_refined | 24, 276 | 0 | |
| cyl2d_snappy | 1084 | 0 | 8 five-face cells, all lined |
| tet_small (Kuhn tets) | 48 | 0 | 1.67 lines per cell |
| prism75, tet75, tetdual (A1) | 6304, 18912, 5346 | 0, 0, 0 | 1.99, 1.99, 7.40 lines per cell |
| tet_regular (one regular tet) | 1 | 1 | refused: cos -1/3 between every face pair |

So the analysts' premise "a tet has no opposite face, sigma 0, bare KG" is false for the Kuhn split, the union-jack
prisms and their polyDual: every face there has a <= 60 deg opposite and the sensor reads lines (issues.md Z11). The
rule refuses near-regular tets (a Delaunay mesher's), not these.

**D3.** Steady KG has an odd-even mode: bare, source flow L1 p 1.255e-3 at 256, order 1.00, sawtooth 1.11e-3.
The ruled term: 3.25e-5, order 1.98, sawtooth 7.2e-6. lambda is |u.S|: 0 on a wall-normal face.

**Dt.** `internal::shockDtEvery`: default cadence until a cell is sensed, then every step. `--cfl` <= 1.2,
no nDim divide: the Sigma form is the axis sum.

## Boundaries

**`total`** (`FvBcKind::Total`): OpenFOAM's `totalPressure` + `totalTemperature` +
`pressureInletOutletVelocity` fixed point, closed form. `u_P.n >= 0`: u_P, T0, p0. Entering: normal velocity
of P, `T = T0 - |u|^2/(2 cp)`, `p = p0 (T/T0)^(gamma/(gamma-1))`. No viscous flux. It reflects.

**`exit`** on `outflow`: `fixed` (default, the shipped row bit for bit), `extrapolate` (zeroGradient),
`riemann` (`fvExitState`, the half-Riemann problem of the owner against p: shock in or out, fan subsonic,
transonic, supersonic, backflow at `tback`). `riemann` is the production row: per face, so a subsonic layer
under a core takes its own branch. On cone6: rest, p_b 0.01: exit M 3.334 at t* 16, `characteristic` hangs at
0.797. Supersonic IC, p_b 0.5: p_e/p0 0.5001, `extrapolate` stays at 0.0162.

**`wedge`**: one case directory in both solvers, the word `wedge` opt-in (`FaceMeshOptions::wedge`) on this route
only. y = r form: (x, r, +-r tan(h)). A z scale is an exact gauge: 5 vs 20 deg 7.0e-15. The cos form is not
(1.3e-2). Sides are a Slip row the tables know: out of the convective dt (`entS` 0, dt x 12 at 5 deg), hoop term
`V sin^2(h)/d_z^2` in the diffusive one. w exactly 0.0 is load-bearing: the dt excludes its side mode. S_z, Cf_z, d_z,
C_z pinned, the second side the first's bit mirror; `w == 0.0` after 1000 viscous steps in FP32 and FP64.

## OpenFOAM map

| dns | OpenFOAM |
|---|---|
| `total` p0, t0 | p `totalPressure` (`gamma 1.4; psi thermo:psi; rho none`), T `totalTemperature`, U `pressureInletOutletVelocity` |
| `outflow`, `exit = "extrapolate"` | p, T, U `zeroGradient` |
| `outflow`, p, `exit = "fixed"` | p `fixedValue`, T, U `zeroGradient` |
| `slip` / `wall` | U `slip` / `noSlip`, p `zeroGradient`, T `zeroGradient` / `fixedValue` |
| axis, planar | `symmetryPlane`, bound `slip` |
| `wedge` | `wedge` |

`hePsiThermo`, `pureMixture`, `const`, `hConst`, `perfectGas`, `sensibleInternalEnergy`; `molWeight` from OpenFOAM's RR,
calibrated once on p/(rho T). Euler, `Kurganov`; `upwind` T1, `vanLeer`/`vanLeerV` T2, `Minmod` the spread. maxCo 0.2
planar, 0.8 wedge 20 deg: foam's Co counts the sides. Tools: `tools/fv/nozzle_mesh.py` (mesh, `bc.toml`, foam case),
`nozzle_cases.py` (T1 references), `nozzle_ladder.py` (T2, T3), `make_test_data.sh nozzle_*`.

## Parity

**T1 is rounding.** `force` (sigma 1, `FvShock::force`, test only) is HLL + Davis, and from `rhoCentralFoam.C`
foam's `phi` on `upwind` reconstruction is that flux on every row. Host, FvRef at foam's field, max |R| / flux scale: A
5.2e-15, B 5.9e-15, W 1.2e-14; the KG flux reads 0.077, 0.115, 0.085. Device, from the q1d IC, 67951 steps, every
cell p, T, U vs foam: A 1.23e-14, B 3.42e-13 (a real shock, p ratio 2.27 across a face), W 20 deg 1.47e-14; exit mdot
6.9e-15, 2.2e-16, 2.0e-15; gamma 1.2 5.3e-14. So mesh, gas, flux, inlet, exit, wall and wedge side are the same
operator, shock included. Foam's own drift at those references: 6e-15.

**T2 is a gap on time means.** rhoCentralFoam `Kurganov` `vanLeer` from its first-order field, `fieldAverage` over t
0.65 to 0.85 s (68 t*), against dns hybrid + D3, rest + ramp 40 t*, FP64, 21 snapshots over the last 87 t*. One
post-processor reads both (`nozzle_ladder.py report`). `Cf` on the exit row. vL vs Mm is foam's own limiter spread.

| case, level | vol L1 p | wall L1 p | exit axis M | mdot | Cf | x_s |
|---|---|---|---|---|---|---|
| A 160x32 | 1.61e-3 | 1.38e-3 | 7.5e-4 (2.1844 / 2.1837) | 1.18e-4 | 3.5e-4 | |
| vL vs Mm | 1.02e-3 | 4.5e-4 | 3.6e-4 | 2.3e-4 | 2.7e-4 | |
| A 320x64 | 1.07e-3 | 9.6e-4 | 1.6e-3 (2.1831 / 2.1847) | 4.7e-5 | 2.4e-4 | |
| vL vs Mm | 4.0e-4 | 3.5e-4 | 2.8e-4 | 3.7e-5 | 9.7e-5 | |
| B 160x32 | 3.84e-3 | 3.32e-3 | 9.6e-3 (0.3587 / 0.3683) | 2.6e-4 | 3.6e-5 | 0.18 h |
| vL vs Mm | 1.13e-3 | 7.8e-4 | 1.2e-2 | 2.3e-4 | 5.1e-4 | |
| B 320x64 | 2.37e-3 | 1.99e-3 | 1.4e-2 (0.3503 / 0.3643) | 1.5e-4 | 2.6e-4 | 0.41 h |
| vL vs Mm | 5.3e-4 | 4.4e-4 | 4.0e-3 | 3.2e-5 | 1.9e-4 | |

Two-level order: A vol 0.59, wall 0.53; B 0.70, 0.74. **Under 1: G23 UNMET on order.** The budget: A at 320 reads wall
gap 3.2e-4 to 1.3e-3 up to x 1.5, and a flat +2.1e-3 past it, dns above foam, at both levels. It is foam's: on A in the
cone x > 1.5, p0 error foam -1.61e-3, -9.66e-4 (order 0.74), T0 error -3.4e-4, -2.4e-4; dns -3.4e-4, -6.9e-5 (order
2.3), T0 5.5e-6, 1.7e-6, -1.8e-5 at 640. Inside x < -1.5 the dns window's own halves differ 3.5e-4, 4.9e-4: the
chamber ring (Z17). Why foam loses p0 and T0 this way is not proved. 640 foam was cut (budget); dns alone: Cd 0.995697,
0.995727, 0.995741, Cf 1.440293, 1.440341, 1.440369. B's axis exit M: dns 0.3587, 0.3503, 0.3618: not monotone (Z5).

**T3, laminar, is not settled and says so.** Re* 1e4, adiabatic, planar, 160x48, 320x96, tanh 2: delta99 17, 36
cells (foam 18, 35), adiabatic heat exactly 0.0 in every row. Cd gap 2.4e-4, 2.5e-4 (0.97980 / 0.97956, 0.97961 /
0.97986), wall L1 p 1.1e-3, 2.0e-3, mean tau gap 16%, 10%, exit axis M gap 4.9e-3, 1.7e-3, Cf 3.8e-4, 3.0e-5. Wall p
rises: **G28 UNMET.** Budget: at 320 the wall gap in the chamber is 1.2e-3 and dns's own window halves differ 1.1e-3:
the ring (Z17) against foam's 17 t* mean window, the most the foam budget paid for. The exit must be foam's: `riemann`
at p_b 0.05 on the layer's subsonic faces moved wall p 14% in the last cells; `extrapolate` 1.6e-2 at x 3.9. D3's shift,
eps 0 vs 1/16 at 160: mean tau 0.09%, Cd 5.6e-5, wall L1 p 1.7e-3. Isothermal T_w/T0 0.5: **G29 UNMET**: foam's t 0.22 (76
t*) is not settled: Cd 0.99982 where dns reads 0.993 at t* 70 and 0.9889 at 140. Wedge 5 deg, the same short window:
wall L1 p 2.1e-3, 3.1e-3, Cd 0.96720 / 0.96862, 0.96718 / 0.96944, delta99 24, 48 cells both; foam's vanLeer stage dies at
maxCo 1 (sigFpe), runs at 0.3.

**Anchors**, no foam: source flow, Sauer, Cd over R: `docs/validation.md` "Nozzle".

Best run end to end: `nc` 640x128 planar, rest + ramp, total + riemann, Cd 0.99574, Cf 1.44037, exit axis M 2.1830; B
p_b 0.6: Cd 0.99572, Cf 1.57464, axis shock at x 3.074. Foam at 320: A Cd 0.99568, B 0.99585, x_s 3.082.

Cost. Dispatches per step, `tools/fv/shock_cost.sh`: none 18.4, sense 20.4, hll 29.0 hot (+58%; band +60%). `--bench`,
20480 cells, FP32, a hot start: none 0.126, sense 0.131, hll 0.210 ms/step (+67%). Launch-bound: the dt every step and three
face passes. The nozzle itself, mostly cold, 0.142.

## Thrust

`[thrust]` in `--bc` (`pa`, required, >= 0, a ratio as `[nozzle] p`; `station`: `dns flame` only, required there,
the burnt stagnation plane along the axis; needs `[nozzle]`) turns on the impulse ledger and writes `PREFIX_thrust.csv`, 17 digits, one row per stats row. Absent: no
launch, no buffer, no file, `_stats.csv` byte for byte. `--checkpoint-dt`, `--restart`, `--wall-limit` with it: exit 2
(the ledger is not checkpointed).

**Ledger** (`DeviceFvComp::setLedger`, `FvLedger`, FvRef's twin `march(..., &ledger)`). Per stage, after the fluxes,
per patch and state row: `S_k = sum_f F_f` over the patch's materialized boundary faces (FP64, one block per patch,
thread partials in ascending face order, a shared-memory tree; no atomics), and `sum_c V omega`. At the step's end,
one thread: `I += (w0 S_0 + w1 S_1) + w2 S_2` with the weights the stage writes used (`fv_comp.hip` `step`):
`w0 = w1 = cb2 (cb1 dtR)`, `w2 = R(cb2 dtR)`, `cb1 = 1/4`, `cb2 = R(2/3)`, `ca + cb = 1` exactly. The add is
compensated (`fvLedgerAdd`, Knuth's two-sum, issues.md N16): `I`, `Omega` stay the plain FP64 sums bit for bit, a low
word carries each add's rounding, every read is `hi + lo` (`FvLedger::at`), so the ledger's own error is one rounding
at any step count (`gpu_fv_nozzle_ledger` N16: 1e5 steps 3.9e-15 flat; the plain hi word 2.6e-13). So per row k

    P_k(t) - P_k(0) + sum_roles I_k = q Omega [E] - Omega [fuel]      to rounding, on any mesh

`P_k = sum_c V q_k` (hi + lo, Neumaier). Roles by BC kind: in (total, inflow, far), exit (outflow), side (`wedge`
rows, a slip on a `symmetryPlane`/`symmetry` patch: the planar axis), wall (every other wall, slip).

**Columns.** `t step dt`, `P_{mass,mx,my,mz,E,ry}`, `I_{in,exit,wall,side}_{...}`, `Omega`, `res_{...}` (the identity's
left side minus its right), then `Fmom Fpres Texit Twall exitSuper mdotIn mdotOut CfVac Cf cstar`:

    Texit = sum_exit (F - pa S).a                   sampled, the current state
    Twall = -sum_{in, wall, side} (F - pa S).a - (sum_c V dq/dt).a      the same number by the gather's telescoping
    window  Tvac = Delta I_exit.a / Delta t         since the previous row: sample-free
    CfVac = Tvac/(p0 A*), Cf = (Tvac - pa sum_exit S.a)/(p0 A*), cstar = p0 A*/mdotOut, mdotIn/Out window means
    Fmom = sum_exit mdot_f (u_B.a), Fpres = sum_exit (p_B - pa) S.a; exitSuper = exit area fraction u_B.n >= c_B

With `station` (the reacting nozzle, masterplan_combnozzle 2.3) the row appends `p0b T0b CdB CfBvac CfB`: over the
internal faces whose owner and neighbour centroids straddle the station, the face state is the mean of the two cells'
primitives, `T0 = T + |u|^2/(2 cp)`, `p0 = p (T0/T)^(gamma/(gamma - 1))`, both weighted by `|rho u.S|` (`fvStation`,
host, off the row's state); `CdB = mdotOut / (Gamma p0b A*/sqrt(R T0b))`, `CfBvac = Tvac/(p0b A*)`, `CfB` at `pa`, with
the window's `mdotOut`, `Tvac`; then `Mst Mth` (masterplan_combclose 2.2, `fvStorage`): `sum V rho` over the cells with
`a.C > station` and over the cells past the throat bin (`fvPastThroat`: the probe bin past `fvThroatBin`, the IC's rule),
Neumaier in ascending cell order of the downloaded state. A window's throat flux `mdotOut + Delta Mth/Delta t` is exact
and sample-free (no station, `dns flow`: no columns, bytes unchanged).

`Cf` at `pa` is ambient-valid only while `exitSuper == 1`; else it is ambient-dependent and needs the plume box (Z15).
The CV is the whole mesh; the plume box's exit plane will be a faceZone (OpenFOAM `surfaceFieldValue` integrates the
same faces), never a cell-centre sum. Any window is `Delta I / Delta t` of two rows.

| gate | rig | measured | band |
|---|---|---|---|
| G2.1 | FvRef, random states, `sum_c V R + sum_bnd F - src` / `sum |F|`, every row | box_cyclic 2.4e-17, box_refined 5.6e-18, cyl2d_snappy 2.9e-18, nozzle_A 1.9e-17, nozzle_W 7.3e-16, q1d 2.3e-16, q1d + C0 1.2e-16 | 1e-14 |
| G2.2 | device FP64, rest + ramp 40, 1000 steps through it: max res_k / S_k | nozzle_A 3.8e-15, nozzle_W 3.7e-15, q1d 4.6e-15; Texit - Twall 1.0e-14, 5.1e-15, 3.3e-14 | 1e-12 |
| G2.2 ctrl | weights 1/3 each | 1.1e-5, 1.3e-5, 2.4e-5 | >= 1e-6 |
| G2.2 | FvRef ledger vs device, 200 steps | 1.2e-16, 2.1e-16, 9.3e-17 | 1e-13 |
| G2.10 | nozzle_A 80x16 through the CLI, every `_thrust.csv` row; window [140, 240] Cf | res 6.5e-15, Texit - Twall 1.8e-16; Cf 1.439005 | [1.43700, 1.44315] |
| L1 | FvRef march 400 steps, inert (hll) and C0 (KG), res / S | 6.7e-16, 5.2e-16 | 1e-12 |

## Quasi-1D

`nozzle_mesh.py --ny 1` (planar): one row of cells, i lines vertical, `wall` on top, `axis` (symmetryPlane) below. It
IS the quasi-1D FV scheme: the wall's `p_P S` is `p dA`, the x faces' S_y is exactly 0.0, and the wall's S_y and the
axis's cancel per cell to an ulp (not bitwise: OpenFOAM's fan on the slanted quad, 2e-19 of 4e-3). So from v = 0 the
march keeps |v|/c at rounding, 1e-17 to 2e-16 (G2.3b; a 1e-9 axis node shift gives 6.7e-12). Closed oracles: area-Mach,
the quasi-1D normal shock (`p_b` sets A_s), `C_F,vac = Gamma sqrt(2g/(g-1)(1 - (p_e/p0)^((g-1)/g))) + (p_e/p0) eps_e`.
Rig (`gpu_fv_nozzle_q1d`, `_q1dshock`, `_ambient`, `_twin`): the generator's q1d IC with v 0, total + riemann, FP64,
hll eps 1/16, every number a window mean from the ledger over the last quarter of tt; A* the contour's (y* 1 x depth),
M_e and p_e from the exit fluxes (`fvExitMach`).

| gate | case | 200 | 400 | 800 | order | band |
|---|---|---|---|---|---|---|
| G2.4 | parabola R 20, eps 1.4: \|Cd - 1\| | 3.82e-6 | 9.80e-7 | 2.48e-7 | 1.96, 1.98 | >= 1.8 |
| | \|CfVac - 1.3794635\| | 5.40e-6 | 1.36e-6 | 3.27e-7 | 1.99, 2.05 | >= 1.8; 1e-5 at 400 |
| | \|M_e - 1.7632052\| | 7.54e-7 | 1.39e-7 | 2.16e-8 | 2.44, 2.69 | >= 1.8 |
| ctrl | `force` sigma 1 CfVac | 1.90e-3 | 9.53e-4 | 4.78e-4 | 0.99, 1.00 | fails: >= 1e-4, <= 1.2 |
| G2.5 | nc A, eps 2: CfVac - 1.457600 | -3.42e-5 | -8.68e-6 | -2.12e-6 | 1.98, 2.03 | 2e-4 at 400, >= 1.5 |
| | Fin + 2.595093 | +3.70e-6 | +1.01e-6 | +3.15e-7 | | reported |
| | wall - 1.137493 | +3.05e-5 | +7.67e-6 | +1.81e-6 | | reported |
| | p_e/p0 - 0.093933 | +5.3e-7 | -1.6e-7 | -3.1e-7 | | reported |
| G2.6 | nc B, p_b 0.6: x_s - 3.08380 | -3.55e-3 (0.07 h) | -1.92e-3 (0.08 h) | -1.40e-3 (0.12 h) | 0.88, 0.46 | <= 1 h; **>= 0.9 UNMET** |
| | Fexit - 1.574057 | -1.32e-5 | -3.19e-6 | -5.88e-7 | 2.05, 2.44 | 5e-4 at 400 |
| | wall - 1.021036 | +9.47e-6 | +2.17e-6 | +2.73e-7 | | reported |
| d4 0 | x_s; Fexit (R6 fallback evidence) | 3.08016; 1.5740346 | 3.08190; 1.5740521 | 3.08241; 1.5740563 | | reported |
| ctrl | p_b 0.606 at 800: x_s moves | | | 6.06e-2 (43x the error) | | fails |

x_s is the linear crossing of (p1 + p2)/2 (the oracle's M1 2.04716), settled from t* 50 at every rung; its sub-cell
offset is not monotone in h, the post-shock integrals converge at order 2 (issues.md N7). G2.7 (nc A 200, 20000
steps at a fixed dt, riemann at p_a/p0 0.01, 0.093933, 0.45): state bytes and every row column but the p_a ones
equal; Cf - (CfVac - (p_a/p0) eps_e) 2.2e-16, 1.1e-16; exitSuper 1 at every sample; `exit = "fixed"` at 0.45 diverges
(the bytes break).

**The react twin** (G2.9; `--contour react --lc 0`, eps_c = eps_e 6.25, P4's nozzle half in nozzle units): CfVac vs the
oracle 1.608145 (gamma 1.4), 1.678517 (gamma 1.2), FP64, riemann p_b 0.005, tt 160:

| gamma | 400 | 800 | 1600 | 3200 | orders |
|---|---|---|---|---|---|
| 1.4 | -4.26e-4 | -1.33e-4 | -4.32e-5 | -1.45e-5 | 1.68, 1.63, 1.57 |
| 1.2 | -4.25e-4 | -1.34e-4 | -4.36e-5 | -1.45e-5 | 1.66, 1.62, 1.59 |

The band (1e-4 at the coarse rung, falling) holds from 1600 (h 0.01; P4's xi grid is h ~ 1.1e-3); 400 and 800 miss it
(recorded). The error is Cd's (2.8e-4 at 400): the cosine throat's curvature radius is 0.62 y* upstream, 5.6 down (C1).
Control: the gamma 1.2 oracle against the 1.4 run, 4.22%.

**T1 on one row** (G2.8, G2.9): `force` sigma 1 from the q1d IC to t 2.4 s vs rhoCentralFoam upwind on the same
directory, every cell p, T, u (v on the speed scale: both are rounding fields): `nozzle_QA` (nc 200) 4.9e-14, mdot vs
foam 3.1e-15, 104442 steps; `nozzle_RT` (twin 400) 2.4e-14, 1.8e-14, 132768 steps. Controls: the route's own flux (KG +
D3) 8.97e-3 (the plan's >= 1e-2 UNMET by threshold, replaced by >= 1e-3), gas R + 1e-6 1.0e-6.

## Reacting

`dns flame --polymesh DIR --bc FILE --shock hll` (compressible, premixed C0; masterplan_combnozzle P3): Shock x C0 is
live (`dns/fvmodules`). No kernel was edited to open it: `faces()` runs `kFvShockFace` after the KG faces, `speciesFaces()`
reads the FINAL `F[0]`, so `F_Y = Y_f(sign of the final mdot) F_rho(final)` (KG + HLL + D3), never the structured `D
Y_f(sign m_KG)`. Under `hll` the `--bc` file takes `type = "total"` with `y` (required: the fresh mass fractions),
`[nozzle]` with `astar` (and `axis`; `p`, `t` exit 2: the flame starts from its composition IC), `[thrust]` with `pa`
and `station` (required on the flame; its columns `p0b T0b CdB CfBvac CfB`, "Thrust"). `wedge`: exit 2 (next cycle). The stats
append `shockCells byteMax` and, with `[nozzle]`, the nozzle columns; `[thrust]` writes `_thrust.csv` from the flame
runner as from the flow's. `--shock-d4` on the flame: hll only, [0, 0.125]. C1, C2b, `--equivalence`, TFLES, LES,
`--turb` with `--shock`: refused by name. Label:

    laminar reacting [premixed single-step C0, calorically perfect gamma G, Le 1, M_flame M (inflated)] species:
    MC-limited MUSCL on the FINAL face mass flux (KG + HLL + D3) | FV path, ... [shock capturing: ... background D4
    eps E |u.S|: NUMERICAL dissipation everywhere, not DNS; ...]

| gate | rig | measured | band |
|---|---|---|---|
| G3.1 | `gpu_fv_matrix` cell C0 + shock, 4 meshes | FvRef 4.5e-15, f32 3.3e-6, numbering 3.5e-16, mass 2.2e-16, E + q rho Y 6.7e-16, pow2 Y exact, passive species 40 steps bitwise, rerun identical | matrix rules |
| G3.1 ctrl | host: one forward-Euler update of uniform Y 1/4 from mutated face fluxes (`fvref_shockspecies`) | "D3 on rho Y itself" bitwise (DEAD: d(rho Y) = d(rho)/4 exactly); replaced once by "species on the KG mass flux": broken | must break |
| G3.2 | `fvref_shockspecies`: 200 cells, 200 steps, cfl 0.8, force sigma 1 / hll | Sod Y in [-2.0e-17, 1 + 2.2e-16], contact 1.69 / 0.94 cells; Mach 3 piston Y in [-2.9e-19, 1 + 4.4e-16] | [-1e-14, 1 + 1e-14]; 2 cells |
| G3.2 ctrl | `D Y_f(sign m_KG)`, sigma 1, same ICs (`shock_ref.hpp` Reuse) | piston min Y -1.08e-2; Sod 0.0 (reported) | < -1e-2 |
| G3.3 | poly1d hll vs d4 0, [7, 14] | see docs/fvcomp.md "Species cross-verification": Delta order 3.1, 3.0 at both Machs, SZ defect hll <= d4 0, shockCells 0 | 1/3 of the d4-0 error, order 1.8 |
| G3.3 ctrl | box 7.5 cells M 0.042, force sigma 1, [2, 4] | S_L 1.631 vs 1.010 (shift 0.62) | >= 3 x band (1.8e-2) |
| G3.4 | hll d4 0 vs none | bytes (15 cells both Machs; 7.5 box state; 7.5 poly1d through the binary) | bytes |
| G3.5 | gamma 1.2, M 0.042, 15 cells, hll | S_L 0.9982783 (P4's design input) | reported |

**R6's fallback did not fire**: D3 moves S_L by 3e-3 at 7.5 cells and at order 3 (h^3: a fourth difference on the
mass flux), below a third of the scheme's own S_L error at every rung, and leaves the Shvab-Zeldovich defect at or
below the d4-0 value (ratio 0.82 to 1.00). `--shock hll` keeps eps 1/16 on the flame route.

### The reacting quasi-1D nozzle (masterplan_combnozzle P4)

A Total fresh inlet (`y = [1]`), a constant-area chamber holding a planar C0 flame, a choked cosine nozzle, a riemann
exit, on ONE row (`nozzle_mesh.py` planar layout, so the scheme is quasi-1D). `[nozzle]` on `dns flame` starts the run from
the **composition design** (`dns/fvnozzle` `fvReactingNozzleDesign`, Python twin `tools/fv/nozzle_react.py`, equal to
the last bit on the rig, `fvref_react` R1):

1. fresh: `T_u = T0u/(1 + (g-1)/2 M_u^2)`, `p_u = p0u (T_u/T0u)^(g/(g-1))`, `u_u = M_u c_u`, `m = rho_u u_u`;
2. constant-area complete burn (Rayleigh-Hugoniot, `p + m u` and `cp T + u^2/2 + q` conserved), the subsonic root by
   bisection on `u_b` in `[u_u, (P/m) g/(g+1)]`; `T0b`, `p0b` of the burnt gas;
3. choke: `m eps_c = Gamma p0b/sqrt(R T0b)`, bisection on `M_u` in `[1e-6, 0.3]`; exit: the supersonic area-Mach root at
   `eps_e`, `CFvac = Gamma sqrt(2g/(g-1)(1 - (p_e/p0b)^((g-1)/g))) + (p_e/p0b) eps_e`.

`q` is the SOLVER's `cp (sigma - 1) Tu` on the box's static `Tu`, so `T0b = T0u + (sigma - 1) Tu` exactly, never
`sigma T0u` (2.8e-4 off at M_u 0.042). Inputs: the Total row, `[nozzle] astar`, `eps_c = A_ref/astar` off the chamber probe,
`eps_e = sum_exit S.a/astar`, gamma, cp. Errors (exit 3): no throat (`eps_c <= 1`), no choke root, a thermally choked
chamber. The IC (`fvReactingNozzleIC`, masterplan_combclose 2.1): the chamber slabs fresh -> burnt on the **Rayleigh line**
of the design's `(m, P, H)`: `p + m u` and `cp T + u^2/2 + q (1 - Y)` conserved (`fvRayleighBurnt` at `q (1 - Y)`), Y =
`(sigma - t)/(sigma - 1)` of the oracle profile with the flame at `--upstream` from the inlet; Y 0 is the design's burnt
state bit for bit (`FvReactingDesign` echoes g, R, cp, q). Every later slab burnt and isentropic at `(p0b, T0b)`,
subsonic to the least-area slab (`fvThroatBin`), supersonic past it (`fvref_react` R4: T0 and entropy per cell to 1e-15;
R5: every chamber cell on the line to 4e-16, T0 continuous into the contraction). Not `premixedPoint` (low-Mach in
energy, T = sigma Tu): its burnt chamber sat +1.7094e-3 over T0b, an entropy slug that crossed the throat at t_e ~ 40 tau
inside every gate window (issues.md N13, N15). The oracle table ends at Y 1e-9 (sigma - 4e-9), so the burnt chamber
carries Y 1e-9 (T0 8.0e-10 under T0b, cp T0 + q Y continuous to 2e-16); G4.12a reads the station's first row 2.7e-10 off
T0b (`gpu_fv_nozzle_react`). `--inflow-speed` with `[nozzle]`: exit 2
(`--inflow-speed: the throat sets the fresh velocity on a [nozzle] run`). The probe reads A_ref and the chamber bins
(docs/fvcomp.md "Stats"); `_thrust.csv` appends the burnt station's `p0b T0b CdB CfBvac CfB Mst Mth` ("Thrust").

**The report's window rule (masterplan_combclose 2.4; supersedes A24's peak list).** `nozzle_react.py modes DIR` is
the linear quasi-1D network (Total inlet, compact moving flame, the exact cosine contraction to M 0.995, throat
regularity): on the 15-cell rig mode 1 P 22.680 tau, e-fold 33.80; mode 2 9.133 / 10.78; the entropy transit x_f0 ->
throat t_e 40.207 tau (chamber 6.134, contraction 34.073), t_flush 43.03; controls p-node inlet e-fold 57.70, a compact
choked nozzle at the chamber's end P 7.283 (the contraction is non-compact). `report` fits `uFresh` on [t_s, tt], t_s =
t_e + 10, with a damped cosine + quadratic (accepted at resid/amp <= 0.05, |P/P_model - 1| <= 0.1; else P_model) and
gates W = [tt - kP, tt], k = floor((tt - t_s)/P) >= 1 (else exit 1); every rate is an endpoint ledger difference over
the same rows; G4.3 on endpoint x_f; G4.5 on the throat flux (exit and station forms beside); G4.6 mdot on the fuel
inventory's jump velocity with the lab-frame fresh state, split exactly into burn + storage; G4.11 the fit vs the model;
G4.12 the IC's T0b on the first row and the slug's 4-tau dip about t_e. The legacy [20, 80] and [40, 80] print beside.
On the archived old-IC runs (K3): fits p15 22.417 / 31.66, p30 22.452 / 32.02, q15 [50, 140] 22.382 / 31.95 (G4.11
pass); rule window p15 [57.58, 80] dCd (exit form) -4.1e-6, mdot -7.8e-6; p30 -1.7e-6, -3.8e-6; q15 k 4 +5.2e-7,
+2.38e-5; the same runs' legacy [20, 80] mdot -2.20e-4, [40, 80] dCd -1.10e-4; G4.12 a +1.709e-3, b -6.99e-4 / -6.94e-4
/ -7.08e-4 (all fail as required: the slug).

**The rig** (`nozzle_react.py design --cells 15 --mach 0.042 --sl 0.9945059`): y* 49.97 delta_D, chamber Lc 1 y* (50
delta_D: the flame at 12 delta_th, 19 delta_th burnt after it), cosine contraction Lcon 6 y* (54 deg max), cosine
divergent Ldiv 4 y* to eps_e 2, h = delta_th/15, 5148 cells (nx a multiple of 11: the twin is the same xi grid from
the contraction start), A* = y* x depth, `pa` 0.01 Tu, station Lc - 2 delta_th, FP64, `--shock hll`, tt 80 tau,
window [40, 80] off the ledger and the probe. eps_c 6.1729 solves `u_u = S_L,num` (P3's 1D 15-cell hll S_L 0.9945059).
Not the plan's Lcon 4 y* (64 deg: R8 refuses 1400 cells, the one-row axis face then has no opposite), eps_e 6.25 (the
exit density sets the viscous dt to 2.5e-5 tau in flame units: 3.1e6 steps), 0.98 S_L (below). 266 s per 80 tau
(0.316 ms/step, 842 393 steps, dt 9.5e-5 viscous-bound).

**The flame is neutral in position; the throat holds the BURNT volume flow.** Kinematically `x_f' = u_u - S_c`, but the
flame jump `rho_b u_b = rho_u S_c + rho_b x_f'` with `rho_b u_b A_c` fixed by the throat gives `x_f' ~ sigma (u_d - S_c)`:
five times the fresh mismatch. The open box's sliding 0.98 S_L drifted -0.1056 S_L (predicted -0.1070), 8.4 delta_D in
80 tau, until its preheat reached the reference slab. At `u_d = S_L,num`: -0.0040. Started 2 delta_th further
downstream, the flame stays 3.20 delta_D (2 delta_th) downstream to the end, and every window number agrees (G4.7).

| gate | relation | measured (15 cells) | band |
|---|---|---|---|
| G4.1 | ledger res_k / (\|P_k\| + sum \|I_k\| (+ q\|Omega\|, \|Omega\|)), every row of 4212 | 8.7e-13 (grows with steps: 2.5e-13 at t 10; the start run 1.08e-12 at t 80); 30 cells **3.51e-12 UNMET**: the FP64 state's rounding (+7e-15 res_mass per step at both h), not the ledger (compensated, P4b) | 1e-12 |
| G4.1 ctrl | Omega dropped from the rho Y row | 0.45 | fails |
| G4.2 | S_c (A_ref) vs the 1D S_L,num 0.9945059; delta S_fresh -5.9e-5 <= band/4: raw | 0.9944148: -9.1e-5 | 5e-4 |
| G4.3 | x_f' (regression) vs u_u - S_c | -0.0040060 vs -0.0036810: 3.3e-4 (the 0.98 run: -0.10557 vs -0.10466, 9.1e-4) | 2e-3 S_L |
| G4.3 ctrl | S_c on vol/L (A_ref/(vol/L) 2.0154) | gap 1.01 | fails |
| G4.4 | T0b vs T0u + (sigma - 1) Tu; fuelOut / burnIn (ledger) | -9.2e-7; 0.0 | 1e-4; 1e-8 |
| G4.4 ctrl | sigma T0u | +2.8e-4 | fails |
| G4.5 | CdB, CfBvac vs the viscous inert twin (Re* 736.26, Pr 0.72, same xi grid, q1d IC, `gpu_fv_nozzle_reacttwin`) | Cd 0.9989657 vs 0.9990755: **-1.10e-4 UNMET**; Cf 1.4546195 vs 1.4546618: -4.2e-5 | 1e-4 |
| G4.5 ctrl | Cd on T0u | -0.552 | fails |
| G4.6 | p0b/p0u - 1 vs the oracle at M_u = S_c/c_u (+ the drift's Galilean term -1.6e-8) | -4.91166e-3 vs -4.91536e-3: 3.7e-6 | 4.9e-4 |
| G4.6 | mdot: (mdotOut - rho_b x_f' A_c) vs rho_u S_c A_c | **-1.37e-4 UNMET** (stationary form -9.4e-4) | 1e-4 |
| G4.6 ctrl | no heat release (p0b = p0u) | 4.9e-3 | fails |
| G4.7 | x_f + 2 delta_th start: window mdot, CfBvac, S_c | 4.4e-5, 2.2e-6, 3.8e-7 relative (bands: 2 x span 1.4e-2, 6.3e-4, 1.4e-4) | 2 x span |
| G4.8 | shockCells | 0 on every row | 0 |

The twin's own numbers: inviscid CfVac 1.4575996 vs the closed oracle 1.4576001 (-5.1e-7), Cd 0.9999996; the viscous loss
at Re* 736: Cd -9.24e-4, CfVac -2.94e-3 (physics: the nozzle Re in flame units is ~ (A_c/A) L/delta_D, R5). The viscous
one-row v is NOT a rounding field: max |v|/c 6.9e-3 in the twin (the slip wall's least-squares gradient carries the
normal stress into y); P_my/P_mx 8.7e-4 in the flame run. The ring: a persistent thermo-acoustic oscillation of the
Total-fed choked chamber, u_u rms 1.0e-3 to 2.6e-3 and p0b rms 2.3e-5 to 4.6e-5 per 20 tau window, not growing over 80
tau (issues.md N14); the gates read ledger window means.

**The 30-cell rung** (P4b, masterplan_combnozzle A24; `design --cells 30 --sl 0.9918456870897155`, eps_c 6.18958, 10307
cells, 3.37e6 steps, 1398 s). The ring is the SAME at 30 cells (p0b rms 2.8e-5, peaks 20, 30, 15, 60 tau): a physical
oscillation of the chamber, not the scheme's. So the window is fixed before the run to **[20, 80]** (60 tau, an integer
number of each peak; `nozzle_react.py report --from 20`), the plan's [40, 80] beside it:

| gate | 15 [20, 80] | 30 [20, 80] | 15 / 30 [40, 80] | band |
|---|---|---|---|---|
| G4.2 S_c - S_L,num | -9.32e-5 | -7.38e-5 | -9.11e-5 / -7.15e-5 | 5e-4 |
| G4.3 | +6.6e-5 | +5.8e-5 | -3.25e-4 / -2.99e-4 | 2e-3 S_L |
| G4.4 T0b | -1.46e-6 | -1.68e-6 | -9.2e-7 / -1.10e-6 | 1e-4 |
| G4.5 CdB - Cd_twin | -9.91e-5 (marginal: the twin's own drift 1.7e-5) | twin cut (13 min GPU) | -1.10e-4 / cut | 1e-4 |
| G4.6 p0b | +2.9e-7 | -1.2e-7 | +3.7e-6 / +3.3e-6 | 4.9e-4 |
| G4.6 mdot | **-2.25e-4** | **-2.23e-4** | **-1.37e-4 / -1.41e-4** | 1e-4 |

The flame half does not move with h (CdB 0.998976 -> 0.998977). G4.6's mdot does not fall (order ~0): the composition
closes (the ledger's burn rate vs rho_u S_c A_c +7.5e-6), A19's drift storage `(rho_u - rho_b) A_c x_f'` over-states the
ledger's exact chamber mass change by 6% at both rungs (issues.md N15). Cut: the eps_c 13.2 row, the gamma 1.2 row (G4.9;
its dead "off >= 4%" control replaced by "CfBvac vs the gamma-1.4 oracle >= 1e-3", not run). The reactingFoam `transonic
yes` scout (G4.10) did not settle in 5 CPU min (7.8 of 80 tau, mdotIn/Out 1.6% apart): stopped.

## Not built, refused

`--steady` (LTS on `Total` stalls on the host, Z7), LES, RANS, two-reactant species (C1, C2b), gravity, body force,
`--turb`: refused by name (`dns/fvmodules`, axis `capture`). Plume box (P7), 2D / wedge reacting nozzles (P6), viscous
walls (P7), Shock x C1 / C2b (P8), gamma(T) (P9): next cycle. Structured D3: no. `[thrust]` with checkpoints: refused
(the ledger is not checkpointed); `[thrust].station` on `dns flow`: refused (the reacting nozzle's burnt station).
`--inflow-speed` on a `[nozzle]` flame: exit 2 (the throat sets u_u). A `[nozzle]` flame whose mesh has no throat (eps_c
<= 1), no choke root or a thermally choked chamber: exit 3 by name (`fvReactingNozzleDesign`). The ledger refuses gravity, a body force, RANS and local time stepping (their
sources are no boundary flux).

## Bent invariants

CLAUDE.md is not edited. For the user:

- "The reader accepts exactly what the OpenFOAM mesh tools write; everything else REJECTED BY NAME": `wedge` opens on
  one route by opt-in; every other route keeps the text.
- "No NSCBC, no sink on a FaceMesh" stands for the flame box. D3 is a mean-flow numerical term on the nozzle route, on by
  default under `hll`; it uses no freestream reference, so it does not justify a sponge.
- "Mean flow gets NO added dissipation" is RANS's. RANS with `--shock` is refused.
- "gamma = 1.4 FIXED": `--gamma` on the polyMesh route only, constant, default 1.4. Since masterplan_combnozzle P1
  also on `dns flame --polymesh` (compressible premixed C0 only): one gamma for fresh AND burnt gas, frozen; Tu =
  1/(gamma M^2), q = cp(gamma) (sigma - 1) Tu; Lambda gamma-free. Gate: the burnt T offset (docs/chem.md "P1").
- "Bit-identical silent `hll` == `none`" is the structured rule; on FV it holds at `--shock-d4 0` only.

## Proposed CLAUDE.md invariants

- **FV `--shock hll` is one face pass on materialized F**; `force` sigma 1 is rhoCentralFoam Kurganov + upwind to
  rounding, on every cell, shock included.
- **The FV sensor reads whole mesh lines only**: a blocked read as the centre fires on a smooth wall (15 cells at 100x20
  on the dns field, 0 with whole lines).
- **Steady KG carries an odd-even mode, first order on a wall**: D3 along mesh lines at lambda |u.S| removes it; lambda
  with c costs a layer +19%.
- **A `wedge` is y = r, sides bit mirrored, w exactly 0**: its side mode is outside the dt.
- **A nozzle's exit is the half-Riemann row.** `Far` at a subsonic duct exit is a radiating far field (M_e 0.4: p_b/p_a 1.79).
- **Compressible walls carry no D.**
- **`--gamma` on FV polyMesh routes is one constant gamma for fresh and burnt gas**; Lambda is gamma-free; the burnt
  T offset -(sigma^2 - 1)(gamma - 1) M^2/(2 sigma) is its gate (P1 G1.7: D +4.81e-5 / +4.75e-5 vs 4.8e-5).
- **Thrust is the impulse ledger**: FP64 sums of the materialized boundary F per stage, weights `cb2 (cb1 dtR), cb2
  (cb1 dtR), R(cb2 dtR)` read from the stage coefficients. The exit form equals the wall form by the identity; window
  thrust = Delta I / Delta t. Never a cell-centre sum, never a sampled mean.
- **A one-row planar FaceMesh is the quasi-1D FV scheme**: v a rounding field (|v|/c <= 1e-12; not 0.0: the wall's and
  the axis's S_y differ by an ulp); area-Mach and the normal shock are its closed oracles.
- **`--shock` is refused on meshes with unlined cells** (wedge axis prisms exempt); Kuhn tets and union-jack prisms
  are lined.
- **Under `--shock` species ride the FINAL F[0]** (species faces after the shock face pass); never upwind on the KG
  mass flux.
- **A premixed flame in a constant-area chamber ahead of a choked throat is neutral in position**: M_u is set by eps_c;
  the throat holds the burnt volume flow, so the drift is `sigma (u_d - S_c)` (kinematically `u_u - S_c`): a gate, not a
  fault. Design `u_d = S_L,num`, never the open box's sliding 0.98.
- **The composition oracle takes the solver's q = cp (sigma - 1) Tu**, not one normalised on T0u.
- **A viscous quasi-1D nozzle in flame units has Re ~ (A_c/A) L/delta_D**: the nozzle half is compared against a matched
  viscous inert twin, the ideal C_F only through the inviscid twin. Its viscous dt binds at the exit density: size eps_e
  and tt before a band.
- **On a `[nozzle]` flame the probe divides by A_ref = V_ref/h and searches the chamber bins only.**
- **chemFoam takes janaf/sutherland thermo only**: constant cp is janaf with only the cp/R coefficient non-zero.
- **Under `--shock` species ride the FINAL F[0]** (the species face pass after the shock face pass); never upwind on
  the KG mass flux (it breaks uniform Y), never `D Y_f(sign m_KG)` (unbounded: piston min Y -1.08e-2). D3 on a flame
  is order h^3 in S_L (G3.3).
- **A foam parity gap is a two-sided number.** In the supersonic cone foam at vanLeer holds p0 to -9.7e-4 at 320 x 64
  and dns to -6.9e-5: the gap is mostly foam's.
