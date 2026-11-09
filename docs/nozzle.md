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

`[thrust]` in `--bc` (`pa`, required, >= 0, a ratio as `[nozzle] p`; `station`: `dns flame` only, P4; needs
`[nozzle]`) turns on the impulse ledger and writes `PREFIX_thrust.csv`, 17 digits, one row per stats row. Absent: no
launch, no buffer, no file, `_stats.csv` byte for byte. `--checkpoint-dt`, `--restart`, `--wall-limit` with it: exit 2
(the ledger is not checkpointed).

**Ledger** (`DeviceFvComp::setLedger`, `FvLedger`, FvRef's twin `march(..., &ledger)`). Per stage, after the fluxes,
per patch and state row: `S_k = sum_f F_f` over the patch's materialized boundary faces (FP64, one block per patch,
thread partials in ascending face order, a shared-memory tree; no atomics), and `sum_c V omega`. At the step's end,
one thread: `I += (w0 S_0 + w1 S_1) + w2 S_2` with the weights the stage writes used (`fv_comp.hip` `step`):
`w0 = w1 = cb2 (cb1 dtR)`, `w2 = R(cb2 dtR)`, `cb1 = 1/4`, `cb2 = R(2/3)`, `ca + cb = 1` exactly. So per row k

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

## Not built, refused

`--steady` (LTS on `Total` stalls on the host, Z7), LES, RANS, species, gravity, body force, `--turb`: refused by name
(`dns/fvmodules`, axis `capture`). Plume box (P7), the reacting chamber (masterplan_combnozzle P3, P4): next. Structured
D3: no. `[thrust]` with checkpoints: refused (the ledger is not checkpointed); `[thrust].station` on `dns flow`: refused
(the reacting nozzle's burnt station, P4). The ledger refuses gravity, a body force, RANS and local time stepping (their
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
- **A foam parity gap is a two-sided number.** In the supersonic cone foam at vanLeer holds p0 to -9.7e-4 at 320 x 64
  and dns to -6.9e-5: the gap is mostly foam's.
