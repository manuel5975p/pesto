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
17 digits, patches from B.

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

## Not built, refused

`--steady` (LTS on `Total` stalls on the host, Z7), LES, RANS, species, gravity, body force, `--turb`: refused by name
(`dns/fvmodules`, axis `capture`). Plume box, chamber, flame: next cycle. Structured D3: no.

## Bent invariants

CLAUDE.md is not edited. For the user:

- "The reader accepts exactly what the OpenFOAM mesh tools write; everything else REJECTED BY NAME": `wedge` opens on
  one route by opt-in; every other route keeps the text.
- "No NSCBC, no sink on a FaceMesh" stands for the flame box. D3 is a mean-flow numerical term on the nozzle route, on by
  default under `hll`; it uses no freestream reference, so it does not justify a sponge.
- "Mean flow gets NO added dissipation" is RANS's. RANS with `--shock` is refused.
- "gamma = 1.4 FIXED": `--gamma` on the polyMesh route only, constant, default 1.4.
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
- **A foam parity gap is a two-sided number.** In the supersonic cone foam at vanLeer holds p0 to -9.7e-4 at 320 x 64
  and dns to -6.9e-5: the gap is mostly foam's.
