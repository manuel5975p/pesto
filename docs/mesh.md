# FaceMesh — the finite-volume mesh of the FV path

`plans/masterplan_meshsolver.md` §2.2–2.3, phases P1–P2. Host only: the
polyMesh reader/writer (`src/dns/polymesh.*`), the FaceMesh builders and
quality report (`src/dns/facemesh.*`), the matrix graph (`src/dns/la/graph.*`),
`dns mesh`, and the FV path's face physics with its CPU FP64 oracle
(`src/dns/fvflux.hpp`, `src/dns/fvgrad.*`, `src/dns/fvref.*`; "The FV face
physics and its FP64 oracle" below). The first GPU solver on a FaceMesh is
the incompressible FV path (`dns flow --polymesh`, P5b, `docs/fv.md`):
OpenFOAM simpleFoam on the identical mesh, with OpenFOAM's MRFZoneList on the
mesh's cellZones (`zoneNames` / `zoneCells`). Every existing path is
untouched by construction.

## What is accepted

`readPolyMesh(dir)` reads `points`, `faces`, `owner`, `neighbour`, `boundary`
and (optional) `cellZones` of a `constant/polyMesh` directory:

| | accepted | rejected (by name, exit 3) |
|---|---|---|
| format | `ascii`, `binary`; `arch` LSB, `label=32` or `label=64` (narrowed to int32, range-checked), `scalar=64` | `*.gz`, MSB, `scalar≠64`, uniform lists `N{…}`, binary non-compact `faceList` |
| faces | `faceList` (ASCII), `faceCompactList` (ASCII or binary) | faces with < 3 nodes, labels out of range, truncated lists |
| topology | owner < neighbour on internal faces; patches tiling `[nInternalFaces, nFaces)` in order | `owner ≥ neighbour`, gaps or overlaps between patches |
| patches | `wall`, `patch`, `symmetry`, `symmetryPlane`, `empty`, `cyclic` (`transform translational`/`unknown`/`coincidentFullMatch`/`noOrdering`), `wedge` (reader only) | `processor*` ("decomposed: run reconstructParMesh"), `cyclicAMI`, `cyclicACMI`, `nonConformal*`, `mapped*`, cyclic `rotational`, unpaired cyclics, any other type |
| dictionaries | plain entries, `inGroups`, header `meta {}` | `#include`, `#calc` and every other `#` directive |
| cellZones | ascending after read | a cell in two zones, a cell listed twice |

That is exactly what blockMesh, snappyHexMesh, extrudeMesh, refineMesh,
foamFormatConvert and plot3dToFoam (v2606) write. The counts come from the
owner header `note`, which every OpenFOAM writer emits; `readPolyManifest`
reads only the note, `boundary` and the zone names (`dns mesh --dry-run`).
`writePolyMesh(dir, m, binary, labelBits)` writes the same layout back
(`faceCompactList` in binary, `faceList` in ASCII, shortest round-trip
doubles); write → read is bitwise (`test_polymesh` T2).

## Geometry: ONE definition, OpenFOAM v2606's

Face centre and area vector: `primitiveMeshTools::makeFaceCentresAndAreas`
(triangles direct; otherwise the fan about the vertex mean, centre
area-weighted). Cell centre and volume: `makeCellCentresAndVols` (pyramids
on the face-centre mean, owner faces then neighbour faces in face order).
Same operand order, FP64. `fromCurvMesh` writes the structured mesh as a
PolyMesh and calls `fromPolyMesh` — there is no second geometry code.

Consequence, stated, not "fixed": on a WARPED hex face the FaceMesh volume
(pyramids on the fan) differs from `CurvMetrics::vol` (consistent diagonal
triangulation) at warp order; on planar faces they agree to rounding. The
area vector is the polygon's vector area in both, equal to rounding on any
quad.

Measured (`test_facemesh` F1/F3/F8):

| mesh | vs | worst |
|---|---|---|
| `cyl2d_snappy`, `box_refined` | their `log.checkMesh` (15 digits) | volume total/min/max, non-orth max/avg, skewness, aspect: every printed digit |
| `mB` (DTMB, 831 578 cells, snappy, local) | its `log.checkMesh` (10 digits) | every printed digit; read 0.31 s + build 0.34 s |
| uniform periodic 6×5×4, power-of-two h | exact | S, V, d exact (bitwise) |
| 2D O-grid 64×24 | `CurvMetrics` | S 4.1e-16, V 2.4e-15 |
| sinusoidal periodic 8×6×4 (amp 0.05) | `CurvMetrics` | S 8.3e-16; V 1.3e-15 (warp order, not gated); Σ V = box 1.1e-15 |

`mB`'s checkMesh ran on 8 ranks: its face count 2 622 899 includes the
processor faces; the reconstructed mesh on disk has 2 600 459, which is what
the FaceMesh holds.

## The FaceMesh contract

- **Faces.** `[0, nI)` internal with `own < nei`, sorted by
  `(own, nei, srcFace)`; `[nI, nI + nB)` boundary, contiguous per patch in
  boundary order, each patch sorted by `(own, srcFace)`. `S` points
  own → nei (boundary: outward). `d` = `C_nei − C_own` (internal),
  `Cf − C_own` (boundary).
- **Cyclic pairs are internal faces.** Face *q* of a cyclic patch pairs with
  face *q* of its `neighbourPatch`. The pair is checked (centre offset equal
  to `separationVector` — or to the first pair's offset when absent — within
  1e-8 of the bounding-box diagonal; areas opposite within 1e-8) and merged
  into one internal face: the side whose owner is the LOWER source cell keeps
  its `S`, `Cf` and `srcFace`; `d = (C_b + (Cf_a − Cf_b)) − C_a`. `Cf` of a
  cyclic face therefore lies in the frame of the srcFace's polyMesh owner. A
  cyclic pair coupling a cell to itself (one cell across a periodic axis) is
  an error. Cyclic patches do not appear in `patches`.
- **2D is a one-cell-thick mesh, not a code path.** Non-empty `empty` patches
  set `emptyAxis`: every empty face normal to that one axis (1e-9), every
  cell exactly one + and one − empty face, every other face
  `|S·e| ≤ 1e-12 |S|` — else "`empty` must be a one-cell extrusion along
  x|y|z". Then the `emptyAxis` component of `S` and `d` is set to EXACTLY 0
  on every non-empty face, so a z-momentum face flux is a product with an
  exact 0 and `w` stays 0. Empty faces keep their area (kernels skip them).
- **Cells.** `C`, `V`; `cellStart`/`cellFace` is the CSR cell → faces in
  ascending face index — the deterministic, atomics-free gather order.
- **Zones.** `zoneNames`/`zoneCells` (ascending new indices): MRF zone,
  injector, flame region.
- **Provenance.** `srcCell`/`srcFace` map new → source (polyMesh index;
  `fromCurvMesh`: `srcCell` = `Grid::idx`, `srcFace` = the `polyOf` face).
- **Validation** (exit 3, named): open cell (`|Σ S| > 1e-9 Σ|S|`), `V ≤ 0`,
  `d·S ≤ 0`, cells with < 4 faces, unmatched cyclics, bad `empty`, `wedge`
  patches (the axisymmetric FaceMesh is P8).

## Ordering

`--order source|morton|colour` (`CellOrder`):

1. Geometry is computed once in SOURCE numbering.
2. Morton: 21 bits per axis of `C` over its bounding box (a degenerate axis
   is 0), interleaved; ties by source index; stable.
3. Colour-major: `greedyColour(graphOf(Morton mesh))` (distance 1, rows in
   order, smallest free colour; < 255 colours required), then stable by
   colour (`colourMajorOrder`). `colour`, `nColours` are filled.
4. Faces are re-oriented (an own/nei swap negates `S` and `d` EXACTLY) and
   re-sorted; every array is permuted.

So geometry is bitwise invariant under the ordering (F4), and the build is
byte-identical for any thread count (`forBlocks` over faces and cells, each
writing its own index; the first error is the lowest block's: F5). Greedy
colouring in Morton order gives 2 colours on an even periodic box; 5 on the
snappy and refined meshes. Which order the GPU path uses is P4's
measurement; it is then pinned as ONE global default.

## CellGraph (`dns/la/graph`)

Neighbour-unique rows in ascending column order, no self entry; per internal
face `ownSlot` (the entry of nei in row own) and `neiSlot` (own in row nei),
as indices into `col`. Two faces between one cell pair (a 2-cell periodic
axis, split faces) share one entry. Matrix assembly scatters face blocks to
those slots in face order: deterministic, no atomics.

The matrices built on this pattern — sliced ELL with `slotPos` from a slot to
its device entry, smoothers, line sets, Krylov, aggregation AMG — are
`gpu/la`: `docs/la.md`. OpenFOAM's `Gauss linear corrected` pressure matrix on
a FaceMesh (`|S|/max(n·d, 0.05|d|)`, fixedValue `|S|/(n·d_b)`) is reproduced
exactly: solved fields equal laplacianFoam's to 1.1e-14 (`docs/la.md`, P5).
`la::structuredGraph(n, periodic)` is the same pattern built in index space
(equal to `graphOf(fromCurvMesh(uniform, Source))`, `test_la`).

## `dns mesh`

```
dns mesh --polymesh DIR [--order source|morton|colour] [--dry-run]
```

Host-only, like `dns prop`: Parse → validate → Acquire (read + build) →
report. `--dry-run` prints the resolved case and the manifest (counts,
format, patches with cyclic partners and separations, zone names) without
reading points or faces. The report: counts and dimension, patches, cell
zones, volume min/max/total, non-orthogonality max/average, skewness and
aspect ratio max (checkMesh's definitions: non-orthogonality over internal
faces incl. merged cyclics, skewness over all faces, aspect over the solution
directions), the CFL width `dim V / Σ|S|` over non-empty faces (min, p1,
median, max — `h/2` on a cube), the max adjacent volume ratio (warns above
1.25: 2:1 jumps reflect near-grid content in a dissipation-free scheme),
faces-per-cell and nodes-per-face histograms, and the colour count. Exit 0,
1 CLI, 3 mesh input. `meshQuality` needs the FaceMesh built by
`fromPolyMesh` from that PolyMesh (`srcFace` indexes it).

## Test meshes (`tests/data/mesh`, 352 KB)

`tools/mesh/make_test_meshes.sh` (dicts in `tools/mesh/dicts/`) regenerates
them with OpenFOAM v2606, each with its own `log.checkMesh` (writePrecision
15), in one `simjob --cpu` job (1.2 s):

| mesh | how | cells | exercises |
|---|---|---|---|
| `box_cyclic` | blockMesh 4×3×2 on 1×0.75×0.5, x/z cyclic translational, y walls, ASCII | 24 | cyclic merge, h = 0.25 exact |
| `box_cyclic_bin` | the same through foamFormatConvert | 24 | binary, `faceCompactList` |
| `box_refined` | 8×6×4 cyclic on every axis + refineMesh of an interior box | 276 | split-face polyhedra (9 faces, 5-node faces), periodic, no boundary |
| `cyl2d_snappy` | snappy cylinder D 1 (surface level 2, wake box level 1, 3 layers) in a slab, extrudeMesh to one layer, `empty` front/back | 1084 | 2D, prism layers, 7-8-face cells, 5-6-node faces, triangles |

`runs/` is git-ignored, so the 831k `mB` gate (`facemesh_mb`) is local and
skips (77) without it.

Prism-layered snappy meshes read unchanged (phase BL, `docs/fv.md` "Prism
layers"): the rotor2 L1 mesh re-meshed with snappy addLayers
(`tools/fv/make_rotor_bl.py`: 8 requested layers, relative first layer 0.03,
expansion 1.3, `nBufferCellsNoExtrude 0`) is 123 511 cells (110 988 hex,
731 prisms, 11 789 polyhedra; layer-truncation cells are polyhedra), checkMesh
non-orth max 65.2°, 2 faces with skewness 17.9 (its one failed check). snappy
extrudes 79% of the 5440 blade faces, 1.52 layers on average, 41% of the
target thickness (issues.md 61a). Its meshWave wall distance equals
simpleFoam's serial one (`fvwalldist_rotor`, local).

## The FV face physics and its FP64 oracle (P2)

Three pieces, all host-side in P2; the GPU explicit path (P4) and the FP64
implicit residual (P7) instantiate the same templates.

| file | content |
|---|---|
| `dns/fvflux.hpp` | DNS_HD, templated on the scalar T (double, float, `Dual<N>`) and the geometry scalar G. `fvPrimOf` (Comp's operand order), `fvTransportAt` (one Sutherland evaluation per face), `fvFaceGrad` / `fvFaceGradBnd`, `fvTraction`, `fvMeanFlux` (KG product of means + face viscous; `wS` = a frame's advected area flux, 0 inertial), `fvEulerFlux`, `FvBc` + `fvBoundaryState` + `fvBoundaryFlux` |
| `dns/fvgrad.{hpp,cpp}` | `FvFaceGeom` (frame-consistent `rOwn`, correction vectors `corr`, `active`), `FvGradWeights` (least squares or Green–Gauss, on differences), `fvCellGradient` |
| `dns/fvref.{hpp,cpp}` | `FvRef`: `rhs`, `evaluate` (every intermediate), `stableDt` (Σ forms, per cell), `march`/`step` (SSP-RK3); `FvState` `[nV][nC]` |

**Scheme.** Restates `faceFluxCurvOf`'s algebra on an arbitrary face (the
structured bodies are not touched):

- convective: Kennedy–Gruber product of face means,
  `F = {ρ̄ U_a, ρ̄ U_a ū + p̄ S, ρ̄ U_a ē + p̄ U}`, `U = ū·S`, `U_a = U − wS`;
- viscous: `−{0, τ·S, ū·τ·S + κ ∇T·S}` with `μ, κ` at the face-mean T;
- face gradient `g_f = ḡ + (φ_N − φ_P − ḡ·d) corr`, `ḡ = ½(g_P + g_N)`,
  `corr = α d/|d|²` (`AlongD`) or `α S/(S·d)` (`AlongNormal`: Nishikawa
  α-damping; α = 1 has the FORM of OpenFOAM's `corrected` snGrad on n, but
  v2606 interpolates ḡ with the `linear` weights, clamps `n·d ≥ 0.05|d|`,
  and applies NO correction on non-coupled boundary faces, where it uses
  `(φ_b − φ_P)/(n·(Cf − C_own))`). Boundary faces here:
  `ḡ = g_P` toward the boundary face value over `d = Cf − C_own`;
- cell gradients on differences, `g_c = Σ_f w_{c,f} (φ_other − φ_c)`:
  least squares `w = G_c⁻¹ d/|d|²`, G from `d dᵀ/|d|²` (NOT v2606's
  `leastSquares`, which weights each face by `(1 − w_f)|S_f|/|d|²` and uses
  the patch-normal `n (n·(Cf − C))` on non-coupled boundary faces: 1.2%
  relative apart on `cyl2d_snappy`, measured; 2D inverts G in the 2D
  subspace, so the emptyAxis weight is exactly 0) or Green–Gauss
  `w = (1 − λ) S/V` with OpenFOAM's `linear` weights (= v2606 `Gauss linear`
  to 2.6e-15 relative on `cyl2d_snappy`, measured). Boundary faces enter
  with their boundary-state values;
- gather: each cell sums its CSR faces in ascending face order,
  `R = −Σ ±F / V + src`. Every pass (primitives → boundary states → cell
  gradients → face fluxes `F[5][nI+nB]` → gather) writes only its own index:
  byte-identical at any thread count.

**Pinned: least squares + `AlongD`, α = 1.** Measured below: least squares
is exact for linear fields on every mesh (Green–Gauss is 16–28% off on the
skewed test meshes) and has the smaller MMS error (Green–Gauss +2–9%);
`AlongD` is the structured scheme on a Cartesian mesh (G1), has the smaller
MMS error (α-damping 4/3: +11–62%), and the better checkerboard efficiency on
the smooth skewed mesh (0.969 vs 0.745), equal on random meshes. Both are
second order (G4).

**Boundaries are face fluxes** (`fvBoundaryState`, `fvBoundaryFlux`); no
state or rhs overwrite anywhere:

| kind | boundary state (also the gradients' face value) | face flux |
|---|---|---|
| `Wall` | ρ_P, `u_w(Cf) = uWall + ω×(Cf − origin)`, p_P (zero gradient), T_w (isothermal) or T_P | `{0, p_P S − τ·S, p_P u_w·S − u_w·τ·S − q}`, `q = κ ∇T·S` isothermal, **exactly 0** adiabatic |
| `Slip` | P with the normal velocity removed | `{0, p_P S − (n·τ·S) n, 0}`: no shear, no heat, no work |
| `Inflow` | u_in, T_in, `p = p_P + ρ_P c_P (u_n,P − u_n,in)` (outgoing acoustic invariant) | Euler flux of the state; no viscous flux |
| `Outflow` | p_out, u_P, T_P; backflow (`u_P·S < 0`): normal u only, T_back | Euler flux; no viscous flux |
| `Far` | 1D characteristic state between P and the freestream (Riemann invariants `u_n ± 2c/(γ−1)`, entropy + tangential u upwind, supersonic all upwind), as differences from the upwind state: P = freestream returns the freestream bit for bit | Euler flux; no viscous flux |
| `Empty` | — | none |
| cyclic | internal face | — |

The freestream of `Far` is given CONSERVED (`qFar`), so its primitives come
out of `fvPrimOf` exactly as an interior cell's. A 2D mesh rejects BC
velocities along the empty axis and rotations about any other axis.

**dt (Σ forms, per cell, LTS-ready).** Convective `2 cfl V / Σ_f (|u·S| +
c|S|)` (cfl 1 = the 1D CFL); diffusive `diff ρ V / (μ(T) max(4/3, γ/Pr) Σ_f
corr·S)`. `Σ_f corr·S` is the cell's compact-Laplacian diagonal, so `diff =
1` puts the checkerboard at λΔt = −2 on any mesh and dimension — issue 15's
lesson applied from day one (E3: decays at the bound, grows at 1.3×). The 4/3
is the momentum checkerboard's normal stress: with 1, Pr > 1.05 put a
longitudinal u checkerboard past −2 (−2.67 from Pr 1.4, past SSP-RK3's −2.51
from Pr 1.32; ×1.8e6 in 60 steps at Pr 2 on 100:1 cells, E6). The structured `Transport::diffusivityFactor` keeps 1.

**SSP-RK3** in the fused form `out = ca a + cb b + cr R(b)`, `ca = 1 − cb`
(cb = 0, ¼, ⅔), on persistent workers (`march`: one thread spawn per call,
barriers between passes).

**Cyclic face centres.** A cyclic face's `Cf` lies in its srcFace side's
polyMesh-owner frame (FaceMesh contract), which may be the neighbour's.
`FvFaceGeom::rOwn` is `Cf − C_own` in the OWNER's frame (translated by
`C_own + d − C_nei` when that brings it nearer); the neighbour offset is
`rOwn − d`. Kernels read `rOwn`, never a raw `Cf − C_own`.

**For later phases** (interfaces only): the state is `[nV][nC]` with nV at
run time (P2 accepts 5); μ, κ are the caller's face values (eddy viscosity
adds there); `wS` carries the MRF frame flux (P9: edge-integrated potential);
`fvMeanFlux<Dual<N>>` gives P7's Jacobians (E5); Green–Gauss with OpenFOAM
linear weights and `AlongNormal` α = 1 are the `Gauss linear` / `corrected`
pieces P5b needs (P5b still has to interpolate ḡ with the linear weights,
not ½); `rOwn` serves reconstruction limiters (P10).

### Gates (`test_fvref`, `test_fvref mms`; host, FP64)

| gate | measured |
|---|---|
| G1 FvRef vs `Comp::rhs`, uniform periodic 8×6×5, h 0.3/0.45/0.2 | 4.1e-15 (μ 0.05, LSQ), 3.4e-15 (Sutherland, colour order), 4.3e-15 (Green–Gauss), 4.0e-15 (inviscid), 6.0e-15 (2D) — gate 1e-13 |
| G2 inviscid KE budget = discrete pressure work `Σ_f p̄ S·(u_N − u_P)` | perturbed periodic hexes (±0.15h, warped faces) 4.5e-17 (uniform p) / 7.4e-17 (random p); `box_refined` 1.4e-16 / 6.2e-17; totals conserved ≤ 1.3e-16 — gate 1e-13 |
| G3 freestream, `cyl2d_snappy`, μ 0.01, M 0.63 | every patch `Far`: max \|rhs\| 2.0e-14; inlet/outlet/slip/far mix: 1.3e-14 — gate 1e-13 |
| G4 MMS, 2D NS (μ 0.01, M ≈ 0.36), sinusoidal amp 0.06 periodic, marched to steady | L2 error ρ 4.01e-4 / 9.95e-5 / 2.48e-5 at N 32/64/128; orders 64→128 (ρ, ρu, ρv, E) 2.003 2.006 2.005 2.002 — gate 1.8 (32→64: 2.01–2.02). Green–Gauss 32→64: 2.01–2.02 (errors +2–9%); α-damping 4/3: 2.00 (errors +11–62%) |
| G5 checkerboard of the face Laplacian (LSQ) | efficiency ρ_cb/λ_max: uniform 1.00 both; random ±0.25h 2D: AlongD 0.904, α 4/3 0.924; sinusoidal 2D: 0.969 vs 0.745; random 3D: 0.985 vs 0.986. After 40 RK3 steps at 2/λ_max the checkerboard is ≤ 1.6e-5 (AlongD) everywhere |
| G6 2D, `cyl2d_snappy` (spinning isothermal wall, inflow, outflow, slip) | every z-momentum face flux, gradient z component, boundary w, rhs ρw exactly 0.0; ρw exactly 0.0 after 20 steps |
| G7 adiabatic wall on a skewed walled box (sinusoidal 0.06, wall non-orthogonality up to 45°, T varying along the wall) | wall mass and energy fluxes exactly 0.0 on all 64 wall faces; Σ V rhs_E / Σ V \|rhs_E\| = 5.9e-18; isothermal contrast 6.0e-3 |
| E1 linear field | LSQ exact to 5.0e-15 (`cyl2d_snappy`) / 1.5e-15 (walled sinusoidal box); Green–Gauss 0.278 / 0.163 |
| E2 threads 1/2/3/7/20 | rhs and a 3-step state byte-identical (9216 cells) |
| E3 Σ-form dt, 3D isotropic 12³, T checkerboard, μ 1 | at the bound: ×2.9e-10 in 20 steps; at 1.3× (λΔt −2.6): grows ×16 |
| E4 Source / Morton / colour-major | rhs equal to 1.8e-16 |
| E5 templates | `fvMeanFlux<Dual<10>>` d/dq vs FP64 central FD 6.0e-11; `Far` boundary flux on Dual 3.7e-10; `float` vs double 1.5e-7 |

Cost: the G4 march on 20 host threads runs N 128 (16 384 cells, 28 350
steps, warm-started from the N 64 error / 4) in 29.7 s (48.9 s with a GPU
job sharing the host), ~1–1.7 ms per SSP-RK3 step; the whole `fvref_mms`
test 38.7–56.4 s, bitwise the same numbers each run. `fvref` takes 0.2 s.

## Not built (P1, P2)

`wedge` patches and axis prisms (P8), decomposed meshes (`processor`),
non-conformal couplings, a FaceMesh of a revolved structured mesh
(`axisFaces`, `MirrorZ`), `meshQuality` on a `fromCurvMesh` mesh, multi-block
builders (P12). FV path (P2 is the host oracle only): no GPU kernels (P4), no
scalars or species rows (`fvScalarFlux`, P8/P10), no MRF frame flux (P9), no
eddy viscosity, no LTS march (per-cell dt exists), per-face boundary data
(profiles) not supported — one `FvBc` per patch. `Far` is 1D characteristic
(no transverse terms), `Outflow` fixes p (reflects acoustics, as OpenFOAM's
fixed p does), `Inflow` is subsonic. See `issues.md`.
