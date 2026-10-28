# FaceMesh — the finite-volume mesh of the FV path

`plans/masterplan_meshsolver.md` §2.2 and phase P1. Host only: the polyMesh
reader/writer (`src/dns/polymesh.*`), the FaceMesh builders and quality
report (`src/dns/facemesh.*`), the matrix graph (`src/dns/la/graph.*`) and
`dns mesh`. No solver runs on a FaceMesh yet (P2 adds the CPU FP64 oracle, P4
the GPU explicit path). Every existing path is untouched by construction.

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

## Not built (P1)

`wedge` patches and axis prisms (P8), decomposed meshes (`processor`),
non-conformal couplings, a FaceMesh of a revolved structured mesh
(`axisFaces`, `MirrorZ`), `meshQuality` on a `fromCurvMesh` mesh, multi-block
builders (P12). See `issues.md`.
