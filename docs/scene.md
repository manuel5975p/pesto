# Scenes — the generalized simulation environment

`dns flow --scene FILE.toml` replaces `--body` with a scene: several solid
objects placed in one free stream, an explicit model selection, and per-object
force integration. Everything else about the run (grid, physics, far field,
timestepping, output) is the ordinary `dns flow` machinery — a scene is a
geometry source, not a new solver. Ready-made examples: `examples/*.toml`.

## File format

TOML (parsed with the vendored [toml++](https://github.com/marzer/tomlplusplus)
v3.4.0, `third_party/tomlplusplus/toml.hpp`), held to the CLI's strictness:
unknown keys, wrong types and duplicate names are hard errors (exit 3), never
silent defaults; duplicate keys the TOML grammar itself rejects.

```toml
[model]
type = "les"
cs = 0.16

[[objects]]
kind = "cylinder"
name = "front"
d = 1.0

[[objects]]
kind = "cylinder"
name = "rear"
x = 2.5
y = 0.25
d = 0.5
```

### Units and placement

Offsets `x, y, z` and diameter `d` are in units of the reference diameter `D`
(the one `--dr` already defines), measured from the standard single-body
placement; all default to 0 (`d` to 1). A lone `[[objects]] kind = "cylinder"`
is therefore the classic cylinder case — but on the generic sampled cut
(`{8,8}` supersampling), not the exact closed-form cut, so coefficients differ
slightly from `--body cylinder`. Objects must stay strictly inside the domain
interior; wall-mounted rigs (fence, floor) are not scene objects — their wall
sealing and mass-balance plumbing is special-cased and stays on `--body`.

### Objects

| kind       | extra keys                                | dimensionality |
|------------|-------------------------------------------|----------------|
| `cylinder` | —                                         | z-invariant    |
| `cube`     | — (45°-rotated cube, side `d/√2`)         | z-finite*      |
| `lens`     | `aspect` (> 1)                            | z-invariant    |
| `sphere`   | —                                         | 3D (needs `--nz`) |
| `mesh`     | `path` (.stl/.gltf/.glb), `scale`, `pitch`, `yaw`, `robust` | 3D (needs `--nz`) |

All objects take `name` (`[A-Za-z0-9_-]`, the force-column label; defaults to
`kind + index`, must be unique), `x`, `y`, `z`, `d`. A 3D object on a native-2D
grid (`nz == 1`) is rejected at geometry build.

*A cube is z-FINITE but not 3D-only: `d` is its frontal width across the
diagonal, so its side — and its z extent, centred on `z` — is `d/√2`, and it
reads `z` like a sphere does. At `nz == 1` it degenerates to the full-depth
prism instead of being rejected, because a cube's 2D reading is its
cross-section. Either way it is excluded from the z-invariant sampling
short-cut: replicating one x-y layer would extrude it back into a pillar.
`dns::sceneCube` is the one definition of all of this, and `--body cube`, the
scene builder and the editor all call it.

### Model selection

`[model]` is `type = "dns"` (the default) or `type = "les"` with `cs` and an
optional `sgs = "smag"` (default) or `sgs = "sigma"` picking the SGS operator
(see `--sgs` in `dns flow --help`; sigma wants `cs ≈ 1.35` and needs nz > 1).
The scene's selection merges into the config before validation, so labels,
`--dry-run` output and the run manifest all carry it. A conflicting `--les`
or `--sgs` on the command line is a config error — the conflict is never
silently resolved. All the usual LES ground rules apply: `cs > 0` means the
run is FILTERED and is labelled LES everywhere (with the model named).

## What the mesh builder does

`buildSceneGeometry()` resolves every object to world coordinates, loads mesh
objects, composes the **union SDF** (min over objects; fluid > 0) and samples
one cut geometry over the whole scene — on a single replicated x-y layer when
every object is z-invariant. Frontal area (and with it `qRef`) is measured
from the resulting apertures exactly as for a single body.

## Integrated quantities

The `stats` series is unchanged: total `cd`, `cl`, `cdf` (friction, ν-only),
kinetic energy, CFL, divergence — the surface-pressure and friction integrals
over the whole scene.

With ≥ 2 objects, incompressible runs additionally write
`<out>_forces.csv`: `t`, then `cd_<name>, cl_<name>, cdf_<name>` per object —
the same discrete surface integral restricted to each object's own index box,
normalized by the same `qRef`, so the per-object columns **sum exactly to the
totals**. If two objects sit so close that their tally boxes overlap, the
per-object series is suppressed with a warning (each box would count the
shared surface) and the totals remain authoritative.

Compressible scene runs report total forces only for now (the ghost-wall force
tally reduces over one global face list; a per-object split needs a per-face
object id and is a follow-up).

## Phases

The scene file is read in *phase 1.5* — after CLI parse, before validation —
because its `[model]` block feeds the validator. Mesh files named by scene
objects are still loaded in phase 3 (acquire), like `--body FILE.stl`.
Exit codes: unreadable/invalid scene file 3, model conflict or `--scene`
+ `--body` 2, geometry errors (3D object at `nz == 1`, object out of bounds,
bad mesh) 3.
