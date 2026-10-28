# `dns-ui` — scene editor, run cockpit and live viewer

`dns-ui` lets you build a case, run it, and watch it. You place bodies, choose a
resolution, and it shows you the grid and the cut cells the solver would
actually receive — the fluid-volume fraction λ per cell and the open aperture
per face, after `sanitizeCut()`, which is exactly what `gpu/cutpoisson` and
`gpu/cutflux` consume. Press **RUN** and it spawns the solver and renders the
live field.

**It still solves nothing itself.** It links `dns_core` only — never `dns_gpu` —
and starts on a machine with no GPU. Running means *spawning `dns` as a child
process*; watching means *mapping the shared-memory segment that run publishes*.
That is what keeps it the second binary in the tree and the only exception to
the one-binary rule in `CLAUDE.md`: it is a front end, not a solver mode.

## Build and run

```
cmake -S . -B build -DDNS_WITH_UI=ON        # SDL3 3.4.0 + HarfBuzz 14.2.1 are fetched
ninja -C build dns-ui
build/dns-ui
build/dns-ui --scene examples/tandem_cylinders.toml   # open a scene
build/dns-ui --check                        # headless self-check, no GPU/window/font
```

`DNS_UI_SCALE=2.0 build/dns-ui` overrides the UI scale (default 1.6).

`DNS_WITH_UI` is **off by default**: turning it on adds a source download and a
few minutes of dependency build to a fresh configure, and the solver does not
need it.

Dependencies are exactly three: **SDL3** (window + input), **Vulkan** (system
loader), **HarfBuzz** (text). There is no FreeType, no ImGui, no GLM — the
widget layer (`src/ui/gui.cpp`), the linear algebra (`src/ui/math.hpp`) and the
glyph rasterizer (`src/ui/text.cpp`) are local.

### About HarfBuzz and "rendering with Vulkan"

HarfBuzz does not render, and has no Vulkan backend at any version, 14.2.1
included. It is a shaping engine. What makes it sufficient as the *only* text
dependency is the `hb-draw` API: `hb_font_draw_glyph_or_fail()` walks a glyph's
outline as move/line/quadratic/cubic callbacks straight out of the font tables,
so `text.cpp` owns the rest of the pipeline —

```
hb_shape  ->  hb-draw outline  ->  flatten  ->  scanline fill  ->  R8 atlas  ->  Vulkan
```

The fill is non-zero winding with 4 sub-rows per pixel row and exact horizontal
span coverage; that combination is what keeps 11–12 px labels legible where
plain 4×4 box supersampling leaves steps on near-vertical stems.

## What it shows

Everything comes from the real pipeline. The document is turned into a
`RawConfig` and put through `dns::validate()` → `dns::buildCaseGeometry()` →
`dns::sanitizeCut()`, so:

- the grid drawn **is** `c.grid`, ghost ring and all (the interior box is drawn
  in a brighter colour — mistaking one for the other is how a body ends up a
  cell from the boundary);
- the cut cells drawn **are** the apertures the runner would get;
- an edit the CLI would reject shows the validator's own message and its exit
  code in a red banner, rather than a second, drifting opinion.

Display layers, all toggleable:

| layer | what it is |
| --- | --- |
| grid lines | cell faces, remapped under `--focal` so clustering reads as clustering; strided on large grids, and the stride is reported |
| body surface | 0-level set of the sdf; marching **tetrahedra** in 3D, marching squares in 2D, sampled over the body's measured box rather than the domain |
| cut cells | λ ∈ (0,1), coloured blue (barely cut) → red (sliver) |
| solid mask | λ = 0, i.e. the staircase the momentum update sees |
| face apertures | `ax`/`ay`/`az` ∈ (0,1) drawn on the faces they belong to |
| focal plateau | the fine-spacing box, in the accent colour |

3D cases restrict the cell layers to a slab about a slice index by default. A
512³ sphere has millions of cut cells; drawn all at once they are a blob, and
the slab is both faster and more readable. "whole body" lifts it, subject to a
cell cap that is reported when hit.

### The resolution read-out

The Grid panel's load-bearing row is

```
BL cells (D/h)/sqrt(Re)
```

the laminar boundary layer ~ `D/√Re` measured in cells. `CLAUDE.md`'s DNS
criterion is **≥ 3**; below that the row goes red and the panel says so. This is
the number that decides whether a run is DNS or an under-resolved run whose
upstream ripples are, in the repo's words, "the honest footprint of
under-resolution". Under `--focal`, `h` is the *fine plateau* spacing, because
that is the spacing the body actually sits on.

The DNS/LES label is `Case::lesLabel` — not a guess — and names the model and Cs
when it is LES.

## Layout

Everything is docked; nothing floats over the viewport.

```
menu:  dns-ui  frame  body  ortho  view  log  undo  redo  RUN/stop
+-----------+----------------------------+---------------+
| Case      |                            | Diagnostics   |
|  material |                            |  cut cells    |
|  flow     |        VIEWPORT            |  layers       |
|  numerics |                            |  live results |
|  grid     |                            |  export       |
+-----------+                            |               |
| Geometry  |                            |               |
+-----------+----------------------------+---------------+
| Run: cd / cl / K / maxDiv plots  |  solver log         |
+------------------------------------------------------- +
| status: hints           BL 2.27 · 1 mm body · Re 197.9  |
+---------------------------------------------------------+
```

The four dividers drag, and they are **painted after the regions**, with a grip
at the centre, so no panel covers them. The dock's divider exists while it is
**collapsed** too: dragging the closed header's top edge upward opens the dock
at the height you dragged it to.

The viewport is a real region: the scene is projected into its rect and
Vulkan-scissored to it, so a panel can never be overdrawn by 3D geometry, and a
press outside it is never stolen by the UI.

## Numeric fields

Every number in the UI is one widget: **drag** to change, **double-click** to
type, **shift** for fine (0.1×), **ctrl** for coarse (10×), **←/→** to nudge one
quantum while the pointer is over it. The filled bar behind the value is where
it sits in its range.

Two rules do the work, and both are in `Gui::dragRate` / `Gui::dragFill`:

* A range that **cannot go negative is a scale**, and drags as a constant
  fraction of its own value (1% per pixel on top of the base step). This is what
  makes a multi-decade field reachable at both ends — `diameter` spans
  `[D/1000, 1000 D]`, and no fixed absolute step both resolves the small end and
  reaches the large one. A **signed** range (a position, an array pitch) stays
  purely linear: dragging an object at `x = 50 D` must not run fifty times
  faster than dragging one at the origin.
* The **track** is drawn only for a range that means something. Positions are
  declared `[-1e4, 1e4] m` for want of a real bound, and a proportion of a
  sentinel is not a proportion. Where it is drawn it is log-mapped past ~1.5
  decades, so 128 cells in `[8, 4096]` reads as mid-track rather than 3% — but a
  range starting *at* zero (LES `Cs` in `[0,1]`) stays linear, because it is an
  ordinary knob and log-mapping it against an invented floor would park
  `Cs = 0.1` at 83%.

The field commits a snap of a **continuous shadow value**, not the sum of its own
committed increments: held against `lo`/`hi` it accumulates no invisible
overshoot (reversing the drag moves the value at once), and a field whose
quantum is coarser than one pixel of travel (`cells z`, 0.25 cells/px against a
quantum of 1) still drags. `checkDragLaw` in `guicheck.cpp` pins all of it.

## Two cases: `flow` and `convection`

The `case` selector at the top of the Case panel chooses which subcommand the
document is. They are two cases, not two settings of one — a convection cell
has no body, no free stream and no far field (its lateral faces are `--sides`,
not a far field), and a body flow has no plates and
no Rayleigh number — so the panels gate wholesale rather than greying out knobs
the other case has no meaning for. In convection mode the Geometry region says
so instead of offering a body picker whose every choice the validator would
reject.

`dns flame` (docs/chem.md) is NOT an editor case: the editor covers `flow`
and `conv` only. A flame run is driven from the CLI; its frames (`temp`,
`fuel`, `omega`) are declared in the run manifest like any other channel.

### `convection`

Same idiom as the rest of the editor: you edit the tangible quantities and the
dimensionless ones are derived read-outs.

| you edit | derived |
| --- | --- |
| layer height (m), temperature drop (K), mean temperature (K), material | **Rayleigh number** |
| | free-fall velocity and time, and the run length in seconds |
| | the **physical** Mach number the layer actually convects at |

`Ra = g·α·ΔT·H³/(ν·κ)` with `α = 1/T` and `κ = ν/Pr`, so one material plus a
height plus a drop fixes it — exactly as one material plus a body size plus a
freestream fixes Re.

**model** picks Boussinesq (incompressible) or fully compressible. They are two
models, not two solvers, and the panel only offers each one's own knobs:
`solver Mach`, `viscosity` and the stratification read-out appear for
compressible only, because the Boussinesq branch *rejects* them by name.

**`solver Mach` is not the physical Mach.** The read-out above it is what a real
layer convects at (~1e-3 for air); the editable one is what sets gravity,
`g = M²γ/dtemp`, and therefore how stratified the layer is and how many steps a
free-fall time costs. The panel prints the **adiabatic fraction** beside it —
the share of the imposed drop gravity's own adiabatic gradient consumes — and
goes red at 100%, where the layer is subadiabatic and the validator rejects it.

**hot patch** heats a centred fraction of the floor. It is a *side* fraction, so
the panel prints the resulting heated **area** (its square in 3D), and warns
that below 1 the case is a localised plume rather than Rayleigh–Bénard.

**sides** sits beside **box aspect** because the two are the container: with
periodic sides the aspect is a sampling window on an unbounded layer, with
closed sides it *is* the cavity and fixes the roll count. Closing them makes x
(and z, in 3D) no-slip adiabatic walls; the panel says so and warns that Nu
becomes an aspect-dependent cavity number below the periodic one, and that the
1708 onset belongs to the unbounded layer. It is emitted only when closed, so a
document that never touches it exports the command it always did. The Geometry
region — the read-out panel in convection mode — reports which faces are walls;
the grid panel notes that the printed grid is the allocation, two ghost rows for
the plates plus two columns per closed side, not the fluid count that was typed.

The grid panel's load-bearing row becomes the **thermal** boundary layer,
`H/(2·Nu·h)`, estimated from the a-priori `Nu ≈ 0.1·Ra^(1/3)`. Five points
across it is the floor; under that, the wall gradient — and therefore the
Nusselt number — is a property of the grid.

The run dock plots **Nu floor / Nu lid / Nu flux / u rms** instead of
cd/cl/K/maxDiv. Floor and lid side by side is the diagnostic: they fluctuate
independently and must agree in the mean, which is the heat balance closing.

### `thermal` (compressible body flow)

Folded away by default, and every knob in it is off by default, because a run
naming none of them is bit-for-bit the constant-viscosity, adiabatic-wall
solver every measured cd/cl anchor was taken on. It carries `--visc sutherland`
(with the `Tinf` anchor Sutherland's dimensional 110.4 K needs), `--prandtl`,
the **body** wall condition (`--wall` / `--twall` — the box walls are always
adiabatic mirrors), `--nscbc-tinflow`, and `--temp`.

`--temp` adds a third published field, `T − Tinf`, which then appears in the
live viewer's `field` selector like any other — the selector is driven by the
segment's own field table, so a run that publishes a temperature is colourable
and foggable by it with no viewer change.

## Physical units

The solver is dimensionless (`uInf = 1`, Re and Mach are its inputs). The editor
takes the tangible quantities instead — a **material** (ν, sound speed, ρ), a
**body size in m**, a **freestream in m/s** — and Re and Mach are DERIVED
read-outs shown beside them, so they cannot disagree. Cell sizes, the domain and
the run time read as `31.2 um`, `8 mm`, `2.667 ms`.

One consequence is physics, not a limitation: **Re and Mach are no longer
independent.** One material plus one freestream fixes both. Re 200 *and* Ma 0.2
in air needs a 44 µm body.

## Editing objects

Objects live in **scene** mode (the `scene` body kind); with a single `--body`
there is nothing to select.

| action | how |
|---|---|
| add | `+cyl` … `+mesh` arms the tool; the next viewport click places it |
| select | click it, or a row in the list; ctrl adds, shift ranges |
| cycle | click the same pixel again to step through overlapping objects |
| move | drag a **gizmo** handle, or type x/y/z in metres |
| scale | `diameter` in metres |
| name | the `name` field — it becomes the `cd_<name>` force column |
| duplicate / array | buttons; array makes N copies at a fixed pitch |
| remove | `remove`, or `Del` |
| undo | `Ctrl+Z` / `Ctrl+Shift+Z`, one entry per gesture |

New objects are placed by a search that never overlaps an existing body and
never crosses the domain boundary, degrading (overlap, then a smaller diameter)
with a notice rather than failing the build.

### The gizmo

Three axis arrows, three plane quads and a camera-facing centre handle, sized
constant in screen space. The drag is **absolute**, not incremental: it solves
from the current ray and the state captured at grab time, so the point you
grabbed stays under the cursor, it cannot drift over a long drag, and zooming
mid-drag is harmless. Snapping is `free` / `grid` (to mesh nodes) / `D/n`, and
**ctrl inverts** whichever is selected.

The z handle is hidden for cylinders and lenses: the scene builder never reads
their `z` (they span the box depth by construction), so a z handle would move a
number that changes nothing. Spheres, meshes and cubes are finite in z and do
read it, so they keep the handle whenever `nz > 1`.

## Running, and watching

**RUN** spawns `build/dns` with the exact command the export panel prints, adding
`--emit shm`. The button is state-aware: **stop** while a run is in flight — ours
*or* one started in a terminal, which it stops by the PID the segment carries.

Live results arrive through **POSIX shared memory**, not files
(`src/dns/shmframes.hpp`). The writer bumps a seqlock to odd, copies, bumps it to
even; the reader copies between two equal even observations and discards a torn
one. **The solver never blocks on the viewer** — a stalled or killed viewer
cannot slow a run down. Nothing is written to disk.

The viewer attaches to *any* `--emit shm` run on the machine, not only one it
started.

**Stopping during start-up.** `stop` sends SIGTERM, and the solver's handler
only sets a flag that its time loop polls — so a run stopped while it is still
acquiring the GPU does not see it until the loop starts. The editor escalates
to SIGKILL after two seconds, and a SIGKILLed run never reaches the destructor
that unlinks its segment, leaving a corpse the next RUN has to step over. A stop
issued once the run is live (i.e. once the viewer has attached) is always clean.

| view | control |
|---|---|
| cut plane | axis (`yz`/`xz`/`xy`) + **plane position**, sliced from the volume by the viewer, so dragging it is instant |
| volumetric fog | the selected field as back-to-front translucent slices; density, threshold, softness, slice count |
| field | `vort` / `speed` — every field's volume is published, so either can be sliced or fogged |
| colour | `field` (the field's own ramp) / `vs mean` / `vs tunnel` — see below |
| timeline | scrub back through the frames already received — see below |
| plots | cd / cl / K / maxDiv in the dock, parsed from the solver's own stdout |

### The frame rate is the run's, not the viewer's

Everything above redraws when a frame *arrives*, so the output frame rate is the
solver's publish cadence: **`--wdt`**, the web/shm frame interval, in solver time
units. The Flow panel edits it as **frame interval** and shows how many frames
that is over the run and how far apart they are in real time; the exported
command carries it, so the printed line reproduces the run that was watched.
The solver's own default (0.15) is a couple of dozen frames for a one
flow-through case — the editor starts at 0.05 instead. It is not free: a frame
every few steps costs the run real time, which the panel warns about past 20k
frames.

### Rewind

Every frame the viewer decodes is kept, so a run can be scrubbed back through
**while it is still running**, and a finished run stays scrubbable until the
window is closed. The timeline slider, the `|< < > live` buttons and the
left/right arrow keys all move it; stepping off the end returns to following the
run, so there is no state to get stuck in.

What is kept is the **planes only**. One 512³ volume is 512 MB and a history of
them would be the machine's memory, so a rewound frame shows the published
centre planes: the fog and the off-centre cut plane follow the live frame, and
the panel says so rather than showing stale data as if it were current. The
history is capped at 256 MB — thousands of frames of a 512² run — and evicts the
oldest first. Switching the coloured field restarts it, because the same frames
decoded as a different quantity are a different series.

**Relative colouring.** A magnitude like `speed` has no natural zero, so its own
ramp answers "how fast" — rarely the question. Re-centring it on a reference
speed answers "where is the flow *not* the tunnel": slower reads blue, faster
reads red, at-reference sinks into the background, exactly like the vorticity
view. `vs tunnel` uses the case's own `uInf`; `vs mean` measures the frame's mean
over the fluid cells (solid cells excluded). The span is measured from the data,
not from the field's fixed range — at Re 200 the wake deficit is a few percent of
uInf, and scaling that against `[0, 1.6]` would leave the picture one flat
colour. `contrast` narrows the span without moving the reference.

It applies to the fog as well, and there it also picks what is *visible*: the
fog threshold drops anything near the middle of the ramp, which with a reference
speed means "show me only where the flow departs from the tunnel".

### Why the fog is not layered

A slice stack is only correct **back to front**, and three things have to hold
or it stops looking like fog:

- the slices are traversed from the far end along the view direction, and the
  mesh records which octant it was built for — orbiting past a diagonal flips
  the slice axis and the stack is rebuilt. Composited the other way round it
  reads inside-out;
- density is authored **per cell of depth**, not per slice drawn. When the stack
  is thinned (`fog slices` below the depth of the volume) each slice carries the
  opacity the cells it stands in for would have, `1-(1-a)^step`, so changing the
  slice count changes how smooth the fog is and not how thick;
- the threshold **ramps** (`fog softness`) and the colour is carried on the quad
  corners, shared between neighbours. A hard cut on a smooth field draws its own
  iso-contour and the fog comes out as nested shells; flat quads make each slice
  a mosaic of tiles. Both read as layering. Drag the softness to zero to see it.

The fog has no solid mask — the segment publishes one per plane but not per
volume — so inside a body the fog shows whatever the field holds there.

Volumes are capped at 512 MB (one field of a 512³ run is 512 MB on its own):
above that the run publishes planes only and the viewer says so.

## Export

The Export panel prints the equivalent `build/dns flow …` command line —
runnable as printed — and writes the object list as a `docs/scene.md` scene TOML.
The editor's scene is *defined* as `parseScene(sceneToml())`: every build round
trips through the real parser, so a file it writes is one the CLI accepts, and an
empty scene or a bad name surfaces as the CLI's own exit-3 message. Nothing else
is ever written.

## Caveat inherited from the solver

Under `--focal` the solver's body is the WORLD sdf at the MAPPED cell centres
(the ghost wall). The viewer draws exactly that: the cut cells come from
`dns::sampleCutMapped` (the world sdf at the mapped sub-samples, in index
space) and the surface samples the sdf at physical positions. The world-lattice
`sampleCut()` the runner keeps is the qRef frontal-area measurement only (see
`dns/geometry.hpp`) and is never drawn. With `--focal` off the two coincide
exactly.
