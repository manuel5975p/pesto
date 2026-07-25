# `dns-ui` — interactive obstacle editor and grid viewer

`dns-ui` is a **pre-run inspector**. It solves nothing. You place bodies, choose
a resolution, and it shows you the grid and the cut cells the solver would
actually receive — the fluid-volume fraction λ per cell and the open aperture
per face, after `sanitizeCut()`, which is exactly what `gpu/cutpoisson` and
`gpu/cutflux` consume.

It is the second binary in the tree and the only exception to the one-binary
rule in `CLAUDE.md`, for that reason: it is not a solver mode, and it never
touches `dns_gpu`.

## Build and run

```
cmake -S . -B build -DDNS_WITH_UI=ON        # SDL3 3.4.0 + HarfBuzz 14.2.1 are fetched
ninja -C build dns-ui
build/dns-ui
build/dns-ui --check                        # headless self-check, no GPU/window/font
```

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

## Interaction

| input | action |
| --- | --- |
| LMB drag | orbit (3D) / pan (2D) |
| RMB or shift+LMB | pan |
| wheel | zoom |
| click an object | select it (scene mode); drag to move it in the view plane |
| double-click a number | type an exact value |
| shift while dragging a number | fine adjust |
| `F` / `B` | frame the domain / the body |
| `G` `C` `M` `A` | grid / cut cells / solid mask / apertures |
| `Del` | remove the selected object |

`nz == 1` is not orbited: the camera is pinned to look straight down −z in
orthographic projection. An oblique view of a 2D cut foreshortens the apertures
the tool exists to show.

Dragging an object queues a rebuild in the background thread; the selection box
follows the pointer immediately because it is computed from the edited offsets,
not from the last completed build.

## Export

The Export panel prints the equivalent `build/dns flow ...` command line —
runnable as printed, `--ft` included — and writes the object list as a
`docs/scene.md` scene TOML. Nothing else is ever written.

## Caveat inherited from the solver

Under `--focal`, `sampleCut()` evaluates the sdf at **reference** coordinates
while the ghost wall classifies against **mapped** centres (see
`dns/geometry.hpp`). The viewer draws the cut-consistent body: sdf evaluated in
reference space, plotted at mapped positions. Surface, cut cells and apertures
therefore agree with each other and with what the runner consumes. With
`--focal` off the two coincide exactly.
