# Web viewer frame format (v1)

Directory-based, streamable, self-describing time series for a browser
viewer. Written live by the apps; a viewer polls the manifest and fetches
frames on demand. Configurable field set — the manifest declares what a run
carries, the viewer renders whatever it finds.

## Layout

    web_<out>/
      manifest.json        rewritten atomically (tmp + rename) every frame
      <field>_NNNNN.bin    one file per field per frame
      <static>.bin         time-independent fields (e.g. mask), written once

## Frame encoding

- u16 little-endian, row-major, i fastest, row j = 0 at the BOTTOM
  (matches the PNG/export convention), nx*ny values, no header.
- Quantized against the field's fixed `range` [lo, hi] declared in the
  manifest: value = lo + (hi - lo) * q / 65535, clipped. Fixed ranges keep
  playback flicker-free; the true per-frame [min, max] is recorded per frame
  for diagnostics/auto-range.
- No compression baked in — serve with HTTP gzip/br if wanted; u16 raw
  decodes to a GPU texture in one typed-array view.
- Static fields: same encoding, range in their manifest entry.

## manifest.json

    {
      "version": 1,
      "grid": { "nx": 2048, "ny": 2048, "h": 0.003068 },
      "sim":  { ... app passthrough: app name, re, mach, D, cx, cy, R, flags },
      "fields": [
        { "name": "vort",  "file": "vort_%05d.bin",
          "kind": "diverging",  "range": [-8.0, 8.0] },
        { "name": "speed", "file": "speed_%05d.bin",
          "kind": "sequential", "range": [0.0, 1.6] }
      ],
      "static": [
        { "name": "mask", "file": "mask.bin", "range": [0.0, 1.0] }
      ],
      "frames": [
        { "t": 0.0, "minmax": { "vort": [-6.1, 5.9], "speed": [0.0, 1.31] } },
        ...
      ]
    }

- `frames[i]` ↔ file index i (NNNNN = i, zero-padded to 5).
- `kind` is a rendering hint (diverging = signed, sequential = magnitude);
  the viewer picks colormaps, the format never bakes colors (that is what
  the PNG export is for).
- Live protocol: poll manifest.json; `frames.length` grows monotonically; a
  frame's files are fully written before the manifest naming them lands
  (manifest rename is the commit point).

## Writer API (`src/dns/webframes.*`; test `webframes`)

    WebFrames wf(dir, w, h, simJson);
    wf.addField("vort", "diverging", -8.0, 8.0);
    wf.addField("speed", "sequential", 0.0, 1.6);
    wf.writeStatic("mask", maskPtr, 0.0, 1.0);
    wf.writeFrame(t, {vortPtr, speedPtr});   // per frame, in declared order

`RunOutput` (`dns/output.cpp`) writes one such directory per published plane
(`web_<out><suffix>`; the `sim` object carries `plane` and `off`), with the
run's frame field set (vort + speed, plus `temp` where the run publishes it).

**The mask is `static` by contract, and a MOVING body breaks that assumption.**
A rotor run (`--rotor-mach`) has a body whose occupancy changes with azimuth. `RunOutput::updateMask` refreshes the mask the sinks that read it
PER FRAME use — the PNG frames and the live shared-memory planes, so images and
the cockpit view follow the blade. This format is not one of them: `mask.bin`
here, and the one-shot `<prefix>_mask.bin`, both hold the mask the run STARTED
with. Making them time-dependent is a format change (a per-frame mask field and
a version bump), not a writer fix — do it deliberately or not at all.

## Sizes

u16, 2048²: 8.4 MB/field/frame raw, ~2-4 MB gzipped on the wire; 512²:
0.5 MB. Same cadence flag as the PNGs (`--pdt`-style, own `--wdt`).

## The run `.json` (not the web manifest)

`<prefix>.json` (`dns/output.cpp`) always carries `nx, ny, nz, frames,
frameDt, D, re, mach, cs, les, sgs, body, far, physics, kick, msPerStep`.
Appended keys, each present ONLY when it applies (every other run's json
reads exactly as before):

| key | when | value |
|---|---|---|
| `rans` | a RANS run (channel or flow) | `"SA-neg"`, `"SST-2003m"`, `"SST-Vm"` |
| `steady` | `dns channel --steady-scheme implicit` | `"implicit"` |
| `ransMode` | `dns flow` RANS, or a compressible `--steady` run | `"urans"`, `"steady-explicit"`, `"steady-implicit-nk"` (`--steady-scheme implicit`) |
| `ransInflow` | `dns flow` RANS | the describe's freestream turbulence string, e.g. `"nu~inf/nu = 3 (TMR)"` |

## `<prefix>_surface.csv` (`--emit surface`, `dns flow --ogrid`)

Written once, at the end of the run. Line 1 is `# ` + the run label
(`dns/rans_label`; `DNS` on a DNS time march), line 2 the header, then one row
per real wall face (i fastest, then k):

| column | meaning |
|---|---|
| `i`, `k` | the wall face's mesh indices (face between ghost row 0 and fluid row 1) |
| `x`, `y`, `z` | face midpoint (mean of its four wall nodes) |
| `theta_deg` | angle from the UPSTREAM axis about the body centre: 0 at front stagnation, ±180 at the rear, positive over the top (y > cy) |
| `area` | face area |
| `cp` | (p_f − p∞)/(½ρ∞U∞²), p_f the flux's own face pressure |
| `cf` | viscous traction on the body along the downstream surface tangent (from front stagnation toward the rear on each side), / ½ρ∞U∞² — negative = reversed flow |
| `tau_x`, `tau_y`, `tau_z` | viscous traction ON the body (force / area): total momentum flux minus p_f S |
| `yplus` | d1 u_τ/ν_f, u_τ from the TANGENTIAL viscous traction, ρ_f, ν_f on the face |
| `d1` | the first fluid cell's wall distance (exact, to the discrete wall) |
| `qw` | energy flux into the body per area (conduction at a resting wall) |

## `<prefix>_surface.csv` (`--emit surface`, compressible immersed body)

The same file name and the same "written once, at the end" contract, for the
compressible IMMERSED ghost wall (no `--ogrid`): the surface tally
(`DeviceCompGhost::force`, which every cd / cl / thrust / torque in the stats
file comes from) per interface face, split (`DeviceCompGhost::surfaceFaces`).
It reads the final state only, so a run with the flag writes stats
bit-identical to one without. 10 significant digits (the parts are gauge
forces summed over thousands of faces). Line 1 is `# ` + the run label (the
RANS label, else the LES label `LES <model> Cs=...`, else `DNS`; an
`--rotor-frame mrf` run appends `[MRF frozen rotor: an APPROXIMATION]`), then
`|`-separated fields: `immersed ghost wall: ...`, `gauge on pInf P`,
`pivot X Y Z` (the moment point: the rotor shaft, else the origin),
`qRef*span Q` (cd = fx / Q), `ctDen` = ρn²D⁴ and `cqDen` = ρn²D⁵ (0 without a
rotor; D = cqDen / ctDen), `shaft A` (axis index, −1 without a rotor). Line 2
is the header, then one row per interface face in table order. All forces are
the flux from the FLUID into the BODY (the force on the body), solver units:

| column | meaning |
|---|---|
| `x`, `y`, `z` | face centre (physical; the focal remap of the reference face) |
| `axis`, `sign` | the face normal's axis (0/1/2) and the side the ghost is on (+1/−1) |
| `area` | \|s\| |
| `sx`, `sy`, `sz` | area vector oriented into the body: sign·area along `axis` (Cartesian), the oriented metric S (focal). Σ s = 0 (closed surface) |
| `p` | the KG face pressure the flux uses, ½(p_fluid + p_ghost), minus pInf |
| `fx`, `fy`, `fz` | the full face momentum flux minus pInf s (gauge); Σ = the tally to round-off |
| `fpx`, `fpy`, `fpz` | its pressure part p s |
| `fcx`, `fcy`, `fcz` | its convective part. The interface faces are impermeable (`DeviceCompGhost::applyRhs`), so `fc` is exactly 0 at a resting wall and under MRF (no advection relative to the wall / frame); on a moving rotor (`--rotor-frame moving`) it is the wall's own normal motion ρ u_wall·n u, u_wall sampled at the face centre |
| `fvx`, `fvy`, `fvz` | the viscous stress, f − fp − fc |

`tools/parity/split_dns.py` sorts a rotor's rows into hub and blade and
prints CT / CQ per part against the same sort of an OpenFOAM case.
