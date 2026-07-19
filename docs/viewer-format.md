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

## Writer API (`src/dns/webframes.*`, implemented; test `webframes`)

    WebFrames wf(dir, grid, simJson);
    wf.field("vort", Kind::Diverging, -8.0, 8.0);
    wf.field("speed", Kind::Sequential, 0.0, 1.6);
    wf.writeStatic("mask", maskPtr, 0.0, 1.0);
    wf.write(t, {{"vort", vortPtr}, {"speed", speedPtr}});   // per frame

Apps choose the set: defaults mirror today's vort+speed; extra primitives
(p, rho, u, v) opt-in via a `--wfields` flag.

## Sizes

u16, 2048²: 8.4 MB/field/frame raw, ~2-4 MB gzipped on the wire; 512²:
0.5 MB. Same cadence flag as the PNGs (`--pdt`-style, own `--wdt`).
