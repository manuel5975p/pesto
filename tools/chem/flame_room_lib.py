#!/usr/bin/env python3
"""The numpy half of tools/chem/flame_room.py (no Blender): the run's files,
the premixed / diffusion split of the light, the chemiluminescence colours,
the proxy lights, the crop and the burner's real scale. flame_room.py
(inside Blender) imports it; its self-test runs without Blender:

    python3 tools/chem/flame_room_lib.py --selftest [PREFIX]

PREFIX (optional): a low-Mach --equivalence burner run with --emit volume and
--emit stats (gpu_burner_round's output): the split of the LAST frame must
reproduce the solver's burnPre / burnDif of the stats row at the frame's
time (the port of dns::lmPremixedWeight against the solver's own sums).
Exit 77 when PREFIX's files are absent.

Colour and brightness basis (docs/chem.md "Low-Mach renderer"):
  inner cone (rich premixed, w_p): CH* 431.4 nm (1.0) and the C2* Swan bands
    516.5 / 473.7 / 563.5 nm (1.2 / 0.4 / 0.25: C2*/CH* rises with phi, ~1.5
    band-integrated at phi 1.3 for CH4/air, e.g. Kojima et al. 2005,
    Hardalupas & Orain 2004) over a weak CO2* continuum (0.2): blue-green;
  outer cone (diffusion, 1 - w_p): the rich products' CO and H2 burning in
    air -- the CO + O -> CO2* continuum (broad, 330-600 nm, peak ~400 nm) and
    a trace of CH* (0.15): violet-blue; per unit heat release DIMMER by
    --dif-gain (0.35, a DISPLAY choice, not a measured yield);
  no soot (laminar phi 1.3: blue flame, no yellow), burnt gas not drawn
    (CO2/H2O radiate in the IR): the light is omega > 0 only.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from turb_render import XYZ2RGB, chemi_rgb, cie_xyz  # noqa: E402


# --------------------------------------------------------------- the split --

def premixed_weight(xi):
    """dns::lmPremixedWeight (dns/lowmach.hpp), the ONE definition of the
    premixed / diffusion split, ported: w_p = smoothstep(-0.2, 0.2, xi) =
    t^2 (3 - 2t), t = clamp((xi + 0.2)/0.4, 0, 1). FP64, any array shape."""
    t = np.clip((np.asarray(xi, np.float64) + 0.2) / 0.4, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def split(omega, xi, ref):
    """(em_pre, em_dif) float32 = max(omega, 0)/ref times w_p(xi) and 1 - w_p:
    the inner and outer cones' emission grids (their sum is the single grid
    max(omega, 0)/ref of the one-colour renderer, to FP32 rounding)."""
    em = np.maximum(np.asarray(omega, np.float64), 0.0) / ref
    w = premixed_weight(xi)
    return (em * w).astype(np.float32), (em * (1.0 - w)).astype(np.float32)


# --------------------------------------------------------------- colours ----

INNER_LINES = ((431.4, 1.0), (516.5, 1.2), (473.7, 0.4), (563.5, 0.25))
INNER_CONT = 0.2
OUTER_LINES = ((431.4, 0.15),)
OUTER_CONT = 1.0


def co2_continuum(lam):
    """The CO + O -> CO2* chemiluminescence continuum's shape (relative): broad
    from the UV with its maximum near 400 nm, falling through the green
    (Gaydon, The Spectroscopy of Flames); a skewed Gaussian (40 nm below the
    peak, 50 above), unit peak."""
    s = np.where(lam < 400.0, 40.0, 50.0)
    return np.exp(-0.5 * ((lam - 400.0) / s) ** 2)


def spectrum_rgb(lines, cont, cont_shape=None, norm="luminance"):
    """Linear sRGB of `lines` [(nm, weight)] plus a continuum carrying `cont`
    in total over 350-600 nm (flat, or cont_shape's relative shape),
    gamut-mapped as turb_render.chemi_rgb (add equal-energy white until the
    smallest channel is 0). norm "luminance": unit luminance (chemi_rgb's);
    "power": the weights rescaled to unit total radiant power, so two
    spectra compare at EQUAL power (the eye's efficacy kept)."""
    lam_l = np.array([l for l, _ in lines])
    w = np.array([v for _, v in lines])
    if norm == "power":
        tot = w.sum() + cont
        w, cont = w / tot, cont / tot
    xyz = (cie_xyz(lam_l) * w[:, None]).sum(0)
    lam = np.arange(350.0, 600.0 + 1e-9, 1.0)
    shape = np.ones_like(lam) if cont_shape is None else cont_shape(lam)
    shape = shape / shape.sum()
    xyz = xyz + cont * (cie_xyz(lam) * shape[:, None]).sum(0)
    rgb = XYZ2RGB @ xyz
    rgb = rgb - min(0.0, rgb.min())
    if norm == "power":
        return rgb
    return rgb / (rgb @ np.array([0.2126, 0.7152, 0.0722]))


LUM = np.array([0.2126, 0.7152, 0.0722])


def inner_rgb(norm="luminance"):
    return spectrum_rgb(INNER_LINES, INNER_CONT, None, norm)


def outer_rgb(norm="luminance"):
    return spectrum_rgb(OUTER_LINES, OUTER_CONT, co2_continuum, norm)


def cone_colours():
    """(inner, outer) emission colours for the renderer at EQUAL radiant power
    per unit grid value, scaled together so the inner's largest channel is
    1: the outer's lower luminance at equal power is the eye's (violet)."""
    ci, co = inner_rgb("power"), outer_rgb("power")
    s = ci.max()
    return tuple(float(c) for c in ci / s), tuple(float(c) for c in co / s)


def display_rgb(rgb):
    """Unit-max colour for a Blender emission node."""
    rgb = np.asarray(rgb, np.float64)
    return tuple(float(c) for c in rgb / rgb.max())


# --------------------------------------------------------------- the run ----

class Run:
    """The --emit volume files of one dns run (nz, ny, nx float32 per frame)
    and its .json (nx, ny, nz, frames, frameDt; low-Mach: "rim" {x, ri, ro,
    h, T} in box units, "si" {deltaD m, SL m/s})."""

    def __init__(self, prefix):
        self.prefix = prefix
        m = json.load(open(prefix + ".json"))
        self.meta = m
        self.nx, self.ny, self.nz, self.frames = m["nx"], m["ny"], m["nz"], m["frames"]
        self.fdt = m.get("frameDt", 1.0)
        self.n3 = self.nx * self.ny * self.nz
        self.rim = m.get("rim")
        self.si = m.get("si")
        size = os.path.getsize(prefix + "_omega.bin")
        on_disk = size // (4 * self.n3)  # a run still writing has frames 0 in its json
        self.frames = min(self.frames, on_disk) if self.frames > 0 else on_disk

    def field(self, name, f):
        a = np.fromfile(f"{self.prefix}_{name}.bin", "<f4", self.n3,
                        offset=4 * self.n3 * f)
        return a.reshape(self.nz, self.ny, self.nx)

    def omega(self, f):
        return self.field("omega", f)

    def has(self, name):
        return os.path.exists(f"{self.prefix}_{name}.bin")


def crop_lo(run, spec):
    """The low crop in cells: an int, or 'auto' = the rim's top face (the
    json "rim" record: round(x/h)), else 0."""
    if spec == "auto":
        if not run.rim:
            return 0
        return int(round(run.rim["x"] / run.rim["h"]))
    return int(spec)


def real_scale(run, height, x_cells, delta_d=0.0):
    """Metres per cell: --height (m) over the cropped cells when given (> 0),
    else the json's rim h x delta_D (delta_d in m when given, else the json's
    si.deltaD: the run's real scale). None if neither."""
    if height > 0:
        return height / x_cells
    dd = delta_d if delta_d > 0 else (run.si["deltaD"] if run.si else 0.0)
    if run.rim and dd > 0:
        return run.rim["h"] * dd
    return None


def tube_geometry(run, voxel):
    """(ri, ro) of the burner tube in metres from the json "rim" (box units
    / h cells x voxel), or None without a rim record."""
    if not run.rim:
        return None
    h = run.rim["h"]
    return run.rim["ri"] / h * voxel, run.rim["ro"] / h * voxel


def light_proxies(grids, gains, voxel, origin, nband, tile_n, power_per_unit):
    """[(location, radius, power W, rgb)] per (height band, span tile) over the
    emission grids (each indexed (x_world, y_world, z_world)), grid g weighted
    by gains[g] = (strength, rgb): the band's summed power and its power-
    weighted colour and centroid."""
    em0 = grids[0]
    nzt, ny, nx = em0.shape[0] // tile_n, em0.shape[1], em0.shape[2]
    out = []
    edges = np.linspace(0, nx, nband + 1).astype(int)
    xs = (np.arange(nzt) + 0.5) * voxel
    ys = (np.arange(ny) + 0.5) * voxel
    zs = (np.arange(nx) + 0.5) * voxel
    for t in range(tile_n):
        for b in range(nband):
            w = 0.0
            col = np.zeros(3)
            for g, (s, rgb) in zip(grids, gains):
                blk = g[t * nzt:(t + 1) * nzt, :, edges[b]:edges[b + 1]]
                pw = s * float(blk.sum(dtype=np.float64))
                w = w + s * blk.astype(np.float64)
                col += pw * np.asarray(rgb, np.float64)
            tot = float(np.sum(w))
            if tot <= 0:
                continue
            px = (w.sum((1, 2)) @ xs) / tot
            py = (w.sum((0, 2)) @ ys) / tot
            pz = (w.sum((0, 1)) @ zs[edges[b]:edges[b + 1]]) / tot
            sy = math.sqrt(max((w.sum((0, 2)) @ ys ** 2) / tot - py ** 2, 0.0))
            loc = origin + np.array([px + t * nzt * voxel, py, pz])
            out.append((loc, max(sy, 2 * voxel), power_per_unit * tot * voxel ** 3,
                        tuple(col / col.max())))
    return out


def omega_ref(run, frames, x0, x1):
    pick = frames[:: max(1, len(frames) // 8)]
    vals = [np.percentile(np.maximum(run.omega(f)[:, :, x0:x1], 0), 99.9) for f in pick]
    return float(np.median(vals))


# --------------------------------------------------------------- self-test --

def _stats(prefix):
    import csv
    return list(csv.DictReader(open(prefix + "_stats.csv")))


def solver_check(prefix):
    """The last frame's split vs the solver's burnPre / burnDif at that time:
    each cell float32(omega w_p) as the runner forms it, summed in FP64 x dV.
    Returns (ok, message) or None when the files are absent."""
    need = [prefix + s for s in (".json", "_omega.bin", "_findex.bin", "_stats.csv")]
    if not all(os.path.exists(p) for p in need):
        return None
    run = Run(prefix)
    f = run.frames - 1
    # The run must END on a frame time (--tt a multiple of --fdt past
    # --frame-start): its last frame and its last (forced) stats row are the
    # same state.
    row = _stats(prefix)[-1]
    t_frame = float(row["t"])
    h = run.rim["h"] if run.rim else None
    if h is None:
        return False, "the run's json has no h (needs the rim record)"
    om = run.omega(f).astype(np.float64)
    w = premixed_weight(run.field("findex", f))
    pre = (om * w).astype(np.float32).astype(np.float64).sum() * h ** 3
    dif = (om * (1.0 - w)).astype(np.float32).astype(np.float64).sum() * h ** 3
    # The CSV prints 6 significant digits: the port must print the SAME row
    # (agreement to the last printed digit; half a unit of it as the bound
    # where a value sits on a rounding edge).
    def same(v, s):
        if f"{v:.6g}" == s.strip():
            return True
        r = float(s)
        return abs(v - r) <= 0.5 * 10.0 ** (math.floor(math.log10(abs(r))) - 5) * (1 + 1e-9)

    ok = same(pre, row["burnPre"]) and same(dif, row["burnDif"])
    msg = (f"frame {f} t={t_frame:g}: burnPre {pre:.9g} vs the row's {row['burnPre']}, "
           f"burnDif {dif:.9g} vs {row['burnDif']} (6 printed digits)")
    return ok, msg


def selftest(prefix=None):
    ok = True

    def check(c, what):
        nonlocal ok
        print(("  ok   " if c else "  FAIL ") + what)
        ok &= bool(c)

    # (a) the port of lmPremixedWeight: closed form, clamps, symmetry.
    xi = np.array([-1.0, -0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 1.0])
    ref = []
    for x in xi:
        t = min(max((x + 0.2) / 0.4, 0.0), 1.0)
        ref.append(t * t * (3.0 - 2.0 * t))
    check(np.array_equal(premixed_weight(xi), np.array(ref)),
          "w_p = smoothstep(-0.2, 0.2, xi) bit for bit (the C++ expression)")
    check(premixed_weight(0.0) == 0.5 and premixed_weight(-0.2) == 0.0 and
          premixed_weight(0.2) == 1.0, "w_p(0) = 1/2, 0 at -0.2, 1 at +0.2")
    g = np.linspace(-1, 1, 2001)
    check(np.all(np.diff(premixed_weight(g)) >= 0) and
          np.max(np.abs(premixed_weight(g) + premixed_weight(-g) - 1)) < 1e-15,
          "w_p monotone, w_p(xi) + w_p(-xi) = 1")
    # (b) the split sums to the single grid.
    rng = np.random.default_rng(1)
    om = rng.normal(0.5, 1.0, (8, 9, 10)).astype(np.float32)
    x = rng.uniform(-1, 1, om.shape).astype(np.float32)
    pre, dif = split(om, x, 0.7)
    em = np.maximum(om.astype(np.float64), 0) / 0.7
    e = np.max(np.abs(pre.astype(np.float64) + dif - em)) / em.max()
    check(e < 1e-6, f"em_pre + em_dif = max(omega, 0)/ref ({e:.1e})")
    check(np.all(pre >= 0) and np.all(dif >= 0) and np.all(pre[om <= 0] == 0),
          "both grids >= 0, zero where omega <= 0 (burnt gas not drawn)")
    # (c) colours: deterministic, the shipped chemi_rgb untouched, hues.
    ci, co = inner_rgb(), outer_rgb()
    check(ci.tobytes() == inner_rgb().tobytes() and co.tobytes() == outer_rgb().tobytes(),
          "colours deterministic")
    base = chemi_rgb()
    check(np.array_equal(spectrum_rgb(((431.4, 1.0), (516.5, 0.6), (473.7, 0.3)), 0.2), base),
          "spectrum_rgb reproduces turb_render.chemi_rgb's weights bit for bit")
    print(f"    inner (CH* + rich C2*) {ci[0]:.3f} {ci[1]:.3f} {ci[2]:.3f}; "
          f"outer (CO2* + CH* trace) {co[0]:.3f} {co[1]:.3f} {co[2]:.3f}")
    check(ci[1] > ci[0] and ci[2] > ci[0], "inner blue-green (g, b > r)")
    check(co[2] > co[1] and co[0] > co[1], "outer violet-blue (b > g, r > g)")
    check(ci[1] / ci[2] > co[1] / co[2], "the inner cone greener than the outer")
    pi_, po = cone_colours()
    lr = (np.array(po) @ LUM) / (np.array(pi_) @ LUM)
    print(f"    renderer colours (equal power, inner max 1): inner {np.round(pi_, 3)}, "
          f"outer {np.round(po, 3)}; outer/inner luminance at equal power {lr:.3f}")
    check(max(pi_) == 1.0 and 0 < lr < 1, "equal-power colours: the violet outer "
          "reads dimmer to the eye")
    # (d) proxies: total power = the gains' sum; colour between the two.
    grids = [np.ones((4, 4, 8), np.float32), 2 * np.ones((4, 4, 8), np.float32)]
    px = light_proxies(grids, [(1.0, ci), (0.35, co)], 0.1, np.zeros(3), 2, 1, 1.0)
    tot = sum(p[2] for p in px)
    want = (1.0 * 128 + 0.35 * 256) * 1e-3
    check(abs(tot / want - 1) < 1e-12, f"proxy power = sum of gain x grid ({tot:.6g})")
    # (e) against the solver.
    if prefix:
        r = solver_check(prefix)
        if r is None:
            print(f"  skip solver check: {prefix} files absent")
            if ok:
                print("SELFTEST SKIP (77)")
                return 77
        else:
            check(r[0], "solver split: " + r[1])
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2] if len(sys.argv) > 2 else None))
    print(__doc__)
    sys.exit(1)
