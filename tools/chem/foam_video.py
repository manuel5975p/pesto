#!/usr/bin/env python3
"""Side-by-side videos, dns vs OpenFOAM v2606 reactingFoam on the IDENTICAL
mesh, of the flame grounding cases (docs/chem.md "Against OpenFOAM
reactingFoam"): G2a slot Bunsen, G2b Darrieus-Landau growth, G3a diffusion
jet.

Inputs, under runs/flames/foam/video/CASE/ (the dense-cadence reruns of the
committed cases, same settings):
  * dns --emit bin frames: <run>/flame.json + flame_<field>.bin, one (ny, nx)
    float32 plane per frame, i fastest. The premixed runs are the z-extruded
    nz 8 twins: their planes carry vort = |curl u| and speed = |u| (2D frames
    carry no velocity). dns frame k sits at the first step with t >= k fdt.
  * foam_mclin/: a reactingFoam case whose writeObjects dump wrote T, U, Qdot
    (+ F, O) every fdt; ascii or binary (LSB, scalar 64) internalField, cell
    order = blockMesh (i, j), i fastest.
Box units come from the committed case (foam_flame.context: si.json, the
oracle, the mesh); observables through foamflame/observe2d, observejet (the
one measurement path of the compare tables).

  foam_video.py render CASE [--workers 8] [--fps F]  video/CASE.mp4 + CASE.png
  foam_video.py check CASE      the rerun against the committed run

Frame: 1920 x 1080, dark; the jet / flow axis x points UP (fresh gas enters at
the bottom) in every panel; one fixed colour scale per field, shared by all
solvers of a video (from the last third of the clip). T: fire_video's fire
map on theta; |curl u|: inferno on sqrt(|curl u| / max); |u|, Z: viridis;
omega: inferno. Text: ffmpeg drawtext rendered into alpha masks. libx264
yuv420p. numpy + ffmpeg only.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import foam_flame  # noqa: E402  (context: spec, oracle, si, committed root)
from fire_video import fire  # noqa: E402
from foamflame import cases, observe2d, observejet  # noqa: E402
from turb_render import FONT, INFERNO, VIRIDIS, lut, resample  # noqa: E402

VIDEO = os.path.join(foam_flame.ROOT, "video")
FFMPEG = "/usr/bin/ffmpeg"
FONT_B = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"
W, H = 1920, 1080
FOOT_SIZE, FOOT_STEP = 14, 18   # footer text px, line pitch
BG = np.array([0.055, 0.058, 0.066], np.float32)
FG = np.array([0.87, 0.87, 0.87], np.float32)
DIM = np.array([0.60, 0.61, 0.64], np.float32)
COL = {"dns": np.array([1.00, 0.63, 0.20], np.float32),      # orange
       "foam": np.array([0.38, 0.72, 1.00], np.float32),     # sky blue
       "dns100": np.array([0.93, 0.42, 0.72], np.float32),   # pink
       "theory": np.array([0.75, 0.75, 0.75], np.float32)}

SQRT = ("vort", "omega")      # fields drawn on a square-root scale
LW = {"foam": 4.0, "dns": 2.0, "dns100": 2.0}   # foam wide underneath: overlaps stay visible

# --- the cases -----------------------------------------------------------------
# runs: (key, label, dir); the first is the dns run the comparison is about.

CFG = {
    "G2a": {
        "kind": "bunsen", "fdt": 0.1, "fps": 20, "unit": "τ",
        "runs": [("dns", "dns  --nscbc-tinflow", "G2a/dns_tinflow_nz8"),
                 ("foam", "OpenFOAM reactingFoam", "G2a/foam_mclin")],
        "fields": ["theta", "vort", "speed"],
        "title": "G2a  2D slot Bunsen flame: dns vs OpenFOAM reactingFoam, identical mesh",
        "subtitle": "U_jet 5 S_L, W 12 δ_th, burnt coflow (T/Tu 5) at 1 S_L; "
                    "8 cells per δ_th; fields from the same initial cone",
        "foot": [
            "dns: compressible Navier-Stokes, explicit central KG flux, FP32, "
            "SSP-RK3 at acoustic CFL 0.4; species MC-MUSCL; per-row NSCBC inflow "
            "WITH --nscbc-tinflow (row T held, as foam's fixedValue T: the "
            "configuration that matches foam), NSCBC outflow p → p0, 16-cell "
            "sink bands; |∇×u|, |u| from a k plane of the z-extruded nz 8 twin "
            "(its temperature frames equal the 2D run's byte for byte).",
            "OpenFOAM v2606 reactingFoam (mclin): PISO nOuter 1 nCorr 2, Euler "
            "dt 4e-3 τ, FP64; U, h Gauss linear, Y MUSCL; fixedValue U, T, Y "
            "inflow rows, zeroGradient p; waveTransmissive p outlet; cyclic y; "
            "|∇×u| by central differences of U (dns: its frame's end rows "
            "i = 0, nx−1 show their neighbours, the stencil wraps across x).",
            "Mesh: 392 × 288 cells, h = δ_th/8 = 0.200 δ_D, every dns cell a "
            "foam cell. Model: single step F → P, σ 5, β 8, Pr 0.72, Le 1, "
            "constant μ, M 0.02.   τ = δ_D/S_L.",
        ],
    },
    "G2b": {
        "kind": "wrinkle", "fdt": 0.1, "fps": 15, "unit": "τ",
        "runs": [("dns", "dns", "G2b/dns_nz8"),
                 ("foam", "OpenFOAM reactingFoam", "G2b/foam_mclin")],
        "fields": ["theta", "vort", "speed"],
        "title": "G2b  Darrieus–Landau growth of one wrinkle mode (σ 5): "
                 "dns vs OpenFOAM reactingFoam, identical mesh",
        "subtitle": "λ = L_y = 30 δ_th = 48.05 δ_D (2.1 λ_c), initial front "
                    "amplitude 0.032 δ_D (kA 0.004), inflow at S_L; the wrinkle "
                    "stays LINEAR over t ≤ 20 τ (no cusp yet)",
        "foot": [
            "dns: compressible Navier-Stokes, explicit central KG flux, FP32, "
            "SSP-RK3 at acoustic CFL 0.4; species MC-MUSCL; NSCBC inflow u = "
            "S_L (etaIn 100), outflow p → p0 − 4, 16-cell sink bands; |∇×u|, "
            "|u| from a k plane of the z-extruded nz 8 twin (temperature frames "
            "byte-identical to the 2D run; the frame's end rows i = 0, nx−1 show "
            "their neighbours: its stencil wraps across x).",
            "OpenFOAM v2606 reactingFoam (mclin): PISO nOuter 1 nCorr 2, Euler "
            "dt 4e-3 τ, FP64; U, h Gauss linear, Y MUSCL; fixedValue U = S_L, T, "
            "Y inlet, waveTransmissive p outlet, cyclic y. Amplitude a = mode 1 "
            "of x_I(y) = h Σ_i Y (both solvers, one code path).",
            "Mesh: 288 × 240 cells, h = δ_th/8 = 0.200 δ_D. Model: single step "
            "F → P, σ 5, β 8, Pr 0.72, Le 1, constant μ, M 0.02. Theory: "
            "Matalon–Matkowsky 1982 (O(k l_f), large β); inviscid Landau = "
            "upper bound.   τ = δ_D/S_L.",
        ],
    },
    "G3a": {
        "kind": "jet", "fdt": 0.4, "fps": 15, "unit": "τ",
        "runs": [("dns", "dns  etaIn 1000", "G3a/dns_etain1000"),
                 ("foam", "OpenFOAM", "G3a/foam_mclin"),
                 ("dns100", "dns  etaIn 100 (default)", "G3a/dns_etain100")],
        "fields": ["theta", "omega", "Z"],
        "title": "G3a  2D slot-jet diffusion flame: dns vs OpenFOAM reactingFoam, "
                 "identical mesh",
        "subtitle": "W 16 l, fuel jet and oxidizer coflow at 1 U, Z_st 0.5 "
                    "(white line), Da 1000; dns at etaIn 1000 matches foam, the "
                    "default etaIn 100 passes 7% more fuel: a longer flame (§68)",
        "foot": [
            "dns: compressible Navier-Stokes, explicit central KG flux, FP32, "
            "SSP-RK3 at acoustic CFL 0.4; species MC-MUSCL, ONE limiter shared by "
            "the species; per-row NSCBC inflow (u, ρ) on the first cell, species "
            "pinned to the Burke–Schumann row, --nscbc-etain 1000 (left) | 100 = "
            "default (right); NSCBC outflow p → p0, 16-cell sink bands.",
            "OpenFOAM v2606 reactingFoam (mclin): PISO nOuter 1 nCorr 2, Euler "
            "dt 0.025 τ, FP64; U, h Gauss linear, each species its own MUSCL; "
            "per-face fixedValue U, T, Y inlet (the same rows), waveTransmissive "
            "p outlet, cyclic y. No velocity panels: dns --mode diffusion is 2D "
            "only and writes no velocity frames.",
            "Mesh: 416 × 192 cells, h = l/4. Model: F + O → 2P, ω = A ρ² Y_F Y_O "
            "e^(−Ta/T), T_ad/T0 5, β 5, Da 1000, Pr 0.72, Le 1, constant μ, M "
            "0.05.   τ = l/U_jet.",
        ],
    },
}

# Fast jets (docs/chem.md "fast jets", "thin, fast nozzles"): the Bunsens are
# dense reruns (fdt 0.09 = 30 / 40 foam steps); the jets read the committed
# runs' own 1 l/U frames.
_FOOT_BUNSEN = [
    "dns: compressible Navier-Stokes, explicit central KG flux, FP32, SSP-RK3 "
    "at acoustic CFL 0.4; species MC-MUSCL; per-row NSCBC inflow WITH "
    "--nscbc-tinflow (row T held, as foam's fixedValue T), NSCBC outflow p → "
    "p0, 16-cell sink bands; fields from a k plane of the z-extruded nz 8 twin.",
    "OpenFOAM v2606 reactingFoam (mclin): PISO nOuter 1 nCorr 2, Euler dt {dt} "
    "τ, FP64; U, h Gauss linear, Y MUSCL; fixedValue U, T, Y inflow rows, "
    "zeroGradient p; waveTransmissive p outlet; cyclic y; |∇×u| by central "
    "differences of U.",
    "Mesh: {nx} × {ny} cells, h = δ_th/8 = 0.200 δ_D, every dns cell a foam "
    "cell. Model: single step F → P, σ 5, β 8, Pr 0.72, Le 1, constant μ, M "
    "{M} (jet Mach {Mj}).   τ = δ_D/S_L.",
]
_FOOT_JET = [
    "dns: compressible Navier-Stokes, explicit central KG flux, FP32, SSP-RK3 "
    "at acoustic CFL 0.4; species MC-MUSCL, ONE limiter shared by the species; "
    "per-row NSCBC inflow (u, ρ), species pinned to the Burke–Schumann row, "
    "--nscbc-etain 1000 | 100 = default; NSCBC outflow p → p0, 16-cell sink "
    "bands. No velocity panels: dns --mode diffusion is 2D only and writes no "
    "velocity frames.",
    "OpenFOAM v2606 reactingFoam (mclin): PISO nOuter 1 nCorr 2, Euler dt 0.025 "
    "τ, FP64; U, h Gauss linear, each species its own MUSCL; per-face "
    "fixedValue U, T, Y inlet, waveTransmissive p outlet (lInf = box/4), "
    "cyclic y.",
    "Mesh: {nx} × {ny} cells, h = l/4. Model: F + O → 2P, ω = A ρ² Y_F Y_O "
    "e^(−Ta/T), T_ad/T0 5, β 5, Da 1000, Pr 0.72, Le 1, constant μ, M 0.2 (jet "
    "Mach).   τ = l/U_jet.",
]
for _c, _U, _W, _M, _nx, _ny, _dt in (("G4b", 15, 12, 0.02, 1000, 288, "3e-3"),
                                      ("G4c", 20, 6, 0.015, 696, 144, "2.25e-3")):
    CFG[_c] = {
        "kind": "bunsen", "fdt": 0.09, "fps": 20, "unit": "τ",
        "window": (20.0, 25.0), "tmax": 25.0,
        "runs": [("dns", "dns  --nscbc-tinflow", f"{_c}/dns_tinflow_nz8"),
                 ("foam", "OpenFOAM reactingFoam", f"{_c}/foam_mclin")],
        "fields": ["theta", "vort", "speed"],
        "title": f"{_c}  2D slot Bunsen, fast jet: dns vs OpenFOAM reactingFoam, "
                 "identical mesh",
        "subtitle": f"U_jet {_U} S_L, W {_W} δ_th, jet Mach {_U * _M:.2g}, burnt "
                    "coflow (T/Tu 5) at 1 S_L; 8 cells per δ_th",
        "foot": [s.format(dt=_dt, nx=_nx, ny=_ny, M=_M, Mj=f"{_U * _M:.2g}")
                 for s in _FOOT_BUNSEN],
    }
for _c, _W, _nx, _ny, _tt, _win, _yl, _ts, _ys in (
        ("G5", 16, 416, 192, 240, (200.0, 240.0), (40.0, 70.0), 20, 5),
        ("G6", 8, 152, 160, 120, (80.0, 120.0), (6.0, 12.0), 10, 1)):
    CFG[_c] = {
        "kind": "jet", "fdt": 1.0, "fps": 15, "unit": "τ",
        "tmax": float(_tt), "window": _win, "ylim": _yl, "tstep": _ts, "ystep": _ys,
        "runs": [("dns", "dns  etaIn 1000", f"../{_c}/dns_c4_etain1000"),
                 ("foam", "OpenFOAM", f"../{_c}/foam_mclin_c4"),
                 ("dns100", "dns  etaIn 100 (default)", f"../{_c}/dns_c4")],
        "fields": ["theta", "omega", "Z"],
        "title": f"{_c}  2D slot-jet diffusion flame at jet Mach 0.2: dns vs "
                 "OpenFOAM reactingFoam, identical mesh",
        "subtitle": f"W {_W} l, fuel jet and oxidizer coflow at 1 U, Z_st 0.5 "
                    "(white line), Da 1000; the box's acoustic mode settles "
                    "slowly at M 0.2",
        "foot": [s.format(nx=_nx, ny=_ny) for s in _FOOT_JET],
    }

FIELD_TITLE = {"theta": "temperature  θ = (T − T_u)/(T_b − T_u)",
               "vort": "vorticity  |∇×u|  [S_L/δ_D]  (√ scale)",
               "speed": "speed  |u|  [S_L]",
               "omega": "heat release  ω  [ρ0 U/l]",
               "Z": "mixture fraction  Z"}
FIELD_TITLE_JET = {"theta": "temperature  θ = (T − T0)/(T_ad − T0)",
                   "omega": "reaction rate  ω  [ρ0 U/l]  (√ scale)",
                   "Z": "mixture fraction  Z"}

# --- OpenFOAM fields -----------------------------------------------------------

_FMT = re.compile(rb"format\s+(\w+)\s*;")
_CLS = re.compile(rb"class\s+(\w+)\s*;")


def read_internal(path, ncell):
    """internalField of a volScalarField / volVectorField file, ascii or
    binary (LSB, scalar 64): float64 (ncell,) or (ncell, 3); a uniform value
    is broadcast. ncell must match a nonuniform list."""
    with open(path, "rb") as f:
        raw = f.read()
    m = _FMT.search(raw, 0, 4000)
    binary = m is not None and m.group(1) == b"binary"
    c = _CLS.search(raw, 0, 4000)
    nc = 3 if c is not None and c.group(1) == b"volVectorField" else 1
    k = raw.index(b"internalField")
    e = raw.index(b";", k) if b"nonuniform" not in raw[k:k + 64] else -1
    if e >= 0:
        v = [float(s) for s in raw[k:e].split(b"uniform", 1)[1]
             .replace(b"(", b" ").replace(b")", b" ").split()]
        return np.tile(np.array(v, float), (ncell, 1)).squeeze()
    a = raw.index(b"(", k)
    n = int(raw[k:a].split()[-1])
    if n != ncell:
        raise ValueError(f"{path}: {n} cells, expected {ncell}")
    if binary:
        v = np.frombuffer(raw, "<f8", n * nc, a + 1).copy()
    else:
        b = raw.index(b"\n)", a) if nc == 3 else raw.index(b")", a)
        v = np.array(raw[a + 1:b].replace(b"(", b" ").replace(b")", b" ").split(),
                     float)
    return v.reshape(n, 3) if nc == 3 else v


def foam_frames(case_dir, fdt, unit_s):
    """{k: time dir} of the dumps at t = k fdt (box), the IC as k = 0."""
    out = {0: os.path.join(case_dir, "0")}
    for t_si, d in observe2d.foam_times(case_dir):
        t = t_si / unit_s
        k = int(round(t / fdt))
        if abs(t - k * fdt) < 1e-6 * max(1.0, t):
            out[k] = d
    return out


def curl_mag(u, v, h):
    """|dv/dx - du/dy| of (ny, nx) cell fields: central differences, y
    periodic, x one-sided at the two ends."""
    dvdx = np.gradient(v, h, axis=1)
    dudy = (np.roll(u, -1, 0) - np.roll(u, 1, 0)) / (2 * h)
    return np.abs(dvdx - dudy)


# --- per-case context (box units) -------------------------------------------------


class Ctx:
    def __init__(self, name):
        self.name = name
        self.cfg = CFG[name]
        spec, o, si, root = foam_flame.context(name)
        self.spec, self.o, self.si, self.root = spec, o, si, root
        N = spec["ladder"][0]
        kind = self.cfg["kind"]
        if kind == "bunsen":
            m = cases.mesh_bunsen2d(spec, N, o.dth)
        elif kind == "wrinkle":
            m = cases.mesh_wrinkle2d(spec, N, o.dth)
        else:
            m = cases.mesh_diffusion2d(spec, N)
        self.m = m
        self.nx, self.ny, self.h = m["nx"], m["ny"], m["h"]
        if kind == "jet":
            self.md = spec["model"]
            self.Tref = si["T0Box"]
            self.sigma = self.md["sigma"]
            self.unit_s = si["tUnit"]
            self.vscale = si["U"]
        else:
            self.Tref = si["TuBox"]
            self.sigma = o.sigma
            self.unit_s = si["tau"]
            self.vscale = si["SL"]
        self.fdt = self.cfg["fdt"]
        self.dns, self.foam = {}, {}
        for key, _, d in self.cfg["runs"]:
            p = os.path.join(VIDEO, d)
            if key.startswith("dns"):
                meta = json.load(open(os.path.join(p, "flame.json")))
                assert (meta["nx"], meta["ny"]) == (self.nx, self.ny), d
                assert abs(meta["frameDt"] - self.fdt) < 1e-12, d
                self.dns[key] = (os.path.join(p, "flame"), meta)
            else:
                self.foam[key] = foam_frames(p, self.fdt, self.unit_s)
        n = [self._dns_count(k) for k in self.dns] + \
            [max(f) + 1 for f in self.foam.values()]
        self.nf = min(n)
        for key, fr in self.foam.items():
            miss = [k for k in range(self.nf) if k not in fr]
            if miss:
                sys.exit(f"{name} {key}: no dump at frames {miss[:5]}")

    def _dns_count(self, key):
        pre, _ = self.dns[key]
        sz = os.path.getsize(pre + "_temp.bin")
        return min(sz // (4 * self.nx * self.ny), self.dns[key][1]["frames"])

    def dns_plane(self, key, field, k):
        pre, _ = self.dns[key]
        n = self.nx * self.ny
        mm = np.memmap(pre + f"_{field}.bin", dtype="<f4", mode="r",
                       offset=4 * n * k, shape=(self.ny, self.nx))
        return np.asarray(mm, dtype=np.float64)

    def frame(self, key, k):
        """Box-unit fields (ny, nx) of solver `key` at frame k: theta, and
        per kind vort, speed (premixed), Y (wrinkle), T, omega (bunsen, jet),
        YF, YO, Z (jet)."""
        sh = (self.ny, self.nx)
        dT = (self.sigma - 1) * self.Tref
        f = {}
        kind = self.cfg["kind"]
        if key in self.dns:
            T = self.dns_plane(key, "temp", k) + self.Tref
            f["T"] = T
            if kind == "jet":
                f["YF"] = self.dns_plane(key, "fuel", k)
                f["YO"] = self.dns_plane(key, "oxid", k)
                f["omega"] = self.dns_plane(key, "omega", k)
            else:
                # the frame's central stencil wraps across x (inflow <->
                # outflow): its two end rows take their neighbours' values
                vo = self.dns_plane(key, "vort", k)
                vo[:, 0], vo[:, -1] = vo[:, 1], vo[:, -2]
                f["vort"] = vo
                f["speed"] = self.dns_plane(key, "speed", k)
                f["omega"] = self.dns_plane(key, "omega", k)
                f["Y"] = self.dns_plane(key, "fuel", k)
        else:
            d = self.foam[key][k]
            nc = self.nx * self.ny
            si = self.si

            def rd(nm):
                p = os.path.join(d, nm)
                return read_internal(p, nc) if os.path.exists(p) else None
            T = rd("T").reshape(sh) / si["TUnit"]
            f["T"] = T
            Q = rd("Qdot")
            f["omega"] = (np.zeros(sh) if Q is None else
                          Q.reshape(sh) / si["q"] / si["omegaUnit"])
            if kind == "jet":
                f["YF"] = rd("F").reshape(sh)
                f["YO"] = rd("O").reshape(sh)
            else:
                U = rd("U")
                u = U[:, 0].reshape(sh) / self.vscale
                v = U[:, 1].reshape(sh) / self.vscale
                f["speed"] = np.hypot(u, v)
                f["vort"] = curl_mag(u, v, self.h)
                Y = rd("F")
                if Y is not None:
                    f["Y"] = Y.reshape(sh)
        f["theta"] = (f["T"] - self.Tref) / dT
        if kind == "jet":
            f["Z"] = cases.mixture_fraction(self.md, f["YF"], f["YO"])
        return f


# --- analysis (pass 1: per frame, in the pool) --------------------------------------

PCT = {"vort": 99.5, "speed": 99.9, "omega": 99.9}


def analyse(k):
    c = G["ctx"]
    kind = c.cfg["kind"]
    out = {"k": k}
    for key, _, _ in c.cfg["runs"]:
        f = c.frame(key, k)
        r = {q: float(np.percentile(f[q], p)) for q, p in PCT.items() if q in f}
        if kind == "bunsen":
            ob, _ = observe2d.bunsen_obs(f["T"], f["omega"], c.h, c.Tref, c.sigma,
                                         c.m["yc"], c.m["geom"]["uJet"], c.o.dth)
            r["xTip"] = ob["xTip"]
        elif kind == "wrinkle":
            xI, _ = observe2d.front_positions(f["T"], f["Y"], c.h,
                                              0.5 * (1 + c.sigma) * c.Tref)
            r["aI"] = observe2d.mode1(xI)[0]
            r["xi"] = (xI - xI.mean()).tolist()
        else:
            r["xTip"] = observejet.measure_jet(c.md, f["YF"], f["YO"], f["omega"],
                                               c.h, c.m["yc"])["xTip"]
        out[key] = r
    return out


def nice_up(x):
    """The smallest 'round' number >= x (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8)."""
    e = 10.0 ** math.floor(math.log10(x))
    for s in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if s * e >= x * (1 - 1e-9):
            return s * e
    return 10 * e


# --- drawing --------------------------------------------------------------------------

_TXT = {}


def text_mask(txt, size, bold=False):
    """Alpha mask (h, w) of `txt` in DejaVu Sans at `size` px (ffmpeg
    drawtext); the row block is the full line height, columns cropped."""
    key = (txt, size, bold)
    if key in _TXT:
        return _TXT[key]
    w = int(size * 0.8 * len(txt)) + 2 * size
    h = int(size * 1.45)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(txt)
    vf = (f"drawtext=fontfile={FONT_B if bold else FONT}:textfile={fh.name}:"
          f"expansion=none:fontsize={size}:fontcolor=white:x={size // 4}:"
          f"y={size // 5}")
    out = subprocess.run([FFMPEG, "-loglevel", "error", "-f", "lavfi", "-i",
                          f"color=c=black:s={w}x{h}", "-vf", vf, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                         capture_output=True, check=True).stdout
    os.unlink(fh.name)
    m = np.frombuffer(out, np.uint8).reshape(h, w).astype(np.float32) / 255
    cols = np.nonzero(m.max(0) > 0.02)[0]
    m = m[:, cols[0]:cols[-1] + 1] if len(cols) else m[:, :1]
    _TXT[key] = m
    return m


def text_width(txt, size, bold=False):
    return text_mask(txt, size, bold).shape[1]


def put_text(img, txt, x, y, size, color=FG, anchor="l", bold=False):
    """anchor l / c / r: x is the left edge / centre / right edge; y = top."""
    m = text_mask(txt, size, bold)
    h, w = m.shape
    x = int(round(x - (w / 2 if anchor == "c" else w if anchor == "r" else 0)))
    y = int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    a = m[y0 - y:y1 - y, x0 - x:x1 - x, None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + color * a


def fill(img, x0, y0, x1, y1, color, alpha=1.0):
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), img.shape[1]), min(int(y1), img.shape[0])
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - alpha) + color * alpha


def polyline(img, xs, ys, color, width=2.0, dash=0.0):
    """Anti-aliased polyline through pixel coordinates (xs, ys): discs of
    radius width/2 at 0.35 px steps, max-composited coverage. dash > 0: on/off
    runs of `dash` px."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    pts = []
    run = 0.0
    for i in range(len(xs) - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        L = math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
        n = max(int(L / 0.35), 1)
        s = np.arange(n) / n
        px = xs[i] + s * (xs[i + 1] - xs[i])
        py = ys[i] + s * (ys[i + 1] - ys[i])
        if dash > 0:
            arc = run + s * L
            keep = (arc // dash) % 2 == 0
            px, py = px[keep], py[keep]
        run += L
        pts.append(np.stack([px, py], 1))
    if not pts:
        return
    p = np.concatenate(pts)
    r = 0.5 * width
    R = int(math.ceil(r + 1))
    cov = np.zeros(img.shape[:2], np.float32)
    bx, by = np.floor(p[:, 0]).astype(int), np.floor(p[:, 1]).astype(int)
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            X, Y = bx + dx, by + dy
            d = np.hypot(X + 0.5 - p[:, 0], Y + 0.5 - p[:, 1])
            a = np.clip(r + 0.5 - d, 0, 1).astype(np.float32)
            k = (X >= 0) & (X < img.shape[1]) & (Y >= 0) & (Y < img.shape[0]) & (a > 0)
            np.maximum.at(cov, (Y[k], X[k]), a[k])
    img[:] = img * (1 - cov[..., None]) + color * cov[..., None]


def dot(img, x, y, color, r=5.0):
    polyline(img, [x, x + 1e-3], [y, y], color, 2 * r)


class Axes:
    """A plot box at pixel (x0, y0, w, h) with data limits; y optionally log."""

    def __init__(self, x0, y0, w, h, xlim, ylim, ylog=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim, self.ylog = xlim, ylim, ylog

    def px(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        fx = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        if self.ylog:
            with np.errstate(divide="ignore", invalid="ignore"):
                fy = (np.log(y) - math.log(self.ylim[0])) / (
                    math.log(self.ylim[1]) - math.log(self.ylim[0]))
        else:
            fy = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return self.x0 + fx * self.w, self.y0 + (1 - fy) * self.h

    def frame(self, img, xticks, yticks, xlabel, ylabel, fmt=lambda v: f"{v:g}"):
        fill(img, self.x0, self.y0, self.x0 + self.w, self.y0 + self.h,
             np.array([0.085, 0.09, 0.10], np.float32))
        grid = np.array([0.17, 0.18, 0.20], np.float32)
        for v in xticks:
            X, _ = self.px(v, self.ylim[0])
            fill(img, X, self.y0, X + 1, self.y0 + self.h, grid)
            put_text(img, fmt(v), X, self.y0 + self.h + 4, 14, DIM, "c")
        for v in yticks:
            _, Y = self.px(self.xlim[0], v)
            fill(img, self.x0, Y, self.x0 + self.w, Y + 1, grid)
            put_text(img, fmt(v), self.x0 - 6, Y - 10, 14, DIM, "r")
        line = np.array([0.45, 0.46, 0.5], np.float32)
        fill(img, self.x0, self.y0 + self.h, self.x0 + self.w, self.y0 + self.h + 1, line)
        fill(img, self.x0, self.y0, self.x0 + 1, self.y0 + self.h, line)
        put_text(img, xlabel, self.x0 + self.w / 2, self.y0 + self.h + 24, 15, DIM, "c")
        put_text(img, ylabel, self.x0, self.y0 - 26, 15, DIM, "l")

    def shade(self, img, xa, xb, color, alpha):
        Xa, _ = self.px(xa, self.ylim[0])
        Xb, _ = self.px(xb, self.ylim[0])
        fill(img, Xa, self.y0, Xb, self.y0 + self.h, color, alpha)

    def line(self, img, x, y, color, width=2.0, dash=0.0):
        X, Y = self.px(x, y)
        Y = np.where((Y >= self.y0 - 0.5) & (Y <= self.y0 + self.h + 0.5), Y, np.nan)
        polyline(img, X, Y, color, width, dash)


def colorbar(img, x0, y0, w, cmap, ticks, label, h=12):
    """Horizontal bar: cmap(q) for q in [0, 1]; ticks [(q, text)]."""
    q = (np.arange(w) + 0.5) / w
    img[y0:y0 + h, x0:x0 + w] = cmap(q)[None, :, :]
    for i, (qq, t) in enumerate(ticks):
        X = x0 + qq * (w - 1)
        fill(img, X, y0 + h, X + 1, y0 + h + 4, DIM)
        put_text(img, t, X, y0 + h + 4, 14, DIM,
                 "l" if i == 0 else "r" if i == len(ticks) - 1 else "c")
    put_text(img, label, x0 + w / 2, y0 - 22, 14, DIM, "c")


# --- the frame (pass 2) -----------------------------------------------------------


def cmap_of(field):
    if field == "theta":
        return lambda v: fire(v).astype(np.float32)
    if field in SQRT:
        return lambda v: lut(np.sqrt(np.clip(v, 0, 1)), INFERNO)
    return lambda v: lut(v, VIRIDIS)


def panel_rgb(field, a, scale, ph, pw):
    """(ny, nx) field -> (ph, pw, 3), x up; Z gets its Z_st isoline."""
    up = np.ascontiguousarray(a.T[::-1])
    v = resample(up.astype(np.float32), ph, pw) / scale
    rgb = cmap_of(field)(v)
    if field == "Z":
        s = v >= 0.5
        e = np.zeros_like(s)
        e[1:] |= s[1:] != s[:-1]
        e[:, 1:] |= s[:, 1:] != s[:, :-1]
        rgb[e] = rgb[e] * 0.15 + 0.85
    return rgb


def compose(k):
    c = G["ctx"]
    L = G["lay"]
    img = np.empty((H, W, 3), np.float32)
    img[:] = BG
    put_text(img, c.cfg["title"], 24, 10, 25, FG, bold=True)
    put_text(img, c.cfg["subtitle"], 24, 48, 17, DIM)
    t = k * c.fdt
    put_text(img, f"t = {t:.1f} {c.cfg['unit']}", W - 28, 12, 30, FG, "r", bold=True)
    fr = {key: c.frame(key, k) for key, _, _ in c.cfg["runs"]}
    for p in L["panels"]:
        rgb = panel_rgb(p["field"], fr[p["run"]][p["field"]], G["scale"][p["field"]],
                        L["ph"], L["pw"])
        img[p["y"]:p["y"] + L["ph"], p["x"]:p["x"] + L["pw"]] = rgb
        if p.get("label"):
            put_text(img, p["label"], p["x"] + L["pw"] / 2, p["y"] - 22, 15,
                     COL[p["run"]], "c", bold=True)
    for lab in L["labels"]:
        put_text(img, lab["text"], lab["x"], lab["y"], lab["size"],
                 COL.get(lab.get("color"), FG) if lab.get("color") else FG,
                 lab.get("anchor", "l"), lab.get("bold", False))
    for cb in L["cbars"]:
        f = cb["field"]
        colorbar(img, cb["x"], cb["y"], cb["w"], cmap_of(f), G["ticks"][f],
                 FIELD_TITLE_JET[f] if c.cfg["kind"] == "jet" else FIELD_TITLE[f])
    PLOT[c.cfg["kind"]](img, k)
    for i, ln in enumerate(G["foot"]):
        put_text(img, ln, 24, H - 12 - FOOT_STEP * (len(G["foot"]) - i), FOOT_SIZE, DIM)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def plot_bunsen(img, k):
    c, S, P = G["ctx"], G["series"], G["lay"]["plot"]
    t = np.arange(c.nf) * c.fdt
    lo, hi = G["ylim"]
    w0, w1 = c.cfg.get("window", (20.0, 25.0))
    tmax = c.cfg.get("tmax", 25.0)
    ystep = 5 if hi - lo <= 60 else 10
    ax = Axes(P[0], P[1], P[2], P[3], (0, max(c.nf * c.fdt, tmax)), (lo, hi))
    ax.frame(img, range(0, int(tmax) + 1, 5), range(int(lo), int(hi) + 1, ystep),
             "t  [τ]", "cone tip height  x_tip  [δ_D]")
    ax.shade(img, w0, w1, FG, 0.06)
    put_text(img, f"window [{w0:g}, {w1:g}]", *ax.px(0.5 * (w0 + w1), hi)[:1],
             P[1] + 4, 13, DIM, "c")
    for key in ("foam", "dns"):
        ax.line(img, t[:k + 1], S[key]["xTip"][:k + 1], COL[key], LW[key])
        X, Y = ax.px(t[k], S[key]["xTip"][k])
        dot(img, X, Y, COL[key], 5)
    y = P[1] + P[3] + 56
    for key, lab, ref in (("dns", "dns --nscbc-tinflow", G["ref"].get("dns")),
                          ("foam", "OpenFOAM mclin", G["ref"].get("foam"))):
        put_text(img, f"{lab}:  x_tip {S[key]['xTip'][k]:6.2f} δ_D", P[0], y, 16,
                 COL[key], bold=True)
        if ref:
            y += 22
            put_text(img, f"docs/chem.md mean [{w0:g}, {w1:g}]: {ref:.2f}", P[0] + 16,
                     y, 15, COL[key])
        y += 28


def plot_wrinkle(img, k):
    c, S, L = G["ctx"], G["series"], G["lay"]
    t = np.arange(c.nf) * c.fdt
    # top: the front displacement xi(y), y as in the panels
    P = L["plot"]
    amax = G["xi_max"]
    y = (np.arange(c.ny) + 0.5) * c.h
    ax = Axes(P[0], P[1], P[2], P[3], (0, c.ny * c.h), (-amax, amax))
    tk = G["xi_ticks"]
    ax.frame(img, range(0, int(c.ny * c.h) + 1, 10), tk, "y  [δ_D]",
             "front displacement  x_I(y) − mean  [δ_D]")
    for key in ("foam", "dns"):
        ax.line(img, y, S[key]["xi"][k], COL[key], LW[key])
    # bottom: ln a(t) with the fits and the theory slopes
    Q = L["plot2"]
    ylo, yhi = G["a_lim"]
    bx = Axes(Q[0], Q[1], Q[2], Q[3], (0, c.nf * c.fdt), (ylo, yhi), ylog=True)
    bx.frame(img, range(0, 21, 4), G["a_ticks"], "t  [τ]",
             "wrinkle amplitude  a  [δ_D]  (log: exponential = straight line)",
             fmt=lambda v: f"{v:g}")
    bx.shade(img, 8, 20, FG, 0.06)
    th = G["theory"]
    for s_, dash, lab in ((th["inviscid"], 4.0, "inviscid Landau"),
                          (th["mm"], 9.0, "Matalon–Matkowsky")):
        tt = np.linspace(0.0, c.nf * c.fdt, 400)   # dense: off-scale points drop
        bx.line(img, tt, G["a_anchor"] * np.exp(s_ * (tt - 8.0)), COL["theory"], 1.6,
                dash)
    for key in ("foam", "dns"):
        bx.line(img, t[:k + 1], S[key]["aI"][:k + 1], COL[key], LW[key])
        X, Y = bx.px(t[k], S[key]["aI"][k])
        dot(img, X, Y, COL[key], 5)
    yy = Q[1] + Q[3] + 50
    rows = [(COL["dns"], f"dns   s = {G['fit']['dns']:.4f} /τ  (fit [8, 20])"),
            (COL["foam"], f"foam  s = {G['fit']['foam']:.4f} /τ"),
            (COL["theory"], f"Matalon–Matkowsky {th['mm']:.4f} /τ (long dash),  "
                            f"inviscid {th['inviscid']:.4f} /τ (short dash)")]
    for col, txt in rows:
        put_text(img, txt, Q[0], yy, 15, col, bold=col is not COL["theory"])
        yy += 22


def plot_jet(img, k):
    c, S, P = G["ctx"], G["series"], G["lay"]["plot"]
    t = np.arange(c.nf) * c.fdt
    lo, hi = G["ylim"]
    tmax = c.cfg.get("tmax", 75.0)
    w0, w1 = c.cfg.get("window", (60.0, 75.0))
    note = ("(no tip before t ≈ 26: the IC's extruded sheet; off scale above 65 "
            "until it settles)" if "tmax" not in c.cfg else
            f"(no closed tip until the IC's sheet flushes; off scale above {hi:g} "
            "until it settles)")
    ax = Axes(P[0], P[1], P[2], P[3], (0, tmax + 1), (lo, hi))
    ax.frame(img, range(0, int(tmax) + 1, c.cfg.get("tstep", 5)),
             range(int(lo), int(hi) + 1, c.cfg.get("ystep", 5)), "t  [τ = l/U]",
             f"flame tip (Z_st on the axis)  x_tip  [l]   {note}")
    ax.shade(img, w0, w1, FG, 0.06)
    for key in ("foam", "dns100", "dns"):
        ax.line(img, t[:k + 1], S[key]["xTip"][:k + 1], COL[key], LW[key])
        if np.isfinite(S[key]["xTip"][k]):
            X, Y = ax.px(t[k], S[key]["xTip"][k])
            dot(img, X, Y, COL[key], 5)
    x = P[0] + P[2] + 30
    y = P[1] + 10
    w0, w1 = c.cfg.get("window", (60.0, 75.0))
    put_text(img, f"x_tip now  (docs/chem.md mean [{w0:g}, {w1:g}])", x, y, 15, DIM)
    y += 28
    for key, lab, _ in c.cfg["runs"]:
        v = S[key]["xTip"][k]
        ref = G["ref"].get(key)
        put_text(img, f"{lab.strip()}", x, y, 16, COL[key], bold=True)
        now = f"{v:6.2f} l" if np.isfinite(v) else "no closed tip yet"
        put_text(img, now + (f"   ({ref:.2f})" if ref else ""), x, y + 22,
                 16, COL[key])
        y += 60


PLOT = {"bunsen": plot_bunsen, "wrinkle": plot_wrinkle, "jet": plot_jet}

G = {}


def init(cfg):
    G.update(cfg)


# --- layout -------------------------------------------------------------------------


def layout(c, foot_top):
    """Panel / bar / plot placement above the footer's first line foot_top."""
    kind = c.cfg["kind"]
    runs = c.cfg["runs"]
    fields = c.cfg["fields"]
    if kind == "bunsen" and c.nx > 2.5 * c.ny:
        # a tall cone: one row, a field group of (dns, foam) panels, full height
        nr = len(runs)
        x0, y0, igap, ggap = 30, 140, 6, 30
        ph = foot_top - 84 - y0
        pw = int(round(c.ny * ph / c.nx))
        gw = nr * pw + (nr - 1) * igap
        panels, cbars = [], []
        for g, f in enumerate(fields):
            gx = x0 + g * (gw + ggap)
            for r, (key, _, _) in enumerate(runs):
                panels.append({"run": key, "field": f, "x": gx + r * (pw + igap),
                               "y": y0, "label": {"dns": "dns tinflow",
                                                  "foam": "OpenFOAM"}[key]})
            cbars.append({"field": f, "x": gx, "y": y0 + ph + 34, "w": gw})
        right = x0 + 3 * gw + 2 * ggap
        px0 = right + 90
        return {"ph": ph, "pw": pw, "panels": panels, "labels": [], "cbars": cbars,
                "plot": (px0, y0 + 30, W - 40 - px0, ph - 200), "right": right}
    if kind in ("bunsen", "wrinkle"):
        # 2 rows (dns, foam) x 3 field columns, colour bars under the columns
        y0, gap = 112, 10
        ph = (foot_top - 72 - y0 - gap) // 2
        s = ph / c.nx
        pw = int(round(c.ny * s))
        x0, pitch = 150, pw + 24
        panels, labels, cbars = [], [], []
        for r, (key, lab, _) in enumerate(runs):
            py = y0 + r * (ph + gap)
            labels.append({"text": "dns" if key == "dns" else "OpenFOAM",
                           "x": x0 - 12, "y": py + ph / 2 - 26, "size": 20,
                           "anchor": "r", "bold": True, "color": key})
            sub = "--nscbc-tinflow" if kind == "bunsen" and key == "dns" else (
                "nz 8 twin" if key == "dns" else "mclin")
            labels.append({"text": sub, "x": x0 - 12, "y": py + ph / 2 + 2,
                           "size": 14, "anchor": "r", "color": key})
            for j, f in enumerate(fields):
                panels.append({"run": key, "field": f, "x": x0 + j * pitch, "y": py})
        for j, f in enumerate(fields):
            cbars.append({"field": f, "x": x0 + j * pitch, "y": y0 + 2 * ph + gap + 34,
                          "w": pw})
        right = x0 + 3 * pitch - 24
        px0 = right + 110
        lay = {"ph": ph, "pw": pw, "panels": panels, "labels": labels,
               "cbars": cbars}
        if kind == "bunsen":
            lay["plot"] = (px0, y0 + 30, W - 40 - px0, 2 * ph + gap - 180)
        else:
            hh = (2 * ph + gap - 150) // 2
            lay["plot"] = (px0, y0 + 30, W - 40 - px0, hh - 40)
            lay["plot2"] = (px0, y0 + 30 + hh + 30, W - 40 - px0, hh)
        lay["right"] = right
        return lay
    # jet: three field groups of len(runs) panels, colour bars under the groups
    nr = len(runs)
    margin, ggap, igap = 30, 40, 6
    pw = (W - 2 * margin - 2 * ggap - 3 * (nr - 1) * igap) // (3 * nr)
    s = pw / c.ny
    ph = int(round(c.nx * s))
    y0 = 140
    panels, labels, cbars = [], [], []
    gw = nr * pw + (nr - 1) * igap
    for g, f in enumerate(fields):
        gx = margin + g * (gw + ggap)
        for r, (key, lab, _) in enumerate(runs):
            short = {"dns": "dns η_in 1000", "foam": "OpenFOAM",
                     "dns100": "dns η_in 100"}[key]
            panels.append({"run": key, "field": f, "x": gx + r * (pw + igap),
                           "y": y0, "label": short})
        cbars.append({"field": f, "x": gx, "y": y0 + ph + 34, "w": gw})
    py = y0 + ph + 104
    lay = {"ph": ph, "pw": pw, "panels": panels, "labels": labels, "cbars": cbars,
           "plot": (130, py, 1100, foot_top - 62 - py)}
    return lay


def ticks_for(field, scale):
    if field in ("theta", "Z"):
        return [(q, f"{q:g}") for q in (0, 0.25, 0.5, 0.75, 1.0)]
    vals = [scale * q for q in (0, 0.25, 0.5, 0.75, 1.0)]
    if field in SQRT:     # sqrt map: the bar position of v is sqrt(v / max)
        vals = [scale * q for q in (0, 0.1, 0.4, 1.0)]
        return [(math.sqrt(v / scale), f"{v:.3g}") for v in vals]
    return [(v / scale, f"{v:.3g}") for v in vals]


def wrap(lines, size, width):
    out = []
    for ln in lines:
        words = ln.split(" ")
        cur = ""
        for w_ in words:
            trial = (cur + " " + w_).strip()
            if cur and text_width(trial, size) > width:
                out.append(cur)
                cur = w_
            else:
                cur = trial
        if cur:
            out.append(cur)
    return out


def dns_label(c):
    """The dns run's DNS/LES label, verbatim from its --dry-run."""
    key, _, d = c.cfg["runs"][0]
    for ln in open(os.path.join(VIDEO, d, "dryrun.txt")):
        if ln.strip().startswith("label"):
            return ln.strip()[len("label"):].strip()
    return "?"


def committed_refs(c):
    p = os.path.join(c.root, "compare.json")
    if not os.path.exists(p):
        return {}
    r = json.load(open(p))
    kind = c.cfg["kind"]
    if kind == "bunsen":
        # dns: its stats rows, foam: its dumps (the docs/chem.md tables)
        return {"dns": r["dns"]["dns_tinflow"]["stats"]["xTip"]["mean"],
                "foam": r["foam"]["xTip"]["mean"]}
    if kind == "jet":
        return {"dns": r["dns_extra"]["dns etaIn 1000"]["stats"]["xTip"]["mean"],
                "dns100": r["dns"]["stats"]["xTip"]["mean"],
                "foam": r["foam"]["foam_mclin_c4"]["means"]["xTip"]["mean"]}
    return {}


def cmd_render(name, workers, fps, out_dir):
    t0 = time.time()
    c = Ctx(name)
    fps = fps or c.cfg["fps"]
    frames = list(range(c.nf))
    ctx = mp.get_context("fork")
    init({"ctx": c})
    with ctx.Pool(workers, initializer=init, initargs=({"ctx": c},)) as pool:
        an = pool.map(analyse, frames, chunksize=4)
    t1 = time.time()
    keys = [r[0] for r in c.cfg["runs"]]
    series = {key: {q: np.array([a[key][q] for a in an])
                    for q in an[0][key] if q != "xi"} for key in keys}
    if c.cfg["kind"] == "wrinkle":
        for key in keys:
            series[key]["xi"] = [np.array(a[key]["xi"]) for a in an]
    # fixed scales: the last third of the clip, the larger of the solvers
    tail = slice(2 * c.nf // 3, c.nf)
    scale = {"theta": 1.0, "Z": 1.0}
    for q in ("vort", "speed", "omega"):
        if q in series[keys[0]]:
            scale[q] = nice_up(max(float(np.median(series[key][q][tail]))
                                   for key in keys))
    foot = c.cfg["foot"] + [f"dns label: {dns_label(c)}"]
    foot = wrap(foot, FOOT_SIZE, W - 48)
    cfg = {"ctx": c, "series": series, "scale": scale, "foot": foot,
           "ticks": {f: ticks_for(f, scale.get(f, 1.0)) for f in c.cfg["fields"]},
           "lay": layout(c, H - 12 - FOOT_STEP * len(foot)),
           "ref": committed_refs(c)}
    if c.cfg["kind"] == "bunsen":
        ys = np.concatenate([series[key]["xTip"] for key in keys])
        cfg["ylim"] = (5 * math.floor(np.nanmin(ys) / 5), 5 * math.ceil(np.nanmax(ys) / 5))
    if c.cfg["kind"] == "jet":   # the settled flames; the start transient runs off scale
        cfg["ylim"] = c.cfg.get("ylim", (40.0, 65.0))
    if c.cfg["kind"] == "wrinkle":
        lam = c.ny * c.h
        th = observe2d.dl_theory(c.sigma, 2 * math.pi / lam)
        cfg["theory"] = th
        t = np.arange(c.nf) * c.fdt
        half = np.abs(t / 0.5 - np.round(t / 0.5)) < 1e-6   # the 0.5 tau dumps
        cfg["fit"] = {key: observe2d.growth(t[half], series[key]["aI"][half],
                                            (8.0, 20.0))[0] for key in keys}
        cfg["a_anchor"] = float(series["dns"]["aI"][int(round(8.0 / c.fdt))])
        amin = min(float(series[key]["aI"].min()) for key in keys)
        amax = max(float(series[key]["aI"].max()) for key in keys)
        cfg["a_lim"] = (0.8 * amin, 1.6 * amax)
        cfg["a_ticks"] = [v for v in (0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5)
                          if cfg["a_lim"][0] <= v <= cfg["a_lim"][1]]
        xm = max(float(np.abs(series[key]["xi"][-1]).max()) for key in keys)
        cfg["xi_max"] = nice_up(1.15 * xm)
        cfg["xi_ticks"] = [-cfg["xi_max"], -cfg["xi_max"] / 2, 0,
                           cfg["xi_max"] / 2, cfg["xi_max"]]
    print(f"{name}: {c.nf} frames, scales " +
          " ".join(f"{q} {v:g}" for q, v in scale.items()) +
          f"; analysis {t1 - t0:.1f} s", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    mp4 = os.path.join(out_dir, f"{name}.mp4")
    png = os.path.join(out_dir, f"{name}.png")
    enc = [FFMPEG, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    p = subprocess.Popen(enc + ["-c:v", "libx264", "-preset", "slow", "-crf", "18",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                mp4], stdin=subprocess.PIPE)
    poster = None
    kp = c.nf - 1
    with ctx.Pool(workers, initializer=init, initargs=(cfg,)) as pool:
        for n, img in enumerate(pool.imap(compose, frames, chunksize=2)):
            p.stdin.write(img.tobytes())
            if n == kp:
                poster = img
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    q = subprocess.run(enc + ["-frames:v", "1", png], input=poster.tobytes())
    if q.returncode != 0:
        sys.exit("ffmpeg png failed")
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump({"frames": c.nf, "fps": fps, "duration_s": c.nf / fps,
                   "scales": scale, "render_s": time.time() - t0,
                   "fit": cfg.get("fit")}, fh, indent=1)
    print(f"{mp4}: {c.nf} frames {W}x{H} @ {fps} fps = {c.nf / fps:.1f} s; "
          f"{png}; {time.time() - t0:.1f} s", flush=True)


# --- the reproduction check ------------------------------------------------------------


def _ascii_tokens(path):
    with open(path, "rb") as f:
        raw = f.read()
    k = raw.index(b"internalField")
    a = raw.index(b"(", k)
    b = raw.index(b"\n)", a)
    return raw[a + 1:b].replace(b"(", b" ").replace(b")", b" ").split()


def check_foam(c, key, old_dir, fields=("T", "U")):
    """Committed (ascii, writePrecision 10) vs rerun dumps at every committed
    time: tokens equal when the rerun is printed at 10 significant digits."""
    old = observe2d.foam_times(old_dir)
    rerun = {r[0]: r[2] for r in c.cfg["runs"]}[key]
    new = {round(t_si / c.unit_s, 6): d for t_si, d in
           observe2d.foam_times(os.path.join(VIDEO, rerun))}
    nc = c.nx * c.ny
    res = {"times": 0, "tokens": 0, "mismatch": 0, "max_rel": 0.0, "per_time": []}
    for t_si, d in old:
        dn = new.get(round(t_si / c.unit_s, 6))
        if dn is None:
            continue
        res["times"] += 1
        worst = 0.0
        for fn in fields:
            tok = _ascii_tokens(os.path.join(d, fn))
            v = read_internal(os.path.join(dn, fn), nc).ravel()
            o = np.array(tok, float)
            mine = [f"{x:.10g}".encode() for x in v]
            res["tokens"] += len(tok)
            res["mismatch"] += sum(1 for x, y in zip(mine, tok) if float(x) != float(y))
            worst = max(worst, float(np.abs(v - o).max() / np.abs(o).max()))
        res["max_rel"] = max(res["max_rel"], worst)
        res["per_time"].append((round(t_si / c.unit_s, 4), worst))
    return res


def check_dns(c, key, old_pre, old_fdt, field="temp"):
    """Byte comparison of the rerun's frames at the committed frame times."""
    meta = json.load(open(old_pre + ".json"))
    n = c.nx * c.ny
    a = np.fromfile(old_pre + f"_{field}.bin", "<f4").reshape(-1, c.ny, c.nx)
    same = diff = 0
    for k in range(min(meta["frames"], a.shape[0])):
        r = k * old_fdt / c.fdt         # the rerun's frame at the same time
        kk = int(round(r))
        if abs(r - kk) > 1e-9 or kk >= c._dns_count(key):
            continue
        b = np.fromfile(c.dns[key][0] + f"_{field}.bin", "<f4", n, offset=4 * n * kk)
        if np.array_equal(a[k].ravel().view(np.uint32), b.view(np.uint32)):
            same += 1
        else:
            diff += 1
    return {"frames_equal": same, "frames_differ": diff, "field": field}


def cmd_check(name):
    c = Ctx(name)
    R = c.root
    out = {"case": name}
    cr = json.load(open(os.path.join(R, "compare.json")))
    if name == "G2a":
        out["dns_vs_dns_c8_tinflow"] = check_dns(c, "dns", os.path.join(
            R, "dns_c8_tinflow", "flame"), 1.0)
        out["dns_vs_dns_c8_nz8_speed"] = "n/a (the committed twin is not --nscbc-tinflow)"
        out["foam"] = check_foam(c, "foam", os.path.join(R, "foam_mclin_c8"))
        rows = {"t": []}
        for k in range(200, c.nf, 10):
            f = c.frame("foam", k)
            ob, _ = observe2d.bunsen_obs(f["T"], f["omega"], c.h, c.Tref, c.sigma,
                                         c.m["yc"], c.m["geom"]["uJet"], c.o.dth)
            rows["t"].append(k * c.fdt)
            for q in ("xTip", "tanA", "sinRatio"):
                rows.setdefault(q, []).append(ob[q])
        # the committed window [20, 25] has 6 dumps; the rerun's t 25 one too
        k25 = int(round(25.0 / c.fdt))
        if k25 in c.foam["foam"]:
            f = c.frame("foam", k25)
            ob, _ = observe2d.bunsen_obs(f["T"], f["omega"], c.h, c.Tref, c.sigma,
                                         c.m["yc"], c.m["geom"]["uJet"], c.o.dth)
            rows["t"].append(25.0)
            for q in ("xTip", "tanA", "sinRatio"):
                rows[q].append(ob[q])
        out["foam_obs"] = {q: {"rerun": float(np.mean(rows[q])),
                               "committed": cr["foam"][q]["mean"]}
                           for q in ("xTip", "tanA", "sinRatio")}
    elif name == "G2b":
        out["dns_vs_dns_c8"] = check_dns(c, "dns", os.path.join(R, "dns_c8", "flame"), 0.5)
        out["foam"] = check_foam(c, "foam", os.path.join(R, "foam_mclin_c8"))
        ts, aI = [], []
        for k in range(0, 201, 5):
            if k not in c.foam["foam"]:
                continue
            f = c.frame("foam", k)
            xI, _ = observe2d.front_positions(f["T"], f["Y"], c.h,
                                              0.5 * (1 + c.sigma) * c.Tref)
            ts.append(k * c.fdt)
            aI.append(observe2d.mode1(xI)[0])
        s = observe2d.growth(ts, aI, (8.0, 20.0))[0]
        out["foam_s_xI_8_20"] = {"rerun": s,
                                 "committed": cr["runs"]["foam"]["rates"]["8-20"]["aI"][0]}
        # dns: the rerun stops at t 19.9 (no t 20 frame): fit [8, 19.5] on both
        old = cr["runs"]["dns"]
        to, ao = np.array(old["t"]), np.array(old["aI"])
        tn = np.arange(0, c.nf, 5) * c.fdt
        an = []
        for k in range(0, c.nf, 5):
            f = c.frame("dns", k)
            xI, _ = observe2d.front_positions(f["T"], f["Y"], c.h,
                                              0.5 * (1 + c.sigma) * c.Tref)
            an.append(observe2d.mode1(xI)[0])
        out["dns_s_xI_8_19.5"] = {"rerun": observe2d.growth(tn, an, (8.0, 19.5))[0],
                                  "committed_frames": observe2d.growth(to, ao, (8.0, 19.5))[0],
                                  "committed_8_20": old["rates"]["8-20"]["aI"][0]}
    elif name in ("G4b", "G4c"):
        # dns: nz 8 mid-span vs the committed 2D tinflow twin (fdt 1: t 0, 9, 18)
        out["dns_vs_dns_c8_tinflow"] = check_dns(c, "dns", os.path.join(
            R, "dns_c8_tinflow", "flame"), 1.0)
        out["foam"] = check_foam(c, "foam", os.path.join(R, "foam_mclin_c8"))
        rows = {}
        for t_si, d in observe2d.foam_times(os.path.join(R, "foam_mclin_c8")):
            t = round(t_si / c.unit_s, 6)
            k = int(round(t / c.fdt))
            if abs(k * c.fdt - t) > 1e-6 or k not in c.foam["foam"]:
                continue
            f = c.frame("foam", k)
            ob, _ = observe2d.bunsen_obs(f["T"], f["omega"], c.h, c.Tref, c.sigma,
                                         c.m["yc"], c.m["geom"]["uJet"], c.o.dth)
            rows.setdefault("t", []).append(t)
            rows.setdefault("xTip", []).append(ob["xTip"])
        out["foam_xTip_at_committed_dumps"] = rows
    else:
        for key, old in (("dns", "dns_c4_etain1000"), ("dns100", "dns_c4")):
            out[f"{key}_frames"] = check_dns(c, key, os.path.join(R, old, "flame"), 1.0)
            a = open(os.path.join(VIDEO, dict((r[0], r[2]) for r in c.cfg["runs"])[key],
                                  "flame_stats.csv"), "rb").read()
            b = open(os.path.join(R, old, "flame_stats.csv"), "rb").read()
            out[f"{key}_stats_csv_identical"] = a == b
        out["foam"] = check_foam(c, "foam", os.path.join(R, "foam_mclin_c4"),
                                 ("T", "F", "O"))
        xs = []
        nc = c.nx * c.ny
        for t_si, d in observe2d.foam_times(os.path.join(VIDEO, "G3a/foam_mclin")):
            t = t_si / c.unit_s     # the committed window: the integer dumps
            if not (60 - 1e-6 <= t <= 75 + 1e-6 and abs(t - round(t)) < 1e-6):
                continue
            sh = (c.ny, c.nx)
            YF = read_internal(os.path.join(d, "F"), nc).reshape(sh)
            YO = read_internal(os.path.join(d, "O"), nc).reshape(sh)
            Q = read_internal(os.path.join(d, "Qdot"), nc).reshape(sh)
            om = Q / c.si["q"] / c.si["omegaUnit"]
            xs.append(observejet.measure_jet(c.md, YF, YO, om, c.h, c.m["yc"])["xTip"])
        out["foam_xTip_60_75"] = {"rerun": float(np.mean(xs)),
                                  "committed": cr["foam"]["foam_mclin_c4"]["means"]["xTip"]["mean"]}
    p = os.path.join(VIDEO, name, "check.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["render", "check"])
    ap.add_argument("case", choices=sorted(CFG))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--out-dir", default=VIDEO)
    a = ap.parse_args()
    if a.cmd == "render":
        cmd_render(a.case, a.workers, a.fps, a.out_dir)
    else:
        cmd_check(a.case)


if __name__ == "__main__":
    main()
