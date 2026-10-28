#!/usr/bin/env python3
"""Render a `dns flame --mode chamber` run's --emit bin frames as a video.

Reads <prefix>.json (nx, ny, frames, mach), <prefix>_temp.bin (T - Tu),
<prefix>_omega.bin and the static <prefix>_mask.bin (1 fluid, 0 solid; one
raw little-endian float32 nx*ny plane per frame, i fastest;
docs/viewer-format.md). Temperature theta = (T - Tu)/((sigma - 1) Tu), Tu =
1/(gamma M^2), goes through a fire colormap; the reaction rate adds a soft
glow; the chamber walls are dark grey with a light outline. Flow runs left
to right (x horizontal, y up). Frames are piped to ffmpeg (libx264, yuv420p,
crf 18); the last frame is also written as a PNG.

numpy only.

    chamber_video.py RUN_PREFIX --out chamber.mp4 [--png last.png]
                     [--fps 30] [--scale 3] [--glow 0.5] [--crop-bands 16]
                     [--sigma 5]
"""

import argparse
import json
import subprocess
import sys

import numpy as np

FIRE = np.array([
    [0.00, 0.00, 0.00, 0.00],
    [0.10, 0.10, 0.00, 0.00],
    [0.25, 0.35, 0.03, 0.00],
    [0.45, 0.70, 0.12, 0.00],
    [0.62, 0.95, 0.35, 0.02],
    [0.78, 1.00, 0.62, 0.08],
    [0.90, 1.00, 0.85, 0.35],
    [1.00, 1.00, 0.97, 0.82],
])
WALL = np.array([0.20, 0.20, 0.22])
EDGE = np.array([0.62, 0.62, 0.66])
COLD = np.array([0.02, 0.03, 0.07])  # the cold ambient: near-black blue
GLOW = {"blue": (0.45, 0.65, 1.0), "white": (1.0, 0.95, 0.85)}


def fire(theta):
    theta = np.clip(theta, 0.0, 1.0)
    rgb = np.empty(theta.shape + (3,))
    for c in range(3):
        rgb[..., c] = np.interp(theta, FIRE[:, 0], FIRE[:, c + 1])
    # Blend the black end toward a dark blue so the cold gas reads as gas.
    w = np.clip(1 - theta / 0.1, 0, 1)[..., None]
    return rgb * (1 - w) + COLD * w


def blur(a, passes):
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = 0.25 * (p[:-2, 1:-1] + 2 * p[1:-1, 1:-1] + p[2:, 1:-1])
        p = np.pad(a, 1, mode="edge")
        a = 0.25 * (p[1:-1, :-2] + 2 * p[1:-1, 1:-1] + p[1:-1, 2:])
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--png")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--glow", type=float, default=0.5)
    ap.add_argument("--crop-bands", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--burnt", type=float, default=0.8,
                    help="colormap position of the burnt plateau (< 1 keeps "
                         "white for the flame glow)")
    ap.add_argument("--glow-color", choices=sorted(GLOW), default="blue")
    ap.add_argument("--first", type=int, default=0)
    a = ap.parse_args()

    meta = json.load(open(a.prefix + ".json"))
    nx, ny, nf = meta["nx"], meta["ny"], meta["frames"]
    n = nx * ny
    Tu = 1.0 / (1.4 * meta["mach"] ** 2)
    tscale = (a.sigma - 1) * Tu
    temp = np.memmap(a.prefix + "_temp.bin", dtype="<f4", mode="r")
    omega = np.memmap(a.prefix + "_omega.bin", dtype="<f4", mode="r")
    if temp.size < nf * n or omega.size < nf * n:
        sys.exit(f"{a.prefix}: frame files shorter than json frames {nf}")
    mask = np.fromfile(a.prefix + "_mask.bin", dtype="<f4")[:n]
    fluid = mask.reshape(ny, nx)[::-1, :] > 0.5
    solid = ~fluid
    # Outline: solid cells with a fluid 4-neighbour (periodic in y).
    nb = np.zeros_like(solid)
    for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb |= np.roll(np.roll(fluid, dj, 0), di, 1)
    edge = solid & nb

    def plane(arr, f):
        p = np.asarray(arr[f * n:(f + 1) * n]).reshape(ny, nx)[::-1, :]
        return np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    # Glow scale: the median rate INSIDE the last frame's reaction zone (cells
    # above 5% of its peak), so the whole front saturates, not just its crest.
    last = plane(omega, nf - 1) * fluid
    zone = last[last > 0.05 * last.max()] if last.max() > 0 else last
    wscale = float(np.median(zone)) if a.glow > 0 and zone.size else 1.0
    if not wscale > 0:
        wscale = 1.0
    c0, c1 = a.crop_bands, nx - a.crop_bands
    H, W = ny * a.scale, (c1 - c0) * a.scale
    H -= H % 2
    W -= W % 2

    def render(f):
        rgb = fire(a.burnt * plane(temp, f) / tscale)
        if a.glow > 0:
            w = np.clip(plane(omega, f) / wscale, 0, 1) * fluid
            g = np.clip(blur(w, 2) * 0.7 + blur(w, 8) * 0.9, 0, 1)
            rgb = 1 - (1 - rgb) * (1 - a.glow * g[..., None] *
                                   np.array(GLOW[a.glow_color]))
        rgb[solid] = WALL
        rgb[edge] = EDGE
        rgb = rgb[:, c0:c1]
        rgb = np.repeat(np.repeat(rgb, a.scale, 0), a.scale, 1)[:H, :W]
        return (np.clip(rgb, 0, 1) ** (1 / 1.1) * 255 + 0.5).astype(np.uint8)

    enc = ["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-"]
    p = subprocess.Popen(enc + ["-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-crf", "18", a.out], stdin=subprocess.PIPE)
    for f in range(a.first, nf):
        p.stdin.write(render(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    if a.png:
        q = subprocess.run(enc + ["-frames:v", "1", a.png],
                           input=render(nf - 1).tobytes())
        if q.returncode != 0:
            sys.exit("ffmpeg png failed")
    print(f"{a.out}: {nf - a.first} frames {W}x{H} @ {a.fps} fps "
          f"(Tb - Tu = {tscale:.6g}, omega scale {wscale:.4g})")


if __name__ == "__main__":
    main()
