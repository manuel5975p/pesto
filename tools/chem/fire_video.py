#!/usr/bin/env python3
"""Render a `dns flame` run's --emit bin frames as a fire-coloured video.

Reads <prefix>.json (nx, ny, frames, frameDt) and <prefix>_<field>.bin (one
raw little-endian float32 nx*ny plane per frame, i fastest; docs/chem.md,
docs/viewer-format.md). The temperature channel `temp` (T - Tu) is mapped
through a fire colormap (black -> dark red -> orange -> yellow -> near white)
on a dark background; an optional glow is composited from `omega` (the
reaction rate, blurred). The jet axis x is drawn UPWARD: fresh gas enters at
the bottom. Frames are piped to ffmpeg (libx264, yuv420p, crf 18); the last
frame is also written as a PNG.

numpy only (no matplotlib).

    fire_video.py RUN_PREFIX --out flame.mp4 [--png last.png] [--fps 30]
                  [--scale 2] [--glow 0.4] [--glow-color blue|white]
                  [--gamma 1] [--height N] [--width N] [--pad-bottom N]

--height / --width keep N cells above the cropped bands / N centred cells
across; --pad-bottom adds N black cells below the inflow plane.

--gamma G maps theta -> theta^G before the colormap: G > 1 darkens the
warm products so the hot sheet carries the light (a diffusion flame's
plume sits near T_ad and saturates at G = 1).
"""

import argparse
import json
import subprocess
import sys

import numpy as np

# Fire colormap control points (position, r, g, b) on theta = (T-Tu)/(Tb-Tu).
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
GLOW = {"blue": (0.35, 0.55, 1.0), "white": (1.0, 0.95, 0.85)}


def fire(theta):
    theta = np.clip(theta, 0.0, 1.0)
    rgb = np.empty(theta.shape + (3,))
    for c in range(3):
        rgb[..., c] = np.interp(theta, FIRE[:, 0], FIRE[:, c + 1])
    return rgb


def blur(a, passes):
    """Separable [1 2 1]/4 smoothing, `passes` times (edge-clamped)."""
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = 0.25 * (p[:-2, 1:-1] + 2 * p[1:-1, 1:-1] + p[2:, 1:-1])
        p = np.pad(a, 1, mode="edge")
        a = 0.25 * (p[1:-1, :-2] + 2 * p[1:-1, 1:-1] + p[1:-1, 2:])
    return a


def upright(plane, nx, ny):
    """(ny, nx) row-major with i fastest -> image rows = x, top = high x."""
    return plane.reshape(ny, nx).T[::-1, :]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prefix", help="the run's --out prefix")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--png", help="final frame .png")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=int, default=2, help="pixel repeat")
    ap.add_argument("--glow", type=float, default=0.4,
                    help="omega glow strength, 0 = off")
    ap.add_argument("--glow-color", choices=sorted(GLOW), default="white")
    ap.add_argument("--crop-bands", type=int, default=0,
                    help="cells dropped at both x ends (the sink bands)")
    ap.add_argument("--first", type=int, default=0, help="first frame")
    ap.add_argument("--gamma", type=float, default=1.0,
                    help="theta -> theta^gamma before the colormap")
    ap.add_argument("--height", type=int, default=0,
                    help="keep only this many cells above the crop (0 = all)")
    ap.add_argument("--width", type=int, default=0,
                    help="keep only this many cells across, centred (0 = all)")
    ap.add_argument("--pad-bottom", type=int, default=0,
                    help="black cells added below the inflow plane")
    ap.add_argument("--smooth", type=int, default=0,
                    help="upscale N x by ffmpeg bicubic instead of --scale's "
                         "pixel repeat (0 = off)")
    a = ap.parse_args()
    if a.smooth > 0:
        a.scale = 1

    meta = json.load(open(a.prefix + ".json"))
    nx, ny, nf = meta["nx"], meta["ny"], meta["frames"]
    n = nx * ny
    temp = np.memmap(a.prefix + "_temp.bin", dtype="<f4", mode="r")
    if temp.size < nf * n:
        sys.exit(f"{a.prefix}_temp.bin holds {temp.size // n} frames, json {nf}")
    omega = None
    if a.glow > 0:
        omega = np.memmap(a.prefix + "_omega.bin", dtype="<f4", mode="r")
    # Normalisation: the burnt rise Tb - Tu, read off the last frame (its
    # 99.9th percentile is the burnt plateau), and omega's peak likewise.
    last = np.asarray(temp[(nf - 1) * n:nf * n])
    tscale = float(np.percentile(last, 99.9))
    wscale = (float(np.percentile(np.asarray(omega[(nf - 1) * n:nf * n]), 99.95))
              if omega is not None else 1.0)
    if tscale <= 0:
        sys.exit("temperature rise is not positive in the last frame")

    c0, c1 = a.crop_bands, nx - a.crop_bands
    if a.height > 0:
        c1 = min(c1, c0 + a.height)
    j0, j1 = 0, ny
    if 0 < a.width < ny:
        j0 = (ny - a.width) // 2
        j1 = j0 + a.width
    H, W = (c1 - c0 + a.pad_bottom) * a.scale, (j1 - j0) * a.scale
    H -= H % 2
    W -= W % 2
    gcol = np.array(GLOW[a.glow_color])

    def render(f):
        th = upright(np.asarray(temp[f * n:(f + 1) * n]), nx, ny)
        rgb = fire(np.clip(th[nx - c1:nx - c0, j0:j1] / tscale, 0, 1) ** a.gamma)
        if omega is not None:
            w = upright(np.asarray(omega[f * n:(f + 1) * n]), nx, ny)
            w = np.clip(w[nx - c1:nx - c0, j0:j1] / wscale, 0, 1)
            g = blur(w, 3) * 0.6 + blur(w, 12) * 0.8
            rgb = 1 - (1 - rgb) * (1 - a.glow * np.clip(g, 0, 1)[..., None] * gcol)
        if a.pad_bottom > 0:
            rgb = np.concatenate(
                [rgb, np.zeros((a.pad_bottom,) + rgb.shape[1:])], 0)
        rgb = np.repeat(np.repeat(rgb, a.scale, 0), a.scale, 1)[:H, :W]
        return (np.clip(rgb, 0, 1) ** (1 / 1.1) * 255 + 0.5).astype(np.uint8)

    enc = ["/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-"]
    if a.smooth > 0:
        enc += ["-vf", f"scale={W * a.smooth}:{H * a.smooth}:flags=bicubic"]
        W, H = W * a.smooth, H * a.smooth
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
