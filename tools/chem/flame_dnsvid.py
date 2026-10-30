#!/usr/bin/env python3
"""Video + still of a 3D DNS flame box: z-integrated chemiluminescence (omega),
mid-span |curl u| and |u|, T, 2x2. Dark theme. usage: flame_dnsvid.py PREFIX TITLE OUTSTEM [T0]
(PREFIX = a `dns flame --emit volume` run: PREFIX.json + PREFIX_{omega,vort,speed,temp}.bin)"""
import json, pathlib, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from turb_render import chemi_rgb, aces, srgb

p, title, stem = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(p + ".json"))
nx, ny, nz, nf = m["nx"], m["ny"], m["nz"], m["frames"]
n3 = nx * ny * nz
t0 = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0


def vol(field, f):
    return np.fromfile(p + "_" + field + ".bin", "<f4", n3, offset=4 * n3 * f).reshape(nz, ny, nx)


def frame(f):
    vo = vol("vort", f)[nz // 2].copy()
    vo[:, :2], vo[:, -2:] = vo[:, 2:3], vo[:, -3:-2]  # end columns: stencil wraps across x
    return (np.maximum(vol("omega", f), 0).sum(0), vo, vol("speed", f)[nz // 2], vol("temp", f)[nz // 2])


# fixed scales from a sample of late frames
samp = [frame(f) for f in range(nf // 2, nf, max(1, nf // 8))]
lmax = np.percentile(np.concatenate([s[0].ravel() for s in samp]), 99.7)
vmax = np.percentile(np.concatenate([s[1].ravel() for s in samp]), 99.7)
smax = np.percentile(np.concatenate([s[2].ravel() for s in samp]), 99.9)
tmax = max(s[3].max() for s in samp)
rgb = chemi_rgb()
plt.style.use("dark_background")
W, H = 19.2, 10.8
fig, ax = plt.subplots(2, 2, figsize=(W, H), facecolor="#0b0d10")
ax = ax.ravel()
heads = ["light ∫ω dz (CH*/C2*, rendered)", "|∇×u| mid-span (√-scaled, fixed)", "|u| mid-span (fixed)",
         "T − Tu mid-span"]
ims, txt = [], fig.text(0.985, 0.955, "", ha="right", fontsize=13, color="#ddd")
for c, a in enumerate(ax):
    im = a.imshow(np.zeros((ny, nx, 3)) if c == 0 else np.zeros((ny, nx)), origin="lower",
                  cmap=None if c == 0 else ["inferno", "viridis", "magma"][c - 1], vmin=0, vmax=1,
                  interpolation="bilinear", aspect="equal")
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_color("#333")
    a.set_title(heads[c], fontsize=11, color="#ddd", loc="left")
    ims.append(im)
fig.suptitle(title, fontsize=14, color="#eee", x=0.015, ha="left", y=0.985)
fig.text(0.5, 0.008, f"DNS · one-step premixed, Le=1, σ=5 · M_flame={m['mach']:g} (inflated) · "
         f"{nx}×{ny}×{nz} · flow left → right",
         ha="center", fontsize=9, color="#999")
fig.tight_layout(rect=(0, 0.025, 1, 0.95))
fig.canvas.draw()
w, h = fig.canvas.get_width_height()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                       "-s", f"{w}x{h}", "-r", "12", "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", stem + ".mp4"],
                      stdin=subprocess.PIPE)
for f in range(nf):
    li, vo, sp, te = frame(f)
    ims[0].set_data(srgb(aces(0.45 * (li / lmax)[..., None] * rgb[None, None, :])))
    ims[1].set_data(np.sqrt(np.clip(vo / vmax, 0, 1)))
    ims[2].set_data(np.clip(sp / smax, 0, 1))
    ims[3].set_data(np.clip(te / tmax, 0, 1))
    txt.set_text(f"t = {t0 + m['frameDt'] * f:5.2f} δ_D/S_L")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    ff.stdin.write(buf.tobytes())
    if f == nf - 1:
        fig.savefig(stem + ".png", dpi=100, facecolor=fig.get_facecolor())
ff.stdin.close(); ff.wait()
print(stem, f"frames {nf} lmax {lmax:.3g} vmax {vmax:.3g} smax {smax:.3g}")
