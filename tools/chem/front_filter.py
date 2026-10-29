#!/usr/bin/env python3
"""A-priori filtering of 3D premixed flame volumes (docs/chem.md, "Why TFLES
misses the U2 DNS"; TPRI lesgap).

Reads `PREFIX_fuel.bin` (float32 volumes, nf x n x n x nx, i fastest; the
run's `--emit volume`, ny = nz = n). Per frame, with c = 1 - Y:
  A      = int |grad c| dV / (L_y L_z)  (resolved flame-surface ratio;
           central differences, one-sided-free interior in x, periodic y, z)
  A_D    = the same of c filtered by Pope's Gaussian G = exp(-6 x^2 / D^2)
           (transfer exp(-k^2 D^2 / 24) per axis; y, z by FFT, x by FFT of an
           edge-padded copy), D = delta x delta_th
  Xi_D   = A / A_D: the sub-filter wrinkling an efficiency is meant to restore
  front  = x_I(y, z) = h sum_i Y minus its mean: rms and the mean-square per
           |k| shell; shells are in units of the box fundamental k0 = 2 pi/L
           (edges 0.765, 1.147, 1.721, 2.677, 4.206, 6.883, 15.29 k0: the
           U2 sigma 5 box's 0.2 ... 4.0 / delta_D), shell 1 = the box mode
  share  = shell-1 mean square / the sum over shells

    front_filter.py PREFIX --n N --cells C [--nx NX] [--delta 1.5,2,3]
                    [--dth DTH] [--t-first T] [--json OUT]

--cells: cells per delta_th; --dth: delta_th/delta_D (default 1.60165,
sigma 5); --nx: default from PREFIX.json; --t-first: first frame time
(default 0, i.e. the run's `--frame-start`: frame k is the first step with
t >= t_first + k frameDt). Lengths in delta_D.
"""
import argparse
import json
import math
import sys

import numpy as np

# vol.py's shell edges (1/delta_D) at the U2 sigma 5 box (90 cells, h =
# 1.60165/6): k0_ref = 2 pi / 24.02475.
_K0_REF = 2 * math.pi / (90 * 1.60165 / 6)
_EDGES_REF = [0.2, 0.3, 0.45, 0.7, 1.1, 1.8, 4.0]


def grad_area(c, h, n):
    gx = np.zeros_like(c)
    gx[..., 1:-1] = (c[..., 2:] - c[..., :-2]) / (2 * h)
    gy = (np.roll(c, -1, 1) - np.roll(c, 1, 1)) / (2 * h)
    gz = (np.roll(c, -1, 0) - np.roll(c, 1, 0)) / (2 * h)
    L = n * h
    return np.sqrt(gx * gx + gy * gy + gz * gz).sum() * h ** 3 / (L * L)


def gauss(c, h, Dw):
    nz, ny, nx = c.shape
    pad = int(math.ceil(2 * Dw / h)) + 4
    cp = np.concatenate([np.repeat(c[..., :1], pad, 2), c,
                         np.repeat(c[..., -1:], pad, 2)], 2)
    ky = 2 * np.pi * np.fft.fftfreq(ny, h)
    kz = 2 * np.pi * np.fft.fftfreq(nz, h)
    kx = 2 * np.pi * np.fft.fftfreq(cp.shape[2], h)
    G = (np.exp(-kz[:, None, None] ** 2 * Dw * Dw / 24) *
         np.exp(-ky[None, :, None] ** 2 * Dw * Dw / 24) *
         np.exp(-kx[None, None, :] ** 2 * Dw * Dw / 24))
    f = np.real(np.fft.ifftn(np.fft.fftn(cp) * G))
    return f[..., pad:-pad]


def front_spectrum(Y, h, n):
    xi = Y.sum(2) * h
    xi = xi - xi.mean()
    P = np.abs(np.fft.fft2(xi)) ** 2 / (n * n) ** 2
    k = 2 * np.pi * np.fft.fftfreq(n, h)
    K = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    k0 = 2 * math.pi / (n * h)
    e = [x * k0 / _K0_REF for x in _EDGES_REF]
    spec = [float(P[(K >= lo) & (K < hi)].sum()) for lo, hi in zip(e, e[1:])]
    return math.sqrt((xi ** 2).mean()), spec


def analyse(prefix, n, cells, nx=None, deltas=(1.5, 2.0, 3.0), dth=1.60165,
            t_first=None):
    meta = json.load(open(prefix + ".json"))
    nx = meta["nx"] if nx is None else nx
    fdt = meta["frameDt"]
    tf = 0.0 if t_first is None else t_first
    a = np.fromfile(prefix + "_fuel.bin", dtype=np.float32)
    per = nx * n * n
    nf = a.size // per
    Yv = a[: nf * per].reshape(nf, n, n, nx)
    h = dth / cells
    out = []
    for f in range(nf):
        Y = Yv[f].astype(np.float64)
        c = 1 - Y
        A = grad_area(c, h, n)
        AD = {d: grad_area(gauss(c, h, d * dth), h, n) for d in deltas}
        rmsx, spec = front_spectrum(Y, h, n)
        out.append(dict(t=tf + f * fdt, A=A, AD=AD,
                        Xi={d: A / v for d, v in AD.items()}, rmsx=rmsx,
                        spec=spec, share=spec[0] / sum(spec)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--cells", type=float, required=True)
    ap.add_argument("--nx", type=int, default=None)
    ap.add_argument("--delta", default="1.5,2,3")
    ap.add_argument("--dth", type=float, default=1.60165)
    ap.add_argument("--t-first", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    deltas = [float(x) for x in a.delta.split(",")]
    rows = analyse(a.prefix, a.n, a.cells, a.nx, deltas, a.dth, a.t_first)
    for r in rows:
        xs = " ".join(f"A_{d:g}={r['AD'][d]:.3f}(Xi {r['Xi'][d]:.3f})"
                      for d in deltas)
        sp = " ".join(f"{math.sqrt(s):.3f}" for s in r["spec"])
        print(f"t={r['t']:g} A={r['A']:.3f} {xs} | xI rms={r['rmsx']:.3f} "
              f"shell rms [{sp}] share1={r['share']:.2f}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
