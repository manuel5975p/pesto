#!/usr/bin/env python3
"""Growth rate of a single-mode 2D wrinkled premixed flame (docs/chem.md,
"Why TFLES misses the U2 DNS"; TPRI lesgap).

Reads `PREFIX_fuel.bin` (float32 frames, nf x ny x nx, i fastest) and
`PREFIX.json` (nx, frameDt) of a `dns flame --wrinkle A --wrinkle-modes 1
--emit bin` run. Per frame: the fresh-mass thickness x_I(y) = h sum_i Y, its
mode-1 amplitude a = 2 |DFT_1(x_I)| / ny (FP64), and the rate
s = d ln a / dt, the least-squares slope of ln a over t in [t0, t1].

    wrinkle_rate.py PREFIX --ny N --cells C [--t0 4] [--t1 16]
                    [--dth DTH] [--t-first T] [--json OUT]

--cells: cells per delta_th (`--cells-per-delta`), so h = dth / cells.
--dth:   delta_th / delta_D of the run's model (the dry-run's
         "delta_th/delta_D"; default 1.60165, sigma 5, beta 8).
--t-first: time of the first frame (default 0: the cadence starts at
         `--frame-start`, 0 by default, and frame k is the first step with
         t >= t_first + k fdt; tt 20 at fdt 0.5 writes 41 frames).
s does not depend on h (a constant factor in a); lambda does.
Units: lengths in delta_D, time in tau = delta_D / S_L.
"""
import argparse
import json
import math
import sys

import numpy as np


def rate(prefix, ny, cells, t0=4.0, t1=16.0, dth=1.60165, t_first=None):
    meta = json.load(open(prefix + ".json"))
    nx = meta["nx"]
    fdt = meta["frameDt"]
    Y = np.fromfile(prefix + "_fuel.bin", dtype=np.float32)
    nf = Y.size // (nx * ny)
    Y = Y[: nf * nx * ny].reshape(nf, ny, nx).astype(np.float64)
    h = dth / cells
    xi = Y.sum(2) * h
    a = np.abs(np.fft.rfft(xi, axis=1)[:, 1]) * 2 / ny
    tf = 0.0 if t_first is None else t_first
    t = tf + fdt * np.arange(nf)
    m = (t >= t0) & (t <= t1)
    s = float(np.polyfit(t[m], np.log(a[m]), 1)[0])
    lam = ny * h
    return dict(prefix=prefix, ny=ny, cells=cells, h=h, dth=dth,
                lam_dD=lam, lam_dth=lam / dth, k=2 * math.pi / lam, s=s,
                t0=t0, t1=t1, t=t.tolist(), a=a.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--cells", type=float, required=True)
    ap.add_argument("--t0", type=float, default=4.0)
    ap.add_argument("--t1", type=float, default=16.0)
    ap.add_argument("--dth", type=float, default=1.60165)
    ap.add_argument("--t-first", type=float, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = rate(a.prefix, a.ny, a.cells, a.t0, a.t1, a.dth, a.t_first)
    amp = " ".join(f"{r['a'][i]:.3f}" for i in range(0, len(r["a"]), 4))
    print(f"{a.prefix}: lambda={r['lam_dD']:.2f} delta_D = "
          f"{r['lam_dth']:.2f} delta_th  k={r['k']:.3f}  "
          f"s={r['s']:+.4f}/tau over [{a.t0:g}, {a.t1:g}]")
    print(f"  a(t={r['t'][0]:g} step {4 * (r['t'][1] - r['t'][0]):g}) = {amp}")
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
