#!/usr/bin/env python3
"""Window statistics of a `dns flame --turb` run (docs/chem.md, "3D and
turbulence"): means over t in [t0, t1] of the consumption speed S_T (column
SL), the flame-surface ratio (area), I0 = (S_T/S_L)/(A_T/A), u' ahead of the
brush (uPrime), the brush thickness (dth), the inflow speed; the two window
halves of S_T; the x_f excursion about x_f0; max Tmax/Tb; yHits / yMax; and the
report-only Peters 1999 / Damkoehler predictions at the measured u'.

    turb_stats.py PREFIX [--t0 T] [--t1 T] [--sl SL_LAMINAR] [--lt LT]
                  [--xf0 X] [--sigma S] [--mach M]

--sl: the laminar S_L of the same grid (a 1D run's window mean; default 1).
--lt: l_t in delta_D (Peters; default 2.5 delta_th = 4.004).
numpy-free: csv only.
"""
import argparse
import csv
import math


def peters(u, l, lf=1.0, a4=0.78, b1=2.0, b3=1.0):
    A = a4 * b3 * b3 / (2 * b1)
    r = A * l / lf
    return 1 - r + math.sqrt(r * r + a4 * b3 * b3 * u * l / lf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--lt", type=float, default=2.5 * 1.60165)
    ap.add_argument("--xf0", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--mach", type=float, default=None)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.prefix + "_stats.csv")))
    t = [float(r["t"]) for r in rows]
    t1 = a.t1 if a.t1 is not None else t[-1]
    t0 = a.t0 if a.t0 is not None else 0.5 * t1
    w = [r for r in rows if t0 <= float(r["t"]) <= t1]
    n = len(w)

    def col(k):
        return [float(r[k]) for r in w]

    def mean(v):
        return sum(v) / len(v)

    S = col("SL")
    h1, h2 = mean(S[: n // 2]), mean(S[n // 2:])
    ST = mean(S) / a.sl
    area = mean(col("area"))
    up = mean(col("uPrime"))
    xf = col("xf")
    out = {
        "window": f"[{t0:g}, {t1:g}] ({n} rows)",
        "S_T/S_L": ST,
        "halves": (h1 / a.sl, h2 / a.sl, abs(h1 - h2) / mean(S)),
        "area": area,
        "I0": ST / area,
        "I0 rows": (min(col("I0")), max(col("I0"))),
        "uPrime": up,
        "brush dth": mean(col("dth")),
        "u_in": mean(col("uInflow")),
        "xf range": (min(xf), max(xf)),
        "yHits max": max(col("yHits")),
        "yMax-1 max": max(col("yMax")) - 1,
        "yMin min": min(col("yMin")),
        "omegaEdge max": max(col("omegaEdge")),
        "Peters(u')": peters(up, a.lt),
        "Damkoehler(u')": 1 + up,
    }
    if a.xf0 is not None:
        out["xf - xf0"] = (min(xf) - a.xf0, max(xf) - a.xf0)
    if a.mach is not None:
        Tb = a.sigma / (1.4 * a.mach * a.mach)
        out["Tmax/Tb max"] = max(col("Tmax")) / Tb
    for k, v in out.items():
        if isinstance(v, tuple):
            print(f"{k:16s} " + "  ".join(f"{x:.5g}" for x in v))
        elif isinstance(v, float):
            print(f"{k:16s} {v:.5g}")
        else:
            print(f"{k:16s} {v}")


if __name__ == "__main__":
    main()
