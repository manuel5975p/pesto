#!/usr/bin/env python3
"""FV vs MAC ladder numbers (docs/fvincomp.md).

  incomp_ladder.py tgv  FV_32.txt MAC_32.txt FV_64.txt MAC_64.txt ...
      `dns tgv` series (t,K,dKdt,epsVisc): peak -dK/dt and epsVisc of each, the
      FV - MAC difference of each pair, its observed order between pairs (N from
      the file name's digits), and the rms difference of the two series.
  incomp_ladder.py cd T0 T1 FV_stats.csv MAC_stats.csv ...
      time mean and range of cd, cl over [T0, T1] of each stats file, the FV -
      MAC difference per pair and its order. Columns by header name.
"""
import csv
import math
import re
import sys


def rows(path):
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#") and ("," in ln)]
    r = list(csv.reader(lines))
    head = r[0]
    out = {h: [] for h in head}
    for row in r[1:]:
        try:
            v = [float(x) for x in row]
        except ValueError:
            continue
        if len(v) != len(head):
            continue
        for h, x in zip(head, v):
            out[h].append(x)
    return out


def level(path):
    m = re.findall(r"\d+", path.split("/")[-1])
    return int(m[-1])


def order(a, b, na, nb):
    return math.log(abs(a) / abs(b)) / math.log(nb / na) if a and b else float("nan")


def tgv(files):
    prev = None
    for fv, mac in zip(files[::2], files[1::2]):
        a, b = rows(fv), rows(mac)
        n = level(fv)
        pa, pb = max(a["dKdt"]), max(b["dKdt"])
        ea, eb = max(a["epsVisc"]), max(b["epsVisc"])
        ta = a["t"][a["epsVisc"].index(ea)]
        tb = b["t"][b["epsVisc"].index(eb)]
        m = min(len(a["t"]), len(b["t"]))
        rms = math.sqrt(sum((a["dKdt"][i] - b["dKdt"][i]) ** 2 for i in range(1, m)) / (m - 1))
        rmsE = math.sqrt(sum((a["epsVisc"][i] - b["epsVisc"][i]) ** 2 for i in range(1, m)) / (m - 1))
        d = pa - pb
        line = (f"N {n}: peak -dK/dt FV {pa:.6f} MAC {pb:.6f} diff {d:+.3e}; peak eps FV {ea:.6f} (t {ta:.2f}) "
                f"MAC {eb:.6f} (t {tb:.2f}) diff {ea - eb:+.3e}; rms(-dK/dt) {rms:.3e} rms(eps) {rmsE:.3e}")
        if prev:
            line += (f"; order peak {order(prev[1], d, prev[0], n):.2f} rms {order(prev[2], rms, prev[0], n):.2f}"
                     f" rms(eps) {order(prev[3], rmsE, prev[0], n):.2f}")
        print(line)
        prev = (n, d, rms, rmsE)


def window(path, t0, t1):
    r = rows(path)
    pick = [i for i, t in enumerate(r["t"]) if t0 <= t <= t1]
    cd = [r["cd"][i] for i in pick]
    cl = [r["cl"][i] for i in pick]
    return sum(cd) / len(cd), min(cd), max(cd), max(abs(c) for c in cl), len(pick)


def cd(t0, t1, files):
    prev = None
    for fv, mac in zip(files[::2], files[1::2]):
        a = window(fv, t0, t1)
        b = window(mac, t0, t1)
        n = level(fv)
        d = a[0] - b[0]
        line = (f"N {n}: cd FV {a[0]:.5f} [{a[1]:.5f}, {a[2]:.5f}] |cl|max {a[3]:.4f} ({a[4]} rows)  "
                f"MAC {b[0]:.5f} [{b[1]:.5f}, {b[2]:.5f}] |cl|max {b[3]:.4f} ({b[4]} rows)  diff {d:+.4e}")
        if prev:
            line += f"  order {order(prev[1], d, prev[0], n):.2f}"
        print(line)
        prev = (n, d)


if __name__ == "__main__":
    if sys.argv[1] == "tgv":
        tgv(sys.argv[2:])
    else:
        cd(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4:])
