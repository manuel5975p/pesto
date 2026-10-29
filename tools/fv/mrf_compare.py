#!/usr/bin/env python3
"""Compares a dns FV-path run with the simpleFoam run of the same case
(docs/fv.md, phase MRF): rotor CT = -Fx / (rho n^2 D^4), CQ = -Mx /
(rho n^2 D^5) from postProcessing/<forces>/0/{force,moment}.dat, their final
relative difference, and the iterations each side needs to settle and to
reach a residual level.

  settle(tol): 1 + the last iteration whose CT or CQ is more than tol x
               |final| away from that side's final value
  res(level):  the first iteration whose initial residuals (every solved U
               component and the first p solve) are all below level

usage: mrf_compare.py FOAM_CASE DNS_OUT [--rho R] [--n N] [--D D] [--json]
  FOAM_CASE  the simpleFoam case (postProcessing/forces/0, log.simpleFoam)
  DNS_OUT    the dns --out prefix (PREFIX/postProcessing/forces/0,
             PREFIX_stats.csv)
"""
import argparse
import json
import os
import re
import sys


def dat(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            rows.append((float(v[0]), [float(x) for x in v[1:]]))
    return rows


def coeffs(root, rho, n, D):
    d = os.path.join(root, "postProcessing", "forces", "0")
    F, M = dat(os.path.join(d, "force.dat")), dat(os.path.join(d, "moment.dat"))
    k = min(len(F), len(M))
    ct = [-F[i][1][0] / (rho * n * n * D ** 4) for i in range(k)]
    cq = [-M[i][1][0] / (rho * n * n * D ** 5) for i in range(k)]
    return ct, cq


def settle(ct, cq, tol):
    last = 0
    for i in range(len(ct)):
        if abs(ct[i] - ct[-1]) > tol * abs(ct[-1]) or abs(cq[i] - cq[-1]) > tol * abs(cq[-1]):
            last = i + 1
    return last + 1


def foam_res(case):
    res, cur, seen = [], None, False
    clock, exe = None, []
    with open(os.path.join(case, "log.simpleFoam")) as f:
        for line in f:
            if line.startswith("Time = "):
                if cur is not None:
                    res.append(max(cur))
                cur, seen = [], False
            m = re.search(r"Solving for (U[xyz]|p), Initial residual = ([^,]+),", line)
            if m and cur is not None:
                if m.group(1) == "p":
                    if not seen:
                        cur.append(float(m.group(2)))
                        seen = True
                else:
                    cur.append(float(m.group(2)))
            m = re.search(r"ExecutionTime = ([0-9.e+-]+) s\s+ClockTime = ([0-9.e+-]+) s", line)
            if m:
                clock = (float(m.group(1)), float(m.group(2)))
                exe.append(float(m.group(1)))
    if cur:
        res.append(max(cur))
    return res, clock, exe


def dns_res(prefix):
    res, ms, cum = [], 0.0, []
    with open(prefix + "_stats.csv") as f:
        head = None
        for line in f:
            if line.startswith("#"):
                continue
            v = line.strip().split(",")
            if head is None:
                head = v
                continue
            r = dict(zip(head, v))
            u = [float(r[k]) for k in ("Ux0", "Uy0", "Uz0") if float(r[k]) > 0]
            res.append(max(u + [float(r["p0"])]))
            ms += float(r["ms"])
            cum.append(ms / 1000.0)
    return res, ms, cum


def first_below(res, level):
    for i, r in enumerate(res):
        if r < level:
            return i + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("foam")
    ap.add_argument("dns")
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--n", type=float, default=1.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    fct, fcq = coeffs(a.foam, a.rho, a.n, a.D)
    dct, dcq = coeffs(a.dns, a.rho, a.n, a.D)
    fres, clock, fexe = foam_res(a.foam)
    dres, dms, dcum = dns_res(a.dns)
    st = {t: (settle(fct, fcq, t), settle(dct, dcq, t)) for t in (1e-2, 1e-3, 1e-4)}
    out = {
        "iters": {"foam": len(fct), "dns": len(dct)},
        "CT": {"foam": fct[-1], "dns": dct[-1], "rel": dct[-1] / fct[-1] - 1},
        "CQ": {"foam": fcq[-1], "dns": dcq[-1], "rel": dcq[-1] / fcq[-1] - 1},
        "settle": {str(t): {"foam": f, "dns": d,
                            "foam_exec_s": fexe[min(f, len(fexe)) - 1] if fexe else None,
                            "dns_loop_s": dcum[min(d, len(dcum)) - 1] if dcum else None}
                   for t, (f, d) in st.items()},
        "residual": {str(l): {"foam": first_below(fres, l), "dns": first_below(dres, l)}
                     for l in (1e-3, 1e-4, 1e-5, 1e-6)},
        "foam_execution_clock_s": clock,
        "dns_loop_s": dms / 1000.0,
    }
    if a.json:
        json.dump(out, sys.stdout, indent=1)
        print()
        return
    print(f"iterations      foam {out['iters']['foam']}  dns {out['iters']['dns']}")
    for q in ("CT", "CQ"):
        o = out[q]
        print(f"{q}  foam {o['foam']:+.10e}  dns {o['dns']:+.10e}  rel {o['rel']:+.3e}")
    for t, o in out["settle"].items():
        print(f"settle {t:>6}   foam {o['foam']:>5} ({o['foam_exec_s']:.2f} s)  dns {o['dns']:>5} "
              f"({o['dns_loop_s']:.2f} s loop)")
    for l, o in out["residual"].items():
        print(f"residual {l:>6} foam {o['foam']}  dns {o['dns']}")
    print(f"foam ExecutionTime/ClockTime {clock} s; dns loop {dms / 1000:.2f} s")


if __name__ == "__main__":
    main()
