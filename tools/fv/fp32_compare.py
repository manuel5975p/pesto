#!/usr/bin/env python3
"""FP32 round (docs/fv.md "Precision"): one dns FV run against a reference
run (FP64 dns or simpleFoam) -- CT = -Fx, CQ = -Mx (n = D = rho = 1) at the
last iteration, their relative difference, the settle-1e-3 iteration and loop
time of each side, the end-of-run jitter (max - min of CT and CQ over the last
100 iterations, relative), and from the dns log every solve's iteration
statistics: max iterations, solves at or above maxIter, floor-limited count,
the median initial residual of the last 100 iterations per field.

usage: fp32_compare.py RUN REF [--maxiter 1000] [--json]
  RUN  a dns --out prefix (PREFIX/postProcessing/forces/0, PREFIX.log, PREFIX_stats.csv)
  REF  a dns --out prefix or a simpleFoam case (postProcessing/forces/0)
"""
import argparse
import json
import os
import re
import statistics


def dat(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            rows.append([float(x) for x in v[1:]])
    return rows


def coeffs(root):
    d = os.path.join(root, "postProcessing", "forces", "0")
    F, M = dat(os.path.join(d, "force.dat")), dat(os.path.join(d, "moment.dat"))
    k = min(len(F), len(M))
    return [-F[i][0] for i in range(k)], [-M[i][0] for i in range(k)]


def settle(ct, cq, tol=1e-3):
    last = 0
    for i in range(len(ct)):
        if abs(ct[i] - ct[-1]) > tol * abs(ct[-1]) or abs(cq[i] - cq[-1]) > tol * abs(cq[-1]):
            last = i + 1
    return last + 1


def jitter(x, n=100):
    t = x[-n:]
    return (max(t) - min(t)) / abs(x[-1])


def mean(x, n=100):
    return sum(x[-n:]) / len(x[-n:])


def loop_ms(prefix, upto):
    """Sum of the stats `ms` column over iterations 1..upto (dns only)."""
    p = prefix + "_stats.csv"
    if not os.path.exists(p):
        return None
    col, s = None, 0.0
    with open(p) as f:
        for line in f:
            if line.startswith("#"):
                continue
            v = line.strip().split(",")
            if v[0] == "iter":
                col = v.index("ms")
                continue
            if int(v[0]) <= upto:
                s += float(v[col])
    return s


pat = re.compile(r"Solving for (\w+), Initial residual = ([^,]+), Final residual = ([^,]+), No Iterations (\d+)(, floor-limited)?")


def solves(log, maxiter):
    per, it = {}, 0
    with open(log) as f:
        for line in f:
            if line.startswith("Time = "):
                it += 1
                continue
            m = pat.search(line)
            if not m:
                continue
            fld = m.group(1)
            if fld in ("Ux", "Uy", "Uz"):
                fld = "U"
            d = per.setdefault(fld, {"n": 0, "maxIt": 0, "atMax": 0, "floor": 0, "its": 0, "r0": []})
            n = int(m.group(4))
            d["n"] += 1
            d["its"] += n
            d["maxIt"] = max(d["maxIt"], n)
            d["atMax"] += n >= maxiter
            d["floor"] += m.group(5) is not None
            d["r0"].append((it, float(m.group(2))))
    for d in per.values():
        last = max(i for i, _ in d["r0"])
        tail = [r for i, r in d["r0"] if i > last - 100]
        d["r0tail"] = statistics.median(tail)
        del d["r0"]
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("ref")
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    ct, cq = coeffs(a.run)
    rct, rcq = coeffs(a.ref)
    s, rs = settle(ct, cq), settle(rct, rcq)
    out = {
        "CT": ct[-1], "CQ": cq[-1], "CTref": rct[-1], "CQref": rcq[-1],
        "dCT": (ct[-1] - rct[-1]) / abs(rct[-1]), "dCQ": (cq[-1] - rcq[-1]) / abs(rcq[-1]),
        "settle": s, "settleRef": rs, "loopSettleMs": loop_ms(a.run, s), "loopSettleRefMs": loop_ms(a.ref, rs),
        "jitCT": jitter(ct), "jitCQ": jitter(cq), "jitCTref": jitter(rct), "jitCQref": jitter(rcq),
        # the last-100 means (the floor's jitter averaged out) vs the reference's
        "dCTmean": (mean(ct) - mean(rct)) / abs(mean(rct)), "dCQmean": (mean(cq) - mean(rcq)) / abs(mean(rcq)),
    }
    if os.path.exists(a.run + ".log"):
        out["solves"] = solves(a.run + ".log", a.maxiter)
    if a.json:
        print(json.dumps(out))
        return
    print(f"CT {out['CT']:.12e} ref {out['CTref']:.12e} rel {out['dCT']:+.2e}; "
          f"CQ {out['CQ']:.12e} ref {out['CQref']:.12e} rel {out['dCQ']:+.2e}")
    ls, lr = out["loopSettleMs"], out["loopSettleRefMs"]
    print(f"settle 1e-3: it {s} (loop {ls / 1000 if ls else float('nan'):.2f} s) ref it {rs}"
          f" (loop {lr / 1000 if lr else float('nan'):.2f} s)")
    print(f"jitter last 100 it: CT {out['jitCT']:.2e} CQ {out['jitCQ']:.2e} (ref {out['jitCTref']:.2e} / {out['jitCQref']:.2e});"
          f" last-100 means: CT {out['dCTmean']:+.2e} CQ {out['dCQmean']:+.2e}")
    for k, d in out.get("solves", {}).items():
        print(f"  {k}: {d['n']} solves, {d['its']} iterations, max {d['maxIt']}, at maxIter {d['atMax']}, "
              f"floor-limited {d['floor']}, median initial residual last 100 it {d['r0tail']:.3e}")


if __name__ == "__main__":
    main()
