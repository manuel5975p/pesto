"""masterplan_combnozzle P3 gates G3.3 / G3.4 / G3.5 on the fv1d ladders (foam_flame.py cases fv1d,
fv1d_hll, fv1d_d40; dns runs only).

    python3 -m tools.chem.foamflame.p3gates      (or: python3 tools/chem/foamflame/p3gates.py)

G3.3: Delta = S(hll) - S(d4 0) per rung, window [7, 14] tau means of the stats' SL; band
|Delta| <= |S(d4 0) - S_ref| / 3, S_ref = 1 - gamma M^2 (sigma - 1); order of |Delta| >= 1.8 on both
pairs; the Shvab-Zeldovich defect max |T + (sigma - 1) Tu Y - sigma Tu| / (sigma Tu) over the window's
profile slabs, hll <= 4/3 x d4 0; shockCells 0 in every hll row. S(d4 0) at 7.5 and 30 cells is fv1d's
no-shock run: G3.4 (d4 0 == none, common stats columns and profiles byte for byte) is measured at 15.
"""
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RUNS = os.path.join(ROOT, "runs", "flames", "foam")
SIGMA, GAMMA, WIN = 5.0, 1.4, (7.0, 14.0)


def table(path):
    rows = [l for l in open(path) if not l.startswith("#")]
    r = list(csv.reader(rows))
    return r[0], r[1:]


def stats(run):
    h, rows = table(os.path.join(RUNS, run, "flame_stats.csv"))
    t, sl = h.index("t"), h.index("SL")
    w = [float(r[sl]) for r in rows if WIN[0] <= float(r[t]) <= WIN[1]]
    cells = max((float(r[h.index("shockCells")]) for r in rows), default=-1) if "shockCells" in h else None
    return {"S": sum(w) / len(w), "n": len(w), "rows": len(rows), "sensedMax": cells}


def sz(run, mach, gamma=GAMMA):
    h, rows = table(os.path.join(RUNS, run, "flame_profiles.csv"))
    it, iT, iY = h.index("t"), h.index("T"), h.index("Y")
    tu = 1.0 / (gamma * mach * mach)
    tad = SIGMA * tu
    return max(abs(float(r[iT]) + (SIGMA - 1.0) * tu * float(r[iY]) - tad) / tad for r in rows
               if WIN[0] <= float(r[it]) <= WIN[1])


def same_bytes(a, b):
    ha, ra = table(os.path.join(RUNS, a, "flame_stats.csv"))
    hb, rb = table(os.path.join(RUNS, b, "flame_stats.csv"))
    idx = [hb.index(x) for x in ha]
    rows = len(ra) == len(rb) and all([y[i] for i in idx] == x for x, y in zip(ra, rb))
    pa = [l for l in open(os.path.join(RUNS, a, "flame_profiles.csv")) if not l.startswith("#")]
    pb = [l for l in open(os.path.join(RUNS, b, "flame_profiles.csv")) if not l.startswith("#")]
    return rows and pa == pb


def main():
    out, ok = {}, True
    lines = []
    for tag, mach in (("", 0.01), ("_mn", 0.042)):
        sref = 1.0 - GAMMA * mach * mach * (SIGMA - 1.0)
        rung = {}
        for N in (7.5, 15, 30):
            base = f"fv1d/dns_c{N:g}{tag}"
            d40 = f"fv1d_d40/dns_c{N:g}{tag}"
            ref = d40 if os.path.exists(os.path.join(RUNS, d40, "flame_stats.csv")) else base
            h, z = stats(f"fv1d_hll/dns_c{N:g}{tag}"), stats(ref)
            d = h["S"] - z["S"]
            band = abs(z["S"] - sref) / 3.0
            szh, sz0 = sz(f"fv1d_hll/dns_c{N:g}{tag}", mach), sz(ref, mach)
            rung[N] = {"S_hll": h["S"], "S_d40": z["S"], "d40_from": ref, "Delta": d, "band": band,
                       "sz_hll": szh, "sz_d40": sz0, "sensedMax": h["sensedMax"], "rows": h["rows"]}
            g = abs(d) <= band and szh <= 4.0 / 3.0 * sz0 and h["sensedMax"] == 0
            ok = ok and g
            lines.append(f"M {mach:g} {N:>4g} cells: S hll {h['S']:.7f}, d4 0 {z['S']:.7f} ({ref}); Delta {d:+.3e} "
                         f"band {band:.3e} {'PASS' if abs(d) <= band else 'FAIL'}; SZ defect hll {szh:.3e} d4-0 {sz0:.3e} "
                         f"(ratio {szh / sz0:.4f}) {'PASS' if szh <= 4 / 3 * sz0 else 'FAIL'}; sensed max "
                         f"{h['sensedMax']:g} over {h['rows']} rows")
        ds = [abs(rung[N]["Delta"]) for N in (7.5, 15, 30)]
        orders = [math.log2(ds[0] / ds[1]) if ds[1] > 0 else math.inf, math.log2(ds[1] / ds[2]) if ds[2] > 0 else math.inf]
        og = all(o >= 1.8 for o in orders)
        ok = ok and og
        lines.append(f"M {mach:g}: order of |Delta| {orders[0]:.3f}, {orders[1]:.3f} {'PASS' if og else 'UNMET'} (>= 1.8)")
        out[f"M{mach:g}"] = {"rungs": {str(k): v for k, v in rung.items()}, "orders": orders}
    g34 = {tag: same_bytes(f"fv1d/dns_c15{tag}", f"fv1d_d40/dns_c15{tag}") for tag in ("", "_mn")}
    lines.append(f"G3.4 15 cells, d4 0 vs no --shock, common stats columns + profiles bytes: M 0.01 {g34['']}, M 0.042 {g34['_mn']}")
    ok = ok and all(g34.values())
    g35 = os.path.join(RUNS, "fv1d_d40/dns_c15_g12_mn_hll", "flame_stats.csv")
    if os.path.exists(g35):
        s = stats("fv1d_d40/dns_c15_g12_mn_hll")
        out["G3.5"] = s
        lines.append(f"G3.5 gamma 1.2 M 0.042 15 cells hll: S_L {s['S']:.7f} (reported; sensed max {s['sensedMax']:g})")
    out["G3.4"] = g34
    print("\n".join(lines))
    print("all counted gates:", "PASS" if ok else "NOT ALL PASS")
    with open(os.path.join(RUNS, "fv1d_hll", "p3gates.json"), "w") as f:
        json.dump(out, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
