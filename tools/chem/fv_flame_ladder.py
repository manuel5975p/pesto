#!/usr/bin/env python3
"""The planar flame ladder of the FV path (docs/fvcomp.md "Species").

Each rung runs through simjob (295 s cap): structured FP32, FV FP32, FV FP64 on the
same box, then with --meshes the FV FP64 flame on the non-orthogonal polyMeshes of
tools/chem/fv_flame_mesh.py. Prints S_L and delta_th means over [tt/2, tt], their
differences and orders.

    tools/chem/fv_flame_ladder.py --cells 7.5,15,30 --tt 14 --meshes rot,shear,wavy,all

--physics lowmach: the same rungs on the low-Mach solvers (docs/fvincomp.md "Low-Mach"),
structured low-Mach FP32 for the structured column, plus the compressible FV flame at
--mach (FP64, tag cm64); --ny-mesh N overrides the polyMesh rung's ny.
"""
import argparse
import csv
import math
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DNS = os.environ.get("DNS", os.path.join(ROOT, "build", "dns"))
SIMJOB = os.environ.get("SIMJOB", os.path.join(ROOT, "build", "simjob"))
MESHES = {"rot": ["--rotate", "30"], "shear": ["--shear", "30"], "wavy": ["--wavy", "2"],
          "all": ["--shear", "30", "--wavy", "2", "--rotate", "20"]}
COMMON = ["--inflow-speed", "0.98", "--emit", "stats", "--quiet"]


def job(label, args):
    if os.path.exists(args[args.index("--out") + 1] + ".json"):  # a finished rung is kept
        return True
    r = subprocess.run([SIMJOB, "submit", "--gpu", "-t", "295", "-m", "8192", "--wait", "--label", label, "--", DNS, *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{label}: FAILED rc {r.returncode}: not measured: dns {' '.join(args)}", flush=True)
        return False
    return True


def window(prefix, tt):
    rows = [r for r in csv.DictReader(l for l in open(prefix + "_stats.csv") if not l.startswith("#"))]
    w = [r for r in rows if float(r["t"]) >= 0.5 * tt]
    mean = lambda k: sum(float(r[k]) for r in w) / len(w)
    return {"SL": mean("SL"), "dth": mean("dth"), "span": max(float(r["SL"]) for r in w) - min(float(r["SL"]) for r in w),
            "hits": max(float(r["yHits"]) for r in rows), "steps": len(rows)}


def box(c, phys, ny=4):
    """nx, ny-free box facts from the dry run of the structured flame."""
    out = subprocess.run([DNS, "flame", *phys, "--cells-per-delta", str(c), "--ny", str(ny), "--dry-run", "--out", "/tmp/dns/sp/x"],
                         capture_output=True, text=True).stdout
    m = re.search(r"grid\s+(\d+) x (\d+) x 1\s+h = ([0-9.e+-]+).*box ([0-9.e+-]+) x ([0-9.e+-]+)", out)
    x = re.search(r"x_f0 = ([0-9.e+-]+)", out)
    return int(m[1]), int(m[2]), float(m[3]), float(m[4]), float(m[5]), float(x[1])


def order(a, b, ra):
    return math.log(abs(a / b)) / math.log(ra) if a != 0 and b != 0 else float("nan")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cells", default="7.5,15,30")
    a.add_argument("--tt", type=float, default=14.0)
    a.add_argument("--meshes", default="")
    a.add_argument("--out", default="/tmp/dns/sp/ladder")
    a.add_argument("--dth", type=float, default=1.60165)
    a.add_argument("--mach", default="0.01")
    a.add_argument("--physics", default="compressible", choices=["compressible", "lowmach"])
    a.add_argument("--ny-mesh", type=int, default=0)
    a.add_argument("--tol", default="")
    a.add_argument("--wrinkle", default="", help="a 2D wrinkled flame, ny 32 c / 15: the momentum path (no meshes)")
    a.add_argument("--tags", default="", help="only these box runs, e.g. fv64")
    o = a.parse_args()
    os.makedirs(o.out, exist_ok=True)
    cells = [float(c) for c in o.cells.split(",")]
    lm = o.physics == "lowmach"
    tt = ["--tt", str(o.tt)]
    ma = ["--mach", o.mach]
    phys = ["--physics", "lowmach"] if lm else []
    tol = ["--tol", o.tol] if o.tol and lm else []
    runs = [("st32", phys + ([] if lm else ma)), ("fv32", phys + ([] if lm else ma) + ["--solver", "fv"] + tol),
            ("fv64", phys + ([] if lm else ma) + ["--solver", "fv", "--fv-precision", "fp64"] + tol)]
    if lm:
        runs.append(("cm64", ma + ["--solver", "fv", "--fv-precision", "fp64"]))
    res = {}
    for c in cells:
        g = ["--cells-per-delta", str(c)]
        if o.wrinkle:
            g += ["--ny", str(int(round(32 * c / 15))), "--wrinkle", o.wrinkle, "--wrinkle-modes", "1"]
        for tag, extra in runs:
            if o.tags and tag not in o.tags.split(","):
                continue
            p = f"{o.out}/{tag}_{c}"
            if job(f"sp-{tag}-{c}", ["flame", *g, *tt, *COMMON, *extra, "--out", p]):
                res[tag, c] = window(p, o.tt)
        nx, _, h, lx, _, xf = box(c, phys)
        ny = o.ny_mesh or int(round(32 * c / 15))
        for m in filter(None, o.meshes.split(",")):
            d = f"{o.out}/mesh_{m}_{c}"
            subprocess.run([sys.executable, os.path.join(ROOT, "tools/chem/fv_flame_mesh.py"), "--nx", str(nx), "--ny", str(ny),
                            "--lx", str(lx), "--ly", str(ny * h), *MESHES[m], *(["--lowmach"] if lm else []), "--out", d],
                           check=True, capture_output=True)
            p = f"{o.out}/{m}_{c}"
            if job(f"sp-{m}-{c}", ["flame", "--polymesh", d, "--bc", d + "/bc.toml", "--upstream", str(xf / o.dth), *tt, *COMMON,
                                   *phys, *([] if lm else ma), *tol, "--fv-precision", "fp64", "--out", p]):
                res[m, c] = window(p, o.tt)
    print("cells | structured FP32 SL, dth | FV FP32 SL, dth | FV FP64 SL, dth | FV32-st32 SL, dth | FV64-oracle SL")
    for c in cells:
        s, f, d = res.get(("st32", c)), res.get(("fv32", c)), res.get(("fv64", c))
        if not (s and f and d):
            continue
        print(f"{c:5g} | {s['SL']:.6f} {s['dth']:.5f} | {f['SL']:.6f} {f['dth']:.5f} | {d['SL']:.6f} {d['dth']:.5f} | "
              f"{f['SL'] - s['SL']:+.2e} {f['dth'] - s['dth']:+.2e} | {d['SL'] - 1:+.2e}  span {d['span']:.1e} hits {d['hits']:g}")
    if lm:
        print("cells | compressible FV FP64 SL, dth | low-Mach FV FP64 - compressible SL, dth")
        for c in cells:
            f, d = res.get(("cm64", c)), res.get(("fv64", c))
            if f and d:
                print(f"{c:5g} | {f['SL']:.6f} {f['dth']:.5f} | {d['SL'] - f['SL']:+.2e} {d['dth'] - f['dth']:+.2e}")
    for m in filter(None, o.meshes.split(",")):
        print(f"mesh {m}: cells, SL, dth, SL-planar, dth-planar")
        prev = None
        for c in cells:
            r, d = res.get((m, c)), res.get(("fv64", c))
            if not (r and d):
                print(f"{c:5g} not measured")
                continue
            e = r["SL"] - d["SL"]
            print(f"{c:5g} {r['SL']:.6f} {r['dth']:.5f} {e:+.2e} {r['dth'] - d['dth']:+.2e}  hits {r['hits']:g}"
                  + (f"  order {order(prev, e, c / pc):.2f}" if prev else ""))
            prev, pc = e, c


if __name__ == "__main__":
    main()
