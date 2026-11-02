#!/usr/bin/env python3
"""The slot-jet diffusion flame ladder of the FV path (docs/fvcomp.md "Diffusion flame").

Each rung runs through simjob (295 s cap): structured default, structured --nscbc-etain
1000 (the stiff inflow foam matches, docs/chem.md G3a), FV FP64 on the same box, then
with --meshes the FV FP64 flame on the polyMeshes of tools/chem/fv_flame_mesh.py --jet.
Prints means over the last quarter of each run. A rung the cap kills prints its command.

    tools/chem/fv_jet_ladder.py --cells 2,3,4.5 --tt 160 --meshes rot,shear
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
COMMON = ["--mode", "diffusion", "--emit", "stats", "--quiet"]
KEYS = ["xTip", "half", "TmaxIn", "burnRatio", "fuelIn", "pMean"]


def job(label, args):
    r = subprocess.run([SIMJOB, "submit", "--gpu", "-t", "295", "-m", "8192", "--wait", "--label", label, "--", DNS, *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"{label}: FAILED rc {r.returncode}: not measured: dns {' '.join(args)}", flush=True)
        return False
    return True


def window(prefix):
    rows = list(csv.DictReader(l for l in open(prefix + "_stats.csv") if not l.startswith("#")))
    tt = float(rows[-1]["t"])
    w = [r for r in rows if float(r["t"]) >= 0.75 * tt]
    o = {k: sum(float(r[k]) for r in w) / len(w) for k in KEYS}
    o["span"] = max(float(r["xTip"]) for r in w) - min(float(r["xTip"]) for r in w)
    o["hits"] = max(float(r["hits"]) for r in rows)
    return o


def box(c):
    out = subprocess.run([DNS, "flame", *COMMON[:2], "--cells-per-unit", str(c), "--dry-run", "--out", "/tmp/dns/df/x"],
                         capture_output=True, text=True).stdout
    m = re.search(r"grid\s+(\d+) x (\d+) x 1\s+h = ([0-9.e+-]+)", out)
    return int(m[1]), int(m[2]), float(m[3])


def line(tag, r):
    return (f"{tag:>10} tip {r['xTip']:.4f} span {r['span']:.1e} half {r['half']:.4f} TmaxIn {r['TmaxIn']:.3f} "
            f"burn/in {r['burnRatio']:.5f} fuelIn {r['fuelIn']:.4f} p/p0 {r['pMean']:.5f} hits {r['hits']:g}")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cells", default="2,3,4.5")
    a.add_argument("--tt", default="160")
    a.add_argument("--meshes", default="")
    a.add_argument("--out", default="/tmp/dns/df/ladder")
    o = a.parse_args()
    os.makedirs(o.out, exist_ok=True)
    tt = ["--tt", o.tt]
    for c in o.cells.split(","):
        g = ["--cells-per-unit", c]
        res = {}
        for tag, extra in (("st", []), ("st1000", ["--nscbc-etain", "1000"]), ("fv64", ["--solver", "fv", "--fv-precision", "fp64"])):
            p = f"{o.out}/{tag}_{c}"
            if job(f"jet-{tag}-{c}", ["flame", *COMMON, *g, *tt, *extra, "--out", p]):
                res[tag] = window(p)
        nx, ny, h = box(c)
        for m in filter(None, o.meshes.split(",")):
            d = f"{o.out}/mesh_{m}_{c}"
            subprocess.run([sys.executable, os.path.join(ROOT, "tools/chem/fv_flame_mesh.py"), "--nx", str(nx), "--ny", str(ny),
                            "--lx", str(nx * h), "--ly", str(ny * h), *MESHES[m], "--jet", "--out", d], check=True, capture_output=True)
            p = f"{o.out}/{m}_{c}"
            if job(f"jet-{m}-{c}", ["flame", *COMMON, "--polymesh", d, "--bc", d + "/bc.toml", *tt, "--fv-precision", "fp64",
                                    "--out", p]):
                res[m] = window(p)
        print(f"cells {c}: {nx} x {ny}, h {h:g}")
        for tag, r in res.items():
            print(line(tag, r), flush=True)
        if "fv64" in res:
            for tag, r in res.items():
                print(f"{tag:>10} - fv64: tip {r['xTip'] - res['fv64']['xTip']:+.3e} half {r['half'] - res['fv64']['half']:+.3e} "
                      f"TmaxIn {r['TmaxIn'] / res['fv64']['TmaxIn'] - 1:+.3e}")


if __name__ == "__main__":
    main()
