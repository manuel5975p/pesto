#!/usr/bin/env python3
"""Grid ladder of the FV family vs the structured solver on one O-grid case
(docs/fvcomp.md "Cross-verification"). Both start from the same quiet ramp on
the same nodes, so cd(t), cl(t) are compared at FIXED t: the difference must
shrink with h whether or not the case has settled. Every run goes through
simjob. Usage:
  comp_ladder.py --re 40 --mach 0.2 --ft 3 --levels 96:48,192:96,384:192 [--far 20] [--trip 0]
                 [--les 0.17 --nz 16] [--rans sa-neg --beta 4] [--t 94] [--compensated] [--out /tmp/dns/ladder]
"""
import argparse, csv, math, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DNS = os.path.join(ROOT, "build", "dns")
SIMJOB = os.path.join(ROOT, "build", "simjob")


def run(tag, a, extra, limit):
    prefix = os.path.join(a.out, tag)
    cmd = [SIMJOB, "submit", "--gpu", "-t", str(limit), "-m", "8192", "--wait", "--label", tag, "--", a.dns, "flow",
           "--physics", "compressible", "--ogrid", "--body", "cylinder", "--re", str(a.re), "--mach", str(a.mach),
           "--ft", str(a.ft), "--ogrid-far", str(a.far), "--emit", "stats", "--out", prefix] + extra
    if a.compensated:
        cmd += ["--compensated"]
    if a.trip > 0:
        cmd += ["--ogrid-trip", str(a.trip)]
    if a.nz > 1:
        cmd += ["--nz", str(a.nz)]
    if a.les > 0:
        cmd += ["--les", str(a.les), "--sgs", a.sgs]
    if a.rans:
        cmd += ["--rans", a.rans]
    if a.beta > 0:
        cmd += ["--ogrid-beta", str(a.beta)]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    if r.returncode != 0:
        sys.exit(f"{tag}: rc {r.returncode}\n{r.stdout}")
    return prefix + "_stats.csv"


def rows(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    return [{k: float(v) for k, v in d.items()} for d in csv.DictReader(lines)]


def at(tb, t, key):
    for a, b in zip(tb, tb[1:]):
        if a["t"] <= t <= b["t"]:
            w = (t - a["t"]) / (b["t"] - a["t"]) if b["t"] > a["t"] else 0.0
            return a[key] + w * (b[key] - a[key])
    return tb[-1][key]


def strouhal(tb, tMin, D=math.pi / 4):
    ups = []
    for a, b in zip(tb, tb[1:]):
        if a["t"] >= tMin and a["cl"] < 0 <= b["cl"]:
            ups.append(a["t"] + (b["t"] - a["t"]) * (-a["cl"]) / (b["cl"] - a["cl"]))
    return (len(ups) - 1) / (ups[-1] - ups[0]) * D if len(ups) >= 3 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--re", type=float, required=True)
    p.add_argument("--mach", type=float, default=0.2)
    p.add_argument("--ft", type=float, required=True)
    p.add_argument("--levels", required=True)
    p.add_argument("--far", type=float, default=20)
    p.add_argument("--trip", type=float, default=0)
    p.add_argument("--les", type=float, default=0)
    p.add_argument("--sgs", default="smag")
    p.add_argument("--rans", default="")
    p.add_argument("--beta", type=float, default=0, help="--ogrid-beta (0: the default)")
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--t", type=float, default=0, help="compare at this t (default: the end)")
    p.add_argument("--st-from", type=float, default=0, help="St over t >= this")
    p.add_argument("--limit", type=int, default=300)
    p.add_argument("--precision", default="fp32")
    p.add_argument("--out", default="/tmp/dns/ladder")
    p.add_argument("--compensated", action="store_true", help="both solvers on the two-word FP32 state")
    p.add_argument("--dns", default=DNS, help="the binary (a copy survives rebuilds)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"# Re {a.re} M {a.mach} far {a.far} D ft {a.ft} trip {a.trip} les {a.les} {a.sgs} nz {a.nz}")
    print("# n ny | t | cd_s cd_fv dcd order | cl_s cl_fv dcl | St_s St_fv")
    prev = None
    for lev in a.levels.split(","):
        n, ny = lev.split(":")
        grid = ["--n", n, "--ny", ny]
        s = rows(run(f"s{n}", a, grid, a.limit))
        f = rows(run(f"f{n}", a, grid + ["--solver", "fv", "--fv-precision", a.precision], a.limit))
        t = a.t if a.t > 0 else min(s[-1]["t"], f[-1]["t"])
        cs, cf, ls, lf = at(s, t, "cd"), at(f, t, "cd"), at(s, t, "cl"), at(f, t, "cl")
        if a.st_from:  # shedding: time means of cd, rms of cl over whole periods' window
            win = lambda tb: [r for r in tb if r["t"] >= a.st_from]
            mean = lambda tb, k: sum(r[k] for r in win(tb)) / len(win(tb))
            rms = lambda tb: math.sqrt(sum(r["cl"] ** 2 for r in win(tb)) / len(win(tb)))
            cs, cf, ls, lf = mean(s, "cd"), mean(f, "cd"), rms(s), rms(f)
        d = abs(cf - cs)
        order = math.log2(prev / d) if prev else float("nan")
        prev = d
        sts = strouhal(s, a.st_from) if a.st_from else float("nan")
        stf = strouhal(f, a.st_from) if a.st_from else float("nan")
        print(f"{n} {ny} | {t:.4g} | {cs:.6f} {cf:.6f} {cf - cs:+.3e} {order:.2f} | {ls:+.6f} {lf:+.6f} {lf - ls:+.3e} | "
              f"{sts:.5f} {stf:.5f}", flush=True)


if __name__ == "__main__":
    main()
