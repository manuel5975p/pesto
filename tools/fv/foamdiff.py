#!/usr/bin/env python3
"""Diffs two OpenFOAM ASCII time directories cell by cell (docs/fv.md, IF1).

usage: foamdiff.py DIR_A DIR_B [field ...]      (default fields: U p phi)

Prints per field: max |a - b|, the scale (max |a|; p: max a - min a, the
pressure is defined up to its level), and their ratio. Both directories must
hold fields in the same (OpenFOAM) cell/face order -- dns writes them so.
"""
import os
import re
import sys


def internal(path):
    txt = open(path).read()
    m = re.search(r"internalField\s+(uniform|nonuniform)", txt)
    if not m:
        raise SystemExit(f"{path}: no internalField")
    p = m.end()
    if m.group(1) == "uniform":
        v = re.match(r"\s*(\([^)]*\)|[-+0-9.eE]+)", txt[p:]).group(1)
        return None, v
    m2 = re.compile(r"\s*List<(\w+)>\s*(\d+)\s*\(").match(txt, p)
    kind, n = m2.group(1), int(m2.group(2))
    p = m2.end()
    vals = []
    if kind == "scalar":
        toks = txt[p:].split(None, n)
        vals = [[float(t)] for t in toks[:n]]
    else:
        body = txt[p:]
        for i, mm in enumerate(re.finditer(r"\(([^()]*)\)", body)):
            if i == n:
                break
            vals.append([float(x) for x in mm.group(1).split()])
    if len(vals) != n:
        raise SystemExit(f"{path}: {len(vals)} values, header says {n}")
    return kind, vals


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    a, b = sys.argv[1], sys.argv[2]
    fields = sys.argv[3:] or ["U", "p", "phi"]
    for f in fields:
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{f}: missing in one directory")
            continue
        ka, va = internal(pa)
        kb, vb = internal(pb)
        if ka is None or kb is None or len(va) != len(vb):
            print(f"{f}: not comparable (uniform or different length)")
            continue
        nc = len(va[0])
        dmax = max(abs(x[q] - y[q]) for x, y in zip(va, vb) for q in range(nc))
        if f == "p":
            flat = [x[0] for x in va]
            scale = max(flat) - min(flat)
        else:
            scale = max(abs(x[q]) for x in va for q in range(nc))
        print(f"{f:4s} n {len(va):8d}  max|diff| {dmax:.3e}  scale {scale:.3e}  rel {dmax / scale if scale else 0:.3e}")


if __name__ == "__main__":
    main()
