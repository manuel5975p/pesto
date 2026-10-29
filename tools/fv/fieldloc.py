#!/usr/bin/env python3
"""Where two OpenFOAM time directories differ (docs/fv.md, debugging aid):
the N cells with the largest |a - b| of one scalar field component, with an
approximate cell centre (mean of its face-point averages) from CASE's polyMesh.

usage: fieldloc.py CASE DIR_A DIR_B FIELD [COMPONENT] [N]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foamdiff import internal  # noqa: E402


def foam_list(path):
    txt = open(path).read()
    txt = re.sub(r"//[^\n]*", "", txt)
    i = txt.index("}") if "FoamFile" in txt else 0
    m = re.compile(r"\s*(\d+)\s*\(").search(txt, i)
    n = int(m.group(1))
    return txt[m.end():], n


def points(path):
    body, n = foam_list(path)
    return [tuple(float(x) for x in t.split()) for t in re.findall(r"\(([^()]*)\)", body)[:n]]


def faces(path):
    body, n = foam_list(path)
    out = re.findall(r"\d+\(([^()]*)\)", body)[:n]
    return [[int(x) for x in f.split()] for f in out]


def labels(path):
    body, n = foam_list(path)
    return [int(x) for x in body.replace(")", " ").split()[:n]]


def main():
    case, a, b, field = sys.argv[1:5]
    comp = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    N = int(sys.argv[6]) if len(sys.argv) > 6 else 10
    pm = os.path.join(case, "constant", "polyMesh")
    P = points(os.path.join(pm, "points"))
    F = faces(os.path.join(pm, "faces"))
    own = labels(os.path.join(pm, "owner"))
    nei = labels(os.path.join(pm, "neighbour"))
    nC = max(own) + 1
    acc = [[0.0, 0.0, 0.0, 0] for _ in range(nC)]
    for fi, f in enumerate(F):
        c = [sum(P[i][q] for i in f) / len(f) for q in range(3)]
        for cell in [own[fi]] + ([nei[fi]] if fi < len(nei) else []):
            for q in range(3):
                acc[cell][q] += c[q]
            acc[cell][3] += 1
    C = [(x / k, y / k, z / k) for x, y, z, k in acc]
    _, va = internal(os.path.join(a, field))
    _, vb = internal(os.path.join(b, field))
    d = sorted(((abs(va[i][comp] - vb[i][comp]), i) for i in range(len(va))), reverse=True)
    for diff, i in d[:N]:
        print(f"cell {i:7d} C ({C[i][0]:+.4f} {C[i][1]:+.5f} {C[i][2]:+.4f})  a {va[i][comp]:+.10e}  "
              f"b {vb[i][comp]:+.10e}  |d| {diff:.3e}")


if __name__ == "__main__":
    main()
