#!/usr/bin/env python3
"""Gate BL-a (docs/fv.md): skin friction and velocity profiles of two
OpenFOAM-layout solutions of the flat plate (tools/fv/make_ras_cases.py
`plate`) on the IDENTICAL mesh -- simpleFoam's and dns's time directories.

cf(x) = 2 tau_w / U^2 from each solution's own fields with the same formula
(the wall-face gradient OpenFOAM's wallShearStress takes on a flat wall:
tau_w = (nu + nut_w) U_P,x / d_P, d_P the cell centre's wall distance), so
the comparison is solution against solution. Profiles u(y)/U at x = 0.5,
1.0, 1.5 (the cell column nearest each station).

usage: plate_compare.py CASE DIR_A DIR_B [U_INF NU]
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fieldloc import faces, labels, points  # noqa: E402
from foamdiff import internal  # noqa: E402


def patch_values(path, patch):
    txt = open(path).read()
    i = txt.index("boundaryField")
    m = re.compile(r"\b" + re.escape(patch) + r"\s*\{").search(txt, i)
    j = txt.index("value", m.end())
    m2 = re.compile(r"value\s+nonuniform\s+List<scalar>\s*(\d+)\s*\(").match(txt, j)
    if not m2:
        m3 = re.compile(r"value\s+uniform\s+([-+0-9.eE]+)").match(txt, j)
        return float(m3.group(1))
    n = int(m2.group(1))
    return [float(t) for t in txt[m2.end():].split(None, n)[:n]]


def boundary(path, name):
    txt = open(path).read()
    m = re.compile(r"\b" + re.escape(name) + r"\s*\{([^}]*)\}").search(txt)
    nf = int(re.search(r"nFaces\s+(\d+)", m.group(1)).group(1))
    sf = int(re.search(r"startFace\s+(\d+)", m.group(1)).group(1))
    return sf, nf


def main():
    case, A, B = sys.argv[1:4]
    Uinf = float(sys.argv[4]) if len(sys.argv) > 4 else 69.4
    nu = float(sys.argv[5]) if len(sys.argv) > 5 else 1.388e-05
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
    sf, nf = boundary(os.path.join(pm, "boundary"), "bottomWall")
    out = {}
    for tag, D in (("A", A), ("B", B)):
        _, U = internal(os.path.join(D, "U"))
        nutw = patch_values(os.path.join(D, "nut"), "bottomWall")
        if isinstance(nutw, float):
            nutw = [nutw] * nf
        cf = []
        for i in range(nf):
            f = F[sf + i]
            fc = [sum(P[j][q] for j in f) / len(f) for q in range(3)]
            c = own[sf + i]
            d = C[c][1] - fc[1]
            cf.append((fc[0], 2 * (nu + nutw[i]) * U[c][0] / d / Uinf ** 2))
            if tag == "A":
                yp = d * ((nu + nutw[i]) * U[c][0] / d) ** 0.5 / nu
                out.setdefault("yplus", []).append(yp)
        out[tag] = (cf, U)
    yp = out["yplus"]
    print(f"first-cell y+ on the plate (A): mean {sum(yp) / len(yp):.3f}, max {max(yp):.3f}")
    cfa, cfb = out["A"][0], out["B"][0]
    worst, at = 0.0, 0.0
    for (x, a), (_, b) in zip(cfa, cfb):
        if x < 0.05:
            continue
        r = abs(a - b) / abs(a)
        if r > worst:
            worst, at = r, x
    print(f"cf: {len(cfa)} wall faces, x >= 0.05: max |cf_B - cf_A| / cf_A = {worst:.3e} at x = {at:.4f}")
    for xs in (0.5, 1.0, 1.5):
        ia = min(range(nf), key=lambda i: abs(cfa[i][0] - xs))
        print(f"  x {cfa[ia][0]:.4f}: cf A {cfa[ia][1]:.6e}  B {cfb[ia][1]:.6e}")
    UA, UB = out["A"][1], out["B"][1]
    for xs in (0.5, 1.0, 1.5):
        col = [c for c in range(nC) if abs(C[c][0] - xs) < 0.02 and C[c][0] > 0]
        x0 = min({round(C[c][0], 9) for c in col}, key=lambda v: abs(v - xs))
        col = sorted((c for c in col if abs(C[c][0] - x0) < 1e-9), key=lambda c: C[c][1])
        d = max(abs(UA[c][0] - UB[c][0]) / Uinf for c in col)
        print(f"profile x {x0:.4f}: {len(col)} cells, max |u_B - u_A| / U = {d:.3e}")


if __name__ == "__main__":
    main()
