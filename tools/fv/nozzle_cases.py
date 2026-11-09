#!/usr/bin/env python3
"""The foam case variants of the nozzle (docs/nozzle.md): nozzle_cases.py ROOT NAME... writes ROOT/NAME and ROOT/NAME.json,
the job spec for `simjob submit --spec ROOT/NAME.json --inherit-env --wait`. Source OpenFOAM's etc/bashrc first.

  A, B, W      tests/data/fv/nozzle_{A,B,W}: 80x16, upwind (T1): supersonic exit, normal shock, 20 deg y = r wedge
  G            tests/data/fv/nozzle_G: B at gamma 1.2
  QA, QV, RT   tests/data/fv/nozzle_{QA,QV,RT}: one row (nc 200 upwind, nc 200 vanLeer, react twin 400 upwind)
"""
import json
import os
import sys

import nozzle_mesh as nm

FOAM = "/opt/OpenFOAM/OpenFOAM-v2606/platforms/linux64GccDPInt32Opt/bin"


def args(**kw):
    """Parsed defaults of nozzle_mesh with kw over them."""
    a = nm.parser().parse_args(["--nx", "1", "--ny", "1", "--out", "x"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# (contour args, runtime cap s, np)
CASES = {
    "A": (dict(nx=80, ny=16, end=2.4, write=0.3, co=0.2, exit="extrapolate"), 280, 1),
    "B": (dict(nx=80, ny=16, end=2.4, write=0.3, co=0.2, exit="fixed", pb=0.6), 280, 1),
    "W": (dict(nx=80, ny=16, mode="wedge", dphi=20.0, end=2.4, write=0.3, co=0.8, exit="extrapolate"), 280, 1),
    "G": (dict(nx=80, ny=16, end=2.4, write=0.3, co=0.2, exit="fixed", pb=0.6, gamma=1.2), 280, 1),
    # masterplan_combnozzle P2: one row (the quasi-1D FV scheme). QA: nc, G2.8's T1; QV: QA at vanLeer (recorded);
    # RT: the react contour's inert twin (lc 0, eps_c = eps_e = 6.25), G2.9's T1.
    "QA": (dict(nx=200, ny=1, end=2.4, write=0.3, co=0.2, exit="extrapolate"), 280, 1),
    "QV": (dict(nx=200, ny=1, end=2.4, write=0.3, co=0.2, exit="extrapolate", recon="vanLeer"), 280, 1),
    "RT": (dict(contour="react", lc=0.0, nx=400, ny=1, end=2.4, write=0.3, co=0.2, exit="extrapolate"), 280, 1),
}


def steps(case, np_, foam="rhoCentralFoam"):
    """Serial or decomposePar + mpirun + reconstructPar."""
    if np_ <= 1:
        return [{"exe": f"{FOAM}/{foam}", "args": ["-case", case]}]
    return [{"exe": f"{FOAM}/decomposePar", "args": ["-case", case, "-force"]},
            {"exe": "/usr/bin/mpirun", "args": ["-np", str(np_), "--bind-to", "core", "--map-by", "core", f"{FOAM}/{foam}", "-case", case, "-parallel"]},
            {"exe": f"{FOAM}/reconstructPar", "args": ["-case", case]}]


def checkStep(case):
    return {"exe": f"{FOAM}/checkMesh", "args": ["-case", case]}


def writeSpec(path, case, label, cap, np_, mem=12000, check=False, cwd=None):
    sp = {"label": label, "resource": "cpu", "max_runtime_s": cap, "max_mem_mb": mem, "cwd": cwd or case,
          "steps": ([checkStep(case)] if check else []) + steps(case, np_)}
    with open(path, "w") as f:
        json.dump(sp, f, indent=1)


def main():
    root = os.path.abspath(sys.argv[1])
    for name in sys.argv[2:]:
        kw, cap, np_ = CASES[name]
        out = os.path.join(root, name)
        a = args(out=out, **kw)
        astar, col, X, Y = nm.generate(a)
        writeSpec(out + ".json", out, f"nozzle-{name}", cap, np_, check=True)
        print(f"{name}: A* {astar:.12g}, column {col}, spec {out}.json")


if __name__ == "__main__":
    main()
