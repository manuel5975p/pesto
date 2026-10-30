#!/usr/bin/env python3
"""dns flame vs OpenFOAM v2606 reactingFoam on the IDENTICAL mesh
(docs/chem.md "Against OpenFOAM reactingFoam"). One case spec
(foamflame/cases.py) -> the dns argv, the reactingFoam case directories, the
simjob job specs, and the comparison table.

  foam_flame.py list
  foam_flame.py setup CASE              write runs/flames/foam/CASE/ (cases, specs)
  foam_flame.py run CASE [--dns] [--foam] [--np K] [--only RUN ...]
                                        dns --dry-run, then simjob (GPU / CPU)
  foam_flame.py compare CASE            table + runs/flames/foam/CASE/compare.json

Every run goes through the job server (docs/jobs.md): dns as one GPU job
(one step per rung), OpenFOAM as one CPU job (blockMesh per case, then the
reactingFoam runs side by side, K at a time, each serial).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from foamflame import cases, observe  # noqa: E402
from foamflame.model import Oracle, si_map, si_map_diff  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DNS = os.path.join(REPO, "build", "dns")
SIMJOB = os.path.join(REPO, "build", "simjob")
# the OpenFOAM v2606 install (its etc/bashrc is sourced for the CPU job)
FOAM = os.environ.get("DNS_FOAM_V2606", "/opt/OpenFOAM/OpenFOAM-v2606")
FOAM_BIN = os.path.join(FOAM, "platforms", "linux64GccDPInt32Opt", "bin")
ROOT = os.path.join(REPO, "runs", "flames", "foam")


def oracle_for(md):
    """The P0 oracle, cached (8 s to recompute)."""
    path = os.path.join(ROOT, f"oracle_s{md['sigma']:g}_b{md['beta']:g}.npz")
    if os.path.exists(path):
        d = np.load(path)
        o = Oracle.__new__(Oracle)
        o.sigma, o.beta = float(d["sigma"]), float(d["beta"])
        o.theta = o.beta * o.sigma ** 2 / (o.sigma - 1)
        o.Lambda, o.dth = float(d["Lambda"]), float(d["dth"])
        o.x, o.T, o.G = d["x"], d["T"], d["G"]
        return o
    o = Oracle(md["sigma"], md["beta"])
    os.makedirs(ROOT, exist_ok=True)
    np.savez(path, sigma=o.sigma, beta=o.beta, Lambda=o.Lambda, dth=o.dth,
             x=o.x, T=o.T, G=o.G)
    return o


def context(name):
    spec = cases.CASES[name]
    md = spec["model"]
    if spec["geometry"] in cases.DIFF_GEOMS:     # model C1: no premixed oracle
        si = si_map_diff(md, gravity=spec["burner"].get("gravity", 0.0),
                         **spec["si"])
        return spec, None, si, os.path.join(ROOT, name)
    o = oracle_for(md)
    si = si_map(md["mach"], md["sigma"], md["beta"], md["Pr"], o.Lambda,
                **spec["si"])
    return spec, o, si, os.path.join(ROOT, name)


def dns_runs(spec, root):
    return [(f"dns_c{N:g}", N, cases.dns_argv(
        spec, N, os.path.join(root, f"dns_c{N:g}", "flame")))
        for N in spec["ladder"]]


def cmd_setup(name):
    spec, o, si, root = context(name)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "si.json"), "w") as f:
        json.dump(si, f, indent=1)
    for run, N, st in cases.foam_runs(spec):
        case = os.path.join(root, run)
        cases.write_foam(spec, N, st, o, si, case)
        with open(os.path.join(case, "settings.json"), "w") as f:
            json.dump({"N": N, **st}, f, indent=1)
        print(f"foam case {case}")
    for run, N, argv in dns_runs(spec, root):
        os.makedirs(os.path.join(root, run), exist_ok=True)
        with open(os.path.join(root, run, "argv.txt"), "w") as f:
            f.write(shlex.join([DNS] + argv) + "\n")
        print(shlex.join([DNS] + argv))
    if o is None:
        print(f"SI: U {si['U']:.6g} m/s, l/U {si['tUnit']:.6g} s, mu {si['mu']:.6g} "
              f"Pa s, A {si['A']:.6g} m^3/kmol/s, Ta {si['Ta']:.6g} K, q "
              f"{si['q']:.6g} J/kg, g {si['g']:.6g} m/s^2")
        return
    print(f"oracle Lambda {o.Lambda:.10g} dth/dD {o.dth:.6f}; "
          f"SI: S_L {si['SL']:.6g} m/s, tau {si['tau']:.6g} s, A {si['A']:.6g} /s")


def submit_spec(job, path, foam_env):
    """Submit and wait; the simjob answer (job id, wall) lands in PATH.log,
    and the job record's per-step walls in PATH.steps.json."""
    with open(path, "w") as f:
        json.dump(job, f, indent=1)
    cmd = f"{shlex.quote(SIMJOB)} submit --spec {shlex.quote(path)} --wait"
    if foam_env:
        cmd = (f"source {FOAM}/etc/bashrc >/dev/null 2>&1; {cmd} --inherit-env")
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    out = p.stdout + p.stderr
    print(out, end="", flush=True)
    base = os.path.splitext(path)[0]
    with open(base + ".log", "w") as f:
        f.write(out)
    for w in out.split():
        if w.startswith("dir="):
            rec = os.path.join(w[4:], "job.json")
            if os.path.exists(rec):
                with open(rec) as f, open(base + ".record.json", "w") as g:
                    g.write(f.read())
    return p.returncode


def cmd_run(name, do_dns, do_foam, np_, only):
    spec, o, si, root = context(name)
    rc = 0
    # a partial run keeps its own job files (the full ladder's stay)
    tag = "" if not only else "_" + "_".join(sorted(only))
    if do_dns:
        runs = [r for r in dns_runs(spec, root) if not only or r[0] in only]
        for run, _, argv in runs:   # dry-run every rung first
            p = subprocess.run([DNS] + argv + ["--dry-run"], capture_output=True,
                               text=True)
            if p.returncode:
                sys.exit(f"dry-run {run}: rc {p.returncode}\n{p.stderr}")
            with open(os.path.join(root, run, "dryrun.txt"), "w") as f:
                f.write(p.stdout)
        if runs:
            job = {"label": f"foamflame {name} dns", "resource": "gpu",
                   "max_runtime_s": 900, "max_mem_mb": 4096, "cwd": root,
                   "steps": [{"exe": DNS, "args": a} for _, _, a in runs]}
            rc |= submit_spec(job, os.path.join(root, f"job_dns{tag}.json"), False)
    if do_foam:
        runs = [r for r in cases.foam_runs(spec) if not only or r[0] in only]
        if runs:
            dirs = [os.path.join(root, r[0]) for r in runs]
            solver = ("rhoReactingBuoyantFoam" if spec["geometry"] == "candle2d"
                      else "reactingFoam")
            for r in runs:          # a restarted check starts from its base's field
                if "restart" in r[2]:
                    restart_link(spec, si, root, r)
            script = os.path.join(root, "run_foam.sh")
            if spec["geometry"] in cases.DIFF_GEOMS:    # one script per job
                script = os.path.join(root, f"run_foam{tag}.sh")
            with open(script, "w") as f:
                if spec["geometry"] == "1d":
                    f.write(FOAM_SCRIPT.format(np=np_, dirs="\n".join(dirs)))
                elif spec["geometry"] in cases.DIFF_GEOMS:
                    f.write(FOAM_PAR_SCRIPT_DIFF.format(
                        np=spec["foam"].get("np", np_), dirs="\n".join(dirs),
                        solver=solver, nrun=len(dirs)))
                else:
                    f.write(FOAM_PAR_SCRIPT.format(
                        np=spec["foam"].get("np", np_), dirs="\n".join(dirs)))
            steps = [{"exe": os.path.join(FOAM_BIN, "blockMesh"),
                      "args": ["-case", d]} for d in dirs]
            steps.append({"exe": "/bin/bash", "args": [script],
                          "verify": [{"path": os.path.join(FOAM_BIN, solver)}]})
            job = {"label": f"foamflame {name} foam", "resource": "cpu",
                   "max_runtime_s": 1200, "max_mem_mb": 8192, "cwd": root,
                   "steps": steps}
            rc |= submit_spec(job, os.path.join(root, f"job_foam{tag}.json"), True)
    sys.exit(rc)


# reactingFoam runs side by side, K at a time (each serial: a 1D row of
# cells does not decompose usefully); per-run wall time in wall.txt.
FOAM_SCRIPT = """#!/bin/bash
set -u
run() {{
    cd "$1" || return 1
    rm -rf dynamicCode
    local t0=$(date +%s.%N)
    reactingFoam > log.reactingFoam 2>&1
    local rc=$?
    echo "$(echo "$(date +%s.%N) - $t0" | bc) rc=$rc" > wall.txt
    echo "$1 $(cat wall.txt)"
    return $rc
}}
export -f run
xargs -P {np} -I{{}} bash -c 'run "$@"' _ {{}} <<'EOF'
{dirs}
EOF
"""


# 2D cases: one case after another, each decomposed (scotch) over K ranks;
# fields reconstructed at the end; wall of the solver alone in wall.txt.
FOAM_PAR_SCRIPT = """#!/bin/bash
set -u
while read -r d; do
    [ -z "$d" ] && continue
    cd "$d" || exit 1
    rm -rf dynamicCode processor*
    decomposePar -force > log.decomposePar 2>&1 || exit 1
    t0=$(date +%s.%N)
    mpirun -np {np} --bind-to none reactingFoam -parallel > log.reactingFoam 2>&1
    rc=$?
    echo "$(echo "$(date +%s.%N) - $t0" | bc) rc=$rc np={np}" > wall.txt
    echo "$d $(cat wall.txt)"
    [ $rc -eq 0 ] || exit $rc
    reconstructPar > log.reconstructPar 2>&1 || exit 1
    rm -rf processor*
done <<'EOF'
{dirs}
EOF
"""


# The two-reactant cases: as FOAM_PAR_SCRIPT, the solver named, a restarted
# check (file `restart_time` = its start time directory, copied in by
# `restart_link`) decomposes that time only, and the cases of one job run
# CONCURRENTLY, K ranks each (wall.txt per case; `concurrent` in the job).
FOAM_PAR_SCRIPT_DIFF = """#!/bin/bash
set -u
one() {{
    cd "$1" || return 1
    rm -rf dynamicCode processor*
    local opts=""
    [ -f restart_time ] && opts="-time $(cat restart_time)"
    decomposePar -force $opts > log.decomposePar 2>&1 || return 1
    local t0=$(date +%s.%N)
    mpirun -np {np} --bind-to none {solver} -parallel > log.{solver} 2>&1
    local rc=$?
    echo "$(echo "$(date +%s.%N) - $t0" | bc) rc=$rc np={np} concurrent={nrun}" > wall.txt
    echo "$1 $(cat wall.txt)"
    [ $rc -eq 0 ] || return $rc
    reconstructPar > log.reconstructPar 2>&1 || return 1
    rm -rf processor*
}}
pids=()
while read -r d; do
    [ -z "$d" ] && continue
    one "$d" &
    pids+=($!)
done <<'EOF'
{dirs}
EOF
rc=0
for p in "${{pids[@]}}"; do wait $p || rc=1; done
exit $rc
"""


def restart_link(spec, si, root, run):
    """Copy the base run's field at the check's start time into the check's
    case (OpenFOAM's time name: 12 significant digits, timePrecision 12)."""
    import shutil
    name, N, st = run
    base = os.path.join(root, f"foam_{st['restart_base']}_c{N:g}")
    tname = f"{st['restart'] * si['tUnit']:.12g}"
    src = os.path.join(base, tname)
    if not os.path.isdir(src):
        sys.exit(f"restart {name}: no base field {src}")
    dst = os.path.join(root, name, tname)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("uniform"))
    shutil.rmtree(os.path.join(root, name, "0"), ignore_errors=True)
    with open(os.path.join(root, name, "restart_time"), "w") as f:
        f.write(tname + "\n")


def cmd_compare(name):
    spec, o, si, root = context(name)
    if spec["geometry"] in cases.DIFF_GEOMS:
        from foamflame import observejet
        res = observejet.compare(spec, si, root)
        with open(os.path.join(root, "compare.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(observejet.table(res))
        return
    if spec["geometry"] != "1d":
        from foamflame import observe2d
        if spec["geometry"] == "bunsen2d":
            res = observe2d.compare_bunsen(spec, o, si, root)
            txt = observe2d.table_bunsen(res)
        else:
            ex = [(n, os.path.join(root, d), N, ny, sg, dth)
                  for n, d, N, ny, sg, dth in spec.get("extra_dns", [])]
            res = observe2d.compare_wrinkle(spec, o, si, root, ex)
            txt = observe2d.table_wrinkle(res)
        with open(os.path.join(root, "compare.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(txt)
        return
    res = observe.compare(spec, o, si, root,
                          [r[0] for r in dns_runs(spec, root)],
                          cases.foam_runs(spec))
    with open(os.path.join(root, "compare.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(observe.table(res))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["list", "setup", "run", "compare"])
    ap.add_argument("case", nargs="?")
    ap.add_argument("--dns", action="store_true")
    ap.add_argument("--foam", action="store_true")
    ap.add_argument("--np", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    if a.cmd == "list":
        for k, v in cases.CASES.items():
            print(f"{k}: {v['title']}")
        return
    if a.case not in cases.CASES:
        sys.exit(f"unknown case {a.case!r} (foam_flame.py list)")
    if a.cmd == "setup":
        cmd_setup(a.case)
    elif a.cmd == "run":
        both = not (a.dns or a.foam)
        cmd_run(a.case, a.dns or both, a.foam or both, a.np, set(a.only))
    else:
        cmd_compare(a.case)


if __name__ == "__main__":
    main()
