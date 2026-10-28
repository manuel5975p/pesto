# simjobd / simjob — the job server

One machine, one GPU, several agents and people: every simulation run goes
through one queue. `simjobd` listens on a unix socket, admits jobs by a fixed
rule, runs each in its own transient systemd scope with a memory cap and a hard
deadline, and records the outcome. `simjob` is the client. Both build with the
tree (`build/simjobd`, `build/simjob`) and link nothing of the solver.

## Guarantees

- **Admission.** At most one `gpu` job and one `cpu` job run at a time; the two
  may overlap. FIFO per resource. A `benchmark` job runs ALONE: it starts only
  when nothing runs and nothing submitted before it still waits, and nothing
  submitted after it starts until it ends.
- **Limits are hard and capped.** `max_runtime_s` ≤ 1800 (30 min) for the whole
  job, all steps; `max_mem_mb` ≤ 16384 (16 GiB) for the whole job's cgroup, swap
  0. Over a cap is a refusal at submit, never a clamp. At the deadline every
  process in the job gets SIGTERM, then SIGKILL 10 s later (`cgroup.kill`).
  VRAM is NOT capped — the cgroup sees host memory only.
- **No stale binaries.** Every step names its executable by absolute path AND
  sha256 (the client hashes at submit time). The server re-hashes at submit and
  the runner re-hashes at launch, then `fexecve`s the very fd it hashed, so a
  rebuild between submit and launch is `rejected`, never run. `verify` pins
  further files a step depends on (e.g. the solver `mpirun` starts).
- **Same user only.** Socket mode 0600 plus an `SO_PEERCRED` uid check.

## Running the server

```
build/simjobd [--socket PATH] [--state DIR] [--gpu-lock FILE]
```

Socket default `$SIMJOBS_SOCKET`, else `$XDG_RUNTIME_DIR/simjobs.sock`; state
default `$XDG_STATE_HOME/simjobs` (`~/.local/state/simjobs`). `--gpu-lock FILE`
makes the server also hold `flock(FILE)` while a gpu or benchmark job runs, so
GPU users outside the server that take the same flock serialize with it.
Needs a systemd user session (`systemd-run --user --scope`). SIGINT/SIGTERM:
running jobs are terminated (`cancelled`, "simjobd shutdown"), queued jobs stay
queued on disk and are re-admitted by the next start; a job that was running
when the server died is marked `lost`.

## Client

```
simjob submit (--gpu|--cpu) -t SEC -m MB [--bench] [--label L] [--cwd DIR]
              [--env K=V]... [--inherit-env] [--verify FILE]... [--wait] -- EXE [ARG...]
simjob submit --spec FILE.json [--inherit-env] [--wait]
simjob status ID | wait ID | cancel ID | log ID | list | ping
```

`submit` prints the job id; with `--wait` (or `simjob wait ID`) it blocks and
exits 0 done, the failing step's rc, 124 timeout, 137 oom, 130 cancelled, 3
rejected, 75 lost. `log` prints the job's combined stdout/stderr.
`--inherit-env` sends the whole shell environment (OpenFOAM needs its sourced
`etc/bashrc`). EXE without a slash is looked up in `PATH`.

```
simjob submit --gpu -t 300 -m 8192 --wait -- build/dns flow --body cylinder --re 100 --out /tmp/r
```

## Multi-step jobs

A job is a list of steps run in order in one scope; the first failing step
ends the job (the limits cover the whole list). OpenFOAM is the usual case:

```json
{
  "label": "cyl40 simpleFoam",
  "resource": "cpu",
  "max_runtime_s": 1200,
  "max_mem_mb": 12000,
  "cwd": "/home/me/runs/cyl40/foam",
  "steps": [
    {"exe": "blockMesh"},
    {"exe": "decomposePar", "args": ["-force"]},
    {"exe": "/usr/bin/mpirun", "args": ["-np", "12", "simpleFoam", "-parallel"],
     "verify": [{"path": "/opt/OpenFOAM/OpenFOAM-v2606/platforms/linux64GccDPInt32Opt/bin/simpleFoam"}]},
    {"exe": "reconstructPar", "args": ["-latestTime"]}
  ]
}
```

`simjob submit --spec job.json --inherit-env --wait` fills the missing `sha256`
fields (and `cwd`) before sending. Wire schema of `job` (strict: unknown keys
are errors): `label` string, `resource` `"gpu"|"cpu"`, `benchmark` bool,
`max_runtime_s` int, `max_mem_mb` int, `cwd` absolute path, `env` object of
strings (overrides on the server's environment), `steps` 1–64 of `{exe`
absolute path, `sha256` 64 hex, `args` string array, `verify` array of
`{path, sha256}}`.

## Wire protocol

One JSON document per line on the socket: the client writes one request line,
the server answers one line and closes. `wait` is answered when the job ends.

| request | answer |
|---|---|
| `{"op":"submit","job":{...}}` | `{"ok":true,"id":N}` or `{"ok":false,"error":"..."}` |
| `{"op":"status","id":N}` / `{"op":"wait","id":N}` | `{"ok":true,"job":{record}}` |
| `{"op":"cancel","id":N}` | `{"ok":true}` (queued: dropped; running: SIGTERM, then SIGKILL) |
| `{"op":"list"}` | `{"ok":true,"jobs":[...]}` (active + last 50 finished) |
| `{"op":"ping"}` | `{"ok":true,"pid":P}` |

Job states: `queued running done failed timeout oom cancelled rejected lost`.
Per job, `<state>/<id>/` holds `job.json` (the record), `steps.json` (per-step
state, rc, wall time), `output.log` (all step output with `=== step` markers)
and `runner.log` (systemd-run's own errors).

## Tests

`jobs` (sha256 vectors, strict JSON, spec caps, the admission rule) and
`jobs_live` (the real binaries on a private socket: every outcome, both
refusal paths, a rebuild while queued, admission timing, cancel, CLI exit
codes; skips without a systemd user session).
