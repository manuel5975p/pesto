#!/usr/bin/env bash
# Chain one long `dns flame` run over successive simjobs (docs/chem.md
# "Checkpoint and restart"; masterplan real_bunsen S2):
#
#   tools/chem/chain.sh [--resume] PREFIX N -- flame ARGS...
#
# Job 1 runs `build/dns flame ARGS --out PREFIX --wall-limit W`; jobs 2..N add
# `--restart PREFIX` (with --resume, job 1 does too: continue a chain that
# stopped). ARGS must carry --checkpoint-dt (the wall limit stops only at a
# checkpoint) and the run's --tt. Every flame route checkpoints: the structured
# solvers and the FV ones (--solver fv, --polymesh DIR --bc FILE: keep the mesh
# and the --bc file unchanged over the chain, a restart compares their
# content). Each job goes through
# `build/simjob submit --gpu -t T -m M --wait`. Stops on the first rc != 0 or
# when PREFIX.json says "complete": true. Run it through run_in_background.
#
# Environment: CHAIN_T (simjob -t, default 1800), CHAIN_M (simjob -m, 16384),
# CHAIN_WALL (--wall-limit, 1700), CHAIN_DNS (the dns binary, default
# build/dns: point it at a COPY to keep building while a chain runs -- a
# restart rejects another binary). dns stops at the first checkpoint after
# which the next one would end past the limit (the last interval's wall time
# ahead), so pick --checkpoint-dt with an interval well under T - W.
#
# Exit: 0 complete; 1 usage; 2 N jobs ran and the run is still incomplete;
# otherwise the failing job's simjob exit code (docs/jobs.md).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DNS=${CHAIN_DNS:-$ROOT/build/dns}
SIMJOB=$ROOT/build/simjob
T=${CHAIN_T:-1800}
M=${CHAIN_M:-16384}
W=${CHAIN_WALL:-1700}

usage() {
    echo "usage: tools/chem/chain.sh [--resume] PREFIX N -- flame ARGS..." >&2
    exit 1
}
resume=0
if [ "${1:-}" = "--resume" ]; then resume=1; shift; fi
[ $# -ge 4 ] || usage
PREFIX=$1
N=$2
[ "$3" = "--" ] || usage
shift 3
case "$N" in '' | *[!0-9]*) usage ;; esac
[ "$N" -ge 1 ] || usage
case " $* " in *" --checkpoint-dt "*) ;; *)
    echo "chain.sh: ARGS need --checkpoint-dt (a --wall-limit stop is a checkpoint stop)" >&2
    exit 1 ;;
esac

complete() { grep -q '"complete":true' "$PREFIX.json" 2>/dev/null; }

for ((job = 1; job <= N; ++job)); do
    restart=()
    if [ $job -gt 1 ] || [ $resume = 1 ]; then restart=(--restart "$PREFIX"); fi
    echo "chain.sh: job $job/$N: dns $* --out $PREFIX --wall-limit $W ${restart[*]}"
    "$SIMJOB" submit --gpu -t "$T" -m "$M" --label "chain $(basename "$PREFIX") $job" \
        --wait -- "$DNS" "$@" --out "$PREFIX" --wall-limit "$W" "${restart[@]}"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "chain.sh: job $job failed (simjob rc $rc); the last checkpoint is" \
             "$PREFIX""_ckpt.bin (continue with --resume)" >&2
        exit $rc
    fi
    if complete; then
        echo "chain.sh: complete after $job job(s)"
        exit 0
    fi
done
echo "chain.sh: $N job(s) ran, the run is incomplete (continue with --resume)" >&2
exit 2
