#!/usr/bin/env bash
# FV RANS channel vs the FP64 oracle (docs/fvcomp.md "RANS"): one simjob per rung,
# three levels at fixed stretch, prints the RESULT lines.
# Usage: rans_ladder.sh TEST_BINARY RETAU BETA "NY NY NY" "MODEL MODEL"
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$1; RETAU=$2; BETA=$3; LEVELS=$4; MODELS=$5
SIMJOB=${SIMJOB:-$ROOT/build/simjob}
for model in $MODELS; do
    for ny in $LEVELS; do
        job=$("$SIMJOB" submit --gpu -t 295 -m 8192 --wait --label "rn-$model-$RETAU-$ny" -- "$BIN" channel "$model" "$RETAU" "$ny" "$BETA" 2>&1 | tail -1)
        id=$(echo "$job" | sed -n 's/^job \([0-9]*\).*/\1/p')
        grep -E "^(RESULT|FATAL)" "$HOME/.local/state/simjobs/$id/output.log" || echo "no result: $job"
    done
done
