#!/usr/bin/env bash
# Run the independent cell and tile/feature preparation branches concurrently
# against a shared Ray cluster. Each command writes to its own <log_dir>/<run_id>
# directory and attaches without shutting down the shared Ray runtime.
#
# Usage:
#   scripts/slurm/run-parallel-branches.sh <dataset-or-cohort.yaml> [ray-address]

set -euo pipefail

DATASET="${1:?Usage: $0 <dataset-or-cohort.yaml> [ray-address]}"
RAY_ADDR="${2:-}"

started_local_ray=0
cleanup() {
    if [ "$started_local_ray" -eq 1 ]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [ -z "$RAY_ADDR" ]; then
    port="${RAY_PORT:-6379}"
    ray start --head --port="$port" --disable-usage-stats >/dev/null
    RAY_ADDR="127.0.0.1:${port}"
    started_local_ray=1
fi

helios prep thumbnails --dataset "$DATASET" --ray-address "$RAY_ADDR" --run-tag thumbnails
helios prep tiles --dataset "$DATASET" --ray-address "$RAY_ADDR" --run-tag tiles

helios prep tile-features --dataset "$DATASET" --ray-address "$RAY_ADDR" --run-tag tile-features &
pid_features=$!

helios prep cells --dataset "$DATASET" --ray-address "$RAY_ADDR" --run-tag cells &
pid_cells=$!

wait "$pid_features"
wait "$pid_cells"

helios prep cell-features --dataset "$DATASET" --ray-address "$RAY_ADDR" --run-tag cell-features
