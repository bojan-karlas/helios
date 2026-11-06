#!/usr/bin/env bash
# Run the independent cell and tile/feature preparation branches concurrently
# against a shared Ray cluster. Each command writes to its own <logs_dir>/<run_id>
# directory and attaches without shutting down the shared Ray runtime.
#
# Usage:
#   scripts/slurm/run-parallel-branches.sh <dataset-or-cohort.yaml> <preprocessing-dir> <output-dir> [ray-address]

set -euo pipefail

DATASET="${1:?Supply dataset-or-cohort.yaml, preprocessing-dir, output-dir, and optional ray-address}"
PREPROCESSING_DIR="${2:?Supply preprocessing-dir}"
OUTPUT_DIR="${3:?Supply output-dir for logs}"
RAY_ADDR="${4:-}"

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

COMMON=(--dataset "$DATASET" --preprocessing-dir "$PREPROCESSING_DIR" --output-dir "$OUTPUT_DIR"
        --ray-address "$RAY_ADDR")

helios prep thumbnails "${COMMON[@]}" --run-tag thumbnails
helios prep tiles "${COMMON[@]}" --run-tag tiles

helios prep tile-features "${COMMON[@]}" --run-tag tile-features &
pid_features=$!

helios prep cells "${COMMON[@]}" --run-tag cells &
pid_cells=$!

wait "$pid_features"
wait "$pid_cells"

helios prep cell-features "${COMMON[@]}" --run-tag cell-features
