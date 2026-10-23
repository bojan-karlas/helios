#!/usr/bin/env bash
# Dynamic SLURM job-array worker pool that joins an existing Ray head node as
# resources become available on the cluster.
#
# Start a Ray head on your coordinator node first:
#   ray start --head --port=6379 --disable-usage-stats
# Then launch elastic workers via SLURM array:
#   RAY_HEAD_ADDRESS="<head-ip>:6379" sbatch --array=0-7 scripts/slurm/ray-array-workers.sbatch
# And run HELIOS against the shared Ray head:
#   helios prep tiles --dataset /data/COHORT1 --ray-address "<head-ip>:6379"

#SBATCH --job-name=helios-ray-worker
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00

set -euo pipefail

if [ -z "${RAY_HEAD_ADDRESS:-}" ]; then
    echo "Error: RAY_HEAD_ADDRESS (<host>:<port>) must be set in the environment." >&2
    exit 2
fi

cleanup() {
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

ray start --address "$RAY_HEAD_ADDRESS" \
    --num-cpus="${SLURM_CPUS_PER_TASK:-1}" \
    --num-gpus="${SLURM_GPUS_ON_NODE:-0}" \
    --disable-usage-stats \
    --block
