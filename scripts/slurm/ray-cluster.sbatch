#!/usr/bin/env bash
# Fixed multi-node SLURM allocation that starts a Ray cluster across all allocated
# nodes and runs a HELIOS command (or user script) on the head node against that
# cluster. HELIOS itself never submits sbatch jobs.
#
# Usage:
#   sbatch scripts/slurm/ray-cluster.sbatch helios prep tile-features --dataset /data/COHORT1

#SBATCH --job-name=helios-ray
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: sbatch scripts/slurm/ray-cluster.sbatch <command> [args...]" >&2
    exit 2
fi

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
read -r -a nodes_array <<< "$(echo "$nodes" | tr '\n' ' ')"
head_node="${nodes_array[0]}"
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
port="${RAY_PORT:-6379}"
ray_address="${head_ip}:${port}"

cleanup() {
    srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_ip" --port="$port" \
    --num-cpus="${SLURM_CPUS_PER_TASK:-1}" --num-gpus="${SLURM_GPUS_ON_NODE:-0}" \
    --disable-usage-stats --block &

sleep 5

worker_num=$((SLURM_NNODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i="${nodes_array[$i]}"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ray_address" \
        --num-cpus="${SLURM_CPUS_PER_TASK:-1}" --num-gpus="${SLURM_GPUS_ON_NODE:-0}" \
        --disable-usage-stats --block &
done

sleep 5

"$@" --ray-address "$ray_address"
