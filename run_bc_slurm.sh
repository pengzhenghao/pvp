#!/bin/bash
#SBATCH --job-name="bc_exp"
#SBATCH --output="/p0/user/caihy/pvp/logs/slurm_bc_%j.out"
#SBATCH --error="/p0/user/caihy/pvp/logs/slurm_bc_%j.err"
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --time=120:00:00
#SBATCH --partition="all"

# ============================================================
# SLURM Script for BC Experiments
# 2 个 GPU，分两轮运行
# ============================================================

if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NUM_GPUS=2
fi

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated GPUs: $NUM_GPUS"
echo "============================================================"

mkdir -p /p0/user/caihy/pvp/logs
export SLURM_NUM_GPUS=$NUM_GPUS

bash /p0/user/caihy/pvp/run_bc_experiments.sh "$@"

echo "SLURM BC job completed!"
