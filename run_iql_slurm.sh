#!/bin/bash
#SBATCH --job-name="iql_exp"
#SBATCH --output="/p0/user/caihy/pvp/logs/slurm_iql_%j.out"
#SBATCH --error="/p0/user/caihy/pvp/logs/slurm_iql_%j.err"
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=120:00:00
#SBATCH --partition="all"

# ============================================================
# SLURM Script for IQL Experiments
# ============================================================
# 用法:
#   sbatch run_iql_slurm.sh                    # 默认 4 GPU
#   sbatch --gres=gpu:8 run_iql_slurm.sh       # 8 GPU
#   sbatch --gres=gpu:1 run_iql_slurm.sh fast  # 快速测试
# ============================================================

if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    NUM_GPUS=4
fi

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated GPUs: $NUM_GPUS"
echo "============================================================"

mkdir -p /p0/user/caihy/pvp/logs
export SLURM_NUM_GPUS=$NUM_GPUS

bash /p0/user/caihy/pvp/run_iql_experiments.sh "$@"

echo "SLURM IQL job completed!"
