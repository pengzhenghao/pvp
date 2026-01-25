#!/bin/bash
#SBATCH --job-name="eval_baseline"
#SBATCH --output="/p0/user/caihy/pvp/logs/slurm_eval_baseline_%j.out"
#SBATCH --error="/p0/user/caihy/pvp/logs/slurm_eval_baseline_%j.err"
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --partition="gpu06"

# ============================================================
# SLURM Script for Baseline Model Evaluation
# Evaluates pretrained model (RGB) and expert model (lidar)
# Uploads results to wandb project: 0123mainexp
# ============================================================

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "============================================================"

mkdir -p /p0/user/caihy/pvp/logs

# Run the evaluation script
bash /p0/user/caihy/pvp/run_eval_baseline.sh

echo "SLURM baseline evaluation job completed!"
