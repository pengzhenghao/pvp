#!/bin/bash
#SBATCH --job-name=eval_dist
#SBATCH --output=/p0/user/caihy/pvp/logs/eval_dist_%A_%a.out
#SBATCH --error=/p0/user/caihy/pvp/logs/eval_dist_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-3

# ============================================================
# Distributed Parallel Evaluation (4 GPUs = 4x30 = 120 envs)
# ============================================================
# Usage:
#   sbatch run_eval_distributed.sh
#
# This launches 4 jobs, each on 1 GPU with 30 parallel envs
# Total: 120 parallel environments across 4 GPUs
# Expected time: ~3-5 minutes for 200 seeds (vs 12 min single GPU)
# ============================================================

# Environment setup
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl
export DISPLAY=

cd /p0/user/caihy/pvp
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

# Configuration
NUM_JOBS=4                  # Total number of distributed jobs (must match --array)
NUM_ENVS_PER_GPU=30         # Each GPU runs 30 parallel envs (max without OOM)
NUM_SEEDS=200               # Total seeds to evaluate
MODEL="both"                # iql, td3, td3bc2, both, all
SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
OUTPUT_DIR="./results/hard_scenario_eval_${SLURM_ARRAY_JOB_ID}"

echo "========================================================"
echo "Distributed Evaluation - Job ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS}"
echo "========================================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Envs per GPU: ${NUM_ENVS_PER_GPU}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run evaluation for this job's portion of seeds
python eval_hard_scenarios_parallel.py \
    --model ${MODEL} \
    --num_seeds ${NUM_SEEDS} \
    --num_envs ${NUM_ENVS_PER_GPU} \
    --output ${OUTPUT_DIR} \
    --job_id ${SLURM_ARRAY_TASK_ID} \
    --num_jobs ${NUM_JOBS} \
    --no_expert

echo ""
echo "Job ${SLURM_ARRAY_TASK_ID} completed!"
echo "After all jobs complete, run: python merge_eval_results.py --input ${OUTPUT_DIR}"
