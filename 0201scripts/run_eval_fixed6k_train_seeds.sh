#!/bin/bash
#SBATCH --job-name=eval-iql-train
#SBATCH --partition=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/%j_eval_iql_train.out
#SBATCH --error=/p0/user/caihy/pvp/logs/%j_eval_iql_train.err

# Usage: sbatch run_eval_fixed6k_train_seeds.sh <checkpoint_path> <step_number>
# Example: sbatch run_eval_fixed6k_train_seeds.sh /data/caihy/iql_fixed_bcdiv/.../iql_step_001000.zip 001000

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

# Accept checkpoint path, step number, and optional model prefix
CHECKPOINT="${1}"
STEP="${2}"
MODEL_PREFIX="${3:-iql}"  # Default to "iql" if not specified
WANDB_PROJECT="${4:-iql-eval-1M}"  # Default wandb project

if [ -z "$CHECKPOINT" ] || [ -z "$STEP" ]; then
    echo "ERROR: Usage: sbatch $0 <checkpoint_path> <step_number> [model_prefix] [wandb_project]"
    echo "Example: sbatch $0 /data/caihy/iql_ckpts/iql_step_001000.zip 001000 iql iql-eval-1M"
    echo "Example: sbatch $0 /data/caihy/bc_ckpts/bc_step_010000.zip 010000 bc bc-eval-1M"
    exit 1
fi

echo "=========================================="
echo "${MODEL_PREFIX} Evaluation - Step ${STEP} - TRAIN SEEDS"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Model: ${MODEL_PREFIX}"
echo "Wandb Project: ${WANDB_PROJECT}"
echo ""
echo "=== Environment Configuration ==="
echo "  daytime: 08:30"
echo "  traffic_density: NOT SET (env default 0.06)"
echo "  random_traffic: NOT SET (env default)"
echo "  image_observation: True"
echo "  crash_vehicle_done: False"
echo "  Seeds: TOP 200 HARD in [0, 1000)"
echo "=========================================="
echo ""

python eval_bc_checkpoint.py \
    --checkpoint "${CHECKPOINT}" \
    --num_envs 10 \
    --num_seeds 200 \
    --daytime "08:30" \
    --wandb \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_team "victorique" \
    --exp_name "${MODEL_PREFIX}_step${STEP}_train"

echo "Evaluation complete!"
