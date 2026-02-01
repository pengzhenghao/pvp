#!/bin/bash
#SBATCH --job-name=repr_12219
#SBATCH --output=/p0/user/caihy/pvp/logs/repr_12219_full_%A_%a.out
#SBATCH --error=/p0/user/caihy/pvp/logs/repr_12219_full_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-19%4
#SBATCH --partition=gpu06
# ============================================================
# FULL Reproduce Job 12219 on GPU07
# - 20 checkpoints (step 1000, 2000, ..., 20000)
# - 4 concurrent jobs
# - 20 envs, 200 seeds each
# - Direct output to SLURM stdout (no tee)
# ============================================================

export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl
export DISPLAY=

cd /p0/user/caihy/pvp
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

TRIAL_DIR="/p0/user/caihy/pvp/runs/bc_train_25k_20260126_013834_freelevel0.95/bc_train_25k_20260126_013834_freelevel0.95_2026-01-26_01-38-40_22c65f75"
CKPT_DIR="$TRIAL_DIR/bc_checkpoints"
EVAL_DIR="$TRIAL_DIR/reproduce_12219_full"
mkdir -p "$EVAL_DIR"

TASK_ID=${SLURM_ARRAY_TASK_ID}
STEP=$((($TASK_ID + 1) * 1000))
STEP_STR=$(printf "%06d" $STEP)
CKPT="$CKPT_DIR/bc_step_${STEP_STR}.zip"

echo "========================================================"
echo "Reproducing Job 12219 - Checkpoint step $STEP"
echo "========================================================"
echo "Task ID: $TASK_ID"
echo "Checkpoint: $CKPT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Node: $(hostname)"
echo "========================================================"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint not found: $CKPT"
    exit 1
fi

# Direct call - same as 12219
python eval_hard_scenarios_parallel.py \
    --model pretrained \
    --pretrained_checkpoint "$CKPT" \
    --num_seeds 200 \
    --num_envs 20 \
    --output "$EVAL_DIR/step_${STEP_STR}" \
    --no_expert

echo ""
echo "Step $STEP evaluation completed at $(date)"
