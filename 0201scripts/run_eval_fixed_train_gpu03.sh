#!/bin/bash
#SBATCH --job-name=eval-fixed-train
#SBATCH --partition=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/%j_eval_fixed_train.out
#SBATCH --error=/p0/user/caihy/pvp/logs/%j_eval_fixed_train.err

# ========================================
# FIXED TRAIN SEEDS EVALUATION
# Bug fix: Now uses correct TOP_200_TRAIN_SEEDS matching data generation
# Seeds[50:60] should be: [799, 816, 740, 781, 382, 738, 609, 765, 930, 932]
# ========================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

# Arguments: $1 = checkpoint path, $2 = model name, $3 = step
CHECKPOINT="$1"
MODEL_NAME="$2"
STEP="$3"

echo "=========================================="
echo "FIXED TRAIN SEEDS EVALUATION"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Step: $STEP"
echo "Checkpoint: $CHECKPOINT"
echo ""
echo "BUG FIX: Using CORRECT TOP_200_TRAIN_SEEDS"
echo "Seeds[50:60]: [799, 816, 740, 781, 382, 738, 609, 765, 930, 932]"
echo "=========================================="

python eval_bc_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --num_envs 10 \
    --num_seeds 200 \
    --daytime "08:30" \
    --wandb \
    --wandb_project "eval-FIXED-train" \
    --wandb_team "victorique" \
    --exp_name "${MODEL_NAME}_step${STEP}_FIXED"

echo "Done!"
