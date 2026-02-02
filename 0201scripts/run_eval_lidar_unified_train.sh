#!/bin/bash
#SBATCH --job-name=lidar-unified-train
#SBATCH --partition=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/%j_lidar_unified_train.out
#SBATCH --error=/p0/user/caihy/pvp/logs/%j_lidar_unified_train.err

# ========================================
# LIDAR EXPERT EVALUATION using UNIFIED CODE
# Uses the SAME eval_bc_checkpoint.py as BC/IQL
# This ensures we use EXACT SAME seeds and env config!
# ========================================

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

echo "=========================================="
echo "LIDAR EXPERT - UNIFIED EVAL CODE"
echo "=========================================="
echo "Using SAME eval_bc_checkpoint.py as BC/IQL"
echo "Seeds[50:60]: [799, 816, 740, 781, 382, 738, 609, 765, 930, 932]"
echo "TRAIN seeds [0, 1000)"
echo "=========================================="

python eval_bc_checkpoint.py \
    --model lidar \
    --num_envs 10 \
    --num_seeds 200 \
    --daytime "08:30" \
    --wandb \
    --wandb_project "eval-FIXED-train" \
    --wandb_team "victorique" \
    --exp_name "lidar_unified_TRAIN"

echo "Done!"
