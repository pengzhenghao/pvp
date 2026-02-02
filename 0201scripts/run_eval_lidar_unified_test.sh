#!/bin/bash
#SBATCH --job-name=lidar-unified-test
#SBATCH --partition=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/%j_lidar_unified_test.out
#SBATCH --error=/p0/user/caihy/pvp/logs/%j_lidar_unified_test.err

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
echo "TEST seeds [1000, 2000)"
echo "=========================================="

python eval_bc_checkpoint.py \
    --model lidar \
    --num_envs 10 \
    --num_seeds 200 \
    --daytime "08:30" \
    --use_test_seeds \
    --wandb \
    --wandb_project "eval-FIXED-test" \
    --wandb_team "victorique" \
    --exp_name "lidar_unified_TEST"

echo "Done!"
