#!/bin/bash
#SBATCH --job-name=verify-bc16k-seeds
#SBATCH --partition=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/%j_verify_bc16k_seeds.out
#SBATCH --error=/p0/user/caihy/pvp/logs/%j_verify_bc16k_seeds.err

# 验证 BC step 16000 使用正确的 seeds

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

CHECKPOINT="/data/caihy/bc_training/bc-1M-data_2026-01-31_22-55-51_5e4ef6f2/bc_step_016000.zip"

echo "=========================================="
echo "VERIFICATION: BC step 16000 with CORRECT seeds"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo ""
echo "Using FIXED TOP_200_TRAIN_SEEDS (matching data generation)"
echo "Seeds[50:60]: [799, 816, 740, 781, 382, 738, 609, 765, 930, 932]"
echo "=========================================="

python eval_bc_checkpoint.py \
    --checkpoint "$CHECKPOINT" \
    --num_envs 10 \
    --num_seeds 200 \
    --daytime "08:30" \
    --wandb \
    --wandb_project "verify-seeds-fix" \
    --wandb_team "victorique" \
    --exp_name "bc16k_FIXED_seeds"

echo "Done!"
