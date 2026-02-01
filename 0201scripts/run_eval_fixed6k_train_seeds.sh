#!/bin/bash
#SBATCH --job-name=eval-fix6k-train
#SBATCH --partition=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/eval_fixed6k_train_%j.out
#SBATCH --error=/p0/user/caihy/pvp/logs/eval_fixed6k_train_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

CHECKPOINT="/p0/user/caihy/iql_ckpts_for_eval/fixed_step_006000.zip"

echo "=========================================="
echo "IQL Fixed BC-Div Step 6000 - TRAIN SEEDS"
echo "=========================================="
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
    --wandb_project "iql-fixed-bcdiv-eval" \
    --wandb_team "victorique" \
    --exp_name "fixed6k_train_seeds"

echo "Evaluation complete!"
