#!/bin/bash
#SBATCH --job-name=eval-lidar-d06
#SBATCH --partition=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/eval_lidar_density06_%j.out
#SBATCH --error=/p0/user/caihy/pvp/logs/eval_lidar_density06_%j.err

# Evaluate Lidar PPO Expert with traffic_density=default (0.06)
# Using TOP 200 EVAL seeds [1000-2000]

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

echo "=========================================="
echo "Evaluating Lidar PPO Expert"
echo "traffic_density=default (0.06)"
echo "TOP 200 EVAL seeds [1000-2000]"
echo "=========================================="

python eval_on_train_seeds.py \
    --model lidar \
    --num_seeds 200 \
    --use_test_seeds \
    --use_original_config \
    --daytime "08:30" \
    --wandb \
    --wandb_project lidar-eval-test \
    --exp_name lidar-test-d06 \
    --output_dir /p0/user/caihy/pvp/results/eval_density06

echo "Done!"
