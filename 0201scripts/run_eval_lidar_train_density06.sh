#!/bin/bash
#SBATCH --job-name=eval-lidar-train-d06
#SBATCH --partition=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/eval_lidar_train_density06_%j.out
#SBATCH --error=/p0/user/caihy/pvp/logs/eval_lidar_train_density06_%j.err

# Evaluate Lidar PPO Expert with traffic_density=default (0.06)
# Using TOP 200 TRAIN seeds [0-1000] - SAME as data generation Job 13165
# daytime=08:30 (default) - SAME as data generation

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

echo "=========================================="
echo "Evaluating Lidar PPO Expert"
echo "traffic_density=default (0.06)"
echo "daytime=08:30 (default)"
echo "TOP 200 TRAIN seeds [0-1000]"
echo "SAME config as data generation Job 13165"
echo "=========================================="

python eval_on_train_seeds.py \
    --model lidar \
    --num_seeds 200 \
    --use_original_config \
    --daytime "08:30" \
    --wandb \
    --wandb_project lidar-eval-train \
    --exp_name lidar-train-d06 \
    --output_dir /p0/user/caihy/pvp/results/eval_lidar_train_density06

echo "Done!"
