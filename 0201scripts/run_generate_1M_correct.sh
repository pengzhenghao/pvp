#!/bin/bash
#SBATCH --job-name=gen-1M-v2
#SBATCH --partition=gpu03
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/gen_1M_correct_%j.out
#SBATCH --error=/p0/user/caihy/pvp/logs/gen_1M_correct_%j.err

# CORRECT data generation:
# - traffic_density NOT SET (uses env default 0.06)
# - random_traffic NOT SET
# - With evaluation metrics tracking
# - With wandb logging

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

echo "=========================================="
echo "CORRECT BC Data Generation with Eval Metrics"
echo "Using env default traffic_density (0.06), NOT 0.3!"
echo "=========================================="

# Total: 1M transitions
# Batch size: 20480 (1024 * 20)
# Num envs: 16

python generate_bc_data_sequential.py \
    --data_dir /data/caihy/bc_data_1M_correct \
    --total_timesteps 1003520 \
    --num_envs 16 \
    --batch_size 20480 \
    --wandb \
    --wandb_project "bc-data-gen" \
    --exp_name "bc-data-1M-correct"

echo "=========================================="
echo "Data generation complete!"
echo "Check wandb for expert evaluation metrics"
echo "=========================================="
