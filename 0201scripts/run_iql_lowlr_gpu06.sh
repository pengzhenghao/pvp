#!/bin/bash
#SBATCH --job-name=iql-lowlr
#SBATCH --partition=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/iql_lowlr_%j.out
#SBATCH --error=/p0/user/caihy/pvp/logs/iql_lowlr_%j.err

# IQL with lower learning rate (1e-4 instead of 3e-4)
# 更慢但可能更稳定
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

python train_iql_from_batches.py \
    --data_dir /data/caihy/bc_data_partial \
    --log_dir /data/caihy/iql_lowlr \
    --training_steps 50000 \
    --save_freq 500 \
    --batch_size 1024 \
    --iql_tau 0.7 \
    --iql_beta 0.5 \
    --learning_rate 1e-4 \
    --reward_normalize \
    --adv_normalize \
    --weight_normalize \
    --max_grad_norm 0.5 \
    --wandb \
    --wandb_project "iql-gpu06-v2" \
    --wandb_team "victorique" \
    --exp_name "iql-lowlr-1e4"
