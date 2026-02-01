#!/bin/bash
#SBATCH --job-name=iql-b05
#SBATCH --partition=gpu06
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/p0/user/caihy/pvp/logs/iql_beta05_%j.out
#SBATCH --error=/p0/user/caihy/pvp/logs/iql_beta05_%j.err

# IQL beta=0.5 (与 GPU03 Fixed BC-Div 相同配置，作为对照)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

cd /p0/user/caihy/pvp/0201scripts

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

python train_iql_from_batches.py \
    --data_dir /data/caihy/bc_data_partial \
    --log_dir /data/caihy/iql_beta05 \
    --training_steps 50000 \
    --save_freq 500 \
    --batch_size 1024 \
    --iql_tau 0.7 \
    --iql_beta 0.5 \
    --learning_rate 3e-4 \
    --reward_normalize \
    --adv_normalize \
    --weight_normalize \
    --max_grad_norm 1.0 \
    --wandb \
    --wandb_project "iql-gpu06-v2" \
    --wandb_team "victorique" \
    --exp_name "iql-beta0.5-baseline"
