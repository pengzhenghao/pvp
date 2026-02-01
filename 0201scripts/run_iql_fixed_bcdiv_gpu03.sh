#!/bin/bash
#SBATCH --job-name=iql-fixbc
#SBATCH --output=./logs/iql_fixed_bcdiv_%j.out
#SBATCH --error=./logs/iql_fixed_bcdiv_%j.err
#SBATCH --time=0-24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition="gpu03"
#SBATCH --gres=gpu:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

echo "=========================================="
echo "IQL with FIXED bc_divergence formula on GPU03"
echo "Using original ESS-based bc_divergence!"
echo "tau=0.7, beta=0.5"
echo "adv_normalize=True, weight_normalize=True (matching original)"
echo "reward_normalize=True"
echo "max_grad_norm=1.0 (matching original run config)"
echo "Data: /data/caihy/bc_data_1M_correct/batches"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0

python /p0/user/caihy/pvp/0201scripts/train_iql_from_batches.py \
    --data_dir /data/caihy/bc_data_1M_correct \
    --log_dir /data/caihy/iql_fixed_bcdiv \
    --training_steps 100000 \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --save_freq 500 \
    --iql_tau 0.7 \
    --iql_beta 0.5 \
    --clip_score 100.0 \
    --max_grad_norm 1.0 \
    --reward_normalize \
    --adv_normalize \
    --weight_normalize \
    --wandb \
    --wandb_project "iql-fixed-bcdiv" \
    --exp_name "iql-fixed-tau0.7-beta0.5"

echo "IQL Fixed BC-Div Training completed!"
