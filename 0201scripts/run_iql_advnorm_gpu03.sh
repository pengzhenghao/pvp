#!/bin/bash
#SBATCH --job-name=iql-advnorm
#SBATCH --output=./logs/iql_advnorm_%j.out
#SBATCH --error=./logs/iql_advnorm_%j.err
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
echo "IQL with Advantage Normalization on GPU03"
echo "tau=0.7, beta=0.5"
echo "WITH advantage normalization"
echo "WITH gradient clipping (max_grad_norm=1.0)"
echo "Data: /data/caihy/bc_data_1M_correct/batches"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0

python /p0/user/caihy/pvp/0201scripts/train_iql_from_batches.py \
    --data_dir /data/caihy/bc_data_1M_correct \
    --log_dir /data/caihy/iql_advnorm \
    --training_steps 100000 \
    --batch_size 1024 \
    --learning_rate 3e-4 \
    --save_freq 500 \
    --iql_tau 0.7 \
    --iql_beta 0.5 \
    --clip_score 100.0 \
    --max_grad_norm 1.0 \
    --adv_normalize \
    --wandb \
    --wandb_project "iql-advnorm-1M" \
    --exp_name "iql-advnorm-tau0.7-beta0.5"

echo "IQL AdvNorm Training completed!"
