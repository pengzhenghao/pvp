#!/bin/bash
#SBATCH --job-name=bc-train
#SBATCH --output=./logs/bc_train_%j.out
#SBATCH --error=./logs/bc_train_%j.err
#SBATCH --time=0-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition="gpu03"
#SBATCH --gres=gpu:1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

echo "=========================================="
echo "BC Training on GPU03"
echo "Data: /data/caihy/bc_data_1M_correct/batches"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0

python /p0/user/caihy/pvp/0201scripts/train_bc_from_batches.py \
    --data_dir /data/caihy/bc_data_1M_correct \
    --log_dir /data/caihy/bc_training \
    --bc_training_steps 100000 \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --save_freq 2000 \
    --wandb \
    --wandb_project "bc-training-1M" \
    --exp_name "bc-1M-data"

echo "BC Training completed!"
