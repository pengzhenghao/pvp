#!/bin/bash
#SBATCH --job-name=bc-train-resume
#SBATCH --output=/p0/user/caihy/pvp/logs/%j_bc_train_resume.out
#SBATCH --error=/p0/user/caihy/pvp/logs/%j_bc_train_resume.err
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
echo "BC Training RESUME on GPU03"
echo "=========================================="
echo "Resuming from: bc_step_040000.zip"
echo "Data: /data/caihy/bc_data_1M_correct/batches (51 batch files)"
echo "Start step: 40000"
echo "End step: 100000"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0

# Resume training from step 40000 using the 40K checkpoint
python /p0/user/caihy/pvp/0201scripts/train_bc_from_batches.py \
    --data_dir /data/caihy/bc_data_1M_correct \
    --log_dir /data/caihy/bc_training \
    --ckpt /data/caihy/bc_training/bc-1M-data_2026-01-31_22-55-51_5e4ef6f2/bc_step_040000.zip \
    --start_step 40000 \
    --bc_training_steps 100000 \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --save_freq 2000 \
    --wandb \
    --wandb_project "bc-training-1M-resume" \
    --exp_name "bc-1M-resume-40K"

echo "BC Training completed!"
