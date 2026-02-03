#!/bin/bash
#SBATCH --job-name=bc-fresh-0610
#SBATCH --output=/p0/user/caihy/pvp/logs/%j_bc_fresh_0610.out
#SBATCH --error=/p0/user/caihy/pvp/logs/%j_bc_fresh_0610.err
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
echo "BC Training on GPU03 - 500K 0610 Dataset (Fresh Experiment)"
echo "=========================================="
echo "Data: /data/caihy/bc_data_500K_0610 (25 batch files, 501K transitions)"
echo "Initial ckpt: bc_step_046000.zip (pretrained weights)"
echo "Start step: 0 (new experiment, counting from 0)"
echo "End step: 100000"
echo "Daytime: 06:10 (matches data generation)"
echo "Rollout collection: ENABLED (every 5 steps for speedup)"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=0
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl

# Train BC on 500K 0610 dataset, starting from 46K pretrained weights
python /p0/user/caihy/pvp/0201scripts/train_bc_from_batches.py \
    --data_dir /data/caihy/bc_data_500K_0610 \
    --log_dir /data/caihy/bc_training \
    --ckpt /data/caihy/bc_training/bc-1M-resume-40K_2026-02-01_16-49-53_261baf3e/bc_step_046000.zip \
    --start_step 0 \
    --bc_training_steps 100000 \
    --batch_size 1024 \
    --learning_rate 1e-4 \
    --save_freq 2000 \
    --log_freq 100 \
    --daytime "06:10" \
    --rollout_log_freq 100 \
    --rollout_step_freq 5 \
    --wandb \
    --wandb_project "bc-training-500K-0610" \
    --exp_name "bc-500K-0610-fresh"

echo "BC Training completed!"
