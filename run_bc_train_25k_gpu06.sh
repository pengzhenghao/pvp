#!/bin/bash
#SBATCH --job-name=bc_train_25k
#SBATCH --output=/p0/user/caihy/pvp/logs/bc_train_25k_%j.out
#SBATCH --error=/p0/user/caihy/pvp/logs/bc_train_25k_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu06
# ============================================================
# BC Training with 25000 data, save checkpoints for batch eval
# ============================================================

export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl
export DISPLAY=

cd /p0/user/caihy/pvp
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

# Configuration
BUFFER_PATH="/bigdata/caihy/data_buffer_domain-adaptation-0120_25000.npz"
BC_TIMESTEPS=20000
BC_SAVE_FREQ=100
EXP_NAME="bc_train_25k_$(date +%Y%m%d_%H%M%S)"

echo "========================================================"
echo "BC Training with 25000 Data"
echo "========================================================"
echo "Experiment: ${EXP_NAME}"
echo "Buffer: ${BUFFER_PATH}"
echo "BC Training steps: ${BC_TIMESTEPS}"
echo "Save frequency: ${BC_SAVE_FREQ}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Node: $(hostname)"
echo "========================================================"

# Verify buffer exists
if [ ! -f "${BUFFER_PATH}" ]; then
    echo "ERROR: Buffer not found: ${BUFFER_PATH}"
    exit 1
fi
echo "Buffer size: $(ls -lh ${BUFFER_PATH} | awk '{print $5}')"

# Run training with --no_eval for batch evaluation later
python train_bc_metadrive_online.py \
    --exp_name "${EXP_NAME}" \
    --load_buffer "${BUFFER_PATH}" \
    --data_collection_timesteps 25000 \
    --bc_training_timesteps ${BC_TIMESTEPS} \
    --bc_save_freq ${BC_SAVE_FREQ} \
    --no_eval \
    --wandb \
    --wandb_project "bc_train_25k"

echo ""
echo "Training completed!"
echo "Checkpoints saved for batch evaluation"
