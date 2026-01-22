#!/bin/bash
# ============================================================
# Run a single LoRA experiment on a specified GPU
# Usage: ./run_single_lora.sh <GPU_ID> <RANK> <ALPHA> <DATA_STEPS>
# Example: ./run_single_lora.sh 3 16 1.0 6000
# ============================================================

# Default values (can be overridden by arguments)
GPU_ID=${1:-0}
LORA_RANK=${2:-16}
LORA_ALPHA=${3:-1.0}
DATA_STEPS=${4:-6000}

# Fixed parameters
LORA_DROPOUT=0.0
LORA_TARGET="actor"
SEED=0

# Paths
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"
BUFFER_PATH="/home/caihy/pvp/data_buffer_20000.npz"

# Training parameters
BC_TRAINING_TIMESTEPS=2000
SAVE_FREQ=1000
EVAL_FREQ=100
N_EVAL_EPISODES=500

# Penalty parameters
CRASH_VEHICLE_PENALTY=5.0
CRASH_OBJECT_PENALTY=5.0
OUT_OF_ROAD_PENALTY=5.0

# Wandb project
WANDB_PROJECT="0121mainexpfull"

# Create experiment name
EXP_NAME="lora_data${DATA_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_d${LORA_DROPOUT}_${LORA_TARGET}_seed${SEED}"

# Create logs directory
mkdir -p ${BASE_DIR}/logs

echo "============================================================"
echo "Running single LoRA experiment"
echo "============================================================"
echo "GPU: ${GPU_ID}"
echo "LoRA rank: ${LORA_RANK}"
echo "LoRA alpha: ${LORA_ALPHA}"
echo "LoRA dropout: ${LORA_DROPOUT}"
echo "LoRA target: ${LORA_TARGET}"
echo "Data steps: ${DATA_STEPS}"
echo "Experiment name: ${EXP_NAME}"
echo "Log file: ${BASE_DIR}/logs/${EXP_NAME}.log"
echo "============================================================"

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
    --exp_name "${EXP_NAME}" \
    --use_lora \
    --lora_rank ${LORA_RANK} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --lora_target ${LORA_TARGET} \
    --load_buffer "${BUFFER_PATH}" \
    --data_collection_timesteps ${DATA_STEPS} \
    --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
    --eval_freq ${EVAL_FREQ} \
    --n_eval_episodes ${N_EVAL_EPISODES} \
    --save_freq ${SAVE_FREQ} \
    --seed ${SEED} \
    --crash_vehicle_penalty ${CRASH_VEHICLE_PENALTY} \
    --crash_object_penalty ${CRASH_OBJECT_PENALTY} \
    --out_of_road_penalty ${OUT_OF_ROAD_PENALTY} \
    --wandb_project "${WANDB_PROJECT}" \
    > "${BASE_DIR}/logs/${EXP_NAME}.log" 2>&1 &

echo "Experiment started in background!"
echo "Monitor with: tail -f ${BASE_DIR}/logs/${EXP_NAME}.log"
echo "============================================================"
