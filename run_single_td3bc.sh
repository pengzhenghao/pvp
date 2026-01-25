#!/bin/bash

# Single TD3+BC Experiment Script
# Usage: ./run_single_td3bc.sh <GPU_ID> <DATA_STEPS> <TD3_BC_ALPHA> [SEED]
# Example: ./run_single_td3bc.sh 0 20000 2.5
# Example: ./run_single_td3bc.sh 1 50000 1.0 42

set -e

# Default values
GPU_ID=${1:-0}
DATA_STEPS=${2:-20000}
TD3_BC_ALPHA=${3:-2.5}
SEED=${4:-0}

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"

# Training parameters
BC_TRAINING_TIMESTEPS=100000
EVAL_FREQ=5000
N_EVAL_EPISODES=50
SAVE_FREQ=5000

# Crash penalty parameters (directly affect reward)
CRASH_VEHICLE_PENALTY=5.0
CRASH_OBJECT_PENALTY=5.0
OUT_OF_ROAD_PENALTY=5.0

# Create descriptive experiment name
EXP_NAME="td3bc_data${DATA_STEPS}_alpha${TD3_BC_ALPHA}_seed${SEED}"

echo "============================================================"
echo "TD3+BC Experiment Configuration"
echo "============================================================"
echo "GPU: ${GPU_ID}"
echo "Data collection steps: ${DATA_STEPS}"
echo "TD3+BC alpha: ${TD3_BC_ALPHA}"
echo "Seed: ${SEED}"
echo "Experiment name: ${EXP_NAME}"
echo "============================================================"
echo ""
echo "TD3+BC Actor Loss Formula:"
echo "  actor_loss = -(α / avg|Q(s,a)|) * Q(s, π(s)) + BC_loss"
echo "  α = ${TD3_BC_ALPHA}"
echo "============================================================"

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# Run experiment
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
    --exp_name "${EXP_NAME}" \
    --use_td3_bc \
    --data_collection_timesteps ${DATA_STEPS} \
    --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
    --td3_bc_alpha ${TD3_BC_ALPHA} \
    --eval_freq ${EVAL_FREQ} \
    --n_eval_episodes ${N_EVAL_EPISODES} \
    --save_freq ${SAVE_FREQ} \
    --seed ${SEED} \
    --crash_vehicle_penalty ${CRASH_VEHICLE_PENALTY} \
    --crash_object_penalty ${CRASH_OBJECT_PENALTY} \
    --out_of_road_penalty ${OUT_OF_ROAD_PENALTY} \
    --wandb_project "0121mainexpfull"
