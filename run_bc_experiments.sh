#!/bin/bash

# Pure BC (Behavioral Cloning) Experiment Script
# This script runs pure BC experiments with different data amounts
# Each data amount runs on a separate GPU (8 data amounts on 8 GPUs)
#
# BC is the simplest baseline - just supervised learning on expert data
# No hyperparameters to tune, only data amount matters

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"
BUFFER_PATH="/home/caihy/pvp/data_buffer_20000.npz"

# Common parameters (optimized)
BC_TRAINING_TIMESTEPS=10000
EVAL_FREQ=1000
N_EVAL_EPISODES=400
SAVE_FREQ=1000
SEED=0

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    
    # Create descriptive experiment name
    local EXP_NAME="bc_data${DATA_STEPS}_seed${SEED}"
    
    echo "Starting experiment: ${EXP_NAME} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --load_buffer "${BUFFER_PATH}" \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
        --eval_freq ${EVAL_FREQ} \
        --n_eval_episodes ${N_EVAL_EPISODES} \
        --save_freq ${SAVE_FREQ} \
        --seed ${SEED} \
        --wandb_project "domain-adaptation-0120" \
        > "${BASE_DIR}/logs/${EXP_NAME}.log" 2>&1 &
    
    echo "Experiment ${EXP_NAME} started with PID $!"
}

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# ============================================================
# Experiment Configuration
# ============================================================
# 8 data amounts for 8 GPUs (from large to small)
DATA_STEPS_LIST=(20000 17500 15000 12500 10000 7500 5000 2500)

# ============================================================
# Run experiments across 8 GPUs
# ============================================================

echo "============================================================"
echo "Starting Pure BC Experiments (8 data amounts on 8 GPUs)"
echo "============================================================"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "Loading buffer from: ${BUFFER_PATH}"
echo "============================================================"
echo ""
echo "Pure BC:"
echo "  - Supervised learning to imitate expert actions"
echo "  - No Q-learning, no conservative penalties"
echo "  - Simplest offline RL baseline"
echo "============================================================"

# Run all 8 experiments in parallel on 8 GPUs
for i in "${!DATA_STEPS_LIST[@]}"; do
    GPU_ID=$i
    DATA_STEPS=${DATA_STEPS_LIST[$i]}
    run_experiment ${GPU_ID} ${DATA_STEPS}
done

# Wait for all experiments to complete
echo "All 8 experiments started. Waiting for completion..."
wait

echo "============================================================"
echo "All Pure BC experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
