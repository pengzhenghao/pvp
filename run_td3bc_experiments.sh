#!/bin/bash

# TD3+BC Experiment Script
# This script runs TD3+BC experiments with different data amounts
# Each data amount runs on a separate GPU (8 data amounts on 8 GPUs)
#
# TD3+BC paper: "A Minimalist Approach to Offline Reinforcement Learning"
# Using optimal hyperparameters found: alpha=0.5 (default)

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"

# Common parameters (optimized)
BC_TRAINING_TIMESTEPS=10000
EVAL_FREQ=1000
N_EVAL_EPISODES=400
SAVE_FREQ=1000
SEED=0

# TD3+BC optimal hyperparameter
TD3_BC_ALPHA=0.5

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    
    # Create descriptive experiment name
    local EXP_NAME="td3bc_data${DATA_STEPS}_alpha${TD3_BC_ALPHA}_seed${SEED}"
    
    echo "Starting experiment: ${EXP_NAME} on GPU ${GPU_ID}"
    
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
        --wandb_project "domain-adaptation" \
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
echo "Starting TD3+BC Experiments (8 data amounts on 8 GPUs)"
echo "============================================================"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "TD3+BC alpha: ${TD3_BC_ALPHA}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "============================================================"
echo ""
echo "TD3+BC Actor Loss Formula:"
echo "  actor_loss = -(α / avg|Q(s,a)|) * Q(s, π(s)) + BC_loss"
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
echo "All TD3+BC experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
