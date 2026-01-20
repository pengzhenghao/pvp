#!/bin/bash

# CQL (Conservative Q-Learning) Experiment Script
# This script runs CQL experiments with different data amounts
# Each data amount runs on a separate GPU (8 data amounts on 8 GPUs)
#
# Using optimal hyperparameters found: alpha=10.0, temp=1.0, num_random=10 (defaults)

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"

# Common parameters (optimized)
BC_TRAINING_TIMESTEPS=10000
EVAL_FREQ=1000
N_EVAL_EPISODES=400
SAVE_FREQ=1000
SEED=0

# CQL optimal hyperparameters
CQL_ALPHA=10.0
CQL_TEMP=1.0
NUM_RANDOM_ACTIONS=10

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    
    # Create descriptive experiment name
    local EXP_NAME="cql_data${DATA_STEPS}_alpha${CQL_ALPHA}_seed${SEED}"
    
    echo "Starting experiment: ${EXP_NAME} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --use_cql \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
        --cql_alpha ${CQL_ALPHA} \
        --cql_temp ${CQL_TEMP} \
        --num_random_actions ${NUM_RANDOM_ACTIONS} \
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
echo "Starting CQL Experiments (8 data amounts on 8 GPUs)"
echo "============================================================"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "CQL alpha: ${CQL_ALPHA}"
echo "CQL temp: ${CQL_TEMP}"
echo "Num random actions: ${NUM_RANDOM_ACTIONS}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "============================================================"
echo ""
echo "CQL adds conservative penalty to prevent Q-value overestimation"
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
echo "All CQL experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
