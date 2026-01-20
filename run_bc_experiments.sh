#!/bin/bash

# Pure BC (Behavioral Cloning) Experiment Script
# This script runs pure BC experiments with different data amounts
# across 8 GPUs in parallel
#
# BC is the simplest baseline - just supervised learning on expert data
# No hyperparameters to tune, only data amount matters

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"

# Common parameters
BC_TRAINING_TIMESTEPS=100000  # Train until manually stopped or converged
EVAL_FREQ=5000
N_EVAL_EPISODES=50
SAVE_FREQ=5000

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    local SEED=$3
    
    # Create descriptive experiment name
    local EXP_NAME="bc_data${DATA_STEPS}_seed${SEED}"
    
    echo "Starting experiment: ${EXP_NAME} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
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
# You can customize the experiments below

# Data collection amounts to test (from large to small)
DATA_STEPS_LIST=(50000 20000 10000 5000)

# Seeds for reproducibility
SEED_LIST=(0)

# ============================================================
# Run experiments across 8 GPUs
# ============================================================

GPU_ID=0
MAX_GPUS=8

echo "============================================================"
echo "Starting Pure BC Experiments"
echo "============================================================"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "Seeds: ${SEED_LIST[@]}"
echo "============================================================"
echo ""
echo "Pure BC:"
echo "  - Supervised learning to imitate expert actions"
echo "  - No Q-learning, no conservative penalties"
echo "  - Simplest offline RL baseline"
echo "============================================================"

for DATA_STEPS in "${DATA_STEPS_LIST[@]}"; do
    for SEED in "${SEED_LIST[@]}"; do
        
        run_experiment ${GPU_ID} ${DATA_STEPS} ${SEED}
        
        # Move to next GPU
        GPU_ID=$(( (GPU_ID + 1) % MAX_GPUS ))
        
        # If we've used all GPUs, wait for them to finish before continuing
        if [ ${GPU_ID} -eq 0 ]; then
            echo "All 8 GPUs are running experiments. Waiting for completion..."
            wait
            echo "Batch completed. Starting next batch..."
        fi
        
    done
done

# Wait for any remaining experiments
echo "Waiting for remaining experiments to complete..."
wait

echo "============================================================"
echo "All Pure BC experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
