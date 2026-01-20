#!/bin/bash

# TD3+BC Hyperparameter Search Script
# This script runs TD3+BC experiments with different data amounts and hyperparameters
# across 8 GPUs in parallel
#
# TD3+BC paper: "A Minimalist Approach to Offline Reinforcement Learning"
# Key hyperparameter: alpha (default 2.5 in paper)
# Actor loss = -(alpha / avg|Q|) * Q(s, π(s)) + BC_loss

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"

# Common parameters
BC_TRAINING_TIMESTEPS=1000000000  # Train until manually stopped or converged
EVAL_FREQ=5000
N_EVAL_EPISODES=50
SAVE_FREQ=10000

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    local TD3_BC_ALPHA=$3
    local SEED=$4
    
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
        > "${BASE_DIR}/logs/${EXP_NAME}.log" 2>&1 &
    
    echo "Experiment ${EXP_NAME} started with PID $!"
}

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# ============================================================
# Experiment Configuration
# ============================================================
# You can customize the experiments below

# Data collection amounts to test
DATA_STEPS_LIST=(5000 10000 20000 50000)

# TD3+BC alpha values to test (paper default is 2.5)
# Higher alpha = more weight on Q-learning, lower alpha = more weight on BC
TD3_BC_ALPHA_LIST=(0.5 1.0 2.5 5.0 10.0)

# Seeds for reproducibility
SEED_LIST=(0)

# ============================================================
# Run experiments across 8 GPUs
# ============================================================

GPU_ID=0
MAX_GPUS=8

echo "============================================================"
echo "Starting TD3+BC Hyperparameter Search"
echo "============================================================"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "TD3+BC alpha: ${TD3_BC_ALPHA_LIST[@]}"
echo "Seeds: ${SEED_LIST[@]}"
echo "============================================================"
echo ""
echo "TD3+BC Actor Loss Formula:"
echo "  actor_loss = -(α / avg|Q(s,a)|) * Q(s, π(s)) + BC_loss"
echo "  - Higher α: more weight on Q-learning (maximize Q)"
echo "  - Lower α: more weight on BC (imitate data)"
echo "============================================================"

for DATA_STEPS in "${DATA_STEPS_LIST[@]}"; do
    for TD3_BC_ALPHA in "${TD3_BC_ALPHA_LIST[@]}"; do
        for SEED in "${SEED_LIST[@]}"; do
            
            run_experiment ${GPU_ID} ${DATA_STEPS} ${TD3_BC_ALPHA} ${SEED}
            
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
done

# Wait for any remaining experiments
echo "Waiting for remaining experiments to complete..."
wait

echo "============================================================"
echo "All TD3+BC experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
