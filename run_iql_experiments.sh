#!/bin/bash

# IQL (Implicit Q-Learning) Hyperparameter Search Script
# This script runs IQL experiments with different data amounts and hyperparameters
# across 8 GPUs in parallel
#
# IQL paper: "Offline Reinforcement Learning with Implicit Q-Learning"
# Key hyperparameters:
# - tau (expectile): Controls how V approximates max Q (0.5=mean, 0.7-0.9 typical)
# - beta (temperature): Controls policy extraction greediness (higher=more greedy)

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
    local IQL_TAU=$3
    local IQL_BETA=$4
    local SEED=$5
    
    # Create descriptive experiment name
    local EXP_NAME="iql_data${DATA_STEPS}_tau${IQL_TAU}_beta${IQL_BETA}_seed${SEED}"
    
    echo "Starting experiment: ${EXP_NAME} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --use_iql \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
        --iql_tau ${IQL_TAU} \
        --iql_beta ${IQL_BETA} \
        --max_grad_norm 1.0 \
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

# Data collection amounts to test
DATA_STEPS_LIST=(50000 20000 10000 5000)

# IQL tau (expectile) values to test
# - 0.5 = mean (no max approximation)
# - 0.7-0.9 = typical values for offline RL
# - Higher = better max approximation but higher variance
IQL_TAU_LIST=(0.5 0.7 0.8 0.9)

# IQL beta (temperature) values to test
# - Higher = more greedy policy extraction
# - Lower = more uniform/exploratory
IQL_BETA_LIST=(1.0 3.0 10.0)

# Seeds for reproducibility
SEED_LIST=(0)

# ============================================================
# Run experiments across 8 GPUs
# ============================================================

GPU_ID=0
MAX_GPUS=8

echo "============================================================"
echo "Starting IQL Hyperparameter Search"
echo "============================================================"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "IQL tau (expectile): ${IQL_TAU_LIST[@]}"
echo "IQL beta (temperature): ${IQL_BETA_LIST[@]}"
echo "Seeds: ${SEED_LIST[@]}"
echo "============================================================"
echo ""
echo "IQL Key Ideas:"
echo "  1. V(s) trained with expectile regression to approximate max_a Q(s,a)"
echo "  2. Q(s,a) trained with TD using V(s') instead of max Q(s',a')"
echo "  3. Policy extracted with advantage-weighted regression (AWR)"
echo ""
echo "  - tau > 0.5: V approximates max Q (higher tau = closer to max)"
echo "  - beta: controls how greedy the policy is (higher = more greedy)"
echo "============================================================"

for DATA_STEPS in "${DATA_STEPS_LIST[@]}"; do
    for IQL_TAU in "${IQL_TAU_LIST[@]}"; do
        for IQL_BETA in "${IQL_BETA_LIST[@]}"; do
            for SEED in "${SEED_LIST[@]}"; do
                
                run_experiment ${GPU_ID} ${DATA_STEPS} ${IQL_TAU} ${IQL_BETA} ${SEED}
                
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
done

# Wait for any remaining experiments
echo "Waiting for remaining experiments to complete..."
wait

echo "============================================================"
echo "All IQL experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
