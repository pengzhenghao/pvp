#!/bin/bash

# IQL (Implicit Q-Learning) Experiment Script
# This script runs IQL experiments with different data amounts
# Each data amount runs on a separate GPU (8 data amounts on 8 GPUs)
#
# IQL paper: "Offline Reinforcement Learning with Implicit Q-Learning"
# Using optimal hyperparameters found: tau=0.5, beta=1.0, max_grad_norm=1.0 (defaults)

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

# IQL optimal hyperparameters
IQL_TAU=0.5
IQL_BETA=1.0
MAX_GRAD_NORM=1.0

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    
    # Create descriptive experiment name
    local EXP_NAME="iql_data${DATA_STEPS}_tau${IQL_TAU}_beta${IQL_BETA}_seed${SEED}"
    
    echo "Starting experiment: ${EXP_NAME} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --use_iql \
        --load_buffer "${BUFFER_PATH}" \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
        --iql_tau ${IQL_TAU} \
        --iql_beta ${IQL_BETA} \
        --max_grad_norm ${MAX_GRAD_NORM} \
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
echo "Starting IQL Experiments (8 data amounts on 8 GPUs)"
echo "============================================================"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "IQL tau (expectile): ${IQL_TAU}"
echo "IQL beta (temperature): ${IQL_BETA}"
echo "Max grad norm: ${MAX_GRAD_NORM}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "Loading buffer from: ${BUFFER_PATH}"
echo "============================================================"
echo ""
echo "IQL Key Ideas:"
echo "  1. V(s) trained with expectile regression to approximate max_a Q(s,a)"
echo "  2. Q(s,a) trained with TD using V(s') instead of max Q(s',a')"
echo "  3. Policy extracted with advantage-weighted regression (AWR)"
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
echo "All IQL experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
