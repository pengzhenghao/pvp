#!/bin/bash

killall python -9 2>/dev/null || true

# CQL (Conservative Q-Learning) Experiment Script
# This script runs CQL experiments with different data amounts
# Each data amount runs on a separate GPU (5 data amounts on 5 GPUs)
#
# Usage:
#   ./run_cql_experiments.sh        # Normal mode (eval_freq=100, 500 episodes)
#   ./run_cql_experiments.sh fast   # Fast mode (eval_freq=1000, 50 episodes)
#
# Using optimal hyperparameters found: alpha=10.0, temp=1.0, num_random=10 (defaults)

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"
BUFFER_PATH="/home/caihy/pvp/data_buffer_20000.npz"

# Check for fast mode
FAST_MODE=false
if [ "$1" == "fast" ]; then
    FAST_MODE=true
fi

# Common parameters
BC_TRAINING_TIMESTEPS=2000
SAVE_FREQ=1000
SEED=0

# Crash penalty parameters (directly affect reward)
CRASH_VEHICLE_PENALTY=5.0
CRASH_OBJECT_PENALTY=5.0
OUT_OF_ROAD_PENALTY=5.0

# Mode-specific parameters
if [ "$FAST_MODE" == "true" ]; then
    EVAL_FREQ=1000
    N_EVAL_EPISODES=50
    echo "*** FAST MODE ENABLED ***"
else
    EVAL_FREQ=100
    N_EVAL_EPISODES=500
fi

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
        --load_buffer "${BUFFER_PATH}" \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
        --cql_alpha ${CQL_ALPHA} \
        --cql_temp ${CQL_TEMP} \
        --num_random_actions ${NUM_RANDOM_ACTIONS} \
        --eval_freq ${EVAL_FREQ} \
        --n_eval_episodes ${N_EVAL_EPISODES} \
        --save_freq ${SAVE_FREQ} \
        --seed ${SEED} \
        --crash_vehicle_penalty ${CRASH_VEHICLE_PENALTY} \
        --crash_object_penalty ${CRASH_OBJECT_PENALTY} \
        --out_of_road_penalty ${OUT_OF_ROAD_PENALTY} \
        --wandb_project "mainexp0121" \
        > "${BASE_DIR}/logs/${EXP_NAME}.log" 2>&1 &
    
    echo "Experiment ${EXP_NAME} started with PID $!"
}

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# ============================================================
# Experiment Configuration
# ============================================================
# 5 data amounts for 5 GPUs (from large to small)
DATA_STEPS_LIST=(6000 5000 4000 3000 2000)

# ============================================================
# Run experiments across 5 GPUs
# ============================================================

echo "============================================================"
echo "Starting CQL Experiments (5 data amounts on 5 GPUs)"
echo "============================================================"
echo "Mode: $([ "$FAST_MODE" == "true" ] && echo "FAST" || echo "NORMAL")"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "CQL alpha: ${CQL_ALPHA}"
echo "CQL temp: ${CQL_TEMP}"
echo "Num random actions: ${NUM_RANDOM_ACTIONS}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "Loading buffer from: ${BUFFER_PATH}"
echo "============================================================"
echo ""
echo "CQL adds conservative penalty to prevent Q-value overestimation"
echo "============================================================"

# Run all 5 experiments in parallel on 5 GPUs
for i in "${!DATA_STEPS_LIST[@]}"; do
    GPU_ID=$i
    DATA_STEPS=${DATA_STEPS_LIST[$i]}
    run_experiment ${GPU_ID} ${DATA_STEPS}
done

# Wait for all experiments to complete
echo "All 5 experiments started. Waiting for completion..."
wait

echo "============================================================"
echo "All CQL experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
