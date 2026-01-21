#!/bin/bash

killall python -9 2>/dev/null || true

# IQL (Implicit Q-Learning) Experiment Script
# This script runs IQL experiments with ALL combinations of hyperparameters and data sizes
# 8 hyperparameter combinations × 5 data sizes = 40 experiments
# Runs in 5 rounds (8 experiments per round on 8 GPUs)
#
# Usage:
#   ./run_iql_experiments.sh        # Normal mode (eval_freq=100, 500 episodes)
#   ./run_iql_experiments.sh fast   # Fast mode (eval_freq=1000, 50 episodes)
#
# IQL paper: "Offline Reinforcement Learning with Implicit Q-Learning"

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
MAX_GRAD_NORM=1.0

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

# ============================================================
# Hyperparameter combinations (8 total)
# ============================================================
declare -a TAU_LIST=(0.5 0.7 0.7 0.8 0.9 0.7 0.8 0.9)
declare -a BETA_LIST=(1.0 1.0 3.0 3.0 3.0 10.0 10.0 10.0)

# Data sizes (5 total, from large to small)
declare -a DATA_STEPS_LIST=(6000 5000 4000 3000 2000)

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local IQL_TAU=$2
    local IQL_BETA=$3
    local DATA_STEPS=$4
    
    # Create descriptive experiment name
    local EXP_NAME="iql_data${DATA_STEPS}_tau${IQL_TAU}_beta${IQL_BETA}_seed${SEED}"
    
    echo "  GPU ${GPU_ID}: ${EXP_NAME}"
    
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
        --crash_vehicle_penalty ${CRASH_VEHICLE_PENALTY} \
        --crash_object_penalty ${CRASH_OBJECT_PENALTY} \
        --out_of_road_penalty ${OUT_OF_ROAD_PENALTY} \
        --wandb_project "mainexp0121" \
        > "${BASE_DIR}/logs/${EXP_NAME}.log" 2>&1 &
}

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# ============================================================
# Run all 40 experiments in 5 rounds
# ============================================================

echo "============================================================"
echo "Starting IQL Full Grid Search"
echo "8 hyperparameter combinations × 5 data sizes = 40 experiments"
echo "Running in 5 rounds (8 experiments per round on 8 GPUs)"
echo "============================================================"
echo "Mode: $([ "$FAST_MODE" == "true" ] && echo "FAST" || echo "NORMAL")"
echo "Data sizes: ${DATA_STEPS_LIST[@]}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "============================================================"
echo ""
echo "Hyperparameter combinations:"
echo "  0: tau=0.5, beta=1.0  (baseline)"
echo "  1: tau=0.7, beta=1.0"
echo "  2: tau=0.7, beta=3.0  (IQL paper default)"
echo "  3: tau=0.8, beta=3.0"
echo "  4: tau=0.9, beta=3.0  (for expert data)"
echo "  5: tau=0.7, beta=10.0"
echo "  6: tau=0.8, beta=10.0"
echo "  7: tau=0.9, beta=10.0 (most aggressive)"
echo "============================================================"

# Run experiments in 5 rounds (one round per data size)
for data_idx in "${!DATA_STEPS_LIST[@]}"; do
    DATA_STEPS=${DATA_STEPS_LIST[$data_idx]}
    ROUND=$((data_idx + 1))
    
    echo ""
    echo "============================================================"
    echo "Round ${ROUND}/5: data_steps=${DATA_STEPS}"
    echo "============================================================"
    
    # Run 8 experiments in parallel (one per GPU)
    for hp_idx in {0..7}; do
        TAU=${TAU_LIST[$hp_idx]}
        BETA=${BETA_LIST[$hp_idx]}
        run_experiment ${hp_idx} ${TAU} ${BETA} ${DATA_STEPS}
    done
    
    echo "Waiting for round ${ROUND} to complete..."
    wait
    echo "Round ${ROUND} completed!"
done

echo ""
echo "============================================================"
echo "All 40 IQL experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
