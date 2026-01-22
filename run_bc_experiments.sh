#!/bin/bash

killall python -9 2>/dev/null || true

# Pure BC (Behavioral Cloning) Experiment Script
# This script runs pure BC experiments with different data amounts
# Each data amount runs on a separate GPU (5 data amounts on 5 GPUs)
#
# Usage:
#   ./run_bc_experiments.sh        # Normal mode (eval_freq=100, 500 episodes)
#   ./run_bc_experiments.sh fast   # Fast mode (eval_freq=1000, 50 episodes)
#
# BC is the simplest baseline - just supervised learning on expert data
# No hyperparameters to tune, only data amount matters

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
    EVAL_FREQ=100
    N_EVAL_EPISODES=25
    SKIP_PRETRAIN_EVAL="--skip_pretrain_eval"
    WANDB_PROJECT="0121mainexp"
    echo "*** FAST MODE ENABLED ***"
else
    EVAL_FREQ=100
    N_EVAL_EPISODES=500
    SKIP_PRETRAIN_EVAL=""
    WANDB_PROJECT="0121mainexpfull"
fi

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    
    # Create descriptive experiment name
    local EXP_NAME="bc_data${DATA_STEPS}_seed${SEED}"
    
    echo "Starting experiment: ${EXP_NAME} on GPU ${GPU_ID}"
    
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --load_buffer "${BUFFER_PATH}" \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
        --eval_freq ${EVAL_FREQ} \
        --n_eval_episodes ${N_EVAL_EPISODES} \
        --save_freq ${SAVE_FREQ} \
        --seed ${SEED} \
        --crash_vehicle_penalty ${CRASH_VEHICLE_PENALTY} \
        --crash_object_penalty ${CRASH_OBJECT_PENALTY} \
        --out_of_road_penalty ${OUT_OF_ROAD_PENALTY} \
        ${SKIP_PRETRAIN_EVAL} \
        --wandb_project "${WANDB_PROJECT}" \
        > "${BASE_DIR}/logs/${EXP_NAME}.log" 2>&1 &
    
    echo "Experiment ${EXP_NAME} started with PID $!"
}

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# ============================================================
# Experiment Configuration
# ============================================================
# Data amounts depend on mode
if [ "$FAST_MODE" == "true" ]; then
    # Fast mode: only 2000 data steps on GPU 0
    DATA_STEPS_LIST=(2000)
else
    # Normal mode: 5 data amounts for 5 GPUs (from large to small)
    DATA_STEPS_LIST=(6000 5000 4000 3000 2000)
fi

# ============================================================
# Run experiments
# ============================================================

echo "============================================================"
echo "Starting Pure BC Experiments"
echo "============================================================"
echo "Mode: $([ "$FAST_MODE" == "true" ] && echo "FAST" || echo "NORMAL")"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "Loading buffer from: ${BUFFER_PATH}"
echo "============================================================"
echo ""
echo "Pure BC:"
echo "  - Supervised learning to imitate expert actions"
echo "  - No Q-learning, no conservative penalties"
echo "  - Simplest offline RL baseline"
echo "============================================================"

# Run experiments in parallel
for i in "${!DATA_STEPS_LIST[@]}"; do
    GPU_ID=$i
    DATA_STEPS=${DATA_STEPS_LIST[$i]}
    run_experiment ${GPU_ID} ${DATA_STEPS}
done

# Wait for all experiments to complete
echo "All ${#DATA_STEPS_LIST[@]} experiments started. Waiting for completion..."
wait

echo "============================================================"
echo "All Pure BC experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
