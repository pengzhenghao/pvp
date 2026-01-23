#!/bin/bash
# ============================================================
# Pure BC (Behavioral Cloning) Experiments
# 2 个 GPU，分两轮运行
# 第一轮: 15000, 10000
# 第二轮: 12500, 7500
# ============================================================

# ============================================================
# GPU Detection
# ============================================================
if [ -n "$SLURM_NUM_GPUS" ]; then
    AVAILABLE_GPUS=$SLURM_NUM_GPUS
elif [ -n "$SLURM_GPUS_ON_NODE" ]; then
    AVAILABLE_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    AVAILABLE_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$AVAILABLE_GPUS" -eq 0 ]; then
        AVAILABLE_GPUS=2
    fi
fi

echo "Detected $AVAILABLE_GPUS available GPU(s)"

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
SCRIPT="train_bc_metadrive_online.py"
BUFFER_PATH="${BASE_DIR}/data_buffer_20000.npz"

# Training parameters
BC_TRAINING_TIMESTEPS=2000
SAVE_FREQ=1000
SEED=0

# Crash penalty parameters
CRASH_VEHICLE_PENALTY=5.0
CRASH_OBJECT_PENALTY=5.0
OUT_OF_ROAD_PENALTY=5.0

# Check for fast mode
FAST_MODE="false"
if [ "$1" == "fast" ]; then
    FAST_MODE="true"
    EVAL_FREQ=100
    N_EVAL_EPISODES=25
    SKIP_PRETRAIN_EVAL="--skip_pretrain_eval"
    WANDB_PROJECT="0123mainexp"
else
    EVAL_FREQ=500
    N_EVAL_EPISODES=200
    SKIP_PRETRAIN_EVAL=""
    WANDB_PROJECT="0123mainexp"
fi

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    
    local EXP_NAME="bc_data${DATA_STEPS}_seed${SEED}"
    
    echo "  GPU ${GPU_ID}: ${EXP_NAME}"
    
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
}

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# ============================================================
# Run experiments - 分两轮
# ============================================================

echo "============================================================"
echo "Starting Pure BC Experiments"
echo "============================================================"
echo "Mode: $([ "$FAST_MODE" == "true" ] && echo "FAST" || echo "NORMAL")"
echo "Available GPUs: ${AVAILABLE_GPUS}"
echo "Round 1: 15000, 10000"
echo "Round 2: 12500, 7500"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "============================================================"

# Round 1: 15000, 10000
echo ""
echo "Round 1: data sizes 15000, 10000"
run_experiment 0 15000
run_experiment 1 10000
echo "Waiting for Round 1 to complete..."
wait
echo "Round 1 completed!"

# Round 2: 12500, 7500
echo ""
echo "Round 2: data sizes 12500, 7500"
run_experiment 0 12500
run_experiment 1 7500
echo "Waiting for Round 2 to complete..."
wait
echo "Round 2 completed!"

echo ""
echo "============================================================"
echo "All BC experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
