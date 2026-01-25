#!/bin/bash
# ============================================================
# TD3+BC Experiments - Adaptive GPU Version
# 自动适应可用的 GPU 数量
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
        AVAILABLE_GPUS=4
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
SAVE_FREQ=500
SEED=0

# Crash penalty parameters
CRASH_VEHICLE_PENALTY=5.0
CRASH_OBJECT_PENALTY=5.0
OUT_OF_ROAD_PENALTY=5.0

# TD3+BC optimal hyperparameter
TD3_BC_ALPHA=0.5

# Check for fast mode
FAST_MODE="false"
if [ "$1" == "fast" ]; then
    FAST_MODE="true"
    EVAL_FREQ=100
    N_EVAL_EPISODES=25
    SKIP_PRETRAIN_EVAL="--skip_pretrain_eval"
    WANDB_PROJECT="0121mainexp"
else
    EVAL_FREQ=500
    N_EVAL_EPISODES=200
    SKIP_PRETRAIN_EVAL=""
    WANDB_PROJECT="0122mainexpfull"
fi

# ============================================================
# Data sizes (from large to small, large first)
# ============================================================
if [ "$FAST_MODE" == "true" ]; then
    declare -a DATA_STEPS_LIST=(15000)
    TOTAL_DATA_SIZES=1
else
    declare -a DATA_STEPS_LIST=(15000 12500 10000 7500)
    TOTAL_DATA_SIZES=4
fi

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    
    local EXP_NAME="td3bc_data${DATA_STEPS}_alpha${TD3_BC_ALPHA}_seed${SEED}"
    
    echo "  GPU ${GPU_ID}: ${EXP_NAME}"
    
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --use_td3_bc \
        --load_buffer "${BUFFER_PATH}" \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
        --td3_bc_alpha ${TD3_BC_ALPHA} \
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
# Run experiments
# ============================================================

echo "============================================================"
echo "Starting TD3+BC Experiments (Adaptive GPU Mode)"
echo "============================================================"
echo "Mode: $([ "$FAST_MODE" == "true" ] && echo "FAST" || echo "NORMAL")"
echo "Available GPUs: ${AVAILABLE_GPUS}"
echo "Data sizes: ${DATA_STEPS_LIST[@]}"
echo "TD3+BC alpha: ${TD3_BC_ALPHA}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "============================================================"

if [ "$FAST_MODE" == "true" ]; then
    run_experiment 0 ${DATA_STEPS_LIST[0]}
    wait
    echo "TD3+BC fast mode experiment completed!"
else
    # Adaptive batching based on available GPUs
    data_idx=0
    while [ $data_idx -lt $TOTAL_DATA_SIZES ]; do
        REMAINING=$((TOTAL_DATA_SIZES - data_idx))
        BATCH_SIZE=$((REMAINING < AVAILABLE_GPUS ? REMAINING : AVAILABLE_GPUS))
        
        echo ""
        echo "Batch: Running ${BATCH_SIZE} experiments in parallel"
        
        for ((gpu=0; gpu<BATCH_SIZE; gpu++)); do
            DATA_STEPS=${DATA_STEPS_LIST[$data_idx]}
            run_experiment ${gpu} ${DATA_STEPS}
            data_idx=$((data_idx + 1))
        done
        
        echo "Waiting for batch to complete..."
        wait
        echo "Batch completed! (${data_idx}/${TOTAL_DATA_SIZES} done)"
    done
    
    echo ""
    echo "============================================================"
    echo "All ${TOTAL_DATA_SIZES} TD3+BC experiments completed!"
fi

echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
