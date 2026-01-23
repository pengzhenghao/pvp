#!/bin/bash
# ============================================================
# LoRA (Low-Rank Adaptation) Experiments
# ============================================================
#
# 运行策略：
#   - 每个 GPU 跑 1 个程序
#   - 每轮运行同一超参数配置的 4 个 data size
#   - 先看 data size 影响，再看超参数影响
#
# 超参数配置（8种）:
#   1. rank=16, alpha=1.0
#   2. rank=8,  alpha=1.0
#   3. rank=4,  alpha=1.0
#   4. rank=2,  alpha=1.0
#   5. rank=16, alpha=4.0
#   6. rank=8,  alpha=4.0
#   7. rank=4,  alpha=4.0
#   8. rank=2,  alpha=4.0
#
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

# ============================================================
# LoRA Hyperparameter configurations (8 种配置)
# ============================================================
declare -a RANK_LIST=(16 8 4 2 16 8 4 2)
declare -a ALPHA_LIST=(1.0 1.0 1.0 1.0 4.0 4.0 4.0 4.0)
declare -a DROPOUT_LIST=(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
declare -a TARGET_LIST=("actor" "actor" "actor" "actor" "actor" "actor" "actor" "actor")
TOTAL_HP_COMBINATIONS=8

# Data sizes
declare -a DATA_STEPS_LIST=(15000 12500 10000 7500)
TOTAL_DATA_SIZES=4

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local LORA_RANK=$2
    local LORA_ALPHA=$3
    local LORA_DROPOUT=$4
    local LORA_TARGET=$5
    local DATA_STEPS=$6
    
    local EXP_NAME="lora_data${DATA_STEPS}_r${LORA_RANK}_a${LORA_ALPHA}_d${LORA_DROPOUT}_${LORA_TARGET}_seed${SEED}"
    
    echo "  GPU ${GPU_ID}: ${EXP_NAME}"
    
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --use_lora \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lora_target ${LORA_TARGET} \
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
# Run experiments
# ============================================================

TOTAL_EXPERIMENTS=$((TOTAL_HP_COMBINATIONS * TOTAL_DATA_SIZES))

echo "============================================================"
echo "Starting LoRA Experiments"
echo "============================================================"
echo "Mode: $([ "$FAST_MODE" == "true" ] && echo "FAST" || echo "NORMAL")"
echo "Available GPUs: ${AVAILABLE_GPUS}"
echo "Total experiments: ${TOTAL_EXPERIMENTS} (8 HP configs × 4 data sizes)"
echo ""
echo "Strategy: 每轮运行 1 个超参数配置的 4 个 data size"
echo "  - 先看 data size 影响，再看超参数影响"
echo "  - 每个 GPU 跑 1 个程序"
echo ""
echo "Data sizes: ${DATA_STEPS_LIST[@]}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "============================================================"

if [ "$FAST_MODE" == "true" ]; then
    run_experiment 0 ${RANK_LIST[0]} ${ALPHA_LIST[0]} ${DROPOUT_LIST[0]} ${TARGET_LIST[0]} ${DATA_STEPS_LIST[0]}
    wait
    echo "LoRA fast mode experiment completed!"
else
    # 外循环：超参数配置
    for hp_idx in $(seq 0 $((TOTAL_HP_COMBINATIONS - 1))); do
        RANK=${RANK_LIST[$hp_idx]}
        ALPHA=${ALPHA_LIST[$hp_idx]}
        DROPOUT=${DROPOUT_LIST[$hp_idx]}
        TARGET=${TARGET_LIST[$hp_idx]}
        
        echo ""
        echo "============================================================"
        echo "Round $((hp_idx + 1))/${TOTAL_HP_COMBINATIONS}: rank=${RANK}, alpha=${ALPHA}"
        echo "============================================================"
        
        # 内循环：data sizes，分批次运行
        data_idx=0
        while [ $data_idx -lt $TOTAL_DATA_SIZES ]; do
            REMAINING=$((TOTAL_DATA_SIZES - data_idx))
            BATCH_SIZE=$((REMAINING < AVAILABLE_GPUS ? REMAINING : AVAILABLE_GPUS))
            
            echo ""
            echo "Starting batch: data_idx=${data_idx}, batch_size=${BATCH_SIZE}"
            
            for ((gpu=0; gpu<BATCH_SIZE; gpu++)); do
                DATA_STEPS=${DATA_STEPS_LIST[$((data_idx + gpu))]}
                run_experiment ${gpu} ${RANK} ${ALPHA} ${DROPOUT} ${TARGET} ${DATA_STEPS}
            done
            
            echo "Waiting for batch to complete..."
            wait
            echo "Batch completed!"
            
            data_idx=$((data_idx + BATCH_SIZE))
        done
        
        echo ""
        echo "Round $((hp_idx + 1)) (rank=${RANK}, alpha=${ALPHA}) completed!"
    done
    
    echo ""
    echo "============================================================"
    echo "All ${TOTAL_EXPERIMENTS} LoRA experiments completed!"
fi

echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
