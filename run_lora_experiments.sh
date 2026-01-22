#!/bin/bash
# ============================================================
# LoRA (Low-Rank Adaptation) Experiments
# Tests different LoRA hyperparameters: rank, alpha, dropout
# ============================================================

# Kill any existing python processes
killall python -9 2>/dev/null || true
sleep 2

# ============================================================
# Configuration
# ============================================================
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"
BUFFER_PATH="/home/caihy/pvp/data_buffer_20000.npz"

# Training parameters
BC_TRAINING_TIMESTEPS=2000
SAVE_FREQ=1000
SEED=0

# Penalty parameters (same as other scripts)
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
    WANDB_PROJECT="0121mainexp"
else
    EVAL_FREQ=100
    N_EVAL_EPISODES=500
    SKIP_PRETRAIN_EVAL=""
    WANDB_PROJECT="0121mainexpfull"
fi

# ============================================================
# LoRA Hyperparameter combinations
# ============================================================
if [ "$FAST_MODE" == "true" ]; then
    # Fast mode: single configuration for quick testing
    declare -a RANK_LIST=(4)
    declare -a ALPHA_LIST=(1.0)
    declare -a DROPOUT_LIST=(0.0)
    declare -a TARGET_LIST=("actor")
    declare -a DATA_STEPS_LIST=(2000)
else
    # Normal mode: grid search over hyperparameters
    # 8 combinations on 8 GPUs, then iterate over data sizes
    declare -a RANK_LIST=(2 4 8 16 2 4 8 16)
    declare -a ALPHA_LIST=(1.0 1.0 1.0 1.0 4.0 4.0 4.0 4.0)
    declare -a DROPOUT_LIST=(0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)
    declare -a TARGET_LIST=("actor" "actor" "actor" "actor" "actor" "actor" "actor" "actor")
    declare -a DATA_STEPS_LIST=(6000 5000 4000 3000 2000)
fi

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local LORA_RANK=$2
    local LORA_ALPHA=$3
    local LORA_DROPOUT=$4
    local LORA_TARGET=$5
    local DATA_STEPS=$6
    
    # Create descriptive experiment name
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

echo "============================================================"
if [ "$FAST_MODE" == "true" ]; then
    echo "Starting LoRA Fast Mode (single experiment)"
else
    echo "Starting LoRA Full Grid Search"
    echo "8 hyperparameter combinations × ${#DATA_STEPS_LIST[@]} data sizes"
    echo "Running in ${#DATA_STEPS_LIST[@]} rounds (8 experiments per round on 8 GPUs)"
fi
echo "============================================================"
echo "Mode: $([ "$FAST_MODE" == "true" ] && echo "FAST" || echo "NORMAL")"
echo "Data sizes: ${DATA_STEPS_LIST[@]}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "============================================================"

if [ "$FAST_MODE" == "true" ]; then
    # Fast mode: single experiment
    echo "Running single LoRA experiment (rank=4, alpha=1.0, dropout=0.0, target=actor)..."
    run_experiment 0 ${RANK_LIST[0]} ${ALPHA_LIST[0]} ${DROPOUT_LIST[0]} ${TARGET_LIST[0]} ${DATA_STEPS_LIST[0]}
    wait
    echo "LoRA fast mode experiment completed!"
else
    # Normal mode: full grid search
    echo ""
    echo "Hyperparameter combinations:"
    echo "  0: rank=2, alpha=1.0 (smallest)"
    echo "  1: rank=4, alpha=1.0 (default)"
    echo "  2: rank=8, alpha=1.0"
    echo "  3: rank=16, alpha=1.0 (largest rank)"
    echo "  4: rank=2, alpha=4.0"
    echo "  5: rank=4, alpha=4.0"
    echo "  6: rank=8, alpha=4.0"
    echo "  7: rank=16, alpha=4.0 (largest)"
    echo "============================================================"

    # Run experiments in rounds (one round per data size)
    for data_idx in "${!DATA_STEPS_LIST[@]}"; do
        DATA_STEPS=${DATA_STEPS_LIST[$data_idx]}
        ROUND=$((data_idx + 1))
        
        echo ""
        echo "============================================================"
        echo "Round ${ROUND}/${#DATA_STEPS_LIST[@]}: data_steps=${DATA_STEPS}"
        echo "============================================================"
        
        # Run 8 experiments in parallel (one per GPU)
        for hp_idx in {0..7}; do
            RANK=${RANK_LIST[$hp_idx]}
            ALPHA=${ALPHA_LIST[$hp_idx]}
            DROPOUT=${DROPOUT_LIST[$hp_idx]}
            TARGET=${TARGET_LIST[$hp_idx]}
            run_experiment ${hp_idx} ${RANK} ${ALPHA} ${DROPOUT} ${TARGET} ${DATA_STEPS}
        done
        
        echo "Waiting for round ${ROUND} to complete..."
        wait
        echo "Round ${ROUND} completed!"
    done
    
    echo ""
    echo "============================================================"
    echo "All LoRA experiments completed!"
fi

echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
