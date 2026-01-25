#!/bin/bash
# ============================================================
# Fast Mode: Run all algorithms (BC, CQL, IQL, TD3+BC) in parallel
# Each algorithm runs on a separate GPU with data_steps=2000
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

# Training parameters (fast mode)
BC_TRAINING_TIMESTEPS=2000
DATA_STEPS=2000
EVAL_FREQ=100
N_EVAL_EPISODES=25
SAVE_FREQ=100
SEED=0

# Penalty parameters
CRASH_VEHICLE_PENALTY=5.0
CRASH_OBJECT_PENALTY=5.0
OUT_OF_ROAD_PENALTY=5.0

# IQL hyperparameters (IQL paper default)
IQL_TAU=0.7
IQL_BETA=3.0
MAX_GRAD_NORM=1.0

# TD3+BC hyperparameters
TD3_BC_ALPHA=2.5

# CQL hyperparameters
CQL_ALPHA=1.0
CQL_TEMP=1.0
NUM_RANDOM_ACTIONS=10

# ============================================================
# Run functions
# ============================================================

run_bc() {
    local GPU_ID=$1
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
        --skip_pretrain_eval \
        --wandb_project "0121mainexp" \
        > "${BASE_DIR}/logs/fast_${EXP_NAME}.log" 2>&1 &
}

run_td3bc() {
    local GPU_ID=$1
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
        --skip_pretrain_eval \
        --wandb_project "0121mainexp" \
        > "${BASE_DIR}/logs/fast_${EXP_NAME}.log" 2>&1 &
}

run_cql() {
    local GPU_ID=$1
    local EXP_NAME="cql_data${DATA_STEPS}_alpha${CQL_ALPHA}_seed${SEED}"
    
    echo "  GPU ${GPU_ID}: ${EXP_NAME}"
    
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
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
        --skip_pretrain_eval \
        --wandb_project "0121mainexp" \
        > "${BASE_DIR}/logs/fast_${EXP_NAME}.log" 2>&1 &
}

run_iql() {
    local GPU_ID=$1
    local EXP_NAME="iql_data${DATA_STEPS}_tau${IQL_TAU}_beta${IQL_BETA}_seed${SEED}"
    
    echo "  GPU ${GPU_ID}: ${EXP_NAME}"
    
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
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
        --skip_pretrain_eval \
        --wandb_project "0121mainexp" \
        > "${BASE_DIR}/logs/fast_${EXP_NAME}.log" 2>&1 &
}

# ============================================================
# Main
# ============================================================

# Create logs directory
mkdir -p ${BASE_DIR}/logs

echo "============================================================"
echo "Fast Mode: Running ALL Algorithms"
echo "============================================================"
echo "Data steps: ${DATA_STEPS}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "Buffer: ${BUFFER_PATH}"
echo "============================================================"
echo ""
echo "Algorithms:"
echo "  GPU 0: BC (pure behavioral cloning)"
echo "  GPU 1: TD3+BC (alpha=${TD3_BC_ALPHA})"
echo "  GPU 2: CQL (alpha=${CQL_ALPHA}, temp=${CQL_TEMP})"
echo "  GPU 3: IQL (tau=${IQL_TAU}, beta=${IQL_BETA})"
echo "============================================================"
echo ""

# Run all 4 algorithms in parallel on 4 GPUs
run_bc 0
run_td3bc 1
run_cql 2
run_iql 3

echo ""
echo "All 4 experiments started. Waiting for completion..."
echo "Logs: ${BASE_DIR}/logs/fast_*.log"
wait

echo ""
echo "============================================================"
echo "All fast mode experiments completed!"
echo "============================================================"
