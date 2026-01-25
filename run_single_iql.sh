#!/bin/bash

# Single GPU IQL Experiment Script
# Runs all experiments for a specific GPU_ID (matching run_iql_experiments.sh)
#
# Usage: ./run_single_iql.sh <GPU_ID>
# Example: ./run_single_iql.sh 0   # Runs GPU 0 experiments (tau=0.5, beta=1.0)
# Example: ./run_single_iql.sh 5   # Runs GPU 5 experiments (tau=0.7, beta=10.0)
#
# GPU_ID -> Hyperparameter mapping:
#   0: tau=0.5, beta=1.0  (baseline)
#   1: tau=0.7, beta=1.0
#   2: tau=0.7, beta=3.0  (IQL paper default)
#   3: tau=0.8, beta=3.0
#   4: tau=0.9, beta=3.0  (for expert data)
#   5: tau=0.7, beta=10.0
#   6: tau=0.8, beta=10.0
#   7: tau=0.9, beta=10.0 (most aggressive)

set -e

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./run_single_iql.sh <GPU_ID>"
    echo "GPU_ID should be 0-7"
    exit 1
fi

GPU_ID=$1

# Validate GPU_ID
if [ "$GPU_ID" -lt 0 ] || [ "$GPU_ID" -gt 7 ]; then
    echo "Error: GPU_ID must be between 0 and 7"
    exit 1
fi

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"
BUFFER_PATH="/home/caihy/pvp/data_buffer_20000.npz"

# Common parameters (matching run_iql_experiments.sh)
BC_TRAINING_TIMESTEPS=2000
SAVE_FREQ=100
SEED=0
MAX_GRAD_NORM=1.0

# Crash penalty parameters
CRASH_VEHICLE_PENALTY=5.0
CRASH_OBJECT_PENALTY=5.0
OUT_OF_ROAD_PENALTY=5.0

# Eval parameters
EVAL_FREQ=100
N_EVAL_EPISODES=500

# Hyperparameter combinations (indexed by GPU_ID)
declare -a TAU_LIST=(0.5 0.7 0.7 0.8 0.9 0.7 0.8 0.9)
declare -a BETA_LIST=(1.0 1.0 3.0 3.0 3.0 10.0 10.0 10.0)

# Data sizes to run
declare -a DATA_STEPS_LIST=(6000 5000 4000 3000 2000)

# Get hyperparameters for this GPU
IQL_TAU=${TAU_LIST[$GPU_ID]}
IQL_BETA=${BETA_LIST[$GPU_ID]}

# Create logs directory
mkdir -p ${BASE_DIR}/logs

echo "============================================================"
echo "Single GPU IQL Experiments"
echo "============================================================"
echo "GPU: ${GPU_ID}"
echo "IQL tau (expectile): ${IQL_TAU}"
echo "IQL beta (temperature): ${IQL_BETA}"
echo "Data sizes: ${DATA_STEPS_LIST[@]}"
echo "Training timesteps: ${BC_TRAINING_TIMESTEPS}"
echo "Eval freq: ${EVAL_FREQ}"
echo "Eval episodes: ${N_EVAL_EPISODES}"
echo "============================================================"
echo ""

# Run experiments for all data sizes sequentially
for data_idx in "${!DATA_STEPS_LIST[@]}"; do
    DATA_STEPS=${DATA_STEPS_LIST[$data_idx]}
    ROUND=$((data_idx + 1))
    
    # Create descriptive experiment name
    EXP_NAME="iql_data${DATA_STEPS}_tau${IQL_TAU}_beta${IQL_BETA}_seed${SEED}"
    
    echo ""
    echo "============================================================"
    echo "Round ${ROUND}/5: ${EXP_NAME}"
    echo "============================================================"
    
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
        --wandb_project "0121mainexpfull" \
        2>&1 | tee "${BASE_DIR}/logs/${EXP_NAME}.log"
    
    echo "Round ${ROUND} completed!"
done

echo ""
echo "============================================================"
echo "All 5 experiments for GPU ${GPU_ID} completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
