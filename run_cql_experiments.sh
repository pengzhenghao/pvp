#!/bin/bash

# CQL Hyperparameter Search Script
# This script runs CQL experiments with different data amounts and hyperparameters
# across 8 GPUs in parallel

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"

# Common parameters
BC_TRAINING_TIMESTEPS=1000000000  # Train until manually stopped or converged
EVAL_FREQ=5000
N_EVAL_EPISODES=50
SAVE_FREQ=10000

# Function to run a single experiment
run_experiment() {
    local GPU_ID=$1
    local DATA_STEPS=$2
    local CQL_ALPHA=$3
    local CQL_TEMP=$4
    local NUM_RANDOM=$5
    local SEED=$6
    
    # Create descriptive experiment name
    local EXP_NAME="cql_data${DATA_STEPS}_alpha${CQL_ALPHA}_temp${CQL_TEMP}_nrand${NUM_RANDOM}_seed${SEED}"
    
    echo "Starting experiment: ${EXP_NAME} on GPU ${GPU_ID}"
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
        --exp_name "${EXP_NAME}" \
        --use_cql \
        --data_collection_timesteps ${DATA_STEPS} \
        --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
        --cql_alpha ${CQL_ALPHA} \
        --cql_temp ${CQL_TEMP} \
        --num_random_actions ${NUM_RANDOM} \
        --eval_freq ${EVAL_FREQ} \
        --n_eval_episodes ${N_EVAL_EPISODES} \
        --save_freq ${SAVE_FREQ} \
        --seed ${SEED} \
        > "${BASE_DIR}/logs/${EXP_NAME}.log" 2>&1 &
    
    echo "Experiment ${EXP_NAME} started with PID $!"
}

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# ============================================================
# Experiment Configuration
# ============================================================
# You can customize the experiments below

# Data collection amounts to test
DATA_STEPS_LIST=(5000 10000 20000 50000)

# CQL alpha values to test (conservative penalty weight)
CQL_ALPHA_LIST=(0.1 1.0 5.0)

# CQL temperature values
CQL_TEMP_LIST=(1.0)

# Number of random actions for CQL loss
NUM_RANDOM_LIST=(10)

# Seeds for reproducibility
SEED_LIST=(0)

# ============================================================
# Run experiments across 8 GPUs
# ============================================================

GPU_ID=0
MAX_GPUS=8

echo "============================================================"
echo "Starting CQL Hyperparameter Search"
echo "============================================================"
echo "Data steps: ${DATA_STEPS_LIST[@]}"
echo "CQL alpha: ${CQL_ALPHA_LIST[@]}"
echo "CQL temp: ${CQL_TEMP_LIST[@]}"
echo "Num random actions: ${NUM_RANDOM_LIST[@]}"
echo "Seeds: ${SEED_LIST[@]}"
echo "============================================================"

for DATA_STEPS in "${DATA_STEPS_LIST[@]}"; do
    for CQL_ALPHA in "${CQL_ALPHA_LIST[@]}"; do
        for CQL_TEMP in "${CQL_TEMP_LIST[@]}"; do
            for NUM_RANDOM in "${NUM_RANDOM_LIST[@]}"; do
                for SEED in "${SEED_LIST[@]}"; do
                    
                    run_experiment ${GPU_ID} ${DATA_STEPS} ${CQL_ALPHA} ${CQL_TEMP} ${NUM_RANDOM} ${SEED}
                    
                    # Move to next GPU
                    GPU_ID=$(( (GPU_ID + 1) % MAX_GPUS ))
                    
                    # If we've used all GPUs, wait for them to finish before continuing
                    if [ ${GPU_ID} -eq 0 ]; then
                        echo "All 8 GPUs are running experiments. Waiting for completion..."
                        wait
                        echo "Batch completed. Starting next batch..."
                    fi
                    
                done
            done
        done
    done
done

# Wait for any remaining experiments
echo "Waiting for remaining experiments to complete..."
wait

echo "============================================================"
echo "All CQL experiments completed!"
echo "Logs are saved in ${BASE_DIR}/logs/"
echo "============================================================"
