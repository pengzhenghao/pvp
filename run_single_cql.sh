#!/bin/bash

# Single CQL Experiment Script
# Usage: ./run_single_cql.sh <GPU_ID> <DATA_STEPS> <CQL_ALPHA> [CQL_TEMP] [NUM_RANDOM] [SEED]
# Example: ./run_single_cql.sh 0 20000 1.0
# Example: ./run_single_cql.sh 1 50000 0.5 1.0 10 42

set -e

# Default values
GPU_ID=${1:-0}
DATA_STEPS=${2:-20000}
CQL_ALPHA=${3:-1.0}
CQL_TEMP=${4:-1.0}
NUM_RANDOM=${5:-10}
SEED=${6:-0}

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"

# Training parameters
BC_TRAINING_TIMESTEPS=100000
EVAL_FREQ=5000
N_EVAL_EPISODES=50
SAVE_FREQ=5000

# Create descriptive experiment name
EXP_NAME="cql_data${DATA_STEPS}_alpha${CQL_ALPHA}_temp${CQL_TEMP}_nrand${NUM_RANDOM}_seed${SEED}"

echo "============================================================"
echo "CQL Experiment Configuration"
echo "============================================================"
echo "GPU: ${GPU_ID}"
echo "Data collection steps: ${DATA_STEPS}"
echo "CQL alpha: ${CQL_ALPHA}"
echo "CQL temperature: ${CQL_TEMP}"
echo "Num random actions: ${NUM_RANDOM}"
echo "Seed: ${SEED}"
echo "Experiment name: ${EXP_NAME}"
echo "============================================================"

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# Run experiment
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
    --wandb_project "domain-adaptation"
