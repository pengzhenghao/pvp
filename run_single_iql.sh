#!/bin/bash

# Single IQL Experiment Script
# Usage: ./run_single_iql.sh <GPU_ID> <DATA_STEPS> <IQL_TAU> <IQL_BETA> [SEED] [MAX_GRAD_NORM]
# Example: ./run_single_iql.sh 0 20000 0.7 3.0
# Example: ./run_single_iql.sh 1 50000 0.8 10.0 42 1.0

set -e

# Default values
GPU_ID=${1:-0}
DATA_STEPS=${2:-20000}
IQL_TAU=${3:-0.7}
IQL_BETA=${4:-3.0}
SEED=${5:-0}
MAX_GRAD_NORM=${6:-1.0}

# Base directory
BASE_DIR="/home/caihy/pvp"
SCRIPT="train_bc_metadrive_online.py"

# Training parameters
BC_TRAINING_TIMESTEPS=1000000000
EVAL_FREQ=5000
N_EVAL_EPISODES=50
SAVE_FREQ=10000

# Create descriptive experiment name
EXP_NAME="iql_data${DATA_STEPS}_tau${IQL_TAU}_beta${IQL_BETA}_seed${SEED}"

echo "============================================================"
echo "IQL Experiment Configuration"
echo "============================================================"
echo "GPU: ${GPU_ID}"
echo "Data collection steps: ${DATA_STEPS}"
echo "IQL tau (expectile): ${IQL_TAU}"
echo "IQL beta (temperature): ${IQL_BETA}"
echo "Max grad norm: ${MAX_GRAD_NORM}"
echo "Seed: ${SEED}"
echo "Experiment name: ${EXP_NAME}"
echo "============================================================"
echo ""
echo "IQL Key Ideas:"
echo "  1. V(s) trained with expectile regression (tau=${IQL_TAU})"
echo "  2. Q(s,a) trained with TD using V(s')"
echo "  3. Policy extracted with AWR (beta=${IQL_BETA})"
echo ""
echo "  - tau=0.5: V = E[Q], no max approximation"
echo "  - tau=${IQL_TAU}: V approximates max_a Q(s,a)"
echo "  - beta=${IQL_BETA}: policy greediness"
echo "============================================================"

# Create logs directory
mkdir -p ${BASE_DIR}/logs

# Run experiment
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${BASE_DIR}/${SCRIPT} \
    --exp_name "${EXP_NAME}" \
    --use_iql \
    --data_collection_timesteps ${DATA_STEPS} \
    --bc_training_timesteps ${BC_TRAINING_TIMESTEPS} \
    --iql_tau ${IQL_TAU} \
    --iql_beta ${IQL_BETA} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --eval_freq ${EVAL_FREQ} \
    --n_eval_episodes ${N_EVAL_EPISODES} \
    --save_freq ${SAVE_FREQ} \
    --seed ${SEED}
