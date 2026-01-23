#!/bin/bash
# ============================================================
# Evaluate Baseline Models (Pretrained and Expert)
# Runs 2 experiments on 2 GPUs in parallel
# Uses EvalCallback with full metrics and uploads to wandb
# ============================================================

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${SCRIPT_DIR}"
SCRIPT="eval_baseline.py"

# Evaluation parameters
N_EVAL_EPISODES=200
SEED=1000
WANDB_PROJECT="0123mainexp"
WANDB_TEAM="victorique"

# Penalty parameters (same as other experiments)
CRASH_VEHICLE_PENALTY=5.0
CRASH_OBJECT_PENALTY=5.0
OUT_OF_ROAD_PENALTY=5.0

# Create logs directory
mkdir -p ${BASE_DIR}/logs

echo "============================================================"
echo "Baseline Model Evaluation"
echo "============================================================"
echo "Pretrained: RGB observations (domain A checkpoint)"
echo "Expert: Lidar observations (PPO 20M steps)"
echo "Episodes per model: ${N_EVAL_EPISODES}"
echo "Seed: ${SEED}"
echo "Wandb Project: ${WANDB_PROJECT}"
echo "============================================================"
echo ""

# Run pretrained model evaluation on GPU 0
echo "Starting PRETRAINED model evaluation on GPU 0..."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python ${BASE_DIR}/${SCRIPT} \
    --model pretrained \
    --n_eval_episodes ${N_EVAL_EPISODES} \
    --seed ${SEED} \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_team "${WANDB_TEAM}" \
    --log_dir "${BASE_DIR}" \
    --crash_vehicle_penalty ${CRASH_VEHICLE_PENALTY} \
    --crash_object_penalty ${CRASH_OBJECT_PENALTY} \
    --out_of_road_penalty ${OUT_OF_ROAD_PENALTY} \
    > "${BASE_DIR}/logs/eval_pretrained.log" 2>&1 &
PRETRAINED_PID=$!
echo "  PID: ${PRETRAINED_PID}"

# Run expert model evaluation on GPU 0 (same GPU, parallel)
echo "Starting EXPERT model evaluation on GPU 0..."
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 python ${BASE_DIR}/${SCRIPT} \
    --model expert \
    --n_eval_episodes ${N_EVAL_EPISODES} \
    --seed ${SEED} \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_team "${WANDB_TEAM}" \
    --log_dir "${BASE_DIR}" \
    --crash_vehicle_penalty ${CRASH_VEHICLE_PENALTY} \
    --crash_object_penalty ${CRASH_OBJECT_PENALTY} \
    --out_of_road_penalty ${OUT_OF_ROAD_PENALTY} \
    > "${BASE_DIR}/logs/eval_expert.log" 2>&1 &
EXPERT_PID=$!
echo "  PID: ${EXPERT_PID}"

echo ""
echo "Both evaluations started. Waiting for completion..."
echo "Logs:"
echo "  - ${BASE_DIR}/logs/eval_pretrained.log"
echo "  - ${BASE_DIR}/logs/eval_expert.log"
echo ""

# Wait for both to complete
wait ${PRETRAINED_PID}
PRETRAINED_STATUS=$?
wait ${EXPERT_PID}
EXPERT_STATUS=$?

echo "============================================================"
echo "Evaluation completed!"
echo "============================================================"
echo ""

# Print results summary
echo "=== PRETRAINED Model Results ==="
grep -A 20 "Evaluation Results" ${BASE_DIR}/logs/eval_pretrained.log 2>/dev/null || tail -30 ${BASE_DIR}/logs/eval_pretrained.log 2>/dev/null || echo "Log not found"
echo ""

echo "=== EXPERT Model Results ==="
grep -A 20 "Evaluation Results" ${BASE_DIR}/logs/eval_expert.log 2>/dev/null || tail -30 ${BASE_DIR}/logs/eval_expert.log 2>/dev/null || echo "Log not found"
echo ""

echo "============================================================"
echo "Results uploaded to wandb project: ${WANDB_PROJECT}"
echo "Full logs saved to ${BASE_DIR}/logs/"
echo "============================================================"
