#!/bin/bash
#SBATCH --job-name=eval_hard
#SBATCH --output=./logs/eval_hard_%j.out
#SBATCH --error=./logs/eval_hard_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --exclude=bolei-gpu05,bolei-gpu03,bolei-gpu07

# Create logs directory
mkdir -p ./logs
mkdir -p ./results/hard_scenario_eval

cd "${SLURM_SUBMIT_DIR:-$(dirname $0)}"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

# Configuration
NUM_SEEDS=${1:-200}       # Default: all 200 seeds
NUM_ENVS=${2:-20}         # Default: 20 parallel envs (optimal speed)
MODEL=${3:-"both"}        # Default: evaluate both iql and td3

echo "Evaluating $MODEL model(s) on $NUM_SEEDS seeds with $NUM_ENVS parallel envs"

# Set display for headless rendering (EGL for GPU offscreen)
export SDL_VIDEODRIVER=offscreen
export PYOPENGL_PLATFORM=egl
export DISPLAY=

# Run parallel evaluation (unbuffered output)
# Output folder includes SLURM job ID to avoid overwriting
OUTPUT_DIR="./results/hard_scenario_eval_${SLURM_JOB_ID}"
mkdir -p ${OUTPUT_DIR}

PYTHONUNBUFFERED=1 python eval_hard_scenarios_parallel.py \
    --model ${MODEL} \
    --no_expert \
    --num_seeds ${NUM_SEEDS} \
    --num_envs ${NUM_ENVS} \
    --output ${OUTPUT_DIR} \
    --iql_checkpoint IQLBEST1.zip \
    --td3_checkpoint TD3BCBEST2.zip

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
