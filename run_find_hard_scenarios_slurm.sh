#!/bin/bash
#SBATCH --job-name=find_hard_scenarios
#SBATCH --output=./logs/find_hard_scenarios_%j.out
#SBATCH --error=./logs/find_hard_scenarios_%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition="gpu04"

# Create logs directory if not exists
mkdir -p ./logs
mkdir -p ./results

# Change to the pvp directory
cd "${SLURM_SUBMIT_DIR:-$(dirname $0)}"

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=============================================="

# Activate conda environment (adjust path as needed)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pvp

# Set environment variables for parallel processing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run the parallel script
# Note: num_envs reduced from 25 to 10 to avoid CUDA OOM
python find_hard_scenarios_parallel.py \
    --start_seed 1000 \
    --num_scenarios 1000 \
    --num_trials 5 \
    --num_envs 10 \
    --save_interval 50 \
    --checkpoint ./pretrained.zip \
    --output ./results/hard_scenarios.json

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
