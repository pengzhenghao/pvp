#!/bin/bash
#SBATCH --job-name=find_hard_expert
#SBATCH --output=./logs/find_hard_expert_%j.out
#SBATCH --error=./logs/find_hard_expert_%j.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# Create directories
mkdir -p ./logs
mkdir -p ./results/expert_difficulty

# Change to the pvp directory
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

# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Run with parallel mode (no debug output, use behavioral metrics)
python find_hard_scenarios_expert.py \
    --start_seed 1000 \
    --num_scenarios 1000 \
    --parallel \
    --num_envs 10 \
    --save_interval 50 \
    --output ./results/expert_difficulty

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
