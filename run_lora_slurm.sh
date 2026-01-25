#!/bin/bash
#SBATCH --job-name="lora_exp"
#SBATCH --output="/p0/user/caihy/pvp/logs/slurm_%j.out"
#SBATCH --error="/p0/user/caihy/pvp/logs/slurm_%j.err"
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --time=120:00:00
#SBATCH --partition="all"

# ============================================================
# SLURM Script for LoRA Experiments
# ============================================================
# 用法:
#   1. 请求 8 个 GPU（默认，完整实验）:
#      sbatch run_lora_slurm.sh
#   
#   2. 请求更少的 GPU（脚本会自动适应）:
#      sbatch --gres=gpu:4 run_lora_slurm.sh
#   
#   3. 快速测试模式（只需 1 个 GPU）:
#      sbatch --gres=gpu:1 run_lora_slurm.sh fast
# ============================================================

# 获取分配的 GPU 数量
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # 计算 CUDA_VISIBLE_DEVICES 中的 GPU 数量
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    # 默认 8 个
    NUM_GPUS=8
fi

echo "============================================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Allocated GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "============================================================"

# 创建日志目录
mkdir -p /p0/user/caihy/pvp/logs

# 设置环境变量传递 GPU 数量给实验脚本
export SLURM_NUM_GPUS=$NUM_GPUS

# 运行改进版的实验脚本
bash /p0/user/caihy/pvp/run_lora_experiments_adaptive.sh "$@"

echo "============================================================"
echo "SLURM job completed!"
echo "============================================================"
