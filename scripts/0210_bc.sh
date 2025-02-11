#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)

filename=$(basename "$0")
extension="${filename##*.}"

EXP_NAME="${filename%.*}"


# Loop over each GPU
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_aim_metadrive_fakehuman.py \
    --exp_name=${EXP_NAME} \
    --wandb=True \
    --wandb_project=aim2025 \
    --wandb_team=victorique \
    --seed=${seeds[$i]} \
    --human_gated \
    --free_level=-100 \
    --name=BC \
    > ${EXP_NAME}_seed${seeds[$i]}.log 2>&1 &
done

