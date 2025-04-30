#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)

filename=$(basename "$0")
extension="${filename##*.}"

EXP_NAME="${filename%.*}"


# Loop over each GPU
for i in {0..4}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_comb_metadrive_fakehuman.py \
    --exp_name=${EXP_NAME} \
    --wandb \
    --seed=${seeds[$i]} \
    > ${EXP_NAME}_seed${seeds[$i]}.log 2>&1 &
done
