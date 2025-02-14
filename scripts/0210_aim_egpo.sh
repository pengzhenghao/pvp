#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)



# Loop over each GPU
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_aim_metadrive_fakehuman.py \
    --exp_name=AIM \
    --wandb \
    --wandb_project=AIM-ICML \
    --wandb_team=victorique \
    --seed=${seeds[$i]} \
    --intervention_start_stop_td=False \
    > "0210_aim_egpo_seed${seeds[$i]}.log" 2>&1 &
done

