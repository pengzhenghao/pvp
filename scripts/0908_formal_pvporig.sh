#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)



# Loop over each GPU
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python pvp/experiments/metadrive/train_pvp_metadrive_fakehuman.py \
    --exp_name=0908_pvporig_noadap_nobcloss_bs1024_seed \
    --bc_loss_weight=0.0 \
    --batch_size=1024 \
    --adaptive_batch_size=False \
    --wandb \
    --wandb_project=pvp2024 \
    --wandb_team=drivingforce \
    --seed=${seeds[$i]} \
    --free_level=0.95 \
    --save_freq=10000 \
    > "0908_pvporig_noadap_nobcloss_bs1024_seed${seeds[$i]}.log" 2>&1 &
done