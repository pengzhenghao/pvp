#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 0 100 200 300)



# Loop over each GPU
for i in {5..8}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python /home/caihy/pvp/pvp/experiments/metadrive/train_pref_metadrive_fakehuman.py \
    --exp_name=see15cbias1 \
    --wandb_project=cpl \
    --wandb_team=victorique \
    --seed=${seeds[$i]} \
    --trial_name=see15cbias1 \
    --free_level=0.9 \
    --cbias=1 \
    > "0219_see5_seed${seeds[$i]}.log" 2>&1 &
done
