#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 200 300 400 500 600 700)



# Loop over each GPU
for i in {0..2}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python /home/caihy/pvp/pvp/experiments/metadrive/train_pref_metadrive_fakehuman.py \
    --exp_name=cpl \
    --wandb_project=cpl \
    --wandb_team=victorique \
    --seed=${seeds[$i]} \
    --trial_name=shah,snan \
    > "0219_cpl_seed${seeds[$i]}.log" 2>&1 &
done

