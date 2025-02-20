#!/bin/bash

# Define the seeds for each GPU
seeds=(0 100 0 100 0 100 0 100)



# Loop over each GPU
for i in {6..7}
do
    CUDA_VISIBLE_DEVICES=$i \
    nohup python /home/caihy/pvp/pvp/experiments/metadrive/train_pref_metadrive_fakehuman.py \
    --exp_name=cpl \
    --wandb_project=cpl \
    --wandb_team=victorique \
    --seed=${seeds[$i]} \
    --trial_name=negtraj \
    --poso=neg_observations \
    --posa=neg_actions_exp \
    --nego=neg_observations \
    --nega=neg_actions_nov \
    > "0219_predt_seed${seeds[$i]}.log" 2>&1 &
done

