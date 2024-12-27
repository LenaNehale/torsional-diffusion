#!/bin/bash

for seed in 1 368 751
do
    for train_mode in on_policy
    do    
        for energy_fn in mmff  
        do 
            sbatch train.sh --train_mode $train_mode --energy_fn $energy_fn --seed $seed --num_points 30 --batch_size_train 4 --batch_size_eval 128
        done
    done
done