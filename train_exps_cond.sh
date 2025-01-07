#!/bin/bash

for rew_temp in 10 
do
    for limit_train_mols in 10 100 1000 
    do
        sbatch train.sh --rew_temp $rew_temp  --limit_train_mols $limit_train_mols
    done     
done