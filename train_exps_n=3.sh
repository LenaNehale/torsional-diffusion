#!/bin/bash

for seed in 1 368 751
do
    for train_mode in on_policy
    do
        for smi in 'CC(=O)n1c(=O)c2ccccc2c(=O)n1-c1ccccc1' 'CC1(C)CC(=O)C2=C(C1)NC(=O)NC2c1ccc(Cl)cc1' 'O=c1oc2ccccc2cc1-c1csc(Nc2nc3ccc(F)cc3s2)n1' 'CC(C)N1CC[NH2+]CC1' 'Brc1ccc(COc2ncnc3ccccc23)cc1'
        do
            for energy_fn in mmff dummy 
            do 
                sbatch train.sh --train_mode $train_mode --smi $smi --energy_fn $energy_fn --seed $seed --num_points 30
            done
        done
    done
done