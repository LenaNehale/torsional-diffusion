#!/bin/bash
for train_mode in diffusion on_policy mle
do
    for smi in 'CC1(C)Cc2[nH]c(=S)c(C#N)cc2CO1' 'Brc1cc2c(cc1Cn1c(-c3cncs3)nc3ccccc31)OCO2'  'CC(=O)N1CCN(c2ncnc3sc(-c4ccccc4)cc23)CC1'
    do
        for energy_fn in dummy mmff 
        do 
            sbatch train.sh --train_mode $train_mode --smi $smi --energy_fn $energy_fn
        done
    done
done