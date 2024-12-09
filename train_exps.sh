#!/bin/bash

for seed in 1 383 749 1092 76
do
    for train_mode in on_policy
    do
        for smi in "Brc1cc2c(cc1Cn1c(-c3cncs3)nc3ccccc31)OCO2" "Brc1ccc(-c2nc(NN=Cc3cccnc3)c3ccccc3n2)cc1" "C(=NNc1nc(Nc2ccccc2)nc(N2CCOCC2)n1)c1c[nH]c2ccccc12"
        do
            for energy_fn in dummy
            do 
                for p_replay in 0.0 0.3 0.5
                do
                sbatch train.sh --train_mode $train_mode --smi $smi --energy_fn $energy_fn --p_replay $p_replay --seed $seed
                done
            done
        done
    done
done