#!/bin/bash

for seed in 1 368 751
do
    for train_mode in on_policy
    do
        for smi in 'C=C(C)CSc1cc(C)c(C#N)c2nc3ccccc3n12' 'O=C(C=Cc1ccco1)OCc1nnc(-c2ccc(Cl)cc2)o1' 'CC1(C)OC2CC3C4CCC5=CC(=O)CCC5(C)C4(F)C(O)CC3(C)C2(C(=O)CCl)O1'  'CC(C)(C)c1[nH]nc2c1C(c1ccsc1)C(C#N)=C(N)O2' 'C=CCNc1cnn(CC(C)=O)c(=O)c1Br'
        do
            for energy_fn in mmff dummy 
            do 
                sbatch train.sh --train_mode $train_mode --smi $smi --energy_fn $energy_fn --seed $seed --num_points 30
            done
        done
    done
done