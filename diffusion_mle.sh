#!/bin/bash



for train_mode in diffusion mle 
do
    for train_smis in " CCS"  "C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)Cl"  "COc1ccccc1"  "CC(=C)c1ccccc1"  "CCc1cccc2c1cccc2 " 
    do
        for max_n_local_structures in 1 10
        do
            sbatch train.sh  --train_mode $train_mode --train_smis $train_smis --max_n_local_structures $max_n_local_structures
        done
    done
done