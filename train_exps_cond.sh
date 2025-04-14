#!/bin/bash

'''
#for smis in "c1ccc(cc1)OC=O"   "CCC(=O)OC"    "CCCCCC(=O)OC"  "CCOP(=O)(OCC)OCC"   "CCCCN(CC)C(=O)SCCC"
for p_expl in 0.1 0.2 0.5 0.8
do 
    for p_replay in 0.0 0.1 0.2
    do
        for diffusion_steps in 20 40 
        do
            sbatch train.sh  --p_expl $p_expl --p_replay $p_replay --diffusion_steps $diffusion_steps
        done
    done     
done


for train_mode in gflownet
do
    for max_n_local_structures in 1 10 100
    do
        sbatch train.sh  --train_mode $train_mode --max_n_local_structures $max_n_local_structures
    done
done

'''

for seed in 0 102 45 76 98
do
    for p_expl in 0.2 
    do
        for lr in 1e-3
        do
            for train_smis in "C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)Cl" 
            do
                for rew_temp in 10 100
                do
                    sbatch train.sh --seed $seed --p_expl $p_expl  --lr $lr --train_smis $train_smis --rew_temp $rew_temp
                done
            done
        done
    done
done
