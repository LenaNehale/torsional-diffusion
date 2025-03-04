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
'''

for train_mode in diffusion mle 
do
    sbatch train.sh  --train_mode $train_mode
done