#!/bin/bash

#for smis in "c1ccc(cc1)OC=O"   "CCC(=O)OC"    "CCCCCC(=O)OC"  "CCOP(=O)(OCC)OCC"   "CCCCN(CC)C(=O)SCCC"
#do
for p_expl in 0.0 0.2
do 
    for p_replay in 0.0
    do
        sbatch train.sh --p_expl $p_expl --p_replay $p_replay
        
    done     
done