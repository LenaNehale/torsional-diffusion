for seed in 0 102 45 76 98
do
    for train_smis in  "C1C=CC[C@@H]2[C@@H]1C(=O)N(C2=O)SC(Cl)(Cl)C"  "COC=O"  "c1ccc2c(c1)C(=O)c3c(ccc(c3C2=O)N)N"  "C[C@@H]1CCCC[C@@H]1C"  "COc1ccccc1"   "CCC"
    do
        for lr in 0.1 0.01 0.0001 0.00001
        do
            sbatch train.sh --seed $seed --train_smis $train_smis --lr $lr
        done
    done
done
