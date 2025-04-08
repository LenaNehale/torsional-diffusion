for seed in 0 102 45 76 98
do
    for p_expl in 0.2 
    do
        for lr in 1e-2 5e-3 1e-3 1e-4
        do
            sbatch train.sh --seed $seed --p_expl $p_expl  --lr $lr      
        done
    done
done