for seed in 0 102 45 76 98
do
    for p_expl in 0.2 
    do
        for p_replay in 0.2
        do
            sbatch train.sh --seed $seed --p_expl $p_expl  --p_replay $p_replay 
        done
    done
done