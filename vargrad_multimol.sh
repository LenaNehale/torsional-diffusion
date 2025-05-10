for seed in 0 102 45 76 98 
do
    for lr in 0.1 0.01 0.0001 0.00001
    do
        sbatch train.sh --seed $seed --lr $lr
    done
done 