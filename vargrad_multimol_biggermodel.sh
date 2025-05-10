for seed in 0 102 45 76 98 
do
    for num_conv_layers in 6 8 
    do 
        sbatch train.sh --seed $seed --num_conv_layers $num_conv_layers --nv 16 
done 