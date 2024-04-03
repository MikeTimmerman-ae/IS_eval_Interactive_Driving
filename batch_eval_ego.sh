#!/bin/bash

###############################################################################
### README ####################################################################
# main script for batch evaluating ego policies
# 1. Specify ego policies to be evaluated (paths)
# 2. Specify naturalistic distribution (mu, sigma)
# 3. Specify evaluation distribution (mu, sigma)
###############################################################################

num_eval=5000
experiment=experiment_2

mean_naturalistic=2.0
std_naturalistic=0.5

eval_mean=(
    2.0
    0.0
)
eval_std=(
    0.5
    0.5
)

normal_mean=(
    -0.5
    0.5
    1.5
    2.5
)

normal_std=(
    0.5
    0.75
    1.0
    1.25
)

for i in "${!normal_mean[@]}"; do
    mean="${normal_mean[i]}"

    for k in "${!normal_std[@]}"; do
        std="${normal_std[k]}"

        for m in "${!eval_mean[@]}"; do
            mean_eval = "${eval_mean[k]}"
            std_eval = "${eval_std[k]}"

            ### evaluate ego policy
            /usr/bin/python3 ego_social_test.py --num_eval $num_eval \
                                                --model_dir "data/${experiment}/rl_ego_${mean}_${std}" \
                                                --experiment $experiment \
                                                --mean_naturalistic $mean_naturalistic \
                                                --std_naturalistic $std_naturalistic \
                                                --mean_eval $mean_eval \
                                                --std_eval $std_eval
        done
    done
done
