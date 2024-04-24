#!/bin/bash

###############################################################################
### README ####################################################################
# main script for batch evaluating ego policies
# 1. Specify ego policies to be evaluated (paths)
# 2. Specify naturalistic distribution (mu, sigma)
# 3. Specify evaluation distribution (mu, sigma)
###############################################################################

num_eval=5000
experiment=experiment_1
naturalistic_eval=true        # True in case of naturalistic eval
IS_eval=true                 # True in case of importance sampling eval

########## Specify in case of naturalistic evaluation #########
naturalistic_dist=(
    kde_irl
    kde_InD1
#    kde_InD2
)
#################################################################

########## Specify in case of importance sampling evaluation #########
eval_mean=(
    1.8
    0.6
)
eval_std=(
    0.5
    0.5
)
######################################################

############ Ego policies to be evaluated ############
normal_mean=(
#   -0.5
#    0.5
    1.5
#    2.5
)

normal_std=(
    0.5
#    0.75
#    1.0
)
######################################################



for k in "${!normal_std[@]}"; do
    std="${normal_std[k]}"

    for i in "${!normal_mean[@]}"; do
        mean="${normal_mean[i]}"

        # Perform naturalistic evaluation of specified policies
        if $naturalistic_eval; then
            for m in "${!naturalistic_dist[@]}"; do
                  dist="${naturalistic_dist[m]}"

                  ### evaluate ego policy
                  /usr/bin/python3 ego_social_test.py --num_eval $num_eval \
                                                      --model_dir "data/${experiment}/rl_ego_normal-13" \
                                                      --eval_type "naturalistic" \
                                                      --naturalistic_dist $dist
#                                                      --model_dir "data/${experiment}/rl_ego_${mean}_${std}" \
            done
        fi

        # Perform importance sampling evaluation of specified policies
        if $IS_eval; then
            for m in "${!eval_mean[@]}"; do
                mean_eval="${eval_mean[m]}"
                std_eval="${eval_std[m]}"

                ### evaluate ego policy
                /usr/bin/python3 ego_social_test.py --num_eval $num_eval \
                                                    --model_dir "data/${experiment}/rl_ego_${mean}_${std}" \
                                                    --eval_type "IS" \
                                                    --mean_eval $mean_eval \
                                                    --std_eval $std_eval
            done
        fi
    done
done
