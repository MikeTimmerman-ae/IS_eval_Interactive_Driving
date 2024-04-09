#!/bin/bash

###############################################################################
### README ####################################################################
# main script for batch training ego policies
###############################################################################

experiment=experiment_1

normal_mean=(
#    -0.5
#    0.5
#    1.5
    2.5
)

normal_std=(
#    0.5
#    0.75
    1.0
#    1.25
)

for i in "${!normal_mean[@]}"; do
    mean="${normal_mean[i]}"

    for k in "${!normal_std[@]}"; do
        std="${normal_std[k]}"

        ### train ego policy
        /usr/bin/python3 train_ego_with_trained_social.py --mean $mean --std  $std --experiment $experiment
    done
done
