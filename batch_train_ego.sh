#!/bin/bash

###############################################################################
### README ####################################################################
# main script for batch training ego policies
# step 1:
###############################################################################

experiment=experiment_1

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

        ### train ego policy
        python test.py --mean $mean --std  $std --experiment=experiment_1
    done
done
