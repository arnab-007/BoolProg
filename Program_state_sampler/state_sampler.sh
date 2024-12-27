#!/bin/bash



RANDOM_SEED=$RANDOM  # Alternatively, you can use: $(date +%s)

# Call CMSGen with the random seed
.././cmsgen/build/cmsgen -s $RANDOM_SEED --samples=100 --samplefile=samples_plus.out ../input-cnf
.././cmsgen/build/cmsgen -s $RANDOM_SEED --samples=100 --samplefile=samples_minus.out ../input-negated-cnf