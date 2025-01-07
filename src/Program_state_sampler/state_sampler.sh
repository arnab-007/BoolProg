#!/bin/bash



RANDOM_SEED=$RANDOM  # Alternatively, you can use: $(date +%s)

# Call CMSGen with the random seed
/usr/local/bin/cmsgen -s $RANDOM_SEED --samples=100 --samplefile=samples_plus.out ../input-cnf
/usr/local/bin/cmsgen -s $RANDOM_SEED --samples=100 --samplefile=samples_minus.out ../input-negated-cnf