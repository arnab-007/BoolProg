#!/bin/bash

# Generate a random seed using $RANDOM or date
RANDOM_SEED=$RANDOM  # Alternatively, you can use: $(date +%s)

# Call CMSGen with the random seed
/usr/local/bin/cmsgen -s $RANDOM_SEED --samples=500 --samplefile=samples.out ../candidate-cnf

# Print the seed for reproducibility (optional)
echo "Used random seed: $RANDOM_SEED"


#./cmsgen/build/cmsgen -s $RANDOM_SEED --samples=600 --samplefile=samples.out input-cnf