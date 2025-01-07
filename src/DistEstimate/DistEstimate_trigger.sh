#!/bin/bash


#gcc ../executable_prog_files/Codex10.c
gcc ../../executable_prog_files/Codex9.c
#gcc ../executable_prog_files/Codex11.c
#gcc ../executable_prog_files/Codex12.c


RANDOM_SEED=$RANDOM  # Alternatively, you can use: $(date +%s)

# Call CMSGen with the random seed
/usr/local/bin/cmsgen -s $RANDOM_SEED --samples=600 --samplefile=samples.out ../../input-cnf
python3 DistEstimate.py