#!/bin/bash
cbmc ex4.c --dimacs --slice-formula --outfile cnf-out
grep 'c ' cnf-out > var-mapping
./cmsgen cnf-out.cnf
sort -n samples.out | uniq -c