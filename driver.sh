#!/bin/bash


cbmc ./program_files/ex12.c --dimacs --slice-formula --outfile cnf-out
grep 'c ' cnf-out > var-mapping
python3 main.py
cryptominisat5 --verb 0 invariant_formula.cnf

