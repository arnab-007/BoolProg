#!/bin/bash
<<<<<<< HEAD
cbmc ./program_files/ex11.c --dimacs --slice-formula --outfile cnf-out
=======
cbmc ex5.c --dimacs --slice-formula --outfile cnf-out
>>>>>>> d24bd43a4e465d7cbddd210d89b87469967e9ddf
grep 'c ' cnf-out > var-mapping
python3 main.py
cryptominisat5 --verb 0 invariant_formula.cnf
