from pysmt.shortcuts import Symbol, And, Not, Or
from pysmt.solvers import Solver

# Define some symbols
x1 = Symbol("x1")
x2 = Symbol("x2")

# Create a logical formula (example)
formula = Or(And(Not(x1), x2), And(x1, Not(x2)))

# Solve using STP (if installed)
with Solver(name="stp") as solver:
    solver.add_assertion(formula)
    result = solver.solve()
    print(result)

