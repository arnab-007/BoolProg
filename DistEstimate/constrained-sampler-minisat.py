import subprocess
import random

def write_cnf_to_file(filename, num_vars, clauses):
    """
    Write the CNF formula to a file in DIMACS format.
    """
    with open(filename, 'w') as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

def run_minisat(cnf_file):
    """
    Run MiniSat on the provided CNF file and return the satisfying assignment.
    """
    result = subprocess.run(['minisat', cnf_file, 'result.out'], capture_output=True, text=True)
    
    if 'UNSAT' in result.stdout:
        return None
    
    with open('result.out', 'r') as f:
        for line in f:
            if line.startswith('v'):
                # Extract the variable assignments from the line starting with 'v'
                return [int(x) for x in line.split()[1:]]  # Skip 'v'

def sample_satisfying_assignments(num_samples, cnf_file, num_vars, cnf_clauses):
    """
    Generate satisfying assignments using MiniSat by sampling multiple times.
    """
    satisfying_assignments = []
    write_cnf_to_file(cnf_file, num_vars, cnf_clauses)

    for _ in range(num_samples):
        assignment = run_minisat(cnf_file)
        if assignment:
            satisfying_assignments.append(assignment)
        else:
            print("No more satisfying assignments.")
            break

    return satisfying_assignments

# Example CNF formula: (x1 || x2) && (!x1 || x3)
cnf_clauses = [[1, 2]]  # (x1 || x2) && (!x1 || x3)
num_vars = 3
num_samples = 50  # Number of samples to generate

# File to store the CNF formula in DIMACS format
cnf_filename = 'formula.cnf'

# Generate samples
satisfying_assignments = sample_satisfying_assignments(num_samples, cnf_filename, num_vars, cnf_clauses)

print("Satisfying Assignments:")
for assignment in satisfying_assignments:
    print(assignment)
