import subprocess
import random
import re
import os
import json
CURRENT_PATH = os.path.realpath("")
PATH = os.path.dirname(CURRENT_PATH)
assumed_shape = " "
'''
"""
[get_config] loads the json object in [progname]'s configuration file
"""
def get_config(progname):
    with open(os.path.join(CURRENT_PATH, "candidate_files", progname + ".json"), "r") as f:
        config = json.load(f)
    return config


with open(os.path.join(CURRENT_PATH, "program-list.txt"), "r") as f:
    prognames = f.read().strip().split("\n")
#print(prognames)

for progname in prognames:
    config = get_config(progname)
    prog_variables = config["Program_variables"]["Bools"]
    cand = config["Candidate"]["Expression"]
    init_states = config["Initial states"]["Expression"]


    
def sample_to_decimal(sample):
    # Create a 12-bit binary string with all bits initially set to '0'
    binary_string = ['0'] * 8
    
    for value in sample:
        if value == 0:
            continue  # Skip the terminating zero
        index = abs(value) - 1  # Convert variable to zero-based index
        if value > 0:
            binary_string[index] = '1'  # Set bit to 1 for positive values
        else:
            binary_string[index] = '0'  # Set bit to 0 for negative values
    
    # Join the bits to form a binary string and convert to decimal
    binary_str = ''.join(binary_string)
    return int(binary_str, 2)

# Read samples from the file and convert each sample to decimal
decimals = []
with open('./DistEstimate/samples.out', 'r') as f:
    for line in f:
        # Convert the line into a list of integers
        sample = list((map(int, line.split())))
        decimal_value = sample_to_decimal(sample)
        decimals.append(decimal_value)

decimals = sorted(list(set(decimals)))
print(decimals)
'''
def read_dimacs_file(file_path):
    clauses = []
    num_variables = 0
    num_clauses = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments
            #if line.startswith('c'):
                #continue
            # Read problem line (e.g. "p cnf 8 4")
            if line.startswith('p'):
                _, _, num_vars, num_clauses = line.split()
                num_variables = int(num_vars)
                num_clauses = int(num_clauses)
            # Read clauses
            else:
                clause = list(map(int, line.split()))[:-1]  # remove the last zero
                print(clause)
                clauses.append(clause)
    return num_variables, clauses


def remove_duplicate_literals(clause):
    cleaned_clause = []
    for literal in clause:
        negated_literal = -literal
        if negated_literal not in clause:
            cleaned_clause.append(literal)
    return cleaned_clause


import itertools

def DNF_to_CNF(dnf_terms):
    # Step 1: Generate all pairs of terms from DNF and take the conjunction of every literal in the pair
    cnf_clauses = []
    for literals in itertools.product(*dnf_terms):
        clause = set()
        for literal in literals:
            # Add the literal to clause if it doesn't conflict with an opposite literal
            if -literal in clause:
                break  # Skip conflicting combinations
            clause.add(literal)
        else:
            # Only add non-conflicting clauses
            cnf_clauses.append(sorted(clause, key=abs))
    
    # Step 2: Remove duplicate clauses
    unique_clauses = []
    seen = set()
    for clause in cnf_clauses:
        clause_tuple = tuple(sorted(clause, key=abs))
        if clause_tuple not in seen:
            unique_clauses.append(clause)
            seen.add(clause_tuple)
    
    return unique_clauses

def negate_clause(clause):
    # Negate each literal in the clause
    return [-lit for lit in clause]

def cnf_to_dimacs(num_vars, clauses, output_file_path):
    with open(output_file_path, 'w') as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(map(str, clause)) + " 0\n")

def negate_cnf(num_vars, clauses):
    # Negate the entire CNF formula (Apply De Morgan's Law)
    L = [negate_clause(clause) for clause in clauses]
    
    return L

def negate_cnf_to_dimacs(input_file_path, output_file_path):
    num_vars, clauses = read_dimacs_file(input_file_path)
    
    negated_clauses = negate_cnf(num_vars, clauses)
    
    negated_clauses = process_and_remove_duplicates(DNF_to_CNF(negated_clauses))
    
    cnf_to_dimacs(num_vars, negated_clauses, output_file_path)

def process_and_remove_duplicates(L):
    # Step 1: Sort each inner list by absolute values
    L_sorted = [sorted(inner_list, key=abs) for inner_list in L]
    
    # Step 2: Remove duplicates by converting to a set of tuples, then back to a list of lists
    L_unique = list(map(list, set(tuple(sublist) for sublist in L_sorted)))
    
    # Step 3: Sort the outer list for consistent order
    L_unique.sort()
    
    return L_unique




# Example usage:
input_file = "../input-cnf"  # Path to the input CNF file
output_file = "../input-negated-cnf"  # Path to the output CNF file for negation
negate_cnf_to_dimacs(input_file, output_file)

