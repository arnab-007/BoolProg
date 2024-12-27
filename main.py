import pandas as pd
import os
import time
from datetime import datetime
import sys
import json
import argparse
import copy
from to_dimacs import generate_init_DIMACS_formula,generate_final_DIMACS_formula
from list_from_dimacs import extract_formula_from_DIMACS



PATH = os.path.realpath("")
assumed_shape = " "



"""
[get_config] loads the json object in [progname]'s configuration file
"""
def get_config(progname):
    with open(os.path.join(PATH, "candidate_files", progname + ".json"), "r") as f:
        config = json.load(f)
    return config


with open(os.path.join(PATH, "program-list.txt"), "r") as f:
    prognames = f.read().strip().split("\n")
print(prognames)


def generate_DIMACS_formula(clauses):
    num_variables = len(set(abs(literal) for clause in clauses for literal in clause))
    num_clauses = len(clauses)

    dimacs_str = f"p cnf {num_variables} {num_clauses}\n"
    
    for clause in clauses:
        dimacs_str += ' '.join(str(i) for i in clause) + ' 0\n'
    
    return dimacs_str


def remove_duplicate_literals(clause):
    cleaned_clause = []
    for literal in clause:
        negated_literal = -literal
        if negated_literal not in clause:
            cleaned_clause.append(literal)
    return cleaned_clause

<<<<<<< HEAD

from sympy.logic.boolalg import Or, And, Not, to_cnf
from sympy import symbols

def parse_dnf_to_expr(dnf_formula):
    """
    Converts a list of DNF terms to a symbolic DNF expression.
    """
    terms = []
    for term in dnf_formula:
        literals = []
        for literal in term:
            if literal < 0:
                literals.append(Not(symbols(f"x{abs(literal)}")))
            else:
                literals.append(symbols(f"x{literal}"))
        terms.append(And(*literals))
    return Or(*terms)

def DNF_to_CNF(dnf_formula):
    """
    Converts a list of DNF terms to a CNF formula using symbolic manipulation.
    """
    # Convert DNF formula into a symbolic expression
    dnf_expr = parse_dnf_to_expr(dnf_formula)

    # Use the global to_cnf function to convert to CNF
    cnf_expr = to_cnf(dnf_expr, simplify=True)

    # Extract CNF clauses as a list of lists
    cnf_clauses = []
    for clause in cnf_expr.args if isinstance(cnf_expr, And) else [cnf_expr]:
        cnf_clause = []
        for literal in clause.args if isinstance(clause, Or) else [clause]:
            if literal.func == Not:
                cnf_clause.append(-int(str(literal.args[0])[1:]))  # Convert ~xN to -N
            else:
                cnf_clause.append(int(str(literal)[1:]))  # Convert xN to N
        cnf_clauses.append(cnf_clause)

    return cnf_clauses


def DNF_to_CNF_old(dnf_formula):
=======
def DNF_to_CNF(dnf_formula):
>>>>>>> d24bd43a4e465d7cbddd210d89b87469967e9ddf
    cnf_formula = []

    # If DNF formula is empty, return empty CNF formula
    if not dnf_formula:
        return cnf_formula

    # Initialize CNF formula with the first clause of DNF
    cnf_formula = [[literal] for literal in dnf_formula[0]]
    
    # Distribute each subsequent clause over the existing CNF formula
    for i in range(1, len(dnf_formula)):
        new_cnf_formula = []
        for clause in cnf_formula:
            for literal in dnf_formula[i]:
                
                new_clause = clause + [literal]
                
                new_cnf_formula.append(list(set(new_clause)))
        cnf_formula = new_cnf_formula

    # Remove duplicate literals (both x and -x) within the same clause
    cnf_formula = [remove_duplicate_literals(clause) for clause in cnf_formula]
    cnf_formula = [elem for elem in cnf_formula if elem]
    return cnf_formula


with open('var-mapping') as f:
    lines = f.readlines() # list containing lines of file
    columns = [] # To store column names


map = lines[[ i for i, word in enumerate(lines) if word.startswith('c ') ][0]:]


variable_mapping = {}
guard_mapping = {}

for line in lines:
    parts = line.split(' ')
    if line.startswith('c main::'):
        variable_parts = parts[1].split('::')[2].split('!')

for line in map:
    parts = line.split()
    if line.startswith('c goto_symex::\\guard#'):
      guard_number = parts[1].split('::')[1].split('\\')[1]
      guard_mapping[f'{guard_number}'] = int(parts[-1])
    elif line.startswith('c main::'):
      variable_parts = parts[1].split('::')[2].split('!') # Extract variable name
      variable_name, instance = variable_parts[0],variable_parts[1].split('#')[1]
      variable_name = f"{variable_name}_{instance}"
      variable_mapping[f"{variable_name}"] = int(parts[-1])

print(variable_mapping)
max_indices = {}

for key in variable_mapping.keys():
    prefix, index = key.split('_')
    index = int(index)
    
    if prefix not in max_indices or index > max_indices[prefix]:
        max_indices[prefix] = index

print(max_indices)
'''
for key, value in max_indices.items():
    print(f"{key}: {value}")
'''

# Print guard mapping

    
results = {}
for progname in prognames:
    config = get_config(progname)
    prog_variables = config["Program_variables"]["Bools"]
    cand = config["Candidate"]["Expression"]
<<<<<<< HEAD
    iterations = config['Program specification']["iterations"]
    cnf_list_init, cnf_str_init = generate_init_DIMACS_formula(cand,variable_mapping,{})
    DIMACS_file = 'cnf-out'
    cnf_prog_formula,num_variables = extract_formula_from_DIMACS(DIMACS_file)[0][:-(len(prog_variables))],extract_formula_from_DIMACS(DIMACS_file)[1]
    #print(cnf_prog_formula)
    cnf_list_final, cnf_str_final = generate_final_DIMACS_formula(cand,variable_mapping,max_indices)
    neg_dnf_list_final = [[-elem for elem in L] for L in cnf_list_final]
    #print(neg_dnf_list_final)
    neg_cnf_list_final = DNF_to_CNF(neg_dnf_list_final)
    neg_cnf_list_final = [elem for elem in neg_cnf_list_final if elem]
    full_formula_list = cnf_list_init + cnf_prog_formula + neg_cnf_list_final
    # Convert to set to remove duplicates and then back to list
    full_formula_list = list(set(tuple(clause) for clause in full_formula_list))
    #print(full_formula_list)
=======
    updates = config["Updates in each iteration"]
    iterations = config["Number of iterations"]
    
    cnf_list_init, cnf_str_init = generate_init_DIMACS_formula(cand,variable_mapping)
    DIMACS_file = 'cnf-out'
    cnf_prog_formula = extract_formula_from_DIMACS(DIMACS_file)[:-(len(prog_variables))]
    cnf_list_final, cnf_str_final = generate_final_DIMACS_formula(cand,variable_mapping,updates)
    neg_dnf_list_final = [[-elem for elem in L] for L in cnf_list_final]
    
    neg_cnf_list_final = DNF_to_CNF(neg_dnf_list_final)
    neg_cnf_list_final = [elem for elem in neg_cnf_list_final if elem]
    

    full_formula_list = cnf_list_init + cnf_prog_formula + neg_cnf_list_final

    # Convert to set to remove duplicates and then back to list
    full_formula_list = list(set(tuple(clause) for clause in full_formula_list))
>>>>>>> d24bd43a4e465d7cbddd210d89b87469967e9ddf
    full_formula_dimacs = generate_DIMACS_formula(full_formula_list)
    file_path = "invariant_formula.cnf"

    # Write the DIMACS formula to the file
    with open(file_path, "w") as file:
        file.write(full_formula_dimacs)

    print("Full DIMACS formula has been written to", file_path)

<<<<<<< HEAD


=======
>>>>>>> d24bd43a4e465d7cbddd210d89b87469967e9ddf
    



'''
./cmsgen cnf-out.cnf
sort -n samples.out | uniq -c
'''