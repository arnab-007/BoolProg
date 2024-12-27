import pandas as pd
import os
import time
from datetime import datetime
import sys
import json
import argparse
import copy
from sampler_for_cmsgen import convert_sample,cnf_to_dimacs
# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from to_dimacs import generate_init_DIMACS_formula,generate_final_DIMACS_formula
from list_from_dimacs import extract_formula_from_DIMACS
import itertools
from collections import Counter
import subprocess

CURRENT_PATH = os.path.realpath("")
PATH = os.path.dirname(CURRENT_PATH)
assumed_shape = " "







import re

def convert_to_decimal(variable_order,expression):
    # Mapping of variables to their positions in the binary representation
    #variable_order = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']
    
    # Initialize a binary string
    binary_str = ""
    
    # Iterate over each variable in the order
    for var in variable_order:
        if f'!{var}' in expression:
            binary_str += '0'  # If the variable is negated, append '0'
        elif var in expression:
            binary_str += '1'  # If the variable is positive, append '1'
        else:
            binary_str += '0'  # If the variable is missing, assume '0'
    
    # Convert the binary string to a decimal number
    return int(binary_str, 2)


def extract_actual_vars(cnf_file_path):
    """
    Extracts actual variable numbers from a DIMACS CNF file.

    Arguments:
    cnf_file_path -- Path to the DIMACS CNF file.

    Returns:
    A list of actual variable numbers used in the CNF file.
    """
    actual_vars = set()
    try:
        with open(cnf_file_path, 'r') as file:
            for line in file:
                if line.startswith('c') or line.startswith('p'):
                    continue  # Skip comments and header lines
                # Extract literals from the clause line
                literals = map(int, line.split())
                for literal in literals:
                    if literal != 0:  # Skip the terminating 0
                        actual_vars.add(abs(literal))
    except Exception as e:
        print(f"Error extracting variables: {e}")

    return sorted(actual_vars)


def create_var_map(cnf_file_path, actual_vars):
    """
    Creates a variable mapping from sequential variables to actual variable numbers.

    Arguments:
    cnf_file_path -- Path to the DIMACS CNF file.
    actual_vars -- List of actual variable numbers.

    Returns:
    A dictionary mapping sequential variable indices to actual variable numbers.
    """
    var_map = {}
    try:
        with open(cnf_file_path, 'r') as file:
            lines = file.readlines()

            # Find the number of variables in the CNF file
            for line in lines:
                if line.startswith('p cnf'):
                    num_vars = int(line.split()[2])
                    break

            if num_vars != len(actual_vars):
                raise ValueError("Number of variables in CNF file does not match the length of actual_vars list.")

            # Create the variable map
            var_map = {i + 1: actual_vars[i] for i in range(num_vars)}

    except Exception as e:
        print(f"Error creating variable map: {e}")

    return var_map


def convert_list_to_dimacs(num_variables, clauses):
    """
    Converts a list of clauses into DIMACS CNF format.

    Arguments:
    clauses -- A list of clauses where each clause is a list of integers.
    
    Returns:
    A string representing the formula in DIMACS CNF format.
    """
    
    # Count the number of clauses
    num_clauses = len(clauses)
    
    # Build the header for DIMACS format (p cnf <num_vars> <num_clauses>)
    dimacs_lines = [f"p cnf {num_variables} {num_clauses}"]
    
    # Add each clause, ending each line with "0"
    for clause in clauses:
        clause_str = " ".join(str(literal) for literal in clause)
        dimacs_lines.append(f"{clause_str} 0")
    
    # Join all lines into a single string, separated by newlines
    return "\n".join(dimacs_lines)

def renumber_dimacs(dimacs_str):
    lines = dimacs_str.strip().split('\n')
    
    # Extract the original number of variables
    header = lines[0]
    num_vars = int(header.split()[2])
    
    # Create a mapping for renumbering
    variable_map = {}
    new_var_id = 1
    
    for line in lines[1:]:
        if line.startswith('p'):
            continue
        for var in re.findall(r'-?\d+', line):
            var = int(var)
            if var != 0 and abs(var) not in variable_map:
                variable_map[abs(var)] = new_var_id
                new_var_id += 1
                if new_var_id > num_vars:
                    break
    
    # Apply the renumbering to the clauses
    new_lines = [header]
    for line in lines[1:]:
        if line.startswith('p'):
            continue
        new_clause = []
        for var in re.findall(r'-?\d+', line):
            var = int(var)
            if var != 0:
                new_var = variable_map.get(abs(var), None)
                if new_var is not None:
                    new_var = new_var if var > 0 else -new_var
                    new_clause.append(str(new_var))
        if new_clause:
            new_lines.append(' '.join(new_clause) + ' 0')
    
    return '\n'.join(new_lines), variable_map



def parse_satisfying_assignment(output):
    lines = output.splitlines()
    #print(lines)
    assignments = list()
    for line in lines:
        if line.startswith('v '):
            # Extract the assignments
            assignments.append(line[2:].split()) # Skip 'v ' at the start
            # Convert to integers and filter out the '0' at the end
    return [item for sublist in assignments for item in sublist if item != '0']
    #return []


def parse_witness(witness):
    """
    Parse the witness string into a dictionary of variable assignments.
    Example: '!x1 && !x2 && x3 && x4 && x5' -> {'x1': False, 'x2': False, 'x3': True, 'x4': True, 'x5': True}
    """
    assignment = {}
    # Remove spaces and split by '&&' to get individual literals
    literals = witness.replace(" ", "").split('&&')
    for literal in literals:
        if literal.startswith('!'):
            assignment[literal[1:]] = False  # Negative literal
        else:
            assignment[literal] = True  # Positive literal
    return assignment


def is_clause_satisfied(clause, assignment):
    """
    Check if at least one literal in the clause is satisfied by the witness.
    """
    for literal in clause:
        if literal.startswith('!'):
            var = literal[1:]
            if not assignment.get(var, False):  # Variable is False in witness
                return True
        else:
            var = literal
            if assignment.get(var, False):  # Variable is True in witness
                return True
    return False


def is_cnf_satisfied(cnf_formula, witness):
    """
    Check if the given witness satisfies the CNF formula.
    """
    # Parse the witness into a dictionary of variable assignments
    assignment = parse_witness(witness)
    
    # Check each clause in the CNF formula
    for clause in cnf_formula:
        if not is_clause_satisfied(clause, assignment):
            return False  # If any clause is not satisfied, the CNF is not satisfied
    return True  # All clauses are satisfied


def solve_cnf(file_path,processed_variable_map):
    result = subprocess.run(['cryptominisat5', file_path], capture_output=True, text=True)
    
    processed_assignment = list()
    assignment = parse_satisfying_assignment(result.stdout)
    if (len(assignment) != 0):
        #print("SAT")
        return 0
        '''
        for valuations in assignment:
            processed_assignment.append(processed_variable_map[valuations])
        '''
        #print(assignment)
    else: 
        #print("UNSAT")
        return 1


def generate_clause(dnf_formula, reverse_mapping, variable_mapping):
    # Split the DNF formula into individual variables
    clauses = []
    terms = dnf_formula.split(' && ')
    
    for term in terms:
        # Remove '!' if it's negated and mark as negative
        if term.startswith('!'):
            var = term[1:]  # Remove '!'
            mapped_var = reverse_mapping.get(var)
            if mapped_var:
                clauses.append([-variable_mapping[mapped_var]])  # Negated variable
        else:
            mapped_var = reverse_mapping.get(term)
            if mapped_var:
                clauses.append([variable_mapping[mapped_var]])  # Positive variable
                
    return clauses


def extract_index(variable_mapping, literal):
    
    
    var_name = literal.lstrip('!')
        
    # Search for the variable name in the dictionary keys using regex
    for key in variable_mapping:
        match = re.match(f'{var_name}_(\d+)', key)  # Match pattern like 'x1_3'
        if match:
        # Extract the index from the matched key
            index = int(match.group(1))
            break

    return index



def parse_dimacs(dimacs_file):
    """
    Parse a DIMACS CNF file and return the clauses and the number of variables.
    
    Parameters:
        dimacs_file (str): Path to the DIMACS CNF file.
    
    Returns:
        (int, list of lists): Number of variables and list of clauses.
    """
    clauses = []
    num_vars = 0
    with open(dimacs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('p'):
                # This is the problem line, e.g., 'p cnf 5 3' -> 5 variables, 3 clauses
                parts = line.split()
                num_vars = int(parts[2])
            elif line.startswith('c') or line == '':
                # Ignore comment lines or empty lines
                continue
            else:
                # Parse the clause and add to the clauses list
                clause = [int(x) for x in line.split() if x != '0']  # Exclude the trailing '0'
                clauses.append(clause)
    return num_vars, clauses


def evaluate_clause(clause, assignment):
    """
    Evaluate a single clause with a given assignment.
    
    Parameters:
        clause (list of int): A clause from the CNF formula.
        assignment (dict): A dictionary where keys are variable indices and values are True or False.
    
    Returns:
        bool: True if the clause is satisfied, False otherwise.
    """
    for literal in clause:
        var = abs(literal)
        value = assignment[var]
        if literal > 0 and value:
            return True
        if literal < 0 and not value:
            return True
    return False


def evaluate_formula(clauses, assignment):
    """
    Evaluate the entire CNF formula with a given assignment.
    
    Parameters:
        clauses (list of list of int): The list of clauses from the CNF formula.
        assignment (dict): A dictionary where keys are variable indices and values are True or False.
    
    Returns:
        bool: True if the entire formula is satisfied, False otherwise.
    """
    return all(evaluate_clause(clause, assignment) for clause in clauses)


def get_unique_assignments(num_vars, clauses, variables_of_interest):
    """
    Get unique satisfying assignments for the specified variables of interest.
    
    Parameters:
        num_vars (int): Total number of variables in the CNF formula.
        clauses (list of list of int): The list of clauses from the CNF formula.
        variables_of_interest (list of int): The list of variables.
    
    Returns:
        set of tuples: Unique satisfying assignments for the variables of interest.
    """
    satisfying_assignments = set()
    
    # Generate all possible assignments (2^num_vars possibilities)
    for assignment_tuple in itertools.product([False, True], repeat=num_vars):
        # Convert tuple to dictionary (1-based index)
        assignment = {i+1: assignment_tuple[i] for i in range(num_vars)}
        
        # Check if the assignment satisfies the formula
        if evaluate_formula(clauses, assignment):
            # Extract only the assignment for the variables of interest
            selected_assignment = tuple(assignment[var] for var in variables_of_interest)
            satisfying_assignments.add(selected_assignment)
    
    return satisfying_assignments




"""
[get_config] loads the json object in [progname]'s configuration file
"""
def get_config(progname):
    with open(os.path.join(PATH, "candidate_files", progname + ".json"), "r") as f:
        config = json.load(f)
    return config


with open(os.path.join(PATH, "program-list.txt"), "r") as f:
    prognames = f.read().strip().split("\n")
#print(prognames)


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

def DNF_to_CNF(dnf_formula):
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

# Subroutine to find last variables and their transformed counterparts
def get_last_and_transformed_vars(original_dict, increment):
    # Dictionary to store the maximum suffix for each base
    max_suffix = {}
    min_suffix = {}
    # Identify the maximum suffix for each base
    for key in original_dict.keys():
        base, suffix = key.split('_')
        suffix = int(suffix)
        # Track the maximum suffix for each base
        if base not in max_suffix or suffix > max_suffix[base]:
            max_suffix[base] = suffix
               # Track the maximum suffix for each base
        if base not in min_suffix or suffix < min_suffix[base]:
            min_suffix[base] = suffix

    # Generate the last and transformed variables sets
    last_variables = [f"{base}_{max_suffix[base]}" for base in max_suffix]
    transformed_variables = [f"{base}_{min_suffix[base] + increment}" for base in min_suffix]

    return last_variables, transformed_variables


# Function to generate equivalence clauses
def generate_equivalence_clauses(var_pairs):
    clauses = []
    for var1, var2 in var_pairs:
        # Create the two clauses for var1 = var2
        clause1 = [f"!{var1}", var2]  # (!var1 || var2)
        clause2 = [var1, f"!{var2}"]  # (var1 || !var2)
        clauses.append(clause1)
        clauses.append(clause2)
    return clauses


def write_dimacs_to_file(dimacs, num_vars, num_clauses, file_name):
    with open(file_name, 'w') as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in dimacs:
            f.write(" ".join(map(str, clause)) + " 0\n")


# Function to replace variables in clauses with their corresponding index in variable_map
def replace_variables(clauses, variable_map):
    replaced_clauses = []

    for clause in clauses:
        new_clause = []
        for var in clause:
            if var.startswith('!'):
                # Negated variable: replace with negative of its index in variable_map
                var_index = variable_map.get(var[1:], None)
                if var_index is not None:
                    new_clause.append(-var_index)
            else:
                # Non-negated variable: replace with its index in variable_map
                var_index = variable_map.get(var, None)
                if var_index is not None:
                    new_clause.append(var_index)
        replaced_clauses.append(new_clause)
    
    return replaced_clauses

def translate_formula(formula, increment):
    return [[literal + increment if literal > 0 else literal - increment for literal in clause] for clause in formula]



#Extract variable mapping for the program from the file var-mapping
with open('../var-mapping') as f:
    lines = f.readlines()
    columns = [] 


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




#Extract maximum and minimum subscripts (updates) of the program variables for a single iteration
max_indices = {}
min_indices = {}
for key in variable_mapping.keys():
    prefix, index = key.split('_')
    index = int(index)
    
    if prefix not in max_indices or index > max_indices[prefix]:
        max_indices[prefix] = index

for key in variable_mapping.keys():
    prefix, index = key.split('_')
    index = int(index)
    
    if prefix not in min_indices or index < min_indices[prefix]:
        min_indices[prefix] = index

print(min_indices)
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
    init_states = config["Initial states"]["Expression"]
    k = config["Program specification"]["iterations"]
    num_ops = config["Program specification"]["operations per line"]
    lines = config["Program specification"]["number of lines"]
    k_step_varmap = list()
    k_step_varmap.append(variable_mapping)
    init_varmap = variable_mapping
    

    DIMACS_file = '../cnf-out'
    cnf_prog_formula, num_variables = extract_formula_from_DIMACS(DIMACS_file)[0][:-(len(prog_variables))], extract_formula_from_DIMACS(DIMACS_file)[1]
    #print(cnf_prog_formula)

    # Transformation from one-step program formula to k-step variable mapping
    for i in range(k):
        new_dict = {}
        for key, value in variable_mapping.items():
            # Split the key into base letter and numeric suffix
            base, suffix = key.split('_')
            new_suffix = int(suffix) + lines  # Increment suffix by 'lines'
 
            # New key: base + incremented suffix
            new_key = f"{base}_{new_suffix}"
            
            # New value: original value + num_ops * lines
            new_value = value + num_variables    
            
            # Add to the new dictionary
            new_dict[new_key] = new_value
        variable_mapping = new_dict
        k_step_varmap.append(new_dict)

    #k_step_varmap := k dictionaries comprising the variable mappings for each of the k iterations

    k_step_flat_varmap = {}
    
    for d in k_step_varmap:
        k_step_flat_varmap.update(d)
    
    with open('../k_step_varmap', "w") as file:
        file.write(json.dumps(k_step_flat_varmap, indent=4))
    output_variables, next_variables =  get_last_and_transformed_vars(k_step_varmap[0],0)
    #print(output_variables)
    #print(next_variables)
    output_variables_indices = [k_step_flat_varmap[element] for element in output_variables]
    #print(output_variables_indices[:5])
    output_variables_dict = {}

    for element in output_variables:
        output_variables_dict[element] = k_step_flat_varmap[element]
    
    output_variables_dict = {key: value for key, value in output_variables_dict.items() if key.startswith('x')}
    #print("Output variables:",output_variables_dict)
    equivalence_clauses = list()
    for i in range (k-1):
        #print(k_step_varmap[i])
        last_variables, transformed_variables = get_last_and_transformed_vars(k_step_varmap[i], lines)
        #print(last_variables,transformed_variables)
        variable_pairs = list(zip(last_variables, transformed_variables))
        #print("Variable pairs:",variable_pairs)
        clauses = generate_equivalence_clauses(variable_pairs)
        clauses = replace_variables(clauses, k_step_flat_varmap)    
        equivalence_clauses.append(clauses)

    #print("Equivalence clauses:",equivalence_clauses)


    cnf_list_init, cnf_str_init = generate_init_DIMACS_formula(init_states,init_varmap,min_indices)
    #print("cnf_list_init:",cnf_list_init)

    k_step_cnf_prog_formula = [cnf_prog_formula]

    for i in range(k-1):
        #k_step_cnf_prog_formula.append(translate_formula(k_step_cnf_prog_formula[-1], num_ops*lines))
        k_step_cnf_prog_formula.append(translate_formula(k_step_cnf_prog_formula[-1], num_variables))


    
    k_step_prog_formula = [clause for sublist in k_step_cnf_prog_formula for clause in sublist]
    #print(k_step_prog_formula)
    k_step_reachability_formula = cnf_list_init + k_step_prog_formula
    #print("k_step_reachability_formula:",k_step_reachability_formula)



    #print(k_step_reachability_formula)
    dimacs_file = "k_step_reachability_formula.cnf"
    # Write the DIMACS formula for the k-step reachability to the file
    with open(dimacs_file, "w") as file:
        file.write(generate_DIMACS_formula(k_step_reachability_formula))



    init_list_clauses = init_states.split('&&')
    #print(cand_list_clauses)
    init_list = list()
    for clause in init_list_clauses:

        literals = clause.split('||')
        literals = [literal.strip("() ") for literal in literals]
        # Filter out empty strings
        literals = list(filter(None, literals))
        init_list.append(literals)

    #print(init_list)
    
    cand_list_clauses = cand.split('&&')
    #print(cand_list_clauses)
    cand_list = list()
    for clause in cand_list_clauses:

        literals = clause.split('||')
        literals = [literal.strip("() ") for literal in literals]
        # Filter out empty strings
        literals = list(filter(None, literals))
        cand_list.append(literals)

    #print(cand_list)



    

    # Read the input file and convert the content
    with open('samples_converted.txt', 'r') as file:
        formulas = file.readlines()

    sampled_output_states = [formula.strip() for formula in formulas]
    #print(sampled_output_states)
    


    # Reverse mapping for easier lookup (mapping x1 to x1_{final})
    reverse_mapping = {key: f"{key}_{value}" for key, value in max_indices.items()}
    #print("reverse_mapping:",reverse_mapping)
    #Debug here for generating reverse_mapping
    num_variables = num_variables*k  # Number of variables
    #num_variables = 29*k  # Number of variables

    #num_variables = 50*k  # Number of variables

    individual_dist = list()
    estimated_dist = 0
    counter_examples = list()
    sampled_output_states_dict = dict(Counter(sampled_output_states))
    #print(sampled_output_states_dict)

    for dnf_formula in sampled_output_states_dict.keys():

        # Generate the clause from the given DNF formula
        formula = generate_clause(dnf_formula, reverse_mapping, output_variables_dict)
        #print(formula)
        #print("Sampled final state:",dnf_formula)



        #print(k_step_reachability_formula)
        tau = 0
        #clauses = []
        #print(k_step_reachability_formula)
        for i in range(k):
            new_clause = []
            for term in formula:
                new_term = [term[0] + num_ops * lines * i] if term[0] > 0 else [term[0] - num_ops * lines * i]
                new_clause.append(new_term)
            #print(new_clause)
            test_formula = k_step_reachability_formula + new_clause
            formula_content = convert_list_to_dimacs(num_variables,test_formula)
            

            # Save to a .cnf file (optional)
            with open("test-reachability.cnf", "w") as file:
                file.write(formula_content + "\n")
            test_reachability_cnf_file = "test-reachability.cnf"  # The path to your CNF file in DIMACS format
            valid_reachability_cnf_file = "valid-reachability.cnf"
            processed_formula_content, processed_variable_map = renumber_dimacs(formula_content)
            with open("valid-reachability.cnf", "w") as file:
                file.write(processed_formula_content + "\n")
            #print(processed_variable_map)
            #actual_vars = extract_actual_vars(cnf_file)
            #var_map = create_var_map(cnf_file, actual_vars)
            tau = tau + solve_cnf(valid_reachability_cnf_file,processed_variable_map)
        if (tau == k):
            if is_cnf_satisfied(init_list, dnf_formula):
                pass
                #print("This state belongs to init_states.")
            else:
                individual_dist.append(sampled_output_states_dict[dnf_formula])
                counter_examples.append(dnf_formula)
        else:
            individual_dist.append(0)
    #print(individual_dist)
    estimated_dist = sum(individual_dist)/sum(sampled_output_states_dict.values())
    print("Validifier outputs: ",estimated_dist)
    #print("Counterexamples generated :",counter_examples)
    state_counter_examples = [convert_to_decimal(prog_variables,state) for state in counter_examples]
    sampled_output_states_decimal = list(set([convert_to_decimal(prog_variables,state) for state in sampled_output_states]))
    # Get unique decimal values
    unique_state_counter_examples = list(set(state_counter_examples))

    #print("Set of counterexamples: ",unique_state_counter_examples)

    '''
    counterexamples_file = 'Validifier_counterexamples'

    with open(counterexamples_file, 'w') as file:
        for counterexample in unique_state_counter_examples:
            file.write(f"{counterexample}\n") 
            
        #estimated_dist = estimated_dist + (individual_dist/20)

        #print("Distance:",individual_dist)
    '''



    counterexamples_dict = {}

    for element in unique_state_counter_examples:
        counterexamples_dict[element] = 1/(len(sampled_output_states_decimal))

    input_dict = {"progname":progname,"candidate":cand,"init_states":init_states,"iterations":k}
    output_dict = {"Sampled output states":sampled_output_states_decimal,"Validifier_value":estimated_dist,"counterexamples":counterexamples_dict}
    parameters_dict = {"epsilon":0.05,"delta":0.1,"t":len(sampled_output_states)}
    total_dict = {"input_dict":input_dict,"parameters_dict":parameters_dict,"output_dict":output_dict}
    #print(len(unique_state_counter_examples))
    #print(len(sampled_output_states))
    results_directory = os.path.join(CURRENT_PATH, 'Validifier_results')
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)



    existing_files = os.listdir(results_directory)
    file_number = 1

    # Loop until you find an unused file name (exp1.json, exp2.json, etc.)
    while f"exp_{progname}_{file_number}.json" in existing_files:
        file_number += 1

    # Create the new filename in the results subdirectory
    filename = os.path.join(results_directory, f"exp_{progname}_{file_number}.json")

    # Write the data to the new file
    with open(filename, 'w') as json_file:
        json.dump(total_dict, json_file, indent=4)




        

    


    

    

    
    












   
    
